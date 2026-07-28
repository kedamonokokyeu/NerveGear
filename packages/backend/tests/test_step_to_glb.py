"""
tests/test_step_to_glb.py

The STEP -> GLB conversion seam (cad/step_to_glb.py + server/cad_api.py).

Unit-header parsing runs everywhere; the conversion round-trip and endpoint
tests need cascadio (OpenCASCADE) and are skipped where it isn't installed,
mirroring how the endpoint itself degrades.
"""

from __future__ import annotations

import pytest

from cad.step_to_glb import read_step_length_unit_microns

cascadio = pytest.importorskip("cascadio", reason="STEP conversion needs cascadio")

from fastapi.testclient import TestClient  # noqa: E402

from server.app import create_app  # noqa: E402


# A minimal STEP header is all the unit parser reads; bodies are irrelevant.
def step_header_with(si_unit_clause: bytes) -> bytes:
    return b"ISO-10303-21;\nDATA;\n#1 = ( LENGTH_UNIT() NAMED_UNIT(*) " + si_unit_clause + b" );\nENDSEC;\n"


def test_millimetre_unit_is_read():
    assert read_step_length_unit_microns(step_header_with(b"SI_UNIT(.MILLI.,.METRE.)")) == 1_000.0


def test_plain_metre_unit_is_read():
    assert read_step_length_unit_microns(step_header_with(b"SI_UNIT($,.METRE.)")) == 1_000_000.0


def test_micrometre_unit_is_read():
    assert read_step_length_unit_microns(step_header_with(b"SI_UNIT(.MICRO.,.METRE.)")) == 1.0


def test_missing_unit_falls_back_to_millimetres():
    assert read_step_length_unit_microns(b"not a real step file") == 1_000.0


@pytest.fixture(scope="module")
def two_part_step_bytes() -> bytes:
    """A real two-solid STEP assembly (named 'substrate' and 'gpu-die', mm)."""
    pytest.importorskip("OCP", reason="fixture generation needs cadquery-ocp")
    import tempfile
    from pathlib import Path

    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    solids = (
        (BRepPrimAPI_MakeBox(gp_Pnt(-15, -1.2, -15), 30, 1.2, 30).Shape(), "substrate"),
        (BRepPrimAPI_MakeBox(gp_Pnt(-6, 0, -5), 12, 0.75, 10).Shape(), "gpu-die"),
    )
    for shape, name in solids:
        label = shape_tool.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))

    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, STEPControl_AsIs)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "two_part.step"
        writer.Write(str(path))
        return path.read_bytes()


def test_endpoint_round_trips_named_parts(two_part_step_bytes):
    trimesh = pytest.importorskip("trimesh", reason="round-trip check reads GLB via trimesh")
    import io

    client = TestClient(create_app(enable_camera=False))
    response = client.post(
        "/cad/convert/step-to-glb",
        files={"file": ("two_part.step", two_part_step_bytes, "application/step")},
    )
    assert response.status_code == 200
    assert response.headers["x-nervegear-microns-per-unit"] == "1000000.0"

    scene = trimesh.load(io.BytesIO(response.content), file_type="glb")
    names = set(scene.geometry.keys())
    assert names == {"substrate", "gpu-die"}

    # The 30 mm substrate must come out 0.030 GLB units (metres): the unit
    # normalisation the microns-per-unit header promises.
    substrate = scene.geometry["substrate"]
    width_units = float(substrate.bounds[1][0] - substrate.bounds[0][0])
    assert width_units == pytest.approx(0.030, rel=1e-3)


def test_endpoint_refuses_wrong_extension():
    client = TestClient(create_app(enable_camera=False))
    response = client.post(
        "/cad/convert/step-to-glb",
        files={"file": ("model.stl", b"solid nope", "model/stl")},
    )
    assert response.status_code == 415


def test_endpoint_refuses_empty_file():
    client = TestClient(create_app(enable_camera=False))
    response = client.post(
        "/cad/convert/step-to-glb",
        files={"file": ("empty.step", b"", "application/step")},
    )
    assert response.status_code == 400
