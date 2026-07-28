"""
cad/step_to_glb.py

Converts a STEP file into GLB for the Studio's 3D viewer.

STEP is the exact-geometry format CAD tools exchange (mathematical surfaces,
not triangles). Browsers can only draw triangles, so before the Studio can
show a STEP file someone has to tessellate it — approximate its surfaces with
a triangle mesh. That someone is this module, because the library that can do
it (OpenCASCADE, via the small `cascadio` wrapper) is far too heavy to ship
to a browser.

Two jobs:
  1. convert_step_to_glb()  — tessellate STEP bytes -> GLB bytes, keeping the
     assembly's part names (the viewer trusts labels in Phase 1). OpenCASCADE
     reads the STEP's declared unit and normalises the GLB to glTF-spec
     METRES (verified: a 30 mm box comes out 0.030 GLB units).
  2. read_step_length_unit_microns() — what the STEP file itself declared one
     unit to mean, in µm. Logging/diagnostics only, given the normalisation
     above.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

# Tessellation tolerances passed to OpenCASCADE. 0.01 model units of linear
# deflection is fine detail at millimetre scale (10 µm sag on curved faces)
# while keeping meshes small; 0.5 rad angular matches cascadio's default.
TESSELLATION_LINEAR_TOLERANCE = 0.01
TESSELLATION_ANGULAR_TOLERANCE = 0.5

# STEP headers declare length units as SI_UNIT tokens, e.g.
#   SI_UNIT(.MILLI.,.METRE.)  -> millimetres (the overwhelming default).
# Map each SI prefix to µm per unit. No prefix means plain metres.
_MICRONS_PER_SI_PREFIX = {
    "": 1_000_000.0,        # metre
    ".CENTI.": 10_000.0,    # centimetre
    ".MILLI.": 1_000.0,     # millimetre
    ".MICRO.": 1.0,         # micrometre
}

# Matches SI_UNIT(.MILLI.,.METRE.) or SI_UNIT($,.METRE.) with any spacing.
_SI_LENGTH_UNIT_PATTERN = re.compile(
    rb"SI_UNIT\s*\(\s*(\$|\.[A-Z]+\.)?\s*,\s*\.METRE\.\s*\)", re.IGNORECASE
)

MICRONS_PER_MILLIMETER = 1_000.0


def read_step_length_unit_microns(step_bytes: bytes) -> float:
    """Return µm per model unit, from the STEP file's declared length unit.

    STEP files are plain text, so we scan for the first SI length-unit token.
    Files that omit it (rare, technically invalid) are assumed millimetres —
    the de-facto CAD default.
    """
    match = _SI_LENGTH_UNIT_PATTERN.search(step_bytes)
    if match:
        prefix = (match.group(1) or b"").decode("ascii", "ignore").upper()
        if prefix == "$":  # "$" is STEP's explicit "no prefix" (plain metres)
            prefix = ""
        if prefix in _MICRONS_PER_SI_PREFIX:
            return _MICRONS_PER_SI_PREFIX[prefix]
    return MICRONS_PER_MILLIMETER


def convert_step_to_glb(step_bytes: bytes) -> bytes:
    """Tessellate STEP bytes into GLB bytes, preserving part names.

    cascadio works on file paths (OpenCASCADE is a C++ library underneath),
    so the bytes take a round-trip through a temporary directory that is
    always cleaned up, even on failure.
    """
    # Imported here, not at module top: cascadio bundles OpenCASCADE (~large,
    # optional). The rest of the backend must keep working without it.
    try:
        import cascadio
    except ImportError as error:
        raise RuntimeError(
            "STEP conversion needs the 'cascadio' package "
            "(pip install cascadio)."
        ) from error

    with tempfile.TemporaryDirectory(prefix="nervegear_step_") as workdir:
        step_path = Path(workdir) / "input.step"
        glb_path = Path(workdir) / "output.glb"
        step_path.write_bytes(step_bytes)

        cascadio.step_to_glb(
            str(step_path),
            str(glb_path),
            tol_linear=TESSELLATION_LINEAR_TOLERANCE,
            tol_angular=TESSELLATION_ANGULAR_TOLERANCE,
        )

        if not glb_path.exists() or glb_path.stat().st_size == 0:
            raise ValueError(
                "OpenCASCADE produced no geometry — the file may be corrupt "
                "or contain no solids."
            )
        return glb_path.read_bytes()
