"""CAD ingestion: mesh -> normalized frame -> 128x128 CFD mask.

Uses trimesh primitives so no external files are needed. Skips cleanly if
trimesh isn't installed (it's an optional, heavy dependency)."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from cad.ingest import ingest, normalize
from cad.voxelize import CFD_GRID_N, mask_summary, mesh_to_mask


def _box(extents=(2.0, 1.0, 0.5), translate=(10.0, -5.0, 3.0)):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(translate)   # move it off the origin to test centering
    return m


def test_normalize_centers_and_unit_scales():
    m, scale = normalize(_box())
    # centered on origin
    assert np.allclose(m.bounding_box.centroid, [0, 0, 0], atol=1e-6)
    # largest dimension is exactly 1.0
    assert np.isclose(m.extents.max(), 1.0, atol=1e-6)
    assert scale > 0


def test_ingest_box_metadata(tmp_path):
    f = tmp_path / "box.stl"
    _box().export(f)
    mesh, info = ingest(str(f))
    assert info.n_faces == 12          # a box is 12 triangles
    assert info.watertight is True
    assert np.isclose(max(info.extents), 1.0, atol=1e-6)
    assert info.volume > 0


def test_mask_shape_and_binary():
    m, _ = normalize(_box())
    mask = mesh_to_mask(m)
    assert mask.shape == (CFD_GRID_N, CFD_GRID_N)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.sum() > 0              # the box is solid somewhere


def test_mask_stays_off_the_edges():
    """Training shapes never touch the domain border; ours shouldn't either."""
    m, _ = normalize(trimesh.creation.icosphere(radius=0.5))
    mask = mesh_to_mask(m, fill_fraction=0.72)
    assert mask[0, :].sum() == 0 and mask[-1, :].sum() == 0
    assert mask[:, 0].sum() == 0 and mask[:, -1].sum() == 0


def test_sphere_mask_is_roughly_centered_and_round():
    m, _ = normalize(trimesh.creation.icosphere(radius=0.5))
    mask = mesh_to_mask(m)
    s = mask_summary(mask)
    cx = (s["bbox_xyxy"][0] + s["bbox_xyxy"][2]) / 2
    cy = (s["bbox_xyxy"][1] + s["bbox_xyxy"][3]) / 2
    assert abs(cx - CFD_GRID_N / 2) < 8        # centered
    assert abs(cy - CFD_GRID_N / 2) < 8
    assert 0.2 < s["solid_fraction"] < 0.6     # a disk fills a chunk, not all


def test_fill_fraction_controls_size():
    m, _ = normalize(trimesh.creation.box(extents=(1, 1, 1)))
    small = mesh_to_mask(m, fill_fraction=0.4).sum()
    big = mesh_to_mask(m, fill_fraction=0.8).sum()
    assert big > small


def test_slice_method_produces_a_mask():
    m, _ = normalize(trimesh.creation.icosphere(radius=0.5))
    mask = mesh_to_mask(m, method="slice")
    assert mask.shape == (CFD_GRID_N, CFD_GRID_N)
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.sum() > 0          # a sphere sliced through center is a disk


def test_degenerate_mesh_returns_zeros_not_crash():
    # a single triangle has zero volume / no closed body -> must not raise
    tri = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    mask = mesh_to_mask(tri)
    assert mask.shape == (CFD_GRID_N, CFD_GRID_N)
    assert set(np.unique(mask)).issubset({0, 1})


def test_empty_extent_returns_zeros():
    pt = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=np.zeros((0, 3), dtype=int))
    assert mesh_to_mask(pt).sum() == 0
