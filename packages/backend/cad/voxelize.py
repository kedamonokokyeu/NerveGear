"""
cad/voxelize.py

Convert normalized 3D mesh from ingest.py to 2D mask. The 2D surrogate is validated 2D
model (not too necessary at the moment), but when 3D model lands, it can serve as a
quick preview in the 2D version. 

Basically, take the 3D import and project it on one axis.
"""

from __future__ import annotations
import numpy as np

CFD_GRID_N = 128  # grid resolution to match model training backend
DEFAULT_FILL = 0.72  # fraction of grid filled in, leaving edge margin 


def _place_centered(occupancy: np.ndarray, N: int) -> np.ndarray:
    """
    Center a 2D boolean occupancy block inside an N x N grid. Crop it if it's
    larger than the grid.
    occupancy is input occupancy array --> 2D grid marks which pixels solid or not.
    """
    mask = np.zeros((N, N), dtype=np.uint8)
    height, width = occupancy.shape
    # crop if needed
    occupancy = occupancy[: min(height, N), : min(width, N)]
    h, w = occupancy.shape
    r0 = (N - h) // 2
    c0 = (N - w) // 2
    mask[r0 : r0 + h, c0 : c0 + w] = occupancy.astype(np.uint8)
    return mask


def mesh_to_mask(
    mesh,
    N: int = CFD_GRID_N,
    axis: int = 2,
    fill_fraction: float = DEFAULT_FILL,
    method: str = "projection",
) -> np.ndarray:
    """Convert a (normalized, origin-centered) mesh to an N x N solid mask.

    axis           : view axis (2 = x-y plane; 0 or 1 give the other two views).
    fill_fraction  : how much of the grid the largest dimension should span.
    method         : "projection" (filled silhouette, robust to messy meshes) or
                     "slice" (a true cross-section through the body center).

    Always returns an (N, N) uint8 array (values in {0, 1}); on any failure it
    returns an all-zero mask rather than raising, so an upload never 500s.
    """
    try:
        extent = float(np.asarray(mesh.extents).max())
        if extent <= 0:
            return np.zeros((N, N), dtype=np.uint8)
        if method == "slice":
            # slice can be empty or error (open mesh / plane on a face / shapely
            # quirk); on any such case, fall through to the robust projection.
            try:
                m = _slice_to_mask(mesh, N, axis, fill_fraction)
                if m is not None and m.sum() > 0:
                    return m
            except Exception:
                pass
        return _projection_to_mask(mesh, N, axis, fill_fraction, extent)
    except Exception:
        return np.zeros((N, N), dtype=np.uint8)


def _projection_to_mask(mesh, N, axis, fill_fraction, extent) -> np.ndarray:
    target_cells = max(8, int(N * fill_fraction))
    pitch = extent / target_cells
    vox = mesh.voxelized(pitch)
    matrix = np.asarray(vox.matrix, dtype=bool)   # 3D occupancy
    if matrix.ndim != 3 or matrix.size == 0:
        return np.zeros((N, N), dtype=np.uint8)
    proj = matrix.any(axis=axis)                  # filled silhouette for closed bodies
    proj = np.flipud(proj.T)                       # orient image-style (up is up)
    return _place_centered(proj, N)


def _slice_to_mask(mesh, N, axis, fill_fraction):
    """Cross-section through the body center, perpendicular to `axis`, rasterized
    onto the grid. Returns None if no section exists. Uses shapely point-tests."""
    normal = [0, 0, 0]; normal[axis] = 1
    section = mesh.section(plane_origin=mesh.centroid, plane_normal=normal)
    if section is None:
        return None
    planar, _to3d = section.to_planar()
    polys = list(getattr(planar, "polygons_full", []))
    if not polys:
        return None

    from shapely.geometry import MultiPolygon
    geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
    minx, miny, maxx, maxy = geom.bounds
    span = max(maxx - minx, maxy - miny) or 1.0
    cells = max(8, int(N * fill_fraction))
    pitch = span / cells

    # sample grid-cell centers across the polygon's bounding box
    xs = minx + (np.arange(cells) + 0.5) * pitch
    ys = miny + (np.arange(cells) + 0.5) * pitch
    gx, gy = np.meshgrid(xs, ys)
    try:
        from shapely import vectorized
        inside = vectorized.contains(geom, gx, gy)
    except Exception:
        from shapely.geometry import Point
        inside = np.array([[geom.contains(Point(x, y)) for x in xs] for y in ys])
    return _place_centered(np.flipud(inside), N)


def mask_summary(mask: np.ndarray) -> dict:
    """Quick stats useful for a UI badge / sanity check."""
    total = mask.size
    solid = int(mask.sum())
    ys, xs = np.where(mask > 0)
    bbox = (
        [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if solid
        else [0, 0, 0, 0]
    )
    return {
        "N": int(mask.shape[0]),
        "solid_cells": solid,
        "solid_fraction": round(solid / total, 4),
        "bbox_xyxy": bbox,
    }
