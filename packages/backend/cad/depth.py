"""
cad/depth.py

Imported CAD mesh → etch depth map, closing the "alter CAD exports" loop:
import an existing cold plate / etched die STL, recover its cavity as a depth
map, edit it with the Studio's etch tools, re-simulate, re-export.

Method: voxelize the normalized mesh (trimesh), take the top-down solid
height per (x, y) column, and read the cavity as (slab top − local top).
Depth comes back normalized 0..1 (the mesh was normalized on ingest); the
Studio maps it onto the die thickness it is editing.
"""

from __future__ import annotations

import numpy as np


def mesh_to_depth(mesh, n: int = 192) -> dict:
    """Normalized mesh → {"ny", "nx", "depth01", "solid", "meta"}.

    depth01[y, x] ∈ [0, 1]: 0 = un-etched top surface, 1 = full slab depth.
    solid[y, x]: 1 where the column contains any material (die footprint).
    """
    extents = np.asarray(mesh.extents, dtype=float)
    largest = float(extents.max()) if extents.size else 0.0
    if largest <= 0:
        raise ValueError("mesh has no extent")
    pitch = largest / max(n, 16)
    vox = mesh.voxelized(pitch)
    m = np.asarray(vox.matrix, dtype=bool)            # (X, Y, Z) occupancy
    if m.ndim != 3 or not m.any():
        raise ValueError("voxelization produced no solid cells")

    # highest solid index per column, top-down along Z
    nzx, nzy, nzz = m.shape
    zidx = np.arange(nzz)[None, None, :]
    top = np.where(m, zidx, -1).max(axis=2)           # (X, Y), -1 = empty column
    solid = top >= 0
    z_top = int(top.max())
    depth_cells = np.where(solid, z_top - top, 0).astype(float)

    # cavity depth is meaningful relative to the slab thickness
    zmin = np.where(m, zidx, nzz).min(axis=2)
    thickness_cells = float(np.median((top - zmin + 1)[solid]))
    depth01 = np.clip(depth_cells / max(thickness_cells, 1.0), 0.0, 1.0)

    # orient image-style: row 0 at the top of the plan view
    depth01 = np.flipud(depth01.T)
    solid_img = np.flipud(solid.T)
    return {
        "ny": int(depth01.shape[0]),
        "nx": int(depth01.shape[1]),
        "depth01": depth01.astype(np.float32),
        "solid": solid_img.astype(np.uint8),
        "meta": {
            "voxel_pitch": pitch,
            "slab_thickness_cells": thickness_cells,
            "grid": list(m.shape),
        },
    }
