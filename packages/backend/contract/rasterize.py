"""
contract/rasterize.py

Rasterizes solve.v1 geometry into a 2-D etch-depth map (µm per pixel).

The reduced Hele-Shaw solver needs this to turn parametric segments and
painted-etch blocks into the depth grid it solves on. The code was lifted
verbatim from the 3-D engine's geometry3d.py when that engine was removed
(2026-07, to be rebuilt from scratch) — the rebuilt engine should import
THIS copy so there is one rasterizer again.
"""

from __future__ import annotations

import numpy as np

from .schema import Geometry

def _bilinear_resample(a: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Cheap bilinear resample of a 2-D array onto (ny, nx)."""
    sy, sx = a.shape
    if (sy, sx) == (ny, nx):
        return a.astype(np.float32)
    yi = np.linspace(0, sy - 1, ny)
    xi = np.linspace(0, sx - 1, nx)
    y0 = np.clip(np.floor(yi).astype(int), 0, sy - 2)
    x0 = np.clip(np.floor(xi).astype(int), 0, sx - 2)
    fy = (yi - y0)[:, None]
    fx = (xi - x0)[None, :]
    a = a.astype(np.float32)
    return ((a[np.ix_(y0, x0)] * (1 - fy) * (1 - fx)) +
            (a[np.ix_(y0 + 1, x0)] * fy * (1 - fx)) +
            (a[np.ix_(y0, x0 + 1)] * (1 - fy) * fx) +
            (a[np.ix_(y0 + 1, x0 + 1)] * fy * fx))



def depth_map(geom: Geometry, ny: int, nx: int) -> np.ndarray:
    """Merged etch depth [µm] over the die footprint on an (ny, nx) grid.
    Union of the freeform etch raster and the stamped segment network (each
    segment carves a w-wide, `depth`-deep slot along its centreline)."""
    depth = np.zeros((ny, nx), dtype=np.float32)
    if geom.etch is not None:
        # nearest, not bilinear: interpolating across a wall invents shallow
        # phantom channels along every edge
        e = geom.etch
        ri = np.clip(np.round(np.linspace(0, e.shape[0] - 1, ny)).astype(int), 0, e.shape[0] - 1)
        ci = np.clip(np.round(np.linspace(0, e.shape[1] - 1, nx)).astype(int), 0, e.shape[1] - 1)
        depth = np.maximum(depth, e[np.ix_(ri, ci)].astype(np.float32))
    if geom.segments:
        dx = geom.die.L / nx
        dy = geom.die.W / ny
        xs = (np.arange(nx) + 0.5) * dx
        ys = (np.arange(ny) + 0.5) * dy
        gx, gy = np.meshgrid(xs, ys)
        for s in geom.segments:
            x0, y0, x1, y1 = s.x0, s.y0, s.x1, s.y1
            ddx, ddy = x1 - x0, y1 - y0
            L2 = max(ddx * ddx + ddy * ddy, 1e-12)
            # bounding box + margin keeps this O(segment area), not O(die area)
            m = s.w / 2 + max(dx, dy)
            ix0 = max(0, int((min(x0, x1) - m) / dx)); ix1 = min(nx, int((max(x0, x1) + m) / dx) + 1)
            iy0 = max(0, int((min(y0, y1) - m) / dy)); iy1 = min(ny, int((max(y0, y1) + m) / dy) + 1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            sgx = gx[iy0:iy1, ix0:ix1]; sgy = gy[iy0:iy1, ix0:ix1]
            t = np.clip(((sgx - x0) * ddx + (sgy - y0) * ddy) / L2, 0.0, 1.0)
            d = np.hypot(sgx - (x0 + t * ddx), sgy - (y0 + t * ddy))
            inside = d <= s.w / 2
            box = depth[iy0:iy1, ix0:ix1]
            box[inside] = np.maximum(box[inside], s.depth)
    max_depth = geom.die.thk - geom.die.base_thk
    return np.clip(depth, 0.0, max_depth)
