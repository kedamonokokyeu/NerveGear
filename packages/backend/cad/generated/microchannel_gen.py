"""
microchannel_gen.py
===================
Parametric generator for chip-level (die-level) microfluidic cooling geometry —
the kind etched directly into the silicon die in the reference images
(Microsoft die-level microfluidics + topology-optimized solid/liquid networks).

Three channel patterns:
  1. "leafvein"   — biomimetic space-colonization branching network (organic
                    trunks splitting into fine capillaries; Murray's-law widths).
  2. "dendritic"  — dense, mirror-symmetric topology-optimized look (inlet at the
                    centerline, outlets at the side edges; like the iteration-200 panel).
  3. "htree"      — recursive fractal H-tree manifold (deterministic, clean).

For each pattern it exports, into ./cad_out/:
  - <name>_die.stl          a die slab with the channels CARVED into the top face
                            (watertight slab-with-grooves — imports cleanly).
  - <name>_fluid.stl        the coolant volume only (the negative space / channel network).
  - <name>_mask128.npy      a 128x128 solid/liquid CFD mask (1 = solid/fin, 0 = channel),
                            matching backend/cad/voxelize.py's convention.
  - <name>_preview.png      quick visual: the channel field + the 128 mask.

Geometry uses the reference die footprint 20.0 mm x 32.5 mm. Channel widths and
depths are real microfluidic scales (hundreds of microns). Everything is
parametric — see PARAMS / the dataclasses below.

Run:
    python microchannel_gen.py                 # all three
    python microchannel_gen.py --pattern leafvein --seed 7
Deps: numpy, scipy, scikit-image, trimesh, matplotlib.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import trimesh

# ----------------------------------------------------------------------------
# Domain / physical parameters (millimetres)
# ----------------------------------------------------------------------------
DIE_W_MM = 20.0          # die width  (x)  — matches the reference figure
DIE_H_MM = 32.5          # die height (y)
DIE_THICK_MM = 2.0       # total die/cold-plate slab thickness (z)
CHANNEL_DEPTH_MM = 1.2   # how deep the channels are etched from the top face
PITCH_MM = 0.0625        # grid resolution of the 2D channel field (16 px / mm)
MESH_PITCH_MM = 0.125    # coarser grid for the 3D mesh (8 px/mm) -> sane file size
NZ = 22                  # vertical voxel layers for the slab
DECIMATE_FRAC = 0.22     # keep this fraction of faces after marching cubes
SMOOTH_ITERS = 4         # Laplacian smoothing passes (de-stairstep the voxels)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad_out")


def _grid_shape():
    """(ny, nx) of the 2D channel field at PITCH_MM resolution."""
    nx = int(round(DIE_W_MM / PITCH_MM))
    ny = int(round(DIE_H_MM / PITCH_MM))
    return ny, nx


# ----------------------------------------------------------------------------
# Rasterization helpers — draw variable-width channels into a 2D float field
# ----------------------------------------------------------------------------
def _stamp_segment(radius_field, p0, p1, r0, r1):
    """Stamp a centerline (with linearly varying radius) into radius_field.

    radius_field[y, x] holds, per pixel, the MAX channel radius (px) of any
    centerline through it; a later EDT fattens these to variable-width channels.
    """
    x0, y0 = p0
    x1, y1 = p1
    length = max(1, int(round(np.hypot(x1 - x0, y1 - y0))))
    ts = np.linspace(0.0, 1.0, length * 2 + 1)
    xs = np.round(x0 + (x1 - x0) * ts).astype(int)
    ys = np.round(y0 + (y1 - y0) * ts).astype(int)
    rs = r0 + (r1 - r0) * ts
    H, W = radius_field.shape
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, rs = xs[ok], ys[ok], rs[ok]
    np.maximum.at(radius_field, (ys, xs), rs)


def _radiusfield_to_channels(radius_field):
    """Sparse centerline-radius map -> boolean channel mask via EDT.
    A pixel is 'channel' if its distance to the nearest centerline pixel is
    within that centerline's radius."""
    seeds = radius_field > 0
    if not seeds.any():
        return np.zeros_like(radius_field, dtype=bool)
    dist, (iy, ix) = ndi.distance_transform_edt(~seeds, return_indices=True)
    nearest_radius = radius_field[iy, ix]
    return dist <= nearest_radius


# ----------------------------------------------------------------------------
# Pattern 1 & 2: space-colonization branching
# ----------------------------------------------------------------------------
@dataclass
class SpaceColParams:
    n_attractors: int = 1400
    influence: float = 9.0     # px, attractor influence radius
    kill: float = 1.6          # px, remove attractor when a node gets this close
    step: float = 2.2          # px, node growth step
    max_iter: int = 1400
    leaf_radius_px: float = 2.2   # tip channel half-width
    murray_exp: float = 2.6       # parent^n = sum(child^n); ~3 is Murray's law
    trunk_cap_px: float = 9.0     # clamp on trunk half-width
    max_nodes: int = 9000         # stop growth here -> controls overall density


def _space_colonization(ny, nx, roots, rng, p: SpaceColParams, region_mask=None):
    """Grow a branching network from `roots` toward random attractors.
    Uses a KD-tree for nearest-node queries (near-linear per iter).
    Returns (nodes (M,2) xy, parent (M,) int with -1 for roots)."""
    from scipy.spatial import cKDTree

    pts = []
    tries = 0
    while len(pts) < p.n_attractors and tries < p.n_attractors * 20:
        x = rng.uniform(0, nx)
        y = rng.uniform(0, ny)
        tries += 1
        if region_mask is not None:
            if not region_mask[min(int(y), ny - 1), min(int(x), nx - 1)]:
                continue
        pts.append((x, y))
    attractors = np.array(pts, dtype=float) if pts else np.zeros((0, 2))

    nodes = [np.array(r, dtype=float) for r in roots]
    parent = [-1] * len(roots)

    for _ in range(p.max_iter):
        if len(attractors) == 0 or len(nodes) >= p.max_nodes:
            break
        node_arr = np.array(nodes)
        tree = cKDTree(node_arr)
        # nearest node to each attractor, capped at influence radius
        dnear, nearest = tree.query(attractors, distance_upper_bound=p.influence)
        within = np.isfinite(dnear)
        if not within.any():
            break
        # accumulate growth directions per node (vectorized grouping)
        a_idx = np.where(within)[0]
        n_idx = nearest[a_idx]
        vecs = attractors[a_idx] - node_arr[n_idx]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        units = vecs / norms
        new_nodes, new_parent = [], []
        for ni in np.unique(n_idx):
            d = units[n_idx == ni].mean(axis=0)
            n = np.linalg.norm(d)
            if n < 1e-9:
                continue
            newp = node_arr[ni] + (d / n) * p.step
            if not (0 <= newp[0] < nx and 0 <= newp[1] < ny):
                continue
            if region_mask is not None and not region_mask[int(newp[1]), int(newp[0])]:
                continue
            new_nodes.append(newp)
            new_parent.append(int(ni))
        if not new_nodes:
            break
        nodes.extend(new_nodes)
        parent.extend(new_parent)
        # prune attractors reached by any node (query attractors against new tree)
        tree = cKDTree(np.array(nodes))
        dmin, _ = tree.query(attractors, distance_upper_bound=p.kill)
        attractors = attractors[~np.isfinite(dmin)]

    return np.array(nodes), np.array(parent)


def _depth(i, parent):
    d, seen = 0, 0
    while parent[i] >= 0 and seen < 100000:
        i = parent[i]; d += 1; seen += 1
    return d


def _murray_radii(nodes, parent, p: SpaceColParams):
    """Leaf nodes get leaf_radius; parents grow by Murray's law
    r_parent^n = sum(r_child^n). Computed deepest-first."""
    M = len(nodes)
    children = [[] for _ in range(M)]
    for i, par in enumerate(parent):
        if par >= 0:
            children[par].append(i)
    radius = np.full(M, p.leaf_radius_px, dtype=float)
    # children always have a higher index than their parent, so high->low is leaf-first
    for i in range(M - 1, -1, -1):
        ch = children[i]
        if ch:
            s = sum(radius[c] ** p.murray_exp for c in ch)
            radius[i] = min(p.trunk_cap_px, max(p.leaf_radius_px, s ** (1.0 / p.murray_exp)))
    return radius


def _network_to_field(ny, nx, nodes, parent, radius):
    rf = np.zeros((ny, nx), dtype=float)
    for i, par in enumerate(parent):
        if par < 0:
            continue
        _stamp_segment(rf, nodes[par], nodes[i], radius[par], radius[i])
    return _radiusfield_to_channels(rf)


def pattern_leafvein(rng):
    """Organic leaf-vein: a trunk enters at bottom-center and branches upward."""
    ny, nx = _grid_shape()
    p = SpaceColParams(n_attractors=2500, influence=17.0, step=2.6, kill=3.4,
                       leaf_radius_px=1.5, trunk_cap_px=9.0, murray_exp=2.8,
                       max_nodes=3500)
    roots = [(nx * 0.5, ny - 2)]
    nodes, parent = _space_colonization(ny, nx, roots, rng, p)
    radius = _murray_radii(nodes, parent, p)
    ch = _network_to_field(ny, nx, nodes, parent, radius)
    ch[ny - 6:ny, int(nx * 0.5) - 5:int(nx * 0.5) + 5] = True  # inlet stub to edge
    return ch


def pattern_dendritic(rng):
    """Dense, mirror-symmetric topology-optimized look: inlet at the vertical
    centerline, growth toward the two side edges (outlets), left/right mirrored."""
    ny, nx = _grid_shape()
    half = nx // 2
    p = SpaceColParams(n_attractors=2600, influence=15.0, step=2.4, kill=3.2,
                       leaf_radius_px=1.3, trunk_cap_px=6.5, murray_exp=2.6,
                       max_nodes=3600)
    region = np.ones((ny, half), dtype=bool)
    roots = [(half - 2, int(ny * f)) for f in (0.18, 0.5, 0.82)]
    nodes, parent = _space_colonization(ny, half, roots, rng, p, region_mask=region)
    radius = _murray_radii(nodes, parent, p)
    left = _network_to_field(ny, half, nodes, parent, radius)
    full = np.zeros((ny, nx), dtype=bool)
    full[:, :half] = left
    full[:, nx - half:] = left[:, ::-1]
    full[:, half - 3:half + 3] = True   # inlet manifold (centerline)
    full[:, 0:3] = True                  # outlet manifolds (side edges)
    full[:, nx - 3:nx] = True
    return full


# ----------------------------------------------------------------------------
# Pattern 3: recursive fractal H-tree
# ----------------------------------------------------------------------------
def pattern_htree(rng=None):
    ny, nx = _grid_shape()
    rf = np.zeros((ny, nx), dtype=float)

    def rec(cx, cy, w, h, level, r):
        if level == 0:
            return
        _stamp_segment(rf, (cx, cy - h / 2), (cx, cy + h / 2), r, r)
        x_left, x_right = cx - w / 2, cx + w / 2
        for (ex, ey) in ((x_left, cy - h / 2), (x_right, cy - h / 2),
                         (x_left, cy + h / 2), (x_right, cy + h / 2)):
            _stamp_segment(rf, (cx, ey), (ex, ey), r, r * 0.8)
        nr = max(1.2, r * 0.66)
        for childx in (x_left, x_right):
            for childy in (cy - h / 2, cy + h / 2):
                rec(childx, childy, w / 2, h / 2, level - 1, nr)

    root_r = 4.5
    _stamp_segment(rf, (nx / 2, ny - 2), (nx / 2, ny * 0.62), root_r, root_r)
    rec(nx / 2, ny * 0.5, nx * 0.5, ny * 0.46, 5, root_r)
    return _radiusfield_to_channels(rf)


PATTERNS = {
    "leafvein": pattern_leafvein,
    "dendritic": pattern_dendritic,
    "htree": pattern_htree,
}


# ----------------------------------------------------------------------------
# 2D channel field -> 3D meshes (carved die + fluid network)
# ----------------------------------------------------------------------------
def _voxels_to_mesh(vol, pitch_xyz, origin):
    """Marching-cubes a boolean volume (indexed [iy,ix,iz]) -> trimesh in mm.
    Padded with False so border surfaces close (=> watertight)."""
    padded = np.pad(vol.astype(np.float32), 1, mode="constant", constant_values=0.0)
    verts, faces, _n, _v = measure.marching_cubes(padded, level=0.5)
    verts -= 1.0
    sy, sx, sz = pitch_xyz
    xyz = np.column_stack([
        origin[0] + verts[:, 1] * sx,   # x from ix
        origin[1] + verts[:, 0] * sy,   # y from iy
        origin[2] + verts[:, 2] * sz,   # z from iz
    ])
    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=True)
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if SMOOTH_ITERS > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=SMOOTH_ITERS)
    if 0 < DECIMATE_FRAC < 1.0:
        try:
            mesh = mesh.simplify_quadric_decimation(DECIMATE_FRAC)
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass
    mesh.process(validate=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def build_meshes(channel2d):
    """(ny,nx) boolean channel field -> (carved-die mesh, fluid-network mesh).
    Resamples the channel field onto a coarser mesh grid for sane file sizes."""
    fny, fnx = channel2d.shape
    nx = max(8, int(round(DIE_W_MM / MESH_PITCH_MM)))
    ny = max(8, int(round(DIE_H_MM / MESH_PITCH_MM)))
    zoom = ndi.zoom(channel2d.astype(np.float32), (ny / fny, nx / fnx), order=1)
    channel2d = zoom >= 0.4
    nz = NZ
    sx = DIE_W_MM / nx
    sy = DIE_H_MM / ny
    sz = DIE_THICK_MM / nz
    depth_layers = max(1, int(round(CHANNEL_DEPTH_MM / sz)))

    solid = np.ones((ny, nx, nz), dtype=bool)
    top = slice(nz - depth_layers, nz)
    carve = np.broadcast_to(channel2d[:, :, None], (ny, nx, depth_layers))
    block = solid[:, :, top].copy()
    block[carve] = False
    solid[:, :, top] = block

    fluid = np.zeros((ny, nx, nz), dtype=bool)
    fblock = fluid[:, :, top].copy()
    fblock[carve] = True
    fluid[:, :, top] = fblock

    origin = (-DIE_W_MM / 2, -DIE_H_MM / 2, -DIE_THICK_MM / 2)
    die_mesh = _voxels_to_mesh(solid, (sy, sx, sz), origin)
    fluid_mesh = _voxels_to_mesh(fluid, (sy, sx, sz), origin)
    return die_mesh, fluid_mesh


# ----------------------------------------------------------------------------
# 128^2 CFD mask (1 = solid/fin, 0 = channel) — backend voxelize convention
# ----------------------------------------------------------------------------
def make_cfd_mask(channel2d, N=128, fill=0.72):
    ny, nx = channel2d.shape
    inner = max(8, int(N * fill))
    if ny >= nx:
        ih = inner
        iw = max(4, int(round(inner * nx / ny)))
    else:
        iw = inner
        ih = max(4, int(round(inner * ny / nx)))
    solid_field = (~channel2d).astype(np.float32)
    small = ndi.zoom(solid_field, (ih / ny, iw / nx), order=1)
    solid_small = (small >= 0.5).astype(np.uint8)
    mask = np.zeros((N, N), dtype=np.uint8)
    r0 = (N - solid_small.shape[0]) // 2
    c0 = (N - solid_small.shape[1]) // 2
    mask[r0:r0 + solid_small.shape[0], c0:c0 + solid_small.shape[1]] = solid_small
    return mask


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def _save_preview(name, channel2d, mask, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    ax[0].imshow(channel2d, cmap="binary", origin="lower")
    ax[0].set_title(f"{name}: channels (black=coolant)")
    ax[0].axis("off")
    ax[1].imshow(mask, cmap="binary", origin="lower")
    ax[1].set_title("128x128 CFD mask (black=solid=1)")
    ax[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def generate(pattern, seed=0):
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    fn = PATTERNS[pattern]
    channel2d = fn(rng) if pattern != "htree" else fn()

    die_mesh, fluid_mesh = build_meshes(channel2d)
    mask = make_cfd_mask(channel2d)

    die_path = os.path.join(OUT_DIR, f"{pattern}_die.stl")
    fluid_path = os.path.join(OUT_DIR, f"{pattern}_fluid.stl")
    mask_path = os.path.join(OUT_DIR, f"{pattern}_mask128.npy")
    prev_path = os.path.join(OUT_DIR, f"{pattern}_preview.png")

    die_mesh.export(die_path)
    fluid_mesh.export(fluid_path)
    np.save(mask_path, mask)
    _save_preview(pattern, channel2d, mask, prev_path)

    return {
        "pattern": pattern,
        "die_watertight": bool(die_mesh.is_watertight),
        "die_faces": int(len(die_mesh.faces)),
        "fluid_faces": int(len(fluid_mesh.faces)),
        "channel_area_fraction": round(float(channel2d.mean()), 4),
        "mask_solid_fraction": round(float(mask.mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Generate microchannel cooling CAD")
    ap.add_argument("--pattern", choices=list(PATTERNS) + ["all"], default="all")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    targets = list(PATTERNS) if args.pattern == "all" else [args.pattern]
    for pat in targets:
        print(generate(pat, seed=args.seed))


if __name__ == "__main__":
    main()
