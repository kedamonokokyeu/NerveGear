"""
transforms.py  —  symmetry canonicalization + input-channel construction.

[ADDED 2026-06-07 — see CHANGELOG.md, items #2 and #3]

Why this file exists
--------------------
The model used to see only left->right flow, so it never had to *learn* the
inlet direction. After adding 4 inlet edges + a +/-30 deg angle, the input space
blew up 4x while the dataset stayed ~500 samples, and the inlet signal (a single
1-cell-wide edge stripe) barely survives the encoder's pooling. Two fixes live here:

  (1) CANONICALIZATION  — every sample is rotated so the inlet is always 'left'.
      This turns a 4-edge problem into a 1-edge problem (4x effective data density)
      and removes the need to learn rotational equivariance. Predictions are rotated
      back to the original frame at inference time.

  (2) RICHER CONDITIONING — in addition to the sparse inlet stripe we broadcast the
      inlet velocity over the whole grid and add CoordConv (x,y) channels, so every
      pixel can read "how fast / what direction / where am I".

Convention (matches lbm_solver / dataset / generate_data):
    vx is the velocity along axis-1 (columns, +x)
    vy is the velocity along axis-0 (rows,    +y)
    edges: left=col0, right=col-1, top=row-1, bottom=row0

np.rot90(field, k) with the k below brings the inlet edge to 'left'. The matching
vector-component map per 90 deg step is (vx, vy) -> (vy, -vx). This was verified
numerically against the LBM fields (flow just inside the canonical left edge equals
the rotated prescribed inlet vector for all four edges).
"""

import numpy as np

# number of CCW np.rot90 turns that move each inlet edge onto the left column
KCAN = {'left': 0, 'bottom': 1, 'right': 2, 'top': 3}

_NORM = 0.15   # max inlet speed used during training (normalizer for velocity channels)


def canon_k(edge: str) -> int:
    return KCAN[edge]


# ----- component transform for a 2D vector under k 90-deg turns -----
def _vec_components_fwd(vx, vy, k):
    """Apply (vx, vy) -> (vy, -vx), k times."""
    k %= 4
    for _ in range(k):
        vx, vy = vy, -vx
    return vx, vy


def _vec_components_inv(vx, vy, k):
    """Inverse of _vec_components_fwd: apply (vx, vy) -> (-vy, vx), k times."""
    k %= 4
    for _ in range(k):
        vx, vy = -vy, vx
    return vx, vy


# ----- forward (raw -> canonical left-inlet frame) -----
def rotate_scalar(field, k):
    return np.rot90(field, k).copy()


def rotate_vector_field(vx, vy, k):
    vx, vy = _vec_components_fwd(vx, vy, k)
    return np.rot90(vx, k).copy(), np.rot90(vy, k).copy()


def rotate_vector_scalar(ux, uy, k):
    return _vec_components_fwd(ux, uy, k)


# ----- inverse (canonical -> original frame), used at inference -----
def inverse_rotate_scalar(field, k):
    return np.rot90(field, -k).copy()


def inverse_rotate_vector_field(vx, vy, k):
    vx = np.rot90(vx, -k)
    vy = np.rot90(vy, -k)
    vx, vy = _vec_components_inv(vx, vy, k)
    return vx.copy(), vy.copy()


# ----- CoordConv channels -----
def coord_channels(N):
    lin = np.linspace(-1.0, 1.0, N).astype(np.float32)
    cx = np.tile(lin, (N, 1))               # varies along columns  (x position)
    cy = np.tile(lin.reshape(N, 1), (1, N)) # varies along rows     (y position)
    return cx, cy


def build_input_channels(sdf, mask, Re, ux_c, uy_c, norm=_NORM):
    """
    Build the 9-channel model input from CANONICAL-frame fields/scalars.

    Channel order (in_channels = 9):
        0 sdf            geometry (signed distance, fluid +, solid -)
        1 mask           1=solid, 0=fluid
        2 Re_map         Re / 400  broadcast
        3 inlet_vx_map   sparse: left column = ux_c / norm
        4 inlet_vy_map   sparse: left column = uy_c / norm
        5 ux_broadcast   ux_c / norm  over the whole grid   [global conditioning]
        6 uy_broadcast   uy_c / norm  over the whole grid   [global conditioning]
        7 coord_x        normalized x position  [CoordConv]
        8 coord_y        normalized y position  [CoordConv]
    """
    N = sdf.shape[0]
    Re_map = np.full((N, N), Re / 400.0, dtype=np.float32)

    inlet_vx = np.zeros((N, N), dtype=np.float32); inlet_vx[:, 0] = ux_c / norm
    inlet_vy = np.zeros((N, N), dtype=np.float32); inlet_vy[:, 0] = uy_c / norm

    bvx = np.full((N, N), ux_c / norm, dtype=np.float32)
    bvy = np.full((N, N), uy_c / norm, dtype=np.float32)

    cx, cy = coord_channels(N)

    return np.stack(
        [sdf.astype(np.float32), mask.astype(np.float32), Re_map,
         inlet_vx, inlet_vy, bvx, bvy, cx, cy],
        axis=0,
    ).astype(np.float32)


IN_CHANNELS = 9
