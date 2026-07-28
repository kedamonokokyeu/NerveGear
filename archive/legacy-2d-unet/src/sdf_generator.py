"""
SDF (Signed Distance Function) shape generator for CFD surrogate model training data.

Each function returns a 2D NumPy array on an NxN grid where:
  - Negative values  = inside the object
  - Positive values  = outside the object
  - Zero             = on the boundary

Usage:
    gx, gy = make_grid(N=128)
    sdf, mask, meta = random_shape(gx, gy)
    visualize_sdf(sdf, mask, meta)
"""

import numpy as np
import matplotlib.pyplot as plt


# Grid Blanket

def make_grid(N: int = 128, domain: float = 1.0):
    lin = np.linspace(-domain / 2, domain / 2, N)
    grid_x, grid_y = np.meshgrid(lin, lin)
    return grid_x, grid_y


# Primitive SDF (Circle, Rectangle, Polygon)

def sdf_circle(gx, gy, cx=0.0, cy=0.0, r=0.2):
    # Centered at cx, cy, and distance is calculated from the center
    return np.sqrt((gx - cx)**2 + (gy - cy)**2) - r


def sdf_rectangle(gx, gy, cx=0.0, cy=0.0, hw=0.2, hh=0.1):
    dx = np.abs(gx - cx) - hw # half-width
    dy = np.abs(gy - cy) - hh # half-height
    return (np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2) + np.minimum(np.maximum(dx, dy), 0))

def sdf_capsule(gx, gy, ax, ay, bx, by, r=0.1):
    """
    (ax, ay) form endpoint 1, (bx, by) form endpoint 2
    Line Segment A-->B is the center for which distance is calculated
    """
    pax, pay = gx - ax, gy - ay
    bax, bay = bx - ax, by - ay
    h = np.clip((pax * bax + pay * bay) / (bax**2 + bay**2 + 1e-12), 0, 1)
    return np.sqrt((pax - h * bax)**2 + (pay - h * bay)**2) - r


def sdf_polygon(gx, gy, vertices):
    """
    For every edge in a polygon, the minimum distance from each edge is calculated.
    """
    n = len(vertices)
    vx = np.array([v[0] for v in vertices])
    vy = np.array([v[1] for v in vertices])

    d = np.full(gx.shape, np.inf)
    s = np.ones(gx.shape)

    j = n - 1
    for i in range(n):
        ex = vx[j] - vx[i]
        ey = vy[j] - vy[i]
        wx = gx - vx[i]
        wy = gy - vy[i]
        t = np.clip((wx * ex + wy * ey) / (ex**2 + ey**2 + 1e-12), 0, 1)
        dist2 = (wx - t * ex)**2 + (wy - t * ey)**2
        d = np.minimum(d, dist2)
        # flipping number sign logic
        c1 = gy >= vy[i]
        c2 = gy < vy[j]
        c3 = ex * wy - ey * wx > 0
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
        j = i

    return s * np.sqrt(d)


# Parametric Blob SDF

def sdf_blob(gx, gy, cx=0.0, cy=0.0, base_r=0.18,
             amplitudes=None, phases=None, n_harmonics=4):
    """
    Parametric blob via radial Fourier harmonics. 
    """
    if amplitudes is None:
        amplitudes = np.zeros(n_harmonics)
    if phases is None:
        phases = np.zeros(n_harmonics)

    theta = np.arctan2(gy - cy, gx - cx)
    r = np.sqrt((gx - cx)**2 + (gy - cy)**2)

    r_surface = base_r
    for k, (amp, phase) in enumerate(zip(amplitudes, phases)):
        r_surface = r_surface + amp * np.sin((k + 2) * theta + phase)

    return r - r_surface


# NACA Airfoil

def _naca_points(thickness=0.12, chord=0.4, n_points=200):
    """Upper + lower surface points for a symmetric NACA 00XX airfoil."""
    t = thickness
    x = np.linspace(0, chord, n_points)
    yt = (5 * t * chord *
          (0.2969 * np.sqrt(x / chord)
           - 0.1260 * (x / chord)
           - 0.3516 * (x / chord)**2
           + 0.2843 * (x / chord)**3
           - 0.1015 * (x / chord)**4))
    upper = list(zip(x, yt))
    lower = list(zip(x[::-1], -yt[::-1]))
    return upper + lower


def sdf_naca(gx, gy, thickness=0.12, chord=0.4, cx=0.0, cy=0.0, n_points=200):
    """
    NACA 4-digit symmetric airfoil SDF, centered at (cx, cy).
    """
    pts_raw = _naca_points(thickness, chord, n_points)
    pts = [(x - chord / 2 + cx, y + cy) for x, y in pts_raw]
    return sdf_polygon(gx, gy, pts)

"""These represent the methods used to create random SDFs later (combining them, intersecting them, subtracting them, etc.)"""
def sdf_union(a, b):
    return np.minimum(a, b)

def sdf_intersect(a, b):
    return np.maximum(a, b)

def sdf_subtract(a, b):
    return np.maximum(a, -b)

def sdf_smooth_union(a, b, k=0.05):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return a * (1 - h) + b * h - k * h * (1 - h)


# Random Shape Generator

def random_shape(gx, gy, rng=None, shape_type=None):
    """
    Generates random training data just using the methods specified above. 
    """
    if rng is None:
        rng = np.random.default_rng()

    choices = ['circle', 'rectangle', 'blob', 'naca', 'polygon']
    if shape_type is None:
        shape_type = rng.choice(choices)

    # Keep shape away from domain edges
    cx = float(rng.uniform(-0.15, 0.15))
    cy = float(rng.uniform(-0.15, 0.15))
    meta = {'shape_type': shape_type, 'cx': cx, 'cy': cy}

    if shape_type == 'circle':
        r = float(rng.uniform(0.08, 0.22))
        sdf = sdf_circle(gx, gy, cx, cy, r)
        meta['r'] = r

    elif shape_type == 'rectangle':
        hw = float(rng.uniform(0.06, 0.22))
        hh = float(rng.uniform(0.04, 0.18))
        sdf = sdf_rectangle(gx, gy, cx, cy, hw, hh)
        meta.update({'hw': hw, 'hh': hh})

    elif shape_type == 'blob':
        base_r = float(rng.uniform(0.10, 0.20))
        n_h = int(rng.integers(2, 6))
        max_amp = base_r * 0.35
        amps = rng.uniform(-max_amp, max_amp, n_h)
        phases = rng.uniform(0, 2 * np.pi, n_h)
        sdf = sdf_blob(gx, gy, cx, cy, base_r, amps, phases, n_h)
        meta.update({'base_r': base_r, 'amplitudes': amps.tolist(),
                     'phases': phases.tolist()})

    elif shape_type == 'naca':
        thickness = float(rng.uniform(0.08, 0.18))
        chord = float(rng.uniform(0.25, 0.45))
        sdf = sdf_naca(gx, gy, thickness, chord, cx, cy)
        meta.update({'thickness': thickness, 'chord': chord})

    elif shape_type == 'polygon':
        n_verts = int(rng.integers(3, 8))
        base_r = float(rng.uniform(0.10, 0.22))
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_verts))
        radii = rng.uniform(0.6, 1.0, n_verts) * base_r
        verts = [(cx + r * np.cos(a), cy + r * np.sin(a))
                 for r, a in zip(radii, angles)]
        sdf = sdf_polygon(gx, gy, verts)
        meta['vertices'] = [(float(x), float(y)) for x, y in verts]

    mask = (sdf <= 0).astype(np.float32)   # 1 = solid, 0 = fluid
    return sdf.astype(np.float32), mask, meta


#  Visualization 

def visualize_sdf(sdf, mask, meta=None, title=None):
    """
    Plot the SDF (with zero contour) alongside the solid mask.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    im = ax.imshow(sdf, origin='lower', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax.contour(sdf, levels=[0], colors='black', linewidths=1.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    label = meta.get('shape_type', '') if meta else ''
    ax.set_title(f"SDF  [{label}]")
    ax.axis('off')

    ax = axes[1]
    ax.imshow(mask, origin='lower', cmap='gray_r')
    ax.set_title("Solid Mask  (white = solid)")
    ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"sdf_{label}.png", dpi=120, bbox_inches='tight')
    plt.show()
    print(f"  Saved sdf_{label}.png")


if __name__ == '__main__':
    N = 128
    gx, gy = make_grid(N=N, domain=1.0)
    rng = np.random.default_rng(seed=42)

    print(f"Grid: {N}x{N}  |  Domain: [-0.5, 0.5]\n")

    for shape in ['circle', 'rectangle', 'blob', 'naca', 'polygon']:
        sdf, mask, meta = random_shape(gx, gy, rng=rng, shape_type=shape)
        solid_pct = 100 * mask.sum() / (N * N)
        print(f"{shape:12s} | SDF range [{sdf.min():.3f}, {sdf.max():.3f}] "
              f"| Solid: {solid_pct:.1f}%")
        visualize_sdf(sdf, mask, meta)
