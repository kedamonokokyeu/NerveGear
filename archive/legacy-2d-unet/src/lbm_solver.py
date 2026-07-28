"""
2D Lattice-Boltzmann (D2Q9) fluid solver for CFD surrogate training data.
It'll take a solid mask (from sdf_generator.py), an inlet velocity, and a Reynolds
number. Runs the simulation to steady state and returns velocity + pressure fields.

Output arrays (all float32, shape N x N):
    vx       — horizontal velocity at every cell
    vy       — vertical velocity at every cell
    pressure — scalar pressure at every cell (rho * cs^2)
"""

import numpy as np


#  D2Q9 Constants 
#
# The D2Q9 model tiles the grid with 9 "channels" per cell — one for each
# direction a particle packet can travel. .
#
# Direction index layout (matches ex/ey below):
#   6  2  5
#   3  0  1
#   7  4  8
#
# Index 0 = stationary (stay put)
# Indices 1-4 = cardinal directions (right, up, left, down)
# Indices 5-8 = diagonals

# Velocity vectors: ex[i], ey[i] is the (dx, dy) for direction i
EX = np.array([ 0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=np.float64) # cardinal directions horizontally
EY = np.array([ 0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=np.float64) # cardinal directions vertically

# Weighted the directions to follow the mathematical derivations for Navier-Stokes equations
W = np.array([
    4/9,                          # 0: stationary
    1/9,  1/9,  1/9,  1/9,       # 1-4: cardinal
    1/36, 1/36, 1/36, 1/36       # 5-8: diagonal
], dtype=np.float64)

# speed of sound in lattice units 
CS2 = 1.0 / 3.0

# Bounce-back pairs: if particle hits wall, it "bounces back"
# BOUNCE[i] gives the index of the opposite direction to i.
BOUNCE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


#  Equilibrium Distribution 
#
# This is the distribution f would have if the fluid were perfectly at rest
# relative to its local mean velocity. The collision step nudges f toward this.
#
# Formula (standard BGK):
#   f_eq[i] = W[i] * rho * (1 + (e·u)/cs² + (e·u)²/(2cs⁴) - |u|²/(2cs²))
#
# rho  : density at each cell, shape (N, N)
# ux   : x-velocity at each cell, shape (N, N)
# uy   : y-velocity at each cell, shape (N, N)
# returns f_eq, shape (9, N, N)

def equilibrium(rho, ux, uy):
    f_eq = np.zeros((9, *rho.shape), dtype=np.float64)
    for i in range(9):
        eu = EX[i] * ux + EY[i] * uy          # dot product e·u at every cell
        u2 = ux**2 + uy**2                     # |u|² at every cell
        f_eq[i] = W[i] * rho * (1.0 + eu/CS2 + eu**2/(2*CS2**2) - u2/(2*CS2))
    return f_eq


#  Single LBM Timestep 
# f     : distribution, shape (9, N, N)
# mask  : solid mask, shape (N, N), 1=solid 0=fluid
# ux_in : inlet x-velocity (left boundary condition)
# omega : relaxation parameter

def lbm_step(f, mask, ux_in, uy_in, omega, inlet_edge='left', outlet_edge='right'):
    solid = mask == 1
    fluid = ~solid

    # 1. Macroscopic quantities from current f (fluid cells only)
    rho = np.maximum(f.sum(axis=0), 1e-10)
    ux  = (f * EX[:, None, None]).sum(axis=0) / rho
    uy  = (f * EY[:, None, None]).sum(axis=0) / rho
    ux[solid] = 0.0
    uy[solid] = 0.0

    # 2. Collision (BGK): only at fluid cells
    f_eq  = equilibrium(rho, ux, uy)
    f_star = f.copy()
    f_star[:, fluid] = f[:, fluid] - omega * (f[:, fluid] - f_eq[:, fluid])
    f_star[:, solid] = 0.0   # solid cells play no role in collision

    # 3. Streaming + mid-link bounce-back.
    #
    #    For each direction i, a particle at fluid cell (y,x) wants to move
    #    to (y+EY[i], x+EX[i]). If that target is solid, the particle can't
    #    go there — it bounces back to (y,x) in the OPPOSITE direction instead.
    #    This keeps particles entirely in the fluid domain; solid cells stay empty.
    #
    #    solid_target[y,x] = True  means the destination of direction i from (y,x)
    #    is inside the solid object.
    f_new = np.zeros_like(f)
    for i in range(9):
        # source_is_solid[y,x] = True if the cell sending to (y,x) in direction i is solid.
        # That source cell is at (y - EY[i], x - EX[i]), so we shift solid FORWARD by (EX,EY).
        # np.roll(A, +k, axis=1)[y,x] = A[y, x-k], which is what we want.
        source_is_solid = np.roll(
            np.roll(solid.astype(np.float64), int(EX[i]), axis=1),
            int(EY[i]), axis=0
        ).astype(bool)

        # dest_is_solid[y,x] = True if the destination of direction i from (y,x) is solid.
        # That destination is at (y + EY[i], x + EX[i]), so shift solid BACKWARD by (EX,EY).
        dest_is_solid = np.roll(
            np.roll(solid.astype(np.float64), -int(EX[i]), axis=1),
            -int(EY[i]), axis=0
        ).astype(bool)

        # Normal stream: fluid cell (y,x) receives f_star[i] from its fluid source.
        # Block arrivals at solid cells, or from solid sources (solid → fluid doesn't happen).
        streamed = np.roll(np.roll(f_star[i], int(EX[i]), axis=1), int(EY[i]), axis=0)
        f_new[i] += np.where(solid | source_is_solid, 0.0, streamed)

        # Bounce-back: fluid cells whose direction-i target is solid reflect back to BOUNCE[i].
        f_new[BOUNCE[i]] += np.where(dest_is_solid & fluid, f_star[i], 0.0)

    # 4. Inlet BC: force equilibrium at prescribed velocity on the chosen edge
    u2_in = ux_in**2 + uy_in**2
    for i in range(9):
        eu  = EX[i] * ux_in + EY[i] * uy_in
        val = W[i] * (1.0 + eu/CS2 + eu**2/(2*CS2**2) - u2_in/(2*CS2))
        if inlet_edge == 'left':   f_new[i, :,  0] = val
        elif inlet_edge == 'right':  f_new[i, :, -1] = val
        elif inlet_edge == 'bottom': f_new[i,  0, :] = val
        elif inlet_edge == 'top':    f_new[i, -1, :] = val

    # 5. Outlet BC: zero-gradient extrapolation on the opposite edge
    if outlet_edge == 'right':  f_new[:, :, -1] = f_new[:, :, -2]
    elif outlet_edge == 'left': f_new[:, :,  0] = f_new[:, :,  1]
    elif outlet_edge == 'top':  f_new[:, -1, :] = f_new[:, -2, :]
    elif outlet_edge == 'bottom': f_new[:,  0, :] = f_new[:,  1, :]

    # 6. Macroscopic output
    rho_new = np.maximum(f_new.sum(axis=0), 1e-10)
    ux_new  = (f_new * EX[:, None, None]).sum(axis=0) / rho_new
    uy_new  = (f_new * EY[:, None, None]).sum(axis=0) / rho_new
    ux_new[solid] = 0.0
    uy_new[solid] = 0.0

    return f_new, rho_new, ux_new, uy_new


"""
Runs the simulation for n_steps timesteps.
Returns the final vx, vy, and pressure fields.
"""
# mask   : solid mask (N, N), float32, 1=solid 0=fluid
# ux_in  : inlet velocity (e.g. 0.1 — keep this small for stability, <0.2)
# Re     : Reynolds number (e.g. 100, 200, 400)
# n_steps: number of timesteps (2000–5000 is usually enough for 128x128)

def run_lbm(mask, ux_in=0.1, uy_in=0.0, inlet_edge='left', Re=100, n_steps=3000,
            tol=1e-5, check_every=250, min_steps=500):
    # [MODIFIED 2026-06-07 — see CHANGELOG.md, item #6]
    # n_steps is now a MAX. We stop early once the velocity field stops changing
    # (relative L2 change between checks < tol), so every saved label is actually
    # at steady state instead of "whatever 3000 fixed steps happened to give".
    OPPOSITE = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
    outlet_edge = OPPOSITE[inlet_edge]

    N = mask.shape[0]

    # Derive relaxation parameter omega from Re and total inlet speed
    # nu (kinematic viscosity in lattice units) = U * N / Re
    # tau = nu/cs2 + 0.5,  omega = 1/tau
    U     = np.sqrt(ux_in**2 + uy_in**2)
    nu    = max(U, 1e-6) * N / Re
    tau   = nu / CS2 + 0.5
    omega = 1.0 / tau

    # Clamp omega for numerical stability
    omega = float(np.clip(omega, 0.51, 1.95))

    # Initialize: uniform flow everywhere, rho=1
    rho0 = np.ones((N, N), dtype=np.float64)
    ux0  = np.full((N, N), ux_in, dtype=np.float64)
    uy0  = np.full((N, N), uy_in, dtype=np.float64)
    ux0[mask == 1] = 0.0
    uy0[mask == 1] = 0.0

    f = equilibrium(rho0, ux0, uy0)

    # Time-stepping loop (runs until steady state or n_steps, whichever first)
    prev_speed = None
    for step in range(n_steps):
        f, rho, ux, uy = lbm_step(f, mask, ux_in, uy_in, omega, inlet_edge, outlet_edge)

        if (step + 1) % check_every == 0:
            speed = np.sqrt(ux**2 + uy**2)
            max_v = speed.max()
            if prev_speed is not None:
                rel = (np.linalg.norm(speed - prev_speed)
                       / (np.linalg.norm(speed) + 1e-12))
                print(f"  step {step+1:>5}/{n_steps}  |  max |u| = {max_v:.4f}  "
                      f"|  rel Δ = {rel:.2e}")
                if (step + 1) >= min_steps and rel < tol:
                    print(f"  ✓ converged at step {step+1}  (rel Δ = {rel:.2e} < {tol:.0e})")
                    break
            else:
                print(f"  step {step+1:>5}/{n_steps}  |  max |u| = {max_v:.4f}")
            prev_speed = speed

    pressure = rho * CS2
    return ux.astype(np.float32), uy.astype(np.float32), pressure.astype(np.float32)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from Source.sdf_generator import make_grid, sdf_circle

    print("Running LBM demo: flow past a circle ...\n")

    N = 128
    gx, gy = make_grid(N=N)
    sdf  = sdf_circle(gx, gy, cx=0.0, cy=0.0, r=0.12)
    mask = (sdf <= 0).astype(np.float32)

    vx, vy, pressure = run_lbm(mask, ux_in=0.1, uy_in=0.05, Re=200, n_steps=3000)

    speed = np.sqrt(vx**2 + vy**2)
    print(f"\nDone. Speed range: [{speed.min():.4f}, {speed.max():.4f}]")
    print(f"Pressure range:    [{pressure.min():.4f}, {pressure.max():.4f}]")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, d, t, c in zip(axes,
        [speed, pressure, mask],
        ['Speed |u|', 'Pressure', 'Solid Mask'],
        ['viridis', 'RdBu_r', 'gray_r']):
        ax.imshow(d, origin='lower', cmap=c)
        ax.set_title(t)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('lbm_demo.png', dpi=120, bbox_inches='tight')
    print("Saved lbm_demo.png")

