"""
micro/heleshaw.py

2.5-D flow + conjugate thermal for FREEFORM etched cavities — what the Studio's
"chip-away" brush produces, where no centreline network exists.

Model: depth-averaged laminar flow (Hele-Shaw) on the etch depth map d(x, y),
with a rectangular-duct correction so straight channels reproduce the exact
Shah & London resistance:

    face conductance  g = C(w_loc, d) · d³ · w_face / (12 µ ℓ)
    C(w, d) = 24 w² / (fRe(α) (w + d)²),  α = aspect ratio(w_loc, d)

C → 1 for wide shallow cavities (parallel plates, fRe = 24) and ≈ 0.42 for a
square duct — the side-wall friction Hele-Shaw alone misses. Local width
w_loc = 2 × EDT of the open region.

Pressure: sparse Laplace solve with p fixed on the inlet/outlet edge cells and
the whole field scaled to the requested flow (linear system ⇒ exact rescale).
Temperature: exact-upwind enthalpy march in descending-pressure order — the
first law closes by construction. Junction map assembled exactly like
network_thermal (nearest-cavity partition + wall superheat + conduction +
spreading), so the two reduced engines agree where their domains overlap.

Validation targets (tests/test_heleshaw.py): straight painted channel vs the
analytic Fanning ΔP (±12%), energy balance < 1%, Tj within ~15% of the
straight-array conjugate model.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import correlations as corr
from . import units
from .materials import Coolant, Solid, coolant as get_coolant, solid as get_solid
from .powermap import PowerMap

MIN_DEPTH = 2e-6          # metres; shallower etch is treated as un-etched


@dataclass
class HeleShawResult:
    metrics: dict
    Tj: np.ndarray
    Tf: np.ndarray                     # NaN off-cavity
    vx: np.ndarray
    vy: np.ndarray
    pressure: np.ndarray               # Pa, NaN off-cavity
    open_mask: np.ndarray              # uint8, 1 = etched cavity
    warnings: list = dc_field(default_factory=list)


def _edge_band(shape, edge, width=1):
    ny, nx = shape
    m = np.zeros(shape, dtype=bool)
    if edge == "left":
        m[:, :width] = True
    elif edge == "right":
        m[:, -width:] = True
    elif edge == "top":
        m[:width, :] = True
    else:
        m[-width:, :] = True
    return m


def solve_etch(depth_m: np.ndarray, die_L: float, die_W: float, power: PowerMap, *,
               coolant: Coolant | str = "water",
               solid: Solid | str = "silicon",
               Q_total_m3s: float | None = None,
               flow_mlmin: float | None = None,
               t_base: float = 200e-6,
               die_thk: float = 500e-6,
               inlet_temp_c: float = 25.0,
               junction_limit_c: float = 105.0,
               inlet_edge: str = "left",
               outlet_edge: str = "right") -> HeleShawResult:
    """Depth map (metres, (ny, nx), 0 = un-etched) → flow + thermal solution."""
    from scipy import sparse
    from scipy.ndimage import distance_transform_edt, label
    from scipy.sparse.linalg import spsolve

    cool = get_coolant(coolant) if isinstance(coolant, str) else coolant
    sol = get_solid(solid) if isinstance(solid, str) else solid
    if Q_total_m3s is None:
        Q_total_m3s = units.mlmin_to_m3s(flow_mlmin if flow_mlmin is not None else 200.0)
    warnings: list[str] = []

    d = np.asarray(depth_m, dtype=float)
    ny, nx = d.shape
    dx, dy = die_L / nx, die_W / ny
    open_ = d > MIN_DEPTH
    if not open_.any():
        raise ValueError("etch is empty — nothing to solve")

    # keep only cavity components that connect inlet edge to outlet edge
    lab, ncomp = label(open_)
    in_band = _edge_band(d.shape, inlet_edge)
    out_band = _edge_band(d.shape, outlet_edge)
    in_ids = set(np.unique(lab[in_band & open_])) - {0}
    out_ids = set(np.unique(lab[out_band & open_])) - {0}
    live_ids = in_ids & out_ids
    if not live_ids:
        raise ValueError("etched cavity does not connect the inlet edge to the outlet edge")
    dead = open_ & ~np.isin(lab, list(live_ids))
    n_dead = int(dead.sum())
    if n_dead:
        warnings.append(f"{n_dead} etched cell(s) not on an inlet→outlet path carry no flow")
    act = open_ & np.isin(lab, list(live_ids))

    # local channel width for the rectangular-duct correction and the
    # convection correlation: 2 × EDT is exact only on the medial axis, so
    # propagate the ridge value across the channel with grey dilations
    # (otherwise near-wall cells look artificially narrow and over-resist)
    from scipy.ndimage import grey_dilation
    edt = distance_transform_edt(act, sampling=(dy, dx))
    # EDT measures to the nearest closed CELL CENTRE — half a cell beyond the
    # wall face — so knock one mean cell off the diameter
    w_loc = np.maximum(2.0 * edt - 0.5 * (dx + dy), 0.5 * min(dx, dy))
    for _ in range(3):
        w_loc = np.maximum(w_loc, grey_dilation(w_loc, size=3))
    w_loc = np.where(act, np.maximum(w_loc, 1e-6), 0.0)

    idx = -np.ones(d.shape, dtype=np.int64)
    ys, xs = np.where(act)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    if n > 400_000:
        raise ValueError(f"etch grid too fine for the reduced solver ({n} cells)")

    def face_g(i0, j0, i1, j1, ell, wf):
        """Conductance between two adjacent active cells."""
        df = min(d[i0, j0], d[i1, j1])
        wl = 0.5 * (w_loc[i0, j0] + w_loc[i1, j1])
        alpha = corr.aspect_ratio(wl, df)
        C = 24.0 * wl * wl / (corr.fRe_fanning(alpha) * (wl + df) ** 2)
        return C * df ** 3 * wf / (12.0 * cool.mu * ell)

    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    gx = np.zeros((ny, nx))            # face conductances to +x / +y neighbour
    gy = np.zeros((ny, nx))
    xi, xj = np.where(act[:, :-1] & act[:, 1:])
    for i, j in zip(xi, xj):
        g = face_g(i, j, i, j + 1, dx, dy)
        a, b = idx[i, j], idx[i, j + 1]
        rows += [a, b]; cols += [b, a]; vals += [-g, -g]
        diag[a] += g; diag[b] += g
        gx[i, j] = g
    yi, yj = np.where(act[:-1, :] & act[1:, :])
    for i, j in zip(yi, yj):
        g = face_g(i, j, i + 1, j, dy, dx)
        a, b = idx[i, j], idx[i + 1, j]
        rows += [a, b]; cols += [b, a]; vals += [-g, -g]
        diag[a] += g; diag[b] += g
        gy[i, j] = g

    # Dirichlet: p = 1 on inlet-edge cells, 0 on outlet-edge cells, then rescale
    p_in = act & in_band
    p_out = act & out_band
    fixed = (p_in | p_out)
    fix_idx = idx[fixed]
    fix_val = np.where(p_in[fixed], 1.0, 0.0)
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr() + sparse.diags(diag)
    # impose Dirichlet rows
    A = A.tolil()
    b = np.zeros(n)
    for k, v in zip(fix_idx, fix_val):
        A.rows[k] = [k]; A.data[k] = [1.0]
        b[k] = v
    A = A.tocsr()
    p_raw = spsolve(A, b)

    # raw flow through the inlet boundary → rescale to the requested Q
    Q_raw = 0.0
    for i, j in zip(xi, xj):
        f = gx[i, j] * (p_raw[idx[i, j]] - p_raw[idx[i, j + 1]])
        if p_in[i, j] and not p_in[i, j + 1]:
            Q_raw += f
        elif p_in[i, j + 1] and not p_in[i, j]:
            Q_raw -= f
    for i, j in zip(yi, yj):
        f = gy[i, j] * (p_raw[idx[i, j]] - p_raw[idx[i + 1, j]])
        if p_in[i, j] and not p_in[i + 1, j]:
            Q_raw += f
        elif p_in[i + 1, j] and not p_in[i, j]:
            Q_raw -= f
    if Q_raw <= 1e-30:
        raise ValueError("no through-flow (inlet and outlet not hydraulically connected)")
    scale = Q_total_m3s / Q_raw
    p_cell = p_raw * scale             # Pa (outlet = 0)
    dP = float(scale)                  # p was 1 at inlet → dP = scale [Pa]

    # cell velocities from face fluxes (depth-average), for display + Re
    fxx = np.zeros((ny, nx)); fyy = np.zeros((ny, nx))
    for i, j in zip(xi, xj):
        f = gx[i, j] * (p_cell[idx[i, j]] - p_cell[idx[i, j + 1]])
        fxx[i, j] += 0.5 * f; fxx[i, j + 1] += 0.5 * f
    for i, j in zip(yi, yj):
        f = gy[i, j] * (p_cell[idx[i, j]] - p_cell[idx[i + 1, j]])
        fyy[i, j] += 0.5 * f; fyy[i + 1, j] += 0.5 * f
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = np.where(act, fxx / (d * dy), 0.0)
        vy = np.where(act, fyy / (d * dx), 0.0)
    speed = np.hypot(vx, vy)

    # ---- heat: assign every die cell to the nearest active cavity cell ----
    q_wm2 = _resample_nearest(power.flux, ny, nx)
    cell_area = dx * dy
    dist, (iyn, ixn) = distance_transform_edt(~act, sampling=(dy, dx), return_indices=True)
    owner = idx[iyn, ixn]                        # active-cell id owning each die cell
    P_col = np.bincount(owner.ravel(), weights=(q_wm2 * cell_area).ravel(),
                        minlength=n)             # W collected per active cell
    P_total = float((q_wm2 * cell_area).sum())

    # exact-upwind enthalpy march, descending pressure
    inflow = np.zeros(n); inflow_H = np.zeros(n)
    outflow = np.zeros(n)
    faces = []
    for i, j in zip(xi, xj):
        f = gx[i, j] * (p_cell[idx[i, j]] - p_cell[idx[i, j + 1]])
        faces.append((idx[i, j], idx[i, j + 1], f))
    for i, j in zip(yi, yj):
        f = gy[i, j] * (p_cell[idx[i, j]] - p_cell[idx[i + 1, j]])
        faces.append((idx[i, j], idx[i + 1, j], f))
    for a, bb, f in faces:
        if f > 0:
            outflow[a] += f
        elif f < 0:
            outflow[bb] += -f
    order = np.argsort(-p_cell)
    T = np.full(n, inlet_temp_c)
    is_in = np.zeros(n, dtype=bool); is_in[idx[p_in]] = True
    adj_out = [[] for _ in range(n)]
    for a, bb, f in faces:
        if f > 0:
            adj_out[a].append((bb, f))
        elif f < 0:
            adj_out[bb].append((a, -f))
    mdot_floor = 0.01 * cool.rho * Q_total_m3s / max(1, int(p_out.sum()))
    n_stag = 0
    for k in order:
        if is_in[k]:
            T[k] = inlet_temp_c
        elif inflow[k] > 0:
            T[k] = inflow_H[k] / inflow[k]
        else:
            T[k] = inlet_temp_c
        mdot_k = cool.rho * max(outflow[k], inflow[k])
        if mdot_k < mdot_floor and P_col[k] > 0:
            n_stag += 1
            mdot_k = mdot_floor
        T_out = T[k] + P_col[k] / max(mdot_k * cool.cp, 1e-12)
        for bb, f in adj_out[k]:
            inflow[bb] += f
            inflow_H[bb] += f * T_out
        T[k] = T_out                       # cell temperature = after pickup
    if n_stag:
        warnings.append(f"{n_stag} near-stagnant cavity cell(s): local temperatures "
                        "there are unreliable")
    out_cells = idx[p_out]
    wsum = np.maximum(inflow[out_cells], 1e-30)
    T_outlet = float((T[out_cells] * wsum).sum() / wsum.sum())
    dT_exact = P_total / max(cool.rho * Q_total_m3s * cool.cp, 1e-12)
    balance_err = abs((T_outlet - inlet_temp_c) - dT_exact) / max(dT_exact, 1e-9)

    # wall superheat per active cell: convection off floor + exposed side walls
    closed_sides = np.zeros((ny, nx))
    for sh, ax in ((1, 1), (-1, 1), (1, 0), (-1, 0)):
        nb = np.roll(act, sh, axis=ax)   # rolled-in border garbage counts as closed — fine
        closed_sides += (~nb & act)
    h_loc = np.zeros(n); A_wet = np.zeros(n); dT_wall_c = np.zeros(n)
    ai, aj = ys, xs
    Dh_loc = 2.0 * w_loc[ai, aj] * d[ai, aj] / np.maximum(w_loc[ai, aj] + d[ai, aj], 1e-9)
    alpha_loc = np.array([corr.aspect_ratio(w, dd) for w, dd in zip(w_loc[ai, aj], d[ai, aj])])
    Nu_loc = np.array([corr.Nu_H1(a) for a in alpha_loc])
    h_all = Nu_loc * cool.k / np.maximum(Dh_loc, 1e-9)
    eta = np.array([corr.fin_efficiency(h, sol.k, 100e-6, dd)
                    for h, dd in zip(h_all, d[ai, aj])])
    A_wet = cell_area + closed_sides[ai, aj] * (0.5 * (dx + dy)) * d[ai, aj] * eta
    dT_wall_cell = P_col / np.maximum(h_all * A_wet, 1e-12)

    Tf_grid = np.full((ny, nx), np.nan)
    Tf_grid[ai, aj] = T
    Tf_near = T[owner]
    dT_wall_near = dT_wall_cell[owner]
    R_base = t_base / sol.k
    R_spread = np.maximum(dist, 0.0) ** 2 / (2.0 * sol.k * max(die_thk, 1e-6))
    Tj = (Tf_near + dT_wall_near + q_wm2 * (R_base + R_spread)).reshape(ny, nx)

    Re_max = float((cool.rho * speed[act].max() * np.median(Dh_loc) / cool.mu)) if act.any() else 0.0
    Tj_max = float(np.nanmax(Tj)); Tj_mean = float(np.nanmean(Tj)); Tj_min = float(np.nanmin(Tj))
    pumping = dP * Q_total_m3s
    R_th = (Tj_max - inlet_temp_c) / max(P_total, 1e-12)
    out_flux = inflow[out_cells]
    cv = float(np.std(out_flux) / np.mean(out_flux)) if len(out_cells) > 1 and np.mean(out_flux) > 0 else 0.0

    metrics = {
        "max_junction_temp": Tj_max,
        "mean_junction_temp": Tj_mean,
        "junction_gradient": Tj_max - Tj_min,
        "junction_cv": float(np.nanstd(Tj) / Tj_mean) if Tj_mean else 0.0,
        "coolant_dT": dT_exact,
        "coolant_outlet_temp": T_outlet,
        "thermal_resistance_KW": R_th,
        "thermal_resistance_Kcm2W": units.rth_to_area_normalised(R_th, die_L * die_W),
        "thermal_margin": junction_limit_c - Tj_max,
        "pressure_drop_kPa": units.to_kPa(dP),
        "pumping_power_W": pumping,
        "flow_mlmin": units.m3s_to_mlmin(Q_total_m3s),
        "COP": P_total / max(pumping, 1e-12),
        "Re": Re_max,
        "laminar": Re_max < 2300.0,
        "maldistribution_cv": cv,
        "total_power_W": P_total,
        "peak_flux_Wcm2": units.wm2_to_wcm2(float(q_wm2.max())),
        "Dh_um": units.to_um(float(np.median(Dh_loc))),
        "min_channel_um": units.to_um(float(w_loc[act].min())),
        "fin_efficiency": float(eta.mean()),
        "h_Wm2K": float(h_all.mean()),
        "n_segments": int(ncomp),
        "n_dead_segments": int(n_dead > 0),
        "n_stagnant_segments": n_stag,
        "energy_balance_error": float(balance_err),
        "etched_area_frac": float(open_.mean()),
    }
    pgrid = np.full((ny, nx), np.nan)
    pgrid[ai, aj] = p_cell
    return HeleShawResult(metrics=metrics, Tj=Tj, Tf=Tf_grid, vx=vx, vy=vy,
                          pressure=pgrid, open_mask=act.astype(np.uint8),
                          warnings=warnings)


def _resample_nearest(a: np.ndarray, ny: int, nx: int) -> np.ndarray:
    sy, sx = a.shape
    ri = np.clip(np.round(np.linspace(0, sy - 1, ny)).astype(int), 0, sy - 1)
    ci = np.clip(np.round(np.linspace(0, sx - 1, nx)).astype(int), 0, sx - 1)
    return a[np.ix_(ri, ci)]
