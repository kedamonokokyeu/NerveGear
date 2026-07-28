"""
micro/network_thermal.py

Conjugate thermal solve for ARBITRARY channel networks — the freeform vein/etch
designs the Studio produces — building on the exact laminar hydraulics of
micro/geometry.Network.

Where micro/conjugate.py is the validated reduced model for straight parallel
arrays, this module generalises the same physics to any connected network:

  1. exact laminar flow split per segment (resistor-network nodal solve),
  2. coolant temperature marched segment-by-segment in flow order
     (enthalpy balance: dT = P_collected / (m_dot·cp), flow-weighted mixing at
     junctions — the first law holds exactly by construction),
  3. junction temperature map: every die cell drains its heat to the nearest
     channel segment (computed from exact segment geometry, NOT from the raster,
     so thin channels never "disappear" at coarse grid resolution);
     Tj = T_coolant(nearest) + wall superheat (Nu correlation, fin efficiency)
        + base conduction + a lateral spreading penalty d²/(2·k·t_die).

Approximations to be honest about (they set the confidence score):
  * heat collection is by nearest-channel partition, not a solved 3-D
    conduction field;
  * wall superheat is segment-averaged (H1 constant-flux Nusselt, fully
    developed);
  * near-stagnant segments (parallel paths with ~zero flow) cannot carry their
    collected heat — their flow is floored at 1% of the mean and flagged.

Validation: for a straight parallel array this must agree with
micro/conjugate.solve (see tests/test_network_thermal.py) and the coolant ΔT
must equal P/(m_dot·cp) exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import correlations as corr
from . import units
from .geometry import Network, rasterize
from .materials import Coolant, Solid, coolant as get_coolant, solid as get_solid
from .powermap import PowerMap


@dataclass
class NetworkThermalResult:
    metrics: dict
    Tj: np.ndarray                 # (ny, nx) junction temperature map [°C]
    Tf: np.ndarray                 # (ny, nx) coolant temp [°C]; NaN off-channel
    raster: dict                   # mask / channel-id / vx / vy grids (display + DRC)
    seg_stats: list = dc_field(default_factory=list)   # per real segment dicts
    warnings: list = dc_field(default_factory=list)


def _resample_nearest(a: np.ndarray, ny: int, nx: int) -> np.ndarray:
    sy, sx = a.shape
    ri = np.clip(np.round(np.linspace(0, sy - 1, ny)).astype(int), 0, sy - 1)
    ci = np.clip(np.round(np.linspace(0, sx - 1, nx)).astype(int), 0, sx - 1)
    return a[np.ix_(ri, ci)]


def _assign_nearest_segment(net: Network, real: list, gx, gy, spacing=None):
    """For every grid cell: nearest REAL segment (index into net.segments), the
    distance to that segment's centreline, and the normalised position t along
    it. Fast path: cKDTree over points sampled along the centrelines at
    `spacing` (distance error ≤ spacing/2). Falls back to the exact per-segment
    sweep without scipy."""
    try:
        from scipy.spatial import cKDTree
    except Exception:  # noqa: BLE001
        cKDTree = None
    if cKDTree is not None:
        if spacing is None:
            span = max(net.die_L or 0.0, net.die_W or 0.0, 1e-6)
            spacing = span / 400.0
        pts, pid, pt = [], [], []
        for i in real:
            s = net.segments[i]
            (x0, y0), (x1, y1) = s.p0, s.p1
            n = max(2, int(np.ceil(s.length / spacing)) + 1)
            ts = np.linspace(0.0, 1.0, n)
            pts.append(np.column_stack([x0 + ts * (x1 - x0), y0 + ts * (y1 - y0)]))
            pid.append(np.full(n, i, dtype=np.int32))
            pt.append(ts)
        P = np.concatenate(pts)
        tree = cKDTree(P)
        d, k = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=1, workers=-1)
        pid = np.concatenate(pid); pt = np.concatenate(pt)
        return (pid[k].reshape(gx.shape), d.reshape(gx.shape).astype(float),
                pt[k].reshape(gx.shape))
    best_d = np.full(gx.shape, np.inf)
    best_i = np.zeros(gx.shape, dtype=np.int32)
    best_t = np.zeros(gx.shape)
    for i in real:
        s = net.segments[i]
        (x0, y0), (x1, y1) = s.p0, s.p1
        dx, dy = x1 - x0, y1 - y0
        L2 = max(dx * dx + dy * dy, 1e-20)
        t = np.clip(((gx - x0) * dx + (gy - y0) * dy) / L2, 0.0, 1.0)
        d = np.hypot(gx - (x0 + t * dx), gy - (y0 + t * dy))
        upd = d < best_d
        best_d[upd] = d[upd]
        best_i[upd] = i
        best_t[upd] = t[upd]
    return best_i, best_d, best_t


def solve_network(net: Network, power: PowerMap, *,
                  coolant: Coolant | str = "water",
                  solid: Solid | str = "silicon",
                  Q_total_m3s: float | None = None,
                  flow_mlmin: float | None = None,
                  t_base: float = 200e-6,
                  die_thk: float = 500e-6,
                  inlet_temp_c: float = 25.0,
                  junction_limit_c: float = 105.0,
                  grid_shape: tuple = (128, 160),
                  supersample: int | None = None) -> NetworkThermalResult:
    """Solve flow + conjugate thermal for an arbitrary channel network.

    `net` comes from geometry.straight_parallel / murray_tree /
    network_from_segments. `t_base` is silicon between channel floor and the
    heated face; `die_thk` the full die thickness (lateral spreading depth).
    """
    cool = get_coolant(coolant) if isinstance(coolant, str) else coolant
    sol = get_solid(solid) if isinstance(solid, str) else solid
    if Q_total_m3s is None:
        Q_total_m3s = units.mlmin_to_m3s(flow_mlmin if flow_mlmin is not None else 200.0)
    warnings: list[str] = []

    # ---- 1. exact laminar flow split ------------------------------------
    flow = net.solve_flow(cool, Q_total_m3s)
    p = flow["pressure"]
    Qs = np.asarray(flow["seg_flow"], dtype=float)
    mdot_total = cool.rho * Q_total_m3s

    real = [i for i, s in enumerate(net.segments) if s.w > 0]
    if not real:
        raise ValueError("network has no real segments")
    n_dead = int(net.meta.get("n_dead", 0))
    if n_dead:
        warnings.append(f"{n_dead} segment(s) not connected inlet→outlet were excluded")

    # ---- 2. geometric nearest-segment assignment ------------------------
    ny, nx = grid_shape
    L = net.die_L or 1.0
    W = net.die_W or 1.0
    dx, dy = L / nx, W / ny
    cell_area = dx * dy
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    gx, gy = np.meshgrid(xs, ys)
    q_wm2 = _resample_nearest(power.flux, ny, nx)

    # ONE assignment pass on a supersampled grid (Voronoi strips between
    # channels otherwise quantise to whole cells: ±25% per-channel power at
    # coarse grids); base-grid maps take the centre subcell of each cell.
    if supersample is None:
        w_min_real = min(net.segments[i].w for i in real)
        cell = max(dx, dy)
        ss = int(min(5, max(1, np.ceil(cell / max(w_min_real / 4.0, 1e-9)))))
    else:
        ss = max(1, int(supersample))
    fxs = (np.arange(nx * ss) + 0.5) * (dx / ss)
    fys = (np.arange(ny * ss) + 0.5) * (dy / ss)
    fgx, fgy = np.meshgrid(fxs, fys)
    spacing = min(dx, dy) / ss
    fseg, fdist, ft = _assign_nearest_segment(net, real, fgx, fgy, spacing=spacing)
    c0 = ss // 2
    seg_of_cell = fseg[c0::ss, c0::ss]
    dist_c = fdist[c0::ss, c0::ss]
    t_of_cell = ft[c0::ss, c0::ss]
    realset = set(real)
    w_of_cell = np.array([net.segments[i].w if i in realset else 0.0
                          for i in range(len(net.segments))])[seg_of_cell]
    dist_edge = np.maximum(dist_c - w_of_cell / 2.0, 0.0)   # to channel EDGE
    in_channel = dist_c <= (w_of_cell / 2.0)

    nseg = len(net.segments)
    fq = np.repeat(np.repeat(q_wm2, ss, axis=0), ss, axis=1)
    P_seg = np.bincount(fseg.ravel(),
                        weights=(fq * (cell_area / (ss * ss))).ravel(),
                        minlength=nseg)          # W collected per segment
    P_total = float((q_wm2 * cell_area).sum())

    # ---- 3. march coolant temperature in flow order ---------------------
    up = np.array([s.a if p[s.a] >= p[s.b] else s.b for s in net.segments])
    dn = np.array([s.b if p[s.a] >= p[s.b] else s.a for s in net.segments])
    mdot_seg = cool.rho * np.abs(Qs)
    n_through = max(1, len([i for i in real if abs(Qs[i]) > 0]))
    mdot_floor = 0.01 * mdot_total / n_through
    stagnant = [i for i in real if mdot_seg[i] < mdot_floor and P_seg[i] > 0]
    # Stagnant branches (dead ends, balanced bridges in symmetric networks)
    # cannot carry their collected heat — physically it conducts through the
    # silicon to the nearest FLOWING channel. Reassign it there, and remember
    # the remap so their die cells read the flowing neighbour's coolant state
    # (with the extra conduction distance added to the spreading penalty).
    seg_remap = np.arange(nseg)
    extra_dist = np.zeros(nseg)
    flowing = [i for i in real if mdot_seg[i] >= mdot_floor]
    if stagnant and flowing:
        cent = {i: ((net.segments[i].p0[0] + net.segments[i].p1[0]) / 2,
                    (net.segments[i].p0[1] + net.segments[i].p1[1]) / 2)
                for i in real}
        for i in stagnant:
            j = min(flowing, key=lambda f: (cent[f][0] - cent[i][0]) ** 2
                                           + (cent[f][1] - cent[i][1]) ** 2)
            P_seg[j] += P_seg[i]
            P_seg[i] = 0.0
            seg_remap[i] = j
            extra_dist[i] = np.hypot(cent[j][0] - cent[i][0], cent[j][1] - cent[i][1])
        warnings.append(f"{len(stagnant)} zero-flow branch(es): their heat is "
                        "re-routed to the nearest flowing channel (conduction)")
    elif stagnant:
        warnings.append(f"{len(stagnant)} near-stagnant channel(s): local "
                        "temperatures there are unreliable")
    mdot_eff = np.maximum(mdot_seg, mdot_floor)

    T_up = np.full(nseg, inlet_temp_c)
    T_dn = np.full(nseg, inlet_temp_c)
    node_T = np.full(len(net.nodes), np.nan)
    node_T[net.inlet] = inlet_temp_c
    sumQ = np.zeros(len(net.nodes))
    sumQT = np.zeros(len(net.nodes))
    order = np.argsort(-p)                       # process nodes high→low pressure
    seg_at = {}
    for i in range(nseg):
        seg_at.setdefault(int(up[i]), []).append(i)
    for n in order:
        if np.isnan(node_T[n]):
            node_T[n] = (sumQT[n] / sumQ[n]) if sumQ[n] > 0 else inlet_temp_c
        for i in seg_at.get(int(n), []):
            T_up[i] = node_T[n]
            T_dn[i] = T_up[i] + P_seg[i] / max(mdot_eff[i] * cool.cp, 1e-12)
            q = abs(Qs[i])
            sumQ[dn[i]] += q
            sumQT[dn[i]] += q * T_dn[i]
    outlet_T = float(node_T[net.outlet])
    dT_exact = P_total / max(mdot_total * cool.cp, 1e-12)
    balance_err = abs((outlet_T - inlet_temp_c) - dT_exact) / max(dT_exact, 1e-9)

    # ---- 4. junction map --------------------------------------------------
    # per-segment wall superheat (segment-averaged H1 convection + fin effect)
    dT_wall = np.zeros(nseg)
    h_seg = np.zeros(nseg)
    eta_seg = np.ones(nseg)
    for i in real:
        s = net.segments[i]
        alpha = corr.aspect_ratio(s.w, s.H)
        Dh = max(s.Dh, 1e-12)
        h = corr.Nu_H1(alpha) * cool.k / Dh
        eta = corr.fin_efficiency(h, sol.k, max(s.w, 1e-6), s.H)
        A_wet = (s.w + 2.0 * eta * s.H) * max(s.length, 1e-9)
        dT_wall[i] = P_seg[i] / max(h * A_wet, 1e-12)
        h_seg[i] = h
        eta_seg[i] = eta

    # coolant temp at the nearest point of the assigned segment (stagnant
    # branches read through their remap to the flowing channel that actually
    # absorbs their heat, at the cost of extra conduction distance)
    own = seg_remap[seg_of_cell]
    Tf_near = T_up[own] + t_of_cell * (T_dn[own] - T_up[own])
    R_base = t_base / sol.k                       # K·m²/W under the channel floor
    d_eff = dist_edge + extra_dist[seg_of_cell]
    R_spread = d_eff ** 2 / (2.0 * sol.k * max(die_thk, 1e-6))
    Tj = Tf_near + dT_wall[own] + q_wm2 * (R_base + R_spread)

    Tf = np.where(in_channel, Tf_near, np.nan)    # display: coolant temp in channels

    # raster grids for display + design-rule checks (mask may under-resolve thin
    # channels; that is reported through the confidence 'resolution' check)
    raster = rasterize(net, seg_flow=Qs, coolant=cool, shape=(ny, nx))

    # ---- 5. metrics -------------------------------------------------------
    dP = flow["dP"]
    pumping_W = dP * Q_total_m3s
    v_seg = np.zeros(nseg)
    Re_seg = np.zeros(nseg)
    for i in real:
        s = net.segments[i]
        v_seg[i] = abs(Qs[i]) / max(s.area, 1e-15)
        Re_seg[i] = corr.reynolds(cool.rho, v_seg[i], max(s.Dh, 1e-12), cool.mu)
    Re_max = float(Re_seg.max()) if len(real) else 0.0
    w_min = min(net.segments[i].w for i in real)
    Dh_min = min(net.segments[i].Dh for i in real)
    Tj_max = float(np.nanmax(Tj)); Tj_min = float(np.nanmin(Tj)); Tj_mean = float(np.nanmean(Tj))
    R_th = (Tj_max - inlet_temp_c) / max(P_total, 1e-12)
    area = L * W
    mdot_w = mdot_seg[real] / max(mdot_seg[real].sum(), 1e-30)

    metrics = {
        "max_junction_temp": Tj_max,
        "mean_junction_temp": Tj_mean,
        "junction_gradient": Tj_max - Tj_min,
        "junction_cv": float(np.nanstd(Tj) / Tj_mean) if Tj_mean else 0.0,
        "coolant_dT": dT_exact,
        "coolant_outlet_temp": outlet_T,
        "thermal_resistance_KW": R_th,
        "thermal_resistance_Kcm2W": units.rth_to_area_normalised(R_th, area),
        "thermal_margin": junction_limit_c - Tj_max,
        "pressure_drop_kPa": units.to_kPa(dP),
        "pumping_power_W": pumping_W,
        "flow_mlmin": units.m3s_to_mlmin(Q_total_m3s),
        "COP": P_total / max(pumping_W, 1e-12),
        "Re": Re_max,
        "laminar": Re_max < 2300.0,
        "maldistribution_cv": flow["maldistribution_cv"],
        "total_power_W": P_total,
        "peak_flux_Wcm2": units.wm2_to_wcm2(float(q_wm2.max())),
        "Dh_um": units.to_um(Dh_min),
        "min_channel_um": units.to_um(w_min),
        "fin_efficiency": float((eta_seg[real] * mdot_w).sum()),
        "h_Wm2K": float((h_seg[real] * mdot_w).sum()),
        "n_segments": len(real),
        "n_dead_segments": n_dead,
        "n_stagnant_segments": len(stagnant),
        "energy_balance_error": float(balance_err),
    }

    seg_stats = [{
        "i": int(i),
        "q_mlmin": units.m3s_to_mlmin(abs(float(Qs[i]))),
        "v_ms": float(v_seg[i]),
        "re": float(Re_seg[i]),
        "dp_kpa": units.to_kPa(abs(float(p[net.segments[i].a] - p[net.segments[i].b]))),
        "t_in_c": float(T_up[i]),
        "t_out_c": float(T_dn[i]),
        "power_w": float(P_seg[i]),
        "stagnant": bool(i in stagnant),
    } for i in real]

    return NetworkThermalResult(metrics=metrics, Tj=Tj, Tf=Tf, raster=raster,
                                seg_stats=seg_stats, warnings=warnings)
