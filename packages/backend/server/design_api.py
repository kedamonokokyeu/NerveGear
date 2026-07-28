"""
server/design_api.py

The design endpoints — the seam between the Studio and the physics,
speaking `axiom/solve.v1` (contract/schema.py — the schema as code).

  POST /api/evaluate       fast reduced solve (network or Hele-Shaw) → metrics,
                           confidence, manufacturability, Tj/Tf maps, per-segment flow
  POST /api/solve          same physics behind the full-solve shape; the 3-D
                           field engine slot is empty (removed 2026-07, to be
                           rebuilt) and `engine` is always tagged honestly
  POST /api/optimize       constrained search over a design's knobs → Pareto
                           history + best feasible designs

All handlers parse/validate through schema.parse_request → SchemaError becomes
a 422 with the exact JSON path.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from fastapi import APIRouter, HTTPException

# The solve.v1 contract lives in-package now (contract/schema.py); the 3-D
# engine that used to own it was removed 2026-07 to be rebuilt from scratch.
from contract import schema  # noqa: E402

from micro import confidence as conf  # noqa: E402
from micro import geometry as mgeo  # noqa: E402
from micro import heleshaw, manufacturing, network_thermal, units  # noqa: E402
from micro.manufacturing import EtchRules  # noqa: E402
from micro.powermap import Hotspot, PowerMap, from_hotspots, uniform  # noqa: E402
from micro.units import UM  # noqa: E402

router = APIRouter()

ENGINE_VERSION = "micro-reduced-1.0"


# ── request → micro-layer objects ────────────────────────────────────────

def _power_map(req: schema.SolveRequest) -> PowerMap:
    p = req.conditions.power
    die_L = req.geometry.die.L * UM
    die_W = req.geometry.die.W * UM
    if p.grid is not None:
        return PowerMap(flux=units.wcm2_to_wm2(np.asarray(p.grid, dtype=float)),
                        die_L=die_L, die_W=die_W)
    hs = [Hotspot(h.cx, h.cy, h.sigma, h.peak_wcm2) for h in p.hotspots]
    if hs:
        return from_hotspots(hs, die_L=die_L, die_W=die_W,
                             background_wcm2=p.background_wcm2)
    return uniform(p.background_wcm2, die_L, die_W)


def _parametric_segments(geom: schema.Geometry) -> list:
    """Expand a parametric pattern into explicit segments (µm) via the single
    Python pattern source, micro/geometry.py."""
    kind = (geom.parametric or {}).get("kind")
    prm = (geom.parametric or {}).get("params") or {}
    die_L, die_W = geom.die.L * UM, geom.die.W * UM
    max_depth = geom.die.thk - geom.die.base_thk
    H = min(float(prm.get("H_um", 300.0)), max_depth) * UM
    if kind == "straight_parallel":
        net = mgeo.straight_parallel(
            n_ch=int(prm.get("n_ch", 40)), w=float(prm.get("w_um", 100.0)) * UM,
            H=H, L=die_L, pitch=(float(prm["pitch_um"]) * UM if "pitch_um" in prm else None),
            header_w=float(prm.get("header_w_um", 600.0)) * UM)
    elif kind == "murray_tree":
        net = mgeo.murray_tree(
            levels=int(prm.get("levels", 5)), w_trunk=float(prm.get("w_trunk_um", 800.0)) * UM,
            H=H, die_L=die_L, die_W=die_W, exponent=float(prm.get("exponent", 3.0)))
    else:
        raise schema.SchemaError("$.geometry.parametric.kind",
                                 f"unknown kind {kind!r} (want straight_parallel|murray_tree)")
    segs = []
    for s in net.segments:
        if s.w <= 0:
            continue
        segs.append({"x0": s.p0[0] / UM, "y0": s.p0[1] / UM,
                     "x1": s.p1[0] / UM, "y1": s.p1[1] / UM,
                     "w": s.w / UM, "depth": s.H / UM, "level": s.level})
    return segs


def _grid_for(geom: schema.Geometry, quality: str) -> tuple:
    """Raster shape for maps/DRC: match die aspect, cap cell count."""
    target = {"draft": 96, "normal": 144, "fine": 192}.get(quality, 96)
    L, W = geom.die.L, geom.die.W
    nx = target if L >= W else max(32, int(round(target * L / W)))
    ny = target if W > L else max(32, int(round(target * W / L)))
    return ny, nx


def _keep_out_violations(geom: schema.Geometry, etched_mask: np.ndarray) -> list:
    """Etched cells inside keep-out rectangles (TSV arrays, power delivery...)."""
    if not geom.keep_out:
        return []
    ny, nx = etched_mask.shape
    dx = geom.die.L / nx
    dy = geom.die.W / ny
    out = []
    for k in geom.keep_out:
        j0, j1 = sorted((int(k.x0 / dx), int(k.x1 / dx)))
        i0, i1 = sorted((int(k.y0 / dy), int(k.y1 / dy)))
        sub = etched_mask[max(0, i0):min(ny, i1 + 1), max(0, j0):min(nx, j1 + 1)]
        n_bad = int(sub.sum())
        if n_bad:
            out.append({"rule": "keep_out", "ok": False, "value": float(n_bad),
                        "limit": 0.0, "margin": -float(n_bad),
                        "note": f"channel crosses keep-out ({k.reason or 'reserved'})"})
    return out


def _manufacturability(geom: schema.Geometry, mask_solid: np.ndarray, dx_m: float,
                       segments: list | None) -> dict:
    rules = EtchRules()
    viols = manufacturing.check_mask(mask_solid, dx_m, rules)
    # partial etches leave every island anchored to the un-etched base — the
    # through-cut floating-silicon rule downgrades to informational
    for v in viols:
        if v.rule == "no_floating_silicon" and not v.ok:
            v.ok = True
            v.margin = 0.0
            v.note = ("silicon island(s) anchored by the un-etched base "
                      "(partial etch); verify lid bonding for tall thin islands")
    if segments:
        w_min = min(s["w"] for s in segments)
        d_max = max(s["depth"] for s in segments)
        ar = d_max / max(w_min, 1e-9)
        viols.append(manufacturing._v(
            "aspect_ratio", ar <= rules.max_aspect_ratio, ar, rules.max_aspect_ratio,
            rules.max_aspect_ratio - ar, "etch depth/width vs DRIE limit"))
        viols.append(manufacturing._v(
            "min_feature", w_min >= rules.min_feature_um, w_min, rules.min_feature_um,
            w_min - rules.min_feature_um, "channel width vs litho/etch floor"))
    summary = manufacturing.summary(viols)
    etched = (mask_solid < 0.5).astype(np.uint8)
    ko = _keep_out_violations(geom, etched)
    if ko:
        summary["details"].extend(ko)
        summary["n_violations"] += len(ko)
        summary["manufacturable"] = False
    # cells to paint red: narrow features + keep-out overlaps
    cells = []
    try:
        from scipy.ndimage import distance_transform_edt
        void = mask_solid < 0.5
        if void.any():
            widths = 2.0 * distance_transform_edt(void) * dx_m
            bad = void & (widths > 0) & (widths < rules.min_feature_um * 1e-6)
            ys, xs = np.where(bad)
            cells = np.column_stack([ys, xs]).tolist()[:2000]
    except Exception:  # noqa: BLE001
        pass
    summary["violation_cells"] = cells
    return summary


# ── the reduced solve shared by /api/evaluate and the /api/solve fallback ─

def _reduced_solve(req: schema.SolveRequest, quality: str | None = None) -> dict:
    t0 = time.time()
    geom = req.geometry
    cond = req.conditions
    power = _power_map(req)
    ny, nx = _grid_for(geom, quality or req.solve.resolution)
    die_L, die_W = geom.die.L * UM, geom.die.W * UM
    t_base = geom.die.base_thk * UM
    die_thk = geom.die.thk * UM
    Q = units.mlmin_to_m3s(cond.flow_mlmin)
    warnings: list[str] = []

    segments = [
        {"x0": s.x0, "y0": s.y0, "x1": s.x1, "y1": s.y1,
         "w": s.w, "depth": s.depth, "level": s.level}
        for s in geom.segments
    ]
    if geom.parametric is not None:
        segments = segments + _parametric_segments(geom)

    if segments and geom.etch is None:
        # centreline network → exact laminar hydraulics + junction-ordered march
        net, info = mgeo.network_from_segments(
            [{**s, "x0": s["x0"] * UM, "y0": s["y0"] * UM, "x1": s["x1"] * UM,
              "y1": s["y1"] * UM, "w": s["w"] * UM, "depth": s["depth"] * UM}
             for s in segments],
            die_L, die_W, inlet_edge=geom.inlet_edge, outlet_edge=geom.outlet_edge)
        res = network_thermal.solve_network(
            net, power, coolant=cond.coolant, solid=geom.die.solid,
            Q_total_m3s=Q, t_base=t_base, die_thk=die_thk,
            inlet_temp_c=cond.inlet_temp_c, junction_limit_c=cond.junction_limit_c,
            grid_shape=(ny, nx))
        mask_solid = res.raster["mask"]
        dx_m = res.raster["dx"]
        maps = {"tj_c": res.Tj, "tf_c": res.Tf}
        seg_stats = res.seg_stats
        model = "network"
        warnings += res.warnings
    else:
        # freeform etch (optionally with segments stamped in) → Hele-Shaw.
        # Solve on the etch canvas's native grid — resampling a painted design
        # corrupts walls — capped for the sparse solve.
        if geom.etch is not None:
            ey, ex = geom.etch.shape
            cap = 240
            if max(ey, ex) > cap:
                f = cap / max(ey, ex)
                ny, nx = max(32, int(ey * f)), max(32, int(ex * f))
            else:
                ny, nx = ey, ex
        from contract.rasterize import depth_map
        d_um = depth_map(geom, ny, nx)
        hs = heleshaw.solve_etch(
            d_um.astype(float) * UM, die_L, die_W, power,
            coolant=cond.coolant, solid=geom.die.solid, Q_total_m3s=Q,
            t_base=t_base, die_thk=die_thk, inlet_temp_c=cond.inlet_temp_c,
            junction_limit_c=cond.junction_limit_c,
            inlet_edge=geom.inlet_edge, outlet_edge=geom.outlet_edge)
        mask_solid = (1 - hs.open_mask).astype(np.uint8)
        dx_m = die_L / nx
        maps = {"tj_c": hs.Tj, "tf_c": hs.Tf,
                "p_kpa": hs.pressure / 1e3,
                "speed_ms": np.hypot(hs.vx, hs.vy)}
        seg_stats = []
        res = hs
        model = "heleshaw"
        warnings += hs.warnings

    mfg = _manufacturability(geom, mask_solid, dx_m, segments or None)
    metrics = dict(res.metrics)
    # the resolution check gauges solve fidelity: it applies to raster-solved
    # models (heleshaw); the network engine's physics is geometric — its
    # raster is only display + DRC
    raster_info = {"dx": dx_m, "dy": die_W / ny} if model == "heleshaw" else None
    cscore = conf.assess(engine="reduced", metrics=metrics,
                         manufacturability=mfg, raster=raster_info)
    q_map = units.wm2_to_wcm2(power.flux)
    maps["q_wcm2"] = q_map
    out = schema.make_response(
        engine="reduced", metrics=metrics, confidence=cscore,
        manufacturability=mfg,
        maps={k: np.nan_to_num(np.asarray(v, dtype=np.float32), nan=-1.0)
              for k, v in maps.items() if v is not None},
        segments=seg_stats,
        meta={"solve_ms": int((time.time() - t0) * 1000),
              "engine_version": ENGINE_VERSION, "model": model,
              "grid": [ny, nx], "warnings": warnings})
    return out


def _parse(payload: dict) -> schema.SolveRequest:
    try:
        return schema.parse_request(payload)
    except schema.SchemaError as e:
        raise HTTPException(422, str(e)) from e


# ── endpoints ────────────────────────────────────────────────────────────

@router.post("/api/evaluate")
async def api_evaluate(payload: dict) -> dict:
    """Fast reduced solve — the Studio's live loop."""
    req = _parse(payload)
    try:
        return _reduced_solve(req)
    except ValueError as e:
        return schema.make_response(
            engine="reduced", metrics={}, confidence=conf.for_error("reduced", str(e)),
            meta={"error": str(e)})


@router.post("/api/solve")
async def api_solve(payload: dict) -> dict:
    """Full solve, served by the validated reduced model.

    The schema still accepts engine "auto" / "solver3d" so stored designs
    keep parsing, but the 3-D field engine was removed (2026-07, to be
    rebuilt from scratch); until it returns, "solver3d" is an honest 503 and
    "auto" falls straight through to the reduced model, tagged as such.
    """
    req = _parse(payload)
    if req.solve.engine == "solver3d":
        raise HTTPException(
            503, "solver3d does not exist yet (the 3-D engine will be rebuilt) "
                 "— use engine 'auto' or 'reduced'")
    out = _reduced_solve(req, quality="normal")
    out["meta"]["note"] = ("3-D field engine not available yet — served by the "
                           "validated reduced model behind the same schema")
    return out


@router.post("/api/optimize")
async def api_optimize(payload: dict) -> dict:
    """Constrained random search over design knobs.

    body: { "request": <solve.v1 request>,
            "n_iter": 120,
            "vary": {"flow_mlmin": [50, 800], "w_scale": [0.6, 2.0],
                     "depth_scale": [0.7, 1.5]},
            "constraints": {"pressure_drop_kPa": 50, "min_channel_um": 30} }

    Objective: minimize max_junction_temp, tie-broken by pressure drop.
    Returns full history for the Pareto view + the best feasible design.
    """
    base = _parse(payload.get("request") or {})
    n_iter = int(np.clip(int(payload.get("n_iter", 100)), 4, 400))
    vary = payload.get("vary") or {"flow_mlmin": [50.0, 800.0],
                                   "w_scale": [0.7, 1.8], "depth_scale": [0.8, 1.4]}
    cons = payload.get("constraints") or {}
    dP_lim = float(cons.get("pressure_drop_kPa", 50.0))
    w_min_lim = float(cons.get("min_channel_um", EtchRules().min_channel_um))
    rules = EtchRules()
    rng = np.random.default_rng(int(payload.get("seed", 0)))

    if not base.geometry.segments and base.geometry.parametric is None:
        raise HTTPException(422, "optimize needs segments or a parametric pattern "
                                 "(freeform etch tuning: vary flow only)")

    base_segments = [
        {"x0": s.x0, "y0": s.y0, "x1": s.x1, "y1": s.y1, "w": s.w,
         "depth": s.depth, "level": s.level} for s in base.geometry.segments]
    if base.geometry.parametric is not None:
        base_segments += _parametric_segments(base.geometry)
    max_depth = base.geometry.die.thk - base.geometry.die.base_thk

    history = []
    t0 = time.time()
    for it in range(n_iter):
        smp = {}
        for k, (lo, hi) in vary.items():
            smp[k] = float(rng.uniform(float(lo), float(hi)))
        ws = smp.get("w_scale", 1.0)
        ds = smp.get("depth_scale", 1.0)
        segs = [{**s, "w": s["w"] * ws, "depth": min(s["depth"] * ds, max_depth)}
                for s in base_segments]
        w_min = min(s["w"] for s in segs)
        try:
            net, _ = mgeo.network_from_segments(
                [{**s, "x0": s["x0"] * UM, "y0": s["y0"] * UM, "x1": s["x1"] * UM,
                  "y1": s["y1"] * UM, "w": s["w"] * UM, "depth": s["depth"] * UM}
                 for s in segs],
                base.geometry.die.L * UM, base.geometry.die.W * UM,
                inlet_edge=base.geometry.inlet_edge, outlet_edge=base.geometry.outlet_edge)
            res = network_thermal.solve_network(
                net, _power_map(base), coolant=base.conditions.coolant,
                solid=base.geometry.die.solid,
                Q_total_m3s=units.mlmin_to_m3s(smp.get("flow_mlmin",
                                                       base.conditions.flow_mlmin)),
                t_base=base.geometry.die.base_thk * UM,
                die_thk=base.geometry.die.thk * UM,
                inlet_temp_c=base.conditions.inlet_temp_c,
                junction_limit_c=base.conditions.junction_limit_c,
                grid_shape=(48, 64))
            m = res.metrics
        except (ValueError, np.linalg.LinAlgError) as e:
            history.append({"i": it, "params": smp, "feasible": False,
                            "error": str(e)[:120]})
            continue
        ar = max(s["depth"] / max(s["w"], 1e-9) for s in segs)
        feasible = (m["pressure_drop_kPa"] <= dP_lim and w_min >= w_min_lim
                    and ar <= rules.max_aspect_ratio and m["laminar"])
        history.append({"i": it, "params": {k: round(v, 3) for k, v in smp.items()},
                        "tj": round(m["max_junction_temp"], 2),
                        "dp_kpa": round(m["pressure_drop_kPa"], 2),
                        "cop": round(m["COP"], 1),
                        "re": round(m["Re"], 0),
                        "feasible": bool(feasible)})
        if time.time() - t0 > 25:
            break

    scored = [h for h in history if h.get("feasible")]
    scored.sort(key=lambda h: (h["tj"], h["dp_kpa"]))
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "n_evaluated": len(history),
        "n_feasible": len(scored),
        "constraints": {"pressure_drop_kPa": dP_lim, "min_channel_um": w_min_lim,
                        "max_aspect_ratio": rules.max_aspect_ratio, "laminar": True},
        "best": scored[:5],
        "history": history,
        "meta": {"elapsed_ms": int((time.time() - t0) * 1000)},
    }
