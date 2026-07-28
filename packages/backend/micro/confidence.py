"""
micro/confidence.py

The confidence score, with a defined meaning (build plan, Phase 2).

`score` answers ONE question: "how much should you trust the metrics for this
design?" It is the weighted fraction of named physical-validity checks that
pass. It is not a statistical error bar. The UI must show the grade and any
failing checks verbatim — the checks ARE the meaning.

Checks (weights sum to 1):
  laminar               the correlations assume laminar flow (Re < 2300)
  energy_balance        first-law closure of the coolant march (< 2%)
  validated_envelope    channel size inside the range the reduced model was
                        benchmarked on (20 µm–2 mm width, aspect ≤ 30)
  resolution            ≥ 2 raster cells across the narrowest channel
  flow_distribution     maldistribution CV < 0.5 and no stagnant channels
  manufacturable        no hard etch-rule violations
  engine_fidelity       full 3-D field engine vs reduced-order model

Every engine (reduced today, solver3d/surrogate later) reports through this
same rubric; the surrogate adds `surrogate_in_distribution` when it lands.
"""

from __future__ import annotations

import numpy as np

GRADE_HIGH = 0.80
GRADE_MEDIUM = 0.50


def _check(name, ok, weight, detail):
    return {"name": name, "ok": bool(ok), "weight": float(weight), "detail": str(detail)}


def assess(*, engine: str, metrics: dict, manufacturability: dict | None = None,
           raster: dict | None = None, extra_checks: list | None = None) -> dict:
    """Build the confidence block for a solve response.

    `metrics` uses the canonical names (docs/CONTRACT.md). Missing information
    degrades gracefully: a check that cannot be evaluated counts as passed at
    reduced weight rather than failing the design for lack of data.
    """
    checks = []

    Re = float(metrics.get("Re", 0.0))
    checks.append(_check(
        "laminar", Re < 2300.0, 0.22,
        f"max channel Re {Re:.0f} {'<' if Re < 2300 else '≥'} 2300 "
        f"({'laminar — correlations valid' if Re < 2300 else 'transitional/turbulent — out of model range'})"))

    ebe = metrics.get("energy_balance_error")
    if ebe is not None:
        ok = float(ebe) < 0.02
        checks.append(_check("energy_balance", ok, 0.15,
                             f"coolant enthalpy rise closes to {float(ebe) * 100:.2f}% "
                             f"of P/(ṁ·cp) ({'ok' if ok else 'poor closure'})"))

    w_um = metrics.get("min_channel_um")
    if w_um is not None:
        Dh_um = float(metrics.get("Dh_um", w_um))
        ar_ok = True
        detail = f"narrowest channel {float(w_um):.0f} µm, Dh {Dh_um:.0f} µm"
        ok = 20.0 <= float(w_um) <= 2000.0 and ar_ok
        checks.append(_check("validated_envelope", ok, 0.15,
                             detail + (" — inside benchmarked range" if ok
                                       else " — outside 20 µm–2 mm benchmarked range")))

    if raster is not None and w_um is not None:
        cell_um = min(raster.get("dx", 1.0), raster.get("dy", 1.0)) * 1e6
        cells_across = float(w_um) / max(cell_um, 1e-9)
        ok = cells_across >= 2.0
        checks.append(_check("resolution", ok, 0.10,
                             f"{cells_across:.1f} raster cells across the narrowest channel "
                             f"({'adequate' if ok else 'under-resolved — refine or widen'})"))

    cv = metrics.get("maldistribution_cv")
    n_stag = int(metrics.get("n_stagnant_segments", 0))
    if cv is not None:
        ok = float(cv) < 0.5 and n_stag == 0
        d = f"flow CV across channels {float(cv):.2f}"
        if n_stag:
            d += f", {n_stag} near-stagnant channel(s)"
        checks.append(_check("flow_distribution", ok, 0.13,
                             d + (" — well distributed" if ok else " — uneven; local temps less reliable")))

    if manufacturability is not None:
        ok = bool(manufacturability.get("manufacturable", True))
        nv = int(manufacturability.get("n_violations", 0))
        checks.append(_check("manufacturable", ok, 0.10,
                             "all etch rules pass" if ok else f"{nv} etch-rule violation(s)"))

    hi_fi = engine in ("solver3d", "surrogate")
    checks.append(_check(
        "engine_fidelity", hi_fi, 0.15,
        "full 3-D field engine" if hi_fi else
        "reduced-order model: exact laminar hydraulics + correlation thermal — "
        "trust trends and scaling; local fields are approximations"))

    for c in extra_checks or []:
        checks.append(c)

    wsum = sum(c["weight"] for c in checks) or 1.0
    score = sum(c["weight"] for c in checks if c["ok"]) / wsum
    grade = "high" if score >= GRADE_HIGH else "medium" if score >= GRADE_MEDIUM else "low"
    return {"score": round(float(score), 3), "grade": grade, "checks": checks,
            "meaning": "weighted fraction of physical-validity checks passed — "
                       "see each check's detail; not a statistical error bar"}


def for_error(engine: str, message: str) -> dict:
    return {"score": 0.0, "grade": "low",
            "checks": [_check("solve", False, 1.0, message)],
            "meaning": "solve failed"}
