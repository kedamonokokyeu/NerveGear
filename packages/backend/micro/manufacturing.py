"""
micro/manufacturing.py

Etch rules and reliability checks for microchannel fabrication.

The thermal optimizer will happily find channel geometries that are
un-manufacturable or clog-prone. This module encodes the constraints it can't
violate: DRIE aspect ratio limits, minimum feature size, wall thickness,
floating-silicon checks, and clogging/maldistribution risk. Each check returns a
margin so the optimizer can treat them as soft penalties or hard constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import units


@dataclass
class EtchRules:
    min_feature_um: float = 20.0       # smallest etchable width
    max_aspect_ratio: float = 25.0     # DRIE depth/width ceiling
    min_wall_um: float = 25.0          # thinnest reliable wall between channels
    min_channel_um: float = 30.0       # clogging floor (particulate)


@dataclass
class Violation:
    rule: str
    ok: bool
    value: float
    limit: float
    margin: float                      # signed; >=0 means satisfied
    note: str = ""


def check_heatsink(hs, rules: EtchRules = EtchRules()) -> list:
    """Run the rule set against a MicroChannelHeatSink (straight parallel)."""
    w_um = units.to_um(hs.w_ch)
    wall_um = units.to_um(hs.w_wall)
    H_um = units.to_um(hs.H_ch)
    ar = H_um / max(w_um, 1e-9)
    V = []
    V.append(_v("min_feature", w_um >= rules.min_feature_um, w_um, rules.min_feature_um,
               w_um - rules.min_feature_um, "channel width vs litho/etch floor"))
    V.append(_v("min_channel_clog", w_um >= rules.min_channel_um, w_um, rules.min_channel_um,
               w_um - rules.min_channel_um, "channel width vs clogging floor"))
    V.append(_v("aspect_ratio", ar <= rules.max_aspect_ratio, ar, rules.max_aspect_ratio,
               rules.max_aspect_ratio - ar, "etch depth/width vs DRIE limit"))
    V.append(_v("min_wall", wall_um >= rules.min_wall_um, wall_um, rules.min_wall_um,
               wall_um - rules.min_wall_um, "wall thickness vs mechanical floor"))
    return V


def check_mask(mask: np.ndarray, dx: float, rules: EtchRules = EtchRules()) -> list:
    """Geometry-level checks on a rasterized network mask (1 = silicon, 0 =
    channel): minimum etched feature and FLOATING-SILICON detection (a solid
    island disconnected from the die body cannot be fabricated / would fall out)."""
    V = []
    # minimum channel feature from a distance transform on the void
    try:
        from scipy.ndimage import distance_transform_edt, label
    except Exception:
        return V
    void = mask < 0.5
    if void.any():
        # 2*EDT ≈ local channel width; min over the channel skeleton-ish (use the
        # max EDT as representative half-width of the narrowest-resolved feature)
        edt = distance_transform_edt(void) * dx
        min_feat_um = units.to_um(2.0 * float(edt[void].max())) if void.any() else 0.0
        # (max EDT is the widest; for a floor check we want the thinnest sustained
        # channel — approximate by the 5th percentile of local widths along voids)
        widths = 2.0 * edt[void]
        thin_um = units.to_um(float(np.percentile(widths[widths > 0], 5))) if (widths > 0).any() else 0.0
        V.append(_v("min_feature_mask", thin_um >= rules.min_feature_um, thin_um,
                   rules.min_feature_um, thin_um - rules.min_feature_um,
                   "narrowest resolved channel (mask)"))
    # floating silicon: solid components not touching the border
    solid = mask > 0.5
    lab, n = label(solid)
    border = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    border.discard(0)
    floating = [c for c in range(1, n + 1) if c not in border]
    V.append(_v("no_floating_silicon", len(floating) == 0, float(len(floating)), 0.0,
               -float(len(floating)), "solid islands disconnected from die body"))
    return V


def clogging_risk(hs) -> float:
    """A 0–1 risk index: narrower channels clog more easily. Heuristic, monotonic
    in 1/width below ~100 µm; meant as a soft objective, not a hard truth."""
    w_um = units.to_um(hs.w_ch)
    return float(np.clip((100.0 - w_um) / 100.0, 0.0, 1.0))


def summary(violations: list) -> dict:
    return {
        "manufacturable": all(v.ok for v in violations),
        "n_violations": sum(0 if v.ok else 1 for v in violations),
        "worst_margin": min((v.margin for v in violations), default=0.0),
        "details": [v.__dict__ for v in violations],
    }


def _v(rule, ok, value, limit, margin, note):
    return Violation(rule=rule, ok=bool(ok), value=float(value), limit=float(limit),
                     margin=float(margin), note=note)
