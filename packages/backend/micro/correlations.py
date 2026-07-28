"""
micro/correlations.py

Closed-form laminar microchannel correlations from Shah & London (1978).

Microchannels run at low Reynolds number (Re < ~1000) — firmly laminar — so
validated closed-form expressions capture most of the physics without a trained
network. Covers friction (fRe), Nusselt number, entrance lengths, and fin
efficiency for rectangular ducts.
"""

from __future__ import annotations

import math

# ── friction: Fanning fRe for a rectangular duct, fully developed laminar ──
# Shah & London polynomial. Limits: α->0 (parallel plates) fRe=24; α=1 (square)
# fRe=14.23.  Darcy f = 4 × Fanning.
def fRe_fanning(alpha: float) -> float:
    a = _clip01(alpha)
    return 24.0 * (1 - 1.3553 * a + 1.9467 * a**2 - 1.7012 * a**3
                   + 0.9564 * a**4 - 0.2537 * a**5)


# ── Nusselt: rectangular duct, fully developed, constant heat flux (H1) ─────
# Shah & London polynomial. Limits: α->0 Nu=8.235 (parallel plates, both walls
# heated); α=1 Nu=3.61 (square duct).
def Nu_H1(alpha: float) -> float:
    a = _clip01(alpha)
    return 8.235 * (1 - 2.0421 * a + 3.0853 * a**2 - 2.4765 * a**3
                    + 1.0578 * a**4 - 0.1861 * a**5)


def reynolds(rho: float, u: float, Dh: float, mu: float) -> float:
    """Re = ρ u D_h / µ.  < ~2300 laminar; microchannels usually < ~1000."""
    return rho * u * Dh / mu


def hydrodynamic_entrance_length(Re: float, Dh: float) -> float:
    """L_h ≈ 0.05 · Re · D_h  — distance for the velocity profile to develop."""
    return 0.05 * Re * Dh


def thermal_entrance_length(Re: float, Pr: float, Dh: float) -> float:
    """L_t ≈ 0.05 · Re · Pr · D_h.  For water (Pr≈6) this is several × longer than
    L_h, so much of a short channel is THERMALLY developing — the regime where the
    coolant heat-up (pivot 2) bites hardest."""
    return 0.05 * Re * Pr * Dh


def developing_nu_factor(x_over_LtGz: float) -> float:
    """Crude entrance enhancement: thermally developing flow has higher local Nu
    near the inlet. Returns a multiplier ≥ 1 on Nu_fd as a function of the inverse
    Graetz position. A conservative engineering bump; replace with the full Graetz
    series or a CFD-trained correction when the surrogate lands."""
    if x_over_LtGz <= 0:
        return 1.0
    # blends ~1.3x near inlet down to 1.0 fully developed
    return 1.0 + 0.3 * math.exp(-3.0 * x_over_LtGz)


def fin_efficiency(h: float, k_solid: float, t_wall: float, H_channel: float) -> float:
    """Channel side-walls between channels are FINS: heat enters at the base and
    convects off both faces along the height. Efficiency η = tanh(mH)/(mH) with
    m = sqrt(2h/(k·t)). Tall thin walls (high aspect microchannels) lose efficiency
    — a key manufacturability/performance trade (deep etch helps area but hurts η
    and raises ΔP)."""
    if t_wall <= 0 or H_channel <= 0:
        return 1.0
    m = math.sqrt(2.0 * h / (k_solid * t_wall))
    mH = m * H_channel
    return math.tanh(mH) / mH if mH > 1e-9 else 1.0


def hydraulic_diameter(w: float, H: float) -> float:
    """D_h = 4 A_c / P_wetted for a rectangular channel w × H."""
    A = w * H
    P = 2.0 * (w + H)
    return 4.0 * A / P if P > 0 else 0.0


def aspect_ratio(w: float, H: float) -> float:
    if w <= 0 or H <= 0:
        return 1.0
    return min(w, H) / max(w, H)


def _clip01(a: float) -> float:
    return 0.0 if a < 0 else 1.0 if a > 1 else a
