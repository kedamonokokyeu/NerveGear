"""
micro/units.py

Unit conversion helpers for the microscale layer.

The macro model used relative units. Microscale work needs real numbers: µm,
W/cm², kPa, °C, mL/min. Everything internally stays SI; this module handles the
conversion at the boundary.
"""

from __future__ import annotations

# ── length ──────────────────────────────────────────────────────────────
UM = 1e-6          # micrometre  -> m   (channel widths/depths live here)
MM = 1e-3          # millimetre  -> m   (die footprints)
CM = 1e-2          # centimetre  -> m

um = lambda x: x * UM            # noqa: E731  (tiny converters read better inline)
mm = lambda x: x * MM            # noqa: E731
to_um = lambda x: x / UM         # noqa: E731
to_mm = lambda x: x / MM         # noqa: E731

# ── pressure ────────────────────────────────────────────────────────────
KPA = 1e3          # kilopascal -> Pa
BAR = 1e5
PSI = 6894.757
to_kPa = lambda pa: pa / KPA      # noqa: E731

# ── volumetric flow ─────────────────────────────────────────────────────
# Coolant pumps are speced in mL/min; the physics needs m^3/s.
ML_PER_MIN = 1e-6 / 60.0          # mL/min -> m^3/s
def mlmin_to_m3s(q_mlmin: float) -> float:
    return q_mlmin * ML_PER_MIN
def m3s_to_mlmin(q_m3s: float) -> float:
    return q_m3s / ML_PER_MIN

# ── power / heat flux ────────────────────────────────────────────────────
# Datasheets give heat flux in W/cm^2; the PDE needs W/m^2.
def wcm2_to_wm2(q_wcm2: float) -> float:
    return q_wcm2 / (CM * CM)     # 1 W/cm^2 = 1e4 W/m^2
def wm2_to_wcm2(q_wm2: float) -> float:
    return q_wm2 * (CM * CM)

# ── temperature ──────────────────────────────────────────────────────────
def c_to_k(t_c: float) -> float:
    return t_c + 273.15
def k_to_c(t_k: float) -> float:
    return t_k - 273.15

# ── thermal-resistance reporting helpers ─────────────────────────────────
# R_th in K/W is scale-dependent; chip people normalise by area -> K·cm²/W.
def rth_to_area_normalised(rth_KW: float, area_m2: float) -> float:
    """K/W -> K·cm²/W (multiply by area in cm²)."""
    return rth_KW * (area_m2 / (CM * CM))
