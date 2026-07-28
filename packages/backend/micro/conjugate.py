"""
micro/conjugate.py

Reduced-order conjugate heat transfer for microchannel heat sinks, in real units.

The dominant effect is coolant heating up as it travels through channels, making
downstream chips run hotter. Based on the validated MCHS thermal-resistance
approach (Tuckerman & Pease 1981). Fast enough for the interactive loop.

The high-fidelity CFD model can drop in later by replacing ConjugateModel.evaluate
— nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import correlations as corr
from . import units
from .materials import Coolant, Solid, solid as get_solid, coolant as get_coolant
from .powermap import PowerMap, uniform


@dataclass
class MicroChannelHeatSink:
    """A straight parallel microchannel heat sink etched into a silicon die — the
    analyzable workhorse the conjugate reduced model is exact-ish for, and the
    geometry the validation benchmarks use."""
    die_L: float = 10e-3          # flow-direction length [m]
    die_W: float = 10e-3          # transverse width [m]
    t_base: float = 100e-6        # silicon between channel floor and heated face [m]
    n_ch: int = 50                # number of parallel channels
    w_ch: float = 100e-6          # channel width [m]
    H_ch: float = 300e-6          # channel depth (etch) [m]
    solid: Solid = dc_field(default_factory=lambda: get_solid("silicon"))
    coolant: Coolant = dc_field(default_factory=lambda: get_coolant("water"))
    inlet_temp_c: float = 25.0
    junction_limit_c: float = 105.0

    @property
    def pitch(self) -> float:
        return self.die_W / self.n_ch

    @property
    def w_wall(self) -> float:
        return max(self.pitch - self.w_ch, 1e-9)

    @property
    def A_c(self) -> float:
        return self.w_ch * self.H_ch

    @property
    def Dh(self) -> float:
        return corr.hydraulic_diameter(self.w_ch, self.H_ch)


@dataclass
class ConjugateResult:
    metrics: dict
    Tj: np.ndarray                # (Ny, Nx) junction temperature map [°C]
    Tf_x: np.ndarray              # (Nx,) coolant bulk temp along flow [°C]
    fields: dict = dc_field(default_factory=dict)


def solve(hs: MicroChannelHeatSink, power: PowerMap | None = None,
          Q_total_m3s: float | None = None, u_in: float | None = None) -> ConjugateResult:
    """Evaluate a heat sink against a power map.

    Specify the flow either as total volumetric flow `Q_total_m3s` OR per-channel
    inlet velocity `u_in` (m/s). Returns junction map + headline metrics in real
    units (°C, kPa, W, K·cm²/W).
    """
    cool = hs.coolant
    power = power or uniform(100.0, hs.die_L, hs.die_W)

    # ---- flow rate ----
    if u_in is not None:
        Q_ch = u_in * hs.A_c
        Q_total = Q_ch * hs.n_ch
    else:
        if Q_total_m3s is None:
            Q_total_m3s = units.mlmin_to_m3s(200.0)   # sensible default 200 mL/min
        Q_total = Q_total_m3s
        Q_ch = Q_total / hs.n_ch
    u = Q_ch / hs.A_c
    mdot_total = cool.rho * Q_total                    # kg/s

    # ---- regime ----
    Re = corr.reynolds(cool.rho, u, hs.Dh, cool.mu)
    alpha = corr.aspect_ratio(hs.w_ch, hs.H_ch)
    Lh = corr.hydrodynamic_entrance_length(Re, hs.Dh)
    Lt = corr.thermal_entrance_length(Re, cool.Pr, hs.Dh)

    # ---- pressure drop (Fanning, fully developed + simple developing bump) ----
    fRe = corr.fRe_fanning(alpha)
    f = fRe / max(Re, 1e-9)
    dP = 2.0 * f * (hs.die_L / hs.Dh) * cool.rho * u * u      # Pa
    pumping_W = dP * Q_total

    # ---- convection + fin model ----
    Nu = corr.Nu_H1(alpha)
    h = Nu * cool.k / hs.Dh                                   # W/m²K
    eta = corr.fin_efficiency(h, hs.solid.k, hs.w_wall, hs.H_ch)
    h_eff = h * (hs.w_ch + 2.0 * eta * hs.H_ch) / hs.pitch    # per base area
    R_conv = 1.0 / max(h_eff, 1e-12)                          # K·m²/W
    R_cond = hs.t_base / hs.solid.k                           # K·m²/W
    R_pp = R_conv + R_cond                                    # area-specific R''

    # ---- power map on a grid; flow along +x ----
    Ny, Nx = power.flux.shape
    q = power.flux                                            # W/m² (Ny, Nx)
    cell_dx = hs.die_L / Nx
    cell_dy = hs.die_W / Ny
    # cumulative power upstream of each x-column -> coolant bulk temp T_f(x)
    col_power = q.sum(axis=0) * cell_dx * cell_dy             # W per x-column
    cum = np.cumsum(col_power) - 0.5 * col_power              # midpoint of each slab
    Tf_x = hs.inlet_temp_c + cum / max(mdot_total * cool.cp, 1e-12)
    # junction map: coolant temp at that x + local q''·R''
    Tj = Tf_x[np.newaxis, :] + q * R_pp

    P_total = power.total_power_W
    dT_coolant = P_total / max(mdot_total * cool.cp, 1e-12)
    Tj_max = float(Tj.max()); Tj_min = float(Tj.min()); Tj_mean = float(Tj.mean())
    R_th = (Tj_max - hs.inlet_temp_c) / max(P_total, 1e-12)   # K/W
    area = hs.die_L * hs.die_W

    metrics = {
        # thermal (the headline numbers)
        "max_junction_temp": Tj_max,
        "mean_junction_temp": Tj_mean,
        "junction_gradient": Tj_max - Tj_min,                # on-die uniformity (warpage driver)
        "junction_cv": float(np.std(Tj) / Tj_mean) if Tj_mean else 0.0,
        "coolant_dT": dT_coolant,
        "thermal_resistance_KW": R_th,
        "thermal_resistance_Kcm2W": units.rth_to_area_normalised(R_th, area),
        "thermal_margin": hs.junction_limit_c - Tj_max,      # >0 = safe
        # hydraulic
        "pressure_drop_kPa": units.to_kPa(dP),
        "pumping_power_W": pumping_W,
        "flow_mlmin": units.m3s_to_mlmin(Q_total),
        "COP": P_total / max(pumping_W, 1e-12),               # heat removed / pump work
        # regime / diagnostics
        "Re": Re, "laminar": Re < 2300.0, "aspect_ratio": alpha,
        "Nu": Nu, "h_Wm2K": h, "fin_efficiency": eta,
        "Dh_um": units.to_um(hs.Dh),
        "thermal_developing_fraction": float(min(1.0, Lt / hs.die_L)),
        "total_power_W": P_total, "peak_flux_Wcm2": power.peak_wcm2,
    }
    return ConjugateResult(metrics=metrics, Tj=Tj, Tf_x=Tf_x,
                           fields={"q_Wm2": q, "R_pp": R_pp})


class ConjugateModel:
    """Thin adapter exposing the analytical model as an injectable evaluator, so it
    can be swapped for the trained surrogate behind one call (mirrors NerveGear's
    existing `velocity_provider` injection pattern). The optimizer and UI call
    `evaluate(params) -> metrics`; whether the metrics come from this closed-form
    model or your friend's trained network is invisible to them."""

    def __init__(self, base: MicroChannelHeatSink, power: PowerMap):
        self.base = base
        self.power = power

    def evaluate(self, params: dict) -> dict:
        """params -> metrics. Recognises w_ch, H_ch, n_ch, t_base [m] and
        flow_mlmin. Unknown params are ignored, so the same evaluator serves
        different optimization specs.

        To upgrade fidelity, replace the `solve(...)` call below with a call to the
        trained surrogate that returns the same metrics dict — nothing upstream
        changes."""
        from dataclasses import replace
        b = self.base
        hs = replace(
            b,
            w_ch=float(params.get("w_ch", b.w_ch)),
            H_ch=float(params.get("H_ch", b.H_ch)),
            n_ch=int(round(params.get("n_ch", b.n_ch))),
            t_base=float(params.get("t_base", b.t_base)),
        )
        Q = units.mlmin_to_m3s(params["flow_mlmin"]) if "flow_mlmin" in params else None
        return solve(hs, self.power, Q_total_m3s=Q).metrics
