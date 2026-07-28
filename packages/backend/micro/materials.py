"""
micro/materials.py

Material property tables for solids (silicon, copper) and coolants (water, glycol,
dielectric).

Microscale conjugate heat transfer is dominated by how well silicon spreads heat
into channel walls and the coolant's thermal and viscous properties. Coolant choice
alone changes viscosity by ~30x, so these are first-class named values, not knobs.
All values SI, near 300–330 K.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Solid:
    name: str
    k: float            # thermal conductivity   [W/m·K]
    rho: float          # density                [kg/m³]
    cp: float           # specific heat          [J/kg·K]


@dataclass(frozen=True)
class Coolant:
    name: str
    rho: float          # density                [kg/m³]
    mu: float           # dynamic viscosity      [Pa·s]
    cp: float           # specific heat          [J/kg·K]
    k: float            # thermal conductivity   [W/m·K]

    @property
    def Pr(self) -> float:
        """Prandtl number = momentum diffusivity / thermal diffusivity = mu·cp/k.
        Sets how much longer the thermal entrance region is than the hydrodynamic
        one (L_t ≈ Pr · L_h) — central to the streamwise heat-up story."""
        return self.mu * self.cp / self.k

    @property
    def nu(self) -> float:
        """Kinematic viscosity mu/rho [m²/s]."""
        return self.mu / self.rho

    def at(self, T_c: float) -> "Coolant":
        """Hook for temperature-dependent properties (identity for now). Wiring a
        T-dependence in later changes nothing downstream — callers already ask the
        coolant for properties rather than hard-coding them."""
        return self


# ── solids (the die + heat-spreader candidates) ──────────────────────────
SOLIDS = {
    "silicon": Solid("Silicon", k=148.0, rho=2330.0, cp=705.0),     # the die itself
    "copper":  Solid("Copper",  k=398.0, rho=8960.0, cp=385.0),
    "aluminium": Solid("Aluminium", k=205.0, rho=2700.0, cp=900.0),
    "sic":     Solid("Silicon carbide", k=370.0, rho=3210.0, cp=690.0),  # WBG power dies
}

# ── coolants (the viscosity/Pr knob the user called out) ──────────────────
COOLANTS = {
    "water":     Coolant("Water (300 K)",            rho=996.0,  mu=8.5e-4, cp=4180.0, k=0.613),
    "water_glycol": Coolant("Water/glycol 25% PG",   rho=1020.0, mu=2.1e-3, cp=3800.0, k=0.430),
    "hfe7100":   Coolant("3M HFE-7100 (dielectric)", rho=1510.0, mu=5.8e-4, cp=1183.0, k=0.069),
    "fc72":      Coolant("3M FC-72 (dielectric)",    rho=1680.0, mu=6.4e-4, cp=1100.0, k=0.057),
}


def solid(name: str) -> Solid:
    if name not in SOLIDS:
        raise KeyError(f"unknown solid {name!r}; have {list(SOLIDS)}")
    return SOLIDS[name]


def coolant(name: str) -> Coolant:
    if name not in COOLANTS:
        raise KeyError(f"unknown coolant {name!r}; have {list(COOLANTS)}")
    return COOLANTS[name]
