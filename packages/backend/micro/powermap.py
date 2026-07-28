"""
micro/powermap.py

The die power map as a first-class field input.

At die level the design input is the heat flux distribution q''(x,y) in W/cm²,
not a list of chip boxes. A PowerMap is a 2D array of heat flux over the die
footprint that can be resampled onto any solver grid and integrated to a total
wattage. Build one from synthetic hotspots now; load a real floorplan export later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import wcm2_to_wm2, wm2_to_wcm2


@dataclass
class Hotspot:
    cx: float            # centre, fraction of die [0,1]
    cy: float
    sigma: float         # gaussian radius, fraction of die
    peak_wcm2: float     # peak heat flux [W/cm²]


@dataclass
class PowerMap:
    flux: np.ndarray     # (Ny, Nx) heat flux [W/m²]
    die_L: float         # x extent [m]
    die_W: float         # y extent [m]

    @property
    def total_power_W(self) -> float:
        cell = (self.die_L / self.flux.shape[1]) * (self.die_W / self.flux.shape[0])
        return float(self.flux.sum() * cell)

    @property
    def peak_wcm2(self) -> float:
        return wm2_to_wcm2(float(self.flux.max()))

    @property
    def mean_wcm2(self) -> float:
        return wm2_to_wcm2(float(self.flux.mean()))

    def flux_wm2(self, grid_n: int) -> np.ndarray:
        """Resample the map onto a grid_n × grid_n grid (nearest — fluxes are
        piecewise smooth and this keeps total power stable enough for the reduced
        model; swap for area-weighted when coupling to the high-fidelity solver)."""
        Ny, Nx = self.flux.shape
        if (Ny, Nx) == (grid_n, grid_n):
            return self.flux
        ri = (np.linspace(0, Ny - 1, grid_n)).round().astype(int)
        ci = (np.linspace(0, Nx - 1, grid_n)).round().astype(int)
        return self.flux[np.ix_(ri, ci)]


def from_hotspots(hotspots, die_L=12e-3, die_W=12e-3, grid=128,
                  background_wcm2=20.0) -> PowerMap:
    """Synthesise a realistic non-uniform die: a uniform background load (the bulk
    logic) plus Gaussian hotspots (cores, AVX units, HBM stacks). Background of
    ~20 W/cm² with hotspots to 300–1000 W/cm² mimics a real GPU/CPU floorplan."""
    xs = (np.arange(grid) + 0.5) / grid
    ys = (np.arange(grid) + 0.5) / grid
    gx, gy = np.meshgrid(xs, ys)
    q_wcm2 = np.full((grid, grid), background_wcm2, dtype=np.float64)
    for h in hotspots:
        q_wcm2 += h.peak_wcm2 * np.exp(-(((gx - h.cx) ** 2 + (gy - h.cy) ** 2) / (2 * h.sigma ** 2)))
    return PowerMap(flux=wcm2_to_wm2(q_wcm2), die_L=die_L, die_W=die_W)


def uniform(peak_wcm2=100.0, die_L=10e-3, die_W=10e-3, grid=128) -> PowerMap:
    """Uniform heat flux — the benchmark case for validation (pivot 8)."""
    return PowerMap(flux=np.full((grid, grid), wcm2_to_wm2(peak_wcm2)), die_L=die_L, die_W=die_W)
