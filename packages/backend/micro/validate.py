"""
micro/validate.py

Analytic benchmarks for the microchannel correlations and reduced-order model.

Checks against cases with known answers: friction (fRe), Nusselt number, energy
balance (first law), and network pressure drop for a single channel. Run with
python -m micro.validate.
"""

from __future__ import annotations

import numpy as np

from . import correlations as corr
from . import units
from .conjugate import MicroChannelHeatSink, solve
from .geometry import straight_parallel
from .materials import coolant
from .powermap import uniform


def _close(a, b, rtol=0.02):
    return abs(a - b) <= rtol * max(abs(b), 1e-12)


def run() -> dict:
    results = []

    # 1. friction limits
    results.append(("fRe parallel-plates -> 24.0", corr.fRe_fanning(0.0), 24.0,
                    _close(corr.fRe_fanning(0.0), 24.0)))
    results.append(("fRe square duct -> 14.23", corr.fRe_fanning(1.0), 14.23,
                    _close(corr.fRe_fanning(1.0), 14.23, 0.03)))

    # 2. Nusselt limits
    results.append(("Nu_H1 parallel-plates -> 8.235", corr.Nu_H1(0.0), 8.235,
                    _close(corr.Nu_H1(0.0), 8.235)))
    results.append(("Nu_H1 square duct -> 3.61", corr.Nu_H1(1.0), 3.61,
                    _close(corr.Nu_H1(1.0), 3.61, 0.03)))

    # 3. energy balance: coolant rise must equal P/(mdot cp)
    hs = MicroChannelHeatSink(die_L=10e-3, die_W=10e-3, n_ch=50, w_ch=100e-6, H_ch=300e-6)
    pw = uniform(120.0, hs.die_L, hs.die_W)             # 120 W/cm² uniform
    res = solve(hs, pw, Q_total_m3s=units.mlmin_to_m3s(300.0))
    cool = hs.coolant
    mdot = cool.rho * units.mlmin_to_m3s(300.0)
    expected_dT = pw.total_power_W / (mdot * cool.cp)
    got_dT = res.metrics["coolant_dT"]
    results.append(("coolant ΔT == P/(ṁcp)", got_dT, expected_dT, _close(got_dT, expected_dT, 1e-6)))

    # 4. network single-channel ΔP vs analytic Fanning ΔP
    w, H, L = 100e-6, 300e-6, 10e-3
    net = straight_parallel(n_ch=1, w=w, H=H, L=L, header_w=w)
    Q = units.mlmin_to_m3s(20.0)
    sol = net.solve_flow(coolant("water"), Q)
    # analytic for the single through-channel (ignore tiny header at n_ch=1)
    A = w * H; Dh = corr.hydraulic_diameter(w, H); u = Q / A
    Re = corr.reynolds(996.0, u, Dh, 8.5e-4); f = corr.fRe_fanning(corr.aspect_ratio(w, H)) / Re
    dP_analytic = 2 * f * (L / Dh) * 996.0 * u * u
    results.append(("network ΔP == analytic", sol["dP"], dP_analytic, _close(sol["dP"], dP_analytic, 0.05)))

    n_pass = sum(1 for *_, ok in results if ok)
    return {"results": results, "n_pass": n_pass, "n_total": len(results),
            "all_pass": n_pass == len(results)}


def main():
    out = run()
    print("=" * 64)
    print("NerveGear micro — analytic validation")
    print("=" * 64)
    for name, got, exp, ok in out["results"]:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name:36s} got={got:11.4g}  expected={exp:11.4g}")
    print("-" * 64)
    print(f"  {out['n_pass']}/{out['n_total']} benchmarks passed")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
