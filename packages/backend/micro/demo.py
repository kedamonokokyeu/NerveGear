"""
micro/demo.py

End-to-end walkthrough of the microscale layer on a synthetic GPU-like die.

No ML or CFD server needed. Builds a non-uniform power map, evaluates a baseline
straight-channel heat sink, shows a Murray-tree network's flow split, runs the
constrained optimizer, and checks the result against manufacturing rules.

    python -m micro.demo
"""

from __future__ import annotations

from . import units
from . import manufacturing as mfg
from .conjugate import MicroChannelHeatSink, solve
from .design import MicroDesign, optimize
from .geometry import murray_tree
from .materials import coolant, solid
from .powermap import Hotspot, from_hotspots


def banner(t):
    print("\n" + "=" * 66 + f"\n  {t}\n" + "=" * 66)


def main():
    banner("1. Power map — non-uniform die (background + hotspots)")
    hotspots = [
        Hotspot(0.30, 0.35, 0.06, 600.0),   # a hot core
        Hotspot(0.70, 0.65, 0.05, 800.0),   # a hotter AVX/tensor block
        Hotspot(0.50, 0.50, 0.10, 200.0),   # warm bulk
    ]
    pw = from_hotspots(hotspots, die_L=12e-3, die_W=12e-3, grid=128, background_wcm2=30.0)
    print(f"  total power = {pw.total_power_W:6.1f} W   peak = {pw.peak_wcm2:6.0f} W/cm²"
          f"   mean = {pw.mean_wcm2:5.0f} W/cm²")

    banner("2. Baseline straight microchannel heat sink (water)")
    hs = MicroChannelHeatSink(die_L=12e-3, die_W=12e-3, t_base=80e-6,
                              n_ch=60, w_ch=90e-6, H_ch=400e-6,
                              solid=solid("silicon"), coolant=coolant("water"),
                              junction_limit_c=105.0)
    res = solve(hs, pw, Q_total_m3s=units.mlmin_to_m3s(300.0))
    m = res.metrics
    for k in ["max_junction_temp", "junction_gradient", "coolant_dT",
              "pressure_drop_kPa", "pumping_power_W", "COP",
              "thermal_resistance_Kcm2W", "Re", "Nu", "fin_efficiency",
              "thermal_developing_fraction", "thermal_margin"]:
        print(f"  {k:28s} = {m[k]:10.3f}")
    print(f"  laminar = {m['laminar']}   Dh = {m['Dh_um']:.1f} µm")

    banner("3. Branching (Murray) network — flow split + ΔP")
    net = murray_tree(levels=4, w_trunk=400e-6, H=350e-6, die_L=12e-3, die_W=12e-3, exponent=3.0)
    sol = net.solve_flow(coolant("water"), units.mlmin_to_m3s(300.0))
    print(f"  segments = {len(net.segments)}   through-channels = {sol['n_through']}")
    print(f"  ΔP = {units.to_kPa(sol['dP']):.2f} kPa   pumping = {sol['pumping_power']*1e3:.2f} mW")
    print(f"  flow maldistribution CV = {sol['maldistribution_cv']*100:.1f} %")

    banner("4. Constrained multi-objective optimization")
    design = MicroDesign(base=hs, power=pw, dP_limit_kPa=40.0, junction_limit_c=105.0)
    out = optimize(design, n_iter=400, seed=1)
    bp, bm = out.best_params, out.best_metrics
    print(f"  feasible = {out.feasible}")
    print(f"  best:  w_ch={units.to_um(bp['w_ch']):.0f}µm  H_ch={units.to_um(bp['H_ch']):.0f}µm"
          f"  n_ch={bp['n_ch']:.0f}  flow={bp['flow_mlmin']:.0f} mL/min")
    print(f"  ->  Tj_max={bm['max_junction_temp']:.1f}°C  grad={bm['junction_gradient']:.1f}°C"
          f"  ΔP={bm['pressure_drop_kPa']:.1f} kPa  COP={bm['COP']:.0f}")
    print(f"  improvement vs baseline: ΔTj = {m['max_junction_temp'] - bm['max_junction_temp']:+.1f} °C")

    banner("5. Manufacturability / reliability check on the winner")
    from dataclasses import replace
    win = replace(hs, w_ch=bp["w_ch"], H_ch=bp["H_ch"], n_ch=int(round(bp["n_ch"])))
    for v in mfg.check_heatsink(win):
        flag = "ok " if v.ok else "XX "
        print(f"  [{flag}] {v.rule:18s} value={v.value:8.1f}  limit={v.limit:8.1f}  ({v.note})")
    print()


if __name__ == "__main__":
    main()
