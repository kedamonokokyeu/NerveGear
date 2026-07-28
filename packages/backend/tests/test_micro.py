"""Tests for the microscale pivot layer (micro/). Fast, deterministic, no model or
CFD server required -- mirrors the style of the existing physics tests."""

import numpy as np
import pytest

from micro import correlations as corr
from micro import units
from micro.conjugate import MicroChannelHeatSink, solve
from micro.design import MicroDesign, optimize
from micro.geometry import straight_parallel, murray_tree, rasterize
from micro.materials import coolant
from micro.powermap import uniform
from micro import manufacturing as mfg
from micro import validate


def test_friction_limits():
    assert corr.fRe_fanning(0.0) == pytest.approx(24.0, rel=1e-6)
    assert corr.fRe_fanning(1.0) == pytest.approx(14.23, rel=0.03)


def test_nusselt_limits():
    assert corr.Nu_H1(0.0) == pytest.approx(8.235, rel=1e-6)
    assert corr.Nu_H1(1.0) == pytest.approx(3.61, rel=0.03)


def test_thermal_entrance_longer():
    Pr = coolant("water").Pr
    assert corr.thermal_entrance_length(500, Pr, 1e-4) > corr.hydrodynamic_entrance_length(500, 1e-4)


def test_unit_roundtrips():
    assert units.to_um(units.um(123.0)) == pytest.approx(123.0)
    assert units.m3s_to_mlmin(units.mlmin_to_m3s(250.0)) == pytest.approx(250.0)
    assert units.wm2_to_wcm2(units.wcm2_to_wm2(500.0)) == pytest.approx(500.0)


def test_energy_balance_exact():
    hs = MicroChannelHeatSink(n_ch=50, w_ch=100e-6, H_ch=300e-6)
    pw = uniform(120.0, hs.die_L, hs.die_W); Q = units.mlmin_to_m3s(300.0)
    res = solve(hs, pw, Q_total_m3s=Q)
    mdot = hs.coolant.rho * Q
    assert res.metrics["coolant_dT"] == pytest.approx(pw.total_power_W / (mdot * hs.coolant.cp), rel=1e-6)


def test_downstream_runs_hotter():
    hs = MicroChannelHeatSink()
    res = solve(hs, uniform(100.0, hs.die_L, hs.die_W), Q_total_m3s=units.mlmin_to_m3s(200.0))
    assert np.all(np.diff(res.Tf_x) >= -1e-9) and res.Tf_x[-1] > res.Tf_x[0]


def test_more_flow_cools_and_costs_pressure():
    hs = MicroChannelHeatSink(); pw = uniform(150.0, hs.die_L, hs.die_W)
    lo = solve(hs, pw, Q_total_m3s=units.mlmin_to_m3s(100.0)).metrics
    hi = solve(hs, pw, Q_total_m3s=units.mlmin_to_m3s(400.0)).metrics
    assert hi["max_junction_temp"] < lo["max_junction_temp"]
    assert hi["pressure_drop_kPa"] > lo["pressure_drop_kPa"]


def test_laminar_regime():
    res = solve(MicroChannelHeatSink(), uniform(100.0), Q_total_m3s=units.mlmin_to_m3s(200.0))
    assert res.metrics["Re"] < 2300.0 and res.metrics["laminar"]


def test_dielectric_differs_from_water():
    base = dict(n_ch=50, w_ch=100e-6, H_ch=300e-6)
    w = solve(MicroChannelHeatSink(coolant=coolant("water"), **base), uniform(100.0),
              Q_total_m3s=units.mlmin_to_m3s(200.0)).metrics
    d = solve(MicroChannelHeatSink(coolant=coolant("fc72"), **base), uniform(100.0),
              Q_total_m3s=units.mlmin_to_m3s(200.0)).metrics
    assert d["pressure_drop_kPa"] != pytest.approx(w["pressure_drop_kPa"], rel=1e-3)


def test_network_single_channel_matches_analytic():
    w, H, L = 100e-6, 300e-6, 10e-3
    net = straight_parallel(n_ch=1, w=w, H=H, L=L, header_w=w)
    Q = units.mlmin_to_m3s(20.0); dP = net.solve_flow(coolant("water"), Q)["dP"]
    A = w * H; Dh = corr.hydraulic_diameter(w, H); u = Q / A
    Re = corr.reynolds(996.0, u, Dh, 8.5e-4); f = corr.fRe_fanning(corr.aspect_ratio(w, H)) / Re
    assert dP == pytest.approx(2 * f * (L / Dh) * 996.0 * u * u, rel=0.05)


def test_murray_widths_taper():
    net = murray_tree(levels=3, w_trunk=400e-6, exponent=3.0)
    assert min(s.w for s in net.segments) < max(s.w for s in net.segments)


def test_murray_network_solves():
    net = murray_tree(levels=4)
    sol = net.solve_flow(coolant("water"), units.mlmin_to_m3s(300.0))
    assert sol["dP"] > 0 and sol["n_through"] > 1


def test_rasterize_inverts_mask():
    net = straight_parallel(n_ch=8, w=120e-6, H=300e-6, L=4e-3)
    sol = net.solve_flow(coolant("water"), units.mlmin_to_m3s(100.0))
    r = rasterize(net, grid_n=128, seg_flow=sol["seg_flow"], coolant=coolant("water"))
    assert (r["mask"] == 1).any() and (r["mask"] == 0).any()
    assert np.abs(r["vx"]).max() + np.abs(r["vy"]).max() > 0


def test_aspect_ratio_violation_flagged():
    hs = MicroChannelHeatSink(w_ch=20e-6, H_ch=900e-6, die_W=10e-3, n_ch=100)
    assert not {v.rule: v for v in mfg.check_heatsink(hs)}["aspect_ratio"].ok


def test_clogging_risk_monotonic():
    assert mfg.clogging_risk(MicroChannelHeatSink(w_ch=30e-6)) > mfg.clogging_risk(MicroChannelHeatSink(w_ch=200e-6))


def test_floating_silicon_detector():
    # detector flags solid enclosed by channel (no path to die border). NB: floored
    # microchannel walls are supported in z, so this 2D check is for through-cut plates.
    iso = np.ones((40, 40), np.uint8)
    iso[10:30, 10] = 0; iso[10:30, 29] = 0; iso[10, 10:30] = 0; iso[29, 10:30] = 0
    assert not {v.rule: v for v in mfg.check_mask(iso, 1e-4)}["no_floating_silicon"].ok
    ok = np.ones((40, 40), np.uint8); ok[:, 18:22] = 0
    assert {v.rule: v for v in mfg.check_mask(ok, 1e-4)}["no_floating_silicon"].ok


def test_optimizer_respects_pressure_constraint():
    hs = MicroChannelHeatSink(die_L=12e-3, die_W=12e-3, n_ch=60, w_ch=90e-6, H_ch=400e-6)
    pw = uniform(150.0, hs.die_L, hs.die_W)
    out = optimize(MicroDesign(base=hs, power=pw, dP_limit_kPa=40.0), n_iter=200, seed=3)
    if out.feasible:
        assert out.best_metrics["pressure_drop_kPa"] <= 40.0 + 1e-6
        assert out.best_metrics["aspect_ratio_AR"] <= mfg.EtchRules().max_aspect_ratio + 1e-6


def test_validation_suite_all_pass():
    assert validate.run()["all_pass"]
