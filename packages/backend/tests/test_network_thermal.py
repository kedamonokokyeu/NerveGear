"""Validation of the freeform-network conjugate model against the benchmarked
straight-array conjugate model and first principles (build plan, Phase 1)."""

import numpy as np
import pytest

from micro import geometry, network_thermal, units
from micro.conjugate import MicroChannelHeatSink, solve as conj_solve
from micro.powermap import Hotspot, from_hotspots, uniform


@pytest.fixture(scope="module")
def straight_case():
    hs = MicroChannelHeatSink(die_L=10e-3, die_W=10e-3, n_ch=40, w_ch=100e-6,
                              H_ch=300e-6, t_base=100e-6)
    pw = uniform(150.0, 10e-3, 10e-3)
    Q = units.mlmin_to_m3s(300.0)
    return hs, pw, Q


def test_agrees_with_conjugate_reference(straight_case):
    """Uniform-flow straight array: network model within 10% of the validated
    conjugate model on the junction-temperature RISE."""
    hs, pw, Q = straight_case
    ref = conj_solve(hs, pw, Q_total_m3s=Q)
    net = geometry.straight_parallel(n_ch=40, w=100e-6, H=300e-6, L=10e-3,
                                     pitch=250e-6, header_w=3e-3)
    r = network_thermal.solve_network(net, pw, Q_total_m3s=Q,
                                      t_base=100e-6, die_thk=500e-6)
    rise_ref = ref.metrics["max_junction_temp"] - 25.0
    rise_net = r.metrics["max_junction_temp"] - 25.0
    assert abs(rise_net - rise_ref) / rise_ref < 0.10
    assert abs(r.metrics["pressure_drop_kPa"] - ref.metrics["pressure_drop_kPa"]) \
        / ref.metrics["pressure_drop_kPa"] < 0.15


def test_energy_balance_exact(straight_case):
    hs, pw, Q = straight_case
    net = geometry.straight_parallel(n_ch=40, w=100e-6, H=300e-6, L=10e-3, pitch=250e-6)
    r = network_thermal.solve_network(net, pw, Q_total_m3s=Q, t_base=100e-6, die_thk=500e-6)
    assert r.metrics["energy_balance_error"] < 1e-6
    mdot = 996.0 * Q
    expected = r.metrics["total_power_W"] / (mdot * 4180.0)
    assert r.metrics["coolant_dT"] == pytest.approx(expected, rel=1e-3)


def test_maldistribution_is_captured():
    """Narrow headers must starve far channels: higher Tj and flow CV than
    wide headers. This is physics the plain conjugate model cannot see."""
    pw = uniform(150.0, 10e-3, 10e-3)
    Q = units.mlmin_to_m3s(300.0)
    out = {}
    for hw in (0.6e-3, 3e-3):
        net = geometry.straight_parallel(n_ch=40, w=100e-6, H=300e-6, L=10e-3,
                                         pitch=250e-6, header_w=hw)
        out[hw] = network_thermal.solve_network(net, pw, Q_total_m3s=Q,
                                                t_base=100e-6, die_thk=500e-6).metrics
    assert out[0.6e-3]["maldistribution_cv"] > 3 * out[3e-3]["maldistribution_cv"]
    assert out[0.6e-3]["max_junction_temp"] > out[3e-3]["max_junction_temp"]


def test_from_segments_prunes_dead_and_connects():
    segs = [
        {"x0": 0, "y0": 5e-3, "x1": 5e-3, "y1": 5e-3, "w": 400e-6, "depth": 300e-6},
        {"x0": 5e-3, "y0": 5e-3, "x1": 10e-3, "y1": 5e-3, "w": 400e-6, "depth": 300e-6},
        {"x0": 2e-3, "y0": 9e-3, "x1": 4e-3, "y1": 9e-3, "w": 200e-6, "depth": 300e-6},
    ]
    net, info = geometry.network_from_segments(segs, 10e-3, 10e-3)
    assert info["n_dead"] == 1
    r = network_thermal.solve_network(net, uniform(50.0, 10e-3, 10e-3), flow_mlmin=50.0)
    assert r.metrics["pressure_drop_kPa"] > 0
    assert r.metrics["energy_balance_error"] < 1e-6


def test_disconnected_raises():
    segs = [{"x0": 0, "y0": 5e-3, "x1": 4e-3, "y1": 5e-3, "w": 300e-6, "depth": 300e-6}]
    with pytest.raises(ValueError, match="edge"):
        geometry.network_from_segments(segs, 10e-3, 10e-3)


def test_hotspot_heats_downstream():
    """A hotspot near the outlet must raise Tj vs the same power near the
    inlet (coolant preheating) — sanity on flow-direction physics."""
    net = geometry.straight_parallel(n_ch=30, w=150e-6, H=300e-6, L=10e-3, pitch=333e-6)
    Q = units.mlmin_to_m3s(200.0)
    tj = {}
    for cx in (0.2, 0.8):
        pw = from_hotspots([Hotspot(cx, 0.5, 0.08, 400.0)], die_L=10e-3, die_W=10e-3)
        tj[cx] = network_thermal.solve_network(net, pw, Q_total_m3s=Q,
                                               t_base=100e-6, die_thk=500e-6
                                               ).metrics["max_junction_temp"]
    assert tj[0.8] > tj[0.2]
