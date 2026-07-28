"""Hele-Shaw freeform-etch solver vs analytic rectangular-duct results."""

import numpy as np
import pytest

from micro import correlations as corr, heleshaw, units
from micro.powermap import uniform


def _straight_channel(ny=96, nx=128, w=500e-6, depth=300e-6, die=10e-3):
    d = np.zeros((ny, nx))
    dyc = die / ny
    rows = max(2, int(round(w / dyc)))
    r0 = ny // 2 - rows // 2
    d[r0:r0 + rows, :] = depth
    return d, rows * dyc


@pytest.mark.parametrize("w,depth,tol", [(500e-6, 300e-6, 0.12),
                                         (800e-6, 200e-6, 0.10)])
def test_straight_channel_dp_matches_analytic(w, depth, tol):
    die = 10e-3
    d, w_eff = _straight_channel(w=w, depth=depth, die=die)
    Q = units.mlmin_to_m3s(20.0)
    r = heleshaw.solve_etch(d, die, die, uniform(50.0, die, die), Q_total_m3s=Q,
                            t_base=100e-6, die_thk=500e-6)
    A = w_eff * depth
    Dh = corr.hydraulic_diameter(w_eff, depth)
    u = Q / A
    Re = corr.reynolds(996.0, u, Dh, 8.5e-4)
    f = corr.fRe_fanning(corr.aspect_ratio(w_eff, depth)) / Re
    dP_analytic = 2 * f * (die / Dh) * 996.0 * u * u
    assert r.metrics["pressure_drop_kPa"] == pytest.approx(dP_analytic / 1e3, rel=tol)


def test_energy_balance_closes():
    die = 10e-3
    d = np.zeros((96, 128))
    for k in range(14):
        rr = int((k + 0.5) / 14 * 96)
        d[rr:rr + 3, :] = 350e-6
    r = heleshaw.solve_etch(d, die, die, uniform(100.0, die, die),
                            Q_total_m3s=units.mlmin_to_m3s(250.0),
                            t_base=120e-6, die_thk=600e-6)
    assert r.metrics["energy_balance_error"] < 0.01


def test_disconnected_cavity_raises():
    d = np.zeros((64, 64))
    d[30:34, 10:50] = 300e-6           # floats in the middle, touches no edge
    with pytest.raises(ValueError):
        heleshaw.solve_etch(d, 10e-3, 10e-3, uniform(50.0, 10e-3, 10e-3),
                            Q_total_m3s=1e-6)


def test_more_flow_cools():
    die = 10e-3
    d, _ = _straight_channel(w=800e-6, depth=400e-6, die=die)
    tj = []
    for q in (50.0, 400.0):
        r = heleshaw.solve_etch(d, die, die, uniform(30.0, die, die),
                                Q_total_m3s=units.mlmin_to_m3s(q),
                                t_base=100e-6, die_thk=500e-6)
        tj.append(r.metrics["max_junction_temp"])
    assert tj[1] < tj[0]
