"""One-Euro filter, angle helpers, and the gesture debouncer."""

from __future__ import annotations

import math

import numpy as np

from gestures.smoothing import (
    AngleFilter,
    Debouncer,
    OneEuroFilter,
    angle_delta,
)


def test_oneeuro_passes_first_sample_through():
    f = OneEuroFilter(1.0, 0.0)
    assert f(0.0, 5.0) == 5.0


def test_oneeuro_smooths_toward_signal():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f(0.0, 0.0)
    # A step to 10: the filtered output should lag (be < 10) but move upward.
    y = f(1 / 30, 10.0)
    assert 0.0 < y < 10.0


def test_oneeuro_reduces_jitter_variance():
    """Feeding a noisy constant signal, the filtered variance must be lower than
    the raw variance."""
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    raw, filt = [], []
    t = 0.0
    for _ in range(200):
        x = 1.0 + rng.normal(0, 0.1)
        raw.append(x)
        filt.append(f(t, x))
        t += 1 / 30
    # ignore warm-up
    assert np.var(filt[20:]) < np.var(raw[20:]) * 0.5


def test_oneeuro_handles_vectors():
    f = OneEuroFilter(1.0, 0.0)
    out0 = f(0.0, np.array([1.0, 2.0]))
    assert np.allclose(out0, [1.0, 2.0])
    out1 = f(1 / 30, np.array([2.0, 4.0]))
    assert out1.shape == (2,)
    assert (out1 > [1.0, 2.0]).all() and (out1 < [2.0, 4.0]).all()


def test_oneeuro_ignores_nonincreasing_time():
    f = OneEuroFilter(1.0, 0.0)
    f(1.0, 3.0)
    # same/earlier timestamp returns the last value, no crash
    assert f(1.0, 99.0) == 3.0


def test_angle_delta_wraps_across_pi():
    # from +179deg to -179deg is +2deg, not -358deg
    a = math.radians(-179)
    b = math.radians(179)
    d = angle_delta(a, b)
    assert math.isclose(d, math.radians(2), abs_tol=1e-6)


def test_angle_filter_handles_seam():
    f = AngleFilter(1.0, 0.0)
    f(0.0, math.pi - 0.01)
    out = f(1 / 30, -math.pi + 0.01)   # crossing the seam
    # result should be near +/-pi, not near 0
    assert abs(abs(out) - math.pi) < 0.2


def test_debouncer_requires_consecutive_frames():
    d = Debouncer(frames_to_confirm=3, initial="idle")
    assert d.update("rotate") == "idle"   # 1
    assert d.update("rotate") == "idle"   # 2
    assert d.update("rotate") == "rotate"  # 3 -> confirmed


def test_debouncer_resets_on_flicker():
    d = Debouncer(frames_to_confirm=3, initial="idle")
    d.update("rotate")
    d.update("rotate")
    d.update("zoom")            # flicker resets the count
    assert d.value == "idle"
    d.update("zoom")
    assert d.update("zoom") == "zoom"
