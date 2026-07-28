"""Future-physics bridge: FlowControl -> /predict params, within trained ranges."""

from __future__ import annotations

import math

import pytest

from gestures.cfd_bridge import (
    ANGLE_LIMIT,
    RE_MAX,
    RE_MIN,
    SPEED_MAX,
    SPEED_MIN,
    flow_to_predict_params,
)
from gestures.commands import FlowControl


def test_defaults_when_fields_missing():
    """A POINT gesture only sets direction + edge; the bridge must fill the rest
    with mid-range defaults and never raise."""
    p = flow_to_predict_params(FlowControl(direction=(1.0, 0.0), inlet_edge="left"))
    assert SPEED_MIN <= p["speed"] <= SPEED_MAX
    assert RE_MIN <= p["Re"] <= RE_MAX
    assert p["inlet_edge"] == "left"
    assert abs(p["angle"]) <= ANGLE_LIMIT + 1e-3  # tolerance for 4-decimal rounding


def test_speed_and_density_map_to_ranges():
    lo = flow_to_predict_params(FlowControl(inlet_edge="left", speed=0.0, density=0.0))
    hi = flow_to_predict_params(FlowControl(inlet_edge="left", speed=1.0, density=1.0))
    assert math.isclose(lo["speed"], SPEED_MIN, abs_tol=1e-6)
    assert math.isclose(hi["speed"], SPEED_MAX, abs_tol=1e-6)
    assert math.isclose(lo["Re"], RE_MIN, abs_tol=1e-3)
    assert math.isclose(hi["Re"], RE_MAX, abs_tol=1e-3)


def test_angle_clamped_to_trained_range():
    # point straight back along the edge -> would be ~180deg, must clamp to +/-30
    p = flow_to_predict_params(FlowControl(inlet_edge="left", direction=(-1.0, 0.0)))
    assert abs(p["angle"]) <= ANGLE_LIMIT + 1e-3  # tolerance for 4-decimal rounding


def test_direction_aligned_with_normal_is_zero_angle():
    # left edge inward normal is (+1, 0); pointing the same way -> angle 0
    p = flow_to_predict_params(FlowControl(inlet_edge="left", direction=(1.0, 0.0)))
    assert abs(p["angle"]) < 1e-6


def test_unknown_edge_raises():
    with pytest.raises(ValueError):
        flow_to_predict_params(FlowControl(inlet_edge="diagonal"))
