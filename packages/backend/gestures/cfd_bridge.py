"""
cfd_bridge.py  --  FUTURE PHYSICS HOOK (not wired in yet)

This is the seam where the gesture system will eventually meet the NerveGear CFD
surrogate. It does ONE thing and does it as a pure function: translate a
FlowControl (what the hand is asking for) into the parameter dict that the
existing /predict endpoint already accepts.

It is included now -- tested and ready -- so the integration path is visible and
type-checked, but nothing calls it during normal operation. When you turn physics
on, the engine will populate FlowControl.speed / .density (today they are None)
and the runner will call `flow_to_predict_params` and POST to the CFD server.

The target schema (from NerveGear_frontend/backend/main.py :: PredictRequest):
    mask       : NxN grid, 1=solid 0=fluid          (comes from the CAD shape, not gestures)
    Re         : Reynolds number, training range 50-400
    speed      : inlet speed magnitude, range 0.05-0.15
    angle      : radians, range -pi/6 .. +pi/6 (tilt of inlet flow)
    inlet_edge : 'left' | 'right' | 'top' | 'bottom'

TODO(physics): decide whether 'density' should drive Reynolds number (as here) or
a separate viscosity channel once the 3D model exposes one.
"""

from __future__ import annotations

import math

from .commands import FlowControl

# Training ranges, mirrored from the CFD model's generate_data.py / main.py.
RE_MIN, RE_MAX = 50.0, 400.0
SPEED_MIN, SPEED_MAX = 0.05, 0.15
ANGLE_LIMIT = math.pi / 6.0  # +/- 30 degrees, the model's trained tilt range

# Inward-pointing normal of each edge, matching EDGE_NORMAL in the CFD backend.
_EDGE_NORMAL = {
    "left": (1.0, 0.0),
    "right": (-1.0, 0.0),
    "top": (0.0, -1.0),
    "bottom": (0.0, 1.0),
}


def _lerp(lo: float, hi: float, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return lo + (hi - lo) * t


def flow_to_predict_params(
    flow: FlowControl,
    *,
    default_re_t: float = 0.43,   # ~200 on the 50..400 scale, the model's mid-range
    default_speed_t: float = 0.5,
) -> dict:
    """Convert a FlowControl into a /predict-ready parameter dict (minus `mask`,
    which comes from the CAD geometry, not from gestures).

    Robust to the partially-populated FlowControl the engine emits today: missing
    fields fall back to sensible mid-range defaults, so this never raises on a
    POINT gesture that only set `direction` and `inlet_edge`.
    """
    edge = flow.inlet_edge or "left"
    if edge not in _EDGE_NORMAL:
        raise ValueError(f"unknown inlet_edge: {edge!r}")

    speed = _lerp(SPEED_MIN, SPEED_MAX, flow.speed if flow.speed is not None else default_speed_t)
    Re = _lerp(RE_MIN, RE_MAX, flow.density if flow.density is not None else default_re_t)

    # angle = signed tilt of the pointing direction away from the edge's inward
    # normal, clamped to the model's trained +/- 30 degrees.
    angle = 0.0
    if flow.direction is not None:
        nx, ny = _EDGE_NORMAL[edge]
        dx, dy = flow.direction
        # signed angle from normal to direction via atan2 of cross/dot.
        cross = nx * dy - ny * dx
        dot = nx * dx + ny * dy
        raw = math.atan2(cross, dot)
        angle = max(-ANGLE_LIMIT, min(ANGLE_LIMIT, raw))

    return {
        "Re": round(Re, 3),
        "speed": round(speed, 4),
        "angle": round(angle, 4),
        "inlet_edge": edge,
    }
