"""ControlCommand serialization shape (the contract the frontend depends on)."""

from __future__ import annotations

import json

from gestures.commands import ControlCommand, FlowControl, PanDelta, RotateDelta


def test_default_command_shape():
    d = ControlCommand().to_dict()
    assert d["mode"] == "idle"
    assert d["zoom"] == 0.0
    assert d["rotate"] == {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    assert d["pan"] == {"x": 0.0, "y": 0.0}
    assert d["frozen"] is False
    assert d["flow"] == {
        "direction": None,
        "speed": None,
        "density": None,
        "inlet_edge": None,
    }


def test_populated_command_roundtrips_json():
    cmd = ControlCommand(
        mode="rotate_zoom",
        rotate=RotateDelta(yaw=0.1, pitch=-0.05, roll=0.2),
        zoom=0.03,
        pan=PanDelta(x=0.01, y=-0.02),
        gestures={"hand0": "open_palm"},
        num_hands=1,
        flow=FlowControl(direction=(1.0, 0.0), inlet_edge="left"),
        timestamp=1.5,
    )
    back = json.loads(json.dumps(cmd.to_dict()))
    assert back["rotate"]["roll"] == 0.2
    assert back["gestures"]["hand0"] == "open_palm"
    assert back["flow"]["direction"] == [1.0, 0.0]   # tuple -> list in JSON
    assert back["timestamp"] == 1.5
