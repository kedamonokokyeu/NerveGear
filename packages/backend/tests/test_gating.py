"""Presence gating: spurious / tiny / low-confidence detections are ignored."""

from __future__ import annotations

from gestures.engine import GestureEngine, Mode
from gestures.landmarks import HandLandmarks
from tools import synthetic as S

from .conftest import drive


def test_low_confidence_hand_is_ignored():
    eng = GestureEngine()
    # an open palm that would normally rotate, but with score below threshold
    low = HandLandmarks(points=S.open_palm().points, handedness="Right", score=0.2)
    cmds = drive(eng, [[low] for _ in range(10)])
    assert all(c.num_hands == 0 for c in cmds)
    assert cmds[-1].mode == Mode.IDLE.value


def test_tiny_hand_is_ignored():
    eng = GestureEngine()
    # shrink the hand far below the min_hand_scale gate (far-away false positive)
    tiny = S.scale_about(S.open_palm(), 0.1)
    cmds = drive(eng, [[tiny] for _ in range(10)])
    assert all(c.num_hands == 0 for c in cmds)


def test_normal_hand_still_passes():
    eng = GestureEngine()
    cmds = drive(eng, [[S.open_palm()] for _ in range(10)])
    assert cmds[-1].num_hands == 1
    assert cmds[-1].mode == Mode.ROTATE.value


def test_gating_thresholds_are_configurable():
    eng = GestureEngine()
    assert eng.cfg.gestures.min_hand_score > 0
    assert eng.cfg.gestures.min_hand_scale > 0
    # raising the score gate above a normal hand's score rejects it
    eng.cfg.gestures.min_hand_score = 1.01
    cmds = drive(eng, [[S.open_palm()] for _ in range(6)])
    assert all(c.num_hands == 0 for c in cmds)
