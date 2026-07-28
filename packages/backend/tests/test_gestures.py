"""Gesture classification: each synthetic pose must map to its intended label."""

from __future__ import annotations

from gestures.config import DEFAULT
from gestures.features import extract
from gestures.gestures import Gesture, classify
from tools import synthetic as S


def _classify(hand):
    return classify(extract(hand), DEFAULT.gestures)


def test_open_palm_classifies():
    assert _classify(S.open_palm()) == Gesture.OPEN_PALM


def test_fist_classifies():
    assert _classify(S.fist()) == Gesture.FIST


def test_pinch_classifies():
    assert _classify(S.pinch()) == Gesture.PINCH


def test_point_classifies():
    assert _classify(S.point()) == Gesture.POINT


def test_fist_is_not_mistaken_for_pinch():
    """A fist also has thumb & index close together; the 'forward' test must keep
    it out of the PINCH bucket."""
    assert _classify(S.fist()) != Gesture.PINCH


def test_classification_stable_under_small_translation():
    for pose in (S.open_palm, S.fist, S.pinch, S.point):
        base = _classify(pose())
        moved = _classify(S.translate(pose(), 0.07, 0.03))
        assert base == moved


def test_classification_stable_under_scale():
    """Distance to camera (apparent size) must not change the label."""
    for pose in (S.open_palm, S.fist, S.pinch):
        base = _classify(pose())
        bigger = _classify(S.scale_about(pose(), 1.3))
        assert base == bigger
