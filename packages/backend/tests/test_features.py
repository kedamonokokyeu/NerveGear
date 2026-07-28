"""Feature extraction: do the geometric quantities match what each pose means?"""

from __future__ import annotations

import numpy as np

from gestures.features import extract
from tools import synthetic as S


def test_open_palm_features():
    f = extract(S.open_palm())
    assert f.num_extended == 5
    assert f.openness > 0.7          # clearly open
    assert f.pinch > 0.45            # thumb and index far apart
    assert f.hand_scale > 0          # positive ruler


def test_fist_features():
    f = extract(S.fist())
    assert f.num_extended <= 1
    assert f.openness < 0.45         # clearly closed
    # contact is NOT forward of the palm (this is what stops fist reading as pinch)
    assert f.pinch_forward < 1.10


def test_pinch_features():
    f = extract(S.pinch())
    assert f.pinch < 0.45            # thumb & index touching
    assert f.pinch_forward > 1.10    # ...and reaching forward of the palm


def test_point_features():
    f = extract(S.point())
    assert f.extended["index"] is True
    assert f.extended["middle"] is False
    assert f.extended["ring"] is False
    assert f.extended["pinky"] is False


def test_hand_scale_grows_when_hand_enlarges():
    small = extract(S.open_palm())
    big = extract(S.scale_about(S.open_palm(), 1.5))
    assert big.hand_scale > small.hand_scale * 1.4


def test_roll_angle_tracks_in_plane_rotation():
    base = extract(S.open_palm())
    rotated = extract(S.rotate_in_plane(S.open_palm(), 0.3))
    # roll_angle should change by roughly the rotation we applied.
    d = abs(rotated.roll_angle - base.roll_angle)
    assert 0.2 < d < 0.4


def test_centroid_is_inside_hand_bounds():
    f = extract(S.open_palm())
    assert 0.3 < f.centroid[0] < 0.7
    assert 0.4 < f.centroid[1] < 0.9


def test_features_invariant_to_translation():
    """Openness/pinch are normalized, so sliding the hand around must not change
    them (only centroid moves)."""
    a = extract(S.open_palm())
    b = extract(S.translate(S.open_palm(), 0.1, -0.05))
    assert np.isclose(a.openness, b.openness, atol=1e-9)
    assert np.isclose(a.pinch, b.pinch, atol=1e-9)
    assert not np.allclose(a.centroid, b.centroid)
