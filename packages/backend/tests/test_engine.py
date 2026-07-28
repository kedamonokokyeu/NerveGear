"""
End-to-end engine behavior. These drive the real GestureEngine with scripted
synthetic frames and assert the control output, which is the closest we can get
to validating the live system without a camera.
"""

from __future__ import annotations

from gestures.engine import GestureEngine, Mode
from gestures.landmarks import Frame
from tools import synthetic as S

from .conftest import drive


def _modes(cmds):
    return [c.mode for c in cmds]


def test_no_hands_is_idle():
    eng = GestureEngine()
    cmds = drive(eng, [[] for _ in range(5)])
    assert all(c.mode == Mode.IDLE.value for c in cmds)
    assert all(c.num_hands == 0 for c in cmds)


def test_sustained_open_palm_enters_rotate():
    eng = GestureEngine()
    cmds = drive(eng, [[S.open_palm()] for _ in range(10)])
    # after debounce, locked into trackball rotate mode
    assert cmds[-1].mode == Mode.ROTATE.value
    assert Mode.ROTATE.value in _modes(cmds)


def test_sustained_fist_freezes():
    eng = GestureEngine()
    cmds = drive(eng, [[S.fist()] for _ in range(8)])
    assert cmds[-1].mode == Mode.HOLD.value
    assert cmds[-1].frozen is True
    # frozen frames emit zero transform
    assert cmds[-1].zoom == 0.0
    assert cmds[-1].rotate.roll == 0.0


def test_still_open_palm_emits_no_delta():
    """A perfectly still hand must produce zero deltas (dead-zone + no motion)."""
    eng = GestureEngine()
    cmds = drive(eng, [[S.open_palm()] for _ in range(12)])
    rotate_frames = [c for c in cmds if c.mode == Mode.ROTATE.value]
    assert rotate_frames
    for c in rotate_frames:
        assert c.rotate.roll == 0.0
        assert c.rotate.yaw == 0.0
        assert c.rotate.pitch == 0.0
        assert c.zoom == 0.0


def test_twisting_palm_produces_roll_rotation():
    eng = GestureEngine()
    # progressively twist the open palm
    seq = [[S.rotate_in_plane(S.open_palm(), 0.05 * i)] for i in range(20)]
    cmds = drive(eng, seq)
    rolls = [c.rotate.roll for c in cmds if c.mode == Mode.ROTATE.value]
    assert sum(rolls) > 0.05           # net positive roll
    # reversing the twist flips the sign
    eng2 = GestureEngine()
    seq2 = [[S.rotate_in_plane(S.open_palm(), -0.05 * i)] for i in range(20)]
    cmds2 = drive(eng2, seq2)
    rolls2 = [c.rotate.roll for c in cmds2 if c.mode == Mode.ROTATE.value]
    assert sum(rolls2) < -0.05


def test_move_mode_yaws_from_hand_movement():
    """In 'move' (trackball) mode, sliding the hand sideways yaws the model -- and
    produces NO zoom, proving the rotate/zoom cross-talk is gone."""
    eng = GestureEngine()
    eng.set_rotation_mode("move")
    seq = [[S.translate(S.open_palm(), 0.015 * i, 0.0)] for i in range(20)]
    cmds = drive(eng, seq)
    rot = [c for c in cmds if c.mode == Mode.ROTATE.value]
    assert rot
    assert sum(c.rotate.yaw for c in rot) > 0.02   # hand right -> yaw
    assert all(c.zoom == 0.0 for c in rot)         # one hand never zooms
    # moving the other way flips yaw
    eng2 = GestureEngine(); eng2.set_rotation_mode("move")
    seq2 = [[S.translate(S.open_palm(), -0.015 * i, 0.0)] for i in range(20)]
    rot2 = [c for c in drive(eng2, seq2) if c.mode == Mode.ROTATE.value]
    assert sum(c.rotate.yaw for c in rot2) < -0.02


def test_tilt_mode_yaws_from_hand_tilt():
    """In 'tilt' (default) mode, angling the hand yaws the model, and a still hand
    that is merely translated does NOT yaw (orientation, not position)."""
    eng = GestureEngine()
    assert eng.rotation_mode == "tilt"
    seq = [[S.tilt(S.open_palm(), yaw=0.03 * i)] for i in range(20)]
    rot = [c for c in drive(eng, seq) if c.mode == Mode.ROTATE.value]
    assert rot
    assert abs(sum(c.rotate.yaw for c in rot)) > 0.02   # tilting the hand yaws
    # pure translation in tilt mode should NOT yaw (no orientation change)
    eng2 = GestureEngine()
    seq2 = [[S.translate(S.open_palm(), 0.02 * i, 0.0)] for i in range(20)]
    rot2 = [c for c in drive(eng2, seq2) if c.mode == Mode.ROTATE.value]
    assert abs(sum(c.rotate.yaw for c in rot2)) < 0.01


def test_set_rotation_mode_validates_and_resets():
    eng = GestureEngine()
    assert eng.set_rotation_mode("move") == "move"
    assert eng.set_rotation_mode("bogus") == "move"   # invalid ignored
    assert eng.set_rotation_mode("tilt") == "tilt"


def test_two_hands_spreading_zooms_in():
    eng = GestureEngine()
    seq = []
    for i in range(20):
        left = S.translate(S.open_palm("Left"), -0.15 - 0.01 * i, 0.0)
        right = S.translate(S.open_palm("Right"), 0.15 + 0.01 * i, 0.0)
        seq.append([left, right])
    cmds = drive(eng, seq)
    two = [c for c in cmds if c.mode == Mode.TWO_HAND.value]
    assert two
    assert sum(c.zoom for c in two) > 0.0   # hands spreading -> zoom in


def test_pinch_drag_pans():
    eng = GestureEngine()
    # pinch moving steadily to the right
    seq = [[S.translate(S.pinch(), 0.02 * i, 0.0)] for i in range(16)]
    cmds = drive(eng, seq)
    pan_frames = [c for c in cmds if c.mode == Mode.PAN.value]
    assert pan_frames
    assert sum(c.pan.x for c in pan_frames) > 0.02
    assert abs(sum(c.pan.y for c in pan_frames)) < 0.02   # no vertical drift


def _two_hands_at_angle(theta: float):
    """Place two open palms diametrically opposite around (0.5, 0.5), so the line
    between them sits at `theta`. Used to simulate the steering-wheel turn."""
    import math

    r = 0.2
    c0 = (0.5 + r * math.cos(theta), 0.5 + r * math.sin(theta))
    c1 = (0.5 - r * math.cos(theta), 0.5 - r * math.sin(theta))
    # open_palm centroid is ~ (0.52, 0.55); translate so each lands on its target.
    h0 = S.translate(S.open_palm("Right"), c0[0] - 0.52, c0[1] - 0.55)
    h1 = S.translate(S.open_palm("Left"), c1[0] - 0.52, c1[1] - 0.55)
    return [h0, h1]


def test_two_hands_steering_wheel_rolls():
    eng = GestureEngine()
    seq = [_two_hands_at_angle(0.06 * i) for i in range(16)]
    cmds = drive(eng, seq)
    two = [c for c in cmds if c.mode == Mode.TWO_HAND.value]
    assert two
    rolls = [c.rotate.roll for c in two]
    assert abs(sum(rolls)) > 0.05            # turning the wheel produces roll
    # reversing the turn flips the sign
    eng2 = GestureEngine()
    seq2 = [_two_hands_at_angle(-0.06 * i) for i in range(16)]
    rolls2 = [c.rotate.roll for c in drive(eng2, seq2) if c.mode == Mode.TWO_HAND.value]
    assert (sum(rolls) > 0) != (sum(rolls2) > 0)


def test_two_hands_with_a_fist_brakes():
    eng = GestureEngine()
    seq = [[S.open_palm("Left"), S.fist("Right")] for _ in range(8)]
    cmds = drive(eng, seq)
    assert cmds[-1].mode == Mode.HOLD.value
    assert cmds[-1].frozen is True


def test_point_sets_flow_but_no_transform():
    eng = GestureEngine()
    cmds = drive(eng, [[S.point()] for _ in range(8)])
    last = cmds[-1]
    assert last.mode == Mode.POINT.value
    # reserved flow direction is populated...
    assert last.flow.direction is not None
    assert last.flow.inlet_edge in {"left", "right", "top", "bottom"}
    # ...but speed/density stay null (physics not enabled) and no transform moves
    assert last.flow.speed is None
    assert last.flow.density is None
    assert last.rotate.roll == 0.0 and last.zoom == 0.0


def test_mode_switch_first_frame_has_zero_delta():
    """The no-lurch guarantee: the first frame of a freshly confirmed mode must
    emit zero deltas, even if the hand was moving in the previous mode."""
    eng = GestureEngine()
    # rotate for a while (moving), then switch to a moving pinch
    seq = [[S.rotate_in_plane(S.open_palm(), 0.05 * i)] for i in range(10)]
    seq += [[S.translate(S.pinch(), 0.03 * i, 0.0)] for i in range(10)]
    cmds = drive(eng, seq)
    modes = _modes(cmds)
    first_pan = modes.index(Mode.PAN.value)
    assert cmds[first_pan].pan.x == 0.0
    assert cmds[first_pan].pan.y == 0.0


def test_delta_is_clamped_on_teleport():
    """A tracking glitch that teleports the hand must be clamped, never flung."""
    eng = GestureEngine()
    g = eng.cfg.gains
    # establish rotate mode, then jump the rotation by a huge amount in one frame
    seq = [[S.open_palm()] for _ in range(5)]
    seq.append([S.rotate_in_plane(S.open_palm(), 2.5)])  # ~143 deg jump
    cmds = drive(eng, seq)
    assert abs(cmds[-1].rotate.roll) <= g.max_rotate_step + 1e-9


def test_command_serializes_to_json():
    import json

    eng = GestureEngine()
    cmd = eng.update(Frame(hands=[S.open_palm()], timestamp=0.0))
    payload = json.dumps(cmd.to_dict())
    back = json.loads(payload)
    assert "mode" in back and "rotate" in back and "flow" in back
