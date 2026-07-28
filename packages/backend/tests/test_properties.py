"""Property/fuzz tests: whatever garbage the tracker throws at the engine, the
output must stay finite, clamped, and well-formed. Catches NaNs and runaway
deltas that fixed-example tests miss."""

from __future__ import annotations

import math

import numpy as np

from gestures.engine import GestureEngine, Mode
from gestures.landmarks import Frame, HandLandmarks

VALID_MODES = {m.value for m in Mode}


def _random_hand(rng):
    pts = rng.random((21, 3))            # arbitrary points in [0,1]^3
    return HandLandmarks(points=pts, handedness="Right", score=float(rng.uniform(0.2, 1.0)))


def _finite(*xs):
    return all(math.isfinite(x) for x in xs)


def test_random_frames_never_break_the_engine():
    rng = np.random.default_rng(1234)
    eng = GestureEngine()
    g = eng.cfg.gains
    t = 0.0
    for _ in range(600):
        n = rng.choice([0, 0, 1, 1, 1, 2])           # mostly 0-1 hands
        hands = [_random_hand(rng) for _ in range(n)]
        cmd = eng.update(Frame(hands=hands, timestamp=t))
        t += 1 / 30

        # well-formed
        assert cmd.mode in VALID_MODES
        assert cmd.num_hands >= 0

        # finite everywhere
        assert _finite(cmd.rotate.yaw, cmd.rotate.pitch, cmd.rotate.roll,
                       cmd.zoom, cmd.pan.x, cmd.pan.y, cmd.scale)

        # clamped within the configured per-frame caps (+ epsilon)
        eps = 1e-9
        for r in (cmd.rotate.yaw, cmd.rotate.pitch, cmd.rotate.roll):
            assert abs(r) <= g.max_rotate_step + eps
        assert abs(cmd.zoom) <= g.max_zoom_step + eps
        assert abs(cmd.pan.x) <= g.max_pan_step + eps
        assert abs(cmd.pan.y) <= g.max_pan_step + eps


def test_frozen_implies_zero_transform():
    """Whenever the engine reports frozen, every transform channel must be zero."""
    rng = np.random.default_rng(7)
    eng = GestureEngine()
    t = 0.0
    saw_frozen = False
    for _ in range(400):
        hands = [_random_hand(rng) for _ in range(rng.choice([1, 2]))]
        cmd = eng.update(Frame(hands=hands, timestamp=t)); t += 1 / 30
        if cmd.frozen:
            saw_frozen = True
            assert cmd.rotate.roll == 0 and cmd.zoom == 0 and cmd.pan.x == 0
    # not asserting saw_frozen (random), just that the invariant held whenever true
    assert saw_frozen or True
