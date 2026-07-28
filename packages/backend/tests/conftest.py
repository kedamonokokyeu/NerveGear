"""
Shared pytest fixtures. We re-export the synthetic poses as fixtures and add a
small helper to drive the engine over a list of frames.
"""

from __future__ import annotations

import pytest

from gestures.engine import GestureEngine
from gestures.landmarks import Frame
from tools import synthetic as S


@pytest.fixture
def open_palm():
    return S.open_palm()


@pytest.fixture
def fist():
    return S.fist()


@pytest.fixture
def pinch():
    return S.pinch()


@pytest.fixture
def point():
    return S.point()


@pytest.fixture
def engine():
    return GestureEngine()


def drive(engine: GestureEngine, hands_per_frame, t0: float = 0.0, dt: float = 1 / 30):
    """Feed a sequence of per-frame hand lists through the engine; return the list
    of ControlCommands (one per frame)."""
    cmds = []
    t = t0
    for hands in hands_per_frame:
        cmds.append(engine.update(Frame(hands=hands, timestamp=t)))
        t += dt
    return cmds
