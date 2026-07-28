"""
smoothing.py

  * OneEuroFilter -- Applies heavy smoothing for when hand is nearly still, 
    but applies smoothing when hand moves fast to prevent lag
    the de-facto standard low-pass filter for interactive hand/    

  * Debouncer -- requires a gesture to persist for N frames before it is accepted,
    so the interaction mode doesn't falsely activate

"""

from __future__ import annotations
import math
from collections import deque
from typing import Generic, TypeVar
import numpy as np


def _alpha(dt: float, cutoff: float) -> float:
    """One-Euro smoothing factor for a given timestep and cutoff frequency."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Adaptive low-pass filter. Call it once per frame with the timestamp and the
    new value; it returns the smoothed value. Handles scalars or numpy vectors
    transparently (the math is elementwise)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None       # last raw value
        self._dx_prev = None      # last smoothed derivative
        self._t_prev: float | None = None

    def reset(self) -> None:
        """Forget history. Call when the tracked hand disappears so the next
        appearance does not get filtered against stale values."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, t: float, x):
        x = np.asarray(x, dtype=np.float64) if not np.isscalar(x) else float(x)

        # if first sample, nothing to smooth
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            self._dx_prev = x * 0.0  # 0 (scalar) or zero-vector, same shape as x
            return x

        dt = t - self._t_prev
        if dt <= 0:
            return self._x_prev

        # estimate and smooth the derivative.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # adapt the cutoff to the speed: faster motion -> higher cutoff -> less lag.
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        # cutoff may be a vector (per-component); _alpha handles arrays fine.
        a = _alpha(dt, cutoff)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class AngleFilter:
    """One-Euro filter specialized for angles, which wrap around at +/-pi. We
    smooth the unit vector (cos, sin) and convert back, so values near the +/-pi
    seam don't average to nonsense."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self._vec = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self._vec.reset()

    def __call__(self, t: float, angle: float) -> float:
        v = np.array([math.cos(angle), math.sin(angle)])
        s = self._vec(t, v)
        return float(math.atan2(s[1], s[0]))


def angle_delta(a: float, b: float) -> float:
    """Shortest signed difference a-b, wrapped into (-pi, pi]. Use this for every
    rotation delta so a twist across the +/-pi seam never produces a giant jump."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


T = TypeVar("T")


class Debouncer(Generic[T]):
    """
    Accept a new discrete value only after it has been observed for
    `frames_to_confirm` consecutive frames.
    Modes shouldn't flicker every frame, so if a mode appeares for N frames, 
    then the mode is switched.
    Keeps interaction mode stable.
    """

    def __init__(self, frames_to_confirm: int, initial: T):
        self.frames_to_confirm = max(1, int(frames_to_confirm))
        self._confirmed: T = initial
        self._candidate: T = initial
        self._count = 0
        self._recent: deque = deque(maxlen=self.frames_to_confirm)

    @property
    def value(self) -> T:
        return self._confirmed

    def update(self, observed: T) -> T:
        if observed == self._candidate:
            self._count += 1
        else:
            self._candidate = observed
            self._count = 1
        if self._count >= self.frames_to_confirm:
            self._confirmed = self._candidate
        return self._confirmed
