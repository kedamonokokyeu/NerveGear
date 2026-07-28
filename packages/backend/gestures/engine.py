"""
engine.py

The engine consumes a stream of frames (sets of han lanmarks) and then creates
ControlCommands that contain the transformation deltas (whether or not to undergo
an interactive state on the previously defined thresholds). 
The brain. It consumes a stream of Frames (sets of hand landmarks) and produces a

  no hands --> IDLE: emit zeros.
  one FIST --> HOLD: BRAKE. freeze the view, emit zeros.
                     ("close your hand to stop", as in the demo)
  one OPEN_PALM --> ROTATE_ZOOM : the main driver.
                        * palm twist (in-plane roll)  -> rotate (roll)
                        * palm tilt (normal x/y)       -> rotate (yaw/pitch)
                        * hand moves toward/away camera -> zoom
  one PINCH  --> PAN : grab and drag the model.
  one POINT --> POINT : reserved. We compute a flow direction for
                                the future CFD hook but emit no transform.
  two OPEN hands --> TWO_HAND : pinch-to-zoom style.
                                * change in distance between hands -> scale
                                * motion of their midpoint          -> pan
  two hands, any FIST --> HOLD : both-hands-closed also brakes.

The delta and baseline should reset after each lock.
When the mode changes, smoothing filters reset. So if I freeze with two closed hands,
then want to perform some rotation motion, the smoothing reapplied based on how quick
the motion is. Every channel reported as change from previous frame.
"""

from __future__ import annotations
from enum import Enum
import numpy as np
from . import features as feat_mod
from .commands import ControlCommand, FlowControl, PanDelta, RotateDelta
from .config import DEFAULT, Config
from .gestures import Gesture, classify
from .landmarks import Frame, HandLandmarks, LM
from .smoothing import AngleFilter, Debouncer, OneEuroFilter, angle_delta


class Mode(str, Enum):
    """All possible interaction states."""
    IDLE = "idle"
    HOLD = "hold"
    ROTATE = "rotate"      # one open hand: rotation (no zoom)
    PAN = "pan"
    POINT = "point"
    TWO_HAND = "two_hand"  # two hands: zoom + roll + pan


def _shape(value: float, deadzone: float, max_step: float) -> float:
    """
    Stabilizer
    Applied deadzone---if less than deadzone for said value, value is ignored.
    If value is huge, stop it at a safe range.
    """
    if abs(value) < deadzone:
        return 0.0
    return float(np.clip(value, -max_step, max_step))


class GestureEngine:
    """Main processor for frames and command outputs."""
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or DEFAULT
        s = self.cfg.smoothing

        # built-in smoothing algorithms
        # AngleFilter is speciailzed version of OneEuroFilter for angles that wrap around at pi
            # averaging angles near wrap point is nonsense, hence why AngleFilter needed
        self._f_roll = AngleFilter(s.min_cutoff, s.beta, s.d_cutoff)       # palm in-plane angle
        self._f_normal = OneEuroFilter(s.min_cutoff, s.beta, s.d_cutoff)   # palm normal (3-vec)
        self._f_scale = OneEuroFilter(s.min_cutoff, s.beta, s.d_cutoff)    # hand apparent size
        self._f_centroid = OneEuroFilter(s.min_cutoff, s.beta, s.d_cutoff) # palm center (2-vec)
        self._f_handdist = OneEuroFilter(s.min_cutoff, s.beta, s.d_cutoff) # two-hand distance
        self._f_midpoint = OneEuroFilter(s.min_cutoff, s.beta, s.d_cutoff) # two-hand midpoint (2-vec)
        self._f_handangle = AngleFilter(s.min_cutoff, s.beta, s.d_cutoff)  # two-hand "wheel" angle
        self._all_filters = [
            self._f_roll, self._f_normal, self._f_scale,
            self._f_centroid, self._f_handdist, self._f_midpoint, self._f_handangle,
        ]

        # previous smoothed values for delta computation
        self._prev_roll: float | None = None
        self._prev_normal: np.ndarray | None = None
        self._prev_scale: float | None = None
        self._prev_centroid: np.ndarray | None = None
        self._prev_handdist: float | None = None
        self._prev_midpoint: np.ndarray | None = None
        self._prev_handangle: float | None = None

        # stability check
        self._mode_debounce: Debouncer[Mode] = Debouncer(self.cfg.debounce.frames_to_confirm, Mode.IDLE)

        self._active_mode = Mode.IDLE

        # rotation_mode default set to tilt so that tilting occurs before moving (the trackball motion)
        self.rotation_mode = "tilt"
        self._prev_signature: tuple | None = None

    def set_rotation_mode(self, mode: str) -> str:
        """Switch one-hand rotation between 'tilt' and 'move'. Resets temporal
        state so the switch doesn't lurch. Returns the active mode."""
        if mode not in ("tilt", "move"):
            return self.rotation_mode
        self.rotation_mode = mode
        self.reset()
        return self.rotation_mode

    def reset(self) -> None:
        """Startup has no prior data --- drop temporal state."""
        for f in self._all_filters:
            f.reset()
        self._prev_roll = None
        self._prev_normal = None
        self._prev_scale = None
        self._prev_centroid = None
        self._prev_handdist = None
        self._prev_midpoint = None
        self._prev_handangle = None

    def update(self, frame: Frame) -> ControlCommand:
        """Advance the engine by one frame and return the control command."""
        t = frame.timestamp

        # ignore hands that are too small, and gestures that have too low confiednce values
        gt = self.cfg.gestures
        candidates = [h for h in frame.hands if h.score >= gt.min_hand_score]
        candidates = sorted(candidates, key=lambda h: float(h.xy(LM.WRIST)[0]))
        hands, feats = [], []
        for h in candidates:
            f = feat_mod.extract(h)
            if f.hand_scale >= gt.min_hand_scale:
                hands.append(h)
                feats.append(f)

        # classify hands present
        raw_gestures = [classify(f, self.cfg.gestures) for f in feats]

        gesture_labels = {f"hand{i}": g.value for i, g in enumerate(raw_gestures)}

        # no differentiation should occur for different gestures (only for delta calculations)
        # wipe out temporal state if gesture changes (e.g., open hand to close hand)
        signature = tuple(g.value for g in raw_gestures)
        if signature != self._prev_signature:
            self.reset()
        self._prev_signature = signature

        raw_mode = self._decide_mode(raw_gestures)
        mode = self._mode_debounce.update(raw_mode)

        if mode != self._active_mode:
            self.reset()
            self._active_mode = mode

        cmd = ControlCommand(
            mode=mode.value,
            gestures=gesture_labels,
            num_hands=len(hands),
            timestamp=t,
        )

        # produce delta calculations for active mode
        if mode == Mode.HOLD:
            cmd.frozen = True
        elif mode == Mode.ROTATE:
            self._do_rotate(t, feats[0], cmd)
        elif mode == Mode.PAN:
            self._do_pan(t, feats[0], cmd)
        elif mode == Mode.TWO_HAND:
            self._do_two_hand(t, feats, cmd)
        elif mode == Mode.POINT:
            self._do_point(hands[0], feats[0], cmd)
        # IDLE: leave all deltas at zero.

        return cmd

    def _decide_mode(self, gestures: list[Gesture]) -> Mode:
        """Decide which mode is active."""
        if len(gestures) == 0:
            return Mode.IDLE
        if len(gestures) >= 2:
            # closed hand on either one is a brake
            if Gesture.FIST in gestures[:2]:
                return Mode.HOLD
            # if both hands usable...
            usable = {Gesture.OPEN_PALM, Gesture.PINCH, Gesture.POINT}
            if all(g in usable for g in gestures[:2]):
                return Mode.TWO_HAND
            return Mode.IDLE

        g = gestures[0]
        if g == Gesture.FIST:
            return Mode.HOLD
        if g == Gesture.OPEN_PALM:
            return Mode.ROTATE
        if g == Gesture.PINCH:
            return Mode.PAN
        if g == Gesture.POINT:
            return Mode.POINT
        return Mode.IDLE

    # per-mode deltas
    def _do_rotate(self, t, f, cmd: ControlCommand) -> None:
        """
        t = timestamp, f = feature set for a hand, gains = how much effect motion should have

        One open hand drives yaw/pitch/roll -- all three combine every frame,
        so you can angle and twist at once. NO zoom here (so a rotate can never
        accidentally zoom). Yaw/pitch source depends on self.rotation_mode:
          'tilt' -> the tilt of your palm (orientation); hold a tilt and the
                    object holds it. A fist freezes + re-grabs (clutch).
          'move' -> trackball: where your hand moves.
        """
        g = self.cfg.gains

        if self.rotation_mode == "tilt":
            # yaw/pitch from change in palm orientation (normal x/y)
            nrm = np.asarray(self._f_normal(t, f.palm_normal))
            if self._prev_normal is None:
                self._prev_normal = nrm
            yaw = float(nrm[0] - self._prev_normal[0]) * g.tilt_gain
            pitch = -float(nrm[1] - self._prev_normal[1]) * g.tilt_gain
            self._prev_normal = nrm
        else:
            # trackball: yaw/pitch from hand position
            c_s = np.asarray(self._f_centroid(t, f.centroid))
            if self._prev_centroid is None:
                self._prev_centroid = c_s
            d = c_s - self._prev_centroid
            self._prev_centroid = c_s
            yaw = float(d[0]) * g.yaw_pitch_gain  # hand right -> turn right
            pitch = -float(d[1]) * g.yaw_pitch_gain   # hand up -> tip forward

        # roll from in-plane palm twist (both modes; very stable)
        roll_s = self._f_roll(t, f.roll_angle)
        if self._prev_roll is None:
            self._prev_roll = roll_s
        droll = angle_delta(roll_s, self._prev_roll)
        self._prev_roll = roll_s

        cmd.rotate = RotateDelta(
            yaw=_shape(yaw, g.rotate_deadzone, g.max_rotate_step),
            pitch=_shape(pitch, g.rotate_deadzone, g.max_rotate_step),
            roll=_shape(droll * g.roll_gain, g.rotate_deadzone, g.max_rotate_step),
        )

    def _do_pan(self, t, f, cmd: ControlCommand) -> None:
        g = self.cfg.gains
        c_s = np.asarray(self._f_centroid(t, f.centroid))
        if self._prev_centroid is None:
            self._prev_centroid = c_s
        d = c_s - self._prev_centroid
        self._prev_centroid = c_s
        cmd.pan = PanDelta(
            x=_shape(float(d[0]) * g.pan_gain, g.pan_deadzone, g.max_pan_step),
            y=_shape(float(d[1]) * g.pan_gain, g.pan_deadzone, g.max_pan_step),
        )

    def _do_two_hand(self, t, feats, cmd: ControlCommand) -> None:
        g = self.cfg.gains
        c0, c1 = feats[0].centroid, feats[1].centroid

        # zoom: relative change in distance between the two hands (pinch-to-zoom).
        # spread apart = zoom in, bring together = zoom out.
        dist = float(np.linalg.norm(c0 - c1))
        dist_s = float(self._f_handdist(t, dist))
        if self._prev_handdist is None:
            self._prev_handdist = dist_s
        rel = (dist_s - self._prev_handdist) / max(self._prev_handdist, 1e-6)
        self._prev_handdist = dist_s
        cmd.zoom = _shape(rel * g.zoom_gain, g.zoom_deadzone, g.max_zoom_step)

        # pan: motion of the midpoint between the hands.
        mid = (c0 + c1) / 2.0
        mid_s = np.asarray(self._f_midpoint(t, mid))
        if self._prev_midpoint is None:
            self._prev_midpoint = mid_s
        d = mid_s - self._prev_midpoint
        self._prev_midpoint = mid_s
        cmd.pan = PanDelta(
            x=_shape(float(d[0]) * g.pan_gain, g.pan_deadzone, g.max_pan_step),
            y=_shape(float(d[1]) * g.pan_gain, g.pan_deadzone, g.max_pan_step),
        )

        # roll: turning both hands like a steering wheel. The angle of the line
        # connecting the two hands, differentiated, drives roll rotation.
        vec = c1 - c0
        angle = float(np.arctan2(vec[1], vec[0]))
        angle_s = self._f_handangle(t, angle)
        if self._prev_handangle is None:
            self._prev_handangle = angle_s
        droll = angle_delta(angle_s, self._prev_handangle)
        self._prev_handangle = angle_s
        cmd.rotate = RotateDelta(
            roll=_shape(droll * g.two_hand_rotate_gain, g.rotate_deadzone, g.max_rotate_step)
        )

    def _do_point(self, hand: HandLandmarks, f, cmd: ControlCommand) -> None:
        """RESERVED for the future physics hook. Still subject to change."""
        v = hand.xy(LM.INDEX_TIP) - hand.xy(LM.INDEX_MCP)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return
        direction = (float(v[0] / n), float(v[1] / n))
        # Nearest edge the flow points *from* is the opposite of where it points to.
        if abs(direction[0]) >= abs(direction[1]):
            edge = "left" if direction[0] > 0 else "right"
        else:
            edge = "top" if direction[1] > 0 else "bottom"
        cmd.flow = FlowControl(direction=direction, inlet_edge=edge, speed=None, density=None)
