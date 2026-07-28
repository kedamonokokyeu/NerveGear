"""
tools/synthetic.py

Hand-crafted 21-point hand poses and geometric transforms for testing the gesture
pipeline without a webcam.

Poses are in MediaPipe's normalized coordinate system (x right, y down, z toward
camera is negative) and verified to classify correctly. Treat them as test
fixtures.
"""

from __future__ import annotations

import numpy as np

from gestures.landmarks import HandLandmarks, Lm


def _pose(points: dict[Lm, tuple[float, float, float]]) -> np.ndarray:
    """Assemble a (21,3) array from a dict of named landmark -> (x,y,z)."""
    arr = np.zeros((21, 3), dtype=np.float64)
    for lm, xyz in points.items():
        arr[int(lm)] = xyz
    return arr


# --- OPEN PALM: all five fingers extended, fingers pointing up ---------------
_OPEN = {
    Lm.WRIST: (0.50, 0.80, 0.0),
    Lm.THUMB_CMC: (0.42, 0.75, 0.0),
    Lm.THUMB_MCP: (0.37, 0.70, 0.0),
    Lm.THUMB_IP: (0.33, 0.66, 0.0),
    Lm.THUMB_TIP: (0.30, 0.62, 0.0),
    Lm.INDEX_MCP: (0.44, 0.62, 0.0),
    Lm.INDEX_PIP: (0.43, 0.52, 0.0),
    Lm.INDEX_DIP: (0.42, 0.45, 0.0),
    Lm.INDEX_TIP: (0.42, 0.38, 0.0),
    Lm.MIDDLE_MCP: (0.50, 0.60, 0.0),
    Lm.MIDDLE_PIP: (0.50, 0.49, 0.0),
    Lm.MIDDLE_DIP: (0.50, 0.42, 0.0),
    Lm.MIDDLE_TIP: (0.50, 0.35, 0.0),
    Lm.RING_MCP: (0.56, 0.62, 0.0),
    Lm.RING_PIP: (0.57, 0.52, 0.0),
    Lm.RING_DIP: (0.58, 0.45, 0.0),
    Lm.RING_TIP: (0.58, 0.39, 0.0),
    Lm.PINKY_MCP: (0.62, 0.66, 0.0),
    Lm.PINKY_PIP: (0.63, 0.58, 0.0),
    Lm.PINKY_DIP: (0.64, 0.53, 0.0),
    Lm.PINKY_TIP: (0.64, 0.48, 0.0),
}

# --- FIST: all fingers curled toward the palm, thumb tucked ------------------
_FIST = {
    Lm.WRIST: (0.50, 0.80, 0.0),
    Lm.THUMB_CMC: (0.42, 0.75, 0.0),
    Lm.THUMB_MCP: (0.40, 0.71, 0.0),
    Lm.THUMB_IP: (0.43, 0.68, 0.0),
    Lm.THUMB_TIP: (0.46, 0.66, 0.0),
    Lm.INDEX_MCP: (0.44, 0.62, 0.0),
    Lm.INDEX_PIP: (0.45, 0.66, 0.0),
    Lm.INDEX_DIP: (0.46, 0.69, 0.0),
    Lm.INDEX_TIP: (0.46, 0.70, 0.0),
    Lm.MIDDLE_MCP: (0.50, 0.60, 0.0),
    Lm.MIDDLE_PIP: (0.50, 0.66, 0.0),
    Lm.MIDDLE_DIP: (0.50, 0.70, 0.0),
    Lm.MIDDLE_TIP: (0.50, 0.71, 0.0),
    Lm.RING_MCP: (0.56, 0.62, 0.0),
    Lm.RING_PIP: (0.55, 0.66, 0.0),
    Lm.RING_DIP: (0.55, 0.70, 0.0),
    Lm.RING_TIP: (0.55, 0.71, 0.0),
    Lm.PINKY_MCP: (0.62, 0.66, 0.0),
    Lm.PINKY_PIP: (0.60, 0.69, 0.0),
    Lm.PINKY_DIP: (0.59, 0.71, 0.0),
    Lm.PINKY_TIP: (0.58, 0.72, 0.0),
}

# --- PINCH: thumb & index tips meet forward; middle/ring/pinky extended ------
_PINCH = dict(_OPEN)
_PINCH.update({
    Lm.THUMB_IP: (0.42, 0.50, 0.0),
    Lm.THUMB_TIP: (0.44, 0.45, 0.0),
    Lm.INDEX_PIP: (0.44, 0.54, 0.0),
    Lm.INDEX_DIP: (0.44, 0.49, 0.0),
    Lm.INDEX_TIP: (0.45, 0.46, 0.0),
})

# --- POINT: index extended, middle/ring/pinky curled, thumb tucked -----------
_POINT = {
    Lm.WRIST: (0.50, 0.80, 0.0),
    Lm.THUMB_CMC: (0.42, 0.75, 0.0),
    Lm.THUMB_MCP: (0.40, 0.71, 0.0),
    Lm.THUMB_IP: (0.43, 0.68, 0.0),
    Lm.THUMB_TIP: (0.46, 0.66, 0.0),
    Lm.INDEX_MCP: (0.44, 0.62, 0.0),
    Lm.INDEX_PIP: (0.44, 0.52, 0.0),
    Lm.INDEX_DIP: (0.44, 0.43, 0.0),
    Lm.INDEX_TIP: (0.44, 0.34, 0.0),
    Lm.MIDDLE_MCP: (0.50, 0.60, 0.0),
    Lm.MIDDLE_PIP: (0.50, 0.66, 0.0),
    Lm.MIDDLE_DIP: (0.50, 0.70, 0.0),
    Lm.MIDDLE_TIP: (0.50, 0.71, 0.0),
    Lm.RING_MCP: (0.56, 0.62, 0.0),
    Lm.RING_PIP: (0.55, 0.66, 0.0),
    Lm.RING_DIP: (0.55, 0.70, 0.0),
    Lm.RING_TIP: (0.55, 0.71, 0.0),
    Lm.PINKY_MCP: (0.62, 0.66, 0.0),
    Lm.PINKY_PIP: (0.60, 0.69, 0.0),
    Lm.PINKY_DIP: (0.59, 0.71, 0.0),
    Lm.PINKY_TIP: (0.58, 0.72, 0.0),
}


def open_palm(handedness: str = "Right") -> HandLandmarks:
    return HandLandmarks(points=_pose(_OPEN), handedness=handedness)


def fist(handedness: str = "Right") -> HandLandmarks:
    return HandLandmarks(points=_pose(_FIST), handedness=handedness)


def pinch(handedness: str = "Right") -> HandLandmarks:
    return HandLandmarks(points=_pose(_PINCH), handedness=handedness)


def point(handedness: str = "Right") -> HandLandmarks:
    return HandLandmarks(points=_pose(_POINT), handedness=handedness)


# --------------------------------------------------------------- transforms
def translate(hand: HandLandmarks, dx: float, dy: float) -> HandLandmarks:
    """Shift the whole hand in the image plane (z unchanged)."""
    pts = hand.points.copy()
    pts[:, 0] += dx
    pts[:, 1] += dy
    return HandLandmarks(points=pts, handedness=hand.handedness, score=hand.score)


def scale_about(hand: HandLandmarks, factor: float, center: Lm = Lm.WRIST) -> HandLandmarks:
    """Scale the hand about a center point. factor>1 simulates the hand moving
    CLOSER to the camera (it appears bigger) -> drives zoom-in."""
    pts = hand.points.copy()
    c = hand.points[int(center), :2]
    pts[:, :2] = c + (pts[:, :2] - c) * factor
    return HandLandmarks(points=pts, handedness=hand.handedness, score=hand.score)


def tilt(hand: HandLandmarks, yaw: float = 0.0, pitch: float = 0.0,
         center: Lm = Lm.WRIST) -> HandLandmarks:
    """Rotate the hand in 3D about the wrist: `yaw` about the vertical (Y) axis,
    `pitch` about the horizontal (X) axis. This changes the palm's orientation
    (and thus its normal), simulating you physically tilting your hand -- which is
    what drives 'tilt' rotation mode."""
    pts = hand.points.copy()
    c = hand.points[int(center)].copy()
    rel = pts - c
    if yaw:
        ca, sa = np.cos(yaw), np.sin(yaw)
        x = rel[:, 0] * ca + rel[:, 2] * sa
        z = -rel[:, 0] * sa + rel[:, 2] * ca
        rel[:, 0], rel[:, 2] = x, z
    if pitch:
        ca, sa = np.cos(pitch), np.sin(pitch)
        y = rel[:, 1] * ca - rel[:, 2] * sa
        z = rel[:, 1] * sa + rel[:, 2] * ca
        rel[:, 1], rel[:, 2] = y, z
    pts = c + rel
    return HandLandmarks(points=pts, handedness=hand.handedness, score=hand.score)


def rotate_in_plane(hand: HandLandmarks, angle: float, center: Lm = Lm.WRIST) -> HandLandmarks:
    """Rotate the hand in the image plane by `angle` radians about a center.
    Simulates a palm 'twist' -> drives roll rotation."""
    pts = hand.points.copy()
    c = hand.points[int(center), :2]
    ca, sa = np.cos(angle), np.sin(angle)
    rel = pts[:, :2] - c
    rot = np.empty_like(rel)
    rot[:, 0] = rel[:, 0] * ca - rel[:, 1] * sa
    rot[:, 1] = rel[:, 0] * sa + rel[:, 1] * ca
    pts[:, :2] = c + rot
    return HandLandmarks(points=pts, handedness=hand.handedness, score=hand.score)
