"""
features.py

From the 21 lanmarks, meaningful geometric feature interpretation occurs here.
Gesture classification and control mapping requires reasoning:
    - e.g., how open a hand is, how fast do we need to rotate, etc.

The goal is to make the features here clean and independent of camera-distance,
ultimately simplifying the logic for gesture classification and control.
Additionally, normalization so that future logic files for hand interpretations
aren't affected by distance.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .landmarks import FINGERS, PALM_POINTS, HandLandmarks, LM


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between 3D points."""
    return float(np.linalg.norm(a - b))


@dataclass(frozen=True)
class HandFeatures:
    """
    Normalized description of the hand.

    hand_scale      : wrist->middle-knuckle distance in image units. This is our
                      "ruler": bigger = hand is closer to the camera. Used both to
                      normalize other distances AND, deliberately, as the zoom
                      signal.
    centroid        : (x, y) palm center in image units (0..1). Drives pan.
    extended        : dict finger-name -> bool, whether each of the 4 fingers is
                      straightened.
    thumb_extended  : bool, handled separately (thumb bends sideways).
    num_extended    : count of straight fingers incl. thumb (0..5).
    openness        : continuous 0..1, how spread/open the hand is. Smoother than
                      the integer finger count, so it is what the zoom/scale logic
                      and the fist/open hysteresis prefer.
    pinch           : thumb-tip to index-tip distance, normalized by hand_scale.
                      ~0 when fingers touch (a pinch), ~1+ when apart.
    pinch_forward   : distance from the thumb/index contact point to the wrist,
                      normalized by hand_scale. This is what separates a *pinch*
                      (contact reaches forward, away from the palm -> large) from
                      a *fist* (everything balls up near the wrist -> small).
    roll_angle      : in-plane palm rotation in radians (atan2 of the index->pinky
                      knuckle vector). Very stable; primary rotation channel.
    palm_normal     : unit 3-vector roughly perpendicular to the palm. Its tilt
                      gives yaw/pitch, but it uses the noisier z axis, so the
                      engine smooths it hard and treats it as secondary.
    """

    hand_scale: float
    centroid: np.ndarray
    extended: dict
    thumb_extended: bool
    num_extended: int
    openness: float
    pinch: float
    pinch_forward: float
    roll_angle: float
    palm_normal: np.ndarray


def _finger_extended(hand: HandLandmarks, mcp: LM, pip: LM, tip: LM, scale: float) -> bool:
    """
    When tip is farther from wrist then PIP joint, then finger is extended.
    A small margin scaled by han size so somewhat curle finger doesn't switch
    classification from frame to frame.
    """
    wrist = hand.xy(LM.WRIST)
    tip_d = _dist(hand.xy(tip), wrist)
    pip_d = _dist(hand.xy(pip), wrist)
    margin = 0.05 * scale
    return tip_d > pip_d + margin


def _thumb_extended(hand: HandLandmarks, scale: float) -> bool:
    """
    The thumb opens sideways across the palm rather than curling toward the
    wrist, so this measures how far the thumb tip has moved away from the index
    knuckle relative to the thumb's own base (MCP). When the thumb tucks in, tip
    and index-knuckle nearly coincide.
    """
    tip_to_index_mcp = _dist(hand.xy(LM.THUMB_TIP), hand.xy(LM.INDEX_MCP))
    return tip_to_index_mcp > 0.5 * scale


def _palm_normal(hand: HandLandmarks) -> np.ndarray:
    """
    Checks whether palm is facing up, down, left, etc.
    Two edge vectors (index_MCP and pinky_MCP), takes cross product,
    normalizes to unit length, which tells which direction the palm faces.
    """
    o = hand.p(LM.WRIST)
    v1 = hand.p(LM.INDEX_MCP) - o
    v2 = hand.p(LM.PINKY_MCP) - o
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def extract(hand: HandLandmarks) -> HandFeatures:
    """Compute the full feature set for one hand."""
    # wrist -> middle knuckle is the most stable span on the hand (it barely
    # changes as fingers move), so this is our normalizer
    scale = _dist(hand.xy(LM.WRIST), hand.xy(LM.MIDDLE_MCP))
    scale = max(scale, 1e-6)  # guard against degenerate/zero hands

    extended = {
        name: _finger_extended(hand, mcp, pip, tip, scale)
        for name, (mcp, pip, _dip, tip) in FINGERS.items()
    }
    thumb_ext = _thumb_extended(hand, scale)
    num_extended = int(sum(extended.values())) + int(thumb_ext)

    # continuous measure of how open the hand is
    # based on average fingertip to wrist distance
    tip_ids = (LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP)
    wrist = hand.xy(LM.WRIST)
    mean_tip = float(np.mean([_dist(hand.xy(t), wrist) for t in tip_ids]))
    openness = float(np.clip(mean_tip / scale / 2.0, 0.0, 1.5))

    # computes thumb-tip to index-tip distance
    thumb_tip = hand.xy(LM.THUMB_TIP)
    index_tip = hand.xy(LM.INDEX_TIP)
    pinch = _dist(thumb_tip, index_tip) / scale
    contact = (thumb_tip + index_tip) / 2.0
    pinch_forward = _dist(contact, wrist) / scale

    # hand's roll angle (rotation in the image plane) 
    idx_to_pinky = hand.xy(LM.PINKY_MCP) - hand.xy(LM.INDEX_MCP)
    roll_angle = float(np.arctan2(idx_to_pinky[1], idx_to_pinky[0]))
    normal = _palm_normal(hand)

    # averages palm landmarks to get hand's center position
    centroid = np.mean([hand.xy(p) for p in PALM_POINTS], axis=0)

    return HandFeatures(
        hand_scale=scale,
        centroid=centroid,
        extended=extended,
        thumb_extended=thumb_ext,
        num_extended=num_extended,
        openness=openness,
        pinch=pinch,
        pinch_forward=pinch_forward,
        roll_angle=roll_angle,
        palm_normal=normal,
    )
