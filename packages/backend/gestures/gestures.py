"""
gestures.py

Takes the numeric hand features from features.py.
Maps them to discrete gesture labels.
"""

from __future__ import annotations
from enum import Enum
from .config import GestureThresholds
from .features import HandFeatures


class Gesture(str, Enum):
    """The vocabulary. str-Enum so it serializes straight to JSON as its name."""

    NONE = "none"            # no hand / undecidable
    FIST = "fist"            # all fingers curled  -> BRAKE / hold (stop motion)
    OPEN_PALM = "open_palm"  # flat open hand      -> rotate + zoom (the main driver)
    PINCH = "pinch"          # thumb + index touch -> grab / pan
    POINT = "point"          # index only extended -> reserved for future flow direction


def classify(f: HandFeatures, cfg: GestureThresholds) -> Gesture:
    """
    Return the gesture label for one hand.
    The order follows the gestures with the highest semantic meaning.
      1. PINCH first  -- it is the explicit "grab" intent; if the fingers are
         pinched and reaching forward, honor it even if openness is ambiguous.
      2. FIST next    -- the safety brake; we want it to win over a vague
         open-ish reading so "close your hand to stop" is reliable.
      3. POINT        -- index-only, reserved for drawing flow direction later.
      4. OPEN_PALM    -- the default manipulation pose.
      5. NONE         -- nothing matched cleanly; engine treats this as a hold.
    """
    fingers_extended = f.extended 

    if f.pinch < cfg.pinch_close and f.pinch_forward > cfg.pinch_forward_min:
        return Gesture.PINCH

    if f.openness < cfg.fist_openness_max and f.num_extended <= 1:
        return Gesture.FIST

    if fingers_extended["index"] and not fingers_extended["middle"] and not fingers_extended["ring"] and not fingers_extended["pinky"]:
        return Gesture.POINT

    if f.openness > cfg.open_openness_min and f.num_extended >= cfg.open_min_fingers:
        return Gesture.OPEN_PALM

    return Gesture.NONE
