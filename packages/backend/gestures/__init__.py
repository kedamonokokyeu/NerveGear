"""
gestures

Hand-gesture control engine for NerveGear's spatial-sculpting prototype.

The camera and MediaPipe adapter (gestures.camera) is imported separately
on demand — the pure engine and test suite don't need OpenCV or MediaPipe.
"""

from .commands import ControlCommand, FlowControl, PanDelta, RotateDelta
from .engine import GestureEngine, Mode
from .features import HandFeatures, extract
from .gestures import Gesture, classify
from .landmarks import LM, Frame, HandLandmarks, Lm, from_normalized_list

__all__ = [
    "GestureEngine",
    "Mode",
    "ControlCommand",
    "RotateDelta",
    "PanDelta",
    "FlowControl",
    "HandFeatures",
    "extract",
    "Gesture",
    "classify",
    "Frame",
    "HandLandmarks",
    "LM", "Lm",
    "from_normalized_list",
]
