"""
Landmark --- single 3D point that markets key spot on the hand.
MediaPipe Hands detects 21 landmarks per hands in the x, y, an z coordinates.
Defines the data contract for the remainder of the gestures pipeline.
"""

# treat type hints as strings until runtime
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class LM(IntEnum):
    """
    LM is short for Landmark.
    This class is an enumeration that gies names to the 21 hand landmarks 
    detected by MediaPipe. Mostly here just for best practices:
        - CMC = carpometacarpal joint (thumb connection to wrist)
        - IP = interphalengeal joint (single bend between thumb and two bones)

        - LM tells that a specific index refers to a landmark for a finger/joint
        - easy indexing in the future
        - MCP = metacarpophalangeal joint (big knuckle)
        - PIP = proximal interphalangeal joint (middle joint of finger)
        - DIP = distal interphalengeal joint (joint closer to finger tip)
        - TIP = fingertip landmark
    """

    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# define fingers for simplicity
# thumb is special landmark
THUMB = (LM.THUMB_CMC, LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP)
FINGERS = {
    "index": (LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP),
    "middle": (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
    "ring": (LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP),
    "pinky": (LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP),
}

# palm points needed to identify when hand is clenched later
PALM_POINTS = (LM.WRIST, LM.INDEX_MCP, LM.MIDDLE_MCP, LM.RING_MCP, LM.PINKY_MCP)

N_LANDMARKS = 21


@dataclass(frozen=True) # once object created, attributes cannot be changed
class HandLandmarks:
    """
    One detected hand in one videoframe.
    points --> NumPy array of shape (21, 3) that contains (x, y, z) position for each landmark
    handeness --> 'left' or 'right' han 
    score --> confidence value between 0 or 1 (how confident is the detector)
    """

    points: np.ndarray
    handedness: str = "Unknown"
    score: float = 1.0

    def __post_init__(self) -> None:
        arr = np.asarray(self.points, dtype=np.float64)
        if arr.shape != (N_LANDMARKS, 3):
            raise ValueError(
                f"HandLandmarks.points must be shape (21, 3); instead got {arr.shape}"
            )
        # frozen dataclass -> must set via object.__setattr__ for low level bypass
        object.__setattr__(self, "points", arr)

    def p(self, idx: LM) -> np.ndarray:
        """Get the (x, y, z) position of specified landmark."""
        return self.points[int(idx)]

    def xy(self, idx: LM) -> np.ndarray:
        """Return just the in-image (x, y) of a landmark. Most 2D geometry
        will only require x and y. Will be use for rotation/panning."""
        return self.points[int(idx), :2]


@dataclass(frozen=True)
class Frame:
    """
    Each frame can contain 0, 1, or 2 hands. Each hand is store as a
    HandLandmarks object (just defined above). 

    For motion tracking, smoothing, and velocity calculations (e.g., how
    fast the CAD import will rotate) timestamp for the frame is necessary.
    """

    hands: list[HandLandmarks]
    timestamp: float

    @property # automatic method running for simplicity
    def num_hands(self) -> int:
        return len(self.hands)


def from_normalized_list(raw_hands: list[dict], timestamp: float) -> Frame:
    """
    Converts JSON-style han data from browser into the define objects (HandLandmarks and Frame)
    for computations. Already accouts for how MediaPipe.js sends from browser.

    raw_hands --> list of dictionaries, describes one detected hand
    timestamp --> when frame was captured

    Expected element shape:
        {
          "handedness": "Left" | "Right",
          "score": 0.97,
          "landmarks": [{"x":.., "y":.., "z":..}, ... 21 of them ...]
        }
    """
    hands: list[HandLandmarks] = []
    for hand in raw_hands:
        landmarks = hand["landmarks"]
        # make z fallback to 0 for 2d computations
        pts = np.array([[p["x"], p["y"], p.get("z", 0.0)] for p in landmarks], dtype=np.float64)
        hands.append(
            HandLandmarks(
                points=pts,
                handedness=hand.get("handedness", "Unknown"),
                score=float(hand.get("score", 1.0)),
            )
        )
    return Frame(hands=hands, timestamp=timestamp)


# Backwards-compat alias (tools/synthetic.py and older callers use `Lm`).
Lm = LM
