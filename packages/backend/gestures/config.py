"""
config.py

Tunable parameters (e.g, zoom sensitivity).

GestureThresholds --- numeric cutoffs for classifying gestures (how close 
two fingers have to be to be considered a specific classification)

SmoothingParams --- how much motion smoothing applied to reduce jitter

DebounceParams --- how many frames a gesture must persist before being classified

ControlGains --- how strongly hand movements affect rotation, zoom, and pan

CameraConfig --- webcam settings, MediaPipe model parameters

ServerConfig --- network settings for broadcasting commands
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GestureThresholds:
    """
    Cutoffs used by gestures.py to turn continuous features into discrete
    gesture labels. All distances are in hand-scale units (1.0 == wrist-to-
    middle-knuckle), so they are camera-distance independent.
    """

    fist_openness_max: float = 0.45
    open_openness_min: float = 0.70
    open_min_fingers: int = 4

    # distance between thumb-tip and fingertip threshold for pinch
    # must also be forward of the wrist 
    pinch_close: float = 0.45
    pinch_forward_min: float = 1.10

    # GATING FOR FALSE POSITIVES
    # drop a detected hand whose detector confidence is below this
    min_hand_score: float = 0.5
    min_hand_scale: float = 0.04 # tiny hand far away guard


@dataclass
class SmoothingParams:
    """
    One-Euro filter parameters (see smoothing.py). 
    Hand moves slowly = strong smoothing
    Hand moves fast = reduced smoothing
    Lower min_cutoff = more
    """
    # baseline smoothing strength
    min_cutoff: float = 0.5
    beta: float = 0.010 # how much filter adapts to speed
    d_cutoff: float = 1.0 # smoothing for derivative


@dataclass
class DebounceParams:
    """
    How many consecutive frames a gesture must hold before we believe it.
    Prevents a single noisy frame from flipping the interaction mode.
    """
    frames_to_confirm: int = 5


@dataclass
class ControlGains:
    """Scale factors mapping raw feature deltas to control-command magnitudes.
    These set the overall 'feel' / sensitivity. Defaults are deliberately gentle
    so the motion is slow and precise (no squiggle)."""

    # ONE-HAND rotation:
    yaw_pitch_gain: float = 2.5       # 'move' mode: hand position movement -> yaw/pitch
    tilt_gain: float = 6.0            # 'tilt' mode: hand tilt (palm normal) -> yaw/pitch
    roll_gain: float = 1.6            # palm twist -> roll (both modes)
    # TWO-HAND:
    zoom_gain: float = 3.5            # spread/squeeze (relative) -> zoom
    two_hand_rotate_gain: float = 1.6  # "steering wheel" angle -> roll
    # PINCH:
    pan_gain: float = 1.5             # hand translation -> pan

    # ignore change valuations below these to prevent residual jitter 
    rotate_deadzone: float = 0.0035
    zoom_deadzone: float = 0.005
    pan_deadzone: float = 0.004

    # Per-frame clamps: cap how far any one frame can move things. Kept small so
    # nothing can lurch -- the model only ever moves a little per frame.
    max_rotate_step: float = 0.06
    max_zoom_step: float = 0.04
    max_pan_step: float = 0.05


@dataclass
class CameraConfig:
    """Webcam / capture settings used by camera.py and the runner."""

    device_index: int = 0          # which webcam (0 is usually the built-in)
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    mirror: bool = True            # selfie-view: flip horizontally so motion feels natural
    max_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    model_asset_path: str | None = None


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8200               # 8000 is taken by the NerveGear CFD /predict server
    broadcast_hz: float = 60.0     # how often the WS pushes the latest command


@dataclass
class Config:
    """
    Each configuration will be instantiate independently.
    FYI, field() creates fresh instance of each section automatically when making new Config() object.
    """
    gestures: GestureThresholds = field(default_factory=GestureThresholds)
    smoothing: SmoothingParams = field(default_factory=SmoothingParams)
    debounce: DebounceParams = field(default_factory=DebounceParams)
    gains: ControlGains = field(default_factory=ControlGains)
    camera: CameraConfig = field(default_factory=CameraConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

DEFAULT = Config()
