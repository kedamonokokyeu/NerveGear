"""
commands.py

Continuous flow of deltas from the frontend. Backend provides commans to
rotate, zoom, etc., whereas frontend applies how fast transformation is or whether
it transforms or not.

Backend is stateless --- does not need to know current camera angle/zoom. Only
tells to rotate, zoom, etc. Frontend applies all non-zero channels every frame
  --- means that two hands, one hand, no difference because unused channels are set to 0

FUTURE-PHYSICS HOOK
The `flow` block is reserved for NerveGear CFD integration (e.g., inflow of liquid coolant
or wind flow). Subject to change.

Yaw = left/right rotation
Pitch = up/down rotation
roll = twist of the hand
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field


@dataclass
class RotateDelta:
    """Incremental rotation in radians. roll is the stable in-plane channel; yaw
    and pitch come from palm-normal tilt and are noisier (see features.py)."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class PanDelta:
    """Incremental translation in normalized screen units. 
    y already follows screen convention (down is positive)."""
    x: float = 0.0
    y: float = 0.0


@dataclass
class FlowControl:
    """RESERVED for future physics. All None until the CFD bridge is enabled.
      direction : (x, y) unit vector for inlet flow, from a POINT gesture.
      speed     : 0..1 normalized inlet speed (maps to /predict 'speed').
      density   : 0..1 proxy (a squeeze gesture), maps toward Reynolds number.
      inlet_edge: 'left'|'right'|'top'|'bottom', nearest edge the flow points from.
    """
    direction: tuple[float, float] | None = None
    speed: float | None = None
    density: float | None = None
    inlet_edge: str | None = None


@dataclass
class ControlCommand:
    """
    Captures one frame, sends it as JSON object. 
    E.g., how much to rotate, zoom, pan, scale a scnee.
    Also includes gesture states and timestamps.
    """

    # current interaction mode (idle, rotate, zoom)
    mode: str = "idle"

    # whatever is nonzero will be applie in frontend
    rotate: RotateDelta = field(default_factory=RotateDelta)
    zoom: float = 0.0
    pan: PanDelta = field(default_factory=PanDelta)
    scale: float = 0.0

    # fist brake locks view
    frozen: bool = False

    # per-hand gesture labels
    gestures: dict = field(default_factory=dict)
    num_hands: int = 0

    # physics plug in for later
    flow: FlowControl = field(default_factory=FlowControl)

    # timestamp (seconds) of the frame this command was derived from.
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Plain nested dict, JSON-serializable. `flow` stays as nested
        Nones/values; the frontend can ignore it until physics is enabled."""
        return asdict(self)
