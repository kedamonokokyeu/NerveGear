"""
schema.py — the `axiom/solve.v1` contract, as code.
===================================================
Single source of truth for the JSON shape that connects the Studio, the reduced
model, and the 3D solver/surrogate. See docs/CONTRACT.md for prose; this module
is the executable version: parse + validate requests, encode/decode array
blocks, and build well-formed responses.

Design rules:
  * stdlib + numpy only — importable from anywhere (server, solver, scripts).
  * Engineering units at the boundary (µm, °C, mL/min, W/cm², kPa); SI inside.
  * Unknown keys are ignored (forward compatibility); missing required keys
    raise SchemaError with a path so the UI can show a precise message.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, asdict

import numpy as np

SCHEMA_VERSION = "axiom/solve.v1"

# Canonical metric names — engines may add keys, never rename these.
METRICS = (
    "max_junction_temp", "mean_junction_temp", "junction_gradient", "junction_cv",
    "coolant_dT", "thermal_resistance_KW", "thermal_resistance_Kcm2W",
    "thermal_margin", "pressure_drop_kPa", "pumping_power_W", "flow_mlmin",
    "COP", "Re", "laminar", "maldistribution_cv", "total_power_W",
    "peak_flux_Wcm2", "Dh_um",
)

EDGES = ("left", "right", "top", "bottom")
ENGINES = ("auto", "solver3d", "reduced")
RESOLUTIONS = {"draft": 200_000, "normal": 1_000_000, "fine": 4_000_000}  # voxel budgets


class SchemaError(ValueError):
    """Request failed validation. `path` locates the offending key."""

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"{path}: {message}")


# ── array blocks ─────────────────────────────────────────────────────────

def encode_array(a: np.ndarray) -> dict:
    """(ny, nx) float array -> transportable block (little-endian f32, b64)."""
    a = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    if a.ndim != 2:
        raise ValueError(f"encode_array expects 2-D, got shape {a.shape}")
    return {"ny": int(a.shape[0]), "nx": int(a.shape[1]),
            "encoding": "b64f32", "data": base64.b64encode(a.tobytes()).decode()}


def decode_array(block: dict, path: str = "array") -> np.ndarray:
    """Transport block -> (ny, nx) float32 array. Accepts b64f32 or nested lists."""
    if not isinstance(block, dict):
        raise SchemaError(path, "expected an array block object")
    try:
        ny, nx = int(block["ny"]), int(block["nx"])
    except (KeyError, TypeError, ValueError) as e:
        raise SchemaError(path, "array block needs integer 'ny' and 'nx'") from e
    if ny <= 0 or nx <= 0 or ny * nx > 16_000_000:
        raise SchemaError(path, f"unreasonable array size {ny}x{nx}")
    enc = block.get("encoding", "b64f32")
    if enc == "b64f32":
        try:
            raw = base64.b64decode(block["data"])
        except Exception as e:  # noqa: BLE001
            raise SchemaError(path, "invalid base64 data") from e
        if len(raw) != ny * nx * 4:
            raise SchemaError(path, f"data length {len(raw)} != ny*nx*4 = {ny * nx * 4}")
        return np.frombuffer(raw, dtype="<f4").reshape(ny, nx).astype(np.float32)
    if enc == "list":
        a = np.asarray(block["data"], dtype=np.float32)
        if a.shape != (ny, nx):
            raise SchemaError(path, f"list shape {a.shape} != ({ny},{nx})")
        return a
    raise SchemaError(path, f"unknown encoding {enc!r}")


# ── request dataclasses ──────────────────────────────────────────────────

@dataclass
class Die:
    L: float = 16000.0        # µm, x (flow)
    W: float = 12000.0        # µm, y
    thk: float = 2400.0       # µm, die thickness (etch comes down from the top)
    base_thk: float = 200.0   # µm, silicon left between channel floor and heated face
    lid: bool = True          # channels capped by a bonded cover
    solid: str = "silicon"


@dataclass
class Segment:
    x0: float; y0: float; x1: float; y1: float   # µm centreline
    w: float                                     # µm channel width
    depth: float                                 # µm etch depth
    level: int = 0

    @property
    def length_um(self) -> float:
        return float(np.hypot(self.x1 - self.x0, self.y1 - self.y0))


@dataclass
class Hotspot:
    cx: float; cy: float          # fraction of die [0,1]
    sigma: float                  # gaussian radius, fraction of die
    peak_wcm2: float


@dataclass
class Power:
    background_wcm2: float = 20.0
    hotspots: list = field(default_factory=list)     # [Hotspot]
    grid: np.ndarray | None = None                   # (ny, nx) W/cm², overrides synth


@dataclass
class Conditions:
    coolant: str = "water"
    flow_mlmin: float = 200.0
    inlet_temp_c: float = 25.0
    junction_limit_c: float = 105.0
    power: Power = field(default_factory=Power)


@dataclass
class KeepOut:
    x0: float; y0: float; x1: float; y1: float       # µm
    reason: str = ""


@dataclass
class Geometry:
    die: Die = field(default_factory=Die)
    etch: np.ndarray | None = None                   # (ny, nx) depth µm
    segments: list = field(default_factory=list)     # [Segment]
    parametric: dict | None = None                   # {"kind": ..., "params": {...}}
    keep_out: list = field(default_factory=list)     # [KeepOut]
    inlet_edge: str = "left"
    outlet_edge: str = "right"

    def has_channels(self) -> bool:
        return self.etch is not None or bool(self.segments) or self.parametric is not None


@dataclass
class SolveOptions:
    engine: str = "auto"
    resolution: str = "draft"
    returns: tuple = ("tj_map", "segment_flow")
    timeout_s: float = 30.0

    @property
    def voxel_budget(self) -> int:
        return RESOLUTIONS.get(self.resolution, RESOLUTIONS["draft"])


@dataclass
class SolveRequest:
    geometry: Geometry
    conditions: Conditions
    solve: SolveOptions

    def to_jsonable(self) -> dict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        g = d["geometry"]
        g["inlet"] = {"edge": g.pop("inlet_edge")}
        g["outlet"] = {"edge": g.pop("outlet_edge")}
        if self.geometry.etch is not None:
            g["etch"] = encode_array(self.geometry.etch)
        if self.conditions.power.grid is not None:
            d["conditions"]["power"]["grid"] = encode_array(self.conditions.power.grid)
        d["solve"]["return"] = list(d["solve"].pop("returns"))
        return d


# ── parsing ──────────────────────────────────────────────────────────────

def _num(d: dict, key: str, default, path: str, lo=None, hi=None) -> float:
    v = d.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError) as e:
        raise SchemaError(f"{path}.{key}", f"expected a number, got {v!r}") from e
    if lo is not None and v < lo:
        raise SchemaError(f"{path}.{key}", f"{v} below minimum {lo}")
    if hi is not None and v > hi:
        raise SchemaError(f"{path}.{key}", f"{v} above maximum {hi}")
    return v


def parse_request(payload: dict) -> SolveRequest:
    """dict (parsed JSON) -> validated SolveRequest. Raises SchemaError."""
    if not isinstance(payload, dict):
        raise SchemaError("$", "request body must be a JSON object")
    ver = payload.get("schema_version", SCHEMA_VERSION)
    if ver != SCHEMA_VERSION:
        raise SchemaError("$.schema_version", f"unsupported version {ver!r} (want {SCHEMA_VERSION})")

    g = payload.get("geometry") or {}
    dd = g.get("die") or {}
    die = Die(
        L=_num(dd, "L", 16000.0, "$.geometry.die", lo=500.0, hi=200_000.0),
        W=_num(dd, "W", 12000.0, "$.geometry.die", lo=500.0, hi=200_000.0),
        thk=_num(dd, "thk", 2400.0, "$.geometry.die", lo=50.0, hi=20_000.0),
        base_thk=_num(dd, "base_thk", 200.0, "$.geometry.die", lo=10.0, hi=10_000.0),
        lid=bool(dd.get("lid", True)),
        solid=str(dd.get("solid", "silicon")),
    )

    etch = None
    if g.get("etch") is not None:
        etch = decode_array(g["etch"], "$.geometry.etch")
        if float(np.nanmax(etch, initial=0.0)) > die.thk - die.base_thk + 1e-6:
            raise SchemaError("$.geometry.etch",
                              f"max etch depth {float(np.nanmax(etch)):.0f}µm exceeds "
                              f"die.thk - die.base_thk = {die.thk - die.base_thk:.0f}µm")
        etch = np.clip(np.nan_to_num(etch, nan=0.0), 0.0, None)

    segments = []
    for i, s in enumerate(g.get("segments") or []):
        p = f"$.geometry.segments[{i}]"
        if not isinstance(s, dict):
            raise SchemaError(p, "expected an object")
        segments.append(Segment(
            x0=_num(s, "x0", None, p), y0=_num(s, "y0", None, p),
            x1=_num(s, "x1", None, p), y1=_num(s, "y1", None, p),
            w=_num(s, "w", None, p, lo=1.0, hi=50_000.0),
            depth=_num(s, "depth", None, p, lo=1.0, hi=die.thk - die.base_thk),
            level=int(s.get("level", 0)),
        ))
    if len(segments) > 20_000:
        raise SchemaError("$.geometry.segments", f"{len(segments)} segments (max 20000)")

    keep_out = []
    for i, k in enumerate(g.get("keep_out") or []):
        p = f"$.geometry.keep_out[{i}]"
        keep_out.append(KeepOut(x0=_num(k, "x0", None, p), y0=_num(k, "y0", None, p),
                                x1=_num(k, "x1", None, p), y1=_num(k, "y1", None, p),
                                reason=str(k.get("reason", ""))))

    inlet = str((g.get("inlet") or {}).get("edge", "left"))
    outlet = str((g.get("outlet") or {}).get("edge", "right"))
    for nm, e in (("inlet", inlet), ("outlet", outlet)):
        if e not in EDGES:
            raise SchemaError(f"$.geometry.{nm}.edge", f"{e!r} not one of {EDGES}")

    parametric = g.get("parametric")
    if parametric is not None and not isinstance(parametric, dict):
        raise SchemaError("$.geometry.parametric", "expected an object {kind, params}")

    geometry = Geometry(die=die, etch=etch, segments=segments, parametric=parametric,
                        keep_out=keep_out, inlet_edge=inlet, outlet_edge=outlet)
    if not geometry.has_channels():
        raise SchemaError("$.geometry", "no channels: provide etch, segments, or parametric")

    c = payload.get("conditions") or {}
    pw = c.get("power") or {}
    hotspots = []
    for i, h in enumerate(pw.get("hotspots") or []):
        p = f"$.conditions.power.hotspots[{i}]"
        hotspots.append(Hotspot(
            cx=_num(h, "cx", None, p, lo=0.0, hi=1.0), cy=_num(h, "cy", None, p, lo=0.0, hi=1.0),
            sigma=_num(h, "sigma", None, p, lo=0.005, hi=1.0),
            peak_wcm2=_num(h, "peak_wcm2", None, p, lo=0.0, hi=100_000.0)))
    grid = decode_array(pw["grid"], "$.conditions.power.grid") if pw.get("grid") else None
    power = Power(background_wcm2=_num(pw, "background_wcm2", 20.0, "$.conditions.power",
                                       lo=0.0, hi=100_000.0),
                  hotspots=hotspots, grid=grid)
    conditions = Conditions(
        coolant=str(c.get("coolant", "water")),
        flow_mlmin=_num(c, "flow_mlmin", 200.0, "$.conditions", lo=0.1, hi=100_000.0),
        inlet_temp_c=_num(c, "inlet_temp_c", 25.0, "$.conditions", lo=-80.0, hi=300.0),
        junction_limit_c=_num(c, "junction_limit_c", 105.0, "$.conditions", lo=0.0, hi=500.0),
        power=power,
    )

    s = payload.get("solve") or {}
    engine = str(s.get("engine", "auto"))
    if engine not in ENGINES:
        raise SchemaError("$.solve.engine", f"{engine!r} not one of {ENGINES}")
    resolution = str(s.get("resolution", "draft"))
    if resolution not in RESOLUTIONS:
        raise SchemaError("$.solve.resolution", f"{resolution!r} not one of {tuple(RESOLUTIONS)}")
    returns = tuple(s.get("return", ("tj_map", "segment_flow")))
    solve = SolveOptions(engine=engine, resolution=resolution, returns=returns,
                         timeout_s=_num(s, "timeout_s", 30.0, "$.solve", lo=1.0, hi=600.0))

    return SolveRequest(geometry=geometry, conditions=conditions, solve=solve)


# ── responses ────────────────────────────────────────────────────────────

def make_response(*, engine: str, metrics: dict, confidence: dict,
                  manufacturability: dict | None = None, maps: dict | None = None,
                  segments: list | None = None, fields: dict | None = None,
                  meta: dict | None = None) -> dict:
    """Assemble a well-formed v1 response. `maps` values may be numpy arrays —
    they are encoded automatically."""
    enc_maps = {}
    for k, v in (maps or {}).items():
        enc_maps[k] = encode_array(v) if isinstance(v, np.ndarray) else v
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "metrics": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                        float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                    for k, v in metrics.items()},
        "confidence": confidence,
        "manufacturability": manufacturability or {},
        "maps": enc_maps,
        "segments": segments or [],
        "fields": fields,
        "meta": meta or {},
    }


# ── examples (used by tests and as living documentation) ────────────────

def example_request() -> dict:
    """A parametric straight-array die at 200 mL/min with one hotspot."""
    return {
        "schema_version": SCHEMA_VERSION,
        "geometry": {
            "die": {"L": 10000, "W": 10000, "thk": 500, "base_thk": 100, "lid": True},
            "parametric": {"kind": "straight_parallel",
                           "params": {"n_ch": 50, "w_um": 100, "H_um": 300}},
            "inlet": {"edge": "left"}, "outlet": {"edge": "right"},
        },
        "conditions": {
            "coolant": "water", "flow_mlmin": 300.0, "inlet_temp_c": 25.0,
            "power": {"background_wcm2": 50.0,
                      "hotspots": [{"cx": 0.7, "cy": 0.5, "sigma": 0.08, "peak_wcm2": 400.0}]},
        },
        "solve": {"engine": "auto", "resolution": "draft",
                  "return": ["tj_map", "segment_flow"]},
    }
