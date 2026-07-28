"""
samples/make_ai_accelerator_glb.py

Generates ai-accelerator-package.glb — a realistic, ~100-part data-center
accelerator package for exercising the Phase 1 viewer. Not a toy cube: it
mirrors the construction of a CoWoS-class part (the kind of chip on-die
microfluidic cooling actually targets), bottom to top:

  BGA ball grid -> organic substrate -> C4 solder -> silicon interposer ->
  micro-bumps -> 2 compute chiplets + 6 HBM stacks (base die + 8 DRAM each) ->
  MLCC capacitor banks + Cu stiffener ring -> TIM pads ->
  an etched-silicon MICROCHANNEL COLD PLATE (14 channels + 2 manifolds)
  under a translucent lid — slice it with the Section tool to see the flow
  path; the channels light up coolant-blue.

Every solid is named so the viewer's label heuristics (materials.ts) resolve
the right material with zero manual overrides. Dimensions are millimetres,
exported in glTF-spec metres; the viewer converts to µm at import.

Run:  python make_ai_accelerator_glb.py   (needs: pip install trimesh numpy)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import box, icosphere

MM = 1e-3  # model in metres (glTF spec); all sizes below are written in mm

OUTPUT_PATH = Path(__file__).parent / "ai-accelerator-package.glb"

# One flat list of (name, mesh); nothing is merged unless it is genuinely one
# manufactured thing (the BGA grid), so the Structure pane mirrors the BOM.
parts: list[tuple[str, trimesh.Trimesh]] = []


def add_box(name: str, size_mm, center_mm) -> None:
    solid = box(extents=[s * MM for s in size_mm])
    solid.apply_translation([c * MM for c in center_mm])
    parts.append((name, solid))


# ── substrate: 55 x 55 organic laminate, top face at y = 0 ─────────────────
SUBSTRATE = dict(w=55.0, t=1.4)
add_box("substrate", (SUBSTRATE["w"], SUBSTRATE["t"], SUBSTRATE["w"]),
        (0, -SUBSTRATE["t"] / 2, 0))

# ── BGA: one merged grid (a single manufactured array, so a single part) ───
BGA = dict(pitch=2.2, rows=23, radius=0.30)
ball_template = icosphere(subdivisions=1, radius=BGA["radius"] * MM)
balls = []
half = (BGA["rows"] - 1) / 2
for ix in range(BGA["rows"]):
    for iz in range(BGA["rows"]):
        ball = ball_template.copy()
        ball.apply_translation((
            (ix - half) * BGA["pitch"] * MM,
            (-SUBSTRATE["t"] - BGA["radius"]) * MM,
            (iz - half) * BGA["pitch"] * MM,
        ))
        balls.append(ball)
parts.append(("bga-ball-grid", trimesh.util.concatenate(balls)))

# ── C4 solder + silicon interposer ─────────────────────────────────────────
INTERPOSER = dict(w=46.0, d=42.0, t=0.15)
add_box("c4-solder-layer", (INTERPOSER["w"] - 1, 0.10, INTERPOSER["d"] - 1), (0, 0.05, 0))
add_box("interposer", (INTERPOSER["w"], INTERPOSER["t"], INTERPOSER["d"]),
        (0, 0.10 + INTERPOSER["t"] / 2, 0))
INTERPOSER_TOP = 0.10 + INTERPOSER["t"]

UBUMP_T = 0.05  # micro-bump layer thickness under every die


def add_die_on_interposer(name: str, w: float, d: float, t: float,
                          x: float, z: float) -> float:
    """Micro-bump sheet + die at (x, z); returns the die's top y (mm)."""
    add_box(f"{name}-microbumps", (w, UBUMP_T, d), (x, INTERPOSER_TOP + UBUMP_T / 2, z))
    base = INTERPOSER_TOP + UBUMP_T
    add_box(name, (w, t, d), (x, base + t / 2, z))
    return base + t


# ── 2 compute chiplets, side by side ───────────────────────────────────────
COMPUTE = dict(w=13.0, d=24.0, t=0.75, gap=1.0)
compute_top = 0.0
for i, sx in enumerate((-1, 1)):
    x = sx * (COMPUTE["w"] + COMPUTE["gap"]) / 2
    compute_top = add_die_on_interposer(f"compute-die-{i}", COMPUTE["w"], COMPUTE["d"],
                                        COMPUTE["t"], x, 0)

# ── 6 HBM stacks (3 per side): buffer die + 8 thinned DRAM dies ────────────
HBM = dict(w=11.0, d=10.0, base_t=0.14, dram_t=0.07, glue=0.015, x=19.0, zs=(-14.0, 0.0, 14.0))
for n, (sx, z) in enumerate((sx, z) for sx in (-1, 1) for z in HBM["zs"]):
    x = sx * HBM["x"]
    y = add_die_on_interposer(f"hbm{n}-base-die", HBM["w"], HBM["d"], HBM["base_t"], x, z)
    for level in range(8):
        add_box(f"hbm{n}-dram-{level + 1}", (HBM["w"], HBM["dram_t"], HBM["d"]),
                (x, y + HBM["dram_t"] / 2, z))
        y += HBM["dram_t"] + HBM["glue"]

# ── MLCC capacitor banks on the substrate (one bank per edge) ──────────────
MLCC = dict(w=1.6, h=0.6, d=0.9, count=17, pitch=2.4, edge=25.5)
for bank, (axis, sign) in (("north", ("z", -1)), ("south", ("z", 1)),
                           ("west", ("x", -1)), ("east", ("x", 1))):
    caps = []
    for i in range(MLCC["count"]):
        offset = (i - (MLCC["count"] - 1) / 2) * MLCC["pitch"]
        cap = box(extents=(MLCC["w"] * MM, MLCC["h"] * MM, MLCC["d"] * MM))
        if axis == "z":
            cap.apply_translation((offset * MM, MLCC["h"] / 2 * MM, sign * MLCC["edge"] * MM))
        else:
            cap.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
            cap.apply_translation((sign * MLCC["edge"] * MM, MLCC["h"] / 2 * MM, offset * MM))
        caps.append(cap)
    parts.append((f"mlcc-bank-{bank}", trimesh.util.concatenate(caps)))

# ── Cu stiffener ring: four walls around the interposer ────────────────────
RING = dict(outer=53.0, wall=3.0, h=1.25)
for side, (w, d, x, z) in {
    "north": (RING["outer"], RING["wall"], 0, -(RING["outer"] - RING["wall"]) / 2),
    "south": (RING["outer"], RING["wall"], 0, (RING["outer"] - RING["wall"]) / 2),
    "west": (RING["wall"], RING["outer"] - 2 * RING["wall"], -(RING["outer"] - RING["wall"]) / 2, 0),
    "east": (RING["wall"], RING["outer"] - 2 * RING["wall"], (RING["outer"] - RING["wall"]) / 2, 0),
}.items():
    add_box(f"stiffener-{side}", (w, RING["h"], d), (x, RING["h"] / 2, z))

# ── TIM pads: one per compute die (HBM runs cooler; thermal spec is the die)─
TIM_T = 0.10
for i, sx in enumerate((-1, 1)):
    x = sx * (COMPUTE["w"] + COMPUTE["gap"]) / 2
    add_box(f"tim-compute-{i}", (COMPUTE["w"], TIM_T, COMPUTE["d"]),
            (x, compute_top + TIM_T / 2, 0))

# ── the microchannel cold plate: THE reason this tool exists ───────────────
# Etched-silicon plate over the whole assembly. Channels run along +X (the
# canonical flow axis), fed by inlet/outlet manifolds across ±X. The lid is
# translucent in the viewer, so the channel field reads even without Section.
PLATE = dict(w=50.0, d=50.0, body_t=0.95)
CHANNEL = dict(n=14, w=1.2, h=0.8, length=42.0, pitch=3.0)
MANIFOLD = dict(w=2.6, h=0.8)
LID_T = 0.6

plate_base = RING["h"]                       # rests on the stiffener ring
channel_base = plate_base + PLATE["body_t"]  # channels etched into the top
lid_base = channel_base + CHANNEL["h"]

add_box("cold-plate-si", (PLATE["w"], PLATE["body_t"], PLATE["d"]),
        (0, plate_base + PLATE["body_t"] / 2, 0))
for i in range(CHANNEL["n"]):
    z = (i - (CHANNEL["n"] - 1) / 2) * CHANNEL["pitch"]
    add_box(f"coolant-channel-{i + 1:02d}", (CHANNEL["length"], CHANNEL["h"], CHANNEL["w"]),
            (0, channel_base + CHANNEL["h"] / 2, z))
span_z = (CHANNEL["n"] - 1) * CHANNEL["pitch"] + CHANNEL["w"]
for name, sx in (("coolant-inlet-manifold", -1), ("coolant-outlet-manifold", 1)):
    add_box(name, (MANIFOLD["w"], MANIFOLD["h"], span_z),
            (sx * (CHANNEL["length"] + MANIFOLD["w"]) / 2, channel_base + MANIFOLD["h"] / 2, 0))
add_box("lid", (PLATE["w"], LID_T, PLATE["d"]), (0, lid_base + LID_T / 2, 0))

# ── export ─────────────────────────────────────────────────────────────────
scene = trimesh.Scene()
for name, mesh in parts:
    scene.add_geometry(mesh, node_name=name, geom_name=name)
scene.export(OUTPUT_PATH)
print(f"wrote {OUTPUT_PATH.name}: {len(parts)} parts, "
      f"{OUTPUT_PATH.stat().st_size / 1024:.0f} KiB")
