"""
tools/replay.py

Offline driver for the gesture engine. Synthesizes a scripted hand sequence,
pushes it through a real GestureEngine, and prints the resulting ControlCommands.
Good for sanity-checking behavior and tuning gains without any hardware.

    python -m tools.replay
"""

from __future__ import annotations

from gestures.engine import GestureEngine
from gestures.landmarks import Frame
from tools import synthetic as S


def _print(tag: str, cmd) -> None:
    r = cmd.rotate
    print(
        f"{tag:>16} | mode={cmd.mode:<11} frozen={int(cmd.frozen)} "
        f"roll={r.roll:+.4f} yaw={r.yaw:+.4f} pitch={r.pitch:+.4f} "
        f"zoom={cmd.zoom:+.4f} pan=({cmd.pan.x:+.3f},{cmd.pan.y:+.3f}) "
        f"scale={cmd.scale:+.4f}"
    )


def scene(engine: GestureEngine, name: str, hands_per_frame, t0: float, dt: float = 1 / 30):
    print(f"\n--- {name} ---")
    t = t0
    for i, hands in enumerate(hands_per_frame):
        cmd = engine.update(Frame(hands=hands, timestamp=t))
        _print(f"f{i}", cmd)
        t += dt
    return t


def main() -> None:
    engine = GestureEngine()
    t = 0.0

    # Scene 1: open palm slowly twisting -> rotation (roll) should ramp up.
    twist = [[S.rotate_in_plane(S.open_palm(), 0.04 * i)] for i in range(8)]
    t = scene(engine, "open palm twist (rotate roll)", twist, t)

    # Scene 2: open palm growing (moving toward camera) -> zoom-in (positive).
    push = [[S.scale_about(S.open_palm(), 1.0 + 0.05 * i)] for i in range(8)]
    t = scene(engine, "open palm push (zoom in)", push, t)

    # Scene 3: fist -> HOLD/brake, all deltas zero, frozen=1.
    brake = [[S.fist()] for _ in range(6)]
    t = scene(engine, "fist (brake / hold)", brake, t)


if __name__ == "__main__":
    main()
