"""
tools/benchmark.py

Measures the gesture engine's per-frame latency. The engine's own cost should be
a small fraction of the ~33 ms frame budget at 30 fps.

    python -m tools.benchmark
"""

from __future__ import annotations

import time

from gestures.engine import GestureEngine
from gestures.landmarks import Frame
from tools import synthetic as S


def run(n: int = 5000) -> None:
    eng = GestureEngine()
    # a realistic mix: a moving, twisting open hand
    frames = [Frame(hands=[S.rotate_in_plane(S.translate(S.open_palm(), 0.001 * i, 0.0), 0.005 * i)],
                    timestamp=i / 30.0) for i in range(n)]
    # warm up
    for f in frames[:100]:
        eng.update(f)

    t0 = time.perf_counter()
    for f in frames:
        eng.update(f)
    dt = time.perf_counter() - t0

    per_us = dt / n * 1e6
    print(f"engine.update: {n} frames in {dt * 1000:.1f} ms")
    print(f"  -> {per_us:.1f} µs/frame  ({1.0 / (dt / n):,.0f} updates/sec)")
    print(f"  -> {per_us / 1000 / 33.3 * 100:.2f}% of a 33 ms (30fps) frame budget")


if __name__ == "__main__":
    run()
