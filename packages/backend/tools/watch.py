"""
tools/watch.py

Subscribes to the control WebSocket and prints the live command stream.

    python run.py           # terminal 1
    python -m tools.watch   # terminal 2

Wave at the webcam to see mode and deltas update in place.
"""

from __future__ import annotations

import asyncio
import json

import websockets

URL = "ws://localhost:8200/ws/control"


async def main() -> None:
    print(f"connecting to {URL} ... (Ctrl+C to quit)")
    async with websockets.connect(URL) as ws:
        async for raw in ws:
            c = json.loads(raw)
            r = c["rotate"]
            line = (
                f"mode={c['mode']:<11} frozen={int(c['frozen'])} hands={c['num_hands']} "
                f"roll={r['roll']:+.3f} yaw={r['yaw']:+.3f} pitch={r['pitch']:+.3f} "
                f"zoom={c['zoom']:+.3f} pan=({c['pan']['x']:+.2f},{c['pan']['y']:+.2f}) "
                f"scale={c['scale']:+.3f}"
            )
            print(line.ljust(110), end="\r", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
