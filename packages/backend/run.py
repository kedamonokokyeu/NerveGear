"""
run.py

Starts the gesture backend: opens the webcam, runs hand tracking, and serves the
control WebSocket.

    python run.py                 # camera on, default port 8200
    python run.py --no-camera     # browser-fed (/ws/landmarks) only
    python run.py --port 9000

Point the frontend's WebSocket at ws://localhost:8200/ws/control.
"""

from __future__ import annotations

import argparse

import uvicorn

from gestures.config import DEFAULT
from server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="backend gesture control backend")
    parser.add_argument("--port", type=int, default=DEFAULT.server.port)
    parser.add_argument("--host", default=DEFAULT.server.host)
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not open the local webcam; serve only the browser /ws/landmarks path.",
    )
    parser.add_argument(
        "--camera-index", type=int, default=DEFAULT.camera.device_index,
        help="Which webcam to use (0 is usually the built-in).",
    )
    args = parser.parse_args()

    cfg = DEFAULT
    cfg.server.port = args.port
    cfg.server.host = args.host
    cfg.camera.device_index = args.camera_index

    app = create_app(cfg, enable_camera=not args.no_camera)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
