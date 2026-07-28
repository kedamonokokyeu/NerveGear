"""
server/runner.py

Runs the webcam capture loop on a background thread.

OpenCV's read() is blocking so it can't run on the asyncio event loop. The thread
grabs frames, runs engine.update(), and pushes results into the Hub. The Hub's
async broadcaster fans commands out to WebSocket clients at a fixed rate, decoupled
from the camera's frame rate. Camera and MediaPipe imports happen only when the
thread starts, so a landmarks-only deployment never needs those packages.
"""

from __future__ import annotations

import threading
import traceback

from gestures.config import Config
from gestures.engine import GestureEngine


class CaptureRunner:
    """Background webcam -> engine -> Hub pump."""

    def __init__(self, cfg: Config, hub):
        self.cfg = cfg
        self.hub = hub
        self.engine = GestureEngine(cfg)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        # Imported here so the heavy CV deps load only when capture is enabled.
        from gestures.camera import HandTracker

        try:
            with HandTracker(self.cfg.camera) as tracker:
                print(f"[capture] webcam open (backend={tracker._backend}). streaming...")
                for frame in tracker.frames():
                    if self._stop.is_set():
                        break
                    cmd = self.engine.update(frame)
                    self.hub.publish(cmd.to_dict())
        except Exception as e:  # noqa: BLE001 - surface any capture failure to /health
            self.last_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            print(
                "[capture] camera loop stopped. The server keeps running so the "
                "browser-fed /ws/landmarks path still works."
            )
