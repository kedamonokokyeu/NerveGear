"""
camera.py

Module for OpenCV and MediaPipe. 
Produces stream of 'Frame' objects in landmark contract defined in landmarks.py
so that it can be sent for interpretation downstream.

Lazy imports to cut startup times.
"""

from __future__ import annotations
import os
import time
import urllib.request
from typing import Iterator
import numpy as np
from .config import CameraConfig
from .landmarks import Frame, HandLandmarks

# URL for MediaPipe hand-tracking model
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _default_model_path() -> str:
    """backend/models/hand_landmarker.task, resolved relative to this file."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__)) # .../gestures
    root = os.path.dirname(pkg_dir)  # .../backend
    return os.path.join(root, "models", "hand_landmarker.task")


def _ensure_model(path: str) -> str:
    """Download the HandLandmarker model to `path` if it isn't there yet. Otherwise, download from Google Cloud."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[camera] downloading hand-tracking model -> {path}")
    urllib.request.urlretrieve(_MODEL_URL, path)
    print(f"[camera] model ready ({os.path.getsize(path) // 1024} KB)")
    return path


class HandTracker:
    """
    Open a webcam and yield Frames. Use as a context manager:

        with HandTracker(cfg.camera) as tracker:
            for frame in tracker.frames():
                cmd = engine.update(frame)
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cv2 = None
        self._cap = None # camera handle that allows for frame reading
        self._backend = None      
        self._detector = None     # the MediaPipe object
        self._mp = None
        self._last_ts_ms = 0    # clock for MediaPipe VIDEO mode

        self._init_capture()
        self._init_detector()

    # capture setup
    def _init_capture(self) -> None:
        import cv2  # lazy
        self._cv2 = cv2
        
        # camera handle --- allows frame reading one-by-one
        cap = cv2.VideoCapture(self.cfg.device_index)

        # configure frame (width + height), and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.target_fps)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at device_index={self.cfg.device_index}. "
                "Check the camera is connected and not in use by another app."
            )
        self._cap = cap

    def _init_detector(self) -> None:
        import mediapipe as mp  # lazy
        self._mp = mp

        # Tasks API = MediaPipe's new interface for hand-tracking
            # for scaling, docker image should contain such dependency
            # if not, use solutions API, older version
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception:  # very old build without the Tasks API
            mp_python = vision = None

        if vision is not None: # only for successful Task API import
            model_path = _ensure_model(self.cfg.model_asset_path or _default_model_path())
            base = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base,
                num_hands=self.cfg.max_hands, # 2 max
                min_hand_detection_confidence=self.cfg.min_detection_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
                running_mode=vision.RunningMode.VIDEO,
            )
            self._detector = vision.HandLandmarker.create_from_options(options) # create hand-tracking object
            self._backend = "tasks" # Tasks backend, different detection methods
            return

        # solutions API (fallback for older installations)
        if hasattr(mp, "solutions"):
            self._detector = mp.solutions.hands.Hands(
                max_num_hands=self.cfg.max_hands,
                min_detection_confidence=self.cfg.min_detection_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
            )
            self._backend = "solutions"
            return

        raise RuntimeError(
            "This MediaPipe build exposes neither the Tasks API nor solutions.hands. "
            "Try: pip install -U mediapipe  (Python 3.9-3.12)."
        )

    # main loop
    def frames(self) -> Iterator[Frame]:
        """Generator yielding one Frame per processed webcam frame until the
        camera stops or the generator is closed."""
        cv2 = self._cv2
        while True:
            # ok = whether read succeeded, bgr = image data in color order
            # OpenCV PULLS IMAGE, CONVERTS TO RGB FOR MediaPipe NEURAL NETWORK
            ok, bgr = self._cap.read()
            if not ok:
                break
            if self.cfg.mirror:
                # for selfie moe, flip image horizontally
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # convert BGR format to RGB --- MediaPipe expects RGP input
            ts = time.monotonic()
            try:
                yield self._detect(rgb, ts)
            except Exception as e:  # noqa: BLE001
                # if frame missed, catch exception, output empty frame (engine treats as idle)
                print(f"[camera] detection hiccup, skipping frame: {type(e).__name__}: {e}")
                yield Frame(hands=[], timestamp=ts)

    def _detect(self, rgb: np.ndarray, ts: float) -> Frame:
        if self._backend == "tasks":
            return self._detect_tasks(rgb, ts)
        return self._detect_solutions(rgb, ts)

    def _detect_solutions(self, rgb: np.ndarray, ts: float) -> Frame:
        results = self._detector.process(rgb) # runs palm detector and landmark model neural network
        hands: list[HandLandmarks] = []
        if results.multi_hand_landmarks:
            handed = results.multi_handedness or []
            for i, lms in enumerate(results.multi_hand_landmarks):
                pts = np.array(
                    [[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float64
                )
                label, score = "Unknown", 1.0
                if i < len(handed) and handed[i].classification:
                    c = handed[i].classification[0]
                    label, score = c.label, float(c.score)
                hands.append(HandLandmarks(points=pts, handedness=label, score=score))
        return Frame(hands=hands, timestamp=ts) # one moment of camera input with timestamp and all detected hans with points, handedness, and confidence

    def _detect_tasks(self, rgb: np.ndarray, ts: float) -> Frame:
        mp = self._mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) # MediaPipe image object
        # millisecond timestamps that increase, any non-increase breaks the video frame stream
        self._last_ts_ms = max(int(ts * 1000), self._last_ts_ms + 1)
        result = self._detector.detect_for_video(image, self._last_ts_ms)
        hands: list[HandLandmarks] = []
        for i, lms in enumerate(result.hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            label, score = "Unknown", 1.0
            if i < len(result.handedness) and result.handedness[i]:
                c = result.handedness[i][0]
                label, score = c.category_name, float(c.score)
            hands.append(HandLandmarks(points=pts, handedness=label, score=score))
        return Frame(hands=hands, timestamp=ts)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._backend == "solutions" and self._detector is not None:
            self._detector.close()
        self._detector = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
