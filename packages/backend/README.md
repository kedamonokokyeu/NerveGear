# backend — Gesture Workspace

An interactive workspace for NerveGear's spatial-sculpting prototype (roadmap Phase 2:
"manipulate the simulation live by using hand movements"). Three connected pieces:

1. **Gesture control** — wave at a webcam; the backend streams transform deltas
   (rotate / zoom / pan) to a 3D viewer over a WebSocket. The Musk/SpaceX
   Leap-Motion demo, done with a plain webcam via MediaPipe.
2. **CAD ingestion** — import a real part (.stl/.obj); the viewer renders it and the
   backend voxelizes it into the 128×128 obstacle mask the CFD model consumes.
3. **RAG knowledge pipeline** — a documented retrieval system over the NerveGear
   roadmap + a CFD glossary (see `rag/README.md`).

The reserved `flow` channel is where fluid-flow control (direction, speed, density)
plugs in later, feeding the existing NerveGear CFD `/predict` model — the seam is
already cut (`nervegear_gestures/cfd_bridge.py`, `cad/` → mask).

---

## Architecture: why Python-backend-with-local-webcam (and how it shaped the code)

There were two real options for where hand tracking runs:

**A. Python backend opens the webcam (chosen).** OpenCV grabs frames, MediaPipe
finds 21 hand landmarks, the engine turns them into control deltas, and a
WebSocket streams those to the frontend.

**B. Browser runs MediaPipe.js**, sends landmarks to a thin backend that only does
mapping + physics.

**For a real, multi-user web product, B is optimal** — the webcam is native to the
browser, detection runs on the client GPU, you never ship video frames over the
network, and the server stays stateless and scalable.

**For what NerveGear is *right now* — a local research prototype tightly coupled to a
PyTorch model — A is optimal, and that's what this is.** One language end-to-end,
direct in-process access to the CFD model when physics lands, no WASM model-loading
friction, and trivial debugging because the whole loop is Python you can set a
breakpoint in. This matches the demo use case (one machine, one camera, lowest
possible latency in a tight loop).

**The key design decision that falls out of this:** don't let "it's Python with a
webcam" leak into the logic. So the codebase is split hard along an I/O boundary:

- `nervegear_gestures/` (minus `camera.py`) is **pure, stateless-where-possible, and
  imports neither OpenCV nor MediaPipe.** It speaks only "landmarks in, commands
  out." This is the entire brain.
- `camera.py` is the **only** module that touches the webcam and MediaPipe, and it
  imports them lazily.
- The input contract is **MediaPipe's normalized 21-landmark format** — which is
  byte-identical between MediaPipe Python and MediaPipe.js.

The payoff: moving to architecture **B** later (for Phase-2 web/XR deployment) is
*not a rewrite*. The browser just sends landmarks to the already-built
`/ws/landmarks` endpoint, and the same engine produces the same commands. The
`/ws/landmarks` path is already wired and tested today. You get the optimal
prototype now and a clean migration path to the optimal product later, for free.

A second consequence: **the protocol is deltas, not absolute state.** The backend
never tracks the scene's camera angle or zoom — it just says "rotate a bit more",
"zoom a touch". The frontend owns the authoritative transform. That keeps the
backend stateless about the 3D scene, makes a dropped frame a non-event, and means
the same command stream works whether CV runs server-side or in the browser.

---

## The gesture vocabulary

| Gesture (one hand)        | Mode          | Effect                                            |
|---------------------------|---------------|---------------------------------------------------|
| ✋ Open palm              | `rotate`      | **trackball**: move hand → spin (L/R = turn, up/down = tilt forward/back); twist palm → **roll**. No zoom here. |
| ✊ Fist                   | `hold`        | **brake** — freezes the view ("close to stop"), and acts as a clutch |
| 🤏 Pinch (thumb+index)    | `pan`         | grab & drag → **pan**                             |
| 👉 Point (index only)     | `point`       | reserved → sets future **flow direction**         |

| Two hands                 | Mode          | Effect                                            |
|---------------------------|---------------|---------------------------------------------------|
| ✋ ✋ both open            | `two_hand`    | spread/close → **zoom**; turn like a wheel → **roll**; move midpoint → **pan** |
| ✊ + anything             | `hold`        | **brake**                                         |

**The mental model:** *one hand* = the object is a ball under your hand (a
trackball). Slide your hand to spin it — left/right turns it (yaw), up/down tips
it forward/back (pitch) — and twist your palm to roll it. Make a fist to "let go"
(freeze), reposition your hand, open again to keep turning — that's the clutch.
*Two hands* = you're framing the object between your palms: spread/squeeze to
zoom, turn both like a steering wheel to roll, move the pair to pan. Rotation
**eases** smoothly toward where you point it (no jitter) and coasts briefly when
you release. Zoom is deliberately ONLY on two hands so a one-hand rotate can
never accidentally zoom.

All thresholds, sensitivities (gains), dead-zones, and clamps live in
`nervegear_gestures/config.py` — tune feel there without touching logic.

---

## Files, top to bottom

**The engine (pure, testable, no CV deps):**

- `nervegear_gestures/landmarks.py` — the data contract. `Lm` names the 21 landmark
  indices; `HandLandmarks`/`Frame` carry one hand / one frame. `from_normalized_list`
  builds a Frame from JSON — this is the browser path's entry point.
- `nervegear_gestures/features.py` — pure geometry. Turns 21 points into normalized
  scalars: hand openness, pinch distance, palm roll angle, palm normal, hand
  "scale" (apparent size = zoom signal), centroid. Everything normalized by hand
  size so it's invariant to camera distance.
- `nervegear_gestures/gestures.py` — threshold-driven classifier: features → one of
  {fist, open_palm, pinch, point, none}. Stateless.
- `nervegear_gestures/smoothing.py` — the One-Euro filter (the standard for low-latency
  hand input: smooths jitter when still, stays responsive when fast), an
  angle-aware variant, and a `Debouncer` so the mode doesn't flicker.
- `nervegear_gestures/commands.py` — the `ControlCommand` emitted every frame, and its
  reserved `flow` block for future physics. Plain JSON.
- `nervegear_gestures/engine.py` — **the brain.** A debounced mode state machine that,
  per frame, smooths the relevant signals, computes deltas against a baseline,
  applies dead-zones and clamps, and resets cleanly on every gesture change so the
  model never lurches when you switch gestures.
- `nervegear_gestures/config.py` — every tunable number, grouped and commented.
- `nervegear_gestures/cfd_bridge.py` — **FUTURE HOOK (not wired in).** Pure mapping from
  a flow gesture to the existing CFD `/predict` params (Re, speed, angle,
  inlet_edge). Built and tested now so the integration is visible and type-checked.

**I/O layer:**

- `nervegear_gestures/camera.py` — the only OpenCV/MediaPipe module (lazy imports).
  Yields Frames from the webcam; supports both the legacy `solutions.hands` API
  (default, no model download) and the newer Tasks `HandLandmarker`.
- `server/app.py` — FastAPI. `GET /health`, `WS /ws/control` (frontend subscribes
  here; also accepts `{action:"toggle_rotation"}`), `WS /ws/landmarks` (browser
  path → same engine), `POST /cad/upload` (mesh → CFD mask), `POST /rag/query`
  (grounded retrieval). A `Hub` fan-outs the latest command, decoupled from camera FPS.
- `server/runner.py` — runs the blocking webcam loop on a background thread and
  publishes commands into the Hub.
- `run.py` — entrypoint. `python run.py` (camera on) / `--no-camera` (browser-fed).

**CAD ingestion (`cad/`):**

- `cad/ingest.py` — load a mesh (STL/OBJ/PLY/GLB via trimesh), center it on the
  origin and uniformly scale so its largest dimension is 1.0 (maps any part, any
  units, onto the model's unit domain), and summarize it (`MeshInfo`).
- `cad/voxelize.py` — `mesh_to_mask` projects/voxelizes the normalized mesh into the
  128×128, 1=solid mask the CFD `/predict` endpoint expects, centered and clear of
  the domain edges (matching how training shapes were generated).

**RAG pipeline (`rag/`):** chunk → embed → store → retrieve → cited prompt. Runs
offline via a hashing embedder; swaps to real semantic embeddings if installed.
Full walkthrough in `rag/README.md`.

**Tooling & tests:**

- `tools/synthetic.py` — hand-built hand poses + transforms (rotate/scale/translate)
  so the whole pipeline is testable with zero hardware.
- `tools/replay.py` — drives the real engine with scripted poses and prints the
  commands. `python -m tools.replay`. Great for tuning gains.
- `tests/` — 57 tests across the gesture engine, CAD ingestion, and the RAG pipeline.

---

## Running it

```bash
cd backend
pip install -r requirements.txt          # core + server + (heavy) opencv/mediapipe + trimesh
python run.py                            # opens webcam, serves http/ws on :8200
```

**Then open `web/viewer.html`** in a browser. It's a clean engineering workspace —
neutral graphite UI, model/session panels, an axis triad, and a live FPS/transform
readout. It connects to the control WebSocket; the part rotates/zooms/pans as you
gesture, eased and stable. **Drag in an .stl/.obj** (or click *Import CAD*) to load
a real part — the viewer renders it and the backend voxelizes it into the CFD mask
(shown as solid-fraction in the Model panel). Mouse-drag rotates as a no-camera
fallback; **T** toggles tilt/move input; **R** resets.

(Port 8200 avoids the CFD `/predict` server on 8000.)

**HTTP/WS endpoints:**
- `WS /ws/control` — subscribe to control commands; send `{action:"toggle_rotation"}` / `{action:"set_rotation","mode":"tilt|move"}`.
- `WS /ws/landmarks` — browser MediaPipe.js path → same engine.
- `POST /cad/upload` (multipart `file`) — returns mesh info + the 128² CFD mask + summary.
- `POST /cfd/predict` — `{mask, N, Re, speed, angle, inlet_edge}` → proxies to the NerveGear
  CFD `/predict` server and returns the velocity/pressure field (closes the loop).
- `POST /rag/query` — `{question, k}` → grounded citations + assembled prompt.
- `POST /cfd/metrics` · `POST /cfd/streamlines` · `POST /cfd/compare` — Tier-A/Tier-B physics post-processing (see `PIVOT.md`).
- `POST /thermal/solve` — geometry + heat sources + boundary conditions → temperature field + cooling metrics (see `THERMAL.md`).
- `GET /health` — status.

Browser-fed (no local webcam): `python run.py --no-camera`.

**No hardware, just want to see it work:** `python -m tools.replay`, `python -m tools.physics_demo`, `python -m tools.thermal_demo`, and `pytest`.

See **`PIVOT.md`** (interactive-physics workspace scaffolding) and **`THERMAL.md`** (the cooling-design pivot) for the post-gesture direction.

---

## The path to physics (later, by design)

When you turn physics on:
1. `engine.py` populates `ControlCommand.flow.speed` / `.density` from gestures
   (e.g. a push for faster inlet flow, a squeeze for higher density). Today those
   are `None`; `flow.direction` and `inlet_edge` are already produced by a Point.
2. `cfd_bridge.flow_to_predict_params(flow)` converts that into the `/predict`
   request the NerveGear CFD model already accepts.
3. The frontend POSTs CAD shape (mask) + those params to `/predict` and renders the
   returned velocity/pressure field — live, as you sculpt.

Nothing in the gesture protocol changes. The seam is already cut.
