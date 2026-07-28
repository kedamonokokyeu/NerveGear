"""
server/app.py

The FastAPI app. It exposes three things to the frontend:

  GET  /health        -> JSON status (is the camera running? any error? client count)
  WS   /ws/control    -> SUBSCRIBE here. The server pushes the latest ControlCommand
                         as JSON at server.broadcast_hz. This is what the Three.js
                         viewer listens to and integrates into the model transform.
  WS   /ws/landmarks  -> OPTIONAL browser path. A MediaPipe.js client POSTs raw
                         landmarks here; the server runs them through the SAME
                         GestureEngine and both echoes the command back and
                         publishes it to /ws/control subscribers.

For reference, payload dictionary might look like:
payload = {
    "mask": [[0, 1, 1, 0], [1, 1, 0, 0]],       # geometry mask grid
    "N": 64,                                   # grid resolution
    "Re": 200.0,                               # Reynolds number
    "speed": 0.10,                             # inlet velocity
    "angle": 0.0,                              # flow direction angle
    "inlet_edge": "left",                      # inlet side
    "volume_m3": 0.002,                        # volume for metrics
    "density": 2700.0,                         # material density
    "field": {                                 # CFD output field
        "vx": [[...]],                         # x-velocity array
        "vy": [[...]],                         # y-velocity array
        "pressure": [[...]],                   # pressure array
        "N": 64
    }
}

Everyone gets same update instantly becaues of concurrency and broadcasting task.
This is most likely to change.
"""

from __future__ import annotations

import asyncio # for concurrency
import json
import time

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from gestures.commands import ControlCommand
from gestures.config import DEFAULT, Config
from gestures.engine import GestureEngine
from gestures.landmarks import from_normalized_list

from .runner import CaptureRunner # camera-capture thread manager


class Hub:
    """
    Mediates between the gesture producers that take in camera/browser landmarks,
    and the viewers who get to see the interaction. 
    
    When new gesture command made, gesture producer calls publish(), and the
    async broadcaster() coroutine runs in the background --- sends that latest command
    to all connected clients at specific rate.
    """

    def __init__(self, broadcast_hz: float): # broadcast_hz specifies how many times per second to send updates
        self.latest: dict = ControlCommand().to_dict()
        self.clients: set[WebSocket] = set() # WebSocket object connection represents new client connect --> latest command will be sent to ALL WebSocket connections
        self.interval = 1.0 / max(broadcast_hz, 1.0)

    def publish(self, cmd_dict: dict) -> None:
        """Called by gesture producer whenever new gesture command is ready."""
        # replaces self.latest with new command dictionary
        self.latest = cmd_dict

    async def broadcaster(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            if not self.clients:
                continue
            payload = json.dumps(self.latest) # convert command dict to JSON string
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(payload) # send string message from server to connected client
                except Exception:  # noqa: BLE001 - client vanished mid-send
                    dead.append(ws) # if send fails, that socket connection is dead
            for ws in dead:
                self.clients.discard(ws) # drop the dead client


def create_app(cfg: Config | None = None, *, enable_camera: bool = True) -> FastAPI:
    cfg = cfg or DEFAULT
    app = FastAPI(title="backend Gesture Control")
    import os as _os
    _origins = _os.environ.get(
        "NERVEGEAR_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8777,http://127.0.0.1:8777,"
        "http://localhost:8200,http://127.0.0.1:8200",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=(["*"] if _origins.strip() == "*" else
                       [o.strip() for o in _origins.split(",") if o.strip()]),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # reject oversized bodies before reading them (CAD uploads are the largest
    # legitimate payload; 40 MB covers real STLs comfortably)
    MAX_BODY = 40 * 1024 * 1024

    @app.middleware("http")
    async def _body_size_guard(request, call_next):
        from starlette.responses import JSONResponse
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    # the design/physics API (axiom/solve.v1) — see server/design_api.py
    from .design_api import router as design_router
    app.include_router(design_router)

    # CAD file conversion (STEP -> GLB for the Studio) — see server/cad_api.py
    from .cad_api import router as cad_router
    app.include_router(cad_router)

    hub = Hub(cfg.server.broadcast_hz)
    landmark_engine = GestureEngine(cfg)
    if enable_camera:
        runner: CaptureRunner | None = CaptureRunner(cfg, hub)
    else:
        runner = None

    # storing these objects into app.state so that any WebSocket handler can access it
    app.state.hub = hub
    app.state.cfg = cfg
    app.state.runner = runner

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.broadcaster_task = asyncio.create_task(hub.broadcaster()) # task object for background job
        if runner is not None:
            runner.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if runner is not None:
            runner.stop() 

        # look inside app.state, get broadcaster_task attribute if exists, otherwise None
        task = getattr(app.state, "broadcaster_task", None)
        if task: 
            task.cancel()

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "camera_enabled": runner is not None,
            "camera_error": runner.last_error if runner else None,
            "control_clients": len(hub.clients),
            "broadcast_hz": round(1.0 / hub.interval, 1),
        }

    # WEBSOCKET ENDPOINTS

    @app.websocket("/ws/control")
    async def control(ws: WebSocket) -> None:
        """
        Browsers connect here to receie gesture commands.
        The continuous datastream comes from the .receive_text() FastAPI WebSocket
        method that continuously waits for the next message from the browser. Extract 
        the 'action' from the JSON, and that determines how the object interacts.
        """
        await ws.accept()
        hub.clients.add(ws) # add new user/client
        try:
            # Control subscribers mostly just receive, but they may send small
            # control messages (e.g. toggle the rotation mode from the viewer).
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                action = data.get("action")
                if action in ("toggle_rotation", "set_rotation"):
                    # runer.engine = live webcam input, landmark_engine is for pre-computed hand coordinates
                    eng = runner.engine if runner else landmark_engine # use the corresponding engine's actions
                    if action == "set_rotation":
                        new_mode = data.get("mode", "tilt")
                    else:
                        new_mode = "move" if eng.rotation_mode == "tilt" else "tilt"
                    if runner:
                        runner.engine.set_rotation_mode(new_mode)
                    landmark_engine.set_rotation_mode(new_mode)
                    # confirm back to the sender
                    await ws.send_text(json.dumps({"info": "rotation_mode", "mode": new_mode}))
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    @app.websocket("/ws/landmarks")
    async def landmarks(ws: WebSocket) -> None:
        """Browser path: receive {"hands": [...], "timestamp": <sec>} per frame."""
        await ws.accept()
        try:
            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)
                ts = float(data.get("timestamp", time.monotonic()))
                # collect all hans, wrap each hand's data into HandLandmarks object, put into Frame object
                frame = from_normalized_list(data.get("hands", []), ts)
                cmd = landmark_engine.update(frame).to_dict()
                # frame into gesture engine
                hub.publish(cmd)            # share with /ws/control subscribers
                await ws.send_text(json.dumps(cmd))  # and echo to the sender
        except WebSocketDisconnect:
            pass

    # CAD SECTION
    CAD_EXTENSIONS = {".stl", ".obj", ".ply", ".glb", ".gltf", ".off"}

    @app.post("/cad/upload")
    async def cad_upload(file: UploadFile = File(...), mode: str = "mask") -> dict:
        """CAD mesh in, physics-ready representation out.

        mode="mask"      -> 128x128 solid mask (the 2D surrogate's input)
        mode="depthmap"  -> normalized etch depth map (the Studio's etch canvas
                            imports this to let users ALTER existing CAD parts)

        Hardening: extension whitelist, 40 MB cap (also enforced by middleware),
        temp file always unlinked.
        """
        import os
        import tempfile
        from cad import ingest, mesh_to_mask
        from cad.voxelize import mask_summary

        suffix = os.path.splitext(file.filename or "")[1].lower() or ".stl"
        if suffix not in CAD_EXTENSIONS:
            raise HTTPException(415, f"unsupported CAD extension {suffix!r}; "
                                     f"accepted: {sorted(CAD_EXTENSIONS)}")
        data = await file.read()
        if len(data) > MAX_BODY:
            raise HTTPException(413, "CAD file too large (40 MB max)")
        if mode not in ("mask", "depthmap"):
            raise HTTPException(422, "mode must be 'mask' or 'depthmap'")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            mesh, info = ingest(tmp_path)
            if mode == "depthmap":
                from cad.depth import mesh_to_depth
                d = mesh_to_depth(mesh)
                return {
                    "info": info.to_dict(),
                    "ny": d["ny"], "nx": d["nx"],
                    "depth01": d["depth01"].flatten().round(4).tolist(),
                    "solid": d["solid"].flatten().astype(int).tolist(),
                    "meta": d["meta"],
                }
            mask = mesh_to_mask(mesh)
            return {
                "info": info.to_dict(),
                "N": int(mask.shape[0]),
                "mask_summary": mask_summary(mask),
                "mask": mask.flatten().astype(int).tolist(),
            }
        except ValueError as e:
            raise HTTPException(422, f"could not process mesh: {e}") from e
        finally:
            os.unlink(tmp_path)

    # RAG SECTION
    # global dictionary to hold the RAG pipeline instance --- lazy initialization
        # loads models, embeddings, and indexes documents, so keep global state
        # each request runs RAG endpoint independently, so this keeps data outside the functions
    _rag = {"pipe": None, "generator": None}

    def _get_rag():
        """Lazy singleton: full NerveGear corpus (knowledge base + repo docs +
        glossary + roadmap), hybrid retrieval, optional generator via
        ANTHROPIC_API_KEY, optional cross-encoder via NerveGear_RAG_RERANK=1."""
        if _rag["pipe"] is None:
            import os as _os
            from rag import RagPipeline, get_default_embedder
            from rag.corpus import nervegear_corpus
            reranker = None
            if _os.environ.get("NerveGear_RAG_RERANK") == "1":
                from rag.rerank import CrossEncoderReranker
                reranker = CrossEncoderReranker.try_create()
            pipe = RagPipeline(get_default_embedder(), reranker=reranker)
            pipe.index_documents(nervegear_corpus())
            from rag.generate import default_generator
            _rag["generator"] = default_generator()
            _rag["pipe"] = pipe
        return _rag["pipe"]

    @app.post("/rag/query")
    async def rag_query(payload: dict) -> dict:
        """
        Retrieve grounded context for a question. Returns citations + the
        assembled prompt (wire a generator server-side to return written answers).
        payload dict example:
        {
            "question": "What is Reynolds number?",
            "k": 3
        }
        """
        question = str(payload.get("question", "")).strip()
        if not question:
            raise HTTPException(400, "missing 'question'")
        if len(question) > 2000:
            raise HTTPException(422, "question too long (2000 chars max)")
        k = max(1, min(int(payload.get("k", 4)), 10))
        pipe = _get_rag()
        out = pipe.answer(question, k=k, generator=_rag["generator"], mode="hybrid")
        out["generated"] = _rag["generator"] is not None
        return out

    @app.post("/rag/explain")
    async def rag_explain(payload: dict) -> dict:
        """Explain the CURRENT design's results, grounded in the knowledge base.

        payload: { "metrics": {...}, "confidence": {...}, "question": optional }
        Builds a question from the design state (worst confidence checks, key
        metrics), retrieves the relevant physics, and answers with citations.
        """
        metrics = payload.get("metrics") or {}
        confidence = payload.get("confidence") or {}
        if len(str(metrics)) > 20000:
            raise HTTPException(422, "metrics payload too large")
        user_q = str(payload.get("question", "")).strip()[:2000]
        failing = [c for c in confidence.get("checks", []) if not c.get("ok")]
        key = {k: metrics.get(k) for k in (
            "max_junction_temp", "thermal_margin", "pressure_drop_kPa", "COP",
            "Re", "coolant_dT", "maldistribution_cv", "junction_gradient") if k in metrics}
        state = "; ".join(f"{k} = {round(v, 2) if isinstance(v, (int, float)) else v}"
                          for k, v in key.items())
        fails = "; ".join(c.get("detail", c.get("name", "")) for c in failing) or "none"
        question = user_q or (
            "Explain these microchannel cooling results to the designer and say "
            "what to improve first.")
        pipe = _get_rag()
        hits = pipe.retrieve_hybrid(question + " " + " ".join(c.get("name", "") for c in failing), k=5)
        prompt = (
            "You are NerveGear's engineering assistant. A designer just simulated a "
            "microfluidic cooling design.\n"
            f"RESULTS: {state or 'no metrics provided'}\n"
            f"FAILING CHECKS: {fails}\n\n"
            "Using ONLY the context below, explain what the numbers mean and give "
            "the single highest-leverage improvement. Cite sources as [1], [2].\n\n"
            "CONTEXT:\n" +
            "\n\n".join(f"[{i}] (source: {h.chunk.source})\n{h.chunk.text}"
                          for i, h in enumerate(hits, 1)) +
            f"\n\nQUESTION: {question}\n\nANSWER:")
        gen = _rag["generator"]
        if gen is not None:
            try:
                text = gen(prompt)
            except Exception as e:  # noqa: BLE001 - API hiccup: degrade, don't 500
                text = f"[generator unavailable: {e}] " + _extractive(hits)
        else:
            text = _extractive(hits)
        return {
            "answer": text,
            "generated": gen is not None,
            "citations": [{"n": i, "source": h.chunk.source,
                           "preview": h.chunk.text[:160]}
                          for i, h in enumerate(hits, 1)],
        }

    def _extractive(hits) -> str:
        if not hits:
            return "No relevant context found in the knowledge base."
        parts = [f"[{i}] {h.chunk.text.strip()}" for i, h in enumerate(hits[:3], 1)]
        return ("Most relevant knowledge-base passages (set ANTHROPIC_API_KEY for "
                "written answers):\n\n" + "\n\n".join(parts))

    # CFD SECTION

    return app


# Module-level app for `uvicorn server.app:app`.
app = create_app()
