"""HTTP endpoint integration tests via FastAPI TestClient.

Camera is disabled (no webcam needed). The CFD proxy is tested with a monkey-
patched client so no CFD server is required. WebSocket paths are exercised
elsewhere; here we cover the request/response endpoints the frontend calls."""

from __future__ import annotations

import io

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
trimesh = pytest.importorskip("trimesh")

from fastapi.testclient import TestClient

from server.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app(enable_camera=False))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["camera_enabled"] is False


def test_cad_upload_returns_mask(client):
    m = trimesh.creation.box(extents=(2, 1, 0.5))
    buf = io.BytesIO()
    m.export(buf, file_type="stl")
    buf.seek(0)
    r = client.post("/cad/upload", files={"file": ("part.stl", buf, "application/octet-stream")})
    assert r.status_code == 200
    d = r.json()
    assert d["N"] == 128
    assert d["info"]["n_faces"] == 12
    assert d["mask_summary"]["solid_cells"] > 0
    assert len(d["mask"]) == 128 * 128


def test_rag_query_returns_citations(client):
    r = client.post("/rag/query", json={"question": "what is the reynolds number?", "k": 2})
    assert r.status_code == 200
    d = r.json()
    assert len(d["citations"]) == 2
    assert any("reynolds" in c["source"].lower() for c in d["citations"])


def test_rag_query_requires_question(client):
    assert client.post("/rag/query", json={}).status_code == 400


