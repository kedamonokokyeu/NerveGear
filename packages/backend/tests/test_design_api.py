"""End-to-end API tests for the axiom/solve.v1 endpoints."""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(enable_camera=False))


def _etch_block(ny=64, nx=96, n_ch=10, depth=350.0):
    d = np.zeros((ny, nx), dtype="<f4")
    for k in range(n_ch):
        rr = int((k + 0.5) / n_ch * ny)
        d[rr:rr + 2, :] = depth
    return {"ny": ny, "nx": nx, "encoding": "b64f32",
            "data": base64.b64encode(d.tobytes()).decode()}


SEG_REQ = {
    "geometry": {
        "die": {"L": 12000, "W": 10000, "thk": 800, "base_thk": 150},
        "segments": [
            {"x0": 0, "y0": 5000, "x1": 6000, "y1": 5000, "w": 600, "depth": 400},
            {"x0": 6000, "y0": 5000, "x1": 12000, "y1": 5000, "w": 600, "depth": 400},
        ],
    },
    "conditions": {"flow_mlmin": 60, "power": {"background_wcm2": 40}},
    "solve": {"engine": "auto"},
}


def test_evaluate_segments(client):
    r = client.post("/api/evaluate", json=SEG_REQ)
    assert r.status_code == 200
    j = r.json()
    assert j["engine"] == "reduced"
    assert j["metrics"]["max_junction_temp"] > 25
    assert j["confidence"]["score"] > 0
    assert j["maps"]["tj_c"]["encoding"] == "b64f32"
    assert len(j["segments"]) == 2
    assert all(k in j["metrics"] for k in
               ("pressure_drop_kPa", "COP", "Re", "thermal_margin"))


def test_evaluate_freeform_etch(client):
    req = {"geometry": {"die": {"L": 10000, "W": 8000, "thk": 600, "base_thk": 120},
                        "etch": _etch_block()},
           "conditions": {"flow_mlmin": 200, "power": {"background_wcm2": 80}},
           "solve": {}}
    j = client.post("/api/evaluate", json=req).json()
    assert j["meta"]["model"] == "heleshaw"
    assert j["metrics"]["energy_balance_error"] < 0.01


def test_solve_falls_back_and_tags_engine(client):
    j = client.post("/api/solve", json=SEG_REQ).json()
    assert j["engine"] == "reduced"
    assert "note" in j["meta"]


def test_keepout_flags_violation(client):
    req = {**SEG_REQ, "geometry": {**SEG_REQ["geometry"],
           "keep_out": [{"x0": 5000, "y0": 4500, "x1": 7000, "y1": 5500, "reason": "TSV"}]}}
    j = client.post("/api/evaluate", json=req).json()
    assert not j["manufacturability"]["manufacturable"]
    assert any(d["rule"] == "keep_out" for d in j["manufacturability"]["details"])


def test_validation_errors_are_422_with_path(client):
    r = client.post("/api/evaluate", json={"geometry": {"die": {"L": -3}}})
    assert r.status_code == 422
    assert "$.geometry.die.L" in r.json()["detail"]


def test_solve_serves_reduced_and_solver3d_is_an_honest_503(client):
    # The 3-D engine was removed (2026-07, to be rebuilt from scratch):
    # "auto" must fall through to the reduced model and say so; asking for
    # "solver3d" explicitly must fail loudly, not silently downgrade.
    j = client.post("/api/solve", json=SEG_REQ).json()
    assert j["engine"] == "reduced"
    explicit = {**SEG_REQ, "solve": {**SEG_REQ.get("solve", {}), "engine": "solver3d"}}
    assert client.post("/api/solve", json=explicit).status_code == 503


def test_optimize_small(client):
    j = client.post("/api/optimize", json={
        "request": SEG_REQ, "n_iter": 12,
        "constraints": {"pressure_drop_kPa": 100}}).json()
    assert j["n_evaluated"] == 12
    assert isinstance(j["history"], list)


def test_rag_query_and_explain(client):
    j = client.post("/rag/query", json={"question": "what is fRe for a square duct?"}).json()
    assert j["citations"]
    j = client.post("/rag/explain", json={
        "metrics": {"max_junction_temp": 140, "Re": 3000},
        "confidence": {"checks": [{"name": "laminar", "ok": False,
                                   "detail": "Re 3000 >= 2300"}]}}).json()
    assert "answer" in j and j["citations"]


def test_rag_query_caps(client):
    r = client.post("/rag/query", json={"question": "x" * 3000})
    assert r.status_code == 422


def test_cad_upload_rejects_bad_extension(client):
    r = client.post("/cad/upload", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
