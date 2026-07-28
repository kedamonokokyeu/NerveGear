# NerveGear — project map

Read this first. What every folder is, who owns it, how they connect.
(Deep-cleaned 2026-07: legacy trees deleted, `packages/` flattened,
AXIOM branding retired everywhere except the `axiom/solve.v1` contract
string, which is load-bearing.)

## The product spine

**`packages/studio/`** — the Studio (frontend; React + Three.js).
- `src/app/` — the React frame: `layout/` (shell, command bar, status bar,
  splitters) · `panes/` (Structure list, Process tree, Properties) ·
  `state/store.ts` (the one UI store) + `state/modelSession.ts` (owns the
  loaded model; the React↔engine seam) · `import/` (drag-drop, unit review
  dialog, bundled-sample boot) · `theme/`, `inputs/` (typed fields — no
  sliders) · `viewport/` (ViewportHost bridge, view gizmo, material legend).
- `src/engine/` — Three.js, no React: `viewport.js` (camera/orbit/render
  loop) · `model/` (Phase 1): `part.ts` (the Part contract + µm units) ·
  `loadModel.ts` (the ONE import door) · `gltfIngest.ts` · `stepConversion.ts`
  (backend round-trip) · `materials.ts` (semantic color = material) ·
  `chipScene.ts` (meshes + feature edges + dispose) · `selection.ts` ·
  `section.ts` · `explode.ts`.
- `src/assets/sample-stacked-die.glb` — bundled default model & test fixture;
  `samples/` — the 96-part realistic accelerator package + its generator.
- `tests/` — `phase1_engine_checks.mjs` (headless pipeline checks),
  `phase1.spec.js` (Playwright e2e + sign-off screenshots),
  `syntax_check.mjs`. Run via `npm run check` / `npm run test:e2e`.
- Three.js is **pinned (0.160.0) and vendored** (`vendor/three/`): the
  no-build path (`python run_studio.py`, port 8777, serves `dist/`) and the
  Vite path (`npm run dev`, port 5173, proxies the backend) run the same
  pinned version. No CDN; offline works.

**`packages/backend/`** — the backend brain (FastAPI, port 8200).
- `server/design_api.py` — the solve.v1 endpoints: `/api/evaluate` (fast
  reduced), `/api/solve` (reduced model behind the full-solve shape; the
  removed 3-D engine's slot 503s honestly), `/api/optimize` (constrained
  search + Pareto history).
- `server/cad_api.py` + `cad/step_to_glb.py` — STEP → GLB tessellation for
  the Studio viewer (`POST /cad/convert/step-to-glb`, cascadio/OpenCASCADE).
- `micro/` — the validated reduced physics: `geometry.py` (exact laminar
  resistor networks), `network_thermal.py`, `heleshaw.py`, `conjugate.py`,
  `manufacturing.py` (DRIE rules), `confidence.py`, `design.py`+`optimize.py`,
  `validate.py` (analytic benchmarks). See `MODEL_CARD.md` for the envelope;
  `THERMAL.md` for the thermal formulation.
- `contract/` — **`axiom/solve.v1` as code**: `schema.py`
  (parse/validate/encode; the contract string keeps its historical name) and
  `rasterize.py` (geometry → etch-depth grid for the reduced solver). Moved
  here when the 3-D engine scaffolding was removed (2026-07, to be rebuilt
  from scratch); a future engine imports the contract from here.
- `rag/` — hybrid retrieval (dense + BM25 + RRF), knowledge base, golden
  eval set (`python -m rag.eval`), optional Anthropic generation.
- `cad/` — mesh ingest → mask / depth map for the CAD-to-etch round-trip
  (rebinds to the UI in Phase 3+).
- `gestures/` + `server/app.py` WebSockets — webcam gesture engine
  (`models/hand_landmarker.task` is its MediaPipe model, via Git LFS).

## `archive/` — frozen, never imported

- `legacy-2d-unet/` — the 2-D LBM + U-Net prototype: source, trained
  weights, write-up. Kept as a learning reference only; see its README.
  (The rest of the old `legacy-2d/` tree — training data, 2-D designer,
  physics-demo — was deleted in the deep-clean.)

## Contracts & docs

- `backend/contract/schema.py` — the JSON seam, as executable code. Change
  only with a version bump.
- `backend/MODEL_CARD.md` — reduced-engine assumptions, envelope, benchmarks.
- `docs/PHASE1_BUILD.md` — the Phase 1 build plan (done);
  `docs/PHASE1_REVIEW_GUIDE.md` — reading order + concepts for review.
- `docs/Axiom_Roadmap.docx`, `docs/NerveGear Doc.pdf` — strategy + original
  spec (historical; keep their filenames).

## Where to work

| You want to… | Go to |
|---|---|
| Studio UI / viewer / import | `packages/studio/src/` |
| Reduced physics, DRC, optimizer, confidence | `packages/backend/micro/` |
| Solve API shapes and endpoints | `packages/backend/server/design_api.py` + `packages/backend/contract/schema.py` |
| STEP→GLB conversion | `packages/backend/cad/step_to_glb.py` |
| Assistant knowledge | `packages/backend/rag/knowledge/` |
| CI / lint | `.github/workflows/ci.yml`, `ruff.toml` |
