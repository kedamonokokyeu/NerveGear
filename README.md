# NerveGear

Sculpt microfluidic cooling directly into silicon, and simulate it live.
The Studio loads real chip CAD as a color-coded 3-D stack you can orbit,
section, and explode; a validated physics backend returns junction
temperatures, pressure drops, and manufacturability verdicts through one
locked JSON contract (`axiom/solve.v1`); and a grounded assistant explains
what the numbers mean. SEMulator-style process realism, for in-chip liquid
cooling.

The frontend is mid-rebuild on a phased plan (`docs/PHASE1_BUILD.md`).
Phase 0 (the enterprise frame) and Phase 1 (the 3-D chip model: CAD import,
semantic materials, selection, section, explode) are done and verified;
etch tools and live physics rebind in Phases 3–4.

```
NerveGear/
├─ packages/
│  ├─ studio/       the Studio frontend — React + Three.js (vendored, pinned,
│  │                fully offline); Phase 1 CAD viewer lives in src/engine/model
│  └─ backend/      the brain the Studio calls — solve API (axiom/solve.v1,
│                   contract/schema.py), micro/ physics, RAG assistant,
│                   STEP→GLB CAD conversion, gestures
├─ archive/
│  └─ legacy-2d-unet/   frozen U-Net keepsake (code + trained weights); never imported
├─ docs/            build plans, review guide, roadmap, original spec
└─ PROJECT_MAP.md   folder-by-folder truth — read this first
```

## Run it

```bash
# 1. backend (physics + RAG + CAD conversion), port 8200
cd packages/backend
pip install -r requirements.txt
python run.py

# 2. studio — either zero-install…
python packages/studio/run_studio.py          # http://localhost:8777 (serves dist/)
#    …or the dev workflow
cd packages/studio && npm install && npm run dev    # :5173, proxies the backend

# tests
cd packages/backend && pytest tests/ && python -m micro.validate && python -m rag.eval
cd packages/studio && npm run check && npm run build && npm run test:e2e
```

Optional: set `ANTHROPIC_API_KEY` before starting the backend and the
assistant writes cited answers instead of returning raw passages.

## Division of labor (locked)

- **`backend/micro`** = the **scoreboard + guardrails + oracle**: exact
  laminar network hydraulics, correlation conjugate thermal (see
  `backend/MODEL_CARD.md`), manufacturability rules, the optimizer, and the
  analytic benchmarks.
- **The 3-D field engine does not exist yet** — the old scaffolding was
  removed (2026-07) to be rebuilt from scratch later. Its slot in the API
  remains: `/api/solve` with engine "solver3d" returns an honest 503, and
  "auto" serves the reduced model, tagged as such. When the engine is
  rebuilt, it plugs into that slot with zero downstream change.
- The Studio speaks **one schema** (`axiom/solve.v1`, as code in
  `backend/contract/schema.py`). The contract string keeps its historical
  `axiom/` prefix on purpose — renaming it would break every stored design.

## What works today

- **Studio (Phase 1)**: import .glb/.gltf/.step/.stp (STEP converts on the
  backend via OpenCASCADE), color = material with a legend, named-part
  structure list, click/list selection with accent outline, axis-selectable
  section plane (typed depth + drag handle), stack-ordered explode, bundled
  sample package so the app is never empty. Generator for a realistic
  96-part test chip in `packages/studio/samples/`.
- **Backend physics** (unchanged by the rebuild): exact laminar
  resistor-network hydraulics, side-wall-corrected Hele-Shaw, conjugate
  junction-temperature maps, per-solve confidence scoring,
  manufacturability DRC, constrained optimizer with Pareto history — all
  validated against Shah & London + analytic benchmarks.
- **Assistant**: hybrid RAG (BM25 + dense + RRF, heading-aware chunks) over
  a microfluidics knowledge base, with citations.

## First-time git setup

Weights (`*.pth`, `*.task`) route through Git LFS (`.gitattributes`):

```bash
git lfs install
```
