# Phase 1 — Review guide

What was built, in what order to read it, and which concepts/libraries to
look up when a file feels unfamiliar. Written for review of the Phase 1 CAD
viewer (see `docs/PHASE1_BUILD.md` for the plan it implements).

## The 60-second architecture

```
                      you pick / drop a file
                                │
   app/state/modelSession.ts ── loadModel.ts (the ONE import door)
                                │
             ┌── .glb/.gltf ────┤──── .step/.stp ──► backend /cad/convert
             │                  │                     (OpenCASCADE → GLB)
             ▼                  ▼
          gltfIngest.ts  →  Part[] (µm, Y-up, names, materials)
                                │
      store.ts (UI facts) ◄─────┤────► ViewportHost.tsx (the React↔Three seam)
                                │              │
         panes read the store   │       chipScene / selection / section /
      (Structure, Properties)   │       explode drive the actual 3D scene
```

Two rules make the whole thing reviewable: geometry enters through exactly
one door (`loadModel.ts`), and React never touches Three.js objects except
inside `ViewportHost.tsx`.

## Reading order (frontend)

1. **`studio/src/engine/model/part.ts`** — the contract everything
   else obeys. Read this first; it defines "a Part", the µm rule, and the
   ×0.001 render scale. Everything downstream is a consequence.
2. **`src/engine/model/loadModel.ts`** — the format dispatcher. Short; shows
   how GLB and STEP end up in the same normalization path.
3. **`src/engine/model/gltfIngest.ts`** — where file-space becomes
   contract-space (unit scale + up-axis baked into vertices, exactly once).
4. **`src/engine/model/materials.ts`** — the three-tier material resolution
   (metadata → name heuristics → manual). Pure data + regexes, easy read.
5. **`src/engine/model/chipScene.ts`** — Parts → meshes + feature edges;
   also the dispose contract (the "no leaks on reimport" requirement).
6. **`src/engine/model/selection.ts`**, **`section.ts`**, **`explode.ts`** —
   one tool per file. Selection = raycasting; Section = one clipping plane
   plus screen-space drag math (the hairiest math in Phase 1, commented
   inline); Explode = bounding-box tiering + a 280 ms animation.
7. **`src/app/state/store.ts`** then **`src/app/state/modelSession.ts`** —
   who owns what state, and the React↔engine seam.
8. **`src/app/viewport/ViewportHost.tsx`** — the wiring hub. If you
   understand this file you understand the app's data flow.
9. The panes (`StructurePane`, `PropertiesPane`, `CommandBar`,
   `LegendOverlay`, `import/*`) — thin store readers/writers; skim.

## Reading order (backend)

1. **`backend/cad/step_to_glb.py`** — tessellation + the unit story
   (OpenCASCADE normalises to metres; the module docstring explains).
2. **`backend/server/cad_api.py`** — the endpoint: validation,
   limits, the `X-NerveGear-Microns-Per-Unit` header contract.
3. **`backend/tests/test_step_to_glb.py`** — generates a real
   two-solid STEP with OpenCASCADE and round-trips it through the endpoint;
   doubles as executable documentation of the whole seam.

## Concepts worth 15 minutes each

- **glTF / GLB** — the "JPEG of 3D": triangles + a node tree + names, metres,
  Y-up by spec. GLB is the binary packing. We treat it as the app's native
  format because Three.js parses it without any server.
- **STEP (ISO 10303)** — exact surfaces (BREP), not triangles; what real CAD
  tools exchange. Key mental model: STEP must be *tessellated* (approximated
  into triangles) before a GPU can draw it — that's the backend's job.
- **Tessellation tolerances** — `tol_linear` is the max distance the mesh may
  sag from the true surface; smaller = smoother = heavier meshes.
- **Raycasting** — "what did I click": shoot a line from the camera through
  the pixel, intersect it with the meshes. `THREE.Raycaster` does the math;
  selection.ts shows the idiomatic use.
- **Clipping planes** — the GPU discards fragments on one side of a plane
  (`renderer.localClippingEnabled`, `material.clippingPlanes`). Uncapped =
  you see hollow shells at the cut; capping (filling the cross-section) is a
  known harder technique (stencil buffer) deferred on purpose.
- **`EdgesGeometry` / feature edges** — lines only where adjacent triangles
  meet above a threshold angle; what makes flat-shaded CAD readable.
- **Matrix baking** — `geometry.applyMatrix4(world)` writes a node's
  placement into its vertices. We bake at import so every later feature
  (framing, section depth, explode offsets) works in one flat space.
- **Units discipline** — the single most bug-prone thing in CAD tooling.
  Our rule: convert to µm once at import, render at ×0.001, display µm/mm.
  Grep for `micronsPerFileUnit` and `RENDER_UNITS_PER_MICRON` to audit it.
- **`useSyncExternalStore`** — the React 18 hook that lets a plain
  dependency-free store drive components (store.ts is ~50 lines because of
  it).

## Libraries that entered the project in Phase 1

- **cascadio** (backend, `requirements.txt`) — a thin pip wrapper around
  OpenCASCADE's STEP reader + glTF writer. We chose it over full
  `pythonocc`/FreeCAD because it's one function and installs from a wheel.
- **@types/three** (frontend, dev-only) — TypeScript types for the pinned
  Three.js 0.160; no runtime change.
- **@playwright/test** (frontend, dev-only) — headless-browser e2e; drives
  the real UI against the built app.

## How to verify locally

```bash
# frontend: types + module syntax + headless engine checks (no browser)
cd packages/studio
npm install
npm run check

# frontend: full browser e2e + screenshots (tests/artifacts/phase1/)
npm run build
npx playwright install chromium   # once
npm run test:e2e

# backend: unit + endpoint round-trip (needs: pip install cascadio cadquery-ocp)
cd ../backend
python -m pytest tests/test_step_to_glb.py -v
```

Everything above except the browser e2e was run and passed during the build
(the CI sandbox couldn't launch Chromium; the spec is ready and the
screenshots it saves are the Step-10 sign-off evidence).

## Known edges (deliberate)

- Section is **uncapped**: at the cut you see part interiors as hollow
  shells. Locked decision; capping is a Phase 2+ pass.
- Phase 1 **trusts names**: a file with unnamed solids gets `part-N`
  stand-ins and `unknown` (grey) material — the inference sweep is Phase 2.
- The unit-review dialog appears only when the imported size is implausible
  (< 0.1 mm or > 1 m across); plausible-but-wrong units can still be fixed
  by re-importing after picking overrides there.
- InstancedMesh-style bulk parts (the demo chip's BGA ball field) are not
  reproduced in the sample GLB — real CAD exports name discrete solids.
