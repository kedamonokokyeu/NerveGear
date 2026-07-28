# AXIOM Studio — Phase 1 Build Plan (The 3D Chip Model)

_Load the CAD export → the whole color-coded 3D stacked chip you can orbit, cut, explode,
and click into. Backend `solve.v1` contract and physics stay untouched._

Companion docs: `FRONTEND_REBUILD.md` (roadmap), `PHASE0_BUILD_PROMPT.md` (the frame, done).

---

## Locked decisions (agreed — do not re-litigate)

| Decision | Choice | Why |
|---|---|---|
| Primary format | **GLTF / GLB** | Mesh + named nodes + units; loads natively in Three.js; fully offline |
| STEP support | **Convert STEP → GLB on the backend** | Observable, loggable, scalable; keeps the client light (OpenCASCADE is heavy) |
| Part structure | **Named / separate solids** | Real CAD assemblies are labeled; Phase 1 trusts labels, Phase 2 infers the rest |
| Material color | **Semantic: CAD metadata → label heuristics → manual** | Color encodes material meaning, consistent legend |
| Canonical units | **Micrometers (µm)** internally | Matches the `solve.v1` contract; convert only at import; render at ×0.001 (µm→mm) |
| Orientation | **Y-up; die flat in X–Z; stack grows +Y; flow +X** | Reads like the PDF; matches glTF's native Y-up; one fixed map to the physics frame |
| Selection highlight | Accent outline | Reads cleanest in an enterprise tool |
| Section capping | Deferred (uncapped cut first) | Stencil capping is a separate, harder pass |

### Units contract (one source of truth)
- Everything internal is **µm**. Convert the file's declared units to µm **once, in the loader**
  (glTF = meters → ×1,000,000; STEP = its embedded unit). Never convert anywhere else.
- **Render** at ×0.001 (µm→mm) so the camera math stays in a ~10–20 unit range.
- **Display** µm by default (etch-feature scale); mm for die-level dimensions.

### Orientation map (viewer ↔ physics)
| Viewer axis | Meaning | `solve.v1` axis |
|---|---|---|
| X | flow (inlet → outlet) | x |
| Y | depth into die (stack up) | z |
| Z | transverse | y |
A well-formed Y-up glTF needs no rotation. A Z-up file gets one −90° X-rotation at import.

---

## Code standards for this build (apply to every file)

The point: anyone (including you, with no CAD or graphics background) can open a file and
understand it. Non-negotiable for Phase 1.

1. **Descriptive names, no cryptic abbreviations.** `dieDepthMicrons`, not `d`. `partsByMaterial`,
   not `pbm`. A name should tell you the *what* and the *unit*. Units go in the name when a value
   has one: `widthMicrons`, `flowMlPerMin`.
2. **Self-documenting code.** Write it so it reads without comments. Small functions that do one
   obvious thing, top-to-bottom flow. If a function needs a paragraph to explain *what* it does,
   split it.
3. **Comments explain _why_, not _what_.** No `// increment i`. Yes `// glTF is metres; the contract
   is microns, so convert once here and nowhere else`.
4. **No jargon in docs.** Every file starts with a short plain-English header: what this file is,
   in one or two sentences a non-specialist understands. Domain terms get a one-line gloss the
   first time they appear.
5. **One concept per file.** Loading, materials, selection, section, explode each live in their
   own module. You should be able to guess a file's contents from its name.
6. **Constants over magic numbers.** `const FEATURE_EDGE_ANGLE_DEGREES = 30;` with a comment on
   why 30, not a bare `30` buried in a call.

---

## The ten steps (each verified before the next)

### Step 1 — Ingestion contract & loader skeleton
Define the internal `Part` shape — `{ id, name, materialKey, geometry, transform, boundingBox }`
— and a format dispatcher keyed on file extension. Wire `GLTFLoader` (native path). Stub the STEP
path as "upload to backend → receive GLB → load it." All imports pass through **one** normalization
step that applies the µm-canonical conversion and the Y-up orientation map.
**Done when:** a `.glb` loads into a list of named `Part`s in µm, correctly oriented.

### Step 2 — Backend STEP → GLB conversion
Add a conversion endpoint in `AXIOM3D_backend/cad/` (OpenCASCADE via `cascadio`/pythonocc) that
tessellates STEP, preserving node names and units, and returns GLB. The client only uploads and
loads the result.
**Done when:** a `.step` file round-trips to named, oriented parts in the viewer.

### Step 3 — Scene assembly
Build the `THREE.Group` from `Part[]`: one mesh per named solid, a flat material per material,
hard feature edges (`EdgesGeometry`, ~30–45°). Full dispose-on-reimport (no memory leaks). Frame
the real model bounding box on load.
**Done when:** a loaded model renders with crisp edges and reimport leaves nothing behind.

### Step 4 — Semantic material system
Resolve each part's material: CAD metadata → label heuristics → manual override. Map material →
the `--mat-*` palette. Add a theme-aware, toggleable **legend** overlay. Edge color follows the
theme (same fix pattern as the background).
**Done when:** color = material is correct and legible in both themes, with a working legend.

### Step 5 — Selection
Raycast pick → accent outline → write `store.selection`. Bidirectional with the left list. The
Properties panel shows the selected part's name, material (editable), and L×W×H in µm/mm.
**Done when:** clicking a part (or its list row) selects it both places and shows its identity.

### Step 6 — Structure pane (flat, from the file)
The left pane lists the named parts with material swatches, synced to selection.
(Hierarchy + inferred dimensions are Phase 2.)
**Done when:** the parts list mirrors the model and drives/reflects selection.

### Step 7 — Section (make the button real)
One clipping plane, axis-selectable, positioned by a **typed depth field + a drag handle** (no
slider). Uncapped first.
**Done when:** Section slices the model to reveal the interior, and toggling it off restores.

### Step 8 — Explode (make the button real)
Offset parts along +Y by stack order; the Explode toggle animates 0→1; a typed amount field for
fine control.
**Done when:** Explode separates the layers and restores cleanly.

### Step 9 — Import UX
Enable **Import CAD**: file picker (`.glb/.gltf/.step/.stp`) + drag-drop, loading/error states,
and a unit/up-axis override dialog for ambiguous files. Ship a **bundled sample stacked-die GLB**
as the default model and the test fixture.
**Done when:** a user can import their own file, and the app is never empty.

### Step 10 — Verify & lock
Load a GLB and a converted STEP; named parts select; color=material + legend correct; section +
explode work; reimport disposes cleanly; both themes correct (background **and** edges).
Screenshots: iso, section, exploded, a selected TSV — both themes.
**Done when:** all of the above pass and you sign off.

---

## Out of scope for Phase 1 (Phase 2+)
The inference/segmentation sweep, captured parametric dimensions, the hierarchical tree, etch /
deposition, the step/checkpoint tree, and any physics.
