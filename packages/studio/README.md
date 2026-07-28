# NerveGear Studio

Parametric CAE prototype: insert a thermal/power model, select any part, and drag
its handles (or type exact mm values) to resize it — fins, dies, lids, channels —
with live derived metrics. Optional webcam **gesture control** moves the camera.

## Run

Double-click `index.html` to use the viewer and all editing (no install).

For **gesture control** the browser needs a secure context (webcam is blocked on
`file://`), so serve it locally:

```bash
cd studio
python run_studio.py          # opens http://localhost:8777
```

Needs internet on first load (Three.js + the MediaPipe model are pulled from a CDN).

## Models

- **Micro-channel cold plate** — base + fin array; per-fin width/height/length, channel gap, plus wetted area A and hydraulic diameter Dₕ.
- **IGBT power module** — Cu baseplate, DBC substrate, IGBT/diode die grid, terminals, bond wires, translucent housing.
- **CPU / GPU package** — substrate, silicon die, IHS lid, BGA solder-ball array.
- **Import CAD** — STL / OBJ / GLTF / GLB / PLY; auto-normalised, with bounding-box scale handles.

## Controls

| Action | Mouse | Gesture |
|---|---|---|
| Orbit | drag | ✋ open palm |
| Pan | right-drag | 🤏 pinch |
| Zoom | scroll | ✋✋ two hands spread |
| Hold / brake | — | ✊ fist |
| Flow direction | — | 👉 point (reserved) |
| Select part | click | — |
| Resize part | drag colored cone handles | — |

## Files

- `index.html` — app shell · `styles.css` — design system
- `studio.js` — viewer, selection/handles, inspector, import, gesture wiring
- `models.js` — parametric geometry + metrics (pure, framework-free)
- `gestures.js` — MediaPipe hand tracking + NerveGear gesture vocabulary
- `run_studio.py` — local server for webcam access

The `models.js` solid model (named bounds per part) is the clean seam to backend's
voxelizer → CFD mask.
