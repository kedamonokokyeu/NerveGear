# Thermal pivot (2D) — cooling design on top of the fluid surrogate

This is the move from "predict airflow" to "predict and improve cooling." It turns
the workspace into a tool for designing how power modules, CPUs, and GPUs are
cooled — and the same abstraction scales up to whole server rooms / HVAC.

## The core idea: couple, don't retrain (yet)

Your fluid surrogate already predicts the **velocity** field for a geometry. Cooling
is what that velocity *does to heat*. So the thermal layer solves the steady heat
equation using your velocity field plus the chips' power:

```
rho*cp (u · grad T) = div(k grad T) + Q
   u = coolant velocity (from your fluid surrogate)
   k = conductivity (high in the metal heat-sink, low in the coolant)
   Q = heat injected by the chips
   T = temperature  (what we solve for)
```

This is a classical, fast, sparse linear solve (`physics/heat.py`). Two payoffs:
1. **Temperature today, no training.** It rides your existing model — you get real,
   physically-grounded cooling behavior immediately.
2. **It generates the ground truth** for the end-to-end thermal *surrogate* you'll
   train later (`tools/thermal_datagen.py`). That ML model is yours to build; this
   solver feeds it.

The numbers are currently **relative** (good vs. bad cooling), not calibrated to
real Kelvin/Watts — calibration to physical units is a later, data-driven step. The
design decisions it supports (hotspot location, does more flow help, which layout
runs cooler) are already correct, which is what interactive iteration needs.

## What was added (all verified, runs with no model)

- **`physics/heat.py`** — `HeatSource` (a chip/module with a power level),
  `power_density_map`, and `solve_temperature` (the conjugate advection-diffusion
  solve). `temperature_field()` adds a `temperature` channel to an existing Field.
- **`physics/metrics.py`** — semiconductor metrics: per-chip junction temps, max
  junction temp (the headline number), thermal resistance (ΔT/power), hotspot
  location, and thermal margin vs. a junction limit (e.g. 175 °C for silicon).
- **`physics/design.py`** — `ThermalDesign` = geometry + heat sources + boundary
  conditions, with `solve(design, velocity_provider)`. The velocity provider is
  injected: `demo_velocity_provider` (no model) or `surrogate_velocity_provider`
  (your CFD server).
- **`POST /thermal/solve`** — design JSON → temperature field + metrics for the frontend.
- **`tools/thermal_demo.py`** — end-to-end proof; shows junction temp dropping as
  coolant flow rises.
- **`tools/thermal_datagen.py`** — turns your LBM samples into thermal training data.

## Verified behavior (tests)

`test_heat.py`, `test_thermal_design.py`, `test_thermal_endpoint.py` check the
physics that matters: heat localizes at the chip, conduction is symmetric, the warm
plume advects downstream, **more coolant flow lowers the peak junction temperature**,
and more power raises it — through the solver, the design abstraction, and the HTTP
endpoint.

## Why it scales (chip → data center)

`ThermalDesign` is scale-invariant. The same object and solve describe:

| scale     | geometry                | heat sources        | coolant            |
|-----------|-------------------------|---------------------|--------------------|
| package   | cold plate / heat-sink  | IGBT / diode dies   | liquid or air      |
| board     | chassis + sinks         | several packages    | chassis airflow    |
| rack      | rack enclosure          | boards              | front-to-back air  |
| **room**  | room + aisles           | **whole racks**     | **CRAC-unit air**  |

Only the physical scale and the source list change — the pipeline (flow → heat →
metrics → optimize) is identical. That's how this grows from "cool a power module"
to "lay out a server room's cooling" without a rewrite.

## Parametric cold-plate geometry (from the NerveGear spec)

The spec is about *stretching cold-plate geometry by hand and watching cooling vs.
pressure trade off live*, grounded in `Heat Removed = h·A·ΔT`: more/thinner/taller
fins add surface area `A` (and turbulence) → cooler chips, but obstruct the coolant
→ higher pressure drop. That trade-off is now first-class:

- **`design.finned_cold_plate(N, n_fins, fin_thickness, fin_height, base_thickness)`**
  — the parametric fin geometry an engineer adjusts. `design.build_mask(spec, N)`
  turns a small JSON spec into the mask, so the frontend's geometry sliders drive it.
- **`metrics.wetted_surface_area(field)`** — the solid/fluid interface length (the
  `A` term). More fins → more `A`.
- **`POST /thermal/solve`** accepts a `geometry` spec and returns the mask, so the
  frontend renders the exact cold plate.
- **`tools/fin_tradeoff.py`** prints the cooling-vs-pressure curve; verified in
  `test_geometry.py` (more fins → lower junction temp, higher pressure drop, more area).

## Frontend: `web/workspace.html`

A 2D thermal design workspace: a temperature heatmap with flow streamlines in the
middle, geometry/coolant/power sliders on the left (fins, fin thickness/height,
pump flow, inlet temp, chip count/power, junction limit), and a live cooling
scoreboard on the right (max junction temp vs. limit, margin, thermal resistance,
pressure drop, surface area, per-chip temps). Drag a slider and everything
re-solves through `/thermal/solve`. "Set baseline" snapshots the current metrics so
each subsequent change shows green/red deltas — the Tier-A compare, live. Flow
toggles between the demo provider and your CFD surrogate.

This is the 2D realization of the spec's "intuitively manipulate the design and see
the result" — the gesture layer can drive these same sliders later.

## What's reserved for you (the ML work)

1. **The thermal surrogate** — train a model to predict temperature directly (fast),
   using `thermal_datagen` output as targets. Add a `power_map` input channel and a
   `temperature` output channel to your UNet when you do.
2. **Unit calibration** — map the relative solve to real K/W with material props.
3. **The real optimizer evaluator** — wire `design.solve` (with the surrogate
   provider) in as the `evaluator` for `physics/optimize.py`, turning the loop into
   real cooling-layout optimization (minimize max junction temp s.t. weight /
   pressure-drop constraints).
