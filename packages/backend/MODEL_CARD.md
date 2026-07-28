# Model card — the NerveGear reduced engine (`micro/`)

**What it is.** The physics behind `engine: "reduced"` responses: exact laminar
resistor-network hydraulics + Shah & London correlation thermal for channel
networks (`micro/network_thermal.py`), and a side-wall-corrected Hele-Shaw
depth-averaged solve for freeform etches (`micro/heleshaw.py`). It is the
scoreboard, guardrail, and oracle for the 3-D solver — not its replacement.

**Intended use.** Interactive design-space exploration, optimizer inner loop,
sanity/validation reference for the 3-D solver and future surrogate. Trends and
scaling are trustworthy inside the envelope; local field values are
approximations.

## Validity envelope

| assumption | valid range | outside it |
|---|---|---|
| laminar flow | Re < 2300 (design target < 1000) | fRe/Nu correlations wrong; confidence fails `laminar` |
| channel size | 20 µm – 2 mm width | outside benchmarked range; flagged |
| fully developed | L ≫ entrance length | entrance effects under-predicted (conservative on ΔP, optimistic on local h near inlets) |
| single-phase liquid | no boiling | Tj > ~110 °C water: two-phase effects unmodeled — treat as "too hot", not as a prediction |
| raster resolution | ≥ 3 cells across narrowest channel | Hele-Shaw width correction quantizes; flagged by `resolution` check |

## Benchmarks (tests/test_network_thermal.py, tests/test_heleshaw.py, micro/validate.py)

- fRe limits: 24.0 (parallel plates), 14.23 (square duct) — exact polynomial.
- Nu_H1 limits: 8.235 / 3.61 — exact polynomial.
- Energy balance: coolant ΔT ≡ P/(ṁ·cp) to < 10⁻⁶ (network) and < 1% (Hele-Shaw march).
- Straight-array Tj rise vs validated conjugate model: within 10% (uniform flow).
- Straight painted channel ΔP vs analytic Fanning: within ~12% (≥3-cell channels).
- Network single-channel ΔP vs analytic: within 5%.

## Known approximations

1. Heat collection by nearest-channel Voronoi partition (not a solved 3-D
   conduction field); lateral spreading approximated as q″·d²/(2·k·t_die).
2. Wall superheat is segment-averaged (constant-flux H1 Nusselt).
3. Zero-flow branches (dead ends, balanced bridges in symmetric networks)
   cannot advect their heat: it is re-routed to the nearest flowing channel
   with the extra conduction distance charged to the spreading term, and the
   response carries a warning. Remaining near-stagnant branches get a 1% flow
   floor and are flagged.
4. Manifolds outside the die are ideal plena (zero resistance).
5. Temperature-independent fluid properties (evaluated near 300 K).

## Confidence reporting

Every response carries `confidence.checks` — the named validity checks above,
each pass/fail with detail. The score is their weighted pass fraction, NOT an
error bar. The `engine_fidelity` check always dings the reduced model so a
reduced solve can never present itself as field-accurate.
