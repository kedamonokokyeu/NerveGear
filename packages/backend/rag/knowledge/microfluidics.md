# Microfluidic chip cooling — the NerveGear knowledge base

This document is indexed by the assistant. It explains the physics, the
manufacturing rules, and the meaning of every number NerveGear shows.

## Why cool inside the chip

Modern compute dies dissipate 100–1000 W/cm² at hotspots — far beyond what
air or even cold plates on the package lid can remove, because every layer
between the junction and the coolant (die, TIM, lid, TIM again, plate) adds
thermal resistance. Etching coolant channels directly into the silicon die
(or a bonded silicon interposer) removes those layers: the coolant flows tens
of micrometres from the transistors. This is microfluidic in-chip cooling,
demonstrated by Tuckerman & Pease in 1981 — 790 W/cm² with a 600 µm silicon
microchannel heat sink — and revived today for AI accelerators and power
electronics.

## Junction temperature (Tj)

The junction temperature is the temperature of the active transistor layer —
the number that must stay below the limit (typically 105 °C for silicon
logic, 150–175 °C for SiC power devices). NerveGear's Tj map decomposes as:

    Tj(x, y) = T_coolant(nearest channel)     — coolant heats up downstream
             + wall superheat                  — convection from wall to fluid
             + conduction through the base     — q'' × t_base / k_silicon
             + lateral spreading penalty       — q'' × d² / (2 k t_die)

Each term suggests its own fix: high coolant temperature → more flow or a
shorter path; high wall superheat → more wetted area (more/narrower channels
where the heat is); high spreading penalty → channels closer to the hotspot.

## Reynolds number and laminar flow

Re = ρ·u·Dh/µ compares inertia to viscosity. In microchannels Re is almost
always below ~1000: the flow is laminar — smooth, predictable, and exactly
solvable. NerveGear's correlations (fRe, Nu) are laminar results; above Re ≈ 2300
the flow transitions to turbulence and those numbers stop being valid. If the
confidence panel flags "laminar", widen channels, add parallel paths, or
reduce flow rate.

## Hydraulic diameter and friction

For a rectangular channel of width w and depth H, the hydraulic diameter is
Dh = 4A/P = 2wH/(w+H). Laminar friction obeys f·Re = fRe(α), a constant that
depends only on the aspect ratio α: 24 for parallel plates (α→0), 14.23 for a
square duct (α=1) — the Shah & London (1978) results NerveGear validates against.
Pressure drop for a channel: ΔP = 2·f·(L/Dh)·ρ·u², and in a network each
segment is a linear hydraulic resistor R = ΔP/Q, so the whole network solves
exactly like a resistor circuit (Kirchhoff nodal analysis).

## Nusselt number and wall superheat

Convection coefficient h = Nu·k_fluid/Dh. Fully developed laminar flow with
constant heat flux (the H1 condition) gives Nu = 8.235 for parallel plates
down to 3.61 for a square duct. Narrower Dh means higher h — the core reason
MICROchannels work at all. The wall superheat is ΔT = q_wall/h where q_wall
is the collected power divided by the wetted area (floor + fin-weighted side
walls).

## Fin efficiency

The silicon walls between channels act as fins: heat enters at the base and
convects off both faces. Efficiency η = tanh(mH)/(mH) with m = √(2h/(k·t)).
Tall thin walls (deep etches) collect more area but lose efficiency and raise
ΔP — one of the central trade-offs the optimizer negotiates.

## Coolant heat-up and energy balance

The first law fixes the coolant temperature rise: ΔT = P_total/(ṁ·cp).
No design choice can beat it — only more flow (higher ṁ) or colder inlet
lowers the outlet temperature. Downstream regions always run hotter;
serpentines concentrate this, parallel arrays dilute it. NerveGear's solvers
close this balance exactly and report the residual as a confidence check.

## Flow maldistribution

Channels fed by a manifold do not share flow equally: channels far from the
inlet see less driving pressure. NerveGear reports the coefficient of variation
(CV) of per-channel flow — above ~0.3 the starved channels run visibly
hotter. Fixes: wider headers, tapered manifolds, or Murray-law branching
trees that deliver flow evenly by construction.

## Murray's law and vein networks

Biological vasculature minimises pumping work when the parent radius cubed
equals the sum of child radii cubed (Murray's law, exponent 3). NerveGear's
branching generator follows w_child = w_parent / 2^(1/n). Trees distribute
coolant with low ΔP; the fine through-channels do the heat collection.

## Pressure drop, pumping power, and COP

ΔP is what the pump must supply; pumping power = ΔP·Q. The cooling COP here
is heat removed / pump work — NerveGear reports it so designs can be compared at
equal pumping budgets. Chasing the last few °C with narrower channels raises
ΔP steeply (R ∝ 1/w³ for fixed depth) — the optimizer treats ΔP as a hard
ceiling for exactly this reason.

## Manufacturing rules (DRIE etching)

Channels are etched with deep reactive-ion etching (DRIE, the Bosch process).
NerveGear enforces: minimum feature ≈ 20 µm (lithography/etch floor), aspect
ratio depth/width ≤ ~25 (etch stability), minimum wall ≈ 25 µm (mechanical
integrity), minimum channel ≈ 30 µm (particulate clogging). A design that
violates these cannot be fabricated no matter how good its thermal numbers —
they are hard constraints in the optimizer, and violations paint red in the
Studio.

## Floating silicon

An etch that fully encircles a solid island leaves it attached to nothing —
it would detach during fabrication. The design-rule checker detects islands
disconnected from the die body; break the moat or bridge the island.

## Keep-out zones and electrical co-design

Dies are not passive slabs: TSV arrays, power-delivery vias, and sensitive
analog blocks forbid etching above them. NerveGear's keep-out rectangles mark
these regions; channels crossing them are flagged as violations. This is the
first step toward full electro-thermal co-design, where routing and cooling
negotiate the same silicon area — the long-term goal being automatic rewiring
of the power/signal network around the coolant network and vice versa.

## The power map

Heat is not uniform: a GPU floorplan is ~20 W/cm² of background logic with
300–1000 W/cm² hotspots (SM clusters, SerDes, HBM PHYs). NerveGear's power map
(background + Gaussian hotspots, or an imported grid) drives all thermal
results. The highest-leverage design move is putting wetted area where the
flux is.

## The confidence score

The confidence score is the weighted fraction of physical-validity checks
that pass: laminar regime, energy-balance closure, geometry inside the
validated envelope (20 µm–2 mm channels), raster resolution (≥2–3 cells
across the narrowest channel), flow distribution, manufacturability, and
engine fidelity (reduced model vs full 3-D solver). It answers "should I
trust these numbers?", not "what is the error bar?" — read the failing
checks; they say exactly what to fix.

## NerveGear's engines

The Studio talks to one schema (axiom/solve.v1). Behind it: (1) the REDUCED
engine — exact laminar network hydraulics + Shah & London conjugate thermal
(this document's physics), instant and validated but approximate on local
fields; (2) the 3-D SOLVER (in progress) — voxel pressure/velocity/
temperature fields, slower but spatially exact; (3) the trained SURROGATE
(future) — a neural network distilled from the solver, giving 3-D-solver
answers at interactive speed. The response's `engine` field always says
which one produced the numbers.

## Reading the maps

Tj map: junction temperature — look for red islands between channels
(spreading-limited) vs red streaks along flow (heat-up-limited). Coolant map:
temperature inside channels only — a steep rise along one branch means it
carries too much of the load. Pressure map: drives the flow; steep local
gradients mark constrictions. Per-segment table: flow, velocity, Re, and ΔP
per channel — starved or stagnant branches show up here first.
