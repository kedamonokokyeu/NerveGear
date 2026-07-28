"""
micro/design.py

Constrained multi-objective optimization for microchannel heat sinks.

Pressure drop becomes a hard constraint here — a thermally greedy search otherwise
drives channels infinitely narrow. Wires the microchannel evaluator into the
existing physics/optimize.py framework. No new optimizer; the seam was already
there.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from . import units
from . import manufacturing as mfg
from .conjugate import MicroChannelHeatSink, ConjugateModel
from .manufacturing import EtchRules
from .powermap import PowerMap, uniform


@dataclass
class MicroDesign:
    base: MicroChannelHeatSink = dc_field(default_factory=MicroChannelHeatSink)
    power: PowerMap = dc_field(default_factory=lambda: uniform(150.0))
    rules: EtchRules = dc_field(default_factory=EtchRules)
    dP_limit_kPa: float = 50.0            # pump ceiling
    junction_limit_c: float = 105.0


def make_evaluator(design: MicroDesign):
    """Return evaluator(params)->metrics combining the conjugate thermal model with
    manufacturability-derived metrics, so constraints can reference either."""
    model = ConjugateModel(design.base, design.power)

    def evaluate(params: dict) -> dict:
        from dataclasses import replace
        m = dict(model.evaluate(params))
        hs = replace(
            design.base,
            w_ch=float(params.get("w_ch", design.base.w_ch)),
            H_ch=float(params.get("H_ch", design.base.H_ch)),
            n_ch=int(round(params.get("n_ch", design.base.n_ch))),
        )
        w_um = units.to_um(hs.w_ch); wall_um = units.to_um(hs.w_wall); H_um = units.to_um(hs.H_ch)
        viols = mfg.check_heatsink(hs, design.rules)
        m["aspect_ratio_AR"] = H_um / max(w_um, 1e-9)
        m["min_wall_um"] = wall_um
        m["min_channel_um"] = w_um
        m["manufacturable"] = 1.0 if all(v.ok for v in viols) else 0.0
        m["clogging_risk"] = mfg.clogging_risk(hs)
        return m

    return evaluate


def build_spec(design: MicroDesign):
    """Construct the OptimizationSpec (objectives + constraints + parameter box)."""
    from micro.optimize import OptimizationSpec, Parameter, Objective, Constraint
    r = design.rules
    parameters = [
        Parameter("w_ch", units.um(r.min_feature_um), units.um(400.0)),
        Parameter("H_ch", units.um(50.0), units.um(1200.0)),
        Parameter("n_ch", 10, max(11, int(design.base.die_W / units.um(60.0)))),
        Parameter("flow_mlmin", 50.0, 1000.0),
    ]
    objectives = [
        Objective("max_junction_temp", "min"),
        Objective("junction_gradient", "min"),   # equal-weighted; reweight later
    ]
    constraints = [
        Constraint("pressure_drop_kPa", "<=", design.dP_limit_kPa),
        Constraint("aspect_ratio_AR", "<=", r.max_aspect_ratio),
        Constraint("min_wall_um", ">=", r.min_wall_um),
        Constraint("min_channel_um", ">=", r.min_channel_um),
    ]
    return OptimizationSpec(parameters=parameters, objectives=objectives, constraints=constraints)


def optimize(design: MicroDesign, n_iter: int = 300, seed: int = 0):
    """Run the constrained search. Returns physics.optimize.OptResult."""
    from micro.optimize import random_search
    spec = build_spec(design)
    return random_search(spec, make_evaluator(design), n_iter=n_iter, seed=seed)
