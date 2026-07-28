"""
physics/optimize.py

Optimization framework for the workspace. Varies design parameters, scores each
candidate with an injected physics evaluator, enforces constraints, and keeps
the best.

The evaluator is injected — today it's a stub, later you wire in the
CAD → mask → surrogate → metrics pipeline. Search is plain random search with a
seed, simple enough to swap for Bayesian or gradient-based later without changing
the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np


@dataclass
class Parameter:
    name: str
    low: float
    high: float

    def sample(self, rng) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class Objective:
    metric: str
    direction: str = "min"      # "min" or "max"


@dataclass
class Constraint:
    metric: str
    op: str                     # "<=" or ">="
    bound: float

    def satisfied(self, value: float) -> bool:
        return value <= self.bound if self.op == "<=" else value >= self.bound

    def violation(self, value: float) -> float:
        """How far (>=0) the constraint is violated; 0 if satisfied."""
        if self.satisfied(value):
            return 0.0
        return abs(value - self.bound)


@dataclass
class OptimizationSpec:
    parameters: list
    objectives: list
    constraints: list = dc_field(default_factory=list)


@dataclass
class OptResult:
    best_params: dict
    best_metrics: dict
    best_score: float
    feasible: bool
    history: list               # list of {params, metrics, score, feasible}


def _score(metrics: dict, objectives: list) -> float:
    """Combine objectives into a single number to MINIMIZE (maximize objectives
    are negated). Equal-weighted; extend with weights as needed."""
    total = 0.0
    for obj in objectives:
        v = float(metrics[obj.metric])
        total += v if obj.direction == "min" else -v
    return total


def random_search(spec: OptimizationSpec, evaluator, n_iter: int = 100, seed: int = 0) -> OptResult:
    """Sample `n_iter` parameter sets, evaluate each, and keep the best FEASIBLE
    one (or, if none are feasible, the one with the least constraint violation).

    evaluator: callable(params: dict[str, float]) -> dict[str, float] of metrics.
               THIS is where the physics surrogate plugs in later.
    """
    rng = np.random.default_rng(seed)
    history = []
    best = None  # (feasible, key, params, metrics, score)

    for _ in range(n_iter):
        params = {p.name: p.sample(rng) for p in spec.parameters}
        metrics = evaluator(params)
        violation = sum(c.violation(float(metrics[c.metric])) for c in spec.constraints)
        feasible = violation == 0.0
        score = _score(metrics, spec.objectives)
        history.append({"params": params, "metrics": metrics, "score": score, "feasible": feasible})

        # ranking key: prefer feasible; among feasible, lowest score; among
        # infeasible, least violation then score.
        key = (0 if feasible else 1, score if feasible else violation)
        if best is None or key < best[0]:
            best = (key, params, metrics, score, feasible)

    _, p, m, s, f = best
    return OptResult(best_params=p, best_metrics=m, best_score=s, feasible=f, history=history)
