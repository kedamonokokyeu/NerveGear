"""
micro

The microscale thermal layer. Sits beside physics/ and reuses its optimization
framework. Covers laminar microchannel hydraulics, conjugate heat transfer,
manufacturing constraints, and constrained multi-objective optimization — all in
real engineering units.

Relationship to the high-fidelity model (your friend's work): this layer is the
fast analytical baseline and ground-truth/sanity check. The trained LBM/conjugate
surrogate plugs in by replacing `conjugate.ConjugateModel.evaluate` — the
optimizer, metrics and UI are untouched, exactly like NerveGear's existing
`velocity_provider` injection.
"""

from . import (
    confidence,
    conjugate,
    correlations,
    design,
    geometry,
    manufacturing,
    materials,
    network_thermal,
    powermap,
    units,
    validate,
)

__all__ = [
    "confidence", "conjugate", "correlations", "design", "geometry",
    "manufacturing", "materials", "network_thermal", "powermap", "units",
    "validate",
]
