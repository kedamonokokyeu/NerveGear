"""
contract/ — the axiom/solve.v1 JSON contract, as code.

This is the one schema the Studio and the backend agree on: geometry and
conditions in, metrics/confidence/maps out. It lived in the model3d package
until 2026-07, when the 3-D engine was removed to be rebuilt from scratch
later; the contract survives here because the backend cannot parse a solve
request without it. Change only with a version bump.
"""
