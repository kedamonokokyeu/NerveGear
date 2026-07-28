"""The axiom/solve.v1 contract: parsing, validation errors, array codecs."""

import os
import sys

import numpy as np
import pytest

# The contract moved in-package when the 3-D engine was removed (2026-07).
from contract import schema


def test_example_request_parses():
    req = schema.parse_request(schema.example_request())
    assert req.geometry.parametric["kind"] == "straight_parallel"
    assert req.conditions.flow_mlmin == 300.0


def test_array_roundtrip():
    a = np.random.rand(37, 53).astype(np.float32) * 500
    block = schema.encode_array(a)
    b = schema.decode_array(block)
    assert np.allclose(a, b)


def test_errors_carry_paths():
    with pytest.raises(schema.SchemaError, match=r"\$\.geometry\.die\.L"):
        schema.parse_request({"geometry": {"die": {"L": -1}}})
    with pytest.raises(schema.SchemaError, match="no channels"):
        schema.parse_request({"geometry": {"die": {"L": 10000}}})
    with pytest.raises(schema.SchemaError, match="exceeds"):
        e = schema.encode_array(np.full((8, 8), 4000.0, dtype=np.float32))
        schema.parse_request({"geometry": {"die": {"L": 10000, "thk": 500, "base_thk": 100},
                                           "etch": e}})
