"""Extreme inputs of kinetic_theory and gravitation give the error of D74, not a Python exception (D95).

The robustness exploration of 25/11/2026 found 119 cases in 11 groups (inputs like 1e-300 and 1e300) where
the answer was the bare text "ZeroDivisionError: float division by zero" or an OverflowError. Every division
in these two operations is by a positive physical size, so the error can only come from the floating-point
range; the singular point of a field (a test point on a point mass) keeps its own INVALID_ARGUMENT.
"""

from __future__ import annotations

import pytest

from oce_math_mcp.errors import NOT_FINITE
from tests.conftest import call

CASES = [
    ("gravitation", {"mode": "potential_difference", "M": 4.94, "r1": 1e-300, "r2": 0.616}),
    ("gravitation", {"mode": "ring", "M": 1000, "R": 1e-300, "z": 1e-300}),
    ("gravitation", {"mode": "sphere", "M": 5.97e24, "R": 1e-300, "r": 1e-300}),
    ("gravitation", {"mode": "sphere", "rho": 1.442, "R": 1e300, "r": 1e300}),
    ("gravitation", {"mode": "two_bodies", "m1": 1, "m2": 1, "r": 1e-300}),
    (
        "gravitation",
        {"mode": "rotating_body", "M": 5.97e24, "R": 1e-300, "omega": 7.29e-5, "latitude_deg": 45},
    ),
    (
        "gravitation",
        {"mode": "rotating_body", "M": 5.97e24, "R": 1e300, "omega": 7.29e-5, "latitude_deg": 45},
    ),
    ("kinetic_theory", {"mode": "speeds", "T": 300, "M_molar": 1e-300}),
    ("kinetic_theory", {"mode": "mean_free_path", "T": 300, "p": 101325, "d": 1e300}),
    (
        "kinetic_theory",
        {"mode": "mean_free_path", "T": 300, "p": 101325, "mu": 1.8e-5, "M_molar": 1e-300},
    ),
    ("kinetic_theory", {"mode": "wall_flux", "T": 300, "p": 101325, "M_molar": 1e-300}),
]


@pytest.mark.parametrize(("op", "params"), CASES)
def test_extreme_inputs_give_the_finite_range_error(op: str, params: dict) -> None:
    resp = call("eng_calc", operation=op, params=params)
    assert not resp["ok"] and resp["error"]["code"] == "MATH_ERROR", resp
    assert resp["error"]["message"] == NOT_FINITE, resp["error"]


def test_singular_point_keeps_its_own_message() -> None:
    resp = call(
        "eng_calc",
        operation="gravitation",
        params={"mode": "point_masses", "masses": [[1, 0, 0, 0]], "point": [0, 0, 0]},
    )
    assert resp["error"]["code"] == "INVALID_ARGUMENT" and "coincides" in resp["error"]["message"]
