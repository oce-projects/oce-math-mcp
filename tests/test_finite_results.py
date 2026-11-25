"""Results that overflow are an error, not an ok answer with "oo" or "nan" (D74, every eng_calc operation).

The cases come from a random exploration with extreme inputs (1e300, 1e-300): before the check each of them returned
ok with a non-finite number. No operation returns an intended infinity (the code filters them, for example C_max of a
heat exchanger with a phase change), and the whole suite passes with the check.
"""

from __future__ import annotations

import pytest

from oce_math_mcp.tools import eng_base
from tests.conftest import call

CASES = [
    ("electric_machine", {"mode": "dc_motor", "V_m": 1e300, "R_m": 2.701, "k": 2.192}),
    ("electric_machine", {"mode": "induction_motor", "f": 1e300, "poles": 10, "s": 1e9}),
    ("electric_machine", {"mode": "synchronous_machine", "V": 1e300, "E": 1e300, "X_s": 2.0}),
    ("electronics", {"mode": "led_resistor", "V_s": 2.493, "V_f": 1.385, "I": 1e300}),
    ("friction", {"mode": "leadscrew", "F": 6.298, "d_m": 1e300, "l": 1.018, "mu": 4.905}),
    (
        "local_loss",
        {"mode": "sudden_contraction", "rho": 2.925, "D1": 1e300, "A2": 1.194, "Q": 66.34},
    ),
    (
        "local_loss",
        {"mode": "sudden_expansion", "rho": 5.536, "D1": 2.37e-7, "A2": 1e300, "Q": 2.036},
    ),
    (
        "plate_bending",
        {
            "shape": "circle",
            "support": "clamped",
            "h": 4.289,
            "E": 1e-300,
            "nu": 0.389,
            "q": 1e300,
            "a": 0.01115,
        },
    ),
    (
        "plate_bending",
        {
            "shape": "rectangle",
            "support": "simply_supported",
            "h": 30632.0,
            "E": 0.4999,
            "nu": 0.282,
            "q": 5.7,
            "a": 3.531,
            "b": 1e-300,
        },
    ),
    ("rolling", {"mode": "incline", "m": 1e300, "r": 3.692, "r_g": 15573819.5, "theta_deg": 3.798}),
    ("rotor_balance", {"mode": "iso_grade", "G": 1e9, "M": 1e300, "n_rpm": 0.567}),
    (
        "rotor_balance",
        {"mode": "unbalance", "masses": [{"m": 1e9, "r": 16524.29, "angle_deg": 3e-8, "z": 1e300}]},
    ),
    ("stress_concentration", {"mode": "elliptical_hole", "sigma": 5.82, "a": 3.73, "b": 1e300}),
    (
        "continuous_beam",
        {"spans": [{"L": 1000000001.0, "w": 1.329e-5, "P": [[1e300, 0.5]]}], "EI": 3.8775},
    ),
    (
        "geometric_optics",
        {"mode": "refracting_surface", "n1": 1e300, "n2": 1e-9, "R": 2.679, "d_i": 3.055},
    ),
    ("geometric_optics", {"mode": "two_lenses", "f1": 1e300, "f2": 1e9}),
    # older operations: the check covers every eng_calc operation
    ("acid_base", {"mode": "weak", "kind": "acid", "c": 1e300, "Ka": 1.387}),
    ("modern_physics", {"mode": "compton", "energy": 1e300, "angle_deg": 2.6118882725549573}),
]


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # numpy overflow on the extreme inputs is expected
@pytest.mark.parametrize(("operation", "params"), CASES)
def test_overflow_is_an_error(operation: str, params: dict) -> None:
    resp = call("eng_calc", operation=operation, params=params)
    assert not resp["ok"], resp["result"]
    assert resp["error"]["code"] == "MATH_ERROR" and "not finite" in resp["error"]["message"], resp[
        "error"
    ]


def test_nested_values() -> None:
    # a non-finite value hidden in a list (support moments of a beam, for example) is found too
    assert not eng_base._all_finite({"M_supports": [0.0, float("inf"), 0.0]})
    assert not eng_base._all_finite({"x": [[1.0, float("nan")]]})
    assert eng_base._all_finite({"M_supports": [0.0, -2.5], "where": "edge", "ok": True, "n": 3})
