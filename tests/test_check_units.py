"""math_verify.check_units (D42): checking the units of a free formula.

Independent methods: SI dimensions written by hand from the definitions (BIPM, SI Brochure 9th ed.);
scale factors checked with convert_to of sympy.physics.units on the same quantities.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import sympy.physics.units as u
from sympy.physics.units import convert_to

from tests.conftest import call, ok


def cu(expr: str, units: dict[str, str], **extra: Any) -> dict[str, Any]:
    return ok("math_verify", operation="check_units", expr=expr, units=units, **extra)


def resp(expr: str, units: dict[str, str], **extra: Any) -> dict[str, Any]:
    return call("math_verify", operation="check_units", expr=expr, units=units, **extra)


# dimensions from the SI definitions: (length, mass, time, current, temperature, amount, intensity)
DEF = {
    "N": {"length": 1, "mass": 1, "time": -2},
    "J": {"length": 2, "mass": 1, "time": -2},
    "W": {"length": 2, "mass": 1, "time": -3},
    "Pa": {"length": -1, "mass": 1, "time": -2},
    "V": {"length": 2, "mass": 1, "time": -3, "current": -1},
    "ohm": {"length": 2, "mass": 1, "time": -3, "current": -2},
    "F": {"length": -2, "mass": -1, "time": 4, "current": 2},
    "H": {"length": 2, "mass": 1, "time": -2, "current": -2},
    "Hz": {"time": -1},
    "kN": {"length": 1, "mass": 1, "time": -2},
    "MPa": {"length": -1, "mass": 1, "time": -2},
    "mA": {"current": 1},
    "km/h": {"length": 1, "time": -1},
    "m2": {"length": 2},
    "kg*m/s^2": {"length": 1, "mass": 1, "time": -2},
    "N*m": {"length": 2, "mass": 1, "time": -2},
    "rpm": {"time": -1},
    "degC": {"temperature": 1},
    "rad": {},
    "deg": {},
    "1": {},
}


@pytest.mark.parametrize("unit", sorted(DEF))
def test_unit_dimensions_match_si_definitions(unit: str) -> None:
    r = cu("x", {"x": unit})
    assert r["consistent"] is True and r["dimension"] == DEF[unit]


@pytest.mark.parametrize(
    ("unit", "sym"),
    [
        ("kN", 1000 * u.newton),
        ("MPa", 10**6 * u.pascal),
        ("km/h", u.kilometer / u.hour),
        ("bar", u.bar),
        ("kWh", 1000 * u.watt * u.hour),
        ("mm2", u.millimeter**2),
        ("psi", u.psi),
        ("L/min", u.liter / u.minute),
    ],
)
def test_scale_factor_matches_sympy_convert_to(unit: str, sym: Any) -> None:
    r = cu("x", {"x": unit})
    si = convert_to(sym, [u.kilogram, u.meter, u.second, u.ampere, u.kelvin]).as_coeff_Mul()[0]
    assert math.isclose(r["scale_to_si"], float(si), rel_tol=1e-12)


def test_drag_force_right_and_wrong() -> None:
    units = {"F": "N", "rho": "kg/m^3", "v": "m/s", "Cd": "1", "A": "m^2"}
    good = cu("F = 0.5*rho*v**2*Cd*A", units)
    assert good["consistent"] is True and good["dimension"] == DEF["N"] and good["problems"] == []
    bad = cu("F = rho*v*A", units)  # mass flow rate (kg/s), not a force
    assert bad["consistent"] is False and bad["problems"]
    assert bad["lhs_dimension"] == DEF["N"] and bad["rhs_dimension"] == {"mass": 1, "time": -1}
    assert bad["scale_ratio"] is None


def test_expression_with_expected_unit() -> None:
    units = {"p": "Pa", "rho": "kg/m3", "v": "m/s", "g": "m/s2", "h": "m"}
    r = cu("p + 0.5*rho*v**2 + rho*g*h", units, expected="kPa")
    assert r["consistent"] is True and r["expected_ok"] is True and r["dimension"] == DEF["Pa"]
    assert cu("rho*g*h", units, expected="N")["expected_ok"] is False


def test_sum_of_different_dimensions() -> None:
    r = cu("x + t", {"x": "m", "t": "s"})
    assert r["consistent"] is False and any("sum" in p for p in r["problems"])


def test_function_arguments_must_be_dimensionless() -> None:
    assert cu("sin(w*t)", {"w": "rad/s", "t": "s"})["consistent"] is True
    r = cu("exp(t)", {"t": "s"})
    assert r["consistent"] is False and r["problems"]


def test_degrees_inside_sin_warn() -> None:
    res = resp("sin(theta)", {"theta": "deg"})
    assert res["ok"] and res["result"]["consistent"] is True
    assert any("degrees" in w or "radians" in w for w in res["warnings"])


def test_same_dimension_different_scale() -> None:
    res = resp("v = d/t", {"v": "km/h", "d": "m", "t": "s"})
    r = res["result"]
    assert r["consistent"] is True and math.isclose(r["scale_ratio"], 3.6, rel_tol=1e-12)
    assert res["warnings"]
    assert cu("v = d/t", {"v": "m/s", "d": "m", "t": "s"})["scale_ratio"] == 1.0


def test_symbolic_exponent_on_dimensioned_base() -> None:
    r = cu("x**n", {"x": "m", "n": "1"})
    assert r["consistent"] is False and r["problems"]
    assert cu("x**2", {"x": "m"})["dimension"] == {"length": 2}


def test_coulomb_vs_celsius_warning() -> None:
    res = resp("T", {"T": "C"})
    assert res["ok"] and res["result"]["dimension"] == {"current": 1, "time": 1}
    assert any("degC" in w for w in res["warnings"])


@pytest.mark.parametrize(
    ("expr", "units", "code"),
    [
        ("F = m*a", {"F": "N", "m": "kg"}, "INVALID_ARGUMENT"),  # the unit of a is missing
        ("x", {"x": "furlong"}, "INVALID_ARGUMENT"),  # unknown unit
        ("x", {"x": "__import__('os')"}, "INVALID_ARGUMENT"),
        ("x", {"x": "m**(1/0)"}, "INVALID_ARGUMENT"),
        ("x", {"x": ""}, "INVALID_ARGUMENT"),
    ],
)
def test_bad_inputs(expr: str, units: dict[str, str], code: str) -> None:
    r = resp(expr, units)
    assert r["ok"] is False and r["error"]["code"] == code


def test_units_field_required() -> None:
    r = call("math_verify", operation="check_units", expr="x")
    assert r["ok"] is False and r["error"]["code"] == "INVALID_ARGUMENT"
