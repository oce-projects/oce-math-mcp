"""D101: math_verify.check_solution with derivatives of the unknowns.

Before D101 the parser computed diff(y, x) on the bare symbol y (= 0) before the solution was put in, so any
function "solved" the wave equation and y' - y = 0 lost its derivative.
"""

from __future__ import annotations

from tests.conftest import call

WAVE = "diff(A, x, 2) + diff(A, y, 2) + diff(A, z, 2) - diff(A, t, 2)/v**2 = 0"


def check(expr: str, solution: dict) -> dict:
    return call("math_verify", operation="check_solution", expr=expr, solution=solution)


def test_ordinary_differential_equation() -> None:
    r = check("diff(y, x) - y = 0", {"y": "exp(x)"})
    assert r["ok"] and r["result"]["valid"] is True and r["result"]["residuals"] == ["0"]
    r = check("diff(y, x) - y = 0", {"y": "x**2"})
    assert r["ok"] and r["result"]["valid"] is False
    assert r["result"]["residuals"] != ["0"]
    r = check("diff(y, x, 2) + y = 0", {"y": "3*sin(x) - cos(x)"})
    assert r["result"]["valid"] is True


def test_wave_equation_needs_the_right_speed() -> None:
    assert check(WAVE, {"A": "exp(-(x - v*t)**2)"})["result"]["valid"] is True
    assert check(WAVE, {"A": "sin(x + v*t)"})["result"]["valid"] is True
    assert check(WAVE, {"A": "exp(-(x - 2*v*t)**2)"})["result"]["valid"] is False
    assert check(WAVE, {"A": "x*t"})["result"]["valid"] is True  # A_xx = A_tt = 0
    assert check(WAVE, {"A": "x**2*t"})["result"]["valid"] is False


def test_kdv_soliton() -> None:
    sol = {"y": "3*c/a/cosh(sqrt(c)/2*(x - c*t))**2"}
    r = check("diff(y, t) + a*y*diff(y, x) + diff(y, x, 3) = 0", sol)
    assert r["ok"] and r["result"]["valid"] is True
    sol = {"y": "3*c/a/cosh(sqrt(c)/2*(x + c*t))**2"}  # travelling the wrong way
    assert check("diff(y, t) + a*y*diff(y, x) + diff(y, x, 3) = 0", sol)["result"]["valid"] is False


def test_undefined_value_and_derivative_with_a_division() -> None:
    assert check("diff(y, x) = 1/x", {"y": "log(x)"})["result"]["valid"] is True
    assert check("diff(y, x) = 1/x", {"y": "log(2*x)"})["result"]["valid"] is True
    assert check("diff(y, x) = 1/x", {"y": "x*log(x)"})["result"]["valid"] is False
    r = check("diff(y, x) - 1/z = 0", {"y": "x", "z": "0"})
    assert r["ok"] and r["result"]["valid"] is False and r["result"]["residuals"] == ["zoo"]


def test_residual_is_simplified_before_the_missing_names() -> None:
    r = check("y = sin(x)**2 + cos(x)**2", {"y": "1"})
    assert r["ok"] and r["result"] == {"valid": True, "residuals": ["0"]}
    # z cancels only after simplification: no value of z is needed
    r = check("y + sin(z)**2 + cos(z)**2 = 1 + x", {"y": "x"})
    assert r["ok"] and r["result"] == {"valid": True, "residuals": ["0"]}


def test_algebraic_equations_unchanged() -> None:
    r = check("y**2 - 4 = 0", {"y": "3"})
    assert r["result"] == {"valid": False, "residuals": ["5"]}
    assert check("y**2 - 4 = 0", {"y": "-2"})["result"]["valid"] is True
    r = check("x + y = 3; x - y = 1", {"x": "2", "y": "1"})
    assert r["result"]["valid"] is True
    r = check(
        "y - 4 = 0", {"x": "2", "y": "x**2"}
    )  # a value may not use another name of the solution
    assert r["ok"] is False and r["error"]["code"] == "INVALID_ARGUMENT"
    r = check("x + y = 3", {"x": "1"})  # a value is missing: still an error
    assert r["ok"] is False and r["error"]["code"] == "INVALID_ARGUMENT"
