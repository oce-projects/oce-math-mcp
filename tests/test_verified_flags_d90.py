"""The `verified` flags of math_solve and math_verify say what was checked (D90).

Found by the audit of 25/11/2026: `diff` gave `verified: false` for every derivative (the numerical reference
went through `complex()` inside `mpmath.diff`); `is_zero` took a difference of 1e-11 and a residual `zoo` as zero;
several operations set `verified: true` without any check (systems of ODEs, `polynomial` nroots, discriminant,
resultant, vieta, coeffs, degree; `partial_diff` compared the derivative with itself). The spec says `true` only if
the check succeeded, so an empty list of solutions is not verified either.
"""

from __future__ import annotations

import pytest
import sympy as sp

from oce_math_mcp.tools import math_solve
from oce_math_mcp.tools.common import is_zero
from oce_math_mcp.tools.math_solve import _check_derivative_numeric, _ode_system_check, _poly_verify
from tests.conftest import ok

x, y, z = sp.symbols("x y z")


def solve(**kw: object) -> dict:
    return ok("math_solve", **kw)


def test_diff_is_verified_by_the_numerical_derivative() -> None:
    for expr, order in (
        ("sin(x)", 1),
        ("x**3", 2),
        ("exp(x**2)", 1),
        ("log(x)", 3),
        ("x*y", 1),
        ("exp(2*x)", 12),
    ):
        r = solve(operation="diff", expr=expr, variables=["x"], order=order)
        assert r["verified"] is True, (expr, order, r)
    # the reference really compares: a wrong derivative fails, at first and higher order
    assert (
        _check_derivative_numeric(sp.sin(x), sp.cos(x) + sp.Rational(1, 10**9), [(x, 1)]) is False
    )
    assert _check_derivative_numeric(x**5, 20 * x**3 + x, [(x, 2)]) is False
    assert _check_derivative_numeric(x**5, 20 * x**3, [(x, 2)]) is True


def test_partial_diff_is_checked_numerically() -> None:
    r = solve(operation="partial_diff", expr="x**2*y**3", variables=["x", "y", "x"])
    assert r["derivative"] == "6*y**2" and r["verified"] is True
    r = solve(operation="partial_diff", expr="sin(x*y)*exp(z)", variables=["x", "y"])
    assert r["verified"] is True
    f = x**2 * y**3
    assert _check_derivative_numeric(f, 6 * y**2, [(x, 2), (y, 1)]) is True
    assert _check_derivative_numeric(f, 6 * y**2 + x, [(x, 2), (y, 1)]) is False
    assert _check_derivative_numeric(f, 6 * y, [(x, 2), (y, 1)]) is False


def test_is_zero_is_strict_on_exact_numbers_and_rejects_undefined() -> None:
    assert is_zero(sp.Rational(-1, 10**11)) is False
    assert is_zero(sp.sin(x) ** 2 + sp.cos(x) ** 2 - 1) is True
    assert is_zero(sp.sqrt(2) * sp.sqrt(3) - sp.sqrt(6)) is True
    for bad in (sp.zoo, sp.nan, sp.oo, x / x - sp.zoo):
        assert is_zero(bad) is False, bad
    r = ok(
        "math_verify", operation="check_derivative", expr="x", expected="1 + 1e-11", variables=["x"]
    )
    assert r["valid"] is False
    r = ok(
        "math_verify", operation="check_derivative", expr="x**2", expected="2*x", variables=["x"]
    )
    assert r["valid"] is True
    r = ok(
        "math_verify",
        operation="check_antiderivative",
        expr="x",
        expected="1 + 1e-11",
        variables=["x"],
    )
    assert r["valid"] is False


def test_check_solution_rejects_a_value_where_the_equation_is_undefined() -> None:
    r = ok("math_verify", operation="check_solution", expr="1/x = 1", solution={"x": "0"})
    assert r["valid"] is False and r["residuals"] == ["zoo"]
    r = ok("math_verify", operation="check_solution", expr="1/x = 1", solution={"x": "1"})
    assert r["valid"] is True


def test_ode_system_is_checked_by_substitution() -> None:
    r = solve(
        operation="ode",
        expr="y' = z; z' = -y",
        variables=["y", "z", "x"],
        ics={"y(0)": 0, "z(0)": 1},
    )
    assert r["verified"] is True and "sin(x)" in r["solution"]
    t = sp.Symbol("t")
    f, g = sp.Function("f"), sp.Function("g")
    eqs = [sp.Eq(f(t).diff(t), g(t)), sp.Eq(g(t).diff(t), -f(t))]
    assert _ode_system_check(eqs, [sp.Eq(f(t), sp.sin(t)), sp.Eq(g(t), sp.cos(t))]) is True
    assert _ode_system_check(eqs, [sp.Eq(f(t), sp.sin(t)), sp.Eq(g(t), sp.sin(t))]) is False


def test_polynomial_operations_are_checked() -> None:
    cases = (
        ("nroots", "x**2 - 2"),
        ("nroots", "x**3 - 6*x**2 + 11*x - 6"),
        ("discriminant", "x**2 - 2*x + 1"),
        ("discriminant", "a*x**2 + b*x + c"),
        ("vieta", "x**3 - 6*x**2 + 11*x - 6"),
        ("vieta", "2*x**2 + a*x - 3"),
        ("coeffs", "3*x**2 - x + 7"),
        ("degree", "a*x**4 + x"),
        ("resultant", "x**2 - 1; x**2 + 2*x + 1"),
        ("resultant", "x**2 + a; x - 3"),
    )
    for pop, expr in cases:
        r = solve(operation="polynomial", expr=expr, variables=["x"], poly_op=pop)
        assert r["verified"] is True, (pop, expr, r)
    a, b, c = sp.symbols("a b c")
    p = a * x**2 + b * x + c
    assert _poly_verify("discriminant", p, None, x, b**2 - 4 * a * c) is True
    assert _poly_verify("discriminant", p, None, x, b**2 - 2 * a * c) is False
    assert _poly_verify("resultant", x**2 + a, x - 3, x, a + 9) is True
    assert _poly_verify("resultant", x**2 + a, x - 3, x, a + 8) is False
    q = x**3 - 6 * x**2 + 11 * x - 6
    assert _poly_verify("vieta", q, None, x, {"e1": 6, "e2": 11, "e3": 6}) is True
    assert _poly_verify("vieta", q, None, x, {"e1": 6, "e2": 12, "e3": 6}) is False
    assert _poly_verify("nroots", x**2 - 2, None, x, [sp.sqrt(2), -sp.sqrt(2)]) is True
    assert (
        _poly_verify("nroots", x**2 - 2, None, x, [sp.Rational(141421, 100000), -sp.sqrt(2)])
        is False
    )
    assert _poly_verify("nroots", x**3 - 1, None, x, [1]) is False
    assert _poly_verify("coeffs", 3 * x**2 - x + 7, None, x, [3, -1, 7]) is True
    assert _poly_verify("coeffs", 3 * x**2 - x + 7, None, x, [3, 1, 7]) is False
    assert _poly_verify("degree", a * x**4 + x, None, x, 4) is True
    assert _poly_verify("degree", a * x**4 + x, None, x, 3) is False


def test_the_operations_use_the_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag comes from the check: a check that fails gives verified false."""
    monkeypatch.setattr(math_solve, "_check_derivative_numeric", lambda *a: False)
    monkeypatch.setattr(math_solve, "_ode_system_check", lambda *a: False)
    assert (
        solve(operation="partial_diff", expr="x**2*y**3", variables=["x", "y"])["verified"] is False
    )
    assert solve(operation="diff", expr="sin(x)", variables=["x"])["verified"] is False
    r = solve(
        operation="ode",
        expr="y' = z; z' = -y",
        variables=["y", "z", "x"],
        ics={"y(0)": 0, "z(0)": 1},
    )
    assert r["verified"] is False


def test_is_zero_does_not_take_a_tiny_exact_number_as_zero() -> None:
    assert is_zero(sp.Rational(1, 10**25)) is False
    assert is_zero(sp.Rational(1, 10**25) * x) is False
    # simplify gives the exact 1/10**25; the numeric fallback alone would see 1e-25 below its floor
    assert is_zero(sp.sin(1) ** 2 + sp.cos(1) ** 2 - 1 + sp.Rational(1, 10**25)) is False


def test_is_zero_is_not_zero_where_a_sample_point_is_undefined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When simplify cannot decide, a sample point where the value is nan means not zero (no proof)."""
    r = sp.Rational(3, 7)  # the first sample point of the fallback
    expr = sp.Add(sp.log(x - r), -sp.log(x - r), evaluate=False)
    assert expr.subs(x, r) is sp.nan
    monkeypatch.setattr(sp, "simplify", lambda e: e)
    assert is_zero(expr) is False


def test_an_empty_solution_list_is_not_verified() -> None:
    r = solve(operation="solve", expr="x**2 + 1 = 0")
    assert r["solutions"] == [] and r["verified"] is False
    r = solve(operation="solve", expr="x**2 - 4 = 0")
    assert r["verified"] is True
    r = solve(operation="solve_system", expr="x + y = 1; x + y = 2", variables=["x", "y"])
    assert r["solutions"] == [] and r["verified"] is False
