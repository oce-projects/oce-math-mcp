"""Counts are integers and the bounds of a multiple integral go from the innermost (D91).

Found by the audit of 25/11/2026: poisson_pmf(2.5, 1) and binom_pmf(1.5, 4, 0.5) gave a probability from the
gamma function; gcd, lcm, mod_inv and round truncated 12.5 to 12 with int(); quad with the bounds in the reversed
order gave 0.0 and multi_integrate gave "x" for the triangle 0 <= y <= x <= 1. Reference values from scipy.stats.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp
from scipy import stats

from oce_math_mcp.tools import math_solve
from oce_math_mcp.tools.math_solve import _numeric_multi
from tests.conftest import err, ok


def val(expr: str) -> float:
    return float(ok("math_eval", expr=expr, exact_mode="float")["value"])


def test_probability_mass_needs_integer_counts() -> None:
    assert math.isclose(val("poisson_pmf(2, 1.5)"), stats.poisson.pmf(2, 1.5), rel_tol=1e-12)
    assert math.isclose(val("poisson_pmf(3.0, 1.5)"), stats.poisson.pmf(3, 1.5), rel_tol=1e-12)
    assert val("poisson_pmf(-1, 1.5)") == 0.0
    assert (
        val("poisson_pmf(-1, 0)") == 0.0 and val("poisson_pmf(0, 0)") == 1.0
    )  # 0**(-1) / factorial(-1) = nan
    assert math.isclose(val("binom_pmf(2, 4, 0.5)"), stats.binom.pmf(2, 4, 0.5), rel_tol=1e-12)
    assert val("binom_pmf(5, 4, 0.5)") == 0.0 and val("binom_pmf(-1, 4, 0.5)") == 0.0
    # with p = 1 or p = 0 a k outside 0..n would give 0 * 0**(-1) = nan without the explicit 0
    assert val("binom_pmf(5, 4, 1)") == 0.0 and val("binom_pmf(-1, 4, 0)") == 0.0
    assert val("binom_pmf(4, 4, 1)") == 1.0
    for bad in (
        "poisson_pmf(2.5, 1)",
        "poisson_pmf(2, -1)",
        "binom_pmf(1.5, 4, 0.5)",
        "binom_pmf(2, 4.5, 0.5)",
        "binom_pmf(2, 4, 1.2)",
        "binom_pmf(2, 4, -0.1)",
        "binom_pmf(2, -4, 0.5)",
        "binom_cdf(2, 4, 1.5)",
        "poisson_cdf(2, -0.5)",
    ):
        assert err("math_eval", expr=bad) == "INVALID_ARGUMENT", bad


def test_cumulative_distributions_take_the_floor_of_k() -> None:
    assert math.isclose(val("poisson_cdf(2.5, 1)"), stats.poisson.cdf(2.5, 1), rel_tol=1e-12)
    assert val("poisson_cdf(-0.5, 1)") == 0.0 == stats.poisson.cdf(-0.5, 1)
    assert math.isclose(val("binom_cdf(1.7, 4, 0.5)"), stats.binom.cdf(1.7, 4, 0.5), rel_tol=1e-12)
    assert val("binom_cdf(10, 4, 0.5)") == 1.0
    assert val("binom_cdf(-1, 4, 0.5)") == 0.0 and val("binom_cdf(-0.5, 4, 0.5)") == 0.0
    assert val("binom_cdf(10, 4, 1)") == 1.0


def test_integer_functions_refuse_fractions() -> None:
    assert val("gcd(12, 18)") == 6 and val("gcd(12.0, 18)") == 6 and val("lcm(4, 6)") == 12
    assert val("mod_inv(3, 7)") == 5 and val("gcd(7)") == 7
    # a float from a scipy function: 12.0 is an integer, 12.5 is not
    assert val("gcd(t_cdf(0, 3)*24, 18)") == 6
    assert err("math_eval", expr="gcd(t_cdf(0, 3)*25, 5)") == "INVALID_ARGUMENT"
    assert ok("math_eval", expr="round(2.675, 2)")["value"] == "2.68"
    for bad in (
        "gcd(12.5, 5)",
        "gcd(7.5)",
        "lcm(4, 6.5)",
        "mod_inv(2.5, 7)",
        "mod_inv(3, 7.5)",
        "round(2.675, 2.5)",
    ):
        assert err("math_eval", expr=bad) == "INVALID_ARGUMENT", bad


def test_multi_integrate_uses_the_numerical_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(math_solve, "_numeric_multi", lambda *a: 2.0)
    r = ok("math_solve", operation="multi_integrate", expr="x*y", bounds=[["x", 0, 1], ["y", 0, 2]])
    assert r["integral"] == "1" and r["verified"] is False


def test_the_first_bound_is_the_innermost_integral() -> None:
    tri = [["y", 0, "x"], ["x", 0, 1]]
    assert float(ok("math_numeric", operation="quad", expr="1", bounds=tri)["value"]) == 0.5
    r = ok("math_solve", operation="multi_integrate", expr="1", bounds=tri)
    assert r["integral"] == "1/2" and r["verified"] is True
    # the back-check is a real numerical integral (before, evalf left an Integral and verified was always false)
    for expr, bounds, coords, exact in (
        ("x*y", [["x", 0, 1], ["y", 0, 2]], "cartesian", 1.0),
        ("1", [["r", 0, 1], ["theta", 0, "2*pi"]], "polar", math.pi),
        ("1", [["r", 0, 1], ["theta", 0, "pi"], ["phi", 0, "2*pi"]], "spherical", 4 * math.pi / 3),
        ("exp(-x**2 - y**2)", [["x", "-oo", "oo"], ["y", "-oo", "oo"]], "cartesian", math.pi),
    ):
        r = ok("math_solve", operation="multi_integrate", expr=expr, bounds=bounds, coords=coords)
        assert r["verified"] is True and math.isclose(r["numeric"], exact, rel_tol=1e-12), (expr, r)
    x, y = sp.symbols("x y")
    assert math.isclose(_numeric_multi(x * y, [(y, 0, x), (x, 0, 1)]), 1 / 8, rel_tol=1e-10)
    for wrong in ([["x", 0, 1], ["y", 0, "x"]], [["x", 0, "x"]], [["x", 0, 1], ["y", "x", 2]]):
        assert (
            err("math_numeric", operation="quad", expr="1", bounds=wrong) == "INVALID_ARGUMENT"
        ), wrong
        assert (
            err("math_solve", operation="multi_integrate", expr="1", bounds=wrong)
            == "INVALID_ARGUMENT"
        ), wrong
