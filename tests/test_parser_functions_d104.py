"""D104: the 10 functions the parser did not read, and sums that cancel in compile_expr and compile_mp.

New names: sech, csch, acoth, asech, acsch, asec, acsc and the orthogonal polynomials legendre(n, x),
hermite(n, x), laguerre(n, x) (SymPy: with an integer degree they are expanded polynomials).
Measured on 2026-11-26 before D104: compile_expr evaluated the expanded polynomials in floating point with an
error, relative to the largest value on the natural domain, of 3.7e-10 for legendre(20, x), 43 for degree 50
and 9.2e19 for degree 100 (hermite 100: 5.2e-5, laguerre 100: 4.9e8): the terms cancel. The references here
are mpmath at 40-60 digits and numpy's Gauss-Legendre nodes.
"""

from __future__ import annotations

import math
import time
from typing import Any

import mpmath
import numpy as np
import pytest
import sympy as sp

from oce_math_mcp import limits
from oce_math_mcp.errors import ToolError
from oce_math_mcp.expr import parse
from oce_math_mcp.numfunc import compile_expr, compile_mp
from oce_math_mcp.tools.common import grammar
from tests.conftest import err, ok

x, n, t = sp.symbols("x n t")
G = grammar(["x", "n", "t"])

HYP = {
    "sech": sp.sech,
    "csch": sp.csch,
    "acoth": sp.acoth,
    "asech": sp.asech,
    "acsch": sp.acsch,
    "asec": sp.asec,
    "acsc": sp.acsc,
}
INSIDE = {  # points of the real domain, the tails included
    "sech": [-800.0, -3.0, -0.5, 0.0, 1e-9, 2.0, 30.0, 800.0],
    "csch": [-800.0, -3.0, -1e-9, 1e-9, 0.5, 2.0, 30.0, 800.0],
    "acoth": [-1e300, -1e6, -3.0, -1.5, 1.5, 7.0, 1e8, 1e300],
    "asech": [1e-300, 1e-9, 0.25, 0.5, 0.75],
    "acsch": [-1e300, -2.0, -1e-9, 1e-9, 0.5, 3.0, 1e300],
    "asec": [-1e300, -2.0, -1.5, 1.5, 3.0, 1e300],
    "acsc": [-1e300, -2.0, -1.5, 1.5, 3.0, 1e300],
}
OUTSIDE = {  # complex values or poles in SymPy: nan
    "sech": [],
    "csch": [0.0],
    "acoth": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "asech": [-0.5, 0.0, 1.5],
    "acsch": [0.0],
    "asec": [-0.5, 0.0, 0.5],
    "acsc": [-0.5, 0.0, 0.5],
}


def p(text: str) -> sp.Expr:
    return parse(text, G)


# ------------------------------------------------------------------ the seven hyperbolic and inverse functions


def test_the_parser_reads_the_seven_functions() -> None:
    for name, fn in HYP.items():
        assert p(f"{name}(x)") == fn(x), name
    assert p("asec(2)") == sp.pi / 3 and p("acsc(2)") == sp.pi / 6 and p("sech(0)") == 1


@pytest.mark.parametrize("name", sorted(HYP))
def test_compile_expr_matches_mpmath(name: str) -> None:
    f = compile_expr(HYP[name](x), [x])
    ref = getattr(mpmath, name)
    mpmath.mp.dps = 40
    for v in INSIDE[name]:
        exact = ref(mpmath.mpf(v))
        got = f(v)
        assert abs(mpmath.mpf(got) - exact) <= mpmath.mpf(1e-15) * abs(exact) + mpmath.mpf(
            1e-320
        ), (
            name,
            v,
            got,
            exact,
        )
    for v in OUTSIDE[name]:
        assert math.isnan(f(v)), (name, v, f(v))


@pytest.mark.parametrize("name", sorted(HYP))
def test_compile_mp_matches_mpmath(name: str) -> None:
    mpmath.mp.dps = 30
    f = compile_mp(HYP[name](x), [x], 30)
    ref = getattr(mpmath, name)
    for v in INSIDE[name]:
        got, exact = f(mpmath.mpf(v)), ref(mpmath.mpf(v))
        assert abs(got - exact) <= mpmath.mpf(10) ** -28 * abs(exact) + mpmath.mpf(10) ** -320, (
            name,
            v,
        )


def test_the_tools_use_the_seven_functions() -> None:
    mpmath.mp.dps = 30
    value = ok("math_eval", expr="sech(1/2) + csch(2) + acoth(3) + asech(1/3) + acsch(4)")
    exact = mpmath.sech(0.5) + mpmath.csch(2) + mpmath.acoth(3) + mpmath.asech(mpmath.mpf(1) / 3)
    exact += mpmath.acsch(4)
    assert abs(value["float"] - float(exact)) < 1e-15
    r = ok("math_solve", operation="diff", expr="asec(x) + acoth(x)", variables=["x"])
    assert sp.simplify(p(r["derivative"]) - sp.diff(sp.asec(x) + sp.acoth(x), x)) == 0


# ------------------------------------------------------------------ orthogonal polynomials


def test_polynomials_are_expanded_and_a_symbolic_degree_stays() -> None:
    assert p("legendre(3, x)") == sp.Rational(5, 2) * x**3 - sp.Rational(3, 2) * x
    assert p("hermite(3, x)") == 8 * x**3 - 12 * x
    assert p("laguerre(2, x)") == x**2 / 2 - 2 * x + 1
    assert p("legendre(n, x)") == sp.legendre(n, x)
    assert p("legendre(-3, x)") == p("legendre(2, x)")  # P_(-n-1) = P_n


def test_degree_limit_and_bad_degrees() -> None:
    top = limits.POLY_DEGREE_MAX
    t0 = time.perf_counter()
    for name in ("legendre", "hermite", "laguerre"):
        assert sp.Poly(p(f"{name}({top}, x)"), x).degree() == top
        for deg in (top + 1, -top - 2):
            with pytest.raises(ToolError) as exc:
                p(f"{name}({deg}, x)")
            assert exc.value.code == "LIMIT_EXCEEDED", (name, deg)
    assert time.perf_counter() - t0 < 5
    assert p("laguerre(-2, x)") == (x + 1) * sp.exp(x)  # SymPy: L_(-n) = exp(x) L_(n-1)(-x)
    for bad in ("hermite(-1, x)", "laguerre(5/2, x)"):
        with pytest.raises(ToolError) as exc:
            p(bad)
        assert exc.value.code == "MATH_ERROR", bad


def test_orthogonality_through_math_solve() -> None:
    cases = [
        ("legendre(3, x)*legendre(5, x)", [-1, 1], 0),
        ("legendre(4, x)**2", [-1, 1], sp.Rational(2, 9)),  # 2 / (2 n + 1)
        ("hermite(3, x)**2*exp(-x**2)", ["-oo", "oo"], 48 * sp.sqrt(sp.pi)),  # sqrt(pi) 2^n n!
        ("laguerre(4, x)**2*exp(-x)", [0, "oo"], 1),
    ]
    for expr, bounds, exact in cases:
        r = ok("math_solve", operation="integrate", expr=expr, variables=["x"], bounds=bounds)
        assert sp.simplify(p(r["integral"]) - exact) == 0, (expr, r["integral"])


# ------------------------------------------------------------------ sums that cancel


def _domain(name: str, deg: int) -> tuple[float, float]:
    if name == "legendre":
        return -1.0, 1.0
    if name == "hermite":
        return -math.sqrt(2 * deg + 1), math.sqrt(2 * deg + 1)
    return 0.0, 4.0 * deg + 2


def _ref(name: str, deg: int, v: float) -> Any:
    if name == "laguerre":
        return mpmath.laguerre(deg, 0, mpmath.mpf(v))
    return getattr(mpmath, name)(deg, mpmath.mpf(v))


@pytest.mark.parametrize("name", ["legendre", "hermite", "laguerre"])
def test_high_degree_polynomials_evaluate_accurately(name: str) -> None:
    deg = 100
    f = compile_expr(p(f"{name}({deg}, x)"), [x])
    lo, hi = _domain(name, deg)
    mpmath.mp.dps = 60
    pts = [lo + (hi - lo) * k / 200 for k in range(201)]
    ref = [_ref(name, deg, v) for v in pts]
    scale = max(abs(r) for r in ref)
    worst = max(abs(mpmath.mpf(f(v)) - r) for v, r in zip(pts, ref, strict=True))
    assert worst <= mpmath.mpf(1e-14) * scale, float(worst / scale)


def test_compile_mp_sums_that_cancel() -> None:
    mpmath.mp.dps = 30
    f = compile_mp(p("legendre(100, x)"), [x], 30)
    for v in (-0.97, -0.3, 0.01, 0.5, 0.999):
        got = f(mpmath.mpf(v))
        mpmath.mp.dps = 60
        exact = mpmath.legendre(100, mpmath.mpf(v))
        mpmath.mp.dps = 30
        assert abs(got - exact) <= mpmath.mpf(10) ** -27, (v, got, exact)


def test_cancellation_inside_larger_expressions() -> None:
    f = compile_expr(p("sin(x) + legendre(60, x)"), [x])
    g = compile_expr(p("legendre(40, cos(t))"), [t])
    mpmath.mp.dps = 60
    for k in range(41):
        v = -1 + k / 20
        exact = mpmath.sin(mpmath.mpf(v)) + mpmath.legendre(60, mpmath.mpf(v))
        assert abs(mpmath.mpf(f(v)) - exact) < 1e-14, v
        c = math.cos(3 * v)  # the float that cos(t) gives: the polynomial is exact on it
        assert abs(mpmath.mpf(g(3 * v)) - mpmath.legendre(40, mpmath.mpf(c))) < 1e-14, v


def test_plain_sums_keep_or_improve_their_values() -> None:
    v = 1 + 2.0**-30
    assert compile_expr(x - 1, [x])(v) == 2.0**-30
    w = 1 + 1e-8
    exact = float((mpmath.mpf(w) - 1) ** 2)  # the float that (w - 1)^2 rounds to
    assert compile_expr(x**2 - 2 * x + 1, [x])(w) == exact
    assert compile_expr(x**2 + 3 * x, [x])(2.0) == 10.0


def test_huge_powers_and_overflow_stay_cheap() -> None:
    t0 = time.perf_counter()
    f = compile_expr(x**100000 - x**99999, [x])
    assert math.isfinite(f(1.0000001))
    h = compile_expr(p(f"hermite({limits.POLY_DEGREE_MAX}, x)"), [x])
    assert math.isnan(h(1e10))  # beyond the float range: nan, no exception
    assert time.perf_counter() - t0 < 2


def test_numeric_tools_on_high_degree_polynomials() -> None:
    r = ok("math_numeric", operation="quad", expr="legendre(60, x)**2", bounds=["-1", "1"])
    assert abs(float(r["value"]) - 2 / 121) < 1e-14, r["value"]
    nodes, _w = np.polynomial.legendre.leggauss(20)
    roots = ok("math_numeric", operation="polyroots", expr="legendre(20, x)", variables=["x"])[
        "roots"
    ]
    assert np.max(np.abs(np.sort([float(r) for r in roots]) - np.sort(nodes))) < 1e-13
    top = limits.POLY_DEGREE_MAX
    assert err("math_numeric", operation="polyroots", expr=f"x**{top + 1} - 1") == "LIMIT_EXCEEDED"


def test_spec_lists_every_function_of_math_eval() -> None:
    """Before D104 the table of the spec missed 43 of the names math_eval reads (gamma, erf, coth, ...)."""
    import re
    from pathlib import Path

    from oce_math_mcp.tools.math_eval import FUNCS

    spec = (Path(__file__).parents[1] / "docs" / "spec" / "SPEC_OCE_MATH_MCP_V3.md").read_text(
        encoding="utf-8"
    )
    table = spec[spec.index("Allowed functions:") : spec.index("## 4. `math_solve`")]
    named = set(re.findall(r"`([A-Za-z_]\w*)", table))
    assert sorted(set(FUNCS) - named) == []


def test_one_variable_polynomials_use_integer_horner() -> None:
    """With Fraction at every step a quad of legendre(200, x)**2 took 57.06 s, with Horner on integers 3.23 s."""
    from oce_math_mcp.numfunc import _one_variable_poly

    assert _one_variable_poly(p("legendre(5, x)")) == (x, [63, 0, -70, 0, 15, 0], 8)
    found = _one_variable_poly(p("legendre(3, cos(t))"))
    assert found is not None and found[0] == sp.cos(t)
    top = limits.POLY_DEGREE_MAX
    others = [sp.sin(x) + x**2, x ** (top + 1) - x, x**2 + sp.Float(0.5) * x, x * t - 1]
    for other in others:
        assert _one_variable_poly(other) is None, other
