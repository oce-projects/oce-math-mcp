"""From a SymPy expression to a numerical Python function, without lambdify (which uses exec).

D104: a sum whose terms cancel (sum of |terms| more than _CANCEL times |sum|) is computed again with exact
rationals from the same inputs and rounded once. Measured on 2026-11-26 before D104: the expanded
legendre(100, x) had an error 9.2e19 times its largest value on [-1, 1] in floating point.

D105: compile_mp works with _MP_GUARD more digits than the caller and rounds the result once, and its
constants follow the working precision (mpmath.quadosc raises it to 340 bits inside). Before, the rounded and
the exact path of a sum differed by up to 50 units in the last place at the caller's precision, and by 1e-30
(the digits of the constants) at a raised one: the Born integral of Woods-Saxon took 17.68 s instead of 1.4 s.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, cast

import sympy as sp

from .errors import ToolError

_CANCEL = 100.0  # digits lost above about 2: the sum is recomputed exactly
_EXACT_POW_MAX = 256  # larger integer powers enter the exact sum as their rounded value
_MP_GUARD = 10  # digits compile_mp adds to the caller's precision (constants and evaluation)


def _sech(v: float) -> float:
    a = abs(v)  # 2 e^-a / (1 + e^-2a): no overflow of cosh for |v| > 710
    return 2.0 * math.exp(-a) / (1.0 + math.exp(-2.0 * a))


def _csch(v: float) -> float:
    a = abs(v)  # expm1 keeps the digits for small |v|; v = 0 is a pole (ZeroDivisionError -> nan)
    return math.copysign(2.0 * math.exp(-a) / -math.expm1(-2.0 * a), v)


def _ratio(v: Any) -> tuple[int, int]:
    """Exact value of a float, an int or an mpf as numerator and denominator (nan, inf, complex raise)."""
    if isinstance(v, float):
        return v.as_integer_ratio()
    if isinstance(v, int):
        return v, 1
    import mpmath

    if isinstance(v, mpmath.mpf) and mpmath.isfinite(v):
        man, exp = v.man_exp
        return (int(man) << exp, 1) if exp >= 0 else (int(man), 1 << -exp)
    raise TypeError(f"no exact value for {v!r}")


Leaf = Callable[[Any], Callable[[Sequence[Any]], Any]]


def _one_variable_poly(e: Any) -> tuple[Any, list[int], int] | None:
    """(generator, integer coefficients from the highest degree, common denominator) of a polynomial
    with rational coefficients in one generator (x, cos(t), ...); None otherwise or above the degree limit."""
    from .limits import POLY_DEGREE_MAX

    for pw in e.atoms(sp.Pow):
        if pw.exp.is_Integer and abs(int(pw.exp)) > POLY_DEGREE_MAX:
            return None
    try:
        poly = sp.Poly(e)
    except sp.PolynomialError:
        return None
    if len(poly.gens) != 1 or poly.get_domain() not in (sp.ZZ, sp.QQ):
        return None
    pq = [(int(r.p), int(r.q)) for r in (cast(Any, sp.Rational(c)) for c in poly.all_coeffs())]
    den = int(sp.ilcm(*[q for _p, q in pq])) if len(pq) > 1 else pq[0][1]
    return poly.gens[0], [p_ * (den // q) for p_, q in pq], den


def _exact_sum(
    e: Any, index: dict[sp.Symbol, int], leaf: Leaf
) -> Callable[[Sequence[Any]], tuple[int, int]]:
    """Exact value of the sum e as (numerator, denominator), rounded once by the caller.

    A polynomial in one generator goes through Horner on integers (no reduction at every step: legendre(200, x)
    with Fraction took 57 s in a quad). Otherwise +, *, integer powers, symbols and rationals are exact Fractions.
    Any other sub-expression (sin(x), pi, a large power) enters with the value of its numerical closure
    `leaf(sub)`, taken as an exact rational: only the sum is made exact, not the functions.
    """
    found = _one_variable_poly(e)
    if found is not None:
        gen, coeffs, den = found
        if isinstance(gen, sp.Symbol) and gen in index:
            i = index[gen]
            value: Callable[[Sequence[Any]], Any] = lambda a: a[i]  # noqa: E731
        else:
            value = leaf(gen)
        degree = len(coeffs) - 1

        def horner(a: Sequence[Any]) -> tuple[int, int]:
            m, q = _ratio(value(a))
            acc, qpow = coeffs[0], 1
            for c in coeffs[1:]:  # sum c_k m^k q^(n-k)
                qpow *= q
                acc = acc * m + c * qpow
            return acc, den * q**degree

        return horner

    def build(node: Any) -> Callable[[Sequence[Any]], Fraction]:
        if isinstance(node, sp.Symbol) and node in index:
            i = index[node]
            return lambda a: Fraction(*_ratio(a[i]))
        if isinstance(node, sp.Rational):
            c = Fraction(int(node.p), int(node.q))
            return lambda a: c
        if isinstance(node, sp.Add | sp.Mul):
            fs = [build(arg) for arg in node.args]
            if isinstance(node, sp.Add):
                return lambda a: sum((f(a) for f in fs), Fraction(0))

            def mul(a: Sequence[Any]) -> Fraction:
                out = Fraction(1)
                for f in fs:
                    out *= f(a)
                return out

            return mul
        if (
            isinstance(node, sp.Pow)
            and node.exp.is_Integer
            and abs(int(node.exp)) <= _EXACT_POW_MAX
        ):
            fb, k = build(node.base), int(node.exp)
            return lambda a: fb(a) ** k
        g = leaf(node)
        return lambda a: Fraction(*_ratio(g(a)))

    total = build(e)

    def generic(a: Sequence[Any]) -> tuple[int, int]:
        q = total(a)
        return q.numerator, q.denominator

    return generic


_F1: dict[Any, Callable[[float], float]] = {
    sp.sin: math.sin,
    sp.cos: math.cos,
    sp.tan: math.tan,
    sp.asin: math.asin,
    sp.acos: math.acos,
    sp.atan: math.atan,
    sp.sinh: math.sinh,
    sp.cosh: math.cosh,
    sp.tanh: math.tanh,
    sp.asinh: math.asinh,
    sp.acosh: math.acosh,
    sp.atanh: math.atanh,
    sp.exp: math.exp,
    sp.log: math.log,
    sp.Abs: abs,
    sp.floor: math.floor,
    sp.ceiling: math.ceil,
    sp.erf: math.erf,
    sp.erfc: math.erfc,
    sp.gamma: math.gamma,
    sp.loggamma: math.lgamma,
    sp.sec: lambda v: 1 / math.cos(v),
    sp.csc: lambda v: 1 / math.sin(v),
    sp.cot: lambda v: 1 / math.tan(v),
    # D104: the inverse functions through 1/v; outside the real domain math raises and the value is nan.
    sp.sech: _sech,
    sp.csch: _csch,
    sp.acoth: lambda v: math.atanh(1 / v),
    sp.asech: lambda v: math.acosh(1 / v),
    sp.acsch: lambda v: math.asinh(1 / v),
    sp.asec: lambda v: math.acos(1 / v),
    sp.acsc: lambda v: math.asin(1 / v),
    sp.sign: lambda v: (v > 0) - (v < 0),
    sp.Heaviside: lambda v: 1.0 if v > 0 else (0.5 if v == 0 else 0.0),
    sp.factorial: lambda v: math.gamma(v + 1),
}


def _pow(x: float, y: float) -> float:
    r = x**y
    if isinstance(r, complex):
        return math.nan
    return r


def compile_expr(expr: Any, symbols: Sequence[sp.Symbol]) -> Callable[..., float]:
    """Returns f(*values) -> float. Values outside the domain give nan."""
    index = {s: i for i, s in enumerate(symbols)}

    def build(e: Any) -> Callable[[Sequence[float]], float]:
        if isinstance(e, sp.Symbol):
            if e not in index:
                raise ToolError(
                    "INVALID_ARGUMENT", f"The symbol '{e}' has no numerical value.", "variables"
                )
            i = index[e]
            return lambda a: a[i]
        if e.is_number:
            try:
                v = complex(e)
            except TypeError:
                raise ToolError("MATH_ERROR", f"Non-numeric value: {e}") from None
            fv = v.real if v.imag == 0 else math.nan
            return lambda a: fv
        if isinstance(e, sp.Add):
            fs = [build(x) for x in e.args]
            exact: list[
                Callable[[Sequence[Any]], tuple[int, int]]
            ] = []  # built at the first cancellation

            def add(a: Sequence[float], node: Any = e) -> float:
                terms = [f(a) for f in fs]
                s = math.fsum(terms)
                size = math.fsum(abs(v) for v in terms)
                if math.isfinite(size) and size > _CANCEL * abs(s):
                    if not exact:
                        exact.append(_exact_sum(node, index, build))
                    try:
                        num, den = exact[0](a)
                    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
                        return s
                    return num / den  # int / int rounds once; OverflowError -> nan in call()
                return s

            return add
        if isinstance(e, sp.Mul):
            fs = [build(x) for x in e.args]

            def mul(a: Sequence[float]) -> float:
                out = 1.0
                for f in fs:
                    out *= f(a)
                return out

            return mul
        if isinstance(e, sp.Pow):
            fb, fe = build(e.base), build(e.exp)
            if e.exp == sp.Rational(1, 2):
                return lambda a: math.sqrt(fb(a))
            return lambda a: _pow(fb(a), fe(a))
        if isinstance(e, sp.atan2):
            fy, fx = build(e.args[0]), build(e.args[1])
            return lambda a: math.atan2(fy(a), fx(a))
        if isinstance(e, sp.Max | sp.Min):
            fs = [build(x) for x in e.args]
            agg = max if isinstance(e, sp.Max) else min
            return lambda a: agg(f(a) for f in fs)
        if e.func in _F1 and len(e.args) == 1:
            f1 = _F1[e.func]
            g = build(e.args[0])
            return lambda a: f1(g(a))
        syms = list(symbols)

        def slow(a: Sequence[float]) -> float:
            v = complex(e.evalf(30, subs=dict(zip(syms, a, strict=True))))
            return v.real if abs(v.imag) < 1e-12 else math.nan

        return slow

    f = build(expr if isinstance(expr, sp.Basic) else sp.Float(expr))

    def call(*vals: float) -> float:
        try:
            return float(f(vals))
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            return math.nan

    return call


_MP1 = {
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.sec: "sec",
    sp.csc: "csc",
    sp.cot: "cot",
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.asinh: "asinh",
    sp.acosh: "acosh",
    sp.atanh: "atanh",
    sp.sech: "sech",
    sp.csch: "csch",
    sp.acoth: "acoth",
    sp.asech: "asech",
    sp.acsch: "acsch",
    sp.asec: "asec",
    sp.acsc: "acsc",
    sp.exp: "exp",
    sp.log: "log",
    sp.Abs: "fabs",
    sp.floor: "floor",
    sp.ceiling: "ceil",
    sp.erf: "erf",
    sp.erfc: "erfc",
    sp.gamma: "gamma",
    sp.loggamma: "loggamma",
    sp.factorial: "factorial",
    sp.zeta: "zeta",
    sp.sign: "sign",
    sp.LambertW: "lambertw",
    sp.Ei: "ei",
    sp.li: "li",
    sp.Si: "si",
    sp.Ci: "ci",
    sp.digamma: "digamma",
}


def to_mp(value: Any, dps: int) -> Any:
    """Exact SymPy number -> mpf/mpc at the given precision (without eval)."""
    import mpmath

    if value in (sp.oo, sp.S.Infinity):
        return mpmath.inf
    if value in (-sp.oo, sp.S.NegativeInfinity):
        return -mpmath.inf
    v = sp.N(value, dps + 10)
    re_, im_ = v.as_real_imag()
    if im_ == 0:
        return mpmath.mpf(str(re_))
    return mpmath.mpc(str(re_), str(im_))


def compile_mp(expr: Any, symbols: Sequence[sp.Symbol], dps: int) -> Callable[..., Any]:
    """Like compile_expr but with mpmath at precision dps (to be set before the call).

    The function computes with _MP_GUARD more digits and rounds its result to the caller's precision, and each
    constant is rounded at the working precision of the call (kept once per precision), so the switch to the
    exact sum of D104 leaves no step that an adaptive mpmath routine (quad, quadosc, findroot) would chase,
    also when it raises the precision.
    """
    import mpmath

    index = {s: i for i, s in enumerate(symbols)}
    syms = list(symbols)

    def build(e: Any) -> Callable[[Sequence[Any]], Any]:
        if isinstance(e, sp.Symbol):
            if e not in index:
                raise ToolError(
                    "INVALID_ARGUMENT", f"The symbol '{e}' has no numerical value.", "variables"
                )
            i = index[e]
            return lambda a: a[i]
        if e.is_number:
            if e.is_Integer:
                c = mpmath.mpf(int(e))  # exact at any precision
                return lambda a: c
            kept: dict[int, Any] = {}

            def const(a: Sequence[Any], e: Any = e, kept: dict[int, Any] = kept) -> Any:
                prec = mpmath.mp.prec
                c = kept.get(prec)
                if c is None:
                    c = kept[prec] = to_mp(e, mpmath.mp.dps)
                return c

            return const
        if isinstance(e, sp.Add):
            fs = [build(x) for x in e.args]
            exact: list[Callable[[Sequence[Any]], tuple[int, int]]] = []

            def add(a: Sequence[Any], node: Any = e) -> Any:
                terms = [f(a) for f in fs]
                s = mpmath.fsum(terms)
                if all(isinstance(v, mpmath.mpf) for v in terms):
                    size = mpmath.fsum(terms, absolute=True)
                    if mpmath.isfinite(size) and size > _CANCEL * abs(s):
                        if not exact:
                            exact.append(_exact_sum(node, index, build))
                        try:
                            num, den = exact[0](a)
                        except (ValueError, ZeroDivisionError, TypeError, OverflowError):
                            return s
                        return mpmath.mpf(num) / den
                return s

            return add
        if isinstance(e, sp.Mul):
            fs = [build(x) for x in e.args]
            return lambda a: mpmath.fprod(f(a) for f in fs)
        if isinstance(e, sp.Pow):
            fb, fe = build(e.base), build(e.exp)
            if e.exp == sp.Rational(1, 2):
                return lambda a: mpmath.sqrt(fb(a))
            if e.exp.is_Integer:
                k = int(e.exp)
                return lambda a: fb(a) ** k
            return lambda a: mpmath.power(fb(a), fe(a))
        if isinstance(e, sp.atan2):
            fy, fx = build(e.args[0]), build(e.args[1])
            return lambda a: mpmath.atan2(fy(a), fx(a))
        if isinstance(e, sp.Max | sp.Min):
            fs = [build(x) for x in e.args]
            agg = max if isinstance(e, sp.Max) else min
            return lambda a: agg(f(a) for f in fs)
        if e.func in _MP1 and len(e.args) == 1:
            f1 = getattr(mpmath, _MP1[e.func])
            g = build(e.args[0])
            return lambda a: f1(g(a))

        def slow(a: Sequence[Any]) -> Any:
            d = mpmath.mp.dps + 5
            subs = {s: sp.Float(mpmath.nstr(v, d), d) for s, v in zip(syms, a, strict=True)}
            return to_mp(e.evalf(d, subs=subs), mpmath.mp.dps)

        return slow

    with mpmath.workdps(dps + _MP_GUARD):
        f = build(expr if isinstance(expr, sp.Basic) else sp.Float(expr))

    def run(*vals: Any) -> Any:
        with mpmath.workdps(mpmath.mp.dps + _MP_GUARD):
            v = f(vals)
        return +v  # rounded once to the caller's precision

    return run
