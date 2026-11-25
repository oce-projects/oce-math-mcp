"""From a SymPy expression to a numerical Python function, without lambdify (which uses exec)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import sympy as sp

from .errors import ToolError

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
            return lambda a: math.fsum(f(a) for f in fs)
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
    """Like compile_expr but with mpmath at precision dps (to be set before the call)."""
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
            c = to_mp(e, dps)
            return lambda a: c
        if isinstance(e, sp.Add):
            fs = [build(x) for x in e.args]
            return lambda a: mpmath.fsum(f(a) for f in fs)
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
            subs = {
                s: sp.Float(mpmath.nstr(v, dps + 5), dps + 5) for s, v in zip(syms, a, strict=True)
            }
            return to_mp(e.evalf(dps + 5, subs=subs), dps)

        return slow

    f = build(expr if isinstance(expr, sp.Basic) else sp.Float(expr))
    return lambda *vals: f(vals)
