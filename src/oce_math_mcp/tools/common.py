"""Functions and grammars shared by the symbolic tools."""

from __future__ import annotations

from typing import Any, cast

import sympy as sp

from .. import limits
from ..errors import ToolError
from ..expr import Grammar, check_identifier, parse, parse_relation


def _log(x: Any, base: Any = None) -> Any:
    return sp.log(x) if base is None else sp.log(x, base)


def _diff(f: Any, *args: Any) -> Any:
    return sp.diff(f, *args)


SOLVE_FUNCS: dict[str, Any] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "acot": sp.acot,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "atan2": sp.atan2,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "coth": sp.coth,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "exp": sp.exp,
    "log": _log,
    "ln": sp.log,
    "log10": lambda x: sp.log(x, 10),
    "log2": lambda x: sp.log(x, 2),
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "root": sp.root,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "sign": sp.sign,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "ceil": sp.ceiling,
    "factorial": sp.factorial,
    "binomial": sp.binomial,
    "comb": sp.binomial,
    "gamma": sp.gamma,
    "beta": sp.beta,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "zeta": sp.zeta,
    "Heaviside": sp.Heaviside,
    "DiracDelta": sp.DiracDelta,
    "Min": sp.Min,
    "Max": sp.Max,
    "min": sp.Min,
    "max": sp.Max,
    "re": sp.re,
    "im": sp.im,
    "arg": sp.arg,
    "conjugate": sp.conjugate,
    "Mod": sp.Mod,
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    "fibonacci": sp.fibonacci,
    "harmonic": sp.harmonic,
    "LambertW": sp.LambertW,
    "besselj": sp.besselj,
    "bessely": sp.bessely,
    "Si": sp.Si,
    "Ci": sp.Ci,
    "Ei": sp.Ei,
    "li": sp.li,
    "digamma": sp.digamma,
    "polygamma": sp.polygamma,
    "diff": _diff,
}


def symbols_for(names: list[str] | None) -> list[sp.Symbol]:
    return [sp.Symbol(check_identifier(n)) for n in (names or [])]


def grammar(
    declared_names: list[str] | None = None, extra: dict[str, Any] | None = None
) -> Grammar:
    declared: dict[str, Any] = {n: sp.Symbol(check_identifier(n)) for n in (declared_names or [])}
    if extra:
        declared.update(extra)
    return Grammar(SOLVE_FUNCS, declared)


def rel_parts(rel: Any) -> tuple[sp.Expr, sp.Expr, str]:
    """Sides and operator of a SymPy relation: 'a >= b' gives (a, b, '>=').

    The SymPy stubs give lhs/rhs the type Basic and do not declare rel_op: here the type is right.
    """
    return cast(sp.Expr, rel.lhs), cast(sp.Expr, rel.rhs), str(rel.rel_op)


def expr_of(text: Any, g: Grammar) -> Any:
    """Expression; an equation 'a = b' becomes a - b."""
    rel = parse_relation(text, g)
    if isinstance(rel, sp.Eq):
        lhs, rhs, _ = rel_parts(rel)
        return lhs - rhs
    if isinstance(rel, sp.core.relational.Relational):
        raise ToolError(
            "PARSE_ERROR",
            "An expression or an equation is needed here, not an inequality.",
            "x**2 - 4",
        )
    return rel


def value_of(raw: Any, g: Grammar) -> Any:
    """JSON number or expression string -> SymPy object."""
    if isinstance(raw, bool):
        raise ToolError("INVALID_ARGUMENT", "Boolean value not allowed here.", "0")
    if isinstance(raw, int):
        return sp.Integer(raw)
    if isinstance(raw, float):
        return sp.Rational(repr(raw))
    if isinstance(raw, str):
        return parse(raw, g)
    raise ToolError("INVALID_ARGUMENT", f"Invalid value: {raw!r}.", '"pi/2"')


def auto_vars(expr: Any, names: list[str] | None) -> list[sp.Symbol]:
    if names:
        return symbols_for(names)
    free = sorted(expr.free_symbols, key=lambda s: s.name)
    if not free:
        raise ToolError("INVALID_ARGUMENT", "No variable found: give 'variables'.", '["x"]')
    return free


def matrix_of(rows: Any, g: Grammar, name: str = "matrix_a") -> sp.Matrix:
    if not isinstance(rows, list) or not rows:
        raise ToolError("INVALID_ARGUMENT", f"'{name}' must be a list of rows.", "[[1, 2], [3, 4]]")
    if not isinstance(rows[0], list):
        rows = [[v] for v in rows]
    nrows, ncols = len(rows), len(rows[0])
    if nrows > limits.MATRIX_MAX or ncols > limits.MATRIX_MAX:
        raise ToolError(
            "LIMIT_EXCEEDED",
            f"Matrix {nrows}x{ncols}: maximum {limits.MATRIX_MAX}x{limits.MATRIX_MAX}.",
        )
    if any(not isinstance(r, list) or len(r) != ncols for r in rows):
        raise ToolError(
            "INVALID_ARGUMENT", f"The rows of '{name}' have different lengths.", "[[1, 2], [3, 4]]"
        )
    return sp.Matrix([[value_of(v, g) for v in r] for r in rows])


def mat_str(m: Any) -> list[list[str]]:
    return [[str(m[i, j]) for j in range(m.shape[1])] for i in range(m.shape[0])]


_UNDEFINED = (sp.zoo, sp.nan, sp.oo, -sp.oo)


def check_bound_order(lims: list[tuple[Any, Any, Any]]) -> None:
    """The first entry of bounds is the innermost integral: a bound may use only the outer variables (D91).

    With the order reversed, quad evaluated the bound at 0 and gave 0.0, multi_integrate left the variable in the
    result ("x" for the triangle 0 <= y <= x <= 1); both said nothing.
    """
    names = [v for v, _, _ in lims]
    for i, (v, a, b) in enumerate(lims):
        used = getattr(a, "free_symbols", set()) | getattr(b, "free_symbols", set())
        bad = used & set(names[: i + 1])
        if bad:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"The bounds of {v} use {', '.join(sorted(map(str, bad)))}, integrated inside it: "
                "the first entry of bounds is the innermost integral.",
                '[["y", 0, "x"], ["x", 0, 1]]',
            )


def is_zero(expr: Any) -> bool:
    """Symbolic zero, with numeric fallback on fixed points.

    D90: an exact nonzero number is not zero (it was, below 1e-10), an undefined value (zoo, nan, oo) is not zero
    (nan passed every comparison), and the numeric fallback is relative to the size of the terms: 1e-20 at 30
    digits for exact expressions, 1e-10 when the expression carries floats.
    """
    try:
        s = sp.simplify(expr)
        if s == 0:
            return True
        if s.has(*_UNDEFINED) or s.is_Rational:
            return False
    except (TypeError, ValueError, NotImplementedError, RecursionError, AttributeError):
        pass
    free = sorted(getattr(expr, "free_symbols", set()), key=lambda s: s.name)
    points = [sp.Rational(3, 7), sp.Rational(11, 13), sp.Rational(-5, 9), sp.Rational(17, 5)]
    rel = 1e-10 if expr.atoms(sp.Float) else 1e-20
    try:
        for k in range(3):
            subs = {s: points[(i + k) % len(points)] for i, s in enumerate(free)}
            e = expr.subs(subs)
            if e.is_Rational:  # exact arithmetic: x / 10**25 at x = 3/7 is not zero
                if e != 0:
                    return False
                continue
            v = abs(complex(sp.N(e, 30)))
            scale = max([1.0] + [abs(complex(sp.N(t, 30))) for t in sp.Add.make_args(e)])
            if not v <= rel * scale:  # also False for nan
                return False
        return True
    except (TypeError, ValueError, ZeroDivisionError, AttributeError):
        return False
