"""math_solve: symbolic SymPy with self-verification (spec V3, section 4)."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

import numpy as np
import sympy as sp
from sympy.integrals import manualintegrate as mi

from ..envelope import Ctx
from ..errors import ToolError
from ..expr import Grammar, check_identifier, parse, parse_relation, split_equation, split_list
from ..numfunc import compile_expr, compile_mp, to_mp
from .common import (
    SOLVE_FUNCS,
    auto_vars,
    check_bound_order,
    expr_of,
    grammar,
    is_zero,
    mat_str,
    matrix_of,
    rel_parts,
    symbols_for,
    value_of,
)

Op = Callable[[dict[str, Any], Ctx], dict[str, Any]]
OPS: dict[str, Op] = {}


def op(name: str) -> Callable[[Op], Op]:
    def deco(fn: Op) -> Op:
        OPS[name] = fn
        return fn

    return deco


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    return OPS[args["operation"]](args, ctx)


def _need(args: dict[str, Any], key: str, example: str) -> Any:
    v = args.get(key)
    if v in (None, "", []):
        raise ToolError(
            "INVALID_ARGUMENT", f"'{args['operation']}' needs the field '{key}'.", example
        )
    return v


def _g(args: dict[str, Any], more: list[str] | None = None) -> Grammar:
    names = list(args.get("variables") or []) + list(more or [])
    return grammar(names)


def _num(v: Any) -> float | None:
    try:
        c = complex(sp.N(v, 30))
    except (TypeError, ValueError):
        return None
    if abs(c.imag) > 1e-12 or not math.isfinite(c.real):
        return None
    return c.real


def _domain(name: str) -> Any:
    return {
        "real": sp.S.Reals,
        "complex": sp.S.Complexes,
        "positive": sp.Interval.open(0, sp.oo),
        "integer": sp.S.Integers,
    }[name]


# ------------------------------------------------------------------ equations


@op("solve")
def _solve(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "x**2 - 5*x + 6 = 0"), g)
    x = auto_vars(f, args.get("variables"))[0]
    dom_name = args.get("domain", "real")
    dom = _domain(dom_name)
    if dom_name == "integer":
        sol_set = sp.solveset(f, x, sp.S.Reals)
        if isinstance(sol_set, sp.FiniteSet):
            sol_set = sp.FiniteSet(*[s for s in sol_set if s.is_integer])
    else:
        sol_set = sp.solveset(f, x, dom)
    sols: list[Any]
    if isinstance(sol_set, sp.FiniteSet):
        sols = sorted(sol_set, key=sp.default_sort_key)
    elif sol_set == sp.S.EmptySet:
        sols = []
    else:
        fallback = sp.solve(f, x)
        if fallback and not isinstance(sol_set, sp.ImageSet | sp.Union):
            sols = [s for s in fallback if dom_name == "complex" or s.is_real is not False]
        else:
            ctx.warn("Infinite or non-elementary solution set: see solution_set.")
            ctx.formatted = f"Solutions: {sol_set}"
            return {"solutions": [], "solution_set": str(sol_set), "count": None, "verified": False}
    # D90: nothing checked = not verified
    verified = bool(sols) and all(is_zero(f.subs(x, s)) for s in sols)
    if ctx.show_steps:
        _solve_steps(ctx, f, x, sols, verified)
    ctx.formatted = (
        f"{x} = " + " or ".join(str(s) for s in sols)
        if sols
        else f"No solution in the domain {dom_name}."
    )
    return {"solutions": [str(s) for s in sols], "count": len(sols), "verified": verified}


def _solve_steps(ctx: Ctx, f: Any, x: Any, sols: list[Any], verified: bool) -> None:
    ctx.step(f"Move everything to the left side: {f} = 0.", f)
    try:
        poly = sp.Poly(f, x)
    except sp.PolynomialError:
        poly = None
    if poly is not None and poly.degree() == 1:
        a, b = poly.all_coeffs()
        ctx.step(f"First-degree equation: a = {a}, b = {b}.")
        ctx.step(f"x = -b/a = {-b / a}.", -b / a)
    elif poly is not None and poly.degree() == 2:
        a, b, c = poly.all_coeffs()
        delta = sp.simplify(b**2 - 4 * a * c)
        ctx.step(f"Second-degree equation: a = {a}, b = {b}, c = {c}.")
        ctx.step(f"Discriminant: Δ = b^2 - 4ac = ({b})^2 - 4·({a})·({c}) = {delta}.", delta)
        sign = (
            "positive: two solutions"
            if delta.is_positive
            else ("zero: one double solution" if delta == 0 else "negative")
        )
        ctx.step(f"The discriminant is {sign}.")
        ctx.step("Quadratic formula: x = (-b ± sqrt(Δ)) / (2a).", sols)
    else:
        ctx.step("Solved with the SymPy algorithm (solveset).", sols)
    checks = ", ".join(f"{x} = {s}: {sp.simplify(f.subs(x, s))}" for s in sols) or "no solution"
    ctx.step(
        f"Back-check (substituting into the equation): {checks}. Outcome: {'passed' if verified else 'failed'}."
    )


@op("solve_system")
def _solve_system(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    eqs = [expr_of(t, g) for t in split_list(_need(args, "expr", "x + y = 3; x - y = 1"))]
    xs = symbols_for(args.get("variables")) or sorted(
        set().union(*[e.free_symbols for e in eqs]), key=lambda s: s.name
    )
    sols = sp.solve(eqs, xs, dict=True)
    if args.get("domain", "real") == "real":
        sols = [s for s in sols if all(v.is_real is not False for v in s.values())]
    verified = bool(sols) and all(is_zero(e.subs(s)) for s in sols for e in eqs)  # D90
    out = [{str(k): str(s[k]) for k in xs if k in s} for s in sols]
    ctx.formatted = (
        "; ".join(", ".join(f"{k} = {v}" for k, v in s.items()) for s in out) or "No solution."
    )
    ctx.step(f"System: {', '.join(f'{e} = 0' for e in eqs)}.")
    ctx.step("Solved by substitution/elimination (SymPy solve).", out)
    ctx.step(
        f"Back-check: substituting the solutions into each equation: {'all 0' if verified else 'not all 0'}."
    )
    return {"solutions": out, "verified": verified}


@op("inequality")
def _inequality(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    rel = parse_relation(_need(args, "expr", "x**2 - 4 > 0"), g)
    if not isinstance(rel, sp.core.relational.Relational) or isinstance(rel, sp.Eq):
        raise ToolError("PARSE_ERROR", "An inequality with <, <=, >, >= is needed.", "x**2 - 4 > 0")
    lhs, rhs, _ = rel_parts(rel)
    xs = auto_vars(lhs - rhs, args.get("variables"))
    x = xs[0]
    sol = sp.solve_univariate_inequality(rel, x, relational=False)
    ok = True
    for k in range(-40, 41):
        pt = sp.Rational(k, 4) + sp.Rational(1, 7)
        try:
            truth = bool(rel.subs(x, pt))
        except TypeError:
            continue
        if truth != bool(sol.contains(pt)):
            ok = False
            break
    ctx.formatted = f"Solution: {x} ∈ {sol}"
    ctx.step(f"Inequality: {rel}.")
    ctx.step(f"Solution set: {sol}.", sol)
    ctx.step(f"Back-check on 81 test points: {'consistent' if ok else 'inconsistent'}.")
    return {"solution_set": str(sol), "verified": ok}


# ------------------------------------------------------------------ derivate


def _rule_name(f: Any, x: Any) -> str:
    if isinstance(f, sp.Add):
        return "sum rule"
    if isinstance(f, sp.Mul):
        _num, den = sp.fraction(f)
        if den != 1 and den.has(x):
            return "quotient rule"
        return (
            "product rule"
            if sum(1 for a in f.args if a.has(x)) > 1
            else "constant times a function"
        )
    if isinstance(f, sp.Pow):
        if f.base.has(x) and not f.exp.has(x):
            return "power rule" + (" with the chain rule" if f.base != x else "")
        return "derivative of an exponential (with the chain rule)"
    if isinstance(f, sp.Function) and f.args and f.args[0] != x:
        return f"chain rule: derivative of {f.func.__name__} times the derivative of the argument {f.args[0]}"
    if isinstance(f, sp.Function):
        return f"elementary derivative of {f.func.__name__}"
    return "elementary derivative"


@op("diff")
def _diff_op(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "sin(x)*x"), g)
    x = auto_vars(f, args.get("variables"))[0]
    n = int(args.get("order", 1) or 1)
    if n < 1 or n > 50:
        raise ToolError("INVALID_ARGUMENT", "order must be between 1 and 50.", "2")
    d = sp.diff(f, x, n)
    ds = sp.simplify(d) if sp.count_ops(d) < 200 else d
    verified = _check_derivative_numeric(f, ds, [(x, n)])
    if ctx.show_steps:
        ctx.step(f"Function: f({x}) = {f}.", f)
        ctx.step(f"Rule: {_rule_name(f, x)}.")
        if isinstance(f, sp.Add | sp.Mul):
            for part in f.args:
                if part.has(x):
                    ctx.step(
                        f"Derivative of {part}: {sp.diff(part, x)} ({_rule_name(part, x)}).",
                        sp.diff(part, x),
                    )
        ctx.step(f"Result: f^({n})({x}) = {ds}.", ds)
        ctx.step(
            f"Back-check with a numerical derivative at 3 points: {'passed' if verified else 'failed'}."
        )
    ctx.formatted = f"Derivative: {ds}"
    return {"derivative": str(ds), "verified": verified}


def _mp_value(e: Any, at: dict[Any, Any]) -> Any:
    """Value of e at the mpmath points `at`, at the working precision of mpmath (D90).

    mpmath.diff raises its precision with the order; the value must keep it: going through complex() (as before
    D90) left 16 digits and the finite differences gave 0.
    """
    import mpmath

    dps = mpmath.mp.dps + 10
    pts = {s: sp.Float(mpmath.nstr(v, dps), dps) for s, v in at.items()}
    re_, im_ = sp.N(e.subs(pts), dps).as_real_imag()
    return mpmath.mpc(mpmath.mpf(str(sp.N(re_, dps))), mpmath.mpf(str(sp.N(im_, dps))))


def _check_derivative_numeric(f: Any, d: Any, orders: list[tuple[Any, int]]) -> bool:
    """d against the numerical derivative of f (mpmath.diff) at 3 points; orders = [(variable, order), ...]."""
    import mpmath

    xs = [v for v, _ in orders]
    others = sorted(f.free_symbols - set(xs), key=lambda s: s.name)
    subs = {s: sp.Rational(7 + i, 5) for i, s in enumerate(others)}
    f0, d0 = f.subs(subs), d.subs(subs)
    good = 0
    with mpmath.workdps(40):
        for pn, pd in ((3, 7), (6, 5), (-4, 9)):
            pts = [Fraction(pn, pd) + Fraction(j, 11) for j in range(len(xs))]
            at = tuple(mpmath.mpf(p.numerator) / p.denominator for p in pts)
            try:
                exact = _mp_value(d0, dict(zip(xs, at, strict=True)))
                num = mpmath.diff(
                    lambda *vs: _mp_value(f0, dict(zip(xs, vs, strict=True))),
                    at,
                    tuple(n for _, n in orders),
                )
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if not (mpmath.isfinite(exact) and mpmath.isfinite(num)):
                continue
            if abs(exact - num) > mpmath.mpf("1e-25") * (1 + abs(exact)):
                return False
            good += 1
    return good > 0


@op("partial_diff")
def _partial(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "x**2*y**3"), g)
    xs = symbols_for(_need(args, "variables", '["x", "y"]'))
    d = sp.simplify(sp.diff(f, *xs))
    # D90: before, the check was the same sympy derivative again; now the numerical mixed derivative
    orders: dict[Any, int] = {}
    for v in xs:
        orders[v] = orders.get(v, 0) + 1
    verified = _check_derivative_numeric(f, d, list(orders.items()))
    ctx.formatted = f"Partial derivative: {d}"
    return {"derivative": str(d), "verified": verified}


# ------------------------------------------------------------------ integrals

_RULE_TEXT = {
    "ConstantRule": "integral of a constant",
    "ConstantTimesRule": "constant taken out of the integral",
    "PowerRule": "power rule: ∫x^n dx = x^(n+1)/(n+1)",
    "ExpRule": "integral of the exponential",
    "ReciprocalRule": "∫1/x dx = log(x)",
    "AddRule": "linearity: integral of the sum = sum of the integrals",
    "URule": "substitution u = ...",
    "PartsRule": "integration by parts: ∫u dv = u v - ∫v du",
    "CyclicPartsRule": "cyclic integration by parts",
    "RewriteRule": "rewriting of the integrand",
    "PiecewiseRule": "piecewise integral",
    "TrigRule": "elementary trigonometric integral",
    "SinRule": "∫sin(x) dx = -cos(x)",
    "CosRule": "∫cos(x) dx = sin(x)",
    "SecTanRule": "∫sec(x)tan(x) dx = sec(x)",
    "ArctanRule": "∫1/(a x^2 + b) dx with the arctangent",
    "PolynomialDivisionRule": "polynomial division",
    "AlternativeRule": "one of the possible routes",
    "DontKnowRule": "no elementary rule: non-elementary integral",
    "DerivativeRule": "integral of a derivative",
    "TrigSubstitutionRule": "trigonometric substitution",
    "PartialFractionRule": "partial fractions",
}


def _walk_rules(rule: Any, out: list[str], depth: int = 0) -> None:
    if rule is None or depth > 12 or len(out) > 20:
        return
    name = type(rule).__name__
    desc = _RULE_TEXT.get(name, name)
    integrand = getattr(rule, "integrand", None)
    out.append(f"{'  ' * depth}{desc}" + (f" su {integrand}" if integrand is not None else ""))
    for field in getattr(rule, "__dataclass_fields__", {}):
        val = getattr(rule, field, None)
        if isinstance(val, mi.Rule):
            _walk_rules(val, out, depth + 1)
        elif isinstance(val, list | tuple):
            for v in val:
                if isinstance(v, mi.Rule):
                    _walk_rules(v, out, depth + 1)


def _bound(v: Any, g: Grammar) -> Any:
    return value_of(v, g)


@op("integrate")
def _integrate(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "x**2"), g)
    x = auto_vars(f, args.get("variables"))[0]
    bounds = args.get("bounds") or []
    if bounds:
        if len(bounds) != 2:
            raise ToolError("INVALID_ARGUMENT", "bounds must be [a, b].", '[0, "pi"]')
        a, b = _bound(bounds[0], g), _bound(bounds[1], g)
        val = sp.integrate(f, (x, a, b))
        val = sp.simplify(val) if not val.has(sp.Integral) else val
        num = _num(val)
        verified = False
        if not val.has(sp.Integral):
            try:
                quad = complex(sp.Integral(f, (x, a, b)).evalf(20))
                verified = num is not None and abs(quad - num) <= 1e-9 * (1 + abs(num))
            except (TypeError, ValueError):
                verified = False
        if ctx.show_steps:
            F = sp.integrate(f, x)
            ctx.step(f"Integrand: {f}.", f)
            ctx.step(f"Antiderivative: F({x}) = {F}.", F)
            ctx.step(f"Values at the bounds: F({b}) - F({a}).")
            ctx.step(f"Result: {val}.", val)
            ctx.step(
                f"Back-check with numerical integration: {'matches' if verified else 'not verified'}."
            )
        ctx.formatted = f"∫ from {a} to {b} of {f} d{x} = {val}"
        return {"integral": str(val), "definite": True, "numeric": num, "verified": verified}
    F = sp.integrate(f, x)
    unsolved = F.has(sp.Integral)
    verified = (not unsolved) and is_zero(sp.diff(F, x) - f)
    if unsolved:
        ctx.warn("Non-elementary antiderivative: SymPy finds no closed form.")
    if ctx.show_steps:
        ctx.step(f"Integrand: {f}.", f)
        rules: list[str] = []
        try:
            _walk_rules(mi.integral_steps(f, x), rules)
        except (NotImplementedError, ValueError, TypeError, RecursionError):
            rules = ["rule not describable: result from the Risch/Meijer algorithm of SymPy"]
        for r in rules:
            ctx.step(f"Rule: {r}.")
        ctx.step(f"Antiderivative: {F} + C.", F)
        ctx.step(
            f"Back-check: differentiating the antiderivative gives back the integrand: {'yes' if verified else 'no'}."
        )
    ctx.formatted = f"∫ {f} d{x} = {F} + C"
    return {"integral": str(F), "definite": False, "numeric": None, "verified": verified}


def _bound_list(args: dict[str, Any], g0: Grammar, example: str) -> tuple[list[Any], Grammar]:
    raw = _need(args, "bounds", example)
    names = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 3:
            raise ToolError("INVALID_ARGUMENT", "Each bound must be [variable, a, b].", example)
        names.append(check_identifier(str(item[0])))
    g = grammar(list(args.get("variables") or []) + names)
    lims = [
        (sp.Symbol(n), value_of(i[1], g), value_of(i[2], g))
        for n, i in zip(names, raw, strict=True)
    ]
    return lims, g


def _numeric_multi(integrand: Any, lims: list[tuple[Any, Any, Any]]) -> float:
    """Nested numerical integral of the region (scipy nquad, the first bound innermost), for the back-check.

    D91: before, Integral(...).evalf(15) stayed an unevaluated Integral for every multiple integral, so `verified`
    was always false in 2D, and in 3D it only said that sympy had found a closed form.
    """
    from scipy import integrate

    syms = [v for v, _, _ in lims]
    f = compile_expr(integrand, syms)

    def bound(e: Any, outer: list[Any]) -> Callable[..., float]:
        if not getattr(e, "free_symbols", None):
            c = float(e)
            return lambda *_: c
        return compile_expr(e, outer)

    ranges = []
    for i, (_, a, b) in enumerate(lims):
        fa, fb = bound(a, syms[i + 1 :]), bound(b, syms[i + 1 :])
        ranges.append(lambda *outer, fa=fa, fb=fb: [fa(*outer), fb(*outer)])
    return float(integrate.nquad(f, ranges, opts={"limit": 100})[0])


@op("multi_integrate")
def _multi(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    lims, g = _bound_list(args, _g(args), '[["x", 0, 1], ["y", 0, 2]]')
    check_bound_order(lims)  # D91
    f = expr_of(_need(args, "expr", "x*y"), g)
    coords = args.get("coords", "cartesian")
    names = {str(v) for v, _, _ in lims}
    jac: Any = 1
    if coords in {"polar", "cylindrical"}:
        if "r" not in names:
            raise ToolError(
                "INVALID_ARGUMENT",
                "In polar/cylindrical coordinates the variable 'r' is needed.",
                '[["r",0,1],["theta",0,"2*pi"]]',
            )
        jac = sp.Symbol("r")
    elif coords == "spherical":
        if not {"r", "theta"} <= names:
            raise ToolError(
                "INVALID_ARGUMENT",
                "In spherical coordinates 'r', 'theta' (polar), 'phi' (azimuth) are needed.",
                '[["r",0,1],["theta",0,"pi"],["phi",0,"2*pi"]]',
            )
        jac = sp.Symbol("r") ** 2 * sp.sin(sp.Symbol("theta"))
    integrand = f * jac
    val = sp.simplify(sp.integrate(integrand, *lims))
    num = _num(val)
    verified = False
    if num is not None:
        try:
            chk = _numeric_multi(integrand, lims)
            verified = abs(chk - num) <= 1e-8 * (1 + abs(num))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            verified = False
    ctx.formatted = f"Multiple integral = {val}"
    if coords != "cartesian":
        ctx.warn(f"Jacobian added automatically: {jac}.")
    return {"integral": str(val), "numeric": num, "verified": verified}


def _param_curve(text: str, g: Grammar) -> dict[str, Any]:
    comps: dict[str, Any] = {}
    for part in split_list(text):
        eq = split_equation(part)
        if eq is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "The curve is written 'x=...; y=...; z=...'.",
                "x=cos(t); y=sin(t); z=0",
            )
        name = check_identifier(eq[0].strip())
        comps[name] = parse(eq[1], g)
    return comps


@op("line_integral")
def _line(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args, ["x", "y", "z"])
    t = symbols_for(_need(args, "variables", '["t"]'))[0]
    curve = _param_curve(_need(args, "expr", "x=cos(t); y=sin(t); z=0"), g)
    bounds = _need(args, "bounds", '[0, "2*pi"]')
    a, b = value_of(bounds[0], g), value_of(bounds[1], g)
    field = [parse(c, g) for c in _need(args, "vector_field", '["-y", "x", "0"]')]
    xyz = [sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")]
    r = [curve.get(str(s), sp.Integer(0)) for s in xyz]
    subs = dict(zip(xyz, r, strict=True))
    dr = [sp.diff(c, t) for c in r]
    if len(field) == 1:
        integrand = field[0].subs(subs) * sp.sqrt(sum(d**2 for d in dr))
    else:
        comps = (field + [sp.Integer(0)] * 3)[:3]
        integrand = sp.Add(*(c.subs(subs) * d for c, d in zip(comps, dr, strict=True)))
    integrand = sp.simplify(integrand)
    val = sp.simplify(sp.integrate(integrand, (t, a, b)))
    num = _num(val)
    try:
        verified = num is not None and abs(
            complex(sp.Integral(integrand, (t, a, b)).evalf(15)) - num
        ) < 1e-8 * (1 + abs(num))
    except (TypeError, ValueError):
        verified = False
    ctx.formatted = f"Line integral = {val}"
    return {"integral": str(val), "numeric": num, "verified": verified}


@op("surface_integral")
def _surface(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    lims, g = _bound_list(args, _g(args), '[["u", 0, "pi"], ["v", 0, "2*pi"]]')
    if len(lims) != 2:
        raise ToolError(
            "INVALID_ARGUMENT", "A surface has two parameters.", '[["u",0,"pi"],["v",0,"2*pi"]]'
        )
    g = grammar([str(v) for v, _, _ in lims] + ["x", "y", "z"])
    surf = _param_curve(_need(args, "expr", "x=sin(u)*cos(v); y=sin(u)*sin(v); z=cos(u)"), g)
    field = [parse(c, g) for c in _need(args, "vector_field", '["x", "y", "z"]')]
    xyz = [sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")]
    rvec = [surf.get(str(s), sp.Integer(0)) for s in xyz]
    r = sp.Matrix(rvec)
    u, v = lims[0][0], lims[1][0]
    n = cast(sp.Matrix, r.diff(u)).cross(cast(sp.Matrix, r.diff(v)))
    subs = dict(zip(xyz, rvec, strict=True))
    if len(field) == 1:
        integrand = field[0].subs(subs) * sp.sqrt(sum(c**2 for c in n))
    else:
        F = sp.Matrix([c.subs(subs) for c in (field + [sp.Integer(0)] * 3)[:3]])
        integrand = F.dot(n)
    integrand = sp.trigsimp(sp.simplify(integrand))
    val = sp.simplify(sp.integrate(integrand, *lims))
    num = _num(val)
    try:
        verified = num is not None and abs(
            complex(sp.Integral(integrand, *lims).evalf(12)) - num
        ) < 1e-7 * (1 + abs(num))
    except (TypeError, ValueError):
        verified = False
    ctx.formatted = f"Surface integral = {val}"
    return {"integral": str(val), "numeric": num, "verified": verified}


# ------------------------------------------------------------------ limits and series


@op("limit")
def _limit(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "sin(x)/x"), g)
    x = auto_vars(f, args.get("variables"))[0]
    bounds = _need(args, "bounds", "[0]")
    point = value_of(bounds[0], g)
    direction = str(bounds[1]) if len(bounds) > 1 else "+-"
    if direction not in {"+", "-", "+-"}:
        raise ToolError("INVALID_ARGUMENT", "The direction is '+', '-' or '+-'.", '[0, "+"]')
    if point in (sp.oo, -sp.oo):
        direction = "-" if point == sp.oo else "+"
    lim = sp.limit(f, x, point, dir=direction)
    verified = False
    lv = _num(lim)
    try:
        # Decimal test points with 50 digits: with exact integers (1 + 1/n)**n for n = 10**8 becomes
        # a fraction with hundreds of millions of digits and the computation never ends.
        if point in (sp.oo, -sp.oo):
            probe = [sp.Float(f"{'' if point == sp.oo else '-'}1e{k}", 50) for k in (6, 8)]
        else:
            sgn = -1 if direction == "-" else 1
            base = sp.N(point, 50)
            probe = [base + sgn * sp.Float(f"1e-{k}", 50) for k in (6, 8)]
        vals = [complex(f.evalf(50, subs={x: p})) for p in probe]
        if lv is not None:
            verified = abs(vals[-1] - lv) < 1e-3 * (1 + abs(lv))
        elif lim in (sp.oo, -sp.oo):
            verified = abs(vals[-1]) > abs(vals[0]) and abs(vals[-1]) > 1e5
    except (TypeError, ValueError, ZeroDivisionError):
        verified = False
    if ctx.show_steps:
        try:
            direct = f.subs(x, point)
        except (TypeError, ValueError):
            direct = sp.nan
        ctx.step(f"Direct substitution {x} = {point}: {direct}.", direct)
        if direct.has(sp.nan, sp.zoo) or direct is sp.nan:
            num, den = sp.fraction(sp.together(f))
            n0, d0 = sp.limit(num, x, point), sp.limit(den, x, point)
            if n0 == 0 and d0 == 0:
                ctx.step(
                    "Indeterminate form 0/0: applying l'Hôpital (differentiating numerator and denominator)."
                )
                ctx.step(f"lim {sp.diff(num, x)} / {sp.diff(den, x)}.")
            else:
                ctx.step(
                    "Indeterminate form: using asymptotic expansions (SymPy Gruntz algorithm)."
                )
        ctx.step(f"Result: {lim}.", lim)
        ctx.step(
            f"Numerical check approaching the point: {'consistent' if verified else 'not verified'}."
        )
    ctx.formatted = f"lim for {x} → {point} of {f} = {lim}"
    return {"limit": str(lim), "verified": verified}


@op("series")
def _series(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "exp(x)"), g)
    x = auto_vars(f, args.get("variables"))[0]
    bounds = args.get("bounds") or [0]
    x0 = value_of(bounds[0], g)
    n = int(args.get("order", 6) or 6)
    if n > 60:
        raise ToolError("LIMIT_EXCEEDED", "order at most 60 for series.")
    s = sp.series(f, x, x0, n)
    poly = s.removeO()
    verified = all(
        is_zero(sp.diff(poly, x, k).subs(x, x0) - sp.limit(sp.diff(f, x, k), x, x0))
        for k in range(min(n, 6))
    )
    ctx.formatted = f"Series: {s}"
    return {"series": str(s), "polynomial": str(poly), "verified": verified}


@op("function_study")
def _study(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "x**3 - 3*x"), g)
    x = auto_vars(f, args.get("variables"))[0]
    from sympy.calculus.util import continuous_domain

    dom = continuous_domain(f, x, sp.S.Reals)
    zeros_set = sp.solveset(f, x, dom)
    zeros = (
        [str(z) for z in sorted(zeros_set, key=sp.default_sort_key)]
        if isinstance(zeros_set, sp.FiniteSet)
        else []
    )
    d1 = sp.simplify(sp.diff(f, x))
    d2 = sp.simplify(sp.diff(f, x, 2))
    crit_set = sp.solveset(d1, x, dom)
    crit = sorted(crit_set, key=sp.default_sort_key) if isinstance(crit_set, sp.FiniteSet) else []
    cps = []
    eps = sp.Rational(1, 10**6)
    for c in crit:
        v2 = sp.simplify(d2.subs(x, c))
        if v2.is_negative:
            kind = "max"
        elif v2.is_positive:
            kind = "min"
        else:
            left, right = _num(d1.subs(x, c - eps)), _num(d1.subs(x, c + eps))
            kind = (
                "flex" if left is not None and right is not None and left * right > 0 else "unknown"
            )
        cps.append({"x": str(c), "y": str(sp.simplify(f.subs(x, c))), "type": kind})
    infl_set = sp.solveset(d2, x, dom)
    infl = []
    if isinstance(infl_set, sp.FiniteSet):
        for c in sorted(infl_set, key=sp.default_sort_key):
            left, right = _num(d2.subs(x, c - eps)), _num(d2.subs(x, c + eps))
            if left is not None and right is not None and left * right < 0:
                infl.append({"x": str(c), "y": str(sp.simplify(f.subs(x, c)))})
    lim_m, lim_p = sp.limit(f, x, -sp.oo), sp.limit(f, x, sp.oo)
    vert = []
    for s in sorted(sp.singularities(f, x, sp.S.Reals) if True else [], key=sp.default_sort_key):
        if s.is_real and sp.limit(f, x, s, dir="+") in (sp.oo, -sp.oo):
            vert.append(str(s))
    horiz = sorted({str(L) for L in (lim_m, lim_p) if L.is_finite})
    obl = []
    for L, inf in ((lim_m, -sp.oo), (lim_p, sp.oo)):
        if L in (sp.oo, -sp.oo):
            m = sp.limit(f / x, x, inf)
            if m.is_finite and m != 0:
                q = sp.limit(f - m * x, x, inf)
                if q.is_finite:
                    obl.append(str(m * x + q))
    y0 = str(sp.simplify(f.subs(x, 0))) if dom.contains(0) == sp.true else None
    ctx.formatted = f"Study of f(x) = {f}: zeros {zeros}, critical points {[c['x'] for c in cps]}"
    return {
        "domain": str(dom),
        "zeros": zeros,
        "y_intercept": y0,
        "derivative": str(d1),
        "critical_points": cps,
        "inflection_points": infl,
        "asymptotes": {"vertical": vert, "horizontal": horiz, "oblique": sorted(set(obl))},
        "limits": {"-oo": str(lim_m), "oo": str(lim_p)},
    }


# ------------------------------------------------------------------ algebra


def _transform(name: str, fn: Callable[[Any, list[Any]], Any]) -> Op:
    def run(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
        g = _g(args)
        f = expr_of(_need(args, "expr", "x**2 - 1"), g)
        xs = symbols_for(args.get("variables")) or sorted(f.free_symbols, key=lambda s: s.name)
        out = fn(f, xs)
        verified = is_zero(out - f)
        ctx.formatted = f"{name}: {out}"
        ctx.step(f"Expression: {f}.", f)
        ctx.step(f"{name}: {out}.", out)
        ctx.step(f"Back-check: difference from the original = 0? {'yes' if verified else 'no'}.")
        return {"expression": str(out), "verified": verified}

    return run


OPS["simplify"] = _transform("Simplification", lambda f, xs: sp.simplify(f))
OPS["factor"] = _transform("Factorisation", lambda f, xs: sp.factor(f))
OPS["expand"] = _transform("Expansion", lambda f, xs: sp.expand(f))
OPS["partial_fractions"] = _transform(
    "Partial fractions", lambda f, xs: sp.apart(f, xs[0]) if xs else sp.apart(f)
)


@op("gradient")
def _gradient(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "x**2 + y**2"), g)
    xs = auto_vars(f, args.get("variables"))
    grad = [sp.simplify(sp.diff(f, v)) for v in xs]
    ctx.formatted = f"∇f = {grad}"
    return {"gradient": [str(c) for c in grad]}


@op("laplacian")
def _laplacian(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "x**2 + y**2"), g)
    xs = auto_vars(f, args.get("variables"))
    lap = sp.simplify(sp.Add(*(sp.diff(f, v, 2) for v in xs)))
    ctx.formatted = f"∇²f = {lap}"
    return {"laplacian": str(lap)}


def _field(args: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    g = _g(args)
    comps = [parse(c, g) for c in _need(args, "vector_field", '["x*y", "x + y"]')]
    xs = symbols_for(args.get("variables")) or sorted(
        set().union(*[c.free_symbols for c in comps]), key=lambda s: s.name
    )
    return comps, xs


@op("divergence")
def _divergence(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    comps, xs = _field(args)
    if len(comps) != len(xs):
        raise ToolError(
            "INVALID_ARGUMENT",
            "As many components as variables are needed.",
            '["x","y","z"] with variables ["x","y","z"]',
        )
    div = sp.simplify(sp.Add(*(sp.diff(c, v) for c, v in zip(comps, xs, strict=True))))
    ctx.formatted = f"∇·F = {div}"
    return {"divergence": str(div)}


@op("curl")
def _curl(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    comps, xs = _field(args)
    if len(comps) != 3 or len(xs) != 3:
        raise ToolError(
            "INVALID_ARGUMENT",
            "The curl holds in 3D: 3 components and 3 variables.",
            '["-y","x","0"]',
        )
    P, Q, Rr = comps
    x, y, z = xs
    c = [
        sp.simplify(sp.diff(Rr, y) - sp.diff(Q, z)),
        sp.simplify(sp.diff(P, z) - sp.diff(Rr, x)),
        sp.simplify(sp.diff(Q, x) - sp.diff(P, y)),
    ]
    ctx.formatted = f"∇×F = {c}"
    return {"curl": [str(v) for v in c]}


@op("jacobian")
def _jacobian(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    comps, xs = _field(args)
    J = sp.Matrix(comps).jacobian(xs)
    det = str(sp.simplify(J.det())) if J.shape[0] == J.shape[1] else None
    ctx.formatted = f"J = {mat_str(J)}"
    return {"jacobian": mat_str(J), "determinant": det}


@op("hessian")
def _hessian(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "x**2 + y**2"), g)
    xs = auto_vars(f, args.get("variables"))
    H = sp.hessian(f, xs)
    crit = sp.solve([sp.diff(f, v) for v in xs], xs, dict=True)
    pts = []
    for c in crit[:20]:
        if any(v.is_real is False for v in c.values()):
            continue
        Hc = H.subs(c)
        try:
            eig = [complex(sp.N(e)) for e in Hc.eigenvals(multiple=True)]
        except (TypeError, ValueError, NotImplementedError):
            eig = [complex(e) for e in np.linalg.eigvalsh(np.array(Hc.evalf(), dtype=float))]
        re_ = [e.real for e in eig]
        if any(abs(e) < 1e-12 for e in re_):
            kind = "degenerate"
        elif all(e > 0 for e in re_):
            kind = "minimum"
        elif all(e < 0 for e in re_):
            kind = "maximum"
        else:
            kind = "saddle"
        pts.append({"point": {str(v): str(c.get(v, v)) for v in xs}, "type": kind})
    ctx.formatted = f"Hessian: {mat_str(H)}"
    return {"hessian": mat_str(H), "critical_points": pts}


@op("lagrange_optimize")
def _lagrange(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "x + y"), g)
    xs = auto_vars(f, args.get("variables"))
    eqs, ineqs = [], []
    for c in _need(args, "constraints", '["x**2 + y**2 = 1"]'):
        rel = parse_relation(c, g)
        if isinstance(rel, sp.Eq):
            lhs, rhs, _ = rel_parts(rel)
            eqs.append(lhs - rhs)
        elif isinstance(rel, sp.core.relational.Relational):
            ineqs.append(rel)
        else:
            eqs.append(rel)
    if len(ineqs) > 4:
        raise ToolError("LIMIT_EXCEEDED", "At most 4 inequality constraints.")
    cands: list[dict[str, Any]] = []
    for k in range(len(ineqs) + 1):
        for active in itertools.combinations(ineqs, k):
            act = eqs + [r.lhs - r.rhs for r in active]
            lams = [sp.Symbol(f"lambda{i + 1}") for i in range(len(act))]
            L = f - sum(lam * c for lam, c in zip(lams, act, strict=True))
            system = [sp.diff(L, v) for v in xs] + act
            for s in sp.solve(system, xs + lams, dict=True):
                if any(s.get(v, v).is_real is False for v in xs):
                    continue
                if not all(v in s for v in xs):
                    continue
                if not all(bool(r.subs(s)) for r in ineqs if r not in active):
                    continue
                val = sp.simplify(f.subs(s))
                cands.append(
                    {
                        "point": {str(v): str(sp.simplify(s[v])) for v in xs},
                        "multipliers": {str(lam): str(s.get(lam, 0)) for lam in lams},
                        "value": str(val),
                        "_num": _num(val),
                    }
                )
    finite = [c for c in cands if c["_num"] is not None]
    best_max = max(finite, key=lambda c: c["_num"]) if finite else None
    best_min = min(finite, key=lambda c: c["_num"]) if finite else None
    for c in cands:
        c.pop("_num", None)
    if ineqs:
        ctx.warn("Inequality constraints handled with KKT over active sets (at most 4).")
    ctx.formatted = f"Maximum: {best_max['value'] if best_max else '-'}; minimum: {best_min['value'] if best_min else '-'}"
    return {"candidates": cands, "maximum": best_max, "minimum": best_min}


# ------------------------------------------------------------------ ODEs and transforms

_PRIME = re.compile(r"(?<![\w.])([A-Za-z]\w*)('+)(?=\s*\()")


def _ode_text(text: str, deps: list[str], indep: str) -> str:
    for name in deps:
        text = re.sub(
            rf"(?<![\w.]){re.escape(name)}('*)(?![\w(])",
            lambda m, name=name: (
                f"{name}({indep})"
                if not m.group(1)
                else f"diff({name}({indep}), {indep}, {len(m.group(1))})"
            ),
            text,
        )
    return text


def _ics(raw: dict[str, Any] | None, funcs: dict[str, Any], x: Any, g: Grammar) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for key, val in (raw or {}).items():
        m = re.fullmatch(r"\s*([A-Za-z]\w*)('*)\((.+)\)\s*", str(key))
        if not m or m.group(1) not in funcs:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Invalid initial condition: {key!r}.",
                '{"y(0)": 1, "y\'(0)": 0}',
            )
        f = funcs[m.group(1)](x)
        k = len(m.group(2))
        x0 = value_of(m.group(3), g)
        lhs = f.subs(x, x0) if k == 0 else sp.diff(f, x, k).subs(x, x0)
        out[lhs] = value_of(val, g)
    return out


def _ode_system_check(eqs: list[Any], sols: list[Any]) -> bool:
    """Each equation of the system, with every function replaced by its solution, simplifies to zero (D90)."""
    rep = {s.lhs: s.rhs for s in sols}
    try:
        return bool(sols) and all(
            is_zero(sp.simplify((e.lhs - e.rhs).subs(rep).doit())) for e in eqs
        )
    except (TypeError, ValueError, NotImplementedError, AttributeError):
        return False


@op("ode")
def _ode(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    names = list(args.get("variables") or ["y", "x"])
    if len(names) < 2:
        raise ToolError(
            "INVALID_ARGUMENT", "variables = [dependent..., independent].", '["y", "x"]'
        )
    deps, indep = [check_identifier(n) for n in names[:-1]], check_identifier(names[-1])
    x = sp.Symbol(indep)
    funcs = {d: sp.Function(d) for d in deps}
    g = Grammar(SOLVE_FUNCS, {indep: x, **funcs})
    text = _ode_text(_need(args, "expr", "y'' + y = 0"), deps, indep)
    eqs = []
    for part in split_list(text):
        rel = parse_relation(part, g)
        eqs.append(rel if isinstance(rel, sp.Eq) else sp.Eq(rel, 0))
    ics = _ics(args.get("ics"), funcs, x, g)
    targets = [funcs[d](x) for d in deps]
    try:
        sol = sp.dsolve(
            eqs[0] if len(eqs) == 1 else eqs,
            targets[0] if len(targets) == 1 else targets,
            ics=ics or None,
        )
    except (NotImplementedError, ValueError) as exc:
        if args.get("bounds") and ics:
            return _ode_numeric(args, ctx, eqs, funcs, deps, x, ics)
        raise ToolError(
            "MATH_ERROR",
            f"ODE not solved in closed form: {str(exc)[:200]}. Pass ics and bounds for the numerical solution.",
        ) from None
    sols: list[Any] = sol if isinstance(sol, list) else [sol]
    verified = (
        all(sp.checkodesol(eqs[0], s)[0] is True for s in sols)
        if len(eqs) == 1
        else _ode_system_check(eqs, sols)  # D90: it was True without any check
    )
    text_sol = str(sols[0].rhs) if len(sols) == 1 else "; ".join(f"{s.lhs} = {s.rhs}" for s in sols)
    ctx.formatted = f"Solution: {text_sol}"
    ctx.step(f"Differential equation: {eqs}.")
    ctx.step(f"Initial conditions: {ics or 'none (constants C1, C2, ...)'}.")
    ctx.step(f"Solution (dsolve): {text_sol}.", text_sol)
    ctx.step(
        f"Back-check: substituting the solution into the equation (checkodesol): {'zero' if verified else 'not zero'}."
    )
    return {"solution": text_sol, "method": "symbolic", "verified": verified}


def _ode_numeric(
    args: dict[str, Any],
    ctx: Ctx,
    eqs: list[Any],
    funcs: dict[str, Any],
    deps: list[str],
    x: Any,
    ics: dict[Any, Any],
) -> dict[str, Any]:
    from scipy.integrate import solve_ivp

    if len(eqs) != 1 or len(deps) != 1:
        raise ToolError(
            "MATH_ERROR",
            "The numerical solution here handles a single equation: use math_numeric.ode_ivp.",
        )
    y = funcs[deps[0]](x)
    eq = eqs[0].lhs - eqs[0].rhs
    order = max((d.derivative_count for d in eq.atoms(sp.Derivative)), default=0)
    top = sp.solve(eq, sp.diff(y, x, order))
    if not top:
        raise ToolError("MATH_ERROR", "Cannot isolate the highest-order derivative.")
    states = [sp.Symbol(f"s{k}") for k in range(order)]
    rhs = top[0]
    for k in range(order - 1, -1, -1):
        rhs = rhs.subs(sp.diff(y, x, k) if k else y, states[k])
    fn = compile_expr(rhs, [x, *states])
    b = args["bounds"]
    g = grammar([])
    t0, t1 = float(value_of(b[0], g)), float(value_of(b[1], g))
    y0 = []
    for k in range(order):
        key = y.subs(x, t0) if k == 0 else sp.diff(y, x, k).subs(x, t0)
        if key not in ics:
            raise ToolError(
                "INVALID_ARGUMENT", f"Missing initial condition of order {k} at x = {t0}."
            )
        y0.append(float(ics[key]))

    def f(t: float, s: Any) -> list[float]:
        return [*list(s[1:]), fn(t, *s)]

    res = solve_ivp(f, (t0, t1), y0, rtol=1e-10, atol=1e-12, t_eval=np.linspace(t0, t1, 11))
    ctx.warn("RK45 numerical solution: SymPy finds no closed form.")
    return {
        "solution": None,
        "method": "numeric",
        "verified": bool(res.success),
        "x": res.t.tolist(),
        "y": res.y[0].tolist(),
    }


@op("laplace")
def _laplace(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    names = list(args.get("variables") or ["t", "s"])
    t = sp.Symbol(check_identifier(names[0]), positive=True)
    s = sp.Symbol(check_identifier(names[1] if len(names) > 1 else "s"))
    g = Grammar(SOLVE_FUNCS, {str(t): t, str(s): s})
    f = expr_of(_need(args, "expr", "exp(-2*t)"), g)
    F = cast(sp.Expr, sp.laplace_transform(f, t, s, noconds=True))
    try:
        back = cast(sp.Expr, sp.inverse_laplace_transform(F, s, t))
        verified = is_zero(sp.simplify(back.subs(sp.Heaviside(t), 1)) - f)
    except (NotImplementedError, ValueError, TypeError):
        verified = False
    ctx.formatted = f"L{{{f}}} = {F}"
    return {"transform": str(sp.simplify(F)), "verified": verified}


@op("inv_laplace")
def _inv_laplace(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    names = list(args.get("variables") or ["s", "t"])
    s = sp.Symbol(check_identifier(names[0]))
    t = sp.Symbol(check_identifier(names[1] if len(names) > 1 else "t"), positive=True)
    g = Grammar(SOLVE_FUNCS, {str(t): t, str(s): s})
    F = expr_of(_need(args, "expr", "1/(s**2 + 1)"), g)
    f = cast(sp.Expr, sp.inverse_laplace_transform(F, s, t))
    try:
        verified = is_zero(sp.laplace_transform(f, t, s, noconds=True) - F)
    except (NotImplementedError, ValueError, TypeError):
        verified = False
    ctx.formatted = f"L^-1{{{F}}} = {f}"
    return {"transform": str(sp.simplify(f)), "verified": verified}


@op("fourier_fft")
def _fourier(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    samples = args.get("numeric_samples")
    if samples:
        data = np.asarray(samples, dtype=float)
        n = len(data)
        bounds = args.get("bounds") or []
        fs = float(bounds[0]) if bounds else 1.0
        spec = np.fft.rfft(data)
        mags = np.abs(spec) / n
        mags[1:] *= 2
        if n % 2 == 0:
            mags[-1] /= 2
        freqs = np.fft.rfftfreq(n, d=1 / fs)
        k = int(np.argmax(mags[1:]) + 1) if n > 2 else 0
        ctx.formatted = f"Dominant frequency: {freqs[k]}"
        if not bounds:
            ctx.warn("Without a sampling frequency the frequencies are in cycles per sample.")
        return {
            "frequencies": freqs.tolist(),
            "magnitudes": [round(float(m), 12) for m in mags],
            "phases": [round(float(p), 12) for p in np.angle(spec)],
            "dominant_frequency": float(freqs[k]),
        }
    g = _g(args)
    f = expr_of(_need(args, "expr", "x"), g)
    x = auto_vars(f, args.get("variables"))[0]
    b = _need(args, "bounds", '["-pi", "pi"]')
    a, c = value_of(b[0], g), value_of(b[1], g)
    T = c - a
    N = int(args.get("order", 5) or 5)
    if N > 50:
        raise ToolError("LIMIT_EXCEEDED", "At most 50 harmonics.")
    w = 2 * sp.pi / T
    a0 = sp.simplify(2 / T * sp.integrate(f, (x, a, c)))
    an = [
        sp.simplify(2 / T * sp.integrate(f * sp.cos(k * w * x), (x, a, c))) for k in range(1, N + 1)
    ]
    bn = [
        sp.simplify(2 / T * sp.integrate(f * sp.sin(k * w * x), (x, a, c))) for k in range(1, N + 1)
    ]
    series = a0 / 2 + sum(
        an[k - 1] * sp.cos(k * w * x) + bn[k - 1] * sp.sin(k * w * x) for k in range(1, N + 1)
    )
    ctx.formatted = f"Fourier series: {series}"
    return {
        "a0": str(a0),
        "an": [str(v) for v in an],
        "bn": [str(v) for v in bn],
        "series": str(series),
    }


@op("riccati_lqr")
def _riccati(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    from scipy import linalg

    def arr(key: str) -> Any:
        rows = _need(args, key, "[[0, 1], [0, 0]]")
        m = matrix_of(rows, grammar([]), key)
        return np.array(m.evalf(), dtype=float)

    A, B, Q, Rm = arr("matrix_a"), arr("matrix_b"), arr("matrix_q"), arr("matrix_r")
    if args.get("time", "continuous") == "discrete":
        P = linalg.solve_discrete_are(A, B, Q, Rm)
        K = np.linalg.solve(Rm + B.T @ P @ B, B.T @ P @ A)
        res = A.T @ P @ A - P - A.T @ P @ B @ K + Q
        eig = np.linalg.eigvals(A - B @ K)
    else:
        P = linalg.solve_continuous_are(A, B, Q, Rm)
        K = np.linalg.solve(Rm, B.T @ P)
        res = A.T @ P + P @ A - P @ B @ K + Q
        eig = np.linalg.eigvals(A - B @ K)
    rn = float(np.linalg.norm(res))
    verified = rn < 1e-8 * (1 + float(np.linalg.norm(P)))
    ctx.formatted = f"Gain K = {K.tolist()}"
    return {
        "P": P.tolist(),
        "K": K.tolist(),
        "closed_loop_eigenvalues": [str(complex(e)) for e in eig],
        "residual_norm": rn,
        "verified": verified,
    }


# ------------------------------------------------------------------ matrices


@op("matrix")
def _matrix(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    A = matrix_of(_need(args, "matrix_a", "[[1, 2], [3, 4]]"), g)
    mop = _need(args, "matrix_op", '"det"')
    exact = (
        all(e.is_Rational or not e.is_Float for e in A.flat())
        and A.shape[0] <= 10
        and not any(e.is_Float for e in A.flat())
    )
    if not exact and mop not in {"svd"}:
        ctx.warn("Large matrix or with decimals: floating-point numerical computation.")
    B = matrix_of(args["matrix_b"], g, "matrix_b") if args.get("matrix_b") else None
    eye = sp.eye(A.shape[0])
    out: dict[str, Any] = {"matrix_op": mop}
    if mop == "det":
        d = A.det() if exact else sp.Float(np.linalg.det(np.array(A.evalf(), dtype=float)))
        out.update(value=str(d), verified=True)
        ctx.step(f"Matrix A = {mat_str(A)}.")
        ctx.step(f"Determinant (expansion/Gaussian elimination): {d}.", d)
    elif mop == "inv":
        inv = A.inv()
        out.update(matrix=mat_str(inv), verified=sp.simplify(A * inv - eye) == sp.zeros(*A.shape))
    elif mop == "pinv":
        P = A.pinv()
        out.update(matrix=mat_str(P), verified=sp.simplify(A * P * A - A) == sp.zeros(*A.shape))
    elif mop == "cofactor":
        C = A.cofactor_matrix()
        out.update(
            matrix=mat_str(C), verified=sp.simplify(A * C.T - A.det() * eye) == sp.zeros(*A.shape)
        )
    elif mop == "projection":
        P = A * (A.T * A).inv() * A.T
        out.update(matrix=mat_str(P), verified=sp.simplify(P * P - P) == sp.zeros(*P.shape))
        if B is not None:
            out["projected"] = mat_str(P * B)
    elif mop == "norm_p":
        p = (args.get("expr") or "2").strip()
        ordv: Any = {"1": 1, "2": 2, "oo": sp.oo, "inf": sp.oo, "fro": "fro"}.get(p)
        if ordv is None:
            raise ToolError("INVALID_ARGUMENT", "Norm: expr = '1', '2', 'oo' or 'fro'.", '"fro"')
        out.update(value=str(sp.simplify(A.norm(ordv))), verified=True)
    elif mop == "transpose":
        out.update(matrix=mat_str(A.T), verified=A.T.T == A)
    elif mop == "rank":
        out.update(value=str(A.rank()), verified=True)
    elif mop == "rref":
        R_, piv = A.rref()
        out.update(matrix=mat_str(R_), pivots=list(piv), verified=R_.rref()[0] == R_)
    elif mop == "lu":
        L, U, perm = A.LUdecomposition()
        Pm = A.copy()
        for i, j in perm:
            Pm.row_swap(i, j)
        out.update(
            L=mat_str(L),
            U=mat_str(U),
            row_swaps=[list(p) for p in perm],
            verified=sp.simplify(L * U - Pm) == sp.zeros(*A.shape),
        )
    elif mop == "qr":
        Qm, Rq = A.QRdecomposition()
        out.update(
            Q=mat_str(Qm), R=mat_str(Rq), verified=sp.simplify(Qm * Rq - A) == sp.zeros(*A.shape)
        )
    elif mop == "svd":
        a = np.array(A.evalf(), dtype=float)
        U, S, Vt = np.linalg.svd(a)
        rec = U[:, : len(S)] @ np.diag(S) @ Vt[: len(S), :]
        out.update(
            U=U.tolist(),
            S=S.tolist(),
            Vt=Vt.tolist(),
            verified=bool(np.allclose(rec, a, atol=1e-10)),
        )
    elif mop == "cholesky":
        L = A.cholesky(hermitian=False)
        out.update(L=mat_str(L), verified=sp.simplify(L * L.T - A) == sp.zeros(*A.shape))
    elif mop == "eigenvals":
        ev = cast(dict[Any, int], A.eigenvals())
        out.update(
            eigenvalues={str(k): int(v) for k, v in ev.items()},
            verified=all(is_zero((A - k * eye).det()) for k in ev),
        )
    elif mop == "eigenvects":
        vecs = A.eigenvects()
        ok = all(
            sp.simplify(A * v - val * v) == sp.zeros(A.shape[0], 1)
            for val, _, vs in vecs
            for v in vs
        )
        out.update(
            eigenvectors=[
                {
                    "value": str(val),
                    "multiplicity": int(m),
                    "vectors": [[str(c) for c in v] for v in vs],
                }
                for val, m, vs in vecs
            ],
            verified=ok,
        )
    elif mop == "solve":
        if B is None:
            raise ToolError(
                "INVALID_ARGUMENT", "'solve' needs matrix_b (right-hand side).", "[[5], [6]]"
            )
        X = A.LUsolve(B) if A.shape[0] == A.shape[1] else A.pinv() * B
        out.update(
            solution=mat_str(sp.simplify(X)), verified=sp.simplify(A * X - B) == sp.zeros(*B.shape)
        )
    elif mop == "multiply":
        if B is None:
            raise ToolError(
                "INVALID_ARGUMENT", "With 'multiply' matrix_b is needed.", "[[1, 0], [0, 1]]"
            )
        out.update(matrix=mat_str(A * B), verified=True)
    out["verified"] = bool(out.get("verified", False))
    ctx.formatted = f"Matrix, {mop}: completed"
    return out


@op("optimize_num")
def _optimize_num(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    from . import math_numeric

    sub = {
        "operation": "minimize",
        "expr": args.get("expr"),
        "variables": args.get("variables"),
        "constraints": args.get("constraints"),
        "x0": args.get("numeric_samples"),
        "bounds": None,
        "method": "auto",
        "direction": "minimize",
    }
    return math_numeric.handle({k: v for k, v in sub.items() if v is not None}, ctx)


@op("complex_ops")
def _complex(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    z = expr_of(_need(args, "expr", "3 + 4*I"), g)
    re_, im_ = sp.re(z), sp.im(z)
    mod = sp.simplify(sp.Abs(z))
    ang = sp.simplify(sp.arg(z))
    ctx.formatted = f"z = {sp.expand_complex(z)}, |z| = {mod}"
    return {
        "rectangular": str(sp.expand_complex(z)),
        "re": str(re_),
        "im": str(im_),
        "modulus": str(mod),
        "argument_rad": str(ang),
        "argument_deg": _num(ang * 180 / sp.pi),
        "conjugate": str(sp.expand_complex(sp.conjugate(z))),
        "polar": f"{mod}*exp(I*{ang})",
    }


@op("summation")
def _summation(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    term = expr_of(_need(args, "expr", "1/k**2"), g)
    k = auto_vars(term, args.get("variables"))[0]
    b = _need(args, "bounds", '[1, "oo"]')
    lo, hi = value_of(b[0], g), value_of(b[1], g)
    S = cast(sp.Expr, sp.simplify(sp.summation(term, (k, lo, hi))))
    verified = False
    if hi.free_symbols:
        n = next(iter(hi.free_symbols))
        verified = all(
            is_zero(
                S.subs(n, m)
                - sp.Add(*(term.subs(k, j) for j in range(int(lo), int(hi.subs(n, m)) + 1)))
            )
            for m in range(int(lo), int(lo) + 5)
        )
    elif hi == sp.oo:
        import mpmath

        fn = compile_expr(term, [k])
        mpmath.mp.dps = 30
        approx = mpmath.nsum(lambda j: fn(float(j)), [int(lo), mpmath.inf])
        sv = _num(S)
        verified = sv is not None and abs(float(approx) - sv) < 1e-9 * (1 + abs(sv))
    else:
        verified = is_zero(S - sp.Add(*(term.subs(k, j) for j in range(int(lo), int(hi) + 1))))
    ctx.formatted = f"Sum = {S}"
    return {"sum": str(S), "numeric": _num(S) if not S.free_symbols else None, "verified": verified}


@op("recurrence")
def _recurrence(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    names = list(args.get("variables") or ["a", "n"])
    aname, nname = check_identifier(names[0]), check_identifier(names[1])
    a, n = sp.Function(aname), sp.Symbol(nname, integer=True)
    g = Grammar(SOLVE_FUNCS, {nname: n, aname: a})
    rel = parse_relation(_need(args, "expr", "a(n) = a(n-1) + a(n-2)"), g)
    if isinstance(rel, sp.Eq):
        lhs, rhs, _ = rel_parts(rel)
        eq = lhs - rhs
    else:
        eq = rel
    ics: dict[Any, Any] = {}
    init: dict[int, Any] = {}
    for key, val in (args.get("ics") or {}).items():
        m = re.fullmatch(rf"\s*{re.escape(aname)}\((-?\d+)\)\s*", str(key))
        if not m:
            raise ToolError(
                "INVALID_ARGUMENT", f"Invalid condition: {key!r}.", '{"a(0)": 0, "a(1)": 1}'
            )
        idx = int(m.group(1))
        v = value_of(val, grammar([]))
        ics[a(idx)] = v
        init[idx] = v
    sol = sp.rsolve(eq, a(n), ics or None)
    if sol is None:
        raise ToolError("MATH_ERROR", "Recurrence not solved in closed form.")
    rhs_list = sp.solve(eq, a(n))
    terms: dict[int, Any] = dict(init)
    if rhs_list and init:
        rhs = rhs_list[0]
        start = max(init) + 1
        for k in range(start, start + 15):
            expr_k = rhs.subs(n, k)
            expr_k = expr_k.replace(
                lambda e: isinstance(e, sp.core.function.AppliedUndef) and e.func == a,
                lambda e: terms.get(int(e.args[0]), e),
            )
            terms[k] = sp.simplify(expr_k)
    first = (
        [terms[k] for k in sorted(terms)][:15]
        if terms
        else [sp.simplify(sol.subs(n, k)) for k in range(10)]
    )
    verified = (
        all(is_zero(sp.simplify(sol.subs(n, k)) - v) for k, v in sorted(terms.items())[:12])
        if terms
        else False
    )
    ctx.formatted = f"{aname}({nname}) = {sol}"
    return {
        "solution": str(sp.simplify(sol)),
        "first_terms": [str(sp.nsimplify(v)) for v in first],
        "verified": verified,
    }


def _at(v: Any, subs: dict[Any, Any]) -> Any:
    return v.subs(subs) if isinstance(v, sp.Basic) else v


def _close(a: Any, b: Any) -> bool:
    va, vb = complex(sp.N(a, 30)), complex(sp.N(b, 30))
    return abs(va - vb) <= 1e-12 * max(1.0, abs(va), abs(vb))


def _poly_verify(pop: str, p: Any, q: Any, x: Any, value: Any) -> bool:
    """Independent check of a `polynomial` result (D90; before, `verified` was True without a check).

    The other symbols get rational values; the roots come from nroots at 30 digits. discriminant =
    a^(2n-2) prod (r_i - r_j)^2, resultant = a^deg(q) prod q(r_i), vieta from prod (t - r_i), nroots by the
    residual relative to the size of the terms, coeffs by rebuilding the polynomial, degree by the limit of
    p / x^n at infinity (finite and nonzero).
    """
    try:
        if pop == "coeffs":
            n = len(value) - 1
            return is_zero(sp.expand(sum(c * x ** (n - i) for i, c in enumerate(value)) - p))
        syms = p.free_symbols | (q.free_symbols if q is not None else set())
        subs = {
            s: sp.Rational(7 + 2 * i, 5)
            for i, s in enumerate(sorted(syms - {x}, key=lambda s: s.name))
        }
        if pop == "degree":
            lim = sp.limit(p.subs(subs) / x ** int(value), x, sp.oo)
            return bool(lim.is_finite) and lim != 0
        P = sp.Poly(p.subs(subs), x)
        if pop == "nroots":
            if len(value) != P.degree():
                return False
            for r in value:
                terms = [
                    abs(complex(sp.N(c * r**k, 30))) for k, c in enumerate(reversed(P.all_coeffs()))
                ]
                if not abs(complex(sp.N(P.as_expr().subs(x, r), 30))) <= 1e-20 * max(terms):
                    return False
            return True
        roots = P.nroots(n=30)
        a, n = P.LC(), int(P.degree())  # -oo (zero polynomial): TypeError, False
        if pop == "discriminant":
            pr = sp.Mul(*[(roots[i] - roots[j]) ** 2 for i in range(n) for j in range(i + 1, n)])
            return _close(a ** (2 * n - 2) * pr, _at(value, subs))
        if pop == "resultant":
            if q is None:
                return False
            Q = sp.Poly(q.subs(subs), x)
            return _close(
                a ** Q.degree() * sp.Mul(*[Q.as_expr().subs(x, r) for r in roots]), _at(value, subs)
            )
        if pop == "vieta":
            t = sp.Dummy("t")
            cs = sp.Poly(sp.Mul(*[t - r for r in roots]), t).all_coeffs()
            return len(value) == n and all(
                _close((-1) ** k * cs[k], _at(value[f"e{k}"], subs)) for k in range(1, n + 1)
            )
    except (TypeError, ValueError, ZeroDivisionError, NotImplementedError, sp.PolynomialError):
        return False
    return False


@op("polynomial")
def _polynomial(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    polys = [expr_of(t, g) for t in split_list(_need(args, "expr", "x**3 - 6*x**2 + 11*x - 6"))]
    x = auto_vars(polys[0], args.get("variables"))[0]
    pop = args.get("poly_op", "roots")
    p = polys[0]
    P = sp.Poly(p, x)
    value: Any
    q = polys[1] if len(polys) > 1 else None
    verified = True
    if pop == "roots":
        r = sp.roots(P)
        value = {str(k): int(v) for k, v in r.items()}
        verified = all(is_zero(p.subs(x, k)) for k in r) and sum(r.values()) == P.degree()
        if sum(r.values()) != P.degree():
            ctx.warn("Some roots have no closed form: use poly_op 'nroots'.")
    elif pop == "nroots":
        nr = P.nroots(n=30)
        value = [str(v) for v in nr]
        verified = _poly_verify(pop, p, None, x, nr)
    elif pop == "factor_list":
        c, fl = cast(tuple[Any, list[tuple[Any, int]]], sp.factor_list(p, x))
        value = {"coefficient": str(c), "factors": [[str(f), int(m)] for f, m in fl]}
        verified = is_zero(c * sp.Mul(*[f**m for f, m in fl]) - p)
    elif pop == "discriminant":
        disc = sp.discriminant(P)
        value = str(disc)
        verified = _poly_verify(pop, p, None, x, disc)
    elif pop in {"resultant", "gcd"}:
        if q is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Two polynomials separated by ';' are needed.",
                "x**2 - 1; x**2 + 2*x + 1",
            )
        if pop == "resultant":
            res = sp.resultant(p, q, x)
            value = str(res)
            verified = _poly_verify(pop, p, q, x, res)
        else:
            gg = sp.gcd(p, q)
            value = str(gg)
            verified = sp.rem(p, gg, x) == 0 and sp.rem(q, gg, x) == 0
    elif pop == "vieta":
        cs = P.all_coeffs()
        lead = cs[0]
        ev = {f"e{k}": sp.simplify((-1) ** k * cs[k] / lead) for k in range(1, len(cs))}
        value = {k: str(v) for k, v in ev.items()}
        verified = _poly_verify(pop, p, None, x, ev)
    elif pop == "coeffs":
        cs = P.all_coeffs()
        value = [str(c) for c in cs]
        verified = _poly_verify(pop, p, None, x, cs)
    else:
        value = str(P.degree())
        verified = _poly_verify("degree", p, None, x, P.degree())
    ctx.formatted = f"{pop}: {value}"
    return {"poly_op": pop, "value": value, "verified": verified}


# ------------------------------------------------------------------ thermodynamic relations (D98)
# D116: Joule-Thomson coefficient mu_JT = (dT/dp)_H for A, G, U and H (page "Joule-Thomson effect").

# natural variables of each potential; "eos" is an equation of state p(V, T)
_THERMO_VARS: dict[str, tuple[str, ...]] = {
    "U": ("S", "V"),
    "H": ("S", "p"),
    "A": ("T", "V"),
    "G": ("T", "p"),
    "S": ("U", "V", "N"),
    "Omega": ("T", "V", "mu"),
    "eos": ("V", "T"),
}
_THERMO_RESULTS = {"T", "p", "V", "S", "U", "H", "A", "G", "mu"}
_THERMO_POSITIVE = {"T", "V", "p", "N"}
# D117: (natural variable, Maxwell relation with N); the left side is d/dN of d(potential)/d(variable)
_THERMO_N_RELS = {
    "U": (("S", "(dT/dN)_S,V = (dmu/dS)_V,N"), ("V", "-(dp/dN)_S,V = (dmu/dV)_S,N")),
    "H": (("S", "(dT/dN)_S,p = (dmu/dS)_p,N"), ("p", "(dV/dN)_S,p = (dmu/dp)_S,N")),
    "A": (("T", "-(dS/dN)_T,V = (dmu/dT)_V,N"), ("V", "-(dp/dN)_T,V = (dmu/dV)_T,N")),
    "G": (("T", "-(dS/dN)_T,p = (dmu/dT)_p,N"), ("p", "(dV/dN)_T,p = (dmu/dp)_T,N")),
}


def _not_finite(e: Any) -> bool:
    """D117: a quantity or a side of a relation with a division by zero (zoo, nan, oo)."""
    return bool(e.has(sp.zoo, sp.nan, sp.oo))


def _thermo_quantities(
    pot: str, f: Any, sy: dict[str, Any]
) -> tuple[dict[str, Any], list[Any], list[Any]]:
    """Quantities, page relations (name, lhs, rhs) and first derivatives to check numerically (f, df, var)."""
    D = sp.diff
    N = sy["N"]
    q: dict[str, Any]
    if pot == "A":
        T, V = sy["T"], sy["V"]
        S, p = -D(f, T), -D(f, V)
        U = f + T * S
        # D117: denominators simplified before dividing, so that a zero sympy does not see at once gives
        # zoo and not a finite value (0 / C_p was 0 with C_p = 0 only after simplify)
        CV = sp.simplify(T * D(S, T))
        kappa = -1 / (V * sp.simplify(D(p, V)))
        alpha = kappa * D(p, T)
        q = {"S": S, "p": p, "U": U, "H": U + p * V, "G": f + p * V, "C_V": CV}
        q.update(
            {"C_p": sp.simplify(CV + T * V * alpha**2 / kappa), "kappa_T": kappa, "alpha": alpha}
        )
        # D116: Joule-Thomson coefficient (dT/dp)_H = V (alpha T - 1) / C_p ("Joule-Thomson effect")
        q["mu_JT"] = V * (T * alpha - 1) / q["C_p"]
        rels = [
            ("mu_JT = -(dH/dp)_T / C_p", q["mu_JT"], -(D(q["H"], V) / D(p, V)) / q["C_p"]),
            ("(dS/dV)_T = (dp/dT)_V", D(S, V), D(p, T)),
            ("U = -T^2 (d(A/T)/dT)_V", U, -(T**2) * D(f / T, T)),
            ("G = -V^2 (d(A/V)/dV)_T", f + p * V, -(V**2) * D(f / V, V)),
            ("(dU/dV)_T = T (dp/dT)_V - p", D(U, V), T * D(p, T) - p),
            ("-(dp/dV)_T = 1/(V kappa_T)", -D(p, V), 1 / (V * kappa)),
            ("(dT/dS)_V = T/C_V", 1 / D(S, T), T / CV),
        ]
        checks = [(f, -S, T), (f, -p, V)]
    elif pot == "G":
        T, P = sy["T"], sy["p"]
        S, V = -D(f, T), D(f, P)
        H = f + T * S
        Cp = sp.simplify(T * D(S, T))  # D117: simplified before dividing
        kappa = -sp.simplify(D(V, P)) / V
        alpha = D(V, T) / V
        q = {"S": S, "V": V, "H": H, "U": H - P * V, "A": f - P * V, "C_p": Cp}
        q.update({"C_V": Cp - T * V * alpha**2 / kappa, "kappa_T": kappa, "alpha": alpha})
        q["mu_JT"] = (T * D(V, T) - V) / Cp  # D116
        rels = [
            ("mu_JT = -(dH/dp)_T / C_p", q["mu_JT"], -D(H, P) / Cp),
            ("-(dS/dp)_T = (dV/dT)_p", -D(S, P), D(V, T)),
            ("H = -T^2 (d(G/T)/dT)_p", H, -(T**2) * D(f / T, T)),
            ("(dH/dp)_T = V - T (dV/dT)_p", D(H, P), V - T * D(V, T)),
            ("(dT/dS)_p = T/C_p", 1 / D(S, T), T / Cp),
            ("-(dp/dV)_T = 1/(V kappa_T)", -1 / D(V, P), 1 / (V * kappa)),
        ]
        checks = [(f, -S, T), (f, V, P)]
    elif pot == "U":
        S, V = sy["S"], sy["V"]
        T, p = D(f, S), -D(f, V)
        q = {"T": T, "p": p, "H": f + p * V, "A": f - T * S, "G": f + p * V - T * S}
        q["C_V"] = T / sp.simplify(D(T, S))  # D117: simplified before dividing
        # D116: mu_JT = d(T, H)/d(p, H) with Jacobians in S and V; the relation uses d(T, p)/d(S, V)
        H = q["H"]
        q["mu_JT"] = (D(T, S) * D(H, V) - D(T, V) * D(H, S)) / sp.simplify(
            D(p, S) * D(H, V) - D(p, V) * D(H, S)
        )
        tp = sp.simplify(D(T, S) * D(p, V) - D(T, V) * D(p, S))
        dv_dt, c_p = -D(p, S) / tp, T * D(p, V) / tp
        rels = [
            ("mu_JT = (T (dV/dT)_p - V) / C_p", q["mu_JT"], (T * dv_dt - V) / c_p),
            ("(dT/dV)_S = -(dp/dS)_V", D(T, V), -D(p, S)),
            ("(dT/dV)_S = d2U/dSdV", D(T, V), D(f, S, V)),
            ("(dT/dS)_V = T/C_V", D(T, S), T / q["C_V"]),
        ]
        checks = [(f, T, S), (f, -p, V)]
    elif pot == "H":
        S, P = sy["S"], sy["p"]
        T, V = D(f, S), D(f, P)
        U = f - P * V
        ts = sp.simplify(D(T, S))  # D117: simplified before dividing
        q = {"T": T, "V": V, "U": U, "G": f - T * S, "A": U - T * S, "C_p": T / ts}
        # D116: (dT/dp)_H = (dT/dp)_S + (dT/dS)_p (dS/dp)_H with (dS/dp)_H = -V / T
        q["mu_JT"] = D(T, P) - D(T, S) * V / T
        rels = [
            ("mu_JT = (T (dV/dT)_p - V) / C_p", q["mu_JT"], (T * D(V, S) / ts - V) / q["C_p"]),
            ("(dT/dp)_S = (dV/dS)_p", D(T, P), D(V, S)),
            ("(dT/dp)_S = d2H/dSdp", D(T, P), D(f, S, P)),
            ("(dT/dS)_p = T/C_p", D(T, S), T / q["C_p"]),
        ]
        checks = [(f, T, S), (f, V, P)]
    elif pot == "S":
        U, V = sy["U"], sy["V"]
        T = 1 / D(f, U)
        p = T * D(f, V)
        q = {"T": T, "p": p, "C_V": -(D(f, U) ** 2) / sp.simplify(D(f, U, 2))}
        rels = [
            ("(dS/dU)_V,N = 1/T", D(f, U), 1 / T),
            ("(dS/dV)_U,N = p/T", D(f, V), p / T),
            ("(d(1/T)/dV)_U = (d(p/T)/dU)_V", D(1 / T, V), D(p / T, U)),
        ]
        checks = [(f, 1 / T, U), (f, p / T, V)]
        if f.has(N):
            q["mu"] = -T * D(f, N)
            rels.append(("(dS/dN)_U,V = -mu/T", D(f, N), -q["mu"] / T))
            checks.append((f, -q["mu"] / T, N))
        return q, rels, checks
    elif pot == "Omega":
        T, V, MU = sy["T"], sy["V"], sy["mu"]
        S, p, Nq = -D(f, T), -D(f, V), -D(f, MU)
        U = f + T * S + MU * Nq
        A = f + MU * Nq
        q = {"S": S, "p": p, "N": Nq, "U": U, "A": A, "G": A + p * V, "H": U + p * V}
        rels = [
            ("(dS/dV)_T,mu = (dp/dT)_V,mu", D(S, V), D(p, T)),
            ("(dS/dmu)_T,V = (dN/dT)_V,mu", D(S, MU), D(Nq, T)),
            ("(dp/dmu)_T,V = (dN/dV)_T,mu", D(p, MU), D(Nq, V)),
        ]
        return q, rels, [(f, -S, T), (f, -p, V), (f, -Nq, MU)]
    else:  # equation of state p(V, T)
        V, T = sy["V"], sy["T"]
        p_t, p_v = D(f, T), D(f, V)
        kappa = -1 / (V * sp.simplify(p_v))
        alpha = kappa * p_t
        q = {"dp_dT_V": p_t, "dp_dV_T": p_v, "dS_dV_T": p_t, "dU_dV_T": T * p_t - f}
        q.update({"kappa_T": kappa, "alpha": alpha, "Cp_minus_Cv": T * V * alpha**2 / kappa})
        rels = [
            ("C_p - C_V = -T (dp/dT)_V^2 / (dp/dV)_T", T * V * alpha**2 / kappa, -T * p_t**2 / p_v),
            ("(dp/dT)_V = alpha / kappa_T", p_t, alpha / kappa),
        ]
        return q, rels, [(f, p_t, T), (f, p_v, V)]
    if f.has(N):
        q["mu"] = D(f, N)
        # D117: the Maxwell relations with N (page "Maxwell relations", "General Maxwell relationships"),
        # d/dN of the conjugate of each natural variable = d(mu)/d(that variable); before, the only relation
        # compared d(potential)/dN with itself
        for x, name in _THERMO_N_RELS[pot]:
            rels.append((name, D(D(f, sy[x]), N), D(q["mu"], sy[x])))
        checks.append((f, q["mu"], N))
    return q, rels, checks


def _thermo_values(
    at: Any, q: dict[str, Any], f: Any, sy: dict[str, Any], ctx: Ctx
) -> dict[str, float]:
    if not isinstance(at, dict) or not at:
        raise ToolError(
            "INVALID_ARGUMENT", "'at' is an object {name: number}.", '{"T": 300, "V": 0.0224}'
        )
    known = {s.name: s for s in f.free_symbols} | sy
    subs: dict[Any, Any] = {}
    for name, raw in at.items():
        if name not in known:
            raise ToolError("INVALID_ARGUMENT", f"at.{name} is not a symbol of the expression.")
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
            raise ToolError("INVALID_ARGUMENT", f"at.{name} must be a finite number.")
        subs[known[name]] = sp.Rational(repr(raw)) if isinstance(raw, float) else sp.Integer(raw)
    values: dict[str, float] = {}
    left: list[str] = []
    for name, e in q.items():
        v = e.subs(subs)
        if v.free_symbols:
            left.append(name)
            continue
        z = complex(sp.N(v, 30))
        if not (math.isfinite(z.real) and abs(z.imag) <= 1e-12 * max(1.0, abs(z.real))):
            left.append(name)
            continue
        values[name] = z.real
    if left:
        ctx.warn(f"No real value at the given point for: {', '.join(left)}.")
    return values


@op("thermo_relations")
def _thermo_relations(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    pot = args.get("potential")
    if pot not in _THERMO_VARS:
        raise ToolError(
            "INVALID_ARGUMENT",
            "'potential' is one of U (of S, V), H (of S, p), A (of T, V), G (of T, p), "
            "S (of U, V, N), Omega (grand potential of T, V, mu), eos (p of V, T).",
            '{"operation": "thermo_relations", "potential": "eos", '
            '"expr": "n*R*T/(V - n*b) - a*n**2/V**2"}',
        )
    names = set(_THERMO_VARS[pot]) | {"N"}
    sy = {
        n: sp.Symbol(n, positive=True) if n in _THERMO_POSITIVE else sp.Symbol(n, real=True)
        for n in names
    }
    f = expr_of(_need(args, "expr", "-n*R*T*(log(V/n) + 3/2*log(T))"), grammar(None, extra=sy))
    if _not_finite(f):  # D117: SymPy absorbs finite terms into oo*V, the quantities had no meaning
        raise ToolError(
            "INVALID_ARGUMENT", f"The expression of {pot} must be finite: no oo, zoo or nan."
        )
    if not any(f.has(sy[n]) for n in _THERMO_VARS[pot][:2]):
        raise ToolError(
            "INVALID_ARGUMENT",
            f"The expression of {pot} must depend on its natural variables "
            f"{', '.join(_THERMO_VARS[pot])}.",
        )
    results = _THERMO_RESULTS | ({"N"} if pot == "Omega" else set())
    clash = sorted({s.name for s in f.free_symbols} & (results - set(_THERMO_VARS[pot])))
    if clash:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"The expression uses {', '.join(clash)}, which this operation computes from {pot}; "
            f"write {pot} with {', '.join(_THERMO_VARS[pot])} and constants only.",
        )
    q, rels, checks = _thermo_quantities(pot, f, sy)
    # D117: a quantity with a division by zero (C_p = 0, (dp/dV)_T = 0) is left out with a warning; simplify
    # is not called on it (sympy stopped with KeyError zoo on alpha = 0 * zoo)
    # (the denominators are simplified in _thermo_quantities, so a zero shows up here as zoo or nan)
    undefined = [k for k, v in q.items() if _not_finite(v)]
    q = {k: sp.simplify(v) for k, v in q.items() if k not in undefined}
    if undefined:
        ctx.warn(f"Not defined for this expression (a division by zero): {', '.join(undefined)}.")
    relations: list[dict[str, Any]] = []
    unchecked: list[str] = []
    for name, lhs, rhs in rels:
        if _not_finite(lhs) or _not_finite(rhs):
            relations.append({"relation": name, "lhs": str(lhs), "rhs": str(rhs), "holds": None})
            unchecked.append(name)
            continue
        relations.append(
            {
                "relation": name,
                "lhs": str(sp.simplify(lhs)),
                "rhs": str(sp.simplify(rhs)),
                "holds": is_zero(lhs - rhs),
            }
        )
    if unchecked:
        ctx.warn(f"Relations not checked, a side is not finite: {'; '.join(unchecked)}.")
    checked = [r["holds"] for r in relations if r["holds"] is not None]
    # independent check: the first derivatives against mpmath finite differences (as D90)
    numeric = all(_check_derivative_numeric(e, d, [(v, 1)]) for e, d, v in checks)
    variables = list(_THERMO_VARS[pot])
    if pot in _THERMO_N_RELS and f.has(sy["N"]):
        variables.append("N")
    out: dict[str, Any] = {
        "potential": pot,
        "variables": variables,
        "quantities": {k: str(v) for k, v in q.items()},
        "relations": relations,
        "verified": numeric and bool(checked) and all(checked),
    }
    if args.get("at") is not None:
        out["values"] = _thermo_values(args["at"], q, f, sy, ctx)
    ctx.formatted = "; ".join(f"{k} = {v}" for k, v in out["quantities"].items())
    return out


# ------------------------------------------------------------------ wavefunctions (D99)

_WF_MAX_VARS = 6  # two particles in 3D
_WF_EXAMPLE = "(2*a/pi)**(1/4)*exp(-a*x**2 + I*k*x)"


def _wf_rename(e: Any, names: list[str]) -> Any:
    """Space variables and t real, every other name a positive constant (hbar, m, a, L, omega, ...)."""
    sub = {
        s: sp.Symbol(s.name, real=True)
        if s.name in names or s.name == "t"
        else sp.Symbol(s.name, positive=True)
        for s in e.free_symbols
    }
    return e.xreplace(sub)


def _wf_const(name: str) -> Any:
    return sp.Symbol(name, positive=True)


def _wf_fixed(e: Any, xs: list[Any]) -> dict[Any, Any]:
    """Values of the constants for the numerical checks: 7/5, 8/5, ... in name order."""
    others = sorted(e.free_symbols - set(xs), key=lambda s: s.name)
    return {s: sp.Rational(7 + i, 5) for i, s in enumerate(others)}


def _wf_check_local(psi: Any, rho: Any, cur: list[Any], xs: list[Any]) -> bool:
    """rho = |Psi|^2 and j = (hbar/m) Im(Psi* dPsi/dx) at 3 points, with mpmath derivatives (as D90)."""
    import mpmath

    hbar, mass = _wf_const("hbar"), _wf_const("m")
    fixed = _wf_fixed(psi + hbar + mass, xs)
    p0 = psi.subs(fixed)
    r0 = rho.subs(fixed)
    c0 = [c.subs(fixed) for c in cur]
    good = 0
    with mpmath.workdps(40):
        h_m = _mp_value(fixed[hbar] / fixed[mass], {}).real
        for pn, pd in ((3, 7), (6, 5), (-4, 9)):
            pts = {x: mpmath.mpf(pn + j) / pd for j, x in enumerate(xs)}
            try:
                val = _mp_value(p0, pts)
                if abs(_mp_value(r0, pts) - abs(val) ** 2) > mpmath.mpf("1e-25") * (
                    1 + abs(val) ** 2
                ):
                    return False
                for i, x in enumerate(xs):
                    rest = {y: v for y, v in pts.items() if y is not x}

                    def f(v: Any, x: Any = x, rest: dict[Any, Any] = rest) -> Any:
                        return _mp_value(p0, {**rest, x: v})

                    want = h_m * mpmath.im(mpmath.conj(val) * mpmath.diff(f, pts[x]))
                    if abs(_mp_value(c0[i], pts) - want) > mpmath.mpf("1e-25") * (1 + abs(want)):
                        return False
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            good += 1
    return good > 0


_WF_GRID_N = (
    48  # Gauss-Legendre points per dimension of the probability check (48^3 = 110592 points)
)
_WF_GRID_MAX_DIM = 3


def _wf_grid(lo: float, hi: float) -> tuple[Any, Any]:
    """Nodes and weights on [lo, hi]; infinite ends mapped by x = u / (1 - u^2) or x = a + v / (1 - v)."""
    u, w = np.polynomial.legendre.leggauss(_WF_GRID_N)
    inf = float("inf")
    if lo == -inf and hi == inf:
        return u / (1 - u * u), w * (1 + u * u) / (1 - u * u) ** 2
    if hi == inf or lo == -inf:
        v = (u + 1) / 2  # 0 < v < 1
        dx = w / 2 / (1 - v) ** 2
        return (lo + v / (1 - v), dx) if hi == inf else (hi - v / (1 - v), dx)
    return (hi - lo) / 2 * u + (hi + lo) / 2, w * (hi - lo) / 2


def _wf_box_integral(fl: Callable[..., float], lims: list[tuple[float, float]]) -> float:
    """Tensor Gauss-Legendre sum of fl over a box (D99: nquad needed 2.9 million calls in 3D)."""
    grids = [_wf_grid(lo, hi) for lo, hi in lims]
    total = 0.0
    for combo in itertools.product(*(range(_WF_GRID_N) for _ in grids)):
        weight = 1.0
        point = []
        for (nodes, weights), i in zip(grids, combo, strict=True):
            weight *= weights[i]
            point.append(nodes[i])
        value = fl(*point)
        if math.isfinite(value):
            total += weight * value
    return total


def _mp_bound(b: Any) -> Any:
    import mpmath

    if b == sp.oo:
        return mpmath.inf
    if b == -sp.oo:
        return -mpmath.inf
    return mpmath.mpf(str(sp.N(b, 30)))


def _wf_probability(
    rho: Any, xs: list[Any], rows: Any, names: list[str], ctx: Ctx
) -> tuple[str | None, bool]:
    """Integral of rho over the region of `bounds`, checked with mpmath.quad."""
    import mpmath

    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(r, list) and len(r) == 3 for r in rows)
    ):
        raise ToolError("INVALID_ARGUMENT", "'bounds' is [[x, a, b], ...].", '[["x", "-oo", "oo"]]')
    by_name = {x.name: x for x in xs}
    g = grammar(names)
    region = []
    for name, a, b in rows:
        if name not in by_name:
            raise ToolError(
                "INVALID_ARGUMENT", f"bounds: '{name}' is not one of the space variables."
            )
        lo, hi = (_wf_rename(value_of(v, g), names) for v in (a, b))
        region.append((by_name[name], lo, hi))
    total = sp.integrate(rho, *region, conds="none")
    result = None
    if total.has(sp.Integral):
        ctx.warn("No closed form for the probability: only the numerical value is checked.")
    else:
        total = sp.simplify(total)
        result = str(total)
    # the symbols of the bounds one by one: -oo + oo would be nan and lose them
    bound_syms = {s for _, lo, hi in region for s in lo.free_symbols | hi.free_symbols}
    fixed = _wf_fixed(rho + sp.Add(*bound_syms), [r[0] for r in region])
    f0 = rho.subs(fixed)
    variables = [r[0] for r in region]
    lims = [(lo.subs(fixed), hi.subs(fixed)) for _, lo, hi in region]
    if any(b.free_symbols for lim in lims for b in lim):
        raise ToolError(
            "INVALID_ARGUMENT", "The bounds of a wavefunction are constants, not space variables."
        )
    try:
        if len(region) == 1:  # mpmath at 20 digits
            with mpmath.workdps(20):
                fn = compile_mp(f0, variables, 20)
                quad = cast(Any, mpmath.quad)
                numeric = complex(quad(fn, [_mp_bound(lims[0][0]), _mp_bound(lims[0][1])]))
        elif len(region) <= _WF_GRID_MAX_DIM:  # fixed Gauss-Legendre grid on the float function
            fl = compile_expr(f0, variables)
            box = [(float(sp.N(lo)), float(sp.N(hi))) for lo, hi in lims]
            numeric = complex(_wf_box_integral(fl, box))
        else:
            ctx.warn(
                f"The probability is checked numerically up to {_WF_GRID_MAX_DIM} dimensions: "
                "not verified here."
            )
            return result, False
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return result, False
    if result is None:
        return None, True
    exact = complex(sp.N(total.subs(fixed), 20))
    tol = 1e-10 if len(region) == 1 else 1e-7
    return result, abs(exact - numeric) <= tol * (1 + abs(exact))


def _wf_momentum(psi: Any, xs: list[Any], ctx: Ctx) -> tuple[str | None, list[str], bool]:
    """Phi(p) = (2 pi hbar)^(-d/2) integral exp(-i p.r / hbar) Psi d^d r; 1D checked by mpmath.quad."""
    import mpmath

    hbar = _wf_const("hbar")
    ps = [sp.Symbol(f"p_{x.name}", real=True) for x in xs]
    names = [p.name for p in ps]
    kernel = sp.exp(-sp.I * sp.Add(*(p * x for p, x in zip(ps, xs, strict=True))) / hbar)
    phi = sp.integrate(kernel * psi, *[(x, -sp.oo, sp.oo) for x in xs], conds="none")
    if phi.has(sp.Integral):
        ctx.warn("No closed form for the momentum-space wavefunction.")
        return None, names, False
    phi = sp.simplify(phi / (2 * sp.pi * hbar) ** sp.Rational(len(xs), 2))
    if len(xs) > 1:
        ctx.warn("The momentum-space wavefunction is checked numerically in 1D only.")
        return str(phi), names, True
    fixed = _wf_fixed(psi + hbar, xs) | {ps[0]: sp.Rational(9, 5)}
    f0 = (kernel * psi).subs(fixed)
    with mpmath.workdps(30):
        try:
            fn = compile_mp(f0, [xs[0]], 30)
            numeric = mpmath.quad(lambda v: fn(v), [-mpmath.inf, mpmath.inf])
            numeric /= mpmath.sqrt(2 * mpmath.pi * _mp_value(fixed[hbar], {}).real)
            exact = _mp_value(phi.subs(fixed), {})
        except (TypeError, ValueError, ZeroDivisionError):
            return str(phi), names, False
        ok = abs(exact - numeric) <= mpmath.mpf("1e-15") * (1 + abs(exact))
    return str(phi), names, bool(ok)


@op("wavefunction")
def _wavefunction(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    names = _need(args, "variables", '["x"]')
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ToolError("INVALID_ARGUMENT", "'variables' is the list of space variables.", '["x"]')
    if len(names) > _WF_MAX_VARS:
        raise ToolError("LIMIT_EXCEEDED", f"At most {_WF_MAX_VARS} space variables.")
    if "t" in names:
        raise ToolError(
            "INVALID_ARGUMENT", "t is the time: list only the space variables.", '["x"]'
        )
    g = grammar(names)
    psi = _wf_rename(expr_of(_need(args, "expr", _WF_EXAMPLE), g), names)
    xs = [sp.Symbol(n, real=True) for n in names]
    t = sp.Symbol("t", real=True)
    hbar, mass = _wf_const("hbar"), _wf_const("m")
    rho = sp.simplify(sp.conjugate(psi) * psi)
    cur = [
        sp.simplify(hbar / mass * sp.im(sp.expand(sp.conjugate(psi) * sp.diff(psi, x)))) for x in xs
    ]
    out: dict[str, Any] = {"density": str(rho), "current": [str(c) for c in cur]}
    checks = [_wf_check_local(psi, rho, cur, xs)]
    if args.get("bounds") is not None:
        out["probability"], ok = _wf_probability(rho, xs, args["bounds"], names, ctx)
        checks.append(ok)
    if args.get("momentum_space"):
        out["momentum_wavefunction"], out["momentum_variables"], ok = _wf_momentum(psi, xs, ctx)
        checks.append(ok)
    if args.get("potential_energy") is not None:
        pot = _wf_rename(expr_of(args["potential_energy"], g), names)
        h_psi = -(hbar**2) / (2 * mass) * sp.Add(*(sp.diff(psi, x, 2) for x in xs)) + pot * psi
        if psi.has(t):
            res = sp.simplify(sp.I * hbar * sp.diff(psi, t) - h_psi)
            out["schrodinger"] = {"residual": str(res), "satisfied": is_zero(res)}
        else:
            e = sp.simplify(h_psi / psi)
            out["schrodinger"] = {"energy": str(e), "eigenstate": not (e.free_symbols & set(xs))}
    out["verified"] = all(checks)
    ctx.formatted = f"|Psi|^2 = {out['density']}"
    return out


# ------------------------------------------------------------------ Lagrangian mechanics (D102)

_LAG_MAX = 6
_LAG_EXAMPLE = "m*x'**2/2 - k*x**2/2"


_LAG_SIMPLIFY_OPS = 100


def _lag_tidy(e: Any) -> Any:
    """simplify up to 100 operations, cancel and factor_terms above: simplify of the double pendulum H
    (315 operations) took 2.44 s and the whole call 12.08 s, over the 5 s limit."""
    if sp.count_ops(e) <= _LAG_SIMPLIFY_OPS:
        return sp.simplify(e)
    return sp.factor_terms(sp.cancel(e))


def _lag_read(text: Any, coords: list[str], g: Grammar) -> Any:
    """L, T or V with x' for the velocity of the coordinate x (the notation of math_solve.ode)."""
    if not isinstance(text, str):
        raise ToolError("INVALID_ARGUMENT", "The Lagrangian is a string.", _LAG_EXAMPLE)
    return expr_of(_ode_text(text, coords, "t"), g)


def _lag_check_form(lag: Any, funcs: dict[str, Any], t: Any) -> None:
    """L may depend on q, q' and t only: no q'' and no q at another time."""
    for d in lag.atoms(sp.Derivative):
        if d.derivative_count > 1:
            name = getattr(d.expr.func, "__name__", str(d.expr))
            raise ToolError(
                "INVALID_ARGUMENT",
                f"L may depend on the coordinates, their first derivatives and t only ({name}'' found).",
                _LAG_EXAMPLE,
            )
    for f in lag.atoms(sp.core.function.AppliedUndef):
        if f.func.__name__ in funcs and f.args != (t,):
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Write the coordinate {f.func.__name__} without an argument: it is a function of t.",
                _LAG_EXAMPLE,
            )


def _lag_symbols(
    lag: Any, coords: list[str], funcs: dict[str, Any], t: Any, positive: bool
) -> tuple[Any, list[Any], list[Any], list[Any], list[Any], dict[str, Any]]:
    """L with plain symbols: q, q' (named x'), q'' (x''), momenta p_x; constants real or positive."""
    qs = [sp.Symbol(c, real=True) for c in coords]
    vs = [sp.Symbol(f"{c}'", real=True) for c in coords]
    accs = [sp.Symbol(f"{c}''", real=True) for c in coords]
    ps = [sp.Symbol(f"p_{c}", real=True) for c in coords]
    qt = [funcs[c](t) for c in coords]
    ls = lag.subs({sp.Derivative(q, t): v for q, v in zip(qt, vs, strict=True)})
    ls = ls.subs(dict(zip(qt, qs, strict=True)))
    taken = {s.name for s in ps} | {s.name for s in accs}
    consts: dict[str, Any] = {}
    for s in sorted(ls.free_symbols - set(qs) - set(vs) - {t}, key=lambda s: s.name):
        if s.name in taken:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"The name {s.name} is a generalized momentum of this operation: rename the constant.",
            )
        consts[s.name] = (
            sp.Symbol(s.name, positive=True) if positive else sp.Symbol(s.name, real=True)
        )
    ls = ls.xreplace({s: consts[s.name] for s in ls.free_symbols if s.name in consts})
    return ls, qs, vs, accs, ps, consts


def _lag_dt(f: Any, qs: list[Any], vs: list[Any], accs: list[Any], t: Any) -> Any:
    """Total time derivative of f(q, q', t) along a motion: sum df/dq q' + df/dq' q'' + df/dt."""
    return (
        sp.Add(*(sp.diff(f, q) * v for q, v in zip(qs, vs, strict=True)))
        + sp.Add(*(sp.diff(f, v) * a for v, a in zip(vs, accs, strict=True)))
        + sp.diff(f, t)
    )


def _lag_euler_check(
    lag: Any, el: list[Any], consts: dict[str, Any], qt: list[Any], syms: tuple[Any, ...], t: Any
) -> bool:
    """el against sympy.calculus.euler.euler_equations, an independent route (dL/dq - d/dt dL/dq' = 0)."""
    from sympy.calculus.euler import euler_equations

    qs, vs, accs = syms
    lag = lag.xreplace({s: consts[s.name] for s in lag.free_symbols if s.name in consts})
    try:
        ref = euler_equations(lag, qt, t)
    except (TypeError, ValueError, NotImplementedError):
        return False
    rep2 = {sp.Derivative(q, (t, 2)): a for q, a in zip(qt, accs, strict=True)}
    rep1 = {sp.Derivative(q, t): v for q, v in zip(qt, vs, strict=True)}
    rep0 = dict(zip(qt, qs, strict=True))
    for e, r in zip(el, ref, strict=True):
        rr = (r.lhs - r.rhs).subs(rep2).subs(rep1).subs(rep0)
        if not is_zero(e + rr):
            return False
    return True


def _lag_zero(e: Any, ls: Any, moving: set[Any]) -> bool:
    """e == 0 where L is real: SymPy first, else 3 points at which L is real (|q'| < c in a relativistic L).

    is_zero would also try points outside that region, where sqrt(1/z) and 1/sqrt(z) are on different branches.
    """
    try:
        if _lag_tidy(e) == 0:
            return True
    except (TypeError, ValueError, NotImplementedError, AttributeError):
        pass
    syms = sorted(e.free_symbols | ls.free_symbols, key=lambda s: s.name)
    good = 0
    for pn, pd in ((3, 7), (6, 5), (-4, 9)):
        subs: dict[Any, Any] = {}
        j = i = 0
        for s in syms:
            if s in moving:
                subs[s] = sp.Rational(pn, pd) + sp.Rational(j, 11)
                j += 1
            else:
                subs[s] = sp.Rational(7 + i, 5)
                i += 1
        try:
            lv = complex(sp.N(ls.subs(subs), 30))
            if not (math.isfinite(lv.real) and abs(lv.imag) <= 1e-25 * (1 + abs(lv.real))):
                continue
            terms = [complex(sp.N(a.subs(subs), 30)) for a in sp.Add.make_args(e)]
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not abs(sum(terms)) <= 1e-20 * (1 + sum(abs(z) for z in terms)):
            return False
        good += 1
    return good > 0


def _lag_hamiltonian(
    ls: Any, h: Any, mom: list[Any], syms: tuple[Any, ...], coords: list[str], ctx: Ctx
) -> tuple[Any, dict[str, str] | None, bool]:
    """H(q, p) by the Legendre transform; None for a singular L or several velocities for one momentum."""
    qs, vs, ps = syms
    hess = sp.Matrix([[sp.diff(ls, a, b) for b in vs] for a in vs])
    if is_zero(hess.det()):
        ctx.warn(
            "The Lagrangian is singular: the matrix of second derivatives of L in the velocities has determinant 0, "
            "so the momenta do not fix the velocities and there is no Hamiltonian H(q, p)."
        )
        return None, None, True
    try:
        sols = sp.solve([sp.Eq(p, m) for p, m in zip(ps, mom, strict=True)], vs, dict=True)
    except (NotImplementedError, ValueError):
        sols = []
    sols = [s for s in sols if all(v in s for v in vs)]
    if len(sols) != 1:
        ctx.warn(
            f"The velocities are not one function of the momenta ({len(sols)} solutions of p = dL/dq'): "
            "no Hamiltonian; energy_function is h(q, q') = sum q' p - L."
        )
        return None, None, True
    ham = _lag_tidy(h.subs(sols[0]))
    at_p = dict(zip(ps, mom, strict=True))
    moving = set(qs) | set(vs) | set(ps) | {s for s in ls.free_symbols if s.name == "t"}
    ok = _lag_zero(ham.subs(at_p) - h, ls, moving) and all(
        _lag_zero(sp.diff(ham, p).subs(at_p) - v, ls, moving) for p, v in zip(ps, vs, strict=True)
    )
    eqs: dict[str, str] = {}
    for c, q, p in zip(coords, qs, ps, strict=True):
        eqs[f"{c}'"] = str(_lag_tidy(sp.diff(ham, p)))
        eqs[f"p_{c}'"] = str(_lag_tidy(-sp.diff(ham, q)))
    return ham, eqs, ok


def _lag_fixed(e: Any, t: Any) -> dict[Any, Any]:
    """Values of the constants for the numerical check of the action: 7/5, 8/5, ... in name order."""
    others = sorted(e.free_symbols - {t}, key=lambda s: s.name)
    return {s: sp.Rational(7 + i, 5) for i, s in enumerate(others)}


def _lag_action(
    args: dict[str, Any],
    ls: Any,
    el: list[Any],
    coords: list[str],
    syms: tuple[Any, ...],
    consts: dict[str, Any],
    t: Any,
    positive: bool,
    ctx: Ctx,
) -> tuple[dict[str, Any], bool]:
    """S = integral of L dt along the path q(t), and the Euler-Lagrange residuals on that path."""
    import mpmath

    qs, vs, accs = syms
    path, bounds = args.get("path"), args.get("bounds")
    if not isinstance(path, dict) or not isinstance(bounds, list) or len(bounds) != 2:
        raise ToolError(
            "INVALID_ARGUMENT",
            "The action needs 'path' ({coordinate: expression of t}) and 'bounds' [t1, t2].",
            '{"path": {"x": "A*cos(w*t)"}, "bounds": [0, 2]}',
        )
    missing = [c for c in coords if c not in path]
    extra = sorted(set(path) - set(coords))
    if missing or extra:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"'path' needs an expression of t for every coordinate and nothing else; missing: "
            f"{', '.join(missing) or 'none'}; not a coordinate: {', '.join(extra) or 'none'}.",
        )
    gp = grammar(None, extra={"t": t, **consts})
    ends = [value_of(b, gp) for b in bounds]
    sub: dict[Any, Any] = {}
    for c, q, v, a in zip(coords, qs, vs, accs, strict=True):
        e = expr_of(path[c], gp)
        if any(s.name in coords for s in e.free_symbols):
            raise ToolError(
                "INVALID_ARGUMENT", "A path uses t and constants only, not the coordinates."
            )
        sub |= {q: e, v: sp.diff(e, t), a: sp.diff(e, t, 2)}

    def assume(e: Any) -> Any:
        new = {
            s: sp.Symbol(s.name, positive=True) if positive else sp.Symbol(s.name, real=True)
            for s in e.free_symbols
            if s != t and s.name not in consts
        }
        return e.xreplace(new)

    sub = {k: assume(v) for k, v in sub.items()}
    ends = [assume(b) for b in ends]
    residuals = [sp.simplify(e.subs(sub)) for e in el]
    lp = sp.simplify(ls.subs(sub))
    s_val = sp.integrate(lp, (t, ends[0], ends[1]), conds="none")  # generic case, checked below
    out: dict[str, Any] = {
        "stationary": all(is_zero(r) for r in residuals),
        "residuals": [str(r) for r in residuals],
    }
    if s_val.has(sp.Integral):
        ctx.warn("The action has no closed form here: value is null.")
        out["value"] = None
        return out, True
    s_val = sp.simplify(s_val)
    out["value"] = str(s_val)
    fixed = _lag_fixed(lp + ends[0] + ends[1], t)
    with mpmath.workdps(30):
        try:
            f = compile_mp(lp.subs(fixed), [t], 30)
            lo, hi = (mpmath.mpf(str(sp.N(b.subs(fixed), 40))) for b in ends)
            quad: Any = mpmath.quad  # the mpmath stubs type quad as a tuple
            ref = quad(f, [lo, hi])
            got = mpmath.mpf(str(sp.N(s_val.subs(fixed), 40)))
            ok = bool(abs(got - ref) <= mpmath.mpf("1e-20") * (1 + abs(ref)))
        except (ToolError, TypeError, ValueError, ZeroDivisionError):
            ok = False
    return out, ok


@op("lagrangian")
def _lagrangian(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    names = args.get("variables")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ToolError(
            "INVALID_ARGUMENT", "'variables' is the list of generalized coordinates.", '["x"]'
        )
    if len(names) > _LAG_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"At most {_LAG_MAX} generalized coordinates.")
    if "t" in names:
        raise ToolError(
            "INVALID_ARGUMENT", "t is the time: list only the generalized coordinates.", '["x"]'
        )
    coords = [check_identifier(n) for n in names]
    if len(set(coords)) != len(coords):
        raise ToolError("INVALID_ARGUMENT", "Each generalized coordinate once.")
    text_l, text_t, text_v = (
        args.get("expr"),
        args.get("kinetic_energy"),
        args.get("potential_energy"),
    )
    if (text_l is None) == (text_t is None) or (text_l is not None and text_v is not None):
        raise ToolError(
            "INVALID_ARGUMENT",
            "Give the Lagrangian L in 'expr', or the kinetic energy T in 'kinetic_energy' and the potential "
            "energy V in 'potential_energy' (L = T - V): one of the two.",
            '{"expr": "' + _LAG_EXAMPLE + '", "variables": ["x"]}',
        )
    domain = args.get("domain", "real")
    if domain not in ("real", "positive"):
        raise ToolError(
            "INVALID_ARGUMENT", "'domain' of the constants for lagrangian: real or positive."
        )
    t = sp.Symbol("t", real=True)
    funcs = {c: sp.Function(c) for c in coords}
    g = Grammar(SOLVE_FUNCS, {"t": t, **funcs})
    if text_l is not None:
        lag = _lag_read(text_l, coords, g)
    else:
        lag = _lag_read(text_t, coords, g) - (_lag_read(text_v, coords, g) if text_v else 0)
    _lag_check_form(lag, funcs, t)
    ls, qs, vs, accs, ps, consts = _lag_symbols(lag, coords, funcs, t, domain == "positive")
    qt = [funcs[c](t) for c in coords]

    mom = [_lag_tidy(sp.diff(ls, v)) for v in vs]
    forces = [sp.diff(ls, q) for q in qs]
    el = [_lag_tidy(_lag_dt(p, qs, vs, accs, t) - f) for p, f in zip(mom, forces, strict=True)]
    checks = [_lag_euler_check(lag, el, consts, qt, (qs, vs, accs), t)]
    checks += [_check_derivative_numeric(ls, m, [(v, 1)]) for m, v in zip(mom, vs, strict=True)]
    checks += [_check_derivative_numeric(ls, f, [(q, 1)]) for f, q in zip(forces, qs, strict=True)]
    h = _lag_tidy(sp.Add(*(v * m for v, m in zip(vs, mom, strict=True))) - ls)
    ham, ham_eqs, ok = _lag_hamiltonian(ls, h, mom, (qs, vs, ps), coords, ctx)
    checks.append(ok)
    out: dict[str, Any] = {
        "coordinates": coords,
        "lagrangian": str(ls),
        "momenta": {f"p_{c}": str(m) for c, m in zip(coords, mom, strict=True)},
        "equations": [f"{e} = 0" for e in el],
        "energy_function": str(h),
        "hamiltonian": None if ham is None else str(ham),
        "hamilton_equations": ham_eqs,
        "cyclic_coordinates": [c for c, f in zip(coords, forces, strict=True) if is_zero(f)],
        "energy_conserved": is_zero(sp.diff(ls, t)),
    }
    if args.get("path") is not None or args.get("bounds") is not None:
        out["action"], ok = _lag_action(
            args, ls, el, coords, (qs, vs, accs), consts, t, domain == "positive", ctx
        )
        checks.append(ok)
    out["verified"] = all(checks)
    ctx.step(f"Lagrangian L = {ls} (x' is dx/dt).")
    ctx.step(f"Generalized momenta p = dL/dq': {out['momenta']}.")
    ctx.step(f"Euler-Lagrange d/dt(dL/dq') - dL/dq = 0: {out['equations']}.")
    ctx.step(f"Hamiltonian H = sum p q' - L: {out['hamiltonian']}.")
    ctx.formatted = "; ".join(out["equations"])
    return out


# ------------------------------------------------------------------ central force, Binet equation (D102)


def _cf_symbols() -> dict[str, Any]:
    return {
        "theta": sp.Symbol("theta", real=True),
        "r": sp.Symbol("r", positive=True),
        "mu": sp.Symbol("mu", positive=True),
        "l": sp.Symbol("l", positive=True),
    }


def _cf_positive(e: Any, sy: dict[str, Any]) -> Any:
    """Every name other than theta, r, mu and l is a positive constant."""
    keep = set(sy.values())
    if any(s.name == "u" for s in e.free_symbols):
        raise ToolError("INVALID_ARGUMENT", "u is 1/r in this operation: rename the constant u.")
    return e.xreplace(
        {s: sp.Symbol(s.name, positive=True) for s in e.free_symbols if s not in keep}
    )


def _cf_newton_force(orbit: Any, sy: dict[str, Any]) -> Any:
    """F from Newton's radial equation mu (r'' - r theta'^2) with theta' = l / (mu r^2): independent of Binet."""
    th, mu, ell = sy["theta"], sy["mu"], sy["l"]
    thd = ell / (mu * orbit**2)
    rd = sp.diff(orbit, th) * thd
    rdd = sp.diff(rd, th) * thd
    return mu * (rdd - orbit * thd**2)


def _cf_from_orbit(text: Any, sy: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    th, r, mu, ell = sy["theta"], sy["r"], sy["mu"], sy["l"]
    orbit = _cf_positive(expr_of(text, grammar(None, extra=sy)), sy)
    if orbit.has(r):
        raise ToolError(
            "INVALID_ARGUMENT",
            "The orbit is r as a function of theta: it must not contain r.",
            "p/(1 + e*cos(theta))",
        )
    u = sp.simplify(1 / orbit)
    f_th = sp.simplify(-(ell**2) * u**2 * (sp.diff(u, th, 2) + u) / mu)
    ok = is_zero(f_th - _cf_newton_force(orbit, sy))
    force = None
    if not orbit.has(th):
        ctx.warn(
            "The orbit is circular: any attractive central force of this size at that radius holds it, so the orbit "
            "does not fix F as a function of r; force_theta is the force on the circle."
        )
    else:
        try:
            sols = sp.solve(sp.Eq(r, orbit), th)
        except (NotImplementedError, ValueError):
            sols = []
        cands = []
        for s in sols:
            c = sp.simplify(f_th.subs(th, s))
            if not c.has(th) and c not in cands and is_zero(c.subs(r, orbit) - f_th):
                cands.append(c)
        if len(cands) == 1:
            force = cands[0]
        else:
            ctx.warn(
                "F could not be written as one function of r: force is null, force_theta is F(theta)."
            )
    return {
        "orbit": str(orbit),
        "u": str(u),
        "binet_equation": "u'' + u = -mu*F/(l**2*u**2)",
        "force_theta": str(f_th),
        "force": None if force is None else str(force),
        "verified": ok,
    }


def _cf_from_force(args: dict[str, Any], sy: dict[str, Any]) -> dict[str, Any]:
    th, r, mu, ell = sy["theta"], sy["r"], sy["mu"], sy["l"]
    force = _cf_positive(expr_of(args["force"], grammar(None, extra=sy)), sy)
    if force.has(th):
        raise ToolError(
            "INVALID_ARGUMENT", "A central force depends on r only, not on theta.", "-k/r**2"
        )
    w = sp.Symbol("u", positive=True)
    rhs = sp.simplify(-mu * force.subs(r, 1 / w) / (ell**2 * w**2))
    binet = f"u'' + u = {rhs}"
    if not (rhs.is_polynomial(w) and sp.Poly(rhs, w).degree() <= 1):
        raise ToolError(
            "MATH_ERROR",
            f"The Binet equation {binet} is not linear in u, so no closed form is tried. Solve it with "
            'math_solve.ode (expr that equation, variables ["u", "theta"], ics and bounds for the '
            "numerical solution).",
        )
    uf = sp.Function("u")
    big_u = uf(th)
    eq = sp.Eq(sp.diff(big_u, th, 2) + big_u, rhs.subs(w, big_u))
    g = grammar(None, extra={"theta": th, "mu": mu, "l": ell})
    ics = {k: _cf_positive(v, sy) for k, v in _ics(args.get("ics"), {"u": uf}, th, g).items()}
    sol = sp.dsolve(eq, big_u, ics=ics or None)
    if isinstance(sol, list):
        raise ToolError(
            "MATH_ERROR", "The Binet equation has several solutions here: use math_solve.ode."
        )
    u_sol: Any = cast(Any, sol).rhs
    orbit = sp.simplify(1 / u_sol)
    ok = sp.checkodesol(eq, sol)[0] is True and is_zero(
        _cf_newton_force(orbit, sy) - force.subs(r, orbit)
    )
    return {
        "force": str(force),
        "binet_equation": binet,
        "u": str(u_sol),
        "orbit": str(orbit),
        "verified": ok,
    }


@op("central_force")
def _central_force(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    has_orbit, has_force = args.get("orbit") is not None, args.get("force") is not None
    if has_orbit == has_force:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Give one of 'orbit' (r as a function of theta) or 'force' (F as a function of r, negative = "
            "attractive); mu is the (reduced) mass and l the angular momentum.",
            '{"operation": "central_force", "orbit": "p/(1 + e*cos(theta))"}',
        )
    sy = _cf_symbols()
    if has_orbit:
        if args.get("ics") is not None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "'ics' is for 'force' (the constants of u(theta)), not for 'orbit'.",
            )
        out = _cf_from_orbit(args["orbit"], sy, ctx)
        ctx.formatted = f"F(r) = {out['force']}; F(theta) = {out['force_theta']}"
    else:
        out = _cf_from_force(args, sy)
        ctx.formatted = f"{out['binet_equation']}; r(theta) = {out['orbit']}"
    ctx.step("Binet equation: u'' + u = -mu F / (l^2 u^2), u = 1/r, ' = d/dtheta.")
    ctx.step(
        "Back-check: Newton's radial equation mu (r'' - r theta'^2) = F with theta' = l / (mu r^2)."
    )
    return out


# ------------------------------------------------------------------ born_scattering (D105)

_BORN_EXAMPLE = '{"operation": "born_scattering", "potential_energy": "-V0*exp(-r/a)/r"}'
_BORN_RESERVED = ("q", "k", "theta", "mu", "hbar")
_BORN_DPS = 20  # digits of the numerical integral (mpmath.quad and quadosc)
_BORN_CHECK_Q = ((3, 2), (7, 3))  # values of q for the numerical check of a closed form
_BORN_CHECK_REL = 1e-12
_BORN_SIMPLIFY_OPS = 60  # simplify a small closed form only; with a Piecewise simplify took 18.7 s
_BORN_MAX_BREAKS = 50
_BORN_BAD = (sp.zoo, sp.nan, sp.oo, -sp.oo)


def _born_symbols() -> dict[str, Any]:
    return {n: sp.Symbol(n, positive=True) for n in ("r", *_BORN_RESERVED)}


def _born_potential(args: dict[str, Any], r: Any) -> Any:
    """V(r) with r and every constant positive; the names of the result are not allowed in V."""
    text = _need(args, "potential_energy", _BORN_EXAMPLE)
    v = expr_of(text, grammar(None))
    for s in v.free_symbols:
        if s.name in _BORN_RESERVED:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"'{s.name}' is a name of the result (f(q), q = 2 k sin(theta/2), mu the reduced mass, "
                "hbar): rename that constant of V.",
                _BORN_EXAMPLE,
            )
    v = v.xreplace({s: sp.Symbol(s.name, positive=True) for s in v.free_symbols})
    if r not in v.free_symbols:
        raise ToolError(
            "INVALID_ARGUMENT", "V must depend on r (the distance from the centre).", _BORN_EXAMPLE
        )
    return v


def _born_limit(e: Any, r: Any, point: Any) -> Any:
    """Limit of e for r -> point (0 from above, or oo); a Heaviside step is replaced by its value there
    first. None when SymPy cannot say."""

    def step(*a: Any) -> Any:
        side = sp.limit(a[0], r, point, "+")
        if side == sp.oo or side.is_positive:
            return sp.Integer(1)
        if side == -sp.oo or side.is_negative:
            return sp.Integer(0)
        return sp.Heaviside(*a)

    try:
        lim = sp.limit(e.replace(sp.Heaviside, step), r, point, "+")
    except (NotImplementedError, ValueError, TypeError, AttributeError):
        return None
    return None if lim.has(sp.Limit) else lim


def _born_split(v: Any, r: Any) -> tuple[Any, Any]:
    """(L, V - L/r): a Coulomb tail L / r goes apart; MATH_ERROR when the integral cannot converge."""
    near = _born_limit(r**3 * v, r, sp.Integer(0))
    if near is not None and near != 0:
        raise ToolError(
            "MATH_ERROR",
            "The Born integral does not converge at r -> 0: r^2 V(r) must be integrable there "
            f"(V weaker than 1/r^3; here r^3 V(r) -> {near}).",
        )
    far = _born_limit(r * v, r, sp.oo)
    if far is None or far == 0:
        return sp.Integer(0), v
    if far.has(*_BORN_BAD, sp.AccumBounds, r):
        raise ToolError(
            "MATH_ERROR",
            "The Born integral does not converge: V(r) must fall at least as 1/r for r -> oo "
            f"(here r V(r) -> {far}).",
        )
    return far, sp.expand(v - far / r)


def _born_integral(rest: Any, r: Any, q: Any) -> Any:
    """integral_0^oo r sin(q r) rest(r) dr for a short-range rest; None when SymPy finds no closed form.

    Only the Meijer G method (meijerg=True): on 15 potentials it gave the same closed forms as the default
    strategy in the same time, and it gives up on Woods-Saxon in 0.10 s instead of more than 12 s (26/11/2026).
    """
    if rest == 0:
        return sp.Integer(0)
    val = sp.integrate(r * sp.sin(q * r) * rest, (r, 0, sp.oo), conds="none", meijerg=True)
    if val.has(sp.Integral):
        return None
    if val.has(*_BORN_BAD, sp.AccumBounds):
        raise ToolError(
            "MATH_ERROR", f"The Born integral of this V(r) does not converge (SymPy: {val})."
        )
    return sp.unpolarify(val)


def _born_tidy(e: Any) -> Any:
    if e.has(sp.Piecewise) or sp.count_ops(e) > _BORN_SIMPLIFY_OPS:
        return e
    return sp.simplify(e)


def _born_breaks(e: Any, r: Any) -> list[Any]:
    """Positive r (numbers) where a Heaviside or a Piecewise condition of e switches, sorted."""
    args = [h.args[0] for h in e.atoms(sp.Heaviside)]
    for pw in e.atoms(sp.Piecewise):
        for _, cond in pw.args:
            rels = (
                cond.atoms(sp.core.relational.Relational) if isinstance(cond, sp.Basic) else set()
            )
            args.extend(lhs - rhs for lhs, rhs, _ in map(rel_parts, rels))
    found: set[Any] = set()
    for a in args:
        if not a.has(r) or a.free_symbols - {r}:
            continue
        try:
            poly = sp.Poly(a, r)
        except sp.PolynomialError:
            continue
        for root in poly.nroots(n=_BORN_DPS):
            re_, im_ = root.as_real_imag()
            if im_ == 0 and re_ > 0:
                found.add(sp.Float(re_, _BORN_DPS))
    return sorted(found)[:_BORN_MAX_BREAKS]


def _born_piece(e: Any, r: Any, at_r: Any) -> Any:
    """e on one interval between breaks: each Heaviside is its value and each Piecewise its branch at r = at_r,
    so the integrand is smooth there (a Heaviside evaluated point by point took 18.9 s)."""
    swap: dict[Any, Any] = {}
    for h in e.atoms(sp.Heaviside):
        val = h.args[0].subs(r, at_r)
        if val.is_positive or val.is_negative:
            swap[h] = sp.Integer(1 if val.is_positive else 0)
    for pw in e.atoms(sp.Piecewise):
        for pair in pw.args:
            cond = pair.cond.subs(r, at_r) if isinstance(pair.cond, sp.Basic) else pair.cond
            if cond is sp.true:
                swap[pw] = pair.expr
                break
    return e.xreplace(swap)


def _born_numeric(rest: Any, r: Any, qv: Any) -> Any:
    """integral_0^oo r sin(q r) rest(r) dr with numbers only: mpmath.quad on each interval between the breaks of
    a step, then mpmath.quadosc (period 2 pi / q) from the last break."""
    import mpmath

    f = r * sp.sin(qv * r) * rest
    pts = [sp.Integer(0), *_born_breaks(rest, r)]
    with mpmath.workdps(_BORN_DPS):
        total = mpmath.mpf(0)
        for a, b in itertools.pairwise(pts):
            piece = _born_piece(f, r, (a + b) / 2)
            if piece != 0:
                total += mpmath.quad(
                    compile_mp(piece, [r], _BORN_DPS), [to_mp(a, _BORN_DPS), to_mp(b, _BORN_DPS)]
                )
        last = _born_piece(f, r, pts[-1] + 1)
        if last != 0:
            fn = compile_mp(last, [r], _BORN_DPS)
            total += mpmath.quadosc(
                fn, [to_mp(pts[-1], _BORN_DPS), mpmath.inf], omega=to_mp(qv, _BORN_DPS)
            )
        return total


def _born_verified(rest: Any, i_rest: Any, r: Any, q: Any) -> bool:
    """The closed form against the numerical integral, constants 7/5, 8/5, ... in name order, 2 values of q."""
    import mpmath

    if rest == 0:
        return True  # only the Coulomb tail: L / q is the limit eps -> 0+ of L q / (q^2 + eps^2)
    names = sorted(rest.free_symbols - {r}, key=lambda s: s.name)
    fixed = {s: sp.Rational(7 + i, 5) for i, s in enumerate(names)}
    for qn, qd in _BORN_CHECK_Q:
        qv = sp.Rational(qn, qd)
        try:
            want = _born_numeric(rest.subs(fixed), r, qv)
            with mpmath.workdps(_BORN_DPS):
                got = to_mp(i_rest.subs({**fixed, q: qv}), _BORN_DPS)
                if not abs(got - want) <= _BORN_CHECK_REL * max(1, abs(want)):
                    return False
        except (TypeError, ValueError, ZeroDivisionError, ToolError):
            return False
    return True


def _born_at(at: Any, v: Any, sy: dict[str, Any]) -> tuple[dict[Any, Any], Any]:
    """Numbers of 'at' for the constants of V, mu and hbar; q from q or from k and theta."""
    if not isinstance(at, dict) or not at:
        raise ToolError(
            "INVALID_ARGUMENT",
            "'at' is an object {name: number}.",
            '{"V0": 1, "a": 1, "mu": 1, "hbar": 1, "q": 1}',
        )
    consts = {s.name: s for s in v.free_symbols if s != sy["r"]}
    known = {**consts, **{n: sy[n] for n in _BORN_RESERVED}}
    nums: dict[str, Any] = {}
    for name, raw in at.items():
        if name not in known:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"at.{name} is not a constant of V(r) nor mu, hbar, q, k or theta.",
            )
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int | float)
            or not math.isfinite(raw)
            or raw <= 0
        ):
            raise ToolError(
                "INVALID_ARGUMENT",
                f"at.{name} must be a finite number > 0 (the constants are positive: write a sign in V).",
            )
        nums[name] = sp.Rational(repr(raw)) if isinstance(raw, float) else sp.Integer(raw)
    for name in (*sorted(consts), "mu", "hbar"):
        if name not in nums:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"'at' has no number for '{name}': give every constant of V(r), mu, hbar and q (or k and theta).",
            )
    with_kt = "k" in nums or "theta" in nums
    if ("q" in nums) == with_kt or (with_kt and not ("k" in nums and "theta" in nums)):
        raise ToolError(
            "INVALID_ARGUMENT",
            "'at' needs q, or k and theta (q = 2 k sin(theta/2)): one of the two.",
        )
    if with_kt and nums["theta"] > sp.pi:
        raise ToolError(
            "INVALID_ARGUMENT", "at.theta is the scattering angle in radians, 0 < theta <= pi."
        )
    qv = nums["q"] if "q" in nums else 2 * nums["k"] * sp.sin(nums["theta"] / 2)
    subs = {consts[n]: nums[n] for n in consts}
    subs.update({sy["mu"]: nums["mu"], sy["hbar"]: nums["hbar"]})
    return subs, qv


def _born_values(at: Any, v: Any, tail: Any, rest: Any, sy: dict[str, Any]) -> dict[str, float]:
    import mpmath

    subs, qv = _born_at(at, v, sy)
    r = sy["r"]
    integral = _born_numeric(rest.subs(subs), r, qv)
    with mpmath.workdps(_BORN_DPS):
        qm = to_mp(qv, _BORN_DPS)
        mu, hbar = to_mp(subs[sy["mu"]], _BORN_DPS), to_mp(subs[sy["hbar"]], _BORN_DPS)
        f = -2 * mu / (hbar**2 * qm) * (integral + to_mp(tail.subs(subs), _BORN_DPS) / qm)
        if isinstance(f, mpmath.mpc):
            if abs(f.imag) > 1e-15 * abs(f):
                raise ToolError("INVALID_ARGUMENT", "V(r) is not real at these numbers.")
            f = f.real
        return {"q": float(qm), "f": float(f), "dsigma_dOmega": float(f * f)}


@op("born_scattering")
def _born_scattering(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    sy = _born_symbols()
    r, q, k, th, mu, hbar = (sy[n] for n in ("r", "q", "k", "theta", "mu", "hbar"))
    v = _born_potential(args, r)
    tail, rest = _born_split(v, r)
    if tail != 0:
        ctx.warn(
            f"V(r) falls as L/r for r -> oo (a Coulomb tail, L = {tail}): the integral of r sin(q r) V(r) does "
            "not converge; the part L/r gives L/q, the limit epsilon -> 0+ of the integral with exp(-epsilon*r) "
            "(a screened potential)."
        )
    ctx.formula = (
        "f(q) = -(2 mu / hbar^2) (1/q) integral_0^oo r sin(q r) V(r) dr, dsigma/dOmega = |f|^2, "
        "q = 2 k sin(theta/2)"
    )
    if args.get("at") is not None:
        vals = _born_values(args["at"], v, tail, rest, sy)
        ctx.formatted = f"f = {vals['f']:.12g}, dsigma/dOmega = {vals['dsigma_dOmega']:.12g}"
        ctx.step("Numerical integral: mpmath.quad up to the steps of V, then mpmath.quadosc.")
        return {"potential": str(v), "regularized": tail != 0, "values": vals}
    i_rest = _born_integral(rest, r, q)
    if i_rest is None:
        raise ToolError(
            "MATH_ERROR",
            "SymPy finds no closed form of the Born integral for this V(r). Give 'at' (numbers for the "
            "constants, mu, hbar and q, or k and theta) for the numerical value.",
            '{"operation": "born_scattering", "potential_energy": "-V0/(1 + exp((r - R)/a))", '
            '"at": {"V0": 50, "R": 4, "a": 0.6, "mu": 1, "hbar": 1, "q": 1}}',
        )
    amp = _born_tidy(-2 * mu / (hbar**2 * q) * (i_rest + tail / q))
    dsig = _born_tidy(sp.Abs(amp) ** 2)
    sub = {q: 2 * k * sp.sin(th / 2)}
    out = {
        "potential": str(v),
        "amplitude": str(amp),
        "dsigma_dOmega": str(dsig),
        "amplitude_theta": str(_born_tidy(amp.subs(sub))),
        "dsigma_dOmega_theta": str(_born_tidy(dsig.subs(sub))),
        "regularized": tail != 0,
        "verified": _born_verified(rest, i_rest, r, q),
    }
    ctx.formatted = f"f(q) = {out['amplitude']}"
    ctx.step("Back-check: the closed form against mpmath.quadosc at 2 values of q.")
    return out
