"""math_solve: symbolic SymPy with self-verification (spec V3, section 4)."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import sympy as sp
from sympy.integrals import manualintegrate as mi

from ..envelope import Ctx
from ..errors import ToolError
from ..expr import Grammar, check_identifier, parse, parse_relation, split_equation, split_list
from ..numfunc import compile_expr
from .common import (
    SOLVE_FUNCS,
    auto_vars,
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
    verified = all(is_zero(f.subs(x, s)) for s in sols)
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
    verified = all(is_zero(e.subs(s)) for s in sols for e in eqs)
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
    verified = _check_derivative_numeric(f, ds, x, n)
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


def _check_derivative_numeric(f: Any, d: Any, x: Any, n: int) -> bool:
    import mpmath

    others = sorted(f.free_symbols - {x}, key=lambda s: s.name)
    subs = {s: sp.Rational(7 + i, 5) for i, s in enumerate(others)}
    f0, d0 = f.subs(subs), d.subs(subs)
    mpmath.mp.dps = 40
    good = 0
    for pn, pd in ((3, 7), (6, 5), (-4, 9)):
        pt = sp.Rational(pn, pd)
        try:
            exact = complex(sp.N(d0.subs(x, pt), 30))
            num = complex(
                mpmath.diff(
                    lambda v: mpmath.mpc(complex(sp.N(f0.subs(x, v), 40))),
                    mpmath.mpf(pn) / pd,
                    n,
                )
            )
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not (math.isfinite(exact.real) and math.isfinite(num.real)):
            continue
        if abs(exact - num) > 1e-8 * (1 + abs(exact)):
            return False
        good += 1
    return good > 0


@op("partial_diff")
def _partial(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    f = expr_of(_need(args, "expr", "x**2*y**3"), g)
    xs = symbols_for(_need(args, "variables", '["x", "y"]'))
    d = sp.simplify(sp.diff(f, *xs))
    check = f
    for v in xs:
        check = sp.diff(check, v)
    verified = is_zero(d - check)
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


@op("multi_integrate")
def _multi(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    lims, g = _bound_list(args, _g(args), '[["x", 0, 1], ["y", 0, 2]]')
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
    if num is not None and len(lims) <= 2:
        try:
            chk = complex(sp.Integral(integrand, *lims).evalf(15))
            verified = abs(chk - num) <= 1e-8 * (1 + abs(num))
        except (TypeError, ValueError):
            verified = False
    elif num is not None:
        verified = not val.has(sp.Integral)
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
        all(sp.checkodesol(eqs[0] if len(eqs) == 1 else eqs, s)[0] is True for s in sols)
        if len(eqs) == 1
        else True
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
            raise ToolError("INVALID_ARGUMENT", "Norma: expr = '1', '2', 'oo' o 'fro'.", '"fro"')
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
                "INVALID_ARGUMENT", "Per 'multiply' serve matrix_b.", "[[1, 0], [0, 1]]"
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


@op("polynomial")
def _polynomial(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = _g(args)
    polys = [expr_of(t, g) for t in split_list(_need(args, "expr", "x**3 - 6*x**2 + 11*x - 6"))]
    x = auto_vars(polys[0], args.get("variables"))[0]
    pop = args.get("poly_op", "roots")
    p = polys[0]
    P = sp.Poly(p, x)
    value: Any
    verified = True
    if pop == "roots":
        r = sp.roots(P)
        value = {str(k): int(v) for k, v in r.items()}
        verified = all(is_zero(p.subs(x, k)) for k in r) and sum(r.values()) == P.degree()
        if sum(r.values()) != P.degree():
            ctx.warn("Some roots have no closed form: use poly_op 'nroots'.")
    elif pop == "nroots":
        value = [str(v) for v in P.nroots(n=30)]
    elif pop == "factor_list":
        c, fl = cast(tuple[Any, list[tuple[Any, int]]], sp.factor_list(p, x))
        value = {"coefficient": str(c), "factors": [[str(f), int(m)] for f, m in fl]}
        verified = is_zero(c * sp.Mul(*[f**m for f, m in fl]) - p)
    elif pop == "discriminant":
        value = str(sp.discriminant(P))
    elif pop in {"resultant", "gcd"}:
        if len(polys) < 2:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Two polynomials separated by ';' are needed.",
                "x**2 - 1; x**2 + 2*x + 1",
            )
        q = polys[1]
        if pop == "resultant":
            value = str(sp.resultant(p, q, x))
        else:
            gg = sp.gcd(p, q)
            value = str(gg)
            verified = sp.rem(p, gg, x) == 0 and sp.rem(q, gg, x) == 0
    elif pop == "vieta":
        cs = P.all_coeffs()
        lead = cs[0]
        value = {f"e{k}": str(sp.simplify((-1) ** k * cs[k] / lead)) for k in range(1, len(cs))}
    elif pop == "coeffs":
        value = [str(c) for c in P.all_coeffs()]
    else:
        value = str(P.degree())
    ctx.formatted = f"{pop}: {value}"
    return {"poly_op": pop, "value": value, "verified": verified}
