"""math_verify: checking an answer before giving it (spec V3, section 7).

equivalent with the numerical method and check_inequality are NOT proofs: they say that no
counterexample was found on fixed-seed points (repeatable result).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

import sympy as sp

from ..envelope import Ctx
from ..errors import ToolError
from ..expr import parse, parse_relation, split_list
from ..numfunc import compile_expr
from ..unitdim import dims_json, parse_unit, si_text
from .common import auto_vars, expr_of, grammar, is_zero, rel_parts, value_of

Op = Callable[[dict[str, Any], Ctx], dict[str, Any]]
OPS: dict[str, Op] = {}
_SPECIAL = [
    sp.Integer(0),
    sp.Integer(1),
    sp.Integer(-1),
    sp.Rational(1, 2),
    sp.Integer(2),
    sp.Rational(-1, 2),
    sp.Rational(1, 3),
    sp.Integer(3),
    sp.Integer(10),
    sp.Integer(-10),
    sp.Rational(1, 10),
    sp.Rational(-3, 2),
]


def op(name: str) -> Callable[[Op], Op]:
    def deco(fn: Op) -> Op:
        OPS[name] = fn
        return fn

    return deco


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    return OPS[args["operation"]](args, ctx)


def _points(xs: list[Any], domain: str, n: int) -> list[dict[Any, Any]]:
    """Repeatable test points: first the special points, then random ones with a fixed seed."""
    rng = random.Random(20261124)  # noqa: S311 - repeatable test points, not cryptography
    pts: list[dict[Any, Any]] = []
    specials = list(_SPECIAL)
    if domain == "positive":
        specials = [s for s in specials if s > 0] + [sp.Rational(1, 100), sp.Integer(100)]
    elif domain == "integer":
        specials = [s for s in specials if s.is_Integer] + [sp.Integer(-2), sp.Integer(7)]
    elif domain == "complex":
        specials = [*specials[:6], sp.I, 1 + sp.I, -sp.I / 2]
    for k in range(min(len(specials), n)):
        pts.append({x: specials[(k + 3 * i) % len(specials)] for i, x in enumerate(xs)})
    while len(pts) < n:
        pt = {}
        for x in xs:
            if domain == "positive":
                v: Any = sp.Rational(round(math.exp(rng.uniform(-4.6, 4.6)) * 1000) or 1, 1000)
            elif domain == "integer":
                v = sp.Integer(rng.randint(-50, 50))
            elif domain == "complex":
                v = sp.Rational(rng.randint(-1000, 1000), 100) + sp.I * sp.Rational(
                    rng.randint(-1000, 1000), 100
                )
            else:
                v = sp.Rational(rng.randint(-10000, 10000), 1000)
            pt[x] = v
        pts.append(pt)
    return pts[:n]


def _val(e: Any, pt: dict[Any, Any]) -> complex | None:
    try:
        v = complex(sp.N(e.subs(pt), 30))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    return v


def _num_pt(pt: dict[Any, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pt.items():
        c = complex(sp.N(v))
        out[str(k)] = c.real if c.imag == 0 else str(v)
    return out


@op("equivalent")
def _equivalent(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = grammar(args.get("variables"))
    a = expr_of(args["expr"], g)
    if not args.get("expected"):
        raise ToolError(
            "INVALID_ARGUMENT", "'expected' is needed, the expression to compare.", "1 - cos(x)**2"
        )
    b = expr_of(args["expected"], g)
    diff = a - b
    xs = sorted((a.free_symbols | b.free_symbols), key=lambda s: s.name)
    try:
        sym_zero = sp.simplify(diff) == 0
    except (TypeError, ValueError, NotImplementedError, RecursionError):
        sym_zero = False
    if sym_zero:
        ctx.step(f"Difference: ({a}) - ({b}).")
        ctx.step("Simplified: 0. The two expressions are identical (symbolic proof).")
        ctx.formatted = "Equivalent (symbolic proof)."
        return {"equivalent": True, "method": "symbolic", "counterexample": None}
    tol = float(args.get("tolerance", 1e-9))
    n = int(args.get("samples", 64))
    domain = args.get("domain", "real")
    checked = 0
    for pt in _points(xs, domain, n):
        va, vb = _val(a, pt), _val(b, pt)
        if va is None and vb is None:
            continue
        if va is None or vb is None or abs(va - vb) > tol * (1 + max(abs(va), abs(vb))):
            ctx.formatted = f"NOT equivalent: at {_num_pt(pt)} the values are {va} and {vb}."
            return {
                "equivalent": False,
                "method": "numeric",
                "counterexample": {
                    "point": _num_pt(pt),
                    "expr_value": str(va),
                    "expected_value": str(vb),
                },
            }
        checked += 1
    ctx.warn(f"No counterexample on {checked} points: this is NOT a proof.")
    ctx.formatted = f"Probably equivalent ({checked} points checked)."
    return {
        "equivalent": True,
        "method": "numeric",
        "counterexample": None,
        "points_checked": checked,
    }


@op("check_solution")
def _check_solution(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    sol = args.get("solution")
    if not isinstance(sol, dict) or not sol:
        raise ToolError(
            "INVALID_ARGUMENT", "'solution' is needed as an object {variable: value}.", '{"x": "2"}'
        )
    names = list(args.get("variables") or []) + [str(k) for k in sol]
    g = grammar(names)
    eqs = [expr_of(t, g) for t in split_list(args["expr"])]
    subs = {sp.Symbol(str(k)): value_of(v, g) for k, v in sol.items()}
    tol = float(args.get("tolerance", 1e-9))
    residuals = []
    valid = True
    approx = False
    for e in eqs:
        r = sp.simplify(e.subs(subs))
        residuals.append(str(r))
        if r.free_symbols:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Missing the value of {sorted(map(str, r.free_symbols))}.",
                '{"x": "2", "y": "1"}',
            )
        if is_zero(r):
            continue
        v = abs(complex(sp.N(r, 30)))
        if v <= tol:
            approx = True
        else:
            valid = False
    if approx and valid:
        ctx.warn(f"Residuals not exactly zero but below the tolerance {tol}.")
    ctx.step(f"Substituting {', '.join(f'{k} = {v}' for k, v in subs.items())} into each equation.")
    ctx.step(f"Residuals: {residuals}.")
    ctx.step(f"Outcome: {'correct solution' if valid else 'wrong solution'}.")
    ctx.formatted = "Correct solution." if valid else "WRONG solution."
    return {"valid": valid, "residuals": residuals}


def _var(args: dict[str, Any], e: Any) -> Any:
    return auto_vars(e, args.get("variables"))[0]


@op("check_derivative")
def _check_derivative(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = grammar(args.get("variables"))
    f = expr_of(args["expr"], g)
    if not args.get("expected"):
        raise ToolError(
            "INVALID_ARGUMENT", "'expected' is needed with the claimed derivative.", "3*x**2"
        )
    fp = expr_of(args["expected"], g)
    x = _var(args, f)
    true = sp.diff(f, x)
    diff = sp.simplify(true - fp)
    valid = is_zero(diff)
    ctx.step(f"Differentiating f = {f}: f' = {true}.")
    ctx.step(f"Difference from the claimed derivative: {diff}.")
    ctx.formatted = "Correct derivative." if valid else f"WRONG derivative: the true one is {true}."
    return {"valid": valid, "difference": str(diff), "true_derivative": str(sp.simplify(true))}


@op("check_antiderivative")
def _check_antiderivative(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = grammar(args.get("variables"))
    F = expr_of(args["expr"], g)
    if not args.get("expected"):
        raise ToolError("INVALID_ARGUMENT", "'expected' is needed with the integrand f.", "log(x)")
    f = expr_of(args["expected"], g)
    x = _var(args, F)
    dF = sp.diff(F, x)
    diff = sp.simplify(dF - f)
    valid = is_zero(diff)
    ctx.step(f"Differentiating the antiderivative F = {F}: F' = {sp.simplify(dF)}.")
    ctx.step(f"Comparison with f = {f}: difference {diff}.")
    ctx.formatted = "Correct antiderivative." if valid else "WRONG antiderivative."
    return {"valid": valid, "difference": str(diff)}


@op("check_numeric")
def _check_numeric(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = grammar(args.get("variables"))
    exact = parse(args["expr"], g)
    if not args.get("expected"):
        raise ToolError(
            "INVALID_ARGUMENT", "'expected' is needed with the claimed value.", "1.41421356"
        )
    claimed = parse(args["expected"], g)
    if exact.free_symbols or claimed.free_symbols:
        raise ToolError(
            "INVALID_ARGUMENT", "check_numeric compares two numbers, without variables.", "sqrt(2)"
        )
    ev = complex(sp.N(exact, 50))
    cv = complex(sp.N(claimed, 50))
    tol = float(args.get("tolerance", 1e-9))
    err_abs = abs(ev - cv)
    rel = err_abs / abs(ev) if ev != 0 else err_abs
    valid = rel <= tol
    ctx.step(f"Exact value: {sp.N(exact, 30)}.")
    ctx.step(f"Claimed value: {claimed}; relative error {rel:.3e} (tolerance {tol:.1e}).")
    ctx.formatted = (
        "Value correct within the tolerance." if valid else "Value OUTSIDE the tolerance."
    )
    return {"valid": valid, "exact_value": str(sp.N(exact, 30)), "relative_error": rel}


@op("check_inequality")
def _check_inequality(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = grammar(args.get("variables"))
    rel = parse_relation(args["expr"], g)
    if not isinstance(rel, sp.core.relational.Relational) or isinstance(rel, sp.Eq | sp.Ne):
        raise ToolError(
            "PARSE_ERROR", "An inequality with <, <=, >, >= is needed.", "(a + b)/2 >= sqrt(a*b)"
        )
    domain = args.get("domain", "real")
    if domain == "complex":
        raise ToolError("INVALID_ARGUMENT", "Inequalities hold only over the reals.", '"real"')
    lhs, rhs, rop = rel_parts(rel)
    diff = lhs - rhs if rop in (">=", ">") else rhs - lhs
    strict = rop in (">", "<")
    xs = sorted(cast(set[sp.Symbol], diff.free_symbols), key=lambda s: s.name)
    n = int(args.get("samples", 64))
    tol = float(args.get("tolerance", 1e-9))
    fast = compile_expr(diff, xs)
    checked = skipped = 0
    for pt in _points(xs, domain, n):
        v = fast(*[float(pt[x]) for x in xs])
        scale = 1.0
        if not math.isfinite(v) or abs(v) < 1e-6:
            exact = _val(diff, pt)
            if exact is None or abs(exact.imag) > 1e-20:
                skipped += 1
                continue
            v = exact.real
            scale = (
                1
                + abs(complex(sp.N(rel.lhs.subs(pt), 30)))
                + abs(complex(sp.N(rel.rhs.subs(pt), 30)))
            )
        bad = v <= tol * scale if strict else v < -tol * scale
        if bad:
            cex = {str(x): float(pt[x]) for x in xs}
            ctx.formatted = f"Inequality FALSE at {cex}."
            return {"holds_on_samples": False, "counterexample": cex, "points_checked": checked + 1}
        checked += 1
    if skipped:
        ctx.warn(f"{skipped} points discarded because outside the domain of the functions.")
    ctx.warn(f"No counterexample on {checked} points: this is NOT a proof.")
    ctx.formatted = f"True on all the {checked} points tried."
    return {"holds_on_samples": True, "counterexample": None, "points_checked": checked}


# ------------------------------------------------------------------ check_units (D42)

_SAME = 1e-12


def _g15(v: float | None) -> float | None:
    """15 significant digits: 3.5999999999999996 becomes 3.6."""
    return None if v is None else float(f"{v:.15g}")


class _UnitCheck:
    """Dimension and SI factor of each node of the formula; collects problems and warnings."""

    def __init__(self, units: dict[Any, tuple[Any, float]]) -> None:
        self.units = units
        self.problems: list[str] = []
        self.warnings: list[str] = []

    def walk(self, e: Any) -> tuple[Any, float | None]:
        if e.is_Number or isinstance(e, sp.NumberSymbol) or e in (sp.I, sp.E):
            return {}, 1.0
        if e.is_Symbol:
            return self.units[e]
        if isinstance(e, sp.Add | sp.Min | sp.Max):
            parts = [(a, *self.walk(a)) for a in e.args]
            known = [(a, d, s) for a, d, s in parts if d is not None]
            if any(d != known[0][1] for _, d, _ in known[1:]):
                text = "; ".join(f"{a}: {si_text(d)}" for a, d, _ in known)
                self.problems.append(
                    f"sum (or comparison) of terms with different dimensions: {text}"
                )
                return None, None
            if len(known) < len(parts):
                return None, None
            scales = [s for _, _, s in known if s is not None]
            if len(scales) == len(parts) and all(
                math.isclose(s, scales[0], rel_tol=_SAME) for s in scales
            ):
                return known[0][1], scales[0]
            if len(scales) == len(parts):
                self.warnings.append(
                    f"In '{e}' the terms have the same dimension but different units (SI factors "
                    f"{sorted({float(f'{s:.6g}') for s in scales})}): convert before adding."
                )
            return known[0][1], None
        if isinstance(e, sp.Mul):
            dims: dict[str, Any] = {}
            scale: float | None = 1.0
            for a in e.args:
                d, s = self.walk(a)
                if d is None:
                    return None, None
                for k, v in d.items():
                    dims[k] = dims.get(k, 0) + v
                scale = None if scale is None or s is None else scale * s
            return {k: v for k, v in dims.items() if v != 0}, scale
        if isinstance(e, sp.Pow):
            bd, bs = self.walk(e.base)
            if e.exp.is_Number:
                q = Fraction(str(e.exp))
                if bd is None:
                    return None, None
                return (
                    {k: v * q for k, v in bd.items() if v * q != 0},
                    None if bs is None else bs ** float(q),
                )
            ed, _ = self.walk(e.exp)
            if ed:
                self.problems.append(f"exponent with dimensions in '{e}': {si_text(ed)}")
            if bd:
                self.problems.append(f"non-numeric exponent on a quantity with dimensions in '{e}'")
                return None, None
            return ({}, 1.0) if ed == {} and bd == {} else (None, None)
        if isinstance(e, sp.Abs):
            return self.walk(e.args[0])
        if isinstance(e, sp.Function):
            ok = True
            for a in e.args:
                d, s = self.walk(a)
                if d is None:
                    ok = False
                elif d:
                    self.problems.append(
                        f"{e.func}: the argument '{a}' has dimension {si_text(d)}, it must be a pure number"
                    )
                    ok = False
                elif s is not None and not math.isclose(s, 1.0, rel_tol=_SAME):
                    self.warnings.append(
                        f"{e.func}: the argument '{a}' is in non-SI units (factor {s:.6g}, for example degrees): "
                        "the functions work in radians and pure numbers; convert first"
                    )
            return ({}, 1.0) if ok else (None, None)
        raise ToolError(
            "INVALID_ARGUMENT", f"check_units does not handle '{type(e).__name__}' in '{e}'."
        )


@op("check_units")
def _check_units(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    raw = args.get("units")
    example = '{"F": "N", "m": "kg", "a": "m/s^2"}'
    if not isinstance(raw, dict) or not raw:
        raise ToolError(
            "INVALID_ARGUMENT", "'units' is needed as an object {variable: unit}.", example
        )
    g = grammar([str(k) for k in raw])
    units: dict[Any, tuple[Any, float]] = {}
    coulomb = False
    for k, v in raw.items():
        scale, dims, is_c = parse_unit(v)
        coulomb = coulomb or is_c
        units[sp.Symbol(str(k))] = (dims, scale)
    rel = parse_relation(args["expr"], g)
    sides = list(rel_parts(rel)[:2]) if isinstance(rel, sp.core.relational.Relational) else [rel]
    missing = sorted(
        {str(s) for side in sides for s in side.free_symbols} - {str(s) for s in units}
    )
    if missing:
        raise ToolError("INVALID_ARGUMENT", f"Missing the unit of {missing}.", example)
    chk = _UnitCheck(units)
    res = [chk.walk(side) for side in sides]
    out: dict[str, Any] = {}
    if len(res) == 2:
        (ld, ls), (rd, rs) = res
        if ld is not None and rd is not None and ld != rd:
            chk.problems.append(
                f"the two sides have different dimensions: {si_text(ld)} and {si_text(rd)}"
            )
        ratio = rs / ls if ls and rs and ld == rd else None  # meaningless if the dimensions differ
        if ratio is not None and ld == rd and not math.isclose(ratio, 1.0, rel_tol=_SAME):
            chk.warnings.append(
                f"Same dimensions but different units: in SI the right-hand side is {ratio:.6g} times the left one; "
                "convert the units before using the formula."
            )
        out |= {
            "lhs_dimension": None if ld is None else dims_json(ld),
            "rhs_dimension": None if rd is None else dims_json(rd),
            "lhs_si_unit": None if ld is None else si_text(ld),
            "rhs_si_unit": None if rd is None else si_text(rd),
            "scale_ratio": _g15(ratio),
        }
        dims, scale = (ld, ls) if ld == rd else (None, None)
    else:
        dims, scale = res[0]
    consistent = not chk.problems and dims is not None
    out = {
        "consistent": consistent,
        "dimension": None if dims is None else dims_json(dims),
        "si_unit": None if dims is None else si_text(dims),
        "scale_to_si": _g15(scale),
        "problems": chk.problems,
        **out,
    }
    if args.get("expected"):
        _, edims, _ = parse_unit(args["expected"])
        out["expected_ok"] = dims is not None and edims == dims
    if coulomb:
        chk.warnings.append("'C' is the coulomb (electric charge); for degrees Celsius use degC.")
    for w in chk.warnings:
        ctx.warn(w)
    ctx.formatted = (
        f"Consistent units: {out['si_unit']}."
        if consistent
        else f"Units NOT consistent: {'; '.join(chk.problems)}."
    )
    return out
