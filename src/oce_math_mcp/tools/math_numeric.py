"""math_numeric: mpmath, SciPy and NumPy (spec V3, section 6)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import mpmath
import numpy as np
import sympy as sp
from mpmath.libmp.libhyper import NoConvergence

from .. import limits, plot
from ..envelope import Ctx
from ..errors import ToolError
from ..expr import parse, parse_relation, split_list
from ..numfunc import compile_expr, compile_mp, to_mp
from .common import auto_vars, check_bound_order, expr_of, grammar, rel_parts, symbols_for, value_of

Op = Callable[[dict[str, Any], Ctx], dict[str, Any]]
OPS: dict[str, Op] = {}
_ODE_METHODS = {"RK45", "DOP853", "Radau", "BDF", "LSODA"}
_MIN_METHODS = {"SLSQP", "trust-constr", "Nelder-Mead", "Powell", "BFGS", "L-BFGS-B"}


def op(name: str) -> Callable[[Op], Op]:
    def deco(fn: Op) -> Op:
        OPS[name] = fn
        return fn

    return deco


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mpmath.mp.dps = 15
    try:
        return OPS[args["operation"]](args, ctx)
    finally:
        mpmath.mp.dps = 15


def _need(args: dict[str, Any], key: str, example: str) -> Any:
    v = args.get(key)
    if v in (None, "", []):
        raise ToolError(
            "INVALID_ARGUMENT", f"'{args['operation']}' needs the field '{key}'.", example
        )
    return v


def _digits(args: dict[str, Any]) -> int:
    d = int(args.get("digits", 15) or 15)
    if d > limits.DIGITS_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"At most {limits.DIGITS_MAX} digits.")
    return d


def _fl(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


# ------------------------------------------------------------------ arbitrary precision


@op("evalf")
def _evalf(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    d = _digits(args)
    e = parse(_need(args, "expr", "pi"), grammar([]))
    if e.free_symbols:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"The expression has free variables: {sorted(map(str, e.free_symbols))}.",
            "sqrt(2) + pi",
        )
    v = sp.N(e, d)
    ctx.formatted = f"{e} ≈ {v}"
    return {"value": str(v), "digits": d}


def _bounds_mp(raw: list[Any], dps: int) -> tuple[Any, Any]:
    g = grammar([])
    return to_mp(value_of(raw[0], g), dps), to_mp(value_of(raw[1], g), dps)


@op("nsum")
def _nsum(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    d = _digits(args)
    g = grammar(args.get("variables"))
    term = expr_of(_need(args, "expr", "1/n**2"), g)
    n = auto_vars(term, args.get("variables"))[0]
    b = _need(args, "bounds", '[1, "oo"]')
    dps = d + 10
    mpmath.mp.dps = dps
    lo, hi = _bounds_mp(b, dps)
    f = compile_mp(term, [n], dps)
    val = mpmath.nsum(f, [lo, hi])
    ctx.formatted = f"Σ {term} ≈ {mpmath.nstr(val, min(d, 20))}"
    ctx.warn(
        "Numerical sum with convergence acceleration (mpmath.nsum): it is not a proof of convergence."
    )
    return {"value": mpmath.nstr(val, d)}


@op("quad")
def _quad(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    d = _digits(args)
    raw = _need(args, "bounds", '["-oo", "oo"]')
    dps = d + 5
    mpmath.mp.dps = dps
    if raw and isinstance(raw[0], list):
        names = [str(item[0]) for item in raw]
        g = grammar(list(args.get("variables") or []) + names)
        lims = [(sp.Symbol(str(i[0])), value_of(i[1], g), value_of(i[2], g)) for i in raw]
    else:
        g = grammar(args.get("variables"))
        f0 = expr_of(_need(args, "expr", "exp(-x**2)"), g)
        x = auto_vars(f0, args.get("variables"))[0]
        lims = [(x, value_of(raw[0], g), value_of(raw[1], g))]
    f = expr_of(_need(args, "expr", "exp(-x**2)"), g)
    syms = [v for v, _, _ in lims]
    extra = f.free_symbols - set(syms)
    if extra:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Variables without bounds: {sorted(map(str, extra))}.",
            '[["x", 0, 1], ["y", 0, "x"]]',
        )
    check_bound_order(lims)  # D91
    fn = compile_mp(f, syms, dps)
    bound_fns = [
        (
            compile_mp(a, syms, dps) if a.free_symbols else None,
            to_mp(a, dps) if not a.free_symbols else None,
            compile_mp(b, syms, dps) if b.free_symbols else None,
            to_mp(b, dps) if not b.free_symbols else None,
        )
        for _, a, b in lims
    ]
    zero = [mpmath.mpf(0)] * len(syms)
    errors: list[Any] = []

    def integrate(level: int, env: list[Any]) -> Any:
        fa, ca, fb, cb = bound_fns[level]
        lo = fa(*env) if fa else ca
        hi = fb(*env) if fb else cb

        def inner(t: Any) -> Any:
            env2 = list(env)
            env2[level] = t
            return fn(*env2) if level == 0 else integrate(level - 1, env2)

        if level == len(lims) - 1:
            val, err = mpmath.quad(inner, [lo, hi], error=True)
            errors.append(err)
            return val
        return mpmath.quad(inner, [lo, hi])

    val = integrate(len(lims) - 1, zero)
    err = errors[-1] if errors else mpmath.mpf(0)
    ctx.formatted = f"∫ ≈ {mpmath.nstr(val, min(d, 20))}"
    return {"value": mpmath.nstr(val, d), "error_estimate": _fl(err)}


# ------------------------------------------------------------------ roots and minima


@op("root")
def _root(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = grammar(args.get("variables"))
    eqs = [expr_of(t, g) for t in split_list(_need(args, "expr", "cos(x) = x"))]
    xs = symbols_for(args.get("variables")) or sorted(
        set().union(*[e.free_symbols for e in eqs]), key=lambda s: s.name
    )
    if len(eqs) != len(xs):
        raise ToolError(
            "INVALID_ARGUMENT",
            f"{len(eqs)} equations and {len(xs)} unknowns: they must be equal.",
            '["x", "y"]',
        )
    d = _digits(args)
    bounds = args.get("bounds") or []
    x0 = args.get("x0") or []
    if len(xs) == 1 and len(bounds) == 2 and not isinstance(bounds[0], list):
        from scipy.optimize import brentq

        f = compile_expr(eqs[0], xs)
        lo, hi = (float(value_of(v, grammar([])).evalf()) for v in bounds)
        if f(lo) * f(hi) > 0:
            raise ToolError(
                "MATH_ERROR",
                f"f({lo}) and f({hi}) have the same sign: the interval does not surely contain a root.",
            )
        r = brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
        if d > 15:
            mpmath.mp.dps = d + 5
            fm = compile_mp(eqs[0], xs, d + 5)
            r_mp = mpmath.findroot(fm, mpmath.mpf(r))
            ctx.formatted = f"{xs[0]} ≈ {mpmath.nstr(r_mp, d)}"
            return {
                "x": {str(xs[0]): float(r_mp)},
                "x_str": {str(xs[0]): mpmath.nstr(r_mp, d)},
                "residual": abs(_fl(fm(r_mp))),
            }
        ctx.formatted = f"{xs[0]} ≈ {r!r}"
        return {"x": {str(xs[0]): r}, "residual": abs(f(r)), "method": "brentq"}
    if len(x0) != len(xs):
        raise ToolError(
            "INVALID_ARGUMENT",
            "'x0' (one value per unknown) or 'bounds' [lo, hi] with one unknown is needed.",
            "[0.5]",
        )
    dps = max(d, 15) + 5
    mpmath.mp.dps = dps
    fms = [compile_mp(e, xs, dps) for e in eqs]
    start = [mpmath.mpf(repr(float(v))) for v in x0]
    try:
        if len(xs) == 1:
            sol = [mpmath.findroot(fms[0], start[0])]
        else:
            res = mpmath.findroot(lambda *v: [fm(*v) for fm in fms], start)
            sol = [res[i] for i in range(len(xs))]
    except (ZeroDivisionError, ValueError) as exc:
        raise ToolError(
            "MATH_ERROR", f"Root not found starting from x0: {str(exc)[:160]}. Try another x0."
        ) from None
    resid = max(abs(_fl(abs(fm(*sol)))) for fm in fms)
    out = {
        str(x): _fl(v.real if isinstance(v, mpmath.mpc) and v.imag == 0 else v)
        for x, v in zip(xs, sol, strict=True)
    }
    ctx.formatted = ", ".join(f"{k} ≈ {v!r}" for k, v in out.items())
    return {
        "x": out,
        "x_str": {str(x): mpmath.nstr(v, d) for x, v in zip(xs, sol, strict=True)},
        "residual": resid,
    }


def _constraints(raw: list[str], g: Any, xs: list[Any]) -> list[dict[str, Any]]:
    out = []
    for text in raw:
        rel = parse_relation(text, g)
        if isinstance(rel, sp.Eq):
            lhs, rhs, _ = rel_parts(rel)
            kind, expr = "eq", lhs - rhs
        elif isinstance(rel, sp.core.relational.Relational):
            lhs, rhs, rop = rel_parts(rel)
            if rop in (">=", ">"):
                kind, expr = "ineq", lhs - rhs
            else:
                kind, expr = "ineq", rhs - lhs
        else:
            raise ToolError("PARSE_ERROR", f"Invalid constraint: {text!r}.", "x + y <= 4")
        f = compile_expr(expr, xs)
        grads = [compile_expr(sp.diff(expr, v), xs) for v in xs]
        out.append(
            {
                "type": kind,
                "fun": (lambda v, f=f: f(*v)),
                "jac": (lambda v, gs=grads: np.array([gg(*v) for gg in gs])),
                "_text": text,
                "_expr": expr,
            }
        )
    return out


def _box(raw: Any, n: int) -> list[tuple[float | None, float | None]] | None:
    if not raw:
        return None
    if len(raw) != n or not all(isinstance(b, list) and len(b) == 2 for b in raw):
        raise ToolError(
            "INVALID_ARGUMENT",
            "bounds for minimize: [[lo, hi], ...] one per variable (null = no bound).",
            "[[0, 10], [null, 5]]",
        )
    g = grammar([])

    def conv(v: Any) -> float | None:
        if v is None:
            return None
        x = float(value_of(v, g).evalf())
        return None if math.isinf(x) else x

    return [(conv(lo), conv(hi)) for lo, hi in raw]


@op("minimize")
def _minimize(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    from scipy import optimize

    g = grammar(args.get("variables"))
    obj = expr_of(_need(args, "expr", "(1 - x)**2 + 100*(y - x**2)**2"), g)
    xs = auto_vars(obj, args.get("variables"))
    sign = -1.0 if args.get("direction") == "maximize" else 1.0
    f = compile_expr(sign * obj, xs)
    grad = [compile_expr(sp.diff(sign * obj, v), xs) for v in xs]
    cons = _constraints(args.get("constraints") or [], g, xs)
    box = _box(args.get("bounds"), len(xs))
    x0 = [float(v) for v in args.get("x0") or []]
    if not x0:
        x0 = [
            ((lo or 0.0) + (hi if hi is not None else (lo or 0.0))) / 2 if box else 0.0
            for lo, hi in (box or [(None, None)] * len(xs))
        ]
        ctx.warn(f"x0 not given: starting from {x0}.")
    if len(x0) != len(xs):
        raise ToolError("INVALID_ARGUMENT", f"x0 must have {len(xs)} values.", "[0, 0]")
    method = args.get("method", "auto")
    if method != "auto" and method not in _MIN_METHODS | {"differential_evolution"}:
        raise ToolError("INVALID_ARGUMENT", f"Method '{method}' not valid for minimize.", '"SLSQP"')

    def fun(v: Any) -> float:
        r = f(*v)
        return r if math.isfinite(r) else 1e300

    def jac(v: Any) -> Any:
        return np.array([gg(*v) for gg in grad])

    if method == "differential_evolution":
        if not box or any(lo is None or hi is None for lo, hi in box):
            raise ToolError(
                "INVALID_ARGUMENT",
                "differential_evolution needs finite bounds for every variable.",
                "[[-5, 5], [-5, 5]]",
            )
        nlc = [
            optimize.NonlinearConstraint(c["fun"], 0, 0 if c["type"] == "eq" else np.inf)
            for c in cons
        ]
        res = optimize.differential_evolution(
            fun, box, constraints=nlc or (), rng=0, tol=1e-12, polish=True
        )
    else:
        if method == "auto":
            method = "SLSQP" if cons else ("L-BFGS-B" if box else "BFGS")
        use_jac = method not in {"Nelder-Mead", "Powell"}
        kw: dict[str, Any] = {"method": method, "bounds": box}
        if use_jac:
            kw["jac"] = jac
        if cons:
            if method not in {"SLSQP", "trust-constr"}:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    "With constraints use method SLSQP or trust-constr.",
                    '"SLSQP"',
                )
            kw["constraints"] = [{k: c[k] for k in ("type", "fun", "jac")} for c in cons]
        if method == "BFGS":
            kw["options"] = {"gtol": 1e-10, "maxiter": 20000}
        elif method == "SLSQP":
            kw["options"] = {"ftol": 1e-14, "maxiter": 2000}
        elif method in {"Nelder-Mead", "Powell"}:
            kw["options"] = (
                {"xatol": 1e-12, "fatol": 1e-14, "maxiter": 50000}
                if method == "Nelder-Mead"
                else {"xtol": 1e-12, "ftol": 1e-14}
            )
        res = optimize.minimize(fun, np.array(x0, dtype=float), **kw)
    xv = {str(v): float(res.x[i]) for i, v in enumerate(xs)}
    violations = [
        c["_text"]
        for c in cons
        if (c["fun"](res.x) < -1e-7 if c["type"] == "ineq" else abs(c["fun"](res.x)) > 1e-7)
    ]
    if violations:
        ctx.warn(f"Constraints not satisfied at the point found: {violations}.")
    ctx.formatted = f"{'Maximum' if sign < 0 else 'Minimum'} {sign * float(res.fun)!r} at {xv}"
    if len(xs) <= 3 and not box and not cons:
        ctx.warn(
            "Local optimum: for a global optimum use bounds and method 'differential_evolution'."
        )
    return {
        "x": xv,
        "fun": sign * float(res.fun),
        "success": bool(res.success) and not violations,
        "message": str(res.message),
        "method": method,
    }


def _linear(expr: Any, xs: list[Any], what: str) -> tuple[list[float], float]:
    coeffs = []
    for v in xs:
        c = sp.diff(expr, v)
        if c.free_symbols & set(xs):
            raise ToolError("INVALID_ARGUMENT", f"{what} is not linear in {v}.", "3*x + 2*y")
        coeffs.append(float(c))
    const = float(expr.subs(dict.fromkeys(xs, 0)))
    return coeffs, const


@op("linprog")
def _linprog(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    from scipy import optimize

    g = grammar(args.get("variables"))
    obj = expr_of(_need(args, "expr", "3*x + 2*y"), g)
    xs = symbols_for(args.get("variables")) or sorted(obj.free_symbols, key=lambda s: s.name)
    c, c0 = _linear(obj, xs, "The objective")
    sign = -1.0 if args.get("direction") == "maximize" else 1.0
    c = [sign * v for v in c]
    A_ub: list[list[float]] = []
    b_ub: list[float] = []
    A_eq: list[list[float]] = []
    b_eq: list[float] = []
    for text in _need(args, "constraints", '["x + y <= 4", "x >= 0"]'):
        rel = parse_relation(text, g)
        if not isinstance(rel, sp.core.relational.Relational):
            raise ToolError("PARSE_ERROR", f"Invalid constraint: {text!r}.", "x + y <= 4")
        lhs, rhs, rop = rel_parts(rel)
        diff = lhs - rhs
        row, k = _linear(diff, xs, f"The constraint {text!r}")
        if isinstance(rel, sp.Eq):
            A_eq.append(row)
            b_eq.append(-k)
        elif rop in ("<=", "<"):
            A_ub.append(row)
            b_ub.append(-k)
        else:
            A_ub.append([-v for v in row])
            b_ub.append(k)
    box = _box(args.get("bounds"), len(xs)) or [(None, None)] * len(xs)
    if args.get("integer"):
        rows = A_ub + A_eq
        lo: Any = [-np.inf] * len(A_ub) + b_eq
        hi: Any = b_ub + b_eq
        cons = [optimize.LinearConstraint(np.array(rows), lo, hi)] if rows else []
        lb: Any = np.array([-np.inf if b[0] is None else b[0] for b in box], dtype=float)
        ub: Any = np.array([np.inf if b[1] is None else b[1] for b in box], dtype=float)
        bnds = optimize.Bounds(lb, ub)
        res = optimize.milp(
            np.array(c), constraints=cons, integrality=np.ones(len(xs)), bounds=bnds
        )
        method = "milp (HiGHS)"
    else:
        res = optimize.linprog(
            c,
            A_ub=A_ub or None,
            b_ub=b_ub or None,
            A_eq=A_eq or None,
            b_eq=b_eq or None,
            bounds=box,
            method="highs",
        )
        method = "linprog (HiGHS)"
    if res.x is None:
        raise ToolError("MATH_ERROR", f"No solution: {res.message}")
    xv = {str(v): float(res.x[i]) for i, v in enumerate(xs)}
    if args.get("integer"):
        xv = {k: float(round(v)) for k, v in xv.items()}
    fun = sign * float(res.fun) + c0
    ctx.formatted = f"Optimum {fun!r} at {xv}"
    return {
        "x": xv,
        "fun": fun,
        "success": bool(res.success),
        "status": int(res.status),
        "method": method,
    }


# ------------------------------------------------------------------ ODEs, polynomials, identification


@op("ode_ivp")
def _ode_ivp(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    from scipy.integrate import solve_ivp

    names = list(_need(args, "variables", '["t", "y1", "y2"]'))
    if len(names) < 2:
        raise ToolError("INVALID_ARGUMENT", "variables = [time, y1, y2, ...].", '["t", "y"]')
    g = grammar(names)
    rhs = [parse(t, g) for t in split_list(_need(args, "expr", "y2; -y1"))]
    syms = symbols_for(names)
    if len(rhs) != len(syms) - 1:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"{len(rhs)} right-hand sides for {len(syms) - 1} unknowns.",
            '"y2; -y1"',
        )
    fs = [compile_expr(e, syms) for e in rhs]
    y0 = [float(v) for v in _need(args, "x0", "[1, 0]")]
    if len(y0) != len(rhs):
        raise ToolError("INVALID_ARGUMENT", f"x0 must have {len(rhs)} values.", "[1, 0]")
    b = _need(args, "bounds", "[0, 10]")
    gg = grammar([])
    t0, t1 = (float(value_of(v, gg).evalf()) for v in b[:2])
    method = args.get("method", "auto")
    method = "RK45" if method == "auto" else method
    if method not in _ODE_METHODS:
        raise ToolError("INVALID_ARGUMENT", f"Method '{method}' not valid for ode_ivp.", '"RK45"')
    n = int(args.get("samples", 11))
    events = _ode_events(args.get("events"), g, syms, rhs, t0, y0, ctx)

    def f(t: float, y: Any) -> list[float]:
        return [fn(t, *y) for fn in fs]

    if not events:
        res = solve_ivp(
            f, (t0, t1), y0, method=method, rtol=1e-10, atol=1e-12, t_eval=np.linspace(t0, t1, n)
        )
        if not res.success:
            raise ToolError("MATH_ERROR", f"Integration failed: {res.message}")
        final = [float(v) for v in res.y[:, -1]]
        ctx.formatted = f"Final state at t = {t1!r}: {final}"
        return {
            "t": res.t.tolist(),
            "y": res.y.tolist(),
            "final": final,
            "method": method,
            "nfev": int(res.nfev),
        }
    res = solve_ivp(
        f,
        (t0, t1),
        y0,
        method=method,
        rtol=1e-10,
        atol=1e-12,
        events=[e for _, e in events],
        dense_output=True,
    )
    if not res.success:
        raise ToolError("MATH_ERROR", f"Integration failed: {res.message}")
    t_end = float(res.t[-1])
    tt = np.linspace(t0, t_end, n)
    final = [float(v) for v in res.y[:, -1]]
    t_events: list[list[float]] = []
    y_events: list[list[list[float]]] = []
    for (text, _), te, ye in zip(events, res.t_events, res.y_events, strict=True):
        if len(te) > _MAX_EVENT_HITS:
            ctx.warn(f"Event '{text}': {len(te)} zeros, the first {_MAX_EVENT_HITS} reported.")
        t_events.append([float(v) for v in te[:_MAX_EVENT_HITS]])
        y_events.append([[float(c) for c in row] for row in ye[:_MAX_EVENT_HITS]])
    stopped = None
    if res.status == 1:
        stopped = next(
            i
            for i, (_, e) in enumerate(events)
            if e.terminal and len(t_events[i]) and t_events[i][-1] == t_end
        )
        ctx.formatted = (
            f"Stopped by event {stopped} ('{events[stopped][0]}') at t = {t_end!r}: {final}"
        )
    else:
        if any(e.terminal for _, e in events):
            ctx.warn(f"No terminal event before t = {t1!r}: integration to the end.")
        ctx.formatted = f"Final state at t = {t_end!r}: {final}"
    return {
        "t": tt.tolist(),
        "y": res.sol(tt).tolist(),
        "final": final,
        "method": method,
        "nfev": int(res.nfev),
        "t_end": t_end,
        "t_events": t_events,
        "y_events": y_events,
        "stopped_by": stopped,
    }


_EVENT_DIRECTIONS = {"both": 0, "rising": 1, "falling": -1}
_MAX_EVENTS = 10
_MAX_EVENT_HITS = 1000
_EVENT_EXAMPLE = '[{"expr": "y", "direction": "falling"}]'


class _Event:
    """Event function for solve_ivp: SciPy reads the attributes terminal and direction."""

    def __init__(
        self, fn: Callable[..., float], t0: float, sign: float, terminal: bool, direction: int
    ) -> None:
        self.fn, self.t0, self.sign = fn, t0, sign
        self.terminal, self.direction = terminal, direction

    def __call__(self, t: float, y: Any) -> float:
        if self.sign and t == self.t0:
            return self.sign * 1e-300  # the zero at t0 does not count: the sign just after is used
        return self.fn(t, *y)


def _sign_after(e: Any, syms: list[Any], rhs: list[Any], t0: float, y0: list[float]) -> float:
    """Sign of e(t) just after t0 along the motion, if e(t0) = 0: first non-zero derivative (up to the 3rd)."""
    t, ys = syms[0], syms[1:]
    d = e
    for _ in range(3):
        d = sp.diff(d, t) + sum(sp.diff(d, yi) * fi for yi, fi in zip(ys, rhs, strict=True))
        if sp.count_ops(d) > 5000:
            break
        v = compile_expr(d, syms)(t0, *y0)
        if math.isfinite(v) and abs(v) > 1e-12:
            return math.copysign(1.0, v)
    return 0.0


def _ode_events(
    raw: Any, g: Any, syms: list[Any], rhs: list[Any], t0: float, y0: list[float], ctx: Ctx
) -> list[tuple[str, _Event]]:
    """Event functions for solve_ivp (D41). A zero exactly at t0 does not count."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ToolError("INVALID_ARGUMENT", "events must be a list of objects.", _EVENT_EXAMPLE)
    if len(raw) > _MAX_EVENTS:
        raise ToolError("LIMIT_EXCEEDED", f"At most {_MAX_EVENTS} events.")
    out: list[tuple[str, _Event]] = []
    for i, ev in enumerate(raw):
        if not isinstance(ev, dict):
            raise ToolError("INVALID_ARGUMENT", f"events[{i}] must be an object.", _EVENT_EXAMPLE)
        extra = sorted(set(ev) - {"expr", "direction", "terminal"})
        if extra:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"events[{i}]: unexpected key '{extra[0]}' (valid: expr, direction, terminal).",
                _EVENT_EXAMPLE,
            )
        text = ev.get("expr")
        if not isinstance(text, str) or not text.strip():
            raise ToolError("INVALID_ARGUMENT", f"events[{i}]: expr is missing.", _EVENT_EXAMPLE)
        direction = ev.get("direction", "both")
        if direction not in _EVENT_DIRECTIONS:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"events[{i}].direction: use both, rising or falling.",
                _EVENT_EXAMPLE,
            )
        terminal = ev.get("terminal", True)
        if not isinstance(terminal, bool):
            raise ToolError("INVALID_ARGUMENT", f"events[{i}].terminal must be true or false.")
        e = parse(text, g)
        unknown = sorted(str(s) for s in getattr(e, "free_symbols", set()) - set(syms))
        if unknown:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"events[{i}]: '{unknown[0]}' is not among the variables {[str(s) for s in syms]}.",
                _EVENT_EXAMPLE,
            )
        fn = compile_expr(e, syms)
        sign = 0.0
        if abs(fn(t0, *y0)) <= 1e-12:
            sign = _sign_after(e, syms, rhs, t0, y0)
            if sign == 0.0:
                ctx.warn(
                    f"Event '{text}': it is zero at t0 and stays zero in the first derivatives."
                )

        out.append((text, _Event(fn, t0, sign, terminal, _EVENT_DIRECTIONS[direction])))
    return out


@op("polyroots")
def _polyroots(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    d = _digits(args)
    g = grammar(args.get("variables"))
    p = expr_of(_need(args, "expr", "x**5 - x - 1"), g)
    x = auto_vars(p, args.get("variables"))[0]
    try:
        coeffs = sp.Poly(p, x).all_coeffs()
    except sp.PolynomialError:
        raise ToolError(
            "INVALID_ARGUMENT",
            "The expression is not a polynomial in the variable.",
            "x**3 - 2*x + 1",
        ) from None
    if len(coeffs) - 1 > limits.POLY_DEGREE_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"Maximum degree {limits.POLY_DEGREE_MAX}.")
    dps = d + 10
    mpmath.mp.dps = dps
    cm = [to_mp(c, dps) for c in coeffs]
    roots = None
    for steps, extra in ((100, 20), (400, 60), (2000, 200)):
        try:
            roots = mpmath.polyroots(cm, maxsteps=steps, extraprec=extra)
            break
        except NoConvergence:
            continue
    if roots is None:
        raise ToolError("MATH_ERROR", "polyroots does not converge.")
    out = []
    for r in roots:
        if isinstance(r, mpmath.mpc) and abs(r.imag) < mpmath.mpf(10) ** (-d - 2):
            r = r.real
        out.append(mpmath.nstr(r, d))
    ctx.formatted = f"{len(out)} roots"
    return {"roots": out, "degree": len(coeffs) - 1}


@op("identify")
def _identify(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    d = _digits(args)
    text = _need(args, "expr", "1.6180339887498948482")
    v = parse(text, grammar([]))
    if v.free_symbols:
        raise ToolError("INVALID_ARGUMENT", "A numerical value is needed.", "1.6180339887498948482")
    sig = sum(ch.isdigit() for ch in text) if isinstance(text, str) else 15
    dps = max(15, min(d, sig, 60))
    mpmath.mp.dps = dps
    x = to_mp(v, dps)
    consts = {
        "pi": +mpmath.pi,
        "e": +mpmath.e,
        "log(2)": mpmath.log(2),
        "sqrt(2)": mpmath.sqrt(2),
        "sqrt(3)": mpmath.sqrt(3),
    }
    tol = mpmath.mpf(10) ** (-(dps - 3))
    # constants as a dictionary of mpf: mpmath does no eval (with a list of strings it would).
    cands = list(mpmath.identify(x, tol=tol, full=True) or [])
    for c in mpmath.identify(x, consts, tol=tol, full=True) or []:
        if c not in cands:
            cands.append(c)
    cands = cands[:10]
    best = None
    for c in cands:
        try:
            e = parse(c, grammar([]))
        except ToolError:
            continue
        if abs(_fl(sp.N(e - v, dps))) <= _fl(tol) * (1 + abs(_fl(v))):
            best = sp.simplify(e)
            break
    if best is None:
        fr = Fraction(str(sp.N(v, dps))).limit_denominator(10**6)
        best = sp.Rational(fr.numerator, fr.denominator)
        ctx.warn("No closed form found: nsimplify is an approximate fraction.")
    ctx.formatted = f"{text} ≈ {best}"
    ctx.warn("Numerical identification: the closed form is a conjecture, not a proof.")
    return {"candidates": cands, "nsimplify": str(best)}


@op("curve_fit")
def _curve_fit(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    from scipy.optimize import curve_fit

    xname = next(iter(args.get("variables") or ["x"]))
    g = grammar([xname])
    model = expr_of(_need(args, "expr", "a*exp(b*x)"), g)
    x = sp.Symbol(xname)
    params = sorted(model.free_symbols - {x}, key=lambda s: s.name)
    if not params:
        raise ToolError(
            "INVALID_ARGUMENT", "The model has no parameters to estimate.", "a*exp(b*x)"
        )
    xd = np.array(_need(args, "data_x", "[0, 1, 2]"), dtype=float)
    yd = np.array(_need(args, "data_y", "[1, 2.7, 7.4]"), dtype=float)
    if len(xd) != len(yd) or len(xd) < len(params):
        raise ToolError(
            "INVALID_ARGUMENT",
            "data_x and data_y must have the same length, at least as many as the parameters.",
        )
    if len(xd) > limits.LIST_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"At most {limits.LIST_MAX} points.")
    f = compile_expr(model, [x, *params])
    jac_fs = [compile_expr(sp.diff(model, p), [x, *params]) for p in params]
    p0 = [float(v) for v in args.get("x0") or [1.0] * len(params)]
    if len(p0) != len(params):
        raise ToolError(
            "INVALID_ARGUMENT",
            f"x0 must have {len(params)} values (parameters {[str(p) for p in params]}).",
        )

    def model_fn(xv: Any, *pv: float) -> Any:
        return np.array([f(xi, *pv) for xi in xv])

    def jac_fn(xv: Any, *pv: float) -> Any:
        return np.array([[jf(xi, *pv) for jf in jac_fs] for xi in xv])

    popt, pcov = curve_fit(
        model_fn, xd, yd, p0=p0, jac=jac_fn, maxfev=20000, ftol=1e-15, xtol=1e-15, gtol=1e-15
    )
    pred = model_fn(xd, *popt)
    resid = yd - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yd - yd.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    with np.errstate(invalid="ignore"):
        stderr = np.sqrt(np.diag(pcov)) if pcov is not None else [math.nan] * len(params)
    out = {str(p): float(popt[i]) for i, p in enumerate(params)}
    ctx.formatted = f"Parameters {out}, R² = {r2:.6f}"
    return {
        "params": out,
        "std_errors": {str(p): _fl(stderr[i]) for i, p in enumerate(params)},
        "r_squared": r2,
        "residuals": [float(v) for v in resid],
    }


# ------------------------------------------------------------------ plot (D111)


def _plot_range(raw: Any, g: Any, key: str, example: str) -> tuple[float, float]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ToolError("INVALID_ARGUMENT", f"'plot' needs '{key}' as [low, high].", example)
    try:
        lo, hi = (float(value_of(v, g)) for v in raw)
    except TypeError:
        raise ToolError("INVALID_ARGUMENT", f"'{key}' must be numbers.", example) from None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ToolError("INVALID_ARGUMENT", f"'{key}' must be finite.", example)
    if not lo < hi:
        raise ToolError(
            "INVALID_ARGUMENT", f"'{key}' must be [low, high] with low < high.", example
        )
    return lo, hi


@op("plot")
def _plot(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    names = args.get("variables") or []
    if len(names) > 1:
        raise ToolError(
            "INVALID_ARGUMENT",
            "'plot' draws functions of one variable: give one name in 'variables'.",
            '["x"]',
        )
    g = grammar(list(names))
    texts = split_list(_need(args, "expr", "sin(x); cos(x)"))
    if len(texts) > limits.PLOT_CURVES_MAX:
        raise ToolError(
            "LIMIT_EXCEEDED", f"At most {limits.PLOT_CURVES_MAX} functions in one plot."
        )
    exprs: list[Any] = []
    labels: list[str] = []
    for text in texts:
        rel = parse_relation(text, g)
        if (
            isinstance(rel, sp.Eq)
            and isinstance(rel.lhs, sp.Symbol)
            and rel.lhs not in rel.rhs.free_symbols
            and rel.lhs.name not in names
            and (rel.rhs.free_symbols or rel.lhs.name != "x")
        ):
            exprs.append(rel.rhs)
            labels.append(f"{rel.lhs} = {rel.rhs}")
        else:
            e = expr_of(text, g)
            exprs.append(e)
            labels.append(str(e))
    free: set[Any] = set().union(*(e.free_symbols for e in exprs))
    if names:
        var = symbols_for(names)[0]
    elif len(free) == 1:
        var = next(iter(free))
    else:
        var = sp.Symbol("x")
    extra = free - {var}
    if extra:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Symbols without a value: {sorted(map(str, extra))}. 'plot' draws functions of one variable; "
            "replace the others with numbers.",
            "sin(2*x)",
        )
    a, b = _plot_range(_need(args, "bounds", '[0, "2*pi"]'), g, "bounds", '[0, "2*pi"]')
    window = None
    if args.get("y_bounds") not in (None, []):
        window = _plot_range(args["y_bounds"], g, "y_bounds", "[-2, 2]")
    samples = int(args.get("samples", 11) or 11)  # 2 to 1001, checked by the schema
    d = min(_digits(args), 17)
    fs = [compile_expr(e, [var]) for e in exprs]
    out = plot.make(
        fs,
        labels,
        a,
        b,
        variable=str(var),
        window=window,
        deadline=time.monotonic() + limits.PLOT_TIME_S,
    )
    for w in out["warnings"]:
        ctx.warn(w)

    def r(v: float) -> float | None:
        return float(f"{v:.{d}g}") if math.isfinite(v) else None

    table = []
    for x in np.linspace(a, b, samples):
        xf = float(x)
        table.append([r(xf)] + [r(f(xf)) for f in fs])
    lo, hi = out["y_range"]
    ctx.formatted = (
        f"Plot of {len(fs)} function(s) for {var} in [{a:.6g}, {b:.6g}], y window [{lo:.6g}, {hi:.6g}], "
        f"{len(out['breaks'])} break(s)."
    )
    return {
        "svg": out["svg"],
        "columns": [str(var), *labels],
        "table": table,
        "x_range": [a, b],
        "y_range": [lo, hi],
        "breaks": out["breaks"],
        "points": out["points"],
    }
