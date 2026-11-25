"""math_eval: exact scalar expression (Decimal/Fraction/float), statistics, finance, units."""

from __future__ import annotations

import ast
import math
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from typing import Any, cast

import mpmath
import sympy as sp

from .. import limits
from ..constants import exact_value as _const_value
from ..envelope import Ctx
from ..errors import ToolError
from ..expr import Grammar, check_identifier, parse
from ..fmt import dec_str, en_number, to_decimal
from .common import SOLVE_FUNCS

# ---------------------------------------------------------------- units (exact factors)
_LBF = "4.4482216152605"
_IN = "0.0254"
_FT = "0.3048"
_UNITS: dict[str, tuple[str, Any]] = {}


def _u(category: str, factor: Any, *names: str) -> None:
    f = factor if isinstance(factor, sp.Basic) else sp.Rational(factor)
    for n in names:
        _UNITS[n] = (category, f)


R = sp.Rational
_u("length", 1, "m")
_u("length", 1000, "km")
_u("length", R(1, 100), "cm")
_u("length", R(1, 1000), "mm")
_u("length", R(1, 10**6), "um")
_u("length", R(1, 10**9), "nm")
_u("length", R(_IN), "in", "inch")
_u("length", R(_FT), "ft")
_u("length", R("0.9144"), "yd")
_u("length", R("1609.344"), "mi")
_u("length", 1852, "nmi")
_u("area", 1, "m2")
_u("area", R(1, 10**4), "cm2")
_u("area", R(1, 10**6), "mm2")
_u("area", 10**6, "km2")
_u("area", 10**4, "ha")
_u("area", R(_IN) ** 2, "in2")
_u("area", R(_FT) ** 2, "ft2")
_u("volume", 1, "m3")
_u("volume", R(1, 1000), "dm3", "L", "l")
_u("volume", R(1, 10**6), "cm3", "mL", "ml")
_u("volume", R(1, 10**9), "mm3")
_u("volume", R(_IN) ** 3, "in3")
_u("volume", R(_FT) ** 3, "ft3")
_u("volume", R("0.003785411784"), "gal")
_u("volume", R("0.00454609"), "gal_uk")
_u("mass", 1, "kg")
_u("mass", R(1, 1000), "g")
_u("mass", R(1, 10**6), "mg")
_u("mass", 1000, "t")
_u("mass", R("0.45359237"), "lb")
_u("mass", R("0.45359237") / 16, "oz")
_u("force", 1, "N")
_u("force", 1000, "kN")
_u("force", 10**6, "MN")
_u("force", R("9.80665"), "kgf")
_u("force", R(_LBF), "lbf")
_u("pressure", 1, "Pa")
_u("pressure", 1000, "kPa")
_u("pressure", 10**6, "MPa", "N/mm2")
_u("pressure", 10**9, "GPa")
_u("pressure", 10**5, "bar")
_u("pressure", 100, "mbar")
_u("pressure", R(_LBF) / R(_IN) ** 2, "psi")
_u("pressure", 1000 * R(_LBF) / R(_IN) ** 2, "ksi")
_u("pressure", 101325, "atm")
_u("pressure", R("133.322387415"), "mmHg")
_u("torque", 1, "Nm")
_u("torque", 1000, "kNm")
_u("torque", R(1, 100), "Ncm")
_u("torque", R(1, 1000), "Nmm")
_u("torque", R("9.80665"), "kgm", "kgfm")
_u("torque", R(_LBF) * R(_FT), "lbf_ft")
_u("torque", R(_LBF) * R(_IN), "lbf_in")
_u("energy", 1, "J")
_u("energy", 1000, "kJ")
_u("energy", 10**6, "MJ")
_u("energy", 3600, "Wh")
_u("energy", 3600000, "kWh")
_u("energy", 3600000000, "MWh")
_u("energy", R("4.184"), "cal")
_u("energy", R("4184"), "kcal")
_u("energy", R("1055.05585262"), "BTU")
_u("energy", R("1.602176634e-19"), "eV")
_u("power", 1, "W")
_u("power", 1000, "kW")
_u("power", 10**6, "MW")
_u("power", 550 * R(_FT) * R(_LBF), "HP", "hp")
_u("power", R("735.49875"), "CV")
_u("power", R("1055.05585262") / 3600, "BTU_h")
_u("angle", 1, "rad")
_u("angle", sp.pi / 180, "deg")
_u("angle", sp.pi / 200, "grad")
_u("angle", 2 * sp.pi, "rev", "turn")
_u("angle", sp.pi / 10800, "arcmin")
_u("angle", sp.pi / 648000, "arcsec")
_u("velocity", 1, "m/s")
_u("velocity", R(1000, 3600), "km/h")
_u("velocity", R("1609.344") / 3600, "mph")
_u("velocity", R(1852, 3600), "kn")
_u("velocity", R(_FT), "ft/s")
_u("flow_rate", 1, "m3/s")
_u("flow_rate", R(1, 3600), "m3/h")
_u("flow_rate", R(1, 1000), "L/s")
_u("flow_rate", R(1, 60000), "L/min")
_u("flow_rate", R("0.003785411784") / 60, "gpm")
_u("time", 1, "s")
_u("time", 60, "min")
_u("time", 3600, "h")
_u("time", 86400, "d")

_TEMPS = {"C", "K", "F"}


def unit_convert(value: Any, src: Any, dst: Any) -> Any:
    if not isinstance(src, str) or not isinstance(dst, str):
        raise ToolError(
            "INVALID_ARGUMENT",
            "Units must be written in quotes.",
            "unit_convert(1, 'in', 'mm')",
        )
    if src in _TEMPS or dst in _TEMPS:
        if src not in _TEMPS or dst not in _TEMPS:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"{src} cannot be converted to {dst}.",
                "unit_convert(100, 'C', 'F')",
            )
        k = {"C": value + R("273.15"), "K": value, "F": (value - 32) * R(5, 9) + R("273.15")}[src]
        return {"C": k - R("273.15"), "K": k, "F": (k - R("273.15")) * R(9, 5) + 32}[dst]
    if src not in _UNITS or dst not in _UNITS:
        bad = src if src not in _UNITS else dst
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Unknown unit '{bad}'. Examples: {', '.join(list(_UNITS)[:20])}.",
            "unit_convert(1, 'bar', 'psi')",
        )
    (c1, f1), (c2, f2) = _UNITS[src], _UNITS[dst]
    if c1 != c2:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"'{src}' is {c1}, '{dst}' is {c2}: not convertible.",
            "unit_convert(1, 'm', 'ft')",
        )
    return value * f1 / f2


def unit_factor(name: str) -> tuple[str, float] | None:
    """Category and factor towards the SI unit of the category; None if unknown (temperatures excluded)."""
    found = _UNITS.get(name)
    return None if found is None else (found[0], float(found[1]))


# ---------------------------------------------------------------- helper functions


_VAT_RATE = sp.Integer(22)  # Italian standard VAT rate, %
_WITHHOLDING_RATE = sp.Integer(20)  # Italian withholding tax on professional fees, %


def _seq(args: tuple[Any, ...]) -> list[Any]:
    data = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
    if not data:
        raise ToolError("INVALID_ARGUMENT", "Empty list.", "mean([1, 2, 3])")
    return [sp.nsimplify(v) if isinstance(v, float) else v for v in data]


def _real(x: Any) -> Any:
    v = sp.sympify(x) if not isinstance(x, sp.Basic) else x
    if not v.is_number:
        raise ToolError("INVALID_ARGUMENT", "A numerical value is needed.", "round(2.675, 2)")
    return v


def _whole(v: Any, what: str, example: str) -> int:
    """An integer argument: 3.0 is accepted, 12.5 is INVALID_ARGUMENT (D91: int() truncated it in silence)."""
    w = _real(v)
    if not (w.is_integer or (w.is_Float and float(w).is_integer())):  # Float(12.0) != Integer(12)
        raise ToolError("INVALID_ARGUMENT", f"{what} must be an integer, not {w}.", example)
    return int(w)


def _round(x: Any, n: Any = 0) -> Any:
    v = _real(x)
    nd = _whole(n, "round: the number of decimals", "round(2.675, 2)")
    d, _ = to_decimal(v, max(nd, 0) + 25)
    q = d.quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)
    fr = Fraction(q)
    return sp.Rational(fr.numerator, fr.denominator)


def _mean(*a: Any) -> Any:
    d = _seq(a)
    return sp.Add(*d) / len(d)


def _median(*a: Any) -> Any:
    d = sorted(_seq(a), key=lambda v: float(v))
    n = len(d)
    return d[n // 2] if n % 2 else (d[n // 2 - 1] + d[n // 2]) / 2


def _mode(*a: Any) -> Any:
    d = _seq(a)
    counts: dict[Any, int] = {}
    for v in d:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _var(d: list[Any], sample: bool) -> Any:
    n = len(d)
    if sample and n < 2:
        raise ToolError("MATH_ERROR", "At least 2 values are needed for the sample variance.")
    m = sp.Add(*d) / n
    return sp.Add(*[(v - m) ** 2 for v in d]) / (n - 1 if sample else n)


def _variance(*a: Any) -> Any:
    return _var(_seq(a), True)


def _pvariance(*a: Any) -> Any:
    return _var(_seq(a), False)


def _stdev(*a: Any) -> Any:
    return sp.sqrt(_var(_seq(a), True))


def _pstdev(*a: Any) -> Any:
    return sp.sqrt(_var(_seq(a), False))


def _cov(x: Any, y: Any) -> Any:
    xs, ys = _seq((x,)), _seq((y,))
    if len(xs) != len(ys) or len(xs) < 2:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Two lists of the same length (at least 2) are needed.",
            "cov([1,2,3],[2,4,7])",
        )
    mx, my = sp.Add(*xs) / len(xs), sp.Add(*ys) / len(ys)
    return sp.Add(*[(a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)]) / (len(xs) - 1)


def _pearson(x: Any, y: Any) -> Any:
    return _cov(x, y) / (sp.sqrt(_var(_seq((x,)), True)) * sp.sqrt(_var(_seq((y,)), True)))


def _norm_pdf(x: Any, mu: Any = 0, s: Any = 1) -> Any:
    return sp.exp(-((x - mu) ** 2) / (2 * s**2)) / (s * sp.sqrt(2 * sp.pi))


def _norm_cdf(x: Any, mu: Any = 0, s: Any = 1) -> Any:
    return (1 + sp.erf((x - mu) / (s * sp.sqrt(2)))) / 2


def _norm_ppf(p: Any, mu: Any = 0, s: Any = 1) -> Any:
    if not (0 < p < 1):
        raise ToolError("MATH_ERROR", "The probability must be between 0 and 1, excluded.")
    return mu + s * sp.sqrt(2) * sp.erfinv(2 * p - 1)


def _lognorm_pdf(x: Any, mu: Any = 0, s: Any = 1) -> Any:
    return sp.exp(-((sp.log(x) - mu) ** 2) / (2 * s**2)) / (x * s * sp.sqrt(2 * sp.pi))


def _lognorm_cdf(x: Any, mu: Any = 0, s: Any = 1) -> Any:
    return (1 + sp.erf((sp.log(x) - mu) / (s * sp.sqrt(2)))) / 2


def _scipy_dist(name: str, fn: str) -> Any:
    def f(x: Any, df: Any) -> Any:
        from scipy import stats

        dist = getattr(stats, name)
        return sp.Float(float(getattr(dist, fn)(float(x), float(df))), 17)

    return f


def _rate(lam: Any) -> Any:
    v = _real(lam)
    if v.is_negative:
        raise ToolError(
            "INVALID_ARGUMENT", f"The Poisson mean must be >= 0, not {v}.", "poisson_pmf(3, 2.5)"
        )
    return v


def _poisson_pmf(k: Any, lam: Any) -> Any:
    # D91: k is a count (2.5 gave gamma(3.5) in the denominator); below 0 the probability is 0
    kk, lv = _whole(k, "poisson_pmf: k", "poisson_pmf(3, 2.5)"), _rate(lam)
    return lv**kk * sp.exp(-lv) / sp.factorial(kk) if kk >= 0 else sp.Integer(0)


def _poisson_cdf(k: Any, lam: Any) -> Any:
    # P(X <= k) for a real k: the sum up to floor(k); 0 below 0 (int(-0.5) gave P(X = 0))
    top, lv = int(sp.floor(_real(k))), _rate(lam)
    return sp.Add(*[lv**i * sp.exp(-lv) / sp.factorial(i) for i in range(top + 1)])  # empty sum = 0


def _binom_args(n: Any, p: Any, fn: str) -> tuple[int, Any]:
    nn = _whole(n, f"{fn}: n", f"{fn}(2, 10, 0.3)")
    pv = _real(p)
    if nn < 0 or pv.is_negative or (pv - 1).is_positive:
        raise ToolError(
            "INVALID_ARGUMENT", f"{fn}: n >= 0 and 0 <= p <= 1 are needed.", f"{fn}(2, 10, 0.3)"
        )
    return nn, pv


def _binom_pmf(k: Any, n: Any, p: Any) -> Any:
    # D91: k and n are counts (1.5 gave a gamma-based binomial); outside 0..n the probability is 0
    kk = _whole(k, "binom_pmf: k", "binom_pmf(2, 10, 0.3)")
    nn, pv = _binom_args(n, p, "binom_pmf")
    return sp.binomial(nn, kk) * pv**kk * (1 - pv) ** (nn - kk) if 0 <= kk <= nn else sp.Integer(0)


def _binom_cdf(k: Any, n: Any, p: Any) -> Any:
    nn, pv = _binom_args(n, p, "binom_cdf")
    top = min(int(sp.floor(_real(k))), nn)
    return sp.Add(*[sp.binomial(nn, i) * pv**i * (1 - pv) ** (nn - i) for i in range(top + 1)])


def _pmt(rate: Any, nper: Any, pv: Any, fv: Any = 0) -> Any:
    if rate == 0:
        return (pv + fv) / nper
    g = (1 + rate) ** nper
    return (pv * g + fv) * rate / (g - 1)


def _npv(rate: Any, flows: Any) -> Any:
    fl = _seq((flows,))
    return sp.Add(*[c / (1 + rate) ** t for t, c in enumerate(fl)])


def _irr(flows: Any) -> Any:
    fl = [float(v) for v in _seq((flows,))]
    mpmath.mp.dps = 50

    def f(r: Any) -> Any:
        return mpmath.fsum(c / (1 + r) ** t for t, c in enumerate(fl))

    try:
        root = mpmath.findroot(f, 0.1)
    except (ValueError, ZeroDivisionError):
        raise ToolError(
            "MATH_ERROR", "IRR not found (no sign change in the cash flows?)."
        ) from None
    return sp.Float(str(root), 40)


def _dcf(rate: Any, flows: Any, g: Any = None) -> Any:
    fl = _seq((flows,))
    pv = sp.Add(*[c / (1 + rate) ** (t + 1) for t, c in enumerate(fl)])
    if g is not None:
        if g >= rate:
            raise ToolError("MATH_ERROR", "Growth g must be less than the rate.")
        tv = fl[-1] * (1 + g) / (rate - g)
        pv += tv / (1 + rate) ** len(fl)
    return pv


def _sind(x: Any) -> Any:
    return sp.sin(x * sp.pi / 180)


def _cosd(x: Any) -> Any:
    return sp.cos(x * sp.pi / 180)


def _tand(x: Any) -> Any:
    return sp.tan(x * sp.pi / 180)


def _mod_inv(a: Any, m: Any) -> Any:
    try:
        return sp.Integer(
            pow(
                _whole(a, "mod_inv: a", "mod_inv(3, 7)"),
                -1,
                _whole(m, "mod_inv: m", "mod_inv(3, 7)"),
            )
        )
    except ValueError:
        raise ToolError("MATH_ERROR", f"{a} is not invertible modulo {m}.") from None


def _gcd(*a: Any) -> Any:
    d = _seq(a)
    w = [_whole(v, "gcd", "gcd(12, 18)") for v in d]  # D91: int() truncated 12.5
    return sp.Integer(sp.igcd(*w)) if len(w) > 1 else sp.Integer(w[0])


def _lcm(*a: Any) -> Any:
    d = _seq(a)
    w = [_whole(v, "lcm", "lcm(4, 6)") for v in d]
    return sp.Integer(sp.ilcm(*w)) if len(w) > 1 else sp.Integer(w[0])


FUNCS: dict[str, Any] = {
    k: v for k, v in SOLVE_FUNCS.items() if k not in {"diff", "gcd", "lcm", "min", "max"}
}
FUNCS.update(
    {
        "round": _round,
        "min": lambda *a: sp.Min(*_seq(a)),
        "max": lambda *a: sp.Max(*_seq(a)),
        "sum": lambda *a: sp.Add(*_seq(a)),
        "prod": lambda *a: sp.Mul(*_seq(a)),
        "sind": _sind,
        "cosd": _cosd,
        "tand": _tand,
        "asind": lambda x: sp.asin(x) * 180 / sp.pi,
        "acosd": lambda x: sp.acos(x) * 180 / sp.pi,
        "atand": lambda x: sp.atan(x) * 180 / sp.pi,
        "deg": lambda x: x * 180 / sp.pi,
        "rad": lambda x: x * sp.pi / 180,
        "perm": lambda n, k: sp.ff(n, k),
        "gcd": _gcd,
        "lcm": _lcm,
        "mod_inv": _mod_inv,
        "mod": sp.Mod,
        "mean": _mean,
        "median": _median,
        "mode": _mode,
        "stdev": _stdev,
        "pstdev": _pstdev,
        "variance": _variance,
        "pvariance": _pvariance,
        "cov": _cov,
        "pearson": _pearson,
        "norm_pdf": _norm_pdf,
        "norm_cdf": _norm_cdf,
        "norm_ppf": _norm_ppf,
        "lognorm_pdf": _lognorm_pdf,
        "lognorm_cdf": _lognorm_cdf,
        "t_cdf": _scipy_dist("t", "cdf"),
        "t_ppf": _scipy_dist("t", "ppf"),
        "chi2_cdf": _scipy_dist("chi2", "cdf"),
        "chi2_ppf": _scipy_dist("chi2", "ppf"),
        "poisson_pmf": _poisson_pmf,
        "poisson_cdf": _poisson_cdf,
        "binom_pmf": _binom_pmf,
        "binom_cdf": _binom_cdf,
        "pmt": _pmt,
        "npv": _npv,
        "irr": _irr,
        "dcf": _dcf,
        "vat": lambda net, rate=_VAT_RATE: net * rate / 100,
        "vat_add": lambda net, rate=_VAT_RATE: net * (1 + rate / 100),
        "vat_remove": lambda gross, rate=_VAT_RATE: gross / (1 + rate / 100),
        "withholding": lambda net, rate=_WITHHOLDING_RATE: net * rate / 100,
        "unit_convert": unit_convert,
        "const": _const_value,
    }
)


def _value_from_json(raw: Any) -> Any:
    if isinstance(raw, bool):
        raise ToolError("INVALID_ARGUMENT", "Boolean value not allowed in variables.", '{"x": 2}')
    if isinstance(raw, int):
        return sp.Integer(raw)
    if isinstance(raw, float):
        fr = Fraction(repr(raw))
        return sp.Rational(fr.numerator, fr.denominator)
    if isinstance(raw, str):
        return parse(raw, Grammar(FUNCS, {}, free_symbols=False))
    raise ToolError("INVALID_ARGUMENT", f"Invalid value in variables: {raw!r}.", '{"x": 2}')


def _fraction_eval(node: ast.AST, values: dict[str, Fraction]) -> Fraction | None:
    """Independent check with fractions.Fraction (only + - * / and integer powers)."""
    if isinstance(node, ast.Expression):
        return _fraction_eval(node.body, values)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        return Fraction(repr(node.value)) if isinstance(node.value, float) else Fraction(node.value)
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        v = _fraction_eval(node.operand, values)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp):
        a, b = _fraction_eval(node.left, values), _fraction_eval(node.right, values)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div) and b != 0:
            return a / b
        if isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 64:
            return a ** int(b)
    return None


def _steps(ctx: Ctx, src: str, value: Any, variables: dict[str, Any]) -> None:
    ctx.step(f"Expression read: {src}", src)
    if variables:
        ctx.step(
            f"Values of the variables: {', '.join(f'{k} = {v}' for k, v in variables.items())}."
        )
    try:
        tree = ast.parse(src.replace("^", "**"), mode="eval")
    except SyntaxError:
        tree = None
    if tree is not None:
        b = tree.body
        if isinstance(b, ast.BinOp) and isinstance(b.op, ast.Add | ast.Sub):
            fl = _fraction_eval(b.left, {}), _fraction_eval(b.right, {})
            if all(f is not None and f.denominator != 1 for f in fl):
                a, c = fl
                assert a is not None and c is not None
                den = math.lcm(a.denominator, c.denominator)
                ctx.step(f"Common denominator: lcm({a.denominator}, {c.denominator}) = {den}.", den)
                ctx.step(
                    f"Equivalent fractions: {a} = {a.numerator * (den // a.denominator)}/{den}, "
                    f"{c} = {c.numerator * (den // c.denominator)}/{den}.",
                )
        ctx.step(f"Exact computation with SymPy: {value}.", value)
        vals = {k: Fraction(str(v)) for k, v in variables.items() if isinstance(v, sp.Rational)}
        indep = _fraction_eval(tree, vals) if tree is not None else None
        if indep is not None and value.is_Rational:
            same = Fraction(int(value.p), int(value.q)) == indep
            ctx.step(
                f"Back-check: independent recomputation with fractions = {indep} "
                f"({'matches' if same else 'does not match'}).",
                indep,
            )
            return
    approx = sp.N(value, 30)
    ctx.step(f"Numerical check with 30 digits: {approx}.", approx)


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    src: str = args["expr"]
    mode: str = args.get("exact_mode", "decimal")
    places: int = args.get("precision", 6)
    raw_vars: dict[str, Any] = args.get("variables") or {}
    if len(raw_vars) > limits.LIST_MAX:
        raise ToolError("LIMIT_EXCEEDED", "Too many variables.")
    declared = {check_identifier(k): _value_from_json(v) for k, v in raw_vars.items()}
    grammar = Grammar(
        FUNCS, declared, free_symbols=False, string_args=frozenset({"unit_convert", "const"})
    )
    value = parse(src, grammar)
    if isinstance(value, list | str):
        raise ToolError("PARSE_ERROR", "The result must be a number.", "mean([1, 2, 3])")
    value = cast(sp.Expr, sp.sympify(value) if not isinstance(value, sp.Basic) else value)
    if isinstance(value, sp.core.relational.Relational | sp.logic.boolalg.Boolean):
        raise ToolError(
            "PARSE_ERROR", "math_eval computes values, not comparisons: use math_verify.", "2**10"
        )
    if value.free_symbols:
        names = ", ".join(sorted(str(s) for s in value.free_symbols))
        raise ToolError(
            "INVALID_ARGUMENT", f"Variables without a value: {names}.", '{"variables": {"x": 2}}'
        )
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ToolError("MATH_ERROR", "Non-finite result (division by zero or function domain).")
    if value.is_real is False or (value.is_real is None and sp.im(sp.N(value)) != 0):
        re_v, im_v = sp.re(value), sp.im(value)
        rd, _ = to_decimal(sp.nsimplify(sp.N(re_v, places + 20)), places)
        idd, _ = to_decimal(sp.nsimplify(sp.N(im_v, places + 20)), places)
        text = f"{dec_str(rd)} + {dec_str(idd)}*I"
        ctx.warn("Complex result.")
        ctx.formatted = f"Result: {text}"
        return {"value": text, "exact": str(value), "float": None, "is_exact": False, "mode": mode}
    if ctx.show_steps:
        _steps(ctx, src, value, declared)
    exact_txt = str(value)
    flt: float | None
    try:
        flt = float(value)
    except (TypeError, OverflowError):
        flt = None
    if mode == "float":
        if flt is None:
            raise ToolError(
                "MATH_ERROR", "Value too large for floating point: use exact_mode decimal."
            )
        text = repr(flt)
        is_exact = bool(value.is_Rational and sp.Rational(Fraction(flt)) == value)
    elif mode == "fraction":
        if value.is_Rational:
            text = str(value)
            is_exact = True
        else:
            d, _ = to_decimal(value, places)
            text = dec_str(d)
            is_exact = False
            ctx.warn(
                f"Irrational result ({exact_txt}): the fraction does not exist, decimal value with {places} digits."
            )
    else:
        d, is_exact = to_decimal(value, places)
        text = dec_str(d)
        if not is_exact:
            ctx.warn(f"Value rounded to {places} decimal places (ROUND_HALF_UP).")
    ctx.formatted = f"Result: {en_number(text)}" + (
        "" if is_exact else f" (exact form: {exact_txt})"
    )
    return {"value": text, "exact": exact_txt, "float": flt, "is_exact": is_exact, "mode": mode}
