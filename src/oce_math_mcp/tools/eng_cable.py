"""eng_calc: catenary cables and journal bearings (D58).

Sources (Wikipedia pages):
- "Catenary": y = a cosh(x / a) with a = T0 / w (T0 horizontal tension, w weight per metre); T = w y;
  supports at horizontal distance H and height difference v with a cable of length L:
  L^2 - v^2 = 4 a^2 sinh^2(H / (2a)); supports at the same height with sag h: H = 2 a arcosh((h + a) / a);
- "Sommerfeld number": S = (r / c)^2 mu N / P (N in revolutions per second, P = W / (2 r l)); with the
  angular velocity S changes by a factor 2 pi; Petroff's law for a centred, lightly loaded journal:
  T = 4 pi^2 r^3 l mu N / c, T = f W r, f = 2 pi^2 (mu N / P) (r / c).
Derived here: a from the relations of the source with brentq; position of the vertex for supports at
different heights from v = 2 a sinh((x1 + x2) / (2a)) sinh(H / (2a)) (difference of the cosh), angles at the supports.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

from scipy.optimize import brentq

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import acosh1p, choice, fmt_sig, need, only, op, units_for

_CAT_MODES: dict[str, set[str]] = {
    "span_length": {"w", "H", "L", "v"},
    "span_sag": {"w", "H", "sag"},
    "span_tension": {"w", "H", "T0"},
}
_CAT_UNITS = {
    "a": "m",
    "T0": "N",
    "T_max": "N",
    "T_supports": "N",
    "sag": "m",
    "L": "m",
    "x_supports": "m",
    "y_supports": "m",
    "angle_support_deg": "deg",
}


def _sinhc_m1(u: float) -> float:
    """sinh(u) / u - 1 for u > 0, by its series for small u: no cancellation for a taut cable (D82)."""
    if u > 0.5:
        return math.sinh(u) / u - 1
    q = u * u
    s = 0.0
    for k in (272, 210, 156, 110, 72, 42, 20):  # (2k + 2)(2k + 3), innermost term first
        s = q / k * (1 + s)
    return q / 6 * (1 + s)


def _root(f: Callable[[float], float], lo: float, hi: float) -> float:
    return cast(float, cast(Any, brentq)(f, lo, hi, xtol=1e-300, rtol=1e-15, maxiter=500))


def _bracket(f: Callable[[float], float], lo: float, hi: float) -> float:
    """Root of f, decreasing or increasing in a: widens hi until the sign changes."""
    flo = f(lo)
    for _ in range(400):
        if (f(hi) > 0) != (flo > 0):
            return _root(f, lo, hi)
        hi *= 2
    raise ToolError("INVALID_ARGUMENT", "Catenary: no solution for these data.")  # pragma: no cover


@op("cable_catenary", ("mode", *sorted(set().union(*_CAT_MODES.values()))))
def _catenary(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_CAT_MODES))
    only(p, mode, _CAT_MODES[mode])
    w = need(p, "w", "line_load")
    span = need(p, "H", "length")
    v = 0.0
    if mode == "span_length":
        length = need(p, "L", "length")
        v = need(p, "v", "length", positive=False) if p.get("v") is not None else 0.0
        if abs(v) >= length:
            raise ToolError(
                "INVALID_ARGUMENT",
                "The height difference |v| must be smaller than the cable length L.",
            )
        # sqrt(L^2 - v^2) - H with L^2 - v^2 - H^2 in exact rationals: the taut cable keeps its digits (D82)
        gap = Fraction(length) ** 2 - Fraction(v) ** 2 - Fraction(span) ** 2
        if gap <= 0:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Cable too short: sqrt(L^2 - v^2) > H is needed (for supports at the same height L > H).",
            )
        # sqrt(L - v) sqrt(L + v) and the excess as one exact quotient: no overflow or underflow of L^2 (D84)
        target = math.sqrt(length - v) * math.sqrt(length + v)
        excess = float(gap / (Fraction(target) + Fraction(span)))

        # 2 a sinh(H / (2a)) decreases from infinity to H as a grows
        # written as H (sinh(u)/u - 1) - excess with u = H / (2a): no cancellation for a taut cable (D82)
        def f_len(a: float) -> float:
            return span * _sinhc_m1(span / (2 * a)) - excess

        a = _bracket(f_len, span / 700, span)  # a = H/700: sinh(350) is still finite
        ctx.formula = (
            "L^2 - v^2 = 4 a^2 sinh^2(H / (2a)) solved for a (brentq); y = a cosh(x / a); T = w y"
        )
    elif mode == "span_sag":
        sag = need(p, "sag", "length")

        def f_sag(a: float) -> float:
            # acosh((h + a) / a) without rounding 1 + h/a (D82)
            return 2 * a * acosh1p(sag / a) - span

        a = _bracket(f_sag, sag * 1e-12, span)
        ctx.formula = (
            "H = 2 a arcosh((h + a) / a) solved for a (brentq); y = a cosh(x / a); T = w y"
        )
    else:
        a = need(p, "T0", "force") / w
        ctx.formula = "a = T0 / w; y = a cosh(x / a); T = w y"
    sh = math.sinh(span / (2 * a))
    mid = a * math.asinh(v / (2 * a * sh))  # (x1 + x2) / 2
    x1, x2 = mid - span / 2, mid + span / 2
    y1, y2 = a * math.cosh(x1 / a), a * math.cosh(x2 / a)
    # heights over the vertex a (cosh(x/a) - 1) = 2 a sinh^2(x/(2a)) and length 2 a cosh(mid/a) sinh(H/(2a)):
    # no cancellation when a is large (taut cable, D82)
    h1, h2 = 2 * a * math.sinh(x1 / (2 * a)) ** 2, 2 * a * math.sinh(x2 / (2 * a)) ** 2
    length_out = 2 * a * math.cosh(mid / a) * sh
    t1, t2 = w * y1, w * y2
    out: dict[str, Any] = {
        "a": a,
        "T0": w * a,
        "L": length_out,
        "x_supports": [x1, x2],
        "y_supports": [h1, h2],
        "T_supports": [t1, t2],
        "T_max": max(t1, t2),
        "angle_support_deg": [
            math.degrees(math.atan(math.sinh(x1 / a))),
            math.degrees(math.atan(math.sinh(x2 / a))),
        ],
    }
    if v == 0:
        out["sag"] = h1
    else:
        out["sag"] = min(h1, h2)
        if x1 > 0 or x2 < 0:
            ctx.warn(
                "The vertex of the catenary falls outside the span: the cable rises from one support to the other."
            )
    ctx.warn(
        "Perfectly flexible and inextensible cable, uniform weight per metre (model of the source)."
    )
    ctx.units = units_for(out, _CAT_UNITS)
    ctx.formatted = (
        f"Parameter a = {fmt_sig(a, 6)} m, horizontal tension {fmt_sig(out['T0'], 6)} N, maximum {fmt_sig(out['T_max'], 6)} N, "
        f"sag {fmt_sig(out['sag'], 6)} m, length {fmt_sig(length_out, 6)} m."
    )
    return out


# ------------------------------------------------------------------ journal_bearing

_BRG_KEYS = {"D", "r", "c", "l", "mu", "N", "rpm", "W", "P"}
_BRG_UNITS = {
    "r": "m",
    "N": "1/s",
    "omega": "rad/s",
    "P": "Pa",
    "S": "1",
    "S_omega": "1",
    "hersey": "1",
    "f": "1",
    "T_friction": "N*m",
    "P_loss": "W",
}


@op("journal_bearing", tuple(sorted(_BRG_KEYS)))
def _bearing(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    only(p, "journal_bearing", _BRG_KEYS, field="operation")
    if (p.get("D") is None) == (p.get("r") is None):
        raise ToolError("INVALID_ARGUMENT", "Give only one of D (journal diameter) and r (radius).")
    r = need(p, "r", "length") if p.get("r") is not None else need(p, "D", "length") / 2
    c = need(p, "c", "length")
    if c >= r:
        raise ToolError(
            "INVALID_ARGUMENT", "The radial clearance c must be much smaller than the radius r."
        )
    length = need(p, "l", "length")
    mu = need(p, "mu", "dynamic_viscosity")
    if (p.get("N") is None) == (p.get("rpm") is None):
        raise ToolError("INVALID_ARGUMENT", "Give only one of N (revolutions per second) and rpm.")
    n = need(p, "N", None) if p.get("N") is not None else need(p, "rpm", None) / 60
    if (p.get("W") is None) == (p.get("P") is None):
        raise ToolError(
            "INVALID_ARGUMENT", "Give only one of W (radial load) and P (pressure W / (2 r l))."
        )
    press = (
        need(p, "P", "pressure")
        if p.get("P") is not None
        else need(p, "W", "force") / (2 * r * length)
    )
    load = press * 2 * r * length
    hersey = mu * n / press
    s = (r / c) ** 2 * hersey
    torque = 4 * math.pi**2 * r**3 * length * mu * n / c
    f = 2 * math.pi**2 * hersey * (r / c)
    out: dict[str, Any] = {
        "r": r,
        "N": n,
        "omega": 2 * math.pi * n,
        "P": press,
        "hersey": hersey,
        "S": s,
        "S_omega": 2 * math.pi * s,
        "f": f,
        "T_friction": torque,
        "P_loss": torque * 2 * math.pi * n,
    }
    ctx.formula = (
        "S = (r / c)^2 mu N / P, P = W / (2 r l), N in rev/s (with w in rad/s the value is 2 pi times larger); "
        "Petroff: T = 4 pi^2 r^3 l mu N / c = f W r, f = 2 pi^2 (mu N / P) (r / c)"
    )
    ctx.warn(
        "Petroff: centred, lightly loaded journal, clearance full of lubricant; under load the hydrodynamic theory is needed."
    )
    ctx.formatted = (
        f"Sommerfeld S = {fmt_sig(s, 5)}, friction coefficient {fmt_sig(f, 5)}, torque {fmt_sig(torque, 5)} N m, "
        f"power loss {fmt_sig(out['P_loss'], 5)} W (load {fmt_sig(load, 5)} N)."
    )
    ctx.units = units_for(out, _BRG_UNITS)
    return out
