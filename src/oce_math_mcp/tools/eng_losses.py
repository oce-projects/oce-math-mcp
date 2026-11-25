"""eng_calc: local losses at sudden changes of section, Borda-Carnot equation (D70).

Source: Wikipedia page "Borda-Carnot equation" in the local offline library.
- Delta E = xi (rho / 2) (v1 - v2)^2, 0 <= xi <= 1, valid only for decreasing velocity (v1 > v2), otherwise the loss is
  zero; total head loss Delta H = Delta E / (rho g).
- Sudden expansion (xi = 1): Delta E = (rho / 2) (1 - A1/A2)^2 v1^2,
  Delta p = p1 - p2 = -rho (A1/A2) (1 - A1/A2) v1^2, Delta h = Delta p / (rho g) (horizontal pipe).
- Sudden contraction: vena contracta A3 = mu A2, Delta E = (rho / 2) (1/mu - 1)^2 v2^2; Weisbach for a sharp-edged
  contraction: mu = 0.63 + 0.37 (A2/A1)^3.
Derived here and checked in the tests: loss coefficients K_1 and K_2 referred to v1 and v2 (Delta E = K rho v^2 / 2),
pressure change for any xi from the energy balance of a horizontal pipe, Delta p = Delta E + (rho / 2) (v2^2 - v1^2)
(equal to the page formula when xi = 1).
"""

from __future__ import annotations

import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, one_of, only, op, qty, units_for
from .eng_thermo import G0

_SECTIONS = {"D1", "A1", "D2", "A2", "Q", "v1", "rho", "g"}
_MODES: dict[str, set[str]] = {
    "sudden_expansion": _SECTIONS | {"xi"},
    "sudden_contraction": _SECTIONS | {"mu"},
    "borda_carnot": {"xi", "v1", "v2", "rho", "g"},
}
_UNITS = {
    "A1": "m2",
    "A2": "m2",
    "A3": "m2",
    "Q": "m3/s",
    "v1": "m/s",
    "v2": "m/s",
    "v3": "m/s",
    "dE": "Pa",
    "dH": "m",
    "dp": "Pa",
    "dh": "m",
}


@op("local_loss", ("mode", *sorted(set().union(*_MODES.values()))))
def _local_loss(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    rho = need(p, "rho", "density")
    g = qty(p, "g", "acceleration", default=G0) or G0
    out = (
        _general(p, rho, g, ctx)
        if mode == "borda_carnot"
        else _section_change(p, mode, rho, g, ctx)
    )
    ctx.units = units_for(out, _UNITS)
    return out


def _xi(p: dict[str, Any]) -> float:
    xi = qty(p, "xi", None, positive=False, default=1.0)
    assert xi is not None
    if not 0 <= xi <= 1:
        raise ToolError("INVALID_ARGUMENT", "params.xi (loss coefficient) must be in [0, 1].")
    return xi


def _area(p: dict[str, Any], n: str) -> float:
    if one_of(p, (f"D{n}", f"A{n}"), f"section {n} (diameter D{n} or area A{n})") == f"D{n}":
        d = need(p, f"D{n}", "length")
        return math.pi * d * d / 4
    return need(p, f"A{n}", "area")


def _general(p: dict[str, Any], rho: float, g: float, ctx: Ctx) -> dict[str, Any]:
    xi = _xi(p)
    v1, v2 = need(p, "v1", "velocity"), need(p, "v2", "velocity")
    if v1 > v2:
        de = xi * rho / 2 * (v1 - v2) ** 2
    else:
        de = 0.0
        ctx.warn(
            "v1 <= v2: the Borda-Carnot loss holds only for decreasing velocity; the loss is zero."
        )
    ctx.formula = "Delta E = xi (rho / 2) (v1 - v2)^2 (v1 > v2); Delta H = Delta E / (rho g)"
    ctx.formatted = (
        f"Energy loss {fmt_sig(de, 6)} Pa (J/m3), head loss {fmt_sig(de / (rho * g), 6)} m."
    )
    return {"dE": de, "dH": de / (rho * g)}


def _section_change(p: dict[str, Any], mode: str, rho: float, g: float, ctx: Ctx) -> dict[str, Any]:
    a1, a2 = _area(p, "1"), _area(p, "2")
    if one_of(p, ("Q", "v1"), "the flow rate (Q) or the velocity in section 1 (v1)") == "Q":
        q = need(p, "Q", "flow_rate")
        v1 = q / a1
    else:
        v1 = need(p, "v1", "velocity")
        q = v1 * a1
    v2 = q / a2
    out: dict[str, Any] = {"A1": a1, "A2": a2, "Q": q, "v1": v1, "v2": v2}
    if mode == "sudden_expansion":
        if a2 <= a1:
            raise ToolError(
                "INVALID_ARGUMENT", "A2 <= A1 is not an expansion: use mode sudden_contraction."
            )
        xi = _xi(p)
        k2 = xi * (a2 / a1 - 1) ** 2
        out["xi"] = xi
        ctx.formula = (
            "Borda-Carnot: Delta E = xi (rho / 2) (v1 - v2)^2 = xi (rho / 2) (1 - A1/A2)^2 v1^2; "
            "Delta p = p1 - p2 = Delta E + (rho / 2) (v2^2 - v1^2)"
        )
    else:
        if a2 >= a1:
            raise ToolError(
                "INVALID_ARGUMENT", "A2 >= A1 is not a contraction: use mode sudden_expansion."
            )
        mu = qty(p, "mu", None, default=0.63 + 0.37 * (a2 / a1) ** 3)
        assert mu is not None
        if mu > 1:
            raise ToolError(
                "INVALID_ARGUMENT", "params.mu (contraction coefficient A3/A2) must be in (0, 1]."
            )
        k2 = (1 / mu - 1) ** 2
        out |= {"mu": mu, "A3": mu * a2, "v3": v2 / mu}
        ctx.formula = (
            "vena contracta A3 = mu A2, mu = 0.63 + 0.37 (A2/A1)^3 (Weisbach, sharp edge) unless given; "
            "Delta E = (rho / 2) (1/mu - 1)^2 v2^2; Delta p = p1 - p2 = Delta E + (rho / 2) (v2^2 - v1^2)"
        )
    de = k2 * rho * v2 * v2 / 2
    dp = de + rho / 2 * (v2 * v2 - v1 * v1)
    out |= {
        "K_1": k2 * (a1 / a2) ** 2,
        "K_2": k2,
        "dE": de,
        "dH": de / (rho * g),
        "dp": dp,
        "dh": dp / (rho * g),
    }
    ctx.warn("Horizontal pipe: dp and dh do not include a change of elevation.")
    ctx.formatted = (
        f"{mode.replace('_', ' ').capitalize()}: head loss {fmt_sig(out['dH'], 6)} m, "
        f"p1 - p2 = {fmt_sig(dp, 6)} Pa."
    )
    return out
