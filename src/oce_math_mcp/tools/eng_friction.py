"""eng_calc: friction in simple machines (D66).

Sources: Wikipedia pages in the local offline library.
- "Inclined plane": force parallel to the plane F_i = F_w (sin(theta) + mu cos(theta)) uphill and
  F_w (sin(theta) - mu cos(theta)) downhill, MA = F_w / F_i, angle of repose phi = atan(mu), frictionless MA = 1/sin(theta).
- "Leadscrew": T_raise = (F d_m/2)(l + pi mu d_m sec(alpha))/(pi d_m - mu l sec(alpha)),
  T_lower = (F d_m/2)(pi mu d_m sec(alpha) - l)/(pi d_m + mu l sec(alpha)), alpha = half the thread angle,
  efficiency (cos(alpha) - mu tan(lambda))/(cos(alpha) + mu cot(lambda)), self-locking when friction holds the load,
  collar T_c = F mu_c d_c / 2.
- "Rolling resistance": F = C_rr N, F = N b / r.
Derived here and checked in the tests: horizontal push on the plane F_w tan(theta +- phi) (equilibrium of the block),
efficiency of the plane (MA with friction over MA without), total torque T_raise + T_c.
"""

from __future__ import annotations

import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, only, op, qty, units_for

_MODES: dict[str, set[str]] = {
    "inclined_plane": {"F_w", "theta_deg", "mu", "push"},
    "leadscrew": {"F", "d_m", "l", "mu", "alpha_deg", "mu_c", "d_c"},
    "rolling_resistance": {"N", "C_rr", "b", "r"},
}
_UNITS = {
    "F_i_up": "N",
    "F_i_down": "N",
    "angle_of_repose_deg": "deg",
    "T_raise": "N*m",
    "T_lower": "N*m",
    "T_c": "N*m",
    "T_raise_total": "N*m",
    "lead_angle_deg": "deg",
    "F_r": "N",
}


@op("friction", ("mode", *sorted(set().union(*_MODES.values()))))
def _friction(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = {"inclined_plane": _plane, "leadscrew": _screw, "rolling_resistance": _rolling}[mode](
        p, ctx
    )
    ctx.units = units_for(out, _UNITS)
    return out


def _mu(p: dict[str, Any], key: str = "mu") -> float:
    v = need(p, key, None, positive=False)
    if v < 0:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be >= 0.")
    return v


def _plane(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    w = need(p, "F_w", "force")
    theta_deg = need(p, "theta_deg", None)
    if theta_deg >= 90:
        raise ToolError("INVALID_ARGUMENT", "params.theta_deg must be between 0 and 90 degrees.")
    mu = _mu(p)
    push = choice(p, "push", ("parallel", "horizontal"), "parallel")
    a = math.radians(theta_deg)
    phi = math.atan(mu)
    if push == "parallel":
        f_up = w * (math.sin(a) + mu * math.cos(a))
        f_down = w * (math.sin(a) - mu * math.cos(a))
        ideal = w * math.sin(a)
        ctx.formula = (
            "F_i = F_w (sin(theta) + mu cos(theta)) uphill, F_w (sin(theta) - mu cos(theta)) downhill "
            "(force parallel to the plane)"
        )
    else:
        if a + phi >= math.pi / 2:
            raise ToolError(
                "INVALID_ARGUMENT",
                "theta + atan(mu) >= 90 degrees: a horizontal push cannot move the load up.",
            )
        f_up = w * math.tan(a + phi)
        f_down = w * math.tan(a - phi)
        ideal = w * math.tan(a)
        ctx.formula = "F_i = F_w tan(theta + phi) uphill, F_w tan(theta - phi) downhill; phi = atan(mu) (horizontal force)"
    out: dict[str, Any] = {"F_i_up": f_up, "F_i_down": f_down, "MA_up": w / f_up}
    out["efficiency_up"] = ideal / f_up
    out["angle_of_repose_deg"] = math.degrees(phi)
    out["self_locking"] = a <= phi
    ctx.formatted = (
        f"Force to move the load up {fmt_sig(f_up, 6)} N; down {fmt_sig(f_down, 6)} N "
        f"({'it holds by friction' if a <= phi else 'it slides by itself'})."
    )
    ctx.warn(
        "F_i_down < 0: a downhill push is needed to move the load; > 0: the load must be held back."
    )
    return out


def _screw(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    f = need(p, "F", "force")
    d_m, lead = need(p, "d_m", "length"), need(p, "l", "length")
    mu = _mu(p)
    alpha = math.radians(qty(p, "alpha_deg", None, positive=False, default=0.0) or 0.0)
    if not 0 <= alpha < math.pi / 2:
        raise ToolError(
            "INVALID_ARGUMENT", "params.alpha_deg (thread half-angle) must be in [0, 90) degrees."
        )
    k = mu / math.cos(alpha)
    den_r = math.pi * d_m - k * lead
    if den_r <= 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            "pi d_m <= mu l sec(alpha): the screw cannot be turned to raise the load.",
        )
    t_raise = f * d_m / 2 * (lead + math.pi * k * d_m) / den_r
    t_lower = f * d_m / 2 * (math.pi * k * d_m - lead) / (math.pi * d_m + k * lead)
    ctx.formula = (
        "T_raise = F d_m/2 (l + pi mu d_m sec a)/(pi d_m - mu l sec a); "
        "T_lower = F d_m/2 (pi mu d_m sec a - l)/(pi d_m + mu l sec a)"
    )
    out: dict[str, Any] = {"T_raise": t_raise, "T_lower": t_lower}
    out["lead_angle_deg"] = math.degrees(math.atan(lead / (math.pi * d_m)))
    if p.get("mu_c") is not None or p.get("d_c") is not None:
        t_c = f * _mu(p, "mu_c") * need(p, "d_c", "length") / 2
        out["T_c"] = t_c
        out["T_raise_total"] = t_raise + t_c
        ctx.formula += "; collar T_c = F mu_c d_c / 2"
    lam = math.atan(lead / (math.pi * d_m))
    out["efficiency"] = (math.cos(alpha) - mu * math.tan(lam)) / (
        math.cos(alpha) + mu / math.tan(lam)
    )
    if alpha > 0:
        ctx.warn(
            "The thread-angle formula neglects the effect of the lead angle on the thread angle."
        )
    out["self_locking"] = t_lower > 0
    ctx.formatted = (
        f"Torque to raise {fmt_sig(t_raise, 6)} N*m, to lower {fmt_sig(t_lower, 6)} N*m, "
        f"efficiency {fmt_sig(100 * out['efficiency'], 4)}%."
    )
    return out


def _rolling(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = need(p, "N", "force")
    if p.get("C_rr") is not None:
        only(p, "rolling_resistance", {"N", "C_rr"})
        c = need(p, "C_rr", None)
        ctx.formula = "F = C_rr N"
    else:
        b, r = need(p, "b", "length"), need(p, "r", "length")
        c = b / r
        ctx.formula = "F = N b / r (b lever arm of the rolling resistance, r wheel radius)"
    ctx.formatted = f"Rolling resistance {fmt_sig(c * n, 6)} N."
    return {"F_r": c * n, "C_rr": c}
