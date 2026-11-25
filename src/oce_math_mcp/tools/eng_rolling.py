"""eng_calc: rolling without slipping of a rigid body (D73).

Source: Wikipedia page "Rolling" in the local offline library.
- v_com = r omega, a = r alpha; a = F_net / m = r tau / I.
- External force through the centre (no torque): F_net = F_external / (1 + I / (m r^2)), a = F_external / (m + I / r^2);
  the tangential force at the contact (static friction) provides the torque; when static friction is not enough the
  body slips.
- Incline with only static friction, normal force and weight: a = g sin(theta) / (1 + (r_g / r)^2), independent of
  scale and density for the same r_g / r.
- Kinetic energy K = K_translation + K_rotation.
Derived here and checked in the tests: friction force f = F_along - m a, normal force, minimum static friction
coefficient f / N for pure rolling, K_rotation = I (v / r)^2 / 2.
"""

from __future__ import annotations

import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, one_of, only, op, qty, units_for
from .eng_thermo import G0

_COMMON = {"m", "r", "I", "r_g", "g", "mu_s", "v"}
_MODES: dict[str, set[str]] = {"incline": _COMMON | {"theta_deg"}, "force": _COMMON | {"F"}}
_UNITS = {
    "a": "m/s2",
    "alpha": "rad/s2",
    "F_net": "N",
    "f": "N",
    "N": "N",
    "K_translation": "J",
    "K_rotation": "J",
    "K": "J",
}


@op("rolling", ("mode", *sorted(set().union(*_MODES.values()))))
def _rolling(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    m, r = need(p, "m", "mass"), need(p, "r", "length")
    if (
        one_of(
            p, ("I", "r_g"), "the rotational inertia (I about the centre or radius of gyration r_g)"
        )
        == "I"
    ):
        inertia = need(p, "I", "mass_moment")
    else:
        inertia = m * need(p, "r_g", "length") ** 2
    g = qty(p, "g", "acceleration", default=G0) or G0
    if mode == "incline":
        theta_deg = need(p, "theta_deg", None, positive=False)
        if not 0 <= theta_deg < 90:
            raise ToolError("INVALID_ARGUMENT", "params.theta_deg must be in [0, 90) degrees.")
        t = math.radians(theta_deg)
        along, normal = m * g * math.sin(t), m * g * math.cos(t)
        ctx.formula = "a = g sin(theta) / (1 + (r_g / r)^2), r_g^2 = I / m"
    else:
        along, normal = need(p, "F", "force", positive=False), m * g
        ctx.formula = (
            "a = F / (m + I / r^2), F_net = F / (1 + I / (m r^2)) (force through the centre)"
        )
    ratio = inertia / (m * r * r)
    a = along / (m + inertia / (r * r))
    f = along - m * a
    out: dict[str, Any] = {
        "a": a,
        "alpha": a / r,
        "F_net": m * a,
        "f": f,
        "N": normal,
        "I_over_m_r2": ratio,
    }
    out["mu_min"] = abs(f) / normal
    ctx.formula += "; friction at the contact f = F_along - m a, pure rolling while |f| <= mu_s N"
    if p.get("mu_s") is not None:
        mu_s = need(p, "mu_s", None, positive=False)
        out["rolls_without_slipping"] = out["mu_min"] <= mu_s
        if not out["rolls_without_slipping"]:
            ctx.warn(
                "Static friction is not enough: the body slips and the pure-rolling acceleration does not hold."
            )
    if p.get("v") is not None:
        v = need(p, "v", "velocity", positive=False)
        out["K_translation"] = m * v * v / 2
        out["K_rotation"] = inertia * (v / r) ** 2 / 2
        out["K"] = out["K_translation"] + out["K_rotation"]
        ctx.formula += "; K = m v^2 / 2 + I (v / r)^2 / 2"
    ctx.formatted = (
        f"Rolling without slipping: a = {fmt_sig(a, 6)} m/s2, friction needed {fmt_sig(f, 6)} N "
        f"(mu at least {fmt_sig(out['mu_min'], 4)})."
    )
    ctx.units = units_for(out, _UNITS)
    return out
