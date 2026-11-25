"""eng_calc: stress concentration at holes in an infinite plate (D71).

Sources: Wikipedia pages in the local offline library.
- "Kirsch equations": circular hole of radius a, far-field stress sigma, angle theta from the direction of the stress:
  s_rr = (sigma/2)(1 - a^2/r^2) + (sigma/2)(1 + 3 a^4/r^4 - 4 a^2/r^2) cos(2 theta),
  s_tt = (sigma/2)(1 + a^2/r^2) - (sigma/2)(1 + 3 a^4/r^4) cos(2 theta),
  s_rt = -(sigma/2)(1 - 3 a^4/r^4 + 2 a^2/r^2) sin(2 theta).
- "Stress concentration": K_t = sigma_max / sigma_nom; K_t = 3 for a circular hole in an infinite plate; Inglis,
  elliptical hole 2a x 2b, sigma_max = sigma_0 (1 + 2 a/b) = sigma (1 + 2 sqrt(a/rho)), rho the radius of curvature;
  not usable for a crack (rho -> 0).
Derived here and checked in the tests: biaxial far field (sigma_x, sigma_y) by superposition (the sigma_y field is the
Kirsch field turned by 90 degrees); hoop stress on the edge 2P - 4D cos(2 theta) with P = (sigma_x + sigma_y)/2,
D = (sigma_x - sigma_y)/2, its extremes, and K_t = largest edge stress magnitude / largest applied stress magnitude.
"""

from __future__ import annotations

import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, one_of, only, op, qty, units_for

_MODES: dict[str, set[str]] = {
    "circular_hole": {"sigma_x", "sigma_y", "a", "r", "theta_deg"},
    "elliptical_hole": {"sigma", "a", "b", "rho"},
}
_UNITS = {
    "sigma_rr": "Pa",
    "sigma_tt": "Pa",
    "sigma_rt": "Pa",
    "sigma_edge_max": "Pa",
    "sigma_edge_min": "Pa",
    "theta_edge_max_deg": "deg",
    "theta_edge_min_deg": "deg",
    "sigma_max": "Pa",
    "b": "m",
    "rho": "m",
}


@op("stress_concentration", ("mode", *sorted(set().union(*_MODES.values()))))
def _stress_concentration(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = _circle(p, ctx) if mode == "circular_hole" else _ellipse(p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _circle(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    sx = need(p, "sigma_x", "pressure", positive=False)
    sy = qty(p, "sigma_y", "pressure", positive=False, default=0.0) or 0.0
    if sx == 0 and sy == 0:
        raise ToolError("INVALID_ARGUMENT", "At least one of sigma_x, sigma_y must be non-zero.")
    a = need(p, "a", "length")
    mean, dev = (sx + sy) / 2, (sx - sy) / 2
    out: dict[str, Any] = {}
    if p.get("r") is not None or p.get("theta_deg") is not None:
        r = need(p, "r", "length")
        theta = math.radians(need(p, "theta_deg", None, positive=False))
        if r < a:
            raise ToolError("INVALID_ARGUMENT", "The point must be in the plate: r >= a.")
        k2, k4 = (a / r) ** 2, (a / r) ** 4
        c2, s2 = math.cos(2 * theta), math.sin(2 * theta)
        out["sigma_rr"] = mean * (1 - k2) + dev * (1 + 3 * k4 - 4 * k2) * c2
        out["sigma_tt"] = mean * (1 + k2) - dev * (1 + 3 * k4) * c2
        out["sigma_rt"] = -dev * (1 - 3 * k4 + 2 * k2) * s2
    hi, lo = 2 * mean + 4 * abs(dev), 2 * mean - 4 * abs(dev)
    out["sigma_edge_max"], out["sigma_edge_min"] = hi, lo
    out["theta_edge_max_deg"] = 90.0 if dev > 0 else 0.0
    out["theta_edge_min_deg"] = 0.0 if dev > 0 else (90.0 if dev < 0 else 0.0)
    out["Kt"] = max(abs(hi), abs(lo)) / max(abs(sx), abs(sy))
    ctx.formula = (
        "Kirsch (superposition of sigma_x and sigma_y): P = (sx + sy)/2, D = (sx - sy)/2, "
        "s_rr = P (1 - a^2/r^2) + D (1 + 3 a^4/r^4 - 4 a^2/r^2) cos 2t, s_tt = P (1 + a^2/r^2) - D (1 + 3 a^4/r^4) cos 2t, "
        "s_rt = -D (1 - 3 a^4/r^4 + 2 a^2/r^2) sin 2t; edge s_tt = 2P - 4D cos 2t"
    )
    ctx.warn("Infinite linear elastic plate: a finite width changes the nominal stress and K_t.")
    ctx.formatted = (
        f"Circular hole: edge stress from {fmt_sig(lo / 1e6, 6)} to {fmt_sig(hi / 1e6, 6)} MPa, "
        f"K_t = {fmt_sig(out['Kt'], 6)}."
    )
    return out


def _ellipse(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    sigma = need(p, "sigma", "pressure", positive=False)
    a = need(p, "a", "length")
    if (
        one_of(p, ("b", "rho"), "the elliptical hole (semi-axis b or radius of curvature rho)")
        == "b"
    ):
        b = need(p, "b", "length")
        rho = b * b / a
        kt = 1 + 2 * a / b
        ctx.formula = "Inglis: sigma_max = sigma (1 + 2 a/b), rho = b^2 / a"
    else:
        rho = need(p, "rho", "length")
        b = math.sqrt(a * rho)
        kt = 1 + 2 * math.sqrt(a / rho)
        ctx.formula = "Inglis: sigma_max = sigma (1 + 2 sqrt(a/rho)), b = sqrt(a rho)"
    ctx.warn(
        "a is the semi-axis perpendicular to the applied stress; infinite plate. For rho -> 0 (a crack) K_t is not "
        "usable: use the stress intensity factor."
    )
    ctx.formatted = f"Elliptical hole: K_t = {fmt_sig(kt, 6)}, maximum stress {fmt_sig(sigma * kt / 1e6, 6)} MPa."
    return {"Kt": kt, "sigma_max": sigma * kt, "b": b, "rho": rho}
