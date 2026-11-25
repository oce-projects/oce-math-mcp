"""eng_calc: torsion of non-circular sections (D60).

Main source: Wikipedia "Torsion (mechanics)", table of sections:
- circle J = pi R^4 / 2, tau = 2T / (pi R^3); annulus J = (pi/2)(Re^4 - Ri^4), tau = 2 T Re / (pi (Re^4 - Ri^4));
- ellipse J = pi p^3 q^3 / (p^2 + q^2), tau = 2T / (pi p q^2) at the ends of the minor axis;
- equilateral triangle J = sqrt(3) a^4 / 80 (exact cubic Prandtl stress function);
- rectangle J = alpha(b/t) b t^3 "from a Fourier-series solution": the series of the Prandtl function is written
  here and checked in the tests against finite differences and against the beta table of "Torsion constant";
- thin-walled open sections J = (1/3) sum b t^3 (tau = 3T / (a s^2) for one strip);
- thin-walled closed section (Bredt) J = 4 A_m^2 / oint ds/t, tau = T / (2 A_m t).
Derived here: triangle tau 20 T / a^3 from the cubic function (tau = G theta' h / 2), wall stresses of open
sections G theta' t_i (same twist for all walls), angle phi = T L / (G J) as in the source.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import zeta

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, one_of, only, op, qty, units_for

_COMMON = {"T", "G", "L"}
_SHAPES: dict[str, set[str]] = {
    "circle": {"R", "D"},
    "annulus": {"R_e", "R_i", "D_e", "D_i"},
    "ellipse": {"p", "q"},
    "equilateral_triangle": {"a"},
    "rectangle": {"b", "t"},
    "thin_open": {"walls"},
    "thin_closed": {"A_m", "walls"},
}
_UNITS = {
    "J": "m4",
    "W_t": "m3",
    "tau_max": "Pa",
    "tau_walls": "Pa",
    "shear_flow": "N/m",
    "twist_rate": "rad/m",
    "theta": "rad",
    "theta_deg": "deg",
    "GJ": "N*m2",
    "k_t": "N*m/rad",
}
_ODD = np.arange(1, 61, 2, dtype=float)  # 30 odd terms: the tails decay like exp(-n pi / 2)
_SUM_ODD_INV5 = 31 / 32 * float(zeta(5))  # sum over odd n of 1/n^5 = (1 - 2^-5) zeta(5)


def _rectangle(b: float, t: float) -> tuple[float, float]:
    """b >= t. Returns (J, k) with tau_max = k G theta' t, from the series of the Prandtl function."""
    e2 = np.exp(-_ODD * math.pi * b / t)  # exp(-2 k_n), k_n = n pi b / (2 t)
    one_minus_tanh = 2 * e2 / (1 + e2)
    s_j = _SUM_ODD_INV5 - float(np.sum(one_minus_tanh / _ODD**5))
    j = b * t**3 / 3 * (1 - 192 / math.pi**5 * (t / b) * s_j)
    e1 = np.sqrt(e2)
    s_t = float(np.sum(2 * e1 / (1 + e2) / _ODD**2))  # 1 / cosh(k_n) without overflow
    return j, 1 - 8 / math.pi**2 * s_t


def _walls(p: dict[str, Any]) -> list[tuple[float, float]]:
    raw = p.get("walls")
    msg = "params.walls: list of walls [length, thickness] in m, both > 0."
    if not isinstance(raw, list) or not raw:
        raise ToolError("INVALID_ARGUMENT", msg)
    out = []
    for row in raw:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or any(isinstance(x, bool) or not isinstance(x, int | float) for x in row)
            or not all(math.isfinite(x) and x > 0 for x in row)
        ):
            raise ToolError("INVALID_ARGUMENT", msg)
        out.append((float(row[0]), float(row[1])))
    return out


@op("torsion_section", ("shape", *sorted(_COMMON.union(*_SHAPES.values()))))
def _torsion(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    shape = choice(p, "shape", tuple(_SHAPES))
    only(p, shape, _SHAPES[shape] | _COMMON, field="shape")
    torque = qty(p, "T", "torque", need=False, positive=False)
    # reference torque for the section: the stresses are linear in T, and W_t = T / tau_max does not depend on it
    # (D75: T = 0 gave 0/0); the outputs are scaled to the real torque below
    t_abs = abs(torque) if torque else 1.0
    out: dict[str, Any] = {}
    if shape == "circle":
        key = one_of(p, ("R", "D"), "the circle")
        r = need(p, key, "length") / (2 if key == "D" else 1)
        j = math.pi * r**4 / 2
        tau = 2 * t_abs / (math.pi * r**3)
        ctx.formula = "J = pi R^4 / 2; tau_max = 2 T / (pi R^3) (exact)"
    elif shape == "annulus":
        ke = one_of(p, ("R_e", "D_e"), "the outer radius")
        ki = one_of(p, ("R_i", "D_i"), "the inner radius")
        r_e = need(p, ke, "length") / (2 if ke == "D_e" else 1)
        r_i = need(p, ki, "length") / (2 if ki == "D_i" else 1)
        if r_i >= r_e:
            raise ToolError(
                "INVALID_ARGUMENT", "params.R_i (or D_i) must be smaller than R_e (or D_e)."
            )
        j = math.pi / 2 * (r_e**4 - r_i**4)
        tau = 2 * t_abs * r_e / (math.pi * (r_e**4 - r_i**4))
        ctx.formula = "J = (pi/2)(Re^4 - Ri^4); tau_max = 2 T Re / (pi (Re^4 - Ri^4)) (exact)"
    elif shape == "ellipse":
        a1, a2 = need(p, "p", "length"), need(p, "q", "length")
        pp, qq = max(a1, a2), min(a1, a2)
        j = math.pi * pp**3 * qq**3 / (pp**2 + qq**2)
        tau = 2 * t_abs / (math.pi * pp * qq**2)
        ctx.formula = "J = pi p^3 q^3 / (p^2 + q^2); tau_max = 2 T / (pi p q^2) at the ends of the minor axis (exact)"
    elif shape == "equilateral_triangle":
        a = need(p, "a", "length")
        j = math.sqrt(3) / 80 * a**4
        tau = 20 * t_abs / a**3
        ctx.formula = "J = sqrt(3) a^4 / 80 (exact); tau_max = 20 T / a^3 at the middle of the sides (from the cubic function)"
    elif shape == "rectangle":
        s1, s2 = need(p, "b", "length"), need(p, "t", "length")
        b, t = max(s1, s2), min(s1, s2)
        j, k = _rectangle(b, t)
        tau = t_abs * k * t / j
        out["beta"] = j / (b * t**3)
        ctx.formula = (
            "J = (b t^3 / 3)(1 - (192 / pi^5)(t / b) sum_odd tanh(n pi b / 2t) / n^5); "
            "tau_max = G theta' t (1 - (8 / pi^2) sum_odd 1 / (n^2 cosh(n pi b / 2t))) at the middle of the long side"
        )
    elif shape == "thin_open":
        walls = _walls(p)
        j = sum(b * t**3 for b, t in walls) / 3
        taus = [t_abs * t / j for _, t in walls]
        tau = max(taus)
        if torque is not None:
            out["tau_walls"] = taus
        ctx.formula = "J = (1/3) sum b t^3; tau_i = T t_i / J (thin strip: 3 T / (a s^2))"
        ctx.warn(
            "Thin strip: slender walls (length much larger than thickness), junctions neglected."
        )
    else:
        a_m = need(p, "A_m", "area")
        walls = _walls(p)
        j = 4 * a_m**2 / sum(ln / t for ln, t in walls)
        q = t_abs / (2 * a_m)
        taus = [q / t for _, t in walls]
        tau = max(taus)
        if torque is not None:
            out["shear_flow"] = q
            out["tau_walls"] = taus
        ctx.formula = "J = 4 A_m^2 / sum(L_i / t_i) (Bredt); q = T / (2 A_m); tau_i = q / t_i"
        ctx.warn("Bredt approximation: thin wall, A_m is the area enclosed by the wall mid-line.")
    out["J"] = j
    out["W_t"] = t_abs / tau
    if torque is not None:
        scale = abs(torque) / t_abs  # 1, or 0 for a zero torque
        tau *= scale
        out["tau_max"] = tau
        if "tau_walls" in out:
            out["tau_walls"] = [x * scale for x in out["tau_walls"]]
        if "shear_flow" in out:
            out["shear_flow"] *= scale
    g_mod = qty(p, "G", "pressure", need=False)
    length = qty(p, "L", "length", need=False)
    if g_mod is not None:
        out["GJ"] = g_mod * j
        if length is not None:
            out["k_t"] = g_mod * j / length
        if torque is not None:
            out["twist_rate"] = abs(torque) / (g_mod * j)
            if length is not None:
                out["theta"] = abs(torque) * length / (g_mod * j)
                out["theta_deg"] = math.degrees(out["theta"])
                ctx.formula += "; theta = T L / (G J)"
    text = f"Section {shape}: J = {fmt_sig(j, 6)} m^4"
    if torque is not None:
        text += f", tau_max = {fmt_sig(tau / 1e6, 5)} MPa"
    if "theta_deg" in out:
        text += f", twist {fmt_sig(out['theta_deg'], 5)} degrees"
    ctx.formatted = text + "."
    ctx.units = units_for(out, _UNITS)
    return out
