"""eng_calc: unbalance and balancing of rigid rotors (D72).

Sources: Wikipedia pages in the local offline library.
- "Rotating unbalance": static unbalance U = m r, couple unbalance U = m r d (two equal unbalances 180 degrees apart,
  d apart along the axis), force F = U omega^2, specific unbalance e = U / m(rotor); ISO balance grade e_per omega = G
  (G in mm/s), U_per = 9.54 G mass / rpm.
- "Balancing of rotating masses": dynamic balance = the rotation produces no resultant centrifugal force or couple;
  one-plane balancing for disk-like rotors, two-plane balancing for long rotors.
Derived here and checked in the tests: unbalances as complex numbers m r e^(i theta); resultant static unbalance
S = sum m r e^(i theta), couple unbalance about z = 0, C = sum m r z e^(i theta); two-plane correction from
U_L + U_R = -S and z_L U_L + z_R U_R = -C; permissible unbalance U_per = mass G / omega (9.54 = 60 / (2 pi) rounded).
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, one_of, only, op, units_for

_MODES: dict[str, set[str]] = {
    "unbalance": {"masses", "M", "n_rpm", "omega", "planes", "r_correction"},
    "iso_grade": {"G", "M", "n_rpm", "omega"},
}
_UNITS = {
    "U": "kg*m",
    "U_angle_deg": "deg",
    "C": "kg*m2",
    "C_angle_deg": "deg",
    "e": "m",
    "F": "N",
    "M_couple": "N*m",
    "U_correction": "kg*m",
    "U_correction_angle_deg": "deg",
    "m_correction": "kg",
    "U_L": "kg*m",
    "U_L_angle_deg": "deg",
    "U_R": "kg*m",
    "U_R_angle_deg": "deg",
    "m_L": "kg",
    "m_R": "kg",
    "e_per": "m",
    "U_per": "kg*m",
    "U_per_g_mm": "g*mm",
}
_EXAMPLE = '{"masses": [{"m": 0.02, "r": 0.1, "angle_deg": 30, "z": 0.05}], "planes": [0, 0.5], "r_correction": 0.1}'


@op("rotor_balance", ("mode", *sorted(set().union(*_MODES.values()))))
def _rotor_balance(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = _unbalance(p, ctx) if mode == "unbalance" else _iso(p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _num(v: Any, name: str, positive: bool = True) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise ToolError("INVALID_ARGUMENT", f"params.{name} must be a finite number.", _EXAMPLE)
    if positive and v <= 0:
        raise ToolError("INVALID_ARGUMENT", f"params.{name} must be > 0.", _EXAMPLE)
    return float(v)


def _omega(p: dict[str, Any], required: bool) -> float | None:
    if not required and p.get("n_rpm") is None and p.get("omega") is None:
        return None
    if one_of(p, ("n_rpm", "omega"), "the speed (n_rpm or omega)") == "n_rpm":
        return need(p, "n_rpm", None) * 2 * math.pi / 60
    return need(p, "omega", "angular_velocity")


def _deg(z: complex) -> float:
    return math.degrees(cmath.phase(z)) % 360.0


def _masses(p: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    raw = p.get("masses")
    if not isinstance(raw, list) or not raw:
        raise ToolError(
            "INVALID_ARGUMENT", "params.masses: a list of {m, r, angle_deg, z}.", _EXAMPLE
        )
    out = []
    for k, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or set(item) - {"m", "r", "angle_deg", "z"}:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"params.masses[{k}]: an object with keys m, r, angle_deg, z.",
                _EXAMPLE,
            )
        m = _num(item.get("m"), f"masses[{k}].m")
        r = _num(item.get("r"), f"masses[{k}].r")
        theta = math.radians(_num(item.get("angle_deg"), f"masses[{k}].angle_deg", positive=False))
        z = _num(item.get("z", 0.0), f"masses[{k}].z", positive=False)
        out.append((m, r, theta, z))
    return out


def _radii(p: dict[str, Any], n: int) -> list[float] | None:
    raw = p.get("r_correction")
    if raw is None:
        return None
    if isinstance(raw, list):
        if len(raw) != n:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"params.r_correction: one number or a list of {n} radii.",
                _EXAMPLE,
            )
        return [_num(v, "r_correction") for v in raw]
    return [_num(raw, "r_correction")] * n


def _unbalance(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    items = _masses(p)
    s = sum(m * r * cmath.exp(1j * t) for m, r, t, _ in items)
    c = sum(m * r * z * cmath.exp(1j * t) for m, r, t, z in items)
    out: dict[str, Any] = {"U": abs(s), "U_angle_deg": _deg(s), "C": abs(c), "C_angle_deg": _deg(c)}
    out |= {"U_correction": abs(s), "U_correction_angle_deg": _deg(-s)}
    formula = "S = sum m r e^(i theta), U = |S|; C = sum m r z e^(i theta) (couple about z = 0)"
    if p.get("M") is not None:
        out["e"] = abs(s) / need(p, "M", "mass")
        formula += "; e = U / M"
    omega = _omega(p, required=False)
    if omega is not None:
        out["F"] = abs(s) * omega**2
        out["M_couple"] = abs(c) * omega**2
        formula += "; F = U omega^2, couple C omega^2"
    planes = p.get("planes")
    if planes is None:
        radii = _radii(p, 1)
        if radii is not None:
            out["m_correction"] = abs(s) / radii[0]
        formula += "; one-plane correction -S"
    else:
        if not isinstance(planes, list) or len(planes) != 2:
            raise ToolError(
                "INVALID_ARGUMENT", "params.planes: two axial positions [z_L, z_R].", _EXAMPLE
            )
        z_l, z_r = (_num(v, "planes", positive=False) for v in planes)
        if z_l == z_r:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.planes: two different axial positions [z_L, z_R].",
                _EXAMPLE,
            )
        u_r = (s * z_l - c) / (z_r - z_l)
        u_l = -s - u_r
        out |= {
            "U_L": abs(u_l),
            "U_L_angle_deg": _deg(u_l),
            "U_R": abs(u_r),
            "U_R_angle_deg": _deg(u_r),
        }
        radii = _radii(p, 2)
        if radii is not None:
            out["m_L"], out["m_R"] = abs(u_l) / radii[0], abs(u_r) / radii[1]
        formula += "; two-plane correction: U_L + U_R = -S, z_L U_L + z_R U_R = -C"
    ctx.formula = formula
    ctx.warn(
        "Rigid rotor with point masses; angles in the rotor frame; the couple is taken about z = 0."
    )
    ctx.formatted = (
        f"Static unbalance {fmt_sig(abs(s) * 1e6, 6)} g*mm at {fmt_sig(out['U_angle_deg'], 6)} deg, "
        f"couple unbalance {fmt_sig(abs(c), 6)} kg*m2."
    )
    return out


def _iso(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    g = need(p, "G", None)
    mass = need(p, "M", "mass")
    omega = _omega(p, required=True)
    assert omega is not None
    e_per = g * 1e-3 / omega
    u_per = mass * e_per
    ctx.formula = (
        "e_per omega = G (G in mm/s); U_per = M e_per (the page form 9.54 G M / rpm gives kg*mm)"
    )
    ctx.formatted = (
        f"Balance grade G{fmt_sig(g, 4)}: permissible residual unbalance {fmt_sig(u_per * 1e6, 6)} g*mm "
        f"(e_per {fmt_sig(e_per * 1e6, 6)} um)."
    )
    return {"e_per": e_per, "U_per": u_per, "U_per_g_mm": u_per * 1e6}
