"""eng_calc: electromagnetic radiation and astronomical photometry (D106).

Source: Wikipedia "List of photonics equations", sections "Physical optics (EM luminal waves)" and
"Astrophysics definitions", in the local offline library. SI units, vacuum.
- Poynting vector N = E x B / mu0 = E x H, its flux through a flat surface Phi = N . n A (the integral over a
  curved surface is math_solve.surface_integral); a plane wave has B0 = E0 / c, E_rms = E0 / sqrt(2),
  intensity I = <N> = E0 B0 / (2 mu0), mean energy density u = I / c, momentum density g = I / c^2.
- Radiation momentum p = U / c and pressure P = I / c = p / (A t) for an absorbing surface at normal incidence;
  here also a reflectance R (P = (1 + R) I cos^2(theta) / c, the normal component) and an angle of incidence.
- Luminosity distance D_L = sqrt(L / (4 pi F)); apparent magnitude m = -(5/2) log10(F / F0); absolute magnitude
  M = m - 5 (log10 D_L[pc] - 1), distance modulus mu = m - M; colour indices U - B, B - V (absolute or apparent
  magnitudes: the same difference); bolometric correction BC = m_bol - V = M_bol - M_V.
"""

from __future__ import annotations

import math
from typing import Any

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import (
    choice,
    fmt_sig,
    log_ratio,
    need,
    one_of,
    only,
    op,
    qty,
    range_guard,
    units_for,
)

_EM_MODES: dict[str, set[str]] = {
    "poynting": {"E", "B", "H", "normal", "A"},
    "plane_wave": {"E0", "B0", "E_rms", "I"},
    "radiation_pressure": {"I", "P", "U", "A", "t", "R", "theta_deg"},
}
_PH_MODES: dict[str, set[str]] = {
    "luminosity_distance": {"L", "F", "D_L"},
    "apparent_magnitude": {"m", "F", "F0"},
    "distance_modulus": {"m", "M", "D_L"},
    "color_index": {"U", "B", "V"},
    "bolometric_correction": {"BC", "m_bol", "V"},
}
_UNITS = {
    "N": "W/m2",
    "N_magnitude": "W/m2",
    "flux": "W",
    "E0": "V/m",
    "E_rms": "V/m",
    "B0": "T",
    "B_rms": "T",
    "I": "W/m2",
    "u": "J/m3",
    "g": "kg/(m2 s)",
    "p_absorbed": "Pa",
    "p_reflected": "Pa",
    "pressure": "Pa",
    "force": "N",
    "momentum": "kg m/s",
    "energy": "J",
    "L": "W",
    "F": "W/m2",
    "F0": "W/m2",
    "D_L": "m",
    "D_L_pc": "pc",
    "D_L_ly": "ly",
    "m": "mag",
    "M": "mag",
    "mu": "mag",
    "U_B": "mag",
    "B_V": "mag",
    "BC": "mag",
    "m_bol": "mag",
    "V": "mag",
}
_PC = 149597870700.0 * 648000 / math.pi  # m: 1 au (exact, IAU 2012) times 648000 / pi (IAU 2015)
_LY = 299792458.0 * 365.25 * 86400  # m: c times a Julian year


def _c() -> float:
    return lookup("c")[1]


def _mu0() -> float:
    return lookup("mu_0")[1]


def _vec3(p: dict[str, Any], key: str) -> list[float]:
    raw = p.get(key)
    if not isinstance(raw, list) or len(raw) != 3:
        raise ToolError(
            "INVALID_ARGUMENT", f"params.{key}: 3 numbers (x, y, z components in SI units)."
        )
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ToolError("INVALID_ARGUMENT", f"params.{key}: finite numbers only.")
    return [float(v) for v in raw]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _two_of(p: dict[str, Any], keys: tuple[str, ...]) -> str:
    """The missing one of three keys, when exactly two are given."""
    missing = [k for k in keys if p.get(k) is None]
    if len(missing) != 1:
        raise ToolError(
            "INVALID_ARGUMENT", f"Give two of {', '.join(keys)}: the third one is computed."
        )
    return missing[0]


@op("em_radiation", ("mode", *sorted(set().union(*_EM_MODES.values()))))
@range_guard
def _em_radiation(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_EM_MODES))
    only(p, mode, _EM_MODES[mode])
    out = {"poynting": _poynting, "plane_wave": _plane_wave, "radiation_pressure": _pressure}[mode](
        p, ctx
    )
    ctx.units = units_for(out, _UNITS)
    return out


def _poynting(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    e = _vec3(p, "E")
    if one_of(p, ("B", "H"), "the magnetic field") == "B":
        mu0 = _mu0()
        n = [x / mu0 for x in _cross(e, _vec3(p, "B"))]
        ctx.formula = "N = E x B / mu0"
    else:
        n = _cross(e, _vec3(p, "H"))
        ctx.formula = "N = E x H"
    size = math.sqrt(sum(x * x for x in n))
    out: dict[str, Any] = {"N": n, "N_magnitude": size}
    if p.get("normal") is not None or p.get("A") is not None:
        if p.get("normal") is None or p.get("A") is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "The flux needs both params.normal (the normal of a flat surface) and params.A (its area).",
            )
        nv = _vec3(p, "normal")
        length = math.sqrt(sum(x * x for x in nv))
        if length == 0:
            raise ToolError("INVALID_ARGUMENT", "params.normal must not be the zero vector.")
        out["flux"] = sum(a * b for a, b in zip(n, nv, strict=True)) / length * need(p, "A", "area")
        ctx.formula += ", Phi = N . n A (flat surface, n the unit normal)"
    ctx.formatted = f"|N| = {fmt_sig(size)} W/m2"
    return out


def _plane_wave(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    c, mu0 = _c(), _mu0()
    src = one_of(p, ("E0", "B0", "E_rms", "I"), "the plane wave")
    if src == "E0":
        e0 = need(p, "E0", "electric_field")
    elif src == "B0":
        e0 = c * need(p, "B0", "magnetic_flux_density")
    elif src == "E_rms":
        e0 = math.sqrt(2) * need(p, "E_rms", "electric_field")
    else:
        e0 = math.sqrt(2 * need(p, "I", "irradiance") * mu0 * c)
    b0 = e0 / c
    i = e0 * b0 / (2 * mu0)
    ctx.formula = (
        "B0 = E0 / c, E_rms = E0 / sqrt(2), I = <N> = E0 B0 / (2 mu0), u = I / c, g = I / c^2, "
        "pressure I / c absorbed and 2 I / c reflected (vacuum)"
    )
    ctx.formatted = f"E0 = {fmt_sig(e0)} V/m, B0 = {fmt_sig(b0)} T, I = {fmt_sig(i)} W/m2"
    return {
        "E0": e0,
        "E_rms": e0 / math.sqrt(2),
        "B0": b0,
        "B_rms": b0 / math.sqrt(2),
        "I": i,
        "u": i / c,
        "g": i / (c * c),
        "p_absorbed": i / c,
        "p_reflected": 2 * i / c,
    }


def _pressure(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    c = _c()
    refl = qty(p, "R", None, positive=False, default=0.0)
    assert refl is not None
    if not 0 <= refl <= 1:
        raise ToolError(
            "INVALID_ARGUMENT", "params.R: the reflectance, 0 <= R <= 1 (0 absorbs, 1 mirrors)."
        )
    deg = qty(p, "theta_deg", None, positive=False, default=0.0)
    assert deg is not None
    if not 0 <= deg < 90:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.theta_deg: the angle of incidence in degrees, 0 <= theta_deg < 90.",
        )
    cos = math.cos(math.radians(deg))
    src = one_of(p, ("I", "P", "U"), "the intensity")
    if src == "I":
        i = need(p, "I", "irradiance")
    elif src == "P":
        i = need(p, "P", "power") / need(p, "A", "area")
    else:  # U falls on the area A in the time t
        i = need(p, "U", "energy") / (need(p, "A", "area") * need(p, "t", "time") * cos)
    pressure = (1 + refl) * i * cos * cos / c
    out: dict[str, Any] = {"I": i, "pressure": pressure}
    if p.get("A") is not None:
        area = need(p, "A", "area")
        out["force"] = pressure * area
        if p.get("t") is not None:
            t = need(p, "t", "time")
            out["momentum"] = out["force"] * t
            out["energy"] = i * area * cos * t
    ctx.formula = "P = (1 + R) I cos^2(theta) / c (normal component; R = 0, theta = 0: P = I / c = p / (A t), p = U / c)"
    ctx.formatted = f"P = {fmt_sig(pressure)} Pa"
    return out


@op("photometry", ("mode", *sorted(set().union(*_PH_MODES.values()))))
@range_guard
def _photometry(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_PH_MODES))
    only(p, mode, _PH_MODES[mode])
    out = {
        "luminosity_distance": _luminosity_distance,
        "apparent_magnitude": _apparent_magnitude,
        "distance_modulus": _distance_modulus,
        "color_index": _color_index,
        "bolometric_correction": _bolometric,
    }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _mag(p: dict[str, Any], key: str) -> float:
    v = need(p, key, None, positive=False)
    return v


def _luminosity_distance(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    missing = _two_of(p, ("L", "F", "D_L"))
    if missing == "D_L":
        lum, f = need(p, "L", "power"), need(p, "F", "irradiance")
        d = math.sqrt(lum / (4 * math.pi * f))
    elif missing == "L":
        f, d = need(p, "F", "irradiance"), need(p, "D_L", "length")
        lum = 4 * math.pi * d * d * f
    else:
        lum, d = need(p, "L", "power"), need(p, "D_L", "length")
        f = lum / (4 * math.pi * d * d)
    ctx.formula = "D_L = sqrt(L / (4 pi F))"
    ctx.formatted = f"D_L = {fmt_sig(d)} m = {fmt_sig(d / _PC)} pc"
    return {"L": lum, "F": f, "D_L": d, "D_L_pc": d / _PC, "D_L_ly": d / _LY}


def _apparent_magnitude(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    missing = _two_of(p, ("m", "F", "F0"))
    if missing == "m":
        f, f0 = need(p, "F", "irradiance"), need(p, "F0", "irradiance")
        m = -2.5 * log_ratio(f, f0) / math.log(10)
    elif missing == "F":
        m, f0 = _mag(p, "m"), need(p, "F0", "irradiance")
        f = f0 * 10 ** (-0.4 * m)
    else:
        m, f = _mag(p, "m"), need(p, "F", "irradiance")
        f0 = f * 10 ** (0.4 * m)
    ctx.formula = "m = -(5/2) log10(F / F0)"
    ctx.formatted = f"m = {fmt_sig(m)}"
    return {"m": m, "F": f, "F0": f0}


def _distance_modulus(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    missing = _two_of(p, ("m", "M", "D_L"))
    if missing == "D_L":
        m, big_m = _mag(p, "m"), _mag(p, "M")
        d_pc = 10 ** ((m - big_m) / 5 + 1)
    else:
        d_pc = need(p, "D_L", "length") / _PC
        if missing == "M":
            m = _mag(p, "m")
            big_m = m - 5 * (math.log10(d_pc) - 1)
        else:
            big_m = _mag(p, "M")
            m = big_m + 5 * (math.log10(d_pc) - 1)
    ctx.formula = "M = m - 5 (log10 D_L - 1), D_L in parsec; mu = m - M"
    ctx.formatted = f"m = {fmt_sig(m)}, M = {fmt_sig(big_m)}, mu = {fmt_sig(m - big_m)}"
    return {"m": m, "M": big_m, "mu": m - big_m, "D_L": d_pc * _PC, "D_L_pc": d_pc}


def _color_index(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if p.get("B") is None or (p.get("U") is None and p.get("V") is None):
        raise ToolError(
            "INVALID_ARGUMENT", "Give two of U, B, V with B: the indices are U - B and B - V."
        )
    b = _mag(p, "B")
    out: dict[str, Any] = {}
    if p.get("U") is not None:
        out["U_B"] = _mag(p, "U") - b
    if p.get("V") is not None:
        out["B_V"] = b - _mag(p, "V")
    ctx.formula = "U - B = M_U - M_B, B - V = M_B - M_V (absolute or apparent magnitudes)"
    ctx.formatted = ", ".join(f"{k.replace('_', ' - ')} = {fmt_sig(v)}" for k, v in out.items())
    return out


def _bolometric(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    missing = _two_of(p, ("BC", "m_bol", "V"))
    if missing == "BC":
        m_bol, v = _mag(p, "m_bol"), _mag(p, "V")
        bc = m_bol - v
    elif missing == "m_bol":
        bc, v = _mag(p, "BC"), _mag(p, "V")
        m_bol = bc + v
    else:
        bc, m_bol = _mag(p, "BC"), _mag(p, "m_bol")
        v = m_bol - bc
    ctx.formula = "BC = m_bol - V = M_bol - M_V"
    ctx.formatted = f"BC = {fmt_sig(bc)}"
    return {"BC": bc, "m_bol": m_bol, "V": v}
