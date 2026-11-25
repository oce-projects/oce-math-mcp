"""eng_calc: psychrometrics of moist air (D51).

Ideal-gas model for dry air and vapour, as in ASHRAE Handbook - Fundamentals (2017) ch. 1:
- saturation pressure p_ws from Eq. 30 of IAPWS-IF97 (`eng_steam.psat`), not from the
  Hyland-Wexler formula (ASHRAE coefficients declared in PsychroLib); measured gap between the two from 0 to
  100 °C: at most 0.0176%;
- W = eps p_w / (p - p_w), eps = M_water / M_air = 18.015268 / 28.966 = 0.621945;
- h = 1.006 t + W (2501 + 1.86 t) kJ/kg of dry air (t in °C);
- v = R_a T (1 + W / eps) / p with R_a = 287.042 J/(kg K) (8,314.472 / 28.966);
- thermodynamic wet bulb from the energy balance of adiabatic saturation, with the liquid
  water h_w = 4.186 t*, solved with brentq;
- dew point with Eq. 31 of IF97 (T_s(p_w)), exact inverse of Eq. 30;
- pressure from altitude with the ISO 2533 standard atmosphere (troposphere, up to 11 km).
Range: from 0 °C to 200 °C (dry bulb); below 0 °C saturation over ice would be needed, which is
not included. The enhancement factor f (about 1.004 at 1 atm) is neglected, as in the ASHRAE model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

from scipy.optimize import brentq

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, only, op, qty, temperature, units_for
from .eng_steam import P_TRIPLE_EQ31, psat, tsat

EPS = 0.621945
CP_DA, H_G0, CP_V, CP_W = 1.006, 2501.0, 1.86, 4.186  # kJ/(kg K), kJ/kg
R_DA = 287.042  # J/(kg K)
P_STD = 101325.0
T_LO, T_HI = 273.15, 473.15
ISA_T0, ISA_L, ISA_G, ISA_R = 288.15, 0.0065, 9.80665, 287.05287

# ------------------------------------------------------------------ relations


def pws(T: float) -> float:
    """Saturation vapour pressure in Pa (IF97, Eq. 30)."""
    return psat(T) * 1e6


def w_from_pw(pw: float, p: float) -> float:
    return EPS * pw / (p - pw)


def pw_from_w(w: float, p: float) -> float:
    return p * w / (EPS + w)


def enthalpy(t: float, w: float) -> float:
    """h of moist air in kJ/kg of dry air; t in °C."""
    return CP_DA * t + w * (H_G0 + CP_V * t)


def w_from_h(t: float, h: float) -> float:
    return (h - CP_DA * t) / (H_G0 + CP_V * t)


def t_from_hw(h: float, w: float) -> float:
    return (h - H_G0 * w) / (CP_DA + CP_V * w)


def w_from_wetbulb(t: float, twb: float, p: float) -> float:
    """Balance of adiabatic saturation: h(t, W) + (Ws* - W) h_w(t*) = h(t*, Ws*)."""
    ws = w_from_pw(pws(twb + 273.15), p)
    return (CP_DA * (twb - t) + ws * (H_G0 + CP_V * twb - CP_W * twb)) / (
        H_G0 + CP_V * t - CP_W * twb
    )


def wetbulb(t: float, w: float, p: float) -> float | None:
    """Wet bulb in °C from t and W; None if below 0 °C (outside the range)."""

    def err(x: float) -> float:
        return w_from_wetbulb(t, x, p) - w

    if err(0.0) > 0:
        return None
    top = err(t)
    if top < 0:
        if top > -1e-9 * max(w, 1e-9):  # saturated air: wet bulb = dry bulb
            return t
        raise ToolError(
            "INVALID_ARGUMENT", "Humidity beyond saturation (fog): the model does not hold."
        )
    return float(cast(float, brentq(err, 0.0, t, xtol=1e-12)))


def dewpoint(pw: float) -> float | None:
    """Dew point in K (IF97 Eq. 31); None if p_w < 611.213 Pa (dew point below 0 °C)."""
    return None if pw < P_TRIPLE_EQ31 * 1e6 else tsat(pw / 1e6)


def p_altitude(z: float) -> float:
    """Pressure of the ISO 2533 standard atmosphere at altitude z (m), troposphere."""
    return P_STD * (1 - ISA_L * z / ISA_T0) ** (ISA_G / (ISA_R * ISA_L))


# ------------------------------------------------------------------ reading the state


def _pressure(p: dict[str, Any], ctx: Ctx) -> float:
    if p.get("p") is not None and p.get("altitude") is not None:
        raise ToolError("INVALID_ARGUMENT", "Give p or altitude, not both.")
    if p.get("altitude") is not None:
        z = need(p, "altitude", "length", positive=False)
        if not -500 <= z <= 11000:
            raise ToolError(
                "INVALID_ARGUMENT", "altitude between -500 m and 11,000 m (ISO 2533 troposphere)."
            )
        ctx.warn(f"Pressure from the ISO 2533 standard atmosphere at {fmt_sig(z, 6)} m.")
        return p_altitude(z)
    v = qty(p, "p", "pressure", default=P_STD)
    assert v is not None
    return v


def _ratio(p: dict[str, Any], key: str) -> float | None:
    """Relative humidity: fraction 0-1 or [value, "%"]."""
    raw = p.get(key)
    if raw is None:
        return None
    if isinstance(raw, list | tuple) and len(raw) == 2 and raw[1] == "%":
        val = raw[0]
        if isinstance(val, bool) or not isinstance(val, int | float):
            raise ToolError("INVALID_ARGUMENT", f"params.{key} must be a number.")
        v = float(val) / 100
    else:
        v = qty(p, key, None, positive=False)
        assert v is not None
        if v > 1:
            raise ToolError(
                "INVALID_ARGUMENT", f'params.{key} is a fraction from 0 to 1 (0.6), or [60, "%"].'
            )
    if not 0 <= v <= 1:
        raise ToolError(
            "INVALID_ARGUMENT", f"params.{key} must be between 0 and 1 (between 0% and 100%)."
        )
    return v


def _hum(p: dict[str, Any], key: str) -> float | None:
    """Humidity ratio W: kg/kg or [value, "g/kg"]."""
    raw = p.get(key)
    if raw is None:
        return None
    if isinstance(raw, list | tuple) and len(raw) == 2 and raw[1] in ("g/kg", "kg/kg"):
        val = raw[0]
        if isinstance(val, bool) or not isinstance(val, int | float):
            raise ToolError("INVALID_ARGUMENT", f"params.{key} must be a number.")
        v = float(val) / (1000.0 if raw[1] == "g/kg" else 1.0)
    else:
        v = qty(p, key, None, positive=False)
        assert v is not None
    if v < 0:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} (humidity ratio) cannot be negative.")
    return v


def _t_c(p: dict[str, Any], key: str, *, req: bool = True) -> float | None:
    T = temperature(p, key, need=req)
    if T is None:
        return None
    if not T_LO <= T <= T_HI:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: psychrometrics here holds from 0 °C to 200 °C (below 0 °C ice would be needed).",
        )
    return T - 273.15


def _state_w(p: dict[str, Any], sfx: str, t: float, pt: float) -> float:
    """W from t and one of RH, W, T_wb, T_dp, h (keys with the suffix sfx: '', '1', '2')."""
    keys = [k + sfx for k in ("RH", "W", "T_wb", "T_dp", "h")]
    given = [k for k in keys if p.get(k) is not None]
    if len(given) != 1:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"With T{sfx} only one of these is needed: {', '.join(keys)}.",
            '{"mode": "state", "T": [25, "C"], "RH": 0.5}',
        )
    k = given[0][: len(given[0]) - len(sfx)] if sfx else given[0]
    if k == "RH":
        rh = _ratio(p, "RH" + sfx)
        assert rh is not None
        pw = rh * pws(t + 273.15)
        if pw >= pt:
            raise ToolError(
                "INVALID_ARGUMENT", "p_w >= p: at this pressure water boils, no moist air."
            )
        return w_from_pw(pw, pt)
    if k == "W":
        w = _hum(p, "W" + sfx)
        assert w is not None
        return w
    if k == "h":
        w = w_from_h(t, need(p, "h" + sfx, "specific_energy", positive=False) / 1000.0)
        if w < 0:
            raise ToolError(
                "INVALID_ARGUMENT", f"h{sfx} is below the enthalpy of dry air at T{sfx}."
            )
        return w
    tx = _t_c(p, k + sfx)
    assert tx is not None
    if tx > t + 1e-12:
        raise ToolError(
            "INVALID_ARGUMENT", f"{k}{sfx} cannot exceed the dry-bulb temperature T{sfx}."
        )
    if k == "T_dp":
        return w_from_pw(pws(tx + 273.15), pt)
    return w_from_wetbulb(t, tx, pt)


def describe(t: float, w: float, pt: float, ctx: Ctx, sfx: str = "") -> dict[str, Any]:
    """All the quantities of the state (t in °C, W in kg/kg, pt in Pa)."""
    T = t + 273.15
    p_s = pws(T)
    pw = pw_from_w(w, pt)
    ws = w_from_pw(p_s, pt) if p_s < pt else math.inf
    rh = pw / p_s
    if rh > 1 + 1e-9:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"State{sfx} beyond saturation (relative humidity {fmt_sig(rh * 100, 4)}%): there is fog, the model does not hold.",
        )
    v = R_DA * T * (1 + w / EPS) / pt
    tdp = dewpoint(pw)
    twb = wetbulb(t, w, pt)
    if tdp is None:
        ctx.warn(f"Dew point{sfx} below 0 °C (frost): outside the range, not computed.")
    if twb is None:
        ctx.warn(f"Wet bulb{sfx} below 0 °C: outside the range, not computed.")
    return {
        "T": T,
        "T_C": t,
        "p": pt,
        "W": w,
        "RH": rh,
        "T_dp": tdp,
        "T_dp_C": None if tdp is None else tdp - 273.15,
        "T_wb": None if twb is None else twb + 273.15,
        "T_wb_C": twb,
        "h": enthalpy(t, w) * 1000.0,
        "v": v,
        "rho": (1 + w) / v,
        "p_w": pw,
        "p_ws": p_s,
        "mu": w / ws,
        "q": w / (1 + w),
        "abs_humidity": w / v,
    }


# ------------------------------------------------------------------ eng_calc operation

_S = ("RH", "W", "T_wb", "T_dp", "h")
_PS: dict[str, set[str]] = {
    "state": {"T", *_S, "p", "altitude"},
    "mixing": {
        "T1",
        "T2",
        "m1",
        "m2",
        "V1",
        "V2",
        "p",
        "altitude",
        *(k + n for k in _S for n in "12"),
    },
    "heating_cooling": {"T1", "T2", "RH2", "m", "V1", "p", "altitude", *(k + "1" for k in _S)},
    "evaporative_cooling": {"T1", "eta", "m", "V1", "p", "altitude", *(k + "1" for k in _S)},
}
_UNITS = {
    "T": "K",
    "T_C": "C",
    "p": "Pa",
    "W": "kg/kg",
    "RH": "1",
    "T_dp": "K",
    "T_dp_C": "C",
    "T_wb": "K",
    "T_wb_C": "C",
    "h": "J/kg",
    "v": "m3/kg",
    "rho": "kg/m3",
    "p_w": "Pa",
    "p_ws": "Pa",
    "mu": "1",
    "q": "kg/kg",
    "abs_humidity": "kg/m3",
    "m_dry_air": "kg/s",
    "Q": "W",
    "q_per_kg": "J/kg",
    "condensate": "kg/s",
    "condensate_per_kg": "kg/kg",
    "water_added": "kg/s",
    "water_added_per_kg": "kg/kg",
}


@op("psychrometrics", ("mode", *sorted(set().union(*_PS.values()))))
def _psychrometrics(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_PS))
    only(p, mode, _PS[mode])
    pt = _pressure(p, ctx)
    fn: Callable[[dict[str, Any], Ctx, float], dict[str, Any]] = {
        "state": _state,
        "mixing": _mixing,
        "heating_cooling": _heating_cooling,
        "evaporative_cooling": _evaporative,
    }[mode]
    out = fn(p, ctx, pt)
    units = {}
    for k in out:
        base = k[:-1] if k[-1:] in ("1", "2", "3") and k[:-1] in _UNITS else k
        if base in _UNITS:
            units[k] = _UNITS[base]
    ctx.units = units_for(out, units)
    return out


def _txt(d: dict[str, Any]) -> str:
    return (
        f"{fmt_sig(d['T_C'], 4)} °C, RH {fmt_sig(d['RH'] * 100, 4)}%, W = {fmt_sig(d['W'] * 1000, 5)} g/kg, "
        f"h = {fmt_sig(d['h'] / 1000, 5)} kJ/kg"
    )


_FORMULA = (
    "W = 0.621945 p_w / (p - p_w); h = 1.006 t + W (2501 + 1.86 t) kJ/kg dry air; "
    "v = 287.042 T (1 + W / 0.621945) / p; p_ws from IF97 Eq. 30; wet bulb from the adiabatic balance"
)


def _state(p: dict[str, Any], ctx: Ctx, pt: float) -> dict[str, Any]:
    t = _t_c(p, "T")
    assert t is not None
    out = describe(t, _state_w(p, "", t, pt), pt, ctx)
    ctx.formula = _FORMULA
    ctx.formatted = f"Moist air at {fmt_sig(pt, 6)} Pa: {_txt(out)}."
    return out


def _flow(p: dict[str, Any], mkey: str, vkey: str, v_spec: float) -> float | None:
    if p.get(mkey) is not None and p.get(vkey) is not None:
        raise ToolError("INVALID_ARGUMENT", f"Give {mkey} or {vkey}, not both.")
    if p.get(vkey) is not None:
        return need(p, vkey, "flow_rate") / v_spec
    return qty(p, mkey, "mass_flow_rate", need=False)


def _suffix(d: dict[str, Any], n: str) -> dict[str, Any]:
    return {k + n: v for k, v in d.items() if k != "p"}


def _mixing(p: dict[str, Any], ctx: Ctx, pt: float) -> dict[str, Any]:
    t1, t2 = _t_c(p, "T1"), _t_c(p, "T2")
    assert t1 is not None and t2 is not None
    s1 = describe(t1, _state_w(p, "1", t1, pt), pt, ctx, " 1")
    s2 = describe(t2, _state_w(p, "2", t2, pt), pt, ctx, " 2")
    m1, m2 = _flow(p, "m1", "V1", s1["v"]), _flow(p, "m2", "V2", s2["v"])
    if m1 is None or m2 is None:
        raise ToolError(
            "INVALID_ARGUMENT", "Mixing needs the dry-air mass flows m1 and m2 (or V1 and V2)."
        )
    w3 = (m1 * s1["W"] + m2 * s2["W"]) / (m1 + m2)
    h3 = (m1 * s1["h"] + m2 * s2["h"]) / (m1 + m2) / 1000.0
    t3 = t_from_hw(h3, w3)
    s3 = describe(t3, w3, pt, ctx, " 3")
    out = {
        "p": pt,
        **_suffix(s1, "1"),
        **_suffix(s2, "2"),
        **_suffix(s3, "3"),
        "m_dry_air": m1 + m2,
    }
    ctx.formula = (
        "W3 = (m1 W1 + m2 W2)/(m1 + m2); h3 = (m1 h1 + m2 h2)/(m1 + m2); t3 from h3 and W3; "
        + _FORMULA
    )
    ctx.formatted = f"Mixture: {_txt(s3)}."
    return out


def _heating_cooling(p: dict[str, Any], ctx: Ctx, pt: float) -> dict[str, Any]:
    t1, t2 = _t_c(p, "T1"), _t_c(p, "T2")
    assert t1 is not None and t2 is not None
    w1 = _state_w(p, "1", t1, pt)
    s1 = describe(t1, w1, pt, ctx, " 1")
    rh2 = _ratio(p, "RH2")
    if rh2 is not None and t2 >= t1:  # D92: heating with RH2 gave condensate
        raise ToolError(
            "INVALID_ARGUMENT",
            "RH2 is for a dehumidifying coil (T2 < T1): heating does not condense water.",
        )
    w_sat2 = w_from_pw(pws(t2 + 273.15), pt)
    if w1 <= w_sat2 and rh2 is None:
        w2, cond = w1, 0.0
        ctx.formula = "Sensible: constant W; q = h2 - h1"
    else:
        w2 = w_from_pw((1.0 if rh2 is None else rh2) * pws(t2 + 273.15), pt)
        if w2 > w1:
            raise ToolError(
                "INVALID_ARGUMENT",
                "RH2 would give more water than the inlet: a humidifier is needed.",
            )
        cond = w1 - w2
        ctx.formula = (
            "Cooling with dehumidification: q = (h2 - h1) + (W1 - W2) h_w(t2), h_w = 4.186 t2"
        )
        if rh2 is None:
            ctx.warn("Saturated outlet (RH2 = 1): give RH2 for a real coil (usually 0.9-0.95).")
    s2 = describe(t2, w2, pt, ctx, " 2")
    q = (s2["h"] - s1["h"]) / 1000.0 + cond * CP_W * t2
    out = {
        "p": pt,
        **_suffix(s1, "1"),
        **_suffix(s2, "2"),
        "q_per_kg": q * 1000.0,
        "condensate_per_kg": cond,
    }
    m = _flow(p, "m", "V1", s1["v"])
    if m is not None:
        out |= {"m_dry_air": m, "Q": m * q * 1000.0, "condensate": m * cond}
    ctx.formula += "; " + _FORMULA
    ctx.formatted = (
        f"From {fmt_sig(t1, 4)} °C to {fmt_sig(t2, 4)} °C: q = {fmt_sig(q, 5)} kJ/kg of dry air "
        f"({'heat removed' if q < 0 else 'heat added'}), condensate {fmt_sig(cond * 1000, 4)} g/kg."
    )
    return out


def _evaporative(p: dict[str, Any], ctx: Ctx, pt: float) -> dict[str, Any]:
    t1 = _t_c(p, "T1")
    assert t1 is not None
    s1 = describe(t1, _state_w(p, "1", t1, pt), pt, ctx, " 1")
    if s1["T_wb_C"] is None:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Wet bulb below 0 °C: evaporative cooling outside the range.",
        )
    eta = qty(p, "eta", None, default=1.0)
    assert eta is not None
    if eta > 1:
        raise ToolError(
            "INVALID_ARGUMENT", "params.eta (saturation effectiveness) must be between 0 and 1."
        )
    twb = s1["T_wb_C"]
    t2 = t1 - eta * (t1 - twb)
    w2 = w_from_wetbulb(t2, twb, pt)
    s2 = describe(t2, w2, pt, ctx, " 2")
    out = {"p": pt, **_suffix(s1, "1"), **_suffix(s2, "2"), "water_added_per_kg": w2 - s1["W"]}
    m = _flow(p, "m", "V1", s1["v"])
    if m is not None:
        out |= {"m_dry_air": m, "water_added": m * (w2 - s1["W"])}
    ctx.formula = "Adiabatic saturation: constant wet bulb; t2 = t1 - eta (t1 - t_wb); " + _FORMULA
    ctx.formatted = f"Evaporative cooling: from {fmt_sig(t1, 4)} °C to {_txt(s2)}."
    return out
