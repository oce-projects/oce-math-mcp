"""eng_calc: water and steam with IAPWS-IF97 (D50).

Industrial formulation IAPWS-IF97 (IAPWS R7-97(2012)); the coefficients are in `if97_data.py`,
generated from the official PDF. Regions: 1 liquid, 2 vapour, 3 zone near the critical point
(Helmholtz free energy in rho and T), 4 saturation (Eq. 30 and 31), 5 high-temperature vapour.
Validity: 273.15 K <= T <= 1,073.15 K with p <= 100 MPa; 1,073.15 K < T <= 2,273.15 K with p <= 50 MPa.

Internal units of IF97: MPa, K, kJ/kg, m3/kg. The result is in SI (Pa, J/kg, J/(kg K), m3/kg).
Choices (D50):
- the density of region 3 for (p, T) is found with brentq on the isotherm (no backward
  equations of 2005); the liquid branch is the densest root, the vapour branch the least dense;
- saturation in region 3 (623.15 K <= T <= Tc) uses p_s from Eq. 30 and the two roots
  of Eq. 28, as in Table 44 of the document;
- the inputs (p, h), (p, s), (p, v), (T, v) are inverted with brentq on the basic equations, from the
  liquid or vapour side according to saturation; no backward equations T(p, h), T(p, s).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ..envelope import Ctx
from ..errors import ToolError
from . import if97_data as D
from .eng_base import choice, fmt_sig, need, only, op, qty, temperature, units_for

R = 0.461526  # kJ/(kg K), Eq. (1)
TC, PC, RHOC = 647.096, 22.064, 322.0  # Eq. (2), (3), (4)
T_MIN, T_13, T_B23_MAX, T_25, T_MAX = 273.15, 623.15, 863.15, 1073.15, 2273.15
P_MAX, P_MAX5 = 100.0, 50.0  # MPa
P_TRIPLE_EQ31 = 611.213e-6  # MPa: Eq. (31) extrapolated to 273.15 K (text of the document)

State = dict[str, Any]

# ------------------------------------------------------------------ basic equations


def p_b23(T: float) -> float:
    """Boundary between regions 2 and 3, Eq. (5): p in MPa from T in K."""
    n = D.B23
    return n[0] + n[1] * T + n[2] * T * T


def t_b23(p: float) -> float:
    """Boundary between regions 2 and 3, Eq. (6): T in K from p in MPa."""
    n = D.B23
    return n[3] + math.sqrt((p - n[4]) / n[2])


def psat(T: float) -> float:
    """Saturation pressure in MPa, Eq. (30), for 273.15 K <= T <= 647.096 K."""
    n = D.R4
    th = T + n[8] / (T - n[9])
    a = th * th + n[0] * th + n[1]
    b = n[2] * th * th + n[3] * th + n[4]
    c = n[5] * th * th + n[6] * th + n[7]
    return (2 * c / (-b + math.sqrt(b * b - 4 * a * c))) ** 4


def tsat(p: float) -> float:
    """Saturation temperature in K, Eq. (31), for 611.213 Pa <= p <= 22.064 MPa."""
    n = D.R4
    be = p**0.25
    e = be * be + n[2] * be + n[5]
    f = n[0] * be * be + n[3] * be + n[6]
    g = n[1] * be * be + n[4] * be + n[7]
    d = 2 * g / (-f - math.sqrt(f * f - 4 * e * g))
    return (n[9] + d - math.sqrt((n[9] + d) ** 2 - 4 * (n[8] + n[9] * d))) / 2


def _gibbs_props(
    p: float, T: float, pi: float, tau: float, d: tuple[float, ...], reg: int
) -> State:
    """Properties from gamma(pi, tau) and derivatives (g, g_p, g_pp, g_t, g_tt, g_pt), Tables 3 and 12."""
    g, gp, gpp, gt, gtt, gpt = d
    rt = R * T
    v = rt * pi * gp / p / 1000.0
    a = (gp - tau * gpt) ** 2
    return {
        "region": reg,
        "p": p,
        "T": T,
        "v": v,
        "rho": 1.0 / v,
        "h": rt * tau * gt,
        "u": rt * (tau * gt - pi * gp),
        "s": R * (tau * gt - g),
        "cp": -R * tau * tau * gtt,
        "cv": R * (-tau * tau * gtt + a / gpp),
        "w": math.sqrt(1000.0 * rt * gp * gp / (a / (tau * tau * gtt) - gpp)),
    }


def region1(p: float, T: float) -> State:
    """Region 1, Eq. (7): pi = p / 16.53 MPa, tau = 1386 K / T."""
    pi, tau = p / 16.53, 1386.0 / T
    a, b = 7.1 - pi, tau - 1.222
    g = gp = gpp = gt = gtt = gpt = 0.0
    for i, j, n in D.R1:
        ai, bj = a**i, b**j
        g += n * ai * bj
        if i:
            gp -= n * i * a ** (i - 1) * bj
            gpp += n * i * (i - 1) * a ** (i - 2) * bj
            if j:
                gpt -= n * i * a ** (i - 1) * j * b ** (j - 1)
        if j:
            gt += n * ai * j * b ** (j - 1)
            gtt += n * ai * j * (j - 1) * b ** (j - 2)
    return _gibbs_props(p, T, pi, tau, (g, gp, gpp, gt, gtt, gpt), 1)


def _ideal_res(
    pi: float,
    tau: float,
    ideal: tuple[tuple[int, float], ...],
    res: tuple[tuple[int, int, float], ...],
    shift: float,
) -> tuple[float, ...]:
    g, gp, gpp = math.log(pi), 1.0 / pi, -1.0 / (pi * pi)
    gt = gtt = gpt = 0.0
    for j, n in ideal:
        g += n * tau**j
        gt += n * j * tau ** (j - 1)
        gtt += n * j * (j - 1) * tau ** (j - 2)
    t = tau - shift
    for i, j, n in res:
        pi_i, t_j = pi**i, t**j
        g += n * pi_i * t_j
        gp += n * i * pi ** (i - 1) * t_j
        gpp += n * i * (i - 1) * pi ** (i - 2) * t_j
        gt += n * pi_i * j * t ** (j - 1)
        gtt += n * pi_i * j * (j - 1) * t ** (j - 2)
        gpt += n * i * pi ** (i - 1) * j * t ** (j - 1)
    return g, gp, gpp, gt, gtt, gpt


def region2(p: float, T: float) -> State:
    """Region 2, Eq. (15)-(17): pi = p / 1 MPa, tau = 540 K / T, residual part in (tau - 0.5)."""
    pi, tau = p, 540.0 / T
    return _gibbs_props(p, T, pi, tau, _ideal_res(pi, tau, D.R2_IDEAL, D.R2_RES, 0.5), 2)


def region5(p: float, T: float) -> State:
    """Region 5, Eq. (32)-(34): pi = p / 1 MPa, tau = 1000 K / T."""
    pi, tau = p, 1000.0 / T
    return _gibbs_props(p, T, pi, tau, _ideal_res(pi, tau, D.R5_IDEAL, D.R5_RES, 0.0), 5)


def region3(rho: float, T: float) -> State:
    """Region 3, Eq. (28): phi(delta, tau) with delta = rho / 322 kg/m3, tau = 647.096 K / T."""
    delta, tau = rho / RHOC, TC / T
    n1 = D.R3_N1
    f, fd, fdd = n1 * math.log(delta), n1 / delta, -n1 / (delta * delta)
    ft = ftt = fdt = 0.0
    for i, j, n in D.R3:
        di, tj = delta**i, tau**j
        f += n * di * tj
        fd += n * i * delta ** (i - 1) * tj
        fdd += n * i * (i - 1) * delta ** (i - 2) * tj
        ft += n * di * j * tau ** (j - 1)
        ftt += n * di * j * (j - 1) * tau ** (j - 2)
        fdt += n * i * delta ** (i - 1) * j * tau ** (j - 1)
    rt = R * T
    dfd = delta * fd
    a = (dfd - delta * tau * fdt) ** 2
    b = 2 * dfd + delta * delta * fdd
    return {
        "region": 3,
        "p": rho * rt * dfd / 1000.0,
        "T": T,
        "v": 1.0 / rho,
        "rho": rho,
        "h": rt * (tau * ft + dfd),
        "u": rt * tau * ft,
        "s": R * (tau * ft - f),
        "cp": R * (-tau * tau * ftt + a / b),
        "cv": -R * tau * tau * ftt,
        "w": math.sqrt(1000.0 * rt * (b - a / (tau * tau * ftt))),
    }


_R3_I = np.array([i for i, _, _ in D.R3], dtype=float)
_R3_J = np.array([j for _, j, _ in D.R3], dtype=float)
_R3_N = np.array([n for _, _, n in D.R3], dtype=float)
_RHO_GRID = np.linspace(40.0, 900.0, 3441)  # step 0.25 kg/m3


def _p3_grid(T: float) -> np.ndarray:
    """p (MPa) of region 3 on the density grid, to find the roots of the isotherm."""
    delta = _RHO_GRID / RHOC
    c = _R3_N * _R3_I * (TC / T) ** _R3_J
    fd = D.R3_N1 / delta + (delta[:, None] ** (_R3_I - 1.0) * c).sum(axis=1)
    return _RHO_GRID * R * T * delta * fd / 1000.0


def rho3(p: float, T: float, branch: str) -> float:
    """Density of region 3 at (p, T): densest root («liq») or least dense («vap»)."""
    f = _p3_grid(T) - p
    k = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) <= 0)[0]
    if k.size == 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Region 3 of IF97: no density for p = {fmt_sig(p)} MPa, T = {fmt_sig(T)} K.",
        )
    i = int(k[-1] if branch == "liq" else k[0])
    return _root(
        lambda r: region3(r, T)["p"] - p, float(_RHO_GRID[i]), float(_RHO_GRID[i + 1]), 1e-13
    )


def _root(f: Callable[[float], float], a: float, b: float, xtol: float) -> float:
    """brentq with the declared result type (default rtol, 4 times the epsilon)."""
    return float(cast(float, brentq(f, a, b, xtol=xtol)))


# ------------------------------------------------------------------ choice of the region


def _check_pT(p: float, T: float) -> None:
    if T < T_MIN or T > T_MAX:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"T = {fmt_sig(T)} K is outside IF97 (from 273.15 K to 2,273.15 K). Below 0 °C there is ice.",
        )
    if p <= 0 or p > P_MAX or (T > T_25 and p > P_MAX5):
        raise ToolError(
            "INVALID_ARGUMENT",
            f"p = {fmt_sig(p)} MPa is outside IF97: up to 100 MPa if T <= 1,073.15 K, "
            "up to 50 MPa above.",
        )


def liquid(p: float, T: float) -> State:
    """Liquid side (T <= T_sat): region 1 or dense branch of region 3."""
    return region1(p, T) if T <= T_13 else region3(rho3(p, T, "liq"), T)


def vapour(p: float, T: float) -> State:
    """Vapour side (T >= T_sat): region 2, least dense branch of region 3, or region 5."""
    if T > T_25:
        return region5(p, T)
    if T <= T_13 or T > T_B23_MAX or p <= p_b23(T):
        return region2(p, T)
    return region3(rho3(p, T, "vap" if T < TC else "liq"), T)


def state_pT(p: float, T: float) -> State:
    """Single-phase state from (p, T); on the saturation line it chooses the vapour."""
    _check_pT(p, T)
    if T > T_25:
        return region5(p, T)
    if T <= T_13:
        return region1(p, T) if p > psat(T) else region2(p, T)
    if T <= T_B23_MAX and p >= p_b23(T):
        branch = "liq" if T >= TC or p > psat(T) else "vap"
        return region3(rho3(p, T, branch), T)
    return region2(p, T)


def sat_T(T: float) -> tuple[float, State, State]:
    """Saturation from T: p_s (Eq. 30), saturated liquid and vapour."""
    if T < T_MIN or T > TC:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Saturation exists from 273.15 K to 647.096 K (critical point); T = {fmt_sig(T)} K.",
        )
    p = psat(T)
    if T <= T_13:
        return p, region1(p, T), region2(p, T)
    return p, region3(rho3(p, T, "liq"), T), region3(rho3(p, T, "vap"), T)


def sat_p(p: float) -> tuple[float, State, State]:
    """Saturation from p: T_s (Eq. 31), saturated liquid and vapour."""
    if p < P_TRIPLE_EQ31 or p > PC:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Saturation exists from 611.213 Pa to 22.064 MPa (critical point); p = {fmt_sig(p * 1e6)} Pa.",
        )
    T = tsat(p)
    if T <= T_13:
        return T, region1(p, T), region2(p, T)
    return T, region3(rho3(p, T, "liq"), T), region3(rho3(p, T, "vap"), T)


def mix(T: float, p: float, x: float, f: State, g: State) -> State:
    """Liquid-vapour mixture with quality x."""
    out: State = {"region": 4, "p": p, "T": T, "x": x}
    for k in ("v", "h", "u", "s"):
        out[k] = f[k] + x * (g[k] - f[k])
    out["rho"] = 1.0 / out["v"]
    return out


def _inside(lo: float, x: float, hi: float) -> bool:
    """x between the saturated values, with a relative margin of 1e-12 (the results have 15 digits)."""
    tol = 1e-12 * max(abs(lo), abs(hi))
    return lo - tol <= x <= hi + tol


def _quality(lo: float, x: float, hi: float) -> float:
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def _t_top(p: float) -> float:
    return T_MAX if p <= P_MAX5 else T_25


def solve_T(p: float, key: str, target: float, ctx: Ctx | None = None) -> State:
    """State from p and one of h, s, v: mixture if the value falls between saturated liquid and vapour."""
    if p <= 0 or p > P_MAX:
        raise ToolError(
            "INVALID_ARGUMENT", f"p = {fmt_sig(p)} MPa is outside IF97 (up to 100 MPa)."
        )
    fn: Callable[[float, float], State] = state_pT
    lo, hi = T_MIN, _t_top(p)
    if P_TRIPLE_EQ31 <= p < PC:
        Ts, f, g = sat_p(p)
        if _inside(f[key], target, g[key]):
            return mix(Ts, p, _quality(f[key], target, g[key]), f, g)
        if target < f[key]:
            fn, hi = liquid, Ts
        else:
            fn, lo = vapour, Ts
    elif p < P_TRIPLE_EQ31:
        fn = vapour
    if key == "v" and fn is liquid:  # density maximum near 4 °C
        m = cast(
            Any,
            minimize_scalar(
                lambda T: liquid(p, T)["v"], bounds=(T_MIN, min(hi, 320.0)), method="bounded"
            ),
        )
        t_rho_max = float(m.x)
        if t_rho_max > T_MIN + 1e-6 and target <= liquid(p, T_MIN)["v"] and ctx is not None:
            ctx.warn(
                f"With this v there is another solution below {fmt_sig(t_rho_max - 273.15)} °C (density maximum)."
            )
        lo = t_rho_max

    def err(T: float) -> float:
        return fn(p, T)[key] - target

    a, b = err(lo), err(hi)
    if a > 0 or b < 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"No IF97 state at p = {fmt_sig(p)} MPa with {key} = {fmt_sig(target)}: "
            f"the range goes from {fmt_sig(fn(p, lo)[key])} to {fmt_sig(fn(p, hi)[key])} (IF97 units).",
        )
    return fn(p, _root(err, lo, hi, 1e-11))


def solve_p(T: float, v: float) -> State:
    """State from T and v: mixture if v falls between saturated liquid and vapour, otherwise brentq on ln p."""
    _check_pT(1.0, T)
    fn: Callable[[float, float], State] = state_pT
    p_hi = P_MAX if T <= T_25 else P_MAX5
    p_lo = min(1e-3, R * T / (v * 1000.0) * 0.1)
    if T <= TC:
        ps, f, g = sat_T(T)
        if _inside(f["v"], v, g["v"]):
            return mix(T, ps, _quality(f["v"], v, g["v"]), f, g)
        if v > g["v"]:
            fn, p_hi = vapour, ps
        else:
            fn, p_lo = liquid, ps

    def err(lnp: float) -> float:
        return fn(min(math.exp(lnp), p_hi), T)["v"] - v

    a, b = err(math.log(p_lo)), err(math.log(p_hi))
    if a < 0 or b > 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"No IF97 state at T = {fmt_sig(T)} K with v = {fmt_sig(v)} m3/kg (pressure beyond the limits).",
        )
    lnp = _root(err, math.log(p_lo), math.log(p_hi), 1e-14)
    return fn(min(math.exp(lnp), p_hi), T)


# ------------------------------------------------------------------ eng_calc operation

_WS: dict[str, set[str]] = {
    "state": {"p", "T", "h", "s", "x", "v", "rho"},
    "saturation": {"p", "T"},
    "rankine": {
        "p_high",
        "p_low",
        "T_high",
        "p_reheat",
        "T_reheat",
        "eta_turbine",
        "eta_pump",
        "P_net",
        "m_dot",
    },
}
_PAIRS = {
    frozenset(("p", "T")),
    frozenset(("p", "x")),
    frozenset(("T", "x")),
    frozenset(("p", "h")),
    frozenset(("p", "s")),
    frozenset(("p", "v")),
    frozenset(("T", "v")),
}
_PHASE = {1: "compressed liquid", 2: "superheated vapour", 5: "high-temperature vapour"}

_UNITS = {
    "p": "Pa",
    "p_bar": "bar",
    "T": "K",
    "T_C": "C",
    "v": "m3/kg",
    "rho": "kg/m3",
    "h": "J/kg",
    "u": "J/kg",
    "s": "J/(kg*K)",
    "cp": "J/(kg*K)",
    "cv": "J/(kg*K)",
    "w": "m/s",
    "x": "1",
    "region": "1",
    "T_sat": "K",
    "p_sat": "Pa",
    "v_f": "m3/kg",
    "v_g": "m3/kg",
    "rho_f": "kg/m3",
    "rho_g": "kg/m3",
    "h_f": "J/kg",
    "h_g": "J/kg",
    "h_fg": "J/kg",
    "u_f": "J/kg",
    "u_g": "J/kg",
    "s_f": "J/(kg*K)",
    "s_g": "J/(kg*K)",
    "s_fg": "J/(kg*K)",
    "cp_f": "J/(kg*K)",
    "cp_g": "J/(kg*K)",
    "w_turbine": "J/kg",
    "w_pump": "J/kg",
    "w_net": "J/kg",
    "q_in": "J/kg",
    "q_out": "J/kg",
    "efficiency": "1",
    "efficiency_carnot": "1",
    "back_work_ratio": "1",
    "x_turbine_exit": "1",
    "T_states": "K",
    "p_states": "Pa",
    "h_states": "J/kg",
    "s_states": "J/(kg*K)",
    "x_states": "1",
    "m_dot": "kg/s",
    "P_net": "W",
    "Q_in": "W",
    "Q_out": "W",
}


def _si(st: State) -> dict[str, Any]:
    """From IF97 units to SI."""
    out: dict[str, Any] = {"region": st["region"], "p": st["p"] * 1e6, "p_bar": st["p"] * 10.0}
    out["T"], out["T_C"] = st["T"], st["T"] - 273.15
    if "x" in st:
        out["x"] = st["x"]
    out["v"], out["rho"] = st["v"], st["rho"]
    for k in ("h", "u", "s", "cp", "cv"):
        if k in st:
            out[k] = st[k] * 1000.0
    if "w" in st:
        out["w"] = st["w"]
    return out


def _phase(st: State) -> str:
    if st["region"] == 4:
        return "liquid-vapour mixture"
    T, p = st["T"], st["p"]
    if T >= TC and p >= PC:
        return "supercritical fluid"
    if st["region"] == 3:
        return "compressed liquid" if T < TC and p > psat(T) else "superheated vapour"
    return _PHASE[st["region"]]


@op("water_steam", ("mode", *sorted(set().union(*_WS.values()))))
def _water_steam(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_WS))
    only(p, mode, _WS[mode])
    out = {"state": _state, "saturation": _saturation, "rankine": _rankine}[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _mpa(p: dict[str, Any], key: str) -> float:
    return need(p, key, "pressure") / 1e6


def _state(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    given = {k for k in _WS["state"] if p.get(k) is not None}
    if "rho" in given and "v" in given:
        raise ToolError("INVALID_ARGUMENT", "Give v or rho, not both.")
    pair = frozenset(("v" if k == "rho" else k) for k in given)
    if pair not in _PAIRS:
        raise ToolError(
            "INVALID_ARGUMENT",
            "The water state needs two quantities: (p, T), (p, x), (T, x), (p, h), (p, s), (p, v) "
            f"or (T, v); rho instead of v. Given: {', '.join(sorted(given)) or 'none'}.",
            '{"mode": "state", "p": [3, "MPa"], "T": [400, "C"]}',
        )
    x = qty(p, "x", None, need=False, positive=False)
    if x is not None and not 0 <= x <= 1:
        raise ToolError("INVALID_ARGUMENT", "params.x (steam quality) must be between 0 and 1.")
    v = None
    if "v" in pair:
        v = need(p, "v", "specific_volume") if "v" in given else 1.0 / need(p, "rho", "density")
    T = temperature(p, "T", need=False)
    if pair == frozenset(("p", "T")):
        assert T is not None
        pm = _mpa(p, "p")
        st = state_pT(pm, T)
        if T <= TC and abs(pm / psat(T) - 1) < 1e-4:
            ctx.warn(
                "p is less than 0.01% away from saturation at this T: if it is a mixture of liquid and vapour "
                "the quality x is needed."
            )
        ctx.formula = "IF97: region from (p, T); v, h, u, s, cp, cv, w from the derivatives of gamma (or phi in region 3)"
    elif "x" in pair:
        assert x is not None
        if T is not None:
            ps, f, g = sat_T(T)
            st = mix(T, ps, x, f, g)
        else:
            Ts, f, g = sat_p(_mpa(p, "p"))
            st = mix(Ts, _mpa(p, "p"), x, f, g)
        ctx.formula = "y = y_f + x (y_g - y_f) per v, h, u, s; p_s(T) Eq. 30, T_s(p) Eq. 31"
    elif pair == frozenset(("T", "v")):
        assert T is not None and v is not None
        st = solve_p(T, v)
        ctx.formula = "IF97: mixture if v_f <= v <= v_g, otherwise p from brentq on v(p, T) = v"
    else:
        key = next(k for k in pair if k != "p")
        kind = {"h": "specific_energy", "s": "specific_heat"}.get(key)
        target = v if key == "v" else need(p, key, kind, positive=False) / 1000.0
        assert target is not None
        st = solve_T(_mpa(p, "p"), key, target, ctx)
        ctx.formula = f"IF97: mixture if {key}_f <= {key} <= {key}_g, otherwise T from brentq on {key}(p, T) = {key}"
    out = _si(st)
    out["phase"] = _phase(st)
    txt = (
        f"Water at {fmt_sig(st['p'], 6)} MPa and {fmt_sig(st['T'] - 273.15, 6)} °C: {out['phase']}"
    )
    if "x" in st:
        txt += f", quality x = {fmt_sig(st['x'], 5)}"
    else:
        txt += f" (region {st['region']})"
    ctx.formatted = (
        txt
        + f"; h = {fmt_sig(st['h'], 6)} kJ/kg, s = {fmt_sig(st['s'], 6)} kJ/(kg K), v = {fmt_sig(st['v'], 6)} m3/kg."
    )
    return out


def _saturation(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    T = temperature(p, "T", need=False)
    if (T is None) == (p.get("p") is None):
        raise ToolError("INVALID_ARGUMENT", "Saturation needs only one of p and T.")
    if T is not None:
        ps, f, g = sat_T(T)
        Ts = T
        ctx.formula = (
            "p_s(T) Eq. 30 of IF97; saturated liquid and vapour from regions 1 and 2 (or 3)"
        )
    else:
        ps = _mpa(p, "p")
        Ts, f, g = sat_p(ps)
        ctx.formula = (
            "T_s(p) Eq. 31 of IF97; saturated liquid and vapour from regions 1 and 2 (or 3)"
        )
    out: dict[str, Any] = {"T_sat": Ts, "T_C": Ts - 273.15, "p_sat": ps * 1e6, "p_bar": ps * 10.0}
    out |= {"v_f": f["v"], "v_g": g["v"], "rho_f": f["rho"], "rho_g": g["rho"]}
    for k in ("h", "u", "s"):
        out[f"{k}_f"], out[f"{k}_g"] = f[k] * 1000.0, g[k] * 1000.0
    out["h_fg"], out["s_fg"] = out["h_g"] - out["h_f"], out["s_g"] - out["s_f"]
    out["cp_f"], out["cp_g"] = f["cp"] * 1000.0, g["cp"] * 1000.0
    if Ts > TC - 0.5:
        ctx.warn("Near the critical point: liquid and vapour almost equal, cp very large.")
    ctx.formatted = (
        f"Saturation at {fmt_sig(ps, 6)} MPa and {fmt_sig(Ts - 273.15, 6)} °C: h_f = {fmt_sig(f['h'], 6)} kJ/kg, "
        f"h_g = {fmt_sig(g['h'], 6)} kJ/kg, h_fg = {fmt_sig(g['h'] - f['h'], 6)} kJ/kg."
    )
    return out


def _eff(p: dict[str, Any], key: str) -> float:
    v = qty(p, key, None, default=1.0)
    assert v is not None
    if v > 1:
        raise ToolError(
            "INVALID_ARGUMENT", f"params.{key} (isentropic efficiency) must be between 0 and 1."
        )
    return v


def _rankine(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    ph, pl = _mpa(p, "p_high"), _mpa(p, "p_low")
    if pl >= ph:
        raise ToolError("INVALID_ARGUMENT", "p_low (condenser) must be less than p_high (boiler).")
    if pl >= PC:
        raise ToolError(
            "INVALID_ARGUMENT", "p_low must be below the critical pressure (22.064 MPa)."
        )
    et, ep = _eff(p, "eta_turbine"), _eff(p, "eta_pump")
    reheat = p.get("p_reheat") is not None or p.get("T_reheat") is not None
    if reheat and (p.get("p_reheat") is None or p.get("T_reheat") is None):
        raise ToolError("INVALID_ARGUMENT", "Reheat needs p_reheat and T_reheat together.")
    # 1: saturated liquid at the condenser outlet
    T1, s1f, _ = sat_p(pl)
    st = [s1f]
    # 2: pump outlet
    h2s = solve_T(ph, "s", s1f["s"])["h"]
    st.append(solve_T(ph, "h", s1f["h"] + (h2s - s1f["h"]) / ep))
    # 3: boiler outlet
    t_high = temperature(p, "T_high", need=False)
    if t_high is None:
        if ph >= PC:
            raise ToolError("INVALID_ARGUMENT", "Above the critical pressure T_high is needed.")
        st.append(sat_p(ph)[2])
    else:
        if ph < PC and t_high <= tsat(ph):
            raise ToolError(
                "INVALID_ARGUMENT",
                f"T_high must exceed the saturation at p_high ({fmt_sig(tsat(ph) - 273.15, 6)} °C).",
            )
        st.append(state_pT(ph, t_high))
    q_in = st[2]["h"] - st[1]["h"]

    def expand(inlet: State, pout: float) -> State:
        """Turbine: h_out = h_in - eta_t (h_in - h_out,s), with h_out,s at constant s."""
        h_s = solve_T(pout, "s", inlet["s"])["h"]
        return solve_T(pout, "h", inlet["h"] - et * (inlet["h"] - h_s))

    if reheat:
        pr = _mpa(p, "p_reheat")
        t_re = temperature(p, "T_reheat")
        assert t_re is not None
        if not pl < pr < ph:
            raise ToolError("INVALID_ARGUMENT", "p_reheat must be between p_low and p_high.")
        if pr < PC and t_re <= tsat(pr):
            raise ToolError("INVALID_ARGUMENT", "T_reheat must exceed the saturation at p_reheat.")
        st.append(expand(st[2], pr))  # 4: high-pressure outlet
        st.append(state_pT(pr, t_re))  # 5: reheater outlet
        q_in += st[4]["h"] - st[3]["h"]
    st.append(expand(st[-1], pl))  # last: turbine outlet
    w_t = st[2]["h"] - st[-1]["h"] + (st[4]["h"] - st[3]["h"] if reheat else 0.0)
    w_p = st[1]["h"] - st[0]["h"]
    w_net = w_t - w_p
    q_out = st[-1]["h"] - st[0]["h"]
    t_max = max(s["T"] for s in st)
    out: dict[str, Any] = {
        "w_turbine": w_t * 1000.0,
        "w_pump": w_p * 1000.0,
        "w_net": w_net * 1000.0,
        "q_in": q_in * 1000.0,
        "q_out": q_out * 1000.0,
        "efficiency": w_net / q_in,
        "efficiency_carnot": 1 - T1 / t_max,
        "back_work_ratio": w_p / w_t,
        "x_turbine_exit": st[-1].get("x"),
        "T_states": [s["T"] for s in st],
        "p_states": [s["p"] * 1e6 for s in st],
        "h_states": [s["h"] * 1000.0 for s in st],
        "s_states": [s["s"] * 1000.0 for s in st],
        "x_states": [s.get("x") for s in st],
    }
    if out["x_turbine_exit"] is None:
        out["x_turbine_exit"] = 1.0 if st[-1]["region"] != 1 else 0.0
        ctx.warn("At the turbine outlet the steam is superheated (quality 1 by convention).")
    elif out["x_turbine_exit"] < 0.88:
        ctx.warn(
            f"Quality at the turbine outlet {fmt_sig(out['x_turbine_exit'], 3)}: below about 0.88 "
            "the droplets erode the blades (rule of thumb)."
        )
    P_net, m_dot = (
        qty(p, "P_net", "power", need=False),
        qty(p, "m_dot", "mass_flow_rate", need=False),
    )
    if P_net is not None and m_dot is not None:
        raise ToolError("INVALID_ARGUMENT", "Give P_net or m_dot, not both.")
    if P_net is not None:
        m_dot = P_net / out["w_net"]
    if m_dot is not None:
        out |= {
            "m_dot": m_dot,
            "P_net": m_dot * out["w_net"],
            "Q_in": m_dot * out["q_in"],
            "Q_out": m_dot * out["q_out"],
        }
    ctx.formula = (
        "Rankine: 1 saturated liquid at p_low; 2 pump h2 = h1 + (h2s - h1)/eta_p; 3 boiler; "
        "turbine h_out = h_in - eta_t (h_in - h_out,s); eta = (w_t - w_p) / q_in"
    )
    ctx.formatted = (
        f"Rankine cycle {fmt_sig(ph, 4)} MPa / {fmt_sig(pl * 1000, 4)} kPa: efficiency "
        f"{fmt_sig(out['efficiency'] * 100, 4)}% (Carnot {fmt_sig(out['efficiency_carnot'] * 100, 4)}%), "
        f"w_net = {fmt_sig(w_net, 5)} kJ/kg, outlet quality {fmt_sig(out['x_turbine_exit'], 4)}."
    )
    return out
