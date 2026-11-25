"""eng_calc: heat exchangers with the e-NTU method, fins, lumped-capacitance transient (D53).

Sources (Wikipedia pages «NTU method», «Fin (extended surface)», «Biot number»):
- e-NTU: C = m cp, Cr = Cmin / Cmax, NTU = U A / Cmin, Q = e Cmin (Th_in - Tc_in); effectiveness for
  parallel flow, counterflow (Cr = 1: NTU / (1 + NTU)), a single fluid changing phase (Cr = 0:
  1 - e^-NTU), crossflow with both unmixed (series of the polynomials P_n), both
  mixed, one mixed (Cmin or Cmax). The inverses (NTU from the effectiveness) are the formulas
  inverted with algebra or, where that is not possible, with brentq. The LMTD F factor and the
  shell-and-tube exchangers are not included: no verified source (D53).
- fins of constant cross-section: m^2 = h P / (k A_c), the 4 tip conditions (convection, adiabatic,
  given temperature, infinite fin), effectiveness, efficiency, overall efficiency of an array.
- transient: Bi = h L_c / k with L_c = V / A, (T - T_inf) / (T_0 - T_inf) = exp(-t / tau),
  tau = rho cp V / (h A); warning above Bi = 0.1.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, only, op, qty, temperature, units_for

_FLOWS = (
    "parallel",
    "counter",
    "crossflow_unmixed",
    "crossflow_mixed",
    "crossflow_one_mixed",
)
_NTU_MAX_SERIES = 500.0

# ------------------------------------------------------------------ effectiveness


def _eps_unmixed(n_tu: float, cr: float) -> float:
    """Both unmixed: 1 - e^-NTU - e^-(1+Cr)NTU sum Cr^n P_n(NTU), terms in logarithms."""
    ln_n, ln_cr, base = math.log(n_tu), math.log(cr), -(1 + cr) * n_tu
    total, n = 0.0, 1
    peak = n_tu * math.sqrt(cr)
    while True:
        j = np.arange(1, n + 1, dtype=float)
        logt = (n + j) * ln_n - gammaln(j + 1) - gammaln(n + 2) + n * ln_cr + base
        term = float(np.sum((n + 1 - j) * np.exp(logt)))
        total += term
        if n > peak + 5 and term <= 1e-17 * total:
            break
        n += 1
        if n > 20000:  # pragma: no cover - NTU limited to 500
            raise ToolError("INVALID_ARGUMENT", "The crossflow series does not converge.")
    return -math.expm1(-n_tu) - total


def effectiveness(flow: str, n_tu: float, cr: float, mixed: str | None = None) -> float:
    """Effectiveness e(NTU, Cr); `mixed` = "cmin" or "cmax" for crossflow with one mixed fluid."""
    if cr == 0:
        return -math.expm1(-n_tu)
    if flow == "parallel":
        return -math.expm1(-n_tu * (1 + cr)) / (1 + cr)
    if flow == "counter":
        if cr == 1:
            return n_tu / (1 + n_tu)
        num = -math.expm1(-n_tu * (1 - cr))
        return num / (1 - cr + cr * num)  # = (1 - e^-x) / (1 - Cr e^-x), stable near Cr = 1
    if flow == "crossflow_unmixed":
        return _eps_unmixed(n_tu, cr)
    if flow == "crossflow_mixed":
        return 1 / (1 / -math.expm1(-n_tu) + cr / -math.expm1(-n_tu * cr) - 1 / n_tu)
    if mixed == "cmin":
        return -math.expm1(math.expm1(-n_tu * cr) / cr)
    return -math.expm1(-cr * -math.expm1(-n_tu)) / cr


def _sup(flow: str, cr: float, mixed: str | None) -> float:
    """Limit effectiveness for NTU -> infinity (not reached)."""
    if cr == 0 or flow in ("counter", "crossflow_unmixed"):
        return 1.0
    if flow in ("parallel", "crossflow_mixed"):
        return 1 / (1 + cr)
    if mixed == "cmin":
        return -math.expm1(-1 / cr)
    return -math.expm1(-cr) / cr


def _root(f: Callable[[float], float], a: float, b: float) -> float:
    return cast(float, cast(Any, brentq)(f, a, b, xtol=1e-300, rtol=1e-15, maxiter=500))


def ntu_from_eps(
    flow: str, eps: float, cr: float, mixed: str | None, ctx: Ctx
) -> tuple[float, float | None]:
    """(NTU, second root for crossflow with both mixed, if there is one)."""
    if flow == "crossflow_mixed" and cr > 0:
        return _ntu_both_mixed(eps, cr, ctx)
    sup = _sup(flow, cr, mixed)
    if eps >= sup:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Effectiveness {fmt_sig(eps, 6)} not reachable: with this flow and Cr = {fmt_sig(cr, 6)} "
            f"the limit for infinite NTU is {fmt_sig(sup, 6)}.",
        )
    if cr == 0:
        return -math.log1p(-eps), None
    if flow == "parallel":
        return -math.log1p(-eps * (1 + cr)) / (1 + cr), None
    if flow == "counter":
        if cr == 1:
            return eps / (1 - eps), None
        return math.log((1 - eps * cr) / (1 - eps)) / (1 - cr), None
    if flow == "crossflow_one_mixed":
        if mixed == "cmin":
            return -math.log1p(cr * math.log1p(-eps)) / cr, None
        return -math.log1p(math.log1p(-eps * cr) / cr), None
    hi = 1.0
    while effectiveness(flow, hi, cr) < eps:
        if hi >= _NTU_MAX_SERIES:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Effectiveness {fmt_sig(eps, 6)} needs NTU above {fmt_sig(_NTU_MAX_SERIES, 3)}: outside the computed range.",
            )
        hi = min(2 * hi, _NTU_MAX_SERIES)
    return _root(lambda n: effectiveness(flow, n, cr) - eps, 1e-300, hi), None


def _ntu_both_mixed(eps: float, cr: float, ctx: Ctx) -> tuple[float, float | None]:
    def f(n: float) -> float:
        return effectiveness("crossflow_mixed", n, cr)

    res = minimize_scalar(
        lambda u: -f(math.exp(u)),
        bounds=(math.log(1e-6), math.log(1e4)),
        method="bounded",
        options={"xatol": 1e-12},
    )
    n_star = math.exp(float(cast(Any, res).x))
    e_max = f(n_star)
    if eps > e_max:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Effectiveness {fmt_sig(eps, 6)} above the maximum of this flow ({fmt_sig(e_max, 6)} at "
            f"NTU = {fmt_sig(n_star, 4)}, Cr = {fmt_sig(cr, 6)}).",
        )
    first = _root(lambda n: f(n) - eps, 1e-300, n_star)
    if eps <= 1 / (1 + cr):
        return first, None
    hi = n_star
    while f(hi) > eps:
        hi *= 2
    second = _root(lambda n: f(n) - eps, n_star, hi)
    ctx.warn(
        f"Crossflow with both mixed: the effectiveness has a maximum, there are two NTU "
        f"({fmt_sig(first, 6)} and {fmt_sig(second, 6)}). The first is the one with the smaller area."
    )
    return first, second


# ------------------------------------------------------------------ heat_exchanger_ntu

_CAP = {"C_h", "m_h", "cp_h", "C_c", "m_c", "cp_c", "phase_change"}
_HX_MODES: dict[str, set[str]] = {
    "effectiveness": {"flow", "mixed", "NTU", "Cr"},
    "ntu": {"flow", "mixed", "epsilon", "Cr"},
    "rating": {"flow", "mixed", "Th_in", "Tc_in", "UA", "U", "A"} | _CAP,
    "sizing": {"flow", "mixed", "Th_in", "Tc_in", "Q", "Th_out", "Tc_out", "U"} | _CAP,
}
_HX_UNITS = {
    "epsilon": "1",
    "NTU": "1",
    "NTU_alt": "1",
    "Cr": "1",
    "C_h": "W/K",
    "C_c": "W/K",
    "C_min": "W/K",
    "C_max": "W/K",
    "UA": "W/K",
    "A": "m2",
    "Q": "W",
    "Q_max": "W",
    "Th_out": "K",
    "Tc_out": "K",
}


def _capacity(p: dict[str, Any], side: str) -> float:
    c, m, cp = f"C_{side}", f"m_{side}", f"cp_{side}"
    if p.get(c) is not None:
        if p.get(m) is not None or p.get(cp) is not None:
            raise ToolError("INVALID_ARGUMENT", f"Give {c} or {m} and {cp}, not all three.")
        return need(p, c, "thermal_conductance")
    if p.get(m) is None or p.get(cp) is None:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"The heat capacity rate of the fluid is needed: {c} (W/K) or {m} and {cp}.",
        )
    return need(p, m, "mass_flow_rate") * need(p, cp, "specific_heat")


def _mixed(p: dict[str, Any], flow: str, allowed: tuple[str, ...]) -> str | None:
    if flow != "crossflow_one_mixed":
        if p.get("mixed") is not None:
            raise ToolError(
                "INVALID_ARGUMENT", "params.mixed is used only with flow 'crossflow_one_mixed'."
            )
        return None
    return choice(p, "mixed", allowed)


def _cr(p: dict[str, Any]) -> float:
    cr = need(p, "Cr", None, positive=False)
    if not 0 <= cr <= 1:
        raise ToolError("INVALID_ARGUMENT", "params.Cr = Cmin / Cmax goes from 0 to 1.")
    return cr


@op("heat_exchanger_ntu", ("mode", *sorted(set().union(*_HX_MODES.values()))))
def _hx(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_HX_MODES))
    only(p, mode, _HX_MODES[mode])
    flow = choice(p, "flow", _FLOWS)
    ctx.formula = (
        "Cr = Cmin / Cmax; NTU = U A / Cmin; Q = e Cmin (Th_in - Tc_in); counterflow "
        "e = [1 - exp(-NTU(1 - Cr))] / [1 - Cr exp(-NTU(1 - Cr))]; parallel flow "
        "e = [1 - exp(-NTU(1 + Cr))] / (1 + Cr); Cr = 0: e = 1 - exp(-NTU)"
    )
    if mode in ("effectiveness", "ntu"):
        mixed = _mixed(p, flow, ("cmin", "cmax"))
        cr = _cr(p)
        if mode == "effectiveness":
            n_tu = need(p, "NTU", None)
            out: dict[str, Any] = {"NTU": n_tu, "Cr": cr, "epsilon": _eps(flow, n_tu, cr, mixed)}
        else:
            eps = need(p, "epsilon", None)
            if eps >= 1:
                raise ToolError(
                    "INVALID_ARGUMENT", "params.epsilon must be between 0 and 1 (excluded)."
                )
            n_tu, alt = ntu_from_eps(flow, eps, cr, mixed, ctx)
            out = {"epsilon": eps, "Cr": cr, "NTU": n_tu}
            if alt is not None:
                out["NTU_alt"] = alt
        ctx.units = units_for(out, _HX_UNITS)
        ctx.formatted = (
            f"{_FLOW_TEXT[flow]}: e = {fmt_sig(out['epsilon'], 6)}, NTU = {fmt_sig(out['NTU'], 6)}."
        )
        return out
    return _hx_streams(p, mode, flow, ctx)


_FLOW_TEXT = {
    "parallel": "Parallel flow",
    "counter": "Counterflow",
    "crossflow_unmixed": "Crossflow, both unmixed",
    "crossflow_mixed": "Crossflow, both mixed",
    "crossflow_one_mixed": "Crossflow, one fluid mixed",
}


def _eps(flow: str, n_tu: float, cr: float, mixed: str | None) -> float:
    if flow == "crossflow_unmixed" and cr > 0 and n_tu > _NTU_MAX_SERIES:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"NTU above {fmt_sig(_NTU_MAX_SERIES, 3)}: outside the computed range of the series (crossflow).",
        )
    return effectiveness(flow, n_tu, cr, mixed)


def _hx_streams(p: dict[str, Any], mode: str, flow: str, ctx: Ctx) -> dict[str, Any]:
    th_in, tc_in = temperature(p, "Th_in"), temperature(p, "Tc_in")
    assert th_in is not None and tc_in is not None
    if th_in <= tc_in:
        raise ToolError("INVALID_ARGUMENT", "Th_in (hot fluid) must be greater than Tc_in.")
    pc = p.get("phase_change")
    if pc is not None:
        pc = choice(p, "phase_change", ("hot", "cold"))
        side = "h" if pc == "hot" else "c"
        if any(p.get(k) is not None for k in (f"C_{side}", f"m_{side}", f"cp_{side}")):
            raise ToolError(
                "INVALID_ARGUMENT",
                f"With phase_change '{pc}' that fluid has infinite capacity: remove C_{side}, m_{side}, cp_{side}.",
            )
    c_h = math.inf if pc == "hot" else _capacity(p, "h")
    c_c = math.inf if pc == "cold" else _capacity(p, "c")
    c_min, c_max = min(c_h, c_c), max(c_h, c_c)
    cr = 0.0 if math.isinf(c_max) else c_min / c_max
    mixed = _mixed(p, flow, ("hot", "cold"))
    if mixed is not None:
        mixed_c = c_h if mixed == "hot" else c_c
        mixed = "cmin" if mixed_c == c_min else "cmax"
    q_max = c_min * (th_in - tc_in)
    out: dict[str, Any] = {}
    if math.isfinite(c_h):
        out["C_h"] = c_h
    if math.isfinite(c_c):
        out["C_c"] = c_c
    out |= {"C_min": c_min, "Cr": cr}
    if math.isfinite(c_max):
        out["C_max"] = c_max
    if mode == "rating":
        if p.get("UA") is not None:
            if p.get("U") is not None or p.get("A") is not None:
                raise ToolError("INVALID_ARGUMENT", "Give UA or U and A, not all three.")
            ua = need(p, "UA", "thermal_conductance")
        else:
            ua = need(p, "U", "heat_transfer_coefficient") * need(p, "A", "area")
        n_tu = ua / c_min
        eps = _eps(flow, n_tu, cr, mixed)
        out |= {"UA": ua, "NTU": n_tu}
    else:
        given = [k for k in ("Q", "Th_out", "Tc_out") if p.get(k) is not None]
        if len(given) != 1:
            raise ToolError("INVALID_ARGUMENT", "Sizing needs only one of Q, Th_out, Tc_out.")
        if given[0] == "Q":
            q = need(p, "Q", "power")
        elif given[0] == "Th_out":
            if math.isinf(c_h):
                raise ToolError(
                    "INVALID_ARGUMENT",
                    "The hot fluid changes phase: Th_out = Th_in. Give Q or Tc_out.",
                )
            t = temperature(p, "Th_out")
            assert t is not None
            q = c_h * (th_in - t)
        else:
            if math.isinf(c_c):
                raise ToolError(
                    "INVALID_ARGUMENT",
                    "The cold fluid changes phase: Tc_out = Tc_in. Give Q or Th_out.",
                )
            t = temperature(p, "Tc_out")
            assert t is not None
            q = c_c * (t - tc_in)
        if not 0 < q < q_max:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Heat {fmt_sig(q, 6)} W outside (0, Q_max = {fmt_sig(q_max, 6)} W).",
            )
        eps = q / q_max
        n_tu, alt = ntu_from_eps(flow, eps, cr, mixed, ctx)
        out |= {"NTU": n_tu, "UA": n_tu * c_min}
        if alt is not None:
            out["NTU_alt"] = alt
        if p.get("U") is not None:
            out["A"] = out["UA"] / need(p, "U", "heat_transfer_coefficient")
    q = eps * q_max
    out |= {
        "epsilon": eps,
        "Q_max": q_max,
        "Q": q,
        "Th_out": th_in if math.isinf(c_h) else th_in - q / c_h,
        "Tc_out": tc_in if math.isinf(c_c) else tc_in + q / c_c,
    }
    ctx.units = units_for(out, _HX_UNITS)
    ctx.formatted = (
        f"{_FLOW_TEXT[flow]}: e = {fmt_sig(eps, 5)}, NTU = {fmt_sig(out['NTU'], 5)}, Q = {fmt_sig(q, 6)} W, "
        f"outlets {fmt_sig(out['Th_out'] - 273.15, 5)} °C (hot) and {fmt_sig(out['Tc_out'] - 273.15, 5)} °C (cold)."
    )
    return out


# ------------------------------------------------------------------ fin_heat_transfer

_FIN_SHAPES: dict[str, set[str]] = {"pin": {"D"}, "rectangular": {"w", "t"}, "custom": {"P", "A_c"}}
_FIN_KEYS = {"tip", "shape", "L", "h", "k", "T_b", "T_inf", "T_L", "x", "n_fins", "A_base"}
_FIN_UNITS = {
    "P": "m",
    "A_c": "m2",
    "m": "1/m",
    "mL": "1",
    "M": "W",
    "Q_f": "W",
    "A_f": "m2",
    "eta_f": "1",
    "eps_f": "1",
    "T_tip": "K",
    "T_x": "K",
    "A_t": "m2",
    "Q_total": "W",
    "eta_o": "1",
}


def _ch_ratio(a: float, b: float, r: float) -> float:
    """(cosh a + r sinh a) / (cosh b + r sinh b) for a, b >= 0, without overflow."""
    ea, eb = math.exp(-2 * a), math.exp(-2 * b)
    sa, sb = -math.expm1(-2 * a), -math.expm1(-2 * b)  # 1 - e^(-2a) without cancellation (D77)
    return math.exp(a - b) * ((1 + ea) + r * sa) / ((1 + eb) + r * sb)


def _sh_ratio(a: float, b: float) -> float:
    """sinh a / sinh b per 0 <= a <= b, b > 0."""
    return math.exp(a - b) * -math.expm1(-2 * a) / -math.expm1(-2 * b)


def _profile(
    tip: str, m: float, length: float, ratio: float, rl: float
) -> Callable[[float], float]:
    """theta(x) / theta_b for the 4 tip conditions (table of the source, cases A-D)."""
    ml = m * length

    def f(x: float) -> float:
        if tip == "infinite":
            return math.exp(-m * x)
        if tip == "convective":
            return _ch_ratio(m * (length - x), ml, ratio)
        if tip == "adiabatic":
            return _ch_ratio(m * (length - x), ml, 0.0)
        return rl * _sh_ratio(m * x, ml) + _sh_ratio(m * (length - x), ml)

    return f


@op("fin_heat_transfer", ("shape", *sorted(_FIN_KEYS | set().union(*_FIN_SHAPES.values()))))
def _fin(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    tip = choice(p, "tip", ("convective", "adiabatic", "fixed_T", "infinite"))
    shape = choice(p, "shape", tuple(_FIN_SHAPES))
    allowed = (_FIN_KEYS - {"tip"}) | _FIN_SHAPES[shape]
    if tip == "infinite":
        allowed -= {"L", "T_L", "n_fins", "A_base"}
    if tip != "fixed_T":
        allowed -= {"T_L"}
    only(p, tip, allowed | {"shape"}, field="tip")
    if shape == "pin":
        d = need(p, "D", "length")
        perim, a_c = math.pi * d, math.pi * d * d / 4
    elif shape == "rectangular":
        w, t = need(p, "w", "length"), need(p, "t", "length")
        perim, a_c = 2 * (w + t), w * t
    else:
        perim, a_c = need(p, "P", "length"), need(p, "A_c", "area")
    h, k = need(p, "h", "heat_transfer_coefficient"), need(p, "k", "thermal_conductivity")
    t_b, t_inf = temperature(p, "T_b"), temperature(p, "T_inf")
    assert t_b is not None and t_inf is not None
    th_b = t_b - t_inf
    if th_b == 0:
        raise ToolError("INVALID_ARGUMENT", "T_b and T_inf are equal: the fin exchanges no heat.")
    m = math.sqrt(h * perim / (k * a_c))
    big_m = math.sqrt(h * perim * k * a_c) * th_b
    out: dict[str, Any] = {"P": perim, "A_c": a_c, "m": m, "M": big_m}
    length = None if tip == "infinite" else need(p, "L", "length")
    x = qty(p, "x", "length", need=False, positive=False)
    if x is not None and (x < 0 or (length is not None and x > length)):
        raise ToolError("INVALID_ARGUMENT", "params.x goes from 0 to the fin length L.")
    if length is None:
        q, prof = big_m, _profile(tip, m, 0.0, 0.0, 0.0)
        ctx.formula = (
            "m = sqrt(h P / (k A_c)); Q_f = sqrt(h P k A_c) theta_b; theta / theta_b = exp(-m x)"
        )
    else:
        ml = m * length
        out["mL"] = ml
        ratio = h / (m * k)
        rl = 0.0
        if tip == "convective":
            tn = math.tanh(ml)
            q = big_m * (tn + ratio) / (1 + ratio * tn)
            ctx.formula = (
                "m = sqrt(h P / (k A_c)); Q_f = sqrt(h P k A_c) theta_b [sinh mL + (h/mk) cosh mL] / "
                "[cosh mL + (h/mk) sinh mL]"
            )
        elif tip == "adiabatic":
            q = big_m * math.tanh(ml)
            ctx.formula = (
                "m = sqrt(h P / (k A_c)); Q_f = sqrt(h P k A_c) theta_b tanh(mL); "
                "theta / theta_b = cosh m(L - x) / cosh mL"
            )
        else:
            t_l = temperature(p, "T_L")
            assert t_l is not None
            rl = (t_l - t_inf) / th_b
            inv_sinh = 2 * math.exp(-ml) / -math.expm1(-2 * ml)
            q = big_m * (1 / math.tanh(ml) - rl * inv_sinh)
            ctx.formula = "m = sqrt(h P / (k A_c)); Q_f = sqrt(h P k A_c) theta_b (cosh mL - theta_L / theta_b) / sinh mL"
        prof = _profile(tip, m, length, ratio, rl)
        out["T_tip"] = t_inf + th_b * prof(length)
        a_f = perim * length + (a_c if tip == "convective" else 0.0)
        out |= {"A_f": a_f, "eta_f": q / (h * a_f * th_b)}
    out |= {"Q_f": q, "eps_f": q / (h * a_c * th_b)}
    if x is not None:
        out["T_x"] = t_inf + th_b * prof(x)
    if p.get("n_fins") is not None or p.get("A_base") is not None:
        n_f = integer(p, "n_fins")
        a_base = need(p, "A_base", "area")
        free = a_base - n_f * a_c
        if free < 0:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"A_base = {fmt_sig(a_base, 5)} m2 is smaller than the fin bases ({fmt_sig(n_f * a_c, 5)} m2).",
            )
        a_t = n_f * out["A_f"] + free
        q_tot = n_f * q + h * free * th_b
        out |= {"A_t": a_t, "Q_total": q_tot, "eta_o": q_tot / (h * a_t * th_b)}
    ctx.units = units_for(out, _FIN_UNITS)
    ctx.formatted = (
        f"Fin: Q_f = {fmt_sig(q, 6)} W"
        + (f", efficiency {fmt_sig(out['eta_f'], 4)}" if "eta_f" in out else "")
        + f", effectiveness {fmt_sig(out['eps_f'], 4)}."
    )
    return out


# ------------------------------------------------------------------ lumped_transient

_LUMP_SHAPES: dict[str, set[str]] = {"sphere": {"D"}, "custom": {"V", "A"}}
_LUMP_KEYS = {"rho", "cp", "h", "k", "T_0", "T_inf", "t", "T"}
_LUMP_UNITS = {
    "V": "m3",
    "A": "m2",
    "L_c": "m",
    "tau": "s",
    "Bi": "1",
    "T": "K",
    "t": "s",
    "theta_ratio": "1",
    "Q": "J",
}


@op("lumped_transient", ("shape", *sorted(_LUMP_KEYS | set().union(*_LUMP_SHAPES.values()))))
def _lumped(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    shape = choice(p, "shape", tuple(_LUMP_SHAPES), default="custom")
    only(p, shape, _LUMP_KEYS | _LUMP_SHAPES[shape], field="shape")
    if shape == "sphere":
        d = need(p, "D", "length")
        vol, area = math.pi * d**3 / 6, math.pi * d * d
    else:
        vol, area = need(p, "V", "volume"), need(p, "A", "area")
    rho, cp = need(p, "rho", "density"), need(p, "cp", "specific_heat")
    h = need(p, "h", "heat_transfer_coefficient")
    t0, t_inf = temperature(p, "T_0"), temperature(p, "T_inf")
    assert t0 is not None and t_inf is not None
    if t0 == t_inf:
        raise ToolError("INVALID_ARGUMENT", "T_0 and T_inf are equal: there is no transient.")
    lc = vol / area
    tau = rho * cp * vol / (h * area)
    out: dict[str, Any] = {"V": vol, "A": area, "L_c": lc, "tau": tau}
    if p.get("k") is not None:
        bi = h * lc / need(p, "k", "thermal_conductivity")
        out["Bi"] = bi
        if bi > 0.1:
            ctx.warn(
                f"Bi = {fmt_sig(bi, 4)} > 0.1: the temperature inside the body is not uniform, "
                "the lumped-capacitance model is not reliable."
            )
    else:
        ctx.warn("Without k the Biot number is not checked (the model holds for Bi < 0.1 roughly).")
    if (p.get("t") is None) == (p.get("T") is None):
        raise ToolError(
            "INVALID_ARGUMENT", "One of t (time, gives T) and T (temperature, gives t) is needed."
        )
    if p.get("t") is not None:
        t = need(p, "t", "time", positive=False)
        if t < 0:
            raise ToolError("INVALID_ARGUMENT", "params.t must be >= 0.")
        ratio = math.exp(-t / tau)
        temp = t_inf + (t0 - t_inf) * ratio
        # Q from 1 - exp(-t / tau) with expm1: T_0 - T cancels at small t (D77)
        q = rho * cp * vol * (t0 - t_inf) * -math.expm1(-t / tau)
    else:
        temp_in = temperature(p, "T")
        assert temp_in is not None
        temp = temp_in
        ratio = (temp - t_inf) / (t0 - t_inf)
        if not 0 < ratio <= 1:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.T must be between T_inf (excluded) and T_0: the body tends to T_inf without passing it.",
            )
        # close to T_0 the ratio rounds near 1: log1p of (T - T_0) / (T_0 - T_inf), T - T_0 exact (D77)
        t = -tau * (math.log1p((temp - t0) / (t0 - t_inf)) if ratio > 0.5 else math.log(ratio))
        q = rho * cp * vol * (t0 - temp)
    out |= {"t": t, "T": temp, "theta_ratio": ratio, "Q": q}
    ctx.formula = (
        "L_c = V / A; Bi = h L_c / k; tau = rho cp V / (h A); (T - T_inf) / (T_0 - T_inf) = exp(-t / tau); "
        "Q = rho cp V (T_0 - T)"
    )
    ctx.units = units_for(out, _LUMP_UNITS)
    ctx.formatted = (
        f"tau = {fmt_sig(tau, 5)} s; after {fmt_sig(t, 6)} s the temperature is {fmt_sig(temp - 273.15, 6)} °C; "
        f"heat released {fmt_sig(out['Q'], 6)} J."
    )
    return out
