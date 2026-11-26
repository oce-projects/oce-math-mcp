"""eng_calc area D (D39): circuits, filters, Clarke-Park, controls, optics and acoustics.

Constants: resistivity of annealed copper at 20 degrees 1/58 ohm mm2/m (IEC 60028); reference sound
pressure 20 micropascal (ISO 1683). Ziegler-Nichols tables from the original paper (1942).
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import choice, need, one_of, only, op, qty, units_for
from .eng_dyn import CIRC_FUNCS, CIRC_MODES, OA_FUNCS, OA_MODES

RHO_CU = 1e-6 / 58  # ohm m, IEC 60028
P_REF = 20e-6  # Pa, ISO 1683


def _brentq(f: Any, a: float, b: float) -> float:
    from scipy.optimize import brentq

    return float(cast(Any, brentq)(f, a, b, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500))


def _phi(p: dict[str, Any]) -> tuple[float, float]:
    """cos(phi) and phi in radians from cos_phi or phi_deg (inductive load, phi >= 0)."""
    if one_of(p, ("cos_phi", "phi_deg"), "the phase shift") == "cos_phi":
        c = need(p, "cos_phi", None)
        if c > 1:
            raise ToolError("INVALID_ARGUMENT", "params.cos_phi must be between 0 and 1.")
        return c, math.acos(c)
    ph = math.radians(need(p, "phi_deg", None, positive=False))
    return math.cos(ph), ph


# ------------------------------------------------------------------ circuit_ac_dc

_CIRC_MODES = {
    "ohm": {"V", "I", "R", "P"},
    "resistors": {"values", "conductances", "connection"},
    "inductors": {"values", "connection", "M"},
    "divider": {"Vin", "R1", "R2"},
    "ac_single": {"V", "I", "P", "cos_phi", "phi_deg"},
    "ac_three": {"V", "I", "P", "cos_phi", "phi_deg"},
    "cable_drop": {"phases", "I", "L", "A_section", "material", "rho", "cos_phi", "V"},
}
_CIRC_MODES |= CIRC_MODES  # D45


def _ohm(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    kinds = {
        "V": "voltage",
        "I": "current",
        "R": "electrical_resistance",
        "P": "power",
    }
    g = {k: need(p, k, kinds[k]) for k in kinds if p.get(k) is not None}
    if len(g) != 2:
        raise ToolError("INVALID_ARGUMENT", "Ohm's law: exactly 2 of V, I, R, P are needed.")
    if "V" in g and "I" in g:
        V, cur = g["V"], g["I"]
    elif "V" in g and "R" in g:
        V, cur = g["V"], g["V"] / g["R"]
    elif "I" in g and "R" in g:
        V, cur = g["I"] * g["R"], g["I"]
    elif "V" in g:
        V, cur = g["V"], g["P"] / g["V"]
    elif "I" in g:
        V, cur = g["P"] / g["I"], g["I"]
    else:
        cur = math.sqrt(g["P"] / g["R"])
        V = cur * g["R"]
    ctx.formula = "V = R I; P = V I = R I^2 = V^2 / R"
    return {"V": V, "I": cur, "R": V / cur, "P": V * cur}


def _positive_list(p: dict[str, Any], key: str, what: str, example: str) -> list[float]:
    vals = p.get(key)
    if (
        not isinstance(vals, list)
        or not vals
        or not all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) and v > 0
            for v in vals
        )
    ):
        raise ToolError("INVALID_ARGUMENT", f"params.{key}: list of {what}, all > 0.", example)
    return [float(v) for v in vals]


def _resistors(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    conductances = p.get("conductances") is not None
    if conductances and p.get("values") is not None:
        one_of(p, ("values", "conductances"), "the elements (resistances or conductances)")
    conn = choice(p, "connection", ("series", "parallel"))
    if conductances:  # D114
        vals = _positive_list(
            p, "conductances", "conductances in S", '{"conductances": [0.1, 0.2]}'
        )
        G = 1 / sum(1 / v for v in vals) if conn == "series" else sum(vals)
        ctx.formula = "series: 1/G = sum(1/G_i); parallel: G = sum(G_i); R = 1/G"
        return {"R_eq": 1 / G, "G_eq": G}
    vals = _positive_list(p, "values", "resistances in ohm", '{"values": [100, 220]}')
    R = sum(vals) if conn == "series" else 1 / sum(1 / v for v in vals)
    ctx.formula = "series: R = sum(R_i); parallel: 1/R = sum(1/R_i); G = 1/R"
    return {"R_eq": R, "G_eq": 1 / R}


def _inductors(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    """D114: uncoupled coils, or 2 coils with mutual inductance M (signed: < 0 for opposing windings)."""
    vals = _positive_list(p, "values", "inductances in H", '{"values": [0.01, 0.02]}')
    conn = choice(p, "connection", ("series", "parallel"))
    if p.get("M") is None:
        L = sum(vals) if conn == "series" else 1 / sum(1 / v for v in vals)
        ctx.formula = "uncoupled coils: series L = sum(L_i); parallel 1/L = sum(1/L_i)"
        return {"L_eq": L}
    if len(vals) != 2:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.M is the mutual inductance of 2 coupled coils: params.values must have 2 values.",
        )
    l1, l2 = vals
    m = need(p, "M", "inductance", positive=False)
    top = math.sqrt(l1 * l2)
    if abs(m) > top:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"|params.M| must be <= sqrt(L1 L2) = {fmt_float(top, 6)} H (coupling coefficient |k| <= 1).",
        )
    k = m / top
    if conn == "series":
        L = l1 + l2 + 2 * m
    else:
        den = l1 + l2 - 2 * m
        if den == 0:
            raise ToolError(
                "INVALID_ARGUMENT",
                "L1 = L2 = M (k = 1) in parallel: (L1 L2 - M^2) / (L1 + L2 - 2 M) is 0/0, the equivalent "
                "inductance is undefined.",
            )
        L = l1 * l2 * (1 - k) * (1 + k) / den
    ctx.formula = (
        "2 coupled coils, M = k sqrt(L1 L2) (M < 0 for opposing windings): series L = L1 + L2 + 2 M; "
        "parallel L = (L1 L2 - M^2) / (L1 + L2 - 2 M)"
    )
    return {"L_eq": L, "k_coupling": k}


def _divider(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    Vin = need(p, "Vin", "voltage", positive=False)
    R1, R2 = need(p, "R1", "electrical_resistance"), need(p, "R2", "electrical_resistance")
    ctx.formula = "Vout = Vin R2 / (R1 + R2) (output without load)"
    return {"Vout": Vin * R2 / (R1 + R2), "I": Vin / (R1 + R2)}


def _ac(p: dict[str, Any], ctx: Ctx, three: bool) -> dict[str, Any]:
    k = math.sqrt(3) if three else 1.0
    c, ph = _phi(p)
    V = need(p, "V", "voltage")
    if one_of(p, ("I", "P"), "the current I or the active power P") == "I":
        cur = need(p, "I", "current")
    else:
        if c == 0:
            raise ToolError(
                "MATH_ERROR", "With cos_phi = 0 the active power does not determine the current."
            )
        cur = need(p, "P", "power") / (k * V * c)
    S = k * V * cur
    ctx.formula = (
        "three-phase (line-to-line V): S = sqrt(3) V I" if three else "single-phase: S = V I"
    ) + "; P = S cos(phi); Q = S sin(phi)"
    return {
        "V": V,
        "I": cur,
        "S": S,
        "P": S * c,
        "Q": S * math.sin(ph),
        "cos_phi": c,
        "phi_deg": math.degrees(ph),
    }


def _cable(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    phases = choice(p, "phases", ("single", "three"))
    cur, L, A = need(p, "I", "current"), need(p, "L", "length"), need(p, "A_section", "area")
    if one_of(p, ("material", "rho"), "the resistivity (material or rho)") == "material":
        choice(p, "material", ("copper",))
        rho = RHO_CU
    else:
        rho = need(p, "rho", "resistivity")
    c = qty(p, "cos_phi", None, default=1.0) or 1.0
    if c > 1:
        raise ToolError("INVALID_ARGUMENT", "params.cos_phi must be between 0 and 1.")
    k = 2.0 if phases == "single" else math.sqrt(3)
    R = rho * L / A
    out: dict[str, Any] = {
        "rho": rho,
        "R_conductor": R,
        "dV": k * R * cur * c,
        "P_loss": (2 if phases == "single" else 3) * R * cur * cur,
    }
    V = qty(p, "V", "voltage", need=False)
    if V is not None:
        out["dV_pct"] = 100 * out["dV"] / V
    ctx.warn(
        "Drop computed with the resistance only (cable reactance neglected), conductor at 20 degrees."
    )
    ctx.formula = (
        "R = rho L / A; dV = k R I cos(phi), k = 2 single-phase, sqrt(3) three-phase; "
        "copper IEC 60028: rho = 1/58 ohm mm2/m"
    )
    return out


@op("circuit_ac_dc", set().union(*_CIRC_MODES.values()) | {"mode"})
def _circuit(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_CIRC_MODES))
    only(p, mode, _CIRC_MODES[mode])
    if mode in CIRC_FUNCS:
        return CIRC_FUNCS[mode](p, ctx)
    if mode in {"ac_single", "ac_three"}:
        out = _ac(p, ctx, mode == "ac_three")
    else:
        out = {
            "ohm": _ohm,
            "resistors": _resistors,
            "inductors": _inductors,
            "divider": _divider,
            "cable_drop": _cable,
        }[mode](p, ctx)
    ctx.units = units_for(
        out,
        {
            "V": "V",
            "I": "A",
            "R": "ohm",
            "P": "W",
            "R_eq": "ohm",
            "G_eq": "S",
            "L_eq": "H",
            "k_coupling": "1",
            "Vout": "V",
            "S": "VA",
            "Q": "var",
            "cos_phi": "1",
            "phi_deg": "deg",
            "rho": "ohm*m",
            "R_conductor": "ohm",
            "dV": "V",
            "P_loss": "W",
            "dV_pct": "%",
        },
    )
    ctx.formatted = (
        "; ".join(f"{k} = {fmt_float(v, 4)}" for k, v in out.items() if isinstance(v, float))
        + " (SI units)."
    )
    return out


# ------------------------------------------------------------------ rlc_filter

_FILTERS = {
    "rc_lowpass": {"R", "C"},
    "rc_highpass": {"R", "C"},
    "rl_lowpass": {"R", "L"},
    "rl_highpass": {"R", "L"},
    "rlc_bandpass": {"R", "L", "C"},
}


@op("rlc_filter", ("type", "R", "L", "C", "f"))
def _filter(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    kind = choice(p, "type", tuple(_FILTERS))
    only(p, kind, _FILTERS[kind] | {"type", "f"})
    R = need(p, "R", "electrical_resistance")
    out: dict[str, Any] = {}
    if kind == "rlc_bandpass":
        L, C = need(p, "L", "inductance"), need(p, "C", "capacitance")
        f0 = 1 / (2 * math.pi * math.sqrt(L * C))
        Q = math.sqrt(L / C) / R
        out |= {"f0": f0, "Q": Q, "bandwidth": f0 / Q, "zeta": 1 / (2 * Q)}
        formulas = [
            "f0 = 1 / (2 pi sqrt(L C)); Q = sqrt(L / C) / R; bandwidth = f0 / Q",
            "H = 1 / (1 + j Q (f/f0 - f0/f)) (output across R)",
        ]
    else:
        fc = (
            1 / (2 * math.pi * R * need(p, "C", "capacitance"))
            if kind.startswith("rc")
            else R / (2 * math.pi * need(p, "L", "inductance"))
        )
        out["fc"] = fc
        low = kind.endswith("lowpass")
        formulas = [
            "fc = 1 / (2 pi R C)" if kind.startswith("rc") else "fc = R / (2 pi L)",
            "low-pass H = 1 / (1 + j f/fc)" if low else "high-pass H = (j f/fc) / (1 + j f/fc)",
        ]
    f = qty(p, "f", "frequency", need=False)
    if f is not None:
        if kind == "rlc_bandpass":
            H = 1 / complex(1, out["Q"] * (f / out["f0"] - out["f0"] / f))
        else:
            x = 1j * f / out["fc"]
            H = 1 / (1 + x) if kind.endswith("lowpass") else x / (1 + x)
        out |= {
            "gain": abs(H),
            "gain_dB": 20 * math.log10(abs(H)),
            "phase_deg": math.degrees(math.atan2(H.imag, H.real)),
        }
    ctx.units = units_for(
        out,
        {
            "fc": "Hz",
            "f0": "Hz",
            "Q": "1",
            "bandwidth": "Hz",
            "zeta": "1",
            "gain": "1",
            "gain_dB": "dB",
            "phase_deg": "deg",
        },
    )
    ctx.formula = "; ".join(formulas)
    key = "fc" if "fc" in out else "f0"
    text = (
        f"{'Cutoff frequency' if key == 'fc' else 'Resonant frequency'} {fmt_float(out[key], 3)} Hz"
    )
    if "gain_dB" in out:
        text += (
            f", at {fmt_float(cast(float, f), 2)} Hz gain {fmt_float(out['gain_dB'], 3)} dB"
            f" and phase {fmt_float(out['phase_deg'], 2)} degrees"
        )
    ctx.formatted = text + "."
    return out


# ------------------------------------------------------------------ foc_clarke_park


@op("foc_clarke_park", ("mode", "ia", "ib", "ic", "d", "q", "zero", "theta_deg", "scaling"))
def _clarke(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", ("forward", "inverse"))
    scaling = choice(p, "scaling", ("amplitude", "power"), "amplitude")
    only(
        p,
        mode,
        ({"ia", "ib", "ic"} if mode == "forward" else {"d", "q", "zero"})
        | {"theta_deg", "scaling"},
    )
    th = math.radians(need(p, "theta_deg", None, positive=False))
    k, k0 = (2 / 3, 1 / 3) if scaling == "amplitude" else (math.sqrt(2 / 3), 1 / math.sqrt(3))
    s3 = math.sqrt(3) / 2
    if mode == "forward":
        a, b = need(p, "ia", None, positive=False), need(p, "ib", None, positive=False)
        c = qty(p, "ic", None, positive=False, need=False)
        c = -a - b if c is None else c
        al, be = k * (a - b / 2 - c / 2), k * s3 * (b - c)
        out: dict[str, Any] = {
            "alpha": al,
            "beta": be,
            "zero": k0 * (a + b + c) if scaling == "amplitude" else (a + b + c) / math.sqrt(3),
            "d": al * math.cos(th) + be * math.sin(th),
            "q": -al * math.sin(th) + be * math.cos(th),
        }
        ctx.formula = (
            f"Clarke ({scaling}): alpha = k (a - b/2 - c/2), beta = k (sqrt(3)/2) (b - c), k = "
            + ("2/3" if scaling == "amplitude" else "sqrt(2/3)")
            + "; Park: d = alpha cos(theta) + beta sin(theta), q = -alpha sin(theta) + beta cos(theta)"
        )
    else:
        d, q = need(p, "d", None, positive=False), need(p, "q", None, positive=False)
        z = qty(p, "zero", None, positive=False, default=0.0) or 0.0
        al, be = d * math.cos(th) - q * math.sin(th), d * math.sin(th) + q * math.cos(th)
        if scaling == "amplitude":
            a, b, c = al + z, -al / 2 + s3 * be + z, -al / 2 - s3 * be + z
        else:
            g = math.sqrt(2 / 3)
            a, b, c = (
                g * al + z / math.sqrt(3),
                g * (-al / 2 + s3 * be) + z / math.sqrt(3),
                g * (-al / 2 - s3 * be) + z / math.sqrt(3),
            )
        out = {"alpha": al, "beta": be, "a": a, "b": b, "c": c}
        ctx.formula = "inverse Park: alpha = d cos(theta) - q sin(theta), beta = d sin(theta) + q cos(theta); inverse Clarke"
    ctx.units = {k_: "same as the input" for k_ in out}
    ctx.formatted = "; ".join(f"{k_} = {fmt_float(v, 6)}" for k_, v in out.items()) + "."
    return out


# ------------------------------------------------------------------ control_system

_CTRL_MODES = {
    "second_order": {"zeta", "wn"},
    "margins": {"num", "den"},
    "pid_zn": {"method", "Ku", "Tu", "K", "L", "T", "controller"},
}


def _step_fn(z: float, wn: float) -> Any:
    if z < 1:
        wd = wn * math.sqrt(1 - z * z)
        return lambda t: (
            1 - np.exp(-z * wn * t) * (np.cos(wd * t) + z / math.sqrt(1 - z * z) * np.sin(wd * t))
        )
    if z == 1:
        return lambda t: 1 - np.exp(-wn * t) * (1 + wn * t)
    r = math.sqrt(z * z - 1)
    # D75: s1 = -wn (z - r) loses digits by cancellation for large z; (z - r)(z + r) = 1 gives the stable form
    s1, s2 = -wn / (z + r), -wn * (z + r)
    return lambda t: 1 + (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s1 - s2)


def _second_order(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    z = need(p, "zeta", None, positive=False)
    if z <= 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.zeta must be > 0 (with zeta = 0 the response does not settle).",
        )
    wn = need(p, "wn", "angular_velocity")
    y = _step_fn(z, wn)
    if z < 1:
        # D92: a grid over 40 / (zeta wn) missed the oscillations for zeta < 1e-4 (zeta = 1e-5: tr 7.30 s
        # instead of 1.0196 s). The extrema of y are at t_k = k pi / wd with |y - 1| = Mp^k: the rise is in
        # [0, t_1], where y grows; ts is in [t_k, t_(k+1)] for the last k with Mp^k > 0.02
        half = math.pi / (wn * math.sqrt(1 - z * z))
        t10 = _brentq(lambda s: float(y(s)) - 0.1, 0.0, half)
        t90 = _brentq(lambda s: float(y(s)) - 0.9, 0.0, half)
        ln_mp, ln_band = -math.pi * z / math.sqrt((1 - z) * (1 + z)), math.log(0.02)
        k = math.floor(ln_band / ln_mp)  # Mp^k >= 0.02 > Mp^(k+1); at equality brentq returns t_k
        ts = _brentq(lambda s: abs(float(y(s)) - 1) - 0.02, k * half, (k + 1) * half)
    else:
        slow = wn / (
            z + math.sqrt(z * z - 1)
        )  # slow pole, stable form (D75); y grows without overshoot
        t = np.linspace(0, 40 / slow, 200001)
        yy = y(t)
        bad = np.nonzero(np.abs(yy - 1) > 0.02)[0]
        i = int(bad[-1])
        ts = _brentq(lambda s: abs(float(y(s)) - 1) - 0.02, float(t[i]), float(t[i + 1]))
        t10 = _brentq(lambda s: float(y(s)) - 0.1, 0.0, float(t[int(np.argmax(yy >= 0.1))]))
        t90 = _brentq(lambda s: float(y(s)) - 0.9, 0.0, float(t[int(np.argmax(yy >= 0.9))]))
    # sqrt(1 - zeta^2) as sqrt((1 - zeta)(1 + zeta)): no cancellation close to 1 (D78)
    out: dict[str, Any] = {
        "wd": wn * math.sqrt((1 - z) * (1 + z)) if z < 1 else None,
        "Mp_pct": 100 * math.exp(-math.pi * z / math.sqrt((1 - z) * (1 + z))) if z < 1 else 0.0,
        "tp": math.pi / (wn * math.sqrt((1 - z) * (1 + z))) if z < 1 else None,
        "ts_2pct": ts,
        "ts_2pct_approx": 4 / (z * wn),
        "tr_10_90": t90 - t10,
    }
    ctx.formula = (
        "G = wn^2 / (s^2 + 2 zeta wn s + wn^2); Mp = exp(-pi zeta / sqrt(1 - zeta^2)); tp = pi / wd; "
        "ts (2 %) and tr (10-90 %) computed on the exact step response; short formula ts = 4 / (zeta wn)"
    )
    return out


def _poly(p: dict[str, Any], key: str) -> list[float]:
    v = p.get(key)
    if (
        not isinstance(v, list)
        or not v
        or not all(isinstance(x, int | float) and not isinstance(x, bool) for x in v)
    ):
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: coefficients of the polynomial in s, from the highest degree.",
            '{"num": [2], "den": [1, 3, 2, 0]}',
        )
    if all(x == 0 for x in v):
        raise ToolError("INVALID_ARGUMENT", f"params.{key} is the zero polynomial.")
    return [float(x) for x in v]


def _margins(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    num, den = _poly(p, "num"), _poly(p, "den")

    def L(w: float) -> complex:
        s = 1j * w
        return complex(np.polyval(num, s) / np.polyval(den, s))

    roots = [abs(r) for r in np.concatenate([np.roots(num), np.roots(den)]) if abs(r) > 1e-12]
    lo = min([1e-3, *(0.001 * r for r in roots)])
    hi = max([1e3, *(1000 * r for r in roots)])
    w = np.logspace(math.log10(lo), math.log10(hi), 40001)
    Lw = np.polyval(num, 1j * w) / np.polyval(den, 1j * w)
    mag = np.abs(Lw)
    out: dict[str, Any] = {
        "wc_gain": None,
        "phase_margin_deg": None,
        "wc_phase": None,
        "gain_margin_dB": None,
    }
    cg = np.nonzero(np.diff(np.sign(mag - 1)))[0]
    if len(cg):
        i = int(cg[0])
        wg = _brentq(lambda x: abs(L(x)) - 1, float(w[i]), float(w[i + 1]))
        ph_unw = np.degrees(np.unwrap(np.angle(Lw)))
        guess = float(np.interp(wg, w, ph_unw))
        exact = math.degrees(math.atan2(L(wg).imag, L(wg).real))
        exact += 360 * round((guess - exact) / 360)
        out |= {"wc_gain": wg, "phase_margin_deg": 180 + exact}
    im = Lw.imag
    cp = [i for i in np.nonzero(np.diff(np.sign(im)))[0] if Lw.real[i] < 0 and Lw.real[i + 1] < 0]
    if cp:
        i = int(cp[0])
        wp = _brentq(lambda x: L(x).imag, float(w[i]), float(w[i + 1]))
        out |= {"wc_phase": wp, "gain_margin_dB": -20 * math.log10(abs(L(wp)))}
    if out["wc_gain"] is None:
        ctx.warn("The magnitude does not cross 1 in the frequency range: phase margin not defined.")
    ctx.formula = "|L(j wc)| = 1, PM = 180 + arg L(j wc); Im L(j wp) = 0 with Re < 0, GM = -20 log10 |L(j wp)|"
    return out


def _zn(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    method = choice(p, "method", ("ultimate", "step"))
    ctrl = choice(p, "controller", ("P", "PI", "PID"), "PID")
    if method == "ultimate":
        Ku, Tu = need(p, "Ku", None), need(p, "Tu", "time")
        Kp, Ti, Td = {
            "P": (0.5 * Ku, None, 0.0),
            "PI": (0.45 * Ku, Tu / 1.2, 0.0),
            "PID": (0.6 * Ku, Tu / 2, Tu / 8),
        }[ctrl]
        ctx.formula = (
            "Ziegler-Nichols closed loop: P Kp = 0.5 Ku; PI Kp = 0.45 Ku, Ti = Tu / 1.2; "
            "PID Kp = 0.6 Ku, Ti = Tu / 2, Td = Tu / 8"
        )
    else:
        K, Ld, T = need(p, "K", None), need(p, "L", "time"), need(p, "T", "time")
        a = T / (K * Ld)
        Kp, Ti, Td = {
            "P": (a, None, 0.0),
            "PI": (0.9 * a, Ld / 0.3, 0.0),
            "PID": (1.2 * a, 2 * Ld, 0.5 * Ld),
        }[ctrl]
        ctx.formula = (
            "Ziegler-Nichols step response: P Kp = T/(K L); PI 0.9 T/(K L), Ti = L/0.3; "
            "PID 1.2 T/(K L), Ti = 2 L, Td = 0.5 L"
        )
    return {"Kp": Kp, "Ti": Ti, "Td": Td, "Ki": Kp / Ti if Ti else 0.0, "Kd": Kp * Td}


@op("control_system", set().union(*_CTRL_MODES.values()) | {"mode"})
def _control(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_CTRL_MODES))
    only(p, mode, _CTRL_MODES[mode])
    out = {"second_order": _second_order, "margins": _margins, "pid_zn": _zn}[mode](p, ctx)
    ctx.units = units_for(
        out,
        {
            "wd": "rad/s",
            "Mp_pct": "%",
            "tp": "s",
            "ts_2pct": "s",
            "ts_2pct_approx": "s",
            "tr_10_90": "s",
            "wc_gain": "rad/s",
            "phase_margin_deg": "deg",
            "wc_phase": "rad/s",
            "gain_margin_dB": "dB",
            "Kp": "1",
            "Ti": "s",
            "Td": "s",
            "Ki": "1/s",
            "Kd": "s",
        },
    )
    ctx.formatted = (
        "; ".join(f"{k} = {fmt_float(v, 4)}" for k, v in out.items() if isinstance(v, float)) + "."
    )
    return out


# ------------------------------------------------------------------ optics_acoustics

_OA_MODES = {
    "snell": {"n1", "n2", "theta1_deg"},
    "thin_lens": {"f", "d_o", "d_i"},
    "db": {"ratio", "dB", "kind"},
    "spl": {"p", "SPL_dB"},
    "sum_levels": {"levels"},
    "inverse_square": {"L1", "r1", "r2"},
    "wave": {"f", "v"},
}
_OA_MODES |= OA_MODES  # D45


def _optics(p: dict[str, Any], mode: str, ctx: Ctx) -> dict[str, Any]:
    if mode == "snell":
        n1, n2 = need(p, "n1", None), need(p, "n2", None)
        t1 = math.radians(need(p, "theta1_deg", None, positive=False))
        s = n1 * math.sin(t1) / n2
        ctx.formula = "n1 sin(theta1) = n2 sin(theta2); critical angle = asin(n2 / n1) if n1 > n2"
        return {
            "theta2_deg": math.degrees(math.asin(s)) if abs(s) <= 1 else None,
            "total_internal_reflection": abs(s) > 1,
            "critical_angle_deg": math.degrees(math.asin(n2 / n1)) if n1 > n2 else None,
        }
    if mode == "thin_lens":
        g = {
            k: need(p, k, "length", positive=k == "d_o")
            for k in ("f", "d_o", "d_i")
            if p.get(k) is not None
        }
        if len(g) != 2:
            raise ToolError("INVALID_ARGUMENT", "Thin lens: exactly 2 of f, d_o, d_i are needed.")
        for k in ("f", "d_i"):
            if g.get(k) == 0:
                raise ToolError("INVALID_ARGUMENT", f"params.{k} must not be 0.")
        # 1/d_o + 1/d_i and 1/f - 1/x as one sum or difference over a product: exact near the focus and
        # with d_i close to -d_o (D82); a ratio first, then one product: no overflow or underflow of the
        # product of two lengths (D84)
        if "f" not in g:
            s = g["d_o"] + g["d_i"]
            if s == 0:
                raise ToolError(
                    "MATH_ERROR", "1/d_o + 1/d_i = 0: the lens has no power, no focal length."
                )
            big, small = sorted((g["d_o"], g["d_i"]), key=abs, reverse=True)
            g["f"] = small * (big / s)
        elif "d_i" not in g:
            if g["d_o"] == g["f"]:
                raise ToolError("MATH_ERROR", "Object at the focus: image at infinity.")
            g["d_i"] = g["f"] * (g["d_o"] / (g["d_o"] - g["f"]))
        else:
            if g["d_i"] == g["f"]:
                raise ToolError("MATH_ERROR", "Image at the focus: object at infinity.")
            g["d_o"] = g["f"] * (g["d_i"] / (g["d_i"] - g["f"]))
        ctx.formula = "1/f = 1/d_o + 1/d_i (real is positive); magnification m = -d_i / d_o"
        return {
            "f": g["f"],
            "d_o": g["d_o"],
            "d_i": g["d_i"],
            "magnification": -g["d_i"] / g["d_o"],
        }
    if mode == "db":
        kind = choice(p, "kind", ("power", "amplitude"))
        k = 10.0 if kind == "power" else 20.0
        ctx.formula = "dB = 10 log10(power ratio) = 20 log10(amplitude ratio)"
        if one_of(p, ("ratio", "dB"), "the value to convert") == "ratio":
            return {"dB": k * math.log10(need(p, "ratio", None))}
        return {"ratio": 10 ** (need(p, "dB", None, positive=False) / k)}
    if mode == "spl":
        ctx.formula = "SPL = 20 log10(p / p0), p0 = 20 micropascal (ISO 1683)"
        if one_of(p, ("p", "SPL_dB"), "the pressure or the level") == "p":
            return {"SPL_dB": 20 * math.log10(need(p, "p", "pressure") / P_REF)}
        return {"p": P_REF * 10 ** (need(p, "SPL_dB", None, positive=False) / 20)}
    if mode == "sum_levels":
        lv = p.get("levels")
        if (
            not isinstance(lv, list)
            or not lv
            or not all(isinstance(x, int | float) and not isinstance(x, bool) for x in lv)
        ):
            raise ToolError(
                "INVALID_ARGUMENT", "params.levels: list of levels in dB.", '{"levels": [80, 83]}'
            )
        ctx.formula = "L_tot = 10 log10(sum(10^(L_i / 10))) (incoherent sources)"
        return {"L_total_dB": 10 * math.log10(sum(10 ** (float(x) / 10) for x in lv))}
    if mode == "inverse_square":
        L1 = need(p, "L1", None, positive=False)
        r1, r2 = need(p, "r1", "length"), need(p, "r2", "length")
        ctx.formula = "free field, point source: L2 = L1 - 20 log10(r2 / r1)"
        return {"L2_dB": L1 - 20 * math.log10(r2 / r1)}
    f, v = need(p, "f", "frequency"), need(p, "v", "velocity")
    ctx.formula = "lambda = v / f; T = 1 / f"
    return {"wavelength": v / f, "period": 1 / f}


@op("optics_acoustics", set().union(*_OA_MODES.values()) | {"mode"})
def _optics_acoustics(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_OA_MODES))
    only(p, mode, _OA_MODES[mode])
    if mode in OA_FUNCS:
        return OA_FUNCS[mode](p, ctx)
    out = _optics(p, mode, ctx)
    ctx.units = units_for(
        out,
        {
            "theta2_deg": "deg",
            "critical_angle_deg": "deg",
            "f": "m",
            "d_o": "m",
            "d_i": "m",
            "magnification": "1",
            "dB": "dB",
            "ratio": "1",
            "SPL_dB": "dB",
            "p": "Pa",
            "L_total_dB": "dB",
            "L2_dB": "dB",
            "wavelength": "m",
            "period": "s",
        },
    )
    ctx.formatted = (
        "; ".join(f"{k} = {fmt_float(v, 4)}" for k, v in out.items() if isinstance(v, float)) + "."
    )
    return out
