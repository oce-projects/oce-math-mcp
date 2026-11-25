"""eng_calc area B (D39): kinematics and dynamics, vibrations, bearings, belts, CNC, gears.

Formulas in the server; the tests check them with independent methods (numerical integration of the
motion, geometric construction of the belt and of the line of action of the gears).
"""

from __future__ import annotations

import math
from typing import Any, cast

import sympy as sp

from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import choice, integer, mm, need, one_of, only, op, qty, units_for
from .eng_dyn import KD_FUNCS, KD_MODES

G0 = 9.80665  # standard acceleration of gravity (m/s2), conventional value CGPM 1901

# ------------------------------------------------------------------ kinematics_dynamics

_MODES = {
    "projectile": {"v0", "angle_deg", "h0", "g"},
    "uniform_acceleration": {"v0", "v", "a", "t", "s"},
    "energy": {"m", "v", "h", "g", "k", "x", "I_m", "omega"},
    "rotation": {"P", "T", "omega"},
    "steiner": {"I_m", "m", "d"},
    "circular": {"m", "v", "omega", "r"},
}
_MODES |= KD_MODES  # D45
_KD_KEYS = set().union(*_MODES.values()) | {"mode"}


def _projectile(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    v0 = need(p, "v0", "velocity")
    ang = need(p, "angle_deg", None, positive=False)
    if not -90 < ang < 90:
        raise ToolError("INVALID_ARGUMENT", "params.angle_deg must be between -90 and 90 degrees.")
    h0 = qty(p, "h0", "length", positive=False, default=0.0) or 0.0
    if h0 < 0:
        raise ToolError(
            "INVALID_ARGUMENT", "params.h0 (starting height above the ground) must be >= 0."
        )
    g = qty(p, "g", "acceleration", default=G0) or G0
    th = math.radians(ang)
    vx, vy = v0 * math.cos(th), v0 * math.sin(th)
    t = (vy + math.sqrt(vy * vy + 2 * g * h0)) / g
    if t <= 0:
        raise ToolError(
            "INVALID_ARGUMENT", "The projectile starts from the ground going downwards: no flight."
        )
    vyi = vy - g * t
    out = {
        "t_flight": t,
        "range": vx * t,
        "t_apex": max(vy / g, 0.0),
        "h_max": h0 + vy * vy / (2 * g) if vy > 0 else h0,
        "v_impact": math.hypot(vx, vyi),
        "impact_angle_deg": math.degrees(math.atan2(-vyi, vx)),
    }
    ctx.units = units_for(
        out,
        {
            "t_flight": "s",
            "range": "m",
            "t_apex": "s",
            "h_max": "m",
            "v_impact": "m/s",
            "impact_angle_deg": "deg",
        },
    )
    ctx.formula = (
        "motion without air drag: vx = v0 cos(a), vy = v0 sin(a); t_flight = (vy + sqrt(vy^2 + 2 g h0)) / g; "
        "range = vx t_flight; h_max = h0 + vy^2 / (2 g)"
    )
    ctx.formatted = (
        f"Range {fmt_float(out['range'], 3)} m in {fmt_float(t, 3)} s, maximum height {fmt_float(out['h_max'], 3)} m, "
        f"impact speed {fmt_float(out['v_impact'], 3)} m/s. Without air drag."
    )
    return out


def _uniform(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    names = ("v0", "v", "a", "t", "s")
    kinds = {
        "v0": "velocity",
        "v": "velocity",
        "a": "acceleration",
        "t": "time",
        "s": "length",
    }
    known = {k: need(p, k, kinds[k], positive=k == "t") for k in names if p.get(k) is not None}
    if len(known) != 3:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Uniformly accelerated motion: exactly 3 of v0, v, a, t, s are needed.",
        )
    sym = {k: sp.Symbol(k, real=True) for k in names}
    eqs = [
        sp.Eq(sym["v"], sym["v0"] + sym["a"] * sym["t"]),
        sp.Eq(sym["s"], sym["v0"] * sym["t"] + sym["a"] * sym["t"] ** 2 / 2),
    ]
    values = cast(Any, {sym[k]: sp.Float(v, 30) for k, v in known.items()})
    eqs = [e.subs(values) for e in eqs]
    unknown = [sym[k] for k in names if k not in known]
    sols = []
    for sol in sp.solve(eqs, unknown, dict=True):
        vals = {str(k): complex(sp.N(v, 20)) for k, v in sol.items()}
        if any(abs(z.imag) > 1e-12 * max(1.0, abs(z.real)) for z in vals.values()):
            continue
        real = {k: z.real for k, z in vals.items()}
        if real.get("t", 0.0) < 0 or len(real) != len(unknown):
            continue
        sols.append(real)
    if not sols:
        raise ToolError("MATH_ERROR", "No solution with t >= 0 for these data.")
    sols.sort(key=lambda d: d.get("t", known.get("t", 0.0)))
    names_out = [str(u) for u in unknown]
    out: dict[str, Any] = {
        k: (sols[0][k] if len(sols) == 1 else [d[k] for d in sols]) for k in names_out
    }
    if len(sols) > 1:
        ctx.warn(f"{len(sols)} solutions with t >= 0: the values are lists in the same order.")
    ctx.units = units_for(out, {"v0": "m/s", "v": "m/s", "a": "m/s2", "t": "s", "s": "m"})
    ctx.formula = "v = v0 + a t; s = v0 t + a t^2 / 2"
    ctx.formatted = "; ".join(f"{k} = {v}" for k, v in out.items()) + " (SI units)."
    return out


def _energy(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = {}
    formulas = []
    g = qty(p, "g", "acceleration", default=G0) or G0
    m = qty(p, "m", "mass", need=False)
    if p.get("v") is not None:
        out["E_kin"] = (
            0.5
            * (m if m is not None else need(p, "m", "mass"))
            * need(p, "v", "velocity", positive=False) ** 2
        )
        formulas.append("E_cin = m v^2 / 2")
    if p.get("h") is not None:
        out["E_pot"] = (
            (m if m is not None else need(p, "m", "mass"))
            * g
            * need(p, "h", "length", positive=False)
        )
        formulas.append("E_pot = m g h")
    if p.get("k") is not None or p.get("x") is not None:
        out["E_spring"] = (
            0.5 * need(p, "k", "stiffness") * need(p, "x", "length", positive=False) ** 2
        )
        formulas.append("E_spring = k x^2 / 2")
    if p.get("I_m") is not None or p.get("omega") is not None:
        out["E_rot"] = (
            0.5
            * need(p, "I_m", "mass_moment")
            * need(p, "omega", "angular_velocity", positive=False) ** 2
        )
        formulas.append("E_rot = I omega^2 / 2")
    if not out:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Energy: m with v or h, or k with x, or I_m with omega are needed.",
        )
    out["E_total"] = sum(out.values())
    ctx.units = {k: "J" for k in out}
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"Total energy {fmt_float(out['E_total'], 3)} J."
    return out


def _rotation(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    given = [k for k in ("P", "T", "omega") if p.get(k) is not None]
    if len(given) != 2:
        raise ToolError(
            "INVALID_ARGUMENT", "Rotation: exactly 2 of P, T, omega are needed (P = T omega)."
        )
    if "P" not in given:
        out = {"P": need(p, "T", "torque") * need(p, "omega", "angular_velocity")}
    elif "T" not in given:
        out = {"T": need(p, "P", "power") / need(p, "omega", "angular_velocity")}
    else:
        out = {"omega": need(p, "P", "power") / need(p, "T", "torque")}
        out["omega_rpm"] = out["omega"] * 60 / (2 * math.pi)
    ctx.units = units_for(out, {"P": "W", "T": "Nm", "omega": "rad/s", "omega_rpm": "rpm"})
    ctx.formula = "P = T omega (omega in rad/s; revolutions per minute: omega = 2 pi n / 60)"
    ctx.formatted = "; ".join(f"{k} = {fmt_float(v, 4)}" for k, v in out.items()) + " (SI units)."
    return out


def _steiner(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    Ic, m, d = (
        need(p, "I_m", "mass_moment"),
        need(p, "m", "mass"),
        need(p, "d", "length", positive=False),
    )
    out = {"I_axis": Ic + m * d * d}
    ctx.units = {"I_axis": "kg*m2"}
    ctx.formula = "I = I_centroid + m d^2 (Huygens-Steiner, parallel axes)"
    ctx.formatted = f"I = {fmt_float(out['I_axis'], 6)} kg·m²."
    return out


def _circular(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    r = need(p, "r", "length")
    if one_of(p, ("v", "omega"), "the circular motion") == "v":
        v = need(p, "v", "velocity")
        w = v / r
    else:
        w = need(p, "omega", "angular_velocity")
        v = w * r
    out: dict[str, Any] = {"v": v, "omega": w, "a_c": v * v / r, "period": 2 * math.pi / w}
    m = qty(p, "m", "mass", need=False)
    if m is not None:
        out["F_c"] = m * v * v / r
    ctx.units = units_for(
        out, {"v": "m/s", "omega": "rad/s", "a_c": "m/s2", "period": "s", "F_c": "N"}
    )
    ctx.formula = "a_c = v^2 / r = omega^2 r; F_c = m v^2 / r; T = 2 pi / omega"
    ctx.formatted = f"Centripetal acceleration {fmt_float(out['a_c'], 4)} m/s²" + (
        f", centripetal force {fmt_float(out['F_c'], 3)} N." if "F_c" in out else "."
    )
    return out


@op("kinematics_dynamics", _KD_KEYS)
def _kinematics(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    fn = {
        "projectile": _projectile,
        "uniform_acceleration": _uniform,
        "energy": _energy,
        "rotation": _rotation,
        "steiner": _steiner,
        "circular": _circular,
        **KD_FUNCS,
    }[mode]
    return fn(p, ctx)


# ------------------------------------------------------------------ vibration_1dof


@op("vibration_1dof", ("m", "k", "c", "zeta", "F0", "omega", "f"))
def _vibration(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m, k = need(p, "m", "mass"), need(p, "k", "stiffness")
    wn = math.sqrt(k / m)
    cc = 2 * math.sqrt(k * m)
    if one_of(p, ("c", "zeta"), "the damping") == "c":
        c = need(p, "c", "damping", positive=False)
        if c < 0:
            raise ToolError("INVALID_ARGUMENT", "params.c must be >= 0.")
        z = c / cc
    else:
        z = need(p, "zeta", None, positive=False)
        if z < 0:
            raise ToolError("INVALID_ARGUMENT", "params.zeta must be >= 0.")
        c = z * cc
    under = z < 1
    # sqrt(1 - zeta^2) as sqrt((1 - zeta)(1 + zeta)): no cancellation close to 1 (D78)
    wd = wn * math.sqrt((1 - z) * (1 + z)) if under else None
    out: dict[str, Any] = {
        "omega_n": wn,
        "f_n": wn / (2 * math.pi),
        "T_n": 2 * math.pi / wn,
        "c_c": cc,
        "c": c,
        "zeta": z,
        "regime": "underdamped" if under else "critically damped" if z == 1 else "overdamped",
        "omega_d": wd,
        "f_d": wd / (2 * math.pi) if wd else None,
        "T_d": 2 * math.pi / wd if wd else None,
        "log_decrement": 2 * math.pi * z / math.sqrt((1 - z) * (1 + z)) if under else None,
        "Q": 1 / (2 * z) if z > 0 else None,
    }
    formulas = [
        "omega_n = sqrt(k / m); c_c = 2 sqrt(k m); zeta = c / c_c",
        "omega_d = omega_n sqrt(1 - zeta^2); delta = 2 pi zeta / sqrt(1 - zeta^2); Q = 1 / (2 zeta)",
    ]
    if p.get("omega") is not None or p.get("f") is not None:
        w = (
            need(p, "omega", "angular_velocity")
            if one_of(p, ("omega", "f"), "the forcing") == "omega"
            else 2 * math.pi * need(p, "f", "frequency")
        )
        r = w / wn
        den = math.hypot(1 - r * r, 2 * z * r)
        if den == 0:
            raise ToolError("MATH_ERROR", "Resonance without damping: infinite amplitude.")
        M = 1 / den
        TR = math.sqrt(1 + (2 * z * r) ** 2) * M
        out |= {
            "r": r,
            "amplification": M,
            "phase_deg": math.degrees(math.atan2(2 * z * r, 1 - r * r)),
            "transmissibility": TR,
        }
        formulas.append(
            "r = omega / omega_n; M = 1 / sqrt((1 - r^2)^2 + (2 zeta r)^2); TR = sqrt(1 + (2 zeta r)^2) M"
        )
        F0 = qty(p, "F0", "force", need=False)
        if F0 is not None:
            out |= {"X": F0 / k * M, "F_T": F0 * TR}
            formulas.append("X = F0 / k * M; F_T = F0 TR")
    elif p.get("F0") is not None:
        raise ToolError(
            "INVALID_ARGUMENT",
            "With F0 the angular frequency omega or the frequency f of the forcing is needed.",
        )
    ctx.units = units_for(
        out,
        {
            "omega_n": "rad/s",
            "f_n": "Hz",
            "T_n": "s",
            "c_c": "N*s/m",
            "c": "N*s/m",
            "zeta": "1",
            "omega_d": "rad/s",
            "f_d": "Hz",
            "T_d": "s",
            "log_decrement": "1",
            "Q": "1",
            "r": "1",
            "amplification": "1",
            "phase_deg": "deg",
            "transmissibility": "1",
            "X": "m",
            "F_T": "N",
        },
    )
    ctx.formula = "; ".join(formulas)
    text = f"Natural frequency {fmt_float(out['f_n'], 4)} Hz, zeta = {fmt_float(z, 4)} ({out['regime']})"
    if "X" in out:
        text += f", steady-state amplitude {mm(out['X'])}"
    ctx.formatted = text + "."
    return out


# ------------------------------------------------------------------ bearing_life


@op("bearing_life", ("C", "P", "Fr", "Fa", "X", "Y", "type", "p", "n", "a1", "L10h_req"))
def _bearing(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if one_of(p, ("type", "p"), "the exponent (balls 3, rollers 10/3)") == "type":
        expo = 3.0 if choice(p, "type", ("ball", "roller")) == "ball" else 10 / 3
    else:
        expo = need(p, "p", None)
    formulas = ["L10 = (C / P)^p million revolutions (p = 3 balls, 10/3 rollers)"]
    if p.get("P") is not None and p.get("Fr") is not None:
        raise ToolError("INVALID_ARGUMENT", "Give P or Fr (with Fa, X, Y), not both.")
    if p.get("Fr") is not None:
        P = (qty(p, "X", None, default=1.0) or 1.0) * need(p, "Fr", "force", positive=False) + (
            qty(p, "Y", None, positive=False, default=0.0) or 0.0
        ) * (qty(p, "Fa", "force", positive=False, default=0.0) or 0.0)
        formulas.insert(0, "P = X Fr + Y Fa")
        if P <= 0:
            raise ToolError("INVALID_ARGUMENT", "The equivalent load P is <= 0.")
    else:
        P = need(p, "P", "force")
    out: dict[str, Any] = {"P": P, "p": expo}
    n = qty(p, "n", "angular_velocity", need=False)
    rev_s = n / (2 * math.pi) if n is not None else None
    a1 = qty(p, "a1", None, default=1.0) or 1.0
    req = qty(p, "L10h_req", "time", need=False)
    if req is not None:
        if rev_s is None:
            raise ToolError("INVALID_ARGUMENT", "L10h_req needs the speed n.")
        out["C_required"] = P * (req * rev_s / 1e6 / a1) ** (1 / expo)
        formulas.append("C_required = P (L_rev / (a1 10^6))^(1/p)")
    C = qty(p, "C", "force", need=req is None)
    if C is not None:
        L10 = (C / P) ** expo * 1e6
        out |= {"L10": L10, "L10_mrev": L10 / 1e6}
        if a1 != 1.0:
            out["L_na"] = a1 * L10
            formulas.append("L_na = a1 L10")
        if rev_s is not None:
            out["L10h"] = L10 / (rev_s * 3600)
            if a1 != 1.0:
                out["L_nah"] = out["L_na"] / (rev_s * 3600)
            formulas.append("L10h = L10 / (60 n_rpm)")
    ctx.units = units_for(
        out,
        {
            "P": "N",
            "p": "1",
            "C_required": "N",
            "L10": "rev",
            "L10_mrev": "million rev",
            "L_na": "rev",
            "L10h": "h",
            "L_nah": "h",
        },
    )
    ctx.formula = "; ".join(formulas)
    parts = []
    if "L10h" in out:
        parts.append(f"L10 life {fmt_float(out['L10h'], 1)} hours")
    elif "L10" in out:
        parts.append(f"L10 life {fmt_float(out['L10_mrev'], 3)} million revolutions")
    if "C_required" in out:
        parts.append(f"required dynamic load rating {fmt_float(out['C_required'] / 1000, 3)} kN")
    ctx.formatted = "; ".join(parts).capitalize() + "."
    return out


# ------------------------------------------------------------------ belt_drive


@op("belt_drive", ("D", "d", "C", "n", "P", "mu", "groove_deg", "m_l"))
def _belt(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    D, d, C = need(p, "D", "length"), need(p, "d", "length"), need(p, "C", "length")
    if d > D:
        raise ToolError("INVALID_ARGUMENT", "params.D is the large pulley: it must be >= d.")
    if (D + d) / 2 >= C:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Centre distance too small: the pulleys touch (C > (D + d) / 2 is needed).",
        )
    alpha = math.asin((D - d) / (2 * C))
    ts, tl = math.pi - 2 * alpha, math.pi + 2 * alpha
    out: dict[str, Any] = {
        "L": 2 * C * math.cos(alpha) + ts * d / 2 + tl * D / 2,
        "theta_small": ts,
        "theta_small_deg": math.degrees(ts),
        "theta_large": tl,
        "ratio": D / d,
    }
    formulas = [
        "sin(alpha) = (D - d) / (2 C); theta_small = pi - 2 alpha; L = 2 C cos(alpha) + theta_s d / 2 + theta_l D / 2 (exact)"
    ]
    n = qty(p, "n", "angular_velocity", need=False)
    if n is not None:
        v = n * d / 2
        out |= {"v": v, "n_large": n * d / D}
        formulas.append("v = omega d / 2; n_large = n d / D")
    Pw = qty(p, "P", "power", need=False)
    if Pw is not None:
        if n is None:
            raise ToolError(
                "INVALID_ARGUMENT", "The power P needs the speed n of the small pulley."
            )
        out["F_eff"] = Pw / out["v"]
        formulas.append("F_eff = T1 - T2 = P / v")
    mu = qty(p, "mu", None, need=False)
    if mu is not None:
        groove = qty(p, "groove_deg", None, need=False)
        mu_eff = mu / math.sin(math.radians(groove) / 2) if groove is not None else mu
        ratio = math.exp(mu_eff * ts)
        out |= {"mu_eff": mu_eff, "tension_ratio": ratio}
        formulas.append(
            "T1 / T2 = e^(mu theta) (Euler-Eytelwein); V-groove: mu' = mu / sin(beta / 2)"
        )
        if "F_eff" in out:
            m_l = qty(p, "m_l", "linear_mass", need=False)
            Fc = m_l * out["v"] ** 2 if m_l is not None else 0.0
            T1 = out["F_eff"] * ratio / (ratio - 1) + Fc
            T2 = out["F_eff"] / (ratio - 1) + Fc
            out |= {"T1": T1, "T2": T2, "T0": (T1 + T2) / 2}
            if m_l is not None:
                out["F_centrifugal"] = Fc
                formulas.append("F_c = m_l v^2 added to T1 and T2")
        elif p.get("m_l") is not None:  # D92: it was ignored in silence
            raise ToolError(
                "INVALID_ARGUMENT", "m_l (centrifugal tension) needs P and n, for T1 and T2."
            )
    elif p.get("groove_deg") is not None or p.get("m_l") is not None:
        raise ToolError("INVALID_ARGUMENT", "groove_deg and m_l are used with mu (friction).")
    ctx.units = units_for(
        out,
        {
            "L": "m",
            "theta_small": "rad",
            "theta_small_deg": "deg",
            "theta_large": "rad",
            "ratio": "1",
            "v": "m/s",
            "n_large": "rad/s",
            "F_eff": "N",
            "mu_eff": "1",
            "tension_ratio": "1",
            "T1": "N",
            "T2": "N",
            "T0": "N",
            "F_centrifugal": "N",
        },
    )
    ctx.formula = "; ".join(formulas)
    text = f"Belt length {mm(out['L'])}, wrap angle {fmt_float(math.degrees(ts), 2)} degrees"
    if "T1" in out:
        text += f", T1 = {fmt_float(out['T1'], 1)} N, T2 = {fmt_float(out['T2'], 1)} N"
    ctx.formatted = text + "."
    return out


# ------------------------------------------------------------------ cnc_machining


@op("cnc_machining", ("mode", "Vc", "n", "D", "f", "fz", "z", "ap", "ae", "kc", "eta"))
def _cnc(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", ("turning", "milling", "drilling"))
    allowed = {
        "turning": {"Vc", "n", "D", "f", "ap", "kc", "eta"},
        "milling": {"Vc", "n", "D", "fz", "z", "ap", "ae", "kc", "eta"},
        "drilling": {"Vc", "n", "D", "f", "kc", "eta"},
    }[mode]
    only(p, mode, allowed)
    D = need(p, "D", "length")
    if one_of(p, ("Vc", "n"), "the speed (cutting speed Vc or spindle speed n)") == "Vc":
        Vc = need(p, "Vc", "velocity")
        rev_s = Vc / (math.pi * D)
    else:
        rev_s = need(p, "n", "angular_velocity") / (2 * math.pi)
        Vc = math.pi * D * rev_s
    out: dict[str, Any] = {
        "n": 2 * math.pi * rev_s,
        "n_rpm": rev_s * 60,
        "Vc": Vc,
        "Vc_m_min": Vc * 60,
    }
    formulas = ["n = Vc / (pi D)"]
    kc = qty(p, "kc", "pressure", need=False)
    if mode == "milling":
        Vf = need(p, "fz", "length") * integer(p, "z") * rev_s
        formulas.append("Vf = fz z n")
        mrr = (
            need(p, "ap", "length") * need(p, "ae", "length") * Vf
            if p.get("ap") or p.get("ae")
            else None
        )
        if mrr is not None:
            formulas.append("MRR = ap ae Vf")
    else:
        f = need(p, "f", "length")
        Vf = f * rev_s
        formulas.append("Vf = f n")
        if mode == "turning":
            ap = qty(p, "ap", "length", need=False)
            mrr = Vc * f * ap if ap is not None else None
            if ap is not None:
                formulas.append("MRR = Vc f ap")
                if kc is not None:
                    out["F_c"] = kc * f * ap
                    formulas.append("F_c = kc f ap")
        else:
            mrr = math.pi * D * D / 4 * Vf
            formulas.append("MRR = pi D^2 / 4 Vf")
    out |= {"Vf": Vf, "Vf_mm_min": Vf * 60000}
    if mrr is not None:
        out |= {"MRR": mrr, "MRR_cm3_min": mrr * 6e7}
        if kc is not None:
            out["P_cut"] = kc * mrr
            out["T_spindle"] = out["P_cut"] / out["n"]
            formulas.append("P_cut = kc MRR; torque = P / omega")
            eta = qty(p, "eta", None, need=False)
            if eta is not None:
                if eta > 1:
                    raise ToolError(
                        "INVALID_ARGUMENT", "params.eta (efficiency) must be between 0 and 1."
                    )
                out["P_motor"] = out["P_cut"] / eta
                formulas.append("P_motor = P_cut / eta")
    ctx.units = units_for(
        out,
        {
            "n": "rad/s",
            "n_rpm": "rpm",
            "Vc": "m/s",
            "Vc_m_min": "m/min",
            "Vf": "m/s",
            "Vf_mm_min": "mm/min",
            "MRR": "m3/s",
            "MRR_cm3_min": "cm3/min",
            "F_c": "N",
            "P_cut": "W",
            "T_spindle": "Nm",
            "P_motor": "W",
        },
    )
    ctx.formula = "; ".join(formulas)
    text = (
        f"n = {fmt_float(out['n_rpm'], 1)} rpm, Vc = {fmt_float(out['Vc_m_min'], 1)} m/min, "
        f"feed {fmt_float(out['Vf_mm_min'], 1)} mm/min"
    )
    if "P_cut" in out:
        text += f", cutting power {fmt_float(out['P_cut'] / 1000, 3)} kW"
    ctx.formatted = text + "."
    return out


# ------------------------------------------------------------------ gear_spur


@op("gear_spur", ("m", "z1", "z2", "alpha_deg", "beta_deg", "T1", "P", "n1", "b", "Y"))
def _gear(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m = need(p, "m", "length")
    z1, z2 = integer(p, "z1"), integer(p, "z2")
    alpha = math.radians(qty(p, "alpha_deg", None, default=20.0) or 20.0)
    if not 0 < alpha < math.radians(45):  # D92: 90 degrees gave a contact ratio of 1e16
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.alpha_deg (pressure angle) must be between 0 and 45 degrees (usually 20).",
        )
    beta = math.radians(qty(p, "beta_deg", None, positive=False, default=0.0) or 0.0)
    if not 0 <= beta < math.radians(45):
        raise ToolError(
            "INVALID_ARGUMENT", "params.beta_deg (helix) must be between 0 and 45 degrees."
        )
    mt = m / math.cos(beta)
    at = math.atan(math.tan(alpha) / math.cos(beta))
    d1, d2 = mt * z1, mt * z2
    ra1, ra2, rb1, rb2 = d1 / 2 + m, d2 / 2 + m, d1 / 2 * math.cos(at), d2 / 2 * math.cos(at)
    a = (d1 + d2) / 2
    eps = (math.sqrt(ra1**2 - rb1**2) + math.sqrt(ra2**2 - rb2**2) - a * math.sin(at)) / (
        math.pi * mt * math.cos(at)
    )
    out: dict[str, Any] = {
        "ratio": z2 / z1,
        "m_t": mt,
        "d1": d1,
        "d2": d2,
        "da1": d1 + 2 * m,
        "da2": d2 + 2 * m,
        "df1": d1 - 2.5 * m,
        "df2": d2 - 2.5 * m,
        "db1": 2 * rb1,
        "db2": 2 * rb2,
        "a": a,
        "contact_ratio": eps,
    }
    formulas = [
        "m_t = m / cos(beta); d = m_t z; d_tip = d + 2 m; d_root = d - 2.5 m; a = (d1 + d2) / 2",
        "epsilon = (sqrt(ra1^2 - rb1^2) + sqrt(ra2^2 - rb2^2) - a sin(alfa_t)) / (pi m_t cos(alfa_t))",
    ]
    if eps < 1:
        ctx.warn(f"Contact ratio {eps:.3g} < 1: the gear does not transmit continuously.")
    if p.get("T1") is not None and p.get("P") is not None:
        raise ToolError("INVALID_ARGUMENT", "Give T1 or P with n1, not both.")
    n1 = qty(p, "n1", "angular_velocity", need=False)
    if n1 is not None:
        out |= {"n2": n1 * z1 / z2, "n2_rpm": n1 * z1 / z2 * 60 / (2 * math.pi)}
        formulas.append("n2 = n1 z1 / z2")
    T1 = qty(p, "T1", "torque", need=False)
    if p.get("P") is not None:
        if n1 is None:
            raise ToolError("INVALID_ARGUMENT", "The power P needs the speed n1 of the pinion.")
        T1 = need(p, "P", "power") / n1
        out["T1"] = T1
        formulas.append("T1 = P / omega1")
    if T1 is not None:
        Ft = 2 * T1 / d1
        out |= {"T2": T1 * z2 / z1, "Ft": Ft, "Fr": Ft * math.tan(at), "Fa": Ft * math.tan(beta)}
        formulas.append(
            "Ft = 2 T1 / d1; Fr = Ft tan(alpha_t); Fa = Ft tan(beta); T2 = T1 z2 / z1 (no losses)"
        )
        b = qty(p, "b", "length", need=False)
        Y = qty(p, "Y", None, need=False)
        if (b is None) != (Y is None):
            raise ToolError(
                "INVALID_ARGUMENT",
                "The Lewis check needs b (face width) and Y (form factor) together.",
            )
        if b is not None and Y is not None:
            out["sigma_lewis"] = Ft / (b * m * Y)
            formulas.append("sigma_Lewis = Ft / (b m Y)")
            if beta > 0:
                out["overlap_ratio"] = b * math.sin(beta) / (math.pi * m)
                formulas.append("epsilon_beta = b sin(beta) / (pi m)")
    elif p.get("b") is not None or p.get("Y") is not None:
        raise ToolError("INVALID_ARGUMENT", "b and Y are used with the torque T1 (or P with n1).")
    ctx.units = units_for(
        out,
        {
            "ratio": "1",
            "m_t": "m",
            "d1": "m",
            "d2": "m",
            "da1": "m",
            "da2": "m",
            "df1": "m",
            "df2": "m",
            "db1": "m",
            "db2": "m",
            "a": "m",
            "contact_ratio": "1",
            "n2": "rad/s",
            "n2_rpm": "rpm",
            "T1": "Nm",
            "T2": "Nm",
            "Ft": "N",
            "Fr": "N",
            "Fa": "N",
            "sigma_lewis": "Pa",
            "overlap_ratio": "1",
        },
    )
    ctx.formula = "; ".join(formulas)
    text = f"Pitch diameters {mm(d1)} and {mm(d2)}, centre distance {mm(a)}, contact ratio {fmt_float(eps, 3)}"
    if "Ft" in out:
        text += f", tangential force {fmt_float(out['Ft'], 1)} N"
    ctx.formatted = text + "."
    return out
