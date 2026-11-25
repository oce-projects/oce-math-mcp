"""eng_calc area C (D39): fluids and heat. Formulas in the server, checked in the tests with independent methods.

Constants: g = 9.80665 m/s2 (standard gravity); sigma = 5.670374419e-8 W/(m2 K4) (Stefan-Boltzmann,
CODATA 2018, exact in the SI since 2019); dry air gamma = 1.4 and R = 287.05287 J/(kg K) (ISO 2533).
"""

from __future__ import annotations

import math
from typing import Any, cast

from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import choice, need, one_of, only, op, qty, temperature, units_for

G0 = 9.80665
SIGMA_SB = 5.670374419e-8
GAMMA_AIR, R_AIR = 1.4, 287.05287


# ------------------------------------------------------------------ fluid_pipe


def colebrook(Re: float, rr: float) -> float:
    """Darcy friction factor from the Colebrook-White formula, solved with brentq on x = 1/sqrt(f)."""
    from scipy.optimize import brentq

    def g(x: float) -> float:
        return x + 2 * math.log10(rr / 3.7 + 2.51 * x / Re)

    x = float(
        cast(Any, brentq)(g, 0.5, 60.0, xtol=1e-15, rtol=4 * 2.220446049250313e-16, maxiter=500)
    )
    return 1 / (x * x)


@op("fluid_pipe", ("D", "L", "Q", "v", "rho", "mu", "nu", "eps", "K", "dz", "p1"))
def _pipe(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    D = need(p, "D", "length")
    A = math.pi * D * D / 4
    if one_of(p, ("Q", "v"), "the flow rate (Q) or the velocity (v)") == "Q":
        Q = need(p, "Q", "flow_rate")
        v = Q / A
    else:
        v = need(p, "v", "velocity")
        Q = v * A
    rho = need(p, "rho", "density")
    if one_of(p, ("mu", "nu"), "the viscosity (dynamic mu or kinematic nu)") == "mu":
        nu = need(p, "mu", "dynamic_viscosity") / rho
    else:
        nu = need(p, "nu", "kinematic_viscosity")
    eps = qty(p, "eps", "length", positive=False, default=0.0) or 0.0
    if eps < 0:
        raise ToolError("INVALID_ARGUMENT", "params.eps (absolute roughness) must be >= 0.")
    Re, rr = v * D / nu, eps / D
    formulas = ["Re = v D / nu"]
    if Re < 2300:
        regime, f = "laminar", 64 / Re
        formulas.append("laminar: f = 64 / Re (Hagen-Poiseuille)")
    else:
        regime = "turbulent" if Re > 4000 else "transitional"
        f = colebrook(Re, rr)
        formulas.append(
            "f from Colebrook-White: 1/sqrt(f) = -2 log10(eps / (3.7 D) + 2.51 / (Re sqrt(f)))"
        )
        if regime == "transitional":
            ctx.warn(f"Re = {Re:.0f} in transition (2300-4000): f from Colebrook, uncertain value.")
    dyn = rho * v * v / 2
    out: dict[str, Any] = {
        "A": A,
        "v": v,
        "Q": Q,
        "Re": Re,
        "regime": regime,
        "relative_roughness": rr,
        "f": f,
    }
    dp_f = dp_l = 0.0
    L = qty(p, "L", "length", need=False)
    if L is not None:
        dp_f = f * L / D * dyn
        out |= {"dp_friction": dp_f, "h_friction": dp_f / (rho * G0)}
        formulas.append("dp = f (L / D) rho v^2 / 2 (Darcy-Weisbach); h = dp / (rho g)")
    K = qty(p, "K", None, positive=False, need=False)
    if K is not None:
        if K < 0:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.K (sum of the minor loss coefficients) must be >= 0.",
            )
        dp_l = K * dyn
        out |= {"dp_local": dp_l, "h_local": dp_l / (rho * G0)}
        formulas.append("minor losses: dp = K rho v^2 / 2")
    out["dp_total"] = dp_f + dp_l
    dz = qty(p, "dz", "length", positive=False, need=False)
    if dz is not None:
        H = (dp_f + dp_l) / (rho * G0) + dz
        out |= {"H_pump": H, "P_hydraulic": rho * G0 * Q * H}
        formulas.append("head H = h_friction + h_local + dz; P = rho g Q H")
    p1 = qty(p, "p1", "pressure", positive=False, need=False)
    if p1 is not None:
        out["p2"] = p1 - dp_f - dp_l - rho * G0 * (dz or 0.0)
        formulas.append("Bernoulli with losses (same cross-section): p2 = p1 - dp - rho g dz")
    ctx.units = units_for(
        out,
        {
            "A": "m2",
            "v": "m/s",
            "Q": "m3/s",
            "Re": "1",
            "relative_roughness": "1",
            "f": "1",
            "dp_friction": "Pa",
            "h_friction": "m",
            "dp_local": "Pa",
            "h_local": "m",
            "dp_total": "Pa",
            "H_pump": "m",
            "P_hydraulic": "W",
            "p2": "Pa",
        },
    )
    ctx.formula = "; ".join(formulas)
    text = f"Re = {fmt_float(Re, 0)} ({regime}), f = {fmt_float(f, 5)}, v = {fmt_float(v, 3)} m/s"
    if L is not None:
        text += f", friction loss {fmt_float(dp_f / 1e5, 4)} bar"
    ctx.formatted = text + "."
    return out


# ------------------------------------------------------------------ aero_hydro_force


@op("aero_hydro_force", ("rho", "v", "A", "Cd", "Cl", "T", "c", "m"))
def _drag(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    rho, A = need(p, "rho", "density"), need(p, "A", "area")
    v = qty(p, "v", "velocity", need=False)
    m = qty(p, "m", "mass", need=False)
    if v is None and m is None:
        raise ToolError(
            "INVALID_ARGUMENT",
            "The velocity v (forces) or the mass m (terminal falling velocity) is needed.",
        )
    Cd = qty(p, "Cd", None, need=False)
    Cl = qty(p, "Cl", None, positive=False, need=False)
    out: dict[str, Any] = {}
    formulas = []
    if v is not None:
        q = rho * v * v / 2
        out["q"] = q
        formulas.append("q = rho v^2 / 2")
        if Cd is not None:
            out |= {"F_D": q * Cd * A, "P_drag": q * Cd * A * v}
            formulas.append("F_D = q Cd A; P = F_D v")
        if Cl is not None:
            out["F_L"] = q * Cl * A
            formulas.append("F_L = q Cl A")
    if p.get("T") is not None or p.get("c") is not None:
        if one_of(p, ("T", "c"), "the speed of sound (air T or c)") == "T":
            T = temperature(p, "T")
            assert T is not None
            c = math.sqrt(GAMMA_AIR * R_AIR * T)
            formulas.append("c = sqrt(gamma R T), dry air gamma = 1.4, R = 287.05287 J/(kg K)")
        else:
            c = need(p, "c", "velocity")
        out["c_sound"] = c
        if v is not None:
            out["Mach"] = v / c
            formulas.append("Mach = v / c")
    if m is not None:
        if Cd is None:
            raise ToolError("INVALID_ARGUMENT", "The terminal velocity needs Cd.")
        out["v_terminal"] = math.sqrt(2 * m * G0 / (rho * Cd * A))
        formulas.append("v_terminal = sqrt(2 m g / (rho Cd A))")
    ctx.units = units_for(
        out,
        {
            "q": "Pa",
            "F_D": "N",
            "P_drag": "W",
            "F_L": "N",
            "c_sound": "m/s",
            "Mach": "1",
            "v_terminal": "m/s",
        },
    )
    ctx.formula = "; ".join(formulas)
    parts = []
    if "F_D" in out:
        parts.append(f"drag {fmt_float(out['F_D'], 2)} N")
    if "F_L" in out:
        parts.append(f"lift {fmt_float(out['F_L'], 2)} N")
    if "Mach" in out:
        parts.append(f"Mach {fmt_float(out['Mach'], 4)}")
    if "v_terminal" in out:
        parts.append(f"terminal velocity {fmt_float(out['v_terminal'], 3)} m/s")
    ctx.formatted = ("; ".join(parts) or "Dynamic pressure computed").capitalize() + "."
    return out


# ------------------------------------------------------------------ heat_transfer

_HT_MODES = {
    "wall": {"layers", "A", "h_in", "h_out", "T1", "T2"},
    "cylinder": {"r1", "r2", "k", "L", "h_in", "h_out", "T1", "T2"},
    "convection": {"h", "A", "T1", "T2"},
    "radiation": {"emissivity", "A", "T1", "T2"},
    "lmtd": {"flow", "Th_in", "Th_out", "Tc_in", "Tc_out", "U", "Q", "A"},
}


def _temps(p: dict[str, Any]) -> tuple[float, float]:
    t1, t2 = temperature(p, "T1"), temperature(p, "T2")
    assert t1 is not None and t2 is not None
    return t1, t2


def _wall(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    layers = p.get("layers")
    if (
        not isinstance(layers, list)
        or not layers
        or not all(
            isinstance(x, list)
            and len(x) == 2
            and all(isinstance(y, int | float) and not isinstance(y, bool) and y > 0 for y in x)
            for x in layers
        )
    ):
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.layers: list of [thickness in m, conductivity in W/(m K)], values > 0.",
            '{"layers": [[0.2, 0.7], [0.05, 0.04]]}',
        )
    A = need(p, "A", "area")
    res = []
    h_in, h_out = (
        qty(p, "h_in", "heat_transfer_coefficient", need=False),
        qty(p, "h_out", "heat_transfer_coefficient", need=False),
    )
    if h_in is not None:
        res.append(1 / (h_in * A))
    res += [float(t) / (float(k) * A) for t, k in layers]
    if h_out is not None:
        res.append(1 / (h_out * A))
    T1, T2 = _temps(p)
    R = sum(res)
    Q = (T1 - T2) / R
    temps = [T1]
    for r in res:
        temps.append(temps[-1] - Q * r)
    ctx.formula = (
        "R = 1 / (h_in A) + sum(s_i / (k_i A)) + 1 / (h_out A); Q = (T1 - T2) / R; U = 1 / (R A)"
    )
    return {"R_total": R, "U": 1 / (R * A), "Q": Q, "q": Q / A, "T_interfaces": temps}


def _cylinder(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    r1, r2 = need(p, "r1", "length"), need(p, "r2", "length")
    if r2 <= r1:
        raise ToolError("INVALID_ARGUMENT", "params.r2 (outer radius) must be greater than r1.")
    k, L = need(p, "k", "thermal_conductivity"), need(p, "L", "length")
    R = math.log(r2 / r1) / (2 * math.pi * k * L)
    h_in, h_out = (
        qty(p, "h_in", "heat_transfer_coefficient", need=False),
        qty(p, "h_out", "heat_transfer_coefficient", need=False),
    )
    if h_in is not None:
        R += 1 / (h_in * 2 * math.pi * r1 * L)
    if h_out is not None:
        R += 1 / (h_out * 2 * math.pi * r2 * L)
    T1, T2 = _temps(p)
    ctx.formula = "R = ln(r2 / r1) / (2 pi k L) (+ 1 / (h 2 pi r L) per i film); Q = (T1 - T2) / R"
    return {"R_total": R, "Q": (T1 - T2) / R, "Q_per_m": (T1 - T2) / R / L}


def _convection(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    h, A = need(p, "h", "heat_transfer_coefficient"), need(p, "A", "area")
    T1, T2 = _temps(p)
    ctx.formula = "Q = h A (T_surface - T_fluid) (Newton)"
    return {"R_total": 1 / (h * A), "Q": h * A * (T1 - T2), "q": h * (T1 - T2)}


def _radiation(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    eps = need(p, "emissivity", None)
    if eps > 1:
        raise ToolError("INVALID_ARGUMENT", "params.emissivity must be between 0 and 1.")
    A = need(p, "A", "area")
    T1, T2 = _temps(p)
    ctx.formula = "Q = eps sigma A (T1^4 - T2^4), sigma = 5.670374419e-8 W/(m2 K4); small surface in large surroundings"
    return {"Q": eps * SIGMA_SB * A * (T1**4 - T2**4), "q": eps * SIGMA_SB * (T1**4 - T2**4)}


def _lmtd(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    flow = choice(p, "flow", ("counter", "parallel"))
    Thi, Tho, Tci, Tco = (temperature(p, k) for k in ("Th_in", "Th_out", "Tc_in", "Tc_out"))
    assert Thi is not None and Tho is not None and Tci is not None and Tco is not None
    if not (Thi > Tho and Tco > Tci):
        raise ToolError(
            "INVALID_ARGUMENT", "The hot fluid must cool down and the cold fluid must warm up."
        )
    d1, d2 = (Thi - Tco, Tho - Tci) if flow == "counter" else (Thi - Tci, Tho - Tco)
    if d1 <= 0 or d2 <= 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Impossible temperatures for this exchanger (differences <= 0 at one end).",
        )
    lm = d1 if math.isclose(d1, d2, rel_tol=1e-12) else (d1 - d2) / math.log(d1 / d2)
    U = need(p, "U", "heat_transfer_coefficient")
    out: dict[str, Any] = {"LMTD": lm}
    if one_of(p, ("Q", "A"), "the heat rate (Q) or the area (A)") == "Q":
        Q = need(p, "Q", "power")
        out |= {"Q": Q, "A_required": Q / (U * lm)}
    else:
        A = need(p, "A", "area")
        Q = U * A * lm
        out |= {"Q": Q, "A": A}
    out |= {"C_hot": Q / (Thi - Tho), "C_cold": Q / (Tco - Tci)}
    ctx.formula = (
        "LMTD = (dT1 - dT2) / ln(dT1 / dT2); counterflow dT1 = Th_in - Tc_out, dT2 = Th_out - Tc_in; "
        "parallel flow dT1 = Th_in - Tc_in, dT2 = Th_out - Tc_out; Q = U A LMTD"
    )
    return out


@op("heat_transfer", set().union(*_HT_MODES.values()) | {"mode", "h", "k", "L", "r1", "r2"})
def _heat(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_HT_MODES))
    only(p, mode, _HT_MODES[mode])
    fn = {
        "wall": _wall,
        "cylinder": _cylinder,
        "convection": _convection,
        "radiation": _radiation,
        "lmtd": _lmtd,
    }[mode]
    out = fn(p, ctx)
    ctx.units = units_for(
        out,
        {
            "R_total": "K/W",
            "U": "W/(m2*K)",
            "Q": "W",
            "q": "W/m2",
            "T_interfaces": "K",
            "Q_per_m": "W/m",
            "LMTD": "K",
            "A_required": "m2",
            "A": "m2",
            "C_hot": "W/K",
            "C_cold": "W/K",
        },
    )
    text = f"Heat rate {fmt_float(out['Q'], 2)} W"
    if "A_required" in out:
        text += f", required area {fmt_float(out['A_required'], 4)} m²"
    ctx.formatted = text + "."
    return out


# ------------------------------------------------------------------ heatsink_thermal


@op("heatsink_thermal", ("P", "T_a", "R_jc", "R_cs", "R_sa", "T_j_max"))
def _heatsink(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    P = need(p, "P", "power")
    Ta = temperature(p, "T_a")
    assert Ta is not None
    Rjc = need(p, "R_jc", "thermal_resistance", positive=False)
    Rcs = qty(p, "R_cs", "thermal_resistance", positive=False, default=0.0) or 0.0
    if Rjc < 0 or Rcs < 0:
        raise ToolError("INVALID_ARGUMENT", "The thermal resistances must be >= 0.")
    Tjmax = temperature(p, "T_j_max", need=False)
    out: dict[str, Any] = {}
    formulas = ["T_j = T_a + P (R_jc + R_cs + R_sa)"]
    Rsa = qty(p, "R_sa", "thermal_resistance", positive=False, need=False)
    if Rsa is None:
        if Tjmax is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "R_sa (heat sink) is needed, or T_j_max to compute the required R_sa.",
            )
        Rsa = (Tjmax - Ta) / P - Rjc - Rcs
        if Rsa <= 0:
            raise ToolError(
                "MATH_ERROR",
                "Even with an ideal heat sink the junction exceeds T_j_max: reduce P or the other resistances.",
            )
        out["R_sa_required"] = Rsa
        formulas.append("R_sa required = (T_j_max - T_a) / P - R_jc - R_cs")
    Tj, Tc, Ts = Ta + P * (Rjc + Rcs + Rsa), Ta + P * (Rcs + Rsa), Ta + P * Rsa
    out |= {
        "R_total": Rjc + Rcs + Rsa,
        "T_j": Tj,
        "T_j_C": Tj - 273.15,
        "T_case": Tc,
        "T_case_C": Tc - 273.15,
        "T_sink": Ts,
        "T_sink_C": Ts - 273.15,
    }
    if Tjmax is not None and "R_sa_required" not in out:
        out["margin_K"] = Tjmax - Tj
        if Tj > Tjmax:
            ctx.warn("The junction exceeds T_j_max.")
    ctx.units = units_for(
        out,
        {
            "R_sa_required": "K/W",
            "R_total": "K/W",
            "T_j": "K",
            "T_j_C": "C",
            "T_case": "K",
            "T_case_C": "C",
            "T_sink": "K",
            "T_sink_C": "C",
            "margin_K": "K",
        },
    )
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"Junction temperature {fmt_float(Tj - 273.15, 2)} °C" + (
        f", required R_sa {fmt_float(Rsa, 3)} K/W." if "R_sa_required" in out else "."
    )
    return out
