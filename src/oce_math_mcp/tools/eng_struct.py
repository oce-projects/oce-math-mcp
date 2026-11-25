"""eng_calc area A (D39): strength of materials. Formulas in the server, checked in the tests."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import (
    choice,
    cube_diff,
    mm,
    mpa,
    need,
    one_of,
    op,
    pow4_diff,
    qty,
    sq_diff,
    units_for,
)

# ------------------------------------------------------------------ hooke_elastic


@op("hooke_elastic", ("E", "nu", "G", "sigma", "epsilon", "F", "A", "L", "tau", "gamma"))
def _hooke(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    E = need(p, "E", "pressure")
    nu = qty(p, "nu", None, need=False, positive=False)
    G = qty(p, "G", "pressure", need=False)
    if nu is not None and not -1 < nu < 0.5:
        raise ToolError("INVALID_ARGUMENT", "params.nu must be between -1 and 0.5.")
    formulas: list[str] = []
    if G is None and nu is not None:
        G = E / (2 * (1 + nu))
        formulas.append("G = E / (2 (1 + nu))")
    elif G is not None and nu is None:
        nu = E / (2 * G) - 1
        if not -1 < nu < 0.5:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"E and G give nu = {nu:.4g}, outside (-1; 0.5): inconsistent values.",
            )
        formulas.append("nu = E / (2 G) - 1")
    elif G is not None and nu is not None and abs(E / (2 * (1 + nu)) - G) > 1e-3 * G:
        ctx.warn("E, G and nu do not satisfy G = E / (2 (1 + nu)); the given values are used.")
    axial = [k for k in ("sigma", "epsilon", "F") if p.get(k) is not None]
    if len(axial) > 1:
        raise ToolError(
            "INVALID_ARGUMENT", "For the axial part give only one of sigma, epsilon, F (with A)."
        )
    if p.get("A") is not None and "F" not in axial:
        raise ToolError("INVALID_ARGUMENT", "params.A is used only with params.F (sigma = F / A).")
    out: dict[str, Any] = {"E": E}
    sigma = eps = None
    if axial == ["F"]:
        sigma = need(p, "F", "force", positive=False) / need(p, "A", "area")
        eps = sigma / E
        formulas += ["sigma = F / A", "epsilon = sigma / E"]
    elif axial == ["sigma"]:
        sigma = need(p, "sigma", "pressure", positive=False)
        eps = sigma / E
        formulas.append("epsilon = sigma / E")
    elif axial == ["epsilon"]:
        eps = need(p, "epsilon", None, positive=False)
        sigma = E * eps
        formulas.append("sigma = E * epsilon")
    if sigma is not None and eps is not None:
        out |= {"sigma": sigma, "epsilon": eps}
        L = qty(p, "L", "length", need=False)
        if L is not None:
            out["delta_L"] = eps * L
            formulas.append("delta_L = epsilon * L")
        if nu is not None:
            out["epsilon_transverse"] = -nu * eps
            formulas.append("epsilon_transverse = -nu * epsilon")
    elif p.get("L") is not None:
        raise ToolError("INVALID_ARGUMENT", "params.L is used with sigma, epsilon or F.")
    if nu is not None:
        out |= {"nu": nu, "G": G}
    shear = [k for k in ("tau", "gamma") if p.get(k) is not None]
    if shear:
        if len(shear) > 1:
            raise ToolError("INVALID_ARGUMENT", "For shear give only one of tau and gamma.")
        if G is None:
            raise ToolError("INVALID_ARGUMENT", "tau or gamma need G, or nu to derive it.")
        if shear == ["tau"]:
            tau = need(p, "tau", "pressure", positive=False)
            out |= {"tau": tau, "gamma": tau / G}
        else:
            gamma = need(p, "gamma", None, positive=False)
            out |= {"tau": G * gamma, "gamma": gamma}
        formulas.append("tau = G * gamma")
    if len(out) == 1:
        raise ToolError(
            "INVALID_ARGUMENT",
            "At least one of sigma, epsilon, F with A, tau, gamma, nu, G is needed.",
        )
    ctx.units = units_for(
        out,
        {
            "E": "Pa",
            "sigma": "Pa",
            "epsilon": "1",
            "delta_L": "m",
            "epsilon_transverse": "1",
            "nu": "1",
            "G": "Pa",
            "tau": "Pa",
            "gamma": "rad",
        },
    )
    ctx.formula = "; ".join(formulas)
    parts = (
        [f"sigma = {mpa(out['sigma'])}, epsilon = {out['epsilon']:.6g}"] if "sigma" in out else []
    )
    if "delta_L" in out:
        parts.append(f"elongation {mm(out['delta_L'])}")
    if "G" in out and out["G"] is not None:
        parts.append(f"G = {mpa(out['G'])}")
    if "tau" in out:
        parts.append(f"tau = {mpa(out['tau'])}")
    ctx.formatted = "; ".join(parts) + "."
    return out


# ------------------------------------------------------------------ beam_calc


def _beam_point(support: str, L: float, F: float, a: float, EI: float) -> dict[str, float]:
    b = L - a
    if support == "cantilever":
        return {
            "R_A": F,
            "M_A": F * a,
            "V_max": F,
            "M_max": F * a,
            "x_M_max": 0.0,
            "f_max": F * a * a * (3 * L - a) / (6 * EI),
            "x_f_max": L,
        }
    if support == "simply_supported":
        s = min(a, b)
        f = F * s * (L * L - s * s) ** 1.5 / (9 * math.sqrt(3) * L * EI)
        xf = math.sqrt((L * L - b * b) / 3) if b <= a else L - math.sqrt((L * L - a * a) / 3)
        RA, RB = F * b / L, F * a / L
        return {
            "R_A": RA,
            "R_B": RB,
            "V_max": max(RA, RB),
            "M_max": F * a * b / L,
            "x_M_max": a,
            "f_max": f,
            "x_f_max": xf,
        }
    RA, RB = F * b * b * (3 * a + b) / L**3, F * a * a * (a + 3 * b) / L**3
    MA, MB, Mc = F * a * b * b / L**2, F * a * a * b / L**2, 2 * F * a * a * b * b / L**3
    Mmax, xM = max((MA, 0.0), (MB, L), (Mc, a), key=lambda t: t[0])
    if a >= b:
        f, xf = 2 * F * a**3 * b * b / (3 * EI * (3 * a + b) ** 2), 2 * a * L / (3 * a + b)
    else:
        f, xf = 2 * F * b**3 * a * a / (3 * EI * (3 * b + a) ** 2), L - 2 * b * L / (3 * b + a)
    return {
        "R_A": RA,
        "R_B": RB,
        "M_A": MA,
        "M_B": MB,
        "V_max": max(RA, RB),
        "M_max": Mmax,
        "x_M_max": xM,
        "f_max": f,
        "x_f_max": xf,
    }


def _beam_uniform(support: str, L: float, q: float, EI: float) -> dict[str, float]:
    if support == "cantilever":
        return {
            "R_A": q * L,
            "M_A": q * L * L / 2,
            "V_max": q * L,
            "M_max": q * L * L / 2,
            "x_M_max": 0.0,
            "f_max": q * L**4 / (8 * EI),
            "x_f_max": L,
        }
    if support == "simply_supported":
        return {
            "R_A": q * L / 2,
            "R_B": q * L / 2,
            "V_max": q * L / 2,
            "M_max": q * L * L / 8,
            "x_M_max": L / 2,
            "f_max": 5 * q * L**4 / (384 * EI),
            "x_f_max": L / 2,
        }
    return {
        "R_A": q * L / 2,
        "R_B": q * L / 2,
        "M_A": q * L * L / 12,
        "M_B": q * L * L / 12,
        "V_max": q * L / 2,
        "M_max": q * L * L / 12,
        "x_M_max": 0.0,
        "M_mid": q * L * L / 24,
        "f_max": q * L**4 / (384 * EI),
        "x_f_max": L / 2,
    }


_BEAM_FORMULAS = {
    (
        "cantilever",
        "point",
    ): "M_A = F a; f_max = F a^2 (3 L - a) / (6 E I) at the free end (F a L: F L^3 / (3 E I))",
    ("cantilever", "uniform"): "M_A = q L^2 / 2; f_max = q L^4 / (8 E I)",
    ("simply_supported", "point"): "R_A = F b / L, R_B = F a / L; M_max = F a b / L; "
    "f_max = F s (L^2 - s^2)^(3/2) / (9 sqrt(3) L E I), s = min(a, b) (a = L/2: F L^3 / (48 E I))",
    ("simply_supported", "uniform"): "R = q L / 2; M_max = q L^2 / 8; f_max = 5 q L^4 / (384 E I)",
    (
        "fixed_fixed",
        "point",
    ): "R_A = F b^2 (3a + b) / L^3; M_A = F a b^2 / L^2, M_B = F a^2 b / L^2, "
    "M_load = 2 F a^2 b^2 / L^3; f_max = 2 F a^3 b^2 / (3 E I (3a + b)^2) with a >= b (a = L/2: F L^3 / (192 E I))",
    (
        "fixed_fixed",
        "uniform",
    ): "M_ends = q L^2 / 12, M_midspan = q L^2 / 24; f_max = q L^4 / (384 E I)",
}


@op("beam_calc", ("support", "load", "L", "F", "q", "a", "E", "I", "W", "c"))
def _beam(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    support = choice(p, "support", ("simply_supported", "cantilever", "fixed_fixed"))
    load = choice(p, "load", ("point", "uniform"))
    L = need(p, "L", "length")
    E = need(p, "E", "pressure")
    Iz = need(p, "I", "area_moment")
    EI = E * Iz
    if load == "point":
        if p.get("q") is not None:
            raise ToolError("INVALID_ARGUMENT", "With load 'point' give F, not q.")
        F = need(p, "F", "force")
        default_a = L if support == "cantilever" else L / 2
        a = qty(p, "a", "length", default=default_a)
        assert a is not None
        if not (0 < a <= L if support == "cantilever" else 0 < a < L):
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.a (distance of the load from the left end) must be inside the beam.",
            )
        out: dict[str, Any] = {"a": a, **_beam_point(support, L, F, a, EI)}
    else:
        if p.get("F") is not None or p.get("a") is not None:
            raise ToolError(
                "INVALID_ARGUMENT", "With load 'uniform' give q (load per metre), not F or a."
            )
        out = _beam_uniform(support, L, need(p, "q", "line_load"), EI)
    if p.get("W") is not None and p.get("c") is not None:
        raise ToolError("INVALID_ARGUMENT", "For sigma_max give W or c, not both.")
    W = qty(p, "W", "volume", need=False)
    c = qty(p, "c", "length", need=False)
    if W is not None:
        out["sigma_max"] = out["M_max"] / W
    elif c is not None:
        out["sigma_max"] = out["M_max"] * c / Iz
    ctx.units = units_for(
        out,
        {
            "a": "m",
            "R_A": "N",
            "R_B": "N",
            "M_A": "Nm",
            "M_B": "Nm",
            "M_mid": "Nm",
            "V_max": "N",
            "M_max": "Nm",
            "x_M_max": "m",
            "f_max": "m",
            "x_f_max": "m",
            "sigma_max": "Pa",
        },
    )
    ctx.formula = _BEAM_FORMULAS[(support, load)] + (
        "; sigma_max = M_max / W"
        if W is not None
        else "; sigma_max = M_max c / I"
        if c is not None
        else ""
    )
    text = f"M_max = {fmt_float(out['M_max'], 3)} Nm, max deflection {mm(out['f_max'])} at x = {fmt_float(out['x_f_max'], 4)} m"
    if "sigma_max" in out:
        text += f", sigma_max = {mpa(out['sigma_max'])}"
    ctx.formatted = (
        text
        + ". Loads downwards, reactions upwards, x from the left end (fixed end of the cantilever)."
    )
    return out


# ------------------------------------------------------------------ shaft_torsion_bending


@op("shaft_torsion_bending", ("d", "d_i", "T", "P", "n", "M", "L", "G", "sigma_allow", "criterion"))
def _shaft(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    criterion = choice(p, "criterion", ("von_mises", "tresca"), "von_mises")
    k_eq = 3.0 if criterion == "von_mises" else 4.0
    formulas: list[str] = []
    out: dict[str, Any] = {}
    if p.get("T") is not None and (p.get("P") is not None or p.get("n") is not None):
        raise ToolError("INVALID_ARGUMENT", "Give T or P with n, not both.")
    if p.get("T") is not None:
        T = abs(need(p, "T", "torque", positive=False))
    elif p.get("P") is not None or p.get("n") is not None:
        T = need(p, "P", "power") / need(p, "n", "angular_velocity")
        out["T"] = T
        formulas.append("T = P / omega")
    else:
        raise ToolError(
            "INVALID_ARGUMENT", "The torque T is needed, or the power P with the speed n."
        )
    M = abs(qty(p, "M", "torque", positive=False, default=0.0) or 0.0)
    sigma_allow = qty(p, "sigma_allow", "pressure", need=False)
    d = qty(p, "d", "length", need=False)
    d_i = qty(p, "d_i", "length", need=False)
    if d is None:
        if sigma_allow is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "The diameter d is needed, or sigma_allow for the minimum diameter.",
            )
        if d_i is not None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "The minimum diameter is computed for a solid shaft: for a hollow shaft give d and d_i.",
            )
        root = 4 * M * M + 3 * T * T if criterion == "von_mises" else 4 * M * M + 4 * T * T
        d = (16 * math.sqrt(root) / (math.pi * sigma_allow)) ** (1 / 3)
        out["d_min"] = d
        formulas.append(
            "d_min = (16 sqrt(4 M^2 + 3 T^2) / (pi sigma_allow))^(1/3)"
            if criterion == "von_mises"
            else "d_min = (32 sqrt(M^2 + T^2) / (pi sigma_allow))^(1/3)"
        )
    di = d_i or 0.0
    if di >= d:
        raise ToolError("INVALID_ARGUMENT", "params.d_i must be smaller than d.")
    J = math.pi * pow4_diff(d, di) / 32  # d^4 - d_i^4, keeps its digits for a thin wall (D80)
    tau = T * (d / 2) / J
    sig = M * (d / 2) / (J / 2)
    seq = math.sqrt(sig * sig + k_eq * tau * tau)
    out |= {"J": J, "I": J / 2, "tau_max": tau, "sigma_b": sig, "sigma_eq": seq}
    formulas += [
        "J = pi (d^4 - d_i^4) / 32",
        "tau = T (d/2) / J (solid: 16 T / (pi d^3))",
        "sigma_b = M (d/2) / I, I = J / 2 (solid: 32 M / (pi d^3))",
        "sigma_eq = sqrt(sigma_b^2 + 3 tau^2)"
        if criterion == "von_mises"
        else "sigma_eq = sqrt(sigma_b^2 + 4 tau^2)",
    ]
    L = qty(p, "L", "length", need=False)
    G = qty(p, "G", "pressure", need=False)
    if (L is None) != (G is None):
        raise ToolError("INVALID_ARGUMENT", "The angle of twist needs L and G together.")
    if L is not None and G is not None:
        out["theta"] = T * L / (G * J)
        out["theta_deg"] = math.degrees(out["theta"])
        formulas.append("theta = T L / (G J)")
    if sigma_allow is not None and "d_min" not in out:
        if seq > 0:
            out["SF"] = sigma_allow / seq
            formulas.append("SF = sigma_allow / sigma_eq")
        else:
            out["SF"] = None
            ctx.warn("Zero equivalent stress: safety factor not defined.")
    ctx.units = units_for(
        out,
        {
            "T": "Nm",
            "d_min": "m",
            "J": "m4",
            "I": "m4",
            "tau_max": "Pa",
            "sigma_b": "Pa",
            "sigma_eq": "Pa",
            "theta": "rad",
            "theta_deg": "deg",
            "SF": "1",
        },
    )
    ctx.formula = "; ".join(formulas)
    text = f"tau = {mpa(tau)}, sigma_b = {mpa(sig)}, sigma_eq ({criterion}) = {mpa(seq)}"
    if "d_min" in out:
        text = f"Minimum diameter {mm(d)}; " + text
    if "theta_deg" in out:
        text += f", twist {fmt_float(out['theta_deg'], 4)} degrees"
    ctx.formatted = text + "."
    return out


# ------------------------------------------------------------------ stress_criterion


@op("stress_criterion", ("sx", "sy", "sz", "txy", "tyz", "tzx", "Sy"))
def _stress(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    sx, sy, sz, txy, tyz, tzx = (
        qty(p, k, "pressure", positive=False, default=0.0) or 0.0
        for k in ("sx", "sy", "sz", "txy", "tyz", "tzx")
    )
    out: dict[str, Any] = {}
    if sz == 0 and tyz == 0 and tzx == 0:
        c, rad = (sx + sy) / 2, math.hypot((sx - sy) / 2, txy)
        principal = sorted([c + rad, c - rad, 0.0], reverse=True)
        out["angle_deg"] = math.degrees(0.5 * math.atan2(2 * txy, sx - sy))
    else:
        tensor = np.array([[sx, txy, tzx], [txy, sy, tyz], [tzx, tyz, sz]])
        principal = sorted((float(v) for v in np.linalg.eigvalsh(tensor)), reverse=True)
    p1, p2, p3 = principal
    vm = math.sqrt(((p1 - p2) ** 2 + (p2 - p3) ** 2 + (p3 - p1) ** 2) / 2)
    out = {
        "principal": principal,
        "tau_max": (p1 - p3) / 2,
        "sigma_vm": vm,
        "sigma_tresca": p1 - p3,
        **out,
    }
    formulas = [
        "sigma_1 >= sigma_2 >= sigma_3 eigenvalues of the stress tensor",
        "sigma_vm = sqrt(((s1 - s2)^2 + (s2 - s3)^2 + (s3 - s1)^2) / 2)",
        "sigma_tresca = s1 - s3, tau_max = (s1 - s3) / 2",
    ]
    Sy = qty(p, "Sy", "pressure", need=False)
    if Sy is not None:
        out["SF_vm"] = Sy / vm if vm > 0 else None
        out["SF_tresca"] = Sy / (p1 - p3) if p1 > p3 else None
        formulas.append("SF = Sy / sigma_eq")
        if vm <= 0:
            ctx.warn("Zero stress state: safety factors not defined.")
    ctx.units = units_for(
        out,
        {
            "principal": "Pa",
            "tau_max": "Pa",
            "sigma_vm": "Pa",
            "sigma_tresca": "Pa",
            "angle_deg": "deg",
            "SF_vm": "1",
            "SF_tresca": "1",
        },
    )
    ctx.formula = "; ".join(formulas)
    text = f"Principal {', '.join(mpa(v) for v in principal)}; Von Mises {mpa(vm)}; Tresca {mpa(p1 - p3)}"
    sf_vm = out.get("SF_vm")
    if isinstance(sf_vm, float):
        text += f"; SF Von Mises {fmt_float(sf_vm, 3)}"
    ctx.formatted = text + "."
    return out


# ------------------------------------------------------------------ euler_buckling


def _k_fixed_pinned() -> float:
    from scipy.optimize import brentq

    x = float(
        cast(Any, brentq)(
            lambda t: math.sin(t) - t * math.cos(t),
            4.0,
            4.6,
            xtol=1e-15,
            rtol=4 * np.finfo(float).eps,
        )
    )
    return math.pi / x


_ENDS = {"pinned_pinned": 1.0, "fixed_free": 2.0, "fixed_fixed": 0.5}


@op("euler_buckling", ("E", "I", "L", "end", "K", "A", "Sy", "F"))
def _euler(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    E = need(p, "E", "pressure")
    Iz = need(p, "I", "area_moment")
    L = need(p, "L", "length")
    if one_of(p, ("end", "K"), "the end conditions") == "K":
        K = need(p, "K", None)
    else:
        end = choice(p, "end", ("pinned_pinned", "fixed_free", "fixed_pinned", "fixed_fixed"))
        K = _k_fixed_pinned() if end == "fixed_pinned" else _ENDS[end]
    L0 = K * L
    Pe = math.pi**2 * E * Iz / L0**2
    out: dict[str, Any] = {"K": K, "L0": L0, "P_cr_euler": Pe, "P_cr": Pe, "method": "euler"}
    formulas = ["L0 = K L", "P_cr = pi^2 E I / L0^2"]
    A = qty(p, "A", "area", need=False)
    Sy = qty(p, "Sy", "pressure", need=False)
    if A is not None:
        r = math.sqrt(Iz / A)
        lam = L0 / r
        out |= {"r": r, "slenderness": lam, "sigma_cr": Pe / A}
        formulas.append("r = sqrt(I / A), lambda = L0 / r")
        if Sy is not None:
            lim = math.sqrt(2 * math.pi**2 * E / Sy)
            out["slenderness_limit"] = lim
            formulas.append("lambda_lim = sqrt(2 pi^2 E / Sy)")
            if lam < lim:
                sc = Sy - (Sy * lam / (2 * math.pi)) ** 2 / E
                out |= {"method": "johnson", "sigma_cr": sc, "P_cr": sc * A}
                formulas.append(
                    "Johnson (lambda < lambda_lim): sigma_cr = Sy - (Sy lambda / (2 pi))^2 / E"
                )
                ctx.warn("Stocky column: the Johnson parabola applies, not Euler.")
    elif Sy is not None:
        ctx.warn("Without params.A the slenderness is not checked: Sy is not used.")
    F = qty(p, "F", "force", need=False)
    if F is not None:
        out["SF"] = out["P_cr"] / F
        formulas.append("SF = P_cr / F")
    ctx.units = units_for(
        out,
        {
            "K": "1",
            "L0": "m",
            "P_cr_euler": "N",
            "P_cr": "N",
            "r": "m",
            "slenderness": "1",
            "sigma_cr": "Pa",
            "slenderness_limit": "1",
            "SF": "1",
        },
    )
    ctx.formula = "; ".join(formulas)
    text = f"Critical load {fmt_float(out['P_cr'] / 1000, 4)} kN ({out['method']}), K = {fmt_float(K, 4)}"
    if "slenderness" in out:
        text += f", slenderness {fmt_float(out['slenderness'], 2)}"
    ctx.formatted = text + "."
    return out


# ------------------------------------------------------------------ pressure_vessel


@op("pressure_vessel", ("p", "p_o", "d_i", "r_i", "t", "d_o", "r_o", "shape", "ends", "Sy"))
def _vessel(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    shape = choice(p, "shape", ("cylinder", "sphere"), "cylinder")
    ends = choice(p, "ends", ("closed", "open"), "closed")
    pi_ = need(p, "p", "pressure", positive=False)
    po = qty(p, "p_o", "pressure", positive=False, default=0.0) or 0.0
    if pi_ < 0 or po < 0 or pi_ == po == 0:
        raise ToolError(
            "INVALID_ARGUMENT", "Pressures p (inner) and p_o (outer) >= 0, at least one > 0."
        )
    inner = one_of(p, ("d_i", "r_i"), "the inner radius")
    ri = need(p, inner, "length") / (2 if inner == "d_i" else 1)
    outer = one_of(p, ("t", "d_o", "r_o"), "the thickness")
    v = need(p, outer, "length")
    ro = ri + v if outer == "t" else v / 2 if outer == "d_o" else v
    if ro <= ri:
        raise ToolError(
            "INVALID_ARGUMENT", "The outer radius must be larger than the inner radius."
        )
    t = ro - ri
    dp, dm = pi_ - po, 2 * ri + t
    out: dict[str, Any] = {"r_i": ri, "r_o": ro, "t": t, "t_over_r_i": t / ri}
    if shape == "cylinder":
        A = (pi_ * ri**2 - po * ro**2) / sq_diff(ro, ri)
        B = (pi_ - po) * ri**2 * ro**2 / sq_diff(ro, ri)
        sa = A if ends == "closed" else 0.0
        out |= {
            "sigma_h_thin": dp * dm / (2 * t),
            "sigma_a_thin": dp * dm / (4 * t) if ends == "closed" else 0.0,
            "sigma_t_inner": A + B / ri**2,
            "sigma_t_outer": A + B / ro**2,
            # sigma_r at the walls is the pressure itself (boundary conditions): exact, while A - B / r^2
            # is a difference of close numbers for a thin wall (D84)
            "sigma_r_inner": -pi_,
            "sigma_r_outer": -po,
            "sigma_a": sa,
        }
        st, sr = out["sigma_t_inner"], out["sigma_r_inner"]
        vm = math.sqrt(((st - sr) ** 2 + (sr - sa) ** 2 + (sa - st) ** 2) / 2)
        formulas = [
            "thin wall (Mariotte, mean diameter d_m = d_i + t): "
            "sigma_h = (p - p_o) d_m / (2 t), sigma_a = (p - p_o) d_m / (4 t)",
            "Lamé: sigma_t(r) = A + B / r^2, sigma_r(r) = A - B / r^2, A = (p r_i^2 - p_o r_o^2) / (r_o^2 - r_i^2), "
            "B = (p - p_o) r_i^2 r_o^2 / (r_o^2 - r_i^2); closed ends: sigma_a = A",
        ]
    else:
        den = cube_diff(ro, ri)  # r_o^3 - r_i^3 without cancellation for a thin wall (D84)
        C = (pi_ * ri**3 - po * ro**3) / den
        D = (pi_ - po) * ri**3 * ro**3 / den
        out |= {
            "sigma_h_thin": dp * dm / (4 * t),
            "sigma_t_inner": C + D / (2 * ri**3),
            "sigma_t_outer": C + D / (2 * ro**3),
            "sigma_r_inner": -pi_,  # boundary conditions, exact (D84)
            "sigma_r_outer": -po,
        }
        vm = abs(out["sigma_t_inner"] - out["sigma_r_inner"])
        formulas = [
            "thin wall (mean diameter d_m = d_i + t): sigma = (p - p_o) d_m / (4 t)",
            "Lamé sphere: sigma_t(r) = C + D / (2 r^3), sigma_r(r) = C - D / r^3, C = (p r_i^3 - p_o r_o^3) / (r_o^3 - r_i^3), "
            "D = (p - p_o) r_i^3 r_o^3 / (r_o^3 - r_i^3)",
        ]
    out["sigma_vm_inner"] = vm
    if t / ri > 0.1:
        ctx.warn(
            f"Thick wall (t / r_i = {t / ri:.3g} > 0.1): the thin-wall formula is not reliable, use the Lamé values."
        )
    Sy = qty(p, "Sy", "pressure", need=False)
    if Sy is not None:
        out["SF"] = Sy / vm if vm > 0 else None
        formulas.append("SF = Sy / sigma_vm at the inner surface")
    ctx.units = {
        k: ("1" if k in {"t_over_r_i", "SF"} else "m" if k in {"r_i", "r_o", "t"} else "Pa")
        for k in out
    }
    ctx.formula = "; ".join(formulas)
    ctx.formatted = (
        f"Inner hoop stress (Lamé) {mpa(out['sigma_t_inner'])}, thin wall {mpa(out['sigma_h_thin'])}; "
        f"inner Von Mises {mpa(vm)}."
    )
    return out
