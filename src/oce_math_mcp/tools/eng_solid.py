"""eng_calc, strength of materials without tables (D46): thermal stresses, Hertz contact, helical
compression springs, interference fits, strain analysis and strain-gauge rosettes.
Formulas in the server, checked in the tests with independent methods; source examples in the tests.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from scipy.optimize import minimize_scalar

from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import choice, mm, mpa, need, one_of, only, op, qty, temperature, units_for

_STRESS = "Pa"


def _poisson(p: dict[str, Any], key: str, *, need_it: bool = True) -> float | None:
    nu = qty(p, key, None, need=need_it, positive=False)
    if nu is not None and not -1 < nu < 0.5:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be between -1 and 0.5.")
    return nu


def _delta_t(p: dict[str, Any]) -> float:
    if p.get("dT") is not None:
        if p.get("T1") is not None or p.get("T2") is not None:
            raise ToolError("INVALID_ARGUMENT", "Give dT or the pair T1, T2.")
        return need(p, "dT", None, positive=False)
    T1, T2 = temperature(p, "T1"), temperature(p, "T2")
    assert T1 is not None and T2 is not None
    return T2 - T1


# ------------------------------------------------------------------ thermal_stress

_TH = {
    "free": {"alpha", "dT", "T1", "T2", "L"},
    "bar": {"alpha", "dT", "T1", "T2", "E", "L", "A", "gap"},
    "plate": {"alpha", "dT", "T1", "T2", "E", "nu"},
}


@op("thermal_stress", ("mode", "alpha", "dT", "T1", "T2", "L", "E", "nu", "A", "gap"))
def _thermal(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_TH))
    only(p, mode, _TH[mode])
    alpha = need(p, "alpha", "thermal_expansion")
    dT = _delta_t(p)
    eps = alpha * dT
    out: dict[str, Any] = {"dT": dT, "epsilon_th": eps}
    L = qty(p, "L", "length", need=False)
    if mode == "free":
        formulas = ["epsilon_th = alpha dT"]
        if L is not None:
            out["dL"] = eps * L
            formulas.append("dL = alpha L dT")
        ctx.formula = "; ".join(formulas)
        ctx.formatted = f"Thermal strain {fmt_float(eps * 1e6, 3)} um/m" + (
            f", elongation {mm(out['dL'])}." if L is not None else "."
        )
    elif mode == "bar":
        E = need(p, "E", "pressure")
        gap = qty(p, "gap", "length", need=False, positive=False)
        if gap is not None:
            if gap < 0:
                raise ToolError("INVALID_ARGUMENT", "params.gap (initial clearance) must be >= 0.")
            if L is None:
                raise ToolError("INVALID_ARGUMENT", "With params.gap params.L is also needed.")
        if L is not None:
            out["delta_free"] = eps * L
        if gap is None:
            sigma = -E * eps
            formulas = ["bar with both ends fixed: sigma = -E alpha dT"]
        else:
            assert L is not None
            closed = eps * L > gap
            out["gap_closed"] = closed
            sigma = -E * (eps * L - gap) / L if closed else 0.0
            formulas = [
                "bar between two rigid walls with total clearance gap: if alpha L dT > gap, "
                "sigma = -E (alpha L dT - gap) / L, otherwise sigma = 0"
            ]
        out["sigma"] = sigma
        A = qty(p, "A", "area", need=False)
        if A is not None:
            out["F"] = sigma * A
            formulas.append("F = sigma A")
        ctx.formula = "; ".join(formulas)
        ctx.formatted = f"Thermal stress {mpa(sigma)} (negative = compression)."
    else:
        E = need(p, "E", "pressure")
        nu = _poisson(p, "nu")
        assert nu is not None
        sigma = -E * eps / (1 - nu)
        out["sigma"] = sigma
        ctx.formula = "plate restrained in both in-plane directions: sigma_x = sigma_y = -E alpha dT / (1 - nu)"
        ctx.formatted = f"Biaxial thermal stress {mpa(sigma)} (negative = compression)."
    ctx.units = units_for(
        out,
        {
            "dT": "K",
            "epsilon_th": "1",
            "dL": "m",
            "delta_free": "m",
            "sigma": _STRESS,
            "F": "N",
            "gap_closed": "1",
        },
    )
    return out


# ------------------------------------------------------------------ contact_hertz


def _tau_sphere(nu: float) -> tuple[float, float]:
    """tau_max / p0 and z / a on the axis (Johnson 1985): sigma_z, sigma_r = sigma_theta."""

    def neg(z: float) -> float:
        sz = -1 / (1 + z * z)
        sr = -((1 + nu) * (1 - z * math.atan(1 / z)) - 1 / (2 * (1 + z * z)))
        return -abs(sr - sz) / 2

    m = cast(Any, minimize_scalar)(
        neg, bounds=(1e-9, 5.0), method="bounded", options={"xatol": 1e-12}
    )
    return -float(m.fun), float(m.x)


def _tau_cylinder(nu: float) -> tuple[float, float]:
    """tau_max / p0 and z / b below the centreline, plane strain (Johnson 1985)."""

    def neg(z: float) -> float:
        s = math.sqrt(1 + z * z)
        sx = -2 * nu * (s - z)
        sy = -((1 + 2 * z * z) / s - 2 * z)
        sz = -1 / s
        return -max(abs(sx - sy), abs(sy - sz), abs(sx - sz)) / 2

    m = cast(Any, minimize_scalar)(
        neg, bounds=(0.0, 5.0), method="bounded", options={"xatol": 1e-12}
    )
    return -float(m.fun), float(m.x)


@op("contact_hertz", ("geometry", "F", "R1", "R2", "L", "E1", "nu1", "E2", "nu2"))
def _hertz(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    geo = choice(p, "geometry", ("sphere", "cylinder"))
    F = need(p, "F", "force")
    R1 = need(p, "R1", "length")
    R2 = qty(p, "R2", "length", need=False, positive=False)
    if R2 is not None and (R2 == 0 or (R2 < 0 and -R2 <= R1)):
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.R2: positive (convex), negative for a concave seat with |R2| > R1, "
            "absent for a plane.",
        )
    E1 = need(p, "E1", "pressure")
    nu1 = _poisson(p, "nu1")
    assert nu1 is not None
    if (p.get("E2") is None) != (p.get("nu2") is None):
        raise ToolError("INVALID_ARGUMENT", "Give E2 and nu2 together, or neither (same material).")
    same = p.get("E2") is None
    E2 = E1 if same else need(p, "E2", "pressure")
    nu2 = nu1 if same else _poisson(p, "nu2")
    assert nu2 is not None
    Es = 1 / ((1 - nu1**2) / E1 + (1 - nu2**2) / E2)
    R = 1 / (1 / R1 + (0.0 if R2 is None else 1 / R2))
    out: dict[str, Any] = {"R": R, "E_star": Es}
    base = [
        "1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2",
        "1/R = 1/R1 + 1/R2 (R2 < 0 for a concave seat, absent for a plane)",
    ]
    if geo == "sphere":
        if p.get("L") is not None:
            raise ToolError("INVALID_ARGUMENT", "params.L is used only with geometry 'cylinder'.")
        a = (3 * F * R / (4 * Es)) ** (1 / 3)
        p0 = 3 * F / (2 * math.pi * a * a)
        out |= {"a": a, "p0": p0, "p_mean": F / (math.pi * a * a), "d": a * a / R}
        taus = [_tau_sphere(nu1), _tau_sphere(nu2)]
        base += [
            "a = (3 F R / (4 E*))^(1/3)",
            "p0 = 3 F / (2 pi a^2)",
            "d = a^2 / R",
            "tau_max: maximum of (sigma_r - sigma_z) / 2 on the axis, sigma_z = -p0 / (1 + z^2/a^2), "
            "sigma_r = -p0 [(1 + nu)(1 - (z/a) atan(a/z)) - 1 / (2 (1 + z^2/a^2))]",
        ]
        size = a
    else:
        L = need(p, "L", "length")
        b = math.sqrt(4 * F * R / (math.pi * L * Es))
        p0 = 2 * F / (math.pi * b * L)
        out |= {"b": b, "p0": p0, "p_mean": F / (2 * b * L)}
        taus = [_tau_cylinder(nu1), _tau_cylinder(nu2)]
        base += [
            "b = sqrt(4 F R / (pi L E*))",
            "p0 = 2 F / (pi b L)",
            "tau_max: maximum of the principal shear stress below the centreline (plane strain), "
            "sigma_x = -2 nu p0 (sqrt(1 + z^2/b^2) - z/b), sigma_y = -p0 ((1 + 2 z^2/b^2) / sqrt(1 + z^2/b^2) - 2 z/b), "
            "sigma_z = -p0 / sqrt(1 + z^2/b^2)",
        ]
        size = b
    for i, (t, z) in enumerate(taus, 1):
        out[f"tau_max_{i}"] = t * p0
        out[f"z_tau_max_{i}"] = z * size
    ctx.formula = "; ".join(base)
    half = "contact radius a" if geo == "sphere" else "half-width b"
    ctx.formatted = (
        f"Maximum pressure {mpa(p0)}, {half} {mm(size)}, maximum tau {mpa(out['tau_max_1'])} "
        f"at {mm(out['z_tau_max_1'])} depth."
    )
    ctx.units = {
        k: ("Pa" if k in {"E_star", "p0", "p_mean"} or k.startswith("tau") else "m") for k in out
    }
    return out


# ------------------------------------------------------------------ spring_helical

_ENDS = {  # coils beyond the active ones, coils in the solid length
    "plain": (0, 1),
    "plain_ground": (1, 0),
    "squared": (2, 1),
    "squared_ground": (2, 0),
}


@op("spring_helical", ("d", "D", "D_o", "Na", "G", "F", "deflection", "ends", "L0", "rho"))
def _spring(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    d = need(p, "d", "length")
    key = one_of(p, ("D", "D_o"), "the mean diameter D or the outer diameter D_o")
    D = need(p, key, "length") - (d if key == "D_o" else 0.0)
    if d >= D:
        raise ToolError(
            "INVALID_ARGUMENT", "The mean diameter must be greater than the wire diameter."
        )
    Na = need(p, "Na", None)
    G = need(p, "G", "pressure")
    C = D / d
    k = G * d**4 / (8 * D**3 * Na)
    KW = (4 * C - 1) / (4 * C - 4) + 0.615 / C
    KB = (4 * C + 2) / (4 * C - 3)
    out: dict[str, Any] = {"D": D, "C": C, "k": k, "K_W": KW, "K_B": KB}
    formulas = [
        "C = D / d",
        "k = G d^4 / (8 D^3 Na)",
        "K_W = (4C - 1)/(4C - 4) + 0.615/C (Wahl)",
        "K_B = (4C + 2)/(4C - 3) (Bergsträsser)",
    ]
    if not 4 <= C <= 12:
        ctx.warn(f"Spring index C = {C:.3g} outside the recommended range 4-12.")
    tk = 8 * D / (math.pi * d**3)
    if p.get("F") is not None and p.get("deflection") is not None:
        raise ToolError("INVALID_ARGUMENT", "Give F or deflection, not both.")
    if p.get("F") is not None or p.get("deflection") is not None:
        if p.get("F") is not None:
            F = need(p, "F", "force")
            y = F / k
        else:
            y = need(p, "deflection", "length")
            F = k * y
        out |= {"F": F, "deflection": y, "tau": KB * F * tk, "tau_wahl": KW * F * tk}
        formulas += ["deflection = F / k", "tau = K_B 8 F D / (pi d^3); tau_wahl with K_W"]
    if p.get("ends") is not None:
        ends = choice(p, "ends", tuple(_ENDS))
        extra, solid = _ENDS[ends]
        Nt = Na + extra
        Ls = d * (Nt + solid)
        out |= {"Nt": Nt, "L_solid": Ls}
        formulas.append(
            "Nt and solid length Ls: plain Nt = Na, Ls = d (Nt + 1); plain_ground Nt = Na + 1, Ls = d Nt; "
            "squared Nt = Na + 2, Ls = d (Nt + 1); squared_ground Nt = Na + 2, Ls = d Nt"
        )
        L0 = qty(p, "L0", "length", need=False)
        if L0 is not None:
            if Ls >= L0:
                raise ToolError("INVALID_ARGUMENT", "params.L0 must exceed the solid length.")
            Fs = k * (L0 - Ls)
            out |= {"F_solid": Fs, "tau_solid": KB * Fs * tk}
            formulas.append("F_solid = k (L0 - Ls)")
    elif p.get("L0") is not None:
        raise ToolError("INVALID_ARGUMENT", "With params.L0 params.ends is needed.")
    rho = qty(p, "rho", "density", need=False)
    if rho is not None:
        m = rho * math.pi * d * d / 4 * math.pi * D * Na
        out |= {"m_active": m, "f_n": 0.5 * math.sqrt(k / m)}
        formulas.append(
            "f_n = (1/2) sqrt(k / m_active), ends on parallel plates, m_active = rho (pi d^2 / 4) pi D Na"
        )
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"Stiffness {fmt_float(k / 1e3, 4)} N/mm, index C = {fmt_float(C, 3)}" + (
        f", tau {mpa(out['tau'])}." if "tau" in out else "."
    )
    ctx.units = units_for(
        out,
        {
            "D": "m",
            "C": "1",
            "k": "N/m",
            "K_W": "1",
            "K_B": "1",
            "F": "N",
            "deflection": "m",
            "tau": "Pa",
            "tau_wahl": "Pa",
            "Nt": "1",
            "L_solid": "m",
            "F_solid": "N",
            "tau_solid": "Pa",
            "m_active": "kg",
            "f_n": "Hz",
        },
    )
    return out


# ------------------------------------------------------------------ interference_fit


@op(
    "interference_fit",
    (
        "d",
        "D_o",
        "d_i",
        "delta",
        "E_hub",
        "nu_hub",
        "E_shaft",
        "nu_shaft",
        "mu",
        "L",
        "alpha_hub",
        "clearance",
    ),
)
def _fit(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    d = need(p, "d", "length")
    Do = need(p, "D_o", "length")
    di = qty(p, "d_i", "length", need=False, positive=False) or 0.0
    if not 0 <= di < d < Do:
        raise ToolError("INVALID_ARGUMENT", "Serve 0 <= d_i < d < D_o.")
    delta = need(p, "delta", "length")
    Eh = need(p, "E_hub", "pressure")
    nh = _poisson(p, "nu_hub")
    assert nh is not None
    if (p.get("E_shaft") is None) != (p.get("nu_shaft") is None):
        raise ToolError(
            "INVALID_ARGUMENT",
            "Give E_shaft and nu_shaft together, or neither (same material as the hub).",
        )
    Es = Eh if p.get("E_shaft") is None else need(p, "E_shaft", "pressure")
    ns = nh if p.get("nu_shaft") is None else _poisson(p, "nu_shaft")
    assert ns is not None
    kh = (Do**2 + d**2) / (Do**2 - d**2)
    ks = (d**2 + di**2) / (d**2 - di**2)
    pc = delta / (d * ((kh + nh) / Eh + (ks - ns) / Es))
    st_h = pc * kh
    out: dict[str, Any] = {
        "p": pc,
        "sigma_t_hub": st_h,
        "sigma_r_hub": -pc,
        "sigma_vm_hub": math.sqrt(st_h**2 + st_h * pc + pc**2),
        "sigma_t_shaft": -pc * ks,
        "sigma_r_shaft": -pc,
    }
    formulas = [
        "p = delta / (d [ (1/E_hub) ((D_o^2 + d^2)/(D_o^2 - d^2) + nu_hub) "
        "+ (1/E_shaft) ((d^2 + d_i^2)/(d^2 - d_i^2) - nu_shaft) ]), delta diametral interference",
        "sigma_t_hub = p (D_o^2 + d^2)/(D_o^2 - d^2), sigma_r = -p at the contact",
        "sigma_t_shaft = -p (d^2 + d_i^2)/(d^2 - d_i^2) at the contact",
        "sigma_vm_hub = sqrt(sigma_t^2 - sigma_t sigma_r + sigma_r^2) at the hub bore",
    ]
    if di > 0:
        out["sigma_t_shaft_inner"] = -2 * pc * d**2 / (d**2 - di**2)
        formulas.append("sigma_t at the bore of the hollow shaft = -2 p d^2 / (d^2 - d_i^2)")
    if (p.get("mu") is None) != (p.get("L") is None):
        raise ToolError("INVALID_ARGUMENT", "Torque and axial force need mu and L together.")
    if p.get("mu") is not None:
        mu, L = need(p, "mu", None), need(p, "L", "length")
        Fa = mu * pc * math.pi * d * L
        out |= {"F_axial": Fa, "T": Fa * d / 2}
        formulas.append("F_axial = mu p pi d L; T = F_axial d / 2")
    if p.get("alpha_hub") is not None:
        a = need(p, "alpha_hub", "thermal_expansion")
        c = qty(p, "clearance", "length", need=False, positive=False) or 0.0
        if c < 0:
            raise ToolError(
                "INVALID_ARGUMENT", "params.clearance (assembly clearance) must be >= 0."
            )
        out["dT_assembly"] = (delta + c) / (a * d)
        formulas.append("shrink fit: dT = (delta + clearance) / (alpha_hub d)")
    elif p.get("clearance") is not None:
        raise ToolError("INVALID_ARGUMENT", "params.clearance is used with params.alpha_hub.")
    ctx.formula = "; ".join(formulas)
    ctx.formatted = f"Contact pressure {mpa(pc)}, hoop stress in the hub {mpa(st_h)}." + (
        f" Transmissible torque {fmt_float(out['T'], 3)} Nm." if "T" in out else ""
    )
    ctx.units = {
        k: ("N" if k == "F_axial" else "Nm" if k == "T" else "K" if k == "dT_assembly" else "Pa")
        for k in out
    }
    return out


# ------------------------------------------------------------------ strain_analysis

_SA = {
    "transform": {"ex", "ey", "gxy", "theta_deg", "E", "nu"},
    "rosette": {"rosette", "angles_deg", "ea", "eb", "ec", "E", "nu"},
}
_ROSETTE = {"rectangular": (0.0, 45.0, 90.0), "delta": (0.0, 60.0, 120.0)}


def _strain(p: dict[str, Any], key: str) -> float:
    return need(p, key, None, positive=False)


@op(
    "strain_analysis",
    ("mode", "ex", "ey", "gxy", "theta_deg", "rosette", "angles_deg", "ea", "eb", "ec", "E", "nu"),
)
def _strain_analysis(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_SA))
    only(p, mode, _SA[mode])
    out: dict[str, Any] = {}
    formulas: list[str] = []
    if mode == "rosette":
        if p.get("rosette") is not None and p.get("angles_deg") is not None:
            raise ToolError("INVALID_ARGUMENT", "Give rosette or angles_deg, not both.")
        if p.get("angles_deg") is not None:
            raw = p["angles_deg"]
            if (
                not isinstance(raw, list)
                or len(raw) != 3
                or any(
                    isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x)
                    for x in raw
                )
            ):
                raise ToolError(
                    "INVALID_ARGUMENT", "params.angles_deg: list of 3 angles in degrees."
                )
            angles = tuple(float(x) for x in raw)
        else:
            angles = _ROSETTE[choice(p, "rosette", tuple(_ROSETTE))]
        th = np.radians(angles)
        M = np.column_stack([np.cos(th) ** 2, np.sin(th) ** 2, np.sin(th) * np.cos(th)])
        if np.linalg.cond(M) > 1e10:
            raise ToolError(
                "INVALID_ARGUMENT",
                "The rosette angles must differ from each other (modulo 180 degrees): singular system.",
            )
        reads = [_strain(p, k) for k in ("ea", "eb", "ec")]
        ex, ey, gxy = (float(v) for v in np.linalg.solve(M, reads))
        out |= {"ex": ex, "ey": ey, "gxy": gxy}
        formulas.append(
            "reading at an angle t: e(t) = ex cos^2 t + ey sin^2 t + gxy sin t cos t; three readings = linear system "
            f"(angles {', '.join(fmt_float(a, 3) for a in angles)} degrees)"
        )
    else:
        ex, ey, gxy = _strain(p, "ex"), _strain(p, "ey"), _strain(p, "gxy")
    avg = (ex + ey) / 2
    rad = math.hypot((ex - ey) / 2, gxy / 2)
    e1, e2 = avg + rad, avg - rad
    out |= {
        "e1": e1,
        "e2": e2,
        "gamma_max": e1 - e2,
        "theta_p_deg": math.degrees(math.atan2(gxy, ex - ey) / 2),
    }
    formulas += [
        "e1,2 = (ex + ey)/2 +- sqrt(((ex - ey)/2)^2 + (gxy/2)^2), gxy engineering shear strain",
        "theta_p = (1/2) atan2(gxy, ex - ey), from x towards the direction of e1",
        "gamma_max = e1 - e2 (in the plane)",
    ]
    if p.get("theta_deg") is not None:
        t = math.radians(need(p, "theta_deg", None, positive=False))
        c2, s2 = math.cos(2 * t), math.sin(2 * t)
        out |= {
            "ex_theta": avg + (ex - ey) / 2 * c2 + gxy / 2 * s2,
            "ey_theta": avg - (ex - ey) / 2 * c2 - gxy / 2 * s2,
            "gxy_theta": -(ex - ey) * s2 + gxy * c2,
        }
        formulas.append(
            "ex' = (ex + ey)/2 + (ex - ey)/2 cos 2t + gxy/2 sin 2t; gx'y' = -(ex - ey) sin 2t + gxy cos 2t"
        )
    if (p.get("E") is None) != (p.get("nu") is None):
        raise ToolError("INVALID_ARGUMENT", "The stresses need E and nu together.")
    if p.get("E") is not None:
        E = need(p, "E", "pressure")
        nu = _poisson(p, "nu")
        assert nu is not None
        c = E / (1 - nu * nu)
        s1, s2_ = c * (e1 + nu * e2), c * (e2 + nu * e1)
        out |= {
            "sigma1": s1,
            "sigma2": s2_,
            "tau_max": (s1 - s2_) / 2,
            "e3": -nu / (1 - nu) * (e1 + e2),
        }
        formulas.append(
            "plane stress: sigma1 = E/(1 - nu^2) (e1 + nu e2), sigma2 = E/(1 - nu^2) (e2 + nu e1); "
            "e3 = -nu/(1 - nu) (e1 + e2)"
        )
    ctx.formula = "; ".join(formulas)
    ctx.formatted = (
        f"Principal strains {fmt_float(e1 * 1e6, 2)} and {fmt_float(e2 * 1e6, 2)} um/m"
        + (
            f"; principal stresses {mpa(out['sigma1'])} and {mpa(out['sigma2'])}."
            if "sigma1" in out
            else "."
        )
    )
    ctx.units = {
        k: ("deg" if k.endswith("_deg") else "Pa" if k.startswith(("sigma", "tau")) else "1")
        for k in out
    }
    return out
