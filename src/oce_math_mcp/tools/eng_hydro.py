"""eng_calc area E (D43): hydrostatics, pumps and cylinders, open channels (Manning), flow meters,
compressible flow. Formulas in the server, checked in the tests with independent methods.

Constants: g = 9.80665 m/s2 (standard gravity); dry air gamma = 1.4 and R = 287.05287 J/(kg K)
(ISO 2533), the same as in area C.
D116: for M >= 1 the isentropic and area-Mach modes give the half-angle of the Mach cone, mu = asin(1/M)
(page "Mach wave").
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import choice, need, one_of, only, op, qty, temperature, units_for
from .eng_thermo import G0, GAMMA_AIR, R_AIR

_EPS = 2.220446049250313e-16


def _brentq(f: Callable[[float], float], a: float, b: float) -> float:
    from scipy.optimize import brentq

    return float(cast(Any, brentq)(f, a, b, xtol=1e-15, rtol=4 * _EPS, maxiter=500))


def _grav(p: dict[str, Any]) -> float:
    g = qty(p, "g", "acceleration", default=G0)
    assert g is not None
    return g


def _eta(p: dict[str, Any]) -> float:
    eta = qty(p, "eta", None, default=1.0)
    assert eta is not None
    if eta > 1:
        raise ToolError("INVALID_ARGUMENT", "params.eta (efficiency) must be between 0 and 1.")
    return eta


# ------------------------------------------------------------------ hydrostatics

_HYD = {
    "pressure": {"rho", "h", "p0", "g"},
    "buoyancy": {"rho", "V", "m", "g"},
    "plane_wall": {"rho", "shape", "b", "L", "D", "theta_deg", "y_top", "g"},
}


@op(
    "hydrostatics",
    ("mode", "rho", "h", "p0", "g", "V", "m", "shape", "b", "L", "D", "theta_deg", "y_top"),
)
def _hydrostatics(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_HYD))
    only(p, mode, _HYD[mode])
    rho, g = need(p, "rho", "density"), _grav(p)
    out: dict[str, Any]
    if mode == "pressure":
        h = need(p, "h", "length", positive=False)
        if h < 0:
            raise ToolError("INVALID_ARGUMENT", "params.h (depth) must be >= 0.")
        p0 = qty(p, "p0", "pressure", positive=False, default=0.0) or 0.0
        ph = rho * g * h
        out = {"p_hydro": ph, "p": p0 + ph}
        ctx.formula = "p = p0 + rho g h (Stevin's law)"
        ctx.formatted = f"p = {fmt_float(out['p'] / 1e5, 5)} bar, of which {fmt_float(ph / 1e3, 4)} kPa of column."
    elif mode == "buoyancy":
        V = need(p, "V", "volume")
        Fb = rho * g * V
        out = {"F_b": Fb}
        ctx.formula = "F_b = rho g V (Archimedes' principle, body fully immersed)"
        m = qty(p, "m", "mass", need=False)
        text = f"Buoyancy of the immersed body {fmt_float(Fb, 4)} N"
        if m is not None:
            W, rb = m * g, m / V
            floats = rb < rho
            out |= {
                "W": W,
                "F_net": Fb - W,
                "rho_body": rb,
                "floats": floats,
                "submerged_fraction": min(1.0, rb / rho),
            }
            ctx.formula += "; floats if rho_body < rho; immersed fraction = rho_body / rho"
            text += (
                f"; floats with {fmt_float(rb / rho * 100, 3)}% of the volume immersed"
                if floats
                else "; sinks"
            )
        ctx.formatted = text + "."
    else:
        shape = choice(p, "shape", ("rectangle", "circle"))
        wrong = ("D",) if shape == "rectangle" else ("b", "L")
        if any(k in p for k in wrong):
            raise ToolError(
                "INVALID_ARGUMENT",
                "plane_wall: the rectangle needs b and L, the circle needs D.",
            )
        if shape == "rectangle":
            b, L = need(p, "b", "length"), need(p, "L", "length")
            A, Ixc, length = b * L, b * L**3 / 12, L
        else:
            D = need(p, "D", "length")
            A, Ixc, length = math.pi * D * D / 4, math.pi * D**4 / 64, D
        theta = need(p, "theta_deg", None)
        if theta >= 180:
            raise ToolError(
                "INVALID_ARGUMENT", "params.theta_deg must be between 0 and 180 degrees."
            )
        y_top = qty(p, "y_top", "length", positive=False, default=0.0) or 0.0
        if y_top < 0:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.y_top < 0: the top edge is above the free surface; use only the wetted part.",
            )
        s = math.sin(math.radians(theta))
        yc = y_top + length / 2
        hc = yc * s
        F = rho * g * hc * A
        off = Ixc / (yc * A)
        out = {
            "A": A,
            "I_xc": Ixc,
            "y_c": yc,
            "h_c": hc,
            "F": F,
            "y_cp": yc + off,
            "h_cp": (yc + off) * s,
            "cp_offset": off,
        }
        ctx.formula = (
            "F = rho g h_c A; y_cp = y_c + I_xc / (y_c A); h = y sin(theta) "
            "(y along the wall from the free surface)"
        )
        ctx.formatted = (
            f"Force {fmt_float(F / 1e3, 4)} kN, centre of pressure at {fmt_float(yc + off, 4)} m "
            "from the free surface along the wall."
        )
    ctx.units = units_for(
        out,
        {
            "p_hydro": "Pa",
            "p": "Pa",
            "F_b": "N",
            "W": "N",
            "F_net": "N",
            "rho_body": "kg/m3",
            "submerged_fraction": "1",
            "A": "m2",
            "I_xc": "m4",
            "y_c": "m",
            "h_c": "m",
            "F": "N",
            "y_cp": "m",
            "h_cp": "m",
            "cp_offset": "m",
        },
    )
    return out


# ------------------------------------------------------------------ pump_system

_PUMP = {
    "power": {"Q", "rho", "H", "dp", "eta", "n", "g"},
    "affinity": {"n1", "n2", "Q1", "H1", "P1", "D1", "D2", "rho1", "rho2"},
    "npsh": {"p_s", "p_v", "rho", "z_s", "h_loss", "NPSH_r", "g"},
    "cylinder": {"D", "d", "p", "Q", "eta"},
}


def _pair(p: dict[str, Any], a: str, b: str, kind: str) -> float:
    """Ratio b/a of two optional keys that must be given together; 1 if both are missing."""
    va, vb = qty(p, a, kind, need=False), qty(p, b, kind, need=False)
    if (va is None) != (vb is None):
        raise ToolError("INVALID_ARGUMENT", f"params.{a} and params.{b} must be given together.")
    return 1.0 if va is None or vb is None else vb / va


@op(
    "pump_system",
    (
        "mode",
        "Q",
        "rho",
        "H",
        "dp",
        "eta",
        "n",
        "g",
        "n1",
        "n2",
        "Q1",
        "H1",
        "P1",
        "D1",
        "D2",
        "rho1",
        "rho2",
        "p_s",
        "p_v",
        "z_s",
        "h_loss",
        "NPSH_r",
        "D",
        "d",
        "p",
    ),
)
def _pump(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_PUMP))
    only(p, mode, _PUMP[mode])
    out: dict[str, Any]
    if mode == "power":
        Q, rho, g, eta = need(p, "Q", "flow_rate"), need(p, "rho", "density"), _grav(p), _eta(p)
        if one_of(p, ("H", "dp"), "the head (H) or the pressure difference (dp)") == "H":
            H = need(p, "H", "length")
            dp = rho * g * H
        else:
            dp = need(p, "dp", "pressure")
            H = dp / (rho * g)
        Ph = dp * Q
        out = {"H": H, "dp": dp, "P_hydraulic": Ph, "P_shaft": Ph / eta}
        ctx.formula = "P_hydraulic = rho g Q H = dp Q; P_shaft = P_hydraulic / eta"
        n = qty(p, "n", "angular_velocity", need=False)
        if n is not None:
            out["T_shaft"] = Ph / eta / n
            ctx.formula += "; T = P_shaft / omega"
        ctx.formatted = f"Hydraulic power {fmt_float(Ph / 1e3, 4)} kW, shaft power {fmt_float(Ph / eta / 1e3, 4)} kW."
    elif mode == "affinity":
        n1, n2 = need(p, "n1", "angular_velocity"), need(p, "n2", "angular_velocity")
        sr, dr = n2 / n1, _pair(p, "D1", "D2", "length")
        rr = _pair(p, "rho1", "rho2", "density")
        if not any(p.get(k) is not None for k in ("Q1", "H1", "P1")):
            raise ToolError("INVALID_ARGUMENT", "affinity: at least one of Q1, H1, P1 is needed.")
        out = {}
        for k1, k2, kind, factor in (
            ("Q1", "Q2", "flow_rate", sr * dr**3),
            ("H1", "H2", "length", sr**2 * dr**2),
            ("P1", "P2", "power", rr * sr**3 * dr**5),
        ):
            v = qty(p, k1, kind, need=False)
            if v is not None:
                out[k2] = v * factor
        out |= {"speed_ratio": sr, "diameter_ratio": dr}
        ctx.formula = (
            "similar pumps: Q2/Q1 = (n2/n1)(D2/D1)^3; H2/H1 = (n2/n1)^2 (D2/D1)^2; "
            "P2/P1 = (rho2/rho1)(n2/n1)^3 (D2/D1)^5"
        )
        if dr != 1:
            ctx.warn(
                "D1 different from D2: similarity laws for geometrically similar pumps; for "
                "impeller trimming of the same pump they hold only approximately."
            )
        ctx.formatted = "Scaled operating point: " + ", ".join(
            f"{k} = {fmt_float(v, 5)}" for k, v in out.items() if k in ("Q2", "H2", "P2")
        )
    elif mode == "npsh":
        ps, pv = need(p, "p_s", "pressure"), need(p, "p_v", "pressure")
        rho, g = need(p, "rho", "density"), _grav(p)
        zs = need(p, "z_s", "length", positive=False)
        hl = qty(p, "h_loss", "length", positive=False, default=0.0) or 0.0
        if hl < 0:
            raise ToolError("INVALID_ARGUMENT", "params.h_loss (losses) must be >= 0.")
        npsh = (ps - pv) / (rho * g) + zs - hl
        out = {"NPSH_a": npsh}
        ctx.formula = "NPSH_a = (p_s - p_v) / (rho g) + z_s - h_losses"
        req = qty(p, "NPSH_r", "length", need=False)
        text = f"Available NPSH {fmt_float(npsh, 4)} m"
        if req is not None:
            out["margin"] = npsh - req
            text += f", margin {fmt_float(npsh - req, 4)} m"
            if npsh < req:
                ctx.warn("Available NPSH below the required NPSH: the pump cavitates.")
        ctx.formatted = text + "."
    else:
        D, pr, eta = need(p, "D", "length"), need(p, "p", "pressure"), _eta(p)
        Ab = math.pi * D * D / 4
        out = {"A_bore": Ab, "F_push": pr * Ab * eta}
        ctx.formula = "F_push = p A eta; A = pi D^2 / 4"
        Q = qty(p, "Q", "flow_rate", need=False)
        if Q is not None:
            out["v_out"] = Q / Ab
        d = qty(p, "d", "length", need=False)
        if d is not None:
            if d >= D:
                raise ToolError("INVALID_ARGUMENT", "params.d (rod) must be smaller than D.")
            Aa = Ab - math.pi * d * d / 4
            out |= {"A_annulus": Aa, "F_pull": pr * Aa * eta, "area_ratio": Ab / Aa}
            ctx.formula += "; F_pull = p (A - pi d^2 / 4) eta"
            if Q is not None:
                out["v_in"] = Q / Aa
        ctx.formatted = f"Push force {fmt_float(pr * Ab * eta / 1e3, 4)} kN."
        if Q is not None:
            ctx.formula += "; v = Q / A"
    ctx.units = units_for(
        out,
        {
            "H": "m",
            "dp": "Pa",
            "P_hydraulic": "W",
            "P_shaft": "W",
            "T_shaft": "Nm",
            "Q2": "m3/s",
            "H2": "m",
            "P2": "W",
            "speed_ratio": "1",
            "diameter_ratio": "1",
            "NPSH_a": "m",
            "margin": "m",
            "A_bore": "m2",
            "A_annulus": "m2",
            "F_push": "N",
            "F_pull": "N",
            "v_out": "m/s",
            "v_in": "m/s",
            "area_ratio": "1",
        },
    )
    return out


# ------------------------------------------------------------------ open_channel


def _channel_geometry(
    shape: str, p: dict[str, Any]
) -> tuple[Callable[[float], tuple[float, float, float]], float | None]:
    """Function y -> (A, wetted P, top width T) and maximum depth (pipe)."""
    allowed = {"rectangular": {"b"}, "trapezoidal": {"b", "z"}, "circular": {"D"}}[shape]
    wrong = sorted({"b", "z", "D"} - allowed & set(p))
    if wrong:
        raise ToolError("INVALID_ARGUMENT", f"shape '{shape}' does not use: {', '.join(wrong)}.")
    if shape == "circular":
        D = need(p, "D", "length")

        def circ(y: float) -> tuple[float, float, float]:
            th = 2 * math.acos(max(-1.0, min(1.0, 1 - 2 * y / D)))
            return D * D / 8 * (th - math.sin(th)), D * th / 2, D * math.sin(th / 2)

        return circ, D
    b = need(p, "b", "length")
    z = 0.0
    if shape == "trapezoidal":
        z = need(p, "z", None, positive=False)
        if z < 0:
            raise ToolError("INVALID_ARGUMENT", "params.z (side slope) must be >= 0.")
    k = 2 * math.sqrt(1 + z * z)
    return (lambda y: ((b + z * y) * y, b + k * y, b + 2 * z * y)), None


def _expand(f: Callable[[float], float], start: float) -> float:
    """First value start * 2^k with f > 0 (f increasing)."""
    hi = start
    for _ in range(200):
        if f(hi) > 0:
            return hi
        hi *= 2
    raise ToolError("MATH_ERROR", "No depth found: check the data.")


@op("open_channel", ("shape", "b", "z", "D", "n", "S", "y", "Q", "g"))
def _channel(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    from scipy.optimize import minimize_scalar

    shape = choice(p, "shape", ("rectangular", "trapezoidal", "circular"))
    geo, D = _channel_geometry(shape, p)
    n, S, g = need(p, "n", None), need(p, "S", None), _grav(p)

    def flow(y: float) -> float:
        A, P, _ = geo(y)
        return A * (A / P) ** (2 / 3) * math.sqrt(S) / n

    out: dict[str, Any] = {}
    y_cap = D if D is not None else math.inf
    if D is not None:
        best = cast(Any, minimize_scalar)(
            lambda y: -flow(y), bounds=(0.5 * D, D), method="bounded", options={"xatol": 1e-13}
        )
        y_qmax, q_max = float(best.x), -float(best.fun)
        out |= {"Q_max": q_max, "y_Q_max": y_qmax}
    if one_of(p, ("y", "Q"), "the depth (y) or the flow rate (Q)") == "y":
        y = need(p, "y", "length")
        if y > y_cap:
            raise ToolError("INVALID_ARGUMENT", "params.y exceeds the pipe diameter.")
        Q = flow(y)
    else:
        Q = need(p, "Q", "flow_rate")
        if D is not None:
            if out["Q_max"] * (1 + 1e-12) < Q:
                raise ToolError(
                    "MATH_ERROR",
                    f"Q above the maximum free-surface flow rate of the pipe ({fmt_float(out['Q_max'], 5)} m3/s).",
                )
            y = _brentq(lambda h: flow(h) - Q, 1e-12 * D, out["y_Q_max"])
        else:
            y = _brentq(lambda h: flow(h) - Q, 0.0 + 1e-300, _expand(lambda h: flow(h) - Q, 1.0))
    A, P, T = geo(y)
    v = Q / A
    out |= {"y": y, "A": A, "P_wet": P, "R_h": A / P, "T_top": T, "v": v, "Q": Q}
    if D is None or y < D * (1 - 1e-12):
        Fr = v / math.sqrt(g * A / T)
        out["Fr"] = Fr
        out["regime"] = "subcritical" if Fr < 1 else "supercritical" if Fr > 1 else "critical"
    else:
        out["Fr"], out["regime"] = None, None
        ctx.warn("Full pipe: Froude number not defined.")

    def crit(h: float) -> float:  # decreasing: Q^2 T / (g A^3) - 1
        Ah, _, Th = geo(h)
        return Q * Q * Th / (g * Ah**3) - 1

    if D is not None:
        out["y_critical"] = _brentq(crit, 1e-9 * D, D * (1 - 1e-12))
    else:
        lo = y
        for _ in range(2000):  # crit grows towards the bottom: halve while it is positive
            if crit(lo) > 0:
                break
            lo /= 2
        out["y_critical"] = _brentq(crit, lo, _expand(lambda h: -crit(h), y))
    ctx.formula = (
        "Manning (SI): v = R_h^(2/3) S^(1/2) / n, Q = v A, R_h = A / P; "
        "Fr = v / sqrt(g A / T); critical depth: Q^2 T / (g A^3) = 1"
    )
    ctx.units = units_for(
        out,
        {
            "y": "m",
            "A": "m2",
            "P_wet": "m",
            "R_h": "m",
            "T_top": "m",
            "v": "m/s",
            "Q": "m3/s",
            "Fr": "1",
            "y_critical": "m",
            "Q_max": "m3/s",
            "y_Q_max": "m",
        },
    )
    ctx.formatted = (
        f"Depth {fmt_float(y, 4)} m, Q = {fmt_float(Q, 5)} m3/s, v = {fmt_float(v, 4)} m/s"
        + (f", Fr = {fmt_float(out['Fr'], 3)}" if out["Fr"] is not None else "")
        + "."
    )
    return out


# ------------------------------------------------------------------ flow_meter

_METER = {
    "orifice": {"D", "d", "Cd", "rho", "dp", "Q", "eps"},
    "nozzle": {"D", "d", "Cd", "rho", "dp", "Q", "eps"},
    "venturi": {"D", "d", "Cd", "rho", "dp", "Q", "eps"},
    "weir_rect": {"b", "Cd", "h", "Q", "g"},
    "weir_v": {"theta_deg", "Cd", "h", "Q", "g"},
}


@op("flow_meter", ("type", "D", "d", "Cd", "rho", "dp", "Q", "eps", "b", "h", "g", "theta_deg"))
def _meter(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    typ = choice(p, "type", tuple(_METER))
    only(p, typ, _METER[typ], field="type")
    Cd = need(p, "Cd", None)
    out: dict[str, Any]
    if typ in ("orifice", "nozzle", "venturi"):
        D, d, rho = need(p, "D", "length"), need(p, "d", "length"), need(p, "rho", "density")
        if d >= D:
            raise ToolError(
                "INVALID_ARGUMENT", "params.d (orifice or throat) must be smaller than D."
            )
        eps = qty(p, "eps", None, default=1.0)
        assert eps is not None
        if eps > 1:
            raise ToolError(
                "INVALID_ARGUMENT", "params.eps (expansibility) must be between 0 and 1."
            )
        beta, At = d / D, math.pi * d * d / 4
        k = Cd * eps * At / math.sqrt(1 - beta**4)
        if one_of(p, ("dp", "Q"), "the pressure difference (dp) or the flow rate (Q)") == "dp":
            dp = need(p, "dp", "pressure")
            Q = k * math.sqrt(2 * dp / rho)
        else:
            Q = need(p, "Q", "flow_rate")
            dp = rho / 2 * (Q / k) ** 2
        out = {
            "beta": beta,
            "A_throat": At,
            "Q": Q,
            "dp": dp,
            "v_throat": Q / At,
            "v_pipe": Q / (math.pi * D * D / 4),
            "m_dot": rho * Q,
        }
        ctx.formula = "Q = Cd eps A_throat sqrt(2 dp / rho) / sqrt(1 - beta^4); beta = d / D"
        ctx.formatted = f"Q = {fmt_float(Q * 3600, 5)} m3/h with dp = {fmt_float(dp / 1e3, 5)} kPa (beta = {fmt_float(beta, 4)})."
    else:
        g = _grav(p)
        if typ == "weir_rect":
            k = Cd * 2 / 3 * need(p, "b", "length") * math.sqrt(2 * g)
            expo = 1.5
            ctx.formula = "rectangular weir: Q = Cd (2/3) b sqrt(2 g) h^(3/2)"
        else:
            th = need(p, "theta_deg", None)
            if th >= 180:
                raise ToolError(
                    "INVALID_ARGUMENT", "params.theta_deg must be between 0 and 180 degrees."
                )
            k = Cd * 8 / 15 * math.tan(math.radians(th) / 2) * math.sqrt(2 * g)
            expo = 2.5
            ctx.formula = "triangular weir: Q = Cd (8/15) tan(theta/2) sqrt(2 g) h^(5/2)"
        if one_of(p, ("h", "Q"), "the head (h) or the flow rate (Q)") == "h":
            h = need(p, "h", "length")
            Q = k * h**expo
        else:
            Q = need(p, "Q", "flow_rate")
            h = (Q / k) ** (1 / expo)
        out = {"Q": Q, "h": h}
        ctx.formatted = f"Q = {fmt_float(Q * 1000, 5)} L/s with head {fmt_float(h * 1000, 4)} mm."
    ctx.units = units_for(
        out,
        {
            "beta": "1",
            "A_throat": "m2",
            "Q": "m3/s",
            "dp": "Pa",
            "v_throat": "m/s",
            "v_pipe": "m/s",
            "m_dot": "kg/s",
            "h": "m",
        },
    )
    return out


# ------------------------------------------------------------------ compressible_flow

_COMP = {
    "isentropic": {"M", "T0", "p0", "R", "gamma"},
    "area_mach": {"A_ratio", "branch", "T0", "p0", "R", "gamma"},
    "normal_shock": {"M1", "gamma"},
    "choked": {"A", "p0", "T0", "R", "gamma"},
}


def _isen(M: float, g: float) -> dict[str, float]:
    t = 1 + (g - 1) / 2 * M * M
    return {
        "M": M,
        "T_T0": 1 / t,
        "p_p0": t ** (-g / (g - 1)),
        "rho_rho0": t ** (-1 / (g - 1)),
        "A_Astar": (2 * t / (g + 1)) ** ((g + 1) / (2 * (g - 1))) / M,
    }


@op("compressible_flow", ("mode", "M", "T0", "p0", "R", "gamma", "A_ratio", "branch", "M1", "A"))
def _compressible(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_COMP))
    only(p, mode, _COMP[mode])
    g = qty(p, "gamma", None, default=GAMMA_AIR)
    assert g is not None
    if g <= 1:
        raise ToolError("INVALID_ARGUMENT", "params.gamma must be > 1.")
    R = qty(p, "R", "specific_heat", default=R_AIR)
    assert R is not None
    out: dict[str, Any]
    if mode in ("isentropic", "area_mach"):
        if mode == "isentropic":
            M = need(p, "M", None)
        else:
            ar = need(p, "A_ratio", None)
            if ar < 1:
                raise ToolError("INVALID_ARGUMENT", "params.A_ratio = A/A* must be >= 1.")
            branch = choice(p, "branch", ("subsonic", "supersonic"))

            def f(m: float) -> float:
                return _isen(m, g)["A_Astar"] - ar

            if ar == 1:
                M = 1.0
            elif branch == "subsonic":
                M = _brentq(f, 1e-12, 1.0)
            else:
                M = _brentq(f, 1.0, _expand(f, 2.0))
        out = _isen(M, g)
        ctx.formula = (
            "T0/T = 1 + (gamma-1)/2 M^2; p/p0 = (T/T0)^(gamma/(gamma-1)); "
            "rho/rho0 = (T/T0)^(1/(gamma-1)); "
            "A/A* = (1/M) [2/(gamma+1) (1 + (gamma-1)/2 M^2)]^((gamma+1)/(2(gamma-1)))"
        )
        if M >= 1:  # D116: half-angle of the Mach cone ("Mach wave")
            out["mach_angle_deg"] = math.degrees(math.asin(1 / M))
            ctx.formula += "; Mach angle mu = asin(1/M) for M >= 1"
        T0 = temperature(p, "T0", need=False)
        p0 = qty(p, "p0", "pressure", need=False)
        if T0 is not None:
            T = T0 * out["T_T0"]
            a = math.sqrt(g * R * T)
            out |= {"T": T, "a": a, "V": M * a}
            ctx.formula += "; a = sqrt(gamma R T), V = M a"
        if p0 is not None:
            out["p"] = p0 * out["p_p0"]
            if T0 is not None:
                out["rho"] = out["p"] / (R * out["T"])
                out["mass_flux"] = out["rho"] * out["V"]
                ctx.formula += "; rho = p / (R T)"
        ctx.formatted = (
            f"M = {fmt_float(M, 5)}: T/T0 = {fmt_float(out['T_T0'], 5)}, p/p0 = {fmt_float(out['p_p0'], 5)}, "
            f"A/A* = {fmt_float(out['A_Astar'], 5)}."
        )
    elif mode == "normal_shock":
        M1 = need(p, "M1", None)
        if M1 <= 1:
            raise ToolError("INVALID_ARGUMENT", "Normal shock: M1 > 1 is needed (supersonic flow).")
        m2 = M1 * M1
        p21 = 1 + 2 * g / (g + 1) * (m2 - 1)
        r21 = (g + 1) * m2 / ((g - 1) * m2 + 2)
        out = {
            "M2": math.sqrt((1 + (g - 1) / 2 * m2) / (g * m2 - (g - 1) / 2)),
            "p2_p1": p21,
            "T2_T1": p21 / r21,
            "rho2_rho1": r21,
            "p02_p01": r21 ** (g / (g - 1)) * p21 ** (-1 / (g - 1)),
        }
        ctx.formula = (
            "Rankine-Hugoniot: M2^2 = (1 + (gamma-1)/2 M1^2) / (gamma M1^2 - (gamma-1)/2); "
            "p2/p1 = 1 + 2 gamma/(gamma+1) (M1^2 - 1); rho2/rho1 = (gamma+1) M1^2 / ((gamma-1) M1^2 + 2); "
            "T2/T1 = (p2/p1)/(rho2/rho1); p02/p01 = (rho2/rho1)^(gamma/(gamma-1)) (p2/p1)^(-1/(gamma-1))"
        )
        ctx.formatted = (
            f"After the shock M2 = {fmt_float(out['M2'], 5)}, p2/p1 = {fmt_float(p21, 5)}, "
            f"total pressure loss {fmt_float((1 - out['p02_p01']) * 100, 4)}%."
        )
    else:
        A, p0 = need(p, "A", "area"), need(p, "p0", "pressure")
        T0 = temperature(p, "T0")
        assert T0 is not None
        Ts = T0 * 2 / (g + 1)
        out = {
            "m_dot": A * p0 * math.sqrt(g / (R * T0)) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1))),
            "p_star": p0 * (2 / (g + 1)) ** (g / (g - 1)),
            "T_star": Ts,
            "V_star": math.sqrt(g * R * Ts),
        }
        ctx.formula = (
            "critical mass flow: m = A p0 sqrt(gamma / (R T0)) (2/(gamma+1))^((gamma+1)/(2(gamma-1))); "
            "p* = p0 (2/(gamma+1))^(gamma/(gamma-1)); T* = 2 T0 / (gamma+1)"
        )
        ctx.formatted = f"Maximum mass flow {fmt_float(out['m_dot'], 5)} kg/s (sonic throat)."
    ctx.units = units_for(
        out,
        {
            "M": "1",
            "T_T0": "1",
            "p_p0": "1",
            "rho_rho0": "1",
            "A_Astar": "1",
            "mach_angle_deg": "deg",
            "T": "K",
            "a": "m/s",
            "V": "m/s",
            "p": "Pa",
            "rho": "kg/m3",
            "mass_flux": "kg/(m2*s)",
            "M2": "1",
            "p2_p1": "1",
            "T2_T1": "1",
            "rho2_rho1": "1",
            "p02_p01": "1",
            "m_dot": "kg/s",
            "p_star": "Pa",
            "T_star": "K",
            "V_star": "m/s",
        },
    )
    return out
