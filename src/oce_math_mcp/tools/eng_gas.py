"""eng_calc area F (D43): ideal gases and processes, ideal air-standard cycles, calorimetry.
Formulas in the server, checked in the tests with independent methods.

Constants: R_u = 8.314462618 J/(mol K) (CODATA 2018, exact in the SI since 2019); dry air
gamma = 1.4 and R = 287.05287 J/(kg K) (ISO 2533), the same as in area C.
"""

from __future__ import annotations

import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import choice, need, one_of, only, op, qty, temperature, units_for
from .eng_thermo import GAMMA_AIR, R_AIR

R_U = 8.314462618


def _gamma(p: dict[str, Any]) -> float:
    g = qty(p, "gamma", None, default=GAMMA_AIR)
    assert g is not None
    if g <= 1:
        raise ToolError("INVALID_ARGUMENT", "params.gamma must be > 1.")
    return g


def _r_gas(p: dict[str, Any]) -> float:
    r = qty(p, "R", "specific_heat", default=R_AIR)
    assert r is not None
    return r


# ------------------------------------------------------------------ ideal_gas

_GAS = {
    "state": {"p", "V", "T", "m", "n_mol", "R", "M_molar"},
    "process": {"process", "p1", "T1", "V1", "m", "p2", "V2", "T2", "gamma", "R", "n_poly"},
}
_PROCESSES = ("isothermal", "isobaric", "isochoric", "isentropic", "polytropic")


def _state(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if p.get("R") is not None and p.get("M_molar") is not None:
        raise ToolError("INVALID_ARGUMENT", "Give R or M_molar, not both.")
    if p.get("m") is not None and p.get("n_mol") is not None:
        raise ToolError("INVALID_ARGUMENT", "The amount of gas is given with m or with n_mol.")
    M_molar = qty(p, "M_molar", "molar_mass", need=False)
    R = R_U / M_molar if M_molar is not None else _r_gas(p)
    M_molar = R_U / R
    given = [k for k in ("p", "V", "T") if p.get(k) is not None]
    has_amount = p.get("m") is not None or p.get("n_mol") is not None
    if len(given) + has_amount != 3:
        raise ToolError(
            "INVALID_ARGUMENT",
            "state: exactly three of p, V, T and the amount of gas (m or n_mol) are needed.",
        )
    pr = qty(p, "p", "pressure", need=False)
    V = qty(p, "V", "volume", need=False)
    T = temperature(p, "T", need=False)
    if p.get("m") is not None:
        N: float | None = need(p, "m", "mass") / M_molar
    elif p.get("n_mol") is not None:
        N = need(p, "n_mol", "amount_of_substance")
    else:
        N = None
    if pr is None:
        assert N is not None and T is not None and V is not None
        pr = N * R_U * T / V
    elif V is None:
        assert N is not None and T is not None
        V = N * R_U * T / pr
    elif T is None:
        assert N is not None
        T = pr * V / (N * R_U)
    else:
        N = pr * V / (R_U * T)
    assert N is not None
    m = N * M_molar
    ctx.formula = "p V = n R_u T = m R T; R = R_u / M"
    ctx.formatted = (
        f"p = {fmt_float(pr / 1e5, 6)} bar, V = {fmt_float(V * 1000, 6)} L, "
        f"T = {fmt_float(T - 273.15, 5)} °C, m = {fmt_float(m, 6)} kg."
    )
    return {"p": pr, "V": V, "T": T, "m": m, "n_mol": N, "rho": m / V, "R": R, "M_molar": M_molar}


def _process(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    proc = choice(p, "process", _PROCESSES)
    if proc != "polytropic" and p.get("n_poly") is not None:
        raise ToolError("INVALID_ARGUMENT", "n_poly is valid only with process 'polytropic'.")
    g, R = _gamma(p), _r_gas(p)
    p1 = need(p, "p1", "pressure")
    T1 = temperature(p, "T1")
    assert T1 is not None
    if one_of(p, ("V1", "m"), "state 1 (V1 or m)") == "V1":
        V1 = need(p, "V1", "volume")
        m = p1 * V1 / (R * T1)
    else:
        m = need(p, "m", "mass")
        V1 = m * R * T1 / p1
    fin = one_of(p, ("p2", "V2", "T2"), "state 2")
    forbidden = {"isothermal": "T2", "isobaric": "p2", "isochoric": "V2"}.get(proc)
    if fin == forbidden:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"In '{proc}' {forbidden} is equal to state 1: give another quantity of state 2.",
        )
    n: float | None = {"isothermal": 1.0, "isobaric": 0.0, "isochoric": None, "isentropic": g}.get(
        proc
    )
    if proc == "polytropic":
        n = need(p, "n_poly", None, positive=False)
        if n < 0:
            raise ToolError("INVALID_ARGUMENT", "params.n_poly must be >= 0.")
    p2 = qty(p, "p2", "pressure", need=False)
    V2 = qty(p, "V2", "volume", need=False)
    T2 = temperature(p, "T2", need=False)
    if n is None:  # isochoric
        V2 = V1
        if p2 is not None:
            T2 = T1 * p2 / p1
        else:
            assert T2 is not None
            p2 = p1 * T2 / T1
    else:
        if fin == "T2":
            assert T2 is not None
            if n == 1:
                raise ToolError("INVALID_ARGUMENT", "With n_poly = 1 (isothermal) T2 = T1.")
            V2 = V1 * (T1 / T2) ** (1 / (n - 1))
        elif fin == "p2":
            assert p2 is not None
            if n == 0:
                raise ToolError("INVALID_ARGUMENT", "With n_poly = 0 (isobaric) p2 = p1.")
            V2 = V1 * (p1 / p2) ** (1 / n)
        assert V2 is not None
        p2 = p1 * (V1 / V2) ** n
        T2 = p2 * V2 / (m * R)
    assert p2 is not None and V2 is not None and T2 is not None
    cv = R / (g - 1)
    cp = g * cv
    dU = m * cv * (T2 - T1)
    if n is None:
        W = 0.0
    elif n == 1:
        W, dU = p1 * V1 * math.log(V2 / V1), 0.0
    else:
        W = (p2 * V2 - p1 * V1) / (1 - n)
    Q = 0.0 if proc == "isentropic" else dU + W
    dS = 0.0 if proc == "isentropic" else m * (cv * math.log(T2 / T1) + R * math.log(V2 / V1))
    dH = 0.0 if n == 1 else m * cp * (T2 - T1)
    ctx.formula = (
        "p V^n = const (n = 1 isothermal, 0 isobaric, gamma isentropic, infinity isochoric); "
        "W = (p2 V2 - p1 V1) / (1 - n), W = p1 V1 ln(V2 / V1) for n = 1; dU = m cv dT; Q = dU + W; "
        "dS = m (cv ln(T2/T1) + R ln(V2/V1)); cv = R / (gamma - 1)"
    )
    ctx.formatted = (
        f"State 2: p = {fmt_float(p2 / 1e5, 6)} bar, T = {fmt_float(T2 - 273.15, 5)} °C; "
        f"work of the gas {fmt_float(W / 1e3, 5)} kJ, heat {fmt_float(Q / 1e3, 5)} kJ."
    )
    return {
        "m": m,
        "p1": p1,
        "V1": V1,
        "T1": T1,
        "p2": p2,
        "V2": V2,
        "T2": T2,
        "W": W,
        "Q": Q,
        "dU": dU,
        "dH": dH,
        "dS": dS,
        "n_poly": n,
    }


@op("ideal_gas", ("mode", *sorted(_GAS["state"] | _GAS["process"])))
def _ideal_gas(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_GAS))
    only(p, mode, _GAS[mode])
    out = _state(p, ctx) if mode == "state" else _process(p, ctx)
    ctx.units = units_for(
        out,
        {
            "p": "Pa",
            "V": "m3",
            "T": "K",
            "m": "kg",
            "n_mol": "mol",
            "rho": "kg/m3",
            "R": "J/(kg*K)",
            "M_molar": "kg/mol",
            "p1": "Pa",
            "V1": "m3",
            "T1": "K",
            "p2": "Pa",
            "V2": "m3",
            "T2": "K",
            "W": "J",
            "Q": "J",
            "dU": "J",
            "dH": "J",
            "dS": "J/K",
            "n_poly": "1",
        },
    )
    return out


# ------------------------------------------------------------------ thermo_cycle

_CYCLES = {
    "carnot": {"T_hot", "T_cold"},
    "otto": {"r", "T1", "T3", "q_in", "p1"},
    "diesel": {"r", "rc", "T1", "p1"},
    "brayton": {"rp", "T1", "T3", "eta_c", "eta_t", "p1"},
}


def _ratio(p: dict[str, Any], key: str, what: str) -> float:
    v = need(p, key, None)
    if v <= 1:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} ({what}) must be > 1.")
    return v


def _eff(p: dict[str, Any], key: str) -> float:
    v = qty(p, key, None, default=1.0)
    assert v is not None
    if v > 1:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} (efficiency) must be between 0 and 1.")
    return v


@op(
    "thermo_cycle",
    ("cycle", "gamma", "R", *sorted(set().union(*_CYCLES.values()))),
)
def _cycle(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    cyc = choice(p, "cycle", tuple(_CYCLES))
    only(p, cyc, _CYCLES[cyc] | {"gamma", "R"}, field="cycle")
    g, R = _gamma(p), _r_gas(p)
    cv = R / (g - 1)
    cp = g * cv
    out: dict[str, Any]
    if cyc == "carnot":
        Th, Tc = temperature(p, "T_hot"), temperature(p, "T_cold")
        assert Th is not None and Tc is not None
        if Th <= Tc:
            raise ToolError("INVALID_ARGUMENT", "T_hot must be larger than T_cold.")
        out = {
            "efficiency": 1 - Tc / Th,
            "COP_refrigerator": Tc / (Th - Tc),
            "COP_heat_pump": Th / (Th - Tc),
        }
        ctx.formula = (
            "eta = 1 - Tc/Th; COP_refrigerator = Tc/(Th - Tc); COP_heat_pump = Th/(Th - Tc)"
        )
        ctx.formatted = f"Carnot efficiency {fmt_float(out['efficiency'] * 100, 4)}%."
        ctx.units = units_for(out, {k: "1" for k in out})
        return out
    T1 = temperature(p, "T1", need=False)
    p1 = qty(p, "p1", "pressure", need=False)
    if p1 is not None and T1 is None:
        raise ToolError("INVALID_ARGUMENT", "params.p1 is used together with T1.")
    T: list[float] | None = None
    pl: list[float] | None = None
    q_in = q_out = 0.0
    bwr: float | None = None
    if cyc == "otto":
        r = _ratio(p, "r", "compression ratio")
        out = {"efficiency": 1 - r ** (1 - g)}
        ctx.formula = "Otto: eta = 1 - r^(1 - gamma)"
        has_heat = p.get("T3") is not None or p.get("q_in") is not None
        if has_heat and T1 is None:
            raise ToolError("INVALID_ARGUMENT", "T3 and q_in are used together with T1.")
        if T1 is not None:
            T2 = T1 * r ** (g - 1)
            if one_of(p, ("T3", "q_in"), "the heat supplied (T3 or q_in)") == "T3":
                T3 = temperature(p, "T3")
                assert T3 is not None
            else:
                T3 = T2 + need(p, "q_in", "specific_energy") / cv
            if T3 <= T2:
                raise ToolError(
                    "INVALID_ARGUMENT", "T3 must exceed the temperature at the end of compression."
                )
            T4 = T3 * r ** (1 - g)
            T, q_in, q_out = [T1, T2, T3, T4], cv * (T3 - T2), cv * (T4 - T1)
            if p1 is not None:
                p2 = p1 * r**g
                pl = [p1, p2, p2 * T3 / T2, p1 * T4 / T1]
            ctx.formula += "; T2 = T1 r^(gamma-1); q_in = cv (T3 - T2); q_out = cv (T4 - T1)"
    elif cyc == "diesel":
        r = _ratio(p, "r", "compression ratio")
        rc = _ratio(p, "rc", "cut-off ratio")
        if rc >= r:
            raise ToolError("INVALID_ARGUMENT", "params.rc must be smaller than r.")
        out = {"efficiency": 1 - r ** (1 - g) * (rc**g - 1) / (g * (rc - 1))}
        ctx.formula = "Diesel: eta = 1 - r^(1-gamma) (rc^gamma - 1) / (gamma (rc - 1))"
        if T1 is not None:
            T2 = T1 * r ** (g - 1)
            T3 = rc * T2
            T4 = T3 * (rc / r) ** (g - 1)
            T, q_in, q_out = [T1, T2, T3, T4], cp * (T3 - T2), cv * (T4 - T1)
            if p1 is not None:
                p2 = p1 * r**g
                pl = [p1, p2, p2, p1 * T4 / T1]
            ctx.formula += "; q_in = cp (T3 - T2); q_out = cv (T4 - T1)"
    else:
        rp = _ratio(p, "rp", "pressure ratio")
        ec, et = _eff(p, "eta_c"), _eff(p, "eta_t")
        T3 = temperature(p, "T3", need=False)
        if (T1 is None) != (T3 is None):
            raise ToolError("INVALID_ARGUMENT", "brayton: T1 and T3 are given together.")
        x = rp ** ((g - 1) / g)
        out = {"efficiency": 1 - 1 / x}
        ctx.formula = "ideal Brayton: eta = 1 - rp^(-(gamma-1)/gamma)"
        if T1 is None and (ec < 1 or et < 1):
            raise ToolError(
                "INVALID_ARGUMENT", "With eta_c or eta_t smaller than 1, T1 and T3 are needed."
            )
        if T1 is not None:
            assert T3 is not None
            T2 = T1 + T1 * (x - 1) / ec
            T4 = T3 - et * (T3 - T3 / x)
            if T3 <= T2:
                raise ToolError(
                    "MATH_ERROR",
                    "T3 does not exceed the compressor outlet: no heat to supply.",
                )
            wc, wt = cp * (T2 - T1), cp * (T3 - T4)
            T, q_in, q_out, bwr = [T1, T2, T3, T4], cp * (T3 - T2), cp * (T4 - T1), wc / wt
            out["efficiency"] = (wt - wc) / q_in
            if p1 is not None:
                pl = [p1, p1 * rp, p1 * rp, p1]
            ctx.formula += (
                "; real: T2 = T1 + T1 (rp^((gamma-1)/gamma) - 1)/eta_c; "
                "T4 = T3 - eta_t (T3 - T3/rp^((gamma-1)/gamma)); eta = (w_t - w_c)/q_in"
            )
            if wt <= wc:
                ctx.warn("The turbine work does not exceed the compressor work: net work <= 0.")
    if T is not None:
        out |= {
            "T": T,
            "q_in": q_in,
            "q_out": q_out,
            "w_net": q_in - q_out,
            "efficiency_carnot": 1 - T[0] / max(T),
        }
        if pl is not None:
            out["p"] = pl
            if cyc in ("otto", "diesel"):
                v1 = R * T[0] / pl[0]
                out["mep"] = (q_in - q_out) / (v1 - v1 / need(p, "r", None))
        if bwr is not None:
            out["back_work_ratio"] = bwr
    ctx.formatted = f"Cycle efficiency {fmt_float(out['efficiency'] * 100, 4)}%" + (
        f", net work {fmt_float(out['w_net'] / 1e3, 5)} kJ/kg." if "w_net" in out else "."
    )
    ctx.units = units_for(
        out,
        {
            "efficiency": "1",
            "T": "K",
            "p": "Pa",
            "q_in": "J/kg",
            "q_out": "J/kg",
            "w_net": "J/kg",
            "efficiency_carnot": "1",
            "mep": "Pa",
            "back_work_ratio": "1",
        },
    )
    return out


# ------------------------------------------------------------------ calorimetry

_CAL = {"sensible": {"m", "c", "dT", "T1", "T2"}, "latent": {"m", "L"}, "mixture": {"bodies"}}


@op("calorimetry", ("mode", "m", "c", "dT", "T1", "T2", "L", "bodies"))
def _calorimetry(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_CAL))
    only(p, mode, _CAL[mode])
    out: dict[str, Any]
    if mode == "sensible":
        m, c = need(p, "m", "mass"), need(p, "c", "specific_heat")
        if p.get("dT") is not None:
            if p.get("T1") is not None or p.get("T2") is not None:
                raise ToolError("INVALID_ARGUMENT", "Give dT or the pair T1, T2.")
            dT = need(p, "dT", None, positive=False)
        else:
            T1, T2 = temperature(p, "T1"), temperature(p, "T2")
            assert T1 is not None and T2 is not None
            dT = T2 - T1
        out = {"Q": m * c * dT, "dT": dT}
        ctx.formula = "Q = m c dT"
        ctx.formatted = f"Heat {fmt_float(m * c * dT / 1e3, 6)} kJ."
    elif mode == "latent":
        Q = need(p, "m", "mass") * need(p, "L", "specific_energy")
        out = {"Q": Q}
        ctx.formula = "Q = m L"
        ctx.formatted = f"Heat {fmt_float(Q / 1e3, 6)} kJ."
    else:
        bodies = p.get("bodies")
        if not isinstance(bodies, list) or not 2 <= len(bodies) <= 100:
            raise ToolError(
                "INVALID_ARGUMENT",
                'params.bodies: list of 2 to 100 objects {"m": ..., "c": ..., "T": ...}.',
            )
        mc: list[tuple[float, float]] = []
        for i, b in enumerate(bodies):
            if not isinstance(b, dict) or set(b) != {"m", "c", "T"}:
                raise ToolError(
                    "INVALID_ARGUMENT", f'params.bodies[{i}]: the keys "m", "c", "T" are needed.'
                )
            Tb = temperature(b, "T")
            assert Tb is not None
            mc.append((need(b, "m", "mass") * need(b, "c", "specific_heat"), Tb))
        Teq = sum(k * t for k, t in mc) / sum(k for k, _ in mc)
        out = {"T_eq": Teq, "T_eq_C": Teq - 273.15, "Q": [k * (Teq - t) for k, t in mc]}
        ctx.formula = "T_eq = sum(m c T) / sum(m c); Q_i = m_i c_i (T_eq - T_i)"
        ctx.formatted = f"Equilibrium temperature {fmt_float(Teq - 273.15, 5)} °C."
    ctx.units = units_for(out, {"Q": "J", "dT": "K", "T_eq": "K", "T_eq_C": "C"})
    return out
