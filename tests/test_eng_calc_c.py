"""eng_calc area C (D39): fluids and heat. Independent methods in the tests.

Pipes: Hagen-Poiseuille in laminar flow, Haaland and Swamee-Jain (different explicit formulas) in turbulent flow.
Exchanger: numerical integration along the area (energy balance step by step).
Layered wall: resistances in series summed by hand. Air: speed of sound from gamma R T.
"""

from __future__ import annotations

import math

import pytest
from scipy.integrate import solve_ivp

from tests.conftest import call, ok


def e(operation: str, **params: object) -> dict:
    return ok("eng_calc", operation=operation, params=params)


def err(operation: str, **params: object) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert resp["ok"] is False
    return resp["error"]["code"]


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------------------------------------------------------ fluid_pipe


def test_laminar_matches_hagen_poiseuille() -> None:
    D, L, Q, mu, rho = 0.01, 5.0, 2e-6, 0.1, 900.0
    r = e("fluid_pipe", D=D, L=L, Q=Q, rho=rho, mu=mu)
    assert r["regime"] == "laminar" and r["Re"] < 2300
    assert rel(r["dp_friction"], 128 * mu * L * Q / (math.pi * D**4), 1e-12)


def haaland(Re: float, rr: float) -> float:
    return (-1.8 * math.log10((rr / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def swamee_jain(Re: float, rr: float) -> float:
    return 0.25 / math.log10(rr / 3.7 + 5.74 / Re**0.9) ** 2


@pytest.mark.parametrize(("Re_target", "rr"), [(1e4, 0.0), (1e5, 1e-4), (1e6, 1e-3), (5e6, 0.01)])
def test_turbulent_colebrook_close_to_explicit_formulas(Re_target: float, rr: float) -> None:
    D, rho, mu = 0.1, 1000.0, 1e-3
    v = Re_target * mu / (rho * D)
    r = e("fluid_pipe", D=D, v=v, rho=rho, mu=mu, eps=rr * D, L=100)
    assert rel(r["Re"], Re_target, 1e-12) and r["regime"] == "turbulent"
    f = r["f"]
    # Colebrook satisfied and close to the two explicit formulas (declared error < 2 %)
    assert (
        abs(1 / math.sqrt(f) + 2 * math.log10(rr / 3.7 + 2.51 / (Re_target * math.sqrt(f)))) < 1e-10
    )
    assert (
        abs(f / haaland(Re_target, rr) - 1) < 0.02
        and abs(f / swamee_jain(Re_target, rr) - 1) < 0.02
    )
    assert rel(r["dp_friction"], f * (100 / D) * rho * v * v / 2)


def test_pipe_local_losses_pump_and_units() -> None:
    r = e(
        "fluid_pipe",
        D=[50, "mm"],
        Q=[10, "m3/h"],
        L=[30, "m"],
        rho=[1000, "kg/m3"],
        mu=[1, "cP"],
        eps=[0.05, "mm"],
        K=4.5,
        dz=[8, "m"],
        p1=[3, "bar"],
    )
    v = (10 / 3600) / (math.pi * 0.05**2 / 4)
    assert rel(r["v"], v) and rel(r["dp_local"], 4.5 * 1000 * v * v / 2)
    H = r["h_friction"] + r["h_local"] + 8
    assert rel(r["H_pump"], H) and rel(r["P_hydraulic"], 1000 * 9.80665 * (10 / 3600) * H)
    assert rel(r["p2"], 3e5 - r["dp_friction"] - r["dp_local"] - 1000 * 9.80665 * 8)
    r2 = e("fluid_pipe", D=0.05, Q=[10, "m3/h"], L=30, rho=1000, nu=[1, "cSt"], eps=0.05e-3)
    assert rel(r2["Re"], r["Re"], 1e-12)


def test_pipe_transition_warns_and_bad_inputs() -> None:
    resp = call(
        "eng_calc", operation="fluid_pipe", params={"D": 0.01, "v": 0.3, "rho": 1000, "mu": 1e-3}
    )
    assert resp["ok"] and resp["result"]["regime"] == "transitional" and resp["warnings"]
    assert err("fluid_pipe", D=0.01, v=1, Q=1, rho=1000, mu=1e-3) == "INVALID_ARGUMENT"
    assert err("fluid_pipe", D=0.01, v=1, rho=1000) == "INVALID_ARGUMENT"
    assert err("fluid_pipe", D=0.01, v=1, rho=1000, mu=[1, "m2/s"]) == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ aero_hydro_force


def test_drag_lift_mach_terminal() -> None:
    r = e("aero_hydro_force", rho=1.225, v=[108, "km/h"], A=2.2, Cd=0.3, Cl=0.1, T=[15, "C"])
    q = 0.5 * 1.225 * 30**2
    assert rel(r["q"], q) and rel(r["F_D"], q * 0.3 * 2.2) and rel(r["F_L"], q * 0.1 * 2.2)
    assert rel(r["P_drag"], r["F_D"] * 30)
    c = math.sqrt(1.4 * 287.05287 * 288.15)  # R of dry air, ISO 2533
    assert rel(r["c_sound"], c) and rel(r["Mach"], 30 / c)
    t = e("aero_hydro_force", rho=1.225, A=0.5, Cd=1.0, m=80)
    assert rel(t["v_terminal"], math.sqrt(2 * 80 * 9.80665 / (1.225 * 1.0 * 0.5)))


def test_drag_needs_speed_or_mass() -> None:
    assert err("aero_hydro_force", rho=1.2, A=1, Cd=1) == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ heat_transfer


def test_conduction_wall_layers_and_films() -> None:
    layers = [[0.2, 0.7], [0.05, 0.04], [0.01, 0.8]]  # thickness m, conductivity W/(m K)
    r = e(
        "heat_transfer",
        mode="wall",
        layers=layers,
        A=10,
        h_in=8,
        h_out=25,
        T1=[20, "C"],
        T2=[-5, "C"],
    )
    R = 1 / (8 * 10) + sum(t / (k * 10) for t, k in layers) + 1 / (25 * 10)
    assert rel(r["R_total"], R) and rel(r["Q"], 25 / R) and rel(r["U"], 1 / (R * 10))
    one = e("heat_transfer", mode="wall", layers=[[0.1, 50]], A=1, T1=400, T2=300)
    assert rel(one["Q"], 50 * 1 * 100 / 0.1)


def test_cylinder_convection_radiation() -> None:
    r = e(
        "heat_transfer",
        mode="cylinder",
        r1=[10, "mm"],
        r2=[30, "mm"],
        k=0.05,
        L=2,
        T1=[150, "C"],
        T2=[25, "C"],
    )
    assert rel(r["Q"], 125 * 2 * math.pi * 0.05 * 2 / math.log(3))
    c = e("heat_transfer", mode="convection", h=15, A=2, T1=[80, "C"], T2=[20, "C"])
    assert rel(c["Q"], 15 * 2 * 60)
    rad = e("heat_transfer", mode="radiation", emissivity=0.9, A=1.5, T1=[500, "C"], T2=[20, "C"])
    assert rel(rad["Q"], 0.9 * 5.670374419e-8 * 1.5 * (773.15**4 - 293.15**4), 1e-12)


def exchanger_area_by_integration(
    Ch: float, Cc: float, U: float, Th_in: float, Tc_in: float, Q: float, counter: bool
) -> float:
    """Area that exchanges Q integrating dT/dA along the exchanger (parallel flow) or by shooting (counterflow)."""
    if not counter:

        def rhs(_a: float, y: list[float]) -> list[float]:
            dq = U * (y[0] - y[1])
            return [-dq / Ch, dq / Cc, dq]

        stop = lambda _a, y: y[2] - Q  # noqa: E731
        stop.terminal = True  # type: ignore[attr-defined]
        sol = solve_ivp(rhs, (0, 1e6), [Th_in, Tc_in, 0.0], events=stop, rtol=1e-12, atol=1e-12)
        return float(sol.t_events[0][0])
    # counterflow: start from the hot inlet side, where the cold fluid leaves (Tc_out known from Q)
    Tc_out = Tc_in + Q / Cc

    def rhs2(_a: float, y: list[float]) -> list[float]:
        dq = U * (y[0] - y[1])
        return [-dq / Ch, -dq / Cc, dq]

    stop2 = lambda _a, y: y[2] - Q  # noqa: E731
    stop2.terminal = True  # type: ignore[attr-defined]
    sol = solve_ivp(rhs2, (0, 1e6), [Th_in, Tc_out, 0.0], events=stop2, rtol=1e-12, atol=1e-12)
    return float(sol.t_events[0][0])


@pytest.mark.parametrize("flow", ["counter", "parallel"])
def test_lmtd_area_against_integration(flow: str) -> None:
    Th_in, Th_out, Tc_in, Tc_out, U = 150.0, 90.0, 20.0, 60.0, 500.0
    Q = 50_000.0
    Ch, Cc = Q / (Th_in - Th_out), Q / (Tc_out - Tc_in)
    r = e(
        "heat_transfer",
        mode="lmtd",
        flow=flow,
        Th_in=[Th_in, "C"],
        Th_out=[Th_out, "C"],
        Tc_in=[Tc_in, "C"],
        Tc_out=[Tc_out, "C"],
        U=U,
        Q=Q,
    )
    area = exchanger_area_by_integration(Ch, Cc, U, Th_in, Tc_in, Q, flow == "counter")
    assert rel(r["A_required"], area, 1e-7)


def test_heat_transfer_bad_inputs() -> None:
    assert (
        err("heat_transfer", mode="wall", layers=[[0.1]], A=1, T1=300, T2=290) == "INVALID_ARGUMENT"
    )
    assert (
        err("heat_transfer", mode="radiation", emissivity=1.2, A=1, T1=300, T2=290)
        == "INVALID_ARGUMENT"
    )
    assert (
        err("heat_transfer", mode="convection", h=1, A=1, T1=[-300, "C"], T2=290)
        == "INVALID_ARGUMENT"
    )
    assert (
        err(
            "heat_transfer",
            mode="lmtd",
            flow="parallel",
            Th_in=100,
            Th_out=50,
            Tc_in=20,
            Tc_out=60,
            U=1,
            Q=1,
        )
        == "INVALID_ARGUMENT"
    )  # in parallel flow the cold fluid does not exceed the hot outlet


# ------------------------------------------------------------------ heatsink_thermal


def test_heatsink_chain_and_required_rsa() -> None:
    r = e("heatsink_thermal", P=[25, "W"], T_a=[40, "C"], R_jc=0.8, R_cs=0.2, R_sa=2.0)
    assert (
        rel(r["T_j"], 313.15 + 25 * 3.0)
        and rel(r["T_j_C"], 40 + 75)
        and rel(r["T_case_C"], 40 + 25 * 2.2)
    )
    req = e("heatsink_thermal", P=25, T_a=[40, "C"], R_jc=0.8, R_cs=0.2, T_j_max=[125, "C"])
    assert rel(req["R_sa_required"], (125 - 40) / 25 - 1.0)
    back = e("heatsink_thermal", P=25, T_a=[40, "C"], R_jc=0.8, R_cs=0.2, R_sa=req["R_sa_required"])
    assert rel(back["T_j_C"], 125.0)
    assert (
        err("heatsink_thermal", P=200, T_a=[40, "C"], R_jc=0.8, R_cs=0.2, T_j_max=[125, "C"])
        == "MATH_ERROR"
    )
