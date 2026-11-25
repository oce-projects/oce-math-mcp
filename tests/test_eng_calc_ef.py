"""eng_calc areas E (fluids) and F (thermodynamics), decision D43.

Independent methods: force on a plane wall with the numerical integral of the pressure; floating
with the equilibrium solved by brentq; affinity with the dimensionless coefficients; critical depth from
the numerical minimum of the specific energy; weirs with the integral of the Torricelli velocity;
meters with Bernoulli and continuity solved by SymPy; normal shock with the conservation of mass,
momentum and energy; critical mass flow from the numerical maximum; work of processes and cycles
with the numerical integral of p dV; entropy with the integral of dQ/T; mixing with a zero balance.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

from tests.conftest import call, ok

G = 9.80665
R_AIR = 287.05287
RU = 8.314462618


def e(operation: str, **params: object) -> dict:
    return ok("eng_calc", operation=operation, params=params)


def full(operation: str, **params: object) -> dict:
    resp = call("eng_calc", operation=operation, params=params)
    assert resp["ok"], resp["error"]
    return resp


def err(operation: str, **params: object) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert resp["ok"] is False, resp["result"]
    return resp["error"]["code"]


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------------------------------------------------------ hydrostatics


def test_pressure() -> None:
    r = e("hydrostatics", mode="pressure", rho=1000, h=10, p0=[1, "bar"])
    assert rel(r["p_hydro"], 1000 * G * 10) and rel(r["p"], 1e5 + 1000 * G * 10)


@pytest.mark.parametrize("m", [1.5, 3.0])
def test_buoyancy_equilibrium(m: float) -> None:
    rho, V = 1000.0, 0.002
    r = e("hydrostatics", mode="buoyancy", rho=rho, V=[2, "L"], m=m)
    assert rel(r["F_b"], rho * G * V) and rel(r["W"], m * G) and rel(r["F_net"], r["F_b"] - r["W"])
    if rho * V > m:  # floats: immersed fraction from the force equilibrium
        x = brentq(lambda f: rho * G * V * f - m * G, 0, 1, xtol=1e-15)
        assert r["floats"] is True and rel(r["submerged_fraction"], x)
    else:
        assert r["floats"] is False and r["submerged_fraction"] == 1.0


def _wall_by_integral(width, y0: float, y1: float, theta: float, rho: float) -> tuple[float, float]:
    s = math.sin(math.radians(theta))
    F = quad(lambda y: rho * G * y * s * width(y), y0, y1, epsabs=0, epsrel=1e-13)[0]
    M = quad(lambda y: rho * G * y * s * width(y) * y, y0, y1, epsabs=0, epsrel=1e-13)[0]
    return F, M / F


def test_plane_wall_rectangle_inclined() -> None:
    r = e(
        "hydrostatics",
        mode="plane_wall",
        rho=1000,
        shape="rectangle",
        b=2,
        L=3,
        theta_deg=60,
        y_top=1.5,
    )
    F, ycp = _wall_by_integral(lambda y: 2.0, 1.5, 4.5, 60, 1000)
    assert rel(r["F"], F, 1e-10) and rel(r["y_cp"], ycp, 1e-10)
    assert rel(r["h_cp"], ycp * math.sin(math.radians(60)), 1e-10)
    assert rel(r["cp_offset"], r["y_cp"] - r["y_c"], 1e-12)


def test_plane_wall_circle_vertical() -> None:
    D, yt = 1.2, 0.5
    r = e("hydrostatics", mode="plane_wall", rho=1025, shape="circle", D=D, theta_deg=90, y_top=yt)
    yc = yt + D / 2
    F, ycp = _wall_by_integral(
        lambda y: 2 * math.sqrt(max(0.0, (D / 2) ** 2 - (y - yc) ** 2)), yt, yt + D, 90, 1025
    )
    assert rel(r["F"], F, 1e-8) and rel(r["y_cp"], ycp, 1e-8)


# ------------------------------------------------------------------ pump_system


def test_pump_power() -> None:
    r = e("pump_system", mode="power", Q=[36, "m3/h"], H=20, rho=1000, eta=0.7, n=[1450, "rpm"])
    Ph = 1000 * G * 0.01 * 20
    assert rel(r["P_hydraulic"], Ph) and rel(r["P_shaft"], Ph / 0.7)
    assert rel(r["T_shaft"], Ph / 0.7 / (1450 * 2 * math.pi / 60)) and rel(r["dp"], 1000 * G * 20)
    r2 = e("pump_system", mode="power", Q=0.01, dp=[2, "bar"], rho=850)
    assert rel(r2["H"], 2e5 / (850 * G)) and rel(r2["P_shaft"], 2e5 * 0.01)


def test_pump_affinity_dimensionless() -> None:
    resp = full(
        "pump_system",
        mode="affinity",
        Q1=0.01,
        H1=20,
        P1=3000,
        n1=[1450, "rpm"],
        n2=[2900, "rpm"],
        D1=0.2,
        D2=0.25,
    )
    r = resp["result"]
    n1, n2 = 1450 * 2 * math.pi / 60, 2900 * 2 * math.pi / 60
    assert rel(r["Q2"] / (n2 * 0.25**3), 0.01 / (n1 * 0.2**3))
    assert rel(G * r["H2"] / (n2**2 * 0.25**2), G * 20 / (n1**2 * 0.2**2))
    assert rel(r["P2"] / (1000 * n2**3 * 0.25**5), 3000 / (1000 * n1**3 * 0.2**5))
    assert resp["warnings"], "D1 != D2 must give the warning about impeller trimming"
    r1 = e("pump_system", mode="affinity", H1=30, n1=1, n2=0.5)
    assert set(r1) >= {"H2"} and "Q2" not in r1 and rel(r1["H2"], 7.5)


def test_npsh() -> None:
    resp = full(
        "pump_system",
        mode="npsh",
        p_s=101325,
        p_v=[2.339, "kPa"],
        rho=998,
        z_s=-3,
        h_loss=0.5,
        NPSH_r=4,
    )
    r = resp["result"]
    # energy per unit weight from the free surface to the inlet, zero height on the pump axis
    head_inlet = 101325 / (998 * G) + (-3) - 0.5
    assert rel(r["NPSH_a"], head_inlet - 2339 / (998 * G))
    assert rel(r["margin"], r["NPSH_a"] - 4)
    bad = full("pump_system", mode="npsh", p_s=101325, p_v=2339, rho=998, z_s=-8, NPSH_r=4)
    assert bad["result"]["margin"] < 0 and bad["warnings"]


def test_cylinder() -> None:
    r = e(
        "pump_system",
        mode="cylinder",
        D=[63, "mm"],
        d=[36, "mm"],
        p=[160, "bar"],
        Q=[20, "L/min"],
        eta=0.95,
    )
    Ab, Ad = math.pi * 0.063**2 / 4, math.pi * 0.036**2 / 4
    assert rel(r["F_push"], 160e5 * Ab * 0.95) and rel(r["F_pull"], 160e5 * (Ab - Ad) * 0.95)
    assert rel(r["v_out"], 20 / 60000 / Ab) and rel(r["v_in"], 20 / 60000 / (Ab - Ad))
    assert rel(r["area_ratio"], Ab / (Ab - Ad))
    assert err("pump_system", mode="cylinder", D=0.05, d=0.06, p=1e6) == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ open_channel


def test_rectangular_manning_and_critical() -> None:
    r = e("open_channel", shape="rectangular", b=3, n=0.015, S=0.001, y=1.2)
    A, P = 3.6, 5.4
    v = (A / P) ** (2 / 3) * 0.001**0.5 / 0.015
    assert rel(r["v"], v) and rel(r["Q"], v * A) and rel(r["Fr"], v / math.sqrt(G * 1.2))
    assert rel(r["y_critical"], (r["Q"] ** 2 / (G * 9)) ** (1 / 3), 1e-9)
    back = e("open_channel", shape="rectangular", b=3, n=0.015, S=0.001, Q=r["Q"])
    assert rel(back["y"], 1.2, 1e-9) and back["regime"] == "subcritical"


def test_trapezoidal_normal_and_critical() -> None:
    b, z, n, S, Q = 2.0, 1.5, 0.025, 0.0005, 5.0
    r = e("open_channel", shape="trapezoidal", b=b, z=z, n=n, S=S, Q=Q)
    y = r["y"]
    A, P = (b + z * y) * y, b + 2 * y * math.sqrt(1 + z * z)
    assert rel((1 / n) * A * (A / P) ** (2 / 3) * S**0.5, Q, 1e-9)

    def energy(h: float) -> float:
        return h + Q**2 / (2 * G * ((b + z * h) * h) ** 2)

    yc = minimize_scalar(energy, bounds=(0.05, 3), method="bounded", options={"xatol": 1e-12}).x
    assert rel(r["y_critical"], yc, 1e-6)


def test_circular_partial_and_max() -> None:
    D, n, S = 1.0, 0.013, 0.002
    half = e("open_channel", shape="circular", D=D, n=n, S=S, y=0.5)
    assert rel(half["A"], math.pi / 8) and rel(half["R_h"], D / 4)
    fullp = e("open_channel", shape="circular", D=D, n=n, S=S, y=D)
    assert rel(fullp["Q"], (1 / n) * math.pi / 4 * (D / 4) ** (2 / 3) * S**0.5, 1e-9)

    def q_of(y: float) -> float:
        th = 2 * math.acos(1 - 2 * y / D)
        A, P = D * D / 8 * (th - math.sin(th)), D * th / 2
        return (1 / n) * A * (A / P) ** (2 / 3) * S**0.5

    best = minimize_scalar(
        lambda y: -q_of(y), bounds=(0.5, 0.9999), method="bounded", options={"xatol": 1e-12}
    )
    assert rel(half["Q_max"], -best.fun, 1e-9) and rel(half["y_Q_max"], best.x, 1e-5)
    assert (
        err("open_channel", shape="circular", D=D, n=n, S=S, Q=half["Q_max"] * 1.01) == "MATH_ERROR"
    )


# ------------------------------------------------------------------ flow_meter


def test_venturi_bernoulli_continuity() -> None:
    D, d, rho, dp, Cd = 0.1, 0.05, 1000.0, 2e4, 0.98
    v1, v2 = sp.symbols("v1 v2", positive=True)
    sol = sp.solve(
        [sp.Eq(dp, rho / 2 * (v2**2 - v1**2)), sp.Eq(v1 * D**2, v2 * d**2)], [v1, v2], dict=True
    )[0]
    Q_ideal = float(sol[v2]) * math.pi * d * d / 4
    r = e("flow_meter", type="venturi", D=D, d=d, Cd=Cd, rho=rho, dp=[20, "kPa"])
    assert rel(r["Q"], Cd * Q_ideal, 1e-12) and rel(r["m_dot"], rho * r["Q"])
    back = e("flow_meter", type="venturi", D=D, d=d, Cd=Cd, rho=rho, Q=r["Q"])
    assert rel(back["dp"], dp, 1e-12)
    gas = e("flow_meter", type="orifice", D=D, d=d, Cd=0.61, rho=1.2, dp=500, eps=0.99)
    assert rel(gas["Q"] / (0.61 * 0.99), Q_ideal * math.sqrt((500 / 1.2) / (dp / rho)), 1e-12)


def test_weirs_torricelli_integral() -> None:
    b, h, Cd = 1.5, 0.3, 0.62
    q = quad(lambda z: b * math.sqrt(2 * G * z), 0, h, epsabs=0, epsrel=1e-13)[0]
    r = e("flow_meter", type="weir_rect", b=b, Cd=Cd, h=h)
    assert rel(r["Q"], Cd * q, 1e-11)
    assert rel(e("flow_meter", type="weir_rect", b=b, Cd=Cd, Q=r["Q"])["h"], h, 1e-12)
    t = math.tan(math.radians(45))
    qv = quad(lambda z: 2 * t * z * math.sqrt(2 * G * (h - z)), 0, h, epsabs=0, epsrel=1e-13)[0]
    rv = e("flow_meter", type="weir_v", theta_deg=90, Cd=0.59, h=h)
    assert rel(rv["Q"], 0.59 * qv, 1e-10)


# ------------------------------------------------------------------ compressible_flow


def _flux(M: float, T0: float, p0: float, g: float = 1.4) -> float:
    T = T0 / (1 + (g - 1) / 2 * M * M)
    p = p0 * (T / T0) ** (g / (g - 1))
    return p / (R_AIR * T) * M * math.sqrt(g * R_AIR * T)


def test_isentropic_continuity() -> None:
    r = e("compressible_flow", mode="isentropic", M=2.0, T0=300, p0=[500, "kPa"])
    assert rel(r["T_T0"], 1 / 1.8) and rel(r["p_p0"], (1 / 1.8) ** 3.5)
    assert rel(r["A_Astar"], _flux(1, 300, 5e5) / _flux(2, 300, 5e5), 1e-12)
    assert rel(r["p"], r["rho"] * R_AIR * r["T"]) and rel(r["mass_flux"], _flux(2, 300, 5e5))


@pytest.mark.parametrize("branch", ["subsonic", "supersonic"])
def test_area_mach_round_trip(branch: str) -> None:
    r = e("compressible_flow", mode="area_mach", A_ratio=2.0, branch=branch)
    assert (r["M"] < 1) if branch == "subsonic" else (r["M"] > 1)
    assert rel(e("compressible_flow", mode="isentropic", M=r["M"])["A_Astar"], 2.0, 1e-10)


def test_normal_shock_conservation() -> None:
    g, M1, T1, p1 = 1.4, 2.5, 250.0, 4e4
    r = e("compressible_flow", mode="normal_shock", M1=M1)
    cp = g * R_AIR / (g - 1)
    rho1, V1 = p1 / (R_AIR * T1), M1 * math.sqrt(g * R_AIR * T1)
    T2, p2 = T1 * r["T2_T1"], p1 * r["p2_p1"]
    rho2, V2 = p2 / (R_AIR * T2), r["M2"] * math.sqrt(g * R_AIR * T2)
    assert rel(rho1 * V1, rho2 * V2, 1e-12) and rel(rho2 / rho1, r["rho2_rho1"], 1e-12)
    assert rel(p1 + rho1 * V1**2, p2 + rho2 * V2**2, 1e-12)
    assert rel(cp * T1 + V1**2 / 2, cp * T2 + V2**2 / 2, 1e-12)
    p01 = p1 * (1 + (g - 1) / 2 * M1**2) ** (g / (g - 1))
    p02 = p2 * (1 + (g - 1) / 2 * r["M2"] ** 2) ** (g / (g - 1))
    assert rel(r["p02_p01"], p02 / p01, 1e-12)
    assert err("compressible_flow", mode="normal_shock", M1=0.8) == "INVALID_ARGUMENT"


def test_choked_is_maximum() -> None:
    A, p0, T0 = 1e-4, 5e5, 300.0
    r = e("compressible_flow", mode="choked", A=[1, "cm2"], p0=[5, "bar"], T0=T0)
    best = minimize_scalar(
        lambda M: -_flux(M, T0, p0), bounds=(0.2, 3), method="bounded", options={"xatol": 1e-12}
    )
    assert rel(r["m_dot"], -best.fun * A, 1e-10) and abs(best.x - 1) < 1e-5
    assert rel(r["p_star"], p0 * (2 / 2.4) ** 3.5) and rel(r["T_star"], T0 / 1.2)


# ------------------------------------------------------------------ ideal_gas


def test_state_molar_volume_codata() -> None:
    r = e("ideal_gas", mode="state", n_mol=1, T=273.15, p=101325)
    assert rel(
        r["V"], 22.41396954e-3, 1e-9
    )  # CODATA 2018, molar volume at 273.15 K and 101.325 kPa
    a = e("ideal_gas", mode="state", p=101325, T=[20, "C"], V=1)
    assert rel(a["m"], 101325 / (R_AIR * 293.15)) and rel(a["M_molar"], RU / R_AIR)
    assert rel(a["rho"], a["m"])
    assert err("ideal_gas", mode="state", p=1e5, T=300) == "INVALID_ARGUMENT"
    assert err("ideal_gas", mode="state", p=1e5, T=300, V=1, m=1) == "INVALID_ARGUMENT"


def _work(p_of_V, V1: float, V2: float) -> float:
    return quad(p_of_V, V1, V2, epsabs=0, epsrel=1e-13)[0]


def test_isothermal_and_isentropic() -> None:
    r = e("ideal_gas", mode="process", process="isothermal", p1=1e5, T1=300, V1=0.1, V2=0.2)
    assert rel(r["W"], _work(lambda V: 1e5 * 0.1 / V, 0.1, 0.2), 1e-11)
    assert rel(r["Q"], r["W"]) and abs(r["dU"]) < 1e-9 and rel(r["dS"], r["Q"] / 300, 1e-12)
    s = e("ideal_gas", mode="process", process="isentropic", p1=1e5, T1=300, V1=0.1, p2=[8, "bar"])
    assert rel(s["T2"], 300 * 8 ** (0.4 / 1.4)) and abs(s["Q"]) < 1e-9 and abs(s["dS"]) < 1e-9
    assert rel(s["W"], _work(lambda V: 1e5 * (0.1 / V) ** 1.4, 0.1, s["V2"]), 1e-11)


def test_polytropic_entropy_integral() -> None:
    n, p1, T1, V1, V2 = 1.3, 2e5, 350.0, 0.05, 0.12
    r = e("ideal_gas", mode="process", process="polytropic", n_poly=n, p1=p1, T1=T1, V1=V1, V2=V2)
    m = p1 * V1 / (R_AIR * T1)
    cv = R_AIR / 0.4

    def p(V: float) -> float:
        return p1 * (V1 / V) ** n

    def T(V: float) -> float:
        return p(V) * V / (m * R_AIR)

    # dQ = m cv dT + p dV along the polytropic, with dT/dV = (1 - n) T / V
    dS = quad(
        lambda V: (m * cv * (1 - n) * T(V) / V + p(V)) / T(V), V1, V2, epsabs=0, epsrel=1e-13
    )[0]
    assert rel(r["W"], _work(p, V1, V2), 1e-11) and rel(r["Q"], r["dU"] + r["W"], 1e-12)
    assert rel(r["dS"], dS, 1e-10)


def test_isobaric_isochoric_and_errors() -> None:
    b = e("ideal_gas", mode="process", process="isobaric", p1=1e5, T1=300, m=1, T2=600)
    cp = 1.4 * R_AIR / 0.4
    assert rel(b["Q"], cp * 300) and rel(b["W"], 1e5 * (b["V2"] - b["V1"]))
    assert rel(b["dS"], cp * math.log(2))
    c = e("ideal_gas", mode="process", process="isochoric", p1=1e5, T1=300, m=1, p2=2e5)
    assert c["W"] == 0 and rel(c["Q"], R_AIR / 0.4 * 300)
    assert (
        err("ideal_gas", mode="process", process="isobaric", p1=1e5, T1=300, m=1, p2=2e5)
        == "INVALID_ARGUMENT"
    )


# ------------------------------------------------------------------ thermo_cycle


def test_carnot() -> None:
    r = e("thermo_cycle", cycle="carnot", T_hot=800, T_cold=300)
    assert rel(r["efficiency"], 0.625) and rel(r["COP_refrigerator"], 0.6)
    assert rel(r["COP_heat_pump"], 1.6)


def test_otto_work_integral() -> None:
    g, r_c, T1, p1, q_in = 1.4, 8.0, 300.0, 1e5, 1.8e6
    r = e("thermo_cycle", cycle="otto", r=r_c, T1=T1, p1=p1, q_in=[1800, "kJ/kg"])
    assert rel(r["efficiency"], 1 - r_c ** (1 - g))
    v1 = R_AIR * T1 / p1
    v2 = v1 / r_c
    p2 = p1 * r_c**g
    T3 = T1 * r_c ** (g - 1) + q_in / (R_AIR / (g - 1))
    p3 = R_AIR * T3 / v2
    w = _work(lambda v: p1 * (v1 / v) ** g, v1, v2) + _work(lambda v: p3 * (v2 / v) ** g, v2, v1)
    assert rel(r["w_net"], w, 1e-10) and rel(r["efficiency"], w / q_in, 1e-10)
    assert rel(r["mep"], w / (v1 - v2), 1e-10) and rel(r["p"][1], p2, 1e-12)


def test_diesel_work_integral() -> None:
    g, rr, rc, T1, p1 = 1.4, 18.0, 2.0, 300.0, 1e5
    r = e("thermo_cycle", cycle="diesel", r=rr, rc=rc, T1=T1, p1=p1)
    v1 = R_AIR * T1 / p1
    v2, p2 = v1 / rr, p1 * rr**g
    v3 = rc * v2
    w = (
        _work(lambda v: p1 * (v1 / v) ** g, v1, v2)
        + p2 * (v3 - v2)
        + _work(lambda v: p2 * (v3 / v) ** g, v3, v1)
    )
    T2 = p2 * v2 / R_AIR
    q_in = g * R_AIR / (g - 1) * (rc * T2 - T2)
    assert rel(r["efficiency"], w / q_in, 1e-10) and rel(r["q_in"], q_in, 1e-12)
    closed = 1 - rr ** (1 - g) * (rc**g - 1) / (g * (rc - 1))
    assert rel(r["efficiency"], closed, 1e-12)


def test_brayton_real_balance() -> None:
    g, rp, T1, T3, ec, et = 1.4, 10.0, 300.0, 1400.0, 0.85, 0.9
    r = e("thermo_cycle", cycle="brayton", rp=rp, T1=T1, T3=T3, eta_c=ec, eta_t=et)
    cp = g * R_AIR / (g - 1)
    T2 = T1 + T1 * (rp ** ((g - 1) / g) - 1) / ec
    T4 = T3 - et * (T3 - T3 / rp ** ((g - 1) / g))
    wc, wt = cp * (T2 - T1), cp * (T3 - T4)
    assert rel(r["w_net"], wt - wc, 1e-12) and rel(r["back_work_ratio"], wc / wt, 1e-12)
    assert rel(r["q_in"] - r["q_out"], r["w_net"], 1e-10)
    ideal = e("thermo_cycle", cycle="brayton", rp=rp)
    assert rel(ideal["efficiency"], 1 - rp ** (-(g - 1) / g))


# ------------------------------------------------------------------ calorimetry


def test_sensible_latent() -> None:
    r = e("calorimetry", mode="sensible", m=2, c=[4.186, "kJ/(kg*K)"], T1=[20, "C"], T2=[80, "C"])
    assert rel(r["Q"], 2 * 4186 * 60, 1e-12) and rel(r["dT"], 60, 1e-12)
    assert rel(e("calorimetry", mode="latent", m=0.5, L=[334, "kJ/kg"])["Q"], 167000)


def test_mixture_balance() -> None:
    bodies = [
        {"m": 1, "c": 4186, "T": [80, "C"]},
        {"m": 0.5, "c": 900, "T": [20, "C"]},
        {"m": [300, "g"], "c": 490, "T": 293.15},
    ]
    r = e("calorimetry", mode="mixture", bodies=bodies)
    mc = [(1, 4186, 353.15), (0.5, 900, 293.15), (0.3, 490, 293.15)]
    Teq = brentq(lambda T: sum(m * c * (T - t) for m, c, t in mc), 250, 400, xtol=1e-13)
    assert rel(r["T_eq"], Teq, 1e-12) and rel(r["T_eq_C"], Teq - 273.15, 1e-9)
    assert abs(sum(r["Q"])) < 1e-6
    assert err("calorimetry", mode="mixture", bodies=bodies[:1]) == "INVALID_ARGUMENT"
