"""eng_calc area B (D39): mechanics, vibrations, transmissions, CNC. Independent methods in the tests.

Projectile and vibrations: numerical integration of the equations of motion (SciPy solve_ivp).
Belt: length built from the tangent points. Gears: contact segment from the intersections of the
line of action with the addendum circles.
"""

from __future__ import annotations

import math

import numpy as np
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


# ------------------------------------------------------------------ kinematics_dynamics


@pytest.mark.parametrize(
    ("v0", "ang", "h0"), [(50.0, 45.0, 0.0), (20.0, 30.0, 12.0), (15.0, -10.0, 40.0)]
)
def test_projectile_against_integration_with_event(v0: float, ang: float, h0: float) -> None:
    g = 9.80665
    r = e("kinematics_dynamics", mode="projectile", v0=v0, angle_deg=ang, h0=h0)
    th = math.radians(ang)

    def ground(_t: float, y: list[float]) -> float:
        return y[1]

    ground.terminal = True  # type: ignore[attr-defined]
    ground.direction = -1  # type: ignore[attr-defined]
    sol = solve_ivp(
        lambda _t, y: [y[2], y[3], 0.0, -g],
        (0, 100),
        [0.0, h0, v0 * math.cos(th), v0 * math.sin(th)],
        events=ground,
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
    )
    t_hit = float(sol.t_events[0][0])
    x_hit, _, vx, vy = sol.y_events[0][0]
    assert rel(r["t_flight"], t_hit, 1e-8) and rel(r["range"], float(x_hit), 1e-8)
    assert rel(r["v_impact"], math.hypot(vx, vy), 1e-8)
    heights = sol.sol(np.linspace(0, t_hit, 200001))[1]
    assert rel(r["h_max"], max(h0, float(np.max(heights))), 1e-8)


def test_projectile_classic_and_custom_g() -> None:
    r = e("kinematics_dynamics", mode="projectile", v0=50, angle_deg=45, g=9.81)
    assert rel(r["range"], 50**2 / 9.81) and rel(r["h_max"], 50**2 / (4 * 9.81))


def test_uniform_acceleration_any_two_unknowns() -> None:
    r = e("kinematics_dynamics", mode="uniform_acceleration", v0=[36, "km/h"], a=2, t=5)
    assert rel(r["v"], 20.0) and rel(r["s"], 75.0)
    r = e("kinematics_dynamics", mode="uniform_acceleration", v0=0, v=30, s=90)
    assert rel(r["a"], 5.0) and rel(r["t"], 6.0)
    r = e("kinematics_dynamics", mode="uniform_acceleration", v0=10, a=-2, s=16)
    assert [round(x, 9) for x in r["t"]] == [
        2.0,
        8.0,
    ]  # two instants, the second after the reversal


def test_energy_rotation_steiner_circular() -> None:
    r = e(
        "kinematics_dynamics",
        mode="energy",
        m=2,
        v=3,
        h=10,
        k=[5, "kN/m"],
        x=[20, "mm"],
        I_m=0.5,
        omega=4,
    )
    assert rel(r["E_kin"], 9.0) and rel(r["E_pot"], 2 * 9.80665 * 10)
    assert rel(r["E_spring"], 0.5 * 5000 * 0.02**2) and rel(r["E_rot"], 0.5 * 0.5 * 16)
    assert rel(r["E_total"], r["E_kin"] + r["E_pot"] + r["E_spring"] + r["E_rot"])
    r = e("kinematics_dynamics", mode="rotation", T=[100, "Nm"], omega=[3000, "rpm"])
    assert rel(r["P"], 100 * 3000 * 2 * math.pi / 60)
    r = e("kinematics_dynamics", mode="rotation", P=[7.5, "kW"], omega=[1450, "rpm"])
    assert rel(r["T"], 7500 / (1450 * 2 * math.pi / 60))
    r = e("kinematics_dynamics", mode="steiner", I_m=0.2, m=4, d=0.5)
    assert rel(r["I_axis"], 0.2 + 4 * 0.25)
    r = e("kinematics_dynamics", mode="circular", m=1200, v=[72, "km/h"], r=50)
    assert rel(r["a_c"], 8.0) and rel(r["F_c"], 9600.0) and rel(r["omega"], 0.4)


def test_kinematics_bad_inputs() -> None:
    assert err("kinematics_dynamics", mode="teleport") == "INVALID_ARGUMENT"
    assert err("kinematics_dynamics", mode="uniform_acceleration", v0=1) == "INVALID_ARGUMENT"
    assert err("kinematics_dynamics", mode="rotation", T=1, P=1, omega=1) == "INVALID_ARGUMENT"
    assert err("kinematics_dynamics", mode="energy") == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ vibration_1dof


def test_vibration_free_decay_against_integration() -> None:
    m, k, zeta = 2.0, 800.0, 0.05
    r = e("vibration_1dof", m=m, k=k, zeta=zeta)
    c = r["c"]
    sol = solve_ivp(
        lambda _t, y: [y[1], -(c * y[1] + k * y[0]) / m],
        (0, 5),
        [1.0, 0.0],
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
    )
    t = np.linspace(0, 5, 200001)
    x = sol.sol(t)[0]
    peaks = [i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1]]
    assert rel(t[peaks[1]] - t[peaks[0]], r["T_d"], 1e-4)
    assert rel(math.log(x[peaks[0]] / x[peaks[1]]), r["log_decrement"], 1e-4)
    assert rel(r["omega_n"], 20.0) and rel(r["f_n"], 20 / (2 * math.pi)) and rel(r["Q"], 10.0)


def test_vibration_forced_steady_state_against_integration() -> None:
    m, k, c, F0, w = 1.0, 400.0, 4.0, 10.0, 15.0
    r = e("vibration_1dof", m=m, k=k, c=c, F0=F0, omega=w)
    sol = solve_ivp(
        lambda tt, y: [y[1], (F0 * math.cos(w * tt) - c * y[1] - k * y[0]) / m],
        (0, 40),
        [0.0, 0.0],
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
    t = np.linspace(30, 40, 100001)
    x = sol.sol(t)[0]
    v = sol.sol(t)[1]
    assert rel(r["X"], float(np.max(np.abs(x))), 1e-5)
    ft = np.abs(k * x + c * v)
    assert rel(r["F_T"], float(np.max(ft)), 1e-4)
    assert rel(r["zeta"], c / (2 * math.sqrt(k * m)))


def test_vibration_inputs() -> None:
    assert err("vibration_1dof", m=1, k=1) == "INVALID_ARGUMENT"  # damping missing
    assert err("vibration_1dof", m=1, k=1, c=1, zeta=0.1) == "INVALID_ARGUMENT"
    r = e("vibration_1dof", m=1, k=100, zeta=1.5)
    assert r["omega_d"] is None and "regime" in r and r["regime"] == "overdamped"


# ------------------------------------------------------------------ bearing_life


def test_bearing_life_ball_roller_and_required_c() -> None:
    r = e("bearing_life", C=[30, "kN"], P=[3, "kN"], type="ball", n=[1500, "rpm"])
    assert rel(r["L10"], 1e9) and rel(r["L10h"], 1e9 / (1500 * 60))
    r2 = e("bearing_life", C=30000, P=3000, type="roller")
    assert rel(r2["L10"], 10 ** (10 / 3) * 1e6)
    r3 = e("bearing_life", P=3000, type="ball", n=[1500, "rpm"], L10h_req=[20000, "h"])
    back = e("bearing_life", C=r3["C_required"], P=3000, type="ball", n=[1500, "rpm"])
    assert rel(back["L10h"], 20000, 1e-12)
    r4 = e("bearing_life", C=30000, Fr=2000, Fa=1000, X=0.56, Y=1.8, type="ball")
    assert rel(r4["P"], 0.56 * 2000 + 1.8 * 1000)
    assert err("bearing_life", C=1, P=1) == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ belt_drive


def belt_length_by_construction(D: float, d: float, C: float) -> float:
    """Tangent points of the outer lines built with vectors, then arcs and straight spans."""
    R, r = D / 2, d / 2
    alpha = math.asin((R - r) / C)
    n = np.array([-math.sin(alpha), math.cos(alpha)])  # normal of the upper tangent line
    p_small, p_big = np.array([0.0, 0.0]) + r * n, np.array([C, 0.0]) + R * n
    straight = 2 * float(np.linalg.norm(p_big - p_small))
    return straight + r * (math.pi - 2 * alpha) + R * (math.pi + 2 * alpha)


def test_belt_geometry_and_tensions() -> None:
    r = e(
        "belt_drive",
        D=[400, "mm"],
        d=[150, "mm"],
        C=[800, "mm"],
        n=[1450, "rpm"],
        P=[11, "kW"],
        mu=0.3,
    )
    assert rel(r["L"], belt_length_by_construction(0.4, 0.15, 0.8), 1e-12)
    v = math.pi * 0.15 * 1450 / 60
    assert rel(r["v"], v) and rel(r["F_eff"], 11000 / v)
    assert rel(r["T1"] - r["T2"], r["F_eff"]) and rel(
        r["T1"] / r["T2"], math.exp(0.3 * r["theta_small"])
    )
    assert rel(r["ratio"], 400 / 150) and rel(r["n_large"], 1450 * 150 / 400 * 2 * math.pi / 60)


def test_belt_v_groove_and_centrifugal() -> None:
    flat = e("belt_drive", D=0.3, d=0.1, C=0.6, n=[1000, "rpm"], P=2000, mu=0.25)
    vee = e("belt_drive", D=0.3, d=0.1, C=0.6, n=[1000, "rpm"], P=2000, mu=0.25, groove_deg=38)
    mu_eff = 0.25 / math.sin(math.radians(19))
    assert rel(vee["T1"] / vee["T2"], math.exp(mu_eff * vee["theta_small"]))
    assert vee["T1"] < flat["T1"]
    cf = e("belt_drive", D=0.3, d=0.1, C=0.6, n=[1000, "rpm"], P=2000, mu=0.25, m_l=0.2)
    assert rel(cf["F_centrifugal"], 0.2 * cf["v"] ** 2) and rel(
        cf["T1"] - flat["T1"], cf["F_centrifugal"]
    )
    assert err("belt_drive", D=0.1, d=0.3, C=0.6) == "INVALID_ARGUMENT"
    assert err("belt_drive", D=0.3, d=0.1, C=0.1) == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ cnc_machining


def test_cnc_turning_milling_drilling() -> None:
    r = e(
        "cnc_machining",
        mode="turning",
        Vc=[200, "m/min"],
        D=[50, "mm"],
        f=[0.2, "mm"],
        ap=[2, "mm"],
        kc=[2000, "MPa"],
        eta=0.8,
    )
    assert rel(r["n_rpm"], 200000 / (math.pi * 50))
    assert rel(r["MRR"], (200 / 60) * 0.2e-3 * 2e-3) and rel(r["P_cut"], 2000e6 * r["MRR"])
    assert rel(r["P_motor"], r["P_cut"] / 0.8) and rel(r["F_c"], 2000e6 * 0.2e-3 * 2e-3)
    m = e(
        "cnc_machining",
        mode="milling",
        n=[8000, "rpm"],
        D=[10, "mm"],
        z=4,
        fz=[0.05, "mm"],
        ap=[5, "mm"],
        ae=[2, "mm"],
    )
    assert rel(m["Vf_mm_min"], 0.05 * 4 * 8000) and rel(m["Vc_m_min"], math.pi * 10 * 8000 / 1000)
    assert rel(m["MRR_cm3_min"], 5 * 2 * 0.05 * 4 * 8000 / 1000)
    d = e("cnc_machining", mode="drilling", Vc=[30, "m/min"], D=[8, "mm"], f=[0.1, "mm"])
    assert rel(d["MRR"], math.pi * 0.008**2 / 4 * d["Vf"])
    assert (
        err("cnc_machining", mode="milling", Vc=100, D=0.01, fz=0.0001, ap=0.001, ae=0.001)
        == "INVALID_ARGUMENT"
    )


# ------------------------------------------------------------------ gear_spur


def contact_ratio_by_geometry(m: float, z1: int, z2: int, alpha: float) -> float:
    """Line of action intersected with the addendum circles (numpy), divided by the base pitch."""
    r1, r2 = m * z1 / 2, m * z2 / 2
    O1, O2 = np.array([0.0, 0.0]), np.array([r1 + r2, 0.0])
    P = np.array([r1, 0.0])
    u = np.array([math.sin(alpha), math.cos(alpha)])  # direction of the line of action

    def hits(center: np.ndarray, ra: float) -> list[float]:
        w = P - center
        b, c = float(w @ u), float(w @ w) - ra * ra
        s = math.sqrt(b * b - c)
        return [-b - s, -b + s]

    a1 = hits(O1, r1 + m)
    a2 = hits(O2, r2 + m)
    start, end = max(min(a1), min(a2)), min(max(a1), max(a2))
    return (end - start) / (math.pi * m * math.cos(alpha))


def test_gear_geometry_forces_and_contact_ratio() -> None:
    r = e("gear_spur", m=[3, "mm"], z1=20, z2=50, T1=[60, "Nm"], n1=[1500, "rpm"])
    assert rel(r["d1"], 0.06) and rel(r["d2"], 0.15) and rel(r["a"], 0.105)
    assert rel(r["da1"], 0.066) and rel(r["df1"], 0.06 - 2.5 * 0.003)
    assert rel(r["Ft"], 2 * 60 / 0.06) and rel(r["Fr"], r["Ft"] * math.tan(math.radians(20)))
    assert rel(r["ratio"], 2.5) and rel(r["T2"], 150.0) and rel(r["n2_rpm"], 600.0)
    assert rel(
        r["contact_ratio"], contact_ratio_by_geometry(0.003, 20, 50, math.radians(20)), 1e-12
    )


def test_gear_helical_and_lewis() -> None:
    r = e("gear_spur", m=[2, "mm"], z1=25, z2=75, beta_deg=15, T1=40, b=[20, "mm"], Y=0.34)
    mt = 0.002 / math.cos(math.radians(15))
    assert rel(r["d1"], mt * 25) and rel(r["Fa"], r["Ft"] * math.tan(math.radians(15)))
    at = math.atan(math.tan(math.radians(20)) / math.cos(math.radians(15)))
    assert rel(r["Fr"], r["Ft"] * math.tan(at))
    assert rel(r["sigma_lewis"], r["Ft"] / (0.02 * 0.002 * 0.34))
    assert err("gear_spur", m=0.002, z1=20.5, z2=40) == "INVALID_ARGUMENT"
