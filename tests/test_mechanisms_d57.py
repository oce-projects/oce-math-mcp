"""Linkages and epicyclic gear trains (D57).

Sources: Wikipedia "Four-bar linkage" (Grashof condition and classification in 8 cases, time ratio),
"Piston motion equations" (slider-crank), "Epicyclic gearing" (N_s w_s + N_r w_r =
(N_s + N_r) w_c, differential). Checks independent of the module formulas:
- four-bar position: the distances between the pins give back the link lengths;
- four-bar velocities against the finite difference of the positions;
- classification against the behaviour: a "crank" input closes for all 360 degrees,
  a "rocker" input does not;
- slider-crank: velocity and acceleration against the numerical derivatives of the position; dead
  centres x = l + r and l - r;
- gear train: numerical example of the source (w_r = -0.7), planet speed equal from the two
  meshes (with the sun and with the ring), differential with the example of the source (w_R = -1.0).
"""

from __future__ import annotations

import math

from tests.conftest import call, ok


def link(**params: object) -> dict:
    return ok("eng_calc", operation="linkage", params=params)


def gear(**params: object) -> dict:
    return ok("eng_calc", operation="planetary_gear", params=params)


def bad(operation: str, params: dict) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


CR = {"a": 1.0, "b": 3.5, "g": 4.0, "h": 3.0}  # T1, T2, T3 = +2.5, +3.5, +1.5: crank-rocker


def test_grashof_classification_table() -> None:
    r = link(mode="grashof", **CR)
    assert r["grashof"] is True and r["input"] == "crank" and r["output"] == "rocker"
    assert (r["T1"], r["T2"], r["T3"]) == (2.5, 3.5, 1.5)
    dc = link(
        mode="grashof", a=4.0, b=3.0, g=1.0, h=3.5
    )  # shortest link is the ground: double crank
    assert dc["grashof"] is True and dc["input"] == "crank" and dc["output"] == "crank"
    ng = link(mode="grashof", a=3.0, b=3.0, g=4.0, h=4.5)
    assert ng["grashof"] is False and ng["S_plus_L"] > ng["P_plus_Q"]


def test_four_bar_full_rotation_matches_class() -> None:
    for deg in range(0, 360, 15):
        r = link(mode="four_bar", theta2_deg=float(deg), assembly=1, **CR)
        ax, ay = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        bx, by = r["B"]
        assert abs(math.hypot(bx - ax, by - ay) - CR["h"]) < 1e-12  # coupler
        assert abs(math.hypot(bx - CR["g"], by) - CR["b"]) < 1e-12  # rocker
        assert rel(r["A"][0], ax, 1e-15) or abs(r["A"][0] - ax) < 1e-15
    # rocker as input (Grashof, rocker-crank): there is an angle where it does not close
    rc = {"a": 3.5, "b": 1.0, "g": 4.0, "h": 3.0}
    assert link(mode="grashof", **rc)["input"] == "rocker"
    msg = bad("linkage", {"mode": "four_bar", "theta2_deg": 180.0, "assembly": 1, **rc})
    assert "does not close" in msg


def test_four_bar_velocity_against_finite_difference() -> None:
    th, w2, h = 40.0, 12.0, 1e-6
    r = link(mode="four_bar", theta2_deg=th, omega2=w2, assembly=-1, **CR)
    p = link(mode="four_bar", theta2_deg=th + math.degrees(h), assembly=-1, **CR)
    m = link(mode="four_bar", theta2_deg=th - math.degrees(h), assembly=-1, **CR)
    w3 = (math.radians(p["theta3_deg"]) - math.radians(m["theta3_deg"])) / (2 * h) * w2
    w4 = (math.radians(p["theta4_deg"]) - math.radians(m["theta4_deg"])) / (2 * h) * w2
    assert rel(r["omega3"], w3, 1e-6) and rel(r["omega4"], w4, 1e-6)
    assert 0 < r["transmission_angle_deg"] <= 90


def test_slider_crank_against_numeric_derivatives() -> None:
    r_, l_, w = 0.05, 0.16, 2 * math.pi * 3000 / 60
    for a_deg in (0.0, 35.0, 90.0, 140.0, 180.0, 260.0):
        r = link(mode="slider_crank", r=r_, l=l_, angle_deg=a_deg, rpm=3000.0)
        a = math.radians(a_deg)

        def x(t: float) -> float:
            return r_ * math.cos(t) + math.sqrt(l_**2 - (r_ * math.sin(t)) ** 2)

        h = 1e-4
        assert rel(r["x"], x(a), 1e-14)
        v_num = (x(a + h) - x(a - h)) / (2 * h) * w
        a_num = (x(a + h) - 2 * x(a) + x(a - h)) / h**2 * w * w
        assert abs(r["v"] - v_num) <= 1e-6 * r_ * w, a_deg
        assert abs(r["a"] - a_num) <= 1e-5 * r_ * w * w, a_deg
    assert rel(link(mode="slider_crank", r=r_, l=l_, angle_deg=0.0)["x"], l_ + r_, 1e-15)
    assert rel(link(mode="slider_crank", r=r_, l=l_, angle_deg=180.0)["x"], l_ - r_, 1e-15)
    assert "l > r" in bad(
        "linkage", {"mode": "slider_crank", "r": 0.2, "l": 0.1, "angle_deg": 10.0}
    )


def test_quick_return_ratio() -> None:
    q = link(mode="quick_return", beta_deg=20.0)
    assert rel(q["Q"], 200 / 160, 1e-15)
    b = link(mode="quick_return", Q=1.25)
    assert rel(b["beta_deg"], 20.0, 1e-13)


def test_planetary_source_example_and_planet_speed() -> None:
    r = gear(mode="planetary", N_s=60, N_r=100, omega_s=0.5, omega_c=-0.25)
    assert rel(r["omega_r"], -0.7, 1e-14)
    n_p = r["N_p"]
    assert n_p == 20
    # planet: same result from the external mesh with the sun and the internal mesh with the ring
    w_p = r["omega_p"]
    assert rel(n_p * (w_p - (-0.25)), -60 * (0.5 - (-0.25)), 1e-14)
    assert rel(n_p * (w_p - (-0.25)), 100 * (-0.7 - (-0.25)), 1e-14)
    fixed_ring = gear(mode="planetary", N_s=24, N_r=72, omega_s=1000.0, omega_r=0.0)
    assert rel(fixed_ring["omega_c"], 1000 * 24 / 96, 1e-14) and rel(
        fixed_ring["ratio_s_to_c"], 4.0, 1e-14
    )
    assert "integer" in bad(
        "planetary_gear",
        {"mode": "planetary", "N_s": 25, "N_r": 72, "omega_s": 1.0, "omega_r": 0.0},
    )


def test_differential_source_example() -> None:
    d = gear(mode="differential", omega_L=0.5, omega_c=-0.25)
    assert rel(d["omega_R"], -1.0, 1e-15)
    assert "Two speeds" in bad("planetary_gear", {"mode": "differential", "omega_L": 1.0})
