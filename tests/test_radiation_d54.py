"""Radiation between grey and diffuse surfaces (D54).

Sources: Wikipedia «View factor», «Thermal radiation», «Radiosity (radiometry)».
Checks independent of the module formulas:
- 2D view factors against the integral of the definition. For infinitely long strips the 3D
  kernel cos(t1) cos(t2) / (pi s^2) integrated along the length is cos(f1) cos(f2) / (2 r), with r and f
  measured in the cross-section (integral of r^2 / (r^2 + z^2)^2 over z = pi / (2 r^3)); the test integrates
  this kernel with dblquad;
- consistency between different formulas of the source: perpendicular plates of equal width = plates
  inclined at 90 degrees; equilateral three-sided enclosure = 1/2;
- two surfaces: formula of the «Thermal radiation» page against the radiosity system of N
  surfaces of the «Radiosity» page; black bodies: sigma A1 F12 (T1^4 - T2^4);
- three surfaces with one reradiating (Q = 0): equivalent circuit of the source solved in the test with
  resistances in series and in parallel; energy conservation (sum of the Q = 0).
"""

from __future__ import annotations

import math

from scipy.integrate import dblquad

from tests.conftest import call, ok

SIGMA = 5.670374419e-8


def rad(**params: object) -> dict:
    return ok("eng_calc", operation="radiation_exchange", params=params)


def full(**params: object) -> dict:
    resp = call("eng_calc", operation="radiation_exchange", params=params)
    assert resp["ok"], resp
    return resp


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="radiation_exchange", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def f2d(
    p1: tuple, d1: tuple, n1: tuple, w1: float, p2: tuple, d2: tuple, n2: tuple, w2: float
) -> float:
    """2D view factor from one strip to the other: points p + s d, normals n towards the other strip."""

    def kern(s2: float, s1: float) -> float:
        x1, y1 = p1[0] + s1 * d1[0], p1[1] + s1 * d1[1]
        x2, y2 = p2[0] + s2 * d2[0], p2[1] + s2 * d2[1]
        rx, ry = x2 - x1, y2 - y1
        r = math.hypot(rx, ry)
        c1 = (n1[0] * rx + n1[1] * ry) / r
        c2 = -(n2[0] * rx + n2[1] * ry) / r
        return c1 * c2 / (2 * r)

    return dblquad(kern, 0, w1, 0, w2, epsabs=1e-12, epsrel=1e-10)[0] / w1


def test_view_factor_parallel_plates_against_integral() -> None:
    for wi, wj, L in ((1.0, 1.0, 0.5), (0.4, 1.5, 1.0), (2.0, 0.3, 0.7)):
        r = rad(mode="view_factor_2d", geometry="parallel_plates", w_i=wi, w_j=wj, L=L)
        # plate i on the x axis centred, plate j at height L centred, normals facing each other
        ref = f2d((-wi / 2, 0), (1, 0), (0, 1), wi, (-wj / 2, L), (1, 0), (0, -1), wj)
        assert rel(r["F_ij"], ref, 1e-8), (wi, wj, L)
        assert rel(r["F_ji"], r["F_ij"] * wi / wj, 1e-12)


def test_view_factor_perpendicular_and_inclined_plates() -> None:
    for wi, wj in ((1.0, 1.0), (0.5, 2.0), (3.0, 1.0)):
        r = rad(mode="view_factor_2d", geometry="perpendicular_plates", w_i=wi, w_j=wj)
        # i on the x axis from 0 to wi (normal upwards), j on the y axis from 0 to wj (normal towards +x)
        ref = f2d((0, 0), (1, 0), (0, 1), wi, (0, 0), (0, 1), (1, 0), wj)
        assert rel(r["F_ij"], ref, 1e-7), (wi, wj)
    perp = rad(mode="view_factor_2d", geometry="perpendicular_plates", w_i=1.0, w_j=1.0)["F_ij"]
    incl = rad(mode="view_factor_2d", geometry="inclined_plates", w=1.0, alpha_deg=90.0)["F_ij"]
    assert rel(perp, incl, 1e-14) and rel(perp, 1 - math.sqrt(0.5), 1e-14)
    a = 60.0
    r = rad(mode="view_factor_2d", geometry="inclined_plates", w=1.0, alpha_deg=a)
    t = math.radians(a)
    ref = f2d(
        (0, 0),
        (1, 0),
        (0, 1),
        1.0,
        (0, 0),
        (math.cos(t), math.sin(t)),
        (math.sin(t), -math.cos(t)),
        1.0,
    )
    assert rel(r["F_ij"], ref, 1e-7)


def test_view_factor_three_sided() -> None:
    assert rel(
        rad(mode="view_factor_2d", geometry="three_sided", w_i=1.0, w_j=1.0, w_k=1.0)["F_ij"],
        0.5,
        1e-15,
    )
    r = rad(mode="view_factor_2d", geometry="three_sided", w_i=3.0, w_j=4.0, w_k=5.0)
    # right triangle 3-4-5: i and j perpendicular with a common edge
    perp = rad(mode="view_factor_2d", geometry="perpendicular_plates", w_i=3.0, w_j=4.0)["F_ij"]
    assert rel(r["F_ij"], perp, 1e-14)
    assert "triangle" in bad(
        {"mode": "view_factor_2d", "geometry": "three_sided", "w_i": 1.0, "w_j": 1.0, "w_k": 3.0}
    )


def test_two_surface_against_radiosity_system() -> None:
    p = {"T1": 800.0, "T2": 400.0, "eps1": 0.6, "eps2": 0.3}
    two = rad(mode="two_surface", config="general", A1=2.0, A2=5.0, F12=0.7, **p)
    # 1 and 2 alone form the enclosure: F11 = 0.3 (1 concave), F21 = A1 F12 / A2 = 0.28, F22 = 0.72
    enc = rad(
        mode="enclosure",
        A=[2.0, 5.0],
        F=[[0.3, 0.7], [0.28, 0.72]],
        eps=[0.6, 0.3],
        T=[800.0, 400.0],
    )
    assert rel(enc["Q"][0], two["Q"], 1e-12) and rel(enc["Q"][1], -two["Q"], 1e-12)
    den = (1 - 0.6) / (2 * 0.6) + 1 / (2 * 0.7) + (1 - 0.3) / (5 * 0.3)
    assert rel(two["Q"], SIGMA * (800.0**4 - 400.0**4) / den, 1e-14)


def test_black_bodies_and_configs() -> None:
    r = rad(
        mode="two_surface",
        config="general",
        A1=1.5,
        A2=3.0,
        F12=0.4,
        T1=600.0,
        T2=300.0,
        eps1=1.0,
        eps2=1.0,
    )
    assert rel(r["Q"], SIGMA * 1.5 * 0.4 * (600.0**4 - 300.0**4), 1e-14)
    plates = rad(
        mode="two_surface", config="parallel_plates", A=1.0, T1=500.0, T2=300.0, eps1=0.8, eps2=0.5
    )
    assert rel(plates["Q"], SIGMA * (500.0**4 - 300.0**4) / (1 / 0.8 + 1 / 0.5 - 1), 1e-14)
    inner = rad(
        mode="two_surface",
        config="enclosed",
        A1=0.1,
        A2=1e9,
        T1=500.0,
        T2=300.0,
        eps1=0.7,
        eps2=0.5,
    )
    # small object in large surroundings: tends to eps1 sigma A1 (T1^4 - T2^4), like heat_transfer radiation
    small = ok(
        "eng_calc",
        operation="heat_transfer",
        params={"mode": "radiation", "emissivity": 0.7, "A": 0.1, "T1": 500.0, "T2": 300.0},
    )
    assert rel(inner["Q"], small["Q"], 1e-8)


def test_radiation_shields() -> None:
    base = {
        "mode": "two_surface",
        "config": "parallel_plates",
        "A": 1.0,
        "T1": 600.0,
        "T2": 300.0,
        "eps1": 0.8,
        "eps2": 0.8,
    }
    q0 = rad(**base)["Q"]
    r = rad(**base, n_shields=1, eps_shield=0.8)
    assert rel(r["Q"], q0 / 2, 1e-14)  # shield equal to the plates: double resistance
    # shield temperature from the balance: the same Q between plate 1 and the shield
    ts = r["T_shields"][0]
    assert rel(r["Q"], SIGMA * (600.0**4 - ts**4) / (1 / 0.8 + 1 / 0.8 - 1), 1e-12)
    r3 = rad(**base, n_shields=3, eps_shield=0.05)
    ts3 = r3["T_shields"]
    assert 600 > ts3[0] > ts3[1] > ts3[2] > 300
    # resistances in series per unit area: plate-shield, 2 shield-shield, shield-plate
    r_ps = 1 / 0.8 + 1 / 0.05 - 1
    r_ss = 2 / 0.05 - 1
    assert rel(r3["Q"], SIGMA * (600.0**4 - 300.0**4) / (2 * r_ps + 2 * r_ss), 1e-12)


def test_three_surfaces_reradiating_against_circuit() -> None:
    # 1 hot, 2 cold, 3 reradiating (insulated, Q = 0): circuit of the source
    A = [1.0, 1.0, 2.0]
    F = [[0.0, 0.25, 0.75], [0.25, 0.0, 0.75], [0.375, 0.375, 0.25]]
    eps = [0.8, 0.6, 0.5]
    resp = full(mode="enclosure", A=A, F=F, eps=eps, T=[900.0, 400.0, None], Q=[None, None, 0.0])
    r = resp["result"]
    r1 = (1 - 0.8) / (1.0 * 0.8)
    r2 = (1 - 0.6) / (1.0 * 0.6)
    r12 = 1 / (1.0 * 0.25)
    r13 = 1 / (1.0 * 0.75)
    r23 = 1 / (1.0 * 0.75)
    par = 1 / (1 / r12 + 1 / (r13 + r23))
    q = SIGMA * (900.0**4 - 400.0**4) / (r1 + par + r2)
    assert rel(r["Q"][0], q, 1e-12) and rel(r["Q"][1], -q, 1e-12) and abs(r["Q"][2]) < 1e-9 * q
    # the reradiating surface has J = sigma T^4 (no net flux)
    assert rel(SIGMA * r["T"][2] ** 4, r["J"][2], 1e-12)
    assert abs(sum(r["Q"])) < 1e-9 * q


def test_enclosure_checks() -> None:
    A = [1.0, 1.0]
    assert "sum" in bad(
        {
            "mode": "enclosure",
            "A": A,
            "F": [[0.0, 0.9], [1.0, 0.0]],
            "eps": [0.5, 0.5],
            "T": [400.0, 300.0],
        }
    )
    assert "Reciprocity" in bad(
        {
            "mode": "enclosure",
            "A": [1.0, 2.0],
            "F": [[0.0, 1.0], [0.6, 0.4]],
            "eps": [0.5, 0.5],
            "T": [400.0, 300.0],
        }
    )
    assert "temperature" in bad(
        {
            "mode": "enclosure",
            "A": A,
            "F": [[0.0, 1.0], [1.0, 0.0]],
            "eps": [0.5, 0.5],
            "T": [None, None],
            "Q": [10.0, -10.0],
        }
    )
    assert "exactly one" in bad(
        {
            "mode": "enclosure",
            "A": A,
            "F": [[0.0, 1.0], [1.0, 0.0]],
            "eps": [0.5, 0.5],
            "T": [400.0, 300.0],
            "Q": [1.0, None],
        }
    )
    assert "eps" in bad(
        {
            "mode": "two_surface",
            "config": "parallel_plates",
            "A": 1.0,
            "T1": 400.0,
            "T2": 300.0,
            "eps1": 1.2,
            "eps2": 0.5,
        }
    )
