"""Electrostatics and capacitance (D55).

Sources: Wikipedia «Coulomb's law» and «Capacitance» (table of the capacitances of simple systems).
Checks independent of the module formulas:
- Coulomb constant: k = mu_0 c^2 / (4 pi) (identity of the source) with the CODATA constants of scipy;
- field of several charges against the numerical gradient (central differences) of the potential;
- capacitance of concentric cylinders and spheres from the potential integrated with quad (V = int E dr);
- wire in front of a wall against two parallel wires with the method of images (C_wall(d) =
  2 C_pair(2d)), eccentric cylinders with d = 0 against the coaxial ones, isolated sphere as the limit of the
  concentric spheres; energy of the parallel-plate capacitor against the density eps E^2 / 2 times the volume.
"""

from __future__ import annotations

import math

import scipy.constants as sc
from scipy.integrate import quad

from tests.conftest import call, ok

EPS0 = sc.epsilon_0
K = sc.mu_0 * sc.c**2 / (4 * math.pi)


def es(**params: object) -> dict:
    return ok("eng_calc", operation="electrostatics", params=params)


def full(**params: object) -> dict:
    resp = call("eng_calc", operation="electrostatics", params=params)
    assert resp["ok"], resp
    return resp


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="electrostatics", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def test_coulomb_constant_and_force() -> None:
    r = es(mode="coulomb", q1=[2, "uC"], q2=[-3, "nC"], r=[5, "cm"])
    assert rel(r["k_e"], K, 1e-9)
    assert rel(r["F"], K * 2e-6 * -3e-9 / 0.05**2, 1e-9) and r["F"] < 0  # negative = attraction
    assert rel(r["E1"], K * 2e-6 / 0.05**2, 1e-9)
    assert rel(r["V1"], K * 2e-6 / 0.05, 1e-9)
    assert rel(r["U"], K * 2e-6 * -3e-9 / 0.05, 1e-9)
    water = es(mode="coulomb", q1=1e-6, q2=1e-6, r=0.1, eps_r=80.0)
    vacuum = es(mode="coulomb", q1=1e-6, q2=1e-6, r=0.1)
    assert rel(water["F"], vacuum["F"] / 80, 1e-12)


def test_point_charges_field_is_minus_gradient() -> None:
    charges = [[1e-9, 0.0, 0.0, 0.0], [-2e-9, 0.1, 0.05, 0.0], [0.5e-9, -0.05, 0.1, 0.08]]
    pt = [0.03, -0.02, 0.04]
    r = es(mode="point_charges", charges=charges, point=pt)
    h = 1e-6
    for axis in range(3):
        a, b = list(pt), list(pt)
        a[axis] += h
        b[axis] -= h
        va = es(mode="point_charges", charges=charges, point=a)["V"]
        vb = es(mode="point_charges", charges=charges, point=b)["V"]
        assert rel(r["E"][axis], -(va - vb) / (2 * h), 1e-6), axis
    assert rel(r["E_mag"], math.sqrt(sum(x * x for x in r["E"])), 1e-12)
    # two charges: the force on the test charge matches Coulomb's law
    two = es(mode="point_charges", charges=[[3e-9, 0.0, 0.0]], point=[0.2, 0.0], q=-1e-9)
    coul = es(mode="coulomb", q1=3e-9, q2=-1e-9, r=0.2)
    assert rel(two["F"][0], coul["F"], 1e-12) and two["F"][1] == 0
    # energy of the system: sum over the pairs
    u = sum(
        K * charges[i][0] * charges[j][0] / math.dist(charges[i][1:], charges[j][1:])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    assert rel(r["U_system"], u, 1e-9)


def test_capacitance_from_integrated_potential() -> None:
    lam, L = 1e-9, 2.0
    ri, ro = 0.001, 0.0035
    v = quad(lambda x: lam / (2 * math.pi * EPS0 * 2.25 * x), ri, ro)[0]
    coax = es(mode="capacitor", geometry="coaxial", R_inner=ri, R_outer=ro, L=L, eps_r=2.25)
    assert rel(coax["C"], lam * L / v, 1e-10)
    q = 1e-10
    vs = quad(lambda x: q / (4 * math.pi * EPS0 * x * x), 0.05, 0.08)[0]
    sph = es(mode="capacitor", geometry="concentric_spheres", R_inner=0.05, R_outer=0.08)
    assert rel(sph["C"], q / vs, 1e-10)
    iso = es(mode="capacitor", geometry="sphere", R=0.05)
    far = es(mode="capacitor", geometry="concentric_spheres", R_inner=0.05, R_outer=5e7)
    assert rel(iso["C"], far["C"], 1e-8) and rel(iso["C"], 4 * math.pi * EPS0 * 0.05, 1e-12)


def test_capacitance_images_and_limits() -> None:
    a, d, L = 0.001, 0.02, 3.0
    wall = es(mode="capacitor", geometry="wire_wall", a=a, d=d, L=L)
    pair = es(mode="capacitor", geometry="parallel_wires", a=a, d=2 * d, L=L)
    assert rel(wall["C"], 2 * pair["C"], 1e-12)  # method of images
    ecc0 = es(
        mode="capacitor", geometry="eccentric_cylinders", R_inner=0.002, R_outer=0.006, d=0.0, L=1.0
    )
    coax = es(mode="capacitor", geometry="coaxial", R_inner=0.002, R_outer=0.006, L=1.0)
    assert rel(ecc0["C"], coax["C"], 1e-12)
    ecc = es(
        mode="capacitor",
        geometry="eccentric_cylinders",
        R_inner=0.002,
        R_outer=0.006,
        d=0.002,
        L=1.0,
    )
    assert ecc["C"] > coax["C"]  # the inner conductor closer to the wall increases C
    disc = es(mode="capacitor", geometry="disc", R=0.1)
    assert rel(disc["C"], 8 * EPS0 * 0.1, 1e-12)


def test_parallel_plate_charge_energy() -> None:
    r = es(
        mode="capacitor", geometry="parallel_plate", A=[100, "cm2"], d=[1, "mm"], eps_r=4.0, V=50.0
    )
    c = 4.0 * EPS0 * 0.01 / 0.001
    assert rel(r["C"], c, 1e-12) and rel(r["Q"], c * 50, 1e-12)
    assert rel(r["E"], 50 / 0.001, 1e-12)
    density = 0.5 * 4.0 * EPS0 * r["E"] ** 2
    assert rel(r["U"], density * 0.01 * 0.001, 1e-12)
    byq = es(mode="capacitor", geometry="parallel_plate", A=0.01, d=0.001, eps_r=4.0, Q=r["Q"])
    assert rel(byq["V"], 50.0, 1e-12)


def test_combination_and_errors() -> None:
    s = es(mode="combination", values=[[10, "uF"], [10, "uF"]], connection="series")
    p = es(mode="combination", values=[1e-6, 2.2e-6, 4.7e-9], connection="parallel")
    assert rel(s["C_eq"], 5e-6, 1e-12) and rel(p["C_eq"], 1e-6 + 2.2e-6 + 4.7e-9, 1e-12)
    assert "R_outer" in bad(
        {"mode": "capacitor", "geometry": "coaxial", "R_inner": 0.01, "R_outer": 0.005, "L": 1.0}
    )
    assert "2a" in bad(
        {"mode": "capacitor", "geometry": "parallel_wires", "a": 0.01, "d": 0.015, "L": 1.0}
    )
    assert "only one" in bad(
        {"mode": "capacitor", "geometry": "sphere", "R": 0.1, "V": 1.0, "Q": 1e-9}
    )
    assert "coincide" in bad(
        {"mode": "point_charges", "charges": [[1e-9, 0.0, 0.0]], "point": [0.0, 0.0]}
    )
    # d + R_inner >= R_outer: the inner conductor touches or leaves the outer one
    assert "touches" in bad(
        {
            "mode": "capacitor",
            "geometry": "eccentric_cylinders",
            "R_inner": 0.002,
            "R_outer": 0.006,
            "d": 0.0045,
            "L": 1.0,
        }
    )
