"""Nusselt correlations for the convection coefficient (D52).

Sources: Wikipedia «Nusselt number», «Churchill–Bernstein equation», «Heat transfer coefficient»,
«Reynolds number». Checks independent of the module formulas:
- laminar plate: the mean is the numerical integral (quad) of the local coefficient along the plate;
- pure conduction limits: sphere Nu -> 2 for Re -> 0 (Faeth) and Ra -> 0 (Yuge);
- laminar pipe: 3.66 against the exact value 3.6567934577632923619 (OEIS A282581, cited by the source);
- Gnielinski at Pr = 1 against the Reynolds analogy; simplified forms of Gnielinski within 15%
  (measured 13.07%, not «very close»);
- Churchill-Chu: laminar form below Ra = 1e9 (at Ra = 4.4e8 the general form gives 27% more);
  Sieder-Tate: viscosity ratio of the example
  of the source (water at 20 °C with the wall at 40 °C and at 100 °C: 1.45 and 3.57);
- Grashof and Rayleigh rebuilt in the test from the definitions; Q = h A dT.
"""

from __future__ import annotations

import math

from scipy.integrate import quad

from tests.conftest import call, ok


def e(**params: object) -> dict:
    return ok("eng_calc", operation="convection_correlation", params=params)


def full(**params: object) -> dict:
    resp = call("eng_calc", operation="convection_correlation", params=params)
    assert resp["ok"], resp
    return resp


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="convection_correlation", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


AIR = {"k": 0.0263, "Pr": 0.707, "nu": 1.589e-5}  # test properties, not from a table


def test_flat_plate_average_is_integral_of_local() -> None:
    v, L = 5.0, 0.8
    r = e(geometry="flat_plate", v=v, L=L, **AIR)
    assert rel(r["Re"], v * L / AIR["nu"], 1e-12)

    def hx(x: float) -> float:
        return 0.332 * (v * x / AIR["nu"]) ** 0.5 * AIR["Pr"] ** (1 / 3) * AIR["k"] / x

    avg = quad(hx, 0, L, limit=200)[0] / L
    assert rel(r["h"], avg, 1e-8)
    assert rel(r["h_local_end"], hx(L), 1e-12)
    assert rel(r["h"], r["Nu"] * AIR["k"] / L, 1e-12)
    # mu and rho instead of nu
    r2 = e(geometry="flat_plate", v=v, L=L, k=AIR["k"], Pr=AIR["Pr"], mu=1.589e-5 * 1.1, rho=1.1)
    assert rel(r2["h"], r["h"], 1e-12)


def test_flat_plate_turbulent_warning_and_heat() -> None:
    resp = full(geometry="flat_plate", v=30.0, L=1.0, T_s=[80, "C"], T_inf=[20, "C"], A=2.0, **AIR)
    r = resp["result"]
    assert any("5e5" in w for w in resp["warnings"])
    assert rel(r["Q"], r["h"] * 2.0 * 60.0, 1e-12) and rel(r["q_flux"], r["h"] * 60.0, 1e-12)


def test_sphere_and_conduction_limits() -> None:
    assert (
        abs(e(geometry="sphere_forced", Re=1e-10, D=0.01, **{"k": 0.6, "Pr": 7.0})["Nu"] - 2) < 1e-4
    )
    assert abs(e(geometry="sphere_natural", Ra=1e-12, D=0.01, k=0.026, Pr=0.7)["Nu"] - 2) < 1e-2
    r = e(geometry="sphere_natural", Ra=1e4, D=0.05, k=0.026, Pr=0.7)
    assert rel(r["Nu"], 2 + 0.43 * 1e4**0.25, 1e-12)


def test_cylinder_crossflow_churchill_bernstein() -> None:
    r = e(geometry="cylinder_crossflow", Re=1e4, D=0.02, k=0.0263, Pr=0.707)
    pr = 0.707
    ref = (
        0.3
        + 0.62
        * 1e4**0.5
        * pr ** (1 / 3)
        / (1 + (0.4 / pr) ** (2 / 3)) ** 0.25
        * (1 + (1e4 / 282000) ** 0.625) ** 0.8
    )
    assert rel(r["Nu"], ref, 1e-12)
    # Re -> 0: the constant 0.3 remains
    assert (
        abs(e(geometry="cylinder_crossflow", Re=1e-12, D=0.02, k=0.0263, Pr=0.707)["Nu"] - 0.3)
        < 1e-5
    )


def test_pipe_laminar_exact_values() -> None:
    t = e(geometry="pipe", Re=1000.0, D=0.01, k=0.6, Pr=7.0)
    q = e(geometry="pipe", Re=1000.0, D=0.01, k=0.6, Pr=7.0, bc="uniform_q")
    assert t["Nu"] == 3.66 and q["Nu"] == 4.36 and t["regime"] == "laminar"
    assert abs(t["Nu"] - 3.6567934577632923619) / 3.6567934577632923619 < 1e-3


def test_gnielinski_reynolds_analogy_and_simplified_forms() -> None:
    # Pr = 1: the denominator is 1 and the Reynolds analogy Nu = (f/8)(Re - 1000) remains
    for re in (1e4, 1e5, 1e6):
        r = e(geometry="pipe", Re=re, D=0.02, k=0.6, Pr=1.0)
        f = (0.79 * math.log(re) - 1.64) ** -2
        assert rel(r["f_darcy"], f, 1e-12) and rel(r["Nu"], f / 8 * (re - 1000), 1e-12)
    # simplified forms of Gnielinski: measured gap at most 13.07% on this grid
    worst = 0.0
    for re in (1e4, 1e5, 1e6):
        for pr in (0.7, 1.2, 5.0, 50.0):
            nu = e(geometry="pipe", Re=re, D=0.02, k=0.6, Pr=pr)["Nu"]
            simple = (
                0.0214 * (re**0.8 - 100) * pr**0.4
                if pr <= 1.5
                else 0.012 * (re**0.87 - 280) * pr**0.4
            )
            worst = max(worst, abs(nu / simple - 1))
    assert worst < 0.15


def test_dittus_boelter_and_sieder_tate() -> None:
    heat = e(
        geometry="pipe",
        Re=5e4,
        D=0.02,
        k=0.6,
        Pr=5.0,
        correlation="dittus_boelter",
        T_s=[60, "C"],
        T_inf=[20, "C"],
    )
    cool = e(
        geometry="pipe", Re=5e4, D=0.02, k=0.6, Pr=5.0, correlation="dittus_boelter", heating=False
    )
    assert rel(heat["Nu"], 0.023 * 5e4**0.8 * 5.0**0.4, 1e-12)
    assert rel(cool["Nu"], 0.023 * 5e4**0.8 * 5.0**0.3, 1e-12)
    st40 = e(
        geometry="pipe",
        Re=5e4,
        D=0.02,
        k=0.6,
        Pr=7.0,
        correlation="sieder_tate",
        mu=10.07e-4,
        mu_w=6.96e-4,
        rho=998.0,
    )
    st100 = e(
        geometry="pipe",
        Re=5e4,
        D=0.02,
        k=0.6,
        Pr=7.0,
        correlation="sieder_tate",
        mu=10.07e-4,
        mu_w=2.82e-4,
        rho=998.0,
    )
    assert round(st40["viscosity_ratio"], 2) == 1.45 and round(st100["viscosity_ratio"], 2) == 3.57
    assert rel(st40["Nu"], 0.027 * 5e4**0.8 * 7.0 ** (1 / 3) * (10.07 / 6.96) ** 0.14, 1e-12)


def test_vertical_plate_grashof_and_churchill_chu() -> None:
    resp = full(geometry="vertical_plate", L=0.5, T_s=[60, "C"], T_inf=[20, "C"], **AIR)
    r = resp["result"]
    tf = (333.15 + 293.15) / 2
    gr = 9.80665 * (1 / tf) * 40 * 0.5**3 / AIR["nu"] ** 2
    assert rel(r["Gr"], gr, 1e-12) and rel(r["Ra"], gr * AIR["Pr"], 1e-12)
    assert any("ideal gases" in w for w in resp["warnings"])
    psi = 1 + (0.492 / AIR["Pr"]) ** (9 / 16)
    assert r["regime"] == "laminar" and "laminar" in r["correlation"]  # Ra < 1e9: laminar form
    assert rel(r["Nu"], 0.68 + 0.67 * r["Ra"] ** 0.25 / psi ** (4 / 9), 1e-12)
    turb = e(geometry="vertical_plate", Ra=1e11, L=3.0, k=0.0263, Pr=0.707)
    assert turb["regime"] == "turbulent" and rel(
        turb["Nu"], (0.825 + 0.387 * 1e11 ** (1 / 6) / psi ** (8 / 27)) ** 2, 1e-12
    )


def test_horizontal_plate_orientation_and_cylinder() -> None:
    up = e(
        geometry="horizontal_plate", A=0.25, P=2.0, Ra=1e6, facing="up", dT=30.0, k=0.0263, Pr=0.707
    )
    down = e(
        geometry="horizontal_plate",
        A=0.25,
        P=2.0,
        Ra=1e6,
        facing="down",
        dT=30.0,
        k=0.0263,
        Pr=0.707,
    )
    cold_down = e(
        geometry="horizontal_plate", L=0.125, Ra=1e6, facing="down", dT=-30.0, k=0.0263, Pr=0.707
    )
    assert up["L_char"] == 0.125 and rel(up["Nu"], 0.54 * 1e6**0.25, 1e-12)
    assert rel(down["Nu"], 0.27 * 1e6**0.25, 1e-12) and rel(cold_down["Nu"], up["Nu"], 1e-12)
    big = e(geometry="horizontal_plate", L=1.0, Ra=1e9, facing="up", dT=30.0, k=0.0263, Pr=0.707)
    assert rel(big["Nu"], 0.15 * 1e9 ** (1 / 3), 1e-12)
    cyl = e(geometry="horizontal_cylinder", D=0.1, Ra=1e6, k=0.0263, Pr=0.707)
    assert rel(
        cyl["Nu"],
        (0.6 + 0.387 * 1e6 ** (1 / 6) / (1 + (0.559 / 0.707) ** (9 / 16)) ** (8 / 27)) ** 2,
        1e-12,
    )
    enc = e(geometry="horizontal_enclosure", L=0.05, Ra=1e6, k=0.0263, Pr=0.707)
    assert rel(enc["Nu"], 0.069 * 1e6 ** (1 / 3) * 0.707**0.074, 1e-12)


def test_errors() -> None:
    assert "v and Re" in bad({"geometry": "flat_plate", "L": 1.0, "k": 0.03, "Pr": 0.7, "nu": 1e-5})
    assert "not all three" in bad(
        {
            "geometry": "flat_plate",
            "L": 1.0,
            "v": 1.0,
            "k": 0.03,
            "Pr": 0.7,
            "nu": 1e-5,
            "mu": 1e-5,
            "rho": 1.0,
        }
    )
    assert "facing" in bad(
        {"geometry": "horizontal_plate", "L": 1.0, "Ra": 1e6, "dT": 10.0, "k": 0.03, "Pr": 0.7}
    )
    assert "mu_w" in bad(
        {
            "geometry": "pipe",
            "D": 0.02,
            "Re": 5e4,
            "k": 0.6,
            "Pr": 7.0,
            "correlation": "sieder_tate",
        }
    )
    assert "heating" in bad(
        {
            "geometry": "pipe",
            "D": 0.02,
            "Re": 5e4,
            "k": 0.6,
            "Pr": 7.0,
            "correlation": "dittus_boelter",
        }
    )
    assert "beta" in bad(
        {"geometry": "vertical_plate", "L": 1.0, "dT": 10.0, "k": 0.03, "Pr": 0.7, "nu": 1e-5}
    )
