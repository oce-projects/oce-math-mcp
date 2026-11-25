"""Catenary cables and journal bearings (D58).

Sources: Wikipedia "Catenary" (y = a cosh(x / a), a = T0 / w, T = w y, L^2 - v^2 = 4 a^2 sinh^2(H / (2a)),
H = 2 a arcosh((h + a) / a)) and "Sommerfeld number" (S, Petroff's law).
Checks independent of the module formulas:
- cable length against the numerical integral of the arc length (quad of sqrt(1 + y'^2));
- heights of the two supports against the given height difference;
- equilibrium: vertical component of the tension at the support = weight of the cable up to the vertex,
  horizontal component = T0;
- small sag against the parabola w H^2 / (8 T0);
- Petroff: torque T = f W r (identity of the source) and torque from the shear stress integrated over the
  surface (quad); Sommerfeld number with w instead of N: factor 2 pi.
"""

from __future__ import annotations

import math

from scipy.integrate import quad

from tests.conftest import call, ok


def cat(**params: object) -> dict:
    return ok("eng_calc", operation="cable_catenary", params=params)


def brg(**params: object) -> dict:
    return ok("eng_calc", operation="journal_bearing", params=params)


def bad(operation: str, params: dict) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def test_level_span_length_against_arc_integral() -> None:
    r = cat(mode="span_length", w=15.0, H=100.0, L=110.0)
    a = r["a"]
    arc = quad(lambda x: math.sqrt(1 + math.sinh(x / a) ** 2), -50.0, 50.0, epsabs=0, epsrel=1e-13)[
        0
    ]
    assert rel(arc, 110.0, 1e-11)
    assert rel(r["sag"], a * (math.cosh(50.0 / a) - 1), 1e-12)
    assert rel(r["T0"], 15.0 * a, 1e-14)
    # equilibrium at the support: vertical = weight of half the cable, horizontal = T0
    tv = r["T_max"] * math.sin(math.radians(r["angle_support_deg"][1]))
    th = r["T_max"] * math.cos(math.radians(r["angle_support_deg"][1]))
    assert rel(tv, 15.0 * 55.0, 1e-10) and rel(th, r["T0"], 1e-10)


def test_unequal_supports() -> None:
    r = cat(mode="span_length", w=8.0, H=60.0, L=75.0, v=12.0)
    a, x1, x2 = r["a"], r["x_supports"][0], r["x_supports"][1]
    assert rel(x2 - x1, 60.0, 1e-12)
    assert rel(a * math.cosh(x2 / a) - a * math.cosh(x1 / a), 12.0, 1e-10)
    arc = quad(lambda x: math.cosh(x / a), x1, x2, epsabs=0, epsrel=1e-13)[0]
    assert rel(arc, 75.0, 1e-10)
    assert rel(r["T_supports"][1], 8.0 * a * math.cosh(x2 / a), 1e-12)
    assert r["T_supports"][1] > r["T_supports"][0]


def test_sag_and_tension_modes_and_parabola_limit() -> None:
    s = cat(mode="span_sag", w=10.0, H=200.0, sag=20.0)
    t = cat(mode="span_tension", w=10.0, H=200.0, T0=s["T0"])
    assert rel(t["sag"], 20.0, 1e-11) and rel(t["L"], s["L"], 1e-12)
    flat = cat(mode="span_tension", w=2.0, H=50.0, T0=1e5)
    assert rel(flat["sag"], 2.0 * 50.0**2 / (8 * 1e5), 1e-4)  # small sag: parabola
    assert "L > H" in bad("cable_catenary", {"mode": "span_length", "w": 1.0, "H": 10.0, "L": 9.0})
    assert "v" in bad(
        "cable_catenary", {"mode": "span_length", "w": 1.0, "H": 10.0, "L": 12.0, "v": 13.0}
    )


def test_petroff_and_sommerfeld() -> None:
    p = {"D": 0.05, "c": 25e-6, "l": 0.04, "mu": 0.03, "rpm": 1800.0, "W": 2000.0}
    r = brg(**p)
    rad, n = 0.025, 30.0
    press = 2000.0 / (2 * rad * 0.04)
    assert rel(r["P"], press, 1e-14)
    assert rel(r["S"], (rad / 25e-6) ** 2 * 0.03 * n / press, 1e-13)
    assert rel(r["T_friction"], r["f"] * 2000.0 * rad, 1e-12)  # T = f W r
    # integrated shear: tau = mu U / c over the surface 2 pi r l, lever arm r
    tau = 0.03 * 2 * math.pi * rad * n / 25e-6
    torque = quad(lambda _phi: tau * rad * 0.04 * rad, 0, 2 * math.pi)[0]
    assert rel(r["T_friction"], torque, 1e-12)
    assert rel(r["P_loss"], r["T_friction"] * 2 * math.pi * n, 1e-12)
    assert rel(r["S_omega"], 2 * math.pi * r["S"], 1e-14)
    assert "only one" in bad("journal_bearing", {**p, "N": 30.0})
