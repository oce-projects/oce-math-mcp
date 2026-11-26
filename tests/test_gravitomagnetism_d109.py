"""D109: eng_calc.gravitation mode `gravitomagnetism` (sections "Gravitoelectromagnetism" and "Weak-field
relativistic equations" of "List of equations in gravitation").

The list page writes the field of a rotating body as G / (2 c^2) (L 3 (L . r) r) / r^3: the minus sign is lost.
The article "Gravitoelectromagnetism" gives xi = B_g = G / (2 c^2) (L - 3 (L . r^) r^) / r^3, its potential
h = A_g = -G / (2 c^2) L x r / r^3, and the force m (v x 4 B_g). The list page writes the force m (v x 2 xi): with
the same xi that is half of it. The tests decide with the Lense-Thirring precession of the orbital node,
2 G J / (c^2 a^3), from the tool's acceleration sampled on a circular orbit: it holds with 4 xi, not with 2 xi.

References: the field by components with numpy; xi = curl h by central differences on the tool's own h; the
equator and the pole of the article; the article's Earth formula G m / (5 c^2 r) 2 pi / T = 2 pi r g0 / (5 c^2 T);
the nodal and the gyroscope (Schiff) frame-dragging rates averaged over an orbit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc

G = 6.67430e-11  # m3/(kg s2), CODATA 2022
C = 299792458.0  # m/s, exact
K = G / (2 * C * C)
J_EARTH = (
    8.034e37 * 7.2921159e-5
)  # kg m2/s: polar moment of inertia times the sidereal rotation rate


def gm(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle(
        {"operation": "gravitation", "params": {"mode": "gravitomagnetism", **params}}, ctx or Ctx()
    )


def approx(v: Any, rel: float = 1e-12) -> Any:
    return pytest.approx(v, rel=rel, abs=0)


def close(got: list[float], want: Any, rel: float = 1e-12) -> None:
    """Vectors: the error measured against the length of the expected vector (components may be ~0)."""
    w = np.asarray(want, dtype=float)
    assert len(got) == 3
    assert float(np.max(np.abs(np.asarray(got) - w))) <= rel * float(np.linalg.norm(w))


def units(ctx: Ctx) -> dict[str, str]:
    assert ctx.units is not None
    return ctx.units


def bad(params: dict, words: str) -> None:
    with pytest.raises(ToolError) as e:
        gm(params)
    assert e.value.code == "INVALID_ARGUMENT"
    assert words in e.value.message


# ------------------------------------------------------------------ the field and its potential


def test_field_at_the_equator_and_at_the_pole() -> None:
    l0, r = 5.0e33, 7.0e6
    ctx = Ctx()
    eq = gm({"L": [0, 0, l0], "point": [r, 0, 0]}, ctx)
    close(eq["xi"], [0, 0, K * l0 / r**3])  # along L, the article's equatorial value
    close(eq["Omega"], [0, 0, 2 * K * l0 / r**3])
    close(eq["gyroscope_precession"], [0, 0, -G * l0 / (C * C * r**3)])
    assert eq["xi_magnitude"] == approx(K * l0 / r**3)
    assert eq["r_magnitude"] == approx(r)
    assert (
        units(ctx)["xi"] == "1/s"
        and units(ctx)["h"] == "m/s"
        and units(ctx)["gyroscope_precession"] == "rad/s"
    )
    pole = gm({"L": [0, 0, l0], "point": [0, 0, r]})
    close(pole["xi"], [0, 0, -2 * K * l0 / r**3])  # L - 3 L = -2 L
    assert pole["h"] == [0, 0, 0]  # L parallel to r


def test_field_and_potential_by_components() -> None:
    lv, x, x0 = [1.2e30, -0.7e30, 2.5e30], [3.1e6, -1.4e6, 2.2e6], [0.2e6, 0.1e6, -0.3e6]
    out = gm({"L": lv, "point": x, "center": x0})
    r = np.asarray(x) - np.asarray(x0)
    n = float(np.linalg.norm(r))
    rh = r / n
    lv_a = np.asarray(lv)
    close(out["r_vector"], r)
    close(out["xi"], K * (lv_a - 3 * float(np.dot(lv_a, rh)) * rh) / n**3)
    close(out["h"], -K * np.cross(lv_a, r) / n**3)
    close(out["Omega"], 2 * np.asarray(out["xi"]))
    close(out["gyroscope_precession"], -2 * np.asarray(out["xi"]))


def test_xi_is_the_curl_of_h() -> None:
    """The page's xi = curl h: central differences of the tool's own h at a generic point."""
    lv, x = [0.4e33, -1.1e33, 2.0e33], np.array([5.0e6, 2.0e6, -4.0e6])
    h = 1.0

    def h_at(pt: Any) -> Any:
        return np.array(gm({"L": lv, "point": [float(v) for v in pt]})["h"])

    jac = np.zeros((3, 3))  # jac[i, j] = dh_i / dx_j
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        jac[:, j] = (h_at(x + e) - h_at(x - e)) / (2 * h)
    curl = [jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]]
    close(curl, gm({"L": lv, "point": [float(v) for v in x]})["xi"], 1e-6)


def test_the_earth_formula_of_the_article() -> None:
    m, radius, period = 5.972e24, 6.371e6, 86164.0
    lv = 2 * m * radius**2 / 5 * 2 * math.pi / period  # uniform ball, I = 2 m R^2 / 5
    out = gm({"L": [0, 0, lv], "point": [radius, 0, 0]})
    assert out["xi"][2] == approx(G / (5 * C * C) * m / radius * 2 * math.pi / period)
    g0 = G * m / radius**2
    assert out["xi"][2] == approx(2 * math.pi * radius * g0 / (5 * C * C * period))
    assert out["xi"][2] > 0  # north, along the angular momentum


# ------------------------------------------------------------------ force: 4 xi, decided by Lense-Thirring


def orbit(a: float, inc: float, n: float, u: float) -> tuple[list[float], list[float]]:
    pos = [a * math.cos(u), a * math.sin(u) * math.cos(inc), a * math.sin(u) * math.sin(inc)]
    vel = [
        -n * a * math.sin(u),
        n * a * math.cos(u) * math.cos(inc),
        n * a * math.cos(u) * math.sin(inc),
    ]
    return pos, vel


def test_the_force_gives_the_lense_thirring_node_precession() -> None:
    """Gauss equation for the node of a circular orbit, averaged over 64 points (exact for these terms)."""
    gm_earth, a, inc = 3.986004418e14, 12270e3, math.radians(109.84)  # LAGEOS
    n = math.sqrt(gm_earth / a**3)
    normal = np.array([0.0, -math.sin(inc), math.cos(inc)])
    rates = []
    for k in range(64):
        u = 2 * math.pi * k / 64
        pos, vel = orbit(a, inc, n, u)
        acc = np.array(gm({"L": [0, 0, J_EARTH], "point": pos, "v": vel})["a_gm"])
        rates.append(a * math.sin(u) * float(acc @ normal) / (n * a * a * math.sin(inc)))
    node = float(np.mean(rates))
    assert node == approx(2 * G * J_EARTH / (C * C * a**3), 1e-10)
    mas_per_year = node * 180 / math.pi * 3600e3 * 365.25 * 86400
    assert 30 < mas_per_year < 32  # LAGEOS: about 31 mas per year


def test_gyroscope_precession_averaged_on_a_polar_orbit() -> None:
    a = 6371e3 + 642e3  # Gravity Probe B
    mean = np.zeros(3)
    for k in range(64):
        u = 2 * math.pi * k / 64
        mean += np.array(
            gm({"L": [0, 0, J_EARTH], "point": [a * math.cos(u), 0, a * math.sin(u)]})[
                "gyroscope_precession"
            ]
        )
    close(list(mean / 64), [0, 0, G * J_EARTH / (2 * C * C * a**3)], 1e-10)


def test_force_on_a_test_mass() -> None:
    lv, x, v, m = [0, 0, 6.0e33], [7.0e6, 1.0e6, 0.5e6], [100.0, -7500.0, 300.0], 12.0
    ctx = Ctx()
    out = gm({"L": lv, "point": x, "v": v, "m_test": m}, ctx)
    xi = np.asarray(out["xi"])
    close(out["a_gm"], 4 * np.cross(v, xi))
    close(out["a_gm"], 2 * np.cross(v, out["Omega"]))
    close(out["F_gm"], m * np.asarray(out["a_gm"]))
    assert abs(float(np.dot(out["F_gm"], v))) <= 1e-12 * float(np.linalg.norm(out["F_gm"])) * float(
        np.linalg.norm(v)
    )
    assert units(ctx)["a_gm"] == "m/s2" and units(ctx)["F_gm"] == "N"
    assert "F_gm" not in gm({"L": lv, "point": x, "v": v})


def test_flux_through_a_flat_surface() -> None:
    out = gm({"L": [0, 0, 5.0e33], "point": [7.0e6, 0, 0], "normal": [0, 0, 2], "A": 3.0})
    assert out["flux_xi"] == approx(out["xi"][2] * 3.0)
    assert out["flux_Omega"] == approx(2 * out["flux_xi"])
    tilted = gm({"L": [0, 0, 5.0e33], "point": [7.0e6, 0, 0], "normal": [1, 0, 1], "A": 3.0})
    assert tilted["flux_xi"] == approx(out["xi"][2] * 3.0 / math.sqrt(2))


def test_gravitomagnetism_errors() -> None:
    bad({"L": [0, 0, 1e33], "point": [1, 2, 3], "center": [1, 2, 3]}, "infinite")
    bad({"L": [0, 0], "point": [1, 2, 3]}, "params.L")
    bad({"L": [0, 0, 1e33], "point": [7e6, 0, 0], "m_test": 2}, "params.m_test needs params.v")
    bad({"L": [0, 0, 1e33], "point": [7e6, 0, 0], "v": [C, 0, 0]}, "less than the speed of light")
    bad({"L": [0, 0, 1e33], "point": [7e6, 0, 0], "normal": [0, 0, 1]}, "The flux needs both")
    bad({"L": [0, 0, 1e33], "point": [7e6, 0, 0], "A": 2.0}, "The flux needs both")
    bad({"L": [0, 0, 1e33], "point": [7e6, 0, 0], "normal": [0, 0, 0], "A": 2.0}, "zero vector")
    bad({"L": [0, 0, 1e33], "point": [7e6, 0, 0], "M": 5e24}, "Keys not used")
