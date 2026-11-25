"""Rolling without slipping (D73).

Checks independent of the server formulas:
- Newton's equations of the rigid body solved as a linear system with numpy: unknowns a, alpha, friction f, normal N
  from m a = F_along - f, I alpha = f r, a = r alpha, N = normal load;
- energy route on the incline: from rest over a distance s, m g s sin(theta) = m v^2 / 2 + I (v / r)^2 / 2, then
  v^2 = 2 a s must hold.
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import call, ok

G0 = 9.80665


def roll(**params: object) -> dict:
    return ok("eng_calc", operation="rolling", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="rolling", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def _newton(m: float, inertia: float, r: float, along: float, normal: float) -> np.ndarray:
    mat = np.array(
        [
            [m, 0.0, 1.0, 0.0],  # m a + f = along
            [0.0, inertia, -r, 0.0],  # I alpha - f r = 0
            [1.0, -r, 0.0, 0.0],  # a - r alpha = 0
            [0.0, 0.0, 0.0, 1.0],  # N = normal
        ]
    )
    return np.linalg.solve(mat, np.array([along, 0.0, 0.0, normal]))


def test_incline_against_newton_and_energy() -> None:
    # solid sphere I = 2/5 m r^2, cylinder 1/2 m r^2, hoop m r^2 (given through I or the radius of gyration)
    for m, r, inertia, theta in (
        (2.0, 0.1, 0.4 * 2.0 * 0.01, 30.0),
        (5.0, 0.2, 0.5 * 5.0 * 0.04, 12.0),
        (1.0, 0.3, 0.09, 45.0),
    ):
        res = roll(mode="incline", m=m, r=r, I=inertia, theta_deg=theta, mu_s=0.9)
        t = math.radians(theta)
        a, alpha, f, n = _newton(m, inertia, r, m * G0 * math.sin(t), m * G0 * math.cos(t))
        assert math.isclose(res["a"], a, rel_tol=1e-12) and math.isclose(
            res["alpha"], alpha, rel_tol=1e-12
        )
        assert math.isclose(res["f"], f, rel_tol=1e-12) and math.isclose(res["N"], n, rel_tol=1e-12)
        assert math.isclose(res["mu_min"], f / n, rel_tol=1e-12)
        assert math.isclose(res["I_over_m_r2"], inertia / (m * r * r), rel_tol=1e-12)
        assert res["rolls_without_slipping"] == (f / n <= 0.9)
        s = 1.7
        v2 = 2 * m * G0 * s * math.sin(t) / (m + inertia / r**2)  # energy route
        assert math.isclose(2 * res["a"] * s, v2, rel_tol=1e-12)
        rg = roll(mode="incline", m=m, r=r, r_g=math.sqrt(inertia / m), theta_deg=theta)
        assert math.isclose(rg["a"], res["a"], rel_tol=1e-12)
    # page: the acceleration does not depend on the scale or the density for the same r_g / r
    big = roll(mode="incline", m=80.0, r=1.5, r_g=1.5 * math.sqrt(0.4), theta_deg=30.0)
    small = roll(mode="incline", m=0.2, r=0.01, r_g=0.01 * math.sqrt(0.4), theta_deg=30.0)
    assert math.isclose(big["a"], small["a"], rel_tol=1e-12)
    steep = call(
        "eng_calc",
        operation="rolling",
        params={"mode": "incline", "m": 1.0, "r": 0.1, "I": 0.005, "theta_deg": 70.0, "mu_s": 0.2},
    )
    assert steep["result"]["rolls_without_slipping"] is False
    assert any("slip" in w for w in steep["warnings"]), steep["warnings"]


def test_external_force_and_energy() -> None:
    m, r, inertia, force = 12.0, 0.25, 0.3, 40.0
    res = roll(mode="force", m=m, r=r, I=inertia, F=force, v=3.0)
    a, alpha, f, n = _newton(m, inertia, r, force, m * G0)
    assert math.isclose(res["alpha"], alpha, rel_tol=1e-12)
    assert math.isclose(res["a"], a, rel_tol=1e-12) and math.isclose(res["f"], f, rel_tol=1e-12)
    assert math.isclose(res["F_net"], m * a, rel_tol=1e-12) and math.isclose(
        res["N"], n, rel_tol=1e-12
    )
    # kinetic energy: translation of the centre plus rotation about it, omega = v / r
    assert math.isclose(res["K_translation"], m * 9.0 / 2, rel_tol=1e-12)
    assert math.isclose(res["K_rotation"], inertia * (3.0 / r) ** 2 / 2, rel_tol=1e-12)
    assert math.isclose(res["K"], res["K_translation"] + res["K_rotation"], rel_tol=1e-12)
    assert "exactly one" in bad({"mode": "force", "m": 1.0, "r": 0.1, "F": 1.0})
    assert "Keys not used" in bad(
        {"mode": "force", "m": 1.0, "r": 0.1, "I": 0.01, "F": 1.0, "theta_deg": 10.0}
    )
