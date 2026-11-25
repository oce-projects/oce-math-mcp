"""Local losses at sudden changes of section, Borda-Carnot equation (D70).

Checks independent of the server formulas:
- sudden expansion: momentum balance on the control volume between the two sections (pressure p1 on the annular
  wall), solved for p2 with numpy; the loss is the drop of p + rho v^2 / 2;
- sudden contraction: Bernoulli without loss from section 1 to the vena contracta 3, then the momentum balance from 3
  to 2 (pressure p3 on the whole section A2), solved as a linear system for p3 and p2.
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import call, ok

G0 = 9.80665


def loss(**params: object) -> dict:
    return ok("eng_calc", operation="local_loss", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="local_loss", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def _expansion_by_momentum(
    a1: float, a2: float, v1: float, rho: float, p1: float
) -> tuple[float, float]:
    """(p2, energy loss per unit volume) from A1 (rho v1^2 + p1) - A2 (rho v2^2 + p2) + (A2 - A1) p1 = 0."""
    v2 = a1 * v1 / a2
    p2 = float(
        np.linalg.solve([[a2]], [a1 * (rho * v1**2 + p1) - a2 * rho * v2**2 + (a2 - a1) * p1])[0]
    )
    return p2, (p1 + rho * v1**2 / 2) - (p2 + rho * v2**2 / 2)


def _contraction_by_momentum(
    a1: float, a2: float, mu: float, v1: float, rho: float, p1: float
) -> tuple[float, float]:
    """(p2, energy loss): Bernoulli 1 -> 3 and momentum 3 -> 2, unknowns p3 and p2."""
    v2 = a1 * v1 / a2
    a3 = mu * a2
    v3 = a1 * v1 / a3
    mat = np.array([[1.0, 0.0], [a2, -a2]])
    rhs = np.array([p1 + rho * (v1**2 - v3**2) / 2, rho * (a2 * v2**2 - a3 * v3**2)])
    _, p2 = np.linalg.solve(mat, rhs)  # p3 is the first unknown
    return float(p2), (p1 + rho * v1**2 / 2) - (float(p2) + rho * v2**2 / 2)


def test_sudden_expansion_against_momentum_balance() -> None:
    rho, p1 = 998.0, 2.0e5
    for d1, d2, q in ((0.05, 0.1, 0.004), (0.08, 0.3, 0.02), (0.1, 0.12, 0.01)):
        a1, a2 = math.pi * d1**2 / 4, math.pi * d2**2 / 4
        v1 = q / a1
        r = loss(mode="sudden_expansion", D1=d1, D2=d2, Q=q, rho=rho)
        p2, de = _expansion_by_momentum(a1, a2, v1, rho, p1)
        assert rel(r["dE"], de, 1e-9), (r, de)
        assert rel(r["dp"], p1 - p2, 1e-9)  # negative: the pressure rises after the expansion
        assert r["dp"] < 0
        assert rel(r["dH"], de / (rho * G0), 1e-9)
        assert rel(r["dh"], (p1 - p2) / (rho * G0), 1e-9)
        assert rel(r["v2"], v1 * a1 / a2, 1e-12)
        assert rel(r["K_1"] * rho * v1**2 / 2, de, 1e-9)
        assert rel(r["K_2"] * rho * r["v2"] ** 2 / 2, de, 1e-9)
    # areas and the velocity instead of diameters and flow rate; a loss coefficient below one
    r = loss(mode="sudden_expansion", A1=0.01, A2=0.04, v1=4.0, rho=1000.0, xi=0.6)
    assert rel(r["dE"], 0.6 * 1000.0 / 2 * (4.0 - 1.0) ** 2, 1e-12)
    assert rel(
        r["dp"], r["dE"] - 1000.0 / 2 * (4.0**2 - 1.0**2), 1e-12
    )  # energy balance, horizontal pipe
    assert "sudden_contraction" in bad(
        {"mode": "sudden_expansion", "D1": 0.1, "D2": 0.05, "Q": 0.01, "rho": 1000.0}
    )
    assert "xi" in bad(
        {"mode": "sudden_expansion", "D1": 0.05, "D2": 0.1, "Q": 0.01, "rho": 1000.0, "xi": 1.5}
    )


def test_sudden_contraction_against_momentum_balance() -> None:
    rho, p1 = 1000.0, 3.0e5
    for d1, d2, v1 in ((0.2, 0.1, 1.5), (0.15, 0.12, 2.0), (0.5, 0.05, 0.05)):
        a1, a2 = math.pi * d1**2 / 4, math.pi * d2**2 / 4
        mu_w = 0.63 + 0.37 * (a2 / a1) ** 3  # Weisbach, sharp-edged contraction
        r = loss(mode="sudden_contraction", D1=d1, D2=d2, v1=v1, rho=rho)
        assert rel(r["mu"], mu_w, 1e-12)
        p2, de = _contraction_by_momentum(a1, a2, mu_w, v1, rho, p1)
        assert rel(r["dE"], de, 1e-9), (r, de)
        assert rel(r["dp"], p1 - p2, 1e-9)
        assert rel(r["v3"], v1 * a1 / (mu_w * a2), 1e-12)
        assert rel(r["A3"], mu_w * a2, 1e-12)
        assert rel(r["K_2"] * rho * r["v2"] ** 2 / 2, de, 1e-9)
        assert rel(r["K_1"] * rho * v1**2 / 2, de, 1e-9)
    # given contraction coefficient (rounded entrance, for example)
    r = loss(mode="sudden_contraction", A1=0.05, A2=0.01, Q=0.03, rho=1000.0, mu=0.9)
    p2, de = _contraction_by_momentum(0.05, 0.01, 0.9, 0.6, 1000.0, 1.0e5)
    assert rel(r["dE"], de, 1e-9) and rel(r["mu"], 0.9, 1e-15)
    assert "mu" in bad(
        {"mode": "sudden_contraction", "A1": 0.05, "A2": 0.01, "Q": 0.03, "rho": 1000.0, "mu": 1.2}
    )
    assert "sudden_expansion" in bad(
        {"mode": "sudden_contraction", "A1": 0.01, "A2": 0.05, "Q": 0.03, "rho": 1000.0}
    )


def test_borda_carnot_general_form() -> None:
    r = loss(mode="borda_carnot", xi=0.5, v1=3.0, v2=1.0, rho=1000.0)
    assert rel(r["dE"], 1000.0, 1e-12) and rel(r["dH"], 1000.0 / (1000.0 * G0), 1e-12)
    r = loss(mode="borda_carnot", xi=1.0, v1=3.0, v2=1.0, rho=1000.0, g=9.81)
    assert rel(r["dH"], 2000.0 / (1000.0 * 9.81), 1e-12)
    # no loss when the velocity does not decrease (no mechanical energy gain without external work)
    slow = {"mode": "borda_carnot", "xi": 1.0, "v1": 1.0, "v2": 2.0, "rho": 1000.0}
    resp = call("eng_calc", operation="local_loss", params=slow)
    assert resp["ok"] and resp["result"]["dE"] == 0.0 and resp["warnings"], resp
    assert "Keys not used" in bad(
        {"mode": "borda_carnot", "xi": 1.0, "v1": 3.0, "v2": 1.0, "rho": 1.0, "D1": 0.1}
    )
