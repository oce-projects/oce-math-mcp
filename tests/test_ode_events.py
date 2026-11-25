"""Events in math_numeric.ode_ivp (D41): stopping when an expression is zero.

Independent methods: flight time and range of the projectile in closed form; linear air
drag with the exact solution and the root found by brentq; zeros of the harmonic oscillator
(pi/2 + k pi).
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
from scipy.optimize import brentq

from tests.conftest import call, ok

G = 9.80665
PROJ = {"expr": "vx; vy; 0; -9.80665", "variables": ["t", "x", "y", "vx", "vy"]}


def ode(**args: Any) -> dict[str, Any]:
    return ok("math_numeric", operation="ode_ivp", **args)


def err(**args: Any) -> str:
    resp = call("math_numeric", operation="ode_ivp", **args)
    assert resp["ok"] is False, resp
    return resp["error"]["code"]


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


@pytest.mark.parametrize("direction", ["falling", "both"])
def test_projectile_from_ground_lands_at_closed_form_time(direction: str) -> None:
    v0, ang = 20.0, math.radians(53.0)
    vx, vy = v0 * math.cos(ang), v0 * math.sin(ang)
    r = ode(
        **PROJ, x0=[0, 0, vx, vy], bounds=[0, 100], events=[{"expr": "y", "direction": direction}]
    )
    t_fly = 2 * vy / G
    # the zero at t0 (starting from the ground) does not count, even with direction "both"
    assert r["stopped_by"] == 0 and rel(r["t_end"], t_fly) and rel(r["t_events"][0][0], t_fly)
    assert rel(r["final"][0], vx * t_fly) and abs(r["final"][1]) < 1e-9
    assert rel(r["y_events"][0][0][0], vx * t_fly)
    assert len(r["t"]) == 11 and rel(r["t"][-1], t_fly) and rel(r["y"][0][-1], vx * t_fly)


def test_apex_not_terminal_and_landing_terminal() -> None:
    vx, vy, h0 = 12.0, 15.0, 3.0
    r = ode(
        **PROJ,
        x0=[0, h0, vx, vy],
        bounds=[0, 50],
        samples=5,
        events=[
            {"expr": "vy", "direction": "falling", "terminal": False},
            {"expr": "y", "direction": "falling"},
        ],
    )
    t_apex = vy / G
    t_land = (vy + math.sqrt(vy * vy + 2 * G * h0)) / G
    assert rel(r["t_events"][0][0], t_apex) and rel(r["y_events"][0][0][1], h0 + vy * vy / (2 * G))
    assert r["stopped_by"] == 1 and rel(r["t_end"], t_land) and len(r["t"]) == 5


def test_linear_drag_against_exact_solution() -> None:
    k, vx0, vy0, h0 = 0.3, 25.0, 18.0, 2.0

    def y_exact(t: float) -> float:
        return h0 + (vy0 + G / k) / k * (1 - math.exp(-k * t)) - G * t / k

    t_land = float(cast(Any, brentq)(y_exact, 0.1, 60, xtol=1e-15))
    r = ode(
        expr="vx; vy; -0.3*vx; -9.80665 - 0.3*vy",
        variables=["t", "x", "y", "vx", "vy"],
        x0=[0, h0, vx0, vy0],
        bounds=[0, 60],
        events=[{"expr": "y", "direction": "falling"}],
    )
    assert rel(r["t_end"], t_land, 1e-8)
    assert rel(r["final"][0], vx0 / k * (1 - math.exp(-k * t_land)), 1e-8)


def test_oscillator_zero_crossings_not_terminal() -> None:
    r = ode(
        expr="v; -x",
        variables=["t", "x", "v"],
        x0=[1, 0],
        bounds=[0, 10],
        events=[{"expr": "x", "terminal": False}],
    )
    zeros = [
        math.pi / 2 + j * math.pi for j in range(3)
    ]  # 1.57 4.71 7.85 (the fourth is beyond 10)
    assert len(r["t_events"][0]) == 3
    assert all(rel(a, b, 1e-8) for a, b in zip(r["t_events"][0], zeros, strict=True))
    assert (
        r["stopped_by"] is None and rel(r["t_end"], 10.0) and rel(r["final"][0], math.cos(10), 1e-8)
    )


def test_terminal_event_never_reached_warns() -> None:
    resp = call(
        "math_numeric",
        operation="ode_ivp",
        **PROJ,
        x0=[0, 100, 1, 0],
        bounds=[0, 1],
        events=[{"expr": "y", "direction": "falling"}],
    )
    assert resp["ok"] and resp["result"]["stopped_by"] is None and resp["warnings"]
    assert resp["result"]["t_events"] == [[]] and rel(resp["result"]["t_end"], 1.0)


def test_without_events_result_keys_unchanged() -> None:
    r = ode(expr="-y", variables=["t", "y"], x0=[1], bounds=[0, 1])
    assert set(r) == {"t", "y", "final", "method", "nfev"}


@pytest.mark.parametrize(
    ("events", "code"),
    [
        ([{"expr": "z"}], "INVALID_ARGUMENT"),  # unknown variable
        ([{"expr": "y", "direction": "down"}], "INVALID_ARGUMENT"),
        ([{"expr": "y", "terminal": "yes"}], "INVALID_ARGUMENT"),
        ([{"expr": "y", "when": 1}], "INVALID_ARGUMENT"),  # unexpected key
        (["y"], "INVALID_ARGUMENT"),  # an object is needed
        ([{"direction": "falling"}], "INVALID_ARGUMENT"),  # expr missing
        ([{"expr": "y"}] * 11, "LIMIT_EXCEEDED"),
    ],
)
def test_bad_events(events: list[Any], code: str) -> None:
    assert err(**PROJ, x0=[0, 0, 1, 1], bounds=[0, 5], events=events) == code


def test_event_expression_goes_through_the_safe_parser() -> None:
    code = err(
        **PROJ, x0=[0, 0, 1, 1], bounds=[0, 5], events=[{"expr": "__import__('os').getcwd()"}]
    )
    assert code in {"FORBIDDEN", "PARSE_ERROR", "INVALID_ARGUMENT"}
