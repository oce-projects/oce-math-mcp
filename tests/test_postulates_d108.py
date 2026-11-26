"""D108: the postulates of special relativity ("List of relativistic equations", section "Postulates of
Special Relativity": its five formulas are the symbol c_0, the speed of light in a classical vacuum).

c_0 is a name of the constant c (exact, 299792458 m/s, SI 2019). The two postulates are checked on the
operations that already exist: c is the same in every inertial frame (velocity addition of c with any v gives
c; an event on the light cone stays on it after a boost) and is the upper bound of every speed (relativistic
velocity addition of two slower speeds stays slower, v = c is refused for a body); the laws are the same in
all inertial frames (the interval is invariant, two boosts make one boost with the added velocity, the boost
with -v undoes the boost with v).
"""

from __future__ import annotations

import itertools

import pytest

from oce_math_mcp.constants import lookup
from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc
from tests.conftest import ok

C = 299792458.0  # m/s, exact
BETAS = [-0.999, -0.9, -0.5, -0.1, 0.0, 0.2, 0.6, 0.95, 0.999]


def sr(params: dict) -> dict:
    return eng_calc.handle({"operation": "special_relativity", "params": params}, Ctx())


def approx(v: float, rel: float = 1e-12) -> object:
    return pytest.approx(v, rel=rel, abs=0)


def refused(params: dict, words: str) -> None:
    with pytest.raises(ToolError) as e:
        sr(params)
    assert e.value.code == "INVALID_ARGUMENT"
    assert words in e.value.message


# ------------------------------------------------------------------ c_0, the constant


@pytest.mark.parametrize("name", ["c_0", "c0"])
def test_c0_is_the_speed_of_light(name: str) -> None:
    assert lookup(name) == lookup("c")
    assert lookup(f"  {name} ")[1] == C
    r = ok("eng_calc", operation="physical_constant", params={"name": name})
    assert r["value"] == C and r["exact"] is True and r["uncertainty"] == 0
    e = ok("math_eval", expr=f"const('{name}') - const('c')", exact_mode="fraction")
    assert e["exact"] == "0"


def test_c0_keeps_c_as_the_symbol_of_the_constant() -> None:
    assert ok("eng_calc", operation="physical_constant", params={"name": "c_0"}) == ok(
        "eng_calc", operation="physical_constant", params={"name": "c"}
    )


# ------------------------------------------------------------------ second postulate: c is the same for all


@pytest.mark.parametrize("bv", BETAS)
def test_light_added_to_any_frame_velocity_is_light(bv: float) -> None:
    assert sr({"mode": "velocity_addition", "u_beta": 1, "v_beta": bv})["beta"] == 1
    assert sr({"mode": "velocity_addition", "u_beta": -1, "v_beta": bv})["beta"] == -1
    out = sr({"mode": "velocity_addition", "u": C, "v": bv * C})
    assert out["beta"] == 1 and out["v"] == C


def test_an_event_on_the_light_cone_stays_on_it() -> None:
    t = 2e-6
    for beta in BETAS:
        out = sr({"mode": "lorentz", "x": C * t, "t": t, "beta": beta})
        assert out["x_prime"] == approx(C * out["t_prime"])
        assert out["interval_type"] == "lightlike"
        # in the moving frame the rounded event is still on the cone (s2 ~ 1e-15 of the squares, not 0)
        again = sr({"mode": "lorentz", "x": out["x_prime"], "t": out["t_prime"], "beta": 0})
        assert again["interval_type"] == "lightlike"


def test_c_is_the_upper_bound_of_every_speed() -> None:
    for bu, bv in itertools.product(BETAS, repeat=2):
        assert abs(sr({"mode": "velocity_addition", "u_beta": bu, "v_beta": bv})["beta"]) < 1
    refused(
        {"mode": "velocity_addition", "u_beta": 1.1, "v_beta": 0.2},
        "|u_beta| must not exceed the speed of light",
    )
    refused(
        {"mode": "velocity_addition", "u": 1.01 * C, "v_beta": 0.2},
        "|u| must not exceed the speed of light",
    )
    refused(
        {"mode": "velocity_addition", "u_beta": 0.5, "v_beta": 1},
        "|v_beta| must be less than the speed of light",
    )
    refused(
        {"mode": "lorentz", "x": 1, "t": 1, "beta": -1},
        "|beta| must be less than the speed of light",
    )
    refused({"mode": "kinematics", "v": C}, "less than the speed of light")


# ------------------------------------------------------------------ first postulate: the same laws in all frames


def test_the_interval_is_the_same_in_every_frame() -> None:
    for x, t in [(3.0, 1e-8), (-120.0, 2e-7), (0.5, -4e-9)]:
        for beta in BETAS:
            out = sr({"mode": "lorentz", "x": x, "t": t, "beta": beta})
            ct2, x2 = (C * out["t_prime"]) ** 2, out["x_prime"] ** 2
            # (c t')^2 - x'^2 cancels: at beta = -0.999 the squares are ~1.8e4 m2 for s2 = -0.0124 m2, so the
            # rounding is measured against the squares (measured: 1.3e-10 m2 = 3.6e-15 of them)
            assert abs((ct2 - x2) - out["interval_s2"]) <= 1e-13 * (ct2 + x2)
            assert out["interval_s2"] == approx((C * t) ** 2 - x * x)


def test_two_boosts_make_one_boost_with_the_added_velocity() -> None:
    x, t = 40.0, 3e-7
    for b1, b2 in [(0.6, 0.3), (-0.9, 0.5), (0.95, 0.95), (0.2, -0.999)]:
        one = sr({"mode": "lorentz", "x": x, "t": t, "beta": b1})
        two = sr({"mode": "lorentz", "x": one["x_prime"], "t": one["t_prime"], "beta": b2})
        b12 = sr({"mode": "velocity_addition", "u_beta": b1, "v_beta": b2})["beta"]
        direct = sr({"mode": "lorentz", "x": x, "t": t, "beta": b12})
        assert two["x_prime"] == approx(direct["x_prime"], 1e-9)
        assert two["t_prime"] == approx(direct["t_prime"], 1e-9)


def test_the_opposite_boost_undoes_the_boost() -> None:
    x, t = -75.0, 6e-7
    for beta in BETAS:
        there = sr({"mode": "lorentz", "x": x, "t": t, "beta": beta})
        back = sr({"mode": "lorentz", "x": there["x_prime"], "t": there["t_prime"], "beta": -beta})
        assert back["x_prime"] == approx(x, 1e-9) and back["t_prime"] == approx(t, 1e-9)
