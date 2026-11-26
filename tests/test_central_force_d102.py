"""D102: math_solve.central_force (the Binet equation u'' + u = -mu F / (l^2 u^2), u = 1/r).

References: textbook force laws of known orbits (conic: inverse square; circle through the centre: 1/r^5;
cardioid: 1/r^4; logarithmic spiral: 1/r^3) and the orbit integrated here with scipy solve_ivp from Newton's
equations in the plane, with the force the server returns.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.expr import parse
from oce_math_mcp.tools import math_solve
from oce_math_mcp.tools.common import grammar


def run(ctx: Ctx | None = None, **args: Any) -> dict:
    return math_solve.handle({"operation": "central_force", **args}, ctx or Ctx())


def expr(text: str) -> Any:
    """x'' -> x_ddot and x' -> x_dot; an equation 'a = b' becomes a - b."""
    if "=" in text:
        lhs, rhs = text.split("=")
        text = f"({lhs}) - ({rhs})"
    text = re.sub(r"([A-Za-z]\w*)''", r"\1_ddot", text)
    return parse(re.sub(r"([A-Za-z]\w*)'", r"\1_dot", text), grammar())


def num(text: str, **values: float) -> float:
    e = expr(text)
    subs = {s: sp.Float(repr(values[s.name]), 30) for s in e.free_symbols}
    z = complex(sp.N(e.subs(subs), 30))
    assert abs(z.imag) <= 1e-25 * (1 + abs(z.real))
    return z.real


def approx(v: float, rel: float = 1e-12) -> Any:
    return pytest.approx(v, rel=rel, abs=1e-14)


def integrate_orbit(force: str, consts: dict[str, float], r0: float, theta_end: float) -> Any:
    """Newton in the plane: mu r'' = F(|r|) r/|r|, start at theta = 0 with r = r0, radial speed 0 and
    angular momentum l. Returns a function theta -> r along the integrated path."""
    mu, ell = consts["mu"], consts["l"]
    e = expr(force)
    fr = sp.lambdify(sp.Symbol("r"), e.subs({sp.Symbol(k): v for k, v in consts.items()}), "math")

    def rhs(_t: float, s: Any) -> list[float]:
        x, y, vx, vy = s
        rr = math.hypot(x, y)
        f = float(fr(rr))
        return [vx, vy, f * x / (rr * mu), f * y / (rr * mu)]

    def reached(_t: float, s: Any) -> float:
        return math.atan2(s[1], s[0]) - theta_end

    reached.terminal = True  # type: ignore[attr-defined]
    vt = ell / (mu * r0)
    sol = solve_ivp(
        rhs, (0, 50), [r0, 0, 0, vt], rtol=1e-11, atol=1e-13, dense_output=True, events=reached
    )
    ts = np.linspace(0, sol.t_events[0][0], 40)
    pts = sol.sol(ts)
    return np.arctan2(pts[1], pts[0]), np.hypot(pts[0], pts[1])


def test_conic_gives_inverse_square_and_reproduces_the_orbit() -> None:
    out = run(orbit="p/(1 + e*cos(theta))")
    assert out["verified"] is True
    mu, ell, p, e = 0.8, 1.3, 2.0, 0.4
    assert num(out["force"], mu=mu, l=ell, p=p, r=1.7) == approx(-(ell**2) / (mu * p * 1.7**2))
    assert out["binet_equation"].startswith("u'' + u = ")
    consts = {"mu": mu, "l": ell, "p": p, "e": e}
    th, rr = integrate_orbit(out["force"], consts, p / (1 + e), 2.5)
    assert np.allclose(rr, p / (1 + e * np.cos(th)), rtol=1e-8, atol=0)


def test_circle_through_the_centre_is_inverse_fifth_power() -> None:
    out = run(orbit="2*a*cos(theta)")
    assert out["verified"] is True
    mu, ell, a, r = 0.8, 1.3, 1.5, 1.1
    assert num(out["force"], mu=mu, l=ell, a=a, r=r) == approx(-8 * a * a * ell**2 / (mu * r**5))
    assert num(out["force_theta"], mu=mu, l=ell, a=a, theta=0.4) == approx(
        -8 * a * a * ell**2 / (mu * (2 * a * math.cos(0.4)) ** 5)
    )


def test_cardioid_and_spiral() -> None:
    card = run(orbit="a*(1 + cos(theta))")
    mu, ell, a, b, r = 0.8, 1.3, 1.5, 0.3, 1.1
    assert num(card["force"], mu=mu, l=ell, a=a, r=r) == approx(-3 * a * ell**2 / (mu * r**4))
    consts = {"mu": mu, "l": ell, "a": a}
    th, rr = integrate_orbit(card["force"], consts, 2 * a, 2.0)
    assert np.allclose(rr, a * (1 + np.cos(th)), rtol=1e-8, atol=0)
    spiral = run(orbit="a*exp(b*theta)")
    assert num(spiral["force"], mu=mu, l=ell, b=b, r=r) == approx(
        -(1 + b * b) * ell**2 / (mu * r**3)
    )


def test_circular_orbit_has_no_force_law() -> None:
    ctx = Ctx()
    out = run(ctx, orbit="a")
    assert out["force"] is None
    assert num(out["force_theta"], mu=0.8, l=1.3, a=1.5) == approx(-(1.3**2) / (0.8 * 1.5**3))
    assert any("circular" in w for w in ctx.warnings)


def test_inverse_square_force_gives_a_conic() -> None:
    out = run(force="-k/r**2")
    assert out["verified"] is True
    assert num(out["binet_equation"], u_ddot=0.3, u=0.2, k=2.0, mu=0.8, l=1.3) == approx(
        0.3 + 0.2 - 2.0 * 0.8 / 1.3**2
    )
    # parabola: u(0) = 2 k mu / l^2, u'(0) = 0 gives u = k mu / l^2 (1 + cos theta)
    out = run(force="-k/r**2", ics={"u(0)": "2*k*mu/l**2", "u'(0)": 0})
    k, mu, ell, th = 2.0, 0.8, 1.3, 0.9
    assert num(out["u"], k=k, mu=mu, l=ell, theta=th) == approx(
        k * mu / ell**2 * (1 + math.cos(th))
    )
    assert num(out["orbit"], k=k, mu=mu, l=ell, theta=th) == approx(
        ell**2 / (k * mu * (1 + math.cos(th)))
    )


def test_inverse_cube_force_is_solved() -> None:
    out = run(force="-k/r**3")
    assert out["verified"] is True
    assert "C1" in out["u"] and "C2" in out["u"]


def test_nonlinear_binet_equation_points_to_ode() -> None:
    with pytest.raises(ToolError, match="ode") as err:
        run(force="-k*r")
    assert err.value.code == "MATH_ERROR"


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({}, "orbit"),
        ({"orbit": "p/(1 + e*cos(theta))", "force": "-k/r**2"}, "one of"),
        ({"orbit": "r*cos(theta)"}, "r"),
        ({"force": "-k*cos(theta)/r**2"}, "theta"),
        ({"orbit": "p/(1 + e*cos(theta))", "ics": {"u(0)": 1}}, "ics"),
    ],
)
def test_errors(args: dict, match: str) -> None:
    with pytest.raises(ToolError, match=match):
        run(**args)
