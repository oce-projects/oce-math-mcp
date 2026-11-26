"""D102: math_solve.lagrangian (Euler-Lagrange equations, generalized momenta, Hamiltonian, action).

References: textbook equations of motion (oscillator, pendulum, Kepler problem in polar coordinates, charged
particle in a uniform magnetic field, relativistic free particle), the Hamiltonian as the Legendre transform
max over v of (p v - L) found here with scipy, and the action integral with mpmath quad.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

import mpmath
import numpy as np
import pytest
import sympy as sp
from scipy.optimize import minimize, minimize_scalar

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.expr import parse
from oce_math_mcp.tools import math_solve
from oce_math_mcp.tools.common import grammar

QUAD: Any = mpmath.quad  # the mpmath stubs type quad as a tuple


def run(ctx: Ctx | None = None, **args: Any) -> dict:
    return math_solve.handle({"operation": "lagrangian", **args}, ctx or Ctx())


def num(text: str, **values: float) -> float:
    """Value of a result string: x'' -> x_ddot and x' -> x_dot, read with the safe parser, 30 digits."""
    text = re.sub(r"([A-Za-z]\w*)''", r"\1_ddot", text)
    text = re.sub(r"([A-Za-z]\w*)'", r"\1_dot", text)
    if "=" in text:
        lhs, rhs = text.split("=")
        text = f"({lhs}) - ({rhs})"
    e = parse(text, grammar())
    subs = {s: sp.Float(repr(values[s.name]), 30) for s in e.free_symbols}
    z = complex(sp.N(e.subs(subs), 30))
    assert abs(z.imag) <= 1e-25 * (1 + abs(z.real))
    return z.real


def approx(v: float, rel: float = 1e-12) -> Any:
    return pytest.approx(v, rel=rel, abs=1e-14)


def legendre_1d(lag: Any, p: float, lo: float, hi: float) -> float:
    """H = max over v of (p v - L(v)), with scipy (convex L in v)."""
    res = minimize_scalar(
        lambda v: -(p * v - lag(v)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-13}
    )
    return -float(res.fun)


def test_harmonic_oscillator() -> None:
    out = run(expr="m*x'**2/2 - k*x**2/2", variables=["x"])
    assert out["verified"] is True
    assert out["coordinates"] == ["x"]
    m, k, x, v, acc, p = 1.3, 2.7, 0.4, -0.9, 0.35, 0.8
    assert out["equations"][0].endswith("= 0")
    assert num(out["equations"][0], m=m, k=k, x=x, x_ddot=acc) == approx(m * acc + k * x)
    assert num(out["momenta"]["p_x"], m=m, x_dot=v) == approx(m * v)
    assert num(out["energy_function"], m=m, k=k, x=x, x_dot=v) == approx(
        m * v * v / 2 + k * x * x / 2
    )
    assert num(out["hamiltonian"], m=m, k=k, x=x, p_x=p) == approx(p * p / (2 * m) + k * x * x / 2)
    he = out["hamilton_equations"]
    assert num(he["x'"], m=m, p_x=p) == approx(p / m)
    assert num(he["p_x'"], k=k, x=x) == approx(-k * x)
    assert out["energy_conserved"] is True
    assert out["cyclic_coordinates"] == []


def test_pendulum_against_textbook_and_legendre() -> None:
    out = run(expr="m*l**2*theta'**2/2 + m*g*l*cos(theta)", variables=["theta"], domain="positive")
    assert out["verified"] is True
    m, ell, g, th, acc, p = 0.7, 1.9, 9.81, 0.6, -1.2, 0.9
    eq = num(out["equations"][0], m=m, l=ell, g=g, theta=th, theta_ddot=acc)
    assert eq == approx(m * ell * ell * acc + m * g * ell * math.sin(th))
    h = num(out["hamiltonian"], m=m, l=ell, g=g, theta=th, p_theta=p)
    ref = legendre_1d(
        lambda v: m * ell * ell * v * v / 2 + m * g * ell * math.cos(th), p, -50.0, 50.0
    )
    assert h == approx(ref, 1e-9)


def test_kepler_polar_cyclic_angle() -> None:
    out = run(expr="m*(r'**2 + r**2*phi'**2)/2 + k/r", variables=["r", "phi"], domain="positive")
    assert out["verified"] is True
    assert out["cyclic_coordinates"] == ["phi"]
    m, k, r, rd, phd, rdd = 1.1, 3.0, 1.7, 0.2, 0.8, -0.4
    vals = {"m": m, "k": k, "r": r, "r_dot": rd, "phi_dot": phd, "r_ddot": rdd}
    assert num(out["equations"][0], **vals) == approx(m * rdd - m * r * phd * phd + k / r**2)
    assert num(out["momenta"]["p_phi"], m=m, r=r, phi_dot=phd) == approx(m * r * r * phd)
    pr, pphi = 0.5, 1.3
    h = num(out["hamiltonian"], m=m, k=k, r=r, p_r=pr, p_phi=pphi)
    assert h == approx(pr**2 / (2 * m) + pphi**2 / (2 * m * r * r) - k / r)
    assert num(out["hamilton_equations"]["p_phi'"], m=m, k=k, r=r, p_r=pr, p_phi=pphi) == 0


def test_charged_particle_negative_charge() -> None:
    out = run(expr="m*(x'**2 + y'**2)/2 + q*B*(x*y' - y*x')/2", variables=["x", "y"])
    assert out["verified"] is True
    m, q, b = 0.9, -1.6, 2.2
    x, y, xd, yd, xdd, ydd = 0.3, -0.7, 1.1, 0.4, -0.2, 0.6
    vals = {"m": m, "q": q, "B": b, "x": x, "y": y, "x_dot": xd, "y_dot": yd}
    vals |= {"x_ddot": xdd, "y_ddot": ydd}
    assert num(out["equations"][0], **vals) == approx(m * xdd - q * b * yd)
    assert num(out["equations"][1], **vals) == approx(m * ydd + q * b * xd)
    px, py = 0.5, -0.8
    h = num(out["hamiltonian"], m=m, q=q, B=b, x=x, y=y, p_x=px, p_y=py)
    textbook = ((px + q * b * y / 2) ** 2 + (py - q * b * x / 2) ** 2) / (2 * m)
    assert h == approx(textbook)

    def minus(v: Any) -> float:
        lag = m * (v[0] ** 2 + v[1] ** 2) / 2 + q * b * (x * v[1] - y * v[0]) / 2
        return -(px * v[0] + py * v[1] - lag)

    res = minimize(minus, np.zeros(2), method="BFGS", options={"gtol": 1e-12})
    assert h == approx(-float(res.fun), 1e-9)


def test_relativistic_free_particle() -> None:
    out = run(expr="-m*c**2*sqrt(1 - x'**2/c**2)", variables=["x"], domain="positive")
    assert out["verified"] is True
    m, c, v, p = 1.2, 2.5, 1.9, 3.1
    assert num(out["momenta"]["p_x"], m=m, c=c, x_dot=v) == approx(
        m * v / math.sqrt(1 - v * v / c / c)
    )
    h = num(out["hamiltonian"], m=m, c=c, p_x=p)
    assert h == approx(math.sqrt(p * p * c * c + m * m * c**4))
    ref = legendre_1d(lambda w: -m * c * c * math.sqrt(1 - w * w / c / c), p, -c, c)
    assert h == approx(ref, 1e-9)


def test_driven_oscillator_energy_not_conserved() -> None:
    out = run(expr="m*x'**2/2 - k*x**2/2 + F0*x*cos(w*t)", variables=["x"])
    assert out["energy_conserved"] is False
    vals = {"m": 1.3, "k": 2.7, "F0": 0.6, "w": 1.7, "t": 0.9, "x": 0.4, "x_ddot": 0.35}
    ref = 1.3 * 0.35 + 2.7 * 0.4 - 0.6 * math.cos(1.7 * 0.9)
    assert num(out["equations"][0], **vals) == approx(ref)


def test_double_pendulum_textbook_and_time() -> None:
    """D102: simplify of the double pendulum H took 2.44 s and the call 12.08 s, over the 5 s limit per call."""
    lag = (
        "(m1 + m2)*l1**2*a'**2/2 + m2*l2**2*b'**2/2 + m2*l1*l2*a'*b'*cos(a - b) + (m1 + m2)*g*l1*cos(a)"
        " + m2*g*l2*cos(b)"
    )
    t0 = time.perf_counter()
    out = run(expr=lag, variables=["a", "b"], domain="positive")
    assert time.perf_counter() - t0 < 5
    assert out["verified"] is True
    m1, m2, l1, l2, g = 1.2, 0.7, 1.5, 0.9, 9.81
    a, b, ad, bd, add, bdd = 0.4, -0.3, 0.8, -1.1, 0.25, -0.6
    vals = {"m1": m1, "m2": m2, "l1": l1, "l2": l2, "g": g, "a": a, "b": b}
    vals |= {"a_dot": ad, "b_dot": bd, "a_ddot": add, "b_ddot": bdd}
    # textbook: (m1 + m2) l1^2 a'' + m2 l1 l2 b'' cos(a - b) + m2 l1 l2 b'^2 sin(a - b) + (m1 + m2) g l1 sin a
    ref = (
        (m1 + m2) * l1 * l1 * add
        + m2 * l1 * l2 * bdd * math.cos(a - b)
        + m2 * l1 * l2 * bd * bd * math.sin(a - b)
        + (m1 + m2) * g * l1 * math.sin(a)
    )
    assert num(out["equations"][0], **vals) == approx(ref)


def test_caldirola_kanai_time_in_the_momentum() -> None:
    """L = exp(g t) (m x'^2/2 - k x^2/2): the damped oscillator m x'' + m g x' + k x = 0 (times exp(g t))."""
    out = run(expr="exp(g*t)*(m*x'**2/2 - k*x**2/2)", variables=["x"])
    assert out["verified"] is True
    assert out["energy_conserved"] is False
    m, k, g, t, x, v, acc, p = 1.3, 2.7, 0.4, 0.9, 0.4, -0.9, 0.35, 0.8
    vals = {"m": m, "k": k, "g": g, "t": t, "x": x, "x_dot": v, "x_ddot": acc}
    ref = math.exp(g * t) * (m * acc + m * g * v + k * x)
    assert num(out["equations"][0], **vals) == approx(ref)
    h = num(out["hamiltonian"], m=m, k=k, g=g, t=t, x=x, p_x=p)
    assert h == approx(p * p * math.exp(-g * t) / (2 * m) + k * x * x * math.exp(g * t) / 2)


def test_several_velocities_for_one_momentum_give_no_hamiltonian() -> None:
    """p = x'^3 - x' has 3 real roots for small p: H(q, p) is not one function."""
    ctx = Ctx()
    out = run(ctx, expr="x'**4/4 - x'**2/2", variables=["x"])
    assert out["hamiltonian"] is None
    assert out["hamilton_equations"] is None
    assert any("not one function" in w for w in ctx.warnings)
    assert num(out["equations"][0], x_dot=0.7, x_ddot=0.3) == approx((3 * 0.49 - 1) * 0.3)


def test_kinetic_and_potential_fields_give_l_equal_t_minus_v() -> None:
    a = run(kinetic_energy="m*(x'**2 + y'**2)/2", potential_energy="m*g*y", variables=["x", "y"])
    b = run(expr="m*(x'**2 + y'**2)/2 - m*g*y", variables=["x", "y"])
    assert a["equations"] == b["equations"]
    assert a["hamiltonian"] == b["hamiltonian"]
    assert a["cyclic_coordinates"] == ["x"]
    assert num(a["lagrangian"], m=2.0, g=9.8, x=0.1, y=0.5, x_dot=0.3, y_dot=-0.2) == approx(
        2.0 * (0.09 + 0.04) / 2 - 2.0 * 9.8 * 0.5
    )


def test_action_free_particle_and_oscillator() -> None:
    out = run(expr="m*x'**2/2", variables=["x"], path={"x": "v0*t"}, bounds=[0, "T"])
    assert num(out["action"]["value"], m=1.5, v0=2.0, T=3.0) == approx(1.5 * 4.0 * 3.0 / 2)
    assert out["action"]["stationary"] is True
    assert out["verified"] is True

    lag = "m*x'**2/2 - k*x**2/2"
    good = run(expr=lag, variables=["x"], path={"x": "A*cos(sqrt(k/m)*t)"}, bounds=[0, 2])
    assert good["action"]["stationary"] is True
    assert good["verified"] is True
    m, k, amp = 1.3, 2.7, 0.5
    w = math.sqrt(k / m)

    def lag_path(t: Any) -> Any:
        return m * (amp * w * mpmath.sin(w * t)) ** 2 / 2 - k * (amp * mpmath.cos(w * t)) ** 2 / 2

    with mpmath.workdps(30):
        ref = float(QUAD(lag_path, [0, 2]))
    assert num(good["action"]["value"], m=m, k=k, A=amp) == approx(ref)

    bad = run(expr=lag, variables=["x"], path={"x": "A*t"}, bounds=[0, 2])
    assert bad["action"]["stationary"] is False
    assert num(bad["action"]["residuals"][0], k=k, A=amp, t=0.7) == approx(k * amp * 0.7)


def test_singular_lagrangian_has_no_hamiltonian() -> None:
    ctx = Ctx()
    out = run(ctx, expr="x*y' - y*x' - x**2 - y**2", variables=["x", "y"])
    assert out["hamiltonian"] is None
    assert out["hamilton_equations"] is None
    assert any("singular" in w for w in ctx.warnings)
    vals = {"x": 0.3, "y": -0.4, "x_dot": 0.6, "y_dot": 0.2}
    assert num(out["equations"][0], **vals) == approx(-2 * 0.2 + 2 * 0.3)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({"variables": ["x"]}, "expr"),
        ({"expr": "x'**2", "kinetic_energy": "x'**2", "variables": ["x"]}, "expr"),
        ({"expr": "x'**2", "variables": ["t"]}, "time"),
        ({"expr": "x'**2", "variables": []}, "variables"),
        ({"expr": "x'**2 + p_x*x", "variables": ["x"]}, "p_x"),
        ({"expr": "x''**2 + x'**2", "variables": ["x"]}, "x''"),
        ({"expr": "x'**2", "variables": ["x"], "path": {"x": "t"}}, "bounds"),
        (
            {
                "expr": "x'**2 + y'**2",
                "variables": ["x", "y"],
                "path": {"x": "t"},
                "bounds": [0, 1],
            },
            "y",
        ),
        ({"expr": "x'**2", "variables": list("abcdefg")}, "6"),
        ({"expr": "x'**2", "potential_energy": "x**2", "variables": ["x"]}, "expr"),
    ],
)
def test_errors(args: dict, match: str) -> None:
    with pytest.raises(ToolError, match=match):
        run(**args)
