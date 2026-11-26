"""D99: math_solve.wavefunction (density, current, probability, momentum space, Schroedinger check).

References: textbook closed forms (Gaussian packet, infinite well, harmonic oscillator) and mpmath integrals and
derivatives written here in plain Python, not with SymPy.
"""

from __future__ import annotations

from typing import Any

import mpmath
import pytest
import sympy as sp

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.expr import parse
from oce_math_mcp.tools import math_solve
from oce_math_mcp.tools.common import grammar

GAUSS = "(2*a/pi)**(1/4)*exp(-a*x**2 + I*k*x)"


def run(expr: str, variables: list[str], **more: Any) -> dict:
    args = {"operation": "wavefunction", "expr": expr, "variables": variables, **more}
    return math_solve.handle(args, Ctx())


def num(text: str, **values: float) -> complex:
    """Value of a result string at the given numbers: read with the safe parser, evaluated at 30 digits."""
    e = parse(text, grammar())
    subs = {s: sp.Float(repr(values[s.name]), 30) for s in e.free_symbols}
    return complex(sp.N(e.subs(subs), 30))


def test_gaussian_packet_density_current_probability() -> None:
    out = run(GAUSS, ["x"], bounds=[["x", "-oo", "oo"]])
    assert out["verified"] is True
    a, k, x, hbar, m = 0.7, 2.0, 0.3, 1.1, 0.9
    rho = num(out["density"], a=a, k=k, x=x).real
    assert rho == pytest.approx(
        float(mpmath.sqrt(2 * a / mpmath.pi) * mpmath.exp(-2 * a * x * x)), rel=1e-13, abs=0
    )
    j = num(out["current"][0], a=a, k=k, x=x, hbar=hbar, m=m).real
    assert j == pytest.approx(hbar * k / m * rho, rel=1e-13, abs=0)
    assert out["probability"] == "1"


def test_gaussian_packet_in_momentum_space() -> None:
    out = run(GAUSS, ["x"], momentum_space=True)
    assert out["verified"] is True
    a, k, hbar, p = 0.7, 2.0, 1.1, 1.9
    with mpmath.workdps(30):
        want = mpmath.quad(
            lambda x: (
                mpmath.exp(-1j * p * x / hbar)
                * (2 * a / mpmath.pi) ** 0.25
                * mpmath.exp(-a * x * x + 1j * k * x)
            ),
            [-mpmath.inf, mpmath.inf],
        ) / mpmath.sqrt(2 * mpmath.pi * hbar)
    got = num(out["momentum_wavefunction"], a=a, k=k, hbar=hbar, p_x=p)
    assert abs(got - complex(want)) < 1e-12 * abs(complex(want))
    assert out["momentum_variables"] == ["p_x"]


def test_infinite_well_eigenstate_energy() -> None:
    out = run("sqrt(2/L)*sin(2*pi*x/L)", ["x"], potential_energy="0", bounds=[["x", "0", "L"]])
    assert out["verified"] is True
    assert out["probability"] == "1"
    sch = out["schrodinger"]
    assert sch["eigenstate"] is True
    e = num(sch["energy"], hbar=1.1, m=0.9, L=2.0, pi=mpmath.pi).real
    assert e == pytest.approx(float(4 * mpmath.pi**2 * 1.1**2 / (2 * 0.9 * 4.0)), rel=1e-13, abs=0)
    # x exp(-x^2) is not an eigenstate of the free Hamiltonian
    out = run("x*exp(-x**2)", ["x"], potential_energy="0")
    assert out["schrodinger"]["eigenstate"] is False


def test_harmonic_oscillator_time_dependent_schroedinger() -> None:
    psi = "(m*w/(pi*hbar))**(1/4)*exp(-m*w*x**2/(2*hbar))*exp(-I*w*t/2)"
    out = run(psi, ["x"], potential_energy="m*w**2*x**2/2", bounds=[["x", "-oo", "oo"]])
    assert out["verified"] is True
    assert out["schrodinger"]["satisfied"] is True
    assert out["schrodinger"]["residual"] == "0"
    assert out["probability"] == "1"
    # a wrong frequency in the phase does not satisfy it
    out = run(psi.replace("w*t/2", "w*t"), ["x"], potential_energy="m*w**2*x**2/2")
    assert out["schrodinger"]["satisfied"] is False


def test_product_state_in_three_dimensions() -> None:
    one = "(2*a/pi)**(1/4)*exp(-a*{v}**2)"
    psi = "*".join(one.format(v=v) for v in "xyz")
    out = run(psi, ["x", "y", "z"], bounds=[[v, "-oo", "oo"] for v in "xyz"])
    assert out["verified"] is True
    assert out["probability"] == "1"
    assert len(out["current"]) == 3 and all(c == "0" for c in out["current"])
    # probability of the octant x, y, z > 0 = 1/8
    out = run(psi, ["x", "y", "z"], bounds=[[v, "0", "oo"] for v in "xyz"])
    assert out["probability"] == "1/8" and out["verified"] is True
    # finite box in 2D (particle in a square well): whole box and one quarter
    box = "(2/L)*sin(pi*x/L)*sin(pi*y/L)"
    out = run(box, ["x", "y"], bounds=[["x", "0", "L"], ["y", "0", "L"]])
    assert out["probability"] == "1" and out["verified"] is True
    out = run(box, ["x", "y"], bounds=[["x", "0", "L/2"], ["y", "0", "L/2"]])
    assert out["probability"] == "1/4" and out["verified"] is True
    # half-infinite on the left, off-centre Gaussian (a symmetric one hides a mirrored grid)
    shifted = "(2*a/pi)**(1/4)*exp(-a*(x - 1)**2)"
    out = run(shifted, ["x", "y"][:1], bounds=[["x", "-oo", "0"]])
    assert out["verified"] is True
    two = shifted + "*" + one.format(v="y")
    out = run(two, ["x", "y"], bounds=[["x", "-oo", "0"], ["y", "-oo", "oo"]])
    assert out["verified"] is True and out["probability"] != "1/2"
    # four dimensions: closed form given, numerical check not done, so not verified (D90)
    psi4 = "*".join(one.format(v=v) for v in ("x", "y", "z", "w"))
    ctx = Ctx()
    args = {"operation": "wavefunction", "expr": psi4, "variables": ["x", "y", "z", "w"]}
    args["bounds"] = [[v, "-oo", "oo"] for v in ("x", "y", "z", "w")]
    out = math_solve.handle(args, ctx)
    assert out["probability"] == "1" and out["verified"] is False
    assert any("up to 3 dimensions" in w for w in ctx.warnings)


def test_errors() -> None:
    with pytest.raises(ToolError):
        math_solve.handle({"operation": "wavefunction", "expr": "exp(-x**2)"}, Ctx())
    with pytest.raises(ToolError):
        run("exp(-x**2)", ["x"], bounds=[["y", "0", "1"]])
    with pytest.raises(ToolError):
        run("exp(-x**2)", ["t"])  # t is the time
    with pytest.raises(ToolError):
        run("exp(-x**2)", [f"x{i}" for i in range(7)])
