"""D118: math_solve.wavefunction with one mass per space variable (argument `masses`).

References: page "Schroedinger equation", section "Time evolution" (multi-particle form):
H = sum_n p_n^2 / (2 m_n) + V = -(hbar^2 / 2) sum_n (1 / m_n) d^2/dx_n^2 + V, and the probability current
j_n = (hbar / m_n) Im(Psi* dPsi/dx_n). Expected values are written here in plain Python (math), and the result
strings are read back with the safe parser (as D99).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import sympy as sp

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.expr import parse
from oce_math_mcp.tools import math_solve
from oce_math_mcp.tools.common import grammar


def run(expr: str, variables: list[str], **more: Any) -> dict[str, Any]:
    args = {"operation": "wavefunction", "expr": expr, "variables": variables, **more}
    return math_solve.handle(args, Ctx())


def num(text: str, **values: float) -> complex:
    e = parse(text, grammar())
    subs = {s: sp.Float(repr(values[s.name]), 30) for s in e.free_symbols}
    return complex(sp.N(e.subs(subs), 30))


def test_the_measure_two_masses() -> None:
    """RAG doc 62, MQ 10 and 12: H Psi with m1 = 1, m2 = 2, hbar = 1 at x1 = 0.5, x2 = 0.3 (V = 0)."""
    out = run("exp(-x1**2/2 - x2**2/2)", ["x1", "x2"], masses=["m1", "m2"], potential_energy="0")
    assert out["verified"] is True
    assert out["masses"] == ["m1", "m2"]
    psi = math.exp(-(0.5**2) / 2 - 0.3**2 / 2)
    want = -0.5 * ((0.5**2 - 1) * psi / 1 + (0.3**2 - 1) * psi / 2)  # the value of the measure
    e = num(out["schrodinger"]["energy"], hbar=1.0, m1=1.0, m2=2.0, x1=0.5, x2=0.3)
    assert abs(e.real * psi - want) < 1e-14 * abs(want)
    assert abs(e.imag) < 1e-30
    assert out["schrodinger"]["eigenstate"] is False


OSC = "exp(-m1*w*x1**2/(2*hbar) - m2*w*x2**2/(2*hbar))"
OSC_V = "m1*w**2*x1**2/2 + m2*w**2*x2**2/2"


def test_two_mass_oscillator_ground_state() -> None:
    """Two oscillators with the same w and masses m1, m2: ground state with E = hbar w (1/2 for each)."""
    out = run(OSC, ["x1", "x2"], masses=["m1", "m2"], potential_energy=OSC_V)
    assert out["schrodinger"] == {"energy": "hbar*w", "eigenstate": True}
    assert out["verified"] is True
    # with the single mass m of D99 the same Psi is not an eigenstate
    old = run(OSC, ["x1", "x2"], potential_energy=OSC_V)
    assert old["schrodinger"]["eigenstate"] is False
    assert "masses" not in old


def test_two_mass_oscillator_time_dependent() -> None:
    """MQ 26, 28: i hbar dPsi/dt = H Psi for Psi exp(-i w t)."""
    out = run(f"({OSC})*exp(-I*w*t)", ["x1", "x2"], masses=["m1", "m2"], potential_energy=OSC_V)
    assert out["schrodinger"] == {"residual": "0", "satisfied": True}
    assert out["verified"] is True


def test_current_has_one_mass_per_variable() -> None:
    out = run("exp(I*(k1*x1 + k2*x2))", ["x1", "x2"], masses=["m1", "m2"])
    assert out["verified"] is True
    hbar, k1, k2, m1, m2 = 1.1, 2.0, 0.7, 0.9, 3.0
    vals = {"hbar": hbar, "k1": k1, "k2": k2, "m1": m1, "m2": m2, "x1": 0.4, "x2": -0.2}
    j1, j2 = (num(c, **vals).real for c in out["current"])
    assert j1 == pytest.approx(hbar * k1 / m1, rel=1e-14, abs=0)
    assert j2 == pytest.approx(hbar * k2 / m2, rel=1e-14, abs=0)


def test_two_particles_in_three_dimensions() -> None:
    """MQ 18, 21: nabla_n^2 for each particle, 3 coordinates per mass; E = 3 hbar w for two 3D oscillators."""
    r1 = "x1**2 + y1**2 + z1**2"
    r2 = "x2**2 + y2**2 + z2**2"
    psi = f"exp(-m1*w*({r1})/(2*hbar) - m2*w*({r2})/(2*hbar))"
    pot = f"m1*w**2*({r1})/2 + m2*w**2*({r2})/2"
    variables = ["x1", "y1", "z1", "x2", "y2", "z2"]
    out = run(psi, variables, masses=["m1"] * 3 + ["m2"] * 3, potential_energy=pot)
    assert out["schrodinger"] == {"energy": "3*hbar*w", "eigenstate": True}
    assert out["verified"] is True


def test_a_mass_can_be_an_expression() -> None:
    out = run("exp(I*(k1*x1 + k2*x2))", ["x1", "x2"], masses=["m", "2*m"])
    assert out["masses"] == ["m", "2*m"]
    vals = {"hbar": 1.3, "k1": 0.5, "k2": 0.8, "m": 1.7, "x1": 0.1, "x2": 0.2}
    j1, j2 = (num(c, **vals).real for c in out["current"])
    assert j1 == pytest.approx(1.3 * 0.5 / 1.7, rel=1e-14, abs=0)
    assert j2 == pytest.approx(1.3 * 0.8 / (2 * 1.7), rel=1e-14, abs=0)
    assert out["verified"] is True


def test_the_same_mass_everywhere_is_the_old_result() -> None:
    expr = "exp(-a*(x**2 + y**2) + I*k*x)"
    new = run(expr, ["x", "y"], masses=["m", "m"], potential_energy="0")
    old = run(expr, ["x", "y"], potential_energy="0")
    assert new.pop("masses") == ["m", "m"]
    assert new == old


@pytest.mark.parametrize(
    ("masses", "words"),
    [
        ("m1", "list"),
        (["m1"], "one mass for each"),
        (["m1", "m2", "m3"], "one mass for each"),
        (["m1", 2], "Each mass is a string"),  # not the generic "The expression must be a string."
        (["m1", "x2"], "space variable"),
        (["m1", "t"], "space variable"),
        (["m1", "-m2"], "positive"),
        (["m1", "0"], "positive"),
        (["m1", "m2 - 1"], "positive"),
    ],
)
def test_errors(masses: Any, words: str) -> None:
    with pytest.raises(ToolError) as err:
        run("exp(-x1**2 - x2**2)", ["x1", "x2"], masses=masses)
    assert err.value.code == "INVALID_ARGUMENT"
    assert words in err.value.message
    assert err.value.example == '["m1", "m2"]'
