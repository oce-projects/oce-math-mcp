"""D99: eng_calc.modern_physics mode angular_momentum (quantum angular momentum and magnetic moments).

Independent references: the Lande factor against the weak-field Zeeman shift <j m| L_z + |g_e| S_z |j m> summed
with the Clebsch-Gordan coefficients of sympy.physics.quantum.cg; magnitudes and moments from CODATA constants.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from sympy import Rational
from sympy.physics.quantum.cg import CG

from oce_math_mcp.constants import lookup
from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc

HBAR = lookup("reduced Planck constant")[1]
MU_B = lookup("Bohr magneton")[1]
G_E = lookup("electron g factor")[1]
E_CH = lookup("elementary charge")[1]


def run(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle(
        {"operation": "modern_physics", "params": {"mode": "angular_momentum", **params}},
        ctx or Ctx(),
    )


def zeeman_cg(ell: int, s: Any, j: Any, m: Any) -> float:
    """<j m | L_z + |g_e| S_z | j m> / hbar from the Clebsch-Gordan expansion of |j m>."""
    total = 0.0
    ms_values = [s - k for k in range(int(2 * s) + 1)]
    for ms in ms_values:
        ml = m - ms
        if abs(ml) > ell:
            continue
        c = float(CG(ell, ml, s, ms, j, m).doit())
        total += c * c * (float(ml) + abs(G_E) * float(ms))
    return total


def test_magnitudes_and_projections_p_electron() -> None:
    out = run({"l": 1, "s": 0.5, "n": 2})
    assert out["L"] == pytest.approx(HBAR * math.sqrt(2), rel=1e-14, abs=0)
    assert out["S"] == pytest.approx(HBAR * math.sqrt(0.75), rel=1e-14, abs=0)
    assert out["l_values"] == [0, 1]
    assert out["m_l"] == [-1, 0, 1]
    assert out["m_s"] == [-0.5, 0.5]
    assert out["L_z"] == pytest.approx([-HBAR, 0.0, HBAR], rel=1e-14, abs=0)
    assert out["S_z"] == pytest.approx([-HBAR / 2, HBAR / 2], rel=1e-14, abs=0)
    assert out["j_values"] == [0.5, 1.5]
    assert out["J"] == pytest.approx(
        [HBAR * math.sqrt(0.75), HBAR * math.sqrt(3.75)], rel=1e-14, abs=0
    )
    assert out["m_j"] == [[-0.5, 0.5], [-1.5, -0.5, 0.5, 1.5]]
    # magnetic moments: mu_l,z = -m_l mu_B, mu_s,z = g_e m_s mu_B (g_e < 0)
    assert out["mu_L"] == pytest.approx(MU_B * math.sqrt(2), rel=1e-14, abs=0)
    assert out["mu_l_z"] == pytest.approx([MU_B, 0.0, -MU_B], rel=1e-14, abs=0)
    assert out["mu_s_z"] == pytest.approx([-G_E * MU_B / 2, G_E * MU_B / 2], rel=1e-14, abs=0)
    assert out["mu_S"] == pytest.approx(abs(G_E) * MU_B * math.sqrt(0.75), rel=1e-14, abs=0)


@pytest.mark.parametrize(("ell", "s"), [(1, 0.5), (2, 0.5), (3, 1.5), (2, 1.0), (0, 0.5)])
def test_lande_factor_against_clebsch_gordan(ell: int, s: float) -> None:
    out = run({"l": ell, "s": s, "B": 0.5})
    srat = Rational(int(2 * s), 2)
    for k, j in enumerate(out["j_values"]):
        jr = Rational(int(2 * j), 2)
        if j == 0:
            assert out["g_J"][k] is None
            continue
        m = jr  # the stretched state |j, j>
        shift = zeeman_cg(ell, srat, jr, m)
        assert out["g_J"][k] == pytest.approx(shift / float(jr), rel=1e-12, abs=0), (ell, s, j)
        # weak-field Zeeman energies g_J m_j mu_B B
        want = [out["g_J"][k] * mj * MU_B * 0.5 for mj in out["m_j"][k]]
        assert out["zeeman"][k] == pytest.approx(want, rel=1e-14, abs=0)


def test_zeeman_energies_of_orbital_and_spin_moments() -> None:
    out = run({"l": 2, "s": 0.5, "B": 1.2})
    # U = -mu_z B
    assert out["U_l"] == pytest.approx([ml * MU_B * 1.2 for ml in range(-2, 3)], rel=1e-14, abs=0)
    assert out["U_s"] == pytest.approx(
        [-G_E * ms * MU_B * 1.2 for ms in (-0.5, 0.5)], rel=1e-14, abs=0
    )
    assert out["zeeman_eV"][1] == pytest.approx(
        [e / E_CH for e in out["zeeman"][1]], rel=1e-14, abs=0
    )


def test_one_j_and_other_spins() -> None:
    out = run({"l": 1, "s": 0.5, "j": 1.5})
    assert out["j_values"] == [1.5]
    assert len(out["m_j"]) == 1
    # total spin of two electrons (LS coupling, term 3P): the spin moments use g_e for each unit of M_S
    out = run({"l": 1, "s": 1.0})
    assert out["j_values"] == [0.0, 1.0, 2.0]
    assert out["mu_s_z"] == pytest.approx([G_E * ms * MU_B for ms in (-1, 0, 1)], rel=1e-14, abs=0)
    assert out["g_J"][0] is None


def test_invalid_quantum_numbers() -> None:
    for bad in (
        {"l": 1.5},
        {"l": -1},
        {"l": 1, "s": 0.3},
        {"l": 1, "s": 0.25},
        {"l": 2, "n": 2},
        {"l": 1, "j": 1.0},
    ):
        with pytest.raises(ToolError):
            run(bad)
    with pytest.raises(ToolError):
        run({"l": 2000})
    with pytest.raises(ToolError):
        run({"l": 1, "s": 1500})
