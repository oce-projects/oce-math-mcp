"""D112: eng_calc.statistical_mechanics modes density_of_states and occupation.

References independent of the module: the page formulas (List of equations in quantum mechanics, "Quantum
uncertainty": N(E) = 8 sqrt(2) pi m^(3/2) E^(1/2) / h^3; Fermi energy page: E_F = hbar^2 / (2 m) (3 pi^2 n)^(2/3)),
mpmath integrals of g(E) at 40 digits and mpmath occupation numbers at 60 digits, also where floats overflow.
"""

from __future__ import annotations

import math
from typing import Any

import mpmath
import pytest

from oce_math_mcp.constants import lookup
from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc

K = lookup("Boltzmann constant")[1]
H = lookup("Planck constant")[1]
HBAR = lookup("reduced Planck constant")[1]
M_E = lookup("electron mass")[1]
M_N = lookup("neutron mass")[1]
EV = lookup("elementary charge")[1]


def run(params: dict[str, Any]) -> tuple[dict[str, Any], Ctx]:
    ctx = Ctx()
    out = eng_calc.handle({"operation": "statistical_mechanics", "params": params}, ctx)
    return out, ctx


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def g_page(m: float, e: float) -> Any:
    """Page formula of the density of states per volume and energy (spin 1/2)."""
    with mpmath.workdps(40):
        return (
            8
            * mpmath.sqrt(2)
            * mpmath.pi
            * mpmath.mpf(m) ** 1.5
            * mpmath.sqrt(e)
            / mpmath.mpf(H) ** 3
        )


def test_density_of_states_page_formula_and_states_below() -> None:
    e = 1.0 * EV
    out, ctx = run({"mode": "density_of_states", "particle": "electron", "E": [1, "eV"]})
    assert rel(out["g_E"], float(g_page(M_E, e))) < 1e-13
    with mpmath.workdps(40):
        below: Any = mpmath.quad(lambda x: g_page(M_E, x), [0, e])
    assert rel(out["N_below"], float(below)) < 1e-12
    assert out["g_s"] == 2
    assert ctx.formula is not None and "8 sqrt(2) pi" in ctx.formula
    assert ctx.units is not None
    assert ctx.units["g_E"] == "1/(J*m3)"
    assert ctx.units["N_below"] == "1/m3"


def test_density_of_states_volume_and_degeneracy() -> None:
    base, _ = run({"mode": "density_of_states", "m": M_N, "E": [2, "MeV"]})
    out, _ = run({"mode": "density_of_states", "m": M_N, "E": [2, "MeV"], "V": 1e-42, "g_s": 4})
    assert rel(out["g_E"], 2 * base["g_E"]) < 1e-14
    assert rel(out["g_E_total"], 2 * base["g_E"] * 1e-42) < 1e-14
    assert rel(out["N_below_total"], 2 * base["N_below"] * 1e-42) < 1e-14


def test_fermi_energy_from_density_and_integral() -> None:
    n = 8.47e28  # electrons per m3, inside the page range 1e28 - 1e29
    out, _ = run({"mode": "density_of_states", "particle": "electron", "n": n})
    with mpmath.workdps(40):
        ef = (
            mpmath.mpf(HBAR) ** 2
            / (2 * mpmath.mpf(M_E))
            * (3 * mpmath.pi**2 * n) ** (mpmath.mpf(2) / 3)
        )
        filled: Any = mpmath.quad(lambda x: g_page(M_E, x), [0, ef])
    assert rel(out["E_F"], float(ef)) < 1e-13
    assert rel(float(filled), n) < 1e-12  # the states below E_F hold exactly n electrons
    assert 2 <= out["E_F_eV"] <= 10  # page: "of the order of 2 to 10 electronvolts"
    assert rel(out["T_F"], float(ef) / K) < 1e-13
    assert rel(out["k_F"], (3 * math.pi**2 * n) ** (1 / 3)) < 1e-13
    assert rel(out["g_E_F"], 1.5 * n / float(ef)) < 1e-12  # g(E_F) = 3 n / (2 E_F)


def test_fermi_energy_general_degeneracy() -> None:
    n = 1e44
    out, ctx = run({"mode": "density_of_states", "particle": "neutron", "n": n, "g_s": 4, "V": 2.0})
    with mpmath.workdps(40):
        ef = (
            mpmath.mpf(HBAR) ** 2
            / (2 * mpmath.mpf(M_N))
            * (6 * mpmath.pi**2 * n / 4) ** (mpmath.mpf(2) / 3)
        )
    assert rel(out["E_F"], float(ef)) < 1e-13
    assert rel(out["g_E_F"], 1.5 * n / float(ef)) < 1e-12  # g(E_F) = 3 n / (2 E_F) for any g_s
    assert out["N_total"] == 2e44
    assert rel(out["g_E_F_total"], 2 * out["g_E_F"]) < 1e-14
    assert any(
        "of m c^2" in w for w in ctx.warnings
    )  # E_F = 26.9 MeV, 2.86% of the neutron rest energy


def test_relativistic_warning() -> None:
    # n = 1e36 electrons per m3: E_F is a large part of m c^2 (white dwarf matter)
    _, ctx = run({"mode": "density_of_states", "particle": "electron", "n": 1e36})
    assert any("of m c^2" in w for w in ctx.warnings)
    _, ctx2 = run({"mode": "density_of_states", "particle": "electron", "E": [10, "keV"]})
    assert any("of m c^2" in w for w in ctx2.warnings)
    _, ctx3 = run({"mode": "density_of_states", "particle": "electron", "E": [4, "keV"]})
    assert ctx3.warnings == []  # 4 keV is 0.78% of m c^2 (10 keV: 1.96%)


def mp_occ_mp(stat: str, x: float) -> Any:
    with mpmath.workdps(60):
        xm = mpmath.mpf(x)
        if stat == "fermi_dirac":
            return 1 / (mpmath.exp(xm) + 1)
        if stat == "bose_einstein":
            return 1 / (mpmath.exp(xm) - 1)
        return mpmath.exp(-xm)


def mp_occ(stat: str, x: float) -> float:
    return float(mp_occ_mp(stat, x))


def mp_ln_occ(stat: str, x: float) -> float:
    with mpmath.workdps(60):
        return float(mpmath.log(mp_occ_mp(stat, x)))


STATS = ["fermi_dirac", "bose_einstein", "maxwell_boltzmann"]


@pytest.mark.parametrize("stat", STATS)
@pytest.mark.parametrize("x", [1e-12, 1e-6, 0.3, 3.8682, 40.0, 690.0])
def test_occupation_against_mpmath(stat: str, x: float) -> None:
    t = 300.0
    de = x * K * t
    out, ctx = run({"mode": "occupation", "statistics": stat, "E": de, "mu": 0.0, "T": t})
    xs = de / (K * t)
    assert rel(out["n"], mp_occ(stat, xs)) < 1e-13
    assert rel(out["ln_n"], mp_ln_occ(stat, xs)) < 1e-13
    assert rel(out["x"], xs) < 1e-14  # outputs are rounded to 15 significant digits
    assert ctx.units is not None and ctx.units["n"] == "1"
    classical_bad = stat == "maxwell_boltzmann" and x < math.log(10)  # n > 0.1
    assert any("n > 0.1" in w for w in ctx.warnings) == classical_bad


@pytest.mark.parametrize("stat", STATS)
def test_occupation_underflow_keeps_ln(stat: str) -> None:
    t = 300.0
    out, ctx = run({"mode": "occupation", "statistics": stat, "E": 800 * K * t, "mu": 0.0, "T": t})
    assert "n" not in out
    assert rel(out["ln_n"], -800.0) < 1e-14
    assert any("beyond" in w for w in ctx.warnings)


def test_occupation_fermi_dirac_below_mu_and_at_mu() -> None:
    t = 300.0
    out, _ = run(
        {"mode": "occupation", "statistics": "fermi_dirac", "E": [5, "eV"], "mu": [5, "eV"], "T": t}
    )
    assert out["n"] == 0.5
    assert rel(out["ln_n"], -math.log(2)) < 1e-15
    deep, _ = run(
        {"mode": "occupation", "statistics": "fermi_dirac", "E": 0.0, "mu": [20, "eV"], "T": t}
    )
    assert deep["n"] == 1.0
    out2, _ = run(
        {
            "mode": "occupation",
            "statistics": "fermi_dirac",
            "E": [4.9, "eV"],
            "mu": [5, "eV"],
            "T": t,
        }
    )
    assert rel(1 - out2["n"], mp_occ("fermi_dirac", 0.1 * EV / (K * t))) < 1e-12  # hole symmetry
    assert rel(out2["ln_n"], mp_ln_occ("fermi_dirac", -0.1 * EV / (K * t))) < 1e-12


def test_occupation_example_and_degeneracy() -> None:
    # doc 63: E - mu = 0.1 eV at 300 K, Fermi-Dirac 0.020469 against Bose-Einstein 0.021343
    common = {"mode": "occupation", "E": [0.1, "eV"], "mu": 0.0, "T": 300}
    fd, _ = run({**common, "statistics": "fermi_dirac"})
    be, _ = run({**common, "statistics": "bose_einstein", "g": 3})
    assert round(fd["n"], 6) == 0.020469
    assert round(be["n"], 6) == 0.021343
    assert rel(be["n_total"], 3 * be["n"]) < 1e-14
    assert "n_total" not in fd


def test_occupation_errors_and_warnings() -> None:
    with pytest.raises(ToolError, match="E > mu"):
        run(
            {"mode": "occupation", "statistics": "bose_einstein", "E": 1e-21, "mu": 1e-21, "T": 300}
        )
    with pytest.raises(ToolError, match="statistics"):
        run({"mode": "occupation", "statistics": "boltzmann", "E": 1e-21, "mu": 0.0, "T": 300})
    with pytest.raises(ToolError, match="Keys not used"):
        run(
            {
                "mode": "occupation",
                "statistics": "fermi_dirac",
                "E": 1e-21,
                "mu": 0.0,
                "T": 300,
                "V": 1,
            }
        )
    # mu - E = 20 eV at 300 K: x = -773.6, exp(773.6) is beyond the floats
    out, ctx = run(
        {
            "mode": "occupation",
            "statistics": "maxwell_boltzmann",
            "E": 0.0,
            "mu": [20, "eV"],
            "T": 300,
        }
    )
    assert "n" not in out
    assert rel(out["ln_n"], 20 * EV / (K * 300)) < 1e-14
    assert any("beyond" in w for w in ctx.warnings)
    with pytest.raises(ToolError, match="g must be >= 1"):
        run(
            {
                "mode": "occupation",
                "statistics": "fermi_dirac",
                "E": 1e-21,
                "mu": 0.0,
                "T": 300,
                "g": 0,
            }
        )
    _, ctx2 = run(
        {
            "mode": "occupation",
            "statistics": "maxwell_boltzmann",
            "E": 0.0,
            "mu": [0.01, "eV"],
            "T": 300,
        }
    )
    assert any("n > 0.1" in w for w in ctx2.warnings)


def test_density_of_states_errors() -> None:
    with pytest.raises(ToolError, match="exactly one"):
        run({"mode": "density_of_states", "particle": "electron"})
    with pytest.raises(ToolError, match="not both"):
        run({"mode": "density_of_states", "particle": "electron", "m": M_E, "E": 1e-19})
    with pytest.raises(ToolError, match="g_s"):
        run({"mode": "density_of_states", "particle": "electron", "E": 1e-19, "g_s": 0})
    with pytest.raises(ToolError, match="> 0"):
        run({"mode": "density_of_states", "particle": "electron", "n": -1.0})
