"""D110: eng_calc.electronegativity (section "Quantum chemistry" of "Defining equation (physical chemistry)").

The page: Pauling chi_A - chi_B = (eV)^(-1/2) sqrt(E_d(AB) - [E_d(AA) + E_d(BB)] / 2) and Mulliken
chi = 10^-3 [187 (E_I + E_EA) + 170], energies in eV. The article "Electronegativity" adds the example
H-Br (3.79, 4.52 and 2.00 eV give 0.73), the inverse formula E_d(AB) = mean + (chi_A - chi_B)^2 eV, the
geometric-mean fit sqrt(E_d(AA) E_d(BB)) + 1.3 (chi_A - chi_B)^2 eV, the absolute electronegativity
(E_i + E_ea) / 2 and the chemical potential -chi.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc

EV = 1.602176634e-19  # J, exact
NA = 6.02214076e23  # 1/mol, exact
FARADAY = EV * NA  # C/mol: 1 eV per particle = FARADAY J/mol


def en(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle({"operation": "electronegativity", "params": params}, ctx or Ctx())


def ev(x: float) -> list:
    return [x, "eV"]


def approx(v: Any, rel: float = 1e-12) -> Any:
    return pytest.approx(v, rel=rel, abs=0)


def units(ctx: Ctx) -> dict[str, str]:
    assert ctx.units is not None
    return ctx.units


def bad(params: dict, words: str) -> None:
    with pytest.raises(ToolError) as e:
        en(params)
    assert e.value.code == "INVALID_ARGUMENT"
    assert words in e.value.message


HBR = {"E_AB": ev(3.79), "E_AA": ev(4.52), "E_BB": ev(2.00)}  # H-Br, H-H, Br-Br


# ------------------------------------------------------------------ Pauling


def test_pauling_hydrogen_bromide_of_the_article() -> None:
    ctx = Ctx()
    out = en({"mode": "pauling", **HBR, "chi_B": 2.20}, ctx)
    mean = (4.52 + 2.00) / 2
    assert out["mean_energy_eV"] == approx(mean)
    assert out["excess_energy_eV"] == approx(3.79 - mean, 1e-11)
    assert out["delta_chi"] == approx(math.sqrt(3.79 - mean), 1e-11)
    assert round(out["delta_chi"], 2) == 0.73  # the article's value
    assert out["chi_A"] == approx(2.20 + out["delta_chi"])
    assert abs(out["chi_A"] - 2.96) < 0.05  # Allred's revised Pauling value of bromine
    assert ctx.units == {
        "mean_energy_eV": "eV",
        "excess_energy_eV": "eV",
        "delta_chi": "1",
        "chi_A": "1",
    }
    assert ctx.warnings == []


def test_pauling_with_the_geometric_mean() -> None:
    out = en({"mode": "pauling", **HBR, "mean": "geometric"})
    mean = math.sqrt(4.52 * 2.00)
    assert out["mean_energy_eV"] == approx(mean)
    assert out["delta_chi"] == approx(math.sqrt((3.79 - mean) / 1.3), 1e-11)
    assert "chi_A" not in out


def test_bond_energy_is_the_inverse_of_pauling() -> None:
    for mean in ("arithmetic", "geometric"):
        d = en({"mode": "pauling", **HBR, "mean": mean})["delta_chi"]
        back = en(
            {
                "mode": "bond_energy",
                "delta_chi": d,
                "E_AA": ev(4.52),
                "E_BB": ev(2.00),
                "mean": mean,
            }
        )
        assert back["E_AB_eV"] == approx(3.79, 1e-11)
        assert back["E_AB_kJ_mol"] == approx(3.79 * FARADAY / 1000, 1e-11)
    ctx = Ctx()
    out = en(
        {"mode": "bond_energy", "chi_A": 2.96, "chi_B": 2.20, "E_AA": ev(4.52), "E_BB": ev(2.00)},
        ctx,
    )
    assert out["delta_chi"] == approx(2.96 - 2.20)
    assert out["E_AB_eV"] == approx((4.52 + 2.00) / 2 + (2.96 - 2.20) ** 2)
    assert units(ctx)["E_AB_kJ_mol"] == "kJ/mol" and units(ctx)["E_AB_eV"] == "eV"
    swapped = en(
        {"mode": "bond_energy", "chi_A": 2.20, "chi_B": 2.96, "E_AA": ev(4.52), "E_BB": ev(2.00)}
    )
    assert swapped["delta_chi"] == approx(2.96 - 2.20)  # the difference, whatever the order


def test_energies_per_mole_are_energies_per_bond() -> None:
    kj = {k: [v[0] * FARADAY / 1000, "kJ/mol"] for k, v in HBR.items()}
    assert en({"mode": "pauling", **kj})["delta_chi"] == approx(
        en({"mode": "pauling", **HBR})["delta_chi"], 1e-11
    )
    joule = {k: [v[0] * EV, "J"] for k, v in HBR.items()}
    assert en({"mode": "pauling", **joule})["delta_chi"] == approx(
        en({"mode": "pauling", **HBR})["delta_chi"], 1e-11
    )
    kcal = en(
        {
            "mode": "pauling",
            "E_AB": [87.4, "kcal/mol"],
            "E_AA": [104.2, "kcal/mol"],
            "E_BB": [46.1, "kcal/mol"],
        }
    )
    want = math.sqrt((87.4 - (104.2 + 46.1) / 2) * 4184 / FARADAY)
    assert kcal["delta_chi"] == approx(want, 1e-11)


# ------------------------------------------------------------------ Mulliken


def test_mulliken_formula_of_the_page() -> None:
    ctx = Ctx()
    out = en({"mode": "mulliken", "E_I": ev(13.598), "E_EA": ev(0.754)}, ctx)  # hydrogen
    s = 13.598 + 0.754
    assert out["E_sum_eV"] == approx(s)
    assert out["chi_pauling_scale"] == approx(1e-3 * (187 * s + 170))
    assert out["chi_absolute_eV"] == approx(s / 2)
    assert out["mu_chemical_eV"] == approx(-s / 2)
    assert units(ctx)["chi_absolute_eV"] == "eV" and units(ctx)["chi_pauling_scale"] == "1"
    assert ctx.warnings == []


def test_mulliken_with_a_negative_electron_affinity_and_kj_per_mol() -> None:
    out = en(
        {"mode": "mulliken", "E_I": ev(14.534), "E_EA": ev(-0.07)}
    )  # nitrogen: negative affinity
    assert out["chi_absolute_eV"] == approx((14.534 - 0.07) / 2)
    kj = en(
        {
            "mode": "mulliken",
            "E_I": [13.598 * FARADAY / 1000, "kJ/mol"],
            "E_EA": [0.754 * FARADAY / 1000, "kJ/mol"],
        }
    )
    assert kj["chi_pauling_scale"] == approx(1e-3 * (187 * (13.598 + 0.754) + 170), 1e-11)


def test_plain_numbers_are_joules_and_a_far_value_is_flagged() -> None:
    ctx = Ctx()
    out = en({"mode": "mulliken", "E_I": 13.598, "E_EA": 0.754}, ctx)  # 13.598 J, not eV
    assert out["E_sum_eV"] == approx((13.598 + 0.754) / EV)
    assert any("joules" in w and "E_I" in w for w in ctx.warnings)
    small = Ctx()
    en({"mode": "mulliken", "E_I": ev(13.598), "E_EA": ev(1e-4)}, small)
    assert any("E_EA" in w for w in small.warnings)
    none = Ctx()
    en({"mode": "mulliken", "E_I": ev(13.598), "E_EA": ev(0)}, none)
    assert none.warnings == []


# ------------------------------------------------------------------ errors


def test_electronegativity_errors() -> None:
    bad({"mode": "pauling", "E_AB": ev(2.0), "E_AA": ev(4.52), "E_BB": ev(2.00)}, "below the mean")
    bad({"mode": "pauling", **HBR, "mean": "harmonic"}, "mean")
    bad({"mode": "pauling", "E_AB": ev(3.79), "E_AA": ev(0), "E_BB": ev(2.00)}, "E_AA")
    bad({"mode": "mulliken", "E_I": ev(-1), "E_EA": ev(0.7)}, "E_I")
    bad(
        {
            "mode": "bond_energy",
            "delta_chi": 0.5,
            "chi_A": 3.0,
            "chi_B": 2.2,
            "E_AA": ev(4.5),
            "E_BB": ev(2.0),
        },
        "delta_chi",
    )
    bad(
        {"mode": "bond_energy", "chi_A": 3.0, "E_AA": ev(4.5), "E_BB": ev(2.0)},
        "params.chi_A and params.chi_B",
    )
    bad({"mode": "bond_energy", "delta_chi": -0.5, "E_AA": ev(4.5), "E_BB": ev(2.0)}, "delta_chi")
    bad({"mode": "pauling", **HBR, "E_I": ev(13.6)}, "Keys not used")
