"""D115: nuclear radius r0 A^(1/3) and the semi-empirical mass formula in eng_calc.nuclide_data.

References independent of the module, from the local offline library:
- "List of equations in nuclear and particle physics", section "Nuclear structure": r = r0 A^(1/3) with
  r0 = 1.2 fm, and E_B = a_v A - a_s A^(2/3) - a_c Z (Z - 1) A^(-1/3) - a_a (N - Z)^2 / A + 12 delta A^(-1/2)
  with delta = +1 (even N, even Z), -1 (odd, odd), 0 (odd A).
- "Semi-empirical mass formula", table "Calculating coefficients", column "Least-squares fit (1)": a_V = 15.8,
  a_S = 18.3, a_C = 0.714, a_A = 23.2, a_P = 12 MeV, k_P = -1/2.
- The measure of the 475 formulas (RAG doc 62): Fe-56 gives r = 4.591034838653734e-15 m and E_B =
  490.6078337619601 MeV.
- AME2020 binding energies of the tool itself: over the 2,455 measured ground states with A >= 20 the formula is
  off by 3.855 MeV rms (RAG doc 67, the same measure decided the coefficients).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import pytest

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc

PAGE = {"a_V": 15.8, "a_S": 18.3, "a_C": 0.714, "a_A": 23.2, "a_P": 12.0, "k_P": -0.5}


def run(params: dict[str, Any]) -> tuple[dict[str, Any], Ctx]:
    ctx = Ctx()
    out = eng_calc.handle({"operation": "nuclide_data", "params": params}, ctx)
    return out, ctx


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def page_semf(z: int, n: int, c: dict[str, float] = PAGE) -> dict[str, float]:
    """The page formula, term by term."""
    a = z + n
    if z % 2 == 0 and n % 2 == 0:
        delta = 1
    elif z % 2 == 1 and n % 2 == 1:
        delta = -1
    else:
        delta = 0
    terms = {
        "semf_volume_MeV": c["a_V"] * a,
        "semf_surface_MeV": -c["a_S"] * a ** (2 / 3),
        "semf_coulomb_MeV": -c["a_C"] * z * (z - 1) / a ** (1 / 3),
        "semf_asymmetry_MeV": -c["a_A"] * float(Fraction((n - z) ** 2, a)),
        "semf_pairing_MeV": delta * c["a_P"] * a ** c["k_P"],
    }
    terms["semf_binding_MeV"] = sum(terms.values())
    return terms


def test_iron_56_values_of_the_measure() -> None:
    out, ctx = run({"nuclide": "Fe-56"})
    assert rel(out["radius_fm"], 4.591034838653734) < 1e-14  # the measure: 4.591034838653734e-15 m
    assert out["r0_fm"] == 1.2
    assert rel(out["semf_binding_MeV"], 490.6078337619601) < 1e-13  # the measure
    assert rel(out["semf_pairing_MeV"], 12 / math.sqrt(56)) < 1e-14  # even-even
    assert (
        rel(out["semf_minus_ame_MeV"], out["semf_binding_MeV"] - out["binding_energy_MeV"]) < 1e-12
    )
    assert (
        rel(out["semf_error_percent"], 100 * out["semf_minus_ame_MeV"] / out["binding_energy_MeV"])
        < 1e-12
    )
    assert rel(out["semf_binding_per_nucleon_MeV"], out["semf_binding_MeV"] / 56) < 1e-14
    assert ctx.units is not None
    assert ctx.units["radius_fm"] == "fm" and ctx.units["semf_binding_MeV"] == "MeV"
    assert ctx.units["semf_error_percent"] == "%"
    assert ctx.warnings == []
    assert ctx.formula is not None and "a_C Z (Z - 1) / A^(1/3)" in ctx.formula


@pytest.mark.parametrize(
    "nuclide", ["U-238", "Bi-209", "K-40", "Pb-208", "Sn-120", "Cs-133", "Ca-48"]
)
def test_page_formula_term_by_term(nuclide: str) -> None:
    out, _ = run({"nuclide": nuclide})
    ref = page_semf(out["Z"], out["N"])
    for key, val in ref.items():
        assert abs(out[key] - val) <= 1e-12 * abs(ref["semf_volume_MeV"]), key
    assert rel(out["radius_fm"], 1.2 * out["A"] ** (1 / 3)) < 1e-14


def test_pairing_signs() -> None:
    even, _ = run({"nuclide": "Ca-40"})
    odd_odd, _ = run({"nuclide": "K-40"})
    odd_a, _ = run({"nuclide": "Ca-41"})
    assert even["semf_pairing_MeV"] > 0
    assert rel(odd_odd["semf_pairing_MeV"], -12 / math.sqrt(40)) < 1e-14
    assert odd_a["semf_pairing_MeV"] == 0


def test_against_ame2020_over_the_table() -> None:
    from oce_math_mcp.tools import eng_nuc

    data = eng_nuc._nuclides()  # pyright: ignore[reportPrivateUsage]
    keys = [
        k
        for k, r in data.items()
        if not r[eng_nuc.I_EXC] and not r[eng_nuc.I_EST] and r[eng_nuc.I_A] >= 20
    ]
    diffs = [run({"nuclide": k})[0]["semf_minus_ame_MeV"] for k in keys]
    rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    assert len(keys) == 2455
    assert round(rms, 3) == 3.855  # the measure that chose the coefficients (RAG doc 67)


def test_nucleus_outside_the_table() -> None:
    out, ctx = run({"Z": 120, "N": 184})
    assert out["Z"] == 120 and out["N"] == 184 and out["A"] == 304
    assert "mass_excess_keV" not in out and "semf_minus_ame_MeV" not in out
    ref = page_semf(120, 184)
    assert rel(out["semf_binding_MeV"], ref["semf_binding_MeV"]) < 1e-13
    assert rel(out["radius_fm"], 1.2 * 304 ** (1 / 3)) < 1e-14
    assert any("not in AME2020" in w for w in ctx.warnings)
    same, _ = run({"Z": 26, "N": 30})  # in the table: the full record
    iron, _ = run({"nuclide": "Fe-56"})
    assert same == iron


def test_radius_and_coefficients_given() -> None:
    a, _ = run({"nuclide": "Pb-208", "r0": [1.25, "fm"]})
    assert rel(a["radius_fm"], 1.25 * 208 ** (1 / 3)) < 1e-14 and a["r0_fm"] == 1.25
    b, _ = run({"nuclide": "Pb-208", "r0": 1.25e-15})
    assert rel(b["radius_fm"], a["radius_fm"]) < 1e-14
    own = {"a_V": 15.75, "a_S": 17.8, "a_C": 0.711, "a_A": 23.7, "a_P": 11.18, "k_P": -0.5}
    c, ctx = run({"nuclide": "Pb-208", "semf": own})
    assert rel(c["semf_binding_MeV"], page_semf(82, 126, own)["semf_binding_MeV"]) < 1e-13
    assert ctx.formula is not None and "a_V = 15.75" in ctx.formula
    part, _ = run({"nuclide": "Pb-208", "semf": {"a_P": 34, "k_P": -0.75}})
    mixed = {**PAGE, "a_P": 34.0, "k_P": -0.75}
    assert rel(part["semf_binding_MeV"], page_semf(82, 126, mixed)["semf_binding_MeV"]) < 1e-13


def test_light_nuclei_warning() -> None:
    _, light = run({"nuclide": "C-12"})
    assert any("light nuclei" in w for w in light.warnings)
    _, ne = run({"nuclide": "Ne-20"})
    assert not any("light nuclei" in w for w in ne.warnings)


def test_errors() -> None:
    with pytest.raises(ToolError, match="exactly one"):
        run({"nuclide": "Fe-56", "Z": 26, "N": 30})
    with pytest.raises(ToolError, match=r"params\.N"):
        run({"Z": 26})
    with pytest.raises(ToolError, match=r"params\.Z must be >= 1"):
        run({"Z": 0, "N": 4})
    with pytest.raises(ToolError, match=r"params\.semf"):
        run({"nuclide": "Fe-56", "semf": {"a_X": 1}})
    with pytest.raises(ToolError, match=r"params\.semf"):
        run({"nuclide": "Fe-56", "semf": [15.8]})
    with pytest.raises(ToolError, match=r"params\.semf\.a_V must be a finite number"):
        run({"nuclide": "Fe-56", "semf": {"a_V": "15.8"}})
    with pytest.raises(ToolError, match=r"params\.r0 must be > 0"):
        run({"nuclide": "Fe-56", "r0": 0})
