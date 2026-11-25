"""CODATA physical constants (D47): `const()` in math_eval and `eng_calc.physical_constant`.

Independent methods: the 7 constants that define the SI since 2019 (exact values by definition,
written here by hand from the SI brochure); exact relations between constants (R = N_A k, F = N_A e,
hbar = h / 2 pi); relations measured within the CODATA uncertainty (Stefan-Boltzmann from the basic
constants, Rydberg from alpha, electron mass, c and h).
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from oce_math_mcp import constants
from tests.conftest import call, ok


def ev(expr: str, **kw: object) -> dict:
    return ok("math_eval", expr=expr, **kw)


def pc(**params: object) -> dict:
    return ok("eng_calc", operation="physical_constant", params=params)


SI_EXACT = {
    "c": "299792458",
    "h": "6.62607015e-34",
    "e": "1.602176634e-19",
    "k_B": "1.380649e-23",
    "N_A": "6.02214076e23",
}


@pytest.mark.parametrize(("sym", "txt"), list(SI_EXACT.items()))
def test_si_defining_constants_exact(sym: str, txt: str) -> None:
    r = pc(name=sym)
    assert r["exact"] is True and r["uncertainty"] == 0
    assert sp.Rational(r["value_text"]) == sp.Rational(txt)
    assert r["codata"] == "CODATA 2022"


def test_exact_relations_in_math_eval() -> None:
    # R = N_A k and F = N_A e are exact in the 2019 SI: the comparison is between rationals
    r = ev("const('R') - const('N_A') * const('k_B')", exact_mode="fraction")
    assert r["exact"] == "0"
    f = ev("const('F') / (const('N_A') * const('e'))", exact_mode="fraction")
    assert f["exact"] in {"1", "1/1"} or abs(float(f["float"]) - 1) < 1e-15
    hb = ev("const('hbar') * 2 * pi / const('h')", exact_mode="float")
    assert abs(hb["float"] - 1) < 1e-15


def test_stefan_boltzmann_from_base_constants() -> None:
    k, h, c = 1.380649e-23, 6.62607015e-34, 299792458.0
    sigma = 2 * math.pi**5 * k**4 / (15 * h**3 * c**2)
    assert abs(pc(name="sigma")["value"] / sigma - 1) < 1e-12


def test_rydberg_relation_within_uncertainty() -> None:
    ra, rm = pc(name="alpha"), pc(name="m_e")
    c, h = pc(name="c")["value"], pc(name="h")["value"]
    rinf = pc(name="R_inf")
    calc = ra["value"] ** 2 * rm["value"] * c / (2 * h)
    # propagated relative uncertainty: 2 u_r(alpha) + u_r(m_e) (c and h are exact)
    tol = 2 * ra["relative_uncertainty"] + rm["relative_uncertainty"]
    assert abs(calc / rinf["value"] - 1) < tol


def test_measured_constant_has_uncertainty() -> None:
    r = pc(name="Newtonian constant of gravitation")
    assert r["exact"] is False and r["uncertainty"] > 0
    assert r["relative_uncertainty"] == pytest.approx(r["uncertainty"] / r["value"])
    assert r["unit"] == "m^3 kg^-1 s^-2"


def test_aliases_all_resolve() -> None:
    for alias, name in constants.ALIASES.items():
        assert constants.lookup(alias)[0] == name


def test_case_insensitive_name() -> None:
    assert pc(name="planck CONSTANT")["name"] == "Planck constant"


def test_search() -> None:
    r = pc(search="Boltzmann")
    assert "Boltzmann constant" in r["matches"] and r["count"] >= 3


def test_unknown_constant_suggests() -> None:
    resp = call("eng_calc", operation="physical_constant", params={"name": "Plank constant"})
    assert (
        resp["error"]["code"] == "INVALID_ARGUMENT"
        and "Planck constant" in resp["error"]["message"]
    )
    resp = call("math_eval", expr="const('no_such_constant')")
    assert resp["error"]["code"] == "INVALID_ARGUMENT"


def test_const_in_formula_photon_energy() -> None:
    # energy of a photon at 500 nm in eV: h c / (lambda e)
    r = ev("const('h') * const('c') / (500e-9 * const('e'))", exact_mode="float")
    assert abs(r["float"] - 6.62607015e-34 * 299792458 / (500e-9 * 1.602176634e-19)) < 1e-12
