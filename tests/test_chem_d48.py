"""Basic chemistry (D48): molar mass, stoichiometry, acids and bases, solutions, kinetics,
electrochemistry.

Independent methods: consistency of the two CIAAW lists (abridged weight inside the interval or within
the uncertainty of the standard weight); atom balance recounted element by element; charge balance of
the solutions solved with brentq; integrated rate laws against solve_ivp;
Nernst against Delta G = -n F E and Delta G = Delta G0 + R T ln Q.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal

import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from oce_math_mcp import constants
from tests.conftest import ROOT, call, ok

R = constants.lookup("R")[1]
F = constants.lookup("F")[1]
DATA = json.loads((ROOT / "src/oce_math_mcp/data/atomic_weights.json").read_text(encoding="utf-8"))


def e(operation: str, **params: object) -> dict:
    return ok("eng_calc", operation=operation, params=params)


def bad(operation: str, params: dict) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def aw(sym: str) -> float:
    return float(DATA["elements"][sym]["abridged"]["value"])


# ------------------------------------------------------------------ CIAAW data


def test_ciaaw_tables_consistent() -> None:
    n = 0
    for sym, el in DATA["elements"].items():
        a, s = el["abridged"], el["standard"]
        if not a:
            assert not s, sym
            continue
        v, u = Decimal(a["value"]), Decimal(a["uncertainty"])
        if "interval" in s:
            lo, hi = (Decimal(x) for x in s["interval"])
            assert v - u <= lo and hi <= v + u, sym  # the abridged value covers the interval
        else:
            assert abs(Decimal(s["value"]) - v) <= u, sym
        n += 1
    assert n == 84


# ------------------------------------------------------------------ molar_mass


@pytest.mark.parametrize(
    ("formula", "counts"),
    [
        ("H2O", {"H": 2, "O": 1}),
        ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}),
        ("K4[Fe(CN)6]", {"K": 4, "Fe": 1, "C": 6, "N": 6}),
        ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
        ("CuSO4*5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
        ("Al2(SO4)3", {"Al": 2, "S": 3, "O": 12}),
    ],
)
def test_molar_mass_counts_and_sum(formula: str, counts: dict[str, int]) -> None:
    r = e("molar_mass", formula=formula)
    assert r["elements"] == counts
    M = sum(k * aw(s) for s, k in counts.items())
    assert rel(r["M_g_mol"], M, 1e-12) and rel(r["M"], M / 1000, 1e-12)
    assert rel(sum(r["mass_fraction"].values()), 1.0, 1e-12)


def test_molar_mass_amount_conversion() -> None:
    r = e("molar_mass", formula="NaCl", m=[117, "g"])
    assert rel(r["n"], 0.117 / r["M"], 1e-12)
    r2 = e("molar_mass", formula="NaCl", n=2)
    assert rel(r2["m"], 2 * r2["M"], 1e-12)


def test_molar_mass_errors() -> None:
    assert "Xx" in bad("molar_mass", {"formula": "Xx2O"})
    assert "Tc" in bad("molar_mass", {"formula": "TcO2"})  # no standard atomic weight
    assert "bracket" in bad("molar_mass", {"formula": "Ca(OH2"}).lower()


# ------------------------------------------------------------------ reaction_stoich


def _count(formula: str) -> dict[str, int]:
    return e("molar_mass", formula=formula)["elements"]


@pytest.mark.parametrize(
    "reaction",
    [
        "C3H8 + O2 -> CO2 + H2O",
        "KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2",
        "Fe2O3 + CO = Fe + CO2",
        "C6H12O6 + O2 → CO2 + H2O",
    ],
)
def test_balance_conserves_atoms(reaction: str) -> None:
    r = e("reaction_stoich", reaction=reaction)
    nr = len(r["reactants"])
    tot: dict[str, int] = {}
    for i, (sp_, k) in enumerate(zip(r["species"], r["coefficients"], strict=True)):
        assert isinstance(k, int) and k > 0
        for el, c in _count(sp_).items():
            tot[el] = tot.get(el, 0) + (k * c if i < nr else -k * c)
    assert all(v == 0 for v in tot.values()), tot
    assert math.gcd(*r["coefficients"]) == 1


def test_propane_coefficients_and_limiting() -> None:
    r = e(
        "reaction_stoich", reaction="C3H8 + O2 -> CO2 + H2O", amounts={"C3H8": [44.1, "g"], "O2": 4}
    )
    assert r["coefficients"] == [1, 5, 3, 4]
    nC3H8 = 0.0441 / e("molar_mass", formula="C3H8")["M"]
    assert r["limiting"] == "O2" and rel(r["extent"], 4 / 5, 1e-12)
    assert rel(r["n_reacted"][2], 3 * 4 / 5, 1e-12) and rel(r["n_left"][0], nC3H8 - 4 / 5, 1e-9)


def test_balance_errors() -> None:
    assert "balance" in bad("reaction_stoich", {"reaction": "H2 -> O2"})


# ------------------------------------------------------------------ acid_base


def _ph_by_charge_balance(c: float, Ka: float | None, Kw: float = 1e-14) -> float:
    if Ka is None:  # strong acid: [H+] = c + Kw/[H+]
        h = brentq(lambda h: h - c - Kw / h, 1e-14, 10, xtol=1e-30, rtol=1e-15)
    else:  # weak acid: [H+] = Ka c / (Ka + [H+]) + Kw/[H+]
        h = brentq(lambda h: h - Ka * c / (Ka + h) - Kw / h, 1e-14, 10, xtol=1e-30, rtol=1e-15)
    return -math.log10(h)


@pytest.mark.parametrize("c", [1e-1, 1e-6, 1e-8])
def test_strong_acid_exact(c: float) -> None:
    r = e("acid_base", mode="strong", kind="acid", c=c)
    assert rel(r["pH"], _ph_by_charge_balance(c, None), 1e-10)
    assert r["pH"] < 7


def test_strong_base() -> None:
    r = e("acid_base", mode="strong", kind="base", c=[0.01, "mol/L"])
    assert rel(r["pH"], 14 - _ph_by_charge_balance(0.01, None), 1e-10)


@pytest.mark.parametrize(("c", "Ka"), [(0.1, 1.8e-5), (1e-4, 1.8e-5), (0.05, 6.3e-5)])
def test_weak_acid_exact(c: float, Ka: float) -> None:
    r = e("acid_base", mode="weak", kind="acid", c=c, Ka=Ka)
    assert rel(r["pH"], _ph_by_charge_balance(c, Ka), 1e-10)
    r2 = e("acid_base", mode="weak", kind="acid", c=c, pKa=-math.log10(Ka))
    assert rel(r2["pH"], r["pH"], 1e-10)


def test_weak_base_mirrors_weak_acid() -> None:
    # weak base with Kb: pOH like the pH of a weak acid with Ka = Kb
    r = e("acid_base", mode="weak", kind="base", c=0.1, Kb=1.8e-5)
    assert rel(r["pOH"], _ph_by_charge_balance(0.1, 1.8e-5), 1e-10)


def test_buffer() -> None:
    ca, cb, Ka = 0.1, 0.2, 1.8e-5
    r = e("acid_base", mode="buffer", c_acid=ca, c_base=cb, Ka=Ka)
    assert rel(r["pH_henderson"], -math.log10(Ka) + math.log10(cb / ca), 1e-12)
    # exact: [H+] + cb = Ka (ca + cb) / (Ka + [H+]) + Kw/[H+]
    h = brentq(
        lambda h: h + cb - Ka * (ca + cb) / (Ka + h) - 1e-14 / h, 1e-14, 1, xtol=1e-30, rtol=1e-15
    )
    assert rel(r["pH"], -math.log10(h), 1e-10)


# ------------------------------------------------------------------ solution_prep


def test_dilution() -> None:
    r = e("solution_prep", mode="dilution", c1=2, V1=[50, "mL"], c2=0.5)
    assert rel(r["V2"], 2 * 50e-6 / 0.5, 1e-12)


def test_molarity_from_mass() -> None:
    r = e("solution_prep", mode="molarity", formula="NaCl", m=[5.844, "g"], V=[1, "L"])
    M = aw("Na") + aw("Cl")
    assert rel(r["c"], 5.844 / M / 1.0, 1e-12) and rel(r["n"], 5.844 / M, 1e-12)


# ------------------------------------------------------------------ reaction_kinetics


def test_arrhenius_round_trip() -> None:
    r = e("reaction_kinetics", mode="arrhenius", k1=1.0e-3, T1=300, k2=4.0e-3, T2=320)
    Ea = r["Ea"]
    assert rel(Ea, R * math.log(4.0) / (1 / 300 - 1 / 320), 1e-12)
    back = e("reaction_kinetics", mode="arrhenius", A=r["A"], Ea=Ea, T=320)
    assert rel(back["k"], 4.0e-3, 1e-10)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_integrated_rate_laws(order: int) -> None:
    k, c0, t = 0.02, 1.0, 15.0
    r = e("reaction_kinetics", mode="integrated", order=order, k=k, c0=c0, t=t)
    sol = solve_ivp(lambda _t, c: [-k * c[0] ** order], (0, t), [c0], rtol=1e-12, atol=1e-14)
    assert rel(r["c"], float(sol.y[0, -1]), 1e-8)
    th = r["t_half"]
    half = solve_ivp(lambda _t, c: [-k * c[0] ** order], (0, th), [c0], rtol=1e-12, atol=1e-14)
    assert rel(float(half.y[0, -1]), c0 / 2, 1e-8)
    back = e("reaction_kinetics", mode="integrated", order=order, k=k, c0=c0, c=r["c"])
    assert rel(back["t"], t, 1e-9)


# ------------------------------------------------------------------ electrochemistry


def test_nernst_against_gibbs() -> None:
    E0, n, Q, T = 1.10, 2, 0.01, 298.15
    r = e("electrochemistry", mode="nernst", E0=E0, n=n, Q=Q, T=T)
    dG = -n * F * E0 + R * T * math.log(Q)  # Delta G = Delta G0 + R T ln Q
    assert rel(r["E"], -dG / (n * F), 1e-12)


def test_electrolysis_faraday() -> None:
    r = e("electrochemistry", mode="electrolysis", I=2.0, t=[1, "h"], n=2, formula="Cu")
    assert rel(r["m"], 2.0 * 3600 * aw("Cu") / 1000 / (2 * F), 1e-12)


def test_gibbs_and_K() -> None:
    r = e("electrochemistry", mode="gibbs", n=2, E=0.5, T=298.15)
    assert rel(r["dG"], -2 * F * 0.5, 1e-12)
    assert rel(math.log(r["K"]), -r["dG"] / (R * 298.15), 1e-10)


# ------------------------------------------------------------------ numerical examples of the sources


@pytest.mark.parametrize(
    ("formula", "value"),
    [
        ("H2O", "18.015"),
        ("NaCl", "58.44"),
        ("C6H12O6", "180.16"),
        ("Ca(OH)2", "74.09"),
        ("K4[Fe(CN)6]", "368.34"),
        ("CuSO4·5H2O", "249.69"),
    ],
)
def test_source_pubchem_molar_mass(formula: str, value: str) -> None:
    """PubChem, Molecular Weight. PubChem uses older atomic weights (S = 32.065): the difference
    must stay within the rounding of the source plus the uncertainty of the IUPAC abridged weights."""
    dec = len(value.split(".")[1])
    r = e("molar_mass", formula=formula)
    assert abs(r["M_g_mol"] - float(value)) <= 0.5 * 10**-dec + r["M_uncertainty_g_mol"]


def test_source_openstax_limiting_reagent() -> None:
    """OpenStax Chemistry 1e 4.5: 3 Si + 2 N2 -> Si3N4 with 2.00 g Si and 1.50 g N2; H2 + Cl2 -> 2 HCl."""
    r = e(
        "reaction_stoich",
        reaction="Si + N2 -> Si3N4",
        amounts={"Si": [2.00, "g"], "N2": [1.50, "g"]},
    )
    assert (
        r["coefficients"] == [3, 2, 1]
        and r["limiting"] == "Si"
        and round(r["n_reacted"][2], 4) == 0.0237
    )
    h = e("reaction_stoich", reaction="H2 + Cl2 -> HCl", amounts={"H2": 6, "Cl2": 4})
    assert h["limiting"] == "Cl2" and rel(h["n_reacted"][2], 8) and rel(h["n_left"][0], 2)


def test_source_strong_acid_dilute() -> None:
    """Chem1 (Lower) 13.02 and UBC: dilute strong acid, with the self-ionization of water."""
    assert round(e("acid_base", mode="strong", kind="acid", c=1.0e-8)["pH"], 2) == 6.98
    assert round(e("acid_base", mode="strong", kind="acid", c=1.0e-6)["pH"], 2) == 6.00
    assert round(e("acid_base", mode="strong", kind="acid", c=2.0e-7)["pH"], 2) == 6.62


def test_source_weak_acid_and_base() -> None:
    """Gonzaga (acetic acid 0.10 M, NH3 0.050 M) and LibreTexts, Exact pH Calculations (cubic)."""
    r = e("acid_base", mode="weak", kind="acid", c=0.10, Ka=1.8e-5)
    assert rel(r["H"], 1.3327e-3, 1e-4) and round(r["pH"], 2) == 2.88
    w = e("acid_base", mode="weak", kind="acid", c=0.0010, Ka=4.0e-11)
    assert round(w["H"], 9) == 2.24e-7 and round(w["pH"], 2) == 6.65
    b = e("acid_base", mode="weak", kind="base", c=0.050, Kb=1.8e-5)
    assert round(b["pH"], 2) == 10.97


def test_source_dilution_and_molarity() -> None:
    """LibreTexts Portland CC 3.03 and OpenStax Atoms First 6.04."""
    assert (
        round(
            e("solution_prep", mode="dilution", c1=2.19, V1=[25.0, "mL"], V2=[72.8, "mL"])["c2"], 3
        )
        == 0.752
    )
    assert (
        round(
            e("solution_prep", mode="dilution", c1=0.885, V1=[76.5, "mL"], c2=0.500)["V2"] * 1e6, 1
        )
        == 135.4
    )
    assert (
        round(
            e("solution_prep", mode="dilution", c1=5.00, V1=[0.850, "L"], V2=[1.80, "L"])["c2"], 2
        )
        == 2.36
    )
    assert (
        round(
            e("solution_prep", mode="dilution", c1=1.59, c2=0.100, V2=[5.00, "L"])["V1"] * 1000, 3
        )
        == 0.314
    )
    m = e("solution_prep", mode="molarity", formula="CH3COOH", m=[25.2, "g"], V=[0.500, "L"])
    assert round(m["c"], 3) == 0.839
    g = e("solution_prep", mode="molarity", formula="NaCl", c=5.30, V=[0.250, "L"])
    assert round(g["m"] * 1000, 1) == 77.4


def test_source_openstax_kinetics() -> None:
    """OpenStax Chemistry 2e 12.4 and 12.5."""
    cyc = e("reaction_kinetics", mode="integrated", order=1, k=9.2e-3, c0=1.0, c=0.20)
    assert round(cyc["t"], -1) == 170
    but = e(
        "reaction_kinetics", mode="integrated", order=2, k=5.76e-2 / 60, c0=0.200, t=[10.0, "min"]
    )
    assert round(but["c"], 3) == 0.179 and round(but["t_half"] / 60, 1) == 86.8
    n2o5 = e("reaction_kinetics", mode="arrhenius", k1=1.66, T1=650, k2=7.39, T2=700)
    assert round(n2o5["Ea"], -4) == 110000


def test_source_nernst_and_K() -> None:
    """LibreTexts, Nernst Equation: Zn concentration cell, K of Fe + Zn2+, Ksp of AgCl."""
    zn = e("electrochemistry", mode="nernst", E0=0.0, n=2, Q=0.024 / 2.4, T=298.15)
    assert round(zn["E"], 4) == 0.0592
    fe = e("electrochemistry", mode="gibbs", n=2, E=-0.353, T=298.15)
    assert round(fe["lnK"] / math.log(10), 2) == -11.93
    ag = e("electrochemistry", mode="gibbs", n=1, E=-0.577, T=298.15)
    assert round(ag["lnK"] / math.log(10), 2) == -9.75


def test_source_openstax_faraday() -> None:
    """OpenStax Chemistry 2e 17.7: silver plating (10.23 A for 1 h) and aluminium (25.0 A for 15.0 min)."""
    ag = e("electrochemistry", mode="electrolysis", I=10.23, t=[1, "h"], n=1, M=[107.9, "g/mol"])
    assert (
        abs(ag["m"] * 1000 - 41.19) <= 0.01
    )  # the source rounds the moles to 0.3817 before multiplying
    al = e("electrochemistry", mode="electrolysis", I=25.0, t=[15.0, "min"], n=3, formula="Al")
    assert round(al["m"] * 1000, 2) == 2.10
