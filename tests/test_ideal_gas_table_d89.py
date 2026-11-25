"""The ideal-gas table of the page "Table of thermodynamic equations" (section Kinetic theory), row by row (D89).

D87 counted the section as covered by `ideal_gas` and `kinetic_theory`; this test checks it. The page works with
n moles and C_V, C_p of the amount; the server works with the mass and the specific gas constant (m R = n R_u).
The page's work W is done ON the gas (Delta U = Q + W); the server's W is done BY the gas, so W_page = -W.
"""

from __future__ import annotations

import math

from tests.conftest import ok

R_U = 1.380649e-23 * 6.02214076e23
N_MOL, M_MOLAR, GAMMA = 2.0, 0.028, 1.4  # diatomic: C_V = 5/2 n R, C_p = 7/2 n R on the page
CV, CP = 2.5 * N_MOL * R_U, 3.5 * N_MOL * R_U
P1, T1 = 2.0e5, 300.0
V1 = N_MOL * R_U * T1 / P1


def proc(process: str, **state2: float) -> dict:
    return ok(
        "eng_calc",
        operation="ideal_gas",
        params={
            "mode": "process",
            "process": process,
            "p1": P1,
            "T1": T1,
            "V1": V1,
            "R": R_U / M_MOLAR,
            "gamma": GAMMA,
            **state2,
        },
    )


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=1e-9)


def test_heat_capacities_of_the_page() -> None:
    for gas, cv_page, cp_page in (("monatomic", 1.5, 2.5), ("diatomic", 2.5, 3.5)):
        eq = ok(
            "eng_calc",
            operation="kinetic_theory",
            params={"mode": "equipartition", "gas": gas, "n_mol": N_MOL},
        )
        assert close(eq["C_V"], cv_page * N_MOL * R_U) and close(eq["C_p"], cp_page * N_MOL * R_U)


def test_isobaric_column() -> None:
    r = proc("isobaric", V2=1.5 * V1)
    t2 = r["T2"]
    w_page = -P1 * (r["V2"] - V1)
    assert close(-r["W"], w_page) and close(r["dU"], CV * (t2 - T1))
    assert close(r["dU"], r["Q"] + w_page)  # Delta U = Q + W
    assert close(r["Q"], CP * (t2 - T1)) and close(r["dH"], CP * (t2 - T1))
    assert close(r["dS"], CP * math.log(t2 / T1))
    assert close(r["V2"] / t2, V1 / T1)  # constant V / T


def test_isochoric_column() -> None:
    r = proc("isochoric", p2=3.0e5)
    t2 = r["T2"]
    assert r["W"] == 0.0 and close(r["Q"], CV * (t2 - T1)) and close(r["dU"], r["Q"])
    assert close(r["dH"], r["Q"] + V1 * (r["p2"] - P1))  # Q_V + V Delta p
    assert close(r["dS"], CV * math.log(t2 / T1))
    assert close(r["p2"] / t2, P1 / T1)  # constant p / T


def test_isothermal_column() -> None:
    r = proc("isothermal", V2=2.5 * V1)
    w_page = -N_MOL * R_U * T1 * math.log(r["V2"] / V1)
    assert close(-r["W"], w_page) and close(-r["W"], -N_MOL * R_U * T1 * math.log(P1 / r["p2"]))
    assert r["dU"] == 0.0 and close(r["Q"], -w_page) and r["dH"] == 0.0
    assert close(r["dS"], N_MOL * R_U * math.log(r["V2"] / V1)) and close(r["dS"], r["Q"] / T1)
    assert close(r["p2"] * r["V2"], P1 * V1)  # constant p V


def test_adiabatic_column_and_general_entropy() -> None:
    r = proc("isentropic", V2=0.4 * V1)
    v2, t2, p2 = r["V2"], r["T2"], r["p2"]
    # page error: the cell writes P V^gamma (V_f^(1-gamma) - V_i^(1-gamma)) / (1 - gamma) = C_V (T2 - T1), but the
    # first expression is the integral of p dV (work BY the gas) and the second is the work ON the gas (Delta U):
    # they have opposite signs. Here a compression: the gas receives work.
    by_gas = P1 * V1**GAMMA * (v2 ** (1 - GAMMA) - V1 ** (1 - GAMMA)) / (1 - GAMMA)
    w_page = CV * (t2 - T1)
    assert close(r["W"], by_gas) and close(-r["W"], w_page) and w_page > 0
    assert r["Q"] == 0.0 and close(r["dU"], w_page) and close(r["dH"], CP * (t2 - T1))
    assert r["dS"] == 0.0 and close(CP * math.log(v2 / V1) + CV * math.log(p2 / P1), 0.0)
    assert close(p2 * v2**GAMMA, P1 * V1**GAMMA)  # constant p V^gamma
    # general entropy change, both forms of the page, on a polytropic process
    pol = proc("polytropic", n_poly=1.25, V2=1.8 * V1)
    ds1 = CV * math.log(pol["T2"] / T1) + N_MOL * R_U * math.log(pol["V2"] / V1)
    ds2 = CP * math.log(pol["T2"] / T1) - N_MOL * R_U * math.log(pol["p2"] / P1)
    assert close(pol["dS"], ds1) and close(pol["dS"], ds2)
