"""Nuclear physics, modern physics and special relativity (D49).

Independent methods: binding energy from the atomic masses against B/A published by AME2020;
Q value against the CODATA masses of proton, alpha and electron and against the binding energies;
threshold against the invariant mass; decays against solve_ivp; maximum of the daughter against
minimize_scalar; Compton against conservation of energy and momentum; Wien against the
numerical maximum of Planck; black-body band against the numerical integral in wavelength;
composition of Lorentz transformations against the velocity addition; velocity addition
against the rapidities. Numerical examples: OpenStax College Physics 2e and University Physics vol. 3
The sources use constants with 3-4 digits, so the comparison is at their digits.
"""

from __future__ import annotations

import gzip
import json
import math

import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize_scalar

from oce_math_mcp import constants
from tests.conftest import ROOT, call, ok

C = 299792458.0
EV = 1.602176634e-19
H = constants.lookup("h")[1]
U_MEV = constants.lookup("atomic mass constant energy equivalent in MeV")[1]
ME_MEV = constants.lookup("electron mass energy equivalent in MeV")[1]
YJ = 365.25 * 86400  # Julian year of the parameters
NUC = json.loads(gzip.decompress((ROOT / "src/oce_math_mcp/data/nuclides.json.gz").read_bytes()))


def e(operation: str, **params: object) -> dict:
    return ok("eng_calc", operation=operation, params=params)


def full(operation: str, **params: object) -> dict:
    resp = call("eng_calc", operation=operation, params=params)
    assert resp["ok"], resp
    return resp


def bad(operation: str, params: dict) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1e-300)


# ------------------------------------------------------------------ nuclide data


def test_data_file_counts_and_source() -> None:
    n = NUC["nuclides"]
    assert len(n) == 5383 and sum(1 for r in n.values() if not r[9]) == 3558
    assert "CC BY 3.0" in NUC["source"]["license"]
    assert len(NUC["source"]["sha256_ame"]) == 64


@pytest.mark.parametrize("name", ["He-4", "C-12", "O-16", "Fe-56", "Ni-62", "Pb-208", "U-238"])
def test_binding_from_atomic_masses_matches_ame(name: str) -> None:
    """B recomputed from the atomic masses in u (other AME column) against the published B/A."""
    r = e("nuclide_data", nuclide=name)
    h, n = e("nuclide_data", nuclide="H-1"), e("nuclide_data", nuclide="n")
    b_u = r["Z"] * h["atomic_mass_u"] + r["N"] * n["atomic_mass_u"] - r["atomic_mass_u"]
    assert rel(b_u * U_MEV, r["binding_energy_MeV"], 2e-7)
    published = float(NUC["nuclides"][name][7]) / 1000
    assert abs(r["binding_per_nucleon_MeV"] - published) < 1e-7


def test_names_isomers_and_errors() -> None:
    for txt in ("U-235", "235U", "u235", "U 235"):
        assert e("nuclide_data", nuclide=txt)["nuclide"] == "U-235"
    tc = e("nuclide_data", nuclide="99mTc")
    assert tc["nuclide"] == "Tc-99m" and tc["isomer"] and rel(tc["excitation_keV"], 142.6836)
    assert rel(tc["half_life_s"], 6.0066 * 3600)
    assert e("nuclide_data", nuclide="alpha")["nuclide"] == "He-4"
    assert "from A =" in bad("nuclide_data", {"nuclide": "U-300"})
    assert "not recognised" in bad("nuclide_data", {"nuclide": "Xx-12"})


def test_half_life_abundance_and_stability() -> None:
    co = e("nuclide_data", nuclide="Co-60")
    assert rel(co["half_life_s"], 5.2714 * 365.2422 * 86400)  # tropical year of NUBASE2020
    assert rel(co["decay_constant"] * co["mean_life_s"], 1.0)
    o = [e("nuclide_data", nuclide=f"O-{a}") for a in (16, 17, 18)]
    assert all(x["stable"] for x in o) and abs(sum(x["abundance_percent"] for x in o) - 100) < 0.01
    assert full("nuclide_data", nuclide="Og-294")["warnings"]


def test_source_binding_energy_he4() -> None:
    """OpenStax College Physics 2e, ex. 31.7: B = 28.3 MeV, B/A = 7.07 MeV (atomic masses)."""
    r = e("nuclide_data", nuclide="He-4")
    assert (
        round(r["binding_energy_MeV"], 1) == 28.3 and round(r["binding_per_nucleon_MeV"], 2) == 7.07
    )


# ------------------------------------------------------------------ decay


def test_decay_law_against_ode() -> None:
    r = e("radioactive_decay", mode="law", nuclide="I-131", N0=1e20, t=[10, "d"])
    lam = r["lambda"]
    sol = solve_ivp(lambda _t, y: [-lam * y[0]], (0, 10 * 86400), [1e20], rtol=1e-11, atol=1)
    assert rel(r["N"], sol.y[0][-1], 1e-8) and rel(r["A"], lam * r["N"])
    back = e("radioactive_decay", mode="law", nuclide="I-131", fraction=r["fraction"])
    assert rel(back["t"], 10 * 86400, 1e-12)


def test_source_sr90_activity() -> None:
    """OpenStax University Physics 3, ex. 10.4: Sr-90, 1.00 g, 89.91 g/mol (year of the source 3.16e7 s)."""
    r = e(
        "radioactive_decay", mode="activity", T_half=[28.8, "y"], M=[89.91, "g/mol"], m=[1.00, "g"]
    )
    assert (
        rel(r["lambda"], 7.61e-10, 5e-3)
        and rel(r["N"], 6.70e21, 5e-3)
        and rel(r["A"], 5.10e12, 5e-3)
    )


def test_source_carbon_dating() -> None:
    """OpenStax University Physics 3, ex. 10.6: N/N0 = 0.80, T = 5730 years, about 1844 years."""
    r = e("radioactive_decay", mode="law", T_half=[5730, "y"], fraction=0.80)
    assert abs(r["t"] / YJ - 1844) < 1


def test_chain_against_ode_and_maximum() -> None:
    p = {"mode": "chain", "parent": "Mo-99", "daughter": "Tc-99m", "N1_0": 1e15, "branching": 0.876}
    r = e("radioactive_decay", **p, t=[24, "h"])
    l1, l2 = r["lambda1"], r["lambda2"]

    def rhs(_t: float, y: list[float]) -> list[float]:
        return [-l1 * y[0], 0.876 * l1 * y[0] - l2 * y[1]]

    sol = solve_ivp(rhs, (0, 24 * 3600), [1e15, 0], rtol=1e-11, atol=1)
    assert rel(r["N1"], sol.y[0][-1], 1e-8) and rel(r["N2"], sol.y[1][-1], 1e-7)
    tmax = e("radioactive_decay", **p, t=0)["t_max_daughter"]

    def n2(t: float) -> float:
        return -e("radioactive_decay", **p, t=t)["N2"]

    found = minimize_scalar(n2, bounds=(3600, 100 * 3600), method="bounded", options={"xatol": 1})
    assert abs(found.x - tmax) < 30 and r["equilibrium"] == "transient"
    ra = e(
        "radioactive_decay",
        mode="chain",
        parent="Ra-226",
        daughter="Rn-222",
        N1_0=1e20,
        t=[60, "d"],
    )
    assert ra["equilibrium"] == "secular" and abs(ra["activity_ratio"] - 1) < 1e-3


def test_decay_errors() -> None:
    assert "stable" in bad("radioactive_decay", {"mode": "law", "nuclide": "C-12", "t": 1})
    assert "exactly one" in bad("radioactive_decay", {"mode": "law", "T_half": 1, "tau": 2, "t": 1})


# ------------------------------------------------------------------ reactions and Q


def test_q_pp_chain_against_codata_masses() -> None:
    """4 p -> alpha + 2 e+ + 2 nu with the CODATA masses of proton, alpha, electron.

    The server uses M_atom - Z m_e and neglects the electron binding: the expected difference is
    B_e(He) - 4 B_e(H), with B_e(H) = Ry mu/m_e and B_e(He) = 24.5874 eV (first ionisation, NIST
    ASD) + 4 Ry mu/m_e (He+).
    """
    r = e("nuclear_reaction", reaction="4 p -> alpha + 2 e+ + 2 nu")
    mp = constants.lookup("proton mass energy equivalent in MeV")[1]
    ma = constants.lookup("alpha particle mass energy equivalent in MeV")[1]
    ry = constants.lookup("Rydberg constant times hc in eV")[1]
    b_h = ry / (1 + ME_MEV / mp)
    b_he = 24.5874 + 4 * ry / (1 + ME_MEV / ma)
    diff_ev = (r["Q_MeV"] - (4 * mp - ma - 2 * ME_MEV)) * 1e6
    assert abs(diff_ev - (b_he - 4 * b_h)) < 0.5


@pytest.mark.parametrize(
    ("reaction", "parent", "products"),
    [
        ("Pu-239 -> U-235 + He-4", "Pu-239", ("U-235", "He-4")),
        ("Ra-226 -> Rn-222 + alpha", "Ra-226", ("Rn-222", "He-4")),
    ],
)
def test_q_alpha_against_binding_and_two_body(
    reaction: str, parent: str, products: tuple[str, str]
) -> None:
    r = e("nuclear_reaction", reaction=reaction)
    b = [e("nuclide_data", nuclide=x)["binding_energy_MeV"] for x in (parent, *products)]
    assert abs(r["Q_MeV"] - (b[1] + b[2] - b[0])) < 1e-6
    t1, t2 = r["T_products_MeV"]
    assert abs(t1 + t2 - r["Q_MeV"]) < 1e-9
    m = [e("nuclide_data", nuclide=x)["nuclear_mass_MeV"] for x in products]
    assert rel(math.sqrt(t1 * (t1 + 2 * m[0])), math.sqrt(t2 * (t2 + 2 * m[1])), 1e-9)


def test_positron_capture_and_isomer() -> None:
    """beta+: Q = Delta(parent) - Delta(daughter) - 2 m_e; capture: Q = atomic difference; isomer = excitation."""
    me = {k: float(NUC["nuclides"][k][2]) for k in ("Na-22", "Ne-22", "Be-7", "Li-7")}
    bp = e("nuclear_reaction", reaction="Na-22 -> Ne-22 + e+ + nu")
    assert rel(bp["Q_keV"], me["Na-22"] - me["Ne-22"] - 2 * ME_MEV * 1e3, 1e-12)
    ec = e("nuclear_reaction", reaction="Be-7 + e- -> Li-7 + nu")
    assert rel(ec["Q_keV"], me["Be-7"] - me["Li-7"], 1e-12)
    it = e("nuclear_reaction", reaction="Tc-99m -> Tc-99 + gamma")
    assert abs(it["Q_keV"] - 142.6836) < 1e-6


def test_threshold_against_invariant_mass() -> None:
    r = e("nuclear_reaction", reaction="N-14 + alpha -> O-17 + p", projectile="alpha")
    assert r["Q_MeV"] < 0 and not r["exothermic"]
    ma, mn = (e("nuclide_data", nuclide=x)["nuclear_mass_MeV"] for x in ("He-4", "N-14"))
    mo, mp = (e("nuclide_data", nuclide=x)["nuclear_mass_MeV"] for x in ("O-17", "H-1"))
    s = (ma + mn) ** 2 + 2 * mn * r["T_threshold_MeV"]
    assert rel(math.sqrt(s), mo + mp, 1e-12)
    assert rel(r["T_threshold_classical_MeV"], -r["Q_MeV"] * (1 + ma / mn), 1e-12)


def test_conservation_checks() -> None:
    assert "Nucleon" in bad("nuclear_reaction", {"reaction": "U-235 + n -> Ba-141 + Kr-92 + 2 n"})
    assert "Charge" in bad("nuclear_reaction", {"reaction": "n -> p + nu"})
    w = full("nuclear_reaction", reaction="n -> p + e-")
    assert any("Lepton" in x for x in w["warnings"])
    beta = e("nuclear_reaction", reaction="n -> p + e- + anti-nu")
    assert rel(
        beta["Q_keV"], float(NUC["nuclides"]["n-1"][2]) - float(NUC["nuclides"]["H-1"][2]), 1e-12
    )


def test_source_openstax_q_values() -> None:
    """College Physics 2e ex. 31.2 and 31.3, University Physics 3 ex. 10.9, College Physics 2e ex. 32.2."""
    assert rel(e("nuclear_reaction", reaction="Pu-239 -> U-235 + He-4")["Q_MeV"], 5.25, 3e-3)
    assert (
        round(e("nuclear_reaction", reaction="Co-60 -> Ni-60 + e- + anti-nu")["Q_MeV"], 2) == 2.82
    )
    fis = e("nuclear_reaction", reaction="U-238 -> Sr-95 + Xe-140 + 3 n")
    assert abs(fis["Q_MeV"] - 171.2) < 0.5
    dt = e("nuclear_reaction", reaction="H-2 + H-3 -> He-4 + n")
    assert round(dt["Q_MeV"], 2) == 17.59 and rel(dt["energy_per_kg_reactants_J"], 3.37e14, 3e-3)


# ------------------------------------------------------------------ modern physics


def test_source_photons() -> None:
    """College Physics 2e ex. 29.3 and 29.5."""
    assert round(e("modern_physics", mode="photon", wavelength=[100, "nm"])["energy_eV"], 1) == 12.4
    r = e("modern_physics", mode="photon", wavelength=[500, "nm"])
    assert rel(r["momentum"], 1.33e-27, 4e-3) and round(r["energy_eV"], 2) == 2.48


def test_source_photoelectric() -> None:
    """University Physics 3, ex. 6.5, 6.6, 6.7."""
    a = e("modern_physics", mode="photoelectric", wavelength=[180, "nm"], stopping_voltage=0.80)
    assert round(a["work_function_eV"], 2) == 6.09 and rel(a["f_cutoff"], 1.47e15, 4e-3)
    b = e(
        "modern_physics", mode="photoelectric", wavelength=[430, "nm"], work_function=[2.71, "eV"]
    )
    assert round(b["photon_energy_eV"], 2) == 2.88 and round(b["K_max_eV"], 2) == 0.17
    c = e("modern_physics", mode="photoelectric", work_function=[4.73, "eV"])
    assert round(c["wavelength_cutoff"] * 1e9) == 262


def test_compton_conservation_and_source() -> None:
    r = e("modern_physics", mode="compton", wavelength=[71, "pm"], angle_deg=30)
    assert round(r["shift"] * 1e12, 3) == 0.325 and rel(r["max_shift"], 4.86e-12, 3e-3)
    th, ph = math.radians(30), math.radians(r["recoil_angle_deg"])
    p1, p2 = r["energy_eV"] * EV / C, r["energy_scattered_eV"] * EV / C
    me = constants.lookup("electron mass")[1]
    ke = r["electron_kinetic_eV"] * EV
    pe = math.sqrt(ke * (ke + 2 * me * C**2)) / C
    assert rel(p1, p2 * math.cos(th) + pe * math.cos(ph), 1e-9)
    assert abs(p2 * math.sin(th) - pe * math.sin(ph)) < 1e-9 * p1


def test_de_broglie_paths_and_source() -> None:
    """University Physics 3, ex. 6.11 and 6.14."""
    a = e("modern_physics", mode="de_broglie", particle="electron", kinetic_energy=[1.0, "eV"])
    assert round(a["wavelength"] * 1e9, 2) == 1.23
    b = e("modern_physics", mode="de_broglie", particle="electron", kinetic_energy=[108, "keV"])
    assert round(b["wavelength"] * 1e12, 2) == 3.55
    p = e("modern_physics", mode="de_broglie", particle="proton", v=0.75 * C)
    # source 1.16 fm with hc = 1.241 eV um and E0 = 938 MeV; CODATA gives 1.1654 fm
    assert abs(p["wavelength"] * 1e15 - 1.16) < 0.01 and rel(p["beta"] * p["gamma"], 1.134, 1e-3)
    assert rel(p["kinetic_energy_eV"] / 1e6, 480.1, 1e-3)
    same = e("modern_physics", mode="de_broglie", particle="electron", voltage=108e3)
    back = e("modern_physics", mode="de_broglie", particle="electron", p=same["p"])
    assert rel(same["wavelength"], b["wavelength"], 1e-12) and rel(back["v"], same["v"], 1e-9)


def test_hydrogen_and_source() -> None:
    """University Physics 3, ex. 6.10; College Physics 2e ex. 30.1."""
    t = e("modern_physics", mode="hydrogen_like", n1=4, n2=2)
    assert round(t["wavelength"] * 1e9) == 486 and t["series"] == "Balmer"
    rinf = constants.lookup("Rydberg constant")[1]
    assert rel(1 / t["wavelength"], rinf * (1 / 4 - 1 / 16), 1e-9)
    ab = e("modern_physics", mode="hydrogen_like", n1=1, wavelength=[93.7, "nm"])
    # the source writes n = 6.07 with 13.6 eV and hc = 1240 eV nm; 1/n^2 = 1 - E/Ry subtracts close
    # numbers and amplifies the rounding: with CODATA it gives 6.03. What counts is n = 6 and E_6.
    assert ab["n2"] == 6 and abs(ab["n2_exact"] - 6) < 0.1 and round(ab["E_n2_eV"], 3) == -0.378
    he = e("modern_physics", mode="hydrogen_like", Z=2)
    assert rel(
        he["ionization_energy_eV"], 4 * constants.lookup("Rydberg constant times hc in eV")[1]
    )
    red = e("modern_physics", mode="hydrogen_like", nuclide="H-1", n=1)
    mu = 1 / (1 + 1 / constants.lookup("proton-electron mass ratio")[1])
    assert rel(red["reduced_mass_factor"], mu, 1e-9)


def test_blackbody_numeric_and_source() -> None:
    """University Physics 3, ex. 6.2; Wien against the numerical maximum of Planck."""
    wd = e("modern_physics", mode="blackbody", T=2.5e4)
    rg = e("modern_physics", mode="blackbody", T=3.0e3)
    assert rel(wd["exitance"], 2.2e10, 2e-2) and rel(rg["exitance"], 4.6e6, 1e-2)
    assert round(wd["exitance"] / rg["exitance"], -1) == 4820
    t = 5800.0
    peak = e("modern_physics", mode="blackbody", T=t)["wavelength_max"]

    def neg(lam_nm: float) -> float:
        return -e("modern_physics", mode="blackbody", T=t, wavelength=lam_nm * 1e-9)[
            "spectral_radiance"
        ]

    found = minimize_scalar(neg, bounds=(300, 800), method="bounded", options={"xatol": 1e-3})
    assert abs(found.x * 1e-9 - peak) < 2e-12
    band = e(
        "modern_physics", mode="blackbody", T=t, wavelength1=[400, "nm"], wavelength2=[700, "nm"]
    )

    def m_lam(lam: float) -> float:
        return e("modern_physics", mode="blackbody", T=t, wavelength=lam)["spectral_exitance"]

    assert rel(band["band_exitance"], quad(m_lam, 400e-9, 700e-9)[0], 1e-6)
    whole = e("modern_physics", mode="blackbody", T=t, wavelength1=[1, "nm"], wavelength2=[1, "m"])
    assert abs(whole["band_fraction"] - 1) < 1e-6


def test_uncertainty() -> None:
    r = e("modern_physics", mode="uncertainty", dx=[0.1, "nm"], particle="electron")
    hbar = constants.lookup("hbar")[1]
    assert rel(r["dp_min"], hbar / 2e-10) and rel(
        r["dv_min"], r["dp_min"] / constants.lookup("m_e")[1]
    )


# ------------------------------------------------------------------ relativity


def test_source_time_length() -> None:
    """University Physics 3, ex. 5.3 (muon) and 5.5 (contraction)."""
    mu = e("special_relativity", mode="kinematics", beta=0.950, proper_time=[2.20, "us"])
    assert round(mu["gamma"], 2) == 3.20 and round(mu["time"] * 1e6, 2) == 7.05
    ln = e("special_relativity", mode="kinematics", gamma=30.0, proper_length=4.300)
    assert round(ln["length"], 4) == 0.1433 and round(ln["beta"], 5) == 0.99944


def test_velocity_addition_rapidity_and_source() -> None:
    """University Physics 3, ex. 5.9 and 5.10; against tanh(atanh(a) + atanh(b))."""
    assert rel(
        e("special_relativity", mode="velocity_addition", u_beta=0.75, v_beta=0.5)["beta"], 10 / 11
    )
    assert rel(
        e("special_relativity", mode="velocity_addition", u_beta=-0.75, v_beta=0.5)["beta"], -0.4
    )
    assert e("special_relativity", mode="velocity_addition", u_beta=1, v_beta=0.5)["beta"] == 1
    r = e("special_relativity", mode="velocity_addition", u_beta=0.3, v_beta=0.6)
    assert rel(r["beta"], math.tanh(math.atanh(0.3) + math.atanh(0.6)), 1e-12)


def test_energy_momentum_and_source() -> None:
    """University Physics 3, ex. 5.12 and 5.13 (source with c = 3.00e8 m/s)."""
    r = e("special_relativity", mode="kinematics", particle="electron", beta=0.990)
    assert rel(r["K_MeV"], 3.12, 5e-3) and rel(r["K_J"] / r["K_classical_J"], 12.4, 5e-3)
    assert rel(r["E_MeV"] ** 2, r["p_MeV_c"] ** 2 + r["E0_MeV"] ** 2, 1e-12)
    me = e("special_relativity", mode="mass_energy", m=[1.00, "g"])
    assert rel(me["energy_J"], 9.00e13, 2e-3)
    k = e("special_relativity", mode="kinematics", particle="proton", kinetic_energy=[480.1, "MeV"])
    assert rel(k["beta"], 0.75, 1e-3)


def test_doppler_and_source() -> None:
    """University Physics 3, ex. 5.11; College Physics 2e 28.4."""
    a = e("special_relativity", mode="doppler", beta=0.825, wavelength=0.525)
    assert round(a["wavelength_observed"], 2) == 1.70
    b = e("special_relativity", mode="doppler", beta=0.350, f=[1.50, "GHz"])
    assert round(b["f_observed"] / 1e9, 2) == 1.04
    inv = e(
        "special_relativity",
        mode="doppler",
        wavelength=0.525,
        wavelength_observed=a["wavelength_observed"],
    )
    assert rel(inv["beta"], 0.825, 1e-12)


def test_lorentz_composition_and_interval() -> None:
    x, t = 1200.0, 3e-6
    one = e("special_relativity", mode="lorentz", x=x, t=t, beta=0.3)
    two = e("special_relativity", mode="lorentz", x=one["x_prime"], t=one["t_prime"], beta=0.6)
    b12 = (0.3 + 0.6) / (1 + 0.18)
    direct = e("special_relativity", mode="lorentz", x=x, t=t, beta=b12)
    assert rel(two["x_prime"], direct["x_prime"], 1e-9) and rel(
        two["t_prime"], direct["t_prime"], 1e-9
    )
    s2 = (C * direct["t_prime"]) ** 2 - direct["x_prime"] ** 2
    assert rel(s2, one["interval_s2"], 1e-9) and one["interval_type"] == "spacelike"


def test_relativity_errors() -> None:
    assert "speed of light" in bad("special_relativity", {"mode": "kinematics", "beta": 1.2})
    assert "mass" in bad("special_relativity", {"mode": "kinematics", "kinetic_energy": 1})
