"""Kinetic theory of gases (D87).

Checks independent of the server formulas:
- typical speeds as moments of the Maxwell-Boltzmann speed distribution, integrated with mpmath at 30 digits; the
  most probable speed as the root of df/dv; the mean relative speed as the mean speed of the distribution with the
  reduced mass m/2;
- the speed distribution against scipy's chi distribution with 3 degrees of freedom (scale a = sqrt(k T / m)) and
  against mpmath integrals of the density, also far in the tail;
- the flux on a wall as the double integral over speed and angle of the page "Kinetic theory of gases";
- page values: N2 at 300 K (v_p about 422 m/s, c about 353 m/s), monatomic gas (12.47 J/K per mole, 3406 J per
  mole and 5.65 zJ = 35.2 meV per molecule at 273.15 K), air at ambient pressure (mean free path 64-68 nm from the
  table and from the viscosity form), Graham's examples (H2 against O2 = 4; 0.25 times He gives 64 g/mol);
- NIST: the heat capacity of argon gas is 20.786 J/(mol K) (5/2 R).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mpmath as mp
from scipy import stats

from tests.conftest import call, ok

K_B = 1.380649e-23
N_A = 6.02214076e23
R = K_B * N_A
EV = 1.602176634e-19


def kt(**params: object) -> dict:
    return ok("eng_calc", operation="kinetic_theory", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="kinetic_theory", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def close(a: float, b: object, tol: float = 1e-12) -> bool:
    return math.isclose(a, float(b), rel_tol=tol)  # type: ignore[arg-type]


def _speed_pdf(m: mp.mpf, t: float) -> Callable[[mp.mpf], mp.mpf]:
    a2 = mp.mpf(K_B) * t / m
    return lambda v: mp.sqrt(2 / mp.pi) * v**2 / a2 ** mp.mpf(1.5) * mp.exp(-(v**2) / (2 * a2))


def test_speeds_are_the_moments_of_the_distribution() -> None:
    mp.mp.dps = 30
    for molar, t in ((0.028, 300.0), (4.0026e-3, 20.0), (0.2, 1500.0)):
        m = mp.mpf(molar) / N_A
        f = _speed_pdf(m, t)
        assert abs(mp.quad(f, [0, mp.inf]) - 1) < 1e-20
        mean = mp.quad(lambda v, f=f: v * f(v), [0, mp.inf])
        rms = mp.sqrt(mp.quad(lambda v, f=f: v**2 * f(v), [0, mp.inf]))
        mode = mp.findroot(lambda v, f=f: mp.diff(f, v), 1.1 * mp.sqrt(2 * K_B * t / m))
        rel_mean = mp.quad(lambda v, g=_speed_pdf(m / 2, t): v * g(v), [0, mp.inf])
        res = kt(mode="speeds", T=t, M_molar=molar, dof=5)
        assert close(res["v_mean"], mean) and close(res["v_rms"], rms)
        assert close(res["v_p"], mode, 1e-10) and close(res["v_rel"], rel_mean)
        assert close(res["m"], m) and close(res["E_k"], m * rms**2 / 2)
        assert close(res["E_k_molar"], 1.5 * R * t)
        # speed of sound of the ideal gas c = sqrt(gamma R T / M), gamma = 7/5 for 5 degrees of freedom
        assert close(res["gamma"], 1.4) and close(res["c_sound"], math.sqrt(1.4 * R * t / molar))
    page = kt(mode="speeds", T=300.0, M_molar=[28, "g/mol"], dof=5)
    assert round(page["v_p"]) == 422 and round(page["c_sound"]) == 353
    # the mass of one molecule in u gives the same speeds (M_u = 0.99999999965 g/mol)
    in_u = kt(mode="speeds", T=[26.85, "C"], m=[28, "u"])
    assert close(in_u["v_rms"], page["v_rms"], 1e-9) and "c_sound" not in in_u


def test_pressure_from_the_mean_square_speed() -> None:
    p_ideal, t, molar = 101325.0, 288.15, 0.0289647
    rho = p_ideal * molar / (R * t)
    v_rms = kt(mode="speeds", T=t, M_molar=molar)["v_rms"]
    res = kt(mode="pressure", rho=rho, v_rms=v_rms, M_molar=molar)
    assert close(res["p"], p_ideal) and close(res["T"], t)
    assert close(res["n"], p_ideal / (K_B * t)) and close(res["K_t_density"], 1.5 * p_ideal)
    alone = kt(mode="pressure", rho=1.2, v_rms=500.0)
    assert close(alone["p"], 1.2 * 500.0**2 / 3) and "T" not in alone


def test_distribution_against_scipy_and_mpmath() -> None:
    mp.mp.dps = 30
    t, molar = 500.0, 0.03199
    m = mp.mpf(molar) / N_A
    f = _speed_pdf(m, t)
    a = math.sqrt(K_B * t / float(m))
    chi = stats.chi(3, scale=a)
    for v in (50.0, 400.0, 900.0, 2500.0):
        res = kt(mode="distribution", T=t, M_molar=molar, v=v)
        assert close(res["pdf"], f(v)) and close(res["pdf"], chi.pdf(v), 1e-10)
        assert close(res["cdf"], mp.quad(f, [0, v]), 1e-11) and close(res["cdf"], chi.cdf(v), 1e-10)
        assert close(res["fraction_above"], mp.quad(f, [v, mp.inf]), 1e-11)
    # far tail: no cancellation in 1 - F (about 4e-173 at 20 times the most probable speed); one plain mpmath
    # quad over [far, inf] is off by 1e-4 there, so the reference is scipy's chi survival function
    far = 20 * math.sqrt(2) * a
    tail = kt(mode="distribution", T=t, M_molar=molar, v=far)["fraction_above"]
    assert close(tail, chi.sf(far), 1e-10) and tail < 1e-170
    band = kt(mode="distribution", T=t, M_molar=molar, v1=300.0, v2=700.0)
    assert close(band["fraction_between"], mp.quad(f, [300, 700]), 1e-11)
    high = kt(mode="distribution", T=t, M_molar=molar, v1=3000.0, v2=3100.0)
    assert close(high["fraction_between"], mp.quad(f, [3000, 3100]), 1e-10)


def test_mean_free_path_and_collisions() -> None:
    mp.mp.dps = 30
    t, p, d, molar = 293.15, 101325.0, 3.64e-10, 0.028
    res = kt(mode="mean_free_path", T=t, p=p, d=d, M_molar=molar, L=1e-6)
    n = p / (K_B * t)
    sigma = math.pi * d * d
    # collision frequency = n sigma <|v1 - v2|>, the mean relative speed from the reduced mass m/2
    m = mp.mpf(molar) / N_A
    v_rel = mp.quad(lambda v: v * _speed_pdf(m / 2, t)(v), [0, mp.inf])
    assert close(res["n"], n) and close(res["sigma"], sigma)
    assert close(res["z"], n * sigma * v_rel) and close(res["tau"], 1 / float(n * sigma * v_rel))
    assert close(res["ell"] * res["z"], res["v_mean"]) and close(res["Kn"], res["ell"] / 1e-6)
    # page table: air at ambient pressure 64-68 nm; same with sigma given as an area
    assert 64e-9 <= res["ell"] <= 68e-9
    assert close(kt(mode="mean_free_path", T=t, p=p, sigma=sigma)["ell"], res["ell"])
    # hard spheres with the viscosity of air at 25 C and 1 bar (18.5 uPa s, R = 287 J/(kg K)): about 68 nm
    visc = kt(mode="mean_free_path", T=[25, "C"], p=[1, "bar"], mu=18.5e-6, M_molar=R / 287.0)
    assert close(visc["ell"], 18.5e-6 / 1e5 * math.sqrt(math.pi * 287.0 * 298.15 / 2))
    assert 64e-9 <= visc["ell"] <= 69e-9
    assert "exactly one" in bad({"mode": "mean_free_path", "T": t, "p": p, "d": d, "sigma": 1e-19})
    assert "molar mass" in bad({"mode": "mean_free_path", "T": t, "p": p, "mu": 1e-5})


def test_wall_flux_and_graham() -> None:
    mp.mp.dps = 30
    t, p, molar, area = 300.0, 10.0, 0.04, 1e-6
    res = kt(mode="wall_flux", T=t, p=p, M_molar=molar, A=area)
    m = mp.mpf(molar) / N_A
    n = mp.mpf(p) / (K_B * t)
    pre = (m / (2 * mp.pi * K_B * t)) ** mp.mpf(1.5)

    def inner(v: mp.mpf) -> mp.mpf:
        weight = mp.quad(lambda th: mp.cos(th) * mp.sin(th), [0, mp.pi / 2])
        return v**3 * pre * mp.exp(-m * v**2 / (2 * K_B * t)) * 2 * mp.pi * weight

    flux = n * mp.quad(inner, [0, mp.inf])
    assert close(res["J"], flux) and close(res["Phi"], flux * area)
    assert close(res["mdot"], flux * area * m) and close(res["J"], res["n"] * res["v_mean"] / 4)
    h2 = kt(mode="graham", M1=[2, "g/mol"], M2=[32, "g/mol"])
    assert close(h2["rate_ratio"], 4.0) and close(h2["time_ratio"], 0.25)
    # page: a gas that effuses 0.25 times as fast as He (rate_He / rate_2 = 4) has M2 = 64 g/mol
    unknown = kt(mode="graham", M1=[4, "g/mol"], rate_ratio=4.0)
    assert close(unknown["M2"], 0.064) and close(unknown["time_ratio"], 0.25)
    # the ratio of two wall fluxes at the same p and T is Graham's ratio
    j1 = kt(mode="wall_flux", T=t, p=p, M_molar=0.002)["J"]
    j2 = kt(mode="wall_flux", T=t, p=p, M_molar=0.032)["J"]
    assert close(j1 / j2, 4.0)
    assert "exactly one" in bad({"mode": "graham", "M1": 0.004})


def test_equipartition() -> None:
    mono = kt(mode="equipartition", gas="monatomic", T=273.15, n_mol=2.0)
    assert mono["dof"] == 3 and close(mono["gamma"], 5 / 3)
    assert round(mono["Cp_molar"], 3) == 20.786  # NIST, argon gas
    assert (
        round(mono["Cv_molar"], 2) == 12.47 and math.floor(mono["U_molar"]) == 3406
    )  # page: 3406 J
    # page values are rounded (5.65 zJ, 35.2 meV; 3/2 k T = 5.657 zJ = 35.31 meV): within 0.5 %
    assert close(mono["E_molecule"], 1.5 * K_B * 273.15)
    assert close(mono["E_molecule"], 5.65e-21, 5e-3) and close(
        mono["E_molecule"] / EV, 0.0352, 5e-3
    )
    assert close(mono["C_V"], 2 * mono["Cv_molar"]) and close(mono["U"], 2 * mono["U_molar"])
    for gas, dof, gamma in (("diatomic", 5, 1.4), ("polyatomic", 6, 4 / 3)):
        r = kt(mode="equipartition", gas=gas)
        assert r["dof"] == dof and close(r["gamma"], gamma)
        assert close(r["Cp_molar"] - r["Cv_molar"], R)  # Mayer
    seven = kt(mode="equipartition", dof=7, T=1000.0)
    assert close(seven["Cv_molar"], 3.5 * R) and close(seven["E_molecule"], 3.5 * K_B * 1000.0)
    assert "exactly one" in bad({"mode": "equipartition", "gas": "monatomic", "dof": 3})


def test_bad_inputs() -> None:
    assert "exactly one" in bad({"mode": "speeds", "T": 300.0})
    assert "exactly one" in bad({"mode": "speeds", "T": 300.0, "m": 1e-26, "M_molar": 0.028})
    assert "Keys not used" in bad({"mode": "speeds", "T": 300.0, "M_molar": 0.028, "p": 1e5})
    assert "v1 < v2" in bad(
        {"mode": "distribution", "T": 300.0, "M_molar": 0.028, "v1": 500.0, "v2": 400.0}
    )
    assert "v" in bad({"mode": "distribution", "T": 300.0, "M_molar": 0.028})
    assert "absolute zero" in bad({"mode": "speeds", "T": [-300, "C"], "M_molar": 0.028})
    assert ">= 1" in bad({"mode": "speeds", "T": 300.0, "M_molar": 0.028, "dof": 0})
