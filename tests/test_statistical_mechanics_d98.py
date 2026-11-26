"""D98: eng_calc.statistical_mechanics (entropy, Maxwell-Juttner, partition functions).

The references are independent of the module: mpmath sums and integrals at 40 digits, the Sackur-Tetrode
equation, the NIST value of the standard entropy of argon and the textbook values of the vibrational and
rotational temperatures of N2.
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
C = lookup("speed of light in vacuum")[1]
N_A = lookup("Avogadro constant")[1]
M_E = lookup("electron mass")[1]


def run(params: dict) -> dict:
    return eng_calc.handle({"operation": "statistical_mechanics", "params": params}, Ctx())


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def test_entropy_gibbs_boltzmann_and_bits() -> None:
    probs = [0.5, 0.25, 0.125, 0.125, 0.0]
    out = run({"mode": "entropy", "probabilities": probs})
    want = -sum(mpmath.mpf(p) * mpmath.log(p) for p in probs if p > 0)
    assert rel(out["S_over_k"], float(want)) < 1e-14
    assert rel(out["S"], float(want) * K) < 1e-14
    assert out["S_bits"] == pytest.approx(1.75, rel=1e-14, abs=0)
    assert out["n_states"] == 5
    assert out["S_max_over_k"] == pytest.approx(math.log(5), rel=1e-14, abs=0)
    assert out["uniform"] is False
    # uniform probabilities: Gibbs equals Boltzmann, S = k ln Omega with P_i = 1 / Omega
    out = run({"mode": "entropy", "probabilities": [0.25] * 4})
    assert out["uniform"] is True
    assert out["S_over_k"] == pytest.approx(math.log(4), rel=1e-14, abs=0)
    out = run({"mode": "entropy", "omega": 1e300})
    assert out["S_over_k"] == pytest.approx(300 * math.log(10), rel=1e-14, abs=0)
    out = run({"mode": "entropy", "ln_omega": 1e25})
    assert out["S"] == pytest.approx(1e25 * K, rel=1e-14, abs=0)
    assert run({"mode": "entropy", "omega": 1})["S"] == 0.0


def test_entropy_rejects_bad_probabilities() -> None:
    for bad in ([0.5, 0.4], [1.2, -0.2], [0.6, 0.6, -0.2], [], [0.5, "a"]):
        with pytest.raises(ToolError):
            run({"mode": "entropy", "probabilities": bad})
    with pytest.raises(ToolError):
        run({"mode": "entropy", "omega": 0.5})
    with pytest.raises(ToolError):
        run({"mode": "entropy", "omega": 10, "ln_omega": 1})
    with pytest.raises(ToolError):
        run({"mode": "entropy", "ln_omega": -0.5})


def test_entropy_heat_against_numeric_integral() -> None:
    # 1 kg of water heated from 20 C to 80 C, c = 4186 J/(kg K): integral of C dT / T
    out = run({"mode": "entropy_heat", "C": 4186.0, "T1": [20, "C"], "T2": [80, "C"]})
    want = _quad(lambda t: 4186 / t, [293.15, 353.15])
    assert rel(out["dS"], float(want)) < 1e-13
    assert out["Q"] == pytest.approx(4186 * 60, rel=1e-13, abs=0)
    # a reservoir at 80 C gives the heat: entropy generated > 0 (second law)
    out = run(
        {"mode": "entropy_heat", "C": 4186.0, "T1": 293.15, "T2": 353.15, "T_reservoir": 353.15}
    )
    assert out["dS_reservoir"] == pytest.approx(-4186 * 60 / 353.15, rel=1e-13, abs=0)
    assert out["S_generated"] == pytest.approx(out["dS"] - 4186 * 60 / 353.15, rel=1e-13, abs=0)
    assert out["S_generated"] > 0
    # isothermal heat (melting 1 kg of ice): dS = Q / T, Q may be negative
    out = run({"mode": "entropy_heat", "Q": 334e3, "T": 273.15})
    assert out["dS"] == pytest.approx(334e3 / 273.15, rel=1e-14, abs=0)
    out = run({"mode": "entropy_heat", "Q": -334e3, "T": 273.15})
    assert out["dS"] == pytest.approx(-334e3 / 273.15, rel=1e-14, abs=0)
    # nearly equal temperatures: no cancellation in ln(T2 / T1)
    t2 = 300.0 * (1 + 1e-12)
    out = run({"mode": "entropy_heat", "C": 1.0, "T1": 300.0, "T2": t2})
    with mpmath.workdps(40):
        want = mpmath.log(mpmath.mpf(t2) / 300)
    assert rel(out["dS"], float(want)) < 1e-12


def _quad(f: Any, pts: list[Any]) -> Any:
    """mpmath.quad with a plain return type (its stubs say tuple or mpf)."""
    return mpmath.quad(f, pts)


def _mj_mean_gamma(theta: float) -> Any:
    """<gamma> by direct integration of the Maxwell-Juttner density in gamma (independent of the Bessel form)."""
    with mpmath.workdps(40):
        th = mpmath.mpf(theta)

        def dens(g):
            return g * g * mpmath.sqrt(1 - 1 / (g * g)) * mpmath.exp(-(g - 1) / th)

        norm = _quad(dens, [1, 1 + th, 1 + 40 * th, mpmath.inf])
        return _quad(lambda g: g * dens(g), [1, 1 + th, 1 + 40 * th, mpmath.inf]) / norm


@pytest.mark.parametrize("temp", [1e5, 1e8, 5.93e9, 1e12])
def test_maxwell_juttner_mean_energy(temp: float) -> None:
    out = run({"mode": "maxwell_juttner", "T": temp, "m": M_E})
    theta = K * temp / (M_E * C * C)
    assert rel(out["theta"], theta) < 1e-14
    assert out["E_kin_classical"] == pytest.approx(1.5 * K * temp, rel=1e-14, abs=0)
    g = _mj_mean_gamma(theta)
    assert rel(out["gamma_mean"], float(g)) < 1e-12
    assert rel(out["E_kin_mean"], float((g - 1) * M_E * C * C)) < 1e-10
    if theta < 1e-4:  # non-relativistic limit: 3/2 k T
        assert out["E_kin_mean"] == pytest.approx(1.5 * K * temp, rel=1e-3, abs=0)
    if theta > 100:  # ultra-relativistic limit: 3 k T
        assert out["E_mean"] == pytest.approx(3 * K * temp, rel=2e-2, abs=0)


def test_maxwell_juttner_densities_are_normalised() -> None:
    temp, m = 3e9, M_E
    mc = m * C
    out0 = run({"mode": "maxwell_juttner", "T": temp, "m": m})
    theta = out0["theta"]

    def f_p(p):
        return run({"mode": "maxwell_juttner", "T": temp, "m": m, "p": float(p) * mc})[
            "pdf_p_magnitude"
        ]

    # integral of the momentum-magnitude density over p (in units of m c) = 1
    total = _quad(lambda x: f_p(x) * mc, [0, theta, 10 * theta, 60 * theta])
    assert abs(total - 1) < 1e-9
    one = run({"mode": "maxwell_juttner", "T": temp, "m": m, "p": 0.7 * mc})
    assert one["gamma_p"] == pytest.approx(math.sqrt(1 + 0.7**2), rel=1e-14, abs=0)
    assert one["pdf_p_magnitude"] == pytest.approx(
        4 * math.pi * (0.7 * mc) ** 2 * one["pdf_p"], rel=1e-14, abs=0
    )
    g = run({"mode": "maxwell_juttner", "T": temp, "m": m, "gamma": 1.5})["pdf_gamma"]
    with mpmath.workdps(40):
        th = mpmath.mpf(theta)
        want = (
            mpmath.mpf(1.5) ** 2
            * mpmath.sqrt(1 - 1 / mpmath.mpf(1.5) ** 2)
            * mpmath.exp(-mpmath.mpf(1.5) / th)
            / (th * mpmath.besselk(2, 1 / th))
        )
    assert rel(g, float(want)) < 1e-12


def test_maxwell_juttner_cold_gas_no_overflow() -> None:
    # theta = 1.7e-8: K2(1 / theta) underflows in plain floats
    out = run({"mode": "maxwell_juttner", "T": 1e2, "m": M_E, "p": 1e-26})
    assert out["E_kin_mean"] == pytest.approx(1.5 * K * 1e2, rel=1e-6, abs=0)
    # <gamma> - 1 = 2.5e-8: at 80 digits the Bessel ratio keeps all the digits the code must keep
    theta = K * 1e2 / (M_E * C * C)
    with mpmath.workdps(80):
        x = 1 / mpmath.mpf(theta)
        g = mpmath.besselk(1, x) / mpmath.besselk(2, x) + 3 / x
        want = (g - 1) * M_E * C * C
    assert rel(out["E_kin_mean"], float(want)) < 1e-12
    with pytest.raises(ToolError):
        run({"mode": "maxwell_juttner", "T": 1e2, "m": M_E, "p": -1e-25})
    assert out["pdf_p"] > 0
    with pytest.raises(ToolError):
        run({"mode": "maxwell_juttner", "T": 1e2, "m": M_E, "gamma": 0.9})


def test_partition_functions_argon_sackur_tetrode_and_nist() -> None:
    # 1 mol of argon at 298.15 K and 1 bar: S = 154.846 J/(mol K) (NIST webbook, standard entropy)
    temp, p0 = 298.15, 1e5
    volume = N_A * K * temp / p0
    out = run(
        {"mode": "partition_functions", "T": temp, "M_molar": 0.039948, "V": volume, "n_mol": 1}
    )
    m = 0.039948 / N_A
    lam = H / math.sqrt(2 * math.pi * m * K * temp)
    assert rel(out["Lambda"], lam) < 1e-14
    assert rel(out["Z_t"], volume / lam**3) < 1e-13
    n = N_A
    st = n * K * (math.log(volume / (n * lam**3)) + 2.5)
    assert rel(out["S"], st) < 1e-12
    assert out["S"] == pytest.approx(154.846, abs=0.01)
    assert out["U"] == pytest.approx(1.5 * n * K * temp, rel=1e-14, abs=0)
    assert out["C_V"] == pytest.approx(1.5 * n * K, rel=1e-14, abs=0)


def test_partition_functions_vibration_and_rotation_against_sums() -> None:
    # N2: theta_vib = 3374 K, theta_rot = 2.88 K (textbook); nu from the wavenumber 2358.6 1/cm
    nu = 2358.6e2 * C
    inertia = HBAR**2 / (2 * K * 2.875)
    temp = 1000.0
    out = run(
        {"mode": "partition_functions", "T": temp, "nu": nu, "I": inertia, "sigma": 2, "N": 1e20}
    )
    assert out["theta_vib"] == pytest.approx(3394, rel=3e-3, abs=0)
    assert out["theta_rot"] == pytest.approx(2.875, rel=1e-12, abs=0)
    x = H * nu / (K * temp)
    with mpmath.workdps(40):
        zv = mpmath.nsum(lambda n: mpmath.exp(-n * x), [0, mpmath.inf])

        # U_vib = N k T^2 d ln Z / dT by numerical differentiation of the sum
        def ln_zv(t):
            return mpmath.log(
                mpmath.nsum(lambda n: mpmath.exp(-n * H * nu / (K * t)), [0, mpmath.inf])
            )

        u_vib = 1e20 * K * temp**2 * mpmath.diff(ln_zv, temp)
    assert rel(out["Z_v"], float(zv)) < 1e-13
    assert rel(out["U_vib"], float(u_vib)) < 1e-10
    # high-temperature rotational sum: exact sum over J within 1/(3 theta) of the formula
    tr = 2.875
    with mpmath.workdps(30):
        zr_sum = (
            mpmath.nsum(
                lambda j: (2 * j + 1) * mpmath.exp(-j * (j + 1) * tr / temp), [0, mpmath.inf]
            )
            / 2
        )
    assert rel(out["Z_r"], float(zr_sum)) < 2e-3
    assert out["U_rot"] == pytest.approx(1e20 * K * temp, rel=1e-14, abs=0)
    assert out["U"] == pytest.approx(out["U_vib"] + out["U_rot"], rel=1e-14, abs=0)
    assert out["A"] == pytest.approx(out["U"] - temp * out["S"], rel=1e-12, abs=0)
    # C_V = dU/dT: numerical derivative of U_vib + U_rot in T
    with mpmath.workdps(40):

        def u_tot(t):
            y = H * nu / (K * t)
            return 1e20 * (H * nu / (mpmath.exp(y) - 1) + K * t)

        cv = mpmath.diff(u_tot, temp)
    assert rel(out["C_V"], float(cv)) < 1e-10
    # omega = 2 pi nu gives the same vibration
    same = run({"mode": "partition_functions", "T": temp, "omega": 2 * math.pi * nu, "N": 1e20})
    assert same["Z_v"] == pytest.approx(out["Z_v"], rel=1e-13, abs=0)
    assert out["Z"] == pytest.approx(out["Z_v"] * out["Z_r"], rel=1e-14, abs=0)
    # cold rotation: warning that the classical formula fails
    ctx = Ctx()
    params = {"mode": "partition_functions", "T": 5.0, "I": inertia, "sigma": 2}
    eng_calc.handle({"operation": "statistical_mechanics", "params": params}, ctx)
    assert any("theta_rot" in w for w in ctx.warnings)
    ctx = Ctx()
    params["T"] = 300.0
    eng_calc.handle({"operation": "statistical_mechanics", "params": params}, ctx)
    assert not ctx.warnings


def test_partition_functions_needs_some_motion() -> None:
    with pytest.raises(ToolError):
        run({"mode": "partition_functions", "T": 300.0})
    with pytest.raises(ToolError):
        run({"mode": "partition_functions", "T": 300.0, "I": 1e-46, "sigma": 1.5})
    with pytest.raises(ToolError):
        run({"mode": "partition_functions", "T": 300.0, "M_molar": 0.04})  # V missing
