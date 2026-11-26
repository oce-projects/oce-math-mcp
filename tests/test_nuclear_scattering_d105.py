"""D105: eng_calc.nuclear_scattering (Breit-Wigner resonance, Rutherford and Mott scattering).

References: the Lorentzian line (half height at E0 +- Gamma/2, area 2 pi / Gamma), the neutron de Broglie
wavelength from h / sqrt(2 m E), hbar = 6.582119569e-16 eV s, e^2 / (4 pi eps0) from e and eps0, the
classical relation dsigma/dOmega = b / sin(theta) |db/dtheta| by finite differences, energy conservation at the
turning point, and the symmetrized Coulomb amplitudes of two identical particles computed here with mpmath
(spin weights (s + 1) / (2s + 1) and s / (2s + 1)).
"""

from __future__ import annotations

import math
from typing import Any

import mpmath
import pytest
from scipy.integrate import quad

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc

EV = 1.602176634e-19  # J, exact
FM = 1e-15
BARN = 1e-28
H = 6.62607015e-34  # J s, exact in the 2019 SI
HBAR = H / (2 * math.pi)
M_N = 1.67492750056e-27  # kg, CODATA 2022 neutron mass
M_P = 1.67262192595e-27  # kg, CODATA 2022 proton mass
E2 = EV * EV / (4 * math.pi * 8.8541878188e-12)  # e^2 / (4 pi eps0), CODATA 2022: 1.43996455 MeV fm


def run(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle({"operation": "nuclear_scattering", "params": params}, ctx or Ctx())


def approx(v: float, rel: float = 1e-12) -> Any:
    """Relative only: the default abs=1e-12 of pytest.approx would accept any cross section (1e-28 m2)."""
    return pytest.approx(v, rel=rel, abs=0)


# ------------------------------------------------------------------ breit_wigner

BW = {
    "mode": "breit_wigner",
    "J": 1,
    "s_a": 0.5,
    "s_b": 0.5,
    "E0": [1.0, "MeV"],
    "Gamma_ab": [20, "keV"],
    "Gamma_c": [30, "keV"],
    "k": [0.2, "1/fm"],
}


def test_breit_wigner_peak_and_statistical_factor() -> None:
    out = run(BW)
    g = 3 / 4  # (2J + 1) / ((2 s_a + 1) (2 s_b + 1)) = 3 / 4
    k = 0.2e15
    gab, gc = 20e3 * EV, 30e3 * EV
    gam = gab + gc
    peak = 4 * math.pi * g * gab * gc / (k * k * gam * gam)
    assert out["g"] == approx(g)
    assert out["Gamma"] == approx(gam)
    assert out["Gamma_MeV"] == approx(0.05)
    assert out["sigma"] == approx(peak)  # E missing: E = E0
    assert out["sigma_peak"] == approx(peak)
    assert out["sigma_barn"] == approx(peak / BARN)
    assert out["k"] == approx(k)


def test_breit_wigner_half_height_at_e0_plus_minus_half_width() -> None:
    peak = run(BW)["sigma_peak"]
    for e_mev in (1.0 - 0.025, 1.0 + 0.025):
        out = run({**BW, "E": [e_mev, "MeV"]})
        assert out["sigma"] == approx(peak / 2, 1e-9)


def test_breit_wigner_area_is_2_pi_over_gamma_times_the_peak_factor() -> None:
    """Lorentzian: integral of sigma dE = 2 pi^2 g Gamma_ab Gamma_c / (k^2 Gamma) at fixed k."""
    gam = 50e3 * EV

    e0 = 1e6 * EV

    def sigma(x: float) -> float:
        """sigma at E = E0 + x Gamma, in units of the peak (quad works on numbers near 1)."""
        return run({**BW, "E": e0 + x * gam})["sigma"] / run(BW)["sigma_peak"]

    area = (
        run(BW)["sigma_peak"]
        * gam
        * sum(
            quad(sigma, a, b, epsabs=0, epsrel=1e-13, limit=200)[0]
            for a, b in ((-15, -1), (-1, 0), (0, 1), (1, 15))
        )
    )
    want = 2 * math.pi**2 * 0.75 * (20e3 * EV) * (30e3 * EV) / (0.2e15**2 * gam)
    # |E - E0| < 15 Gamma holds the fraction (2 / pi) atan(30) of the whole line
    assert area == approx(want * 2 / math.pi * math.atan(30), 1e-9)


def test_breit_wigner_k_from_the_mass_is_the_de_broglie_wavenumber() -> None:
    out = run(
        {
            "mode": "breit_wigner",
            "J": 0.5,
            "s_a": 0.5,
            "s_b": 0,
            "E0": [1, "eV"],
            "Gamma_ab": [1e-3, "eV"],
            "Gamma_c": [2e-3, "eV"],
            "mu": M_N,
        }
    )
    lam = H / math.sqrt(2 * M_N * EV)  # 2.86e-11 m for a 1 eV neutron
    assert out["k"] == approx(2 * math.pi / lam, 1e-10)
    assert out["g"] == approx(1.0)


def test_breit_wigner_lifetime_is_hbar_over_gamma() -> None:
    out = run({**BW, "Gamma_ab": [0.4, "eV"], "Gamma_c": [0.6, "eV"]})
    assert out["tau"] == approx(6.582119569e-16, 1e-9)  # hbar / (1 eV)


def test_breit_wigner_with_the_mass_uses_k_at_each_energy() -> None:
    base = {**BW, "mu": M_N}
    del base["k"]
    lo = run({**base, "E": [0.5, "MeV"]})
    hi = run({**base, "E": [2.0, "MeV"]})
    assert hi["k"] / lo["k"] == approx(2.0, 1e-12)  # k proportional to sqrt(E)
    assert lo["sigma_peak"] == approx(hi["sigma_peak"], 1e-12)  # the peak uses k(E0)


# ------------------------------------------------------------------ rutherford

GOLD = {"mode": "rutherford", "E": [7.7, "MeV"], "Z1": 2, "Z2": 79}


def test_rutherford_alpha_on_gold_at_90_degrees() -> None:
    out = run({**GOLD, "theta_deg": 90})
    alpha = 2 * 79 * E2
    e = 7.7e6 * EV
    want = (alpha / (4 * e)) ** 2 * 4  # csc^4(45 deg) = 4
    assert out["alpha"] == approx(alpha, 1e-8)
    assert out["dsigma_dOmega"] == approx(want, 1e-8)
    assert out["dsigma_dOmega_barn"] == approx(want / BARN, 1e-8)
    assert out["dsigma_dOmega_barn"] == approx(2.1826, 1e-4)  # 2.18 b/sr, Geiger-Marsden


def test_rutherford_is_b_over_sin_times_db_dtheta() -> None:
    """Classical orbit: the ring of impact parameters b to b + db goes to theta to theta + dtheta."""
    h = 1e-5
    for deg in (20.0, 60.0, 150.0):
        out = run({**GOLD, "theta_deg": deg})
        bp = run({**GOLD, "theta_deg": deg + h})["b"]
        bm = run({**GOLD, "theta_deg": deg - h})["b"]
        db = (bp - bm) / (2 * math.radians(h))
        th = math.radians(deg)
        assert out["dsigma_dOmega"] == approx(out["b"] / math.sin(th) * abs(db), 1e-8)


@pytest.mark.parametrize("z1", [2, -1])
def test_rutherford_turning_point_conserves_energy(z1: int) -> None:
    """E = alpha / r_min + E b^2 / r_min^2 (energy and angular momentum at the turning point)."""
    out = run({**GOLD, "Z1": z1, "theta_deg": 40})
    e = 7.7e6 * EV
    r, b = out["r_min"], out["b"]
    assert out["alpha"] / r + e * b * b / (r * r) == approx(e, 1e-12)
    assert r > 0


def test_rutherford_sign_of_the_coupling_leaves_the_cross_section() -> None:
    rep = run({**GOLD, "theta_deg": 30})
    att = run({**GOLD, "Z1": -2, "theta_deg": 30})
    assert att["dsigma_dOmega"] == approx(rep["dsigma_dOmega"])
    assert att["r_min"] < rep["r_min"]


def test_rutherford_counts_from_the_fluence_and_alpha_in_mev_fm() -> None:
    out = run(
        {
            "mode": "rutherford",
            "E": [5, "MeV"],
            "alpha": [100, "MeV*fm"],
            "theta_deg": 60,
            "fluence": [1e10, "1/cm2"],
        }
    )
    want = (100 / 20) ** 2 * FM * FM / math.sin(math.radians(30)) ** 4
    assert out["dsigma_dOmega"] == approx(want, 1e-12)
    assert out["dN_dOmega"] == approx(1e14 * want, 1e-12)


# ------------------------------------------------------------------ mott


def coulomb_pair(e: float, deg: float, alpha: float, s: float, m: float) -> float:
    """Two identical particles: |f(t) +- f(pi - t)|^2 with spin weights, f the exact Coulomb amplitude."""
    mpmath.mp.dps = 30
    mu = m / 2
    v = math.sqrt(2 * e / mu)
    k = mu * v / HBAR
    eta = alpha / (HBAR * v)
    sigma0 = mpmath.arg(mpmath.gamma(1 + 1j * eta))

    def f(t: float) -> Any:
        s2 = mpmath.sin(t / 2) ** 2
        return -eta / (2 * k * s2) * mpmath.exp(-1j * eta * mpmath.log(s2) + 2j * sigma0)

    t = mpmath.radians(deg)
    plus = abs(f(t) + f(mpmath.pi - t)) ** 2
    minus = abs(f(t) - f(mpmath.pi - t)) ** 2
    n = 2 * s + 1
    if round(2 * s) % 2 == 0:  # bosons: symmetric spin states with symmetric space
        return float(((s + 1) * plus + s * minus) / n)
    return float((s * plus + (s + 1) * minus) / n)  # fermions


@pytest.mark.parametrize(
    ("s", "m", "z"),
    [(0, 6.6446573357e-27, 2), (0.5, M_P, 1), (1, 3.3435837724e-27, 1)],
)
@pytest.mark.parametrize("deg", [30.0, 90.0, 137.0])
def test_mott_is_the_symmetrized_coulomb_amplitude(s: float, m: float, z: int, deg: float) -> None:
    e = 2e6 * EV
    out = run({"mode": "mott", "E": e, "theta_deg": deg, "Z": z, "s": s, "m": m})
    want = coulomb_pair(e, deg, z * z * E2, s, m)
    assert out["dsigma_dOmega"] == approx(want, 1e-7)
    assert out["A"] == approx(2 * (-1) ** round(2 * s) / (2 * s + 1))


def test_mott_spin_zero_at_90_degrees_is_four_times_rutherford() -> None:
    e = 1e6 * EV
    out = run({"mode": "mott", "E": e, "theta_deg": 90, "Z": 2, "s": 0, "m": 6.6446573357e-27})
    unit = (4 * E2 / (4 * e)) ** 2
    assert out["dsigma_dOmega"] == approx(16 * unit, 1e-8)  # 4 + 4 + 2 * 4
    assert out["dsigma_no_interference"] == approx(8 * unit, 1e-8)
    assert out["dsigma_rutherford"] == approx(4 * unit, 1e-8)


def test_mott_speed_instead_of_mass() -> None:
    e = 2e6 * EV
    v = math.sqrt(4 * e / M_P)
    a = run({"mode": "mott", "E": e, "theta_deg": 50, "Z": 1, "s": 0.5, "m": M_P})
    b = run({"mode": "mott", "E": e, "theta_deg": 50, "Z": 1, "s": 0.5, "v": v})
    assert a["v"] == approx(v)
    assert b["dsigma_dOmega"] == approx(a["dsigma_dOmega"], 1e-12)
    assert a["eta"] == approx(E2 / (HBAR * v), 1e-8)
    assert a["k"] == approx(M_P / 2 * v / HBAR, 1e-10)  # mu = m / 2
    assert "k" not in b


def test_mott_warns_when_the_speed_is_relativistic() -> None:
    ctx = Ctx()
    run(
        {"mode": "mott", "E": [5, "MeV"], "theta_deg": 50, "Z": 1, "s": 0.5, "m": M_P}, ctx
    )  # v = 0.146 c
    assert any("relativistic" in w for w in ctx.warnings)
    quiet = Ctx()
    run(
        {"mode": "mott", "E": [1, "MeV"], "theta_deg": 50, "Z": 1, "s": 0.5, "m": M_P}, quiet
    )  # v = 0.065 c
    assert not quiet.warnings


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"mode": "fission"}, "mode"),
        ({**BW, "J": 0.3}, "multiple of 1/2"),
        ({**BW, "s_a": -0.5}, "multiple of 1/2"),
        ({**BW, "mu": M_N}, "exactly one of"),
        ({**BW, "Z1": 2}, "Z1"),
        ({**GOLD, "theta_deg": 0}, "theta_deg"),
        ({**GOLD, "theta_deg": 180}, "theta_deg"),
        ({**GOLD, "theta_deg": 30, "alpha": 1e-26}, "exactly one of"),
        ({**GOLD, "theta_deg": 30, "Z2": 0}, "Z2"),
        ({**GOLD, "theta_deg": 30, "Z2": 7.5}, "Z2"),
        ({"mode": "mott", "E": 1e-13, "theta_deg": 30, "Z": 1, "s": 0.5}, "exactly one of"),
        (
            {"mode": "mott", "E": 1e-13, "theta_deg": 30, "Z": 1, "s": 0.7, "m": M_P},
            "multiple of 1/2",
        ),
        ({"mode": "mott", "E": 1e-13, "theta_deg": 30, "Z": 1, "m": M_P}, "params.s"),
    ],
)
def test_errors(params: dict, match: str) -> None:
    with pytest.raises(ToolError, match=match):
        run(params)
