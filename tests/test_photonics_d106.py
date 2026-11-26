"""D106: eng_calc.em_radiation (Poynting vector, plane waves, radiation pressure) and eng_calc.photometry
(luminosity distance, magnitudes, distance modulus, colour indices, bolometric correction).

References: the Poynting vector written out by components, its time average over one period of a plane wave
computed here with numpy, the IAU 2015 nominal solar values (L = 3.828e26 W, S = 1361 W/m2, 1 au), the Sun's
absolute visual magnitude 4.83 from m = -26.74, Sirius (m = -1.46 at 2.64 pc: M = 1.43), the definition
"5 magnitudes = a factor 100 in flux" and the round trips of every relation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc

C = 299792458.0  # m/s, exact
MU0 = 1.25663706127e-6  # N/A2, CODATA 2022
AU = 149597870700.0  # m, exact (IAU 2012)
PC = AU * 648000 / math.pi  # m, IAU 2015
LY = C * 365.25 * 86400  # m, Julian year


def em(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle({"operation": "em_radiation", "params": params}, ctx or Ctx())


def ph(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle({"operation": "photometry", "params": params}, ctx or Ctx())


def approx(v: Any, rel: float = 1e-12) -> Any:
    """Relative only: pytest's default abs = 1e-12 would accept any small SI value."""
    return pytest.approx(v, rel=rel, abs=0)


# ------------------------------------------------------------------ em_radiation: poynting


def cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def test_poynting_of_a_plane_wave_points_along_the_propagation() -> None:
    e0 = 1000.0
    out = em({"mode": "poynting", "E": [e0, 0, 0], "B": [0, e0 / C, 0]})
    assert out["N"][0] == 0 and out["N"][1] == 0
    assert out["N"][2] == approx(e0 * e0 / (MU0 * C), 1e-10)
    assert out["N_magnitude"] == approx(out["N"][2])


def test_poynting_by_components_and_with_h() -> None:
    e, b = [3.0, -1.5, 2.0], [0.4e-6, 1.1e-6, -0.7e-6]
    out = em({"mode": "poynting", "E": e, "B": b})
    want = [x / MU0 for x in cross(e, b)]
    assert out["N"] == [approx(x, 1e-10) for x in want]
    assert np.dot(out["N"], e) == pytest.approx(
        0, abs=1e-12 * out["N_magnitude"] * np.linalg.norm(e)
    )
    h = [x / MU0 for x in b]
    with_h = em({"mode": "poynting", "E": e, "H": h})
    assert with_h["N"] == [approx(x, 1e-10) for x in cross(e, h)]


def test_poynting_flux_through_a_tilted_flat_surface() -> None:
    e0 = 500.0
    tilt = math.radians(60)
    out = em(
        {
            "mode": "poynting",
            "E": [e0, 0, 0],
            "B": [0, e0 / C, 0],
            "normal": [0, 2 * math.sin(tilt), 2 * math.cos(tilt)],  # not a unit vector: normalised
            "A": [3, "cm2"],
        }
    )
    assert out["flux"] == approx(e0 * e0 / (MU0 * C) * 3e-4 * 0.5, 1e-10)


# ------------------------------------------------------------------ em_radiation: plane_wave


def test_plane_wave_from_sunlight() -> None:
    out = em({"mode": "plane_wave", "I": 1361})
    e0 = math.sqrt(2 * 1361 * MU0 * C)
    assert out["E0"] == approx(e0, 1e-12)
    assert out["E0"] == approx(1012.6, 1e-4)  # about 1.01 kV/m, textbook value
    assert out["E_rms"] == approx(e0 / math.sqrt(2), 1e-12)
    assert out["B0"] == approx(e0 / C, 1e-12)
    assert out["B_rms"] == approx(e0 / C / math.sqrt(2), 1e-12)
    assert out["u"] == approx(1361 / C, 1e-12)
    assert out["g"] == approx(1361 / C**2, 1e-12)
    assert out["p_absorbed"] == approx(1361 / C, 1e-12)
    assert out["p_reflected"] == approx(2 * 1361 / C, 1e-12)
    assert em({"mode": "plane_wave", "I": [136.1, "mW/cm2"]})["E0"] == approx(e0, 1e-12)


def test_plane_wave_intensity_is_the_time_average_of_the_poynting_vector() -> None:
    """numpy: E(t) = E0 cos(wt), B(t) = B0 cos(wt) on 100000 points of one period."""
    out = em({"mode": "plane_wave", "E0": [2, "kV/m"]})
    assert out["E0"] == 2000.0
    t = np.linspace(0, 2 * math.pi, 100_000, endpoint=False)
    e = out["E0"] * np.cos(t)
    b = out["B0"] * np.cos(t)
    assert out["I"] == approx(float(np.mean(e * b / MU0)), 1e-12)
    assert out["E_rms"] == approx(float(np.sqrt(np.mean(e * e))), 1e-12)


@pytest.mark.parametrize("key", ["E0", "B0", "E_rms"])
def test_plane_wave_round_trip(key: str) -> None:
    base = em({"mode": "plane_wave", "I": 250.0})
    back = em({"mode": "plane_wave", key: base[key]})
    assert back["I"] == approx(250.0, 1e-12)


# ------------------------------------------------------------------ em_radiation: radiation_pressure


def test_radiation_pressure_of_sunlight() -> None:
    absorbed = em({"mode": "radiation_pressure", "I": 1361})
    mirror = em({"mode": "radiation_pressure", "I": 1361, "R": 1})
    assert absorbed["pressure"] == approx(1361 / C, 1e-12)
    assert absorbed["pressure"] == approx(4.54e-6, 1e-3)  # 4.5 uPa, textbook
    assert mirror["pressure"] == approx(2 * 1361 / C, 1e-12)


def test_radiation_pressure_is_momentum_over_area_and_time() -> None:
    """Page: p_EM = U / c and P_EM = I / c = p_EM / (A t), for an absorbing surface at normal incidence."""
    out = em({"mode": "radiation_pressure", "U": 50.0, "A": 0.2, "t": 10.0})
    assert out["I"] == approx(50.0 / (0.2 * 10.0))
    assert out["momentum"] == approx(50.0 / C)
    assert out["pressure"] == approx(out["momentum"] / (0.2 * 10.0))
    assert out["force"] == approx(out["pressure"] * 0.2)


def test_radiation_pressure_at_an_angle_and_from_a_beam_power() -> None:
    out = em({"mode": "radiation_pressure", "P": 2.0, "A": [1, "cm2"], "R": 0.5, "theta_deg": 60})
    i = 2.0 / 1e-4
    assert out["I"] == approx(i)
    assert out["pressure"] == approx(1.5 * i * 0.25 / C, 1e-12)  # (1 + R) I cos^2(theta) / c


def test_radiation_pressure_from_the_energy_on_an_inclined_surface() -> None:
    """U falls on A in t at 60 degrees: the beam intensity is U / (A t cos(theta)), the energy stays U."""
    out = em({"mode": "radiation_pressure", "U": 30.0, "A": 0.5, "t": 4.0, "theta_deg": 60})
    assert out["I"] == approx(30.0 / (0.5 * 4.0 * 0.5), 1e-12)
    assert out["energy"] == approx(30.0, 1e-12)
    beam = em({"mode": "radiation_pressure", "I": 15.0, "A": 0.5, "t": 4.0, "theta_deg": 60})
    assert beam["energy"] == approx(15.0 * 0.5 * 0.5 * 4.0, 1e-12)  # I A cos(theta) t


# ------------------------------------------------------------------ photometry


def test_luminosity_distance_of_the_earth_is_one_au() -> None:
    out = ph({"mode": "luminosity_distance", "L": 3.828e26, "F": 1361})
    assert out["D_L"] == approx(AU, 2e-4)
    assert out["D_L_pc"] == approx(out["D_L"] / PC)
    assert out["D_L_ly"] == approx(out["D_L"] / LY)
    back = ph({"mode": "luminosity_distance", "L": 3.828e26, "D_L": [out["D_L_pc"], "pc"]})
    assert back["F"] == approx(1361, 1e-12)
    lum = ph({"mode": "luminosity_distance", "F": 1361, "D_L": [1, "au"]})
    assert lum["L"] == approx(4 * math.pi * AU * AU * 1361, 1e-12)


def test_apparent_magnitude_five_magnitudes_are_a_factor_100() -> None:
    out = ph({"mode": "apparent_magnitude", "F": 1e-10, "F0": 1e-8})
    assert out["m"] == approx(5.0, 1e-14)
    back = ph({"mode": "apparent_magnitude", "m": -1.46, "F0": 3.6e-8})
    assert back["F"] == approx(3.6e-8 * 10 ** (1.46 / 2.5), 1e-12)
    ref = ph({"mode": "apparent_magnitude", "m": 2.0, "F": 1e-9})
    assert ref["F0"] == approx(1e-9 * 10 ** (2.0 / 2.5), 1e-12)


def test_distance_modulus_of_the_sun_and_sirius() -> None:
    sun = ph({"mode": "distance_modulus", "m": -26.74, "D_L": [1, "au"]})
    assert sun["M"] == pytest.approx(4.83, abs=0.006)
    assert sun["mu"] == approx(-26.74 - sun["M"])
    sirius = ph({"mode": "distance_modulus", "m": -1.46, "D_L": [2.64, "pc"]})
    assert sirius["M"] == pytest.approx(1.43, abs=0.006)
    ten = ph({"mode": "distance_modulus", "m": 7.0, "M": 7.0})
    assert ten["D_L_pc"] == approx(10.0, 1e-12)  # mu = 0 at 10 pc
    far = ph({"mode": "distance_modulus", "M": -19.3, "D_L": [100, "Mpc"]})
    assert far["m"] == approx(-19.3 + 5 * (math.log10(1e8) - 1), 1e-12)  # mu = 35


def test_colour_indices_and_bolometric_correction() -> None:
    out = ph({"mode": "color_index", "U": 0.85, "B": 1.30, "V": 0.65})
    assert out["U_B"] == approx(0.85 - 1.30)
    assert out["B_V"] == approx(1.30 - 0.65)
    only_bv = ph({"mode": "color_index", "B": 1.30, "V": 0.65})
    assert "U_B" not in only_bv
    bc = ph({"mode": "bolometric_correction", "m_bol": 4.74, "V": 4.83})
    assert bc["BC"] == approx(4.74 - 4.83)
    m_bol = ph({"mode": "bolometric_correction", "BC": -0.09, "V": 4.83})
    assert m_bol["m_bol"] == approx(4.74)


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize(
    ("op", "params", "match"),
    [
        ("em_radiation", {"mode": "poynting", "E": [1, 0, 0]}, "exactly one of"),
        ("em_radiation", {"mode": "poynting", "E": [1, 0], "B": [0, 1, 0]}, "3 numbers"),
        ("em_radiation", {"mode": "poynting", "E": [1, 0, 0], "B": [0, 1, 0], "A": 1}, "normal"),
        (
            "em_radiation",
            {"mode": "poynting", "E": [1, 0, 0], "B": [0, 1, 0], "normal": [0, 0, 0], "A": 1},
            "zero",
        ),
        ("em_radiation", {"mode": "plane_wave", "I": 1, "E0": 1}, "exactly one of"),
        ("em_radiation", {"mode": "radiation_pressure", "I": 1, "R": 1.5}, "R"),
        ("em_radiation", {"mode": "radiation_pressure", "I": 1, "theta_deg": 90}, "theta_deg"),
        ("em_radiation", {"mode": "radiation_pressure", "U": 1, "A": 1}, "t"),
        ("em_radiation", {"mode": "radiation_pressure", "P": 1}, "A"),
        ("photometry", {"mode": "luminosity_distance", "L": 1}, "two of"),
        ("photometry", {"mode": "luminosity_distance", "L": 1, "F": 1, "D_L": 1}, "two of"),
        ("photometry", {"mode": "apparent_magnitude", "F": -1, "F0": 1}, "> 0"),
        ("photometry", {"mode": "color_index", "V": 0.6}, "two of"),
        ("photometry", {"mode": "bolometric_correction", "BC": 0.1}, "two of"),
        ("photometry", {"mode": "distance_modulus", "m": 1, "M": 2, "D_L": 3}, "two of"),
    ],
)
def test_errors(op: str, params: dict, match: str) -> None:
    with pytest.raises(ToolError, match=match):
        eng_calc.handle({"operation": op, "params": params}, Ctx())
