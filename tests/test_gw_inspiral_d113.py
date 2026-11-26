"""D113: eng_calc.gravitation mode `gw_inspiral` (circular orbit, quadrupole formula).

References independent of the module, all from the local offline library:
- "Two-body problem in general relativity", section "Gravitational radiation": energy, angular momentum and period
  loss averaged over an orbit, here with e = 0 and a = r.
- "Gravitational wave", section "Binaries": dr/dt = -64/5 G^3/c^5 m1 m2 (m1 + m2) / r^3, the time to merger
  5/256 c^5/G^3 r^4 / (m1 m2 (m1 + m2)), the radiation at twice the orbital frequency, and the page values of a
  pair of solar-mass neutron stars and of the Earth-Sun system (rounded on the page: compared at 1% or at one
  significant digit).
- "Chirp mass": the definition (m1 m2)^(3/5) / (m1 + m2)^(1/5), (1/4)^(3/5) M = 0.435 M for equal masses and
  df/dt = 96/5 pi^(8/3) (G Mc / c^3)^(5/3) f^(11/3).
- "Solar mass" (nominal 1.988416e30 kg) and "Earth mass" (5.9722e24 kg).
The formulas are evaluated in mpmath at 40 digits; the time to coalescence is also the integral of dr / |dr/dt|,
and the chirp-mass frequency law is checked against the chain rule on the Kepler frequency.
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

G = lookup("Newtonian constant of gravitation")[1]
C = lookup("c")[1]
M_SUN = 1.988416e30  # kg, nominal solar mass ("Solar mass" page)
M_EARTH = 5.9722e24  # kg ("Earth mass" page)
AU = 149597870700.0  # m, exact


def run(params: dict[str, Any]) -> tuple[dict[str, Any], Ctx]:
    ctx = Ctx()
    out = eng_calc.handle(
        {"operation": "gravitation", "params": {"mode": "gw_inspiral", **params}}, ctx
    )
    return out, ctx


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def page(m1: float, m2: float, r: float) -> dict[str, float]:
    """Page formulas at 40 digits."""
    with mpmath.workdps(40):
        g, c = mpmath.mpf(G), mpmath.mpf(C)
        a, b, s = mpmath.mpf(m1), mpmath.mpf(m2), mpmath.mpf(r)
        m = a + b
        w = mpmath.sqrt(g * m / s**3)
        p_orb = 2 * mpmath.pi / w
        f_gw = 2 / p_orb
        chirp = (a * b) ** (mpmath.mpf(3) / 5) / m ** (mpmath.mpf(1) / 5)
        ref = {
            "M_total": m,
            "reduced_mass": a * b / m,
            "chirp_mass": chirp,
            "power_gw": 32 * g**4 * a**2 * b**2 * m / (5 * c**5 * s**5),
            "dL_dt": -32 * g**3.5 * a**2 * b**2 * mpmath.sqrt(m) / (5 * c**5 * s**3.5),
            "dP_orb_dt": -192
            * mpmath.pi
            * g ** (mpmath.mpf(5) / 3)
            * a
            * b
            / (5 * c**5 * m ** (mpmath.mpf(1) / 3))
            * (p_orb / (2 * mpmath.pi)) ** (-mpmath.mpf(5) / 3),
            "dr_dt": -mpmath.mpf(64) / 5 * g**3 / c**5 * a * b * m / s**3,
            "t_coalescence": mpmath.mpf(5) / 256 * c**5 / g**3 * s**4 / (a * b * m),
            "omega_orb": w,
            "P_orb": p_orb,
            "f_orb": 1 / p_orb,
            "f_gw": f_gw,
            "df_gw_dt": mpmath.mpf(96)
            / 5
            * mpmath.pi ** (mpmath.mpf(8) / 3)
            * (g * chirp / c**3) ** (mpmath.mpf(5) / 3)
            * f_gw ** (mpmath.mpf(11) / 3),
            "E_orb": -g * a * b / (2 * s),
            "v_over_c": mpmath.sqrt(g * m / s) / c,
        }
        return {k: float(v) for k, v in ref.items()}


CASES = [
    (M_SUN, M_SUN, 1.89e8),  # the page's neutron-star pair
    (1.4 * M_SUN, 10 * M_SUN, 3e6),
    (36 * M_SUN, 29 * M_SUN, 5e7),
    (M_SUN, M_EARTH, AU),  # Earth-Sun
    (1e9 * M_SUN, 3e8 * M_SUN, 1e15),  # supermassive pair
]


@pytest.mark.parametrize(("m1", "m2", "r"), CASES)
def test_against_page_formulas(m1: float, m2: float, r: float) -> None:
    out, _ = run({"m1": m1, "m2": m2, "r": r})
    ref = page(m1, m2, r)
    assert set(out) == set(ref)
    for key, val in ref.items():
        assert rel(out[key], val) < 1e-13, key


@pytest.mark.parametrize(("m1", "m2", "r"), CASES[:3])
def test_time_is_integral_and_energy_balance(m1: float, m2: float, r: float) -> None:
    out, _ = run({"m1": m1, "m2": m2, "r": r})
    k = 64 / 5 * G**3 * m1 * m2 * (m1 + m2) / C**5
    with mpmath.workdps(40):
        t: Any = mpmath.quad(lambda x: x**3 / mpmath.mpf(k), [0, r])
    assert rel(out["t_coalescence"], float(t)) < 1e-12
    dr = out["dr_dt"]
    # E = -G m1 m2 / (2 r): the radiated power is -dE/dt
    assert rel(out["power_gw"], -G * m1 * m2 / (2 * r * r) * dr) < 1e-12
    # L = mu sqrt(G M r) on a circular orbit, and dE = omega dL
    mu = m1 * m2 / (m1 + m2)
    assert rel(out["dL_dt"], mu * math.sqrt(G * (m1 + m2) / r) / 2 * dr) < 1e-12
    assert rel(out["dL_dt"], -out["power_gw"] / out["omega_orb"]) < 1e-12
    # Kepler: f ~ r^(-3/2), P ~ r^(3/2): the chirp-mass law and the period law follow from dr/dt
    assert rel(out["df_gw_dt"], -1.5 * out["f_gw"] / r * dr) < 1e-12
    assert rel(out["dP_orb_dt"], 1.5 * out["P_orb"] / r * dr) < 1e-12


def test_page_values_neutron_stars() -> None:
    out, _ = run({"m1": M_SUN, "m2": M_SUN, "r": 1.89e8})
    assert rel(out["P_orb"], 1000) < 0.005  # page: "an orbital period of 1,000 seconds"
    assert rel(out["t_coalescence"], 1.30e13) < 0.01  # page: "1.30×10^13 seconds"
    close, _ = run({"m1": M_SUN, "m2": M_SUN, "r": 1.89e6})
    assert rel(close["t_coalescence"], 130_000) < 0.01  # page: "about 130,000 seconds"
    assert rel(close["f_orb"], 1) < 0.005  # page: "1 orbit per second at the start"
    merger, _ = run({"m1": M_SUN, "m2": M_SUN, "r": 2e4})
    assert rel(merger["f_orb"], 918) < 0.005  # page: "918 orbits per second" at 20 km
    assert rel(merger["f_gw"], 2 * merger["f_orb"]) < 1e-14  # "twice the orbital frequency"


def test_page_values_earth_sun() -> None:
    out, _ = run({"m1": M_SUN, "m2": M_EARTH, "r": AU})
    assert round(out["power_gw"], -2) == 200  # page: "only 200 watts"
    assert f"{out['dr_dt'] * 86400:.0e}" == "-1e-15"  # page: "about 1×10^-15 meters per day"


def test_chirp_mass() -> None:
    m = 2.8 * M_SUN
    out, _ = run({"m1": m / 2, "m2": m / 2, "r": 1e6})
    assert rel(out["chirp_mass"], 0.25**0.6 * m) < 1e-14
    assert round(out["chirp_mass"] / m, 3) == 0.435  # page: (1/4)^(3/5) M = 0.435 M
    a, b = 7 * M_SUN, 2 * M_SUN
    one, _ = run({"m1": a, "m2": b, "r": 1e6})
    two, _ = run({"m1": b, "m2": a, "r": 1e6})
    assert one == two
    mu = a * b / (a + b)
    assert rel(one["chirp_mass"], mu**0.6 * (a + b) ** 0.4) < 1e-14  # page: mu^(3/5) M^(2/5)


def test_relativistic_warnings() -> None:
    # 2 solar masses: G M / (r c^2) = 1% at r = 295,326 m; 6 G M / c^2 = 17,719.6 m
    _, ctx = run({"m1": M_SUN, "m2": M_SUN, "r": 3e5})
    assert ctx.warnings == []
    _, ctx = run({"m1": M_SUN, "m2": M_SUN, "r": 2.9e5})
    assert len(ctx.warnings) == 1 and "post-Newtonian" in ctx.warnings[0]
    _, ctx = run({"m1": M_SUN, "m2": M_SUN, "r": 2e4})
    assert len(ctx.warnings) == 1 and "v/c = 0.384" in ctx.warnings[0]
    _, ctx = run({"m1": M_SUN, "m2": M_SUN, "r": 1.7e4})
    assert len(ctx.warnings) == 2
    assert "innermost stable circular orbit" in ctx.warnings[1]
    assert "17,719.6 m" in ctx.warnings[1]


def test_units_formula_and_text() -> None:
    out, ctx = run({"m1": [1.4 * M_SUN, "kg"], "m2": 1.4 * M_SUN, "r": [1000, "km"]})
    ref = page(1.4 * M_SUN, 1.4 * M_SUN, 1e6)
    assert rel(out["t_coalescence"], ref["t_coalescence"]) < 1e-13
    assert ctx.units is not None
    expected = {
        "M_total": "kg",
        "reduced_mass": "kg",
        "chirp_mass": "kg",
        "power_gw": "W",
        "dL_dt": "N*m",
        "dP_orb_dt": "1",
        "dr_dt": "m/s",
        "t_coalescence": "s",
        "omega_orb": "rad/s",
        "P_orb": "s",
        "f_orb": "Hz",
        "f_gw": "Hz",
        "df_gw_dt": "Hz/s",
        "E_orb": "J",
        "v_over_c": "1",
    }
    assert ctx.units == expected
    assert ctx.formula is not None
    assert "P = 32 G^4 m1^2 m2^2 (m1 + m2) / (5 c^5 r^5)" in ctx.formula
    assert "f_gw = 2 f_orb" in ctx.formula
    assert ctx.formatted is not None and "coalescence" in ctx.formatted


def test_errors() -> None:
    with pytest.raises(ToolError, match=r"Missing params\.r "):
        run({"m1": M_SUN, "m2": M_SUN})
    with pytest.raises(ToolError, match="m2 must be > 0"):
        run({"m1": M_SUN, "m2": 0, "r": 1e6})
    with pytest.raises(ToolError, match="Keys not used"):
        run({"m1": M_SUN, "m2": M_SUN, "r": 1e6, "M": M_SUN})
    with pytest.raises(ToolError, match="The result is not finite"):
        run({"m1": 1e300, "m2": 1e300, "r": 1e-300})
