"""Six eng_calc answers that the audit of 25/11/2026 found wrong or silent (D92).

second_order with zeta < 1e-4 missed the oscillations (zeta = 1e-5: tr_10_90 7.30 s instead of 1.0196 s);
the beta thermistor gave -208 K for R = 1e-10 ohm; bolt_joint took a negative shear as a negative utilization
(ok = true); gear_spur accepted a pressure angle of 90 degrees; belt_drive ignored m_l without P; psychrometrics
condensed water while heating when RH2 was given.
"""

from __future__ import annotations

import math

import mpmath as mp

from tests.conftest import err, ok


def e(op: str, **params: object) -> dict:
    return ok("eng_calc", operation=op, params=params)


def bad(op: str, **params: object) -> str:
    return err("eng_calc", operation=op, params=params)


def _y(z: mp.mpf, wn: mp.mpf, t: mp.mpf) -> mp.mpf:
    wd = wn * mp.sqrt(1 - z * z)
    return 1 - mp.e ** (-z * wn * t) * (mp.cos(wd * t) + z / mp.sqrt(1 - z * z) * mp.sin(wd * t))


def test_second_order_underdamped_down_to_tiny_damping() -> None:
    mp.mp.dps = 30
    for zf, wnf in (
        (1e-6, 1.0),
        (1e-5, 1.0),
        (1e-4, 2.0),
        (1e-3, 1.0),
        (0.2, 5.0),
        (0.7, 3.0),
        (0.999, 1.0),
    ):
        r = e("control_system", mode="second_order", zeta=zf, wn=wnf)
        z, wn = mp.mpf(zf), mp.mpf(wnf)
        half = mp.pi / (wn * mp.sqrt(1 - z * z))
        # rise: the first crossings, found by bisection on the first half period (y grows there)
        t10 = mp.findroot(lambda t: _y(z, wn, t) - mp.mpf("0.1"), (0, half), solver="bisect")
        t90 = mp.findroot(lambda t: _y(z, wn, t) - mp.mpf("0.9"), (0, half), solver="bisect")
        assert math.isclose(r["tr_10_90"], float(t90 - t10), rel_tol=1e-9), (
            zf,
            r["tr_10_90"],
            t90 - t10,
        )
        # settling: |y - 1| = 0.02 at ts, and no extremum after ts leaves the band
        ts = mp.mpf(r["ts_2pct"])
        assert abs(abs(_y(z, wn, ts) - 1) - mp.mpf("0.02")) < mp.mpf("1e-9"), zf
        k_next = int(mp.floor(ts / half)) + 1
        assert abs(_y(z, wn, k_next * half) - 1) < mp.mpf("0.02"), zf
        assert abs(_y(z, wn, (k_next - 1) * half) - 1) > mp.mpf("0.02"), zf
        # and ts is within half a period of the envelope bound
        t_env = -mp.log(mp.mpf("0.02") * mp.sqrt(1 - z * z)) / (z * wn)
        assert t_env - half <= ts <= t_env, zf


def test_thermistor_never_below_absolute_zero() -> None:
    base = {"mode": "thermistor", "model": "beta", "R0": 10000, "T0": [25, "C"], "B": 3950}
    assert math.isclose(e("electronics", **base, R=10000)["T"], 298.15, rel_tol=1e-12)
    expect = 1 / (1 / 298.15 + math.log(1e6 / 1e4) / 3950)
    assert math.isclose(e("electronics", **base, R=1e6)["T"], expect, rel_tol=1e-12)
    for r in (1e-10, 1e-3):
        assert bad("electronics", **base, R=r) == "INVALID_ARGUMENT", r
    sh = {"mode": "thermistor", "model": "steinhart_hart", "a": -1.0, "b": 2.4e-4, "c": 1.5e-7}
    assert bad("electronics", **sh, R=10000) == "INVALID_ARGUMENT"
    # 1/T exactly 0 (no division by zero): beta with 1/T0 + ln(R/R0)/B = 2 - 2, Steinhart-Hart with a = 0, R = 1
    zero = {
        "mode": "thermistor",
        "model": "beta",
        "R0": 1,
        "T0": [0.5, "K"],
        "B": 0.5,
        "R": math.exp(-1),
    }
    assert bad("electronics", **zero) == "INVALID_ARGUMENT"
    assert bad("electronics", **{**sh, "a": 0.0}, R=1) == "INVALID_ARGUMENT"


def test_bolt_shear_sign_and_tension_only() -> None:
    plus = e("bolt_joint", size="M12", grade="8.8", Fv=[100, "kN"])
    minus = e("bolt_joint", size="M12", grade="8.8", Fv=[-100, "kN"])
    assert minus["util_shear"] == plus["util_shear"] > 3 and minus["ok"] is False
    assert e("bolt_joint", size="M12", grade="8.8", Ft=[10, "kN"])["util_tension"] > 0
    assert bad("bolt_joint", size="M12", grade="8.8", Ft=[-10, "kN"]) == "INVALID_ARGUMENT"


def test_gear_pressure_angle_range() -> None:
    assert e("gear_spur", m=2, z1=20, z2=40, alpha_deg=20)["contact_ratio"] < 2
    assert e("gear_spur", m=2, z1=20, z2=40, alpha_deg=44)["contact_ratio"] > 0
    for a in (45, 60, 90):
        assert bad("gear_spur", m=2, z1=20, z2=40, alpha_deg=a) == "INVALID_ARGUMENT", a


def test_belt_centrifugal_mass_needs_the_power() -> None:
    geo = {"D": 0.3, "d": 0.1, "C": 0.5, "mu": 0.3}
    assert bad("belt_drive", **geo, m_l=1) == "INVALID_ARGUMENT"
    r = e("belt_drive", **geo, m_l=0.2, P=[2, "kW"], n=[1450, "rpm"])
    v = 1450 * 2 * math.pi / 60 * 0.05
    assert math.isclose(r["F_centrifugal"], 0.2 * v * v, rel_tol=1e-12)


def test_psychrometrics_heating_does_not_condense() -> None:
    assert (
        bad(
            "psychrometrics",
            mode="heating_cooling",
            T1=[20, "C"],
            RH1=0.9,
            T2=[40, "C"],
            RH2=0.2,
            m=1,
        )
        == "INVALID_ARGUMENT"
    )
    heat = e("psychrometrics", mode="heating_cooling", T1=[20, "C"], RH1=0.9, T2=[40, "C"], m=1)
    assert heat["condensate_per_kg"] == 0.0 and heat["q_per_kg"] > 0
    cool = e(
        "psychrometrics", mode="heating_cooling", T1=[30, "C"], RH1=0.6, T2=[15, "C"], RH2=0.9, m=1
    )
    assert cool["condensate_per_kg"] > 0 and cool["q_per_kg"] < 0
