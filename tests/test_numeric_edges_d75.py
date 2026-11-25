"""Numeric edge cases found by the random exploration (D75).

- torsion_section with a zero torque: 0/0 in the section modulus W_t = T / tau_max (it does not depend on T);
- modern_physics blackbody at a large c2 / (lambda T): exp overflow, while the radiance is tiny but finite;
- control_system second_order heavily overdamped: the slow pole wn (zeta - sqrt(zeta^2 - 1)) loses digits by
  cancellation (zero for zeta = 1e9). References computed with mpmath at 50 digits.
"""

from __future__ import annotations

import math

import mpmath as mp

from tests.conftest import ok

H, C, K = 6.62607015e-34, 299792458.0, 1.380649e-23  # exact SI values


def test_torsion_zero_torque() -> None:
    shapes = [
        ("circle", {"R": 0.02}),
        ("annulus", {"R_e": 0.03, "R_i": 0.02}),
        ("ellipse", {"p": 0.06, "q": 0.025}),
        ("equilateral_triangle", {"a": 0.0165}),
        ("rectangle", {"b": 0.01, "t": 0.004}),
        ("thin_open", {"walls": [[0.5, 0.01], [0.2, 0.006]]}),
        ("thin_closed", {"A_m": 0.01, "walls": [[0.4, 0.002], [0.1, 0.003]]}),
    ]
    for shape, geo in shapes:
        ref = ok(
            "eng_calc",
            operation="torsion_section",
            params={"shape": shape, **geo, "T": 100.0, "G": 80e9, "L": 2.0},
        )
        zero = ok(
            "eng_calc",
            operation="torsion_section",
            params={"shape": shape, **geo, "T": 0.0, "G": 80e9, "L": 2.0},
        )
        assert zero["J"] == ref["J"] and math.isclose(zero["W_t"], ref["W_t"], rel_tol=1e-13), (
            shape,
            zero,
            ref,
        )
        assert zero["tau_max"] == 0.0 and zero["twist_rate"] == 0.0 and zero["theta"] == 0.0, (
            shape,
            zero,
        )
        assert zero.get("tau_walls", [0.0]) == [0.0] * len(ref.get("tau_walls", [0.0]))
        assert zero.get("shear_flow", 0.0) == 0.0
        # the stresses are linear in the torque
        assert math.isclose(ref["tau_max"], 100.0 / ref["W_t"], rel_tol=1e-13)


def _planck(lam: float, t: float) -> float:
    mp.mp.dps = 50
    lam_m, t_m = mp.mpf(lam), mp.mpf(t)
    c1l = 2 * mp.mpf(H) * mp.mpf(C) ** 2
    c2 = mp.mpf(H) * mp.mpf(C) / mp.mpf(K)
    return float(c1l / lam_m**5 / mp.expm1(c2 / (lam_m * t_m)))


def test_blackbody_far_wien_tail() -> None:
    for lam, t in ((1e-6, 20.0), (5e-7, 41.0), (1e-6, 1000.0)):  # c2 / (lambda T) = 719, 702, 14
        r = ok(
            "eng_calc",
            operation="modern_physics",
            params={"mode": "blackbody", "T": t, "wavelength": lam},
        )
        assert math.isclose(r["spectral_radiance"], _planck(lam, t), rel_tol=1e-12), (lam, t, r)
    cold = ok(
        "eng_calc",
        operation="modern_physics",
        params={"mode": "blackbody", "T": 0.6139646112805491, "wavelength": 4.909830056250526e-07},
    )
    assert cold["spectral_radiance"] == 0.0  # e^-47730: below the smallest float


def _settling(z: float, wn: float) -> float:
    mp.mp.dps = 50
    zm, wm = mp.mpf(z), mp.mpf(wn)
    r = mp.sqrt(zm * zm - 1)
    s1, s2 = -wm / (zm + r), -wm * (zm + r)

    def y(t: mp.mpf) -> mp.mpf:
        return 1 + (s2 * mp.e ** (s1 * t) - s1 * mp.e ** (s2 * t)) / (s1 - s2)

    # overdamped: monotone response, the 2 % settling time is where y = 0.98
    return float(mp.findroot(lambda t: y(t) - mp.mpf("0.98"), mp.log(50) / -s1))


def test_second_order_heavily_overdamped() -> None:
    for z in (3.0, 1e4, 1e9):
        r = ok(
            "eng_calc",
            operation="control_system",
            params={"mode": "second_order", "zeta": z, "wn": 5.0},
        )
        assert math.isclose(r["ts_2pct"], _settling(z, 5.0), rel_tol=1e-11), (
            z,
            r["ts_2pct"],
            _settling(z, 5.0),
        )
