"""Formulas that lost digits by cancellation (D76). References with mpmath at 50 digits.

Measured before the fix (relative error): Compton shift at 0.01 degrees 3.5e-9; view factor of parallel strips with
w / L = 1e-6 8.9e-5; perpendicular strips with w_j / w_i = 1e-6 1.3e-10; inclined strips at 179.999 degrees 8.0e-7.
"""

from __future__ import annotations

import mpmath as mp

from tests.conftest import ok

mp.mp.dps = 50
H, C, M_E = mp.mpf("6.62607015e-34"), mp.mpf(299792458), mp.mpf("9.1093837139e-31")  # CODATA 2022


def rel(a: float, b: mp.mpf) -> float:
    return float(abs(mp.mpf(a) - b) / abs(b))


def vf(**params: object) -> float:
    return ok(
        "eng_calc", operation="radiation_exchange", params={"mode": "view_factor_2d", **params}
    )["F_ij"]


def test_compton_small_angles() -> None:
    for deg in (0.01, 0.1, 1.0, 90.0, 179.0):
        r = ok(
            "eng_calc",
            operation="modern_physics",
            params={"mode": "compton", "energy": 1e-15, "angle_deg": deg},
        )
        ref = H / (M_E * C) * (1 - mp.cos(mp.radians(mp.mpf(deg))))
        assert rel(r["shift"], ref) < 1e-13, (deg, r["shift"], ref)


def test_view_factors_of_strips() -> None:
    for w in (1e-2, 1e-4, 1e-6, 3.0):
        a = b = mp.mpf(w)
        ref = (mp.sqrt((a + b) ** 2 + 4) - mp.sqrt((b - a) ** 2 + 4)) / (2 * a)
        assert rel(vf(geometry="parallel_plates", w_i=w, w_j=w, L=1.0), ref) < 1e-13, w
    assert (
        rel(
            vf(geometry="parallel_plates", w_i=0.3, w_j=1e-5, L=2.0),
            (lambda a, b: (mp.sqrt((a + b) ** 2 + 4) - mp.sqrt((b - a) ** 2 + 4)) / (2 * a))(
                mp.mpf(0.3) / 2, mp.mpf(1e-5) / 2
            ),
        )
        < 1e-13
    )
    for r in (1e-2, 1e-4, 1e-6, 50.0):
        rr = mp.mpf(r)
        assert (
            rel(
                vf(geometry="perpendicular_plates", w_i=1.0, w_j=r),
                (1 + rr - mp.sqrt(1 + rr * rr)) / 2,
            )
            < 1e-13
        ), r
    for alpha in (10.0, 170.0, 179.9, 179.999):
        ref = 1 - mp.sin(mp.radians(mp.mpf(alpha)) / 2)
        assert rel(vf(geometry="inclined_plates", w=1.0, alpha_deg=alpha), ref) < 1e-13, alpha
    # no overflow inside the roots for huge strips: the limit of the perpendicular case is 1/2
    assert (
        rel(vf(geometry="perpendicular_plates", w_i=2.511125015260122, w_j=1e300), mp.mpf("0.5"))
        < 1e-13
    )
    assert rel(vf(geometry="parallel_plates", w_i=1e300, w_j=1e300, L=1.0), mp.mpf(1)) < 1e-13
