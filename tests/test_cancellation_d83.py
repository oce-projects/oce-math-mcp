"""Hertz contact in a concave seat close to conforming (D83). References with mpmath at 50 digits.

Measured before the fix (relative error of the equivalent radius): seat with R2 = -R1 (1 + 1e-8) 2.7e-9, R2 = -R1
(1 + 1e-4) 1.7e-13, bearing conformity R2 = -1.04 R1 2.5e-15. 1/R = 1/R1 + 1/R2 is a difference of close reciprocals
there; R = R1 R2 / (R1 + R2) keeps the digits, because R1 + R2 is exact for close |R1| and |R2|.
"""

from __future__ import annotations

import mpmath as mp

from tests.conftest import ok

mp.mp.dps = 50


def rel(a: float, b: mp.mpf) -> float:
    return float(abs(mp.mpf(a) - b) / abs(b))


def test_hertz_close_to_conforming() -> None:
    for geo in ("sphere", "cylinder"):
        for d in (1e-12, 1e-8, 1e-4, 0.04, None, 1.0):
            params: dict[str, object] = {
                "geometry": geo,
                "F": 100.0,
                "R1": 0.01,
                "E1": 2.1e11,
                "nu1": 0.3,
            }
            if geo == "cylinder":
                params["L"] = 0.02
            if d is None:
                ref = mp.mpf(0.01)  # plane
            else:
                r2 = -0.01 * (1 + d) if d < 1 else 0.03  # concave seat, or a convex body
                params["R2"] = r2
                ref = mp.mpf(0.01) * mp.mpf(r2) / (mp.mpf(0.01) + mp.mpf(r2))
            r = ok("eng_calc", operation="contact_hertz", params=params)
            assert rel(r["R"], ref) < 1e-13, (geo, d, r["R"])
            es = 1 / (2 * (1 - mp.mpf(0.3) ** 2) / mp.mpf(2.1e11))
            if geo == "sphere":
                assert rel(r["a"], (3 * 100 * ref / (4 * es)) ** (mp.mpf(1) / 3)) < 1e-13, (
                    d,
                    r["a"],
                )
            else:
                assert rel(r["b"], mp.sqrt(4 * 100 * ref / (mp.pi * mp.mpf(0.02) * es))) < 1e-13, (
                    d,
                    r["b"],
                )
