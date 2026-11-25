"""Extreme ranges after the cancellation fixes (D84). References with mpmath at 50 digits.

An independent review of D79-D83 found that some exact forms multiply two lengths (f d_o, R1 R2, n2 R d_o), which
overflow or underflow where the old reciprocals did not: a mirror with d_o = 1e308 (a client writing infinity as
1e308) answered "at the focus", a thin lens with f = 1e-200 answered d_i = 0. The thin lens also accepted f = 0 and
d_i = 0. Lame's sigma_r at the walls lost up to 3e-11 against the exact -p. And `fmt_float`, used in the
`formatted` texts, raised decimal.InvalidOperation from 1e22 on (28 digits of the Decimal context): ten operations
answered "InvalidOperation" for large but finite results.
"""

from __future__ import annotations

import mpmath as mp

from oce_math_mcp.fmt import fmt_float
from tests.conftest import call, ok

mp.mp.dps = 50


def rel(a: float, b: mp.mpf) -> float:
    return float(abs(mp.mpf(a) - b) / abs(b))


def e(op: str, **params: object) -> dict:
    return ok("eng_calc", operation=op, params=params)


def test_mirror_and_thin_lens_far_and_tiny() -> None:
    assert rel(e("geometric_optics", mode="mirror", f=2.0, d_o=1e308)["d_i"], mp.mpf(2)) < 1e-13
    assert rel(e("geometric_optics", mode="mirror", R=4.0, d_i=1e308)["d_o"], mp.mpf(2)) < 1e-13
    ref = 1 / (1 / mp.mpf(1e308) + 1 / mp.mpf(3))
    assert (
        rel(e("geometric_optics", mode="mirror", d_o=1e308, d_i=3.0)["f"], ref) < 1e-13
    )  # d_o d_i > 1.8e308
    # d_o = 0.3, d_i = 1e308 is refused: the magnification -d_i / d_o is beyond the float range
    for f, x in ((2.0, 1e308), (1e-200, 3e-200), (1e200, 3e200), (0.1, 0.3)):
        ref = 1 / (1 / mp.mpf(f) - 1 / mp.mpf(x))
        assert rel(e("optics_acoustics", mode="thin_lens", f=f, d_o=x)["d_i"], ref) < 1e-13, (f, x)
        assert rel(e("optics_acoustics", mode="thin_lens", f=f, d_i=x)["d_o"], ref) < 1e-13, (f, x)
        ref = 1 / (1 / mp.mpf(f) + 1 / mp.mpf(x))
        assert rel(e("optics_acoustics", mode="thin_lens", d_o=f, d_i=x)["f"], ref) < 1e-13, (f, x)
        assert rel(e("optics_acoustics", mode="thin_lens", d_o=x, d_i=f)["f"], ref) < 1e-13, (f, x)


def test_thin_lens_rejects_zero() -> None:
    for params in ({"f": 0.0, "d_o": 0.3}, {"d_i": 0.0, "d_o": 0.3}, {"f": 0.0, "d_i": 0.3}):
        resp = call(
            "eng_calc", operation="optics_acoustics", params={"mode": "thin_lens", **params}
        )
        assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", (params, resp)


def test_lensmaker_and_surface_extreme_radii() -> None:
    for r1, r2 in ((1e160, -1e160), (1e-300, -1e-300), (1e-160, -2e-160), (1e300, 2e300)):
        r = e("geometric_optics", mode="lensmaker", n=1.5, R1=r1, R2=r2)
        ref = 1 / (mp.mpf(0.5) * (1 / mp.mpf(r1) - 1 / mp.mpf(r2)))
        assert r["afocal"] is False and rel(r["f"], ref) < 1e-13, (r1, r2, r)
    for n2, rr, x in ((1.5, 2.0, 1e308), (4.0, 0.1, 1e308), (1.5, 1e-200, 4e-200)):
        a, b = mp.mpf(1), mp.mpf(n2)
        r = e("geometric_optics", mode="refracting_surface", n1=1.0, n2=n2, R=rr, d_o=x)
        assert rel(r["d_i"], b / ((b - a) / mp.mpf(rr) - a / mp.mpf(x))) < 1e-13, (n2, rr, x, r)
        if n2 == 4.0:
            continue  # with d_i = 1e308 the magnification is beyond the float range
        r = e("geometric_optics", mode="refracting_surface", n1=1.0, n2=n2, R=rr, d_i=x)
        assert rel(r["d_o"], a / ((b - a) / mp.mpf(rr) - b / mp.mpf(x))) < 1e-13, (n2, rr, x, r)


def test_surface_beyond_the_float_range_is_a_clean_error() -> None:
    resp = call(
        "eng_calc",
        operation="geometric_optics",
        params={"mode": "refracting_surface", "n1": 1.0, "n2": 1.5, "R": 1e-320, "d_o": 1.0},
    )
    assert resp["error"]["message"].startswith("The result is not finite"), resp["error"]


def test_hertz_with_a_huge_radius() -> None:
    for r2 in (1e308, -1e308, 1e-300):
        r = e("contact_hertz", geometry="sphere", F=100.0, R1=2.0, R2=r2, E1=2.1e11, nu1=0.3)
        ref = mp.mpf(2) * mp.mpf(r2) / (2 + mp.mpf(r2))
        assert rel(r["R"], ref) < 1e-13, (r2, r["R"])


def test_catenary_extreme_lengths() -> None:
    resp = call(
        "eng_calc",
        operation="cable_catenary",
        params={"mode": "span_length", "w": 1.0, "H": 100.0, "L": 1e200},
    )
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp["error"]
    for h, ln in ((1e-170, 1.5e-170), (1e150, 1.5e150)):
        r = e("cable_catenary", mode="span_length", w=1.0, H=h, L=ln)
        # normalised: a = y H, so that the tolerance of findroot does not depend on the scale
        y0 = mp.mpf(r["a"]) / mp.mpf(h)
        y = mp.findroot(
            lambda q: 2 * q * mp.sinh(1 / (2 * q)) - mp.mpf(ln) / mp.mpf(h),
            (y0 * 0.999, y0 * 1.001),
        )
        a = y * mp.mpf(h)
        assert rel(r["a"], a) < 1e-13, (h, r["a"])
        assert rel(r["L"], mp.mpf(ln)) < 1e-13, (h, r["L"])


def test_lame_radial_stress_at_the_walls_is_the_pressure() -> None:
    for t in (1e-9, 1e-6, 1e-3, 0.5):
        for po in (0.0, 2e5):
            for shape in ("cylinder", "sphere"):
                r = e("pressure_vessel", p=1e6, p_o=po, r_i=0.5, t=t, shape=shape)
                assert r["sigma_r_inner"] == -1e6 and r["sigma_r_outer"] == -po, (t, po, shape, r)


def test_thick_sphere_thin_wall() -> None:
    for t in (1e-9, 1e-6, 1e-3, 0.5):
        r = e("pressure_vessel", p=1e6, r_i=0.5, t=t, shape="sphere")
        a3, b3 = mp.mpf(0.5) ** 3, mp.mpf(0.5 + t) ** 3  # r_o as the server computes it
        c = mp.mpf(1e6) * a3 / (b3 - a3)
        d = mp.mpf(1e6) * a3 * b3 / (b3 - a3)
        assert rel(r["sigma_t_inner"], c + d / (2 * a3)) < 1e-13, (t, r["sigma_t_inner"])
        assert rel(r["sigma_t_outer"], c + d / (2 * b3)) < 1e-13, (t, r["sigma_t_outer"])


def test_fmt_float_beyond_28_digits() -> None:
    assert fmt_float(1e21) == "1,000,000,000,000,000,000,000"  # unchanged below the limit
    assert fmt_float(1e22) == "1e+22"
    assert fmt_float(-2.5e300, 3) == "-2.5e+300"
    assert fmt_float(float("inf")) == "inf"
    r = e("bearing_life", type="roller", P=1e-9, C=3.779)
    assert r["L10"] > 1e25, r
    resp = call(
        "eng_calc",
        operation="heat_transfer",
        params={"mode": "radiation", "emissivity": 1.0, "A": 1.0, "T1": 1e80, "T2": 300.0},
    )
    assert resp["error"]["code"] == "MATH_ERROR", resp["error"]
    assert resp["error"]["message"].startswith("The result is not finite"), resp["error"]
