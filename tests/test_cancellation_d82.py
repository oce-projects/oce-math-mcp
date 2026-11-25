"""Lenses, refracting surface, wires and catenary near their singular points (D82). References with mpmath.

Measured before the fix (relative error, case 1e-8): thin lens and mirror near the focus or with d_i close to -d_o
about 4.5e-9 (1/f - 1/d_o, 1/d_o + 1/d_i); thin meniscus with R2 = R1 (1 + 1e-8) 4.5e-9 (1/R1 - 1/R2); refracting
surface with the object near the focus 4.5e-9 and 5.2e-9 ((n2 - n1)/R - n1/d_o); two wires and wire near a wall almost
touching 1.0e-9 (acosh close to 1); catenary: sag with a = 1e4 H 7.1e-8 (a cosh(x/a) - a), taut cable with
L = H (1 + 1e-8) 4.8e-9 in a and 5.8e-9 in the sag (2 a sinh(H/(2a)) - L), sag 1e-4 H 5.4e-10 (acosh close to 1).
"""

from __future__ import annotations

import math

import mpmath as mp

from oce_math_mcp.tools.eng_em import EPS0
from oce_math_mcp.tools.eng_mag import MU0
from tests.conftest import call, ok

mp.mp.dps = 50
CLOSE = (1e-4, 1e-8, 1e-12, -1e-8, 2.0)


def rel(a: float, b: mp.mpf) -> float:
    return float(abs(mp.mpf(a) - b) / abs(b))


def e(op: str, **params: object) -> dict:
    return ok("eng_calc", operation=op, params=params)


def test_thin_lens_near_the_focus() -> None:
    for d in CLOSE:
        x = 0.1 * (1 + d)
        ref = 1 / (1 / mp.mpf(0.1) - 1 / mp.mpf(x))
        r = e("optics_acoustics", mode="thin_lens", f=0.1, d_o=x)
        assert rel(r["d_i"], ref) < 1e-13, (d, r["d_i"])
        r = e("optics_acoustics", mode="thin_lens", f=0.1, d_i=x)
        assert rel(r["d_o"], ref) < 1e-13, (d, r["d_o"])
        di = -0.3 * (1 + d)
        ref = 1 / (1 / mp.mpf(0.3) + 1 / mp.mpf(di))
        r = e("optics_acoustics", mode="thin_lens", d_o=0.3, d_i=di)
        assert rel(r["f"], ref) < 1e-13, (d, r["f"])
        r = e("geometric_optics", mode="mirror", d_o=0.3, d_i=di)
        assert rel(r["f"], ref) < 1e-13, (d, r["f"])


def test_thin_lens_singular_points_have_a_clear_message() -> None:
    for params, word in (
        ({"mode": "thin_lens", "f": 0.1, "d_i": 0.1}, "focus"),
        ({"mode": "thin_lens", "d_o": 0.3, "d_i": -0.3}, "no power"),
    ):
        resp = call("eng_calc", operation="optics_acoustics", params=params)
        assert not resp["ok"] and resp["error"]["code"] == "MATH_ERROR", resp
        assert word in resp["error"]["message"], resp["error"]
        assert "ZeroDivision" not in resp["error"]["message"], resp["error"]


def test_meniscus_lens() -> None:
    for d in (1e-4, 1e-8, 1e-11, -1e-8, 0.5):
        r2 = 0.1 * (1 + d)
        for th, nm in ((0.0, None), (0.004, None), (0.0, 1.33)):
            params: dict[str, object] = {
                "mode": "lensmaker",
                "n": 1.5,
                "R1": 0.1,
                "R2": r2,
                "d": th,
            }
            if nm is not None:
                params["n_medium"] = nm
            r = e("geometric_optics", **params)
            n = mp.mpf(1.5 / nm) if nm is not None else mp.mpf(1.5)  # the server divides in floats
            a1, a2, t = mp.mpf(0.1), mp.mpf(r2), mp.mpf(th)
            ref = 1 / ((n - 1) * (1 / a1 - 1 / a2 + (n - 1) * t / (n * a1 * a2)))
            assert rel(r["f"], ref) < 1e-13, (d, th, nm, r["f"])


def test_refracting_surface_near_the_focus() -> None:
    for n1, n2, rr in ((1.0, 1.5, 0.1), (1.333, 1.52, -0.05)):
        fo, fi = n1 * rr / (n2 - n1), n2 * rr / (n2 - n1)
        a, b, c = mp.mpf(n1), mp.mpf(n2), mp.mpf(rr)
        for d in CLOSE:
            x = fo * (1 + d)
            r = e("geometric_optics", mode="refracting_surface", n1=n1, n2=n2, R=rr, d_o=x)
            assert rel(r["d_i"], b / ((b - a) / c - a / mp.mpf(x))) < 1e-13, (n1, d, r["d_i"])
            x = fi * (1 + d)
            r = e("geometric_optics", mode="refracting_surface", n1=n1, n2=n2, R=rr, d_i=x)
            assert rel(r["d_o"], a / ((b - a) / c - b / mp.mpf(x))) < 1e-13, (n1, d, r["d_o"])


def test_wires_almost_touching() -> None:
    for d in (1e-4, 1e-8, 1e-12, 0.5, 10.0, 1e6, 1e160):
        x = 2e-3 * (1 + d)
        r = e("electrostatics", mode="capacitor", geometry="parallel_wires", a=1e-3, d=x, L=1.0)
        ref = mp.pi * mp.mpf(EPS0) / mp.acosh(mp.mpf(x) / mp.mpf(2e-3))
        assert rel(r["C"], ref) < 1e-13, (d, r["C"])
        r = e("inductance", geometry="parallel_wires", s=x, a=1e-3, L=1.0, hf=True)
        assert rel(r["L"], mp.mpf(MU0) / mp.pi * mp.acosh(mp.mpf(x) / mp.mpf(2e-3))) < 1e-13, d
        x = 1e-3 * (1 + d)
        r = e("electrostatics", mode="capacitor", geometry="wire_wall", a=1e-3, d=x, L=1.0)
        ref = 2 * mp.pi * mp.mpf(EPS0) / mp.acosh(mp.mpf(x) / mp.mpf(1e-3))
        assert rel(r["C"], ref) < 1e-13, (d, r["C"])


def _sag(a: mp.mpf, x: float) -> mp.mpf:
    return a * (mp.cosh(mp.mpf(x) / a) - 1)


def test_catenary_from_the_tension() -> None:
    for t0 in (10.0, 1e4, 1e6, 1e9):
        r = e("cable_catenary", mode="span_tension", w=1.0, H=100.0, T0=t0)
        a = mp.mpf(t0)
        assert rel(r["sag"], _sag(a, 50.0)) < 1e-13, (t0, r["sag"])
        assert rel(r["L"], 2 * a * mp.sinh(50 / a)) < 1e-13, (t0, r["L"])


def test_catenary_taut_cable() -> None:
    for d in (1e-2, 1e-4, 1e-8, 1e-12):
        ln = 100.0 * (1 + d)
        r = e("cable_catenary", mode="span_length", w=1.0, H=100.0, L=ln)
        a = mp.findroot(lambda x: 2 * x * mp.sinh(50 / x) - mp.mpf(ln), mp.mpf(r["a"]))
        assert rel(r["a"], a) < 1e-13, (d, r["a"])
        assert rel(r["sag"], _sag(a, 50.0)) < 1e-13, (d, r["sag"])
        assert rel(r["L"], mp.mpf(ln)) < 1e-13, (d, r["L"])


def test_catenary_with_a_height_difference() -> None:
    for d in (1e-2, 1e-4, 1e-8):
        ln = 50.0 * (1 + d)
        h = 0.5 * math.sqrt(ln * ln - 2500.0)
        r = e("cable_catenary", mode="span_length", w=1.0, H=h, L=ln, v=50.0)
        tg = mp.sqrt(mp.mpf(ln) ** 2 - 2500)
        a = mp.findroot(lambda x: 2 * x * mp.sinh(mp.mpf(h) / 2 / x) - tg, mp.mpf(r["a"]))
        assert rel(r["a"], a) < 1e-13, (d, r["a"])
        assert rel(r["L"], mp.mpf(ln)) < 1e-13, (d, r["L"])
    for d in (1e-2, 1e-6, 1e-9, 1e-12):
        ln = math.hypot(100.0, 10.0) * (1 + d)
        r = e("cable_catenary", mode="span_length", w=1.0, H=100.0, L=ln, v=10.0)
        tg = mp.sqrt(mp.mpf(ln) ** 2 - 100)
        a = mp.findroot(lambda x: 2 * x * mp.sinh(50 / x) - tg, mp.mpf(r["a"]))
        assert rel(r["a"], a) < 1e-13, (d, r["a"])
        assert rel(r["L"], mp.mpf(ln)) < 1e-13, (d, r["L"])


def test_catenary_from_the_sag() -> None:
    for d in (1e-1, 1e-4, 1e-8, 1e-11):
        sg = 100.0 * d
        r = e("cable_catenary", mode="span_sag", w=1.0, H=100.0, sag=sg)
        a = mp.findroot(lambda x: 2 * x * mp.acosh((mp.mpf(sg) + x) / x) - 100, mp.mpf(r["a"]))
        assert rel(r["a"], a) < 1e-13, (d, r["a"])
        assert rel(r["sag"], mp.mpf(sg)) < 1e-13, (d, r["sag"])
