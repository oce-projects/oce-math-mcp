"""Stress concentration at holes in an infinite plate (D71).

Checks independent of the server formulas, on the field returned by the server:
- equilibrium in polar coordinates, d(s_rr)/dr + (1/r) d(s_rt)/dt + (s_rr - s_tt)/r = 0 and
  d(s_rt)/dr + (1/r) d(s_tt)/dt + 2 s_rt / r = 0, by central finite differences;
- compatibility of plane elasticity without body forces, Laplacian(s_rr + s_tt) = 0, by finite differences;
- traction-free hole edge (s_rr = s_rt = 0 at r = a) and the far field equal to the applied stresses (Cartesian
  components at r = 1e5 a);
- edge maximum and minimum against a scan of the hoop stress around the hole.
These conditions determine the elastic solution; the uniaxial case must give the page value K_t = 3.
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import call, ok


def sc(**params: object) -> dict:
    return ok("eng_calc", operation="stress_concentration", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="stress_concentration", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def _field(sx: float, sy: float, a: float, r: float, t: float) -> np.ndarray:
    res = sc(mode="circular_hole", sigma_x=sx, sigma_y=sy, a=a, r=r, theta_deg=math.degrees(t))
    return np.array([res["sigma_rr"], res["sigma_tt"], res["sigma_rt"]])


CASES = [(100e6, 0.0), (-60e6, 0.0), (50e6, 50e6), (80e6, -80e6), (30e6, 120e6)]


def test_kirsch_field_is_an_elastic_solution() -> None:
    a = 0.01
    for sx, sy in CASES:
        s = max(abs(sx), abs(sy))
        for r, t in ((1.3 * a, 0.4), (2.0 * a, 1.1), (3.7 * a, 2.5), (1.05 * a, 4.0)):
            hr, ht = 1e-4 * a, 1e-4
            d_r = (_field(sx, sy, a, r + hr, t) - _field(sx, sy, a, r - hr, t)) / (2 * hr)
            d_t = (_field(sx, sy, a, r, t + ht) - _field(sx, sy, a, r, t - ht)) / (2 * ht)
            srr, stt, srt = _field(sx, sy, a, r, t)
            eq1 = d_r[0] + d_t[2] / r + (srr - stt) / r
            eq2 = d_r[2] + d_t[1] / r + 2 * srt / r
            assert abs(eq1) < 1e-6 * s / a and abs(eq2) < 1e-6 * s / a, (sx, sy, r, t, eq1, eq2)
            h2r, h2t = 1e-3 * a, 1e-3

            def tr(rr: float, tt: float) -> float:
                f = _field(sx, sy, a, rr, tt)
                return float(f[0] + f[1])

            c = tr(r, t)
            lap = (
                (tr(r + h2r, t) - 2 * c + tr(r - h2r, t)) / h2r**2
                + (tr(r + h2r, t) - tr(r - h2r, t)) / (2 * h2r) / r
                + (tr(r, t + h2t) - 2 * c + tr(r, t - h2t)) / h2t**2 / r**2
            )
            assert abs(lap) < 1e-5 * s / a**2, (sx, sy, r, t, lap)
        for t in np.linspace(0, 2 * math.pi, 13):
            srr, _, srt = _field(sx, sy, a, a, float(t))
            assert abs(srr) < 1e-12 * s and abs(srt) < 1e-12 * s
            srr, stt, srt = _field(sx, sy, a, 1e5 * a, float(t))
            c, n = math.cos(t), math.sin(t)
            sxx = srr * c * c + stt * n * n - 2 * srt * n * c
            syy = srr * n * n + stt * c * c + 2 * srt * n * c
            txy = (srr - stt) * n * c + srt * (c * c - n * n)
            assert abs(sxx - sx) < 1e-8 * s and abs(syy - sy) < 1e-8 * s and abs(txy) < 1e-8 * s


def test_edge_extremes_and_kt() -> None:
    a = 0.02
    thetas = np.linspace(0.0, 180.0, 361)
    for sx, sy in CASES:
        r = sc(mode="circular_hole", sigma_x=sx, sigma_y=sy, a=a)
        hoop = [
            sc(mode="circular_hole", sigma_x=sx, sigma_y=sy, a=a, r=a, theta_deg=float(t))[
                "sigma_tt"
            ]
            for t in thetas
        ]
        assert math.isclose(r["sigma_edge_max"], max(hoop), rel_tol=1e-12, abs_tol=1e-6)
        assert math.isclose(r["sigma_edge_min"], min(hoop), rel_tol=1e-12, abs_tol=1e-6)
        at_max = sc(
            mode="circular_hole",
            sigma_x=sx,
            sigma_y=sy,
            a=a,
            r=a,
            theta_deg=r["theta_edge_max_deg"],
        )
        assert math.isclose(at_max["sigma_tt"], r["sigma_edge_max"], rel_tol=1e-12)
        kt = max(abs(max(hoop)), abs(min(hoop))) / max(abs(sx), abs(sy))
        assert math.isclose(r["Kt"], kt, rel_tol=1e-12)
    assert sc(mode="circular_hole", sigma_x=100e6, a=a)["Kt"] == 3.0  # page: circular hole, K_t = 3
    assert "r >= a" in bad(
        {"mode": "circular_hole", "sigma_x": 1e6, "a": 0.02, "r": 0.01, "theta_deg": 0}
    )
    assert "theta_deg" in bad({"mode": "circular_hole", "sigma_x": 1e6, "a": 0.02, "r": 0.03})
    assert "non-zero" in bad({"mode": "circular_hole", "sigma_x": 0.0, "a": 0.02})


def test_inglis_elliptical_hole() -> None:
    # a = b is the circular hole: same K_t as the Kirsch solution (two different pages)
    circle = sc(mode="elliptical_hole", sigma=100e6, a=0.01, b=0.01)
    assert math.isclose(
        circle["Kt"], sc(mode="circular_hole", sigma_x=100e6, a=0.01)["Kt"], rel_tol=1e-15
    )
    e = sc(mode="elliptical_hole", sigma=50e6, a=0.01, b=0.002)
    assert math.isclose(e["Kt"], 1 + 2 * 0.01 / 0.002, rel_tol=1e-15)
    assert math.isclose(e["sigma_max"], 50e6 * e["Kt"], rel_tol=1e-15)
    assert math.isclose(e["rho"], 0.002**2 / 0.01, rel_tol=1e-15)
    # the radius of curvature form 1 + 2 sqrt(a / rho) gives the same result
    f = sc(mode="elliptical_hole", sigma=50e6, a=0.01, rho=0.002**2 / 0.01)
    assert math.isclose(f["Kt"], e["Kt"], rel_tol=1e-13) and math.isclose(
        f["b"], 0.002, rel_tol=1e-13
    )
    assert "exactly one" in bad({"mode": "elliptical_hole", "sigma": 1e6, "a": 0.01})
