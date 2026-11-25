"""Differences of powers and of reciprocals of close numbers (D80). References with mpmath at 50 digits.

Measured before the fix (relative error): radiation with T1 = T2 (1 + 1e-9) 1.0e-8 (T1^4 - T2^4); eccentric cylinders
almost touching 1.5e-9 (acosh close to 1); mirror with the object 1e-8 f beyond the focus 4.5e-9 (1/f - 1/d_o);
thick cylinder with t = 1e-6 r_i 4.4e-11 (r_o^2 - r_i^2); hollow shaft and annulus with d_i = d (1 - 1e-6) 2.1e-11
(d^4 - d_i^4). a^2 - b^2 = (a - b)(a + b) and a^4 - b^4 = (a - b)(a + b)(a^2 + b^2) keep the digits, because a - b
is exact for close a and b.
"""

from __future__ import annotations

import mpmath as mp

from oce_math_mcp.tools.eng_em import EPS0
from oce_math_mcp.tools.eng_thermo import SIGMA_SB
from tests.conftest import ok

mp.mp.dps = 50
CLOSE = (1e-3, 1e-6, 1e-9, 0.3)
SG = mp.mpf(SIGMA_SB)


def rel(a: float, b: mp.mpf) -> float:
    return float(abs(mp.mpf(a) - b) / abs(b))


def e(op: str, **params: object) -> dict:
    return ok("eng_calc", operation=op, params=params)


def test_radiation_small_temperature_difference() -> None:
    for d in (*CLOSE, -1e-9):
        t1 = 300.0 * (1 + d)
        d4 = mp.mpf(t1) ** 4 - mp.mpf(300) ** 4
        r = e("heat_transfer", mode="radiation", emissivity=0.9, A=2.0, T1=t1, T2=300.0)
        assert rel(r["Q"], mp.mpf(0.9) * SG * 2 * d4) < 1e-13, (d, r["Q"])
        r = e(
            "radiation_exchange",
            mode="two_surface",
            config="parallel_plates",
            A=1.0,
            T1=t1,
            T2=300.0,
            eps1=0.8,
            eps2=0.8,
        )
        assert rel(r["Q"], SG * d4 / (2 / mp.mpf(0.8) - 1)) < 1e-13, (d, r["Q"])


def test_thin_walled_sections() -> None:
    for d in CLOSE:
        di = 0.05 * (1 - d)
        r = e("shaft_torsion_bending", d=0.05, d_i=di, T=100.0)
        j = mp.pi * (mp.mpf(0.05) ** 4 - mp.mpf(di) ** 4) / 32
        assert rel(r["J"], j) < 1e-13, (d, r["J"])
        assert rel(r["tau_max"], 100 * mp.mpf(0.025) / j) < 1e-13, (d, r["tau_max"])
        r = e("torsion_section", shape="annulus", R_e=0.05, R_i=di, T=100.0)
        j = mp.pi / 2 * (mp.mpf(0.05) ** 4 - mp.mpf(di) ** 4)
        assert rel(r["J"], j) < 1e-13, (d, r["J"])
        assert rel(r["tau_max"], 100 * mp.mpf(0.05) / j) < 1e-13, (d, r["tau_max"])


def test_thin_thick_cylinder_and_fit() -> None:
    for d in CLOSE:
        t = 0.5 * d
        ro = 0.5 + t  # as the server computes it
        r = e("pressure_vessel", p=1e6, r_i=0.5, t=t, shape="cylinder")
        a2, b2 = mp.mpf(0.5) ** 2, mp.mpf(ro) ** 2
        assert rel(r["sigma_t_inner"], mp.mpf(1e6) * (a2 + b2) / (b2 - a2)) < 1e-13, (
            d,
            r["sigma_t_inner"],
        )
        assert rel(r["sigma_t_outer"], mp.mpf(1e6) * 2 * a2 / (b2 - a2)) < 1e-13, (
            d,
            r["sigma_t_outer"],
        )
        big, small = 0.05 * (1 + d), 0.05 * (1 - d)
        r = e("interference_fit", d=0.05, D_o=big, d_i=small, delta=2e-5, E_hub=2.1e11, nu_hub=0.3)
        dm, do, di = mp.mpf(0.05), mp.mpf(big), mp.mpf(small)
        kh, ks = (do**2 + dm**2) / (do**2 - dm**2), (dm**2 + di**2) / (dm**2 - di**2)
        pc = mp.mpf(2e-5) / (dm * ((kh + mp.mpf(0.3)) + (ks - mp.mpf(0.3))) / mp.mpf(2.1e11))
        assert rel(r["p"], pc) < 1e-13, (d, r["p"])
        assert rel(r["sigma_t_shaft_inner"], -2 * pc * dm**2 / (dm**2 - di**2)) < 1e-13, d


def test_mirror_near_the_focus() -> None:
    for d in (1e-4, 1e-8, 1e-12, -1e-8, 2.0):
        do = 0.1 * (1 + d)
        r = e("geometric_optics", mode="mirror", f=0.1, d_o=do)
        assert rel(r["d_i"], 1 / (1 / mp.mpf(0.1) - 1 / mp.mpf(do))) < 1e-13, (d, r["d_i"])
        r = e("geometric_optics", mode="mirror", R=0.2, d_i=do)
        assert rel(r["d_o"], 1 / (1 / mp.mpf(0.1) - 1 / mp.mpf(do))) < 1e-13, (d, r["d_o"])


def test_eccentric_cylinders_almost_touching() -> None:
    for d in (1e-4, 1e-8, 1e-12, 0.5, 1.0):
        dd = 1e-3 * (1 - d)
        r = e(
            "electrostatics",
            mode="capacitor",
            geometry="eccentric_cylinders",
            R_inner=1e-3,
            R_outer=2e-3,
            d=dd,
            L=1.0,
        )
        ri, ro, dm = mp.mpf(1e-3), mp.mpf(2e-3), mp.mpf(dd)
        ref = 2 * mp.pi * mp.mpf(EPS0) / mp.acosh((ro**2 + ri**2 - dm**2) / (2 * ro * ri))
        assert rel(r["C"], ref) < 1e-13, (d, r["C"])
