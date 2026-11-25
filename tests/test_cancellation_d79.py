"""Logarithm of a ratio close to 1 (D79). References with mpmath at 50 digits.

Measured before the fix (relative error): LMTD with d1 / d2 = 1 + 1e-11 1.1e-5 (1 + 1e-8: 8.7e-9); coaxial capacitor
with a gap of 1e-9 R_inner 8.7e-8; Arrhenius Ea from two temperatures 1e-9 apart 4.8e-8; cylindrical wall, toroid and
isothermal work with a ratio 1 + 1e-6 2.8e-11. The error of ln(a / b) comes from rounding a / b: log1p((a - b) / b)
avoids it, since a - b is exact for b / 2 <= a <= 2 b.
"""

from __future__ import annotations

import mpmath as mp

from oce_math_mcp.tools.eng_chem import R_GAS
from oce_math_mcp.tools.eng_em import EPS0
from oce_math_mcp.tools.eng_gas import GAMMA_AIR, R_AIR
from oce_math_mcp.tools.eng_mag import MU0
from tests.conftest import ok

mp.mp.dps = 50
GAPS = (1e-3, 1e-6, 1e-9, 1e-12, 9.0)


def rel(a: float, b: mp.mpf) -> float:
    return float(abs(mp.mpf(a) - b) / abs(b))


def e(op: str, **params: object) -> dict:
    return ok("eng_calc", operation=op, params=params)


def lnr(a: float, b: float) -> mp.mpf:
    return mp.log(mp.mpf(a) / mp.mpf(b))


def test_lmtd_with_close_differences() -> None:
    for d in (1e-3, 1e-8, 1e-11, 9e-13, 1e-14, 0.0, 0.5):
        for flow in ("counter", "parallel"):
            if flow == "counter":
                t = {"Th_in": 400.0, "Th_out": 350.0, "Tc_in": 300.0 + 50.0 * d, "Tc_out": 350.0}
                d1, d2 = mp.mpf(400) - mp.mpf(350), mp.mpf(350) - mp.mpf(t["Tc_in"])
            else:
                t = {
                    "Th_in": 400.0,
                    "Th_out": 380.0,
                    "Tc_in": 300.0,
                    "Tc_out": 300.0 + 20.0 * (1 - d),
                }
                d1, d2 = mp.mpf(100), mp.mpf(380) - mp.mpf(t["Tc_out"])
            r = e("heat_transfer", mode="lmtd", flow=flow, U=100.0, A=1.0, **t)
            ref = d1 if d1 == d2 else (d1 - d2) / mp.log(d1 / d2)
            assert rel(r["LMTD"], ref) < 1e-13, (flow, d, r["LMTD"], ref)


def test_thin_walls_and_gaps() -> None:
    for d in GAPS:
        r2 = 0.05 * (1 + d)
        r = e("heat_transfer", mode="cylinder", r1=0.05, r2=r2, k=50.0, L=1.0, T1=400.0, T2=300.0)
        assert rel(r["R_total"], lnr(r2, 0.05) / (2 * mp.pi * 50)) < 1e-13, ("cylinder", d)
        ro = 1e-3 * (1 + d)
        r = e(
            "electrostatics", mode="capacitor", geometry="coaxial", R_inner=1e-3, R_outer=ro, L=1.0
        )
        assert rel(r["C"], 2 * mp.pi * mp.mpf(EPS0) / lnr(ro, 1e-3)) < 1e-13, ("coaxial C", d)
        r = e(
            "electrostatics",
            mode="capacitor",
            geometry="concentric_spheres",
            R_inner=1e-3,
            R_outer=ro,
        )
        ref = 4 * mp.pi * mp.mpf(EPS0) / (1 / mp.mpf(1e-3) - 1 / mp.mpf(ro))
        assert rel(r["C"], ref) < 1e-13, ("spheres C", d)
        r = e("inductance", geometry="toroid", N=100, h=0.01, r_inner=0.05, r_outer=r2)
        assert (
            rel(r["L"], mp.mpf(MU0) * 100**2 * mp.mpf(0.01) * lnr(r2, 0.05) / (2 * mp.pi)) < 1e-13
        ), ("toroid", d)
        r = e("inductance", geometry="coaxial", a=1e-3, b=ro, L=1.0)
        assert rel(r["L"], mp.mpf(MU0) / (2 * mp.pi) * lnr(ro, 1e-3)) < 1e-13, ("coaxial L", d)


def test_isothermal_gas() -> None:
    for d in (1e-6, 1e-9, 1e-12, 1.0, -0.5):
        v2 = 0.1 * (1 + d)
        r = e("ideal_gas", mode="process", process="isothermal", p1=1e5, T1=300.0, V1=0.1, V2=v2)
        w = mp.mpf(1e5) * mp.mpf(0.1) * lnr(v2, 0.1)
        assert rel(r["W"], w) < 1e-13, (d, r["W"])
        assert rel(r["dS"], w / 300) < 1e-13, (d, r["dS"])


def test_two_point_fits() -> None:
    for d in (1e-6, 1e-9, 0.1):
        t2, k2 = 300.0 * (1 + d), 1.0 + 10 * d
        r = e("reaction_kinetics", mode="arrhenius", k1=1.0, T1=300.0, k2=k2, T2=t2)
        ref = mp.mpf(R_GAS) * lnr(k2, 1.0) / (1 / mp.mpf(300) - 1 / mp.mpf(t2))
        assert rel(r["Ea"], ref) < 1e-13, ("Arrhenius", d, r["Ea"])
        r1, r2 = 10000.0, 10000.0 * (1 - 4 * d)
        r = e("electronics", mode="thermistor", model="fit", points=[[300.0, r1], [t2, r2]])
        ref = lnr(r1, r2) / (1 / mp.mpf(300) - 1 / mp.mpf(t2))
        assert rel(r["B"], ref) < 1e-13, ("thermistor", d, r["B"])


def test_isobaric_entropy() -> None:
    cv = mp.mpf(R_AIR) / (mp.mpf(GAMMA_AIR) - 1)
    for d in (1e-6, 1e-9, 1e-12, 1.0):
        v2 = 0.1 * (1 + d)
        r = e("ideal_gas", mode="process", process="isobaric", p1=1e5, T1=300.0, V1=0.1, V2=v2)
        # T2 and m as the server computes them in floats: the test isolates the two logarithms
        m = 1e5 * 0.1 / (R_AIR * 300.0)
        t2 = 1e5 * (0.1 / v2) ** 0.0 * v2 / (m * R_AIR)
        ref = mp.mpf(m) * (cv * lnr(t2, 300.0) + mp.mpf(R_AIR) * lnr(v2, 0.1))
        assert rel(r["dS"], ref) < 1e-13, (d, r["dS"])
