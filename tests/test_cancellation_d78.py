"""Third group of formulas that lost digits by cancellation (D78). References with mpmath at 50 digits.

Measured before the fix (relative error): relativistic kinetic energy (gamma - 1) m c^2 at beta = 1e-6 8.9e-5, from a
small momentum (E - m c^2) 9.5e-6, de Broglie kinetic energy of an electron at v = 1000 m/s 1.4e-5; round trip of a
kinetic energy of 1e-19 J through gamma = 1 + K / (m c^2) 7.4e-11; Compton electron energy E - E' at 0.1 degrees
3.3e-9; Doppler z = r - 1 at beta = 1e-9 8.2e-8 and beta from f / f_observed = 1 + 1e-9 8.1e-8; damped frequency
sqrt(1 - zeta^2) at zeta = 1 - 1e-8 2.8e-10.
"""

from __future__ import annotations

import mpmath as mp

from oce_math_mcp.tools.eng_modern import E_CH as ECH_F
from oce_math_mcp.tools.eng_modern import M_E as ME_F
from oce_math_mcp.tools.eng_modern import C as C_F
from tests.conftest import ok

mp.mp.dps = 50
C, ECH, M_E = mp.mpf(C_F), mp.mpf(ECH_F), mp.mpf(ME_F)  # the server's constants: formula error only
E0 = M_E * C**2


def rel(a: float, b: mp.mpf) -> float:
    return float(abs(mp.mpf(a) - b) / abs(b))


def e(op: str, **params: object) -> dict:
    return ok("eng_calc", operation=op, params=params)


def test_compton_electron_energy() -> None:
    for deg in (0.01, 0.1, 1.0, 90.0, 180.0):
        r = e("modern_physics", mode="compton", energy=1e-15, angle_deg=deg)
        x = mp.mpf(1e-15) / E0 * 2 * mp.sin(mp.radians(mp.mpf(deg)) / 2) ** 2
        ref = mp.mpf(1e-15) * x / (1 + x) / ECH
        assert rel(r["electron_kinetic_eV"], ref) < 1e-13, (deg, r["electron_kinetic_eV"], ref)


def test_de_broglie_kinetic_energy() -> None:
    for v in (1e3, 1e5, 1e7, 2.9e8):
        r = e("modern_physics", mode="de_broglie", particle="electron", v=v)
        b = mp.mpf(v) / C
        ref = (1 / mp.sqrt(1 - b * b) - 1) * E0 / ECH
        assert rel(r["kinetic_energy_eV"], ref) < 1e-13, (v, r["kinetic_energy_eV"], ref)
    for pm in (1e-27, 1e-24, 1e-18):
        r = e("modern_physics", mode="de_broglie", particle="electron", p=pm)
        pc = mp.mpf(pm) * C
        ref = (mp.sqrt(pc**2 + E0**2) - E0) / ECH
        assert rel(r["kinetic_energy_eV"], ref) < 1e-13, (pm, r["kinetic_energy_eV"], ref)


def test_relativistic_kinematics_at_low_speed() -> None:
    def check(r: dict, gamma: mp.mpf, case: object) -> None:
        beta = mp.sqrt(1 - 1 / gamma**2)
        assert rel(r["K_J"], (gamma - 1) * E0) < 1e-13, (case, r["K_J"])
        assert rel(r["K_MeV"], (gamma - 1) * E0 / (10**6 * ECH)) < 1e-13, (case, r["K_MeV"])
        assert rel(r["beta"], beta) < 1e-13, (case, r["beta"], beta)
        assert rel(r["gamma"], gamma) < 1e-13, (case, r["gamma"])

    for b in (1e-6, 1e-4, 0.5, 0.999):
        r = e("special_relativity", mode="kinematics", particle="electron", beta=b)
        check(r, 1 / mp.sqrt(1 - mp.mpf(b) ** 2), ("beta", b))
    for k in (1e-19, 1e-16, 1e-10):
        r = e("special_relativity", mode="kinematics", particle="electron", kinetic_energy=k)
        check(r, 1 + mp.mpf(k) / E0, ("K", k))
    for pm in (1e-27, 1e-22, 1e-18):
        r = e("special_relativity", mode="kinematics", particle="electron", p=pm)
        check(r, mp.sqrt((mp.mpf(pm) * C) ** 2 + E0**2) / E0, ("p", pm))
    r = e("special_relativity", mode="kinematics", particle="electron", gamma=1 + 1e-10)
    check(r, mp.mpf(1 + 1e-10), ("gamma", 1 + 1e-10))
    e0f = (
        ME_F * C_F**2
    )  # the server's rest energy (a float): the test isolates the subtraction E - m c^2
    for d in (1e-9, 1e-3, 1.0):
        etot = e0f * (1 + d)
        r = e("special_relativity", mode="kinematics", particle="electron", total_energy=etot)
        em, e0m = mp.mpf(etot), mp.mpf(e0f)
        assert rel(r["K_J"], em - e0m) < 1e-13, (d, r["K_J"])
        assert rel(r["beta"], mp.sqrt(1 - (e0m / em) ** 2)) < 1e-13, (d, r["beta"])


def test_doppler_at_low_speed() -> None:
    for b in (1e-9, 1e-6, -1e-6, 0.5, -0.9):
        r = e("special_relativity", mode="doppler", beta=b, f=1e9)
        bm = mp.mpf(b)
        assert rel(r["z"], mp.sqrt((1 + bm) / (1 - bm)) - 1) < 1e-13, (b, r["z"])
    for d in (1e-9, 1e-6, -1e-6, 0.3):
        fo = 1e9 * (1 - d)
        r = e("special_relativity", mode="doppler", f=1e9, f_observed=fo)
        rr = mp.mpf(1e9) / mp.mpf(fo)
        assert rel(r["beta"], (rr**2 - 1) / (rr**2 + 1)) < 1e-13, (d, r["beta"])
        assert rel(r["z"], rr - 1) < 1e-13, (d, r["z"])
        lo = 5e-7 * (1 + d)
        r = e("special_relativity", mode="doppler", wavelength=5e-7, wavelength_observed=lo)
        rr = mp.mpf(lo) / mp.mpf(5e-7)
        assert rel(r["beta"], (rr**2 - 1) / (rr**2 + 1)) < 1e-13, (d, r["beta"])
    for z in (1e-9, 2.0):
        r = e("special_relativity", mode="doppler", z=z)
        rr = 1 + mp.mpf(z)
        assert rel(r["beta"], (rr**2 - 1) / (rr**2 + 1)) < 1e-13, (z, r["beta"])


def test_damping_close_to_critical() -> None:
    for z in (1 - 1e-8, 1 - 1e-4, 0.5, 0.01):
        zm = mp.mpf(z)
        sq = mp.sqrt(1 - zm * zm)
        r = e("control_system", mode="second_order", zeta=z, wn=1.0)
        assert rel(r["wd"], sq) < 1e-13, (z, r["wd"])
        assert rel(r["tp"], mp.pi / sq) < 1e-13, (z, r["tp"])
        if (
            z < 0.9999999
        ):  # at 1 - 1e-8 the true Mp is about exp(-22214): below the smallest float, 0 is right
            # Mp = 100 exp(-pi zeta / sqrt(1 - zeta^2)): the exponent (222 at 1 - 1e-4) multiplies the rounding
            assert rel(r["Mp_pct"], 100 * mp.exp(-mp.pi * zm / sq)) < 1e-12, (z, r["Mp_pct"])
        r = e("vibration_1dof", m=1.0, k=1.0, zeta=z)
        assert rel(r["omega_d"], sq) < 1e-13, (z, r["omega_d"])
        assert rel(r["log_decrement"], 2 * mp.pi * zm / sq) < 1e-13, (z, r["log_decrement"])
