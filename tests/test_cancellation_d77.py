"""Second group of formulas that lost digits by cancellation (D77). References with mpmath at 50 digits.

Measured before the fix (relative error): RC and RL charge at t / tau = 1e-8 1.0e-8; RLC step response at small
times 2.2e-8 (underdamped), 3.0e-10 (critical), 5.7e-10 (overdamped), heavily overdamped circuit 8.6e-9 (cancelling
root s1 = -alpha + sqrt(alpha^2 - omega0^2)); lumped heat Q at t / tau = 1.5e-10 1.1e-6, time from a temperature
close to T_0 3.6e-8; decayed fraction at lambda t = 7e-10 3.8e-8; Bateman chain with half-lives equal within 1e-11
1.6e-5 (t_max 2.5e-6); first-order kinetics with c = c0 (1 - 1e-10) 1.0e-10; fin ratio with a = 1e-8, r = 1e6
1.6e-11.
"""

from __future__ import annotations

import math

import mpmath as mp

from oce_math_mcp.tools.eng_hx import _ch_ratio
from tests.conftest import ok

mp.mp.dps = 50


def rel(a: float, b: mp.mpf) -> float:
    return float(abs(mp.mpf(a) - b) / abs(b))


def e(op: str, **params: object) -> dict:
    return ok("eng_calc", operation=op, params=params)


def test_rc_rl_charge_small_times() -> None:
    for t in (1e-12, 1e-8, 1e-4, 1.0, 30.0):
        for circ, key in (("RC", "v_C"), ("RL", "i")):
            extra = {"C": 1.0} if circ == "RC" else {"L": 1.0}
            r = e("circuit_ac_dc", mode="transient", circuit=circ, R=1.0, V=1.0, t=t, **extra)
            ref = -mp.expm1(-mp.mpf(t))
            assert rel(r[key], ref) < 1e-13, (circ, t, r[key], ref)
            d = e(
                "circuit_ac_dc",
                mode="transient",
                circuit=circ,
                R=1.0,
                V=1.0,
                t=t,
                kind="discharge",
                **extra,
            )
            assert rel(d[key], mp.exp(-mp.mpf(t))) < 1e-13, (circ, t)


def _rlc_reference(R: float, t: float) -> tuple[mp.mpf, mp.mpf]:
    """v_C / V and i L / V of the step response from rest, L = C = 1 (omega0 = 1)."""
    a, tt = mp.mpf(R) / 2, mp.mpf(t)
    if a < 1:
        wd = mp.sqrt(1 - a * a)
        c, s = mp.cos(wd * tt), mp.sin(wd * tt) / wd
    elif a == 1:
        c, s = mp.mpf(1), tt
    else:
        q = mp.sqrt(a * a - 1)
        c, s = mp.cosh(q * tt), mp.sinh(q * tt) / q
    ex = mp.exp(-a * tt)
    return 1 - ex * (c + a * s), ex * s


def test_rlc_step_response() -> None:
    cases = [
        (R, t)
        for R in (0.2, 1.999, 2.0, 2.00000002, 3.0, 2e4)
        for t in (1e-6, 1e-4, 0.1, 1.0, 10.0)
    ]
    cases += [(2e4, 1e4), (2e4, 1e-9), (0.2, 100.0)]
    for R, t in cases:
        r = e("circuit_ac_dc", mode="transient", circuit="RLC", R=R, L=1.0, C=1.0, V=1.0, t=t)
        v, i = _rlc_reference(R, t)
        assert rel(r["v_C"], v) < 1e-13, (R, t, r["regime"], r["v_C"], v)
        assert rel(r["i"], i) < 1e-13, (R, t, r["regime"], r["i"], i)


def test_lumped_transient_small_changes() -> None:
    base = {
        "shape": "custom",
        "V": 1e-6,
        "A": 6e-4,
        "rho": 8000.0,
        "cp": 500.0,
        "h": 10.0,
        "k": 50.0,
    }
    base |= {"T_0": 373.15, "T_inf": 293.15}
    vol, area, rho, cp, h = (mp.mpf(base[k]) for k in ("V", "A", "rho", "cp", "h"))
    t0, ti = mp.mpf(373.15), mp.mpf(293.15)
    tau = rho * cp * vol / (h * area)
    for t in (1e-9, 1e-3, 1e3, 1e4):
        r = e("lumped_transient", **base, t=t)
        q = rho * cp * vol * (t0 - ti) * -mp.expm1(-mp.mpf(t) / tau)
        assert rel(r["Q"], q) < 1e-13, (t, r["Q"], q)
    for temp in (373.15 - 1e-9, 373.15 - 1e-3, 330.0, 293.15 + 1e-3, 293.15 + 1e-9):
        r = e("lumped_transient", **base, T=temp)
        ref = -tau * mp.log((mp.mpf(temp) - ti) / (t0 - ti))
        assert rel(r["t"], ref) < 1e-13, (temp, r["t"], ref)


def test_decayed_fraction_small_times() -> None:
    for t in (1e-12, 1e-9, 1e-5, 1.0, 50.0):
        r = e("radioactive_decay", mode="law", T_half=1.0, t=t)
        lam_t = mp.log(2) / mp.mpf(1) * mp.mpf(t)
        assert rel(r["decayed_fraction"], -mp.expm1(-lam_t)) < 1e-13, (t, r["decayed_fraction"])
        assert rel(r["fraction"], mp.exp(-lam_t)) < 1e-13, t


def test_chain_with_close_or_far_half_lives() -> None:
    cases = [(1.0, 1.0 + d, 1.0) for d in (1e-3, 1e-8, 1e-11, 1e-14)]
    cases += [
        (1.0, 1.0 + 9e-13, 100.0),
        (1.0, 1.0, 1.0),
        (1e-3, 1.0, 10.0),
        (1.0, 1e-3, 10.0),
        (1.0, 2.0, 1e-9),
    ]
    for th1, th2, t in cases:
        r = e("radioactive_decay", mode="chain", T_half1=th1, T_half2=th2, N1_0=1.0, t=t)
        l1, l2 = (
            mp.mpf(r["lambda1"]),
            mp.mpf(r["lambda2"]),
        )  # the server's float lambdas: formula error only
        tt = mp.mpf(t)
        n2 = (
            l1 * tt * mp.exp(-l1 * tt)
            if l1 == l2
            else l1 / (l2 - l1) * (mp.exp(-l1 * tt) - mp.exp(-l2 * tt))
        )
        assert rel(r["N2"], n2) < 1e-13, (th1, th2, t, r["N2"], n2)
        if l1 != l2:
            tm = mp.log(l2 / l1) / (l2 - l1)
            assert rel(r["t_max_daughter"], tm) < 1e-13, (th1, th2, r["t_max_daughter"], tm)


def test_kinetics_close_to_initial_concentration() -> None:
    for d in (1e-12, 1e-10, 1e-6, 0.3, 0.999):
        c = 2.0 * (1 - d)
        cm, c0 = mp.mpf(c), mp.mpf(2)
        r1 = e("reaction_kinetics", mode="integrated", order=1, k=0.5, c0=2.0, c=c)
        assert rel(r1["t"], mp.log(c0 / cm) / mp.mpf(0.5)) < 1e-13, (d, r1["t"])
        r2 = e("reaction_kinetics", mode="integrated", order=2, k=0.5, c0=2.0, c=c)
        assert rel(r2["t"], (1 / cm - 1 / c0) / mp.mpf(0.5)) < 1e-13, (d, r2["t"])


def test_fin_ratio_small_argument() -> None:
    for a, b, r in (
        (1e-8, 1.0, 1e6),
        (1e-6, 1.0, 1e3),
        (1e-3, 1.0, 10.0),
        (0.0, 2.0, 5.0),
        (3.0, 40.0, 0.0),
    ):
        am, bm, rm = mp.mpf(a), mp.mpf(b), mp.mpf(r)
        ref = (mp.cosh(am) + rm * mp.sinh(am)) / (mp.cosh(bm) + rm * mp.sinh(bm))
        assert rel(_ch_ratio(a, b, r), ref) < 1e-13, (a, b, r)
    assert math.isfinite(_ch_ratio(700.0, 800.0, 1e3))
