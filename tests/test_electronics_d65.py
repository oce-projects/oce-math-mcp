"""Practical electronics (D65).

Checks independent of the server formulas:
- operational amplifiers: Kirchhoff current law at the inverting node with a finite-gain model
  Vout = A (V+ - V-), solved as a linear system (A = 1e12 for the ideal case, small A for the finite-gain
  formula); integrator and differentiator with complex impedances;
- 555 timer: capacitor charge and discharge integrated with solve_ivp until the 1/3 and 2/3 thresholds;
- LED resistor: Kirchhoff voltage law around the loop;
- thermistor: inverse models solved with brentq, Steinhart-Hart fit against numpy least squares.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from tests.conftest import call, ok


def el(**params: object) -> dict:
    return ok("eng_calc", operation="electronics", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="electronics", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def solve_root(fun: Callable[[float], float], a: float, b: float) -> float:
    return cast(float, brentq(fun, a, b, xtol=1e-15))


# ---------------------------------------------------------------- operational amplifiers


def _kcl(
    gain: complex, rows: Sequence[tuple[float, complex]], r_f: complex, v_plus: float
) -> complex:
    """Vout of an op amp: inputs (V_k, Z_k) into the inverting node, feedback Z_f, V+ given, Vout = A (V+ - V-).
    Unknowns [V-, Vout]."""
    a = np.zeros((2, 2), dtype=complex)
    b = np.zeros(2, dtype=complex)
    for v_k, z_k in rows:  # sum (V_k - V-)/Z_k + (Vout - V-)/Z_f = 0
        a[0, 0] -= 1 / z_k
        b[0] -= v_k / z_k
    a[0, 0] -= 1 / r_f
    a[0, 1] += 1 / r_f
    a[1] = [gain, 1.0]  # Vout + A V- = A V+
    b[1] = gain * v_plus
    return complex(np.linalg.solve(a, b)[1])


def test_inverting_and_non_inverting_against_kcl() -> None:
    for r_in, r_f, v in ((1e3, 10e3, 0.25), (4.7e3, 47e3, -1.2), (10e3, 2.2e3, 3.0)):
        r = el(mode="opamp", config="inverting", R_in=r_in, R_f=r_f, V_in=v)
        assert rel(r["V_out"], _kcl(1e12, [(v, r_in)], r_f, 0.0).real, 1e-9), r
        assert rel(r["gain"], -r_f / r_in, 1e-13) and rel(r["R_input"], r_in, 1e-13)
        fin = el(mode="opamp", config="inverting", R_in=r_in, R_f=r_f, V_in=v, A=2e3)
        assert rel(fin["V_out_finite_A"], _kcl(2e3, [(v, r_in)], r_f, 0.0).real, 1e-12), fin
        # non-inverting: R_g from the inverting node to ground, the input on V+
        r = el(mode="opamp", config="non_inverting", R_g=r_in, R_f=r_f, V_in=v)
        assert rel(r["V_out"], _kcl(1e12, [(0.0, r_in)], r_f, v).real, 1e-9), r
        assert rel(r["gain"], 1 + r_f / r_in, 1e-13)
        fin = el(mode="opamp", config="non_inverting", R_g=r_in, R_f=r_f, V_in=v, A=500.0)
        assert rel(fin["V_out_finite_A"], _kcl(500.0, [(0.0, r_in)], r_f, v).real, 1e-12), fin


def test_bandwidth_from_gain_bandwidth_product() -> None:
    r = el(mode="opamp", config="inverting", R_in=1e3, R_f=99e3, GBW=1e6)
    assert rel(r["noise_gain"], 100.0, 1e-13) and rel(
        r["bandwidth"], 1e4, 1e-13
    )  # GBW / noise gain
    r = el(mode="opamp", config="non_inverting", R_g=1e3, R_f=9e3, GBW=10e6)
    assert rel(r["bandwidth"], 1e6, 1e-13)


def test_summing_against_kcl() -> None:
    inputs = [[0.5, 10e3], [-0.2, 20e3], [1.0, 5e3]]
    r = el(mode="opamp", config="summing", R_f=10e3, inputs=inputs)
    rows = [(float(v), float(z)) for v, z in inputs]
    assert rel(r["V_out"], _kcl(1e12, rows, 10e3, 0.0).real, 1e-9), r


def test_difference_amplifier_against_kcl() -> None:
    for r1, rf, r2, rg, v1, v2 in (
        (10e3, 47e3, 10e3, 47e3, 1.2, 1.5),
        (1e3, 2e3, 3.3e3, 5.6e3, -0.4, 0.9),
    ):
        r = el(mode="opamp", config="difference", R_1=r1, R_f=rf, R_2=r2, R_g=rg, V_1=v1, V_2=v2)
        v_plus = v2 * rg / (r2 + rg)  # divider R_2 - R_g on the non-inverting input
        assert rel(r["V_out"], _kcl(1e12, [(v1, r1)], rf, v_plus).real, 1e-9), r
        cm = _kcl(1e12, [(1.0, r1)], rf, rg / (r2 + rg)).real  # V_1 = V_2 = 1 V
        assert abs(r["common_mode_gain"] - cm) < 1e-9
    bal = el(
        mode="opamp", config="difference", R_1=10e3, R_f=47e3, R_2=10e3, R_g=47e3, V_1=1.2, V_2=1.5
    )
    assert bal["balanced"] is True and rel(bal["V_out"], 4.7 * 0.3, 1e-12)


def test_bandwidth_against_single_pole_op_amp() -> None:
    """A(f) = A0 / (1 + j f / f_p) with A0 f_p = GBW: the -3 dB frequency of the closed loop is GBW / noise gain."""
    gbw, a0 = 1e6, 1e9
    for config, keys, rows, r_f, v_plus in (
        ("inverting", {"R_in": 1e3}, [(1.0, 1e3)], 99e3, 0.0),
        ("non_inverting", {"R_g": 1e3}, [(0.0, 1e3)], 9e3, 1.0),
    ):
        r = el(mode="opamp", config=config, R_f=r_f, GBW=gbw, **keys)
        dc = abs(_kcl(a0, rows, r_f, v_plus))

        def drop(f: float, rows=rows, r_f=r_f, v_plus=v_plus, dc=dc) -> float:
            a = a0 / (1 + 1j * f * a0 / gbw)
            return abs(_kcl(a, rows, r_f, v_plus)) - dc / math.sqrt(2)

        f3 = solve_root(drop, 1.0, gbw)
        assert rel(r["bandwidth"], f3, 1e-6), (r, f3)


def test_wheatstone_against_nodal_analysis() -> None:
    def nodes(
        r1: float, r2: float, r3: float, rx: float, vs: float, rg: float
    ) -> tuple[float, float]:
        # top at V_s; R_1 top-D, R_2 D-ground, R_3 top-B, R_x B-ground; galvanometer R_G between D and B
        a = np.array([[-1 / r1 - 1 / r2 - 1 / rg, 1 / rg], [1 / rg, -1 / r3 - 1 / rx - 1 / rg]])
        vd, vb = np.linalg.solve(a, [-vs / r1, -vs / r3])
        return float(vd - vb), float((vd - vb) / rg)

    for r1, r2, r3, rx in (
        (100.0, 120.0, 330.0, 390.0),
        (1e3, 1e3, 1e3, 1010.0),
        (2.2e3, 470.0, 10e3, 1.8e3),
    ):
        r = el(mode="wheatstone", R_1=r1, R_2=r2, R_3=r3, R_x=rx, V_s=10.0, R_G=50.0)
        v_open, _ = nodes(r1, r2, r3, rx, 10.0, 1e15)
        _, i_g = nodes(r1, r2, r3, rx, 10.0, 50.0)
        assert abs(r["V_G"] - v_open) < 1e-9 and rel(r["I_G"], i_g, 1e-9), (r, v_open, i_g)
        back = el(mode="wheatstone", R_1=r1, R_2=r2, R_3=r3, V_G=r["V_G"], V_s=10.0)
        assert rel(back["R_x"], rx, 1e-12)
    bal = el(mode="wheatstone", R_1=100.0, R_2=150.0, R_3=220.0)
    assert rel(bal["R_x_balance"], 330.0, 1e-13)
    v0, _ = nodes(100.0, 150.0, 220.0, 330.0, 5.0, 1e15)
    assert abs(v0) < 1e-12


def test_integrator_and_differentiator_with_complex_impedances() -> None:
    res, cap, f = 10e3, 100e-9, 250.0
    z_c = 1 / (2j * math.pi * f * cap)
    r = el(mode="opamp", config="integrator", R=res, C=cap, f=f)
    g = _kcl(1e12, [(1.0, res)], z_c, 0.0)
    assert (
        rel(r["gain_magnitude"], abs(g), 1e-9)
        and abs(r["phase_deg"] - math.degrees(np.angle(g))) < 1e-6
    )
    assert rel(r["tau"], res * cap, 1e-13) and rel(
        r["f_unity"], 1 / (2 * math.pi * res * cap), 1e-13
    )
    r = el(mode="opamp", config="differentiator", R=res, C=cap, f=f)
    g = _kcl(1e12, [(1.0, z_c)], res, 0.0)
    assert (
        rel(r["gain_magnitude"], abs(g), 1e-9)
        and abs(r["phase_deg"] - math.degrees(np.angle(g))) < 1e-6
    )


# ---------------------------------------------------------------- 555 timer


def _charge_time(v0: float, v1: float, target: float, tau: float) -> float:
    """Time for dV/dt = (target - V)/tau to go from v0 to v1, integrated numerically."""

    def hit(_t: float, y: np.ndarray) -> float:
        return float(y[0] - v1)

    hit.terminal = True  # type: ignore[attr-defined]
    sol = solve_ivp(
        lambda _t, y: [(target - y[0]) / tau],
        (0, 20 * tau),
        [v0],
        events=hit,
        rtol=1e-12,
        atol=1e-14,
    )
    return float(sol.t_events[0][0])


def test_555_astable_and_monostable_against_capacitor_ode() -> None:
    vcc = 9.0
    for r1, r2, c in ((1e3, 10e3, 10e-6), (4.7e3, 4.7e3, 100e-9), (68e3, 1e3, 1e-6)):
        r = el(mode="timer_555", circuit="astable", R1=r1, R2=r2, C=c)
        t_high = _charge_time(vcc / 3, 2 * vcc / 3, vcc, (r1 + r2) * c)
        t_low = _charge_time(2 * vcc / 3, vcc / 3, 0.0, r2 * c)
        assert rel(r["t_high"], t_high, 1e-8) and rel(r["t_low"], t_low, 1e-8), (r, t_high, t_low)
        assert rel(r["f"], 1 / (t_high + t_low), 1e-8)
        assert rel(r["duty"], t_high / (t_high + t_low), 1e-8)
    m = el(mode="timer_555", circuit="monostable", R=100e3, C=10e-6)
    assert rel(m["t_pulse"], _charge_time(0.0, 2 * vcc / 3, vcc, 1.0), 1e-8)  # tau = R C = 1 s
    # tables of the page "555 timer IC": R1 8.2 k, R2 68 k, 10 uF -> 1 Hz +0.048%, duty 52.8%; 91 k, 10 uF -> 1 s -0.026%
    a = el(mode="timer_555", circuit="astable", R1=8.2e3, R2=68e3, C=10e-6)
    assert round((a["f"] - 1) * 100, 3) == 0.048 and round(a["duty"] * 100, 1) == 52.8
    assert (
        round((el(mode="timer_555", circuit="monostable", R=91e3, C=10e-6)["t_pulse"] - 1) * 100, 3)
        == -0.026
    )


# ---------------------------------------------------------------- LED


def test_led_resistor_against_kirchhoff() -> None:
    for v_s, v_f, i, n in ((5.0, 2.0, 0.02, 1), (12.0, 3.1, 0.015, 3), (24.0, 2.1, 0.01, 8)):
        r = el(mode="led_resistor", V_s=v_s, V_f=v_f, I=i, n=n)
        assert abs(v_s - n * v_f - i * r["R"]) < 1e-12 * v_s  # loop: V_s = n V_f + I R
        assert rel(r["P_R"], i * i * r["R"], 1e-13) and rel(r["P_led"], n * v_f * i, 1e-13)
        assert rel(r["efficiency"], n * v_f / v_s, 1e-13)
    assert "V_f" in bad({"mode": "led_resistor", "V_s": 5.0, "V_f": 2.0, "I": 0.02, "n": 3})
    # examples of the page "LED circuit" (V_switch = 0): 2040, 1440, 3200 and 1500 ohm
    for v_s, i, expect in (
        (12.0, 0.005, 2040.0),
        (9.0, 0.005, 1440.0),
        (5.0, 0.001, 3200.0),
        (3.3, 0.001, 1500.0),
    ):
        assert rel(el(mode="led_resistor", V_s=v_s, V_f=1.8, I=i)["R"], expect, 1e-12)
    sw = el(mode="led_resistor", V_s=12.0, V_f=1.8, I=0.005, V_switch=0.2)  # BJT switch V_CE(sat)
    assert rel(sw["R"], 2000.0, 1e-12)


# ---------------------------------------------------------------- thermistor


def test_thermistor_beta_model_both_ways() -> None:
    r0, t0, beta = 10e3, 298.15, 3950.0
    r = el(mode="thermistor", model="beta", R0=r0, T0=t0, B=beta, T=[50, "C"])
    t = 323.15
    assert rel(r["R"], r0 * math.exp(beta * (1 / t - 1 / t0)), 1e-13)
    back = el(mode="thermistor", model="beta", R0=r0, T0=t0, B=beta, R=r["R"])
    assert abs(back["T"] - t) < 1e-9 and abs(back["T_C"] - 50.0) < 1e-9
    found = solve_root(lambda tk: r0 * math.exp(beta * (1 / tk - 1 / t0)) - 2500.0, 250.0, 450.0)
    assert (
        abs(el(mode="thermistor", model="beta", R0=r0, T0=t0, B=beta, R=2500.0)["T"] - found) < 1e-9
    )


def test_steinhart_hart_both_ways_and_fit() -> None:
    a, b, c = 1.129148e-3, 2.34125e-4, 8.76741e-8
    for res in (1e3, 10e3, 100e3):
        r = el(mode="thermistor", model="steinhart_hart", a=a, b=b, c=c, R=res)
        lr = math.log(res)
        assert rel(r["T"], 1 / (a + b * lr + c * lr**3), 1e-13)
        back = el(mode="thermistor", model="steinhart_hart", a=a, b=b, c=c, T=r["T"])
        assert rel(back["R"], res, 1e-10), (back, res)
    pts = [[273.15, 32650.0], [298.15, 10000.0], [358.15, 1200.0]]
    fit = el(mode="thermistor", model="fit", points=pts)
    m = np.array([[1, math.log(rr), math.log(rr) ** 3] for _, rr in pts])
    ref = np.linalg.lstsq(m, [1 / tt for tt, _ in pts], rcond=None)[0]
    for key, val in zip(("a", "b", "c"), ref, strict=True):
        assert rel(fit[key], float(val), 1e-9), (fit, ref)
    page = el(mode="thermistor", model="steinhart_hart", a=1.40e-3, b=2.37e-4, c=9.90e-8, R=3000.0)
    assert (
        abs(page["T_C"] - 25.0) < 1.0
    )  # page "Thermistor": 3 kOhm at 25 C (coefficients with 3 digits)
    two = el(mode="thermistor", model="fit", points=pts[:2])
    assert rel(two["B"], math.log(32650.0 / 10000.0) / (1 / 273.15 - 1 / 298.15), 1e-12)
