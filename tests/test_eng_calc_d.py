"""eng_calc area D (D39): circuits, filters, Clarke-Park, controls, optics and acoustics.

Independent methods: filters and impedances with numpy complex numbers (voltage divider);
second-order system simulated with solve_ivp; margins on cases with a closed-form solution;
Clarke-Park with balanced three-phase currents and power invariance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tests.conftest import call, ok


def e(operation: str, **params: object) -> dict:
    return ok("eng_calc", operation=operation, params=params)


def err(operation: str, **params: object) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert resp["ok"] is False
    return resp["error"]["code"]


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------------------------------------------------------ circuit_ac_dc


@pytest.mark.parametrize(
    "given",
    [
        {"V": 12, "I": 0.5},
        {"V": 12, "R": 24},
        {"I": 0.5, "R": 24},
        {"P": 6, "V": 12},
        {"P": 6, "I": 0.5},
        {"P": 6, "R": 24},
    ],
)
def test_ohm_any_two(given: dict) -> None:
    r = e("circuit_ac_dc", mode="ohm", **given)
    assert rel(r["V"], 12) and rel(r["I"], 0.5) and rel(r["R"], 24) and rel(r["P"], 6)


def test_resistors_divider_ac() -> None:
    assert rel(
        e("circuit_ac_dc", mode="resistors", values=[100, 220, 330], connection="series")["R_eq"],
        650,
    )
    par = e("circuit_ac_dc", mode="resistors", values=[100, 100, 50], connection="parallel")["R_eq"]
    assert rel(par, 25)
    d = e("circuit_ac_dc", mode="divider", Vin=[5, "V"], R1=[10, "kohm"], R2=[4.7, "kohm"])
    assert rel(d["Vout"], 5 * 4.7 / 14.7) and rel(d["I"], 5 / 14700)
    s = e("circuit_ac_dc", mode="ac_single", V=230, I=10, cos_phi=0.8)
    assert rel(s["S"], 2300) and rel(s["P"], 1840) and rel(s["Q"], 1380)
    t = e("circuit_ac_dc", mode="ac_three", V=[400, "V"], P=[15, "kW"], cos_phi=0.85)
    assert rel(t["I"], 15000 / (math.sqrt(3) * 400 * 0.85))


def test_cable_drop_copper_iec() -> None:
    r = e(
        "circuit_ac_dc",
        mode="cable_drop",
        phases="single",
        I=16,
        L=[25, "m"],
        A_section=[2.5, "mm2"],
        material="copper",
        V=230,
    )
    rho = 1 / 58 * 1e-6  # IEC 60028: annealed copper at 20 degrees, 1/58 ohm mm2/m
    assert rel(r["dV"], 2 * rho * 25 * 16 / 2.5e-6) and rel(r["dV_pct"], 100 * r["dV"] / 230)
    assert (
        err("circuit_ac_dc", mode="cable_drop", phases="single", I=16, L=25, A_section=2.5e-6)
        == "INVALID_ARGUMENT"
    )


# ------------------------------------------------------------------ rlc_filter


def divider(kind: str, R: float, L: float, C: float, f: float) -> complex:
    w = 2 * math.pi * f
    zR, zL, zC = complex(R), 1j * w * L, 1 / (1j * w * C) if C else 0j
    return {
        "rc_lowpass": zC / (zR + zC),
        "rc_highpass": zR / (zR + zC),
        "rl_lowpass": zR / (zR + zL),
        "rl_highpass": zL / (zR + zL),
        "rlc_bandpass": zR / (zR + zL + zC),
    }[kind]


@pytest.mark.parametrize(
    "kind", ["rc_lowpass", "rc_highpass", "rl_lowpass", "rl_highpass", "rlc_bandpass"]
)
def test_filters_against_complex_divider(kind: str) -> None:
    R, L, C, f = 1000.0, 0.1, 1e-7, 1234.5
    params: dict[str, object] = {"type": kind, "R": R, "f": f}
    if "l" in kind.split("_")[0]:
        params["L"] = L
    if "c" in kind.split("_")[0]:
        params["C"] = C
    r = e("rlc_filter", **params)
    H = divider(kind, R, L if "L" in params else 0.0, C if "C" in params else 0.0, f)
    assert rel(r["gain"], abs(H), 1e-12) and rel(r["phase_deg"], math.degrees(np.angle(H)), 1e-9)
    assert rel(r["gain_dB"], 20 * math.log10(abs(H)), 1e-9)


def test_filter_cutoff_is_minus_3db_and_rlc_q() -> None:
    fc = e("rlc_filter", type="rc_lowpass", R=[10, "kohm"], C=[100, "nF"])["fc"]
    assert rel(abs(divider("rc_lowpass", 1e4, 0, 1e-7, fc)), 1 / math.sqrt(2), 1e-12)
    r = e("rlc_filter", type="rlc_bandpass", R=50, L=[10, "mH"], C=[1, "uF"])
    assert rel(r["f0"], 1 / (2 * math.pi * math.sqrt(0.01 * 1e-6))) and rel(
        r["Q"], math.sqrt(0.01 / 1e-6) / 50
    )
    assert rel(r["bandwidth"], r["f0"] / r["Q"])
    assert rel(abs(divider("rlc_bandpass", 50, 0.01, 1e-6, r["f0"])), 1.0, 1e-12)
    assert err("rlc_filter", type="rc_lowpass", R=1, L=1) == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ foc_clarke_park


def test_clarke_park_balanced_and_round_trip() -> None:
    amp, th = 7.0, 35.0
    t = math.radians(th)
    ia, ib, ic = (amp * math.cos(t - k * 2 * math.pi / 3) for k in range(3))
    r = e("foc_clarke_park", mode="forward", ia=ia, ib=ib, ic=ic, theta_deg=th)
    assert rel(r["d"], amp, 1e-12) and abs(r["q"]) < 1e-12 and abs(r["zero"]) < 1e-12
    assert rel(math.hypot(r["alpha"], r["beta"]), amp, 1e-12)
    back = e("foc_clarke_park", mode="inverse", d=r["d"], q=2.5, theta_deg=th)
    fwd = e(
        "foc_clarke_park", mode="forward", ia=back["a"], ib=back["b"], ic=back["c"], theta_deg=th
    )
    assert rel(fwd["d"], amp, 1e-12) and rel(fwd["q"], 2.5, 1e-12)
    two = e("foc_clarke_park", mode="forward", ia=ia, ib=ib, theta_deg=th)
    assert rel(two["d"], amp, 1e-12)


def test_power_invariant_scaling_keeps_power() -> None:
    va, vb, vc = 100.0, -30.0, -70.0
    ia, ib, ic = 4.0, 1.0, -5.0
    v = e("foc_clarke_park", mode="forward", ia=va, ib=vb, ic=vc, theta_deg=20, scaling="power")
    i = e("foc_clarke_park", mode="forward", ia=ia, ib=ib, ic=ic, theta_deg=20, scaling="power")
    assert rel(
        v["d"] * i["d"] + v["q"] * i["q"] + v["zero"] * i["zero"],
        va * ia + vb * ib + vc * ic,
        1e-12,
    )


# ------------------------------------------------------------------ control_system


def simulated_step(zeta: float, wn: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        lambda _t, y: [y[1], wn * wn * (1 - y[0]) - 2 * zeta * wn * y[1]],
        (0, t_end),
        [0.0, 0.0],
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )
    t = np.linspace(0, t_end, 400001)
    return t, sol.sol(t)[0]


@pytest.mark.parametrize(("zeta", "wn"), [(0.2, 5.0), (0.6, 2.0), (1.0, 3.0), (1.8, 1.0)])
def test_second_order_against_simulation(zeta: float, wn: float) -> None:
    r = e("control_system", mode="second_order", zeta=zeta, wn=wn)
    t, y = simulated_step(zeta, wn, 40 / (zeta * wn))
    assert abs(r["Mp_pct"] - max(0.0, 100 * (y.max() - 1))) < 1e-4
    outside = np.nonzero(np.abs(y - 1) > 0.02)[0]
    assert abs(r["ts_2pct"] - t[outside[-1] + 1]) < 2 * (t[1] - t[0])
    tr = t[np.argmax(y >= 0.9)] - t[np.argmax(y >= 0.1)]
    assert abs(r["tr_10_90"] - tr) < 2 * (t[1] - t[0])
    if zeta < 1:
        assert rel(r["Mp_pct"], 100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta * zeta)), 1e-9)


def test_margins_closed_form_cases() -> None:
    r = e("control_system", mode="margins", num=[1], den=[1, 1, 0])  # L = 1 / (s (s + 1))
    w = math.sqrt((math.sqrt(5) - 1) / 2)
    assert rel(r["wc_gain"], w, 1e-9) and rel(
        r["phase_margin_deg"], 90 - math.degrees(math.atan(w)), 1e-9
    )
    assert r["gain_margin_dB"] is None
    r2 = e(
        "control_system", mode="margins", num=[2], den=[1, 3, 2, 0]
    )  # L = 2 / (s (s + 1) (s + 2))
    assert rel(r2["wc_phase"], math.sqrt(2), 1e-9) and rel(
        r2["gain_margin_dB"], 20 * math.log10(3), 1e-9
    )


def test_ziegler_nichols_tables() -> None:
    u = e("control_system", mode="pid_zn", method="ultimate", Ku=10, Tu=2, controller="PID")
    assert (
        (u["Kp"], u["Ti"], u["Td"]) == (6.0, 1.0, 0.25) and rel(u["Ki"], 6.0) and rel(u["Kd"], 1.5)
    )
    s = e("control_system", mode="pid_zn", method="step", K=2, L=0.5, T=4, controller="PI")
    assert rel(s["Kp"], 0.9 * 4 / (2 * 0.5)) and rel(s["Ti"], 0.5 / 0.3) and s["Td"] == 0


# ------------------------------------------------------------------ optics_acoustics


def test_snell_lens_levels() -> None:
    r = e("optics_acoustics", mode="snell", n1=1.5, n2=1.0, theta1_deg=30)
    assert rel(1.5 * math.sin(math.radians(30)), math.sin(math.radians(r["theta2_deg"])), 1e-12)
    assert rel(r["critical_angle_deg"], math.degrees(math.asin(1 / 1.5)))
    tir = e("optics_acoustics", mode="snell", n1=1.5, n2=1.0, theta1_deg=60)
    assert tir["total_internal_reflection"] is True and tir["theta2_deg"] is None
    lens = e("optics_acoustics", mode="thin_lens", f=[50, "mm"], d_o=[200, "mm"])
    assert rel(1 / 0.05, 1 / 0.2 + 1 / lens["d_i"]) and rel(
        lens["magnification"], -lens["d_i"] / 0.2
    )
    assert rel(e("optics_acoustics", mode="spl", p=[1, "Pa"])["SPL_dB"], 20 * math.log10(1 / 2e-5))
    tot = e("optics_acoustics", mode="sum_levels", levels=[80, 80])["L_total_dB"]
    assert rel(tot, 80 + 10 * math.log10(2))
    far = e("optics_acoustics", mode="inverse_square", L1=90, r1=1, r2=[10, "m"])
    assert rel(far["L2_dB"], 70.0)
    assert rel(e("optics_acoustics", mode="db", ratio=100, kind="power")["dB"], 20.0)
    assert rel(e("optics_acoustics", mode="db", dB=6, kind="amplitude")["ratio"], 10 ** (6 / 20))
