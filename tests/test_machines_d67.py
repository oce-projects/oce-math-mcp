"""Electric machines (D67).

Checks independent of the server formulas:
- DC motor: the electrical and mechanical differential equations (L di/dt = V - R i - k w, J dw/dt = k i - T_load)
  integrated with solve_ivp to the steady state; maximum power found numerically;
- transformer: EMF from the numerical derivative of the flux N A B sin(w t) and its RMS value; regulation and
  efficiency from the equivalent circuit solved with complex phasors; maximum efficiency found numerically;
- induction motor: the per-phase equivalent circuit solved with complex phasors (air-gap power, torque),
  maximum torque found numerically;
- synchronous machine: power from the phasors V and E with the synchronous reactance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from tests.conftest import call, ok


def em(**params: object) -> dict:
    return ok("eng_calc", operation="electric_machine", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="electric_machine", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def test_dc_motor_against_differential_equations() -> None:
    v, r_a, k, ind, j = 24.0, 0.8, 0.05, 2e-3, 1e-4
    for t_load in (0.0, 0.1, 0.5):
        r = em(mode="dc_motor", V_m=v, R_m=r_a, k=k, T_load=t_load)
        sol = solve_ivp(
            lambda _t, y: [(v - r_a * y[0] - k * y[1]) / ind, (k * y[0] - t_load) / j],
            (0, 5.0),
            [0.0, 0.0],
            method="LSODA",
            rtol=1e-11,
            atol=1e-12,
        )
        i_ss, w_ss = sol.y[0][-1], sol.y[1][-1]
        assert rel(r["omega"], w_ss, 1e-7) and abs(r["I_a"] - i_ss) < 1e-7, (r, i_ss, w_ss)
        assert rel(r["n_rpm"], w_ss * 60 / (2 * math.pi), 1e-7)
    r = em(mode="dc_motor", V_m=v, R_m=r_a, k=k)
    assert rel(r["omega_0"], v / k, 1e-13) and rel(r["T_stall"], k * v / r_a, 1e-13)
    best: Any = minimize_scalar(
        lambda t: -t * (v / k - r_a * t / k**2),
        bounds=(0, k * v / r_a),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert rel(r["P_max"], -float(best.fun), 1e-9) and rel(r["T_at_P_max"], float(best.x), 1e-6)


def _induction(
    v_ph: float,
    f: float,
    poles: int,
    r1: float,
    x1: float,
    r2: float,
    x2: float,
    xm: float,
    s: float,
):
    """Per-phase equivalent circuit (stator R1 + jX1, magnetising jXm, rotor R2/s + jX2) solved with phasors."""
    z_rot = r2 / s + 1j * x2
    z_par = 1 / (1 / (1j * xm) + 1 / z_rot)
    i1 = v_ph / (r1 + 1j * x1 + z_par)
    i2 = i1 * z_par / z_rot
    p_ag = 3 * abs(i2) ** 2 * r2 / s
    w_s = 2 * math.pi * f / (poles / 2)
    return p_ag, p_ag / w_s, i1


def test_induction_motor_against_equivalent_circuit() -> None:
    v_ll, f, poles = 400.0, 50.0, 4
    c = {"R_s": 0.5, "X_s": 1.2, "R_r": 0.4, "X_r": 1.2, "X_m": 40.0}
    for s in (0.01, 0.04, 0.2, 1.0):
        r = em(mode="induction_motor", V=v_ll, f=f, poles=poles, s=s, **c)
        p_ag, torque, i1 = _induction(
            v_ll / math.sqrt(3), f, poles, c["R_s"], c["X_s"], c["R_r"], c["X_r"], c["X_m"], s
        )
        assert rel(r["P_ag"], p_ag, 1e-12) and rel(r["T"], torque, 1e-12), (r, p_ag, torque)
        assert rel(r["P_cu_rotor"], s * p_ag, 1e-12) and rel(r["P_mech"], (1 - s) * p_ag, 1e-12)
        assert rel(r["I_stator"], abs(i1), 1e-12)
    r = em(mode="induction_motor", V=v_ll, f=f, poles=poles, s=0.03, **c)
    assert rel(r["n_s_rpm"], 1500.0, 1e-15) and rel(r["n_rpm"], 1455.0, 1e-13)
    assert rel(r["f_rotor"], 1.5, 1e-13)
    best: Any = minimize_scalar(
        lambda s: (
            -_induction(
                v_ll / math.sqrt(3), f, poles, c["R_s"], c["X_s"], c["R_r"], c["X_r"], c["X_m"], s
            )[1]
        ),
        bounds=(1e-4, 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert rel(r["s_T_max"], float(best.x), 1e-6) and rel(r["T_max"], -float(best.fun), 1e-9), (
        r,
        best,
    )


def test_synchronous_machine_against_phasors() -> None:
    v_ph, e_ph, x_s = 400 / math.sqrt(3), 260.0, 2.5
    for delta in (5.0, 30.0, 60.0, 90.0):
        r = em(mode="synchronous_machine", V=400.0, E=e_ph * math.sqrt(3), X_s=x_s, delta_deg=delta)
        e = e_ph * np.exp(1j * math.radians(delta))
        i = (e - v_ph) / (1j * x_s)  # generator convention: current from E to the network V
        p = 3 * (v_ph * np.conj(i)).real
        assert rel(r["P"], p, 1e-12), (r, p)
    assert rel(
        em(mode="synchronous_machine", V=400.0, E=450.0, X_s=x_s)["P_max"],
        400.0 * 450.0 / x_s,
        1e-13,
    )
    t = em(mode="synchronous_machine", V=400.0, E=450.0, X_s=x_s, delta_deg=30.0, f=50.0, poles=4)
    assert rel(t["T"], t["P"] / (2 * math.pi * 1500 / 60), 1e-13)
    # examples of the page "Synchronous motor": 4 poles 50 Hz -> 1500 rpm; 12 poles 60 Hz -> 600 rpm
    assert em(mode="synchronous_machine", f=50.0, poles=4)["n_s_rpm"] == 1500.0
    assert em(mode="synchronous_machine", f=60.0, poles=12)["n_s_rpm"] == 600.0
    assert "even" in bad({"mode": "synchronous_machine", "f": 50.0, "poles": 3})


def test_transformer_ratio_against_coupled_coils() -> None:
    n_p, n_s, f, lam = (
        500.0,
        50.0,
        50.0,
        1e3,
    )  # perfect coupling, permeance lam -> magnetising current negligible
    w = 2 * math.pi * f
    l1, l2, m = n_p**2 * lam, n_s**2 * lam, n_p * n_s * lam
    for z_load in (2.0, 10.0):
        # V1 = j w (L1 I1 - M I2); 0 = j w (M I1 - L2 I2) - Z I2, with V1 = 230 V; unknowns I1, I2
        a = np.array([[1j * w * l1, -1j * w * m], [1j * w * m, -1j * w * l2 - z_load]])
        i1, i2 = np.linalg.solve(a, [230.0, 0.0])
        r = em(mode="transformer", N_p=n_p, N_s=n_s, V_p=230.0, I_p=abs(i1), Z_s=z_load)
        assert rel(r["V_s"], abs(z_load * i2), 1e-9), (r, abs(z_load * i2))
        assert rel(r["I_s"], abs(i2), 1e-6) and rel(r["Z_p"], abs(230.0 / i1), 1e-6)


def test_transformer_emf_against_flux_derivative() -> None:
    f, n, area, b = 50.0, 200.0, 0.01, 1.2
    t = np.linspace(0, 1 / f, 200001)
    e = n * np.gradient(area * b * np.sin(2 * math.pi * f * t), t)
    rms = float(np.sqrt(np.mean(e[:-1] ** 2)))
    r = em(mode="transformer", calc="emf", f=f, N=n, A=area, B_peak=b)
    assert rel(r["E_rms"], rms, 1e-8), (r, rms)
    assert round(r["E_rms"] / (f * n * area * b), 2) == 4.44
    assert rel(
        em(mode="transformer", calc="emf", f=f, N=n, A=area, E_rms=r["E_rms"])["B_peak"], b, 1e-13
    )
