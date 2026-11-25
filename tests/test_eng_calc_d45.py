"""eng_calc, modes added by D45: collisions, orbits, gyroscope, flywheel, Doppler, strings, pipes,
sound intensity, impedances, transients.

Independent methods: conservation of momentum; numerical integration of the two-body problem
(solve_ivp) over one period; vis-viva and angular momentum at the apsides; arrival times of the
wave fronts solved with brentq (Doppler); eigenvalues of the discrete string (eigh_tridiagonal);
boundary condition of the pipe solved with brentq; complex arithmetic of numpy; circuit
equations integrated with solve_ivp.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from tests.conftest import ok

G0 = 9.80665
MU_E = 3.986004418e14


def e(operation: str, **params: object) -> dict:
    return ok("eng_calc", operation=operation, params=params)


def kd(**p: object) -> dict:
    return e("kinematics_dynamics", **p)


def oa(**p: object) -> dict:
    return e("optics_acoustics", **p)


def ci(**p: object) -> dict:
    return e("circuit_ac_dc", **p)


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------------------------------------------------------ collision


@pytest.mark.parametrize("coef", [0.0, 0.6, 1.0])
def test_collision_momentum_restitution(coef: float) -> None:
    r = kd(mode="collision", m1=2, v1=3, m2=1, v2=-1, e=coef)
    a, b = r["v1_after"], r["v2_after"]
    assert rel(2 * a + 1 * b, 2 * 3 + 1 * -1) and rel(b - a, coef * (3 - -1))
    assert rel(r["J"], 2 * (a - 3)) and rel(r["J"], -1 * (b - -1))
    assert r["KE_lost"] >= -1e-12 and (coef < 1 or abs(r["KE_lost"]) < 1e-9)


def test_collision_wall() -> None:
    assert rel(kd(mode="collision", m1=1, v1=5, e=0.8)["v1_after"], -4)
    assert rel(kd(mode="collision", m1=1, v1=5, v2=1, e=0.8)["v1_after"], 1 - 0.8 * 4)


# ------------------------------------------------------------------ orbit


def _orbit_return(mu: float, r0: float, v0: float, T: float) -> float:
    def f(_t: float, s: np.ndarray) -> list[float]:
        x, y, vx, vy = s
        d3 = (x * x + y * y) ** 1.5
        return [vx, vy, -mu * x / d3, -mu * y / d3]

    sol = solve_ivp(f, (0, T), [r0, 0, 0, v0], rtol=1e-12, atol=1e-6, method="DOP853")
    x, y = sol.y[0, -1], sol.y[1, -1]
    return math.hypot(x - r0, y) / r0


def test_orbit_circular_numeric_period() -> None:
    r = kd(mode="orbit", mu=MU_E, r=7e6)
    assert rel(r["v_circular"], math.sqrt(MU_E / 7e6)) and rel(
        r["v_escape"], math.sqrt(2) * r["v_circular"]
    )
    assert _orbit_return(MU_E, 7e6, r["v_circular"], r["period"]) < 1e-7
    assert rel(r["energy"], 0.5 * r["v_circular"] ** 2 - MU_E / 7e6)
    m = kd(mode="orbit", M=5.972e24, r=7e6)
    assert rel(m["v_circular"], math.sqrt(6.67430e-11 * 5.972e24 / 7e6))


def test_orbit_ellipse_and_hohmann() -> None:
    rp, ra = 7e6, 4.2e7
    r = kd(mode="orbit", mu=MU_E, r_p=rp, r_a=ra)
    a = r["a"]
    assert rel(r["v_p"] ** 2, MU_E * (2 / rp - 1 / a)) and rel(
        r["v_a"] ** 2, MU_E * (2 / ra - 1 / a)
    )
    assert rel(rp * r["v_p"], ra * r["v_a"]) and rel(r["h"], rp * r["v_p"])
    assert _orbit_return(MU_E, rp, r["v_p"], r["period"]) < 1e-6
    h = kd(mode="orbit", mu=MU_E, r=rp, r_target=ra)
    assert rel(h["dv1"], r["v_p"] - math.sqrt(MU_E / rp)) and rel(
        h["dv2"], math.sqrt(MU_E / ra) - r["v_a"]
    )
    assert rel(h["t_transfer"], r["period"] / 2) and rel(h["dv_total"], h["dv1"] + h["dv2"])


# ------------------------------------------------------------------ gyroscope, flywheel


def test_gyroscope() -> None:
    w = 3000 * 2 * math.pi / 60
    r = kd(mode="gyroscope", I_m=0.01, omega=[3000, "rpm"], m=0.5, r=0.05)
    T = 0.5 * G0 * 0.05
    assert rel(r["T"], T) and rel(r["Omega_p"], T / (0.01 * w)) and rel(r["L"], 0.01 * w)
    # |dL/dt| of a vector L rotating at Omega_p, centred finite difference
    Om, h = r["Omega_p"], 1e-6
    dL = (
        r["L"]
        * math.hypot(math.cos(Om * h) - math.cos(-Om * h), math.sin(Om * h) - math.sin(-Om * h))
        / (2 * h)
    )
    assert rel(dL, T, 1e-9)
    back = kd(mode="gyroscope", I_m=0.01, omega=w, Omega_p=r["Omega_p"])
    assert rel(back["T"], T)


def test_flywheel_energy_and_design() -> None:
    r = kd(
        mode="flywheel", shape="disk", m=50, R=0.3, omega_max=[1600, "rpm"], omega_min=[1400, "rpm"]
    )
    Iv = 0.5 * 50 * 0.09
    w1, w2 = 1600 * math.pi / 30, 1400 * math.pi / 30
    assert rel(r["I_m"], Iv) and rel(r["dE"], 0.5 * Iv * (w1**2 - w2**2))
    assert rel(r["dE"], Iv * r["Cs"] * r["omega_mean"] ** 2)
    d = kd(mode="flywheel", dE=5000, omega_mean=[1500, "rpm"], Cs=0.02)
    wm = 1500 * math.pi / 30
    hi, lo = wm * 1.01, wm * 0.99
    assert rel(0.5 * d["I_required"] * (hi**2 - lo**2), 5000)


# ------------------------------------------------------------------ doppler, string, pipe, intensity


def test_doppler_wavefront_arrivals() -> None:
    f, c, vs, vo, D = 440.0, 343.0, 30.0, 10.0, 1000.0
    r = oa(mode="doppler", f=f, c=c, v_source=vs, v_observer=vo)

    def arrival(k: int) -> float:
        te = k / f  # emission instant of front k

        def gap(t: float) -> float:  # front k against the observer position
            return vs * te + c * (t - te) - (D - vo * t)

        return brentq(gap, te, te + 100, xtol=1e-15, rtol=1e-15)

    assert rel(r["f_observed"], 1 / (arrival(1) - arrival(0)), 1e-9)


def test_string_discrete_eigenvalues() -> None:
    L, T, mu = 0.65, 70.0, 1.2e-3
    r = oa(mode="string", L=L, T=T, mu=[1.2, "g/m"])
    n = 4000
    h = L / (n + 1)
    lam = eigh_tridiagonal(
        np.full(n, 2 / h**2),
        np.full(n - 1, -1 / h**2),
        select="i",
        select_range=(0, 2),
        eigvals_only=True,
    )
    v = math.sqrt(T / mu)
    f_num = v * np.sqrt(lam) / (2 * math.pi)
    assert rel(r["f_n"], f_num[0], 1e-6) and rel(r["v"], v)
    assert rel(oa(mode="string", L=L, T=T, mu=mu, n=3)["f_n"], f_num[2], 1e-6)


@pytest.mark.parametrize("n", [1, 2])
def test_pipe_boundary_condition(n: int) -> None:
    L, c = 0.8, 343.0
    oc = oa(mode="pipe", L=L, c=c, ends="open_closed", n=n)
    k = brentq(
        lambda k: math.cos(k * L), (n - 1) * math.pi / L + 1e-9, n * math.pi / L - 1e-9, xtol=1e-15
    )
    assert rel(oc["f_n"], c * k / (2 * math.pi), 1e-12)
    oo = oa(mode="pipe", L=L, c=c, ends="open_open", n=n)
    k2 = brentq(
        lambda k: math.sin(k * L), (n - 0.5) * math.pi / L, (n + 0.5) * math.pi / L, xtol=1e-15
    )
    assert rel(oo["f_n"], c * k2 / (2 * math.pi), 1e-12)


def test_intensity() -> None:
    W, r0 = 0.01, 5.0
    r = oa(mode="intensity", W=W, r=r0, rho=1.204, c=343)
    flux = r["I"] * 4 * math.pi * r0**2  # integral of the intensity over the sphere
    assert (
        rel(flux, W) and rel(r["L_W_dB"], 100) and rel(r["L_I_dB"], 10 * math.log10(r["I"] / 1e-12))
    )
    p = math.sqrt(r["I"] * 1.204 * 343)
    assert (
        rel(r["p_rms"], p)
        and rel(r["SPL_dB"], 20 * math.log10(p / 2e-5))
        and rel(r["Z"], 1.204 * 343)
    )
    assert rel(oa(mode="intensity", W=W, r=r0, Q_dir=2)["I"], 2 * r["I"])


# ------------------------------------------------------------------ impedance, transient


@pytest.mark.parametrize("conn", ["series", "parallel"])
def test_impedance_complex(conn: str) -> None:
    f, R, L, C = 50.0, 10.0, 0.05, 1e-4
    w = 2 * math.pi * f
    zs = [complex(R, 0), complex(0, w * L), 1 / complex(0, w * C)]
    Z = sum(zs) if conn == "series" else 1 / sum(1 / z for z in zs)
    r = ci(
        mode="impedance",
        f=f,
        connection=conn,
        V=230,
        elements=[
            {"type": "R", "value": R},
            {"type": "L", "value": [50, "mH"]},
            {"type": "C", "value": [100, "uF"]},
        ],
    )
    assert rel(r["Z_re"], Z.real) and rel(r["Z_im"], Z.imag) and rel(r["Z_abs"], abs(Z))
    assert rel(r["phase_deg"], math.degrees(np.angle(Z)))
    S = 230 * (230 / Z).conjugate()
    assert rel(r["P"], S.real) and rel(r["Q"], S.imag) and rel(r["S"], abs(S))


def _ivp(f, y0: list[float], t: float) -> np.ndarray:
    return solve_ivp(f, (0, t), y0, rtol=1e-12, atol=1e-14, method="DOP853").y[:, -1]


def test_rc_rl_transients() -> None:
    R, C, V, t = 1e3, 1e-4, 12.0, 0.05
    r = ci(mode="transient", circuit="RC", R=R, C=[100, "uF"], V=V, t=t)
    assert rel(r["tau"], 0.1) and rel(
        r["v_C"], _ivp(lambda _t, y: [(V - y[0]) / (R * C)], [0], t)[0], 1e-9
    )
    d = ci(mode="transient", circuit="RC", R=R, C=C, V=V, t=t, kind="discharge")
    assert rel(d["v_C"], _ivp(lambda _t, y: [-y[0] / (R * C)], [V], t)[0], 1e-9)
    L = 0.2
    rl = ci(mode="transient", circuit="RL", R=50, L=L, V=V, t=2e-3)
    assert rel(rl["i"], _ivp(lambda _t, y: [(V - 50 * y[0]) / L], [0], 2e-3)[0], 1e-9)


@pytest.mark.parametrize("R", [10.0, 2 * math.sqrt(0.01 / 1e-5), 200.0])
def test_rlc_step(R: float) -> None:
    L, C, V, t = 0.01, 1e-5, 5.0, 1e-3
    r = ci(mode="transient", circuit="RLC", R=R, L=[10, "mH"], C=[10, "uF"], V=V, t=[1, "ms"])
    q, i = _ivp(lambda _t, y: [y[1], (V - R * y[1] - y[0] / C) / L], [0, 0], t)
    assert rel(r["v_C"], q / C, 1e-8) and rel(r["i"], i, 1e-8)
    want = {10.0: "underdamped", 200.0: "overdamped"}.get(R, "critical")
    assert r["regime"] == want
