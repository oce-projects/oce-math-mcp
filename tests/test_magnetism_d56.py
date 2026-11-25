"""Magnetic fields, forces, inductance and induction (D56).

Sources: Wikipedia «Magnetic field» (table of the fields), «Biot–Savart law», «Solenoid», «Ampère's force
law», «Lorentz force», «Inductance» (table of the inductances, Rosa formula), «Faraday's law of
induction». Checks independent of the module formulas:
- loop, wire and finite solenoid against the Biot-Savart law integrated numerically (quad);
- force between wires against the force on a wire in the field of the other (B I L, page «Lorentz force»);
- cyclotron radius against the trajectory integrated with solve_ivp;
- inductances of toroid and coaxial cable against the integrated flux linkage (L = N Phi / I) and
  against the magnetic energy B^2 / (2 mu) integrated over the volume;
- Wheeler formula against the long solenoid; parallel wires HF (arcosh) against the form with the
  logarithm for s much greater than a;
- rotating coil against the numerical derivative of the flux.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.constants as sc
from scipy.integrate import quad, solve_ivp

from tests.conftest import call, ok

MU0 = sc.mu_0


def mag(**params: object) -> dict:
    return ok("eng_calc", operation="magnetism", params=params)


def ind(**params: object) -> dict:
    return ok("eng_calc", operation="inductance", params=params)


def emf(**params: object) -> dict:
    return ok("eng_calc", operation="electromagnetic_induction", params=params)


def bad(operation: str, params: dict) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def loop_axis_numeric(current: float, radius: float, x: float) -> float:
    """B_x on the axis from Biot-Savart: dB = mu0 I dl x r / (4 pi r^3), loop in the yz plane."""

    def integrand(phi: float) -> float:
        # element at (0, R cos, R sin), dl = R (0, -sin, cos) dphi; point (x, 0, 0)
        ry, rz = -radius * math.cos(phi), -radius * math.sin(phi)
        dly, dlz = -radius * math.sin(phi), radius * math.cos(phi)
        r3 = (x * x + ry * ry + rz * rz) ** 1.5
        return MU0 * current / (4 * math.pi) * (dly * rz - dlz * ry) / r3

    return quad(integrand, 0, 2 * math.pi, epsabs=1e-20, epsrel=1e-12)[0]


def test_loop_and_wire_against_biot_savart() -> None:
    for x in (0.0, 0.03, 0.2):
        r = mag(mode="loop", I=2.0, R=0.05, x=x)
        assert rel(r["B"], loop_axis_numeric(2.0, 0.05, x), 1e-10), x
    coil = mag(mode="loop", I=2.0, R=0.05, N=10)
    assert rel(coil["B"], 10 * loop_axis_numeric(2.0, 0.05, 0.0), 1e-10)
    # straight wire 2 km long against the infinite wire, at 1 cm
    half = 1000.0

    def db(z: float) -> float:
        return MU0 * 5.0 / (4 * math.pi) * 0.01 / (0.01**2 + z * z) ** 1.5

    # symmetric: twice from 0 to 2 km, split at 1 m because the peak is 1 cm wide
    num = 2 * (
        quad(db, 0, 1.0, epsabs=0, epsrel=1e-12)[0] + quad(db, 1.0, half, epsabs=0, epsrel=1e-12)[0]
    )
    r = mag(mode="wire", I=5.0, r=0.01)
    assert rel(r["B"], num, 1e-6) and rel(r["H"], r["B"] / MU0, 1e-12)
    inside = mag(mode="wire", I=5.0, r=0.001, R_wire=0.002)
    assert rel(inside["B"], MU0 * 5.0 * 0.001 / (2 * math.pi * 0.002**2), 1e-12)


def test_solenoid_finite_against_loops_and_long_limit() -> None:
    n_turns, current, length, radius = 500, 0.8, 0.2, 0.02
    for z in (0.0, 0.07, 0.15):
        r = mag(mode="solenoid", N=n_turns, I=current, L=length, R=radius, z=z)
        # continuous sum of loops: density N / L, each with the numerical Biot-Savart field
        num = quad(
            lambda zp, z=z: n_turns / length * loop_axis_numeric(current, radius, z - zp),
            -length / 2,
            length / 2,
            epsrel=1e-11,
        )[0]
        assert rel(r["B"], num, 1e-8), z
    longc = mag(mode="solenoid", N=n_turns, I=current, L=length)
    assert rel(longc["B"], MU0 * n_turns * current / length, 1e-12)
    center = mag(mode="solenoid", N=n_turns, I=current, L=500.0, R=0.001)
    assert rel(
        center["B"], MU0 * n_turns * current / 500.0, 1e-6
    )  # R << L: tends to the long solenoid
    tor = mag(mode="toroid", N=200, I=1.5, r=0.05)
    sol = mag(mode="solenoid", N=200, I=1.5, L=2 * math.pi * 0.05)
    assert rel(tor["B"], sol["B"], 1e-12)


def test_forces_between_wires_and_on_wire() -> None:
    fw = mag(mode="force_wires", I1=10.0, I2=-4.0, d=0.02, L=3.0)
    b1 = mag(mode="wire", I=10.0, r=0.02)["B"]
    on = mag(mode="force_on_wire", I=4.0, L=3.0, B=b1, angle_deg=90.0)
    assert (
        rel(abs(fw["F"]), on["F"], 1e-12) and fw["F"] < 0
    )  # opposite currents: repulsion (negative)
    assert rel(fw["F_per_L"], fw["F"] / 3.0, 1e-12)
    half = mag(mode="force_on_wire", I=4.0, L=3.0, B=b1, angle_deg=30.0)
    assert rel(half["F"], on["F"] / 2, 1e-12)


def test_lorentz_force_and_cyclotron_against_ivp() -> None:
    q, m = 1.602176634e-19, 9.1093837015e-31
    v, b = [2e6, 0.0, 1e6], [0.0, 0.0, 0.01]
    r = mag(mode="lorentz", q=q, v=v, B=b, E=[0.0, 1e3, 0.0], m=m)
    f = np.array([0.0, q * 1e3, 0.0]) + q * np.cross(v, b)
    assert np.allclose(r["F"], f, rtol=1e-12, atol=0)
    c = mag(mode="lorentz", q=q, v=v, B=b, m=m)
    vperp = 2e6
    period = c["T_c"]

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        acc = q / m * np.cross(y[3:], b)
        return np.concatenate([y[3:], acc])

    sol = solve_ivp(
        rhs,
        (0, period),
        np.array([0, 0, 0, *v], dtype=float),
        rtol=1e-11,
        atol=1e-14,
        dense_output=True,
    )
    xs = sol.sol(np.linspace(0, period, 2001))
    diameter = float(np.max(np.hypot(xs[0], xs[1])))
    assert rel(c["r_c"], diameter / 2, 1e-6) and rel(c["r_c"], m * vperp / (q * 0.01), 1e-12)
    assert np.allclose(
        sol.sol(period)[:2], [0.0, 0.0], atol=1e-6 * c["r_c"]
    )  # back after one period


def test_inductance_against_flux_and_energy() -> None:
    n, h, a, b = 300, 0.01, 0.02, 0.03
    tor = ind(geometry="toroid", N=n, h=h, r_inner=a, r_outer=b, mu_r=1.0, I=2.0)
    flux = quad(lambda r: MU0 * n * 2.0 / (2 * math.pi * r) * h, a, b)[0]
    assert rel(tor["L"], n * flux / 2.0, 1e-12) and rel(tor["W"], 0.5 * tor["L"] * 4.0, 1e-12)
    coax = ind(geometry="coaxial", a=0.0005, b=0.002, L=10.0, I=3.0)
    energy = quad(
        lambda r: (MU0 * 3.0 / (2 * math.pi * r)) ** 2 / (2 * MU0) * 2 * math.pi * r * 10.0,
        0.0005,
        0.002,
    )[0]
    assert rel(coax["W"], energy, 1e-12)
    sol = ind(geometry="solenoid", N=1000, D=0.01, L=10.0)
    wheel = ind(geometry="wheeler", N=1000, D=0.01, L=10.0)
    assert rel(
        wheel["L"], sol["L"], 4e-3
    )  # measured -0.32% at l = 10 m (-0.72% at 1 m): surface-current model
    hf = ind(geometry="parallel_wires", s=1.0, a=0.001, L=2.0, hf=True)
    lf0 = ind(geometry="parallel_wires", s=1.0, a=0.001, L=2.0, Y=0.0)
    assert rel(hf["L"], lf0["L"], 1e-6)  # arcosh(s/2a) ~ ln(s/a) per s >> a
    lf1 = ind(geometry="parallel_wires", s=1.0, a=0.001, L=2.0, Y=1.0)
    assert rel(lf1["L"] - lf0["L"], MU0 / math.pi * 2.0 / 4, 1e-12)


def test_wire_shapes_and_warnings() -> None:
    rosa = ind(geometry="straight_wire", L=1.0, r=0.0005)
    assert rel(rosa["L"], 200e-9 * (math.log(2 / 0.0005) - 0.75), 1e-12)
    ac = ind(geometry="straight_wire", L=1.0, r=0.0005, hf=True)
    assert rel(rosa["L"] - ac["L"], 200e-9 * 0.25, 1e-12)
    loop = ind(geometry="circular_loop", r=0.1, a=0.001, Y=1.0)
    assert rel(loop["L"], MU0 * 0.1 * (math.log(800) - 2 + 0.25), 1e-12)
    sq = ind(geometry="rectangle", l1=0.2, l2=0.2, a=0.001, Y=0.0)
    expected = (
        MU0
        / math.pi
        * (2 * 0.2 * math.log(400) + 2 * math.sqrt(0.08) - 2 * 0.2 * math.asinh(1.0) - 2 * 0.4)
    )
    assert rel(sq["L"], expected, 1e-12)
    resp = call(
        "eng_calc",
        operation="inductance",
        params={"geometry": "wheeler", "N": 10, "D": 0.05, "L": 0.01},
    )
    assert resp["ok"] and any("0.4" in w for w in resp["warnings"])


def test_faraday_induction() -> None:
    avg = emf(mode="flux_change", N=50, A=0.01, B1=0.2, B2=0.8, dt=0.1)
    assert rel(avg["emf"], -50 * 0.01 * 0.6 / 0.1, 1e-12)
    rot = emf(mode="rotating_coil", N=100, A=0.02, B=0.5, rpm=3000.0)
    w = 3000 * 2 * math.pi / 60
    t, h = 0.0013, 1e-7

    def flux(tt: float) -> float:
        return 100 * 0.5 * 0.02 * math.cos(w * tt)

    num = -(flux(t + h) - flux(t - h)) / (2 * h)
    assert rel(rot["emf_peak"], 100 * 0.5 * 0.02 * w, 1e-12)
    assert rel(rot["emf_peak"] * math.sin(w * t), num, 1e-6)
    assert rel(rot["emf_rms"], rot["emf_peak"] / math.sqrt(2), 1e-12)
    mot = emf(mode="motional", B=0.4, L=0.5, v=3.0)
    assert rel(mot["emf"], 0.6, 1e-12)


def test_errors() -> None:
    assert "x" in bad("magnetism", {"mode": "wire", "I": 1.0, "r": 0.01, "x": 1.0})
    assert "q" in bad("magnetism", {"mode": "lorentz", "v": [1.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0]})
    assert "b" in bad("inductance", {"geometry": "coaxial", "a": 0.003, "b": 0.002, "L": 1.0})
    assert "only one" in bad(
        "electromagnetic_induction",
        {"mode": "flux_change", "N": 1, "dPhi": 0.1, "A": 0.01, "B1": 0.1, "B2": 0.2, "dt": 0.1},
    )
