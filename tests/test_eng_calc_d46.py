"""eng_calc, operations of D46 (strength without tables): thermal stresses, Hertz contact,
helical compression springs, interference fits, strain analysis and rosettes.

Independent methods: plane Hooke's law solved as a linear system (numpy); Hertz force-approach
relation inverted with brentq and integral of the pressure (quad); torsion energy of the wire
integrated along the helix (quad, Castigliano's theorem); Lamé radial displacements at the
contact solved with brentq; strain tensor diagonalised with numpy (eigh) and rosette readings
rebuilt from the tensor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq, minimize_scalar

from tests.conftest import call, ok


def e(operation: str, **params: object) -> dict:
    return ok("eng_calc", operation=operation, params=params)


def bad(operation: str, params: dict) -> str:
    """Expects INVALID_ARGUMENT and returns the message."""
    resp = call("eng_calc", operation=operation, params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


IN = 0.0254  # m
LBF = 4.4482216152605  # N
PSI = 6894.757293168361  # Pa


# ------------------------------------------------------------------ thermal_stress


def test_thermal_free_expansion() -> None:
    r = e("thermal_stress", mode="free", alpha=[12, "um/(m*K)"], dT=50, L=[2, "m"])
    assert rel(r["epsilon_th"], 12e-6 * 50) and rel(r["dL"], 12e-6 * 50 * 2)
    r2 = e("thermal_stress", mode="free", alpha=12e-6, T1=[20, "C"], T2=[70, "C"], L=2)
    assert rel(r2["dL"], r["dL"])


def test_thermal_bar_fully_constrained() -> None:
    r = e("thermal_stress", mode="bar", alpha=12e-6, dT=40, E=[210, "GPa"], A=[400, "mm2"])
    # constraint: zero total strain, sigma / E + alpha dT = 0, solved with brentq
    s = brentq(lambda s: s / 210e9 + 12e-6 * 40, -1e10, 1e10, xtol=1e-6)
    assert rel(r["sigma"], s, 1e-9) and r["sigma"] < 0
    assert rel(r["F"], s * 400e-6, 1e-9)


def test_thermal_bar_with_gap() -> None:
    base = {"mode": "bar", "alpha": 12e-6, "E": 200e9, "L": 1.0, "gap": [0.3, "mm"]}
    small = e("thermal_stress", dT=20, **base)  # 0.24 mm < 0.3 mm: the clearance does not close
    assert small["sigma"] == 0 and small["gap_closed"] is False
    big = e("thermal_stress", dT=50, **base)  # 0.6 mm: 0.3 mm restrained
    s = brentq(lambda s: s / 200e9 * 1.0 + 12e-6 * 50 * 1.0 - 0.3e-3, -1e10, 1e10, xtol=1e-6)
    assert big["gap_closed"] is True and rel(big["sigma"], s, 1e-9)
    cold = e("thermal_stress", dT=-50, **base)
    assert cold["sigma"] == 0


def test_thermal_plate_biaxial() -> None:
    E, nu, a, dT = 70e9, 0.33, 23e-6, 60
    r = e("thermal_stress", mode="plate", alpha=a, dT=dT, E=E, nu=nu)
    # plane Hooke: eps_x = (sx - nu sy) / E + a dT = 0, eps_y = (sy - nu sx) / E + a dT = 0
    M = np.array([[1, -nu], [-nu, 1]]) / E
    sx, sy = np.linalg.solve(M, [-a * dT, -a * dT])
    assert rel(r["sigma"], float(sx), 1e-9) and rel(float(sx), float(sy), 1e-12)


def test_thermal_errors() -> None:
    msg = bad(
        operation="thermal_stress", params={"mode": "plate", "alpha": 1e-5, "dT": 10, "E": 2e11}
    )
    assert "nu" in msg
    msg = bad(
        operation="thermal_stress", params={"mode": "free", "alpha": 1e-5, "dT": 10, "E": 2e11}
    )
    assert "E" in msg


# ------------------------------------------------------------------ contact_hertz


def _estar(E1: float, n1: float, E2: float, n2: float) -> float:
    return 1 / ((1 - n1**2) / E1 + (1 - n2**2) / E2)


@pytest.mark.parametrize("R2", [None, 0.03, -0.05])
def test_hertz_sphere_force_approach(R2: float | None) -> None:
    F, R1, E, nu = 500.0, 0.01, 210e9, 0.3
    kw = {} if R2 is None else {"R2": R2}
    r = e("contact_hertz", geometry="sphere", F=F, R1=R1, E1=E, nu1=nu, **kw)
    R = 1 / (1 / R1 + (0 if R2 is None else 1 / R2))
    Es = _estar(E, nu, E, nu)
    assert rel(r["E_star"], Es) and rel(r["R"], R)
    # F = 4/3 E* R^0.5 d^1.5 inverted numerically
    d = brentq(
        lambda d: 4 / 3 * Es * math.sqrt(R) * d**1.5 - F, 1e-12, 1e-2, xtol=1e-20, rtol=1e-15
    )
    assert rel(r["d"], d, 1e-9) and rel(r["a"], math.sqrt(R * d), 1e-9)
    # the integral of the elliptical pressure over the contact area equals F
    a, p0 = r["a"], r["p0"]
    Fint = quad(lambda s: p0 * math.sqrt(1 - s * s / (a * a)) * 2 * math.pi * s, 0, a)[0]
    assert rel(Fint, F, 1e-8)
    assert rel(r["p_mean"], F / (math.pi * a * a))


def test_hertz_sphere_subsurface_shear() -> None:
    r = e("contact_hertz", geometry="sphere", F=1000, R1=0.02, E1=200e9, nu1=0.3)
    a, p0 = r["a"], r["p0"]

    def tau(z: float) -> float:  # Johnson 1985, stresses on the axis
        zeta = z / a
        sz = -p0 / (1 + zeta**2)
        sr = -p0 * ((1 + 0.3) * (1 - zeta * math.atan(1 / zeta)) - 1 / (2 * (1 + zeta**2)))
        return (sr - sz) / 2

    m = minimize_scalar(
        lambda z: -abs(tau(z)), bounds=(1e-6 * a, 3 * a), method="bounded", options={"xatol": 1e-12}
    )
    assert rel(r["tau_max_1"], abs(tau(m.x)), 1e-6) and rel(r["z_tau_max_1"], m.x, 1e-4)
    assert 0.30 < r["tau_max_1"] / p0 < 0.32 and 0.46 < r["z_tau_max_1"] / a < 0.49


def test_hertz_cylinder() -> None:
    F, L, R1, R2 = 2000.0, 0.02, 0.015, 0.03
    r = e(
        "contact_hertz",
        geometry="cylinder",
        F=F,
        L=L,
        R1=R1,
        R2=R2,
        E1=210e9,
        nu1=0.3,
        E2=110e9,
        nu2=0.34,
    )
    Es, R = _estar(210e9, 0.3, 110e9, 0.34), 1 / (1 / R1 + 1 / R2)
    # F = pi L E* b^2 / (4 R) inverted numerically
    b = brentq(lambda b: math.pi * L * Es * b * b / (4 * R) - F, 1e-9, 1e-2, xtol=1e-20, rtol=1e-15)
    assert rel(r["b"], b, 1e-9)
    p0 = r["p0"]
    Fint = quad(lambda x: p0 * math.sqrt(1 - x * x / (b * b)) * L, -b, b)[0]
    assert rel(Fint, F, 1e-8)
    assert 0.29 < r["tau_max_1"] / p0 < 0.31 and 0.77 < r["z_tau_max_1"] / b < 0.80


def test_hertz_errors() -> None:
    msg = bad(
        operation="contact_hertz",
        params={"geometry": "cylinder", "F": 1, "R1": 0.01, "E1": 2e11, "nu1": 0.3},
    )
    assert "L" in msg
    msg = bad(
        operation="contact_hertz",
        params={"geometry": "sphere", "F": 1, "R1": 0.01, "R2": -0.005, "E1": 2e11, "nu1": 0.3},
    )
    assert "R2" in msg


# ------------------------------------------------------------------ spring_helical


def test_spring_rate_castigliano() -> None:
    d, D, Na, G, F = 0.003, 0.024, 8.0, 79.3e9, 100.0
    r = e("spring_helical", d=d, D=D, Na=Na, G=G, F=F)
    J = math.pi * d**4 / 32
    length = math.pi * D * Na
    # torsion energy U = integral of T^2 / (2 G J) along the wire; delta = 2 U / F
    U = quad(lambda s: (F * D / 2) ** 2 / (2 * G * J), 0, length)[0]
    assert rel(r["deflection"], 2 * U / F, 1e-9) and rel(r["k"], F / (2 * U / F), 1e-9)
    C = D / d
    assert rel(r["C"], C)
    assert rel(r["K_W"], (4 * C - 1) / (4 * C - 4) + 0.615 / C)
    assert rel(r["K_B"], (4 * C + 2) / (4 * C - 3))
    assert rel(r["tau"], r["K_B"] * 8 * F * D / (math.pi * d**3))
    assert rel(r["tau_wahl"], r["K_W"] * 8 * F * D / (math.pi * d**3))


@pytest.mark.parametrize(
    ("ends", "extra", "solid"),
    [("plain", 0, 1), ("plain_ground", 1, 0), ("squared", 2, 1), ("squared_ground", 2, 0)],
)
def test_spring_ends(ends: str, extra: int, solid: int) -> None:
    r = e("spring_helical", d=0.002, D_o=0.02, Na=10, G=79.3e9, ends=ends, L0=0.06)
    assert rel(r["D"], 0.018) and r["Nt"] == 10 + extra
    assert rel(r["L_solid"], 0.002 * (10 + extra + solid))
    assert rel(r["F_solid"], r["k"] * (0.06 - r["L_solid"]))


def test_spring_deflection_input_and_frequency() -> None:
    r = e("spring_helical", d=0.004, D=0.03, Na=6, G=80e9, deflection=[5, "mm"], rho=7850)
    k = 80e9 * 0.004**4 / (8 * 0.03**3 * 6)
    assert rel(r["F"], k * 0.005)
    m = 7850 * math.pi * 0.004**2 / 4 * math.pi * 0.03 * 6
    assert rel(r["f_n"], 0.5 * math.sqrt(k / m))


# ------------------------------------------------------------------ interference_fit


def _lame_u(r: float, ri: float, ro: float, pi_: float, po: float, E: float, nu: float) -> float:
    """Radial displacement of a thick cylinder in plane stress, from the Lamé sigma_t and sigma_r."""
    A = (pi_ * ri**2 - po * ro**2) / (ro**2 - ri**2)
    B = (pi_ - po) * ri**2 * ro**2 / (ro**2 - ri**2)
    st, sr = A + B / r**2, A - B / r**2
    return r / E * (st - nu * sr)


@pytest.mark.parametrize("d_i", [0.0, 0.02])
def test_interference_pressure_by_displacements(d_i: float) -> None:
    d, Do, delta = 0.05, 0.1, 40e-6
    Eh, nh, Es, ns = 200e9, 0.3, 110e9, 0.34
    kw = {"d_i": d_i} if d_i else {}
    r = e(
        "interference_fit",
        d=d,
        D_o=Do,
        delta=delta,
        E_hub=Eh,
        nu_hub=nh,
        E_shaft=Es,
        nu_shaft=ns,
        mu=0.15,
        L=0.04,
        **kw,
    )
    R, ro, ri = d / 2, Do / 2, d_i / 2

    def gap(p: float) -> float:
        u_hub = _lame_u(R, R, ro, p, 0.0, Eh, nh)
        u_sh = -p * R * (1 - ns) / Es if ri == 0 else _lame_u(R, ri, R, 0.0, p, Es, ns)
        return u_hub - u_sh - delta / 2

    p = brentq(gap, 0, 1e10, xtol=1e-6, rtol=1e-15)
    assert rel(r["p"], p, 1e-9)
    assert rel(r["sigma_t_hub"], p * (ro**2 + R**2) / (ro**2 - R**2))
    st_sh = -p if ri == 0 else -p * (R**2 + ri**2) / (R**2 - ri**2)
    assert rel(r["sigma_t_shaft"], st_sh)
    assert rel(r["T"], 0.15 * p * math.pi * d * 0.04 * d / 2) and rel(
        r["F_axial"], 0.15 * p * math.pi * d * 0.04
    )


def test_interference_same_material_and_shrink() -> None:
    r = e(
        "interference_fit",
        d=0.05,
        D_o=0.1,
        delta=[0.04, "mm"],
        E_hub=[210, "GPa"],
        nu_hub=0.3,
        alpha_hub=12e-6,
        clearance=[0.02, "mm"],
    )
    # same material, solid shaft: p = E delta (D_o^2 - d^2) / (2 d D_o^2)
    p = 210e9 * 40e-6 * (0.1**2 - 0.05**2) / (2 * 0.05 * 0.1**2)
    assert rel(r["p"], p, 1e-12)
    assert rel(r["dT_assembly"], (40e-6 + 20e-6) / (12e-6 * 0.05))


# ------------------------------------------------------------------ strain_analysis


def test_strain_transform_principal() -> None:
    ex, ey, gxy = 500e-6, -200e-6, 300e-6
    r = e("strain_analysis", mode="transform", ex=ex, ey=ey, gxy=gxy, theta_deg=30)
    w, v = np.linalg.eigh(np.array([[ex, gxy / 2], [gxy / 2, ey]]))
    assert rel(r["e1"], float(w[1]), 1e-12) and rel(r["e2"], float(w[0]), 1e-12)
    assert rel(r["gamma_max"], float(w[1] - w[0]), 1e-12)
    ang = math.degrees(math.atan2(v[1, 1], v[0, 1])) % 180
    assert abs((r["theta_p_deg"] % 180) - ang) < 1e-9
    t = math.radians(30)
    n = np.array([math.cos(t), math.sin(t)])
    m = np.array([-math.sin(t), math.cos(t)])
    Tn = np.array([[ex, gxy / 2], [gxy / 2, ey]])
    assert rel(r["ex_theta"], float(n @ Tn @ n), 1e-12)
    assert rel(r["gxy_theta"], float(2 * n @ Tn @ m), 1e-12)


@pytest.mark.parametrize(
    ("rosette", "angles"),
    [("rectangular", (0, 45, 90)), ("delta", (0, 60, 120)), (None, (10, 55, 130))],
)
def test_rosette_recovers_tensor(rosette: str | None, angles: tuple[float, float, float]) -> None:
    ex, ey, gxy = 420e-6, -130e-6, 260e-6
    Tn = np.array([[ex, gxy / 2], [gxy / 2, ey]])
    reads = []
    for a in angles:
        n = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        reads.append(float(n @ Tn @ n))
    kw: dict[str, object] = {"rosette": rosette} if rosette else {"angles_deg": list(angles)}
    r = e(
        "strain_analysis",
        mode="rosette",
        ea=reads[0],
        eb=reads[1],
        ec=reads[2],
        E=200e9,
        nu=0.3,
        **kw,
    )
    assert rel(r["ex"], ex, 1e-12) and rel(r["ey"], ey, 1e-12) and rel(r["gxy"], gxy, 1e-12)
    w = np.linalg.eigvalsh(Tn)
    c = 200e9 / (1 - 0.09)
    assert rel(r["sigma1"], c * (w[1] + 0.3 * w[0]), 1e-9) and rel(
        r["sigma2"], c * (w[0] + 0.3 * w[1]), 1e-9
    )


def test_rosette_errors() -> None:
    msg = bad(
        operation="strain_analysis",
        params={"mode": "rosette", "ea": 1e-4, "eb": 1e-4, "ec": 1e-4, "angles_deg": [0, 180, 90]},
    )
    assert "angles" in msg


# ------------------------------------------------------------------ numerical examples of the sources


def test_source_shigley_hertz_cylinder() -> None:
    """Shigley, example T3.19.1: cast-iron wheel (6 in, 2 in wide) on flat steel, 800 lbf."""
    r = e(
        "contact_hertz",
        geometry="cylinder",
        F=800 * LBF,
        R1=3 * IN,
        L=2 * IN,
        E1=14.5e6 * PSI,
        nu1=0.211,
        E2=30e6 * PSI,
        nu2=0.292,
    )
    assert rel(r["b"] / IN, 1.214e-2, 1e-3) and rel(r["p0"] / PSI, 20980, 1e-3)
    # steel (nu >= 0.25): maximum tau 0.30 p0 at 0.786 b, as in the source (6298 psi)
    assert rel(r["tau_max_2"] / PSI, 6298, 1e-3) and rel(r["z_tau_max_2"] / r["b"], 0.786, 2e-3)
    # cast iron (nu = 0.211): the true maximum is higher up and larger (the source computes at 0.786 b)
    assert r["tau_max_1"] > r["tau_max_2"] and r["z_tau_max_1"] < 0.5 * r["b"]


def test_source_shigley_shrink_fit() -> None:
    """Shigley, example T4.17.1: steel hub and shaft, 1 in bore, hub 0.5 in thick."""
    kw = {"d": 1 * IN, "D_o": 2 * IN, "E_hub": 30e6 * PSI, "nu_hub": 0.292}
    loose = e("interference_fit", delta=(1.0018 - 1.0008) * IN, **kw)
    tight = e("interference_fit", delta=(1.0023 - 1.0000) * IN, **kw)
    assert rel(loose["p"] / PSI, 11250, 1e-3) and rel(loose["sigma_t_hub"] / PSI, 18750, 1e-3)
    assert rel(tight["p"] / PSI, 25875, 1e-3) and rel(tight["sigma_t_hub"] / PSI, 43125, 1e-3)
    assert rel(tight["sigma_t_shaft"], tight["sigma_r_shaft"]) and rel(
        tight["sigma_r_shaft"], -tight["p"]
    )


def test_source_uw_thick_cylinders() -> None:
    """Univ. of Washington ME354, problem 4: 400/240/140 mm, radial interference 0.3 mm."""
    r = e(
        "interference_fit",
        d=[240, "mm"],
        D_o=[400, "mm"],
        d_i=[140, "mm"],
        delta=[0.6, "mm"],
        E_hub=[200, "GPa"],
        nu_hub=0.3,
    )
    assert round(r["p"] / 1e6, 1) == 120.3
    assert round(r["sigma_t_shaft"] / 1e6) == -244 and round(r["sigma_t_shaft_inner"] / 1e6) == -365
    assert round(r["sigma_t_hub"] / 1e6) == 256


def test_source_ansys_vm27_gap() -> None:
    """ANSYS VM27: bar with 0.002 in clearance, L = 3 in, from 70 to 170 degrees F."""
    r = e(
        "thermal_stress",
        mode="bar",
        alpha=12.5e-6 * 1.8,
        T1=[70, "F"],
        T2=[170, "F"],
        E=10.5e6 * PSI,
        L=3 * IN,
        gap=0.002 * IN,
    )
    assert rel(r["sigma"] / PSI, -6125, 1e-9) and rel(r["epsilon_th"], 0.00125, 1e-9)


def test_source_roylance_strain_rotation() -> None:
    """Roylance (LibreTexts 3.3), example 2: ex = 0.01, ey = -0.01, gxy = 0, rotation of 45 degrees."""
    r = e("strain_analysis", mode="transform", ex=0.01, ey=-0.01, gxy=0, theta_deg=45)
    assert (
        abs(r["ex_theta"]) < 1e-15
        and abs(r["ey_theta"]) < 1e-15
        and rel(r["gxy_theta"], -0.02, 1e-12)
    )


@pytest.mark.parametrize("rosette", ["rectangular", "delta"])
def test_source_tn515_direct_stresses(rosette: str) -> None:
    """Micro-Measurements TN-515, equations 11 (rectangular) and 12 (delta): stresses from the readings."""
    e1, e2, e3, E, nu = 900e-6, 150e-6, -350e-6, 200e9, 0.3
    r = e("strain_analysis", mode="rosette", rosette=rosette, ea=e1, eb=e2, ec=e3, E=E, nu=nu)
    if rosette == "rectangular":
        mid = (e1 + e3) / (1 - nu)
        root = math.sqrt((e1 - e3) ** 2 + (2 * e2 - e1 - e3) ** 2) / (1 + nu)
        sp, sq = E / 2 * (mid + root), E / 2 * (mid - root)
    else:
        mid = (e1 + e2 + e3) / (3 * (1 - nu))
        root = (
            math.sqrt(2)
            / (3 * (1 + nu))
            * math.sqrt((e1 - e2) ** 2 + (e2 - e3) ** 2 + (e3 - e1) ** 2)
        )
        sp, sq = E * (mid + root), E * (mid - root)
    assert rel(r["sigma1"], sp, 1e-9) and rel(r["sigma2"], sq, 1e-9)


def test_source_shigley_spring_example() -> None:
    """Shigley (lecture notes), example 7-1: wire 0.037 in, outer diameter 7/16 in, squared ends, 12.5 coils."""
    r = e("spring_helical", d=0.037 * IN, D_o=7 / 16 * IN, Na=10.5, G=11.85e6 * PSI, ends="squared")
    assert round(r["C"], 1) == 10.8 and r["Nt"] == 12.5 and round(r["L_solid"] / IN, 2) == 0.5


def test_spring_frequency_discrete_chain() -> None:
    """f_n of the spring between two plates against the chain of n segments (masses and springs), fixed ends."""
    r = e("spring_helical", d=0.002, D=0.016, Na=7, G=79.3e9, rho=7850)
    k, m, n = r["k"], r["m_active"], 4000
    w2 = eigh_tridiagonal(
        np.full(n - 1, 2 * n * k / (m / n)),
        np.full(n - 2, -n * k / (m / n)),
        select="i",
        select_range=(0, 0),
        eigvals_only=True,
    )[0]
    assert rel(r["f_n"], math.sqrt(w2) / (2 * math.pi), 1e-6)
