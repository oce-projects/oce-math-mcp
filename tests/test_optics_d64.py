"""Geometric and wave optics (D64).

Checks independent of the server formulas:
- lenses, mirrors and a single refracting surface: exact meridional ray tracing written in the test
  (vector Snell law and reflection on spheres), with paraxial rays (height 1e-6 of the radius);
- two thin lenses: ray transfer (ABCD) matrices;
- Malus: Jones vectors, and the average over random input polarisations;
- Fresnel and Brewster: the 2 x 2 boundary-condition systems solved with numpy, Brewster as the root of R_p;
- double slit: exact path difference from the two slits to a screen at a finite distance;
- single slit and grating: phasor integral and phasor sum;
- thin film: characteristic-matrix reflectance with multiple reflections, extrema found numerically;
- Rayleigh: first zero of the Airy pattern from scipy (J1).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j1, jn_zeros

from tests.conftest import call, ok


def geo(**params: object) -> dict:
    return ok("eng_calc", operation="geometric_optics", params=params)


def wave(**params: object) -> dict:
    return ok("eng_calc", operation="wave_optics", params=params)


def bad(operation: str, params: dict) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def solve_root(fun: Callable[[float], float], a: float, b: float, xtol: float) -> float:
    return cast(float, brentq(fun, a, b, xtol=xtol))


# ---------------------------------------------------------------- exact ray tracing (2D: z along the axis, y up)


def _hit(p: np.ndarray, u: np.ndarray, z_v: float, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Point where the ray meets the surface with vertex z_v and curvature c (0 = plane), and the unit normal
    pointing against the ray."""
    if c == 0:
        q = p + (z_v - p[0]) / u[0] * u
        n = np.array([-1.0, 0.0])
    else:
        r = 1 / c
        centre = np.array([z_v + r, 0.0])
        b = float(np.dot(u, p - centre))
        disc = b * b - float(np.dot(p - centre, p - centre)) + r * r
        roots = (-b - math.sqrt(disc), -b + math.sqrt(disc))
        q = min((p + t * u for t in roots), key=lambda pt: abs(pt[0] - z_v))
        n = (q - centre) / abs(r)
    if np.dot(u, n) > 0:
        n = -n
    return q, n


def _refract(u: np.ndarray, n: np.ndarray, n1: float, n2: float) -> np.ndarray:
    cos_i = -float(np.dot(u, n))
    eta = n1 / n2
    k = 1 - eta * eta * (1 - cos_i * cos_i)
    return eta * u + (eta * cos_i - math.sqrt(k)) * n


def _axis_z(p: np.ndarray, u: np.ndarray) -> float:
    return float(p[0] - p[1] / u[1] * u[0])


def trace_lens(n: float, c1: float, c2: float, d: float, n_m: float = 1.0) -> dict[str, float]:
    """Parallel ray from the left: effective focal length, back focal distance, H2 from the second vertex."""
    h = 1e-7
    p, u = np.array([-1.0, h]), np.array([1.0, 0.0])
    q, nn = _hit(p, u, 0.0, c1)
    u = _refract(u, nn, n_m, n)
    q, nn = _hit(q, u, d, c2)
    u = _refract(u, nn, n, n_m)
    z_f = _axis_z(q, u)
    f = -h / (u[1] / u[0])
    return {"f": f, "BFD": z_f - d, "H2": z_f - f - d}


def test_thin_lensmaker_against_ray_tracing() -> None:
    for n, r1, r2 in ((1.5, 0.1, -0.1), (1.5168, 0.2, 0.5), (1.7, -0.08, 0.12)):
        r = geo(mode="lensmaker", n=n, R1=r1, R2=r2)
        tr = trace_lens(n, 1 / r1, 1 / r2, 0.0)
        assert rel(r["f"], tr["f"], 1e-8), (r, tr)
        assert rel(r["P"], 1 / tr["f"], 1e-8)


def test_thick_lens_and_principal_planes_against_ray_tracing() -> None:
    for n, r1, r2, d, n_m in (
        (1.5, 0.05, -0.05, 0.01, 1.0),
        (1.6, 0.1, 0.3, 0.008, 1.0),
        (1.5, 0.05, -0.08, 0.012, 1.333),
    ):
        r = geo(mode="lensmaker", n=n, R1=r1, R2=r2, d=d, n_medium=n_m)
        tr = trace_lens(n, 1 / r1, 1 / r2, d, n_m)
        assert rel(r["f"], tr["f"], 1e-7), (r, tr)
        assert rel(r["BFD"], tr["BFD"], 1e-7)
        assert abs(r["H2"] - tr["H2"]) <= 1e-7 * abs(tr["f"])
        back = trace_lens(n, -1 / r2, -1 / r1, d, n_m)  # the same lens seen from the right
        assert rel(r["FFD"], back["BFD"], 1e-7)
        assert abs(r["H1"] + back["H2"]) <= 1e-7 * abs(tr["f"])
        assert rel(r["P"], n_m / tr["f"], 1e-7)


def test_plano_convex_with_curvature_zero() -> None:
    r = geo(mode="lensmaker", n=1.5, C1=0, R2=-0.05, d=0.006)
    tr = trace_lens(1.5, 0.0, -20.0, 0.006)
    assert rel(r["f"], tr["f"], 1e-7) and rel(
        r["f"], 0.1, 1e-12
    )  # flat face first: f = R / (n - 1)
    assert abs(r["H2"]) < 1e-15  # the curved face is the second one: H2 on its vertex


def test_mirror_against_ray_tracing() -> None:
    for radius, d_o in (
        (0.4, 0.6),
        (0.4, 0.1),
        (-0.3, 0.5),
    ):  # concave real, concave virtual, convex
        r = geo(mode="mirror", R=radius, d_o=d_o)
        a = 1e-7
        p, u = np.array([-d_o, 0.0]), np.array([math.cos(a), math.sin(a)])
        q, nn = _hit(
            p, u, 0.0, -1 / radius
        )  # real-is-positive: the centre of a concave mirror is in front
        refl = u - 2 * float(np.dot(u, nn)) * nn
        d_i = -_axis_z(q, refl)
        assert rel(r["d_i"], d_i, 1e-7), (r, d_i)
        assert rel(r["f"], radius / 2, 1e-13)
        assert rel(r["magnification"], -d_i / d_o, 1e-7)
    both = geo(mode="mirror", d_o=0.6, d_i=0.3)
    assert rel(both["f"], 0.2, 1e-12) and rel(both["R"], 0.4, 1e-12)


def test_refracting_surface_against_ray_tracing() -> None:
    for n1, n2, radius, s_o in (
        (1.0, 1.5, 0.05, 0.3),
        (1.0, 1.333, 0.1, 0.05),
        (1.5, 1.0, -0.2, 0.5),
    ):
        r = geo(mode="refracting_surface", n1=n1, n2=n2, R=radius, d_o=s_o)
        a = 1e-7
        p, u = np.array([-s_o, 0.0]), np.array([math.cos(a), math.sin(a)])
        q, nn = _hit(p, u, 0.0, 1 / radius)
        s_i = _axis_z(q, _refract(u, nn, n1, n2))
        assert rel(r["d_i"], s_i, 1e-7), (r, s_i)
        assert rel(r["P"], (n2 - n1) / radius, 1e-13)
        assert rel(r["f_object"], n1 * radius / (n2 - n1), 1e-13)
        assert rel(r["f_image"], n2 * radius / (n2 - n1), 1e-13)
        assert rel(r["magnification"], -(n1 * s_i) / (n2 * s_o), 1e-7)


def test_two_lenses_against_abcd_matrices() -> None:
    for f1, f2, d in ((0.1, 0.2, 0.05), (0.1, -0.05, 0.02), (0.2, 0.3, 0.0), (0.05, 0.05, 0.3)):
        m = (
            np.array([[1, 0], [-1 / f2, 1]])
            @ np.array([[1, d], [0, 1]])
            @ np.array([[1, 0], [-1 / f1, 1]])
        )
        (a, _), (c, dd) = m
        r = geo(mode="two_lenses", f1=f1, f2=f2, d=d)
        assert rel(r["f"], -1 / c, 1e-12), (r, m)
        assert rel(r["BFD"], -a / c, 1e-12)
        assert rel(r["FFD"], -dd / c, 1e-12)


def test_afocal_systems_and_bad_input() -> None:
    tele = geo(mode="two_lenses", f1=0.1, f2=0.05, d=0.15)  # Keplerian telescope: d = f1 + f2
    assert (
        tele["afocal"] is True
        and tele["f"] is None
        and rel(tele["angular_magnification"], -0.5, 1e-13)
    )
    flat = geo(mode="lensmaker", n=1.5, R1=0.1, R2=0.1)  # thin meniscus with equal radii: no power
    assert flat["afocal"] is True and flat["f"] is None and flat["P"] == 0.0
    assert geo(mode="lensmaker", n=1.5, R1=0.1, R2=-0.1)["afocal"] is False
    assert "R1" in bad("geometric_optics", {"mode": "lensmaker", "n": 1.5, "R2": 0.1})
    assert "exactly two" in bad("geometric_optics", {"mode": "mirror", "R": 0.4})
    assert "Keys not used" in bad(
        "geometric_optics", {"mode": "mirror", "R": 0.4, "d_o": 1, "n": 1.5}
    )


# ---------------------------------------------------------------- wave optics


def test_malus_against_jones_vectors() -> None:
    for theta in (0.0, 30.0, 45.0, 72.5, 90.0):
        r = wave(mode="malus", I0=800.0, theta_deg=theta)
        e = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
        assert abs(r["I"] - 800.0 * float(np.dot(e, [1.0, 0.0])) ** 2) <= 1e-9 * 800.0
        phis = np.linspace(0, 2 * math.pi, 4001)[:-1]
        mean = (
            float(np.mean(np.cos(phis) ** 2)) * math.cos(math.radians(theta)) ** 2
        )  # unpolarised input
        u = wave(mode="malus", I0=800.0, theta_deg=theta, incident="unpolarized")
        assert abs(u["I"] - 800.0 * mean) <= 1e-9 * 800.0


def _boundary(n1: float, n2: float, theta_deg: float) -> tuple[float, float]:
    r_s, r_p = _amplitudes(n1, n2, theta_deg)
    return r_s**2, r_p**2


def _amplitudes(n1: float, n2: float, theta_deg: float) -> tuple[float, float]:
    """r_s and r_p from the continuity of the tangential fields, solved as two 2 x 2 linear systems."""
    ti = math.radians(theta_deg)
    tt = math.asin(n1 * math.sin(ti) / n2)
    ci, ct = math.cos(ti), math.cos(tt)
    # s: E_y continuous (1 + r = t); H_x continuous (n1 ci (1 - r) = n2 ct t)
    r_s, _ = np.linalg.solve([[1.0, -1.0], [-n1 * ci, -n2 * ct]], [-1.0, -n1 * ci])
    # p: H_y continuous (1 + r = t); E_x continuous (ci / n1 (1 - r) = ct / n2 t)
    r_p, _ = np.linalg.solve([[1.0, -1.0], [-ci / n1, -ct / n2]], [-1.0, -ci / n1])
    return float(r_s), float(r_p)


def test_fresnel_against_boundary_conditions() -> None:
    for n1, n2, theta in (
        (1.0, 1.5, 0.0),
        (1.0, 1.5, 30.0),
        (1.0, 1.5, 80.0),
        (1.5, 1.0, 35.0),
        (1.33, 2.4, 60.0),
    ):
        r = wave(mode="fresnel", n1=n1, n2=n2, theta_deg=theta)
        r_s, r_p = _boundary(n1, n2, theta)
        assert abs(r["R_s"] - r_s) < 1e-12 and abs(r["R_p"] - r_p) < 1e-12, (r, r_s, r_p)
        assert abs(r["T_s"] + r["R_s"] - 1) < 1e-12 and abs(r["T_p"] + r["R_p"] - 1) < 1e-12
        assert abs(r["R_unpolarized"] - (r_s + r_p) / 2) < 1e-12
        assert r["total_internal_reflection"] is False
    assert abs(wave(mode="fresnel", n1=1.0, n2=1.5)["R_s"] - 0.04) < 1e-15  # page example: about 4%
    at45 = wave(mode="fresnel", n1=1.0, n2=1.5, theta_deg=45.0)
    assert abs(at45["R_p"] - at45["R_s"] ** 2) < 1e-15  # page: R_p = R_s^2 at 45 degrees
    tir = wave(mode="fresnel", n1=1.5, n2=1.0, theta_deg=50.0)
    assert tir["total_internal_reflection"] is True and tir["R_s"] == 1.0 and tir["R_p"] == 1.0
    assert tir["theta_t_deg"] is None


def test_brewster_is_the_zero_of_r_p() -> None:
    for n1, n2 in ((1.0, 1.5), (1.33, 1.0), (1.0, 2.4)):
        r = wave(mode="brewster", n1=n1, n2=n2)
        crit = math.degrees(math.asin(min(1.0, n2 / n1))) - 1e-9
        zero = solve_root(lambda t: _amplitudes(n1, n2, t)[1], 1.0, crit, 1e-12)
        assert abs(r["theta_B_deg"] - zero) < 1e-9, (r, zero)
        assert abs(r["theta_B_deg"] + r["theta_t_deg"] - 90.0) < 1e-12
        assert r["R_p"] < 1e-30
    # examples of the page "Brewster's angle": glass about 56 degrees, water about 53 degrees
    assert round(wave(mode="brewster", n1=1.0, n2=1.5)["theta_B_deg"]) == 56
    assert round(wave(mode="brewster", n1=1.0, n2=1.33)["theta_B_deg"]) == 53


def test_double_slit_against_exact_path_difference() -> None:
    lam, d, length = 632.8e-9, 0.25e-3, 1.2
    for m in (1, 2, 5):
        r = wave(mode="double_slit", wavelength=lam, d=d, L=length, m=m)

        def delta(y: float) -> float:
            return math.hypot(length, y + d / 2) - math.hypot(length, y - d / 2) - m * lam

        y_exact = solve_root(delta, 0.0, 0.5, 1e-15)
        assert rel(r["y"], y_exact, 1e-6), (r, y_exact)
        assert rel(r["y_small_angle"], m * lam * length / d, 1e-13)
        assert rel(r["fringe_spacing"], lam * length / d, 1e-13)
    assert wave(mode="double_slit", wavelength=500e-9, d=2e-6)["m_max"] == 3
    page = wave(mode="double_slit", wavelength=0.6e-6, d=0.5e-3, L=1.0)  # page example: 1.2 mm
    assert round(page["fringe_spacing"] * 1000, 6) == 1.2
    assert "order" in bad(
        "wave_optics", {"mode": "double_slit", "wavelength": 500e-9, "d": 2e-6, "m": 5}
    )


def test_single_slit_minimum_against_phasor_integral() -> None:
    lam, a = 550e-9, 20e-6
    k = 2 * math.pi / lam

    def amplitude(theta: float) -> float:
        return quad(lambda x: math.cos(k * x * math.sin(theta)), -a / 2, a / 2, limit=200)[0]

    for m in (1, 2, 3):
        r = wave(mode="single_slit", wavelength=lam, a=a, m=m, L=1.0)
        guess = math.asin(m * lam / a)
        root = solve_root(amplitude, guess * 0.9, guess * 1.1, 1e-14)
        assert abs(math.radians(r["theta_deg"]) - root) < 1e-9, (r, root)
        if m == 1:
            assert rel(r["central_width"], 2 * math.tan(root), 1e-8)


def test_grating_normal_incidence_against_phasor_sum() -> None:
    lam, lines = 589.0e-9, 600.0
    d = 1e-3 / lines
    n_lines = 50
    for m in (1, 2):
        r = wave(mode="grating", wavelength=lam, lines_per_mm=lines, m=m, N=n_lines)
        th = math.radians(r["theta_deg"])
        phase = 2 * math.pi * d * math.sin(th) / lam
        s = abs(sum(np.exp(1j * phase * np.arange(n_lines)))) ** 2
        assert abs(s - n_lines**2) < 1e-6 * n_lines**2  # principal maximum: all phasors in step
        assert r["resolving_power"] == n_lines * m
        # Rayleigh on the N-slit pattern: the maximum of lambda + delta_lambda falls on the first zero of lambda
        lam2 = lam + r["delta_lambda"]
        th2 = math.asin(m * lam2 / d)
        phase2 = 2 * math.pi * d * math.sin(th2) / lam
        assert abs(sum(np.exp(1j * phase2 * np.arange(n_lines)))) ** 2 < 1e-12 * n_lines**2
    assert wave(mode="grating", wavelength=lam, d=d)["m_max"] == 2


def test_grating_oblique_incidence_convention() -> None:
    lam, d = 600e-9, 2e-6
    for th_i, m in ((20.0, 1), (20.0, -1), (-10.0, 2), (30.0, 0)):
        r = wave(mode="grating", wavelength=lam, d=d, m=m, theta_i_deg=th_i)
        # page convention: sin(theta_i) + sin(theta_m) = m lambda / d; m = 0 gives theta_m = -theta_i
        lhs = math.sin(math.radians(th_i)) + math.sin(math.radians(r["theta_deg"]))
        assert abs(lhs - m * lam / d) < 1e-14, (r, th_i, m)
    assert "order" in bad(
        "wave_optics", {"mode": "grating", "wavelength": lam, "d": d, "m": 4, "theta_i_deg": 0}
    )


def _film_r(n1: float, nf: float, n2: float, thickness: float, lam: float) -> float:
    """Normal-incidence reflectance of one layer with the characteristic matrix (all multiple reflections)."""
    delta = 2 * math.pi * nf * thickness / lam
    m = np.array(
        [[math.cos(delta), 1j * math.sin(delta) / nf], [1j * nf * math.sin(delta), math.cos(delta)]]
    )
    b, c = m @ np.array([1.0, n2])
    r = (n1 * b - c) / (n1 * b + c)
    return float(abs(r) ** 2)


def test_thin_film_extrema_against_characteristic_matrix() -> None:
    lam = 550e-9
    for n1, nf, n2 in ((1.0, 1.33, 1.0), (1.0, 1.38, 1.52), (1.0, 2.0, 1.5), (1.0, 1.2, 1.0)):
        r = wave(mode="thin_film", n1=n1, n_film=nf, n2=n2, wavelength=lam)
        for key, sign in (("d_constructive", -1), ("d_destructive", 1)):
            t = r[key]
            here = _film_r(n1, nf, n2, t, lam)
            side = (
                _film_r(n1, nf, n2, t * (1 - 1e-3), lam),
                _film_r(n1, nf, n2, t * (1 + 1e-3), lam),
            )
            assert all(sign * (s - here) > 0 for s in side), (key, r, here, side)  # local max / min
        assert rel(r["period"], lam / (2 * nf), 1e-13)
    ar = wave(mode="thin_film", n1=1.0, n_film=math.sqrt(1.52), n2=1.52, wavelength=lam)
    assert (
        _film_r(1.0, math.sqrt(1.52), 1.52, ar["d_destructive"], lam) < 1e-30
    )  # ideal quarter-wave coating
    soap = wave(mode="thin_film", n1=1.0, n_film=1.33, n2=1.0, wavelength=lam, d=lam / (4 * 1.33))
    assert soap["reflection"] == "constructive"


def test_rayleigh_against_airy_zero() -> None:
    lam, diam = 550e-9, 0.1
    r = wave(mode="rayleigh", wavelength=lam, D=diam, L=1000.0)
    x0 = float(jn_zeros(1, 1)[0])
    th = math.asin(x0 / math.pi * lam / diam)
    assert rel(math.radians(r["theta_deg"]), th, 1e-12)
    x = math.pi * diam / lam * math.sin(math.radians(r["theta_deg"]))
    assert abs(j1(x)) < 1e-12  # first dark ring of the Airy pattern
    assert rel(r["theta_arcsec"], math.degrees(th) * 3600, 1e-12)
    assert rel(r["x"], 1000.0 * math.tan(th), 1e-12)


def test_wave_optics_units_and_errors() -> None:
    r = wave(mode="double_slit", wavelength=[632.8, "nm"], d=[0.25, "mm"], L=[1.2, "m"])
    assert rel(r["fringe_spacing"], 632.8e-9 * 1.2 / 0.25e-3, 1e-12)
    assert "mode" in bad("wave_optics", {"mode": "laser"})
    assert "Keys not used" in bad(
        "wave_optics", {"mode": "malus", "I0": 1, "theta_deg": 0, "n1": 1}
    )
