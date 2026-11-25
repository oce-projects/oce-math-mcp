"""Newtonian gravitation (D88).

Checks independent of the server formulas:
- G from scipy.constants (CODATA) and the inverse-square law summed with numpy over point masses; the field must be
  minus the numerical gradient of the server's own potential;
- shell theorem: the field of a uniform solid sphere and of a hollow shell built from thin rings and shells and
  integrated with mpmath; the potential as minus the integral of the field from infinity; the binding energy as the
  integral of G m(r) dm / r; page values for the Earth as a uniform sphere (2.24e32 J, 37.5 MJ/kg, 60 %);
- ring on its axis against 8 point masses on the ring (exact by symmetry) and the maximum of the field as the root
  of its derivative;
- rotating body: gravity plus the centrifugal acceleration -omega x (omega x r) as numpy vectors;
- Gauss's law: the flux of a point mass through the 6 faces of a cube around it, integrated with scipy dblquad;
- potential difference against the mpmath integral of the field, also for radii 1 mm apart.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
from scipy import constants, integrate

from tests.conftest import call, ok

G = constants.G


def grav(**params: object) -> dict:
    return ok("eng_calc", operation="gravitation", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="gravitation", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def close(a: float, b: object, tol: float = 1e-12) -> bool:
    return math.isclose(a, float(b), rel_tol=tol)  # type: ignore[arg-type]


def test_two_bodies_earth_moon() -> None:
    m1, m2, r = 5.972e24, 7.342e22, 3.844e8
    res = grav(mode="two_bodies", m1=m1, m2=m2, r=r)
    assert close(res["F"], G * m1 * m2 / r**2) and close(res["U"], -res["F"] * r)
    assert close(res["g1"], G * m1 / r**2) and close(res["g2"], G * m2 / r**2)
    assert close(res["mu1"], G * m1) and close(res["mu2"], G * m2)
    assert 1.97e20 < res["F"] < 1.99e20  # the well-known 1.98e20 N


def test_point_masses_against_numpy_and_gradient() -> None:
    masses = [[5.0e3, 0.0, 0.0, 0.0], [2.0e3, 3.0, 1.0, -2.0], [8.0e2, -1.5, 4.0, 0.5]]
    point = [1.0, 2.0, 0.5]
    res = grav(mode="point_masses", masses=masses, point=point, m_test=2.5)
    arr = np.array(masses)
    d = arr[:, 1:] - np.array(point)
    dist = np.linalg.norm(d, axis=1)
    g = (G * arr[:, 0] / dist**3) @ d
    phi = -np.sum(G * arr[:, 0] / dist)
    assert np.allclose(res["g"], g, rtol=1e-12, atol=0) and close(
        res["g_magnitude"], np.linalg.norm(g)
    )
    assert close(res["potential"], phi) and close(res["U"], 2.5 * phi)
    assert np.allclose(res["F"], 2.5 * g, rtol=1e-12, atol=0)
    assert close(res["M_total"], 7.8e3)
    assert np.allclose(res["center_of_mass"], (arr[:, 0] @ arr[:, 1:]) / 7.8e3, rtol=1e-12)
    # the field is minus the gradient of the potential (central differences on the server's potential)
    h = 1e-4
    for axis in range(3):
        up, down = list(point), list(point)
        up[axis] += h
        down[axis] -= h
        p_up = grav(mode="point_masses", masses=masses, point=up)["potential"]
        p_dn = grav(mode="point_masses", masses=masses, point=down)["potential"]
        assert math.isclose(-(p_up - p_dn) / (2 * h), res["g"][axis], rel_tol=1e-6)
    assert "coincides" in bad({"mode": "point_masses", "masses": masses, "point": [3.0, 1.0, -2.0]})
    assert "masses" in bad({"mode": "point_masses", "masses": [], "point": point})


def _shell_element(a: mp.mpf, u: mp.mpf, s: float) -> mp.mpf:
    """Field towards the centre at distance s from the ring of a shell of radius a at cos(theta) = u, per unit
    surface density: ring area 2 pi a^2 du, axial distance s - a u, distance^2 = s^2 + a^2 - 2 a s u."""
    return 2 * mp.pi * a**2 * (s - a * u) / (s * s + a * a - 2 * a * s * u) ** mp.mpf(1.5)


def test_shell_theorem_from_rings() -> None:
    mp.mp.dps = 15
    big_m, big_r = 1.0e20, 1.0e5
    rho = big_m / (4 / 3 * math.pi * big_r**3)
    for s in (0.75e5, 1.2e5, 1.6e5):
        # solid sphere = shells of rings, a 2D integral over the radius a and u = cos(theta); the integrand is
        # singular where the shell passes the point, so the radius is split there
        cuts = [0, s, big_r] if s < big_r else [0, big_r]
        solid = G * rho * mp.quad(lambda a, u, s=s: _shell_element(a, u, s), cuts, [-1, 1])
        res = grav(mode="sphere", M=big_m, R=big_r, r=s)
        assert close(res["g"], solid, 1e-11)
    for s in (0.3e5, 0.75e5, 1.6e5):
        # hollow shell of radius R: a 1D integral over its rings
        shell = (
            G
            * big_m
            / (4 * mp.pi * big_r**2)
            * mp.quad(lambda u, s=s: _shell_element(big_r, u, s), [-1, 1])
        )
        hollow = grav(mode="sphere", M=big_m, R=big_r, r=s, shell=True)
        assert math.isclose(hollow["g"], shell, rel_tol=1e-11, abs_tol=1e-14)
        assert "binding_energy" not in hollow and "rho" in hollow
        # potential = - integral of the field from r to infinity
        res = grav(mode="sphere", M=big_m, R=big_r, r=s)
        g_of = lambda x: G * big_m * (x / big_r**3 if x < big_r else 1 / x**2)  # noqa: E731
        pts = [s, big_r, mp.inf] if s < big_r else [s, mp.inf]
        assert close(res["potential"], -mp.quad(g_of, pts), 1e-12)
        g_shell = lambda x: 0 if x < big_r else G * big_m / x**2  # noqa: E731
        assert close(hollow["potential"], -mp.quad(g_shell, pts), 1e-12)
    surface = grav(mode="sphere", M=big_m, R=big_r, r=big_r)
    assert close(surface["g"], surface["g_surface"]) and close(surface["rho"], rho)
    assert close(grav(mode="sphere", rho=rho, R=big_r, r=2e5)["g"], G * big_m / 4e10)
    # binding energy = - integral of G m(r) dm / r for a uniform sphere
    bind = -mp.quad(
        lambda x: G * (4 / 3 * mp.pi * x**3 * rho) * 4 * mp.pi * x**2 * rho / x, [0, big_r]
    )
    assert close(surface["binding_energy"], bind)
    earth = grav(mode="sphere", M=5.97e24, R=6.37e6, r=6.37e6)
    assert round(-earth["binding_energy"] / 1e32, 2) == 2.24
    assert round(-earth["binding_energy_per_kg"] / 1e6, 1) == 37.5
    assert close(earth["binding_energy_per_kg"] / earth["potential_surface"], 0.6)
    assert "binding_energy" not in grav(mode="sphere", mu=G * 5.97e24, R=6.37e6, r=7e6)


def test_ring_on_axis() -> None:
    mp.mp.dps = 25
    m, radius = 4.0e9, 2.0
    for z in (0.0, 0.7, 1.4142, 5.0):
        res = grav(mode="ring", M=m, R=radius, z=z)
        ring_pts = [
            [m / 8, radius * math.cos(k * math.pi / 4), radius * math.sin(k * math.pi / 4), 0.0]
            for k in range(8)
        ]
        pm = grav(mode="point_masses", masses=ring_pts, point=[0.0, 0.0, z])
        assert math.isclose(res["g"], -pm["g"][2], rel_tol=1e-12, abs_tol=1e-30)
        assert close(res["potential"], pm["potential"])
    z_max = mp.findroot(lambda zz: mp.diff(lambda t: t / (t**2 + radius**2) ** 1.5, zz), 1.0)
    res = grav(mode="ring", M=m, R=radius, z=1.0)
    assert close(res["z_max"], z_max) and close(res["g_max"], G * m * z_max / (z_max**2 + 4) ** 1.5)


def test_rotating_body_against_vectors() -> None:
    gm, radius, omega = 3.986004418e14, 6.371e6, 7.2921159e-5
    w = np.array([0.0, 0.0, omega])
    for lat in (0.0, 30.0, 45.0, 60.0, 90.0, -30.0, -60.0, -90.0):
        t = math.radians(lat)
        r_vec = radius * np.array([math.cos(t), 0.0, math.sin(t)])
        g_vec = -gm / radius**3 * r_vec - np.cross(w, np.cross(w, r_vec))
        up = r_vec / radius
        north = np.array([-math.sin(t), 0.0, math.cos(t)])
        to_equator = -north if lat >= 0 else north  # along the meridian, towards the equator
        res = grav(mode="rotating_body", mu=gm, R=radius, omega=omega, latitude_deg=lat)
        assert close(res["g_eff"], np.linalg.norm(g_vec))
        assert close(res["g_radial"], -g_vec @ up)
        assert math.isclose(res["g_tangential"], g_vec @ to_equator, rel_tol=1e-9, abs_tol=1e-15)
        assert res["g_tangential"] >= 0 and res["deflection_deg"] >= 0
        ang = math.degrees(math.atan2(g_vec @ to_equator, -g_vec @ up))
        assert math.isclose(res["deflection_deg"], ang, rel_tol=1e-9, abs_tol=1e-12)
    eq = grav(mode="rotating_body", mu=gm, R=radius, omega=omega, latitude_deg=0.0)
    assert close(eq["g_eff"], gm / radius**2 - omega**2 * radius)
    by_period = grav(
        mode="rotating_body", mu=gm, R=radius, T_rot=2 * math.pi / omega, latitude_deg=0.0
    )
    assert close(by_period["g_eff"], eq["g_eff"])


def test_gauss_flux_through_a_cube() -> None:
    m, src = 3.0e8, np.array([0.2, -0.1, 0.3])  # off-centre mass inside the cube [-1, 1]^3

    def face(axis: int, sign: float) -> float:
        def normal_field(v: float, u: float) -> float:
            pt = np.zeros(3)
            pt[axis] = sign
            pt[[i for i in range(3) if i != axis]] = (u, v)
            d = pt - src
            return float(-G * m * (d[axis] * sign) / np.linalg.norm(d) ** 3)

        return integrate.dblquad(normal_field, -1, 1, -1, 1, epsabs=0, epsrel=1e-11)[0]

    flux = sum(face(ax, sg) for ax in range(3) for sg in (1.0, -1.0))
    res = grav(mode="gauss", M=m)
    assert close(res["Phi_G"], flux, 1e-9) and close(res["Phi_G"], -4 * math.pi * G * m)
    assert close(grav(mode="gauss", Phi_G=res["Phi_G"])["M_enclosed"], m)


def test_potential_difference_and_uniform_field() -> None:
    mp.mp.dps = 30
    gm = 3.986004418e14
    for r1, r2 in ((6.371e6, 6.371e6 + 1e-3), (6.371e6, 6.771e6), (7e6, 4.2e7)):
        res = grav(mode="potential_difference", mu=gm, r1=r1, r2=r2, m=10.0)
        ref = mp.quad(lambda x: gm / x**2, [r1, r2])
        assert close(res["dPhi"], ref) and close(res["W"], 10 * ref)
        approx = 10 * mp.mpf(gm) / mp.mpf(r1) ** 2 * (mp.mpf(r2) - r1)
        assert close(res["W_uniform"], approx)
        assert close(res["uniform_field_error"], (approx - 10 * ref) / (10 * ref), 1e-12)
    assert "r1" in bad({"mode": "potential_difference", "mu": gm, "r1": 7e6, "r2": 7e6})
    assert "exactly one" in bad(
        {"mode": "potential_difference", "mu": gm, "M": 1e24, "r1": 1.0, "r2": 2.0}
    )
    assert "Keys not used" in bad({"mode": "gauss", "M": 1.0, "r": 2.0})
