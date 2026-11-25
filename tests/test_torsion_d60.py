"""Torsion of non-circular sections (D60).

Sources: Wikipedia "Torsion (mechanics)" (table of sections: exact formulas, thin strip, Bredt) and
"Torsion constant" (beta table of the rectangle, Roark formula with error <= 4%).
Checks independent of the server formulas:
- ellipse and equilateral triangle: Prandtl stress function written in the test (Laplacian -2 checked by
  finite differences), J = 2 int phi dA with dblquad, tau = |grad phi| on the boundary;
- rectangle: Prandtl equation solved by finite differences (grids 40 and 80, Richardson);
- thin walls: shear strain energy against the work of the torque;
- thin closed tube against the exact circular annulus (difference (t / (2 r_m))^2);
- circle against the operation shaft_torsion_bending.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import dblquad

from tests.conftest import call, ok


def tors(**params: object) -> dict:
    return ok("eng_calc", operation="torsion_section", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="torsion_section", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def test_circle_and_annulus_against_shaft_torsion() -> None:
    for d, d_i in ((0.05, None), (0.08, 0.06)):
        shaft = {"d": d, "T": 1500.0, "L": 2.0, "G": 80e9}
        if d_i is None:
            r = tors(shape="circle", D=d, T=1500.0, L=2.0, G=80e9)
            assert rel(tors(shape="circle", R=d / 2, T=1500.0)["J"], r["J"], 1e-15)
        else:
            shaft["d_i"] = d_i
            r = tors(shape="annulus", R_e=d / 2, R_i=d_i / 2, T=1500.0, L=2.0, G=80e9)
        s = ok("eng_calc", operation="shaft_torsion_bending", params=shaft)
        for k in ("J", "tau_max", "theta"):
            assert rel(r[k], s[k], 1e-12), (k, r[k], s[k])
    u = tors(shape="circle", D=[50, "mm"], T=[1.5, "kNm"], L=[2000, "mm"], G=[80, "GPa"])
    assert rel(u["theta"], tors(shape="circle", D=0.05, T=1500.0, L=2.0, G=80e9)["theta"], 1e-12)


def test_ellipse_against_prandtl_function() -> None:
    p, q = 0.06, 0.025
    c = (
        p * p * q * q / (p * p + q * q)
    )  # phi = c (1 - x^2/p^2 - y^2/q^2): lap(phi) = -2 (G theta = 1)
    j_num, _ = dblquad(
        lambda y, x: 2 * c * (1 - x * x / p**2 - y * y / q**2),
        -p,
        p,
        lambda x: -q * math.sqrt(1 - (x / p) ** 2),
        lambda x: q * math.sqrt(1 - (x / p) ** 2),
        epsabs=0,
        epsrel=1e-12,
    )
    s = np.linspace(0, 2 * math.pi, 20001)
    grad = np.hypot(
        2 * c * np.cos(s) / p, 2 * c * np.sin(s) / q
    )  # |grad phi| on the boundary (p cos s, q sin s)
    r = tors(shape="ellipse", p=p, q=q, T=1.0)
    assert rel(r["J"], j_num, 1e-10)
    assert rel(r["tau_max"], grad.max() / j_num, 1e-8)  # T = 1: G theta = 1 / J
    assert rel(tors(shape="ellipse", p=q, q=p, T=1.0)["J"], r["J"], 1e-15)  # semi-axes in any order
    assert rel(tors(shape="ellipse", p=0.03, q=0.03)["J"], tors(shape="circle", R=0.03)["J"], 1e-14)


def test_equilateral_triangle_against_prandtl_function() -> None:
    a = 0.05
    h = math.sqrt(3) / 2 * a
    normals = [
        np.array([math.cos(t), math.sin(t)])
        for t in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)
    ]

    def dist(
        x: float, y: float
    ) -> list[float]:  # distances from the three sides, centroid at the origin
        return [h / 3 - (n[0] * x + n[1] * y) for n in normals]

    def phi(x: float, y: float) -> float:
        d1, d2, d3 = dist(x, y)
        return 2 / h * d1 * d2 * d3

    e = 1e-4  # numerical Laplacian at interior points: must be -2
    for x, y in ((0.0, 0.0), (0.004, -0.003), (-0.006, 0.002)):
        lap = (phi(x + e, y) + phi(x - e, y) + phi(x, y + e) + phi(x, y - e) - 4 * phi(x, y)) / e**2
        assert abs(lap + 2) < 1e-6
    # triangle with one side at the top (y = h/3) and the vertex at the bottom (y = -2h/3)
    j_num, _ = dblquad(
        lambda x, y: 2 * phi(x, y),
        -2 * h / 3,
        h / 3,
        lambda y: -(y + 2 * h / 3) / math.sqrt(3),
        lambda y: (y + 2 * h / 3) / math.sqrt(3),
        epsabs=0,
        epsrel=1e-12,
    )
    xs = np.linspace(-a / 2, a / 2, 20001)
    grad_top = [
        abs((phi(x, h / 3 - 1e-7) - phi(x, h / 3)) / 1e-7) for x in xs
    ]  # |grad phi| on the top side
    r = tors(shape="equilateral_triangle", a=a, T=1.0)
    assert rel(r["J"], j_num, 1e-10)
    assert rel(r["tau_max"], max(grad_top) / j_num, 1e-5)


def _prandtl_fd(b: float, t: float, m: int) -> tuple[float, float]:
    h = t / m
    nx, ny = round(b / h) - 1, m - 1
    lx = sp.diags([np.ones(nx - 1), -2 * np.ones(nx), np.ones(nx - 1)], [-1, 0, 1]) / h**2
    ly = sp.diags([np.ones(ny - 1), -2 * np.ones(ny), np.ones(ny - 1)], [-1, 0, 1]) / h**2
    lap = sp.kron(sp.identity(ny), lx) + sp.kron(ly, sp.identity(nx))
    phi = spla.spsolve(lap.tocsc(), -2 * np.ones(nx * ny)).reshape(ny, nx)
    col = nx // 2
    return 2 * phi.sum() * h * h, (4 * phi[0, col] - phi[1, col]) / (2 * h)


def test_rectangle_against_finite_differences_and_tables() -> None:
    for ratio in (1.0, 2.0, 5.0):
        t = 0.01
        (j1, t1), (j2, t2) = _prandtl_fd(ratio * t, t, 40), _prandtl_fd(ratio * t, t, 80)
        j_fd, tau_fd = (4 * j2 - j1) / 3, (4 * t2 - t1) / 3  # G theta = 1
        r = tors(shape="rectangle", b=ratio * t, t=t, T=1.0)
        assert rel(r["J"], j_fd, 1e-5) and rel(r["tau_max"], tau_fd / j_fd, 1e-5), ratio
        assert rel(
            tors(shape="rectangle", b=t, t=ratio * t)["J"], r["J"], 1e-15
        )  # sides in any order
    # beta table of the source "Torsion constant" (3 digits) and Roark formula (error <= 4%)
    table = {
        1.0: 0.141,
        1.5: 0.196,
        2.0: 0.229,
        2.5: 0.249,
        3.0: 0.263,
        4.0: 0.281,
        5.0: 0.291,
        6.0: 0.299,
        10.0: 0.312,
    }
    for ratio, beta in table.items():
        j = tors(shape="rectangle", b=ratio, t=1.0)["J"]
        assert abs(j / ratio - beta) < 1e-3, ratio
        a, bb = ratio / 2, 0.5
        roark = a * bb**3 * (16 / 3 - 3.36 * bb / a * (1 - bb**4 / (12 * a**4)))
        assert rel(roark, j, 0.04)
    assert rel(
        tors(shape="rectangle", b=2.0, t=2.0)["J"], 2.25, 1e-3
    )  # square: J ~ 2.25 a^4 (a = half side)


def test_thin_open_walls_energy() -> None:
    walls = [[0.2, 0.012], [0.3, 0.008], [0.2, 0.012]]  # I-section: flanges and web
    T, G = 300.0, 80e9
    r = tors(shape="thin_open", walls=walls, T=T, G=G)
    assert rel(r["J"], sum(b * t**3 for b, t in walls) / 3, 1e-14)
    # tau linear across the thickness (maximum at the surface): energy sum b int tau^2 / (2G) dy = T theta' / 2
    energy = sum(
        b * (tau**2 / (2 * G)) * t / 3 for (b, t), tau in zip(walls, r["tau_walls"], strict=True)
    )
    assert rel(energy, T * r["twist_rate"] / 2, 1e-12)
    assert rel(r["tau_max"], max(r["tau_walls"]), 1e-15)
    strip = tors(shape="thin_open", walls=[[0.5, 0.01]], T=T)
    assert rel(strip["tau_max"], 3 * T / (0.5 * 0.01**2), 1e-14)  # thin strip: 3T / (a s^2)
    # thin strip against the exact rectangle at b/t = 50: difference about 0.63 t / b
    exact = tors(shape="rectangle", b=0.5, t=0.01)["J"]
    assert 0.012 < 1 - exact / strip["J"] < 0.013


def test_thin_closed_bredt() -> None:
    r_m, t = 0.1, 0.002
    tube = tors(shape="thin_closed", A_m=math.pi * r_m**2, walls=[[2 * math.pi * r_m, t]], T=1000.0)
    ann = tors(shape="annulus", R_e=r_m + t / 2, R_i=r_m - t / 2, T=1000.0)
    assert rel(ann["J"] / tube["J"] - 1, (t / (2 * r_m)) ** 2, 1e-8)  # exact identity
    # box section with walls of different thickness: constant shear flow, energy against work
    w, hh, t1, t2, T, G = 0.3, 0.2, 0.004, 0.006, 5000.0, 80e9
    walls = [[w, t1], [hh, t2], [w, t1], [hh, t2]]
    box = tors(shape="thin_closed", A_m=w * hh, walls=walls, T=T, G=G)
    q = T / (2 * w * hh)
    assert rel(box["shear_flow"], q, 1e-14)
    assert all(
        rel(tau * tt, q, 1e-14) for tau, (_, tt) in zip(box["tau_walls"], walls, strict=True)
    )
    energy = sum(
        tau**2 / (2 * G) * tt * ln for tau, (ln, tt) in zip(box["tau_walls"], walls, strict=True)
    )
    assert rel(energy, T * box["twist_rate"] / 2, 1e-12)
    assert rel(box["tau_max"], q / t1, 1e-14)


def test_errors() -> None:
    assert "R_i" in bad({"shape": "annulus", "R_e": 0.02, "R_i": 0.03})
    assert "shape" in bad({"shape": "hexagon", "a": 1.0})
    assert "walls" in bad({"shape": "thin_open", "walls": [[0.1]]})
    assert "A_m" in bad({"shape": "thin_closed", "walls": [[0.1, 0.01]]})
    assert "Missing params.a" in bad({"shape": "equilateral_triangle"})
