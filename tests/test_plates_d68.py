"""Thin plates (D68).

Checks independent of the server formulas:
- rectangular simply supported plate: the biharmonic equation solved by finite differences as two Poisson problems
  (w = 0 and Laplacian(w) = 0 on the edge), grids n and 2n with Richardson extrapolation;
- circular plates: the general axisymmetric solution w = C1 + C2 r^2 + q r^4 / (64 D) with the two constants found
  from the edge conditions by a 2 x 2 linear system (clamped: w = 0, w' = 0; simply supported: w = 0, M_r = 0),
  and the plate equation D (w'''' + 2 w'''/r - w''/r^2 + w'/r^3) = q checked by finite differences.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tests.conftest import call, ok


def plate(**params: object) -> dict:
    return ok("eng_calc", operation="plate_bending", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="plate_bending", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def _rect_fd(a: float, b: float, n: int, q_over_d: float) -> tuple[float, float, float]:
    """Simply supported a x b plate with uniform load, two Poisson problems: centre deflection and the second
    derivatives w_xx, w_yy at the centre (central differences on the grid)."""
    hx, hy = a / n, b / n
    m = n - 1
    tx = sp.diags([np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)], [-1, 0, 1]) / hx**2
    ty = sp.diags([np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)], [-1, 0, 1]) / hy**2
    lap = (sp.kron(sp.identity(m), tx) + sp.kron(ty, sp.identity(m))).tocsc()
    u = spla.spsolve(lap, np.full(m * m, q_over_d))
    w = spla.spsolve(lap, u)
    g = np.asarray(w, dtype=float).reshape(m, m)  # g[j, i]: row j along y, column i along x
    c = m // 2
    w_xx = (g[c, c + 1] - 2 * g[c, c] + g[c, c - 1]) / hx**2
    w_yy = (g[c + 1, c] - 2 * g[c, c] + g[c - 1, c]) / hy**2
    return float(g[c, c]), float(w_xx), float(w_yy)


def test_rectangular_simply_supported_against_finite_differences() -> None:
    e, nu, h, q = 70e9, 0.33, 0.004, 2000.0
    d = e * h**3 / (12 * (1 - nu**2))
    for a, b in ((1.0, 1.0), (1.2, 0.6), (0.5, 1.5)):
        r = plate(shape="rectangle", support="simply_supported", a=a, b=b, h=h, E=e, nu=nu, q=q)
        assert rel(r["D"], d, 1e-13)
        coarse, fine = _rect_fd(a, b, 40, q / d), _rect_fd(a, b, 80, q / d)
        w_c, w_xx, w_yy = ((4 * f - c) / 3 for f, c in zip(fine, coarse, strict=True))
        assert rel(r["w_max"], w_c, 1e-6), (r, coarse, fine)
        assert rel(r["Mx_center"], -d * (w_xx + nu * w_yy), 1e-4), (
            r,
            w_xx,
            w_yy,
        )  # Mx = -D (w_xx + nu w_yy)
        assert rel(r["My_center"], -d * (w_yy + nu * w_xx), 1e-4)
        assert rel(r["sigma_max"], 6 * max(r["Mx_center"], r["My_center"]) / h**2, 1e-13)


def _circle_constants(a: float, q_over_d: float, nu: float, clamped: bool) -> tuple[float, float]:
    """C1, C2 of w = C1 + C2 r^2 + q r^4 / (64 D) from the edge conditions at r = a."""
    k = q_over_d / 64
    if clamped:  # w(a) = 0, w'(a) = 0
        mat = np.array([[1.0, a * a], [0.0, 2 * a]])
        rhs = np.array([-k * a**4, -4 * k * a**3])
    else:  # w(a) = 0, M_r ~ w'' + nu w'/r = 0
        mat = np.array([[1.0, a * a], [0.0, 2 + 2 * nu]])
        rhs = np.array([-k * a**4, -(12 * k * a * a + nu * 4 * k * a * a)])
    c1, c2 = np.linalg.solve(mat, rhs)
    return float(c1), float(c2)


def test_circular_plates_against_axisymmetric_solution() -> None:
    e, nu, h, q, a = 200e9, 0.3, 0.01, 50e3, 0.4
    d = e * h**3 / (12 * (1 - nu**2))
    for support, clamped in (("clamped", True), ("simply_supported", False)):
        r = plate(shape="circle", support=support, a=a, h=h, E=e, nu=nu, q=q)
        c1, c2 = _circle_constants(a, q / d, nu, clamped)
        assert rel(r["w_max"], c1, 1e-12), (r, c1)  # maximum at the centre
        # the plate equation holds for w = C1 + C2 r^2 + q r^4 / (64 D): finite differences at r = a / 2
        rr, dr = a / 2, 1e-3 * a

        def w(x: float) -> float:
            return c1 + c2 * x * x + q / d * x**4 / 64

        d1 = (w(rr + dr) - w(rr - dr)) / (2 * dr)
        d2 = (w(rr + dr) - 2 * w(rr) + w(rr - dr)) / dr**2
        d3 = (w(rr + 2 * dr) - 2 * w(rr + dr) + 2 * w(rr - dr) - w(rr - 2 * dr)) / (2 * dr**3)
        d4 = (w(rr + 2 * dr) - 4 * w(rr + dr) + 6 * w(rr) - 4 * w(rr - dr) + w(rr - 2 * dr)) / dr**4
        assert rel(d * (d4 + 2 * d3 / rr - d2 / rr**2 + d1 / rr**3), q, 1e-4)
        # moments from the solution: M_r = -D (w'' + nu w'/r), M_t = -D (w'/r + nu w''), largest magnitude on 0 < r <= a
        rs = np.linspace(a * 1e-6, a, 20001)
        w1, w2 = 2 * c2 * rs + q / d * rs**3 / 16, 2 * c2 + 3 * q / d * rs**2 / 16
        m_abs = np.max(np.abs(np.concatenate([d * (w2 + nu * w1 / rs), d * (w1 / rs + nu * w2)])))
        assert rel(r["M_max"], float(m_abs), 1e-9), (r, m_abs)
        assert rel(r["sigma_max"], 6 * float(m_abs) / h**2, 1e-9)
    # clamped edge: the page gives the maximum radial stress 3 q a^2 / (4 H^2)
    cl = plate(shape="circle", support="clamped", a=a, h=h, E=e, nu=nu, q=q)
    assert rel(cl["sigma_max"], 3 * q * a * a / (4 * h * h), 1e-13) and cl["sigma_max_at"] == "edge"
    assert "Keys not used" in bad(
        {
            "shape": "circle",
            "support": "clamped",
            "a": a,
            "b": 1.0,
            "h": h,
            "E": e,
            "nu": nu,
            "q": q,
        }
    )
