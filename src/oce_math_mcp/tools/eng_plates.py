"""eng_calc: thin plates in bending, Kirchhoff-Love theory (D68).

Source: Wikipedia page "Bending of plates" in the local offline library.
- Flexural rigidity D = E H^3 / (12 (1 - nu^2)), H thickness.
- Circular plate, radius a, uniform load q, clamped edge: w(r) = (q / (64 D)) (a^2 - r^2)^2 (sign: deflection along q),
  M_rr = -(q/16) [(1 + nu) a^2 - (3 + nu) r^2], M_thth = -(q/16) [(1 + nu) a^2 - (1 + 3 nu) r^2], edge moment q a^2 / 8,
  maximum radial stress 3 q a^2 / (4 H^2) at the edge.
- General axisymmetric solution w(r) = q r^4 / (64 D) + C2 r^2 + C4 (terms in ln r are excluded by a finite centre).
- Rectangular plate a x b, simply supported, uniform load q0 (Navier, odd m and n):
  w = (16 q0 / (pi^6 D)) sum sum sin(m pi x/a) sin(n pi y/b) / (m n (m^2/a^2 + n^2/b^2)^2),
  Mx = (16 q0 / pi^4) sum sum (m^2/a^2 + nu n^2/b^2) sin sin / (m n (m^2/a^2 + n^2/b^2)^2), My with the roles swapped.
Derived here and checked in the tests: simply supported circular plate from the general solution with w(a) = 0 and
M_rr(a) = 0 (centre deflection (5 + nu) q a^4 / (64 (1 + nu) D), centre moment (3 + nu) q a^2 / 16); bending stress
6 M / H^2 at the surface (the page gives it for the clamped circle).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, only, op, units_for

_SHAPES: dict[str, set[str]] = {"circle": {"a"}, "rectangle": {"a", "b"}}
_COMMON = {"shape", "support", "h", "E", "nu", "q"}
_UNITS = {
    "D": "N*m",
    "w_max": "m",
    "M_max": "N*m/m",
    "M_center": "N*m/m",
    "M_edge": "N*m/m",
    "sigma_max": "Pa",
}
_UNITS |= {
    "Mx_center": "N*m/m",
    "My_center": "N*m/m",
    "sigma_x_center": "Pa",
    "sigma_y_center": "Pa",
}
_ODD = np.arange(
    1, 2000, 2, dtype=float
)  # 1000 odd terms per direction: the moment series tails are ~1e-10


@op("plate_bending", tuple(sorted(_COMMON | {"a", "b"})))
def _plate(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    shape = choice(p, "shape", tuple(_SHAPES))
    only(p, shape, _SHAPES[shape] | _COMMON, field="shape")
    h, e = need(p, "h", "length"), need(p, "E", "pressure")
    nu = need(p, "nu", None, positive=False)
    if not 0 <= nu < 0.5:
        raise ToolError("INVALID_ARGUMENT", "params.nu (Poisson ratio) must be in [0, 0.5).")
    q = need(p, "q", "pressure")
    d = e * h**3 / (12 * (1 - nu * nu))
    out = _circle(p, d, h, nu, q, ctx) if shape == "circle" else _rectangle(p, d, h, nu, q, ctx)
    ctx.warn(
        "Kirchhoff-Love thin plate: small deflection (w much less than h) and h much less than the span."
    )
    if out["w_max"] > 0.2 * h:
        ctx.warn(
            "w_max > 0.2 h: membrane effects are no longer negligible, the linear result overestimates w."
        )
    ctx.units = units_for(out, _UNITS)
    return out


def _circle(p: dict[str, Any], d: float, h: float, nu: float, q: float, ctx: Ctx) -> dict[str, Any]:
    support = choice(p, "support", ("clamped", "simply_supported"))
    a = need(p, "a", "length")
    if support == "clamped":
        w_max = q * a**4 / (64 * d)
        m_center = (1 + nu) * q * a * a / 16
        m_edge = q * a * a / 8
        ctx.formula = (
            "D = E H^3 / (12 (1 - nu^2)); w(r) = q (a^2 - r^2)^2 / (64 D); M_rr = -(q/16)[(1 + nu) a^2 - (3 + nu) r^2]; "
            "sigma_max = 3 q a^2 / (4 H^2) at the edge"
        )
    else:
        w_max = (5 + nu) * q * a**4 / (64 * (1 + nu) * d)
        m_center = (3 + nu) * q * a * a / 16
        m_edge = 0.0
        ctx.formula = (
            "D = E H^3 / (12 (1 - nu^2)); w(r) = q r^4 / (64 D) + C2 r^2 + C4 with w(a) = 0, M_rr(a) = 0: "
            "w(0) = (5 + nu) q a^4 / (64 (1 + nu) D), M(0) = (3 + nu) q a^2 / 16; sigma = 6 M / H^2"
        )
    m_max = max(m_center, m_edge)
    sigma = 6 * m_max / (h * h)
    where = "edge" if m_edge > m_center else "centre"
    ctx.formatted = (
        f"Circular plate {support.replace('_', ' ')}: centre deflection {fmt_sig(w_max * 1000, 6)} mm, "
        f"maximum stress {fmt_sig(sigma / 1e6, 6)} MPa at the {where}."
    )
    return {
        "D": d,
        "w_max": w_max,
        "M_center": m_center,
        "M_edge": m_edge,
        "M_max": m_max,
        "sigma_max": sigma,
        "sigma_max_at": where,
    }


def _rectangle(
    p: dict[str, Any], d: float, h: float, nu: float, q: float, ctx: Ctx
) -> dict[str, Any]:
    support = choice(p, "support", ("simply_supported",))
    a, b = need(p, "a", "length"), need(p, "b", "length")
    m, n = np.meshgrid(_ODD, _ODD, indexing="ij")
    sign = np.sin(m * math.pi / 2) * np.sin(n * math.pi / 2)  # value of sin sin at the centre: +-1
    den = m * n * (m * m / (a * a) + n * n / (b * b)) ** 2
    w_c = 16 * q / (math.pi**6 * d) * float(np.sum(sign / den))
    mx = 16 * q / math.pi**4 * float(np.sum(sign * (m * m / (a * a) + nu * n * n / (b * b)) / den))
    my = 16 * q / math.pi**4 * float(np.sum(sign * (n * n / (b * b) + nu * m * m / (a * a)) / den))
    ctx.formula = (
        "Navier: w = (16 q0 / (pi^6 D)) sum sum sin(m pi x/a) sin(n pi y/b) / (m n (m^2/a^2 + n^2/b^2)^2), "
        "Mx = (16 q0 / pi^4) sum sum (m^2/a^2 + nu n^2/b^2) sin sin / (m n (...)^2), odd m, n up to 1999; "
        "sigma = 6 M / H^2"
    )
    m_max = max(mx, my)
    ctx.formatted = (
        f"Rectangular plate {support.replace('_', ' ')} {fmt_sig(a, 6)} m x {fmt_sig(b, 6)} m: centre deflection "
        f"{fmt_sig(w_c * 1000, 6)} mm, stress {fmt_sig(6 * mx / h**2 / 1e6, 6)} MPa (x) and "
        f"{fmt_sig(6 * my / h**2 / 1e6, 6)} MPa (y) at the centre."
    )
    return {
        "D": d,
        "w_max": w_c,
        "Mx_center": mx,
        "My_center": my,
        "M_max": m_max,
        "sigma_x_center": 6 * mx / h**2,
        "sigma_y_center": 6 * my / h**2,
        "sigma_max": 6 * m_max / h**2,
        "sigma_max_at": "centre",
    }
