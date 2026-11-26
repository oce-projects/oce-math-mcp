"""eng_calc: electrostatics and capacitance (D55).

Sources (Wikipedia pages):
- «Coulomb's law»: F = k_e q1 q2 / r^2, k_e = 1 / (4 pi eps0) = mu0 c^2 / (4 pi), field |E| = k_e |Q| / r^2,
  superposition for several charges;
- «Capacitance», table «Capacitance of simple systems»: parallel-plate capacitor eps A / d, concentric
  cylinders 2 pi eps l / ln(R2 / R1), eccentric cylinders, two parallel wires, wire parallel to a
  wall, concentric spheres, isolated sphere 4 pi eps a, disc 8 eps a; energy Q V / 2 = C V^2 / 2;
- «Electric dipole moment» (D114): p = q d (d from -q to +q); in a uniform field U = -p . E and tau = p x E;
  ideal dipole at the centre c, R = x - c: potential V = k_e p . R / R^3 and field
  E = k_e (3 (p . R^) R^ - p) / R^3, the limit of two charges +-q at distances much larger than d.
CODATA 2022 constants from scipy (eps0 = 8.8541878188e-12 F/m). Homogeneous medium with eps = eps_r eps0.
"""

from __future__ import annotations

import math
from typing import Any

import scipy.constants as _sc

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import (
    acosh1p,
    choice,
    fmt_sig,
    log_ratio,
    need,
    one_of,
    only,
    op,
    qty,
    range_guard,
    units_for,
)

EPS0 = float(_sc.epsilon_0)
K_E = 1 / (4 * math.pi * EPS0)

_CAP_GEOMETRIES: dict[str, set[str]] = {
    "parallel_plate": {"A", "d"},
    "coaxial": {"R_inner", "R_outer", "L"},
    "eccentric_cylinders": {"R_inner", "R_outer", "d", "L"},
    "parallel_wires": {"a", "d", "L"},
    "wire_wall": {"a", "d", "L"},
    "concentric_spheres": {"R_inner", "R_outer"},
    "sphere": {"R"},
    "disc": {"R"},
}
_MODES: dict[str, set[str]] = {
    "coulomb": {"q1", "q2", "r", "eps_r"},
    "point_charges": {"charges", "point", "q", "eps_r"},
    "capacitor": {"geometry", "eps_r", "V", "Q"} | set().union(*_CAP_GEOMETRIES.values()),
    "combination": {"values", "connection"},
    "dipole": {"p", "q", "d", "E", "point", "center", "eps_r"},
}
# |d| / |R| above which the field of the two charges is not that of the ideal dipole
_DIPOLE_NEAR = 0.1
_UNITS = {
    "k_e": "N*m2/C2",
    "F": "N",
    "E1": "V/m",
    "V1": "V",
    "U": "J",
    "E": "V/m",
    "E_mag": "V/m",
    "V": "V",
    "U_system": "J",
    "F_mag": "N",
    "C": "F",
    "Q": "C",
    "C_eq": "F",
    "p": "C*m",
    "p_magnitude": "C*m",
    "tau": "N*m",
    "tau_magnitude": "N*m",
    "angle_deg": "deg",
    "R_vector": "m",
    "R_magnitude": "m",
    "E_dipole": "V/m",
    "E_dipole_mag": "V/m",
    "V_dipole": "V",
}


def _eps_r(p: dict[str, Any]) -> float:
    return need(p, "eps_r", None) if p.get("eps_r") is not None else 1.0


@op("electrostatics", ("mode", *sorted(set().union(*_MODES.values()))))
def _electrostatics(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    if mode == "capacitor":
        geo = choice(p, "geometry", tuple(_CAP_GEOMETRIES))
        only(p, mode, _CAP_GEOMETRIES[geo] | {"geometry", "eps_r", "V", "Q"})
        out = _capacitor(geo, p, ctx)
    else:
        only(p, mode, _MODES[mode])
        out = {
            "coulomb": _coulomb,
            "point_charges": _point_charges,
            "combination": _combination,
            "dipole": _dipole,
        }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _coulomb(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    q1, q2 = need(p, "q1", "charge", positive=False), need(p, "q2", "charge", positive=False)
    r = need(p, "r", "length")
    k = K_E / _eps_r(p)
    f = k * q1 * q2 / r**2
    ctx.formula = (
        "F = k_e q1 q2 / (eps_r r^2), k_e = 1 / (4 pi eps0); E1 = k_e |q1| / (eps_r r^2); "
        "V1 = k_e q1 / (eps_r r); U = q2 V1"
    )
    ctx.formatted = (
        f"F = {fmt_sig(f, 6)} N ({'repulsion' if f > 0 else 'attraction' if f < 0 else 'zero'})."
    )
    return {"k_e": K_E, "F": f, "E1": k * abs(q1) / r**2, "V1": k * q1 / r, "U": k * q1 * q2 / r}


def _vec(raw: Any, key: str, sizes: tuple[int, ...]) -> list[float]:
    if (
        not isinstance(raw, list)
        or len(raw) not in sizes
        or any(
            isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(float(x))
            for x in raw
        )
    ):
        raise ToolError(
            "INVALID_ARGUMENT", f"params.{key}: list of {' or '.join(map(str, sizes))} numbers."
        )
    return [float(x) for x in raw]


def _point_charges(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    raw = p.get("charges")
    if not isinstance(raw, list) or not raw:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.charges: list of charges [q in C, x, y] or [q, x, y, z] (positions in m).",
            '{"charges": [[1e-9, 0, 0], [-2e-9, 0.1, 0]], "point": [0.05, 0.05]}',
        )
    rows = [_vec(c, "charges", (3, 4)) for c in raw]
    dim = len(rows[0]) - 1
    if any(len(c) - 1 != dim for c in rows):
        raise ToolError("INVALID_ARGUMENT", "params.charges: all the charges in 2D or all in 3D.")
    pt = _vec(p.get("point"), "point", (dim,))
    k = K_E / _eps_r(p)
    e = [0.0] * dim
    v = 0.0
    for q, *pos in rows:
        d = [pt[i] - pos[i] for i in range(dim)]
        r = math.sqrt(sum(x * x for x in d))
        if r == 0:
            raise ToolError(
                "INVALID_ARGUMENT",
                "The point coincides with a charge: infinite field and potential.",
            )
        for i in range(dim):
            e[i] += k * q * d[i] / r**3
        v += k * q / r
    out: dict[str, Any] = {"E": e, "E_mag": math.sqrt(sum(x * x for x in e)), "V": v}
    if len(rows) > 1:
        u = 0.0
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                rij = math.dist(rows[i][1:], rows[j][1:])
                if rij == 0:
                    raise ToolError("INVALID_ARGUMENT", f"Charges {i + 1} and {j + 1} coincide.")
                u += k * rows[i][0] * rows[j][0] / rij
        out["U_system"] = u
    if p.get("q") is not None:
        q = need(p, "q", "charge", positive=False)
        out["F"] = [q * x for x in e]
        out["F_mag"] = abs(q) * out["E_mag"]
    ctx.formula = (
        "E = sum k_e q_i (r - r_i) / |r - r_i|^3; V = sum k_e q_i / |r - r_i|; "
        "U = sum_(i<j) k_e q_i q_j / r_ij; F = q E"
    )
    ctx.formatted = f"|E| = {fmt_sig(out['E_mag'], 6)} V/m, V = {fmt_sig(v, 6)} V."
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@range_guard
def _dipole(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    sep: list[float] | None = None
    if one_of(p, ("p", "q"), "the dipole moment (p, or q with d)") == "p":
        if p.get("d") is not None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.d goes with params.q (p = q d): with params.p leave it out.",
            )
        pv = _vec(p.get("p"), "p", (3,))
    else:
        q = need(p, "q", "charge", positive=False)
        sep = _vec(p.get("d"), "d", (3,))
        pv = [q * x for x in sep]
    if p.get("point") is None:
        for key in ("center", "eps_r"):
            if p.get(key) is not None:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    f"params.{key} goes with params.point (the field of the dipole at a point).",
                )
    size_p = math.sqrt(_dot(pv, pv))
    out: dict[str, Any] = {"p": pv, "p_magnitude": size_p}
    parts = ["p = q d"]
    if p.get("E") is not None:
        ev = _vec(p.get("E"), "E", (3,))
        tau = [
            pv[1] * ev[2] - pv[2] * ev[1],
            pv[2] * ev[0] - pv[0] * ev[2],
            pv[0] * ev[1] - pv[1] * ev[0],
        ]
        size_tau = math.sqrt(_dot(tau, tau))
        u = -_dot(pv, ev)
        out |= {"tau": tau, "tau_magnitude": size_tau, "U": u}
        if size_p > 0 and _dot(ev, ev) > 0:
            out["angle_deg"] = math.degrees(math.atan2(size_tau, -u))
        parts.append("uniform field: U = -p . E, tau = p x E")
    if p.get("point") is not None:
        x = _vec(p.get("point"), "point", (3,))
        c = _vec(p.get("center"), "center", (3,)) if p.get("center") is not None else [0.0] * 3
        r = [a - b for a, b in zip(x, c, strict=True)]
        size = math.sqrt(_dot(r, r))
        if size == 0:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.point is the position of the dipole (params.center): the field of a point dipole "
                "is infinite there.",
            )
        k = K_E / _eps_r(p)
        pr = _dot(pv, r)
        e = [k * (3 * pr * ri / size**5 - pc / size**3) for ri, pc in zip(r, pv, strict=True)]
        out |= {
            "R_vector": r,
            "R_magnitude": size,
            "E_dipole": e,
            "E_dipole_mag": math.sqrt(_dot(e, e)),
            "V_dipole": k * pr / size**3,
        }
        parts.append(
            "ideal dipole, R = point - center: V = k_e p . R / R^3, E = k_e (3 (p . R^) R^ - p) / R^3"
        )
        if sep is not None and math.sqrt(_dot(sep, sep)) > _DIPOLE_NEAR * size:
            ctx.warn(
                f"|d| = {fmt_sig(math.sqrt(_dot(sep, sep)))} m is more than {_DIPOLE_NEAR:g} |R|: the field of "
                "the two charges differs from the ideal dipole by terms of order (d / R)^2; mode "
                "point_charges gives it exactly."
            )
    ctx.formula = "; ".join(parts) + "; k_e = 1 / (4 pi eps0 eps_r)"
    ctx.formatted = f"|p| = {fmt_sig(size_p, 6)} C m" + (
        f", |E| = {fmt_sig(out['E_dipole_mag'], 6)} V/m" if "E_dipole" in out else ""
    )
    return out


def _capacitor(geo: str, p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    eps = _eps_r(p) * EPS0
    out: dict[str, Any] = {}
    if geo == "parallel_plate":
        area, d = need(p, "A", "area"), need(p, "d", "length")
        c = eps * area / d
        ctx.formula = "C = eps A / d (edge effects neglected)"
    elif geo in ("coaxial", "eccentric_cylinders", "concentric_spheres"):
        ri, ro = need(p, "R_inner", "length"), need(p, "R_outer", "length")
        if ri >= ro:
            raise ToolError("INVALID_ARGUMENT", "R_outer must be greater than R_inner.")
        if geo == "concentric_spheres":
            # 1/ri - 1/ro = (ro - ri) / (ri ro), without cancellation (D79)
            c = 4 * math.pi * eps * ri * ro / (ro - ri)
            ctx.formula = "C = 4 pi eps / (1/R_inner - 1/R_outer)"
        else:
            length = need(p, "L", "length")
            if geo == "coaxial":
                c = 2 * math.pi * eps * length / log_ratio(ro, ri)  # ln(ro / ri), thin gap (D79)
                ctx.formula = "C = 2 pi eps L / ln(R_outer / R_inner)"
            else:
                d = need(p, "d", "length", positive=False)
                if d < 0 or d + ri >= ro:
                    raise ToolError(
                        "INVALID_ARGUMENT",
                        "With d >= R_outer - R_inner the inner conductor touches the outer one.",
                    )
                # acosh(1 + y) = log1p(y + sqrt(y (y + 2))) with y = ((ro - ri)^2 - d^2) / (2 ro ri): no
                # cancellation when the conductors almost touch (D80)
                g = ro - ri
                y = (g - d) * (g + d) / (2 * ro * ri)
                c = 2 * math.pi * eps * length / acosh1p(y)
                ctx.formula = (
                    "C = 2 pi eps L / arcosh[(R_outer^2 + R_inner^2 - d^2) / (2 R_outer R_inner)]"
                )
    elif geo in ("parallel_wires", "wire_wall"):
        a, d, length = (
            need(p, "a", "length"),
            need(p, "d", "length"),
            need(p, "L", "length"),
        )
        if geo == "parallel_wires":
            if d <= 2 * a:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    "Parallel wires: the distance d between the axes must exceed 2a.",
                )
            # acosh(d / (2a)) = acosh(1 + (d - 2a) / (2a)), exact difference for close wires (D82)
            c = math.pi * eps * length / acosh1p((d - 2 * a) / (2 * a))
            ctx.formula = "C = pi eps L / arcosh(d / (2a))"
        else:
            if d <= a:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    "Wire and wall: the distance d of the axis must exceed the radius a.",
                )
            # acosh(d / a) = acosh(1 + (d - a) / a), exact difference close to the wall (D82)
            c = 2 * math.pi * eps * length / acosh1p((d - a) / a)
            ctx.formula = "C = 2 pi eps L / arcosh(d / a)"
    else:
        radius = need(p, "R", "length")
        c = 4 * math.pi * eps * radius if geo == "sphere" else 8 * eps * radius
        ctx.formula = (
            "C = 4 pi eps R (isolated sphere)" if geo == "sphere" else "C = 8 eps R (isolated disc)"
        )
    out["C"] = c
    if p.get("V") is not None and p.get("Q") is not None:
        raise ToolError("INVALID_ARGUMENT", "Give only one of V and Q.")
    if p.get("V") is not None or p.get("Q") is not None:
        if p.get("V") is not None:
            v = need(p, "V", "voltage", positive=False)
            q = c * v
        else:
            q = need(p, "Q", "charge", positive=False)
            v = q / c
        out |= {"Q": q, "V": v, "U": 0.5 * c * v * v}
        if geo == "parallel_plate":
            out["E"] = v / need(p, "d", "length")
    ctx.formatted = (
        f"C = {fmt_sig(c, 6)} F" + (f", U = {fmt_sig(out['U'], 6)} J" if "U" in out else "") + "."
    )
    return out


def _combination(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    raw = p.get("values")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ToolError(
            "INVALID_ARGUMENT", 'params.values: at least 2 capacitances (F or [value, "uF"]).'
        )
    vals = []
    for i, x in enumerate(raw):
        key = f"values[{i}]"
        v = qty({key: x}, key, "capacitance")
        assert v is not None
        vals.append(v)
    conn = choice(p, "connection", ("series", "parallel"))
    c = sum(vals) if conn == "parallel" else 1 / sum(1 / x for x in vals)
    ctx.formula = "parallel: C = sum C_i; series: 1/C = sum 1/C_i"
    ctx.formatted = f"Equivalent C = {fmt_sig(c, 6)} F."
    return {"C_eq": c}
