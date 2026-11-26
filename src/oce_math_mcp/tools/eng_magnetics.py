"""eng_calc: magnetic dipoles and magnetic circuits (D107), Larmor precession (D114).

Source: Wikipedia "List of electromagnetism equations", sections "Magnetic circuits" and "Magnetic fields and
moments", in the local offline library. SI units, vacuum (mu0) around the dipole.
- Magnetomotive force F = N I; Hopkinson's law F = Phi R (the law of a magnetic circuit that goes with the
  page's F), reluctance of a uniform segment R = l / (mu0 mu_r A), segments in series add up.
- Point dipole m at x': r = x - x', A = mu0 / (4 pi) m x r / |r|^3, B = curl A = mu0 / (4 pi)
  (3 r (m . r) / |r|^5 - m / |r|^3).
- Moment m = (1/2) int r x J dV: for a closed loop of current I (J dV = I dl) m = (N I / 2) sum r_k x r_(k+1)
  over the vertices of a polygon, exact; for a flat loop m = N I A n. A volume distribution J(x, y, z) is
  math_solve.multi_integrate of the components of (1/2) r x J.
- Torque tau = int dm x B and energy U = - int dm . B: m x B and - m . B for a dipole in a uniform field (or a
  point dipole); the sum over pieces dm_i in their fields B_i for a distribution.
- Larmor precession (D114; page "Larmor precession"): the moment precesses about B at omega = |gamma B|, whatever the
  angle between them; f = omega / (2 pi). gamma is given, or taken from CODATA 2022 for a particle (magnitudes;
  the electron, neutron and helion moments point against their spin, so they precess in the other sense).
"""

from __future__ import annotations

import math
from typing import Any

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, one_of, only, op, range_guard, units_for

_DIP_MODES: dict[str, set[str]] = {
    "field": {"m", "x", "x0"},
    "moment": {"I", "N", "A", "normal", "vertices"},
    "torque": {"m", "B"},
    "larmor": {"B", "gamma", "particle"},
}
_GAMMA = {  # CODATA 2022 names of the gyromagnetic ratios (magnitudes, rad/(s T))
    "electron": "electron gyromag. ratio",
    "proton": "proton gyromag. ratio",
    "neutron": "neutron gyromag. ratio",
    "shielded_proton": "shielded proton gyromag. ratio",
    "shielded_helion": "shielded helion gyromag. ratio",
}
_CIRCUIT_KEYS = ("N", "I", "F", "Phi", "R", "segments")
_MAX_POINTS = 10000
_MAX_SEGMENTS = 1000
_UNITS = {
    "r": "m",
    "r_magnitude": "m",
    "A": "T*m",
    "B": "T",
    "B_magnitude": "T",
    "m": "A*m2",
    "m_magnitude": "A*m2",
    "vector_area": "m2",
    "tau": "N*m",
    "tau_magnitude": "N*m",
    "U": "J",
    "angle_deg": "deg",
    "F": "A",
    "Phi": "Wb",
    "R": "1/H",
    "I": "A",
    "R_segments": "1/H",
    "B_segments": "T",
    "H_segments": "A/m",
    "drop_segments": "A",
    "gamma": "rad/(s*T)",
    "gamma_over_2pi": "Hz/T",
    "omega": "rad/s",
    "f": "Hz",
}


def _mu0() -> float:
    return lookup("mu_0")[1]


def _vector(raw: Any, where: str) -> list[float]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ToolError("INVALID_ARGUMENT", f"{where}: 3 numbers (x, y, z components in SI units).")
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ToolError("INVALID_ARGUMENT", f"{where}: finite numbers only.")
    return [float(v) for v in raw]


def _vec3(p: dict[str, Any], key: str) -> list[float]:
    return _vector(p.get(key), f"params.{key}")


def _vectors(p: dict[str, Any], key: str, minimum: int) -> list[list[float]]:
    raw = p.get(key)
    if not isinstance(raw, list) or not minimum <= len(raw) <= _MAX_POINTS:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: a list of {minimum} to {_MAX_POINTS} vectors [x, y, z] in SI units.",
        )
    return [_vector(v, f"params.{key}[{i}]") for i, v in enumerate(raw)]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


@op("magnetic_dipole", ("mode", *sorted(set().union(*_DIP_MODES.values()))))
@range_guard
def _magnetic_dipole(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_DIP_MODES))
    only(p, mode, _DIP_MODES[mode])
    out = {"field": _field, "moment": _moment, "torque": _torque, "larmor": _larmor}[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _field(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m, x = _vec3(p, "m"), _vec3(p, "x")
    x0 = _vec3(p, "x0") if p.get("x0") is not None else [0.0, 0.0, 0.0]
    r = [a - b for a, b in zip(x, x0, strict=True)]
    size = _norm(r)
    if size == 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.x is the position of the dipole (params.x0): the field of a point dipole is infinite there.",
        )
    k = _mu0() / (4 * math.pi)
    a = [k * v / size**3 for v in _cross(m, r)]
    mr = _dot(m, r)
    b = [k * (3 * ri * mr / size**5 - mi / size**3) for ri, mi in zip(r, m, strict=True)]
    b_size = _norm(b)
    ctx.formula = "r = x - x', A = mu0 / (4 pi) m x r / |r|^3, B = curl A = mu0 / (4 pi) (3 r (m . r) / |r|^5 - m / |r|^3)"
    ctx.formatted = f"|B| = {fmt_sig(b_size)} T at |r| = {fmt_sig(size)} m"
    return {"r": r, "r_magnitude": size, "A": a, "B": b, "B_magnitude": b_size}


def _moment(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    cur = need(p, "I", "current", positive=False)
    turns = integer(p, "N") if p.get("N") is not None else 1
    out: dict[str, Any]
    if one_of(p, ("A", "vertices"), "the shape of the loop") == "A":
        size = turns * cur * need(p, "A", "area")
        out = {"m_magnitude": abs(size)}
        if p.get("normal") is not None:
            n = _vec3(p, "normal")
            length = _norm(n)
            if length == 0:
                raise ToolError("INVALID_ARGUMENT", "params.normal must not be the zero vector.")
            out["m"] = [size * v / length for v in n]
        ctx.formula = "flat loop: m = N I A n (n the unit normal, right-hand rule with the current)"
    else:
        if p.get("normal") is not None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.normal goes with params.A: the vertices give the direction themselves.",
            )
        pts = _vectors(p, "vertices", 3)
        rel = [[a - b for a, b in zip(v, pts[0], strict=True)] for v in pts]  # less rounding
        area = [0.0, 0.0, 0.0]
        for a, b in zip(rel, rel[1:] + rel[:1], strict=True):
            area = [s + c for s, c in zip(area, _cross(a, b), strict=True)]
        area = [v / 2 for v in area]
        mv = [turns * cur * v for v in area]
        out = {"m": mv, "m_magnitude": _norm(mv), "vector_area": area}
        ctx.formula = "closed polygon (J dV = I dl): m = (1/2) int r x J dV = (N I / 2) sum r_k x r_(k+1), exact"
    ctx.formatted = f"|m| = {fmt_sig(out['m_magnitude'])} A m2"
    return out


def _torque(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    raw = p.get("m")
    single = not (isinstance(raw, list) and raw and all(isinstance(v, list) for v in raw))
    if single:
        ms, bs = [_vec3(p, "m")], [_vec3(p, "B")]
    else:
        ms, bs = _vectors(p, "m", 1), _vectors(p, "B", 1)
        if len(ms) != len(bs):
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.m and params.B: the same number of pieces (each dm_i in its field B_i).",
            )
    tau = [0.0, 0.0, 0.0]
    u = 0.0
    for m, b in zip(ms, bs, strict=True):
        tau = [s + c for s, c in zip(tau, _cross(m, b), strict=True)]
        u -= _dot(m, b)
    size = _norm(tau)
    out: dict[str, Any] = {"tau": tau, "tau_magnitude": size, "U": u}
    if single:
        if _norm(ms[0]) > 0 and _norm(bs[0]) > 0:
            out["angle_deg"] = math.degrees(math.atan2(size, -u))
        ctx.formula = "tau = m x B, U = -m . B (dipole in a uniform field, or a point dipole)"
    else:
        ctx.formula = "tau = int dm x B = sum dm_i x B_i, U = -int dm . B = -sum dm_i . B_i"
    ctx.formatted = f"|tau| = {fmt_sig(size)} N m, U = {fmt_sig(u)} J"
    return out


def _larmor(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    raw = p.get("B")
    if (
        isinstance(raw, list)
        and len(raw) == 3
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in raw)
    ):
        b = _norm(_vec3(p, "B"))  # a field vector: its magnitude
    else:
        b = abs(need(p, "B", "magnetic_flux_density", positive=False))
    if one_of(p, ("gamma", "particle"), "the gyromagnetic ratio") == "gamma":
        gamma = need(p, "gamma", None, positive=False)
        source = "gamma given"
    else:
        name = _GAMMA[choice(p, "particle", tuple(_GAMMA))]
        gamma = lookup(name)[1]
        source = f"gamma = CODATA 2022 {name}"
    omega = abs(gamma * b)
    ctx.formula = (
        f"omega = |gamma B| (Larmor frequency, the same at any angle between the moment and B), "
        f"f = omega / (2 pi); {source}"
    )
    ctx.formatted = (
        f"Larmor frequency f = {fmt_sig(omega / (2 * math.pi), 6)} Hz at |B| = {fmt_sig(b)} T"
    )
    return {
        "gamma": gamma,
        "gamma_over_2pi": gamma / (2 * math.pi),
        "omega": omega,
        "f": omega / (2 * math.pi),
    }


def _segments(p: dict[str, Any]) -> list[tuple[float, float, float]]:
    raw = p.get("segments")
    if not isinstance(raw, list) or not 1 <= len(raw) <= _MAX_SEGMENTS:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.segments: a list of 1 to {_MAX_SEGMENTS} segments [l, A, mu_r] in series "
            "(m, m2, relative permeability).",
        )
    out = []
    for i, s in enumerate(raw):
        if (
            not isinstance(s, list)
            or len(s) != 3
            or any(
                isinstance(v, bool)
                or not isinstance(v, int | float)
                or not math.isfinite(v)
                or v <= 0
                for v in s
            )
        ):
            raise ToolError(
                "INVALID_ARGUMENT",
                f"params.segments[{i}]: [l, A, mu_r], three positive numbers (m, m2, relative permeability).",
            )
        out.append((float(s[0]), float(s[1]), float(s[2])))
    return out


@op("magnetic_circuit", _CIRCUIT_KEYS)
@range_guard
def _magnetic_circuit(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mu0 = _mu0()
    seg_out: dict[str, Any] = {}
    segs = None
    if one_of(p, ("R", "segments"), "the reluctance") == "R":
        rel = need(p, "R", "reluctance")
    else:
        segs = _segments(p)
        r_seg = [length / (mu0 * mu_r * area) for length, area, mu_r in segs]
        rel = sum(r_seg)
        seg_out["R_segments"] = r_seg
    turns = integer(p, "N") if p.get("N") is not None else None
    src = one_of(p, ("I", "F", "Phi"), "the drive of the circuit")
    if src == "I":
        if turns is None:
            raise ToolError(
                "INVALID_ARGUMENT", "params.I needs params.N (the turns of the coil): F = N I."
            )
        mmf = turns * need(p, "I", "current", positive=False)
        phi = mmf / rel
    elif src == "F":
        mmf = need(p, "F", "current", positive=False)
        phi = mmf / rel
    else:
        phi = need(p, "Phi", "magnetic_flux", positive=False)
        mmf = phi * rel
    out: dict[str, Any] = {"F": mmf, "Phi": phi, "R": rel, **seg_out}
    if turns is not None:
        out["I"] = mmf / turns
    if segs is not None:
        b = [phi / area for _, area, _ in segs]
        h = [bi / (mu0 * mu_r) for bi, (_, _, mu_r) in zip(b, segs, strict=True)]
        out["B_segments"] = b
        out["H_segments"] = h
        out["drop_segments"] = [hi * length for hi, (length, _, _) in zip(h, segs, strict=True)]
    ctx.formula = "F = N I = Phi R (Hopkinson), R = sum l / (mu0 mu_r A) in series, B = Phi / A, H = B / (mu0 mu_r)"
    ctx.formatted = f"F = {fmt_sig(mmf)} A, Phi = {fmt_sig(phi)} Wb, R = {fmt_sig(rel)} 1/H"
    ctx.units = units_for(out, _UNITS)
    return out
