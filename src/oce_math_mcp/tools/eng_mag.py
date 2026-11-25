"""eng_calc: magnetic fields, forces, inductance and induction (D56).

Sources (Wikipedia pages):
- «Magnetic field», table of the fields: infinite wire mu0 I / (2 pi x) outside and mu0 I x / (2 pi R^2)
  inside; loop on the axis mu0 I R^2 / (2 (x^2 + R^2)^(3/2)) (also «Biot–Savart law»); infinite
  solenoid mu0 n I; toroid mu0 N I / (2 pi R);
- «Solenoid»: finite solenoid on the axis, B_z = mu0 N I / 2 [(z + l/2) / (l sqrt(R^2 + (z + l/2)^2))
  - (z - l/2) / (l sqrt(R^2 + (z - l/2)^2))];
- «Ampère's force law»: F / L = 2 k_A I1 I2 / r, k_A = mu0 / (4 pi);
- «Lorentz force»: F = q (E + v x B), F = B I l sin(theta) on a wire;
- «Inductance»: solenoid mu0 N^2 A / l, L = N Phi / i, coaxial cable mu / (2 pi) l ln(b / a), Rosa
  straight wire 200 nH/m l [ln(2 l / r) - 0.75] (at high frequency - 1), table of the thin-wire
  shapes (Wheeler, loop, rectangle, two wires with the parameter Y);
- «Faraday's law of induction»: emf = -d Phi / dt.
Derived here (stated in the `formula` field): cyclotron radius and period from the Lorentz force and
the circular motion; inductance of the toroid with rectangular cross-section from the integrated flux of the
toroid field; rotating coil and moving bar from Faraday's law.
"""

from __future__ import annotations

import math
from typing import Any

import scipy.constants as _sc

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, only, op, qty, units_for

MU0 = float(_sc.mu_0)

_MAG_MODES: dict[str, set[str]] = {
    "wire": {"I", "r", "R_wire"},
    "loop": {"I", "R", "x", "N"},
    "solenoid": {"N", "I", "L", "R", "z", "mu_r"},
    "toroid": {"N", "I", "r", "mu_r"},
    "force_wires": {"I1", "I2", "d", "L"},
    "force_on_wire": {"I", "L", "B", "angle_deg"},
    "lorentz": {"q", "v", "E", "B", "m"},
}
_MAG_UNITS = {
    "B": "T",
    "H": "A/m",
    "n": "1/m",
    "F": "N",
    "F_per_L": "N/m",
    "F_mag": "N",
    "r_c": "m",
    "T_c": "s",
    "f_c": "Hz",
    "v_perp": "m/s",
}


def _b(p: dict[str, Any], key: str) -> float:
    return need(p, key, "magnetic_flux_density", positive=False)


def _vec3(p: dict[str, Any], key: str, *, required: bool) -> list[float] | None:
    raw = p.get(key)
    if raw is None:
        if required:
            raise ToolError(
                "INVALID_ARGUMENT", f"Missing params.{key} (vector [x, y, z] in SI units)."
            )
        return None
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or any(
            isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(float(x))
            for x in raw
        )
    ):
        raise ToolError(
            "INVALID_ARGUMENT", f"params.{key}: vector of 3 numbers [x, y, z] in SI units."
        )
    return [float(x) for x in raw]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


@op("magnetism", ("mode", *sorted(set().union(*_MAG_MODES.values()))))
def _magnetism(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MAG_MODES))
    only(p, mode, _MAG_MODES[mode])
    out: dict[str, Any]
    if mode == "wire":
        cur, r = need(p, "I", "current", positive=False), need(p, "r", "length")
        rw = qty(p, "R_wire", "length", need=False)
        if rw is not None and r < rw:
            b = MU0 * cur * r / (2 * math.pi * rw * rw)
            ctx.formula = "inside the wire: B = mu0 I r / (2 pi R_wire^2)"
        else:
            b = MU0 * cur / (2 * math.pi * r)
            ctx.formula = "infinite straight wire: B = mu0 I / (2 pi r)"
        out = {"B": b, "H": b / MU0}
    elif mode == "loop":
        cur, radius = need(p, "I", "current", positive=False), need(p, "R", "length")
        x = qty(p, "x", "length", need=False, positive=False) or 0.0
        turns = integer(p, "N") if p.get("N") is not None else 1
        b = turns * MU0 * cur * radius**2 / (2 * (x * x + radius**2) ** 1.5)
        out = {"B": b, "H": b / MU0}
        ctx.formula = "loop on the axis: B = N mu0 I R^2 / [2 (x^2 + R^2)^(3/2)]"
    elif mode in ("solenoid", "toroid"):
        turns = integer(p, "N")
        cur = need(p, "I", "current", positive=False)
        mu = MU0 * (need(p, "mu_r", None) if p.get("mu_r") is not None else 1.0)
        if mode == "toroid":
            r = need(p, "r", "length")
            b = mu * turns * cur / (2 * math.pi * r)
            ctx.formula = "toroid: B = mu N I / (2 pi r)"
            out = {"B": b}
        else:
            length = need(p, "L", "length")
            out = {"n": turns / length}
            if p.get("R") is None:
                if p.get("z") is not None:
                    raise ToolError(
                        "INVALID_ARGUMENT", "params.z needs the radius R (finite solenoid)."
                    )
                b = mu * turns * cur / length
                ctx.formula = "long solenoid: B = mu N I / L"
            else:
                radius = need(p, "R", "length")
                z = qty(p, "z", "length", need=False, positive=False) or 0.0
                hp, hm = z + length / 2, z - length / 2
                b = (
                    mu
                    * turns
                    * cur
                    / 2
                    * (
                        hp / (length * math.hypot(radius, hp))
                        - hm / (length * math.hypot(radius, hm))
                    )
                )
                ctx.formula = (
                    "finite solenoid on the axis: B = mu N I / 2 [(z + L/2) / (L sqrt(R^2 + (z + L/2)^2)) "
                    "- (z - L/2) / (L sqrt(R^2 + (z - L/2)^2))]"
                )
            out["B"] = b
        if p.get("mu_r") is not None:
            ctx.warn("With mu_r the material fills the whole field space; core not saturated.")
        out["H"] = out["B"] / mu
    elif mode == "force_wires":
        i1, i2 = (
            need(p, "I1", "current", positive=False),
            need(p, "I2", "current", positive=False),
        )
        d = need(p, "d", "length")
        fl = MU0 * i1 * i2 / (2 * math.pi * d)
        out = {"F_per_L": fl}
        if p.get("L") is not None:
            out["F"] = fl * need(p, "L", "length")
        ctx.formula = (
            "F / L = mu0 I1 I2 / (2 pi d) (positive = attraction, currents in the same direction)"
        )
    elif mode == "force_on_wire":
        cur, length = need(p, "I", "current", positive=False), need(p, "L", "length")
        b = _b(p, "B")
        ang = need(p, "angle_deg", None, positive=False) if p.get("angle_deg") is not None else 90.0
        out = {"F": abs(b * cur * length * math.sin(math.radians(ang)))}
        ctx.formula = "F = B I L sin(theta)"
    else:
        out = _lorentz(p, ctx)
    ctx.units = units_for(out, _MAG_UNITS)
    first = next(iter(out))
    val = out[first]
    ctx.formatted = (
        f"{first} = {fmt_sig(val, 6)} {_MAG_UNITS.get(first, '')}."
        if not isinstance(val, list)
        else f"{first} = [{', '.join(fmt_sig(x, 6) for x in val)}] {_MAG_UNITS.get(first, '')}."
    )
    return out


def _lorentz(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    q = need(p, "q", "charge", positive=False)
    v = _vec3(p, "v", required=True)
    assert v is not None
    e = _vec3(p, "E", required=False) or [0.0, 0.0, 0.0]
    b = _vec3(p, "B", required=False) or [0.0, 0.0, 0.0]
    vxb = _cross(v, b)
    f = [q * (e[i] + vxb[i]) for i in range(3)]
    out: dict[str, Any] = {"F": f, "F_mag": math.sqrt(sum(x * x for x in f))}
    ctx.formula = "F = q (E + v x B)"
    bm = math.sqrt(sum(x * x for x in b))
    if p.get("m") is not None and bm > 0 and q != 0:
        m = need(p, "m", "mass")
        v_par = sum(v[i] * b[i] for i in range(3)) / bm
        v_perp = math.sqrt(max(sum(x * x for x in v) - v_par * v_par, 0.0))
        out |= {
            "v_perp": v_perp,
            "r_c": m * v_perp / (abs(q) * bm),
            "T_c": 2 * math.pi * m / (abs(q) * bm),
            "f_c": abs(q) * bm / (2 * math.pi * m),
        }
        ctx.formula += (
            "; cyclotron (derived, uniform B field): r = m v_perp / (|q| B), T = 2 pi m / (|q| B)"
        )
        if any(x != 0 for x in e):
            ctx.warn(
                "With E other than zero the motion is not a circle: r_c and T_c hold only for the B field."
            )
        ctx.warn("Non-relativistic cyclotron: holds for v much less than c.")
    return out


# ------------------------------------------------------------------ inductance

_IND_GEOMETRIES: dict[str, set[str]] = {
    "solenoid": {"N", "A", "D", "L", "mu_r"},
    "toroid": {"N", "h", "r_inner", "r_outer", "mu_r"},
    "coaxial": {"a", "b", "L", "mu_r"},
    "straight_wire": {"L", "r", "hf"},
    "circular_loop": {"r", "a", "Y"},
    "rectangle": {"l1", "l2", "a", "Y"},
    "parallel_wires": {"s", "a", "L", "Y", "hf"},
    "wheeler": {"N", "D", "L"},
}
_IND_UNITS = {"L": "H", "W": "J", "flux_linkage": "Wb"}


def _y(p: dict[str, Any]) -> float:
    y = need(p, "Y", None, positive=False) if p.get("Y") is not None else 1.0
    if not 0 <= y <= 1:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.Y goes from 0 (surface only, high frequency) to 1 (uniform current).",
        )
    return y


def _flag(p: dict[str, Any], key: str) -> bool:
    v = p.get(key, False)
    if not isinstance(v, bool):
        raise ToolError("INVALID_ARGUMENT", f"params.{key}: true or false.")
    return v


@op("inductance", ("geometry", "I", *sorted(set().union(*_IND_GEOMETRIES.values()))))
def _inductance(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    geo = choice(p, "geometry", tuple(_IND_GEOMETRIES))
    only(p, geo, _IND_GEOMETRIES[geo] | {"I"}, field="geometry")
    mu = MU0 * (need(p, "mu_r", None) if p.get("mu_r") is not None else 1.0)
    if geo == "solenoid":
        turns = integer(p, "N")
        if (p.get("A") is None) == (p.get("D") is None):
            raise ToolError("INVALID_ARGUMENT", "Solenoid: give the area A or the diameter D.")
        area = (
            need(p, "A", "area")
            if p.get("A") is not None
            else math.pi * need(p, "D", "length") ** 2 / 4
        )
        ind = mu * turns**2 * area / need(p, "L", "length")
        ctx.formula = "L = mu N^2 A / l (long solenoid)"
    elif geo == "toroid":
        turns = integer(p, "N")
        h, a, b = (
            need(p, "h", "length"),
            need(p, "r_inner", "length"),
            need(p, "r_outer", "length"),
        )
        if b <= a:
            raise ToolError("INVALID_ARGUMENT", "r_outer must be greater than r_inner.")
        ind = mu * turns**2 * h * math.log(b / a) / (2 * math.pi)
        ctx.formula = (
            "L = N Phi / I with B = mu N I / (2 pi r) integrated over the rectangular cross-section: "
            "L = mu N^2 h ln(r_outer / r_inner) / (2 pi)"
        )
    elif geo == "coaxial":
        a, b, length = (
            need(p, "a", "length"),
            need(p, "b", "length"),
            need(p, "L", "length"),
        )
        if b <= a:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Coaxial cable: the inner radius b of the outer conductor must exceed a.",
            )
        ind = mu / (2 * math.pi) * length * math.log(b / a)
        ctx.formula = "L = mu / (2 pi) l ln(b / a) (high frequency: no flux inside the conductors)"
    elif geo == "straight_wire":
        length, r = need(p, "L", "length"), need(p, "r", "length")
        const = 1.0 if _flag(p, "hf") else 0.75
        ind = 200e-9 * length * (math.log(2 * length / r) - const)
        ctx.formula = f"Rosa: L = 200 nH/m l [ln(2 l / r) - {const}]"
        if length < 10 * r:
            ctx.warn("Rosa formula for a wire that is long compared with its radius.")
    elif geo == "circular_loop":
        r, a = need(p, "r", "length"), need(p, "a", "length")
        ind = MU0 * r * (math.log(8 * r / a) - 2 + _y(p) / 4)
        ctx.formula = "L = mu0 r [ln(8 r / a) - 2 + Y / 4] (thin wire, a << r)"
    elif geo == "rectangle":
        l1, l2, a = (
            need(p, "l1", "length"),
            need(p, "l2", "length"),
            need(p, "a", "length"),
        )
        ind = (
            MU0
            / math.pi
            * (
                l1 * math.log(2 * l1 / a)
                + l2 * math.log(2 * l2 / a)
                + 2 * math.hypot(l1, l2)
                - l1 * math.asinh(l1 / l2)
                - l2 * math.asinh(l2 / l1)
                - (2 - _y(p) / 4) * (l1 + l2)
            )
        )
        ctx.formula = (
            "L = mu0 / pi [l1 ln(2 l1 / a) + l2 ln(2 l2 / a) + 2 sqrt(l1^2 + l2^2) - l1 asinh(l1 / l2) "
            "- l2 asinh(l2 / l1) - (2 - Y / 4)(l1 + l2)]"
        )
    elif geo == "parallel_wires":
        s, a, length = (
            need(p, "s", "length"),
            need(p, "a", "length"),
            need(p, "L", "length"),
        )
        if s < 2 * a:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Two wires: the distance s between the axes must be at least 2a.",
            )
        if _flag(p, "hf"):
            if p.get("Y") is not None:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    "With hf = true Y is not needed (current only at the surface).",
                )
            ind = MU0 / math.pi * length * math.acosh(s / (2 * a))
            ctx.formula = "high frequency: L = mu0 / pi l arcosh(s / (2a))"
        else:
            ind = MU0 / math.pi * length * (math.log(s / a) + _y(p) / 4)
            ctx.formula = "L = mu0 / pi l [ln(s / a) + Y / 4]"
    else:
        turns = integer(p, "N")
        d_cm, l_cm = need(p, "D", "length") * 100, need(p, "L", "length") * 100
        ind = turns**2 * d_cm**2 / (45.72 * d_cm + 101.6 * l_cm) * 1e-6
        ctx.formula = "Wheeler: L [uH] = N^2 D^2 / (45.72 D + 101.6 l), D and l in cm"
        if l_cm <= 0.4 * d_cm:
            ctx.warn("Wheeler: error within 1% only for l > 0.4 D.")
    out: dict[str, Any] = {"L": ind}
    if p.get("I") is not None:
        cur = need(p, "I", "current", positive=False)
        out |= {"flux_linkage": ind * cur, "W": 0.5 * ind * cur * cur}
    if (
        geo in ("circular_loop", "rectangle", "parallel_wires")
        and p.get("Y") is None
        and not p.get("hf")
    ):
        ctx.warn("Y = 1 (uniform current, low frequency) if not given.")
    ctx.units = units_for(out, _IND_UNITS)
    ctx.formatted = (
        f"L = {fmt_sig(ind, 6)} H"
        + (f", energy {fmt_sig(out['W'], 6)} J" if "W" in out else "")
        + "."
    )
    return out


# ------------------------------------------------------------------ electromagnetic_induction

_EMF_MODES: dict[str, set[str]] = {
    "flux_change": {"N", "dPhi", "A", "B1", "B2", "dt"},
    "rotating_coil": {"N", "A", "B", "omega", "rpm"},
    "motional": {"B", "L", "v"},
}
_EMF_UNITS = {
    "emf": "V",
    "emf_peak": "V",
    "emf_rms": "V",
    "dPhi": "Wb",
    "f": "Hz",
    "omega": "rad/s",
}


@op("electromagnetic_induction", ("mode", *sorted(set().union(*_EMF_MODES.values()))))
def _induction(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_EMF_MODES))
    only(p, mode, _EMF_MODES[mode])
    if mode == "flux_change":
        turns = integer(p, "N") if p.get("N") is not None else 1
        has_phi = p.get("dPhi") is not None
        has_b = any(p.get(k) is not None for k in ("A", "B1", "B2"))
        if has_phi == has_b:
            raise ToolError(
                "INVALID_ARGUMENT", "Give only one form: dPhi (Wb) or A with B1 and B2."
            )
        dphi = (
            need(p, "dPhi", "magnetic_flux", positive=False)
            if has_phi
            else need(p, "A", "area") * (_b(p, "B2") - _b(p, "B1"))
        )
        e = -turns * dphi / need(p, "dt", "time")
        out: dict[str, Any] = {"dPhi": dphi, "emf": e}
        ctx.formula = (
            "mean emf = -N dPhi / dt (Faraday's law), dPhi = A (B2 - B1) with B normal to the loop"
        )
    elif mode == "rotating_coil":
        turns = integer(p, "N") if p.get("N") is not None else 1
        if (p.get("omega") is None) == (p.get("rpm") is None):
            raise ToolError("INVALID_ARGUMENT", "Give only one of omega (rad/s) and rpm.")
        w = (
            need(p, "omega", "angular_velocity")
            if p.get("omega") is not None
            else need(p, "rpm", None) * 2 * math.pi / 60
        )
        peak = turns * _b(p, "B") * need(p, "A", "area") * w
        out = {
            "omega": w,
            "f": w / (2 * math.pi),
            "emf_peak": abs(peak),
            "emf_rms": abs(peak) / math.sqrt(2),
        }
        ctx.formula = (
            "Phi = N B A cos(w t), emf = -dPhi/dt = N B A w sin(w t) (derived from Faraday's law)"
        )
    else:
        e = _b(p, "B") * need(p, "L", "length") * need(p, "v", "velocity", positive=False)
        out = {"emf": e}
        ctx.formula = "bar moving perpendicular to B: emf = B L v (derived from Faraday's law)"
    ctx.units = units_for(out, _EMF_UNITS)
    key = "emf" if "emf" in out else "emf_peak"
    ctx.formatted = f"{'Electromotive force' if key == 'emf' else 'Peak electromotive force'}: {fmt_sig(out[key], 6)} V."
    return out
