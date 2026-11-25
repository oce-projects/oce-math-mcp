"""eng_calc: Newtonian gravitation (D88).

Sources: Wikipedia pages "List of equations in gravitation", "Shell theorem", "Gravitational binding energy" and
"Gauss's law for gravity" in the local offline library.
- Two bodies: F = G m1 m2 / r^2, U = -G m1 m2 / r, field g = G m / r^2, mu = G m.
- Point masses: g = G sum m_i (r_i - r) / |r_i - r|^3, potential Phi = -G sum m_i / |r_i - r|; F = m g, U = m Phi.
- Uniform sphere (shell theorem): outside g = G M / r^2, Phi = -G M / r; inside a solid sphere g = G M r / R^3
  and Phi = -G M (3 R^2 - r^2) / (2 R^3); inside a hollow shell g = 0 and Phi = -G M / R. Binding energy of a
  uniform solid sphere U = -3 G M^2 / (5 R).
- Ring on its axis: g = G M z / (z^2 + R^2)^(3/2), Phi = -G M / sqrt(z^2 + R^2), largest field at z = R / sqrt 2.
- Rotating body: effective gravity = gravity + centrifugal acceleration omega^2 r cos(latitude) away from the axis.
  The list page writes the centrifugal term with a minus sign while its unit vector points away from the axis;
  the centrifugal acceleration points away from the axis, so it is added here with that direction.
- Gauss's law: the flux of g through a closed surface is -4 pi G M_enclosed.
- Potential difference between r1 and r2 of a point mass: G M (r2 - r1) / (r1 r2), written without the
  cancellation of 1/r1 - 1/r2; uniform-field approximation m g(r1) h, relative error h / r1.
"""

from __future__ import annotations

import math
from typing import Any

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, one_of, only, op, qty, units_for

G = lookup("Newtonian constant of gravitation")[1]
_MAX_MASSES = 1000

_CENTRAL = {"M", "mu"}
_MODES: dict[str, set[str]] = {
    "two_bodies": {"m1", "m2", "r"},
    "point_masses": {"masses", "point", "m_test"},
    "sphere": _CENTRAL | {"rho", "R", "r", "shell", "m_test"},
    "ring": _CENTRAL | {"R", "z"},
    "rotating_body": _CENTRAL | {"R", "omega", "T_rot", "latitude_deg"},
    "gauss": {"M", "Phi_G"},
    "potential_difference": _CENTRAL | {"r1", "r2", "m"},
}
_UNITS = {
    "F": "N",
    "U": "J",
    "g1": "m/s2",
    "g2": "m/s2",
    "mu": "m3/s2",
    "mu1": "m3/s2",
    "mu2": "m3/s2",
    "g": "m/s2",
    "g_magnitude": "m/s2",
    "F_magnitude": "N",
    "potential": "J/kg",
    "M_total": "kg",
    "center_of_mass": "m",
    "g_surface": "m/s2",
    "potential_surface": "J/kg",
    "rho": "kg/m3",
    "binding_energy": "J",
    "binding_energy_per_kg": "J/kg",
    "z_max": "m",
    "g_max": "m/s2",
    "omega": "rad/s",
    "g_grav": "m/s2",
    "a_centrifugal": "m/s2",
    "g_radial": "m/s2",
    "g_tangential": "m/s2",
    "g_eff": "m/s2",
    "deflection_deg": "deg",
    "Phi_G": "m3/s2",
    "M_enclosed": "kg",
    "dPhi": "J/kg",
    "W": "J",
    "W_uniform": "J",
    "uniform_field_error": "1",
}


def _mu(p: dict[str, Any]) -> float:
    if one_of(p, ("M", "mu"), "the central body (M or mu = G M)") == "M":
        return G * need(p, "M", "mass")
    return need(p, "mu", None)


@op("gravitation", ("mode", *sorted(set().union(*_MODES.values()))))
def _gravitation(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = {
        "two_bodies": _two_bodies,
        "point_masses": _point_masses,
        "sphere": _sphere,
        "ring": _ring,
        "rotating_body": _rotating,
        "gauss": _gauss,
        "potential_difference": _potential_difference,
    }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _two_bodies(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m1, m2, r = need(p, "m1", "mass"), need(p, "m2", "mass"), need(p, "r", "length")
    force = G * m1 * m2 / (r * r)
    ctx.formula = "F = G m1 m2 / r^2, U = -G m1 m2 / r, g_i = G m_i / r^2, mu_i = G m_i"
    ctx.formatted = f"Gravitational force F = {fmt_sig(force, 6)} N at r = {fmt_sig(r, 6)} m."
    return {
        "F": force,
        "U": -force * r,
        "g1": G * m1 / (r * r),
        "g2": G * m2 / (r * r),
        "mu1": G * m1,
        "mu2": G * m2,
    }


def _vector(raw: Any, key: str) -> tuple[float, float, float]:
    if not isinstance(raw, list | tuple) or len(raw) != 3:
        raise ToolError("INVALID_ARGUMENT", f"params.{key}: a point [x, y, z] in m.")
    vals = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(float(v)):
            raise ToolError(
                "INVALID_ARGUMENT", f"params.{key}: the coordinates must be finite numbers."
            )
        vals.append(float(v))
    return vals[0], vals[1], vals[2]


def _masses(raw: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(raw, list | tuple) or not raw:
        raise ToolError(
            "INVALID_ARGUMENT", "params.masses: a non-empty list of [m, x, y, z] (kg, m)."
        )
    if len(raw) > _MAX_MASSES:
        raise ToolError("INVALID_ARGUMENT", f"params.masses: at most {_MAX_MASSES} point masses.")
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, list | tuple) or len(item) != 4:
            raise ToolError("INVALID_ARGUMENT", f"params.masses[{i}]: [m, x, y, z] (kg, m).")
        m = item[0]
        if isinstance(m, bool) or not isinstance(m, int | float) or not float(m) > 0:
            raise ToolError("INVALID_ARGUMENT", f"params.masses[{i}]: the mass must be > 0.")
        x, y, z = _vector(item[1:], f"masses[{i}]")
        out.append((float(m), x, y, z))
    return out


def _point_masses(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    masses = _masses(p.get("masses"))
    px, py, pz = _vector(p.get("point"), "point")
    g = [0.0, 0.0, 0.0]
    phi = 0.0
    for i, (m, x, y, z) in enumerate(masses):
        d = (x - px, y - py, z - pz)
        dist = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if dist == 0:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"params.point coincides with masses[{i}]: the field there is infinite.",
            )
        for k in range(3):
            g[k] += G * m * d[k] / dist**3
        phi -= G * m / dist
    total = sum(m for m, *_ in masses)
    com = [sum(m * c[k] for m, *c in masses) / total for k in range(3)]
    out: dict[str, Any] = {
        "g": g,
        "g_magnitude": math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2),
        "potential": phi,
        "M_total": total,
        "center_of_mass": com,
    }
    ctx.formula = "g = G sum m_i (r_i - r) / |r_i - r|^3, Phi = -G sum m_i / |r_i - r|"
    if p.get("m_test") is not None:
        mt = need(p, "m_test", "mass")
        out["F"] = [mt * c for c in g]
        out["F_magnitude"] = mt * out["g_magnitude"]
        out["U"] = mt * phi
        ctx.formula += "; F = m g, U = m Phi"
    ctx.formatted = (
        f"Field of {len(masses)} point masses: |g| = {fmt_sig(out['g_magnitude'], 6)} m/s2, "
        f"potential {fmt_sig(phi, 6)} J/kg."
    )
    return out


def _sphere(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    big_r, r = need(p, "R", "length"), need(p, "r", "length", positive=False)
    if r < 0:
        raise ToolError("INVALID_ARGUMENT", "params.r (distance from the centre) must be >= 0.")
    how = one_of(p, ("M", "mu", "rho"), "the sphere (M, mu = G M or the mean density rho)")
    mass: float | None
    if how == "rho":
        mass = 4 / 3 * math.pi * big_r**3 * need(p, "rho", "density")
        mu = G * mass
    elif how == "M":
        mass = need(p, "M", "mass")
        mu = G * mass
    else:
        mass, mu = None, need(p, "mu", None)
    shell = p.get("shell", False)
    if not isinstance(shell, bool):
        raise ToolError(
            "INVALID_ARGUMENT", "params.shell: true (hollow thin shell) or false (solid)."
        )
    if r >= big_r:
        g, phi = mu / (r * r), -mu / r
        ctx.formula = "outside: g = G M / r^2, Phi = -G M / r"
    elif shell:
        g, phi = 0.0, -mu / big_r
        ctx.formula = "inside a hollow shell: g = 0, Phi = -G M / R"
    else:
        g = mu * r / big_r**3
        phi = -mu * (3 * big_r**2 - r * r) / (2 * big_r**3)
        ctx.formula = (
            "inside a uniform solid sphere: g = G M r / R^3, Phi = -G M (3 R^2 - r^2) / (2 R^3)"
        )
    out: dict[str, Any] = {
        "mu": mu,
        "g": g,
        "potential": phi,
        "g_surface": mu / big_r**2,
        "potential_surface": -mu / big_r,
    }
    if mass is not None:
        out["rho"] = mass / (4 / 3 * math.pi * big_r**3)
        if not shell:
            out["binding_energy"] = -3 * mu * mass / (5 * big_r)
            out["binding_energy_per_kg"] = -3 * mu / (5 * big_r)
            ctx.formula += "; binding energy U = -3 G M^2 / (5 R)"
    if p.get("m_test") is not None:
        mt = need(p, "m_test", "mass")
        out["F"], out["U"] = mt * g, mt * phi
        ctx.formula += "; F = m g, U = m Phi"
    ctx.formatted = f"{'Hollow shell' if shell else 'Uniform sphere'}: g = {fmt_sig(g, 6)} m/s2 at r = {fmt_sig(r, 6)} m."
    return out


def _ring(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mu, big_r = _mu(p), need(p, "R", "length")
    z = need(p, "z", "length", positive=False)
    s2 = z * z + big_r * big_r
    z_max = big_r / math.sqrt(2)
    ctx.formula = (
        "g = G M z / (z^2 + R^2)^(3/2) towards the plane of the ring, Phi = -G M / sqrt(z^2 + R^2); "
        "largest field at z = R / sqrt 2"
    )
    out = {
        "mu": mu,
        "g": mu * abs(z) / s2**1.5,
        "potential": -mu / math.sqrt(s2),
        "z_max": z_max,
        "g_max": mu * z_max / (z_max * z_max + big_r * big_r) ** 1.5,
    }
    ctx.formatted = f"Ring on its axis: g = {fmt_sig(out['g'], 6)} m/s2 at z = {fmt_sig(z, 6)} m."
    return out


def _rotating(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mu, big_r = _mu(p), need(p, "R", "length")
    if one_of(p, ("omega", "T_rot"), "the rotation (omega or the period T_rot)") == "omega":
        omega = need(p, "omega", "angular_velocity")
    else:
        omega = 2 * math.pi / need(p, "T_rot", "time")
    lat = need(p, "latitude_deg", None, positive=False)
    if not -90 <= lat <= 90:
        raise ToolError("INVALID_ARGUMENT", "params.latitude_deg must be in [-90, 90] degrees.")
    t = math.radians(lat)
    g_grav = mu / big_r**2
    a_c = omega * omega * big_r * math.cos(t)
    g_radial = g_grav - a_c * math.cos(t)
    # towards the equator in both hemispheres (D89): the sign of sin(lat) only says which way that is
    g_tang = a_c * abs(math.sin(t))
    out = {
        "omega": omega,
        "g_grav": g_grav,
        "a_centrifugal": a_c,
        "g_radial": g_radial,
        "g_tangential": g_tang,
        "g_eff": math.hypot(g_radial, g_tang),
        "deflection_deg": math.degrees(math.atan2(g_tang, g_radial)),
    }
    ctx.formula = (
        "g_eff = G M / R^2 (towards the centre) + omega^2 R cos(lat) (away from the axis); radial part "
        "G M / R^2 - omega^2 R cos^2(lat), part towards the equator omega^2 R cos(lat) |sin(lat)| (>= 0 in both "
        "hemispheres)"
    )
    ctx.formatted = (
        f"Effective gravity at latitude {fmt_sig(lat, 6)} deg: {fmt_sig(out['g_eff'], 7)} m/s2 "
        f"(plumb line {fmt_sig(out['deflection_deg'], 4)} deg from the radial direction, towards the equator)."
    )
    return out


def _gauss(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if one_of(p, ("M", "Phi_G"), "Gauss's law (the enclosed mass M or the flux Phi_G)") == "M":
        mass = need(p, "M", "mass")
        flux = -4 * math.pi * G * mass
    else:
        flux = qty(p, "Phi_G", None, positive=False)
        assert flux is not None
        if flux >= 0:
            raise ToolError("INVALID_ARGUMENT", "params.Phi_G: the gravitational flux is negative.")
        mass = -flux / (4 * math.pi * G)
    ctx.formula = "closed-surface flux of g = -4 pi G M_enclosed"
    ctx.formatted = f"Gauss: flux {fmt_sig(flux, 6)} m3/s2 for {fmt_sig(mass, 6)} kg enclosed."
    return {"Phi_G": flux, "M_enclosed": mass}


def _potential_difference(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mu, r1, r2 = _mu(p), need(p, "r1", "length"), need(p, "r2", "length")
    if r1 == r2:
        raise ToolError("INVALID_ARGUMENT", "params.r1 and r2 must differ.")
    h = r2 - r1
    d_phi = mu * h / (r1 * r2)
    out: dict[str, Any] = {"dPhi": d_phi, "g1": mu / (r1 * r1), "uniform_field_error": h / r1}
    ctx.formula = (
        "dPhi = Phi(r2) - Phi(r1) = G M (r2 - r1) / (r1 r2); uniform field g(r1) (r2 - r1), "
        "relative error (r2 - r1) / r1"
    )
    if p.get("m") is not None:
        m = need(p, "m", "mass")
        out["W"] = m * d_phi
        out["W_uniform"] = m * out["g1"] * h
        ctx.formula += "; W = m dPhi"
    ctx.formatted = f"Potential difference {fmt_sig(d_phi, 6)} J/kg from r1 to r2."
    return out
