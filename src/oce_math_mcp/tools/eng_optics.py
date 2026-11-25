"""eng_calc: geometric and wave optics (D64).

Sources: Wikipedia pages in the local offline library.
- "Lens": thick-lens lensmaker equation 1/f = (n-1)[1/R1 - 1/R2 + (n-1) d / (n R1 R2)], principal planes
  h1 = -(n-1) f d / (n R2), h2 = -(n-1) f d / (n R1) (positive to the right of the vertex); thin lens in a
  medium with the relative index; spherical surface n1/u + n2/v = (n2 - n1)/R, f_0 = n1 R / (n2 - n1),
  f_i = n2 R / (n2 - n1); compound lenses 1/f = 1/f1 + 1/f2 - d/(f1 f2), FFD = f1 (d - f2) / (d - (f1 + f2)),
  BFD = f2 (d - f1) / (d - (f1 + f2)), afocal case d = f1 + f2 with M = -f2/f1. Sign of R: positive when the
  centre of curvature is further along the direction of the light.
- "Focal length": FFD = f (1 + (n-1) d / (n R2)), BFD = f (1 - (n-1) d / (n R1)); |f| = R/2 for a mirror.
- "Curved mirror": 1/d_o + 1/d_i = 1/f, m = -d_i/d_o, f > 0 for a concave mirror, distances positive in
  front of the mirror (real).
- "Polarizer" (Malus): I = I0 cos^2(theta); unpolarised light through an ideal polariser gives I0/2.
- "Brewster's angle": theta_B = arctan(n2/n1), theta_B + theta_t = 90 degrees.
- "Fresnel equations": R_s, R_p with cos(theta_t) = sqrt(1 - ((n1/n2) sin(theta_i))^2), T = 1 - R,
  unpolarised R = (R_s + R_p)/2, total internal reflection R_s = R_p = 1.
- "Double-slit experiment": maxima d sin(theta) = n lambda, fringe spacing z lambda / d.
- "Diffraction": single slit intensity I0 sinc^2((a pi / lambda) sin(theta)); first minimum a sin(theta) = lambda.
- "Diffraction grating": sin(theta_i) + sin(theta_m) = m lambda / d.
- "Thin-film interference": OPD = 2 n_film d cos(theta_2), a half-wave shift on reflection from a higher
  index; constructive reflection OPD = (m - 1/2) lambda with one shift, m lambda with zero or two.
- "Angular resolution": Rayleigh criterion, factor 1.21966989... (first zero of J1 divided by pi).
Derived here and checked in the tests: thick lens in a medium (relative index n / n_medium, same medium on
both sides; exact ray tracing), single-slit minima of order m (zeros of the sinc), grating resolving power
N |m| (Rayleigh criterion on the N-slit pattern; phasor sum).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from scipy.special import jn_zeros

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, one_of, only, op, qty, units_for

_RAYLEIGH = float(jn_zeros(1, 1)[0]) / math.pi  # 1.2196698912665045
_REL = 1e-12

_GEO: dict[str, set[str]] = {
    "lensmaker": {"n", "R1", "C1", "R2", "C2", "d", "n_medium"},
    "mirror": {"R", "f", "d_o", "d_i"},
    "refracting_surface": {"n1", "n2", "R", "d_o", "d_i"},
    "two_lenses": {"f1", "f2", "d"},
}
_WAVE: dict[str, set[str]] = {
    "malus": {"I0", "theta_deg", "incident"},
    "brewster": {"n1", "n2"},
    "fresnel": {"n1", "n2", "theta_deg"},
    "double_slit": {"wavelength", "d", "m", "L"},
    "single_slit": {"wavelength", "a", "m", "L"},
    "grating": {"wavelength", "d", "lines_per_mm", "m", "theta_i_deg", "N"},
    "thin_film": {"n1", "n_film", "n2", "wavelength", "theta_deg", "d"},
    "rayleigh": {"wavelength", "D", "L"},
}
_UNITS = {
    "f": "m",
    "R": "m",
    "d_o": "m",
    "d_i": "m",
    "P": "1/m",
    "BFD": "m",
    "FFD": "m",
    "H1": "m",
    "H2": "m",
    "f_object": "m",
    "f_image": "m",
    "theta_deg": "deg",
    "theta_t_deg": "deg",
    "theta_B_deg": "deg",
    "critical_angle_deg": "deg",
    "theta_rad": "rad",
    "theta_arcsec": "arcsec",
    "y": "m",
    "y_small_angle": "m",
    "fringe_spacing": "m",
    "central_width": "m",
    "x": "m",
    "delta_lambda": "m",
    "d_constructive": "m",
    "d_destructive": "m",
    "period": "m",
    "opd": "m",
    "phase_difference_rad": "rad",
}


def _signed(p: dict[str, Any], key: str) -> float:
    """Signed length, not zero (radii, distances with the sign convention)."""
    v = need(p, key, "length", positive=False)
    if v == 0:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must not be 0.")
    return v


def _angle(p: dict[str, Any], key: str, low: float, high: float) -> float:
    v = qty(p, key, None, positive=False, default=0.0)
    assert v is not None
    if not low <= v < high:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be in [{low:g}, {high:g}) degrees.")
    return v


def _order(p: dict[str, Any], minimum: int, default: int = 1) -> int:
    return integer(p, "m", minimum) if p.get("m") is not None else default


def _no_order(m: int, s: float, limit: str) -> ToolError:
    return ToolError(
        "INVALID_ARGUMENT",
        f"The order m = {m} does not exist: {limit} = {fmt_sig(s, 6)}, beyond 1 (the light would leave at more "
        "than 90 degrees).",
    )


def _top_order(x: float) -> int:
    """Largest integer m with m < x (strict: m = x would leave at exactly 90 degrees)."""
    m = math.floor(x)
    return m - 1 if m >= x * (1 - _REL) else m


# ---------------------------------------------------------------- geometric optics


@op("geometric_optics", ("mode", *sorted(set().union(*_GEO.values()))))
def _geometric(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_GEO))
    only(p, mode, _GEO[mode])
    if mode == "lensmaker":
        out = _lensmaker(p, ctx)
    elif mode == "mirror":
        out = _mirror(p, ctx)
    elif mode == "refracting_surface":
        out = _surface(p, ctx)
    else:
        out = _two_lenses(p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _curvature(p: dict[str, Any], r_key: str, c_key: str, what: str) -> float:
    key = one_of(p, (r_key, c_key), what)
    if key == c_key:
        return need(p, c_key, None, positive=False)  # 1/m, 0 = flat face
    return 1 / _signed(p, r_key)


def _lensmaker(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n = need(p, "n", None)
    n_m = need(p, "n_medium", None) if p.get("n_medium") is not None else 1.0
    c1 = _curvature(p, "R1", "C1", "the first surface")
    c2 = _curvature(p, "R2", "C2", "the second surface")
    d = qty(p, "d", "length", positive=False, default=0.0)
    assert d is not None
    if d < 0:
        raise ToolError("INVALID_ARGUMENT", "params.d (thickness) must be >= 0.")
    nr = n / n_m
    if nr == 1:
        raise ToolError("INVALID_ARGUMENT", "n equals n_medium: the lens does not refract.")
    k = (nr - 1) * d / nr
    dc = c1 - c2
    if p.get("R1") is not None and p.get("R2") is not None:
        # 1/R1 - 1/R2 = (R2 - R1) / (R1 R2): exact difference for a meniscus (D82); two divisions, no
        # product R1 R2 that overflows or underflows (D84)
        r1, r2 = _signed(p, "R1"), _signed(p, "R2")
        dc = (r2 - r1) / r1 / r2
    power = (nr - 1) * (dc + k * c1 * c2)
    ctx.formula = "1/f = (n-1) [1/R1 - 1/R2 + (n-1) d / (n R1 R2)] (thick lens, paraxial)"
    if n_m != 1:
        ctx.formula += "; n -> n / n_medium (same medium on both sides)"
    if d > 0:
        ctx.formula += "; BFD = f (1 - (n-1) d / (n R1)); FFD = f (1 + (n-1) d / (n R2)); H1 = -f (n-1) d / (n R2)"
        ctx.formula += "; H2 = -f (n-1) d / (n R1)"
    scale = abs(nr - 1) * (abs(c1) + abs(c2) + abs(k * c1 * c2))
    if abs(power) <= _REL * scale:
        ctx.formatted = "Lens with no optical power (afocal): parallel light leaves parallel."
        return {
            "afocal": True,
            "f": None,
            "P": 0.0,
            "BFD": None,
            "FFD": None,
            "H1": None,
            "H2": None,
        }
    f = 1 / power
    out: dict[str, Any] = {"afocal": False, "f": f, "P": n_m * power}
    out["BFD"] = f * (1 - k * c1)
    out["FFD"] = f * (1 + k * c2)
    out["H1"] = -f * k * c2
    out["H2"] = -f * k * c1
    kind = "converging" if f > 0 else "diverging"
    ctx.formatted = (
        f"Lens {kind}: f = {fmt_sig(f * 1000, 6)} mm, power {fmt_sig(n_m * power, 6)} D."
    )
    return out


def _mirror(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if p.get("R") is not None and p.get("f") is not None:
        raise ToolError("INVALID_ARGUMENT", "Mirror: give f or R, not both (f = R/2).")
    f = (
        _signed(p, "f")
        if p.get("f") is not None
        else (_signed(p, "R") / 2 if p.get("R") is not None else None)
    )
    d_o = _signed(p, "d_o") if p.get("d_o") is not None else None
    d_i = _signed(p, "d_i") if p.get("d_i") is not None else None
    if sum(v is not None for v in (f, d_o, d_i)) != 2:
        raise ToolError("INVALID_ARGUMENT", "Mirror: exactly two of f (or R), d_o, d_i are needed.")
    ctx.formula = "1/d_o + 1/d_i = 1/f; f = R/2; m = -d_i/d_o (real is positive, concave f > 0)"
    if f is None:
        assert d_o is not None and d_i is not None
        # 1/d_o + 1/d_i = (d_o + d_i) / (d_o d_i): exact sum when d_i is close to -d_o (D82)
        s = d_o + d_i
        if s == 0:
            raise ToolError("INVALID_ARGUMENT", "1/d_o + 1/d_i = 0: plane mirror, no focal length.")
        big, small = sorted((d_o, d_i), key=abs, reverse=True)
        f = small * (big / s)  # a ratio, then one product: no overflow of d_o d_i (D84)
    elif d_i is None:
        assert d_o is not None
        # 1/f - 1/d_o = (d_o - f) / (f d_o): exact difference near the focus (D80)
        # divided in two steps: no overflow of f d_o for a far object (D84)
        d_i = _conjugate((d_o - f) / d_o / f, ctx)
    else:
        d_o = _conjugate((d_i - f) / d_i / f, ctx)
    out: dict[str, Any] = {"f": f, "R": 2 * f, "d_o": d_o, "d_i": d_i, "magnification": None}
    if d_o is not None and d_i is not None:
        out["magnification"] = -d_i / d_o
        out["real_image"] = d_i > 0
        out["upright"] = out["magnification"] > 0
    kind = "concave" if f > 0 else "convex"
    text = f"Mirror {kind}: f = {fmt_sig(f * 1000, 6)} mm"
    if d_i is not None and d_o is not None:
        text += f", d_o = {fmt_sig(d_o * 1000, 6)} mm, d_i = {fmt_sig(d_i * 1000, 6)} mm"
        text += f", magnification {fmt_sig(out['magnification'], 5)}"
    ctx.formatted = text + "."
    return out


def _to_float(q: Fraction) -> float:
    """Correctly rounded float of an exact rational, +-inf beyond the float range (D84)."""
    try:
        return float(q)
    except OverflowError:
        return math.inf if q > 0 else -math.inf


def _conjugate(inv: float, ctx: Ctx) -> float | None:
    if inv == 0:
        ctx.warn("Point at the focus: the conjugate point is at infinity (parallel light).")
        return None
    return 1 / inv


def _surface(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n1, n2 = need(p, "n1", None), need(p, "n2", None)
    if n1 == n2:
        raise ToolError("INVALID_ARGUMENT", "n1 equals n2: the surface does not refract.")
    r = _signed(p, "R")
    power = (n2 - n1) / r
    key = one_of(p, ("d_o", "d_i"), "the refracting surface")
    ctx.formula = (
        "n1/d_o + n2/d_i = (n2 - n1)/R; f_object = n1 R/(n2 - n1); f_image = n2 R/(n2 - n1)"
    )
    ctx.formula += "; m = -(n1 d_i)/(n2 d_o)"
    # (n2 - n1)/R - n1/d_o = ((n2 - n1) d_o - n1 R) / (R d_o), with the numerator in exact rationals: no
    # cancellation near the focus (D82); the same for d_i
    dn = Fraction(n2) - Fraction(n1)
    if key == "d_o":
        x = _signed(p, "d_o")
        d_o: float | None = x
        num = dn * Fraction(x) - Fraction(n1) * Fraction(r)
        d_i = _conjugate(_to_float(num / (Fraction(n2) * Fraction(r) * Fraction(x))), ctx)
    else:
        y = _signed(p, "d_i")
        d_i = y
        num = dn * Fraction(y) - Fraction(n2) * Fraction(r)
        d_o = _conjugate(_to_float(num / (Fraction(n1) * Fraction(r) * Fraction(y))), ctx)
    out: dict[str, Any] = {
        "P": power,
        "f_object": n1 / power,
        "f_image": n2 / power,
        "d_o": d_o,
        "d_i": d_i,
    }
    out["magnification"] = -(n1 * d_i) / (n2 * d_o) if d_o is not None and d_i is not None else None
    text = f"Surface: power {fmt_sig(power, 6)} D"
    if d_i is not None and d_o is not None:
        text += f", d_o = {fmt_sig(d_o * 1000, 6)} mm, d_i = {fmt_sig(d_i * 1000, 6)} mm"
    ctx.formatted = text + "."
    return out


def _two_lenses(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    f1, f2 = _signed(p, "f1"), _signed(p, "f2")
    d = qty(p, "d", "length", positive=False, default=0.0)
    assert d is not None
    if d < 0:
        raise ToolError("INVALID_ARGUMENT", "params.d (distance between the lenses) must be >= 0.")
    den = f1 + f2 - d
    ctx.formula = "1/f = 1/f1 + 1/f2 - d/(f1 f2); FFD = f1 (d - f2)/(d - (f1 + f2)); BFD = f2 (d - f1)/(d - (f1 + f2))"
    if abs(den) <= _REL * (abs(f1) + abs(f2) + d):
        ctx.formula = "d = f1 + f2: afocal system, angular magnification M = -f2/f1"
        mag = -f2 / f1
        ctx.formatted = f"Afocal system (telescope): angular magnification {fmt_sig(mag, 6)}."
        return {
            "afocal": True,
            "f": None,
            "P": 0.0,
            "BFD": None,
            "FFD": None,
            "angular_magnification": mag,
        }
    f = f1 * f2 / den
    out = {
        "afocal": False,
        "f": f,
        "P": 1 / f,
        "BFD": f2 * (d - f1) / (d - (f1 + f2)),
        "FFD": f1 * (d - f2) / (d - (f1 + f2)),
        "angular_magnification": None,
    }
    ctx.formatted = (
        f"Two lenses: f = {fmt_sig(f * 1000, 6)} mm, BFD = {fmt_sig(out['BFD'] * 1000, 6)} mm, "
        f"FFD = {fmt_sig(out['FFD'] * 1000, 6)} mm."
    )
    return out


# ---------------------------------------------------------------- wave optics


@op("wave_optics", ("mode", *sorted(set().union(*_WAVE.values()))))
def _wave(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_WAVE))
    only(p, mode, _WAVE[mode])
    out = {
        "malus": _malus,
        "brewster": _brewster,
        "fresnel": _fresnel_mode,
        "double_slit": _double_slit,
        "single_slit": _single_slit,
        "grating": _grating,
        "thin_film": _thin_film,
        "rayleigh": _rayleigh,
    }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _malus(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    i0 = need(p, "I0", None)
    theta = qty(p, "theta_deg", None, positive=False, default=0.0)
    assert theta is not None
    incident = choice(p, "incident", ("polarized", "unpolarized"), "polarized")
    c2 = math.cos(math.radians(theta)) ** 2
    if incident == "polarized":
        ratio = c2
        ctx.formula = "I = I0 cos^2(theta) (Malus)"
    else:
        ratio = c2 / 2
        ctx.formula = "I = (I0/2) cos^2(theta): the first polariser halves unpolarised light, theta between the two axes"
    ctx.formatted = (
        f"Transmitted intensity {fmt_sig(i0 * ratio, 6)} ({fmt_sig(100 * ratio, 5)}% of I0)."
    )
    ctx.warn("Ideal polarisers; I has the unit of I0.")
    return {"I": i0 * ratio, "ratio": ratio}


def _fresnel(n1: float, n2: float, theta: float) -> dict[str, Any]:
    ti = math.radians(theta)
    s = n1 * math.sin(ti) / n2
    out: dict[str, Any] = {
        "critical_angle_deg": math.degrees(math.asin(n2 / n1)) if n1 > n2 else None
    }
    if s >= 1:
        out.update(
            theta_t_deg=None,
            R_s=1.0,
            R_p=1.0,
            T_s=0.0,
            T_p=0.0,
            R_unpolarized=1.0,
            T_unpolarized=0.0,
        )
        out["total_internal_reflection"] = True
        return out
    ci, ct = math.cos(ti), math.sqrt(1 - s * s)
    r_s = ((n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)) ** 2
    r_p = ((n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)) ** 2
    out.update(theta_t_deg=math.degrees(math.asin(s)), R_s=r_s, R_p=r_p, T_s=1 - r_s, T_p=1 - r_p)
    out.update(
        R_unpolarized=(r_s + r_p) / 2,
        T_unpolarized=1 - (r_s + r_p) / 2,
        total_internal_reflection=False,
    )
    return out


def _fresnel_mode(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n1, n2 = need(p, "n1", None), need(p, "n2", None)
    theta = _angle(p, "theta_deg", 0.0, 90.0)
    out = _fresnel(n1, n2, theta)
    ctx.formula = (
        "R_s = ((n1 cos ti - n2 cos tt)/(n1 cos ti + n2 cos tt))^2; R_p = ((n1 cos tt - n2 cos ti)/(n1 cos tt + "
        "n2 cos ti))^2; T = 1 - R; unpolarised R = (R_s + R_p)/2"
    )
    if out["total_internal_reflection"]:
        ctx.formatted = f"Total internal reflection (critical angle {fmt_sig(out['critical_angle_deg'], 6)} degrees)."
    else:
        ctx.formatted = (
            f"Reflectance R_s = {fmt_sig(out['R_s'], 6)}, R_p = {fmt_sig(out['R_p'], 6)}, "
            f"unpolarised {fmt_sig(out['R_unpolarized'], 6)}."
        )
    return out


def _brewster(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n1, n2 = need(p, "n1", None), need(p, "n2", None)
    if n1 == n2:
        raise ToolError("INVALID_ARGUMENT", "n1 equals n2: no interface, no Brewster angle.")
    theta_b = math.degrees(math.atan2(n2, n1))
    fr = _fresnel(n1, n2, theta_b)
    ctx.formula = "theta_B = arctan(n2/n1); theta_B + theta_t = 90 degrees"
    ctx.formatted = (
        f"Brewster angle {fmt_sig(theta_b, 6)} degrees: reflected light fully s-polarised."
    )
    return {
        "theta_B_deg": theta_b,
        "theta_t_deg": 90.0 - theta_b,
        "R_s": fr["R_s"],
        "R_p": fr["R_p"],
    }


def _double_slit(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    lam, d = need(p, "wavelength", "length"), need(p, "d", "length")
    m = _order(p, 0)
    s = m * lam / d
    m_max = _top_order(d / lam)
    if s >= 1:
        raise _no_order(m, s, "m lambda / d")
    theta = math.asin(s)
    out: dict[str, Any] = {"theta_deg": math.degrees(theta), "m_max": m_max}
    ctx.formula = "d sin(theta) = m lambda (bright fringes)"
    length = qty(p, "L", "length", need=False)
    if length is not None:
        out["y"] = length * math.tan(theta)
        out["y_small_angle"] = m * lam * length / d
        out["fringe_spacing"] = lam * length / d
        ctx.formula += "; y = L tan(theta); fringe spacing = L lambda / d"
        ctx.formatted = (
            f"Bright fringe m = {m} at {fmt_sig(out['theta_deg'], 6)} degrees, y = {fmt_sig(out['y'] * 1000, 6)} mm; "
            f"fringe spacing {fmt_sig(out['fringe_spacing'] * 1000, 6)} mm."
        )
    else:
        ctx.formatted = f"Bright fringe m = {m} at {fmt_sig(out['theta_deg'], 6)} degrees; highest order {m_max}."
    return out


def _single_slit(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    lam, a = need(p, "wavelength", "length"), need(p, "a", "length")
    m = _order(p, 1)
    s = m * lam / a
    if s >= 1:
        raise _no_order(m, s, "m lambda / a")
    theta = math.asin(s)
    out: dict[str, Any] = {"theta_deg": math.degrees(theta)}
    ctx.formula = "a sin(theta) = m lambda (minima, zeros of I0 sinc^2((a pi / lambda) sin(theta)))"
    length = qty(p, "L", "length", need=False)
    if length is not None:
        out["y"] = length * math.tan(theta)
        out["central_width"] = 2 * length * math.tan(math.asin(lam / a))
        ctx.formula += "; y = L tan(theta); central maximum 2 L tan(theta_1)"
    ctx.formatted = f"Minimum m = {m} at {fmt_sig(out['theta_deg'], 6)} degrees."
    return out


def _grating(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    lam = need(p, "wavelength", "length")
    key = one_of(p, ("d", "lines_per_mm"), "the grating period")
    d = need(p, "d", "length") if key == "d" else 1e-3 / need(p, "lines_per_mm", None)
    m = integer(p, "m", -(10**9)) if p.get("m") is not None else 1
    s_i = math.sin(math.radians(_angle(p, "theta_i_deg", -90.0, 90.0)))
    m_max = _top_order((1 + s_i) * d / lam)
    m_min = -_top_order((1 - s_i) * d / lam)
    s = m * lam / d - s_i
    if abs(s) >= 1:
        raise _no_order(m, abs(s), "|m lambda / d - sin(theta_i)|")
    out: dict[str, Any] = {
        "theta_deg": math.degrees(math.asin(s)),
        "d": d,
        "m_max": m_max,
        "m_min": m_min,
    }
    ctx.formula = "sin(theta_i) + sin(theta_m) = m lambda / d (m = 0 gives theta_m = -theta_i)"
    if p.get("N") is not None and m != 0:
        n_lines = integer(p, "N", 1)
        out["resolving_power"] = n_lines * abs(m)
        out["delta_lambda"] = lam / (n_lines * abs(m))
        ctx.formula += "; resolving power N |m| (Rayleigh criterion on the N-slit pattern)"
    ctx.formatted = (
        f"Order m = {m} at {fmt_sig(out['theta_deg'], 6)} degrees; orders from {m_min} to {m_max}."
    )
    return out


def _thin_film(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n1 = need(p, "n1", None) if p.get("n1") is not None else 1.0
    nf, n2 = need(p, "n_film", None), need(p, "n2", None)
    lam = need(p, "wavelength", "length")
    if nf in (n1, n2):
        raise ToolError(
            "INVALID_ARGUMENT", "n_film equals n1 or n2: one interface does not reflect."
        )
    s = n1 * math.sin(math.radians(_angle(p, "theta_deg", 0.0, 90.0))) / nf
    if s >= 1:
        raise ToolError(
            "INVALID_ARGUMENT", "Total internal reflection at the top of the film: no light enters."
        )
    cos2 = math.sqrt(1 - s * s)
    half = (n1 < nf) != (
        nf < n2
    )  # one half-wave shift (reflection from a higher index) and not two
    period = lam / (2 * nf * cos2)
    out: dict[str, Any] = {"half_wave_shift": half, "period": period}
    out["d_constructive"] = period / 2 if half else period
    out["d_destructive"] = period if half else period / 2
    ctx.formula = (
        "OPD = 2 n_film d cos(theta_2); reflection constructive at OPD = (m - 1/2) lambda with one "
    )
    ctx.formula += "half-wave shift, m lambda with zero or two"
    text = (
        f"Thinnest film for a reflected maximum {fmt_sig(out['d_constructive'] * 1e9, 6)} nm, "
        f"for a minimum {fmt_sig(out['d_destructive'] * 1e9, 6)} nm"
    )
    d = qty(p, "d", "length", need=False)
    if d is not None:
        opd = 2 * nf * d * cos2
        phase = 2 * math.pi * opd / lam + (math.pi if half else 0.0)
        frac = (phase / (2 * math.pi)) % 1.0
        dist = min(frac, 1 - frac)
        out["opd"] = opd
        out["phase_difference_rad"] = phase
        out["reflection"] = (
            "constructive"
            if dist < 1e-9
            else ("destructive" if abs(dist - 0.5) < 1e-9 else "partial")
        )
        text += f"; with d = {fmt_sig(d * 1e9, 6)} nm the reflection is {out['reflection']}"
    ctx.formatted = text + "."
    ctx.warn(
        "Two-beam condition: it gives the thicknesses of the extrema, not the reflectance value."
    )
    return out


def _rayleigh(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    lam, diam = need(p, "wavelength", "length"), need(p, "D", "length")
    s = _RAYLEIGH * lam / diam
    if s >= 1:
        raise ToolError(
            "INVALID_ARGUMENT", "1.22 lambda / D >= 1: the aperture is smaller than the wavelength."
        )
    theta = math.asin(s)
    out: dict[str, Any] = {"theta_rad": theta, "theta_deg": math.degrees(theta)}
    out["theta_arcsec"] = math.degrees(theta) * 3600
    ctx.formula = (
        "sin(theta) = 1.21967 lambda / D (first zero of the Airy pattern; about 1.22 lambda / D)"
    )
    length = qty(p, "L", "length", need=False)
    if length is not None:
        out["x"] = length * math.tan(theta)
        ctx.formula += "; x = L tan(theta)"
    ctx.formatted = (
        f"Angular resolution {fmt_sig(out['theta_arcsec'], 6)} arcsec ({fmt_sig(theta, 6)} rad)."
    )
    return out
