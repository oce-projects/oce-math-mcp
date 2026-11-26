"""eng_calc: nuclear scattering, the Breit-Wigner resonance and the Rutherford and Mott cross sections (D105).

Source: Wikipedia "List of equations in nuclear and particle physics", section "Nuclear scattering theory", in
the local offline library. E is the kinetic energy in the centre-of-mass frame, theta (chi on the page) the
scattering angle in that frame, alpha = Z1 Z2 e^2 / (4 pi eps0) the coupling of the Coulomb potential
V = alpha / r (the page writes -alpha / r: the sign leaves the cross sections, it moves the turning point).
- Breit-Wigner: sigma(E) = (pi g / k^2) Gamma_ab Gamma_c / ((E - E0)^2 + Gamma^2 / 4) with
  g = (2J + 1) / ((2 s_a + 1)(2 s_b + 1)), Gamma = Gamma_ab + Gamma_c, mean lifetime tau = hbar / Gamma;
  k = sqrt(2 mu E) / hbar when the reduced mass mu is given instead of k.
- Rutherford: dsigma/dOmega = (alpha / (4E))^2 csc^4(theta / 2) = (1/n) dN/dOmega (n: the fluence); the classical
  orbit has impact parameter b = |alpha| / (2E) cot(theta / 2) and turning point
  r_min = (alpha + |alpha| csc(theta / 2)) / (2E).
- Mott (two identical particles of spin s): dsigma/dOmega = (alpha / (4E))^2 [csc^4(theta / 2) + sec^4(theta / 2)
  + A cos(eta ln tan^2(theta / 2)) csc^2(theta / 2) sec^2(theta / 2)], A = 2 (-1)^(2s) / (2s + 1),
  eta = alpha / (hbar v), v the relative speed (mu = m / 2). The page prints the argument of the cosine and
  the last factor with misprints; this is the form of the symmetrized Coulomb amplitudes (Landau-Lifshitz,
  Quantum mechanics, par. 137), checked in tests/test_nuclear_scattering_d105.py.
"""

from __future__ import annotations

import math
from typing import Any

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, one_of, only, op, qty, range_guard, units_for

_MODES: dict[str, set[str]] = {
    "breit_wigner": {"J", "s_a", "s_b", "E", "E0", "Gamma_ab", "Gamma_c", "k", "mu"},
    "rutherford": {"E", "theta_deg", "alpha", "Z1", "Z2", "fluence"},
    "mott": {"E", "theta_deg", "alpha", "Z", "s", "m", "v"},
}
_UNITS = {
    "Gamma": "J",
    "Gamma_MeV": "MeV",
    "tau": "s",
    "k": "1/m",
    "sigma": "m2",
    "sigma_barn": "b",
    "sigma_peak": "m2",
    "sigma_peak_barn": "b",
    "alpha": "J m",
    "dsigma_dOmega": "m2/sr",
    "dsigma_dOmega_barn": "b/sr",
    "b": "m",
    "r_min": "m",
    "dN_dOmega": "1/sr",
    "v": "m/s",
    "dsigma_no_interference": "m2/sr",
    "dsigma_rutherford": "m2/sr",
}
_BARN = 1e-28  # m2
_EV = 1.602176634e-19  # J, exact in the 2019 SI


def _hbar() -> float:
    return lookup("hbar")[1]


def _e2() -> float:
    """e^2 / (4 pi eps0) in J m (1.44 MeV fm)."""
    return lookup("e")[1] ** 2 / (4 * math.pi * lookup("epsilon_0")[1])


@op("nuclear_scattering", ("mode", *sorted(set().union(*_MODES.values()))))
@range_guard
def _scattering(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = {"breit_wigner": _breit_wigner, "rutherford": _rutherford, "mott": _mott}[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _spin(p: dict[str, Any], key: str) -> float:
    """A spin: 0, 1/2, 1, 3/2, ..."""
    v = need(p, key, None, positive=False)
    if v < 0 or abs(2 * v - round(2 * v)) > 1e-9:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: a spin, a non-negative multiple of 1/2 (0, 0.5, 1, 1.5, ...).",
        )
    return round(2 * v) / 2


def _angle(p: dict[str, Any]) -> float:
    """Scattering angle in the centre-of-mass frame, degrees, 0 < theta < 180; returns theta / 2 in radians."""
    v = need(p, "theta_deg", None, positive=False)
    if not 0 < v < 180:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.theta_deg: the scattering angle in degrees (centre-of-mass frame), 0 < theta_deg < 180.",
        )
    return math.radians(v) / 2


def _charge(p: dict[str, Any], key: str) -> int:
    v = p.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float) or v != int(v) or v == 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: a charge number, a non-zero integer (1 for a proton).",
        )
    return int(v)


def _coupling(p: dict[str, Any], charges: tuple[str, ...]) -> float:
    """alpha in J m, from params.alpha or from the charge numbers (one Z: two identical particles)."""
    src = one_of(p, ("alpha", charges[0]), "the coupling alpha = Z1 Z2 e^2 / (4 pi eps0)")
    if src == "alpha":
        for key in charges[1:]:
            if p.get(key) is not None:
                raise ToolError(
                    "INVALID_ARGUMENT", f"params.{key} goes with {charges[0]}, not with alpha."
                )
        v = qty(p, "alpha", "coupling", positive=False)
        assert v is not None
        if v == 0:
            raise ToolError("INVALID_ARGUMENT", "params.alpha must not be 0.")
        return v
    zs = [_charge(p, key) for key in charges]
    prod = zs[0] * zs[0] if len(zs) == 1 else zs[0] * zs[1]
    return prod * _e2()


def _breit_wigner(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    j, sa, sb = (_spin(p, key) for key in ("J", "s_a", "s_b"))
    g = (2 * j + 1) / ((2 * sa + 1) * (2 * sb + 1))
    e0 = need(p, "E0", "energy")
    gab, gc = need(p, "Gamma_ab", "energy"), need(p, "Gamma_c", "energy")
    e = qty(p, "E", "energy", default=e0)
    assert e is not None
    hbar = _hbar()
    if one_of(p, ("k", "mu"), "the wavenumber") == "k":
        k = need(p, "k", "wavenumber")
        k0 = k
    else:
        mu = need(p, "mu", "mass")
        k, k0 = math.sqrt(2 * mu * e) / hbar, math.sqrt(2 * mu * e0) / hbar
    gam = gab + gc
    sigma = math.pi * g / (k * k) * gab * gc / ((e - e0) ** 2 + gam * gam / 4)
    peak = 4 * math.pi * g * gab * gc / (k0 * k0 * gam * gam)
    ctx.formula = (
        "sigma(E) = (pi g / k^2) Gamma_ab Gamma_c / ((E - E0)^2 + Gamma^2 / 4), "
        "g = (2J + 1) / ((2 s_a + 1)(2 s_b + 1)), Gamma = Gamma_ab + Gamma_c, tau = hbar / Gamma"
        + ("" if p.get("k") is not None else ", k = sqrt(2 mu E) / hbar")
    )
    ctx.formatted = f"sigma = {fmt_sig(sigma / _BARN)} b (peak {fmt_sig(peak / _BARN)} b at E0)"
    return {
        "g": g,
        "Gamma": gam,
        "Gamma_MeV": gam / (1e6 * _EV),
        "tau": hbar / gam,
        "k": k,
        "sigma": sigma,
        "sigma_barn": sigma / _BARN,
        "sigma_peak": peak,
        "sigma_peak_barn": peak / _BARN,
    }


def _rutherford(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    e = need(p, "E", "energy")
    half = _angle(p)
    alpha = _coupling(p, ("Z1", "Z2"))
    s = math.sin(half)
    dsig = (alpha / (4 * e)) ** 2 / s**4
    out: dict[str, Any] = {
        "alpha": alpha,
        "dsigma_dOmega": dsig,
        "dsigma_dOmega_barn": dsig / _BARN,
        "b": abs(alpha) / (2 * e) * math.cos(half) / s,
        "r_min": (alpha + abs(alpha) / s) / (2 * e),
    }
    n = qty(p, "fluence", "fluence", need=False)
    if n is not None:
        out["dN_dOmega"] = n * dsig
    ctx.formula = (
        "dsigma/dOmega = (alpha / (4E))^2 csc^4(theta / 2) = (1/n) dN/dOmega, b = |alpha| / (2E) cot(theta / 2), "
        "r_min = (alpha + |alpha| csc(theta / 2)) / (2E)"
    )
    ctx.formatted = f"dsigma/dOmega = {fmt_sig(dsig)} m2/sr = {fmt_sig(dsig / _BARN)} b/sr"
    return out


def _mott(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    e = need(p, "E", "energy")
    half = _angle(p)
    alpha = _coupling(p, ("Z",))
    spin = _spin(p, "s")
    hbar = _hbar()
    if one_of(p, ("m", "v"), "the relative speed") == "m":
        m = need(p, "m", "mass")
        v = math.sqrt(4 * e / m)  # E = mu v^2 / 2 with mu = m / 2
        out_k: float | None = (m / 2) * v / hbar
    else:
        v = need(p, "v", "velocity")
        out_k = None
    c = lookup("c")[1]
    if v > 0.1 * c:
        ctx.warn(
            f"v = {fmt_sig(v / c, 3)} c: the Mott formula here is non-relativistic (corrections of the order of "
            "(v/c)^2)."
        )
    eta = alpha / (hbar * v)
    a_coef = 2 * (-1) ** round(2 * spin) / (2 * spin + 1)
    s2, c2 = math.sin(half) ** 2, math.cos(half) ** 2
    pref = (alpha / (4 * e)) ** 2
    plain = pref * (1 / (s2 * s2) + 1 / (c2 * c2))
    dsig = plain + pref * a_coef * math.cos(eta * math.log(s2 / c2)) / (s2 * c2)
    out: dict[str, Any] = {
        "eta": eta,
        "A": a_coef,
        "v": v,
        "dsigma_dOmega": dsig,
        "dsigma_dOmega_barn": dsig / _BARN,
        "dsigma_no_interference": plain,
        "dsigma_rutherford": pref / (s2 * s2),
    }
    if out_k is not None:
        out["k"] = out_k
    ctx.formula = (
        "dsigma/dOmega = (alpha / (4E))^2 [csc^4(theta/2) + sec^4(theta/2) + A cos(eta ln tan^2(theta/2)) "
        "csc^2(theta/2) sec^2(theta/2)], A = 2 (-1)^(2s) / (2s + 1), eta = alpha / (hbar v)"
    )
    ctx.formatted = f"dsigma/dOmega = {fmt_sig(dsig)} m2/sr = {fmt_sig(dsig / _BARN)} b/sr"
    return out
