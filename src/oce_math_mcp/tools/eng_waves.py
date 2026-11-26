"""eng_calc: general wave functions and modulation (D100).

Source: Wikipedia "List of equations in wave theory", sections "Modulation indices" and "General wave functions",
in the local offline library.
- Modulation indices: h_AM = A_m / A_c (amplitude of the modulating signal over the carrier), h_FM = Delta f / f_m,
  h_PM = Delta phi. AM: envelope A_c (1 +- h), sidebands carry h^2 / (2 + h^2) of the power (sine modulation);
  FM: Carson bandwidth 2 (Delta f + f_m).
- Damped wave A = A0 exp(-b t) sin(k x - omega t + phi).
- Superposition of A_n = |A_n| exp(i (k_n . r - omega_n t + phi_n)): A = sum A_n and
  |A| = sqrt(sum_n sum_m |A_n| |A_m| cos((k_n - k_m) . r - (omega_n - omega_m) t + phi_n - phi_m)).
- Standing wave: A sin(k x - omega t) + A sin(k x + omega t) = 2 A sin(k x) cos(omega t); nodes at x = n pi / k.
- Beats: A sin(k1 x - omega1 t) + A sin(k2 x - omega2 t) = 2 A sin(<k> x - <omega> t) cos(Delta k x / 2 -
  Delta omega t / 2), <omega> = (omega1 + omega2) / 2, Delta omega = omega1 - omega2 (the page writes + omega2 t
  in the second wave, a misprint: the product form needs both waves travelling the same way).
- Phase difference: A sin(k x - omega t) + A sin(k x - omega t + phi) = 2 A cos(phi / 2) sin(k x - omega t + phi / 2)
  (the page's left side has 2 A in the first term, a misprint).
- KdV soliton A(x, t) = (3 v / alpha) sech^2(sqrt(v) / 2 (x - v t)) of dy/dt + alpha y dy/dx + d3y/dx3 = 0.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, one_of, only, op, qty, range_guard, units_for

_MODES: dict[str, set[str]] = {
    "modulation": {"kind", "A_mod", "A_carrier", "delta_f", "f_m", "delta_phi"},
    "damped": {"A0", "b", "k", "omega", "phi", "x", "t"},
    "superposition": {"waves", "point", "t"},
    "standing": {"A", "k", "wavelength", "omega", "f", "x", "t", "L"},
    "beats": {"A", "k1", "omega1", "k2", "omega2", "x", "t"},
    "phase_shift": {"A", "phi", "k", "omega", "x", "t"},
    "soliton": {"v", "alpha", "x", "t"},
}
# amplitudes carry the unit of the input amplitude: no entry here
_UNITS = {
    "h": "1",
    "sideband_power_fraction": "1",
    "carson_bandwidth": "Hz",
    "delta_f": "Hz",
    "tau": "s",
    "half_life": "s",
    "phase": "rad",
    "nodes": "m",
    "antinodes": "m",
    "k": "rad/m",
    "omega": "rad/s",
    "wavelength": "m",
    "omega_mean": "rad/s",
    "k_mean": "rad/m",
    "delta_omega": "rad/s",
    "delta_k": "rad/m",
    "f_beat": "Hz",
    "T_beat": "s",
    "v_phase": "m/s",
    "v_group": "m/s",
    "fwhm": "m",
}
_NODES_MAX = 10_000


@op("waves", ("mode", *sorted(set().union(*_MODES.values()))))
@range_guard
def _waves(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = {
        "modulation": _modulation,
        "damped": _damped,
        "superposition": _superposition,
        "standing": _standing,
        "beats": _beats,
        "phase_shift": _phase_shift,
        "soliton": _soliton,
    }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _real(p: dict[str, Any], key: str, default: float | None = None) -> float:
    """Signed pure number (positions, times, phases); default when missing."""
    v = qty(p, key, None, need=default is None, positive=False, default=default)
    assert v is not None
    return v


def _modulation(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    kind = choice(p, "kind", ("AM", "FM", "PM"))
    allowed = {"AM": {"A_mod", "A_carrier"}, "FM": {"delta_f", "f_m"}, "PM": {"delta_phi", "f_m"}}[
        kind
    ]
    extra = sorted(k for k in p if k not in allowed | {"mode", "kind"})
    if extra:
        raise ToolError("INVALID_ARGUMENT", f"Keys not used by kind '{kind}': {', '.join(extra)}.")
    if kind == "AM":
        a_m, a_c = need(p, "A_mod", None), need(p, "A_carrier", None)
        h = a_m / a_c
        if h > 1:
            ctx.warn("h_AM > 1: over-modulation, the envelope crosses zero and is distorted.")
        ctx.formula = (
            "h_AM = A_m / A_c; envelope A_c (1 +- h); sideband power h^2 / (2 + h^2) of the total"
        )
        return {
            "h": h,
            "envelope_max": a_c * (1 + h),
            "envelope_min": a_c * (1 - h),
            "sideband_power_fraction": h * h / (2 + h * h),
        }
    if kind == "FM":
        df, fm = need(p, "delta_f", "frequency"), need(p, "f_m", "frequency")
        ctx.formula = "h_FM = Delta f / f_m; Carson bandwidth 2 (Delta f + f_m)"
        return {"h": df / fm, "carson_bandwidth": 2 * (df + fm)}
    h = need(p, "delta_phi", None)
    ctx.formula = "h_PM = Delta phi (peak phase deviation, rad)"
    out: dict[str, Any] = {"h": h}
    if p.get("f_m") is not None:
        out["delta_f"] = h * need(p, "f_m", "frequency")
        ctx.formula += "; equivalent peak frequency deviation Delta f = h f_m"
    return out


def _damped(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a0, b = _real(p, "A0"), need(p, "b", None)
    k, w, phi = _real(p, "k"), _real(p, "omega"), _real(p, "phi", 0.0)
    x, t = _real(p, "x"), _real(p, "t")
    env = a0 * math.exp(-b * t)
    ctx.formula = "A = A0 exp(-b t) sin(k x - omega t + phi); tau = 1 / b"
    return {
        "A": env * math.sin(k * x - w * t + phi),
        "envelope": env,
        "tau": 1 / b,
        "half_life": math.log(2) / b,
    }


def _vector(raw: Any, what: str) -> list[float]:
    vals = raw if isinstance(raw, list) else [raw]
    if not 1 <= len(vals) <= 3:
        raise ToolError("INVALID_ARGUMENT", f"{what}: a number or a list of 1 to 3 numbers.")
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ToolError("INVALID_ARGUMENT", f"{what}: finite numbers only.")
    return [float(v) for v in vals]


def _superposition(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    waves = p.get("waves")
    if not isinstance(waves, list) or not waves:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.waves is a list of [A, k, omega, phi] (k a number or [kx, ky, kz]).",
            '{"waves": [[1, 2, 3, 0], [0.5, 2.1, 3.2, 1]], "point": 0.4, "t": 0}',
        )
    if len(waves) > 1000:
        raise ToolError("LIMIT_EXCEEDED", "At most 1000 waves.")
    r = _vector(p.get("point", 0.0), "params.point")
    t = _real(p, "t", 0.0)
    terms: list[tuple[float, float]] = []  # (|A_n|, phase_n)
    for i, wv in enumerate(waves):
        if not isinstance(wv, list) or len(wv) != 4:
            raise ToolError("INVALID_ARGUMENT", f"params.waves[{i}] is [A, k, omega, phi].")
        amp = _vector(wv[0], f"params.waves[{i}] A")[0]
        kv = _vector(wv[1], f"params.waves[{i}] k")
        w = _vector(wv[2], f"params.waves[{i}] omega")[0]
        phi = _vector(wv[3], f"params.waves[{i}] phi")[0]
        if len(kv) != len(r):
            raise ToolError(
                "INVALID_ARGUMENT", f"params.waves[{i}]: k and point need the same size."
            )
        dot = math.fsum(a * b for a, b in zip(kv, r, strict=True))
        terms.append((abs(amp), dot - w * t + phi + (math.pi if amp < 0 else 0.0)))
    total = sum((a * cmath.exp(1j * ph) for a, ph in terms), 0j)
    pairs = math.fsum(an * am * math.cos(pn - pm) for an, pn in terms for am, pm in terms)
    ctx.formula = (
        "A = sum |A_n| exp(i (k_n . r - omega_n t + phi_n)); "
        "|A| = sqrt(sum_n sum_m |A_n| |A_m| cos(phase_n - phase_m))"
    )
    return {
        "re": total.real,
        "im": total.imag,
        "amplitude": abs(total),
        "amplitude_pairs": math.sqrt(max(pairs, 0.0)),
        "phase": cmath.phase(total),
    }


def _wave_numbers(p: dict[str, Any]) -> tuple[float, float]:
    if one_of(p, ("k", "wavelength"), "the wave number (k or wavelength)") == "k":
        k = need(p, "k", None)
    else:
        k = 2 * math.pi / need(p, "wavelength", "length")
    if one_of(p, ("omega", "f"), "the frequency (omega or f)") == "omega":
        w = need(p, "omega", None)
    else:
        w = 2 * math.pi * need(p, "f", "frequency")
    return k, w


def _standing(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a = _real(p, "A")
    k, w = _wave_numbers(p)
    out: dict[str, Any] = {"k": k, "omega": w, "wavelength": 2 * math.pi / k}
    ctx.formula = "y = A sin(k x - omega t) + A sin(k x + omega t) = 2 A sin(k x) cos(omega t)"
    if p.get("x") is not None or p.get("t") is not None:
        x, t = _real(p, "x", 0.0), _real(p, "t", 0.0)
        out["y"] = 2 * a * math.sin(k * x) * math.cos(w * t)
    if p.get("L") is not None:
        length = need(p, "L", "length")
        step = math.pi / k
        count = math.floor(length / step * (1 + 1e-12)) + 1
        if count > _NODES_MAX:
            raise ToolError("LIMIT_EXCEEDED", f"More than {_NODES_MAX} nodes in L.")
        out["nodes"] = [n * step for n in range(count)]
        out["antinodes"] = [(n + 0.5) * step for n in range(count) if (n + 0.5) * step <= length]
        ctx.formula += "; nodes x = n pi / k, antinodes x = (n + 1/2) pi / k in [0, L]"
    return out


def _beats(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a = _real(p, "A", 1.0)
    k1, w1, k2, w2 = (need(p, key, None) for key in ("k1", "omega1", "k2", "omega2"))
    wm, km, dw, dk = (w1 + w2) / 2, (k1 + k2) / 2, w1 - w2, k1 - k2
    out: dict[str, Any] = {
        "omega_mean": wm,
        "k_mean": km,
        "delta_omega": dw,
        "delta_k": dk,
        "f_beat": abs(dw) / (2 * math.pi),
        "v_phase": wm / km,
    }
    if dw != 0:
        out["T_beat"] = 2 * math.pi / abs(dw)
    if dk != 0:
        out["v_group"] = dw / dk
    ctx.formula = (
        "<omega> = (omega1 + omega2) / 2, Delta omega = omega1 - omega2; f_beat = |Delta omega| / (2 pi); "
        "y = 2 A sin(<k> x - <omega> t) cos(Delta k x / 2 - Delta omega t / 2)"
    )
    if p.get("x") is not None or p.get("t") is not None:
        x, t = _real(p, "x", 0.0), _real(p, "t", 0.0)
        out["y"] = 2 * a * math.sin(km * x - wm * t) * math.cos(dk * x / 2 - dw * t / 2)
        out["y_direct"] = a * math.sin(k1 * x - w1 * t) + a * math.sin(k2 * x - w2 * t)
    return out


def _phase_shift(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a, phi = _real(p, "A"), _real(p, "phi")
    c = 2 * a * math.cos(phi / 2)  # a negative factor is a phase of pi more
    out: dict[str, Any] = {"amplitude": abs(c), "phase": phi / 2 + (math.pi if c < 0 else 0.0)}
    ctx.formula = "A sin(k x - omega t) + A sin(k x - omega t + phi) = 2 A cos(phi / 2) sin(k x - omega t + phi / 2)"
    if any(p.get(key) is not None for key in ("k", "omega", "x", "t")):
        k, w = _real(p, "k", 0.0), _real(p, "omega", 0.0)
        x, t = _real(p, "x", 0.0), _real(p, "t", 0.0)
        arg = k * x - w * t
        out["y"] = 2 * a * math.cos(phi / 2) * math.sin(arg + phi / 2)
        out["y_direct"] = a * math.sin(arg) + a * math.sin(arg + phi)
    return out


def _soliton(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    v, alpha = need(p, "v", None), _real(p, "alpha")
    if alpha == 0:
        raise ToolError("INVALID_ARGUMENT", "params.alpha must not be 0.")
    peak = 3 * v / alpha
    half = math.sqrt(v) / 2
    out: dict[str, Any] = {
        "peak": peak,
        "fwhm": 2 * math.acosh(math.sqrt(2)) / half,
        "area": 2 * peak / half,
    }
    ctx.formula = (
        "A(x, t) = (3 v / alpha) sech^2(sqrt(v) / 2 (x - v t)), solution of dy/dt + alpha y dy/dx + d3y/dx3 = 0; "
        "full width at half maximum 4 acosh(sqrt 2) / sqrt v; area 12 sqrt(v) / alpha"
    )
    if p.get("x") is not None or p.get("t") is not None:
        z = half * (_real(p, "x", 0.0) - v * _real(p, "t", 0.0))
        out["y"] = peak / math.cosh(z) ** 2 if abs(z) < 350 else 0.0
    ctx.formatted = f"KdV soliton: peak {fmt_sig(peak, 6)}, width {fmt_sig(out['fwhm'], 6)}."
    return out
