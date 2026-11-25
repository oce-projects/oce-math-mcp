"""eng_calc, modes added by D45 to existing operations (JSON schemas unchanged).

kinematics_dynamics: collision, orbit, gyroscope, flywheel. optics_acoustics: doppler, string,
pipe, intensity. circuit_ac_dc: impedance, transient. Each function sets formula, units and
text; the operation modules call it from their list of modes.

Constants: G = 6.67430e-11 m3/(kg s2) (CODATA 2018); g = 9.80665 m/s2; acoustic references
1 pW/m2, 1 pW and 20 micropascal (ISO 1683).
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .eng_base import choice, integer, need, one_of, qty, units_for

G_NEWTON = 6.67430e-11
G0 = 9.80665
Fn = Callable[[dict[str, Any], Ctx], dict[str, Any]]


def _text(out: dict[str, Any]) -> str:
    return (
        "; ".join(f"{k} = {fmt_float(v, 5)}" for k, v in out.items() if isinstance(v, float)) + "."
    )


# ------------------------------------------------------------------ kinematics_dynamics


def _collision(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    m1 = need(p, "m1", "mass")
    v1 = need(p, "v1", "velocity", positive=False)
    v2 = qty(p, "v2", "velocity", positive=False, default=0.0) or 0.0
    e = qty(p, "e", None, positive=False, default=1.0)
    assert e is not None
    if not 0 <= e <= 1:
        raise ToolError("INVALID_ARGUMENT", "params.e (restitution) must be between 0 and 1.")
    m2 = qty(p, "m2", "mass", need=False)
    out: dict[str, Any]
    if m2 is None:
        a = v2 - e * (v1 - v2)
        out = {"v1_after": a, "v2_after": v2, "J": m1 * (a - v1)}
        ctx.formula = "wall: v1' = v2 - e (v1 - v2); J = m1 (v1' - v1)"
    else:
        tot = m1 + m2
        a = (m1 * v1 + m2 * v2 + m2 * e * (v2 - v1)) / tot
        b = (m1 * v1 + m2 * v2 + m1 * e * (v1 - v2)) / tot
        k0 = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2
        k1 = 0.5 * m1 * a * a + 0.5 * m2 * b * b
        out = {
            "v1_after": a,
            "v2_after": b,
            "J": m1 * (a - v1),
            "KE_before": k0,
            "KE_after": k1,
            "KE_lost": k0 - k1,
        }
        ctx.formula = (
            "momentum conserved, e = (v2' - v1') / (v1 - v2): "
            "v1' = (m1 v1 + m2 v2 + m2 e (v2 - v1)) / (m1 + m2)"
        )
    ctx.units = units_for(
        out,
        {
            "v1_after": "m/s",
            "v2_after": "m/s",
            "J": "N*s",
            "KE_before": "J",
            "KE_after": "J",
            "KE_lost": "J",
        },
    )
    ctx.formatted = _text(out)
    return out


def _orbit(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if one_of(p, ("mu", "M"), "the central body (mu = G M or M)") == "mu":
        mu = need(p, "mu", None)
    else:
        mu = G_NEWTON * need(p, "M", "mass")
    out: dict[str, Any]
    if p.get("r") is not None:
        if p.get("r_p") is not None or p.get("r_a") is not None:
            raise ToolError(
                "INVALID_ARGUMENT", "Give r (circular orbit) or r_p with r_a (ellipse)."
            )
        r = need(p, "r", "length")
        vc = math.sqrt(mu / r)
        out = {
            "v_circular": vc,
            "v_escape": math.sqrt(2 * mu / r),
            "period": 2 * math.pi * math.sqrt(r**3 / mu),
            "energy": -mu / (2 * r),
        }
        ctx.formula = (
            "v = sqrt(mu / r); v_escape = sqrt(2 mu / r); T = 2 pi sqrt(r^3 / mu); E = -mu / (2 r)"
        )
        r2 = qty(p, "r_target", "length", need=False)
        if r2 is not None:
            at = (r + r2) / 2
            v_dep = math.sqrt(mu * (2 / r - 1 / at))
            v_arr = math.sqrt(mu * (2 / r2 - 1 / at))
            dv1, dv2 = abs(v_dep - vc), abs(math.sqrt(mu / r2) - v_arr)
            out |= {
                "dv1": dv1,
                "dv2": dv2,
                "dv_total": dv1 + dv2,
                "t_transfer": math.pi * math.sqrt(at**3 / mu),
            }
            ctx.formula += (
                "; Hohmann: ellipse with a = (r1 + r2) / 2, vis-viva v^2 = mu (2/r - 1/a)"
            )
    else:
        rp, ra = need(p, "r_p", "length"), need(p, "r_a", "length")
        if rp > ra:
            raise ToolError("INVALID_ARGUMENT", "r_p (periapsis) cannot exceed r_a (apoapsis).")
        a = (rp + ra) / 2
        vp, va = math.sqrt(mu * (2 / rp - 1 / a)), math.sqrt(mu * (2 / ra - 1 / a))
        out = {
            "a": a,
            "eccentricity": (ra - rp) / (ra + rp),
            "v_p": vp,
            "v_a": va,
            "h": rp * vp,
            "period": 2 * math.pi * math.sqrt(a**3 / mu),
            "energy": -mu / (2 * a),
        }
        ctx.formula = "a = (r_p + r_a) / 2; vis-viva v^2 = mu (2/r - 1/a); T = 2 pi sqrt(a^3 / mu); h = r_p v_p"
    ctx.units = units_for(
        out,
        {
            "v_circular": "m/s", "v_escape": "m/s", "period": "s", "energy": "J/kg", "dv1": "m/s",
            "dv2": "m/s", "dv_total": "m/s", "t_transfer": "s", "a": "m", "eccentricity": "1",
            "v_p": "m/s", "v_a": "m/s", "h": "m2/s",
        },
    )  # fmt: skip
    ctx.formatted = _text(out)
    return out


def _gyroscope(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    Iv, w = need(p, "I_m", "mass_moment"), need(p, "omega", "angular_velocity")
    L = Iv * w
    given = [k for k in ("T", "Omega_p", "m") if p.get(k) is not None]
    if len(given) != 1:
        raise ToolError(
            "INVALID_ARGUMENT", "gyroscope: give only one of T, Omega_p and the torque m with r."
        )
    if given[0] == "T":
        T = need(p, "T", "torque")
        Om = T / L
    elif given[0] == "Omega_p":
        Om = need(p, "Omega_p", "angular_velocity")
        T = L * Om
    else:
        g = qty(p, "g", "acceleration", default=G0)
        assert g is not None
        T = need(p, "m", "mass") * g * need(p, "r", "length")
        Om = T / L
    out = {"L": L, "T": T, "Omega_p": Om, "precession_period": 2 * math.pi / Om}
    ctx.formula = (
        "L = I omega; T = L Omega_p (slow precession); T = m g r for the weight with arm r"
    )
    ctx.units = {"L": "kg*m2/s", "T": "Nm", "Omega_p": "rad/s", "precession_period": "s"}
    ctx.formatted = _text(out)
    return out


def _flywheel(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any]
    if p.get("dE") is not None:
        extra = sorted(k for k in ("I_m", "shape", "m", "R", "omega_max", "omega_min") if k in p)
        if extra:
            raise ToolError(
                "INVALID_ARGUMENT", f"Flywheel design (dE): extra keys {', '.join(extra)}."
            )
        dE, wm, Cs = (
            need(p, "dE", "energy"),
            need(p, "omega_mean", "angular_velocity"),
            need(p, "Cs", None),
        )
        out = {"I_required": dE / (Cs * wm * wm), "dE": dE, "omega_mean": wm, "Cs": Cs}
        ctx.formula = "I = dE / (Cs omega_media^2)"
    else:
        w1, w2 = (
            need(p, "omega_max", "angular_velocity"),
            need(p, "omega_min", "angular_velocity"),
        )
        if w2 >= w1:
            raise ToolError("INVALID_ARGUMENT", "omega_min must be less than omega_max.")
        if p.get("I_m") is not None:
            if any(k in p for k in ("shape", "m", "R")):
                raise ToolError("INVALID_ARGUMENT", "Give I_m or shape with m and R.")
            Iv = need(p, "I_m", "mass_moment")
        else:
            shape = choice(p, "shape", ("disk", "ring"))
            mR2 = need(p, "m", "mass") * need(p, "R", "length") ** 2
            Iv = mR2 / 2 if shape == "disk" else mR2
        wm = (w1 + w2) / 2
        e1, e2 = 0.5 * Iv * w1 * w1, 0.5 * Iv * w2 * w2
        out = {
            "I_m": Iv,
            "E_max": e1,
            "E_min": e2,
            "dE": e1 - e2,
            "omega_mean": wm,
            "Cs": (w1 - w2) / wm,
        }
        ctx.formula = (
            "E = I omega^2 / 2; Cs = (omega_max - omega_min) / omega_mean; disc I = m R^2 / 2"
        )
    ctx.units = units_for(
        out,
        {
            "I_required": "kg*m2",
            "I_m": "kg*m2",
            "E_max": "J",
            "E_min": "J",
            "dE": "J",
            "omega_mean": "rad/s",
            "Cs": "1",
        },
    )
    ctx.formatted = _text(out)
    return out


KD_MODES: dict[str, set[str]] = {
    "collision": {"m1", "v1", "m2", "v2", "e"},
    "orbit": {"mu", "M", "r", "r_target", "r_p", "r_a"},
    "gyroscope": {"I_m", "omega", "T", "Omega_p", "m", "r", "g"},
    "flywheel": {"I_m", "shape", "m", "R", "omega_max", "omega_min", "dE", "omega_mean", "Cs"},
}
KD_FUNCS: dict[str, Fn] = {
    "collision": _collision,
    "orbit": _orbit,
    "gyroscope": _gyroscope,
    "flywheel": _flywheel,
}


# ------------------------------------------------------------------ optics_acoustics


def _doppler(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    f, c = need(p, "f", "frequency"), need(p, "c", "velocity")
    vs = qty(p, "v_source", "velocity", positive=False, default=0.0) or 0.0
    vo = qty(p, "v_observer", "velocity", positive=False, default=0.0) or 0.0
    if abs(vs) >= c or vo <= -c:
        raise ToolError("INVALID_ARGUMENT", "The speeds must stay below the wave speed c.")
    ratio = (c + vo) / (c - vs)
    out = {"f_observed": f * ratio, "ratio": ratio}
    ctx.formula = "f' = f (c + v_observer) / (c - v_source), speeds positive towards each other"
    ctx.units = {"f_observed": "Hz", "ratio": "1"}
    ctx.formatted = f"Observed frequency {fmt_float(f * ratio, 5)} Hz."
    return out


def _string(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    L, T, mu = need(p, "L", "length"), need(p, "T", "force"), need(p, "mu", "linear_mass")
    n = integer(p, "n") if p.get("n") is not None else 1
    v = math.sqrt(T / mu)
    out = {"v": v, "f_n": n * v / (2 * L), "wavelength": 2 * L / n}
    ctx.formula = "v = sqrt(T / mu); f_n = n v / (2 L); lambda_n = 2 L / n"
    ctx.units = {"v": "m/s", "f_n": "Hz", "wavelength": "m"}
    ctx.formatted = f"Harmonic {n}: {fmt_float(out['f_n'], 5)} Hz."
    return out


def _pipe(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    L, c = need(p, "L", "length"), need(p, "c", "velocity")
    ends = choice(p, "ends", ("open_open", "open_closed"))
    n = integer(p, "n") if p.get("n") is not None else 1
    if ends == "open_open":
        f = n * c / (2 * L)
        ctx.formula = "open-open pipe: f_n = n c / (2 L)"
    else:
        f = (2 * n - 1) * c / (4 * L)
        ctx.formula = "open-closed pipe: f_n = (2n - 1) c / (4 L), odd harmonics only"
    out = {"f_n": f, "wavelength": c / f}
    ctx.units = {"f_n": "Hz", "wavelength": "m"}
    ctx.formatted = f"Resonance {n}: {fmt_float(f, 5)} Hz."
    return out


def _intensity(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    W, r = need(p, "W", "power"), need(p, "r", "length")
    Q = qty(p, "Q_dir", None, default=1.0)
    assert Q is not None
    Iv = Q * W / (4 * math.pi * r * r)
    out: dict[str, Any] = {
        "I": Iv,
        "L_I_dB": 10 * math.log10(Iv / 1e-12),
        "L_W_dB": 10 * math.log10(W / 1e-12),
    }
    ctx.formula = "I = Q W / (4 pi r^2); L_I = 10 log10(I / 1 pW/m2); L_W = 10 log10(W / 1 pW)"
    if (p.get("rho") is None) != (p.get("c") is None):
        raise ToolError("INVALID_ARGUMENT", "rho and c must be given together.")
    if p.get("rho") is not None:
        Z = need(p, "rho", "density") * need(p, "c", "velocity")
        pr = math.sqrt(Iv * Z)
        out |= {"Z": Z, "p_rms": pr, "SPL_dB": 20 * math.log10(pr / 2e-5)}
        ctx.formula += "; Z = rho c; p_rms = sqrt(I rho c); SPL = 20 log10(p / 20 microPa)"
    ctx.units = units_for(
        out,
        {"I": "W/m2", "L_I_dB": "dB", "L_W_dB": "dB", "Z": "Pa*s/m", "p_rms": "Pa", "SPL_dB": "dB"},
    )
    ctx.formatted = f"Intensity {fmt_float(Iv, 4)} W/m2, level {fmt_float(out['L_I_dB'], 4)} dB."
    return out


OA_MODES: dict[str, set[str]] = {
    "doppler": {"f", "c", "v_source", "v_observer"},
    "string": {"L", "T", "mu", "n"},
    "pipe": {"L", "c", "ends", "n"},
    "intensity": {"W", "r", "Q_dir", "rho", "c"},
}
OA_FUNCS: dict[str, Fn] = {
    "doppler": _doppler,
    "string": _string,
    "pipe": _pipe,
    "intensity": _intensity,
}


# ------------------------------------------------------------------ circuit_ac_dc

_ELEM_KIND = {"R": "electrical_resistance", "L": "inductance", "C": "capacitance"}


def _impedance(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    f = need(p, "f", "frequency")
    conn = choice(p, "connection", ("series", "parallel"))
    elems = p.get("elements")
    if not isinstance(elems, list) or not 1 <= len(elems) <= 50:
        raise ToolError(
            "INVALID_ARGUMENT",
            'params.elements: list of 1 to 50 objects {"type": "R", "value": 10}.',
        )
    w = 2 * math.pi * f
    zs: list[complex] = []
    X: list[float] = []
    for i, el in enumerate(elems):
        if not isinstance(el, dict) or set(el) != {"type", "value"} or el["type"] not in _ELEM_KIND:
            raise ToolError(
                "INVALID_ARGUMENT", f'params.elements[{i}]: {{"type": "R"/"L"/"C", "value": ...}}.'
            )
        v = need(el, "value", _ELEM_KIND[el["type"]])
        z = (
            complex(v, 0)
            if el["type"] == "R"
            else complex(0, w * v)
            if el["type"] == "L"
            else complex(0, -1 / (w * v))
        )
        zs.append(z)
        X.append(z.imag)
    Z = sum(zs, 0j) if conn == "series" else 1 / sum((1 / z for z in zs), 0j)
    out: dict[str, Any] = {
        "Z_re": Z.real,
        "Z_im": Z.imag,
        "Z_abs": abs(Z),
        "phase_deg": math.degrees(cmath.phase(Z)),
        "X": X,
    }
    ctx.formula = (
        "Z_R = R; Z_L = j 2 pi f L; Z_C = 1 / (j 2 pi f C); series: sum; parallel: 1/Z = sum(1/Z_k)"
    )
    V = qty(p, "V", "voltage", need=False)
    if V is not None:
        Iv = V / Z
        S = V * Iv.conjugate()
        out |= {"I": abs(Iv), "P": S.real, "Q": S.imag, "S": abs(S), "cos_phi": S.real / abs(S)}
        ctx.formula += "; I = V / Z; S = V I*"
    ctx.units = units_for(
        out,
        {"Z_re": "ohm", "Z_im": "ohm", "Z_abs": "ohm", "phase_deg": "deg", "X": "ohm", "I": "A", "P": "W",
         "Q": "var", "S": "VA", "cos_phi": "1"},
    )  # fmt: skip
    ctx.formatted = (
        f"|Z| = {fmt_float(abs(Z), 5)} ohm, phase {fmt_float(out['phase_deg'], 4)} degrees."
    )
    return out


def _rlc_series(q: float, p: float) -> tuple[float, float]:
    """Step response of the series RLC from rest by its Taylor series (D77): v_C / V and t i / (C V).

    q = alpha t and p = omega0 t, with q + p <= 0.5. With y = v_C / V and T_n = c_n t^n, the equation
    y'' = omega0^2 (1 - y) - 2 alpha y' gives
    T_(n+2) = (p^2 (delta_n0 - T_n) - 2 q (n + 1) T_(n+1)) / ((n + 2)(n + 1)).
    30 terms; the test checks the result against mpmath at 50 digits.
    """
    t0 = t1 = y = ty = 0.0
    for n in range(30):
        t2 = (p * p * ((1.0 if n == 0 else 0.0) - t0) - 2 * q * (n + 1) * t1) / ((n + 2) * (n + 1))
        y += t2
        ty += (n + 2) * t2
        t0, t1 = t1, t2
    return y, ty


def _transient(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    circ = choice(p, "circuit", ("RC", "RL", "RLC"))
    need_keys = {"RC": ("C",), "RL": ("L",), "RLC": ("L", "C")}[circ]
    wrong = sorted({"L", "C"} - set(need_keys) & set(p))
    if wrong:
        raise ToolError("INVALID_ARGUMENT", f"The {circ} circuit does not use: {', '.join(wrong)}.")
    R, V = need(p, "R", "electrical_resistance"), need(p, "V", "voltage")
    t = qty(p, "t", "time", positive=False, need=False)
    if t is not None and t < 0:
        raise ToolError("INVALID_ARGUMENT", "params.t must be >= 0.")
    kind = choice(p, "kind", ("charge", "discharge"), "charge")
    out: dict[str, Any]
    if circ == "RC":
        C = need(p, "C", "capacitance")
        tau = R * C
        out = {"tau": tau}
        if t is not None:
            # charge: 1 - e^(-t/tau) with expm1, no cancellation at small t (D77)
            out["v_C"] = V * -math.expm1(-t / tau) if kind == "charge" else V * math.exp(-t / tau)
        ctx.formula = "tau = R C; charge v = V (1 - e^(-t/tau)); discharge v = V e^(-t/tau)"
    elif circ == "RL":
        L = need(p, "L", "inductance")
        tau = L / R
        out = {"tau": tau}
        if t is not None:
            i0 = V / R
            out["i"] = i0 * -math.expm1(-t / tau) if kind == "charge" else i0 * math.exp(-t / tau)
        ctx.formula = "tau = L / R; rise i = V/R (1 - e^(-t/tau)); decay i = V/R e^(-t/tau)"
    else:
        if p.get("kind") is not None:
            raise ToolError(
                "INVALID_ARGUMENT", "RLC: only the step response from a discharged circuit."
            )
        L, C = need(p, "L", "inductance"), need(p, "C", "capacitance")
        a, w0 = R / (2 * L), 1 / math.sqrt(L * C)
        zeta = a / w0
        regime = "critical" if abs(zeta - 1) < 1e-9 else "underdamped" if zeta < 1 else "overdamped"
        out = {"alpha": a, "omega0": w0, "zeta": zeta, "regime": regime}
        if regime == "underdamped":
            wd = math.sqrt(w0 * w0 - a * a)
            out["omega_d"] = wd
        ctx.formula = "alpha = R / (2 L); omega0 = 1 / sqrt(L C); zeta = alpha / omega0; step response from rest"
        if t is not None:
            ex = math.exp(-a * t)
            if (a + w0) * t <= 0.5:
                # small times: the closed forms below compute 1 - (a number close to 1), D77
                y, ty = _rlc_series(a * t, w0 * t)
                out["v_C"], out["i"] = V * y, (C * V * ty / t if t else 0.0)
            elif regime == "underdamped":
                wd = out["omega_d"]
                out["v_C"] = V * (1 - ex * (math.cos(wd * t) + a / wd * math.sin(wd * t)))
                out["i"] = V / (L * wd) * ex * math.sin(wd * t)
            elif regime == "critical":
                out["v_C"] = V * (1 - ex * (1 + a * t))
                out["i"] = V / L * t * ex
            else:
                s = math.sqrt(a * a - w0 * w0)
                s2 = -(a + s)
                s1 = w0 * w0 / s2  # = -a + s without cancellation, since s1 s2 = omega0^2 (D77)
                e1, e2 = math.exp(s1 * t), math.exp(s2 * t)
                # e1 - e2 without cancellation near critical damping
                d12 = 2 * ex * math.sinh(s * t) if s * t < 1 else e1 - e2
                if 2 * s > a:  # clearly overdamped: the two modes do not cancel
                    vy = (s2 * math.expm1(s1 * t) - s1 * math.expm1(s2 * t)) / (2 * s)
                else:  # close to critical damping
                    vy = -math.expm1(s2 * t) + s2 * d12 / (2 * s)
                out["v_C"] = V * vy
                out["i"] = V / L * d12 / (2 * s)
    ctx.units = units_for(
        out,
        {
            "tau": "s",
            "v_C": "V",
            "i": "A",
            "alpha": "1/s",
            "omega0": "rad/s",
            "zeta": "1",
            "omega_d": "rad/s",
        },
    )
    ctx.formatted = _text(out)
    return out


CIRC_MODES: dict[str, set[str]] = {
    "impedance": {"f", "elements", "connection", "V"},
    "transient": {"circuit", "R", "C", "L", "V", "kind", "t"},
}
CIRC_FUNCS: dict[str, Fn] = {"impedance": _impedance, "transient": _transient}
