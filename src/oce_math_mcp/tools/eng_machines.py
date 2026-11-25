"""eng_calc: electric machines (D67).

Sources: Wikipedia pages in the local offline library.
- "Transformer": ideal transformer (voltage ratio = turns ratio, inverse current ratio, impedance referred with the
  turns ratio squared, i_p n_p - i_s n_s = 0); EMF equation E_rms = 2 pi f N A B_peak / sqrt(2) ~ 4.44 f N A B_peak.
- "Brushed DC electric motor": V_m = E_b + R_m I_a, back EMF proportional to the speed, torque proportional to the
  current (T = k_T I_a Phi with k_T = k_b / (2 pi)), linear torque-speed line, quadratic power.
- "Induction motor": n_s = 120 f / p with p poles, slip s = (n_s - n_r)/n_s, P_r = 3 R_r' I_r'^2,
  P_gap = P_r / s, P_em = P_gap (1 - s), T_em = 3 I_r'^2 R_r' / (omega_s s), input impedance
  Z_m = R_s + j X_s + ((R_r'/s + j X_r') j X_m) / (R_r'/s + j (X_r' + X_m)), breakdown torque
  T_max = 3 V_TE^2 / (2 omega_s (R_TE + sqrt(R_TE^2 + (X_TE + X_r')^2))) at s = R_r' / sqrt(R_TE^2 + (X_TE + X_r')^2).
- "Synchronous motor": N_s = 120 f / p, T = T_max sin(delta), T_max = 3 V E / (X_s omega_s).
Here, derived and checked in the tests: the DC motor in SI with one constant k (E_b = k omega, T = k I_a; the page
writes the speed n in rpm with T = k_b Phi I_a / (2 pi), and the power balance E_b I_a = T omega holds with n in
revolutions per second), no-load speed V_m/k, stall torque k V_m/R_m, maximum power V_m^2/(4 R_m); rotor frequency
s f; power of the synchronous machine P = T omega_s = 3 V E sin(delta) / X_s.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, one_of, only, op, qty, units_for

_IM_CIRCUIT = {"R_s", "X_s", "R_r", "X_r", "X_m"}
_MODES: dict[str, set[str]] = {
    "transformer": {"calc", "N_p", "N_s", "V_p", "I_p", "Z_s", "f", "N", "A", "B_peak", "E_rms"},
    "dc_motor": {"V_m", "R_m", "k", "T_load"},
    "induction_motor": {"f", "poles", "s", "n_rpm", "V"} | _IM_CIRCUIT,
    "synchronous_machine": {"V", "E", "X_s", "delta_deg", "f", "poles"},
}
_TR_CALC = {"ratio": {"N_p", "N_s", "V_p", "I_p", "Z_s"}, "emf": {"f", "N", "A", "B_peak", "E_rms"}}
_UNITS = {
    "V_s": "V",
    "I_s": "A",
    "Z_p": "ohm",
    "E_rms": "V",
    "B_peak": "T",
    "omega_0": "rad/s",
    "n_0_rpm": "rpm",
    "T_stall": "N*m",
    "I_stall": "A",
    "P_max": "W",
    "T_at_P_max": "N*m",
    "omega": "rad/s",
    "n_rpm": "rpm",
    "I_a": "A",
    "E_b": "V",
    "T": "N*m",
    "P_out": "W",
    "P_in": "W",
    "n_s_rpm": "rpm",
    "omega_s": "rad/s",
    "f_rotor": "Hz",
    "I_stator": "A",
    "P_ag": "W",
    "P_cu_rotor": "W",
    "P_mech": "W",
    "T_max": "N*m",
    "P": "W",
}


@op("electric_machine", ("mode", *sorted(set().union(*_MODES.values()))))
def _machine(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = {
        "transformer": _transformer,
        "dc_motor": _dc_motor,
        "induction_motor": _induction,
        "synchronous_machine": _synchronous,
    }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _transformer(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    calc = choice(p, "calc", tuple(_TR_CALC), "ratio")
    only(p, calc, _TR_CALC[calc] | {"mode"}, field="calc")
    if calc == "emf":
        f, n, area = need(p, "f", "frequency"), need(p, "N", None), need(p, "A", "area")
        k = 2 * math.pi * f * n * area / math.sqrt(2)
        ctx.formula = (
            "E_rms = 2 pi f N A B_peak / sqrt(2) (about 4.44 f N A B_peak, sinusoidal flux)"
        )
        if one_of(p, ("B_peak", "E_rms"), "the EMF equation") == "B_peak":
            b = need(p, "B_peak", "magnetic_flux_density")
            e = k * b
        else:
            e = need(p, "E_rms", "voltage")
            b = e / k
        ctx.formatted = f"EMF {fmt_sig(e, 6)} V rms with B_peak = {fmt_sig(b, 6)} T."
        return {"E_rms": e, "B_peak": b}
    n_p, n_s = need(p, "N_p", None), need(p, "N_s", None)
    a = n_p / n_s
    out: dict[str, Any] = {"a": a}
    ctx.formula = "ideal: V_s = V_p N_s / N_p; I_p N_p = I_s N_s; Z_p = (N_p / N_s)^2 Z_s"
    v_p = qty(p, "V_p", "voltage", need=False)
    if v_p is not None:
        out["V_s"] = v_p / a
    i_p = qty(p, "I_p", "current", need=False)
    if i_p is not None:
        out["I_s"] = i_p * a
    z_s = qty(p, "Z_s", "electrical_resistance", need=False)
    if z_s is not None:
        out["Z_p"] = a * a * z_s
    ctx.formatted = f"Ideal transformer, turns ratio {fmt_sig(a, 6)}."
    return out


def _dc_motor(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    v, r = need(p, "V_m", "voltage"), need(p, "R_m", "electrical_resistance")
    k = need(p, "k", None)
    w0 = v / k
    t_stall = k * v / r
    out: dict[str, Any] = {
        "omega_0": w0,
        "n_0_rpm": w0 * 60 / (2 * math.pi),
        "T_stall": t_stall,
        "I_stall": v / r,
    }
    out["P_max"] = v * v / (4 * r)
    out["T_at_P_max"] = t_stall / 2
    ctx.formula = "V_m = E_b + R_m I_a; E_b = k omega; T = k I_a; omega = (V_m - R_m T / k) / k"
    t_load = qty(p, "T_load", "torque", need=False, positive=False)
    if t_load is None:
        ctx.formatted = f"DC motor: no-load speed {fmt_sig(out['n_0_rpm'], 6)} rpm, stall torque {fmt_sig(t_stall, 6)} N*m."
        return out
    if t_load < 0 or t_load > t_stall:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"T_load must be between 0 and the stall torque {fmt_sig(t_stall, 6)} N*m.",
        )
    i_a = t_load / k
    w = (v - r * i_a) / k
    out.update(I_a=i_a, E_b=k * w, omega=w, n_rpm=w * 60 / (2 * math.pi), T=t_load)
    out["P_out"] = t_load * w
    out["P_in"] = v * i_a
    out["efficiency"] = (t_load * w / (v * i_a)) if i_a > 0 else None
    ctx.formatted = f"DC motor at {fmt_sig(t_load, 6)} N*m: {fmt_sig(out['n_rpm'], 6)} rpm, I_a = {fmt_sig(i_a, 6)} A."
    return out


def _sync_speed(p: dict[str, Any]) -> tuple[float, float, int]:
    f = need(p, "f", "frequency")
    poles = integer(p, "poles", 2)
    if poles % 2:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.poles is the number of poles: an even number (2, 4, 6, ...).",
        )
    n_s = 120 * f / poles
    return n_s, 4 * math.pi * f / poles, poles


def _induction(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    n_s, w_s, _ = _sync_speed(p)
    if one_of(p, ("s", "n_rpm"), "the operating point (slip or speed)") == "s":
        s = need(p, "s", None, positive=False)
    else:
        s = (n_s - need(p, "n_rpm", None, positive=False)) / n_s
    if s == 0:
        raise ToolError(
            "INVALID_ARGUMENT", "Slip 0 (synchronous speed): no torque, no rotor current."
        )
    out: dict[str, Any] = {
        "n_s_rpm": n_s,
        "omega_s": w_s,
        "s": s,
        "n_rpm": n_s * (1 - s),
        "f_rotor": s * need(p, "f", "frequency"),
    }
    ctx.formula = "n_s = 120 f / p (p poles); s = (n_s - n_r) / n_s; f_rotor = s f"
    given = [k for k in sorted(_IM_CIRCUIT | {"V"}) if p.get(k) is not None]
    if not given:
        ctx.formatted = f"Synchronous speed {fmt_sig(n_s, 6)} rpm, slip {fmt_sig(s, 6)}."
        return out
    v_ph = need(p, "V", "voltage") / math.sqrt(3)
    r_s, x_s = (
        need(p, "R_s", "electrical_resistance", positive=False),
        need(p, "X_s", "electrical_resistance", positive=False),
    )
    r_r, x_r = (
        need(p, "R_r", "electrical_resistance"),
        need(p, "X_r", "electrical_resistance", positive=False),
    )
    x_m = need(p, "X_m", "electrical_resistance")
    z_rot = complex(r_r / s, x_r)
    z_par = z_rot * 1j * x_m / complex(r_r / s, x_r + x_m)
    i_stator = v_ph / complex(r_s + z_par.real, x_s + z_par.imag)
    i_rotor = i_stator * z_par / z_rot
    p_ag = 3 * abs(i_rotor) ** 2 * r_r / s
    out.update(I_stator=abs(i_stator), power_factor=math.cos(cmath.phase(i_stator)))
    out.update(
        P_in=3 * (v_ph * i_stator.conjugate()).real,
        P_ag=p_ag,
        P_cu_rotor=s * p_ag,
        P_mech=(1 - s) * p_ag,
        T=p_ag / w_s,
    )
    z_th = 1j * x_m * complex(r_s, x_s) / complex(r_s, x_s + x_m)
    v_th = abs(v_ph * 1j * x_m / complex(r_s, x_s + x_m))
    root = math.hypot(z_th.real, z_th.imag + x_r)
    out["s_T_max"] = r_r / root
    out["T_max"] = 3 * v_th**2 / (2 * w_s * (z_th.real + root))
    ctx.formula += (
        "; Z_m = R_s + j X_s + ((R_r/s + j X_r) j X_m) / (R_r/s + j (X_r + X_m)); P_ag = 3 I_r^2 R_r / s; "
        "P_mech = (1 - s) P_ag; T = P_ag / omega_s; T_max = 3 V_TE^2 / (2 omega_s (R_TE + sqrt(R_TE^2 + (X_TE + X_r)^2)))"
    )
    ctx.formatted = (
        f"Induction motor at {fmt_sig(out['n_rpm'], 6)} rpm: torque {fmt_sig(out['T'], 6)} N*m, "
        f"stator current {fmt_sig(abs(i_stator), 6)} A, breakdown torque {fmt_sig(out['T_max'], 6)} N*m."
    )
    ctx.warn(
        "V is the line-to-line voltage of a star-connected three-phase motor; circuit values per phase."
    )
    return out


def _synchronous(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = {}
    w_s = None
    if p.get("f") is not None or p.get("poles") is not None:
        n_s, w_s, _ = _sync_speed(p)
        out.update(n_s_rpm=n_s, omega_s=w_s)
        ctx.formula = "N_s = 120 f / p (p poles)"
    if all(p.get(k) is None for k in ("V", "E", "X_s", "delta_deg")):
        if w_s is None:
            raise ToolError(
                "INVALID_ARGUMENT", "Synchronous machine: give f and poles, or V, E and X_s."
            )
        ctx.formatted = f"Synchronous speed {fmt_sig(out['n_s_rpm'], 6)} rpm."
        return out
    v, e = need(p, "V", "voltage"), need(p, "E", "voltage")
    x_s = need(p, "X_s", "electrical_resistance")
    p_max = v * e / x_s  # 3 V_ph E_ph / X_s with line-to-line V and E
    out["P_max"] = p_max
    formula = "P = 3 V_ph E_ph sin(delta) / X_s = V E sin(delta) / X_s (line-to-line V, E)"
    ctx.formula = f"{ctx.formula}; {formula}" if ctx.formula else formula
    delta = qty(p, "delta_deg", None, need=False, positive=False)
    if delta is not None:
        out["P"] = p_max * math.sin(math.radians(delta))
    if w_s is not None:
        out["T_max"] = p_max / w_s
        if delta is not None:
            out["T"] = out["P"] / w_s
        ctx.formula += "; T = T_max sin(delta), T_max = 3 V E / (X_s omega_s)"
    ctx.formatted = f"Synchronous machine: maximum power {fmt_sig(p_max, 6)} W" + (
        f", at {fmt_sig(delta, 6)} degrees {fmt_sig(out['P'], 6)} W." if delta is not None else "."
    )
    return out
