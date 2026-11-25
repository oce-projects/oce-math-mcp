"""eng_calc: practical electronics (D65).

Sources: Wikipedia pages in the local offline library.
- "LED circuit": R = (V_power - V_f - V_switch) / I_led; LEDs in series need the sum of their voltages.
- "Operational amplifier applications": inverting V_out = -(R_f/R_in) V_in with input impedance R_in;
  summing V_out = -R_f sum(V_n/R_n); difference V_out = ((R_f + R_1) R_g)/((R_g + R_2) R_1) V_2 - (R_f/R_1) V_1,
  zero common-mode gain with R_1/R_f = R_2/R_g; integrator V_out = -(1/(R C)) integral V_in dt;
  differentiator V_out = -R C dV_in/dt. "Non-inverting amplifier": A_CL = 1 + R_f/R_g.
- "Negative-feedback amplifier": A_FB = A_OL / (1 + beta A_OL). "Operational amplifier": cutoff = GBWP divided
  by the closed-loop gain (one dominant pole).
- "Wheatstone bridge": R_x = (R_2/R_1) R_3; V_G = (R_2/(R_1 + R_2) - R_x/(R_x + R_3)) V_s and its inverse for R_x.
- "Thermistor": Steinhart-Hart 1/T = a + b ln R + c (ln R)^3 and its inverse; B equation
  R = R_0 exp(B (1/T - 1/T_0)), T = T_0 B / (T_0 ln(R/R_0) + B).
- "555 timer IC": t_high = ln(2) (R_1 + R_2) C, t_low = ln(2) R_2 C, f = 1/(ln(2)(R_1 + 2 R_2) C), monostable
  t = ln(3) R C.
Derived here and checked in the tests: inverting amplifier with a finite open-loop gain
G = -(R_f/R_in) A beta / (1 + A beta) (Kirchhoff law at the inverting node), bandwidth with the noise gain
1 + R_f/R_in for the inverting amplifier (single-pole model), integrator and differentiator gain and phase at a
frequency (complex impedances), galvanometer current I_G with Thevenin (nodal analysis), B from two points and
Steinhart-Hart coefficients from three points (3 x 3 linear system).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, one_of, only, op, qty, temperature, units_for

_OPAMP: dict[str, set[str]] = {
    "inverting": {"R_in", "R_f", "V_in", "A", "GBW"},
    "non_inverting": {"R_g", "R_f", "V_in", "A", "GBW"},
    "summing": {"R_f", "inputs"},
    "difference": {"R_1", "R_f", "R_2", "R_g", "V_1", "V_2"},
    "integrator": {"R", "C", "f"},
    "differentiator": {"R", "C", "f"},
}
_THERM: dict[str, set[str]] = {
    "beta": {"R0", "T0", "B", "T", "R"},
    "steinhart_hart": {"a", "b", "c", "T", "R"},
    "fit": {"points"},
}
_TIMER: dict[str, set[str]] = {"astable": {"R1", "R2", "C"}, "monostable": {"R", "C"}}
_MODES: dict[str, set[str]] = {
    "led_resistor": {"V_s", "V_f", "I", "n", "V_switch"},
    "wheatstone": {"R_1", "R_2", "R_3", "R_x", "V_s", "V_G", "R_G"},
    "opamp": {"config"}.union(*_OPAMP.values()),
    "timer_555": {"circuit"}.union(*_TIMER.values()),
    "thermistor": {"model"}.union(*_THERM.values()),
}
_UNITS = {
    "R": "ohm",
    "V_R": "V",
    "P_R": "W",
    "P_led": "W",
    "V_out": "V",
    "V_G": "V",
    "I_G": "A",
    "R_x": "ohm",
    "R_x_balance": "ohm",
    "R_th": "ohm",
    "V_out_finite_A": "V",
    "R_input": "ohm",
    "bandwidth": "Hz",
    "tau": "s",
    "f_unity": "Hz",
    "phase_deg": "deg",
    "t_high": "s",
    "t_low": "s",
    "period": "s",
    "f": "Hz",
    "t_pulse": "s",
    "T": "K",
    "T_C": "degC",
    "B": "K",
    "R0": "ohm",
    "T0": "K",
}


@op("electronics", ("mode", *sorted(set().union(*_MODES.values()))))
def _electronics(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    if mode in ("led_resistor", "wheatstone"):
        only(p, mode, _MODES[mode])
        out = _led(p, ctx) if mode == "led_resistor" else _wheatstone(p, ctx)
    elif mode == "opamp":
        config = choice(p, "config", tuple(_OPAMP))
        only(p, config, _OPAMP[config] | {"mode"}, field="config")
        out = _opamp(config, p, ctx)
    elif mode == "timer_555":
        circuit = choice(p, "circuit", tuple(_TIMER))
        only(p, circuit, _TIMER[circuit] | {"mode"}, field="circuit")
        out = _timer(circuit, p, ctx)
    else:
        model = choice(p, "model", tuple(_THERM))
        only(p, model, _THERM[model] | {"mode"}, field="model")
        out = _thermistor(model, p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _led(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    v_s, v_f = need(p, "V_s", "voltage"), need(p, "V_f", "voltage")
    i = need(p, "I", "current")
    n = integer(p, "n", 1) if p.get("n") is not None else 1
    v_sw = qty(p, "V_switch", "voltage", positive=False, default=0.0)
    assert v_sw is not None
    v_r = v_s - n * v_f - v_sw
    if v_r <= 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"n V_f + V_switch = {fmt_sig(n * v_f + v_sw, 6)} V is not below V_s = {fmt_sig(v_s, 6)} V: no voltage is left for "
            "the resistor.",
        )
    res = v_r / i
    ctx.formula = "R = (V_s - n V_f - V_switch) / I (n LEDs in series); P_R = I^2 R"
    ctx.formatted = (
        f"Series resistor {fmt_sig(res, 6)} ohm, it dissipates {fmt_sig(i * i * res, 5)} W."
    )
    return {
        "R": res,
        "V_R": v_r,
        "P_R": i * i * res,
        "P_led": n * v_f * i,
        "efficiency": n * v_f / v_s,
    }


def _opamp(config: str, p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if config in ("integrator", "differentiator"):
        return _rc_opamp(config, p, ctx)
    r_f = need(p, "R_f", "electrical_resistance")
    if config == "summing":
        rows = p.get("inputs")
        if not isinstance(rows, list) or not rows:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.inputs: a list of [V, R] pairs in volt and ohm.",
                '{"inputs": [[0.5, 1e4]]}',
            )
        total = 0.0
        for row in rows:
            ok_row = isinstance(row, list | tuple) and len(row) == 2
            if not ok_row or not all(
                isinstance(x, int | float) and not isinstance(x, bool) for x in row
            ):
                raise ToolError(
                    "INVALID_ARGUMENT", "params.inputs: each input is [V, R] in volt and ohm."
                )
            if float(row[1]) <= 0:
                raise ToolError("INVALID_ARGUMENT", "params.inputs: every R must be > 0.")
            total += float(row[0]) / float(row[1])
        ctx.formula = "V_out = -R_f sum(V_k / R_k) (ideal op amp)"
        v_out = -r_f * total
        ctx.formatted = f"Summing amplifier: V_out = {fmt_sig(v_out, 6)} V."
        return {"V_out": v_out}
    if config == "difference":
        return _difference(r_f, p, ctx)
    if config == "inverting":
        r_in = need(p, "R_in", "electrical_resistance")
        gain = -r_f / r_in
        beta = r_in / (r_in + r_f)
        ctx.formula = "G = -R_f / R_in; input resistance R_in"
        out: dict[str, Any] = {"gain": gain, "R_input": r_in}
    else:
        r_g = need(p, "R_g", "electrical_resistance")
        gain = 1 + r_f / r_g
        beta = r_g / (r_g + r_f)
        ctx.formula = "G = 1 + R_f / R_g"
        out = {"gain": gain, "R_input": None}
    out["noise_gain"] = 1 / beta
    v_in = qty(p, "V_in", "voltage", need=False, positive=False)
    if v_in is not None:
        out["V_out"] = gain * v_in
    big_a = qty(p, "A", None, need=False)
    if big_a is not None:
        loop = big_a * beta / (1 + big_a * beta)
        out["gain_finite_A"] = gain * loop if config == "inverting" else big_a / (1 + big_a * beta)
        ctx.formula += (
            "; finite A: G = A / (1 + A beta) (non-inverting), -(R_f/R_in) A beta / (1 + A beta)"
        )
        if v_in is not None:
            out["V_out_finite_A"] = out["gain_finite_A"] * v_in
    gbw = qty(p, "GBW", "frequency", need=False)
    if gbw is not None:
        out["bandwidth"] = gbw * beta
        ctx.formula += "; bandwidth = GBW / noise gain, noise gain 1/beta = 1 + R_f/R"
    ctx.formatted = f"{config.replace('_', '-').capitalize()} amplifier: gain {fmt_sig(gain, 6)}."
    return out


def _difference(r_f: float, p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    r_1, r_2 = need(p, "R_1", "electrical_resistance"), need(p, "R_2", "electrical_resistance")
    r_g = need(p, "R_g", "electrical_resistance")
    v_1 = need(p, "V_1", "voltage", positive=False)
    v_2 = need(p, "V_2", "voltage", positive=False)
    k2 = (r_f + r_1) * r_g / ((r_g + r_2) * r_1)
    v_out = k2 * v_2 - r_f / r_1 * v_1
    balanced = abs(r_1 / r_f - r_2 / r_g) <= 1e-12 * (r_1 / r_f)
    ctx.formula = "V_out = ((R_f + R_1) R_g) / ((R_g + R_2) R_1) V_2 - (R_f / R_1) V_1"
    out: dict[str, Any] = {
        "V_out": v_out,
        "gain_V2": k2,
        "gain_V1": -r_f / r_1,
        "balanced": balanced,
    }
    out["common_mode_gain"] = k2 - r_f / r_1
    if balanced:
        ctx.formula += "; with R_1/R_f = R_2/R_g: V_out = (R_f/R_1)(V_2 - V_1)"
    else:
        ctx.warn("R_1/R_f differs from R_2/R_g: the common-mode gain is not zero.")
    ctx.formatted = f"Difference amplifier: V_out = {fmt_sig(v_out, 6)} V."
    return out


def _wheatstone(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    r_1, r_2 = need(p, "R_1", "electrical_resistance"), need(p, "R_2", "electrical_resistance")
    r_3 = need(p, "R_3", "electrical_resistance")
    out: dict[str, Any] = {"R_x_balance": r_2 * r_3 / r_1}
    ctx.formula = "balance R_2 / R_1 = R_x / R_3, so R_x = (R_2 / R_1) R_3"
    given = [k for k in ("R_x", "V_G") if p.get(k) is not None]
    if len(given) == 2:
        raise ToolError("INVALID_ARGUMENT", "Wheatstone: give R_x or V_G, not both.")
    if not given:
        only(p, "wheatstone", {"R_1", "R_2", "R_3"})
        ctx.formatted = f"Bridge balanced with R_x = {fmt_sig(out['R_x_balance'], 6)} ohm."
        return out
    v_s = need(p, "V_s", "voltage")
    if given[0] == "V_G":
        if p.get("R_G") is not None:
            raise ToolError(
                "INVALID_ARGUMENT", "Wheatstone: R_G is used only with R_x (galvanometer current)."
            )
        v_g = need(p, "V_G", "voltage", positive=False)
        den = r_1 * v_s + (r_1 + r_2) * v_g
        num = r_2 * v_s - (r_1 + r_2) * v_g
        if den <= 0 or num <= 0:
            raise ToolError(
                "INVALID_ARGUMENT", "This V_G gives no positive R_x: check its sign and size."
            )
        out["R_x"] = num / den * r_3
        ctx.formula += "; R_x = (R_2 V_s - (R_1 + R_2) V_G) / (R_1 V_s + (R_1 + R_2) V_G) R_3"
        ctx.formatted = f"Unknown resistor R_x = {fmt_sig(out['R_x'], 6)} ohm."
        return out
    r_x = need(p, "R_x", "electrical_resistance")
    v_g = (r_2 / (r_1 + r_2) - r_x / (r_x + r_3)) * v_s
    out["V_G"] = v_g
    ctx.formula += "; V_G = (R_2/(R_1 + R_2) - R_x/(R_x + R_3)) V_s (node D relative to node B, open galvanometer)"
    r_th = r_1 * r_2 / (r_1 + r_2) + r_3 * r_x / (r_3 + r_x)
    out["R_th"] = r_th
    r_g = qty(p, "R_G", "electrical_resistance", need=False)
    if r_g is not None:
        out["I_G"] = v_g / (r_th + r_g)
        ctx.formula += "; I_G = V_G / (R_1 || R_2 + R_3 || R_x + R_G) (Thevenin)"
    ctx.formatted = f"Bridge output V_G = {fmt_sig(v_g * 1000, 6)} mV."
    return out


def _rc_opamp(config: str, p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    res, cap = need(p, "R", "electrical_resistance"), need(p, "C", "capacitance")
    tau = res * cap
    out: dict[str, Any] = {"tau": tau, "f_unity": 1 / (2 * math.pi * tau)}
    if config == "integrator":
        ctx.formula = "V_out = -(1/(R C)) integral V_in dt; G(f) = -1/(j 2 pi f R C)"
    else:
        ctx.formula = "V_out = -R C dV_in/dt; G(f) = -j 2 pi f R C"
    f = qty(p, "f", "frequency", need=False)
    if f is not None:
        wt = 2 * math.pi * f * tau
        out["gain_magnitude"] = 1 / wt if config == "integrator" else wt
        out["phase_deg"] = 90.0 if config == "integrator" else -90.0
    ctx.formatted = f"{config.capitalize()}: R C = {fmt_sig(tau, 6)} s, unity gain at {fmt_sig(out['f_unity'], 6)} Hz."
    return out


def _timer(circuit: str, p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    cap = need(p, "C", "capacitance")
    if circuit == "monostable":
        res = need(p, "R", "electrical_resistance")
        t = math.log(3) * res * cap
        ctx.formula = "t = ln(3) R C (about 1.1 R C)"
        ctx.formatted = f"555 monostable: pulse {fmt_sig(t, 6)} s."
        return {"t_pulse": t}
    r1, r2 = need(p, "R1", "electrical_resistance"), need(p, "R2", "electrical_resistance")
    t_high = math.log(2) * (r1 + r2) * cap
    t_low = math.log(2) * r2 * cap
    period = t_high + t_low
    ctx.formula = "t_high = ln(2) (R1 + R2) C; t_low = ln(2) R2 C; f = 1 / (ln(2) (R1 + 2 R2) C)"
    ctx.formatted = (
        f"555 astable: f = {fmt_sig(1 / period, 6)} Hz, high {fmt_sig(t_high, 6)} s, low {fmt_sig(t_low, 6)} s, "
        f"duty cycle {fmt_sig(100 * t_high / period, 5)}%."
    )
    return {
        "t_high": t_high,
        "t_low": t_low,
        "period": period,
        "f": 1 / period,
        "duty": t_high / period,
    }


def _thermistor(model: str, p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if model == "fit":
        return _fit(p, ctx)
    key = one_of(p, ("T", "R"), "the thermistor (temperature or resistance)")
    if model == "beta":
        r0 = need(p, "R0", "electrical_resistance")
        t0 = temperature(p, "T0")
        beta = need(p, "B", None)
        assert t0 is not None
        ctx.formula = "R = R0 exp(B (1/T - 1/T0)); 1/T = 1/T0 + ln(R/R0) / B"
        if key == "T":
            t = temperature(p, "T")
            assert t is not None
            res = r0 * math.exp(beta * (1 / t - 1 / t0))
        else:
            res = need(p, "R", "electrical_resistance")
            t = 1 / (1 / t0 + math.log(res / r0) / beta)
    else:
        a, b = need(p, "a", None, positive=False), need(p, "b", None, positive=False)
        c = need(p, "c", None, positive=False)
        ctx.formula = "1/T = a + b ln R + c (ln R)^3 (Steinhart-Hart)"
        if key == "R":
            res = need(p, "R", "electrical_resistance")
            lr = math.log(res)
            t = 1 / (a + b * lr + c * lr**3)
        else:
            t = temperature(p, "T")
            assert t is not None
            if c <= 0 or b <= 0:
                raise ToolError(
                    "INVALID_ARGUMENT", "R from T needs b > 0 and c > 0 (Steinhart-Hart inverse)."
                )
            y = (a - 1 / t) / (2 * c)
            x = y + math.sqrt((b / (3 * c)) ** 3 + y * y)
            res = math.exp(b / (3 * c * math.cbrt(x)) - math.cbrt(x))
            ctx.formula += "; inverse ln R = b/(3 c x^(1/3)) - x^(1/3), y = (a - 1/T)/(2c), x = y + sqrt((b/(3c))^3 + y^2)"
        if t <= 0:
            raise ToolError(
                "INVALID_ARGUMENT", "The coefficients give a temperature <= 0 K for this R."
            )
    ctx.formatted = f"Thermistor: R = {fmt_sig(res, 6)} ohm at T = {fmt_sig(t, 6)} K ({fmt_sig(t - 273.15, 6)} C)."
    return {"R": res, "T": t, "T_C": t - 273.15}


def _fit(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    pts = p.get("points")
    if not isinstance(pts, list) or len(pts) not in (2, 3):
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.points: 2 points [T in K, R in ohm] for B, 3 points for Steinhart-Hart.",
            '{"points": [[273.15, 32650], [298.15, 10000]]}',
        )
    rows: list[tuple[float, float]] = []
    for pt in pts:
        good = isinstance(pt, list | tuple) and len(pt) == 2
        if not good or not all(
            isinstance(x, int | float) and not isinstance(x, bool) and x > 0 for x in pt
        ):
            raise ToolError(
                "INVALID_ARGUMENT", "params.points: each point is [T in K, R in ohm], both > 0."
            )
        rows.append((float(pt[0]), float(pt[1])))
    if len({t for t, _ in rows}) != len(rows):
        raise ToolError("INVALID_ARGUMENT", "params.points: the temperatures must be different.")
    if len(rows) == 2:
        (t1, r1), (t2, r2) = rows
        beta = math.log(r1 / r2) / (1 / t1 - 1 / t2)
        ctx.formula = "B = ln(R1/R2) / (1/T1 - 1/T2)"
        ctx.formatted = f"Thermistor: B = {fmt_sig(beta, 6)} K (R0 = {fmt_sig(r1, 6)} ohm at T0 = {fmt_sig(t1, 6)} K)."
        return {"B": beta, "R0": r1, "T0": t1}
    m = np.array([[1.0, math.log(r), math.log(r) ** 3] for _, r in rows])
    a, b, c = (float(v) for v in np.linalg.solve(m, [1 / t for t, _ in rows]))
    ctx.formula = "1/T_i = a + b ln R_i + c (ln R_i)^3 for the 3 points (3 x 3 linear system)"
    ctx.formatted = (
        f"Steinhart-Hart: a = {fmt_sig(a, 6)}, b = {fmt_sig(b, 6)}, c = {fmt_sig(c, 6)}."
    )
    return {"a": a, "b": b, "c": c}
