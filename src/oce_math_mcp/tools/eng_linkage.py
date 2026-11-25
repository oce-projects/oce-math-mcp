"""eng_calc: linkages and epicyclic gear trains (D57).

Sources (Wikipedia pages):
- "Four-bar linkage": Grashof condition S + L <= P + Q; classification in 8 cases with
  T1 = g + h - a - b, T2 = b + g - a - h, T3 = b + h - a - g (a input crank, b output link,
  g ground link, h coupler); time ratio Q = (180 + beta) / (180 - beta);
- "Piston motion equations": x = r cos A + sqrt(l^2 - r^2 sin^2 A), v = x' w, a = x'' w^2;
- "Epicyclic gearing": N_s w_s + N_r w_r = (N_s + N_r) w_c; differential w_c = (w_L + w_R) / 2.
Derived here (shown in the `formula` field): position of the four-bar as the intersection of two
circles, velocities from the derivative of the loop-closure equation (2x2 linear system), transmission
angle, number of planet teeth N_p = (N_r - N_s) / 2 and planet speed from the mesh with the sun
N_p (w_p - w_c) = -N_s (w_s - w_c).
"""

from __future__ import annotations

import math
from typing import Any

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, only, op, qty, units_for

_LINK_MODES: dict[str, set[str]] = {
    "grashof": {"a", "b", "g", "h"},
    "four_bar": {"a", "b", "g", "h", "theta2_deg", "omega2", "assembly"},
    "slider_crank": {"r", "l", "angle_deg", "omega", "rpm"},
    "quick_return": {"beta_deg", "Q"},
}
_CLASSES = {
    ("-", "-", "+"): (True, "crank", "crank"),
    ("+", "+", "+"): (True, "crank", "rocker"),
    ("+", "-", "-"): (True, "rocker", "crank"),
    ("-", "+", "-"): (True, "rocker", "rocker"),
    ("-", "-", "-"): (False, "0-rocker", "0-rocker"),
    ("-", "+", "+"): (False, "pi-rocker", "pi-rocker"),
    ("+", "-", "+"): (False, "pi-rocker", "0-rocker"),
    ("+", "+", "-"): (False, "0-rocker", "pi-rocker"),
}
_LINK_UNITS = {
    "T1": "m",
    "T2": "m",
    "T3": "m",
    "S_plus_L": "m",
    "P_plus_Q": "m",
    "A": "m",
    "B": "m",
    "theta3_deg": "deg",
    "theta4_deg": "deg",
    "omega3": "rad/s",
    "omega4": "rad/s",
    "transmission_angle_deg": "deg",
    "x": "m",
    "dx_dA": "m/rad",
    "d2x_dA2": "m/rad2",
    "v": "m/s",
    "a": "m/s2",
    "omega": "rad/s",
    "Q": "1",
    "beta_deg": "deg",
}


def _lengths(p: dict[str, Any]) -> tuple[float, float, float, float]:
    a, b, g, h = (need(p, k, "length") for k in ("a", "b", "g", "h"))
    return a, b, g, h


@op("linkage", ("mode", *sorted(set().union(*_LINK_MODES.values()))))
def _linkage(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_LINK_MODES))
    only(p, mode, _LINK_MODES[mode])
    if mode == "grashof":
        out = _grashof(p, ctx)
    elif mode == "four_bar":
        out = _four_bar(p, ctx)
    elif mode == "slider_crank":
        out = _slider_crank(p, ctx)
    else:
        if (p.get("beta_deg") is None) == (p.get("Q") is None):
            raise ToolError("INVALID_ARGUMENT", "Give only one of beta_deg and Q.")
        if p.get("Q") is not None:
            q = need(p, "Q", None)
            if q < 1:
                raise ToolError(
                    "INVALID_ARGUMENT", "params.Q (slow time / fast time) must be >= 1."
                )
            beta = 180 * (q - 1) / (q + 1)
        else:
            beta = need(p, "beta_deg", None, positive=False)
            if not 0 <= beta < 180:
                raise ToolError(
                    "INVALID_ARGUMENT", "params.beta_deg goes from 0 to 180 degrees (excluded)."
                )
            q = (180 + beta) / (180 - beta)
        out = {"Q": q, "beta_deg": beta}
        ctx.formula = "Q = (180 + beta) / (180 - beta); beta = 180 (Q - 1) / (Q + 1)"
        ctx.formatted = f"Time ratio Q = {fmt_sig(q, 6)}, beta = {fmt_sig(beta, 6)} degrees."
    ctx.units = units_for(out, _LINK_UNITS)
    return out


def _grashof(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a, b, g, h = _lengths(p)
    t1, t2, t3 = g + h - a - b, b + g - a - h, b + h - a - g
    srt = sorted((a, b, g, h))
    signs = tuple("+" if t >= 0 else "-" for t in (t1, t2, t3))
    grashof, inp, outp = _CLASSES[signs]  # type: ignore[index]
    if any(t == 0 for t in (t1, t2, t3)):
        ctx.warn(
            "One term T is zero: change-point (transition) linkage, the class is on the limit."
        )
    ctx.formula = (
        "Grashof: S + L <= P + Q; T1 = g + h - a - b, T2 = b + g - a - h, T3 = b + h - a - g; "
        "class from the signs of T1, T2, T3 (table of the source)"
    )
    kind = "Grashof" if grashof else "non-Grashof"
    ctx.formatted = f"Four-bar linkage {kind}: input {inp}, output {outp}."
    return {
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "S_plus_L": srt[0] + srt[3],
        "P_plus_Q": srt[1] + srt[2],
        "grashof": grashof,
        "input": inp,
        "output": outp,
    }


def _four_bar(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a, b, g, h = _lengths(p)
    th2 = math.radians(need(p, "theta2_deg", None, positive=False))
    sign = p.get("assembly", 1)
    if sign not in (1, -1) or isinstance(sign, bool):
        raise ToolError("INVALID_ARGUMENT", "params.assembly: 1 or -1 (the two assembly modes).")
    ax, ay = a * math.cos(th2), a * math.sin(th2)
    dx, dy = g - ax, -ay  # from A to the fixed pivot O4 = (g, 0)
    d = math.hypot(dx, dy)
    if d > h + b or d < abs(h - b) or d == 0:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"The four-bar does not close at theta2 = {fmt_sig(math.degrees(th2), 6)} degrees: the distance "
            f"A-O4 = {fmt_sig(d, 6)} m is outside [|h - b|, h + b].",
        )
    # intersection of the circles (A, h) and (O4, b)
    along = (h * h - b * b + d * d) / (2 * d)
    perp = math.sqrt(max(h * h - along * along, 0.0))
    ux, uy = dx / d, dy / d
    bx = ax + along * ux - sign * perp * uy
    by = ay + along * uy + sign * perp * ux
    th3 = math.atan2(by - ay, bx - ax)
    th4 = math.atan2(by, bx - g)
    cos_mu = ((bx - ax) * (bx - g) + (by - ay) * by) / (h * b)
    mu = math.degrees(math.acos(max(-1.0, min(1.0, cos_mu))))
    out: dict[str, Any] = {
        "A": [ax, ay],
        "B": [bx, by],
        "theta3_deg": math.degrees(th3),
        "theta4_deg": math.degrees(th4),
        "transmission_angle_deg": min(mu, 180 - mu),
    }
    ctx.formula = (
        "A = a (cos t2, sin t2); B = intersection of the circles (A, h) and (O4 = (g, 0), b); "
        "velocities: a w2 (-sin t2, cos t2) + h w3 (-sin t3, cos t3) = b w4 (-sin t4, cos t4)"
    )
    if p.get("omega2") is not None:
        w2 = need(p, "omega2", "angular_velocity", positive=False)
        # h w3 (-s3, c3) - b w4 (-s4, c4) = -a w2 (-s2, c2)
        m11, m12 = -h * math.sin(th3), b * math.sin(th4)
        m21, m22 = h * math.cos(th3), -b * math.cos(th4)
        r1, r2 = a * w2 * math.sin(th2), -a * w2 * math.cos(th2)
        det = m11 * m22 - m12 * m21
        if abs(det) < 1e-12 * h * b:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Singular position (coupler and rocker aligned): velocities are not defined.",
            )
        out["omega3"] = (r1 * m22 - m12 * r2) / det
        out["omega4"] = (m11 * r2 - r1 * m21) / det
    ctx.formatted = (
        f"theta3 = {fmt_sig(out['theta3_deg'], 6)} degrees, theta4 = {fmt_sig(out['theta4_deg'], 6)} degrees, "
        f"transmission angle {fmt_sig(out['transmission_angle_deg'], 4)} degrees."
    )
    return out


def _slider_crank(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    r, rod = need(p, "r", "length"), need(p, "l", "length")
    if rod <= r:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Slider-crank: l > r is needed (connecting rod longer than the crank).",
        )
    a = math.radians(need(p, "angle_deg", None, positive=False))
    s, c = math.sin(a), math.cos(a)
    root = math.sqrt(rod * rod - r * r * s * s)
    x = r * c + root
    x1 = -r * s - r * r * s * c / root
    x2 = -r * c - r * r * (c * c - s * s) / root - r**4 * s * s * c * c / root**3
    out: dict[str, Any] = {"x": x, "dx_dA": x1, "d2x_dA2": x2}
    ctx.formula = (
        "x = r cos A + sqrt(l^2 - r^2 sin^2 A); x' = -r sin A - r^2 sin A cos A / sqrt(l^2 - r^2 sin^2 A); "
        "x'' = -r cos A - r^2 (cos^2 A - sin^2 A) / sqrt(...) - r^4 sin^2 A cos^2 A / sqrt(...)^3; v = x' w, a = x'' w^2"
    )
    if p.get("omega") is not None and p.get("rpm") is not None:
        raise ToolError("INVALID_ARGUMENT", "Give only one of omega and rpm.")
    if p.get("omega") is not None or p.get("rpm") is not None:
        w = (
            need(p, "omega", "angular_velocity", positive=False)
            if p.get("omega") is not None
            else need(p, "rpm", None, positive=False) * 2 * math.pi / 60
        )
        out |= {"omega": w, "v": x1 * w, "a": x2 * w * w}
    ctx.formatted = f"Piston at x = {fmt_sig(x, 6)} m from the crank centre" + (
        f", v = {fmt_sig(out['v'], 6)} m/s, a = {fmt_sig(out['a'], 6)} m/s2." if "v" in out else "."
    )
    return out


# ------------------------------------------------------------------ planetary_gear

_GEAR_MODES: dict[str, set[str]] = {
    "planetary": {"N_s", "N_r", "omega_s", "omega_r", "omega_c"},
    "differential": {"omega_L", "omega_R", "omega_c"},
}
_GEAR_UNITS: dict[str, str] = {}


@op("planetary_gear", ("mode", *sorted(set().union(*_GEAR_MODES.values()))))
def _planetary(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_GEAR_MODES))
    only(p, mode, _GEAR_MODES[mode])
    keys = (
        ("omega_s", "omega_r", "omega_c")
        if mode == "planetary"
        else ("omega_L", "omega_R", "omega_c")
    )
    known = {k: qty(p, k, None, need=False, positive=False) for k in keys}
    missing = [k for k, v in known.items() if v is None]
    if len(missing) != 1:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Two speeds out of three are needed ({', '.join(keys)}), in the same unit; the server computes the third.",
        )
    if mode == "differential":
        wl, wr, wc = known["omega_L"], known["omega_R"], known["omega_c"]
        if wc is None:
            assert wl is not None and wr is not None
            wc = (wl + wr) / 2
        elif wr is None:
            assert wl is not None
            wr = 2 * wc - wl
        else:
            assert wr is not None
            wl = 2 * wc - wr
        ctx.formula = "w_c = (w_L + w_R) / 2"
        ctx.formatted = f"w_L = {fmt_sig(wl, 6)}, w_R = {fmt_sig(wr, 6)}, w_c = {fmt_sig(wc, 6)}."
        return {"omega_L": wl, "omega_R": wr, "omega_c": wc}
    ns, nr = integer(p, "N_s"), integer(p, "N_r")
    if nr <= ns:
        raise ToolError(
            "INVALID_ARGUMENT", "The ring gear N_r must have more teeth than the sun gear N_s."
        )
    if (nr - ns) % 2:
        raise ToolError(
            "INVALID_ARGUMENT",
            "N_r - N_s must be even: the planet has N_p = (N_r - N_s) / 2 teeth, an integer.",
        )
    n_p = (nr - ns) // 2
    ws, wr, wc = known["omega_s"], known["omega_r"], known["omega_c"]
    if wc is None:
        assert ws is not None and wr is not None
        wc = (ns * ws + nr * wr) / (ns + nr)
    elif wr is None:
        assert ws is not None
        wr = ((ns + nr) * wc - ns * ws) / nr
    else:
        assert wr is not None
        ws = ((ns + nr) * wc - nr * wr) / ns
    wp = wc - ns * (ws - wc) / n_p
    out: dict[str, Any] = {"omega_s": ws, "omega_r": wr, "omega_c": wc, "omega_p": wp, "N_p": n_p}
    if wr == 0 and wc != 0:
        out["ratio_s_to_c"] = ws / wc
    if ws == 0 and wc != 0:
        out["ratio_r_to_c"] = wr / wc
    if wc == 0 and wr != 0:
        out["ratio_s_to_r"] = ws / wr
    ctx.formula = (
        "N_s w_s + N_r w_r = (N_s + N_r) w_c; planet (derived): N_p = (N_r - N_s) / 2, "
        "N_p (w_p - w_c) = -N_s (w_s - w_c)"
    )
    ctx.formatted = (
        f"Sun {fmt_sig(ws, 6)}, ring {fmt_sig(wr, 6)}, carrier {fmt_sig(wc, 6)}, planet {fmt_sig(wp, 6)} "
        "(same unit as the inputs)."
    )
    return out
