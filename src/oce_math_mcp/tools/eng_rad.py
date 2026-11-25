"""eng_calc: radiation between grey and diffuse surfaces (D54).

Sources (Wikipedia pages):
- «View factor»: view factors of infinitely long 2D geometries (parallel plates with aligned
  midlines, inclined plates with a common edge, perpendicular plates with a common
  edge, three-sided enclosure), reciprocity A_i F_ij = A_j F_ji, row sum = 1;
- «Thermal radiation»: two grey surfaces that form an enclosure,
  Q = sigma (T1^4 - T2^4) / [(1 - e1) / (A1 e1) + 1 / (A1 F12) + (1 - e2) / (A2 e2)];
- «Radiosity (radiometry)»: enclosure of N surfaces, J_i = e_i sigma T_i^4 + (1 - e_i) sum_j F_ij J_j,
  Q_i = A_i (J_i - G_i), and the circuit analogy (surface and space resistances), from which
  come the shields between parallel plates (resistances in series).
3D view factors (discs, rectangles) are not included: the source refers to external tables (D54).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, integer, need, only, op, temperature, units_for
from .eng_thermo import SIGMA_SB

_VF_GEOMETRIES: dict[str, set[str]] = {
    "parallel_plates": {"w_i", "w_j", "L"},
    "inclined_plates": {"w", "alpha_deg"},
    "perpendicular_plates": {"w_i", "w_j"},
    "three_sided": {"w_i", "w_j", "w_k"},
}
_TS_CONFIGS: dict[str, set[str]] = {
    "general": {"A1", "A2", "F12"},
    "parallel_plates": {"A", "n_shields", "eps_shield"},
    "enclosed": {"A1", "A2"},
}
_TS_COMMON = {"T1", "T2", "eps1", "eps2"}
_KEYS = (
    {"mode", "geometry", "config", "F", "eps", "T", "Q"}
    | set().union(*_VF_GEOMETRIES.values())
    | set().union(*_TS_CONFIGS.values())
    | _TS_COMMON
)
_UNITS = {
    "F_ij": "1",
    "F_ji": "1",
    "F12": "1",
    "F21": "1",
    "Q": "W",
    "q1": "W/m2",
    "R_total": "1/m2",
    "T_shields": "K",
    "J": "W/m2",
    "G": "W/m2",
    "T": "K",
}


def _eps(p: dict[str, Any], key: str) -> float:
    e = need(p, key, None)
    if e > 1:
        raise ToolError(
            "INVALID_ARGUMENT", f"params.{key} (emissivity) must be between 0 (excluded) and 1."
        )
    return e


@op("radiation_exchange", tuple(sorted(_KEYS)))
def _radiation(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", ("view_factor_2d", "two_surface", "enclosure"))
    if mode == "view_factor_2d":
        geo = choice(p, "geometry", tuple(_VF_GEOMETRIES))
        only(p, mode, _VF_GEOMETRIES[geo] | {"geometry"})
        out = _view_factor(geo, p, ctx)
        ctx.formatted = (
            f"View factor F_ij = {fmt_sig(out['F_ij'], 6)}, F_ji = {fmt_sig(out['F_ji'], 6)}."
        )
    elif mode == "two_surface":
        cfg = choice(p, "config", tuple(_TS_CONFIGS))
        only(p, mode, _TS_CONFIGS[cfg] | _TS_COMMON | {"config"})
        out = _two_surface(cfg, p, ctx)
        ctx.formatted = f"Net heat from 1 to 2: Q = {fmt_sig(out['Q'], 6)} W."
    else:
        only(p, mode, {"A", "F", "eps", "T", "Q"})
        out = _enclosure(p, ctx)
        ctx.formatted = (
            "Net heat of each surface: " + ", ".join(f"{fmt_sig(q, 6)} W" for q in out["Q"]) + "."
        )
    ctx.units = units_for(out, _UNITS)
    return out


# ------------------------------------------------------------------ 2D view factors


def _view_factor(geo: str, p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if geo == "parallel_plates":
        wi, wj, L = (need(p, k, "length") for k in ("w_i", "w_j", "L"))
        a, b = wi / L, wj / L
        # D76: same value without the difference of two close roots, 4ab / (s1 + s2) / (2a)
        f = 2 * b / (math.hypot(a + b, 2) + math.hypot(b - a, 2))  # hypot: no overflow
        ctx.formula = (
            "F_ij = {[(W_i + W_j)^2 + 4]^(1/2) - [(W_j - W_i)^2 + 4]^(1/2)} / (2 W_i), W = w / L "
            "(computed as 2 W_j / ([...]^(1/2) + [...]^(1/2)), same value without cancellation)"
        )
    elif geo == "inclined_plates":
        wi = wj = need(p, "w", "length")
        alpha = need(p, "alpha_deg", None)
        if alpha >= 180:
            raise ToolError(
                "INVALID_ARGUMENT", "params.alpha_deg must be between 0 and 180 degrees (excluded)."
            )
        # = 1 - sin(alpha / 2); 180 - alpha is exact in degrees, so it stays accurate near 180 (D76)
        f = 2 * math.sin(math.radians(180 - alpha) / 4) ** 2
        ctx.formula = "F_ij = 1 - sin(alpha / 2), plates of equal width with a common edge"
    elif geo == "perpendicular_plates":
        wi, wj = need(p, "w_i", "length"), need(p, "w_j", "length")
        r = wj / wi
        # = (1 + r - sqrt(1 + r^2)) / 2 without cancellation; hypot does not overflow (D76)
        f = r / (1 + r + math.hypot(1.0, r))
        ctx.formula = "F_ij = {1 + w_j / w_i - [1 + (w_j / w_i)^2]^(1/2)} / 2"
    else:
        wi, wj, wk = (need(p, k, "length") for k in ("w_i", "w_j", "w_k"))
        if not (wi < wj + wk and wj < wi + wk and wk < wi + wj):
            raise ToolError(
                "INVALID_ARGUMENT",
                "The three widths do not form a triangle: each must be less than the sum of the others.",
            )
        f = (wi + wj - wk) / (2 * wi)
        ctx.formula = "F_ij = (w_i + w_j - w_k) / (2 w_i)"
    ctx.warn("2D geometry: infinitely long surfaces; the widths act as areas in the reciprocity.")
    return {"F_ij": f, "F_ji": f * wi / wj}


# ------------------------------------------------------------------ two surfaces


def _two_surface(cfg: str, p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    t1, t2 = temperature(p, "T1"), temperature(p, "T2")
    assert t1 is not None and t2 is not None
    e1, e2 = _eps(p, "eps1"), _eps(p, "eps2")
    dt4 = SIGMA_SB * (t1**4 - t2**4)
    if cfg == "parallel_plates":
        area = need(p, "A", "area")
        n = integer(p, "n_shields", minimum=0) if p.get("n_shields") is not None else 0
        if n > 0:
            es = _eps(p, "eps_shield")
        elif p.get("eps_shield") is not None:
            raise ToolError(
                "INVALID_ARGUMENT", "params.eps_shield is used only with n_shields > 0."
            )
        else:
            es = 1.0
        r_plates = 1 / e1 + 1 / e2 - 1
        r_shield = 2 / es - 1
        r_area = r_plates + n * r_shield
        q = dt4 / r_area
        out: dict[str, Any] = {"Q": q * area, "q1": q, "R_total": r_area / area, "F12": 1.0}
        if n > 0:
            eb = SIGMA_SB * t1**4 - q * (1 / e1 + 1 / es - 1)
            temps = []
            for _ in range(n):
                temps.append((eb / SIGMA_SB) ** 0.25)
                eb -= q * r_shield
            out["T_shields"] = temps
        ctx.formula = (
            "infinite parallel plates: Q = sigma A (T1^4 - T2^4) / [1/e1 + 1/e2 - 1 + N (2/e_s - 1)], "
            "N shields of emissivity e_s (resistances in series)"
        )
        return out
    a1, a2 = need(p, "A1", "area"), need(p, "A2", "area")
    if cfg == "enclosed":
        if a1 > a2:
            raise ToolError(
                "INVALID_ARGUMENT",
                "The inner surface 1 cannot be larger than surface 2 that encloses it.",
            )
        f12 = 1.0
    else:
        f12 = need(p, "F12", None)
        if f12 > 1:
            raise ToolError("INVALID_ARGUMENT", "params.F12 must be between 0 (excluded) and 1.")
        if a1 * f12 > a2 * (1 + 1e-12):
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Reciprocity impossible: F21 = A1 F12 / A2 = {fmt_sig(a1 * f12 / a2, 6)} > 1.",
            )
    r = (1 - e1) / (a1 * e1) + 1 / (a1 * f12) + (1 - e2) / (a2 * e2)
    q = dt4 / r
    ctx.formula = "Q = sigma (T1^4 - T2^4) / [(1 - e1)/(A1 e1) + 1/(A1 F12) + (1 - e2)/(A2 e2)]"
    if cfg == "general":
        ctx.warn("The formula holds if surfaces 1 and 2 alone form the enclosure (F11 = 1 - F12).")
    return {"Q": q, "q1": q / a1, "R_total": r, "F12": f12, "F21": a1 * f12 / a2}


# ------------------------------------------------------------------ enclosure of N surfaces


def _num_list(
    p: dict[str, Any], key: str, n: int | None, *, nulls: bool = False
) -> list[float | None]:
    raw = p.get(key)
    ok = isinstance(raw, list) and (n is None or len(raw) == n) and len(raw) >= 2
    if ok:
        assert isinstance(raw, list)
        for x in raw:
            if x is None and nulls:
                continue
            if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(float(x)):
                ok = False
    if not ok:
        size = f"of {n} numbers" if n else "of at least 2 numbers"
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: list {size}"
            + (" (null where the value is unknown)." if nulls else "."),
        )
    assert isinstance(raw, list)
    return [None if x is None else float(x) for x in raw]


def _enclosure(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    areas = _num_list(p, "A", None)
    n = len(areas)
    a = np.array(areas, dtype=float)
    if np.any(a <= 0):
        raise ToolError("INVALID_ARGUMENT", "params.A: areas > 0 in m2.")
    raw_f = p.get("F")
    if not isinstance(raw_f, list) or len(raw_f) != n:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.F: {n} x {n} matrix of the view factors (list of rows).",
        )
    rows = [_num_list({"r": r}, "r", n) for r in raw_f]
    f = np.array(rows, dtype=float)
    if np.any(f < 0) or np.any(f > 1):
        raise ToolError("INVALID_ARGUMENT", "params.F: view factors between 0 and 1.")
    for i in range(n):
        s = float(f[i].sum())
        if abs(s - 1) > 1e-6:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Row {i + 1} of F: the sum is {fmt_sig(s, 8)}, it must be 1 (summation rule).",
            )
    af = a[:, None] * f
    if np.any(np.abs(af - af.T) > 1e-6 * np.maximum(np.maximum(af, af.T), 1e-300)):
        i, j = (int(v) for v in np.unravel_index(int(np.argmax(np.abs(af - af.T))), af.shape))
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Reciprocity rule violated: A{i + 1} F{i + 1}{j + 1} = {fmt_sig(af[i, j], 6)} but "
            f"A{j + 1} F{j + 1}{i + 1} = {fmt_sig(af[j, i], 6)}.",
        )
    eps = _num_list(p, "eps", n)
    e = np.array(eps, dtype=float)
    if np.any(e <= 0) or np.any(e > 1):
        raise ToolError("INVALID_ARGUMENT", "params.eps: emissivity between 0 (excluded) and 1.")
    temps = _num_list(p, "T", n, nulls=True)
    heats = _num_list(p, "Q", n, nulls=True) if p.get("Q") is not None else [None] * n
    for i in range(n):
        if (temps[i] is None) == (heats[i] is None):
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Surface {i + 1}: exactly one of the temperature T and the heat Q is needed (the other null).",
            )
    if all(t is None for t in temps):
        raise ToolError(
            "INVALID_ARGUMENT", "At least one surface with known temperature is needed."
        )
    m = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        t = temps[i]
        if t is not None:
            if t <= 0:
                raise ToolError("INVALID_ARGUMENT", f"params.T: temperature {i + 1} in K, > 0.")
            m[i] = -(1 - e[i]) * f[i]
            m[i, i] += 1
            b[i] = e[i] * SIGMA_SB * t**4
        else:
            q = heats[i]
            assert q is not None
            m[i] = -f[i]
            m[i, i] += 1
            b[i] = q / a[i]
    try:
        j = np.linalg.solve(m, b)
    except np.linalg.LinAlgError as exc:
        raise ToolError(
            "INVALID_ARGUMENT", "Singular radiosity system: check F and the known data."
        ) from exc
    g = f @ j
    q_out = a * (j - g)
    t_out: list[float] = []
    for i in range(n):
        t = temps[i]
        if t is not None:
            t_out.append(t)
            continue
        eb = (j[i] - (1 - e[i]) * g[i]) / e[i]
        if eb <= 0:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Surface {i + 1}: inconsistent data, the temperature comes out negative.",
            )
        t_out.append(float((eb / SIGMA_SB) ** 0.25))
    ctx.formula = (
        "J_i = e_i sigma T_i^4 + (1 - e_i) sum_j F_ij J_j (T known); J_i - sum_j F_ij J_j = Q_i / A_i (Q known); "
        "G_i = sum_j F_ij J_j; Q_i = A_i (J_i - G_i)"
    )
    total = float(np.sum(np.abs(q_out)))
    if (
        total > 0 and abs(float(np.sum(q_out))) > 1e-9 * total
    ):  # pragma: no cover - consistency check
        ctx.warn("The sum of the net heats is not zero: check the data.")
    return {
        "J": [float(x) for x in j],
        "G": [float(x) for x in g],
        "Q": [float(x) for x in q_out],
        "T": t_out,
    }
