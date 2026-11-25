"""eng_calc: continuous beams with the theorem of three moments (D69).

Source: Wikipedia page "Theorem of three moments" in the local offline library:
M1 L1/(E1 I1) + 2 M2 (L1/(E1 I1) + L2/(E2 I2)) + M3 L2/(E2 I2) = 6 [settlement terms] - 6 [A1 X1/(E1 I1 L1) + A2 X2/(E2 I2 L2)],
A = area of the free bending-moment diagram of the span, X1 measured from the far support of the left span (A),
X2 from the far support of the right span (C); moments positive when sagging, settlements positive downward.
Load terms built here from area times centroid, as the page says: uniform load w on L: A X = w L^4 / 24 (the page's
uniform form 1/4 w l^3); point load P at a from the left end (b = L - a): A = P a b / 2, centroid (L + a)/3 from the left
and (L + b)/3 from the right.
Settlement term: with the page's conventions (sagging positive, settlement positive downward) a support that settles
must increase the sagging moment over it; the stiffness method in the tests gives
+6 [(D_B - D_A)/L1 + (D_B - D_C)/L2], the opposite sign of the formula printed on the page. The server uses the sign
checked by the stiffness method.
A fixed end is a support with an imaginary span of length 0 (standard use of the theorem). Reactions and span moments
from the statics of each span with its end moments.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, op, qty

_KEYS = ("spans", "EI", "settlements", "left", "right")
_UNITS = {
    "M_supports": "N*m",
    "reactions": "N",
    "M_span_max": "N*m",
    "x_span_max": "m",
    "M_max_abs": "N*m",
}


def _num(v: Any, what: str, positive: bool = True) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(float(v)):
        raise ToolError("INVALID_ARGUMENT", f"{what} must be a number (SI units).")
    if positive and float(v) <= 0:
        raise ToolError("INVALID_ARGUMENT", f"{what} must be > 0.")
    return float(v)


def _spans(p: dict[str, Any]) -> list[dict[str, Any]]:
    raw = p.get("spans")
    example = '{"spans": [{"L": 4, "w": 10000}, {"L": 5, "P": [[20000, 2]]}], "EI": 2.1e7}'
    if not isinstance(raw, list) or not raw:
        raise ToolError("INVALID_ARGUMENT", "params.spans: a list of spans {L, w, P, EI}.", example)
    ei_all = qty(p, "EI", None, need=False)
    spans = []
    for k, s in enumerate(raw, start=1):
        if not isinstance(s, dict) or set(s) - {"L", "w", "P", "EI"}:
            raise ToolError(
                "INVALID_ARGUMENT", f"params.spans[{k}]: an object with keys L, w, P, EI.", example
            )
        length = _num(s.get("L"), f"spans[{k}].L")
        w = _num(s.get("w", 0.0), f"spans[{k}].w", positive=False)
        ei = s.get("EI", ei_all)
        if ei is None:
            raise ToolError(
                "INVALID_ARGUMENT", f"spans[{k}]: EI missing (give params.EI for all spans)."
            )
        loads = []
        for pl in s.get("P", []) or []:
            if not isinstance(pl, list | tuple) or len(pl) != 2:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    f"spans[{k}].P: a list of [P in N, a in m from the left end].",
                )
            force, a = (
                _num(pl[0], f"spans[{k}].P force", positive=False),
                _num(pl[1], f"spans[{k}].P position", False),
            )
            if not 0 < a < length:
                raise ToolError(
                    "INVALID_ARGUMENT",
                    f"spans[{k}].P: the position must be inside the span (0 < a < L).",
                )
            loads.append((force, a))
        spans.append({"L": length, "w": w, "P": loads, "EI": _num(ei, f"spans[{k}].EI")})
    return spans


def _ax(span: dict[str, Any], from_left: bool) -> float:
    """Area of the free moment diagram times the distance of its centroid from the left (or right) end."""
    length, w = span["L"], span["w"]
    total = w * length**4 / 24
    for force, a in span["P"]:
        b = length - a
        total += force * a * b / 2 * ((length + a) / 3 if from_left else (length + b) / 3)
    return total


@op("continuous_beam", _KEYS)
def _beam(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    spans = _spans(p)
    n = len(spans)
    left = choice(p, "left", ("pinned", "fixed"), "pinned")
    right = choice(p, "right", ("pinned", "fixed"), "pinned")
    sett = p.get("settlements")
    if sett is None:
        delta = [0.0] * (n + 1)
    else:
        if not isinstance(sett, list) or len(sett) != n + 1:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"params.settlements: {n + 1} values in m (one per support, down +).",
            )
        delta = [_num(v, "settlements", positive=False) for v in sett]
    single = n == 1 and left == right == "pinned"
    moments = [0.0, 0.0] if single else _solve(spans, delta, left, right)
    out = _statics(spans, moments)
    out["M_supports"] = moments
    ctx.formula = (
        "M_{j-1} L_j/EI_j + 2 M_j (L_j/EI_j + L_{j+1}/EI_{j+1}) + M_{j+1} L_{j+1}/EI_{j+1} = "
        "6 [(D_j - D_{j-1})/L_j + (D_j - D_{j+1})/L_{j+1}] - 6 [A_j X_j/(EI_j L_j) + A_{j+1} X_{j+1}/(EI_{j+1} L_{j+1})]; "
        "uniform load A X = w L^4/24, point load A = P a b/2 at (L + a)/3; fixed end = span of length 0"
    )
    ctx.warn(
        "Moments positive when sagging; loads and settlements positive downward; Euler-Bernoulli beam."
    )
    ctx.units = {k: v for k, v in _UNITS.items() if k in out}
    inner = ", ".join(fmt_sig(m / 1000, 6) for m in moments)
    ctx.formatted = (
        f"Support moments {inner} kN*m; largest |M| = {fmt_sig(out['M_max_abs'] / 1000, 6)} kN*m."
    )
    return out


def _solve(spans: list[dict[str, Any]], delta: list[float], left: str, right: str) -> list[float]:
    n = len(spans)
    unknown = [
        j
        for j in range(n + 1)
        if (j == 0 and left == "fixed") or (j == n and right == "fixed") or 0 < j < n
    ]
    idx = {j: i for i, j in enumerate(unknown)}
    mat = np.zeros((len(unknown), len(unknown)))
    rhs = np.zeros(len(unknown))
    for j in unknown:
        row = idx[j]
        for side in (-1, 1):  # span on the left of support j (index j-1) and on the right (index j)
            k = j - 1 if side < 0 else j
            if not 0 <= k < n:
                continue  # imaginary span of length 0 at a fixed end
            s = spans[k]
            flex = s["L"] / s["EI"]
            mat[row, row] += 2 * flex
            other = j + side
            if other in idx:
                mat[row, idx[other]] += flex
            rhs[row] += 6 * (delta[j] - delta[other]) / s["L"]
            rhs[row] -= 6 * _ax(s, from_left=side < 0) / (s["EI"] * s["L"])
    sol = np.linalg.solve(mat, rhs)
    return [float(sol[idx[j]]) if j in idx else 0.0 for j in range(n + 1)]


def _statics(spans: list[dict[str, Any]], moments: list[float]) -> dict[str, Any]:
    reactions = [0.0] * (len(spans) + 1)
    m_max, x_max, m_abs = [], [], 0.0
    for k, s in enumerate(spans):
        length, w = s["L"], s["w"]
        loads = sorted(s["P"], key=lambda t: t[1])
        total = w * length + sum(f for f, _ in loads)
        r_free = w * length / 2 + sum(f * (length - a) / length for f, a in loads)
        m_l, m_r = moments[k], moments[k + 1]
        r_l = r_free + (m_r - m_l) / length
        reactions[k] += r_l
        reactions[k + 1] += total - r_l

        def moment(
            x: float, m_l: float = m_l, r_l: float = r_l, w: float = w, loads: list = loads
        ) -> float:
            return m_l + r_l * x - w * x * x / 2 - sum(f * (x - a) for f, a in loads if a < x)

        cands = [0.0, length] + [a for _, a in loads]
        if w != 0:  # zero shear inside each segment between point loads
            edges = [0.0, *[a for _, a in loads], length]
            v = r_l
            for i in range(len(edges) - 1):
                x0 = v / w
                if edges[i] < x0 < edges[i + 1]:
                    cands.append(x0)
                if i < len(loads):
                    v -= loads[i][0]
        best = max(cands, key=moment)
        m_max.append(moment(best))
        x_max.append(best)
        m_abs = max(m_abs, *(abs(moment(c)) for c in cands))
    return {"reactions": reactions, "M_span_max": m_max, "x_span_max": x_max, "M_max_abs": m_abs}
