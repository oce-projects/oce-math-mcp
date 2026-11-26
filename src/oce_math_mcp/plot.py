"""Function plots as SVG (D111): adaptive sampling, breaks, ticks, clipping and drawing.

No plotting library: the SVG text is written here. The functions are float callables such as the ones of
numfunc.compile_expr (nan outside the domain). Steps for each curve:

1. a uniform grid of 513 points, evaluated coarse to fine so a time limit still leaves an even grid;
2. refinement: a segment is halved while its midpoint is more than 0.4 px from the chord (at most 12
   halvings and 10,000 evaluations per curve), with the y values clamped to the window +- one window
   height; every tested midpoint stays as a point, so the drawn segments are halves of accepted ones.
   The grid and the refinement stop at 70% of the time limit, so the break searches always run;
3. breaks: a bisection across the segments still unresolved at the last halving (jumps, odd poles,
   oscillation), a bisection toward each undefined point (domain edges, holes, poles where the function
   is undefined) and a golden-section search on |f| around each spike (poles that keep their sign);
4. drawing: Liang-Barsky clipping to the window, at most 2 points per pixel column, Ramer-Douglas-Peucker
   simplification at 0.25 px, coordinates with 1 decimal.

A search decides from the values of its last third of steps: |f| that keeps growing is an asymptote,
values that stay within 1% of the window height have a limit, anything else is an oscillation.
"""

from __future__ import annotations

import html
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from . import limits
from .errors import ToolError

Fn = Callable[[float], float]

GRID_LEVELS = 9  # 2**9 + 1 = 513 grid points
MAX_DEPTH = 12  # halvings of a grid step
REFINE_MAX = 10_000  # refinement evaluations per curve
RESERVE = 0.3  # fraction of PLOT_TIME_S kept for the break searches
SEARCH_STEPS = 60  # steps of each break search
BREAKS_MAX = 50  # searches of each kind per curve
TOL_PX = 0.4  # largest distance of a midpoint from the chord
SIMPLIFY_PX = 0.25  # Ramer-Douglas-Peucker tolerance
SETTLE = 0.01  # a value with a limit moves less than this fraction of the window height
SAME_VALUE = 1e-6  # two one-sided limits closer than this fraction of the window height are equal
WIDTH, PLOT_H = 640, 324
LEFT, RIGHT, BOTTOM = 64, 16, 40
PLOT_W = WIDTH - LEFT - RIGHT
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")  # Okabe-Ito, colour-blind safe
DASHES = ("", "6 3", "2 3", "8 3 2 3", "4 4")
LABEL_MAX = 40
_INVPHI = (math.sqrt(5.0) - 1.0) / 2.0


class _Clock:
    """Time limit of the sampling; once passed, every later step is skipped.

    over(reserve) is also True `reserve` seconds before the limit (trimmed): the grid and the
    refinement leave that time to the break searches."""

    def __init__(self, deadline: float | None) -> None:
        self.deadline = deadline
        self.stopped = False
        self.trimmed = False

    def over(self, reserve: float = 0.0) -> bool:
        if self.stopped:
            return True
        if self.deadline is None:
            return False
        now = time.monotonic()
        if now > self.deadline:
            self.stopped = True
            return True
        if reserve > 0 and now > self.deadline - reserve:
            self.trimmed = True
            return True
        return False


def _safe(f: Fn) -> Fn:
    """f with nan for every error and for infinite values; each x is evaluated once (the second pass
    in a narrower window reuses the values)."""
    memo: dict[float, float] = {}

    def ev(x: float) -> float:
        v = memo.get(x)
        if v is None:
            try:
                v = float(f(x))
            except (ValueError, OverflowError, ZeroDivisionError, TypeError):
                v = math.nan
            if not math.isfinite(v):
                v = math.nan
            memo[x] = v
        return v

    return ev


def _fin(v: float) -> bool:
    return math.isfinite(v)


# ------------------------------------------------------------------ ticks and labels


def nice_ticks(lo: float, hi: float, most: int = 8) -> list[float]:
    """Tick values k * step: the smallest step 1, 2 or 5 times a power of ten with at most `most` ticks."""
    e0 = math.floor(math.log10((hi - lo) / most))
    for e in range(e0, e0 + 3):
        for m in (1.0, 2.0, 5.0):
            step = m * 10.0**e
            k0 = math.ceil(lo / step - 1e-9)
            k1 = math.floor(hi / step + 1e-9)
            if k1 - k0 + 1 <= most:
                return [k * step for k in range(k0, k1 + 1)]
    return [lo, hi]


def tick_label(v: float, step: float) -> str:
    """Short label of a tick: fixed decimals from the step, exponent form for very small or large values."""
    if abs(v) < abs(step) * 1e-6:
        return "0"
    if abs(step) < 1e-4 or abs(v) >= 1e6:
        return f"{v:.6g}"
    decimals = max(0, -math.floor(math.log10(abs(step)) + 1e-9))
    return f"{v:.{decimals}f}"


# ------------------------------------------------------------------ window


def auto_window(values: Sequence[float], poles: bool = False) -> tuple[float, float, bool]:
    """y window from sampled values: (low, high, values_cut).

    The full range plus 5%. With poles (an asymptote was found) and a range more than 10 times the
    5%-95% quantile range, that quantile range plus 25% on each side (never past the full range plus 5%
    of the quantile range), and values_cut is True. Without poles a narrow peak is never cut."""
    v = np.array([y for y in values if _fin(y)], dtype=float)
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        d = abs(lo) * 0.1 or 1.0
        return lo - d, hi + d, False
    q_lo, q_hi = (float(q) for q in np.quantile(v, [0.05, 0.95]))
    if poles and q_hi > q_lo and hi - lo > 10 * (q_hi - q_lo):
        s = q_hi - q_lo
        return max(q_lo - 0.25 * s, lo - 0.05 * s), min(q_hi + 0.25 * s, hi + 0.05 * s), True
    s = hi - lo
    return lo - 0.05 * s, hi + 0.05 * s, False


# ------------------------------------------------------------------ sampling


def _grid(evs: Sequence[Fn], a: float, b: float, clock: _Clock) -> list[dict[float, float]]:
    """Grid of 2**GRID_LEVELS + 1 points, coarse levels first; the first level (a and b) always runs."""
    n = 2**GRID_LEVELS
    xs = [a + (b - a) * (i / n) for i in range(n + 1)]
    xs[-1] = b
    done: list[dict[int, float]] = [{} for _ in evs]
    for level in range(GRID_LEVELS + 1):
        step = n >> level
        for c, ev in enumerate(evs):
            if level > 0 and clock.over(RESERVE * limits.PLOT_TIME_S):
                break
            for i in range(0, n + 1, step):
                if i not in done[c]:
                    done[c][i] = ev(xs[i])
    return [{xs[i]: y for i, y in sorted(d.items())} for d in done]


@dataclass
class _Curve:
    pts: dict[float, float]
    lo: float
    hi: float
    breaks: list[dict[str, Any]] = field(default_factory=list)
    cuts: list[tuple[float, float]] = field(default_factory=list)  # no segment crosses (l, r)
    # values seen by a search on the way to a point
    hist: dict[float, list[float]] = field(default_factory=dict)
    exhausted: bool = False
    capped: bool = False

    def py(self, y: float) -> float:
        """Height in pixels above the bottom of the window, clamped to one window height outside it."""
        v = (y - self.lo) / (self.hi - self.lo) * PLOT_H
        return min(max(v, -PLOT_H), 2.0 * PLOT_H)

    @property
    def span(self) -> float:
        return self.hi - self.lo

    def taken(self, x0: float, x1: float) -> bool:
        """True when a break already found lies in [x0, x1]."""
        return any(x0 <= c1 and c0 <= x1 for c0, c1 in self.cuts) or any(
            x0 <= br["x"] <= x1 for br in self.breaks
        )


def _refine(ev: Fn, cv: _Curve, clock: _Clock) -> list[tuple[float, float, float, float]]:
    """Halves the segments whose midpoint is off the chord; returns the segments still unresolved."""
    used = 0

    def val(x: float) -> float:
        nonlocal used
        if x not in cv.pts:
            cv.pts[x] = ev(x)
            used += 1
        return cv.pts[x]

    xs = sorted(cv.pts)
    current = [(xs[i], cv.pts[xs[i]], xs[i + 1], cv.pts[xs[i + 1]]) for i in range(len(xs) - 1)]
    unresolved: list[tuple[float, float, float, float]] = []
    for depth in range(MAX_DEPTH + 1):
        nxt: list[tuple[float, float, float, float]] = []
        stop = False
        for seg in current:
            x0, y0, x1, y1 = seg
            if not (_fin(y0) and _fin(y1)):
                continue
            if stop or used >= REFINE_MAX or clock.over(RESERVE * limits.PLOT_TIME_S):
                if used >= REFINE_MAX:
                    cv.exhausted = True
                stop = True
                if depth > 0:
                    unresolved.append(seg)
                continue
            xm = 0.5 * (x0 + x1)
            if not x0 < xm < x1:
                continue
            ym = val(xm)
            if _fin(ym) and abs(cv.py(ym) - 0.5 * (cv.py(y0) + cv.py(y1))) <= TOL_PX:
                continue
            if depth == MAX_DEPTH:
                if _fin(ym):
                    unresolved.append(
                        (x0, y0, xm, ym) if abs(ym - y0) >= abs(y1 - ym) else (xm, ym, x1, y1)
                    )
                continue
            nxt.append((x0, y0, xm, ym))
            nxt.append((xm, ym, x1, y1))
        if stop:
            unresolved.extend(s for s in nxt if _fin(s[1]) and _fin(s[3]))
            break
        current = nxt
        if not current:
            break
    return unresolved


def _trend(hist: Sequence[float], span: float) -> str:
    """'grows', 'settles' or 'wanders', from the last third of the values of a search."""
    if len(hist) < 6:
        return "settles"
    tail = hist[(2 * len(hist)) // 3 - 1 :]
    mags = [abs(v) for v in tail]
    if all(m1 >= m0 for m0, m1 in pairwise(mags)) and mags[-1] - mags[0] > SETTLE * span:
        return "grows"
    if max(tail) - min(tail) <= SETTLE * span:
        return "settles"
    return "wanders"


def _jump(ev: Fn, cv: _Curve, seg: tuple[float, float, float, float]) -> None:
    """Bisection toward the largest change inside an unresolved segment."""
    left, yl, right, yr = seg
    hl, hr = [yl], [yr]
    for _ in range(SEARCH_STEPS):
        m = 0.5 * (left + right)
        if not left < m < right:
            break
        ym = ev(m)
        if not _fin(ym):  # undefined point: the edge search decides, with the values seen so far
            cv.pts[left], cv.pts[m], cv.pts[right] = yl, ym, yr
            cv.hist[left], cv.hist[right] = hl, hr
            return
        if abs(ym - yl) >= abs(yr - ym):
            right, yr = m, ym
        else:
            left, yl = m, ym
        hl.append(yl)
        hr.append(yr)
    tl, tr = _trend(hl, cv.span), _trend(hr, cv.span)
    if "grows" in (tl, tr):
        kind, lv, rv = "asymptote", None, None
    elif tl == tr == "settles":
        if abs(yr - yl) <= SAME_VALUE * cv.span:
            return  # continuous, only steep
        kind, lv, rv = "jump", yl, yr
    else:
        kind, lv, rv = "oscillation", None, None
    cv.pts[left], cv.pts[right] = yl, yr
    x = 0.5 * (left + right)
    cv.cuts.append((left, right))
    cv.breaks.append({"x": x, "kind": kind, "left": lv, "right": rv})


def _edge(ev: Fn, cv: _Curve, xf: float, yf: float, xn: float) -> tuple[float, str, float]:
    """Bisection from a defined point toward an undefined one: (edge x, trend, last defined value)."""
    hist = list(cv.hist.get(xf, [yf]))
    for _ in range(SEARCH_STEPS):
        m = 0.5 * (xf + xn)
        if m in (xf, xn):
            break
        ym = ev(m)
        if _fin(ym):
            xf, yf = m, ym
            hist.append(yf)
        else:
            xn = m
    cv.pts[xf] = yf
    cv.pts[xn] = math.nan
    return 0.5 * (xf + xn), _trend(hist, cv.span), yf


def _edges(ev: Fn, cv: _Curve, clock: _Clock, tol_x: float) -> None:
    """Domain edges, holes, undefined jumps and poles at undefined points."""
    xs = sorted(cv.pts)
    found: list[tuple[float, str, str, float]] = []  # x, side of the defined values, trend, value
    pairs = [
        (xs[i], xs[i + 1])
        for i in range(len(xs) - 1)
        if _fin(cv.pts[xs[i]]) != _fin(cv.pts[xs[i + 1]])
    ]
    if len(pairs) > 2 * BREAKS_MAX:
        cv.capped = True
        pairs = pairs[: 2 * BREAKS_MAX]
    for x0, x1 in pairs:
        if clock.over():
            return
        if _fin(cv.pts[x0]):
            x, trend, y = _edge(ev, cv, x0, cv.pts[x0], x1)
            found.append((x, "left", trend, y))
        else:
            x, trend, y = _edge(ev, cv, x1, cv.pts[x1], x0)
            found.append((x, "right", trend, y))
    found.sort()
    i = 0
    while i < len(found):
        x, side, trend, y = found[i]
        nxt = found[i + 1] if i + 1 < len(found) else None
        if nxt is not None and side == "left" and nxt[1] == "right" and nxt[0] - x <= tol_x:
            trends = {trend, nxt[2]}
            if "grows" in trends:
                br = {"x": 0.5 * (x + nxt[0]), "kind": "asymptote", "left": None, "right": None}
            elif "wanders" in trends:
                br = {"x": 0.5 * (x + nxt[0]), "kind": "oscillation", "left": None, "right": None}
            else:
                same = abs(nxt[3] - y) <= SAME_VALUE * cv.span
                br = {
                    "x": 0.5 * (x + nxt[0]),
                    "kind": "hole" if same else "jump",
                    "left": y,
                    "right": nxt[3],
                }
            i += 2
        else:
            kind = {"grows": "asymptote", "wanders": "oscillation"}.get(trend, "domain_edge")
            br = {"x": x, "kind": kind, "left": None, "right": None}
            i += 1
        cv.breaks.append(br)


def _peak(ev: Fn, cv: _Curve, left: float, right: float) -> None:
    """Golden-section search for the largest |f| in [left, right]: an asymptote when it keeps growing."""

    def g(x: float) -> float:
        v = ev(x)
        return abs(v) if _fin(v) else math.inf

    c = right - _INVPHI * (right - left)
    d = left + _INVPHI * (right - left)
    gc, gd = g(c), g(d)
    hist: list[float] = []
    for _ in range(SEARCH_STEPS):
        if gc >= gd:
            right, d, gd = d, c, gc
            c = right - _INVPHI * (right - left)
            gc = g(c)
        else:
            left, c, gc = c, d, gd
            d = left + _INVPHI * (right - left)
            gd = g(d)
        hist.append(max(gc, gd))
        if not left < c < d < right:
            break
    k = (2 * len(hist)) // 3 - 1
    if len(hist) < 6 or not (hist[-1] == math.inf or hist[-1] - hist[k] > SETTLE * cv.span):
        return
    yl, yr = ev(left), ev(right)
    if _fin(yl):
        cv.pts[left] = yl
    if _fin(yr):
        cv.pts[right] = yr
    x = 0.5 * (left + right)
    cv.cuts.append((left, right))
    cv.breaks.append({"x": x, "kind": "asymptote", "left": None, "right": None})


def _peaks(ev: Fn, cv: _Curve, clock: _Clock) -> None:
    """Poles between samples: local maxima of |f| outside the window or more than twice their neighbours."""
    xs = sorted(cv.pts)
    ys = [cv.pts[x] for x in xs]
    cands: list[tuple[float, int]] = []
    for k in range(1, len(xs) - 1):
        y0, y1, y2 = ys[k - 1], ys[k], ys[k + 1]
        if not (_fin(y0) and _fin(y1) and _fin(y2)):
            continue
        m0, m1, m2 = abs(y0), abs(y1), abs(y2)
        if not (m1 >= m0 and m1 >= m2 and (m1 > m0 or m1 > m2)):
            continue
        outside = y1 < cv.lo - cv.span or y1 > cv.hi + cv.span
        if outside or m1 > 2 * max(m0, m2):
            cands.append((m1, k))
    if len(cands) > BREAKS_MAX:
        cv.capped = True
    best = sorted(sorted(cands, reverse=True)[:BREAKS_MAX], key=lambda t: t[1])
    for _, k in best:
        if clock.over():
            return
        if not cv.taken(xs[k - 1], xs[k + 1]):
            _peak(ev, cv, xs[k - 1], xs[k + 1])


def _analyse(
    ev: Fn, grid: dict[float, float], lo: float, hi: float, a: float, b: float, clock: _Clock
) -> _Curve:
    cv = _Curve(pts=dict(grid), lo=lo, hi=hi)
    unresolved = _refine(ev, cv, clock)
    if len(unresolved) > BREAKS_MAX:
        cv.capped = True
        unresolved.sort(key=lambda s: -abs(cv.py(s[3]) - cv.py(s[1])))
        unresolved = unresolved[:BREAKS_MAX]
    for seg in sorted(unresolved):
        if clock.over():
            break
        if not cv.taken(seg[0], seg[2]):
            _jump(ev, cv, seg)
    _edges(ev, cv, clock, 1e-9 * (b - a))
    _peaks(ev, cv, clock)
    cv.breaks.sort(key=lambda br: br["x"])
    return cv


def _pieces(cv: _Curve) -> list[list[tuple[float, float]]]:
    """Runs of defined points, split at undefined points and at the cuts of the breaks."""
    xs = sorted(cv.pts)
    cuts = sorted(cv.cuts)
    out: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    ci = 0
    prev = -math.inf
    for x in xs:
        y = cv.pts[x]
        while ci < len(cuts) and cuts[ci][1] <= prev:
            ci += 1
        if ci < len(cuts) and cuts[ci][0] < x and cur:
            out.append(cur)
            cur = []
        if _fin(y):
            cur.append((x, y))
        elif cur:
            out.append(cur)
            cur = []
        prev = x
    if cur:
        out.append(cur)
    return out


# ------------------------------------------------------------------ drawing


def clip_segment(
    x0: float, y0: float, x1: float, y1: float, xmin: float, xmax: float, ymin: float, ymax: float
) -> tuple[float, float, float, float] | None:
    """Liang-Barsky: the part of the segment inside the rectangle, or None."""
    t0, t1 = 0.0, 1.0
    dx, dy = x1 - x0, y1 - y0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    sx, sy = (x0, y0) if t0 == 0.0 else (x0 + t0 * dx, y0 + t0 * dy)
    ex, ey = (x1, y1) if t1 == 1.0 else (x0 + t1 * dx, y0 + t1 * dy)
    return sx, sy, ex, ey


def _columns(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """At most the lowest and highest point of each pixel column, plus the two ends."""
    if len(pts) <= 2:
        return pts
    out: list[tuple[float, float]] = [pts[0]]
    i, n = 1, len(pts) - 1
    while i < n:
        col = math.floor(pts[i][0])
        j = i
        while j < n and math.floor(pts[j][0]) == col:
            j += 1
        group = pts[i:j]
        if len(group) > 2:
            k_lo = min(range(len(group)), key=lambda k: group[k][1])
            k_hi = max(range(len(group)), key=lambda k: group[k][1])
            out.extend(group[k] for k in sorted({k_lo, k_hi}))
        else:
            out.extend(group)
        i = j
    out.append(pts[-1])
    return out


def _simplify(pts: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker without recursion, distances with numpy."""
    n = len(pts)
    if n < 3:
        return pts
    arr = np.array(pts, dtype=float)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        p0, p1 = arr[i], arr[j]
        dx, dy = p1 - p0
        seg = arr[i + 1 : j]
        norm = math.hypot(dx, dy)
        if norm > 0:
            dist = np.abs(dy * (seg[:, 0] - p0[0]) - dx * (seg[:, 1] - p0[1])) / norm
        else:
            dist = np.hypot(seg[:, 0] - p0[0], seg[:, 1] - p0[1])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            idx = i + 1 + k
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
    return [p for p, kp in zip(pts, keep, strict=True) if kp]


def _polylines(
    pieces: list[list[tuple[float, float]]], a: float, b: float, lo: float, hi: float, top: float
) -> list[list[tuple[float, float]]]:
    """Pixel polylines of the visible parts of the pieces."""
    span = hi - lo
    far_lo, far_hi = lo - 1e6 * span, hi + 1e6 * span

    def px(x: float, y: float) -> tuple[float, float]:
        return LEFT + (x - a) / (b - a) * PLOT_W, top + PLOT_H - (y - lo) / span * PLOT_H

    out: list[list[tuple[float, float]]] = []
    for piece in pieces:
        line: list[tuple[float, float]] = []
        pts = [(x, min(max(y, far_lo), far_hi)) for x, y in piece]
        for (x0, y0), (x1, y1) in pairwise(pts):
            c = clip_segment(x0, y0, x1, y1, a, b, lo, hi)
            if c is None:
                continue
            start, end = (c[0], c[1]), (c[2], c[3])
            if not line or line[-1] != start:
                if len(line) > 1:
                    out.append(line)
                line = [start]
            line.append(end)
        if len(line) > 1:
            out.append(line)
    return [_simplify(_columns([px(x, y) for x, y in line]), SIMPLIFY_PX) for line in out]


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _path(lines: list[list[tuple[float, float]]]) -> str:
    parts: list[str] = []
    for line in lines:
        last = ""
        for k, (x, y) in enumerate(line):
            p = f"{x:.1f} {y:.1f}"
            if p == last:
                continue
            parts.append(("M" if k == 0 else "L") + p)
            last = p
    return "".join(parts)


def _legend(labels: Sequence[str]) -> tuple[list[tuple[float, float, str]], float]:
    """Position of each legend entry, rows filled from the left; returns the entries and the top margin."""
    rows, x, row = [], float(LEFT), 0
    for label in labels:
        text = label if len(label) <= LABEL_MAX else label[: LABEL_MAX - 1] + "…"
        width = 30 + 7 * len(text) + 16
        if x > LEFT and x + width > WIDTH - RIGHT:
            row += 1
            x = float(LEFT)
        rows.append((x, float(row), text))
        x += width
    return rows, 14.0 + 18.0 * (row + 1)


def _render(
    pieces: list[list[list[tuple[float, float]]]],
    labels: Sequence[str],
    asymptotes: list[float],
    a: float,
    b: float,
    lo: float,
    hi: float,
    variable: str,
) -> str:
    entries, top = _legend(labels)
    height = top + PLOT_H + BOTTOM
    lines = [_polylines(p, a, b, lo, hi, top) for p in pieces]

    def X(x: float) -> float:
        return LEFT + (x - a) / (b - a) * PLOT_W

    def Y(y: float) -> float:
        return top + PLOT_H - (y - lo) / (hi - lo) * PLOT_H

    title = f"Plot of {', '.join(labels)} for {variable} from {a:.6g} to {b:.6g}"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height:.0f}" '
        f'viewBox="0 0 {WIDTH} {height:.0f}" role="img" aria-label="{_esc(title)}">',
        f"<title>{_esc(title)}</title>",
        f'<rect width="{WIDTH}" height="{height:.0f}" fill="#ffffff"/>',
    ]
    xt, yt = nice_ticks(a, b), nice_ticks(lo, hi)
    xs_step = xt[1] - xt[0] if len(xt) > 1 else b - a
    ys_step = yt[1] - yt[0] if len(yt) > 1 else hi - lo
    bottom = top + PLOT_H
    for t in xt:
        out.append(
            f'<line class="grid" x1="{X(t):.1f}" y1="{top:.1f}" x2="{X(t):.1f}" y2="{bottom:.1f}" stroke="#e6e6e6"/>'
        )
    for t in yt:
        out.append(
            f'<line class="grid" x1="{LEFT}" y1="{Y(t):.1f}" x2="{LEFT + PLOT_W}" y2="{Y(t):.1f}" stroke="#e6e6e6"/>'
        )
    if a < 0 < b:
        out.append(
            f'<line class="axis" x1="{X(0):.1f}" y1="{top:.1f}" x2="{X(0):.1f}" y2="{bottom:.1f}" stroke="#888888"/>'
        )
    if lo < 0 < hi:
        out.append(
            f'<line class="axis" x1="{LEFT}" y1="{Y(0):.1f}" x2="{LEFT + PLOT_W}" y2="{Y(0):.1f}" stroke="#888888"/>'
        )
    out.append(
        f'<rect class="frame" x="{LEFT}" y="{top:.1f}" width="{PLOT_W}" height="{PLOT_H}" fill="none" stroke="#444444"/>'
    )
    out.append('<g font-family="sans-serif" font-size="12" fill="#333333">')
    for t in xt:
        out.append(
            f'<text class="tick" x="{X(t):.1f}" y="{bottom + 16:.1f}" text-anchor="middle">{_esc(tick_label(t, xs_step))}</text>'
        )
    for t in yt:
        out.append(
            f'<text class="tick" x="{LEFT - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{_esc(tick_label(t, ys_step))}</text>'
        )
    out.append(
        f'<text class="axis-name" x="{LEFT + PLOT_W}" y="{bottom + 34:.1f}" text-anchor="end">{_esc(variable)}</text>'
    )
    out.append("</g>")
    for x in asymptotes:
        out.append(
            f'<line class="asymptote" x1="{X(x):.1f}" y1="{top:.1f}" x2="{X(x):.1f}" y2="{bottom:.1f}" '
            'stroke="#999999" stroke-dasharray="4 4"/>'
        )
    for i, pl in enumerate(lines):
        dash = f' stroke-dasharray="{DASHES[i]}"' if DASHES[i] else ""
        out.append(
            f'<path class="curve c{i}" d="{_path(pl)}" fill="none" stroke="{COLORS[i]}" stroke-width="1.8" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash}/>'
        )
    for i, (x, row, text) in enumerate(entries):
        y = 16.0 + 18.0 * row
        dash = f' stroke-dasharray="{DASHES[i]}"' if DASHES[i] else ""
        out.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 24:.1f}" y2="{y:.1f}" stroke="{COLORS[i]}" stroke-width="2.5"{dash}/>'
        )
        out.append(
            f'<text class="legend" x="{x + 30:.1f}" y="{y + 4:.1f}" font-family="sans-serif" font-size="12" '
            f'fill="#333333">{_esc(text)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def _which(nums: list[int]) -> str:
    """'Curve 2' or 'Curves 1, 2 and 5'."""
    if len(nums) == 1:
        return f"Curve {nums[0]}"
    return "Curves " + ", ".join(map(str, nums[:-1])) + f" and {nums[-1]}"


def make(
    fs: Sequence[Fn],
    labels: Sequence[str],
    a: float,
    b: float,
    *,
    variable: str = "x",
    window: tuple[float, float] | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Samples and draws up to 5 functions on [a, b].

    Returns svg, y_range, breaks (curve, x, kind, left, right), points (samples per curve) and warnings.
    Raises MATH_ERROR when no function has a real value on the grid and LIMIT_EXCEEDED when the SVG is
    longer than limits.PLOT_SVG_MAX_CHARS."""
    clock = _Clock(deadline)
    evs = [_safe(f) for f in fs]
    warnings: list[str] = []
    grids = _grid(evs, a, b, clock)
    empty = [not any(_fin(y) for y in g.values()) for g in grids]
    if all(empty):
        raise ToolError(
            "MATH_ERROR",
            f"No function has a real value on [{a:.6g}, {b:.6g}].",
            "sqrt(x) with bounds [0, 4]",
        )
    for i, e in enumerate(empty):
        if e:
            warnings.append(f"Curve {i + 1} has no real values on [{a:.6g}, {b:.6g}]: {labels[i]}.")

    def analyse(lo: float, hi: float) -> list[_Curve]:
        return [_analyse(ev, g, lo, hi, a, b, clock) for ev, g in zip(evs, grids, strict=True)]

    if window is not None:
        lo, hi = window
        curves = analyse(lo, hi)
    else:
        grid_values = [y for g in grids for y in g.values()]
        lo, hi, _ = auto_window(grid_values)
        curves = analyse(lo, hi)
        if any(br["kind"] == "asymptote" for cv in curves for br in cv.breaks):
            lo2, hi2, cut = auto_window(grid_values, poles=True)
            if cut:  # sample again in the window of the regular part
                lo, hi = lo2, hi2
                curves = analyse(lo, hi)
                warnings.append(
                    f"Some values are outside the y window [{lo:.6g}, {hi:.6g}] and are cut; "
                    "pass y_bounds to choose the window."
                )
        else:  # the refined points too, so a narrow peak missed by the grid stays inside
            lo, hi, _ = auto_window([y for cv in curves for y in cv.pts.values()])
    if clock.stopped or clock.trimmed:
        warnings.append(
            f"Sampling stopped at the time limit of {limits.PLOT_TIME_S} s: the plot uses fewer points and "
            "some breaks may be missing."
        )
    spent = [i + 1 for i, cv in enumerate(curves) if cv.exhausted]
    if spent:
        warnings.append(
            f"{_which(spent)}: the refinement budget of {REFINE_MAX} evaluations ran out; details finer "
            "than the sampling may be missing."
        )
    capped = [i + 1 for i, cv in enumerate(curves) if cv.capped]
    if capped:
        warnings.append(
            f"{_which(capped)}: more than {BREAKS_MAX} possible breaks of one kind; only {BREAKS_MAX} checked."
        )
    asym = [br["x"] for cv in curves for br in cv.breaks if br["kind"] == "asymptote"]
    svg = _render([_pieces(cv) for cv in curves], labels, asym, a, b, lo, hi, variable)
    if len(svg) > limits.PLOT_SVG_MAX_CHARS:
        raise ToolError(
            "LIMIT_EXCEEDED",
            f"The SVG has {len(svg)} characters, more than {limits.PLOT_SVG_MAX_CHARS}.",
            "fewer functions or a smaller interval",
        )
    breaks = [{"curve": i, **br} for i, cv in enumerate(curves) for br in cv.breaks]
    return {
        "svg": svg,
        "y_range": [lo, hi],
        "breaks": breaks,
        "points": [len(cv.pts) for cv in curves],
        "warnings": warnings,
    }
