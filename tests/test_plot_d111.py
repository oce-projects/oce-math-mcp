"""D111: math_numeric.plot, an SVG plot written by the server with a table of points.

The checks read the SVG back: the frame rectangle gives the pixel mapping, so every drawn point and
every segment midpoint is compared with the function itself.
"""

from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from oce_math_mcp import limits, plot
from tests.conftest import call, err, ok

_POINT = re.compile(r"([ML])(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)")


@pytest.fixture(autouse=True)
def _no_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checks of the drawing and of the breaks must not depend on the load of the machine. With the
    server's 2.5 s limit, the full suite under load from another program cut the sampling of zeta(x) and lost
    the pole at 1, with the time-limit warning (26/11/2026). The time limit has its own tests below, which give
    the deadline to plot.make."""
    monkeypatch.setattr(limits, "PLOT_TIME_S", 600.0)


def elements(svg: str, kind: str) -> list[ET.Element]:
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    return [e for e in root.iter() if e.tag.endswith(kind)]


def frame(svg: str) -> tuple[float, float, float, float]:
    rects = [e for e in elements(svg, "rect") if e.get("class") == "frame"]
    assert len(rects) == 1
    r = rects[0]
    return (
        float(r.get("x", "")),
        float(r.get("y", "")),
        float(r.get("width", "")),
        float(r.get("height", "")),
    )


def subpaths(svg: str, curve: int = 0) -> list[list[tuple[float, float]]]:
    """Pixel points of one curve, one list per 'M'."""
    paths = [e for e in elements(svg, "path") if e.get("class") == f"curve c{curve}"]
    assert len(paths) == 1
    out: list[list[tuple[float, float]]] = []
    for cmd, x, y in _POINT.findall(paths[0].get("d", "")):
        if cmd == "M":
            out.append([])
        out[-1].append((float(x), float(y)))
    return out


class Mapping:
    def __init__(self, res: dict[str, Any]) -> None:
        self.x0, self.y0, self.w, self.h = frame(res["svg"])
        self.a, self.b = res["x_range"]
        self.lo, self.hi = res["y_range"]

    def data_x(self, px: float) -> float:
        return self.a + (px - self.x0) / self.w * (self.b - self.a)

    def px(self, x: float) -> float:
        return self.x0 + (x - self.a) / (self.b - self.a) * self.w

    def py(self, y: float) -> float:
        return self.y0 + self.h - (y - self.lo) / (self.hi - self.lo) * self.h


def _value(f: Any, x: float) -> float:
    try:
        return float(f(x))
    except (ValueError, ZeroDivisionError, OverflowError):
        return math.nan


def worst_gap(res: dict[str, Any], f: Any, curve: int = 0) -> tuple[float, float]:
    """Largest pixel distance of the drawn points and of the segment midpoints from the curve.

    The curve is a polyline of 20 samples per pixel; the distance is to its nearest segment within
    0.6 px. Points within 0.1 px of a reported break are skipped: there f has no single value."""
    m = Mapping(res)
    xs = np.linspace(m.a, m.b, int(20 * m.w) + 1)
    dpx = np.array([m.px(float(x)) for x in xs])
    dpy = np.array([m.py(_value(f, float(x))) for x in xs])
    cuts = [m.px(b["x"]) for b in res["breaks"] if b["curve"] == curve]

    def dist(px: float, py: float) -> float:
        i0 = max(0, int((px - 0.6 - m.x0) * 20))
        i1 = min(len(xs) - 1, int((px + 0.6 - m.x0) * 20) + 1)
        ax, ay, bx, by = dpx[i0:i1], dpy[i0:i1], dpx[i0 + 1 : i1 + 1], dpy[i0 + 1 : i1 + 1]
        keep = np.isfinite(ay) & np.isfinite(by)
        ax, ay, bx, by = ax[keep], ay[keep], bx[keep], by[keep]
        if ax.size == 0:
            return math.inf
        vx, vy = bx - ax, by - ay
        ll = vx * vx + vy * vy
        s = np.clip(((px - ax) * vx + (py - ay) * vy) / np.where(ll > 0, ll, 1.0), 0.0, 1.0)
        return float(np.min(np.hypot(ax + s * vx - px, ay + s * vy - py)))

    pts_gap = mid_gap = 0.0
    for sub in subpaths(res["svg"], curve):
        for px, py in sub:
            if all(abs(px - c) > 0.1 for c in cuts):
                pts_gap = max(pts_gap, dist(px, py))
        for (p0, q0), (p1, q1) in pairwise(sub):
            pm, qm = (p0 + p1) / 2, (q0 + q1) / 2
            if all(abs(pm - c) > 0.1 for c in cuts):
                mid_gap = max(mid_gap, dist(pm, qm))
    return pts_gap, mid_gap


def warnings_of(**args: Any) -> list[str]:
    resp = call("math_numeric", operation="plot", **args)
    assert resp["ok"], resp["error"]
    return resp["warnings"]


def test_sine_points_on_the_curve_and_table() -> None:
    res = ok("math_numeric", operation="plot", expr="sin(x)", bounds=[0, "2*pi"])
    assert res["x_range"] == [0.0, 2 * math.pi]
    assert res["y_range"] == pytest.approx([-1.1, 1.1], rel=1e-12, abs=0)
    assert res["breaks"] == []
    subs = subpaths(res["svg"])
    assert len(subs) == 1
    m = Mapping(res)
    assert subs[0][0][0] == pytest.approx(m.x0, abs=0.06)
    assert subs[0][-1][0] == pytest.approx(m.x0 + m.w, abs=0.06)
    pts_gap, mid_gap = worst_gap(res, math.sin)
    assert pts_gap <= 0.1
    assert mid_gap <= 0.5
    assert len(subs[0]) < 100  # 52 measured: the straight runs are simplified
    assert res["columns"] == ["x", "sin(x)"]
    xs = np.linspace(0, 2 * math.pi, 11)
    assert len(res["table"]) == 11
    for row, x in zip(res["table"], xs, strict=True):
        assert row[0] == pytest.approx(float(x), rel=5e-15, abs=0)  # 15 significant digits
        assert row[1] == pytest.approx(math.sin(x), rel=1e-14, abs=1e-15)


def test_table_samples_and_digits() -> None:
    res = ok("math_numeric", operation="plot", expr="exp(x)", bounds=[0, 1], samples=3, digits=4)
    assert res["table"] == [[0.0, 1.0], [0.5, 1.649], [1.0, 2.718]]


def test_odd_pole_breaks_the_curve() -> None:
    res = ok("math_numeric", operation="plot", expr="1/(x - 0.3)", bounds=[-1, 1])
    assert len(res["breaks"]) == 1
    br = res["breaks"][0]
    assert br["kind"] == "asymptote" and br["curve"] == 0
    assert br["x"] == pytest.approx(0.3, rel=0, abs=1e-9)
    assert br["left"] is None and br["right"] is None
    m = Mapping(res)
    subs = subpaths(res["svg"])
    assert len(subs) == 2
    cut = m.px(0.3)
    assert max(p for p, _ in subs[0]) <= cut + 0.06
    assert min(p for p, _ in subs[1]) >= cut - 0.06
    lines = [e for e in elements(res["svg"], "line") if e.get("class") == "asymptote"]
    assert len(lines) == 1
    assert float(lines[0].get("x1", "")) == pytest.approx(cut, abs=0.06)
    assert m.lo < 0 < m.hi and m.hi - m.lo < 100
    pts_gap, mid_gap = worst_gap(res, lambda x: 1 / (x - 0.3))
    assert pts_gap <= 0.1 and mid_gap <= 0.5
    assert any("outside the y window" in w for w in warnings_of(expr="1/(x - 0.3)", bounds=[-1, 1]))


def test_even_pole_between_grid_points() -> None:
    res = ok("math_numeric", operation="plot", expr="1/x**2", bounds=[-1, 2])
    assert [b["kind"] for b in res["breaks"]] == ["asymptote"]
    assert res["breaks"][0]["x"] == pytest.approx(0.0, rel=0, abs=1e-9)
    assert len(subpaths(res["svg"])) == 2


def test_tangent_four_asymptotes() -> None:
    res = ok("math_numeric", operation="plot", expr="tan(x)", bounds=[-5, 5])
    xs = [b["x"] for b in res["breaks"]]
    assert [b["kind"] for b in res["breaks"]] == ["asymptote"] * 4
    want = [-1.5 * math.pi, -0.5 * math.pi, 0.5 * math.pi, 1.5 * math.pi]
    for x, w in zip(xs, want, strict=True):
        assert x == pytest.approx(w, rel=0, abs=1e-9)
    assert len(subpaths(res["svg"])) == 5


def test_poles_exactly_on_floats_are_asymptotes() -> None:
    """gamma at -2 and -1 and zeta at 1: the bisection lands on the pole itself (a float); the values seen
    on the way decide, not the last two points 1 ulp apart."""
    res = ok("math_numeric", operation="plot", expr="gamma(x)", bounds=[-3, 3])
    assert [b["kind"] for b in res["breaks"]] == ["asymptote"] * 4
    for br, x in zip(res["breaks"], [-3, -2, -1, 0], strict=True):
        assert br["x"] == pytest.approx(x, rel=0, abs=1e-9)
    res = ok("math_numeric", operation="plot", expr="zeta(x)", bounds=[-5, 5])
    assert [(b["kind"], round(b["x"], 9)) for b in res["breaks"]] == [("asymptote", 1.0)]


def test_floor_jumps_with_side_values() -> None:
    res = ok("math_numeric", operation="plot", expr="floor(x)", bounds=[-0.5, 2.5])
    assert [b["kind"] for b in res["breaks"]] == ["jump"] * 3
    for br, (x, left, right) in zip(res["breaks"], [(0, -1, 0), (1, 0, 1), (2, 1, 2)], strict=True):
        assert br["x"] == pytest.approx(x, rel=0, abs=1e-9)
        assert br["left"] == left and br["right"] == right
    assert len(subpaths(res["svg"])) == 4
    pts_gap, mid_gap = worst_gap(res, math.floor)
    assert pts_gap <= 0.1 and mid_gap <= 0.5


def test_narrow_peak_is_drawn_whole() -> None:
    """The grid misses the top of the peak and most values are near 0: no pole, so no cut."""
    res = ok("math_numeric", operation="plot", expr="exp(-(x - 0.3)**2/1e-5)", bounds=[0, 1])
    assert res["breaks"] == []
    assert res["y_range"][1] > 1.0
    m = Mapping(res)
    top = min(py for sub in subpaths(res["svg"]) for _, py in sub)
    assert top <= m.py(0.99)
    assert not any(
        "outside the y window" in w
        for w in warnings_of(expr="exp(-(x - 0.3)**2/1e-5)", bounds=[0, 1])
    )


def test_oscillating_sines_within_half_a_pixel() -> None:
    """Measured 2026-11-26: midpoints 0.28 px (sin(50x)) and 0.24 px (sin(100x)); the quarter points of
    long segments catch the S shapes whose midpoint is an inflection."""
    for k in (50, 100):
        res = ok("math_numeric", operation="plot", expr=f"sin({k}*x)", bounds=[0, "2*pi"])
        pts_gap, mid_gap = worst_gap(res, lambda x, k=k: math.sin(k * x))
        assert pts_gap <= 0.1 and mid_gap <= 0.5


def test_smooth_peak_outside_a_small_window_is_not_a_pole() -> None:
    res = ok(
        "math_numeric",
        operation="plot",
        expr="exp(-(x - 0.3)**2/1e-5)",
        bounds=[0, 1],
        y_bounds=[0, 0.1],
    )
    assert res["breaks"] == []


def test_slow_singularities_between_grid_points() -> None:
    for expr in ("log(abs(x))", "abs(x)**(-0.5)"):
        res = ok("math_numeric", operation="plot", expr=expr, bounds=[-1, 2])
        assert [b["kind"] for b in res["breaks"]] == ["asymptote"]
        assert res["breaks"][0]["x"] == pytest.approx(0.0, rel=0, abs=1e-9)


def test_pole_inside_a_huge_window() -> None:
    res = ok("math_numeric", operation="plot", expr="1/x**2", bounds=[-1, 2], y_bounds=[0, 1e12])
    assert [b["kind"] for b in res["breaks"]] == ["asymptote"]
    assert res["breaks"][0]["x"] == pytest.approx(0.0, rel=0, abs=1e-9)


def test_steep_but_continuous_is_not_a_break() -> None:
    res = ok("math_numeric", operation="plot", expr="tanh(1000*x)", bounds=[-1, 1])
    assert res["breaks"] == []
    assert len(subpaths(res["svg"])) == 1
    pts_gap, mid_gap = worst_gap(res, lambda x: math.tanh(1000 * x))
    assert pts_gap <= 0.1 and mid_gap <= 0.5


def test_domain_edge_and_logarithm() -> None:
    res = ok("math_numeric", operation="plot", expr="sqrt(x)", bounds=[-1, 1])
    assert [b["kind"] for b in res["breaks"]] == ["domain_edge"]
    assert res["breaks"][0]["x"] == pytest.approx(0.0, rel=0, abs=1e-9)
    assert [row[1] for row in res["table"][:5]] == [None] * 5
    assert res["table"][5][1] == 0.0
    m = Mapping(res)
    subs = subpaths(res["svg"])
    assert len(subs) == 1
    assert subs[0][0][0] == pytest.approx(m.px(0.0), abs=0.06)
    res = ok("math_numeric", operation="plot", expr="log(x)", bounds=[-1, 2])
    assert [b["kind"] for b in res["breaks"]] == ["asymptote"]
    assert res["breaks"][0]["x"] == pytest.approx(0.0, rel=0, abs=1e-9)


def test_hole_jump_at_undefined_point_and_oscillation() -> None:
    res = ok("math_numeric", operation="plot", expr="sin(x)/x", bounds=[-1, 1])
    assert [b["kind"] for b in res["breaks"]] == ["hole"]
    br = res["breaks"][0]
    assert br["x"] == pytest.approx(0.0, rel=0, abs=1e-9)
    assert br["left"] == pytest.approx(1.0, rel=1e-9, abs=0)
    assert br["right"] == pytest.approx(1.0, rel=1e-9, abs=0)
    res = ok("math_numeric", operation="plot", expr="x/abs(x)", bounds=[-1, 1])
    assert [(b["kind"], b["left"], b["right"]) for b in res["breaks"]] == [("jump", -1.0, 1.0)]
    res = ok("math_numeric", operation="plot", expr="sin(1/x)", bounds=[-1, 1])
    assert [b["kind"] for b in res["breaks"]] == ["oscillation"]
    assert res["breaks"][0]["x"] == pytest.approx(0.0, rel=0, abs=1e-9)
    assert len(res["svg"]) < 20_000  # 4,013 measured: at most 2 points per pixel column


def test_second_pass_uses_the_narrow_window() -> None:
    """With the thresholds of the first window (-281 to 283, from the pole) the oscillation at 0 came out
    as a jump (measured 2026-11-26); the second pass in the window -14 to 16 sees it."""
    res = ok("math_numeric", operation="plot", expr="sin(1/x) + 1/(x - 0.5)", bounds=[-1, 1])
    assert [b["kind"] for b in res["breaks"]] == ["oscillation", "asymptote"]
    assert [b["x"] for b in res["breaks"]] == pytest.approx([0.0, 0.5], rel=0, abs=1e-9)


def test_five_oscillating_curves_stay_small() -> None:
    """16,771 characters measured; 24,225 without the limit of 2 points per pixel column."""
    expr = "sin(1/x); sin(2/x); sin(3/x); sin(5/x); sin(7/x)"
    res = ok("math_numeric", operation="plot", expr=expr, bounds=[-1, 1])
    assert len(res["svg"]) < 20_000
    assert [b["kind"] for b in res["breaks"]] == ["oscillation"] * 5


def test_two_curves_named_and_styled() -> None:
    res = ok("math_numeric", operation="plot", expr="sin(x); y = cos(x)", bounds=[0, 3])
    assert res["columns"] == ["x", "sin(x)", "y = cos(x)"]
    p0 = next(e for e in elements(res["svg"], "path") if e.get("class") == "curve c0")
    p1 = next(e for e in elements(res["svg"], "path") if e.get("class") == "curve c1")
    assert p0.get("stroke") != p1.get("stroke")
    texts = [e.text for e in elements(res["svg"], "text") if e.get("class") == "legend"]
    assert texts == ["sin(x)", "y = cos(x)"]
    _, mid_gap = worst_gap(res, math.cos, curve=1)
    assert mid_gap <= 0.5
    assert res["table"][-1] == [
        3.0,
        pytest.approx(math.sin(3.0), rel=1e-14),
        pytest.approx(math.cos(3.0), rel=1e-14),
    ]


def test_user_window_clips_inside_the_frame() -> None:
    res = ok("math_numeric", operation="plot", expr="x**3", bounds=[-2, 2], y_bounds=[-1, 1])
    assert res["y_range"] == [-1.0, 1.0]
    m = Mapping(res)
    subs = subpaths(res["svg"])
    assert len(subs) == 1
    for px, py in subs[0]:
        assert m.y0 - 0.06 <= py <= m.y0 + m.h + 0.06
        assert m.px(-1) - 0.06 <= px <= m.px(1) + 0.06
    assert not any(
        "outside the y window" in w
        for w in warnings_of(expr="x**3", bounds=[-2, 2], y_bounds=[-1, 1])
    )


def test_constant_functions_and_variable_name() -> None:
    res = ok("math_numeric", operation="plot", expr="3", bounds=[0, 1])
    assert res["y_range"] == pytest.approx([2.7, 3.3], rel=1e-12, abs=0)
    assert all(row[1] == 3.0 for row in res["table"])
    res = ok("math_numeric", operation="plot", expr="0", bounds=[0, 1])
    assert res["y_range"] == [-1.0, 1.0]
    res = ok("math_numeric", operation="plot", expr="x = 0.5", bounds=[0, 1])
    assert res["columns"] == ["x", "x - 1/2"]
    res = ok("math_numeric", operation="plot", expr="t**2", bounds=[0, 1])
    assert res["columns"] == ["t", "t**2"]
    texts = [e.text for e in elements(res["svg"], "text") if e.get("class") == "axis-name"]
    assert texts == ["t"]


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"expr": "sin(x)"}, "INVALID_ARGUMENT"),
        ({"expr": "sin(x)", "bounds": [1, 0]}, "INVALID_ARGUMENT"),
        ({"expr": "sin(x)", "bounds": ["-oo", 1]}, "INVALID_ARGUMENT"),
        ({"expr": "sin(x)", "bounds": [0, 1, 2]}, "INVALID_ARGUMENT"),
        ({"expr": "x; x; x; x; x; x", "bounds": [0, 1]}, "LIMIT_EXCEEDED"),
        ({"expr": "x*y", "bounds": [0, 1]}, "INVALID_ARGUMENT"),
        ({"expr": "x", "bounds": [0, 1], "variables": ["x", "y"]}, "INVALID_ARGUMENT"),
        ({"expr": "x", "bounds": [0, 1], "y_bounds": [1, 1]}, "INVALID_ARGUMENT"),
        ({"expr": "sqrt(-1 - x**2)", "bounds": [0, 1]}, "MATH_ERROR"),
    ],
)
def test_errors(args: dict[str, Any], code: str) -> None:
    assert err("math_numeric", operation="plot", **args) == code


def test_one_curve_without_values_is_a_warning() -> None:
    warns = warnings_of(expr="x; sqrt(-1 - x**2)", bounds=[0, 1])
    assert any("Curve 2 has no real values" in w for w in warns)


def test_nice_ticks_and_labels() -> None:
    assert plot.nice_ticks(0, 1) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0], abs=1e-15)
    assert plot.nice_ticks(-1.1, 1.1) == pytest.approx([-1, -0.5, 0, 0.5, 1], abs=1e-15)
    assert plot.nice_ticks(0, 7) == pytest.approx([0, 1, 2, 3, 4, 5, 6, 7], abs=1e-15)
    assert plot.nice_ticks(0, 8) == pytest.approx([0, 2, 4, 6, 8], abs=1e-15)
    assert plot.nice_ticks(3, 3.6) == pytest.approx([3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6], abs=1e-12)
    assert plot.tick_label(0.30000000000000004, 0.1) == "0.3"
    assert plot.tick_label(-1e-17, 0.5) == "0"
    assert plot.tick_label(-0.5, 0.5) == "-0.5"
    assert plot.tick_label(20.0, 5.0) == "20"
    assert plot.tick_label(0.05, 0.05) == "0.05"
    assert plot.tick_label(2e7, 1e7) == "2e+07"
    assert plot.tick_label(3e-5, 1e-5) == "3e-05"


def test_clip_segment() -> None:
    box = (0.0, 1.0, 0.0, 1.0)
    assert plot.clip_segment(0.5, 0.5, 0.5, 10.0, *box) == pytest.approx((0.5, 0.5, 0.5, 1.0))
    assert plot.clip_segment(-1.0, 0.5, 2.0, 0.5, *box) == pytest.approx((0.0, 0.5, 1.0, 0.5))
    assert plot.clip_segment(-1.0, -1.0, 2.0, 2.0, *box) == pytest.approx((0.0, 0.0, 1.0, 1.0))
    assert plot.clip_segment(2.0, 2.0, 3.0, 0.5, *box) is None
    assert plot.clip_segment(0.2, 2.0, 0.8, 2.0, *box) is None
    assert plot.clip_segment(0.2, 0.3, 0.4, 0.6, *box) == pytest.approx((0.2, 0.3, 0.4, 0.6))


def test_labels_are_escaped() -> None:
    out = plot.make([lambda x: x], ['a<b & "c"'], 0.0, 1.0)
    assert "a&lt;b &amp; &quot;c&quot;" in out["svg"]
    texts = [e.text for e in elements(out["svg"], "text") if e.get("class") == "legend"]
    assert texts == ['a<b & "c"']


def test_time_limit_keeps_a_coarse_plot() -> None:
    out = plot.make([math.sin], ["sin(x)"], 0.0, 1.0, deadline=time.monotonic() - 1.0)
    assert any("time limit" in w for w in out["warnings"])
    assert out["points"] == [2]
    assert len(subpaths(out["svg"])) == 1


def test_time_reserve_stops_the_sampling_first() -> None:
    """Deadline 0.3 s away, inside the reserve of 0.75 s kept for the break searches: the grid and the
    refinement stop at once, without the time limit itself being reached."""
    deadline = time.monotonic() + 0.3
    out = plot.make([math.sin], ["sin(x)"], 0.0, 1.0, deadline=deadline)
    assert out["points"] == [2]
    assert any("time limit" in w for w in out["warnings"])


def test_clock_reserve_and_single_evaluation() -> None:
    clock = plot._Clock(time.monotonic() + 10.0)
    assert not clock.over()
    assert clock.over(20.0) and clock.trimmed and not clock.stopped
    assert not clock.over()
    assert plot._Clock(None).over(1e9) is False
    calls: list[float] = []

    def f(x: float) -> float:
        calls.append(x)
        return 1 / x

    ev = plot._safe(f)
    assert ev(2.0) == 0.5 and ev(2.0) == 0.5 and math.isnan(ev(0.0)) and math.isnan(ev(0.0))
    assert calls == [2.0, 0.0]
    assert math.isnan(plot._safe(lambda x: math.inf)(1.0))


def test_svg_size_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limits, "PLOT_SVG_MAX_CHARS", 500)
    assert err("math_numeric", operation="plot", expr="sin(x)", bounds=[0, 1]) == "LIMIT_EXCEEDED"
