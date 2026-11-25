"""geom3d_calc: integer counts, sizes checked first, no nan inside an ok answer (D93).

Found by the audit of 25/11/2026: pcd_holes with n = 5.9 drew 5 holes; pattern built a list of `count` points
before the size check; mesh_properties of a flat mesh gave centroid and inertia nan with ok = true; curve_points
of 1/t on [0, 1] gave a nan point and a negative length (-0.847) with ok = true. Two messages of pattern were in
Italian. Found while testing the fix: an involute with base_radius 1e306 gave ok = true with length "oo"; a curve
without a domain (helix, involute, arc...) can only leave the float range, so that is the MATH_ERROR of D74.
"""

from __future__ import annotations

import math
import time

from oce_math_mcp.errors import NOT_FINITE
from tests.conftest import call, err, ok


def g(**kw: object) -> dict:
    return ok("geom3d_calc", **kw)


def gerr(**kw: object) -> str:
    return err("geom3d_calc", **kw)


def test_counts_must_be_integers() -> None:
    assert len(g(operation="pcd_holes", params={"n": 6, "pcd_diameter": 100})["holes"]) == 6
    assert len(g(operation="pcd_holes", params={"n": 6.0, "pcd_diameter": 100})["holes"]) == 6
    for n in (5.9, 0, -3, "6", True, float("nan")):
        code = gerr(operation="pcd_holes", params={"n": n, "pcd_diameter": 100})
        assert code == "INVALID_ARGUMENT", n
    assert gerr(operation="pcd_holes", params={"pcd_diameter": 100}) == "INVALID_ARGUMENT"
    assert (
        gerr(operation="curve_points", params={"curve": "circle", "radius": 1, "n": 10.5})
        == "INVALID_ARGUMENT"
    )
    assert (
        gerr(operation="curve_points", params={"curve": "circle", "radius": 1, "n": 1})
        == "INVALID_ARGUMENT"
    )
    assert (
        gerr(operation="pattern", params={"kind": "linear", "count": 2.5, "spacing": 1})
        == "INVALID_ARGUMENT"
    )
    assert (
        gerr(operation="pattern", params={"kind": "circular", "count": 3.5, "radius": 1})
        == "INVALID_ARGUMENT"
    )
    assert (
        gerr(operation="pattern", params={"kind": "grid", "count": [2, 1.5], "spacing": [1, 1]})
        == "INVALID_ARGUMENT"
    )
    # n_holes is read by solid_properties (feature_points has no flange: it failed for another reason)
    flange = {"shape": "flange", "D": 100, "d": 20, "thickness": 10, "hole_d": 8, "pcd": 70}
    vols = [
        g(operation="solid_properties", params={**flange, "n_holes": n})["volume"] for n in (4, 4.0)
    ]
    ring_minus_holes = math.pi * (50**2 - 10**2) * 10 - 4 * math.pi * 4**2 * 10
    assert vols[0] == vols[1] and math.isclose(vols[0], ring_minus_holes, rel_tol=1e-12)
    for n in (4.5, -1, True):
        code = gerr(operation="solid_properties", params={**flange, "n_holes": n})
        assert code == "INVALID_ARGUMENT", n


def test_sizes_are_checked_before_building_the_list() -> None:
    t = time.perf_counter()
    for params in (
        {"kind": "linear", "count": 10_000_000, "spacing": 1},
        {"kind": "circular", "count": 10_000_000, "radius": 1},
        {"kind": "grid", "count": [100_000, 100_000], "spacing": [1, 1]},
        {"kind": "grid", "count": [101, 100], "spacing": [1, 1]},
    ):
        assert gerr(operation="pattern", params=params) == "LIMIT_EXCEEDED", params
    assert gerr(operation="pcd_holes", params={"n": 1e300, "pcd_diameter": 100}) == "LIMIT_EXCEEDED"
    assert time.perf_counter() - t < 2.0  # no list of 1e7 points (that takes tens of seconds)
    assert (
        gerr(operation="pcd_holes", params={"n": 10_001, "pcd_diameter": 100}) == "LIMIT_EXCEEDED"
    )
    assert (
        gerr(operation="curve_points", params={"curve": "circle", "radius": 1, "n": 10_001})
        == "LIMIT_EXCEEDED"
    )
    assert (
        len(g(operation="pcd_holes", params={"n": 10_000, "pcd_diameter": 100})["holes"]) == 10_000
    )
    assert (
        g(operation="pattern", params={"kind": "grid", "count": [100, 100], "spacing": [1, 1]})[
            "count"
        ]
        == 10_000
    )


def test_flat_mesh_has_no_centroid() -> None:
    resp = call(
        "geom3d_calc",
        operation="mesh_properties",
        params={
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            "faces": [[0, 1, 2], [1, 3, 2]],
        },
    )
    r = resp["result"]
    assert resp["ok"] and r["volume"] == 0.0 and r["centroid"] is None and r["inertia"] is None
    assert math.isclose(r["area"], 1.0) and any("not defined" in w for w in resp["warnings"])


def test_curve_outside_its_domain_is_an_error() -> None:
    bad = {"curve": "parametric", "x": "1/t", "y": "t", "z": "0", "t_range": [0, 1], "n": 3}
    assert gerr(operation="curve_points", params=bad) == "INVALID_ARGUMENT"
    # a reversed t_range runs the curve backwards: the length stays positive
    fwd = g(
        operation="curve_points",
        params={"curve": "parametric", "x": "t", "y": "t**2", "z": "0", "t_range": [0, 1]},
    )
    back = g(
        operation="curve_points",
        params={"curve": "parametric", "x": "t", "y": "t**2", "z": "0", "t_range": [1, 0]},
    )
    exact = math.sqrt(5) / 2 + math.asinh(2) / 4
    assert math.isclose(fwd["length"], exact, rel_tol=1e-10) and math.isclose(
        back["length"], exact, rel_tol=1e-10
    )
    # finite points but a divergent length: t sin(1/t) has infinite length on (0, 1]
    wild = {
        "curve": "parametric",
        "x": "t",
        "y": "t*sin(1/t)",
        "z": "0",
        "t_range": [1e-9, 1],
        "n": 5,
    }
    resp = call("geom3d_calc", operation="curve_points", params=wild)
    assert not resp["ok"] and resp["error"]["code"] == "MATH_ERROR", resp


def test_curves_beyond_the_float_range_give_the_d74_error() -> None:
    """Points or length that overflow are the MATH_ERROR of D74, as in D95: these curves have no domain."""
    for params in (
        {"curve": "helix", "radius": 1, "pitch": 1e308, "turns": 10, "n": 5},
        {"curve": "involute", "base_radius": 1e306, "max_angle_deg": 3000, "n": 5},
    ):
        resp = call("geom3d_calc", operation="curve_points", params=params)
        assert not resp["ok"] and resp["error"]["code"] == "MATH_ERROR", resp
        assert resp["error"]["message"] == NOT_FINITE
    r = g(
        operation="curve_points",
        params={"curve": "involute", "base_radius": 10, "max_angle_deg": 40, "n": 5},
    )
    assert math.isclose(r["length"], 10 * math.radians(40) ** 2 / 2, rel_tol=1e-9)


def test_pattern_messages_are_in_english() -> None:
    for params in ({"kind": "spiral", "count": 3}, {"kind": "grid", "count": 3, "spacing": 1}):
        resp = call("geom3d_calc", operation="pattern", params=params)
        msg = resp["error"]["message"]
        assert " o " not in msg and " e " not in msg, msg
