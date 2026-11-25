"""geom3d_calc: 3D geometry, CAD, robotics and computations for 3D drawing."""

from __future__ import annotations

import math

import pytest

from tests.conftest import ok


def g(**kw: object) -> dict:
    return ok("geom3d_calc", **kw)


def close(a: list[float], b: list[float], tol: float = 1e-9) -> bool:
    return all(abs(p - q) < tol for p, q in zip(a, b, strict=True))


def test_vec_ops() -> None:
    r = g(operation="vec_ops", v1=[1, 0, 0], v2=[0, 1, 0])
    assert r["cross"] == [0, 0, 1] and r["dot"] == 0 and r["norm_v1"] == 1


def test_angle_between() -> None:
    assert abs(g(operation="angle_between", v1=[1, 0, 0], v2=[0, 1, 0])["angle_deg"] - 90) < 1e-12


def test_rotate_point_axis_angle() -> None:
    r = g(operation="rotate_point", v1=[1, 0, 0], axis=[0, 0, 1], angles_deg=[90])
    assert close(r["point"], [0, 1, 0])


def test_transform_and_inverse() -> None:
    r = g(operation="transform_4x4", angles_deg=[0, 0, 90], translation=[1, 2, 3], v1=[1, 0, 0])
    assert close(r["point"], [1, 3, 3])
    m, inv = r["matrix"], r["inverse"]
    prod = [[sum(m[i][k] * inv[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    assert close(
        [v for row in prod for v in row],
        [1.0 if i == j else 0.0 for i in range(4) for j in range(4)],
    )


def test_quaternion_roundtrip() -> None:
    r = g(operation="quat_euler", angles_deg=[10, 20, 30])
    back = g(operation="quat_euler", quat=r["quat"])
    assert close(back["angles_deg"], [10, 20, 30], 1e-9)
    assert abs(sum(q * q for q in r["quat"]) - 1) < 1e-12


def test_dh_planar_two_links() -> None:
    r = g(operation="dh_kinematics", dh_table=[[1, 0, 0, 0], [1, 0, 0, 90]])
    assert close(r["position"], [1, 1, 0])


def test_point_plane_and_line() -> None:
    r = g(
        operation="point_plane_distance",
        v1=[0, 0, 5],
        plane_point=[0, 0, 0],
        plane_normal=[0, 0, 2],
    )
    assert r["distance"] == 5 and r["signed_distance"] == 5
    r = g(
        operation="line_plane_intersect",
        v1=[0, 0, 5],
        v2=[0, 0, -1],
        plane_point=[0, 0, 1],
        plane_normal=[0, 0, 1],
    )
    assert close(r["intersection"], [0, 0, 1])
    r = g(
        operation="point_line_distance",
        v1=[0, 3, 0],
        params={"line_point": [0, 0, 0], "line_dir": [1, 0, 0]},
    )
    assert abs(r["distance"] - 3) < 1e-12


def test_coord_convert_roundtrip() -> None:
    sph = g(operation="coord_convert", v1=[1, 1, 1], coord_from="cartesian", coord_to="spherical")[
        "coords"
    ]
    assert abs(sph[0] - math.sqrt(3)) < 1e-12
    back = g(operation="coord_convert", v1=sph, coord_from="spherical", coord_to="cartesian")[
        "coords"
    ]
    assert close(back, [1, 1, 1])


def test_pcd_holes() -> None:
    r = g(operation="pcd_holes", params={"n": 4, "pcd_diameter": 100})
    assert close([c for h in r["holes"] for c in h], [50, 0, 0, 50, -50, 0, 0, -50], 1e-6)


def test_solid_cylinder() -> None:
    r = g(
        operation="solid_properties",
        params={"shape": "cylinder", "radius": 1, "height": 2, "density": 1000},
    )
    assert abs(r["volume"] - 2 * math.pi) < 1e-12
    assert abs(r["mass"] - 2000 * math.pi) < 1e-9


def test_section_rectangle_and_hollow_circle() -> None:
    r = g(operation="section_properties", params={"shape": "rectangle", "b": 100, "h": 200})
    assert abs(r["Ix"] - 100 * 200**3 / 12) < 1e-6
    r = g(operation="section_properties", params={"shape": "hollow_circle", "D": 50, "d": 40})
    assert abs(r["Ix"] - math.pi * (50**4 - 40**4) / 64) < 1e-6


def test_polygon_2d_square_and_l_shape() -> None:
    r = g(operation="polygon_2d", params={"points": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    assert r["area"] == 1 and close(r["centroid"], [0.5, 0.5]) and r["is_convex"]
    r = g(
        operation="polygon_2d", params={"points": [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]}
    )
    assert r["area"] == 3 and not r["is_convex"]


def test_circle_3pts() -> None:
    r = g(operation="circle_3pts", params={"points": [[1, 0], [0, 1], [-1, 0]]})
    assert close(r["center"], [0, 0]) and abs(r["radius"] - 1) < 1e-12


def test_arc_fillet_right_angle() -> None:
    r = g(
        operation="arc_fillet",
        params={"line1": [[0, 0], [10, 0]], "line2": [[0, 0], [0, 10]], "radius": 5},
    )
    assert close(r["center"], [5, 5])
    assert sorted([tuple(r["tangent1"]), tuple(r["tangent2"])]) == [(0.0, 5.0), (5.0, 0.0)]


def test_triangle_solve() -> None:
    r = g(operation="triangle_solve", params={"a": 3, "b": 4, "c": 5})
    assert abs(r["solutions"][0]["C"] - 90) < 1e-9 and abs(r["solutions"][0]["area"] - 6) < 1e-12
    r = g(operation="triangle_solve", params={"a": 6, "b": 8, "A": 30})
    assert len(r["solutions"]) == 2


def test_curve_points_helix_length() -> None:
    r = g(
        operation="curve_points",
        params={"curve": "helix", "radius": 10, "pitch": 2, "turns": 3, "n": 301},
    )
    expected = 3 * math.hypot(2 * math.pi * 10, 2)
    assert abs(r["length"] - expected) < 1e-9
    assert len(r["points"]) == 301


def test_curve_points_involute_starts_on_base_circle() -> None:
    r = g(
        operation="curve_points",
        params={"curve": "involute", "base_radius": 20, "n": 10, "max_angle_deg": 60},
    )
    p0 = r["points"][0]
    assert abs(math.hypot(p0[0], p0[1]) - 20) < 1e-9


@pytest.mark.parametrize(
    ("a", "b", "kind"),
    [
        (
            {"type": "line", "point": [-5, 0, 0], "dir": [1, 0, 0]},
            {"type": "sphere", "center": [0, 0, 0], "radius": 2},
            "points",
        ),
        (
            {"type": "plane", "point": [0, 0, 0], "normal": [0, 0, 1]},
            {"type": "plane", "point": [0, 0, 0], "normal": [1, 0, 0]},
            "line",
        ),
        (
            {"type": "sphere", "center": [0, 0, 0], "radius": 2},
            {"type": "sphere", "center": [2, 0, 0], "radius": 2},
            "circle",
        ),
        (
            {"type": "line", "point": [0, 0, 0], "dir": [1, 0, 0]},
            {"type": "line", "point": [0, 1, 1], "dir": [0, 1, 0]},
            "none",
        ),
    ],
)
def test_intersect(a: dict, b: dict, kind: str) -> None:
    r = g(operation="intersect", params={"a": a, "b": b})
    assert r["kind"] == kind


def test_skew_lines_distance() -> None:
    r = g(
        operation="intersect",
        params={
            "a": {"type": "line", "point": [0, 0, 0], "dir": [1, 0, 0]},
            "b": {"type": "line", "point": [0, 1, 1], "dir": [0, 1, 0]},
        },
    )
    assert abs(r["distance"] - 1) < 1e-12


def test_feature_points_box() -> None:
    r = g(operation="feature_points", params={"shape": "box", "size": [10, 20, 30]})
    assert (
        len(r["vertices"]) == 8 and len(r["edge_midpoints"]) == 12 and len(r["face_centers"]) == 6
    )
    tops = [f for f in r["face_centers"] if f["name"] == "top"]
    assert close(tops[0]["point"], [5, 10, 30])


def test_pattern_circular() -> None:
    r = g(operation="pattern", params={"kind": "circular", "count": 6, "radius": 10})
    assert len(r["points"]) == 6


def test_point_cloud_hull() -> None:
    pts = [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    r = g(operation="point_cloud", params={"points": pts})
    assert abs(r["hull_volume"] - 1) < 1e-12 and r["size"] == [1, 1, 1]


def test_mesh_properties_unit_cube() -> None:
    v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    f = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    r = g(operation="mesh_properties", params={"vertices": v, "faces": f})
    assert abs(r["volume"] - 1) < 1e-12 and abs(r["area"] - 6) < 1e-12
    assert r["watertight"] is True and close(r["centroid"], [0.5, 0.5, 0.5])


def test_composite_block_minus_hole() -> None:
    r = g(
        operation="composite_properties",
        params={
            "parts": [
                {"shape": "box", "size": [100, 50, 20], "position": [0, 0, 0], "op": "add"},
                {
                    "shape": "cylinder",
                    "radius": 5,
                    "height": 20,
                    "position": [50, 25, 0],
                    "op": "subtract",
                },
            ]
        },
    )
    assert abs(r["volume"] - (100 * 50 * 20 - math.pi * 25 * 20)) < 1e-9
    assert close(r["bbox_max"], [100, 50, 20])


def test_geometry_steps() -> None:
    r = g(operation="angle_between", v1=[1, 0, 0], v2=[1, 1, 0], show_steps=True)
    assert len(r["steps"]) >= 3
