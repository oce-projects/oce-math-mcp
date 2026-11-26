"""D107: eng_calc.magnetic_dipole (field and vector potential of a point dipole, moment of a current loop,
torque and energy in a field) and eng_calc.magnetic_circuit (magnetomotive force, reluctance, Hopkinson's law).

References: the dipole field written out by components here with numpy; B = curl A checked by central
differences on the tool's own A; the far field of a circular loop from magnetism.loop on its axis; the
moment of a spinning charged sphere q omega R^2 / 5 from math_solve.multi_integrate of (1/2) r x J; the area
of a regular polygon (n / 2) a^2 sin(2 pi / n); the closed core of magnetism.toroid.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc
from tests.conftest import ok

MU0 = 1.25663706127e-6  # N/A2, CODATA 2022
K = MU0 / (4 * math.pi)


def dip(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle({"operation": "magnetic_dipole", "params": params}, ctx or Ctx())


def circ(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle({"operation": "magnetic_circuit", "params": params}, ctx or Ctx())


def mag(params: dict) -> dict:
    return eng_calc.handle({"operation": "magnetism", "params": params}, Ctx())


def approx(v: Any, rel: float = 1e-12) -> Any:
    """Relative only: pytest's default abs = 1e-12 would accept any small SI value."""
    return pytest.approx(v, rel=rel, abs=0)


def units(ctx: Ctx) -> dict[str, str]:
    assert ctx.units is not None
    return ctx.units


def close(got: list[float], want: Any, rel: float = 1e-12) -> None:
    """Vectors: the error measured against the length of the expected vector (components may be ~0)."""
    w = np.asarray(want, dtype=float)
    assert len(got) == 3
    assert float(np.max(np.abs(np.asarray(got) - w))) <= rel * float(np.linalg.norm(w))


def bad(fn: Any, params: dict, words: str) -> None:
    with pytest.raises(ToolError) as e:
        fn(params)
    assert e.value.code == "INVALID_ARGUMENT"
    assert words in e.value.message


def ref_field(m: list[float], x: list[float], x0: list[float]) -> tuple[Any, Any, Any]:
    r = np.asarray(x, dtype=float) - np.asarray(x0, dtype=float)
    n = float(np.linalg.norm(r))
    mv = np.asarray(m, dtype=float)
    a = K * np.cross(mv, r) / n**3
    b = K * (3 * r * float(np.dot(mv, r)) / n**5 - mv / n**3)
    return r, a, b


# ------------------------------------------------------------------ magnetic_dipole: field


def test_field_on_the_axis_of_the_dipole() -> None:
    m0, z = 0.8, 0.05
    out = dip({"mode": "field", "m": [0, 0, m0], "x": [0, 0, z]})
    assert out["A"] == [0, 0, 0]
    assert out["B"][0] == 0 and out["B"][1] == 0
    assert out["B"][2] == approx(2 * K * m0 / z**3)
    assert out["B_magnitude"] == approx(2 * K * m0 / z**3)
    assert out["r"] == [0, 0, z] and out["r_magnitude"] == approx(z)


def test_field_in_the_equatorial_plane() -> None:
    m0, d = 1.5, 0.2
    out = dip({"mode": "field", "m": [0, 0, m0], "x": [d, 0, 0]})
    close(out["B"], [0, 0, -K * m0 / d**3])
    close(out["A"], [0, K * m0 / d**2, 0])


def test_field_by_components_away_from_the_origin() -> None:
    m, x, x0 = [0.3, -0.8, 1.1], [0.12, -0.05, 0.31], [0.02, 0.04, 0.01]
    ctx = Ctx()
    out = dip({"mode": "field", "m": m, "x": x, "x0": x0}, ctx)
    r, a, b = ref_field(m, x, x0)
    close(out["r"], r)
    assert out["r_magnitude"] == approx(float(np.linalg.norm(r)))
    close(out["A"], a)
    close(out["B"], b)
    assert out["B_magnitude"] == approx(float(np.linalg.norm(b)))
    assert units(ctx)["A"] == "T*m" and units(ctx)["B"] == "T" and units(ctx)["r"] == "m"


def test_b_is_the_curl_of_a() -> None:
    """The page's B = curl A: central differences of the tool's own A at a generic point."""
    m, x = [0.3, -0.8, 1.1], np.array([0.05, 0.02, -0.04])
    h = 1e-6

    def a_at(pt: Any) -> Any:
        return np.array(dip({"mode": "field", "m": m, "x": [float(v) for v in pt]})["A"])

    jac = np.zeros((3, 3))  # jac[i, j] = dA_i / dx_j
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        jac[:, j] = (a_at(x + e) - a_at(x - e)) / (2 * h)
    curl = [jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]]
    close(curl, dip({"mode": "field", "m": m, "x": [float(v) for v in x]})["B"], 1e-7)


def test_far_field_of_a_circular_loop_on_its_axis() -> None:
    cur, radius, z = 2.0, 0.01, 10.0
    loop = mag({"mode": "loop", "I": cur, "R": radius, "x": z})["B"]
    out = dip({"mode": "field", "m": [0, 0, cur * math.pi * radius**2], "x": [0, 0, z]})
    assert out["B"][2] == approx(loop, 2e-6)
    assert out["B"][2] != approx(
        loop, 1e-7
    )  # the loop field is not a dipole field at finite distance


def test_field_errors() -> None:
    bad(
        dip,
        {"mode": "field", "m": [1, 0, 0], "x": [0.1, 0.2, 0.3], "x0": [0.1, 0.2, 0.3]},
        "infinite",
    )
    bad(dip, {"mode": "field", "m": [1, 0], "x": [0, 0, 1]}, "params.m: 3 numbers")
    bad(dip, {"mode": "field", "m": [1, 0, 0], "x": [0, 0, float("inf")]}, "finite numbers")
    bad(dip, {"mode": "field", "m": [1, 0, 0], "x": [0, 0, 1], "I": 2}, "Keys not used")


# ------------------------------------------------------------------ magnetic_dipole: moment


def test_moment_of_a_flat_loop() -> None:
    ctx = Ctx()
    out = dip({"mode": "moment", "I": 2, "N": 10, "A": 0.01, "normal": [0, 0, 2]}, ctx)
    close(out["m"], [0, 0, 0.2])
    assert out["m_magnitude"] == approx(0.2)
    assert units(ctx)["m"] == "A*m2"
    out = dip({"mode": "moment", "I": -2, "A": [100, "cm2"]})
    assert out == {"m_magnitude": approx(0.02)}


def test_moment_of_a_square_follows_the_direction_of_the_current() -> None:
    a, cur = 0.02, 3.0
    square = [[0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0]]
    out = dip({"mode": "moment", "I": cur, "vertices": square})
    close(out["m"], [0, 0, cur * a * a])
    close(out["vector_area"], [0, 0, a * a])
    assert out["m_magnitude"] == approx(cur * a * a)
    back = dip({"mode": "moment", "I": cur, "vertices": square[::-1]})
    close(back["m"], [0, 0, -cur * a * a])
    moved = [[v[0] + 5, v[1] - 7, v[2] + 2] for v in square]
    close(dip({"mode": "moment", "I": cur, "vertices": moved})["m"], [0, 0, cur * a * a], 1e-10)


def test_moment_of_a_tilted_triangle_with_turns() -> None:
    pa, pb, pc = np.array([0.1, 0.0, 0.2]), np.array([0.0, 0.3, -0.1]), np.array([-0.2, 0.1, 0.4])
    out = dip(
        {"mode": "moment", "I": 1.5, "N": 5, "vertices": [pa.tolist(), pb.tolist(), pc.tolist()]}
    )
    area = 0.5 * np.cross(pb - pa, pc - pa)
    close(out["vector_area"], area)
    close(out["m"], 7.5 * area)


def test_polygon_of_many_sides_tends_to_the_circle() -> None:
    n, a, cur = 10000, 0.05, 2.0
    t = 2 * math.pi * np.arange(n) / n
    pts = np.stack([a * np.cos(t), a * np.sin(t), np.zeros(n)], axis=1).tolist()
    out = dip({"mode": "moment", "I": cur, "vertices": pts})
    assert out["m"][2] == approx(cur * n / 2 * a * a * math.sin(2 * math.pi / n), 1e-11)
    assert out["m"][2] == approx(cur * math.pi * a * a, 1e-6)


def test_volume_moment_of_a_spinning_charged_sphere_with_multi_integrate() -> None:
    """m = (1/2) int r x J dV with J = rho omega z x r (rho = omega = R = 1): m_z = 4 pi / 15 = q omega R^2 / 5."""
    bounds = [["r", 0, 1], ["theta", 0, "pi"], ["phi", 0, "2*pi"]]
    r = ok(
        "math_solve",
        operation="multi_integrate",
        expr="r**2*sin(theta)**2/2",
        bounds=bounds,
        coords="spherical",
    )
    q = 4 * math.pi / 3
    assert r["numeric"] == approx(q / 5) and r["verified"] is True
    x = ok(
        "math_solve",
        operation="multi_integrate",
        expr="-r**2*sin(theta)*cos(theta)*cos(phi)/2",
        bounds=bounds,
        coords="spherical",
    )
    assert x["numeric"] == 0


def test_moment_errors() -> None:
    sq = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    bad(dip, {"mode": "moment", "I": 1, "vertices": sq[:2]}, "3 to 10000")
    bad(dip, {"mode": "moment", "I": 1, "vertices": [[0, 0, 0]] * 10001}, "3 to 10000")
    bad(
        dip,
        {"mode": "moment", "I": 1, "vertices": [[0, 0], [1, 0, 0], [1, 1, 0]]},
        "params.vertices[0]",
    )
    bad(dip, {"mode": "moment", "I": 1, "A": 0.1, "vertices": sq}, "exactly one of these")
    bad(dip, {"mode": "moment", "I": 1}, "exactly one of these")
    bad(
        dip,
        {"mode": "moment", "I": 1, "vertices": sq, "normal": [0, 0, 1]},
        "params.normal goes with params.A",
    )
    bad(dip, {"mode": "moment", "I": 1, "A": 0.1, "normal": [0, 0, 0]}, "zero vector")
    bad(dip, {"mode": "moment", "I": 1, "A": 0.1, "N": 1.5}, "integer")


# ------------------------------------------------------------------ magnetic_dipole: torque


def test_torque_and_energy_of_a_dipole() -> None:
    ctx = Ctx()
    out = dip({"mode": "torque", "m": [2, 0, 0], "B": [0, 0.5, 0]}, ctx)
    close(out["tau"], [0, 0, 1])
    assert out["tau_magnitude"] == approx(1.0)
    assert out["U"] == 0 and out["angle_deg"] == approx(90.0)
    assert units(ctx)["tau"] == "N*m" and units(ctx)["U"] == "J"
    par = dip({"mode": "torque", "m": [0, 0, 3], "B": [0, 0, 0.2]})
    assert par["tau"] == [0, 0, 0] and par["U"] == approx(-0.6) and par["angle_deg"] == 0
    anti = dip({"mode": "torque", "m": [0, 0, -3], "B": [0, 0, 0.2]})
    assert anti["U"] == approx(0.6) and anti["angle_deg"] == approx(180.0)
    assert "angle_deg" not in dip({"mode": "torque", "m": [0, 0, 0], "B": [0, 0, 0.2]})


def test_torque_by_components() -> None:
    m, b = [0.4, -1.2, 0.7], [0.03, 0.05, -0.02]
    out = dip({"mode": "torque", "m": m, "B": b})
    close(out["tau"], np.cross(m, b))
    assert out["U"] == approx(-float(np.dot(m, b)))
    cos = float(np.dot(m, b) / (np.linalg.norm(m) * np.linalg.norm(b)))
    assert out["angle_deg"] == approx(math.degrees(math.acos(cos)), 1e-10)


def test_torque_is_minus_the_derivative_of_the_energy() -> None:
    m0, b0, th, h = 2.5, 0.4, 0.7, 1e-6

    def at(angle: float) -> dict:
        return dip(
            {
                "mode": "torque",
                "m": [m0 * math.cos(angle), m0 * math.sin(angle), 0],
                "B": [b0, 0, 0],
            }
        )

    du = (at(th + h)["U"] - at(th - h)["U"]) / (2 * h)
    assert at(th)["tau"][2] == approx(-du, 1e-8)
    assert at(th)["U"] == approx(-m0 * b0 * math.cos(th))


def test_torque_of_pieces_in_their_own_fields() -> None:
    out = dip({"mode": "torque", "m": [[1, 0, 0], [0, 1, 0]], "B": [[2, 0, 1], [0, 3, 2]]})
    close(out["tau"], [2, -1, 0])
    assert out["U"] == approx(-5.0)
    assert "angle_deg" not in out
    pair = dip({"mode": "torque", "m": [[1, 0, 0], [-1, 0, 0]], "B": [[0, 1, 0], [0, 1, 0]]})
    assert pair["tau"] == [0, 0, 0] and pair["U"] == 0


def test_torque_errors() -> None:
    bad(dip, {"mode": "torque", "m": [[1, 0, 0], [0, 1, 0]], "B": [[0, 0, 1]]}, "same number")
    bad(dip, {"mode": "torque", "m": [[1, 0, 0]], "B": [0, 0, 1]}, "params.B[0]")
    bad(dip, {"mode": "torque", "m": [1, 0, 0], "B": [[0, 0, 1]]}, "params.B: 3 numbers")
    bad(dip, {"mode": "torque", "m": [], "B": []}, "params.m: 3 numbers")


# ------------------------------------------------------------------ magnetic_circuit


CORE = [0.3, 1e-4, 2000.0]  # l = 30 cm, A = 1 cm2, mu_r = 2000
GAP = [0.001, 1e-4, 1.0]  # 1 mm of air


def test_closed_core_is_the_toroid() -> None:
    ctx = Ctx()
    out = circ({"N": 500, "I": 0.2, "segments": [CORE]}, ctx)
    rel = 0.3 / (MU0 * 2000 * 1e-4)
    assert out["R"] == approx(rel) and out["R_segments"] == [approx(rel)]
    assert out["F"] == approx(100.0)
    assert out["Phi"] == approx(100.0 / rel)
    assert out["I"] == approx(0.2)
    tor = mag({"mode": "toroid", "N": 500, "I": 0.2, "r": 0.3 / (2 * math.pi), "mu_r": 2000})["B"]
    assert out["B_segments"] == [approx(tor)]
    assert out["H_segments"] == [approx(tor / (MU0 * 2000))]
    assert out["drop_segments"] == [approx(100.0)]
    assert units(ctx)["R"] == "1/H" and units(ctx)["Phi"] == "Wb" and units(ctx)["F"] == "A"


def test_air_gap_in_series_takes_most_of_the_drop() -> None:
    out = circ({"N": 500, "I": 0.2, "segments": [CORE, GAP]})
    r_core, r_gap = 0.3 / (MU0 * 2000 * 1e-4), 0.001 / (MU0 * 1e-4)
    assert out["R_segments"] == [approx(r_core), approx(r_gap)]
    assert out["R"] == approx(r_core + r_gap)
    phi = 100.0 / (r_core + r_gap)
    assert out["Phi"] == approx(phi)
    assert out["B_segments"] == [approx(phi / 1e-4), approx(phi / 1e-4)]
    assert out["H_segments"][1] == approx(phi / 1e-4 / MU0)
    assert sum(out["drop_segments"]) == approx(100.0)
    assert out["drop_segments"][1] == approx(100.0 * r_gap / (r_core + r_gap))


def test_segments_of_different_area() -> None:
    out = circ({"F": 50, "segments": [[0.2, 1e-4, 1000], [0.1, 2e-4, 1000]]})
    assert out["B_segments"][0] == approx(2 * out["B_segments"][1])
    assert "I" not in out


def test_flux_given_gives_the_current() -> None:
    phi = circ({"N": 500, "I": 0.2, "segments": [CORE, GAP]})["Phi"]
    back = circ({"N": 500, "Phi": phi, "segments": [CORE, GAP]})
    assert back["I"] == approx(0.2) and back["F"] == approx(100.0)


def test_reluctance_and_force_with_units() -> None:
    out = circ({"F": [100, "A"], "R": [2, "MA/Wb"]})
    assert out["Phi"] == approx(5e-5) and out["R"] == approx(2e6)
    assert "R_segments" not in out
    out = circ({"F": 120, "N": 300, "R": [4e5, "1/H"]})
    assert out["I"] == approx(0.4) and out["Phi"] == approx(3e-4)


def test_circuit_errors() -> None:
    bad(circ, {"I": 0.2, "segments": [CORE]}, "params.I needs params.N")
    bad(circ, {"N": 5, "I": 0.2, "R": 1e6, "segments": [CORE]}, "exactly one of these")
    bad(circ, {"N": 5, "R": 1e6}, "exactly one of these")
    bad(circ, {"N": 5, "I": 0.2, "Phi": 1e-4, "R": 1e6}, "exactly one of these")
    bad(circ, {"F": 1, "segments": [[0.3, 1e-4, 0]]}, "params.segments[0]")
    bad(circ, {"F": 1, "segments": [[0.3, 1e-4]]}, "params.segments[0]")
    bad(circ, {"F": 1, "segments": []}, "1 to 1000")
    bad(circ, {"F": 1, "R": 1e6, "mode": "x"}, "Unexpected keys")
