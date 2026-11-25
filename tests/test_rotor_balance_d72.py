"""Unbalance and balancing of rotors (D72).

Checks independent of the server formulas: the centrifugal force of every mass as a 3D vector at the speed omega,
m omega^2 (r cos(theta), r sin(theta), 0) at the point (r cos(theta), r sin(theta), z), summed with numpy; the resultant
force and the resultant moment about the origin (cross products). After the two-plane correction both must vanish
(dynamic balance: no resultant centrifugal force or couple).
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import call, ok


def bal(**params: object) -> dict:
    return ok("eng_calc", operation="rotor_balance", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="rotor_balance", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def _resultants(masses: list[dict], omega: float) -> tuple[np.ndarray, np.ndarray]:
    force, moment = np.zeros(3), np.zeros(3)
    for item in masses:
        t = math.radians(item["angle_deg"])
        pos = np.array([item["r"] * math.cos(t), item["r"] * math.sin(t), item.get("z", 0.0)])
        f = item["m"] * omega**2 * np.array([pos[0], pos[1], 0.0])
        force += f
        moment += np.cross(pos, f)
    return force, moment


MASSES = [
    {"m": 0.02, "r": 0.1, "angle_deg": 30.0, "z": 0.05},
    {"m": 0.015, "r": 0.08, "angle_deg": 200.0, "z": 0.3},
    {"m": 0.01, "r": 0.12, "angle_deg": 110.0, "z": 0.45},
]


def test_two_plane_balancing_against_vector_resultants() -> None:
    n = 3000.0
    omega = n * 2 * math.pi / 60
    r = bal(mode="unbalance", masses=MASSES, M=20.0, n_rpm=n, planes=[0.0, 0.5], r_correction=0.1)
    force, moment = _resultants(MASSES, omega)
    assert math.isclose(r["F"], float(np.linalg.norm(force)), rel_tol=1e-12)
    assert math.isclose(r["U"] * omega**2, float(np.linalg.norm(force)), rel_tol=1e-12)
    assert math.isclose(r["M_couple"], float(np.linalg.norm(moment)), rel_tol=1e-12)
    assert math.isclose(r["C"] * omega**2, float(np.linalg.norm(moment)), rel_tol=1e-12)
    assert math.isclose(
        math.radians(r["U_angle_deg"]),
        math.atan2(force[1], force[0]) % (2 * math.pi),
        rel_tol=1e-12,
    )
    assert math.isclose(r["e"], r["U"] / 20.0, rel_tol=1e-13)
    corrected = [
        *MASSES,
        {"m": r["m_L"], "r": 0.1, "angle_deg": r["U_L_angle_deg"], "z": 0.0},
        {"m": r["m_R"], "r": 0.1, "angle_deg": r["U_R_angle_deg"], "z": 0.5},
    ]
    force, moment = _resultants(corrected, omega)
    scale = r["F"] * 0.5
    assert np.linalg.norm(force) < 1e-12 * scale and np.linalg.norm(moment) < 1e-12 * scale, (
        force,
        moment,
    )
    # speed given as an angular velocity: same result
    w = bal(mode="unbalance", masses=MASSES, omega=omega)
    assert math.isclose(w["F"], r["F"], rel_tol=1e-13)


def test_static_and_couple_unbalance() -> None:
    # page: U = m r (static), U = m r d (couple of two equal unbalances 180 degrees apart): statically balanced
    pair = [
        {"m": 0.01, "r": 0.1, "angle_deg": 0.0, "z": 0.0},
        {"m": 0.01, "r": 0.1, "angle_deg": 180.0, "z": 0.4},
    ]
    r = bal(mode="unbalance", masses=pair)
    assert r["U"] < 1e-15 and math.isclose(r["C"], 0.01 * 0.1 * 0.4, rel_tol=1e-12)
    one = bal(mode="unbalance", masses=[{"m": 0.05, "r": 0.2, "angle_deg": 75.0}])
    assert math.isclose(one["U"], 0.05 * 0.2, rel_tol=1e-13) and math.isclose(
        one["U_angle_deg"], 75.0, rel_tol=1e-12
    )
    # single-plane correction: opposite to the unbalance
    assert math.isclose(one["U_correction"], one["U"], rel_tol=1e-13)
    assert math.isclose(one["U_correction_angle_deg"], 255.0, rel_tol=1e-12)
    m_one = bal(
        mode="unbalance", masses=[{"m": 0.05, "r": 0.2, "angle_deg": 75.0}], r_correction=0.1
    )
    assert math.isclose(m_one["m_correction"], 0.05 * 0.2 / 0.1, rel_tol=1e-13)
    assert "planes" in bad({"mode": "unbalance", "masses": MASSES, "planes": [0.2, 0.2]})
    assert "masses" in bad({"mode": "unbalance", "masses": []})
    assert "> 0" in bad({"mode": "unbalance", "masses": [{"m": -1.0, "r": 0.1, "angle_deg": 0.0}]})


def test_iso_balance_grade() -> None:
    r = bal(mode="iso_grade", G=2.5, M=100.0, n_rpm=3000.0)
    omega = 3000.0 * 2 * math.pi / 60
    assert math.isclose(r["e_per"] * omega, 2.5e-3, rel_tol=1e-13)  # e_per omega = G (mm/s)
    assert math.isclose(r["U_per"], 100.0 * r["e_per"], rel_tol=1e-13)
    # page: U_per = 9.54 G M / rpm, in kg*mm with M in kg (9.54 = 60 / (2 pi) rounded)
    page = 9.54 * 2.5 * 100.0 / 3000.0
    assert abs(r["U_per"] * 1e3 - page) / page < 1e-3
    assert math.isclose(r["U_per_g_mm"], r["U_per"] * 1e6, rel_tol=1e-13)
    w = bal(mode="iso_grade", G=2.5, M=100.0, omega=omega)
    assert math.isclose(w["U_per"], r["U_per"], rel_tol=1e-13)
