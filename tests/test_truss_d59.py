"""2D and 3D trusses with the direct stiffness method (D59).

Source: Wikipedia "Direct stiffness method" (member matrix EA/L [c c^T, -c c^T; -c c^T, c c^T] with the
direction cosines, assembly, r = K^-1 (R - R^o)). Checks independent of the stiffness
method:
- determinate truss against the method of joints (equilibrium of every joint, linear system in the test);
- displacement of a node against the principle of virtual work (unit load: sum N n L / (E A));
- strain energy sum N^2 L / (2 E A) equal to the work of the loads (1/2) sum F u (Clapeyron);
- indeterminate truss: global equilibrium of forces and moments with the reactions;
- 3D tripod against the equilibrium of the top node.
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import call, ok


def truss(**params: object) -> dict:
    return ok("eng_calc", operation="truss", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="truss", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


# Pratt truss with 4 bays of 3 m, 3 m high: nodes 1-5 at the bottom, 6-8 at the top
NODES = [[0, 0], [3, 0], [6, 0], [9, 0], [12, 0], [3, 3], [6, 3], [9, 3]]
MEMBERS = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],  # bottom chord
    [6, 7],
    [7, 8],  # top chord
    [1, 6],
    [8, 5],  # end struts
    [2, 6],
    [3, 7],
    [4, 8],  # verticals
    [6, 3],
    [8, 3],  # diagonals
]
SUPPORTS = [[1, 1, 1], [5, 0, 1]]
LOADS = [[2, 0, -10000], [3, 0, -20000], [4, 0, -10000], [7, 5000, 0]]
E, A = 210e9, 0.002


def joints_method(
    nodes: list, members: list, supports: list, loads: list
) -> tuple[np.ndarray, np.ndarray]:
    """Member forces (tension +) and reactions from the equilibrium of the joints, without stiffness."""
    n = len(nodes)
    react = [(s[0] - 1, d) for s in supports for d in range(2) if s[1 + d]]
    m = len(members)
    mat = np.zeros((2 * n, m + len(react)))
    rhs = np.zeros(2 * n)
    for k, (i, j) in enumerate(members):
        pi, pj = np.array(nodes[i - 1], float), np.array(nodes[j - 1], float)
        u = (pj - pi) / np.linalg.norm(pj - pi)
        mat[2 * (i - 1) : 2 * (i - 1) + 2, k] += u  # tension pulls node i towards j
        mat[2 * (j - 1) : 2 * (j - 1) + 2, k] -= u
    for k, (node, d) in enumerate(react):
        mat[2 * node + d, m + k] = 1.0
    for node, fx, fy in loads:
        rhs[2 * (node - 1)] -= fx
        rhs[2 * (node - 1) + 1] -= fy
    sol = np.linalg.solve(mat, rhs)
    return sol[:m], sol[m:]


def test_determinate_truss_against_method_of_joints() -> None:
    r = truss(nodes=NODES, members=MEMBERS, supports=SUPPORTS, loads=LOADS, E=E, A=A)
    assert r["determinacy"] == "determinate"
    forces, reacts = joints_method(NODES, MEMBERS, SUPPORTS, LOADS)
    assert np.allclose(r["forces"], forces, rtol=1e-9, atol=1e-6)
    # rows [node, Rx, Ry]: zero in the free directions (node 5 free in x)
    got = [x for row in r["reactions"] for x in row[1:]]
    assert np.allclose(got, [reacts[0], reacts[1], 0.0, reacts[2]], rtol=1e-9, atol=1e-6)
    assert np.allclose(r["stresses"], np.array(forces) / A, rtol=1e-9, atol=1e-3)


def test_virtual_work_and_clapeyron() -> None:
    r = truss(nodes=NODES, members=MEMBERS, supports=SUPPORTS, loads=LOADS, E=E, A=A)
    forces, _ = joints_method(NODES, MEMBERS, SUPPORTS, LOADS)
    unit, _ = joints_method(
        NODES, MEMBERS, SUPPORTS, [[3, 0, -1.0]]
    )  # unit load downwards at node 3
    lengths = [math.dist(NODES[i - 1], NODES[j - 1]) for i, j in MEMBERS]
    delta = sum(n_ * u_ * L / (E * A) for n_, u_, L in zip(forces, unit, lengths, strict=True))
    assert rel(
        -r["displacements"][2][1], delta, 1e-9
    )  # vertical displacement of node 3 (downwards)
    energy = sum(n_**2 * L / (2 * E * A) for n_, L in zip(forces, lengths, strict=True))
    work = 0.5 * sum(
        fx * r["displacements"][nd - 1][0] + fy * r["displacements"][nd - 1][1]
        for nd, fx, fy in LOADS
    )
    assert rel(r["strain_energy"], energy, 1e-9) and rel(work, energy, 1e-9)


def test_indeterminate_truss_equilibrium() -> None:
    members = [*MEMBERS, [2, 7], [7, 4]]  # two more diagonals: indeterminate of degree 2
    r = truss(
        nodes=NODES, members=members, supports=SUPPORTS, loads=LOADS, E=E, A=[A] * len(members)
    )
    assert r["determinacy"] == "indeterminate" and r["degree"] == 2
    rx = sum(row[1] for row in r["reactions"])
    ry = sum(row[2] for row in r["reactions"])
    assert abs(rx + 5000) < 1e-6 and abs(ry - 40000) < 1e-6
    moment = sum(
        (NODES[row[0] - 1][0] * row[2] - NODES[row[0] - 1][1] * row[1]) for row in r["reactions"]
    )
    moment += sum(NODES[nd - 1][0] * fy - NODES[nd - 1][1] * fx for nd, fx, fy in LOADS)
    assert abs(moment) < 1e-6


def test_tripod_3d() -> None:
    nodes = [[0, 0, 4], [2, 0, 0], [-1, 1.7320508075688772, 0], [-1, -1.7320508075688772, 0]]
    members = [[1, 2], [1, 3], [1, 4]]
    supports = [[2, 1, 1, 1], [3, 1, 1, 1], [4, 1, 1, 1]]
    r = truss(
        nodes=nodes, members=members, supports=supports, loads=[[1, 0, 0, -30000]], E=70e9, A=5e-4
    )
    # equilibrium of the top node: sum of the member forces along their directions = load
    total = np.zeros(3)
    for k, (i, j) in enumerate(members):
        u = np.array(nodes[j - 1], float) - np.array(nodes[i - 1], float)
        total += r["forces"][k] * u / np.linalg.norm(u)
    assert np.allclose(total, [0, 0, 30000], atol=1e-6)
    assert (
        all(rel(f, r["forces"][0], 1e-12) for f in r["forces"]) and r["forces"][0] < 0
    )  # compressed, equal


def test_units_and_per_member_lists() -> None:
    base = truss(nodes=NODES, members=MEMBERS, supports=SUPPORTS, loads=LOADS, E=E, A=A)
    units = truss(
        nodes=NODES,
        members=MEMBERS,
        supports=SUPPORTS,
        loads=LOADS,
        E=[210, "GPa"],
        A=[2000, "mm2"],
    )
    lists = truss(
        nodes=NODES,
        members=MEMBERS,
        supports=SUPPORTS,
        loads=LOADS,
        E=[E] * 13,
        A=[[20, "cm2"]] * 13,
    )
    for r in (units, lists):
        assert np.allclose(r["forces"], base["forces"], rtol=1e-12, atol=1e-9)
        assert np.allclose(r["displacements"], base["displacements"], rtol=1e-12, atol=1e-15)
    # determinate: forces do not depend on stiffness, displacements do (double area in member 1)
    stiff = truss(
        nodes=NODES, members=MEMBERS, supports=SUPPORTS, loads=LOADS, E=E, A=[2 * A] + [A] * 12
    )
    assert np.allclose(stiff["forces"], base["forces"], rtol=1e-12, atol=1e-9)
    assert rel(
        stiff["displacements"][1][0], base["displacements"][1][0] / 2, 1e-12
    )  # node 2 stretches less


def test_errors() -> None:
    assert "Unstable" in bad(
        {
            "nodes": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "members": [[1, 2], [3, 4], [1, 3], [2, 4]],
            "supports": [[1, 1, 1], [2, 0, 1]],
            "loads": [[3, 100, 0]],
            "E": E,
            "A": A,
        }
    )
    assert "does not exist" in bad(
        {
            "nodes": [[0, 0], [1, 0]],
            "members": [[1, 3]],
            "supports": [[1, 1, 1]],
            "loads": [],
            "E": E,
            "A": A,
        }
    )
    assert "zero length" in bad(
        {
            "nodes": [[0, 0], [0, 0]],
            "members": [[1, 2]],
            "supports": [[1, 1, 1], [2, 1, 1]],
            "loads": [],
            "E": E,
            "A": A,
        }
    )
