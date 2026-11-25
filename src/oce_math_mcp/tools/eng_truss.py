"""eng_calc: 2D and 3D pin-jointed trusses with the direct stiffness method (D59).

Source: Wikipedia "Direct stiffness method". Member matrix in global coordinates
k = (E A / L) [c c^T, -c c^T; -c c^T, c c^T] with c the direction cosines (the same pattern in 3D, as the
source says); assembly into the system matrix K; r = K^-1 R on the free degrees of freedom.
Member force N = (E A / L) c . (u_j - u_i), positive in tension. Degree of static indeterminacy from the
counting rule m + r - d n (d = 2 or 3; page "Truss": m >= 2j - r is necessary, not sufficient),
confirmed by a non-singular matrix.
Frames (members with bending) are not included: the source writes the 6 x 6 matrix only with symbols (D59).
Nodes are numbered from 1, as in textbooks.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import fmt_sig, need, only, op, units_for

_KEYS = {"nodes", "members", "supports", "loads", "E", "A"}
_UNITS = {
    "displacements": "m",
    "forces": "N",
    "stresses": "Pa",
    "reactions": "N",
    "lengths": "m",
    "strain_energy": "J",
}


def _rows(raw: Any, key: str, sizes: tuple[int, ...], example: str) -> list[list[float]]:
    if not isinstance(raw, list):
        raise ToolError("INVALID_ARGUMENT", f"params.{key}: list of rows {example}.")
    out = []
    for k, row in enumerate(raw, 1):
        if (
            not isinstance(row, list)
            or len(row) not in sizes
            or any(isinstance(x, bool) and key != "supports" for x in row)
            or any(not isinstance(x, int | float) or not math.isfinite(float(x)) for x in row)
        ):
            raise ToolError("INVALID_ARGUMENT", f"params.{key}, row {k}: {example} is needed.")
        out.append([float(x) for x in row])
    return out


def _node(idx: float, n: int, what: str) -> int:
    if not float(idx).is_integer() or not 1 <= idx <= n:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"{what}: node {fmt_sig(idx, 6)} does not exist (nodes are numbered from 1 to {n}).",
        )
    return int(idx) - 1


def _per_member(p: dict[str, Any], key: str, kind: str, m: int) -> list[float]:
    """One value for all members (number or [value, "unit"]) or a list with one value per member."""
    raw = p.get(key)
    with_unit = isinstance(raw, list) and len(raw) == 2 and isinstance(raw[1], str)
    if not isinstance(raw, list) or with_unit:
        return [need(p, key, kind)] * m
    if len(raw) != m:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: one value for all members or a list of {m} values.",
        )
    return [need({f"{key}[{k + 1}]": x}, f"{key}[{k + 1}]", kind) for k, x in enumerate(raw)]


@op("truss", tuple(sorted(_KEYS)))
def _truss(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    only(p, "truss", _KEYS, field="operation")
    nodes = _rows(p.get("nodes"), "nodes", (2, 3), "[x, y] or [x, y, z] in m")
    if not nodes:
        raise ToolError("INVALID_ARGUMENT", "params.nodes: at least 2 nodes.")
    dim = len(nodes[0])
    if any(len(r) != dim for r in nodes):
        raise ToolError("INVALID_ARGUMENT", "params.nodes: all nodes in 2D or all in 3D.")
    n = len(nodes)
    members = _rows(p.get("members"), "members", (2,), "[node i, node j]")
    if not members:
        raise ToolError("INVALID_ARGUMENT", "params.members: at least one member.")
    m = len(members)
    pairs = [
        (_node(i, n, f"Member {k}"), _node(j, n, f"Member {k}"))
        for k, (i, j) in enumerate(members, 1)
    ]
    e_mod = _per_member(p, "E", "pressure", m)
    area = _per_member(p, "A", "area", m)
    fix_names = ", ".join(["fixed_x", "fixed_y", "fixed_z"][:dim])
    load_names = ", ".join(["Fx", "Fy", "Fz"][:dim])
    supports = _rows(p.get("supports"), "supports", (1 + dim,), f"[node, {fix_names}] with 1 or 0")
    raw_loads = p.get("loads")
    loads = _rows(
        raw_loads if raw_loads is not None else [],
        "loads",
        (1 + dim,),
        f"[node, {load_names}] in N",
    )
    coords = np.array(nodes, dtype=float)
    ndof = dim * n
    kmat = np.zeros((ndof, ndof))
    dirs, lengths = [], []
    for k, (i, j) in enumerate(pairs):
        d = coords[j] - coords[i]
        length = float(np.linalg.norm(d))
        if length == 0:
            raise ToolError("INVALID_ARGUMENT", f"Member {k + 1}: coincident nodes, zero length.")
        c = d / length
        ke = e_mod[k] * area[k] / length * np.outer(c, c)
        si, sj = slice(dim * i, dim * i + dim), slice(dim * j, dim * j + dim)
        kmat[si, si] += ke
        kmat[sj, sj] += ke
        kmat[si, sj] -= ke
        kmat[sj, si] -= ke
        dirs.append(c)
        lengths.append(length)
    fixed = np.zeros(ndof, dtype=bool)
    for row in supports:
        nd = _node(row[0], n, "Support")
        for a in range(dim):
            if row[1 + a] not in (0.0, 1.0):
                raise ToolError(
                    "INVALID_ARGUMENT", "params.supports: 1 = fixed direction, 0 = free."
                )
            fixed[dim * nd + a] |= row[1 + a] == 1.0
    force = np.zeros(ndof)
    for row in loads:
        nd = _node(row[0], n, "Load")
        force[dim * nd : dim * nd + dim] += row[1:]
    free = ~fixed
    kff = kmat[np.ix_(free, free)]
    scale = float(np.max(np.abs(np.diag(kmat)))) if ndof else 1.0
    if kff.size and (np.linalg.matrix_rank(kff, tol=1e-10 * scale) < kff.shape[0]):
        raise ToolError(
            "INVALID_ARGUMENT",
            "Unstable structure (mechanism): the stiffness matrix is singular. Add members or supports.",
        )
    u = np.zeros(ndof)
    if kff.size:
        u[free] = np.linalg.solve(kff, force[free])
    reac = kmat @ u - force
    forces, stresses = [], []
    for k, (i, j) in enumerate(pairs):
        du = u[dim * j : dim * j + dim] - u[dim * i : dim * i + dim]
        nk = e_mod[k] * area[k] / lengths[k] * float(dirs[k] @ du)
        forces.append(nk)
        stresses.append(nk / area[k])
    sup_nodes = sorted({_node(r[0], n, "Support") for r in supports})
    reactions = [
        [nd + 1, *(float(reac[dim * nd + a]) if fixed[dim * nd + a] else 0.0 for a in range(dim))]
        for nd in sup_nodes
    ]
    degree = m + int(fixed.sum()) - ndof
    energy = 0.5 * float(u @ kmat @ u)
    out: dict[str, Any] = {
        "displacements": [[float(x) for x in u[dim * a : dim * a + dim]] for a in range(n)],
        "forces": forces,
        "stresses": stresses,
        "reactions": reactions,
        "lengths": lengths,
        "strain_energy": energy,
        "degree": degree,
        "determinacy": "determinate" if degree == 0 else "indeterminate",
    }
    ctx.formula = (
        "k_member = (E A / L) [c c^T, -c c^T; -c c^T, c c^T]; K u = F on the free degrees of freedom; "
        "N = (E A / L) c . (u_j - u_i), positive in tension; reactions = K u - F"
    )
    ctx.warn(
        "Members pinned at both ends, loads only at the nodes, small displacements, linear elastic material."
    )
    k_max = int(np.argmax(np.abs(forces)))
    ctx.formatted = (
        f"Truss {out['determinacy']} (degree {degree}); most loaded member {k_max + 1}: "
        f"N = {fmt_sig(forces[k_max], 6)} N ({'tension' if forces[k_max] > 0 else 'compression'})."
    )
    ctx.units = units_for(out, _UNITS)
    return out
