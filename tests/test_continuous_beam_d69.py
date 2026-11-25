"""Continuous beams with the theorem of three moments (D69).

Check independent of the server formulas: the stiffness method with Euler-Bernoulli beam elements (cubic Hermite,
nodes at the supports and under the point loads, consistent nodal loads for the uniform load: nodal values exact),
supports as prescribed displacements (settlements) with free or fixed rotation. Reactions from K d - F; moments by
statics of the free body on the left of each section, sagging positive.
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import call, ok


def beam(**params: object) -> dict:
    return ok("eng_calc", operation="continuous_beam", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="continuous_beam", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def _fem(spans: list[dict], ei_all: float, settle: list[float], left: str, right: str):
    """Returns (support positions, reactions [N, up +], moment function M(x) sagging +)."""
    supports = np.cumsum([0.0] + [s["L"] for s in spans])
    nodes = set(supports.tolist())
    point = []  # (x, P down +)
    for k, s in enumerate(spans):
        for force, a in s.get("P", []):
            nodes.add(float(supports[k] + a))
            point.append((float(supports[k] + a), force))
    xs = sorted(nodes)
    nd = len(xs)
    kg = np.zeros((2 * nd, 2 * nd))
    fg = np.zeros(2 * nd)

    def span_of(x0: float) -> dict:
        k = int(np.searchsorted(supports, x0, side="right") - 1)
        return spans[min(k, len(spans) - 1)]

    for i in range(nd - 1):
        x0, x1 = xs[i], xs[i + 1]
        le = x1 - x0
        s = span_of(x0)
        ei = s.get("EI", ei_all)
        w = s.get("w", 0.0)
        ke = (
            ei
            / le**3
            * np.array(
                [
                    [12, 6 * le, -12, 6 * le],
                    [6 * le, 4 * le**2, -6 * le, 2 * le**2],
                    [-12, -6 * le, 12, -6 * le],
                    [6 * le, 2 * le**2, -6 * le, 4 * le**2],
                ]
            )
        )
        fe = -w * np.array(
            [le / 2, le**2 / 12, le / 2, -(le**2) / 12]
        )  # w down +, dofs (v up, theta ccw)
        dofs = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
        kg[np.ix_(dofs, dofs)] += ke
        fg[dofs] += fe
    for x, force in point:
        fg[2 * xs.index(x)] -= force
    fixed = {}
    for j, xsup in enumerate(supports.tolist()):
        fixed[2 * xs.index(xsup)] = -settle[j]  # settlement down + = v negative
    if left == "fixed":
        fixed[1] = 0.0
    if right == "fixed":
        fixed[2 * nd - 1] = 0.0
    free = [d for d in range(2 * nd) if d not in fixed]
    dvec = np.zeros(2 * nd)
    for dof, val in fixed.items():
        dvec[dof] = val
    dvec[free] = np.linalg.solve(
        kg[np.ix_(free, free)], fg[free] - kg[np.ix_(free, list(fixed))] @ dvec[list(fixed)]
    )
    react = kg @ dvec - fg  # forces of the supports on the beam, at every dof
    r_vert = [float(react[2 * xs.index(x)]) for x in supports.tolist()]
    m_left = float(react[1]) if left == "fixed" else 0.0  # reaction moment at x = 0, ccw +

    def moment(x: float) -> float:
        total = -m_left
        for xsup, r in zip(supports.tolist(), r_vert, strict=True):
            if xsup < x:
                total += r * (x - xsup)
        for xp, force in point:
            if xp < x:
                total -= force * (x - xp)
        for k, s in enumerate(spans):
            a0, a1 = supports[k], min(supports[k + 1], x)
            if a1 > a0:
                total -= s.get("w", 0.0) * (a1 - a0) * (x - (a0 + a1) / 2)
        return total

    return supports, r_vert, moment


CASES = [
    ([{"L": 4.0, "w": 10e3}, {"L": 4.0, "w": 10e3}], [0, 0, 0], "pinned", "pinned"),
    (
        [
            {"L": 3.0, "w": 5e3},
            {"L": 5.0, "P": [[20e3, 2.0]]},
            {"L": 4.0, "w": 8e3, "P": [[5e3, 1.0], [7e3, 3.2]]},
        ],
        [0, 0, 0, 0],
        "pinned",
        "pinned",
    ),
    (
        [{"L": 6.0, "w": 12e3, "EI": 4e7}, {"L": 4.0, "P": [[30e3, 1.5]], "EI": 2e7}],
        [0, 0.004, 0.001],
        "pinned",
        "pinned",
    ),
    (
        [{"L": 5.0, "w": 9e3}, {"L": 5.0, "w": 9e3}, {"L": 3.0, "P": [[15e3, 1.0]]}],
        [0, 0, 0, 0],
        "fixed",
        "pinned",
    ),
    ([{"L": 4.0, "P": [[10e3, 1.0]]}], [0, 0], "fixed", "fixed"),
    ([{"L": 5.0, "w": 6e3}, {"L": 6.0, "w": 6e3}], [0.002, 0, 0.003], "fixed", "fixed"),
]


def test_three_moments_against_stiffness_method() -> None:
    ei = 3e7
    for spans, settle, left, right in CASES:
        r = beam(spans=spans, EI=ei, settlements=settle, left=left, right=right)
        supports, r_vert, moment = _fem(spans, ei, settle, left, right)
        scale = max(abs(moment(x)) for x in np.linspace(0, supports[-1], 2001)) or 1.0
        for j, xsup in enumerate(supports.tolist()):
            m_fem = moment(xsup + 1e-12) if j == 0 else moment(xsup)
            assert abs(r["M_supports"][j] - m_fem) < 1e-8 * scale, (
                j,
                r["M_supports"],
                m_fem,
                spans,
            )
            assert abs(r["reactions"][j] - r_vert[j]) < 1e-8 * max(map(abs, r_vert)), (
                j,
                r["reactions"],
                r_vert,
            )
        for k, (m_max, x_at) in enumerate(zip(r["M_span_max"], r["x_span_max"], strict=True)):
            xs = np.linspace(supports[k], supports[k + 1], 4001)
            assert (
                abs(m_max - max(moment(float(x)) for x in xs)) < 1e-5 * scale
            )  # grid of 4000 cells
            assert abs(moment(supports[k] + x_at) - m_max) < 1e-8 * scale


def test_settlement_sign_and_classic_results() -> None:
    # page uniform form for two equal spans: M_B = -w L^2 / 8 (hogging)
    r = beam(spans=[{"L": 4.0, "w": 10e3}, {"L": 4.0, "w": 10e3}], EI=3e7)
    assert (
        abs(r["M_supports"][1] + 10e3 * 16 / 8) < 1e-8
        and abs(r["reactions"][1] - 1.25 * 10e3 * 4) < 1e-6
    )
    # a middle support that settles relieves the hogging moment over it (stiffness method, case above)
    s = beam(
        spans=[{"L": 4.0, "w": 10e3}, {"L": 4.0, "w": 10e3}], EI=3e7, settlements=[0, 0.002, 0]
    )
    assert s["M_supports"][1] > r["M_supports"][1]
    # fixed-fixed beam with a central point load: P L / 8 at the ends and at the centre
    f = beam(spans=[{"L": 4.0, "P": [[10e3, 2.0]]}], EI=3e7, left="fixed", right="fixed")
    assert (
        abs(f["M_supports"][0] + 10e3 * 4 / 8) < 1e-8
        and abs(f["M_span_max"][0] - 10e3 * 4 / 8) < 1e-8
    )
    assert math.isclose(f["x_span_max"][0], 2.0)
    assert "EI" in bad({"spans": [{"L": 4.0, "w": 1.0}]})
    assert "inside" in bad({"spans": [{"L": 4.0, "P": [[1.0, 5.0]]}], "EI": 1.0})
