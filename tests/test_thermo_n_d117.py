"""D117: thermo_relations with N in U, H, A and G; relations with a side that is not finite.

References: page "Maxwell relations", section "General Maxwell relationships": with the number of particles N as
a natural variable of the four potentials there are more Maxwell relations, e.g. (dmu/dp)_S,N = (dV/dN)_S,p for H;
mu = (dU/dN)_S,V = (dH/dN)_S,p = (dA/dN)_T,V = (dG/dN)_T,p. The expressions of the measure of the 475 formulas
(RAG doc 62, TD 13, 14 and 18) have no heat capacity (C_p or C_V = 0): some relations divide by zero there.
Expected values are written here in plain Python, and read back with the safe parser (as D99).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import sympy as sp

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.expr import parse
from oce_math_mcp.tools import math_solve
from oce_math_mcp.tools.common import grammar

R = 8.314462618


def run(
    expr: str, potential: str, at: dict[str, float] | None = None
) -> tuple[dict[str, Any], Ctx]:
    args: dict[str, Any] = {"operation": "thermo_relations", "expr": expr, "potential": potential}
    if at is not None:
        args["at"] = at
    ctx = Ctx()
    return math_solve.handle(args, ctx), ctx


def num(text: str, **values: float) -> float:
    e = parse(text, grammar())
    subs = {s: sp.Float(repr(values[s.name]), 30) for s in e.free_symbols}
    return float(sp.N(e.subs(subs), 30))


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def relations(out: dict[str, Any]) -> dict[str, Any]:
    return {r["relation"]: r["holds"] for r in out["relations"]}


def lhs(out: dict[str, Any], name: str) -> str:
    return next(r["lhs"] for r in out["relations"] if r["relation"] == name)


def test_mu_from_u_with_n() -> None:
    out, ctx = run("S**2/V + N**2*S", "U")  # the measure, TD 13: mu = 2 N S
    assert out["variables"] == ["S", "V", "N"]
    assert rel(num(out["quantities"]["mu"], S=1.3, V=0.7, N=2.0), 2 * 2.0 * 1.3) < 1e-14
    holds = relations(out)
    assert holds["(dT/dN)_S,V = (dmu/dS)_V,N"] is True
    assert holds["-(dp/dN)_S,V = (dmu/dV)_S,N"] is True
    assert lhs(out, "(dT/dN)_S,V = (dmu/dS)_V,N") == "2*N"  # d/dN of T = 2 S / V + N^2
    assert lhs(out, "-(dp/dN)_S,V = (dmu/dV)_S,N") == "0"  # d/dN of -p = -S^2 / V^2
    assert "mu = (dU/dN)_S,V" not in holds  # it compared dU/dN with itself
    # d(T, p)/d(S, V) = 0: (dV/dT)_p and C_p do not exist, the relation of mu_JT is not checked
    assert holds["mu_JT = (T (dV/dT)_p - V) / C_p"] is None
    assert any("not finite" in w and "mu_JT" in w for w in ctx.warnings)
    assert out["verified"] is True


def test_mu_from_a_with_n_no_crash() -> None:
    """The measure, TD 14: before D117 sympy.simplify stopped with KeyError zoo (alpha = 0 * zoo)."""
    out, ctx = run("-T*V*log(N) - N*T", "A")
    q = out["quantities"]
    assert rel(num(q["mu"], T=300.0, V=0.02, N=2.0), -300 * 0.02 / 2 - 300) < 1e-14
    for name in ("kappa_T", "alpha", "C_p", "mu_JT"):
        assert name not in q
    assert any("Not defined" in w and "kappa_T" in w for w in ctx.warnings)
    holds = relations(out)
    assert holds["-(dS/dN)_T,V = (dmu/dT)_V,N"] is True
    assert holds["-(dp/dN)_T,V = (dmu/dV)_T,N"] is True
    # d/dN of -S = -V log N - N and of -p = -T log N
    assert rel(num(lhs(out, "-(dS/dN)_T,V = (dmu/dT)_V,N"), V=0.02, N=2.0), -0.02 / 2 - 1) < 1e-14
    assert rel(num(lhs(out, "-(dp/dN)_T,V = (dmu/dV)_T,N"), T=300.0, N=2.0), -300 / 2) < 1e-14
    assert holds["(dT/dS)_V = T/C_V"] is None  # C_V = 0
    assert out["verified"] is True


def test_mu_from_g_with_n() -> None:
    out, _ = run("N*T*log(p) + N**2", "G")  # the measure, TD 18: mu = T log p + 2 N
    assert out["variables"] == ["T", "p", "N"]
    assert rel(num(out["quantities"]["mu"], T=300.0, p=1e5, N=2.0), 300 * math.log(1e5) + 4) < 1e-14
    holds = relations(out)
    assert holds["-(dS/dN)_T,p = (dmu/dT)_p,N"] is True
    assert holds["(dV/dN)_T,p = (dmu/dp)_T,N"] is True
    assert lhs(out, "-(dS/dN)_T,p = (dmu/dT)_p,N") == "log(p)"  # d/dN of -S = N log p
    assert lhs(out, "(dV/dN)_T,p = (dmu/dp)_T,N") == "T/p"  # d/dN of V = N T / p
    assert holds["(dT/dS)_p = T/C_p"] is None  # C_p = 0
    assert out["verified"] is True


def test_mu_from_h_with_n_and_the_page_relation() -> None:
    """Ideal gas: S = N (c ln T - R ln p + s0), so H(S, p, N) = N c T with T = exp((S/N + R ln p - s0)/c)."""
    at = {"S": 150.0, "p": 1e5, "N": 2.0, "R": R, "c": 29.1, "s0": 10.0}
    out, ctx = run("N*c*exp((S/N + R*log(p) - s0)/c)", "H", at)
    t = math.exp((at["S"] / at["N"] + R * math.log(at["p"]) - at["s0"]) / at["c"])
    v = out["values"]
    assert rel(v["T"], t) < 1e-14
    assert rel(v["mu"], t * (at["c"] - at["S"] / at["N"])) < 1e-13  # mu = G / N = c T - T S / N
    assert abs(v["mu_JT"]) < 1e-30  # ideal gas
    holds = relations(out)
    assert holds["(dV/dN)_S,p = (dmu/dp)_S,N"] is True  # the relation written on the page
    assert holds["(dT/dN)_S,p = (dmu/dS)_p,N"] is True
    # V = N R T / p and dT/dN = -T S / (N^2 c) at fixed S and p
    dv_dn = R * t / at["p"] * (1 - at["S"] / (at["N"] * at["c"]))
    assert rel(num(lhs(out, "(dV/dN)_S,p = (dmu/dp)_S,N"), **at), dv_dn) < 1e-13
    dt_dn = -t * at["S"] / (at["N"] ** 2 * at["c"])
    assert rel(num(lhs(out, "(dT/dN)_S,p = (dmu/dS)_p,N"), **at), dt_dn) < 1e-13
    assert out["verified"] is True
    assert ctx.warnings == []


def test_ideal_gas_a_with_n_euler() -> None:
    at = {"T": 300.0, "V": 0.03, "N": 1.5, "R": R, "c": 12.5}
    out, _ = run("-N*R*T*(log(V/N) + c/R*log(T) + 1)", "A", at)
    v = out["values"]
    t, vol, n = at["T"], at["V"], at["N"]
    assert rel(v["p"], n * R * t / vol) < 1e-14
    assert rel(v["mu"], -R * t * (math.log(vol / n) + at["c"] / R * math.log(t))) < 1e-13
    assert rel(v["G"], n * v["mu"]) < 1e-13  # Euler: G = N mu for an extensive A
    assert all(h is True for h in relations(out).values())
    assert out["verified"] is True


def test_without_n_the_variables_are_unchanged() -> None:
    out, _ = run("S**2/V", "U")
    assert out["variables"] == ["S", "V"]
    assert "mu" not in out["quantities"]
    assert not any("dN" in r["relation"] for r in out["relations"])


def test_n_is_added_to_the_variables_of_u_h_a_g_only() -> None:
    assert run("N*log(U*V/N)", "S")[0]["variables"] == ["U", "V", "N"]  # N is already there
    assert run("N*R*T/V", "eos")[0]["variables"] == ["V", "T"]  # N is a constant of p(V, T)


def test_an_infinite_expression_is_refused() -> None:
    """SymPy absorbs the finite terms: -N T log V + oo V gave p = 0 and S without N."""
    with pytest.raises(ToolError) as err:
        run("-N*T*log(V) - c*T*log(T) + oo*V", "A")
    assert err.value.code == "INVALID_ARGUMENT"
    assert "finite" in err.value.message


def test_zero_heat_capacity_seen_only_after_simplify() -> None:
    """atan(T) + atan(1/T) = pi/2 for T > 0: C_p = T (dS/dT)_p is 0 only after simplify. Before D117 sympy took
    0 / C_p as 0 and mu_JT came out 0; it is 0/0, not defined."""
    out, ctx = run("N*T*log(p) - c*T*(atan(T) + atan(1/T))", "G")
    q = out["quantities"]
    assert q["C_p"] == "0"
    assert "mu_JT" not in q
    assert any("Not defined" in w and "mu_JT" in w for w in ctx.warnings)
    holds = relations(out)
    assert holds["(dT/dS)_p = T/C_p"] is None
    assert holds["mu_JT = -(dH/dp)_T / C_p"] is None
    assert rel(num(q["mu"], T=300.0, p=1e5), 300 * math.log(1e5)) < 1e-14
    assert out["verified"] is True


def hidden_one(x: str) -> str:
    """(x + 1)^3 - x^3 - 3 x^2 - 3 x = 1, but SymPy sees it only after simplify: its derivative is a hidden 0."""
    return f"(({x} + 1)**3 - {x}**3 - 3*{x}**2 - 3*{x})"


HIDDEN = [
    # potential, expression, quantities left out, relations not checked, verified
    (
        "G",
        f"N*T*log(p) - c*T*{hidden_one('T')}",
        {"mu_JT"},
        {"mu_JT = -(dH/dp)_T / C_p", "(dT/dS)_p = T/C_p"},
        True,
    ),
    (
        "G",
        f"c*T*(1 - log(T)) + p*T*{hidden_one('p')}",
        {"C_V"},
        {"-(dp/dV)_T = 1/(V kappa_T)"},
        True,
    ),
    ("A", f"-N*R*T*log(V) - c*T*{hidden_one('T')}", set(), {"(dT/dS)_V = T/C_V"}, True),
    (
        "A",
        f"-T*V*{hidden_one('V')} - c*T*log(T)",
        {"C_p", "kappa_T", "alpha", "mu_JT"},
        {"mu_JT = -(dH/dp)_T / C_p"},
        True,
    ),
    ("U", f"V*S*{hidden_one('S')}", {"C_V", "mu_JT"}, {"mu_JT = (T (dV/dT)_p - V) / C_p"}, True),
    # d(T, p)/d(S, V) = 0: no C_p and no (dV/dT)_p, mu_JT = V / S exists
    ("U", f"S**2/V*{hidden_one('S')}", set(), {"mu_JT = (T (dV/dT)_p - V) / C_p"}, True),
    # p = -c is constant, H = S^2: (dT/dp)_H does not exist
    ("U", f"S**2 + c*V*{hidden_one('V')}", {"mu_JT"}, {"mu_JT = (T (dV/dT)_p - V) / C_p"}, True),
    ("H", f"p*S*{hidden_one('S')} + p**2", {"C_p"}, {"mu_JT = (T (dV/dT)_p - V) / C_p"}, True),
    ("S", f"N*U*{hidden_one('U')} + V", {"C_V"}, set(), True),
    # every relation divides by (dp/dV)_T = 0: nothing is checked, so not verified
    (
        "eos",
        f"R*T*{hidden_one('V')}",
        {"kappa_T", "alpha", "Cp_minus_Cv"},
        {"C_p - C_V = -T (dp/dT)_V^2 / (dp/dV)_T", "(dp/dT)_V = alpha / kappa_T"},
        False,
    ),
]


@pytest.mark.parametrize(("potential", "expr", "left_out", "unchecked", "verified"), HIDDEN)
def test_hidden_zero_in_each_denominator(
    potential: str, expr: str, left_out: set[str], unchecked: set[str], verified: bool
) -> None:
    out, ctx = run(expr, potential)
    assert not left_out & set(out["quantities"])
    for name in left_out:
        assert any("Not defined" in w and name in w for w in ctx.warnings)
    assert {r["relation"] for r in out["relations"] if r["holds"] is None} == unchecked
    assert all(r["holds"] is True for r in out["relations"] if r["holds"] is not None)
    assert out["verified"] is verified
