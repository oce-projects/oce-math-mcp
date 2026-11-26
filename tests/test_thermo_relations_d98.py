"""D98: math_solve.thermo_relations (thermodynamic potentials and Maxwell relations).

The references do not use SymPy: values at a point come from finite differences of plain Python functions with
mpmath at 40 digits, and the ideal gas and van der Waals results are the textbook closed forms.
"""

from __future__ import annotations

import mpmath
import pytest

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import math_solve

R = 8.314462618


def run(expr: str, potential: str, at: dict | None = None) -> dict:
    args: dict = {"operation": "thermo_relations", "expr": expr, "potential": potential}
    if at is not None:
        args["at"] = at
    return math_solve.handle(args, Ctx())


def holds_all(out: dict) -> bool:
    return bool(out["relations"]) and all(r["holds"] for r in out["relations"])


def test_van_der_waals_equation_of_state() -> None:
    at = {"n": 1, "R": R, "a": 0.1378, "b": 3.183e-5, "T": 300, "V": 0.0224}
    out = run("n*R*T/(V - n*b) - a*n**2/V**2", "eos", at)
    assert out["verified"] is True and holds_all(out)

    def p(v, t):
        return at["n"] * R * t / (v - at["n"] * at["b"]) - at["a"] * at["n"] ** 2 / v**2

    with mpmath.workdps(40):
        v0, t0 = mpmath.mpf(at["V"]), mpmath.mpf(at["T"])
        p_t = mpmath.diff(lambda t: p(v0, t), t0)
        p_v = mpmath.diff(lambda v: p(v, t0), v0)
        want = {
            "dp_dT_V": p_t,
            "dp_dV_T": p_v,
            "dU_dV_T": t0 * p_t - p(v0, t0),
            "kappa_T": -1 / (v0 * p_v),
            "alpha": -p_t / (v0 * p_v),
            "Cp_minus_Cv": -t0 * p_t**2 / p_v,
        }
    for key, ref in want.items():
        assert out["values"][key] == pytest.approx(float(ref), rel=1e-12, abs=0), key
    # (dU/dV)_T of van der Waals is a n^2 / V^2 exactly
    assert out["values"]["dU_dV_T"] == pytest.approx(0.1378 / 0.0224**2, rel=1e-12, abs=0)


def test_helmholtz_ideal_gas_gives_the_textbook_results() -> None:
    expr = "-n*R*T*(log(V/n) + 3/2*log(T) + c)"
    out = run(expr, "A", {"n": 2, "R": R, "c": 1.5, "T": 400, "V": 0.05})
    assert out["verified"] is True and holds_all(out)
    v = out["values"]
    assert v["p"] == pytest.approx(2 * R * 400 / 0.05, rel=1e-13, abs=0)
    assert v["U"] == pytest.approx(1.5 * 2 * R * 400, rel=1e-13, abs=0)
    assert v["H"] == pytest.approx(2.5 * 2 * R * 400, rel=1e-13, abs=0)
    assert v["C_V"] == pytest.approx(1.5 * 2 * R, rel=1e-13, abs=0)
    assert v["C_p"] == pytest.approx(2.5 * 2 * R, rel=1e-13, abs=0)
    assert v["kappa_T"] == pytest.approx(0.05 / (2 * R * 400), rel=1e-13, abs=0)  # 1 / p
    assert v["alpha"] == pytest.approx(1 / 400, rel=1e-13, abs=0)  # 1 / T
    names = {r["relation"] for r in out["relations"]}
    assert "(dS/dV)_T = (dp/dT)_V" in names
    assert "U = -T^2 (d(A/T)/dT)_V" in names
    assert "G = -V^2 (d(A/V)/dV)_T" in names
    assert "(dU/dV)_T = T (dp/dT)_V - p" in names


def test_gibbs_with_moles_gives_chemical_potential() -> None:
    expr = "N*cp*(T - T*log(T)) + N*R*T*log(p) + N*R*T*log(N)"
    out = run(expr, "G", {"N": 1, "cp": 29.1, "R": R, "T": 350, "p": 2e5})
    assert out["verified"] is True and holds_all(out)
    v = out["values"]
    assert v["V"] == pytest.approx(R * 350 / 2e5, rel=1e-13, abs=0)
    assert v["H"] == pytest.approx(29.1 * 350, rel=1e-13, abs=0)
    assert v["C_p"] == pytest.approx(29.1, rel=1e-13, abs=0)
    assert v["C_V"] == pytest.approx(29.1 - R, rel=1e-12, abs=0)

    def g(nn, t, pp):
        return (
            nn * 29.1 * (t - t * mpmath.log(t))
            + nn * R * t * mpmath.log(pp)
            + nn * R * t * mpmath.log(nn)
        )

    with mpmath.workdps(40):
        mu = mpmath.diff(lambda nn: g(nn, 350, 2e5), 1)
        s = -mpmath.diff(lambda t: g(1, t, 2e5), 350)
    assert v["mu"] == pytest.approx(float(mu), rel=1e-12, abs=0)
    assert v["S"] == pytest.approx(float(s), rel=1e-12, abs=0)
    names = {r["relation"] for r in out["relations"]}
    assert "-(dS/dp)_T = (dV/dT)_p" in names
    assert "H = -T^2 (d(G/T)/dT)_p" in names
    assert "(dH/dp)_T = V - T (dV/dT)_p" in names


def test_internal_energy_and_enthalpy_fundamental_relations() -> None:
    # monatomic ideal gas: U(S, V) = c V^(-2/3) exp(2 S / (3 n R))
    out = run("c*V**(-2/3)*exp(2*S/(3*n*R))", "U", {"c": 1e3, "n": 1, "R": R, "S": 10, "V": 0.02})
    assert out["verified"] is True and holds_all(out)
    v = out["values"]
    u = 1e3 * 0.02 ** (-2 / 3) * mpmath.exp(2 * 10 / (3 * R))
    assert v["T"] == pytest.approx(float(2 * u / (3 * R)), rel=1e-12, abs=0)
    assert v["p"] == pytest.approx(float(2 * u / (3 * 0.02)), rel=1e-12, abs=0)
    assert v["p"] * 0.02 == pytest.approx(R * v["T"], rel=1e-12, abs=0)  # pV = nRT
    assert v["C_V"] == pytest.approx(1.5 * R, rel=1e-12, abs=0)
    assert "(dT/dV)_S = -(dp/dS)_V" in {r["relation"] for r in out["relations"]}
    # ideal gas H(S, p) = c p^(2/5) exp(2 S / (5 n R))
    out = run("c*p**(2/5)*exp(2*S/(5*n*R))", "H", {"c": 50, "n": 1, "R": R, "S": 5, "p": 1e5})
    assert out["verified"] is True and holds_all(out)
    v = out["values"]
    assert v["V"] * 1e5 == pytest.approx(R * v["T"], rel=1e-12, abs=0)
    assert v["C_p"] == pytest.approx(2.5 * R, rel=1e-12, abs=0)
    assert "(dT/dp)_S = (dV/dS)_p" in {r["relation"] for r in out["relations"]}


def test_entropy_fundamental_relation() -> None:
    expr = "N*k*log(V*U**(3/2)/N**(5/2)) + N*k*s0"
    out = run(expr, "S", {"N": 1e22, "k": 1.380649e-23, "s0": 2, "U": 60, "V": 1e-3})
    assert out["verified"] is True and holds_all(out)
    v = out["values"]
    assert v["T"] == pytest.approx(2 * 60 / (3 * 1e22 * 1.380649e-23), rel=1e-12, abs=0)
    assert v["p"] * 1e-3 == pytest.approx(1e22 * 1.380649e-23 * v["T"], rel=1e-12, abs=0)
    assert v["C_V"] == pytest.approx(1.5 * 1e22 * 1.380649e-23, rel=1e-12, abs=0)
    names = {r["relation"] for r in out["relations"]}
    assert {"(dS/dU)_V,N = 1/T", "(dS/dV)_U,N = p/T", "(dS/dN)_U,V = -mu/T"} <= names


def test_symbolic_results_without_values() -> None:
    out = run("n*R*T/V", "eos")
    assert "values" not in out
    assert out["quantities"]["dU_dV_T"] == "0"
    assert out["verified"] is True


def test_errors() -> None:
    with pytest.raises(ToolError):
        run("n*R*T/V", "Q")
    with pytest.raises(ToolError):  # no natural variable of A
        run("x**2 + y", "A")
    with pytest.raises(ToolError):
        math_solve.handle({"operation": "thermo_relations", "expr": "T*V"}, Ctx())
    with pytest.raises(ToolError):
        run("n*R*T/V", "eos", {"T": "300"})


def test_grand_potential_of_the_ideal_gas() -> None:
    # Omega(T, V, mu) = -k T V exp(mu / k T) / Lambda^3 with Lambda = h / sqrt(2 pi m k T)
    expr = "-k*T*V*exp(mu/(k*T))*(2*pi*m*k*T/h**2)**(3/2)"
    at = {
        "k": 1.380649e-23,
        "h": 6.62607015e-34,
        "m": 6.6335e-26,
        "T": 300,
        "V": 1e-3,
        "mu": -6e-20,
    }
    out = run(expr, "Omega", at)
    assert out["verified"] is True and holds_all(out)
    v = out["values"]
    kt = 1.380649e-23 * 300
    # N = -dOmega/dmu = -Omega / (k T) and p = -Omega / V: p V = N k T
    assert v["p"] * 1e-3 == pytest.approx(v["N"] * kt, rel=1e-12, abs=0)
    assert v["U"] == pytest.approx(1.5 * v["N"] * kt, rel=1e-12, abs=0)
    assert v["G"] == pytest.approx(-6e-20 * v["N"], rel=1e-12, abs=0)

    def omega(t, vol, mu):
        k, h, m = 1.380649e-23, 6.62607015e-34, 6.6335e-26
        return -k * t * vol * mpmath.exp(mu / (k * t)) * (2 * mpmath.pi * m * k * t / h**2) ** 1.5

    with mpmath.workdps(40):
        s = -mpmath.diff(lambda t: omega(t, mpmath.mpf("1e-3"), mpmath.mpf("-6e-20")), 300)
    assert v["S"] == pytest.approx(float(s), rel=1e-11, abs=0)
    names = {r["relation"] for r in out["relations"]}
    assert {
        "(dS/dV)_T,mu = (dp/dT)_V,mu",
        "(dS/dmu)_T,V = (dN/dT)_V,mu",
        "(dp/dmu)_T,V = (dN/dV)_T,mu",
    } <= names
    with pytest.raises(ToolError):  # N is a result of Omega
        run("-T*V*exp(mu/T)*N", "Omega")
