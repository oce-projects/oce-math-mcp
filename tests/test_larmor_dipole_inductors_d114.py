"""D114: Larmor precession, electric dipole, inductors in series and parallel, conductances.

References independent of the module, from the local offline library and CODATA 2022:
- "Larmor precession": omega = |gamma B|, independent of the angle; the CODATA entries "... gyromag. ratio in
  MHz/T" are separate tabulated values of gamma / (2 pi).
- "Electric dipole moment": U = -p . E and tau = p x E in a uniform field; p = q d with d from -q to +q; the
  potential p . R / (4 pi eps0 R^3) and the field (3 (p . R^) R^ - p) / (4 pi eps0 R^3) of an ideal dipole, the limit
  of two charges +-q at distances much larger than d. Checked with numpy, with E = -grad V by central differences
  of the tool's own potential, and with the exact field of the two charges from `point_charges`.
- "Inductor", section "Inductor networks": series sum L_i, parallel (sum 1/L_i)^-1, M = k sqrt(L1 L2) with
  |k| <= 1; "Inductance", section "Coupling coefficient" (negative k for opposing windings). With mutual inductance
  the equivalent inductance comes from the 2 x 2 inductance matrix [[L1, M], [M, L2]]: series 1^T L 1, parallel
  1 / (1^T L^-1 1).
- "List of electromagnetism equations", section "Electric circuits and electronics": conductances in series
  1/G = sum 1/G_i (the measure of the 475 formulas, RAG doc 62).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from oce_math_mcp.constants import lookup
from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc

EPS0 = lookup("epsilon_0")[1]
KE = 1 / (4 * math.pi * EPS0)


def run(operation: str, params: dict[str, Any]) -> tuple[dict[str, Any], Ctx]:
    ctx = Ctx()
    out = eng_calc.handle({"operation": operation, "params": params}, ctx)
    return out, ctx


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def vrel(a: list[float], b: Any) -> float:
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))


# ------------------------------------------------------------------ Larmor


def larmor(**params: Any) -> tuple[dict[str, Any], Ctx]:
    return run("magnetic_dipole", {"mode": "larmor", **params})


def test_larmor_with_gamma() -> None:
    out, ctx = larmor(gamma=2.675e8, B=1.5)
    assert rel(out["omega"], 4.0125e8) < 1e-14  # the measure: gamma B = 401,250,000 rad/s
    assert rel(out["f"], 4.0125e8 / (2 * math.pi)) < 1e-14
    assert rel(out["gamma_over_2pi"], 2.675e8 / (2 * math.pi)) < 1e-14
    assert out["gamma"] == 2.675e8
    assert ctx.units is not None
    assert ctx.units["omega"] == "rad/s" and ctx.units["f"] == "Hz"
    assert ctx.units["gamma"] == "rad/(s*T)" and ctx.units["gamma_over_2pi"] == "Hz/T"
    assert ctx.formula is not None and "omega = |gamma B|" in ctx.formula


@pytest.mark.parametrize(
    ("particle", "codata"),
    [
        ("electron", "electron gyromag. ratio in MHz/T"),
        ("proton", "proton gyromag. ratio in MHz/T"),
        ("neutron", "neutron gyromag. ratio in MHz/T"),
        ("shielded_proton", "shielded proton gyromag. ratio in MHz/T"),
        ("shielded_helion", "shielded helion gyromag. ratio in MHz/T"),
    ],
)
def test_larmor_particles_against_codata_mhz_per_tesla(particle: str, codata: str) -> None:
    per_tesla = lookup(codata)[1] * 1e6  # Hz/T
    out, _ = larmor(particle=particle, B=1.5)
    # the two CODATA entries are printed with 9 to 11 significant digits (neutron: 1.7e-9 apart)
    assert rel(out["gamma_over_2pi"], per_tesla) < 5e-9
    assert rel(out["f"], 1.5 * per_tesla) < 5e-9
    assert out["gamma"] > 0


def test_larmor_field_forms_and_sign() -> None:
    out, _ = larmor(gamma=-1.0e8, B=[0, 3, 4])  # a vector: its magnitude, 5 T
    assert rel(out["omega"], 5e8) < 1e-14  # omega = |gamma B|
    assert out["gamma"] == -1.0e8
    gauss, _ = larmor(particle="proton", B=[1.5e4, "G"])
    tesla, _ = larmor(particle="proton", B=1.5)
    assert rel(gauss["omega"], tesla["omega"]) < 1e-14


def test_larmor_errors() -> None:
    with pytest.raises(ToolError, match="exactly one"):
        larmor(gamma=1e8, particle="proton", B=1)
    with pytest.raises(ToolError, match="exactly one"):
        larmor(B=1)
    with pytest.raises(ToolError, match=r"params\.particle"):
        larmor(particle="muon", B=1)
    with pytest.raises(ToolError, match="Keys not used"):
        larmor(particle="proton", B=1, m=[0, 0, 1])
    with pytest.raises(ToolError, match=r"params\.B"):
        larmor(particle="proton", B=[0, 1, "T"])


# ------------------------------------------------------------------ electric dipole


def dipole(**params: Any) -> tuple[dict[str, Any], Ctx]:
    return run("electrostatics", {"mode": "dipole", **params})


def test_dipole_in_uniform_field() -> None:
    p = [1e-9, 2e-9, -3e-9]
    e = [1e3, -2e3, 5e2]
    out, ctx = dipole(p=p, E=e)
    assert vrel(out["tau"], np.cross(p, e)) < 1e-14
    assert rel(out["tau_magnitude"], float(np.linalg.norm(np.cross(p, e)))) < 1e-14
    assert rel(out["U"], -float(np.dot(p, e))) < 1e-14
    cos = float(np.dot(p, e)) / float(np.linalg.norm(p) * np.linalg.norm(e))
    assert rel(out["angle_deg"], math.degrees(math.acos(cos))) < 1e-12
    assert vrel(out["p"], p) < 1e-15
    assert ctx.units is not None
    assert ctx.units["p"] == "C*m" and ctx.units["tau"] == "N*m" and ctx.units["U"] == "J"
    assert ctx.formula is not None and "tau = p x E" in ctx.formula


def test_dipole_from_charges_and_aligned() -> None:
    out, _ = dipole(q=2e-9, d=[0, 0, 1e-3], E=[0, 0, 1e4])
    assert vrel(out["p"], [0, 0, 2e-12]) < 1e-15  # p = q d
    assert rel(out["p_magnitude"], 2e-12) < 1e-15
    assert out["tau_magnitude"] == 0
    zero, _ = dipole(p=[0, 0, 1e-12], E=[0, 0, 0])
    assert zero["U"] == 0 and "angle_deg" not in zero  # no direction without a field
    assert rel(out["U"], -2e-8) < 1e-14  # parallel: lowest energy
    assert out["angle_deg"] == 0
    anti, _ = dipole(q=-2e-9, d=[0, 0, 1e-3], E=[0, 0, 1e4])
    assert rel(anti["U"], 2e-8) < 1e-14 and anti["angle_deg"] == 180


def test_dipole_field_and_potential() -> None:
    p = [2e-12, -1e-12, 3e-12]
    c = [0.1, -0.2, 0.05]
    x = [0.4, 0.3, -0.25]
    out, ctx = dipole(p=p, point=x, center=c)
    r = np.subtract(x, c)
    size = float(np.linalg.norm(r))
    hat = r / size
    ref = KE * (3 * np.dot(p, hat) * hat - np.asarray(p)) / size**3
    assert vrel(out["E_dipole"], ref) < 1e-13
    assert rel(out["E_dipole_mag"], float(np.linalg.norm(ref))) < 1e-13
    assert rel(out["V_dipole"], KE * float(np.dot(p, r)) / size**3) < 1e-13
    assert vrel(out["R_vector"], r) < 1e-15 and rel(out["R_magnitude"], size) < 1e-15
    # E = -grad V, central differences of the tool's own potential
    h = 1e-5
    grad = []
    for i in range(3):
        up, dn = list(x), list(x)
        up[i] += h
        dn[i] -= h
        vu = dipole(p=p, point=up, center=c)[0]["V_dipole"]
        vd = dipole(p=p, point=dn, center=c)[0]["V_dipole"]
        grad.append(-(vu - vd) / (2 * h))
    assert vrel(grad, out["E_dipole"]) < 1e-8
    assert ctx.units is not None and ctx.units["E_dipole"] == "V/m" and ctx.units["V_dipole"] == "V"
    medium, _ = dipole(p=p, point=x, center=c, eps_r=4)
    assert vrel(medium["E_dipole"], ref / 4) < 1e-13


def test_dipole_is_the_far_field_of_two_charges() -> None:
    q, d = 3e-9, 1e-4
    x = [0.3, 0.7, 1.1]  # |x| = 1.338 m, 13,379 d
    out, ctx = dipole(q=q, d=[0, 0, d], point=x)
    pair, _ = run(
        "electrostatics",
        {"mode": "point_charges", "charges": [[q, 0, 0, d / 2], [-q, 0, 0, -d / 2]], "point": x},
    )
    assert vrel(out["E_dipole"], pair["E"]) < 1e-7  # the next terms are of order (d / R)^2
    assert rel(out["V_dipole"], pair["V"]) < 1e-7
    assert ctx.warnings == []


def test_dipole_warning_near_the_charges() -> None:
    _, near = dipole(q=1e-9, d=[0, 0, 0.2], point=[0, 0, 1.0])  # |d| = 0.2 |R|
    assert any("point_charges" in w for w in near.warnings)
    _, far = dipole(q=1e-9, d=[0, 0, 0.05], point=[0, 0, 1.0])  # |d| = 0.05 |R|
    assert far.warnings == []
    _, ideal = dipole(p=[0, 0, 1e-12], point=[0, 0, 1e-3])  # an ideal dipole has no size
    assert ideal.warnings == []


def test_dipole_errors() -> None:
    with pytest.raises(ToolError, match="exactly one"):
        dipole(p=[0, 0, 1], q=1, d=[0, 0, 1])
    with pytest.raises(ToolError, match=r"params\.d"):
        dipole(q=1e-9)
    with pytest.raises(ToolError, match=r"params\.d goes with params\.q"):
        dipole(p=[0, 0, 1e-12], d=[0, 0, 1])
    with pytest.raises(ToolError, match=r"params\.center goes with params\.point"):
        dipole(p=[0, 0, 1e-12], center=[0, 0, 0])
    with pytest.raises(ToolError, match=r"params\.eps_r goes with params\.point"):
        dipole(p=[0, 0, 1e-12], E=[1, 0, 0], eps_r=2)
    with pytest.raises(ToolError, match="the position of the dipole"):
        dipole(p=[0, 0, 1e-12], point=[1, 2, 3], center=[1, 2, 3])
    with pytest.raises(ToolError, match=r"params\.p:"):
        dipole(p=[0, 1e-12])


# ------------------------------------------------------------------ inductors and conductances


def circuit(**params: Any) -> tuple[dict[str, Any], Ctx]:
    return run("circuit_ac_dc", params)


def test_inductors_without_coupling() -> None:
    s, ctx = circuit(mode="inductors", values=[0.01, 0.02], connection="series")
    assert rel(s["L_eq"], 0.03) < 1e-15  # the measure: 0.03 H
    par, _ = circuit(mode="inductors", values=[0.01, 0.02], connection="parallel")
    assert rel(par["L_eq"], 1 / (1 / 0.01 + 1 / 0.02)) < 1e-14  # the measure: 0.006667 H
    assert "k_coupling" not in s
    assert ctx.units is not None and ctx.units["L_eq"] == "H"


@pytest.mark.parametrize("k", [0.9, 0.3, 0.0, -0.4, -0.95, 1.0])
def test_coupled_inductors_against_the_matrix(k: float) -> None:
    l1, l2 = 4e-3, 9e-3
    m = k * math.sqrt(l1 * l2)
    mat = np.array([[l1, m], [m, l2]])
    one = np.ones(2)
    s, ctx = circuit(mode="inductors", values=[l1, l2], connection="series", M=m)
    assert rel(s["L_eq"], float(one @ mat @ one)) < 1e-14  # L1 + L2 + 2 M
    assert rel(s["k_coupling"], k) < 1e-14 if k else s["k_coupling"] == 0
    assert ctx.units is not None and ctx.units["k_coupling"] == "1"
    par, _ = circuit(mode="inductors", values=[l1, l2], connection="parallel", M=m)
    if k == 1.0:
        assert par["L_eq"] == 0  # L1 L2 - M^2 = 0: the coupled coils short each other
    else:
        assert rel(par["L_eq"], 1 / float(one @ np.linalg.solve(mat, one))) < 1e-12


def test_inductor_errors() -> None:
    with pytest.raises(ToolError, match="sqrt"):
        circuit(mode="inductors", values=[1e-3, 4e-3], connection="series", M=2.1e-3)
    with pytest.raises(ToolError, match="sqrt"):
        circuit(mode="inductors", values=[1e-3, 4e-3], connection="parallel", M=-2.1e-3)
    with pytest.raises(ToolError, match="2 coupled coils"):
        circuit(mode="inductors", values=[1e-3, 2e-3, 3e-3], connection="series", M=1e-4)
    with pytest.raises(ToolError, match="undefined"):
        circuit(mode="inductors", values=[0.25, 0.25], connection="parallel", M=0.25)
    with pytest.raises(ToolError, match="inductances in H"):
        circuit(mode="inductors", values=[1e-3, -2e-3], connection="series")
    with pytest.raises(ToolError, match="Keys not used"):
        circuit(mode="inductors", values=[1e-3], connection="series", conductances=[1])


def test_conductances() -> None:
    s, ctx = circuit(mode="resistors", conductances=[0.1, 0.2], connection="series")
    assert rel(s["G_eq"], 1 / (1 / 0.1 + 1 / 0.2)) < 1e-14  # the measure: 0.06667 S
    assert rel(s["R_eq"], 15) < 1e-14
    par, _ = circuit(mode="resistors", conductances=[0.1, 0.2], connection="parallel")
    assert rel(par["G_eq"], 0.3) < 1e-14 and rel(par["R_eq"], 1 / 0.3) < 1e-14
    r, _ = circuit(mode="resistors", values=[100, 220, 330], connection="series")
    assert rel(r["R_eq"], 650) < 1e-15 and rel(r["G_eq"], 1 / 650) < 1e-15
    assert ctx.units is not None and ctx.units["G_eq"] == "S" and ctx.units["R_eq"] == "ohm"
    with pytest.raises(ToolError, match="exactly one"):
        circuit(mode="resistors", values=[1], conductances=[1], connection="series")
    with pytest.raises(ToolError, match="conductances in S"):
        circuit(mode="resistors", conductances=[0.1, 0], connection="series")
