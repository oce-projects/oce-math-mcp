"""D116: Joule-Thomson coefficient, Mach angle, Cherenkov radiation, Fick's first law, ionic strength.

References independent of the module, from the local offline library and CODATA 2022:
- "Joule-Thomson effect": mu_JT = (dT/dp)_H = -mu_T / C_p with mu_T = (dH/dp)_T (cyclic rule), and
  mu_JT = (V / C_p) (alpha T - 1); for a van der Waals gas mu_JT = -(V_m / C_p) (R T V_m^2 b - 2 a (V_m - b)^2) /
  (R T V_m^3 - 2 a (V_m - b)^2). The gas with covolume, V = R T / p + b and C_p = c, has mu_JT = -b / c (the
  measure of the 475 formulas, RAG doc 62: b = 3e-5, c = 29.1).
- "Mach wave": mu = arcsin(1 / M) for M >= 1 (the measure: M = 2 gives pi / 6).
- "Cherenkov radiation", section "Emission angle": cos theta = 1 / (n beta), emission only for beta > 1 / n
  (the measure: n = 1.33, beta = 0.99 gives 0.7082914737803945 rad).
- "Kinetic theory of gases", section "Diffusion coefficient and diffusion flux": J = -D dn/dy (Fick's first law)
  and D_0 = (1/3) v_mean l for a dilute gas (the measure: D = 1e-9, gradient -50 gives 5e-8).
- "Ionic strength": I = 1/2 sum c_i z_i^2; the page example 0.050 M Na2SO4 with 0.020 M KCl gives 0.17 M (the
  measure: z = 2, c = 0.1 and z = -1, c = 0.2 give 0.3).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from oce_math_mcp.constants import lookup
from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc, math_solve

R = 8.314462618
C = 299792458.0
E_CH = 1.602176634e-19
K_B = lookup("Boltzmann constant")[1]
N_A = lookup("Avogadro constant")[1]
M_E = lookup("electron mass")[1]


def run(operation: str, params: dict[str, Any]) -> tuple[dict[str, Any], Ctx]:
    ctx = Ctx()
    out = eng_calc.handle({"operation": operation, "params": params}, ctx)
    return out, ctx


def thermo(expr: str, potential: str, at: dict[str, float]) -> dict[str, Any]:
    args = {"operation": "thermo_relations", "expr": expr, "potential": potential, "at": at}
    return math_solve.handle(args, Ctx())


def units(ctx: Ctx) -> dict[str, str]:
    assert ctx.units is not None
    return ctx.units


def formula(ctx: Ctx) -> str:
    assert ctx.formula is not None
    return ctx.formula


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def relation(out: dict[str, Any], name: str) -> dict[str, Any]:
    found = [r for r in out["relations"] if r["relation"] == name]
    assert len(found) == 1, [r["relation"] for r in out["relations"]]
    return found[0]


def bad(operation: str, params: dict[str, Any], match: str) -> None:
    with pytest.raises(ToolError, match=match) as info:
        run(operation, params)
    assert info.value.code == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ Joule-Thomson

COV = {"R": R, "b": 3e-5, "c": 29.1}
VDW = {"R": R, "a": 0.1382, "b": 3.87e-5, "c": 20.8}


def vdw_reference(t: float, v: float) -> float:
    """The page formula, with C_p = C_V - T (dp/dT)_V^2 / (dp/dV)_T of the same gas (C_V = c)."""
    a, b, c = VDW["a"], VDW["b"], VDW["c"]
    p_t = R / (v - b)
    p_v = -R * t / (v - b) ** 2 + 2 * a / v**3
    cp = c - t * p_t**2 / p_v
    num = R * t * v**2 * b - 2 * a * (v - b) ** 2
    den = R * t * v**3 - 2 * a * (v - b) ** 2
    return -(v / cp) * num / den


def test_jt_covolume_gas_from_g() -> None:
    out = thermo("R*T*log(p) + b*p - c*T*log(T)", "G", {"T": 300, "p": 1e5, **COV})
    assert out["quantities"]["mu_JT"] == "-b/c"
    assert rel(out["values"]["mu_JT"], -1.0309278350515464e-06) < 1e-14  # the measure: -b / c
    assert relation(out, "mu_JT = -(dH/dp)_T / C_p")["holds"] is True
    assert out["verified"] is True


def test_jt_same_gas_from_h() -> None:
    """H(S, p) of the same gas: T = exp((S + R ln p - c) / c), H = c T + b p."""
    out = thermo("c*exp((S + R*log(p) - c)/c) + b*p", "H", {"S": 150, "p": 1e5, **COV})
    assert rel(out["values"]["mu_JT"], -3e-5 / 29.1) < 1e-14
    assert relation(out, "mu_JT = (T (dV/dT)_p - V) / C_p")["holds"] is True
    assert out["verified"] is True


def test_jt_ideal_gas_is_zero() -> None:
    out = thermo("R*T*log(p) - c*T*log(T)", "G", {"T": 300, "p": 1e5, "R": R, "c": 29.1})
    assert out["quantities"]["mu_JT"] == "0"
    assert out["values"]["mu_JT"] == 0


def test_jt_van_der_waals_from_a() -> None:
    expr = "-R*T*log(V - b) - a/V - c*T*log(T)"
    out = thermo(expr, "A", {"T": 300, "V": 0.0224, **VDW})
    assert rel(out["values"]["mu_JT"], vdw_reference(300, 0.0224)) < 1e-12
    assert out["values"]["mu_JT"] > 0  # below the inversion temperature: the gas cools on expansion
    assert relation(out, "mu_JT = -(dH/dp)_T / C_p")["holds"] is True
    assert out["verified"] is True


def test_jt_van_der_waals_from_u() -> None:
    """U(S, V) of the same gas: S = R ln(V - b) + c ln T, U = c T - a / V."""
    expr = "c*exp((S - R*log(V - b))/c) - a/V"
    s0 = R * math.log(0.0224 - VDW["b"]) + VDW["c"] * math.log(300)
    out = thermo(expr, "U", {"S": s0, "V": 0.0224, **VDW})
    assert rel(out["values"]["T"], 300) < 1e-13
    assert rel(out["values"]["mu_JT"], vdw_reference(300, 0.0224)) < 1e-12
    assert relation(out, "mu_JT = (T (dV/dT)_p - V) / C_p")["holds"] is True
    assert out["verified"] is True


def test_jt_van_der_waals_low_pressure_limit() -> None:
    """At low pressure mu_JT -> (2 a / (R T) - b) / C_p with C_p -> c + R."""
    out = thermo("-R*T*log(V - b) - a/V - c*T*log(T)", "A", {"T": 300, "V": 1000, **VDW})
    limit = (2 * VDW["a"] / (R * 300) - VDW["b"]) / (VDW["c"] + R)
    assert rel(out["values"]["mu_JT"], limit) < 1e-6


def test_no_jt_without_enthalpy_variables() -> None:
    for expr, pot in (("n*R*T/V", "eos"), ("N*log(U*V/N)", "S")):
        out = math_solve.handle(
            {"operation": "thermo_relations", "expr": expr, "potential": pot}, Ctx()
        )
        assert "mu_JT" not in out["quantities"]


# ------------------------------------------------------------------ Mach angle


def test_mach_angle_isentropic() -> None:
    out, ctx = run("compressible_flow", {"mode": "isentropic", "M": 2})
    # the measure: pi / 6
    assert rel(math.radians(out["mach_angle_deg"]), 0.5235987755982988) < 1e-14
    assert units(ctx)["mach_angle_deg"] == "deg"
    assert "asin(1/M)" in formula(ctx)
    assert run("compressible_flow", {"mode": "isentropic", "M": 1})[0]["mach_angle_deg"] == 90
    sub, sctx = run("compressible_flow", {"mode": "isentropic", "M": 0.5})
    assert "mach_angle_deg" not in sub
    assert "asin(1/M)" not in formula(sctx)


def test_mach_angle_area_mach() -> None:
    sup, _ = run(
        "compressible_flow", {"mode": "area_mach", "A_ratio": 1.6875, "branch": "supersonic"}
    )
    assert rel(sup["M"], 2) < 1e-9  # A/A* = 1.6875 at M = 2 for gamma = 1.4
    assert rel(sup["mach_angle_deg"], math.degrees(math.asin(1 / sup["M"]))) < 1e-14
    sub, _ = run(
        "compressible_flow", {"mode": "area_mach", "A_ratio": 1.6875, "branch": "subsonic"}
    )
    assert "mach_angle_deg" not in sub


# ------------------------------------------------------------------ Cherenkov


def cherenkov(**params: Any) -> tuple[dict[str, Any], Ctx]:
    return run("modern_physics", {"mode": "cherenkov", **params})


def gamma_threshold(n: float) -> float:
    return 1 / math.sqrt(1 - 1 / n**2)


def test_cherenkov_angle_of_the_measure() -> None:
    out, ctx = cherenkov(n=1.33, beta=0.99)
    assert rel(math.radians(out["theta_deg"]), 0.7082914737803945) < 1e-14
    assert out["emits"] is True
    assert rel(out["beta_threshold"], 1 / 1.33) < 1e-14  # outputs have 15 significant digits
    assert rel(out["v_threshold"], C / 1.33) < 1e-14
    assert rel(out["gamma_threshold"], gamma_threshold(1.33)) < 1e-14
    assert rel(out["theta_max_deg"], math.degrees(math.acos(1 / 1.33))) < 1e-14
    assert rel(out["gamma"], 1 / math.sqrt(1 - 0.99**2)) < 1e-14
    assert "kinetic_threshold_eV" not in out  # no mass given
    assert units(ctx)["theta_deg"] == "deg" and units(ctx)["v_threshold"] == "m/s"
    assert ctx.warnings == []
    same, _ = cherenkov(n=1.33, v=0.99 * C)
    assert rel(same["theta_deg"], out["theta_deg"]) < 1e-14


def test_cherenkov_electron_threshold_in_water() -> None:
    k_j = 1e6 * E_CH  # 1 MeV
    out, _ = cherenkov(n=1.33, particle="electron", kinetic_energy=k_j)
    mc2 = M_E * C**2
    gamma = 1 + k_j / mc2
    beta = math.sqrt(1 - 1 / gamma**2)
    assert rel(out["beta"], beta) < 1e-14
    assert rel(out["theta_deg"], math.degrees(math.acos(1 / (1.33 * beta)))) < 1e-13
    k_t = mc2 * (gamma_threshold(1.33) - 1) / E_CH
    assert rel(out["kinetic_threshold_eV"], k_t) < 1e-13
    assert 0.26e6 < out["kinetic_threshold_eV"] < 0.27e6  # about 0.26 MeV for electrons in water
    assert rel(out["kinetic_energy_eV"], 1e6) < 1e-14


def test_cherenkov_from_momentum_and_mass() -> None:
    m, mom = 1.67e-27, 2e-18
    out, _ = cherenkov(n=1.5, m=m, p=mom)
    beta = mom * C / math.hypot(mom * C, m * C**2)
    assert rel(out["beta"], beta) < 1e-14
    assert rel(out["kinetic_threshold_eV"], m * C**2 * (gamma_threshold(1.5) - 1) / E_CH) < 1e-13


def test_cherenkov_below_threshold_warns() -> None:
    out, ctx = cherenkov(n=1.33, beta=0.7)
    assert out["emits"] is False and out["theta_deg"] is None
    assert any("threshold" in w for w in ctx.warnings)


def test_cherenkov_errors() -> None:
    bad("modern_physics", {"mode": "cherenkov", "n": 1.0, "beta": 0.9}, r"params\.n .* must be > 1")
    bad(
        "modern_physics",
        {"mode": "cherenkov", "n": 1.33, "beta": 1.0},
        "less than the speed of light",
    )
    bad(
        "modern_physics",
        {"mode": "cherenkov", "n": 1.33, "kinetic_energy": 1e-13},
        "the mass is needed",
    )
    bad("modern_physics", {"mode": "cherenkov", "n": 1.33, "beta": 0.9, "v": 2e8}, "exactly one")
    bad("modern_physics", {"mode": "cherenkov", "beta": 0.9}, "params.n")


# ------------------------------------------------------------------ Fick's first law


def diffusion(**params: Any) -> tuple[dict[str, Any], Ctx]:
    return run("kinetic_theory", {"mode": "diffusion", **params})


def test_fick_with_a_given_coefficient() -> None:
    out, ctx = diffusion(D=1e-9, dn_dx=-50)
    assert rel(out["J"], 5e-8) < 1e-15  # the measure
    assert units(ctx)["J"] == "1/(m2*s)" and units(ctx)["D"] == "m2/s"
    assert units(ctx)["dn_dx"] == "1/m4"
    same, _ = diffusion(D=[1e-5, "cm2/s"], dn_dx=-50)
    assert rel(same["J"], 5e-8) < 1e-15
    back, _ = diffusion(D=1e-9, dn_dx=50)
    assert rel(back["J"], -5e-8) < 1e-15  # the flux goes down the gradient


def test_fick_with_the_kinetic_coefficient() -> None:
    t, pressure, d, mm = 300.0, 101325.0, 3.7e-10, 0.028
    out, _ = diffusion(T=t, p=pressure, d=d, M_molar=mm, dn_dx=-1e24)
    m = mm / N_A
    ell = K_B * t / (math.sqrt(2) * math.pi * d**2 * pressure)
    v_mean = math.sqrt(8 * K_B * t / (math.pi * m))
    assert rel(out["ell"], ell) < 1e-14
    assert rel(out["v_mean"], v_mean) < 1e-14
    assert rel(out["D"], ell * v_mean / 3) < 1e-14
    assert rel(out["J"], ell * v_mean / 3 * 1e24) < 1e-14
    mfp, _ = run(
        "kinetic_theory", {"mode": "mean_free_path", "T": t, "p": pressure, "d": d, "M_molar": mm}
    )
    assert rel(out["ell"], mfp["ell"]) < 1e-15


def test_fick_errors() -> None:
    bad("kinetic_theory", {"mode": "diffusion", "D": 1e-9}, "params.dn_dx")
    bad("kinetic_theory", {"mode": "diffusion", "D": 1e-9, "T": 300, "dn_dx": 1}, "not both")
    bad(
        "kinetic_theory",
        {"mode": "diffusion", "T": 300, "p": 1e5, "d": 3e-10, "dn_dx": 1},
        "molecule mass",
    )


# ------------------------------------------------------------------ ionic strength


def ionic(ions: Any) -> tuple[dict[str, Any], Ctx]:
    return run("electrochemistry", {"mode": "ionic_strength", "ions": ions})


def test_ionic_strength_of_the_measure() -> None:
    out, ctx = ionic([[2, 0.1], [-1, 0.2]])
    assert rel(out["ionic_strength"], 0.3) < 1e-15
    assert out["charge_sum"] == 0
    assert units(ctx)["ionic_strength"] == "mol/L"
    assert ctx.warnings == []
    assert "1/2" in formula(ctx)


def test_ionic_strength_page_example() -> None:
    """0.050 M Na2SO4 and 0.020 M KCl: Na+ 0.100, SO4 2- 0.050, K+ 0.020, Cl- 0.020 (mol/L)."""
    out, ctx = ionic([[1, 0.1], [-2, [50, "mmol/L"]], [1, 0.02], [-1, [20, "mmol/L"]]])
    assert rel(out["ionic_strength"], 0.17) < 1e-14
    assert abs(out["charge_sum"]) < 1e-15
    assert ctx.warnings == []


def test_ionic_strength_not_neutral_warns() -> None:
    out, ctx = ionic([[1, 0.1]])
    assert rel(out["ionic_strength"], 0.05) < 1e-15
    assert rel(out["charge_sum"], 0.1) < 1e-15
    assert any("not electrically neutral" in w for w in ctx.warnings)


def test_ionic_strength_errors() -> None:
    bad("electrochemistry", {"mode": "ionic_strength"}, "params.ions")
    bad("electrochemistry", {"mode": "ionic_strength", "ions": []}, "params.ions")
    bad("electrochemistry", {"mode": "ionic_strength", "ions": [[1.5, 0.1]]}, "nonzero integer")
    bad("electrochemistry", {"mode": "ionic_strength", "ions": [[0, 0.1]]}, "nonzero integer")
    bad("electrochemistry", {"mode": "ionic_strength", "ions": [[True, 0.1]]}, "nonzero integer")
    bad("electrochemistry", {"mode": "ionic_strength", "ions": [[1, -0.1]]}, "must be > 0")
    bad("electrochemistry", {"mode": "ionic_strength", "ions": [[1]]}, r"\[z, c\]")
    bad("electrochemistry", {"mode": "ionic_strength", "ions": [[1, 0.1]] * 101}, "at most 100")
