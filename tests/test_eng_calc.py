"""eng_calc area A (D39): every formula checked with an independent method.

Beams: SymPy `Beam` (singularity functions, other code). Stresses: eigenvalues of the tensor and
formula by components. Lamé: equilibrium of half a tube. Shafts: stresses recomputed from the diameter.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
from sympy import Rational
from sympy.physics.continuum_mechanics.beam import Beam

from tests.conftest import call, ok


def e(operation: str, **params: object) -> dict:
    return ok("eng_calc", operation=operation, params=params)


def err(operation: str, **params: object) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert resp["ok"] is False
    return resp["error"]["code"]


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------------------------------------------------------ input units


def test_units_pair_converted_to_si() -> None:
    a = e("hooke_elastic", E=[210, "GPa"], sigma=[100, "MPa"])
    b = e("hooke_elastic", E=210e9, sigma=100e6)
    assert rel(a["epsilon"], b["epsilon"]) and rel(a["epsilon"], 100e6 / 210e9)


def test_unit_of_wrong_kind_is_refused() -> None:
    resp = call("eng_calc", operation="hooke_elastic", params={"E": [210, "mm"], "sigma": 1e6})
    assert resp["error"]["code"] == "INVALID_ARGUMENT"
    assert "length" in resp["error"]["message"]


def test_unknown_key_is_refused_with_list() -> None:
    resp = call("eng_calc", operation="hooke_elastic", params={"E": 1e9, "sigmaa": 1e6})
    assert resp["error"]["code"] == "INVALID_ARGUMENT"
    assert "sigmaa" in resp["error"]["message"] and "sigma" in resp["error"]["message"]


def test_bad_values_refused() -> None:
    assert err("hooke_elastic", E=-1, sigma=1) == "INVALID_ARGUMENT"
    assert err("hooke_elastic", E="much", sigma=1) == "INVALID_ARGUMENT"
    assert err("hooke_elastic", E=True, sigma=1) == "INVALID_ARGUMENT"
    assert err("hooke_elastic", E=[1, "GPa", 3], sigma=1) == "INVALID_ARGUMENT"
    assert err("hooke_elastic", E=[1, "parsec"], sigma=1) == "INVALID_ARGUMENT"


def test_unknown_operation_refused() -> None:
    """Since D44 every enum entry has its code; a name outside the contract is rejected."""
    assert err("material_lookup_v2") == "INVALID_ARGUMENT"


def test_units_and_formula_in_envelope() -> None:
    resp = call("eng_calc", operation="hooke_elastic", params={"E": 200e9, "sigma": 200e6})
    assert resp["units"]["sigma"] == "Pa" and resp["units"]["epsilon"] == "1"
    assert "E" in resp["formula_used"]


# ------------------------------------------------------------------ hooke_elastic


def test_hooke_bar_elongation_and_g() -> None:
    r = e("hooke_elastic", E=[200, "GPa"], nu=0.3, F=[10, "kN"], A=[100, "mm2"], L=[2, "m"])
    assert rel(r["sigma"], 100e6)
    assert rel(r["epsilon"], 5e-4)
    assert rel(r["delta_L"], 1e-3)
    assert rel(r["G"], 200e9 / 2.6)
    assert rel(r["epsilon_transverse"], -0.3 * 5e-4)


def test_hooke_shear_and_nu_from_g() -> None:
    r = e("hooke_elastic", E=210e9, G=80e9, gamma=1e-3)
    assert rel(r["nu"], 210 / 160 - 1)
    assert rel(r["tau"], 80e6)


def test_hooke_only_one_axial_input() -> None:
    assert err("hooke_elastic", E=1e9, sigma=1e6, epsilon=1e-3) == "INVALID_ARGUMENT"
    assert err("hooke_elastic", E=1e9, nu=0.7) == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ beam_calc against SymPy Beam

L_, F_, Q_, E_, I_ = 3, 1000, 2000, 210 * 10**9, Rational(8, 10**6)


def sympy_beam(support: str, load: str, a: Rational) -> Beam:
    b = Beam(L_, E_, I_)
    sup = {
        "simply_supported": [(0, "pin"), (L_, "roller")],
        "fixed_fixed": [(0, "fixed"), (L_, "fixed")],
        "cantilever": [(0, "fixed")],
    }[support]
    syms = []
    for pos, kind in sup:
        r = b.apply_support(pos, kind)
        syms += list(r) if isinstance(r, tuple) else [r]
    if load == "point":
        b.apply_load(F_, a, -1)
    else:
        b.apply_load(Q_, 0, 0, end=L_)
    b.solve_for_reaction_loads(*syms)
    return b


CASES = [
    ("simply_supported", "point", Rational(1)),
    ("simply_supported", "point", Rational(2)),
    ("simply_supported", "uniform", None),
    ("fixed_fixed", "point", Rational(1)),
    ("fixed_fixed", "point", Rational(9, 4)),
    ("fixed_fixed", "uniform", None),
    ("cantilever", "point", Rational(2)),
    ("cantilever", "uniform", None),
]


@pytest.mark.parametrize(("support", "load", "a"), CASES)
def test_beam_matches_sympy_singularity_method(support: str, load: str, a: Rational | None) -> None:
    params: dict[str, object] = {"support": support, "load": load, "L": L_, "E": E_, "I": float(I_)}
    if load == "point":
        params |= {"F": F_, "a": float(a)}  # type: ignore[arg-type]
    else:
        params |= {"q": Q_}
    r = e("beam_calc", **params)
    b: Any = sympy_beam(support, load, a if a is not None else Rational(0))
    x = b.variable
    m, d, s = b.bending_moment(), b.deflection(), b.slope()
    # maximum moment where the tool says, and no point with a larger moment
    # at the ends SymPy also adds the reaction (moment 0): evaluate just inside the beam
    eps = Rational(1, 10**12)
    xm = min(max(cast(Any, Rational(r["x_M_max"])).limit_denominator(10**6), eps), L_ - eps)
    assert rel(abs(float(m.subs(x, xm))), r["M_max"], 1e-9)
    grid = [abs(float(m.subs(x, Rational(i, 40) * L_))) for i in range(41)]
    assert max(grid) <= r["M_max"] * (1 + 1e-9)
    # maximum deflection: value and position (zero slope, or free end of the cantilever)
    xf = cast(Any, Rational(r["x_f_max"])).limit_denominator(10**9)
    assert rel(abs(float(d.subs(x, xf))), r["f_max"], 1e-7)
    if support != "cantilever":
        assert abs(float(s.subs(x, xf))) < 1e-9
    assert max(abs(float(d.subs(x, Rational(i, 30) * L_))) for i in range(31)) <= r["f_max"] * (
        1 + 1e-9
    )
    # reactions: same magnitudes (SymPy gives them with the sign of its convention)
    reac = {str(k): abs(float(v)) for k, v in b.reaction_loads.items()}
    assert rel(reac["R_0"], r["R_A"], 1e-9)
    if support != "cantilever":
        assert rel(reac[f"R_{L_}"], r["R_B"], 1e-9)
    if support in {"fixed_fixed", "cantilever"}:
        assert rel(reac["M_0"], r["M_A"], 1e-9)


def test_beam_classic_midspan_values() -> None:
    EI = 210e9 * 8e-6
    r = e(
        "beam_calc",
        support="simply_supported",
        load="point",
        L=4,
        F=[5, "kN"],
        E=210e9,
        I=[800, "cm4"],
    )
    assert rel(r["f_max"], 5000 * 4**3 / (48 * EI)) and rel(r["M_max"], 5000 * 4 / 4)
    r = e(
        "beam_calc",
        support="cantilever",
        load="uniform",
        L=2,
        q=[3, "kN/m"],
        E=210e9,
        I=8e-6,
        W=[80, "cm3"],
    )
    assert rel(r["f_max"], 3000 * 2**4 / (8 * EI)) and rel(r["sigma_max"], (3000 * 4 / 2) / 80e-6)


def test_beam_sigma_from_c_and_defaults() -> None:
    r = e("beam_calc", support="cantilever", load="point", L=1, F=100, E=1e9, I=1e-8, c=0.01)
    assert rel(r["a"], 1.0) and rel(r["sigma_max"], 100 * 1 * 0.01 / 1e-8)
    r = e("beam_calc", support="fixed_fixed", load="point", L=2, F=100, E=1e9, I=1e-8)
    assert rel(r["a"], 1.0) and rel(r["f_max"], 100 * 8 / (192 * 1e9 * 1e-8))


def test_beam_bad_inputs() -> None:
    base = {"L": 2, "E": 1e9, "I": 1e-8}
    assert err("beam_calc", support="propped", load="point", F=1, **base) == "INVALID_ARGUMENT"
    assert err("beam_calc", support="cantilever", load="point", **base) == "INVALID_ARGUMENT"
    assert (
        err("beam_calc", support="simply_supported", load="point", F=1, a=2, **base)
        == "INVALID_ARGUMENT"
    )
    assert (
        err("beam_calc", support="simply_supported", load="uniform", F=1, **base)
        == "INVALID_ARGUMENT"
    )


# ------------------------------------------------------------------ shaft_torsion_bending


def test_shaft_solid_stresses_and_twist() -> None:
    r = e(
        "shaft_torsion_bending",
        d=[25, "mm"],
        T=[200, "Nm"],
        M=[150, "Nm"],
        L=[1, "m"],
        G=[80, "GPa"],
    )
    d = 0.025
    tau, sig = 16 * 200 / (math.pi * d**3), 32 * 150 / (math.pi * d**3)
    assert rel(r["tau_max"], tau) and rel(r["sigma_b"], sig)
    assert rel(r["sigma_eq"], math.sqrt(sig**2 + 3 * tau**2))
    J = math.pi * d**4 / 32
    assert rel(r["theta"], 200 * 1 / (80e9 * J)) and rel(r["theta_deg"], math.degrees(r["theta"]))


def test_shaft_power_rpm_and_hollow() -> None:
    r = e("shaft_torsion_bending", P=[15, "kW"], n=[1450, "rpm"], d=0.04, d_i=0.03)
    T = 15000 / (1450 * 2 * math.pi / 60)
    assert rel(r["T"], T)
    assert rel(r["tau_max"], T * 0.02 / (math.pi * (0.04**4 - 0.03**4) / 32))


@pytest.mark.parametrize("criterion", ["von_mises", "tresca"])
def test_shaft_min_diameter_gives_allowed_stress(criterion: str) -> None:
    r = e("shaft_torsion_bending", T=300, M=400, sigma_allow=[120, "MPa"], criterion=criterion)
    again = e("shaft_torsion_bending", T=300, M=400, d=r["d_min"], criterion=criterion)
    assert rel(again["sigma_eq"], 120e6, 1e-9)


def test_shaft_bad_inputs() -> None:
    assert err("shaft_torsion_bending", T=1) == "INVALID_ARGUMENT"
    assert err("shaft_torsion_bending", d=0.02, d_i=0.03, T=1) == "INVALID_ARGUMENT"
    assert err("shaft_torsion_bending", d=0.02, T=1, P=100, n=10) == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ stress_criterion


def vm_components(sx: float, sy: float, sz: float, txy: float, tyz: float, tzx: float) -> float:
    return math.sqrt(
        sx * sx
        + sy * sy
        + sz * sz
        - sx * sy
        - sy * sz
        - sz * sx
        + 3 * (txy * txy + tyz * tyz + tzx * tzx)
    )


def test_stress_3d_against_component_formula() -> None:
    s = {"sx": 80e6, "sy": -40e6, "sz": 25e6, "txy": 30e6, "tyz": -15e6, "tzx": 10e6}
    r = e("stress_criterion", Sy=[250, "MPa"], **s)
    vm = vm_components(*s.values())
    assert rel(r["sigma_vm"], vm, 1e-12)
    p1, p2, p3 = r["principal"]
    assert p1 >= p2 >= p3
    assert rel(p1 + p2 + p3, 80e6 - 40e6 + 25e6, 1e-12)
    assert rel(r["tau_max"], (p1 - p3) / 2) and rel(r["sigma_tresca"], p1 - p3)
    assert rel(r["SF_vm"], 250e6 / vm) and rel(r["SF_tresca"], 250e6 / (p1 - p3))


def test_stress_plane_mohr() -> None:
    r = e("stress_criterion", sx=[100, "MPa"], sy=[20, "MPa"], txy=[30, "MPa"])
    c, rad = 60e6, math.hypot(40e6, 30e6)
    assert (
        rel(r["principal"][0], c + rad)
        and rel(r["principal"][1], c - rad)
        and r["principal"][2] == 0
    )
    assert rel(r["angle_deg"], math.degrees(0.5 * math.atan2(60e6, 80e6)))


def test_stress_zero_state_has_no_safety_factor() -> None:
    resp = call("eng_calc", operation="stress_criterion", params={"sx": 0, "Sy": 1e8})
    assert resp["ok"] and resp["result"]["SF_vm"] is None and resp["warnings"]


# ------------------------------------------------------------------ euler_buckling


def test_euler_cases_and_fixed_pinned_constant() -> None:
    base = {"E": 210e9, "I": 1e-8, "L": 2.0}
    pin = e("euler_buckling", end="pinned_pinned", **base)["P_cr"]
    assert rel(pin, math.pi**2 * 210e9 * 1e-8 / 4)
    assert rel(e("euler_buckling", end="fixed_free", **base)["P_cr"], pin / 4)
    assert rel(e("euler_buckling", end="fixed_fixed", **base)["P_cr"], pin * 4)
    k = e("euler_buckling", end="fixed_pinned", **base)["K"]
    x = math.pi / k  # root of tan(x) = x
    assert abs(math.tan(x) - x) < 1e-9 and 4.49 < x < 4.50


def test_euler_slenderness_and_johnson() -> None:
    # solid round bar d = 20 mm: r = d/4
    A, Iz = math.pi * 0.02**2 / 4, math.pi * 0.02**4 / 64
    r = e(
        "euler_buckling",
        E=210e9,
        I=Iz,
        A=A,
        L=[1.5, "m"],
        end="pinned_pinned",
        Sy=[235, "MPa"],
        F=[5, "kN"],
    )
    assert rel(r["slenderness"], 1.5 / 0.005) and r["method"] == "euler"
    assert rel(r["SF"], r["P_cr"] / 5000)
    short = e("euler_buckling", E=210e9, I=Iz, A=A, L=0.3, end="pinned_pinned", Sy=235e6)
    lam = 0.3 / 0.005
    assert short["method"] == "johnson" and lam < short["slenderness_limit"]
    assert rel(short["P_cr"], (235e6 - (235e6 * lam / (2 * math.pi)) ** 2 / 210e9) * A)
    assert rel(short["slenderness_limit"], math.sqrt(2 * math.pi**2 * 210e9 / 235e6))


def test_euler_needs_end_or_k() -> None:
    assert err("euler_buckling", E=1e9, I=1e-8, L=1) == "INVALID_ARGUMENT"
    assert rel(e("euler_buckling", E=1e9, I=1e-8, L=1, K=0.7)["L0"], 0.7)


# ------------------------------------------------------------------ pressure_vessel


def test_lame_boundary_conditions_and_equilibrium() -> None:
    pi_, po, ri, ro = 20e6, 2e6, 0.05, 0.08
    r = e("pressure_vessel", p=pi_, p_o=po, d_i=2 * ri, t=ro - ri)
    assert rel(r["sigma_r_inner"], -pi_) and rel(r["sigma_r_outer"], -po)
    # equilibrium of half a tube: integral of sigma_t over the thickness = p_i r_i - p_o r_o
    A = (pi_ * ri**2 - po * ro**2) / (ro**2 - ri**2)
    B = (r["sigma_t_inner"] - A) * ri**2
    integral = A * (ro - ri) + B * (1 / ri - 1 / ro)
    assert rel(integral, pi_ * ri - po * ro, 1e-12)
    assert rel(r["sigma_t_outer"], A + B / ro**2)
    assert rel(r["sigma_a"], A)


def test_thin_wall_close_to_lame_and_warning() -> None:
    thin = e("pressure_vessel", p=[10, "bar"], d_i=[1000, "mm"], t=[5, "mm"])
    assert rel(thin["sigma_h_thin"], 1e6 * 1.005 / (2 * 0.005))
    assert rel(thin["sigma_a_thin"], thin["sigma_h_thin"] / 2)
    assert abs(thin["sigma_t_inner"] / thin["sigma_h_thin"] - 1) < 1e-4
    resp = call("eng_calc", operation="pressure_vessel", params={"p": 1e7, "d_i": 0.1, "t": 0.02})
    assert any("thin-wall" in w for w in resp["warnings"])


def test_sphere_boundary_and_thin() -> None:
    pi_, ri, ro = 5e6, 0.5, 0.52
    r = e("pressure_vessel", shape="sphere", p=pi_, d_i=2 * ri, t=ro - ri, Sy=[300, "MPa"])
    assert rel(r["sigma_r_inner"], -pi_) and abs(r["sigma_r_outer"]) < 1e-6
    # equilibrium of half a sphere: integral of sigma_t * r dr = p r_i^2 / 2
    n = 20000
    h = (ro - ri) / n
    C = (pi_ * ri**3) / (ro**3 - ri**3)
    D = (r["sigma_t_inner"] - C) * ri**3
    integral = sum((C + D / (ri + (i + 0.5) * h) ** 3) * (ri + (i + 0.5) * h) * h for i in range(n))
    assert rel(integral, pi_ * ri**2 / 2, 1e-8)
    assert rel(r["sigma_h_thin"], pi_ * (2 * ri + 0.02) / (4 * 0.02))
    assert r["SF"] > 1


def test_vessel_bad_inputs() -> None:
    assert err("pressure_vessel", p=1e6, d_i=0.1) == "INVALID_ARGUMENT"
    assert err("pressure_vessel", p=1e6, d_i=0.1, t=0.01, d_o=0.12) == "INVALID_ARGUMENT"
    assert err("pressure_vessel", p=1e6, d_i=0.1, t=0.01, shape="cone") == "INVALID_ARGUMENT"
