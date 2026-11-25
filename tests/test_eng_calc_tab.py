"""eng_calc, the 4 operations with data tables (D44): fatigue, bolts, welds, materials.

Independent methods: numerical examples of the sources (Shigley: ka = 0.860, kb = 0.969, a = 161.4 kpsi,
b = -0.0716); tensile stress area with the alternative formula 0.7854 (D - 0.938194 P)^2; Gerber
substituted back into its equation; Basquin line through the two points (10^3, f Sut) and (10^6, Se);
continuity of the EN 1993-1-9 curves at the slope-change points; directional method reduced to the
simplified one for longitudinal shear only.
"""

from __future__ import annotations

import math

import pytest

from tests.conftest import call, ok

MPA = 1e6
KSI = 6.894757293168361e6


def e(operation: str, **params: object) -> dict:
    return ok("eng_calc", operation=operation, params=params)


def full(operation: str, **params: object) -> dict:
    resp = call("eng_calc", operation=operation, params=params)
    assert resp["ok"], resp["error"]
    return resp


def err(operation: str, **params: object) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert resp["ok"] is False, resp["result"]
    return resp["error"]["code"]


def rel(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


# ------------------------------------------------------------------ fatigue_life: endurance


def test_marin_source_examples() -> None:
    # uoanbar, table 3-1: "ka = 4.51(520)^-0.265 = 0.860"
    r = e("fatigue_life", mode="endurance", Sut=[520, "MPa"], surface="machined", load="axial")
    assert abs(r["ka"] - 0.860) < 5e-4 and r["kb"] == 1.0 and r["kc"] == 0.85
    # oakland, eq. 6-19: "kb = 1.24(10)^-0.107 = 0.969"
    r = e("fatigue_life", mode="endurance", Sut=[600, "MPa"], surface="ground", d=[10, "mm"])
    assert abs(r["kb"] - 0.969) < 5e-4 and r["kc"] == 1.0


def test_endurance_chain() -> None:
    r = e(
        "fatigue_life",
        mode="endurance",
        Sut=[600, "MPa"],
        surface="hot_rolled",
        load="torsion",
        d=[25, "mm"],
        reliability=99,
        kd=0.95,
    )
    ka = 57.7 * 600**-0.718
    kb = 1.24 * 25**-0.107
    assert rel(r["Se_prime"], 300 * MPA) and rel(r["ka"], ka) and rel(r["kb"], kb)
    assert r["kc"] == 0.59 and r["ke"] == 0.814 and r["kd"] == 0.95
    assert rel(r["Se"], 300 * MPA * ka * kb * 0.59 * 0.95 * 0.814)
    hi = e("fatigue_life", mode="endurance", Sut=[1600, "MPa"], surface="ground", load="axial")
    assert rel(hi["Se_prime"], 700 * MPA)


@pytest.mark.parametrize(
    "bad",
    [
        {"d": [60, "mm"]},
        {"d": [10, "mm"], "reliability": 80},
        {"d": [10, "mm"], "surface": "polished"},
        {},
    ],
)
def test_endurance_errors(bad: dict) -> None:
    params = {"mode": "endurance", "Sut": [600, "MPa"], "surface": "machined"} | bad
    assert err("fatigue_life", **params) == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ fatigue_life: mean_stress


def test_mean_stress_criteria() -> None:
    sa, sm, Se, Sut, Sy = 100.0, 150.0, 250.0, 600.0, 450.0
    r = e(
        "fatigue_life",
        mode="mean_stress",
        sigma_a=[sa, "MPa"],
        sigma_m=[sm, "MPa"],
        Se=[Se, "MPa"],
        Sut=[Sut, "MPa"],
        Sy=[Sy, "MPa"],
    )
    assert rel(r["n_goodman"], 1 / (sa / Se + sm / Sut))
    assert rel(r["n_soderberg"], 1 / (sa / Se + sm / Sy))
    n = r["n_gerber"]
    assert rel(n * sa / Se + (n * sm / Sut) ** 2, 1.0, 1e-12)
    assert rel(r["sigma_ar_goodman"], sa / (1 - sm / Sut) * MPA)
    z = e(
        "fatigue_life",
        mode="mean_stress",
        sigma_a=[sa, "MPa"],
        sigma_m=0,
        Se=[Se, "MPa"],
        Sut=[Sut, "MPa"],
    )
    assert rel(z["n_goodman"], 2.5) and rel(z["n_gerber"], 2.5) and "n_soderberg" not in z


# ------------------------------------------------------------------ fatigue_life: sn_life


def test_basquin_source_example_and_points() -> None:
    # oakland, eq. 6-13 and 6-14: a = 0.82(120)^2/60 = 161.4 kpsi; b = -0.0716
    r = e("fatigue_life", mode="sn_life", Sut=120 * KSI, Se=60 * KSI, f=0.82, sigma_a=70 * KSI)
    assert abs(r["a"] / KSI - 161.4) < 0.05 and abs(r["b"] + 0.0716) < 5e-5
    top = e(
        "fatigue_life", mode="sn_life", Sut=120 * KSI, Se=60 * KSI, f=0.82, sigma_a=0.82 * 120 * KSI
    )
    assert rel(top["N"], 1e3, 1e-9)
    low = e(
        "fatigue_life",
        mode="sn_life",
        Sut=120 * KSI,
        Se=60 * KSI,
        f=0.82,
        sigma_a=60 * KSI * (1 + 1e-9),
    )
    assert rel(low["N"], 1e6, 1e-6)
    inf = e("fatigue_life", mode="sn_life", Sut=120 * KSI, Se=60 * KSI, f=0.82, sigma_a=50 * KSI)
    assert inf["N"] is None


# ------------------------------------------------------------------ fatigue_life: ec3, miner


def test_ec3_normal_curve_continuity() -> None:
    C = 71.0
    r = e("fatigue_life", mode="ec3", delta_sigma=[100, "MPa"], delta_sigma_C=[C, "MPa"])
    assert rel(r["N"], 2e6 * (C / 100) ** 3)
    D, L = r["delta_sigma_D"], r["delta_sigma_L"]
    assert abs(D / (C * MPA) - 0.737) < 5e-4 and abs(L / D - 0.549) < 5e-4  # values of the source
    at_D = e("fatigue_life", mode="ec3", delta_sigma=D, delta_sigma_C=[C, "MPa"])
    assert rel(at_D["N"], 5e6, 1e-12)
    at_L = e("fatigue_life", mode="ec3", delta_sigma=L, delta_sigma_C=[C, "MPa"])
    assert rel(at_L["N"], 1e8, 1e-12)
    assert (
        e("fatigue_life", mode="ec3", delta_sigma=L * 0.99, delta_sigma_C=[C, "MPa"])["N"] is None
    )
    g = e(
        "fatigue_life",
        mode="ec3",
        delta_sigma=[100, "MPa"],
        delta_sigma_C=[C, "MPa"],
        gamma_Mf=1.35,
        gamma_Ff=1.1,
    )
    assert rel(g["N"], 2e6 * (C / 1.35 / 110) ** 3)


def test_ec3_shear_and_miner() -> None:
    r = e("fatigue_life", mode="ec3", delta_tau=[100, "MPa"], delta_tau_C=[80, "MPa"])
    assert rel(r["N"], 2e6 * 0.8**5) and abs(r["delta_tau_L"] / (80 * MPA) - 0.457) < 5e-4
    assert rel(
        e("fatigue_life", mode="ec3", delta_tau=r["delta_tau_L"], delta_tau_C=[80, "MPa"])["N"],
        1e8,
        1e-12,
    )
    m = e("fatigue_life", mode="miner", blocks=[[1e4, 1e5], [2e5, 1e6], [1e3, 5e4]])
    assert rel(m["D"], 0.32) and rel(m["remaining"], 0.68) and m["ok"] is True
    assert e("fatigue_life", mode="miner", blocks=[[2e5, 1e5]])["ok"] is False


# ------------------------------------------------------------------ bolt_joint


def test_bolt_m12_88() -> None:
    r = e("bolt_joint", size="M12", grade="8.8")
    P, d = 1.75e-3, 12e-3
    d2, d3 = d - 0.649519 * P, d - 1.226869 * P
    As = math.pi / 4 * ((d2 + d3) / 2) ** 2
    assert rel(r["P"], P) and rel(r["As"], As, 1e-6)
    assert rel(r["As"], 0.7854 * (d - 0.938194 * P) ** 2, 1e-4)  # engineersedge form
    assert rel(r["fub"], 800 * MPA) and rel(r["fyb"], 640 * MPA)
    assert rel(r["Fv_Rd"], 0.6 * 800 * MPA * r["As"] / 1.25)
    assert rel(r["Ft_Rd"], 0.9 * 800 * MPA * r["As"] / 1.25)
    assert rel(r["Fp_C"], 0.7 * 800 * MPA * r["As"])


def test_bolt_checks_torque_slip() -> None:
    r = e(
        "bolt_joint",
        size="M16",
        grade="10.9",
        Fv=[30, "kN"],
        Ft=[40, "kN"],
        shear_planes=2,
        K=0.2,
        mu_thread=0.12,
        mu_head=0.12,
        D_km=[21, "mm"],
        mu_slip=0.5,
    )
    As, fub = r["As"], 1000 * MPA
    FvRd = 2 * 0.5 * fub * As / 1.25
    FtRd = 0.9 * fub * As / 1.25
    Fp = 0.7 * fub * As
    assert rel(r["Fv_Rd"], FvRd) and rel(r["util_combined"], 30e3 / FvRd + 40e3 / (1.4 * FtRd))
    assert rel(r["T_K"], 0.2 * 0.016 * Fp)
    assert rel(r["T_vdi"], Fp * (0.159 * 2e-3 + 0.578 * r["d2"] * 0.12 + 0.021 * 0.12 / 2))
    assert rel(r["Fs_Rd"], 1 * 2 * 0.5 * (Fp - 0.8 * 40e3) / 1.25)
    assert r["ok"] is (r["util_combined"] <= 1 and r["util_shear"] <= 1 and r["util_tension"] <= 1)


def test_bolt_variants_and_errors() -> None:
    sh = e("bolt_joint", size="M20", grade="4.8", threads_in_shear=False)
    assert rel(sh["A"], math.pi * 0.02**2 / 4) and rel(
        sh["Fv_Rd"], 0.6 * 400 * MPA * sh["A"] / 1.25
    )
    cs = e("bolt_joint", d=[10, "mm"], P=[1.25, "mm"], grade="8.8", countersunk=True)
    assert rel(cs["Ft_Rd"], 0.63 * 800 * MPA * cs["As"] / 1.25)
    w = full("bolt_joint", size="M8", grade="12.9")
    assert (
        rel(w["result"]["fub"], 1200 * MPA)
        and rel(w["result"]["fyb"], 1080 * MPA)
        and w["warnings"]
    )
    assert err("bolt_joint", size="M13", grade="8.8") == "INVALID_ARGUMENT"
    assert err("bolt_joint", size="M12", grade="9.9") == "INVALID_ARGUMENT"
    assert err("bolt_joint", size="M12", d=0.012, grade="8.8") == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ weld_fillet


def test_weld_simplified_s355() -> None:
    r = e("weld_fillet", a=[5, "mm"], steel="S355", L=[200, "mm"], F=[150, "kN"])
    fvw = 490 * MPA / (math.sqrt(3) * 0.9 * 1.25)
    assert rel(r["fu"], 490 * MPA) and r["beta_w"] == 0.9 and rel(r["fvw_d"], fvw)
    assert rel(r["Fw_Rd"], fvw * 5e-3) and rel(r["F_Rd"], fvw * 5e-3 * 0.2)
    assert rel(r["util"], 150e3 / r["F_Rd"]) and r["ok"] is (r["util"] <= 1)
    z = e("weld_fillet", z=[7, "mm"], steel="S235")
    assert rel(z["a"], 7e-3 / math.sqrt(2)) and rel(z["fu"], 360 * MPA) and z["beta_w"] == 0.8


def test_weld_directional_matches_simplified() -> None:
    s = e("weld_fillet", a=[4, "mm"], steel="S275", t=[50, "mm"])
    assert rel(s["fu"], 410 * MPA) and s["beta_w"] == 0.85
    d = e(
        "weld_fillet",
        a=[4, "mm"],
        steel="S275",
        t=[50, "mm"],
        sigma_perp=0,
        tau_perp=0,
        tau_par=s["fvw_d"],
    )
    assert rel(d["util_eq"], 1.0, 1e-12)
    g = e(
        "weld_fillet",
        a=[4, "mm"],
        fu=[430, "MPa"],
        beta_w=0.85,
        sigma_perp=[150, "MPa"],
        tau_perp=[150, "MPa"],
        tau_par=[80, "MPa"],
    )
    eq = math.sqrt(150**2 + 3 * (150**2 + 80**2)) * MPA
    assert rel(g["sigma_eq"], eq) and rel(g["util_eq"], eq / (430 * MPA / (0.85 * 1.25)))
    assert rel(g["util_perp"], 150 * MPA / (0.9 * 430 * MPA / 1.25))
    assert err("weld_fillet", a=[4, "mm"], steel="S420", t=[50, "mm"]) == "INVALID_ARGUMENT"
    assert err("weld_fillet", a=[4, "mm"], z=[6, "mm"], steel="S235") == "INVALID_ARGUMENT"


# ------------------------------------------------------------------ material_lookup


def test_material_structural_steel() -> None:
    r = e("material_lookup", material="S355", t=[30, "mm"])
    pr = r["properties"]
    assert rel(pr["fy"], 355 * MPA) and rel(pr["fu"], 490 * MPA) and rel(pr["E"], 210e9)
    assert pr["nu"] == 0.3 and rel(pr["rho"], 7850) and rel(pr["alpha"], 12e-6)
    thick = e("material_lookup", material="S355", t=[50, "mm"])["properties"]
    assert rel(thick["fy"], 335 * MPA) and rel(thick["fu"], 470 * MPA)
    assert err("material_lookup", material="S355", t=[90, "mm"]) == "INVALID_ARGUMENT"


def test_material_sources_complete() -> None:
    names = [
        "S235",
        "S275",
        "S355",
        "C45+N",
        "C45+QT",
        "AISI304",
        "Al6061-T6",
        "Ti6Al4V",
        "Cu",
        "C25/30",
        "C24",
        "PLA",
        "ABS",
    ]
    for name in names:
        r = e("material_lookup", material=name)
        for key, value in r["properties"].items():
            if value is not None:
                assert r["source"][key].startswith("https://"), (name, key)
                if isinstance(value, list):
                    assert len(value) == 2 and value[0] <= value[1], (name, key)
    al = e("material_lookup", material="Al6061-T6")["properties"]
    assert rel(al["E"], 68e9) and rel(al["fy"], 240 * MPA) and al["k"] == [151, 202]
    ss = e("material_lookup", material="AISI304")["properties"]
    assert ss["E"] == [183e9, 200e9] and ss["fy"] is None
    assert err("material_lookup", material="unobtainium") == "INVALID_ARGUMENT"
