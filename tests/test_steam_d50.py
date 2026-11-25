"""Water and steam with IAPWS-IF97 (D50).

Single source: IAPWS R7-97(2012), IF97-Rev.pdf. Checks:
- verification tables of the document (1, 5, 15, 33, 35, 36, 42): same values to 9 significant digits;
- inconsistencies at the boundaries between regions (Table 43) and on the saturation line (Table 44): below the
  published maxima, on points spread along each boundary;
- thermodynamic identities with finite differences (cp = dh/dT at constant p, v = dg/dp at constant T,
  cv = du/dT at constant rho): method independent of the formulas of the derivatives;
- inversions (p, h), (p, s), (p, v), (T, v) against the direct state (p, T);
- Rankine cycle against the energy balance and against the isentropic transformations.
"""

from __future__ import annotations

import pytest

from oce_math_mcp.tools import eng_steam as W
from tests.conftest import call, ok

# Values of the verification tables of IF97-Rev, written by estrai_if97.py 1.1.0 (9 significant digits).
# Order in tables 5, 15, 33, 42: per row (v or p, h, u, s, cp, w) the 3 columns of the 3 points.
VERIFY = {
    "v5_r1": [
        "0.100215168e-2",
        "0.971180894e-3",
        "0.120241800e-2",
        "0.115331273e3",
        "0.184142828e3",
        "0.975542239e3",
        "0.112324818e3",
        "0.106448356e3",
        "0.971934985e3",
        "0.392294792",
        "0.368563852",
        "0.258041912e1",
        "0.417301218e1",
        "0.401008987e1",
        "0.465580682e1",
        "0.150773921e4",
        "0.163469054e4",
        "0.124071337e4",
    ],
    "v15_r2": [
        "0.394913866e2",
        "0.923015898e2",
        "0.542946619e-2",
        "0.254991145e4",
        "0.333568375e4",
        "0.263149474e4",
        "0.241169160e4",
        "0.301262819e4",
        "0.246861076e4",
        "0.852238967e1",
        "0.101749996e2",
        "0.517540298e1",
        "0.191300162e1",
        "0.208141274e1",
        "0.103505092e2",
        "0.427920172e3",
        "0.644289068e3",
        "0.480386523e3",
    ],
    "v33_r3": [
        "0.255837018e2",
        "0.222930643e2",
        "0.783095639e2",
        "0.186343019e4",
        "0.237512401e4",
        "0.225868845e4",
        "0.181226279e4",
        "0.226365868e4",
        "0.210206932e4",
        "0.405427273e1",
        "0.485438792e1",
        "0.446971906e1",
        "0.138935717e2",
        "0.446579342e2",
        "0.634165359e1",
        "0.502005554e3",
        "0.383444594e3",
        "0.760696041e3",
    ],
    "v35_ps": [
        "0.353658941e-2",
        "0.263889776e1",
        "0.123443146e2",
    ],
    "v36_ts": [
        "0.372755919e3",
        "0.453035632e3",
        "0.584149488e3",
    ],
    "v42_r5": [
        "0.138455090e1",
        "0.230761299e-1",
        "0.311385219e-1",
        "0.521976855e4",
        "0.516723514e4",
        "0.657122604e4",
        "0.452749310e4",
        "0.447495124e4",
        "0.563707038e4",
        "0.965408875e1",
        "0.772970133e1",
        "0.853640523e1",
        "0.261609445e1",
        "0.272724317e1",
        "0.288569882e1",
        "0.917068690e3",
        "0.928548002e3",
        "0.106736948e4",
    ],
}


def e(**params: object) -> dict:
    return ok("eng_calc", operation="water_steam", params=params)


def full(**params: object) -> dict:
    resp = call("eng_calc", operation="water_steam", params=params)
    assert resp["ok"], resp
    return resp


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="water_steam", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def same9(calc: float, ref: str) -> bool:
    """The computed value, rounded to 9 significant digits, is the one of the document."""
    return float(f"{calc:.8e}") == float(ref)


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


# ------------------------------------------------------------------ verification tables

GIBBS = [
    ("v5_r1", 1, [(300.0, 3.0), (300.0, 80.0), (500.0, 3.0)]),
    ("v15_r2", 2, [(300.0, 0.0035), (700.0, 0.0035), (700.0, 30.0)]),
    ("v42_r5", 5, [(1500.0, 0.5), (1500.0, 30.0), (2000.0, 30.0)]),
]


@pytest.mark.parametrize(("table", "region", "points"), GIBBS)
def test_gibbs_regions_match_verification_tables(table: str, region: int, points: list) -> None:
    ref = VERIFY[table]
    for k, (T, pm) in enumerate(points):
        r = e(mode="state", p=[pm, "MPa"], T=T)
        assert r["region"] == region
        got = [r["v"], r["h"] / 1000, r["u"] / 1000, r["s"] / 1000, r["cp"] / 1000, r["w"]]
        for q in range(6):
            assert same9(got[q], ref[3 * q + k]), (table, T, pm, q, got[q], ref[3 * q + k])


def test_region3_matches_table_33() -> None:
    ref = VERIFY["v33_r3"]
    for k, (T, rho) in enumerate([(650.0, 500.0), (650.0, 200.0), (750.0, 500.0)]):
        r = W.region3(rho, T)
        got = [r["p"], r["h"], r["u"], r["s"], r["cp"], r["w"]]
        for q in range(6):
            assert same9(got[q], ref[3 * q + k]), (T, rho, q, got[q], ref[3 * q + k])


def test_saturation_tables_35_36_and_b23_point() -> None:
    for T, ref in zip((300.0, 500.0, 600.0), VERIFY["v35_ps"], strict=True):
        assert same9(W.psat(T), ref)
        r = e(mode="saturation", T=T)
        assert same9(r["p_sat"] / 1e6, ref)
    for pm, ref in zip((0.1, 1.0, 10.0), VERIFY["v36_ts"], strict=True):
        assert same9(W.tsat(pm), ref)
        assert same9(e(mode="saturation", p=[pm, "MPa"])["T_sat"], ref)
    # Table 1: T = 0.623 150 000 x 10^3 K, p = 0.165 291 643 x 10^2 MPa (exponents on a new line in the PDF)
    assert same9(W.p_b23(623.15), "0.165291643e2")
    assert same9(W.t_b23(16.5291643), "0.623150000e3")


# ------------------------------------------------------------------ boundaries (tables 43 and 44)

# Published maxima, with half a unit of the last digit: v and cp and w as a fraction, h and g in kJ/kg,
# s in kJ/(kg K).
LIM = {
    "13": {
        "v": 0.0045e-2,
        "h": 0.0315,
        "cp": 0.1955e-2,
        "s": 0.0425e-3,
        "g": 0.0055,
        "w": 0.2995e-2,
    },
    "23": {
        "v": 0.0185e-2,
        "h": 0.1345,
        "cp": 0.3535e-2,
        "s": 0.1775e-3,
        "g": 0.0055,
        "w": 0.4035e-2,
    },
    "25": {
        "v": 0.0125e-2,
        "h": 0.0965,
        "cp": 0.0745e-2,
        "s": 0.1425e-3,
        "g": 0.0875,
        "w": 0.0465e-2,
    },
}


def _diff(a: dict, b: dict, lim: dict) -> None:
    for k in ("v", "cp", "w"):
        assert abs(a[k] - b[k]) <= lim[k] * abs(a[k]), (k, a, b)  # relative to the first equation
    for k in ("h", "s"):
        assert abs(a[k] - b[k]) <= lim[k], (k, a, b)
    ga, gb = a["h"] - a["T"] * a["s"], b["h"] - b["T"] * b["s"]
    assert abs(ga - gb) <= lim["g"], ("g", a, b)


def test_boundary_1_3_table_43() -> None:
    T = 623.15
    p0 = W.psat(T)
    for k in range(25):
        p = p0 + (100.0 - p0) * k / 24
        _diff(W.region1(p, T), W.region3(W.rho3(p, T, "liq"), T), LIM["13"])


def test_boundary_2_3_table_43() -> None:
    for k in range(25):
        T = 623.15 + (863.15 - 623.15) * k / 24
        p = W.p_b23(T)
        branch = "vap" if T < W.TC else "liq"
        _diff(W.region2(p, T), W.region3(W.rho3(p, T, branch), T), LIM["23"])


def test_boundary_2_5_table_43() -> None:
    T = 1073.15
    for k in range(25):
        p = 0.01 + (50.0 - 0.01) * k / 24
        _diff(W.region2(p, T), W.region5(p, T), LIM["25"])


def test_saturation_gibbs_table_44() -> None:
    for k in range(40):
        T = 273.16 + (623.15 - 273.16) * k / 39
        p = W.psat(T)
        f, g = W.region1(p, T), W.region2(p, T)
        assert abs((f["h"] - T * f["s"]) - (g["h"] - T * g["s"])) <= 0.0125, T
    for k in range(20):
        T = 623.2 + (647.0 - 623.2) * k / 19
        _, f, g = W.sat_T(T)
        assert f["rho"] > W.RHOC > g["rho"]
        assert abs((f["h"] - T * f["s"]) - (g["h"] - T * g["s"])) <= 0.0025, T


# ------------------------------------------------------------------ thermodynamic identities

POINTS = [
    (300.0, 3.0),
    (500.0, 3.0),
    (700.0, 0.0035),
    (700.0, 30.0),
    (640.0, 25.0),
    (750.0, 40.0),
    (1500.0, 30.0),
]


@pytest.mark.parametrize(("T", "pm"), POINTS)
def test_cp_and_v_from_finite_differences(T: float, pm: float) -> None:
    dT, dp = 1e-3, 1e-5 * pm
    up, dn = W.state_pT(pm, T + dT), W.state_pT(pm, T - dT)
    st = W.state_pT(pm, T)
    assert rel(st["cp"], (up["h"] - dn["h"]) / (2 * dT), 1e-6)
    a, b = W.state_pT(pm + dp, T), W.state_pT(pm - dp, T)
    dg = (a["h"] - T * a["s"]) - (b["h"] - T * b["s"])  # kJ/kg; v = dg/dp with p in kPa
    assert rel(st["v"], dg / (2 * dp * 1000), 1e-6)


def test_region3_cv_and_pressure_derivative() -> None:
    for rho, T in ((500.0, 650.0), (200.0, 700.0), (600.0, 640.0)):
        dT = 1e-4
        a, b, c = W.region3(rho, T + dT), W.region3(rho, T - dT), W.region3(rho, T)
        assert rel(c["cv"], (a["u"] - b["u"]) / (2 * dT), 1e-6)


# ------------------------------------------------------------------ inversions

INV = [
    (300.0, 3.0),
    (500.0, 3.0),
    (700.0, 0.0035),
    (640.0, 25.0),
    (645.0, 21.0),
    (750.0, 40.0),
    (1500.0, 30.0),
]


@pytest.mark.parametrize(("T", "pm"), INV)
def test_inverse_pairs_return_the_temperature(T: float, pm: float) -> None:
    ref = e(mode="state", p=[pm, "MPa"], T=T)
    for key in ("h", "s", "v"):
        r = e(mode="state", p=[pm, "MPa"], **{key: ref[key]})
        assert rel(r["T"], T, 1e-9), (key, r["T"])
        assert r["region"] == ref["region"]
    r = e(mode="state", T=T, v=ref["v"])
    assert rel(r["p"], pm * 1e6, 1e-8)
    r = e(mode="state", T=T, rho=[ref["rho"], "kg/m3"])
    assert rel(r["p"], pm * 1e6, 1e-8)


def test_region3_round_trip_both_branches() -> None:
    T = 640.0
    ps = W.psat(T)
    liq, vap = W.region3(560.0, T), W.region3(130.0, T)
    assert liq["p"] > ps > vap["p"]
    assert rel(W.state_pT(liq["p"], T)["rho"], 560.0, 1e-10)
    assert rel(W.state_pT(vap["p"], T)["rho"], 130.0, 1e-10)
    for rho, T2 in ((500.0, 650.0), (200.0, 650.0), (500.0, 750.0)):
        p = W.region3(rho, T2)["p"]
        assert rel(W.state_pT(p, T2)["rho"], rho, 1e-10)


def test_two_phase_quality_from_h_s_v() -> None:
    sat = e(mode="saturation", p=[1, "MPa"])
    for x in (0.0, 0.3, 1.0):
        r = e(mode="state", p=[1, "MPa"], x=x)
        assert r["region"] == 4 and r["phase"] == "liquid-vapour mixture"
        assert rel(r["h"], sat["h_f"] + x * sat["h_fg"], 1e-12)
        for key in ("h", "s"):
            back = e(mode="state", p=[1, "MPa"], **{key: r[key]})
            assert abs(back["x"] - x) < 1e-12
        back = e(mode="state", p=[1, "MPa"], v=r["v"])
        assert abs(back["x"] - x) < 1e-12
    # rigid vessel: 2 kg in 0.5 m3 at 150 °C
    r = e(mode="state", T=[150, "C"], v=0.25)
    s150 = e(mode="saturation", T=[150, "C"])
    assert rel(r["x"], (0.25 - s150["v_f"]) / (s150["v_g"] - s150["v_f"]), 1e-12)
    assert rel(r["p"], s150["p_sat"], 1e-12)


def test_saturation_in_region3_and_critical_side() -> None:
    r = e(mode="saturation", T=640.0)
    assert r["rho_f"] > W.RHOC > r["rho_g"] and r["h_fg"] > 0
    st = e(mode="state", T=640.0, x=0.5)
    assert rel(st["h"], (r["h_f"] + r["h_g"]) / 2, 1e-12)
    sc = e(mode="state", p=[25, "MPa"], T=[400, "C"])
    assert sc["phase"] == "supercritical fluid"


def test_phases_and_warning_near_saturation() -> None:
    assert e(mode="state", p=[3, "MPa"], T=300.0)["phase"] == "compressed liquid"
    assert e(mode="state", p=[3, "MPa"], T=[400, "C"])["phase"] == "superheated vapour"
    ps = W.psat(373.15)
    resp = full(mode="state", p=[ps * (1 - 1e-6), "MPa"], T=373.15)
    assert any("saturation" in w for w in resp["warnings"])


# ------------------------------------------------------------------ Rankine


def test_rankine_ideal_energy_balance_and_isentropes() -> None:
    r = e(mode="rankine", p_high=[8, "MPa"], p_low=[10, "kPa"], T_high=[480, "C"])
    assert rel(r["q_in"] - r["q_out"], r["w_net"], 1e-12)
    assert rel(r["efficiency"], r["w_net"] / r["q_in"], 1e-12)
    assert 0 < r["efficiency"] < r["efficiency_carnot"] < 1
    s = r["s_states"]
    assert rel(s[1], s[0], 1e-9) and rel(s[3], s[2], 1e-9)
    # pump: work close to v1 (p2 - p1) of the almost incompressible liquid
    v1 = e(mode="saturation", p=[10, "kPa"])["v_f"]
    assert rel(r["w_pump"], v1 * (8e6 - 1e4), 1e-2)
    sat = e(mode="saturation", p=[10, "kPa"])
    assert rel(r["h_states"][3], sat["h_f"] + r["x_turbine_exit"] * sat["h_fg"], 1e-9)


def test_rankine_efficiencies_reheat_and_power() -> None:
    base = e(mode="rankine", p_high=[8, "MPa"], p_low=[10, "kPa"], T_high=[480, "C"])
    real = e(
        mode="rankine",
        p_high=[8, "MPa"],
        p_low=[10, "kPa"],
        T_high=[480, "C"],
        eta_turbine=0.85,
        eta_pump=0.8,
    )
    assert real["efficiency"] < base["efficiency"]
    h3, h4 = real["h_states"][2], real["h_states"][3]
    h4s = e(mode="state", p=[10, "kPa"], s=real["s_states"][2])["h"]
    assert rel(h3 - h4, 0.85 * (h3 - h4s), 1e-9)
    rh = e(
        mode="rankine",
        p_high=[8, "MPa"],
        p_low=[10, "kPa"],
        T_high=[480, "C"],
        p_reheat=[0.7, "MPa"],
        T_reheat=[440, "C"],
        P_net=[100, "MW"],
    )
    assert len(rh["h_states"]) == 6 and rh["x_turbine_exit"] > base["x_turbine_exit"]
    assert rel(rh["m_dot"] * rh["w_net"], 1e8, 1e-12)
    assert rel(rh["q_in"] - rh["q_out"], rh["w_net"], 1e-12)
    sat = e(mode="rankine", p_high=[4, "MPa"], p_low=[20, "kPa"])
    assert rel(sat["h_states"][2], e(mode="saturation", p=[4, "MPa"])["h_g"], 1e-12)


# ------------------------------------------------------------------ errors


def test_errors() -> None:
    assert "273.15" in bad({"mode": "state", "p": [1, "MPa"], "T": [-5, "C"]})
    assert "100 MPa" in bad({"mode": "state", "p": [120, "MPa"], "T": 500.0})
    assert "50 MPa" in bad({"mode": "state", "p": [60, "MPa"], "T": 1500.0})
    assert "(p, T)" in bad({"mode": "state", "h": [2800, "kJ/kg"], "s": [6, "kJ/(kg*K)"]})
    assert "between 0 and 1" in bad({"mode": "state", "p": [1, "MPa"], "x": 1.2})
    assert "647.096" in bad({"mode": "saturation", "T": 700.0})
    assert "p_low" in bad({"mode": "rankine", "p_high": [1, "MPa"], "p_low": [2, "MPa"]})
    assert "No IF97 state" in bad({"mode": "state", "p": [1, "MPa"], "h": [9000, "kJ/kg"]})
