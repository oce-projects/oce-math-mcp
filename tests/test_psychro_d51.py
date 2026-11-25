"""Psychrometrics of moist air (D51).

Checks:
- comparison with PsychroLib 2.5.0 (MIT, independent implementation of ASHRAE 2017 ch. 1) in 6 states:
  the only model difference is p_ws (IF97 here, Hyland-Wexler in PsychroLib), measured gap at
  most 0.0176% between 0 and 100 °C; tolerances chosen accordingly; the PsychroLib wet bulb
  is iterated with a tolerance of about 0.001 °C;
- internal consistency: the state rebuilt from (T, T_wb), (T, T_dp), (T, h) gives the same W;
- energy balance of adiabatic saturation written here, not with the module formula;
- mixing: conservation of dry air, water and energy; cooling coil: balance with the condensate;
  evaporative: constant wet bulb.
"""

from __future__ import annotations

import pytest

from oce_math_mcp.tools import eng_steam
from tests.conftest import call, ok

# PsychroLib 2.5.0 values (SI), written by confronto_psychrolib.py 1.0.1: input [type, t °C, x, p Pa].
REF = {
    "source": "PsychroLib 2.5.0 (MIT), SI units",
    "cases": [
        {
            "input": ["rh", 25.0, 0.5, 101325.0],
            "psychrolib": {
                "W": 0.009881043690749623,
                "RH": 0.5000000000000001,
                "T_wb": 17.889432148552928,
                "T_dp": 13.863973265964844,
                "p_w": 1584.608235071814,
                "h": 50321.958802184665,
                "v": 0.8580432638526019,
                "mu": 0.4920563364198204,
                "rho": 1.17695818641638,
            },
        },
        {
            "input": ["rh", 35.0, 0.4, 101325.0],
            "psychrolib": {
                "W": 0.01413165383445261,
                "RH": 0.4,
                "T_wb": 23.934202171156727,
                "T_dp": 19.38464909971421,
                "p_w": 2251.1277786160963,
                "h": 71473.23690458885,
                "v": 0.8927882682968101,
                "mu": 0.386366974088269,
                "rho": 1.1359150762242114,
            },
        },
        {
            "input": ["rh", 10.0, 0.8, 101325.0],
            "psychrolib": {
                "W": 0.006089102676754953,
                "RH": 0.8,
                "T_wb": 8.290297409650204,
                "T_dp": 6.713005486940737,
                "p_w": 982.3962203526237,
                "h": 25402.10310435178,
                "v": 0.8099843799409112,
                "mu": 0.7980419160289882,
                "rho": 1.2421092648109455,
            },
        },
        {
            "input": ["wb", 30.0, 20.0, 101325.0],
            "psychrolib": {
                "W": 0.010516728279452336,
                "RH": 0.39680755464866885,
                "T_wb": 19.99965558258757,
                "T_dp": 14.811527885526276,
                "p_w": 1684.856877924273,
                "h": 57069.17086490373,
                "v": 0.873310479885915,
                "mu": 0.3866079211634277,
                "rho": 1.1571105025688702,
            },
        },
        {
            "input": ["dp", 20.0, 10.0, 101325.0],
            "psychrolib": {
                "W": 0.0076300537032615865,
                "RH": 0.5250527333276988,
                "T_wb": 14.130554198264292,
                "T_dp": 9.99999999837385,
                "p_w": 1227.9952754407793,
                "h": 39486.60230961856,
                "v": 0.8406481468689843,
                "mu": 0.5192260554849202,
                "rho": 1.1986347171003777,
            },
        },
        {
            "input": ["rh", 45.0, 0.3, 90000.0],
            "psychrolib": {
                "W": 0.020545164857277174,
                "RH": 0.3,
                "T_wb": 28.262147714549613,
                "T_dp": 23.392816448100596,
                "p_w": 2877.9659803285817,
                "h": 98373.08760660431,
                "v": 1.0482125574698338,
                "mu": 0.2768763871401908,
                "rho": 0.9736051696620198,
            },
        },
        {"altitude_m": 1000.0, "p_std": 89874.51941577366},
    ],
}


def e(**params: object) -> dict:
    return ok("eng_calc", operation="psychrometrics", params=params)


def full(**params: object) -> dict:
    resp = call("eng_calc", operation="psychrometrics", params=params)
    assert resp["ok"], resp
    return resp


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="psychrometrics", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


KEY = {"rh": "RH", "wb": "T_wb", "dp": "T_dp"}


@pytest.mark.parametrize(
    "case", REF["cases"][:-1], ids=lambda c: "-".join(str(x) for x in c["input"])
)
def test_matches_psychrolib(case: dict) -> None:
    kind, t, x, p = case["input"]
    ref = case["psychrolib"]
    val = x if kind == "rh" else [x, "C"]
    r = e(mode="state", T=[t, "C"], p=p, **{KEY[kind]: val})
    assert rel(r["W"], ref["W"], 3e-4), (r["W"], ref["W"])
    assert abs(r["RH"] - ref["RH"]) <= 2e-4
    assert abs(r["T_dp_C"] - ref["T_dp"]) <= 5e-3
    assert abs(r["T_wb_C"] - ref["T_wb"]) <= 1e-2
    assert rel(r["p_w"], ref["p_w"], 3e-4)
    assert rel(r["h"], ref["h"], 2e-4)
    assert rel(r["v"], ref["v"], 1e-4)
    assert rel(r["rho"], ref["rho"], 1e-4)
    assert rel(r["mu"], ref["mu"], 3e-4)


def test_altitude_pressure_matches_psychrolib() -> None:
    last = REF["cases"][-1]
    r = full(mode="state", T=[20, "C"], RH=0.5, altitude=[last["altitude_m"], "m"])
    assert rel(r["result"]["p"], last["p_std"], 1e-5)
    assert any("ISO 2533" in w for w in r["warnings"])


def test_inputs_are_consistent() -> None:
    base = e(mode="state", T=[30, "C"], RH=[55, "%"])
    for key, val in (
        ("T_wb", [base["T_wb_C"], "C"]),
        ("T_dp", [base["T_dp_C"], "C"]),
        ("h", base["h"]),
        ("W", [base["W"] * 1000, "g/kg"]),
    ):
        r = e(mode="state", T=[30, "C"], **{key: val})
        assert rel(r["W"], base["W"], 1e-9), key
    assert rel(base["RH"], 0.55, 1e-12)
    # dew point: p_ws(T_dp) = p_w
    assert rel(eng_steam.psat(base["T_dp"]) * 1e6, base["p_w"], 1e-12)


def test_wet_bulb_energy_balance() -> None:
    r = e(mode="state", T=[35, "C"], RH=0.3)
    t, w, ts = r["T_C"], r["W"], r["T_wb_C"]
    pws = eng_steam.psat(ts + 273.15) * 1e6
    ws = 0.621945 * pws / (101325 - pws)
    h_in = 1.006 * t + w * (2501 + 1.86 * t) + (ws - w) * 4.186 * ts
    h_out = 1.006 * ts + ws * (2501 + 1.86 * ts)
    assert abs(h_in - h_out) < 1e-9


def test_saturated_air_and_fog() -> None:
    r = e(mode="state", T=[20, "C"], RH=1.0)
    assert (
        abs(r["T_wb_C"] - 20) < 1e-9 and abs(r["T_dp_C"] - 20) < 1e-9 and abs(r["mu"] - 1) < 1e-12
    )
    assert "fog" in bad({"mode": "state", "T": [20, "C"], "W": 0.05})


def test_mixing_conserves_mass_and_energy() -> None:
    r = e(mode="mixing", T1=[32, "C"], RH1=0.6, m1=2.0, T2=[24, "C"], RH2=0.5, m2=6.0)
    assert rel(r["W3"], (2 * r["W1"] + 6 * r["W2"]) / 8, 1e-12)
    assert rel(r["h3"], (2 * r["h1"] + 6 * r["h2"]) / 8, 1e-12)
    assert 24 < r["T_C3"] < 32 and r["m_dry_air"] == 8.0
    v = e(
        mode="mixing", T1=[32, "C"], RH1=0.6, V1=[1, "m3/s"], T2=[24, "C"], RH2=0.5, V2=[3, "m3/s"]
    )
    assert rel(
        v["W3"], (v["W1"] / v["v1"] + 3 * v["W2"] / v["v2"]) / (1 / v["v1"] + 3 / v["v2"]), 1e-12
    )


def test_heating_and_cooling_coil() -> None:
    hot = e(mode="heating_cooling", T1=[10, "C"], RH1=0.7, T2=[30, "C"], m=1.5)
    assert hot["W2"] == hot["W1"] and hot["condensate_per_kg"] == 0
    assert rel(hot["Q"], 1.5 * (hot["h2"] - hot["h1"]), 1e-12)
    cold = e(mode="heating_cooling", T1=[30, "C"], RH1=0.6, T2=[14, "C"], RH2=0.95, m=2.0)
    assert cold["W2"] < cold["W1"] and abs(cold["RH2"] - 0.95) < 1e-12
    q = (cold["h2"] - cold["h1"]) + (cold["W1"] - cold["W2"]) * 4186 * 14
    assert rel(cold["q_per_kg"], q, 1e-12) and cold["Q"] < 0
    assert rel(cold["condensate"], 2.0 * (cold["W1"] - cold["W2"]), 1e-12)
    sat = full(mode="heating_cooling", T1=[30, "C"], RH1=0.6, T2=[10, "C"])
    assert abs(sat["result"]["RH2"] - 1) < 1e-9 and sat["warnings"]


def test_evaporative_cooling_keeps_wet_bulb() -> None:
    r = e(mode="evaporative_cooling", T1=[38, "C"], RH1=0.2, eta=0.8, m=1.0)
    assert abs(r["T_wb_C2"] - r["T_wb_C1"]) < 1e-9
    assert abs(r["T_C2"] - (38 - 0.8 * (38 - r["T_wb_C1"]))) < 1e-9
    assert rel(r["water_added"], r["W2"] - r["W1"], 1e-12) and r["W2"] > r["W1"]


def test_errors_and_cold_limits() -> None:
    assert "0 °C" in bad({"mode": "state", "T": [-5, "C"], "RH": 0.5})
    assert '[60, "%"]' in bad({"mode": "state", "T": [20, "C"], "RH": 60})
    assert "only one" in bad({"mode": "state", "T": [20, "C"], "RH": 0.5, "W": 0.01})
    assert "exceed" in bad({"mode": "state", "T": [20, "C"], "T_wb": [25, "C"]})
    assert "m1 and m2" in bad({"mode": "mixing", "T1": 300.0, "RH1": 0.5, "T2": 290.0, "RH2": 0.5})
    cold = full(mode="state", T=[5, "C"], RH=0.2)
    assert cold["result"]["T_dp"] is None and any("Dew point" in w for w in cold["warnings"])
