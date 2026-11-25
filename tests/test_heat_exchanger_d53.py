"""e-NTU exchangers, fins, lumped-capacitance transient (D53).

Sources: Wikipedia «NTU method», «Fin (extended surface)», «Biot number».
Checks independent of the module formulas:
- parallel flow and counterflow: differential equations solved with solve_ivp and solve_bvp;
- crossflow: grid of n x n cells with trapezoidal exchange (mixed fluid = mean after
  each column); measured (script misura_d53.py): unmixed 1.1e-5 at n = 100, one fluid
  mixed with Richardson extrapolation from n = 100 and 200 at most 4.9e-6;
- both mixed: model of the mean temperatures solved as a 2x2 linear system;
- cross-check with the operation `heat_transfer` mode `lmtd` (Q = U A LMTD);
- fins: solve_bvp of the equation theta'' = m^2 theta with the 4 tip conditions, and the energy
  balance integrated with quad;
- transient: solve_ivp of rho c V dT/dt = -h A (T - T_inf).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, solve_bvp, solve_ivp

from tests.conftest import call, ok


def hx(**params: object) -> dict:
    return ok("eng_calc", operation="heat_exchanger_ntu", params=params)


def fin(**params: object) -> dict:
    return ok("eng_calc", operation="fin_heat_transfer", params=params)


def lump(**params: object) -> dict:
    return ok("eng_calc", operation="lumped_transient", params=params)


def full(operation: str, **params: object) -> dict:
    resp = call("eng_calc", operation=operation, params=params)
    assert resp["ok"], resp
    return resp


def bad(operation: str, params: dict) -> str:
    resp = call("eng_calc", operation=operation, params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


# ------------------------------------------------------------------ independent numerical models


def eps_parallel_ivp(N: float, cr: float) -> float:
    def f(_x: float, y: list[float]) -> list[float]:
        d = y[0] - y[1]
        return [-N * d, N * cr * d]

    sol = solve_ivp(f, (0, 1), [1.0, 0.0], rtol=1e-12, atol=1e-14)
    return 1 - float(sol.y[0, -1])


def eps_counter_bvp(N: float, cr: float) -> float:
    def f(_x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = y[0] - y[1]
        return np.vstack([-N * d, -N * cr * d])

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.array([ya[0] - 1.0, yb[1]])

    x = np.linspace(0, 1, 50)
    sol = solve_bvp(f, bc, x, np.vstack([1 - 0.5 * x, 0.5 * (1 - x)]), tol=1e-10, max_nodes=100000)
    assert sol.success
    return 1 - float(sol.sol(1.0)[0])


def cross_grid(cx: float, cy: float, ua: float, x_mixed: bool, n: int) -> float:
    """Heat with Tx_in = 1 and Ty_in = 0; fluid x flows along the columns, one row per cell."""
    u = ua / (n * n)
    chx, chy = cx / n, cy / n
    den = 1 + u / (2 * chx) + u / (2 * chy)
    tx = [1.0] * n
    q_tot = 0.0
    for _i in range(n):
        ty = 0.0
        for j in range(n):
            q = u * (tx[j] - ty) / den
            tx[j] -= q / chx
            ty += q / chy
            q_tot += q
        if x_mixed:
            tx = [sum(tx) / n] * n
    return q_tot


def eps_both_mixed_model(N: float, cr: float) -> float:
    a = (1 - math.exp(-N)) / N
    b = (1 - math.exp(-N * cr)) / (N * cr)
    _th_bar, tc_bar = np.linalg.solve(np.array([[1.0, -(1 - a)], [-(1 - b), 1.0]]), [a, 0.0])
    return 1 - (tc_bar + (1 - tc_bar) * math.exp(-N))


# ------------------------------------------------------------------ e-NTU


def test_parallel_and_counter_against_ode() -> None:
    for N in (0.1, 1.0, 4.0):
        for cr in (0.3, 0.8, 1.0):
            par = hx(mode="effectiveness", flow="parallel", NTU=N, Cr=cr)["epsilon"]
            cnt = hx(mode="effectiveness", flow="counter", NTU=N, Cr=cr)["epsilon"]
            assert rel(par, eps_parallel_ivp(N, cr), 1e-9)
            assert rel(cnt, eps_counter_bvp(N, cr), 1e-8)


def test_counter_balanced_and_single_stream() -> None:
    assert rel(hx(mode="effectiveness", flow="counter", NTU=3.0, Cr=1)["epsilon"], 0.75, 1e-15)
    near = hx(mode="effectiveness", flow="counter", NTU=3.0, Cr=1 - 1e-10)["epsilon"]
    assert rel(near, 0.75, 1e-8)  # stable form near Cr = 1
    for flow, extra in (
        ("parallel", {}),
        ("counter", {}),
        ("crossflow_unmixed", {}),
        ("crossflow_mixed", {}),
        ("crossflow_one_mixed", {"mixed": "cmin"}),
        ("crossflow_one_mixed", {"mixed": "cmax"}),
    ):
        r = hx(mode="effectiveness", flow=flow, NTU=2.0, Cr=0, **extra)
        assert rel(r["epsilon"], 1 - math.exp(-2.0), 1e-14), flow


def test_crossflow_unmixed_against_grid() -> None:
    for N, cr in ((0.5, 0.25), (1.0, 0.75), (3.0, 1.0)):
        r = hx(mode="effectiveness", flow="crossflow_unmixed", NTU=N, Cr=cr)
        grid = cross_grid(1.0, 1.0 / cr, N, False, 100)
        assert rel(r["epsilon"], grid, 5e-5)
    # large NTU: the series converges and the effectiveness grows towards 1
    e = [
        hx(mode="effectiveness", flow="crossflow_unmixed", NTU=n, Cr=1.0)["epsilon"]
        for n in (5.0, 20.0, 100.0)
    ]
    assert e[0] < e[1] < e[2] < 1


def test_crossflow_one_mixed_against_grid() -> None:
    for N, cr in ((0.5, 0.25), (1.0, 0.75), (3.0, 1.0)):
        for mixed in ("cmin", "cmax"):
            r = hx(mode="effectiveness", flow="crossflow_one_mixed", mixed=mixed, NTU=N, Cr=cr)
            cx, cy = (1.0, 1.0 / cr) if mixed == "cmin" else (1.0 / cr, 1.0)
            g1, g2 = cross_grid(cx, cy, N, True, 100), cross_grid(cx, cy, N, True, 200)
            assert rel(r["epsilon"], 2 * g2 - g1, 5e-5), (mixed, N, cr)


def test_crossflow_both_mixed_model_and_two_roots() -> None:
    for N in (0.2, 1.0, 5.0):
        for cr in (0.3, 1.0):
            r = hx(mode="effectiveness", flow="crossflow_mixed", NTU=N, Cr=cr)
            assert rel(r["epsilon"], eps_both_mixed_model(N, cr), 1e-12)
    # the effectiveness has a maximum: below the maximum and above the limit 1/(1+Cr) there are two NTU
    resp = full("heat_exchanger_ntu", mode="ntu", flow="crossflow_mixed", epsilon=0.55, Cr=1.0)
    r = resp["result"]
    assert r["NTU"] < r["NTU_alt"]
    for n in (r["NTU"], r["NTU_alt"]):
        assert rel(
            hx(mode="effectiveness", flow="crossflow_mixed", NTU=n, Cr=1.0)["epsilon"], 0.55, 1e-9
        )
    assert any("two" in w for w in resp["warnings"])
    assert "maximum" in bad(
        "heat_exchanger_ntu", {"mode": "ntu", "flow": "crossflow_mixed", "epsilon": 0.6, "Cr": 1.0}
    )


def test_inverse_round_trip() -> None:
    cases = [
        ("parallel", {}, 0.4),
        ("counter", {}, 0.9),
        ("crossflow_unmixed", {}, 0.7),
        ("crossflow_one_mixed", {"mixed": "cmin"}, 0.6),
        ("crossflow_one_mixed", {"mixed": "cmax"}, 0.6),
    ]
    for flow, extra, eps in cases:
        for cr in (0.0, 0.5, 1.0):
            n = hx(mode="ntu", flow=flow, epsilon=eps, Cr=cr, **extra)["NTU"]
            back = hx(mode="effectiveness", flow=flow, NTU=n, Cr=cr, **extra)["epsilon"]
            assert rel(back, eps, 1e-9), (flow, cr)
    assert "not reachable" in bad(
        "heat_exchanger_ntu", {"mode": "ntu", "flow": "parallel", "epsilon": 0.6, "Cr": 1.0}
    )
    # bug found on 24/11: the doubling jumped from 256 to 512 and rejected NTU between 256 and 500
    n = hx(mode="ntu", flow="crossflow_unmixed", epsilon=0.97, Cr=1.0)["NTU"]
    assert 256 < n < 500
    assert "500" in bad(
        "heat_exchanger_ntu",
        {"mode": "ntu", "flow": "crossflow_unmixed", "epsilon": 0.99, "Cr": 1.0},
    )


def test_rating_energy_balance_and_lmtd() -> None:
    r = hx(
        mode="rating",
        flow="counter",
        m_h=[2.0, "kg/s"],
        cp_h=[4.18, "kJ/(kg*K)"],
        m_c=3.0,
        cp_c=4180.0,
        Th_in=[90, "C"],
        Tc_in=[20, "C"],
        U=1500.0,
        A=5.0,
    )
    assert rel(r["C_h"], 8360.0, 1e-12) and rel(r["Cr"], 2 / 3, 1e-12)
    assert rel(r["NTU"], 7500 / 8360, 1e-12)
    assert rel(r["Q"], r["C_h"] * (363.15 - r["Th_out"]), 1e-12)
    assert rel(r["Q"], r["C_c"] * (r["Tc_out"] - 293.15), 1e-12)
    lm = ok(
        "eng_calc",
        operation="heat_transfer",
        params={
            "mode": "lmtd",
            "flow": "counter",
            "Th_in": 363.15,
            "Th_out": r["Th_out"],
            "Tc_in": 293.15,
            "Tc_out": r["Tc_out"],
            "U": 1500.0,
            "A": 5.0,
        },
    )
    assert rel(lm["Q"], r["Q"], 1e-9)


def test_rating_crossflow_mixed_side_and_phase_change() -> None:
    base = {"Th_in": 400.0, "Tc_in": 300.0, "UA": 2000.0}
    hot_mixed = hx(
        mode="rating", flow="crossflow_one_mixed", mixed="hot", C_h=1000.0, C_c=3000.0, **base
    )
    cmin = hx(mode="effectiveness", flow="crossflow_one_mixed", mixed="cmin", NTU=2.0, Cr=1 / 3)
    assert rel(hot_mixed["epsilon"], cmin["epsilon"], 1e-14)
    cold_mixed = hx(
        mode="rating", flow="crossflow_one_mixed", mixed="cold", C_h=1000.0, C_c=3000.0, **base
    )
    cmax = hx(mode="effectiveness", flow="crossflow_one_mixed", mixed="cmax", NTU=2.0, Cr=1 / 3)
    assert rel(cold_mixed["epsilon"], cmax["epsilon"], 1e-14)
    # condenser: the hot fluid changes phase, Cr = 0 and its temperature stays constant
    cond = hx(
        mode="rating",
        flow="counter",
        phase_change="hot",
        C_c=500.0,
        Th_in=373.15,
        Tc_in=293.15,
        UA=1000.0,
    )
    assert cond["Cr"] == 0 and cond["Th_out"] == 373.15
    assert rel(cond["Q"], (1 - math.exp(-2.0)) * 500 * 80, 1e-12)


def test_sizing_round_trip() -> None:
    common = {"flow": "counter", "C_h": 1000.0, "C_c": 1500.0, "Th_in": 400.0, "Tc_in": 300.0}
    s = hx(mode="sizing", Th_out=340.0, U=250.0, **common)
    assert rel(s["Q"], 60000.0, 1e-12) and rel(s["A"], s["UA"] / 250.0, 1e-12)
    back = hx(mode="rating", UA=s["UA"], **common)
    assert rel(back["Th_out"], 340.0, 1e-12)
    s2 = hx(mode="sizing", Q=[60, "kW"], **common)
    assert rel(s2["UA"], s["UA"], 1e-12)


def test_hx_errors() -> None:
    assert "Cr" in bad(
        "heat_exchanger_ntu", {"mode": "effectiveness", "flow": "counter", "NTU": 1.0, "Cr": 1.2}
    )
    assert "mixed" in bad(
        "heat_exchanger_ntu",
        {"mode": "effectiveness", "flow": "crossflow_one_mixed", "NTU": 1.0, "Cr": 0.5},
    )
    assert "Th_in" in bad(
        "heat_exchanger_ntu",
        {
            "mode": "rating",
            "flow": "counter",
            "C_h": 1.0,
            "C_c": 1.0,
            "Th_in": 300.0,
            "Tc_in": 350.0,
            "UA": 1.0,
        },
    )
    assert "C_h" in bad(
        "heat_exchanger_ntu",
        {
            "mode": "rating",
            "flow": "counter",
            "C_h": 1.0,
            "m_h": 1.0,
            "cp_h": 1.0,
            "C_c": 1.0,
            "Th_in": 400.0,
            "Tc_in": 300.0,
            "UA": 1.0,
        },
    )


# ------------------------------------------------------------------ fins


def fin_bvp(m: float, L: float, tip: str, r: float = 0.0, thl: float = 0.0) -> tuple[float, float]:
    """(-theta'(0), theta(L/2)) with theta(0) = 1, solved with solve_bvp."""

    def f(_x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.vstack([y[1], m * m * y[0]])

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        if tip == "convective":
            return np.array([ya[0] - 1, yb[1] + r * m * yb[0]])  # h theta = -k theta'
        if tip == "adiabatic":
            return np.array([ya[0] - 1, yb[1]])
        return np.array([ya[0] - 1, yb[0] - thl])

    x = np.linspace(0, L, 200)
    sol = solve_bvp(
        f, bc, x, np.vstack([np.ones_like(x), np.zeros_like(x)]), tol=1e-10, max_nodes=200000
    )
    assert sol.success
    return -float(sol.sol(0.0)[1]), float(sol.sol(L / 2)[0])


def test_fin_tips_against_bvp() -> None:
    h, k, D, L = 25.0, 200.0, 0.005, 0.05
    P, Ac = math.pi * D, math.pi * D * D / 4
    m = math.sqrt(h * P / (k * Ac))
    base = {
        "shape": "pin",
        "D": D,
        "L": L,
        "h": h,
        "k": k,
        "T_b": [100, "C"],
        "T_inf": [25, "C"],
        "x": L / 2,
    }
    for tip, extra, kw in (
        ("convective", {}, {"r": h / (m * k)}),
        ("adiabatic", {}, {}),
        ("fixed_T", {"T_L": [40, "C"]}, {"thl": 15 / 75}),
    ):
        r = fin(tip=tip, **base, **extra)
        dth0, th_mid = fin_bvp(m, L, tip, **kw)
        assert rel(r["m"], m, 1e-12) and rel(r["P"], P, 1e-12) and rel(r["A_c"], Ac, 1e-12)
        assert rel(r["Q_f"], k * Ac * dth0 * 75, 1e-7), tip
        assert rel(r["T_x"] - 298.15, th_mid * 75, 1e-7), tip


def test_fin_energy_balance_and_efficiency() -> None:
    h, k, w, t, L = 40.0, 180.0, 0.1, 0.002, 0.03
    r = fin(tip="convective", shape="rectangular", w=w, t=t, L=L, h=h, k=k, T_b=380.0, T_inf=300.0)
    P, Ac = 2 * (w + t), w * t
    m = math.sqrt(h * P / (k * Ac))
    ratio = h / (m * k)

    def theta(x: float) -> float:
        return (
            80
            * (math.cosh(m * (L - x)) + ratio * math.sinh(m * (L - x)))
            / (math.cosh(m * L) + ratio * math.sinh(m * L))
        )

    lateral = quad(lambda x: h * P * theta(x), 0, L, epsabs=1e-13)[0]
    assert rel(r["Q_f"], lateral + h * Ac * theta(L), 1e-10)  # balance: sides + tip
    assert rel(r["A_f"], P * L + Ac, 1e-12) and rel(
        r["eta_f"], r["Q_f"] / (h * r["A_f"] * 80), 1e-12
    )
    assert rel(r["eps_f"], r["Q_f"] / (h * Ac * 80), 1e-12)
    adi = fin(tip="adiabatic", shape="rectangular", w=w, t=t, L=L, h=h, k=k, T_b=380.0, T_inf=300.0)
    assert rel(
        adi["eta_f"], math.tanh(m * L) / (m * L), 1e-12
    )  # algebraic identity of the adiabatic tip
    inf = fin(tip="infinite", shape="custom", P=P, A_c=Ac, h=h, k=k, T_b=380.0, T_inf=300.0)
    long = fin(
        tip="adiabatic", shape="custom", P=P, A_c=Ac, L=50.0, h=h, k=k, T_b=380.0, T_inf=300.0
    )
    assert rel(inf["Q_f"], long["Q_f"], 1e-12) and "eta_f" not in inf


def test_fin_array_overall_efficiency() -> None:
    r = fin(
        tip="adiabatic",
        shape="pin",
        D=0.004,
        L=0.04,
        h=30.0,
        k=170.0,
        T_b=360.0,
        T_inf=300.0,
        n_fins=100,
        A_base=0.01,
    )
    a_t = 100 * r["A_f"] + 0.01 - 100 * r["A_c"]
    assert rel(r["A_t"], a_t, 1e-12)
    assert rel(r["eta_o"], 1 - 100 * r["A_f"] / a_t * (1 - r["eta_f"]), 1e-12)
    assert rel(r["Q_total"], r["eta_o"] * 30.0 * a_t * 60.0, 1e-12)
    assert "A_base" in bad(
        "fin_heat_transfer",
        {
            "tip": "adiabatic",
            "shape": "pin",
            "D": 0.02,
            "L": 0.04,
            "h": 30.0,
            "k": 170.0,
            "T_b": 360.0,
            "T_inf": 300.0,
            "n_fins": 100,
            "A_base": 0.01,
        },
    )


def test_fin_errors() -> None:
    assert "integer" in bad(
        "fin_heat_transfer",
        {
            "tip": "adiabatic",
            "shape": "pin",
            "D": 0.004,
            "L": 0.04,
            "h": 30.0,
            "k": 170.0,
            "T_b": 360.0,
            "T_inf": 300.0,
            "n_fins": 2.5,
            "A_base": 0.01,
        },
    )
    assert "T_L" in bad(
        "fin_heat_transfer",
        {
            "tip": "fixed_T",
            "shape": "pin",
            "D": 0.01,
            "L": 0.05,
            "h": 10.0,
            "k": 50.0,
            "T_b": 350.0,
            "T_inf": 300.0,
        },
    )
    assert "L" in bad(
        "fin_heat_transfer",
        {
            "tip": "infinite",
            "shape": "pin",
            "D": 0.01,
            "L": 0.05,
            "h": 10.0,
            "k": 50.0,
            "T_b": 350.0,
            "T_inf": 300.0,
        },
    )
    assert "equal" in bad(
        "fin_heat_transfer",
        {
            "tip": "adiabatic",
            "shape": "pin",
            "D": 0.01,
            "L": 0.05,
            "h": 10.0,
            "k": 50.0,
            "T_b": 300.0,
            "T_inf": 300.0,
        },
    )


# ------------------------------------------------------------------ transient


def test_lumped_against_ivp_and_inverse() -> None:
    p = {"shape": "sphere", "D": 0.01, "rho": 8933.0, "cp": 385.0, "k": 401.0, "h": 50.0}
    r = lump(T_0=[200, "C"], T_inf=[20, "C"], t=[2, "min"], **p)
    V, A = math.pi * 0.01**3 / 6, math.pi * 0.01**2
    assert rel(r["L_c"], 0.01 / 6, 1e-12) and rel(r["Bi"], 50 * 0.01 / 6 / 401, 1e-12)
    sol = solve_ivp(
        lambda _t, y: [-50 * A * (y[0] - 293.15) / (8933 * 385 * V)],
        (0, 120),
        [473.15],
        rtol=1e-12,
        atol=1e-12,
    )
    assert rel(r["T"], float(sol.y[0, -1]), 1e-9)
    assert rel(r["Q"], 8933 * 385 * V * (473.15 - r["T"]), 1e-12)
    back = lump(T_0=[200, "C"], T_inf=[20, "C"], T=r["T"], **p)
    assert rel(back["t"], 120.0, 1e-9)


def test_lumped_biot_warning_and_errors() -> None:
    resp = full(
        "lumped_transient",
        V=1e-3,
        A=0.06,
        rho=2700.0,
        cp=900.0,
        k=0.5,
        h=100.0,
        T_0=400.0,
        T_inf=300.0,
        t=60.0,
    )
    assert any("0.1" in w for w in resp["warnings"])
    assert "between" in bad(
        "lumped_transient",
        {
            "V": 1e-3,
            "A": 0.06,
            "rho": 2700.0,
            "cp": 900.0,
            "h": 100.0,
            "T_0": 400.0,
            "T_inf": 300.0,
            "T": 250.0,
        },
    )
    assert "t" in bad(
        "lumped_transient",
        {
            "V": 1e-3,
            "A": 0.06,
            "rho": 2700.0,
            "cp": 900.0,
            "h": 100.0,
            "T_0": 400.0,
            "T_inf": 300.0,
        },
    )
