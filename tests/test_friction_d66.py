"""Friction in simple machines (D66).

Checks independent of the server formulas:
- inclined plane: equilibrium of the block (normal force and push as unknowns) solved with numpy, Coulomb
  friction against the motion;
- leadscrew: the thread unrolled as an inclined plane with lead angle atan(l / (pi d_m)); for a trapezoidal
  thread the normal force of the flank found from the 3D force balance;
- efficiency against the work: useful work F l over the work of the torque 2 pi T in one turn;
- rolling resistance: moment balance of the wheel about the contact point.
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import call, ok


def fr(**params: object) -> dict:
    return ok("eng_calc", operation="friction", params=params)


def bad(params: dict) -> str:
    resp = call("eng_calc", operation="friction", params=params)
    assert not resp["ok"] and resp["error"]["code"] == "INVALID_ARGUMENT", resp
    return resp["error"]["message"]


def rel(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * abs(b)


def _push(w: float, alpha: float, mu: float, beta: float, up: bool) -> float:
    """Push P at angle beta above the slope that moves a block of weight W up (or down) the slope at constant
    speed. Unknowns [N, P]; friction mu N opposes the motion."""
    s = 1.0 if up else -1.0  # direction of motion along the slope
    # along the slope: P cos(beta) - W sin(alpha) - s mu N = 0 (s = +1 up)
    # normal to the slope: N + P sin(beta) - W cos(alpha) = 0
    a = np.array([[-s * mu, math.cos(beta)], [1.0, math.sin(beta)]])
    b = np.array([w * math.sin(alpha), w * math.cos(alpha)])
    n, p = np.linalg.solve(a, b)
    assert n > 0
    return float(p)


def test_inclined_plane_parallel_and_horizontal_push() -> None:
    w, mu = 1000.0, 0.25
    for theta_deg in (10.0, 30.0, 45.0):
        a = math.radians(theta_deg)
        r = fr(mode="inclined_plane", F_w=w, theta_deg=theta_deg, mu=mu)
        assert rel(r["F_i_up"], _push(w, a, mu, 0.0, True), 1e-12), r
        assert abs(r["F_i_down"] - _push(w, a, mu, 0.0, False)) <= 1e-12 * w
        assert rel(r["MA_up"], w / r["F_i_up"], 1e-13)
        h = fr(mode="inclined_plane", F_w=w, theta_deg=theta_deg, mu=mu, push="horizontal")
        assert rel(h["F_i_up"], _push(w, a, mu, -a, True), 1e-12), (
            h
        )  # horizontal = -theta from the slope
        assert rel(r["efficiency_up"], w * math.sin(a) / r["F_i_up"], 1e-12)
    repose = fr(mode="inclined_plane", F_w=1.0, theta_deg=5.0, mu=0.3)["angle_of_repose_deg"]
    assert rel(repose, math.degrees(math.atan(0.3)), 1e-13)
    assert fr(mode="inclined_plane", F_w=1.0, theta_deg=10.0, mu=0.3)["self_locking"] is True
    assert fr(mode="inclined_plane", F_w=1.0, theta_deg=20.0, mu=0.3)["self_locking"] is False
    # examples of the page "Inclined plane" (no friction): H 1 m on L 5 m gives MA 5; Minard plane 1804 m by 37.50 m, 48.1
    assert rel(
        fr(mode="inclined_plane", F_w=100.0, theta_deg=math.degrees(math.asin(1 / 5)), mu=0)[
            "MA_up"
        ],
        5.0,
        1e-13,
    )
    minard = fr(
        mode="inclined_plane", F_w=1.0, theta_deg=math.degrees(math.asin(37.5 / 1804)), mu=0
    )
    assert round(minard["MA_up"], 1) == 48.1


def _screw_torque(
    f: float, d_m: float, lead: float, mu: float, alpha_half: float, raise_: bool
) -> float:
    """Torque on a screw thread: force balance on a flank element in 3D. The flank normal leans by the lead
    angle lam along the helix and by the thread half-angle about the radial direction."""
    lam = math.atan(lead / (math.pi * d_m))
    # local axes: t = tangential (direction of the push), z = axial (load F along -z when raising)
    # unit vector of the helix (direction of motion of the nut relative to the screw when raising)
    v = np.array([math.cos(lam), math.sin(lam)]) * (1 if raise_ else -1)  # (t, z)
    # flank normal: in the (t, z) plane it is perpendicular to the helix; the thread angle tilts it out of
    # that plane, so its (t, z) part has length cos(alpha_n) with tan(alpha_n) = tan(alpha) cos(lam)
    alpha_n = math.atan(math.tan(alpha_half) * math.cos(lam))
    n_tz = np.array([-math.sin(lam), math.cos(lam)]) * math.cos(alpha_n)
    # unknowns [N, P]: P tangential push, N normal force; friction mu N against v; load F along -z
    # t: P + N n_t - mu N v_t = 0 ; z: -F + N n_z - mu N v_z = 0
    a = np.array([[n_tz[0] - mu * v[0], 1.0], [n_tz[1] - mu * v[1], 0.0]])
    _, p = np.linalg.solve(a, [0.0, f])
    return float(
        p * d_m / 2 if raise_ else -p * d_m / 2
    )  # lowering: the torque that holds the load back


def test_leadscrew_square_thread_against_unrolled_plane() -> None:
    for f, d_m, lead, mu in (
        (10e3, 0.032, 0.006, 0.15),
        (2e3, 0.02, 0.004, 0.08),
        (50e3, 0.06, 0.024, 0.1),
    ):
        r = fr(mode="leadscrew", F=f, d_m=d_m, l=lead, mu=mu)
        assert rel(r["T_raise"], _screw_torque(f, d_m, lead, mu, 0.0, True), 1e-12), r
        assert rel(r["T_lower"], _screw_torque(f, d_m, lead, mu, 0.0, False), 1e-12), r
        assert rel(
            r["efficiency"], f * lead / (2 * math.pi * r["T_raise"]), 1e-12
        )  # work in one turn
        assert r["self_locking"] is (mu > lead / (math.pi * d_m))
        assert rel(r["lead_angle_deg"], math.degrees(math.atan(lead / (math.pi * d_m))), 1e-13)


def test_leadscrew_trapezoidal_thread_and_collar() -> None:
    f, d_m, lead, mu = 10e3, 0.032, 0.006, 0.15
    for half in (14.5, 15.0):
        r = fr(mode="leadscrew", F=f, d_m=d_m, l=lead, mu=mu, alpha_deg=half)
        exact = _screw_torque(f, d_m, lead, mu, math.radians(half), True)
        assert rel(r["T_raise"], exact, 2e-4), (
            r,
            exact,
        )  # the page neglects the lead angle in the thread angle
        assert abs(r["T_raise"] - exact) > 1e-7 * exact  # and the gap is real, not rounding
        low = _screw_torque(f, d_m, lead, mu, math.radians(half), False)
        assert rel(r["T_lower"], low, 3e-4), (r, low)
        assert rel(r["efficiency"], f * lead / (2 * math.pi * r["T_raise"]), 1e-12)
        assert any(
            "lead angle" in w
            for w in call(
                "eng_calc",
                operation="friction",
                params={
                    "mode": "leadscrew",
                    "F": f,
                    "d_m": d_m,
                    "l": lead,
                    "mu": mu,
                    "alpha_deg": half,
                },
            )["warnings"]
        )
    c = fr(mode="leadscrew", F=f, d_m=d_m, l=lead, mu=mu, mu_c=0.12, d_c=0.05)
    assert rel(c["T_c"], f * 0.12 * 0.05 / 2, 1e-13) and rel(
        c["T_raise_total"], c["T_raise"] + c["T_c"], 1e-13
    )
    steep = fr(
        mode="leadscrew", F=f, d_m=0.02, l=0.02, mu=0.1
    )  # lead angle 17.7 degrees: it runs back
    assert steep["self_locking"] is False and steep["T_lower"] < 0


def test_rolling_resistance_and_moment_balance() -> None:
    assert rel(
        fr(mode="rolling_resistance", N=1000.0, C_rr=0.01)["F_r"], 10.0, 1e-13
    )  # page: 1000 lb -> 10 lb
    n, b, r = 5000.0, 0.0005, 0.35
    out = fr(mode="rolling_resistance", N=n, b=b, r=r)
    # moment about the contact point: the normal force acts b ahead of the centre, the push at height r
    assert rel(out["F_r"] * r, n * b, 1e-13) and rel(out["C_rr"], b / r, 1e-13)
    assert "Keys not used" in bad(
        {"mode": "rolling_resistance", "N": 1.0, "C_rr": 0.01, "b": 0.001}
    )
