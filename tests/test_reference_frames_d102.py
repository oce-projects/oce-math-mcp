"""D102: eng_calc.reference_frames (Galilean and rotating frames, Euler's equations of a rigid body).

References: the motion seen from the other frame computed here with rotation matrices (Rodrigues) and finite
differences, the textbook component form of Euler's equations on principal axes, and the same equations in a
rotated basis (I' = Q I Q^T, omega' = Q omega).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from oce_math_mcp.envelope import Ctx
from oce_math_mcp.errors import ToolError
from oce_math_mcp.tools import eng_calc


def run(params: dict, ctx: Ctx | None = None) -> dict:
    return eng_calc.handle({"operation": "reference_frames", "params": params}, ctx or Ctx())


def close(a: Any, b: Any, rel: float = 1e-12, tol: float = 1e-12) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=rel, atol=tol))


def rodrigues(axis: Any, angle: float) -> Any:
    n = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


# ------------------------------------------------------------------ galilean


def test_galilean_constant_velocity_is_the_page_formulas() -> None:
    out = run(
        {
            "mode": "galilean",
            "r": [1, 2, 3],
            "v": [0.5, -1, 2],
            "a": [0, 0, -9.8],
            "V": [3, 0, -1],
            "t": 2.0,
            "m": 2.5,
        }
    )
    assert close(out["r_prime"], [1 + 6, 2, 3 - 2])
    assert close(out["v_prime"], [3.5, -1, 1])
    assert close(out["a_prime"], [0, 0, -9.8])
    assert close(out["F_app"], [0, 0, 0])
    assert close(out["F_prime"], out["F"])


def test_galilean_accelerated_frame_by_finite_differences() -> None:
    r0, v0, acc = np.array([1.0, -2.0]), np.array([0.3, 0.8]), np.array([0.1, -0.4])
    big_v, big_a, m = np.array([2.0, 1.0]), np.array([-0.5, 0.25]), 1.5

    def seen(t: float) -> dict:
        r = r0 + v0 * t + acc * t * t / 2
        v = v0 + acc * t
        return run(
            {
                "mode": "galilean",
                "r": r.tolist(),
                "v": v.tolist(),
                "a": acc.tolist(),
                "V": big_v.tolist(),
                "A": big_a.tolist(),
                "t": t,
                "m": m,
            }
        )

    t, h = 1.3, 1e-4
    mid = seen(t)
    lo, hi = seen(t - h), seen(t + h)
    dr = (np.array(hi["r_prime"]) - np.array(lo["r_prime"])) / (2 * h)
    dv = (np.array(hi["v_prime"]) - np.array(lo["v_prime"])) / (2 * h)
    assert close(dr, mid["v_prime"], 1e-8, 1e-8)
    assert close(dv, mid["a_prime"], 1e-8, 1e-8)
    assert close(mid["a_prime"], acc + big_a)
    assert close(mid["F_app"], -m * big_a)
    assert close(np.array(mid["F"]) - np.array(mid["F_app"]), mid["F_prime"])


def test_galilean_distance_between_two_bodies_is_invariant() -> None:
    common = {"mode": "galilean", "V": [1.0, -2.0, 0.5], "A": [0.2, 0.0, -0.1], "t": 3.0}
    one = run({**common, "r": [1.0, 1.0, 1.0]})
    two = run({**common, "r": [4.0, -3.0, 2.0]})
    d = np.linalg.norm(np.array(one["r_prime"]) - np.array(two["r_prime"]))
    assert d == pytest.approx(math.sqrt(9 + 16 + 1), rel=1e-14)


# ------------------------------------------------------------------ rotation about one axis


def test_rotation_about_a_fixed_axis() -> None:
    def seen(t: float) -> dict:
        th, w, al = 0.2 + 0.5 * t + 0.15 * t * t, 0.5 + 0.3 * t, 0.3
        return run(
            {
                "mode": "rotation",
                "theta": th,
                "omega": w,
                "alpha": al,
                "Omega": 1.2,
                "Lambda": -0.4,
                "t": t,
                "I": 2.0,
            }
        )

    t, h = 0.8, 1e-4
    mid, lo, hi = seen(t), seen(t - h), seen(t + h)
    assert (hi["theta_prime"] - lo["theta_prime"]) / (2 * h) == pytest.approx(
        mid["omega_prime"], rel=1e-8
    )
    assert (hi["omega_prime"] - lo["omega_prime"]) / (2 * h) == pytest.approx(
        mid["alpha_prime"], rel=1e-8
    )
    assert mid["alpha_prime"] == pytest.approx(0.3 - 0.4, rel=1e-14)
    assert mid["tau_app"] == pytest.approx(2.0 * 0.4, rel=1e-14)
    assert mid["tau"] - mid["tau_app"] == pytest.approx(mid["tau_prime"], rel=1e-14)
    still = run(
        {"mode": "rotation", "theta": 1.0, "omega": 0.5, "alpha": 0.25, "Omega": 2.0, "t": 3.0}
    )
    assert still["theta_prime"] == pytest.approx(1.0 + 6.0, rel=1e-14)
    assert still["omega_prime"] == pytest.approx(2.5, rel=1e-14)
    assert still["alpha_prime"] == pytest.approx(0.25, rel=1e-14)


# ------------------------------------------------------------------ rotating frame


def test_rate_of_a_vector_in_the_rotating_frame() -> None:
    omega, t0, d = np.array([0.3, -0.5, 1.1]), np.array([2.0, 1.0, -0.5]), np.array([0.4, 0.1, 0.7])
    out = run({"mode": "rotating", "Omega": omega.tolist(), "T": t0.tolist(), "dT_dt": d.tolist()})

    def comp(t: float) -> Any:  # components in the rotating frame of T(t) = T0 + D t
        return rodrigues(omega, -np.linalg.norm(omega) * t) @ (t0 + d * t)

    h = 1e-5
    fd = (comp(h) - comp(-h)) / (2 * h)
    assert close(out["dT_dt_prime"], fd, 1e-9, 1e-9)
    assert close(out["dT_dt_prime"], d - np.cross(omega, t0))


def test_fictitious_forces_of_a_free_particle() -> None:
    axis, w0, wdot, m = np.array([0.0, 0.6, 0.8]), 1.4, 0.3, 2.0
    r0, vel = np.array([1.0, -0.5, 2.0]), np.array([0.2, 0.9, -0.3])

    def rot(
        t: float,
    ) -> Any:  # inertial free motion seen from the frame turning by w0 t + wdot t^2 / 2
        return rodrigues(axis, -(w0 * t + wdot * t * t / 2)) @ (r0 + vel * t)

    h = 1e-4
    v_rot = (rot(h) - rot(-h)) / (2 * h)
    a_rot = (rot(h) - 2 * rot(0) + rot(-h)) / (h * h)
    out = run(
        {
            "mode": "rotating",
            "Omega": (w0 * axis).tolist(),
            "Omega_dot": (wdot * axis).tolist(),
            "r": r0.tolist(),
            "v": v_rot.tolist(),
            "m": m,
        }
    )
    total = np.array(out["a_coriolis"]) + np.array(out["a_centrifugal"]) + np.array(out["a_euler"])
    assert close(total, a_rot, 1e-6, 1e-6)
    assert close(out["v_inertial"], vel, 1e-8, 1e-8)
    assert close(out["F_coriolis"], m * np.array(out["a_coriolis"]))
    assert close(out["a_centrifugal"], -np.cross(w0 * axis, np.cross(w0 * axis, r0)))


# ------------------------------------------------------------------ Euler's equations


def components(i: Any, w: Any, al: Any) -> list[float]:
    """Textbook form on principal axes: tau1 = I1 alpha1 + (I3 - I2) omega2 omega3, and cyclic."""
    i1, i2, i3 = i
    return [
        i1 * al[0] + (i3 - i2) * w[1] * w[2],
        i2 * al[1] + (i1 - i3) * w[2] * w[0],
        i3 * al[2] + (i2 - i1) * w[0] * w[1],
    ]


def test_euler_principal_axes_textbook() -> None:
    i, w, al = [2.0, 3.0, 4.5], [0.7, -1.2, 2.0], [0.3, 0.1, -0.5]
    out = run({"mode": "euler_rigid_body", "I": i, "omega": w, "alpha": al})
    assert close(out["torque"], components(i, w, al))
    assert close(out["angular_momentum"], [2.0 * 0.7, 3.0 * -1.2, 4.5 * 2.0])
    assert out["kinetic_energy"] == pytest.approx(
        (2.0 * 0.49 + 3.0 * 1.44 + 4.5 * 4.0) / 2, rel=1e-14
    )
    back = run({"mode": "euler_rigid_body", "I": i, "omega": w, "torque": out["torque"]})
    assert close(back["alpha"], al)


def test_euler_full_tensor_in_a_rotated_basis() -> None:
    q = rodrigues([1.0, 2.0, -0.5], 0.9)
    i, w, al = np.diag([2.0, 3.0, 4.5]), np.array([0.7, -1.2, 2.0]), np.array([0.3, 0.1, -0.5])
    tau = np.array(components([2.0, 3.0, 4.5], w, al))
    out = run(
        {
            "mode": "euler_rigid_body",
            "I": (q @ i @ q.T).tolist(),
            "omega": (q @ w).tolist(),
            "alpha": (q @ al).tolist(),
        }
    )
    assert close(out["torque"], q @ tau)
    back = run(
        {
            "mode": "euler_rigid_body",
            "I": (q @ i @ q.T).tolist(),
            "omega": (q @ w).tolist(),
            "torque": (q @ tau).tolist(),
        }
    )
    assert close(back["alpha"], q @ al, 1e-11, 1e-12)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"mode": "euler_rigid_body", "I": [1, 2, 3], "omega": [1, 0, 0]}, "alpha"),
        (
            {
                "mode": "euler_rigid_body",
                "I": [1, 2, 3],
                "omega": [1, 0, 0],
                "alpha": [0, 0, 0],
                "torque": [0, 0, 0],
            },
            "alpha",
        ),
        (
            {
                "mode": "euler_rigid_body",
                "I": [[1, 0.5, 0], [0, 2, 0], [0, 0, 3]],
                "omega": [1, 0, 0],
                "alpha": [0, 0, 0],
            },
            "symmetric",
        ),
        (
            {"mode": "euler_rigid_body", "I": [1, -2, 3], "omega": [1, 0, 0], "alpha": [0, 0, 0]},
            "positive",
        ),
        (
            {"mode": "euler_rigid_body", "I": [1, 1, 3], "omega": [1, 0, 0], "alpha": [0, 0, 0]},
            "triangle",
        ),
        ({"mode": "euler_rigid_body", "I": [1, 2, 3], "omega": [1, 0], "alpha": [0, 0, 0]}, "3"),
        ({"mode": "rotating", "Omega": [0, 0, 1]}, "T"),
        ({"mode": "rotating", "Omega": [0, 0, 1], "T": [1, 0, 0]}, "dT_dt"),
        ({"mode": "rotating", "Omega": [0, 0, 1], "r": [1, 0, 0], "m": 2}, "v"),
        ({"mode": "galilean", "r": [1, 2], "V": [1, 2, 3]}, "size"),
        ({"mode": "galilean", "r": [1, 2], "V": [1, 2], "Omega": 3}, "Omega"),
        ({"mode": "rotation", "theta": [1, 2], "Omega": 1}, "number"),
    ],
)
def test_errors(params: dict, match: str) -> None:
    with pytest.raises(ToolError, match=match):
        run(params)
