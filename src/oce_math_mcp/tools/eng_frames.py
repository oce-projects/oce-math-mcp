"""eng_calc: reference frames and Euler's equations of a rigid body (D102).

Source: Wikipedia "List of equations in classical mechanics", sections "Galilean frame transforms" and
"Euler's equations for rigid body dynamics", in the local offline library.
- Galilean frames: the primed frame moves with velocity -V (and acceleration -A) relative to the unprimed one,
  origins together at t = 0: r' = r + V t + A t^2 / 2, v' = v + V + A t, a' = a + A (a' = a for A = 0);
  F' = F - F_app with F = m a, F' = m a' and the apparent force F_app = -m A.
- Rotation about one fixed axis: theta' = theta + Omega t + Lambda t^2 / 2, omega' = omega + Omega + Lambda t,
  alpha' = alpha + Lambda (alpha' = alpha for Lambda = 0); tau' = tau - tau_app with tau = I alpha,
  tau' = I alpha' and tau_app = -I Lambda.
- Rotating frame (angular velocity Omega, its rate Omega_dot): dT'/dt = dT/dt - Omega x T for any vector T;
  a particle at r with velocity v in that frame has v_inertial = v + Omega x r and the apparent accelerations
  -2 Omega x v (Coriolis), -Omega x (Omega x r) (centrifugal), -Omega_dot x r (Euler).
- Euler's equations: I . alpha + omega x (I . omega) = tau (body axes), the rate of L = I omega seen from the
  body plus omega x L. I is the inertia tensor: 3 principal moments or a symmetric 3 x 3 matrix.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import choice, fmt_sig, need, only, op, qty, range_guard, units_for

_MODES: dict[str, set[str]] = {
    "galilean": {"r", "v", "a", "V", "A", "t", "m"},
    "rotation": {"theta", "omega", "alpha", "Omega", "Lambda", "t", "I"},
    "rotating": {"Omega", "Omega_dot", "T", "dT_dt", "r", "v", "m"},
    "euler_rigid_body": {"I", "omega", "alpha", "torque"},
}
_UNITS = {
    "r_prime": "m",
    "v_prime": "m/s",
    "a_prime": "m/s2",
    "F": "N",
    "F_prime": "N",
    "F_app": "N",
    "theta_prime": "rad",
    "omega_prime": "rad/s",
    "alpha_prime": "rad/s2",
    "tau": "N m",
    "tau_prime": "N m",
    "tau_app": "N m",
    "v_inertial": "m/s",
    "a_coriolis": "m/s2",
    "a_centrifugal": "m/s2",
    "a_euler": "m/s2",
    "F_coriolis": "N",
    "F_centrifugal": "N",
    "F_euler": "N",
    "torque": "N m",
    "alpha": "rad/s2",
    "angular_momentum": "kg m2/s",
    "kinetic_energy": "J",
    "gyroscopic_torque": "N m",
}


@op("reference_frames", ("mode", *sorted(set().union(*_MODES.values()))))
@range_guard
def _frames(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    mode = choice(p, "mode", tuple(_MODES))
    only(p, mode, _MODES[mode])
    out = {
        "galilean": _galilean,
        "rotation": _rotation,
        "rotating": _rotating,
        "euler_rigid_body": _euler,
    }[mode](p, ctx)
    ctx.units = units_for(out, _UNITS)
    return out


def _vec(p: dict[str, Any], key: str, size: int | None = None) -> Any:
    """A vector of SI numbers: a number (1D) or a list of 1 to 3 numbers; size 3 when given."""
    raw = p.get(key)
    vals = raw if isinstance(raw, list) else [raw]
    if not 1 <= len(vals) <= 3 or (size is not None and len(vals) != size):
        want = f"{size} numbers" if size else "a number or a list of 1 to 3 numbers"
        raise ToolError("INVALID_ARGUMENT", f"params.{key}: {want}.")
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ToolError("INVALID_ARGUMENT", f"params.{key}: finite numbers only.")
    return np.array(vals, dtype=float)


def _real(p: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Signed pure number; default when missing."""
    v = qty(p, key, None, need=False, positive=False, default=default)
    return float(default if v is None else v)


def _galilean(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    given = [k for k in ("r", "v", "a", "V", "A") if p.get(k) is not None]
    vecs = {k: _vec(p, k) for k in given}
    sizes = {len(v) for v in vecs.values()}
    if len(sizes) > 1:
        raise ToolError(
            "INVALID_ARGUMENT", "r, v, a, V and A need the same size (1, 2 or 3 numbers)."
        )
    n = sizes.pop() if sizes else 1
    zero = np.zeros(n)
    r, v, a, big_v, big_a = (vecs.get(k, zero) for k in ("r", "v", "a", "V", "A"))
    t = _real(p, "t")
    out: dict[str, Any] = {
        "r_prime": (r + big_v * t + big_a * t * t / 2).tolist(),
        "v_prime": (v + big_v + big_a * t).tolist(),
        "a_prime": (a + big_a).tolist(),
    }
    ctx.formula = (
        "r' = r + V t + A t^2 / 2, v' = v + V + A t, a' = a + A (the primed frame moves with -V, -A; "
        "A = 0: r' = r + V t, v' = v + V, a' = a)"
    )
    if p.get("m") is not None:
        m = need(p, "m", "mass")
        out["F"] = (m * a).tolist()
        out["F_app"] = (-m * big_a).tolist()
        out["F_prime"] = (m * (a + big_a)).tolist()
        ctx.formula += "; F = m a, F_app = -m A, F' = F - F_app = m a'"
    return out


def _rotation(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    th, w, al = _real(p, "theta"), _real(p, "omega"), _real(p, "alpha")
    big_w, lam, t = _real(p, "Omega"), _real(p, "Lambda"), _real(p, "t")
    out: dict[str, Any] = {
        "theta_prime": th + big_w * t + lam * t * t / 2,
        "omega_prime": w + big_w + lam * t,
        "alpha_prime": al + lam,
    }
    ctx.formula = (
        "about one fixed axis: theta' = theta + Omega t + Lambda t^2 / 2, omega' = omega + Omega + Lambda t, "
        "alpha' = alpha + Lambda"
    )
    if p.get("I") is not None:
        inertia = need(p, "I", None)
        out["tau"] = inertia * al
        out["tau_app"] = -inertia * lam
        out["tau_prime"] = inertia * (al + lam)
        ctx.formula += "; tau = I alpha, tau_app = -I Lambda, tau' = tau - tau_app = I alpha'"
    return out


def _rotating(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    big_w = _vec(p, "Omega", 3)
    w_dot = _vec(p, "Omega_dot", 3) if p.get("Omega_dot") is not None else np.zeros(3)
    has_t, has_r = p.get("T") is not None, p.get("r") is not None
    if not has_t and not has_r:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Mode rotating needs T with dT_dt (rate of a vector), or r (position in the rotating frame, "
            "with v and m for the forces).",
            '{"mode": "rotating", "Omega": [0, 0, 7.29e-5], "r": [6.4e6, 0, 0], "v": [0, 10, 0], "m": 1}',
        )
    out: dict[str, Any] = {}
    formulas = []
    if has_t:
        if p.get("dT_dt") is None:
            raise ToolError(
                "INVALID_ARGUMENT",
                "params.dT_dt (rate of T in the inertial frame) is needed with T.",
            )
        vt, dt = _vec(p, "T", 3), _vec(p, "dT_dt", 3)
        out["dT_dt_prime"] = (dt - np.cross(big_w, vt)).tolist()
        formulas.append("dT'/dt = dT/dt - Omega x T")
    elif p.get("dT_dt") is not None:
        raise ToolError("INVALID_ARGUMENT", "params.dT_dt goes with params.T.")
    if has_r:
        r = _vec(p, "r", 3)
        v = _vec(p, "v", 3) if p.get("v") is not None else None
        if p.get("m") is not None and v is None:
            raise ToolError(
                "INVALID_ARGUMENT", "The forces need params.v (velocity in the rotating frame)."
            )
        acc = {
            "a_centrifugal": -np.cross(big_w, np.cross(big_w, r)),
            "a_euler": -np.cross(w_dot, r),
        }
        formulas.append("a_centrifugal = -Omega x (Omega x r), a_euler = -Omega_dot x r")
        if v is not None:
            acc["a_coriolis"] = -2 * np.cross(big_w, v)
            out["v_inertial"] = (v + np.cross(big_w, r)).tolist()
            formulas.append("a_coriolis = -2 Omega x v, v_inertial = v + Omega x r")
        out |= {k: x.tolist() for k, x in acc.items()}
        if p.get("m") is not None:
            m = need(p, "m", "mass")
            for k, x in acc.items():
                out["F_" + k[2:]] = (m * x).tolist()
            formulas.append("F = m a for each apparent acceleration")
    elif p.get("v") is not None or p.get("m") is not None:
        raise ToolError("INVALID_ARGUMENT", "params.v and params.m go with params.r.")
    ctx.formula = "; ".join(formulas)
    return out


def _inertia(p: dict[str, Any]) -> Any:
    raw = p.get("I")
    if isinstance(raw, list) and raw and all(isinstance(row, list) for row in raw):
        if len(raw) != 3 or any(len(row) != 3 for row in raw):
            raise ToolError("INVALID_ARGUMENT", "params.I: 3 principal moments or a 3 x 3 matrix.")
        vals = [x for row in raw for x in row]
        if any(
            isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x)
            for x in vals
        ):
            raise ToolError("INVALID_ARGUMENT", "params.I: finite numbers only.")
        i = np.array(raw, dtype=float)
    else:
        i = np.diag(_vec(p, "I", 3))
    scale = float(np.max(np.abs(i)))
    if scale == 0 or not np.allclose(i, i.T, rtol=0, atol=1e-12 * scale):
        raise ToolError(
            "INVALID_ARGUMENT", "params.I must be symmetric (an inertia tensor), and not zero."
        )
    i = (i + i.T) / 2
    moments = np.linalg.eigvalsh(i)
    if moments[0] <= 1e-12 * scale:
        raise ToolError(
            "INVALID_ARGUMENT", "params.I must be positive definite (principal moments > 0)."
        )
    if moments[0] + moments[1] < moments[2] * (1 - 1e-12):
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.I breaks the triangle inequality of the principal moments (I1 + I2 >= I3): "
            "no rigid body has this tensor.",
        )
    return i


def _euler(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    i = _inertia(p)
    w = _vec(p, "omega", 3)
    has_a, has_tau = p.get("alpha") is not None, p.get("torque") is not None
    if has_a == has_tau:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Mode euler_rigid_body needs exactly one of alpha (gives the torque) and torque (gives alpha).",
            '{"mode": "euler_rigid_body", "I": [2, 3, 4.5], "omega": [0.7, -1.2, 2], "alpha": [0.3, 0.1, -0.5]}',
        )
    ang = i @ w
    gyro = np.cross(w, ang)
    out: dict[str, Any] = {}
    if has_a:
        al = _vec(p, "alpha", 3)
        tau = i @ al + gyro
        out["torque"] = tau.tolist()
    else:
        tau = _vec(p, "torque", 3)
        al = np.linalg.solve(i, tau - gyro)
        out["alpha"] = al.tolist()
    resid = i @ al + gyro - tau
    size = float(np.linalg.norm(i @ al)) + float(np.linalg.norm(gyro)) + float(np.linalg.norm(tau))
    if float(np.linalg.norm(resid)) > 1e-12 * max(size, 1e-300):
        ctx.warn(
            "The back-substitution of Euler's equations left a residual above 1e-12 of the terms."
        )
    out["angular_momentum"] = ang.tolist()
    out["kinetic_energy"] = float(w @ ang) / 2
    out["gyroscopic_torque"] = gyro.tolist()
    ctx.formula = (
        "I . alpha + omega x (I . omega) = tau (body axes); L = I . omega, K = omega . L / 2"
    )
    ctx.formatted = f"Euler's equations: |tau| = {fmt_sig(float(np.linalg.norm(tau)), 6)} N m."
    return out
