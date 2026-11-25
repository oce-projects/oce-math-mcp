"""geom3d_calc: 3D geometry, CAD, robotics and computations for 3D drawing (spec V3 section 11, D22).

Conventions: angles in degrees, lengths in the user's unit, quaternions [w, x, y, z],
cylindrical [r, theta_deg, z], spherical ISO 80000-2 [r, theta_deg polar, phi_deg azimuth].
Euler sequences: "XYZ" = fixed axes in the written order. Box with a corner at the origin;
cylinder and cone with the base centred at the origin and axis z.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .. import limits
from ..envelope import Ctx
from ..errors import NOT_FINITE, ToolError

Op = Callable[[dict[str, Any], Ctx], dict[str, Any]]
OPS: dict[str, Op] = {}
EPS = 1e-12


def op(name: str) -> Callable[[Op], Op]:
    def deco(fn: Op) -> Op:
        OPS[name] = fn
        return fn

    return deco


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    return clean(OPS[args["operation"]](args, ctx))


def clean(obj: Any) -> Any:
    """Rounds to 12 decimals and removes -0.0 (float noise)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return obj
        v = round(obj, 12)
        return 0.0 if v == 0 else v
    return obj


# ------------------------------------------------------------------ reading the data


def vec(raw: Any, name: str, n: int | None = 3) -> np.ndarray:
    if not isinstance(raw, list | tuple) or not raw:
        raise ToolError(
            "INVALID_ARGUMENT", f"'{name}' is needed as a list of numbers.", "[1, 0, 0]"
        )
    try:
        v = np.array([float(x) for x in raw], dtype=float)
    except (TypeError, ValueError):
        raise ToolError(
            "INVALID_ARGUMENT", f"'{name}' contains non-numeric values.", "[1, 0, 0]"
        ) from None
    if n is not None and len(v) != n:
        raise ToolError("INVALID_ARGUMENT", f"'{name}' must have {n} components.", "[1, 0, 0]")
    if not np.all(np.isfinite(v)):
        raise ToolError("INVALID_ARGUMENT", f"'{name}' contains non-finite values.")
    return v


def unit(v: np.ndarray, name: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        raise ToolError("INVALID_ARGUMENT", f"'{name}' is the zero vector.", "[0, 0, 1]")
    return v / n


def P(args: dict[str, Any]) -> dict[str, Any]:
    p = args.get("params") or {}
    if not isinstance(p, dict):
        raise ToolError(
            "INVALID_ARGUMENT",
            "'params' must be an object.",
            '{"shape": "box", "size": [10, 20, 30]}',
        )
    return p


def num(p: dict[str, Any], key: str, default: float | None = None, positive: bool = True) -> float:
    v = p.get(key, default)
    if v is None:
        raise ToolError("INVALID_ARGUMENT", f"Missing params.{key}.", f'{{"{key}": 10}}')
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be a number.") from None
    if positive and f <= 0:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be > 0.")
    return f


def count(v: Any, name: str, lo: int = 1, hi: int = limits.LIST_MAX) -> int:
    """An integer count in [lo, hi], checked before any list is built (D93).

    Before, int() truncated 5.9 holes to 5 and a pattern of 1e8 points was built before the size check.
    """
    if v is None:
        raise ToolError("INVALID_ARGUMENT", f"Missing {name}.")
    if isinstance(v, bool) or not isinstance(v, int | float) or not float(v).is_integer():
        raise ToolError("INVALID_ARGUMENT", f"{name} must be an integer, not {v}.")
    n = int(v)
    if n < lo:
        raise ToolError("INVALID_ARGUMENT", f"{name} must be >= {lo}.")
    if n > hi:
        raise ToolError("LIMIT_EXCEEDED", f"{name}: at most {hi}.")
    return n


def points(raw: Any, name: str, dim: int | None = None, min_count: int = 1) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) < min_count:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"'{name}' is needed with at least {min_count} points.",
            "[[0, 0], [1, 0], [1, 1]]",
        )
    if len(raw) > limits.LIST_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"At most {limits.LIST_MAX} points.")
    try:
        arr = np.array(raw, dtype=float)
    except (TypeError, ValueError):
        raise ToolError(
            "INVALID_ARGUMENT", f"'{name}': invalid points (same dimension for all)."
        ) from None
    if arr.ndim != 2 or (dim is not None and arr.shape[1] != dim) or arr.shape[1] not in (2, 3):
        raise ToolError(
            "INVALID_ARGUMENT",
            f"'{name}': {'2D or 3D' if dim is None else f'{dim}D'} points are needed.",
        )
    if not np.all(np.isfinite(arr)):
        raise ToolError("INVALID_ARGUMENT", f"'{name}' contains non-finite values.")
    return arr


# ------------------------------------------------------------------ rotations


def rot_axis(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rodrigues(axis: np.ndarray, deg: float) -> np.ndarray:
    k = unit(axis, "axis")
    a = math.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def euler_matrix(angles: np.ndarray, sequence: str) -> np.ndarray:
    seq = sequence.upper()
    if len(seq) != 3 or any(c not in "XYZ" for c in seq):
        raise ToolError(
            "INVALID_ARGUMENT", "params.sequence: three letters among X, Y, Z.", '"XYZ"'
        )
    R = np.eye(3)
    for ax, deg in zip(seq, angles, strict=True):
        R = rot_axis(ax, float(deg)) @ R  # fixed axes: each rotation is applied after
    return R


def _scipy_rot() -> Any:
    from scipy.spatial.transform import Rotation

    return Rotation


@op("vec_ops")
def _vec_ops(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a = vec(args.get("v1"), "v1", None)
    out: dict[str, Any] = {"norm_v1": float(np.linalg.norm(a)), "unit_v1": (unit(a, "v1")).tolist()}
    if args.get("v2"):
        b = vec(args["v2"], "v2", len(a))
        out.update(
            dot=float(a @ b),
            norm_v2=float(np.linalg.norm(b)),
            sum=(a + b).tolist(),
            difference=(a - b).tolist(),
        )
        if len(a) == 3:
            out["cross"] = np.cross(a, b).tolist()
        nb = float(b @ b)
        if nb > EPS:
            out["projection_v1_on_v2"] = (float(a @ b) / nb * b).tolist()
    ctx.formatted = f"|v1| = {out['norm_v1']:.6g}"
    return out


@op("angle_between")
def _angle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    a = vec(args.get("v1"), "v1", None)
    b = vec(args.get("v2"), "v2", len(a))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        raise ToolError("MATH_ERROR", "Angle undefined with a zero vector.")
    dot = float(a @ b)
    if len(a) == 3:
        ang = math.atan2(float(np.linalg.norm(np.cross(a, b))), dot)
    else:
        ang = math.acos(max(-1.0, min(1.0, dot / (na * nb))))
    ctx.step(f"Dot product v1·v2 = {dot:.12g}.")
    ctx.step(f"Norms: |v1| = {na:.12g}, |v2| = {nb:.12g}.")
    ctx.step(f"cos θ = v1·v2 / (|v1| |v2|) = {dot / (na * nb):.12g}.")
    ctx.step(f"θ = {math.degrees(ang):.12g}° = {ang:.12g} rad.")
    ctx.formatted = f"θ = {math.degrees(ang):.6g}°"
    return {"angle_deg": math.degrees(ang), "angle_rad": ang}


@op("rotate_point")
def _rotate(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    pt = vec(args.get("v1"), "v1")
    angles = vec(args.get("angles_deg"), "angles_deg", None)
    if args.get("axis"):
        R = rodrigues(vec(args["axis"], "axis"), float(angles[0]))
        ctx.step(f"Rotation of {angles[0]}° about the axis {args['axis']} (Rodrigues formula).")
    else:
        if len(angles) != 3:
            raise ToolError(
                "INVALID_ARGUMENT",
                "Without 'axis' three angles are needed in angles_deg.",
                "[0, 0, 90]",
            )
        R = euler_matrix(angles, str(p.get("sequence", "XYZ")))
        ctx.step(f"Fixed-axis rotations {p.get('sequence', 'XYZ')}: {angles.tolist()}°.")
    center = vec(p["center"], "params.center") if p.get("center") else np.zeros(3)
    out = R @ (pt - center) + center
    ctx.step(f"Rotated point: {out.round(12).tolist()}.")
    ctx.formatted = f"Rotated point: {out.round(6).tolist()}"
    return {"point": out.tolist(), "rotation_matrix": R.tolist()}


def _hom(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@op("transform_4x4")
def _transform(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    if p.get("matrices"):
        T = np.eye(4)
        for i, m in enumerate(p["matrices"]):
            M = np.array(m, dtype=float)
            if M.shape != (4, 4):
                raise ToolError("INVALID_ARGUMENT", f"params.matrices[{i}] is not 4x4.")
            T = T @ M
    else:
        angles = vec(args["angles_deg"], "angles_deg") if args.get("angles_deg") else np.zeros(3)
        t = vec(args["translation"], "translation") if args.get("translation") else np.zeros(3)
        T = _hom(euler_matrix(angles, str(p.get("sequence", "XYZ"))), t)
    inv = np.linalg.inv(T)
    if p.get("invert"):
        T, inv = inv, T
    out: dict[str, Any] = {"matrix": T.tolist(), "inverse": inv.tolist()}
    if args.get("v1"):
        x = T @ np.append(vec(args["v1"], "v1"), 1.0)
        out["point"] = (x[:3] / x[3]).tolist()
    ctx.formatted = "4x4 matrix computed"
    return out


@op("quat_euler")
def _quat(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    seq = str(p.get("sequence", "XYZ"))
    Rot = _scipy_rot()
    scipy_seq = seq.upper() if p.get("intrinsic") else seq.lower()
    if args.get("quat"):
        q = vec(args["quat"], "quat", 4)
        q = unit(q, "quat")
        r = Rot.from_quat([q[1], q[2], q[3], q[0]])
    elif args.get("angles_deg"):
        r = Rot.from_euler(scipy_seq, vec(args["angles_deg"], "angles_deg"), degrees=True)
    else:
        raise ToolError(
            "INVALID_ARGUMENT", "'quat' [w, x, y, z] or 'angles_deg' is needed.", "[10, 20, 30]"
        )
    x, y, z, w = r.as_quat()
    qq = np.array([w, x, y, z])
    if qq[0] < 0:
        qq = -qq
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ang = r.as_euler(scipy_seq, degrees=True)
    if caught:
        ctx.warn("Gimbal lock: the angles are not unique.")
    ctx.formatted = f"q = {np.round(qq, 9).tolist()}"
    return {
        "quat": qq.tolist(),
        "angles_deg": ang.tolist(),
        "rotation_matrix": r.as_matrix().tolist(),
        "sequence": seq,
    }


@op("dh_kinematics")
def _dh(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    table = args.get("dh_table")
    if not isinstance(table, list) or not table:
        raise ToolError(
            "INVALID_ARGUMENT",
            "'dh_table' is needed with rows [a, alpha_deg, d, theta_deg].",
            "[[1, 0, 0, 0], [1, 0, 0, 90]]",
        )
    T = np.eye(4)
    frames = []
    for i, row in enumerate(table):
        if not isinstance(row, list) or len(row) != 4:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"dh_table[{i}] must have 4 values [a, alpha_deg, d, theta_deg].",
            )
        a, alpha, d, theta = (float(v) for v in row)
        ct, st = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        ca, sa = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
        A = np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ]
        )
        T = T @ A
        frames.append(T[:3, 3].tolist())
        ctx.step(f"Joint {i + 1}: position {np.round(T[:3, 3], 9).tolist()}.")
    ctx.formatted = f"Final position: {np.round(T[:3, 3], 6).tolist()}"
    return {"matrix": T.tolist(), "position": T[:3, 3].tolist(), "frames": frames}


@op("point_plane_distance")
def _pp_dist(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = vec(args.get("v1"), "v1")
    p0 = vec(args.get("plane_point"), "plane_point")
    n = unit(vec(args.get("plane_normal"), "plane_normal"), "plane_normal")
    s = float((p - p0) @ n)
    ctx.formatted = f"Distance {abs(s):.6g}"
    return {"distance": abs(s), "signed_distance": s, "foot": (p - s * n).tolist()}


@op("line_plane_intersect")
def _lp(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = vec(args.get("v1"), "v1")
    d = unit(vec(args.get("v2"), "v2"), "v2")
    p0 = vec(args.get("plane_point"), "plane_point")
    n = unit(vec(args.get("plane_normal"), "plane_normal"), "plane_normal")
    den = float(d @ n)
    if abs(den) < 1e-12:
        contained = abs(float((p - p0) @ n)) < 1e-9
        ctx.formatted = "Line contained in the plane" if contained else "Line parallel to the plane"
        return {"intersection": None, "parallel": True, "contained": contained}
    t = float((p0 - p) @ n) / den
    pt = p + t * d
    ctx.formatted = f"Intersection {np.round(pt, 6).tolist()}"
    return {"intersection": pt.tolist(), "parallel": False, "contained": False, "t": t}


@op("point_line_distance")
def _pl(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    q = vec(args.get("v1"), "v1")
    a = vec(p.get("line_point"), "params.line_point")
    d = unit(vec(p.get("line_dir"), "params.line_dir"), "params.line_dir")
    foot = a + float((q - a) @ d) * d
    dist = float(np.linalg.norm(q - foot))
    ctx.formatted = f"Distance {dist:.6g}"
    return {"distance": dist, "foot": foot.tolist()}


@op("coord_convert")
def _coord(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    v = vec(args.get("v1"), "v1")
    src, dst = args.get("coord_from") or "cartesian", args.get("coord_to") or "cartesian"
    if src == "cylindrical":
        r, th, z = v
        x, y = r * math.cos(math.radians(th)), r * math.sin(math.radians(th))
    elif src == "spherical":
        r, th, ph = v
        x = r * math.sin(math.radians(th)) * math.cos(math.radians(ph))
        y = r * math.sin(math.radians(th)) * math.sin(math.radians(ph))
        z = r * math.cos(math.radians(th))
    else:
        x, y, z = v
    if dst == "cylindrical":
        out = [math.hypot(x, y), math.degrees(math.atan2(y, x)), z]
    elif dst == "spherical":
        r = math.sqrt(x * x + y * y + z * z)
        out = [r, math.degrees(math.acos(z / r)) if r > 0 else 0.0, math.degrees(math.atan2(y, x))]
    else:
        out = [x, y, z]
    ctx.formatted = f"{dst}: {np.round(out, 6).tolist()}"
    return {
        "coords": out,
        "convention": "cylindrical [r, theta_deg, z]; spherical ISO [r, polar theta, azimuth phi]",
    }


@op("pcd_holes")
def _pcd(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    n = count(p.get("n"), "params.n")
    r = num(p, "pcd_diameter") / 2
    start = num(p, "start_angle_deg", 0.0, positive=False)
    c = vec(p.get("center", [0, 0]), "params.center", 2)
    sign = -1 if p.get("clockwise") else 1
    step = 360.0 / n
    holes = []
    for k in range(n):
        a = math.radians(start + sign * k * step)
        holes.append([round(c[0] + r * math.cos(a), 6), round(c[1] + r * math.sin(a), 6)])
    ctx.formatted = f"{n} holes on Ø {2 * r:g}, pitch {step:g}°"
    return {"holes": holes, "angle_step_deg": step}


# ------------------------------------------------------------------ solids and sections

_UNIT = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}


def _solid(p: dict[str, Any]) -> dict[str, Any]:
    """Volume, area, centroid and second moment ∫ for density 1 (user's units)."""
    shape = p.get("shape")
    if shape == "cylinder":
        r, h = num(p, "radius"), num(p, "height")
        V = math.pi * r * r * h
        return {
            "volume": V,
            "surface_area": 2 * math.pi * r * r + 2 * math.pi * r * h,
            "centroid": [0.0, 0.0, h / 2],
            "I": [V * (3 * r * r + h * h) / 12, V * (3 * r * r + h * h) / 12, V * r * r / 2],
            "bbox": ([-r, -r, 0.0], [r, r, h]),
        }
    if shape == "tube":
        R = num(p, "outer_radius") if "outer_radius" in p else num(p, "D") / 2
        r = num(p, "inner_radius") if "inner_radius" in p else num(p, "d") / 2
        h = num(p, "height")
        if r >= R:
            raise ToolError("INVALID_ARGUMENT", "The inner radius must be less than the outer one.")
        V = math.pi * (R * R - r * r) * h
        return {
            "volume": V,
            "surface_area": 2 * math.pi * (R * R - r * r) + 2 * math.pi * (R + r) * h,
            "centroid": [0.0, 0.0, h / 2],
            "I": [V * (3 * (R * R + r * r) + h * h) / 12] * 2 + [V * (R * R + r * r) / 2],
            "bbox": ([-R, -R, 0.0], [R, R, h]),
        }
    if shape == "box":
        size = (
            vec(p.get("size"), "params.size")
            if p.get("size")
            else np.array([num(p, "a"), num(p, "b"), num(p, "c")])
        )
        if np.any(size <= 0):
            raise ToolError("INVALID_ARGUMENT", "params.size: values > 0.")
        a, b, c = (float(v) for v in size)
        V = a * b * c
        return {
            "volume": V,
            "surface_area": 2 * (a * b + b * c + a * c),
            "centroid": [a / 2, b / 2, c / 2],
            "I": [V * (b * b + c * c) / 12, V * (a * a + c * c) / 12, V * (a * a + b * b) / 12],
            "bbox": ([0.0, 0.0, 0.0], [a, b, c]),
        }
    if shape == "sphere":
        r = num(p, "radius")
        V = 4 / 3 * math.pi * r**3
        return {
            "volume": V,
            "surface_area": 4 * math.pi * r * r,
            "centroid": [0.0, 0.0, 0.0],
            "I": [0.4 * V * r * r] * 3,
            "bbox": ([-r, -r, -r], [r, r, r]),
        }
    if shape == "cone":
        r, h = num(p, "radius"), num(p, "height")
        V = math.pi * r * r * h / 3
        Ixx = V * (3 * r * r / 20 + 3 * h * h / 80)
        return {
            "volume": V,
            "surface_area": math.pi * r * r + math.pi * r * math.hypot(r, h),
            "centroid": [0.0, 0.0, h / 4],
            "I": [Ixx, Ixx, 0.3 * V * r * r],
            "bbox": ([-r, -r, 0.0], [r, r, h]),
        }
    if shape == "flange":
        D, d, t = num(p, "D"), num(p, "d"), num(p, "thickness")
        n = count(p.get("n_holes", 0), "params.n_holes", 0)
        hd = num(p, "hole_d") if n else 0.0
        pcd = num(p, "pcd") if n else 0.0
        R, r, rh, rho = D / 2, d / 2, hd / 2, pcd / 2
        if r >= R or (n and (rho + rh > R or rho - rh < r)):
            raise ToolError("INVALID_ARGUMENT", "Holes or centre hole outside the flange.")
        Vr = math.pi * (R * R - r * r) * t
        Vh = math.pi * rh * rh * t
        V = Vr - n * Vh
        Ixx = Vr * (3 * (R * R + r * r) + t * t) / 12
        Iyy = Ixx
        Izz = Vr * (R * R + r * r) / 2
        for k in range(n):
            a = 2 * math.pi * k / n
            x, y = rho * math.cos(a), rho * math.sin(a)
            Ixx -= Vh * (3 * rh * rh + t * t) / 12 + Vh * y * y
            Iyy -= Vh * (3 * rh * rh + t * t) / 12 + Vh * x * x
            Izz -= Vh * rh * rh / 2 + Vh * (x * x + y * y)
        area = (
            2 * (math.pi * (R * R - r * r) - n * math.pi * rh * rh)
            + 2 * math.pi * (R + r) * t
            + n * 2 * math.pi * rh * t
        )
        return {
            "volume": V,
            "surface_area": area,
            "centroid": [0.0, 0.0, t / 2],
            "I": [Ixx, Iyy, Izz],
            "bbox": ([-R, -R, 0.0], [R, R, t]),
        }
    raise ToolError(
        "INVALID_ARGUMENT",
        "params.shape: cylinder, tube, box, sphere, cone, flange.",
        '{"shape": "cylinder", "radius": 10, "height": 20}',
    )


def _mass_scale(p: dict[str, Any], ctx: Ctx) -> tuple[float | None, float]:
    if p.get("density") is None:
        return None, 1.0
    rho = num(p, "density")
    lu = str(p.get("length_unit", "m"))
    if lu not in _UNIT:
        raise ToolError("INVALID_ARGUMENT", "params.length_unit: m, cm or mm.", '"mm"')
    if "length_unit" not in p:
        ctx.warn(
            "Mass with lengths in metres and density in kg/m³: for millimetres use params.length_unit = 'mm'."
        )
    return rho, _UNIT[lu]


@op("solid_properties")
def _solid_props(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    s = _solid(p)
    rho, f = _mass_scale(p, ctx)
    out: dict[str, Any] = {
        "volume": s["volume"],
        "surface_area": s["surface_area"],
        "centroid": s["centroid"],
    }
    inertia = np.diag(s["I"])
    if rho is not None:
        mass = rho * s["volume"] * f**3
        out["mass"] = mass
        out["inertia"] = (inertia * rho * f**5).tolist()
        out["inertia_units"] = "kg·m² about the centroid"
    else:
        out["mass"] = None
        out["inertia"] = inertia.tolist()
        out["inertia_units"] = "∫r² dV about the centroid (density 1, user's unit^5)"
    ctx.formatted = f"Volume {s['volume']:.6g}"
    return out


def _rects(p: dict[str, Any]) -> tuple[list[tuple[float, float, float, float, int]], float, float]:
    shape = p.get("shape")
    if shape == "rectangle":
        b, h = num(p, "b"), num(p, "h")
        return [(0, 0, b, h, 1)], b, h
    if shape == "rect_tube":
        B = num(p, "B") if "B" in p else num(p, "b")
        H = num(p, "H") if "H" in p else num(p, "h")
        t = num(p, "t")
        if 2 * t >= min(B, H):
            raise ToolError("INVALID_ARGUMENT", "Thickness too large for the tube.")
        return [(0, 0, B, H, 1), (t, t, B - 2 * t, H - 2 * t, -1)], B, H
    if shape in {"i_beam", "t_section", "c_section"}:
        b, h, tf, tw = num(p, "b"), num(p, "h"), num(p, "tf"), num(p, "tw")
        if shape == "i_beam":
            if 2 * tf >= h or tw >= b:
                raise ToolError("INVALID_ARGUMENT", "Inconsistent I-beam dimensions.")
            return (
                [(0, 0, b, tf, 1), ((b - tw) / 2, tf, tw, h - 2 * tf, 1), (0, h - tf, b, tf, 1)],
                b,
                h,
            )
        if shape == "t_section":
            if tf >= h or tw >= b:
                raise ToolError("INVALID_ARGUMENT", "Inconsistent T-section dimensions.")
            return [(0, h - tf, b, tf, 1), ((b - tw) / 2, 0, tw, h - tf, 1)], b, h
        if 2 * tf >= h or tw >= b:
            raise ToolError("INVALID_ARGUMENT", "Inconsistent C-section dimensions.")
        return [(0, 0, tw, h, 1), (tw, 0, b - tw, tf, 1), (tw, h - tf, b - tw, tf, 1)], b, h
    raise ToolError(
        "INVALID_ARGUMENT",
        "params.shape: rectangle, circle, hollow_circle, rect_tube, i_beam, t_section, c_section.",
        '{"shape": "rectangle", "b": 100, "h": 200}',
    )


@op("section_properties")
def _section(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    shape = p.get("shape")
    if shape in {"circle", "hollow_circle"}:
        D = num(p, "D")
        d = num(p, "d") if shape == "hollow_circle" else 0.0
        if d >= D:
            raise ToolError("INVALID_ARGUMENT", "d must be less than D.")
        A = math.pi * (D * D - d * d) / 4
        Ix = math.pi * (D**4 - d**4) / 64
        Iy = Ix
        xc = yc = D / 2
        W, H = D, D
    else:
        rects, W, H = _rects(p)
        A = sum(s * w * h for _, _, w, h, s in rects)
        xc = sum(s * w * h * (x + w / 2) for x, _, w, h, s in rects) / A
        yc = sum(s * w * h * (y + h / 2) for _, y, w, h, s in rects) / A
        Ix = sum(s * (w * h**3 / 12 + w * h * (y + h / 2 - yc) ** 2) for _, y, w, h, s in rects)
        Iy = sum(s * (h * w**3 / 12 + w * h * (x + w / 2 - xc) ** 2) for x, _, w, h, s in rects)
    out = {
        "area": A,
        "centroid": [xc, yc],
        "Ix": Ix,
        "Iy": Iy,
        "Wx": Ix / max(yc, H - yc),
        "Wy": Iy / max(xc, W - xc),
        "Ip": Ix + Iy,
        "rx": math.sqrt(Ix / A),
        "ry": math.sqrt(Iy / A),
    }
    ctx.step(f"Area A = {A:.12g}.")
    ctx.step(f"Centroid: ({xc:.12g}, {yc:.12g}) from the bottom-left corner.")
    ctx.step(f"Ix = {Ix:.12g}, Iy = {Iy:.12g} (centroidal axes, Huygens-Steiner theorem).")
    ctx.formatted = f"A = {A:.6g}, Ix = {Ix:.6g}"
    return out


# ------------------------------------------------------------------ 2D drawing


@op("polygon_2d")
def _polygon(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    pts = points(P(args).get("points"), "params.points", 2, 3)
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a2 = float(cr.sum())
    if abs(a2) < EPS:
        raise ToolError("MATH_ERROR", "Degenerate polygon (zero area).")
    A = a2 / 2
    cx = float(((x + x1) * cr).sum()) / (3 * a2)
    cy = float(((y + y1) * cr).sum()) / (3 * a2)
    Ix0 = float(((y * y + y * y1 + y1 * y1) * cr).sum()) / 12
    Iy0 = float(((x * x + x * x1 + x1 * x1) * cr).sum()) / 12
    Ixy0 = float(((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum()) / 24
    s = 1 if A > 0 else -1
    Ix, Iy, Ixy = (
        s * Ix0 - abs(A) * cy * cy,
        s * Iy0 - abs(A) * cx * cx,
        s * Ixy0 - abs(A) * cx * cy,
    )
    edges = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    nz = turns[np.abs(turns) > EPS]
    convex = bool(np.all(nz > 0) or np.all(nz < 0))
    ctx.step(f"Gauss area formula (shoelace): A = {abs(A):.12g}.")
    ctx.step(f"Centroid: ({cx:.12g}, {cy:.12g}).")
    ctx.formatted = f"Area {abs(A):.6g}, perimeter {float(np.linalg.norm(edges, axis=1).sum()):.6g}"
    return {
        "area": abs(A),
        "signed_area": A,
        "perimeter": float(np.linalg.norm(edges, axis=1).sum()),
        "centroid": [cx, cy],
        "is_convex": convex,
        "orientation": "ccw" if A > 0 else "cw",
        "Ix": Ix,
        "Iy": Iy,
        "Ixy": Ixy,
    }


@op("circle_3pts")
def _circle3(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    pts = points(P(args).get("points"), "params.points", None, 3)
    if len(pts) != 3:
        raise ToolError("INVALID_ARGUMENT", "Exactly 3 points are needed.")
    dim = pts.shape[1]
    a, b, c = (np.append(q, 0.0) if dim == 2 else q for q in pts)
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = float(n @ n)
    if nn < EPS:
        raise ToolError("MATH_ERROR", "The three points are collinear: no circle.")
    center = a + (np.cross(n, ab) * float(ac @ ac) + np.cross(ac, n) * float(ab @ ab)) / (2 * nn)
    r = float(np.linalg.norm(center - a))
    out: dict[str, Any] = {"center": center[:dim].tolist(), "radius": r, "diameter": 2 * r}
    if dim == 3:
        out["normal"] = (n / math.sqrt(nn)).tolist()
    ctx.formatted = f"Centre {np.round(center[:dim], 6).tolist()}, radius {r:.6g}"
    return out


def _line2(raw: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    pts = points(raw, name, 2, 2)
    if len(pts) != 2 or np.linalg.norm(pts[1] - pts[0]) < EPS:
        raise ToolError("INVALID_ARGUMENT", f"'{name}': two distinct points [[x, y], [x, y]].")
    return pts[0], pts[1]


@op("arc_fillet")
def _fillet(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    a0, a1 = _line2(p.get("line1"), "params.line1")
    b0, b1 = _line2(p.get("line2"), "params.line2")
    r = num(p, "radius")
    da, db = a1 - a0, b1 - b0
    den = da[0] * db[1] - da[1] * db[0]
    if abs(den) < EPS:
        raise ToolError("MATH_ERROR", "Parallel lines: no fillet.")
    t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / den
    X = a0 + t * da
    u1 = (a1 if np.linalg.norm(a1 - X) >= np.linalg.norm(a0 - X) else a0) - X
    u2 = (b1 if np.linalg.norm(b1 - X) >= np.linalg.norm(b0 - X) else b0) - X
    u1, u2 = u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)
    phi = math.acos(max(-1.0, min(1.0, float(u1 @ u2))))
    if phi < 1e-9 or abs(phi - math.pi) < 1e-9:
        raise ToolError("MATH_ERROR", "Collinear lines: no fillet.")
    half = phi / 2
    bis = (u1 + u2) / np.linalg.norm(u1 + u2)
    center = X + bis * r / math.sin(half)
    t1 = X + u1 * r / math.tan(half)
    t2 = X + u2 * r / math.tan(half)
    a_start = math.degrees(math.atan2(t1[1] - center[1], t1[0] - center[0]))
    a_end = math.degrees(math.atan2(t2[1] - center[1], t2[0] - center[0]))
    ctx.step(f"The lines meet at {np.round(X, 9).tolist()} with angle {math.degrees(phi):.9g}°.")
    ctx.step(f"Distance of the centre from the vertex: r / sin(θ/2) = {r / math.sin(half):.9g}.")
    ctx.step(
        f"Distance of the tangent points from the vertex: r / tan(θ/2) = {r / math.tan(half):.9g}."
    )
    ctx.formatted = f"Centre {np.round(center, 6).tolist()}"
    return {
        "center": center.tolist(),
        "tangent1": t1.tolist(),
        "tangent2": t2.tolist(),
        "start_angle_deg": a_start,
        "end_angle_deg": a_end,
        "arc_length": r * (math.pi - phi),
        "corner": X.tolist(),
    }


def _tri(a: float, b: float, c: float) -> dict[str, float]:
    if a + b <= c + 1e-12 or a + c <= b + 1e-12 or b + c <= a + 1e-12:
        raise ToolError("MATH_ERROR", "The sides do not satisfy the triangle inequality.")
    A = math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c)))))
    B = math.degrees(math.acos(max(-1.0, min(1.0, (a * a + c * c - b * b) / (2 * a * c)))))
    C = 180.0 - A - B
    s = (a + b + c) / 2
    area = math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
    return {
        "a": a,
        "b": b,
        "c": c,
        "A": A,
        "B": B,
        "C": C,
        "area": area,
        "perimeter": a + b + c,
        "circumradius": a * b * c / (4 * area),
        "inradius": area / s,
    }


@op("triangle_solve")
def _triangle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    sides = {k: num(p, k) for k in ("a", "b", "c") if p.get(k) is not None}
    angs = {k: num(p, k) for k in ("A", "B", "C") if p.get(k) is not None}
    if len(sides) + len(angs) != 3 or not sides:
        raise ToolError(
            "INVALID_ARGUMENT",
            "3 values among a, b, c, A, B, C are needed, with at least one side.",
            '{"a": 3, "b": 4, "c": 5}',
        )
    if sum(angs.values()) >= 180:
        raise ToolError("MATH_ERROR", "The sum of the angles exceeds 180°.")
    opp = {"a": "A", "b": "B", "c": "C", "A": "a", "B": "b", "C": "c"}
    sols: list[dict[str, float]] = []
    if len(sides) == 3:
        sols = [_tri(sides["a"], sides["b"], sides["c"])]
    elif len(angs) == 2:
        third = ({"A", "B", "C"} - set(angs)).pop()
        angs[third] = 180 - sum(angs.values())
        ((side, val),) = sides.items()
        k = val / math.sin(math.radians(angs[opp[side]]))
        sols = [_tri(*(k * math.sin(math.radians(angs[X])) for X in "ABC"))]
    else:
        ((ang, val),) = angs.items()
        if opp[ang] not in sides:
            x, y = sorted(sides)
            c = math.sqrt(
                sides[x] ** 2
                + sides[y] ** 2
                - 2 * sides[x] * sides[y] * math.cos(math.radians(val))
            )
            vals = {**sides, opp[ang]: c}
            sols = [_tri(vals["a"], vals["b"], vals["c"])]
        else:
            known = opp[ang]
            other = next(s for s in sides if s != known)
            sin_o = sides[other] * math.sin(math.radians(val)) / sides[known]
            if sin_o > 1 + 1e-12:
                raise ToolError("MATH_ERROR", "No triangle with these data (impossible SSA case).")
            base = math.degrees(math.asin(min(1.0, sin_o)))
            for cand in {round(base, 12), round(180 - base, 12)}:
                if val + cand < 180 - 1e-9:
                    third_ang = 180 - val - cand
                    k = sides[known] / math.sin(math.radians(val))
                    angles = {ang: val, opp[other]: cand}
                    last = ({"A", "B", "C"} - set(angles)).pop()
                    angles[last] = third_ang
                    sols.append(_tri(*(k * math.sin(math.radians(angles[X])) for X in "ABC")))
            if len(sols) == 2:
                ctx.warn(
                    "Ambiguous case (two sides and a non-included angle): two possible triangles."
                )
    ctx.formatted = f"{len(sols)} solution(s)"
    return {"solutions": sols}


# ------------------------------------------------------------------ curve


@op("curve_points")
def _curve(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    kind = p.get("curve")
    n = count(p.get("n", 50), "params.n", 2)
    c = (
        vec(p.get("center", [0, 0, 0]), "params.center")
        if kind in {"arc", "circle", "ellipse", "helix"}
        else np.zeros(3)
    )
    if kind == "helix":
        r, pitch, turns = num(p, "radius"), num(p, "pitch"), num(p, "turns")
        t = np.linspace(0, 2 * math.pi * turns, n)
        sign = -1 if p.get("left_hand") else 1
        pts = np.column_stack([r * np.cos(t), sign * r * np.sin(t), pitch * t / (2 * math.pi)]) + c
        length = turns * math.hypot(2 * math.pi * r, pitch)
    elif kind == "involute":
        rb = num(p, "base_radius")
        tmax = math.radians(num(p, "max_angle_deg"))
        t = np.linspace(0, tmax, n)
        pts = np.column_stack(
            [rb * (np.cos(t) + t * np.sin(t)), rb * (np.sin(t) - t * np.cos(t)), np.zeros(n)]
        )
        length = rb * tmax * tmax / 2
    elif kind in {"arc", "circle"}:
        r = num(p, "radius")
        a0 = math.radians(num(p, "start_angle_deg", 0.0, positive=False)) if kind == "arc" else 0.0
        a1 = (
            math.radians(num(p, "end_angle_deg", 360.0, positive=False))
            if kind == "arc"
            else 2 * math.pi
        )
        t = np.linspace(a0, a1, n)
        pts = np.column_stack([r * np.cos(t), r * np.sin(t), np.zeros(n)]) + c
        length = r * abs(a1 - a0)
    elif kind == "ellipse":
        from scipy.special import ellipe

        a, b = num(p, "a"), num(p, "b")
        t = np.linspace(0, 2 * math.pi, n)
        pts = np.column_stack([a * np.cos(t), b * np.sin(t), np.zeros(n)]) + c
        big, small = max(a, b), min(a, b)
        length = 4 * big * float(ellipe(1 - (small / big) ** 2))
    elif kind == "spline":
        from scipy.integrate import quad
        from scipy.interpolate import CubicSpline

        cp = points(p.get("points"), "params.points", None, 2)
        seg = np.linalg.norm(np.diff(cp, axis=0), axis=1)
        if np.any(seg < EPS):
            raise ToolError("INVALID_ARGUMENT", "Repeated spline points.")
        u = np.concatenate([[0], np.cumsum(seg)])
        cs = CubicSpline(u, cp, bc_type="natural" if len(cp) > 2 else "not-a-knot")
        t = np.linspace(0, u[-1], n)
        pts = cs(t)
        d = cs.derivative()
        length = float(
            sum(
                quad(
                    lambda s: float(np.linalg.norm(d(s))),
                    u[i],
                    u[i + 1],
                    epsabs=1e-12,
                    epsrel=1e-12,
                )[0]
                for i in range(len(u) - 1)
            )
        )
        ctx.warn("Natural cubic spline through the given points, parameter = chord length.")
    elif kind == "parametric":
        import sympy as sp
        from scipy.integrate import quad

        from ..expr import parse
        from ..numfunc import compile_expr
        from .common import grammar, value_of

        g = grammar(["t"])
        tt = sp.Symbol("t")
        comps = [parse(str(p.get(k, "0")), g) for k in ("x", "y", "z")]
        rng = p.get("t_range") or [0, 1]
        t0, t1 = (float(value_of(v, grammar([])).evalf()) for v in rng)
        fs = [compile_expr(e, [tt]) for e in comps]
        ds = [compile_expr(sp.diff(e, tt), [tt]) for e in comps]
        t = np.linspace(t0, t1, n)
        pts = np.array([[f(float(v)) for f in fs] for v in t])
        if not np.all(np.isfinite(pts)):  # D93: before, ok with nan points and a negative length
            raise ToolError(
                "INVALID_ARGUMENT",
                "The curve is not defined at some points of t_range (e.g. 1/t at t = 0).",
            )
        import warnings

        from scipy.integrate import IntegrationWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            try:
                length = abs(
                    float(
                        quad(
                            lambda s: math.sqrt(sum(df(s) ** 2 for df in ds)),
                            t0,
                            t1,
                            epsabs=1e-12,
                            epsrel=1e-12,
                            limit=200,
                        )[0]
                    )
                )
            except IntegrationWarning:
                raise ToolError(
                    "MATH_ERROR", "The length integral does not converge on t_range."
                ) from None
    else:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.curve: helix, involute, spline, parametric, arc, circle, ellipse.",
            '{"curve": "helix", "radius": 10, "pitch": 2, "turns": 3, "n": 100}',
        )
    # D93: the other curves have no domain, so a non-finite point or length is the float range (D74, D95)
    if not (np.all(np.isfinite(pts)) and math.isfinite(length)):
        raise ToolError("MATH_ERROR", NOT_FINITE)
    ctx.formatted = f"{n} points, length {length:.6g}"
    return {"points": pts.tolist(), "length": length}


# ------------------------------------------------------------------ intersections


def _obj(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("type") not in {"line", "plane", "sphere", "cylinder"}:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{name}: object with type line, plane, sphere or cylinder.",
            '{"type": "line", "point": [0,0,0], "dir": [1,0,0]}',
        )
    t = raw["type"]
    o: dict[str, Any] = {"type": t}
    if t == "line":
        o["p"], o["d"] = (
            vec(raw.get("point"), f"{name}.point"),
            unit(vec(raw.get("dir"), f"{name}.dir"), f"{name}.dir"),
        )
    elif t == "plane":
        o["p"], o["n"] = (
            vec(raw.get("point"), f"{name}.point"),
            unit(vec(raw.get("normal"), f"{name}.normal"), f"{name}.normal"),
        )
    elif t == "sphere":
        o["c"], o["r"] = vec(raw.get("center"), f"{name}.center"), num(raw, "radius")
    else:
        o["p"], o["d"], o["r"] = (
            vec(raw.get("point"), f"{name}.point"),
            unit(vec(raw.get("axis"), f"{name}.axis"), f"{name}.axis"),
            num(raw, "radius"),
        )
    return o


def _line_line(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    w0 = a["p"] - b["p"]
    cr = np.cross(a["d"], b["d"])
    if float(np.linalg.norm(cr)) < 1e-12:
        dist = float(np.linalg.norm(np.cross(w0, a["d"])))
        if dist < 1e-9:
            return {"kind": "coincident", "distance": 0.0}
        return {"kind": "none", "parallel": True, "distance": dist}
    bb = float(a["d"] @ b["d"])
    d, e = float(a["d"] @ w0), float(b["d"] @ w0)
    den = 1 - bb * bb
    s = (bb * e - d) / den
    t = (e - bb * d) / den
    pa, pb = a["p"] + s * a["d"], b["p"] + t * b["d"]
    dist = float(np.linalg.norm(pa - pb))
    if dist < 1e-9:
        return {"kind": "point", "point": ((pa + pb) / 2).tolist(), "distance": 0.0}
    return {
        "kind": "none",
        "skew": True,
        "closest_points": [pa.tolist(), pb.tolist()],
        "distance": dist,
    }


def _line_sphere(a: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    oc = a["p"] - s["c"]
    bq = float(a["d"] @ oc)
    disc = bq * bq - (float(oc @ oc) - s["r"] ** 2)
    if disc < -1e-12:
        return {"kind": "none", "distance": float(np.linalg.norm(np.cross(oc, a["d"]))) - s["r"]}
    if abs(disc) <= 1e-12:
        return {"kind": "point", "point": (a["p"] - bq * a["d"]).tolist()}
    sq = math.sqrt(disc)
    return {
        "kind": "points",
        "points": [
            (a["p"] + (-bq - sq) * a["d"]).tolist(),
            (a["p"] + (-bq + sq) * a["d"]).tolist(),
        ],
    }


def _plane_plane(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    d = np.cross(a["n"], b["n"])
    nd = float(np.linalg.norm(d))
    if nd < 1e-12:
        if abs(float((b["p"] - a["p"]) @ a["n"])) < 1e-9:
            return {"kind": "coincident"}
        return {
            "kind": "none",
            "parallel": True,
            "distance": abs(float((b["p"] - a["p"]) @ a["n"])),
        }
    d = d / nd
    h1, h2 = float(a["n"] @ a["p"]), float(b["n"] @ b["p"])
    M = np.array([a["n"], b["n"], d])
    pt = np.linalg.solve(M, np.array([h1, h2, 0.0]))
    return {"kind": "line", "point": pt.tolist(), "dir": d.tolist()}


def _plane_sphere(pl: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    dist = float((s["c"] - pl["p"]) @ pl["n"])
    foot = s["c"] - dist * pl["n"]
    if abs(dist) > s["r"] + 1e-12:
        return {"kind": "none", "distance": abs(dist) - s["r"]}
    if abs(abs(dist) - s["r"]) <= 1e-12:
        return {"kind": "point", "point": foot.tolist()}
    return {
        "kind": "circle",
        "center": foot.tolist(),
        "radius": math.sqrt(s["r"] ** 2 - dist * dist),
        "normal": pl["n"].tolist(),
    }


def _sphere_sphere(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    v = b["c"] - a["c"]
    d = float(np.linalg.norm(v))
    if d < 1e-12:
        return {"kind": "coincident"} if abs(a["r"] - b["r"]) < 1e-12 else {"kind": "none"}
    if d > a["r"] + b["r"] + 1e-12 or d < abs(a["r"] - b["r"]) - 1e-12:
        return {"kind": "none", "distance": d}
    n = v / d
    x = (d * d + a["r"] ** 2 - b["r"] ** 2) / (2 * d)
    h2 = a["r"] ** 2 - x * x
    c = a["c"] + x * n
    if abs(h2) <= 1e-12:
        return {"kind": "point", "point": c.tolist()}
    return {"kind": "circle", "center": c.tolist(), "radius": math.sqrt(h2), "normal": n.tolist()}


def _line_plane(ln: dict[str, Any], pl: dict[str, Any]) -> dict[str, Any]:
    den = float(ln["d"] @ pl["n"])
    off = float((pl["p"] - ln["p"]) @ pl["n"])
    if abs(den) < 1e-12:
        return (
            {"kind": "line", "point": ln["p"].tolist(), "dir": ln["d"].tolist()}
            if abs(off) < 1e-9
            else {"kind": "none", "parallel": True, "distance": abs(off)}
        )
    return {"kind": "point", "point": (ln["p"] + off / den * ln["d"]).tolist()}


def _line_cyl(ln: dict[str, Any], cy: dict[str, Any]) -> dict[str, Any]:
    w = ln["p"] - cy["p"]
    dp = ln["d"] - float(ln["d"] @ cy["d"]) * cy["d"]
    wp = w - float(w @ cy["d"]) * cy["d"]
    A, B, C = float(dp @ dp), 2 * float(dp @ wp), float(wp @ wp) - cy["r"] ** 2
    if A < 1e-12:
        return (
            {"kind": "line", "point": ln["p"].tolist(), "dir": ln["d"].tolist()}
            if abs(C) < 1e-9
            else {"kind": "none", "parallel": True}
        )
    disc = B * B - 4 * A * C
    if disc < -1e-12:
        return {"kind": "none"}
    if abs(disc) <= 1e-12:
        return {"kind": "point", "point": (ln["p"] - B / (2 * A) * ln["d"]).tolist()}
    sq = math.sqrt(disc)
    return {
        "kind": "points",
        "points": [
            (ln["p"] + (-B - sq) / (2 * A) * ln["d"]).tolist(),
            (ln["p"] + (-B + sq) / (2 * A) * ln["d"]).tolist(),
        ],
    }


_PAIRS: dict[tuple[str, str], Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
    ("line", "line"): _line_line,
    ("line", "sphere"): _line_sphere,
    ("line", "plane"): _line_plane,
    ("line", "cylinder"): _line_cyl,
    ("plane", "plane"): _plane_plane,
    ("plane", "sphere"): _plane_sphere,
    ("sphere", "sphere"): _sphere_sphere,
}


@op("intersect")
def _intersect(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    a, b = _obj(p.get("a"), "a"), _obj(p.get("b"), "b")
    key = (a["type"], b["type"])
    if key not in _PAIRS:
        a, b = b, a
        key = (a["type"], b["type"])
    if key not in _PAIRS:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Pair {key[1]}-{key[0]} not supported: {sorted(f'{x}-{y}' for x, y in _PAIRS)}.",
        )
    out = _PAIRS[key](a, b)
    ctx.formatted = f"Intersection: {out['kind']}"
    return out


# ------------------------------------------------------------------ points for CAD


@op("feature_points")
def _features(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    origin = vec(p.get("origin", [0, 0, 0]), "params.origin")
    R = (
        euler_matrix(vec(p["rotation_deg"], "params.rotation_deg"), "XYZ")
        if p.get("rotation_deg")
        else np.eye(3)
    )

    def tr(q: list[float]) -> list[float]:
        return (R @ np.array(q, dtype=float) + origin).tolist()

    if p.get("shape") == "box":
        a, b, c = (float(v) for v in vec(p.get("size"), "params.size"))
        vs = [[x, y, z] for z in (0, c) for y in (0, b) for x in (0, a)]
        edges = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        mids = [[(vs[i][k] + vs[j][k]) / 2 for k in range(3)] for i, j in edges]
        faces = [
            ("bottom", [a / 2, b / 2, 0], [0, 0, -1]),
            ("top", [a / 2, b / 2, c], [0, 0, 1]),
            ("front", [a / 2, 0, c / 2], [0, -1, 0]),
            ("back", [a / 2, b, c / 2], [0, 1, 0]),
            ("left", [0, b / 2, c / 2], [-1, 0, 0]),
            ("right", [a, b / 2, c / 2], [1, 0, 0]),
        ]
        out = {
            "vertices": [tr(v) for v in vs],
            "edge_midpoints": [tr(m) for m in mids],
            "face_centers": [
                {"name": n, "point": tr(q), "normal": (R @ np.array(nv, dtype=float)).tolist()}
                for n, q, nv in faces
            ],
        }
    elif p.get("shape") == "cylinder":
        r, h = num(p, "radius"), num(p, "height")
        out = {
            "vertices": [],
            "edge_midpoints": [tr([r, 0, 0]), tr([r, 0, h])],
            "face_centers": [
                {
                    "name": "bottom",
                    "point": tr([0, 0, 0]),
                    "normal": (R @ np.array([0.0, 0, -1])).tolist(),
                },
                {
                    "name": "top",
                    "point": tr([0, 0, h]),
                    "normal": (R @ np.array([0.0, 0, 1])).tolist(),
                },
                {
                    "name": "side",
                    "point": tr([r, 0, h / 2]),
                    "normal": (R @ np.array([1.0, 0, 0])).tolist(),
                },
            ],
            "axis": [tr([0, 0, 0]), tr([0, 0, h])],
        }
        ctx.warn("edge_midpoints of the cylinder: one point on each of the two circular edges.")
    else:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.shape: box or cylinder.",
            '{"shape": "box", "size": [10, 20, 30]}',
        )
    ctx.formatted = f"{len(out['vertices'])} vertices, {len(out['face_centers'])} faces"
    return out


@op("pattern")
def _pattern(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    kind = p.get("kind")
    origin = vec(p.get("origin", [0, 0, 0]), "params.origin")
    pts: list[list[float]] = []
    if kind == "linear":
        n = count(p.get("count"), "params.count")
        d = unit(vec(p.get("axis", [1, 0, 0]), "params.axis"), "params.axis")
        s = num(p, "spacing", positive=False)
        pts = [(origin + k * s * d).tolist() for k in range(n)]
    elif kind == "grid":
        cnt = p.get("count")
        sp_ = p.get("spacing")
        if not (
            isinstance(cnt, list) and len(cnt) == 2 and isinstance(sp_, list) and len(sp_) == 2
        ):
            raise ToolError(
                "INVALID_ARGUMENT",
                "grid: count [nx, ny] and spacing [dx, dy].",
                '{"kind": "grid", "count": [3, 2], "spacing": [10, 20]}',
            )
        nx, ny = count(cnt[0], "params.count[0]"), count(cnt[1], "params.count[1]")
        if nx * ny > limits.LIST_MAX:
            raise ToolError("LIMIT_EXCEEDED", f"At most {limits.LIST_MAX} points.")
        pts = [
            (origin + np.array([i * float(sp_[0]), j * float(sp_[1]), 0.0])).tolist()
            for j in range(ny)
            for i in range(nx)
        ]
    elif kind == "circular":
        n = count(p.get("count"), "params.count")
        r = num(p, "radius")
        total = num(p, "angle_deg", 360.0)
        start = num(p, "start_angle_deg", 0.0, positive=False)
        step = total / n if abs(total - 360.0) < 1e-12 else total / max(n - 1, 1)
        axis = unit(vec(p.get("axis", [0, 0, 1]), "params.axis"), "params.axis")
        ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1, 0])
        u = unit(ref - float(ref @ axis) * axis, "ref")
        v = np.cross(axis, u)
        for k in range(n):
            a = math.radians(start + k * step)
            pts.append((origin + r * (math.cos(a) * u + math.sin(a) * v)).tolist())
    else:
        raise ToolError(
            "INVALID_ARGUMENT",
            "params.kind: linear, grid or circular.",
            '{"kind": "circular", "count": 6, "radius": 10}',
        )
    if len(pts) > limits.LIST_MAX:
        raise ToolError("LIMIT_EXCEEDED", f"At most {limits.LIST_MAX} points.")
    ctx.formatted = f"{len(pts)} points"
    return {"points": pts, "count": len(pts)}


@op("point_cloud")
def _cloud(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    pts = points(P(args).get("points"), "params.points", 3, 1)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    out: dict[str, Any] = {
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "size": (hi - lo).tolist(),
        "center": ((lo + hi) / 2).tolist(),
        "centroid": pts.mean(axis=0).tolist(),
        "hull_volume": None,
        "hull_area": None,
    }
    if len(pts) >= 4:
        from scipy.spatial import ConvexHull, QhullError

        try:
            h = ConvexHull(pts)
            out["hull_volume"], out["hull_area"] = float(h.volume), float(h.area)
        except QhullError:
            ctx.warn("Coplanar or collinear points: 3D convex hull not defined.")
    ctx.formatted = f"Bounding box {np.round(hi - lo, 6).tolist()}"
    return out


@op("mesh_properties")
def _mesh(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    V = points(p.get("vertices"), "params.vertices", 3, 4)
    F_raw = p.get("faces")
    if not isinstance(F_raw, list) or not F_raw:
        raise ToolError(
            "INVALID_ARGUMENT", "params.faces is needed (triangles, indices from 0).", "[[0, 1, 2]]"
        )
    try:
        F = np.array(F_raw, dtype=int)
    except (TypeError, ValueError):
        raise ToolError("INVALID_ARGUMENT", "params.faces: triangles of integer indices.") from None
    if F.ndim != 2 or F.shape[1] != 3 or F.min() < 0 or F.max() >= len(V):
        raise ToolError("INVALID_ARGUMENT", "params.faces: triangles [i, j, k] with valid indices.")
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = float(0.5 * np.linalg.norm(cr, axis=1).sum())
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = float(det.sum() / 6)
    cen = (
        (det[:, None] * (a + b + c)).sum(axis=0) / 24 / vol
        if abs(vol) > EPS
        else np.full(3, math.nan)
    )
    C0 = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 120
    Cov = np.zeros((3, 3))
    for i in range(len(F)):
        A = np.column_stack([a[i], b[i], c[i]])
        Cov += det[i] * (A @ C0 @ A.T)
    edges: dict[tuple[int, int], int] = {}
    for tri in F:
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[(int(i), int(j))] = edges.get((int(i), int(j)), 0) + 1
    watertight = all(cnt == 1 and edges.get((j, i), 0) == 1 for (i, j), cnt in edges.items())
    if vol < 0:
        ctx.warn("Normals pointing inwards: volume with negative sign, reported as absolute value.")
        Cov, vol = -Cov, -vol
    if not watertight:
        ctx.warn(
            "Mesh not closed or with inconsistent orientation: volume and centroid not reliable."
        )
    Cc = Cov - vol * np.outer(cen, cen)
    inertia = np.trace(Cc) * np.eye(3) - Cc
    flat = abs(vol) <= EPS
    if flat:  # D93: before, centroid and inertia were nan inside ok (D74)
        ctx.warn("Volume zero (flat or open mesh): centroid and inertia are not defined.")
    ctx.formatted = f"Volume {vol:.6g}, area {area:.6g}"
    return {
        "volume": vol,
        "area": area,
        "centroid": None if flat else cen.tolist(),
        "bbox_min": V.min(axis=0).tolist(),
        "bbox_max": V.max(axis=0).tolist(),
        "watertight": watertight,
        "inertia": None if flat else inertia.tolist(),
        "inertia_units": "∫r² dV about the centroid (density 1)",
    }


@op("composite_properties")
def _composite(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    p = P(args)
    parts = p.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ToolError(
            "INVALID_ARGUMENT",
            "Missing params.parts.",
            '[{"shape": "box", "size": [10, 10, 10], "position": [0, 0, 0], "op": "add"}]',
        )
    vol = 0.0
    moment = np.zeros(3)
    adds: list[tuple[np.ndarray, np.ndarray]] = []
    subs: list[tuple[np.ndarray, np.ndarray]] = []
    for i, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ToolError("INVALID_ARGUMENT", f"params.parts[{i}] must be an object.")
        s = _solid(part)
        pos = vec(part.get("position", [0, 0, 0]), f"parts[{i}].position")
        sign = {"add": 1.0, "subtract": -1.0}.get(part.get("op", "add"))
        if sign is None:
            raise ToolError("INVALID_ARGUMENT", f"parts[{i}].op: add or subtract.")
        vol += sign * s["volume"]
        moment += sign * s["volume"] * (np.array(s["centroid"]) + pos)
        box = (np.array(s["bbox"][0]) + pos, np.array(s["bbox"][1]) + pos)
        (adds if sign > 0 else subs).append(box)
    if vol <= 0:
        raise ToolError(
            "MATH_ERROR", "Total volume not positive: the subtracted parts exceed the added ones."
        )
    for i in range(len(adds)):
        for j in range(i + 1, len(adds)):
            if np.all(adds[i][0] < adds[j][1] - EPS) and np.all(adds[j][0] < adds[i][1] - EPS):
                ctx.warn(
                    f"The added parts {i} and {j} may overlap: the volume would count the common part twice."
                )
    for k, (lo, hi) in enumerate(subs):
        if not any(np.all(lo >= a_lo - 1e-9) and np.all(hi <= a_hi + 1e-9) for a_lo, a_hi in adds):
            ctx.warn(
                f"The subtracted part {k} is not fully inside an added bounding box: check in the CAD."
            )
    lo = np.min([b[0] for b in adds], axis=0)
    hi = np.max([b[1] for b in adds], axis=0)
    out: dict[str, Any] = {
        "volume": vol,
        "centroid": (moment / vol).tolist(),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "mass": None,
    }
    rho, f = _mass_scale(p, ctx)
    if rho is not None:
        out["mass"] = rho * vol * f**3
    ctx.formatted = f"Volume {vol:.6g}"
    return out
