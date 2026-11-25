"""Base of eng_calc: registry of the operations, reading of values with units, formatting.

Inputs in `params`: a number is in the SI unit of the quantity (m, N, Pa, Nm, m4, N/m, rad/s, W);
otherwise `[value, "unit"]` with the units of `unit_convert` plus those of `_EXTRA`.
A unit of the wrong kind (a length instead of a stress) and an unexpected key give
INVALID_ARGUMENT.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from ..constants import lookup
from ..envelope import Ctx
from ..errors import ToolError
from ..fmt import fmt_float
from .math_eval import unit_factor

Op = Callable[[dict[str, Any], Ctx], dict[str, Any]]
_EV = 1.602176634e-19  # J, exact in the 2019 SI
_C = 299792458.0  # m/s, exact
_U = lookup("atomic mass constant")[1]  # kg, CODATA
_YEAR = 365.25 * 86400.0  # Julian year (IAU), for inputs in y, a, ky, My
OPS: dict[str, tuple[Op, frozenset[str]]] = {}

_EXTRA: dict[str, dict[str, float]] = {
    "area_moment": {"m4": 1.0, "cm4": 1e-8, "mm4": 1e-12, "in4": 0.0254**4},
    "line_load": {"N/m": 1.0, "kN/m": 1e3, "N/mm": 1e3},
    "angular_velocity": {"rad/s": 1.0, "rpm": 2 * math.pi / 60},
    "velocity": {"m/min": 1 / 60, "mm/min": 1 / 60000, "mm/s": 1e-3, "km/s": 1e3},
    "stiffness": {"N/m": 1.0, "kN/m": 1e3, "N/mm": 1e3, "kN/mm": 1e6},
    "damping": {"N*s/m": 1.0, "Ns/m": 1.0, "kg/s": 1.0, "N*s/mm": 1e3},
    "frequency": {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12, "PHz": 1e15},
    "acceleration": {"m/s2": 1.0},
    "mass_moment": {"kg*m2": 1.0, "kgm2": 1.0, "kg*cm2": 1e-4, "kg*mm2": 1e-6},
    "linear_mass": {"kg/m": 1.0, "g/m": 1e-3},
    "density": {"kg/m3": 1.0, "g/cm3": 1e3, "kg/L": 1e3, "kg/dm3": 1e3},
    "specific_volume": {"m3/kg": 1.0, "L/kg": 1e-3, "dm3/kg": 1e-3, "cm3/g": 1e-3},
    "mass_flow_rate": {"kg/s": 1.0, "kg/h": 1 / 3600, "t/h": 1000 / 3600, "kg/min": 1 / 60},
    "dynamic_viscosity": {"Pa*s": 1.0, "Pas": 1.0, "mPa*s": 1e-3, "cP": 1e-3, "P": 0.1},
    "kinematic_viscosity": {"m2/s": 1.0, "mm2/s": 1e-6, "cSt": 1e-6, "St": 1e-4},
    "thermal_conductivity": {"W/(m*K)": 1.0, "W/mK": 1.0},
    "heat_transfer_coefficient": {"W/(m2*K)": 1.0, "W/m2K": 1.0},
    "thermal_resistance": {"K/W": 1.0, "C/W": 1.0},
    "thermal_conductance": {"W/K": 1.0, "kW/K": 1e3, "W/C": 1.0, "kW/C": 1e3},
    "magnetic_flux_density": {"T": 1.0, "mT": 1e-3, "uT": 1e-6, "nT": 1e-9, "G": 1e-4},
    "magnetic_flux": {"Wb": 1.0, "mWb": 1e-3, "uWb": 1e-6},
    "voltage": {"V": 1.0, "mV": 1e-3, "kV": 1e3, "MV": 1e6},
    "current": {"A": 1.0, "mA": 1e-3, "uA": 1e-6, "kA": 1e3},
    "electrical_resistance": {"ohm": 1.0, "mohm": 1e-3, "kohm": 1e3, "Mohm": 1e6},
    "resistivity": {"ohm*m": 1.0, "ohm*mm2/m": 1e-6},
    "inductance": {"H": 1.0, "mH": 1e-3, "uH": 1e-6, "nH": 1e-9},
    "capacitance": {"F": 1.0, "mF": 1e-3, "uF": 1e-6, "nF": 1e-9, "pF": 1e-12},
    "reactive_power": {"var": 1.0, "kvar": 1e3},
    "apparent_power": {"VA": 1.0, "kVA": 1e3},
    "specific_heat": {"J/(kg*K)": 1.0, "J/kgK": 1.0, "kJ/(kg*K)": 1e3, "kJ/kgK": 1e3},
    "specific_energy": {"J/kg": 1.0, "kJ/kg": 1e3, "MJ/kg": 1e6},
    "molar_mass": {"kg/mol": 1.0, "g/mol": 1e-3},
    "amount_of_substance": {"mol": 1.0, "kmol": 1e3, "mmol": 1e-3},
    "concentration": {"mol/L": 1.0, "M": 1.0, "mmol/L": 1e-3, "umol/L": 1e-6, "mol/m3": 1e-3},
    "molar_energy": {"J/mol": 1.0, "kJ/mol": 1e3, "kcal/mol": 4184.0},
    "charge": {"C": 1.0, "Ah": 3600.0, "mAh": 3.6, "mC": 1e-3, "uC": 1e-6, "nC": 1e-9, "pC": 1e-12},
    "time": {
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "ps": 1e-12,
        "y": _YEAR,
        "a": _YEAR,
        "ky": 1e3 * _YEAR,
        "My": 1e6 * _YEAR,
    },
    "energy": {"keV": 1e3 * _EV, "MeV": 1e6 * _EV, "GeV": 1e9 * _EV},
    "length": {"pm": 1e-12, "fm": 1e-15, "angstrom": 1e-10, "Å": 1e-10},
    "mass": {"u": _U, "Da": _U, "MeV/c2": 1e6 * _EV / _C**2, "GeV/c2": 1e9 * _EV / _C**2},
    "activity": {
        "Bq": 1.0,
        "kBq": 1e3,
        "MBq": 1e6,
        "GBq": 1e9,
        "TBq": 1e12,
        "Ci": 3.7e10,
        "mCi": 3.7e7,
        "uCi": 3.7e4,
    },
    "decay_constant": {
        "1/s": 1.0,
        "1/min": 1 / 60,
        "1/h": 1 / 3600,
        "1/d": 1 / 86400,
        "1/y": 1 / _YEAR,
    },
    "momentum": {
        "kg*m/s": 1.0,
        "N*s": 1.0,
        "eV/c": _EV / _C,
        "keV/c": 1e3 * _EV / _C,
        "MeV/c": 1e6 * _EV / _C,
        "GeV/c": 1e9 * _EV / _C,
    },
    "thermal_expansion": {
        "1/K": 1.0,
        "1/C": 1.0,
        "um/(m*K)": 1e-6,
        "um/(m*C)": 1e-6,
        "ppm/K": 1e-6,
    },
}
# kind -> (description, SI unit, example unit)
_KIND: dict[str, tuple[str, str, str]] = {
    "length": ("a length", "m", "mm"),
    "area": ("an area", "m2", "mm2"),
    "volume": ("a volume (or a section modulus)", "m3", "cm3"),
    "energy": ("an energy", "J", "kJ"),
    "specific_heat": ("a specific heat or a gas constant", "J/(kg*K)", "kJ/(kg*K)"),
    "specific_energy": ("an energy per kg", "J/kg", "kJ/kg"),
    "molar_mass": ("a molar mass", "kg/mol", "g/mol"),
    "amount_of_substance": ("an amount of substance", "mol", "mol"),
    "force": ("a force", "N", "kN"),
    "pressure": ("a stress or a pressure", "Pa", "MPa"),
    "torque": ("a moment or a torque", "Nm", "Nm"),
    "power": ("a power", "W", "kW"),
    "area_moment": ("a second moment of area", "m4", "cm4"),
    "line_load": ("a distributed load", "N/m", "kN/m"),
    "angular_velocity": ("an angular velocity", "rad/s", "rpm"),
    "velocity": ("a velocity", "m/s", "m/min"),
    "mass": ("a mass", "kg", "kg"),
    "time": ("a time", "s", "h"),
    "stiffness": ("a stiffness", "N/m", "N/mm"),
    "damping": ("a viscous damping", "N*s/m", "N*s/m"),
    "frequency": ("a frequency", "Hz", "Hz"),
    "acceleration": ("an acceleration", "m/s2", "m/s2"),
    "mass_moment": ("a mass moment of inertia", "kg*m2", "kg*m2"),
    "linear_mass": ("a mass per metre", "kg/m", "kg/m"),
    "flow_rate": ("a volume flow rate", "m3/s", "m3/h"),
    "density": ("a density", "kg/m3", "kg/m3"),
    "specific_volume": ("a specific volume", "m3/kg", "m3/kg"),
    "mass_flow_rate": ("a mass flow rate", "kg/s", "t/h"),
    "dynamic_viscosity": ("a dynamic viscosity", "Pa*s", "cP"),
    "kinematic_viscosity": ("a kinematic viscosity", "m2/s", "cSt"),
    "thermal_conductivity": ("a thermal conductivity", "W/(m*K)", "W/(m*K)"),
    "heat_transfer_coefficient": ("a heat transfer coefficient", "W/(m2*K)", "W/(m2*K)"),
    "thermal_resistance": ("a thermal resistance", "K/W", "K/W"),
    "magnetic_flux_density": ("a magnetic flux density (B field)", "T", "mT"),
    "magnetic_flux": ("a magnetic flux", "Wb", "mWb"),
    "thermal_conductance": (
        "a thermal conductance (U A) or a heat capacity rate (m cp)",
        "W/K",
        "kW/K",
    ),
    "voltage": ("a voltage", "V", "V"),
    "current": ("a current", "A", "mA"),
    "electrical_resistance": ("an electrical resistance", "ohm", "kohm"),
    "resistivity": ("a resistivity", "ohm*m", "ohm*mm2/m"),
    "inductance": ("an inductance", "H", "mH"),
    "capacitance": ("a capacitance", "F", "uF"),
    "thermal_expansion": ("a thermal expansion coefficient", "1/K", "um/(m*K)"),
    "concentration": ("a molar concentration (mol/L basis, D48)", "mol/L", "mol/L"),
    "molar_energy": ("an energy per mole", "J/mol", "kJ/mol"),
    "charge": ("an electric charge", "C", "Ah"),
    "activity": ("a radioactive activity", "Bq", "MBq"),
    "momentum": ("a momentum", "kg*m/s", "MeV/c"),
    "decay_constant": ("a decay constant", "1/s", "1/y"),
}
_EXAMPLE = (
    '{"operation": "beam_calc", "params": {"support": "simply_supported", "load": "point", '
    '"L": [2, "m"], "F": [5, "kN"], "E": [210, "GPa"], "I": [800, "cm4"]}}'
)


def op(name: str, keys: Iterable[str]) -> Callable[[Op], Op]:
    def deco(fn: Op) -> Op:
        OPS[name] = (fn, frozenset(keys))
        return fn

    return deco


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    name = args["operation"]
    if name not in OPS:
        raise ToolError(
            "NOT_IMPLEMENTED",
            f"eng_calc.{name} is not written yet (D39, it comes with the next areas). "
            f"Ready: {', '.join(sorted(OPS))}.",
            _EXAMPLE,
        )
    fn, keys = OPS[name]
    p = args.get("params")
    if not isinstance(p, dict):
        raise ToolError("INVALID_ARGUMENT", "'params' must be an object.", _EXAMPLE)
    extra = sorted(set(p) - keys)
    if extra:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Unexpected keys in params: {', '.join(extra)}. "
            f"Keys of eng_calc.{name}: {', '.join(sorted(keys))}.",
        )
    result = fn(p, ctx)
    # D74: an ok answer never carries "oo" or "nan"; an overflow of the inputs is an error
    if not _all_finite(result):
        raise ToolError(
            "MATH_ERROR",
            "The result is not finite: the inputs are outside the floating-point range (overflow or underflow).",
        )
    return {k: clean(v) for k, v in result.items()}


def _all_finite(v: Any) -> bool:
    if isinstance(v, dict):
        return all(_all_finite(x) for x in v.values())
    if isinstance(v, list | tuple):
        return all(_all_finite(x) for x in v)
    return not isinstance(v, float) or math.isfinite(v)


def clean(v: Any) -> Any:
    """15 significant digits (removes float noise, keeps small values) and no -0.0."""
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, float) and math.isfinite(v):
        r = float(f"{v:.15g}")
        return 0.0 if r == 0 else r
    return v


# ------------------------------------------------------------------ reading the inputs


def factor(unit: str, kind: str, key: str) -> float:
    if unit in _EXTRA.get(kind, {}):
        return _EXTRA[kind][unit]
    found = unit_factor(unit)
    other = next((k for k, t in _EXTRA.items() if unit in t), None)
    desc, _, ex = _KIND[kind]
    if found is None and other is None:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: unknown unit '{unit}'. Here {desc} is needed (for example [1, \"{ex}\"]).",
        )
    got = found[0] if found is not None else other
    if got != kind:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: '{unit}' is a unit of {str(got).replace('_', ' ')}; "
            f'here {desc} is needed (for example [1, "{ex}"]).',
        )
    assert found is not None
    return found[1]


def qty(
    p: dict[str, Any],
    key: str,
    kind: str | None,
    *,
    need: bool = True,
    positive: bool = True,
    default: float | None = None,
) -> float | None:
    """Reads an SI number or a pair [value, "unit"]; kind None = pure number."""
    raw = p.get(key)
    if raw is None:
        if default is not None or not need:
            return default
        unit_txt = f'number in {_KIND[kind][1]} or [value, "unit"]' if kind else "pure number"
        raise ToolError("INVALID_ARGUMENT", f"Missing params.{key} ({unit_txt}).", _EXAMPLE)
    unit: str | None = None
    if isinstance(raw, list | tuple):
        if kind is None:
            raise ToolError("INVALID_ARGUMENT", f"params.{key} is a pure number, without units.")
        if len(raw) != 2 or not isinstance(raw[1], str):
            raise ToolError(
                "INVALID_ARGUMENT",
                f'params.{key}: the form with units is [value, "unit"].',
                f'{{"{key}": [10, "mm"]}}',
            )
        raw, unit = raw
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be a number.")
    v = float(raw)
    if not math.isfinite(v):
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be finite.")
    if unit is not None and kind is not None:
        v *= factor(unit, kind, key)
    if positive and v <= 0:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be > 0.")
    return v


def need(p: dict[str, Any], key: str, kind: str | None, *, positive: bool = True) -> float:
    v = qty(p, key, kind, positive=positive)
    assert v is not None
    return v


def choice(
    p: dict[str, Any], key: str, options: tuple[str, ...], default: str | None = None
) -> str:
    v = p.get(key, default)
    if v not in options:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"params.{key}: one of {', '.join(options)}.",
            f'{{"{key}": "{options[0]}"}}',
        )
    return str(v)


def one_of(p: dict[str, Any], keys: tuple[str, ...], what: str) -> str:
    given = [k for k in keys if p.get(k) is not None]
    if len(given) != 1:
        raise ToolError(
            "INVALID_ARGUMENT", f"For {what} exactly one of these is needed: {', '.join(keys)}."
        )
    return given[0]


def units_for(result: dict[str, Any], table: dict[str, str]) -> dict[str, str]:
    return {k: table[k] for k in result if k in table}


def fmt_sig(v: float, digits: int = 5) -> str:
    """Number with `digits` significant digits (1.2346e-10)."""
    return f"{v:.{digits}g}"


def mpa(v: float) -> str:
    return f"{fmt_float(v / 1e6, 3)} MPa"


def mm(v: float) -> str:
    return f"{fmt_float(v * 1000, 4)} mm"


def integer(p: dict[str, Any], key: str, minimum: int = 1) -> int:
    """Pure integer (number of teeth, of cutting edges) >= minimum."""
    raw = p.get(key)
    if raw is None:
        raise ToolError("INVALID_ARGUMENT", f"Missing params.{key} (integer).")
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not float(raw).is_integer():
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be an integer.")
    if int(raw) < minimum:
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be >= {minimum}.")
    return int(raw)


def only(p: dict[str, Any], mode: str, allowed: set[str], field: str = "mode") -> None:
    """For operations with `mode` (or `type`, `cycle`): keys not used by that mode give INVALID_ARGUMENT."""
    extra = sorted(set(p) - allowed - {field})
    if extra:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Keys not used by {field} '{mode}': {', '.join(extra)}. Keys of this {field}: {', '.join(sorted(allowed))}.",
        )


_TEMP_UNITS = {
    "K": (1.0, 0.0),
    "C": (1.0, 273.15),
    "°C": (1.0, 273.15),
    "F": (5 / 9, 273.15 - 32 * 5 / 9),
    "°F": (5 / 9, 273.15 - 32 * 5 / 9),
}


def temperature(p: dict[str, Any], key: str, *, need: bool = True) -> float | None:
    """Absolute temperature in K: number in K or [value, "C" / "K" / "F"]."""
    raw = p.get(key)
    if raw is None:
        if not need:
            return None
        raise ToolError(
            "INVALID_ARGUMENT", f'Missing params.{key} (temperature in K or [value, "C"]).'
        )
    unit = "K"
    if isinstance(raw, list | tuple):
        if len(raw) != 2 or raw[1] not in _TEMP_UNITS:
            raise ToolError(
                "INVALID_ARGUMENT",
                f'params.{key}: temperature as a number in K or [value, "C"], "K", "F".',
            )
        raw, unit = raw
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(float(raw)):
        raise ToolError("INVALID_ARGUMENT", f"params.{key} must be a number.")
    scale, offset = _TEMP_UNITS[unit]
    t = float(raw) * scale + offset
    if t <= 0:
        raise ToolError("INVALID_ARGUMENT", f"params.{key}: temperature below absolute zero.")
    return t


def log_ratio(a: float, b: float) -> float:
    """ln(a / b) for a, b > 0 without the rounding of a / b when a is close to b (D79).

    For b / 2 <= a <= 2 b the difference a - b is exact (Sterbenz), so log1p((a - b) / b) keeps all the digits
    that log(a / b) loses: with a / b = 1 + 1e-9 the plain form is right to about 7 digits only.
    """
    if 0.5 * b <= a <= 2 * b:
        return math.log1p((a - b) / b)
    return math.log(a / b)


def sq_diff(a: float, b: float) -> float:
    """a^2 - b^2 as (a - b)(a + b): no cancellation for close a and b, where a - b is exact (D80)."""
    return (a - b) * (a + b)


def pow4_diff(a: float, b: float) -> float:
    """a^4 - b^4 as (a - b)(a + b)(a^2 + b^2), without cancellation for close a and b (D80)."""
    return (a - b) * (a + b) * (a * a + b * b)


def cube_diff(a: float, b: float) -> float:
    """a^3 - b^3 as (a - b)(a^2 + ab + b^2), without cancellation for close a and b (D84)."""
    return (a - b) * (a * a + a * b + b * b)


def acosh1p(y: float) -> float:
    """acosh(1 + y) for y >= 0 as log1p(y + sqrt(y (y + 2))): no rounding of 1 + y when y is small (D82)."""
    if y < 1:
        return math.log1p(y + math.sqrt(y * (y + 2)))
    return math.acosh(1 + y)
