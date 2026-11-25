"""SI dimensions and factors of the units for math_verify.check_units (D42).

Units come from sympy.physics.units: only Quantity objects looked up by name (no eval),
plus the SI prefixes and a few aliases (min, rpm, degC, degF, Nm, cP, cSt, Wh). A unit text
is read with a hand-written parser: names, numbers, `*`, `/`, `^` or `**`, parentheses.
SymPy measures mass in grams: the factors are scaled to the kilogram.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from functools import cache
from typing import Any

from .errors import ToolError

Dims = dict[str, Fraction]
Unit = tuple[float, Dims]  # (factor to SI, dimensions)

ORDER = (
    "mass",
    "length",
    "time",
    "current",
    "temperature",
    "amount_of_substance",
    "luminous_intensity",
)
SYMBOL = {
    "mass": "kg",
    "length": "m",
    "time": "s",
    "current": "A",
    "temperature": "K",
    "amount_of_substance": "mol",
    "luminous_intensity": "cd",
}
PREFIX = {
    "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "h": 1e2,
    "da": 1e1, "d": 1e-1, "c": 1e-2, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "n": 1e-9, "p": 1e-12,
    "f": 1e-15,
}  # fmt: skip
PREFIXABLE = {"m", "g", "s", "N", "Pa", "J", "W", "V", "A", "ohm", "F", "H", "T", "Wb", "Hz", "C", "L", "l",
              "mol", "eV", "bar", "Wh", "K"}  # fmt: skip
ALIASES: dict[str, tuple[float, str]] = {
    "min": (1.0, "minute"),
    "rpm": (
        2 * math.pi,
        "1/minute",
    ),  # revolutions per minute as angular velocity: 1 rev = 2 pi rad
    "rev": (2 * math.pi, "1"),
    "degC": (1.0, "K"),  # temperature differences: the shifted zero does not matter for dimensions
    "°C": (1.0, "K"),
    "degF": (5 / 9, "K"),
    "°F": (5 / 9, "K"),
    "°": (1.0, "deg"),
    "Nm": (1.0, "N*m"),
    "cP": (1e-3, "Pa*s"),
    "cSt": (1e-6, "m^2/s"),
    "Wh": (1.0, "W*h"),
    "Ohm": (1.0, "ohm"),
    "Ω": (1.0, "ohm"),
}
_MAX_LEN = 100
_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_°µμΩ]+)(?P<pow>\d+)?"
    r"|(?P<op>\*\*|[*/^()·-]))"
)
_EXAMPLE = '{"F": "N", "rho": "kg/m^3", "v": "m/s", "Cd": "1"}'


def warm_up() -> None:
    """Builds the unit table when the computing process starts (not inside the timeout)."""
    _sympy_table()


@cache
def _sympy_table() -> dict[str, Unit]:
    import sympy.physics.units as u
    from sympy.physics.units.quantities import Quantity
    from sympy.physics.units.systems.si import SI, dimsys_SI

    out: dict[str, Unit] = {}
    for name in dir(u):
        q = getattr(u, name)
        if not isinstance(q, Quantity):
            continue
        try:
            deps = dimsys_SI.get_dimensional_dependencies(SI.get_quantity_dimension(q))
            dims = {str(k.name): Fraction(str(v)) for k, v in deps.items() if v != 0}
            scale = float(SI.get_quantity_scale_factor(q)) / 1000.0 ** float(dims.get("mass", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isfinite(scale) and scale > 0 and set(dims) <= set(ORDER):
            out[name] = (scale, dims)
    return out


def _lookup(name: str, depth: int = 0) -> tuple[Unit, bool]:
    """Unit by name; the second value is True for the coulomb written 'C' (maybe degrees Celsius)."""
    table = _sympy_table()
    if name in table:
        return table[name], name == "C"
    if name in ALIASES and depth < 3:
        k, text = ALIASES[name]
        scale, dims, _ = parse_unit(text, depth + 1)
        return (k * scale, dims), False
    for p in sorted(PREFIX, key=len, reverse=True):
        rest = name[len(p) :]
        if name.startswith(p) and rest in PREFIXABLE:
            (scale, dims), _ = _lookup(rest, depth + 1)
            return (PREFIX[p] * scale, dims), False
    raise ToolError("INVALID_ARGUMENT", f"Unknown unit '{name}'.", _EXAMPLE)


def _mul(a: Unit, b: Unit, sign: int = 1) -> Unit:
    dims = dict(a[1])
    for k, v in b[1].items():
        dims[k] = dims.get(k, Fraction(0)) + sign * v
    return (a[0] * b[0] ** sign, {k: v for k, v in dims.items() if v != 0})


def _pow(a: Unit, e: Fraction) -> Unit:
    return (a[0] ** float(e), {k: v * e for k, v in a[1].items() if v * e != 0})


def parse_unit(text: Any, depth: int = 0) -> tuple[float, Dims, bool]:
    """SI factor, dimensions and 'contains C (coulomb)' of a text such as 'kg/m^3' or 'N*m'."""
    if not isinstance(text, str) or not text.strip():
        raise ToolError("INVALID_ARGUMENT", 'Empty unit: for a pure number write "1".', _EXAMPLE)
    if len(text) > _MAX_LEN:
        raise ToolError("LIMIT_EXCEEDED", f"Unit longer than {_MAX_LEN} characters.")
    toks: list[tuple[str, str, str | None]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Unit '{text}': character not allowed at position {pos}.",
                _EXAMPLE,
            )
        pos = m.end()
        if m.group("num"):
            toks.append(("num", m.group("num"), None))
        elif m.group("name"):
            toks.append(("name", m.group("name"), m.group("pow")))
        else:
            toks.append(("op", m.group("op"), None))
    i = 0
    coulomb = False

    def peek() -> tuple[str, str, str | None] | None:
        return toks[i] if i < len(toks) else None

    def take(kind: str, value: str | None = None) -> tuple[str, str, str | None]:
        nonlocal i
        t = peek()
        if t is None or t[0] != kind or (value is not None and t[1] != value):
            raise ToolError("INVALID_ARGUMENT", f"Unit '{text}': invalid notation.", _EXAMPLE)
        i += 1
        return t

    def number() -> Fraction:
        t = peek()
        neg = False
        if t and t[0] == "op" and t[1] == "-":
            take("op", "-")
            neg = True
        v = Fraction(take("num")[1])
        return -v if neg else v

    def exponent() -> Fraction:
        t = peek()
        if t and t[0] == "op" and t[1] == "(":
            take("op", "(")
            v = number()
            t2 = peek()
            if t2 and t2[0] == "op" and t2[1] == "/":
                take("op", "/")
                d = number()
                if d == 0:
                    raise ToolError(
                        "INVALID_ARGUMENT", f"Unit '{text}': exponent with division by zero."
                    )
                v = v / d
            take("op", ")")
            return v
        return number()

    def atom() -> Unit:
        nonlocal coulomb
        t = peek()
        if t is None:
            raise ToolError("INVALID_ARGUMENT", f"Unit '{text}': incomplete.", _EXAMPLE)
        if t[0] == "num":
            take("num")
            return (float(Fraction(t[1])), {})
        if t[0] == "name":
            take("name")
            unit, is_c = _lookup(t[1], depth)
            coulomb = coulomb or is_c
            return _pow(unit, Fraction(int(t[2]))) if t[2] else unit
        take("op", "(")
        v = expr()
        take("op", ")")
        return v

    def term() -> Unit:
        a = atom()
        t = peek()
        if t and t[0] == "op" and t[1] in ("^", "**"):
            take("op")
            a = _pow(a, exponent())
        return a

    def expr() -> Unit:
        a = term()
        while True:
            t = peek()
            if t is None or (t[0] == "op" and t[1] == ")"):
                return a
            if t[0] == "op" and t[1] in ("*", "·", "/"):
                take("op")
                a = _mul(a, term(), -1 if t[1] == "/" else 1)
            else:  # implicit multiplication: "N m"
                a = _mul(a, term())

    result = expr()
    if i != len(toks):
        raise ToolError("INVALID_ARGUMENT", f"Unit '{text}': invalid notation.", _EXAMPLE)
    if not math.isfinite(result[0]) or result[0] <= 0:
        raise ToolError("INVALID_ARGUMENT", f"Unit '{text}': invalid factor.")
    return result[0], result[1], coulomb


def dims_json(d: Dims) -> dict[str, int | str]:
    return {k: (int(d[k]) if d[k].denominator == 1 else str(d[k])) for k in ORDER if k in d}


def si_text(d: Dims) -> str:
    def part(k: str, e: Fraction) -> str:
        return SYMBOL[k] if e == 1 else f"{SYMBOL[k]}^{e}"

    num = [part(k, d[k]) for k in ORDER if d.get(k, 0) > 0]
    den = [part(k, -d[k]) for k in ORDER if d.get(k, 0) < 0]
    top = "*".join(num) or "1"
    if not den:
        return top
    bottom = den[0] if len(den) == 1 else "(" + "*".join(den) + ")"
    return f"{top}/{bottom}"
