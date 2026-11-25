"""Number formatting in English (1,234.56) and conversions to strings."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Any

import sympy as sp

from . import limits

_NUM = re.compile(r"(-?)(\d+)(?:\.(\d+))?")


def en_number(text: str) -> str:
    """'1234567.89' -> '1,234,567.89'. Non-numeric strings are returned unchanged."""
    m = _NUM.fullmatch(text)
    if not m:
        return text
    sign, intpart, dec = m.groups()
    groups = []
    while len(intpart) > 3:
        groups.insert(0, intpart[-3:])
        intpart = intpart[:-3]
    groups.insert(0, intpart)
    out = sign + ",".join(groups)
    if dec:
        out += "." + dec
    return out


def fmt_float(value: float, digits: int = 6) -> str:
    try:
        d = Decimal(repr(value)).quantize(Decimal(1).scaleb(-digits), ROUND_HALF_UP)
    except InvalidOperation:
        # beyond the 28 digits of the Decimal context (from about 1e22), or infinite: scientific notation (D84)
        return f"{value:.6g}"
    return en_number(dec_str(d))


def dec_str(d: Decimal) -> str:
    """Decimal without scientific notation and without useless trailing zeros."""
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in {"-0", ""}:
        s = "0"
    return s


def to_decimal(value: sp.Expr, places: int) -> tuple[Decimal, bool]:
    """Real value -> Decimal rounded with ROUND_HALF_UP, and a flag telling whether it is exact."""
    quant = Decimal(1).scaleb(-places)
    if value.is_Rational:
        p, q = int(value.p), int(value.q)  # type: ignore[attr-defined]
        with localcontext() as ctx:
            ctx.prec = len(str(abs(p))) + places + 30 if abs(p) < 10**4000 else 6000
            exact = Decimal(p) / Decimal(q)
            rounded = exact.quantize(quant, ROUND_HALF_UP)
        is_exact = rounded * q == p
        return rounded, bool(is_exact)
    approx = sp.N(value, places + 30)
    with localcontext() as ctx:
        ctx.prec = places + 60
        rounded = Decimal(str(approx)).quantize(quant, ROUND_HALF_UP)
    return rounded, False


def int_text(n: int) -> str:
    """Integer as text; beyond 10,000 digits: a summary with the first and last digits."""
    if n.bit_length() <= int(limits.INT_SUMMARY_DIGITS * 3.3219):
        s = str(n)
        if len(s) <= limits.INT_SUMMARY_DIGITS:
            return s
    d = Decimal(n)
    count = d.adjusted() + 1
    with localcontext() as ctx:
        ctx.prec = 50
        head = format(+d, "f")[:50].lstrip("-")
    tail = str(abs(n) % 10**50).zfill(50)
    sign = "-" if n < 0 else ""
    return f"{sign}{head}...{tail} ({count} digits)"


def sstr(obj: Any) -> Any:
    """SymPy objects -> strings in SymPy syntax (recursive on lists and dicts)."""
    if isinstance(obj, dict):
        return {str(k): sstr(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [sstr(v) for v in obj]
    if isinstance(obj, sp.Integer):
        return int_text(int(obj))
    if isinstance(obj, sp.Basic):
        return str(obj)
    return obj
