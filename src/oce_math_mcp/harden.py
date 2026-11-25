"""Protections of the computing process: SymPy no longer converts arbitrary text with eval.

SymPy's `sympify("text")` calls `parse_expr`, which uses `eval`. After `harden()` the conversion
accepts only simple names and numbers and builds them without eval; everything else is FORBIDDEN.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from .errors import ToolError

_DONE = False
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_FRAC = re.compile(r"^([+-]?\d{1,50})/(\d{1,50})$")
# Numeric forms that `nsimplify` receives from `mpmath.identify`, for example "(sqrt(2)/2)".
_NUMERIC_TEXT = re.compile(r"^(?:[\d\s+\-*/().]|sqrt|exp|log|pi|E|I)+$")


def _safe_parse_expr(s: Any, local_dict: Any = None, *_args: Any, **_kwargs: Any) -> Any:
    import sympy as sp

    if not isinstance(s, str):
        raise ToolError("FORBIDDEN", "Conversion from text not allowed.")
    t = s.strip()
    if _NAME.match(t) and "__" not in t:
        if isinstance(local_dict, dict) and t in local_dict:
            return local_dict[t]
        return sp.Symbol(t)
    if _NUM.match(t):
        if re.match(r"^[+-]?\d+$", t):
            return sp.Integer(int(t))
        return sp.Float(t)
    frac = _FRAC.match(t)
    if frac:
        # SymPy itself writes S('3/2') in meijerint and hyperexpand: a fraction of two integers, no eval.
        return sp.Rational(int(frac.group(1)), int(frac.group(2)))
    if _NUMERIC_TEXT.match(t):
        # D38: `nsimplify` hands to sympify the form found by `mpmath.identify` ("(3/4)",
        # "(sqrt(2)/2)"). Our allow-list AST parser reads it, with its limits: never eval.
        from .expr import Grammar, parse

        funcs = {"sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log}
        return parse(t, Grammar(funcs, {}, free_symbols=False))
    raise ToolError(
        "FORBIDDEN",
        "Conversion of text to expression not allowed (sympify/parse_expr): use SymPy objects.",
        "x = sympy.Symbol('x'); x + 1",
    )


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise ToolError("FORBIDDEN", "SymPy function that uses eval: not allowed.")


def harden() -> None:
    """Applies the protections only once per process."""
    global _DONE
    if _DONE:
        return
    import sympy
    import sympy.parsing
    import sympy.parsing.sympy_parser as spp

    spp.parse_expr = _safe_parse_expr  # type: ignore[assignment]
    spp.eval_expr = _blocked  # type: ignore[assignment]
    sympy.parsing.parse_expr = _safe_parse_expr  # type: ignore[attr-defined]
    if hasattr(sympy, "parse_expr"):
        sympy.parse_expr = _safe_parse_expr  # type: ignore[attr-defined]
    sys.set_int_max_str_digits(200_000)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 6_000))
    _DONE = True
