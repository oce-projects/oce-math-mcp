"""Safe parser: model text -> SymPy objects, without eval/exec/sympify (D5).

The text goes through `ast.parse` (which executes nothing) and every node is translated by hand.
Only these are allowed: numbers, names, arithmetic operators, comparisons, calls to functions of the
tool's list, lists as arguments. Everything else is FORBIDDEN.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, cast

import sympy as sp

from . import limits
from .errors import ToolError

CONSTANTS: dict[str, Any] = {
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
    "I": sp.I,
    "oo": sp.oo,
    "inf": sp.oo,
    "tau": 2 * sp.pi,
    "GoldenRatio": sp.GoldenRatio,
    "phi": sp.GoldenRatio,
}

_IMPLICIT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![eE][+-]?\d)([A-Za-z_(])")
_COMMA_DEC = re.compile(r"(?<![\w.])(\d+),(\d+)")
_IDENT = re.compile(r"^[^\W\d]\w*$")
_FLOAT_EXP = re.compile(r"[eE]([+-]?\d+)")

_FORBIDDEN_NAMES = {
    ast.Attribute: "attribute access (dot)",
    ast.Subscript: "indexing with square brackets",
    ast.Lambda: "lambda",
    ast.IfExp: "conditional expression",
    ast.Dict: "dictionaries",
    ast.Set: "sets in braces",
    ast.ListComp: "comprehension",
    ast.SetComp: "comprehension",
    ast.DictComp: "comprehension",
    ast.GeneratorExp: "generators",
    ast.JoinedStr: "f-string",
    ast.NamedExpr: "assignment :=",
    ast.BoolOp: "and/or",
    ast.Await: "await",
    ast.Yield: "yield",
    ast.YieldFrom: "yield",
    ast.Starred: "arguments with *",
    ast.Slice: "slice",
}


@dataclass
class Grammar:
    """What is allowed in an expression field."""

    functions: Mapping[str, Callable[..., Any]]
    declared: Mapping[str, Any] = field(default_factory=dict)
    free_symbols: bool = True
    string_args: frozenset[str] = frozenset()
    relations: bool = False


def check_identifier(name: str) -> str:
    if not isinstance(name, str) or not _IDENT.match(name) or "__" in name:
        raise ToolError("INVALID_ARGUMENT", f"Invalid variable name: {name!r}.", '["x", "y"]')
    return name


def _prepare(text: Any) -> str:
    if not isinstance(text, str):
        raise ToolError("INVALID_ARGUMENT", "The expression must be a string.", '"x**2 + 1"')
    if len(text) > limits.EXPR_MAX_CHARS:
        raise ToolError(
            "LIMIT_EXCEEDED",
            f"Expression of {len(text)} characters: the maximum is {limits.EXPR_MAX_CHARS}.",
        )
    src = text.strip()
    if not src:
        raise ToolError("PARSE_ERROR", "Empty expression.", "x**2 + 1")
    if "__" in src:
        raise ToolError("FORBIDDEN", "Names with a double underscore are not allowed.", "x**2 + 1")
    m = _IMPLICIT.search(src)
    if m:
        raise ToolError(
            "PARSE_ERROR",
            f"Implicit multiplication not allowed: '{m.group(0)}'. Write the * sign.",
            _IMPLICIT.sub(r"\1*\2", src),
        )
    return src.replace("^", "**")


def split_equation(src: str) -> tuple[str, str] | None:
    """Splits 'a = b' on the single '=' that is not part of '==', '<=', '>=', '!='."""
    pos = []
    for i, ch in enumerate(src):
        if ch != "=":
            continue
        prev = src[i - 1] if i else ""
        nxt = src[i + 1] if i + 1 < len(src) else ""
        if prev in "<>!=" or nxt == "=":
            continue
        pos.append(i)
    if not pos:
        return None
    if len(pos) > 1:
        raise ToolError("PARSE_ERROR", "More than one '=' in the same equation.", "x**2 = 4")
    i = pos[0]
    return src[:i], src[i + 1 :]


class _Builder:
    def __init__(self, grammar: Grammar, src: str) -> None:
        self.g = grammar
        self.src = src
        self.cache: dict[str, sp.Symbol] = {}

    def build(self, node: ast.AST, *, in_call: str | None = None) -> Any:
        method = getattr(self, "_" + type(node).__name__, None)
        if method is not None:
            return method(node, in_call)
        label = _FORBIDDEN_NAMES.get(type(node), type(node).__name__)
        raise ToolError(
            "FORBIDDEN", f"Construct not allowed in expressions: {label}.", "sqrt(x) + 1"
        )

    def _Expression(self, node: ast.Expression, in_call: str | None) -> Any:
        if isinstance(node.body, ast.Tuple | ast.List):
            if _COMMA_DEC.search(self.src):
                raise ToolError(
                    "PARSE_ERROR",
                    "The decimal comma is not allowed: use the point.",
                    _COMMA_DEC.sub(r"\1.\2", self.src),
                )
            raise ToolError(
                "PARSE_ERROR", "A list is not an expression: separate with ';'.", "x + 1"
            )
        return self.build(node.body)

    def _Constant(self, node: ast.Constant, in_call: str | None) -> Any:
        v = node.value
        if isinstance(v, bool) or v is None or v is Ellipsis or isinstance(v, bytes):
            raise ToolError("FORBIDDEN", f"Constant not allowed: {v!r}.", "1")
        if isinstance(v, str):
            if in_call is not None and in_call in self.g.string_args:
                return v
            raise ToolError("FORBIDDEN", "Quoted text is not allowed here.", "x + 1")
        if isinstance(v, int):
            if v.bit_length() > limits.INT_BITS_MAX:
                raise ToolError("LIMIT_EXCEEDED", "Integer too large.")
            return sp.Integer(v)
        if isinstance(v, float):
            seg = ast.get_source_segment(self.src, node) or repr(v)
            exp = _FLOAT_EXP.search(seg)
            if exp and abs(int(exp.group(1))) > limits.EXPONENT_MAX:
                # 1e99999999 as an exact fraction would need 10**99999999: endless computation.
                raise ToolError(
                    "LIMIT_EXCEEDED", f"Decimal exponent beyond {limits.EXPONENT_MAX}.", "1e300"
                )
            try:
                fr = Fraction(seg.replace("_", ""))
            except ValueError:
                fr = Fraction(repr(v))
            return sp.Rational(fr.numerator, fr.denominator)
        if isinstance(v, complex):
            return sp.Rational(Fraction(repr(v.imag))) * sp.I
        raise ToolError("FORBIDDEN", f"Constant not allowed: {v!r}.", "1")

    def _Name(self, node: ast.Name, in_call: str | None) -> Any:
        name = node.id
        if "__" in name:
            raise ToolError("FORBIDDEN", "Names with a double underscore are not allowed.", "x + 1")
        if name in self.g.declared:
            return self.g.declared[name]
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name in self.g.functions:
            raise ToolError(
                "PARSE_ERROR", f"The function '{name}' must be used with parentheses.", f"{name}(x)"
            )
        if not self.g.free_symbols:
            raise ToolError(
                "INVALID_ARGUMENT",
                f"The variable '{name}' has no value: pass it in 'variables'.",
                f'{{"expr": "2*{name}", "variables": {{"{name}": 3}}}}',
            )
        if name not in self.cache:
            self.cache[name] = sp.Symbol(name)
        return self.cache[name]

    def _UnaryOp(self, node: ast.UnaryOp, in_call: str | None) -> Any:
        v = self.build(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ToolError("FORBIDDEN", "Operator not allowed (not, ~).", "-x")

    def _BinOp(self, node: ast.BinOp, in_call: str | None) -> Any:
        a = self.build(node.left)
        b = self.build(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            if b == 0:
                raise ToolError("MATH_ERROR", "Division by zero.")
            return a / b
        if isinstance(op, ast.FloorDiv):
            if b == 0:
                raise ToolError("MATH_ERROR", "Division by zero.")
            return sp.floor(a / b)
        if isinstance(op, ast.Mod):
            if b == 0:
                raise ToolError("MATH_ERROR", "Modulo per zero.")
            return sp.Mod(a, b)
        if isinstance(op, ast.Pow | ast.BitXor):
            return safe_pow(a, b)
        raise ToolError("FORBIDDEN", "Operator not allowed.", "x**2 + 1")

    def _Compare(self, node: ast.Compare, in_call: str | None) -> Any:
        if not self.g.relations:
            raise ToolError("PARSE_ERROR", "Comparisons are not allowed in this field.", "x**2 - 4")
        ops = {
            ast.Lt: sp.Lt,
            ast.LtE: sp.Le,
            ast.Gt: sp.Gt,
            ast.GtE: sp.Ge,
            ast.Eq: sp.Eq,
            ast.NotEq: sp.Ne,
        }
        left = self.build(node.left)
        parts = []
        for op, comp in zip(node.ops, node.comparators, strict=True):
            fn = ops.get(type(op))
            if fn is None:
                raise ToolError("FORBIDDEN", "Comparison not allowed (is, in).", "x < 2")
            right = self.build(comp)
            parts.append(fn(left, right, evaluate=False))
            left = right
        return parts[0] if len(parts) == 1 else sp.And(*parts)

    def _List(self, node: ast.List, in_call: str | None) -> Any:
        if in_call is None:
            raise ToolError(
                "PARSE_ERROR",
                "Lists are allowed only as function arguments.",
                "mean([1, 2, 3])",
            )
        if len(node.elts) > limits.LIST_MAX:
            raise ToolError("LIMIT_EXCEEDED", f"List beyond {limits.LIST_MAX} elements.")
        return [self.build(e, in_call=in_call) for e in node.elts]

    _Tuple = _List

    def _Call(self, node: ast.Call, in_call: str | None) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ToolError(
                "PARSE_ERROR",
                "Call not allowed: is the * sign missing between the parentheses?",
                "(x + 1)*(x - 1)",
            )
        name = node.func.id
        if "__" in name:
            raise ToolError(
                "FORBIDDEN", "Names with a double underscore are not allowed.", "sin(x)"
            )
        if node.keywords:
            raise ToolError(
                "FORBIDDEN",
                "Keyword arguments are not allowed: use positional arguments only.",
                f"{name}(x)",
            )
        target = self.g.declared.get(name)
        if target is not None and isinstance(target, sp.FunctionClass):
            fn: Callable[..., Any] = target
        elif name in self.g.functions:
            fn = self.g.functions[name]
        else:
            some = ", ".join(sorted(self.g.functions)[:12])
            raise ToolError(
                "FORBIDDEN", f"Function '{name}' not allowed. Some allowed ones: {some}.", "sqrt(x)"
            )
        args = [self.build(a, in_call=name) for a in node.args]
        try:
            return fn(*args)
        except ToolError:
            raise
        except RecursionError:
            raise
        except TypeError as exc:
            raise ToolError(
                "INVALID_ARGUMENT", f"Invalid arguments for {name}: {exc}", f"{name}(x)"
            ) from None
        except (ZeroDivisionError, ValueError, ArithmeticError) as exc:
            raise ToolError("MATH_ERROR", f"{name}: {exc}") from None


def safe_pow(a: Any, b: Any) -> Any:
    """Power with the limits of D6: exponent <= 10,000 and integer result not huge."""
    if isinstance(b, sp.Basic) and b.is_number and b.is_real is not False and b.is_finite:
        try:
            mag = abs(float(cast(Any, b)))
        except (TypeError, ValueError):
            mag = 0.0
        if mag > limits.EXPONENT_MAX:
            raise ToolError(
                "LIMIT_EXCEEDED",
                f"Exponent {b} beyond the limit of {limits.EXPONENT_MAX}.",
                "pow(a, b, m) with number_theory.mod_pow for modular powers",
            )
        if isinstance(a, sp.Integer) and isinstance(b, sp.Integer) and b > 0:
            bits = int(a).bit_length() * int(b)
            if bits > limits.INT_BITS_MAX:
                raise ToolError(
                    "LIMIT_EXCEEDED", "Integer result too large (beyond 4,000,000 bits)."
                )
    return a**b


def parse(text: Any, grammar: Grammar) -> Any:
    """Translates an expression (without '=') into a SymPy object."""
    src = _prepare(text)
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        if _COMMA_DEC.search(src):
            raise ToolError(
                "PARSE_ERROR",
                "The decimal comma is not allowed: use the point.",
                _COMMA_DEC.sub(r"\1.\2", src),
            ) from None
        col = f" (column {exc.offset})" if exc.offset else ""
        raise ToolError(
            "PARSE_ERROR", f"Invalid syntax{col}: {exc.msg}.", "x**2 - 5*x + 6"
        ) from None
    except (ValueError, MemoryError) as exc:
        raise ToolError("PARSE_ERROR", f"Unreadable expression: {exc}.", "x + 1") from None
    return _Builder(grammar, src).build(tree)


def parse_relation(text: Any, grammar: Grammar) -> Any:
    """'a = b' -> Eq(a, b); 'a < b' -> relation; otherwise expression."""
    src = _prepare(text)
    parts = split_equation(src)
    if parts is not None:
        left = parse(parts[0], grammar)
        right = parse(parts[1], grammar)
        return sp.Eq(left, right, evaluate=False)
    g = Grammar(
        grammar.functions, grammar.declared, grammar.free_symbols, grammar.string_args, True
    )
    out = parse(src, g)
    if isinstance(out, sp.Eq):
        return out
    return out


def split_list(text: Any) -> list[str]:
    """'a; b; c' -> ['a', 'b', 'c'] (without empty parts)."""
    if not isinstance(text, str):
        raise ToolError("INVALID_ARGUMENT", "A string is needed.", "x + y = 3; x - y = 1")
    if len(text) > limits.EXPR_MAX_CHARS:
        raise ToolError("LIMIT_EXCEEDED", f"Text beyond {limits.EXPR_MAX_CHARS} characters.")
    return [p.strip() for p in text.split(";") if p.strip()]
