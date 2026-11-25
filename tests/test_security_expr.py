"""Expression parser security: no eval, only allowed nodes, anti-DoS limits."""

from __future__ import annotations

import builtins
import sysconfig
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from oce_math_mcp.dispatch import run_tool
from tests.conftest import call

ATTACKS = [
    "__import__('os').system('dir')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('C:/Windows/win.ini')",
    "lambda: 1",
    "[x for x in range(3)]",
    "x.real",
    "a[0]",
    "exec('1')",
    "eval('1')",
    "sympify('1')",
    "parse_expr('1')",
    "1 if 1 else 2",
    "{1: 2}",
    "f'{1}'",
    "import os",
    "globals()",
    "getattr(1, 'real')",
    "x := 3",
    "__builtins__",
    "sin.__globals__",
    "'abc'",
    "unit_convert(1, 'm', 'mm').__class__",
]


@pytest.mark.parametrize("expr", ATTACKS)
def test_math_eval_rejects_attack(expr: str) -> None:
    resp = call("math_eval", expr=expr)
    assert not resp["ok"]
    assert resp["error"]["code"] in {"FORBIDDEN", "PARSE_ERROR", "INVALID_ARGUMENT"}


@pytest.mark.parametrize("expr", ["__import__('os')", "x.__class__", "open('a')", "solve(x)"])
def test_math_solve_rejects_attack(expr: str) -> None:
    resp = call("math_solve", operation="simplify", expr=expr)
    assert not resp["ok"]
    assert resp["error"]["code"] in {"FORBIDDEN", "PARSE_ERROR"}


def test_expression_length_limit() -> None:
    assert call("math_eval", expr="1+" * 1000 + "1")["error"]["code"] == "LIMIT_EXCEEDED"


@pytest.mark.parametrize(
    "expr", ["2**10001", "2^20000", "10**(10**5)", "x**20000", "1e99999999", "x*1e-99999999"]
)
def test_exponent_limit(expr: str) -> None:
    resp = call("math_solve", operation="expand", expr=expr)
    assert resp["error"]["code"] == "LIMIT_EXCEEDED"


def test_exponent_at_limit_is_allowed() -> None:
    resp = call("math_eval", expr="2**10000", exact_mode="fraction")
    assert resp["ok"]


def test_implicit_multiplication_has_corrected_example() -> None:
    resp = call("math_eval", expr="2x + 1", variables={"x": 3})
    assert resp["error"]["code"] == "PARSE_ERROR"
    assert resp["error"]["example"] == "2*x + 1"


def test_italian_decimal_comma_has_corrected_example() -> None:
    resp = call("math_eval", expr="3,5 * 2")
    assert resp["error"]["code"] == "PARSE_ERROR"
    assert "3.5" in resp["error"]["example"]


def test_caret_is_power() -> None:
    assert call("math_eval", expr="2^10")["result"]["value"] == "1024"


# Allowed exceptions (D31), all with fixed templates and without model text:
# - SymPy generates code from an integer (number of variables of a polynomial ring);
# - `collections.namedtuple` and `dataclasses` of the standard library, called by numpy,
#   platform and SymPy when they load their modules (field names written in the library).
#   From the sandbox `namedtuple` stays blocked, even with `from collections import *`, and
#   `dataclasses` cannot be imported (D38).
ALLOWED_EXEC_CALLERS = ("sympy/polys/monomials.py",)
_STDLIB = sysconfig.get_paths()["stdlib"].replace("\\", "/").rstrip("/") + "/"
ALLOWED_STDLIB_CALLERS = frozenset(
    {_STDLIB + "collections/__init__.py", _STDLIB + "dataclasses.py"}
)


def _allowed_caller(caller: str) -> bool:
    return (
        any(caller.endswith(ok) for ok in ALLOWED_EXEC_CALLERS)
        or caller in ALLOWED_STDLIB_CALLERS
        or caller.startswith("<frozen")
    )


SENTINEL = "qwzk"


def test_eval_and_exec_never_see_model_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    calls: list[tuple[str, str]] = []
    real = {"eval": builtins.eval, "exec": builtins.exec}

    def trap(name: str) -> Any:
        def _f(*a: Any, **k: Any) -> Any:
            caller = sys._getframe(1).f_code.co_filename.replace("\\", "/")
            code = a[0] if a and isinstance(a[0], str) else ""
            calls.append((caller, code))
            return real[name](*a, **k)

        return _f

    monkeypatch.setattr(builtins, "eval", trap("eval"))
    monkeypatch.setattr(builtins, "exec", trap("exec"))
    q = SENTINEL
    run_tool("math_eval", {"expr": f"sqrt(2)*sind(30) + log(8, 2) + {q} - {q}"})
    run_tool("math_solve", {"operation": "solve", "expr": f"{q}**2 - 5*{q} + 6 = 0"})
    run_tool("math_solve", {"operation": "integrate", "expr": f"{q}*exp({q})", "variables": [q]})
    run_tool("math_solve", {"operation": "factor", "expr": f"{q}**2 - y**2"})
    run_tool(
        "math_verify",
        {"operation": "equivalent", "expr": f"sin({q})**2", "expected": f"1-cos({q})**2"},
    )
    run_tool("number_theory", {"operation": "diophantine", "expr": f"3*{q} + 5*y = 1"})
    # D38: nsimplify and star import in the sandbox, with the sentinel name in the model code.
    run_tool(
        "python_sandbox",
        {
            "code": f"from sympy import *\n{q} = symbols('{q}')\nprint(nsimplify(0.75), solve({q}**2 - 2, {q}))"
        },
    )
    for caller, code in calls:
        assert SENTINEL not in code, (caller, code[:200])
        assert "oce_math_mcp" not in caller, caller
        assert _allowed_caller(caller), caller


def test_sympy_string_parsing_is_neutralised() -> None:
    import sympy

    from oce_math_mcp.errors import ToolError

    assert sympy.sympify("x") == sympy.Symbol("x")
    assert sympy.sympify("12") == 12
    # SymPy uses S('3/2') inside meijerint and hyperexpand: the simple fraction must pass.
    assert sympy.S("3/2") == sympy.Rational(3, 2)
    assert sympy.S("-1/3") == sympy.Rational(-1, 3)
    with pytest.raises(ToolError):
        sympy.sympify("3/2 + x")
    with pytest.raises(ToolError):
        sympy.sympify("__import__('os')")
    with pytest.raises(ToolError):
        sympy.parsing.sympy_parser.parse_expr("x + 1")


SAFE_TEXT = st.text(alphabet=st.characters(codec="utf-8"), max_size=120)
MATHY = st.text(alphabet="0123456789+-*/^()., =<>xyzepiIsqrtlogsincoa_[]'", max_size=60)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(SAFE_TEXT)
def test_fuzz_random_text_never_raises(text: str) -> None:
    resp = run_tool("math_eval", {"expr": text})
    assert isinstance(resp["ok"], bool)
    if not resp["ok"]:
        assert resp["error"]["code"] != "INTERNAL_ERROR", resp["error"]


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(MATHY)
def test_fuzz_math_like_text_never_raises(text: str) -> None:
    for tool, args in (
        ("math_eval", {"expr": text}),
        ("math_solve", {"operation": "simplify", "expr": text}),
    ):
        resp = run_tool(tool, args)
        assert isinstance(resp["ok"], bool)
        if not resp["ok"]:
            assert resp["error"]["code"] != "INTERNAL_ERROR", (text, resp["error"])


def test_sympify_reads_numeric_forms_of_mpmath_identify() -> None:
    """D38: nsimplify hands the mpmath.identify text to sympify; the AST parser reads it, no eval."""
    import sympy

    from oce_math_mcp.errors import ToolError

    assert sympy.sympify("(3/4)") == sympy.Rational(3, 4)
    assert sympy.sympify("(sqrt(2)/2)") == sympy.sqrt(2) / 2
    assert sympy.sympify("((1+sqrt(5))/2)") == (1 + sympy.sqrt(5)) / 2
    assert sympy.sympify("(pi/4)") == sympy.pi / 4
    for bad in ("(x/2)", "(sqrt(2)/2) + __import__('os')", "exp(1)**10**10**10", "3/2 + x"):
        with pytest.raises(ToolError):
            sympy.sympify(bad)
