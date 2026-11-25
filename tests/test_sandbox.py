"""python_sandbox: AST interpreter with an allow list, without exec."""

from __future__ import annotations

import math
import textwrap

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from oce_math_mcp.dispatch import run_tool
from tests.conftest import call, ok


def run(code: str) -> dict:
    return ok("python_sandbox", code=textwrap.dedent(code))


def test_digit_sum_example() -> None:
    r = run("sum(1 for n in range(1, 1001) if sum(map(int, str(n))) == 10)")
    assert r["value"] == "63"


def test_print_and_last_value() -> None:
    r = run(
        """
        x = 6
        print("x is", x)
        x * 7
        """
    )
    assert r["stdout"] == "x is 6\n"
    assert r["value"] == "42"


def test_def_recursion_and_lru_cache() -> None:
    r = run(
        """
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def fib(n):
            return n if n < 2 else fib(n - 1) + fib(n - 2)

        fib(90)
        """
    )
    a, b = 0, 1
    for _ in range(90):
        a, b = b, a + b
    assert r["value"] == str(a)


def test_sympy_solve() -> None:
    r = run(
        """
        import sympy as sp
        x = sp.symbols('x')
        sorted(sp.solve(x**2 - 5*x + 6, x))
        """
    )
    assert r["value"] == "[2, 3]"


def test_sympy_from_import_and_subs_with_name_string() -> None:
    r = run(
        """
        from sympy import Symbol, Rational
        x = Symbol('x')
        (x + Rational(1, 2)).subs('x', 2)
        """
    )
    assert r["value"] == "5/2"


def test_numpy_and_itertools() -> None:
    r = run(
        """
        import numpy as np
        from itertools import combinations
        a = np.arange(10)
        print(int(a.sum()))
        len(list(combinations(range(10), 3)))
        """
    )
    assert r["stdout"] == "45\n"
    assert r["value"] == str(math.comb(10, 3))


def test_fractions_decimal_collections() -> None:
    r = run(
        """
        from fractions import Fraction
        from collections import Counter
        c = Counter("mississippi")
        (Fraction(1, 3) + Fraction(1, 6), c["s"])
        """
    )
    assert r["value"] == "(Fraction(1, 2), 4)"


def test_mpmath_precision() -> None:
    r = run(
        """
        import mpmath
        mpmath.mp.dps = 50
        str(mpmath.pi)
        """
    )
    assert r["value"].startswith("'3.14159265358979323846264338327950288419716939937")


def test_try_except_while_fstring() -> None:
    r = run(
        """
        n = 0
        while True:
            n += 1
            if n > 4:
                break
        try:
            1 / 0
        except ZeroDivisionError:
            msg = f"n={n}"
        msg
        """
    )
    assert r["value"] == "'n=5'"


def test_comprehensions_lambda_sorted() -> None:
    r = run("sorted({k: k * k for k in range(5)}.items(), key=lambda kv: -kv[1])[0]")
    assert r["value"] == "(4, 16)"


def test_random_is_deterministic() -> None:
    a = run("import random\n[random.randint(1, 100) for _ in range(5)]")["value"]
    b = run("import random\n[random.randint(1, 100) for _ in range(5)]")["value"]
    assert a == b


def test_scipy_optimize() -> None:
    r = run(
        """
        from scipy.optimize import brentq
        import math
        round(brentq(lambda x: math.cos(x) - x, 0, 1), 6)
        """
    )
    assert r["value"] == "0.739085"


ESCAPES = [
    "import os",
    "import subprocess",
    "from os import system",
    "import sys",
    "import operator",
    "import importlib",
    "__import__('os')",
    "open('C:/Windows/win.ini')",
    "().__class__",
    "x = 1\nx.__class__",
    "type(1)",
    "getattr(1, 'real')",
    "eval('1')",
    "exec('1')",
    "compile('1', '', 'exec')",
    "globals()",
    "vars()",
    "class A:\n    pass",
    "with open('x') as f:\n    pass",
    "def g():\n    yield 1",
    "import sympy\nsympy.sympify('1')",
    "import sympy as sp\nsp.parse_expr('1')",
    "import sympy as sp\nsp.lambdify",
    'import sympy as sp\nsp.S(\'__import__("os").system("dir")\')',
    "import sympy as sp\nsp.preview",
    "import numpy as np\nnp.load('x.npy')",
    "import numpy as np\nnp.zeros(3).ctypes",
    "import numpy as np\nnp.zeros(3).tofile('x')",
    "import numpy as np\nnp.lib",
    "import scipy\nscipy.io",
    "'{0.__class__}'.format(1)",
    "'{0}'.format_map({})",
    "import mpmath\nmpmath.identify(1.0, ['__import__(\"os\")'])",
    "(lambda: 0).__code__",
    "g = (i for i in range(3))\ng.gi_frame",
    "print.__self__",
    "import functools\nfunctools.partial(print).func.__module__",
    "x = [1]\nx.__len__()",
    "def f():\n    nonlocal y",
    "async def f():\n    pass",
    "import math\nmath.__dict__",
    "breakpoint()",
    "input()",
]


@pytest.mark.parametrize("code", ESCAPES)
def test_sandbox_blocks_escape(code: str) -> None:
    resp = call("python_sandbox", code=code)
    assert not resp["ok"]
    assert resp["error"]["code"] in {"FORBIDDEN", "MATH_ERROR", "PARSE_ERROR"}


def test_code_length_limit() -> None:
    assert call("python_sandbox", code="x = 1\n" * 2000)["error"]["code"] == "LIMIT_EXCEEDED"


def test_output_is_truncated_with_warning() -> None:
    resp = call("python_sandbox", code="for i in range(5000):\n    print('abcdefghij')")
    assert resp["ok"]
    assert resp["result"]["truncated"] is True
    assert len(resp["result"]["stdout"]) <= 10_000
    assert resp["warnings"]


@pytest.mark.parametrize(
    "code",
    ["'a' * 10**9", "[0] * 10**9", "list(range(10**9))", "2**100000", "10**10**10"],
)
def test_memory_and_size_guards(code: str) -> None:
    assert call("python_sandbox", code=code)["error"]["code"] == "LIMIT_EXCEEDED"


def test_runtime_error_reports_line() -> None:
    resp = call("python_sandbox", code="a = 1\nb = 0\na / b")
    assert resp["error"]["code"] == "MATH_ERROR"
    assert "ZeroDivisionError" in resp["error"]["message"]
    assert "line 3" in resp["error"]["message"]


CODE_FUZZ = st.text(alphabet="abcxyz0123456789 ()[]{}:=+-*/%.,'\"\n_#!<>@", max_size=80)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(CODE_FUZZ)
def test_fuzz_sandbox_never_raises(code: str) -> None:
    resp = run_tool("python_sandbox", {"code": code})
    assert isinstance(resp["ok"], bool)
    if not resp["ok"]:
        assert resp["error"]["code"] != "INTERNAL_ERROR", (code, resp["error"])


# ------------------------------------------------------------------ TIR constructs (D38)


def test_star_import_binds_public_names_but_not_blocked_ones() -> None:
    r = run("from sympy import *\nx = symbols('x')\nprint(solve(x**2 - 4, x))")
    assert r["stdout"].strip() == "[-2, 2]"
    bad = call("python_sandbox", code="from sympy import *\nsympify('x')")
    assert bad["ok"] is False
    # namedtuple uses exec: the star import must not bring it in.
    nt = call("python_sandbox", code="from collections import *\nnamedtuple('P', ['a'])")
    assert nt["ok"] is False


def test_global_statement_in_function() -> None:
    r = run(
        """
        total = 0
        def add(k):
            global total
            total += k
        for i in range(5):
            add(i)
        total
        """
    )
    assert r["value"] == "10"


def test_global_cannot_reach_forbidden_names() -> None:
    resp = call("python_sandbox", code="def f():\n    global __builtins__\n    return 1\nf()")
    assert resp["ok"] is False and resp["error"]["code"] == "FORBIDDEN"


def test_sympy_S_constants_allowed_but_not_text() -> None:
    r = run("import sympy as sp\nprint(sp.S.Reals, sp.S(1) / 3, sp.S.Half)")
    assert r["stdout"].strip() == "Reals 1/3 1/2"
    for code in (
        "import sympy as sp\nsp.S('x + 1')",
        "from sympy import S\nS('1')",
        "import sympy as sp\nsp.S(['x'])",
    ):
        resp = call("python_sandbox", code=code)
        assert resp["ok"] is False and resp["error"]["code"] == "FORBIDDEN", code


def test_comparison_returns_the_object_like_python() -> None:
    """`x**2 > 4` is a SymPy relation, not a true/false: solveset needs it (found by the TIR probe)."""
    r = run(
        """
        import sympy as sp
        x = sp.symbols('x', real=True)
        print(x**2 > 4)
        print(sp.solveset(x**2 > 4, x, sp.S.Reals))
        print(1 < 2 < 3, 1 < 3 < 2)
        """
    )
    lines = r["stdout"].strip().splitlines()
    assert lines == [
        "x**2 > 4",
        "Union(Interval.open(-oo, -2), Interval.open(2, oo))",
        "True False",
    ]


def test_nsimplify_of_numbers_but_not_of_text() -> None:
    r = run("import sympy as sp\nprint(sp.nsimplify(0.75), sp.nsimplify(1.4142135623730951))")
    assert r["stdout"].strip() == "3/4 sqrt(2)"
    resp = call("python_sandbox", code="import sympy as sp\nsp.nsimplify('x + 1')")
    assert resp["ok"] is False
