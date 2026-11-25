"""Typical constructs of TIR programs (NuminaMath style): the sandbox must run them (D13, D38).

List born from the probe of 2026-11-24: 29 constructs out of 30 pass, `class` stays forbidden.
"""

from __future__ import annotations

import pytest

from oce_math_mcp.dispatch import run_tool

CASES = {
    "star_import": "from sympy import *\nx = symbols('x')\nprint(solve(x**2 - 4, x))",
    "from_import": (
        "from sympy import symbols, solve, Eq\nx, y = symbols('x y')\n"
        "print(solve([Eq(x + y, 10), Eq(x - y, 2)], [x, y]))"
    ),
    "sp_rational": "import sympy as sp\nprint(sp.Rational(1, 3) + sp.Rational(1, 6))",
    "itertools": "import itertools\nprint(sum(1 for p in itertools.permutations(range(5)) if p[0] < p[1]))",
    "fractions": "from fractions import Fraction\nprint(sum(Fraction(1, k) for k in range(1, 11)))",
    "math_comb": "import math\nprint(math.comb(10, 3), math.gcd(84, 36), math.isqrt(99))",
    "def_func": "def f(n):\n    return n * (n + 1) // 2\nprint(f(100))",
    "while_loop": "n = 0\ni = 1\nwhile i < 1000:\n    n += i\n    i *= 2\nprint(n)",
    "list_comp": "print([k for k in range(2, 50) if all(k % d for d in range(2, k))][:10])",
    "dict_comp": "d = {k: k**2 for k in range(5)}\nprint(d)",
    "fstring": "x = 3.14159\nprint(f'{x:.2f}')",
    "sp_simplify": "import sympy as sp\nx = sp.symbols('x')\nprint(sp.simplify((x**2 - 1)/(x - 1)))",
    "sp_integrate": "import sympy as sp\nx = sp.symbols('x')\nprint(sp.integrate(sp.sin(x)**2, (x, 0, sp.pi)))",
    "sp_nsimplify": "import sympy as sp\nprint(sp.nsimplify(0.75))",
    "sp_N": "import sympy as sp\nprint(sp.N(sp.pi, 30))",
    "numpy": "import numpy as np\nA = np.array([[2, 1], [1, 3]])\nprint(round(float(np.linalg.det(A)), 9))",
    "collections": "from collections import Counter\nprint(Counter('mississippi').most_common(2))",
    "sympy_primes": "from sympy import isprime, primerange\nprint(sum(primerange(1, 100)), isprime(97))",
    "lambda_map": "print(list(map(lambda t: t**2, range(5))))",
    "try_except": "try:\n    v = 1 / 0\nexcept ZeroDivisionError:\n    v = 'inf'\nprint(v)",
    "decimal": "from decimal import Decimal, getcontext\ngetcontext().prec = 30\nprint(Decimal(1) / Decimal(7))",
    "sympy_solve_set": (
        "import sympy as sp\nx = sp.symbols('x', real=True)\nprint(sp.solveset(x**2 > 4, x, sp.S.Reals))"
    ),
    "sp_symbols_positive": "import sympy as sp\na, b = sp.symbols('a b', positive=True)\nprint(sp.sqrt(a**2 * b**2))",
    "global_var_in_func": (
        "total = 0\ndef add(k):\n    global total\n    total += k\nfor i in range(5):\n    add(i)\nprint(total)"
    ),
    "nested_def_closure": "def mk(n):\n    def g(x):\n        return x + n\n    return g\nprint(mk(5)(3))",
    "math_floor_log": "import math\nprint(math.floor(math.log(1000, 10) + 1e-9))",
    "sorted_key": "print(sorted([(1, 'b'), (2, 'a')], key=lambda t: t[1]))",
    "string_methods": "s = 'Hello'\nprint(s.lower().count('l'), s[::-1])",
    "sympy_matrix": "import sympy as sp\nM = sp.Matrix([[1, 2], [3, 4]])\nprint(M.det(), M.inv())",
}

# Outputs checked by hand on real Python and SymPy, outside the sandbox (same code).
EXPECTED = {
    "star_import": "[-2, 2]",
    "from_import": "{x: 6, y: 4}",
    "sp_rational": "1/2",
    "itertools": "60",
    "fractions": "7381/2520",
    "math_comb": "120 12 9",
    "def_func": "5050",
    "while_loop": "1023",
    "list_comp": "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]",
    "sp_simplify": "x + 1",
    "sp_integrate": "pi/2",
    "sp_nsimplify": "3/4",
    "numpy": "5.0",
    "sympy_primes": "1060 True",
    "sympy_solve_set": "Union(Interval.open(-oo, -2), Interval.open(2, oo))",
    "sp_symbols_positive": "a*b",
    "global_var_in_func": "10",
    "nested_def_closure": "8",
    "math_floor_log": "3",
    "sympy_matrix": "-2 Matrix([[-2, 1], [3/2, -1/2]])",
}


def _plain_python(code: str) -> str:
    """Runs the same code with real Python: it is the reference of the check, test code."""
    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "<tir>", "exec"), {})  # fixed test code, never model text
    return out.getvalue().strip()


@pytest.mark.parametrize("name", sorted(CASES))
def test_tir_construct_runs_like_python(name: str) -> None:
    code = CASES[name]
    resp = run_tool("python_sandbox", {"code": code})
    assert resp["ok"], (name, resp.get("error"))
    got = resp["result"]["stdout"].strip()
    assert got == _plain_python(code), name
    if name in EXPECTED:
        assert got == EXPECTED[name], name


def test_class_stays_forbidden() -> None:
    resp = run_tool(
        "python_sandbox",
        {"code": "class P:\n    def __init__(self, a):\n        self.a = a\nprint(P(3).a)"},
    )
    assert resp["ok"] is False
    assert resp["error"]["code"] == "FORBIDDEN"
