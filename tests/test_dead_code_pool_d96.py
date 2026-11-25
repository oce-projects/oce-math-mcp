"""Parser branches and the pool size (D96).

From the audit of 25/11/2026. The branch for complex literals was reported as dead: 2j is refused first as an
implicit product, but exponent forms (1e5j) do reach it, and 1e400j failed on Fraction(repr(inf)); now it is
exact from the source text, like a float. A redundant `if isinstance(out, sp.Eq): return out` is gone.
OCE_MATH_POOL_SIZE=0 or a negative --pool-size started no computing process, so every call waited for the
timeout and failed; now the pool refuses to start.
"""

from __future__ import annotations

import pytest
import sympy as sp

from oce_math_mcp.errors import ToolError
from oce_math_mcp.pool import ComputePool
from oce_math_mcp.tools.common import expr_of, grammar
from tests.conftest import call, ok


def test_imaginary_literals() -> None:
    # 2j and 0.5j are implicit products (refused); exponent forms are exact, also beyond the double range
    for expr in ("1 + 2j", "0.5j"):
        resp = call("math_eval", expr=expr)
        assert not resp["ok"] and resp["error"]["code"] == "PARSE_ERROR", (expr, resp)
    assert ok("math_eval", expr="1 + 2*I")["value"] == "1 + 2*I"
    g = grammar([])
    assert expr_of("1e5j", g) == 100000 * sp.I
    assert expr_of("1e400j", g) == sp.Integer(10) ** 400 * sp.I  # beyond the double range, exact
    assert expr_of("1.5e2j", g) == 150 * sp.I
    with pytest.raises(ToolError) as exc:
        expr_of("1e99999999j", g)
    assert exc.value.code == "LIMIT_EXCEEDED"


def test_relations_still_parse() -> None:
    r = ok("math_solve", operation="solve", expr="x**2 - 4 = 0")
    assert sorted(r["solutions"]) == ["-2", "2"]
    assert ok("math_solve", operation="inequality", expr="x - 1 > 0")["solution_set"]


def test_pool_size_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ComputePool(size=-1)
    monkeypatch.setenv("OCE_MATH_POOL_SIZE", "0")
    with pytest.raises(ValueError, match="at least 1"):
        ComputePool()
