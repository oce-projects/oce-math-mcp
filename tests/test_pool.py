"""Separate computing processes: timeout that kills, memory limit, restart."""

from __future__ import annotations

import time

import pytest

from oce_math_mcp import limits
from oce_math_mcp.pool import ComputePool


@pytest.fixture(scope="module")
def pool() -> ComputePool:
    p = ComputePool(size=2, timeout_s=2.0, memory_mb=1024)
    yield p
    p.close()


def test_default_timeout_is_five_seconds() -> None:
    assert limits.TIMEOUT_S == 5.0
    assert limits.LEAN_TIMEOUT_S == 60.0


def test_pool_returns_envelope(pool: ComputePool) -> None:
    resp = pool.call("math_eval", {"expr": "1/3", "exact_mode": "fraction"})
    assert resp["ok"] and resp["result"]["value"] == "1/3"


def test_infinite_loop_is_killed_and_pool_recovers(pool: ComputePool) -> None:
    t0 = time.perf_counter()
    resp = pool.call("python_sandbox", {"code": "while True:\n    pass"})
    elapsed = time.perf_counter() - t0
    assert resp["error"]["code"] == "TIMEOUT"
    assert elapsed < 2.0 + 3.0
    assert pool.call("math_eval", {"expr": "2+2"})["result"]["value"] == "4"


def test_memory_limit_protects_host(pool: ComputePool) -> None:
    resp = pool.call(
        "python_sandbox",
        {"code": "import numpy as np\na = np.ones((40000, 40000))\nfloat(a.sum())"},
    )
    assert not resp["ok"]
    assert resp["error"]["code"] in {"LIMIT_EXCEEDED", "MATH_ERROR"}
    assert pool.call("math_eval", {"expr": "3*3"})["result"]["value"] == "9"


def test_parallel_calls(pool: ComputePool) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(lambda k: pool.call("math_eval", {"expr": f"{k}*2"}), range(8)))
    assert [r["result"]["value"] for r in out] == [str(k * 2) for k in range(8)]
