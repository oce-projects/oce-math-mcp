"""Benchmark of the computing processes (oce-math-mcp --benchmark). Measured numbers, not estimates."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from typing import Any

from . import __version__


def fmt_num(value: float, decimals: int) -> str:
    """Number in English format: 1,234.56."""
    text = f"{value:,.{decimals}f}"
    return text


def to_markdown(d: dict[str, Any]) -> str:
    """Benchmark table for docs/BENCHMARK.md, from the data of run()."""

    def s(v: float) -> str:
        return f"{fmt_num(v, 3)} s"

    def ms(v: float) -> str:
        return f"{fmt_num(v, 2)} ms"

    hot, inproc = d["warm_call_in_pool"], d["call_in_same_process"]
    lean = d.get("lean_check_s")
    rows = [
        (
            f"Start of one computing process (median of {d['startup_rounds']})",
            s(d["process_startup_median_s"]),
        ),
        ("Single starts", "; ".join(s(v) for v in d["process_startup_s"])),
        (
            f"Warm `math_eval` call in the pool (median, p95; {hot['n']} calls)",
            f"{ms(hot['median_ms'])}; {ms(hot['p95_ms'])}",
        ),
        (
            f"`math_eval` call in the same process (median, p95; {inproc['n']} calls)",
            f"{ms(inproc['median_ms'])}; {ms(inproc['p95_ms'])}",
        ),
        (
            "Response to an endless loop with a 1 s timeout",
            f"{s(d['timeout_1s_response_s'])} (code {d['timeout_code']})",
        ),
        ("First call after the kill (pool of 1 process)", s(d["restart_after_kill_s"])),
        ("First call after `python_sandbox` (pool of 1)", s(d["wait_after_sandbox_pool_1_s"])),
        (
            "`lean_check` of `2 + 2 = 4` with Mathlib",
            s(lean) if lean is not None else "Lean not installed",
        ),
    ]
    lines = [
        "# oce-math-mcp benchmark",
        "",
        "File written by `oce-math-mcp --benchmark --benchmark-md docs/BENCHMARK.md`: the numbers come",
        "from the program, they are not written by hand.",
        "",
        f"- Version {d['version']}, measured on {d['date_utc']} UTC.",
        f"- Python {d['python']}, {d['system']}, {d['cpu']} logical CPUs.",
        "",
        "| Measurement | Value |",
        "|---|---|",
        *[f"| {a} | {b} |" for a, b in rows],
        "",
    ]
    return "\n".join(lines)


def _ms(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    return {
        "median_ms": round(statistics.median(s) * 1000, 2),
        "p95_ms": round(s[max(0, int(len(s) * 0.95) - 1)] * 1000, 2),
        "min_ms": round(s[0] * 1000, 2),
        "max_ms": round(s[-1] * 1000, 2),
        "n": len(s),
    }


def _wait_ready(pool: Any, n: int, limit_s: float = 180) -> None:
    t0 = time.perf_counter()
    while len(pool.startup_times) < n:
        if time.perf_counter() - t0 > limit_s:
            raise RuntimeError("computing processes not ready")
        time.sleep(0.02)


def run(rounds: int = 3, calls: int = 50) -> dict[str, Any]:
    from .dispatch import run_tool
    from .harden import harden
    from .pool import ComputePool

    harden()
    out: dict[str, Any] = {
        "version": __version__,
        "date_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "python": sys.version.split()[0],
        "system": platform.platform(),
        "cpu": os.cpu_count(),
        "startup_rounds": rounds,
    }
    startups = []
    for _ in range(rounds):
        pool = ComputePool(size=1)
        _wait_ready(pool, 1)
        startups.append(pool.startup_times[0])
        pool.close()
    out["process_startup_s"] = [round(v, 3) for v in startups]
    out["process_startup_median_s"] = round(statistics.median(startups), 3)

    pool = ComputePool(size=1, timeout_s=1.0)
    _wait_ready(pool, 1)
    lat = []
    for k in range(calls):
        t0 = time.perf_counter()
        r = pool.call("math_eval", {"expr": f"{k}+2"})
        lat.append(time.perf_counter() - t0)
        assert r["ok"]
    out["warm_call_in_pool"] = _ms(lat)
    inproc = []
    for k in range(calls):
        t0 = time.perf_counter()
        run_tool("math_eval", {"expr": f"{k}+2"})
        inproc.append(time.perf_counter() - t0)
    out["call_in_same_process"] = _ms(inproc)

    t0 = time.perf_counter()
    r = pool.call("python_sandbox", {"code": "while True:\n    pass"})
    out["timeout_1s_response_s"] = round(time.perf_counter() - t0, 3)
    out["timeout_code"] = r["error"]["code"] if r.get("error") else None
    t1 = time.perf_counter()
    r = pool.call("math_eval", {"expr": "2+2"})
    out["restart_after_kill_s"] = round(time.perf_counter() - t1, 3)

    pool.call("python_sandbox", {"code": "1+1"})
    t2 = time.perf_counter()
    pool.call("math_eval", {"expr": "2+2"})
    out["wait_after_sandbox_pool_1_s"] = round(time.perf_counter() - t2, 3)
    pool.close()

    from .tools import lean_check

    if lean_check.lean_available():
        t3 = time.perf_counter()
        r = run_tool("lean_check", {"code": "theorem t : 2 + 2 = 4 := by norm_num"})
        out["lean_check_s"] = round(time.perf_counter() - t3, 3)
        out["lean_check_valid"] = bool(r.get("result", {}) and r["result"].get("valid"))
    else:
        out["lean_check_s"] = None
    return out
