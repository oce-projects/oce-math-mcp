"""Benchmark table: English number format and no hand-written numbers."""

from __future__ import annotations

from oce_math_mcp.benchmark import fmt_num, to_markdown

SAMPLE = {
    "version": "0.1.0",
    "date_utc": "2026-11-24 02:30",
    "python": "3.13.13",
    "system": "Windows-10",
    "cpu": 12,
    "startup_rounds": 3,
    "process_startup_s": [3.064, 2.81, 2.853],
    "process_startup_median_s": 2.853,
    "warm_call_in_pool": {"median_ms": 0.33, "p95_ms": 0.38, "n": 50},
    "call_in_same_process": {"median_ms": 0.09, "p95_ms": 0.14, "n": 50},
    "timeout_1s_response_s": 1.013,
    "timeout_code": "TIMEOUT",
    "restart_after_kill_s": 2.751,
    "wait_after_sandbox_pool_1_s": 2.71,
    "lean_check_s": None,
}


def test_fmt_num_uses_english_separators() -> None:
    assert fmt_num(1234.5678, 2) == "1,234.57"
    assert fmt_num(0.33, 2) == "0.33"
    assert fmt_num(1234567.0, 1) == "1,234,567.0"


def test_markdown_table_has_measured_values() -> None:
    md = to_markdown(SAMPLE)
    assert "| Start of one computing process (median of 3) | 2.853 s |" in md
    assert "3.064 s; 2.810 s; 2.853 s" in md
    assert "1.013 s (code TIMEOUT)" in md
    assert "Lean not installed" in md
    assert md.endswith("\n") and "\r" not in md


def test_markdown_with_lean_time() -> None:
    md = to_markdown({**SAMPLE, "lean_check_s": 16.632})
    assert "| `lean_check` of `2 + 2 = 4` with Mathlib | 16.632 s |" in md
