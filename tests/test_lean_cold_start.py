"""Lean after a restart (D86): the first check loads Mathlib from disk and can pass the 60 s limit.

Measured on 2026-11-25: two checks after hours of inactivity went beyond 60 s, the next ones took 16.9-20.1 s. The
timeout stays (D16), but its message now tells the user what happened and what to do; `--self-check` says whether
Lean is installed at all.
"""

from __future__ import annotations

import pytest

from oce_math_mcp import limits, selfcheck
from oce_math_mcp.tools import lean_check
from tests.conftest import call

LEAN_READY = lean_check.lean_available()


@pytest.mark.skipif(not LEAN_READY, reason="Lean + Mathlib not installed")
def test_timeout_message_explains_the_cold_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limits, "LEAN_TIMEOUT_S", 0.05)
    err = call("lean_check", code="theorem t : (1 : ℕ) = 1 := rfl")["error"]
    assert err["code"] == "TIMEOUT", err
    msg = err["message"]
    assert msg.startswith("Lean beyond 0 s."), msg
    assert "Mathlib" in msg and "again" in msg, msg


def test_self_check_reports_lean(monkeypatch: pytest.MonkeyPatch) -> None:
    for ready in (True, False):
        monkeypatch.setattr(lean_check, "lean_available", lambda ready=ready: ready)
        assert selfcheck.lean_status() == (
            "installed" if ready else "not installed (optional: --install-lean)"
        )
