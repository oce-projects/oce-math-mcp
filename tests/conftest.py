"""Shared fixtures of the oce-math-mcp test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from oce_math_mcp.dispatch import run_tool
from oce_math_mcp.harden import harden

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _harden_session() -> None:
    """The suite runs with the same protections as the computation process."""
    harden()


def call(tool: str, **args: Any) -> dict[str, Any]:
    """Calls a tool in the test process (no pool) and checks the shape of the response."""
    resp = run_tool(tool, args)
    assert list(resp.keys()) == [
        "ok",
        "tool",
        "operation",
        "result",
        "units",
        "formula_used",
        "formatted",
        "warnings",
        "error",
    ]
    assert resp["tool"] == tool
    assert isinstance(resp["formatted"], str)
    assert isinstance(resp["warnings"], list)
    if resp["ok"]:
        assert resp["error"] is None
        assert isinstance(resp["result"], dict)
    else:
        assert resp["result"] is None
        assert set(resp["error"]) == {"code", "message", "example"}
    return resp


def ok(tool: str, **args: Any) -> dict[str, Any]:
    """Calls a tool and requires `ok: true`; returns `result`."""
    resp = call(tool, **args)
    assert resp["ok"], resp["error"]
    return resp["result"]


def err(tool: str, **args: Any) -> str:
    """Calls a tool and requires `ok: false`; returns the error code."""
    resp = call(tool, **args)
    assert not resp["ok"], resp["result"]
    return resp["error"]["code"]
