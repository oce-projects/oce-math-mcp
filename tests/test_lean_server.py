"""lean_check and real checks of the MCP server (stdio and streamable-http)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import anyio
import pytest

from oce_math_mcp.tools import lean_check
from tests.conftest import ROOT, call

LEAN_READY = lean_check.lean_available()


def test_lean_missing_is_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.setenv("OCE_MATH_LEAN_PROJECT", str(ROOT / "does-not-exist"))
    assert call("lean_check", code="theorem t : 1 = 1 := rfl")["error"]["code"] == "UNAVAILABLE"


def test_lean_code_limit() -> None:
    assert call("lean_check", code="-- x\n" * 5000)["error"]["code"] == "LIMIT_EXCEEDED"


@pytest.mark.skipif(not LEAN_READY, reason="Lean + Mathlib not installed")
def test_lean_valid_and_invalid_proof() -> None:
    r = call("lean_check", code="theorem t : (2 : ℕ) + 2 = 4 := by norm_num")
    assert r["ok"] and r["result"]["valid"] is True
    r = call("lean_check", code="theorem t : (2 : ℕ) + 2 = 5 := by norm_num")
    assert r["ok"] and r["result"]["valid"] is False and r["result"]["errors"]


@pytest.mark.skipif(not LEAN_READY, reason="Lean + Mathlib not installed")
def test_lean_sorry_detected() -> None:
    r = call("lean_check", code="theorem t (n : ℕ) : n + 0 = n := by sorry")
    assert r["result"]["has_sorry"] is True and r["result"]["valid"] is False


@pytest.mark.skipif(not LEAN_READY, reason="Lean + Mathlib not installed")
def test_lean_axioms_are_checked() -> None:
    """D36: valid only with the standard axioms; the check must really run."""
    r = call("lean_check", code="theorem t (p : Prop) : p ∨ ¬p := Classical.em p")
    assert r["result"]["valid"] is True
    assert set(r["result"]["axioms"]) <= lean_check.STANDARD_AXIOMS
    assert "Classical.choice" in r["result"]["axioms"]
    r = call("lean_check", code="theorem t (n : ℕ) : n + 0 = n := by sorry", allow_sorry=True)
    assert "sorryAx" in r["result"]["axioms"] and r["result"]["valid"] is True
    # A comment left open swallows the check: never "valid".
    r = call("lean_check", code="theorem t : 1 = 1 := rfl\n/- open")
    assert r["result"]["valid"] is False and r["result"]["axioms"] is None


def _server_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OCE_MATH_POOL_SIZE"] = "1"
    return env


TIR_CODE = "from sympy import *\nx = symbols('x', real=True)\nprint(solveset(x**2 > 4, x, S.Reals), nsimplify(0.75))"


async def _stdio_roundtrip() -> tuple[list[str], dict, dict]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "oce_math_mcp"], env=_server_env()
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        res = await session.call_tool("math_eval", {"expr": "1/3", "exact_mode": "fraction"})
        tir = await session.call_tool("python_sandbox", {"code": TIR_CODE})
        return (
            [t.name for t in tools.tools],
            json.loads(res.content[0].text),
            json.loads(tir.content[0].text),
        )


def test_stdio_real_roundtrip() -> None:
    t0 = time.perf_counter()
    names, resp, tir = anyio.run(_stdio_roundtrip)
    # D32: with multiprocessing the first child stayed stuck and the first call arrived after 132.6 s.
    assert time.perf_counter() - t0 < 60
    assert len(names) == 10
    assert resp["ok"] and resp["result"]["value"] == "1/3"
    # D38 through the real transport and the computing process: star import, S, comparison, nsimplify.
    assert tir["ok"], tir
    assert (
        tir["result"]["stdout"].strip() == "Union(Interval.open(-oo, -2), Interval.open(2, oo)) 3/4"
    )


def _free_port() -> int:
    with socket.socket() as sck:
        sck.bind(("127.0.0.1", 0))
        return sck.getsockname()[1]


async def _http_roundtrip(url: str) -> tuple[list[str], dict]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with (
        streamable_http_client(url) as streams,
        ClientSession(streams[0], streams[1]) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        res = await session.call_tool("number_theory", {"operation": "is_prime", "n": 97})
        return [t.name for t in tools.tools], json.loads(res.content[0].text)


def test_http_real_roundtrip() -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "oce_math_mcp",
            "--http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=_server_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            with socket.socket() as sck:
                if sck.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.2)
        names, resp = anyio.run(_http_roundtrip, f"http://127.0.0.1:{port}/mcp")
        assert len(names) == 10
        assert resp["ok"] and resp["result"]["is_prime"] is True
    finally:
        proc.terminate()
        proc.wait(timeout=20)
