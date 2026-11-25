"""`--self-check`: proves that an installation works, as an MCP client sees it (D85).

The server is started as a child process over stdio with the same Python, then the check runs initialize,
tools/list and two tools/call with known answers (exact arithmetic, and an `eng_calc` operation that goes
through the computing processes). The report is JSON; `ok` is true only if every check passes.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import time
from typing import Any

from . import __version__
from .schemas import tool_names

EXPECTED_TOOLS: tuple[str, ...] = tuple(tool_names())
TIMEOUT_S = 120.0
_EXACT = "3802951800684688204490109616129/3"  # 2**100 + 1/3


def lean_status() -> str:
    """Lean is optional: it is reported, never counted as a failure."""
    from .tools import lean_check

    return (
        "installed" if lean_check.lean_available() else "not installed (optional: --install-lean)"
    )


def _check(name: str, ok: Any, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def child_errlog() -> Any:
    """A stderr with a real file descriptor for the child process (pytest and some hosts replace sys.stderr).

    `mcp.client.stdio.stdio_client` binds `errlog=sys.stderr` when the module is imported, so it is always passed.
    """
    for stream in (sys.stderr, sys.__stderr__):
        try:
            if stream is not None:
                stream.fileno()
                return stream
        except (AttributeError, OSError, ValueError):
            continue
    return open(os.devnull, "w", encoding="utf-8")


def _body(result: Any) -> dict[str, Any]:
    try:
        return json.loads(result.content[0].text)
    except (AttributeError, IndexError, TypeError, ValueError):
        return {}


async def _checks() -> tuple[str | None, int, list[dict[str, Any]]]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "oce_math_mcp"], env=dict(os.environ)
    )
    checks: list[dict[str, Any]] = []
    async with (
        stdio_client(params, errlog=child_errlog()) as (read, write),
        ClientSession(read, write) as session,
    ):
        init = await session.initialize()
        version = init.server_info.version
        checks.append(
            _check("initialize", version == __version__, f"{init.server_info.name} {version}")
        )
        listed = [t.name for t in (await session.list_tools()).tools]
        missing = sorted(set(EXPECTED_TOOLS) - set(listed))
        extra = sorted(set(listed) - set(EXPECTED_TOOLS))
        detail = f"{len(listed)} tools"
        if missing:
            detail += f"; missing: {', '.join(missing)}"
        if extra:
            detail += f"; unexpected: {', '.join(extra)}"
        checks.append(_check("tools/list", not missing and not extra, detail))
        body = _body(await session.call_tool("math_eval", {"expr": "2**100 + 1/3"}))
        exact = body.get("ok") is True and _EXACT in json.dumps(body.get("result"))
        checks.append(_check("math_eval", exact, f"2**100 + 1/3 = {_EXACT}"))
        mirror = {
            "operation": "geometric_optics",
            "params": {"mode": "mirror", "f": 0.1, "d_o": 0.3},
        }
        body = _body(await session.call_tool("eng_calc", mirror))
        d_i = (body.get("result") or {}).get("d_i")
        good = body.get("ok") is True and isinstance(d_i, float) and abs(d_i - 0.15) <= 1e-12
        checks.append(
            _check(
                "eng_calc", good, f"mirror f = 0.1 m, d_o = 0.3 m: d_i = {d_i} m (0.15 expected)"
            )
        )
    return version, len(listed), checks


def run() -> dict[str, Any]:
    """Runs the checks; never raises: a failure is reported with `ok: false` and the error."""
    t0 = time.perf_counter()
    rep: dict[str, Any] = {"ok": False, "server_version": None, "python": platform.python_version()}
    try:
        version, tools, checks = asyncio.run(asyncio.wait_for(_checks(), TIMEOUT_S))
    except Exception as exc:  # the report is the answer: any failure of the client or server
        rep |= {"tools": 0, "checks": [], "error": f"{type(exc).__name__}: {exc}"}
    else:
        rep |= {
            "ok": len(checks) == 4 and all(c["ok"] for c in checks),
            "server_version": version,
            "tools": tools,
            "checks": checks,
        }
    rep["lean"] = lean_status()
    rep["seconds"] = round(time.perf_counter() - t0, 2)
    return rep
