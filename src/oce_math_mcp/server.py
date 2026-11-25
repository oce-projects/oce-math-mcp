"""MCP server (mcp 2.x, MCPServer, decision D25): lists the 10 tools with the contract schemas."""

from __future__ import annotations

import json
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, TextContent
from mcp_types import Tool as MCPTool

from . import __version__, schemas
from .dispatch import run_tool
from .pool import ComputePool

INSTRUCTIONS = (
    "Deterministic and verifiable mathematics tools. Every response has 9 fixed fields: ok, tool, "
    "operation, result, units, formula_used, formatted, warnings, error. Use show_steps for the steps "
    "and math_verify to check an answer before giving it."
)


class OceMathServer(MCPServer):
    """MCPServer with listing and calling of the tools taken from the JSON-Schema contract."""

    def __init__(self, pool: ComputePool | None) -> None:
        super().__init__(name="oce-math-mcp", version=__version__, instructions=INSTRUCTIONS)
        self._pool = pool

    async def list_tools(self) -> list[MCPTool]:  # type: ignore[override]
        return [
            MCPTool(name=t["name"], description=t["description"], input_schema=t["parameters"])
            for t in schemas.load_tools()
        ]

    def _tool_input_schema(self, name: str) -> dict[str, Any] | None:
        try:
            return schemas.get_tool(name)["parameters"]
        except Exception:
            return None

    async def call_tool(  # type: ignore[override]
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> CallToolResult:
        pool = self._pool
        if pool is not None:
            resp = await anyio.to_thread.run_sync(pool.call, name, arguments)
        else:
            resp = await anyio.to_thread.run_sync(run_tool, name, arguments)
        text = json.dumps(resp, ensure_ascii=False)
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structured_content=resp,
            is_error=not resp["ok"],
        )


def build_server(pool: ComputePool | None) -> OceMathServer:
    return OceMathServer(pool)
