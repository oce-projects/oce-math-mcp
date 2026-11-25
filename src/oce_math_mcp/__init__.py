"""oce-math-mcp: mathematics MCP server for Qwen3.6 and Claude."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path


def _read_version() -> str:
    repo_version = Path(__file__).resolve().parents[2] / "VERSION"
    if repo_version.is_file():
        return repo_version.read_text(encoding="utf-8").strip()
    try:
        return metadata.version("oce-math-mcp")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "0.0.0"


__version__ = _read_version()
