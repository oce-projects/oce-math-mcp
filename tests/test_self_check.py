"""`oce-math-mcp --self-check`: one command that proves an installation works (D85).

It starts the server as a real MCP client would (stdio, initialize, tools/list, tools/call), so an agent that
installs the server can verify it without an MCP client of its own. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import json

import pytest

from oce_math_mcp import __version__, selfcheck
from oce_math_mcp.cli import main


def test_self_check_passes(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--self-check"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 0, rep
    assert rep["ok"] is True
    assert rep["server_version"] == __version__
    assert rep["tools"] == 10
    names = [c["name"] for c in rep["checks"]]
    assert names == ["initialize", "tools/list", "math_eval", "eng_calc"], names
    assert all(c["ok"] for c in rep["checks"]), rep["checks"]
    assert rep["lean"] == selfcheck.lean_status()


def test_self_check_reports_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(selfcheck, "EXPECTED_TOOLS", (*selfcheck.EXPECTED_TOOLS, "no_such_tool"))
    code = main(["--self-check"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 1 and rep["ok"] is False, rep
    failed = [c["name"] for c in rep["checks"] if not c["ok"]]
    assert failed == ["tools/list"], rep["checks"]
    assert "no_such_tool" in next(c["detail"] for c in rep["checks"] if c["name"] == "tools/list")
