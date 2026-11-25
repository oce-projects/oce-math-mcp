"""Install and contribution documents (D85) stay true to the code.

An installation guide that quotes an option the program does not have stops the reader at the first step,
so the documents are checked against `--help`, and the issue form against the real tool names.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from oce_math_mcp.cli import main
from oce_math_mcp.schemas import tool_names

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    "README.md",
    "llms-install.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".github/pull_request_template.md",
)
CLONE = "git clone https://github.com/oce-projects/oce-math-mcp.git"


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _cli_options(capsys: pytest.CaptureFixture[str]) -> set[str]:
    with pytest.raises(SystemExit):
        main(["--help"])
    return set(re.findall(r"--[a-z][a-z-]*", capsys.readouterr().out))


def _quoted_options(text: str) -> set[str]:
    """Options written after the program name, as in `uv run oce-math-mcp --self-check`."""
    found: set[str] = set()
    for tail in re.findall(r"oce-math-mcp ([^`\n]*)", text):
        found |= set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", tail))
    return found


def test_documents_quote_only_real_options(capsys: pytest.CaptureFixture[str]) -> None:
    options = _cli_options(capsys)
    for name in DOCS:
        unknown = _quoted_options(_text(name)) - options
        assert not unknown, f"{name}: {sorted(unknown)}"


def test_readme_options_table_is_complete(capsys: pytest.CaptureFixture[str]) -> None:
    options = _cli_options(capsys) - {"--help", "--version"}
    rows = [r for r in _text("README.md").splitlines() if r.startswith("| `--")]
    in_table = {o for r in rows for o in re.findall(r"--[a-z][a-z-]*", r.split("|")[1])}
    assert in_table == options, (sorted(options - in_table), sorted(in_table - options))


def test_guides_use_the_same_clone_command() -> None:
    for name in ("README.md", "llms-install.md", "CONTRIBUTING.md"):
        assert CLONE in _text(name), name


def test_citation_follows_the_last_release() -> None:
    # CITATION.cff describes a released version (D89): the first CHANGELOG entry that is not "in progress"
    released = [
        h
        for h in re.findall(r"^## (\S+ - \S+.*)$", _text("CHANGELOG.md"), re.M)
        if "in progress" not in h
    ]
    version, date = released[0].split(" - ")[:2]
    cff = _text("CITATION.cff")
    assert f'version: "{version}"' in cff and f'date-released: "{date}"' in cff, (version, date)
    current = _text("VERSION").strip()
    top = re.findall(r"^## (\S+) - ", _text("CHANGELOG.md"), re.M)[0]
    assert top == current, (top, current)


def test_issue_form_lists_the_real_tools() -> None:
    form = _text(".github/ISSUE_TEMPLATE/feature_request.yml")
    block = form.split("id: tool", 1)[1].split("validations:", 1)[0]
    listed = re.findall(r"^\s+- (\S+)$", block, flags=re.MULTILINE)
    assert sorted(listed) == sorted(tool_names()), listed
