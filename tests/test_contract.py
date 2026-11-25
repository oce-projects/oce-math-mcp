"""Schema contract: repo file, spec, dataset mirror, --dump-schemas and server must match."""

from __future__ import annotations

import json

import anyio
import pytest

import oce_math_mcp
from oce_math_mcp import schemas
from oce_math_mcp.server import build_server
from oce_math_mcp.sync_spec import env_path
from tests.conftest import ROOT, call

SPEC_JSON = ROOT / "docs" / "spec" / "oce_math_mcp_tools.json"
SPEC_MD = ROOT / "docs" / "spec" / "SPEC_OCE_MATH_MCP_V3.md"
DATASET_DIR = env_path("OCE_MATH_DATASET_DIR")

NAMES = [
    "math_eval",
    "math_solve",
    "number_theory",
    "math_numeric",
    "math_verify",
    "python_sandbox",
    "lean_check",
    "date_calc",
    "geom3d_calc",
    "eng_calc",
]


def test_package_schema_equals_spec_file() -> None:
    assert schemas.schemas_text().encode("utf-8") == SPEC_JSON.read_bytes()


def test_spec_markdown_embeds_same_json() -> None:
    md = SPEC_MD.read_text(encoding="utf-8")
    start = md.index("<!-- SCHEMAS-START -->")
    block = md[start:].split("```json\n", 1)[1].split("```\n", 1)[0]
    assert block == SPEC_JSON.read_text(encoding="utf-8")


@pytest.mark.skipif(
    DATASET_DIR is None or not DATASET_DIR.exists(),
    reason="OCE_MATH_DATASET_DIR not set or not reachable",
)
def test_dataset_mirror_is_identical() -> None:
    assert DATASET_DIR is not None
    assert (DATASET_DIR / "oce_math_mcp_tools.json").read_bytes() == SPEC_JSON.read_bytes()
    assert (DATASET_DIR / "SPEC_OCE_MATH_MCP_V3.md").read_bytes() == SPEC_MD.read_bytes()


def test_ten_tools_in_order() -> None:
    assert [t["name"] for t in schemas.load_tools()] == NAMES


def test_dump_schemas_default_format() -> None:
    assert json.loads(schemas.dump("mcp")) == json.loads(SPEC_JSON.read_text(encoding="utf-8"))


def test_dump_schemas_openai_format() -> None:
    out = json.loads(schemas.dump("openai"))
    assert [t["type"] for t in out] == ["function"] * 10
    assert [t["function"]["name"] for t in out] == NAMES
    assert out[0]["function"]["parameters"] == schemas.load_tools()[0]["parameters"]


def test_server_lists_exact_schemas() -> None:
    server = build_server(pool=None)
    tools = anyio.run(server.list_tools)
    assert [t.name for t in tools] == NAMES
    by_name = {t["name"]: t for t in schemas.load_tools()}
    for t in tools:
        assert t.input_schema == by_name[t.name]["parameters"]
        assert t.description == by_name[t.name]["description"]


def test_version_matches_file() -> None:
    assert oce_math_mcp.__version__ == (ROOT / "VERSION").read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("name", [n for n in NAMES if n not in {"python_sandbox", "lean_check"}])
def test_missing_required_field_is_invalid_argument(name: str) -> None:
    resp = call(name)
    assert not resp["ok"]
    assert resp["error"]["code"] == "INVALID_ARGUMENT"


def test_unknown_field_rejected_with_list_of_valid_fields() -> None:
    resp = call("math_eval", expr="1+1", expression="1+1")
    assert resp["error"]["code"] == "INVALID_ARGUMENT"
    assert "expression" in resp["error"]["message"]
    assert "expr" in resp["error"]["message"]


def test_enum_violation_rejected() -> None:
    resp = call("math_solve", operation="integral", expr="x")
    assert resp["error"]["code"] == "INVALID_ARGUMENT"
    assert "integrate" in resp["error"]["message"]


def test_unknown_tool() -> None:
    from oce_math_mcp.dispatch import run_tool

    resp = run_tool("calculate", {"expr": "1"})
    assert resp["ok"] is False
    assert resp["error"]["code"] == "INVALID_ARGUMENT"


def test_eng_calc_all_contract_operations_implemented() -> None:
    """D44: since 0.2.0 every operation of the eng_calc enum has code."""
    from oce_math_mcp.tools import eng_calc

    tools = json.loads(SPEC_JSON.read_text(encoding="utf-8"))
    enum = next(t for t in tools if t["name"] == "eng_calc")["parameters"]["properties"][
        "operation"
    ]["enum"]
    assert sorted(enum) == sorted(eng_calc.OPS)
    resp = call("eng_calc", operation="material_lookup", params={})
    assert resp["error"]["code"] == "INVALID_ARGUMENT"
