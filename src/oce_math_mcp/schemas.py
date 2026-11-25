"""JSON-Schema contract: loading, export and validation of the arguments."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any

from .errors import ToolError

EXAMPLES: dict[str, str] = {
    "math_eval": '{"expr": "sqrt(2)*sind(45)"}',
    "math_solve": '{"operation": "solve", "expr": "x**2 - 5*x + 6 = 0"}',
    "number_theory": '{"operation": "factorint", "n": 360}',
    "math_numeric": '{"operation": "evalf", "expr": "pi", "digits": 50}',
    "math_verify": '{"operation": "equivalent", "expr": "sin(x)**2", "expected": "1 - cos(x)**2"}',
    "python_sandbox": '{"code": "print(sum(range(10)))"}',
    "lean_check": '{"code": "theorem t : 2 + 2 = 4 := by norm_num"}',
    "date_calc": '{"operation": "age", "birth_date": "2015-03-10"}',
    "geom3d_calc": '{"operation": "angle_between", "v1": [1, 0, 0], "v2": [0, 1, 0]}',
    "eng_calc": '{"operation": "beam_calc", "params": {}}',
}


@cache
def schemas_text() -> str:
    """Exact text of the package `tools.json` file (byte-exact contract)."""
    return resources.files("oce_math_mcp").joinpath("tools.json").read_text(encoding="utf-8")


@cache
def _tools() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(schemas_text()))


def load_tools() -> list[dict[str, Any]]:
    return [json.loads(json.dumps(t)) for t in _tools()]


def tool_names() -> list[str]:
    return [t["name"] for t in _tools()]


def get_tool(name: str) -> dict[str, Any]:
    for t in _tools():
        if t["name"] == name:
            return t
    raise ToolError(
        "INVALID_ARGUMENT",
        f"Unknown tool '{name}'. Available tools: {', '.join(tool_names())}.",
        EXAMPLES["math_eval"],
    )


def dump(fmt: str = "mcp") -> str:
    tools = load_tools()
    if fmt == "openai":
        tools = [{"type": "function", "function": t} for t in tools]
    elif fmt != "mcp":
        raise ValueError("unknown format: use 'mcp' or 'openai'")
    return json.dumps(tools, ensure_ascii=False, indent=2)


_PY_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def _type_ok(value: Any, expected: str | list[str]) -> bool:
    kinds = expected if isinstance(expected, list) else [expected]
    for kind in kinds:
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "boolean" and isinstance(value, bool):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "number" and isinstance(value, int | float) and not isinstance(value, bool):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
    return False


def _coerce(value: Any, expected: str | list[str]) -> Any:
    """Accepts 2.0 where an integer is needed (some clients send numbers as floats)."""
    kinds = expected if isinstance(expected, list) else [expected]
    if "integer" in kinds and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def validate(name: str, args: Any) -> dict[str, Any]:
    """Checks the arguments against the schema and applies the default values."""
    tool = get_tool(name)
    example = EXAMPLES.get(name)
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError("INVALID_ARGUMENT", "The arguments must be a JSON object.", example)
    params = tool["parameters"]
    props: dict[str, Any] = params["properties"]
    unknown = [k for k in args if k not in props]
    if unknown:
        raise ToolError(
            "INVALID_ARGUMENT",
            f"Unknown field '{unknown[0]}'. Valid fields: {', '.join(props)}.",
            example,
        )
    for req in params.get("required", []):
        if req not in args:
            raise ToolError("INVALID_ARGUMENT", f"Missing required field '{req}'.", example)
    out: dict[str, Any] = {}
    for key, spec in props.items():
        if key not in args:
            if "default" in spec:
                out[key] = spec["default"]
            continue
        value = args[key]
        if value is None:
            continue
        if "type" in spec:
            value = _coerce(value, spec["type"])
            if not _type_ok(value, spec["type"]):
                raise ToolError(
                    "INVALID_ARGUMENT",
                    f"The field '{key}' must be of type {spec['type']}, not {type(value).__name__}.",
                    example,
                )
        if "enum" in spec and value not in spec["enum"]:
            allowed = ", ".join(str(v) for v in spec["enum"])
            raise ToolError(
                "INVALID_ARGUMENT",
                f"Value '{value}' not allowed for '{key}'. Allowed values: {allowed}.",
                example,
            )
        if "minimum" in spec and isinstance(value, int | float) and value < spec["minimum"]:
            raise ToolError("INVALID_ARGUMENT", f"'{key}' must be >= {spec['minimum']}.", example)
        if "maximum" in spec and isinstance(value, int | float) and value > spec["maximum"]:
            raise ToolError("INVALID_ARGUMENT", f"'{key}' must be <= {spec['maximum']}.", example)
        items = spec.get("items")
        if isinstance(value, list) and isinstance(items, dict) and "type" in items:
            value = [_coerce(v, items["type"]) for v in value]
            for v in value:
                if not _type_ok(v, items["type"]):
                    raise ToolError(
                        "INVALID_ARGUMENT",
                        f"The items of '{key}' must be of type {items['type']}.",
                        example,
                    )
        out[key] = value
    return out
