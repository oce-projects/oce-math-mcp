"""Single response shape (spec V3, section 2.1) and the context of one call."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any

FIELDS = (
    "ok",
    "tool",
    "operation",
    "result",
    "units",
    "formula_used",
    "formatted",
    "warnings",
    "error",
)


@dataclass
class Ctx:
    """Collects warnings, units, formula, readable text and steps during a computation."""

    show_steps: bool = False
    warnings: list[str] = field(default_factory=list)
    units: dict[str, str] | None = None
    formula: str | None = None
    formatted: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def step(self, text: str, expr: Any = None) -> None:
        if self.show_steps:
            self.steps.append(
                {
                    "n": len(self.steps) + 1,
                    "text": text,
                    "expr": None if expr is None else str(expr),
                }
            )


def sanitize(obj: Any) -> Any:
    """Converts to JSON types: SymPy and NumPy objects become strings or Python numbers."""
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return "nan" if math.isnan(obj) else ("oo" if obj > 0 else "-oo")
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, Decimal | Fraction):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return sanitize(obj.tolist())
        if isinstance(obj, np.generic):
            return sanitize(obj.item())
    except ImportError:  # pragma: no cover
        pass
    return str(obj)


def build_ok(tool: str, operation: str | None, result: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if ctx.show_steps:
        result = {**result, "steps": ctx.steps}
    return {
        "ok": True,
        "tool": tool,
        "operation": operation,
        "result": sanitize(result),
        "units": ctx.units,
        "formula_used": ctx.formula,
        "formatted": ctx.formatted or "Computation completed.",
        "warnings": list(ctx.warnings),
        "error": None,
    }


def build_error(
    tool: str,
    operation: str | None,
    code: str,
    message: str,
    example: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool,
        "operation": operation,
        "result": None,
        "units": None,
        "formula_used": None,
        "formatted": f"Error {code}: {message}",
        "warnings": list(warnings or []),
        "error": {"code": code, "message": message, "example": example},
    }
