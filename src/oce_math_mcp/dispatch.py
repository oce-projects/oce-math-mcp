"""Execution of a tool in the current process: validation, computation, uniform response."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from . import schemas
from .envelope import Ctx, build_error, build_ok
from .errors import ToolError

log = logging.getLogger("oce_math_mcp")

Handler = Callable[[dict[str, Any], Ctx], dict[str, Any]]


def _handlers() -> dict[str, Handler]:
    from .tools import (
        date_calc,
        eng_calc,
        geom3d_calc,
        lean_check,
        math_eval,
        math_numeric,
        math_solve,
        math_verify,
        number_theory,
        python_sandbox,
    )

    return {
        "math_eval": math_eval.handle,
        "math_solve": math_solve.handle,
        "number_theory": number_theory.handle,
        "math_numeric": math_numeric.handle,
        "math_verify": math_verify.handle,
        "python_sandbox": python_sandbox.handle,
        "lean_check": lean_check.handle,
        "date_calc": date_calc.handle,
        "geom3d_calc": geom3d_calc.handle,
        "eng_calc": eng_calc.handle,
    }


_HANDLERS: dict[str, Handler] | None = None


def handlers() -> dict[str, Handler]:
    global _HANDLERS
    if _HANDLERS is None:
        _HANDLERS = _handlers()
    return _HANDLERS


def _generic_steps(
    ctx: Ctx, operation: str | None, args: dict[str, Any], result: dict[str, Any]
) -> None:
    shown = {
        k: v
        for k, v in args.items()
        if k not in {"show_steps"} and v not in (None, [], {}, 0, False)
    }
    ctx.step(f"Data: {shown}.")
    ctx.step(
        f"Method: operation '{operation}' computed with exact or verified numerical algorithms."
    )
    main = {k: v for k, v in result.items() if k not in {"steps"}}
    ctx.step(f"Result: {main}.")
    if "verified" in result:
        outcome = "passed" if result["verified"] else "failed: check the result with math_verify"
        ctx.step(f"Back-check: {outcome}.")
    else:
        ctx.step("Check: no automatic check for this operation; use math_verify if needed.")
    ctx.warn("Short steps: this operation has no detailed step-by-step explanation.")


def run_tool(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Runs the tool and always returns the 9-field response (never exceptions)."""
    operation = args.get("operation") if isinstance(args, dict) else None
    op_str = operation if isinstance(operation, str) else None
    ctx = Ctx()
    try:
        clean = schemas.validate(name, args)
        ctx.show_steps = bool(clean.get("show_steps", False))
        result = handlers()[name](clean, ctx)
        if ctx.show_steps and not ctx.steps:
            _generic_steps(ctx, op_str, clean, result)
        return build_ok(name, op_str, result, ctx)
    except ToolError as exc:
        return build_error(name, op_str, exc.code, exc.message, exc.example, ctx.warnings)
    except RecursionError:
        return build_error(
            name, op_str, "LIMIT_EXCEEDED", "Expression or computation nested too deeply."
        )
    except MemoryError:
        return build_error(name, op_str, "LIMIT_EXCEEDED", "Out of memory: computation too large.")
    except NotImplementedError as exc:
        msg = str(exc).splitlines()[0][:300] if str(exc) else "method not available"
        return build_error(name, op_str, "MATH_ERROR", f"SymPy cannot solve this case: {msg}.")
    except (ZeroDivisionError, ArithmeticError, ValueError, TypeError) as exc:
        msg = str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
        return build_error(name, op_str, "MATH_ERROR", f"{type(exc).__name__}: {msg}")
    except Exception as exc:  # pragma: no cover - safety net
        log.exception("internal error in %s", name)
        return build_error(
            name, op_str, "INTERNAL_ERROR", f"Internal error ({type(exc).__name__})."
        )
