"""eng_calc, basic sciences (D47 and later): CODATA physical constants.

The table is in `oce_math_mcp.constants` and is the same as `const()` in math_eval.
"""

from __future__ import annotations

from typing import Any

from ..constants import CODATA, SOURCE, SYMBOL, TABLE, lookup, search
from ..envelope import Ctx
from ..errors import ToolError
from .eng_base import op

# ------------------------------------------------------------------ physical_constant


@op("physical_constant", ("name", "search"))
def _physical_constant(p: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    if (p.get("name") is None) == (p.get("search") is None):
        raise ToolError(
            "INVALID_ARGUMENT",
            "Give params.name (CODATA name or short symbol) or params.search (text to search for).",
            '{"name": "h"}',
        )
    if p.get("search") is not None:
        if not isinstance(p["search"], str):
            raise ToolError("INVALID_ARGUMENT", "params.search is a text.")
        everything = search(p["search"], limit=len(TABLE))
        out: dict[str, Any] = {"matches": everything[:25], "count": len(everything)}
        ctx.formula = f"search in the names of the constants {CODATA}"
        ctx.formatted = f"{len(everything)} constants contain '{p['search']}'."
        return out
    key, v, unit, unc = lookup(p["name"])
    out = {
        "name": key,
        "symbol": SYMBOL.get(key),
        "value": v,
        "value_text": repr(v),
        "unit": unit,
        "uncertainty": unc,
        "relative_uncertainty": unc / abs(v) if v else None,
        "exact": unc == 0,
        "codata": CODATA,
        "source": SOURCE,
    }
    u = unit or "1"
    ctx.units = {"value": u, "uncertainty": u, "relative_uncertainty": "1"}
    ctx.formula = f"recommended value {CODATA} (scipy.constants)"
    ctx.formatted = f"{key} = {v!r} {unit}".rstrip() + (
        " (exact)." if unc == 0 else f", standard uncertainty {unc:.2g}."
    )
    return out
