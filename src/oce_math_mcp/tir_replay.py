"""Replay of TIR datasets on the sandbox (oce-math-mcp --tir-replay, decision D13).

Two row formats:

- **tool calls** (Italian datasets): `messages` with `tool_calls` (`calcola` with `expr`,
  or Python code) and the tool answer in the `tool` message. The old `calcola` tool
  already had `math` and `sp`: here the imports are added;
- **NuminaMath-TIR**: `solution` field with ```` ```python ```` blocks followed by their
  ```` ```output ````. If a block uses a name from a previous block (NameError), it is retried with
  the previous blocks in front and the last lines of the output are compared.

The code runs in `python_sandbox` and the result is compared with the one recorded in the dataset.
Files `.jsonl` or `.parquet` (`pyarrow` needed: `uv run --with pyarrow ...`). With `sample` a
random sample with a fixed seed is replayed; with `call` the process pool is used (real timeout).
"""

from __future__ import annotations

import ast
import bisect
import hashlib
import json
import math
import random
import re
import textwrap
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PRELUDE = "import math\nimport sympy as sp\n"
CODE_KEYS = ("expr", "code")
_NOT_FENCE = r"(?:(?!```).)*"
_BLOCK = re.compile(
    rf"```python[ \t]*\r?\n({_NOT_FENCE})```[ \t]*\r?\n\s*```output[ \t]*\r?\n({_NOT_FENCE})```",
    re.S,
)
_ERROR_OUT = re.compile(
    r"Traceback \(most recent call last\)|^(?:[A-Za-z_.]*(?:Error|Exception)\b|Timeout|Timed out\b)",
    re.M,
)

_NOT_TAG = r"(?:(?!</tool_call>).)*"
# OpenMathReasoning (tir part): code between <tool_call> and </tool_call>, then the output block.
_TOOL_CALL = re.compile(
    rf"<tool_call>[ \t]*\r?\n({_NOT_TAG})</tool_call>[ \t]*\r?\n\s*```output[ \t]*\r?\n({_NOT_FENCE})```",
    re.S,
)

Call = Callable[[str, dict[str, Any]], dict[str, Any]]


def _parse(text: Any) -> Any:
    """Text -> Python value: JSON, then Python literal (ast.literal_eval, no eval), then text."""
    if not isinstance(text, str):
        return text
    s = text.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return s


def _number(v: Any) -> int | float | None:
    """Number or None. Integers stay integers: an integer beyond float range is not converted."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return v
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _numbers_equal(a: int | float, b: int | float) -> bool:
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except OverflowError:
        return a == b


def results_equal(a: Any, b: Any) -> bool:
    """Tolerant equality: numbers with relative error 1e-9, dicts, lists and tuples, text."""
    a, b = _parse(a), _parse(b)
    na, nb = _number(a), _number(b)
    if na is not None and nb is not None:
        return _numbers_equal(na, nb)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(results_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(results_equal(x, y) for x, y in zip(a, b, strict=True))
    return str(a).replace(" ", "") == str(b).replace(" ", "")


def _lines(text: str) -> list[str]:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


TRUNCATED = "... (output truncated)"
# Cut markers of long outputs: NuminaMath-TIR "... (output truncated)", OpenMathReasoning
# "<output cut>" (also in the middle of a line).
CUT_MARKERS = (TRUNCATED, "<output cut>")


def _cut(expected: str) -> str | None:
    """Part of the recorded output before the cut marker, or None if it is not cut."""
    s = expected.rstrip()
    for marker in CUT_MARKERS:
        if s.endswith(marker):
            return s[: -len(marker)]
    return None


def is_truncated(expected: str) -> bool:
    return _cut(expected) is not None


def outputs_equal(got: str, expected: str, tail: bool = False) -> bool:
    """Printed output against recorded output: equal lines, or equal values line by line.

    With `tail` only the last lines of `got` are compared (the previous blocks print first).
    A recorded output that ends with a cut marker (`CUT_MARKERS`) counts as the start of the output.
    """
    cut = _cut(expected)
    if cut is not None:
        prefix = "\n".join(_lines(cut.rstrip()))
        text = "\n".join(_lines(got))
        return bool(prefix) and (text.startswith(prefix) or (tail and ("\n" + prefix) in text))
    g, e = _lines(got), _lines(expected)
    if tail and len(g) > len(e):
        g = g[len(g) - len(e) :]
    if g == e:
        return True
    if len(g) == len(e) and all(results_equal(x, y) for x, y in zip(g, e, strict=True)):
        return True
    return results_equal("\n".join(g), "\n".join(e))


# Environment differences seen in NuminaMath-TIR, not caused by the sandbox: IPython warning at the top
# of the output, numpy 2 scalars (`np.float64(1.5)` where numpy 1 printed `1.5`), `Fraction(a, b)`
# shown by the notebook instead of `a/b`, long tuples split over several lines by the notebook.
_IPYTHON_NOISE = re.compile(
    r"^The history saving thread hit an unexpected error.*?History will not be written to the database\.",
    re.M,
)
_NP_SCALAR = re.compile(r"\bnp\.(?:float|int|uint|complex)\d*\(([^()]*)\)")
_NP_BOOL = re.compile(r"\bnp\.(True|False)_")
_FRACTION = re.compile(r"\bFraction\((-?\d+),\s*(-?\d+)\)")
_WS = re.compile(r"\s+")


def _env_fix(text: str) -> str:
    t = _IPYTHON_NOISE.sub("", text)
    t = _NP_SCALAR.sub(r"\1", t)
    t = _NP_BOOL.sub(r"\1", t)
    return _FRACTION.sub(r"\1/\2", t)


def compare_outputs(got: str, expected: str, tail: bool = False) -> str | None:
    """'exact', 'normalised' (equal only after the environment fixes) or None."""
    if outputs_equal(got, expected, tail):
        return "exact"
    g, e = _env_fix(got), _env_fix(expected)
    if outputs_equal(g, e, tail):
        return "normalised"
    gw = _WS.sub("", g)
    if is_truncated(e):
        # the notebook wraps lines inside long structures: the start is compared without spaces, both on
        # the original text (the cut may fall inside "np.float64(") and on the fixed one
        for got_text, exp_text in ((got, expected), (g, e)):
            cw, pw = _WS.sub("", got_text), _WS.sub("", _cut(exp_text) or "")
            if pw and (cw.startswith(pw) or (tail and pw in cw)):
                return "normalised"
        return None
    ew = _WS.sub("", e)
    if ew and (gw == ew or (tail and gw.endswith(ew))):
        return "normalised"
    return None


def _calls(record: dict[str, Any]) -> list[tuple[str, str]]:
    """Pairs (code, expected answer) of one example, in message order."""
    pending: list[str] = []
    pairs: list[tuple[str, str]] = []
    for m in record.get("messages") or []:
        if not isinstance(m, dict):
            continue
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                args = _parse(args)
            if isinstance(args, dict):
                code = next((args[k] for k in CODE_KEYS if isinstance(args.get(k), str)), None)
                if code is not None:
                    pending.append(code)
        if m.get("role") == "tool" and pending:
            pairs.append((pending.pop(0), str(m.get("content", ""))))
    return pairs


def code_format(record: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    """Solution format ("numina", "omr" or "") and pairs (code, recorded output)."""
    sol = record.get("solution")
    if isinstance(sol, str):
        found = _BLOCK.findall(sol)
        if found:
            return "numina", [(c.strip("\r\n"), o.strip("\r\n")) for c, o in found]
    gen = record.get("generated_solution")
    if isinstance(gen, str):
        found = _TOOL_CALL.findall(gen)
        if found:
            return "omr", [(c.strip("\r\n"), o.strip("\r\n")) for c, o in found]
    return "", []


def numina_blocks(record: dict[str, Any]) -> list[tuple[str, str]]:
    """Pairs (code, recorded output): NuminaMath-TIR (```python) or OpenMathReasoning (<tool_call>)."""
    return code_format(record)[1]


def _parquet(path: Path) -> Any:
    try:
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            ".parquet files need pyarrow: uv run --with pyarrow oce-math-mcp --tir-replay ..."
        ) from exc
    return pq.ParquetFile(path)


def _count(path: Path) -> int:
    """Rows of a file without reading them all: metadata for .parquet, non-empty lines for .jsonl."""
    if path.suffix.lower() == ".parquet":
        return int(_parquet(path).metadata.num_rows)
    with path.open(encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def _records(
    path: Path, wanted: list[int] | None = None
) -> Iterator[tuple[int, dict[str, Any] | None]]:
    """(index in the file, row): dict, or None for an unreadable JSONL row.

    With `wanted` (sorted indices) it reads only those rows; for .parquet only the groups that hold them.
    """
    if path.suffix.lower() == ".parquet":
        pf = _parquet(path)
        names = pf.schema_arrow.names
        cols = next(([c] for c in ("solution", "generated_solution", "messages") if c in names), [])
        start = 0
        for rg in range(pf.metadata.num_row_groups):
            size = pf.metadata.row_group(rg).num_rows
            if wanted is None:
                for k, rec in enumerate(pf.read_row_group(rg, columns=cols).to_pylist()):
                    yield start + k, rec
            else:
                local = wanted[
                    bisect.bisect_left(wanted, start) : bisect.bisect_left(wanted, start + size)
                ]
                if local:
                    table = pf.read_row_group(rg, columns=cols)
                    for j in local:
                        yield j, table.slice(j - start, 1).to_pylist()[0]
            start += size
        return
    keep = set(wanted) if wanted is not None else None
    with path.open(encoding="utf-8", errors="replace") as fh:
        j = -1
        for line in fh:
            if not line.strip():
                continue
            j += 1
            if keep is not None and j not in keep:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                yield j, None
                continue
            yield j, rec if isinstance(rec, dict) else None


def _got(resp: dict[str, Any], prefer_value: bool) -> str:
    res = resp["result"]
    if prefer_value and res["value"] is not None:
        return str(res["value"])
    out = res["stdout"]
    return out if out.strip() or res["value"] is None else str(res["value"])


def error_kind(code: str, message: str) -> str:
    """Error kind for the count: exception class, or message without the line number."""
    msg = re.sub(r"\s*\(line \d+\)\.?", "", message).strip()
    if code == "MATH_ERROR":
        return f"{code}: {msg.split(':', 1)[0]}"
    if code == "FORBIDDEN":
        return f"{code}: {msg[:100]}"
    return code


Unit = tuple[str, str, str, list[str], int]  # format, code, expected output, previous blocks, row


def notebook_program(previous: list[str], code: str) -> str:
    """Previous blocks inside `try` (in the notebook a failing block keeps the earlier variables), then the block.

    Blocks that cannot be parsed (code cut in the dataset) stay out: `ast.parse` reads the text
    and does not run it. The sandbox stops (TIMEOUT, LIMIT_EXCEEDED, FORBIDDEN) are not caught.
    """
    parts: list[str] = []
    for block in previous:
        try:
            ast.parse(block)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
        parts.append(
            "try:\n" + textwrap.indent(block, "    ") + "\n    pass\nexcept Exception:\n    pass\n"
        )
    return "".join(parts) + code


def _unit(call: Call, unit: Unit) -> dict[str, Any]:
    """Runs one call and classifies it: equal, different, error, expected_error."""
    kind, code, expected, previous, row = unit
    resp = call("python_sandbox", {"code": PRELUDE + code if kind == "tool" else code})
    used_prev = False
    if kind == "numina" and previous and not resp["ok"] and "NameError" in resp["error"]["message"]:
        resp = call("python_sandbox", {"code": notebook_program(previous, code)})
        used_prev = True
    out: dict[str, Any] = {"row": row, "code": code, "previous": used_prev}
    if kind == "numina" and _ERROR_OUT.search(expected):
        out["outcome"] = "expected_error"
        return out
    if not resp["ok"]:
        out["outcome"] = "error"
        out["error_code"] = resp["error"]["code"]
        out["error"] = resp["error"]["message"][:300]
        out["expected"] = expected[:300]
        return out
    got = _got(resp, prefer_value=kind == "tool")
    try:
        match = (
            ("exact" if results_equal(got.strip(), expected) else None)
            if kind == "tool"
            else compare_outputs(got, expected, used_prev)
        )
    except (ArithmeticError, ValueError, TypeError, MemoryError, RecursionError) as exc:
        # A comparison that fails must not stop the whole replay: it counts as different.
        match = None
        out["comparison_failed"] = f"{type(exc).__name__}: {exc}"[:200]
    same = match is not None
    out["normalised"] = match == "normalised"
    out["outcome"] = "equal" if same else "different"
    out["truncated"] = kind == "numina" and is_truncated(expected)
    if not same:
        out["expected"] = expected[:300]
        out["got"] = got[:300]
    return out


def run(
    paths: list[Path],
    keep: int = 5,
    sample: int | None = None,
    seed: int = 0,
    call: Call | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Replays the files and returns the report with the counts (all produced by the code).

    `keep`: examples saved for every error kind (and for the different ones), with the row number.
    """
    if call is None:
        from .dispatch import run_tool as call
    t0 = time.perf_counter()
    counts = [_count(p) for p in paths]
    total = sum(counts)
    report: dict[str, Any] = {
        "file": [str(p) for p in paths],
        "rows_total": total,
        "rows": 0,
        "rows_unreadable": 0,
        "calls": 0,
        "equal": 0,
        "different": 0,
        "errors": 0,
        "expected_error": 0,
        "with_previous_blocks": 0,
        "equal_on_truncated_output": 0,
        "equal_after_normalisation": 0,
    }
    picked: list[int] | None = None
    if sample is not None and 0 < sample < total:
        picked = sorted(random.Random(seed).sample(range(total), sample))  # noqa: S311 - sampling, not cryptography
        report["sample"] = sample
        report["seed"] = seed
        report["sample_fingerprint"] = hashlib.sha256(
            ",".join(map(str, picked)).encode()
        ).hexdigest()[:16]
    formats: Counter[str] = Counter()
    units: list[Unit] = []
    rows: list[tuple[int, dict[str, Any] | None]] = []
    offset = 0
    for path, count in zip(paths, counts, strict=True):
        wanted = None
        if picked is not None:
            lo, hi = bisect.bisect_left(picked, offset), bisect.bisect_left(picked, offset + count)
            wanted = [g - offset for g in picked[lo:hi]]
        rows += [(offset + j, rec) for j, rec in _records(path, wanted)]
        offset += count
    for i, rec in rows:
        report["rows"] += 1
        if rec is None:
            report["rows_unreadable"] += 1
            continue
        fmt, blocks = code_format(rec)
        if blocks:
            formats[fmt] += 1
            units += [
                ("numina", c, o, [b[0] for b in blocks[:k]], i) for k, (c, o) in enumerate(blocks)
            ]
            continue
        pairs = _calls(rec)
        formats["tool_calls" if pairs else "no_code"] += 1
        units += [("tool", c, o, [], i) for c, o in pairs]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda u: _unit(call, u), units))
    else:
        results = [_unit(call, u) for u in units]
    codes: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    saved: Counter[str] = Counter()
    key = {
        "equal": "equal",
        "different": "different",
        "error": "errors",
        "expected_error": "expected_error",
    }
    for r in results:
        report["calls"] += 1
        report[key[r["outcome"]]] += 1
        if r.get("normalised") and r["outcome"] == "equal":
            report["equal_after_normalisation"] += 1
        if r.get("truncated") and r["outcome"] == "equal":
            report["equal_on_truncated_output"] += 1
        if r["previous"] and r["outcome"] == "equal":
            report["with_previous_blocks"] += 1
        if r["outcome"] == "error":
            codes[r["error_code"]] += 1
            kinds[error_kind(r["error_code"], r["error"])] += 1
        if r["outcome"] in ("error", "different"):
            label = (
                error_kind(r["error_code"], r["error"]) if r["outcome"] == "error" else "DIFFERENT"
            )
            if saved[label] < keep:
                saved[label] += 1
                kept = {
                    k: v
                    for k, v in r.items()
                    if k not in ("outcome", "previous", "truncated", "normalised")
                }
                failures.append({"kind": label, **kept})
    report["formats"] = dict(formats)
    report["errors_by_code"] = dict(codes)
    report["errors_by_kind"] = dict(kinds.most_common(40))
    report["duration_s"] = round(time.perf_counter() - t0, 1)
    report["examples_per_kind"] = keep
    report["failed_examples"] = failures
    return report
