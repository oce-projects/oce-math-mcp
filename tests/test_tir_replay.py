"""Replay of TIR datasets (D13): how many examples the sandbox runs and with what result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oce_math_mcp.tir_replay import results_equal, run


def _record(expr: str, expected: str) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": "compute"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "calcola", "arguments": {"expr": expr}},
                        }
                    ],
                },
                {"role": "tool", "name": "calcola", "content": expected},
                {"role": "assistant", "content": "done"},
            ]
        },
        ensure_ascii=False,
    )


def test_results_equal_tolerates_float_noise_and_json_vs_repr() -> None:
    assert results_equal({"a": 0.1 + 0.2}, {"a": 0.3})
    assert results_equal([1, 2.0], (1, 2))
    assert results_equal("sqrt(2)", "sqrt(2)")
    assert results_equal(12420.62, "12420.62")
    assert not results_equal({"a": 1}, {"a": 2})
    assert not results_equal([1, 2], [1, 2, 3])


def test_run_counts_same_different_and_errors(tmp_path: Path) -> None:
    f = tmp_path / "tir.jsonl"
    lines = [
        _record("round(14050.47*(1-15/100), 2)", "11942.9"),
        _record("{'vat': round(100*22/100, 2)}", '{"vat": 22.0}'),
        _record("sp.sqrt(8)", "2*sqrt(2)"),
        _record("2 + 2", "5"),
        _record("open('x')", "0"),
    ]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = run([f])
    assert rep["calls"] == 5
    assert rep["equal"] == 3
    assert rep["different"] == 1
    assert rep["errors"] == 1
    assert rep["errors_by_code"] == {"FORBIDDEN": 1}
    assert len(rep["failed_examples"]) == 2


def test_run_skips_lines_that_are_not_json(tmp_path: Path) -> None:
    f = tmp_path / "tir.jsonl"
    f.write_text("non json\n" + _record("1 + 1", "2") + "\n", encoding="utf-8")
    rep = run([f])
    assert rep["rows_unreadable"] == 1 and rep["equal"] == 1


# ------------------------------------------------------------- NuminaMath-TIR format (D13)

FENCE = "`" * 3


def _numina(*blocks: tuple[str, str]) -> str:
    """Row in the NuminaMath-TIR format: text with python blocks followed by their output."""
    parts = ["Let us reason."]
    for code, out in blocks:
        parts.append(f"{FENCE}python\n{code}\n{FENCE}\n{FENCE}output\n{out}\n{FENCE}\nSo.")
    return json.dumps({"problem": "p", "solution": "\n".join(parts)}, ensure_ascii=False)


def test_numina_blocks_are_extracted_in_order() -> None:
    from oce_math_mcp.tir_replay import numina_blocks

    rec = json.loads(_numina(("print(1)", "1"), ("x = 2\nprint(x)", "2")))
    assert numina_blocks(rec) == [("print(1)", "1"), ("x = 2\nprint(x)", "2")]
    assert numina_blocks({"solution": "no code"}) == []


def test_numina_run_reuses_previous_blocks_when_a_name_is_missing(tmp_path: Path) -> None:
    f = tmp_path / "numina.jsonl"
    lines = [
        _numina(("from sympy import *\nx = symbols('x')\nprint(solve(x**2 - 4, x))", "[-2, 2]")),
        # the second block uses `a` from the first: alone it gives NameError, with the first in front it works
        _numina(("a = 7\nprint(a * 2)", "14"), ("print(a + 1)", "8")),
        _numina(("print(2 + 2)", "5")),
        # the recorded output is an error: it is not counted as equal or different
        _numina(
            (
                "print(1 / 0)",
                "Traceback (most recent call last):\nZeroDivisionError: division by zero",
            )
        ),
        _numina(("import os\nprint(os.getcwd())", "/home")),
    ]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = run([f])
    assert rep["rows"] == 5
    assert rep["calls"] == 6
    assert rep["equal"] == 3
    assert rep["with_previous_blocks"] == 1
    assert rep["different"] == 1
    assert rep["expected_error"] == 1
    assert rep["errors"] == 1
    assert rep["errors_by_code"] == {"FORBIDDEN": 1}
    assert sum(rep["errors_by_kind"].values()) == rep["errors"] == 1
    assert rep["formats"] == {"numina": 5}
    assert all("expected" in e for e in rep["failed_examples"])


def test_sample_is_reproducible_with_seed(tmp_path: Path) -> None:
    f = tmp_path / "numina.jsonl"
    f.write_text(
        "\n".join(_numina((f"print({i})", str(i))) for i in range(20)) + "\n", encoding="utf-8"
    )
    a = run([f], sample=5, seed=3)
    b = run([f], sample=5, seed=3)
    assert a["rows"] == 5 and a["rows_total"] == 20
    assert (
        a["sample_fingerprint"]
        == b["sample_fingerprint"]
        != run([f], sample=5, seed=4)["sample_fingerprint"]
    )
    assert a["equal"] == 5


def test_parquet_input_is_read(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    sol = json.loads(_numina(("print(6 * 7)", "42")))["solution"]
    pq.write_table(pa.table({"problem": ["p"], "solution": [sol]}), tmp_path / "t.parquet")
    rep = run([tmp_path / "t.parquet"])
    assert rep["rows"] == 1 and rep["equal"] == 1


def test_truncated_expected_output_is_compared_as_prefix() -> None:
    """NuminaMath-TIR records long outputs cut with "... (output truncated)"."""
    from oce_math_mcp.tir_replay import outputs_equal

    full = "R1: 5/2\nR2: 16*sqrt(15)/15\nR3: 13/2"
    assert outputs_equal(full, "R1: 5/2\nR2: 16*sqrt(1 ... (output truncated)")
    assert not outputs_equal(full, "R1: 5/2\nR2: 17 ... (output truncated)")
    # with the previous blocks the recorded output may come after the lines of the earlier blocks
    assert outputs_equal("old\n" + full, "R1: 5/2\nR2 ... (output truncated)", tail=True)


def test_openmathreasoning_output_cut_marker() -> None:
    """OpenMathReasoning cuts long outputs with "<output cut>", also in the middle of a line."""
    from oce_math_mcp.tir_replay import compare_outputs, is_truncated

    full = "LHS: 2*a**2*x**4 + b**2 + 2*c**2\nRHS: 2*a**2*x**4"
    assert is_truncated("LHS: 2*a**2*x**4 + b**2 + 2<output cut>")
    assert compare_outputs(full, "LHS: 2*a**2*x**4 + b**2 + 2<output cut>") == "exact"
    assert compare_outputs(full, "LHS: 2*a**2*x**4 + b**2 + 3<output cut>") is None
    # the notebook wraps lines inside long dicts and matrices: only the text without spaces counts
    got = "{(np.float64(1.5), np.float64(0.0)): np.float64(1.0), (np.float64(0.0), np.float64(2.5)): 3}"
    exp = "{(np.float64(1.5),\n  np.float64(0.0)): np.float64(1.0),\n (np.float64(0.0),\n  np.floa<output cut>"
    assert compare_outputs(got, exp) == "normalised"
    assert compare_outputs(got, "{(np.float64(1.6),\n  np.float64(0.0)<output cut>") is None


def test_failures_keep_row_number_and_a_few_per_kind(tmp_path: Path) -> None:
    f = tmp_path / "numina.jsonl"
    rows = [_numina((f"print({i})", "wrong")) for i in range(4)] + [_numina(("import os", "x"))]
    f.write_text("\n".join(rows) + "\n", encoding="utf-8")
    rep = run([f], keep=2)
    kinds = [e["kind"] for e in rep["failed_examples"]]
    assert kinds.count("DIFFERENT") == 2 and len(kinds) == 3
    assert rep["examples_per_kind"] == 2
    assert [e["row"] for e in rep["failed_examples"] if e["kind"] != "DIFFERENT"] == [4]


def test_environment_differences_are_normalised_but_counted() -> None:
    """Environment differences seen in NuminaMath-TIR (numpy 1, notebook, IPython), not caused by the sandbox."""
    from oce_math_mcp.tir_replay import compare_outputs

    assert compare_outputs("(np.float64(1.5), np.float64(2.0))", "(1.5, 2.0)") == "normalised"
    assert compare_outputs("230/33", "Fraction(230, 33)") == "normalised"
    assert compare_outputs("((a, b), [1, 2])", "((a, b),\n [1, 2])") == "normalised"
    noise = (
        "The history saving thread hit an unexpected error (OperationalError('database is locked'))."
        "History will not be written to the database.\n8"
    )
    assert compare_outputs("8", noise) == "normalised"
    assert compare_outputs("8", "8") == "exact"
    assert compare_outputs("231/33", "Fraction(230, 33)") is None
    assert compare_outputs("np.float64(1.5)", "1.6") is None


def test_dataset_timeout_output_counts_as_expected_error(tmp_path: Path) -> None:
    f = tmp_path / "numina.jsonl"
    f.write_text(
        _numina(("while True:\n    pass", "Timed out after 5 seconds."))
        + "\n"
        # OpenMathReasoning writes only "Timed out"
        + _omr(("while True:\n    pass", "Timed out"))
        + "\n",
        encoding="utf-8",
    )
    rep = run([f])
    assert rep["expected_error"] == 2 and rep["errors"] == 0


def _omr(*blocks: tuple[str, str]) -> str:
    """Row in the OpenMathReasoning format (tir part): code between <tool_call> and </tool_call>."""
    parts = ["<think>Let us reason."]
    for code, out in blocks:
        parts.append(
            f"<tool_call>\n{code}\n</tool_call>\n{FENCE}output\n{out}\n{FENCE}\n"
            f"{FENCE}system\nRemaining code executions: 3.\n{FENCE}\nSo."
        )
    return json.dumps({"problem": "p", "generated_solution": "\n".join(parts)}, ensure_ascii=False)


def test_openmathreasoning_tool_call_format(tmp_path: Path) -> None:
    from oce_math_mcp.tir_replay import numina_blocks

    rec = json.loads(_omr(("print(6 * 7)", "42"), ("import os", "x")))
    assert numina_blocks(rec) == [("print(6 * 7)", "42"), ("import os", "x")]
    f = tmp_path / "omr.jsonl"
    f.write_text(
        _omr(("print(6 * 7)", "42")) + "\n" + _numina(("print(1)", "1")) + "\n", encoding="utf-8"
    )
    rep = run([f])
    assert rep["equal"] == 2
    assert rep["formats"] == {"omr": 1, "numina": 1}


def test_sample_spans_files_and_reads_only_picked_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    a = tmp_path / "a.jsonl"
    a.write_text(
        "\n".join(_numina((f"print({i})", str(i))) for i in range(30)) + "\n", encoding="utf-8"
    )
    sols = [json.loads(_numina((f"print({i})", str(i))))["solution"] for i in range(30, 70)]
    pq.write_table(pa.table({"solution": sols}), tmp_path / "b.parquet", row_group_size=8)
    rep = run([a, tmp_path / "b.parquet"], sample=12, seed=7)
    assert rep["rows_total"] == 70 and rep["rows"] == 12 and rep["equal"] == 12
    # same sample from the same seed on a single file with the same 70 rows
    c = tmp_path / "c.jsonl"
    c.write_text(
        "\n".join(_numina((f"print({i})", str(i))) for i in range(70)) + "\n", encoding="utf-8"
    )
    assert run([c], sample=12, seed=7)["sample_fingerprint"] == rep["sample_fingerprint"]


def test_previous_block_that_failed_in_the_notebook_keeps_its_variables(tmp_path: Path) -> None:
    """In the notebook a failing block keeps the variables defined before: the previous blocks go in try."""
    jupyter_tb = (
        "-" * 75 + " TypeError                                 Traceback (most recent call last)"
    )
    f = tmp_path / "numina.jsonl"
    f.write_text(
        _numina(("a = 7\nprint(a)", "7"), ("print(a + 'x')", jupyter_tb), ("print(a + 1)", "8"))
        + "\n",
        encoding="utf-8",
    )
    rep = run([f])
    assert rep["calls"] == 3
    assert rep["expected_error"] == 1
    assert rep["equal"] == 2 and rep["with_previous_blocks"] == 1


def test_huge_integers_are_compared_exactly() -> None:
    """OpenMathReasoning: an integer beyond float range stopped the whole replay (OverflowError)."""
    big = 10**400
    assert results_equal(big, str(big))
    assert not results_equal(big, str(big + 1))
    assert results_equal(str(big) + "\n7", f"{big}\n7")
    from oce_math_mcp.tir_replay import compare_outputs

    assert compare_outputs(f"{big}\n7", f"{big + 1}\n7") is None


def test_a_failing_comparison_counts_as_different(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import oce_math_mcp.tir_replay as pt

    def boom(*_a: object, **_k: object) -> None:
        raise OverflowError("probe")

    monkeypatch.setattr(pt, "compare_outputs", boom)
    f = tmp_path / "numina.jsonl"
    f.write_text(
        _numina(("print(1)", "1")) + "\n" + _numina(("print(2)", "2")) + "\n", encoding="utf-8"
    )
    rep = pt.run([f])
    assert rep["calls"] == 2 and rep["different"] == 2
    assert rep["failed_examples"][0]["comparison_failed"] == "OverflowError: probe"
