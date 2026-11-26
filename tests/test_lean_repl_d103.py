"""D103: lean_check through a persistent Lean REPL (Mathlib loaded once, each check from env 0).

Measured on 2026-11-26: `lake env lean` took 15.46 s even for `#check Nat` (loading Mathlib), the REPL 16.16 s for
`import Mathlib` once and then 0.004-0.034 s per check. The logic is tested with a fake REPL in Python
(`tests/fake_lean_repl.py`, same JSON protocol); with Lean installed the REPL is compared with `lake env lean`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from oce_math_mcp import lean_setup, limits
from oce_math_mcp.tools import lean_check, lean_repl
from tests.conftest import call

FAKE = Path(__file__).with_name("fake_lean_repl.py")
LEAN_READY = lean_check.lean_available() and lean_check.repl_exe().is_file()


def make(
    tmp_path: Path, recycle: int = 1000, start_timeout: float = 10.0, **extra: str
) -> lean_repl.LeanRepl:
    env = dict(
        os.environ,
        FAKE_REPL_PIDS=str(tmp_path / "pids.txt"),
        FAKE_REPL_LOG=str(tmp_path / "log.txt"),
    )
    env.update(extra)
    return lean_repl.LeanRepl(
        [sys.executable, str(FAKE)], tmp_path, env, recycle=recycle, start_timeout=start_timeout
    )


def starts(tmp_path: Path) -> int:
    p = tmp_path / "pids.txt"
    return len(p.read_text(encoding="utf-8").split()) if p.exists() else 0


# ------------------------------------------------------------------ the REPL manager, fake REPL


def test_one_process_for_many_checks(tmp_path: Path) -> None:
    repl = make(tmp_path)
    try:
        envs = [repl.check("theorem a : True := trivial", 10)[0]["env"] for _ in range(3)]
    finally:
        repl.close()
    assert envs == [1, 2, 3]
    assert starts(tmp_path) == 1


def test_every_check_starts_from_env_0(tmp_path: Path) -> None:
    repl = make(tmp_path)
    try:
        for _ in range(3):
            repl.check("theorem a : True := trivial", 10)
    finally:
        repl.close()
    sent = (tmp_path / "log.txt").read_text(encoding="utf-8").split()
    assert sent == ["env=None", "env=0", "env=0", "env=0"]


def test_timeout_kills_the_repl_and_the_next_check_restarts_it(tmp_path: Path) -> None:
    repl = make(tmp_path)
    try:
        repl.check("theorem a : True := trivial", 10)
        old = repl._proc
        assert old is not None
        t0 = time.perf_counter()
        with pytest.raises(subprocess.TimeoutExpired):
            repl.check("SLEEP", 1.0)
        assert time.perf_counter() - t0 < 10
        old.wait(timeout=10)
        assert old.poll() is not None
        assert repl.check("theorem a : True := trivial", 10)[0]["env"] == 1
    finally:
        repl.close()
    assert starts(tmp_path) == 2


def test_a_crash_is_a_repl_error_and_the_repl_restarts(tmp_path: Path) -> None:
    repl = make(tmp_path)
    try:
        with pytest.raises(lean_repl.ReplError):
            repl.check("CRASH", 10)
        assert repl.check("theorem a : True := trivial", 10)[0]["env"] == 1
    finally:
        repl.close()
    assert starts(tmp_path) == 2


def test_recycle_after_n_checks(tmp_path: Path) -> None:
    repl = make(tmp_path, recycle=2)
    try:
        for _ in range(3):
            repl.check("theorem a : True := trivial", 10)
    finally:
        repl.close()
    assert starts(tmp_path) == 2


def test_slow_mathlib_load_is_a_start_timeout(tmp_path: Path) -> None:
    repl = make(tmp_path, start_timeout=0.5, FAKE_REPL_IMPORT_S="5")
    t0 = time.perf_counter()
    try:
        with pytest.raises(lean_repl.ReplStartTimeout):
            repl.check("theorem a : True := trivial", 10)
    finally:
        repl.close()
    assert time.perf_counter() - t0 < 4


def test_failed_import_is_a_repl_error(tmp_path: Path) -> None:
    repl = make(tmp_path, FAKE_REPL_IMPORT_FAILS="1")
    try:
        with pytest.raises(lean_repl.ReplError, match="Mathlib"):
            repl.check("theorem a : True := trivial", 10)
    finally:
        repl.close()


def test_a_refused_command_is_a_repl_error_and_the_repl_restarts(tmp_path: Path) -> None:
    repl = make(tmp_path)
    try:
        with pytest.raises(lean_repl.ReplError, match="refused"):
            repl.check("REFUSE", 10)
        assert repl.check("theorem a : True := trivial", 10)[0]["env"] == 1
    finally:
        repl.close()
    assert starts(tmp_path) == 2


def test_concurrent_checks_do_not_mix(tmp_path: Path) -> None:
    repl = make(tmp_path)

    def one(i: int) -> str:
        resp, _ = repl.check(f"theorem t{i} : True := trivial -- OCE_AXIOMS_{i:04d}x", 10)
        return resp["messages"][0]["data"]

    try:
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(one, range(16)))
    finally:
        repl.close()
    assert got == [f"OCE_AXIOMS_{i:04d}x [propext]" for i in range(16)]
    assert starts(tmp_path) == 1


# ------------------------------------------------------------------ lean_check with the fake REPL


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[lean_repl.LeanRepl]:
    repl = make(tmp_path)
    monkeypatch.setattr(lean_repl, "shared", lambda: repl)
    monkeypatch.setattr(lean_repl, "enabled", lambda: True)
    monkeypatch.setattr(lean_check, "lean_available", lambda: True)
    yield repl
    repl.close()


def old_path(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []

    def run_tree(cmd: list[str], *_a: Any) -> tuple[int, str, str]:
        calls.append(cmd)
        return 1, "", "lake env lean was called"

    monkeypatch.setattr(lean_check, "_run_tree", run_tree)
    return calls


def test_lean_check_goes_through_the_repl(
    fake: lean_repl.LeanRepl, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = old_path(monkeypatch)
    r = call("lean_check", code="theorem t : 1 = 1 := rfl")
    assert r["ok"], r
    assert r["result"]["valid"] is True
    assert r["result"]["axioms"] == ["propext"]
    assert calls == []


def test_repl_errors_keep_line_and_column(fake: lean_repl.LeanRepl) -> None:
    r = call("lean_check", code="theorem t : 1 = 1 := by\n    BAD")
    assert r["result"]["valid"] is False
    assert r["result"]["errors"] == [{"line": 2, "col": 4, "text": "boom"}]


def test_code_with_its_own_import_uses_lake_env_lean(
    fake: lean_repl.LeanRepl, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = old_path(monkeypatch)
    r = call("lean_check", code="import Mathlib.Tactic\ntheorem t : 1 = 1 := rfl")
    assert r["result"]["valid"] is False
    assert len(calls) == 1 and calls[0][1:3] == ["env", "lean"]
    assert starts(tmp_path) == 0


def test_a_failing_repl_falls_back_to_lake_env_lean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repl = make(tmp_path, FAKE_REPL_IMPORT_FAILS="1")
    monkeypatch.setattr(lean_repl, "shared", lambda: repl)
    monkeypatch.setattr(lean_repl, "enabled", lambda: True)
    monkeypatch.setattr(lean_check, "lean_available", lambda: True)
    calls = old_path(monkeypatch)
    try:
        r = call("lean_check", code="theorem t : 1 = 1 := rfl")
    finally:
        repl.close()
    assert len(calls) == 1
    assert any("REPL" in w for w in r["warnings"]), r["warnings"]


def test_a_slow_proof_is_a_timeout_and_says_why(
    fake: lean_repl.LeanRepl, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits, "LEAN_TIMEOUT_S", 1.0)
    err = call("lean_check", code="theorem t : 1 = 1 := by SLEEP")["error"]
    assert err["code"] == "TIMEOUT", err
    assert "Mathlib" in err["message"] and "again" in err["message"], err


def test_a_sorry_listed_only_in_sorries_is_a_sorry(fake: lean_repl.LeanRepl) -> None:
    res = call("lean_check", code="theorem t : 1 = 1 := by ADMIT")["result"]
    assert res["has_sorry"] is True
    assert res["valid"] is False


def test_a_slow_mathlib_load_is_a_timeout_that_names_mathlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repl = make(tmp_path, start_timeout=0.5, FAKE_REPL_IMPORT_S="5")
    monkeypatch.setattr(lean_repl, "shared", lambda: repl)
    monkeypatch.setattr(lean_repl, "enabled", lambda: True)
    monkeypatch.setattr(lean_check, "lean_available", lambda: True)
    calls = old_path(monkeypatch)
    try:
        err = call("lean_check", code="theorem t : 1 = 1 := rfl")["error"]
    finally:
        repl.close()
    assert err["code"] == "TIMEOUT", err
    assert "Mathlib" in err["message"] and "again" in err["message"], err
    assert calls == []


def test_switch_and_missing_binary_turn_the_repl_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lean_check, "repl_exe", lambda: FAKE)
    monkeypatch.delenv("OCE_MATH_LEAN_REPL", raising=False)
    assert lean_repl.enabled() is True
    monkeypatch.setenv("OCE_MATH_LEAN_REPL", "0")
    assert lean_repl.enabled() is False
    monkeypatch.delenv("OCE_MATH_LEAN_REPL")
    monkeypatch.setattr(lean_check, "repl_exe", lambda: tmp_path / "missing" / "repl.exe")
    assert lean_repl.enabled() is False


def test_repl_dir_follows_elan_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OCE_MATH_LEAN_REPL_DIR", raising=False)
    monkeypatch.setenv("OCE_MATH_ELAN_HOME", str(tmp_path / "tools" / "elan"))
    assert lean_check.repl_dir() == tmp_path / "tools" / "repl"
    monkeypatch.setenv("OCE_MATH_LEAN_REPL_DIR", str(tmp_path / "x"))
    assert lean_check.repl_dir() == tmp_path / "x"


def test_lean_repl_switch_warm_starts_the_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lean_check, "repl_exe", lambda: FAKE)
    for value, warm in (("warm", True), ("1", False), ("0", False)):
        monkeypatch.setenv("OCE_MATH_LEAN_REPL", value)
        assert lean_repl.warm_at_start() is warm


@pytest.mark.parametrize("warm", [True, False])
def test_server_warms_the_repl_only_when_asked_and_always_shuts_it(
    warm: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oce_math_mcp import cli, server

    seen: list[str] = []

    class FakeServer:
        def run(self, *_a: object, **_k: object) -> None:
            seen.append("run")

    monkeypatch.setattr(server, "build_server", lambda _pool: FakeServer())
    monkeypatch.setattr(lean_repl, "warm_at_start", lambda: warm)
    monkeypatch.setattr(lean_check, "lean_available", lambda: True)
    monkeypatch.setattr(lean_repl, "warm_up_in_background", lambda: seen.append("warm"))
    monkeypatch.setattr(lean_repl, "shutdown", lambda: seen.append("shutdown"))
    assert cli.main(["--no-pool"]) == 0
    assert seen == (["warm", "run", "shutdown"] if warm else ["run", "shutdown"])


# ------------------------------------------------------------------ installation of the REPL


def _quiet(_msg: str) -> None:
    return None


def test_install_repl_clones_the_toolchain_tag_and_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "repl"
    monkeypatch.setenv("OCE_MATH_LEAN_REPL_DIR", str(target))
    runs: list[tuple[list[str], Path]] = []

    def fake_run(cmd: list[str], cwd: Path, timeout: float, log: object) -> float:
        runs.append((cmd, cwd))
        if "clone" in cmd:
            target.mkdir()
            (target / "lean-toolchain").write_text(
                lean_setup.LEAN_TOOLCHAIN + "\n", encoding="utf-8"
            )
        return 0.0

    monkeypatch.setattr(lean_setup, "_run", fake_run)
    lean_setup.install_repl(_quiet)
    assert lean_setup.LEAN_TOOLCHAIN.split(":")[1] == lean_setup.REPL_TAG
    clone, build = runs
    assert clone[0][1:] == [
        "clone",
        "--depth",
        "1",
        "--branch",
        lean_setup.REPL_TAG,
        lean_setup.REPL_URL,
        str(target),
    ]
    assert build == ([str(lean_check.lake_exe()), "build", "repl"], target)


def test_install_repl_refuses_a_repl_for_another_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "repl"
    target.mkdir()
    (target / "lean-toolchain").write_text("leanprover/lean4:v4.35.0\n", encoding="utf-8")
    monkeypatch.setenv("OCE_MATH_LEAN_REPL_DIR", str(target))
    monkeypatch.setattr(lean_setup, "_run", lambda *_a: 0.0)
    with pytest.raises(RuntimeError, match=r"v4\.35\.0"):
        lean_setup.install_repl(_quiet)


def test_install_repl_skips_when_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCE_MATH_LEAN_REPL_DIR", str(tmp_path / "repl"))
    lean_check.repl_exe().parent.mkdir(parents=True)
    lean_check.repl_exe().write_bytes(b"x")

    def must_not_run(*_a: object) -> float:
        raise AssertionError("git and lake must not start if the REPL is already built")

    monkeypatch.setattr(lean_setup, "_run", must_not_run)
    assert lean_setup.install_repl(_quiet) == 0.0


# ------------------------------------------------------------------ the real REPL (Lean + Mathlib installed)

CASES = [
    "theorem t : (2 : ℕ) + 2 = 4 := by norm_num",
    "theorem t : (2 : ℕ) + 2 = 5 := by norm_num",
    "theorem t (n : ℕ) : n + 0 = n := by sorry",
    "theorem t (p : Prop) : p ∨ ¬p := Classical.em p",
    "theorem t (𝔸 : Type) (a : 𝔸) : a = a := rfl\n\ntheorem u : (1 : ℕ) = 2 := by\n  simp",
]


def summary(r: dict) -> tuple[Any, ...]:
    res = r["result"]
    return (
        res["valid"],
        res["has_sorry"],
        res["axioms"],
        [(e["line"], e["col"]) for e in res["errors"]],
    )


@pytest.mark.skipif(not LEAN_READY, reason="Lean + Mathlib + REPL not installed")
def test_real_repl_gives_the_same_answers_as_lake_env_lean(monkeypatch: pytest.MonkeyPatch) -> None:
    with_repl = [call("lean_check", code=c) for c in CASES]
    monkeypatch.setenv("OCE_MATH_LEAN_REPL", "0")
    without = [call("lean_check", code=c) for c in CASES]
    assert [summary(r) for r in with_repl] == [summary(r) for r in without]
    assert [r["result"]["valid"] for r in with_repl] == [True, False, False, True, False]
    lines = [e["line"] for e in with_repl[4]["result"]["errors"]]
    assert lines and all(n in (3, 4) for n in lines), lines


@pytest.mark.skipif(not LEAN_READY, reason="Lean + Mathlib + REPL not installed")
def test_real_repl_is_fast_after_the_first_check_and_isolated() -> None:
    assert call("lean_check", code="def oceSegreto : ℕ := 1")["result"]["valid"] is True
    r = call("lean_check", code="theorem t : oceSegreto = 1 := rfl")
    assert r["result"]["valid"] is False
    assert any("oceSegreto" in e["text"] for e in r["result"]["errors"]), r["result"]["errors"]
    assert r["result"]["elapsed_s"] < 2
