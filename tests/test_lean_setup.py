"""Lean installation: toolchain rename on a network drive (D35).

elan extracts the toolchain into `<name>.tmp` and then renames it. On a network drive the rename
failed 2 times out of 2 with "Access denied" (os error 5); with the process closed it succeeded in 81 ms.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oce_math_mcp import lean_setup

FINAL = "leanprover--lean4---v4.36.0"
TMP = "leanprover--lean4---v4.36.tmp"


def _quiet(_msg: str) -> None:
    return None


def test_toolchain_dir_name_matches_elan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCE_MATH_ELAN_HOME", str(tmp_path))
    assert lean_setup.toolchain_dir() == tmp_path / "toolchains" / FINAL
    assert lean_setup.toolchain_dir().with_suffix(".tmp").name == TMP


def test_finish_rename_moves_tmp_dir(tmp_path: Path) -> None:
    tmp = tmp_path / TMP
    (tmp / "bin").mkdir(parents=True)
    (tmp / "bin" / "lean.exe").write_bytes(b"x")
    assert lean_setup.finish_rename(tmp_path / FINAL, _quiet, tries=3, wait_s=0)
    assert (tmp_path / FINAL / "bin" / "lean.exe").is_file()
    assert not tmp.exists()


def test_finish_rename_retries_when_access_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / TMP).mkdir()
    calls = {"n": 0}
    real_rename = Path.rename

    def flaky(self: Path, target: Path) -> Path:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access denied")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    assert lean_setup.finish_rename(tmp_path / FINAL, _quiet, tries=5, wait_s=0)
    assert calls["n"] == 3
    assert (tmp_path / FINAL).is_dir()


def test_finish_rename_gives_up_after_tries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / TMP).mkdir()

    def always_denied(self: Path, target: Path) -> Path:
        raise PermissionError(5, "Access denied")

    monkeypatch.setattr(Path, "rename", always_denied)
    assert not lean_setup.finish_rename(tmp_path / FINAL, _quiet, tries=3, wait_s=0)
    assert (tmp_path / TMP).is_dir()


def test_finish_rename_without_tmp_returns_false(tmp_path: Path) -> None:
    assert not lean_setup.finish_rename(tmp_path / FINAL, _quiet, tries=2, wait_s=0)


def test_install_toolchain_skips_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OCE_MATH_ELAN_HOME", str(tmp_path))
    lean_setup.toolchain_dir().mkdir(parents=True)

    def must_not_run(*_a: object, **_k: object) -> float:
        raise AssertionError("elan must not start if the toolchain is already there")

    monkeypatch.setattr(lean_setup, "_run", must_not_run)
    assert lean_setup.install_toolchain(_quiet) == 0.0


def test_install_toolchain_finishes_rename_after_elan_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OCE_MATH_ELAN_HOME", str(tmp_path))
    final = lean_setup.toolchain_dir()

    def elan_rename_fails(cmd: list[str], cwd: Path, timeout: float, log: object) -> float:
        final.with_suffix(".tmp").mkdir(parents=True)
        raise RuntimeError(
            "command failed (1): elan toolchain install\n"
            "error: could not rename temp toolchain directory directory from 'a' to 'b'"
        )

    monkeypatch.setattr(lean_setup, "_run", elan_rename_fails)
    lean_setup.install_toolchain(_quiet)
    assert final.is_dir()


def test_install_toolchain_never_renames_after_other_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted download leaves an incomplete .tmp: it must not be promoted to toolchain."""
    monkeypatch.setenv("OCE_MATH_ELAN_HOME", str(tmp_path))
    final = lean_setup.toolchain_dir()

    def elan_download_fails(cmd: list[str], cwd: Path, timeout: float, log: object) -> float:
        final.with_suffix(".tmp").mkdir(parents=True)
        raise RuntimeError("command failed (1): elan toolchain install\nerror: could not download")

    monkeypatch.setattr(lean_setup, "_run", elan_download_fails)
    with pytest.raises(RuntimeError, match="could not download"):
        lean_setup.install_toolchain(_quiet)
    assert not final.exists()


def test_run_error_carries_command_output(tmp_path: Path) -> None:
    import sys

    # The searched word does not appear in the command: it must come from the output.
    with pytest.raises(RuntimeError, match="real-output"):
        lean_setup._run(
            [sys.executable, "-c", "import sys; print('real' + '-output'); sys.exit(3)"],
            tmp_path,
            60,
            _quiet,
        )


def test_half_installed_mathlib_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mathlib cloned but not compiled (lake update failed): no real Lean, UNAVAILABLE answer."""
    from oce_math_mcp.dispatch import run_tool
    from oce_math_mcp.tools import lean_check

    elan = tmp_path / "elan"
    (elan / "bin").mkdir(parents=True)
    monkeypatch.setenv("OCE_MATH_ELAN_HOME", str(elan))
    lean_check.lake_exe().write_bytes(b"")
    proj = tmp_path / "lean"
    (proj / ".lake" / "packages" / "mathlib").mkdir(parents=True)
    (proj / "lean-toolchain").write_text("leanprover/lean4:v4.36.0\n", encoding="utf-8")
    monkeypatch.setenv("OCE_MATH_LEAN_PROJECT", str(proj))
    assert not lean_check.lean_available()
    resp = run_tool("lean_check", {"code": "theorem t : 2 + 2 = 4 := by norm_num"})
    assert resp["error"]["code"] == "UNAVAILABLE"


def _alive(pid: int) -> bool:
    import os
    import subprocess
    import sys

    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_timeout_kills_the_whole_process_tree(tmp_path: Path) -> None:
    """The parent starts a sleeping child: at the timeout both must die."""
    import os
    import subprocess
    import sys
    import time

    from oce_math_mcp.tools import lean_check

    pidfile = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(60)\n"
    )
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        lean_check._run_tree([sys.executable, "-c", script], tmp_path, dict(os.environ), 3.0)
    # If the grandchild stayed alive, the pipe would stay open until its 60 s of sleep end.
    assert time.monotonic() - t0 < 30
    child = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(child)


def test_network_paths_are_detected() -> None:
    assert lean_setup.on_network(Path("//server/share/lean"))
    assert lean_setup.on_network(Path(r"\\server\share\lean"))
    assert not lean_setup.on_network(Path("C:/Users/x/lean-does-not-exist"))
