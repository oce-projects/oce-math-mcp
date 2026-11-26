"""Installation of Lean 4 and Mathlib inside the repo (decision D17): oce-math-mcp --install-lean.

- elan (the Lean version manager) in .lean-tools/elan, without touching the system PATH;
- Lean project in lean/ with versioned lean-toolchain and lakefile.toml (Mathlib pinned);
- dependencies and already compiled .olean files with `lake exe cache get` in lean/.lake (excluded from git).

Sources: official GitHub releases of leanprover/elan and the Mathlib cache.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .tools import lean_check

ELAN_VERSION = "v4.4.0"
MATHLIB_REV = "v4.36.0"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.36.0"
# D103: the Lean REPL (Apache-2.0) at the tag of the same Lean version: an .olean of another version does not load.
REPL_URL = "https://github.com/leanprover-community/repl.git"
REPL_TAG = LEAN_TOOLCHAIN.split(":", 1)[1]

LAKEFILE = f"""name = "oce_lean"
version = "0.1.0"
defaultTargets = []

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "{MATHLIB_REV}"
"""


def _asset() -> str:
    if sys.platform == "win32":
        return "elan-x86_64-pc-windows-msvc.zip"
    if sys.platform == "darwin":
        return "elan-x86_64-apple-darwin.tar.gz"
    return "elan-x86_64-unknown-linux-gnu.tar.gz"


def lean_env() -> dict[str, str]:
    env = dict(os.environ)
    home = lean_check.elan_home()
    env["ELAN_HOME"] = str(home)
    env["PATH"] = str(home / "bin") + os.pathsep + env.get("PATH", "")
    # Long Mathlib paths on Windows: only for these git commands, no global change.
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "core.longpaths"
    env["GIT_CONFIG_VALUE_0"] = "true"
    return env


def _run(cmd: list[str], cwd: Path, timeout: float, log: Any) -> float:
    t0 = time.perf_counter()
    log(f"$ {' '.join(cmd)}  (in {cwd})")
    proc = subprocess.run(  # noqa: S603 - fixed commands
        cmd,
        cwd=cwd,
        env=lean_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
    for line in tail:
        log(f"  {line}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail)
        )
    return time.perf_counter() - t0


def _size_mb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                total += (Path(root) / f).stat().st_size
    return round(total / 1024 / 1024, 1)


def install_elan(log: Any) -> float:
    home = lean_check.elan_home()
    if lean_check.lake_exe().is_file():
        log(f"elan already present in {home}")
        return 0.0
    t0 = time.perf_counter()
    url = f"https://github.com/leanprover/elan/releases/download/{ELAN_VERSION}/{_asset()}"
    tmp = Path(tempfile.mkdtemp(prefix="oce-elan-"))
    try:
        archive = tmp / _asset()
        log(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, archive.open("wb") as out:
            shutil.copyfileobj(resp, out)
        log(f"  {archive.stat().st_size} byte")
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmp / "x")  # noqa: S202 - zipfile strips ".." and absolute paths
        else:
            with tarfile.open(archive) as t:
                t.extractall(tmp / "x", filter="data")
        init = next(p for p in (tmp / "x").rglob("elan-init*") if p.is_file())
        home.mkdir(parents=True, exist_ok=True)
        _run([str(init), "-y", "--no-modify-path", "--default-toolchain", "none"], tmp, 600, log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return time.perf_counter() - t0


def toolchain_dir() -> Path:
    """Folder where elan puts the toolchain: 'leanprover/lean4:v4.36.0' -> 'leanprover--lean4---v4.36.0'."""
    name = LEAN_TOOLCHAIN.replace("/", "--").replace(":", "---")
    return lean_check.elan_home() / "toolchains" / name


def finish_rename(final: Path, log: Any, tries: int = 15, wait_s: float = 2.0) -> bool:
    """Completes the rename '<name>.tmp' -> '<name>' that elan could not do (D35).

    On a network drive the elan rename failed 2 times out of 2 with "Access denied" (os error 5):
    a file just written was still open. With the process closed the same rename succeeded in 81 ms.
    """
    tmp = final.with_suffix(".tmp")
    for k in range(tries):
        if final.is_dir():
            return True
        if not tmp.is_dir():
            return False
        try:
            tmp.rename(final)
            log(f"rename completed at attempt {k + 1}: {final.name}")
            return True
        except OSError as exc:
            log(f"rename failed ({exc}); new attempt in {wait_s} s")
            time.sleep(wait_s)
    return final.is_dir()


def install_toolchain(log: Any) -> float:
    """Installs the pinned toolchain before `lake update`, so a failed rename can be resumed."""
    final = toolchain_dir()
    if final.is_dir():
        log(f"toolchain already present in {final}")
        return 0.0
    t0 = time.perf_counter()
    elan = lean_check.elan_home() / "bin" / ("elan.exe" if os.name == "nt" else "elan")
    try:
        _run([str(elan), "toolchain", "install", LEAN_TOOLCHAIN], lean_check.elan_home(), 3600, log)
    except RuntimeError as exc:
        # Only the final rename is resumed: a .tmp left by an interrupted download is incomplete.
        if "could not rename temp toolchain" not in str(exc) or not finish_rename(final, log):
            raise
    return time.perf_counter() - t0


def write_project(log: Any) -> None:
    proj = lean_check.project_dir()
    proj.mkdir(parents=True, exist_ok=True)
    tc = proj / "lean-toolchain"
    if not tc.is_file() or tc.read_text(encoding="utf-8").strip() != LEAN_TOOLCHAIN:
        tc.write_text(LEAN_TOOLCHAIN + "\n", encoding="utf-8")
        log(f"written {tc}")
    lf = proj / "lakefile.toml"
    if not lf.is_file() or lf.read_text(encoding="utf-8") != LAKEFILE:
        lf.write_text(LAKEFILE, encoding="utf-8")
        log(f"written {lf}")


def on_network(path: Path) -> bool:
    """True if the path is on a network drive (UNC `\\\\server\\...`, also behind a drive letter)."""
    try:
        full = str(path.resolve()) if path.exists() else str(path)
    except OSError:
        full = str(path)
    return full.startswith(("\\\\", "//"))


NETWORK_WARNING = (
    "WARNING: {what} is on a network drive. There the installation failed 2 times out of 2 "
    "(toolchain rename denied, git index.lock). Proposal D37: set OCE_MATH_ELAN_HOME and "
    "OCE_MATH_LEAN_PROJECT to a local drive."
)


def install_repl(log: Any) -> float:
    """Clones the Lean REPL at REPL_TAG and builds it with lake (D103; measured 27 s on 2026-11-26)."""
    if lean_check.repl_exe().is_file():
        log(f"Lean REPL already built in {lean_check.repl_exe()}")
        return 0.0
    t0 = time.perf_counter()
    target = lean_check.repl_dir()
    if not target.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        git = shutil.which("git") or "git"
        cmd = [git, "clone", "--depth", "1", "--branch", REPL_TAG, REPL_URL, str(target)]
        _run(cmd, target.parent, 600, log)
    toolchain = target / "lean-toolchain"
    found = toolchain.read_text(encoding="utf-8").strip() if toolchain.is_file() else "none"
    if found != LEAN_TOOLCHAIN:
        raise RuntimeError(
            f"the Lean REPL in {target} is for {found}, Mathlib needs {LEAN_TOOLCHAIN}: "
            "remove that folder and run --install-lean again"
        )
    _run([str(lean_check.lake_exe()), "build", "repl"], target, 3600, log)
    return time.perf_counter() - t0


def install(log: Any = print) -> dict[str, Any]:
    t_all = time.perf_counter()
    report: dict[str, Any] = {
        "elan": ELAN_VERSION,
        "toolchain": LEAN_TOOLCHAIN,
        "mathlib": MATHLIB_REV,
    }
    for what, where in (
        ("elan", lean_check.elan_home()),
        ("the Lean project", lean_check.project_dir()),
    ):
        if on_network(where):
            log(NETWORK_WARNING.format(what=what))
            report.setdefault("warnings", []).append(NETWORK_WARNING.format(what=what))
    report["elan_s"] = round(install_elan(log), 1)
    report["toolchain_s"] = round(install_toolchain(log), 1)
    write_project(log)
    proj = lean_check.project_dir()
    lake = str(lean_check.lake_exe())
    report["lake_update_s"] = round(_run([lake, "update"], proj, 7200, log), 1)
    report["cache_get_s"] = round(_run([lake, "exe", "cache", "get"], proj, 7200, log), 1)
    report["repl"] = REPL_TAG
    report["repl_s"] = round(install_repl(log), 1)
    report["total_s"] = round(time.perf_counter() - t_all, 1)
    report["elan_size_mb"] = _size_mb(lean_check.elan_home())
    report["lake_size_mb"] = _size_mb(proj / ".lake")
    report["lean_available"] = lean_check.lean_available()
    return report
