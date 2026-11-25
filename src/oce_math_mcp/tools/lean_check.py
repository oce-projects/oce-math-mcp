"""lean_check: verification of Lean 4 + Mathlib proofs (D16 timeout 60 s, D17 Lean in the repo)."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .. import limits
from ..envelope import Ctx
from ..errors import ToolError

REPO = Path(__file__).resolve().parents[3]
_MSG = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+): (?P<kind>error|warning|info): (?P<text>.*)$"
)

# ---------------------------------------------------------------- D36: Lean code filter
# Lean runs code while it compiles: `#eval` of an IO action, commands that define elaborators
# (`elab`, `macro`, `syntax`), attributes that register a user function as an elaborator
# (`@[command_elab]`, `@[tactic]`, `@[init]`). `axiom`, `native_decide` and the like would make
# a false proof "valid". The filter works on the raw text, comments included: a false
# positive costs a rewording, a false negative the execution of a program.

_HASH_ALLOWED = frozenset(
    {"check", "check_failure", "print", "reduce", "synth", "simp", "norm_num", "guard_msgs"}
)
_FORBIDDEN_WORDS = (
    "run_cmd",
    "run_tac",
    "run_elab",
    "run_meta",
    "elab",
    "elab_rules",
    "macro",
    "macro_rules",
    "syntax",
    "initialize",
    "builtin_initialize",
    "unsafe",
    "implemented_by",
    "extern",
    "axiom",
    "native_decide",
    "simproc",
    "dsimproc",
    "builtin_simproc",
    "polyrith",
    "bv_decide",
    "bv_check",
)
# Constant names that are also written with the prefix (`Lean.ofReduceBool`).
_FORBIDDEN_NAMES = ("ofReduceBool", "ofReduceNat", "trustCompiler", "sorryAx")
_WORD = re.compile(
    r"(?<![\w.'«])(" + "|".join(sorted(_FORBIDDEN_WORDS, key=len, reverse=True)) + r")(?![\w'!?»])"
)
_NAME = re.compile(r"(?<![\w'«])(" + "|".join(_FORBIDDEN_NAMES) + r")(?![\w'!?»])")
_PREFIXED = re.compile(r"(?<![\w.'«])((?:register|declare)_\w+)")
_NATIVE = re.compile(r"\+\s*native\b|\bnative\s*:=")
_HASH = re.compile(r"(?<![\w.'«])#([A-Za-z_][\w!?]*)")
_SET_OPTION = re.compile(r"\bset_option\s+([\w.]+)")
_OPTION_PREFIXES = (
    "maxHeartbeats",
    "maxRecDepth",
    "maxSynthPendingDepth",
    "synthInstance.",
    "pp.",
    "linter.",
    "trace.",
    "autoImplicit",
    "relaxedAutoImplicit",
    "push_neg.",
    "exponentiation.",
)
_IMPORT = re.compile(r"^\s*import\s+(.+)$", flags=re.MULTILINE)
_MODULE_OK = re.compile(r"^Mathlib(\.[A-Za-z_][\w]*)*$")
_ATTR_OPEN = re.compile(r"@\[|\battribute\s*\[")
_ATTR_ALLOWED = frozenset(
    {
        "simp",
        "ext",
        "reducible",
        "irreducible",
        "semireducible",
        "instance",
        "inline",
        "noinline",
        "specialize",
        "refl",
        "symm",
        "trans",
        "congr",
        "gcongr",
        "mono",
        "nolint",
        "deprecated",
        "match_pattern",
        "elab_as_elim",
        "elab_without_expected_type",
        "norm_cast",
        "push_cast",
        "coe",
        "simps",
        "to_additive",
        "fun_prop",
        "continuity",
        "measurability",
        "field_simps",
        "induction_eliminator",
        "cases_eliminator",
        "inherit_doc",
        "pp_nodot",
        "class",
    }
)
_ATTR_NAME = re.compile(r"[A-Za-z_][\w.]*")
_EXAMPLE = "theorem t : 2 + 2 = 4 := by norm_num"


def _forbid(what: str) -> ToolError:
    return ToolError(
        "FORBIDDEN",
        f"Construct forbidden in lean_check: {what}. Theorems, definitions and tactics are allowed; "
        "no commands that run code, no new axioms, no imports outside Mathlib.",
        _EXAMPLE,
    )


def _attribute_items(code: str, start: int) -> tuple[list[str], int]:
    """Items of an attribute list '[a, b c, local d]' that starts at code[start] == '['."""
    depth, i, cur, items = 0, start, [], []
    while i < len(code):
        ch = code[i]
        if ch == "[":
            depth += 1
            if depth > 1:
                cur.append(ch)
        elif ch == "]":
            depth -= 1
            if depth == 0:
                items.append("".join(cur))
                return items, i
            cur.append(ch)
        elif ch == "," and depth == 1:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    raise _forbid("attribute list not closed")


def check_source(code: str) -> None:
    """Rejects Lean code that could run programs or add axioms (D36)."""
    m = _WORD.search(code) or _NAME.search(code)
    if m:
        raise _forbid(m.group(1))
    m = _PREFIXED.search(code)
    if m:
        raise _forbid(m.group(1))
    if _NATIVE.search(code):
        raise _forbid("native decide (native := true / +native)")
    for m in _HASH.finditer(code):
        if m.group(1) not in _HASH_ALLOWED:
            raise _forbid("#" + m.group(1))
    for m in _SET_OPTION.finditer(code):
        if not m.group(1).startswith(_OPTION_PREFIXES):
            raise _forbid(f"set_option {m.group(1)}")
    for m in _IMPORT.finditer(code):
        for module in m.group(1).split():
            if not _MODULE_OK.match(module):
                raise _forbid(f"import {module}")
    for m in _ATTR_OPEN.finditer(code):
        items, _end = _attribute_items(code, m.end() - 1)
        for item in items:
            words = item.strip().lstrip("-").split()
            while words and words[0] in ("local", "scoped"):
                words = words[1:]
            name = _ATTR_NAME.match(words[0]) if words else None
            if name is None or name.group(0) not in _ATTR_ALLOWED:
                raise _forbid(f"attribute [{item.strip()}]")


def project_dir() -> Path:
    return Path(os.environ.get("OCE_MATH_LEAN_PROJECT", str(REPO / "lean")))


def elan_home() -> Path:
    return Path(os.environ.get("OCE_MATH_ELAN_HOME", str(REPO / ".lean-tools" / "elan")))


def lake_exe() -> Path:
    name = "lake.exe" if os.name == "nt" else "lake"
    return elan_home() / "bin" / name


def mathlib_olean() -> Path:
    """Compiled Mathlib file: it exists only after a successful `lake exe cache get`."""
    return (
        project_dir()
        / ".lake"
        / "packages"
        / "mathlib"
        / ".lake"
        / "build"
        / "lib"
        / "lean"
        / "Mathlib.olean"
    )


def lean_available() -> bool:
    """Lean is ready only if Mathlib is compiled: a half-cloned copy would make Lean run for nothing."""
    proj = project_dir()
    return (
        lake_exe().is_file() and (proj / "lean-toolchain").is_file() and mathlib_olean().is_file()
    )


STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})


def _axiom_trailer(marker: str) -> str:
    """Our command, after the user code: lists the axioms used by the declarations (D36).

    `_root_.Lean.collectAxioms` by full name: a user `collectAxioms` does not replace it.
    `example` blocks do not enter the environment: for them the `check_source` filter applies.
    """
    return (
        "\n\nrun_cmd do\n"
        "  let mut all : Array Lean.Name := #[]\n"
        "  for (n, _) in (← Lean.getEnv).constants.map₂.toList do\n"
        "    for a in (← _root_.Lean.collectAxioms n) do\n"
        "      unless all.contains a do\n"
        "        all := all.push a\n"
        f'  Lean.logInfo m!"{marker} {{all.toList}}"\n'
    )


def _run_tree(
    cmd: list[str], cwd: Path, env: dict[str, str], timeout: float
) -> tuple[int, str, str]:
    """Runs `lake env lean` and, at the timeout or at the end, closes the whole process tree.

    `lake` starts `lean` as a child: killing only `lake` leaves `lean` orphaned (seen on 2026-11-24
    with a stopped test). Windows: `taskkill /T` while the parent is alive (it follows the children by PID), then
    the Job Object with KILL_ON_JOB_CLOSE as a safety net. The Job Object alone is not enough: tested
    on 2026-11-24, a grandchild created by the venv launcher left the job and stayed alive.
    Linux: process group.
    """
    from ..pool import WinJob

    win = sys.platform == "win32"
    proc = subprocess.Popen(  # noqa: S603 - fixed executable and arguments, code in a file
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=0x08000000 if win else 0,  # CREATE_NO_WINDOW
        start_new_session=not win,
    )
    job: WinJob | None = None
    if win:
        try:
            job = WinJob(0)
            job.assign(proc.pid)
        except OSError:
            job = None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        _kill_tree(proc, job)
        proc.communicate()
        raise
    finally:
        if job is not None:
            job.close()


def _kill_tree(proc: subprocess.Popen[str], job: Any) -> None:
    if sys.platform == "win32":
        subprocess.run(  # noqa: S603 - fixed system command
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],  # noqa: S607
            capture_output=True,
            check=False,
        )
        if job is not None:
            job.close()
    else:
        import signal

        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    proc.kill()


def handle(args: dict[str, Any], ctx: Ctx) -> dict[str, Any]:
    code: str = args["code"]
    if len(code) > limits.LEAN_CODE_MAX:
        raise ToolError(
            "LIMIT_EXCEEDED",
            f"Lean code of {len(code)} characters: maximum {limits.LEAN_CODE_MAX}.",
        )
    check_source(code)
    if not lean_available():
        raise ToolError(
            "UNAVAILABLE",
            "Lean 4 + Mathlib are not installed: oce-math-mcp --install-lean "
            "(folders from OCE_MATH_ELAN_HOME and OCE_MATH_LEAN_PROJECT, inside the repo by default).",
        )
    if not re.search(r"^\s*import\s", code, flags=re.MULTILINE):
        code = "import Mathlib\n" + code
    proj = project_dir()
    tmp_dir = proj / ".oce_tmp"
    tmp_dir.mkdir(exist_ok=True)
    src = tmp_dir / f"Check_{uuid.uuid4().hex}.lean"
    marker = f"OCE_AXIOMS_{uuid.uuid4().hex[:16]}"  # random: the user code does not know it
    src.write_text(code + _axiom_trailer(marker), encoding="utf-8")
    env = dict(os.environ)
    env["ELAN_HOME"] = str(elan_home())
    env["PATH"] = str(elan_home() / "bin") + os.pathsep + env.get("PATH", "")
    t0 = time.perf_counter()
    try:
        returncode, stdout, stderr = _run_tree(
            [str(lake_exe()), "env", "lean", str(src)], proj, env, limits.LEAN_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        raise ToolError("TIMEOUT", f"Lean beyond {limits.LEAN_TIMEOUT_S:.0f} s.") from None
    finally:
        src.unlink(missing_ok=True)
    elapsed = time.perf_counter() - t0
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    messages = []
    axioms: list[str] | None = None
    for raw in (stdout + "\n" + stderr).splitlines():
        if marker in raw:
            listed = raw.split(marker, 1)[1].strip().strip("[]")
            axioms = sorted({a.strip() for a in listed.split(",") if a.strip()})
            continue
        m = _MSG.match(raw.strip())
        if not m:
            if raw.strip():
                messages.append(raw.strip())
            continue
        item = {
            "line": int(m["line"]) - (0 if code is args["code"] else 1),
            "col": int(m["col"]),
            "text": m["text"],
        }
        (errors if m["kind"] == "error" else warnings).append(item)
        messages.append(f"{m['kind']}: {m['text']}")
    has_sorry = (
        any("sorry" in w["text"] for w in warnings)
        or bool(re.search(r"\bsorry\b", args["code"]))
        or "sorryAx" in (axioms or [])
    )
    if axioms is None:
        nonstandard: list[str] = []
        messages.append(
            "Axiom check not run: the code does not close (comment, block or proof left open)."
        )
    else:
        nonstandard = [a for a in axioms if a not in STANDARD_AXIOMS and a != "sorryAx"]
        if nonstandard:
            messages.append(f"Axioms not allowed: {', '.join(nonstandard)}.")
    valid = (
        returncode == 0
        and not errors
        and axioms is not None
        and not nonstandard
        and (args.get("allow_sorry", False) or not has_sorry)
    )
    ctx.formatted = "Proof valid." if valid else "Proof NOT valid."
    ctx.step("Code passed to Lean 4 with Mathlib (lake env lean).")
    ctx.step(
        f"Errors: {len(errors)}; warnings: {len(warnings)}; sorry: {'yes' if has_sorry else 'no'}."
    )
    ctx.step(f"Axioms used: {', '.join(axioms) if axioms else 'none or not checked'}.")
    ctx.step(f"Check: {'passed' if valid else 'failed'}.")
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "has_sorry": has_sorry,
        "axioms": axioms,
        "messages": messages[:200],
        "elapsed_s": round(elapsed, 3),
    }
