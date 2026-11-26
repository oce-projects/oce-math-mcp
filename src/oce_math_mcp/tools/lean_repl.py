"""Persistent Lean REPL for lean_check (D103): Mathlib is loaded once, every check starts from that environment.

Measured on 2026-11-26 on the development machine (Windows, local disk):
- `lake env lean` loads Mathlib at every check: 15.46 s even for `#check Nat`; this is why checks after a
  restart went beyond the 60 s limit (D86);
- the REPL (leanprover-community/repl, tag v4.36.0, Apache-2.0): `import Mathlib` once in 16.16 s, then
  0.004-0.034 s per check;
- 1,000 checks from env 0: memory in use from 3,130 MB to 3,374 MB (about +0.24 MB per check, every environment
  stays in the REPL), so the REPL is restarted every `limits.LEAN_REPL_RECYCLE` checks;
- a definition made in one check is not visible in the next one: each check starts from env 0.

Protocol: one JSON command per block, blocks separated by a blank line; each answer is a JSON object followed by a
blank line. {"cmd": "import Mathlib"} gives env 0, every check sends {"cmd": code, "env": 0}. An answer
{"message": ...} is an error of the REPL itself (unknown environment, bad JSON).

Switch OCE_MATH_LEAN_REPL: 0 = REPL off (lake env lean for every check, as before D103); warm = Mathlib is loaded
when the server starts; any other value or unset = Mathlib is loaded at the first check.
One REPL serves one check at a time (lock): concurrent checks wait instead of loading Mathlib again in parallel.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any

from .. import limits


class ReplError(RuntimeError):
    """The REPL cannot answer: it stopped, refused the command or could not load Mathlib."""


class ReplStartTimeout(ReplError):
    """Mathlib did not load within the start limit."""


def _pump(stream: IO[str], out: queue.Queue[str | None]) -> None:
    for line in stream:
        out.put(line)
    out.put(None)


def _drain(stream: IO[str], tail: collections.deque[str]) -> None:
    for line in stream:
        tail.append(line.rstrip())


class LeanRepl:
    """One REPL process: started on demand, restarted after a timeout, a crash or `recycle` checks."""

    def __init__(
        self,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        *,
        recycle: int,
        start_timeout: float,
    ) -> None:
        self._cmd = list(cmd)
        self._cwd = cwd
        self._env = env
        self._recycle = recycle
        self._start_timeout = start_timeout
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._job: Any = None
        self._out: queue.Queue[str | None] = queue.Queue()
        self._err: collections.deque[str] = collections.deque(maxlen=20)
        self._base = 0
        self._checks = 0

    # ---------------------------------------------------------------- process

    def _start(self) -> None:
        from ..pool import WinJob

        win = sys.platform == "win32"
        proc = subprocess.Popen(  # noqa: S603 - fixed executable and arguments, code goes through stdin as JSON
            self._cmd,
            cwd=self._cwd,
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=0x08000000 if win else 0,  # CREATE_NO_WINDOW
            start_new_session=not win,
        )
        job = None
        if win:
            try:
                job = WinJob(0)
                job.assign(proc.pid)
            except OSError:
                job = None
        # New queues for every process: lines of a killed process never reach the next one.
        out: queue.Queue[str | None] = queue.Queue()
        err: collections.deque[str] = collections.deque(maxlen=20)
        assert proc.stdout is not None and proc.stderr is not None
        threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True).start()
        threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True).start()
        self._proc, self._job, self._out, self._err = proc, job, out, err
        self._checks = 0
        try:
            resp = self._send({"cmd": "import Mathlib"}, self._start_timeout)
        except subprocess.TimeoutExpired:
            self._stop()
            raise ReplStartTimeout(
                f"the Lean REPL did not load Mathlib within {self._start_timeout:.0f} s"
            ) from None
        except ReplError:
            self._stop()
            raise
        errors = [
            str(m.get("data", "")) for m in resp.get("messages", []) if m.get("severity") == "error"
        ]
        if "message" in resp or errors or not isinstance(resp.get("env"), int):
            self._stop()
            why = "; ".join(errors) if errors else str(resp.get("message", resp))
            raise ReplError(f"import Mathlib failed in the Lean REPL: {why[:400]}")
        self._base = int(resp["env"])

    def _stop(self) -> None:
        proc, job = self._proc, self._job
        self._proc, self._job = None, None
        if proc is None:
            return
        if proc.stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                proc.stdin.close()
        if proc.poll() is None:
            from .lean_check import _kill_tree

            _kill_tree(proc, job)
        if job is not None:
            job.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)

    def _tail(self) -> str:
        lines = [ln for ln in self._err if ln.strip()]
        return f"; stderr: {' | '.join(lines[-5:])[:400]}" if lines else ""

    def _send(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ReplError("the Lean REPL is not running")
        try:
            # ensure_ascii=False: the code goes as UTF-8, shorter than \u escapes. ensure_ascii=True (surrogate
            # pairs for 𝔸) gave the same answers on 2026-11-26: the REPL reads both.
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise ReplError(f"the Lean REPL closed its input ({exc}){self._tail()}") from None
        deadline = time.monotonic() + timeout
        lines: list[str] = []
        while True:
            left = deadline - time.monotonic()
            try:
                if left <= 0:
                    raise queue.Empty
                line = self._out.get(timeout=left)
            except queue.Empty:
                raise subprocess.TimeoutExpired(self._cmd, timeout) from None
            if line is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=2)
                raise ReplError(f"the Lean REPL stopped (exit code {proc.poll()}){self._tail()}")
            if line.strip():
                lines.append(line)
                continue
            if not lines:
                continue
            try:
                resp = json.loads("".join(lines))
            except json.JSONDecodeError:
                continue  # a blank line inside an answer: keep reading
            if not isinstance(resp, dict):
                raise ReplError(
                    f"the Lean REPL gave an answer that is not an object: {resp!r:.200}"
                )
            return resp

    # ---------------------------------------------------------------- public

    def check(self, code: str, timeout: float) -> tuple[dict[str, Any], float]:
        """Checks `code` from the Mathlib environment; returns the REPL answer and the seconds of the check.

        Raises subprocess.TimeoutExpired (the REPL is killed), ReplStartTimeout or ReplError.
        """
        with self._lock:
            if self._proc is not None and (
                self._proc.poll() is not None or self._checks >= self._recycle
            ):
                self._stop()
            if self._proc is None:
                self._start()
            t0 = time.perf_counter()
            try:
                resp = self._send({"cmd": code, "env": self._base}, timeout)
            except (subprocess.TimeoutExpired, ReplError):
                self._stop()
                raise
            self._checks += 1
            if "message" in resp:
                self._stop()
                raise ReplError(f"the Lean REPL refused the command: {str(resp['message'])[:400]}")
            return resp, time.perf_counter() - t0

    def warm_up(self) -> None:
        """Starts the REPL and loads Mathlib, if it is not running already."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._stop()
            self._start()

    def close(self) -> None:
        got = self._lock.acquire(timeout=1.0)
        try:
            self._stop()
        finally:
            if got:
                self._lock.release()


# -------------------------------------------------------------------- the REPL of the server

_SHARED: LeanRepl | None = None
_SHARED_LOCK = threading.Lock()


def _switch() -> str:
    return os.environ.get("OCE_MATH_LEAN_REPL", "1").strip().lower()


def enabled() -> bool:
    """REPL in use: not switched off and built (`oce-math-mcp --install-lean` builds it)."""
    from . import lean_check

    return _switch() not in ("0", "off", "false", "no") and lean_check.repl_exe().is_file()


def warm_at_start() -> bool:
    return _switch() == "warm" and enabled()


def shared() -> LeanRepl:
    global _SHARED
    from . import lean_check

    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = LeanRepl(
                [str(lean_check.lake_exe()), "env", str(lean_check.repl_exe())],
                lean_check.project_dir(),
                lean_check.lean_run_env(),
                recycle=limits.LEAN_REPL_RECYCLE,
                start_timeout=limits.LEAN_REPL_START_S,
            )
        return _SHARED


def warm_up_in_background() -> threading.Thread:
    def run() -> None:
        try:
            shared().warm_up()
        except Exception as exc:  # the first check starts it again and reports the error
            sys.stderr.write(f"oce-math-mcp: Lean REPL warm-up failed: {exc}\n")

    thread = threading.Thread(target=run, name="lean-repl-warm-up", daemon=True)
    thread.start()
    return thread


def shutdown() -> None:
    global _SHARED
    with _SHARED_LOCK:
        repl, _SHARED = _SHARED, None
    if repl is not None:
        repl.close()


atexit.register(shutdown)
