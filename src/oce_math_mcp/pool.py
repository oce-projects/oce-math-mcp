"""Separate computing processes (D7): timeout that kills, memory limit, automatic restart.

A thread cannot be stopped: that is why every computation runs in a child process
(`python -m oce_math_mcp.worker`). When the time runs out the child is killed and another one
starts in the background.

The children have their own stdin and stdout (dedicated pipes, D32): with `multiprocessing`
on Windows the first child, started while the stdio server was reading its stdin, stayed stuck
and was not ready within 120 s (measured: first stdio call after 132.6 s).

Memory: on Windows a Job Object with JOB_OBJECT_LIMIT_PROCESS_MEMORY; on Linux RLIMIT_AS.
The Job Object also has KILL_ON_JOB_CLOSE: if the server dies, the children die too.
After every python_sandbox the process is replaced (D28).
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any

from . import limits
from .envelope import build_error
from .worker import read_frame, write_frame

log = logging.getLogger("oce_math_mcp.pool")

_RECYCLE_AFTER = frozenset({"python_sandbox"})
_IN_PARENT = frozenset({"lean_check"})
_EOF = object()


# ------------------------------------------------------------------ Job Object (Windows)


class WinJob:
    """Job Object with a per-process memory limit; the children close with the parent."""

    def __init__(self, memory_mb: int) -> None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimit),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k32 = k32
        self.handle = k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
        info = ExtendedLimit()
        flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_mb > 0:
            flags |= 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = memory_mb * 1024 * 1024
        info.BasicLimitInformation.LimitFlags = flags
        ok = k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject")

    def close(self) -> None:
        """Closes the Job Object: with KILL_ON_JOB_CLOSE every assigned process and its children die."""
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None

    def assign(self, pid: int) -> None:
        import ctypes

        h = self._k32.OpenProcess(
            0x0100 | 0x0001, False, pid
        )  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        if not h:
            raise OSError(ctypes.get_last_error(), "OpenProcess")
        try:
            if not self._k32.AssignProcessToJobObject(self.handle, h):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
        finally:
            self._k32.CloseHandle(h)


# ------------------------------------------------------------------ pool


class _Worker:
    def __init__(self, memory_mb: int, job: WinJob | None, start_timeout: float = 120) -> None:
        t0 = time.perf_counter()
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        self.proc = subprocess.Popen(  # noqa: S603 - fixed interpreter and module
            [sys.executable, "-m", "oce_math_mcp.worker", str(memory_mb)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
        if job is not None:
            try:
                job.assign(self.proc.pid)
            except OSError as exc:  # pragma: no cover - depends on the system
                log.warning("Job Object not assigned: %s", exc)
        self.replies: queue.Queue[Any] = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            first = self.replies.get(timeout=start_timeout)
        except queue.Empty:
            self.kill()
            raise RuntimeError(f"computing process not ready within {start_timeout:g} s") from None
        if not (isinstance(first, dict) and "ready" in first):
            self.kill()
            raise RuntimeError("computing process: start failed")
        self.startup_s = time.perf_counter() - t0

    def _read(self) -> None:
        stream = self.proc.stdout
        assert stream is not None
        try:
            while True:
                msg = read_frame(stream)
                if msg is None:
                    break
                self.replies.put(msg)
        except (OSError, ValueError):
            pass
        self.replies.put(_EOF)

    def send(self, tool: str, args: Any) -> None:
        stream = self.proc.stdin
        assert stream is not None
        write_frame(stream, {"tool": tool, "args": args})

    def kill(self) -> None:
        with contextlib.suppress(OSError):
            self.proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            self.proc.wait(5)
        for stream in (self.proc.stdin, self.proc.stdout):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()


class ComputePool:
    """Pool of computing processes. call() is safe from several threads at once."""

    def __init__(
        self,
        size: int | None = None,
        timeout_s: float = limits.TIMEOUT_S,
        memory_mb: int = limits.MEMORY_MB,
    ) -> None:
        env_size = os.environ.get("OCE_MATH_POOL_SIZE")
        self.size = size or (
            int(env_size) if env_size else max(1, min(4, (os.cpu_count() or 2) - 1))
        )
        # D96: 0 or a negative size started no process, every call waited and failed
        if self.size < 1:
            raise ValueError(
                f"The pool needs at least 1 computing process, not {self.size} (OCE_MATH_POOL_SIZE)."
            )
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self._job: WinJob | None = None
        if sys.platform == "win32":
            try:
                self._job = WinJob(memory_mb)
            except OSError as exc:  # pragma: no cover
                log.warning("Job Object not available: %s (no memory limit)", exc)
        self._idle: queue.Queue[_Worker] = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()
        self._all: set[_Worker] = set()
        self.startup_times: list[float] = []
        for _ in range(self.size):
            threading.Thread(target=self._spawn, daemon=True).start()

    def _spawn(self) -> None:
        for attempt in range(3):
            if self._closed:
                return
            try:
                w = _Worker(self.memory_mb, self._job)
            except (RuntimeError, OSError) as exc:
                log.error("computing process start failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(1)
                continue
            with self._lock:
                if self._closed:
                    w.kill()
                    return
                self._all.add(w)
                self.startup_times.append(w.startup_s)
            self._idle.put(w)
            return

    def _replace(self, w: _Worker) -> None:
        with self._lock:
            self._all.discard(w)
        w.kill()
        threading.Thread(target=self._spawn, daemon=True).start()

    def call(self, tool: str, args: dict[str, Any] | None) -> dict[str, Any]:
        """Runs a tool in a computing process and returns the 9-field envelope."""
        op = (
            args.get("operation")
            if isinstance(args, dict) and isinstance(args.get("operation"), str)
            else None
        )
        if tool in _IN_PARENT:
            from .dispatch import run_tool

            return run_tool(tool, args)
        try:
            w = self._idle.get(timeout=180)
        except queue.Empty:
            return build_error(tool, op, "UNAVAILABLE", "No free computing process within 180 s.")
        t0 = time.perf_counter()
        try:
            w.send(tool, args)
        except (TypeError, ValueError):
            self._idle.put(w)
            return build_error(tool, op, "INVALID_ARGUMENT", "Arguments not serialisable to JSON.")
        except OSError:
            self._replace(w)
            return build_error(tool, op, "INTERNAL_ERROR", "Computing process unreachable: retry.")
        try:
            resp = w.replies.get(timeout=self.timeout_s)
        except queue.Empty:
            self._replace(w)
            return build_error(
                tool,
                op,
                "TIMEOUT",
                f"Computation beyond {self.timeout_s:g} s: process stopped. Simplify the problem or use math_numeric.",
            )
        if resp is _EOF:
            elapsed = time.perf_counter() - t0
            self._replace(w)
            return build_error(
                tool,
                op,
                "LIMIT_EXCEEDED",
                f"Computing process ended after {elapsed:.2f} s (memory beyond {self.memory_mb} MB or crash).",
            )
        if tool in _RECYCLE_AFTER:
            self._replace(w)
        else:
            self._idle.put(w)
        return resp

    def close(self) -> None:
        with self._lock:
            self._closed = True
            workers = list(self._all)
            self._all.clear()
        for w in workers:
            stream = w.proc.stdin
            if stream is not None:
                with contextlib.suppress(OSError, ValueError):
                    stream.write(bytes(4))
                    stream.flush()
            w.kill()
