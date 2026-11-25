"""Computing process: python -m oce_math_mcp.worker <memory_mb>.

Protocol on the child's stdin/stdout (dedicated pipes, never the MCP ones): frames with a 4-byte
length (big endian) and UTF-8 JSON. First frame from the child: {"ready": pid}. Then, for every frame
{"tool": ..., "args": ...} received, a frame with the 9-field envelope. Empty frame = end.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from typing import IO, Any

_HDR = struct.Struct(">I")
MAX_FRAME = 64 * 1024 * 1024


def read_frame(stream: IO[bytes]) -> Any | None:
    head = stream.read(_HDR.size)
    if len(head) < _HDR.size:
        return None
    (n,) = _HDR.unpack(head)
    if n == 0:
        return None
    if n > MAX_FRAME:
        raise ValueError(f"frame of {n} bytes beyond the limit")
    data = stream.read(n)
    if len(data) < n:
        return None
    return json.loads(data.decode("utf-8"))


def write_frame(stream: IO[bytes], obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    stream.write(_HDR.pack(len(data)) + data)
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    memory_mb = int(args[0]) if args else 0
    for var in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    # The channel to the parent is a copy of stdout; the real stdout goes to devnull, so the prints of
    # libraries (SymPy, NumPy) do not pollute the protocol.
    out = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - open for the whole life of the process
    inp = sys.stdin.buffer
    if sys.platform != "win32" and memory_mb > 0:
        import resource

        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    from .harden import harden

    harden()
    import mpmath  # noqa: F401 - heavy imports before the first call
    import numpy  # noqa: F401
    import scipy.optimize  # noqa: F401
    import sympy  # noqa: F401

    from .dispatch import handlers, run_tool
    from .envelope import build_error
    from .unitdim import warm_up

    handlers()
    warm_up()  # unit table of check_units: 1.46 s measured on 2026-11-24, outside the timeout
    write_frame(out, {"ready": os.getpid()})
    while True:
        try:
            msg = read_frame(inp)
        except (OSError, ValueError):
            return 1
        if msg is None:
            return 0
        tool, targs = msg.get("tool"), msg.get("args")
        try:
            resp = run_tool(str(tool), targs)
        except MemoryError:
            resp = build_error(
                str(tool), None, "LIMIT_EXCEEDED", "Out of memory in the computing process."
            )
        try:
            write_frame(out, resp)
        except (OSError, ValueError, MemoryError):
            return 1


if __name__ == "__main__":
    sys.exit(main())
