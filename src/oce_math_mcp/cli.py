"""Command line: stdio or HTTP server, schema dump, spec synchronisation, benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="oce-math-mcp", description="Mathematics MCP server (oce-math-mcp)."
    )
    ap.add_argument("--version", action="version", version=f"oce-math-mcp {__version__}")
    ap.add_argument(
        "--http", action="store_true", help="Streamable HTTP instead of stdio (path /mcp)."
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Computing processes (default: OCE_MATH_POOL_SIZE or CPU-1, max 4).",
    )
    ap.add_argument(
        "--no-pool",
        action="store_true",
        help="Compute in the server process (tests only: no killing timeout).",
    )
    ap.add_argument(
        "--dump-schemas", action="store_true", help="Print the schemas of the 10 tools and exit."
    )
    ap.add_argument("--format", choices=["mcp", "openai"], default="mcp")
    ap.add_argument(
        "--sync-spec",
        action="store_true",
        help="Copy the contract into the package, the spec and the optional mirrors (OCE_MATH_DATASET_DIR, OCE_MATH_RAG_DOC).",
    )
    ap.add_argument(
        "--note",
        default=None,
        help="With --sync-spec: changelog line of the mirrored spec document for the new version.",
    )
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure start, latency and restart of the computing processes.",
    )
    ap.add_argument(
        "--tir-replay",
        nargs="+",
        metavar="FILE",
        default=None,
        help="Run the code of TIR datasets (.jsonl or .parquet) in the sandbox and compare it with the recorded result (D13).",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="With --tir-replay: replay only N rows picked at random (seed --seed).",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="With --sample: seed of the sample (default 0)."
    )
    ap.add_argument(
        "--examples",
        type=int,
        default=5,
        help="With --tir-replay: examples saved for every error kind, with the row number (default 5).",
    )
    ap.add_argument(
        "--benchmark-md",
        default=None,
        help="With --benchmark: also write the Markdown table to this file.",
    )
    ap.add_argument(
        "--install-lean",
        action="store_true",
        help="Install elan, Lean 4 and Mathlib inside the repo (D17).",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.dump_schemas:
        from .schemas import dump

        sys.stdout.write(dump(args.format) + "\n")
        return 0
    if args.sync_spec:
        from .sync_spec import sync

        report = sync(note=args.note)
        sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0 if report["ok"] else 1
    if args.install_lean:
        from .lean_setup import install

        def log(msg: str) -> None:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

        report = install(log)
        sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0 if report["lean_available"] else 1
    if args.tir_replay:
        from pathlib import Path

        from .tir_replay import run as replay

        files = [Path(p) for p in args.tir_replay]
        try:
            if args.no_pool:
                rep = replay(files, keep=args.examples, sample=args.sample, seed=args.seed)
            else:
                # Process pool: real timeout and a new process after every python_sandbox (D29).
                from .pool import ComputePool

                tir_pool = ComputePool(size=args.pool_size)
                try:
                    rep = replay(
                        files,
                        keep=args.examples,
                        sample=args.sample,
                        seed=args.seed,
                        call=tir_pool.call,
                        workers=tir_pool.size,
                    )
                    rep["processes"] = tir_pool.size
                finally:
                    tir_pool.close()
        except RuntimeError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        sys.stdout.write(json.dumps(rep, ensure_ascii=False, indent=2) + "\n")
        return 0
    if args.benchmark:
        from .benchmark import run, to_markdown

        result = run()
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if args.benchmark_md:
            with open(args.benchmark_md, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(to_markdown(result))
        return 0

    from .pool import ComputePool
    from .server import build_server

    pool = None if args.no_pool else ComputePool(size=args.pool_size)
    server = build_server(pool)
    try:
        if args.http:
            server.run("streamable-http", host=args.host, port=args.port)
        else:
            server.run("stdio")
    finally:
        if pool is not None:
            pool.close()
    return 0
