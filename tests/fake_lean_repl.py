"""A fake Lean REPL for the tests of lean_repl (D103): same JSON protocol, no Lean.

Commands are JSON objects separated by blank lines; each answer is an indented JSON object and a blank line.
- {"cmd": "import Mathlib"} without env gives env 0 (after FAKE_REPL_IMPORT_S seconds, default 0).
- A command with an env other than 0 gets {"message": "Unknown environment."}.
- SLEEP in the command: no answer for 60 s. CRASH: the process exits with code 3. REFUSE: {"message": ...}.
- BAD: an error at line 2, column 4. ADMIT: a sorry only in "sorries".
- The axiom marker of the command is echoed with [propext].
Every start appends the PID to the file FAKE_REPL_PIDS, every command its env to FAKE_REPL_LOG.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time


def log(name: str, text: str) -> None:
    path = os.environ.get(name)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def answer(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, indent=1) + "\n\n")
    sys.stdout.flush()


def main() -> None:
    log("FAKE_REPL_PIDS", str(os.getpid()))
    counter = 0
    block: list[str] = []
    for line in sys.stdin:
        if line.strip():
            block.append(line)
            continue
        if not block:
            continue
        req = json.loads("".join(block))
        block = []
        cmd = req.get("cmd", "")
        log("FAKE_REPL_LOG", f"env={req.get('env')}")
        if "env" not in req:
            time.sleep(float(os.environ.get("FAKE_REPL_IMPORT_S", "0")))
            if os.environ.get("FAKE_REPL_IMPORT_FAILS"):
                answer(
                    {
                        "messages": [
                            {
                                "severity": "error",
                                "pos": {"line": 1, "column": 0},
                                "data": "unknown module prefix 'Mathlib'",
                            }
                        ],
                        "env": 0,
                    }
                )
                continue
            answer({"env": 0})
            continue
        if req["env"] != 0:
            answer({"message": "Unknown environment."})
            continue
        if "SLEEP" in cmd:
            time.sleep(60)
        if "CRASH" in cmd:
            sys.exit(3)
        if "REFUSE" in cmd:
            answer({"message": "Could not parse the command."})
            continue
        counter += 1
        messages = []
        if "BAD" in cmd:
            messages.append({"severity": "error", "pos": {"line": 2, "column": 4}, "data": "boom"})
        reply: dict = {}
        if (
            "ADMIT" in cmd
        ):  # a sorry reported only in "sorries", without the word and without a warning
            reply["sorries"] = [{"pos": {"line": 1, "column": 24}, "goal": "⊢ 1 = 1"}]
        marker = re.search(r"OCE_AXIOMS_\w+", cmd)
        if marker:
            messages.append(
                {
                    "severity": "info",
                    "pos": {"line": 9, "column": 0},
                    "data": f"{marker.group(0)} [propext]",
                }
            )
        answer({"messages": messages, **reply, "env": counter})


if __name__ == "__main__":
    main()
