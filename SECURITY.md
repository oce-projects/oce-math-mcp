# Security policy

## Supported versions

| Version | Security fixes |
|---|---|
| 0.3.x | yes |
| older than 0.3 | no |

## Reporting a vulnerability

Do not open a public issue. Use GitHub's private vulnerability reporting: open the **Security** tab of
this repository and choose **Report a vulnerability**. Only the maintainers can read the report.

Please include the version (`uv run oce-math-mcp --version`), the operating system, the exact tool call or
program that shows the problem, and what happens. This is a volunteer project: there is no guaranteed
response time, but every report is read.

## What the server protects against

The server runs mathematics written by a language model. It treats that text as untrusted.

| Protection | How | Decision |
|---|---|---|
| no code execution from model text | allow-list AST parser; no `eval`, `exec`, `sympify` or `parse_expr` on input | D5, D31 |
| `python_sandbox` | an AST interpreter written for this project, not `exec`; modules and functions from an allow list; dangerous attributes blocked | D13, D28, D29 |
| `lean_check` | code that can run programs or add axioms is refused (`#eval`, `elab`, `macro`, `run_cmd`, `axiom`, `native_decide` and others) | D36 |
| runaway computations | every call runs in a separate process; after 5 s the process is killed and replaced; Lean has 60 s | D6, D16, D32 |
| memory | 2,048 MB per computing process (Job Object on Windows, `RLIMIT_AS` on Linux) | D13 |
| input size | expressions up to 2,000 characters, exponents up to 10,000 in absolute value | D6 |

## In scope

Any way to make the server run code, read or write files, open network connections, or use more time or
memory than the limits above, starting from a tool call. Also any way to make `lean_check` accept a proof
that depends on a non-standard axiom.

## Out of scope

- The streamable HTTP transport has no authentication by design. It listens on `127.0.0.1` by default;
  exposing it with `--host 0.0.0.0` is the operator's choice.
- `--no-pool` disables the computing processes and with them the timeout. It exists for tests only.
- The Linux memory limit and process groups are not tested (the server is tested on Windows).
- A wrong mathematical result is a bug, not a vulnerability: report it with the "Bug or wrong result"
  issue form.
