[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-d9342b?logo=buymeacoffee&logoColor=white)](https://buymeacoffee.com/oscarcontatoengineering)
[![Donate via PayPal](https://img.shields.io/badge/Donate%20via%20PayPal-0070ba?logo=paypal&logoColor=white)](https://www.paypal.biz/OSCARCONTATO94)

# oce-math-mcp

## Badges

![Version](https://img.shields.io/badge/version-0.4.0-blue)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux-lightgrey)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

A mathematics MCP (Model Context Protocol) server for local language models. It is built to train
**Qwen3.6** and to let it work on calculations that a model cannot do in its head: exact
arithmetic, symbolic algebra, calculus, number theory, numerical methods, answer checking, the
Italian calendar, 3D geometry for CAD/CNC, formal proofs in Lean 4 and 89 engineering, physics
and chemistry operations.

**Status: version 0.4.0.** All 10 tools answer.
The contract defines 208 operations and all 208 are implemented (for `eng_calc` a test compares
the contract with the code).

## Why

The base model gets almost every complex calculation wrong: in the starting benchmark
Qwen3.6-35B-A3B gave the exact result in 1 problem out of 12. A model should not compute in its
head: it should call a deterministic, verifiable tool. The same server also produces the true
answers of the training dataset.

## Tools

| Tool | What it does | Operations |
|---|---|---|
| `math_eval` | exact arithmetic (decimal, fraction, float), functions, statistics, Italian VAT and withholding tax, units, physical constants | expression |
| `math_solve` | symbolic algebra and calculus with SymPy, with a check of the solutions | 39 |
| `number_theory` | primes, factorisation, modular arithmetic, Diophantine equations | 26 |
| `math_numeric` | arbitrary precision, numerical sums and integrals, optimisation, ODEs with events, function plots (SVG) | 11 |
| `math_verify` | checks an answer: equivalence, substitution, derivatives, inequalities, units | 7 |
| `python_sandbox` | short Python programs in a restricted interpreter (counting, brute force, sequences) | program |
| `lean_check` | checks a proof in Lean 4 with Mathlib (60 s limit) | proof |
| `date_calc` | Italian calendar `Europe/Rome`: age, holidays, working days, time zones | 12 |
| `geom3d_calc` | vectors, rotations, quaternions, solids, CNC, 3D drawing calculations | 24 |
| `eng_calc` | engineering formulas with SI units and the formula used | 89 |

`eng_calc` covers: strength of materials, machine elements, fluids and heat transfer, circuits,
control, optics and acoustics, hydrostatics, pumps, open channels, flow meters, compressible flow,
ideal gases and cycles, calorimetry, fatigue, bolts, welds, a material database with cited
sources, thermal stresses, Hertz contact, springs, interference fits, strain rosettes, CODATA 2022
physical constants, chemistry (molar mass, stoichiometry, pH, solutions, kinetics,
electrochemistry), nuclear physics (AME2020 and NUBASE2020 data), modern physics and special
relativity, water and steam (IAPWS-IF97), psychrometrics, Nusselt correlations, e-NTU heat
exchangers, fins, lumped transients, radiation between surfaces, electrostatics, magnetism and
induction, linkages and epicyclic gear trains, catenaries, plain bearings, 2D and 3D trusses,
torsion of non-circular sections, geometric optics (thick lenses, mirrors, refracting surfaces,
lens pairs), wave optics (Malus, Brewster, Fresnel, slits, gratings, thin films, Rayleigh) and practical
electronics (LED resistor, op-amp circuits, Wheatstone bridge, 555 timer, thermistors), friction in simple
machines (inclined plane, leadscrew, rolling resistance) and electric machines (transformer, DC, induction
and synchronous motors), thin plates (circular and rectangular) and continuous beams (theorem of three
moments, with settlements and fixed ends) local losses at sudden changes of section (Borda-Carnot) and stress concentration at holes (Kirsch,
Inglis) rotor balancing (one and two planes, ISO balance grades), rolling without slipping and the kinetic theory
of gases (molecular speeds, Maxwell-Boltzmann distribution, mean free path, effusion, equipartition) and
Newtonian gravitation (point masses, spheres and shells, rings, rotating bodies, Gauss's law) and the
gravitomagnetism of a rotating body (Lense-Thirring precession) and the gravitational waves of a circular
binary (power, orbital decay, time to coalescence, chirp mass).

The full contract (names, JSON schemas, examples) is in `docs/spec/SPEC_OCE_MATH_MCP_V3.md` and in
`docs/spec/oce_math_mcp_tools.json`. Every response has 9 fixed fields: `ok`, `tool`,
`operation`, `result`, `units`, `formula_used`, `formatted`, `warnings`, `error`.

## Installation

Five steps. An AI agent that installs the server for you can follow
[`llms-install.md`](llms-install.md) instead: the same steps, with a check after each one.

### 1. Requirements

- [Git](https://git-scm.com/downloads).
- [uv](https://docs.astral.sh/uv/getting-started/installation/), the Python package manager. If
  Python 3.12 or newer is missing, uv downloads it.
- About 0.6 GB of disk for the Python environment (564 MB measured on Windows, development tools
  included). Lean 4 is optional and needs about 10 GB more (see [Lean 4 and Mathlib](#lean-4-and-mathlib)).

Install uv on Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install uv on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal afterwards, so that `uv` is on the `PATH`.

### 2. Download and install

```powershell
git clone https://github.com/oce-projects/oce-math-mcp.git
```

```powershell
cd oce-math-mcp
```

```powershell
uv sync --compile-bytecode
```

`--compile-bytecode` is needed once: without it every computing process recompiles SymPy and SciPy
when `PYTHONDONTWRITEBYTECODE` is set (start measured at 4.997 s against 2.853 s, decision D34).

If the repo is on a network drive, keep the Python environment on a local drive (decision D26):

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\oce-math-mcp"; uv sync --compile-bytecode
```

### 3. Check the installation

```powershell
uv run oce-math-mcp --self-check
```

The command starts the server as an MCP client does (stdio), lists the tools and runs two
calls with known answers. It prints a JSON report and exits with code 0 only if every check passes:

```json
{
  "ok": true,
  "server_version": "0.4.0",
  "python": "3.13.13",
  "tools": 10,
  "checks": [
    {
      "name": "initialize",
      "ok": true,
      "detail": "oce-math-mcp 0.4.0"
    },
    {
      "name": "tools/list",
      "ok": true,
      "detail": "10 tools"
    },
    {
      "name": "math_eval",
      "ok": true,
      "detail": "2**100 + 1/3 = 3802951800684688204490109616129/3"
    },
    {
      "name": "eng_calc",
      "ok": true,
      "detail": "mirror f = 0.1 m, d_o = 0.3 m: d_i = 0.15 m (0.15 expected)"
    }
  ],
  "lean": "not installed (optional: --install-lean)",
  "seconds": 4.41
}
```

`lean` is information only: without Lean the other 9 tools work and the check still passes.

### 4. Register the server in your MCP client

Use the absolute path of the folder you cloned in step 2.

Claude Code (one command, available in every project with `--scope user`):

```powershell
claude mcp add --scope user oce-math -- uv run --directory C:\path\to\oce-math-mcp oce-math-mcp
```

Claude Desktop and other clients that read an `mcpServers` JSON file. For Claude Desktop the file
is `%APPDATA%\Claude\claude_desktop_config.json` on Windows and
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS:

```json
{
  "mcpServers": {
    "oce-math": {
      "command": "uv",
      "args": ["run", "--directory", "C:/path/to/oce-math-mcp", "oce-math-mcp"]
    }
  }
}
```

On Windows write the path with forward slashes, as above, or write every backslash twice: JSON reads
a single backslash as an escape. Restart the client after the change.

A local model without an MCP client (for example through an OpenAI-compatible API) can use the
same tools as function definitions:

```powershell
uv run oce-math-mcp --dump-schemas --format openai
```

### 5. Optional: Lean 4

Only `lean_check` needs it. See [Lean 4 and Mathlib](#lean-4-and-mathlib).

### Platform

Tested on Windows 10 with Python 3.12.13 and 3.13.13, and on Ubuntu 22.04 (x86_64) with Python
3.12.13 (version 0.4.0, 2026-11-27): `--self-check` passes and the test suite gives 1,591
passed and 9 skipped (Lean, `pyarrow` and the dataset mirror not installed). On Linux the memory
limit of the computing process is `RLIMIT_AS`: `/proc/<pid>/limits` of a worker shows 1,073,741,824
bytes for `memory_mb = 1024`, and a 12.8 GB array is refused. Not tested: Lean on Linux, and macOS.

### Common problems

| Problem | Cause and fix |
|---|---|
| `uv` is not found | open a new terminal after installing uv |
| the client cannot start the server, but `--self-check` passes | desktop apps do not always see the terminal `PATH`: put the full path of `uv` in `command` (`where.exe uv` on Windows, `which uv` on macOS and Linux) |
| the first call takes a few seconds | the server starts its computing processes: 3.603 s for one process, measured in `docs/BENCHMARK.md` |
| the first `lean_check` takes about 20 s | the Lean REPL loads Mathlib once (16.16 s measured); the next checks take less than 0.1 s. `OCE_MATH_LEAN_REPL=warm` loads it when the server starts (see below) |
| `uv sync` is slow or fails on a network drive | use a local environment with `UV_PROJECT_ENVIRONMENT` (step 2) |

## Running

The MCP client starts the server by itself (step 4). To start it by hand, from the repo folder:

`stdio` transport (the one MCP clients use):

```powershell
uv run oce-math-mcp
```

Streamable HTTP on `http://127.0.0.1:8012/mcp`:

```powershell
uv run oce-math-mcp --http --port 8012
```

The HTTP transport has no authentication. Keep the default host `127.0.0.1`; do not use
`--host 0.0.0.0` on a network you do not control.

## Options

| Option | Effect |
|---|---|
| `--http`, `--host`, `--port` | streamable HTTP instead of stdio (defaults `127.0.0.1` and `8012`) |
| `--pool-size N` | computing processes (default: `OCE_MATH_POOL_SIZE`, otherwise CPU-1, at most 4) |
| `--no-pool` | compute in the server process, for tests only: the timeout cannot stop the computation |
| `--self-check` | start the server over stdio, run 4 checks and print a JSON report (exit code 0 only if all pass) |
| `--dump-schemas --format mcp\|openai` | print the schemas of the 10 tools and exit |
| `--benchmark` | measure start, latency and restart of the computing processes (JSON result) |
| `--benchmark-md FILE` | with `--benchmark`: also write the Markdown table (for example `docs/BENCHMARK.md`) |
| `--tir-replay FILE...` | run the code of TIR files (`.jsonl` or `.parquet`) in `python_sandbox` and compare it with the recorded outputs (decision D13). Formats: tool calls, NuminaMath-TIR, OpenMathReasoning. Uses the process pool (real timeout) unless `--no-pool`; `.parquet` needs `uv run --with pyarrow` |
| `--sample N`, `--seed S` | with `--tir-replay`: replay N rows picked at random with seed S (default 0), a repeatable measurement |
| `--examples N` | with `--tir-replay`: examples saved for every error kind, with the row number (default 5) |
| `--install-lean` | install elan, Lean 4 and Mathlib inside the repo (decision D17) |
| `--sync-spec --note "..."` | copy the contract into the spec and into the optional mirrors |

| Environment variable | Effect |
|---|---|
| `OCE_MATH_POOL_SIZE` | number of computing processes |
| `OCE_MATH_ELAN_HOME` | elan folder (default `.lean-tools/elan` in the repo) |
| `OCE_MATH_LEAN_PROJECT` | Lean project with Mathlib (default `lean/` in the repo) |
| `OCE_MATH_LEAN_REPL` | `0`: no Lean REPL, every check runs `lake env lean`; `warm`: load Mathlib into the REPL when the server starts; unset: at the first check (decision D103) |
| `OCE_MATH_LEAN_REPL_DIR` | folder of the Lean REPL (default `repl` next to the elan folder) |
| `OCE_MATH_DATASET_DIR` | optional mirror folder of the spec for `--sync-spec` |
| `OCE_MATH_RAG_DOC` | optional knowledge-base copy of the spec for `--sync-spec` |

## Lean 4 and Mathlib

`lean_check` works only after the installation:

```powershell
uv run oce-math-mcp --install-lean
```

The command downloads elan, installs the toolchain pinned in `lean/lean-toolchain`, then downloads
Mathlib and its precompiled files (`lake exe cache get`). Last it clones the
[Lean REPL](https://github.com/leanprover-community/repl) (Apache-2.0) at the tag of the same Lean
version and builds it with `lake build repl` (27 s measured; `git` must be installed). Without Lean,
`lean_check` answers with a clear error and the other tools keep working. After the installation
`--self-check` reports `"lean": "installed"`. About 10 GB are needed: measured 3,287.2 MB for elan,
6,993.7 MB for Mathlib and 11 MB for the REPL. Measured times: `docs/BENCHMARK.md`.

If the repo is on a network drive, install Lean on a local drive (decision D37: on the network
drive the installation failed 2 times out of 2). Give the same variables to the server:

```powershell
$env:OCE_MATH_ELAN_HOME = "$HOME\.oce-math-mcp\elan"; $env:OCE_MATH_LEAN_PROJECT = "$HOME\.oce-math-mcp\lean"; uv run oce-math-mcp --install-lean
```

The server keeps one Lean REPL running with Mathlib loaded (decision D103). Every check starts from
the same Mathlib environment, so a definition made in one check is not visible in the next one.
Measured on 26/11/2026:

| Step | Time |
|---|---|
| `lake env lean` for every check (before D103) | 15.46 s for `#check Nat` |
| loading Mathlib into the REPL, once | 16.16 s |
| one check in the REPL | 0.004 to 0.034 s |

The REPL uses memory while the server runs: 3,130 MB in use after loading Mathlib, 3,374 MB after
1,000 checks. It is restarted every 1,000 checks, after a check that passes the 60 s limit
(decision D16), and after a crash. Code with its own `import` lines, or a REPL that cannot start,
is checked with `lake env lean` as before; `OCE_MATH_LEAN_REPL=0` always does that.

`lean_check` rejects code that can run programs or add axioms (`#eval`, `elab`, `macro`,
`run_cmd`, `axiom`, `native_decide` and others, decision D36). A proof is valid only if it uses
the standard axioms (`propext`, `Classical.choice`, `Quot.sound`).

## Security

- No `eval`, `exec`, `sympify` or `parse_expr` on model text: an allow-list AST parser builds the
  SymPy expressions (decisions D5 and D31). A test intercepts `eval` and `exec` during the
  computations and fails if model text reaches either of them.
- `python_sandbox` is an AST interpreter written here, not `exec`: modules and functions from an
  allow list, dangerous attributes blocked (D13, D28, D29). It accepts the usual constructs of TIR
  programs: `from sympy import *`, `global`, `sympy.S`, `nsimplify` (D38); `class` is refused.
- Limits: expressions up to 2,000 characters, exponents up to 10,000, a 5 s timeout per call.
  Computations run in separate processes: when the time runs out the process is killed and
  replaced (D32). Memory limit per process (Job Object on Windows, `RLIMIT_AS` on Linux).
- No exception leaves the server: every error has `code`, `message` and `example`.
- The streamable HTTP transport has no authentication: it listens on `127.0.0.1` by default. Do not
  bind it to `0.0.0.0` on a network you do not control.
- To report a vulnerability, see `SECURITY.md`.

## Development

How to propose a change, the test rules and how to open an issue: `CONTRIBUTING.md`. Tests run on a
local machine, not in GitHub Actions.

```powershell
uv run ruff check src tests
```

```powershell
uv run ruff format --check src tests
```

```powershell
uv run basedpyright
```

```powershell
uv run pytest -q
```

The suite includes real server runs over stdio and HTTP. The `lean_check` tests are skipped when
Lean is not installed. A test checks that the version badge above matches `VERSION`, and another
one scans the repo for personal data before publication.

## Documents

- `docs/spec/SPEC_OCE_MATH_MCP_V3.md`: contract of the 10 tools
- `docs/DECISIONS.md`: decision log (the D numbers quoted in the code)
- `docs/BENCHMARK.md`: measured times
- `CHANGELOG.md`: version history
- `llms-install.md`: installation guide for AI agents
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`: rules for contributors and for security reports

## Support

If this project is useful to you, you can support it with the buttons at the top of this page.

## Licence

Apache License 2.0, see `LICENSE` and `NOTICE`.

Third-party data inside the package (`src/oce_math_mcp/data/`), with source and attribution in
`NOTICE`: nuclide masses and decays from AME2020 and NUBASE2020 (CC BY 3.0 licence); abridged
atomic weights from IUPAC CIAAW. The CODATA physical constants come from SciPy (a dependency).
