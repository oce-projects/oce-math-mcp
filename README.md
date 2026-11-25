[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-d9342b?logo=buymeacoffee&logoColor=white)](https://buymeacoffee.com/oscarcontatoengineering)
[![Donate via PayPal](https://img.shields.io/badge/Donate%20via%20PayPal-0070ba?logo=paypal&logoColor=white)](https://www.paypal.biz/OSCARCONTATO94)

# oce-math-mcp

## Badges

![Version](https://img.shields.io/badge/version-0.3.0-blue)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Platform](https://img.shields.io/badge/platform-windows-lightgrey)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

A mathematics MCP (Model Context Protocol) server for local language models. It is built to train
**Qwen3.6** and to let it work on calculations that a model cannot do in its head: exact
arithmetic, symbolic algebra, calculus, number theory, numerical methods, answer checking, the
Italian calendar, 3D geometry for CAD/CNC, formal proofs in Lean 4 and 78 engineering, physics
and chemistry operations.

**Status: version 0.3.0 (in progress; 0.2.0 is the last published version).** All 10 tools answer.
The contract defines 191 operations and all 191 are implemented (for `eng_calc` a test compares
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
| `math_solve` | symbolic algebra and calculus with SymPy, with a check of the solutions | 34 |
| `number_theory` | primes, factorisation, modular arithmetic, Diophantine equations | 26 |
| `math_numeric` | arbitrary precision, numerical sums and integrals, optimisation, ODEs with events | 10 |
| `math_verify` | checks an answer: equivalence, substitution, derivatives, inequalities, units | 7 |
| `python_sandbox` | short Python programs in a restricted interpreter (counting, brute force, sequences) | program |
| `lean_check` | checks a proof in Lean 4 with Mathlib (60 s limit) | proof |
| `date_calc` | Italian calendar `Europe/Rome`: age, holidays, working days, time zones | 12 |
| `geom3d_calc` | vectors, rotations, quaternions, solids, CNC, 3D drawing calculations | 24 |
| `eng_calc` | engineering formulas with SI units and the formula used | 78 |

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
Inglis) rotor balancing (one and two planes, ISO balance grades) and rolling without slipping.

The full contract (names, JSON schemas, examples) is in `docs/spec/SPEC_OCE_MATH_MCP_V3.md` and in
`docs/spec/oce_math_mcp_tools.json`. Every response has 9 fixed fields: `ok`, `tool`,
`operation`, `result`, `units`, `formula_used`, `formatted`, `warnings`, `error`.

## Installation

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --compile-bytecode
```

`--compile-bytecode` is needed once: without it every computing process recompiles SymPy and SciPy
when `PYTHONDONTWRITEBYTECODE` is set (start measured at 4.997 s against 2.853 s, decision D34).

If the repo is on a network drive, keep the Python environment on a local drive (decision D26):

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\oce-math-mcp"; uv sync --compile-bytecode
```

Platform: tested on Windows. The Linux code path (memory limit with `RLIMIT_AS`, process groups)
exists but is not tested.

## Running

`stdio` transport (the one MCP clients use):

```powershell
uv run oce-math-mcp
```

Streamable HTTP on `http://127.0.0.1:8012/mcp`:

```powershell
uv run oce-math-mcp --http --port 8012
```

Example configuration for an MCP client:

```json
{
  "mcpServers": {
    "oce-math": { "command": "uv", "args": ["run", "--directory", "<path of the repo>", "oce-math-mcp"] }
  }
}
```

## Options

| Option | Effect |
|---|---|
| `--http`, `--host`, `--port` | streamable HTTP instead of stdio (defaults `127.0.0.1` and `8012`) |
| `--pool-size N` | computing processes (default: `OCE_MATH_POOL_SIZE`, otherwise CPU-1, at most 4) |
| `--no-pool` | compute in the server process, for tests only: the timeout cannot stop the computation |
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
| `OCE_MATH_DATASET_DIR` | optional mirror folder of the spec for `--sync-spec` |
| `OCE_MATH_RAG_DOC` | optional knowledge-base copy of the spec for `--sync-spec` |

## Lean 4 and Mathlib

`lean_check` works only after the installation:

```powershell
uv run oce-math-mcp --install-lean
```

The command downloads elan, installs the toolchain pinned in `lean/lean-toolchain`, then downloads
Mathlib and its precompiled files (`lake exe cache get`). Without Lean, `lean_check` answers with a
clear error and the other tools keep working. About 10 GB are needed: measured 3,287.2 MB for elan
and 6,993.7 MB for Mathlib. Measured times: `docs/BENCHMARK.md`.

If the repo is on a network drive, install Lean on a local drive (decision D37: on the network
drive the installation failed 2 times out of 2). Give the same variables to the server:

```powershell
$env:OCE_MATH_ELAN_HOME = "$HOME\.oce-math-mcp\elan"; $env:OCE_MATH_LEAN_PROJECT = "$HOME\.oce-math-mcp\lean"; uv run oce-math-mcp --install-lean
```

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

## Development

Tests run on a local machine, not in GitHub Actions.

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

## Support

If this project is useful to you, you can support it with the buttons at the top of this page.

## Licence

Apache License 2.0, see `LICENSE` and `NOTICE`.

Third-party data inside the package (`src/oce_math_mcp/data/`), with source and attribution in
`NOTICE`: nuclide masses and decays from AME2020 and NUBASE2020 (CC BY 3.0 licence); abridged
atomic weights from IUPAC CIAAW. The CODATA physical constants come from SciPy (a dependency).
