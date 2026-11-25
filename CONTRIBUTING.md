# Contributing to oce-math-mcp

Thank you for helping. This server gives numbers that people and models trust, so every change needs a
proof that it is right. This page says how to propose a change and what a change must contain.

## Before you start

- Read `docs/DECISIONS.md`. The D numbers quoted in the code are listed there. A decision already taken is
  not argued again in a pull request; open an issue first if you think one is wrong.
- For anything larger than a small fix, open an issue first and describe the change. It avoids work that
  cannot be merged.
- Everything in the repo is in English: code, comments, documents, commit messages, issues.

## Set up

```powershell
git clone https://github.com/oce-projects/oce-math-mcp.git
```

```powershell
cd oce-math-mcp
```

```powershell
uv sync --compile-bytecode
```

```powershell
uv run oce-math-mcp --self-check
```

The last command must print `"ok": true`. Lean 4 is needed only for the `lean_check` tests; without it they
are skipped.

## Rules for every change

1. **Test first.** Write the test, run it and see it fail, then write the code. Every bug found becomes a
   test.
2. **Check that the test can fail.** Change the new formula or condition on purpose (a sign, a constant,
   a comparison) and run the test again: it must fail. A test that still passes does not prove anything.
3. **Independent reference values.** The expected value in a test comes from a source independent of the
   code under test: a published worked example, a standard, a table, or a separate computation with
   `mpmath` or SymPy. Write the source in the test.
4. **Measured numbers only.** Times, counts and percentages in documents come from a script or a test.
   Anything not measured is written as "not tested".
5. **Security.** Never call `eval`, `exec`, `compile`, `sympify`, `parse_expr` or `__import__` on input
   text. Expressions are built by the allow-list AST parser (decisions D5 and D31). Every computation that
   can run long goes through the process pool, so that the timeout can stop it (D32).
6. **Contract.** Tool names and JSON schemas are a contract shared with training data. A change goes into
   `docs/spec/oce_math_mcp_tools.json` and `docs/spec/SPEC_OCE_MATH_MCP_V3.md` first; then
   `uv run oce-math-mcp --sync-spec --note "..."` copies it into the package.
7. **Version and history.** Update `VERSION`, the version badge in `README.md` (a test compares them),
   `CHANGELOG.md`, and `docs/DECISIONS.md` when the change takes a decision.
8. **No personal data.** No names, e-mail addresses, tokens, private network addresses or local paths.
   `tests/test_publishable.py` scans the files git would publish.

## New formula or operation

An `eng_calc` operation (or another tool operation) is complete when it has:

- SI units in and out, with the unit of every result in `units`;
- `formula_used`, the formula in plain text;
- a validity range: a warning when the inputs leave it, an error when the formula has no meaning;
- the source of the formula (book, standard or paper) in the code;
- tests against an independent reference (rule 3), including at least one limit or extreme case;
- an entry in the contract (rule 6).

## Checks before a pull request

All four must pass, on your machine. There is no GitHub Actions pipeline (decision taken for this
project); paste the last line of each command in the pull request.

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

## Pull requests

- One topic per pull request.
- Fill in the template: what changes, why, how it was tested, which reference values you used.
- Keep commits small and their messages in English.

## Issues

- Wrong result, crash or bad error message: use the "Bug or wrong result" form.
- New formula, operation or tool: use the "New formula or operation" form.
- Security problem: do not open a public issue; follow `SECURITY.md`.

Include the output of `uv run oce-math-mcp --self-check` and the exact tool call. Remove personal data
first.

## Licence

By sending a contribution you agree that it is released under the Apache License 2.0 of this repo
(section 5 of `LICENSE`).

## Conduct

Everyone taking part follows `CODE_OF_CONDUCT.md`.
