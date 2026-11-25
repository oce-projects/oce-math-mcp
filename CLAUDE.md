# oce-math-mcp: instructions for agents

## What it is

A mathematics MCP server for Qwen3.6 and Claude. Goal: an extreme level of Qwen in mathematics,
with deterministic and verifiable tools. The server also produces the true answers of the
training dataset `mix_v4`.

## Read before working

1. `docs/DECISIONS.md`: decisions already taken are not argued again.
2. `docs/spec/`: the JSON-Schema contract (`SPEC_OCE_MATH_MCP_V3.md`, `oce_math_mcp_tools.json`).
3. `CLAUDE.local.md`, if present: private local notes (paths, knowledge base). It is not in git.

## Rules

- **Schema contract**: tool names and JSON schemas are shared with the training pipeline. A change
  goes into `docs/spec/oce_math_mcp_tools.json` and the spec, then `oce-math-mcp --sync-spec`
  copies it to the package and to the optional mirrors (`OCE_MATH_DATASET_DIR`, `OCE_MATH_RAG_DOC`).
- **Security**: never `eval`, `exec`, `compile`, `sympify`, `parse_expr`, `__import__` on input
  text. Allow-list AST parser. Timeout in a separate process (a thread cannot be stopped).
- **TDD**: first the failing test, then the code. Every bug found becomes a test.
- **Measured numbers**: times, counts and percentages come from a script or a test, never from a
  guess. What is not measured is written as "not tested, assumption".
- **Version and changelog** updated at every change (`VERSION` and `CHANGELOG.md`).
- **Quality**: `ruff check`, `ruff format --check`, `basedpyright`, `pytest` all green before
  saying "done", with the output as proof.
- **No GitHub Actions**: tests run on a local machine.
- **No multi-agent workflows**: at most one focused agent.
- **Document** every decision in `docs/DECISIONS.md`.
- **Public repo**: English only; no personal data, passwords, tokens, private network addresses or
  local paths. `tests/test_publishable.py` checks the files git would publish.
