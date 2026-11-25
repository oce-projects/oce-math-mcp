## What changes

<!-- One topic per pull request. Link the issue: "Closes #NUMBER". -->

## Why

## How it was tested

<!-- Reference values and their source (book, standard, paper, mpmath or SymPy computation). -->

## Checklist

- [ ] The new tests failed before the change and pass after it.
- [ ] Changing the new formula or condition on purpose makes a test fail.
- [ ] `uv run ruff check src tests` passes.
- [ ] `uv run ruff format --check src tests` passes.
- [ ] `uv run basedpyright` passes.
- [ ] `uv run pytest -q` passes (paste the last line below).
- [ ] `VERSION`, the README version badge and `CHANGELOG.md` are updated.
- [ ] A decision taken here is written in `docs/DECISIONS.md`.
- [ ] A change to a tool schema is in `docs/spec/` and was copied with `--sync-spec`.
- [ ] No personal data, tokens, private addresses or local paths.

```
<!-- last line of pytest -->
```
