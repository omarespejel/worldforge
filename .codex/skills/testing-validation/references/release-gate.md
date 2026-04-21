# Release Gate

The full local gate. Run this before opening a public PR, tagging, publishing,
or merging anything that changes public behavior. Each line maps to a CI job
that protects against a specific class of regression — none of them are
optional.

```bash
uv lock --check                                                # dep drift
uv run ruff check src tests examples scripts                    # lint
uv run ruff format --check src tests examples scripts           # format drift
uv run python scripts/generate_provider_docs.py --check         # README catalog drift
uv run pytest                                                   # behavior
uv run --extra harness pytest \
  --cov=src/worldforge --cov-report=term-missing \
  --cov-fail-under=90                                           # coverage incl. TUI
bash scripts/test_package.sh                                    # wheel/sdist contract
```

## What each step catches

| Step | Class of regression |
| --- | --- |
| `uv lock --check` | dependency drift — quietly diverges contributor resolutions |
| `ruff check` | banned patterns, dead imports, simple bugs |
| `ruff format --check` | hand-edits that bypassed the formatter |
| `generate_provider_docs.py --check` | README/provider catalog drift; catalog is generated, never hand-edited |
| `pytest` | behavior, including failure-path contracts |
| coverage ≥ 90 % with `--extra harness` | silent untested branches; harness extra includes TUI |
| `test_package.sh` | wheel/sdist actually installs and imports — catches hatch include/exclude mistakes |

## If a step fails

Fix at its level. The wrong moves:

- lowering `--cov-fail-under=90` to make coverage pass
- hand-editing the README catalog block instead of regenerating
- dropping `scripts` from the ruff invocation
- broadening base dependencies to satisfy an optional-runtime test
- skipping `test_package.sh` because "pytest passed"

If you genuinely need to weaken a gate, that's a `<gated>`-zone change in
`CLAUDE.md` and needs explicit user approval first.
