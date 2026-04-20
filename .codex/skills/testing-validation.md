---
name: testing-validation
description: Use when selecting or running WorldForge tests, reproducing CI failures, fixing ruff/format/coverage failures, checking generated provider docs, validating package builds, or preparing a public branch/release. Also use when a task says CI is failing, docs drift, coverage, package contract, lint, or release gate.
prerequisites: uv, bash, pytest, ruff.
---

# Testing And Validation

<purpose>
Choose the smallest reliable validation loop while preserving the full release gate for public behavior.
</purpose>

<context>
- CI quality job runs lock check, uv sync, ruff check, ruff format check, and coverage >=90.
- CI test matrix runs pytest on Ubuntu, macOS, Windows with Python 3.10-3.13.
- Package job runs `bash scripts/test_package.sh`.
- Provider catalog docs are generated into `README.md` and `docs/src/providers/README.md`.
</context>

<procedure>
1. Reproduce a reported failure with the exact failing command when available.
2. During iteration, run the narrowest tests that cover changed files.
3. For provider/catalog/docs work, always run `uv run python scripts/generate_provider_docs.py --check`.
4. Before publishing, run:
```bash
uv lock --check
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run python scripts/generate_provider_docs.py --check
uv run pytest
uv run pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90
bash scripts/test_package.sh
```
5. Inspect `git status --short --ignored` only when needed to ensure generated artifacts are not staged.
6. If a local environment lacks optional live runtimes, run checkout-safe demos and state the limitation.
</procedure>

<test_selection>
| Change area | Focused command |
| --- | --- |
| provider adapter | `uv run pytest tests/test_provider_contracts.py tests/test_<provider>_provider.py` |
| provider catalog/docs | `uv run pytest tests/test_provider_catalog.py tests/test_provider_catalog_docs.py` |
| CLI | `uv run pytest tests/test_cli_doctor.py tests/test_cli_help_snapshots.py` |
| evaluation/planning | `uv run pytest tests/test_evaluation_and_planning.py` |
| benchmarking | `uv run pytest tests/test_benchmark.py` |
| persistence/world state | `uv run pytest tests/test_world_lifecycle.py tests/test_helper_validations.py` |
| public exports | `uv run pytest tests/test_public_api.py` |
| package/import surface | `bash scripts/test_package.sh` |
</test_selection>

<patterns>
<do>
- Add failure-path tests for every documented failure mode.
- Run formatter only after behavior is stable.
- Treat generated docs drift as a code issue, not a manual README edit.
- Keep validation artifacts out of git unless the task explicitly asks for them.
</do>
<dont>
- Do not lower `--cov-fail-under=90`.
- Do not remove `scripts` from ruff command scope.
- Do not skip package validation after public API, package metadata, or include/exclude changes.
- Do not broaden dependencies to make an optional runtime test pass locally.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| `scripts/test_package.sh` fails but `pytest` passes | sdist/wheel missing files or import path issue | inspect hatch config and package import traceback |
| provider docs check fails | generated catalog stale | run `uv run python scripts/generate_provider_docs.py` and review diff |
| ruff format check fails | formatting drift | run `uv run ruff format src tests examples scripts` |
| coverage fails | new branch lacks tests | add focused behavior/error tests; do not weaken gate |
</troubleshooting>

<references>
- `.github/workflows/ci.yml`: CI quality, matrix, package jobs.
- `.github/workflows/security.yml`: dependency review and pip-audit.
- `Makefile`: local command aliases.
- `scripts/test_package.sh`: wheel install contract.
</references>
