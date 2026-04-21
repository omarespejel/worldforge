---
name: testing-validation
description: Use whenever the task involves running, selecting, or fixing WorldForge tests, reproducing a CI failure, fixing ruff / format / coverage / docs-drift errors, validating the wheel/sdist via the package contract script, or preparing a public branch or release. Trigger on phrases like "tests are failing", "CI red", "lint", "ruff", "coverage below 90", "package check fails", "scripts/test_package.sh", "docs check", "release gate", "regenerate provider docs", "before publishing". Also trigger when the user asks "what should I run to verify this?" — that question almost always wants the right slice of the gate.
---

# Testing And Validation

The repo runs a multi-job CI gate (lint, format, docs-check, pytest matrix on Ubuntu/macOS/Windows × Py 3.10–3.13, coverage ≥ 90 %, and the wheel/sdist contract). Reproducing the right slice locally is faster and cheaper than waiting for CI. This skill picks the smallest reliable loop for the change at hand and keeps the full release gate intact for public behavior.

## Fast start

Most calls are one of these. Pick by the change you just made.

```bash
# Touched ONE file or a small slice — fastest feedback
uv run pytest tests/test_<area>.py -x

# Touched a provider / catalog / docs block
uv run pytest tests/test_provider_contracts.py tests/test_provider_catalog*.py
uv run python scripts/generate_provider_docs.py --check

# About to publish, open a PR, or cut a release
# (full gate — see references/release-gate.md for the canonical bundle)
uv run python scripts/generate_provider_docs.py --check && \
uv lock --check && \
uv run ruff check src tests examples scripts && \
uv run ruff format --check src tests examples scripts && \
uv run pytest && \
uv run --extra harness pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90 && \
bash scripts/test_package.sh
```

If any step in the release gate fails, fix it at its level — do not lower the coverage gate, do not skip the docs check, do not narrow the ruff scope.

## Why this skill exists

The gate exists because each job catches a class of regression that the others can't:

- **`uv lock --check`** — dependency drift. A merged change without a lock refresh quietly diverges every contributor's resolution.
- **`ruff check`** + **`ruff format --check`** — style / banned-pattern drift. Format failures usually mean a hand-edit slipped past the formatter.
- **`generate_provider_docs.py --check`** — README / docs catalog drift. The catalog block is generated; a hand-edit is a real bug.
- **`pytest`** — behavior. Failing-path coverage is non-optional because adapters lie about their failure shape more often than their happy path.
- **coverage ≥ 90 %** — silent untested branches. Lowering the gate to land a change is the wrong move; add the tests.
- **`scripts/test_package.sh`** — wheel / sdist actually installs and imports. Catches `pyproject.toml` `include`/`exclude` mistakes that pytest cannot see.
- **harness extra in coverage** — `--extra harness` ensures Textual-gated TUI tests run; without it, the TUI module quietly drops out of coverage.

When something in the gate is "annoying", the right move is to understand which class of regression it's catching, not to disable it.

## Test selection — the smallest reliable slice

| Change area | Focused command |
| --- | --- |
| provider adapter | `uv run pytest tests/test_provider_contracts.py tests/test_<provider>_provider.py` |
| provider catalog or generated docs | `uv run pytest tests/test_provider_catalog.py tests/test_provider_catalog_docs.py` and `uv run python scripts/generate_provider_docs.py --check` |
| CLI surfaces | `uv run pytest tests/test_cli_doctor.py tests/test_cli_help_snapshots.py` |
| evaluation / planning | `uv run pytest tests/test_evaluation_and_planning.py` |
| benchmarking | `uv run pytest tests/test_benchmark.py` |
| persistence / world state | `uv run pytest tests/test_world_lifecycle.py tests/test_helper_validations.py` |
| public exports | `uv run pytest tests/test_public_api.py` |
| package metadata, hatch include/exclude, wheel surface | `bash scripts/test_package.sh` |
| TUI (Textual) | `uv run --extra harness pytest tests/test_harness*.py` |

After the focused slice passes, run the full gate (`references/release-gate.md`) before declaring done on anything that changes public behavior.

## Activation cues

Trigger on:
- "tests failing", "CI red", "fix lint", "ruff complaining", "format check fails"
- "coverage dropped", "coverage below 90", "what coverage is this missing"
- "docs check fails", "provider catalog drift", "README differs"
- "package contract", "test_package.sh", "wheel import broke"
- "what should I run to verify <change>"
- preparing a PR, opening a release, tagging, publishing

Do **not** trigger for:
- adding / changing a provider's behavior — load `provider-adapter-development` first; this skill comes later for the gate
- evaluation suite semantics — load `evaluation-benchmarking`
- writing the change itself — this skill is about *validating* changes

## Stop and ask the user

- before lowering `--cov-fail-under=90`, removing `scripts` from ruff scope, or skipping `scripts/test_package.sh`
- before modifying anything under `.github/workflows/` (gated)
- before adding or removing a test gate from CI

## Patterns

**Do:**
- Add a failure-path test for every documented failure mode you change. Happy paths drift less than error envelopes.
- Run formatter only after behavior is stable; reformat-first hides which lines you actually changed.
- Treat generated-docs drift as a code issue — regenerate, do not hand-edit the catalog block.
- Keep generated artifacts (dist/, build/, .worldforge/, coverage HTML) out of git.

**Don't:**
- Lower the coverage gate to land a change.
- Drop `scripts` from the ruff invocation; helper scripts ship with the repo.
- Skip the package contract after public-API, package-metadata, or hatch include/exclude changes.
- Broaden base dependencies to make an optional-runtime test pass locally — that's a different skill.

## Troubleshooting

| Symptom | Likely cause | First fix |
| --- | --- | --- |
| `scripts/test_package.sh` fails but `pytest` passes | wheel/sdist missing a file or a bad import path | inspect the hatch `[tool.hatch.build.targets.*]` and the test_package.sh traceback |
| provider docs check fails | generated catalog stale | `uv run python scripts/generate_provider_docs.py` and review the diff |
| ruff format check fails | formatting drift | `uv run ruff format src tests examples scripts` then re-check |
| coverage dropped below 90 | new branch lacks tests | add focused behavior + error-path tests; never weaken the gate |
| harness tests not counted | missing `--extra harness` | use the harness coverage command from the fast start |
| `uv lock --check` fails | dependency change without lock refresh | confirm the dependency change is approved, then refresh the lock |

## References

- `references/release-gate.md` — the canonical full-gate command bundle and what each step protects against
- `.github/workflows/ci.yml` — CI quality, matrix, package jobs (gated; ask before editing)
- `.github/workflows/security.yml` — dependency review and pip-audit
- `Makefile` — local command aliases
- `scripts/test_package.sh` — wheel install contract
