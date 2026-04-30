# Contributing

WorldForge contributions should keep code, tests, docs, and agent context in sync.

```bash
uv sync --group dev
uv lock --check
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run python scripts/generate_provider_docs.py --check
uv run mkdocs build --strict
uv run pytest
uv run --extra harness pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90
bash scripts/test_package.sh
uv build --out-dir dist --clear --no-build-logs
```

Before tags or package publishing, also run the locked dependency audit from
[Operations](./operations.md). `uv run python scripts/generate_provider_docs.py --check` plus
`uv run mkdocs build --strict` verifies generated provider docs and builds the MkDocs Material site
in strict mode. `bash scripts/test_package.sh` checks the wheel/sdist contents before installing
the built wheel and running tests against the installed package.

Key directories:

- `src/worldforge/models.py`: public data contracts and validation.
- `src/worldforge/framework.py`: runtime facade, worlds, planning, persistence, and diagnostics.
- `src/worldforge/providers/`: provider interfaces, catalog, adapters, and scaffolds.
- `src/worldforge/testing/`: reusable provider contract helpers.
- `src/worldforge/evaluation/`: deterministic evaluation suites.
- `src/worldforge/benchmark.py`: provider benchmark harness.
- `src/worldforge/observability.py`: provider event sinks.
- `docs/src/`: user docs, architecture, playbooks, provider pages, and API notes.
- `tests/`: behavior and regression tests.
- `examples/`: runnable examples and compatibility wrappers.
- `scripts/`: docs generation, scaffolding, package validation, and optional smokes.

Provider work belongs in `src/worldforge/providers/`. Keep adapter capabilities honest and add
tests for every new supported path.

For adapter packages and in-repo providers, use the reusable contract helper:

```python
from worldforge.testing import assert_provider_contract

report = assert_provider_contract(provider)
print(report.exercised_operations)
```

Score-capable providers must pass provider-specific score fixtures:

```python
report = assert_provider_contract(
    provider,
    score_info=score_fixture["info"],
    score_action_candidates=score_fixture["action_candidates"],
)
```

Before publishing a branch:

- run the full release gate from [User And Operator Playbooks](./playbooks.md).
- update provider docs and generated catalog tables for provider behavior changes.
- update [Python API](./api/python.md) for public API or exception changes.
- update [Architecture](./architecture.md) for new flows or ownership boundaries.
- update [Operations](./operations.md) and [Playbooks](./playbooks.md) for new operator work.
- update `CHANGELOG.md` for user-visible changes.
- update `mkdocs.yml` when the docs navigation changes.
- update `AGENTS.md` for new commands, constraints, gotchas, or architecture facts.
