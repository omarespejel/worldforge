# Contributing

WorldForge changes should keep the provider boundary truthful. A contribution is not complete when
the code compiles; it is complete when the public contract, tests, docs, and agent context agree.

## Workflow

WorldForge follows the standard GitHub fork-and-pull-request flow against `main`.

1. Fork the repository and clone your fork.
2. Create a topic branch: `git checkout -b feat/<short-description>` (or `fix/`, `docs/`, `chore/`).
3. Run `uv sync --group dev` and the validation gate below before opening the PR.
4. Open the pull request against `AbdelStark/worldforge:main`. Keep the title imperative and
   under ~70 characters; describe user-visible changes and link related issues in the body.
5. Address review feedback with new commits on the same branch; squash-merging is the default.

Direct pushes to `main` are reserved for maintainers performing release operations.

## Setup

```bash
uv sync --group dev
uv run worldforge doctor
uv run worldforge examples
```

## Validation Gates

Run the focused gate while iterating, then run the full gate before publishing work.

```bash
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

Before tagging or publishing, run the same gate plus the locked dependency audit documented in
[docs/src/playbooks.md](./docs/src/playbooks.md).

`bash scripts/test_package.sh` is the packaging contract check. It builds the wheel and sdist with `uv`,
checks the distribution contents, installs the wheel into an isolated virtual environment, and runs
the root test suite against the installed package.
`uv run python scripts/generate_provider_docs.py --check` plus `uv run mkdocs build --strict`
verifies the generated provider catalog and builds the MkDocs Material site in strict mode.

The exact release gate is documented in [docs/src/playbooks.md](./docs/src/playbooks.md).

## Repository Layout

- `src/worldforge/models.py`: public domain models, validation helpers, request policies, and
  serialization contracts.
- `src/worldforge/framework.py`: `WorldForge`, `World`, persistence, prediction, comparison,
  planning, diagnostics, and evaluation entry points.
- `src/worldforge/providers/`: provider interfaces, catalog, concrete adapters, and scaffolds.
- `src/worldforge/testing/`: reusable provider contract helpers for adapter packages.
- `src/worldforge/evaluation/`: deterministic evaluation suites and report rendering.
- `src/worldforge/benchmark.py`: provider benchmark harness.
- `src/worldforge/observability.py`: provider event sinks.
- `docs/src/`: user docs, architecture, playbooks, provider docs, and API notes.
- `tests/`: unit, contract, CLI, packaging, and regression tests.
- `examples/`: runnable checkout examples and compatibility wrappers.
- `scripts/`: docs generation, provider scaffolding, package validation, and optional smokes.

## Standards

- Keep public APIs typed, serializable, and explicit about failure modes.
- Fail fast on invalid inputs instead of silently coercing them.
- Use `ProviderError` for provider/runtime failures.
- Use `WorldForgeError` for invalid caller input and public model validation failures.
- Use `WorldStateError` for malformed persisted or provider-supplied world state.
- Do not advertise a provider capability that is not implemented end to end.
- Keep optional model runtimes, robot stacks, checkpoints, datasets, and credentials out of the
  base dependency set and out of the repository.
- Keep local JSON persistence documented as single-writer unless a dedicated persistence adapter is
  designed and reviewed.
- Keep docs aligned with the live package surface.

## Adding Or Promoting A Provider

1. Start from [docs/src/provider-authoring-guide.md](./docs/src/provider-authoring-guide.md).
2. Use `scripts/scaffold_provider.py` for new adapter skeletons.
3. Declare only the capabilities the adapter actually supports.
4. Fail clearly on missing credentials, optional dependencies, malformed inputs, malformed
   upstream outputs, partial outputs, expired artifacts, and unsupported flows.
5. Register the provider only when auto-detection is safe.
6. Add fixture-driven tests for happy paths and every documented failure mode.
7. Run `worldforge.testing.assert_provider_contract()` in adapter tests for supported surfaces.
8. Update provider docs, generated provider catalog tables, README or API docs when public
   behavior changes, `CHANGELOG.md`, and `AGENTS.md` when future contributors need new context.

## Documentation Changes

Public behavior changes need docs in the same branch. Use this routing:

- README for the front-door story and common commands.
- `docs/src/architecture.md` for new components, flows, or ownership boundaries.
- `docs/src/playbooks.md` for operator or maintainer runbooks.
- provider pages for provider-specific config, limits, examples, failure modes, and validation.
- `docs/src/api/python.md` for public API and exception behavior.
- `CHANGELOG.md` for user-visible changes.
- `AGENTS.md` for repo commands, constraints, gotchas, and agent-facing context.

## Pull Request Checklist

- lint, docs check, tests, coverage, package check, and build pass.
- new behavior has tests, including error paths.
- provider docs and generated catalog tables are current.
- README and API docs reflect public contract changes.
- `CHANGELOG.md` records user-visible changes.
- `AGENTS.md` records new constraints, commands, or gotchas.
- no secrets, `.env` files, checkpoints, datasets, generated runtime artifacts, or optional heavy
  dependencies are committed accidentally.
