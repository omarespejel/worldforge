# Engineering Quality Standards

WorldForge treats engineering quality as part of the public API. Provider capabilities, package
metadata, docs, tests, and optional robotics runtimes must stay aligned so users can reason about
what is installed, what is deterministic, and what remains host-owned.

## Reference Baseline

WorldForge's quality rules are grounded in the same upstream sources a production Python ML
framework should track:

- [Python Packaging User Guide: `pyproject.toml` metadata](https://packaging.python.org/specifications/declaring-project-metadata/)
  for static project metadata, build-system configuration, SPDX license metadata, entry points, and
  dependency declarations.
- [Python Packaging User Guide: packaging projects](https://packaging.python.org/tutorials/packaging-projects/)
  for `src` layout packaging and distribution structure.
- [uv package guide](https://docs.astral.sh/uv/guides/package/) for building and publishing
  packages through a modern build frontend while preserving the backend contract.
- [pytest good integration practices](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html)
  for `src` layout test isolation and `--import-mode=importlib`.
- [Ruff linter configuration](https://docs.astral.sh/ruff/linter/) and
  [Ruff formatter guidance](https://docs.astral.sh/ruff/formatter/) for a single fast lint and
  format toolchain.
- [PEP 561](https://peps.python.org/pep-0561/) and the
  [typing distribution spec](https://typing.python.org/en/latest/spec/distributing.html) for
  shipping inline type information through `py.typed`.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) for
  honest ML determinism boundaries: seeds reduce nondeterminism within a platform and release, but
  exact results are not guaranteed across releases, devices, or platforms.
- [scikit-learn developer guide](https://scikit-learn.org/dev/developers/develop.html) for stable
  public imports, estimator-style validation discipline, and explicit randomness contracts.
- [Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/) for the expectation
  that scientific Python projects document support windows and dependency policy rather than
  letting compatibility drift silently.

## Project Rules

### Packaging

- The package source lives under `src/worldforge`, and tests run against installed/importable
  package semantics rather than accidental repository-root imports.
- `pyproject.toml` is the single source of truth for package metadata, Python support, scripts,
  optional extras, uv package mode, and tool configuration.
- Wheels contain runtime package files only. Source distributions contain tests, docs, examples,
  scripts, and release metadata so downstream users can inspect and rebuild the project.
- `src/worldforge/py.typed` is part of the wheel contract. Removing it is a typing regression.
- Optional ML and robotics runtimes stay outside the base dependency set. `torch`, LeRobot,
  LeWorldModel, GR00T, CUDA, robot controllers, checkpoints, and datasets are supplied by the host
  environment for the specific smoke or showcase that needs them.

### Testing

- `pytest` runs with `--import-mode=importlib` so tests do not depend on implicit `sys.path`
  mutation.
- Test fixtures must be deterministic unless a test explicitly validates nondeterministic runtime
  handling.
- Every provider capability must have both a positive contract test and a failure-mode test for the
  boundary it documents.
- Public exception assertions should match literal messages precisely enough to catch regressions
  without depending on unrelated text.
- `xfail` is strict. A test that starts passing should be investigated and either promoted or
  removed.

### Linting And Style

- Ruff owns formatting-compatible linting and import ordering for `src`, `tests`, `examples`, and
  `scripts`.
- Public `__all__` exports stay sorted so public API diffs are reviewable.
- Mutable class metadata such as Textual `BINDINGS` and `SCREENS` must be annotated as `ClassVar`
  to separate framework declarations from instance state.
- Tests use direct `pytest` imports and split compound assertions when doing so improves failure
  localization.

### ML And Robotics Boundaries

- Deterministic in-repo suites are contract harnesses, not evidence of physical fidelity.
- Real robotics showcase paths must state which runtime owns preprocessing, checkpoints,
  observations, action translation, safety checks, and hardware execution.
- WorldForge validates tensor and action boundaries; it must not pad, project, or reinterpret
  mismatched action spaces.
- Score providers expose `score`. Policy providers expose `policy`. Predictive world models expose
  `predict`. Branding must not override executable capability truth.
- Provider events are log-facing records. Targets, messages, and metadata must remain sanitized
  before reaching JSON logs, in-memory sinks, or metrics aggregation.

## Local Gate

Run the full gate from the repository root before publishing behavior, docs, or distribution
changes:

```bash
uv lock --check
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run python scripts/generate_provider_docs.py --check
uv run mkdocs build --strict
uv run pytest
uv run --extra harness pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90
bash scripts/test_package.sh
```

For release hardening, also run the dependency audit documented in `docs/src/operations.md`.
