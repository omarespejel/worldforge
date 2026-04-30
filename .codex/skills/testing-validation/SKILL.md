---
name: testing-validation
description: "Use when selecting, running, or fixing WorldForge tests and validation gates, including pytest, coverage, ruff, provider docs drift, MkDocs, package contract failures, CI failures, and release checks."
prerequisites: "uv, pytest, ruff, mkdocs, bash"
---

# Testing And Validation

<purpose>
Choose the smallest credible validation path first, then scale to the relevant release gate.
</purpose>

<context>
WorldForge is Python 3.13-only, uv-native, and coverage-gated at 90 percent with the `harness` extra. CI runs lock check, ruff, generated provider docs, strict MkDocs, pytest, coverage, and package contract. Local tests should reproduce the failing command before broad cleanup.
</context>

<procedure>
1. Identify the changed surface: provider, benchmark, persistence, CLI, TUI, docs, packaging, or public API.
2. Reproduce failures with the narrowest command, usually one `uv run pytest tests/<file>.py -q` or the exact CI command.
3. Fix root cause and add regression coverage for success plus failure paths when behavior changes.
4. Run `uv run ruff check src tests examples scripts` and `uv run ruff format --check src tests examples scripts` before broad pytest.
5. For docs/provider/catalog changes, run `uv run python scripts/generate_provider_docs.py --check` and `uv run mkdocs build --strict`.
6. For public API, package, CLI, or release behavior, run the package contract and coverage gate from `references/release-gate.md`.
7. Report skipped gates explicitly with the blocker.
</procedure>

<patterns>
<do>
- Use `tmp_path` state dirs and `sys.argv` monkeypatching for CLI persistence tests.
- Keep provider payload fixtures under `tests/fixtures/providers/`.
- Use explicit `AssertionError` messages in helpers under `src/worldforge/testing/`.
</do>
<dont>
- Do not lower `--cov-fail-under=90`.
- Do not drop `scripts` from ruff targets.
- Do not replace deterministic tests with live-service requirements.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| Provider docs check fails | Generated README/provider catalog stale | Run generator without `--check`, inspect diff |
| Coverage barely fails | New branch lacks focused tests | Add direct failure-path tests instead of weakening gate |
| Package contract fails only in isolated venv | Missing package data or import path | Inspect `pyproject.toml` hatch settings and `scripts/test_package.sh` |
| MkDocs strict fails | Nav/SUMMARY/docs link drift | Sync `mkdocs.yml`, `docs/src/SUMMARY.md`, and links |
</troubleshooting>

<references>
- `references/release-gate.md`: validation command sets by change type.
- `README.md` and `docs/src/playbooks.md`: canonical direct local gate commands.
- `.github/workflows/ci.yml`: CI quality, test, and package jobs.
- `scripts/test_package.sh`: built package contract.
</references>
