---
name: public-docs-release
description: "Use for WorldForge README, docs, changelog, generated provider docs, MkDocs navigation, version/release metadata, public positioning, and release or publish readiness checks."
prerequisites: "uv, mkdocs, ruff"
---

# Public Docs And Release

<purpose>
Keep public-facing text, generated docs, package metadata, and release gates synchronized.
</purpose>

<context>
WorldForge's public front face must stay serious and precise. The strongest current positioning is an integration layer for physical-AI world-model workflows; typed/local-first details support that claim but should not become hype. Public behavior changes usually touch README, docs, changelog, AGENTS, CLAUDE, CLI help snapshots, and generated provider catalog blocks together.
</context>

<procedure>
1. Identify whether the change affects public API, CLI, provider surface, runtime workflow, release process, or only wording.
2. Edit source docs, not generated blocks. For provider tables, change provider metadata/catalog and run `uv run python scripts/generate_provider_docs.py`.
3. Keep `mkdocs.yml` nav synchronized with `docs/src/SUMMARY.md` when adding or removing docs pages.
4. Update `CHANGELOG.md` for user-visible behavior changes.
5. For version/release changes, align `pyproject.toml`, `uv.lock`, README badges/text, `CITATION.cff`, changelog, docs, and release notes.
6. Run docs check and the relevant validation gate from `testing-validation`.
</procedure>

<patterns>
<do>
- Keep README concise and route detailed operations to `docs/src/playbooks.md`.
- State command, expected success signal, and first triage step for operational docs.
- Preserve claim boundaries around mock providers, deterministic suites, and optional runtimes.
</do>
<dont>
- Do not duplicate generated provider catalog edits by hand.
- Do not mention agent/tool branding in public contribution artifacts.
- Do not present scaffold providers as real integrations or deterministic evaluations as fidelity evidence.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| MkDocs strict warning | Bad link/nav/SUMMARY drift | Fix source page and sync nav/SUMMARY |
| Provider table changes disappear | Edited generated block by hand | Change catalog/profile metadata and regenerate |
| CLI snapshot fails after copy edit | Help text changed | Update `tests/test_cli_help_snapshots.py` intentionally |
</troubleshooting>

<references>
- `README.md`: front-door positioning and quickstart.
- `docs/src/playbooks.md`: operational runbooks.
- `docs/src/SUMMARY.md` and `mkdocs.yml`: docs navigation contract.
- `scripts/generate_provider_docs.py`: generated provider docs.
- `CHANGELOG.md`: public change log.
</references>
