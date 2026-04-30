# WorldForge Skill Registry

Last updated: 2026-04-30

Canonical location: `.codex/skills/`.
Compatibility links: `.claude/skills -> ../.codex/skills`, `.agents/skills -> ../.codex/skills`.

| Skill | File | Triggers | Priority |
| --- | --- | --- | --- |
| Provider Adapter Development | `provider-adapter-development/SKILL.md` | provider, adapter, capability, catalog, scaffold, Cosmos, Runway, LeWorldModel, GR00T, LeRobot, JEPA-WMS | Core |
| Testing And Validation | `testing-validation/SKILL.md` | test, pytest, coverage, ruff, CI, package, validation, failing check | Core |
| Evaluation And Benchmarking | `evaluation-benchmarking/SKILL.md` | eval, benchmark, report, budget, input fixture, claim, throughput, latency | Core |
| Optional Runtime Smokes | `optional-runtime-smokes/SKILL.md` | smoke, LeWorldModel, GR00T, LeRobot, checkpoint, PushT, live runtime | Core |
| Persistence And State | `persistence-state/SKILL.md` | world CLI, state dir, import, export, history, world id, snapshot | Core |
| TUI Development | `tui-development/SKILL.md` | harness, TheWorldHarness, Textual, TUI, screen, flow, screenshot | Core |
| Public Docs And Release | `public-docs-release/SKILL.md` | README, docs, changelog, release, mkdocs, provider docs, version | Core |

## Skill Gap Analysis

Covered high-frequency domains:
- Provider capability truthfulness and adapter promotion.
- Local validation, package contract, docs drift, and release gates.
- Benchmark/evaluation claim boundaries.
- Host-owned optional runtime smokes.
- Local JSON persistence and history contracts.
- TheWorldHarness TUI boundaries.
- Public docs and release-surface synchronization.

Recommended future skills:
- `security-observability-review`: provider-event sanitization, signed URL handling, dependency audit triage.
- `refactoring-playbook`: large cross-module refactors, shared helper extraction, API compatibility review.
- `dependency-release-management`: version bumps, lockfile refresh, tag/release workflow, PyPI publish risk review.

