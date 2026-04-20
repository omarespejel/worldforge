# Skill Registry

Last updated: 2026-04-20

<inventory>
| Skill | File | Triggers | Priority |
| --- | --- | --- | --- |
| Provider Adapter Development | `provider-adapter-development.md` | provider, adapter, capability, catalog, scaffold, remote, parser, fixture | Core |
| Testing And Validation | `testing-validation.md` | test, coverage, CI, lint, docs drift, package, release gate | Core |
| Evaluation And Benchmarking | `evaluation-benchmarking.md` | eval, benchmark, report, latency, throughput, claim, suite | Core |
| Optional Runtime Smokes | `optional-runtime-smokes.md` | LeWorldModel, stable-worldmodel, GR00T, LeRobot, torch, checkpoint, live smoke | Core |
| Persistence And State | `persistence-state.md` | world ID, `.worldforge`, save, load, import, export, history, JSON state | Core |
</inventory>

<context_audit>
| Context surface | Status | Finding | Action |
| --- | --- | --- | --- |
| `AGENTS.md` | Keep | Accurate, detailed repo guide with architecture map, commands, constraints, gotchas | Leave as detailed secondary context |
| `CLAUDE.md` | Add | Missing compact primary cognitive context | Added machine-first context with boundaries and workflow routing |
| `.codex/skills/` | Add | Missing progressive-disclosure skills | Added core WorldForge skills |
| `.claude/skills`, `.agents/skills` | Add | Missing cross-tool discovery links | Create symlinks to `.codex/skills` |
| `agents.md` | Not separate | `agents.md` resolves to existing `AGENTS.md` on this checkout | Do not create a case-colliding file; use `AGENTS.md` plus skill-scoped delegation guidance |
</context_audit>

<skill_gap_analysis>
Covered high-priority gaps:
- provider adapter development: highest impact and frequent; provider boundaries are the main project risk.
- testing/validation: highest frequency; CI, docs generation, coverage, and package contract are release gates.
- evaluation/benchmarking: high impact; prevents overstated claims from deterministic suites and one-off benchmark runs.
- optional runtime smokes: high risk; prevents base dependency bloat and mislabeling injected demos as neural inference.
- persistence/state: high impact; world IDs and history validation protect local JSON coherence.

Recommended lower-priority future skills:
- security-review.md: sanitized provider errors, dependency audit triage, credential leakage checks.
- release-management.md: changelog, tags, PyPI release, GitHub release, rollback evidence.
- documentation-maintenance.md: README/docs routing, generated block drift, provider page consistency.
- refactoring-playbook.md: safe extraction patterns across `models.py`, `framework.py`, providers, and tests.
</skill_gap_analysis>

<activation>
Read only the skill needed for the current task. Do not load all skills by default.
</activation>
