# Skill Registry

Last updated: 2026-04-21

Project-local agent skills for WorldForge. Each skill follows the
Claude-Code progressive-disclosure layout: a `<name>/SKILL.md` is loaded when
the skill triggers; bundled `references/` files are read on demand from
inside the skill body.

## Inventory

| Skill | Path | What it owns |
| --- | --- | --- |
| Provider Adapter Development | `provider-adapter-development/SKILL.md` | Adapter classification, capability truthfulness, scaffold → promote, fixtures, contract tests, generated provider docs. |
| Testing And Validation | `testing-validation/SKILL.md` | Smallest reliable test loop, full release gate, lint / format / coverage / docs-drift / package-contract triage. |
| Evaluation And Benchmarking | `evaluation-benchmarking/SKILL.md` | Deterministic evaluation suites, benchmark harness, renderers, claim hygiene, capability-gated runs. |
| Optional Runtime Smokes | `optional-runtime-smokes/SKILL.md` | LeWorldModel / GR00T / LeRobot smokes, host-owned dependencies, demo-vs-real labelling. |
| Persistence And State | `persistence-state/SKILL.md` | World IDs, local JSON persistence, save / load / import / export / fork, history validation. |

## Disambiguation — which skill for what

Many tasks touch multiple areas. Use this table to pick the *primary* skill.
Secondary skills can be loaded as the task crosses boundaries.

| If the user is asking about… | Load |
| --- | --- |
| "this provider's capability flag is wrong" / scaffold / parser / fixture | `provider-adapter-development` |
| "tests / CI failed" / "what should I run before merging" / coverage / package contract | `testing-validation` |
| "run / publish / interpret an eval or benchmark number" | `evaluation-benchmarking` |
| "live LeWorldModel / GR00T / LeRobot run" / "is this real or injected?" | `optional-runtime-smokes` |
| "save / load / export / import a world" / `.worldforge/` / history | `persistence-state` |
| Adding a database / lock file / migration / multi-writer persistence | `persistence-state` (the answer is *stop and ask*) |
| Adding torch / CUDA / robot SDK to base deps | `optional-runtime-smokes` (the answer is *stop and ask*) |
| Lowering coverage / dropping a CI gate | `testing-validation` (the answer is *stop and ask*) |

## Activation rules

- Read only the skill needed for the current task. Do not preload the set.
- Read a skill's `references/<file>.md` only when its `SKILL.md` directs you to.
- If a task crosses skills, load them one at a time as you reach each boundary
  rather than loading everything up front — context is the scarce resource.

## Skill format (SOTA)

Each `SKILL.md` follows the Anthropic Claude-Code skill convention:

- YAML frontmatter with `name` + `description` (description is the trigger;
  it includes both positive triggers and an explicit negative-trigger section
  in the body).
- A `## Fast start` block leading with the 1–3 commands that solve the most
  common case for the skill.
- A `## Why this skill exists` section that names the failure modes the skill
  defends against — explanation beats prohibition.
- A `## The procedure` section in imperative voice with annotated rationale.
- A `## Activation cues` section with positive triggers and explicit
  *do-not-trigger* cases (the most common cause of skill mis-routing is
  shared keywords with an adjacent skill).
- A `## Stop and ask the user` section that maps to the project's `<gated>`
  list in `CLAUDE.md`.
- A `## Patterns` Do/Don't block kept short; rationale lives above it.
- A `## Troubleshooting` table for symptom → likely cause → first fix.
- A `## References` section pointing at code, docs, tests, and any bundled
  `references/` files.

Bundled `references/` files exist only where a SKILL.md genuinely reuses
dense lookup data; keep them lean.

## Future skills (not yet authored)

Tracked here so we don't re-rediscover the gap:

- `security-review/` — sanitised provider errors, dependency audit triage, credential leakage checks.
- `release-management/` — changelog, tags, PyPI release, GitHub release, rollback evidence.
- `documentation-maintenance/` — README / docs routing, generated-block drift, provider page consistency.
- `refactoring-playbook/` — safe extraction patterns across `models.py`, `framework.py`, providers, and tests.

Author each on demand, not speculatively.
