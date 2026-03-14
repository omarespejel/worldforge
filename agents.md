# WorldForge Multi-Agent Orchestration

<applicability>
This project benefits from multi-agent coordination because:
— 6 crates with distinct domains (core types, providers, eval, verify, server, CLI)
— Tasks frequently decompose: implement types → write tests → implement provider → integration test
— Multiple skill domains intersect: Rust systems, ML/AI provider APIs, ZK cryptography, REST APIs
</applicability>

<roles>

| Role           | Model Tier | Responsibility                                        | Boundaries                                   |
|----------------|------------|-------------------------------------------------------|----------------------------------------------|
| Orchestrator   | Frontier   | Decompose tasks, plan implementation order, review     | NEVER writes implementation code directly    |
| Implementer    | Mid-tier   | Write Rust code, implement types/traits, fix bugs      | NEVER makes architectural decisions          |
| Tester         | Mid-tier   | Write tests, run test suite, verify correctness        | NEVER modifies non-test production code      |
| Spec Reviewer  | Frontier   | Cross-reference implementation against SPECIFICATION.md| NEVER modifies code — only reports deviations|

</roles>

<delegation_protocol>
The Orchestrator follows this decision tree:

1. ANALYZE: What crate and module does the task affect?
2. DECOMPOSE: Break into atomic sub-tasks scoped to single files.
3. CLASSIFY:
   — Type implementation (well-defined in spec) → Implementer
   — Test writing → Tester
   — Spec compliance check → Spec Reviewer
   — Architectural question → Orchestrator handles or escalates
4. PLAN: Determine dependency order:
   — types.rs must be done before scene.rs, state.rs, action.rs
   — provider.rs trait must be done before any provider adapter
   — error.rs should be done early (all modules depend on it)
5. DELEGATE: Issue task with context package (see format below).
6. REVIEW: Verify implementation matches spec before marking complete.
</delegation_protocol>

<task_format>
Every delegated task must include:

## Task: [Clear, actionable title]

**Objective**: [What "done" looks like — one sentence]

**Context**:
- Spec section: [SPECIFICATION.md section number and title]
- Files to read: [exact paths]
- Files to modify: [exact paths]
- Dependencies: [what must be implemented first]

**Acceptance criteria**:
- [ ] Types match SPECIFICATION.md definitions
- [ ] All public items have doc comments
- [ ] Unit tests in #[cfg(test)] mod tests {}
- [ ] `cargo test -p [crate]` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo fmt -- --check` passes

**Constraints**:
- Do NOT modify files outside the specified crate
- Do NOT deviate from spec without documenting in ADR.md
- Do NOT add dependencies not in workspace Cargo.toml

**Handoff**: Report completion with list of types implemented and test count.
</task_format>

<parallel_execution>
Safe to parallelize:
— types.rs and error.rs (no mutual dependency)
— Tests for different modules (after types are stable)
— Provider adapters (independent of each other, depend only on core trait)
— Documentation updates

Must serialize:
— types.rs → scene.rs, state.rs (depend on spatial/temporal types)
— provider.rs trait → provider implementations
— core crate → all other crates
— Any Cargo.toml changes (workspace-wide impact)
</parallel_execution>

<implementation_waves>
Recommended implementation sequence:

**Wave 1 — Foundation** (parallelize within wave):
  — types.rs: Tensor, Spatial, Temporal, Media types (Spec §3)
  — error.rs: WorldForgeError enum (Spec §14)

**Wave 2 — Scene & State** (depends on Wave 1):
  — scene.rs: SceneGraph, SceneObject, PhysicsProperties (Spec §5.1)
  — action.rs: Action enum, ActionTranslator trait (Spec §6)
  — state.rs: StateStore trait, StateHistory (Spec §5.2-5.3)

**Wave 3 — Provider & Prediction** (depends on Wave 2):
  — provider.rs: WorldModelProvider trait, ProviderRegistry (Spec §4)
  — prediction.rs: Prediction, PredictionConfig, PhysicsScores (Spec §7)
  — guardrail.rs: Guardrail enum, GuardrailResult (Spec §10)

**Wave 4 — World Orchestration** (depends on Wave 3):
  — world.rs: World struct, plan(), predict() orchestration (Spec §5, §8)
  — lib.rs: WorldForge entry point

**Wave 5 — Outer Crates** (depends on Wave 4):
  — worldforge-providers: Mock provider, then Cosmos adapter
  — worldforge-eval: EvalSuite, EvalScenario (Spec §9)
  — worldforge-cli: CLI commands via clap
  — worldforge-server: REST API
  — worldforge-verify: ZK verification (last)
</implementation_waves>

<escalation>
Escalate to human when:
— Spec is ambiguous or contradictory (e.g., type referenced but never defined)
— A dependency (burn, PyO3, sqlx) has a breaking change or won't compile
— ZK verification design requires cryptographic expertise
— Provider API requires paid access for testing
— Blocked for >30 minutes without progress

Escalation format:
**ESCALATION**: [one-line summary]
**Context**: [what was being done]
**Blocker**: [specific issue]
**Options**: [numbered alternatives with tradeoffs]
**Recommendation**: [which option and why]
</escalation>
