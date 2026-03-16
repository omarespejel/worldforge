<identity>
WorldForge: Unified orchestration layer for world foundation models (WFMs).
Rust core library with Python bindings (PyO3). Pre-alpha — the workspace has implemented core orchestration, providers, eval, verification, CLI, server, and Python bindings, with ongoing depth work tracked against SPECIFICATION.md.
</identity>

<stack>

| Layer        | Technology   | Version  | Notes                                              |
|--------------|--------------|----------|----------------------------------------------------|
| Language     | Rust         | 1.80+    | Edition 2021, workspace resolver v2                |
| Async        | Tokio        | 1.x      | Full features                                      |
| Serialization| Serde + JSON | 1.x      | + MessagePack (rmp-serde) for binary               |
| HTTP         | Reqwest      | 0.12     | rustls-tls, JSON feature                           |
| ML Framework | Burn         | 0.16     | ndarray + wgpu backends                            |
| Database     | SQLx         | 0.8      | SQLite, runtime-tokio-rustls                       |
| Python       | PyO3         | 0.22     | auto-initialize                                    |
| Error        | thiserror    | 2.x      | Typed errors; anyhow for ad-hoc contexts           |
| Logging      | tracing      | 0.1      | + tracing-subscriber 0.3                           |
| CLI          | Clap         | 4.x      | Derive API                                         |
| Testing      | proptest     | 1.x      | Property-based tests for core types                |
| Benchmarks   | Criterion    | 0.5      | Performance benchmarks                             |
| IDs          | UUID         | 1.x      | v4, serde-compatible                               |
| Package mgr  | Cargo        | —        | NEVER use npm/pip for Rust code                    |

</stack>

<structure>
```
worldforge/
├── Cargo.toml                  # Workspace root — defines all members and shared deps
├── SPECIFICATION.md            # Detailed technical spec — the source of truth for all types and APIs
├── CONTRIBUTING.md             # Dev setup and contribution guide
├── architecture/
│   └── ADR.md                  # Architecture Decision Records (8 accepted ADRs)
├── business/
│   └── BUSINESS_PLAN.md        # Business strategy (read-only context)
├── go-to-market/
│   └── 90_DAY_SPRINT.md        # Launch plan (read-only context)
├── research/
│   └── MARKET_INTELLIGENCE.md  # Market research (read-only context)
├── crates/
│   ├── worldforge-core/        # Core library: types, traits, state management
│   │   └── src/
│   │       ├── lib.rs          # Crate root — WorldForge struct
│   │       ├── types.rs        # Tensor, spatial, temporal, media types
│   │       ├── world.rs        # World orchestration + planning
│   │       ├── action.rs       # Action type system
│   │       ├── prediction.rs   # Prediction engine + planning types
│   │       ├── provider.rs     # WorldModelProvider trait + registry
│   │       ├── scene.rs        # Scene graph
│   │       ├── guardrail.rs    # Safety constraints
│   │       ├── state.rs        # State persistence
│   │       └── error.rs        # WorldForgeError enum
│   ├── worldforge-providers/   # Provider adapters: Cosmos, GWM, JEPA, Genie
│   │   └── src/lib.rs          # Auto-detection + adapter exports
│   ├── worldforge-eval/        # Evaluation framework
│   │   └── src/lib.rs          # Built-in suites + reports
│   ├── worldforge-verify/      # ZK verification — optional module
│   │   └── src/lib.rs          # Mock verifier + proof types
│   ├── worldforge-server/      # REST API server
│   │   └── src/lib.rs          # HTTP routing + persistence
│   └── worldforge-cli/         # CLI tool
│       └── src/lib.rs          # Command parsing + orchestration
└── .codex/skills/              # Agentic skill files
```
</structure>

<commands>

| Task             | Command                              | Notes                                    |
|------------------|--------------------------------------|------------------------------------------|
| Build all        | `cargo build`                        | Workspace build, all crates              |
| Build one crate  | `cargo build -p worldforge-core`     | Replace crate name as needed             |
| Test all         | `cargo test`                         | Runs all workspace tests                 |
| Test one crate   | `cargo test -p worldforge-core`      | Replace crate name as needed             |
| Test specific    | `cargo test -p worldforge-core test_name` | Run a single test                   |
| Clippy (lint)    | `cargo clippy -- -D warnings`        | Treat warnings as errors                 |
| Format           | `cargo fmt`                          | Rustfmt — run before every commit        |
| Format check     | `cargo fmt -- --check`               | CI mode — no modifications               |
| Doc generation   | `cargo doc --no-deps --open`         | Generate and view docs                   |
| Bench            | `cargo bench -p worldforge-core`     | Criterion benchmarks                     |

</commands>

<conventions>
<code_style>
  Naming: snake_case for functions/variables/modules, PascalCase for types/traits/enums, SCREAMING_SNAKE for constants.
  Files: snake_case.rs — one module per file, match module name to filename.
  Imports: Group as std → external crates → workspace crates → local modules. Separate groups with blank line.
  Visibility: Prefer pub(crate) over pub unless the item is part of the public API.
  Error handling: Use thiserror for typed errors (WorldForgeError enum). Use anyhow only in CLI/server binaries, never in library code. No .unwrap() in library code — always propagate with `?`.
  Async: All provider interactions are async (tokio). Use async-trait for trait methods. Internal pure logic is sync.
  Documentation: All public items must have `///` doc comments. Include `# Examples` for complex APIs.
</code_style>

<patterns>
  <do>
    — Implement types exactly as specified in SPECIFICATION.md sections 3-11.
    — Use #[derive(Debug, Clone, Serialize, Deserialize)] on all data types.
    — Use builder pattern for complex config types (PredictionConfig, PlanRequest).
    — Use #[cfg(test)] mod tests {} in each source file for unit tests.
    — Use proptest for property-based testing of core types (serialization roundtrip, invariant checking).
    — Place integration tests in tests/ directory at crate root.
    — Return Result<T, WorldForgeError> from all fallible operations.
    — Use tracing::instrument on async functions for observability.
    — Keep provider adapters in separate submodules (cosmos.rs, runway.rs, jepa.rs, genie.rs).
  </do>
  <dont>
    — Don't deviate from SPECIFICATION.md type definitions without documenting the change in ADR.md.
    — Don't use .unwrap(), .expect() in library code — use proper error propagation.
    — Don't add provider-specific types to worldforge-core — keep provider details in worldforge-providers.
    — Don't use println! — use tracing::{info, debug, warn, error}.
    — Don't add dependencies without checking if they're already in [workspace.dependencies].
    — Don't implement ZK features outside worldforge-verify crate.
  </dont>
</patterns>

<commit_conventions>
  Format: type(scope): description
  Types: feat, fix, refactor, test, docs, chore, perf
  Scopes: core, providers, eval, verify, server, cli, workspace
  Examples:
    feat(core): implement WorldState and SceneGraph types
    test(core): add proptest roundtrip tests for spatial types
    fix(providers): handle timeout in Cosmos API calls
    docs(workspace): update SPECIFICATION.md with new guardrail types
</commit_conventions>
</conventions>

<workflows>
<implement_types>
  1. Read the relevant section of SPECIFICATION.md for the type definitions.
  2. Implement types in the appropriate file in worldforge-core/src/.
  3. Add #[derive(Debug, Clone, Serialize, Deserialize)] and any other needed derives.
  4. Add doc comments on every public item.
  5. Write unit tests in #[cfg(test)] mod tests {} at bottom of file.
  6. Add proptest strategies for roundtrip serialization.
  7. Run `cargo test -p worldforge-core` — all must pass.
  8. Run `cargo clippy -- -D warnings` — zero warnings.
  9. Run `cargo fmt`.
  10. Commit: feat(core): implement [type names] per specification.
</implement_types>

<implement_provider>
  1. Read SPECIFICATION.md sections 4 and 13 for provider trait and mapping.
  2. Create provider module in worldforge-providers/src/ (e.g., cosmos.rs).
  3. Implement WorldModelProvider trait.
  4. Implement ActionTranslator for provider-specific action mapping.
  5. Add integration tests (can use mock HTTP responses).
  6. Run full test suite: `cargo test`.
  7. Commit: feat(providers): implement [provider name] adapter.
</implement_provider>

<add_feature>
  1. Check SPECIFICATION.md for the feature's design.
  2. Identify which crate(s) need changes.
  3. Implement in the correct crate, following existing patterns.
  4. Write tests (unit + integration as appropriate).
  5. Run `cargo test && cargo clippy -- -D warnings && cargo fmt`.
  6. Self-review: no secrets, no debug prints, no TODO without tracking issue.
  7. Commit with conventional format.
</add_feature>

<fix_bug>
  1. Write a failing test that reproduces the bug.
  2. Fix the code.
  3. Verify the test passes.
  4. Run full suite: `cargo test`.
  5. Commit: fix(scope): description of what was broken and why.
</fix_bug>
</workflows>

<boundaries>
<forbidden>
  DO NOT modify under any circumstances:
  — .env, .env.* (credentials, API keys)
  — Any file containing API keys, secrets, or tokens
  — LICENSE file
</forbidden>

<gated>
  Modify ONLY with explicit human approval:
  — Cargo.toml (workspace root) — dependency changes affect all crates
  — SPECIFICATION.md — source of truth for the entire system
  — architecture/ADR.md — architectural decisions
  — business/ and go-to-market/ — business documents
</gated>

<autonomous>
  Safe to modify without approval:
  — All src/ files in crates/ — this is where implementation happens
  — Test files
  — CONTRIBUTING.md
  — CLAUDE.md, agents.md, .codex/skills/ — agentic context
</autonomous>

<safety_checks>
  Before ANY destructive operation (delete file, drop table, reset state):
  1. State what you're about to do
  2. State what could go wrong
  3. Wait for confirmation
</safety_checks>
</boundaries>

<troubleshooting>
<known_issues>

| Symptom                              | Cause                          | Fix                                    |
|--------------------------------------|--------------------------------|----------------------------------------|
| `cargo build` fails on burn          | Missing GPU drivers for wgpu   | Use `--no-default-features` or ndarray backend only |
| PyO3 build fails                     | Python dev headers missing     | Install python3-dev / python3-devel    |
| SQLx compile error                   | Missing SQLite dev libs        | Install libsqlite3-dev                 |
| `unresolved import` in IDE           | Workspace not detected         | Open project from workspace root       |

</known_issues>

<recovery_patterns>
  When stuck, follow this cascade:
  1. Read the error message fully — Rust errors are verbose but precise.
  2. Check if the type/function exists in SPECIFICATION.md.
  3. Run `cargo clean && cargo build` (stale build artifacts).
  4. Check Cargo.toml for missing dependencies.
  5. If still stuck, state the problem clearly and ask for help.
</recovery_patterns>
</troubleshooting>

<architecture_context>
Key decisions (from ADR.md) — do not re-litigate:
— ADR-001: Rust core + Python bindings via PyO3 (performance + ecosystem reach)
— ADR-002: Provider trait pattern (compile-time interface verification)
— ADR-003: Scene graph for world state (industry standard, provider-agnostic)
— ADR-004: ZK verification as optional separate crate (keeps core lightweight)
— ADR-005: Pluggable state stores (file, SQLite, Redis, S3)
— ADR-006: Evaluation as first-class crate (drives adoption as reference standard)
— ADR-007: Apache 2.0 core, proprietary cloud (open-core model)
— ADR-008: CLI-first development (rapid experimentation, natural integration test)
</architecture_context>

<implementation_priority>
The codebase is pre-alpha with stub modules. Implementation order should be:
1. worldforge-core: types.rs → error.rs → scene.rs → state.rs → action.rs → provider.rs → prediction.rs → guardrail.rs → world.rs → lib.rs
2. worldforge-providers: Start with a mock provider, then Cosmos adapter
3. worldforge-eval: After core types are stable
4. worldforge-cli: After core + at least one provider
5. worldforge-server: After CLI proves the API
6. worldforge-verify: Last — depends on stable core + JEPA provider
</implementation_priority>

<skills>
Modular skills are in .codex/skills/ (symlinked at .claude/skills/ and .agents/skills/).

Available skills:
— rust-development.md: Rust patterns, idioms, and workspace conventions for this project
— provider-integration.md: How to implement a WorldModelProvider adapter
— testing-strategy.md: Testing approach — unit, property-based, integration, benchmarks
— specification-driven-dev.md: How to translate SPECIFICATION.md into implementation
</skills>
