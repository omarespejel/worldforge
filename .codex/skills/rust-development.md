---
name: rust-development
description: Rust coding patterns, idioms, and workspace conventions for WorldForge. Activate when implementing any Rust code, adding types, writing modules, or working with the Cargo workspace. Also activate for refactoring, fixing clippy warnings, or reviewing Rust code.
prerequisites: Rust 1.80+, cargo
---

# Rust Development

<purpose>
Guides implementation of Rust code within the WorldForge workspace. Covers type design, error handling, async patterns, and workspace conventions.
</purpose>

<context>
— Workspace with 6 crates in crates/ directory.
— All shared dependencies pinned in root Cargo.toml [workspace.dependencies].
— Core crate (worldforge-core) defines all public types and traits.
— Other crates depend on worldforge-core.
— Async runtime: Tokio with full features.
— Error pattern: thiserror for library errors, anyhow only in binaries.
</context>

<procedure>
1. Identify which crate the code belongs in (core types → worldforge-core, provider code → worldforge-providers, etc.).
2. Check if types/traits are defined in SPECIFICATION.md — implement to match.
3. Add derives: `#[derive(Debug, Clone, Serialize, Deserialize)]` minimum for data types.
4. Add `///` doc comments on all public items.
5. Use `pub(crate)` for internal-only items.
6. Write unit tests in `#[cfg(test)] mod tests {}` at bottom of file.
7. Run `cargo clippy -- -D warnings` and `cargo fmt`.
</procedure>

<patterns>
<do>
  — Use `Result<T, WorldForgeError>` for all fallible operations in library code.
  — Use `?` operator for error propagation — avoid explicit match on Result when possible.
  — Use `#[instrument]` from tracing on async functions.
  — Use workspace dependency references: `dep = { workspace = true }` in crate Cargo.toml.
  — Use `impl From<ExternalError> for WorldForgeError` for error conversion.
  — Use builder pattern for config structs with many optional fields.
  — Group imports: std → external → workspace → local, separated by blank lines.
</do>
<dont>
  — Don't use `.unwrap()` or `.expect()` in library code — propagate errors.
  — Don't use `println!` — use `tracing::{info, debug, warn, error}`.
  — Don't add deps to individual crate Cargo.toml without adding to workspace first.
  — Don't use `Box<dyn Error>` — use the typed WorldForgeError enum.
  — Don't make fields `pub` by default — use accessor methods for encapsulation.
</dont>
</patterns>

<examples>
Example: Adding a new type to worldforge-core

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a world instance.
pub type WorldId = Uuid;

/// Complete state of a simulated world at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    /// Unique world identifier.
    pub id: WorldId,
    /// Current simulation time.
    pub time: SimTime,
    /// Scene graph with all objects.
    pub scene: SceneGraph,
    /// History of previous states.
    pub history: StateHistory,
    /// Additional metadata.
    pub metadata: WorldMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_state_serialization_roundtrip() {
        let state = WorldState { /* ... */ };
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: WorldState = serde_json::from_str(&json).unwrap();
        assert_eq!(state.id, deserialized.id);
    }
}
```
</examples>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `workspace dependency not found` | Dep not in root Cargo.toml | Add to `[workspace.dependencies]` first |
| `trait bound not satisfied: Send` | Holding non-Send type across await | Use `Arc<Mutex<T>>` or restructure to drop before await |
| `lifetime error on async trait` | Missing `async-trait` macro | Add `#[async_trait]` to trait and impl |
| `unused import` warning | Over-importing | Remove unused, run `cargo fix` |

</troubleshooting>

<references>
— Cargo.toml (workspace root): workspace dependency definitions
— crates/worldforge-core/Cargo.toml: core crate dependencies
— SPECIFICATION.md: type definitions and API contracts
— architecture/ADR.md: rationale for architectural decisions
</references>
