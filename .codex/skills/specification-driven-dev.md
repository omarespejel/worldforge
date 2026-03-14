---
name: specification-driven-dev
description: How to translate SPECIFICATION.md into working Rust implementation. Activate when implementing types from the spec, cross-referencing spec sections, or resolving ambiguities between spec and code. Also activate when the spec needs updating after implementation discoveries.
prerequisites: SPECIFICATION.md read, worldforge-core crate
---

# Specification-Driven Development

<purpose>
WorldForge's implementation is driven by SPECIFICATION.md, which defines all types, traits, APIs, and behaviors. This skill guides the process of faithfully translating spec sections into Rust code while handling ambiguities and edge cases.
</purpose>

<context>
— SPECIFICATION.md is the source of truth (32K+ words, 15 sections).
— All modules in worldforge-core are currently stubs referencing the spec.
— Types in the spec use Rust syntax — implement them as-is unless there's a compile-time reason to deviate.
— Any deviation from spec must be documented in architecture/ADR.md.
</context>

<procedure>
1. Identify the spec section for the module you're implementing:
   — types.rs → Section 3 (Core Type System: Tensor, Spatial, Temporal, Media)
   — error.rs → Section 14 (Error Handling)
   — scene.rs → Section 5.1 (Scene Graph, SceneObject, PhysicsProperties, SpatialRelationship)
   — state.rs → Section 5.2-5.3 (State Persistence, State History)
   — action.rs → Section 6 (Action System)
   — provider.rs → Section 4 (Provider Trait, Registry, Capabilities)
   — prediction.rs → Section 7 (Prediction Engine)
   — guardrail.rs → Section 10 (Guardrails & Safety)
   — world.rs → Section 5.1 (WorldState) + orchestration logic
2. Read the entire relevant section — don't cherry-pick.
3. Copy type definitions from spec into Rust code.
4. Add necessary derives: Debug, Clone, Serialize, Deserialize.
5. Resolve spec ambiguities:
   — If spec uses `Box<dyn Trait>`, consider if a generic parameter works better.
   — If spec uses `String` for IDs, consider if a newtype wrapper is better.
   — If spec has `Option<Vec<T>>`, consider if `Vec<T>` (empty = none) is simpler.
6. Document any deviation with a comment: `// DEVIATION from spec: [reason]`.
7. Write tests that validate the type matches spec behavior.
8. Cross-reference with other spec sections for consistency.
</procedure>

<patterns>
<do>
  — Implement types exactly as specified unless there's a Rust-specific reason to deviate.
  — Keep spec section numbers as comments in code: `// Spec §3.2: Spatial Types`.
  — Implement all enum variants listed in the spec — don't skip "for later".
  — Use the same field names as the spec for API consistency.
  — When the spec defines a trait, implement it as a Rust trait with async-trait if async.
</do>
<dont>
  — Don't rename spec types without documenting why (breaks API contracts).
  — Don't add fields/variants not in the spec (scope creep).
  — Don't implement "improvements" over the spec — file an issue instead.
  — Don't skip spec sections because they seem complex — implement stubs at minimum.
  — Don't modify SPECIFICATION.md without approval — it's a gated file.
</dont>
</patterns>

<examples>
Example: Translating spec §3.2 (Spatial Types)

Spec says:
```rust
pub struct Position { pub x: f32, pub y: f32, pub z: f32 }
```

Implementation:
```rust
// Spec §3.2: Spatial Types

/// 3D position in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position {
    /// Origin point (0, 0, 0).
    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
}
```

Note: Added `Copy` (small struct), `PartialEq` (testing), constructor. These are additive, not deviations.
</examples>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| Spec type doesn't compile | Missing import or dependency | Check workspace deps, add to crate Cargo.toml |
| Spec uses `Box<dyn CostFn>` but trait isn't defined | Spec references forward-declared types | Implement a placeholder trait, mark with TODO |
| Circular dependency between spec types | Types in different modules reference each other | Use the types.rs module as the shared type foundation |
| Spec enum has too many variants for one file | Large enum like Action | Keep in one file — split only if >500 lines |

</troubleshooting>

<references>
— SPECIFICATION.md: all type definitions and API contracts
— architecture/ADR.md: rationale for deviations
— crates/worldforge-core/src/: implementation targets
</references>
