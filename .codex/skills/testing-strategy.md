---
name: testing-strategy
description: Testing approach for WorldForge — unit tests, property-based tests (proptest), integration tests, and benchmarks (criterion). Activate when writing tests, setting up test infrastructure, debugging test failures, or evaluating test coverage. Also activate when discussing mocking, test fixtures, or CI test configuration.
prerequisites: cargo, proptest (in workspace deps), criterion (in workspace deps)
---

# Testing Strategy

<purpose>
Defines how to test WorldForge code across all crates. Covers unit tests, property-based tests with proptest, integration tests, and performance benchmarks with criterion.
</purpose>

<context>
— proptest 1.x for property-based testing (in workspace deps).
— criterion 0.5 for benchmarks (in workspace deps).
— No external test runner — standard `cargo test`.
— No mock framework specified — use hand-written mocks or conditional compilation.
— Pre-alpha: focus on core type correctness before integration tests.
</context>

<procedure>
1. For every new type: write serialization roundtrip test (JSON and MessagePack).
2. For types with invariants: write proptest strategies to verify invariants hold for arbitrary inputs.
3. For async provider code: write tests with mock HTTP responses (no live API dependency).
4. For planning/guardrail logic: write scenario-based integration tests.
5. Run `cargo test` after every change — all tests must pass.
6. Run `cargo test -- --nocapture` to see tracing output during tests.
</procedure>

<patterns>
<do>
  — Place unit tests in `#[cfg(test)] mod tests {}` at bottom of each source file.
  — Place integration tests in `crates/{crate}/tests/` directory.
  — Use `proptest! {}` macro for property-based tests of core types.
  — Test error paths — verify correct WorldForgeError variants are returned.
  — Use `#[tokio::test]` for async test functions.
  — Name tests descriptively: `test_{what}_{condition}_{expected}`.
</do>
<dont>
  — Don't use .unwrap() in test setup without a descriptive message — use .expect("reason").
  — Don't write tests that depend on external APIs — use mocks.
  — Don't skip testing error cases — they're often where bugs hide.
  — Don't write tests that depend on execution order.
</dont>
</patterns>

<examples>
Example: Proptest roundtrip for spatial types

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    prop_compose! {
        fn arb_position()(
            x in -1000.0f32..1000.0,
            y in -1000.0f32..1000.0,
            z in -1000.0f32..1000.0,
        ) -> Position {
            Position { x, y, z }
        }
    }

    proptest! {
        #[test]
        fn position_json_roundtrip(pos in arb_position()) {
            let json = serde_json::to_string(&pos).unwrap();
            let decoded: Position = serde_json::from_str(&json).unwrap();
            prop_assert!((pos.x - decoded.x).abs() < f32::EPSILON);
            prop_assert!((pos.y - decoded.y).abs() < f32::EPSILON);
            prop_assert!((pos.z - decoded.z).abs() < f32::EPSILON);
        }
    }
}
```

Example: Async provider test with mock

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn mock_provider() -> MockProvider {
        MockProvider {
            predictions: vec![/* pre-configured responses */],
        }
    }

    #[tokio::test]
    async fn test_predict_returns_valid_state() {
        let provider = mock_provider();
        let state = WorldState::default();
        let action = Action::Move { target: Position::origin(), speed: 1.0 };
        let config = PredictionConfig::default();

        let result = provider.predict(&state, &action, &config).await;
        assert!(result.is_ok());

        let prediction = result.unwrap();
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }
}
```
</examples>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| proptest timeout | Strategy generates too many values | Narrow ranges, reduce `PROPTEST_CASES` |
| async test hangs | Missing tokio runtime | Use `#[tokio::test]` not `#[test]` |
| test passes locally, fails in CI | Non-deterministic behavior | Check for time-dependent or order-dependent logic |
| criterion benchmark won't compile | Missing bench harness config | Add `[[bench]]` section to Cargo.toml with `harness = false` |

</troubleshooting>

<references>
— Cargo.toml: proptest and criterion in workspace deps
— CONTRIBUTING.md: test commands
— crates/worldforge-core/src/: files needing unit tests
</references>
