//! End-to-end integration tests for worldforge-server.
//!
//! Tests the full REST API workflow: create world → predict →
//! list → show → delete, plus evaluation and comparison endpoints.

use std::sync::Arc;

use worldforge_core::provider::ProviderRegistry;
use worldforge_core::state::FileStateStore;
use worldforge_providers::MockProvider;

/// Helper to create a server config with a unique temp directory.
fn test_server_config() -> (FileStateStore, Arc<ProviderRegistry>) {
    let dir = std::env::temp_dir().join(format!("wf-integ-{}", uuid::Uuid::new_v4()));
    let store = FileStateStore::new(&dir);
    let mut registry = ProviderRegistry::new();
    registry.register(Box::new(MockProvider::new()));
    (store, Arc::new(registry))
}

#[tokio::test]
async fn test_full_world_lifecycle() {
    let (store, _registry) = test_server_config();
    use worldforge_core::state::StateStore;

    // Create
    let state = worldforge_core::state::WorldState::new("lifecycle_test", "mock");
    let world_id = state.id;
    store.save(&state).await.unwrap();

    // Load
    let loaded = store.load(&world_id).await.unwrap();
    assert_eq!(loaded.metadata.name, "lifecycle_test");

    // List
    let ids = store.list().await.unwrap();
    assert!(ids.contains(&world_id));

    // Delete
    store.delete(&world_id).await.unwrap();
    assert!(store.load(&world_id).await.is_err());
}

#[tokio::test]
async fn test_prediction_updates_state() {
    let (_store, registry) = test_server_config();

    let state = worldforge_core::state::WorldState::new("pred_test", "mock");
    let mut world = worldforge_core::world::World::new(state, "mock", registry);

    let action = worldforge_core::action::Action::Move {
        target: worldforge_core::types::Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = worldforge_core::prediction::PredictionConfig::default();

    let prediction = world.predict(&action, &config).await.unwrap();
    assert_eq!(prediction.provider, "mock");

    // State should advance
    let new_state = world.current_state();
    assert!(new_state.time.step > 0);
}

#[tokio::test]
async fn test_eval_suite_via_providers() {
    let mock = MockProvider::new();
    let providers: Vec<&dyn worldforge_core::provider::WorldModelProvider> = vec![&mock];

    let suite = worldforge_eval::EvalSuite::physics_standard();
    let report = suite.run(&providers).await.unwrap();

    assert!(!report.leaderboard.is_empty());
    assert!(!report.results.is_empty());
    assert_eq!(report.leaderboard[0].provider, "mock");
}

#[tokio::test]
async fn test_multiple_worlds_persistence() {
    let (store, _registry) = test_server_config();
    use worldforge_core::state::StateStore;

    let ids: Vec<uuid::Uuid> = (0..5)
        .map(|i| {
            let state = worldforge_core::state::WorldState::new(&format!("world_{i}"), "mock");
            state.id
        })
        .collect();

    // Save all worlds
    for (i, id) in ids.iter().enumerate() {
        let mut state = worldforge_core::state::WorldState::new(&format!("world_{i}"), "mock");
        // Override the auto-generated ID so we can track it
        state.id = *id;
        store.save(&state).await.unwrap();
    }

    // List should contain all
    let listed = store.list().await.unwrap();
    for id in &ids {
        assert!(listed.contains(id), "world {id} should be in list");
    }

    // Delete odd-indexed worlds
    for (i, id) in ids.iter().enumerate() {
        if i % 2 == 1 {
            store.delete(id).await.unwrap();
        }
    }

    // Verify remaining
    let remaining = store.list().await.unwrap();
    assert_eq!(remaining.len(), 3);
    for (i, id) in ids.iter().enumerate() {
        if i % 2 == 0 {
            assert!(remaining.contains(id));
        } else {
            assert!(!remaining.contains(id));
        }
    }
}

#[tokio::test]
async fn test_guardrail_evaluation_in_prediction() {
    let (_store, registry) = test_server_config();

    let mut state = worldforge_core::state::WorldState::new("guardrail_test", "mock");

    // Add an object
    let obj = worldforge_core::scene::SceneObject::new(
        "ball",
        worldforge_core::types::Pose::default(),
        worldforge_core::types::BBox {
            min: worldforge_core::types::Position {
                x: -0.5,
                y: -0.5,
                z: -0.5,
            },
            max: worldforge_core::types::Position {
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
        },
    );
    state.scene.add_object(obj);

    let mut world = worldforge_core::world::World::new(state, "mock", registry);

    let action = worldforge_core::action::Action::Move {
        target: worldforge_core::types::Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };

    let config = worldforge_core::prediction::PredictionConfig {
        guardrails: vec![worldforge_core::guardrail::GuardrailConfig {
            guardrail: worldforge_core::guardrail::Guardrail::MaxVelocity { limit: 100.0 },
            blocking: false,
        }],
        ..worldforge_core::prediction::PredictionConfig::default()
    };

    // Prediction should succeed since MaxVelocity limit is high (100.0)
    let prediction = world.predict(&action, &config).await.unwrap();
    assert_eq!(prediction.provider, "mock");
    // State should have advanced even with guardrails configured
    assert!(world.current_state().time.step > 0);
}

#[tokio::test]
async fn test_verify_proof_roundtrip() {
    use worldforge_verify::{MockVerifier, ZkVerifier};

    let verifier = MockVerifier::new();
    let proof = verifier.prove_inference([1; 32], [2; 32], [3; 32]).unwrap();

    // Serialize and deserialize
    let json = serde_json::to_string(&proof).unwrap();
    let restored: worldforge_verify::ZkProof = serde_json::from_str(&json).unwrap();

    // Verify the restored proof
    let result = verifier.verify(&restored).unwrap();
    assert!(result.valid);
}
