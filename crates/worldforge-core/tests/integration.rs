//! Integration tests for worldforge-core.
//!
//! Tests the end-to-end flow from WorldForge creation through
//! world state management, scene manipulation, and state persistence.

use worldforge_core::action::Action;
use worldforge_core::error::WorldForgeError;
use worldforge_core::guardrail::{
    evaluate_guardrails, has_blocking_violation, Guardrail, GuardrailConfig,
};
use worldforge_core::prediction::PredictionConfig;
use worldforge_core::scene::{PhysicsProperties, SceneObject};
use worldforge_core::state::{FileStateStore, StateStore, WorldState};
use worldforge_core::types::{BBox, Pose, Position};

// ---------------------------------------------------------------------------
// State persistence integration tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_state_store_save_load_roundtrip() {
    let dir = std::env::temp_dir().join(format!("wf-integ-{}", uuid::Uuid::new_v4()));
    let store = FileStateStore::new(&dir);

    let mut state = WorldState::new("integration-test", "mock");
    state.scene.add_object(SceneObject::new(
        "table",
        Pose::default(),
        BBox {
            min: Position {
                x: -1.0,
                y: -1.0,
                z: -1.0,
            },
            max: Position {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
        },
    ));
    state.scene.add_object(SceneObject::new(
        "mug",
        Pose {
            position: Position {
                x: 0.5,
                y: 1.5,
                z: 0.0,
            },
            ..Pose::default()
        },
        BBox {
            min: Position {
                x: 0.3,
                y: 1.3,
                z: -0.1,
            },
            max: Position {
                x: 0.7,
                y: 1.7,
                z: 0.1,
            },
        },
    ));

    let id = state.id;

    // Save
    store.save(&state).await.unwrap();

    // Load and verify
    let loaded = store.load(&id).await.unwrap();
    assert_eq!(loaded.id, id);
    assert_eq!(loaded.metadata.name, "integration-test");
    assert_eq!(loaded.scene.objects.len(), 2);

    // Verify objects roundtripped correctly
    let table = loaded
        .scene
        .objects
        .values()
        .find(|o| o.name == "table")
        .unwrap();
    assert_eq!(table.pose.position.x, 0.0);
    let mug = loaded
        .scene
        .objects
        .values()
        .find(|o| o.name == "mug")
        .unwrap();
    assert_eq!(mug.pose.position.x, 0.5);

    // List
    let ids = store.list().await.unwrap();
    assert!(ids.contains(&id));

    // Delete
    store.delete(&id).await.unwrap();
    assert!(store.load(&id).await.is_err());

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn test_state_store_multiple_worlds() {
    let dir = std::env::temp_dir().join(format!("wf-integ-multi-{}", uuid::Uuid::new_v4()));
    let store = FileStateStore::new(&dir);

    let state1 = WorldState::new("world-1", "mock");
    let state2 = WorldState::new("world-2", "mock");
    let state3 = WorldState::new("world-3", "mock");

    let id1 = state1.id;
    let id2 = state2.id;
    let id3 = state3.id;

    store.save(&state1).await.unwrap();
    store.save(&state2).await.unwrap();
    store.save(&state3).await.unwrap();

    let ids = store.list().await.unwrap();
    assert_eq!(ids.len(), 3);
    assert!(ids.contains(&id1));
    assert!(ids.contains(&id2));
    assert!(ids.contains(&id3));

    // Delete one and verify
    store.delete(&id2).await.unwrap();
    let ids = store.list().await.unwrap();
    assert_eq!(ids.len(), 2);
    assert!(!ids.contains(&id2));

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn test_state_store_not_found_error() {
    let dir = std::env::temp_dir().join(format!("wf-integ-nf-{}", uuid::Uuid::new_v4()));
    let store = FileStateStore::new(&dir);
    tokio::fs::create_dir_all(&dir).await.unwrap();

    let fake_id = uuid::Uuid::new_v4();
    let result = store.load(&fake_id).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        WorldForgeError::WorldNotFound(id) => assert_eq!(id, fake_id),
        other => panic!("expected WorldNotFound, got: {other}"),
    }

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

// ---------------------------------------------------------------------------
// Scene graph integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_scene_graph_complex_operations() {
    use worldforge_core::scene::{SceneGraph, SpatialRelationship};

    let mut sg = SceneGraph::new();

    let table = SceneObject::new(
        "table",
        Pose::default(),
        BBox {
            min: Position {
                x: -1.0,
                y: -1.0,
                z: -1.0,
            },
            max: Position {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
        },
    );
    let table_id = table.id;

    let mug = SceneObject::new(
        "mug",
        Pose {
            position: Position {
                x: 0.0,
                y: 1.1,
                z: 0.0,
            },
            ..Pose::default()
        },
        BBox {
            min: Position {
                x: -0.05,
                y: 1.05,
                z: -0.05,
            },
            max: Position {
                x: 0.05,
                y: 1.15,
                z: 0.05,
            },
        },
    );
    let mug_id = mug.id;

    sg.add_object(table);
    sg.add_object(mug);

    // Add relationship
    sg.relationships.push(SpatialRelationship::On {
        subject: mug_id,
        surface: table_id,
    });

    // Verify
    assert_eq!(sg.objects.len(), 2);
    assert_eq!(sg.relationships.len(), 1);

    // Modify object
    let mug_mut = sg.get_object_mut(&mug_id).unwrap();
    mug_mut.pose.position.y = 1.2;
    assert_eq!(sg.get_object(&mug_id).unwrap().pose.position.y, 1.2);

    // Remove mug — should also remove relationships
    sg.remove_object(&mug_id);
    assert_eq!(sg.objects.len(), 1);
    assert_eq!(sg.relationships.len(), 0);
}

#[test]
fn test_scene_object_with_physics() {
    let mut obj = SceneObject::new(
        "heavy_block",
        Pose::default(),
        BBox {
            min: Position {
                x: -0.5,
                y: -0.5,
                z: -0.5,
            },
            max: Position {
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
        },
    );
    obj.physics = PhysicsProperties {
        mass: Some(10.0),
        friction: Some(0.8),
        restitution: Some(0.2),
        is_static: false,
        is_graspable: true,
        material: Some("steel".to_string()),
    };
    obj.semantic_label = Some("block".to_string());

    assert_eq!(obj.physics.mass, Some(10.0));
    assert!(obj.physics.is_graspable);

    // Verify serialization with physics
    let json = serde_json::to_string(&obj).unwrap();
    let obj2: SceneObject = serde_json::from_str(&json).unwrap();
    assert_eq!(obj2.physics.mass, Some(10.0));
    assert_eq!(obj2.physics.material, Some("steel".to_string()));
}

// ---------------------------------------------------------------------------
// Guardrail integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_guardrails_on_complex_scene() {
    let mut state = WorldState::new("guardrail-test", "mock");

    // Add two objects that don't overlap
    let obj_a = SceneObject::new(
        "box_a",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            ..Pose::default()
        },
        BBox {
            min: Position {
                x: -0.5,
                y: -0.5,
                z: -0.5,
            },
            max: Position {
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
        },
    );
    let obj_b = SceneObject::new(
        "box_b",
        Pose {
            position: Position {
                x: 5.0,
                y: 0.0,
                z: 0.0,
            },
            ..Pose::default()
        },
        BBox {
            min: Position {
                x: 4.5,
                y: -0.5,
                z: -0.5,
            },
            max: Position {
                x: 5.5,
                y: 0.5,
                z: 0.5,
            },
        },
    );

    state.scene.add_object(obj_a);
    state.scene.add_object(obj_b);

    // All guardrails should pass
    let configs = vec![
        GuardrailConfig {
            guardrail: Guardrail::NoCollisions,
            blocking: true,
        },
        GuardrailConfig {
            guardrail: Guardrail::BoundaryConstraint {
                bounds: BBox {
                    min: Position {
                        x: -10.0,
                        y: -10.0,
                        z: -10.0,
                    },
                    max: Position {
                        x: 10.0,
                        y: 10.0,
                        z: 10.0,
                    },
                },
            },
            blocking: true,
        },
        GuardrailConfig {
            guardrail: Guardrail::MaxVelocity { limit: 10.0 },
            blocking: false,
        },
    ];

    let results = evaluate_guardrails(&configs, &state);
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|r| r.passed));
    assert!(!has_blocking_violation(&results));
}

#[test]
fn test_guardrails_mixed_pass_fail() {
    let mut state = WorldState::new("mixed-guardrail", "mock");

    // Object inside bounds
    let inside = SceneObject::new(
        "inside",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            ..Pose::default()
        },
        BBox {
            min: Position {
                x: -0.5,
                y: -0.5,
                z: -0.5,
            },
            max: Position {
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
        },
    );
    // Object outside bounds (bbox matches position)
    let mut outside = SceneObject::new(
        "outside",
        Pose::default(),
        BBox {
            min: Position {
                x: 99.5,
                y: -0.5,
                z: -0.5,
            },
            max: Position {
                x: 100.5,
                y: 0.5,
                z: 0.5,
            },
        },
    );
    outside.pose.position = Position {
        x: 100.0,
        y: 0.0,
        z: 0.0,
    };

    state.scene.add_object(inside);
    state.scene.add_object(outside);

    let configs = vec![
        GuardrailConfig {
            guardrail: Guardrail::NoCollisions,
            blocking: true,
        },
        GuardrailConfig {
            guardrail: Guardrail::BoundaryConstraint {
                bounds: BBox {
                    min: Position {
                        x: -10.0,
                        y: -10.0,
                        z: -10.0,
                    },
                    max: Position {
                        x: 10.0,
                        y: 10.0,
                        z: 10.0,
                    },
                },
            },
            blocking: true,
        },
    ];

    let results = evaluate_guardrails(&configs, &state);
    // NoCollisions should pass, BoundaryConstraint should fail
    assert!(results[0].passed); // No collision
    assert!(!results[1].passed); // Out of bounds
    assert!(has_blocking_violation(&results));
}

// ---------------------------------------------------------------------------
// Action system integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_complex_compound_action_serialization() {
    use worldforge_core::action::{Condition, Weather};

    let obj_id = uuid::Uuid::new_v4();

    let complex_action = Action::Sequence(vec![
        Action::Move {
            target: Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            speed: 0.5,
        },
        Action::Conditional {
            condition: Condition::ObjectExists { object: obj_id },
            then: Box::new(Action::Parallel(vec![
                Action::Grasp {
                    object: obj_id,
                    grip_force: 5.0,
                },
                Action::SetWeather {
                    weather: Weather::Rain,
                },
            ])),
            otherwise: Some(Box::new(Action::SpawnObject {
                template: "mug".to_string(),
                pose: Pose::default(),
            })),
        },
        Action::SetLighting { time_of_day: 18.0 },
    ]);

    let json = serde_json::to_string(&complex_action).unwrap();
    let deserialized: Action = serde_json::from_str(&json).unwrap();

    match deserialized {
        Action::Sequence(actions) => {
            assert_eq!(actions.len(), 3);
            match &actions[1] {
                Action::Conditional {
                    then, otherwise, ..
                } => {
                    match then.as_ref() {
                        Action::Parallel(inner) => assert_eq!(inner.len(), 2),
                        _ => panic!("expected Parallel"),
                    }
                    assert!(otherwise.is_some());
                }
                _ => panic!("expected Conditional"),
            }
        }
        _ => panic!("expected Sequence"),
    }
}

// ---------------------------------------------------------------------------
// State history integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_state_history_evolution() {
    use worldforge_core::state::{HistoryEntry, PredictionSummary};
    use worldforge_core::types::SimTime;

    let mut state = WorldState::new("history-test", "mock");
    assert!(state.history.is_empty());

    // Simulate multiple prediction steps
    for i in 0..5 {
        state.history.push(HistoryEntry {
            time: SimTime {
                step: i,
                seconds: i as f64 * 0.1,
                dt: 0.1,
            },
            state_hash: [i as u8; 32],
            action: Some(Action::Move {
                target: Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
                speed: 1.0,
            }),
            prediction: Some(PredictionSummary {
                confidence: 0.9 - (i as f32 * 0.02),
                physics_score: 0.85,
                latency_ms: 100 + i * 10,
            }),
            provider: "mock".to_string(),
        });
    }

    assert_eq!(state.history.len(), 5);
    let latest = state.history.latest().unwrap();
    assert_eq!(latest.time.step, 4);
    assert_eq!(latest.provider, "mock");

    // Verify it serializes correctly
    let json = serde_json::to_string(&state).unwrap();
    let state2: WorldState = serde_json::from_str(&json).unwrap();
    assert_eq!(state2.history.len(), 5);
}

// ---------------------------------------------------------------------------
// World orchestration async integration tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_world_predict_basic() {
    use std::sync::Arc;
    use worldforge_core::prediction::PredictionConfig;
    use worldforge_core::provider::ProviderRegistry;
    use worldforge_core::world::World;
    use worldforge_providers::MockProvider;

    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::new()));
        r
    });

    let state = WorldState::new("predict-test", "mock");
    let initial_step = state.time.step;
    let mut world = World::new(state, "mock", registry);

    let action = Action::Move {
        target: Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = PredictionConfig::default();

    let prediction = world.predict(&action, &config).await.unwrap();

    assert_eq!(prediction.provider, "mock");
    assert!(prediction.confidence > 0.0);
    assert!(prediction.physics_scores.overall > 0.0);
    // World state should have advanced
    assert!(world.current_state().time.step > initial_step);
    // History should have one entry
    assert_eq!(world.current_state().history.len(), 1);
}

#[tokio::test]
async fn test_world_predict_multi_compares_providers() {
    use std::sync::Arc;
    use worldforge_core::prediction::PredictionConfig;
    use worldforge_core::provider::ProviderRegistry;
    use worldforge_core::world::World;
    use worldforge_providers::MockProvider;

    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::with_name("provider-a")));
        r.register(Box::new(MockProvider::with_name("provider-b")));
        r
    });

    let state = WorldState::new("multi-test", "mock");
    let world = World::new(state, "provider-a", registry);

    let action = Action::Move {
        target: Position {
            x: 2.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = PredictionConfig::default();

    let multi = world
        .predict_multi(&action, &["provider-a", "provider-b"], &config)
        .await
        .unwrap();

    assert_eq!(multi.predictions.len(), 2);
    assert!(multi.agreement_score > 0.0);
    assert!(multi.agreement_score <= 1.0);
    assert_eq!(multi.comparison.scores.len(), 2);
    assert_eq!(multi.comparison.scores[0].provider, "provider-a");
    assert_eq!(multi.comparison.scores[1].provider, "provider-b");
}

#[tokio::test]
async fn test_world_predict_with_guardrails_pass() {
    use std::sync::Arc;
    use worldforge_core::prediction::PredictionConfig;
    use worldforge_core::provider::ProviderRegistry;
    use worldforge_core::world::World;
    use worldforge_providers::MockProvider;

    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::new()));
        r
    });

    let state = WorldState::new("guardrail-predict", "mock");
    let mut world = World::new(state, "mock", registry);

    let action = Action::Move {
        target: Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = PredictionConfig {
        guardrails: vec![GuardrailConfig {
            guardrail: Guardrail::BoundaryConstraint {
                bounds: BBox {
                    min: Position {
                        x: -100.0,
                        y: -100.0,
                        z: -100.0,
                    },
                    max: Position {
                        x: 100.0,
                        y: 100.0,
                        z: 100.0,
                    },
                },
            },
            blocking: true,
        }],
        ..PredictionConfig::default()
    };

    // Should succeed — everything is within bounds
    let result = world.predict(&action, &config).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_world_predict_unknown_provider_errors() {
    use std::sync::Arc;
    use worldforge_core::prediction::PredictionConfig;
    use worldforge_core::provider::ProviderRegistry;
    use worldforge_core::world::World;
    use worldforge_providers::MockProvider;

    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::new()));
        r
    });

    let state = WorldState::new("error-test", "nonexistent");
    let mut world = World::new(state, "nonexistent", registry);

    let action = Action::Move {
        target: Position::default(),
        speed: 1.0,
    };
    let config = PredictionConfig::default();

    let result = world.predict(&action, &config).await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Prediction config integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_prediction_config_builder_pattern() {
    let config = PredictionConfig {
        steps: 10,
        resolution: (1920, 1080),
        fps: 30.0,
        return_video: true,
        return_depth: true,
        return_segmentation: false,
        guardrails: vec![
            GuardrailConfig {
                guardrail: Guardrail::NoCollisions,
                blocking: true,
            },
            GuardrailConfig {
                guardrail: Guardrail::MaxVelocity { limit: 5.0 },
                blocking: false,
            },
        ],
        max_latency_ms: Some(10_000),
        fallback_provider: Some("mock".to_string()),
        num_samples: 5,
        temperature: 0.8,
    };

    let json = serde_json::to_string(&config).unwrap();
    let config2: PredictionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config2.steps, 10);
    assert_eq!(config2.resolution, (1920, 1080));
    assert_eq!(config2.guardrails.len(), 2);
    assert_eq!(config2.fallback_provider, Some("mock".to_string()));
}
