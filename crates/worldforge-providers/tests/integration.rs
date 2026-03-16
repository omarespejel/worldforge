//! Cross-crate integration tests for worldforge-providers.
//!
//! Tests the full workflow of provider registration, capability
//! querying, prediction, health checks, and multi-provider comparison.

use std::fs;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use worldforge_core::action::Action;
use worldforge_core::prediction::PredictionConfig;
use worldforge_core::provider::{ProviderRegistry, WorldModelProvider};
use worldforge_core::scene::SceneObject;
use worldforge_core::state::WorldState;
use worldforge_core::types::{BBox, Pose, Position, Vec3};
use worldforge_core::world::World;
use worldforge_providers::{auto_detect, JepaBackend, JepaProvider, MockProvider};

struct TestModelDir {
    path: std::path::PathBuf,
}

impl TestModelDir {
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "worldforge-providers-integration-{name}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn write_assets(&self) {
        fs::write(self.path.join("model.safetensors"), b"jepa-weights").unwrap();
        fs::write(
            self.path.join("worldforge-jepa.json"),
            r#"{
                "model_name": "vjepa2-local",
                "representation_dim": 2048,
                "action_gain": 1.25,
                "temporal_smoothness": 0.9,
                "gravity_bias": 0.95,
                "collision_bias": 0.88,
                "confidence_bias": 0.08
            }"#,
        )
        .unwrap();
    }
}

impl Drop for TestModelDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn sample_jepa_state() -> (WorldState, uuid::Uuid) {
    let mut state = WorldState::new("jepa-world", "jepa");
    let object = SceneObject::new(
        "crate",
        Pose {
            position: Position {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            ..Default::default()
        },
        BBox {
            min: Position {
                x: -0.15,
                y: 0.85,
                z: -0.15,
            },
            max: Position {
                x: 0.15,
                y: 1.15,
                z: 0.15,
            },
        },
    );
    let object_id = object.id;
    state.scene.add_object(object);
    (state, object_id)
}

#[test]
fn test_auto_detect_registry_mock_present() {
    let registry = auto_detect();
    assert!(registry.get("mock").is_ok());
}

#[test]
fn test_provider_capabilities_querying() {
    let mock = MockProvider::new();
    let caps = mock.capabilities();
    assert!(caps.predict);
    assert!(caps.generate);
    assert!(caps.action_conditioned);
    assert!(!caps.supported_action_spaces.is_empty());
}

#[test]
fn test_registry_find_by_capability() {
    let mut registry = ProviderRegistry::new();
    registry.register(Box::new(MockProvider::new()));

    let predictors = registry.find_by_capability("predict");
    assert_eq!(predictors.len(), 1);
    assert_eq!(predictors[0].name(), "mock");

    let planners = registry.find_by_capability("planning");
    // MockProvider may or may not support planning
    assert!(planners.len() <= 1);
}

#[tokio::test]
async fn test_mock_provider_predict_workflow() {
    let mock = MockProvider::new();
    let state = WorldState::new("test_world", "mock");
    let action = Action::Move {
        target: Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = PredictionConfig::default();

    let prediction = mock.predict(&state, &action, &config).await.unwrap();
    assert_eq!(prediction.provider, "mock");
    assert!(prediction.confidence >= 0.0);
    assert!(prediction.confidence <= 1.0);
}

#[tokio::test]
async fn test_mock_provider_health_check() {
    let mock = MockProvider::new();
    let status = mock.health_check().await.unwrap();
    assert!(status.healthy);
}

#[tokio::test]
async fn test_world_predict_with_mock() {
    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::new()));
        r
    });

    let state = WorldState::new("integration_world", "mock");
    let mut world = World::new(state, "mock", registry);

    let action = Action::Move {
        target: Position {
            x: 5.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 2.0,
    };
    let config = PredictionConfig::default();

    let prediction = world.predict(&action, &config).await.unwrap();
    assert_eq!(prediction.provider, "mock");

    // State should have advanced
    assert!(world.current_state().time.step > 0);
}

#[tokio::test]
async fn test_multi_provider_comparison() {
    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::new()));
        r.register(Box::new(MockProvider::with_name("mock2")));
        r
    });

    let state = WorldState::new("compare_world", "mock");
    let world = World::new(state, "mock", registry);

    let action = Action::Move {
        target: Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = PredictionConfig::default();

    let multi = world
        .predict_multi(&action, &["mock", "mock2"], &config)
        .await
        .unwrap();
    assert_eq!(multi.predictions.len(), 2);
    assert!(multi.agreement_score >= 0.0);
    assert!(multi.agreement_score <= 1.0);
}

#[tokio::test]
async fn test_provider_cost_estimation() {
    let mock = MockProvider::new();
    let cost = mock.estimate_cost(&worldforge_core::provider::Operation::Predict {
        steps: 10,
        resolution: (1280, 720),
    });
    // Mock provider should have zero or minimal cost
    assert!(cost.usd >= 0.0);
    assert_eq!(cost.estimated_latency_ms, mock.latency_ms);
}

#[tokio::test]
async fn test_jepa_provider_predict_workflow() {
    let model_dir = TestModelDir::new("jepa-predict");
    model_dir.write_assets();

    let provider = JepaProvider::new(&model_dir.path, JepaBackend::Burn);
    let (state, object_id) = sample_jepa_state();
    let action = Action::Push {
        object: object_id,
        direction: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        force: 2.0,
    };
    let config = PredictionConfig {
        steps: 4,
        fps: 8.0,
        ..PredictionConfig::default()
    };

    let prediction = provider.predict(&state, &action, &config).await.unwrap();
    let before = state.scene.get_object(&object_id).unwrap().pose.position;
    let after = prediction
        .output_state
        .scene
        .get_object(&object_id)
        .unwrap()
        .pose
        .position;

    assert!(after.x > before.x);
    assert!(prediction.confidence > 0.4);
    assert!(prediction.physics_scores.overall > 0.4);
}

#[tokio::test]
async fn test_world_predict_with_jepa() {
    let model_dir = TestModelDir::new("jepa-world");
    model_dir.write_assets();

    let registry = Arc::new({
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(MockProvider::new()));
        registry.register(Box::new(JepaProvider::new(
            &model_dir.path,
            JepaBackend::Burn,
        )));
        registry
    });

    let (state, object_id) = sample_jepa_state();
    let mut world = World::new(state, "jepa", registry);

    let action = Action::Release { object: object_id };
    let config = PredictionConfig {
        steps: 3,
        fps: 10.0,
        ..PredictionConfig::default()
    };

    let prediction = world.predict(&action, &config).await.unwrap();
    let updated = world.current_state().scene.get_object(&object_id).unwrap();

    assert_eq!(prediction.provider, "jepa");
    assert!(world.current_state().time.step > 0);
    assert!(updated.pose.position.y <= 1.0);
}

#[test]
fn test_all_provider_names_unique() {
    let registry = auto_detect();
    let names = registry.list();
    let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
    assert_eq!(names.len(), unique.len(), "provider names must be unique");
}
