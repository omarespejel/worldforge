//! Cross-crate integration tests for worldforge-providers.
//!
//! Tests the full workflow of provider registration, capability
//! querying, prediction, health checks, and multi-provider comparison.

use std::sync::Arc;

use worldforge_core::action::Action;
use worldforge_core::prediction::PredictionConfig;
use worldforge_core::provider::{ProviderRegistry, WorldModelProvider};
use worldforge_core::state::WorldState;
use worldforge_core::types::Position;
use worldforge_core::world::World;
use worldforge_providers::{auto_detect, MockProvider};

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

#[test]
fn test_all_provider_names_unique() {
    let registry = auto_detect();
    let names = registry.list();
    let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
    assert_eq!(names.len(), unique.len(), "provider names must be unique");
}
