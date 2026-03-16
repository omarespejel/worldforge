//! Mock provider for testing and development.
//!
//! Returns deterministic, plausible predictions without calling
//! any real model API. Useful for unit tests, integration tests,
//! and offline development.

use async_trait::async_trait;
use worldforge_core::action::{Action, ActionSpaceType};
use worldforge_core::error::Result;
use worldforge_core::prediction::{PhysicsScores, Prediction, PredictionConfig};
use worldforge_core::provider::{
    CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
    ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
    WorldModelProvider,
};
use worldforge_core::state::WorldState;
use worldforge_core::types::VideoClip;

/// A mock provider that returns deterministic predictions.
#[derive(Debug, Clone)]
pub struct MockProvider {
    /// Name of this mock instance.
    name: String,
    /// Simulated latency in milliseconds.
    pub latency_ms: u64,
    /// Default confidence score for predictions.
    pub default_confidence: f32,
}

impl MockProvider {
    /// Create a new mock provider with default settings.
    pub fn new() -> Self {
        Self {
            name: "mock".to_string(),
            latency_ms: 10,
            default_confidence: 0.85,
        }
    }

    /// Create a named mock provider.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::new()
        }
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorldModelProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: true,
            transfer: false,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 10.0,
            max_resolution: (1920, 1080),
            fps_range: (8.0, 30.0),
            supported_action_spaces: vec![
                ActionSpaceType::Continuous,
                ActionSpaceType::Discrete,
                ActionSpaceType::Language,
            ],
            supports_depth: true,
            supports_segmentation: false,
            supports_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 10,
                p95_ms: 20,
                p99_ms: 50,
                throughput_fps: 60.0,
            },
        }
    }

    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        _config: &PredictionConfig,
    ) -> Result<Prediction> {
        // Simulate latency
        if self.latency_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
        }

        // Create a copy of the input state as the output
        let mut output_state = state.clone();
        output_state.time.step += 1;
        output_state.time.seconds += output_state.time.dt.max(1.0 / 24.0);

        // Apply simple action effects to the output state
        apply_mock_action(&mut output_state, action);

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: self.name.clone(),
            model: "mock-v1".to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video: None,
            confidence: self.default_confidence,
            physics_scores: PhysicsScores {
                overall: 0.9,
                object_permanence: 0.95,
                gravity_compliance: 0.9,
                collision_accuracy: 0.85,
                spatial_consistency: 0.9,
                temporal_consistency: 0.92,
            },
            latency_ms: self.latency_ms,
            cost: CostEstimate::default(),
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip> {
        tracing::info!(prompt = %prompt.text, "mock: generating video");
        Ok(VideoClip {
            frames: Vec::new(),
            fps: config.fps,
            resolution: config.resolution,
            duration: config.duration_seconds,
        })
    }

    async fn reason(&self, _input: &ReasoningInput, query: &str) -> Result<ReasoningOutput> {
        Ok(ReasoningOutput {
            answer: format!("Mock reasoning response to: {query}"),
            confidence: 0.8,
            evidence: vec!["mock evidence".to_string()],
        })
    }

    async fn transfer(
        &self,
        source: &VideoClip,
        _controls: &SpatialControls,
        _config: &TransferConfig,
    ) -> Result<VideoClip> {
        // Just return the source as-is for mock
        Ok(source.clone())
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            healthy: true,
            message: "mock provider is always healthy".to_string(),
            latency_ms: 1,
        })
    }

    fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
        CostEstimate {
            usd: 0.0,
            credits: 0.0,
            estimated_latency_ms: self.latency_ms,
        }
    }
}

/// Apply simple mock effects of an action on the world state.
fn apply_mock_action(state: &mut WorldState, action: &Action) {
    match action {
        Action::Move { target, .. } => {
            // Move the first object to the target position
            if let Some(obj) = state.scene.objects.values_mut().next() {
                obj.pose.position = *target;
            }
        }
        Action::RemoveObject { object } => {
            state.scene.remove_object(object);
        }
        Action::SpawnObject { template, pose } => {
            use worldforge_core::scene::SceneObject;
            use worldforge_core::types::{BBox, Position};
            let obj = SceneObject {
                id: uuid::Uuid::new_v4(),
                name: template.clone(),
                pose: *pose,
                bbox: BBox {
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
                velocity: Default::default(),
                mesh: None,
                physics: Default::default(),
                semantic_label: Some(template.clone()),
                visual_embedding: None,
            };
            state.scene.add_object(obj);
        }
        Action::Sequence(actions) => {
            for a in actions {
                apply_mock_action(state, a);
            }
        }
        // Other actions are no-ops in the mock
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::types::Position;

    #[tokio::test]
    async fn test_mock_predict() {
        let provider = MockProvider::new();
        let state = WorldState::new("test", "mock");
        let action = Action::Move {
            target: Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            speed: 1.0,
        };
        let config = PredictionConfig::default();
        let prediction = provider.predict(&state, &action, &config).await.unwrap();
        assert_eq!(prediction.provider, "mock");
        assert!(prediction.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_mock_health() {
        let provider = MockProvider::new();
        let status = provider.health_check().await.unwrap();
        assert!(status.healthy);
    }

    #[tokio::test]
    async fn test_mock_generate() {
        let provider = MockProvider::new();
        let prompt = GenerationPrompt {
            text: "A kitchen with a mug".to_string(),
            reference_image: None,
            negative_prompt: None,
        };
        let config = GenerationConfig {
            resolution: (640, 360),
            fps: 12.0,
            duration_seconds: 5.0,
            ..GenerationConfig::default()
        };
        let clip = provider.generate(&prompt, &config).await.unwrap();
        assert_eq!(clip.fps, 12.0);
        assert_eq!(clip.resolution, (640, 360));
        assert_eq!(clip.duration, 5.0);
    }

    #[tokio::test]
    async fn test_mock_reason() {
        let provider = MockProvider::new();
        let input = ReasoningInput {
            video: None,
            state: None,
        };
        let output = provider.reason(&input, "will it fall?").await.unwrap();
        assert!(!output.answer.is_empty());
    }

    #[test]
    fn test_mock_capabilities() {
        let provider = MockProvider::new();
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(caps.generate);
        assert!(!caps.transfer);
    }
}
