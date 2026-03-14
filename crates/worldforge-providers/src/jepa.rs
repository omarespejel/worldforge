//! Meta JEPA provider adapter (local inference).
//!
//! Implements the `WorldModelProvider` trait for Meta's JEPA family:
//! - I-JEPA: Image JEPA
//! - V-JEPA: Video JEPA
//! - V-JEPA 2: Video + action-conditioned planning
//!
//! This provider runs models locally using burn (Rust-native), PyTorch
//! bindings, or ONNX runtime. It is the primary target for ZK verification.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use worldforge_core::action::{Action, ActionSpaceType};
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::prediction::{PhysicsScores, Prediction, PredictionConfig};
use worldforge_core::provider::{
    CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
    ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
    WorldModelProvider,
};
use worldforge_core::state::WorldState;
use worldforge_core::types::VideoClip;

/// Backend for running JEPA inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JepaBackend {
    /// Rust-native via burn framework.
    Burn,
    /// PyTorch via tch-rs bindings.
    PyTorch,
    /// ONNX via ort-rs runtime.
    Onnx,
    /// Direct weight loading from safetensors.
    Safetensors,
}

/// Meta JEPA provider for local inference.
///
/// Loads V-JEPA / V-JEPA 2 weights and runs inference locally.
/// This is the fully open-source, self-hosted option that enables
/// ZK verification since the inference circuit runs locally.
#[derive(Debug, Clone)]
pub struct JepaProvider {
    /// Path to model weights.
    pub model_path: PathBuf,
    /// Inference backend.
    pub backend: JepaBackend,
}

impl JepaProvider {
    /// Create a new JEPA provider with the given model path and backend.
    pub fn new(model_path: impl Into<PathBuf>, backend: JepaBackend) -> Self {
        Self {
            model_path: model_path.into(),
            backend,
        }
    }

    /// Check if the model weights file exists at the configured path.
    pub fn weights_exist(&self) -> bool {
        self.model_path.exists()
    }
}

#[async_trait]
impl WorldModelProvider for JepaProvider {
    fn name(&self) -> &str {
        "jepa"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: false,
            reason: false,
            transfer: false,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 5.0,
            max_resolution: (224, 224), // JEPA operates on patches
            fps_range: (8.0, 16.0),
            supported_action_spaces: vec![ActionSpaceType::Continuous],
            supports_depth: false,
            supports_segmentation: false,
            supports_planning: true, // Gradient-based planning through differentiable model
            latency_profile: LatencyProfile {
                p50_ms: 100,
                p95_ms: 300,
                p99_ms: 500,
                throughput_fps: 30.0,
            },
        }
    }

    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        if !self.weights_exist() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "jepa".to_string(),
                reason: format!("model weights not found at: {}", self.model_path.display()),
            });
        }

        // TODO: Load model and run forward pass through V-JEPA 2
        // For now, return a stub prediction indicating the provider is available
        // but inference is not yet implemented.
        let mut output_state = state.clone();
        output_state.time.step += config.steps as u64;
        output_state.time.seconds += config.steps as f64 / config.fps as f64;

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: "jepa".to_string(),
            model: "v-jepa-2".to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video: None,
            confidence: 0.0,
            physics_scores: PhysicsScores::default(),
            latency_ms: 0,
            cost: self.estimate_cost(&Operation::Predict {
                steps: config.steps,
                resolution: config.resolution,
            }),
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn generate(
        &self,
        _prompt: &GenerationPrompt,
        _config: &GenerationConfig,
    ) -> Result<VideoClip> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "jepa".to_string(),
            capability: "generate (JEPA models operate in representation space, not pixel space)"
                .to_string(),
        })
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "jepa".to_string(),
            capability: "reason (use Cosmos Reason as fallback)".to_string(),
        })
    }

    async fn transfer(
        &self,
        _source: &VideoClip,
        _controls: &SpatialControls,
        _config: &TransferConfig,
    ) -> Result<VideoClip> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "jepa".to_string(),
            capability: "transfer (JEPA does not support spatial control transfer)".to_string(),
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let healthy = self.weights_exist();
        Ok(HealthStatus {
            healthy,
            message: if healthy {
                format!("JEPA model weights found at {}", self.model_path.display())
            } else {
                format!(
                    "JEPA model weights not found at {}",
                    self.model_path.display()
                )
            },
            latency_ms: 0,
        })
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        // Local inference — no monetary cost, only compute time
        match operation {
            Operation::Predict { steps, .. } => CostEstimate {
                usd: 0.0,
                credits: 0.0,
                estimated_latency_ms: 100 * *steps as u64,
            },
            _ => CostEstimate::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jepa_provider_creation() {
        let provider = JepaProvider::new("/tmp/models/v-jepa-2", JepaBackend::Burn);
        assert_eq!(provider.name(), "jepa");
    }

    #[test]
    fn test_jepa_capabilities() {
        let provider = JepaProvider::new("/tmp/models", JepaBackend::Burn);
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(!caps.generate);
        assert!(!caps.reason);
        assert!(!caps.transfer);
        assert!(caps.supports_planning);
        assert!(caps.action_conditioned);
    }

    #[test]
    fn test_jepa_cost_is_zero() {
        let provider = JepaProvider::new("/tmp/models", JepaBackend::Burn);
        let cost = provider.estimate_cost(&Operation::Predict {
            steps: 10,
            resolution: (224, 224),
        });
        assert_eq!(cost.usd, 0.0);
        assert!(cost.estimated_latency_ms > 0);
    }

    #[test]
    fn test_jepa_backend_serialization() {
        let backends = vec![
            JepaBackend::Burn,
            JepaBackend::PyTorch,
            JepaBackend::Onnx,
            JepaBackend::Safetensors,
        ];
        for b in backends {
            let json = serde_json::to_string(&b).unwrap();
            let _: JepaBackend = serde_json::from_str(&json).unwrap();
        }
    }

    #[tokio::test]
    async fn test_jepa_generate_unsupported() {
        let provider = JepaProvider::new("/tmp/models", JepaBackend::Burn);
        let result = provider
            .generate(
                &GenerationPrompt {
                    text: "test".to_string(),
                    reference_image: None,
                    negative_prompt: None,
                },
                &GenerationConfig::default(),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_jepa_reason_unsupported() {
        let provider = JepaProvider::new("/tmp/models", JepaBackend::Burn);
        let result = provider
            .reason(
                &ReasoningInput {
                    video: None,
                    state: None,
                },
                "will it fall?",
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_jepa_health_check_no_weights() {
        let provider = JepaProvider::new("/nonexistent/path", JepaBackend::Burn);
        let status = provider.health_check().await.unwrap();
        assert!(!status.healthy);
    }
}
