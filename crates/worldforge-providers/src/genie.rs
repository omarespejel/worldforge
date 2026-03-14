//! Google Genie provider adapter (stub).
//!
//! Implements the `WorldModelProvider` trait for Google's Genie models.
//! Genie is currently in research preview — this provider is stubbed
//! until a public API becomes available.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use worldforge_core::action::{Action, ActionSpaceType};
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::prediction::{Prediction, PredictionConfig};
use worldforge_core::provider::{
    CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
    ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
    WorldModelProvider,
};
use worldforge_core::state::WorldState;
use worldforge_core::types::VideoClip;

/// Google Genie model variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenieModel {
    /// Genie 3 — interactive world generation.
    Genie3,
}

/// Google Genie provider adapter (research preview).
///
/// Genie generates interactive, playable environments from text or
/// image prompts. This adapter is a stub awaiting public API access.
#[derive(Debug, Clone)]
pub struct GenieProvider {
    /// Model variant.
    pub model: GenieModel,
    /// API key for authentication (used when public API becomes available).
    #[allow(dead_code)]
    api_key: String,
    /// API endpoint URL.
    pub endpoint: String,
}

impl GenieProvider {
    /// Create a new Genie provider.
    pub fn new(model: GenieModel, api_key: impl Into<String>) -> Self {
        Self {
            model,
            api_key: api_key.into(),
            endpoint: "https://generativelanguage.googleapis.com".to_string(),
        }
    }

    /// Create a Genie provider with a custom endpoint.
    pub fn with_endpoint(
        model: GenieModel,
        api_key: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Self::new(model, api_key)
        }
    }
}

#[async_trait]
impl WorldModelProvider for GenieProvider {
    fn name(&self) -> &str {
        "genie"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: false,
            transfer: false,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 8.0,
            max_resolution: (256, 256), // Genie operates at lower resolution
            fps_range: (8.0, 16.0),
            supported_action_spaces: vec![ActionSpaceType::Discrete],
            supports_depth: false,
            supports_segmentation: false,
            supports_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 500,
                p95_ms: 1500,
                p99_ms: 3000,
                throughput_fps: 16.0,
            },
        }
    }

    async fn predict(
        &self,
        _state: &WorldState,
        _action: &Action,
        _config: &PredictionConfig,
    ) -> Result<Prediction> {
        Err(WorldForgeError::ProviderUnavailable {
            provider: "genie".to_string(),
            reason: "Genie is in research preview — public API not yet available".to_string(),
        })
    }

    async fn generate(
        &self,
        _prompt: &GenerationPrompt,
        _config: &GenerationConfig,
    ) -> Result<VideoClip> {
        Err(WorldForgeError::ProviderUnavailable {
            provider: "genie".to_string(),
            reason: "Genie is in research preview — public API not yet available".to_string(),
        })
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "genie".to_string(),
            capability: "reason (Genie does not support physical reasoning)".to_string(),
        })
    }

    async fn transfer(
        &self,
        _source: &VideoClip,
        _controls: &SpatialControls,
        _config: &TransferConfig,
    ) -> Result<VideoClip> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "genie".to_string(),
            capability: "transfer (Genie does not support spatial control transfer)".to_string(),
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            healthy: false,
            message: "Genie is in research preview — not yet operational".to_string(),
            latency_ms: 0,
        })
    }

    fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
        CostEstimate::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genie_provider_creation() {
        let provider = GenieProvider::new(GenieModel::Genie3, "test-key");
        assert_eq!(provider.name(), "genie");
    }

    #[test]
    fn test_genie_capabilities() {
        let provider = GenieProvider::new(GenieModel::Genie3, "test-key");
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(caps.generate);
        assert!(!caps.reason);
        assert!(!caps.transfer);
        assert!(caps.action_conditioned);
    }

    #[tokio::test]
    async fn test_genie_predict_unavailable() {
        let provider = GenieProvider::new(GenieModel::Genie3, "test-key");
        let state = WorldState::new("test", "genie");
        let action = Action::Move {
            target: worldforge_core::types::Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            speed: 1.0,
        };
        let result = provider
            .predict(&state, &action, &PredictionConfig::default())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_genie_health_not_operational() {
        let provider = GenieProvider::new(GenieModel::Genie3, "test-key");
        let status = provider.health_check().await.unwrap();
        assert!(!status.healthy);
    }

    #[test]
    fn test_genie_model_serialization() {
        let model = GenieModel::Genie3;
        let json = serde_json::to_string(&model).unwrap();
        let model2: GenieModel = serde_json::from_str(&json).unwrap();
        assert!(matches!(model2, GenieModel::Genie3));
    }

    #[test]
    fn test_custom_endpoint() {
        let provider =
            GenieProvider::with_endpoint(GenieModel::Genie3, "key", "http://localhost:8080");
        assert_eq!(provider.endpoint, "http://localhost:8080");
    }
}
