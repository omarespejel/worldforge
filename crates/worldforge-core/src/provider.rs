//! Provider abstraction layer for world foundation models.
//!
//! Defines the `WorldModelProvider` trait that all model adapters must implement,
//! along with the `ProviderRegistry` for managing multiple providers.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionSpaceType};
use crate::error::{Result, WorldForgeError};
use crate::prediction::PredictionConfig;
use crate::state::WorldState;
use crate::types::VideoClip;

/// Capabilities declared by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Whether the provider supports forward prediction.
    pub predict: bool,
    /// Whether the provider supports video generation from prompts.
    pub generate: bool,
    /// Whether the provider supports physical reasoning.
    pub reason: bool,
    /// Whether the provider supports spatial-control transfer.
    pub transfer: bool,
    /// Whether predictions can be conditioned on actions.
    pub action_conditioned: bool,
    /// Whether the provider supports multi-view rendering.
    pub multi_view: bool,
    /// Maximum video length in seconds.
    pub max_video_length_seconds: f32,
    /// Maximum output resolution `(width, height)`.
    pub max_resolution: (u32, u32),
    /// Supported FPS range `(min, max)`.
    pub fps_range: (f32, f32),
    /// Supported action space types.
    pub supported_action_spaces: Vec<ActionSpaceType>,
    /// Whether depth maps are supported.
    pub supports_depth: bool,
    /// Whether semantic segmentation is supported.
    pub supports_segmentation: bool,
    /// Whether planning is supported natively.
    pub supports_planning: bool,
    /// Latency profile for the provider.
    pub latency_profile: LatencyProfile,
}

/// Latency percentile profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyProfile {
    /// 50th percentile latency in milliseconds.
    pub p50_ms: u32,
    /// 95th percentile latency in milliseconds.
    pub p95_ms: u32,
    /// 99th percentile latency in milliseconds.
    pub p99_ms: u32,
    /// Maximum throughput in frames per second.
    pub throughput_fps: f32,
}

/// Health status of a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Whether the provider is healthy.
    pub healthy: bool,
    /// Human-readable status message.
    pub message: String,
    /// Latency of the health check in milliseconds.
    pub latency_ms: u64,
}

/// Cost estimate for an operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Estimated cost in USD.
    pub usd: f64,
    /// Estimated compute credits consumed.
    pub credits: f64,
    /// Estimated latency in milliseconds.
    pub estimated_latency_ms: u64,
}

/// Describes the type of operation for cost estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// A prediction operation.
    Predict { steps: u32, resolution: (u32, u32) },
    /// A generation operation.
    Generate {
        duration_seconds: f64,
        resolution: (u32, u32),
    },
    /// A reasoning query.
    Reason,
    /// A transfer operation.
    Transfer { duration_seconds: f64 },
}

/// Prompt for video generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPrompt {
    /// Text description of the scene to generate.
    pub text: String,
    /// Optional reference image.
    pub reference_image: Option<crate::types::Tensor>,
    /// Optional negative prompt.
    pub negative_prompt: Option<String>,
}

/// Configuration for video generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Output resolution `(width, height)`.
    pub resolution: (u32, u32),
    /// Output frames per second.
    pub fps: f32,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Sampling temperature.
    pub temperature: f32,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

/// Input for reasoning queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningInput {
    /// Video clip to reason about.
    pub video: Option<VideoClip>,
    /// Scene state to reason about.
    pub state: Option<WorldState>,
}

/// Output from reasoning queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningOutput {
    /// Natural language answer.
    pub answer: String,
    /// Confidence score.
    pub confidence: f32,
    /// Supporting evidence or references.
    pub evidence: Vec<String>,
}

/// Spatial controls for transfer operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialControls {
    /// Camera trajectory for the output.
    pub camera_trajectory: Option<crate::types::Trajectory>,
    /// Depth guidance.
    pub depth_map: Option<crate::types::Tensor>,
    /// Segmentation guidance.
    pub segmentation_map: Option<crate::types::Tensor>,
}

/// Configuration for transfer operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferConfig {
    /// Output resolution.
    pub resolution: (u32, u32),
    /// Output FPS.
    pub fps: f32,
    /// Strength of spatial control (0.0–1.0).
    pub control_strength: f32,
}

/// Trait that all world model provider adapters must implement.
#[async_trait::async_trait]
pub trait WorldModelProvider: Send + Sync {
    /// Human-readable name of the provider.
    fn name(&self) -> &str;

    /// Declared capabilities of this provider.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Predict the next world state given an action.
    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<crate::prediction::Prediction>;

    /// Generate a video clip from a text/image prompt.
    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip>;

    /// Perform physical reasoning on input.
    async fn reason(&self, input: &ReasoningInput, query: &str) -> Result<ReasoningOutput>;

    /// Transfer spatial controls to produce a video.
    async fn transfer(
        &self,
        source: &VideoClip,
        controls: &SpatialControls,
        config: &TransferConfig,
    ) -> Result<VideoClip>;

    /// Check provider health and connectivity.
    async fn health_check(&self) -> Result<HealthStatus>;

    /// Estimate cost for an operation.
    fn estimate_cost(&self, operation: &Operation) -> CostEstimate;
}

/// Registry that manages multiple providers.
pub struct ProviderRegistry {
    providers: HashMap<String, Box<dyn WorldModelProvider>>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider.
    pub fn register(&mut self, provider: Box<dyn WorldModelProvider>) {
        let name = provider.name().to_string();
        self.providers.insert(name, provider);
    }

    /// Get a provider by name.
    pub fn get(&self, name: &str) -> Result<&dyn WorldModelProvider> {
        self.providers
            .get(name)
            .map(|p| p.as_ref())
            .ok_or_else(|| WorldForgeError::ProviderNotFound(name.to_string()))
    }

    /// List all registered provider names.
    pub fn list(&self) -> Vec<&str> {
        self.providers.keys().map(|k| k.as_str()).collect()
    }

    /// Find providers that support a given capability.
    pub fn find_by_capability(&self, capability: &str) -> Vec<&dyn WorldModelProvider> {
        self.providers
            .values()
            .filter(|p| {
                let caps = p.capabilities();
                match capability {
                    "predict" => caps.predict,
                    "generate" => caps.generate,
                    "reason" => caps.reason,
                    "transfer" => caps.transfer,
                    "planning" => caps.supports_planning,
                    _ => false,
                }
            })
            .map(|p| p.as_ref())
            .collect()
    }

    /// Number of registered providers.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Consume the registry and return the registered provider instances.
    pub fn into_providers(self) -> Vec<Box<dyn WorldModelProvider>> {
        self.providers.into_values().collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CostEstimate {
    fn default() -> Self {
        Self {
            usd: 0.0,
            credits: 0.0,
            estimated_latency_ms: 0,
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            resolution: (1280, 720),
            fps: 24.0,
            duration_seconds: 4.0,
            temperature: 1.0,
            seed: None,
        }
    }
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            resolution: (1280, 720),
            fps: 24.0,
            control_strength: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_registry() {
        let registry = ProviderRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_provider_not_found() {
        let registry = ProviderRegistry::new();
        let result = registry.get("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_capabilities_serialization() {
        let caps = ProviderCapabilities {
            predict: true,
            generate: true,
            reason: false,
            transfer: false,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 10.0,
            max_resolution: (1920, 1080),
            fps_range: (8.0, 30.0),
            supported_action_spaces: vec![ActionSpaceType::Continuous],
            supports_depth: true,
            supports_segmentation: false,
            supports_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 200,
                p95_ms: 500,
                p99_ms: 1000,
                throughput_fps: 24.0,
            },
        };
        let json = serde_json::to_string(&caps).unwrap();
        let caps2: ProviderCapabilities = serde_json::from_str(&json).unwrap();
        assert!(caps2.predict);
    }
}
