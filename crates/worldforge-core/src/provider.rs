//! Provider abstraction layer for world foundation models.
//!
//! Defines the `WorldModelProvider` trait that all model adapters must implement,
//! along with the `ProviderRegistry` for managing multiple providers.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionSpaceType};
use crate::async_utils::{join_all_ordered, BoxFuture};
use crate::error::{Result, WorldForgeError};
use crate::prediction::{Plan, PlanRequest, PredictionConfig};
use crate::state::WorldState;
use crate::types::{Tensor, VideoClip};

/// Capabilities declared by a provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Whether the provider supports forward prediction.
    pub predict: bool,
    /// Whether the provider supports video generation from prompts.
    pub generate: bool,
    /// Whether the provider supports physical reasoning.
    pub reason: bool,
    /// Whether the provider supports spatial-control transfer.
    pub transfer: bool,
    /// Whether the provider supports text/video embeddings.
    pub embed: bool,
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
    /// Whether the provider can return depth maps in generated or predicted output.
    pub supports_depth: bool,
    /// Whether the provider can return semantic segmentation in generated or predicted output.
    pub supports_segmentation: bool,
    /// Whether planning is supported natively.
    pub supports_planning: bool,
    /// Latency profile for the provider.
    pub latency_profile: LatencyProfile,
}

/// Latency percentile profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Whether the provider is healthy.
    pub healthy: bool,
    /// Human-readable status message.
    pub message: String,
    /// Latency of the health check in milliseconds.
    pub latency_ms: u64,
}

/// Cost estimate for an operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Estimated cost in USD.
    pub usd: f64,
    /// Estimated compute credits consumed.
    pub credits: f64,
    /// Estimated latency in milliseconds.
    pub estimated_latency_ms: u64,
}

/// Describes the type of operation for cost estimation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

/// Describes a registered provider and its advertised capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderDescriptor {
    /// Provider identifier.
    pub name: String,
    /// Declared capabilities for this provider.
    pub capabilities: ProviderCapabilities,
}

/// Provider metadata paired with the latest live health-check result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderHealthReport {
    /// Provider identifier.
    pub name: String,
    /// Declared capabilities for this provider.
    pub capabilities: ProviderCapabilities,
    /// Latest health status when the live check completed successfully.
    pub status: Option<HealthStatus>,
    /// Error returned while attempting the health check, if any.
    pub error: Option<String>,
}

impl ProviderHealthReport {
    /// Whether the provider completed a live health check and reported healthy.
    pub fn is_healthy(&self) -> bool {
        self.error.is_none() && self.status.as_ref().is_some_and(|status| status.healthy)
    }
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
#[serde(default)]
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

/// Input for embedding requests.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingInput {
    /// Optional text to embed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Optional video clip to embed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoClip>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct EmbeddingInputSerde {
    text: Option<String>,
    video: Option<VideoClip>,
}

impl EmbeddingInput {
    /// Create a new embedding input from optional text and video.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` when both inputs are absent.
    pub fn new(text: Option<String>, video: Option<VideoClip>) -> Result<Self> {
        let input = Self { text, video };
        input.validate()?;
        Ok(input)
    }

    /// Create an embedding input from text only.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            video: None,
        }
    }

    /// Create an embedding input from video only.
    pub fn from_video(video: VideoClip) -> Self {
        Self {
            text: None,
            video: Some(video),
        }
    }

    /// Validate that the input contains at least one modality.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` when both `text` and `video`
    /// are absent.
    pub fn validate(&self) -> Result<()> {
        if self.text.is_none() && self.video.is_none() {
            return Err(WorldForgeError::InvalidState(
                "embedding input must include text and/or video".to_string(),
            ));
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for EmbeddingInput {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = EmbeddingInputSerde::deserialize(deserializer)?;
        Self::new(raw.text, raw.video).map_err(serde::de::Error::custom)
    }
}

/// Embedding output returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingOutput {
    /// Provider identifier that produced the embedding.
    pub provider: String,
    /// Model identifier used for the embedding request.
    pub model: String,
    /// Returned embedding tensor.
    pub embedding: Tensor,
}

/// Spatial controls for transfer operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[serde(default)]
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

    /// Embed text and/or video input into a provider-specific representation.
    async fn embed(&self, _input: &EmbeddingInput) -> Result<EmbeddingOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: self.name().to_string(),
            capability: "embed".to_string(),
        })
    }

    /// Transfer spatial controls to produce a video.
    async fn transfer(
        &self,
        source: &VideoClip,
        controls: &SpatialControls,
        config: &TransferConfig,
    ) -> Result<VideoClip>;

    /// Check provider health and connectivity.
    async fn health_check(&self) -> Result<HealthStatus>;

    /// Produce a provider-native plan when the model supports built-in planning.
    ///
    /// Providers that do not implement native planning should rely on the
    /// default implementation, which returns `UnsupportedCapability`.
    async fn plan(&self, _request: &PlanRequest) -> Result<Plan> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: self.name().to_string(),
            capability: "native planning".to_string(),
        })
    }

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

    /// Describe a provider by name.
    pub fn describe(&self, name: &str) -> Result<ProviderDescriptor> {
        let provider = self.get(name)?;
        Ok(ProviderDescriptor {
            name: name.to_string(),
            capabilities: provider.capabilities(),
        })
    }

    /// Run a live health check for one provider.
    pub async fn health_check(&self, name: &str) -> Result<ProviderHealthReport> {
        let provider = self.get(name)?;
        Ok(build_health_report(name, provider).await)
    }

    /// List all registered provider names.
    pub fn list(&self) -> Vec<&str> {
        self.providers.keys().map(|k| k.as_str()).collect()
    }

    /// Describe all registered providers.
    pub fn describe_all(&self) -> Vec<ProviderDescriptor> {
        let mut descriptors: Vec<_> = self
            .providers
            .iter()
            .map(|(name, provider)| ProviderDescriptor {
                name: name.clone(),
                capabilities: provider.capabilities(),
            })
            .collect();
        descriptors.sort_by(|left, right| left.name.cmp(&right.name));
        descriptors
    }

    /// Find providers that support a given capability.
    pub fn find_by_capability(&self, capability: &str) -> Vec<&dyn WorldModelProvider> {
        self.providers
            .values()
            .filter(|p| {
                let caps = p.capabilities();
                supports_capability(&caps, capability)
            })
            .map(|p| p.as_ref())
            .collect()
    }

    /// Describe providers that support a given capability.
    pub fn describe_by_capability(&self, capability: &str) -> Vec<ProviderDescriptor> {
        let mut descriptors: Vec<_> = self
            .providers
            .iter()
            .filter_map(|(name, provider)| {
                let capabilities = provider.capabilities();
                supports_capability(&capabilities, capability).then(|| ProviderDescriptor {
                    name: name.clone(),
                    capabilities,
                })
            })
            .collect();
        descriptors.sort_by(|left, right| left.name.cmp(&right.name));
        descriptors
    }

    /// Run live health checks for all registered providers.
    pub async fn health_check_all(&self) -> Vec<ProviderHealthReport> {
        self.health_check_filtered(None).await
    }

    /// Run live health checks for providers matching a capability filter.
    pub async fn health_check_by_capability(&self, capability: &str) -> Vec<ProviderHealthReport> {
        self.health_check_filtered(Some(capability)).await
    }

    /// Estimate the cost of an operation on a provider.
    pub fn estimate_cost(&self, name: &str, operation: &Operation) -> Result<CostEstimate> {
        Ok(self.get(name)?.estimate_cost(operation))
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

    async fn health_check_filtered(&self, capability: Option<&str>) -> Vec<ProviderHealthReport> {
        let mut futures = Vec::<BoxFuture<'_, ProviderHealthReport>>::new();

        for (name, provider) in &self.providers {
            let capabilities = provider.capabilities();
            if capability.is_some_and(|capability| !supports_capability(&capabilities, capability))
            {
                continue;
            }

            futures.push(Box::pin(build_health_report(name, provider.as_ref())));
        }

        let mut reports = join_all_ordered(futures).await;
        reports.sort_by(|left, right| left.name.cmp(&right.name));
        reports
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn supports_capability(capabilities: &ProviderCapabilities, capability: &str) -> bool {
    match capability {
        "predict" => capabilities.predict,
        "generate" => capabilities.generate,
        "reason" => capabilities.reason,
        "transfer" => capabilities.transfer,
        "embed" => capabilities.embed,
        "planning" => capabilities.supports_planning,
        "action-conditioned" | "action_conditioned" => capabilities.action_conditioned,
        "multi-view" | "multi_view" => capabilities.multi_view,
        "depth" => capabilities.supports_depth,
        "segmentation" => capabilities.supports_segmentation,
        _ => false,
    }
}

async fn build_health_report(
    name: &str,
    provider: &dyn WorldModelProvider,
) -> ProviderHealthReport {
    let capabilities = provider.capabilities();
    match provider.health_check().await {
        Ok(status) => ProviderHealthReport {
            name: name.to_string(),
            capabilities,
            status: Some(status),
            error: None,
        },
        Err(error) => ProviderHealthReport {
            name: name.to_string(),
            capabilities,
            status: None,
            error: Some(error.to_string()),
        },
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::error::WorldForgeError;
    use crate::prediction::{PlanGoal, PlannerType};

    #[derive(Debug, Default)]
    struct ConcurrencyTracker {
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    impl ConcurrencyTracker {
        fn enter(&self) {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
        }

        fn exit(&self) {
            self.active.fetch_sub(1, Ordering::SeqCst);
        }

        fn max_active(&self) -> usize {
            self.max_active.load(Ordering::SeqCst)
        }
    }

    struct TestProvider {
        name: &'static str,
        capabilities: ProviderCapabilities,
        estimate: CostEstimate,
    }

    struct FailingHealthProvider {
        name: &'static str,
        capabilities: ProviderCapabilities,
    }

    struct DelayedHealthProvider {
        name: &'static str,
        capabilities: ProviderCapabilities,
        delay_ms: u64,
        tracker: Arc<ConcurrencyTracker>,
    }

    #[async_trait::async_trait]
    impl WorldModelProvider for TestProvider {
        fn name(&self) -> &str {
            self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            self.capabilities.clone()
        }

        async fn predict(
            &self,
            _state: &crate::state::WorldState,
            _action: &crate::action::Action,
            _config: &crate::prediction::PredictionConfig,
        ) -> Result<crate::prediction::Prediction> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "predict".to_string(),
            })
        }

        async fn generate(
            &self,
            _prompt: &GenerationPrompt,
            _config: &GenerationConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "generate".to_string(),
            })
        }

        async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "reason".to_string(),
            })
        }

        async fn transfer(
            &self,
            _source: &VideoClip,
            _controls: &SpatialControls,
            _config: &TransferConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "transfer".to_string(),
            })
        }

        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                healthy: true,
                message: "healthy".to_string(),
                latency_ms: 1,
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            self.estimate.clone()
        }
    }

    #[async_trait::async_trait]
    impl WorldModelProvider for FailingHealthProvider {
        fn name(&self) -> &str {
            self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            self.capabilities.clone()
        }

        async fn predict(
            &self,
            _state: &crate::state::WorldState,
            _action: &crate::action::Action,
            _config: &crate::prediction::PredictionConfig,
        ) -> Result<crate::prediction::Prediction> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "predict".to_string(),
            })
        }

        async fn generate(
            &self,
            _prompt: &GenerationPrompt,
            _config: &GenerationConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "generate".to_string(),
            })
        }

        async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "reason".to_string(),
            })
        }

        async fn transfer(
            &self,
            _source: &VideoClip,
            _controls: &SpatialControls,
            _config: &TransferConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "transfer".to_string(),
            })
        }

        async fn health_check(&self) -> Result<HealthStatus> {
            Err(WorldForgeError::ProviderUnavailable {
                provider: self.name.to_string(),
                reason: "offline".to_string(),
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }

    #[async_trait::async_trait]
    impl WorldModelProvider for DelayedHealthProvider {
        fn name(&self) -> &str {
            self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            self.capabilities.clone()
        }

        async fn predict(
            &self,
            _state: &crate::state::WorldState,
            _action: &crate::action::Action,
            _config: &crate::prediction::PredictionConfig,
        ) -> Result<crate::prediction::Prediction> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "predict".to_string(),
            })
        }

        async fn generate(
            &self,
            _prompt: &GenerationPrompt,
            _config: &GenerationConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "generate".to_string(),
            })
        }

        async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "reason".to_string(),
            })
        }

        async fn transfer(
            &self,
            _source: &VideoClip,
            _controls: &SpatialControls,
            _config: &TransferConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "transfer".to_string(),
            })
        }

        async fn health_check(&self) -> Result<HealthStatus> {
            self.tracker.enter();
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            self.tracker.exit();

            Ok(HealthStatus {
                healthy: true,
                message: "healthy".to_string(),
                latency_ms: self.delay_ms,
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }

    fn test_capabilities() -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: false,
            transfer: false,
            embed: false,
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
        }
    }

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

    #[tokio::test]
    async fn test_health_check_returns_report_for_provider() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(TestProvider {
            name: "alpha",
            capabilities: test_capabilities(),
            estimate: CostEstimate::default(),
        }));

        let report = registry.health_check("alpha").await.unwrap();
        assert_eq!(report.name, "alpha");
        assert!(report.is_healthy());
        assert_eq!(report.status.unwrap().message, "healthy");
        assert!(report.error.is_none());
    }

    #[tokio::test]
    async fn test_health_check_records_provider_errors() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(FailingHealthProvider {
            name: "beta",
            capabilities: test_capabilities(),
        }));

        let report = registry.health_check("beta").await.unwrap();
        assert_eq!(report.name, "beta");
        assert!(!report.is_healthy());
        assert!(report.status.is_none());
        assert!(report
            .error
            .as_deref()
            .is_some_and(|error| error.contains("provider unavailable")));
    }

    #[tokio::test]
    async fn test_health_check_by_capability_filters_and_sorts() {
        let mut registry = ProviderRegistry::new();

        let mut predict_caps = test_capabilities();
        predict_caps.supports_planning = true;
        let mut reason_only_caps = test_capabilities();
        reason_only_caps.predict = false;
        reason_only_caps.generate = false;
        reason_only_caps.reason = true;

        registry.register(Box::new(TestProvider {
            name: "gamma",
            capabilities: reason_only_caps,
            estimate: CostEstimate::default(),
        }));
        registry.register(Box::new(TestProvider {
            name: "alpha",
            capabilities: predict_caps.clone(),
            estimate: CostEstimate::default(),
        }));
        registry.register(Box::new(TestProvider {
            name: "beta",
            capabilities: predict_caps,
            estimate: CostEstimate::default(),
        }));

        let reports = registry.health_check_by_capability("planning").await;
        let names: Vec<_> = reports.iter().map(|report| report.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
        assert!(reports.iter().all(ProviderHealthReport::is_healthy));
    }

    #[tokio::test]
    async fn test_health_check_all_runs_concurrently_and_sorts() {
        let tracker = Arc::new(ConcurrencyTracker::default());
        let mut registry = ProviderRegistry::new();

        for name in ["gamma", "alpha", "beta"] {
            registry.register(Box::new(DelayedHealthProvider {
                name,
                capabilities: test_capabilities(),
                delay_ms: 25,
                tracker: Arc::clone(&tracker),
            }));
        }

        let reports = registry.health_check_all().await;
        let names: Vec<_> = reports.iter().map(|report| report.name.as_str()).collect();

        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
        assert!(reports.iter().all(ProviderHealthReport::is_healthy));
        assert!(tracker.max_active() >= 2);
    }

    #[test]
    fn test_capabilities_serialization() {
        let caps = test_capabilities();
        let json = serde_json::to_string(&caps).unwrap();
        let caps2: ProviderCapabilities = serde_json::from_str(&json).unwrap();
        assert!(caps2.predict);
    }

    #[test]
    fn test_spatial_controls_default() {
        let controls = SpatialControls::default();
        assert!(controls.camera_trajectory.is_none());
        assert!(controls.depth_map.is_none());
        assert!(controls.segmentation_map.is_none());
    }

    #[test]
    fn test_describe_all_and_capability_filtering() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(TestProvider {
            name: "alpha",
            capabilities: test_capabilities(),
            estimate: CostEstimate {
                usd: 0.42,
                credits: 3.0,
                estimated_latency_ms: 120,
            },
        }));
        registry.register(Box::new(TestProvider {
            name: "beta",
            capabilities: ProviderCapabilities {
                generate: false,
                ..test_capabilities()
            },
            estimate: CostEstimate::default(),
        }));

        let descriptors = registry.describe_all();
        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].name, "alpha");
        assert_eq!(descriptors[1].name, "beta");

        let generators = registry.describe_by_capability("generate");
        assert_eq!(generators.len(), 1);
        assert_eq!(generators[0].name, "alpha");

        let action_conditioned = registry.describe_by_capability("action-conditioned");
        assert_eq!(action_conditioned.len(), 2);

        let embeds = registry.describe_by_capability("embed");
        assert_eq!(embeds.len(), 0);
    }

    #[test]
    fn test_estimate_cost_delegates_to_provider() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(TestProvider {
            name: "alpha",
            capabilities: test_capabilities(),
            estimate: CostEstimate {
                usd: 1.25,
                credits: 8.0,
                estimated_latency_ms: 750,
            },
        }));

        let estimate = registry
            .estimate_cost(
                "alpha",
                &Operation::Predict {
                    steps: 8,
                    resolution: (1280, 720),
                },
            )
            .unwrap();

        assert_eq!(
            estimate,
            CostEstimate {
                usd: 1.25,
                credits: 8.0,
                estimated_latency_ms: 750,
            }
        );
    }

    #[tokio::test]
    async fn test_default_native_planning_is_unsupported() {
        let provider = TestProvider {
            name: "alpha",
            capabilities: test_capabilities(),
            estimate: CostEstimate::default(),
        };
        let request = PlanRequest {
            current_state: crate::state::WorldState::new("planning", "alpha"),
            goal: PlanGoal::Description("move block to position (1.0, 0.0, 0.0)".to_string()),
            max_steps: 3,
            guardrails: Vec::new(),
            planner: PlannerType::ProviderNative,
            timeout_seconds: 1.0,
            fallback_provider: None,
        };

        let error = provider.plan(&request).await.unwrap_err();
        assert!(matches!(
            error,
            WorldForgeError::UnsupportedCapability { provider, capability }
                if provider == "alpha" && capability == "native planning"
        ));
    }
}
