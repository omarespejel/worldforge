//! WorldForge Core Library
//!
//! The orchestration layer for world foundation models (WFMs).
//! Provides unified types, traits, and state management for
//! interacting with multiple world model providers.
//!
//! # Modules
//!
//! - [`types`] — Tensor, spatial, temporal, and media types
//! - [`error`] — Error types and result alias
//! - [`scene`] — Scene graph for spatial representation
//! - [`goal_image`] — Goal-image rendering and similarity helpers
//! - [`state`] — World state and persistence
//! - [`action`] — Action type system
//! - [`provider`] — Provider abstraction and registry
//! - [`prediction`] — Prediction engine and planning
//! - [`proof`] — Shared proof metadata types
//! - [`guardrail`] — Safety constraints
//! - [`world`] — World orchestration

pub mod action;
mod async_utils;
mod bootstrap;
pub mod error;
/// Goal-image rendering and similarity utilities.
pub mod goal_image;
pub mod guardrail;
pub mod prediction;
pub mod proof;
pub mod provider;
pub mod scene;
pub mod state;
pub mod types;
pub mod world;

use std::sync::Arc;

use action::Action;
use error::Result;
use prediction::{MultiPrediction, Prediction, PredictionConfig};
use provider::{
    CostEstimate, EmbeddingInput, EmbeddingOutput, Operation, ProviderDescriptor,
    ProviderHealthReport, ProviderRegistry, ReasoningInput, ReasoningOutput,
};
use state::{DynStateStore, WorldState};
use world::World;

/// The main entry point for WorldForge.
///
/// Manages provider registration and world creation.
pub struct WorldForge {
    /// Provider registry.
    registry: Arc<ProviderRegistry>,
    /// Optional state store for persistence helpers.
    state_store: Option<DynStateStore>,
}

impl WorldForge {
    /// Create a new WorldForge instance.
    pub fn new() -> Self {
        Self::from_registry(ProviderRegistry::new())
    }

    /// Create a new WorldForge instance with an attached state store.
    pub fn with_state_store(store: DynStateStore) -> Self {
        Self::from_registry_with_state_store(ProviderRegistry::new(), store)
    }

    /// Create a new WorldForge instance from an existing provider registry.
    pub fn from_registry(registry: ProviderRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
            state_store: None,
        }
    }

    /// Create a new WorldForge instance from an existing registry and state store.
    pub fn from_registry_with_state_store(
        registry: ProviderRegistry,
        store: DynStateStore,
    ) -> Self {
        Self {
            registry: Arc::new(registry),
            state_store: Some(store),
        }
    }

    /// Register a world model provider.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if worlds are already active
    /// (i.e., the registry `Arc` has been cloned).
    pub fn register_provider(
        &mut self,
        provider: Box<dyn provider::WorldModelProvider>,
    ) -> Result<()> {
        Arc::get_mut(&mut self.registry)
            .ok_or_else(|| {
                error::WorldForgeError::InvalidState(
                    "cannot register provider while worlds are active".to_string(),
                )
            })?
            .register(provider);
        Ok(())
    }

    /// List all registered provider names.
    pub fn providers(&self) -> Vec<&str> {
        self.registry.list()
    }

    /// Describe all registered providers, optionally filtering by capability.
    pub fn provider_infos(&self, capability: Option<&str>) -> Vec<ProviderDescriptor> {
        match capability {
            Some(capability) => self.registry.describe_by_capability(capability),
            None => self.registry.describe_all(),
        }
    }

    /// Describe a single provider by name.
    pub fn provider_info(&self, provider: &str) -> Result<ProviderDescriptor> {
        self.registry.describe(provider)
    }

    /// Run live health checks across the registered providers.
    #[allow(clippy::manual_async_fn)]
    pub fn provider_healths<'a>(
        &'a self,
        capability: Option<&'a str>,
    ) -> impl std::future::Future<Output = Vec<ProviderHealthReport>> + 'a {
        async move {
            match capability {
                Some(capability) => self.registry.health_check_by_capability(capability).await,
                None => self.registry.health_check_all().await,
            }
        }
    }

    /// Run a live health check for one provider.
    pub async fn provider_health(&self, provider: &str) -> Result<ProviderHealthReport> {
        self.registry.health_check(provider).await
    }

    /// Estimate the cost of an operation for a provider.
    pub fn estimate_cost(&self, provider: &str, operation: &Operation) -> Result<CostEstimate> {
        self.registry.estimate_cost(provider, operation)
    }

    /// Ask a specific provider to reason about supplied state or video input.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider is unknown, reasoning is unsupported,
    /// or the input does not include a state or video payload.
    pub async fn reason(
        &self,
        provider: &str,
        input: &ReasoningInput,
        query: &str,
    ) -> Result<ReasoningOutput> {
        self.reason_with_fallback(provider, input, query, None)
            .await
            .map(|(_, output)| output)
    }

    /// Ask a specific provider to reason about supplied state or video input
    /// with an optional fallback provider.
    ///
    /// Returns the provider name that ultimately satisfied the request alongside
    /// the output.
    ///
    /// # Errors
    ///
    /// Returns an error if the primary request fails and no fallback succeeds,
    /// or if the input does not include a state or video payload.
    pub async fn reason_with_fallback(
        &self,
        provider: &str,
        input: &ReasoningInput,
        query: &str,
        fallback_provider: Option<&str>,
    ) -> Result<(String, ReasoningOutput)> {
        validate_reasoning_input(input)?;

        match self.registry.get(provider) {
            Ok(provider_ref) => match provider_ref.reason(input, query).await {
                Ok(output) => Ok((provider.to_string(), output)),
                Err(primary_error) => {
                    let Some(fallback_provider) =
                        fallback_provider.filter(|fallback| *fallback != provider)
                    else {
                        return Err(primary_error);
                    };

                    tracing::warn!(
                        provider,
                        fallback = fallback_provider,
                        error = %primary_error,
                        "reasoning failed on primary provider, attempting fallback"
                    );

                    match self.registry.get(fallback_provider)?.reason(input, query).await {
                        Ok(output) => Ok((fallback_provider.to_string(), output)),
                        Err(fallback_error) => Err(error::WorldForgeError::ProviderUnavailable {
                            provider: provider.to_string(),
                            reason: format!(
                                "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                            ),
                        }),
                    }
                }
            },
            Err(primary_error) => {
                let Some(fallback_provider) =
                    fallback_provider.filter(|fallback| *fallback != provider)
                else {
                    return Err(primary_error);
                };

                tracing::warn!(
                    provider,
                    fallback = fallback_provider,
                    error = %primary_error,
                    "reasoning failed on primary provider, attempting fallback"
                );

                match self.registry.get(fallback_provider)?.reason(input, query).await {
                    Ok(output) => Ok((fallback_provider.to_string(), output)),
                    Err(fallback_error) => Err(error::WorldForgeError::ProviderUnavailable {
                        provider: provider.to_string(),
                        reason: format!(
                            "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                        ),
                    }),
                }
            }
        }
    }

    /// Request an embedding from a specific provider.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider is unknown or does not support embeddings.
    pub async fn embed(&self, provider: &str, input: &EmbeddingInput) -> Result<EmbeddingOutput> {
        input.validate()?;
        self.embed_with_fallback(provider, input, None)
            .await
            .map(|(_, output)| output)
    }

    /// Request an embedding from a specific provider with an optional fallback provider.
    ///
    /// Returns the provider name that ultimately satisfied the request alongside the output.
    ///
    /// # Errors
    ///
    /// Returns an error if the primary request fails and no fallback succeeds.
    pub async fn embed_with_fallback(
        &self,
        provider: &str,
        input: &EmbeddingInput,
        fallback_provider: Option<&str>,
    ) -> Result<(String, EmbeddingOutput)> {
        input.validate()?;

        match self.registry.get(provider) {
            Ok(provider_ref) => match provider_ref.embed(input).await {
                Ok(output) => Ok((provider.to_string(), output)),
                Err(primary_error) => {
                    let Some(fallback_provider) =
                        fallback_provider.filter(|fallback| *fallback != provider)
                    else {
                        return Err(primary_error);
                    };

                    tracing::warn!(
                        provider,
                        fallback = fallback_provider,
                        error = %primary_error,
                        "embedding failed on primary provider, attempting fallback"
                    );

                    match self.registry.get(fallback_provider)?.embed(input).await {
                        Ok(output) => Ok((fallback_provider.to_string(), output)),
                        Err(fallback_error) => Err(error::WorldForgeError::ProviderUnavailable {
                            provider: provider.to_string(),
                            reason: format!(
                                "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                            ),
                        }),
                    }
                }
            },
            Err(primary_error) => {
                let Some(fallback_provider) =
                    fallback_provider.filter(|fallback| *fallback != provider)
                else {
                    return Err(primary_error);
                };

                tracing::warn!(
                    provider,
                    fallback = fallback_provider,
                    error = %primary_error,
                    "embedding failed on primary provider, attempting fallback"
                );

                match self.registry.get(fallback_provider)?.embed(input).await {
                    Ok(output) => Ok((fallback_provider.to_string(), output)),
                    Err(fallback_error) => Err(error::WorldForgeError::ProviderUnavailable {
                        provider: provider.to_string(),
                        reason: format!(
                            "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                        ),
                    }),
                }
            }
        }
    }

    /// Compare previously generated predictions.
    pub fn compare(&self, predictions: Vec<Prediction>) -> Result<MultiPrediction> {
        MultiPrediction::try_from_predictions(predictions)
    }

    /// Compare provider predictions for a supplied world state without persisting it.
    pub async fn compare_world_state(
        &self,
        state: WorldState,
        default_provider: impl Into<String>,
        action: &Action,
        providers: &[&str],
        config: &PredictionConfig,
    ) -> Result<MultiPrediction> {
        let world = self.load_world(state, default_provider)?;
        world.predict_multi(action, providers, config).await
    }

    /// Create a new world with the given name and default provider.
    pub fn create_world(
        &self,
        name: impl Into<String>,
        provider_name: impl Into<String>,
    ) -> Result<World> {
        let provider_name = provider_name.into();
        // Verify the provider exists
        self.registry.get(&provider_name)?;
        let state = WorldState::new(name, &provider_name);
        Ok(World::new(state, provider_name, Arc::clone(&self.registry)))
    }

    /// Create a new world seeded from a natural-language prompt.
    ///
    /// The prompt is stored in metadata and used to synthesize a deterministic
    /// starter scene.
    pub async fn create_world_from_prompt(
        &self,
        prompt: &str,
        provider_name: impl Into<String>,
        name_override: Option<&str>,
    ) -> Result<World> {
        let provider_name = provider_name.into();
        self.registry.get(&provider_name)?;
        let state = WorldState::from_prompt(prompt, &provider_name, name_override)?;
        Ok(World::new(state, provider_name, Arc::clone(&self.registry)))
    }

    /// Load a world from a state store.
    pub fn load_world(
        &self,
        state: WorldState,
        default_provider: impl Into<String>,
    ) -> Result<World> {
        let provider_name = default_provider.into();
        self.registry.get(&provider_name)?;
        Ok(World::new(state, provider_name, Arc::clone(&self.registry)))
    }

    /// Save a world state into the configured state store.
    pub async fn save_state(&self, state: &WorldState) -> Result<crate::types::WorldId> {
        let store = self.state_store()?;
        store.save(state).await?;
        Ok(state.id)
    }

    /// Save a live world into the configured state store.
    pub async fn save_world(&self, world: &World) -> Result<crate::types::WorldId> {
        self.save_state(&world.state).await
    }

    /// Load a world state from the configured state store.
    pub async fn load_state(&self, id: &crate::types::WorldId) -> Result<WorldState> {
        let store = self.state_store()?;
        store.load(id).await
    }

    /// Load a world from the configured state store.
    pub async fn load_world_from_store(&self, id: &crate::types::WorldId) -> Result<World> {
        let state = self.load_state(id).await?;
        let provider_name = state.current_state_provider();
        self.registry.get(&provider_name)?;
        Ok(World::new(state, provider_name, Arc::clone(&self.registry)))
    }

    /// Fork a persisted world from the configured state store and save the branch.
    ///
    /// If `history_index` is provided, the fork starts from that recorded
    /// checkpoint; otherwise it forks the world's current materialized state.
    ///
    /// # Errors
    ///
    /// Returns an error if the world cannot be loaded, the requested history
    /// checkpoint is unavailable, the forked provider is not registered, or
    /// the branch cannot be saved.
    pub async fn fork_world(
        &self,
        id: &crate::types::WorldId,
        history_index: Option<usize>,
        name_override: Option<&str>,
    ) -> Result<World> {
        let state = self.load_state(id).await?;
        let forked = match history_index {
            Some(index) => state.fork_from_history(index, name_override)?,
            None => state.fork(name_override)?,
        };
        let provider_name = forked.current_state_provider();
        self.registry.get(&provider_name)?;

        let store = self.state_store()?;
        store.save(&forked).await?;

        Ok(World::new(
            forked,
            provider_name,
            Arc::clone(&self.registry),
        ))
    }

    /// List the IDs of all saved worlds in the configured state store.
    pub async fn list_worlds(&self) -> Result<Vec<crate::types::WorldId>> {
        let store = self.state_store()?;
        store.list().await
    }

    /// Delete a saved world from the configured state store.
    pub async fn delete_world(&self, id: &crate::types::WorldId) -> Result<()> {
        let store = self.state_store()?;
        store.delete(id).await
    }

    /// Get a reference to the provider registry.
    pub fn registry(&self) -> &ProviderRegistry {
        &self.registry
    }

    /// Clone the shared provider registry handle.
    pub fn registry_arc(&self) -> Arc<ProviderRegistry> {
        Arc::clone(&self.registry)
    }

    fn state_store(&self) -> Result<&DynStateStore> {
        self.state_store.as_ref().ok_or_else(|| {
            error::WorldForgeError::InvalidState(
                "no state store configured for persistence operations".to_string(),
            )
        })
    }
}

fn validate_reasoning_input(input: &ReasoningInput) -> Result<()> {
    if input.state.is_none() && input.video.is_none() {
        return Err(error::WorldForgeError::InvalidState(
            "reasoning input must include state and/or video".to_string(),
        ));
    }

    Ok(())
}

impl Default for WorldForge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{
        EmbeddingInput, EmbeddingOutput, HealthStatus, LatencyProfile, ProviderCapabilities,
        ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig, WorldModelProvider,
    };
    use crate::types::{DType, Device, Tensor, TensorData};
    use async_trait::async_trait;

    #[test]
    fn test_worldforge_new() {
        let wf = WorldForge::new();
        assert!(wf.providers().is_empty());
    }

    #[tokio::test]
    async fn test_create_world_from_prompt_seeds_scene() {
        let mut wf = WorldForge::new();
        wf.register_provider(Box::new(EmbedProvider)).unwrap();

        let world = wf
            .create_world_from_prompt("A kitchen with a mug", "embedder", None)
            .await
            .unwrap();

        assert_eq!(
            world.current_state().metadata.description,
            "A kitchen with a mug"
        );
        assert_eq!(world.current_state().history.len(), 1);
        assert!(!world.current_state().scene.objects.is_empty());
    }

    #[tokio::test]
    async fn test_fork_world_persists_branch() {
        let dir = std::env::temp_dir().join(format!("worldforge-fork-{}", uuid::Uuid::new_v4()));
        let store = crate::state::StateStoreKind::File(dir)
            .open()
            .await
            .unwrap();
        let mut wf = WorldForge::with_state_store(store);
        wf.register_provider(Box::new(EmbedProvider)).unwrap();

        let mut world = wf.create_world("source world", "embedder").unwrap();
        world.state.ensure_history_initialized("embedder").unwrap();
        world.state.metadata.description = "source description".to_string();
        world.state.time = crate::types::SimTime {
            step: 1,
            seconds: 1.0,
            dt: 1.0,
        };
        world
            .state
            .record_current_state(None, None, "embedder")
            .unwrap();
        let source_id = world.id();
        wf.save_world(&world).await.unwrap();

        let forked = wf
            .fork_world(&source_id, None, Some("branch world"))
            .await
            .unwrap();

        assert_ne!(forked.id(), source_id);
        assert_eq!(forked.current_state().metadata.name, "branch world");
        assert_eq!(forked.current_state().history.len(), 1);

        let persisted = wf.load_state(&forked.id()).await.unwrap();
        assert_eq!(persisted.metadata.name, "branch world");
        assert_eq!(persisted.history.len(), 1);
        assert_eq!(persisted.current_state_provider(), "embedder");
    }

    #[tokio::test]
    async fn test_fork_world_uses_history_checkpoint() {
        let dir =
            std::env::temp_dir().join(format!("worldforge-fork-history-{}", uuid::Uuid::new_v4()));
        let store = crate::state::StateStoreKind::File(dir)
            .open()
            .await
            .unwrap();
        let mut wf = WorldForge::with_state_store(store);
        wf.register_provider(Box::new(EmbedProvider)).unwrap();

        let mut world = wf.create_world("history source", "embedder").unwrap();
        world.state.ensure_history_initialized("embedder").unwrap();
        world.state.time = crate::types::SimTime {
            step: 1,
            seconds: 1.0,
            dt: 1.0,
        };
        world.state.metadata.name = "checkpoint".to_string();
        world
            .state
            .record_current_state(None, None, "embedder")
            .unwrap();
        world.state.time = crate::types::SimTime {
            step: 2,
            seconds: 2.0,
            dt: 1.0,
        };
        world.state.metadata.name = "latest".to_string();
        world
            .state
            .record_current_state(None, None, "embedder")
            .unwrap();
        let source_id = world.id();
        wf.save_world(&world).await.unwrap();

        let forked = wf.fork_world(&source_id, Some(1), None).await.unwrap();

        assert_ne!(forked.id(), source_id);
        assert_eq!(forked.current_state().metadata.name, "checkpoint Fork");
        assert_eq!(forked.current_state().time.step, 1);
        assert_eq!(forked.current_state().history.len(), 1);

        let persisted = wf.load_state(&forked.id()).await.unwrap();
        assert_eq!(persisted.metadata.name, "checkpoint Fork");
        assert_eq!(persisted.time.step, 1);
        assert_eq!(persisted.history.len(), 1);
    }

    #[test]
    fn test_create_world_without_provider() {
        let wf = WorldForge::new();
        let result = wf.create_world("test", "nonexistent");
        assert!(result.is_err());
    }

    struct EmbedProvider;

    struct ReasonProvider {
        name: &'static str,
        should_fail: bool,
    }

    #[async_trait]
    impl WorldModelProvider for EmbedProvider {
        fn name(&self) -> &str {
            "embedder"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                predict: false,
                generate: false,
                reason: false,
                transfer: false,
                embed: true,
                action_conditioned: false,
                multi_view: false,
                max_video_length_seconds: 0.0,
                max_resolution: (0, 0),
                fps_range: (0.0, 0.0),
                supported_action_spaces: Vec::new(),
                supports_depth: false,
                supports_segmentation: false,
                supports_planning: false,
                latency_profile: LatencyProfile {
                    p50_ms: 1,
                    p95_ms: 1,
                    p99_ms: 1,
                    throughput_fps: 1.0,
                },
            }
        }

        async fn predict(
            &self,
            _state: &crate::state::WorldState,
            _action: &crate::action::Action,
            _config: &crate::prediction::PredictionConfig,
        ) -> Result<crate::prediction::Prediction> {
            Err(crate::error::WorldForgeError::UnsupportedCapability {
                provider: self.name().to_string(),
                capability: "predict".to_string(),
            })
        }

        async fn generate(
            &self,
            _prompt: &crate::provider::GenerationPrompt,
            _config: &crate::provider::GenerationConfig,
        ) -> Result<crate::types::VideoClip> {
            Err(crate::error::WorldForgeError::UnsupportedCapability {
                provider: self.name().to_string(),
                capability: "generate".to_string(),
            })
        }

        async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
            Err(crate::error::WorldForgeError::UnsupportedCapability {
                provider: self.name().to_string(),
                capability: "reason".to_string(),
            })
        }

        async fn embed(&self, input: &EmbeddingInput) -> Result<EmbeddingOutput> {
            input.validate()?;
            Ok(EmbeddingOutput {
                provider: self.name().to_string(),
                model: "embedder-v1".to_string(),
                embedding: Tensor {
                    data: TensorData::Float32(vec![0.25, 0.5, 0.75]),
                    shape: vec![3],
                    dtype: DType::Float32,
                    device: Device::Cpu,
                },
            })
        }

        async fn transfer(
            &self,
            _source: &crate::types::VideoClip,
            _controls: &SpatialControls,
            _config: &TransferConfig,
        ) -> Result<crate::types::VideoClip> {
            Err(crate::error::WorldForgeError::UnsupportedCapability {
                provider: self.name().to_string(),
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

        fn estimate_cost(
            &self,
            _operation: &crate::provider::Operation,
        ) -> crate::provider::CostEstimate {
            crate::provider::CostEstimate::default()
        }
    }

    #[async_trait]
    impl WorldModelProvider for ReasonProvider {
        fn name(&self) -> &str {
            self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                predict: false,
                generate: false,
                reason: true,
                transfer: false,
                embed: false,
                action_conditioned: false,
                multi_view: false,
                max_video_length_seconds: 0.0,
                max_resolution: (0, 0),
                fps_range: (0.0, 0.0),
                supported_action_spaces: Vec::new(),
                supports_depth: false,
                supports_segmentation: false,
                supports_planning: false,
                latency_profile: LatencyProfile {
                    p50_ms: 1,
                    p95_ms: 1,
                    p99_ms: 1,
                    throughput_fps: 1.0,
                },
            }
        }

        async fn predict(
            &self,
            _state: &crate::state::WorldState,
            _action: &crate::action::Action,
            _config: &crate::prediction::PredictionConfig,
        ) -> Result<crate::prediction::Prediction> {
            Err(crate::error::WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "predict".to_string(),
            })
        }

        async fn generate(
            &self,
            _prompt: &crate::provider::GenerationPrompt,
            _config: &crate::provider::GenerationConfig,
        ) -> Result<crate::types::VideoClip> {
            Err(crate::error::WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "generate".to_string(),
            })
        }

        async fn reason(&self, input: &ReasoningInput, query: &str) -> Result<ReasoningOutput> {
            if self.should_fail {
                return Err(crate::error::WorldForgeError::UnsupportedCapability {
                    provider: self.name.to_string(),
                    capability: "reason".to_string(),
                });
            }

            Ok(ReasoningOutput {
                answer: format!(
                    "{}:{}:{}:{}",
                    self.name,
                    query,
                    input.state.is_some(),
                    input.video.is_some()
                ),
                confidence: 0.99,
                evidence: vec![
                    format!("state={}", input.state.is_some()),
                    format!("video={}", input.video.is_some()),
                ],
            })
        }

        async fn embed(&self, _input: &EmbeddingInput) -> Result<EmbeddingOutput> {
            Err(crate::error::WorldForgeError::UnsupportedCapability {
                provider: self.name.to_string(),
                capability: "embed".to_string(),
            })
        }

        async fn transfer(
            &self,
            _source: &crate::types::VideoClip,
            _controls: &SpatialControls,
            _config: &TransferConfig,
        ) -> Result<crate::types::VideoClip> {
            Err(crate::error::WorldForgeError::UnsupportedCapability {
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

        fn estimate_cost(
            &self,
            _operation: &crate::provider::Operation,
        ) -> crate::provider::CostEstimate {
            crate::provider::CostEstimate::default()
        }
    }

    #[tokio::test]
    async fn test_worldforge_embed_delegates_to_provider() {
        let mut registry = crate::provider::ProviderRegistry::new();
        registry.register(Box::new(EmbedProvider));
        let wf = WorldForge::from_registry(registry);

        let result = wf
            .embed("embedder", &EmbeddingInput::from_text("hello world"))
            .await
            .unwrap();

        assert_eq!(result.provider, "embedder");
        assert_eq!(result.model, "embedder-v1");
        assert_eq!(result.embedding.shape, vec![3]);
    }

    #[tokio::test]
    async fn test_worldforge_embed_uses_fallback_provider() {
        let mut registry = crate::provider::ProviderRegistry::new();
        registry.register(Box::new(EmbedProvider));
        let wf = WorldForge::from_registry(registry);

        let (provider, result) = wf
            .embed_with_fallback(
                "missing",
                &EmbeddingInput::from_text("fallback please"),
                Some("embedder"),
            )
            .await
            .unwrap();

        assert_eq!(provider, "embedder");
        assert_eq!(result.provider, "embedder");
        assert_eq!(result.model, "embedder-v1");
    }

    #[tokio::test]
    async fn test_worldforge_reason_delegates_to_provider() {
        let mut registry = crate::provider::ProviderRegistry::new();
        registry.register(Box::new(ReasonProvider {
            name: "reasoner",
            should_fail: false,
        }));
        let wf = WorldForge::from_registry(registry);

        let output = wf
            .reason(
                "reasoner",
                &ReasoningInput {
                    state: None,
                    video: Some(crate::types::VideoClip {
                        frames: Vec::new(),
                        fps: 12.0,
                        resolution: (320, 180),
                        duration: 1.5,
                    }),
                },
                "what do you see?",
            )
            .await
            .unwrap();

        assert!(output.answer.contains("reasoner:what do you see?"));
        assert!(output.evidence.iter().any(|entry| entry == "video=true"));
    }

    #[tokio::test]
    async fn test_worldforge_reason_uses_fallback_provider() {
        let mut registry = crate::provider::ProviderRegistry::new();
        registry.register(Box::new(ReasonProvider {
            name: "primary",
            should_fail: true,
        }));
        registry.register(Box::new(ReasonProvider {
            name: "fallback",
            should_fail: false,
        }));
        let wf = WorldForge::from_registry(registry);

        let (provider, output) = wf
            .reason_with_fallback(
                "primary",
                &ReasoningInput {
                    state: Some(crate::state::WorldState::new("reason-world", "primary")),
                    video: None,
                },
                "count the objects",
                Some("fallback"),
            )
            .await
            .unwrap();

        assert_eq!(provider, "fallback");
        assert!(output.answer.contains("fallback:count the objects"));
    }

    #[tokio::test]
    async fn test_worldforge_reason_rejects_empty_input() {
        let wf = WorldForge::new();
        let error = wf
            .reason(
                "missing",
                &ReasoningInput {
                    state: None,
                    video: None,
                },
                "what happens?",
            )
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            crate::error::WorldForgeError::InvalidState(_)
        ));
    }
}
