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
//! - [`state`] — World state and persistence
//! - [`action`] — Action type system
//! - [`provider`] — Provider abstraction and registry
//! - [`prediction`] — Prediction engine and planning
//! - [`guardrail`] — Safety constraints
//! - [`world`] — World orchestration

pub mod action;
pub mod error;
pub mod guardrail;
pub mod prediction;
pub mod provider;
pub mod scene;
pub mod state;
pub mod types;
pub mod world;

use std::sync::Arc;

use error::Result;
use prediction::{MultiPrediction, Prediction};
use provider::{
    CostEstimate, Operation, ProviderDescriptor, ProviderHealthReport, ProviderRegistry,
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
        Self {
            registry: Arc::new(ProviderRegistry::new()),
            state_store: None,
        }
    }

    /// Create a new WorldForge instance with an attached state store.
    pub fn with_state_store(store: DynStateStore) -> Self {
        Self {
            registry: Arc::new(ProviderRegistry::new()),
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

    /// Compare previously generated predictions.
    pub fn compare(&self, predictions: Vec<Prediction>) -> Result<MultiPrediction> {
        MultiPrediction::try_from_predictions(predictions)
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
        let provider_name = state.metadata.created_by.clone();
        self.registry.get(&provider_name)?;
        Ok(World::new(state, provider_name, Arc::clone(&self.registry)))
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

    fn state_store(&self) -> Result<&DynStateStore> {
        self.state_store.as_ref().ok_or_else(|| {
            error::WorldForgeError::InvalidState(
                "no state store configured for persistence operations".to_string(),
            )
        })
    }
}

impl Default for WorldForge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worldforge_new() {
        let wf = WorldForge::new();
        assert!(wf.providers().is_empty());
    }

    #[test]
    fn test_create_world_without_provider() {
        let wf = WorldForge::new();
        let result = wf.create_world("test", "nonexistent");
        assert!(result.is_err());
    }
}
