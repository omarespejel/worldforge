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
use provider::ProviderRegistry;
use state::WorldState;
use world::World;

/// The main entry point for WorldForge.
///
/// Manages provider registration and world creation.
pub struct WorldForge {
    /// Provider registry.
    registry: Arc<ProviderRegistry>,
}

impl WorldForge {
    /// Create a new WorldForge instance.
    pub fn new() -> Self {
        Self {
            registry: Arc::new(ProviderRegistry::new()),
        }
    }

    /// Register a world model provider.
    pub fn register_provider(&mut self, provider: Box<dyn provider::WorldModelProvider>) {
        Arc::get_mut(&mut self.registry)
            .expect("cannot register provider while worlds are active")
            .register(provider);
    }

    /// List all registered provider names.
    pub fn providers(&self) -> Vec<&str> {
        self.registry.list()
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

    /// Get a reference to the provider registry.
    pub fn registry(&self) -> &ProviderRegistry {
        &self.registry
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
