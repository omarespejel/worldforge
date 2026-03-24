//! WorldForge Provider Adapters
//!
//! Concrete implementations of the `WorldModelProvider` trait for
//! various world foundation models, plus a mock provider for testing.
//!
//! # Providers
//!
//! - [`mock`] — Deterministic mock for testing and development
//! - [`cosmos`] — NVIDIA Cosmos (Predict, Transfer, Reason, Embed)
//! - [`runway`] — Runway GWM (Worlds, Robotics, Avatars)
//! - [`jepa`] — Meta JEPA (local deterministic inference, ZK-compatible)
//! - [`genie`] — Google Genie (deterministic local surrogate for predict/generate)

pub mod cosmos;
pub mod genie;
pub mod jepa;
pub mod mock;
pub mod runway;

pub use cosmos::CosmosProvider;
pub use genie::GenieProvider;
pub use jepa::{JepaBackend, JepaModelManifest, JepaProvider};
pub use mock::MockProvider;
pub use runway::RunwayProvider;

use std::path::PathBuf;

use worldforge_core::provider::ProviderRegistry;

/// Auto-detect available providers from environment variables.
///
/// Checks for:
/// - `NVIDIA_API_KEY` → registers `CosmosProvider` (Predict 2.5)
/// - `RUNWAY_API_SECRET` → registers `RunwayProvider` (GWM-1 Worlds)
/// - `JEPA_MODEL_PATH` → registers `JepaProvider`
/// - `JEPA_BACKEND` → optional backend override (`burn`, `pytorch`, `onnx`, `safetensors`)
/// - `GENIE_API_KEY` → registers `GenieProvider` (Genie 3 surrogate + future remote hint)
///
/// A `MockProvider` is always registered for testing.
/// The Genie surrogate currently supports `predict` and `generate`; `reason`,
/// `transfer`, and native planning remain unsupported.
///
/// # Examples
///
/// ```
/// use worldforge_providers::auto_detect;
/// let registry = auto_detect();
/// // Mock provider is always available
/// assert!(registry.get("mock").is_ok());
/// ```
pub fn auto_detect() -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();

    // Mock is always available
    registry.register(Box::new(MockProvider::new()));

    // Cosmos: requires NVIDIA_API_KEY
    if let Ok(api_key) = std::env::var("NVIDIA_API_KEY") {
        let endpoint = std::env::var("NVIDIA_API_ENDPOINT")
            .map(cosmos::CosmosEndpoint::NimApi)
            .unwrap_or_else(|_| {
                cosmos::CosmosEndpoint::NimApi("https://ai.api.nvidia.com".to_string())
            });
        registry.register(Box::new(CosmosProvider::new(
            cosmos::CosmosModel::Predict2_5,
            api_key,
            endpoint,
        )));
    }

    // Runway: requires RUNWAY_API_SECRET
    if let Ok(api_secret) = std::env::var("RUNWAY_API_SECRET") {
        registry.register(Box::new(RunwayProvider::new(
            runway::RunwayModel::Gwm1Worlds,
            api_secret,
        )));
    }

    // JEPA: requires JEPA_MODEL_PATH pointing to model weights
    if let Ok(model_path) = std::env::var("JEPA_MODEL_PATH") {
        let path = PathBuf::from(&model_path);
        let backend = std::env::var("JEPA_BACKEND")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(jepa::JepaBackend::Burn);
        registry.register(Box::new(JepaProvider::new(path, backend)));
    }

    // Genie: optional credentials act as a future remote hint, but the
    // local surrogate is what actually powers the adapter today.
    if let Ok(api_key) = std::env::var("GENIE_API_KEY") {
        registry.register(Box::new(GenieProvider::new(
            genie::GenieModel::Genie3,
            api_key,
        )));
    }

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect_always_has_mock() {
        let registry = auto_detect();
        assert!(registry.get("mock").is_ok());
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_auto_detect_no_env_vars() {
        // Without env vars set, only mock should be registered
        // (We can't guarantee env vars aren't set in CI, so just
        // verify mock is present)
        let registry = auto_detect();
        assert!(registry.get("mock").is_ok());
    }
}
