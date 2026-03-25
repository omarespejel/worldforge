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
//! - [`genie`] — Google Genie (deterministic local surrogate for prediction, reasoning, transfer, and native planning)

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
use worldforge_core::state::DynStateStore;
use worldforge_core::WorldForge;

/// Auto-detect available providers from environment variables.
///
/// Checks for:
/// - `NVIDIA_API_KEY` → registers a full-stack `CosmosProvider`
/// - `RUNWAY_API_SECRET` → registers a full-stack `RunwayProvider`
/// - `JEPA_MODEL_PATH` → registers `JepaProvider`
/// - `JEPA_BACKEND` → optional backend override (`burn`, `pytorch`, `onnx`, `safetensors`)
/// - `GENIE_API_KEY` → registers `GenieProvider` (Genie 3 surrogate + future remote hint)
///
/// A `MockProvider` is always registered for testing.
/// The auto-detected Cosmos and Runway entries are capability-complete and
/// preserve vendor-wide predict/generate/reason/transfer coverage under the
/// stable `"cosmos"` and `"runway"` names.
/// The Genie surrogate currently supports `predict`, `generate`, `reason`,
/// `transfer`, and provider-native planning through the local deterministic
/// backend.
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
        registry.register(Box::new(CosmosProvider::full_stack(api_key, endpoint)));
    }

    // Runway: requires RUNWAY_API_SECRET
    if let Ok(api_secret) = std::env::var("RUNWAY_API_SECRET") {
        registry.register(Box::new(RunwayProvider::full_stack(api_secret)));
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

/// Build a `WorldForge` instance backed by the auto-detected provider registry.
///
/// This keeps provider discovery in the providers crate while exposing an
/// ergonomic Rust entry point that is immediately usable with the detected
/// adapters.
pub fn auto_detect_worldforge() -> WorldForge {
    WorldForge::from_registry(auto_detect())
}

/// Build a `WorldForge` instance with auto-detected providers and a state store.
pub fn auto_detect_worldforge_with_state_store(store: DynStateStore) -> WorldForge {
    WorldForge::from_registry_with_state_store(auto_detect(), store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_lock() -> &'static Mutex<()> {
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        previous: Vec<(String, Option<std::ffi::OsString>)>,
    }

    impl EnvVarGuard {
        fn new(vars: &[(&str, &str)]) -> Self {
            let previous = vars
                .iter()
                .map(|(name, _)| ((*name).to_string(), std::env::var_os(name)))
                .collect();

            for (name, value) in vars {
                std::env::set_var(name, value);
            }

            Self { previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            for (name, previous_value) in self.previous.drain(..) {
                match previous_value {
                    Some(value) => std::env::set_var(&name, value),
                    None => std::env::remove_var(&name),
                }
            }
        }
    }

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

    #[test]
    fn test_auto_detect_registers_full_stack_cosmos() {
        let _guard = env_lock().lock().unwrap();
        let _env = EnvVarGuard::new(&[
            ("NVIDIA_API_KEY", "cosmos-test-key"),
            ("NVIDIA_API_ENDPOINT", "https://example.invalid/cosmos"),
        ]);

        let registry = auto_detect();
        let capabilities = registry.get("cosmos").unwrap().capabilities();

        assert!(capabilities.predict);
        assert!(capabilities.generate);
        assert!(capabilities.reason);
        assert!(capabilities.transfer);
    }

    #[test]
    fn test_auto_detect_registers_full_stack_runway() {
        let _guard = env_lock().lock().unwrap();
        let _env = EnvVarGuard::new(&[("RUNWAY_API_SECRET", "runway-test-secret")]);

        let registry = auto_detect();
        let capabilities = registry.get("runway").unwrap().capabilities();

        assert!(capabilities.predict);
        assert!(capabilities.generate);
        assert!(capabilities.transfer);
        assert!(capabilities.action_conditioned);
        assert!(capabilities.multi_view);
    }
}
