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
//! - [`jepa`] — Meta JEPA (local deterministic inference, reasoning, embeddings, and native planning)
//! - [`genie`] — Google Genie (deterministic local surrogate for prediction, reasoning, transfer, depth/segmentation outputs, and native planning)
//! - [`marble`] — Experimental deterministic local surrogate for Marble with native planning
//! - [`native_planning`] — shared deterministic adapter-native planning helper

pub mod cosmos;
pub mod genie;
pub mod jepa;
pub mod marble;
pub mod mock;
mod native_planning;
pub mod runway;

/// Backward-compatible Cosmos action translator helper.
pub use cosmos::CosmosActionTranslator;
pub use cosmos::CosmosProvider;
pub use genie::GenieProvider;
pub use jepa::{JepaBackend, JepaModelManifest, JepaProvider};
pub use marble::MarbleProvider;
pub use mock::MockProvider;
/// Backward-compatible Runway action translator helper.
pub use runway::RunwayActionTranslator;
pub use runway::RunwayProvider;

use std::path::PathBuf;

use worldforge_core::provider::ProviderRegistry;
use worldforge_core::state::DynStateStore;
use worldforge_core::WorldForge;

/// Auto-detect available providers from environment variables.
///
/// Checks for:
/// - `NVIDIA_API_KEY` → registers a full-stack `CosmosProvider`
/// - `RUNWAY_API_SECRET` → registers a `RunwayProvider`
/// - `JEPA_MODEL_PATH` → registers `JepaProvider`
/// - `JEPA_BACKEND` → optional backend override (`burn`, `pytorch`, `onnx`, `safetensors`)
/// - `GENIE_API_KEY` → optional credential hint for `GenieProvider`
/// - `GENIE_API_ENDPOINT` → optional endpoint override for `GenieProvider`
/// - `MarbleProvider` → always registered as an experimental local surrogate with native planning
///
/// A `MockProvider` is always registered for testing.
/// The auto-detected Cosmos entry is capability-complete for its documented
/// surface: predict/generate/reason/transfer/embed plus adapter-native
/// planning under the stable `"cosmos"` name. The Runway entry exposes
/// predict/generate/transfer plus adapter-native planning under the stable
/// `"runway"` name, and adds Cosmos-backed reasoning when both provider
/// credentials are available.
/// The JEPA adapter currently supports local `predict`, `reason`, `embed`,
/// and provider-native planning through inspected model assets. The Genie
/// surrogate currently supports `predict`, `generate`, `reason`, `transfer`,
/// depth/segmentation prediction outputs, and provider-native planning through
/// the local deterministic backend.
/// Marble is an experimental deterministic local surrogate that registers by
/// default and exposes prediction, generation, reasoning, planning, transfer,
/// and embedding capabilities without a remote transport.
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

    let cosmos_reason_fallback = if let Ok(api_key) = std::env::var("NVIDIA_API_KEY") {
        let endpoint = std::env::var("NVIDIA_API_ENDPOINT")
            .map(cosmos::CosmosEndpoint::NimApi)
            .unwrap_or_else(|_| {
                cosmos::CosmosEndpoint::NimApi("https://ai.api.nvidia.com".to_string())
            });
        let reason_fallback = cosmos::CosmosProvider::new(
            cosmos::CosmosModel::Reason2,
            api_key.clone(),
            endpoint.clone(),
        );
        registry.register(Box::new(cosmos::CosmosProvider::full_stack(
            api_key, endpoint,
        )));
        Some(reason_fallback)
    } else {
        None
    };

    // Cosmos-backed reasoning fallback requires NVIDIA_API_KEY.
    if let Ok(api_secret) = std::env::var("RUNWAY_API_SECRET") {
        let runway = match cosmos_reason_fallback {
            Some(reason_fallback) => RunwayProvider::full_stack_with_reason_fallback(
                api_secret,
                "https://api.runwayml.com",
                reason_fallback,
            ),
            None => RunwayProvider::full_stack(api_secret),
        };
        registry.register(Box::new(runway));
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

    // Genie is always available as a deterministic local surrogate.
    // Environment variables only provide optional remote hints/overrides.
    let genie_api_key = std::env::var("GENIE_API_KEY").unwrap_or_default();
    let genie = match std::env::var("GENIE_API_ENDPOINT") {
        Ok(endpoint) => {
            GenieProvider::with_endpoint(genie::GenieModel::Genie3, genie_api_key, endpoint)
        }
        Err(_) => GenieProvider::new(genie::GenieModel::Genie3, genie_api_key),
    };
    registry.register(Box::new(genie));

    registry.register(Box::new(MarbleProvider::new()));

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
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::sync::{Mutex, OnceLock};
    use std::thread;
    use std::time::Duration;

    use worldforge_core::provider::ReasoningInput;
    use worldforge_core::state::WorldState;

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_lock() -> &'static Mutex<()> {
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        previous: Vec<(String, Option<std::ffi::OsString>)>,
    }

    impl EnvVarGuard {
        fn set(vars: &[(&str, &str)]) -> Self {
            let previous = vars
                .iter()
                .map(|(name, _)| ((*name).to_string(), std::env::var_os(name)))
                .collect();

            for (name, value) in vars {
                std::env::set_var(name, value);
            }

            Self { previous }
        }

        fn clear(vars: &[&str]) -> Self {
            let previous = vars
                .iter()
                .map(|name| ((*name).to_string(), std::env::var_os(name)))
                .collect();

            for name in vars {
                std::env::remove_var(name);
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

    fn spawn_response_server(
        response_body: String,
    ) -> (
        String,
        mpsc::Receiver<(String, String, String)>,
        thread::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());

            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let request_line = request_line.trim_end_matches(['\r', '\n']);
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let path = parts.next().unwrap_or_default().to_string();

            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let line = line.trim_end_matches(['\r', '\n']);
                if line.is_empty() {
                    break;
                }

                if let Some((name, value)) = line.split_once(':') {
                    if name.trim().eq_ignore_ascii_case("content-length") {
                        content_length = value.trim().parse().unwrap_or(0);
                    }
                }
            }

            let mut body_bytes = vec![0u8; content_length];
            if content_length > 0 {
                reader.read_exact(&mut body_bytes).unwrap();
            }

            let body = String::from_utf8(body_bytes).unwrap_or_default();
            tx.send((method, path, body)).unwrap();

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        (format!("http://{}", addr), rx, handle)
    }

    #[test]
    fn test_auto_detect_always_has_mock() {
        let registry = auto_detect();
        assert!(registry.get("mock").is_ok());
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_auto_detect_no_env_vars() {
        let _guard = env_lock().lock().unwrap();
        let _env = EnvVarGuard::clear(&[
            "NVIDIA_API_KEY",
            "NVIDIA_API_ENDPOINT",
            "RUNWAY_API_SECRET",
            "JEPA_MODEL_PATH",
            "JEPA_BACKEND",
            "GENIE_API_KEY",
            "GENIE_API_ENDPOINT",
        ]);

        let registry = auto_detect();
        assert!(registry.get("mock").is_ok());
        assert!(registry.get("genie").is_ok());
        assert!(registry.get("marble").is_ok());
        assert!(registry
            .find_by_capability("reason")
            .iter()
            .any(|provider| provider.name() == "genie"));
    }

    #[test]
    fn test_auto_detect_registers_marble_local_surrogate() {
        let registry = auto_detect();
        let capabilities = registry.get("marble").unwrap().capabilities();

        assert!(capabilities.predict);
        assert!(capabilities.generate);
        assert!(capabilities.reason);
        assert!(capabilities.transfer);
        assert!(capabilities.embed);
        assert!(capabilities.action_conditioned);
        assert!(capabilities.supports_depth);
        assert!(capabilities.supports_segmentation);
        assert!(capabilities.supports_planning);
        assert!(registry
            .find_by_capability("embed")
            .iter()
            .any(|provider| provider.name() == "marble"));
    }

    #[test]
    fn test_auto_detect_registers_genie_depth_outputs() {
        let registry = auto_detect();
        let capabilities = registry.get("genie").unwrap().capabilities();

        assert!(capabilities.predict);
        assert!(capabilities.generate);
        assert!(capabilities.reason);
        assert!(capabilities.transfer);
        assert!(capabilities.supports_depth);
        assert!(capabilities.supports_segmentation);
    }

    #[test]
    fn test_auto_detect_registers_jepa_reasoning_and_embeddings() {
        let _guard = env_lock().lock().unwrap();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let model_dir = std::env::temp_dir().join(format!(
            "worldforge-jepa-auto-detect-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.safetensors"), b"jepa-weights").unwrap();
        std::fs::write(
            model_dir.join("worldforge-jepa.json"),
            r#"{
                "model_name": "vjepa2-local",
                "representation_dim": 1536,
                "action_gain": 1.2,
                "temporal_smoothness": 0.88,
                "gravity_bias": 0.92,
                "collision_bias": 0.9,
                "confidence_bias": 0.07
            }"#,
        )
        .unwrap();
        let model_dir_string = model_dir.to_string_lossy().to_string();
        let _env = EnvVarGuard::set(&[
            ("JEPA_MODEL_PATH", &model_dir_string),
            ("JEPA_BACKEND", "burn"),
        ]);

        let registry = auto_detect();
        let capabilities = registry.get("jepa").unwrap().capabilities();

        assert!(capabilities.predict);
        assert!(capabilities.reason);
        assert!(capabilities.embed);
        assert!(capabilities.supports_planning);
        assert!(registry
            .find_by_capability("reason")
            .iter()
            .any(|provider| provider.name() == "jepa"));
        assert!(registry
            .find_by_capability("embed")
            .iter()
            .any(|provider| provider.name() == "jepa"));

        let _ = std::fs::remove_dir_all(&model_dir);
    }

    #[test]
    fn test_auto_detect_registers_full_stack_cosmos() {
        let _guard = env_lock().lock().unwrap();
        let _env = EnvVarGuard::set(&[
            ("NVIDIA_API_KEY", "cosmos-test-key"),
            ("NVIDIA_API_ENDPOINT", "https://example.invalid/cosmos"),
        ]);

        let registry = auto_detect();
        let capabilities = registry.get("cosmos").unwrap().capabilities();

        assert!(capabilities.predict);
        assert!(capabilities.generate);
        assert!(capabilities.reason);
        assert!(capabilities.transfer);
        assert!(capabilities.embed);

        let embed_providers = registry.find_by_capability("embed");
        assert!(embed_providers
            .iter()
            .any(|provider| provider.name() == "mock"));
        assert!(embed_providers
            .iter()
            .any(|provider| provider.name() == "cosmos"));
    }

    #[test]
    fn test_auto_detect_registers_full_stack_runway() {
        let _guard = env_lock().lock().unwrap();
        let _env = EnvVarGuard::set(&[("RUNWAY_API_SECRET", "runway-test-secret")]);

        let registry = auto_detect();
        let capabilities = registry.get("runway").unwrap().capabilities();

        assert!(capabilities.predict);
        assert!(capabilities.generate);
        assert!(capabilities.transfer);
        assert!(capabilities.action_conditioned);
        assert!(capabilities.multi_view);
        assert!(!capabilities.embed);
    }

    #[test]
    fn test_auto_detect_registers_runway_reason_fallback_when_cosmos_available() {
        let _guard = env_lock().lock().unwrap();
        let (reason_endpoint, request_rx, handle) = spawn_response_server(
            serde_json::json!({
                "request_id": "reason-route",
                "answer": "The scene is stable.",
                "confidence": 0.93,
                "evidence": ["cosmos-backed reasoning"]
            })
            .to_string(),
        );
        let _env = EnvVarGuard::set(&[
            ("NVIDIA_API_KEY", "cosmos-test-key"),
            ("NVIDIA_API_ENDPOINT", &reason_endpoint),
            ("RUNWAY_API_SECRET", "runway-test-secret"),
        ]);

        let registry = auto_detect();
        let capabilities = registry.get("runway").unwrap().capabilities();
        assert!(capabilities.reason);
        assert!(registry
            .find_by_capability("reason")
            .iter()
            .any(|provider| provider.name() == "runway"));

        let provider = registry.get("runway").unwrap();
        let input = ReasoningInput {
            video: None,
            state: Some(WorldState::new("auto-detect-reason", "runway")),
        };

        let output = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(provider.reason(&input, "Is the scene stable?"))
            .unwrap();
        let (method, path, body) = request_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        handle.join().unwrap();

        assert_eq!(method, "POST");
        assert_eq!(path, "/v1/reason");
        assert!(body.contains(r#""model":"nvidia/cosmos-reason-2""#));
        assert_eq!(output.answer, "The scene is stable.");
        assert!(output.confidence > 0.9);
        assert_eq!(output.evidence, vec!["cosmos-backed reasoning"]);
    }
}
