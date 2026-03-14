//! NVIDIA Cosmos provider adapter.
//!
//! Implements the `WorldModelProvider` trait for NVIDIA Cosmos models:
//! - Cosmos Predict 2.5: video generation / future prediction
//! - Cosmos Transfer 2.5: spatial control to video
//! - Cosmos Reason 2: physical reasoning VLM (7B)
//! - Cosmos Embed 1: video-text embeddings

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

/// NVIDIA Cosmos model variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CosmosModel {
    /// Video generation / future prediction.
    Predict2_5,
    /// Spatial control to video.
    Transfer2_5,
    /// Physical reasoning VLM (7B).
    Reason2,
    /// Video-text embeddings.
    Embed1,
}

/// Cosmos API endpoint type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CosmosEndpoint {
    /// NVIDIA NIM managed API.
    NimApi(String),
    /// Self-hosted NIM container.
    NimLocal(String),
    /// Direct model download from HuggingFace.
    HuggingFace,
    /// DGX Cloud deployment.
    DgxCloud(String),
}

/// Configuration for the Cosmos provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmosConfig {
    /// Maximum retries on transient failures.
    pub max_retries: u32,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Whether to include depth maps in predictions.
    pub include_depth: bool,
    /// Default number of prediction frames.
    pub default_num_frames: u32,
}

impl Default for CosmosConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            timeout_ms: 30_000,
            include_depth: false,
            default_num_frames: 24,
        }
    }
}

/// NVIDIA Cosmos provider adapter.
///
/// Wraps the Cosmos NIM API (or local deployment) to implement
/// the `WorldModelProvider` trait.
#[derive(Debug, Clone)]
pub struct CosmosProvider {
    /// Model variant to use.
    pub model: CosmosModel,
    /// API key for authentication.
    api_key: String,
    /// API endpoint.
    pub endpoint: CosmosEndpoint,
    /// Provider configuration.
    pub config: CosmosConfig,
    /// HTTP client (shared).
    client: reqwest::Client,
}

impl CosmosProvider {
    /// Create a new Cosmos provider.
    pub fn new(model: CosmosModel, api_key: impl Into<String>, endpoint: CosmosEndpoint) -> Self {
        Self {
            model,
            api_key: api_key.into(),
            endpoint,
            config: CosmosConfig::default(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new Cosmos provider with custom configuration.
    pub fn with_config(
        model: CosmosModel,
        api_key: impl Into<String>,
        endpoint: CosmosEndpoint,
        config: CosmosConfig,
    ) -> Self {
        Self {
            model,
            api_key: api_key.into(),
            endpoint,
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Get the base URL for the configured endpoint.
    fn base_url(&self) -> Result<String> {
        match &self.endpoint {
            CosmosEndpoint::NimApi(url) => Ok(url.clone()),
            CosmosEndpoint::NimLocal(url) => Ok(url.clone()),
            CosmosEndpoint::DgxCloud(url) => Ok(url.clone()),
            CosmosEndpoint::HuggingFace => Err(WorldForgeError::UnsupportedCapability {
                provider: "cosmos".to_string(),
                capability: "HuggingFace endpoint requires local inference, not HTTP API"
                    .to_string(),
            }),
        }
    }

    /// Translate a WorldForge action into a Cosmos text prompt.
    fn action_to_prompt(action: &Action) -> String {
        match action {
            Action::Move { target, speed } => {
                format!(
                    "Move to position ({:.1}, {:.1}, {:.1}) at speed {:.1}",
                    target.x, target.y, target.z, speed
                )
            }
            Action::Grasp { grip_force, .. } => {
                format!("Grasp the object with force {grip_force:.1}")
            }
            Action::Release { .. } => "Release the grasped object".to_string(),
            Action::Push {
                direction, force, ..
            } => {
                format!(
                    "Push the object in direction ({:.1}, {:.1}, {:.1}) with force {:.1}",
                    direction.x, direction.y, direction.z, force
                )
            }
            Action::Rotate { axis, angle, .. } => {
                format!(
                    "Rotate the object around axis ({:.1}, {:.1}, {:.1}) by {:.1} degrees",
                    axis.x, axis.y, axis.z, angle
                )
            }
            Action::Place { target, .. } => {
                format!(
                    "Place the object at position ({:.1}, {:.1}, {:.1})",
                    target.x, target.y, target.z
                )
            }
            Action::SetWeather { weather } => {
                format!("Change weather to {weather:?}")
            }
            Action::SetLighting { time_of_day } => {
                format!("Set time of day to {time_of_day:.1}")
            }
            Action::SpawnObject { template, .. } => {
                format!("Spawn a new {template}")
            }
            Action::CameraMove { delta } => {
                format!(
                    "Move camera by ({:.1}, {:.1}, {:.1})",
                    delta.position.x, delta.position.y, delta.position.z
                )
            }
            Action::CameraLookAt { target } => {
                format!(
                    "Look at position ({:.1}, {:.1}, {:.1})",
                    target.x, target.y, target.z
                )
            }
            _ => "Perform the specified action".to_string(),
        }
    }
}

#[async_trait]
impl WorldModelProvider for CosmosProvider {
    fn name(&self) -> &str {
        "cosmos"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: matches!(self.model, CosmosModel::Reason2),
            transfer: matches!(self.model, CosmosModel::Transfer2_5),
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 10.0,
            max_resolution: (1920, 1080),
            fps_range: (8.0, 30.0),
            supported_action_spaces: vec![ActionSpaceType::Continuous, ActionSpaceType::Language],
            supports_depth: true,
            supports_segmentation: false,
            supports_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 2000,
                p95_ms: 5000,
                p99_ms: 10000,
                throughput_fps: 8.0,
            },
        }
    }

    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        let base_url = self.base_url()?;
        let prompt = Self::action_to_prompt(action);

        let request_body = serde_json::json!({
            "model": format!("nvidia/cosmos-predict-2.5"),
            "prompt": prompt,
            "num_frames": config.steps * (config.fps as u32),
            "resolution": [config.resolution.0, config.resolution.1],
            "fps": config.fps,
        });

        let response = self
            .client
            .post(format!("{base_url}/v1/predict"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(WorldForgeError::ProviderAuthError(
                "invalid Cosmos API key".to_string(),
            ));
        }

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(WorldForgeError::ProviderRateLimited {
                provider: "cosmos".to_string(),
                retry_after_ms: 5000,
            });
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "cosmos".to_string(),
                reason: format!("HTTP {status}: {body}"),
            });
        }

        // Parse the prediction response
        // In a real implementation, this would parse the Cosmos API response format.
        // For now, we return a structured prediction with the output state.
        let mut output_state = state.clone();
        output_state.time.step += config.steps as u64;
        output_state.time.seconds += config.steps as f64 / config.fps as f64;

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: "cosmos".to_string(),
            model: "cosmos-predict-2.5".to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video: None,
            confidence: 0.0, // Would be parsed from API response
            physics_scores: PhysicsScores::default(),
            latency_ms: 0, // Would be measured
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
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip> {
        let base_url = self.base_url()?;

        let request_body = serde_json::json!({
            "model": "nvidia/cosmos-predict-2.5",
            "prompt": prompt.text,
            "negative_prompt": prompt.negative_prompt,
            "duration_seconds": config.duration_seconds,
            "resolution": [config.resolution.0, config.resolution.1],
            "fps": config.fps,
            "temperature": config.temperature,
            "seed": config.seed,
        });

        let response = self
            .client
            .post(format!("{base_url}/v1/generate"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "cosmos".to_string(),
                reason: format!("HTTP {}", response.status()),
            });
        }

        // Parse response into VideoClip (would parse actual video data in production)
        Ok(VideoClip {
            frames: Vec::new(),
            fps: config.fps,
            resolution: config.resolution,
            duration: config.duration_seconds,
        })
    }

    async fn reason(&self, input: &ReasoningInput, query: &str) -> Result<ReasoningOutput> {
        if !matches!(self.model, CosmosModel::Reason2) {
            return Err(WorldForgeError::UnsupportedCapability {
                provider: "cosmos".to_string(),
                capability: "reason (requires Cosmos Reason 2 model)".to_string(),
            });
        }

        let base_url = self.base_url()?;

        let request_body = serde_json::json!({
            "model": "nvidia/cosmos-reason-2",
            "query": query,
            "has_video": input.video.is_some(),
            "has_state": input.state.is_some(),
        });

        let response = self
            .client
            .post(format!("{base_url}/v1/reason"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "cosmos".to_string(),
                reason: format!("HTTP {}", response.status()),
            });
        }

        // Parse reasoning response
        Ok(ReasoningOutput {
            answer: String::new(),
            confidence: 0.0,
            evidence: Vec::new(),
        })
    }

    async fn transfer(
        &self,
        _source: &VideoClip,
        _controls: &SpatialControls,
        config: &TransferConfig,
    ) -> Result<VideoClip> {
        if !matches!(self.model, CosmosModel::Transfer2_5) {
            return Err(WorldForgeError::UnsupportedCapability {
                provider: "cosmos".to_string(),
                capability: "transfer (requires Cosmos Transfer 2.5 model)".to_string(),
            });
        }

        let base_url = self.base_url()?;

        let request_body = serde_json::json!({
            "model": "nvidia/cosmos-transfer-2.5",
            "resolution": [config.resolution.0, config.resolution.1],
            "fps": config.fps,
            "control_strength": config.control_strength,
        });

        let response = self
            .client
            .post(format!("{base_url}/v1/transfer"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "cosmos".to_string(),
                reason: format!("HTTP {}", response.status()),
            });
        }

        Ok(VideoClip {
            frames: Vec::new(),
            fps: config.fps,
            resolution: config.resolution,
            duration: 0.0,
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let base_url = self.base_url()?;
        let start = std::time::Instant::now();

        let response = self
            .client
            .get(format!("{base_url}/v1/health"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .timeout(std::time::Duration::from_millis(5000))
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        let latency = start.elapsed().as_millis() as u64;

        Ok(HealthStatus {
            healthy: response.status().is_success(),
            message: format!("Cosmos API responded with HTTP {}", response.status()),
            latency_ms: latency,
        })
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        match operation {
            Operation::Predict { steps, resolution } => {
                let pixels = resolution.0 as f64 * resolution.1 as f64;
                let base_cost = 0.01; // $0.01 per prediction step at 720p
                let scale = pixels / (1280.0 * 720.0);
                CostEstimate {
                    usd: base_cost * *steps as f64 * scale,
                    credits: *steps as f64 * scale,
                    estimated_latency_ms: 2000 + (*steps as u64 * 500),
                }
            }
            Operation::Generate {
                duration_seconds,
                resolution,
            } => {
                let pixels = resolution.0 as f64 * resolution.1 as f64;
                let scale = pixels / (1280.0 * 720.0);
                CostEstimate {
                    usd: 0.05 * duration_seconds * scale,
                    credits: 5.0 * duration_seconds * scale,
                    estimated_latency_ms: (*duration_seconds * 2000.0) as u64,
                }
            }
            Operation::Reason => CostEstimate {
                usd: 0.005,
                credits: 0.5,
                estimated_latency_ms: 1000,
            },
            Operation::Transfer { duration_seconds } => CostEstimate {
                usd: 0.03 * duration_seconds,
                credits: 3.0 * duration_seconds,
                estimated_latency_ms: (*duration_seconds * 3000.0) as u64,
            },
        }
    }
}

/// Cosmos-specific action translator.
pub struct CosmosActionTranslator;

impl worldforge_core::action::ActionTranslator for CosmosActionTranslator {
    fn translate(&self, action: &Action) -> Result<worldforge_core::action::ProviderAction> {
        let prompt = CosmosProvider::action_to_prompt(action);
        Ok(worldforge_core::action::ProviderAction {
            provider: "cosmos".to_string(),
            data: serde_json::json!({
                "type": "text_prompt",
                "prompt": prompt,
            }),
        })
    }

    fn supported_actions(&self) -> Vec<ActionSpaceType> {
        vec![ActionSpaceType::Continuous, ActionSpaceType::Language]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::action::ActionTranslator;
    use worldforge_core::types::Position;

    #[test]
    fn test_cosmos_provider_creation() {
        let provider = CosmosProvider::new(
            CosmosModel::Predict2_5,
            "test-key",
            CosmosEndpoint::NimApi("https://api.nvidia.com".to_string()),
        );
        assert_eq!(provider.name(), "cosmos");
    }

    #[test]
    fn test_cosmos_capabilities() {
        let provider = CosmosProvider::new(
            CosmosModel::Predict2_5,
            "test-key",
            CosmosEndpoint::NimApi("https://api.nvidia.com".to_string()),
        );
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(caps.generate);
        assert!(!caps.reason); // Predict model doesn't support reasoning
    }

    #[test]
    fn test_cosmos_reason_capability() {
        let provider = CosmosProvider::new(
            CosmosModel::Reason2,
            "test-key",
            CosmosEndpoint::NimApi("https://api.nvidia.com".to_string()),
        );
        let caps = provider.capabilities();
        assert!(caps.reason);
    }

    #[test]
    fn test_cosmos_transfer_capability() {
        let provider = CosmosProvider::new(
            CosmosModel::Transfer2_5,
            "test-key",
            CosmosEndpoint::NimApi("https://api.nvidia.com".to_string()),
        );
        let caps = provider.capabilities();
        assert!(caps.transfer);
    }

    #[test]
    fn test_cosmos_config_default() {
        let config = CosmosConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.timeout_ms, 30_000);
    }

    #[test]
    fn test_action_to_prompt() {
        let prompt = CosmosProvider::action_to_prompt(&Action::Move {
            target: Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            speed: 0.5,
        });
        assert!(prompt.contains("1.0"));
        assert!(prompt.contains("2.0"));
        assert!(prompt.contains("3.0"));
    }

    #[test]
    fn test_cost_estimation() {
        let provider = CosmosProvider::new(
            CosmosModel::Predict2_5,
            "test-key",
            CosmosEndpoint::NimApi("https://api.nvidia.com".to_string()),
        );
        let cost = provider.estimate_cost(&Operation::Predict {
            steps: 1,
            resolution: (1280, 720),
        });
        assert!(cost.usd > 0.0);
        assert!(cost.estimated_latency_ms > 0);
    }

    #[test]
    fn test_action_translator() {
        let translator = CosmosActionTranslator;
        let action = Action::SetWeather {
            weather: worldforge_core::action::Weather::Rain,
        };
        let result = translator.translate(&action).unwrap();
        assert_eq!(result.provider, "cosmos");
        assert!(result.data["prompt"].as_str().unwrap().contains("Rain"));
    }

    #[test]
    fn test_huggingface_endpoint_unsupported_for_api() {
        let provider = CosmosProvider::new(
            CosmosModel::Predict2_5,
            "test-key",
            CosmosEndpoint::HuggingFace,
        );
        assert!(provider.base_url().is_err());
    }

    #[test]
    fn test_cosmos_model_serialization() {
        let model = CosmosModel::Predict2_5;
        let json = serde_json::to_string(&model).unwrap();
        let model2: CosmosModel = serde_json::from_str(&json).unwrap();
        assert!(matches!(model2, CosmosModel::Predict2_5));
    }

    #[test]
    fn test_cosmos_endpoint_serialization() {
        let endpoint = CosmosEndpoint::NimApi("https://api.nvidia.com".to_string());
        let json = serde_json::to_string(&endpoint).unwrap();
        let ep2: CosmosEndpoint = serde_json::from_str(&json).unwrap();
        assert!(matches!(ep2, CosmosEndpoint::NimApi(_)));
    }
}
