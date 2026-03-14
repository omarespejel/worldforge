//! Runway GWM provider adapter.
//!
//! Implements the `WorldModelProvider` trait for Runway's General World
//! Models family:
//! - GWM-1 Worlds: explorable environment generation
//! - GWM-1 Robotics: action-conditioned video prediction
//! - GWM-1 Avatars: audio-driven character generation

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

/// Runway model variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunwayModel {
    /// Explorable environment generation.
    Gwm1Worlds,
    /// Action-conditioned robot video prediction.
    Gwm1Robotics,
    /// Audio-driven conversational characters.
    Gwm1Avatars,
}

/// Configuration for the Runway provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunwayConfig {
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum retries on transient failures.
    pub max_retries: u32,
}

impl Default for RunwayConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 60_000,
            max_retries: 3,
        }
    }
}

/// Runway GWM provider adapter.
///
/// Wraps the Runway HTTP API to implement the `WorldModelProvider` trait.
#[derive(Debug, Clone)]
pub struct RunwayProvider {
    /// Model variant to use.
    pub model: RunwayModel,
    /// API secret for authentication.
    api_secret: String,
    /// API endpoint URL.
    pub endpoint: String,
    /// Provider configuration.
    pub config: RunwayConfig,
    /// HTTP client.
    client: reqwest::Client,
}

impl RunwayProvider {
    /// Create a new Runway provider.
    pub fn new(model: RunwayModel, api_secret: impl Into<String>) -> Self {
        Self {
            model,
            api_secret: api_secret.into(),
            endpoint: "https://api.runwayml.com".to_string(),
            config: RunwayConfig::default(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a Runway provider with a custom endpoint.
    pub fn with_endpoint(
        model: RunwayModel,
        api_secret: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Self::new(model, api_secret)
        }
    }

    /// Translate a WorldForge action to Runway's robot command format.
    fn action_to_robot_command(action: &Action) -> serde_json::Value {
        match action {
            Action::Move { target, speed } => {
                serde_json::json!({
                    "type": "move",
                    "target": [target.x, target.y, target.z],
                    "speed": speed,
                })
            }
            Action::Grasp { grip_force, .. } => {
                serde_json::json!({
                    "type": "grasp",
                    "grip_force": grip_force,
                })
            }
            Action::Release { .. } => {
                serde_json::json!({
                    "type": "release",
                })
            }
            Action::Push {
                direction, force, ..
            } => {
                serde_json::json!({
                    "type": "push",
                    "direction": [direction.x, direction.y, direction.z],
                    "force": force,
                })
            }
            Action::Rotate { axis, angle, .. } => {
                serde_json::json!({
                    "type": "rotate",
                    "axis": [axis.x, axis.y, axis.z],
                    "angle": angle,
                })
            }
            Action::Place { target, .. } => {
                serde_json::json!({
                    "type": "place",
                    "target": [target.x, target.y, target.z],
                })
            }
            Action::CameraMove { delta } => {
                serde_json::json!({
                    "type": "camera_move",
                    "position": [delta.position.x, delta.position.y, delta.position.z],
                })
            }
            Action::CameraLookAt { target } => {
                serde_json::json!({
                    "type": "camera_look_at",
                    "target": [target.x, target.y, target.z],
                })
            }
            _ => {
                serde_json::json!({
                    "type": "raw",
                    "action": format!("{action:?}"),
                })
            }
        }
    }
}

#[async_trait]
impl WorldModelProvider for RunwayProvider {
    fn name(&self) -> &str {
        "runway"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let (predict, generate, transfer) = match self.model {
            RunwayModel::Gwm1Worlds => (false, true, true),
            RunwayModel::Gwm1Robotics => (true, false, false),
            RunwayModel::Gwm1Avatars => (false, true, false),
        };

        ProviderCapabilities {
            predict,
            generate,
            reason: false, // Runway does not support reasoning; use Cosmos as fallback
            transfer,
            action_conditioned: matches!(self.model, RunwayModel::Gwm1Robotics),
            multi_view: matches!(self.model, RunwayModel::Gwm1Worlds),
            max_video_length_seconds: 16.0,
            max_resolution: (1920, 1080),
            fps_range: (12.0, 30.0),
            supported_action_spaces: match self.model {
                RunwayModel::Gwm1Robotics => {
                    vec![ActionSpaceType::Continuous, ActionSpaceType::Discrete]
                }
                RunwayModel::Gwm1Worlds => vec![ActionSpaceType::Language, ActionSpaceType::Visual],
                RunwayModel::Gwm1Avatars => vec![ActionSpaceType::Language],
            },
            supports_depth: matches!(self.model, RunwayModel::Gwm1Worlds),
            supports_segmentation: false,
            supports_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 3000,
                p95_ms: 8000,
                p99_ms: 15000,
                throughput_fps: 4.0,
            },
        }
    }

    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        if !matches!(self.model, RunwayModel::Gwm1Robotics) {
            return Err(WorldForgeError::UnsupportedCapability {
                provider: "runway".to_string(),
                capability: "predict (requires GWM-1 Robotics model)".to_string(),
            });
        }

        let command = Self::action_to_robot_command(action);

        let request_body = serde_json::json!({
            "model": "gwm-1-robotics",
            "action": command,
            "num_frames": config.steps * (config.fps as u32),
            "resolution": [config.resolution.0, config.resolution.1],
        });

        let response = self
            .client
            .post(format!("{}/v1/robotics/predict", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_secret))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(WorldForgeError::ProviderAuthError(
                "invalid Runway API secret".to_string(),
            ));
        }

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(WorldForgeError::ProviderRateLimited {
                provider: "runway".to_string(),
                retry_after_ms: 10_000,
            });
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "runway".to_string(),
                reason: format!("HTTP {status}: {body}"),
            });
        }

        let mut output_state = state.clone();
        output_state.time.step += config.steps as u64;
        output_state.time.seconds += config.steps as f64 / config.fps as f64;

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: "runway".to_string(),
            model: "gwm-1-robotics".to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video: None,
            confidence: 0.0,
            physics_scores: PhysicsScores::default(),
            latency_ms: 0,
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
        if !matches!(
            self.model,
            RunwayModel::Gwm1Worlds | RunwayModel::Gwm1Avatars
        ) {
            return Err(WorldForgeError::UnsupportedCapability {
                provider: "runway".to_string(),
                capability: "generate (requires GWM-1 Worlds or Avatars model)".to_string(),
            });
        }

        let request_body = serde_json::json!({
            "model": match self.model {
                RunwayModel::Gwm1Worlds => "gwm-1-worlds",
                RunwayModel::Gwm1Avatars => "gwm-1-avatars",
                _ => "gwm-1-worlds",
            },
            "prompt": prompt.text,
            "negative_prompt": prompt.negative_prompt,
            "duration_seconds": config.duration_seconds,
            "resolution": [config.resolution.0, config.resolution.1],
            "fps": config.fps,
        });

        let response = self
            .client
            .post(format!("{}/v1/worlds/generate", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_secret))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "runway".to_string(),
                reason: format!("HTTP {}", response.status()),
            });
        }

        Ok(VideoClip {
            frames: Vec::new(),
            fps: config.fps,
            resolution: config.resolution,
            duration: config.duration_seconds,
        })
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "runway".to_string(),
            capability: "reason (use Cosmos Reason as fallback)".to_string(),
        })
    }

    async fn transfer(
        &self,
        _source: &VideoClip,
        _controls: &SpatialControls,
        config: &TransferConfig,
    ) -> Result<VideoClip> {
        if !matches!(self.model, RunwayModel::Gwm1Worlds) {
            return Err(WorldForgeError::UnsupportedCapability {
                provider: "runway".to_string(),
                capability: "transfer (requires GWM-1 Worlds model)".to_string(),
            });
        }

        let request_body = serde_json::json!({
            "model": "gwm-1-worlds",
            "resolution": [config.resolution.0, config.resolution.1],
            "fps": config.fps,
            "control_strength": config.control_strength,
        });

        let response = self
            .client
            .post(format!("{}/v1/worlds/transfer", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_secret))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "runway".to_string(),
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
        let start = std::time::Instant::now();

        let response = self
            .client
            .get(format!("{}/v1/health", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_secret))
            .timeout(std::time::Duration::from_millis(5000))
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        let latency = start.elapsed().as_millis() as u64;

        Ok(HealthStatus {
            healthy: response.status().is_success(),
            message: format!("Runway API responded with HTTP {}", response.status()),
            latency_ms: latency,
        })
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        match operation {
            Operation::Predict { steps, resolution } => {
                let pixels = resolution.0 as f64 * resolution.1 as f64;
                let scale = pixels / (1280.0 * 720.0);
                CostEstimate {
                    usd: 0.02 * *steps as f64 * scale,
                    credits: 2.0 * *steps as f64 * scale,
                    estimated_latency_ms: 3000 + (*steps as u64 * 1000),
                }
            }
            Operation::Generate {
                duration_seconds,
                resolution,
            } => {
                let pixels = resolution.0 as f64 * resolution.1 as f64;
                let scale = pixels / (1280.0 * 720.0);
                CostEstimate {
                    usd: 0.10 * duration_seconds * scale,
                    credits: 10.0 * duration_seconds * scale,
                    estimated_latency_ms: (*duration_seconds * 5000.0) as u64,
                }
            }
            Operation::Reason => CostEstimate::default(),
            Operation::Transfer { duration_seconds } => CostEstimate {
                usd: 0.08 * duration_seconds,
                credits: 8.0 * duration_seconds,
                estimated_latency_ms: (*duration_seconds * 4000.0) as u64,
            },
        }
    }
}

/// Runway-specific action translator for GWM-1 Robotics.
pub struct RunwayActionTranslator;

impl worldforge_core::action::ActionTranslator for RunwayActionTranslator {
    fn translate(&self, action: &Action) -> Result<worldforge_core::action::ProviderAction> {
        let command = RunwayProvider::action_to_robot_command(action);
        Ok(worldforge_core::action::ProviderAction {
            provider: "runway".to_string(),
            data: command,
        })
    }

    fn supported_actions(&self) -> Vec<ActionSpaceType> {
        vec![ActionSpaceType::Continuous, ActionSpaceType::Discrete]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::action::ActionTranslator;
    use worldforge_core::types::Position;

    #[test]
    fn test_runway_provider_creation() {
        let provider = RunwayProvider::new(RunwayModel::Gwm1Robotics, "test-secret");
        assert_eq!(provider.name(), "runway");
    }

    #[test]
    fn test_runway_robotics_capabilities() {
        let provider = RunwayProvider::new(RunwayModel::Gwm1Robotics, "test-secret");
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(!caps.generate);
        assert!(!caps.reason);
        assert!(caps.action_conditioned);
    }

    #[test]
    fn test_runway_worlds_capabilities() {
        let provider = RunwayProvider::new(RunwayModel::Gwm1Worlds, "test-secret");
        let caps = provider.capabilities();
        assert!(!caps.predict);
        assert!(caps.generate);
        assert!(caps.transfer);
        assert!(caps.multi_view);
        assert!(caps.supports_depth);
    }

    #[test]
    fn test_runway_avatars_capabilities() {
        let provider = RunwayProvider::new(RunwayModel::Gwm1Avatars, "test-secret");
        let caps = provider.capabilities();
        assert!(!caps.predict);
        assert!(caps.generate);
        assert!(!caps.transfer);
    }

    #[test]
    fn test_runway_config_default() {
        let config = RunwayConfig::default();
        assert_eq!(config.timeout_ms, 60_000);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_action_to_robot_command() {
        let cmd = RunwayProvider::action_to_robot_command(&Action::Move {
            target: Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            speed: 0.5,
        });
        assert_eq!(cmd["type"], "move");
        assert_eq!(cmd["speed"], 0.5);
    }

    #[test]
    fn test_action_translator() {
        let translator = RunwayActionTranslator;
        let action = Action::Grasp {
            object: uuid::Uuid::new_v4(),
            grip_force: 5.0,
        };
        let result = translator.translate(&action).unwrap();
        assert_eq!(result.provider, "runway");
        assert_eq!(result.data["type"], "grasp");
    }

    #[test]
    fn test_cost_estimation() {
        let provider = RunwayProvider::new(RunwayModel::Gwm1Robotics, "test-secret");
        let cost = provider.estimate_cost(&Operation::Predict {
            steps: 1,
            resolution: (1280, 720),
        });
        assert!(cost.usd > 0.0);
    }

    #[test]
    fn test_runway_model_serialization() {
        let model = RunwayModel::Gwm1Robotics;
        let json = serde_json::to_string(&model).unwrap();
        let model2: RunwayModel = serde_json::from_str(&json).unwrap();
        assert!(matches!(model2, RunwayModel::Gwm1Robotics));
    }

    #[test]
    fn test_custom_endpoint() {
        let provider =
            RunwayProvider::with_endpoint(RunwayModel::Gwm1Worlds, "secret", "http://localhost");
        assert_eq!(provider.endpoint, "http://localhost");
    }
}
