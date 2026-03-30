//! MBZUAI PAN provider adapter.
//!
//! Implements the `WorldModelProvider` trait for the MBZUAI PAN (Perception
//! and Action Network) world model. PAN is a stateful multi-round generation
//! model that maintains server-side state across generation rounds, making it
//! especially capable for planning tasks.
//!
//! # API Flow
//!
//! PAN uses a unique stateful multi-round API:
//!
//! 1. **Round 1:** `POST <endpoint>/first_round` with `{ prompt, image_path, state_id }`
//!    - Returns `{ frames, state_id, video_id }`
//! 2. **Round N:** `POST <endpoint>/continue` with `{ prompt, state_id, video_id }`
//!    - Returns `{ frames, state_id, video_id }`
//!
//! The server maintains internal state across rounds, so frames do not need
//! to be sent back.
//!
//! # Prompt Upsampling
//!
//! PAN supports server-side prompt enrichment (prompt upsampling). All
//! prompts are prefixed with `"FPS-{fps} "` before sending.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use worldforge_core::action::{Action, ActionSpaceType, ActionType};
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::prediction::{PhysicsScores, Plan, PlanRequest, Prediction, PredictionConfig};
use worldforge_core::provider::{
    CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
    ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
    WorldModelProvider,
};
use worldforge_core::state::WorldState;
use worldforge_core::types::VideoClip;

use crate::native_planning;
use crate::polling::{build_stub_video_clip, check_http_response};

/// Default PAN API endpoint.
const DEFAULT_ENDPOINT: &str = "https://ifm.mbzuai.ac.ae/pan";

/// Default output resolution `(width, height)`.
const DEFAULT_RESOLUTION: (u32, u32) = (832, 480);

/// Number of frames per generation round.
const FRAMES_PER_ROUND: u32 = 41;

/// Maximum video duration in seconds.
const MAX_VIDEO_LENGTH: f64 = 5.0;

/// Default FPS for generation.
const DEFAULT_FPS: f32 = 24.0;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// Request body for the PAN first-round endpoint.
#[derive(Debug, Serialize)]
struct PanFirstRoundRequest {
    /// Text prompt describing the desired scene or action.
    prompt: String,
    /// Path or URL to the initial image.
    image_path: String,
    /// Unique identifier for this generation session.
    state_id: String,
}

/// Request body for the PAN continuation endpoint.
#[derive(Debug, Serialize)]
struct PanContinueRequest {
    /// Text prompt describing the next action or scene progression.
    prompt: String,
    /// Session state identifier from a prior round.
    state_id: String,
    /// Video identifier from a prior round.
    video_id: String,
}

/// Response from both PAN endpoints.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct PanResponse {
    /// Generated frame data (opaque per the API).
    #[serde(default)]
    pub frames: Vec<serde_json::Value>,
    /// Server-side state identifier to use in subsequent rounds.
    pub state_id: String,
    /// Video identifier to use in subsequent rounds.
    pub video_id: String,
    /// Optional status message from the server.
    #[serde(default)]
    pub status: Option<String>,
    /// Optional error message from the server.
    #[serde(default)]
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Session tracking
// ---------------------------------------------------------------------------

/// Tracks the server-side state for a multi-round PAN generation session.
#[derive(Debug, Clone)]
pub struct PanSession {
    /// Server-side state identifier.
    pub state_id: String,
    /// Server-side video identifier.
    pub video_id: String,
    /// Current round number (1-based).
    pub round: u32,
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// MBZUAI PAN world model provider.
///
/// PAN maintains server-side state across generation rounds, making it
/// uniquely suited for multi-step planning tasks. The provider tracks
/// active sessions locally via an `Arc<Mutex<HashMap>>`.
#[derive(Debug, Clone)]
pub struct PanProvider {
    /// Bearer token for API authentication.
    api_key: String,
    /// Base API endpoint URL.
    endpoint: String,
    /// Reusable HTTP client.
    client: reqwest::Client,
    /// Active multi-round sessions keyed by state ID.
    sessions: Arc<Mutex<HashMap<String, PanSession>>>,
}

impl PanProvider {
    /// Create a new PAN provider with the default endpoint.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use worldforge_providers::pan::PanProvider;
    /// let provider = PanProvider::new("my-api-key");
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            client: reqwest::Client::new(),
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new PAN provider with a custom endpoint.
    ///
    /// Useful for testing against a local mock server.
    pub fn with_endpoint(api_key: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: endpoint.into(),
            client: reqwest::Client::new(),
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Format a prompt with PAN's FPS prefix convention.
    ///
    /// PAN requires prompts to be prefixed with `"FPS-{fps} "` before
    /// submission to the API.
    fn format_prompt(prompt: &str, fps: f32) -> String {
        format!("FPS-{fps} {prompt}")
    }

    /// Start a new multi-round generation session.
    ///
    /// Sends the first-round request with an image and prompt, then records
    /// the returned session state for subsequent rounds.
    #[tracing::instrument(skip(self, image_path))]
    pub async fn start_session(
        &self,
        state_id: impl Into<String> + std::fmt::Debug,
        prompt: &str,
        image_path: &str,
        fps: f32,
    ) -> Result<PanSession> {
        let state_id = state_id.into();
        let formatted_prompt = Self::format_prompt(prompt, fps);

        let body = PanFirstRoundRequest {
            prompt: formatted_prompt,
            image_path: image_path.to_string(),
            state_id: state_id.clone(),
        };

        let response = self
            .client
            .post(format!("{}/first_round", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        check_http_response("pan", status, &text)?;

        let pan_response: PanResponse = serde_json::from_str(&text)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        if let Some(error) = &pan_response.error {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "pan".to_string(),
                reason: error.clone(),
            });
        }

        let session = PanSession {
            state_id: pan_response.state_id,
            video_id: pan_response.video_id,
            round: 1,
        };

        self.sessions.lock().await.insert(state_id, session.clone());

        Ok(session)
    }

    /// Continue an existing multi-round generation session.
    ///
    /// Sends a continuation request using the server-side state and video
    /// identifiers from a previous round.
    #[tracing::instrument(skip(self))]
    pub async fn continue_session(
        &self,
        session: &mut PanSession,
        prompt: &str,
        fps: f32,
    ) -> Result<PanResponse> {
        let formatted_prompt = Self::format_prompt(prompt, fps);

        let body = PanContinueRequest {
            prompt: formatted_prompt,
            state_id: session.state_id.clone(),
            video_id: session.video_id.clone(),
        };

        let response = self
            .client
            .post(format!("{}/continue", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        check_http_response("pan", status, &text)?;

        let pan_response: PanResponse = serde_json::from_str(&text)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        if let Some(error) = &pan_response.error {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "pan".to_string(),
                reason: error.clone(),
            });
        }

        // Update session state for the next round.
        session.state_id = pan_response.state_id.clone();
        session.video_id = pan_response.video_id.clone();
        session.round += 1;

        // Update the tracked session.
        let original_key = session.state_id.clone();
        self.sessions
            .lock()
            .await
            .insert(original_key, session.clone());

        Ok(pan_response)
    }

    /// Cost estimate helper for a given operation.
    fn cost_for_operation(operation: &Operation) -> CostEstimate {
        match operation {
            Operation::Predict {
                steps, resolution, ..
            } => {
                let pixels = resolution.0 as f64 * resolution.1 as f64;
                let base = 0.02 * (*steps as f64);
                let scale = pixels / (DEFAULT_RESOLUTION.0 as f64 * DEFAULT_RESOLUTION.1 as f64);
                CostEstimate {
                    usd: base * scale,
                    credits: base * scale * 100.0,
                    estimated_latency_ms: 3000 + (*steps as u64) * 500,
                }
            }
            Operation::Generate {
                duration_seconds,
                resolution,
            } => {
                let pixels = resolution.0 as f64 * resolution.1 as f64;
                let base = 0.04 * duration_seconds;
                let scale = pixels / (DEFAULT_RESOLUTION.0 as f64 * DEFAULT_RESOLUTION.1 as f64);
                CostEstimate {
                    usd: base * scale,
                    credits: base * scale * 100.0,
                    estimated_latency_ms: 5000 + (*duration_seconds * 1000.0) as u64,
                }
            }
            // PAN does not support reasoning or transfer.
            Operation::Reason | Operation::Transfer { .. } => CostEstimate::default(),
        }
    }
}

#[async_trait]
impl WorldModelProvider for PanProvider {
    fn name(&self) -> &str {
        "pan"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: false,
            transfer: false,
            embed: false,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: MAX_VIDEO_LENGTH as f32,
            max_resolution: DEFAULT_RESOLUTION,
            fps_range: (8.0, 24.0),
            supported_action_spaces: vec![ActionSpaceType::Continuous, ActionSpaceType::Language],
            supports_depth: false,
            supports_segmentation: false,
            supports_planning: true,
            supports_gradient_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 3000,
                p95_ms: 8000,
                p99_ms: 15000,
                throughput_fps: 8.0,
            },
        }
    }

    #[tracing::instrument(skip(self, state, action, config))]
    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        let prompt = format!("{action:?}");
        let fps = config.fps.clamp(8.0, 24.0);
        let formatted_prompt = Self::format_prompt(&prompt, fps);
        let start = std::time::Instant::now();

        let request_body = serde_json::json!({
            "prompt": formatted_prompt,
            "image_path": "",
            "state_id": uuid::Uuid::new_v4().to_string(),
        });

        let response = self
            .client
            .post(format!("{}/first_round", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        check_http_response("pan", status, &text)?;

        let pan_response: PanResponse = serde_json::from_str(&text)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        if let Some(error) = &pan_response.error {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "pan".to_string(),
                reason: error.clone(),
            });
        }

        let latency_ms = start.elapsed().as_millis() as u64;

        let mut output_state = state.clone();
        output_state.time.step += config.steps as u64;
        output_state.time.seconds += config.steps as f64 / fps as f64;
        output_state.time.dt = 1.0 / fps as f64;

        let num_frames = if pan_response.frames.is_empty() {
            FRAMES_PER_ROUND as usize
        } else {
            pan_response.frames.len()
        };
        let duration = num_frames as f64 / fps as f64;
        let video = if config.return_video {
            Some(build_stub_video_clip(
                config.resolution,
                fps,
                duration,
                latency_ms,
            ))
        } else {
            None
        };

        let confidence = 0.85;

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: "pan".to_string(),
            model: "pan-v1".to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video,
            confidence,
            physics_scores: PhysicsScores {
                overall: confidence,
                object_permanence: confidence,
                gravity_compliance: confidence,
                collision_accuracy: confidence,
                spatial_consistency: confidence,
                temporal_consistency: confidence,
            },
            latency_ms,
            cost: Self::cost_for_operation(&Operation::Predict {
                steps: config.steps,
                resolution: config.resolution,
            }),
            provenance: None,
            sampling: None,
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    #[tracing::instrument(skip(self, prompt, config))]
    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip> {
        let fps = config.fps.clamp(8.0, 24.0);
        let formatted_prompt = Self::format_prompt(&prompt.text, fps);
        let state_id = uuid::Uuid::new_v4().to_string();

        let body = PanFirstRoundRequest {
            prompt: formatted_prompt,
            image_path: String::new(),
            state_id,
        };

        let response = self
            .client
            .post(format!("{}/first_round", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        check_http_response("pan", status, &text)?;

        let pan_response: PanResponse = serde_json::from_str(&text)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        if let Some(error) = &pan_response.error {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "pan".to_string(),
                reason: error.clone(),
            });
        }

        let duration = config.duration_seconds.min(MAX_VIDEO_LENGTH);
        Ok(build_stub_video_clip(
            config.resolution,
            fps,
            duration,
            pan_response.frames.len() as u64,
        ))
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "pan".to_string(),
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
            provider: "pan".to_string(),
            capability: "transfer".to_string(),
        })
    }

    #[tracing::instrument(skip(self))]
    async fn health_check(&self) -> Result<HealthStatus> {
        let start = std::time::Instant::now();

        let response = self
            .client
            .get(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .timeout(std::time::Duration::from_millis(5000))
            .send()
            .await;

        let latency = start.elapsed().as_millis() as u64;

        match response {
            Ok(resp) => Ok(HealthStatus {
                healthy: resp.status().is_success(),
                message: format!("PAN API responded with HTTP {}", resp.status()),
                latency_ms: latency,
            }),
            Err(error) => Ok(HealthStatus {
                healthy: false,
                message: format!("PAN health check failed: {error}"),
                latency_ms: latency,
            }),
        }
    }

    async fn plan(&self, request: &PlanRequest) -> Result<Plan> {
        let step_cost = self.estimate_cost(&Operation::Predict {
            steps: 1,
            resolution: DEFAULT_RESOLUTION,
        });
        native_planning::plan_native("pan", request, step_cost)
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        Self::cost_for_operation(operation)
    }

    fn translate_action(&self, action: &Action) -> Result<worldforge_core::action::ProviderAction> {
        let prompt = format!("{action:?}");
        Ok(worldforge_core::action::ProviderAction {
            provider: "pan".to_string(),
            data: serde_json::json!({
                "prompt": prompt,
                "fps": DEFAULT_FPS,
            }),
        })
    }

    fn supported_actions(&self) -> Vec<ActionType> {
        vec![
            ActionType::Move,
            ActionType::Grasp,
            ActionType::Release,
            ActionType::Push,
            ActionType::Rotate,
            ActionType::Place,
            ActionType::CameraMove,
            ActionType::CameraLookAt,
            ActionType::Navigate,
            ActionType::SetWeather,
            ActionType::SetLighting,
            ActionType::SpawnObject,
            ActionType::RemoveObject,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::provider::WorldModelProvider;

    #[test]
    fn test_pan_provider_creation() {
        let provider = PanProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(provider.name(), "pan");
    }

    #[test]
    fn test_pan_provider_with_endpoint() {
        let provider = PanProvider::with_endpoint("test-key", "https://custom.pan.example.com");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.endpoint, "https://custom.pan.example.com");
        assert_eq!(provider.name(), "pan");
    }

    #[test]
    fn test_pan_capabilities_include_planning() {
        let provider = PanProvider::new("test-key");
        let caps = provider.capabilities();

        assert!(caps.predict);
        assert!(caps.generate);
        assert!(!caps.reason);
        assert!(!caps.transfer);
        assert!(!caps.embed);
        assert!(caps.action_conditioned);
        assert!(!caps.multi_view);
        assert!(!caps.supports_depth);
        assert!(!caps.supports_segmentation);
        assert!(caps.supports_planning);
        assert!(!caps.supports_gradient_planning);
        assert_eq!(caps.max_resolution, (832, 480));
        assert_eq!(caps.max_video_length_seconds, 5.0);
        assert_eq!(caps.fps_range, (8.0, 24.0));
    }

    #[test]
    fn test_pan_session_tracking() {
        let provider = PanProvider::new("test-key");

        // Verify that the sessions map starts empty.
        let sessions = provider.sessions.clone();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let count = rt.block_on(async { sessions.lock().await.len() });
        assert_eq!(count, 0);

        // Manually insert a session and verify tracking.
        let session = PanSession {
            state_id: "test-state".to_string(),
            video_id: "test-video".to_string(),
            round: 1,
        };

        rt.block_on(async {
            sessions
                .lock()
                .await
                .insert("test-state".to_string(), session.clone());
        });

        let tracked = rt.block_on(async { sessions.lock().await.get("test-state").cloned() });

        assert!(tracked.is_some());
        let tracked = tracked.unwrap();
        assert_eq!(tracked.state_id, "test-state");
        assert_eq!(tracked.video_id, "test-video");
        assert_eq!(tracked.round, 1);
    }

    #[test]
    fn test_pan_cost_estimation() {
        let provider = PanProvider::new("test-key");

        // Predict cost
        let predict_cost = provider.estimate_cost(&Operation::Predict {
            steps: 10,
            resolution: (832, 480),
        });
        assert!(predict_cost.usd > 0.0);
        assert!(predict_cost.credits > 0.0);
        assert!(predict_cost.estimated_latency_ms > 0);

        // Generate cost
        let generate_cost = provider.estimate_cost(&Operation::Generate {
            duration_seconds: 5.0,
            resolution: (832, 480),
        });
        assert!(generate_cost.usd > 0.0);
        assert!(generate_cost.credits > 0.0);
        assert!(generate_cost.estimated_latency_ms > 0);

        // Unsupported operations return zero cost
        let reason_cost = provider.estimate_cost(&Operation::Reason);
        assert_eq!(reason_cost.usd, 0.0);
        assert_eq!(reason_cost.credits, 0.0);

        let transfer_cost = provider.estimate_cost(&Operation::Transfer {
            duration_seconds: 3.0,
        });
        assert_eq!(transfer_cost.usd, 0.0);
        assert_eq!(transfer_cost.credits, 0.0);
    }

    #[test]
    fn test_pan_cost_scales_with_resolution() {
        let provider = PanProvider::new("test-key");

        let low_res = provider.estimate_cost(&Operation::Generate {
            duration_seconds: 5.0,
            resolution: (416, 240),
        });
        let high_res = provider.estimate_cost(&Operation::Generate {
            duration_seconds: 5.0,
            resolution: (832, 480),
        });
        assert!(high_res.usd > low_res.usd);
    }

    #[test]
    fn test_pan_prompt_formatting() {
        let formatted = PanProvider::format_prompt("a robot moving forward", 24.0);
        assert_eq!(formatted, "FPS-24 a robot moving forward");

        let formatted_8fps = PanProvider::format_prompt("slow motion", 8.0);
        assert_eq!(formatted_8fps, "FPS-8 slow motion");
    }

    #[tokio::test]
    async fn test_pan_reason_unsupported() {
        let provider = PanProvider::new("test-key");
        let input = ReasoningInput {
            video: None,
            state: None,
        };
        let result = provider.reason(&input, "Why?").await;
        assert!(matches!(
            result,
            Err(WorldForgeError::UnsupportedCapability { .. })
        ));
    }

    #[tokio::test]
    async fn test_pan_transfer_unsupported() {
        let provider = PanProvider::new("test-key");
        let source = build_stub_video_clip((640, 480), 24.0, 1.0, 0);
        let controls = SpatialControls {
            camera_trajectory: None,
            depth_map: None,
            segmentation_map: None,
        };
        let config = TransferConfig::default();
        let result = provider.transfer(&source, &controls, &config).await;
        assert!(matches!(
            result,
            Err(WorldForgeError::UnsupportedCapability { .. })
        ));
    }

    #[test]
    fn test_pan_supported_actions() {
        let provider = PanProvider::new("test-key");
        let actions = provider.supported_actions();
        assert!(!actions.is_empty());
        assert!(actions.contains(&ActionType::Move));
        assert!(actions.contains(&ActionType::CameraMove));
        assert!(actions.contains(&ActionType::Navigate));
    }

    #[test]
    fn test_pan_translate_action() {
        let provider = PanProvider::new("test-key");
        let action = Action::Move {
            target: worldforge_core::types::Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            speed: 1.5,
        };
        let result = provider.translate_action(&action);
        assert!(result.is_ok());
        let provider_action = result.unwrap();
        assert_eq!(provider_action.provider, "pan");
    }

    #[test]
    fn test_pan_session_struct() {
        let session = PanSession {
            state_id: "abc123".to_string(),
            video_id: "vid456".to_string(),
            round: 3,
        };
        assert_eq!(session.state_id, "abc123");
        assert_eq!(session.video_id, "vid456");
        assert_eq!(session.round, 3);
    }

    #[test]
    fn test_pan_default_endpoint() {
        assert_eq!(DEFAULT_ENDPOINT, "https://ifm.mbzuai.ac.ae/pan");
    }

    #[test]
    fn test_pan_default_resolution() {
        assert_eq!(DEFAULT_RESOLUTION, (832, 480));
    }

    #[test]
    fn test_pan_frames_per_round() {
        assert_eq!(FRAMES_PER_ROUND, 41);
    }
}
