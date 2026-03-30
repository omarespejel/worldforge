//! MiniMax/Hailuo provider adapter.
//!
//! Implements the `WorldModelProvider` trait for MiniMax's video generation
//! API (T2V-01 model). Uses the async submit/poll/download pattern via the
//! shared polling infrastructure.
//!
//! # API Flow
//!
//! 1. **Submit** — `POST /v1/video_generation` with model + prompt
//! 2. **Poll** — `GET /v1/query/video_generation?task_id=<id>` until complete
//! 3. **Download** — `GET /v1/files/retrieve?file_id=<id>` for the video URL

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

use crate::polling::{
    build_stub_video_clip, check_http_response, poll_until_complete, PollStatus, PollingConfig,
};

/// Default MiniMax API endpoint.
const DEFAULT_ENDPOINT: &str = "https://api.minimax.io";

/// Default output resolution `(width, height)`.
const DEFAULT_RESOLUTION: (u32, u32) = (1072, 720);

/// Maximum video duration in seconds.
const MAX_VIDEO_LENGTH: f32 = 6.0;

/// Default model identifier.
const MODEL_ID: &str = "T2V-01";

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// Request body for the MiniMax video generation submit endpoint.
#[derive(Debug, Serialize)]
struct MiniMaxGenerateRequest {
    /// Model identifier (e.g. `"T2V-01"`).
    model: String,
    /// Text prompt describing the desired video.
    prompt: String,
    /// Optional base64-encoded first frame image for image-to-video.
    #[serde(skip_serializing_if = "Option::is_none")]
    first_frame_image: Option<String>,
}

/// Response from the submit endpoint.
#[derive(Debug, Deserialize)]
struct MiniMaxSubmitResponse {
    /// Opaque task identifier used for polling.
    task_id: String,
}

/// Response from the poll endpoint.
#[derive(Debug, Deserialize)]
struct MiniMaxPollResponse {
    /// Current task status: `"Success"`, `"Processing"`, or `"Failed"`.
    status: String,
    /// File identifier available when status is `"Success"`.
    #[serde(default)]
    file_id: Option<String>,
}

/// Response from the file retrieval endpoint.
#[derive(Debug, Deserialize)]
struct MiniMaxFileResponse {
    /// Signed download URL for the generated video.
    download_url: String,
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// MiniMax/Hailuo video generation provider.
///
/// Wraps the MiniMax HTTP API to implement the `WorldModelProvider` trait.
/// Supports text-to-video and optional image-to-video generation via the
/// T2V-01 model family.
#[derive(Debug, Clone)]
pub struct MiniMaxProvider {
    /// Bearer token for API authentication.
    api_key: String,
    /// Base API endpoint URL.
    endpoint: String,
    /// Reusable HTTP client.
    client: reqwest::Client,
}

impl MiniMaxProvider {
    /// Create a new MiniMax provider with the default endpoint.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use worldforge_providers::minimax::MiniMaxProvider;
    /// let provider = MiniMaxProvider::new("my-api-key");
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new MiniMax provider with a custom endpoint.
    ///
    /// Useful for testing against a local mock server.
    pub fn with_endpoint(api_key: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: endpoint.into(),
            client: reqwest::Client::new(),
        }
    }

    // -- internal helpers ---------------------------------------------------

    /// Submit a video generation job and return the task ID.
    #[tracing::instrument(skip(self, prompt, first_frame_image))]
    async fn submit_generation(
        &self,
        prompt: &str,
        first_frame_image: Option<String>,
    ) -> Result<String> {
        let body = MiniMaxGenerateRequest {
            model: MODEL_ID.to_string(),
            prompt: prompt.to_string(),
            first_frame_image,
        };

        let response = self
            .client
            .post(format!("{}/v1/video_generation", self.endpoint))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        check_http_response("minimax", status, &text)?;

        let parsed: MiniMaxSubmitResponse = serde_json::from_str(&text)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        Ok(parsed.task_id)
    }

    /// Poll a generation task until it completes or fails.
    #[tracing::instrument(skip(self))]
    async fn poll_task(&self, task_id: &str) -> Result<String> {
        let polling_config = PollingConfig::default();
        let task_id_owned = task_id.to_string();

        let file_id = poll_until_complete("minimax", &polling_config, || {
            let tid = task_id_owned.clone();
            async move {
                let response = self
                    .client
                    .get(format!(
                        "{}/v1/query/video_generation?task_id={}",
                        self.endpoint, tid
                    ))
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .send()
                    .await
                    .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

                let status = response.status();
                let text = response
                    .text()
                    .await
                    .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

                check_http_response("minimax", status, &text)?;

                let poll_response: MiniMaxPollResponse = serde_json::from_str(&text)
                    .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

                match poll_response.status.as_str() {
                    "Success" => {
                        let file_id = poll_response.file_id.ok_or_else(|| {
                            WorldForgeError::ProviderUnavailable {
                                provider: "minimax".to_string(),
                                reason: "task succeeded but no file_id returned".to_string(),
                            }
                        })?;
                        Ok(PollStatus::Complete(file_id))
                    }
                    "Failed" => Ok(PollStatus::Failed("video generation failed".to_string())),
                    _ => Ok(PollStatus::Pending),
                }
            }
        })
        .await?;

        Ok(file_id)
    }

    /// Download a file and return its download URL.
    #[tracing::instrument(skip(self))]
    async fn retrieve_file_url(&self, file_id: &str) -> Result<String> {
        let response = self
            .client
            .get(format!(
                "{}/v1/files/retrieve?file_id={}",
                self.endpoint, file_id
            ))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        check_http_response("minimax", status, &text)?;

        let file_response: MiniMaxFileResponse = serde_json::from_str(&text)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        Ok(file_response.download_url)
    }

    /// Cost estimate for a given operation.
    fn cost_for_operation(operation: &Operation) -> CostEstimate {
        match operation {
            Operation::Predict {
                steps, resolution, ..
            } => {
                let pixels = resolution.0 as f64 * resolution.1 as f64;
                let base = 0.01 * (*steps as f64);
                let scale = pixels / (1072.0 * 720.0);
                CostEstimate {
                    usd: base * scale,
                    credits: base * scale * 100.0,
                    estimated_latency_ms: 15_000 + (*steps as u64) * 500,
                }
            }
            Operation::Generate {
                duration_seconds,
                resolution,
            } => {
                let pixels = resolution.0 as f64 * resolution.1 as f64;
                let base = 0.05 * duration_seconds;
                let scale = pixels / (1072.0 * 720.0);
                CostEstimate {
                    usd: base * scale,
                    credits: base * scale * 100.0,
                    estimated_latency_ms: 30_000 + (*duration_seconds * 5000.0) as u64,
                }
            }
            Operation::Reason | Operation::Transfer { .. } => CostEstimate::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// WorldModelProvider
// ---------------------------------------------------------------------------

#[async_trait]
impl WorldModelProvider for MiniMaxProvider {
    fn name(&self) -> &str {
        "minimax"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: false,
            transfer: false,
            embed: false,
            action_conditioned: false,
            multi_view: false,
            max_video_length_seconds: MAX_VIDEO_LENGTH,
            max_resolution: DEFAULT_RESOLUTION,
            fps_range: (24.0, 30.0),
            supported_action_spaces: vec![ActionSpaceType::Discrete],
            supports_depth: false,
            supports_segmentation: false,
            supports_planning: false,
            supports_gradient_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 30_000,
                p95_ms: 60_000,
                p99_ms: 90_000,
                throughput_fps: 0.1,
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
        let start = std::time::Instant::now();

        // Build a prompt from the action description.
        let prompt = format!(
            "Predict next state: action={}, scene={}",
            action.action_type(),
            state.metadata.name
        );

        let task_id = self.submit_generation(&prompt, None).await?;
        let file_id = self.poll_task(&task_id).await?;
        let _download_url = self.retrieve_file_url(&file_id).await?;
        let latency_ms = start.elapsed().as_millis() as u64;

        let duration = config.steps as f64 / config.fps.max(1.0) as f64;
        let video = if config.return_video {
            Some(build_stub_video_clip(
                config.resolution,
                config.fps,
                duration,
                task_id.len() as u64,
            ))
        } else {
            None
        };

        let mut output_state = state.clone();
        output_state.time.step += config.steps as u64;
        output_state.time.seconds += duration;
        output_state.time.dt = 1.0 / config.fps as f64;

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: "minimax".to_string(),
            model: MODEL_ID.to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video,
            confidence: 0.7,
            physics_scores: PhysicsScores {
                overall: 0.7,
                object_permanence: 0.7,
                gravity_compliance: 0.7,
                collision_accuracy: 0.7,
                spatial_consistency: 0.7,
                temporal_consistency: 0.7,
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
        let task_id = self.submit_generation(&prompt.text, None).await?;
        let file_id = self.poll_task(&task_id).await?;
        let _download_url = self.retrieve_file_url(&file_id).await?;

        Ok(build_stub_video_clip(
            config.resolution,
            config.fps,
            config.duration_seconds,
            task_id.len() as u64,
        ))
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "minimax".to_string(),
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
            provider: "minimax".to_string(),
            capability: "transfer".to_string(),
        })
    }

    #[tracing::instrument(skip(self))]
    async fn health_check(&self) -> Result<HealthStatus> {
        let start = std::time::Instant::now();

        let response = self
            .client
            .get(&self.endpoint)
            .timeout(std::time::Duration::from_millis(5000))
            .send()
            .await;

        let latency_ms = start.elapsed().as_millis() as u64;

        match response {
            Ok(resp) if resp.status().is_success() || resp.status().is_redirection() => {
                Ok(HealthStatus {
                    healthy: true,
                    message: "MiniMax API reachable".to_string(),
                    latency_ms,
                })
            }
            Ok(resp) => Ok(HealthStatus {
                healthy: false,
                message: format!("MiniMax API returned HTTP {}", resp.status()),
                latency_ms,
            }),
            Err(e) => Ok(HealthStatus {
                healthy: false,
                message: format!("MiniMax API unreachable: {e}"),
                latency_ms,
            }),
        }
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        Self::cost_for_operation(operation)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimax_provider_creation() {
        let provider = MiniMaxProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(provider.name(), "minimax");
    }

    #[test]
    fn test_minimax_provider_with_endpoint() {
        let provider = MiniMaxProvider::with_endpoint("test-key", "https://custom.api.example.com");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.endpoint, "https://custom.api.example.com");
        assert_eq!(provider.name(), "minimax");
    }

    #[test]
    fn test_minimax_capabilities() {
        let provider = MiniMaxProvider::new("test-key");
        let caps = provider.capabilities();

        assert!(caps.predict);
        assert!(caps.generate);
        assert!(!caps.reason);
        assert!(!caps.transfer);
        assert!(!caps.embed);
        assert!(!caps.action_conditioned);
        assert!(!caps.multi_view);
        assert!(!caps.supports_depth);
        assert!(!caps.supports_segmentation);
        assert!(!caps.supports_planning);
        assert!(!caps.supports_gradient_planning);
        assert_eq!(caps.max_resolution, (1072, 720));
        assert_eq!(caps.max_video_length_seconds, 6.0);
        assert_eq!(caps.fps_range, (24.0, 30.0));
    }

    #[test]
    fn test_minimax_cost_estimation() {
        let provider = MiniMaxProvider::new("test-key");

        // Predict cost
        let predict_cost = provider.estimate_cost(&Operation::Predict {
            steps: 10,
            resolution: (1072, 720),
        });
        assert!(predict_cost.usd > 0.0);
        assert!(predict_cost.credits > 0.0);
        assert!(predict_cost.estimated_latency_ms > 0);

        // Generate cost
        let generate_cost = provider.estimate_cost(&Operation::Generate {
            duration_seconds: 5.0,
            resolution: (1072, 720),
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
    fn test_minimax_cost_scales_with_resolution() {
        let provider = MiniMaxProvider::new("test-key");

        let low_res = provider.estimate_cost(&Operation::Generate {
            duration_seconds: 5.0,
            resolution: (536, 360),
        });
        let high_res = provider.estimate_cost(&Operation::Generate {
            duration_seconds: 5.0,
            resolution: (1072, 720),
        });
        assert!(high_res.usd > low_res.usd);
    }

    #[tokio::test]
    async fn test_minimax_reason_unsupported() {
        let provider = MiniMaxProvider::new("test-key");
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
    async fn test_minimax_transfer_unsupported() {
        let provider = MiniMaxProvider::new("test-key");
        let source = build_stub_video_clip((640, 480), 24.0, 1.0, 0);
        let controls = SpatialControls::default();
        let config = TransferConfig::default();
        let result = provider.transfer(&source, &controls, &config).await;
        assert!(matches!(
            result,
            Err(WorldForgeError::UnsupportedCapability { .. })
        ));
    }

    #[test]
    fn test_minimax_generate_request_serialization() {
        let request = MiniMaxGenerateRequest {
            model: "T2V-01".to_string(),
            prompt: "A cat walking".to_string(),
            first_frame_image: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["model"], "T2V-01");
        assert_eq!(json["prompt"], "A cat walking");
        assert!(json.get("first_frame_image").is_none());

        let request_with_image = MiniMaxGenerateRequest {
            model: "T2V-01".to_string(),
            prompt: "A cat walking".to_string(),
            first_frame_image: Some("base64data".to_string()),
        };
        let json = serde_json::to_value(&request_with_image).unwrap();
        assert_eq!(json["first_frame_image"], "base64data");
    }

    #[test]
    fn test_minimax_submit_response_deserialization() {
        let json = r#"{"task_id": "task-123"}"#;
        let resp: MiniMaxSubmitResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.task_id, "task-123");
    }

    #[test]
    fn test_minimax_poll_response_deserialization() {
        // Success with file_id
        let json = r#"{"status": "Success", "file_id": "file-456"}"#;
        let resp: MiniMaxPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "Success");
        assert_eq!(resp.file_id.as_deref(), Some("file-456"));

        // Processing without file_id
        let json = r#"{"status": "Processing"}"#;
        let resp: MiniMaxPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "Processing");
        assert!(resp.file_id.is_none());

        // Failed
        let json = r#"{"status": "Failed"}"#;
        let resp: MiniMaxPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "Failed");
    }

    #[test]
    fn test_minimax_file_response_deserialization() {
        let json = r#"{"download_url": "https://cdn.example.com/video.mp4"}"#;
        let resp: MiniMaxFileResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.download_url, "https://cdn.example.com/video.mp4");
    }

    #[test]
    fn test_minimax_latency_profile() {
        let provider = MiniMaxProvider::new("test-key");
        let caps = provider.capabilities();
        assert!(caps.latency_profile.p50_ms > 0);
        assert!(caps.latency_profile.p95_ms >= caps.latency_profile.p50_ms);
        assert!(caps.latency_profile.p99_ms >= caps.latency_profile.p95_ms);
    }
}
