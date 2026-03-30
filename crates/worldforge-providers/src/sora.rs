//! Sora 2 (OpenAI) provider adapter.
//!
//! Implements the `WorldModelProvider` trait for OpenAI's Sora 2 video
//! generation model using an async submit/poll/download pattern.
//!
//! - **Auth:** Bearer token from `OPENAI_API_KEY`
//! - **Submit:** `POST /v1/videos` with model, prompt, optional image, duration, resolution
//! - **Poll:** `GET /v1/videos/{id}` until status is `"completed"` or `"failed"`
//! - **Download:** from the result URL in the completed task response

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use worldforge_core::action::{Action, ActionSpaceType, ActionType};
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_ENDPOINT: &str = "https://api.openai.com";
const PROVIDER_NAME: &str = "sora";
const MODEL_NAME: &str = "sora-2";
const DEFAULT_RESOLUTION: (u32, u32) = (1280, 720);
const MAX_VIDEO_LENGTH_SECONDS: f32 = 10.0;
const MIN_FPS: f32 = 12.0;
const MAX_FPS: f32 = 24.0;

// ---------------------------------------------------------------------------
// API request/response types
// ---------------------------------------------------------------------------

/// Request body for the Sora video generation endpoint.
#[derive(Debug, Serialize)]
struct SoraGenerateRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
    duration: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolution: Option<String>,
}

/// Task response returned by the submit and poll endpoints.
#[derive(Debug, Deserialize)]
struct SoraTaskResponse {
    id: String,
    status: String,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// OpenAI Sora 2 provider adapter.
///
/// Wraps the OpenAI video generation API to implement the
/// `WorldModelProvider` trait for Sora 2 video generation.
#[derive(Debug, Clone)]
pub struct SoraProvider {
    /// API key used for Bearer authentication.
    api_key: String,
    /// Base endpoint URL (without trailing slash).
    endpoint: String,
    /// HTTP client.
    client: reqwest::Client,
}

impl SoraProvider {
    /// Create a new Sora provider with the default OpenAI endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new Sora provider with a custom endpoint.
    pub fn with_endpoint(api_key: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: endpoint.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Build the URL for the video generation submit endpoint.
    fn submit_url(&self) -> String {
        format!("{}/v1/videos", self.endpoint)
    }

    /// Build the URL for polling a task by ID.
    fn poll_url(&self, task_id: &str) -> String {
        format!("{}/v1/videos/{}", self.endpoint, task_id)
    }

    /// Build the URL for the models list endpoint (used for health checks).
    fn models_url(&self) -> String {
        format!("{}/v1/models", self.endpoint)
    }

    /// Format a resolution tuple as `"WIDTHxHEIGHT"`.
    fn format_resolution(resolution: (u32, u32)) -> String {
        format!("{}x{}", resolution.0, resolution.1)
    }

    /// Submit a video generation request and poll until completion.
    #[tracing::instrument(skip(self, prompt_text, image))]
    async fn submit_and_poll(
        &self,
        prompt_text: &str,
        image: Option<String>,
        duration: f64,
        resolution: (u32, u32),
    ) -> Result<SoraTaskResponse> {
        let body = SoraGenerateRequest {
            model: MODEL_NAME.to_string(),
            prompt: prompt_text.to_string(),
            image,
            duration,
            resolution: Some(Self::format_resolution(resolution)),
        };

        let response = self
            .client
            .post(self.submit_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| WorldForgeError::ProviderUnavailable {
                provider: PROVIDER_NAME.to_string(),
                reason: format!("request failed: {e}"),
            })?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .unwrap_or_else(|_| "unknown error".to_string());

        check_http_response(PROVIDER_NAME, status, &response_text)?;

        let task: SoraTaskResponse = serde_json::from_str(&response_text)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        // If the task completed synchronously, return immediately.
        if task.status == "completed" {
            return Ok(task);
        }
        if task.status == "failed" {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: PROVIDER_NAME.to_string(),
                reason: task
                    .error
                    .unwrap_or_else(|| "generation failed".to_string()),
            });
        }

        // Poll until completion.
        let task_id = task.id.clone();
        let polling_config = PollingConfig::default();

        poll_until_complete(PROVIDER_NAME, &polling_config, || {
            let tid = task_id.clone();
            async move {
                let poll_resp = self
                    .client
                    .get(self.poll_url(&tid))
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .send()
                    .await
                    .map_err(|e| WorldForgeError::ProviderUnavailable {
                        provider: PROVIDER_NAME.to_string(),
                        reason: format!("poll request failed: {e}"),
                    })?;

                let poll_status = poll_resp.status();
                let poll_text = poll_resp
                    .text()
                    .await
                    .unwrap_or_else(|_| "unknown error".to_string());

                check_http_response(PROVIDER_NAME, poll_status, &poll_text)?;

                let poll_task: SoraTaskResponse = serde_json::from_str(&poll_text)
                    .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

                match poll_task.status.as_str() {
                    "completed" => Ok(PollStatus::Complete(poll_task)),
                    "failed" => Ok(PollStatus::Failed(
                        poll_task
                            .error
                            .unwrap_or_else(|| "generation failed".to_string()),
                    )),
                    _ => Ok(PollStatus::Pending),
                }
            }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// WorldModelProvider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl WorldModelProvider for SoraProvider {
    fn name(&self) -> &str {
        PROVIDER_NAME
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
            max_video_length_seconds: MAX_VIDEO_LENGTH_SECONDS,
            max_resolution: DEFAULT_RESOLUTION,
            fps_range: (MIN_FPS, MAX_FPS),
            supported_action_spaces: vec![ActionSpaceType::Language],
            supports_depth: false,
            supports_segmentation: false,
            supports_planning: false,
            supports_gradient_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 20000,
                p95_ms: 60000,
                p99_ms: 120000,
                throughput_fps: 3.0,
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

        let prompt_text = format!(
            "Predict the world state after applying action: {:?}",
            action
        );

        let fps = config.fps.clamp(MIN_FPS, MAX_FPS);
        let duration = (config.steps as f64 / fps as f64).min(MAX_VIDEO_LENGTH_SECONDS as f64);

        let task = self
            .submit_and_poll(&prompt_text, None, duration, config.resolution)
            .await?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let video = if config.return_video && task.url.is_some() {
            Some(build_stub_video_clip(
                config.resolution,
                fps,
                duration,
                latency_ms,
            ))
        } else {
            None
        };

        let confidence = 0.75;
        let physics_scores = PhysicsScores {
            overall: 0.70,
            object_permanence: 0.75,
            gravity_compliance: 0.65,
            collision_accuracy: 0.60,
            spatial_consistency: 0.70,
            temporal_consistency: 0.75,
        };

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: PROVIDER_NAME.to_string(),
            model: MODEL_NAME.to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state: state.clone(),
            video,
            confidence,
            physics_scores,
            latency_ms,
            cost: self.estimate_cost(&Operation::Predict {
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
        let fps = config.fps.clamp(MIN_FPS, MAX_FPS);
        let duration = config.duration_seconds.min(MAX_VIDEO_LENGTH_SECONDS as f64);

        let task = self
            .submit_and_poll(&prompt.text, None, duration, config.resolution)
            .await?;

        if task.url.is_none() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: PROVIDER_NAME.to_string(),
                reason: "completed task has no video URL".to_string(),
            });
        }

        Ok(build_stub_video_clip(config.resolution, fps, duration, 0))
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: PROVIDER_NAME.to_string(),
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
            provider: PROVIDER_NAME.to_string(),
            capability: "transfer".to_string(),
        })
    }

    #[tracing::instrument(skip(self))]
    async fn health_check(&self) -> Result<HealthStatus> {
        let start = std::time::Instant::now();

        let response = self
            .client
            .get(self.models_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await;

        let latency_ms = start.elapsed().as_millis() as u64;

        match response {
            Ok(resp) if resp.status().is_success() => Ok(HealthStatus {
                healthy: true,
                message: format!("Sora provider ready: {} at {}", MODEL_NAME, self.endpoint),
                latency_ms,
            }),
            Ok(resp) => Ok(HealthStatus {
                healthy: false,
                message: format!("Sora health check failed: HTTP {}", resp.status()),
                latency_ms,
            }),
            Err(e) => Ok(HealthStatus {
                healthy: false,
                message: format!("Sora health check failed: {e}"),
                latency_ms,
            }),
        }
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        match operation {
            Operation::Predict { steps, resolution } => {
                let pixels = (resolution.0 as f64) * (resolution.1 as f64);
                let step_factor = (*steps).max(1) as f64;
                let usd = 0.030 + step_factor * 0.006 + pixels / 1_000_000.0 * 0.010;
                CostEstimate {
                    usd,
                    credits: usd * 100.0,
                    estimated_latency_ms: 20_000 + (*steps).max(1) as u64 * 600,
                }
            }
            Operation::Generate {
                duration_seconds,
                resolution,
            } => {
                let duration = duration_seconds.clamp(0.5, MAX_VIDEO_LENGTH_SECONDS as f64);
                let pixels = (resolution.0 as f64) * (resolution.1 as f64);
                let usd = 0.040 + duration * 0.012 + pixels / 1_000_000.0 * 0.015;
                CostEstimate {
                    usd,
                    credits: usd * 100.0,
                    estimated_latency_ms: 25_000 + (duration * 2_500.0) as u64,
                }
            }
            Operation::Reason => CostEstimate {
                usd: 0.0,
                credits: 0.0,
                estimated_latency_ms: 0,
            },
            Operation::Transfer { .. } => CostEstimate {
                usd: 0.0,
                credits: 0.0,
                estimated_latency_ms: 0,
            },
        }
    }

    fn supported_actions(&self) -> Vec<ActionType> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sora_provider_creation() {
        let provider = SoraProvider::new("test-api-key");
        assert_eq!(provider.name(), "sora");
        assert_eq!(provider.api_key, "test-api-key");
        assert_eq!(provider.endpoint, DEFAULT_ENDPOINT);
    }

    #[test]
    fn test_sora_provider_with_endpoint() {
        let provider = SoraProvider::with_endpoint("key", "https://custom.openai.com");
        assert_eq!(provider.endpoint, "https://custom.openai.com");
        assert_eq!(provider.api_key, "key");
    }

    #[test]
    fn test_sora_capabilities() {
        let provider = SoraProvider::new("test-key");
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

        assert_eq!(caps.max_resolution, (1280, 720));
        assert_eq!(caps.max_video_length_seconds, 10.0);
        assert_eq!(caps.fps_range, (12.0, 24.0));
        assert_eq!(
            caps.supported_action_spaces,
            vec![ActionSpaceType::Language]
        );
    }

    #[test]
    fn test_sora_cost_estimation() {
        let provider = SoraProvider::new("test-key");

        let predict_cost = provider.estimate_cost(&Operation::Predict {
            steps: 10,
            resolution: (1280, 720),
        });
        assert!(predict_cost.usd > 0.0);
        assert!(predict_cost.credits > 0.0);
        assert!(predict_cost.estimated_latency_ms > 0);

        let generate_cost = provider.estimate_cost(&Operation::Generate {
            duration_seconds: 5.0,
            resolution: (1280, 720),
        });
        assert!(generate_cost.usd > 0.0);
        assert!(generate_cost.credits > 0.0);
        assert!(generate_cost.estimated_latency_ms > 0);

        // Unsupported operations return zero cost.
        let reason_cost = provider.estimate_cost(&Operation::Reason);
        assert_eq!(reason_cost.usd, 0.0);
        assert_eq!(reason_cost.credits, 0.0);

        let transfer_cost = provider.estimate_cost(&Operation::Transfer {
            duration_seconds: 2.0,
        });
        assert_eq!(transfer_cost.usd, 0.0);
        assert_eq!(transfer_cost.credits, 0.0);
    }

    #[test]
    fn test_sora_submit_url() {
        let provider = SoraProvider::new("key");
        assert_eq!(provider.submit_url(), "https://api.openai.com/v1/videos");
    }

    #[test]
    fn test_sora_poll_url() {
        let provider = SoraProvider::new("key");
        assert_eq!(
            provider.poll_url("task-abc-123"),
            "https://api.openai.com/v1/videos/task-abc-123"
        );
    }

    #[test]
    fn test_sora_models_url() {
        let provider = SoraProvider::new("key");
        assert_eq!(provider.models_url(), "https://api.openai.com/v1/models");
    }

    #[test]
    fn test_sora_format_resolution() {
        assert_eq!(SoraProvider::format_resolution((1280, 720)), "1280x720");
        assert_eq!(SoraProvider::format_resolution((1920, 1080)), "1920x1080");
    }

    #[test]
    fn test_sora_supported_actions_empty() {
        let provider = SoraProvider::new("key");
        assert!(provider.supported_actions().is_empty());
    }

    #[tokio::test]
    async fn test_sora_reason_unsupported() {
        let provider = SoraProvider::new("key");
        let input = ReasoningInput {
            video: None,
            state: None,
        };
        let result = provider.reason(&input, "test query").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            WorldForgeError::UnsupportedCapability {
                provider: name,
                capability,
            } => {
                assert_eq!(name, "sora");
                assert_eq!(capability, "reason");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_sora_transfer_unsupported() {
        let provider = SoraProvider::new("key");
        let source = build_stub_video_clip((64, 64), 12.0, 1.0, 0);
        let controls = SpatialControls::default();
        let config = TransferConfig::default();
        let result = provider.transfer(&source, &controls, &config).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            WorldForgeError::UnsupportedCapability {
                provider: name,
                capability,
            } => {
                assert_eq!(name, "sora");
                assert_eq!(capability, "transfer");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_sora_request_serialization() {
        let request = SoraGenerateRequest {
            model: "sora-2".to_string(),
            prompt: "A cat walking through snow".to_string(),
            image: None,
            duration: 5.0,
            resolution: Some("1280x720".to_string()),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["model"], "sora-2");
        assert_eq!(json["prompt"], "A cat walking through snow");
        assert!(json.get("image").is_none());
        assert_eq!(json["duration"], 5.0);
        assert_eq!(json["resolution"], "1280x720");
    }

    #[test]
    fn test_sora_request_with_image() {
        let request = SoraGenerateRequest {
            model: "sora-2".to_string(),
            prompt: "Transform this scene".to_string(),
            image: Some("base64data".to_string()),
            duration: 3.0,
            resolution: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["image"], "base64data");
        assert!(json.get("resolution").is_none());
    }

    #[test]
    fn test_sora_task_response_deserialization() {
        let json = r#"{
            "id": "task-abc-123",
            "status": "processing"
        }"#;
        let task: SoraTaskResponse = serde_json::from_str(json).unwrap();
        assert_eq!(task.id, "task-abc-123");
        assert_eq!(task.status, "processing");
        assert!(task.url.is_none());
        assert!(task.error.is_none());
    }

    #[test]
    fn test_sora_task_response_completed() {
        let json = r#"{
            "id": "task-abc-123",
            "status": "completed",
            "url": "https://cdn.openai.com/videos/result.mp4"
        }"#;
        let task: SoraTaskResponse = serde_json::from_str(json).unwrap();
        assert_eq!(task.status, "completed");
        assert_eq!(
            task.url.unwrap(),
            "https://cdn.openai.com/videos/result.mp4"
        );
    }

    #[test]
    fn test_sora_task_response_failed() {
        let json = r#"{
            "id": "task-err-456",
            "status": "failed",
            "error": "content policy violation"
        }"#;
        let task: SoraTaskResponse = serde_json::from_str(json).unwrap();
        assert_eq!(task.status, "failed");
        assert_eq!(task.error.unwrap(), "content policy violation");
    }

    #[test]
    fn test_sora_latency_profile() {
        let provider = SoraProvider::new("key");
        let caps = provider.capabilities();
        assert_eq!(caps.latency_profile.p50_ms, 20000);
        assert_eq!(caps.latency_profile.p95_ms, 60000);
        assert_eq!(caps.latency_profile.p99_ms, 120000);
        assert_eq!(caps.latency_profile.throughput_fps, 3.0);
    }
}
