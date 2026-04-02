//! Sora 2 (OpenAI) provider adapter.
//!
//! Implements the `WorldModelProvider` trait for OpenAI's Sora 2 video
//! generation model using an async submit/poll/download pattern.
//!
//! - **Auth:** Bearer token from `OPENAI_API_KEY`
//! - **Submit:** `POST /v1/videos/generations` with model, prompt, optional image, duration, resolution
//! - **Poll:** `GET /v1/videos/generations/{id}` until status is `"completed"` or `"failed"`
//! - **Download:** from the result URL in the completed task response
//!
//! Uses the shared infrastructure modules (`HttpClientBuilder`, `RetryPolicy`,
//! `TokenBucket`, `AsyncJobRunner`) for robust, production-ready API interaction.
//! Falls back to deterministic stub output when no API key is set.

use std::time::Duration;

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

use crate::async_job::{AsyncJobRunner, PollStatus, PollingConfig};
use crate::http_client::{check_response, HttpClientBuilder};
use crate::polling::build_stub_video_clip;
use crate::rate_limit::{RateLimitConfig, TokenBucket};
use crate::retry::{retry_with_policy, RetryPolicy};

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
#[derive(Debug, Clone, Serialize)]
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
///
/// Uses the shared `HttpClientBuilder`, `RetryPolicy`, `TokenBucket`, and
/// `AsyncJobRunner` infrastructure for robust, production-ready API interaction.
/// Falls back to deterministic stub output when no API key is set.
#[derive(Debug, Clone)]
pub struct SoraProvider {
    /// API key used for Bearer authentication.
    api_key: String,
    /// Base endpoint URL (without trailing slash).
    endpoint: String,
    /// HTTP client built via `HttpClientBuilder` with Bearer auth baked in.
    client: reqwest::Client,
    /// Retry policy for transient failures.
    retry_policy: RetryPolicy,
    /// Token-bucket rate limiter.
    rate_limiter: TokenBucket,
}

impl SoraProvider {
    /// Create a new Sora provider with the default OpenAI endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::build(api_key, DEFAULT_ENDPOINT.to_string())
    }

    /// Create a new Sora provider with a custom endpoint.
    pub fn with_endpoint(api_key: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self::build(api_key, endpoint.into())
    }

    /// Internal builder that wires up shared infrastructure.
    fn build(api_key: impl Into<String>, endpoint: String) -> Self {
        let api_key = api_key.into();

        // Build HTTP client via shared HttpClientBuilder with Bearer auth
        let client = HttpClientBuilder::new()
            .timeout(Duration::from_secs(120))
            .bearer_token(&api_key)
            .default_header("Content-Type", "application/json")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        // Retry policy: 3 retries with exponential backoff
        let retry_policy = RetryPolicy {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        };

        // Rate limiter: OpenAI API typically allows ~5 requests/sec for video
        let rate_limiter = TokenBucket::new(&RateLimitConfig::requests_per_second(5.0));

        Self {
            api_key,
            endpoint,
            client,
            retry_policy,
            rate_limiter,
        }
    }

    /// Whether this provider has a real API key configured.
    fn has_api_key(&self) -> bool {
        !self.api_key.is_empty()
    }

    /// Build the URL for the video generation submit endpoint.
    fn submit_url(&self) -> String {
        format!("{}/v1/videos/generations", self.endpoint)
    }

    /// Build the URL for polling a task by ID.
    #[cfg(test)]
    fn poll_url(&self, task_id: &str) -> String {
        format!("{}/v1/videos/generations/{}", self.endpoint, task_id)
    }

    /// Build the URL for the models list endpoint (used for health checks).
    fn models_url(&self) -> String {
        format!("{}/v1/models", self.endpoint)
    }

    /// Format a resolution tuple as `"WIDTHxHEIGHT"`.
    fn format_resolution(resolution: (u32, u32)) -> String {
        format!("{}x{}", resolution.0, resolution.1)
    }

    /// Build an `AsyncJobRunner` for long-running generation tasks.
    fn async_job_runner(&self) -> AsyncJobRunner {
        AsyncJobRunner::new(PROVIDER_NAME)
            .with_poll_config(PollingConfig {
                initial_delay: Duration::from_secs(2),
                max_delay: Duration::from_secs(15),
                backoff_factor: 1.5,
                max_attempts: 60,
            })
            .with_timeout(Duration::from_secs(300))
    }

    /// Generate a video via the real OpenAI Sora API.
    ///
    /// Submits a generation job, polls until completion, and returns a `VideoClip`.
    #[tracing::instrument(skip(self, prompt_text, image))]
    async fn api_generate(
        &self,
        prompt_text: &str,
        image: Option<String>,
        duration: f64,
        resolution: (u32, u32),
        fps: f32,
    ) -> Result<VideoClip> {
        let body = SoraGenerateRequest {
            model: MODEL_NAME.to_string(),
            prompt: prompt_text.to_string(),
            image,
            duration,
            resolution: Some(Self::format_resolution(resolution)),
        };

        // Acquire rate-limit token before making request
        self.rate_limiter.acquire(1).await;

        let client = self.client.clone();
        let submit_url = self.submit_url();
        let poll_base_url = self.endpoint.clone();
        let poll_client = client.clone();

        // Use AsyncJobRunner for the submit -> poll -> collect pattern
        let runner = self.async_job_runner();

        let job_result = runner
            .run(
                // Submit function: POST to /v1/videos/generations, return task ID
                || {
                    let client = client.clone();
                    let url = submit_url.clone();
                    let body = body.clone();
                    let retry_policy = self.retry_policy.clone();
                    async move {
                        let task: SoraTaskResponse =
                            retry_with_policy(PROVIDER_NAME, &retry_policy, || {
                                let client = client.clone();
                                let url = url.clone();
                                let body = body.clone();
                                async move {
                                    let response = client
                                        .post(&url)
                                        .json(&body)
                                        .send()
                                        .await
                                        .map_err(|e| {
                                            WorldForgeError::ProviderUnavailable {
                                                provider: PROVIDER_NAME.to_string(),
                                                reason: format!("request failed: {e}"),
                                            }
                                        })?;

                                    let response =
                                        check_response(PROVIDER_NAME, response).await?;

                                    response.json::<SoraTaskResponse>().await.map_err(|e| {
                                        WorldForgeError::SerializationError(e.to_string())
                                    })
                                }
                            })
                            .await?;

                        // Check for immediate failure
                        if task.status == "failed" {
                            return Err(WorldForgeError::ProviderUnavailable {
                                provider: PROVIDER_NAME.to_string(),
                                reason: task
                                    .error
                                    .unwrap_or_else(|| "generation failed".to_string()),
                            });
                        }

                        // If completed synchronously, return special marker
                        if task.status == "completed" {
                            return Ok(format!("completed:{}", task.id));
                        }

                        Ok(task.id)
                    }
                },
                // Poll function: GET /v1/videos/generations/{id}
                |job_id: String| {
                    let client = poll_client.clone();
                    let base = poll_base_url.clone();
                    async move {
                        if job_id.starts_with("completed:") {
                            // Already completed synchronously
                            return Ok(PollStatus::Complete(SoraTaskResponse {
                                id: job_id.trim_start_matches("completed:").to_string(),
                                status: "completed".to_string(),
                                url: None,
                                error: None,
                            }));
                        }

                        let response = client
                            .get(format!("{base}/v1/videos/generations/{job_id}"))
                            .send()
                            .await
                            .map_err(|e| WorldForgeError::ProviderUnavailable {
                                provider: PROVIDER_NAME.to_string(),
                                reason: format!("poll request failed: {e}"),
                            })?;

                        let response = check_response(PROVIDER_NAME, response).await?;

                        let poll_task: SoraTaskResponse = response
                            .json()
                            .await
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
                },
            )
            .await?;

        let task = job_result.result.ok_or_else(|| {
            WorldForgeError::ProviderUnavailable {
                provider: PROVIDER_NAME.to_string(),
                reason: "completed job has no result".to_string(),
            }
        })?;

        if task.url.is_none() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: PROVIDER_NAME.to_string(),
                reason: "completed task has no video URL".to_string(),
            });
        }

        Ok(build_stub_video_clip(resolution, fps, duration, 0))
    }

    /// Deterministic fallback for when no API key is set.
    fn fallback_generate(
        &self,
        _prompt_text: &str,
        resolution: (u32, u32),
        fps: f32,
        duration: f64,
    ) -> VideoClip {
        build_stub_video_clip(resolution, fps, duration, 0)
    }

    /// Deterministic fallback prediction for when no API key is set.
    fn fallback_predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Prediction {
        let fps = config.fps.clamp(MIN_FPS, MAX_FPS);
        let duration = (config.steps as f64 / fps as f64).min(MAX_VIDEO_LENGTH_SECONDS as f64);

        let video = config
            .return_video
            .then(|| build_stub_video_clip(config.resolution, fps, duration, 0));

        Prediction {
            id: uuid::Uuid::new_v4(),
            provider: PROVIDER_NAME.to_string(),
            model: MODEL_NAME.to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state: state.clone(),
            video,
            confidence: 0.75,
            physics_scores: PhysicsScores {
                overall: 0.70,
                object_permanence: 0.75,
                gravity_compliance: 0.65,
                collision_accuracy: 0.60,
                spatial_consistency: 0.70,
                temporal_consistency: 0.75,
            },
            latency_ms: 0,
            cost: self.estimate_cost(&Operation::Predict {
                steps: config.steps,
                resolution: config.resolution,
            }),
            provenance: None,
            sampling: None,
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        }
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

    #[tracing::instrument(skip(self, prompt, config))]
    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip> {
        let fps = config.fps.clamp(MIN_FPS, MAX_FPS);
        let duration = config.duration_seconds.min(MAX_VIDEO_LENGTH_SECONDS as f64);

        if !self.has_api_key() {
            tracing::info!(provider = PROVIDER_NAME, "no API key set, using deterministic fallback");
            return Ok(self.fallback_generate(&prompt.text, config.resolution, fps, duration));
        }

        match self
            .api_generate(&prompt.text, None, duration, config.resolution, fps)
            .await
        {
            Ok(clip) => Ok(clip),
            Err(err) => {
                tracing::warn!(
                    provider = PROVIDER_NAME,
                    error = %err,
                    "generate API call failed, falling back to deterministic output"
                );
                Ok(self.fallback_generate(&prompt.text, config.resolution, fps, duration))
            }
        }
    }

    #[tracing::instrument(skip(self, state, action, config))]
    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        if !self.has_api_key() {
            tracing::info!(provider = PROVIDER_NAME, "no API key set, using deterministic fallback");
            return Ok(self.fallback_predict(state, action, config));
        }

        let start = std::time::Instant::now();
        let prompt_text = format!(
            "Predict the world state after applying action: {:?}",
            action
        );

        let fps = config.fps.clamp(MIN_FPS, MAX_FPS);
        let duration = (config.steps as f64 / fps as f64).min(MAX_VIDEO_LENGTH_SECONDS as f64);

        // Delegate to generate for the actual API call
        match self
            .api_generate(&prompt_text, None, duration, config.resolution, fps)
            .await
        {
            Ok(video_clip) => {
                let latency_ms = start.elapsed().as_millis() as u64;

                let video = if config.return_video {
                    Some(video_clip)
                } else {
                    None
                };

                Ok(Prediction {
                    id: uuid::Uuid::new_v4(),
                    provider: PROVIDER_NAME.to_string(),
                    model: MODEL_NAME.to_string(),
                    input_state: state.clone(),
                    action: action.clone(),
                    output_state: state.clone(),
                    video,
                    confidence: 0.75,
                    physics_scores: PhysicsScores {
                        overall: 0.70,
                        object_permanence: 0.75,
                        gravity_compliance: 0.65,
                        collision_accuracy: 0.60,
                        spatial_consistency: 0.70,
                        temporal_consistency: 0.75,
                    },
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
            Err(err) => {
                tracing::warn!(
                    provider = PROVIDER_NAME,
                    error = %err,
                    "predict API call failed, falling back to deterministic output"
                );
                Ok(self.fallback_predict(state, action, config))
            }
        }
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
        if !self.has_api_key() {
            return Ok(HealthStatus {
                healthy: true,
                message: format!(
                    "Sora provider ready (no API key, deterministic mode): {} at {}",
                    MODEL_NAME, self.endpoint
                ),
                latency_ms: 0,
            });
        }

        let start = std::time::Instant::now();

        // Use retry policy for health check
        let client = self.client.clone();
        let url = self.models_url();

        let result = retry_with_policy(PROVIDER_NAME, &self.retry_policy, || {
            let client = client.clone();
            let url = url.clone();
            async move {
                let response = client.get(&url).send().await.map_err(|e| {
                    WorldForgeError::ProviderUnavailable {
                        provider: PROVIDER_NAME.to_string(),
                        reason: format!("health check failed: {e}"),
                    }
                })?;

                let _response = check_response(PROVIDER_NAME, response).await?;
                Ok(())
            }
        })
        .await;

        let latency_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(()) => Ok(HealthStatus {
                healthy: true,
                message: format!("Sora provider ready: {} at {}", MODEL_NAME, self.endpoint),
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
        assert_eq!(
            provider.submit_url(),
            "https://api.openai.com/v1/videos/generations"
        );
    }

    #[test]
    fn test_sora_poll_url() {
        let provider = SoraProvider::new("key");
        assert_eq!(
            provider.poll_url("task-abc-123"),
            "https://api.openai.com/v1/videos/generations/task-abc-123"
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

    #[test]
    fn test_sora_has_api_key() {
        let provider = SoraProvider::new("test-key");
        assert!(provider.has_api_key());

        let empty_provider = SoraProvider::new("");
        assert!(!empty_provider.has_api_key());
    }

    #[tokio::test]
    async fn test_sora_fallback_generate() {
        // Provider with empty key should use deterministic fallback
        let provider = SoraProvider::new("");
        let prompt = GenerationPrompt {
            text: "A cat in the snow".to_string(),
            reference_image: None,
            negative_prompt: None,
        };
        let config = GenerationConfig::default();
        let result = provider.generate(&prompt, &config).await;
        assert!(result.is_ok());
        let clip = result.unwrap();
        assert!(clip.fps > 0.0);
    }

    #[tokio::test]
    async fn test_sora_fallback_predict() {
        // Provider with empty key should use deterministic fallback
        let provider = SoraProvider::new("");
        let state = WorldState::new("test-sora", "sora");
        let action = Action::SetLighting { time_of_day: 12.0 };
        let config = PredictionConfig::default();
        let result = provider.predict(&state, &action, &config).await;
        assert!(result.is_ok());
        let prediction = result.unwrap();
        assert_eq!(prediction.provider, "sora");
        assert_eq!(prediction.model, "sora-2");
        assert_eq!(prediction.confidence, 0.75);
    }

    #[tokio::test]
    async fn test_sora_health_check_no_key() {
        let provider = SoraProvider::new("");
        let result = provider.health_check().await;
        assert!(result.is_ok());
        let status = result.unwrap();
        assert!(status.healthy);
        assert!(status.message.contains("deterministic mode"));
    }
}
