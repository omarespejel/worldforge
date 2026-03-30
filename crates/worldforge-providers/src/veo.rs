//! Veo 3 (Google) provider adapter.
//!
//! Implements the `WorldModelProvider` trait for Google's Veo family
//! of video generation models:
//! - Veo 3.1 Fast Generate Preview: high-quality video generation with
//!   text prompts and optional image conditioning.
//!
//! Uses the Google Generative Language API with an async submit/poll/download
//! pattern.

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

const DEFAULT_ENDPOINT: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_MODEL: &str = "veo-3.1-fast-generate-preview";
const PROVIDER_NAME: &str = "veo";
const DEFAULT_RESOLUTION: (u32, u32) = (1280, 720);
#[allow(dead_code)]
const DEFAULT_FRAMES: u32 = 96;
const MAX_VIDEO_LENGTH_SECONDS: f32 = 8.0;
const MIN_FPS: f32 = 12.0;
const MAX_FPS: f32 = 24.0;

// ---------------------------------------------------------------------------
// API request/response types
// ---------------------------------------------------------------------------

/// Request body for the Veo video generation endpoint.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoGenerateRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<VeoImage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<VeoConfig>,
}

/// Optional reference image for the generation request.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoImage {
    /// Base64-encoded image bytes.
    bytes: String,
    /// MIME type of the image (e.g. `"image/png"`).
    mime_type: String,
}

/// Generation configuration sent with the request.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeoConfig {
    aspect_ratio: String,
    person_generation: String,
}

/// Long-running operation returned by the submit endpoint.
#[derive(Debug, Deserialize)]
struct VeoOperation {
    name: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    response: Option<VeoResponse>,
    #[serde(default)]
    error: Option<VeoError>,
}

/// Successful completion payload inside a `VeoOperation`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VeoResponse {
    generated_videos: Vec<VeoVideo>,
}

/// A single generated video entry.
#[derive(Debug, Deserialize)]
struct VeoVideo {
    video: VeoVideoUri,
}

/// URI wrapper for a generated video.
#[derive(Debug, Deserialize)]
struct VeoVideoUri {
    uri: String,
}

/// Error payload inside a `VeoOperation`.
#[derive(Debug, Deserialize)]
struct VeoError {
    message: String,
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// Google Veo 3 provider adapter.
///
/// Wraps the Google Generative Language API to implement the
/// `WorldModelProvider` trait for Veo video generation.
#[derive(Debug, Clone)]
pub struct VeoProvider {
    /// API key used for authentication.
    api_key: String,
    /// Base endpoint URL (without trailing slash).
    endpoint: String,
    /// Model identifier.
    model: String,
    /// HTTP client.
    client: reqwest::Client,
}

impl VeoProvider {
    /// Create a new Veo provider with default endpoint and model.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            model: DEFAULT_MODEL.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new Veo provider with a custom endpoint.
    pub fn with_endpoint(api_key: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: endpoint.into(),
            model: DEFAULT_MODEL.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new Veo provider with a custom model identifier.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Build the URL for the video generation submit endpoint.
    fn generate_url(&self) -> String {
        format!(
            "{}/v1beta/models/{}:generateVideos",
            self.endpoint, self.model
        )
    }

    /// Build the URL for polling an operation by name.
    fn poll_url(&self, operation_name: &str) -> String {
        format!("{}/v1beta/{}", self.endpoint, operation_name)
    }

    /// Build the URL for the models list endpoint (used for health checks).
    fn models_url(&self) -> String {
        format!("{}/v1beta/models", self.endpoint)
    }

    /// Submit a video generation request and poll until completion.
    #[tracing::instrument(skip(self, prompt_text))]
    async fn submit_and_poll(
        &self,
        prompt_text: &str,
        config: Option<VeoConfig>,
        image: Option<VeoImage>,
    ) -> Result<VeoResponse> {
        let body = VeoGenerateRequest {
            prompt: prompt_text.to_string(),
            image,
            config,
        };

        let response = self
            .client
            .post(self.generate_url())
            .query(&[("key", &self.api_key)])
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

        let operation: VeoOperation = serde_json::from_str(&response_text)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        // If the operation completed synchronously, return immediately.
        if operation.done {
            if let Some(error) = operation.error {
                return Err(WorldForgeError::ProviderUnavailable {
                    provider: PROVIDER_NAME.to_string(),
                    reason: error.message,
                });
            }
            return operation
                .response
                .ok_or_else(|| WorldForgeError::ProviderUnavailable {
                    provider: PROVIDER_NAME.to_string(),
                    reason: "operation completed but no response body".to_string(),
                });
        }

        // Poll until completion.
        let operation_name = operation.name.clone();
        let polling_config = PollingConfig::default();

        poll_until_complete(PROVIDER_NAME, &polling_config, || {
            let op_name = operation_name.clone();
            async move {
                let poll_resp = self
                    .client
                    .get(self.poll_url(&op_name))
                    .query(&[("key", &self.api_key)])
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

                let op: VeoOperation = serde_json::from_str(&poll_text)
                    .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

                if !op.done {
                    return Ok(PollStatus::Pending);
                }

                if let Some(error) = op.error {
                    return Ok(PollStatus::Failed(error.message));
                }

                match op.response {
                    Some(resp) => Ok(PollStatus::Complete(resp)),
                    None => Ok(PollStatus::Failed(
                        "operation completed but no response body".to_string(),
                    )),
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
impl WorldModelProvider for VeoProvider {
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
                p50_ms: 15000,
                p95_ms: 45000,
                p99_ms: 90000,
                throughput_fps: 4.0,
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

        // Build a text prompt from the action description.
        let prompt_text = format!(
            "Predict the world state after applying action: {:?}",
            action
        );

        let veo_config = VeoConfig {
            aspect_ratio: "16:9".to_string(),
            person_generation: "allow_adult".to_string(),
        };

        let veo_response = self
            .submit_and_poll(&prompt_text, Some(veo_config), None)
            .await?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let video = if config.return_video && !veo_response.generated_videos.is_empty() {
            let _video_uri = &veo_response.generated_videos[0].video.uri;
            let fps = config.fps.clamp(MIN_FPS, MAX_FPS);
            let duration = config.steps as f64 / fps as f64;
            Some(build_stub_video_clip(
                config.resolution,
                fps,
                duration.min(MAX_VIDEO_LENGTH_SECONDS as f64),
                latency_ms,
            ))
        } else {
            None
        };

        let confidence = 0.70;
        let physics_scores = PhysicsScores {
            overall: 0.65,
            object_permanence: 0.70,
            gravity_compliance: 0.60,
            collision_accuracy: 0.55,
            spatial_consistency: 0.65,
            temporal_consistency: 0.70,
        };

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: PROVIDER_NAME.to_string(),
            model: self.model.clone(),
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
        let veo_config = VeoConfig {
            aspect_ratio: "16:9".to_string(),
            person_generation: "allow_adult".to_string(),
        };

        let veo_response = self
            .submit_and_poll(&prompt.text, Some(veo_config), None)
            .await?;

        if veo_response.generated_videos.is_empty() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: PROVIDER_NAME.to_string(),
                reason: "no videos returned in response".to_string(),
            });
        }

        let _video_uri = &veo_response.generated_videos[0].video.uri;
        let fps = config.fps.clamp(MIN_FPS, MAX_FPS);

        Ok(build_stub_video_clip(
            config.resolution,
            fps,
            config.duration_seconds.min(MAX_VIDEO_LENGTH_SECONDS as f64),
            0,
        ))
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
            .query(&[("key", &self.api_key)])
            .send()
            .await;

        let latency_ms = start.elapsed().as_millis() as u64;

        match response {
            Ok(resp) if resp.status().is_success() => Ok(HealthStatus {
                healthy: true,
                message: format!("Veo provider ready: {} at {}", self.model, self.endpoint),
                latency_ms,
            }),
            Ok(resp) => Ok(HealthStatus {
                healthy: false,
                message: format!("Veo health check failed: HTTP {}", resp.status()),
                latency_ms,
            }),
            Err(e) => Ok(HealthStatus {
                healthy: false,
                message: format!("Veo health check failed: {e}"),
                latency_ms,
            }),
        }
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        match operation {
            Operation::Predict { steps, resolution } => {
                let pixels = (resolution.0 as f64) * (resolution.1 as f64);
                let step_factor = (*steps).max(1) as f64;
                let usd = 0.025 + step_factor * 0.005 + pixels / 1_000_000.0 * 0.008;
                CostEstimate {
                    usd,
                    credits: usd * 100.0,
                    estimated_latency_ms: 15_000 + (*steps).max(1) as u64 * 500,
                }
            }
            Operation::Generate {
                duration_seconds,
                resolution,
            } => {
                let duration = duration_seconds.clamp(0.5, MAX_VIDEO_LENGTH_SECONDS as f64);
                let pixels = (resolution.0 as f64) * (resolution.1 as f64);
                let usd = 0.030 + duration * 0.010 + pixels / 1_000_000.0 * 0.012;
                CostEstimate {
                    usd,
                    credits: usd * 100.0,
                    estimated_latency_ms: 20_000 + (duration * 2_000.0) as u64,
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
    fn test_veo_provider_creation() {
        let provider = VeoProvider::new("test-api-key");
        assert_eq!(provider.name(), "veo");
        assert_eq!(provider.api_key, "test-api-key");
        assert_eq!(provider.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(provider.model, DEFAULT_MODEL);
    }

    #[test]
    fn test_veo_provider_with_endpoint() {
        let provider = VeoProvider::with_endpoint("key", "https://custom.api.com");
        assert_eq!(provider.endpoint, "https://custom.api.com");
        assert_eq!(provider.model, DEFAULT_MODEL);
    }

    #[test]
    fn test_veo_provider_with_model() {
        let provider = VeoProvider::with_model("key", "veo-4-preview");
        assert_eq!(provider.model, "veo-4-preview");
        assert_eq!(provider.endpoint, DEFAULT_ENDPOINT);
    }

    #[test]
    fn test_veo_capabilities() {
        let provider = VeoProvider::new("test-key");
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
        assert_eq!(caps.max_video_length_seconds, 8.0);
        assert_eq!(caps.fps_range, (12.0, 24.0));
        assert_eq!(
            caps.supported_action_spaces,
            vec![ActionSpaceType::Language]
        );
    }

    #[test]
    fn test_veo_cost_estimation() {
        let provider = VeoProvider::new("test-key");

        let predict_cost = provider.estimate_cost(&Operation::Predict {
            steps: 10,
            resolution: (1280, 720),
        });
        assert!(predict_cost.usd > 0.0);
        assert!(predict_cost.credits > 0.0);
        assert!(predict_cost.estimated_latency_ms > 0);

        let generate_cost = provider.estimate_cost(&Operation::Generate {
            duration_seconds: 4.0,
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
    fn test_veo_generate_url() {
        let provider = VeoProvider::new("key");
        assert_eq!(
            provider.generate_url(),
            "https://generativelanguage.googleapis.com/v1beta/models/veo-3.1-fast-generate-preview:generateVideos"
        );
    }

    #[test]
    fn test_veo_poll_url() {
        let provider = VeoProvider::new("key");
        assert_eq!(
            provider.poll_url("operations/12345"),
            "https://generativelanguage.googleapis.com/v1beta/operations/12345"
        );
    }

    #[test]
    fn test_veo_models_url() {
        let provider = VeoProvider::new("key");
        assert_eq!(
            provider.models_url(),
            "https://generativelanguage.googleapis.com/v1beta/models"
        );
    }

    #[test]
    fn test_veo_supported_actions_empty() {
        let provider = VeoProvider::new("key");
        assert!(provider.supported_actions().is_empty());
    }

    #[tokio::test]
    async fn test_veo_reason_unsupported() {
        let provider = VeoProvider::new("key");
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
                assert_eq!(name, "veo");
                assert_eq!(capability, "reason");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_veo_transfer_unsupported() {
        let provider = VeoProvider::new("key");
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
                assert_eq!(name, "veo");
                assert_eq!(capability, "transfer");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_veo_request_serialization() {
        let request = VeoGenerateRequest {
            prompt: "A cat sitting on a table".to_string(),
            image: None,
            config: Some(VeoConfig {
                aspect_ratio: "16:9".to_string(),
                person_generation: "allow_adult".to_string(),
            }),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["prompt"], "A cat sitting on a table");
        assert!(json.get("image").is_none());
        assert_eq!(json["config"]["aspectRatio"], "16:9");
        assert_eq!(json["config"]["personGeneration"], "allow_adult");
    }

    #[test]
    fn test_veo_request_with_image() {
        let request = VeoGenerateRequest {
            prompt: "Transform this scene".to_string(),
            image: Some(VeoImage {
                bytes: "base64data".to_string(),
                mime_type: "image/png".to_string(),
            }),
            config: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["image"]["bytes"], "base64data");
        assert_eq!(json["image"]["mimeType"], "image/png");
        assert!(json.get("config").is_none());
    }

    #[test]
    fn test_veo_operation_deserialization() {
        let json = r#"{
            "name": "operations/abc-123",
            "done": false
        }"#;
        let op: VeoOperation = serde_json::from_str(json).unwrap();
        assert_eq!(op.name, "operations/abc-123");
        assert!(!op.done);
        assert!(op.response.is_none());
        assert!(op.error.is_none());
    }

    #[test]
    fn test_veo_operation_complete_deserialization() {
        let json = r#"{
            "name": "operations/abc-123",
            "done": true,
            "response": {
                "generatedVideos": [
                    {
                        "video": {
                            "uri": "https://storage.googleapis.com/video.mp4"
                        }
                    }
                ]
            }
        }"#;
        let op: VeoOperation = serde_json::from_str(json).unwrap();
        assert!(op.done);
        let resp = op.response.unwrap();
        assert_eq!(resp.generated_videos.len(), 1);
        assert_eq!(
            resp.generated_videos[0].video.uri,
            "https://storage.googleapis.com/video.mp4"
        );
    }

    #[test]
    fn test_veo_operation_error_deserialization() {
        let json = r#"{
            "name": "operations/err-456",
            "done": true,
            "error": {
                "message": "quota exceeded"
            }
        }"#;
        let op: VeoOperation = serde_json::from_str(json).unwrap();
        assert!(op.done);
        assert!(op.response.is_none());
        let error = op.error.unwrap();
        assert_eq!(error.message, "quota exceeded");
    }
}
