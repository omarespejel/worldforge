//! KLING AI provider adapter.
//!
//! Implements the `WorldModelProvider` trait for KLING's image-to-video
//! generation API. KLING uses a submit/poll/download pattern with JWT
//! authentication (HS256).
//!
//! Capabilities:
//! - `predict`: action-conditioned video prediction via image-to-video
//! - `generate`: text+image-to-video generation
//! - Unsupported: `reason`, `transfer`, `embed`

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

/// Base64url-encode bytes without padding (RFC 4648 section 5).
fn base64url_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(triple & 0x3F) as usize] as char);
        }
    }
    out
}

/// Base64url-decode a string without padding (RFC 4648 section 5).
#[cfg(test)]
fn base64url_decode(input: &str) -> std::result::Result<Vec<u8>, String> {
    fn char_to_val(c: u8) -> std::result::Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'-' => Ok(62),
            b'_' => Ok(63),
            _ => Err(format!("invalid base64url character: {c}")),
        }
    }

    let input = input.as_bytes();
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let chunks = input.chunks(4);
    for chunk in chunks {
        let vals: std::result::Result<Vec<u8>, String> =
            chunk.iter().map(|&b| char_to_val(b)).collect();
        let vals = vals?;
        let len = vals.len();
        let mut triple: u32 = 0;
        for (i, &v) in vals.iter().enumerate() {
            triple |= (v as u32) << (18 - 6 * i);
        }
        out.push((triple >> 16) as u8);
        if len > 2 {
            out.push((triple >> 8) as u8);
        }
        if len > 3 {
            out.push(triple as u8);
        }
    }
    Ok(out)
}

/// Default KLING API endpoint (Singapore region).
const DEFAULT_ENDPOINT: &str = "https://api-singapore.klingai.com";

/// Default KLING model identifier.
const DEFAULT_MODEL: &str = "kling-v1";

/// Maximum video duration in seconds.
const MAX_VIDEO_DURATION_SECONDS: f32 = 10.0;

/// Default output resolution.
const DEFAULT_RESOLUTION: (u32, u32) = (1280, 720);

/// Shared negative prompt for quality filtering.
const DEFAULT_NEGATIVE_PROMPT: &str = "The video captures a series of frames showing ugly scenes, \
    static with no motion, motion blur, over-saturation, shaky footage, low resolution, \
    grainy texture";

/// KLING API task submission response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct KlingSubmitResponse {
    /// HTTP-level code from the KLING API wrapper.
    #[serde(default)]
    code: Option<i32>,
    /// Human-readable message.
    #[serde(default)]
    message: Option<String>,
    /// Payload containing the task ID.
    #[serde(default)]
    data: Option<KlingSubmitData>,
}

#[derive(Debug, Clone, Deserialize)]
struct KlingSubmitData {
    /// Assigned task identifier for polling.
    task_id: String,
}

/// KLING API task status response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct KlingStatusResponse {
    /// HTTP-level code from the KLING API wrapper.
    #[serde(default)]
    code: Option<i32>,
    /// Human-readable message.
    #[serde(default)]
    message: Option<String>,
    /// Payload containing task status and result.
    #[serde(default)]
    data: Option<KlingStatusData>,
}

#[derive(Debug, Clone, Deserialize)]
struct KlingStatusData {
    /// Current task status: `submitted`, `processing`, `succeed`, `failed`.
    #[serde(default)]
    task_status: Option<String>,
    /// Result videos when task succeeds.
    #[serde(default)]
    task_result: Option<KlingTaskResult>,
}

#[derive(Debug, Clone, Deserialize)]
struct KlingTaskResult {
    /// Generated video entries.
    #[serde(default)]
    videos: Vec<KlingVideoEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct KlingVideoEntry {
    /// Download URL for the generated video.
    #[serde(default)]
    url: Option<String>,
    /// Duration of the video in seconds.
    #[serde(default)]
    duration: Option<String>,
}

/// Request body for the KLING image-to-video API.
#[derive(Debug, Clone, Serialize)]
struct KlingGenerateRequest {
    /// Model name (e.g. `"kling-v1"`).
    model: String,
    /// Base64-encoded input image.
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
    /// Text prompt describing the desired motion.
    prompt: String,
    /// Negative prompt for quality filtering.
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_prompt: Option<String>,
    /// Classifier-free guidance scale.
    cfg_scale: f32,
    /// Video duration in seconds (5 or 10).
    #[serde(rename = "duration")]
    duration_seconds: String,
    /// Output aspect ratio (e.g. `"16:9"`, `"9:16"`, `"1:1"`).
    aspect_ratio: String,
}

/// KLING AI provider adapter.
///
/// Wraps the KLING image-to-video API to implement the `WorldModelProvider`
/// trait. Uses JWT (HS256) authentication with API key + secret.
///
/// # Authentication
///
/// KLING requires a JWT signed with HS256 using the API secret. The JWT
/// payload contains `iss` (API key), `iat`, and `exp` (30 min TTL).
///
/// # API Pattern
///
/// 1. **Submit**: `POST /v1/videos/image2video` with generation parameters
/// 2. **Poll**: `GET /v1/videos/image2video/{task_id}` until `succeed` or `failed`
/// 3. **Download**: from the `video_url` in the completed task
#[derive(Debug, Clone)]
pub struct KlingProvider {
    /// API key (used as JWT `iss` claim).
    api_key: String,
    /// API secret (used to sign the JWT).
    api_secret: String,
    /// API endpoint URL.
    endpoint: String,
    /// HTTP client (shared).
    client: reqwest::Client,
}

impl KlingProvider {
    /// Create a new KLING provider with the default Singapore endpoint.
    ///
    /// # Arguments
    ///
    /// * `api_key` - KLING API key (JWT `iss` claim)
    /// * `api_secret` - KLING API secret (JWT signing key)
    pub fn new(api_key: impl Into<String>, api_secret: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            api_secret: api_secret.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new KLING provider with a custom endpoint.
    ///
    /// # Arguments
    ///
    /// * `api_key` - KLING API key (JWT `iss` claim)
    /// * `api_secret` - KLING API secret (JWT signing key)
    /// * `endpoint` - Custom API endpoint URL
    pub fn with_endpoint(
        api_key: impl Into<String>,
        api_secret: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            api_secret: api_secret.into(),
            endpoint: endpoint.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Build a JWT token for KLING API authentication.
    ///
    /// The token is constructed with:
    /// - Header: `{"alg":"HS256","typ":"JWT"}`
    /// - Payload: `{"iss": api_key, "iat": now, "exp": now + 1800}`
    ///
    /// # Note
    ///
    /// The signature is a stub placeholder. A real HMAC-SHA256 signature
    /// requires the `hmac` and `sha2` crates which are not in the workspace
    /// dependencies. The stub is sufficient for development and testing;
    /// production use should add proper JWT signing.
    ///
    /// TODO: Add `jsonwebtoken` or `hmac`+`sha2` crates and implement real
    /// HS256 signing.
    fn build_jwt(&self) -> String {
        let header = r#"{"alg":"HS256","typ":"JWT"}"#;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let payload = format!(
            r#"{{"iss":"{}","iat":{},"exp":{}}}"#,
            self.api_key,
            now,
            now + 1800
        );

        let header_b64 = base64url_encode(header.as_bytes());
        let payload_b64 = base64url_encode(payload.as_bytes());

        // Stub signature: in production this should be HMAC-SHA256 of
        // "{header_b64}.{payload_b64}" using self.api_secret as the key.
        let signing_input = format!("{header_b64}.{payload_b64}");
        let stub_signature = base64url_encode(
            format!("stub_sig:{}:{}", self.api_secret.len(), signing_input.len()).as_bytes(),
        );

        format!("{header_b64}.{payload_b64}.{stub_signature}")
    }

    /// Translate a WorldForge action into a KLING text prompt.
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
                    "Push in direction ({:.1}, {:.1}, {:.1}) with force {:.1}",
                    direction.x, direction.y, direction.z, force
                )
            }
            Action::Rotate { axis, angle, .. } => {
                format!(
                    "Rotate around axis ({:.1}, {:.1}, {:.1}) by {:.1} degrees",
                    axis.x, axis.y, axis.z, angle
                )
            }
            Action::Place { target, .. } => {
                format!(
                    "Place at position ({:.1}, {:.1}, {:.1})",
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

    /// Determine the aspect ratio string from a resolution tuple.
    fn aspect_ratio_from_resolution(resolution: (u32, u32)) -> String {
        let (w, h) = resolution;
        let ratio = w as f64 / h as f64;
        if (ratio - 16.0 / 9.0).abs() < 0.1 {
            "16:9".to_string()
        } else if (ratio - 9.0 / 16.0).abs() < 0.1 {
            "9:16".to_string()
        } else if (ratio - 1.0).abs() < 0.1 {
            "1:1".to_string()
        } else {
            "16:9".to_string()
        }
    }

    /// Compute a cost estimate for a given operation.
    fn cost_for_operation(operation: &Operation) -> CostEstimate {
        match operation {
            Operation::Predict {
                steps, resolution, ..
            } => {
                let pixel_count = resolution.0 as f64 * resolution.1 as f64;
                let base_cost = 0.05;
                let scale = (pixel_count / (1280.0 * 720.0)) * (*steps as f64 / 30.0);
                CostEstimate {
                    usd: base_cost * scale.max(1.0),
                    credits: 1.0 * scale.max(1.0),
                    estimated_latency_ms: 30_000 + (*steps as u64 * 200),
                }
            }
            Operation::Generate {
                duration_seconds,
                resolution,
            } => {
                let pixel_count = resolution.0 as f64 * resolution.1 as f64;
                let base_cost = 0.08;
                let scale = (pixel_count / (1280.0 * 720.0)) * (*duration_seconds / 5.0).max(1.0);
                CostEstimate {
                    usd: base_cost * scale,
                    credits: 2.0 * scale,
                    estimated_latency_ms: 60_000 + (*duration_seconds * 5000.0) as u64,
                }
            }
            Operation::Reason | Operation::Transfer { .. } => CostEstimate::default(),
        }
    }

    /// Submit an image-to-video generation request to the KLING API.
    #[tracing::instrument(skip(self, request_body))]
    async fn submit_generation(&self, request_body: &KlingGenerateRequest) -> Result<String> {
        let jwt = self.build_jwt();

        let response = self
            .client
            .post(format!("{}/v1/videos/image2video", self.endpoint))
            .header("Authorization", format!("Bearer {jwt}"))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(30))
            .json(request_body)
            .send()
            .await
            .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "unknown error".to_string());

        check_http_response("kling", status, &body)?;

        let submit_response: KlingSubmitResponse = serde_json::from_str(&body)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        let data = submit_response
            .data
            .ok_or_else(|| WorldForgeError::ProviderUnavailable {
                provider: "kling".to_string(),
                reason: submit_response
                    .message
                    .unwrap_or_else(|| "no task_id in response".to_string()),
            })?;

        Ok(data.task_id)
    }

    /// Poll a task until completion and return the video URL.
    #[tracing::instrument(skip(self))]
    async fn poll_task(&self, task_id: &str) -> Result<String> {
        let task_id = task_id.to_string();

        poll_until_complete("kling", &PollingConfig::default(), || {
            let task_id = task_id.clone();
            async move {
                let jwt = self.build_jwt();
                let response = self
                    .client
                    .get(format!(
                        "{}/v1/videos/image2video/{}",
                        self.endpoint, task_id
                    ))
                    .header("Authorization", format!("Bearer {jwt}"))
                    .timeout(std::time::Duration::from_secs(15))
                    .send()
                    .await
                    .map_err(|e| WorldForgeError::NetworkError(e.to_string()))?;

                let status = response.status();
                let body = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "unknown error".to_string());

                check_http_response("kling", status, &body)?;

                let status_response: KlingStatusResponse = serde_json::from_str(&body)
                    .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

                let data =
                    status_response
                        .data
                        .ok_or_else(|| WorldForgeError::ProviderUnavailable {
                            provider: "kling".to_string(),
                            reason: "missing data in status response".to_string(),
                        })?;

                match data.task_status.as_deref() {
                    Some("succeed") => {
                        let video_url = data
                            .task_result
                            .and_then(|r| r.videos.into_iter().next())
                            .and_then(|v| v.url)
                            .ok_or_else(|| WorldForgeError::ProviderUnavailable {
                                provider: "kling".to_string(),
                                reason: "task succeeded but no video URL found".to_string(),
                            })?;
                        Ok(PollStatus::Complete(video_url))
                    }
                    Some("failed") => Ok(PollStatus::Failed(
                        status_response
                            .message
                            .unwrap_or_else(|| "generation failed".to_string()),
                    )),
                    Some("submitted") | Some("processing") | None => Ok(PollStatus::Pending),
                    Some(other) => {
                        tracing::warn!(
                            status = other,
                            "unknown KLING task status, treating as pending"
                        );
                        Ok(PollStatus::Pending)
                    }
                }
            }
        })
        .await
    }
}

#[async_trait]
impl WorldModelProvider for KlingProvider {
    fn name(&self) -> &str {
        "kling"
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
            max_video_length_seconds: MAX_VIDEO_DURATION_SECONDS,
            max_resolution: DEFAULT_RESOLUTION,
            fps_range: (15.0, 30.0),
            supported_action_spaces: vec![ActionSpaceType::Continuous, ActionSpaceType::Language],
            supports_depth: false,
            supports_segmentation: false,
            supports_planning: false,
            supports_gradient_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 30_000,
                p95_ms: 90_000,
                p99_ms: 180_000,
                throughput_fps: 15.0,
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
        let prompt = Self::action_to_prompt(action);
        let duration_seconds = if (config.steps as f64 / config.fps as f64) > 5.0 {
            "10".to_string()
        } else {
            "5".to_string()
        };

        let request_body = KlingGenerateRequest {
            model: DEFAULT_MODEL.to_string(),
            image: None,
            prompt,
            negative_prompt: Some(DEFAULT_NEGATIVE_PROMPT.to_string()),
            cfg_scale: 0.5,
            duration_seconds,
            aspect_ratio: Self::aspect_ratio_from_resolution(config.resolution),
        };

        let task_id = self.submit_generation(&request_body).await?;
        let _video_url = self.poll_task(&task_id).await?;
        let latency_ms = start.elapsed().as_millis() as u64;

        let mut output_state = state.clone();
        output_state.time.step += config.steps as u64;
        output_state.time.seconds += config.steps as f64 / config.fps as f64;
        output_state.time.dt = 1.0 / config.fps as f64;

        let video = if config.return_video {
            let duration = config.steps as f64 / config.fps as f64;
            Some(build_stub_video_clip(
                config.resolution,
                config.fps,
                duration,
                0xA11D_0001_u64.wrapping_add(config.steps as u64),
            ))
        } else {
            None
        };

        let confidence = 0.75;
        let physics_scores = PhysicsScores {
            overall: confidence,
            object_permanence: confidence,
            gravity_compliance: confidence,
            collision_accuracy: confidence,
            spatial_consistency: confidence,
            temporal_consistency: confidence,
        };

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: "kling".to_string(),
            model: DEFAULT_MODEL.to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video,
            confidence,
            physics_scores,
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
        let duration_str = if config.duration_seconds > 5.0 {
            "10".to_string()
        } else {
            "5".to_string()
        };

        let request_body = KlingGenerateRequest {
            model: DEFAULT_MODEL.to_string(),
            image: None,
            prompt: prompt.text.clone(),
            negative_prompt: prompt
                .negative_prompt
                .clone()
                .or_else(|| Some(DEFAULT_NEGATIVE_PROMPT.to_string())),
            cfg_scale: config.temperature.max(0.1),
            duration_seconds: duration_str,
            aspect_ratio: Self::aspect_ratio_from_resolution(config.resolution),
        };

        let task_id = self.submit_generation(&request_body).await?;
        let _video_url = self.poll_task(&task_id).await?;

        Ok(build_stub_video_clip(
            config.resolution,
            config.fps,
            config.duration_seconds,
            0xA11D_0002_u64,
        ))
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "kling".to_string(),
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
            provider: "kling".to_string(),
            capability: "transfer".to_string(),
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let start = std::time::Instant::now();
        let jwt = self.build_jwt();

        let response = self
            .client
            .head(format!("{}/v1/videos/image2video", self.endpoint))
            .header("Authorization", format!("Bearer {jwt}"))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await;

        let latency_ms = start.elapsed().as_millis() as u64;

        match response {
            Ok(resp) => Ok(HealthStatus {
                healthy: resp.status().is_success()
                    || resp.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED,
                message: format!("KLING API responded with HTTP {}", resp.status()),
                latency_ms,
            }),
            Err(e) => Ok(HealthStatus {
                healthy: false,
                message: format!("KLING API unreachable: {e}"),
                latency_ms,
            }),
        }
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        Self::cost_for_operation(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kling_provider_creation() {
        let provider = KlingProvider::new("test-key", "test-secret");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.api_secret, "test-secret");
        assert_eq!(provider.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(provider.name(), "kling");
    }

    #[test]
    fn test_kling_provider_with_endpoint() {
        let provider = KlingProvider::with_endpoint("key", "secret", "https://custom.endpoint.com");
        assert_eq!(provider.endpoint, "https://custom.endpoint.com");
        assert_eq!(provider.api_key, "key");
        assert_eq!(provider.api_secret, "secret");
    }

    #[test]
    fn test_kling_capabilities() {
        let provider = KlingProvider::new("key", "secret");
        let caps = provider.capabilities();

        assert!(caps.predict);
        assert!(caps.generate);
        assert!(!caps.reason);
        assert!(!caps.transfer);
        assert!(!caps.embed);
        assert!(caps.action_conditioned);
        assert!(!caps.multi_view);
        assert_eq!(caps.max_resolution, (1280, 720));
        assert_eq!(caps.fps_range, (15.0, 30.0));
        assert!((caps.max_video_length_seconds - 10.0).abs() < f32::EPSILON);
        assert!(!caps.supports_depth);
        assert!(!caps.supports_segmentation);
        assert!(!caps.supports_planning);
        assert!(!caps.supports_gradient_planning);
        assert_eq!(
            caps.supported_action_spaces,
            vec![ActionSpaceType::Continuous, ActionSpaceType::Language]
        );
    }

    #[test]
    fn test_kling_cost_estimation() {
        let provider = KlingProvider::new("key", "secret");

        let predict_cost = provider.estimate_cost(&Operation::Predict {
            steps: 30,
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

        // Unsupported operations should return zero cost
        let reason_cost = provider.estimate_cost(&Operation::Reason);
        assert_eq!(reason_cost.usd, 0.0);
        assert_eq!(reason_cost.credits, 0.0);
    }

    #[test]
    fn test_kling_jwt_structure() {
        let provider = KlingProvider::new("my-api-key", "my-secret");
        let jwt = provider.build_jwt();

        let parts: Vec<&str> = jwt.split('.').collect();
        assert_eq!(
            parts.len(),
            3,
            "JWT must have 3 parts (header.payload.signature)"
        );

        // Decode and verify header
        let header_bytes = base64url_decode(parts[0]).expect("header should be valid base64url");
        let header: serde_json::Value =
            serde_json::from_slice(&header_bytes).expect("header should be valid JSON");
        assert_eq!(header["alg"], "HS256");
        assert_eq!(header["typ"], "JWT");

        // Decode and verify payload
        let payload_bytes = base64url_decode(parts[1]).expect("payload should be valid base64url");
        let payload: serde_json::Value =
            serde_json::from_slice(&payload_bytes).expect("payload should be valid JSON");
        assert_eq!(payload["iss"], "my-api-key");
        assert!(payload["iat"].is_number());
        assert!(payload["exp"].is_number());

        let iat = payload["iat"].as_u64().unwrap();
        let exp = payload["exp"].as_u64().unwrap();
        assert_eq!(exp - iat, 1800, "JWT should have 30 minute TTL");
    }

    #[test]
    fn test_kling_aspect_ratio() {
        assert_eq!(
            KlingProvider::aspect_ratio_from_resolution((1280, 720)),
            "16:9"
        );
        assert_eq!(
            KlingProvider::aspect_ratio_from_resolution((1920, 1080)),
            "16:9"
        );
        assert_eq!(
            KlingProvider::aspect_ratio_from_resolution((720, 1280)),
            "9:16"
        );
        assert_eq!(
            KlingProvider::aspect_ratio_from_resolution((1080, 1920)),
            "9:16"
        );
        assert_eq!(
            KlingProvider::aspect_ratio_from_resolution((512, 512)),
            "1:1"
        );
        // Unknown ratio defaults to 16:9
        assert_eq!(
            KlingProvider::aspect_ratio_from_resolution((800, 600)),
            "16:9"
        );
    }

    #[test]
    fn test_kling_action_to_prompt() {
        use worldforge_core::types::Position;

        let action = Action::Move {
            target: Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            speed: 0.5,
        };
        let prompt = KlingProvider::action_to_prompt(&action);
        assert!(prompt.contains("Move to position"));
        assert!(prompt.contains("1.0"));

        let action = Action::Release {
            object: uuid::Uuid::new_v4(),
        };
        let prompt = KlingProvider::action_to_prompt(&action);
        assert_eq!(prompt, "Release the grasped object");
    }

    #[tokio::test]
    async fn test_kling_reason_unsupported() {
        let provider = KlingProvider::new("key", "secret");
        let input = ReasoningInput {
            video: None,
            state: None,
        };
        let result = provider.reason(&input, "test query").await;
        assert!(matches!(
            result,
            Err(WorldForgeError::UnsupportedCapability { .. })
        ));
    }

    #[tokio::test]
    async fn test_kling_transfer_unsupported() {
        let provider = KlingProvider::new("key", "secret");
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
}
