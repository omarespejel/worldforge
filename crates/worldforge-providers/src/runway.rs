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
use worldforge_core::types::{DType, Device, Frame, SimTime, Tensor, TensorData, VideoClip};

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

#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
struct RunwayResponsePayload {
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    score: Option<f32>,
    #[serde(default)]
    processing_time_ms: Option<u64>,
    #[serde(default)]
    latency_ms: Option<u64>,
    #[serde(default)]
    physics_scores: Option<RunwayPhysicsScores>,
    #[serde(default)]
    output_state: Option<WorldState>,
    #[serde(default)]
    state: Option<WorldState>,
    #[serde(default)]
    video_url: Option<String>,
    #[serde(default)]
    media_url: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    resolution: Option<[u32; 2]>,
    #[serde(default)]
    fps: Option<f32>,
    #[serde(default)]
    duration_seconds: Option<f64>,
    #[serde(default)]
    video: Option<RunwayMediaPayload>,
    #[serde(default)]
    media: Vec<RunwayMediaPayload>,
    #[serde(default)]
    frames: Vec<RunwayMediaPayload>,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    evidence: Vec<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    transcript: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
struct RunwayMediaPayload {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    href: Option<String>,
    #[serde(default)]
    media_url: Option<String>,
    #[serde(default)]
    frame_url: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    frame_count: Option<usize>,
    #[serde(default)]
    resolution: Option<[u32; 2]>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    fps: Option<f32>,
    #[serde(default)]
    duration_seconds: Option<f64>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    timestamp_seconds: Option<f64>,
    #[serde(default)]
    timestamp: Option<f64>,
    #[serde(default)]
    frames: Vec<RunwayMediaPayload>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RunwayPhysicsScores {
    #[serde(default)]
    overall: Option<f32>,
    #[serde(default)]
    object_permanence: Option<f32>,
    #[serde(default)]
    gravity_compliance: Option<f32>,
    #[serde(default)]
    collision_accuracy: Option<f32>,
    #[serde(default)]
    spatial_consistency: Option<f32>,
    #[serde(default)]
    temporal_consistency: Option<f32>,
}

impl RunwayPhysicsScores {
    fn to_physics_scores(&self, fallback_overall: Option<f32>) -> PhysicsScores {
        let overall = self.overall.or(fallback_overall).unwrap_or(0.0);
        PhysicsScores {
            overall,
            object_permanence: self.object_permanence.unwrap_or(overall),
            gravity_compliance: self.gravity_compliance.unwrap_or(overall),
            collision_accuracy: self.collision_accuracy.unwrap_or(overall),
            spatial_consistency: self.spatial_consistency.unwrap_or(overall),
            temporal_consistency: self.temporal_consistency.unwrap_or(overall),
        }
    }
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

    fn model_name(&self) -> &'static str {
        match self.model {
            RunwayModel::Gwm1Worlds => "gwm-1-worlds",
            RunwayModel::Gwm1Robotics => "gwm-1-robotics",
            RunwayModel::Gwm1Avatars => "gwm-1-avatars",
        }
    }

    fn unwrap_response_payload(mut value: serde_json::Value) -> serde_json::Value {
        for key in ["data", "result", "output", "response", "payload"] {
            let next = value
                .as_object()
                .and_then(|object| object.get(key))
                .cloned();
            if let Some(next) = next {
                value = next;
            }
        }
        value
    }

    fn parse_response_payload(body: serde_json::Value) -> Result<RunwayResponsePayload> {
        let value = Self::unwrap_response_payload(body);
        serde_json::from_value::<RunwayResponsePayload>(value)
            .map_err(|error| WorldForgeError::SerializationError(error.to_string()))
    }

    fn response_media_payloads(payload: &RunwayResponsePayload) -> Vec<RunwayMediaPayload> {
        let mut media = Vec::new();
        if let Some(video) = payload.video.clone() {
            Self::push_media_payload(&mut media, video);
        }
        for item in payload.media.iter().cloned() {
            Self::push_media_payload(&mut media, item);
        }
        for item in payload.frames.iter().cloned() {
            Self::push_media_payload(&mut media, item);
        }
        if let Some(url) = payload.video_url.clone() {
            media.push(RunwayMediaPayload {
                url: Some(url),
                ..Default::default()
            });
        }
        if let Some(url) = payload.media_url.clone() {
            media.push(RunwayMediaPayload {
                url: Some(url),
                ..Default::default()
            });
        }
        if let Some(url) = payload.url.clone() {
            media.push(RunwayMediaPayload {
                url: Some(url),
                ..Default::default()
            });
        }
        media
    }

    fn push_media_payload(out: &mut Vec<RunwayMediaPayload>, payload: RunwayMediaPayload) {
        if payload.url.is_some()
            || payload.href.is_some()
            || payload.media_url.is_some()
            || payload.frame_url.is_some()
            || payload.id.is_some()
        {
            out.push(payload.clone());
        }
        for nested in payload.frames {
            Self::push_media_payload(out, nested);
        }
    }

    fn response_resolution(
        payload: &RunwayResponsePayload,
        media: &[RunwayMediaPayload],
        fallback: (u32, u32),
    ) -> (u32, u32) {
        payload
            .resolution
            .map(|resolution| (resolution[0].max(1), resolution[1].max(1)))
            .or_else(|| {
                media.iter().find_map(|item| {
                    item.resolution
                        .map(|resolution| (resolution[0].max(1), resolution[1].max(1)))
                        .or_else(|| match (item.width, item.height) {
                            (Some(width), Some(height)) => Some((width.max(1), height.max(1))),
                            _ => None,
                        })
                })
            })
            .unwrap_or(fallback)
    }

    fn response_fps(
        payload: &RunwayResponsePayload,
        media: &[RunwayMediaPayload],
        fallback: f32,
    ) -> f32 {
        payload
            .fps
            .or_else(|| media.iter().find_map(|item| item.fps))
            .unwrap_or(fallback)
            .max(1.0)
    }

    fn response_duration(
        payload: &RunwayResponsePayload,
        media: &[RunwayMediaPayload],
        fallback: f64,
    ) -> f64 {
        payload
            .duration_seconds
            .or_else(|| {
                media
                    .iter()
                    .find_map(|item| item.duration_seconds.or(item.duration))
            })
            .unwrap_or(fallback)
            .max(0.0)
    }

    fn media_anchor(payload: &RunwayMediaPayload, fallback: &str, index: usize) -> String {
        payload
            .url
            .clone()
            .or_else(|| payload.href.clone())
            .or_else(|| payload.media_url.clone())
            .or_else(|| payload.frame_url.clone())
            .or_else(|| payload.id.clone())
            .unwrap_or_else(|| format!("{fallback}#{index}"))
    }

    fn synthetic_frame(
        anchor: String,
        resolution: (u32, u32),
        step: u64,
        seconds: f64,
        fps: f32,
    ) -> Frame {
        let width = resolution.0.max(1);
        let height = resolution.1.max(1);
        let (preview_width, preview_height) = if width <= 96 {
            (width, height)
        } else {
            let scale = 96.0 / width as f32;
            let preview_height = ((height as f32 * scale).round() as u32).max(1);
            (96, preview_height)
        };
        Frame {
            data: Tensor {
                data: TensorData::UInt8(vec![
                    0;
                    preview_width as usize * preview_height as usize * 3
                ]),
                shape: vec![preview_height as usize, preview_width as usize, 3],
                dtype: DType::UInt8,
                device: Device::Remote(anchor),
            },
            timestamp: SimTime {
                step,
                seconds,
                dt: 1.0 / fps.max(1.0) as f64,
            },
            camera: None,
            depth: None,
            segmentation: None,
        }
    }

    fn build_video_clip(
        payload: &RunwayResponsePayload,
        media: &[RunwayMediaPayload],
        fallback: String,
        resolution: (u32, u32),
        fps: f32,
        duration: f64,
    ) -> VideoClip {
        let resolved_resolution = Self::response_resolution(payload, media, resolution);
        let resolved_fps = Self::response_fps(payload, media, fps);
        let resolved_duration = Self::response_duration(payload, media, duration);

        let mut frames = Vec::new();
        for (index, item) in media.iter().enumerate() {
            let frame_count = item.frame_count.unwrap_or(1).max(1);
            let frame_resolution = item
                .resolution
                .map(|resolution| (resolution[0].max(1), resolution[1].max(1)))
                .or_else(|| match (item.width, item.height) {
                    (Some(width), Some(height)) => Some((width.max(1), height.max(1))),
                    _ => None,
                })
                .unwrap_or(resolved_resolution);
            let item_duration = item
                .duration_seconds
                .or(item.duration)
                .unwrap_or(resolved_duration.max(frame_count as f64 / resolved_fps as f64));
            let base_seconds = item
                .timestamp_seconds
                .or(item.timestamp)
                .unwrap_or_else(|| index as f64 * item_duration / frame_count as f64);
            let anchor = Self::media_anchor(item, &fallback, index);

            for frame_index in 0..frame_count {
                let seconds = if frame_count == 1 {
                    base_seconds
                } else {
                    base_seconds + (item_duration / frame_count as f64) * frame_index as f64
                };
                frames.push(Self::synthetic_frame(
                    if frame_count == 1 {
                        anchor.clone()
                    } else {
                        format!("{anchor}#frame={frame_index}")
                    },
                    frame_resolution,
                    frames.len() as u64,
                    seconds,
                    resolved_fps,
                ));
            }
        }

        if frames.is_empty() {
            frames.push(Self::synthetic_frame(
                fallback,
                resolved_resolution,
                0,
                0.0,
                resolved_fps,
            ));
        }

        VideoClip {
            frames,
            fps: resolved_fps,
            resolution: resolved_resolution,
            duration: resolved_duration.max(1.0 / resolved_fps as f64),
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
        let start = std::time::Instant::now();

        let request_body = serde_json::json!({
            "model": "gwm-1-robotics",
            "action": command,
            "num_frames": config.steps * (config.fps as u32),
            "resolution": [config.resolution.0, config.resolution.1],
            "return_video": config.return_video,
            "return_depth": config.return_depth,
            "return_segmentation": config.return_segmentation,
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

        let latency_ms = start.elapsed().as_millis() as u64;
        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;
        let response_payload = Self::parse_response_payload(response_body)?;
        let media = Self::response_media_payloads(&response_payload);

        let output_state = if let Some(output_state) = response_payload
            .output_state
            .clone()
            .or(response_payload.state.clone())
        {
            output_state
        } else {
            let mut output_state = state.clone();
            output_state.time.step += config.steps as u64;
            output_state.time.seconds += config.steps as f64 / config.fps as f64;
            output_state.time.dt = 1.0 / config.fps as f64;
            output_state
        };

        let physics_scores = response_payload
            .physics_scores
            .as_ref()
            .map(|scores| {
                scores.to_physics_scores(response_payload.confidence.or(response_payload.score))
            })
            .unwrap_or_else(|| {
                let confidence = response_payload
                    .confidence
                    .or(response_payload.score)
                    .unwrap_or(0.0);
                PhysicsScores {
                    overall: confidence,
                    object_permanence: confidence,
                    gravity_compliance: confidence,
                    collision_accuracy: confidence,
                    spatial_consistency: confidence,
                    temporal_consistency: confidence,
                }
            });
        let confidence = response_payload
            .confidence
            .or(response_payload.score)
            .unwrap_or(physics_scores.overall);
        let video = if config.return_video {
            Some(Self::build_video_clip(
                &response_payload,
                &media,
                response_payload
                    .request_id
                    .clone()
                    .or_else(|| response_payload.video_url.clone())
                    .or_else(|| response_payload.media_url.clone())
                    .or_else(|| response_payload.url.clone())
                    .or_else(|| response_payload.status.clone())
                    .unwrap_or_else(|| format!("runway://{}/predict", self.model_name())),
                config.resolution,
                config.fps,
                config.steps as f64 / config.fps as f64,
            ))
        } else {
            None
        };

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: "runway".to_string(),
            model: "gwm-1-robotics".to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video,
            confidence,
            physics_scores,
            latency_ms: response_payload
                .processing_time_ms
                .or(response_payload.latency_ms)
                .unwrap_or(latency_ms),
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

        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;
        let response_payload = Self::parse_response_payload(response_body)?;
        let media = Self::response_media_payloads(&response_payload);
        Ok(Self::build_video_clip(
            &response_payload,
            &media,
            response_payload
                .request_id
                .clone()
                .or_else(|| response_payload.video_url.clone())
                .or_else(|| response_payload.media_url.clone())
                .or_else(|| response_payload.url.clone())
                .or_else(|| response_payload.status.clone())
                .unwrap_or_else(|| format!("runway://{}/generate", self.model_name())),
            config.resolution,
            config.fps,
            config.duration_seconds,
        ))
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "runway".to_string(),
            capability: "reason (use Cosmos Reason as fallback)".to_string(),
        })
    }

    async fn transfer(
        &self,
        source: &VideoClip,
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

        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;
        let response_payload = Self::parse_response_payload(response_body)?;
        let media = Self::response_media_payloads(&response_payload);
        let duration = if let Some(duration) = response_payload.duration_seconds {
            duration
        } else {
            source
                .duration
                .max(source.frames.len() as f64 / config.fps.max(1.0) as f64)
        };
        Ok(Self::build_video_clip(
            &response_payload,
            &media,
            response_payload
                .request_id
                .clone()
                .or_else(|| response_payload.video_url.clone())
                .or_else(|| response_payload.media_url.clone())
                .or_else(|| response_payload.url.clone())
                .or_else(|| response_payload.status.clone())
                .unwrap_or_else(|| format!("runway://{}/transfer", self.model_name())),
            config.resolution,
            config.fps,
            duration,
        ))
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
    use std::io::{BufRead, Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;
    use worldforge_core::action::ActionTranslator;
    use worldforge_core::types::{Position, SimTime};

    fn spawn_response_server(
        response_body: String,
    ) -> (String, mpsc::Receiver<String>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());

            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
                let lower = line.to_ascii_lowercase();
                if let Some(value) = lower.strip_prefix("content-length:") {
                    content_length = value.trim().parse().unwrap_or(0);
                }
            }

            let mut request_body = vec![0u8; content_length];
            if content_length > 0 {
                reader.read_exact(&mut request_body).unwrap();
            }
            tx.send(String::from_utf8(request_body).unwrap()).unwrap();

            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        (format!("http://{}", address), rx, handle)
    }

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

    #[test]
    fn test_unwrap_response_payload() {
        let body = serde_json::json!({
            "data": {
                "result": {
                    "url": "https://example.com/video.mp4"
                }
            }
        });
        let payload = RunwayProvider::parse_response_payload(body).unwrap();
        let media = RunwayProvider::response_media_payloads(&payload);
        assert_eq!(media.len(), 1);
        assert_eq!(
            media[0].url.as_deref(),
            Some("https://example.com/video.mp4")
        );
    }

    #[tokio::test]
    async fn test_predict_maps_response_payload() {
        let state = {
            let mut world = WorldState::new("runway-test", "runway");
            world.time = worldforge_core::types::SimTime {
                step: 2,
                seconds: 0.5,
                dt: 0.25,
            };
            world
        };
        let (endpoint, request_rx, handle) = spawn_response_server(
            serde_json::json!({
                "result": {
                    "request_id": "pred-123",
                    "confidence": 0.82,
                    "score": 0.91,
                    "processing_time_ms": 123,
                    "physics_scores": {
                        "overall": 0.77,
                        "object_permanence": 0.8
                    },
                    "output_state": state,
                    "video_url": "https://cdn.runwayml.com/prediction.mp4"
                }
            })
            .to_string(),
        );
        let provider = RunwayProvider::with_endpoint(RunwayModel::Gwm1Robotics, "secret", endpoint);
        let action = Action::Move {
            target: Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            speed: 1.0,
        };
        let config = PredictionConfig {
            return_video: true,
            ..PredictionConfig::default()
        };
        let prediction = provider.predict(&state, &action, &config).await.unwrap();

        let request_body = request_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(request_body.contains(r#""return_video":true"#));
        assert!(request_body.contains(r#""return_depth":false"#));
        assert!(request_body.contains(r#""return_segmentation":false"#));
        handle.join().unwrap();

        assert_eq!(prediction.provider, "runway");
        assert_eq!(prediction.confidence, 0.82);
        assert_eq!(prediction.physics_scores.overall, 0.77);
        assert_eq!(prediction.latency_ms, 123);
        assert!(prediction.video.is_some());
        assert_eq!(
            prediction
                .video
                .as_ref()
                .and_then(|clip| clip.frames.first())
                .map(|frame| frame.data.device.clone()),
            Some(Device::Remote(
                "https://cdn.runwayml.com/prediction.mp4".to_string()
            ))
        );
        assert_eq!(prediction.output_state.time.step, 2);
        assert_eq!(prediction.output_state.time.seconds, 0.5);
    }

    #[tokio::test]
    async fn test_generate_uses_response_media_metadata() {
        let (endpoint, request_rx, handle) = spawn_response_server(
            serde_json::json!({
                "data": {
                    "request_id": "gen-42",
                    "fps": 12.0,
                    "resolution": [800, 600],
                    "duration_seconds": 4.0,
                    "frames": [
                        {"url": "https://cdn.runwayml.com/frame-1.png", "timestamp_seconds": 0.0},
                        {"url": "https://cdn.runwayml.com/frame-2.png", "timestamp_seconds": 0.5}
                    ]
                }
            })
            .to_string(),
        );
        let provider = RunwayProvider::with_endpoint(RunwayModel::Gwm1Worlds, "secret", endpoint);
        let prompt = GenerationPrompt {
            text: "A rolling cube".to_string(),
            reference_image: None,
            negative_prompt: None,
        };
        let config = GenerationConfig {
            resolution: (640, 360),
            fps: 24.0,
            duration_seconds: 2.0,
            temperature: 1.0,
            seed: None,
        };
        let clip = provider.generate(&prompt, &config).await.unwrap();

        let request_body = request_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(request_body.contains("A rolling cube"));
        handle.join().unwrap();

        assert_eq!(clip.fps, 12.0);
        assert_eq!(clip.resolution, (800, 600));
        assert_eq!(clip.duration, 4.0);
        assert_eq!(clip.frames.len(), 2);
        assert_eq!(
            clip.frames[0].data.device,
            Device::Remote("https://cdn.runwayml.com/frame-1.png".to_string())
        );
    }

    #[tokio::test]
    async fn test_transfer_falls_back_to_synthetic_remote_frames() {
        let (endpoint, request_rx, handle) = spawn_response_server(
            serde_json::json!({
                "status": "ok",
                "request_id": "transfer-7",
                "media_url": "https://cdn.runwayml.com/transfer.mp4",
                "resolution": [1024, 576],
                "fps": 18.0
            })
            .to_string(),
        );
        let provider = RunwayProvider::with_endpoint(RunwayModel::Gwm1Worlds, "secret", endpoint);
        let source = VideoClip {
            frames: vec![Frame {
                data: Tensor::zeros(vec![1, 1, 3], worldforge_core::types::DType::UInt8),
                timestamp: SimTime::default(),
                camera: None,
                depth: None,
                segmentation: None,
            }],
            fps: 10.0,
            resolution: (320, 240),
            duration: 1.2,
        };
        let controls = SpatialControls::default();
        let config = TransferConfig {
            resolution: (320, 240),
            fps: 24.0,
            control_strength: 0.8,
        };
        let clip = provider
            .transfer(&source, &controls, &config)
            .await
            .unwrap();

        let _ = request_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        handle.join().unwrap();

        assert_eq!(clip.fps, 18.0);
        assert_eq!(clip.resolution, (1024, 576));
        assert_eq!(clip.duration, 1.2);
        assert!(!clip.frames.is_empty());
        assert_eq!(
            clip.frames[0].data.device,
            Device::Remote("https://cdn.runwayml.com/transfer.mp4".to_string())
        );
    }
}
