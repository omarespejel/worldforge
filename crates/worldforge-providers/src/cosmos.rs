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
use worldforge_core::types::{
    CameraPose, DType, Device, Frame, Pose, Position, Rotation, SimTime, Tensor, TensorData,
    VideoClip,
};

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

impl CosmosConfig {
    /// Validate the configuration.
    ///
    /// Returns an error if any configuration values are out of range.
    pub fn validate(&self) -> Result<()> {
        if self.timeout_ms == 0 {
            return Err(WorldForgeError::InvalidState(
                "Cosmos timeout_ms must be > 0".to_string(),
            ));
        }
        if self.default_num_frames == 0 || self.default_num_frames > 300 {
            return Err(WorldForgeError::InvalidState(
                "Cosmos default_num_frames must be 1..=300".to_string(),
            ));
        }
        Ok(())
    }
}

/// Request payload for the Cosmos Predict API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmosPredictRequest {
    /// Model identifier.
    pub model: String,
    /// Text prompt describing the desired action.
    pub prompt: String,
    /// Number of frames to generate.
    pub num_frames: u32,
    /// Output resolution `[width, height]`.
    pub resolution: [u32; 2],
    /// Frames per second.
    pub fps: f32,
    /// Whether to include depth maps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_depth: Option<bool>,
}

/// Response payload from the Cosmos Predict API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmosPredictResponse {
    /// Request identifier.
    pub request_id: String,
    /// Status of the prediction.
    pub status: String,
    /// Confidence score (0.0–1.0).
    pub confidence: Option<f32>,
    /// Physics plausibility scores.
    pub physics_scores: Option<CosmosPhysicsScores>,
    /// Processing time in milliseconds.
    pub processing_time_ms: Option<u64>,
}

/// Physics scores returned by the Cosmos API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmosPhysicsScores {
    pub overall: Option<f32>,
    pub object_permanence: Option<f32>,
    pub gravity_compliance: Option<f32>,
    pub collision_accuracy: Option<f32>,
    pub spatial_consistency: Option<f32>,
    pub temporal_consistency: Option<f32>,
}

impl CosmosPhysicsScores {
    /// Convert to the core PhysicsScores type.
    pub fn to_physics_scores(&self) -> PhysicsScores {
        PhysicsScores {
            overall: self.overall.unwrap_or(0.0),
            object_permanence: self.object_permanence.unwrap_or(0.0),
            gravity_compliance: self.gravity_compliance.unwrap_or(0.0),
            collision_accuracy: self.collision_accuracy.unwrap_or(0.0),
            spatial_consistency: self.spatial_consistency.unwrap_or(0.0),
            temporal_consistency: self.temporal_consistency.unwrap_or(0.0),
        }
    }
}

/// Request payload for the Cosmos Generate API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmosGenerateRequest {
    /// Model identifier.
    pub model: String,
    /// Text prompt.
    pub prompt: String,
    /// Negative prompt (things to avoid).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Output resolution `[width, height]`.
    pub resolution: [u32; 2],
    /// Frames per second.
    pub fps: f32,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Random seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CosmosMediaFrame {
    #[serde(default, alias = "index")]
    index: Option<u32>,
    #[serde(default, alias = "timestampSeconds", alias = "timestamp")]
    timestamp_seconds: Option<f64>,
    #[serde(default, alias = "url", alias = "frameUrl")]
    url: Option<String>,
    #[serde(default, alias = "dataUrl", alias = "imageUrl")]
    data_url: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CosmosVideoEnvelope {
    #[serde(default, alias = "url", alias = "videoUrl")]
    url: Option<String>,
    #[serde(default)]
    fps: Option<f32>,
    #[serde(default, alias = "durationSeconds")]
    duration_seconds: Option<f64>,
    #[serde(default, alias = "resolution")]
    resolution: Option<[u32; 2]>,
    #[serde(default, alias = "frameCount")]
    frame_count: Option<u32>,
    #[serde(default)]
    frames: Vec<CosmosMediaFrame>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default, alias = "requestId")]
    request_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CosmosReasoningEnvelope {
    #[serde(default, alias = "response", alias = "text", alias = "explanation")]
    answer: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    evidence: Vec<String>,
    #[serde(default)]
    citations: Vec<String>,
    #[serde(default)]
    rationale: Option<String>,
    #[serde(default)]
    summary: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CosmosPredictionEnvelope {
    #[serde(default, alias = "requestId")]
    request_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default, alias = "modelId")]
    model: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    physics_scores: Option<CosmosPhysicsScores>,
    #[serde(default, alias = "processingTimeMs")]
    processing_time_ms: Option<u64>,
    #[serde(default, alias = "latencyMs")]
    latency_ms: Option<u64>,
    #[serde(default, alias = "outputState")]
    output_state: Option<serde_json::Value>,
    #[serde(default, alias = "state")]
    state: Option<serde_json::Value>,
    #[serde(default)]
    video: Option<CosmosVideoEnvelope>,
    #[serde(default)]
    media: Option<CosmosVideoEnvelope>,
    #[serde(default)]
    output: Option<CosmosVideoEnvelope>,
    #[serde(default)]
    result: Option<CosmosVideoEnvelope>,
    #[serde(default, alias = "videoUrl")]
    video_url: Option<String>,
    #[serde(default, alias = "mediaUrl")]
    media_url: Option<String>,
    #[serde(default)]
    frames: Vec<CosmosMediaFrame>,
    #[serde(default)]
    reasoning: Option<CosmosReasoningEnvelope>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CosmosVideoResponse {
    #[serde(default, alias = "requestId")]
    request_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default, alias = "negativePrompt")]
    negative_prompt: Option<String>,
    #[serde(default)]
    fps: Option<f32>,
    #[serde(default, alias = "durationSeconds")]
    duration_seconds: Option<f64>,
    #[serde(default)]
    resolution: Option<[u32; 2]>,
    #[serde(default, alias = "frameCount")]
    frame_count: Option<u32>,
    #[serde(default)]
    video: Option<CosmosVideoEnvelope>,
    #[serde(default)]
    media: Option<CosmosVideoEnvelope>,
    #[serde(default)]
    output: Option<CosmosVideoEnvelope>,
    #[serde(default)]
    result: Option<CosmosVideoEnvelope>,
    #[serde(default, alias = "videoUrl")]
    video_url: Option<String>,
    #[serde(default, alias = "mediaUrl")]
    media_url: Option<String>,
    #[serde(default)]
    frames: Vec<CosmosMediaFrame>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CosmosReasoningResponse {
    #[serde(default, alias = "requestId")]
    request_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default, alias = "answer", alias = "text", alias = "response")]
    answer: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    evidence: Vec<String>,
    #[serde(default)]
    citations: Vec<String>,
    #[serde(default)]
    rationale: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    reasoning: Option<CosmosReasoningEnvelope>,
    #[serde(default)]
    output: Option<CosmosReasoningEnvelope>,
    #[serde(default)]
    result: Option<CosmosReasoningEnvelope>,
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
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn with_config(
        model: CosmosModel,
        api_key: impl Into<String>,
        endpoint: CosmosEndpoint,
        config: CosmosConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            model,
            api_key: api_key.into(),
            endpoint,
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Get the model identifier string for API requests.
    fn model_id(&self) -> &'static str {
        match self.model {
            CosmosModel::Predict2_5 => "nvidia/cosmos-predict-2.5",
            CosmosModel::Transfer2_5 => "nvidia/cosmos-transfer-2.5",
            CosmosModel::Reason2 => "nvidia/cosmos-reason-2",
            CosmosModel::Embed1 => "nvidia/cosmos-embed-1",
        }
    }

    /// Send an HTTP request with retry logic for transient failures.
    async fn send_with_retry(
        &self,
        request: reqwest::RequestBuilder,
    ) -> std::result::Result<reqwest::Response, WorldForgeError> {
        let mut last_err = WorldForgeError::NetworkError("no attempts made".to_string());
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(100 * 2u64.pow(attempt - 1));
                tokio::time::sleep(delay).await;
            }

            match request
                .try_clone()
                .ok_or_else(|| {
                    WorldForgeError::InternalError("request cannot be cloned".to_string())
                })?
                .send()
                .await
            {
                Ok(resp) => {
                    // Don't retry client errors (4xx), only server errors (5xx)
                    if resp.status().is_server_error() && attempt < self.config.max_retries {
                        last_err = WorldForgeError::ProviderUnavailable {
                            provider: "cosmos".to_string(),
                            reason: format!("HTTP {}", resp.status()),
                        };
                        continue;
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    last_err = WorldForgeError::NetworkError(e.to_string());
                    if attempt >= self.config.max_retries {
                        break;
                    }
                }
            }
        }
        Err(last_err)
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

fn prediction_id_from_request_id(request_id: Option<&str>) -> uuid::Uuid {
    request_id
        .and_then(|value| uuid::Uuid::parse_str(value).ok())
        .unwrap_or_else(uuid::Uuid::new_v4)
}

fn response_marker(parts: &[Option<&str>]) -> String {
    let mut marker = parts
        .iter()
        .filter_map(|part| part.map(str::trim))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");

    if marker.is_empty() {
        marker.push_str("cosmos");
    }

    marker
}

fn tensor_from_marker(marker: &str, width: u32, height: u32) -> Tensor {
    let width = width.max(1);
    let height = height.max(1);
    let (preview_width, preview_height) = if width <= 96 {
        (width, height)
    } else {
        let scale = 96.0 / width as f32;
        let preview_height = ((height as f32 * scale).round() as u32).max(1);
        (96, preview_height)
    };
    let size = (preview_width as usize) * (preview_height as usize) * 3;
    let bytes = if marker.is_empty() {
        vec![0u8; size]
    } else {
        marker
            .as_bytes()
            .iter()
            .copied()
            .cycle()
            .take(size)
            .collect()
    };

    Tensor {
        data: TensorData::UInt8(bytes),
        shape: vec![preview_height as usize, preview_width as usize, 3],
        dtype: DType::UInt8,
        device: Device::Cpu,
    }
}

fn default_camera() -> CameraPose {
    CameraPose {
        extrinsics: Pose {
            position: Position::default(),
            rotation: Rotation::default(),
        },
        fov: 60.0,
        near_clip: 0.1,
        far_clip: 100.0,
    }
}

fn frame_timestamp(index: usize, fps: f32, timestamp_seconds: Option<f64>) -> SimTime {
    let seconds = timestamp_seconds.unwrap_or_else(|| {
        let fps = fps.max(1.0) as f64;
        index as f64 / fps
    });
    let dt = if index == 0 {
        0.0
    } else {
        1.0 / (fps.max(1.0) as f64)
    };

    SimTime {
        step: index as u64,
        seconds,
        dt,
    }
}

fn synthesize_frame(
    marker: &str,
    index: usize,
    width: u32,
    height: u32,
    fps: f32,
    timestamp_seconds: Option<f64>,
) -> Frame {
    let seed = format!("{marker}#{index}");
    Frame {
        data: tensor_from_marker(&seed, width, height),
        timestamp: frame_timestamp(index, fps, timestamp_seconds),
        camera: Some(default_camera()),
        depth: None,
        segmentation: None,
    }
}

fn collect_markers_from_frame(frame: &CosmosMediaFrame, index: usize) -> String {
    let index_string = frame
        .index
        .map(|value| value.to_string())
        .unwrap_or_else(|| index.to_string());
    response_marker(&[
        frame.url.as_deref(),
        frame.data_url.as_deref(),
        frame.prompt.as_deref(),
        Some(index_string.as_str()),
    ])
}

fn merge_video_envelope(
    base: Option<CosmosVideoEnvelope>,
    fallback_url: Option<&str>,
    fallback_frames: &[CosmosMediaFrame],
    fallback_request_id: Option<&str>,
    fallback_status: Option<&str>,
) -> Option<CosmosVideoEnvelope> {
    let mut envelope = base.unwrap_or_default();
    if envelope.url.is_none() {
        envelope.url = fallback_url.map(ToOwned::to_owned);
    }
    if envelope.frames.is_empty() && !fallback_frames.is_empty() {
        envelope.frames = fallback_frames.to_vec();
    }
    if envelope.request_id.is_none() {
        envelope.request_id = fallback_request_id.map(ToOwned::to_owned);
    }
    if envelope.status.is_none() {
        envelope.status = fallback_status.map(ToOwned::to_owned);
    }

    let has_media = envelope.url.is_some()
        || !envelope.frames.is_empty()
        || envelope.frame_count.is_some()
        || envelope.prompt.is_some()
        || envelope.request_id.is_some();

    has_media.then_some(envelope)
}

fn materialize_video_clip(
    envelope: Option<CosmosVideoEnvelope>,
    fallback_marker: String,
    resolution: (u32, u32),
    fps: f32,
    duration_seconds: f64,
) -> VideoClip {
    let envelope = envelope.unwrap_or_default();
    let resolution = envelope
        .resolution
        .map(|value| (value[0].max(1), value[1].max(1)))
        .unwrap_or((resolution.0.max(1), resolution.1.max(1)));
    let fps = envelope.fps.unwrap_or(fps).max(1.0);
    let duration = envelope
        .duration_seconds
        .unwrap_or(duration_seconds)
        .max(0.0);

    let mut frame_markers: Vec<String> = envelope
        .frames
        .iter()
        .enumerate()
        .map(|(index, frame)| collect_markers_from_frame(frame, index))
        .collect();

    if frame_markers.is_empty() {
        if let Some(url) = envelope.url.as_deref() {
            frame_markers.push(response_marker(&[Some(url), envelope.prompt.as_deref()]));
        } else {
            frame_markers.push(fallback_marker.clone());
        }
    }

    let desired_frames = envelope
        .frame_count
        .map(|count| count.max(1) as usize)
        .unwrap_or_else(|| {
            if !envelope.frames.is_empty() {
                envelope.frames.len().max(1)
            } else {
                (duration * fps as f64).round().max(1.0) as usize
            }
        });

    let mut frames = Vec::with_capacity(desired_frames);
    for index in 0..desired_frames {
        let marker = frame_markers.get(index).cloned().unwrap_or_else(|| {
            frame_markers
                .last()
                .cloned()
                .unwrap_or_else(|| fallback_marker.clone())
        });
        let source_timestamp = envelope
            .frames
            .get(index)
            .and_then(|frame| frame.timestamp_seconds);
        frames.push(synthesize_frame(
            &marker,
            index,
            resolution.0,
            resolution.1,
            fps,
            source_timestamp,
        ));
    }

    VideoClip {
        frames,
        fps,
        resolution,
        duration: if duration > 0.0 {
            duration
        } else {
            desired_frames as f64 / (fps as f64)
        },
    }
}

fn parse_world_state(value: Option<serde_json::Value>) -> Option<WorldState> {
    value.and_then(|value| serde_json::from_value(value).ok())
}

fn reasoning_envelope(response: &CosmosReasoningResponse) -> Option<CosmosReasoningEnvelope> {
    merge_reasoning_envelope(
        response
            .reasoning
            .clone()
            .or_else(|| response.output.clone())
            .or_else(|| response.result.clone()),
        ReasoningEnvelopeSeed {
            answer: response.answer.as_deref(),
            status: response.status.as_deref(),
            evidence: &response.evidence,
            citations: &response.citations,
            summary: response.summary.as_deref(),
            rationale: response.rationale.as_deref(),
            request_id: response.request_id.as_deref(),
        },
    )
}

struct ReasoningEnvelopeSeed<'a> {
    answer: Option<&'a str>,
    status: Option<&'a str>,
    evidence: &'a [String],
    citations: &'a [String],
    summary: Option<&'a str>,
    rationale: Option<&'a str>,
    request_id: Option<&'a str>,
}

fn merge_reasoning_envelope(
    base: Option<CosmosReasoningEnvelope>,
    seed: ReasoningEnvelopeSeed<'_>,
) -> Option<CosmosReasoningEnvelope> {
    let mut envelope = base.unwrap_or_default();
    if envelope.answer.is_none() {
        envelope.answer = seed
            .answer
            .map(ToOwned::to_owned)
            .or_else(|| seed.summary.map(ToOwned::to_owned))
            .or_else(|| seed.rationale.map(ToOwned::to_owned))
            .or_else(|| seed.status.map(ToOwned::to_owned));
    }
    if envelope.summary.is_none() {
        envelope.summary = seed.summary.map(ToOwned::to_owned);
    }
    if envelope.rationale.is_none() {
        envelope.rationale = seed.rationale.map(ToOwned::to_owned);
    }
    if envelope.evidence.is_empty() && !seed.evidence.is_empty() {
        envelope.evidence = seed.evidence.to_vec();
    }
    if envelope.citations.is_empty() && !seed.citations.is_empty() {
        envelope.citations = seed.citations.to_vec();
    }
    if envelope.answer.is_none() && seed.request_id.is_some() {
        envelope.answer = seed
            .request_id
            .map(|request_id| format!("Cosmos response {request_id}"));
    }

    (envelope.answer.is_some()
        || envelope.confidence.is_some()
        || !envelope.evidence.is_empty()
        || !envelope.citations.is_empty()
        || envelope.summary.is_some()
        || envelope.rationale.is_some())
    .then_some(envelope)
}

fn build_prediction_from_response(
    provider: &CosmosProvider,
    state: &WorldState,
    action: &Action,
    config: &PredictionConfig,
    response: CosmosPredictionEnvelope,
    latency_ms: u64,
) -> Prediction {
    let physics_scores = response
        .physics_scores
        .map(|scores| scores.to_physics_scores())
        .unwrap_or_default();

    let output_state = parse_world_state(response.output_state)
        .or_else(|| parse_world_state(response.state))
        .unwrap_or_else(|| {
            let mut predicted_state = state.clone();
            predicted_state.time.step += config.steps as u64;
            predicted_state.time.seconds += config.steps as f64 / config.fps as f64;
            predicted_state
        });

    let video = if config.return_video {
        let media = merge_video_envelope(
            response
                .video
                .clone()
                .or(response.media.clone())
                .or(response.output.clone())
                .or(response.result.clone()),
            response
                .video_url
                .as_deref()
                .or(response.media_url.as_deref()),
            &response.frames,
            response.request_id.as_deref(),
            response.status.as_deref(),
        );
        media.map(|envelope| {
            materialize_video_clip(
                Some(envelope),
                response_marker(&[
                    response.request_id.as_deref(),
                    response.status.as_deref(),
                    response.model.as_deref(),
                    response.video_url.as_deref(),
                    response.media_url.as_deref(),
                ]),
                config.resolution,
                config.fps,
                config.steps as f64 / config.fps as f64,
            )
        })
    } else {
        None
    };

    Prediction {
        id: prediction_id_from_request_id(response.request_id.as_deref()),
        provider: provider.name().to_string(),
        model: response
            .model
            .unwrap_or_else(|| provider.model_id().to_string()),
        input_state: state.clone(),
        action: action.clone(),
        output_state,
        video,
        confidence: response
            .confidence
            .or(Some(physics_scores.overall))
            .unwrap_or(0.0),
        physics_scores,
        latency_ms: response
            .processing_time_ms
            .or(response.latency_ms)
            .unwrap_or(latency_ms),
        cost: provider.estimate_cost(&Operation::Predict {
            steps: config.steps,
            resolution: config.resolution,
        }),
        guardrail_results: Vec::new(),
        timestamp: chrono::Utc::now(),
    }
}

fn build_video_clip_from_response(
    response: CosmosVideoResponse,
    fallback_marker: String,
    resolution: (u32, u32),
    fps: f32,
    duration_seconds: f64,
) -> VideoClip {
    let envelope = merge_video_envelope(
        response
            .video
            .clone()
            .or(response.media.clone())
            .or(response.output.clone())
            .or(response.result.clone()),
        response
            .video_url
            .as_deref()
            .or(response.media_url.as_deref()),
        &response.frames,
        response.request_id.as_deref(),
        response.status.as_deref(),
    );
    materialize_video_clip(
        envelope,
        fallback_marker,
        resolution,
        response.fps.unwrap_or(fps),
        response.duration_seconds.unwrap_or(duration_seconds),
    )
}

fn build_reasoning_output_from_response(response: CosmosReasoningResponse) -> ReasoningOutput {
    let envelope = reasoning_envelope(&response);
    let mut evidence = envelope
        .as_ref()
        .map(|value| value.evidence.clone())
        .unwrap_or_default();
    if let Some(envelope) = envelope.as_ref() {
        for citation in &envelope.citations {
            if !evidence.contains(citation) {
                evidence.push(citation.clone());
            }
        }
    }

    ReasoningOutput {
        answer: envelope
            .as_ref()
            .and_then(|value| value.answer.clone())
            .unwrap_or_else(|| {
                response_marker(&[
                    response.request_id.as_deref(),
                    response.status.as_deref(),
                    response.summary.as_deref(),
                    response.rationale.as_deref(),
                ])
            }),
        confidence: envelope
            .as_ref()
            .and_then(|value| value.confidence)
            .or(response.confidence)
            .unwrap_or(0.0),
        evidence,
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
        let start = std::time::Instant::now();
        let prompt = Self::action_to_prompt(action);

        let request_body = CosmosPredictRequest {
            model: self.model_id().to_string(),
            prompt,
            num_frames: config.steps * (config.fps as u32),
            resolution: [config.resolution.0, config.resolution.1],
            fps: config.fps,
            include_depth: if self.config.include_depth {
                Some(true)
            } else {
                None
            },
        };

        let request = self
            .client
            .post(format!("{base_url}/v1/predict"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .json(&request_body);

        let response = self.send_with_retry(request).await?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(WorldForgeError::ProviderAuthError(
                "invalid Cosmos API key".to_string(),
            ));
        }

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(5000);
            return Err(WorldForgeError::ProviderRateLimited {
                provider: "cosmos".to_string(),
                retry_after_ms: retry_after,
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

        let latency_ms = start.elapsed().as_millis() as u64;

        let api_response: CosmosPredictionEnvelope = response
            .json()
            .await
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        Ok(build_prediction_from_response(
            self,
            state,
            action,
            config,
            api_response,
            latency_ms,
        ))
    }

    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip> {
        let base_url = self.base_url()?;

        let request_body = CosmosGenerateRequest {
            model: self.model_id().to_string(),
            prompt: prompt.text.clone(),
            negative_prompt: prompt.negative_prompt.clone(),
            duration_seconds: config.duration_seconds,
            resolution: [config.resolution.0, config.resolution.1],
            fps: config.fps,
            temperature: Some(config.temperature),
            seed: config.seed,
        };

        let request = self
            .client
            .post(format!("{base_url}/v1/generate"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .json(&request_body);

        let response = self.send_with_retry(request).await?;

        if !response.status().is_success() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "cosmos".to_string(),
                reason: format!("HTTP {}", response.status()),
            });
        }

        let api_response: CosmosVideoResponse = response
            .json()
            .await
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;
        Ok(build_video_clip_from_response(
            api_response,
            response_marker(&[
                Some(prompt.text.as_str()),
                prompt.negative_prompt.as_deref(),
                Some(self.model_id()),
            ]),
            config.resolution,
            config.fps,
            config.duration_seconds,
        ))
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

        let api_response: CosmosReasoningResponse = response
            .json()
            .await
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;
        Ok(build_reasoning_output_from_response(api_response))
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

        let api_response: CosmosVideoResponse = response
            .json()
            .await
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;
        Ok(build_video_clip_from_response(
            api_response,
            response_marker(&[Some("cosmos transfer"), Some(self.model_id())]),
            config.resolution,
            config.fps,
            0.0,
        ))
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

    #[test]
    fn test_config_validation_valid() {
        let config = CosmosConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_timeout() {
        let config = CosmosConfig {
            timeout_ms: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_too_many_frames() {
        let config = CosmosConfig {
            default_num_frames: 500,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_with_config_validates() {
        let bad_config = CosmosConfig {
            timeout_ms: 0,
            ..Default::default()
        };
        let result = CosmosProvider::with_config(
            CosmosModel::Predict2_5,
            "key",
            CosmosEndpoint::NimApi("https://api.nvidia.com".to_string()),
            bad_config,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_model_id() {
        let provider = CosmosProvider::new(
            CosmosModel::Predict2_5,
            "key",
            CosmosEndpoint::NimApi("https://api.nvidia.com".to_string()),
        );
        assert_eq!(provider.model_id(), "nvidia/cosmos-predict-2.5");

        let provider = CosmosProvider::new(
            CosmosModel::Reason2,
            "key",
            CosmosEndpoint::NimApi("https://api.nvidia.com".to_string()),
        );
        assert_eq!(provider.model_id(), "nvidia/cosmos-reason-2");
    }

    #[test]
    fn test_predict_request_serialization() {
        let req = CosmosPredictRequest {
            model: "nvidia/cosmos-predict-2.5".to_string(),
            prompt: "Move forward".to_string(),
            num_frames: 24,
            resolution: [1280, 720],
            fps: 24.0,
            include_depth: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("include_depth")); // skip_serializing_if
        let req2: CosmosPredictRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req2.num_frames, 24);
    }

    #[test]
    fn test_predict_request_with_depth() {
        let req = CosmosPredictRequest {
            model: "nvidia/cosmos-predict-2.5".to_string(),
            prompt: "Move forward".to_string(),
            num_frames: 24,
            resolution: [1280, 720],
            fps: 24.0,
            include_depth: Some(true),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("include_depth"));
    }

    #[test]
    fn test_cosmos_physics_scores_conversion() {
        let api_scores = CosmosPhysicsScores {
            overall: Some(0.9),
            object_permanence: Some(0.85),
            gravity_compliance: None,
            collision_accuracy: Some(0.7),
            spatial_consistency: None,
            temporal_consistency: Some(0.95),
        };
        let scores = api_scores.to_physics_scores();
        assert_eq!(scores.overall, 0.9);
        assert_eq!(scores.object_permanence, 0.85);
        assert_eq!(scores.gravity_compliance, 0.0); // None => 0.0
        assert_eq!(scores.collision_accuracy, 0.7);
        assert_eq!(scores.temporal_consistency, 0.95);
    }

    #[test]
    fn test_build_prediction_from_response_materializes_state_and_video() {
        let provider = CosmosProvider::new(
            CosmosModel::Predict2_5,
            "key",
            CosmosEndpoint::NimApi("https://api.nvidia.com".to_string()),
        );
        let input_state = WorldState::new("input", "cosmos");
        let output_state = WorldState::new("output", "cosmos");
        let action = Action::Move {
            target: Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            speed: 0.75,
        };
        let response = CosmosPredictionEnvelope {
            request_id: Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
            status: Some("completed".to_string()),
            model: Some("nvidia/cosmos-predict-2.5".to_string()),
            confidence: Some(0.88),
            physics_scores: Some(CosmosPhysicsScores {
                overall: Some(0.91),
                object_permanence: Some(0.89),
                gravity_compliance: Some(0.93),
                collision_accuracy: Some(0.84),
                spatial_consistency: Some(0.92),
                temporal_consistency: Some(0.87),
            }),
            processing_time_ms: Some(42),
            latency_ms: None,
            output_state: Some(serde_json::to_value(&output_state).unwrap()),
            state: None,
            video: Some(CosmosVideoEnvelope {
                url: Some("https://example.com/cosmos-prediction.mp4".to_string()),
                fps: Some(12.0),
                duration_seconds: Some(1.5),
                resolution: Some([320, 180]),
                frame_count: Some(2),
                frames: vec![
                    CosmosMediaFrame {
                        index: Some(0),
                        timestamp_seconds: Some(0.0),
                        url: Some("https://example.com/frame-0.png".to_string()),
                        data_url: None,
                        prompt: Some("frame zero".to_string()),
                        width: Some(320),
                        height: Some(180),
                    },
                    CosmosMediaFrame {
                        index: Some(1),
                        timestamp_seconds: Some(0.5),
                        url: None,
                        data_url: Some("data:image/png;base64,frame-1".to_string()),
                        prompt: Some("frame one".to_string()),
                        width: Some(320),
                        height: Some(180),
                    },
                ],
                prompt: Some("predict a cube rolling".to_string()),
                request_id: Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
                status: Some("completed".to_string()),
            }),
            media: None,
            output: None,
            result: None,
            video_url: None,
            media_url: None,
            frames: Vec::new(),
            reasoning: None,
        };
        let config = PredictionConfig {
            return_video: true,
            steps: 4,
            resolution: (640, 360),
            fps: 24.0,
            ..Default::default()
        };

        let prediction =
            build_prediction_from_response(&provider, &input_state, &action, &config, response, 17);

        assert_eq!(
            prediction.id.to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
        assert_eq!(prediction.output_state.metadata.name, "output");
        assert_eq!(prediction.confidence, 0.88);
        assert_eq!(prediction.physics_scores.overall, 0.91);
        assert_eq!(prediction.latency_ms, 42);
        let clip = prediction
            .video
            .expect("expected response media to be materialized");
        assert_eq!(clip.frames.len(), 2);
        assert_eq!(clip.resolution, (320, 180));
        assert!((clip.duration - 1.5).abs() < f64::EPSILON);
        match &clip.frames[0].data.data {
            TensorData::UInt8(bytes) => assert!(bytes.iter().any(|byte| *byte != 0)),
            other => panic!("unexpected tensor data: {other:?}"),
        }
    }

    #[test]
    fn test_build_video_clip_from_response_preserves_metadata() {
        let response = CosmosVideoResponse {
            request_id: Some("clip-request".to_string()),
            status: Some("finished".to_string()),
            model: Some("nvidia/cosmos-transfer-2.5".to_string()),
            prompt: Some("transfer motion".to_string()),
            negative_prompt: None,
            fps: Some(15.0),
            duration_seconds: Some(2.0),
            resolution: Some([640, 360]),
            frame_count: None,
            video: None,
            media: Some(CosmosVideoEnvelope {
                url: Some("https://example.com/transfer.mp4".to_string()),
                fps: Some(15.0),
                duration_seconds: Some(2.0),
                resolution: Some([640, 360]),
                frame_count: Some(2),
                frames: vec![CosmosMediaFrame {
                    index: Some(0),
                    timestamp_seconds: Some(0.0),
                    url: Some("https://example.com/transfer-0.png".to_string()),
                    data_url: None,
                    prompt: Some("first frame".to_string()),
                    width: Some(640),
                    height: Some(360),
                }],
                prompt: Some("transfer motion".to_string()),
                request_id: Some("clip-request".to_string()),
                status: Some("finished".to_string()),
            }),
            output: None,
            result: None,
            video_url: None,
            media_url: None,
            frames: vec![CosmosMediaFrame {
                index: Some(1),
                timestamp_seconds: Some(1.0),
                url: None,
                data_url: Some("data:image/png;base64,transfer-1".to_string()),
                prompt: Some("second frame".to_string()),
                width: Some(640),
                height: Some(360),
            }],
        };

        let clip = build_video_clip_from_response(
            response,
            "transfer fallback".to_string(),
            (1280, 720),
            24.0,
            4.0,
        );

        assert_eq!(clip.fps, 15.0);
        assert_eq!(clip.resolution, (640, 360));
        assert_eq!(clip.frames.len(), 2);
        assert!((clip.duration - 2.0).abs() < f64::EPSILON);
        match &clip.frames[1].data.data {
            TensorData::UInt8(bytes) => assert!(bytes.iter().any(|byte| *byte != 0)),
            other => panic!("unexpected tensor data: {other:?}"),
        }
    }

    #[test]
    fn test_build_reasoning_output_from_response_collects_evidence() {
        let response = CosmosReasoningResponse {
            request_id: Some("reason-request".to_string()),
            status: Some("ok".to_string()),
            answer: None,
            confidence: Some(0.77),
            evidence: vec!["surface is stable".to_string()],
            citations: vec!["frame-12".to_string()],
            rationale: Some("The mug remains supported".to_string()),
            summary: Some("stable mug".to_string()),
            reasoning: Some(CosmosReasoningEnvelope {
                answer: Some("The mug stays upright".to_string()),
                confidence: Some(0.82),
                evidence: vec!["support surface".to_string()],
                citations: vec!["frame-1".to_string()],
                rationale: None,
                summary: None,
            }),
            output: None,
            result: None,
        };

        let output = build_reasoning_output_from_response(response);
        assert_eq!(output.answer, "The mug stays upright");
        assert_eq!(output.confidence, 0.82);
        assert!(output.evidence.iter().any(|item| item == "support surface"));
        assert!(output.evidence.iter().any(|item| item == "frame-1"));
    }

    #[test]
    fn test_generate_request_serialization() {
        let req = CosmosGenerateRequest {
            model: "nvidia/cosmos-predict-2.5".to_string(),
            prompt: "A ball rolling down a hill".to_string(),
            negative_prompt: None,
            duration_seconds: 5.0,
            resolution: [1920, 1080],
            fps: 30.0,
            temperature: Some(0.8),
            seed: Some(42),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("negative_prompt")); // None => skipped
        let req2: CosmosGenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req2.seed, Some(42));
    }

    #[test]
    fn test_all_action_prompts() {
        // Ensure all action variants produce reasonable prompts
        let actions = vec![
            Action::Move {
                target: Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                speed: 1.0,
            },
            Action::Grasp {
                object: uuid::Uuid::new_v4(),
                grip_force: 5.0,
            },
            Action::Release {
                object: uuid::Uuid::new_v4(),
            },
            Action::Push {
                object: uuid::Uuid::new_v4(),
                direction: worldforge_core::types::Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                force: 3.0,
            },
            Action::Rotate {
                object: uuid::Uuid::new_v4(),
                axis: worldforge_core::types::Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                angle: 1.57,
            },
            Action::Place {
                object: uuid::Uuid::new_v4(),
                target: Position {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
            Action::SetWeather {
                weather: worldforge_core::action::Weather::Snow,
            },
            Action::SetLighting { time_of_day: 12.0 },
            Action::SpawnObject {
                template: "sphere".to_string(),
                pose: worldforge_core::types::Pose::default(),
            },
        ];
        for action in &actions {
            let prompt = CosmosProvider::action_to_prompt(action);
            assert!(!prompt.is_empty());
        }
    }
}
