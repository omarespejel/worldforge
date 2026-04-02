# RFC-0002: NVIDIA Cosmos Provider Integration

| Field     | Value                                      |
|-----------|--------------------------------------------|
| Title     | NVIDIA Cosmos Provider Integration         |
| Status    | Draft                                      |
| Authors   | WorldForge Core Team                       |
| Created   | 2026-04-02                                 |
| Updated   | 2026-04-02                                 |
| RFC       | 0002                                       |
| Depends   | RFC-0001 (Provider Integration Protocol)   |

---

## Abstract

This RFC specifies the complete integration of the NVIDIA Cosmos world
foundation model as a WorldForge provider. NVIDIA Cosmos is the
highest-priority provider for WorldForge because it is the most capable,
supporting prediction, reasoning, embedding, and generation operations
with physics-aware simulation.

This document covers the Cosmos API surface, authentication via NVIDIA
NGC API keys, request/response schemas, type mappings to WorldForge
abstractions, video frame handling, rate limits, pricing, error handling,
testing, and known limitations.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Cosmos API Overview](#cosmos-api-overview)
   - 2.1 [API Architecture](#21-api-architecture)
   - 2.2 [Endpoints](#22-endpoints)
   - 2.3 [Authentication](#23-authentication)
3. [Detailed Design](#detailed-design)
   - 3.1 [Provider Struct and Configuration](#31-provider-struct-and-configuration)
   - 3.2 [Capability Declaration](#32-capability-declaration)
   - 3.3 [Predict Operation](#33-predict-operation)
   - 3.4 [Reason Operation](#34-reason-operation)
   - 3.5 [Embed Operation](#35-embed-operation)
   - 3.6 [Generate Operation](#36-generate-operation)
   - 3.7 [Plan Operation](#37-plan-operation)
   - 3.8 [Health Check](#38-health-check)
   - 3.9 [Cost Estimate](#39-cost-estimate)
   - 3.10 [Type Mappings](#310-type-mappings)
   - 3.11 [Video Frame Pipeline](#311-video-frame-pipeline)
   - 3.12 [Error Handling](#312-error-handling)
4. [Rate Limits and Pricing](#rate-limits-and-pricing)
5. [Cosmos-Specific Features](#cosmos-specific-features)
6. [Implementation Plan](#implementation-plan)
7. [Testing Strategy](#testing-strategy)
8. [Example Usage](#example-usage)
9. [Known Limitations and Workarounds](#known-limitations-and-workarounds)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Open Questions](#open-questions)

---

## Motivation

NVIDIA Cosmos is a family of world foundation models designed for
physical AI development. It can:

- **Predict** future world states conditioned on actions (video prediction)
- **Reason** about physical scenes (object detection, physics understanding)
- **Embed** world states into dense vectors for similarity and retrieval
- **Generate** photorealistic worlds from text descriptions

This makes Cosmos the most complete single provider for WorldForge's
vision. Other providers (Runway, Google Genie) cover subsets of these
capabilities, but Cosmos covers 4 of the 6 core operations natively.

Additionally, NVIDIA provides enterprise-grade infrastructure through
NVIDIA Cloud Functions (NVCF) with:
- Low-latency inference via GPU-optimized endpoints
- SLA-backed availability
- Transparent per-request pricing

Integrating Cosmos first establishes the patterns that other providers
will follow.

---

## Cosmos API Overview

### 2.1 API Architecture

NVIDIA Cosmos is accessed through the NVIDIA Cloud Functions (NVCF)
platform. The API follows a RESTful design with JSON request/response
bodies and video data returned as downloadable assets.

```
┌──────────────┐          ┌──────────────────┐
│  WorldForge  │  HTTPS   │   NVIDIA NVCF    │
│  Client      │◄────────►│   API Gateway    │
└──────────────┘          └────────┬─────────┘
                                   │
                          ┌────────┴─────────┐
                          │  Cosmos Model    │
                          │  Inference GPU   │
                          │  Cluster         │
                          └──────────────────┘
```

Base URL: `https://api.nvcf.nvidia.com/v2/nvcf`

All requests are authenticated via Bearer token using the NVIDIA NGC
API key. Responses are JSON with video/binary data delivered via
signed download URLs.

### 2.2 Endpoints

| Endpoint                           | Method | WorldForge Op | Description                     |
|------------------------------------|--------|---------------|---------------------------------|
| `/functions/{predict_id}/invoke`   | POST   | predict       | World state prediction          |
| `/functions/{reason_id}/invoke`    | POST   | reason        | Scene reasoning & QA            |
| `/functions/{embed_id}/invoke`     | POST   | embed         | World state embedding           |
| `/functions/{generate_id}/invoke`  | POST   | generate      | World generation from prompt    |
| `/functions/{id}/status/{req_id}`  | GET    | (internal)    | Check async request status      |
| `/functions/{id}/results/{req_id}` | GET    | (internal)    | Retrieve completed results      |

NVIDIA Cosmos uses function IDs to identify model versions:
- Predict: `cosmos-1.0-predict-nvcf`
- Reason: `cosmos-1.0-reason-nvcf`
- Embed: `cosmos-1.0-embed-nvcf`
- Generate: `cosmos-1.0-generate-nvcf`

These function IDs may change with model versions and MUST be configurable.

### 2.3 Authentication

NVIDIA Cosmos uses NGC API key authentication:

1. User obtains an API key from https://ngc.nvidia.com
2. Key is passed as `Authorization: Bearer $NGC_API_KEY` on every request
3. Key format: `nvapi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`
4. Keys have per-organization quotas

```rust
pub struct CosmosAuth {
    api_key: SecretString,
}

impl CosmosAuth {
    pub fn new(api_key: SecretString) -> Self {
        Self { api_key }
    }

    pub fn from_env() -> Result<Self, WorldForgeError> {
        let key = std::env::var("WORLDFORGE_NVIDIA_COSMOS_API_KEY")
            .or_else(|_| std::env::var("NGC_API_KEY"))
            .map_err(|_| WorldForgeError::Auth {
                provider: "nvidia-cosmos".into(),
                message: "NGC API key not found. Set WORLDFORGE_NVIDIA_COSMOS_API_KEY or NGC_API_KEY".into(),
            })?;

        if !key.starts_with("nvapi-") {
            return Err(WorldForgeError::Auth {
                provider: "nvidia-cosmos".into(),
                message: "NGC API key must start with 'nvapi-'".into(),
            });
        }

        Ok(Self {
            api_key: SecretString::new(key),
        })
    }
}

#[async_trait]
impl ProviderAuth for CosmosAuth {
    async fn apply(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, WorldForgeError> {
        Ok(request.bearer_auth(self.api_key.expose_secret()))
    }

    async fn validate(&self) -> Result<(), WorldForgeError> {
        // Attempt a lightweight API call to validate the key
        // The health check endpoint is free
        Ok(())
    }
}
```

---

## Detailed Design

### 3.1 Provider Struct and Configuration

```rust
use crate::providers::prelude::*;

pub struct NvidiaCosmosProvider {
    /// HTTP client with auth, rate limiting, retry
    http: ProviderHttpClient,
    /// Provider-specific configuration
    config: CosmosConfig,
    /// Cost tracker (shared with WorldForge runtime)
    cost_tracker: Arc<CostTracker>,
    /// Video decoder for processing Cosmos video responses
    video_decoder: VideoDecoder,
    /// Video encoder for preparing input frames
    video_encoder: VideoEncoder,
    /// Cached health status
    cached_health: Arc<RwLock<Option<(Instant, HealthStatus)>>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CosmosConfig {
    /// NVIDIA NVCF base URL
    pub base_url: String,

    /// Function IDs for each Cosmos model
    pub predict_function_id: String,
    pub reason_function_id: String,
    pub embed_function_id: String,
    pub generate_function_id: String,

    /// Default model parameters
    pub default_resolution: (u32, u32),
    pub max_input_frames: usize,
    pub max_output_frames: usize,
    pub default_fps: f64,

    /// Timeout for async operations (Cosmos uses async invocation)
    pub async_poll_interval: Duration,
    pub async_max_wait: Duration,

    /// Rate limiting
    pub max_requests_per_second: f64,
    pub max_concurrent_requests: usize,

    /// Request timeout
    pub request_timeout: Duration,
}

impl Default for CosmosConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.nvcf.nvidia.com/v2/nvcf".to_string(),
            predict_function_id: "cosmos-1.0-predict-nvcf".to_string(),
            reason_function_id: "cosmos-1.0-reason-nvcf".to_string(),
            embed_function_id: "cosmos-1.0-embed-nvcf".to_string(),
            generate_function_id: "cosmos-1.0-generate-nvcf".to_string(),
            default_resolution: (1280, 720),
            max_input_frames: 60,
            max_output_frames: 120,
            default_fps: 24.0,
            async_poll_interval: Duration::from_secs(2),
            async_max_wait: Duration::from_secs(300),
            max_requests_per_second: 5.0,
            max_concurrent_requests: 5,
            request_timeout: Duration::from_secs(120),
        }
    }
}
```

#### Constructor

```rust
impl NvidiaCosmosProvider {
    pub async fn new(
        config: CosmosConfig,
        cost_tracker: Arc<CostTracker>,
    ) -> Result<Self, WorldForgeError> {
        let auth = CosmosAuth::from_env()?;

        let http = ProviderHttpClient::new(ProviderHttpConfig {
            base_url: config.base_url.clone(),
            auth: Box::new(auth),
            rate_limiter: RateLimiter::new(
                config.max_requests_per_second,
                config.max_concurrent_requests,
            ),
            retry_policy: RetryPolicy {
                max_retries: 3,
                initial_backoff: Duration::from_secs(1),
                max_backoff: Duration::from_secs(60),
                backoff_multiplier: 2.0,
                retryable_errors: vec![
                    RetryableError::RateLimit,
                    RetryableError::ServerError,
                    RetryableError::Timeout,
                ],
                jitter: JitterStrategy::Full,
            },
            provider_name: "nvidia-cosmos".to_string(),
            request_timeout: config.request_timeout,
            connect_timeout: Duration::from_secs(10),
            max_idle_connections: 10,
        })?;

        let video_decoder = VideoDecoder {
            max_frames: Some(config.max_output_frames),
            target_resolution: Some(config.default_resolution),
            target_format: PixelFormat::Rgb8,
            target_fps: Some(config.default_fps),
        };

        let video_encoder = VideoEncoder {
            codec: VideoCodec::H264,
            quality: 23,
            fps: config.default_fps,
        };

        Ok(Self {
            http,
            config,
            cost_tracker,
            video_decoder,
            video_encoder,
            cached_health: Arc::new(RwLock::new(None)),
        })
    }
}
```

### 3.2 Capability Declaration

```rust
impl WorldModelProvider for NvidiaCosmosProvider {
    fn name(&self) -> &str {
        "nvidia-cosmos"
    }

    fn describe(&self) -> String {
        format!(
            "NVIDIA Cosmos World Foundation Model — Physics-aware world simulation \
             supporting predict, reason, embed, and generate operations. \
             Resolution: {}x{}, Max frames: {}",
            self.config.default_resolution.0,
            self.config.default_resolution.1,
            self.config.max_output_frames,
        )
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
            .with_predict()
            .with_reason()
            .with_embed()
            .with_generate()
            .with_spatial_control()
            .with_action_conditioning()
    }
}
```

Cosmos supports 4 of 6 core operations plus spatial control and action
conditioning. It does NOT support:
- `plan` — No native planning API (could be composed from predict)
- `transfer` — No native style transfer API
- `real_time` — Inference latency is too high for real-time (2-30 seconds)

### 3.3 Predict Operation

The predict operation maps to Cosmos's world state prediction endpoint.
Given a sequence of input frames (current state) and an action description,
Cosmos generates predicted future frames.

#### Request Schema

```rust
#[derive(Debug, Serialize)]
struct CosmosPredictRequest {
    /// Base64-encoded input video or image frames
    input_video: String,

    /// Action conditioning — text description or structured action
    action: CosmosPredictAction,

    /// Number of frames to predict
    num_output_frames: u32,

    /// Output resolution
    resolution: CosmosResolution,

    /// Physics simulation parameters
    physics: Option<CosmosPhysicsParams>,

    /// Random seed for reproducibility
    seed: Option<u64>,

    /// Output format
    output_format: String,
}

#[derive(Debug, Serialize)]
struct CosmosPredictAction {
    /// Action type: "text", "trajectory", "control"
    action_type: String,

    /// Text description of the action (if action_type == "text")
    text: Option<String>,

    /// Camera trajectory points (if action_type == "trajectory")
    trajectory: Option<Vec<TrajectoryPoint>>,

    /// Control signals (if action_type == "control")
    controls: Option<CosmosControlSignals>,
}

#[derive(Debug, Serialize)]
struct TrajectoryPoint {
    /// Timestamp in seconds
    t: f64,
    /// Camera position [x, y, z]
    position: [f64; 3],
    /// Camera rotation [roll, pitch, yaw] in radians
    rotation: [f64; 3],
}

#[derive(Debug, Serialize)]
struct CosmosControlSignals {
    /// Robot joint positions
    joint_positions: Option<Vec<f64>>,
    /// End effector pose
    end_effector: Option<[f64; 6]>,
    /// Linear/angular velocity
    velocity: Option<[f64; 6]>,
}

#[derive(Debug, Serialize)]
struct CosmosResolution {
    width: u32,
    height: u32,
}

#[derive(Debug, Serialize)]
struct CosmosPhysicsParams {
    /// Enable physics simulation
    enabled: bool,
    /// Gravity vector [x, y, z] m/s^2
    gravity: Option<[f64; 3]>,
    /// Time step in seconds
    time_step: Option<f64>,
    /// Friction coefficient
    friction: Option<f64>,
    /// Restitution (bounciness)
    restitution: Option<f64>,
}
```

#### Response Schema

```rust
#[derive(Debug, Deserialize)]
struct CosmosPredictResponse {
    /// Request ID for async status tracking
    request_id: String,

    /// Status: "pending", "running", "completed", "failed"
    status: String,

    /// Result (present when status == "completed")
    result: Option<CosmosPredictResult>,

    /// Error details (present when status == "failed")
    error: Option<CosmosError>,
}

#[derive(Debug, Deserialize)]
struct CosmosPredictResult {
    /// URL to download the predicted video
    video_url: String,

    /// Video duration in seconds
    duration: f64,

    /// Number of frames generated
    num_frames: u32,

    /// Resolution of output
    resolution: CosmosResolution,

    /// Model confidence score
    confidence: f64,

    /// Physics simulation metrics (if physics was enabled)
    physics_metrics: Option<PhysicsMetrics>,

    /// Per-frame metadata
    frame_metadata: Vec<FrameMetadata>,
}

#[derive(Debug, Deserialize)]
struct PhysicsMetrics {
    /// Whether the simulation was physically plausible
    plausible: bool,
    /// Energy conservation error
    energy_error: f64,
    /// Collision detection accuracy
    collision_accuracy: f64,
}

#[derive(Debug, Deserialize)]
struct FrameMetadata {
    /// Frame index
    index: u32,
    /// Timestamp in seconds
    timestamp: f64,
    /// Detected objects in this frame
    objects: Vec<DetectedObject>,
    /// Camera pose (if available)
    camera_pose: Option<CameraPose>,
}
```

#### Implementation

```rust
async fn predict(
    &self,
    state: &WorldState,
    action: &Action,
    options: PredictOptions,
) -> Result<Prediction, WorldForgeError> {
    // 1. Encode input frames to video
    let input_video = self.encode_world_state_to_video(state).await?;
    let input_b64 = base64::engine::general_purpose::STANDARD.encode(&input_video);

    // 2. Map WorldForge Action to Cosmos action
    let cosmos_action = self.map_action(action)?;

    // 3. Build request
    let num_frames = options.horizon.unwrap_or(24) as u32;
    let resolution = options.resolution.unwrap_or(self.config.default_resolution);

    let request = CosmosPredictRequest {
        input_video: input_b64,
        action: cosmos_action,
        num_output_frames: num_frames,
        resolution: CosmosResolution {
            width: resolution.0,
            height: resolution.1,
        },
        physics: options.physics.map(|p| CosmosPhysicsParams {
            enabled: true,
            gravity: p.gravity,
            time_step: p.time_step,
            friction: p.friction,
            restitution: p.restitution,
        }),
        seed: options.seed,
        output_format: "mp4".to_string(),
    };

    // 4. Submit to Cosmos (async invocation)
    let invoke_url = format!(
        "/functions/{}/invoke",
        self.config.predict_function_id
    );
    let invoke_response: CosmosPredictResponse =
        self.http.post(&invoke_url, &request).await?;

    // 5. Poll for completion
    let result = self.wait_for_result::<CosmosPredictResult>(
        &self.config.predict_function_id,
        &invoke_response.request_id,
    ).await?;

    // 6. Download and decode video
    let video_bytes = self.http.download_video(&result.video_url).await?;
    let frames = self.video_decoder.decode(&video_bytes).await?;

    // 7. Record cost
    self.cost_tracker.record(
        "nvidia-cosmos",
        OperationType::Predict,
        self.estimate_predict_cost(num_frames, resolution),
        &invoke_response.request_id,
    )?;

    // 8. Build WorldForge Prediction
    Ok(Prediction {
        next_state: WorldState {
            frames: frames.clone(),
            metadata: result.frame_metadata.into_iter().map(|fm| {
                (format!("frame_{}", fm.index), serde_json::to_value(fm).unwrap())
            }).collect(),
            timestamp: chrono::Utc::now(),
            ..Default::default()
        },
        confidence: result.confidence,
        frames,
        metadata: {
            let mut meta = HashMap::new();
            if let Some(pm) = result.physics_metrics {
                meta.insert("physics_plausible".into(), json!(pm.plausible));
                meta.insert("energy_error".into(), json!(pm.energy_error));
            }
            meta.insert("request_id".into(), json!(invoke_response.request_id));
            meta
        },
        latency_ms: 0, // Will be set by the timing wrapper
    })
}
```

#### Async Polling

Cosmos uses asynchronous invocation for long-running inference. The
client must poll for results:

```rust
async fn wait_for_result<T: DeserializeOwned>(
    &self,
    function_id: &str,
    request_id: &str,
) -> Result<T, WorldForgeError> {
    let status_url = format!(
        "/functions/{}/status/{}",
        function_id, request_id
    );
    let result_url = format!(
        "/functions/{}/results/{}",
        function_id, request_id
    );

    let start = Instant::now();
    loop {
        if start.elapsed() > self.config.async_max_wait {
            return Err(WorldForgeError::Timeout {
                provider: "nvidia-cosmos".into(),
                timeout_ms: self.config.async_max_wait.as_millis() as u64,
            });
        }

        let status: AsyncStatusResponse = self.http.get(&status_url).await?;

        match status.status.as_str() {
            "completed" => {
                let result: T = self.http.get(&result_url).await?;
                return Ok(result);
            }
            "failed" => {
                return Err(WorldForgeError::ProviderError {
                    provider: "nvidia-cosmos".into(),
                    status: 500,
                    message: status.error.unwrap_or_else(|| "Unknown error".into()),
                });
            }
            "pending" | "running" => {
                tracing::debug!(
                    request_id = request_id,
                    status = status.status.as_str(),
                    elapsed_ms = start.elapsed().as_millis(),
                    "Waiting for Cosmos result"
                );
                tokio::time::sleep(self.config.async_poll_interval).await;
            }
            other => {
                return Err(WorldForgeError::InvalidResponse {
                    provider: "nvidia-cosmos".into(),
                    message: format!("Unknown status: {other}"),
                });
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct AsyncStatusResponse {
    status: String,
    progress: Option<f64>,
    error: Option<String>,
}
```

### 3.4 Reason Operation

The reason operation leverages Cosmos's scene understanding capabilities
to answer questions about a world state.

#### Request Schema

```rust
#[derive(Debug, Serialize)]
struct CosmosReasonRequest {
    /// Base64-encoded input video or image
    input_video: String,

    /// The reasoning query
    query: String,

    /// Reasoning mode: "descriptive", "causal", "counterfactual", "predictive"
    mode: String,

    /// Maximum response tokens
    max_tokens: u32,

    /// Whether to return bounding boxes for referenced objects
    return_spatial_references: bool,

    /// Whether to return a causal graph
    return_causal_chain: bool,

    /// Temperature for response generation
    temperature: f64,
}
```

#### Response Schema

```rust
#[derive(Debug, Deserialize)]
struct CosmosReasonResponse {
    request_id: String,
    status: String,
    result: Option<CosmosReasonResult>,
    error: Option<CosmosError>,
}

#[derive(Debug, Deserialize)]
struct CosmosReasonResult {
    /// The reasoning answer
    answer: String,

    /// Confidence in the answer
    confidence: f64,

    /// Spatial references (bounding boxes, segmentation masks)
    spatial_references: Vec<SpatialReference>,

    /// Causal chain (if requested)
    causal_chain: Option<Vec<CausalLink>>,

    /// Objects detected and referenced
    referenced_objects: Vec<ReferencedObject>,

    /// Token usage
    token_usage: TokenUsage,
}

#[derive(Debug, Deserialize)]
struct SpatialReference {
    /// Object label
    label: String,
    /// Bounding box [x1, y1, x2, y2] normalized to [0, 1]
    bbox: [f64; 4],
    /// Frame index this reference applies to
    frame_index: u32,
    /// Confidence
    confidence: f64,
}

#[derive(Debug, Deserialize)]
struct CausalLink {
    cause: String,
    effect: String,
    confidence: f64,
    temporal_relation: String,  // "before", "during", "after"
}

#[derive(Debug, Deserialize)]
struct TokenUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}
```

#### Implementation

```rust
async fn reason(
    &self,
    state: &WorldState,
    query: &str,
    options: ReasonOptions,
) -> Result<Reasoning, WorldForgeError> {
    let input_video = self.encode_world_state_to_video(state).await?;
    let input_b64 = base64::engine::general_purpose::STANDARD.encode(&input_video);

    let mode = match options.mode {
        ReasoningMode::Descriptive => "descriptive",
        ReasoningMode::Causal => "causal",
        ReasoningMode::Counterfactual => "counterfactual",
        ReasoningMode::Predictive => "predictive",
    };

    let request = CosmosReasonRequest {
        input_video: input_b64,
        query: query.to_string(),
        mode: mode.to_string(),
        max_tokens: options.max_tokens.unwrap_or(1024),
        return_spatial_references: true,
        return_causal_chain: matches!(options.mode, ReasoningMode::Causal),
        temperature: options.temperature.unwrap_or(0.7),
    };

    let invoke_url = format!(
        "/functions/{}/invoke",
        self.config.reason_function_id
    );
    let response: CosmosReasonResponse =
        self.http.post(&invoke_url, &request).await?;

    let result = self.wait_for_result::<CosmosReasonResult>(
        &self.config.reason_function_id,
        &response.request_id,
    ).await?;

    // Record cost based on token usage
    let cost = self.estimate_reason_cost(&result.token_usage);
    self.cost_tracker.record(
        "nvidia-cosmos",
        OperationType::Reason,
        cost,
        &response.request_id,
    )?;

    // Map to WorldForge types
    Ok(Reasoning {
        answer: result.answer,
        confidence: result.confidence,
        evidence: result.spatial_references.into_iter().map(|sr| {
            Evidence::Spatial {
                label: sr.label,
                bbox: sr.bbox,
                frame_index: sr.frame_index as usize,
                confidence: sr.confidence,
            }
        }).collect(),
        causal_chain: result.causal_chain.map(|chain| {
            chain.into_iter().map(|link| {
                WorldForgeCausalLink {
                    cause: link.cause,
                    effect: link.effect,
                    confidence: link.confidence,
                }
            }).collect()
        }),
    })
}
```

### 3.5 Embed Operation

The embed operation maps world states to dense vectors for similarity
search and downstream tasks.

#### Request Schema

```rust
#[derive(Debug, Serialize)]
struct CosmosEmbedRequest {
    /// Base64-encoded input video or image
    input_video: String,

    /// Embedding model variant
    model: String,

    /// Target embedding dimension (if the model supports variable dimensions)
    dimension: Option<u32>,

    /// Whether to L2-normalize the output
    normalize: bool,

    /// Pooling strategy: "mean", "max", "cls"
    pooling: String,
}
```

#### Response Schema

```rust
#[derive(Debug, Deserialize)]
struct CosmosEmbedResponse {
    request_id: String,
    status: String,
    result: Option<CosmosEmbedResult>,
    error: Option<CosmosError>,
}

#[derive(Debug, Deserialize)]
struct CosmosEmbedResult {
    /// The embedding vector
    embedding: Vec<f32>,

    /// Dimensionality
    dimension: u32,

    /// Whether the vector is normalized
    normalized: bool,

    /// The model used
    model: String,

    /// Per-frame embeddings (if requested)
    frame_embeddings: Option<Vec<Vec<f32>>>,
}
```

#### Implementation

```rust
async fn embed(
    &self,
    state: &WorldState,
    options: EmbedOptions,
) -> Result<Embedding, WorldForgeError> {
    let input_video = self.encode_world_state_to_video(state).await?;
    let input_b64 = base64::engine::general_purpose::STANDARD.encode(&input_video);

    let request = CosmosEmbedRequest {
        input_video: input_b64,
        model: options.model.unwrap_or_else(|| "cosmos-embed-v1".to_string()),
        dimension: options.dimension.map(|d| d as u32),
        normalize: options.normalize.unwrap_or(true),
        pooling: options.pooling.unwrap_or_else(|| "mean".to_string()),
    };

    let invoke_url = format!(
        "/functions/{}/invoke",
        self.config.embed_function_id
    );
    let response: CosmosEmbedResponse =
        self.http.post(&invoke_url, &request).await?;

    let result = self.wait_for_result::<CosmosEmbedResult>(
        &self.config.embed_function_id,
        &response.request_id,
    ).await?;

    // Validate embedding dimension
    if result.embedding.len() != result.dimension as usize {
        return Err(WorldForgeError::InvalidResponse {
            provider: "nvidia-cosmos".into(),
            message: format!(
                "Embedding dimension mismatch: declared {} but got {}",
                result.dimension,
                result.embedding.len()
            ),
        });
    }

    // Record cost
    self.cost_tracker.record(
        "nvidia-cosmos",
        OperationType::Embed,
        self.estimate_embed_cost(),
        &response.request_id,
    )?;

    Ok(Embedding {
        vector: result.embedding,
        dimension: result.dimension as usize,
        model: result.model,
        normalized: result.normalized,
    })
}
```

### 3.6 Generate Operation

The generate operation creates new world video from text prompts.

#### Request Schema

```rust
#[derive(Debug, Serialize)]
struct CosmosGenerateRequest {
    /// Text prompt describing the world to generate
    prompt: String,

    /// Negative prompt (what to avoid)
    negative_prompt: Option<String>,

    /// Number of frames to generate
    num_frames: u32,

    /// Output resolution
    resolution: CosmosResolution,

    /// Frames per second
    fps: f64,

    /// Random seed
    seed: Option<u64>,

    /// Guidance scale (higher = more prompt-adherent)
    guidance_scale: f64,

    /// Number of diffusion steps
    num_inference_steps: u32,

    /// Conditioning image (optional, for image-to-video)
    conditioning_image: Option<String>,

    /// Camera trajectory (optional, for controlled generation)
    camera_trajectory: Option<Vec<TrajectoryPoint>>,

    /// Output format: "mp4", "webm", "frames"
    output_format: String,
}
```

#### Response Schema

```rust
#[derive(Debug, Deserialize)]
struct CosmosGenerateResponse {
    request_id: String,
    status: String,
    result: Option<CosmosGenerateResult>,
    error: Option<CosmosError>,
}

#[derive(Debug, Deserialize)]
struct CosmosGenerateResult {
    /// URL to download the generated video
    video_url: String,

    /// Video duration in seconds
    duration: f64,

    /// Number of frames generated
    num_frames: u32,

    /// Resolution
    resolution: CosmosResolution,

    /// The seed used (for reproducibility)
    seed: u64,

    /// Generation quality metrics
    quality_metrics: Option<QualityMetrics>,
}

#[derive(Debug, Deserialize)]
struct QualityMetrics {
    /// FID (Frechet Inception Distance) if available
    fid: Option<f64>,
    /// CLIP similarity to prompt
    clip_score: Option<f64>,
    /// Temporal consistency score
    temporal_consistency: Option<f64>,
}
```

#### Implementation

```rust
async fn generate(
    &self,
    prompt: &str,
    options: GenerateOptions,
) -> Result<GeneratedWorld, WorldForgeError> {
    let num_frames = options.num_frames.unwrap_or(48) as u32;
    let resolution = options.resolution.unwrap_or(self.config.default_resolution);

    let conditioning_image = if let Some(ref img) = options.conditioning_image {
        Some(base64::engine::general_purpose::STANDARD.encode(img))
    } else {
        None
    };

    let request = CosmosGenerateRequest {
        prompt: prompt.to_string(),
        negative_prompt: options.negative_prompt.clone(),
        num_frames,
        resolution: CosmosResolution {
            width: resolution.0,
            height: resolution.1,
        },
        fps: options.fps.unwrap_or(self.config.default_fps),
        seed: options.seed,
        guidance_scale: options.guidance_scale.unwrap_or(7.5),
        num_inference_steps: options.num_steps.unwrap_or(50),
        conditioning_image,
        camera_trajectory: options.camera_trajectory.clone(),
        output_format: "mp4".to_string(),
    };

    let invoke_url = format!(
        "/functions/{}/invoke",
        self.config.generate_function_id
    );
    let response: CosmosGenerateResponse =
        self.http.post(&invoke_url, &request).await?;

    let result = self.wait_for_result::<CosmosGenerateResult>(
        &self.config.generate_function_id,
        &response.request_id,
    ).await?;

    // Download and decode video
    let video_bytes = self.http.download_video(&result.video_url).await?;
    let frames = self.video_decoder.decode(&video_bytes).await?;

    // Validate frame count
    if frames.len() != num_frames as usize {
        tracing::warn!(
            expected = num_frames,
            actual = frames.len(),
            "Cosmos returned different frame count than requested"
        );
    }

    // Record cost
    let cost = self.estimate_generate_cost(num_frames, resolution);
    self.cost_tracker.record(
        "nvidia-cosmos",
        OperationType::Generate,
        cost,
        &response.request_id,
    )?;

    Ok(GeneratedWorld {
        frames,
        scene: None, // Cosmos doesn't return 3D scene graphs
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("seed".into(), json!(result.seed));
            meta.insert("request_id".into(), json!(response.request_id));
            if let Some(qm) = result.quality_metrics {
                if let Some(clip) = qm.clip_score {
                    meta.insert("clip_score".into(), json!(clip));
                }
                if let Some(tc) = qm.temporal_consistency {
                    meta.insert("temporal_consistency".into(), json!(tc));
                }
            }
            meta
        },
        seed: result.seed,
    })
}
```

### 3.7 Plan Operation

Cosmos does NOT natively support planning. The plan operation returns
`UnsupportedOperation`:

```rust
async fn plan(
    &self,
    _current: &WorldState,
    _goal: &WorldState,
    _options: PlanOptions,
) -> Result<Plan, WorldForgeError> {
    Err(WorldForgeError::UnsupportedOperation {
        operation: "plan".to_string(),
        provider: "nvidia-cosmos".to_string(),
    })
}
```

Future work: A composed planner could use `predict` iteratively with
search to build plans, but this is outside the scope of the provider.

### 3.8 Health Check

```rust
async fn health_check(&self) -> Result<HealthStatus, WorldForgeError> {
    // Check cache
    if let Some((when, status)) = self.cached_health.read().await.as_ref() {
        if when.elapsed() < Duration::from_secs(30) {
            return Ok(status.clone());
        }
    }

    let start = Instant::now();

    // Hit the NVCF status endpoint (free, no billing)
    let url = format!(
        "/functions/{}/status",
        self.config.predict_function_id
    );

    let result = self.http.get::<NvcfFunctionStatus>(&url).await;

    let status = match result {
        Ok(nvcf_status) => HealthStatus {
            healthy: nvcf_status.status == "active",
            latency_ms: start.elapsed().as_millis() as u64,
            api_version: nvcf_status.version.unwrap_or_else(|| "unknown".into()),
            quota_remaining: nvcf_status.quota_remaining,
            message: Some(format!("Function status: {}", nvcf_status.status)),
        },
        Err(e) => HealthStatus {
            healthy: false,
            latency_ms: start.elapsed().as_millis() as u64,
            api_version: "unknown".to_string(),
            quota_remaining: None,
            message: Some(format!("Health check failed: {e}")),
        },
    };

    // Cache the result
    *self.cached_health.write().await = Some((Instant::now(), status.clone()));

    Ok(status)
}

#[derive(Debug, Deserialize)]
struct NvcfFunctionStatus {
    status: String,
    version: Option<String>,
    quota_remaining: Option<u64>,
    instances: Option<u32>,
}
```

### 3.9 Cost Estimate

```rust
async fn cost_estimate(
    &self,
    request: &CostEstimateRequest,
) -> Result<CostEstimate, WorldForgeError> {
    // All cost estimation is local computation — no API call
    let (cost, latency, breakdown) = match request.operation {
        OperationType::Predict => {
            let frames = request.options.get("num_frames")
                .and_then(|v| v.as_u64())
                .unwrap_or(24);
            let resolution = request.input_size.pixels();
            let cost = self.estimate_predict_cost(frames as u32, resolution);
            let latency = self.estimate_predict_latency(frames as u32, resolution);
            let breakdown = HashMap::from([
                ("inference".into(), cost * 0.7),
                ("video_encoding".into(), cost * 0.2),
                ("network".into(), cost * 0.1),
            ]);
            (cost, latency, breakdown)
        }
        OperationType::Reason => {
            let max_tokens = request.options.get("max_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(1024);
            let cost = 0.001 * (max_tokens as f64 / 1000.0); // $0.001 per 1K tokens
            let latency = 2000 + max_tokens * 5; // ~5ms per token
            let breakdown = HashMap::from([
                ("prompt_tokens".into(), cost * 0.3),
                ("completion_tokens".into(), cost * 0.7),
            ]);
            (cost, latency, breakdown)
        }
        OperationType::Embed => {
            let cost = 0.0005; // $0.0005 per embedding
            let latency = 500;
            let breakdown = HashMap::from([
                ("inference".into(), cost),
            ]);
            (cost, latency, breakdown)
        }
        OperationType::Generate => {
            let frames = request.options.get("num_frames")
                .and_then(|v| v.as_u64())
                .unwrap_or(48);
            let resolution = request.input_size.pixels();
            let cost = self.estimate_generate_cost(frames as u32, resolution);
            let latency = self.estimate_generate_latency(frames as u32, resolution);
            let breakdown = HashMap::from([
                ("diffusion_steps".into(), cost * 0.8),
                ("video_encoding".into(), cost * 0.15),
                ("network".into(), cost * 0.05),
            ]);
            (cost, latency, breakdown)
        }
        _ => {
            return Err(WorldForgeError::UnsupportedOperation {
                operation: format!("{:?}", request.operation),
                provider: "nvidia-cosmos".into(),
            });
        }
    };

    Ok(CostEstimate {
        estimated_cost_usd: cost,
        estimated_latency_ms: latency,
        estimated_tokens: None,
        confidence: 0.7, // Estimates are approximate
        breakdown,
    })
}
```

#### Cost Estimation Helpers

```rust
impl NvidiaCosmosProvider {
    fn estimate_predict_cost(&self, num_frames: u32, resolution: (u32, u32)) -> f64 {
        let base_cost = 0.005; // $0.005 per prediction
        let frame_multiplier = num_frames as f64 / 24.0;
        let resolution_multiplier = (resolution.0 * resolution.1) as f64
            / (1280.0 * 720.0);
        base_cost * frame_multiplier * resolution_multiplier
    }

    fn estimate_predict_latency(&self, num_frames: u32, resolution: (u32, u32)) -> u64 {
        let base_latency = 3000; // 3 seconds base
        let frame_latency = num_frames as u64 * 50; // 50ms per frame
        let resolution_factor = ((resolution.0 * resolution.1) as f64
            / (1280.0 * 720.0)) as u64;
        base_latency + frame_latency * resolution_factor
    }

    fn estimate_generate_cost(&self, num_frames: u32, resolution: (u32, u32)) -> f64 {
        let base_cost = 0.02; // $0.02 per generation
        let frame_multiplier = num_frames as f64 / 48.0;
        let resolution_multiplier = (resolution.0 * resolution.1) as f64
            / (1280.0 * 720.0);
        base_cost * frame_multiplier * resolution_multiplier
    }

    fn estimate_generate_latency(&self, num_frames: u32, resolution: (u32, u32)) -> u64 {
        let base_latency = 10000; // 10 seconds base
        let frame_latency = num_frames as u64 * 100; // 100ms per frame
        let resolution_factor = ((resolution.0 * resolution.1) as f64
            / (1280.0 * 720.0)) as u64;
        base_latency + frame_latency * resolution_factor
    }

    fn estimate_reason_cost(&self, usage: &TokenUsage) -> f64 {
        let prompt_cost = usage.prompt_tokens as f64 * 0.0000005; // $0.0005 per 1K
        let completion_cost = usage.completion_tokens as f64 * 0.0000015; // $0.0015 per 1K
        prompt_cost + completion_cost
    }

    fn estimate_embed_cost(&self) -> f64 {
        0.0005 // Flat rate per embedding
    }
}
```

### 3.10 Type Mappings

#### WorldState to Cosmos Input

```rust
impl NvidiaCosmosProvider {
    /// Encode a WorldState into a video for Cosmos input.
    async fn encode_world_state_to_video(
        &self,
        state: &WorldState,
    ) -> Result<Vec<u8>, WorldForgeError> {
        if state.frames.is_empty() {
            return Err(WorldForgeError::Validation {
                message: "WorldState has no frames to encode".into(),
            });
        }

        // Limit frames to max input
        let frames: Vec<&Frame> = state.frames.iter()
            .take(self.config.max_input_frames)
            .collect();

        // Encode frames to MP4 video
        self.video_encoder.encode_frames(&frames).await
    }

    /// Map a WorldForge Action to a Cosmos action spec.
    fn map_action(&self, action: &Action) -> Result<CosmosPredictAction, WorldForgeError> {
        match action {
            Action::Text(text) => Ok(CosmosPredictAction {
                action_type: "text".to_string(),
                text: Some(text.clone()),
                trajectory: None,
                controls: None,
            }),
            Action::Trajectory(points) => Ok(CosmosPredictAction {
                action_type: "trajectory".to_string(),
                text: None,
                trajectory: Some(points.iter().map(|p| TrajectoryPoint {
                    t: p.timestamp,
                    position: p.position,
                    rotation: p.rotation,
                }).collect()),
                controls: None,
            }),
            Action::Control(ctrl) => Ok(CosmosPredictAction {
                action_type: "control".to_string(),
                text: None,
                trajectory: None,
                controls: Some(CosmosControlSignals {
                    joint_positions: ctrl.joint_positions.clone(),
                    end_effector: ctrl.end_effector,
                    velocity: ctrl.velocity,
                }),
            }),
            Action::Noop => Ok(CosmosPredictAction {
                action_type: "text".to_string(),
                text: Some("no action".to_string()),
                trajectory: None,
                controls: None,
            }),
        }
    }
}
```

#### Type Mapping Summary Table

| WorldForge Type       | Cosmos Concept          | Mapping Notes                          |
|-----------------------|-------------------------|----------------------------------------|
| `WorldState`          | Input video (MP4)       | Frames encoded to video                |
| `WorldState.frames`   | Video frames            | RGB8 frames at configured FPS          |
| `WorldState.metadata` | Frame metadata          | JSON key-value pairs                   |
| `Action::Text`        | Text action             | Direct string mapping                  |
| `Action::Trajectory`  | Camera trajectory       | [t, xyz, rpy] points                   |
| `Action::Control`     | Control signals         | Joint/end-effector/velocity            |
| `Prediction`          | Predict result          | Video decoded back to frames           |
| `Embedding`           | Embed result            | f32 vector                             |
| `Reasoning`           | Reason result           | Text + spatial references              |
| `GeneratedWorld`      | Generate result         | Video decoded to frames                |
| `ProviderCapabilities`| N/A                     | Hardcoded based on Cosmos features     |

### 3.11 Video Frame Pipeline

#### Input Pipeline (WorldState -> Cosmos)

```
WorldState.frames (Vec<Frame>)
        │
        ▼
┌───────────────────┐
│ Validate frames   │  Check resolution consistency, format
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Resize if needed  │  Scale to Cosmos-supported resolution
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Encode to MP4     │  H.264, CRF 23, configured FPS
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Base64 encode     │  For JSON request body
└───────────────────┘
```

#### Output Pipeline (Cosmos -> WorldState)

```
Cosmos video_url (signed URL)
        │
        ▼
┌───────────────────┐
│ Stream download   │  Chunked HTTP download with progress
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Verify integrity  │  Check Content-Length, file magic bytes
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Decode with FFmpeg│  Extract raw RGB frames
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Validate frames   │  Count, resolution, format
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Build Vec<Frame>  │  Assign indices, timestamps
└───────────────────┘
```

#### Supported Resolutions

Cosmos supports the following resolutions:
- 1280 x 720 (720p) — Default, best quality/speed tradeoff
- 1920 x 1080 (1080p) — Higher quality, ~2x slower
- 640 x 480 (480p) — Fast preview mode
- 512 x 512 — Square format, for certain use cases

The provider MUST validate requested resolution and fall back to the
nearest supported resolution with a warning.

```rust
fn nearest_supported_resolution(w: u32, h: u32) -> (u32, u32) {
    const SUPPORTED: [(u32, u32); 4] = [
        (640, 480),
        (512, 512),
        (1280, 720),
        (1920, 1080),
    ];

    SUPPORTED.iter()
        .min_by_key(|(sw, sh)| {
            let dw = (*sw as i64 - w as i64).abs();
            let dh = (*sh as i64 - h as i64).abs();
            dw + dh
        })
        .copied()
        .unwrap_or((1280, 720))
}
```

### 3.12 Error Handling

#### Cosmos-Specific Error Codes

```rust
#[derive(Debug, Deserialize)]
struct CosmosError {
    code: String,
    message: String,
    details: Option<Value>,
}

fn map_cosmos_error(error: &CosmosError) -> WorldForgeError {
    match error.code.as_str() {
        "INVALID_INPUT" | "INVALID_VIDEO_FORMAT" => WorldForgeError::Validation {
            message: format!("[nvidia-cosmos] {}: {}", error.code, error.message),
        },
        "AUTHENTICATION_FAILED" | "INVALID_API_KEY" => WorldForgeError::Auth {
            provider: "nvidia-cosmos".into(),
            message: error.message.clone(),
        },
        "RATE_LIMIT_EXCEEDED" => WorldForgeError::RateLimited {
            provider: "nvidia-cosmos".into(),
            retry_after_ms: error.details
                .as_ref()
                .and_then(|d| d.get("retry_after_ms"))
                .and_then(|v| v.as_u64())
                .unwrap_or(5000),
        },
        "QUOTA_EXCEEDED" => WorldForgeError::BudgetExceeded {
            estimated: 0.0,
            budget: 0.0,
        },
        "MODEL_UNAVAILABLE" | "FUNCTION_NOT_FOUND" => WorldForgeError::ProviderError {
            provider: "nvidia-cosmos".into(),
            status: 503,
            message: error.message.clone(),
        },
        "INFERENCE_TIMEOUT" => WorldForgeError::Timeout {
            provider: "nvidia-cosmos".into(),
            timeout_ms: 0,
        },
        "INTERNAL_ERROR" | "GPU_ERROR" => WorldForgeError::ProviderError {
            provider: "nvidia-cosmos".into(),
            status: 500,
            message: error.message.clone(),
        },
        "VIDEO_TOO_LARGE" | "RESOLUTION_UNSUPPORTED" => WorldForgeError::Validation {
            message: format!("[nvidia-cosmos] {}", error.message),
        },
        _ => WorldForgeError::ProviderError {
            provider: "nvidia-cosmos".into(),
            status: 500,
            message: format!("{}: {}", error.code, error.message),
        },
    }
}
```

#### Error Recovery Strategies

| Error Code             | Strategy                                           |
|------------------------|----------------------------------------------------|
| RATE_LIMIT_EXCEEDED    | Exponential backoff, respect Retry-After            |
| QUOTA_EXCEEDED         | Fail immediately, alert user                        |
| MODEL_UNAVAILABLE      | Retry 3x with 30s backoff, then fail                |
| INFERENCE_TIMEOUT      | Retry once with 2x timeout, then fail               |
| GPU_ERROR              | Retry 2x with 10s backoff                           |
| INVALID_VIDEO_FORMAT   | Re-encode video with conservative settings, retry   |
| VIDEO_TOO_LARGE        | Downsample resolution, retry                        |

---

## Rate Limits and Pricing

### Rate Limits

| Tier        | Requests/sec | Concurrent | Daily Limit | Monthly Limit |
|-------------|-------------|------------|-------------|---------------|
| Free        | 1           | 1          | 100         | 1,000         |
| Developer   | 5           | 5          | 1,000       | 25,000        |
| Enterprise  | 50          | 25         | Unlimited   | Unlimited     |

### Pricing (Estimated)

| Operation    | Unit              | Free Tier | Developer   | Enterprise   |
|-------------|-------------------|-----------|-------------|--------------|
| Predict     | per 24-frame req  | $0.00     | $0.005      | $0.003       |
| Reason      | per 1K tokens     | $0.00     | $0.001      | $0.0006      |
| Embed       | per request       | $0.00     | $0.0005     | $0.0003      |
| Generate    | per 48-frame req  | $0.00     | $0.02       | $0.012       |

Notes:
- Free tier has usage caps, not billing.
- Costs scale linearly with frame count beyond base units.
- 1080p costs approximately 2x the 720p rate.
- Enterprise pricing requires direct NVIDIA agreement.

### Budget Guard Configuration

```toml
[providers.nvidia-cosmos.budget]
max_cost_per_request_usd = 0.10
max_daily_cost_usd = 10.00
max_monthly_cost_usd = 200.00
alert_threshold_percent = 80
```

---

## Cosmos-Specific Features

### 5.1 Physics Simulation

Cosmos can simulate physical interactions with configurable parameters:

```rust
let options = PredictOptions {
    physics: Some(PhysicsOptions {
        gravity: Some([0.0, -9.81, 0.0]),
        time_step: Some(0.016), // 60Hz physics
        friction: Some(0.5),
        restitution: Some(0.3),
    }),
    ..Default::default()
};

let prediction = cosmos.predict(&state, &action, options).await?;

// Check physics plausibility in response metadata
if let Some(plausible) = prediction.metadata.get("physics_plausible") {
    println!("Physics plausible: {}", plausible);
}
```

### 5.2 Camera Control

Cosmos supports precise camera trajectory specification:

```rust
let action = Action::Trajectory(vec![
    TrajectoryPoint3D { timestamp: 0.0, position: [0.0, 1.5, 0.0], rotation: [0.0, 0.0, 0.0] },
    TrajectoryPoint3D { timestamp: 1.0, position: [2.0, 1.5, 0.0], rotation: [0.0, 0.3, 0.0] },
    TrajectoryPoint3D { timestamp: 2.0, position: [4.0, 2.0, 1.0], rotation: [0.0, 0.5, 0.0] },
]);
```

### 5.3 Action Conditioning

Robot/agent action conditioning for embodied AI:

```rust
let action = Action::Control(ControlAction {
    joint_positions: Some(vec![0.0, 0.5, -0.3, 0.0, 0.7, 0.0]),
    end_effector: Some([0.5, 0.3, 0.2, 0.0, 0.0, 0.0]),
    velocity: None,
});
```

### 5.4 Counterfactual Reasoning

```rust
let reasoning = cosmos.reason(
    &state,
    "What would happen if the ball were twice as heavy?",
    ReasonOptions {
        mode: ReasoningMode::Counterfactual,
        ..Default::default()
    },
).await?;

println!("Answer: {}", reasoning.answer);
if let Some(chain) = &reasoning.causal_chain {
    for link in chain {
        println!("  {} → {} (confidence: {})", link.cause, link.effect, link.confidence);
    }
}
```

---

## Implementation Plan

| Phase | Task                                          | Duration | Dependencies |
|-------|-----------------------------------------------|----------|--------------|
| 1     | Implement `CosmosConfig` and `CosmosAuth`     | 2 days   | RFC-0001     |
| 2     | Implement `NvidiaCosmosProvider` struct        | 1 day    | Phase 1      |
| 3     | Implement `health_check` and `cost_estimate`  | 2 days   | Phase 2      |
| 4     | Implement video encode/decode pipeline        | 3 days   | Phase 2      |
| 5     | Implement `embed` operation                   | 2 days   | Phase 4      |
| 6     | Implement `reason` operation                  | 2 days   | Phase 4      |
| 7     | Implement `predict` operation                 | 3 days   | Phase 4      |
| 8     | Implement `generate` operation                | 3 days   | Phase 4      |
| 9     | Implement type mappings and validation        | 2 days   | Phases 5-8   |
| 10    | Implement error mapping                       | 1 day    | Phases 5-8   |
| 11    | Record replay fixtures                        | 2 days   | Phases 5-8   |
| 12    | Write integration tests                       | 3 days   | Phase 11     |
| 13    | Write documentation and examples              | 2 days   | Phase 12     |
| 14    | Performance benchmarking                      | 2 days   | Phase 12     |
| 15    | Code review and merge                         | 2 days   | Phase 14     |

Total estimated duration: **4 weeks**

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosmos_config_default() {
        let config = CosmosConfig::default();
        assert_eq!(config.default_resolution, (1280, 720));
        assert_eq!(config.max_output_frames, 120);
    }

    #[test]
    fn test_map_text_action() {
        let provider = create_test_provider();
        let action = Action::Text("move forward".into());
        let cosmos_action = provider.map_action(&action).unwrap();
        assert_eq!(cosmos_action.action_type, "text");
        assert_eq!(cosmos_action.text.unwrap(), "move forward");
    }

    #[test]
    fn test_map_trajectory_action() {
        let provider = create_test_provider();
        let action = Action::Trajectory(vec![
            TrajectoryPoint3D {
                timestamp: 0.0,
                position: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0],
            },
        ]);
        let cosmos_action = provider.map_action(&action).unwrap();
        assert_eq!(cosmos_action.action_type, "trajectory");
        assert_eq!(cosmos_action.trajectory.unwrap().len(), 1);
    }

    #[test]
    fn test_nearest_resolution() {
        assert_eq!(nearest_supported_resolution(1300, 730), (1280, 720));
        assert_eq!(nearest_supported_resolution(1900, 1000), (1920, 1080));
        assert_eq!(nearest_supported_resolution(500, 500), (512, 512));
    }

    #[test]
    fn test_cosmos_error_mapping() {
        let error = CosmosError {
            code: "RATE_LIMIT_EXCEEDED".into(),
            message: "Too many requests".into(),
            details: Some(json!({"retry_after_ms": 5000})),
        };
        let wf_error = map_cosmos_error(&error);
        assert!(matches!(wf_error, WorldForgeError::RateLimited { .. }));
    }

    #[test]
    fn test_cost_estimate_predict() {
        let provider = create_test_provider();
        let cost = provider.estimate_predict_cost(24, (1280, 720));
        assert!(cost > 0.0);
        assert!(cost < 0.10); // Sanity check

        // Higher resolution should cost more
        let cost_hd = provider.estimate_predict_cost(24, (1920, 1080));
        assert!(cost_hd > cost);
    }

    #[test]
    fn test_capabilities() {
        let provider = create_test_provider();
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(caps.reason);
        assert!(caps.embed);
        assert!(caps.generate);
        assert!(!caps.plan);
        assert!(!caps.transfer);
        assert!(!caps.real_time);
        assert!(caps.spatial_control);
        assert!(caps.action_conditioning);
    }
}
```

### Integration Tests (Replay)

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::testing::ReplayServer;

    #[tokio::test]
    async fn test_predict_with_replay() {
        let server = ReplayServer::from_fixture(
            "tests/fixtures/nvidia-cosmos/predict_basic.json"
        ).await;

        let config = CosmosConfig {
            base_url: server.url(),
            ..Default::default()
        };
        let provider = NvidiaCosmosProvider::new_with_auth(
            config,
            Arc::new(CostTracker::new(None)),
            Box::new(MockAuth),
        ).await.unwrap();

        let state = test_world_state_with_frames(4);
        let action = Action::Text("move forward".into());
        let prediction = provider.predict(&state, &action, PredictOptions::default()).await.unwrap();

        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(!prediction.frames.is_empty());
    }

    #[tokio::test]
    async fn test_reason_with_replay() {
        let server = ReplayServer::from_fixture(
            "tests/fixtures/nvidia-cosmos/reason_basic.json"
        ).await;

        let config = CosmosConfig {
            base_url: server.url(),
            ..Default::default()
        };
        let provider = NvidiaCosmosProvider::new_with_auth(
            config,
            Arc::new(CostTracker::new(None)),
            Box::new(MockAuth),
        ).await.unwrap();

        let state = test_world_state_with_frames(1);
        let reasoning = provider.reason(
            &state,
            "What objects are visible?",
            ReasonOptions::default(),
        ).await.unwrap();

        assert!(!reasoning.answer.is_empty());
    }

    #[tokio::test]
    async fn test_embed_deterministic() {
        let server = ReplayServer::from_fixture(
            "tests/fixtures/nvidia-cosmos/embed_deterministic.json"
        ).await;

        let config = CosmosConfig {
            base_url: server.url(),
            ..Default::default()
        };
        let provider = NvidiaCosmosProvider::new_with_auth(
            config,
            Arc::new(CostTracker::new(None)),
            Box::new(MockAuth),
        ).await.unwrap();

        let state = test_world_state_with_frames(1);
        let e1 = provider.embed(&state, EmbedOptions::default()).await.unwrap();
        let e2 = provider.embed(&state, EmbedOptions::default()).await.unwrap();
        assert_eq!(e1.vector, e2.vector);
    }

    #[tokio::test]
    async fn test_rate_limit_handling() {
        let server = ReplayServer::from_fixture(
            "tests/fixtures/nvidia-cosmos/rate_limit_then_success.json"
        ).await;

        let config = CosmosConfig {
            base_url: server.url(),
            ..Default::default()
        };
        let provider = NvidiaCosmosProvider::new_with_auth(
            config,
            Arc::new(CostTracker::new(None)),
            Box::new(MockAuth),
        ).await.unwrap();

        // Should succeed after automatic retry
        let health = provider.health_check().await.unwrap();
        assert!(health.healthy);
    }

    #[tokio::test]
    async fn test_unsupported_plan() {
        let provider = create_test_cosmos_provider().await;
        let result = provider.plan(
            &WorldState::default(),
            &WorldState::default(),
            PlanOptions::default(),
        ).await;
        assert!(matches!(result, Err(WorldForgeError::UnsupportedOperation { .. })));
    }

    #[tokio::test]
    async fn test_unsupported_transfer() {
        let provider = create_test_cosmos_provider().await;
        let result = provider.transfer(
            &WorldState::default(),
            &WorldState::default(),
            TransferOptions::default(),
        ).await;
        assert!(matches!(result, Err(WorldForgeError::UnsupportedOperation { .. })));
    }
}
```

### Live Integration Tests (Nightly CI)

```rust
#[cfg(test)]
mod live_tests {
    use super::*;

    /// These tests require WORLDFORGE_NVIDIA_COSMOS_API_KEY to be set.
    /// They are only run in nightly CI.

    fn skip_if_no_api_key() {
        if std::env::var("WORLDFORGE_NVIDIA_COSMOS_API_KEY").is_err() {
            println!("Skipping live test: no API key");
            return;
        }
    }

    #[tokio::test]
    #[ignore] // Only run with --include-ignored
    async fn live_health_check() {
        skip_if_no_api_key();
        let provider = NvidiaCosmosProvider::new(
            CosmosConfig::default(),
            Arc::new(CostTracker::new(Some(1.0))), // $1 budget for tests
        ).await.unwrap();

        let health = provider.health_check().await.unwrap();
        assert!(health.healthy);
        println!("Cosmos health: {:?}", health);
    }

    #[tokio::test]
    #[ignore]
    async fn live_embed() {
        skip_if_no_api_key();
        let provider = create_live_provider().await;
        let state = create_simple_test_state();
        let embedding = provider.embed(&state, EmbedOptions::default()).await.unwrap();
        assert!(embedding.dimension > 0);
        assert_eq!(embedding.vector.len(), embedding.dimension);
        println!("Embedding dim: {}", embedding.dimension);
    }

    #[tokio::test]
    #[ignore]
    async fn live_predict() {
        skip_if_no_api_key();
        let provider = create_live_provider().await;
        let state = create_simple_test_state();
        let action = Action::Text("camera pans right slowly".into());
        let prediction = provider.predict(
            &state,
            &action,
            PredictOptions {
                horizon: Some(8),
                ..Default::default()
            },
        ).await.unwrap();
        assert!(!prediction.frames.is_empty());
        println!("Predicted {} frames, confidence: {}", prediction.frames.len(), prediction.confidence);
    }
}
```

---

## Example Usage

### Rust Example

```rust
use worldforge::prelude::*;
use worldforge::providers::nvidia_cosmos::{NvidiaCosmosProvider, CosmosConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the Cosmos provider
    let config = CosmosConfig::default();
    let cost_tracker = Arc::new(CostTracker::new(Some(10.0))); // $10 budget
    let cosmos = NvidiaCosmosProvider::new(config, cost_tracker.clone()).await?;

    // Check health
    let health = cosmos.health_check().await?;
    println!("Provider: {} — Healthy: {}", cosmos.name(), health.healthy);

    // Generate a world
    let world = cosmos.generate(
        "A robot arm picking up a red cube from a table in a well-lit lab",
        GenerateOptions {
            num_frames: Some(48),
            resolution: Some((1280, 720)),
            seed: Some(42),
            guidance_scale: Some(7.5),
            ..Default::default()
        },
    ).await?;
    println!("Generated {} frames", world.frames.len());

    // Reason about the generated world
    let state = WorldState::from_frames(world.frames);
    let reasoning = cosmos.reason(
        &state,
        "What is the robot doing and what objects are on the table?",
        ReasonOptions::default(),
    ).await?;
    println!("Reasoning: {}", reasoning.answer);

    // Predict what happens next
    let action = Action::Control(ControlAction {
        joint_positions: Some(vec![0.0, 0.8, -0.5, 0.0, 0.6, 0.0]),
        end_effector: None,
        velocity: None,
    });
    let prediction = cosmos.predict(&state, &action, PredictOptions::default()).await?;
    println!(
        "Prediction: {} frames, confidence: {:.2}",
        prediction.frames.len(),
        prediction.confidence
    );

    // Embed for similarity search
    let embedding = cosmos.embed(&state, EmbedOptions::default()).await?;
    println!("Embedding: {} dimensions", embedding.dimension);

    // Check total cost
    println!("Total cost: ${:.4}", cost_tracker.total_cost_usd());

    Ok(())
}
```

### Python Example (via PyO3 bindings)

```python
import worldforge

async def main():
    # Initialize provider
    cosmos = await worldforge.NvidiaCosmosProvider.create(
        api_key_env="WORLDFORGE_NVIDIA_COSMOS_API_KEY",
        resolution=(1280, 720),
    )

    # Check health
    health = await cosmos.health_check()
    print(f"Healthy: {health.healthy}, Latency: {health.latency_ms}ms")

    # Generate a world
    world = await cosmos.generate(
        prompt="A sunny park with children playing on a playground",
        num_frames=48,
        seed=42,
    )
    print(f"Generated {len(world.frames)} frames")

    # Save frames as images
    for i, frame in enumerate(world.frames):
        frame.save(f"output/frame_{i:04d}.png")

    # Reason about the scene
    state = worldforge.WorldState.from_frames(world.frames)
    reasoning = await cosmos.reason(
        state=state,
        query="How many children are visible and what are they doing?",
    )
    print(f"Answer: {reasoning.answer}")
    print(f"Confidence: {reasoning.confidence:.2f}")

    # Embed for similarity
    embedding = await cosmos.embed(state)
    print(f"Embedding shape: ({embedding.dimension},)")

    # Predict with camera movement
    action = worldforge.Action.trajectory([
        worldforge.TrajectoryPoint(t=0.0, pos=[0, 1.5, 0], rot=[0, 0, 0]),
        worldforge.TrajectoryPoint(t=1.0, pos=[3, 1.5, 0], rot=[0, 0.3, 0]),
    ])
    prediction = await cosmos.predict(state, action, horizon=24)
    print(f"Predicted {len(prediction.frames)} frames")

    # Save prediction as video
    prediction.save_video("output/prediction.mp4", fps=24)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## Known Limitations and Workarounds

### Limitation 1: No Real-Time Inference

**Problem:** Cosmos inference takes 2-30 seconds per request, making it
unsuitable for real-time applications (games, robotics control loops).

**Workaround:**
- Use Cosmos for planning/look-ahead in a separate thread.
- Cache predictions for repeated states.
- Use a lighter model for real-time and Cosmos for validation.

### Limitation 2: No Native Planning

**Problem:** Cosmos cannot generate multi-step plans directly.

**Workaround:**
- Compose planning from iterated predict calls:
```rust
async fn composed_plan(cosmos: &NvidiaCosmosProvider, current: &WorldState, goal: &WorldState) -> Vec<Action> {
    let mut state = current.clone();
    let mut plan = Vec::new();
    for _ in 0..max_steps {
        let embedding_current = cosmos.embed(&state, Default::default()).await?;
        let embedding_goal = cosmos.embed(goal, Default::default()).await?;
        if cosine_similarity(&embedding_current.vector, &embedding_goal.vector) > 0.95 {
            break;
        }
        // Use reasoning to suggest next action
        let suggestion = cosmos.reason(&state, "What action should be taken to reach the goal?", Default::default()).await?;
        let action = parse_action(&suggestion.answer);
        let prediction = cosmos.predict(&state, &action, Default::default()).await?;
        state = prediction.next_state;
        plan.push(action);
    }
    Ok(plan)
}
```

### Limitation 3: No Style Transfer

**Problem:** Cosmos does not support direct style/property transfer between states.

**Workaround:** Use the generate endpoint with a prompt that describes
the source style applied to the target content.

### Limitation 4: Video Upload Size Limits

**Problem:** Large video inputs (>50MB) may be rejected or cause timeouts.

**Workaround:**
- Downsample to 720p before upload.
- Limit input to 60 frames (2.5 seconds at 24fps).
- Use keyframe extraction for long sequences.

### Limitation 5: Async Polling Overhead

**Problem:** Cosmos uses async invocation, requiring polling which adds latency.

**Workaround:**
- Start with aggressive polling (1s) then back off.
- Use webhook callbacks when available (future NVCF feature).
- Pipeline: submit multiple requests, poll in parallel.

### Limitation 6: Embedding Model Compatibility

**Problem:** Embeddings from different Cosmos model versions are not comparable.

**Workaround:**
- Always store the model version alongside embeddings.
- Re-embed when upgrading model versions.
- Use the `model` field in `EmbedOptions` to pin a version.

---

## Performance Benchmarks

### Target Benchmarks

| Operation    | Resolution | Frames | Target Latency | Target Throughput |
|-------------|-----------|--------|----------------|-------------------|
| predict     | 720p      | 24     | < 5s           | 12 req/min        |
| predict     | 1080p     | 24     | < 10s          | 6 req/min         |
| predict     | 720p      | 48     | < 8s           | 8 req/min         |
| reason      | 720p      | 1      | < 3s           | 20 req/min        |
| reason      | 720p      | 24     | < 5s           | 12 req/min        |
| embed       | 720p      | 1      | < 1s           | 60 req/min        |
| embed       | 720p      | 24     | < 2s           | 30 req/min        |
| generate    | 720p      | 48     | < 15s          | 4 req/min         |
| generate    | 1080p     | 48     | < 30s          | 2 req/min         |
| health_check| N/A       | N/A    | < 500ms        | N/A               |
| cost_estimate| N/A      | N/A    | < 1ms          | N/A               |

### Benchmark Methodology

1. Run each operation 100 times (or 20 for expensive operations).
2. Record P50, P95, P99 latencies.
3. Record actual cost per operation.
4. Compare against target benchmarks.
5. Run benchmarks on a standard network connection (100 Mbps+).

### Benchmark Suite

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    #[ignore]
    async fn benchmark_predict_720p_24frames() {
        let provider = create_live_provider().await;
        let state = create_benchmark_state(1280, 720, 4);
        let action = Action::Text("camera pans right".into());
        let options = PredictOptions {
            horizon: Some(24),
            resolution: Some((1280, 720)),
            ..Default::default()
        };

        let mut latencies = Vec::new();
        for i in 0..20 {
            let start = Instant::now();
            let result = provider.predict(&state, &action, options.clone()).await;
            let elapsed = start.elapsed();
            latencies.push(elapsed.as_millis());

            match result {
                Ok(p) => println!(
                    "  Run {}: {}ms, {} frames, confidence: {:.2}",
                    i, elapsed.as_millis(), p.frames.len(), p.confidence
                ),
                Err(e) => println!("  Run {}: ERROR: {}", i, e),
            }
        }

        latencies.sort();
        println!("Predict 720p/24f benchmark:");
        println!("  P50: {}ms", latencies[latencies.len() / 2]);
        println!("  P95: {}ms", latencies[latencies.len() * 95 / 100]);
        println!("  P99: {}ms", latencies[latencies.len() * 99 / 100]);
        assert!(latencies[latencies.len() / 2] < 5000, "P50 exceeds 5s target");
    }

    #[tokio::test]
    #[ignore]
    async fn benchmark_embed_720p() {
        let provider = create_live_provider().await;
        let state = create_benchmark_state(1280, 720, 1);

        let mut latencies = Vec::new();
        for _ in 0..100 {
            let start = Instant::now();
            let _ = provider.embed(&state, EmbedOptions::default()).await;
            latencies.push(start.elapsed().as_millis());
        }

        latencies.sort();
        println!("Embed 720p benchmark:");
        println!("  P50: {}ms", latencies[50]);
        println!("  P95: {}ms", latencies[95]);
        println!("  P99: {}ms", latencies[99]);
        assert!(latencies[50] < 1000, "P50 exceeds 1s target");
    }
}
```

---

## Open Questions

1. **Cosmos model versioning:** How should we handle Cosmos model updates?
   Pin to a specific version or auto-upgrade?

2. **Webhook support:** NVCF may add webhook callbacks for async results.
   Should we design for this now or add later?

3. **Multi-GPU inference:** For enterprise users with private Cosmos
   deployments, how should we handle custom endpoints?

4. **Batch API:** Should we implement batch predict/embed for efficiency?
   NVCF may add batch endpoints.

5. **Streaming generation:** Cosmos may support frame-by-frame streaming
   in future versions. How should this integrate with the current
   frame-based API?

6. **Fine-tuned models:** How should we handle Cosmos models fine-tuned
   on user data? Custom function IDs? Config-based?

7. **Video caching:** Should we cache downloaded prediction videos for
   replay/debugging? How much disk space to allocate?

8. **Multimodal prompts:** Cosmos may support image+text prompts for
   generation. How should this map to the current prompt string API?

9. **Regional endpoints:** NVIDIA may offer region-specific NVCF endpoints
   for lower latency. Should we auto-select or let users configure?

10. **Fallback providers:** If Cosmos is down, should the provider
    automatically fall back to another provider, or should this be
    handled by the WorldForge router?

---

## Appendix A: Environment Variable Reference

| Variable                              | Required | Description                      |
|---------------------------------------|----------|----------------------------------|
| `WORLDFORGE_NVIDIA_COSMOS_API_KEY`    | Yes*     | NGC API key (preferred)          |
| `NGC_API_KEY`                         | Yes*     | NGC API key (fallback)           |
| `WORLDFORGE_COSMOS_BASE_URL`          | No       | Override NVCF base URL           |
| `WORLDFORGE_COSMOS_PREDICT_FUNC_ID`   | No       | Override predict function ID     |
| `WORLDFORGE_COSMOS_REASON_FUNC_ID`    | No       | Override reason function ID      |
| `WORLDFORGE_COSMOS_EMBED_FUNC_ID`     | No       | Override embed function ID       |
| `WORLDFORGE_COSMOS_GENERATE_FUNC_ID`  | No       | Override generate function ID    |

*At least one of the API key variables must be set.

## Appendix B: Configuration File Example

```toml
[providers.nvidia-cosmos]
enabled = true
api_key_env = "WORLDFORGE_NVIDIA_COSMOS_API_KEY"
base_url = "https://api.nvcf.nvidia.com/v2/nvcf"

# Model function IDs
predict_function_id = "cosmos-1.0-predict-nvcf"
reason_function_id = "cosmos-1.0-reason-nvcf"
embed_function_id = "cosmos-1.0-embed-nvcf"
generate_function_id = "cosmos-1.0-generate-nvcf"

# Defaults
default_resolution = [1280, 720]
max_input_frames = 60
max_output_frames = 120
default_fps = 24.0

# Async polling
async_poll_interval_secs = 2
async_max_wait_secs = 300

# Rate limiting
max_requests_per_second = 5.0
max_concurrent_requests = 5

# Timeouts
request_timeout_secs = 120

# Budget
[providers.nvidia-cosmos.budget]
max_cost_per_request_usd = 0.10
max_daily_cost_usd = 10.00
max_monthly_cost_usd = 200.00
alert_threshold_percent = 80
```

---

*End of RFC-0002*
