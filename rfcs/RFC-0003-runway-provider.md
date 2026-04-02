# RFC-0003: Runway Gen-4 / GWM Provider Integration

| Field   | Value                                      |
|---------|--------------------------------------------|
| Title   | Runway Gen-4 / GWM Provider Integration    |
| Status  | Draft                                      |
| Author  | WorldForge Contributors                    |
| Created | 2026-04-02                                 |
| RFC     | 0003                                       |

---

## Abstract

This RFC proposes the integration of Runway's Gen-4 Turbo and GWM-1 (General
World Model) APIs as a first-class provider within the WorldForge framework.
Runway represents one of the most capable commercial video generation and world
simulation platforms available today, offering text-to-video, image-to-video,
and video-to-video generation through Gen-4 Turbo, alongside physics-aware
world simulation through GWM-1. This document describes the architecture of the
`RunwayProvider`, the mapping of Runway API concepts to WorldForge trait
methods, the async job lifecycle, camera control and motion brush features,
error handling, rate limiting, and testing strategy.

---

## Motivation

### Why Runway?

Runway is a leading generative AI company whose Gen-4 Turbo model produces
state-of-the-art video generation results. More importantly, their GWM-1
(General World Model) represents a significant step toward true world
simulation—predicting physically plausible future states of visual environments.
This aligns directly with WorldForge's core mission of orchestrating world
foundation models.

### Key Capabilities

1. **Gen-4 Turbo**: High-quality video generation with fine-grained camera
   control, motion brushes, and style transfer. Supports text-to-video,
   image-to-video, and video-to-video pipelines.

2. **GWM-1**: A world simulation model that goes beyond generation—it reasons
   about physics, object permanence, and spatial relationships, making it
   uniquely suited for WorldForge's `predict()` and `reason()` trait methods.

3. **Mature API**: Runway provides a well-documented REST API with async job
   semantics, making it straightforward to integrate into Rust async workflows.

### WorldForge Alignment

The Runway provider maps cleanly to the `WorldModelProvider` trait:

- `predict()` → GWM-1 world simulation (next-state prediction)
- `generate()` → Gen-4 Turbo video generation
- `transfer()` → Gen-4 Turbo video-to-video style transfer
- `reason()` → GWM-1 physics/spatial reasoning
- `embed()` → Not directly supported (see Open Questions)
- `plan()` → Multi-step generation via chained predictions
- `health_check()` → API availability ping
- `cost_estimate()` → Token/credit-based cost calculation

---

## Detailed Design

### 1. Runway API Overview

#### 1.1 API Base URL and Versioning

Runway's API is accessed through a versioned REST endpoint:

```
Base URL: https://api.dev.runwayml.com/v1
```

The API follows REST conventions with JSON request/response bodies. All
endpoints require authentication and return standard HTTP status codes.

#### 1.2 Authentication

Runway uses bearer token authentication. API tokens are generated from the
Runway dashboard and scoped to an organization.

```rust
pub struct RunwayConfig {
    /// Bearer token for API authentication
    pub api_token: String,
    /// Optional organization ID for multi-org accounts
    pub organization_id: Option<String>,
    /// Base URL override (for testing/staging)
    pub base_url: Option<String>,
    /// Maximum concurrent jobs
    pub max_concurrent_jobs: usize,
    /// Polling interval for async jobs (milliseconds)
    pub poll_interval_ms: u64,
    /// Maximum polling duration before timeout (seconds)
    pub max_poll_duration_secs: u64,
}
```

Authentication is performed by including the token in the `Authorization`
header:

```
Authorization: Bearer <RUNWAY_API_TOKEN>
```

The provider reads the token from:
1. `RunwayConfig.api_token` (explicit)
2. `RUNWAY_API_TOKEN` environment variable (fallback)
3. `~/.worldforge/runway_token` file (last resort)

#### 1.3 Async Job Model

All generation requests in Runway follow an asynchronous job pattern:

```
Submit Job → Receive Job ID → Poll Status → Download Result
```

The lifecycle is:

1. **Submit**: POST to the generation endpoint with parameters. Returns a
   job ID immediately (HTTP 200).

2. **Poll**: GET the job status endpoint with the job ID. Status transitions:
   - `PENDING` → Job is queued
   - `RUNNING` → Generation in progress
   - `SUCCEEDED` → Output ready for download
   - `FAILED` → Generation failed (includes error details)
   - `CANCELLED` → Job was cancelled by user

3. **Download**: On `SUCCEEDED`, the response includes a presigned URL for
   the generated output. URLs expire after a configurable duration.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunwayJob {
    pub id: String,
    pub status: RunwayJobStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub output_url: Option<String>,
    pub error: Option<RunwayJobError>,
    pub progress: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RunwayJobStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}
```

#### 1.4 Polling Strategy

The provider implements an adaptive polling strategy:

```rust
pub struct AdaptivePoller {
    /// Initial interval between polls
    initial_interval: Duration,
    /// Maximum interval (cap for backoff)
    max_interval: Duration,
    /// Backoff multiplier
    backoff_factor: f64,
    /// Total timeout
    timeout: Duration,
}

impl AdaptivePoller {
    pub fn new(config: &RunwayConfig) -> Self {
        Self {
            initial_interval: Duration::from_millis(config.poll_interval_ms),
            max_interval: Duration::from_secs(30),
            backoff_factor: 1.5,
            timeout: Duration::from_secs(config.max_poll_duration_secs),
        }
    }

    pub async fn poll_until_complete(
        &self,
        client: &reqwest::Client,
        job_id: &str,
    ) -> Result<RunwayJob, RunwayError> {
        let start = Instant::now();
        let mut interval = self.initial_interval;

        loop {
            if start.elapsed() > self.timeout {
                return Err(RunwayError::Timeout {
                    job_id: job_id.to_string(),
                    elapsed: start.elapsed(),
                });
            }

            tokio::time::sleep(interval).await;
            let job = self.fetch_job_status(client, job_id).await?;

            match job.status {
                RunwayJobStatus::Succeeded => return Ok(job),
                RunwayJobStatus::Failed => {
                    return Err(RunwayError::JobFailed {
                        job_id: job_id.to_string(),
                        error: job.error,
                    });
                }
                RunwayJobStatus::Cancelled => {
                    return Err(RunwayError::JobCancelled {
                        job_id: job_id.to_string(),
                    });
                }
                _ => {
                    interval = Duration::from_secs_f64(
                        (interval.as_secs_f64() * self.backoff_factor)
                            .min(self.max_interval.as_secs_f64()),
                    );
                }
            }
        }
    }
}
```

### 2. Gen-4 Turbo Modes

#### 2.1 Text-to-Video

Generate video from a text prompt. The simplest generation mode.

```rust
#[derive(Debug, Clone, Serialize)]
pub struct TextToVideoRequest {
    /// Text prompt describing the desired video
    pub prompt: String,
    /// Duration in seconds (5 or 10)
    pub duration: u8,
    /// Aspect ratio: "16:9", "9:16", "1:1"
    pub aspect_ratio: String,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
    /// Camera control settings
    pub camera: Option<CameraControl>,
    /// Style reference image URL
    pub style_reference: Option<String>,
}
```

API endpoint: `POST /v1/gen4-turbo/text-to-video`

#### 2.2 Image-to-Video

Animate a reference image into a video sequence. Preserves visual identity
from the source image while adding motion and dynamics.

```rust
#[derive(Debug, Clone, Serialize)]
pub struct ImageToVideoRequest {
    /// URL or base64-encoded reference image
    pub image: ImageInput,
    /// Text prompt for motion/action guidance
    pub prompt: String,
    /// Duration in seconds (5 or 10)
    pub duration: u8,
    /// Motion brush regions for selective animation
    pub motion_brush: Option<MotionBrush>,
    /// Camera control settings
    pub camera: Option<CameraControl>,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ImageInput {
    Url(String),
    Base64 { data: String, mime_type: String },
}
```

API endpoint: `POST /v1/gen4-turbo/image-to-video`

#### 2.3 Video-to-Video

Transform an existing video, applying style transfer or modifications while
preserving the original motion structure.

```rust
#[derive(Debug, Clone, Serialize)]
pub struct VideoToVideoRequest {
    /// URL or reference to source video
    pub video: String,
    /// Text prompt for transformation guidance
    pub prompt: String,
    /// Strength of transformation (0.0 = no change, 1.0 = full restyle)
    pub strength: f32,
    /// Preserve motion from source
    pub preserve_motion: bool,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}
```

API endpoint: `POST /v1/gen4-turbo/video-to-video`

### 3. Camera Control

Gen-4 Turbo provides granular camera control through parameterized camera
movements. These map to cinematic concepts:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct CameraControl {
    /// Camera movement type
    pub movement: CameraMovement,
    /// Intensity of the movement (0.0 to 1.0)
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraMovement {
    /// Static camera, no movement
    Static,
    /// Pan left/right
    Pan { direction: PanDirection },
    /// Tilt up/down
    Tilt { direction: TiltDirection },
    /// Dolly zoom in/out
    Dolly { direction: DollyDirection },
    /// Orbit around subject
    Orbit { direction: OrbitDirection },
    /// Crane shot up/down
    Crane { direction: CraneDirection },
    /// Free-form camera path (advanced)
    Custom {
        /// Keyframe positions as [x, y, z] at normalized timestamps
        keyframes: Vec<CameraKeyframe>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct CameraKeyframe {
    /// Normalized time (0.0 to 1.0)
    pub time: f32,
    /// Camera position [x, y, z]
    pub position: [f32; 3],
    /// Camera look-at target [x, y, z]
    pub look_at: [f32; 3],
}
```

### 4. Motion Brush

The motion brush allows selective animation of regions within an image:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct MotionBrush {
    /// Regions to animate
    pub regions: Vec<MotionRegion>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MotionRegion {
    /// Mask image (binary, same dimensions as input)
    pub mask: ImageInput,
    /// Direction of motion in this region
    pub direction: MotionDirection,
    /// Speed of motion (0.0 to 1.0)
    pub speed: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MotionDirection {
    /// Horizontal component (-1.0 to 1.0)
    pub x: f32,
    /// Vertical component (-1.0 to 1.0)
    pub y: f32,
}
```

### 5. Mapping to WorldForge Types

#### 5.1 Frame Mapping

Runway outputs video files (MP4). WorldForge needs individual frames and
video clips:

```rust
impl RunwayProvider {
    /// Download the generated video and extract frames
    async fn download_and_extract(
        &self,
        output_url: &str,
    ) -> Result<VideoClip, RunwayError> {
        // 1. Download MP4 from presigned URL
        let video_bytes = self.client
            .get(output_url)
            .send()
            .await?
            .bytes()
            .await?;

        // 2. Save to temporary file
        let temp_path = self.temp_dir.join(format!("{}.mp4", Uuid::new_v4()));
        tokio::fs::write(&temp_path, &video_bytes).await?;

        // 3. Extract metadata
        let metadata = extract_video_metadata(&temp_path)?;

        // 4. Create VideoClip
        Ok(VideoClip {
            id: Uuid::new_v4().to_string(),
            path: temp_path,
            duration: metadata.duration,
            fps: metadata.fps,
            resolution: Resolution {
                width: metadata.width,
                height: metadata.height,
            },
            frames: None, // Lazily extracted on demand
            format: VideoFormat::Mp4,
            created_at: chrono::Utc::now(),
        })
    }

    /// Extract individual frames from a video clip
    async fn extract_frames(
        &self,
        clip: &VideoClip,
    ) -> Result<Vec<Frame>, RunwayError> {
        // Use ffmpeg or a Rust video decoder to extract frames
        let frames = video_decoder::extract_frames(
            &clip.path,
            clip.fps,
        )?;

        Ok(frames.into_iter().enumerate().map(|(i, raw)| {
            Frame {
                id: format!("{}-frame-{}", clip.id, i),
                index: i as u64,
                timestamp: i as f64 / clip.fps as f64,
                data: raw,
                resolution: clip.resolution.clone(),
            }
        }).collect())
    }
}
```

#### 5.2 WorldModelProvider Implementation

```rust
#[async_trait]
impl WorldModelProvider for RunwayProvider {
    async fn predict(
        &self,
        input: &WorldState,
        params: &PredictionParams,
    ) -> Result<WorldState, ProviderError> {
        // Use GWM-1 for world state prediction
        let request = self.build_gwm_request(input, params)?;
        let job = self.submit_gwm_job(request).await?;
        let result = self.poller.poll_until_complete(&self.client, &job.id).await?;
        self.parse_gwm_output(result, input).await
    }

    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        params: &GenerationParams,
    ) -> Result<GenerationOutput, ProviderError> {
        // Use Gen-4 Turbo for video generation
        let request = match prompt {
            GenerationPrompt::Text(text) => {
                self.build_text_to_video(text, params)?
            }
            GenerationPrompt::Image(img) => {
                self.build_image_to_video(img, params)?
            }
            GenerationPrompt::Video(vid) => {
                self.build_video_to_video(vid, params)?
            }
        };

        let job = self.submit_gen4_job(request).await?;
        let result = self.poller.poll_until_complete(&self.client, &job.id).await?;
        let clip = self.download_and_extract(&result.output_url.unwrap()).await?;

        Ok(GenerationOutput::Video(clip))
    }

    async fn transfer(
        &self,
        source: &MediaInput,
        target_style: &StyleParams,
    ) -> Result<GenerationOutput, ProviderError> {
        // Video-to-video with style transfer
        let request = VideoToVideoRequest {
            video: source.to_url()?,
            prompt: target_style.description.clone(),
            strength: target_style.strength.unwrap_or(0.7),
            preserve_motion: true,
            seed: target_style.seed,
        };

        let job = self.submit_gen4_job(
            Gen4Request::VideoToVideo(request)
        ).await?;
        let result = self.poller.poll_until_complete(&self.client, &job.id).await?;
        let clip = self.download_and_extract(&result.output_url.unwrap()).await?;

        Ok(GenerationOutput::Video(clip))
    }

    async fn health_check(&self) -> Result<HealthStatus, ProviderError> {
        let response = self.client
            .get(format!("{}/health", self.base_url))
            .bearer_auth(&self.config.api_token)
            .send()
            .await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(HealthStatus::Healthy),
            reqwest::StatusCode::UNAUTHORIZED => {
                Ok(HealthStatus::Degraded("Invalid API token".into()))
            }
            status => Ok(HealthStatus::Unhealthy(
                format!("API returned status {}", status)
            )),
        }
    }

    async fn cost_estimate(
        &self,
        params: &GenerationParams,
    ) -> Result<CostEstimate, ProviderError> {
        let credits = match params.duration_secs {
            d if d <= 5 => 50,
            d if d <= 10 => 100,
            _ => 200,
        };

        Ok(CostEstimate {
            credits,
            estimated_usd: credits as f64 * 0.01, // $0.01 per credit
            currency: "USD".to_string(),
            breakdown: vec![
                CostItem {
                    description: format!("Gen-4 Turbo {}s generation", params.duration_secs),
                    amount: credits as f64 * 0.01,
                },
            ],
        })
    }
}
```

### 6. Rate Limits

Runway enforces rate limits at the organization level:

| Tier       | Requests/min | Concurrent Jobs | Monthly Credits |
|------------|-------------|-----------------|-----------------|
| Free       | 5           | 1               | 125             |
| Standard   | 30          | 5               | 625             |
| Pro        | 60          | 10              | 2250            |
| Enterprise | Custom      | Custom          | Custom          |

The provider implements a token bucket rate limiter:

```rust
pub struct RateLimiter {
    tokens: Arc<AtomicU32>,
    max_tokens: u32,
    refill_rate: Duration,
    concurrent_jobs: Arc<Semaphore>,
}

impl RateLimiter {
    pub async fn acquire(&self) -> Result<RateLimitGuard, RunwayError> {
        let permit = self.concurrent_jobs
            .acquire()
            .await
            .map_err(|_| RunwayError::RateLimitExhausted)?;

        // Wait for token availability
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current > 0 {
                if self.tokens.compare_exchange(
                    current, current - 1,
                    Ordering::SeqCst, Ordering::Relaxed
                ).is_ok() {
                    return Ok(RateLimitGuard { permit });
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}
```

### 7. Pricing Estimate

Based on Runway's current pricing model:

| Operation          | Credits | Estimated USD |
|--------------------|---------|---------------|
| Text-to-Video 5s   | 50      | $0.50         |
| Text-to-Video 10s  | 100     | $1.00         |
| Image-to-Video 5s  | 50      | $0.50         |
| Image-to-Video 10s | 100     | $1.00         |
| Video-to-Video 5s  | 75      | $0.75         |
| Video-to-Video 10s | 150     | $1.50         |
| GWM-1 Simulation   | 25      | $0.25         |

### 8. Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum RunwayError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Authentication failed: invalid or expired API token")]
    AuthenticationFailed,

    #[error("Rate limit exceeded, retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },

    #[error("Job {job_id} failed: {error:?}")]
    JobFailed {
        job_id: String,
        error: Option<RunwayJobError>,
    },

    #[error("Job {job_id} was cancelled")]
    JobCancelled { job_id: String },

    #[error("Job {job_id} timed out after {elapsed:?}")]
    Timeout {
        job_id: String,
        elapsed: Duration,
    },

    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },

    #[error("Content policy violation: {message}")]
    ContentFiltered { message: String },

    #[error("Insufficient credits: have {available}, need {required}")]
    InsufficientCredits { available: u64, required: u64 },

    #[error("API returned unexpected response: {0}")]
    UnexpectedResponse(String),

    #[error("Video processing error: {0}")]
    VideoProcessing(String),
}

impl From<RunwayError> for ProviderError {
    fn from(e: RunwayError) -> Self {
        match e {
            RunwayError::AuthenticationFailed => ProviderError::Authentication(e.to_string()),
            RunwayError::RateLimited { .. } => ProviderError::RateLimit(e.to_string()),
            RunwayError::ContentFiltered { .. } => ProviderError::ContentPolicy(e.to_string()),
            RunwayError::Timeout { .. } => ProviderError::Timeout(e.to_string()),
            _ => ProviderError::Provider(e.to_string()),
        }
    }
}
```

### 9. Provider Struct

```rust
pub struct RunwayProvider {
    config: RunwayConfig,
    client: reqwest::Client,
    base_url: String,
    poller: AdaptivePoller,
    rate_limiter: RateLimiter,
    temp_dir: PathBuf,
    active_jobs: Arc<DashMap<String, RunwayJob>>,
}

impl RunwayProvider {
    pub fn new(config: RunwayConfig) -> Result<Self, RunwayError> {
        let client = reqwest::Client::builder()
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", config.api_token).parse().unwrap(),
                );
                headers.insert(
                    reqwest::header::CONTENT_TYPE,
                    "application/json".parse().unwrap(),
                );
                headers
            })
            .timeout(Duration::from_secs(30))
            .build()?;

        let base_url = config.base_url.clone()
            .unwrap_or_else(|| "https://api.dev.runwayml.com/v1".to_string());

        Ok(Self {
            poller: AdaptivePoller::new(&config),
            rate_limiter: RateLimiter::new(&config),
            config,
            client,
            base_url,
            temp_dir: std::env::temp_dir().join("worldforge-runway"),
            active_jobs: Arc::new(DashMap::new()),
        })
    }
}
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

1. Create `crates/worldforge-providers/src/runway/` module
2. Implement `RunwayConfig` with environment variable support
3. Implement HTTP client with authentication
4. Implement async job submission and polling
5. Implement adaptive polling with exponential backoff
6. Add basic error types and conversions

### Phase 2: Gen-4 Turbo Integration (Week 3-4)

1. Implement text-to-video generation
2. Implement image-to-video generation
3. Implement video-to-video transformation
4. Add camera control parameter mapping
5. Add motion brush support
6. Implement video download and frame extraction

### Phase 3: GWM-1 Integration (Week 5-6)

1. Implement GWM-1 world simulation requests
2. Map GWM-1 outputs to WorldForge `WorldState`
3. Implement `predict()` using GWM-1
4. Implement `reason()` for physics-aware queries
5. Implement `plan()` for multi-step predictions

### Phase 4: Production Hardening (Week 7-8)

1. Implement rate limiting with token bucket
2. Add retry logic with exponential backoff
3. Implement job cancellation and cleanup
4. Add metrics and logging
5. Implement cost tracking
6. Performance optimization

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::{method, path, header};

    #[tokio::test]
    async fn test_text_to_video_submission() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/gen4-turbo/text-to-video"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "job-123",
                "status": "PENDING"
            })))
            .mount(&mock_server)
            .await;

        let config = RunwayConfig {
            api_token: "test-token".to_string(),
            base_url: Some(mock_server.uri()),
            ..Default::default()
        };

        let provider = RunwayProvider::new(config).unwrap();
        // Test submission logic...
    }

    #[tokio::test]
    async fn test_polling_with_backoff() {
        // Mock a sequence: PENDING -> RUNNING -> SUCCEEDED
    }

    #[tokio::test]
    async fn test_rate_limit_handling() {
        // Mock 429 response and verify retry behavior
    }

    #[tokio::test]
    async fn test_authentication_failure() {
        // Mock 401 and verify error mapping
    }

    #[tokio::test]
    async fn test_content_filter_rejection() {
        // Mock content policy violation response
    }
}
```

### Integration Tests

Integration tests require a valid Runway API token and are gated behind a
feature flag:

```rust
#[cfg(feature = "runway-integration-tests")]
mod integration {
    #[tokio::test]
    #[ignore = "Requires RUNWAY_API_TOKEN"]
    async fn test_real_text_to_video() {
        let token = std::env::var("RUNWAY_API_TOKEN")
            .expect("RUNWAY_API_TOKEN must be set");

        let config = RunwayConfig {
            api_token: token,
            ..Default::default()
        };

        let provider = RunwayProvider::new(config).unwrap();

        let health = provider.health_check().await.unwrap();
        assert!(matches!(health, HealthStatus::Healthy));

        let result = provider.generate(
            &GenerationPrompt::Text("A serene mountain lake at sunset".into()),
            &GenerationParams {
                duration_secs: 5,
                aspect_ratio: "16:9".into(),
                ..Default::default()
            },
        ).await.unwrap();

        match result {
            GenerationOutput::Video(clip) => {
                assert!(clip.duration > 0.0);
                assert!(clip.path.exists());
            }
            _ => panic!("Expected video output"),
        }
    }
}
```

### Mock Server Tests

A full mock Runway server is provided for development without API access:

```rust
pub struct MockRunwayServer {
    server: MockServer,
    job_store: Arc<DashMap<String, RunwayJob>>,
}

impl MockRunwayServer {
    pub async fn start() -> Self {
        let server = MockServer::start().await;
        let job_store = Arc::new(DashMap::new());
        // Configure standard mock endpoints...
        Self { server, job_store }
    }

    pub fn uri(&self) -> String {
        self.server.uri()
    }
}
```

---

## Open Questions

1. **GWM-1 API Availability**: The GWM-1 API may not yet be publicly available.
   The implementation should gracefully degrade when GWM-1 endpoints are
   unavailable, falling back to Gen-4 Turbo for prediction tasks.

2. **Waitlist Access**: Runway API access may require waitlist approval. The
   provider should detect and report waitlist status in `health_check()`.

3. **Embedding Support**: Runway does not natively expose embedding endpoints.
   Should we implement `embed()` by extracting intermediate representations
   from generated videos, or mark it as unsupported?

4. **Video Format Support**: Should we support formats beyond MP4? WebM and
   GIF could be useful for certain downstream applications.

5. **Streaming Output**: Gen-4 Turbo may support progressive video delivery
   in the future. Should we design the interface to accommodate streaming?

6. **Credit Management**: Should the provider track credit usage locally and
   refuse requests when credits are estimated to be exhausted, or always
   defer to the API for credit checking?

7. **Multi-region Support**: Runway may offer regional endpoints for lower
   latency. Should we implement region selection?

8. **Webhook Support**: Instead of polling, Runway may support webhooks for
   job completion notifications. This would be more efficient for long-running
   jobs.

9. **Caching Generated Videos**: Should we implement a local cache for
   deterministic generations (same prompt + seed = same output)?

10. **Batch Generation**: Should we support submitting multiple generation
    requests as a batch for improved throughput?
