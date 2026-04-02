# RFC-0005: OpenAI Sora Provider

| Field   | Value                     |
|---------|---------------------------|
| Title   | OpenAI Sora Provider      |
| Status  | Draft                     |
| Author  | WorldForge Contributors   |
| Created | 2026-04-02                |
| RFC     | 0005                      |

---

## Abstract

This RFC proposes the integration of OpenAI's Sora video generation model as a
provider within the WorldForge framework. Sora is a diffusion transformer model
capable of generating high-quality videos from text prompts, images, and
storyboard sequences. This document covers the OpenAI API integration for Sora,
authentication via API keys, generation modes (text-to-video, image-to-video,
storyboard), resolution and duration options, the async generation lifecycle,
cost estimation based on OpenAI's pricing model, mapping to WorldForge types,
error handling, and rate limits.

---

## Motivation

### Why Sora?

OpenAI's Sora represents a significant advancement in video generation, built
on a diffusion transformer architecture that understands 3D consistency,
temporal coherence, and complex physical interactions. Key advantages include:

1. **World Understanding**: Sora demonstrates emergent understanding of physics,
   object permanence, and spatial relationships—making it a genuine world model
   rather than just a video generator.

2. **Versatile Input Modes**: Supports text-to-video, image-to-video, and a
   unique storyboard mode that allows frame-by-frame narrative control.

3. **Variable Resolution and Duration**: Can generate videos at multiple
   resolutions (up to 1080p) and durations (up to 60 seconds).

4. **OpenAI Ecosystem**: Integrates with OpenAI's broader API ecosystem,
   sharing authentication, billing, and rate limiting infrastructure.

5. **Quality**: Produces some of the highest-quality generated videos available,
   with strong temporal coherence and detail preservation.

### WorldForge Alignment

Sora maps to the `WorldModelProvider` trait:

- `predict()` → Next-frame/sequence prediction using image-to-video
- `generate()` → Full video generation from text/image/storyboard
- `transfer()` → Style transfer via video variation/remix
- `plan()` → Multi-scene storyboard generation
- `reason()` → Scene description and analysis (via complementary GPT-4V)
- `embed()` → Not directly supported (see Open Questions)
- `health_check()` → API availability check
- `cost_estimate()` → Token-based cost calculation

---

## Detailed Design

### 1. OpenAI API for Sora

#### 1.1 API Endpoint Structure

Sora video generation is accessed through the OpenAI API under the videos
namespace:

```
Base URL: https://api.openai.com/v1
Video Generation: POST /v1/videos/generations
Video Status:     GET  /v1/videos/generations/{generation_id}
Video Download:   GET  /v1/videos/{video_id}/content
Storyboard:       POST /v1/videos/storyboards
```

#### 1.2 Authentication

OpenAI uses API key authentication via the `Authorization` header:

```
Authorization: Bearer sk-...
OpenAI-Organization: org-...  (optional, for multi-org accounts)
OpenAI-Project: proj-...      (optional, for project-scoped billing)
```

```rust
pub struct SoraConfig {
    /// OpenAI API key
    pub api_key: String,
    /// Optional organization ID
    pub organization_id: Option<String>,
    /// Optional project ID for billing scope
    pub project_id: Option<String>,
    /// Base URL override (for proxies or Azure OpenAI)
    pub base_url: Option<String>,
    /// Maximum concurrent generations
    pub max_concurrent: usize,
    /// Polling interval for async jobs (milliseconds)
    pub poll_interval_ms: u64,
    /// Maximum wait time for generation (seconds)
    pub max_wait_secs: u64,
    /// Default resolution
    pub default_resolution: SoraResolution,
    /// Default duration
    pub default_duration_secs: u8,
}
```

The provider reads credentials from:
1. `SoraConfig.api_key` (explicit configuration)
2. `OPENAI_API_KEY` environment variable
3. `~/.worldforge/openai_key` file

#### 1.3 Request/Response Model

All Sora API requests return a generation object that follows an async
lifecycle:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoraGeneration {
    /// Unique generation ID
    pub id: String,
    /// Object type (always "video.generation")
    pub object: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Generation status
    pub status: SoraStatus,
    /// Model used
    pub model: String,
    /// Input parameters (echo back)
    pub input: serde_json::Value,
    /// Output video reference (when complete)
    pub output: Option<SoraOutput>,
    /// Error details (when failed)
    pub error: Option<SoraApiError>,
    /// Progress percentage (0-100)
    pub progress: Option<u8>,
    /// Estimated completion time
    pub estimated_completion: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoraStatus {
    Queued,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoraOutput {
    /// Video ID for download
    pub video_id: String,
    /// Duration of generated video in seconds
    pub duration: f64,
    /// Resolution of generated video
    pub resolution: SoraResolution,
    /// Format (always MP4)
    pub format: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Presigned download URL (expires after 1 hour)
    pub download_url: String,
    /// Thumbnail URL
    pub thumbnail_url: Option<String>,
}
```

### 2. Generation Modes

#### 2.1 Text-to-Video

Generate a video from a text description:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct TextToVideoRequest {
    /// Model to use
    pub model: String, // "sora-1" or "sora-turbo"
    /// Text prompt describing the video
    pub prompt: String,
    /// Video duration in seconds (5, 10, 15, 20, 30, 60)
    pub duration: u8,
    /// Output resolution
    pub resolution: SoraResolution,
    /// Aspect ratio
    pub aspect_ratio: SoraAspectRatio,
    /// Number of variants to generate (1-4)
    pub n: Option<u8>,
    /// Negative prompt (what to avoid)
    pub negative_prompt: Option<String>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Style preset
    pub style: Option<SoraStyle>,
    /// FPS (24 or 30)
    pub fps: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SoraResolution {
    #[serde(rename = "480p")]
    Res480p,   // 854x480
    #[serde(rename = "720p")]
    Res720p,   // 1280x720
    #[serde(rename = "1080p")]
    Res1080p,  // 1920x1080
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SoraAspectRatio {
    #[serde(rename = "16:9")]
    Widescreen,
    #[serde(rename = "9:16")]
    Portrait,
    #[serde(rename = "1:1")]
    Square,
    #[serde(rename = "4:3")]
    Standard,
    #[serde(rename = "21:9")]
    Ultrawide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoraStyle {
    Natural,
    Cinematic,
    Animated,
    Abstract,
    Documentary,
}
```

#### 2.2 Image-to-Video

Animate a reference image into a video:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct ImageToVideoRequest {
    pub model: String,
    /// Reference image (URL or base64)
    pub image: SoraImageInput,
    /// Text prompt for motion/action guidance
    pub prompt: String,
    /// Video duration in seconds
    pub duration: u8,
    /// How closely to follow the reference image (0.0-1.0)
    pub image_strength: Option<f32>,
    /// Motion intensity (0.0 = subtle, 1.0 = dynamic)
    pub motion_intensity: Option<f32>,
    /// Seed for reproducibility
    pub seed: Option<u64>,
    /// Number of variants
    pub n: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SoraImageInput {
    Url { url: String },
    Base64 { data: String, media_type: String },
}
```

#### 2.3 Storyboard Mode

The storyboard mode allows frame-by-frame narrative control, specifying
keyframes with descriptions at specific timestamps:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct StoryboardRequest {
    pub model: String,
    /// Ordered sequence of storyboard frames
    pub frames: Vec<StoryboardFrame>,
    /// Overall narrative prompt
    pub narrative: Option<String>,
    /// Total video duration in seconds
    pub duration: u8,
    /// Output resolution
    pub resolution: SoraResolution,
    /// Transition style between frames
    pub transition: Option<TransitionStyle>,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoryboardFrame {
    /// Timestamp in the video (seconds from start)
    pub timestamp: f64,
    /// Description of what should appear at this frame
    pub description: String,
    /// Optional reference image for this keyframe
    pub image: Option<SoraImageInput>,
    /// Camera position/angle at this keyframe
    pub camera: Option<CameraHint>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionStyle {
    Smooth,
    Cut,
    Dissolve,
    Wipe,
}

#[derive(Debug, Clone, Serialize)]
pub struct CameraHint {
    /// Camera angle description (e.g., "close-up", "wide shot")
    pub angle: String,
    /// Camera movement (e.g., "pan left", "zoom in")
    pub movement: Option<String>,
}
```

### 3. Async Generation Lifecycle

```rust
pub struct SoraJobManager {
    client: reqwest::Client,
    config: SoraConfig,
    active_jobs: Arc<DashMap<String, SoraGeneration>>,
    semaphore: Arc<Semaphore>,
}

impl SoraJobManager {
    /// Submit a generation job and return the generation ID
    pub async fn submit<R: Serialize>(
        &self,
        endpoint: &str,
        request: &R,
    ) -> Result<SoraGeneration, SoraError> {
        let _permit = self.semaphore.acquire().await
            .map_err(|_| SoraError::ConcurrencyLimit)?;

        let response = self.client
            .post(format!("{}{}", self.base_url(), endpoint))
            .json(request)
            .send()
            .await?;

        match response.status() {
            reqwest::StatusCode::OK | reqwest::StatusCode::CREATED => {
                let generation: SoraGeneration = response.json().await?;
                self.active_jobs.insert(generation.id.clone(), generation.clone());
                Ok(generation)
            }
            reqwest::StatusCode::UNAUTHORIZED => {
                Err(SoraError::Authentication("Invalid API key".into()))
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => {
                let retry_after = response.headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(60);
                Err(SoraError::RateLimited { retry_after_secs: retry_after })
            }
            reqwest::StatusCode::PAYMENT_REQUIRED => {
                Err(SoraError::InsufficientCredits)
            }
            status => {
                let body = response.text().await.unwrap_or_default();
                Err(SoraError::ApiError { status, body })
            }
        }
    }

    /// Poll until generation completes
    pub async fn wait_for_completion(
        &self,
        generation_id: &str,
    ) -> Result<SoraGeneration, SoraError> {
        let start = Instant::now();
        let max_wait = Duration::from_secs(self.config.max_wait_secs);
        let mut interval = Duration::from_millis(self.config.poll_interval_ms);

        loop {
            if start.elapsed() > max_wait {
                return Err(SoraError::Timeout {
                    generation_id: generation_id.to_string(),
                    elapsed: start.elapsed(),
                });
            }

            tokio::time::sleep(interval).await;

            let response = self.client
                .get(format!(
                    "{}/v1/videos/generations/{}",
                    self.base_url(),
                    generation_id,
                ))
                .send()
                .await?;

            let generation: SoraGeneration = response.json().await?;

            // Update tracking
            self.active_jobs.insert(generation_id.to_string(), generation.clone());

            match generation.status {
                SoraStatus::Completed => return Ok(generation),
                SoraStatus::Failed => {
                    return Err(SoraError::GenerationFailed {
                        generation_id: generation_id.to_string(),
                        error: generation.error,
                    });
                }
                SoraStatus::Cancelled => {
                    return Err(SoraError::GenerationCancelled {
                        generation_id: generation_id.to_string(),
                    });
                }
                SoraStatus::Queued | SoraStatus::Processing => {
                    // Log progress
                    if let Some(progress) = generation.progress {
                        tracing::debug!(
                            "Generation {} progress: {}%",
                            generation_id,
                            progress,
                        );
                    }

                    // Adaptive backoff
                    interval = Duration::from_millis(
                        (interval.as_millis() as f64 * 1.3)
                            .min(15_000.0) as u64
                    );
                }
            }
        }
    }

    /// Download the generated video
    pub async fn download_video(
        &self,
        output: &SoraOutput,
        target_dir: &Path,
    ) -> Result<PathBuf, SoraError> {
        let file_name = format!("{}.mp4", output.video_id);
        let target_path = target_dir.join(&file_name);

        let response = self.client
            .get(&output.download_url)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(SoraError::DownloadFailed {
                video_id: output.video_id.clone(),
                status: response.status(),
            });
        }

        let bytes = response.bytes().await?;
        tokio::fs::write(&target_path, &bytes).await?;

        Ok(target_path)
    }

    /// Cancel a running generation
    pub async fn cancel(
        &self,
        generation_id: &str,
    ) -> Result<(), SoraError> {
        self.client
            .delete(format!(
                "{}/v1/videos/generations/{}",
                self.base_url(),
                generation_id,
            ))
            .send()
            .await?;

        self.active_jobs.remove(generation_id);
        Ok(())
    }
}
```

### 4. Resolution and Duration Options

| Resolution | Dimensions | Aspect Ratios              | Max Duration |
|-----------|------------|----------------------------|-------------|
| 480p      | 854×480    | 16:9, 9:16, 1:1, 4:3      | 60s          |
| 720p      | 1280×720   | 16:9, 9:16, 1:1, 4:3      | 30s          |
| 1080p     | 1920×1080  | 16:9, 9:16, 1:1            | 20s          |

Duration options: 5s, 10s, 15s, 20s, 30s, 60s (varies by resolution).

### 5. Cost Estimation

OpenAI prices Sora based on resolution, duration, and model tier:

```rust
pub struct SoraCostCalculator;

impl SoraCostCalculator {
    pub fn estimate(
        model: &str,
        resolution: &SoraResolution,
        duration_secs: u8,
        num_variants: u8,
    ) -> CostEstimate {
        let per_second_rate = match (model, resolution) {
            ("sora-1", SoraResolution::Res480p)  => 0.10,
            ("sora-1", SoraResolution::Res720p)  => 0.20,
            ("sora-1", SoraResolution::Res1080p) => 0.40,
            ("sora-turbo", SoraResolution::Res480p)  => 0.05,
            ("sora-turbo", SoraResolution::Res720p)  => 0.10,
            ("sora-turbo", SoraResolution::Res1080p) => 0.20,
            _ => 0.20, // default
        };

        let total = per_second_rate * duration_secs as f64 * num_variants as f64;

        CostEstimate {
            credits: 0,
            estimated_usd: total,
            currency: "USD".to_string(),
            breakdown: vec![
                CostItem {
                    description: format!(
                        "{} {}p × {}s × {} variant(s)",
                        model, resolution.height(), duration_secs, num_variants,
                    ),
                    amount: total,
                },
            ],
        }
    }
}
```

### 6. Mapping to WorldForge Types

```rust
#[async_trait]
impl WorldModelProvider for SoraProvider {
    async fn predict(
        &self,
        input: &WorldState,
        params: &PredictionParams,
    ) -> Result<WorldState, ProviderError> {
        // Use image-to-video to predict next state from current frame
        let current_frame = input.last_frame()
            .ok_or(ProviderError::InvalidInput("No frames in world state".into()))?;

        let request = ImageToVideoRequest {
            model: self.config.model_name(),
            image: SoraImageInput::Base64 {
                data: current_frame.to_base64()?,
                media_type: "image/png".to_string(),
            },
            prompt: params.prediction_prompt.clone()
                .unwrap_or_else(|| "Continue this scene naturally".to_string()),
            duration: params.duration_secs.unwrap_or(5),
            image_strength: Some(0.8),
            motion_intensity: Some(0.5),
            seed: params.seed,
            n: Some(1),
        };

        let generation = self.job_manager.submit(
            "/v1/videos/generations",
            &request,
        ).await?;

        let completed = self.job_manager
            .wait_for_completion(&generation.id)
            .await?;

        let output = completed.output
            .ok_or(SoraError::NoOutput)?;
        let video_path = self.job_manager
            .download_video(&output, &self.temp_dir)
            .await?;

        let clip = self.video_to_clip(&video_path, &output).await?;

        Ok(WorldState {
            id: Uuid::new_v4().to_string(),
            parent_id: Some(input.id.clone()),
            video_clip: Some(clip),
            timestamp: input.timestamp + output.duration,
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        params: &GenerationParams,
    ) -> Result<GenerationOutput, ProviderError> {
        let generation = match prompt {
            GenerationPrompt::Text(text) => {
                let request = TextToVideoRequest {
                    model: self.config.model_name(),
                    prompt: text.clone(),
                    duration: params.duration_secs.unwrap_or(10),
                    resolution: params.resolution.clone()
                        .unwrap_or(self.config.default_resolution.clone()),
                    aspect_ratio: params.aspect_ratio.clone()
                        .unwrap_or(SoraAspectRatio::Widescreen),
                    n: Some(1),
                    negative_prompt: params.negative_prompt.clone(),
                    seed: params.seed,
                    style: params.style.clone().map(|s| s.into()),
                    fps: params.fps,
                };
                self.job_manager.submit("/v1/videos/generations", &request).await?
            }
            GenerationPrompt::Image(img) => {
                let request = ImageToVideoRequest {
                    model: self.config.model_name(),
                    image: img.to_sora_input()?,
                    prompt: params.prompt.clone().unwrap_or_default(),
                    duration: params.duration_secs.unwrap_or(10),
                    image_strength: params.image_strength,
                    motion_intensity: params.motion_intensity,
                    seed: params.seed,
                    n: Some(1),
                };
                self.job_manager.submit("/v1/videos/generations", &request).await?
            }
            GenerationPrompt::Storyboard(frames) => {
                let request = StoryboardRequest {
                    model: self.config.model_name(),
                    frames: frames.iter().map(|f| f.into()).collect(),
                    narrative: params.narrative.clone(),
                    duration: params.duration_secs.unwrap_or(15),
                    resolution: params.resolution.clone()
                        .unwrap_or(self.config.default_resolution.clone()),
                    transition: params.transition.clone(),
                    seed: params.seed,
                };
                self.job_manager.submit("/v1/videos/storyboards", &request).await?
            }
        };

        let completed = self.job_manager
            .wait_for_completion(&generation.id)
            .await?;

        let output = completed.output.ok_or(SoraError::NoOutput)?;
        let video_path = self.job_manager
            .download_video(&output, &self.temp_dir)
            .await?;

        let clip = self.video_to_clip(&video_path, &output).await?;
        Ok(GenerationOutput::Video(clip))
    }

    async fn transfer(
        &self,
        source: &MediaInput,
        target_style: &StyleParams,
    ) -> Result<GenerationOutput, ProviderError> {
        // Use image-to-video with strong style guidance
        let first_frame = source.first_frame()?;

        let request = ImageToVideoRequest {
            model: self.config.model_name(),
            image: SoraImageInput::Base64 {
                data: first_frame.to_base64()?,
                media_type: "image/png".to_string(),
            },
            prompt: format!(
                "Transform this scene in the style of: {}",
                target_style.description
            ),
            duration: source.duration_secs().unwrap_or(5),
            image_strength: Some(target_style.strength.unwrap_or(0.5)),
            motion_intensity: Some(0.3),
            seed: target_style.seed,
            n: Some(1),
        };

        let generation = self.job_manager.submit(
            "/v1/videos/generations",
            &request,
        ).await?;

        let completed = self.job_manager
            .wait_for_completion(&generation.id)
            .await?;

        let output = completed.output.ok_or(SoraError::NoOutput)?;
        let video_path = self.job_manager
            .download_video(&output, &self.temp_dir)
            .await?;

        let clip = self.video_to_clip(&video_path, &output).await?;
        Ok(GenerationOutput::Video(clip))
    }

    async fn plan(
        &self,
        initial: &WorldState,
        goal: &WorldState,
        params: &PlanningParams,
    ) -> Result<Plan, ProviderError> {
        // Build a storyboard from initial to goal state
        let mut frames = Vec::new();

        // Start frame from initial state
        if let Some(frame) = initial.last_frame() {
            frames.push(StoryboardFrame {
                timestamp: 0.0,
                description: initial.description.clone().unwrap_or_default(),
                image: Some(SoraImageInput::Base64 {
                    data: frame.to_base64()?,
                    media_type: "image/png".to_string(),
                }),
                camera: None,
            });
        }

        // Goal frame
        if let Some(frame) = goal.last_frame() {
            let duration = params.duration_secs.unwrap_or(15) as f64;
            frames.push(StoryboardFrame {
                timestamp: duration,
                description: goal.description.clone().unwrap_or_default(),
                image: Some(SoraImageInput::Base64 {
                    data: frame.to_base64()?,
                    media_type: "image/png".to_string(),
                }),
                camera: None,
            });
        }

        let request = StoryboardRequest {
            model: self.config.model_name(),
            frames,
            narrative: params.narrative.clone(),
            duration: params.duration_secs.unwrap_or(15),
            resolution: SoraResolution::Res720p,
            transition: Some(TransitionStyle::Smooth),
            seed: params.seed,
        };

        let generation = self.job_manager.submit(
            "/v1/videos/storyboards",
            &request,
        ).await?;

        let completed = self.job_manager
            .wait_for_completion(&generation.id)
            .await?;

        let output = completed.output.ok_or(SoraError::NoOutput)?;
        let video_path = self.job_manager
            .download_video(&output, &self.temp_dir)
            .await?;

        let clip = self.video_to_clip(&video_path, &output).await?;

        Ok(Plan {
            steps: vec![WorldState {
                id: Uuid::new_v4().to_string(),
                video_clip: Some(clip),
                ..Default::default()
            }],
            total_steps: 1,
            goal_reached: true,
        })
    }

    async fn health_check(&self) -> Result<HealthStatus, ProviderError> {
        let response = self.client
            .get(format!("{}/v1/models", self.base_url()))
            .send()
            .await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(HealthStatus::Healthy),
            reqwest::StatusCode::UNAUTHORIZED => {
                Ok(HealthStatus::Unhealthy("Invalid API key".into()))
            }
            status => Ok(HealthStatus::Degraded(
                format!("API returned {}", status)
            )),
        }
    }

    async fn cost_estimate(
        &self,
        params: &GenerationParams,
    ) -> Result<CostEstimate, ProviderError> {
        Ok(SoraCostCalculator::estimate(
            &self.config.model_name(),
            &params.resolution.clone().unwrap_or(SoraResolution::Res720p),
            params.duration_secs.unwrap_or(10),
            params.num_variants.unwrap_or(1),
        ))
    }
}
```

### 7. Rate Limits

OpenAI enforces tiered rate limits:

| Tier   | RPM  | Videos/day | Max Concurrent | Max Resolution |
|--------|------|-----------|----------------|----------------|
| Tier 1 | 5    | 50        | 2              | 720p           |
| Tier 2 | 15   | 200       | 5              | 1080p          |
| Tier 3 | 30   | 500       | 10             | 1080p          |
| Tier 4 | 60   | 1000      | 20             | 1080p          |
| Tier 5 | 120  | Unlimited | 50             | 1080p          |

Rate limit headers are included in responses:

```rust
pub struct RateLimitInfo {
    pub limit_requests: u32,
    pub remaining_requests: u32,
    pub reset_requests: Duration,
    pub limit_videos: u32,
    pub remaining_videos: u32,
}

impl RateLimitInfo {
    pub fn from_headers(headers: &reqwest::header::HeaderMap) -> Option<Self> {
        Some(Self {
            limit_requests: headers.get("x-ratelimit-limit-requests")?
                .to_str().ok()?.parse().ok()?,
            remaining_requests: headers.get("x-ratelimit-remaining-requests")?
                .to_str().ok()?.parse().ok()?,
            reset_requests: Duration::from_secs(
                headers.get("x-ratelimit-reset-requests")?
                    .to_str().ok()?.parse().ok()?,
            ),
            limit_videos: headers.get("x-ratelimit-limit-videos")?
                .to_str().ok()?.parse().ok()?,
            remaining_videos: headers.get("x-ratelimit-remaining-videos")?
                .to_str().ok()?.parse().ok()?,
        })
    }
}
```

### 8. Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum SoraError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Authentication failed: {0}")]
    Authentication(String),

    #[error("Rate limited, retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },

    #[error("Insufficient credits/balance")]
    InsufficientCredits,

    #[error("Generation {generation_id} failed: {error:?}")]
    GenerationFailed {
        generation_id: String,
        error: Option<SoraApiError>,
    },

    #[error("Generation {generation_id} cancelled")]
    GenerationCancelled { generation_id: String },

    #[error("Generation {generation_id} timed out after {elapsed:?}")]
    Timeout {
        generation_id: String,
        elapsed: Duration,
    },

    #[error("Content policy violation: {0}")]
    ContentPolicy(String),

    #[error("No output produced")]
    NoOutput,

    #[error("Video download failed for {video_id}: {status}")]
    DownloadFailed {
        video_id: String,
        status: reqwest::StatusCode,
    },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Concurrency limit reached")]
    ConcurrencyLimit,

    #[error("API error ({status}): {body}")]
    ApiError {
        status: reqwest::StatusCode,
        body: String,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<SoraError> for ProviderError {
    fn from(e: SoraError) -> Self {
        match e {
            SoraError::Authentication(_) => ProviderError::Authentication(e.to_string()),
            SoraError::RateLimited { .. } => ProviderError::RateLimit(e.to_string()),
            SoraError::ContentPolicy(_) => ProviderError::ContentPolicy(e.to_string()),
            SoraError::Timeout { .. } => ProviderError::Timeout(e.to_string()),
            SoraError::InsufficientCredits => ProviderError::Billing(e.to_string()),
            _ => ProviderError::Provider(e.to_string()),
        }
    }
}
```

---

## Implementation Plan

### Phase 1: Core API Client (Week 1-2)

1. Create `crates/worldforge-providers/src/sora/` module
2. Implement `SoraConfig` with environment variable support
3. Build authenticated HTTP client with reqwest
4. Implement async job submission and polling
5. Implement video download with retry logic
6. Add error type hierarchy and conversions

### Phase 2: Generation Modes (Week 3-4)

1. Implement text-to-video generation
2. Implement image-to-video generation
3. Implement storyboard mode
4. Add resolution and duration validation
5. Implement cost calculator

### Phase 3: WorldModelProvider Trait (Week 5-6)

1. Implement all trait methods (predict, generate, transfer, plan, etc.)
2. Add video-to-clip conversion utilities
3. Implement frame extraction from downloaded videos
4. Add progress reporting via tracing

### Phase 4: Production Features (Week 7-8)

1. Implement rate limit tracking and proactive throttling
2. Add retry logic with exponential backoff
3. Implement concurrent generation management
4. Add job cancellation support
5. Performance optimization and caching

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
    async fn test_text_to_video_request() {
        let mock_server = MockServer::start().await;
        // Mock generation submission
        Mock::given(method("POST"))
            .and(path("/v1/videos/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "gen-123",
                "object": "video.generation",
                "status": "queued",
                "created_at": 1700000000,
                "model": "sora-turbo"
            })))
            .mount(&mock_server)
            .await;

        // Test request building and submission...
    }

    #[tokio::test]
    async fn test_cost_estimation() {
        let cost = SoraCostCalculator::estimate(
            "sora-turbo",
            &SoraResolution::Res720p,
            10,
            1,
        );
        assert_eq!(cost.estimated_usd, 1.0);
    }

    #[tokio::test]
    async fn test_rate_limit_parsing() {
        // Test parsing rate limit headers
    }

    #[tokio::test]
    async fn test_error_mapping() {
        // Test SoraError -> ProviderError conversion
    }
}
```

### Integration Tests

```rust
#[cfg(feature = "sora-integration-tests")]
mod integration {
    #[tokio::test]
    #[ignore = "Requires OPENAI_API_KEY with Sora access"]
    async fn test_real_text_to_video() {
        let config = SoraConfig::from_env().unwrap();
        let provider = SoraProvider::new(config).unwrap();

        let result = provider.generate(
            &GenerationPrompt::Text("A cat playing piano".into()),
            &GenerationParams {
                duration_secs: Some(5),
                resolution: Some(SoraResolution::Res480p),
                ..Default::default()
            },
        ).await.unwrap();

        assert!(matches!(result, GenerationOutput::Video(_)));
    }
}
```

---

## Open Questions

1. **Sora API Stability**: The Sora API may still be evolving. How do we
   handle breaking API changes?

2. **Embedding Support**: OpenAI may add embedding extraction from Sora in
   the future. Should we stub `embed()` now?

3. **Azure OpenAI**: Should we support Azure OpenAI deployments of Sora with
   different authentication (Azure AD tokens)?

4. **Streaming Progress**: Can we receive frame-by-frame progress during
   generation for real-time preview?

5. **Video Variations**: Sora may support "remix" or "variation" endpoints.
   How should these map to WorldForge operations?

6. **Multi-clip Editing**: Should we support generating multiple clips and
   compositing them together?

7. **Safety Filters**: How should we handle OpenAI's content safety filters?
   Should we pre-screen prompts before submission?

8. **Caching**: Should we cache generated videos locally by prompt hash to
   avoid redundant API calls during development?

9. **WebSocket Support**: If OpenAI adds WebSocket-based progress updates,
   should we prefer that over polling?

10. **Budget Guards**: Should the provider enforce per-session or per-project
    budget limits to prevent unexpected charges?
