# RFC-0006: Google Veo Provider via Vertex AI

| Field   | Value                              |
|---------|------------------------------------|
| Title   | Google Veo Provider via Vertex AI  |
| Status  | Draft                              |
| Author  | WorldForge Contributors            |
| Created | 2026-04-02                         |
| RFC     | 0006                               |

---

## Abstract

This RFC proposes the integration of Google's Veo video generation model as a
WorldForge provider, accessed through the Vertex AI platform. Veo is Google
DeepMind's state-of-the-art video generation model capable of producing
high-quality, high-fidelity videos from text and image prompts. This document
covers the Vertex AI API integration, Google Cloud authentication (service
accounts, Application Default Credentials), Veo generation modes, image-to-video
transfer, resolution options, the async job model, mapping to WorldForge types,
pricing, and error handling.

---

## Motivation

### Why Google Veo?

Google's Veo model, available through Vertex AI, brings several unique
advantages to the WorldForge ecosystem:

1. **Google Cloud Integration**: Veo is deeply integrated with Google Cloud
   Platform (GCP), providing enterprise-grade infrastructure, security,
   compliance, and observability out of the box.

2. **High Fidelity**: Veo generates videos with exceptional visual quality,
   temporal consistency, and natural motion at resolutions up to 4K.

3. **Imagen Synergy**: Veo shares architecture components with Google's Imagen
   image generation model, enabling seamless image-to-video workflows.

4. **Enterprise Features**: Vertex AI provides built-in monitoring, logging,
   IAM-based access control, VPC Service Controls, and CMEK encryption.

5. **SynthID Watermarking**: All Veo-generated content includes invisible
   SynthID watermarks for responsible AI provenance tracking.

6. **Grounding**: Veo can be grounded with Google Search results for factual
   consistency in generated content.

### WorldForge Alignment

Veo maps to the `WorldModelProvider` trait:

- `predict()` → Image-to-video continuation of current world state
- `generate()` → Text-to-video and image-to-video generation
- `transfer()` → Style-guided video generation
- `plan()` → Sequential scene generation
- `reason()` → Via complementary Gemini integration for scene analysis
- `embed()` → Via Video Intelligence API for feature extraction
- `health_check()` → Vertex AI endpoint health check
- `cost_estimate()` → GCP pricing-based cost calculation

---

## Detailed Design

### 1. Vertex AI API

#### 1.1 API Structure

Veo is accessed through the Vertex AI Generative AI endpoints:

```
Base URL: https://{REGION}-aiplatform.googleapis.com/v1
Projects: /projects/{PROJECT_ID}/locations/{REGION}
Endpoint: /publishers/google/models/veo-2:generateVideo
Status:   /projects/{PROJECT_ID}/locations/{REGION}/operations/{OPERATION_ID}
```

The full endpoint URL for video generation:

```
POST https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/{MODEL_ID}:predictLongRunning
```

#### 1.2 Supported Regions

| Region          | Location        | Veo Availability |
|----------------|-----------------|------------------|
| us-central1    | Iowa, USA       | GA               |
| us-east4       | Virginia, USA   | GA               |
| europe-west4   | Netherlands     | Preview          |
| asia-northeast1| Tokyo, Japan    | Preview          |

#### 1.3 Model Versions

| Model ID   | Description                    | Max Resolution | Max Duration |
|-----------|--------------------------------|----------------|-------------|
| veo-2     | Latest production model         | 4K (3840×2160) | 60s         |
| veo-2-hd  | Optimized for HD content        | 1080p          | 120s        |
| veo-3     | Next-gen (preview)              | 4K             | 60s         |

### 2. Google Cloud Authentication

#### 2.1 Service Account Authentication

The recommended authentication method for production deployments:

```rust
pub struct VeoConfig {
    /// GCP Project ID
    pub project_id: String,
    /// GCP Region
    pub region: String,
    /// Model ID to use
    pub model_id: String,
    /// Authentication method
    pub auth: GcpAuth,
    /// Maximum concurrent operations
    pub max_concurrent: usize,
    /// Polling interval for long-running operations (ms)
    pub poll_interval_ms: u64,
    /// Maximum wait time (seconds)
    pub max_wait_secs: u64,
    /// Default output resolution
    pub default_resolution: VeoResolution,
    /// Default duration
    pub default_duration_secs: u8,
    /// Enable SynthID watermarking metadata retrieval
    pub enable_synthid_metadata: bool,
}

#[derive(Debug, Clone)]
pub enum GcpAuth {
    /// Service account key file (JSON)
    ServiceAccountKey { key_path: PathBuf },
    /// Service account key from environment variable
    ServiceAccountKeyJson { json: String },
    /// Application Default Credentials (gcloud auth)
    ApplicationDefault,
    /// Workload Identity (for GKE)
    WorkloadIdentity,
    /// Access token (short-lived, for testing)
    AccessToken { token: String },
}
```

#### 2.2 Application Default Credentials (ADC)

ADC provides automatic credential discovery following Google's standard
credential chain:

```rust
pub struct GcpTokenProvider {
    auth: GcpAuth,
    cached_token: Arc<RwLock<Option<CachedToken>>>,
}

#[derive(Debug, Clone)]
struct CachedToken {
    access_token: String,
    expires_at: Instant,
}

impl GcpTokenProvider {
    pub async fn new(auth: GcpAuth) -> Result<Self, VeoError> {
        let provider = Self {
            auth,
            cached_token: Arc::new(RwLock::new(None)),
        };
        // Validate credentials on creation
        provider.get_token().await?;
        Ok(provider)
    }

    pub async fn get_token(&self) -> Result<String, VeoError> {
        // Check cache first
        {
            let cached = self.cached_token.read().await;
            if let Some(token) = cached.as_ref() {
                if token.expires_at > Instant::now() + Duration::from_secs(60) {
                    return Ok(token.access_token.clone());
                }
            }
        }

        // Refresh token
        let new_token = match &self.auth {
            GcpAuth::ServiceAccountKey { key_path } => {
                self.token_from_service_account(key_path).await?
            }
            GcpAuth::ServiceAccountKeyJson { json } => {
                self.token_from_service_account_json(json).await?
            }
            GcpAuth::ApplicationDefault => {
                self.token_from_adc().await?
            }
            GcpAuth::WorkloadIdentity => {
                self.token_from_metadata_server().await?
            }
            GcpAuth::AccessToken { token } => {
                CachedToken {
                    access_token: token.clone(),
                    expires_at: Instant::now() + Duration::from_secs(3600),
                }
            }
        };

        let token_str = new_token.access_token.clone();
        *self.cached_token.write().await = Some(new_token);
        Ok(token_str)
    }

    async fn token_from_service_account(
        &self,
        key_path: &Path,
    ) -> Result<CachedToken, VeoError> {
        let key_json = tokio::fs::read_to_string(key_path).await
            .map_err(|e| VeoError::AuthError(format!(
                "Failed to read service account key: {}", e
            )))?;

        self.token_from_service_account_json(&key_json).await
    }

    async fn token_from_service_account_json(
        &self,
        json: &str,
    ) -> Result<CachedToken, VeoError> {
        let sa: ServiceAccountKey = serde_json::from_str(json)
            .map_err(|e| VeoError::AuthError(format!(
                "Invalid service account key JSON: {}", e
            )))?;

        // Create JWT and exchange for access token
        let jwt = self.create_jwt(&sa)?;
        let token_response = self.exchange_jwt_for_token(&jwt).await?;

        Ok(CachedToken {
            access_token: token_response.access_token,
            expires_at: Instant::now() + Duration::from_secs(
                token_response.expires_in.unwrap_or(3600)
            ),
        })
    }

    async fn token_from_adc(&self) -> Result<CachedToken, VeoError> {
        // Check well-known locations:
        // 1. GOOGLE_APPLICATION_CREDENTIALS env var
        // 2. gcloud default credentials file
        // 3. GCE metadata server

        if let Ok(cred_path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            return self.token_from_service_account(Path::new(&cred_path)).await;
        }

        let default_path = dirs::config_dir()
            .unwrap_or_default()
            .join("gcloud")
            .join("application_default_credentials.json");

        if default_path.exists() {
            return self.token_from_service_account(&default_path).await;
        }

        // Try metadata server (for GCE/GKE)
        self.token_from_metadata_server().await
    }

    async fn token_from_metadata_server(&self) -> Result<CachedToken, VeoError> {
        let response = reqwest::Client::new()
            .get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
            .header("Metadata-Flavor", "Google")
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|_| VeoError::AuthError(
                "Metadata server unavailable (not running on GCE?)".into()
            ))?;

        let token_data: MetadataTokenResponse = response.json().await?;

        Ok(CachedToken {
            access_token: token_data.access_token,
            expires_at: Instant::now() + Duration::from_secs(token_data.expires_in),
        })
    }
}
```

### 3. Veo Generation Modes

#### 3.1 Text-to-Video

```rust
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoTextToVideoRequest {
    /// Instances containing the generation parameters
    pub instances: Vec<VeoInstance>,
    /// Generation configuration
    pub parameters: VeoParameters,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoInstance {
    /// Text prompt for video generation
    pub prompt: String,
    /// Optional reference image for style/content guidance
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<VeoImage>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoImage {
    /// Base64-encoded image bytes
    pub bytes_base64_encoded: String,
    /// MIME type
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoParameters {
    /// Video duration in seconds
    pub video_duration_seconds: u8,
    /// Output resolution
    pub output_resolution: VeoResolution,
    /// Aspect ratio
    pub aspect_ratio: VeoAspectRatio,
    /// Number of videos to generate (1-4)
    pub sample_count: u8,
    /// Random seed for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Guidance scale (how closely to follow the prompt)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guidance_scale: Option<f32>,
    /// Negative prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// FPS for output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u8>,
    /// Enable person/face generation (requires additional approval)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub person_generation: Option<PersonGenerationSetting>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum VeoResolution {
    #[serde(rename = "480p")]
    Res480p,
    #[serde(rename = "720p")]
    Res720p,
    #[serde(rename = "1080p")]
    Res1080p,
    #[serde(rename = "4k")]
    Res4K,
}

#[derive(Debug, Clone, Serialize)]
pub enum VeoAspectRatio {
    #[serde(rename = "16:9")]
    Widescreen,
    #[serde(rename = "9:16")]
    Portrait,
    #[serde(rename = "1:1")]
    Square,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PersonGenerationSetting {
    AllowAdult,
    DontAllow,
}
```

#### 3.2 Image-to-Video

Image-to-video uses the same endpoint but includes a reference image:

```rust
impl VeoProvider {
    fn build_image_to_video_request(
        &self,
        image: &MediaInput,
        prompt: &str,
        params: &VeoParameters,
    ) -> Result<VeoTextToVideoRequest, VeoError> {
        let image_bytes = image.to_bytes()?;
        let base64 = base64::engine::general_purpose::STANDARD
            .encode(&image_bytes);

        Ok(VeoTextToVideoRequest {
            instances: vec![VeoInstance {
                prompt: prompt.to_string(),
                image: Some(VeoImage {
                    bytes_base64_encoded: base64,
                    mime_type: image.mime_type().to_string(),
                }),
            }],
            parameters: params.clone(),
        })
    }
}
```

#### 3.3 Video Extension

Veo supports extending existing videos by appending or prepending content:

```rust
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoExtendRequest {
    pub instances: Vec<VeoExtendInstance>,
    pub parameters: VeoParameters,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoExtendInstance {
    /// Text prompt for the extension
    pub prompt: String,
    /// Reference video to extend
    pub video: VeoVideoRef,
    /// Extension direction
    pub extend_direction: ExtendDirection,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoVideoRef {
    /// GCS URI of the source video
    pub gcs_uri: Option<String>,
    /// Base64-encoded video bytes
    pub bytes_base64_encoded: Option<String>,
    /// MIME type
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExtendDirection {
    Forward,
    Backward,
}
```

### 4. Async Job Model (Long-Running Operations)

Vertex AI uses Google's Long-Running Operations (LRO) pattern:

```rust
pub struct VeoOperationManager {
    client: reqwest::Client,
    token_provider: GcpTokenProvider,
    config: VeoConfig,
    active_operations: Arc<DashMap<String, VeoOperation>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoOperation {
    /// Operation resource name
    pub name: String,
    /// Whether the operation is complete
    pub done: bool,
    /// Operation metadata
    pub metadata: Option<serde_json::Value>,
    /// Result (on success)
    pub response: Option<VeoResponse>,
    /// Error (on failure)
    pub error: Option<GcpStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoResponse {
    /// Generated video predictions
    pub predictions: Vec<VeoPrediction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VeoPrediction {
    /// GCS URI of generated video
    pub gcs_uri: String,
    /// Video duration in seconds
    pub duration_seconds: f64,
    /// Resolution
    pub resolution: String,
    /// MIME type
    pub mime_type: String,
    /// SynthID watermark metadata
    pub synthid_metadata: Option<SynthIdMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcpStatus {
    pub code: i32,
    pub message: String,
    pub details: Option<Vec<serde_json::Value>>,
}

impl VeoOperationManager {
    pub async fn submit(
        &self,
        request: &VeoTextToVideoRequest,
    ) -> Result<VeoOperation, VeoError> {
        let token = self.token_provider.get_token().await?;
        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:predictLongRunning",
            self.config.region,
            self.config.project_id,
            self.config.region,
            self.config.model_id,
        );

        let response = self.client
            .post(&url)
            .bearer_auth(&token)
            .json(request)
            .send()
            .await?;

        match response.status() {
            reqwest::StatusCode::OK => {
                let operation: VeoOperation = response.json().await?;
                self.active_operations.insert(
                    operation.name.clone(),
                    operation.clone(),
                );
                Ok(operation)
            }
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                Err(VeoError::AuthError("Insufficient permissions".into()))
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => {
                Err(VeoError::RateLimited)
            }
            status => {
                let body = response.text().await.unwrap_or_default();
                Err(VeoError::ApiError {
                    status: status.as_u16(),
                    message: body,
                })
            }
        }
    }

    pub async fn poll_operation(
        &self,
        operation_name: &str,
    ) -> Result<VeoOperation, VeoError> {
        let token = self.token_provider.get_token().await?;
        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/{}",
            self.config.region,
            operation_name,
        );

        let response = self.client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        let operation: VeoOperation = response.json().await?;
        self.active_operations.insert(
            operation_name.to_string(),
            operation.clone(),
        );

        Ok(operation)
    }

    pub async fn wait_for_completion(
        &self,
        operation_name: &str,
    ) -> Result<VeoOperation, VeoError> {
        let start = Instant::now();
        let max_wait = Duration::from_secs(self.config.max_wait_secs);
        let mut interval = Duration::from_millis(self.config.poll_interval_ms);

        loop {
            if start.elapsed() > max_wait {
                return Err(VeoError::Timeout {
                    operation: operation_name.to_string(),
                    elapsed: start.elapsed(),
                });
            }

            tokio::time::sleep(interval).await;
            let operation = self.poll_operation(operation_name).await?;

            if operation.done {
                if let Some(error) = operation.error {
                    return Err(VeoError::OperationFailed {
                        operation: operation_name.to_string(),
                        code: error.code,
                        message: error.message,
                    });
                }
                return Ok(operation);
            }

            // Adaptive backoff
            interval = Duration::from_millis(
                (interval.as_millis() as f64 * 1.5).min(20_000.0) as u64
            );
        }
    }

    pub async fn cancel_operation(
        &self,
        operation_name: &str,
    ) -> Result<(), VeoError> {
        let token = self.token_provider.get_token().await?;
        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/{}:cancel",
            self.config.region,
            operation_name,
        );

        self.client
            .post(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        self.active_operations.remove(operation_name);
        Ok(())
    }
}
```

### 5. Downloading Results from GCS

Veo outputs are stored in Google Cloud Storage. We need to download them:

```rust
pub struct GcsDownloader {
    client: reqwest::Client,
    token_provider: GcpTokenProvider,
}

impl GcsDownloader {
    pub async fn download(
        &self,
        gcs_uri: &str,
        target_path: &Path,
    ) -> Result<(), VeoError> {
        // Parse gs://bucket/path format
        let (bucket, object) = parse_gcs_uri(gcs_uri)?;

        let token = self.token_provider.get_token().await?;
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}?alt=media",
            bucket,
            urlencoding::encode(object),
        );

        let response = self.client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await?;

        let bytes = response.bytes().await?;
        tokio::fs::write(target_path, &bytes).await?;

        Ok(())
    }
}

fn parse_gcs_uri(uri: &str) -> Result<(&str, &str), VeoError> {
    let stripped = uri.strip_prefix("gs://")
        .ok_or_else(|| VeoError::InvalidGcsUri(uri.to_string()))?;
    let slash_pos = stripped.find('/')
        .ok_or_else(|| VeoError::InvalidGcsUri(uri.to_string()))?;
    Ok((&stripped[..slash_pos], &stripped[slash_pos + 1..]))
}
```

### 6. Mapping to WorldForge Types

```rust
#[async_trait]
impl WorldModelProvider for VeoProvider {
    async fn predict(
        &self,
        input: &WorldState,
        params: &PredictionParams,
    ) -> Result<WorldState, ProviderError> {
        let current_frame = input.last_frame()
            .ok_or(ProviderError::InvalidInput("No frames in state".into()))?;

        let request = self.build_image_to_video_request(
            &current_frame.into(),
            &params.prediction_prompt.clone()
                .unwrap_or("Continue this scene".to_string()),
            &VeoParameters {
                video_duration_seconds: params.duration_secs.unwrap_or(5),
                output_resolution: VeoResolution::Res720p,
                aspect_ratio: VeoAspectRatio::Widescreen,
                sample_count: 1,
                seed: params.seed,
                ..Default::default()
            },
        )?;

        let operation = self.op_manager.submit(&request).await?;
        let completed = self.op_manager
            .wait_for_completion(&operation.name)
            .await?;

        let prediction = &completed.response
            .ok_or(VeoError::NoOutput)?
            .predictions[0];

        let video_path = self.temp_dir.join(format!("{}.mp4", Uuid::new_v4()));
        self.gcs_downloader.download(&prediction.gcs_uri, &video_path).await?;

        let clip = self.video_to_clip(&video_path, prediction).await?;

        Ok(WorldState {
            id: Uuid::new_v4().to_string(),
            parent_id: Some(input.id.clone()),
            video_clip: Some(clip),
            timestamp: input.timestamp + prediction.duration_seconds,
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        params: &GenerationParams,
    ) -> Result<GenerationOutput, ProviderError> {
        let request = match prompt {
            GenerationPrompt::Text(text) => {
                VeoTextToVideoRequest {
                    instances: vec![VeoInstance {
                        prompt: text.clone(),
                        image: None,
                    }],
                    parameters: VeoParameters {
                        video_duration_seconds: params.duration_secs.unwrap_or(10),
                        output_resolution: params.resolution.clone()
                            .unwrap_or(VeoResolution::Res1080p),
                        aspect_ratio: params.aspect_ratio.clone()
                            .unwrap_or(VeoAspectRatio::Widescreen),
                        sample_count: params.num_variants.unwrap_or(1),
                        seed: params.seed,
                        negative_prompt: params.negative_prompt.clone(),
                        guidance_scale: params.guidance_scale,
                        ..Default::default()
                    },
                }
            }
            GenerationPrompt::Image(img) => {
                self.build_image_to_video_request(
                    img,
                    &params.prompt.clone().unwrap_or_default(),
                    &VeoParameters {
                        video_duration_seconds: params.duration_secs.unwrap_or(10),
                        output_resolution: params.resolution.clone()
                            .unwrap_or(VeoResolution::Res1080p),
                        sample_count: 1,
                        ..Default::default()
                    },
                )?
            }
            _ => return Err(ProviderError::UnsupportedMode(
                "Veo only supports text and image prompts".into()
            )),
        };

        let operation = self.op_manager.submit(&request).await?;
        let completed = self.op_manager
            .wait_for_completion(&operation.name)
            .await?;

        let response = completed.response.ok_or(VeoError::NoOutput)?;
        let prediction = &response.predictions[0];

        let video_path = self.temp_dir.join(format!("{}.mp4", Uuid::new_v4()));
        self.gcs_downloader.download(&prediction.gcs_uri, &video_path).await?;

        let clip = self.video_to_clip(&video_path, prediction).await?;
        Ok(GenerationOutput::Video(clip))
    }

    async fn transfer(
        &self,
        source: &MediaInput,
        target_style: &StyleParams,
    ) -> Result<GenerationOutput, ProviderError> {
        // Use image-to-video with style-focused prompt
        let first_frame = source.first_frame()?;

        let request = self.build_image_to_video_request(
            &first_frame.into(),
            &format!("Transform in style: {}", target_style.description),
            &VeoParameters {
                video_duration_seconds: source.duration_secs().unwrap_or(5) as u8,
                output_resolution: VeoResolution::Res1080p,
                aspect_ratio: VeoAspectRatio::Widescreen,
                sample_count: 1,
                guidance_scale: Some(target_style.strength.unwrap_or(0.7) * 20.0),
                ..Default::default()
            },
        )?;

        let operation = self.op_manager.submit(&request).await?;
        let completed = self.op_manager
            .wait_for_completion(&operation.name)
            .await?;

        let prediction = &completed.response.ok_or(VeoError::NoOutput)?.predictions[0];
        let video_path = self.temp_dir.join(format!("{}.mp4", Uuid::new_v4()));
        self.gcs_downloader.download(&prediction.gcs_uri, &video_path).await?;

        let clip = self.video_to_clip(&video_path, prediction).await?;
        Ok(GenerationOutput::Video(clip))
    }

    async fn health_check(&self) -> Result<HealthStatus, ProviderError> {
        match self.token_provider.get_token().await {
            Ok(_) => {
                // Try a lightweight API call
                let url = format!(
                    "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}",
                    self.config.region,
                    self.config.project_id,
                    self.config.region,
                );

                match self.client.get(&url)
                    .bearer_auth(&self.token_provider.get_token().await?)
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        Ok(HealthStatus::Healthy)
                    }
                    Ok(resp) => Ok(HealthStatus::Degraded(
                        format!("API returned {}", resp.status())
                    )),
                    Err(e) => Ok(HealthStatus::Unhealthy(e.to_string())),
                }
            }
            Err(e) => Ok(HealthStatus::Unhealthy(
                format!("Authentication failed: {}", e)
            )),
        }
    }

    async fn cost_estimate(
        &self,
        params: &GenerationParams,
    ) -> Result<CostEstimate, ProviderError> {
        Ok(VeoCostCalculator::estimate(
            &self.config.model_id,
            &params.resolution.clone().unwrap_or(VeoResolution::Res1080p),
            params.duration_secs.unwrap_or(10),
            params.num_variants.unwrap_or(1),
        ))
    }
}
```

### 7. Pricing

Vertex AI Veo pricing (estimated, per second of generated video):

| Resolution | Price/second | 10s Video | 30s Video |
|-----------|-------------|-----------|-----------|
| 480p      | $0.05       | $0.50     | $1.50     |
| 720p      | $0.10       | $1.00     | $3.00     |
| 1080p     | $0.20       | $2.00     | $6.00     |
| 4K        | $0.50       | $5.00     | $15.00    |

```rust
pub struct VeoCostCalculator;

impl VeoCostCalculator {
    pub fn estimate(
        model: &str,
        resolution: &VeoResolution,
        duration_secs: u8,
        sample_count: u8,
    ) -> CostEstimate {
        let per_second = match resolution {
            VeoResolution::Res480p => 0.05,
            VeoResolution::Res720p => 0.10,
            VeoResolution::Res1080p => 0.20,
            VeoResolution::Res4K => 0.50,
        };

        let total = per_second * duration_secs as f64 * sample_count as f64;

        CostEstimate {
            credits: 0,
            estimated_usd: total,
            currency: "USD".to_string(),
            breakdown: vec![
                CostItem {
                    description: format!(
                        "Veo {} {:?} × {}s × {} sample(s)",
                        model, resolution, duration_secs, sample_count,
                    ),
                    amount: total,
                },
            ],
        }
    }
}
```

### 8. Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum VeoError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Authentication error: {0}")]
    AuthError(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Operation {operation} failed (code {code}): {message}")]
    OperationFailed {
        operation: String,
        code: i32,
        message: String,
    },

    #[error("Operation {operation} timed out after {elapsed:?}")]
    Timeout {
        operation: String,
        elapsed: Duration,
    },

    #[error("No output produced")]
    NoOutput,

    #[error("Invalid GCS URI: {0}")]
    InvalidGcsUri(String),

    #[error("GCS download failed: {0}")]
    GcsDownloadFailed(String),

    #[error("API error (HTTP {status}): {message}")]
    ApiError { status: u16, message: String },

    #[error("Content safety violation")]
    SafetyViolation,

    #[error("Quota exceeded for project {project_id}")]
    QuotaExceeded { project_id: String },

    #[error("Region {region} does not support Veo")]
    UnsupportedRegion { region: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<VeoError> for ProviderError {
    fn from(e: VeoError) -> Self {
        match e {
            VeoError::AuthError(_) => ProviderError::Authentication(e.to_string()),
            VeoError::RateLimited | VeoError::QuotaExceeded { .. } => {
                ProviderError::RateLimit(e.to_string())
            }
            VeoError::SafetyViolation => ProviderError::ContentPolicy(e.to_string()),
            VeoError::Timeout { .. } => ProviderError::Timeout(e.to_string()),
            _ => ProviderError::Provider(e.to_string()),
        }
    }
}
```

---

## Implementation Plan

### Phase 1: GCP Authentication (Week 1-2)

1. Create `crates/worldforge-providers/src/veo/` module
2. Implement service account key parsing and JWT creation
3. Implement Application Default Credentials discovery
4. Implement token caching and refresh
5. Add workload identity support for GKE

### Phase 2: Vertex AI Client (Week 3-4)

1. Implement Long-Running Operations submit/poll/cancel
2. Implement GCS download for video outputs
3. Add adaptive polling with backoff
4. Implement error type hierarchy

### Phase 3: Veo Generation (Week 5-6)

1. Implement text-to-video generation
2. Implement image-to-video generation
3. Implement video extension
4. Add resolution and duration validation
5. Implement cost calculator

### Phase 4: WorldModelProvider Trait (Week 7-8)

1. Implement all trait methods
2. Add video-to-clip conversion
3. Implement SynthID metadata extraction
4. Add production hardening (retries, logging, metrics)

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcs_uri_parsing() {
        let (bucket, path) = parse_gcs_uri("gs://my-bucket/path/to/video.mp4").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(path, "path/to/video.mp4");
    }

    #[tokio::test]
    async fn test_token_caching() {
        // Test that tokens are cached and reused
    }

    #[tokio::test]
    async fn test_operation_polling() {
        // Mock Vertex AI LRO responses
    }

    #[test]
    fn test_cost_estimation() {
        let cost = VeoCostCalculator::estimate(
            "veo-2", &VeoResolution::Res1080p, 10, 1
        );
        assert_eq!(cost.estimated_usd, 2.0);
    }
}
```

### Integration Tests

```rust
#[cfg(feature = "veo-integration-tests")]
mod integration {
    #[tokio::test]
    #[ignore = "Requires GCP credentials and Vertex AI access"]
    async fn test_real_text_to_video() {
        let config = VeoConfig::from_env().unwrap();
        let provider = VeoProvider::new(config).await.unwrap();

        let health = provider.health_check().await.unwrap();
        assert!(matches!(health, HealthStatus::Healthy));

        let result = provider.generate(
            &GenerationPrompt::Text("A butterfly landing on a flower".into()),
            &GenerationParams {
                duration_secs: Some(5),
                resolution: Some(VeoResolution::Res480p),
                ..Default::default()
            },
        ).await.unwrap();

        assert!(matches!(result, GenerationOutput::Video(_)));
    }
}
```

---

## Open Questions

1. **GCS Bucket Configuration**: Should we require users to specify an output
   GCS bucket, or use Vertex AI's default temporary storage?

2. **VPC Service Controls**: How should we handle VPC-SC restricted projects
   where the API endpoint may be different?

3. **Regional Failover**: Should we implement automatic failover to another
   region if the primary region is unavailable?

4. **Gemini Integration**: Should we integrate Gemini for the `reason()` trait
   method, providing scene understanding for Veo-generated content?

5. **Batch Prediction**: Vertex AI supports batch prediction jobs. Should we
   implement a bulk generation mode?

6. **Custom Endpoints**: Should we support Vertex AI Model Garden custom
   deployments of Veo?

7. **Billing Alerts**: Should we integrate with GCP billing alerts to warn
   users about spending thresholds?

8. **SynthID Verification**: Should we implement SynthID watermark verification
   for ingested videos to detect AI-generated content?

9. **Video Intelligence API**: Should we use Google's Video Intelligence API
   for the `embed()` method?

10. **Streaming Output**: Does Vertex AI support streaming partial video
    output? If so, should we implement progressive delivery?
