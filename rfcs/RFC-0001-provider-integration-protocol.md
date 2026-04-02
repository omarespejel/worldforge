# RFC-0001: WorldForge Provider Integration Protocol

| Field     | Value                                      |
|-----------|--------------------------------------------|
| Title     | WorldForge Provider Integration Protocol   |
| Status    | Draft                                      |
| Authors   | WorldForge Core Team                       |
| Created   | 2026-04-02                                 |
| Updated   | 2026-04-02                                 |
| RFC       | 0001                                       |

---

## Abstract

This RFC defines the standard protocol, trait interface, and operational
requirements for integrating a new world model provider into WorldForge.
It covers the full lifecycle from initial stub implementation through
production deployment, including the `WorldModelProvider` trait with its
8 core operations, authentication patterns, error handling, rate limiting,
video/frame pipelines, testing contracts, and health monitoring.

Any provider — whether NVIDIA Cosmos, Google DeepMind, Runway, or a
custom local model — MUST conform to this protocol to be accepted into
the WorldForge provider registry.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Detailed Design](#detailed-design)
   - 2.1 [WorldModelProvider Trait](#21-worldmodelprovider-trait)
   - 2.2 [Core Operations](#22-core-operations)
   - 2.3 [Provider Metadata Methods](#23-provider-metadata-methods)
   - 2.4 [ProviderCapabilities](#24-providercapabilities)
   - 2.5 [Authentication Patterns](#25-authentication-patterns)
   - 2.6 [Rate Limiting and Retry Strategy](#26-rate-limiting-and-retry-strategy)
   - 2.7 [Error Mapping](#27-error-mapping)
   - 2.8 [Async HTTP Client Patterns](#28-async-http-client-patterns)
   - 2.9 [Response Parsing and Validation](#29-response-parsing-and-validation)
   - 2.10 [Video and Frame Handling Pipeline](#210-video-and-frame-handling-pipeline)
   - 2.11 [Cost Tracking](#211-cost-tracking)
   - 2.12 [Provider Health Monitoring](#212-provider-health-monitoring)
3. [Step-by-Step Provider Implementation Guide](#step-by-step-provider-implementation-guide)
4. [Integration Test Contract](#integration-test-contract)
5. [Mock and Replay Testing Strategy](#mock-and-replay-testing-strategy)
6. [Implementation Plan](#implementation-plan)
7. [Testing Strategy](#testing-strategy)
8. [Open Questions](#open-questions)

---

## Motivation

WorldForge is designed as a multi-provider world model framework. Users
should be able to swap between providers (NVIDIA Cosmos, Google Genie,
Runway Gen-3, local models) without changing application logic. To achieve
this, every provider must implement a uniform interface and adhere to
consistent behavioral contracts.

Without a formal protocol:
- Each provider would have ad-hoc error handling, making debugging painful.
- Authentication would be inconsistent, forcing users to learn per-provider quirks.
- Testing would be fragmented, with no guarantee of baseline quality.
- Video/frame handling would be duplicated across providers.
- Cost tracking would be impossible to aggregate.

This RFC establishes the single source of truth for provider integration.

---

## Detailed Design

### 2.1 WorldModelProvider Trait

The `WorldModelProvider` trait is the central abstraction. Every provider
MUST implement this trait. It is defined as an async trait using Rust's
native async trait support (Rust 1.75+).

```rust
use async_trait::async_trait;
use crate::types::{
    WorldState, Action, Prediction, Plan, Embedding,
    Reasoning, GeneratedWorld, TransferResult,
    HealthStatus, CostEstimate, ProviderCapabilities,
    WorldForgeError, PredictOptions, PlanOptions,
    EmbedOptions, ReasonOptions, GenerateOptions,
    TransferOptions, CostEstimateRequest,
};

#[async_trait]
pub trait WorldModelProvider: Send + Sync + 'static {
    // ── Metadata ──────────────────────────────────────────────
    /// Unique provider identifier, e.g. "nvidia-cosmos", "runway-gen3"
    fn name(&self) -> &str;

    /// Human-readable description of the provider and its capabilities.
    fn describe(&self) -> String;

    /// Declares which operations this provider supports.
    fn capabilities(&self) -> ProviderCapabilities;

    // ── Core Operations (8) ───────────────────────────────────

    /// Predict the next world state given current state and action.
    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        options: PredictOptions,
    ) -> Result<Prediction, WorldForgeError>;

    /// Generate a multi-step plan from current state to goal state.
    async fn plan(
        &self,
        current: &WorldState,
        goal: &WorldState,
        options: PlanOptions,
    ) -> Result<Plan, WorldForgeError>;

    /// Embed a world state into a latent vector space.
    async fn embed(
        &self,
        state: &WorldState,
        options: EmbedOptions,
    ) -> Result<Embedding, WorldForgeError>;

    /// Reason about the world state: causal analysis, counterfactuals.
    async fn reason(
        &self,
        state: &WorldState,
        query: &str,
        options: ReasonOptions,
    ) -> Result<Reasoning, WorldForgeError>;

    /// Generate a new world (video frames, 3D scene, etc.) from a prompt.
    async fn generate(
        &self,
        prompt: &str,
        options: GenerateOptions,
    ) -> Result<GeneratedWorld, WorldForgeError>;

    /// Transfer style/properties from one world state to another.
    async fn transfer(
        &self,
        source: &WorldState,
        target: &WorldState,
        options: TransferOptions,
    ) -> Result<TransferResult, WorldForgeError>;

    /// Check provider health and availability.
    async fn health_check(&self) -> Result<HealthStatus, WorldForgeError>;

    /// Estimate cost for a given operation before executing it.
    async fn cost_estimate(
        &self,
        request: &CostEstimateRequest,
    ) -> Result<CostEstimate, WorldForgeError>;
}
```

The trait is object-safe. Provider instances are typically stored as
`Arc<dyn WorldModelProvider>` to allow sharing across async tasks.

### 2.2 Core Operations

#### 2.2.1 predict

**Purpose:** Given a world state and an action, predict the resulting world state.

**Input:**
- `state: &WorldState` — The current world state (frames, metadata, scene graph).
- `action: &Action` — The action to simulate (movement, force, manipulation).
- `options: PredictOptions` — Horizon length, confidence threshold, frame format.

**Output:**
- `Prediction` containing:
  - `next_state: WorldState` — The predicted next state.
  - `confidence: f64` — Model confidence in [0.0, 1.0].
  - `frames: Vec<Frame>` — If the prediction includes video frames.
  - `metadata: HashMap<String, Value>` — Provider-specific metadata.
  - `latency_ms: u64` — Time taken for the prediction.

**Behavioral Contract:**
- MUST return within `options.timeout` or return `WorldForgeError::Timeout`.
- MUST NOT modify the input state (immutable borrow).
- If the provider does not support prediction, return `WorldForgeError::UnsupportedOperation`.
- Confidence MUST be normalized to [0.0, 1.0].

#### 2.2.2 plan

**Purpose:** Generate a sequence of actions to transition from current state to goal state.

**Input:**
- `current: &WorldState` — Starting state.
- `goal: &WorldState` — Desired end state.
- `options: PlanOptions` — Max steps, time budget, optimization criteria.

**Output:**
- `Plan` containing:
  - `steps: Vec<PlanStep>` — Ordered sequence of actions with predicted intermediate states.
  - `total_cost: f64` — Estimated total cost (provider-defined units).
  - `estimated_duration: Duration` — How long the plan would take to execute.
  - `feasibility: f64` — Confidence the plan is achievable, in [0.0, 1.0].

**Behavioral Contract:**
- MUST return at least one step or an error explaining why no plan exists.
- Steps MUST be in chronological order.
- Each step MUST include a predicted intermediate state.

#### 2.2.3 embed

**Purpose:** Map a world state into a dense vector embedding for similarity search,
clustering, or downstream ML tasks.

**Input:**
- `state: &WorldState` — The state to embed.
- `options: EmbedOptions` — Embedding dimension, normalization, model variant.

**Output:**
- `Embedding` containing:
  - `vector: Vec<f32>` — The embedding vector.
  - `dimension: usize` — Vector dimensionality.
  - `model: String` — Which embedding model was used.
  - `normalized: bool` — Whether the vector is L2-normalized.

**Behavioral Contract:**
- Vector length MUST equal the declared dimension.
- If `options.normalize` is true, the vector MUST be L2-normalized.
- Embeddings for identical states MUST be deterministic (same input -> same output).

#### 2.2.4 reason

**Purpose:** Perform causal reasoning, counterfactual analysis, or question-answering
about a world state.

**Input:**
- `state: &WorldState` — The world state to reason about.
- `query: &str` — Natural language question or structured reasoning query.
- `options: ReasonOptions` — Reasoning depth, causal mode, max tokens.

**Output:**
- `Reasoning` containing:
  - `answer: String` — The reasoning output (natural language or structured).
  - `confidence: f64` — Confidence in the reasoning, [0.0, 1.0].
  - `evidence: Vec<Evidence>` — Supporting evidence (frame references, spatial refs).
  - `causal_chain: Option<Vec<CausalLink>>` — If causal reasoning was requested.

**Behavioral Contract:**
- MUST provide a non-empty answer or an error.
- If causal reasoning is requested but not supported, return `UnsupportedOperation`.

#### 2.2.5 generate

**Purpose:** Generate a new world (video, 3D scene, image sequence) from a text prompt
or structured specification.

**Input:**
- `prompt: &str` — Text description of the world to generate.
- `options: GenerateOptions` — Resolution, frame count, style, seed, format.

**Output:**
- `GeneratedWorld` containing:
  - `frames: Vec<Frame>` — Generated visual frames.
  - `scene: Option<SceneGraph>` — Optional 3D scene representation.
  - `metadata: HashMap<String, Value>` — Generation metadata.
  - `seed: u64` — The seed used (for reproducibility).

**Behavioral Contract:**
- Frame count MUST match `options.num_frames` if specified.
- Resolution MUST match `options.resolution` or return an error.
- If a seed is provided, regeneration with the same seed SHOULD produce identical output.

#### 2.2.6 transfer

**Purpose:** Transfer visual style, physics properties, or semantic content from one
world state to another.

**Input:**
- `source: &WorldState` — The state to transfer FROM.
- `target: &WorldState` — The state to transfer TO.
- `options: TransferOptions` — Transfer mode (style, physics, semantic), strength.

**Output:**
- `TransferResult` containing:
  - `result_state: WorldState` — The target state with transferred properties.
  - `transfer_map: HashMap<String, f64>` — What was transferred and by how much.

**Behavioral Contract:**
- The result MUST have the same spatial dimensions as the target.
- Transfer strength of 0.0 MUST return the target unchanged.
- Transfer strength of 1.0 SHOULD maximally apply source properties.

#### 2.2.7 health_check

**Purpose:** Verify the provider is reachable, authenticated, and operational.

**Input:** None.

**Output:**
- `HealthStatus` containing:
  - `healthy: bool` — Overall health.
  - `latency_ms: u64` — Round-trip time to the provider.
  - `api_version: String` — Provider API version.
  - `quota_remaining: Option<u64>` — Remaining API quota if applicable.
  - `message: Option<String>` — Human-readable status message.

**Behavioral Contract:**
- MUST complete within 10 seconds.
- MUST NOT consume billable API calls if possible.
- SHOULD cache results for at least 30 seconds.

#### 2.2.8 cost_estimate

**Purpose:** Estimate the cost of an operation before executing it, enabling
budget-aware orchestration.

**Input:**
- `request: &CostEstimateRequest` containing:
  - `operation: OperationType` — Which operation to estimate.
  - `input_size: InputSize` — Approximate input dimensions.
  - `options: HashMap<String, Value>` — Operation-specific parameters.

**Output:**
- `CostEstimate` containing:
  - `estimated_cost_usd: f64` — Estimated cost in USD.
  - `estimated_latency_ms: u64` — Estimated wall-clock time.
  - `estimated_tokens: Option<u64>` — Token count if applicable.
  - `confidence: f64` — How confident the estimate is, [0.0, 1.0].
  - `breakdown: HashMap<String, f64>` — Cost breakdown by component.

**Behavioral Contract:**
- MUST NOT execute the actual operation.
- MUST return in under 1 second (local computation or cached pricing).
- Cost MUST be in USD for cross-provider comparability.

### 2.3 Provider Metadata Methods

#### name()

Returns a unique, lowercase, hyphenated identifier:
- `"nvidia-cosmos"`
- `"runway-gen3"`
- `"google-genie"`
- `"local-diffusion"`

This identifier is used in configuration files, logging, metrics, and the
provider registry.

#### describe()

Returns a human-readable description including:
- Provider name and version
- Supported operations
- Notable features or limitations

Example: `"NVIDIA Cosmos v2.1 — World foundation model supporting predict,
reason, embed, and generate operations with physics-aware simulation."`

#### capabilities()

Returns a `ProviderCapabilities` struct declaring supported operations.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub predict: bool,
    pub plan: bool,
    pub embed: bool,
    pub reason: bool,
    pub generate: bool,
    pub transfer: bool,
    pub spatial_control: bool,
    pub real_time: bool,
    pub action_conditioning: bool,
}
```

Rules:
- If a capability is `false`, calling the corresponding operation MUST return
  `WorldForgeError::UnsupportedOperation { operation, provider }`.
- Capabilities MUST NOT change during the lifetime of a provider instance.
- The provider registry uses capabilities for routing and fallback logic.

### 2.4 ProviderCapabilities

The `ProviderCapabilities` struct has 9 boolean fields:

| Capability            | Description                                            |
|-----------------------|--------------------------------------------------------|
| `predict`             | Can predict next world state given state + action      |
| `plan`                | Can generate multi-step action plans                   |
| `embed`               | Can produce vector embeddings of world states          |
| `reason`              | Can perform causal/counterfactual reasoning            |
| `generate`            | Can generate new worlds from prompts                   |
| `transfer`            | Can transfer properties between world states           |
| `spatial_control`     | Supports fine-grained spatial/camera control            |
| `real_time`           | Can operate at real-time or near-real-time speeds      |
| `action_conditioning` | Supports conditioning generation on action sequences   |

Providers SHOULD implement a `ProviderCapabilities::none()` constructor
(all false) and builder methods:

```rust
impl ProviderCapabilities {
    pub fn none() -> Self {
        Self {
            predict: false,
            plan: false,
            embed: false,
            reason: false,
            generate: false,
            transfer: false,
            spatial_control: false,
            real_time: false,
            action_conditioning: false,
        }
    }

    pub fn with_predict(mut self) -> Self { self.predict = true; self }
    pub fn with_plan(mut self) -> Self { self.plan = true; self }
    pub fn with_embed(mut self) -> Self { self.embed = true; self }
    pub fn with_reason(mut self) -> Self { self.reason = true; self }
    pub fn with_generate(mut self) -> Self { self.generate = true; self }
    pub fn with_transfer(mut self) -> Self { self.transfer = true; self }
    pub fn with_spatial_control(mut self) -> Self { self.spatial_control = true; self }
    pub fn with_real_time(mut self) -> Self { self.real_time = true; self }
    pub fn with_action_conditioning(mut self) -> Self { self.action_conditioning = true; self }

    /// Returns true if this provider supports the given operation.
    pub fn supports(&self, op: OperationType) -> bool {
        match op {
            OperationType::Predict => self.predict,
            OperationType::Plan => self.plan,
            OperationType::Embed => self.embed,
            OperationType::Reason => self.reason,
            OperationType::Generate => self.generate,
            OperationType::Transfer => self.transfer,
        }
    }
}
```

### 2.5 Authentication Patterns

WorldForge supports three authentication mechanisms. Providers MUST use
at least one.

#### 2.5.1 API Key Authentication

The simplest pattern. An API key is passed as a header on every request.

```rust
pub struct ApiKeyAuth {
    header_name: String,  // e.g., "Authorization", "X-API-Key"
    key: SecretString,     // From the `secrecy` crate for safe handling
}

impl ApiKeyAuth {
    pub fn apply(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        request.header(&self.header_name, self.key.expose_secret())
    }
}
```

Key management rules:
- API keys MUST be loaded from environment variables or a secrets manager.
- API keys MUST NOT appear in logs, error messages, or serialized configs.
- The `secrecy` crate MUST be used to wrap keys (`SecretString`).
- Keys SHOULD be validated on provider construction (call `health_check`).

Environment variable convention: `WORLDFORGE_{PROVIDER}_API_KEY`
- `WORLDFORGE_NVIDIA_COSMOS_API_KEY`
- `WORLDFORGE_RUNWAY_API_KEY`
- `WORLDFORGE_GOOGLE_GENIE_API_KEY`

#### 2.5.2 JWT Authentication

For providers that issue short-lived tokens from a long-lived credential.

```rust
pub struct JwtAuth {
    token: Arc<RwLock<String>>,
    refresh_token: SecretString,
    token_url: String,
    expires_at: Arc<RwLock<Instant>>,
    client: reqwest::Client,
}

impl JwtAuth {
    pub async fn apply(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, WorldForgeError> {
        let token = self.get_valid_token().await?;
        Ok(request.bearer_auth(&token))
    }

    async fn get_valid_token(&self) -> Result<String, WorldForgeError> {
        let expires_at = *self.expires_at.read().await;
        if Instant::now() < expires_at - Duration::from_secs(60) {
            // Token still valid (with 60s buffer)
            return Ok(self.token.read().await.clone());
        }
        self.refresh().await
    }

    async fn refresh(&self) -> Result<String, WorldForgeError> {
        let resp = self.client
            .post(&self.token_url)
            .json(&serde_json::json!({
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token.expose_secret(),
            }))
            .send()
            .await
            .map_err(|e| WorldForgeError::Auth(format!("Token refresh failed: {e}")))?;

        let body: TokenResponse = resp.json().await
            .map_err(|e| WorldForgeError::Auth(format!("Invalid token response: {e}")))?;

        let mut token = self.token.write().await;
        *token = body.access_token.clone();
        let mut expires = self.expires_at.write().await;
        *expires = Instant::now() + Duration::from_secs(body.expires_in);

        Ok(body.access_token)
    }
}
```

#### 2.5.3 OAuth2 Authentication

For enterprise providers requiring OAuth2 client credentials flow.

```rust
pub struct OAuth2Auth {
    client_id: String,
    client_secret: SecretString,
    token_url: String,
    scopes: Vec<String>,
    token_cache: Arc<RwLock<Option<CachedToken>>>,
    client: reqwest::Client,
}
```

The implementation follows the standard OAuth2 client credentials grant
(RFC 6749 Section 4.4). Token caching and automatic refresh are mandatory.

#### 2.5.4 Authentication Trait

All auth mechanisms implement a common trait:

```rust
#[async_trait]
pub trait ProviderAuth: Send + Sync {
    async fn apply(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, WorldForgeError>;

    async fn validate(&self) -> Result<(), WorldForgeError>;
}
```

### 2.6 Rate Limiting and Retry Strategy

#### 2.6.1 Rate Limiter

Every provider MUST use the WorldForge rate limiter to respect API quotas.

```rust
pub struct RateLimiter {
    /// Maximum requests per second
    max_rps: f64,
    /// Maximum concurrent requests
    max_concurrent: usize,
    /// Token bucket for rate limiting
    bucket: Arc<Mutex<TokenBucket>>,
    /// Semaphore for concurrency limiting
    semaphore: Arc<Semaphore>,
}

impl RateLimiter {
    pub async fn acquire(&self) -> Result<RateLimitGuard, WorldForgeError> {
        // Acquire concurrency permit
        let permit = self.semaphore
            .acquire()
            .await
            .map_err(|_| WorldForgeError::Internal("Semaphore closed".into()))?;

        // Wait for rate limit token
        self.bucket.lock().await.wait_for_token().await;

        Ok(RateLimitGuard { _permit: permit })
    }
}
```

Default limits per provider tier:
- Free tier: 1 RPS, 2 concurrent
- Standard tier: 10 RPS, 10 concurrent
- Enterprise tier: 100 RPS, 50 concurrent

#### 2.6.2 Retry Strategy

WorldForge uses exponential backoff with jitter for retries.

```rust
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub backoff_multiplier: f64,
    pub retryable_errors: Vec<RetryableError>,
    pub jitter: JitterStrategy,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            retryable_errors: vec![
                RetryableError::RateLimit,
                RetryableError::ServerError,
                RetryableError::Timeout,
                RetryableError::ConnectionReset,
            ],
            jitter: JitterStrategy::Full,
        }
    }
}

pub enum JitterStrategy {
    /// No jitter — exact exponential backoff
    None,
    /// Full jitter — uniform random between 0 and backoff
    Full,
    /// Equal jitter — half backoff + random half
    Equal,
    /// Decorrelated jitter — based on previous sleep time
    Decorrelated,
}
```

Retry decision logic:
- HTTP 429 (Too Many Requests): ALWAYS retry, respect `Retry-After` header.
- HTTP 500, 502, 503, 504: Retry up to `max_retries`.
- HTTP 401, 403: Do NOT retry (auth problem).
- HTTP 400, 404, 422: Do NOT retry (client error).
- Network timeout: Retry up to `max_retries`.
- Connection refused: Retry with longer backoff.

```rust
async fn execute_with_retry<F, T>(
    policy: &RetryPolicy,
    operation: F,
) -> Result<T, WorldForgeError>
where
    F: Fn() -> Pin<Box<dyn Future<Output = Result<T, WorldForgeError>> + Send>>,
{
    let mut attempt = 0;
    let mut last_error = None;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) if e.is_retryable() && attempt < policy.max_retries => {
                last_error = Some(e);
                let backoff = calculate_backoff(policy, attempt);
                tracing::warn!(
                    attempt = attempt + 1,
                    max = policy.max_retries,
                    backoff_ms = backoff.as_millis(),
                    "Retrying after error"
                );
                tokio::time::sleep(backoff).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}
```

### 2.7 Error Mapping

All provider-specific errors MUST be mapped to `WorldForgeError`.

```rust
#[derive(Debug, thiserror::Error)]
pub enum WorldForgeError {
    #[error("Provider '{provider}' does not support operation '{operation}'")]
    UnsupportedOperation {
        operation: String,
        provider: String,
    },

    #[error("Authentication failed for provider '{provider}': {message}")]
    Auth {
        provider: String,
        message: String,
    },

    #[error("Rate limited by provider '{provider}': retry after {retry_after_ms}ms")]
    RateLimited {
        provider: String,
        retry_after_ms: u64,
    },

    #[error("Provider '{provider}' request timed out after {timeout_ms}ms")]
    Timeout {
        provider: String,
        timeout_ms: u64,
    },

    #[error("Provider '{provider}' returned invalid response: {message}")]
    InvalidResponse {
        provider: String,
        message: String,
    },

    #[error("Provider '{provider}' server error (HTTP {status}): {message}")]
    ProviderError {
        provider: String,
        status: u16,
        message: String,
    },

    #[error("Network error communicating with '{provider}': {message}")]
    Network {
        provider: String,
        message: String,
    },

    #[error("Video processing error: {message}")]
    VideoProcessing {
        message: String,
    },

    #[error("Validation error: {message}")]
    Validation {
        message: String,
    },

    #[error("Cost budget exceeded: estimated ${estimated} exceeds budget ${budget}")]
    BudgetExceeded {
        estimated: f64,
        budget: f64,
    },

    #[error("Internal error: {message}")]
    Internal {
        message: String,
    },
}

impl WorldForgeError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            WorldForgeError::RateLimited { .. }
            | WorldForgeError::Timeout { .. }
            | WorldForgeError::Network { .. }
            | WorldForgeError::ProviderError { status, .. } if *status >= 500
        )
    }
}
```

#### Error Mapping from HTTP Responses

```rust
fn map_http_error(
    provider: &str,
    status: reqwest::StatusCode,
    body: &str,
) -> WorldForgeError {
    match status.as_u16() {
        401 | 403 => WorldForgeError::Auth {
            provider: provider.to_string(),
            message: body.to_string(),
        },
        429 => WorldForgeError::RateLimited {
            provider: provider.to_string(),
            retry_after_ms: parse_retry_after(body).unwrap_or(1000),
        },
        400 | 422 => WorldForgeError::Validation {
            message: format!("[{provider}] {body}"),
        },
        404 => WorldForgeError::ProviderError {
            provider: provider.to_string(),
            status: 404,
            message: "Endpoint not found".to_string(),
        },
        408 => WorldForgeError::Timeout {
            provider: provider.to_string(),
            timeout_ms: 0,
        },
        500..=599 => WorldForgeError::ProviderError {
            provider: provider.to_string(),
            status: status.as_u16(),
            message: body.to_string(),
        },
        _ => WorldForgeError::Internal {
            message: format!("[{provider}] Unexpected HTTP {}: {}", status, body),
        },
    }
}
```

### 2.8 Async HTTP Client Patterns

WorldForge uses `reqwest` 0.12 with `rustls-tls` for all HTTP communication.

#### 2.8.1 Client Construction

Each provider MUST construct its HTTP client once and reuse it:

```rust
pub struct ProviderHttpClient {
    client: reqwest::Client,
    base_url: String,
    auth: Box<dyn ProviderAuth>,
    rate_limiter: RateLimiter,
    retry_policy: RetryPolicy,
    provider_name: String,
}

impl ProviderHttpClient {
    pub fn new(config: ProviderHttpConfig) -> Result<Self, WorldForgeError> {
        let client = reqwest::Client::builder()
            .use_rustls_tls()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .pool_max_idle_per_host(config.max_idle_connections)
            .pool_idle_timeout(Duration::from_secs(90))
            .user_agent(format!("WorldForge/{}", env!("CARGO_PKG_VERSION")))
            .default_headers({
                let mut headers = HeaderMap::new();
                headers.insert("Accept", "application/json".parse().unwrap());
                headers
            })
            .build()
            .map_err(|e| WorldForgeError::Internal {
                message: format!("Failed to build HTTP client: {e}"),
            })?;

        Ok(Self {
            client,
            base_url: config.base_url,
            auth: config.auth,
            rate_limiter: config.rate_limiter,
            retry_policy: config.retry_policy,
            provider_name: config.provider_name,
        })
    }
}
```

#### 2.8.2 Request Execution

```rust
impl ProviderHttpClient {
    pub async fn post<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        path: &str,
        body: &Req,
    ) -> Result<Resp, WorldForgeError> {
        execute_with_retry(&self.retry_policy, || {
            let client = self.client.clone();
            let url = format!("{}{}", self.base_url, path);
            let auth = &self.auth;
            let rate_limiter = &self.rate_limiter;
            let provider = &self.provider_name;

            Box::pin(async move {
                let _guard = rate_limiter.acquire().await?;

                let request = client.post(&url).json(body);
                let request = auth.apply(request).await?;

                let response = request.send().await.map_err(|e| {
                    if e.is_timeout() {
                        WorldForgeError::Timeout {
                            provider: provider.clone(),
                            timeout_ms: 0,
                        }
                    } else {
                        WorldForgeError::Network {
                            provider: provider.clone(),
                            message: e.to_string(),
                        }
                    }
                })?;

                let status = response.status();
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    return Err(map_http_error(provider, status, &body));
                }

                response.json::<Resp>().await.map_err(|e| {
                    WorldForgeError::InvalidResponse {
                        provider: provider.clone(),
                        message: format!("Failed to parse response: {e}"),
                    }
                })
            })
        })
        .await
    }
}
```

#### 2.8.3 Streaming Responses

For large video responses, providers SHOULD use streaming:

```rust
pub async fn download_video(
    &self,
    url: &str,
) -> Result<Vec<u8>, WorldForgeError> {
    let response = self.client.get(url).send().await.map_err(|e| {
        WorldForgeError::Network {
            provider: self.provider_name.clone(),
            message: e.to_string(),
        }
    })?;

    let total_size = response.content_length();
    let mut bytes = Vec::with_capacity(total_size.unwrap_or(1024 * 1024) as usize);
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| WorldForgeError::Network {
            provider: self.provider_name.clone(),
            message: format!("Stream interrupted: {e}"),
        })?;
        bytes.extend_from_slice(&chunk);
    }

    Ok(bytes)
}
```

### 2.9 Response Parsing and Validation

Every response from a provider API MUST be validated before returning to the caller.

#### 2.9.1 Schema Validation

```rust
pub trait ResponseValidator {
    type Response;
    type Validated;

    fn validate(
        &self,
        response: Self::Response,
    ) -> Result<Self::Validated, WorldForgeError>;
}

// Example: Prediction response validator
pub struct PredictionValidator {
    expected_frame_count: Option<usize>,
    expected_resolution: Option<(u32, u32)>,
}

impl ResponseValidator for PredictionValidator {
    type Response = RawPredictionResponse;
    type Validated = Prediction;

    fn validate(
        &self,
        response: RawPredictionResponse,
    ) -> Result<Prediction, WorldForgeError> {
        // Validate confidence is in range
        if response.confidence < 0.0 || response.confidence > 1.0 {
            return Err(WorldForgeError::InvalidResponse {
                provider: "unknown".into(),
                message: format!(
                    "Confidence {} out of range [0, 1]",
                    response.confidence
                ),
            });
        }

        // Validate frame count if expected
        if let Some(expected) = self.expected_frame_count {
            if response.frames.len() != expected {
                return Err(WorldForgeError::InvalidResponse {
                    provider: "unknown".into(),
                    message: format!(
                        "Expected {} frames, got {}",
                        expected,
                        response.frames.len()
                    ),
                });
            }
        }

        // Validate frame resolution
        if let Some((w, h)) = self.expected_resolution {
            for (i, frame) in response.frames.iter().enumerate() {
                if frame.width != w || frame.height != h {
                    return Err(WorldForgeError::InvalidResponse {
                        provider: "unknown".into(),
                        message: format!(
                            "Frame {} has resolution {}x{}, expected {}x{}",
                            i, frame.width, frame.height, w, h
                        ),
                    });
                }
            }
        }

        Ok(response.into())
    }
}
```

### 2.10 Video and Frame Handling Pipeline

Many world model providers return video (MP4, WebM) rather than individual
frames. WorldForge provides a standard pipeline for handling this.

#### 2.10.1 Pipeline Overview

```
Provider API Response
        │
        ▼
┌───────────────┐
│  Download      │  ← Streaming download with progress
│  Video Bytes   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Decode Video  │  ← FFmpeg or pure-Rust decoder
│  to Frames     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Validate &    │  ← Resolution, frame count, format check
│  Normalize     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Convert to    │  ← WorldForge Frame type (RGB, f32)
│  WorldForge    │
│  Frame Type    │
└───────────────┘
```

#### 2.10.2 Frame Type

```rust
#[derive(Debug, Clone)]
pub struct Frame {
    /// Raw pixel data in RGB format, row-major
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Pixel format
    pub format: PixelFormat,
    /// Frame index in sequence
    pub index: usize,
    /// Timestamp in milliseconds from start
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Rgb32F,
    Grayscale8,
    Grayscale32F,
}
```

#### 2.10.3 Video Decoder

```rust
pub struct VideoDecoder {
    max_frames: Option<usize>,
    target_resolution: Option<(u32, u32)>,
    target_format: PixelFormat,
    target_fps: Option<f64>,
}

impl VideoDecoder {
    pub async fn decode(
        &self,
        video_bytes: &[u8],
    ) -> Result<Vec<Frame>, WorldForgeError> {
        // Write bytes to temp file
        let temp_path = write_temp_video(video_bytes).await?;

        // Use ffmpeg to extract frames
        let frames = self.extract_frames(&temp_path).await?;

        // Clean up
        tokio::fs::remove_file(&temp_path).await.ok();

        Ok(frames)
    }

    async fn extract_frames(
        &self,
        path: &Path,
    ) -> Result<Vec<Frame>, WorldForgeError> {
        let mut cmd = tokio::process::Command::new("ffmpeg");
        cmd.arg("-i").arg(path)
           .arg("-f").arg("rawvideo")
           .arg("-pix_fmt").arg("rgb24");

        if let Some(fps) = self.target_fps {
            cmd.arg("-vf").arg(format!("fps={fps}"));
        }

        if let Some((w, h)) = self.target_resolution {
            cmd.arg("-s").arg(format!("{w}x{h}"));
        }

        cmd.arg("-");

        let output = cmd.output().await.map_err(|e| {
            WorldForgeError::VideoProcessing {
                message: format!("FFmpeg execution failed: {e}"),
            }
        })?;

        if !output.status.success() {
            return Err(WorldForgeError::VideoProcessing {
                message: format!(
                    "FFmpeg error: {}",
                    String::from_utf8_lossy(&output.stderr)
                ),
            });
        }

        self.raw_bytes_to_frames(&output.stdout)
    }
}
```

#### 2.10.4 Frame Encoder (for uploading)

```rust
pub struct VideoEncoder {
    codec: VideoCodec,
    quality: u32,
    fps: f64,
}

pub enum VideoCodec {
    H264,
    H265,
    VP9,
    AV1,
}

impl VideoEncoder {
    pub async fn encode(
        &self,
        frames: &[Frame],
    ) -> Result<Vec<u8>, WorldForgeError> {
        let (width, height) = if let Some(f) = frames.first() {
            (f.width, f.height)
        } else {
            return Err(WorldForgeError::Validation {
                message: "No frames to encode".into(),
            });
        };

        let mut cmd = tokio::process::Command::new("ffmpeg");
        cmd.arg("-f").arg("rawvideo")
           .arg("-pix_fmt").arg("rgb24")
           .arg("-s").arg(format!("{width}x{height}"))
           .arg("-r").arg(self.fps.to_string())
           .arg("-i").arg("-")
           .arg("-c:v").arg(self.codec.ffmpeg_name())
           .arg("-crf").arg(self.quality.to_string())
           .arg("-f").arg("mp4")
           .arg("-movflags").arg("frag_keyframe+empty_moov")
           .arg("-");

        // ... pipe frame data to stdin, collect output
        todo!()
    }
}
```

### 2.11 Cost Tracking

WorldForge tracks costs across all providers to enable budget management.

```rust
pub struct CostTracker {
    /// Running total cost in USD
    total_cost: Arc<AtomicU64>,  // Stored as microdollars (1 USD = 1_000_000)
    /// Per-provider costs
    provider_costs: Arc<DashMap<String, u64>>,
    /// Per-operation costs
    operation_costs: Arc<DashMap<OperationType, u64>>,
    /// Cost log for auditing
    cost_log: Arc<Mutex<Vec<CostEntry>>>,
    /// Budget limit in USD (None = unlimited)
    budget_limit: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CostEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub provider: String,
    pub operation: OperationType,
    pub cost_usd: f64,
    pub request_id: String,
    pub metadata: HashMap<String, Value>,
}

impl CostTracker {
    pub fn record(
        &self,
        provider: &str,
        operation: OperationType,
        cost_usd: f64,
        request_id: &str,
    ) -> Result<(), WorldForgeError> {
        // Check budget
        if let Some(limit) = self.budget_limit {
            let current = self.total_cost.load(Ordering::Relaxed) as f64 / 1_000_000.0;
            if current + cost_usd > limit {
                return Err(WorldForgeError::BudgetExceeded {
                    estimated: cost_usd,
                    budget: limit - current,
                });
            }
        }

        let microdollars = (cost_usd * 1_000_000.0) as u64;
        self.total_cost.fetch_add(microdollars, Ordering::Relaxed);
        self.provider_costs
            .entry(provider.to_string())
            .and_modify(|v| *v += microdollars)
            .or_insert(microdollars);
        self.operation_costs
            .entry(operation)
            .and_modify(|v| *v += microdollars)
            .or_insert(microdollars);

        Ok(())
    }

    pub fn total_cost_usd(&self) -> f64 {
        self.total_cost.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    pub fn provider_cost_usd(&self, provider: &str) -> f64 {
        self.provider_costs
            .get(provider)
            .map(|v| *v as f64 / 1_000_000.0)
            .unwrap_or(0.0)
    }
}
```

Providers MUST call `CostTracker::record` after every successful API call.
The cost tracker is injected via the provider configuration.

### 2.12 Provider Health Monitoring

WorldForge runs continuous health checks on all registered providers.

```rust
pub struct HealthMonitor {
    providers: Vec<Arc<dyn WorldModelProvider>>,
    check_interval: Duration,
    health_history: Arc<DashMap<String, VecDeque<HealthRecord>>>,
    alert_callback: Option<Box<dyn Fn(&str, &HealthAlert) + Send + Sync>>,
}

#[derive(Debug, Clone)]
pub struct HealthRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub status: HealthStatus,
    pub check_duration: Duration,
}

#[derive(Debug, Clone)]
pub enum HealthAlert {
    ProviderDown { provider: String, since: chrono::DateTime<chrono::Utc> },
    HighLatency { provider: String, latency_ms: u64, threshold_ms: u64 },
    QuotaLow { provider: String, remaining: u64, threshold: u64 },
    ErrorRateHigh { provider: String, rate: f64, window: Duration },
}

impl HealthMonitor {
    pub async fn start(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.check_interval);
            loop {
                interval.tick().await;
                for provider in &self.providers {
                    let name = provider.name().to_string();
                    let start = Instant::now();
                    let result = tokio::time::timeout(
                        Duration::from_secs(10),
                        provider.health_check(),
                    ).await;

                    let record = match result {
                        Ok(Ok(status)) => HealthRecord {
                            timestamp: chrono::Utc::now(),
                            status,
                            check_duration: start.elapsed(),
                        },
                        Ok(Err(e)) => HealthRecord {
                            timestamp: chrono::Utc::now(),
                            status: HealthStatus {
                                healthy: false,
                                latency_ms: start.elapsed().as_millis() as u64,
                                api_version: String::new(),
                                quota_remaining: None,
                                message: Some(e.to_string()),
                            },
                            check_duration: start.elapsed(),
                        },
                        Err(_) => HealthRecord {
                            timestamp: chrono::Utc::now(),
                            status: HealthStatus {
                                healthy: false,
                                latency_ms: 10000,
                                api_version: String::new(),
                                quota_remaining: None,
                                message: Some("Health check timed out".into()),
                            },
                            check_duration: start.elapsed(),
                        },
                    };

                    // Check for alerts
                    self.evaluate_alerts(&name, &record);

                    // Store history (keep last 100)
                    self.health_history
                        .entry(name)
                        .or_insert_with(VecDeque::new)
                        .push_back(record);

                    // Trim to 100 entries
                    if let Some(mut history) = self.health_history.get_mut(&name) {
                        while history.len() > 100 {
                            history.pop_front();
                        }
                    }
                }
            }
        })
    }
}
```

---

## Step-by-Step Provider Implementation Guide

### Step 1: Create the Provider Module

```
src/providers/
├── mod.rs
├── registry.rs
├── nvidia_cosmos.rs   ← Your new provider
├── mock.rs
└── ...
```

Add to `src/providers/mod.rs`:
```rust
pub mod nvidia_cosmos;
```

### Step 2: Define the Provider Struct

```rust
pub struct NvidiaCosmosProvider {
    http: ProviderHttpClient,
    config: CosmosConfig,
    cost_tracker: Arc<CostTracker>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CosmosConfig {
    pub api_key: SecretString,
    pub base_url: String,
    pub model: String,
    pub max_frames: usize,
    pub default_resolution: (u32, u32),
}
```

### Step 3: Implement the Constructor

```rust
impl NvidiaCosmosProvider {
    pub async fn new(config: CosmosConfig) -> Result<Self, WorldForgeError> {
        let auth = Box::new(ApiKeyAuth {
            header_name: "Authorization".to_string(),
            key: config.api_key.clone(),
        });

        let http = ProviderHttpClient::new(ProviderHttpConfig {
            base_url: config.base_url.clone(),
            auth,
            rate_limiter: RateLimiter::new(10.0, 10),
            retry_policy: RetryPolicy::default(),
            provider_name: "nvidia-cosmos".to_string(),
            request_timeout: Duration::from_secs(120),
            connect_timeout: Duration::from_secs(10),
            max_idle_connections: 10,
        })?;

        let provider = Self {
            http,
            config,
            cost_tracker: Arc::new(CostTracker::new(None)),
        };

        // Validate connectivity
        provider.health_check().await?;

        Ok(provider)
    }
}
```

### Step 4: Implement the Trait

Implement all 8 operations plus the 3 metadata methods. For operations
your provider does not support, return `UnsupportedOperation`.

### Step 5: Register the Provider

```rust
// In your application setup
let cosmos = NvidiaCosmosProvider::new(config).await?;
registry.register(Arc::new(cosmos));
```

### Step 6: Write Integration Tests

See the Integration Test Contract section below.

### Step 7: Add Configuration

Add provider config to `worldforge.toml`:
```toml
[providers.nvidia-cosmos]
enabled = true
api_key_env = "WORLDFORGE_NVIDIA_COSMOS_API_KEY"
base_url = "https://api.nvcf.nvidia.com/v2"
model = "cosmos-1.0-world-foundation"
max_frames = 120
default_resolution = [1280, 720]
```

### Step 8: Documentation

- Add provider to the README provider matrix.
- Document any provider-specific options.
- Add example code to `examples/`.

---

## Integration Test Contract

Before a provider can be merged, it MUST pass ALL of the following tests.

### Tier 1: Mandatory (ALL providers)

```rust
#[tokio::test]
async fn test_provider_name_is_valid() {
    let provider = create_test_provider().await;
    let name = provider.name();
    assert!(!name.is_empty());
    assert!(name.chars().all(|c| c.is_ascii_lowercase() || c == '-'));
}

#[tokio::test]
async fn test_provider_capabilities_are_consistent() {
    let provider = create_test_provider().await;
    let caps = provider.capabilities();
    // At least one capability must be true
    assert!(
        caps.predict || caps.plan || caps.embed || caps.reason
        || caps.generate || caps.transfer
    );
}

#[tokio::test]
async fn test_health_check_succeeds() {
    let provider = create_test_provider().await;
    let health = provider.health_check().await.unwrap();
    assert!(health.healthy);
    assert!(health.latency_ms < 10_000);
}

#[tokio::test]
async fn test_unsupported_operations_return_error() {
    let provider = create_test_provider().await;
    let caps = provider.capabilities();
    if !caps.predict {
        let result = provider.predict(&WorldState::default(), &Action::default(), PredictOptions::default()).await;
        assert!(matches!(result, Err(WorldForgeError::UnsupportedOperation { .. })));
    }
    // ... repeat for all operations
}

#[tokio::test]
async fn test_cost_estimate_is_fast() {
    let provider = create_test_provider().await;
    let start = Instant::now();
    let _ = provider.cost_estimate(&CostEstimateRequest::default()).await;
    assert!(start.elapsed() < Duration::from_secs(1));
}
```

### Tier 2: Per-Capability Tests

For each capability the provider declares as `true`:

```rust
// If predict == true
#[tokio::test]
async fn test_predict_returns_valid_prediction() {
    let provider = create_test_provider().await;
    let state = test_world_state();
    let action = test_action();
    let prediction = provider.predict(&state, &action, PredictOptions::default()).await.unwrap();
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    assert!(!prediction.next_state.frames.is_empty());
}

// If embed == true
#[tokio::test]
async fn test_embed_returns_valid_vector() {
    let provider = create_test_provider().await;
    let state = test_world_state();
    let embedding = provider.embed(&state, EmbedOptions::default()).await.unwrap();
    assert_eq!(embedding.vector.len(), embedding.dimension);
    assert!(embedding.dimension > 0);
}

// If embed == true
#[tokio::test]
async fn test_embed_is_deterministic() {
    let provider = create_test_provider().await;
    let state = test_world_state();
    let e1 = provider.embed(&state, EmbedOptions::default()).await.unwrap();
    let e2 = provider.embed(&state, EmbedOptions::default()).await.unwrap();
    assert_eq!(e1.vector, e2.vector);
}

// If reason == true
#[tokio::test]
async fn test_reason_returns_non_empty_answer() {
    let provider = create_test_provider().await;
    let state = test_world_state();
    let reasoning = provider.reason(&state, "What objects are present?", ReasonOptions::default()).await.unwrap();
    assert!(!reasoning.answer.is_empty());
    assert!(reasoning.confidence >= 0.0 && reasoning.confidence <= 1.0);
}

// If generate == true
#[tokio::test]
async fn test_generate_returns_frames() {
    let provider = create_test_provider().await;
    let result = provider.generate("A sunny meadow with a river", GenerateOptions {
        num_frames: Some(16),
        resolution: Some((640, 480)),
        ..Default::default()
    }).await.unwrap();
    assert_eq!(result.frames.len(), 16);
}
```

### Tier 3: Error Handling Tests

```rust
#[tokio::test]
async fn test_invalid_auth_returns_auth_error() {
    let provider = create_provider_with_bad_auth().await;
    let result = provider.health_check().await;
    assert!(matches!(result, Err(WorldForgeError::Auth { .. })));
}

#[tokio::test]
async fn test_timeout_returns_timeout_error() {
    // Use a mock server with artificial delay
    let provider = create_provider_with_slow_server().await;
    let result = provider.predict(&test_state(), &test_action(), PredictOptions {
        timeout: Some(Duration::from_millis(100)),
        ..Default::default()
    }).await;
    assert!(matches!(result, Err(WorldForgeError::Timeout { .. })));
}
```

---

## Mock and Replay Testing Strategy

### Mock Provider

WorldForge ships a `MockProvider` for unit testing user code:

```rust
pub struct MockProvider {
    name: String,
    capabilities: ProviderCapabilities,
    predict_responses: Arc<Mutex<VecDeque<Result<Prediction, WorldForgeError>>>>,
    plan_responses: Arc<Mutex<VecDeque<Result<Plan, WorldForgeError>>>>,
    embed_responses: Arc<Mutex<VecDeque<Result<Embedding, WorldForgeError>>>>,
    // ... for each operation
    call_log: Arc<Mutex<Vec<MockCall>>>,
}

impl MockProvider {
    pub fn new(name: &str) -> Self { ... }

    pub fn expect_predict(&self, response: Result<Prediction, WorldForgeError>) {
        self.predict_responses.lock().unwrap().push_back(response);
    }

    pub fn calls(&self) -> Vec<MockCall> {
        self.call_log.lock().unwrap().clone()
    }

    pub fn assert_called(&self, operation: &str, times: usize) {
        let count = self.calls().iter().filter(|c| c.operation == operation).count();
        assert_eq!(count, times, "Expected {operation} to be called {times} times, got {count}");
    }
}
```

### Replay Testing

For integration tests that should not hit real APIs, WorldForge supports
HTTP replay via recorded fixtures:

```rust
pub struct ReplayProvider {
    inner: Box<dyn WorldModelProvider>,
    recordings_dir: PathBuf,
    mode: ReplayMode,
}

pub enum ReplayMode {
    /// Record real API calls to disk
    Record,
    /// Replay from recorded fixtures
    Replay,
    /// Try replay first, fall back to real API
    Hybrid,
}
```

Recording format (JSON):
```json
{
    "request": {
        "method": "POST",
        "url": "https://api.nvcf.nvidia.com/v2/predict",
        "headers": { "Content-Type": "application/json" },
        "body": { "...": "..." }
    },
    "response": {
        "status": 200,
        "headers": { "Content-Type": "application/json" },
        "body": { "...": "..." }
    },
    "recorded_at": "2026-04-02T19:00:00Z",
    "latency_ms": 1250
}
```

### Testing Pyramid

```
               ╱╲
              ╱  ╲
             ╱ E2E╲          ← Full provider round-trip (CI nightly)
            ╱──────╲
           ╱        ╲
          ╱Integration╲      ← Replay fixtures (CI on every PR)
         ╱────────────╲
        ╱              ╲
       ╱  Unit (Mock)   ╲    ← MockProvider (local, fast)
      ╱──────────────────╲
```

---

## Implementation Plan

| Phase | Task                                    | Duration | Owner |
|-------|-----------------------------------------|----------|-------|
| 1     | Finalize WorldModelProvider trait        | 1 week   | Core  |
| 2     | Implement ProviderHttpClient             | 1 week   | Core  |
| 3     | Implement auth patterns                  | 3 days   | Core  |
| 4     | Implement rate limiter + retry           | 3 days   | Core  |
| 5     | Implement video pipeline                 | 1 week   | Core  |
| 6     | Implement MockProvider                   | 3 days   | Core  |
| 7     | Implement CostTracker                    | 2 days   | Core  |
| 8     | Implement HealthMonitor                  | 2 days   | Core  |
| 9     | Write integration test harness           | 3 days   | Core  |
| 10    | First provider (NVIDIA Cosmos)           | 2 weeks  | Core  |
| 11    | Provider documentation template          | 2 days   | Docs  |

---

## Testing Strategy

### Unit Tests
- Every public method on ProviderHttpClient has unit tests with a mock HTTP server (wiremock).
- RateLimiter tested for correctness under concurrent load.
- RetryPolicy tested with deterministic delays.
- CostTracker tested for thread safety and budget enforcement.
- Error mapping tested for every HTTP status code.

### Integration Tests
- Each provider runs the full Integration Test Contract.
- Replay fixtures are stored in `tests/fixtures/{provider}/`.
- CI runs replay tests on every PR.
- Nightly CI runs live API tests (requires secrets).

### Property Tests
- Embedding determinism: same input always produces same output.
- Cost estimate: never negative, scales with input size.
- Plan steps: always in order, always reachable from previous step.

### Load Tests
- Rate limiter correctness under 1000 concurrent requests.
- Provider graceful degradation under rate limiting.

---

## Open Questions

1. **Should providers support batch operations?** Some APIs are more efficient
   with batch requests. We may need `predict_batch`, `embed_batch`, etc.

2. **Should the trait support streaming predictions?** Some providers can
   stream frames as they are generated rather than waiting for the full
   result.

3. **How should we handle provider versioning?** If a provider updates their
   API, should we support multiple versions simultaneously?

4. **Should cost tracking be opt-in or always-on?** Always-on has overhead
   but ensures budget safety.

5. **Should we support local/self-hosted providers?** This changes auth
   patterns significantly (no API key needed, but need connection info).

6. **What is the maximum video size we should support?** Need to set
   reasonable limits to avoid OOM.

7. **Should providers declare their latency characteristics?** This would
   help the router make better decisions about which provider to use.

8. **How should we handle multi-modal inputs?** Some providers accept
   text + image + video. The current WorldState may need extension.

---

## Appendix A: Provider Checklist

Before submitting a provider PR, verify:

- [ ] Implements `WorldModelProvider` trait completely
- [ ] `name()` returns a valid, unique identifier
- [ ] `capabilities()` accurately reflects supported operations
- [ ] Unsupported operations return `UnsupportedOperation`
- [ ] Uses `ProviderHttpClient` for HTTP communication
- [ ] Uses `SecretString` for all credentials
- [ ] Rate limiting is configured correctly
- [ ] Retry policy is appropriate for the provider's API
- [ ] All errors are mapped to `WorldForgeError`
- [ ] Response validation is implemented
- [ ] Cost tracking is wired up
- [ ] Health check works correctly
- [ ] All Tier 1 integration tests pass
- [ ] All Tier 2 tests pass for declared capabilities
- [ ] All Tier 3 error handling tests pass
- [ ] Replay fixtures are recorded and committed
- [ ] Configuration documented in `worldforge.toml` example
- [ ] Provider added to README provider matrix
- [ ] Example code in `examples/`

---

## Appendix B: Error Code Reference

| Error Variant         | HTTP Status | Retryable | Description                         |
|-----------------------|-------------|-----------|-------------------------------------|
| UnsupportedOperation  | N/A         | No        | Provider doesn't support operation  |
| Auth                  | 401, 403    | No        | Authentication/authorization failed |
| RateLimited           | 429         | Yes       | Too many requests                   |
| Timeout               | 408         | Yes       | Request timed out                   |
| InvalidResponse       | N/A         | No        | Response failed validation          |
| ProviderError         | 500-599     | Yes       | Server-side error                   |
| Network               | N/A         | Yes       | Connection/DNS/TLS failure          |
| VideoProcessing       | N/A         | No        | FFmpeg or frame processing failed   |
| Validation            | 400, 422    | No        | Invalid request parameters          |
| BudgetExceeded        | N/A         | No        | Cost would exceed budget limit      |
| Internal              | N/A         | No        | Bug in WorldForge itself            |

---

*End of RFC-0001*
