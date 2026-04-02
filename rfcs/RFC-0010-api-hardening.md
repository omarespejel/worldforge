# RFC-0010: REST API Production Hardening

| Field   | Value                              |
|---------|------------------------------------|
| Title   | REST API Production Hardening      |
| Status  | Draft                              |
| Author  | WorldForge Core Team               |
| Created | 2026-04-02                         |
| Updated | 2026-04-02                         |

---

## Abstract

This RFC proposes a comprehensive overhaul of the WorldForge REST API server,
migrating from the current hand-rolled tokio TCP server with manual HTTP parsing
(crates/worldforge-server/src/lib.rs, ~9,588 lines) to the axum web framework
with tower middleware. The migration introduces OpenAPI 3.1 specification
generation, request validation, pagination, WebSocket streaming, CORS,
compression, connection pooling, graceful shutdown, API versioning, and
establishes load testing targets for production readiness.

---

## Motivation

The current WorldForge server implementation suffers from several critical
limitations that prevent production deployment:

1. **Manual HTTP Parsing**: The server hand-parses HTTP requests from raw TCP
   streams. This is fragile, incomplete (missing chunked transfer encoding,
   multipart handling, HTTP/2), and a security liability. Every HTTP edge case
   is a potential bug or exploit.

2. **No Framework Middleware**: There is no middleware stack. Cross-cutting
   concerns like logging, authentication, CORS, compression, and rate limiting
   must each be implemented from scratch in the request handling code. This
   leads to code duplication and inconsistency across the 27+ routes.

3. **No API Specification**: Without an OpenAPI spec, client SDK generation is
   manual, documentation drifts from implementation, and integration testing
   requires reading source code to understand request/response shapes.

4. **No Request Validation**: Input payloads are deserialized but not validated.
   Missing fields produce cryptic internal errors rather than 422 responses with
   actionable messages.

5. **No Pagination**: List endpoints (providers, models, capabilities) return
   unbounded result sets. As the provider ecosystem grows, these responses will
   become unmanageable.

6. **No Streaming Support**: World model predictions can take 10-30 seconds.
   Clients must poll or wait with no progress indication. WebSocket support
   would enable streaming partial results and progress updates.

7. **No CORS**: Browser-based clients cannot call the API due to missing CORS
   headers, blocking dashboard and web UI development.

8. **No Compression**: Large prediction responses (especially those containing
   generated images or point clouds) are sent uncompressed, wasting bandwidth.

9. **No Graceful Shutdown**: The server terminates immediately on SIGTERM,
   dropping in-flight requests. This makes rolling deployments unreliable.

10. **No Connection Pooling**: Each outbound provider API call creates a new
    HTTP connection. Connection reuse would significantly reduce latency.

These issues collectively make the current server unsuitable for any deployment
beyond local development. This RFC addresses all of them systematically.

---

## Detailed Design

### 1. Framework Migration: axum + tower

#### 1.1 Why axum

axum is chosen over alternatives (actix-web, warp, rocket) for the following
reasons:

- Built on top of tokio and hyper, which the project already depends on
- Uses tower middleware ecosystem, giving access to dozens of production-tested
  middleware components
- Extractor-based request handling aligns with our typed request/response model
- First-class WebSocket support via axum::extract::ws
- Maintained by the tokio team, ensuring long-term compatibility
- No macros required; purely function-based handlers

#### 1.2 Router Structure

The 27+ existing routes will be organized into nested routers by domain:

```rust
use axum::{Router, routing::{get, post, put, delete}};

fn app() -> Router<AppState> {
    Router::new()
        .nest("/v1", api_v1_router())
        .layer(middleware_stack())
        .with_state(app_state)
}

fn api_v1_router() -> Router<AppState> {
    Router::new()
        .nest("/providers", providers_router())
        .nest("/models", models_router())
        .nest("/predictions", predictions_router())
        .nest("/capabilities", capabilities_router())
        .nest("/health", health_router())
        .nest("/guardrails", guardrails_router())
        .nest("/planning", planning_router())
}

fn providers_router() -> Router<AppState> {
    Router::new()
        .route("/", get(list_providers))
        .route("/:id", get(get_provider))
        .route("/:id/models", get(list_provider_models))
        .route("/:id/status", get(get_provider_status))
}

fn predictions_router() -> Router<AppState> {
    Router::new()
        .route("/", post(create_prediction))
        .route("/:id", get(get_prediction))
        .route("/:id/cancel", post(cancel_prediction))
        .route("/stream", get(stream_prediction_ws))
}
```

#### 1.3 Application State

Shared state will be managed through axum's State extractor:

```rust
#[derive(Clone)]
struct AppState {
    provider_registry: Arc<ProviderRegistry>,
    prediction_engine: Arc<PredictionEngine>,
    guardrail_runner: Arc<GuardrailRunner>,
    http_client_pool: reqwest::Client,
    config: Arc<ServerConfig>,
    shutdown_signal: Arc<tokio::sync::watch::Sender<bool>>,
}
```

#### 1.4 Handler Migration Strategy

Each existing route handler in the monolithic lib.rs will be extracted into
its own module. The migration will proceed route-by-route, with each handler
converted from raw byte manipulation to axum extractors:

Before (current):
```rust
// Manually parsing request body from raw TCP bytes
let body = read_body(&mut stream, content_length).await?;
let request: PredictionRequest = serde_json::from_slice(&body)
    .map_err(|e| write_400(&mut stream, &e.to_string()))?;
```

After (axum):
```rust
async fn create_prediction(
    State(state): State<AppState>,
    ValidatedJson(request): ValidatedJson<PredictionRequest>,
) -> Result<Json<PredictionResponse>, ApiError> {
    let response = state.prediction_engine.predict(request).await?;
    Ok(Json(response))
}
```

### 2. OpenAPI 3.1 Specification Generation

#### 2.1 utoipa Integration

We will use the utoipa crate to generate OpenAPI 3.1 specifications directly
from Rust types and handler annotations:

```rust
use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "WorldForge API",
        version = "1.0.0",
        description = "Unified interface for world model providers",
        license(name = "MIT OR Apache-2.0"),
    ),
    paths(
        providers::list_providers,
        providers::get_provider,
        predictions::create_prediction,
        predictions::get_prediction,
        // ... all routes
    ),
    components(schemas(
        PredictionRequest,
        PredictionResponse,
        ProviderInfo,
        ModelCapabilities,
        GuardrailConfig,
        // ... all types
    )),
    tags(
        (name = "providers", description = "Provider management"),
        (name = "predictions", description = "World model predictions"),
        (name = "guardrails", description = "Safety guardrails"),
        (name = "planning", description = "Planning pipelines"),
    )
)]
struct ApiDoc;
```

#### 2.2 Spec Serving

The OpenAPI JSON spec will be served at `/v1/openapi.json` and a Swagger UI
will be available at `/v1/docs` using utoipa-swagger-ui:

```rust
use utoipa_swagger_ui::SwaggerUi;

fn app() -> Router<AppState> {
    Router::new()
        .merge(SwaggerUi::new("/v1/docs").url("/v1/openapi.json", ApiDoc::openapi()))
        .nest("/v1", api_v1_router())
}
```

#### 2.3 Schema Derivation

All request and response types will derive `utoipa::ToSchema`:

```rust
#[derive(Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "provider": "cosmos",
    "model": "cosmos-7b",
    "input": { "image": "base64...", "action": "move_forward" }
}))]
pub struct PredictionRequest {
    /// Provider identifier (e.g., "cosmos", "genie2", "wayve")
    #[schema(example = "cosmos")]
    pub provider: String,

    /// Model identifier within the provider
    #[schema(example = "cosmos-7b")]
    pub model: String,

    /// Input data for the prediction
    pub input: PredictionInput,

    /// Optional guardrail configuration
    #[schema(nullable)]
    pub guardrails: Option<GuardrailConfig>,
}
```

### 3. Request Validation

#### 3.1 Validation Framework

We will use the validator crate integrated with a custom axum extractor:

```rust
use validator::Validate;

#[derive(Deserialize, Validate, ToSchema)]
pub struct PredictionRequest {
    #[validate(length(min = 1, max = 64))]
    pub provider: String,

    #[validate(length(min = 1, max = 128))]
    pub model: String,

    #[validate]
    pub input: PredictionInput,

    #[validate(range(min = 1, max = 100))]
    pub max_frames: Option<u32>,
}
```

#### 3.2 Custom ValidatedJson Extractor

```rust
pub struct ValidatedJson<T>(pub T);

#[async_trait]
impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(req, state)
            .await
            .map_err(|e| ApiError::InvalidJson(e.to_string()))?;

        value.validate()
            .map_err(|e| ApiError::ValidationError(e))?;

        Ok(ValidatedJson(value))
    }
}
```

#### 3.3 Error Response Format

All errors will follow a consistent JSON format:

```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Request validation failed",
        "details": [
            {
                "field": "provider",
                "message": "must not be empty",
                "code": "length"
            }
        ]
    },
    "request_id": "req_abc123"
}
```

### 4. Pagination for List Endpoints

#### 4.1 Cursor-Based Pagination

We will use cursor-based pagination (not offset-based) for stable iteration
over growing datasets:

```rust
#[derive(Deserialize, Validate, ToSchema)]
pub struct PaginationParams {
    /// Maximum number of items to return (1-100, default 20)
    #[validate(range(min = 1, max = 100))]
    #[serde(default = "default_limit")]
    pub limit: u32,

    /// Cursor for the next page (opaque string from previous response)
    pub cursor: Option<String>,

    /// Sort field
    #[serde(default)]
    pub sort_by: SortField,

    /// Sort direction
    #[serde(default)]
    pub sort_order: SortOrder,
}

#[derive(Serialize, ToSchema)]
pub struct PaginatedResponse<T: Serialize + ToSchema> {
    pub data: Vec<T>,
    pub pagination: PaginationInfo,
}

#[derive(Serialize, ToSchema)]
pub struct PaginationInfo {
    pub total: u64,
    pub limit: u32,
    pub has_more: bool,
    pub next_cursor: Option<String>,
    pub previous_cursor: Option<String>,
}
```

#### 4.2 Affected Endpoints

The following endpoints will gain pagination support:

- `GET /v1/providers` — List all registered providers
- `GET /v1/providers/:id/models` — List models for a provider
- `GET /v1/capabilities` — List all capabilities across providers
- `GET /v1/predictions` — List prediction history
- `GET /v1/guardrails` — List available guardrail checks

### 5. WebSocket Support for Streaming Predictions

#### 5.1 WebSocket Endpoint

```rust
async fn stream_prediction_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_prediction_stream(socket, state))
}

async fn handle_prediction_stream(
    mut socket: WebSocket,
    state: AppState,
) {
    // Receive prediction request
    let msg = socket.recv().await;
    let request: StreamPredictionRequest = match msg {
        Some(Ok(Message::Text(text))) => serde_json::from_str(&text).unwrap(),
        _ => return,
    };

    // Create prediction channel
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    // Spawn prediction task
    let engine = state.prediction_engine.clone();
    tokio::spawn(async move {
        engine.predict_streaming(request, tx).await;
    });

    // Stream results back to client
    while let Some(update) = rx.recv().await {
        let msg = Message::Text(serde_json::to_string(&update).unwrap());
        if socket.send(msg).await.is_err() {
            break;
        }
    }

    // Send completion message
    let _ = socket.send(Message::Text(
        serde_json::to_string(&StreamUpdate::Complete).unwrap()
    )).await;
}
```

#### 5.2 Stream Message Protocol

```rust
#[derive(Serialize, ToSchema)]
#[serde(tag = "type")]
enum StreamUpdate {
    #[serde(rename = "progress")]
    Progress {
        percent: f32,
        stage: String,
        message: String,
    },
    #[serde(rename = "partial_result")]
    PartialResult {
        frame_index: u32,
        data: PredictionOutput,
    },
    #[serde(rename = "guardrail_check")]
    GuardrailCheck {
        check_name: String,
        passed: bool,
        details: String,
    },
    #[serde(rename = "complete")]
    Complete {
        prediction_id: String,
        total_frames: u32,
        elapsed_ms: u64,
    },
    #[serde(rename = "error")]
    Error {
        code: String,
        message: String,
    },
}
```

#### 5.3 Connection Management

WebSocket connections will be managed with:

- Ping/pong heartbeat every 30 seconds
- Idle timeout of 5 minutes
- Maximum message size of 16 MB
- Per-connection backpressure via bounded channel (32 messages)
- Graceful close on server shutdown (close frame with reason)

### 6. CORS Configuration

```rust
use tower_http::cors::{CorsLayer, Any};

fn cors_layer(config: &ServerConfig) -> CorsLayer {
    if config.cors_permissive {
        // Development mode
        CorsLayer::permissive()
    } else {
        // Production mode
        CorsLayer::new()
            .allow_origin(
                config.allowed_origins.iter()
                    .map(|o| o.parse::<HeaderValue>().unwrap())
                    .collect::<Vec<_>>()
            )
            .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
            .allow_headers([
                header::CONTENT_TYPE,
                header::AUTHORIZATION,
                header::ACCEPT,
                HeaderName::from_static("x-request-id"),
            ])
            .expose_headers([
                HeaderName::from_static("x-request-id"),
                HeaderName::from_static("x-ratelimit-remaining"),
            ])
            .max_age(Duration::from_secs(86400))
    }
}
```

### 7. Request/Response Compression

```rust
use tower_http::compression::CompressionLayer;
use tower_http::decompression::RequestDecompressionLayer;

fn compression_layer() -> CompressionLayer {
    CompressionLayer::new()
        .br(true)      // Brotli for text responses
        .gzip(true)    // Gzip fallback
        .zstd(true)    // Zstandard for large payloads
        .quality(tower_http::compression::CompressionLevel::Default)
}
```

Compression will be applied selectively:
- Always compress JSON responses > 1 KB
- Always compress OpenAPI spec responses
- Never compress already-compressed binary data (images, video)
- Support Accept-Encoding negotiation

### 8. Connection Pooling

#### 8.1 Outbound HTTP Client Pool

A shared reqwest::Client with connection pooling for provider API calls:

```rust
fn build_http_client(config: &ServerConfig) -> reqwest::Client {
    reqwest::Client::builder()
        .pool_max_idle_per_host(config.pool_max_idle_per_host.unwrap_or(10))
        .pool_idle_timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(120))
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(format!("worldforge-server/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .expect("Failed to build HTTP client")
}
```

#### 8.2 Connection Limits

- Maximum 100 connections per provider host
- Maximum 500 total outbound connections
- Idle connection timeout: 90 seconds
- TCP keepalive: 60 seconds
- Connect timeout: 10 seconds
- Request timeout: 120 seconds (predictions can be slow)

### 9. Graceful Shutdown

```rust
async fn run_server(config: ServerConfig) -> Result<()> {
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let app = build_app(config.clone(), shutdown_tx.clone());

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;

    tracing::info!("WorldForge server listening on {}", config.bind_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_rx))
        .await?;

    tracing::info!("Server shutdown complete");
    Ok(())
}

async fn shutdown_signal(mut rx: tokio::sync::watch::Receiver<bool>) {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("install Ctrl+C handler");
    };

    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received Ctrl+C"),
        _ = terminate => tracing::info!("Received SIGTERM"),
        _ = rx.changed() => tracing::info!("Received programmatic shutdown"),
    }
}
```

Graceful shutdown behavior:
- Stop accepting new connections immediately
- Wait up to 30 seconds for in-flight requests to complete
- Send WebSocket close frames to all connected clients
- Flush any pending metrics/logs
- Close outbound connection pool
- Exit with code 0

### 10. Load Testing Targets

#### 10.1 Performance Requirements

| Endpoint Category | Target RPS | p50 Latency | p99 Latency | Error Rate |
|-------------------|-----------|-------------|-------------|------------|
| Health check      | 5,000     | < 1 ms      | < 5 ms      | < 0.01%    |
| Provider metadata | 1,000     | < 10 ms     | < 50 ms     | < 0.1%     |
| Model listing     | 1,000     | < 20 ms     | < 100 ms    | < 0.1%     |
| Prediction submit | 100       | < 100 ms    | < 500 ms    | < 0.5%     |
| Prediction result | 500       | < 15 ms     | < 75 ms     | < 0.1%     |
| WebSocket stream  | 200 conn  | N/A         | N/A         | < 0.5%     |

#### 10.2 Load Testing Tools

We will use a combination of tools:

- **k6**: Primary load testing tool for HTTP endpoints. Scripts will be
  committed to `tests/load/` directory.
- **websocat**: WebSocket load testing for streaming endpoints.
- **Custom Rust harness**: For complex multi-step scenarios (create prediction,
  poll for result, verify output).

#### 10.3 Load Test Scenarios

Scenario 1: Metadata Storm
- 50 virtual users querying provider/model metadata
- Sustained for 5 minutes
- Target: 1,000 req/s with p99 < 50 ms

Scenario 2: Prediction Pipeline
- 20 virtual users submitting predictions in sequence
- Each user: submit -> poll -> retrieve -> verify
- Target: 100 predictions/s submission rate

Scenario 3: Mixed Workload
- 70% metadata queries, 20% prediction submissions, 10% WebSocket streams
- 100 virtual users, 10-minute sustained run
- Target: No degradation in metadata latency under prediction load

Scenario 4: Spike Test
- Baseline of 200 req/s, spike to 2,000 req/s for 30 seconds
- Target: No errors, p99 < 200 ms during spike recovery

### 11. API Versioning Strategy

#### 11.1 URL Prefix Versioning

All API endpoints will be prefixed with `/v1/`:

```
GET  /v1/providers
GET  /v1/providers/:id
POST /v1/predictions
GET  /v1/predictions/:id
WS   /v1/predictions/stream
```

#### 11.2 Version Lifecycle

- **v1**: Current version, stable API surface
- **v2** (future): Breaking changes, coexists with v1 during migration period
- Deprecation policy: Minimum 6 months notice before removing a version
- Sunset header: Deprecated versions include `Sunset` HTTP header

#### 11.3 Non-Breaking Changes (No Version Bump)

The following changes are considered non-breaking and do not require a new
API version:

- Adding new optional fields to request bodies
- Adding new fields to response bodies
- Adding new endpoints
- Adding new enum values to response fields
- Relaxing validation constraints
- Adding new query parameters with defaults

#### 11.4 Breaking Changes (Require Version Bump)

- Removing or renaming fields
- Changing field types
- Tightening validation constraints
- Removing endpoints
- Changing error response format
- Changing authentication requirements

### 12. Middleware Stack

The complete tower middleware stack, applied in order (outermost first):

```rust
fn middleware_stack(config: &ServerConfig) -> ServiceBuilder<...> {
    ServiceBuilder::new()
        // 1. Request ID generation and propagation
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        // 2. Request/response logging
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().include_headers(true))
                .on_response(DefaultOnResponse::new().include_headers(true))
        )
        // 3. CORS
        .layer(cors_layer(config))
        // 4. Compression
        .layer(CompressionLayer::new())
        .layer(RequestDecompressionLayer::new())
        // 5. Request body size limit (16 MB)
        .layer(RequestBodyLimitLayer::new(16 * 1024 * 1024))
        // 6. Timeout (request-level, 120 seconds)
        .layer(TimeoutLayer::new(Duration::from_secs(120)))
        // 7. Concurrency limit
        .layer(ConcurrencyLimitLayer::new(1000))
        // 8. Metrics (Prometheus)
        .layer(PrometheusMetricsLayer::new())
}
```

### 13. Error Handling

#### 13.1 Unified Error Type

```rust
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Invalid JSON: {0}")]
    InvalidJson(String),

    #[error("Validation error")]
    ValidationError(validator::ValidationErrors),

    #[error("Provider error: {0}")]
    ProviderError(#[from] ProviderError),

    #[error("Guardrail violation: {0}")]
    GuardrailViolation(String),

    #[error("Rate limit exceeded")]
    RateLimited { retry_after: Duration },

    #[error("Internal server error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "NOT_FOUND", msg.clone()),
            ApiError::InvalidJson(msg) => (StatusCode::BAD_REQUEST, "INVALID_JSON", msg.clone()),
            ApiError::ValidationError(e) => (StatusCode::UNPROCESSABLE_ENTITY, "VALIDATION_ERROR", e.to_string()),
            ApiError::ProviderError(e) => (StatusCode::BAD_GATEWAY, "PROVIDER_ERROR", e.to_string()),
            ApiError::GuardrailViolation(msg) => (StatusCode::FORBIDDEN, "GUARDRAIL_VIOLATION", msg.clone()),
            ApiError::RateLimited { retry_after } => (StatusCode::TOO_MANY_REQUESTS, "RATE_LIMITED", format!("Retry after {} seconds", retry_after.as_secs())),
            ApiError::Internal(e) => {
                tracing::error!("Internal error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "An internal error occurred".to_string())
            }
        };

        let body = json!({
            "error": {
                "code": code,
                "message": message,
            }
        });

        (status, Json(body)).into_response()
    }
}
```

---

## Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

1. Add axum, tower, tower-http, utoipa dependencies to Cargo.toml
2. Create `crates/worldforge-server/src/router/` module structure
3. Implement AppState and shared state initialization
4. Implement middleware stack (logging, request ID, CORS, compression)
5. Migrate health check endpoint as proof-of-concept
6. Implement unified error type and error handling

### Phase 2: Route Migration (Weeks 3-4)

7. Migrate provider endpoints (list, get, status)
8. Migrate model endpoints (list, get, capabilities)
9. Migrate prediction endpoints (create, get, cancel)
10. Migrate guardrail endpoints
11. Migrate planning endpoints
12. Remove old hand-rolled HTTP handling code

### Phase 3: New Features (Weeks 5-6)

13. Implement request validation with validator crate
14. Add pagination to all list endpoints
15. Implement WebSocket streaming for predictions
16. Add OpenAPI spec generation with utoipa
17. Set up Swagger UI at /v1/docs
18. Implement graceful shutdown

### Phase 4: Production Readiness (Weeks 7-8)

19. Implement connection pooling for outbound requests
20. Write k6 load test scripts
21. Run load tests, profile and optimize hot paths
22. Write API versioning documentation
23. Update Python SDK for new API structure
24. Cut v1.0.0 release of worldforge-server

### Migration Safety

The migration will use a feature flag `new-server` to allow running both
the old and new server implementations during the transition. Integration
tests will run against both implementations to verify behavioral equivalence.

---

## Testing Strategy

### Unit Tests

- Each handler function tested in isolation with mock state
- Request validation tested with valid, invalid, and edge-case inputs
- Error mapping tested for all error variants
- Pagination cursor encoding/decoding tested
- WebSocket message serialization/deserialization tested

### Integration Tests

- Full server startup with test configuration
- Round-trip tests for all 27+ endpoints via reqwest client
- CORS preflight request/response verification
- Compression negotiation verification
- WebSocket connection lifecycle tests
- Graceful shutdown behavior verification
- Concurrent request handling tests

### Load Tests

- k6 scripts for all four load test scenarios
- Automated benchmark regression detection in CI
- Memory leak detection via long-running soak tests
- Connection exhaustion tests

### Contract Tests

- OpenAPI spec snapshot tests (detect unintended API changes)
- Response schema validation against OpenAPI spec
- Python SDK compatibility tests against new server

---

## Open Questions

1. **Rate Limiting Strategy**: Should we implement rate limiting in the server
   or defer to an API gateway (e.g., Kong, Envoy)? Server-side is simpler for
   self-hosted deployments, but gateway-based is more flexible for cloud.

2. **Authentication**: This RFC intentionally omits authentication. Should we
   add API key authentication as part of this work, or defer to RFC-0014
   (Cloud Service)?

3. **HTTP/2 and HTTP/3**: axum supports HTTP/2 via hyper. Should we enable
   HTTP/2 by default? HTTP/3 (QUIC) support is experimental in hyper.

4. **Request Logging and PII**: How should we handle request/response logging
   for endpoints that may contain PII (user-uploaded images, prompts)?
   Options: redaction, separate audit log, configurable per-field.

5. **WebSocket Authentication**: How should WebSocket connections be
   authenticated? Options: token in query parameter, token in first message,
   cookie-based.

6. **Backwards Compatibility**: Should the server continue to support the
   non-versioned URL paths (`/providers` instead of `/v1/providers`) during
   a transition period, or make a clean break?

7. **Binary Protocols**: Some providers return large binary payloads (point
   clouds, video frames). Should we support alternative serialization formats
   (MessagePack, Protocol Buffers) for these endpoints, or stick with
   base64-encoded JSON?

8. **Server-Sent Events vs WebSocket**: For streaming predictions, SSE is
   simpler (unidirectional, automatic reconnection). Should we support both
   SSE and WebSocket, or only WebSocket?
