# RFC-0020: Observability & Telemetry

| Field   | Value                          |
|---------|--------------------------------|
| Status  | Draft                          |
| Author  | WorldForge Core Team           |
| Created | 2026-04-02                     |
| Updated | 2026-04-02                     |

## Abstract

This RFC defines the observability and telemetry strategy for WorldForge. It
builds on the existing `tracing` 0.1 and `tracing-subscriber` 0.3 dependencies
to add OpenTelemetry integration for distributed tracing, Prometheus metrics
export for monitoring, structured health check endpoints, Grafana dashboard
templates, and alerting rules. The goal is complete visibility into system
behavior, performance, and cost across all WorldForge components.

## Motivation

WorldForge already uses the `tracing` crate for structured logging, which is an
excellent foundation. However, production observability requires much more:

- **No metrics export**: There are no quantitative measurements (latency, error
  rates, throughput) being collected or exported.
- **No distributed tracing**: When a request spans provider calls, queue workers,
  and state backends, there is no way to trace the full request lifecycle.
- **No health checks**: No endpoint reports whether the system and its
  dependencies (Redis, S3, providers) are healthy.
- **No dashboards**: Operators have no pre-built views for understanding system
  behavior.
- **No alerting**: There are no defined thresholds for when to page an operator.
- **No cost visibility**: Per-request cost tracking is absent.

Without observability, operating WorldForge in production is flying blind. This
RFC provides the instrumentation, export, and visualization layers needed for
reliable operations.

## Detailed Design

### 1. Structured Logging with Tracing

WorldForge already depends on `tracing` 0.1 and `tracing-subscriber` 0.3. We
standardize logging patterns and add structured context.

```rust
use tracing::{info, warn, error, debug, instrument, Span};

/// Standard request span with all relevant context.
#[instrument(
    name = "handle_request",
    skip(request, state),
    fields(
        request_id = %request_id,
        method = %request.method(),
        path = %request.uri().path(),
        account_id = tracing::field::Empty,
        duration_ms = tracing::field::Empty,
        status_code = tracing::field::Empty,
    )
)]
pub async fn handle_request(
    request: Request,
    state: AppState,
    request_id: String,
) -> Response {
    let start = Instant::now();

    // Authentication fills in account_id
    let auth = authenticate(&request).await;
    if let Ok(ref ctx) = auth {
        Span::current().record("account_id", &ctx.account_id.to_string().as_str());
    }

    let response = process_request(request, state).await;

    let duration = start.elapsed();
    Span::current().record("duration_ms", &duration.as_millis());
    Span::current().record("status_code", &response.status().as_u16());

    response
}

/// Configure the tracing subscriber pipeline.
pub fn init_tracing(config: &TracingConfig) -> Result<(), TracingError> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.default_level));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true);

    let fmt_layer = if config.json_logs {
        fmt_layer.json().flatten_event(true).boxed()
    } else {
        fmt_layer.pretty().boxed()
    };

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer);

    // Add OpenTelemetry layer if configured
    let registry = if let Some(ref otel_config) = config.opentelemetry {
        let otel_layer = init_opentelemetry(otel_config)?;
        registry.with(otel_layer).boxed()
    } else {
        registry.boxed()
    };

    tracing::subscriber::set_global_default(registry)?;
    Ok(())
}

pub struct TracingConfig {
    /// Default log level (e.g., "info", "debug").
    pub default_level: String,
    /// Output logs as JSON (for log aggregation).
    pub json_logs: bool,
    /// OpenTelemetry configuration.
    pub opentelemetry: Option<OpenTelemetryConfig>,
}
```

#### Log Level Guidelines

| Level | Use Case                                                |
|-------|---------------------------------------------------------|
| ERROR | Unrecoverable failures, data loss risk                  |
| WARN  | Degraded behavior, retryable failures, approaching limits|
| INFO  | Request lifecycle, state changes, configuration         |
| DEBUG | Detailed flow, intermediate values                      |
| TRACE | Very verbose, per-frame/per-step logging                |

### 2. OpenTelemetry Integration

Distributed tracing via OpenTelemetry, using `tracing-opentelemetry` to bridge
the existing `tracing` instrumentation.

```rust
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::WithExportConfig;
use tracing_opentelemetry::OpenTelemetryLayer;

pub struct OpenTelemetryConfig {
    /// OTLP endpoint (e.g., "http://localhost:4317").
    pub endpoint: String,
    /// Service name reported to the collector.
    pub service_name: String,
    /// Service version.
    pub service_version: String,
    /// Sampling rate (0.0 to 1.0, 1.0 = sample everything).
    pub sampling_rate: f64,
    /// Export protocol.
    pub protocol: OtlpProtocol,
    /// Additional resource attributes.
    pub resource_attributes: HashMap<String, String>,
}

pub enum OtlpProtocol {
    Grpc,
    Http,
}

pub fn init_opentelemetry(
    config: &OpenTelemetryConfig,
) -> Result<OpenTelemetryLayer<Registry, opentelemetry_sdk::trace::Tracer>, TracingError> {
    let mut resource_attrs = vec![
        opentelemetry::KeyValue::new("service.name", config.service_name.clone()),
        opentelemetry::KeyValue::new("service.version", config.service_version.clone()),
    ];

    for (k, v) in &config.resource_attributes {
        resource_attrs.push(opentelemetry::KeyValue::new(k.clone(), v.clone()));
    }

    let resource = opentelemetry_sdk::Resource::new(resource_attrs);

    let exporter = match config.protocol {
        OtlpProtocol::Grpc => {
            opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .with_endpoint(&config.endpoint)
                .build()?
        }
        OtlpProtocol::Http => {
            opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_endpoint(&config.endpoint)
                .build()?
        }
    };

    let sampler = opentelemetry_sdk::trace::Sampler::TraceIdRatioBased(
        config.sampling_rate
    );

    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .with_sampler(sampler)
        .build();

    let tracer = provider.tracer("worldforge");

    opentelemetry::global::set_tracer_provider(provider);

    Ok(tracing_opentelemetry::layer().with_tracer(tracer))
}
```

#### Trace Context Propagation

```rust
use opentelemetry::propagation::{TextMapPropagator, Injector, Extractor};
use opentelemetry_sdk::propagation::TraceContextPropagator;

pub struct TracePropagation;

impl TracePropagation {
    /// Extract trace context from incoming HTTP request headers.
    pub fn extract_from_request(headers: &HeaderMap) -> opentelemetry::Context {
        let propagator = TraceContextPropagator::new();
        let extractor = HeaderExtractor(headers);
        propagator.extract(&extractor)
    }

    /// Inject trace context into outgoing HTTP request headers.
    pub fn inject_into_request(
        cx: &opentelemetry::Context,
        headers: &mut HeaderMap,
    ) {
        let propagator = TraceContextPropagator::new();
        let mut injector = HeaderInjector(headers);
        propagator.inject_context(cx, &mut injector);
    }
}

/// Use in provider API calls to propagate trace context.
#[instrument(skip(client, request_body))]
pub async fn call_provider(
    client: &reqwest::Client,
    provider: &str,
    url: &str,
    request_body: &serde_json::Value,
) -> Result<ProviderResponse, ProviderError> {
    let mut request = client.post(url)
        .json(request_body)
        .build()?;

    // Inject trace context so provider calls appear in the same trace
    let cx = Span::current().context();
    TracePropagation::inject_into_request(&cx, request.headers_mut());

    let start = Instant::now();
    let response = client.execute(request).await?;
    let duration = start.elapsed();

    // Record metrics
    metrics::histogram!("provider.call.duration_ms", "provider" => provider.to_string())
        .record(duration.as_millis() as f64);
    metrics::counter!("provider.calls.total", "provider" => provider.to_string())
        .increment(1);

    if !response.status().is_success() {
        metrics::counter!(
            "provider.calls.errors",
            "provider" => provider.to_string(),
            "status" => response.status().as_u16().to_string()
        ).increment(1);
    }

    Ok(parse_provider_response(response).await?)
}
```

### 3. Prometheus Metrics Export

Using the `metrics` crate with `metrics-exporter-prometheus` for Prometheus-
compatible metrics export.

```rust
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;

pub struct MetricsConfig {
    /// Enable Prometheus metrics export.
    pub enabled: bool,
    /// Port for the metrics endpoint (default: 9090).
    pub port: u16,
    /// Endpoint path (default: "/metrics").
    pub path: String,
    /// Global labels added to all metrics.
    pub global_labels: HashMap<String, String>,
    /// Histogram bucket boundaries.
    pub histogram_buckets: Vec<f64>,
}

pub fn init_metrics(config: &MetricsConfig) -> Result<(), MetricsError> {
    if !config.enabled {
        return Ok(());
    }

    let mut builder = PrometheusBuilder::new()
        .with_http_listener(([0, 0, 0, 0], config.port));

    // Set default histogram buckets for latency measurements
    let buckets = if config.histogram_buckets.is_empty() {
        vec![5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
             1000.0, 2500.0, 5000.0, 10000.0, 30000.0]
    } else {
        config.histogram_buckets.clone()
    };

    builder = builder.set_buckets(&buckets)?;

    for (key, value) in &config.global_labels {
        builder = builder.add_global_label(key, value);
    }

    builder.install()?;
    Ok(())
}
```

### 4. Key Metrics

#### Request Metrics

```rust
pub fn record_request_metrics(
    method: &str,
    path: &str,
    status: u16,
    duration: Duration,
    account_id: &str,
) {
    let labels = [
        ("method", method.to_string()),
        ("path", normalize_path(path)),
        ("status", status.to_string()),
        ("status_class", format!("{}xx", status / 100)),
    ];

    // Request duration histogram
    histogram!("http_request_duration_ms", &labels)
        .record(duration.as_millis() as f64);

    // Request count
    counter!("http_requests_total", &labels).increment(1);

    // Active requests gauge (increment on start, decrement on end)
    // Handled by middleware wrapping
}

/// Normalize path to avoid high-cardinality labels.
fn normalize_path(path: &str) -> String {
    // Replace UUIDs and IDs with placeholders
    let re = regex::Regex::new(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    ).unwrap();
    re.replace_all(path, ":id").to_string()
}
```

#### Provider Metrics

```rust
pub fn record_provider_metrics(
    provider: &str,
    model: &str,
    duration: Duration,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
    success: bool,
) {
    let labels = [
        ("provider", provider.to_string()),
        ("model", model.to_string()),
    ];

    histogram!("provider_call_duration_ms", &labels)
        .record(duration.as_millis() as f64);

    counter!("provider_calls_total", &labels).increment(1);

    if !success {
        counter!("provider_calls_errors_total", &labels).increment(1);
    }

    counter!("provider_tokens_input_total", &labels).increment(input_tokens);
    counter!("provider_tokens_output_total", &labels).increment(output_tokens);

    histogram!("provider_cost_usd", &labels).record(cost_usd);
    counter!("provider_cost_usd_total", &labels).increment(cost_usd as u64);
}
```

#### System Metrics

```rust
pub fn record_system_metrics() {
    // Active worlds
    gauge!("worldforge_active_worlds").set(/* query from state */);

    // Prediction cache
    gauge!("worldforge_cache_size").set(/* cache entries */);
    counter!("worldforge_cache_hits_total").increment(/* on hit */);
    counter!("worldforge_cache_misses_total").increment(/* on miss */);

    // Queue depth (for async prediction queue)
    gauge!("worldforge_queue_depth").set(/* pending items */);
    gauge!("worldforge_queue_workers_active").set(/* active workers */);

    // State backend metrics
    histogram!("state_operation_duration_ms",
        "backend" => "sqlite",
        "operation" => "read"
    ).record(/* duration */);
}

/// Background task that periodically records gauge metrics.
pub async fn metrics_collection_loop(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(15));
    loop {
        interval.tick().await;

        // Process metrics
        if let Ok(info) = sys_info::mem_info() {
            gauge!("process_memory_rss_bytes").set(info.total as f64 * 1024.0);
        }

        // World count
        if let Ok(count) = state.world_store.count().await {
            gauge!("worldforge_active_worlds").set(count as f64);
        }

        // Cache stats
        let cache_stats = state.prediction_cache.stats();
        gauge!("worldforge_cache_size").set(cache_stats.entries as f64);
        gauge!("worldforge_cache_memory_bytes").set(cache_stats.memory_bytes as f64);

        // Queue depth
        gauge!("worldforge_queue_depth").set(state.task_queue.len() as f64);
    }
}
```

#### Complete Metrics Reference

| Metric Name                          | Type      | Labels                        | Description                                |
|--------------------------------------|-----------|-------------------------------|--------------------------------------------|
| http_request_duration_ms             | Histogram | method, path, status          | Request latency in milliseconds            |
| http_requests_total                  | Counter   | method, path, status_class    | Total HTTP requests                        |
| http_requests_active                 | Gauge     | -                             | Currently in-flight requests               |
| provider_call_duration_ms            | Histogram | provider, model               | Provider API call latency                  |
| provider_calls_total                 | Counter   | provider, model               | Total provider API calls                   |
| provider_calls_errors_total          | Counter   | provider, model, error_type   | Failed provider API calls                  |
| provider_tokens_input_total          | Counter   | provider, model               | Total input tokens sent to providers       |
| provider_tokens_output_total         | Counter   | provider, model               | Total output tokens from providers         |
| provider_cost_usd                    | Histogram | provider, model               | Per-call cost in USD                       |
| worldforge_active_worlds             | Gauge     | -                             | Number of active worlds                    |
| worldforge_cache_size                | Gauge     | -                             | Prediction cache entries                   |
| worldforge_cache_hits_total          | Counter   | -                             | Cache hit count                            |
| worldforge_cache_misses_total        | Counter   | -                             | Cache miss count                           |
| worldforge_queue_depth               | Gauge     | -                             | Pending items in task queue                |
| state_operation_duration_ms          | Histogram | backend, operation            | State backend operation latency            |
| auth_attempts_total                  | Counter   | method, outcome               | Authentication attempts                    |
| process_memory_rss_bytes             | Gauge     | -                             | Process resident memory                    |

### 5. Health Check Endpoint

```rust
/// Health check response with dependency status.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub version: String,
    pub uptime_seconds: u64,
    pub checks: HashMap<String, DependencyHealth>,
}

#[derive(Debug, Serialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Serialize)]
pub struct DependencyHealth {
    pub status: HealthStatus,
    pub latency_ms: Option<u64>,
    pub message: Option<String>,
    pub last_checked: DateTime<Utc>,
}

pub struct HealthChecker {
    state: Arc<AppState>,
    start_time: Instant,
}

impl HealthChecker {
    pub async fn check(&self) -> HealthResponse {
        let mut checks = HashMap::new();

        // Check state backend
        checks.insert(
            "state_backend".to_string(),
            self.check_state_backend().await,
        );

        // Check Redis (if configured)
        if self.state.redis.is_some() {
            checks.insert(
                "redis".to_string(),
                self.check_redis().await,
            );
        }

        // Check S3 (if configured)
        if self.state.s3.is_some() {
            checks.insert(
                "s3".to_string(),
                self.check_s3().await,
            );
        }

        // Check provider connectivity
        for provider in &self.state.configured_providers {
            checks.insert(
                format!("provider_{}", provider),
                self.check_provider(provider).await,
            );
        }

        // Determine overall status
        let status = if checks.values().all(|c| c.status == HealthStatus::Healthy) {
            HealthStatus::Healthy
        } else if checks.values().any(|c| c.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        };

        HealthResponse {
            status,
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            checks,
        }
    }

    async fn check_state_backend(&self) -> DependencyHealth {
        let start = Instant::now();
        match self.state.world_store.ping().await {
            Ok(_) => DependencyHealth {
                status: HealthStatus::Healthy,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                message: None,
                last_checked: Utc::now(),
            },
            Err(e) => DependencyHealth {
                status: HealthStatus::Unhealthy,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                message: Some(e.to_string()),
                last_checked: Utc::now(),
            },
        }
    }

    async fn check_redis(&self) -> DependencyHealth {
        let start = Instant::now();
        if let Some(ref redis) = self.state.redis {
            match redis.ping().await {
                Ok(_) => DependencyHealth {
                    status: HealthStatus::Healthy,
                    latency_ms: Some(start.elapsed().as_millis() as u64),
                    message: None,
                    last_checked: Utc::now(),
                },
                Err(e) => DependencyHealth {
                    status: HealthStatus::Degraded,
                    latency_ms: Some(start.elapsed().as_millis() as u64),
                    message: Some(format!("Redis error: {}", e)),
                    last_checked: Utc::now(),
                },
            }
        } else {
            DependencyHealth {
                status: HealthStatus::Healthy,
                latency_ms: None,
                message: Some("Not configured".to_string()),
                last_checked: Utc::now(),
            }
        }
    }

    async fn check_provider(&self, provider: &str) -> DependencyHealth {
        let start = Instant::now();
        match self.state.provider_registry.health_check(provider).await {
            Ok(_) => DependencyHealth {
                status: HealthStatus::Healthy,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                message: None,
                last_checked: Utc::now(),
            },
            Err(e) => DependencyHealth {
                status: HealthStatus::Degraded,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                message: Some(format!("Provider health check failed: {}", e)),
                last_checked: Utc::now(),
            },
        }
    }
}

/// API endpoints.
/// GET /health -> full health check (may be slow due to dependency checks)
/// GET /health/live -> liveness probe (always returns 200 if process is running)
/// GET /health/ready -> readiness probe (returns 200 if ready to serve traffic)
pub async fn health_handler(state: Arc<AppState>) -> impl IntoResponse {
    let checker = HealthChecker::new(state);
    let health = checker.check().await;
    let status_code = match health.status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded => StatusCode::OK,  // Still serving
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };
    (status_code, Json(health))
}

pub async fn liveness_handler() -> impl IntoResponse {
    StatusCode::OK
}

pub async fn readiness_handler(state: Arc<AppState>) -> impl IntoResponse {
    // Check critical dependencies only
    if state.world_store.ping().await.is_ok() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}
```

### 6. Grafana Dashboard Templates

Dashboard JSON templates are provided in `deploy/grafana/`:

#### Overview Dashboard

```json
{
  "title": "WorldForge Overview",
  "panels": [
    {
      "title": "Request Rate",
      "type": "timeseries",
      "targets": [{
        "expr": "rate(http_requests_total[5m])",
        "legendFormat": "{{method}} {{path}} {{status_class}}"
      }]
    },
    {
      "title": "Request Latency (p50, p90, p99)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(http_request_duration_ms_bucket[5m]))",
          "legendFormat": "p50"
        },
        {
          "expr": "histogram_quantile(0.90, rate(http_request_duration_ms_bucket[5m]))",
          "legendFormat": "p90"
        },
        {
          "expr": "histogram_quantile(0.99, rate(http_request_duration_ms_bucket[5m]))",
          "legendFormat": "p99"
        }
      ]
    },
    {
      "title": "Error Rate",
      "type": "stat",
      "targets": [{
        "expr": "sum(rate(http_requests_total{status_class='5xx'}[5m])) / sum(rate(http_requests_total[5m])) * 100",
        "legendFormat": "Error %"
      }]
    },
    {
      "title": "Active Worlds",
      "type": "stat",
      "targets": [{
        "expr": "worldforge_active_worlds",
        "legendFormat": "Worlds"
      }]
    },
    {
      "title": "Provider Calls by Provider",
      "type": "timeseries",
      "targets": [{
        "expr": "rate(provider_calls_total[5m])",
        "legendFormat": "{{provider}}/{{model}}"
      }]
    },
    {
      "title": "Cache Hit Rate",
      "type": "gauge",
      "targets": [{
        "expr": "rate(worldforge_cache_hits_total[5m]) / (rate(worldforge_cache_hits_total[5m]) + rate(worldforge_cache_misses_total[5m])) * 100",
        "legendFormat": "Hit Rate %"
      }]
    },
    {
      "title": "Queue Depth",
      "type": "timeseries",
      "targets": [{
        "expr": "worldforge_queue_depth",
        "legendFormat": "Pending Tasks"
      }]
    },
    {
      "title": "Provider Cost (Cumulative, 24h)",
      "type": "stat",
      "targets": [{
        "expr": "increase(provider_cost_usd_total[24h])",
        "legendFormat": "{{provider}}"
      }]
    }
  ]
}
```

#### Provider Performance Dashboard

```json
{
  "title": "WorldForge Provider Performance",
  "panels": [
    {
      "title": "Provider Latency by Model",
      "type": "heatmap",
      "targets": [{
        "expr": "rate(provider_call_duration_ms_bucket[5m])",
        "legendFormat": "{{provider}}/{{model}}"
      }]
    },
    {
      "title": "Provider Error Rate",
      "type": "timeseries",
      "targets": [{
        "expr": "rate(provider_calls_errors_total[5m]) / rate(provider_calls_total[5m]) * 100",
        "legendFormat": "{{provider}} error %"
      }]
    },
    {
      "title": "Token Throughput",
      "type": "timeseries",
      "targets": [
        {
          "expr": "rate(provider_tokens_input_total[5m])",
          "legendFormat": "{{provider}} input"
        },
        {
          "expr": "rate(provider_tokens_output_total[5m])",
          "legendFormat": "{{provider}} output"
        }
      ]
    },
    {
      "title": "Cost per Request",
      "type": "timeseries",
      "targets": [{
        "expr": "rate(provider_cost_usd[5m])",
        "legendFormat": "{{provider}}/{{model}}"
      }]
    }
  ]
}
```

### 7. Alerting Rules

Prometheus alerting rules for critical conditions:

```yaml
# deploy/prometheus/alerts.yml
groups:
  - name: worldforge.rules
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status_class="5xx"}[5m]))
          / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: >
            Error rate is {{ $value | humanizePercentage }} over the last 5
            minutes. Threshold is 5%.

      # High latency
      - alert: HighLatencyP99
        expr: |
          histogram_quantile(0.99, rate(http_request_duration_ms_bucket[5m]))
          > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency exceeds 10 seconds"
          description: >
            P99 request latency is {{ $value }}ms. Threshold is 10000ms.

      # Provider degraded
      - alert: ProviderHighErrorRate
        expr: |
          rate(provider_calls_errors_total[5m])
          / rate(provider_calls_total[5m]) > 0.10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Provider {{ $labels.provider }} error rate high"
          description: >
            Provider {{ $labels.provider }} error rate is
            {{ $value | humanizePercentage }}. Check provider status.

      # Provider latency spike
      - alert: ProviderHighLatency
        expr: |
          histogram_quantile(0.95, rate(provider_call_duration_ms_bucket[5m]))
          > 30000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Provider {{ $labels.provider }} latency spike"
          description: >
            Provider {{ $labels.provider }} p95 latency is {{ $value }}ms.

      # Queue backing up
      - alert: QueueBacklog
        expr: worldforge_queue_depth > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Task queue backlog growing"
          description: >
            Queue depth is {{ $value }}. Tasks may be processing slowly.

      # Cache degraded
      - alert: LowCacheHitRate
        expr: |
          rate(worldforge_cache_hits_total[15m])
          / (rate(worldforge_cache_hits_total[15m])
             + rate(worldforge_cache_misses_total[15m])) < 0.5
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "Prediction cache hit rate below 50%"
          description: >
            Cache hit rate is {{ $value | humanizePercentage }}.
            Consider increasing cache size.

      # High cost rate
      - alert: HighCostRate
        expr: |
          increase(provider_cost_usd_total[1h]) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Provider costs exceeding $100/hour"
          description: >
            Hourly provider cost is ${{ $value }}. Check for runaway usage.

      # Service unhealthy
      - alert: ServiceUnhealthy
        expr: |
          up{job="worldforge"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "WorldForge service is down"
          description: "The WorldForge instance has been unreachable for 1 minute."

      # Memory pressure
      - alert: HighMemoryUsage
        expr: |
          process_memory_rss_bytes / 1024 / 1024 / 1024 > 4
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage (>4GB)"
          description: >
            Process RSS is {{ $value }}GB. Check for memory leaks.
```

### 8. Cost Tracking Per Request

```rust
/// Middleware that tracks the total cost of a request.
pub struct CostTracker {
    request_costs: DashMap<String, RequestCost>,
}

#[derive(Debug, Default)]
pub struct RequestCost {
    pub provider_calls: Vec<ProviderCallCost>,
    pub compute_cost: f64,
    pub storage_cost: f64,
    pub total_cost: f64,
}

#[derive(Debug)]
pub struct ProviderCallCost {
    pub provider: String,
    pub model: String,
    pub cost_usd: f64,
    pub tokens_in: u64,
    pub tokens_out: u64,
}

impl CostTracker {
    pub fn start_request(&self, request_id: &str) {
        self.request_costs.insert(
            request_id.to_string(),
            RequestCost::default(),
        );
    }

    pub fn add_provider_cost(
        &self,
        request_id: &str,
        cost: ProviderCallCost,
    ) {
        if let Some(mut entry) = self.request_costs.get_mut(request_id) {
            entry.total_cost += cost.cost_usd;
            entry.provider_calls.push(cost);
        }
    }

    pub fn finish_request(&self, request_id: &str) -> Option<RequestCost> {
        self.request_costs.remove(request_id).map(|(_, cost)| {
            // Record cost metrics
            histogram!("request_total_cost_usd").record(cost.total_cost);
            if cost.total_cost > 1.0 {
                tracing::warn!(
                    request_id = request_id,
                    cost = cost.total_cost,
                    "High-cost request detected"
                );
            }
            cost
        })
    }
}
```

The request cost is optionally included in response headers:

```
X-WorldForge-Cost: 0.0234
X-WorldForge-Tokens-In: 150
X-WorldForge-Tokens-Out: 500
```

## Implementation Plan

### Phase 1: Metrics Foundation (2 weeks)
- Add `metrics` and `metrics-exporter-prometheus` dependencies.
- Initialize Prometheus exporter with configurable port.
- Instrument HTTP request handler with latency and count metrics.
- Instrument provider calls with duration, token counts, and error metrics.
- Add system metrics collection background task.
- Expose `/metrics` endpoint.

### Phase 2: Distributed Tracing (2 weeks)
- Add `tracing-opentelemetry` and `opentelemetry-otlp` dependencies.
- Configure OpenTelemetry tracer provider.
- Implement trace context propagation for provider calls.
- Add trace context to async task queue.
- Test with Jaeger or Tempo collector.

### Phase 3: Health Checks (1 week)
- Implement `/health`, `/health/live`, `/health/ready` endpoints.
- Add dependency health checks (state backend, Redis, S3, providers).
- Configure Kubernetes probe compatibility.

### Phase 4: Dashboards & Alerts (1 week)
- Create Grafana dashboard JSON templates.
- Write Prometheus alerting rules.
- Document Alertmanager configuration.
- Create docker-compose for local observability stack.

### Phase 5: Cost Tracking (1 week)
- Implement per-request cost tracking middleware.
- Add cost response headers.
- Cost metrics (histogram of request costs).
- Cost alerting rules.

### Phase 6: Documentation & Deployment (1 week)
- Deployment guide for observability stack.
- Runbook for each alert.
- Log level tuning guide.
- Performance impact documentation.

## Testing Strategy

### Unit Tests
- Metric recording produces expected Prometheus output.
- Health check returns correct status for various dependency states.
- Cost tracker accumulates costs correctly across concurrent requests.
- Path normalization removes high-cardinality segments.
- Trace context serialization/deserialization roundtrip.

### Integration Tests
- Full request flow produces expected spans in OpenTelemetry collector.
- Prometheus scrape endpoint returns valid metrics.
- Health endpoint reflects actual dependency state (mock down dependencies).
- Metrics survive process restart (counter persistence is not required, but
  the exporter should start cleanly).

### Load Tests
- Metrics collection overhead: < 1% latency increase at p99.
- Tracing overhead with 100% sampling vs 1% sampling.
- Prometheus scrape with 10K+ unique metric series.
- Health check timeout handling (slow dependency doesn't block readiness).

### Observability Stack Tests
- Docker-compose stack starts and Grafana dashboards load.
- Alerting rules fire correctly with synthetic metrics.
- Jaeger/Tempo shows traces with correct parent-child relationships.

## Open Questions

1. **Metrics cardinality**: High-cardinality labels (per-user, per-world) can
   overwhelm Prometheus. Should we limit labels to low-cardinality dimensions
   and use logs/traces for per-entity debugging?

2. **Sampling strategy**: 100% trace sampling is expensive in production. Should
   we default to 1% with head-based sampling, or use tail-based sampling
   (sample all errors and slow requests)?

3. **Metrics vs tracing overlap**: Both systems can record latency. Should we
   derive metrics from traces (using span metrics connector in OpenTelemetry
   Collector) to avoid double instrumentation?

4. **Log aggregation**: This RFC doesn't prescribe a log aggregation backend
   (Loki, Elasticsearch, etc.). Should we provide configuration for common
   backends?

5. **Custom metrics API**: Should users be able to define custom metrics for
   their worlds (e.g., prediction accuracy over time)?

6. **SLA monitoring**: Should we provide built-in SLO/SLI tracking (error
   budget, availability percentage)?

7. **Profiling**: Should we integrate continuous profiling (e.g., `pprof` via
   `pprof-rs`) alongside metrics and tracing for the full observability trifecta?
