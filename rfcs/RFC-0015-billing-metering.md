# RFC-0015: Billing & Usage Metering

| Field   | Value                          |
|---------|--------------------------------|
| Status  | Draft                          |
| Author  | WorldForge Core Team           |
| Created | 2026-04-02                     |
| Updated | 2026-04-02                     |

## Abstract

This RFC defines the billing and usage metering system for WorldForge. It covers
the complete lifecycle of usage tracking—from individual API events through
aggregation, cost attribution, and invoice generation. The system integrates with
Stripe for payment processing and provides usage dashboards, budget caps, cost
alerts, and webhook notifications. Three billing tiers (Free, Pro, Enterprise)
gate access to platform resources with corresponding rate limits.

## Motivation

WorldForge proxies calls to multiple AI providers (OpenAI, Anthropic, Google,
Replicate, etc.), each with their own pricing models. Users consume storage,
compute, and bandwidth in varying amounts. Without a metering and billing system:

- There is no way to recoup provider API costs or sustain the platform.
- Users have no visibility into their usage or spending.
- There is no mechanism to prevent runaway costs from misconfigured automations.
- The platform cannot offer differentiated service tiers.
- Provider cost attribution is impossible, making pricing decisions guesswork.

A robust metering pipeline is foundational infrastructure that must be in place
before any commercial offering of WorldForge.

## Detailed Design

### 1. Usage Event Model

Every billable action in WorldForge produces a `UsageEvent`. Events are the
atomic unit of metering and are immutable once created.

```rust
/// A single billable usage event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    /// Globally unique event ID (UUIDv7 for time-ordering).
    pub id: Uuid,
    /// Timestamp when the event occurred (UTC).
    pub timestamp: DateTime<Utc>,
    /// Account that owns this usage.
    pub account_id: AccountId,
    /// Optional project/world scoping.
    pub project_id: Option<ProjectId>,
    /// The type of usage.
    pub event_type: UsageEventType,
    /// Quantity consumed (interpretation depends on event_type).
    pub quantity: f64,
    /// Unit of the quantity (e.g., "tokens", "bytes", "ms").
    pub unit: String,
    /// Provider cost in USD (if applicable, before markup).
    pub provider_cost_usd: Option<f64>,
    /// Metadata for debugging and attribution.
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsageEventType {
    /// AI provider API call (prediction, embedding, etc.)
    ProviderCall {
        provider: String,
        model: String,
        input_tokens: u64,
        output_tokens: u64,
    },
    /// Storage operation (read, write, delete).
    StorageOp {
        operation: StorageOperation,
        backend: String,
        bytes: u64,
    },
    /// Compute time (physics simulation, video processing, etc.)
    ComputeTime {
        task_type: String,
        duration_ms: u64,
        gpu: bool,
    },
    /// Network bandwidth consumed.
    Bandwidth {
        direction: BandwidthDirection,
        bytes: u64,
        endpoint: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageOperation {
    Read,
    Write,
    Delete,
    List,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BandwidthDirection {
    Ingress,
    Egress,
}
```

Events are produced by instrumentation points throughout the codebase:

- **Provider calls**: The provider proxy layer emits events after each API call,
  capturing token counts, latency, and the provider's reported cost.
- **Storage ops**: State backends (file, SQLite, Redis, S3) emit events on every
  read/write/delete with byte counts.
- **Compute time**: Physics simulation, video encoding, and other CPU/GPU tasks
  emit events with wall-clock duration.
- **Bandwidth**: The HTTP server layer tracks request/response sizes.

### 2. Cost Attribution

Provider API costs are the largest variable expense. WorldForge passes these
through with a configurable markup:

```rust
pub struct CostAttribution {
    /// Raw cost reported by the provider.
    pub provider_cost: Decimal,
    /// Markup percentage (e.g., 0.20 for 20%).
    pub markup_rate: Decimal,
    /// Final cost to the user.
    pub user_cost: Decimal,
    /// Currency (always USD for now).
    pub currency: Currency,
}

impl CostAttribution {
    pub fn calculate(provider_cost: Decimal, markup_rate: Decimal) -> Self {
        let user_cost = provider_cost * (Decimal::ONE + markup_rate);
        Self {
            provider_cost,
            markup_rate,
            user_cost,
            currency: Currency::USD,
        }
    }
}
```

Cost attribution rules by event type:

| Event Type     | Cost Basis                        | Default Markup |
|----------------|-----------------------------------|----------------|
| ProviderCall   | Provider-reported cost per call   | 20%            |
| StorageOp      | $0.023/GB/month (S3-equivalent)   | 30%            |
| ComputeTime    | $0.10/hour CPU, $1.00/hour GPU    | 25%            |
| Bandwidth      | $0.09/GB egress                   | 20%            |

Markup rates are configurable per account (for enterprise negotiated pricing)
and globally via configuration.

### 3. Metering Pipeline

The metering pipeline processes events from emission to billing:

```
[Instrumentation Points]
         |
         v
  [Event Emitter] -- async channel (tokio::sync::mpsc)
         |
         v
  [Event Buffer]  -- batches events (100 events or 5s, whichever first)
         |
         v
  [Event Store]   -- persistent storage (PostgreSQL / SQLite)
         |
         v
  [Aggregator]    -- periodic roll-ups (hourly, daily, monthly)
         |
         v
  [Billing Engine] -- applies pricing, generates line items
         |
         v
  [Stripe Sync]   -- reports usage to Stripe for invoicing
```

#### Event Emitter

```rust
pub struct MeteringEmitter {
    tx: mpsc::Sender<UsageEvent>,
}

impl MeteringEmitter {
    /// Emit a usage event. Non-blocking; drops events if buffer is full
    /// rather than blocking the hot path.
    pub fn emit(&self, event: UsageEvent) {
        if self.tx.try_send(event).is_err() {
            tracing::warn!("Metering buffer full, dropping usage event");
            metrics::counter!("metering.events.dropped").increment(1);
        }
    }
}
```

#### Event Buffer and Store

Events are buffered in memory and flushed in batches to the event store.
The store uses an append-only table:

```sql
CREATE TABLE usage_events (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    account_id UUID NOT NULL,
    project_id UUID,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    unit TEXT NOT NULL,
    provider_cost_usd DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_usage_events_account_time
    ON usage_events (account_id, timestamp);

CREATE INDEX idx_usage_events_type_time
    ON usage_events (event_type, timestamp);
```

#### Aggregator

A background task runs aggregation queries at configurable intervals:

```rust
pub struct UsageAggregate {
    pub account_id: AccountId,
    pub period: AggregationPeriod,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub event_type: String,
    pub total_quantity: f64,
    pub total_provider_cost: Decimal,
    pub total_user_cost: Decimal,
    pub event_count: u64,
}

pub enum AggregationPeriod {
    Hourly,
    Daily,
    Monthly,
}
```

Aggregates are stored in a separate table for fast dashboard queries:

```sql
CREATE TABLE usage_aggregates (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL,
    period TEXT NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    total_quantity DOUBLE PRECISION NOT NULL,
    total_provider_cost DOUBLE PRECISION NOT NULL,
    total_user_cost DOUBLE PRECISION NOT NULL,
    event_count BIGINT NOT NULL,
    UNIQUE (account_id, period, period_start, event_type)
);
```

### 4. Stripe Integration

WorldForge uses Stripe for payment processing, subscription management, and
invoicing. The integration uses the `stripe-rust` crate.

#### Subscription Model

Each billing tier maps to a Stripe Product + Price:

```rust
pub struct StripeIntegration {
    client: stripe::Client,
    config: StripeConfig,
}

pub struct StripeConfig {
    pub api_key: SecretString,
    pub webhook_secret: SecretString,
    pub free_price_id: String,
    pub pro_price_id: String,
    pub metered_price_id: String,  // For usage-based overage billing
}
```

#### Usage Reporting

At the end of each billing period, WorldForge reports metered usage to Stripe:

```rust
impl StripeIntegration {
    /// Report usage for a subscription item (called by billing engine).
    pub async fn report_usage(
        &self,
        subscription_item_id: &str,
        quantity: u64,
        timestamp: DateTime<Utc>,
    ) -> Result<(), BillingError> {
        let params = CreateUsageRecord {
            quantity: quantity as i64,
            timestamp: Some(timestamp.timestamp()),
            action: Some(UsageRecordAction::Set),
            ..Default::default()
        };
        UsageRecord::create(
            &self.client,
            &subscription_item_id.parse()?,
            params,
        ).await?;
        Ok(())
    }
}
```

#### Webhook Handling

Stripe webhooks notify WorldForge of payment events:

```rust
pub async fn handle_stripe_webhook(
    payload: Bytes,
    signature: &str,
    config: &StripeConfig,
) -> Result<(), BillingError> {
    let event = Webhook::construct_event(
        &String::from_utf8_lossy(&payload),
        signature,
        &config.webhook_secret.expose_secret(),
    )?;

    match event.type_ {
        EventType::InvoicePaymentSucceeded => {
            // Update account status, unlock features
        }
        EventType::InvoicePaymentFailed => {
            // Notify user, start grace period
        }
        EventType::CustomerSubscriptionUpdated => {
            // Update tier, adjust rate limits
        }
        EventType::CustomerSubscriptionDeleted => {
            // Downgrade to free tier
        }
        _ => {
            tracing::debug!(?event.type_, "Unhandled Stripe event");
        }
    }
    Ok(())
}
```

### 5. Usage Dashboards

The API exposes endpoints for usage visibility:

```
GET /api/v1/usage/summary
    ?period=monthly&start=2026-03-01&end=2026-04-01
    -> { predictions_used: 847, predictions_limit: 10000,
         storage_gb: 2.3, compute_hours: 4.1,
         total_cost_usd: 23.47, breakdown: [...] }

GET /api/v1/usage/events
    ?type=provider_call&limit=100&offset=0
    -> { events: [...], total: 847 }

GET /api/v1/usage/timeseries
    ?metric=predictions&granularity=daily&start=...&end=...
    -> { data_points: [{ timestamp: ..., value: 42 }, ...] }

GET /api/v1/billing/invoices
    -> { invoices: [{ id: ..., amount: 23.47, status: "paid", ... }] }

GET /api/v1/billing/current
    -> { tier: "pro", period_start: ..., period_end: ...,
         usage: {...}, projected_cost: 31.20 }
```

### 6. Billing Tiers

Three tiers with clear limits:

| Feature              | Free          | Pro ($49/mo)  | Enterprise     |
|----------------------|---------------|---------------|----------------|
| Predictions/month    | 100           | 10,000        | Custom         |
| Storage              | 1 GB          | 50 GB         | Custom         |
| Compute hours        | 1 hr/mo       | 50 hr/mo      | Custom         |
| Bandwidth            | 5 GB/mo       | 200 GB/mo     | Custom         |
| Worlds               | 3             | Unlimited     | Unlimited      |
| API rate limit       | 10 req/min    | 300 req/min   | Custom         |
| Support              | Community     | Email         | Dedicated      |
| Provider passthrough | +30% markup   | +20% markup   | Negotiable     |
| Overage              | Hard cap      | $0.005/pred   | Negotiable     |

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BillingTier {
    Free,
    Pro,
    Enterprise { contract_id: String },
}

#[derive(Debug, Clone)]
pub struct TierLimits {
    pub predictions_per_month: u64,
    pub storage_bytes: u64,
    pub compute_ms_per_month: u64,
    pub bandwidth_bytes_per_month: u64,
    pub max_worlds: Option<u64>,
    pub rate_limit_per_minute: u64,
    pub markup_rate: Decimal,
    pub overage_allowed: bool,
    pub overage_cost_per_prediction: Option<Decimal>,
}

impl BillingTier {
    pub fn limits(&self) -> TierLimits {
        match self {
            BillingTier::Free => TierLimits {
                predictions_per_month: 100,
                storage_bytes: 1_073_741_824,       // 1 GB
                compute_ms_per_month: 3_600_000,     // 1 hour
                bandwidth_bytes_per_month: 5_368_709_120, // 5 GB
                max_worlds: Some(3),
                rate_limit_per_minute: 10,
                markup_rate: dec!(0.30),
                overage_allowed: false,
                overage_cost_per_prediction: None,
            },
            BillingTier::Pro => TierLimits {
                predictions_per_month: 10_000,
                storage_bytes: 53_687_091_200,       // 50 GB
                compute_ms_per_month: 180_000_000,   // 50 hours
                bandwidth_bytes_per_month: 214_748_364_800, // 200 GB
                max_worlds: None,
                rate_limit_per_minute: 300,
                markup_rate: dec!(0.20),
                overage_allowed: true,
                overage_cost_per_prediction: Some(dec!(0.005)),
            },
            BillingTier::Enterprise { .. } => {
                // Loaded from contract configuration
                unimplemented!("Enterprise limits are contract-specific")
            }
        }
    }
}
```

### 7. Rate Limiting Tied to Billing Tiers

Rate limiting uses a token bucket algorithm, with bucket sizes determined by the
account's billing tier:

```rust
pub struct RateLimiter {
    buckets: DashMap<AccountId, TokenBucket>,
    tier_resolver: Arc<dyn TierResolver>,
}

pub struct TokenBucket {
    tokens: AtomicU64,
    max_tokens: u64,
    refill_rate: u64,  // tokens per second
    last_refill: AtomicI64,
}

impl RateLimiter {
    pub async fn check_rate_limit(
        &self,
        account_id: &AccountId,
    ) -> Result<(), RateLimitError> {
        let tier = self.tier_resolver.resolve(account_id).await?;
        let limits = tier.limits();
        let bucket = self.buckets.entry(*account_id).or_insert_with(|| {
            TokenBucket::new(limits.rate_limit_per_minute, limits.rate_limit_per_minute / 60)
        });
        bucket.try_consume(1).map_err(|remaining_ms| {
            RateLimitError::Exceeded {
                retry_after_ms: remaining_ms,
                limit: limits.rate_limit_per_minute,
                tier: tier.clone(),
            }
        })
    }

    pub async fn check_monthly_quota(
        &self,
        account_id: &AccountId,
        event_type: &UsageEventType,
    ) -> Result<(), QuotaError> {
        let tier = self.tier_resolver.resolve(account_id).await?;
        let limits = tier.limits();
        let current_usage = self.get_current_month_usage(account_id, event_type).await?;

        match event_type {
            UsageEventType::ProviderCall { .. } => {
                if current_usage >= limits.predictions_per_month as f64 {
                    if limits.overage_allowed {
                        return Ok(());  // Will be billed as overage
                    }
                    return Err(QuotaError::MonthlyLimitReached {
                        limit: limits.predictions_per_month,
                        used: current_usage as u64,
                    });
                }
            }
            // ... similar checks for other event types
            _ => {}
        }
        Ok(())
    }
}
```

### 8. Invoice Generation

Invoices are generated monthly via a scheduled task:

```rust
pub struct Invoice {
    pub id: InvoiceId,
    pub account_id: AccountId,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub line_items: Vec<InvoiceLineItem>,
    pub subtotal: Decimal,
    pub tax: Decimal,
    pub total: Decimal,
    pub status: InvoiceStatus,
    pub stripe_invoice_id: Option<String>,
    pub pdf_url: Option<String>,
}

pub struct InvoiceLineItem {
    pub description: String,
    pub event_type: String,
    pub quantity: f64,
    pub unit: String,
    pub unit_price: Decimal,
    pub amount: Decimal,
}

pub enum InvoiceStatus {
    Draft,
    Finalized,
    Sent,
    Paid,
    Overdue,
    Void,
}
```

The invoice generation flow:
1. Monthly cron job triggers at period end.
2. Aggregator finalizes monthly totals per account.
3. Billing engine computes line items with cost attribution.
4. Invoice is created in WorldForge database.
5. Corresponding Stripe invoice is finalized.
6. PDF is generated and stored.
7. Notification is sent to the account owner.

### 9. Cost Alerts and Budget Caps

Users can configure alerts and hard caps:

```rust
pub struct BudgetConfig {
    pub account_id: AccountId,
    /// Alert thresholds as percentages of the monthly budget.
    pub alert_thresholds: Vec<AlertThreshold>,
    /// Hard cap: reject requests when exceeded. None = no cap.
    pub hard_cap_usd: Option<Decimal>,
    /// Monthly budget for projection alerts.
    pub monthly_budget_usd: Decimal,
}

pub struct AlertThreshold {
    /// Percentage of budget (e.g., 50, 80, 100).
    pub percent: u32,
    /// Notification channels.
    pub notify: Vec<NotificationChannel>,
    /// Whether this threshold has been triggered this period.
    pub triggered: bool,
}

pub enum NotificationChannel {
    Email { address: String },
    Webhook { url: String },
    Slack { channel: String },
}
```

Budget enforcement runs as middleware in the request pipeline:

```rust
pub async fn budget_enforcement_middleware(
    account_id: &AccountId,
    budget_config: &BudgetConfig,
    current_spend: Decimal,
) -> Result<(), BudgetError> {
    // Check hard cap
    if let Some(cap) = budget_config.hard_cap_usd {
        if current_spend >= cap {
            return Err(BudgetError::HardCapReached {
                cap,
                current_spend,
            });
        }
    }

    // Check alert thresholds
    let usage_percent = (current_spend / budget_config.monthly_budget_usd * dec!(100))
        .to_u32()
        .unwrap_or(0);

    for threshold in &budget_config.alert_thresholds {
        if usage_percent >= threshold.percent && !threshold.triggered {
            send_budget_alert(account_id, threshold, current_spend).await?;
        }
    }

    Ok(())
}
```

### 10. Webhook Notifications for Billing Events

WorldForge emits webhooks for billing events to allow integrations:

```rust
pub enum BillingWebhookEvent {
    /// Usage approaching tier limit.
    UsageThresholdReached {
        account_id: AccountId,
        event_type: String,
        current_usage: f64,
        limit: f64,
        percent: u32,
    },
    /// Invoice generated.
    InvoiceCreated {
        invoice_id: InvoiceId,
        amount: Decimal,
    },
    /// Payment succeeded.
    PaymentSucceeded {
        invoice_id: InvoiceId,
        amount: Decimal,
    },
    /// Payment failed.
    PaymentFailed {
        invoice_id: InvoiceId,
        amount: Decimal,
        reason: String,
    },
    /// Budget alert triggered.
    BudgetAlert {
        account_id: AccountId,
        threshold_percent: u32,
        current_spend: Decimal,
        budget: Decimal,
    },
    /// Tier changed.
    TierChanged {
        account_id: AccountId,
        old_tier: BillingTier,
        new_tier: BillingTier,
    },
}
```

Webhooks are delivered with HMAC-SHA256 signatures for verification, with
exponential backoff retry (1s, 5s, 30s, 5m, 30m) for failed deliveries.

## Implementation Plan

### Phase 1: Event Collection (2 weeks)
- Define `UsageEvent` types and serialization.
- Instrument provider proxy layer to emit events.
- Instrument storage backends to emit events.
- Implement in-memory event buffer with async channel.
- Add SQLite event store for single-node deployments.

### Phase 2: Aggregation & Dashboards (2 weeks)
- Implement hourly/daily/monthly aggregation background task.
- Build aggregation tables and queries.
- Expose usage summary, events, and timeseries API endpoints.
- Create basic usage dashboard views.

### Phase 3: Billing Tiers & Rate Limiting (2 weeks)
- Define tier model with limits.
- Implement token bucket rate limiter.
- Add monthly quota checking middleware.
- Wire tier changes to rate limit adjustments.

### Phase 4: Stripe Integration (3 weeks)
- Set up Stripe products, prices, and webhook endpoints.
- Implement subscription lifecycle management.
- Build usage reporting to Stripe.
- Handle webhook events (payment success/failure, subscription changes).
- Invoice generation and PDF storage.

### Phase 5: Alerts & Webhooks (1 week)
- Implement budget configuration API.
- Build threshold checking in request pipeline.
- Implement webhook delivery with retry logic.
- Add email notification integration.

### Phase 6: Hardening (1 week)
- Idempotency for all billing operations.
- Reconciliation job to detect metering gaps.
- Load testing of metering pipeline.
- Audit logging for all billing state changes.

## Testing Strategy

### Unit Tests
- Cost attribution calculation with various markup rates.
- Token bucket rate limiter behavior (refill, consume, overflow).
- Tier limit enforcement for each event type.
- Aggregation correctness (hourly rollups match event sums).
- Invoice line item calculation.

### Integration Tests
- End-to-end event emission through aggregation pipeline.
- Stripe webhook signature validation.
- Rate limiting under concurrent requests.
- Budget cap enforcement stops requests.
- Webhook delivery and retry logic.

### Property-Based Tests
- Aggregated totals always match sum of individual events.
- Rate limiter never allows more than configured burst.
- Cost attribution is always non-negative with valid markup.

### Load Tests
- Metering pipeline throughput: target 10K events/second.
- Event buffer behavior under sustained load (no unbounded growth).
- Aggregation query performance with 1M+ events.

## Open Questions

1. **PostgreSQL requirement**: The metering pipeline assumes a relational store.
   Should we support SQLite for small deployments, or require PostgreSQL for
   any deployment with billing enabled?

2. **Multi-currency support**: Current design is USD-only. When should we add
   support for EUR, GBP, etc.? Stripe handles currency conversion, but our
   internal cost model would need adjustment.

3. **Usage pre-authorization**: Should we check provider cost estimates before
   making calls (to enforce budgets preemptively), or only record costs after?
   Pre-auth adds latency but prevents surprise overages.

4. **Free tier sustainability**: 100 predictions/month may be too generous or
   too stingy. How should we determine the right free tier allocation?

5. **Provider cost tracking accuracy**: Some providers report costs asynchronously
   or don't report them at all. How do we handle cost estimation vs actual cost
   for these providers?

6. **Data retention**: How long should raw usage events be retained? Aggregates
   may suffice for billing, but raw events are useful for debugging. Consider
   tiered retention (30 days raw, 1 year aggregated, indefinite invoices).

7. **Stripe alternatives**: Should we support other payment processors (Paddle,
   LemonSqueezy) from the start, or design the billing engine with pluggable
   payment backends?
