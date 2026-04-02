# RFC-0014: Cloud Service Architecture

| Field   | Value                              |
|---------|------------------------------------|
| Title   | Cloud Service Architecture         |
| Status  | Draft                              |
| Author  | WorldForge Core Team               |
| Created | 2026-04-02                         |
| Updated | 2026-04-02                         |

---

## Abstract

This RFC defines the architecture for running WorldForge as a managed cloud
service, covering multi-tenant isolation, horizontal scaling, credential
management, async job queues, result caching, CDN for generated assets,
Kubernetes deployment, Terraform infrastructure, auto-scaling, monitoring,
and SLA targets. The cloud service transforms WorldForge from a self-hosted
tool into a production platform that multiple organizations can use
simultaneously with isolation, security, and reliability guarantees.

---

## Motivation

WorldForge is currently designed as a single-user, self-hosted server. To
reach a broader audience and generate sustainable revenue, we need a managed
cloud offering. Key drivers:

1. **Ease of Adoption**: Most potential users do not want to manage
   infrastructure. A cloud service eliminates setup friction and lets
   developers start making predictions in minutes.

2. **Multi-Tenancy**: Organizations need isolated environments with separate
   credentials, rate limits, and billing. The current single-instance model
   cannot support this.

3. **Scalability**: Individual predictions can take 10-30 seconds and consume
   significant compute. A cloud service must handle hundreds of concurrent
   predictions from multiple tenants without degradation.

4. **Credential Management**: Each tenant may have their own API keys for
   different providers (e.g., their own NVIDIA Cosmos key). These credentials
   must be encrypted, rotatable, and never exposed.

5. **Reliability**: A cloud service needs formal SLA targets, health monitoring,
   alerting, and automated recovery. Self-hosted deployments have no such
   guarantees.

6. **Cost Optimization**: Shared infrastructure across tenants enables better
   resource utilization than per-tenant deployments. Caching and CDN reduce
   redundant computation and bandwidth costs.

### Target SLA

| Metric                           | Target           |
|----------------------------------|------------------|
| Overall availability             | 99.9% (8.76h/yr) |
| Metadata API p99 latency         | < 500 ms         |
| Prediction API p99 latency       | < 30 seconds     |
| Prediction API p50 latency       | < 10 seconds     |
| Time to first byte (streaming)   | < 2 seconds      |
| Error rate (5xx)                 | < 0.1%           |
| Data durability (results)        | 99.99%           |
| Planned maintenance windows      | < 4 hours/month  |

---

## Detailed Design

### 1. Multi-Tenant Architecture

#### 1.1 Tenant Model

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    /// Unique tenant identifier
    pub id: TenantId,

    /// Human-readable organization name
    pub name: String,

    /// Tenant tier (determines rate limits, features, SLA)
    pub tier: TenantTier,

    /// API keys for this tenant (multiple allowed)
    pub api_keys: Vec<ApiKeyInfo>,

    /// Provider credentials (encrypted)
    pub provider_credentials: HashMap<String, EncryptedCredential>,

    /// Rate limit configuration
    pub rate_limits: RateLimitConfig,

    /// Feature flags
    pub features: TenantFeatures,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Status
    pub status: TenantStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TenantTier {
    /// Free tier: limited rate, shared resources
    Free,
    /// Pro tier: higher limits, priority queue
    Pro,
    /// Enterprise tier: dedicated resources, custom SLA
    Enterprise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum metadata requests per minute
    pub metadata_rpm: u32,
    /// Maximum prediction requests per minute
    pub prediction_rpm: u32,
    /// Maximum concurrent predictions
    pub max_concurrent_predictions: u32,
    /// Maximum WebSocket connections
    pub max_websocket_connections: u32,
    /// Maximum request body size (bytes)
    pub max_request_body_bytes: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            metadata_rpm: 100,
            prediction_rpm: 10,
            max_concurrent_predictions: 5,
            max_websocket_connections: 10,
            max_request_body_bytes: 16 * 1024 * 1024, // 16 MB
        }
    }
}
```

#### 1.2 Tenant Isolation

Isolation is enforced at multiple layers:

| Layer              | Isolation Mechanism                              |
|--------------------|--------------------------------------------------|
| Network            | Tenant-specific API keys, not network-level      |
| Authentication     | API key → tenant ID mapping, JWT tokens          |
| Authorization      | Tenant can only access own resources             |
| Data               | All data tagged with tenant_id, filtered in queries|
| Rate Limiting      | Per-tenant rate limits enforced at API gateway    |
| Compute            | Per-tenant concurrency limits, priority queues    |
| Credentials        | Per-tenant encrypted credential storage           |
| Billing            | Per-tenant usage metering and invoicing           |

#### 1.3 Authentication Flow

```
Client                    API Gateway              WorldForge Server
  │                           │                           │
  │  Request + API Key        │                           │
  │──────────────────────────►│                           │
  │                           │  Validate API Key         │
  │                           │  Look up Tenant ID        │
  │                           │  Check Rate Limits        │
  │                           │                           │
  │                           │  Request + Tenant Context │
  │                           │──────────────────────────►│
  │                           │                           │
  │                           │  Response                 │
  │                           │◄──────────────────────────│
  │  Response                 │                           │
  │◄──────────────────────────│                           │
```

API keys are prefixed with the tenant tier for quick identification:

```
wf_free_a1b2c3d4e5f6...     # Free tier key
wf_pro_x9y8z7w6v5u4...      # Pro tier key
wf_ent_m1n2o3p4q5r6...      # Enterprise tier key
```

### 2. Horizontal Scaling Architecture

#### 2.1 Stateless Server Design

```
                    ┌─────────────────────┐
                    │    Load Balancer     │
                    │  (AWS ALB / GCP LB) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼────────┐ ┌─────▼──────────┐
    │  Server Pod 1  │ │  Server Pod 2 │ │  Server Pod N  │
    │  (stateless)   │ │  (stateless)  │ │  (stateless)   │
    └───────┬────────┘ └───────┬───────┘ └───────┬────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
   ┌─────▼─────┐      ┌───────▼──────┐     ┌───────▼──────┐
   │   Redis   │      │  PostgreSQL  │     │  Object Store│
   │  (queue,  │      │  (tenants,   │     │  (S3/GCS,    │
   │   cache)  │      │   metadata)  │     │   results)   │
   └───────────┘      └──────────────┘     └──────────────┘
```

All server pods are stateless and interchangeable. Shared state is stored in:

- **Redis**: Job queue, result cache, rate limit counters, session state
- **PostgreSQL**: Tenant configuration, API keys, prediction history, billing
- **Object Storage (S3/GCS)**: Prediction results (images, videos, point clouds)

#### 2.2 Shared State Components

```rust
#[derive(Clone)]
pub struct CloudAppState {
    /// Tenant registry (backed by PostgreSQL)
    pub tenants: Arc<TenantStore>,

    /// Job queue (backed by Redis)
    pub job_queue: Arc<JobQueue>,

    /// Result cache (backed by Redis)
    pub result_cache: Arc<ResultCache>,

    /// Object storage (S3/GCS)
    pub object_store: Arc<dyn ObjectStore>,

    /// Provider registry (in-memory, loaded from config)
    pub providers: Arc<ProviderRegistry>,

    /// Outbound HTTP client pool
    pub http_client: reqwest::Client,

    /// Metrics collector
    pub metrics: Arc<MetricsCollector>,
}
```

### 3. Provider Credential Management

#### 3.1 Credential Storage

Provider credentials are encrypted at rest using AES-256-GCM with per-tenant
key wrapping:

```rust
pub struct CredentialStore {
    /// Master encryption key (loaded from KMS at startup)
    master_key: [u8; 32],

    /// Database connection for encrypted credential storage
    db: PgPool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EncryptedCredential {
    /// Encrypted credential data
    pub ciphertext: Vec<u8>,

    /// AES-GCM nonce (12 bytes)
    pub nonce: [u8; 12],

    /// Key ID (for key rotation)
    pub key_id: String,

    /// Provider identifier
    pub provider: String,

    /// Credential type
    pub credential_type: CredentialType,

    /// Last rotation timestamp
    pub last_rotated: DateTime<Utc>,

    /// Expiration (if applicable)
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum CredentialType {
    ApiKey,
    OAuth2ClientCredentials {
        client_id: String,
        // client_secret is in the encrypted ciphertext
    },
    ServiceAccountJson,
    AwsCredentials,
}

impl CredentialStore {
    /// Store a credential (encrypts before storage)
    pub async fn store(
        &self,
        tenant_id: &TenantId,
        provider: &str,
        credential: &PlaintextCredential,
    ) -> Result<()> {
        let nonce = generate_nonce();
        let key = self.derive_tenant_key(tenant_id);
        let ciphertext = aes_gcm_encrypt(&key, &nonce, &credential.to_bytes())?;

        sqlx::query!(
            "INSERT INTO credentials (tenant_id, provider, ciphertext, nonce, key_id, created_at)
             VALUES ($1, $2, $3, $4, $5, NOW())
             ON CONFLICT (tenant_id, provider) DO UPDATE
             SET ciphertext = $3, nonce = $4, key_id = $5, updated_at = NOW()",
            tenant_id.as_str(),
            provider,
            &ciphertext,
            &nonce[..],
            &self.current_key_id(),
        )
        .execute(&self.db)
        .await?;

        Ok(())
    }

    /// Retrieve and decrypt a credential
    pub async fn get(
        &self,
        tenant_id: &TenantId,
        provider: &str,
    ) -> Result<PlaintextCredential> {
        let row = sqlx::query!(
            "SELECT ciphertext, nonce, key_id FROM credentials
             WHERE tenant_id = $1 AND provider = $2",
            tenant_id.as_str(),
            provider,
        )
        .fetch_one(&self.db)
        .await?;

        let key = self.derive_tenant_key(tenant_id);
        let nonce: [u8; 12] = row.nonce.try_into()?;
        let plaintext = aes_gcm_decrypt(&key, &nonce, &row.ciphertext)?;

        PlaintextCredential::from_bytes(&plaintext)
    }

    /// Rotate credentials (re-encrypt with new key)
    pub async fn rotate_keys(&self, new_master_key: &[u8; 32]) -> Result<u64> {
        // Fetch all credentials, decrypt with old key, re-encrypt with new key
        let rows = sqlx::query!("SELECT id, tenant_id, ciphertext, nonce, key_id FROM credentials")
            .fetch_all(&self.db)
            .await?;

        let mut rotated = 0u64;
        for row in rows {
            let old_key = self.derive_key_for_id(&row.key_id, &row.tenant_id);
            let new_key = derive_key(new_master_key, &row.tenant_id);
            let nonce: [u8; 12] = row.nonce.try_into()?;

            let plaintext = aes_gcm_decrypt(&old_key, &nonce, &row.ciphertext)?;
            let new_nonce = generate_nonce();
            let new_ciphertext = aes_gcm_encrypt(&new_key, &new_nonce, &plaintext)?;

            sqlx::query!(
                "UPDATE credentials SET ciphertext = $1, nonce = $2, key_id = $3, updated_at = NOW()
                 WHERE id = $4",
                &new_ciphertext,
                &new_nonce[..],
                &self.current_key_id(),
                row.id,
            )
            .execute(&self.db)
            .await?;

            rotated += 1;
        }

        Ok(rotated)
    }

    fn derive_tenant_key(&self, tenant_id: &TenantId) -> [u8; 32] {
        derive_key(&self.master_key, tenant_id.as_str())
    }
}

fn derive_key(master: &[u8; 32], context: &str) -> [u8; 32] {
    use hkdf::Hkdf;
    use sha2::Sha256;

    let hk = Hkdf::<Sha256>::new(None, master);
    let mut key = [0u8; 32];
    hk.expand(context.as_bytes(), &mut key).expect("valid length");
    key
}
```

#### 3.2 Credential Lifecycle

1. **Creation**: Tenant provides credentials via API or dashboard. Credentials
   are encrypted and stored immediately. Plaintext never written to logs.

2. **Usage**: When a prediction needs provider credentials, the server decrypts
   them in memory, uses them for the API call, and zeroes the memory after use.

3. **Rotation**: Credentials can be rotated (updated) by the tenant at any time.
   Key wrapping keys can be rotated by the operator (re-encrypts all credentials).

4. **Deletion**: When a tenant deletes credentials or is deactivated, the
   encrypted credential is deleted from the database. No soft-delete for
   security-sensitive data.

5. **Expiration**: Credentials with expiration dates trigger alerts 7 days
   before expiry and are automatically disabled on expiration.

### 4. Job Queue for Async Predictions

#### 4.1 Queue Architecture

```rust
pub struct JobQueue {
    redis: redis::aio::ConnectionManager,
    config: JobQueueConfig,
}

#[derive(Debug, Clone)]
pub struct JobQueueConfig {
    /// Redis key prefix for queue data
    pub prefix: String,
    /// Maximum job age before expiry (default: 1 hour)
    pub max_job_age: Duration,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Retry backoff base (exponential)
    pub retry_backoff_base: Duration,
    /// Number of worker tasks per server pod
    pub workers_per_pod: usize,
    /// Priority queue names
    pub priority_queues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionJob {
    /// Unique job ID
    pub id: JobId,
    /// Tenant that submitted the job
    pub tenant_id: TenantId,
    /// Prediction request
    pub request: PredictionRequest,
    /// Priority (higher = processed first)
    pub priority: u32,
    /// Job status
    pub status: JobStatus,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Number of retry attempts
    pub attempts: u32,
    /// Last error (if any)
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Processing { worker_id: String, started_at: DateTime<Utc> },
    Completed { completed_at: DateTime<Utc> },
    Failed { error: String, failed_at: DateTime<Utc> },
    Cancelled { cancelled_at: DateTime<Utc> },
}
```

#### 4.2 Queue Operations

```rust
impl JobQueue {
    /// Submit a prediction job to the queue
    pub async fn submit(&self, job: PredictionJob) -> Result<JobId> {
        let job_data = serde_json::to_string(&job)?;
        let queue_key = self.queue_key(&job);

        // Store job data
        let data_key = format!("{}:job:{}", self.config.prefix, job.id);
        self.redis.set_ex(&data_key, &job_data, self.config.max_job_age.as_secs() as u64).await?;

        // Add to priority queue (sorted set, score = priority * 1M + timestamp)
        let score = (job.priority as f64) * 1_000_000.0
            + (u64::MAX - job.created_at.timestamp() as u64) as f64;
        self.redis.zadd(&queue_key, &job.id.to_string(), score).await?;

        // Publish notification for waiting workers
        self.redis.publish(&format!("{}:notify", self.config.prefix), "new_job").await?;

        Ok(job.id)
    }

    /// Dequeue the highest-priority job (called by workers)
    pub async fn dequeue(&self, worker_id: &str) -> Result<Option<PredictionJob>> {
        // Atomically pop from sorted set and update job status
        // Uses Redis MULTI/EXEC for atomicity
        let queue_keys: Vec<String> = self.config.priority_queues.iter()
            .map(|q| format!("{}:queue:{}", self.config.prefix, q))
            .collect();

        for queue_key in &queue_keys {
            let result: Option<(String, f64)> = redis::cmd("ZPOPMAX")
                .arg(queue_key)
                .query_async(&mut self.redis.clone())
                .await?;

            if let Some((job_id, _score)) = result {
                let data_key = format!("{}:job:{}", self.config.prefix, job_id);
                let job_data: String = self.redis.get(&data_key).await?;
                let mut job: PredictionJob = serde_json::from_str(&job_data)?;

                job.status = JobStatus::Processing {
                    worker_id: worker_id.to_string(),
                    started_at: Utc::now(),
                };
                job.attempts += 1;

                self.redis.set_ex(&data_key, &serde_json::to_string(&job)?, self.config.max_job_age.as_secs() as u64).await?;

                return Ok(Some(job));
            }
        }

        Ok(None)
    }

    /// Complete a job with results
    pub async fn complete(&self, job_id: &JobId, result: &PredictionResponse) -> Result<()> {
        let data_key = format!("{}:job:{}", self.config.prefix, job_id);
        let result_key = format!("{}:result:{}", self.config.prefix, job_id);

        // Store result
        self.redis.set_ex(
            &result_key,
            &serde_json::to_string(result)?,
            3600, // Results available for 1 hour
        ).await?;

        // Update job status
        let job_data: String = self.redis.get(&data_key).await?;
        let mut job: PredictionJob = serde_json::from_str(&job_data)?;
        job.status = JobStatus::Completed { completed_at: Utc::now() };
        self.redis.set_ex(&data_key, &serde_json::to_string(&job)?, 3600).await?;

        // Notify waiting clients
        self.redis.publish(&format!("{}:result:{}", self.config.prefix, job_id), "completed").await?;

        Ok(())
    }

    /// Fail a job (with optional retry)
    pub async fn fail(&self, job_id: &JobId, error: &str) -> Result<bool> {
        let data_key = format!("{}:job:{}", self.config.prefix, job_id);
        let job_data: String = self.redis.get(&data_key).await?;
        let mut job: PredictionJob = serde_json::from_str(&job_data)?;

        if job.attempts < self.config.max_retries {
            // Re-queue with exponential backoff
            let backoff = self.config.retry_backoff_base * 2u32.pow(job.attempts - 1);
            job.status = JobStatus::Queued;
            job.last_error = Some(error.to_string());

            // Delayed re-queue using Redis sorted set with future timestamp
            let requeue_at = Utc::now() + chrono::Duration::from_std(backoff)?;
            let queue_key = self.queue_key(&job);
            let score = (job.priority as f64) * 1_000_000.0
                + (u64::MAX - requeue_at.timestamp() as u64) as f64;

            self.redis.set_ex(&data_key, &serde_json::to_string(&job)?, self.config.max_job_age.as_secs() as u64).await?;
            self.redis.zadd(&queue_key, &job_id.to_string(), score).await?;

            Ok(true) // Will retry
        } else {
            job.status = JobStatus::Failed {
                error: error.to_string(),
                failed_at: Utc::now(),
            };
            self.redis.set_ex(&data_key, &serde_json::to_string(&job)?, 3600).await?;
            self.redis.publish(&format!("{}:result:{}", self.config.prefix, job_id), "failed").await?;

            Ok(false) // Permanently failed
        }
    }

    fn queue_key(&self, job: &PredictionJob) -> String {
        let tier = match job.priority {
            0..=3 => "free",
            4..=7 => "pro",
            _ => "enterprise",
        };
        format!("{}:queue:{}", self.config.prefix, tier)
    }
}
```

#### 4.3 Worker Pool

```rust
pub struct WorkerPool {
    workers: Vec<JoinHandle<()>>,
    shutdown: Arc<tokio::sync::watch::Sender<bool>>,
}

impl WorkerPool {
    pub fn start(
        num_workers: usize,
        queue: Arc<JobQueue>,
        providers: Arc<ProviderRegistry>,
        credentials: Arc<CredentialStore>,
        cache: Arc<ResultCache>,
    ) -> Self {
        let (shutdown_tx, _) = tokio::sync::watch::channel(false);
        let shutdown = Arc::new(shutdown_tx);

        let workers = (0..num_workers)
            .map(|i| {
                let worker_id = format!("worker-{}-{}", hostname(), i);
                let queue = queue.clone();
                let providers = providers.clone();
                let credentials = credentials.clone();
                let cache = cache.clone();
                let mut shutdown_rx = shutdown.subscribe();

                tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            _ = shutdown_rx.changed() => break,
                            job = queue.dequeue(&worker_id) => {
                                match job {
                                    Ok(Some(job)) => {
                                        process_job(&queue, &providers, &credentials, &cache, job).await;
                                    }
                                    Ok(None) => {
                                        // No jobs available, wait for notification
                                        tokio::time::sleep(Duration::from_millis(100)).await;
                                    }
                                    Err(e) => {
                                        tracing::error!("Worker {} dequeue error: {}", worker_id, e);
                                        tokio::time::sleep(Duration::from_secs(1)).await;
                                    }
                                }
                            }
                        }
                    }
                    tracing::info!("Worker {} shutting down", worker_id);
                })
            })
            .collect();

        Self { workers, shutdown }
    }
}

async fn process_job(
    queue: &JobQueue,
    providers: &ProviderRegistry,
    credentials: &CredentialStore,
    cache: &ResultCache,
    job: PredictionJob,
) {
    let span = tracing::info_span!("process_job",
        job_id = %job.id,
        tenant_id = %job.tenant_id,
        provider = %job.request.provider,
    );

    async move {
        // Check cache first
        let cache_key = cache.key_for_request(&job.request);
        if let Some(cached) = cache.get(&cache_key).await {
            tracing::info!("Cache hit for job {}", job.id);
            queue.complete(&job.id, &cached).await.ok();
            return;
        }

        // Get provider credentials
        let creds = match credentials.get(&job.tenant_id, &job.request.provider).await {
            Ok(c) => c,
            Err(e) => {
                queue.fail(&job.id, &format!("Credential error: {}", e)).await.ok();
                return;
            }
        };

        // Execute prediction
        let provider = match providers.get(&job.request.provider) {
            Some(p) => p,
            None => {
                queue.fail(&job.id, &format!("Unknown provider: {}", job.request.provider)).await.ok();
                return;
            }
        };

        match provider.predict_with_credentials(&job.request, &creds).await {
            Ok(response) => {
                // Cache result
                cache.set(&cache_key, &response, cache.ttl_for_request(&job.request)).await.ok();
                // Complete job
                queue.complete(&job.id, &response).await.ok();
            }
            Err(e) => {
                queue.fail(&job.id, &e.to_string()).await.ok();
            }
        }
    }
    .instrument(span)
    .await;
}
```

### 5. Result Caching

#### 5.1 Content-Addressed Cache

Prediction results are cached using a content-addressed key derived from the
request parameters. Identical requests return cached results without re-running
the prediction.

```rust
pub struct ResultCache {
    redis: redis::aio::ConnectionManager,
    config: CacheConfig,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Redis key prefix
    pub prefix: String,
    /// Default TTL for cached results
    pub default_ttl: Duration,
    /// Maximum cache entry size (bytes)
    pub max_entry_size: usize,
    /// TTL overrides by provider
    pub provider_ttl: HashMap<String, Duration>,
}

impl ResultCache {
    /// Generate a content-addressed cache key from the request
    pub fn key_for_request(&self, request: &PredictionRequest) -> String {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(request.provider.as_bytes());
        hasher.update(request.model.as_bytes());
        hasher.update(&serde_json::to_vec(&request.input).unwrap_or_default());

        // Include guardrail config in cache key (different guardrails = different results)
        if let Some(ref guardrails) = request.guardrails {
            hasher.update(&serde_json::to_vec(guardrails).unwrap_or_default());
        }

        let hash = hasher.finalize();
        format!("{}:cache:{}", self.config.prefix, hash.to_hex())
    }

    /// Get a cached result
    pub async fn get(&self, key: &str) -> Option<PredictionResponse> {
        let data: Option<Vec<u8>> = self.redis.get(key).await.ok()?;
        data.and_then(|d| serde_json::from_slice(&d).ok())
    }

    /// Store a result in cache
    pub async fn set(
        &self,
        key: &str,
        response: &PredictionResponse,
        ttl: Duration,
    ) -> Result<()> {
        let data = serde_json::to_vec(response)?;
        if data.len() > self.config.max_entry_size {
            // Too large for cache, store reference to object storage instead
            return Ok(());
        }
        self.redis.set_ex(key, &data, ttl.as_secs() as u64).await?;
        Ok(())
    }

    /// TTL for a specific request (provider-specific overrides)
    pub fn ttl_for_request(&self, request: &PredictionRequest) -> Duration {
        self.config.provider_ttl
            .get(&request.provider)
            .copied()
            .unwrap_or(self.config.default_ttl)
    }

    /// Invalidate all cache entries for a tenant
    pub async fn invalidate_tenant(&self, tenant_id: &TenantId) -> Result<u64> {
        let pattern = format!("{}:cache:*:{}:*", self.config.prefix, tenant_id);
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query_async(&mut self.redis.clone())
            .await?;

        if keys.is_empty() {
            return Ok(0);
        }

        let count = redis::cmd("DEL")
            .arg(&keys)
            .query_async(&mut self.redis.clone())
            .await?;

        Ok(count)
    }
}
```

#### 5.2 Cache TTL Strategy

| Content Type                  | Default TTL    | Rationale                           |
|-------------------------------|---------------|-------------------------------------|
| Provider metadata             | 5 minutes     | Changes infrequently                |
| Model capabilities            | 5 minutes     | Changes infrequently                |
| Prediction results (deterministic) | 24 hours | Same input always produces same output |
| Prediction results (stochastic)    | No cache | Different each time                 |
| Guardrail results             | 24 hours      | Deterministic given same input       |
| Health check                  | 30 seconds    | Needs to be fresh                   |

### 6. CDN for Generated Assets

#### 6.1 Asset Storage and Delivery

Large prediction outputs (video frames, point clouds, 3D meshes) are stored
in object storage and served via CDN:

```rust
pub struct AssetStore {
    /// Object storage backend (S3, GCS, or local)
    object_store: Arc<dyn ObjectStore>,

    /// CDN base URL for public asset access
    cdn_base_url: String,

    /// Signing key for pre-signed URLs
    signing_key: SigningKey,
}

impl AssetStore {
    /// Store a prediction asset and return its CDN URL
    pub async fn store_asset(
        &self,
        tenant_id: &TenantId,
        prediction_id: &str,
        asset_type: AssetType,
        data: &[u8],
    ) -> Result<AssetUrl> {
        // Content-addressed path for deduplication
        let hash = blake3::hash(data);
        let extension = asset_type.extension();
        let path = format!(
            "assets/{}/{}/{}.{}",
            tenant_id, prediction_id, hash.to_hex(), extension
        );

        // Upload to object storage
        self.object_store.put(
            &path.into(),
            data.into(),
        ).await?;

        // Generate pre-signed CDN URL (valid for 1 hour)
        let signed_url = self.generate_signed_url(&path, Duration::from_secs(3600))?;

        Ok(AssetUrl {
            path,
            url: signed_url,
            size_bytes: data.len() as u64,
            content_type: asset_type.content_type().to_string(),
            hash: hash.to_hex().to_string(),
        })
    }

    fn generate_signed_url(&self, path: &str, ttl: Duration) -> Result<String> {
        let expires = Utc::now() + chrono::Duration::from_std(ttl)?;
        let signature = self.signing_key.sign(
            format!("{}:{}", path, expires.timestamp()).as_bytes()
        );

        Ok(format!(
            "{}/{}?expires={}&signature={}",
            self.cdn_base_url,
            path,
            expires.timestamp(),
            hex::encode(signature.as_bytes()),
        ))
    }
}

#[derive(Debug, Clone)]
pub enum AssetType {
    VideoMp4,
    ImagePng,
    ImageJpeg,
    PointCloudPly,
    PointCloudPcd,
    Mesh3dGltf,
    AudioWav,
}

impl AssetType {
    fn extension(&self) -> &str {
        match self {
            Self::VideoMp4 => "mp4",
            Self::ImagePng => "png",
            Self::ImageJpeg => "jpg",
            Self::PointCloudPly => "ply",
            Self::PointCloudPcd => "pcd",
            Self::Mesh3dGltf => "gltf",
            Self::AudioWav => "wav",
        }
    }

    fn content_type(&self) -> &str {
        match self {
            Self::VideoMp4 => "video/mp4",
            Self::ImagePng => "image/png",
            Self::ImageJpeg => "image/jpeg",
            Self::PointCloudPly => "application/octet-stream",
            Self::PointCloudPcd => "application/octet-stream",
            Self::Mesh3dGltf => "model/gltf+json",
            Self::AudioWav => "audio/wav",
        }
    }
}
```

#### 6.2 CDN Configuration

- **CloudFront (AWS)** or **Cloud CDN (GCP)** for global edge caching
- Cache-Control headers: `public, max-age=86400` for prediction assets
- Pre-signed URLs with 1-hour expiry for access control
- Automatic content-type detection from file extension
- Brotli/gzip compression for text-based formats (GLTF, PLY)
- No compression for already-compressed formats (MP4, PNG, JPEG)

### 7. Kubernetes Deployment

#### 7.1 Helm Chart Structure

```
charts/worldforge/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── serviceaccount.yaml
│   ├── pdb.yaml
│   └── tests/
│       └── test-connection.yaml
```

#### 7.2 Key Kubernetes Resources

```yaml
# values.yaml
replicaCount: 3

image:
  repository: ghcr.io/worldforge/worldforge-server
  tag: latest
  pullPolicy: IfNotPresent

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 50
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60

redis:
  enabled: true
  architecture: replication
  replica:
    replicaCount: 3
  master:
    resources:
      requests:
        cpu: 250m
        memory: 256Mi

postgresql:
  enabled: true
  architecture: replication
  readReplicas:
    replicaCount: 2
  primary:
    resources:
      requests:
        cpu: 500m
        memory: 512Mi

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
  hosts:
    - host: api.worldforge.dev
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: worldforge-tls
      hosts:
        - api.worldforge.dev

podDisruptionBudget:
  enabled: true
  minAvailable: 2

serviceMonitor:
  enabled: true
  interval: 15s
```

### 8. Terraform Infrastructure

#### 8.1 AWS Configuration

```hcl
# terraform/aws/main.tf

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "worldforge-${var.environment}"
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    general = {
      desired_size = 3
      min_size     = 2
      max_size     = 10

      instance_types = ["m6i.xlarge"]
      capacity_type  = "ON_DEMAND"
    }

    prediction_workers = {
      desired_size = 2
      min_size     = 1
      max_size     = 20

      instance_types = ["c6i.2xlarge"]
      capacity_type  = "SPOT"

      labels = {
        workload = "prediction"
      }

      taints = [{
        key    = "workload"
        value  = "prediction"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "worldforge-${var.environment}"

  engine               = "postgres"
  engine_version       = "16.1"
  instance_class       = "db.r6g.large"
  allocated_storage    = 100
  max_allocated_storage = 500

  multi_az               = true
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [module.security_group.security_group_id]

  backup_retention_period = 30
  deletion_protection     = true

  performance_insights_enabled = true
}

module "elasticache" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  cluster_id           = "worldforge-${var.environment}"
  engine               = "redis"
  node_type            = "cache.r6g.large"
  num_cache_nodes      = 3
  parameter_group_name = "default.redis7"

  subnet_group_name    = module.vpc.elasticache_subnet_group_name
  security_group_ids   = [module.security_group.security_group_id]

  automatic_failover_enabled = true
  multi_az_enabled           = true
}

module "s3" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 4.0"

  bucket = "worldforge-assets-${var.environment}"

  versioning = {
    enabled = true
  }

  lifecycle_rule = [{
    id      = "cleanup"
    enabled = true
    expiration = {
      days = 30
    }
  }]

  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        sse_algorithm = "aws:kms"
      }
    }
  }
}

module "cloudfront" {
  source  = "terraform-aws-modules/cloudfront/aws"
  version = "~> 3.0"

  origin = {
    s3 = {
      domain_name = module.s3.s3_bucket_bucket_regional_domain_name
      s3_origin_config = {
        origin_access_identity = module.cloudfront.cloudfront_origin_access_identity_iam_arns[0]
      }
    }
  }

  default_cache_behavior = {
    target_origin_id       = "s3"
    viewer_protocol_policy = "redirect-to-https"
    compress               = true
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]

    min_ttl     = 0
    default_ttl = 86400
    max_ttl     = 604800
  }
}
```

#### 8.2 GCP Configuration

A parallel Terraform configuration for GCP using:
- GKE for Kubernetes
- Cloud SQL for PostgreSQL
- Memorystore for Redis
- Cloud Storage for objects
- Cloud CDN for asset delivery

### 9. Auto-Scaling Policies

#### 9.1 Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: worldforge-server
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: worldforge-server
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: worldforge_active_predictions
        target:
          type: AverageValue
          averageValue: "10"
    - type: External
      external:
        metric:
          name: worldforge_queue_depth
        target:
          type: Value
          value: "50"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
        - type: Pods
          value: 5
          periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
      selectPolicy: Min
```

#### 9.2 Cluster Auto-Scaling

Node pools auto-scale based on pod scheduling pressure:

- **General pool**: 2-10 nodes, m6i.xlarge (4 vCPU, 16 GB RAM)
- **Prediction pool**: 1-20 nodes, c6i.2xlarge (8 vCPU, 16 GB RAM), spot instances
- Scale-up trigger: pods pending for > 30 seconds
- Scale-down trigger: node utilization < 50% for > 10 minutes

### 10. Health Monitoring

#### 10.1 Health Endpoints

```rust
/// Liveness probe — is the server process alive?
async fn liveness() -> StatusCode {
    StatusCode::OK
}

/// Readiness probe — is the server ready to accept traffic?
async fn readiness(State(state): State<CloudAppState>) -> Result<Json<ReadinessResponse>, StatusCode> {
    let redis_ok = state.job_queue.ping().await.is_ok();
    let db_ok = state.tenants.ping().await.is_ok();

    if redis_ok && db_ok {
        Ok(Json(ReadinessResponse {
            status: "ready",
            checks: vec![
                HealthCheck { name: "redis", status: "ok" },
                HealthCheck { name: "postgres", status: "ok" },
            ],
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Detailed health check (not exposed to load balancer)
async fn health_detailed(State(state): State<CloudAppState>) -> Json<DetailedHealth> {
    let redis_latency = measure_latency(|| state.job_queue.ping()).await;
    let db_latency = measure_latency(|| state.tenants.ping()).await;
    let queue_depth = state.job_queue.queue_depth().await.unwrap_or(0);
    let active_predictions = state.metrics.active_predictions();
    let cache_hit_rate = state.result_cache.hit_rate().await;

    Json(DetailedHealth {
        status: "healthy",
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds: state.metrics.uptime().as_secs(),
        checks: vec![
            DetailedCheck { name: "redis", status: "ok", latency_ms: redis_latency },
            DetailedCheck { name: "postgres", status: "ok", latency_ms: db_latency },
        ],
        metrics: HealthMetrics {
            queue_depth,
            active_predictions,
            cache_hit_rate,
            requests_per_second: state.metrics.rps(),
            error_rate: state.metrics.error_rate(),
        },
    })
}
```

#### 10.2 Prometheus Metrics

```rust
// Key metrics exported at /metrics
pub struct MetricsCollector {
    pub http_requests_total: IntCounterVec,       // by method, path, status
    pub http_request_duration: HistogramVec,       // by method, path
    pub prediction_duration: HistogramVec,         // by provider, model
    pub prediction_queue_depth: IntGauge,
    pub active_predictions: IntGauge,
    pub cache_hits_total: IntCounter,
    pub cache_misses_total: IntCounter,
    pub provider_errors_total: IntCounterVec,      // by provider, error_type
    pub tenant_requests_total: IntCounterVec,      // by tenant_id, endpoint
    pub credential_operations_total: IntCounterVec, // by operation (get, store, rotate)
    pub websocket_connections: IntGauge,
}
```

#### 10.3 Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: worldforge
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 1% for 5 minutes"

      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(http_request_duration_bucket{path=~"/v1/providers.*"}[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p99 metadata latency above 500ms"

      - alert: PredictionLatency
        expr: histogram_quantile(0.99, rate(prediction_duration_bucket[5m])) > 30
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "p99 prediction latency above 30 seconds"

      - alert: QueueBacklog
        expr: prediction_queue_depth > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Prediction queue depth above 100"

      - alert: LowCacheHitRate
        expr: rate(cache_hits_total[1h]) / (rate(cache_hits_total[1h]) + rate(cache_misses_total[1h])) < 0.3
        for: 1h
        labels:
          severity: info
        annotations:
          summary: "Cache hit rate below 30%"

      - alert: ServerDown
        expr: up{job="worldforge-server"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "WorldForge server instance is down"
```

#### 10.4 Dashboards

Grafana dashboards for:

1. **Overview**: Request rate, error rate, latency percentiles, active pods
2. **Predictions**: Prediction volume by provider, latency by model, queue depth
3. **Tenants**: Per-tenant request rates, rate limit hits, error rates
4. **Infrastructure**: CPU, memory, disk, network across all pods
5. **Cache**: Hit/miss rates, eviction rates, memory usage
6. **Providers**: Per-provider error rates, latency, availability

---

## Implementation Plan

### Phase 1: Foundation (Weeks 1-3)

1. Design PostgreSQL schema for tenants, API keys, credentials
2. Implement tenant store and API key authentication
3. Implement credential store with AES-256-GCM encryption
4. Set up Redis for job queue and caching
5. Implement basic job queue (submit, dequeue, complete, fail)

### Phase 2: Core Services (Weeks 4-6)

6. Implement worker pool with Tokio tasks
7. Implement result caching with content-addressed keys
8. Implement asset storage and CDN URL generation
9. Add per-tenant rate limiting
10. Add Prometheus metrics and health endpoints

### Phase 3: Infrastructure (Weeks 7-9)

11. Create Helm chart for Kubernetes deployment
12. Write Terraform configurations for AWS
13. Write Terraform configurations for GCP
14. Set up CI/CD for infrastructure changes
15. Configure auto-scaling policies

### Phase 4: Observability (Weeks 10-11)

16. Deploy Prometheus and Grafana
17. Create alerting rules
18. Build Grafana dashboards
19. Set up PagerDuty/OpsGenie integration
20. Implement structured logging with correlation IDs

### Phase 5: Hardening (Weeks 12-14)

21. Security audit (credential handling, API authentication)
22. Load testing (target: 99.9% availability under load)
23. Chaos engineering (pod failure, Redis failure, DB failover)
24. Documentation (operator guide, runbooks)
25. Compliance review (SOC 2 readiness, GDPR data handling)

---

## Testing Strategy

### Unit Tests

- Tenant CRUD operations
- API key generation, validation, and revocation
- Credential encryption/decryption round-trip
- Job queue operations (submit, dequeue, complete, fail, retry)
- Cache key generation determinism
- Rate limiting logic (token bucket algorithm)
- Asset URL generation and signing

### Integration Tests

- Full tenant lifecycle: create → configure → use → deactivate
- Job queue under concurrent access (multiple workers)
- Credential rotation without service disruption
- Cache invalidation propagation
- Redis failover behavior
- PostgreSQL failover behavior

### Load Tests

- Sustained 1,000 metadata req/s across 100 tenants
- Sustained 100 prediction req/s with queue processing
- Spike test: 10x traffic surge for 60 seconds
- Soak test: 24-hour sustained load at 50% capacity
- Cache effectiveness: verify hit rate > 50% under realistic workload

### Chaos Tests

- Kill random server pods during load test
- Redis primary failover during active job processing
- PostgreSQL failover during credential operations
- Network partition between server pods and Redis
- Object storage unavailability (graceful degradation)

### Security Tests

- Verify tenant isolation: tenant A cannot access tenant B's data
- Verify credential isolation: credentials never appear in logs or responses
- Verify rate limiting: exceed rate limit and verify 429 responses
- Verify API key revocation: revoked key is immediately rejected
- Penetration testing by third-party security firm

---

## Open Questions

1. **Multi-Region**: Should the initial deployment be single-region or
   multi-region? Multi-region provides better latency and availability but
   adds significant complexity (data replication, conflict resolution).

2. **Billing Integration**: How should usage metering and billing work?
   Options: Stripe Billing, custom metering with Stripe invoicing, AWS
   Marketplace, or all of the above.

3. **Tenant Onboarding**: Should tenant creation be self-service (sign up
   on website) or sales-assisted? Self-service scales better but requires
   abuse prevention (email verification, credit card on file).

4. **Data Residency**: Should we support data residency requirements (e.g.,
   EU data stays in EU region)? This affects database architecture and
   object storage configuration.

5. **Provider Cost Pass-Through**: When tenants use their own provider
   credentials, WorldForge doesn't incur provider costs. When they use
   shared credentials, we do. How should pricing differ?

6. **Free Tier Limits**: What should the free tier include? Proposal:
   100 predictions/month, 10 req/min rate limit, community support only.

7. **SOC 2 Compliance**: Should we pursue SOC 2 Type II certification
   before launch? Enterprise customers often require it, but the audit
   process takes 6-12 months.

8. **Managed vs. Self-Hosted Parity**: Should the cloud service have
   features not available in the self-hosted version (e.g., multi-tenant
   dashboard, usage analytics)? Or should they be feature-equivalent?

9. **Database Selection**: PostgreSQL is proposed for the primary datastore.
   Should we consider alternatives (CockroachDB for multi-region, DynamoDB
   for operational simplicity, Turso/libsql for edge deployment)?

10. **Queue Technology**: Redis is proposed for the job queue. Should we
    consider alternatives (RabbitMQ for more robust queuing semantics,
    SQS for fully managed, NATS for lower latency)?
