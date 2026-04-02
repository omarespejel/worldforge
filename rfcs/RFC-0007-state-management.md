# RFC-0007: Real-Time State Management & Persistence

| Field   | Value                                       |
|---------|---------------------------------------------|
| Title   | Real-Time State Management & Persistence    |
| Status  | Draft                                       |
| Author  | WorldForge Contributors                     |
| Created | 2026-04-02                                  |
| RFC     | 0007                                        |

---

## Abstract

This RFC describes the comprehensive state management and persistence layer for
WorldForge, covering the hardening and scaling of existing file/JSON and SQLite
backends, the introduction of Redis and S3 backends as feature-gated options,
state versioning and migration, concurrent write conflict resolution via
optimistic locking, state compaction and garbage collection, snapshot
export/import, performance benchmarks targeting 10K worlds and 1M state
transitions, and real-time state change notifications via pub/sub. This RFC
transforms WorldForge's state management from a basic persistence layer into a
production-grade, horizontally scalable system.

---

## Motivation

### Current State

WorldForge currently has basic state persistence through:

1. **File/JSON Backend**: Stores world states as JSON files on disk using
   `serde_json`. Simple but doesn't scale beyond single-node, single-writer
   scenarios.

2. **SQLite Backend**: Uses `sqlx 0.8` for structured storage. Better than
   files but limited to single-node and has write contention under concurrent
   access.

### Why Enhance State Management?

As WorldForge scales to production workloads, the state management layer
becomes a critical bottleneck:

1. **Multi-Process/Multi-Node**: Production deployments need multiple WorldForge
   instances sharing state. File/JSON and SQLite don't support this.

2. **High Throughput**: World simulation can generate thousands of state
   transitions per second. Current backends aren't optimized for this.

3. **Data Safety**: No versioning, no conflict resolution, no garbage collection.
   State can become corrupted or bloated over time.

4. **Observability**: No notifications when state changes. Downstream consumers
   must poll for updates.

5. **Portability**: No way to export/import world states between environments.

### Target Scale

| Metric                      | Current Capability | Target     |
|-----------------------------|--------------------|------------|
| Concurrent worlds           | ~100               | 10,000+    |
| State transitions/sec       | ~50                | 10,000+    |
| Total stored transitions    | ~10,000            | 1,000,000+ |
| Concurrent writers          | 1                  | 100+       |
| State query latency (p99)   | ~50ms              | < 5ms      |
| Notification latency        | N/A (no pub/sub)   | < 10ms     |

---

## Detailed Design

### 1. State Backend Trait

The unified interface for all state backends:

```rust
#[async_trait]
pub trait StateBackend: Send + Sync + 'static {
    /// Store a world state, returning the assigned version
    async fn put_state(
        &self,
        world_id: &str,
        state: &WorldState,
        expected_version: Option<u64>,
    ) -> Result<StateVersion, StateError>;

    /// Retrieve the latest state for a world
    async fn get_state(
        &self,
        world_id: &str,
    ) -> Result<Option<VersionedState>, StateError>;

    /// Retrieve a specific version of a world state
    async fn get_state_version(
        &self,
        world_id: &str,
        version: u64,
    ) -> Result<Option<VersionedState>, StateError>;

    /// List all versions for a world (paginated)
    async fn list_versions(
        &self,
        world_id: &str,
        offset: u64,
        limit: u64,
    ) -> Result<Vec<StateVersion>, StateError>;

    /// List all world IDs (paginated)
    async fn list_worlds(
        &self,
        offset: u64,
        limit: u64,
    ) -> Result<Vec<WorldInfo>, StateError>;

    /// Delete a world and all its states
    async fn delete_world(
        &self,
        world_id: &str,
    ) -> Result<(), StateError>;

    /// Delete versions older than a threshold, keeping at least min_versions
    async fn compact(
        &self,
        world_id: &str,
        keep_versions: u64,
    ) -> Result<CompactionResult, StateError>;

    /// Export a world's complete state history as a portable snapshot
    async fn export_snapshot(
        &self,
        world_id: &str,
    ) -> Result<WorldSnapshot, StateError>;

    /// Import a world from a snapshot
    async fn import_snapshot(
        &self,
        snapshot: &WorldSnapshot,
    ) -> Result<(), StateError>;

    /// Subscribe to state changes for a world (or all worlds)
    async fn subscribe(
        &self,
        filter: SubscriptionFilter,
    ) -> Result<StateSubscription, StateError>;

    /// Health check for the backend
    async fn health_check(&self) -> Result<BackendHealth, StateError>;

    /// Run garbage collection across all worlds
    async fn garbage_collect(
        &self,
        policy: &GcPolicy,
    ) -> Result<GcResult, StateError>;
}

#[derive(Debug, Clone)]
pub struct VersionedState {
    pub world_id: String,
    pub state: WorldState,
    pub version: u64,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub checksum: String,
}

#[derive(Debug, Clone)]
pub struct StateVersion {
    pub version: u64,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub size_bytes: u64,
    pub checksum: String,
}
```

### 2. Stress Testing File/JSON Backend

The existing file/JSON backend needs hardening for reliability:

```rust
pub struct FileJsonBackend {
    base_dir: PathBuf,
    /// Per-world file locks to prevent corruption
    locks: Arc<DashMap<String, Arc<RwLock<()>>>>,
    /// Write-ahead log for crash recovery
    wal: WriteAheadLog,
    /// Fsync policy
    fsync: FsyncPolicy,
}

#[derive(Debug, Clone)]
pub enum FsyncPolicy {
    /// Fsync after every write (safest, slowest)
    Always,
    /// Fsync periodically (configurable interval)
    Periodic { interval: Duration },
    /// Never fsync (fastest, risk of data loss)
    Never,
}

impl FileJsonBackend {
    /// Directory structure:
    /// base_dir/
    ///   worlds/
    ///     {world_id}/
    ///       current.json          <- Latest state (symlink or copy)
    ///       versions/
    ///         v000001.json
    ///         v000002.json
    ///       metadata.json         <- World metadata, current version
    ///   wal/
    ///     pending/                <- In-progress writes
    ///     committed/              <- Completed writes (pruned after apply)

    pub fn new(base_dir: PathBuf, fsync: FsyncPolicy) -> Result<Self, StateError> {
        std::fs::create_dir_all(base_dir.join("worlds"))?;
        std::fs::create_dir_all(base_dir.join("wal/pending"))?;
        std::fs::create_dir_all(base_dir.join("wal/committed"))?;

        let wal = WriteAheadLog::new(base_dir.join("wal"))?;

        // Recover any pending WAL entries from crash
        wal.recover()?;

        Ok(Self {
            base_dir,
            locks: Arc::new(DashMap::new()),
            wal,
            fsync,
        })
    }

    fn world_dir(&self, world_id: &str) -> PathBuf {
        self.base_dir.join("worlds").join(world_id)
    }

    fn version_path(&self, world_id: &str, version: u64) -> PathBuf {
        self.world_dir(world_id)
            .join("versions")
            .join(format!("v{:06}.json", version))
    }
}

#[async_trait]
impl StateBackend for FileJsonBackend {
    async fn put_state(
        &self,
        world_id: &str,
        state: &WorldState,
        expected_version: Option<u64>,
    ) -> Result<StateVersion, StateError> {
        let lock = self.locks
            .entry(world_id.to_string())
            .or_insert_with(|| Arc::new(RwLock::new(())))
            .clone();

        let _guard = lock.write().await;

        // Read current version
        let metadata = self.read_metadata(world_id).await?;
        let current_version = metadata.as_ref().map(|m| m.current_version).unwrap_or(0);

        // Optimistic locking check
        if let Some(expected) = expected_version {
            if current_version != expected {
                return Err(StateError::VersionConflict {
                    world_id: world_id.to_string(),
                    expected,
                    actual: current_version,
                });
            }
        }

        let new_version = current_version + 1;

        // Serialize state
        let json = serde_json::to_vec_pretty(state)?;
        let checksum = sha256_hash(&json);

        // Write to WAL first
        let wal_entry = self.wal.write_entry(
            world_id, new_version, &json
        ).await?;

        // Write version file
        let version_path = self.version_path(world_id, new_version);
        tokio::fs::create_dir_all(version_path.parent().unwrap()).await?;
        self.atomic_write(&version_path, &json).await?;

        // Update current.json
        let current_path = self.world_dir(world_id).join("current.json");
        self.atomic_write(&current_path, &json).await?;

        // Update metadata
        self.write_metadata(world_id, &WorldMetadata {
            current_version: new_version,
            total_versions: new_version,
            created_at: metadata.as_ref()
                .map(|m| m.created_at)
                .unwrap_or_else(chrono::Utc::now),
            updated_at: chrono::Utc::now(),
        }).await?;

        // Mark WAL entry as committed
        self.wal.commit_entry(wal_entry).await?;

        // Fsync if needed
        if matches!(self.fsync, FsyncPolicy::Always) {
            self.fsync_dir(&self.world_dir(world_id)).await?;
        }

        Ok(StateVersion {
            version: new_version,
            created_at: chrono::Utc::now(),
            size_bytes: json.len() as u64,
            checksum,
        })
    }

    async fn get_state(
        &self,
        world_id: &str,
    ) -> Result<Option<VersionedState>, StateError> {
        let current_path = self.world_dir(world_id).join("current.json");

        if !current_path.exists() {
            return Ok(None);
        }

        let lock = self.locks
            .entry(world_id.to_string())
            .or_insert_with(|| Arc::new(RwLock::new(())))
            .clone();

        let _guard = lock.read().await;

        let json = tokio::fs::read(&current_path).await?;
        let state: WorldState = serde_json::from_slice(&json)?;
        let metadata = self.read_metadata(world_id).await?
            .ok_or_else(|| StateError::NotFound(world_id.to_string()))?;

        Ok(Some(VersionedState {
            world_id: world_id.to_string(),
            state,
            version: metadata.current_version,
            created_at: metadata.updated_at,
            checksum: sha256_hash(&json),
        }))
    }

    // ... other methods ...
}
```

### 3. Stress Testing SQLite Backend

```rust
pub struct SqliteBackend {
    pool: sqlx::SqlitePool,
    /// Enable WAL mode for better concurrent read performance
    wal_mode: bool,
}

impl SqliteBackend {
    pub async fn new(database_url: &str) -> Result<Self, StateError> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(10)
            .min_connections(2)
            .acquire_timeout(Duration::from_secs(5))
            .idle_timeout(Duration::from_secs(300))
            .connect(database_url)
            .await?;

        let backend = Self {
            pool,
            wal_mode: true,
        };

        backend.initialize().await?;
        Ok(backend)
    }

    async fn initialize(&self) -> Result<(), StateError> {
        // Enable WAL mode for concurrent reads
        if self.wal_mode {
            sqlx::query("PRAGMA journal_mode=WAL")
                .execute(&self.pool)
                .await?;
            sqlx::query("PRAGMA synchronous=NORMAL")
                .execute(&self.pool)
                .await?;
            sqlx::query("PRAGMA wal_autocheckpoint=1000")
                .execute(&self.pool)
                .await?;
        }

        // Create tables
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS worlds (
                id TEXT PRIMARY KEY,
                current_version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        "#).execute(&self.pool).await?;

        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS world_states (
                world_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                state_json BLOB NOT NULL,
                checksum TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (world_id, version),
                FOREIGN KEY (world_id) REFERENCES worlds(id) ON DELETE CASCADE
            )
        "#).execute(&self.pool).await?;

        sqlx::query(r#"
            CREATE INDEX IF NOT EXISTS idx_world_states_world_version
            ON world_states(world_id, version DESC)
        "#).execute(&self.pool).await?;

        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS state_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        "#).execute(&self.pool).await?;

        Ok(())
    }
}

#[async_trait]
impl StateBackend for SqliteBackend {
    async fn put_state(
        &self,
        world_id: &str,
        state: &WorldState,
        expected_version: Option<u64>,
    ) -> Result<StateVersion, StateError> {
        let json = serde_json::to_vec(state)?;
        let checksum = sha256_hash(&json);
        let now = chrono::Utc::now();

        // Use a transaction for atomicity
        let mut tx = self.pool.begin().await?;

        // Get current version with row-level lock
        let current: Option<(i64,)> = sqlx::query_as(
            "SELECT current_version FROM worlds WHERE id = ? FOR UPDATE"
        )
        .bind(world_id)
        .fetch_optional(&mut *tx)
        .await?;

        let current_version = current.map(|r| r.0 as u64).unwrap_or(0);

        // Optimistic locking
        if let Some(expected) = expected_version {
            if current_version != expected {
                tx.rollback().await?;
                return Err(StateError::VersionConflict {
                    world_id: world_id.to_string(),
                    expected,
                    actual: current_version,
                });
            }
        }

        let new_version = current_version + 1;

        // Upsert world record
        sqlx::query(r#"
            INSERT INTO worlds (id, current_version, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                current_version = ?,
                updated_at = ?
        "#)
        .bind(world_id)
        .bind(new_version as i64)
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind(new_version as i64)
        .bind(now.to_rfc3339())
        .execute(&mut *tx)
        .await?;

        // Insert state version
        sqlx::query(r#"
            INSERT INTO world_states (world_id, version, state_json, checksum, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        "#)
        .bind(world_id)
        .bind(new_version as i64)
        .bind(&json)
        .bind(&checksum)
        .bind(json.len() as i64)
        .bind(now.to_rfc3339())
        .execute(&mut *tx)
        .await?;

        // Record notification
        sqlx::query(r#"
            INSERT INTO state_notifications (world_id, version, event_type, created_at)
            VALUES (?, ?, 'state_updated', ?)
        "#)
        .bind(world_id)
        .bind(new_version as i64)
        .bind(now.to_rfc3339())
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;

        Ok(StateVersion {
            version: new_version,
            created_at: now,
            size_bytes: json.len() as u64,
            checksum,
        })
    }

    // ... other methods ...
}
```

### 4. Redis Backend

The Redis backend provides high-performance, distributed state management:

```rust
pub struct RedisBackend {
    pool: RedisPool,
    config: RedisConfig,
    /// Pub/sub connection for notifications
    pubsub: Arc<RwLock<Option<PubSubConnection>>>,
}

pub struct RedisConfig {
    /// Redis connection URL(s)
    pub urls: Vec<String>,
    /// Connection pool size
    pub pool_size: u32,
    /// Minimum idle connections
    pub min_idle: u32,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Command timeout
    pub command_timeout: Duration,
    /// Reconnection strategy
    pub reconnect: ReconnectStrategy,
    /// Key prefix for namespacing
    pub key_prefix: String,
    /// Enable cluster mode
    pub cluster_mode: bool,
    /// TLS configuration
    pub tls: Option<RedisTlsConfig>,
}

#[derive(Debug, Clone)]
pub enum ReconnectStrategy {
    /// Exponential backoff
    ExponentialBackoff {
        initial_delay: Duration,
        max_delay: Duration,
        max_retries: u32,
    },
    /// Fixed interval
    FixedInterval {
        interval: Duration,
        max_retries: u32,
    },
    /// No reconnection
    Never,
}

pub struct RedisPool {
    pool: bb8::Pool<bb8_redis::RedisConnectionManager>,
    health: Arc<AtomicBool>,
}

impl RedisPool {
    pub async fn new(config: &RedisConfig) -> Result<Self, StateError> {
        let manager = bb8_redis::RedisConnectionManager::new(
            config.urls[0].as_str()
        )?;

        let pool = bb8::Pool::builder()
            .max_size(config.pool_size)
            .min_idle(Some(config.min_idle))
            .connection_timeout(config.connect_timeout)
            .build(manager)
            .await?;

        Ok(Self {
            pool,
            health: Arc::new(AtomicBool::new(true)),
        })
    }

    pub async fn get_conn(&self) -> Result<PooledConnection, StateError> {
        self.pool.get().await.map_err(|e| {
            self.health.store(false, Ordering::Relaxed);
            StateError::ConnectionFailed(e.to_string())
        })
    }

    /// Background task to monitor connection health and reconnect
    pub async fn run_health_monitor(&self, config: &RedisConfig) {
        let mut consecutive_failures = 0u32;

        loop {
            tokio::time::sleep(Duration::from_secs(10)).await;

            match self.get_conn().await {
                Ok(mut conn) => {
                    match redis::cmd("PING").query_async::<_, String>(&mut *conn).await {
                        Ok(_) => {
                            self.health.store(true, Ordering::Relaxed);
                            consecutive_failures = 0;
                        }
                        Err(e) => {
                            tracing::warn!("Redis health check failed: {}", e);
                            consecutive_failures += 1;
                            self.health.store(false, Ordering::Relaxed);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Redis connection failed: {}", e);
                    consecutive_failures += 1;
                    self.health.store(false, Ordering::Relaxed);

                    // Apply reconnection strategy
                    self.handle_reconnect(config, consecutive_failures).await;
                }
            }
        }
    }
}
```

#### Redis Data Model

```
Keys:
  {prefix}:world:{world_id}:meta         -> Hash (current_version, created_at, updated_at)
  {prefix}:world:{world_id}:state:latest  -> JSON blob (latest state)
  {prefix}:world:{world_id}:state:{ver}   -> JSON blob (versioned state)
  {prefix}:world:{world_id}:versions      -> Sorted set (version -> timestamp)
  {prefix}:worlds                          -> Set of all world IDs
  {prefix}:channel:state_changes           -> Pub/sub channel
  {prefix}:channel:world:{world_id}        -> Per-world pub/sub channel
```

```rust
#[async_trait]
impl StateBackend for RedisBackend {
    async fn put_state(
        &self,
        world_id: &str,
        state: &WorldState,
        expected_version: Option<u64>,
    ) -> Result<StateVersion, StateError> {
        let json = serde_json::to_vec(state)?;
        let checksum = sha256_hash(&json);
        let now = chrono::Utc::now();

        let mut conn = self.pool.get_conn().await?;

        // Use Lua script for atomic optimistic locking
        let script = redis::Script::new(r#"
            local world_key = KEYS[1]
            local state_key = KEYS[2]
            local latest_key = KEYS[3]
            local versions_key = KEYS[4]
            local worlds_key = KEYS[5]

            local expected_version = tonumber(ARGV[1])
            local state_json = ARGV[2]
            local checksum = ARGV[3]
            local size_bytes = tonumber(ARGV[4])
            local timestamp = ARGV[5]

            -- Get current version
            local current_version = tonumber(redis.call('HGET', world_key, 'current_version') or '0')

            -- Optimistic locking check
            if expected_version ~= -1 and current_version ~= expected_version then
                return {0, current_version}  -- Conflict
            end

            local new_version = current_version + 1

            -- Update world metadata
            redis.call('HSET', world_key,
                'current_version', new_version,
                'updated_at', timestamp)
            if current_version == 0 then
                redis.call('HSET', world_key, 'created_at', timestamp)
            end

            -- Store versioned state
            redis.call('SET', state_key .. ':' .. new_version, state_json)

            -- Update latest pointer
            redis.call('SET', latest_key, state_json)

            -- Add to version sorted set
            redis.call('ZADD', versions_key, timestamp, new_version)

            -- Add to worlds set
            redis.call('SADD', worlds_key, KEYS[6])

            -- Publish notification
            local notification = cjson.encode({
                world_id = KEYS[6],
                version = new_version,
                event = 'state_updated',
                timestamp = timestamp
            })
            redis.call('PUBLISH', KEYS[7], notification)
            redis.call('PUBLISH', KEYS[8], notification)

            return {1, new_version}
        "#);

        let prefix = &self.config.key_prefix;
        let expected = expected_version.map(|v| v as i64).unwrap_or(-1);

        let result: Vec<i64> = script
            .key(format!("{}:world:{}:meta", prefix, world_id))
            .key(format!("{}:world:{}:state", prefix, world_id))
            .key(format!("{}:world:{}:state:latest", prefix, world_id))
            .key(format!("{}:world:{}:versions", prefix, world_id))
            .key(format!("{}:worlds", prefix))
            .key(world_id)
            .key(format!("{}:channel:state_changes", prefix))
            .key(format!("{}:channel:world:{}", prefix, world_id))
            .arg(expected)
            .arg(&json)
            .arg(&checksum)
            .arg(json.len() as i64)
            .arg(now.timestamp())
            .invoke_async(&mut *conn)
            .await?;

        if result[0] == 0 {
            return Err(StateError::VersionConflict {
                world_id: world_id.to_string(),
                expected: expected_version.unwrap_or(0),
                actual: result[1] as u64,
            });
        }

        Ok(StateVersion {
            version: result[1] as u64,
            created_at: now,
            size_bytes: json.len() as u64,
            checksum,
        })
    }

    async fn subscribe(
        &self,
        filter: SubscriptionFilter,
    ) -> Result<StateSubscription, StateError> {
        let mut pubsub_conn = self.pool.get_pubsub_conn().await?;
        let prefix = &self.config.key_prefix;

        match filter {
            SubscriptionFilter::AllWorlds => {
                pubsub_conn.subscribe(
                    format!("{}:channel:state_changes", prefix)
                ).await?;
            }
            SubscriptionFilter::World(world_id) => {
                pubsub_conn.subscribe(
                    format!("{}:channel:world:{}", prefix, world_id)
                ).await?;
            }
        }

        let (tx, rx) = tokio::sync::mpsc::channel(1000);

        tokio::spawn(async move {
            let mut stream = pubsub_conn.on_message();
            while let Some(msg) = stream.next().await {
                if let Ok(payload) = msg.get_payload::<String>() {
                    if let Ok(notification) = serde_json::from_str::<StateNotification>(&payload) {
                        if tx.send(notification).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                }
            }
        });

        Ok(StateSubscription {
            receiver: rx,
        })
    }

    // ... other methods ...
}
```

### 5. S3 Backend

The S3 backend provides durable, scalable storage for world states:

```rust
pub struct S3Backend {
    client: aws_sdk_s3::Client,
    config: S3Config,
    /// Local cache for frequently accessed states
    cache: Arc<Mutex<LruCache<String, Vec<u8>>>>,
}

pub struct S3Config {
    /// S3 bucket name
    pub bucket: String,
    /// Key prefix within the bucket
    pub key_prefix: String,
    /// AWS region
    pub region: String,
    /// Custom endpoint URL (for MinIO, LocalStack)
    pub endpoint_url: Option<String>,
    /// Force path-style addressing (needed for MinIO)
    pub force_path_style: bool,
    /// Local cache size (MB)
    pub cache_size_mb: u64,
    /// Enable server-side encryption
    pub server_side_encryption: Option<S3Encryption>,
    /// Storage class for versioned states
    pub version_storage_class: S3StorageClass,
}

#[derive(Debug, Clone)]
pub enum S3Encryption {
    Sse,           // SSE-S3 (AES-256)
    SseKms(String), // SSE-KMS with key ARN
}

#[derive(Debug, Clone)]
pub enum S3StorageClass {
    Standard,
    StandardIa,       // Infrequent Access
    IntelligentTiering,
    Glacier,          // For very old versions
}

impl S3Backend {
    pub async fn new(config: S3Config) -> Result<Self, StateError> {
        let mut s3_config = aws_config::from_env()
            .region(aws_config::Region::new(config.region.clone()))
            .load()
            .await;

        let mut s3_builder = aws_sdk_s3::config::Builder::from(&s3_config);

        if let Some(endpoint) = &config.endpoint_url {
            s3_builder = s3_builder.endpoint_url(endpoint);
        }

        if config.force_path_style {
            s3_builder = s3_builder.force_path_style(true);
        }

        let client = aws_sdk_s3::Client::from_conf(s3_builder.build());

        // Verify bucket access
        client.head_bucket()
            .bucket(&config.bucket)
            .send()
            .await
            .map_err(|e| StateError::ConnectionFailed(
                format!("Cannot access S3 bucket {}: {}", config.bucket, e)
            ))?;

        let cache_entries = (config.cache_size_mb * 1024 * 1024 / 65536) as usize;

        Ok(Self {
            client,
            config,
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_entries.max(1)).unwrap()
            ))),
        })
    }

    fn state_key(&self, world_id: &str, version: u64) -> String {
        format!(
            "{}/worlds/{}/versions/v{:010}.json",
            self.config.key_prefix, world_id, version
        )
    }

    fn latest_key(&self, world_id: &str) -> String {
        format!(
            "{}/worlds/{}/latest.json",
            self.config.key_prefix, world_id
        )
    }

    fn metadata_key(&self, world_id: &str) -> String {
        format!(
            "{}/worlds/{}/metadata.json",
            self.config.key_prefix, world_id
        )
    }
}

#[async_trait]
impl StateBackend for S3Backend {
    async fn put_state(
        &self,
        world_id: &str,
        state: &WorldState,
        expected_version: Option<u64>,
    ) -> Result<StateVersion, StateError> {
        let json = serde_json::to_vec(state)?;
        let checksum = sha256_hash(&json);
        let now = chrono::Utc::now();

        // Read current metadata for version check
        let metadata = self.read_metadata(world_id).await?;
        let current_version = metadata.as_ref()
            .map(|m| m.current_version)
            .unwrap_or(0);

        // Optimistic locking
        if let Some(expected) = expected_version {
            if current_version != expected {
                return Err(StateError::VersionConflict {
                    world_id: world_id.to_string(),
                    expected,
                    actual: current_version,
                });
            }
        }

        let new_version = current_version + 1;

        // Upload versioned state
        let version_key = self.state_key(world_id, new_version);
        let mut put_req = self.client.put_object()
            .bucket(&self.config.bucket)
            .key(&version_key)
            .body(json.clone().into())
            .content_type("application/json")
            .metadata("worldforge-version", &new_version.to_string())
            .metadata("worldforge-checksum", &checksum);

        // Apply storage class
        put_req = match &self.config.version_storage_class {
            S3StorageClass::Standard => put_req,
            S3StorageClass::StandardIa => {
                put_req.storage_class(aws_sdk_s3::types::StorageClass::StandardIa)
            }
            S3StorageClass::IntelligentTiering => {
                put_req.storage_class(aws_sdk_s3::types::StorageClass::IntelligentTiering)
            }
            S3StorageClass::Glacier => {
                put_req.storage_class(aws_sdk_s3::types::StorageClass::Glacier)
            }
        };

        // Apply encryption
        if let Some(encryption) = &self.config.server_side_encryption {
            put_req = match encryption {
                S3Encryption::Sse => {
                    put_req.server_side_encryption(
                        aws_sdk_s3::types::ServerSideEncryption::Aes256
                    )
                }
                S3Encryption::SseKms(key_arn) => {
                    put_req
                        .server_side_encryption(
                            aws_sdk_s3::types::ServerSideEncryption::AwsKms
                        )
                        .ssekms_key_id(key_arn)
                }
            };
        }

        put_req.send().await?;

        // Update latest pointer
        self.client.put_object()
            .bucket(&self.config.bucket)
            .key(&self.latest_key(world_id))
            .body(json.clone().into())
            .content_type("application/json")
            .send()
            .await?;

        // Update metadata
        let new_metadata = WorldMetadata {
            current_version: new_version,
            total_versions: new_version,
            created_at: metadata.as_ref()
                .map(|m| m.created_at)
                .unwrap_or(now),
            updated_at: now,
        };

        self.write_metadata(world_id, &new_metadata).await?;

        // Update local cache
        {
            let cache_key = format!("{}:latest", world_id);
            self.cache.lock().unwrap().put(cache_key, json.clone());
        }

        Ok(StateVersion {
            version: new_version,
            created_at: now,
            size_bytes: json.len() as u64,
            checksum,
        })
    }

    // ... other methods ...
}
```

### 6. State Versioning and Migration

```rust
pub struct StateMigration {
    /// Source schema version
    pub from_version: u32,
    /// Target schema version
    pub to_version: u32,
    /// Migration function
    pub migrate: Box<dyn Fn(serde_json::Value) -> Result<serde_json::Value, StateError> + Send + Sync>,
    /// Description of the migration
    pub description: String,
}

pub struct MigrationRegistry {
    migrations: Vec<StateMigration>,
}

impl MigrationRegistry {
    pub fn new() -> Self {
        let mut registry = Self { migrations: Vec::new() };

        // Register known migrations
        registry.register(StateMigration {
            from_version: 1,
            to_version: 2,
            description: "Add latent_representation field".to_string(),
            migrate: Box::new(|mut state| {
                if let Some(obj) = state.as_object_mut() {
                    obj.entry("latent_representation")
                        .or_insert(serde_json::Value::Null);
                }
                Ok(state)
            }),
        });

        registry.register(StateMigration {
            from_version: 2,
            to_version: 3,
            description: "Rename timestamp to created_at, add updated_at".to_string(),
            migrate: Box::new(|mut state| {
                if let Some(obj) = state.as_object_mut() {
                    if let Some(ts) = obj.remove("timestamp") {
                        obj.insert("created_at".to_string(), ts.clone());
                        obj.insert("updated_at".to_string(), ts);
                    }
                }
                Ok(state)
            }),
        });

        registry
    }

    pub fn migrate(
        &self,
        state: serde_json::Value,
        from: u32,
        to: u32,
    ) -> Result<serde_json::Value, StateError> {
        let mut current = state;
        let mut current_version = from;

        while current_version < to {
            let migration = self.migrations.iter()
                .find(|m| m.from_version == current_version)
                .ok_or_else(|| StateError::MigrationNotFound {
                    from: current_version,
                    to: current_version + 1,
                })?;

            current = (migration.migrate)(current)?;
            current_version = migration.to_version;
        }

        Ok(current)
    }
}
```

### 7. Concurrent Write Conflict Resolution (Optimistic Locking)

```rust
/// Retry helper for optimistic locking conflicts
pub async fn with_optimistic_retry<F, Fut, T>(
    max_retries: u32,
    backoff: Duration,
    mut operation: F,
) -> Result<T, StateError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, StateError>>,
{
    let mut attempts = 0;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(StateError::VersionConflict { .. }) if attempts < max_retries => {
                attempts += 1;
                let jitter = rand::thread_rng().gen_range(0..backoff.as_millis() as u64);
                tokio::time::sleep(backoff * attempts + Duration::from_millis(jitter)).await;
                tracing::debug!("Optimistic lock conflict, retry {} of {}", attempts, max_retries);
            }
            Err(e) => return Err(e),
        }
    }
}
```

### 8. State Compaction and Garbage Collection

```rust
#[derive(Debug, Clone)]
pub struct GcPolicy {
    /// Keep at least this many versions per world
    pub min_versions_to_keep: u64,
    /// Delete versions older than this duration
    pub max_version_age: Option<Duration>,
    /// Maximum total storage per world (bytes)
    pub max_storage_per_world: Option<u64>,
    /// Maximum total storage across all worlds (bytes)
    pub max_total_storage: Option<u64>,
    /// Dry run (report what would be deleted without actually deleting)
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct GcResult {
    pub worlds_processed: u64,
    pub versions_deleted: u64,
    pub bytes_freed: u64,
    pub duration: Duration,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub versions_before: u64,
    pub versions_after: u64,
    pub bytes_freed: u64,
}

pub struct GarbageCollector;

impl GarbageCollector {
    pub async fn run(
        backend: &dyn StateBackend,
        policy: &GcPolicy,
    ) -> Result<GcResult, StateError> {
        let start = Instant::now();
        let mut result = GcResult {
            worlds_processed: 0,
            versions_deleted: 0,
            bytes_freed: 0,
            duration: Duration::ZERO,
            errors: Vec::new(),
        };

        let mut offset = 0;
        let page_size = 100;

        loop {
            let worlds = backend.list_worlds(offset, page_size).await?;
            if worlds.is_empty() {
                break;
            }

            for world in &worlds {
                match backend.compact(
                    &world.id,
                    policy.min_versions_to_keep,
                ).await {
                    Ok(compaction) => {
                        result.versions_deleted += compaction.versions_before - compaction.versions_after;
                        result.bytes_freed += compaction.bytes_freed;
                    }
                    Err(e) => {
                        result.errors.push(format!(
                            "World {}: {}", world.id, e
                        ));
                    }
                }
                result.worlds_processed += 1;
            }

            offset += page_size;
        }

        result.duration = start.elapsed();
        Ok(result)
    }
}
```

### 9. Snapshot Export/Import

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSnapshot {
    /// Snapshot format version
    pub format_version: u32,
    /// World ID
    pub world_id: String,
    /// All state versions
    pub states: Vec<SnapshotEntry>,
    /// World metadata
    pub metadata: WorldMetadata,
    /// Export timestamp
    pub exported_at: chrono::DateTime<chrono::Utc>,
    /// Checksum of the entire snapshot
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotEntry {
    pub version: u64,
    pub state: WorldState,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub checksum: String,
}

pub struct SnapshotManager;

impl SnapshotManager {
    /// Export a world to a compressed snapshot file
    pub async fn export_to_file(
        backend: &dyn StateBackend,
        world_id: &str,
        output_path: &Path,
    ) -> Result<(), StateError> {
        let snapshot = backend.export_snapshot(world_id).await?;

        let json = serde_json::to_vec(&snapshot)?;

        // Compress with zstd
        let compressed = zstd::encode_all(&json[..], 3)?;

        tokio::fs::write(output_path, &compressed).await?;

        tracing::info!(
            "Exported world {} ({} versions, {} bytes compressed)",
            world_id,
            snapshot.states.len(),
            compressed.len(),
        );

        Ok(())
    }

    /// Import a world from a compressed snapshot file
    pub async fn import_from_file(
        backend: &dyn StateBackend,
        input_path: &Path,
    ) -> Result<String, StateError> {
        let compressed = tokio::fs::read(input_path).await?;

        // Decompress
        let json = zstd::decode_all(&compressed[..])?;

        let snapshot: WorldSnapshot = serde_json::from_slice(&json)?;

        // Verify checksum
        let computed_checksum = compute_snapshot_checksum(&snapshot);
        if computed_checksum != snapshot.checksum {
            return Err(StateError::ChecksumMismatch {
                expected: snapshot.checksum.clone(),
                actual: computed_checksum,
            });
        }

        let world_id = snapshot.world_id.clone();
        backend.import_snapshot(&snapshot).await?;

        tracing::info!(
            "Imported world {} ({} versions)",
            world_id,
            snapshot.states.len(),
        );

        Ok(world_id)
    }
}
```

### 10. Performance Benchmarks

Target benchmark scenarios:

```rust
#[cfg(test)]
mod benchmarks {
    use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

    fn bench_put_state(c: &mut Criterion) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut group = c.benchmark_group("put_state");

        for backend_name in &["file_json", "sqlite", "redis"] {
            group.bench_with_input(
                BenchmarkId::new("sequential", backend_name),
                backend_name,
                |b, name| {
                    b.to_async(&rt).iter(|| async {
                        let backend = create_backend(name).await;
                        let state = create_test_state();
                        backend.put_state("bench-world", &state, None).await.unwrap();
                    });
                },
            );
        }

        group.finish();
    }

    fn bench_get_state(c: &mut Criterion) {
        // Pre-populate and then benchmark reads
    }

    fn bench_concurrent_writes(c: &mut Criterion) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut group = c.benchmark_group("concurrent_writes");

        for num_writers in &[1, 10, 50, 100] {
            group.bench_with_input(
                BenchmarkId::new("writers", num_writers),
                num_writers,
                |b, &num_writers| {
                    b.to_async(&rt).iter(|| async {
                        let backend = Arc::new(create_backend("sqlite").await);
                        let mut handles = Vec::new();

                        for i in 0..num_writers {
                            let backend = backend.clone();
                            handles.push(tokio::spawn(async move {
                                let state = create_test_state();
                                backend.put_state(
                                    &format!("world-{}", i),
                                    &state,
                                    None,
                                ).await.unwrap();
                            }));
                        }

                        for handle in handles {
                            handle.await.unwrap();
                        }
                    });
                },
            );
        }

        group.finish();
    }

    fn bench_10k_worlds(c: &mut Criterion) {
        // Create 10K worlds, benchmark list and random access
    }

    fn bench_1m_transitions(c: &mut Criterion) {
        // Write 1M transitions to a single world, benchmark read and compaction
    }
}
```

#### Performance Targets

| Operation                     | File/JSON | SQLite  | Redis   | S3      |
|-------------------------------|-----------|---------|---------|---------|
| Put state (single)            | < 5ms     | < 2ms   | < 1ms   | < 50ms  |
| Get latest state              | < 2ms     | < 1ms   | < 0.5ms | < 30ms  |
| Get versioned state           | < 3ms     | < 1ms   | < 0.5ms | < 30ms  |
| List worlds (10K)             | < 100ms   | < 10ms  | < 5ms   | < 200ms |
| Concurrent writes (100)       | N/A       | < 50ms  | < 10ms  | < 500ms |
| Compaction (1K versions)      | < 500ms   | < 100ms | < 50ms  | < 2s    |
| Snapshot export (1K versions) | < 1s      | < 500ms | < 300ms | < 5s    |
| Pub/sub notification latency  | N/A       | N/A     | < 5ms   | N/A     |

### 11. State Change Notifications (Pub/Sub)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateNotification {
    pub world_id: String,
    pub version: u64,
    pub event: StateEvent,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateEvent {
    StateUpdated,
    WorldCreated,
    WorldDeleted,
    CompactionCompleted { versions_removed: u64 },
    SnapshotExported,
    SnapshotImported,
}

pub enum SubscriptionFilter {
    AllWorlds,
    World(String),
}

pub struct StateSubscription {
    pub receiver: tokio::sync::mpsc::Receiver<StateNotification>,
}

impl StateSubscription {
    pub async fn next(&mut self) -> Option<StateNotification> {
        self.receiver.recv().await
    }

    pub fn try_next(&mut self) -> Option<StateNotification> {
        self.receiver.try_recv().ok()
    }
}

/// In-process pub/sub for backends that don't have native pub/sub
pub struct InProcessPubSub {
    subscribers: Arc<RwLock<Vec<tokio::sync::mpsc::Sender<StateNotification>>>>,
}

impl InProcessPubSub {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn publish(&self, notification: StateNotification) {
        let subscribers = self.subscribers.read().await;
        let mut dead_indices = Vec::new();

        for (i, sub) in subscribers.iter().enumerate() {
            if sub.send(notification.clone()).await.is_err() {
                dead_indices.push(i);
            }
        }

        // Clean up dead subscribers
        if !dead_indices.is_empty() {
            drop(subscribers);
            let mut subs = self.subscribers.write().await;
            for i in dead_indices.into_iter().rev() {
                subs.swap_remove(i);
            }
        }
    }

    pub async fn subscribe(&self) -> StateSubscription {
        let (tx, rx) = tokio::sync::mpsc::channel(1000);
        self.subscribers.write().await.push(tx);
        StateSubscription { receiver: rx }
    }
}
```

---

## Implementation Plan

### Phase 1: Backend Trait & File/JSON Hardening (Week 1-2)

1. Define unified `StateBackend` trait
2. Add WAL (write-ahead log) to file backend
3. Implement atomic writes with temp files
4. Add fsync policies
5. Implement optimistic locking for file backend
6. Add version management

### Phase 2: SQLite Hardening (Week 3-4)

1. Enable WAL mode and tune pragmas
2. Implement transactional optimistic locking
3. Add connection pool tuning
4. Implement compaction via SQL DELETE
5. Add notification table and polling

### Phase 3: Redis Backend (Week 5-7)

1. Implement connection pooling with bb8-redis
2. Implement health monitoring and reconnection
3. Implement Lua-scripted atomic operations
4. Add Redis pub/sub for notifications
5. Implement cluster mode support
6. Add TLS support

### Phase 4: S3 Backend (Week 8-9)

1. Implement aws-sdk-s3 integration
2. Add local LRU caching layer
3. Implement storage class tiering
4. Add server-side encryption support
5. Test with real S3 and LocalStack

### Phase 5: Cross-Cutting Concerns (Week 10-12)

1. Implement state migration framework
2. Implement snapshot export/import with compression
3. Implement garbage collector
4. Add in-process pub/sub for non-Redis backends
5. Performance benchmarking with criterion
6. Load testing at 10K worlds / 1M transitions

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Test suite that runs against all backends
    async fn test_backend_contract(backend: &dyn StateBackend) {
        // Test put_state
        let state = create_test_state();
        let v1 = backend.put_state("world-1", &state, None).await.unwrap();
        assert_eq!(v1.version, 1);

        // Test get_state
        let retrieved = backend.get_state("world-1").await.unwrap().unwrap();
        assert_eq!(retrieved.version, 1);

        // Test optimistic locking
        let result = backend.put_state("world-1", &state, Some(0)).await;
        assert!(matches!(result, Err(StateError::VersionConflict { .. })));

        // Test successful optimistic lock
        let v2 = backend.put_state("world-1", &state, Some(1)).await.unwrap();
        assert_eq!(v2.version, 2);

        // Test version history
        let versions = backend.list_versions("world-1", 0, 100).await.unwrap();
        assert_eq!(versions.len(), 2);

        // Test compaction
        let compaction = backend.compact("world-1", 1).await.unwrap();
        assert_eq!(compaction.versions_after, 1);

        // Test snapshot export/import
        let snapshot = backend.export_snapshot("world-1").await.unwrap();
        backend.delete_world("world-1").await.unwrap();
        assert!(backend.get_state("world-1").await.unwrap().is_none());
        backend.import_snapshot(&snapshot).await.unwrap();
        assert!(backend.get_state("world-1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_file_json_backend() {
        let dir = tempdir::TempDir::new("worldforge-test").unwrap();
        let backend = FileJsonBackend::new(dir.path().to_path_buf(), FsyncPolicy::Never).unwrap();
        test_backend_contract(&backend).await;
    }

    #[tokio::test]
    async fn test_sqlite_backend() {
        let backend = SqliteBackend::new("sqlite::memory:").await.unwrap();
        test_backend_contract(&backend).await;
    }

    #[tokio::test]
    async fn test_concurrent_writes_sqlite() {
        let backend = Arc::new(SqliteBackend::new("sqlite::memory:").await.unwrap());
        let mut handles = Vec::new();

        for i in 0..50 {
            let backend = backend.clone();
            handles.push(tokio::spawn(async move {
                let state = create_test_state();
                backend.put_state(
                    &format!("world-{}", i % 10),
                    &state,
                    None,
                ).await
            }));
        }

        let results: Vec<_> = futures::future::join_all(handles).await;
        let successes = results.iter().filter(|r| r.as_ref().unwrap().is_ok()).count();
        assert!(successes > 0);
    }

    #[tokio::test]
    async fn test_migration_v1_to_v3() {
        let registry = MigrationRegistry::new();
        let v1_state = json!({
            "id": "test",
            "timestamp": 1000
        });

        let v3_state = registry.migrate(v1_state, 1, 3).unwrap();
        assert!(v3_state.get("created_at").is_some());
        assert!(v3_state.get("updated_at").is_some());
        assert!(v3_state.get("latent_representation").is_some());
    }
}
```

### Integration Tests (Redis)

```rust
#[cfg(feature = "redis-integration-tests")]
mod redis_tests {
    #[tokio::test]
    #[ignore = "Requires running Redis instance"]
    async fn test_redis_backend() {
        let config = RedisConfig {
            urls: vec!["redis://127.0.0.1:6379".to_string()],
            pool_size: 5,
            key_prefix: format!("worldforge-test-{}", Uuid::new_v4()),
            ..Default::default()
        };

        let backend = RedisBackend::new(config).await.unwrap();
        test_backend_contract(&backend).await;

        // Test pub/sub
        let mut sub = backend.subscribe(SubscriptionFilter::AllWorlds).await.unwrap();
        let state = create_test_state();
        backend.put_state("pub-sub-test", &state, None).await.unwrap();

        let notification = tokio::time::timeout(
            Duration::from_secs(5),
            sub.next(),
        ).await.unwrap().unwrap();

        assert_eq!(notification.world_id, "pub-sub-test");
        assert!(matches!(notification.event, StateEvent::StateUpdated));
    }
}
```

### Integration Tests (S3)

```rust
#[cfg(feature = "s3-integration-tests")]
mod s3_tests {
    #[tokio::test]
    #[ignore = "Requires S3 or LocalStack"]
    async fn test_s3_backend() {
        let config = S3Config {
            bucket: std::env::var("TEST_S3_BUCKET")
                .unwrap_or("worldforge-test".to_string()),
            key_prefix: format!("test/{}", Uuid::new_v4()),
            region: "us-east-1".to_string(),
            endpoint_url: std::env::var("S3_ENDPOINT_URL").ok(),
            force_path_style: true, // for LocalStack
            ..Default::default()
        };

        let backend = S3Backend::new(config).await.unwrap();
        test_backend_contract(&backend).await;
    }
}
```

### Load Tests

```rust
#[cfg(feature = "load-tests")]
mod load_tests {
    #[tokio::test]
    #[ignore = "Long-running load test"]
    async fn test_10k_worlds() {
        let backend = create_test_backend().await;
        let start = Instant::now();

        // Create 10K worlds
        let mut handles = Vec::new();
        for i in 0..10_000 {
            let backend = backend.clone();
            handles.push(tokio::spawn(async move {
                let state = create_test_state();
                backend.put_state(
                    &format!("world-{:05}", i),
                    &state,
                    None,
                ).await.unwrap();
            }));
        }

        futures::future::join_all(handles).await;
        let create_duration = start.elapsed();
        println!("Created 10K worlds in {:?}", create_duration);

        // List all worlds
        let start = Instant::now();
        let worlds = backend.list_worlds(0, 10_000).await.unwrap();
        let list_duration = start.elapsed();
        assert_eq!(worlds.len(), 10_000);
        println!("Listed 10K worlds in {:?}", list_duration);
    }

    #[tokio::test]
    #[ignore = "Long-running load test"]
    async fn test_1m_transitions() {
        let backend = create_test_backend().await;
        let start = Instant::now();

        // Write 1M transitions to one world
        for i in 0..1_000_000 {
            let state = create_test_state_with_index(i);
            backend.put_state("heavy-world", &state, None).await.unwrap();

            if i % 100_000 == 0 {
                println!("Written {} transitions in {:?}", i, start.elapsed());
            }
        }

        let write_duration = start.elapsed();
        println!("Wrote 1M transitions in {:?}", write_duration);

        // Read latest
        let start = Instant::now();
        let state = backend.get_state("heavy-world").await.unwrap().unwrap();
        let read_duration = start.elapsed();
        assert_eq!(state.version, 1_000_000);
        println!("Read latest state in {:?}", read_duration);

        // Compact to last 100 versions
        let start = Instant::now();
        let result = backend.compact("heavy-world", 100).await.unwrap();
        let compact_duration = start.elapsed();
        println!(
            "Compacted from {} to {} versions in {:?}, freed {} bytes",
            result.versions_before,
            result.versions_after,
            compact_duration,
            result.bytes_freed,
        );
    }
}
```

---

## Open Questions

1. **Redis Cluster vs Sentinel**: Should we support both Redis Cluster and
   Redis Sentinel for high availability? Which should be the default?

2. **S3 Consistency**: S3 now provides strong read-after-write consistency.
   Do we still need a DynamoDB-based locking layer for concurrent writes?

3. **State Compression**: Should we compress state JSON before storage?
   zstd compression could reduce storage by 5-10x.

4. **Time-Series Optimization**: For backends storing millions of transitions,
   should we use a time-series-optimized storage format instead of individual
   JSON documents?

5. **Cross-Backend Migration**: Should we provide tools for migrating data
   between backends (e.g., SQLite → Redis)?

6. **Encryption at Rest**: Beyond S3's SSE, should we implement client-side
   encryption for the file and SQLite backends?

7. **Multi-Tenancy**: Should the state backend support multiple tenants with
   isolated storage and access control?

8. **DynamoDB Backend**: Should we add a DynamoDB backend for AWS-native
   deployments that don't want to manage Redis?

9. **Event Sourcing**: Should we fully embrace event sourcing, where states
   are derived from a log of events rather than stored directly?

10. **Write Batching**: For high-throughput scenarios, should we batch multiple
    state writes into single backend operations?
