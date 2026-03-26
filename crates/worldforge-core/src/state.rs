//! State persistence for WorldForge worlds.
//!
//! Provides the `StateStore` trait and built-in file/SQLite
//! implementations for saving and loading world state.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::bootstrap::seed_world_state_from_prompt;
use crate::error::{Result, WorldForgeError};
use crate::scene::SceneGraph;
use crate::types::{SimTime, WorldId};

const SHA256_INITIAL_STATE: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Complete state of a world instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    /// Unique identifier for this world.
    pub id: WorldId,
    /// Current simulation time.
    pub time: SimTime,
    /// Spatial scene representation.
    pub scene: SceneGraph,
    /// History of past states and actions.
    pub history: StateHistory,
    /// Metadata about the world.
    pub metadata: WorldMetadata,
}

/// Metadata describing a world instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldMetadata {
    /// Human-readable name.
    pub name: String,
    /// Description or creation prompt.
    pub description: String,
    /// Provider used to create the world.
    pub created_by: String,
    /// Timestamp of creation.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Tags for categorization.
    pub tags: Vec<String>,
}

/// Rolling history of state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateHistory {
    /// History entries in chronological order.
    pub states: VecDeque<HistoryEntry>,
    /// Maximum number of entries to keep.
    pub max_entries: usize,
    /// Compression mode for stored states.
    pub compression: Compression,
}

/// A single entry in the state history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Simulation time of this entry.
    pub time: SimTime,
    /// SHA-256 fingerprint of the serialized state snapshot.
    pub state_hash: [u8; 32],
    /// Action that caused this transition (if any).
    pub action: Option<Action>,
    /// Summary of the prediction (if any).
    pub prediction: Option<PredictionSummary>,
    /// Provider that generated this state.
    pub provider: String,
}

/// Lightweight summary of a prediction for history storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionSummary {
    /// Confidence score.
    pub confidence: f32,
    /// Overall physics score.
    pub physics_score: f32,
    /// Latency in milliseconds.
    pub latency_ms: u64,
}

/// Compression mode for state history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    /// No compression.
    #[default]
    None,
    /// LZ4 compression.
    Lz4,
    /// Zstandard compression.
    Zstd,
}

impl Default for StateHistory {
    fn default() -> Self {
        Self {
            states: VecDeque::new(),
            max_entries: 1000,
            compression: Compression::None,
        }
    }
}

impl StateHistory {
    /// Add an entry, evicting the oldest if at capacity.
    pub fn push(&mut self, entry: HistoryEntry) {
        if self.states.len() >= self.max_entries {
            self.states.pop_front();
        }
        self.states.push_back(entry);
    }

    /// Get the most recent entry.
    pub fn latest(&self) -> Option<&HistoryEntry> {
        self.states.back()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl WorldState {
    /// Create a new world state with default settings.
    pub fn new(name: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            time: SimTime::default(),
            scene: SceneGraph::new(),
            history: StateHistory::default(),
            metadata: WorldMetadata {
                name: name.into(),
                description: String::new(),
                created_by: provider.into(),
                created_at: chrono::Utc::now(),
                tags: Vec::new(),
            },
        }
    }

    /// Create a new world state seeded from a natural-language prompt.
    ///
    /// The resulting state stores the prompt in metadata, materializes a
    /// deterministic starter scene, and records the initial snapshot in
    /// history.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` when the prompt is empty.
    pub fn from_prompt(
        prompt: &str,
        provider: impl Into<String>,
        name_override: Option<&str>,
    ) -> Result<Self> {
        let provider = provider.into();
        seed_world_state_from_prompt(prompt, &provider, name_override)
    }

    /// Return the provider most likely responsible for the current state snapshot.
    pub fn current_state_provider(&self) -> String {
        self.history
            .latest()
            .map(|entry| entry.provider.clone())
            .filter(|provider| !provider.is_empty())
            .unwrap_or_else(|| self.metadata.created_by.clone())
    }

    /// Record the current state as a history checkpoint.
    pub fn record_current_state(
        &mut self,
        action: Option<Action>,
        prediction: Option<PredictionSummary>,
        provider: impl Into<String>,
    ) -> Result<()> {
        let state_hash = canonical_state_hash(self)?;
        self.history.push(HistoryEntry {
            time: self.time,
            state_hash,
            action,
            prediction,
            provider: provider.into(),
        });
        Ok(())
    }

    /// Ensure the initial snapshot exists in history.
    pub fn ensure_history_initialized(&mut self, provider: impl Into<String>) -> Result<bool> {
        if self.history.is_empty() {
            self.record_current_state(None, None, provider)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Ensure the current state matches the latest history entry.
    ///
    /// This repairs legacy states whose last checkpoint was recorded with the
    /// previous non-canonical hash format before a new provider transition is appended.
    pub fn ensure_current_state_recorded(&mut self, provider: impl Into<String>) -> Result<bool> {
        if current_state_matches_latest_history(self)? {
            return Ok(false);
        }

        self.record_current_state(None, None, provider)?;
        Ok(true)
    }
}

/// Compute the SHA-256 hash of a byte slice.
pub fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let mut message = data.to_vec();
    let bit_len = (message.len() as u64) * 8;
    message.push(0x80);
    while (message.len() % 64) != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut hash = SHA256_INITIAL_STATE;
    let mut schedule = [0u32; 64];

    for chunk in message.chunks(64) {
        for (index, word) in schedule.iter_mut().take(16).enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }

        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let mut a = hash[0];
        let mut b = hash[1];
        let mut c = hash[2];
        let mut d = hash[3];
        let mut e = hash[4];
        let mut f = hash[5];
        let mut g = hash[6];
        let mut h = hash[7];

        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_ROUND_CONSTANTS[index])
                .wrapping_add(schedule[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        hash[0] = hash[0].wrapping_add(a);
        hash[1] = hash[1].wrapping_add(b);
        hash[2] = hash[2].wrapping_add(c);
        hash[3] = hash[3].wrapping_add(d);
        hash[4] = hash[4].wrapping_add(e);
        hash[5] = hash[5].wrapping_add(f);
        hash[6] = hash[6].wrapping_add(g);
        hash[7] = hash[7].wrapping_add(h);
    }

    let mut output = [0u8; 32];
    for (index, word) in hash.iter().enumerate() {
        output[index * 4..(index + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    output
}

/// Compute the canonical SHA-256 hash for a serialized world-state snapshot.
pub fn canonical_state_hash(state: &WorldState) -> Result<[u8; 32]> {
    let bytes = serde_json::to_vec(state)
        .map_err(|error| WorldForgeError::SerializationError(error.to_string()))?;
    Ok(sha256_hash(&bytes))
}

fn current_state_matches_latest_history(state: &WorldState) -> Result<bool> {
    let Some(latest) = state.history.latest() else {
        return Ok(false);
    };

    let mut snapshot = state.clone();
    snapshot.history.states.pop_back();
    Ok(snapshot.time == latest.time && canonical_state_hash(&snapshot)? == latest.state_hash)
}

/// Trait for persisting world state.
#[async_trait::async_trait]
pub trait StateStore: Send + Sync {
    /// Save world state to the store.
    async fn save(&self, state: &WorldState) -> Result<()>;

    /// Load world state by ID.
    async fn load(&self, id: &WorldId) -> Result<WorldState>;

    /// List all stored world IDs.
    async fn list(&self) -> Result<Vec<WorldId>>;

    /// Delete a world from the store.
    async fn delete(&self, id: &WorldId) -> Result<()>;
}

/// Shared pointer to a dynamically selected state store implementation.
pub type DynStateStore = Arc<dyn StateStore>;

/// Serialization format for file-backed world-state persistence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateFileFormat {
    /// Human-readable JSON files.
    Json,
    /// Compact MessagePack files.
    MessagePack,
}

impl StateFileFormat {
    /// Return the canonical user-facing identifier for this format.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::MessagePack => "msgpack",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::MessagePack => "msgpack",
        }
    }

    fn alternate(self) -> Self {
        match self {
            Self::Json => Self::MessagePack,
            Self::MessagePack => Self::Json,
        }
    }
}

/// Serialize a world state using the requested snapshot format.
pub fn serialize_world_state(format: StateFileFormat, state: &WorldState) -> Result<Vec<u8>> {
    match format {
        StateFileFormat::Json => serde_json::to_vec_pretty(state)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string())),
        StateFileFormat::MessagePack => rmp_serde::to_vec_named(state)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string())),
    }
}

/// Deserialize a world state using the requested snapshot format.
pub fn deserialize_world_state(format: StateFileFormat, data: &[u8]) -> Result<WorldState> {
    match format {
        StateFileFormat::Json => serde_json::from_slice(data)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string())),
        StateFileFormat::MessagePack => rmp_serde::from_slice(data)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string())),
    }
}

/// Infer the snapshot format from a file path extension.
///
/// Recognized extensions are `json`, `msgpack`, and `messagepack`.
pub fn infer_state_file_format(path: impl AsRef<Path>) -> Result<StateFileFormat> {
    let path = path.as_ref();
    let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
        return Err(WorldForgeError::InvalidState(format!(
            "snapshot path '{}' is missing a file extension",
            path.display()
        )));
    };

    extension.parse::<StateFileFormat>().map_err(|_| {
        WorldForgeError::InvalidState(format!(
            "snapshot path '{}' uses an unknown extension '{extension}'",
            path.display()
        ))
    })
}

impl std::str::FromStr for StateFileFormat {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "msgpack" | "messagepack" => Ok(Self::MessagePack),
            other => Err(format!(
                "unknown state file format: {other}. Available formats: json, msgpack"
            )),
        }
    }
}

/// Concrete state-store implementation to open at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateStoreKind {
    /// Persist each world state as a JSON file in the given directory.
    File(PathBuf),
    /// Persist each world state as a file in the given directory using an explicit format.
    FileWithFormat {
        /// Directory for persisted state files.
        path: PathBuf,
        /// Serialization format for the files in this store.
        format: StateFileFormat,
    },
    /// Persist all world states in a SQLite database file.
    #[cfg(feature = "sqlite")]
    Sqlite(PathBuf),
}

impl StateStoreKind {
    /// Open the configured state store implementation.
    pub async fn open(&self) -> Result<DynStateStore> {
        match self {
            Self::File(path) => Ok(Arc::new(FileStateStore::new(path.clone()))),
            Self::FileWithFormat { path, format } => Ok(Arc::new(FileStateStore::new_with_format(
                path.clone(),
                *format,
            ))),
            #[cfg(feature = "sqlite")]
            Self::Sqlite(path) => Ok(Arc::new(SqliteStateStore::from_path(path).await?)),
        }
    }
}

/// File-based state store using JSON serialization.
#[derive(Debug, Clone)]
pub struct FileStateStore {
    /// Directory for state files.
    pub path: PathBuf,
    /// Serialization format used when writing state files.
    pub format: StateFileFormat,
}

impl FileStateStore {
    /// Create a new file-based state store at the given directory.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::new_with_format(path, StateFileFormat::Json)
    }

    /// Create a new file-based state store with an explicit on-disk format.
    pub fn new_with_format(path: impl Into<PathBuf>, format: StateFileFormat) -> Self {
        Self {
            path: path.into(),
            format,
        }
    }

    fn state_path_for_format(&self, id: &WorldId, format: StateFileFormat) -> PathBuf {
        self.path.join(format!("{}.{}", id, format.extension()))
    }

    fn candidate_formats(&self) -> [StateFileFormat; 2] {
        [self.format, self.format.alternate()]
    }
}

#[async_trait::async_trait]
impl StateStore for FileStateStore {
    async fn save(&self, state: &WorldState) -> Result<()> {
        let mut normalized = state.clone();
        let provider = normalized.current_state_provider();
        normalized.ensure_history_initialized(provider)?;
        tokio::fs::create_dir_all(&self.path)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("failed to create dir: {e}")))?;
        let payload = serialize_world_state(self.format, &normalized)?;
        tokio::fs::write(
            self.state_path_for_format(&normalized.id, self.format),
            payload,
        )
        .await
        .map_err(|e| WorldForgeError::InternalError(format!("failed to write state: {e}")))?;
        Ok(())
    }

    async fn load(&self, id: &WorldId) -> Result<WorldState> {
        for format in self.candidate_formats() {
            let path = self.state_path_for_format(id, format);
            match tokio::fs::read(&path).await {
                Ok(data) => return deserialize_world_state(format, &data),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(WorldForgeError::InternalError(format!(
                        "failed to read state: {error}"
                    )))
                }
            }
        }

        Err(WorldForgeError::WorldNotFound(*id))
    }

    async fn list(&self) -> Result<Vec<WorldId>> {
        if !tokio::fs::try_exists(&self.path)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("failed to inspect dir: {e}")))?
        {
            return Ok(Vec::new());
        }

        let mut ids = HashSet::new();
        let mut entries = tokio::fs::read_dir(&self.path)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("failed to read dir: {e}")))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| WorldForgeError::InternalError(e.to_string()))?
        {
            if let Some(name) = entry.file_name().to_str() {
                if infer_state_file_format(name).is_ok() {
                    if let Some(id_str) = name
                        .strip_suffix(".json")
                        .or_else(|| name.strip_suffix(".msgpack"))
                        .or_else(|| name.strip_suffix(".messagepack"))
                    {
                        if let Ok(id) = id_str.parse::<WorldId>() {
                            ids.insert(id);
                        }
                    }
                }
            }
        }

        let mut ids = ids.into_iter().collect::<Vec<_>>();
        ids.sort_unstable_by_key(|id| id.as_u128());
        Ok(ids)
    }

    async fn delete(&self, id: &WorldId) -> Result<()> {
        let mut deleted_any = false;

        for format in self.candidate_formats() {
            let path = self.state_path_for_format(id, format);
            match tokio::fs::remove_file(&path).await {
                Ok(()) => deleted_any = true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(WorldForgeError::InternalError(format!(
                        "failed to delete state: {error}"
                    )))
                }
            }
        }

        if deleted_any {
            Ok(())
        } else {
            Err(WorldForgeError::WorldNotFound(*id))
        }
    }
}

// ---------------------------------------------------------------------------
// SQLite-based state store
// ---------------------------------------------------------------------------

/// SQLite-backed state store using sqlx.
///
/// Stores world states in a single `world_states` table with the world ID as
/// primary key and the JSON-serialized state as a TEXT column.
#[cfg(feature = "sqlite")]
#[derive(Debug, Clone)]
pub struct SqliteStateStore {
    pool: sqlx::SqlitePool,
}

#[cfg(feature = "sqlite")]
impl SqliteStateStore {
    /// Create a new SQLite state store and initialize the schema.
    ///
    /// The `url` should be a valid SQLite connection string, e.g.
    /// `"sqlite:worldforge.db"` or `"sqlite::memory:"`.
    pub async fn new(url: &str) -> Result<Self> {
        let pool = sqlx::SqlitePool::connect(url)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("SQLite connect failed: {e}")))?;

        Self::initialize_schema(&pool).await?;

        Ok(Self { pool })
    }

    /// Create a SQLite state store from a filesystem path, creating parent
    /// directories and the database file as needed.
    pub async fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                WorldForgeError::InternalError(format!(
                    "failed to create SQLite parent directory: {e}"
                ))
            })?;
        }

        let options = sqlx::sqlite::SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = sqlx::SqlitePool::connect_with(options)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("SQLite connect failed: {e}")))?;

        Self::initialize_schema(&pool).await?;

        Ok(Self { pool })
    }

    async fn initialize_schema(pool: &sqlx::SqlitePool) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS world_states (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL
            )",
        )
        .execute(pool)
        .await
        .map_err(|e| WorldForgeError::InternalError(format!("schema creation failed: {e}")))?;

        Ok(())
    }
}

#[cfg(feature = "sqlite")]
#[async_trait::async_trait]
impl StateStore for SqliteStateStore {
    async fn save(&self, state: &WorldState) -> Result<()> {
        let mut normalized = state.clone();
        let provider = normalized.current_state_provider();
        normalized.ensure_history_initialized(provider)?;
        let id = normalized.id.to_string();
        let json = serde_json::to_string(&normalized)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;

        sqlx::query("INSERT OR REPLACE INTO world_states (id, state) VALUES (?, ?)")
            .bind(&id)
            .bind(&json)
            .execute(&self.pool)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("SQLite save failed: {e}")))?;

        Ok(())
    }

    async fn load(&self, id: &WorldId) -> Result<WorldState> {
        let id_str = id.to_string();
        let row: Option<(String,)> = sqlx::query_as("SELECT state FROM world_states WHERE id = ?")
            .bind(&id_str)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("SQLite load failed: {e}")))?;

        match row {
            Some((json,)) => serde_json::from_str(&json)
                .map_err(|e| WorldForgeError::SerializationError(e.to_string())),
            None => Err(WorldForgeError::WorldNotFound(*id)),
        }
    }

    async fn list(&self) -> Result<Vec<WorldId>> {
        let rows: Vec<(String,)> = sqlx::query_as("SELECT id FROM world_states")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("SQLite list failed: {e}")))?;

        let mut ids = Vec::new();
        for (id_str,) in rows {
            if let Ok(id) = id_str.parse::<WorldId>() {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    async fn delete(&self, id: &WorldId) -> Result<()> {
        let id_str = id.to_string();
        let result = sqlx::query("DELETE FROM world_states WHERE id = ?")
            .bind(&id_str)
            .execute(&self.pool)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("SQLite delete failed: {e}")))?;

        if result.rows_affected() == 0 {
            return Err(WorldForgeError::WorldNotFound(*id));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_state_new() {
        let ws = WorldState::new("test-world", "mock");
        assert_eq!(ws.metadata.name, "test-world");
        assert_eq!(ws.metadata.created_by, "mock");
        assert_eq!(ws.time.step, 0);
        assert!(ws.history.is_empty());
    }

    #[test]
    fn test_world_state_from_prompt_seeds_metadata_and_scene() {
        let state =
            WorldState::from_prompt("A kitchen with a mug", "mock", Some("seeded")).unwrap();

        assert_eq!(state.metadata.name, "seeded");
        assert_eq!(state.metadata.description, "A kitchen with a mug");
        assert_eq!(state.metadata.created_by, "mock");
        assert_eq!(state.history.len(), 1);
        assert!(state.scene.find_object_by_name("counter").is_some());
        assert!(state.scene.find_object_by_name("mug").is_some());
    }

    #[test]
    fn test_world_state_ensure_history_initialized_records_initial_snapshot() {
        let mut state = WorldState::new("test-world", "mock");

        let recorded = state.ensure_history_initialized("mock").unwrap();

        assert!(recorded);
        assert_eq!(state.history.len(), 1);
        let latest = state.history.latest().unwrap();
        assert_eq!(latest.provider, "mock");
        assert_eq!(latest.time, state.time);
    }

    #[test]
    fn test_world_state_ensure_current_state_recorded_repairs_stale_latest_hash() {
        let mut state = WorldState::new("legacy", "mock");
        state.history.push(HistoryEntry {
            time: state.time,
            state_hash: [7; 32],
            action: None,
            prediction: None,
            provider: "mock".to_string(),
        });

        let repaired = state.ensure_current_state_recorded("mock").unwrap();

        assert!(repaired);
        assert_eq!(state.history.len(), 2);
        assert_ne!(state.history.latest().unwrap().state_hash, [7; 32]);
    }

    #[test]
    fn test_state_history_push_and_eviction() {
        let mut history = StateHistory {
            states: VecDeque::new(),
            max_entries: 3,
            compression: Compression::None,
        };
        for i in 0..5 {
            history.push(HistoryEntry {
                time: SimTime {
                    step: i,
                    seconds: i as f64,
                    dt: 1.0,
                },
                state_hash: [0u8; 32],
                action: None,
                prediction: None,
                provider: "mock".to_string(),
            });
        }
        assert_eq!(history.len(), 3);
        assert_eq!(history.latest().unwrap().time.step, 4);
    }

    #[test]
    fn test_world_state_serialization() {
        let ws = WorldState::new("test", "mock");
        let json = serde_json::to_string(&ws).unwrap();
        let ws2: WorldState = serde_json::from_str(&json).unwrap();
        assert_eq!(ws.id, ws2.id);
        assert_eq!(ws.metadata.name, ws2.metadata.name);
    }

    #[test]
    fn test_state_file_format_parsing() {
        assert_eq!(
            "json".parse::<StateFileFormat>().unwrap(),
            StateFileFormat::Json
        );
        assert_eq!(
            "msgpack".parse::<StateFileFormat>().unwrap(),
            StateFileFormat::MessagePack
        );
        assert_eq!(
            "messagepack".parse::<StateFileFormat>().unwrap(),
            StateFileFormat::MessagePack
        );
        assert!("yaml".parse::<StateFileFormat>().is_err());
    }

    #[test]
    fn test_infer_state_file_format() {
        assert_eq!(
            infer_state_file_format(Path::new("snapshot.json")).unwrap(),
            StateFileFormat::Json
        );
        assert_eq!(
            infer_state_file_format(Path::new("snapshot.msgpack")).unwrap(),
            StateFileFormat::MessagePack
        );
        assert_eq!(
            infer_state_file_format(Path::new("snapshot.messagepack")).unwrap(),
            StateFileFormat::MessagePack
        );
        assert!(matches!(
            infer_state_file_format(Path::new("snapshot")),
            Err(WorldForgeError::InvalidState(message)) if message.contains("missing a file extension")
        ));
        assert!(matches!(
            infer_state_file_format(Path::new("snapshot.yaml")),
            Err(WorldForgeError::InvalidState(message)) if message.contains("unknown extension")
        ));
    }

    #[test]
    fn test_world_state_snapshot_codec_roundtrip_json() {
        let mut state = WorldState::new("codec-json", "mock");
        state.metadata.description = "json roundtrip".to_string();

        let bytes = serialize_world_state(StateFileFormat::Json, &state).unwrap();
        let restored = deserialize_world_state(StateFileFormat::Json, &bytes).unwrap();

        assert_eq!(restored.id, state.id);
        assert_eq!(restored.metadata.name, state.metadata.name);
        assert_eq!(restored.metadata.description, state.metadata.description);
    }

    #[test]
    fn test_world_state_snapshot_codec_roundtrip_msgpack() {
        let mut state = WorldState::new("codec-msgpack", "mock");
        state.metadata.description = "msgpack roundtrip".to_string();

        let bytes = serialize_world_state(StateFileFormat::MessagePack, &state).unwrap();
        let restored = deserialize_world_state(StateFileFormat::MessagePack, &bytes).unwrap();

        assert_eq!(restored.id, state.id);
        assert_eq!(restored.metadata.name, state.metadata.name);
        assert_eq!(restored.metadata.description, state.metadata.description);
    }

    #[test]
    fn test_world_state_snapshot_codec_rejects_invalid_bytes() {
        let err = deserialize_world_state(StateFileFormat::Json, b"not json").unwrap_err();
        assert!(matches!(err, WorldForgeError::SerializationError(_)));
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn test_sqlite_state_store() {
        let store = SqliteStateStore::new("sqlite::memory:").await.unwrap();

        let state = WorldState::new("sqlite-test", "mock");
        let id = state.id;

        // Save
        store.save(&state).await.unwrap();

        // Load
        let loaded = store.load(&id).await.unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.metadata.name, "sqlite-test");

        // List
        let ids = store.list().await.unwrap();
        assert!(ids.contains(&id));

        // Overwrite (upsert)
        let mut updated = state.clone();
        updated.time.step = 42;
        store.save(&updated).await.unwrap();
        let reloaded = store.load(&id).await.unwrap();
        assert_eq!(reloaded.time.step, 42);

        // Delete
        store.delete(&id).await.unwrap();
        assert!(store.load(&id).await.is_err());

        // Delete nonexistent
        assert!(store.delete(&id).await.is_err());
    }

    #[tokio::test]
    async fn test_file_state_store() {
        let dir = std::env::temp_dir().join(format!("worldforge-test-{}", uuid::Uuid::new_v4()));
        let store = FileStateStore::new(&dir);

        let state = WorldState::new("test-world", "mock");
        let id = state.id;

        store.save(&state).await.unwrap();
        assert!(dir.join(format!("{id}.json")).exists());
        let loaded = store.load(&id).await.unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.history.len(), 1);

        let ids = store.list().await.unwrap();
        assert!(ids.contains(&id));

        store.delete(&id).await.unwrap();
        assert!(store.load(&id).await.is_err());

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn test_file_state_store_lists_empty_directory_when_missing() {
        let dir = std::env::temp_dir().join(format!("worldforge-missing-{}", uuid::Uuid::new_v4()));
        let store = FileStateStore::new(&dir);

        let ids = store.list().await.unwrap();
        assert!(ids.is_empty());
    }

    #[tokio::test]
    async fn test_file_state_store_msgpack_roundtrip() {
        let dir = std::env::temp_dir().join(format!("worldforge-msgpack-{}", uuid::Uuid::new_v4()));
        let store = FileStateStore::new_with_format(&dir, StateFileFormat::MessagePack);

        let state = WorldState::new("msgpack-world", "mock");
        let id = state.id;

        store.save(&state).await.unwrap();
        assert!(dir.join(format!("{id}.msgpack")).exists());

        let loaded = store.load(&id).await.unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.metadata.name, "msgpack-world");

        let ids = store.list().await.unwrap();
        assert_eq!(ids, vec![id]);

        store.delete(&id).await.unwrap();
        assert!(store.load(&id).await.is_err());

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn test_file_state_store_loads_alternate_format() {
        let dir =
            std::env::temp_dir().join(format!("worldforge-alt-format-{}", uuid::Uuid::new_v4()));
        let json_store = FileStateStore::new(&dir);
        let msgpack_store = FileStateStore::new_with_format(&dir, StateFileFormat::MessagePack);

        let state = WorldState::new("alternate-format", "mock");
        let id = state.id;

        json_store.save(&state).await.unwrap();
        let loaded_from_msgpack_store = msgpack_store.load(&id).await.unwrap();
        assert_eq!(loaded_from_msgpack_store.id, id);

        msgpack_store.delete(&id).await.unwrap();
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn test_file_state_store_list_deduplicates_multiple_formats() {
        let dir = std::env::temp_dir().join(format!("worldforge-dedup-{}", uuid::Uuid::new_v4()));
        let json_store = FileStateStore::new(&dir);
        let msgpack_store = FileStateStore::new_with_format(&dir, StateFileFormat::MessagePack);

        let state = WorldState::new("dedup-world", "mock");
        let id = state.id;

        json_store.save(&state).await.unwrap();
        msgpack_store.save(&state).await.unwrap();

        let ids = json_store.list().await.unwrap();
        assert_eq!(ids, vec![id]);

        json_store.delete(&id).await.unwrap();
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn test_state_store_kind_opens_file_store() {
        let dir = std::env::temp_dir().join(format!("worldforge-kind-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.clone()).open().await.unwrap();
        let state = WorldState::new("kind-test", "mock");

        store.save(&state).await.unwrap();
        let loaded = store.load(&state.id).await.unwrap();
        assert_eq!(loaded.id, state.id);
        assert_eq!(loaded.history.len(), 1);

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn test_state_store_kind_opens_file_store_with_explicit_format() {
        let dir =
            std::env::temp_dir().join(format!("worldforge-kind-msgpack-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::FileWithFormat {
            path: dir.clone(),
            format: StateFileFormat::MessagePack,
        }
        .open()
        .await
        .unwrap();
        let state = WorldState::new("kind-msgpack", "mock");

        store.save(&state).await.unwrap();
        let loaded = store.load(&state.id).await.unwrap();
        assert_eq!(loaded.metadata.name, "kind-msgpack");
        assert!(dir.join(format!("{}.msgpack", state.id)).exists());

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn test_sqlite_state_store_from_path() {
        let db_path = std::env::temp_dir()
            .join(format!("worldforge-sqlite-{}", uuid::Uuid::new_v4()))
            .join("nested")
            .join("worldforge.db");
        let store = SqliteStateStore::from_path(&db_path).await.unwrap();
        let state = WorldState::new("sqlite-path-test", "mock");

        store.save(&state).await.unwrap();
        let ids = store.list().await.unwrap();
        assert!(ids.contains(&state.id));
        assert!(db_path.exists());

        let _ = tokio::fs::remove_file(&db_path).await;
        if let Some(parent) = db_path.parent().and_then(Path::parent) {
            let _ = tokio::fs::remove_dir_all(parent).await;
        }
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn test_state_store_kind_opens_sqlite_store() {
        let db_path = std::env::temp_dir()
            .join(format!("worldforge-kind-sqlite-{}", uuid::Uuid::new_v4()))
            .join("state.db");
        let store = StateStoreKind::Sqlite(db_path.clone())
            .open()
            .await
            .unwrap();
        let state = WorldState::new("sqlite-kind", "mock");

        store.save(&state).await.unwrap();
        let loaded = store.load(&state.id).await.unwrap();
        assert_eq!(loaded.metadata.name, "sqlite-kind");

        let _ = tokio::fs::remove_file(&db_path).await;
        if let Some(parent) = db_path.parent() {
            let _ = tokio::fs::remove_dir_all(parent).await;
        }
    }
}
