//! State persistence for WorldForge worlds.
//!
//! Provides the `StateStore` trait and built-in file/SQLite
//! implementations for saving and loading world state.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use reqwest::{Method, StatusCode, Url};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use crate::action::Action;
use crate::bootstrap::seed_world_state_from_prompt;
use crate::error::{Result, WorldForgeError};
use crate::prediction::{PredictionProvenance, StoredPlanRecord};
use crate::scene::SceneGraph;
use crate::types::{PlanId, SimTime, WorldId};

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
    /// Persisted plan artifacts associated with this world.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stored_plans: BTreeMap<PlanId, StoredPlanRecord>,
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
    /// Recoverable snapshot of the world at this checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<HistorySnapshot>,
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
    /// Provider model identifier, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Execution provenance for the underlying prediction, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<PredictionProvenance>,
}

/// Recoverable world checkpoint stored alongside a history entry.
///
/// The snapshot captures the materialized world fields for a checkpoint
/// without recursively embedding the history log itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistorySnapshot {
    /// Simulation time captured by the checkpoint.
    pub time: SimTime,
    /// Scene graph materialized at this checkpoint.
    pub scene: SceneGraph,
    /// World metadata active at this checkpoint.
    pub metadata: WorldMetadata,
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
            stored_plans: BTreeMap::new(),
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

    /// Fork this world state into a new world with a fresh identity.
    ///
    /// The fork preserves the materialized world contents and creates a new,
    /// self-consistent history rooted at the fork point.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::SerializationError` if the forked state
    /// cannot be hashed for history tracking.
    pub fn fork(&self, name_override: Option<&str>) -> Result<Self> {
        self.fork_with_snapshot(self.snapshot(), name_override)
    }

    /// Fork a historical checkpoint into a new world with a fresh identity.
    ///
    /// The fork preserves the materialized checkpoint state and creates a new,
    /// self-consistent history rooted at the checkpoint.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if the checkpoint cannot be
    /// reconstructed or `WorldForgeError::SerializationError` if the forked
    /// state cannot be hashed for history tracking.
    pub fn fork_from_history(&self, index: usize, name_override: Option<&str>) -> Result<Self> {
        let checkpoint = self.history_state(index)?;
        checkpoint.fork(name_override)
    }

    /// Return the provider most likely responsible for the current state snapshot.
    pub fn current_state_provider(&self) -> String {
        self.history
            .latest()
            .map(|entry| entry.provider.clone())
            .filter(|provider| !provider.is_empty())
            .unwrap_or_else(|| self.metadata.created_by.clone())
    }

    /// Capture the current world fields as a recoverable history snapshot.
    pub fn snapshot(&self) -> HistorySnapshot {
        HistorySnapshot {
            time: self.time,
            scene: self.scene.clone(),
            metadata: self.metadata.clone(),
        }
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
            snapshot: Some(self.snapshot()),
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

        if self
            .history
            .latest()
            .map(|entry| entry.time == self.time)
            .unwrap_or(false)
        {
            let repaired_hash = canonical_hash_without_latest_history(self)?;
            let snapshot = self.snapshot();
            if let Some(entry) = self.history.states.back_mut() {
                entry.state_hash = repaired_hash;
                entry.snapshot = Some(snapshot);
                return Ok(true);
            }
        }

        self.record_current_state(None, None, provider)?;
        Ok(true)
    }

    /// Ensure the latest history entry has a recoverable snapshot payload.
    ///
    /// This upgrades legacy states written before checkpoint payloads were added.
    pub fn ensure_latest_history_snapshot(&mut self) -> Result<bool> {
        if self.history.is_empty() {
            return Ok(false);
        }

        let needs_snapshot = self
            .history
            .latest()
            .map(|entry| entry.snapshot.is_none())
            .unwrap_or(false);
        if !needs_snapshot {
            return Ok(false);
        }

        if !current_state_matches_latest_history(self)? {
            return Ok(false);
        }

        let snapshot = self.snapshot();
        if let Some(entry) = self.history.states.back_mut() {
            entry.snapshot = Some(snapshot);
            return Ok(true);
        }

        Ok(false)
    }

    /// Reconstruct the world state captured by a specific history entry.
    ///
    /// The returned state preserves the world ID and truncates history after the
    /// requested checkpoint so future operations continue from that point.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if the index is out of bounds or
    /// the requested checkpoint predates recoverable snapshots.
    pub fn history_state(&self, index: usize) -> Result<Self> {
        let entry_count = self.history.len();
        let Some(entry) = self.history.states.get(index) else {
            return Err(WorldForgeError::InvalidState(format!(
                "history index {index} out of range for {entry_count} entries"
            )));
        };

        let snapshot = match &entry.snapshot {
            Some(snapshot) => snapshot.clone(),
            None if index + 1 == entry_count && current_state_matches_latest_history(self)? => {
                self.snapshot()
            }
            None => {
                return Err(WorldForgeError::InvalidState(format!(
                    "history checkpoint {index} does not include a recoverable snapshot"
                )))
            }
        };

        let mut history = self.history.clone();
        history.states.truncate(index + 1);

        Ok(Self {
            id: self.id,
            time: snapshot.time,
            scene: snapshot.scene,
            history,
            metadata: snapshot.metadata,
            stored_plans: self.stored_plans.clone(),
        })
    }

    /// Restore this world state in place to a specific history checkpoint.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if the checkpoint cannot be
    /// reconstructed.
    pub fn restore_history(&mut self, index: usize) -> Result<()> {
        *self = self.history_state(index)?;
        Ok(())
    }

    fn fork_with_snapshot(
        &self,
        snapshot: HistorySnapshot,
        name_override: Option<&str>,
    ) -> Result<Self> {
        let provider = self.current_state_provider();
        let mut forked = Self {
            id: uuid::Uuid::new_v4(),
            time: snapshot.time,
            scene: snapshot.scene,
            history: StateHistory {
                states: VecDeque::new(),
                max_entries: self.history.max_entries,
                compression: self.history.compression,
            },
            metadata: snapshot.metadata,
            stored_plans: self.stored_plans.clone(),
        };
        forked.metadata.name = derive_fork_name(&forked.metadata.name, name_override);
        forked.metadata.created_by = provider.clone();
        forked.metadata.created_at = chrono::Utc::now();
        forked.record_current_state(None, None, provider)?;
        Ok(forked)
    }

    /// Persist a plan artifact alongside this world state.
    pub fn store_plan_record(&mut self, record: StoredPlanRecord) -> PlanId {
        let id = record.id;
        self.stored_plans.insert(id, record);
        id
    }

    /// List all persisted plan artifacts associated with this world.
    pub fn list_stored_plans(&self) -> Vec<StoredPlanRecord> {
        self.stored_plans.values().cloned().collect()
    }

    /// Return a persisted plan artifact by ID.
    pub fn get_stored_plan(&self, id: &PlanId) -> Option<&StoredPlanRecord> {
        self.stored_plans.get(id)
    }

    /// Return a persisted plan artifact by ID.
    pub fn stored_plan(&self, id: &PlanId) -> Option<&StoredPlanRecord> {
        self.get_stored_plan(id)
    }

    /// Remove a persisted plan artifact by ID and return it if present.
    pub fn remove_stored_plan(&mut self, id: &PlanId) -> Option<StoredPlanRecord> {
        self.stored_plans.remove(id)
    }
}

fn derive_fork_name(source_name: &str, name_override: Option<&str>) -> String {
    if let Some(name) = name_override.map(str::trim).filter(|name| !name.is_empty()) {
        return name.to_string();
    }

    let source_name = source_name.trim();
    if source_name.is_empty() {
        "Forked World".to_string()
    } else {
        format!("{source_name} Fork")
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
    let mut snapshot = state.clone();
    snapshot.stored_plans.clear();
    let bytes = serde_json::to_vec(&snapshot)
        .map_err(|error| WorldForgeError::SerializationError(error.to_string()))?;
    Ok(sha256_hash(&bytes))
}

fn current_state_matches_latest_history(state: &WorldState) -> Result<bool> {
    let Some(latest) = state.history.latest() else {
        return Ok(false);
    };

    let snapshot = current_state_without_latest_history(state)?;
    Ok(snapshot.time == latest.time && canonical_state_hash(&snapshot)? == latest.state_hash)
}

fn current_state_without_latest_history(state: &WorldState) -> Result<WorldState> {
    let mut snapshot = state.clone();
    if snapshot.history.states.pop_back().is_none() {
        return Err(WorldForgeError::InvalidState(
            "cannot compare current state without history entries".to_string(),
        ));
    }
    Ok(snapshot)
}

fn canonical_hash_without_latest_history(state: &WorldState) -> Result<[u8; 32]> {
    let snapshot = current_state_without_latest_history(state)?;
    canonical_state_hash(&snapshot)
}

fn normalize_world_state(mut state: WorldState) -> Result<WorldState> {
    let provider = state.current_state_provider();
    state.ensure_history_initialized(provider.as_str())?;
    state.ensure_current_state_recorded(provider.as_str())?;
    state.ensure_latest_history_snapshot()?;
    Ok(state)
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

/// Schema version written into persisted world-state snapshots.
pub const WORLD_STATE_SNAPSHOT_SCHEMA_VERSION: u32 = 1;

/// Metadata discovered while inspecting a serialized world-state snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSnapshotMetadata {
    /// Effective schema version associated with the snapshot payload.
    ///
    /// Legacy raw `WorldState` payloads report the current schema version and
    /// set `legacy_payload=true`.
    pub schema_version: u32,
    /// Whether the payload used the legacy raw-`WorldState` shape.
    pub legacy_payload: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorldStateSnapshotEnvelope {
    schema_version: u32,
    state: WorldState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum WorldStateSnapshotDocument {
    Envelope(WorldStateSnapshotEnvelope),
    Legacy(WorldState),
}

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
    let document = WorldStateSnapshotEnvelope {
        schema_version: WORLD_STATE_SNAPSHOT_SCHEMA_VERSION,
        state: state.clone(),
    };
    match format {
        StateFileFormat::Json => serde_json::to_vec_pretty(&document)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string())),
        StateFileFormat::MessagePack => rmp_serde::to_vec_named(&document)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string())),
    }
}

/// Inspect a serialized world-state snapshot without materializing the state.
pub fn inspect_world_state_snapshot(
    format: StateFileFormat,
    data: &[u8],
) -> Result<StateSnapshotMetadata> {
    let (_, metadata) = decode_world_state_snapshot(format, data)?;
    Ok(metadata)
}

/// Deserialize a world state using the requested snapshot format.
///
/// The returned state is normalized so legacy snapshots regain a recoverable
/// latest history entry and a materialized checkpoint snapshot when needed.
pub fn deserialize_world_state(format: StateFileFormat, data: &[u8]) -> Result<WorldState> {
    let (state, _) = decode_world_state_snapshot(format, data)?;
    normalize_world_state(state)
}

fn decode_world_state_snapshot(
    format: StateFileFormat,
    data: &[u8],
) -> Result<(WorldState, StateSnapshotMetadata)> {
    let document = match format {
        StateFileFormat::Json => serde_json::from_slice(data)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?,
        StateFileFormat::MessagePack => rmp_serde::from_slice(data)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?,
    };

    match document {
        WorldStateSnapshotDocument::Envelope(envelope) => {
            validate_snapshot_schema_version(envelope.schema_version)?;
            Ok((
                envelope.state,
                StateSnapshotMetadata {
                    schema_version: envelope.schema_version,
                    legacy_payload: false,
                },
            ))
        }
        WorldStateSnapshotDocument::Legacy(state) => Ok((
            state,
            StateSnapshotMetadata {
                schema_version: WORLD_STATE_SNAPSHOT_SCHEMA_VERSION,
                legacy_payload: true,
            },
        )),
    }
}

fn validate_snapshot_schema_version(schema_version: u32) -> Result<()> {
    if schema_version == 0 || schema_version > WORLD_STATE_SNAPSHOT_SCHEMA_VERSION {
        return Err(WorldForgeError::InvalidState(format!(
            "unsupported world-state snapshot schema version: {schema_version} (supported: 1..={})",
            WORLD_STATE_SNAPSHOT_SCHEMA_VERSION
        )));
    }

    Ok(())
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
    /// Persist all world states in a Redis database.
    Redis(String),
    /// Persist world states as objects in an S3-compatible bucket.
    S3 {
        /// S3 connection and authentication settings.
        config: S3Config,
        /// Serialization format used for object payloads.
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
            Self::Redis(url) => Ok(Arc::new(RedisStateStore::new(url.clone()).await?)),
            Self::S3 { config, format } => {
                Ok(Arc::new(S3StateStore::new(config.clone(), *format)?))
            }
            #[cfg(feature = "sqlite")]
            Self::Sqlite(path) => Ok(Arc::new(SqliteStateStore::from_path(path).await?)),
        }
    }
}

const S3_SIGNING_ALGORITHM: &str = "AWS4-HMAC-SHA256";
const S3_SERVICE: &str = "s3";
const HMAC_SHA256_BLOCK_SIZE: usize = 64;
const REDIS_STATE_KEY_PREFIX: &str = "worldforge:state:";
const REDIS_STATE_INDEX_KEY: &str = "worldforge:state:index";

/// Configuration for an S3-compatible state store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct S3Config {
    /// Bucket that stores world snapshots.
    pub bucket: String,
    /// AWS region or S3-compatible region label used for request signing.
    pub region: String,
    /// Optional custom endpoint for S3-compatible services such as MinIO.
    pub endpoint: Option<String>,
    /// Access key used for SigV4 request signing.
    pub access_key_id: String,
    /// Secret key used for SigV4 request signing.
    pub secret_access_key: String,
    /// Optional session token for temporary credentials.
    pub session_token: Option<String>,
    /// Object-key prefix under which world snapshots are stored.
    pub prefix: String,
}

impl S3Config {
    fn validate(&self) -> Result<()> {
        if self.bucket.trim().is_empty() {
            return Err(WorldForgeError::InvalidState(
                "s3 bucket cannot be empty".to_string(),
            ));
        }
        if self.region.trim().is_empty() {
            return Err(WorldForgeError::InvalidState(
                "s3 region cannot be empty".to_string(),
            ));
        }
        if self.access_key_id.trim().is_empty() {
            return Err(WorldForgeError::InvalidState(
                "s3 access key id cannot be empty".to_string(),
            ));
        }
        if self.secret_access_key.trim().is_empty() {
            return Err(WorldForgeError::InvalidState(
                "s3 secret access key cannot be empty".to_string(),
            ));
        }

        self.endpoint_url()?;
        Ok(())
    }

    fn normalized_prefix(&self) -> String {
        let trimmed = self.prefix.trim().trim_matches('/');
        if trimmed.is_empty() {
            String::new()
        } else {
            format!("{trimmed}/")
        }
    }

    fn endpoint_url(&self) -> Result<Url> {
        let raw = self
            .endpoint
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("https://s3.{}.amazonaws.com", self.region.trim()));
        Url::parse(&raw).map_err(|error| {
            WorldForgeError::InvalidState(format!("invalid s3 endpoint '{raw}': {error}"))
        })
    }

    fn object_key(&self, id: &WorldId, format: StateFileFormat) -> String {
        format!("{}{}.{}", self.normalized_prefix(), id, format.extension())
    }
}

/// S3-compatible state store using SigV4-signed HTTP requests.
#[derive(Debug, Clone)]
pub struct S3StateStore {
    /// S3 connection and authentication settings.
    pub config: S3Config,
    /// Serialization format used for persisted object payloads.
    pub format: StateFileFormat,
    client: reqwest::Client,
}

impl S3StateStore {
    /// Create a new S3-backed state store.
    pub fn new(config: S3Config, format: StateFileFormat) -> Result<Self> {
        config.validate()?;
        let client = reqwest::Client::builder().build().map_err(|error| {
            WorldForgeError::NetworkError(format!("failed to build s3 client: {error}"))
        })?;
        Ok(Self {
            config,
            format,
            client,
        })
    }

    fn candidate_formats(&self) -> [StateFileFormat; 2] {
        [self.format, self.format.alternate()]
    }

    fn bucket_path(&self) -> Result<String> {
        let base = self.config.endpoint_url()?;
        Ok(s3_canonical_path(base.path(), &self.config.bucket, None))
    }

    fn object_path(&self, key: &str) -> Result<String> {
        let base = self.config.endpoint_url()?;
        Ok(s3_canonical_path(
            base.path(),
            &self.config.bucket,
            Some(key),
        ))
    }

    fn request_url(&self, canonical_path: &str, canonical_query: &str) -> Result<Url> {
        let mut url = self.config.endpoint_url()?;
        url.set_path(canonical_path);
        if canonical_query.is_empty() {
            url.set_query(None);
        } else {
            url.set_query(Some(canonical_query));
        }
        Ok(url)
    }

    async fn send_request(
        &self,
        method: Method,
        canonical_path: &str,
        query_pairs: &[(String, String)],
        body: Vec<u8>,
        content_type: Option<&str>,
    ) -> Result<reqwest::Response> {
        let canonical_query = canonical_query_string(query_pairs);
        let url = self.request_url(canonical_path, &canonical_query)?;
        let host = url_authority(&url)?;
        let amz_date = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = &amz_date[..8];
        let payload_hash = hex_sha256(&body);

        let mut headers = vec![
            ("host".to_string(), host.clone()),
            ("x-amz-content-sha256".to_string(), payload_hash.clone()),
            ("x-amz-date".to_string(), amz_date.clone()),
        ];
        if let Some(token) = self
            .config
            .session_token
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            headers.push(("x-amz-security-token".to_string(), token.to_string()));
        }
        headers.sort_by(|left, right| left.0.cmp(&right.0));

        let canonical_headers = headers
            .iter()
            .map(|(name, value)| format!("{name}:{value}\n"))
            .collect::<String>();
        let signed_headers = headers
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(";");

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method.as_str(),
            canonical_path,
            canonical_query,
            canonical_headers,
            signed_headers,
            payload_hash
        );
        let credential_scope = format!(
            "{date_stamp}/{}/{S3_SERVICE}/aws4_request",
            self.config.region.trim()
        );
        let string_to_sign = format!(
            "{S3_SIGNING_ALGORITHM}\n{amz_date}\n{credential_scope}\n{}",
            hex_sha256(canonical_request.as_bytes())
        );
        let signing_key = derive_s3_signing_key(
            self.config.secret_access_key.as_bytes(),
            date_stamp.as_bytes(),
            self.config.region.trim().as_bytes(),
        );
        let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));
        let authorization = format!(
            "{S3_SIGNING_ALGORITHM} Credential={}/{credential_scope}, SignedHeaders={}, Signature={signature}",
            self.config.access_key_id.trim(),
            signed_headers
        );

        let mut request = self
            .client
            .request(method, url)
            .header("host", host)
            .header("x-amz-content-sha256", payload_hash)
            .header("x-amz-date", amz_date)
            .header("authorization", authorization);
        if let Some(token) = self
            .config
            .session_token
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            request = request.header("x-amz-security-token", token);
        }
        if let Some(content_type) = content_type {
            request = request.header("content-type", content_type);
        }
        if !body.is_empty() {
            request = request.body(body);
        }

        request
            .send()
            .await
            .map_err(|error| WorldForgeError::NetworkError(format!("s3 request failed: {error}")))
    }

    async fn put_object(&self, key: &str, body: Vec<u8>) -> Result<()> {
        let path = self.object_path(key)?;
        let response = self
            .send_request(
                Method::PUT,
                &path,
                &[],
                body,
                Some(match self.format {
                    StateFileFormat::Json => "application/json",
                    StateFileFormat::MessagePack => "application/msgpack",
                }),
            )
            .await?;
        s3_expect_success(&self.config, response, "PUT object").await
    }

    async fn get_object(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let path = self.object_path(key)?;
        let response = self
            .send_request(Method::GET, &path, &[], Vec::new(), None)
            .await?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let response = s3_expect_success_response(&self.config, response, "GET object").await?;
        let bytes = response.bytes().await.map_err(|error| {
            WorldForgeError::NetworkError(format!("failed to read s3 response body: {error}"))
        })?;
        Ok(Some(bytes.to_vec()))
    }

    async fn object_exists(&self, key: &str) -> Result<bool> {
        let path = self.object_path(key)?;
        let response = self
            .send_request(Method::HEAD, &path, &[], Vec::new(), None)
            .await?;
        match response.status() {
            StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            _ => {
                s3_expect_success(&self.config, response, "HEAD object").await?;
                Ok(true)
            }
        }
    }

    async fn delete_object(&self, key: &str) -> Result<()> {
        let path = self.object_path(key)?;
        let response = self
            .send_request(Method::DELETE, &path, &[], Vec::new(), None)
            .await?;
        s3_expect_success(&self.config, response, "DELETE object").await
    }

    async fn list_keys(&self) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        let mut continuation_token = None;

        loop {
            let mut query_pairs = vec![
                ("list-type".to_string(), "2".to_string()),
                ("prefix".to_string(), self.config.normalized_prefix()),
            ];
            if let Some(token) = continuation_token.clone() {
                query_pairs.push(("continuation-token".to_string(), token));
            }

            let path = self.bucket_path()?;
            let response = self
                .send_request(Method::GET, &path, &query_pairs, Vec::new(), None)
                .await?;
            let response =
                s3_expect_success_response(&self.config, response, "ListObjectsV2").await?;
            let body = response.text().await.map_err(|error| {
                WorldForgeError::NetworkError(format!("failed to read s3 list response: {error}"))
            })?;
            keys.extend(xml_tag_values(&body, "Key"));

            if xml_tag_value(&body, "IsTruncated").as_deref() == Some("true") {
                continuation_token = xml_tag_value(&body, "NextContinuationToken");
                if continuation_token.is_none() {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(keys)
    }
}

#[async_trait::async_trait]
impl StateStore for S3StateStore {
    async fn save(&self, state: &WorldState) -> Result<()> {
        let normalized = normalize_world_state(state.clone())?;
        let payload = serialize_world_state(self.format, &normalized)?;
        self.put_object(
            &self.config.object_key(&normalized.id, self.format),
            payload,
        )
        .await
    }

    async fn load(&self, id: &WorldId) -> Result<WorldState> {
        for format in self.candidate_formats() {
            let key = self.config.object_key(id, format);
            if let Some(payload) = self.get_object(&key).await? {
                return deserialize_world_state(format, &payload);
            }
        }

        Err(WorldForgeError::WorldNotFound(*id))
    }

    async fn list(&self) -> Result<Vec<WorldId>> {
        let mut ids = HashSet::new();
        for key in self.list_keys().await? {
            let Some(name) = Path::new(&key).file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if infer_state_file_format(name).is_err() {
                continue;
            }
            let Some(id_str) = name
                .strip_suffix(".json")
                .or_else(|| name.strip_suffix(".msgpack"))
                .or_else(|| name.strip_suffix(".messagepack"))
            else {
                continue;
            };
            if let Ok(id) = id_str.parse::<WorldId>() {
                ids.insert(id);
            }
        }

        let mut ids = ids.into_iter().collect::<Vec<_>>();
        ids.sort_unstable_by_key(|id| id.as_u128());
        Ok(ids)
    }

    async fn delete(&self, id: &WorldId) -> Result<()> {
        let mut deleted_any = false;
        for format in self.candidate_formats() {
            let key = self.config.object_key(id, format);
            if self.object_exists(&key).await? {
                self.delete_object(&key).await?;
                deleted_any = true;
            }
        }

        if deleted_any {
            Ok(())
        } else {
            Err(WorldForgeError::WorldNotFound(*id))
        }
    }
}

fn s3_canonical_path(base_path: &str, bucket: &str, key: Option<&str>) -> String {
    let mut segments = base_path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .map(|segment| aws_uri_encode(segment, true))
        .collect::<Vec<_>>();
    segments.push(aws_uri_encode(bucket.trim(), true));
    if let Some(key) = key {
        let encoded = aws_uri_encode(key, false);
        if !encoded.is_empty() {
            segments.push(encoded);
        }
    }
    format!("/{}", segments.join("/"))
}

fn canonical_query_string(pairs: &[(String, String)]) -> String {
    let mut pairs = pairs
        .iter()
        .map(|(key, value)| (aws_uri_encode(key, true), aws_uri_encode(value, true)))
        .collect::<Vec<_>>();
    pairs.sort();
    pairs
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn url_authority(url: &Url) -> Result<String> {
    let host = url.host_str().ok_or_else(|| {
        WorldForgeError::InvalidState("s3 endpoint is missing a host".to_string())
    })?;
    Ok(match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_string(),
    })
}

fn aws_uri_encode(value: &str, encode_slash: bool) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        let is_unreserved = matches!(
            byte,
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~'
        );
        if is_unreserved || (!encode_slash && byte == b'/') {
            encoded.push(byte as char);
        } else {
            encoded.push('%');
            encoded.push(nibble_to_hex(byte >> 4));
            encoded.push(nibble_to_hex(byte & 0x0f));
        }
    }
    encoded
}

fn derive_s3_signing_key(secret: &[u8], date: &[u8], region: &[u8]) -> [u8; 32] {
    let mut prefixed = Vec::with_capacity(4 + secret.len());
    prefixed.extend_from_slice(b"AWS4");
    prefixed.extend_from_slice(secret);
    let date_key = hmac_sha256(&prefixed, date);
    let region_key = hmac_sha256(&date_key, region);
    let service_key = hmac_sha256(&region_key, S3_SERVICE.as_bytes());
    hmac_sha256(&service_key, b"aws4_request")
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut normalized_key = [0u8; HMAC_SHA256_BLOCK_SIZE];
    if key.len() > HMAC_SHA256_BLOCK_SIZE {
        normalized_key[..32].copy_from_slice(&sha256_hash(key));
    } else {
        normalized_key[..key.len()].copy_from_slice(key);
    }

    let mut inner = Vec::with_capacity(HMAC_SHA256_BLOCK_SIZE + data.len());
    let mut outer = Vec::with_capacity(HMAC_SHA256_BLOCK_SIZE + 32);
    for byte in normalized_key {
        inner.push(byte ^ 0x36);
        outer.push(byte ^ 0x5c);
    }
    inner.extend_from_slice(data);
    outer.extend_from_slice(&sha256_hash(&inner));
    sha256_hash(&outer)
}

fn hex_sha256(data: &[u8]) -> String {
    hex_encode(&sha256_hash(data))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(nibble_to_hex(byte >> 4));
        encoded.push(nibble_to_hex(byte & 0x0f));
    }
    encoded
}

fn nibble_to_hex(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'a' + (nibble - 10)) as char,
        _ => unreachable!("nibble must be less than 16"),
    }
}

async fn s3_expect_success(
    config: &S3Config,
    response: reqwest::Response,
    action: &str,
) -> Result<()> {
    s3_expect_success_response(config, response, action).await?;
    Ok(())
}

async fn s3_expect_success_response(
    config: &S3Config,
    response: reqwest::Response,
    action: &str,
) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = response
        .text()
        .await
        .unwrap_or_else(|_| "<unreadable body>".to_string());
    let detail = body.trim();
    let detail = if detail.is_empty() {
        "no response body"
    } else {
        detail
    };
    Err(WorldForgeError::NetworkError(format!(
        "s3 {action} failed for bucket '{}' with status {}: {detail}",
        config.bucket, status
    )))
}

fn xml_tag_value(xml: &str, tag: &str) -> Option<String> {
    xml_tag_values(xml, tag).into_iter().next()
}

fn xml_tag_values(xml: &str, tag: &str) -> Vec<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let mut cursor = 0usize;
    let mut values = Vec::new();

    while let Some(start) = xml[cursor..].find(&start_tag) {
        let value_start = cursor + start + start_tag.len();
        let Some(end) = xml[value_start..].find(&end_tag) else {
            break;
        };
        let value_end = value_start + end;
        values.push(xml_unescape(&xml[value_start..value_end]));
        cursor = value_end + end_tag.len();
    }

    values
}

fn xml_unescape(value: &str) -> String {
    value
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RedisEndpoint {
    host: String,
    port: u16,
    database: Option<u32>,
}

impl RedisEndpoint {
    fn parse(url: &str) -> Result<Self> {
        let url = url.trim();
        let Some(rest) = url.strip_prefix("redis://") else {
            return Err(WorldForgeError::InvalidState(format!(
                "unsupported Redis URL '{url}': expected redis://host[:port][/db]"
            )));
        };

        if rest.is_empty() {
            return Err(WorldForgeError::InvalidState(
                "redis URL is missing a host".to_string(),
            ));
        }
        if rest.contains('@') {
            return Err(WorldForgeError::InvalidState(
                "redis URLs with credentials are not supported".to_string(),
            ));
        }
        if rest.contains('?') || rest.contains('#') {
            return Err(WorldForgeError::InvalidState(
                "redis URLs with query parameters or fragments are not supported".to_string(),
            ));
        }

        let (authority, path) = match rest.split_once('/') {
            Some((authority, path)) => (authority, path),
            None => (rest, ""),
        };
        let (host, port) = parse_redis_authority(authority)?;
        let database = parse_redis_database(path)?;

        Ok(Self {
            host,
            port,
            database,
        })
    }

    fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

fn parse_redis_authority(authority: &str) -> Result<(String, u16)> {
    if authority.is_empty() {
        return Err(WorldForgeError::InvalidState(
            "redis URL is missing a host".to_string(),
        ));
    }

    if let Some(stripped) = authority.strip_prefix('[') {
        let Some((host, remainder)) = stripped.split_once(']') else {
            return Err(WorldForgeError::InvalidState(
                "redis IPv6 host must be bracketed".to_string(),
            ));
        };

        let port = match remainder {
            "" => 6379,
            rest if rest.starts_with(':') => rest[1..].parse::<u16>().map_err(|_| {
                WorldForgeError::InvalidState(format!("invalid Redis port in URL '{authority}'"))
            })?,
            _ => {
                return Err(WorldForgeError::InvalidState(format!(
                    "unexpected Redis authority suffix '{remainder}'"
                )))
            }
        };

        if host.is_empty() {
            return Err(WorldForgeError::InvalidState(
                "redis IPv6 host is empty".to_string(),
            ));
        }

        return Ok((host.to_string(), port));
    }

    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() && !host.contains(':') => {
            let port = port.parse::<u16>().map_err(|_| {
                WorldForgeError::InvalidState(format!("invalid Redis port in URL '{authority}'"))
            })?;
            (host, port)
        }
        Some((host, _)) if host.contains(':') => {
            return Err(WorldForgeError::InvalidState(
                "redis IPv6 hosts must be bracketed".to_string(),
            ))
        }
        _ => (authority, 6379),
    };

    if host.is_empty() {
        return Err(WorldForgeError::InvalidState(
            "redis URL is missing a host".to_string(),
        ));
    }

    Ok((host.to_string(), port))
}

fn parse_redis_database(path: &str) -> Result<Option<u32>> {
    let path = path.trim();
    if path.is_empty() {
        return Ok(None);
    }
    if path.contains('/') {
        return Err(WorldForgeError::InvalidState(
            "redis URLs support at most one database path segment".to_string(),
        ));
    }

    let database = path.parse::<u32>().map_err(|_| {
        WorldForgeError::InvalidState(format!("invalid Redis database index '{path}' in URL"))
    })?;
    Ok(Some(database))
}

#[derive(Debug)]
struct RedisConnection {
    stream: BufReader<TcpStream>,
}

#[derive(Debug)]
enum RedisValue {
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Vec<u8>),
    Array(Vec<RedisValue>),
    Null,
}

impl RedisConnection {
    async fn connect(endpoint: &RedisEndpoint) -> Result<Self> {
        let stream = TcpStream::connect(endpoint.address())
            .await
            .map_err(|error| {
                WorldForgeError::InternalError(format!("failed to connect to Redis: {error}"))
            })?;

        Ok(Self {
            stream: BufReader::new(stream),
        })
    }

    async fn command(&mut self, args: &[&[u8]]) -> Result<RedisValue> {
        let payload = encode_redis_command(args);
        self.stream
            .get_mut()
            .write_all(&payload)
            .await
            .map_err(|error| {
                WorldForgeError::InternalError(format!("failed to write Redis command: {error}"))
            })?;
        read_redis_value(&mut self.stream).await
    }
}

fn encode_redis_command(args: &[&[u8]]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(format!("*{}\r\n", args.len()).as_bytes());
    for arg in args {
        payload.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
        payload.extend_from_slice(arg);
        payload.extend_from_slice(b"\r\n");
    }
    payload
}

async fn read_redis_line(reader: &mut BufReader<TcpStream>) -> Result<String> {
    let mut line = String::new();
    let bytes = reader.read_line(&mut line).await.map_err(|error| {
        WorldForgeError::InternalError(format!("failed to read Redis line: {error}"))
    })?;

    if bytes == 0 {
        return Err(WorldForgeError::InternalError(
            "unexpected EOF while reading Redis response".to_string(),
        ));
    }

    if line.ends_with('\n') {
        line.pop();
    }
    if line.ends_with('\r') {
        line.pop();
    }

    Ok(line)
}

fn read_redis_value<'a>(
    reader: &'a mut BufReader<TcpStream>,
) -> Pin<Box<dyn Future<Output = Result<RedisValue>> + Send + 'a>> {
    Box::pin(async move {
        let mut prefix = [0u8; 1];
        reader.read_exact(&mut prefix).await.map_err(|error| {
            WorldForgeError::InternalError(format!("failed to read Redis response: {error}"))
        })?;

        match prefix[0] {
            b'+' => Ok(RedisValue::SimpleString(read_redis_line(reader).await?)),
            b'-' => Ok(RedisValue::Error(read_redis_line(reader).await?)),
            b':' => {
                let line = read_redis_line(reader).await?;
                let value = line.parse::<i64>().map_err(|_| {
                    WorldForgeError::InvalidState(format!(
                        "invalid Redis integer response '{line}'"
                    ))
                })?;
                Ok(RedisValue::Integer(value))
            }
            b'$' => {
                let line = read_redis_line(reader).await?;
                let length = line.parse::<isize>().map_err(|_| {
                    WorldForgeError::InvalidState(format!("invalid Redis bulk length '{line}'"))
                })?;
                if length < 0 {
                    return Ok(RedisValue::Null);
                }
                let mut payload = vec![0u8; length as usize];
                reader.read_exact(&mut payload).await.map_err(|error| {
                    WorldForgeError::InternalError(format!(
                        "failed to read Redis bulk payload: {error}"
                    ))
                })?;
                let mut crlf = [0u8; 2];
                reader.read_exact(&mut crlf).await.map_err(|error| {
                    WorldForgeError::InternalError(format!(
                        "failed to read Redis bulk terminator: {error}"
                    ))
                })?;
                if crlf != *b"\r\n" {
                    return Err(WorldForgeError::InvalidState(
                        "invalid Redis bulk string terminator".to_string(),
                    ));
                }
                Ok(RedisValue::BulkString(payload))
            }
            b'*' => {
                let line = read_redis_line(reader).await?;
                let count = line.parse::<isize>().map_err(|_| {
                    WorldForgeError::InvalidState(format!("invalid Redis array length '{line}'"))
                })?;
                if count < 0 {
                    return Ok(RedisValue::Null);
                }

                let mut values = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    values.push(read_redis_value(reader).await?);
                }
                Ok(RedisValue::Array(values))
            }
            other => Err(WorldForgeError::InvalidState(format!(
                "unsupported Redis response prefix '{}'",
                other as char
            ))),
        }
    })
}

fn redis_error(message: impl Into<String>) -> WorldForgeError {
    WorldForgeError::InvalidState(message.into())
}

fn expect_redis_ok(value: RedisValue) -> Result<()> {
    match value {
        RedisValue::SimpleString(message) if message.eq_ignore_ascii_case("OK") => Ok(()),
        RedisValue::BulkString(bytes) if bytes.eq_ignore_ascii_case(b"OK") => Ok(()),
        RedisValue::Integer(1) => Ok(()),
        RedisValue::Error(message) => Err(WorldForgeError::InternalError(message)),
        other => Err(redis_error(format!("unexpected Redis response: {other:?}"))),
    }
}

fn expect_redis_pong(value: RedisValue) -> Result<()> {
    match value {
        RedisValue::SimpleString(message) if message.eq_ignore_ascii_case("PONG") => Ok(()),
        RedisValue::BulkString(bytes) if bytes.eq_ignore_ascii_case(b"PONG") => Ok(()),
        RedisValue::Error(message) => Err(WorldForgeError::InternalError(message)),
        other => Err(redis_error(format!("unexpected Redis response: {other:?}"))),
    }
}

fn redis_key_for_world(id: &WorldId) -> String {
    format!("{REDIS_STATE_KEY_PREFIX}{id}")
}

#[derive(Debug, Clone)]
pub struct RedisStateStore {
    /// Redis connection URL used to establish short-lived command connections.
    pub url: String,
    endpoint: RedisEndpoint,
}

impl RedisStateStore {
    /// Create a new Redis-backed state store from a Redis connection URL.
    pub async fn new(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        let endpoint = RedisEndpoint::parse(&url)?;
        let store = Self { url, endpoint };
        let response = store.command(&[b"PING"]).await?;
        expect_redis_pong(response)?;
        Ok(store)
    }

    async fn with_connection(&self) -> Result<RedisConnection> {
        let mut connection = RedisConnection::connect(&self.endpoint).await?;

        if let Some(database) = self.endpoint.database {
            let database = database.to_string();
            let response = connection
                .command(&[b"SELECT", database.as_bytes()])
                .await?;
            expect_redis_ok(response)?;
        }

        Ok(connection)
    }

    async fn command(&self, args: &[&[u8]]) -> Result<RedisValue> {
        let mut connection = self.with_connection().await?;
        connection.command(args).await
    }
}

#[async_trait::async_trait]
impl StateStore for RedisStateStore {
    async fn save(&self, state: &WorldState) -> Result<()> {
        let normalized = normalize_world_state(state.clone())?;

        let payload = serde_json::to_vec(&normalized)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;
        let key = redis_key_for_world(&normalized.id);
        let id = normalized.id.to_string();

        let response = self
            .command(&[b"SET", key.as_bytes(), payload.as_slice()])
            .await?;
        expect_redis_ok(response)?;

        let response = self
            .command(&[b"SADD", REDIS_STATE_INDEX_KEY.as_bytes(), id.as_bytes()])
            .await?;
        match response {
            RedisValue::Integer(value) if value >= 0 => Ok(()),
            RedisValue::Error(message) => Err(WorldForgeError::InternalError(message)),
            other => Err(redis_error(format!(
                "unexpected Redis response for SADD: {other:?}"
            ))),
        }
    }

    async fn load(&self, id: &WorldId) -> Result<WorldState> {
        let key = redis_key_for_world(id);
        let response = self.command(&[b"GET", key.as_bytes()]).await?;
        match response {
            RedisValue::BulkString(bytes) => deserialize_world_state(StateFileFormat::Json, &bytes),
            RedisValue::Null => Err(WorldForgeError::WorldNotFound(*id)),
            RedisValue::Error(message) => Err(WorldForgeError::InternalError(message)),
            other => Err(redis_error(format!(
                "unexpected Redis response for GET: {other:?}"
            ))),
        }
    }

    async fn list(&self) -> Result<Vec<WorldId>> {
        let response = self
            .command(&[b"SMEMBERS", REDIS_STATE_INDEX_KEY.as_bytes()])
            .await?;

        let mut ids = HashSet::new();
        let items = match response {
            RedisValue::Array(items) => items,
            RedisValue::Null => Vec::new(),
            RedisValue::Error(message) => return Err(WorldForgeError::InternalError(message)),
            other => {
                return Err(redis_error(format!(
                    "unexpected Redis response for SMEMBERS: {other:?}"
                )))
            }
        };

        for item in items {
            let key = match item {
                RedisValue::BulkString(bytes) => String::from_utf8(bytes)
                    .map_err(|error| WorldForgeError::SerializationError(error.to_string()))?,
                RedisValue::SimpleString(text) => text,
                RedisValue::Error(message) => return Err(WorldForgeError::InternalError(message)),
                other => {
                    return Err(redis_error(format!(
                        "unexpected Redis key response: {other:?}"
                    )))
                }
            };

            if let Ok(id) = key.parse::<WorldId>() {
                ids.insert(id);
            }
        }

        let mut ids = ids.into_iter().collect::<Vec<_>>();
        ids.sort_unstable_by_key(|id| id.as_u128());
        Ok(ids)
    }

    async fn delete(&self, id: &WorldId) -> Result<()> {
        let key = redis_key_for_world(id);
        let response = self.command(&[b"DEL", key.as_bytes()]).await?;
        match response {
            RedisValue::Integer(0) => Err(WorldForgeError::WorldNotFound(*id)),
            RedisValue::Integer(_) => {
                let id = id.to_string();
                let response = self
                    .command(&[b"SREM", REDIS_STATE_INDEX_KEY.as_bytes(), id.as_bytes()])
                    .await?;
                match response {
                    RedisValue::Integer(value) if value >= 0 => Ok(()),
                    RedisValue::Error(message) => Err(WorldForgeError::InternalError(message)),
                    other => Err(redis_error(format!(
                        "unexpected Redis response for SREM: {other:?}"
                    ))),
                }
            }
            RedisValue::Error(message) => Err(WorldForgeError::InternalError(message)),
            other => Err(redis_error(format!(
                "unexpected Redis response for DEL: {other:?}"
            ))),
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
        let normalized = normalize_world_state(state.clone())?;
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
        let normalized = normalize_world_state(state.clone())?;
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
            Some((json,)) => deserialize_world_state(StateFileFormat::Json, json.as_bytes()),
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
    use std::collections::{HashMap, HashSet};
    use std::net::SocketAddr;
    use std::sync::Arc;

    use uuid::Uuid;

    use super::*;
    use crate::prediction::{Plan, PlanGoal, PlanRequest, PlannerType, StoredPlanRecord};
    use tokio::net::TcpListener;
    use tokio::sync::Mutex;
    use tokio::task::JoinHandle;

    fn legacy_world_state_without_snapshot() -> WorldState {
        let mut state = WorldState::new("legacy-world", "mock");
        state.history.push(HistoryEntry {
            time: state.time,
            state_hash: [7; 32],
            action: None,
            prediction: None,
            provider: "mock".to_string(),
            snapshot: None,
        });
        state
    }

    #[test]
    fn test_world_state_new() {
        let ws = WorldState::new("test-world", "mock");
        assert_eq!(ws.metadata.name, "test-world");
        assert_eq!(ws.metadata.created_by, "mock");
        assert_eq!(ws.time.step, 0);
        assert!(ws.history.is_empty());
        assert!(ws.list_stored_plans().is_empty());
    }

    fn sample_stored_plan_record(id: PlanId, name: &str) -> StoredPlanRecord {
        let state = WorldState::new(name, "mock");
        let request = PlanRequest {
            current_state: state.clone(),
            goal: PlanGoal::Description(format!("reach {name}")),
            max_steps: 2,
            guardrails: Vec::new(),
            planner: PlannerType::Sampling {
                num_samples: 1,
                top_k: 1,
            },
            timeout_seconds: 5.0,
            fallback_provider: None,
        };
        let plan = Plan {
            actions: Vec::new(),
            predicted_states: vec![state],
            predicted_videos: None,
            total_cost: 0.0,
            success_probability: 1.0,
            guardrail_compliance: Vec::new(),
            planning_time_ms: 0,
            iterations_used: 0,
            stored_plan_id: Some(id),
            verification_proof: None,
        };
        StoredPlanRecord::from_request("mock", &request, &plan)
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
        assert!(latest.snapshot.is_some());
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
            snapshot: None,
        });

        let repaired = state.ensure_current_state_recorded("mock").unwrap();

        assert!(repaired);
        assert_eq!(state.history.len(), 1);
        assert_ne!(state.history.latest().unwrap().state_hash, [7; 32]);
        assert!(state.history.latest().unwrap().snapshot.is_some());
    }

    #[test]
    fn test_normalize_world_state_bootstraps_empty_history() {
        let state = WorldState::new("normalize-empty", "mock");

        let normalized = normalize_world_state(state).unwrap();

        assert_eq!(normalized.history.len(), 1);
        let latest = normalized.history.latest().unwrap();
        assert_eq!(latest.provider, "mock");
        assert!(latest.snapshot.is_some());
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
                snapshot: None,
            });
        }
        assert_eq!(history.len(), 3);
        assert_eq!(history.latest().unwrap().time.step, 4);
    }

    #[test]
    fn test_world_state_history_state_restores_prior_checkpoint() {
        let mut state = WorldState::new("restore", "mock");
        state.ensure_history_initialized("mock").unwrap();

        state.time = SimTime {
            step: 1,
            seconds: 0.5,
            dt: 0.5,
        };
        state.metadata.name = "restore-step-1".to_string();
        state.record_current_state(None, None, "mock").unwrap();

        state.time = SimTime {
            step: 2,
            seconds: 1.0,
            dt: 0.5,
        };
        state.metadata.name = "restore-step-2".to_string();
        state.record_current_state(None, None, "mock").unwrap();

        let restored = state.history_state(1).unwrap();
        assert_eq!(restored.time.step, 1);
        assert_eq!(restored.metadata.name, "restore-step-1");
        assert_eq!(restored.history.len(), 2);
        assert_eq!(restored.history.latest().unwrap().time.step, 1);
    }

    #[test]
    fn test_world_state_restore_history_mutates_in_place() {
        let mut state = WorldState::new("restore-in-place", "mock");
        state.ensure_history_initialized("mock").unwrap();

        state.time = SimTime {
            step: 1,
            seconds: 0.25,
            dt: 0.25,
        };
        state.metadata.name = "step-one".to_string();
        state.record_current_state(None, None, "mock").unwrap();

        state.time = SimTime {
            step: 2,
            seconds: 0.5,
            dt: 0.25,
        };
        state.metadata.name = "step-two".to_string();
        state.record_current_state(None, None, "mock").unwrap();

        state.restore_history(0).unwrap();
        assert_eq!(state.time.step, 0);
        assert_eq!(state.metadata.name, "restore-in-place");
        assert_eq!(state.history.len(), 1);
    }

    #[test]
    fn test_world_state_stored_plan_lifecycle() {
        let mut state = WorldState::new("plan-world", "mock");
        let first_id = Uuid::from_u128(1);
        let second_id = Uuid::from_u128(2);
        let first = sample_stored_plan_record(first_id, "first");
        let second = sample_stored_plan_record(second_id, "second");

        assert_eq!(state.store_plan_record(first.clone()), first_id);
        assert_eq!(state.store_plan_record(second.clone()), second_id);

        let listed = state.list_stored_plans();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].id, first_id);
        assert_eq!(listed[1].id, second_id);

        let fetched = state.get_stored_plan(&second_id).unwrap();
        assert_eq!(fetched.goal_summary, second.goal_summary);
        assert_eq!(state.stored_plan(&first_id).unwrap().provider, "mock");

        let removed = state.remove_stored_plan(&first_id).unwrap();
        assert_eq!(removed.id, first_id);
        assert!(state.get_stored_plan(&first_id).is_none());
        assert_eq!(state.list_stored_plans().len(), 1);
        assert!(state.remove_stored_plan(&first_id).is_none());
    }

    #[test]
    fn test_world_state_fork_creates_fresh_history() {
        let mut state = WorldState::new("source", "primary");
        state.ensure_history_initialized("primary").unwrap();
        state.metadata.description = "original description".to_string();
        state.metadata.tags.push("tagged".to_string());
        state.time = SimTime {
            step: 3,
            seconds: 1.5,
            dt: 0.5,
        };
        state.metadata.name = "source world".to_string();
        state.record_current_state(None, None, "primary").unwrap();

        let forked = state.fork(None).unwrap();

        assert_ne!(forked.id, state.id);
        assert_eq!(forked.metadata.name, "source world Fork");
        assert_eq!(forked.metadata.description, state.metadata.description);
        assert_eq!(forked.metadata.tags, state.metadata.tags);
        assert_eq!(forked.metadata.created_by, "primary");
        assert!(forked.metadata.created_at >= state.metadata.created_at);
        assert_eq!(forked.time, state.time);
        assert_eq!(forked.scene.objects.len(), state.scene.objects.len());
        assert_eq!(forked.history.len(), 1);
        let latest = forked.history.latest().unwrap();
        assert_eq!(latest.provider, "primary");
        assert_eq!(latest.time, forked.time);
        assert!(latest.snapshot.is_some());
    }

    #[test]
    fn test_world_state_fork_from_history_uses_checkpoint_state() {
        let mut state = WorldState::new("source", "primary");
        state.ensure_history_initialized("primary").unwrap();

        state.time = SimTime {
            step: 1,
            seconds: 0.5,
            dt: 0.5,
        };
        state.metadata.name = "checkpoint".to_string();
        state.metadata.description = "checkpoint description".to_string();
        state.record_current_state(None, None, "secondary").unwrap();

        state.time = SimTime {
            step: 2,
            seconds: 1.0,
            dt: 0.5,
        };
        state.metadata.name = "latest".to_string();
        state.record_current_state(None, None, "tertiary").unwrap();

        let forked = state.fork_from_history(1, Some("branch-a")).unwrap();

        assert_ne!(forked.id, state.id);
        assert_eq!(forked.metadata.name, "branch-a");
        assert_eq!(forked.metadata.description, "checkpoint description");
        assert_eq!(forked.metadata.created_by, "secondary");
        assert_eq!(forked.time.step, 1);
        assert_eq!(forked.history.len(), 1);
        assert_eq!(forked.history.latest().unwrap().provider, "secondary");
    }

    #[test]
    fn test_world_state_history_state_rejects_missing_legacy_snapshot() {
        let mut state = WorldState::new("legacy-restore", "mock");
        state.history.push(HistoryEntry {
            time: state.time,
            state_hash: [1; 32],
            action: None,
            prediction: None,
            provider: "mock".to_string(),
            snapshot: None,
        });
        state.history.push(HistoryEntry {
            time: SimTime {
                step: 1,
                seconds: 1.0,
                dt: 1.0,
            },
            state_hash: [2; 32],
            action: None,
            prediction: None,
            provider: "mock".to_string(),
            snapshot: None,
        });

        let error = state.history_state(0).unwrap_err();
        assert!(
            matches!(error, WorldForgeError::InvalidState(message) if message.contains("recoverable snapshot"))
        );
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
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let metadata = inspect_world_state_snapshot(StateFileFormat::Json, &bytes).unwrap();
        let restored = deserialize_world_state(StateFileFormat::Json, &bytes).unwrap();

        assert_eq!(
            payload["schema_version"],
            serde_json::json!(WORLD_STATE_SNAPSHOT_SCHEMA_VERSION)
        );
        assert_eq!(payload["state"]["metadata"]["name"], state.metadata.name);
        assert_eq!(metadata.schema_version, WORLD_STATE_SNAPSHOT_SCHEMA_VERSION);
        assert!(!metadata.legacy_payload);
        assert_eq!(restored.id, state.id);
        assert_eq!(restored.metadata.name, state.metadata.name);
        assert_eq!(restored.metadata.description, state.metadata.description);
        assert_eq!(restored.history.len(), 1);
        assert!(restored.history.latest().unwrap().snapshot.is_some());
    }

    #[test]
    fn test_world_state_snapshot_codec_roundtrip_msgpack() {
        let mut state = WorldState::new("codec-msgpack", "mock");
        state.metadata.description = "msgpack roundtrip".to_string();

        let bytes = serialize_world_state(StateFileFormat::MessagePack, &state).unwrap();
        let metadata = inspect_world_state_snapshot(StateFileFormat::MessagePack, &bytes).unwrap();
        let restored = deserialize_world_state(StateFileFormat::MessagePack, &bytes).unwrap();

        assert_eq!(metadata.schema_version, WORLD_STATE_SNAPSHOT_SCHEMA_VERSION);
        assert!(!metadata.legacy_payload);
        assert_eq!(restored.id, state.id);
        assert_eq!(restored.metadata.name, state.metadata.name);
        assert_eq!(restored.metadata.description, state.metadata.description);
        assert_eq!(restored.history.len(), 1);
        assert!(restored.history.latest().unwrap().snapshot.is_some());
    }

    #[test]
    fn test_world_state_snapshot_codec_normalizes_legacy_history() {
        let state = legacy_world_state_without_snapshot();
        let bytes = serde_json::to_vec_pretty(&state).unwrap();
        let metadata = inspect_world_state_snapshot(StateFileFormat::Json, &bytes).unwrap();

        let restored = deserialize_world_state(StateFileFormat::Json, &bytes).unwrap();
        assert_eq!(metadata.schema_version, WORLD_STATE_SNAPSHOT_SCHEMA_VERSION);
        assert!(metadata.legacy_payload);
        assert_eq!(restored.id, state.id);
        assert_eq!(restored.history.len(), 1);
        assert!(restored.history.latest().unwrap().snapshot.is_some());
        assert_ne!(restored.history.latest().unwrap().state_hash, [7; 32]);
    }

    #[test]
    fn test_world_state_snapshot_codec_accepts_legacy_msgpack_payload() {
        let state = legacy_world_state_without_snapshot();
        let bytes = rmp_serde::to_vec_named(&state).unwrap();
        let metadata = inspect_world_state_snapshot(StateFileFormat::MessagePack, &bytes).unwrap();
        let restored = deserialize_world_state(StateFileFormat::MessagePack, &bytes).unwrap();

        assert_eq!(metadata.schema_version, WORLD_STATE_SNAPSHOT_SCHEMA_VERSION);
        assert!(metadata.legacy_payload);
        assert_eq!(restored.id, state.id);
        assert_eq!(restored.history.len(), 1);
        assert!(restored.history.latest().unwrap().snapshot.is_some());
    }

    #[test]
    fn test_world_state_snapshot_codec_rejects_invalid_bytes() {
        let err = deserialize_world_state(StateFileFormat::Json, b"not json").unwrap_err();
        assert!(matches!(err, WorldForgeError::SerializationError(_)));
    }

    #[test]
    fn test_world_state_snapshot_codec_rejects_future_schema_version() {
        let state = WorldState::new("future-snapshot", "mock");
        let bytes = serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": WORLD_STATE_SNAPSHOT_SCHEMA_VERSION + 1,
            "state": state,
        }))
        .unwrap();

        let error = deserialize_world_state(StateFileFormat::Json, &bytes).unwrap_err();
        assert!(matches!(
            error,
            WorldForgeError::InvalidState(message)
                if message.contains("unsupported world-state snapshot schema version")
        ));
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

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn test_sqlite_state_store_load_normalizes_legacy_raw_record() {
        let store = SqliteStateStore::new("sqlite::memory:").await.unwrap();
        let state = WorldState::new("sqlite-legacy-load", "mock");
        let id = state.id;

        sqlx::query("INSERT INTO world_states (id, state) VALUES (?, ?)")
            .bind(id.to_string())
            .bind(serde_json::to_string(&state).unwrap())
            .execute(&store.pool)
            .await
            .unwrap();

        let loaded = store.load(&id).await.unwrap();
        assert_eq!(loaded.history.len(), 1);
        assert!(loaded.history.latest().unwrap().snapshot.is_some());
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
    async fn test_file_state_store_backfills_latest_legacy_snapshot() {
        let dir = std::env::temp_dir().join(format!("worldforge-legacy-{}", uuid::Uuid::new_v4()));
        let store = FileStateStore::new(&dir);

        let state = legacy_world_state_without_snapshot();
        let payload = serialize_world_state(StateFileFormat::Json, &state).unwrap();
        let path = store.state_path_for_format(&state.id, StateFileFormat::Json);

        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(&path, payload).await.unwrap();

        let loaded = store.load(&state.id).await.unwrap();
        assert_eq!(loaded.id, state.id);
        assert_eq!(loaded.history.len(), 1);
        assert!(loaded.history.latest().unwrap().snapshot.is_some());

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

    #[tokio::test]
    async fn test_redis_state_store_rejects_invalid_url() {
        let error = RedisStateStore::new("http://127.0.0.1:6379")
            .await
            .unwrap_err();
        assert!(
            matches!(error, WorldForgeError::InvalidState(message) if message.contains("redis://"))
        );
    }

    #[tokio::test]
    async fn test_redis_state_store_roundtrip_with_fake_server() {
        let server = FakeRedisServer::spawn().await;
        let store = StateStoreKind::Redis(format!("redis://{}/1", server.address))
            .open()
            .await
            .unwrap();

        let first = WorldState::new("redis-a", "mock");
        let second = WorldState::new("redis-b", "mock");

        store.save(&first).await.unwrap();
        store.save(&second).await.unwrap();

        let loaded = store.load(&first.id).await.unwrap();
        assert_eq!(loaded.id, first.id);
        assert_eq!(loaded.metadata.name, "redis-a");

        let listed = store.list().await.unwrap();
        let mut expected = vec![first.id, second.id];
        expected.sort_unstable_by_key(|id| id.as_u128());
        assert_eq!(listed, expected);

        store.delete(&first.id).await.unwrap();
        assert!(matches!(
            store.load(&first.id).await.unwrap_err(),
            WorldForgeError::WorldNotFound(id) if id == first.id
        ));

        let commands = server.commands.lock().await;
        assert!(commands
            .iter()
            .any(|command| command == &vec!["SELECT".to_string(), "1".to_string()]));
        assert!(commands
            .iter()
            .any(|command| command.first().map(String::as_str) == Some("PING")));
        assert!(commands
            .iter()
            .any(|command| command.first().map(String::as_str) == Some("SADD")));
        assert!(commands
            .iter()
            .any(|command| command.first().map(String::as_str) == Some("SMEMBERS")));
        assert!(commands
            .iter()
            .any(|command| command.first().map(String::as_str) == Some("DEL")));
        assert!(commands
            .iter()
            .any(|command| command.first().map(String::as_str) == Some("SREM")));
    }

    #[tokio::test]
    async fn test_s3_state_store_roundtrip_with_fake_server() {
        let server = FakeS3Server::spawn().await;
        let config = test_s3_config(server.endpoint());
        let store = StateStoreKind::S3 {
            config,
            format: StateFileFormat::Json,
        }
        .open()
        .await
        .unwrap();

        let state = WorldState::new("s3-world", "mock");
        let id = state.id;

        store.save(&state).await.unwrap();

        let loaded = store.load(&id).await.unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.metadata.name, "s3-world");
        assert_eq!(loaded.history.len(), 1);

        let ids = store.list().await.unwrap();
        assert_eq!(ids, vec![id]);

        store.delete(&id).await.unwrap();
        assert!(matches!(
            store.load(&id).await.unwrap_err(),
            WorldForgeError::WorldNotFound(found) if found == id
        ));

        let requests = server.requests.lock().await;
        assert!(requests.iter().any(|request| request.method == "PUT"));
        assert!(requests.iter().any(|request| {
            request
                .path
                .ends_with(&format!("/worldforge-tests/states/{id}.json"))
        }));
        assert!(requests
            .iter()
            .any(|request| request.query.contains("list-type=2")));
        assert!(requests
            .iter()
            .all(|request| request.headers.get("authorization").is_some_and(
                |value| value.starts_with("AWS4-HMAC-SHA256 Credential=test-access/")
            )));
        assert!(requests
            .iter()
            .all(|request| request.headers.contains_key("x-amz-date")));
        assert!(requests
            .iter()
            .all(|request| request.headers.contains_key("x-amz-content-sha256")));
    }

    #[tokio::test]
    async fn test_s3_state_store_loads_alternate_format_and_deduplicates_listing() {
        let server = FakeS3Server::spawn().await;
        let json_store =
            S3StateStore::new(test_s3_config(server.endpoint()), StateFileFormat::Json).unwrap();
        let msgpack_store = S3StateStore::new(
            test_s3_config(server.endpoint()),
            StateFileFormat::MessagePack,
        )
        .unwrap();

        let state = WorldState::new("s3-alt", "mock");
        let id = state.id;

        json_store.save(&state).await.unwrap();
        let loaded = msgpack_store.load(&id).await.unwrap();
        assert_eq!(loaded.id, id);

        msgpack_store.save(&state).await.unwrap();
        assert_eq!(json_store.list().await.unwrap(), vec![id]);

        json_store.delete(&id).await.unwrap();
        assert!(matches!(
            json_store.load(&id).await.unwrap_err(),
            WorldForgeError::WorldNotFound(found) if found == id
        ));
    }

    fn test_s3_config(endpoint: &str) -> S3Config {
        S3Config {
            bucket: "worldforge-tests".to_string(),
            region: "us-east-1".to_string(),
            endpoint: Some(endpoint.to_string()),
            access_key_id: "test-access".to_string(),
            secret_access_key: "test-secret".to_string(),
            session_token: Some("test-session".to_string()),
            prefix: "states".to_string(),
        }
    }

    #[derive(Debug, Clone)]
    struct RecordedS3Request {
        method: String,
        path: String,
        query: String,
        headers: HashMap<String, String>,
    }

    struct FakeS3Server {
        endpoint: String,
        requests: Arc<Mutex<Vec<RecordedS3Request>>>,
        handle: JoinHandle<()>,
    }

    impl FakeS3Server {
        async fn spawn() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let endpoint = format!("http://{address}");
            let requests = Arc::new(Mutex::new(Vec::new()));
            let objects = Arc::new(Mutex::new(HashMap::new()));
            let requests_for_task = Arc::clone(&requests);
            let objects_for_task = Arc::clone(&objects);

            let handle = tokio::spawn(async move {
                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(pair) => pair,
                        Err(_) => break,
                    };
                    let requests = Arc::clone(&requests_for_task);
                    let objects = Arc::clone(&objects_for_task);
                    tokio::spawn(async move {
                        let _ = handle_fake_s3_connection(stream, requests, objects).await;
                    });
                }
            });

            Self {
                endpoint,
                requests,
                handle,
            }
        }

        fn endpoint(&self) -> &str {
            &self.endpoint
        }
    }

    impl Drop for FakeS3Server {
        fn drop(&mut self) {
            self.handle.abort();
        }
    }

    async fn handle_fake_s3_connection(
        stream: tokio::net::TcpStream,
        requests: Arc<Mutex<Vec<RecordedS3Request>>>,
        objects: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    ) -> Result<()> {
        let mut reader = BufReader::new(stream);
        let mut request_line = String::new();
        if reader
            .read_line(&mut request_line)
            .await
            .map_err(|error| WorldForgeError::InternalError(error.to_string()))?
            == 0
        {
            return Ok(());
        }

        let mut parts = request_line.split_whitespace();
        let method = parts
            .next()
            .ok_or_else(|| WorldForgeError::InvalidState("missing fake s3 method".to_string()))?
            .to_string();
        let target = parts
            .next()
            .ok_or_else(|| WorldForgeError::InvalidState("missing fake s3 target".to_string()))?
            .to_string();

        let mut headers = HashMap::new();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            let read = reader
                .read_line(&mut line)
                .await
                .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
            if read == 0 || line == "\r\n" {
                break;
            }

            let trimmed = line.trim_end();
            if let Some((name, value)) = trimmed.split_once(':') {
                let key = name.trim().to_ascii_lowercase();
                let value = value.trim().to_string();
                if key == "content-length" {
                    content_length = value.parse::<usize>().map_err(|error| {
                        WorldForgeError::InvalidState(format!(
                            "invalid fake s3 content length '{value}': {error}"
                        ))
                    })?;
                }
                headers.insert(key, value);
            }
        }

        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader
                .read_exact(&mut body)
                .await
                .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
        }

        let (path, query) = match target.split_once('?') {
            Some((path, query)) => (path.to_string(), query.to_string()),
            None => (target, String::new()),
        };

        requests.lock().await.push(RecordedS3Request {
            method: method.clone(),
            path: path.clone(),
            query: query.clone(),
            headers: headers.clone(),
        });

        let mut stream = reader.into_inner();
        let query_params = parse_fake_query(&query);
        if query_params.get("list-type").map(String::as_str) == Some("2") {
            let prefix = query_params.get("prefix").cloned().unwrap_or_default();
            let objects = objects.lock().await;
            let body = build_fake_s3_list_response(&objects, &prefix);
            write_fake_http_response(
                &mut stream,
                200,
                "OK",
                body.as_bytes(),
                Some("application/xml"),
            )
            .await?;
            return Ok(());
        }

        let key = fake_s3_object_key(&path);
        match method.as_str() {
            "PUT" => {
                objects.lock().await.insert(key, body);
                write_fake_http_response(&mut stream, 200, "OK", b"", None).await?;
            }
            "GET" => {
                if let Some(payload) = objects.lock().await.get(&key).cloned() {
                    write_fake_http_response(
                        &mut stream,
                        200,
                        "OK",
                        &payload,
                        Some("application/octet-stream"),
                    )
                    .await?;
                } else {
                    write_fake_http_response(&mut stream, 404, "Not Found", b"", None).await?;
                }
            }
            "HEAD" => {
                let status = if objects.lock().await.contains_key(&key) {
                    200
                } else {
                    404
                };
                let reason = if status == 200 { "OK" } else { "Not Found" };
                write_fake_http_response(&mut stream, status, reason, b"", None).await?;
            }
            "DELETE" => {
                objects.lock().await.remove(&key);
                write_fake_http_response(&mut stream, 204, "No Content", b"", None).await?;
            }
            other => {
                let body = format!("unsupported method: {other}");
                write_fake_http_response(
                    &mut stream,
                    405,
                    "Method Not Allowed",
                    body.as_bytes(),
                    Some("text/plain"),
                )
                .await?;
            }
        }

        Ok(())
    }

    fn fake_s3_object_key(path: &str) -> String {
        path.trim_start_matches('/')
            .split_once('/')
            .map(|(_, key)| percent_decode(key))
            .unwrap_or_default()
    }

    fn parse_fake_query(query: &str) -> HashMap<String, String> {
        query
            .split('&')
            .filter(|pair| !pair.is_empty())
            .map(|pair| match pair.split_once('=') {
                Some((key, value)) => (percent_decode(key), percent_decode(value)),
                None => (percent_decode(pair), String::new()),
            })
            .collect()
    }

    fn build_fake_s3_list_response(objects: &HashMap<String, Vec<u8>>, prefix: &str) -> String {
        let mut keys = objects
            .keys()
            .filter(|key| key.starts_with(prefix))
            .cloned()
            .collect::<Vec<_>>();
        keys.sort();

        let contents = keys
            .iter()
            .map(|key| format!("<Contents><Key>{key}</Key></Contents>"))
            .collect::<String>();

        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult><Name>worldforge-tests</Name><Prefix>{prefix}</Prefix><KeyCount>{}</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>{contents}</ListBucketResult>",
            keys.len()
        )
    }

    fn percent_decode(value: &str) -> String {
        let bytes = value.as_bytes();
        let mut index = 0usize;
        let mut decoded = Vec::with_capacity(bytes.len());
        while index < bytes.len() {
            match bytes[index] {
                b'%' if index + 2 < bytes.len() => {
                    if let (Some(high), Some(low)) =
                        (hex_value(bytes[index + 1]), hex_value(bytes[index + 2]))
                    {
                        decoded.push((high << 4) | low);
                        index += 3;
                        continue;
                    }
                    decoded.push(bytes[index]);
                    index += 1;
                }
                b'+' => {
                    decoded.push(b' ');
                    index += 1;
                }
                other => {
                    decoded.push(other);
                    index += 1;
                }
            }
        }

        String::from_utf8(decoded).unwrap_or_default()
    }

    fn hex_value(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    async fn write_fake_http_response(
        stream: &mut tokio::net::TcpStream,
        status: u16,
        reason: &str,
        body: &[u8],
        content_type: Option<&str>,
    ) -> Result<()> {
        let mut response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        if let Some(content_type) = content_type {
            response.push_str(&format!("Content-Type: {content_type}\r\n"));
        }
        response.push_str("\r\n");
        stream
            .write_all(response.as_bytes())
            .await
            .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
        stream
            .write_all(body)
            .await
            .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
        stream
            .flush()
            .await
            .map_err(|error| WorldForgeError::InternalError(error.to_string()))?;
        Ok(())
    }

    struct FakeRedisServer {
        address: SocketAddr,
        commands: Arc<Mutex<Vec<Vec<String>>>>,
        handle: JoinHandle<()>,
    }

    impl FakeRedisServer {
        async fn spawn() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let commands = Arc::new(Mutex::new(Vec::new()));
            let values = Arc::new(Mutex::new(HashMap::new()));
            let sets = Arc::new(Mutex::new(HashMap::new()));
            let commands_for_task = Arc::clone(&commands);
            let values_for_task = Arc::clone(&values);
            let sets_for_task = Arc::clone(&sets);

            let handle = tokio::spawn(async move {
                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(pair) => pair,
                        Err(_) => break,
                    };
                    let commands = Arc::clone(&commands_for_task);
                    let values = Arc::clone(&values_for_task);
                    let sets = Arc::clone(&sets_for_task);
                    tokio::spawn(async move {
                        let _ = handle_fake_redis_connection(stream, commands, values, sets).await;
                    });
                }
            });

            Self {
                address,
                commands,
                handle,
            }
        }
    }

    impl Drop for FakeRedisServer {
        fn drop(&mut self) {
            self.handle.abort();
        }
    }

    async fn handle_fake_redis_connection(
        stream: tokio::net::TcpStream,
        commands: Arc<Mutex<Vec<Vec<String>>>>,
        values: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        sets: Arc<Mutex<HashMap<String, HashSet<String>>>>,
    ) -> Result<()> {
        let mut reader = BufReader::new(stream);

        loop {
            let request = match read_redis_value(&mut reader).await {
                Ok(request) => request,
                Err(WorldForgeError::InternalError(message))
                    if message.contains("unexpected EOF while reading Redis response") =>
                {
                    break;
                }
                Err(error) => return Err(error),
            };

            let RedisValue::Array(items) = request else {
                return Err(WorldForgeError::InvalidState(
                    "redis request must be an array".to_string(),
                ));
            };

            let mut command = Vec::with_capacity(items.len());
            for item in items {
                command.push(redis_value_to_string(item)?);
            }
            if command.is_empty() {
                return Err(WorldForgeError::InvalidState(
                    "redis request is empty".to_string(),
                ));
            }

            commands.lock().await.push(command.clone());

            let response = match command[0].as_str() {
                "PING" => redis_simple_string_response(reader.get_mut(), "PONG").await,
                "SELECT" => redis_simple_string_response(reader.get_mut(), "OK").await,
                "SET" => {
                    if command.len() != 3 {
                        return Err(WorldForgeError::InvalidState(
                            "SET requires key and value".to_string(),
                        ));
                    }
                    let key = command[1].clone();
                    let value = command[2].clone().into_bytes();
                    values.lock().await.insert(key, value);
                    redis_simple_string_response(reader.get_mut(), "OK").await
                }
                "GET" => {
                    if command.len() != 2 {
                        return Err(WorldForgeError::InvalidState(
                            "GET requires key".to_string(),
                        ));
                    }
                    let key = &command[1];
                    let value = values.lock().await.get(key).cloned();
                    redis_bulk_response(reader.get_mut(), value.as_deref()).await
                }
                "DEL" => {
                    if command.len() != 2 {
                        return Err(WorldForgeError::InvalidState(
                            "DEL requires key".to_string(),
                        ));
                    }
                    let removed = values.lock().await.remove(&command[1]).is_some();
                    redis_integer_response(reader.get_mut(), if removed { 1 } else { 0 }).await
                }
                "SADD" => {
                    if command.len() != 3 {
                        return Err(WorldForgeError::InvalidState(
                            "SADD requires key and member".to_string(),
                        ));
                    }
                    let mut sets = sets.lock().await;
                    let members = sets.entry(command[1].clone()).or_default();
                    let inserted = members.insert(command[2].clone());
                    redis_integer_response(reader.get_mut(), if inserted { 1 } else { 0 }).await
                }
                "SMEMBERS" => {
                    if command.len() != 2 {
                        return Err(WorldForgeError::InvalidState(
                            "SMEMBERS requires key".to_string(),
                        ));
                    }
                    let mut members = sets
                        .lock()
                        .await
                        .get(&command[1])
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .collect::<Vec<_>>();
                    members.sort_unstable();
                    let members = members
                        .into_iter()
                        .map(|member| member.into_bytes())
                        .collect::<Vec<_>>();
                    redis_array_response(reader.get_mut(), &members).await
                }
                "SREM" => {
                    if command.len() != 3 {
                        return Err(WorldForgeError::InvalidState(
                            "SREM requires key and member".to_string(),
                        ));
                    }
                    let mut sets = sets.lock().await;
                    let removed = sets
                        .get_mut(&command[1])
                        .map(|members| members.remove(&command[2]))
                        .unwrap_or(false);
                    redis_integer_response(reader.get_mut(), if removed { 1 } else { 0 }).await
                }
                other => {
                    redis_error_response(reader.get_mut(), &format!("unknown command '{other}'"))
                        .await
                }
            };

            response.map_err(|error| {
                WorldForgeError::InternalError(format!("fake redis server write failed: {error}"))
            })?;
        }

        Ok(())
    }

    async fn redis_simple_string_response(
        stream: &mut tokio::net::TcpStream,
        value: &str,
    ) -> std::io::Result<()> {
        stream.write_all(format!("+{value}\r\n").as_bytes()).await
    }

    async fn redis_error_response(
        stream: &mut tokio::net::TcpStream,
        message: &str,
    ) -> std::io::Result<()> {
        stream
            .write_all(format!("-ERR {message}\r\n").as_bytes())
            .await
    }

    async fn redis_integer_response(
        stream: &mut tokio::net::TcpStream,
        value: i64,
    ) -> std::io::Result<()> {
        stream.write_all(format!(":{value}\r\n").as_bytes()).await
    }

    async fn redis_bulk_response(
        stream: &mut tokio::net::TcpStream,
        value: Option<&[u8]>,
    ) -> std::io::Result<()> {
        match value {
            Some(bytes) => {
                stream
                    .write_all(format!("${}\r\n", bytes.len()).as_bytes())
                    .await?;
                stream.write_all(bytes).await?;
                stream.write_all(b"\r\n").await
            }
            None => stream.write_all(b"$-1\r\n").await,
        }
    }

    async fn redis_array_response(
        stream: &mut tokio::net::TcpStream,
        values: &[Vec<u8>],
    ) -> std::io::Result<()> {
        stream
            .write_all(format!("*{}\r\n", values.len()).as_bytes())
            .await?;
        for value in values {
            redis_bulk_response(stream, Some(value)).await?;
        }
        Ok(())
    }

    fn redis_value_to_string(value: RedisValue) -> Result<String> {
        match value {
            RedisValue::BulkString(bytes) => String::from_utf8(bytes)
                .map_err(|error| WorldForgeError::SerializationError(error.to_string())),
            RedisValue::SimpleString(text) => Ok(text),
            RedisValue::Error(message) => Err(WorldForgeError::InternalError(message)),
            other => Err(WorldForgeError::InvalidState(format!(
                "expected Redis string, got {other:?}"
            ))),
        }
    }
}
