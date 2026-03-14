//! State persistence for WorldForge worlds.
//!
//! Provides the `StateStore` trait and file-based implementation
//! for saving and loading world state.

use std::collections::VecDeque;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::error::{Result, WorldForgeError};
use crate::scene::SceneGraph;
use crate::types::{SimTime, WorldId};

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
    /// Fingerprint of the serialized state (non-cryptographic).
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

/// File-based state store using JSON serialization.
#[derive(Debug, Clone)]
pub struct FileStateStore {
    /// Directory for state files.
    pub path: PathBuf,
}

impl FileStateStore {
    /// Create a new file-based state store at the given directory.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    fn state_path(&self, id: &WorldId) -> PathBuf {
        self.path.join(format!("{}.json", id))
    }
}

#[async_trait::async_trait]
impl StateStore for FileStateStore {
    async fn save(&self, state: &WorldState) -> Result<()> {
        tokio::fs::create_dir_all(&self.path)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("failed to create dir: {e}")))?;
        let json = serde_json::to_string_pretty(state)
            .map_err(|e| WorldForgeError::SerializationError(e.to_string()))?;
        tokio::fs::write(self.state_path(&state.id), json)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("failed to write state: {e}")))?;
        Ok(())
    }

    async fn load(&self, id: &WorldId) -> Result<WorldState> {
        let path = self.state_path(id);
        let data = tokio::fs::read_to_string(&path)
            .await
            .map_err(|_| WorldForgeError::WorldNotFound(*id))?;
        serde_json::from_str(&data).map_err(|e| WorldForgeError::SerializationError(e.to_string()))
    }

    async fn list(&self) -> Result<Vec<WorldId>> {
        let mut ids = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.path)
            .await
            .map_err(|e| WorldForgeError::InternalError(format!("failed to read dir: {e}")))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| WorldForgeError::InternalError(e.to_string()))?
        {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(id_str) = name.strip_suffix(".json") {
                    if let Ok(id) = id_str.parse::<WorldId>() {
                        ids.push(id);
                    }
                }
            }
        }
        Ok(ids)
    }

    async fn delete(&self, id: &WorldId) -> Result<()> {
        let path = self.state_path(id);
        tokio::fs::remove_file(&path)
            .await
            .map_err(|_| WorldForgeError::WorldNotFound(*id))?;
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

    #[tokio::test]
    async fn test_file_state_store() {
        let dir = std::env::temp_dir().join(format!("worldforge-test-{}", uuid::Uuid::new_v4()));
        let store = FileStateStore::new(&dir);

        let state = WorldState::new("test-world", "mock");
        let id = state.id;

        store.save(&state).await.unwrap();
        let loaded = store.load(&id).await.unwrap();
        assert_eq!(loaded.id, id);

        let ids = store.list().await.unwrap();
        assert!(ids.contains(&id));

        store.delete(&id).await.unwrap();
        assert!(store.load(&id).await.is_err());

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }
}
