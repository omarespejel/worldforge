//! WR-Arena benchmark dataset loaders.
//!
//! Provides structured loading of WR-Arena evaluation datasets in their
//! native JSON format. Supports three dataset types:
//!
//! - **Action Simulation Fidelity**: sequential prompt scenarios for agent/environment evaluation
//! - **Smoothness Evaluation**: multi-round scenes with camera paths
//! - **Generation Consistency**: multi-round scenes for 3D/visual consistency checks

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use worldforge_core::error::{Result, WorldForgeError};

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Camera motion type for multi-round evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraMotion {
    PanLeft,
    PanRight,
    PanUp,
    PanDown,
    ZoomIn,
    ZoomOut,
    TiltUp,
    TiltDown,
    Orbit,
    Static,
}

/// Whether an evaluation instance tests agent or environment simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationType {
    /// Agent actions: robot picks up cup, person walks to door.
    Agent,
    /// Environment changes: shadow moves, water flows, leaves rustle.
    Environment,
}

/// Structured metadata about a scenario within a multi-round dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioMetadata {
    /// Scenario identifier.
    pub sid: String,
    /// Human-readable label.
    pub label: String,
    /// Full text definition of the scenario.
    #[serde(default, alias = "s.definition")]
    pub definition: String,
}

// ---------------------------------------------------------------------------
// Action Simulation Fidelity
// ---------------------------------------------------------------------------

/// A single instance from the WR-Arena action simulation fidelity dataset.
///
/// Each instance consists of an initial image and a list of sequential
/// text prompts describing a multi-step action scenario.
///
/// # JSON format
///
/// ```json
/// {
///   "id": "agent_000_1",
///   "image_path": "initial_state/000.png",
///   "prompt_list": [
///     "A man walks toward the toll booth",
///     "He reaches into his pocket for change",
///     "He hands the coins to the attendant"
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSimulationInstance {
    /// Unique instance identifier (e.g. `"agent_000_1"` or `"env_003_2"`).
    pub id: String,
    /// Path to the initial frame image (relative to dataset root).
    pub image_path: PathBuf,
    /// Sequential action prompts — typically 3 per instance.
    pub prompt_list: Vec<String>,
}

impl ActionSimulationInstance {
    /// Infer the simulation type from the instance id prefix.
    pub fn simulation_type(&self) -> SimulationType {
        if self.id.starts_with("env") {
            SimulationType::Environment
        } else {
            SimulationType::Agent
        }
    }

    /// Number of rounds (prompts) in this instance.
    pub fn num_rounds(&self) -> usize {
        self.prompt_list.len()
    }
}

/// A loaded action simulation fidelity dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSimulationDataset {
    /// All instances in the dataset.
    pub instances: Vec<ActionSimulationInstance>,
}

impl ActionSimulationDataset {
    /// Load a dataset from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_json_path(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            WorldForgeError::InvalidState(format!(
                "cannot read action simulation dataset at {}: {e}",
                path.display()
            ))
        })?;
        Self::from_json_str(&content)
    }

    /// Parse a dataset from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let instances: Vec<ActionSimulationInstance> = serde_json::from_str(json).map_err(|e| {
            WorldForgeError::SerializationError(format!(
                "invalid action simulation dataset JSON: {e}"
            ))
        })?;
        Ok(Self { instances })
    }

    /// Return only agent-type instances.
    pub fn agent_instances(&self) -> Vec<&ActionSimulationInstance> {
        self.instances
            .iter()
            .filter(|i| i.simulation_type() == SimulationType::Agent)
            .collect()
    }

    /// Return only environment-type instances.
    pub fn environment_instances(&self) -> Vec<&ActionSimulationInstance> {
        self.instances
            .iter()
            .filter(|i| i.simulation_type() == SimulationType::Environment)
            .collect()
    }

    /// Total number of instances.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Multi-Round Evaluation (Smoothness + Consistency)
// ---------------------------------------------------------------------------

/// A single instance from the WR-Arena multi-round evaluation dataset.
///
/// Used for both smoothness and generation consistency evaluation.
/// Contains rich metadata about the scene and a 10-round prompt sequence.
///
/// # JSON format
///
/// ```json
/// {
///   "id": "scene_001",
///   "visual_movement": "dynamic",
///   "visual_style": "photorealistic",
///   "scene_type": "outdoor",
///   "category": "diverse",
///   "scenario": { "sid": "S001", "label": "park_scene", "definition": "..." },
///   "camera_path": ["pan_right", "pan_right", "zoom_in"],
///   "content_list": ["A park with oak trees", ...],
///   "prompt_list": ["Camera pans right over the park", ...],
///   "image_path": "static/photorealistic/outdoor/diverse/001_1.png"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRoundInstance {
    /// Unique instance identifier.
    pub id: String,
    /// Path to the initial frame image (relative to dataset root).
    pub image_path: PathBuf,
    /// Sequential prompts for each round (typically 10 rounds).
    pub prompt_list: Vec<String>,
    /// Camera motion type for each round.
    #[serde(default)]
    pub camera_path: Vec<CameraMotion>,
    /// Type of visual movement in the scene.
    #[serde(default)]
    pub visual_movement: String,
    /// Visual style of the scene (e.g. "photorealistic").
    #[serde(default)]
    pub visual_style: String,
    /// Scene type (e.g. "outdoor", "indoor").
    #[serde(default)]
    pub scene_type: String,
    /// Scene category (e.g. "diverse", "urban").
    #[serde(default)]
    pub category: String,
    /// Structured scenario metadata.
    #[serde(default)]
    pub scenario: Option<ScenarioMetadata>,
    /// Content descriptions for each round.
    #[serde(default)]
    pub content_list: Vec<String>,
}

impl MultiRoundInstance {
    /// Number of rounds (prompts) in this instance.
    pub fn num_rounds(&self) -> usize {
        self.prompt_list.len()
    }
}

/// A loaded multi-round evaluation dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRoundDataset {
    /// The type of evaluation this dataset is intended for.
    pub eval_type: MultiRoundEvalType,
    /// All instances in the dataset.
    pub instances: Vec<MultiRoundInstance>,
}

/// The type of multi-round evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiRoundEvalType {
    /// Transition smoothness evaluation.
    Smoothness,
    /// Generation consistency evaluation.
    Consistency,
}

impl MultiRoundDataset {
    /// Load a dataset from a JSON file.
    pub fn from_json_path(path: &Path, eval_type: MultiRoundEvalType) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            WorldForgeError::InvalidState(format!(
                "cannot read multi-round dataset at {}: {e}",
                path.display()
            ))
        })?;
        Self::from_json_str(&content, eval_type)
    }

    /// Parse a dataset from a JSON string.
    pub fn from_json_str(json: &str, eval_type: MultiRoundEvalType) -> Result<Self> {
        let instances: Vec<MultiRoundInstance> = serde_json::from_str(json).map_err(|e| {
            WorldForgeError::SerializationError(format!("invalid multi-round dataset JSON: {e}"))
        })?;
        Ok(Self {
            eval_type,
            instances,
        })
    }

    /// Total number of instances.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Filter instances by scene type.
    pub fn by_scene_type(&self, scene_type: &str) -> Vec<&MultiRoundInstance> {
        self.instances
            .iter()
            .filter(|i| i.scene_type == scene_type)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_simulation_instance_type_detection() {
        let agent = ActionSimulationInstance {
            id: "agent_000_1".to_string(),
            image_path: PathBuf::from("initial_state/000.png"),
            prompt_list: vec!["walk forward".to_string()],
        };
        assert_eq!(agent.simulation_type(), SimulationType::Agent);
        assert_eq!(agent.num_rounds(), 1);

        let env = ActionSimulationInstance {
            id: "env_003_2".to_string(),
            image_path: PathBuf::from("initial_state/003.png"),
            prompt_list: vec!["shadow extends".to_string(), "light fades".to_string()],
        };
        assert_eq!(env.simulation_type(), SimulationType::Environment);
        assert_eq!(env.num_rounds(), 2);
    }

    #[test]
    fn test_action_simulation_dataset_from_json() {
        let json = r#"[
            {
                "id": "agent_000_1",
                "image_path": "initial_state/000.png",
                "prompt_list": ["walk forward", "pick up cup", "place on table"]
            },
            {
                "id": "env_000_1",
                "image_path": "initial_state/000.png",
                "prompt_list": ["shadow extends left", "cloud passes over", "wind blows leaves"]
            }
        ]"#;

        let dataset = ActionSimulationDataset::from_json_str(json).unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.agent_instances().len(), 1);
        assert_eq!(dataset.environment_instances().len(), 1);
        assert_eq!(dataset.instances[0].num_rounds(), 3);
    }

    #[test]
    fn test_action_simulation_dataset_empty() {
        let dataset = ActionSimulationDataset::from_json_str("[]").unwrap();
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
    }

    #[test]
    fn test_multi_round_dataset_from_json() {
        let json = r#"[
            {
                "id": "scene_001",
                "image_path": "static/outdoor/001.png",
                "prompt_list": ["Camera pans right", "Camera continues right"],
                "camera_path": ["pan_right", "pan_right"],
                "visual_movement": "dynamic",
                "visual_style": "photorealistic",
                "scene_type": "outdoor",
                "category": "diverse"
            }
        ]"#;

        let dataset =
            MultiRoundDataset::from_json_str(json, MultiRoundEvalType::Smoothness).unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.eval_type, MultiRoundEvalType::Smoothness);
        assert_eq!(dataset.instances[0].num_rounds(), 2);
        assert_eq!(
            dataset.instances[0].camera_path,
            vec![CameraMotion::PanRight, CameraMotion::PanRight]
        );
    }

    #[test]
    fn test_multi_round_dataset_scene_type_filter() {
        let json = r#"[
            { "id": "s1", "image_path": "a.png", "prompt_list": ["p1"], "scene_type": "outdoor" },
            { "id": "s2", "image_path": "b.png", "prompt_list": ["p2"], "scene_type": "indoor" },
            { "id": "s3", "image_path": "c.png", "prompt_list": ["p3"], "scene_type": "outdoor" }
        ]"#;

        let dataset =
            MultiRoundDataset::from_json_str(json, MultiRoundEvalType::Consistency).unwrap();
        assert_eq!(dataset.by_scene_type("outdoor").len(), 2);
        assert_eq!(dataset.by_scene_type("indoor").len(), 1);
        assert_eq!(dataset.by_scene_type("underwater").len(), 0);
    }

    #[test]
    fn test_camera_motion_serde_roundtrip() {
        let motions = vec![
            CameraMotion::PanLeft,
            CameraMotion::ZoomIn,
            CameraMotion::Static,
        ];
        let json = serde_json::to_string(&motions).unwrap();
        let parsed: Vec<CameraMotion> = serde_json::from_str(&json).unwrap();
        assert_eq!(motions, parsed);
    }

    #[test]
    fn test_simulation_type_serde_roundtrip() {
        let agent_json = serde_json::to_string(&SimulationType::Agent).unwrap();
        assert_eq!(agent_json, r#""agent""#);
        let parsed: SimulationType = serde_json::from_str(&agent_json).unwrap();
        assert_eq!(parsed, SimulationType::Agent);

        let env_json = serde_json::to_string(&SimulationType::Environment).unwrap();
        assert_eq!(env_json, r#""environment""#);
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let result = ActionSimulationDataset::from_json_str("not json");
        assert!(result.is_err());
    }
}
