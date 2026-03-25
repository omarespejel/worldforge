//! WorldForge Evaluation Framework
//!
//! Provides standardized evaluation of world foundation models across
//! dimensions like physics plausibility, spatial consistency, and
//! temporal coherence.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use worldforge_core::action::{evaluate_condition, Action, Condition};
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::prediction::{PhysicsScores, PredictionConfig};
use worldforge_core::provider::WorldModelProvider;
use worldforge_core::state::WorldState;
use worldforge_core::types::{Position, Tensor, TensorData, VideoClip};

const BUILTIN_SUITE_NAMES: [&str; 4] = ["physics", "manipulation", "spatial", "comprehensive"];

/// Dimension along which a provider is evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvalDimension {
    /// Do objects persist when occluded?
    ObjectPermanence,
    /// Do unsupported objects fall?
    GravityCompliance,
    /// Are collisions physically accurate?
    CollisionAccuracy,
    /// Is the scene spatially consistent across viewpoints?
    SpatialConsistency,
    /// Is the scene temporally consistent across time?
    TemporalConsistency,
    /// Does the action produce the expected physical effect?
    ActionPredictionAccuracy,
    /// Does the model understand material properties?
    MaterialUnderstanding,
    /// Can the model reason about depth, scale, and distance?
    SpatialReasoning,
    /// Custom evaluation dimension.
    Custom { name: String },
}

impl EvalDimension {
    fn key(&self) -> String {
        match self {
            Self::ObjectPermanence => "object_permanence".to_string(),
            Self::GravityCompliance => "gravity_compliance".to_string(),
            Self::CollisionAccuracy => "collision_accuracy".to_string(),
            Self::SpatialConsistency => "spatial_consistency".to_string(),
            Self::TemporalConsistency => "temporal_consistency".to_string(),
            Self::ActionPredictionAccuracy => "action_prediction_accuracy".to_string(),
            Self::MaterialUnderstanding => "material_understanding".to_string(),
            Self::SpatialReasoning => "spatial_reasoning".to_string(),
            Self::Custom { name } => name.clone(),
        }
    }
}

/// A single evaluation scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalScenario {
    /// Human-readable name of the scenario.
    pub name: String,
    /// Description of what is being tested.
    pub description: String,
    /// Initial world state.
    pub initial_state: WorldState,
    /// Actions to perform.
    pub actions: Vec<Action>,
    /// Expected outcomes to check.
    pub expected_outcomes: Vec<ExpectedOutcome>,
    /// Ground truth video for comparison (if available).
    pub ground_truth: Option<VideoClip>,
}

/// An expected outcome to verify after prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpectedOutcome {
    /// An object should exist in the scene.
    ObjectExists { name: String },
    /// An object should not exist in the scene.
    ObjectNotExists { name: String },
    /// An object should end up near the target position.
    ObjectPosition {
        /// Human-readable object name.
        name: String,
        /// Expected world-space position.
        position: Position,
        /// Maximum allowed Euclidean distance from the target.
        tolerance: f32,
    },
    /// An object should carry the expected semantic label.
    ObjectSemanticLabel {
        /// Human-readable object name.
        name: String,
        /// Expected semantic label value.
        label: String,
    },
    /// The minimum physics score threshold.
    MinPhysicsScore {
        dimension: EvalDimension,
        threshold: f32,
    },
    /// The prediction confidence should be above a threshold.
    MinConfidence { threshold: f32 },
    /// The predicted clip should be sufficiently similar to the supplied ground truth.
    MinVideoSimilarity { threshold: f32 },
    /// The final state should satisfy a core `Condition`.
    FinalStateCondition { condition: Condition },
}

/// A suite of evaluation scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSuite {
    /// Name of this evaluation suite.
    pub name: String,
    /// Scenarios in this suite.
    pub scenarios: Vec<EvalScenario>,
    /// Dimensions to evaluate.
    pub dimensions: Vec<EvalDimension>,
}

/// Result of evaluating one scenario with one provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Provider that was evaluated.
    pub provider: String,
    /// Scenario that was evaluated.
    pub scenario: String,
    /// Scores per dimension.
    pub scores: HashMap<String, f32>,
    /// Latency of the evaluation in milliseconds.
    pub latency_ms: u64,
    /// Final predicted clip retained for this scenario, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoClip>,
    /// Derived similarity metrics between the predicted clip and ground truth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_metrics: Option<VideoMetrics>,
    /// Whether each expected outcome was met.
    pub outcomes: Vec<OutcomeResult>,
}

/// Ground-truth comparison metrics for a predicted clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetrics {
    /// Aggregate similarity score across metadata and aligned frame content.
    pub overall_similarity: f32,
    /// Similarity of the declared resolution.
    pub resolution_similarity: f32,
    /// Similarity of the declared FPS.
    pub fps_similarity: f32,
    /// Similarity of the declared duration.
    pub duration_similarity: f32,
    /// Similarity of frame counts.
    pub frame_count_similarity: f32,
    /// Average similarity of aligned RGB frame tensors.
    pub frame_similarity: Option<f32>,
    /// Average similarity of aligned depth tensors when present in both clips.
    pub depth_similarity: Option<f32>,
    /// Average similarity of aligned segmentation tensors when present in both clips.
    pub segmentation_similarity: Option<f32>,
}

/// Whether an expected outcome was met.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeResult {
    /// Description of the outcome.
    pub description: String,
    /// Whether it was met.
    pub passed: bool,
    /// Explanation.
    pub details: Option<String>,
}

/// Aggregated results across all scenarios and providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Suite name.
    pub suite: String,
    /// Per-provider, per-scenario results.
    pub results: Vec<EvalResult>,
    /// Leaderboard ranking.
    pub leaderboard: Vec<LeaderboardEntry>,
    /// Per-provider rollups across the suite.
    pub provider_summaries: Vec<ProviderSummary>,
    /// Per-dimension rollups across all providers.
    pub dimension_summaries: Vec<DimensionSummary>,
    /// Per-scenario comparisons across providers.
    pub scenario_summaries: Vec<ScenarioSummary>,
    /// Number of outcomes that passed across the full report.
    pub outcomes_passed: usize,
    /// Total number of evaluated outcomes across the full report.
    pub total_outcomes: usize,
}

/// One row in the evaluation leaderboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    /// Provider name.
    pub provider: String,
    /// Average score across all dimensions.
    pub average_score: f32,
    /// Average latency in milliseconds.
    pub average_latency_ms: u64,
    /// Number of scenarios passed.
    pub scenarios_passed: usize,
    /// Total number of scenarios.
    pub total_scenarios: usize,
}

/// Aggregated metrics for a single provider across a suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSummary {
    /// Provider name.
    pub provider: String,
    /// Average overall score across all scenarios with scores.
    pub average_score: f32,
    /// Average end-to-end latency across all scenarios.
    pub average_latency_ms: u64,
    /// Number of scenarios where every outcome passed.
    pub scenarios_passed: usize,
    /// Total number of scenarios in the suite.
    pub total_scenarios: usize,
    /// Fraction of scenarios that fully passed.
    pub scenario_pass_rate: f32,
    /// Number of passed outcomes across all scenarios.
    pub outcomes_passed: usize,
    /// Total number of outcomes across all scenarios.
    pub total_outcomes: usize,
    /// Fraction of individual outcomes that passed.
    pub outcome_pass_rate: f32,
    /// Average score per dimension across the suite.
    pub dimension_scores: HashMap<String, f32>,
    /// Scenario-level overall scores keyed by scenario name.
    pub scenario_scores: HashMap<String, f32>,
}

/// Aggregated metrics for a dimension across all providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionSummary {
    /// Dimension identifier.
    pub dimension: String,
    /// Average score for each provider on this dimension.
    pub provider_scores: HashMap<String, f32>,
    /// Provider with the highest average score for this dimension.
    pub best_provider: Option<String>,
    /// Highest average score observed for this dimension.
    pub best_score: Option<f32>,
}

/// Aggregated comparison for a scenario across all providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSummary {
    /// Scenario identifier.
    pub scenario: String,
    /// Human-readable scenario description.
    pub description: String,
    /// Average overall score by provider.
    pub provider_scores: HashMap<String, f32>,
    /// Providers that passed every outcome for this scenario.
    pub passed_by: Vec<String>,
    /// Providers that failed at least one outcome for this scenario.
    pub failed_by: Vec<String>,
    /// Best-performing provider for this scenario, if any.
    pub best_provider: Option<String>,
    /// Best overall score recorded for this scenario, if any.
    pub best_score: Option<f32>,
    /// Number of passed outcomes across every provider evaluated on this scenario.
    pub outcomes_passed: usize,
    /// Total number of evaluated outcomes across every provider for this scenario.
    pub total_outcomes: usize,
}

impl EvalSuite {
    /// List the built-in evaluation suites.
    pub fn builtin_names() -> &'static [&'static str] {
        &BUILTIN_SUITE_NAMES
    }

    /// Load one of the built-in evaluation suites by name.
    pub fn from_builtin(name: &str) -> Result<Self> {
        let suite = match name {
            "physics" => Self::physics_standard(),
            "manipulation" => Self::manipulation_standard(),
            "spatial" => Self::spatial_reasoning(),
            "comprehensive" => Self::comprehensive(),
            other => {
                return Err(WorldForgeError::InvalidState(format!(
                    "unknown eval suite: {other}. Available: {}",
                    Self::builtin_names().join(", ")
                )))
            }
        };
        suite.validate()?;
        Ok(suite)
    }

    /// Deserialize an evaluation suite from JSON.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let suite: Self = serde_json::from_str(json)
            .map_err(|error| WorldForgeError::SerializationError(error.to_string()))?;
        suite.validate()?;
        Ok(suite)
    }

    /// Read and deserialize an evaluation suite from a JSON file.
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|error| {
            WorldForgeError::SerializationError(format!(
                "failed to read {}: {error}",
                path.display()
            ))
        })?;
        Self::from_json_str(&contents)
    }

    /// Serialize the suite to pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| WorldForgeError::SerializationError(error.to_string()))
    }

    /// Validate that the suite is structurally usable.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(WorldForgeError::InvalidState(
                "evaluation suite name cannot be empty".to_string(),
            ));
        }
        if self.scenarios.is_empty() {
            return Err(WorldForgeError::InvalidState(format!(
                "evaluation suite '{}' must contain at least one scenario",
                self.name
            )));
        }
        if self.dimensions.is_empty() {
            return Err(WorldForgeError::InvalidState(format!(
                "evaluation suite '{}' must declare at least one dimension",
                self.name
            )));
        }

        let mut seen_names = HashSet::new();
        for scenario in &self.scenarios {
            if scenario.name.trim().is_empty() {
                return Err(WorldForgeError::InvalidState(format!(
                    "evaluation suite '{}' contains a scenario with an empty name",
                    self.name
                )));
            }
            if scenario.description.trim().is_empty() {
                return Err(WorldForgeError::InvalidState(format!(
                    "scenario '{}' must include a description",
                    scenario.name
                )));
            }
            if !seen_names.insert(scenario.name.as_str()) {
                return Err(WorldForgeError::InvalidState(format!(
                    "duplicate evaluation scenario name: {}",
                    scenario.name
                )));
            }
            if scenario.ground_truth.is_none()
                && scenario
                    .expected_outcomes
                    .iter()
                    .any(|expected| matches!(expected, ExpectedOutcome::MinVideoSimilarity { .. }))
            {
                return Err(WorldForgeError::InvalidState(format!(
                    "scenario '{}' uses video similarity checks but does not provide ground truth",
                    scenario.name
                )));
            }
        }

        Ok(())
    }

    /// Create a standard physics evaluation suite.
    pub fn physics_standard() -> Self {
        Self {
            name: "Physics Standard".to_string(),
            scenarios: vec![
                EvalScenario {
                    name: "object_drop".to_string(),
                    description: "Drop an object — it should fall due to gravity".to_string(),
                    initial_state: WorldState::new("gravity_test", "eval"),
                    actions: vec![Action::Release {
                        object: uuid::Uuid::new_v4(),
                    }],
                    expected_outcomes: vec![ExpectedOutcome::MinPhysicsScore {
                        dimension: EvalDimension::GravityCompliance,
                        threshold: 0.7,
                    }],
                    ground_truth: None,
                },
                EvalScenario {
                    name: "object_collision".to_string(),
                    description: "Push object into another — should collide".to_string(),
                    initial_state: WorldState::new("collision_test", "eval"),
                    actions: vec![Action::Push {
                        object: uuid::Uuid::new_v4(),
                        direction: worldforge_core::types::Vec3 {
                            x: 1.0,
                            y: 0.0,
                            z: 0.0,
                        },
                        force: 5.0,
                    }],
                    expected_outcomes: vec![ExpectedOutcome::MinPhysicsScore {
                        dimension: EvalDimension::CollisionAccuracy,
                        threshold: 0.7,
                    }],
                    ground_truth: None,
                },
            ],
            dimensions: vec![
                EvalDimension::ObjectPermanence,
                EvalDimension::GravityCompliance,
                EvalDimension::CollisionAccuracy,
                EvalDimension::SpatialConsistency,
                EvalDimension::TemporalConsistency,
            ],
        }
    }

    /// Create a manipulation evaluation suite.
    ///
    /// Tests object grasping, placement, and compound manipulation tasks.
    pub fn manipulation_standard() -> Self {
        use worldforge_core::scene::SceneObject;
        use worldforge_core::types::{BBox, Pose, Position};

        let mut state = WorldState::new("manipulation_test", "eval");
        let mug = SceneObject::new(
            "mug",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                ..Default::default()
            },
            BBox {
                min: Position {
                    x: -0.05,
                    y: 0.9,
                    z: -0.05,
                },
                max: Position {
                    x: 0.05,
                    y: 1.1,
                    z: 0.05,
                },
            },
        );
        let mug_id = mug.id;
        state.scene.add_object(mug);

        let mut table_state = WorldState::new("table_test", "eval");
        let mut table = SceneObject::new(
            "table",
            Pose::default(),
            BBox {
                min: Position {
                    x: -0.5,
                    y: 0.0,
                    z: -0.5,
                },
                max: Position {
                    x: 0.5,
                    y: 0.8,
                    z: 0.5,
                },
            },
        );
        table.physics.is_static = true;
        let block = SceneObject::new(
            "block",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.9,
                    z: 0.0,
                },
                ..Default::default()
            },
            BBox {
                min: Position {
                    x: -0.05,
                    y: 0.8,
                    z: -0.05,
                },
                max: Position {
                    x: 0.05,
                    y: 1.0,
                    z: 0.05,
                },
            },
        );
        let block_id = block.id;
        table_state.scene.add_object(table);
        table_state.scene.add_object(block);

        Self {
            name: "Manipulation Standard".to_string(),
            scenarios: vec![
                EvalScenario {
                    name: "grasp_object".to_string(),
                    description: "Grasp a mug — object should remain in scene".to_string(),
                    initial_state: state.clone(),
                    actions: vec![Action::Grasp {
                        object: mug_id,
                        grip_force: 5.0,
                    }],
                    expected_outcomes: vec![
                        ExpectedOutcome::ObjectExists {
                            name: "mug".to_string(),
                        },
                        ExpectedOutcome::MinConfidence { threshold: 0.5 },
                    ],
                    ground_truth: None,
                },
                EvalScenario {
                    name: "place_object".to_string(),
                    description: "Place an object at a target — should reach destination"
                        .to_string(),
                    initial_state: state,
                    actions: vec![Action::Place {
                        object: mug_id,
                        target: Position {
                            x: 1.0,
                            y: 0.8,
                            z: 0.0,
                        },
                    }],
                    expected_outcomes: vec![
                        ExpectedOutcome::FinalStateCondition {
                            condition: Condition::And(vec![
                                Condition::ObjectExists { object: mug_id },
                                Condition::ObjectAt {
                                    object: mug_id,
                                    position: Position {
                                        x: 1.0,
                                        y: 0.8,
                                        z: 0.0,
                                    },
                                    tolerance: 0.001,
                                },
                            ]),
                        },
                        ExpectedOutcome::MinPhysicsScore {
                            dimension: EvalDimension::SpatialConsistency,
                            threshold: 0.5,
                        },
                    ],
                    ground_truth: None,
                },
                EvalScenario {
                    name: "push_on_surface".to_string(),
                    description: "Push a block along a table surface".to_string(),
                    initial_state: table_state,
                    actions: vec![Action::Push {
                        object: block_id,
                        direction: worldforge_core::types::Vec3 {
                            x: 0.3,
                            y: 0.0,
                            z: 0.0,
                        },
                        force: 2.0,
                    }],
                    expected_outcomes: vec![
                        ExpectedOutcome::ObjectExists {
                            name: "block".to_string(),
                        },
                        ExpectedOutcome::ObjectExists {
                            name: "table".to_string(),
                        },
                        ExpectedOutcome::MinPhysicsScore {
                            dimension: EvalDimension::GravityCompliance,
                            threshold: 0.5,
                        },
                    ],
                    ground_truth: None,
                },
            ],
            dimensions: vec![
                EvalDimension::ObjectPermanence,
                EvalDimension::GravityCompliance,
                EvalDimension::SpatialConsistency,
                EvalDimension::ActionPredictionAccuracy,
            ],
        }
    }

    /// Create a spatial reasoning evaluation suite.
    ///
    /// Tests understanding of spatial relationships, occlusion, and depth.
    pub fn spatial_reasoning() -> Self {
        use worldforge_core::scene::SceneObject;
        use worldforge_core::types::{BBox, Pose, Position};

        let mut state = WorldState::new("spatial_test", "eval");
        let box_a = SceneObject::new(
            "box_a",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.5,
                    z: 0.0,
                },
                ..Default::default()
            },
            BBox {
                min: Position {
                    x: -0.2,
                    y: 0.0,
                    z: -0.2,
                },
                max: Position {
                    x: 0.2,
                    y: 1.0,
                    z: 0.2,
                },
            },
        );
        let box_b = SceneObject::new(
            "box_b",
            Pose {
                position: Position {
                    x: 2.0,
                    y: 0.5,
                    z: 0.0,
                },
                ..Default::default()
            },
            BBox {
                min: Position {
                    x: 1.8,
                    y: 0.0,
                    z: -0.2,
                },
                max: Position {
                    x: 2.2,
                    y: 1.0,
                    z: 0.2,
                },
            },
        );
        let box_a_id = box_a.id;
        let box_b_id = box_b.id;
        state.scene.add_object(box_a);
        state.scene.add_object(box_b);

        let mut occl_state = WorldState::new("occlusion_test", "eval");
        let front_obj = SceneObject::new(
            "front_wall",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.5,
                    z: -1.0,
                },
                ..Default::default()
            },
            BBox {
                min: Position {
                    x: -1.0,
                    y: 0.0,
                    z: -1.1,
                },
                max: Position {
                    x: 1.0,
                    y: 1.0,
                    z: -0.9,
                },
            },
        );
        let hidden_obj = SceneObject::new(
            "hidden_ball",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.5,
                    z: 0.0,
                },
                ..Default::default()
            },
            BBox {
                min: Position {
                    x: -0.1,
                    y: 0.4,
                    z: -0.1,
                },
                max: Position {
                    x: 0.1,
                    y: 0.6,
                    z: 0.1,
                },
            },
        );
        occl_state.scene.add_object(front_obj);
        occl_state.scene.add_object(hidden_obj);

        Self {
            name: "Spatial Reasoning".to_string(),
            scenarios: vec![
                EvalScenario {
                    name: "spatial_separation".to_string(),
                    description: "Two separated boxes — moving one should not affect the other"
                        .to_string(),
                    initial_state: state,
                    actions: vec![Action::Push {
                        object: box_a_id,
                        direction: worldforge_core::types::Vec3 {
                            x: -1.0,
                            y: 0.0,
                            z: 0.0,
                        },
                        force: 1.0,
                    }],
                    expected_outcomes: vec![
                        ExpectedOutcome::FinalStateCondition {
                            condition: Condition::And(vec![
                                Condition::ObjectExists { object: box_a_id },
                                Condition::ObjectExists { object: box_b_id },
                                Condition::Not(Box::new(Condition::ObjectsTouching {
                                    a: box_a_id,
                                    b: box_b_id,
                                })),
                            ]),
                        },
                        ExpectedOutcome::MinPhysicsScore {
                            dimension: EvalDimension::SpatialConsistency,
                            threshold: 0.5,
                        },
                    ],
                    ground_truth: None,
                },
                EvalScenario {
                    name: "object_permanence_occlusion".to_string(),
                    description: "Object behind wall should persist even when occluded".to_string(),
                    initial_state: occl_state,
                    actions: vec![Action::CameraLookAt {
                        target: Position {
                            x: 0.0,
                            y: 0.5,
                            z: -5.0,
                        },
                    }],
                    expected_outcomes: vec![
                        ExpectedOutcome::ObjectExists {
                            name: "hidden_ball".to_string(),
                        },
                        ExpectedOutcome::ObjectExists {
                            name: "front_wall".to_string(),
                        },
                        ExpectedOutcome::MinPhysicsScore {
                            dimension: EvalDimension::ObjectPermanence,
                            threshold: 0.6,
                        },
                    ],
                    ground_truth: None,
                },
            ],
            dimensions: vec![
                EvalDimension::ObjectPermanence,
                EvalDimension::SpatialConsistency,
                EvalDimension::SpatialReasoning,
            ],
        }
    }

    /// Create a comprehensive evaluation suite that combines all standard suites.
    pub fn comprehensive() -> Self {
        let physics = Self::physics_standard();
        let manipulation = Self::manipulation_standard();
        let spatial = Self::spatial_reasoning();

        let mut all_scenarios = physics.scenarios;
        all_scenarios.extend(manipulation.scenarios);
        all_scenarios.extend(spatial.scenarios);

        Self {
            name: "Comprehensive".to_string(),
            scenarios: all_scenarios,
            dimensions: vec![
                EvalDimension::ObjectPermanence,
                EvalDimension::GravityCompliance,
                EvalDimension::CollisionAccuracy,
                EvalDimension::SpatialConsistency,
                EvalDimension::TemporalConsistency,
                EvalDimension::ActionPredictionAccuracy,
                EvalDimension::SpatialReasoning,
            ],
        }
    }

    /// Run the evaluation suite against a set of providers.
    pub async fn run(&self, providers: &[&dyn WorldModelProvider]) -> Result<EvalReport> {
        self.validate()?;
        let mut all_results = Vec::new();

        for provider in providers {
            if !provider.capabilities().predict {
                for scenario in &self.scenarios {
                    all_results.push(EvalResult {
                        provider: provider.name().to_string(),
                        scenario: scenario.name.clone(),
                        scores: HashMap::new(),
                        latency_ms: 0,
                        video: None,
                        video_metrics: None,
                        outcomes: vec![OutcomeResult {
                            description: "provider supports prediction".to_string(),
                            passed: false,
                            details: Some(
                                "evaluation requires predict capability for every scenario"
                                    .to_string(),
                            ),
                        }],
                    });
                }
                continue;
            }

            for scenario in &self.scenarios {
                let start = std::time::Instant::now();
                let config = prediction_config_for_scenario(scenario);
                let mut score_accumulator = ScenarioAccumulator::default();
                let mut outcomes = Vec::new();

                // Run prediction for each action
                let mut current_state = scenario.initial_state.clone();
                let mut prediction_failed = false;
                for action in &scenario.actions {
                    match provider.predict(&current_state, action, &config).await {
                        Ok(prediction) => {
                            score_accumulator.record(&prediction);
                            current_state = prediction.output_state.clone();
                        }
                        Err(e) => {
                            outcomes.push(OutcomeResult {
                                description: "prediction".to_string(),
                                passed: false,
                                details: Some(e.to_string()),
                            });
                            prediction_failed = true;
                            break;
                        }
                    }
                }

                let average_scores = score_accumulator.average_scores();
                let average_confidence = score_accumulator.average_confidence();
                let mut scores = HashMap::new();
                if let Some(average_scores) = average_scores.as_ref() {
                    record_physics_scores(average_scores, &mut scores);
                }
                let predicted_video = if prediction_failed {
                    None
                } else {
                    score_accumulator.final_video.clone()
                };
                let video_metrics = if prediction_failed {
                    None
                } else {
                    predicted_video
                        .as_ref()
                        .zip(scenario.ground_truth.as_ref())
                        .map(|(predicted, ground_truth)| {
                            compare_video_clips(predicted, ground_truth)
                        })
                };
                if let Some(metrics) = video_metrics.as_ref() {
                    scores.insert("video_similarity".to_string(), metrics.overall_similarity);
                }
                ensure_overall_score(&mut scores);

                if !prediction_failed {
                    for expected in &scenario.expected_outcomes {
                        outcomes.push(check_outcome(
                            expected,
                            &current_state,
                            average_scores.as_ref(),
                            average_confidence,
                            scenario.ground_truth.as_ref(),
                            video_metrics.as_ref(),
                        ));
                    }
                }

                all_results.push(EvalResult {
                    provider: provider.name().to_string(),
                    scenario: scenario.name.clone(),
                    scores,
                    latency_ms: start.elapsed().as_millis() as u64,
                    video: predicted_video,
                    video_metrics,
                    outcomes,
                });
            }
        }

        let provider_summaries = build_provider_summaries(&all_results, self.scenarios.len());
        let leaderboard = build_leaderboard(&provider_summaries);
        let dimension_summaries = build_dimension_summaries(&all_results, &self.dimensions);
        let scenario_summaries = build_scenario_summaries(&all_results, &self.scenarios);
        let (outcomes_passed, total_outcomes) = all_results
            .iter()
            .flat_map(|result| result.outcomes.iter())
            .fold((0usize, 0usize), |(passed, total), outcome| {
                (passed + usize::from(outcome.passed), total + 1)
            });

        Ok(EvalReport {
            suite: self.name.clone(),
            results: all_results,
            leaderboard,
            provider_summaries,
            dimension_summaries,
            scenario_summaries,
            outcomes_passed,
            total_outcomes,
        })
    }
}

#[derive(Debug, Default)]
struct ScenarioAccumulator {
    physics: PhysicsScoreAccumulator,
    total_confidence: f32,
    count: usize,
    final_video: Option<VideoClip>,
}

impl ScenarioAccumulator {
    fn record(&mut self, prediction: &worldforge_core::prediction::Prediction) {
        self.physics.record(&prediction.physics_scores);
        self.total_confidence += prediction.confidence;
        self.count += 1;
        if let Some(video) = &prediction.video {
            self.final_video = Some(video.clone());
        }
    }

    fn average_scores(&self) -> Option<PhysicsScores> {
        self.physics.average()
    }

    fn average_confidence(&self) -> Option<f32> {
        (self.count > 0).then_some(self.total_confidence / self.count as f32)
    }
}

#[derive(Debug, Default)]
struct PhysicsScoreAccumulator {
    total: PhysicsScores,
    count: usize,
}

impl PhysicsScoreAccumulator {
    fn record(&mut self, scores: &PhysicsScores) {
        self.total.overall += scores.overall;
        self.total.object_permanence += scores.object_permanence;
        self.total.gravity_compliance += scores.gravity_compliance;
        self.total.collision_accuracy += scores.collision_accuracy;
        self.total.spatial_consistency += scores.spatial_consistency;
        self.total.temporal_consistency += scores.temporal_consistency;
        self.count += 1;
    }

    fn average(&self) -> Option<PhysicsScores> {
        if self.count == 0 {
            return None;
        }

        let count = self.count as f32;
        Some(PhysicsScores {
            overall: self.total.overall / count,
            object_permanence: self.total.object_permanence / count,
            gravity_compliance: self.total.gravity_compliance / count,
            collision_accuracy: self.total.collision_accuracy / count,
            spatial_consistency: self.total.spatial_consistency / count,
            temporal_consistency: self.total.temporal_consistency / count,
        })
    }
}

fn record_physics_scores(scores: &PhysicsScores, map: &mut HashMap<String, f32>) {
    map.insert("overall".to_string(), scores.overall);
    map.insert("object_permanence".to_string(), scores.object_permanence);
    map.insert("gravity_compliance".to_string(), scores.gravity_compliance);
    map.insert("collision_accuracy".to_string(), scores.collision_accuracy);
    map.insert(
        "spatial_consistency".to_string(),
        scores.spatial_consistency,
    );
    map.insert(
        "temporal_consistency".to_string(),
        scores.temporal_consistency,
    );
}

fn prediction_config_for_scenario(scenario: &EvalScenario) -> PredictionConfig {
    let needs_video = scenario_requires_video_artifacts(scenario);
    PredictionConfig {
        return_video: needs_video,
        return_depth: scenario
            .ground_truth
            .as_ref()
            .is_some_and(video_has_depth_maps),
        return_segmentation: scenario
            .ground_truth
            .as_ref()
            .is_some_and(video_has_segmentation_maps),
        ..PredictionConfig::default()
    }
}

fn scenario_requires_video_artifacts(scenario: &EvalScenario) -> bool {
    scenario.ground_truth.is_some()
        || scenario
            .expected_outcomes
            .iter()
            .any(|expected| matches!(expected, ExpectedOutcome::MinVideoSimilarity { .. }))
}

fn video_has_depth_maps(video: &VideoClip) -> bool {
    video.frames.iter().any(|frame| frame.depth.is_some())
}

fn video_has_segmentation_maps(video: &VideoClip) -> bool {
    video
        .frames
        .iter()
        .any(|frame| frame.segmentation.is_some())
}

fn check_outcome(
    expected: &ExpectedOutcome,
    state: &WorldState,
    average_scores: Option<&PhysicsScores>,
    average_confidence: Option<f32>,
    ground_truth: Option<&VideoClip>,
    video_metrics: Option<&VideoMetrics>,
) -> OutcomeResult {
    match expected {
        ExpectedOutcome::MinPhysicsScore {
            dimension,
            threshold,
        } => match average_scores {
            Some(scores) => {
                let score = match dimension {
                    EvalDimension::ObjectPermanence => scores.object_permanence,
                    EvalDimension::GravityCompliance => scores.gravity_compliance,
                    EvalDimension::CollisionAccuracy => scores.collision_accuracy,
                    EvalDimension::SpatialConsistency => scores.spatial_consistency,
                    EvalDimension::TemporalConsistency => scores.temporal_consistency,
                    _ => scores.overall,
                };
                OutcomeResult {
                    description: format!("{} >= {threshold}", dimension.key()),
                    passed: score >= *threshold,
                    details: Some(format!("score: {score:.3}")),
                }
            }
            None => OutcomeResult {
                description: format!("{} >= {threshold}", dimension.key()),
                passed: false,
                details: Some("requires at least one prediction step".to_string()),
            },
        },
        ExpectedOutcome::MinConfidence { threshold } => match average_confidence {
            Some(confidence) => OutcomeResult {
                description: format!("confidence >= {threshold}"),
                passed: confidence >= *threshold,
                details: Some(format!("confidence: {confidence:.3}")),
            },
            None => OutcomeResult {
                description: format!("confidence >= {threshold}"),
                passed: false,
                details: Some("requires at least one prediction step".to_string()),
            },
        },
        ExpectedOutcome::ObjectExists { name } => {
            let exists = state.scene.objects.values().any(|o| o.name == *name);
            OutcomeResult {
                description: format!("object '{name}' exists"),
                passed: exists,
                details: None,
            }
        }
        ExpectedOutcome::ObjectNotExists { name } => {
            let exists = state.scene.objects.values().any(|o| o.name == *name);
            OutcomeResult {
                description: format!("object '{name}' does not exist"),
                passed: !exists,
                details: None,
            }
        }
        ExpectedOutcome::ObjectPosition {
            name,
            position,
            tolerance,
        } => match state
            .scene
            .objects
            .values()
            .find(|object| object.name == *name)
        {
            Some(object) => {
                let distance = object.pose.position.distance(*position);
                OutcomeResult {
                    description: format!("object '{name}' is within {tolerance}m of target"),
                    passed: distance <= *tolerance,
                    details: Some(format!(
                        "actual=({:.3}, {:.3}, {:.3}), expected=({:.3}, {:.3}, {:.3}), distance={distance:.3}",
                        object.pose.position.x,
                        object.pose.position.y,
                        object.pose.position.z,
                        position.x,
                        position.y,
                        position.z,
                    )),
                }
            }
            None => OutcomeResult {
                description: format!("object '{name}' is within {tolerance}m of target"),
                passed: false,
                details: Some("object not found".to_string()),
            },
        },
        ExpectedOutcome::ObjectSemanticLabel { name, label } => {
            match state
                .scene
                .objects
                .values()
                .find(|object| object.name == *name)
            {
                Some(object) => {
                    let actual = object.semantic_label.as_deref();
                    OutcomeResult {
                        description: format!("object '{name}' has semantic label '{label}'"),
                        passed: actual == Some(label.as_str()),
                        details: Some(format!("actual label: {}", actual.unwrap_or("<missing>"))),
                    }
                }
                None => OutcomeResult {
                    description: format!("object '{name}' has semantic label '{label}'"),
                    passed: false,
                    details: Some("object not found".to_string()),
                },
            }
        }
        ExpectedOutcome::MinVideoSimilarity { threshold } => match (ground_truth, video_metrics) {
            (None, _) => OutcomeResult {
                description: format!("video similarity >= {threshold}"),
                passed: false,
                details: Some("scenario does not define ground truth video".to_string()),
            },
            (Some(_), Some(metrics)) => OutcomeResult {
                description: format!("video similarity >= {threshold}"),
                passed: metrics.overall_similarity >= *threshold,
                details: Some(format!(
                    "video similarity: {:.3}",
                    metrics.overall_similarity
                )),
            },
            (Some(_), None) => OutcomeResult {
                description: format!("video similarity >= {threshold}"),
                passed: false,
                details: Some("provider did not return a comparable video clip".to_string()),
            },
        },
        ExpectedOutcome::FinalStateCondition { condition } => {
            let passed = evaluate_condition(condition, state);
            OutcomeResult {
                description: format!("final state matches {}", describe_condition(condition)),
                passed,
                details: Some(if passed {
                    "condition matched".to_string()
                } else {
                    format!("condition did not match: {:?}", condition)
                }),
            }
        }
    }
}

fn describe_condition(condition: &Condition) -> String {
    match condition {
        Condition::ObjectAt {
            object,
            position,
            tolerance,
        } => format!(
            "object {object} at ({:.3}, {:.3}, {:.3}) within {tolerance}",
            position.x, position.y, position.z
        ),
        Condition::ObjectsTouching { a, b } => format!("objects {a} and {b} touching"),
        Condition::ObjectExists { object } => format!("object {object} exists"),
        Condition::And(conditions) => format!(
            "all of ({})",
            conditions
                .iter()
                .map(describe_condition)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Condition::Or(conditions) => format!(
            "any of ({})",
            conditions
                .iter()
                .map(describe_condition)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Condition::Not(inner) => format!("not ({})", describe_condition(inner)),
    }
}

fn ensure_overall_score(scores: &mut HashMap<String, f32>) {
    if scores.contains_key("overall") || scores.is_empty() {
        return;
    }

    let total: f32 = scores.values().copied().sum();
    scores.insert("overall".to_string(), total / scores.len() as f32);
}

fn compare_video_clips(predicted: &VideoClip, ground_truth: &VideoClip) -> VideoMetrics {
    let resolution_similarity = resolution_similarity(predicted, ground_truth);
    let fps_similarity = ratio_similarity(predicted.fps as f64, ground_truth.fps as f64);
    let duration_similarity = ratio_similarity(predicted.duration, ground_truth.duration);
    let frame_count_similarity =
        count_similarity(predicted.frames.len(), ground_truth.frames.len());
    let frame_similarity = average_frame_similarity(predicted, ground_truth, |pred, truth| {
        tensor_similarity(&pred.data, &truth.data)
    });
    let depth_similarity = average_frame_similarity(predicted, ground_truth, |pred, truth| {
        pred.depth
            .as_ref()
            .zip(truth.depth.as_ref())
            .and_then(|(left, right)| tensor_similarity(left, right))
    });
    let segmentation_similarity =
        average_frame_similarity(predicted, ground_truth, |pred, truth| {
            pred.segmentation
                .as_ref()
                .zip(truth.segmentation.as_ref())
                .and_then(|(left, right)| tensor_similarity(left, right))
        });

    let mut components = vec![
        resolution_similarity,
        fps_similarity,
        duration_similarity,
        frame_count_similarity,
    ];
    if let Some(score) = frame_similarity {
        components.push(score);
    }
    if let Some(score) = depth_similarity {
        components.push(score);
    }
    if let Some(score) = segmentation_similarity {
        components.push(score);
    }
    let overall_similarity = if components.is_empty() {
        0.0
    } else {
        components.iter().copied().sum::<f32>() / components.len() as f32
    };

    VideoMetrics {
        overall_similarity,
        resolution_similarity,
        fps_similarity,
        duration_similarity,
        frame_count_similarity,
        frame_similarity,
        depth_similarity,
        segmentation_similarity,
    }
}

fn resolution_similarity(predicted: &VideoClip, ground_truth: &VideoClip) -> f32 {
    let width = ratio_similarity(
        predicted.resolution.0 as f64,
        ground_truth.resolution.0 as f64,
    );
    let height = ratio_similarity(
        predicted.resolution.1 as f64,
        ground_truth.resolution.1 as f64,
    );
    (width + height) / 2.0
}

fn ratio_similarity(left: f64, right: f64) -> f32 {
    if left == 0.0 && right == 0.0 {
        return 1.0;
    }

    let baseline = left.abs().max(right.abs()).max(1.0);
    (1.0 - ((left - right).abs() / baseline) as f32).clamp(0.0, 1.0)
}

fn count_similarity(left: usize, right: usize) -> f32 {
    ratio_similarity(left as f64, right as f64)
}

fn average_frame_similarity(
    predicted: &VideoClip,
    ground_truth: &VideoClip,
    cmp: impl Fn(&worldforge_core::types::Frame, &worldforge_core::types::Frame) -> Option<f32>,
) -> Option<f32> {
    let frame_count = predicted.frames.len().min(ground_truth.frames.len());
    if frame_count == 0 {
        return None;
    }

    let sample_count = frame_count.min(8);
    let mut total = 0.0;
    let mut seen = 0usize;
    for sample_index in 0..sample_count {
        let frame_index = sample_index * frame_count / sample_count;
        if let Some(score) = cmp(
            &predicted.frames[frame_index],
            &ground_truth.frames[frame_index],
        ) {
            total += score;
            seen += 1;
        }
    }

    (seen > 0).then_some(total / seen as f32)
}

fn tensor_similarity(left: &Tensor, right: &Tensor) -> Option<f32> {
    if left.shape != right.shape {
        return Some(0.0);
    }

    let left_values = tensor_values(&left.data);
    let right_values = tensor_values(&right.data);
    let value_count = left_values.len().min(right_values.len());
    if value_count == 0 {
        return Some(1.0);
    }

    let baseline = left_values
        .iter()
        .chain(right_values.iter())
        .map(|value| value.abs())
        .fold(0.0f64, f64::max)
        .max(default_tensor_scale(left))
        .max(default_tensor_scale(right));
    let total_error = left_values
        .iter()
        .zip(right_values.iter())
        .take(value_count)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum::<f64>();
    Some((1.0 - (total_error / value_count as f64 / baseline) as f32).clamp(0.0, 1.0))
}

fn tensor_values(data: &TensorData) -> Vec<f64> {
    match data {
        TensorData::Float32(values) => values.iter().map(|value| *value as f64).collect(),
        TensorData::Float64(values) => values.clone(),
        TensorData::UInt8(values) => values.iter().map(|value| *value as f64).collect(),
        TensorData::Int32(values) => values.iter().map(|value| *value as f64).collect(),
        TensorData::Int64(values) => values.iter().map(|value| *value as f64).collect(),
    }
}

fn default_tensor_scale(tensor: &Tensor) -> f64 {
    match tensor.dtype {
        worldforge_core::types::DType::UInt8 => 255.0,
        _ => 1.0,
    }
}

fn build_leaderboard(provider_summaries: &[ProviderSummary]) -> Vec<LeaderboardEntry> {
    provider_summaries
        .iter()
        .map(|summary| LeaderboardEntry {
            provider: summary.provider.clone(),
            average_score: summary.average_score,
            average_latency_ms: summary.average_latency_ms,
            scenarios_passed: summary.scenarios_passed,
            total_scenarios: summary.total_scenarios,
        })
        .collect()
}

fn build_provider_summaries(
    results: &[EvalResult],
    total_scenarios: usize,
) -> Vec<ProviderSummary> {
    let mut by_provider: HashMap<String, Vec<&EvalResult>> = HashMap::new();
    for r in results {
        by_provider.entry(r.provider.clone()).or_default().push(r);
    }

    let mut summaries: Vec<ProviderSummary> = by_provider
        .into_iter()
        .map(|(provider, results)| {
            let mut overall_total = 0.0;
            let mut overall_count = 0usize;
            let mut dimension_scores: HashMap<String, (f32, usize)> = HashMap::new();
            let mut scenario_scores = HashMap::new();
            let mut outcomes_passed = 0usize;
            let mut total_outcomes = 0usize;

            for result in &results {
                if let Some(score) = result.scores.get("overall") {
                    overall_total += score;
                    overall_count += 1;
                    scenario_scores.insert(result.scenario.clone(), *score);
                }

                for (dimension, score) in &result.scores {
                    let entry = dimension_scores
                        .entry(dimension.clone())
                        .or_insert((0.0, 0));
                    entry.0 += score;
                    entry.1 += 1;
                }

                for outcome in &result.outcomes {
                    outcomes_passed += usize::from(outcome.passed);
                    total_outcomes += 1;
                }
            }

            let average_score = if overall_count == 0 {
                0.0
            } else {
                overall_total / overall_count as f32
            };
            let avg_latency = if results.is_empty() {
                0
            } else {
                results.iter().map(|r| r.latency_ms).sum::<u64>() / results.len() as u64
            };
            let passed = results
                .iter()
                .filter(|r| r.outcomes.iter().all(|o| o.passed))
                .count();

            ProviderSummary {
                provider,
                average_score,
                average_latency_ms: avg_latency,
                scenarios_passed: passed,
                total_scenarios,
                scenario_pass_rate: if total_scenarios == 0 {
                    0.0
                } else {
                    passed as f32 / total_scenarios as f32
                },
                outcomes_passed,
                total_outcomes,
                outcome_pass_rate: if total_outcomes == 0 {
                    1.0
                } else {
                    outcomes_passed as f32 / total_outcomes as f32
                },
                dimension_scores: dimension_scores
                    .into_iter()
                    .map(|(dimension, (total, count))| (dimension, total / count as f32))
                    .collect(),
                scenario_scores,
            }
        })
        .collect();

    summaries.sort_by(|a, b| {
        b.average_score
            .partial_cmp(&a.average_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.provider.cmp(&b.provider))
    });
    summaries
}

fn build_dimension_summaries(
    results: &[EvalResult],
    dimensions: &[EvalDimension],
) -> Vec<DimensionSummary> {
    dimensions
        .iter()
        .map(|dimension| {
            let key = dimension.key();
            let mut by_provider: HashMap<String, (f32, usize)> = HashMap::new();
            for result in results {
                if let Some(score) = result.scores.get(&key) {
                    let entry = by_provider
                        .entry(result.provider.clone())
                        .or_insert((0.0, 0));
                    entry.0 += score;
                    entry.1 += 1;
                }
            }

            let provider_scores: HashMap<String, f32> = by_provider
                .into_iter()
                .map(|(provider, (total, count))| (provider, total / count as f32))
                .collect();

            let best = provider_scores.iter().max_by(|a, b| {
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.0.cmp(a.0))
            });
            let best_provider = best.map(|(provider, _)| provider.clone());
            let best_score = best.map(|(_, score)| *score);

            DimensionSummary {
                dimension: key,
                provider_scores,
                best_provider,
                best_score,
            }
        })
        .collect()
}

fn build_scenario_summaries(
    results: &[EvalResult],
    scenarios: &[EvalScenario],
) -> Vec<ScenarioSummary> {
    scenarios
        .iter()
        .map(|scenario| {
            let scenario_results: Vec<_> = results
                .iter()
                .filter(|result| result.scenario == scenario.name)
                .collect();
            let provider_scores: HashMap<String, f32> = scenario_results
                .iter()
                .filter_map(|result| {
                    result
                        .scores
                        .get("overall")
                        .map(|score| (result.provider.clone(), *score))
                })
                .collect();
            let mut passed_by = Vec::new();
            let mut failed_by = Vec::new();
            let mut outcomes_passed = 0usize;
            let mut total_outcomes = 0usize;

            for result in &scenario_results {
                if result.outcomes.iter().all(|outcome| outcome.passed) {
                    passed_by.push(result.provider.clone());
                } else {
                    failed_by.push(result.provider.clone());
                }

                for outcome in &result.outcomes {
                    outcomes_passed += usize::from(outcome.passed);
                    total_outcomes += 1;
                }
            }

            passed_by.sort();
            failed_by.sort();

            let best = provider_scores.iter().max_by(|a, b| {
                a.1.partial_cmp(b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.0.cmp(a.0))
            });
            let best_provider = best.map(|(provider, _)| provider.clone());
            let best_score = best.map(|(_, score)| *score);

            ScenarioSummary {
                scenario: scenario.name.clone(),
                description: scenario.description.clone(),
                provider_scores,
                passed_by,
                failed_by,
                best_provider,
                best_score,
                outcomes_passed,
                total_outcomes,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use worldforge_core::error::WorldForgeError;
    use worldforge_core::prediction::Prediction;
    use worldforge_core::provider::{
        CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
        ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
    };
    use worldforge_core::types::{BBox, DType, Device, Frame, Pose, SimTime, Tensor, TensorData};
    use worldforge_providers::MockProvider;

    #[derive(Debug)]
    struct SequencedEvalProvider {
        name: String,
        steps: Mutex<Vec<(PhysicsScores, f32)>>,
    }

    impl SequencedEvalProvider {
        fn new(name: &str, steps: Vec<(PhysicsScores, f32)>) -> Self {
            Self {
                name: name.to_string(),
                steps: Mutex::new(steps),
            }
        }
    }

    #[derive(Debug)]
    struct VisualFixtureProvider {
        name: String,
        output_state: WorldState,
        video: VideoClip,
        last_config: Mutex<Option<PredictionConfig>>,
    }

    impl VisualFixtureProvider {
        fn new(name: &str, output_state: WorldState, video: VideoClip) -> Self {
            Self {
                name: name.to_string(),
                output_state,
                video,
                last_config: Mutex::new(None),
            }
        }

        fn last_config(&self) -> Option<PredictionConfig> {
            self.last_config
                .lock()
                .expect("fixture config poisoned")
                .clone()
        }
    }

    fn visual_fixture_state(position: Position, semantic_label: Option<&str>) -> WorldState {
        use worldforge_core::scene::SceneObject;

        let mut state = WorldState::new("visual-fixture", "eval");
        let mut mug = SceneObject::new(
            "mug",
            Pose {
                position,
                ..Default::default()
            },
            BBox {
                min: Position {
                    x: position.x - 0.05,
                    y: position.y - 0.05,
                    z: position.z - 0.05,
                },
                max: Position {
                    x: position.x + 0.05,
                    y: position.y + 0.05,
                    z: position.z + 0.05,
                },
            },
        );
        mug.semantic_label = semantic_label.map(ToOwned::to_owned);
        state.scene.add_object(mug);
        state
    }

    fn visual_fixture_clip() -> VideoClip {
        VideoClip {
            frames: vec![Frame {
                data: Tensor {
                    data: TensorData::Float32(vec![0.1, 0.2, 0.3]),
                    shape: vec![1, 1, 3],
                    dtype: DType::Float32,
                    device: Device::Cpu,
                },
                timestamp: SimTime {
                    step: 0,
                    seconds: 0.0,
                    dt: 0.0,
                },
                camera: None,
                depth: Some(Tensor {
                    data: TensorData::Float32(vec![0.5]),
                    shape: vec![1, 1],
                    dtype: DType::Float32,
                    device: Device::Cpu,
                }),
                segmentation: Some(Tensor {
                    data: TensorData::UInt8(vec![7]),
                    shape: vec![1, 1],
                    dtype: DType::UInt8,
                    device: Device::Cpu,
                }),
            }],
            fps: 24.0,
            resolution: (1, 1),
            duration: 0.5,
        }
    }

    #[async_trait]
    impl WorldModelProvider for SequencedEvalProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                predict: true,
                generate: false,
                reason: false,
                transfer: false,
                action_conditioned: true,
                multi_view: false,
                max_video_length_seconds: 0.0,
                max_resolution: (0, 0),
                fps_range: (0.0, 0.0),
                supported_action_spaces: Vec::new(),
                supports_depth: false,
                supports_segmentation: false,
                supports_planning: false,
                latency_profile: LatencyProfile {
                    p50_ms: 1,
                    p95_ms: 1,
                    p99_ms: 1,
                    throughput_fps: 1.0,
                },
            }
        }

        async fn predict(
            &self,
            state: &WorldState,
            action: &Action,
            _config: &PredictionConfig,
        ) -> Result<Prediction> {
            let (physics_scores, confidence) =
                self.steps.lock().expect("sequence poisoned").remove(0);
            let mut output_state = state.clone();
            output_state.time.step += 1;

            Ok(Prediction {
                id: uuid::Uuid::new_v4(),
                provider: self.name.clone(),
                model: "sequenced".to_string(),
                input_state: state.clone(),
                action: action.clone(),
                output_state,
                video: None,
                confidence,
                physics_scores,
                latency_ms: 1,
                cost: CostEstimate::default(),
                guardrail_results: Vec::new(),
                timestamp: chrono::Utc::now(),
            })
        }

        async fn generate(
            &self,
            _prompt: &GenerationPrompt,
            _config: &GenerationConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.clone(),
                capability: "generate".to_string(),
            })
        }

        async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.clone(),
                capability: "reason".to_string(),
            })
        }

        async fn transfer(
            &self,
            _source: &VideoClip,
            _controls: &SpatialControls,
            _config: &TransferConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.clone(),
                capability: "transfer".to_string(),
            })
        }

        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                healthy: true,
                message: "healthy".to_string(),
                latency_ms: 1,
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }

    #[async_trait]
    impl WorldModelProvider for VisualFixtureProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                predict: true,
                generate: false,
                reason: false,
                transfer: false,
                action_conditioned: true,
                multi_view: false,
                max_video_length_seconds: 10.0,
                max_resolution: (1, 1),
                fps_range: (24.0, 24.0),
                supported_action_spaces: Vec::new(),
                supports_depth: true,
                supports_segmentation: true,
                supports_planning: false,
                latency_profile: LatencyProfile {
                    p50_ms: 1,
                    p95_ms: 1,
                    p99_ms: 1,
                    throughput_fps: 1.0,
                },
            }
        }

        async fn predict(
            &self,
            state: &WorldState,
            action: &Action,
            config: &PredictionConfig,
        ) -> Result<Prediction> {
            *self.last_config.lock().expect("fixture config poisoned") = Some(config.clone());

            Ok(Prediction {
                id: uuid::Uuid::new_v4(),
                provider: self.name.clone(),
                model: "visual-fixture".to_string(),
                input_state: state.clone(),
                action: action.clone(),
                output_state: self.output_state.clone(),
                video: config.return_video.then_some(self.video.clone()),
                confidence: 0.95,
                physics_scores: PhysicsScores {
                    overall: 0.92,
                    object_permanence: 0.9,
                    gravity_compliance: 0.91,
                    collision_accuracy: 0.92,
                    spatial_consistency: 0.93,
                    temporal_consistency: 0.94,
                },
                latency_ms: 1,
                cost: CostEstimate::default(),
                guardrail_results: Vec::new(),
                timestamp: chrono::Utc::now(),
            })
        }

        async fn generate(
            &self,
            _prompt: &GenerationPrompt,
            _config: &GenerationConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.clone(),
                capability: "generate".to_string(),
            })
        }

        async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.clone(),
                capability: "reason".to_string(),
            })
        }

        async fn transfer(
            &self,
            _source: &VideoClip,
            _controls: &SpatialControls,
            _config: &TransferConfig,
        ) -> Result<VideoClip> {
            Err(WorldForgeError::UnsupportedCapability {
                provider: self.name.clone(),
                capability: "transfer".to_string(),
            })
        }

        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                healthy: true,
                message: "healthy".to_string(),
                latency_ms: 1,
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }

    #[test]
    fn test_eval_suite_creation() {
        let suite = EvalSuite::physics_standard();
        assert_eq!(suite.name, "Physics Standard");
        assert!(!suite.scenarios.is_empty());
        assert!(!suite.dimensions.is_empty());
    }

    #[tokio::test]
    async fn test_eval_suite_run() {
        let suite = EvalSuite::physics_standard();
        let provider = MockProvider::new();
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];
        let report = suite.run(&providers).await.unwrap();
        assert!(!report.results.is_empty());
        assert!(!report.leaderboard.is_empty());
        assert_eq!(report.provider_summaries.len(), 1);
        assert_eq!(report.dimension_summaries.len(), suite.dimensions.len());
        assert_eq!(report.scenario_summaries.len(), suite.scenarios.len());
        assert_eq!(report.leaderboard[0].provider, "mock");
    }

    #[test]
    fn test_manipulation_suite_creation() {
        let suite = EvalSuite::manipulation_standard();
        assert_eq!(suite.name, "Manipulation Standard");
        assert_eq!(suite.scenarios.len(), 3);
        assert!(suite.dimensions.contains(&EvalDimension::ObjectPermanence));
        assert!(suite
            .dimensions
            .contains(&EvalDimension::ActionPredictionAccuracy));
    }

    #[tokio::test]
    async fn test_manipulation_suite_run() {
        let suite = EvalSuite::manipulation_standard();
        let provider = MockProvider::new();
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];
        let report = suite.run(&providers).await.unwrap();
        assert_eq!(report.results.len(), 3);
        assert_eq!(report.leaderboard[0].provider, "mock");
    }

    #[test]
    fn test_spatial_reasoning_suite_creation() {
        let suite = EvalSuite::spatial_reasoning();
        assert_eq!(suite.name, "Spatial Reasoning");
        assert_eq!(suite.scenarios.len(), 2);
        assert!(suite.dimensions.contains(&EvalDimension::SpatialReasoning));
    }

    #[tokio::test]
    async fn test_spatial_reasoning_suite_run() {
        let suite = EvalSuite::spatial_reasoning();
        let provider = MockProvider::new();
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];
        let report = suite.run(&providers).await.unwrap();
        assert_eq!(report.results.len(), 2);
    }

    #[test]
    fn test_comprehensive_suite_creation() {
        let suite = EvalSuite::comprehensive();
        assert_eq!(suite.name, "Comprehensive");
        // 2 physics + 3 manipulation + 2 spatial = 7
        assert_eq!(suite.scenarios.len(), 7);
        assert_eq!(suite.dimensions.len(), 7);
    }

    #[tokio::test]
    async fn test_comprehensive_suite_run() {
        let suite = EvalSuite::comprehensive();
        let provider = MockProvider::new();
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];
        let report = suite.run(&providers).await.unwrap();
        assert_eq!(report.results.len(), 7);
        assert_eq!(report.leaderboard[0].total_scenarios, 7);
        assert!(report.total_outcomes >= report.outcomes_passed);
    }

    #[test]
    fn test_builtin_names_are_exposed() {
        assert_eq!(
            EvalSuite::builtin_names(),
            &["physics", "manipulation", "spatial", "comprehensive"]
        );
    }

    #[test]
    fn test_suite_json_roundtrip_and_lookup() {
        let suite = EvalSuite::from_builtin("physics").unwrap();
        let json = suite.to_json_pretty().unwrap();
        let restored = EvalSuite::from_json_str(&json).unwrap();
        assert_eq!(restored.name, suite.name);
        assert_eq!(restored.scenarios.len(), suite.scenarios.len());
    }

    #[tokio::test]
    async fn test_expected_outcomes_are_checked_once_per_scenario() {
        let suite = EvalSuite {
            name: "Two-step".to_string(),
            scenarios: vec![EvalScenario {
                name: "two_moves".to_string(),
                description: "Run two actions and validate once at the end".to_string(),
                initial_state: WorldState::new("two-step", "eval"),
                actions: vec![
                    Action::Move {
                        target: worldforge_core::types::Position {
                            x: 1.0,
                            y: 0.0,
                            z: 0.0,
                        },
                        speed: 1.0,
                    },
                    Action::Move {
                        target: worldforge_core::types::Position {
                            x: 2.0,
                            y: 0.0,
                            z: 0.0,
                        },
                        speed: 1.0,
                    },
                ],
                expected_outcomes: vec![ExpectedOutcome::MinConfidence { threshold: 0.5 }],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::ActionPredictionAccuracy],
        };
        let provider = MockProvider::new();
        let report = suite
            .run(&[&provider as &dyn WorldModelProvider])
            .await
            .unwrap();
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].outcomes.len(), 1);
        assert!(report.results[0].outcomes[0].passed);
    }

    #[tokio::test]
    async fn test_eval_report_builds_rollups() {
        let suite = EvalSuite::physics_standard();
        let provider = MockProvider::new();
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];
        let report = suite.run(&providers).await.unwrap();

        let provider_summary = &report.provider_summaries[0];
        assert_eq!(provider_summary.provider, "mock");
        assert!(provider_summary.dimension_scores.contains_key("overall"));
        assert_eq!(provider_summary.total_scenarios, suite.scenarios.len());

        let dimension_summary = report
            .dimension_summaries
            .iter()
            .find(|summary| summary.dimension == "gravity_compliance")
            .unwrap();
        assert_eq!(dimension_summary.best_provider.as_deref(), Some("mock"));

        let scenario_summary = report
            .scenario_summaries
            .iter()
            .find(|summary| summary.scenario == "object_drop")
            .unwrap();
        assert!(scenario_summary.provider_scores.contains_key("mock"));
        assert!(scenario_summary.passed_by.contains(&"mock".to_string()));
    }

    #[tokio::test]
    async fn test_min_threshold_outcomes_use_multi_step_averages() {
        let scenario = EvalScenario {
            name: "two_step_average".to_string(),
            description: "Average scores and confidence should drive threshold checks".to_string(),
            initial_state: WorldState::new("two_step_average", "eval"),
            actions: vec![
                Action::SetLighting { time_of_day: 0.25 },
                Action::SetLighting { time_of_day: 0.75 },
            ],
            expected_outcomes: vec![
                ExpectedOutcome::MinPhysicsScore {
                    dimension: EvalDimension::GravityCompliance,
                    threshold: 0.55,
                },
                ExpectedOutcome::MinConfidence { threshold: 0.60 },
            ],
            ground_truth: None,
        };
        let suite = EvalSuite {
            name: "Average Threshold".to_string(),
            scenarios: vec![scenario],
            dimensions: vec![EvalDimension::GravityCompliance],
        };
        let provider = SequencedEvalProvider::new(
            "sequenced",
            vec![
                (
                    PhysicsScores {
                        overall: 0.2,
                        object_permanence: 0.2,
                        gravity_compliance: 0.2,
                        collision_accuracy: 0.2,
                        spatial_consistency: 0.2,
                        temporal_consistency: 0.2,
                    },
                    0.9,
                ),
                (
                    PhysicsScores {
                        overall: 0.9,
                        object_permanence: 0.9,
                        gravity_compliance: 0.9,
                        collision_accuracy: 0.9,
                        spatial_consistency: 0.9,
                        temporal_consistency: 0.9,
                    },
                    0.3,
                ),
            ],
        );
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];

        let report = suite.run(&providers).await.unwrap();
        let result = &report.results[0];

        assert_eq!(result.scores["gravity_compliance"], 0.55);
        assert_eq!(result.scores["overall"], 0.55);
        assert!(result.outcomes.iter().all(|outcome| outcome.passed));
    }

    #[tokio::test]
    async fn test_object_position_and_label_outcomes_pass_for_mock_provider() {
        let mut initial_state = WorldState::new("scene_assertions", "mock");
        let target = Position {
            x: 0.4,
            y: 0.8,
            z: 0.1,
        };
        let mut object = worldforge_core::scene::SceneObject::new(
            "red_mug",
            worldforge_core::types::Pose::default(),
            worldforge_core::types::BBox::from_center_half_extents(
                Position::default(),
                worldforge_core::types::Vec3 {
                    x: 0.05,
                    y: 0.05,
                    z: 0.05,
                },
            ),
        );
        object.semantic_label = Some("mug".to_string());
        let object_id = object.id;
        initial_state.scene.add_object(object);

        let suite = EvalSuite {
            name: "Scene Assertions".to_string(),
            scenarios: vec![EvalScenario {
                name: "move_and_check".to_string(),
                description: "Move a labeled object and verify the final scene".to_string(),
                initial_state,
                actions: vec![Action::Move { target, speed: 1.0 }],
                expected_outcomes: vec![
                    ExpectedOutcome::FinalStateCondition {
                        condition: Condition::And(vec![
                            Condition::ObjectExists { object: object_id },
                            Condition::ObjectAt {
                                object: object_id,
                                position: target,
                                tolerance: 0.001,
                            },
                        ]),
                    },
                    ExpectedOutcome::ObjectSemanticLabel {
                        name: "red_mug".to_string(),
                        label: "mug".to_string(),
                    },
                ],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::ActionPredictionAccuracy],
        };
        let provider = MockProvider::new();
        let report = suite
            .run(&[&provider as &dyn WorldModelProvider])
            .await
            .unwrap();
        let result = &report.results[0];

        assert!(result.outcomes.iter().all(|outcome| outcome.passed));
    }

    #[tokio::test]
    async fn test_final_state_condition_outcome_fails_and_passes() {
        let mut state = WorldState::new("condition_checks", "eval");
        let object = worldforge_core::scene::SceneObject::new(
            "crate",
            worldforge_core::types::Pose {
                position: Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                ..Default::default()
            },
            worldforge_core::types::BBox::from_center_half_extents(
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                worldforge_core::types::Vec3 {
                    x: 0.1,
                    y: 0.1,
                    z: 0.1,
                },
            ),
        );
        let object_id = object.id;
        state.scene.add_object(object);

        let passing_suite = EvalSuite {
            name: "Condition Pass".to_string(),
            scenarios: vec![EvalScenario {
                name: "object_exists_and_position".to_string(),
                description: "Final state should satisfy a compound condition".to_string(),
                initial_state: state.clone(),
                actions: vec![Action::Move {
                    target: Position {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                    speed: 1.0,
                }],
                expected_outcomes: vec![ExpectedOutcome::FinalStateCondition {
                    condition: Condition::And(vec![
                        Condition::ObjectExists { object: object_id },
                        Condition::ObjectAt {
                            object: object_id,
                            position: Position {
                                x: 0.0,
                                y: 1.0,
                                z: 0.0,
                            },
                            tolerance: 0.001,
                        },
                    ]),
                }],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::ActionPredictionAccuracy],
        };

        let passing_report = passing_suite
            .run(&[&MockProvider::new() as &dyn WorldModelProvider])
            .await
            .unwrap();
        assert!(passing_report.results[0].outcomes[0].passed);

        let failing_suite = EvalSuite {
            name: "Condition Fail".to_string(),
            scenarios: vec![EvalScenario {
                name: "wrong_position".to_string(),
                description: "Final state should fail a condition when the target is wrong"
                    .to_string(),
                initial_state: state,
                actions: vec![Action::Move {
                    target: Position {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                    speed: 1.0,
                }],
                expected_outcomes: vec![ExpectedOutcome::FinalStateCondition {
                    condition: Condition::ObjectAt {
                        object: object_id,
                        position: Position {
                            x: 1.0,
                            y: 1.0,
                            z: 0.0,
                        },
                        tolerance: 0.001,
                    },
                }],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::ActionPredictionAccuracy],
        };

        let failing_report = failing_suite
            .run(&[&MockProvider::new() as &dyn WorldModelProvider])
            .await
            .unwrap();
        assert!(!failing_report.results[0].outcomes[0].passed);
    }

    #[test]
    fn test_final_state_condition_outcome_json_roundtrip() {
        let object_id = uuid::Uuid::new_v4();
        let outcome = ExpectedOutcome::FinalStateCondition {
            condition: Condition::Or(vec![
                Condition::ObjectExists { object: object_id },
                Condition::Not(Box::new(Condition::ObjectsTouching {
                    a: object_id,
                    b: uuid::Uuid::new_v4(),
                })),
            ]),
        };
        let json = serde_json::to_string(&outcome).unwrap();
        let restored: ExpectedOutcome = serde_json::from_str(&json).unwrap();

        match restored {
            ExpectedOutcome::FinalStateCondition { condition } => match condition {
                Condition::Or(conditions) => assert_eq!(conditions.len(), 2),
                _ => panic!("expected Or condition"),
            },
            _ => panic!("expected FinalStateCondition"),
        }
    }

    #[tokio::test]
    async fn test_ground_truth_video_similarity_is_reported() {
        let provider = MockProvider::new();
        let state = WorldState::new("ground_truth_video", "mock");
        let action = Action::SetLighting { time_of_day: 0.4 };
        let config = PredictionConfig {
            return_video: true,
            ..PredictionConfig::default()
        };
        let prediction = provider.predict(&state, &action, &config).await.unwrap();
        let ground_truth = prediction.video.clone().unwrap();

        let suite = EvalSuite {
            name: "Ground Truth Video".to_string(),
            scenarios: vec![EvalScenario {
                name: "video_match".to_string(),
                description: "Compare the generated clip against known ground truth".to_string(),
                initial_state: state,
                actions: vec![action],
                expected_outcomes: vec![ExpectedOutcome::MinVideoSimilarity { threshold: 0.95 }],
                ground_truth: Some(ground_truth),
            }],
            dimensions: vec![EvalDimension::Custom {
                name: "video_similarity".to_string(),
            }],
        };

        let report = suite
            .run(&[&provider as &dyn WorldModelProvider])
            .await
            .unwrap();
        let result = &report.results[0];

        assert!(result.video.is_some());
        assert!(result.video_metrics.is_some());
        assert!(result.scores["video_similarity"] >= 0.95);
        assert!(result.outcomes.iter().all(|outcome| outcome.passed));
        assert_eq!(
            report.dimension_summaries[0].best_provider.as_deref(),
            Some("mock")
        );
    }

    #[test]
    fn test_validate_rejects_duplicate_scenario_names() {
        let suite = EvalSuite {
            name: "bad".to_string(),
            scenarios: vec![
                EvalScenario {
                    name: "duplicate".to_string(),
                    description: "first".to_string(),
                    initial_state: WorldState::new("a", "eval"),
                    actions: vec![],
                    expected_outcomes: vec![],
                    ground_truth: None,
                },
                EvalScenario {
                    name: "duplicate".to_string(),
                    description: "second".to_string(),
                    initial_state: WorldState::new("b", "eval"),
                    actions: vec![],
                    expected_outcomes: vec![],
                    ground_truth: None,
                },
            ],
            dimensions: vec![EvalDimension::SpatialConsistency],
        };

        assert!(suite.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_video_similarity_without_ground_truth() {
        let suite = EvalSuite {
            name: "video-bad".to_string(),
            scenarios: vec![EvalScenario {
                name: "needs_video".to_string(),
                description: "requires a reference clip".to_string(),
                initial_state: WorldState::new("needs_video", "eval"),
                actions: vec![],
                expected_outcomes: vec![ExpectedOutcome::MinVideoSimilarity { threshold: 0.9 }],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::SpatialConsistency],
        };

        assert!(suite.validate().is_err());
    }

    #[tokio::test]
    async fn test_visual_scenario_requests_and_scores_video_artifacts() {
        let target = Position {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let ground_truth = visual_fixture_clip();
        let suite = EvalSuite {
            name: "visual-fixture".to_string(),
            scenarios: vec![EvalScenario {
                name: "mug_alignment".to_string(),
                description: "The mug should move to the target pose and preserve its label"
                    .to_string(),
                initial_state: visual_fixture_state(
                    Position {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    Some("cup"),
                ),
                actions: vec![Action::Move { target, speed: 1.0 }],
                expected_outcomes: vec![
                    ExpectedOutcome::ObjectPosition {
                        name: "mug".to_string(),
                        position: target,
                        tolerance: 0.001,
                    },
                    ExpectedOutcome::ObjectSemanticLabel {
                        name: "mug".to_string(),
                        label: "mug".to_string(),
                    },
                    ExpectedOutcome::MinVideoSimilarity { threshold: 0.99 },
                ],
                ground_truth: Some(ground_truth.clone()),
            }],
            dimensions: vec![EvalDimension::SpatialConsistency],
        };
        let provider = VisualFixtureProvider::new(
            "visual-fixture",
            visual_fixture_state(target, Some("mug")),
            ground_truth,
        );

        let report = suite
            .run(&[&provider as &dyn WorldModelProvider])
            .await
            .unwrap();
        let result = &report.results[0];
        let last_config = provider.last_config().expect("config should be captured");

        assert!(last_config.return_video);
        assert!(last_config.return_depth);
        assert!(last_config.return_segmentation);
        assert!(result.video.is_some());
        assert!(result.video_metrics.is_some());
        assert_eq!(
            result.video_metrics.as_ref().unwrap().overall_similarity,
            1.0
        );
        assert_eq!(result.scores["video_similarity"], 1.0);
        assert_eq!(
            result.video_metrics.as_ref().unwrap().frame_similarity,
            Some(1.0)
        );
        assert_eq!(
            result.video_metrics.as_ref().unwrap().depth_similarity,
            Some(1.0)
        );
        assert_eq!(
            result
                .video_metrics
                .as_ref()
                .unwrap()
                .segmentation_similarity,
            Some(1.0)
        );
        assert!(result.outcomes.iter().all(|outcome| outcome.passed));
    }
}
