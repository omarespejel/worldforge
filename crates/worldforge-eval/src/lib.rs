//! WorldForge Evaluation Framework
//!
//! Provides standardized evaluation of world foundation models across
//! dimensions like physics plausibility, spatial consistency, and
//! temporal coherence.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use worldforge_core::action::Action;
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::prediction::{PhysicsScores, PredictionConfig};
use worldforge_core::provider::WorldModelProvider;
use worldforge_core::state::WorldState;
use worldforge_core::types::VideoClip;

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
    /// The minimum physics score threshold.
    MinPhysicsScore {
        dimension: EvalDimension,
        threshold: f32,
    },
    /// The prediction confidence should be above a threshold.
    MinConfidence { threshold: f32 },
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
    /// Whether each expected outcome was met.
    pub outcomes: Vec<OutcomeResult>,
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
                        ExpectedOutcome::ObjectExists {
                            name: "mug".to_string(),
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
                        ExpectedOutcome::ObjectExists {
                            name: "box_a".to_string(),
                        },
                        ExpectedOutcome::ObjectExists {
                            name: "box_b".to_string(),
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
        let config = PredictionConfig::default();
        let mut all_results = Vec::new();

        for provider in providers {
            if !provider.capabilities().predict {
                for scenario in &self.scenarios {
                    all_results.push(EvalResult {
                        provider: provider.name().to_string(),
                        scenario: scenario.name.clone(),
                        scores: HashMap::new(),
                        latency_ms: 0,
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

                if !prediction_failed {
                    for expected in &scenario.expected_outcomes {
                        outcomes.push(check_outcome(
                            expected,
                            &current_state,
                            average_scores.as_ref(),
                            average_confidence,
                        ));
                    }
                }

                all_results.push(EvalResult {
                    provider: provider.name().to_string(),
                    scenario: scenario.name.clone(),
                    scores,
                    latency_ms: start.elapsed().as_millis() as u64,
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
}

impl ScenarioAccumulator {
    fn record(&mut self, prediction: &worldforge_core::prediction::Prediction) {
        self.physics.record(&prediction.physics_scores);
        self.total_confidence += prediction.confidence;
        self.count += 1;
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

fn check_outcome(
    expected: &ExpectedOutcome,
    state: &WorldState,
    average_scores: Option<&PhysicsScores>,
    average_confidence: Option<f32>,
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
}
