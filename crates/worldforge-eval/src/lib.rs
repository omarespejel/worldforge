//! WorldForge Evaluation Framework
//!
//! Provides standardized evaluation of world foundation models across
//! dimensions like physics plausibility, spatial consistency, and
//! temporal coherence.

mod async_utils;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::async_utils::{join_all_ordered, BoxFuture};
use worldforge_core::action::{evaluate_condition, Action, Condition};
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::prediction::{PhysicsScores, PredictionConfig};
use worldforge_core::provider::WorldModelProvider;
use worldforge_core::state::{WorldMetadata, WorldState};
use worldforge_core::types::{Position, Tensor, TensorData, VideoClip};

const BUILTIN_SUITE_NAMES: [&str; 4] = ["physics", "manipulation", "spatial", "comprehensive"];
const SUPPORTED_CUSTOM_DIMENSION_NAMES: [&str; 8] = [
    "overall",
    "object_permanence",
    "gravity_compliance",
    "collision_accuracy",
    "spatial_consistency",
    "temporal_consistency",
    "confidence",
    "video_similarity",
];

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
    /// Default providers to evaluate when callers do not override them.
    #[serde(default)]
    pub providers: Vec<String>,
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

/// Supported serialization and rendering formats for evaluation reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EvalReportFormat {
    /// Pretty-printed JSON with the full structured report payload.
    #[default]
    Json,
    /// A human-readable Markdown report with summary tables.
    Markdown,
    /// A CSV export with one row per provider/scenario result.
    Csv,
}

impl EvalReportFormat {
    /// Canonical lowercase identifier for this format.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Markdown => "markdown",
            Self::Csv => "csv",
        }
    }
}

impl fmt::Display for EvalReportFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EvalReportFormat {
    type Err = WorldForgeError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "markdown" | "md" => Ok(Self::Markdown),
            "csv" => Ok(Self::Csv),
            other => Err(WorldForgeError::InvalidState(format!(
                "unknown evaluation report format: {other}. Available: json, markdown, csv"
            ))),
        }
    }
}

impl EvalReport {
    /// Deserialize a full evaluation report from JSON.
    pub fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|error| WorldForgeError::SerializationError(error.to_string()))
    }

    /// Serialize the report to pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| WorldForgeError::SerializationError(error.to_string()))
    }

    /// Render the report using one of the supported export formats.
    pub fn render(&self, format: EvalReportFormat) -> Result<String> {
        match format {
            EvalReportFormat::Json => self.to_json_pretty(),
            EvalReportFormat::Markdown => self.to_markdown(),
            EvalReportFormat::Csv => self.to_csv(),
        }
    }

    /// Render the report in multiple formats, deduplicating repeats while
    /// preserving the first-requested order.
    pub fn render_many<I>(&self, formats: I) -> Result<Vec<(EvalReportFormat, String)>>
    where
        I: IntoIterator<Item = EvalReportFormat>,
    {
        let mut rendered = Vec::new();

        for format in formats {
            if rendered
                .iter()
                .any(|(existing_format, _)| *existing_format == format)
            {
                continue;
            }

            rendered.push((format, self.render(format)?));
        }

        Ok(rendered)
    }

    /// Render a deterministic Markdown summary of the report.
    pub fn to_markdown(&self) -> Result<String> {
        let mut output = String::new();
        output.push_str("# Evaluation Report: ");
        output.push_str(&self.suite);
        output.push_str("\n\n");
        output.push_str("## Summary\n\n");
        output.push_str(&format!(
            "- Providers evaluated: {}\n- Scenarios evaluated: {}\n- Outcomes passed: {}/{}\n\n",
            self.provider_summaries.len(),
            self.scenario_summaries.len(),
            self.outcomes_passed,
            self.total_outcomes,
        ));

        output.push_str("## Leaderboard\n\n");
        if self.leaderboard.is_empty() {
            output.push_str("_No leaderboard entries recorded._\n\n");
        } else {
            output.push_str(
                "| Provider | Avg Score | Avg Latency (ms) | Scenarios Passed |\n| --- | ---: | ---: | ---: |\n",
            );
            for entry in &self.leaderboard {
                output.push_str(&format!(
                    "| {} | {:.3} | {} | {}/{} |\n",
                    markdown_cell(&entry.provider),
                    entry.average_score,
                    entry.average_latency_ms,
                    entry.scenarios_passed,
                    entry.total_scenarios,
                ));
            }
            output.push('\n');
        }

        output.push_str("## Provider Summaries\n\n");
        if self.provider_summaries.is_empty() {
            output.push_str("_No provider summaries recorded._\n\n");
        } else {
            for summary in &self.provider_summaries {
                output.push_str("### ");
                output.push_str(&summary.provider);
                output.push_str("\n\n");
                output.push_str(&format!(
                    "- Average score: {:.3}\n- Average latency: {} ms\n- Scenario pass rate: {} ({}/{})\n- Outcome pass rate: {} ({}/{})\n",
                    summary.average_score,
                    summary.average_latency_ms,
                    format_percent(summary.scenario_pass_rate),
                    summary.scenarios_passed,
                    summary.total_scenarios,
                    format_percent(summary.outcome_pass_rate),
                    summary.outcomes_passed,
                    summary.total_outcomes,
                ));

                if summary.dimension_scores.is_empty() {
                    output.push_str("\n_No dimension scores recorded._\n\n");
                } else {
                    output.push_str("\n| Dimension | Score |\n| --- | ---: |\n");
                    let mut dimensions: Vec<_> = summary.dimension_scores.iter().collect();
                    dimensions.sort_by(|left, right| left.0.cmp(right.0));
                    for (dimension, score) in dimensions {
                        output.push_str(&format!(
                            "| {} | {:.3} |\n",
                            markdown_cell(dimension),
                            score,
                        ));
                    }
                    output.push('\n');
                }
            }
        }

        output.push_str("## Dimension Summaries\n\n");
        if self.dimension_summaries.is_empty() {
            output.push_str("_No dimension summaries recorded._\n\n");
        } else {
            output.push_str(
                "| Dimension | Best Provider | Best Score | Provider Scores |\n| --- | --- | ---: | --- |\n",
            );
            for summary in &self.dimension_summaries {
                let provider_scores = provider_score_list(&summary.provider_scores);
                output.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    markdown_cell(&summary.dimension),
                    markdown_optional(summary.best_provider.as_deref()),
                    format_optional_score(summary.best_score),
                    markdown_cell(&provider_scores),
                ));
            }
            output.push('\n');
        }

        output.push_str("## Scenario Summaries\n\n");
        if self.scenario_summaries.is_empty() {
            output.push_str("_No scenario summaries recorded._\n");
        } else {
            output.push_str("| Scenario | Best Provider | Best Score | Outcomes | Passed By | Failed By |\n| --- | --- | ---: | ---: | --- | --- |\n");
            for summary in &self.scenario_summaries {
                output.push_str(&format!(
                    "| {} | {} | {} | {}/{} | {} | {} |\n",
                    markdown_cell(&summary.scenario),
                    markdown_optional(summary.best_provider.as_deref()),
                    format_optional_score(summary.best_score),
                    summary.outcomes_passed,
                    summary.total_outcomes,
                    markdown_joined_list(&summary.passed_by),
                    markdown_joined_list(&summary.failed_by),
                ));
            }
        }

        Ok(output)
    }

    /// Render the report as CSV with one row per provider/scenario result.
    pub fn to_csv(&self) -> Result<String> {
        let mut output = String::new();
        let score_keys = report_score_keys(&self.results);

        let mut header = vec![
            "suite".to_string(),
            "provider".to_string(),
            "scenario".to_string(),
            "overall_score".to_string(),
            "latency_ms".to_string(),
            "passed_outcomes".to_string(),
            "total_outcomes".to_string(),
            "all_outcomes_passed".to_string(),
        ];
        header.extend(score_keys.iter().cloned());
        output.push_str(&header.join(","));
        output.push('\n');

        for result in &self.results {
            let passed_outcomes = result
                .outcomes
                .iter()
                .filter(|outcome| outcome.passed)
                .count();
            let total_outcomes = result.outcomes.len();
            let all_outcomes_passed = total_outcomes > 0 && passed_outcomes == total_outcomes;
            let mut row = vec![
                csv_cell(&self.suite),
                csv_cell(&result.provider),
                csv_cell(&result.scenario),
                csv_cell(
                    &result
                        .scores
                        .get("overall")
                        .map(|score| format!("{score:.3}"))
                        .unwrap_or_default(),
                ),
                result.latency_ms.to_string(),
                passed_outcomes.to_string(),
                total_outcomes.to_string(),
                all_outcomes_passed.to_string(),
            ];

            for score_key in &score_keys {
                row.push(csv_cell(
                    &result
                        .scores
                        .get(score_key)
                        .map(|score| format!("{score:.3}"))
                        .unwrap_or_default(),
                ));
            }

            output.push_str(&row.join(","));
            output.push('\n');
        }

        Ok(output)
    }
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

        for dimension in &self.dimensions {
            let EvalDimension::Custom { name } = dimension else {
                continue;
            };
            let normalized = name.trim();
            if normalized.is_empty() {
                return Err(WorldForgeError::InvalidState(format!(
                    "evaluation suite '{}' declares an empty custom dimension name",
                    self.name
                )));
            }
            if !is_supported_custom_dimension(normalized) {
                return Err(WorldForgeError::InvalidState(format!(
                    "unsupported custom evaluation dimension '{normalized}'. Supported names: {}",
                    SUPPORTED_CUSTOM_DIMENSION_NAMES.join(", ")
                )));
            }
            if normalized.eq_ignore_ascii_case("video_similarity")
                && self
                    .scenarios
                    .iter()
                    .any(|scenario| scenario.ground_truth.is_none())
            {
                return Err(WorldForgeError::InvalidState(format!(
                    "custom dimension 'video_similarity' requires ground truth video for every scenario in suite '{}'",
                    self.name
                )));
            }
        }

        let mut seen_providers = HashSet::new();
        for provider in &self.providers {
            let provider = provider.trim();
            if provider.is_empty() {
                return Err(WorldForgeError::InvalidState(format!(
                    "evaluation suite '{}' contains an empty provider name",
                    self.name
                )));
            }
            if !seen_providers.insert(provider.to_string()) {
                return Err(WorldForgeError::InvalidState(format!(
                    "duplicate evaluation provider name: {provider}"
                )));
            }
        }

        Ok(())
    }

    /// Return the suite's default provider names.
    pub fn providers(&self) -> &[String] {
        &self.providers
    }

    /// Resolve the provider names to use for an evaluation run.
    pub fn effective_provider_names(&self, requested: &[String]) -> Vec<String> {
        let requested = normalize_provider_names(requested);
        if !requested.is_empty() {
            return requested;
        }

        let defaults = normalize_provider_names(&self.providers);
        if !defaults.is_empty() {
            return defaults;
        }

        vec!["mock".to_string()]
    }

    /// Run the suite against a supplied world state.
    ///
    /// The supplied world is merged over each scenario's fixture state before
    /// evaluation. This preserves the scenario's action fixtures while letting
    /// persisted world context influence the results.
    pub async fn run_with_world_state(
        &self,
        providers: &[&dyn WorldModelProvider],
        world_state: &WorldState,
    ) -> Result<EvalReport> {
        self.run_internal(providers, Some(world_state)).await
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
            providers: vec!["mock".to_string()],
        }
    }

    /// Create a manipulation evaluation suite.
    ///
    /// Tests object grasping, placement, and compound manipulation tasks.
    pub fn manipulation_standard() -> Self {
        use worldforge_core::scene::SceneObject;
        use worldforge_core::types::{BBox, Pose, Position};

        let mut state = WorldState::new("manipulation_test", "eval");
        let mut mug = SceneObject::new(
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
        mug.semantic_label = Some("mug".to_string());
        mug.physics.material = Some("ceramic".to_string());
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
        table.semantic_label = Some("table".to_string());
        table.physics.material = Some("wood".to_string());
        let mut block = SceneObject::new(
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
        block.semantic_label = Some("block".to_string());
        block.physics.material = Some("rubber".to_string());
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
                        ExpectedOutcome::ObjectSemanticLabel {
                            name: "mug".to_string(),
                            label: "mug".to_string(),
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
                        ExpectedOutcome::ObjectSemanticLabel {
                            name: "block".to_string(),
                            label: "block".to_string(),
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
                EvalDimension::MaterialUnderstanding,
            ],
            providers: vec!["mock".to_string()],
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
            providers: vec!["mock".to_string()],
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
                EvalDimension::MaterialUnderstanding,
                EvalDimension::SpatialReasoning,
            ],
            providers: vec!["mock".to_string()],
        }
    }

    /// Run the evaluation suite against a set of providers.
    pub async fn run(&self, providers: &[&dyn WorldModelProvider]) -> Result<EvalReport> {
        self.run_internal(providers, None).await
    }

    async fn run_internal(
        &self,
        providers: &[&dyn WorldModelProvider],
        world_state: Option<&WorldState>,
    ) -> Result<EvalReport> {
        self.validate()?;
        let mut futures = Vec::<BoxFuture<'_, EvalResult>>::new();

        for &provider in providers {
            for scenario in &self.scenarios {
                futures.push(Box::pin(evaluate_provider_scenario(
                    provider,
                    scenario,
                    world_state,
                    &self.dimensions,
                )));
            }
        }

        let all_results = join_all_ordered(futures).await;

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

async fn evaluate_provider_scenario(
    provider: &dyn WorldModelProvider,
    scenario: &EvalScenario,
    world_state: Option<&WorldState>,
    dimensions: &[EvalDimension],
) -> EvalResult {
    if !provider.capabilities().predict {
        return EvalResult {
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
                    "evaluation requires predict capability for every scenario".to_string(),
                ),
            }],
        };
    }

    let start = std::time::Instant::now();
    let config = prediction_config_for_scenario(scenario);
    let mut score_accumulator = ScenarioAccumulator::default();
    let mut outcomes = Vec::new();
    let mut current_state = scenario.initial_state.clone();

    if let Some(world_state) = world_state {
        current_state = merge_world_state(&current_state, world_state);
    }
    let starting_state = current_state.clone();

    let mut prediction_failed = false;
    for action in &scenario.actions {
        match provider.predict(&current_state, action, &config).await {
            Ok(prediction) => {
                score_accumulator.record(&prediction);
                current_state = prediction.output_state.clone();
            }
            Err(error) => {
                outcomes.push(OutcomeResult {
                    description: "prediction".to_string(),
                    passed: false,
                    details: Some(error.to_string()),
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
            .map(|(predicted, ground_truth)| compare_video_clips(predicted, ground_truth))
    };
    if let Some(metrics) = video_metrics.as_ref() {
        scores.insert("video_similarity".to_string(), metrics.overall_similarity);
    }
    ensure_overall_score(&mut scores);

    if custom_metric_requested(dimensions, &scenario.expected_outcomes, "confidence") {
        if let Some(confidence) = average_confidence {
            scores.insert("confidence".to_string(), confidence);
        }
    }

    if !prediction_failed {
        if derived_metric_requested(
            dimensions,
            &scenario.expected_outcomes,
            EvalDimension::ActionPredictionAccuracy,
            outcome_implies_action_prediction_accuracy,
        ) {
            scores.insert(
                "action_prediction_accuracy".to_string(),
                action_prediction_accuracy_score(
                    scenario,
                    &starting_state,
                    &current_state,
                    average_scores.as_ref(),
                ),
            );
        }

        if derived_metric_requested(
            dimensions,
            &scenario.expected_outcomes,
            EvalDimension::SpatialReasoning,
            outcome_implies_spatial_reasoning,
        ) {
            scores.insert(
                "spatial_reasoning".to_string(),
                spatial_reasoning_score(scenario, &current_state, average_scores.as_ref()),
            );
        }

        if derived_metric_requested(
            dimensions,
            &scenario.expected_outcomes,
            EvalDimension::MaterialUnderstanding,
            outcome_implies_material_understanding,
        ) {
            scores.insert(
                "material_understanding".to_string(),
                material_understanding_score(&starting_state, scenario, &current_state),
            );
        }
    }

    if !prediction_failed {
        for expected in &scenario.expected_outcomes {
            outcomes.push(check_outcome(
                expected,
                &current_state,
                &scores,
                average_confidence,
                scenario.ground_truth.as_ref(),
                video_metrics.as_ref(),
            ));
        }
    }

    EvalResult {
        provider: provider.name().to_string(),
        scenario: scenario.name.clone(),
        scores,
        latency_ms: start.elapsed().as_millis() as u64,
        video: predicted_video,
        video_metrics,
        outcomes,
    }
}

fn normalize_provider_names(provider_names: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();

    for provider in provider_names {
        let provider = provider.trim();
        if provider.is_empty() {
            continue;
        }

        if seen.insert(provider.to_string()) {
            normalized.push(provider.to_string());
        }
    }

    normalized
}

fn merge_world_state(base: &WorldState, overlay: &WorldState) -> WorldState {
    let mut merged = base.clone();

    merged.time = overlay.time;
    merged.history = if overlay.history.is_empty() {
        base.history.clone()
    } else {
        overlay.history.clone()
    };
    merged.metadata = merge_world_metadata(&base.metadata, &overlay.metadata);

    for object in sorted_scene_objects(&overlay.scene) {
        if merged.scene.objects.contains_key(&object.id) {
            let _ = merged.scene.replace_object(object.clone());
        } else {
            merged.scene.add_object(object.clone());
        }
    }

    merged
}

fn merge_world_metadata(base: &WorldMetadata, overlay: &WorldMetadata) -> WorldMetadata {
    let mut tags = base.tags.clone();
    for tag in &overlay.tags {
        if !tags.contains(tag) {
            tags.push(tag.clone());
        }
    }

    WorldMetadata {
        name: if overlay.name.is_empty() {
            base.name.clone()
        } else {
            overlay.name.clone()
        },
        description: if overlay.description.is_empty() {
            base.description.clone()
        } else {
            overlay.description.clone()
        },
        created_by: if overlay.created_by.is_empty() {
            base.created_by.clone()
        } else {
            overlay.created_by.clone()
        },
        created_at: overlay.created_at,
        tags,
    }
}

fn sorted_scene_objects(
    scene: &worldforge_core::scene::SceneGraph,
) -> Vec<worldforge_core::scene::SceneObject> {
    let mut objects: Vec<_> = scene.objects.values().cloned().collect();
    objects.sort_by(|left, right| left.id.as_bytes().cmp(right.id.as_bytes()));
    objects
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

fn is_supported_custom_dimension(name: &str) -> bool {
    SUPPORTED_CUSTOM_DIMENSION_NAMES
        .iter()
        .any(|supported| supported.eq_ignore_ascii_case(name))
}

fn custom_metric_requested(
    dimensions: &[EvalDimension],
    expected_outcomes: &[ExpectedOutcome],
    key: &str,
) -> bool {
    dimensions.iter().any(|dimension| {
        matches!(
            dimension,
            EvalDimension::Custom { name } if name.eq_ignore_ascii_case(key)
        )
    }) || expected_outcomes.iter().any(|expected| {
        matches!(
            expected,
            ExpectedOutcome::MinPhysicsScore {
                dimension: EvalDimension::Custom { name },
                ..
            } if name.eq_ignore_ascii_case(key)
        )
    })
}

fn derived_metric_requested(
    dimensions: &[EvalDimension],
    expected_outcomes: &[ExpectedOutcome],
    target: EvalDimension,
    implied_by_outcome: fn(&ExpectedOutcome) -> bool,
) -> bool {
    dimensions.contains(&target)
        || expected_outcomes.iter().any(implied_by_outcome)
        // Threshold checks can require this key, but score derivation itself
        // intentionally ignores MinPhysicsScore to avoid circular logic.
        || expected_outcomes.iter().any(|expected| {
            matches!(
                expected,
                ExpectedOutcome::MinPhysicsScore { dimension, .. } if *dimension == target
            )
        })
}

fn outcome_implies_action_prediction_accuracy(expected: &ExpectedOutcome) -> bool {
    matches!(
        expected,
        ExpectedOutcome::ObjectExists { .. }
            | ExpectedOutcome::ObjectNotExists { .. }
            | ExpectedOutcome::ObjectPosition { .. }
            | ExpectedOutcome::ObjectSemanticLabel { .. }
            | ExpectedOutcome::FinalStateCondition { .. }
    )
}

fn outcome_implies_spatial_reasoning(expected: &ExpectedOutcome) -> bool {
    matches!(
        expected,
        ExpectedOutcome::ObjectPosition { .. } | ExpectedOutcome::FinalStateCondition { .. }
    )
}

fn outcome_implies_material_understanding(expected: &ExpectedOutcome) -> bool {
    matches!(expected, ExpectedOutcome::ObjectSemanticLabel { .. })
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

fn average_components(components: &[f32]) -> Option<f32> {
    (!components.is_empty()).then(|| components.iter().sum::<f32>() / components.len() as f32)
}

fn push_component(components: &mut Vec<f32>, score: Option<f32>) {
    if let Some(score) = score {
        components.push(score.clamp(0.0, 1.0));
    }
}

fn scalar_similarity(actual: f32, target: f32, tolerance: f32) -> f32 {
    let tolerance = tolerance.max(0.001);
    let delta = (actual - target).abs();
    if delta <= tolerance {
        1.0
    } else {
        (1.0 - (delta - tolerance) / (tolerance * 2.0)).clamp(0.0, 1.0)
    }
}

fn position_similarity(actual: Position, target: Position, tolerance: f32) -> f32 {
    scalar_similarity(actual.distance(target), 0.0, tolerance)
}

fn objects_touching(state: &WorldState, a: uuid::Uuid, b: uuid::Uuid) -> bool {
    state.scene.relationships.iter().any(|relationship| {
        matches!(
            relationship,
            worldforge_core::scene::SpatialRelationship::Touching { a: left, b: right }
                if (*left == a && *right == b) || (*left == b && *right == a)
        )
    })
}

fn condition_score(condition: &Condition, state: &WorldState) -> f32 {
    match condition {
        Condition::ObjectAt {
            object,
            position,
            tolerance,
        } => state
            .scene
            .get_object(object)
            .map(|item| position_similarity(item.pose.position, *position, *tolerance))
            .unwrap_or(0.0),
        Condition::ObjectsTouching { a, b } => f32::from(objects_touching(state, *a, *b)),
        Condition::ObjectExists { object } => f32::from(state.scene.get_object(object).is_some()),
        Condition::And(conditions) => average_components(
            &conditions
                .iter()
                .map(|condition| condition_score(condition, state))
                .collect::<Vec<_>>(),
        )
        .unwrap_or(1.0),
        Condition::Or(conditions) => conditions
            .iter()
            .map(|condition| condition_score(condition, state))
            .fold(0.0, f32::max),
        Condition::Not(inner) => 1.0 - condition_score(inner, state),
    }
}

fn object_named<'a>(
    state: &'a WorldState,
    name: &str,
) -> Option<&'a worldforge_core::scene::SceneObject> {
    state
        .scene
        .objects
        .values()
        .find(|object| object.name == name)
}

fn object_exists_score(state: &WorldState, name: &str) -> f32 {
    f32::from(object_named(state, name).is_some())
}

fn object_not_exists_score(state: &WorldState, name: &str) -> f32 {
    1.0 - object_exists_score(state, name)
}

fn object_position_score(
    state: &WorldState,
    name: &str,
    position: Position,
    tolerance: f32,
) -> f32 {
    object_named(state, name)
        .map(|object| position_similarity(object.pose.position, position, tolerance))
        .unwrap_or(0.0)
}

fn object_semantic_label_score(state: &WorldState, name: &str, label: &str) -> f32 {
    object_named(state, name)
        .map(|object| f32::from(object.semantic_label.as_deref() == Some(label)))
        .unwrap_or(0.0)
}

fn non_threshold_outcome_score(expected: &ExpectedOutcome, state: &WorldState) -> Option<f32> {
    match expected {
        ExpectedOutcome::ObjectExists { name } => Some(object_exists_score(state, name)),
        ExpectedOutcome::ObjectNotExists { name } => Some(object_not_exists_score(state, name)),
        ExpectedOutcome::ObjectPosition {
            name,
            position,
            tolerance,
        } => Some(object_position_score(state, name, *position, *tolerance)),
        ExpectedOutcome::ObjectSemanticLabel { name, label } => {
            Some(object_semantic_label_score(state, name, label))
        }
        ExpectedOutcome::FinalStateCondition { condition } => {
            Some(condition_score(condition, state))
        }
        _ => None,
    }
}

fn best_object_position_score(state: &WorldState, target: Position, tolerance: f32) -> Option<f32> {
    state
        .scene
        .objects
        .values()
        .map(|object| position_similarity(object.pose.position, target, tolerance))
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn tag_value<'a>(state: &'a WorldState, prefix: &str) -> Option<&'a str> {
    state
        .metadata
        .tags
        .iter()
        .find_map(|tag| tag.strip_prefix(prefix))
}

fn action_effect_score(
    action: &Action,
    initial_state: &WorldState,
    final_state: &WorldState,
) -> Option<f32> {
    match action {
        Action::Move { target, .. } => best_object_position_score(final_state, *target, 0.2),
        Action::Place { object, target } => final_state
            .scene
            .get_object(object)
            .map(|item| position_similarity(item.pose.position, *target, 0.05)),
        Action::Push {
            object, direction, ..
        } => {
            let initial = initial_state.scene.get_object(object)?;
            let final_object = final_state.scene.get_object(object)?;
            let displacement = (
                final_object.pose.position.x - initial.pose.position.x,
                final_object.pose.position.y - initial.pose.position.y,
                final_object.pose.position.z - initial.pose.position.z,
            );
            let normalized = direction.normalized();
            let projection = displacement.0 * normalized.x
                + displacement.1 * normalized.y
                + displacement.2 * normalized.z;
            Some((projection.max(0.0) / 0.1).clamp(0.0, 1.0))
        }
        Action::Rotate { object, angle, .. } => {
            let initial = initial_state.scene.get_object(object)?;
            let final_object = final_state.scene.get_object(object)?;
            let delta = (final_object.pose.rotation.w - initial.pose.rotation.w).abs()
                + (final_object.pose.rotation.x - initial.pose.rotation.x).abs()
                + (final_object.pose.rotation.y - initial.pose.rotation.y).abs()
                + (final_object.pose.rotation.z - initial.pose.rotation.z).abs();
            Some(if angle.abs() <= f32::EPSILON {
                1.0
            } else {
                (delta / 0.5).clamp(0.0, 1.0)
            })
        }
        Action::Navigate { waypoints } => waypoints
            .last()
            .copied()
            .and_then(|target| best_object_position_score(final_state, target, 0.2)),
        Action::Teleport { destination } => {
            best_object_position_score(final_state, destination.position, 0.05)
        }
        Action::CameraMove { .. } => Some(f32::from(
            tag_value(final_state, "camera-offset:").is_some(),
        )),
        Action::CameraLookAt { .. } => Some(f32::from(
            tag_value(final_state, "camera-look-at:").is_some(),
        )),
        Action::SetWeather { weather } => tag_value(final_state, "weather:")
            .map(|value| f32::from(value.eq_ignore_ascii_case(&format!("{weather:?}")))),
        Action::SetLighting { time_of_day } => tag_value(final_state, "lighting:")
            .and_then(|value| value.parse::<f32>().ok())
            .map(|value| scalar_similarity(value, *time_of_day, 0.5)),
        Action::SpawnObject { template, pose } => Some(
            object_named(final_state, template)
                .map(|object| position_similarity(object.pose.position, pose.position, 0.2))
                .unwrap_or_else(|| {
                    final_state
                        .scene
                        .objects
                        .values()
                        .find(|object| object.semantic_label.as_deref() == Some(template.as_str()))
                        .map(|object| position_similarity(object.pose.position, pose.position, 0.2))
                        .unwrap_or(0.0)
                }),
        ),
        Action::RemoveObject { object } => {
            Some(f32::from(final_state.scene.get_object(object).is_none()))
        }
        Action::Grasp { object, .. } => Some(f32::from(
            final_state
                .metadata
                .tags
                .iter()
                .any(|tag| tag == &format!("grasped:{object}")),
        )),
        Action::Release { object } => Some(f32::from(
            !final_state
                .metadata
                .tags
                .iter()
                .any(|tag| tag == &format!("grasped:{object}")),
        )),
        Action::Sequence(actions) | Action::Parallel(actions) => average_components(
            &actions
                .iter()
                .filter_map(|action| action_effect_score(action, initial_state, final_state))
                .collect::<Vec<_>>(),
        ),
        Action::Conditional {
            condition,
            then,
            otherwise,
        } => {
            if evaluate_condition(condition, initial_state) {
                action_effect_score(then, initial_state, final_state)
            } else {
                otherwise
                    .as_deref()
                    .and_then(|action| action_effect_score(action, initial_state, final_state))
                    .or(Some(1.0))
            }
        }
        Action::Raw { provider, .. } => {
            tag_value(final_state, "raw-provider:").map(|value| f32::from(value == provider))
        }
    }
}

fn action_prediction_accuracy_score(
    scenario: &EvalScenario,
    starting_state: &WorldState,
    final_state: &WorldState,
    average_scores: Option<&PhysicsScores>,
) -> f32 {
    let outcome_scores = scenario
        .expected_outcomes
        .iter()
        .filter_map(|expected| match expected {
            ExpectedOutcome::ObjectExists { .. }
            | ExpectedOutcome::ObjectNotExists { .. }
            | ExpectedOutcome::ObjectPosition { .. }
            | ExpectedOutcome::ObjectSemanticLabel { .. }
            | ExpectedOutcome::FinalStateCondition { .. } => {
                non_threshold_outcome_score(expected, final_state)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let action_scores = scenario
        .actions
        .iter()
        .filter_map(|action| action_effect_score(action, starting_state, final_state))
        .collect::<Vec<_>>();

    if let Some(outcome_score) = average_components(&outcome_scores) {
        return outcome_score;
    }

    if let Some(action_score) = average_components(&action_scores) {
        return action_score;
    }

    average_scores.map_or(0.5, |scores| scores.overall)
}

fn range_overlap_ratio(a_min: f32, a_max: f32, b_min: f32, b_max: f32) -> f32 {
    let overlap = (a_max.min(b_max) - a_min.max(b_min)).max(0.0);
    let baseline = (a_max - a_min).min(b_max - b_min).max(0.001);
    (overlap / baseline).clamp(0.0, 1.0)
}

fn spatial_relationship_score(
    state: &WorldState,
    relationship: &worldforge_core::scene::SpatialRelationship,
) -> Option<f32> {
    use worldforge_core::scene::SpatialRelationship;

    match relationship {
        SpatialRelationship::Touching { a, b } => Some(f32::from(objects_touching(state, *a, *b))),
        SpatialRelationship::Near { a, b, distance } => {
            let left = state.scene.get_object(a)?;
            let right = state.scene.get_object(b)?;
            Some(scalar_similarity(
                left.pose.position.distance(right.pose.position),
                *distance,
                distance.max(0.05),
            ))
        }
        SpatialRelationship::Above { subject, reference } => {
            let subject = state.scene.get_object(subject)?;
            let reference = state.scene.get_object(reference)?;
            let delta = subject.pose.position.y - reference.pose.position.y;
            Some(if delta >= 0.0 {
                1.0
            } else {
                (1.0 + (delta / 0.5)).clamp(0.0, 1.0)
            })
        }
        SpatialRelationship::Below { subject, reference } => {
            let subject = state.scene.get_object(subject)?;
            let reference = state.scene.get_object(reference)?;
            let delta = reference.pose.position.y - subject.pose.position.y;
            Some(if delta >= 0.0 {
                1.0
            } else {
                (1.0 + (delta / 0.5)).clamp(0.0, 1.0)
            })
        }
        SpatialRelationship::On { subject, surface } => {
            let subject = state.scene.get_object(subject)?;
            let surface = state.scene.get_object(surface)?;
            let vertical = scalar_similarity(subject.bbox.min.y, surface.bbox.max.y, 0.1);
            let overlap_x = range_overlap_ratio(
                subject.bbox.min.x,
                subject.bbox.max.x,
                surface.bbox.min.x,
                surface.bbox.max.x,
            );
            let overlap_z = range_overlap_ratio(
                subject.bbox.min.z,
                subject.bbox.max.z,
                surface.bbox.min.z,
                surface.bbox.max.z,
            );
            Some(((vertical + overlap_x + overlap_z) / 3.0).clamp(0.0, 1.0))
        }
        SpatialRelationship::In { subject, container } => {
            let subject = state.scene.get_object(subject)?;
            let container = state.scene.get_object(container)?;
            Some(if container.bbox.contains(&subject.bbox) {
                1.0
            } else if container.bbox.intersects_or_touches(&subject.bbox) {
                0.5
            } else {
                0.0
            })
        }
    }
}

fn spatial_relationship_signal_score(state: &WorldState) -> Option<f32> {
    average_components(
        &state
            .scene
            .relationships
            .iter()
            .filter_map(|relationship| spatial_relationship_score(state, relationship))
            .collect::<Vec<_>>(),
    )
}

fn spatial_reasoning_score(
    scenario: &EvalScenario,
    final_state: &WorldState,
    average_scores: Option<&PhysicsScores>,
) -> f32 {
    let mut components = scenario
        .expected_outcomes
        .iter()
        .filter_map(|expected| match expected {
            ExpectedOutcome::ObjectPosition { .. }
            | ExpectedOutcome::FinalStateCondition { .. } => {
                non_threshold_outcome_score(expected, final_state)
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    push_component(
        &mut components,
        average_scores.map(|scores| scores.spatial_consistency),
    );
    push_component(
        &mut components,
        spatial_relationship_signal_score(final_state),
    );

    average_components(&components).unwrap_or(0.5)
}

fn material_understanding_score(
    starting_state: &WorldState,
    scenario: &EvalScenario,
    final_state: &WorldState,
) -> f32 {
    let mut components = Vec::new();

    for expected in &scenario.expected_outcomes {
        if let ExpectedOutcome::ObjectSemanticLabel { .. } = expected {
            push_component(
                &mut components,
                non_threshold_outcome_score(expected, final_state),
            );
        }
    }

    for initial_object in starting_state.scene.objects.values() {
        let Some(final_object) = final_state.scene.get_object(&initial_object.id) else {
            continue;
        };

        if initial_object.semantic_label.is_some() {
            components.push(f32::from(
                final_object.semantic_label == initial_object.semantic_label,
            ));
        }
        if initial_object.physics.material.is_some() {
            components.push(f32::from(
                final_object.physics.material == initial_object.physics.material,
            ));
        }
    }

    average_components(&components).unwrap_or(0.5)
}

fn check_outcome(
    expected: &ExpectedOutcome,
    state: &WorldState,
    scores: &HashMap<String, f32>,
    average_confidence: Option<f32>,
    ground_truth: Option<&VideoClip>,
    video_metrics: Option<&VideoMetrics>,
) -> OutcomeResult {
    match expected {
        ExpectedOutcome::MinPhysicsScore {
            dimension,
            threshold,
        } => match scores.get(&dimension.key()) {
            Some(score) => OutcomeResult {
                description: format!("{} >= {threshold}", dimension.key()),
                passed: *score >= *threshold,
                details: Some(format!("score: {score:.3}")),
            },
            None => OutcomeResult {
                description: format!("{} >= {threshold}", dimension.key()),
                passed: false,
                details: Some("score unavailable for this scenario".to_string()),
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
        TensorData::Float16(values) => values
            .iter()
            .map(|value| half_bits_to_f32(*value) as f64)
            .collect(),
        TensorData::BFloat16(values) => values
            .iter()
            .map(|value| bfloat16_bits_to_f32(*value) as f64)
            .collect(),
        TensorData::UInt8(values) => values.iter().map(|value| *value as f64).collect(),
        TensorData::Int32(values) => values.iter().map(|value| *value as f64).collect(),
        TensorData::Int64(values) => values.iter().map(|value| *value as f64).collect(),
    }
}

fn half_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits & 0x7c00) >> 10) as u32;
    let mantissa = (bits & 0x03ff) as u32;

    let f32_bits = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                let mut mantissa = mantissa;
                let mut exponent = -14i32;
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | (((exponent as i32 - 15 + 127) as u32) << 23) | (mantissa << 13),
    };

    f32::from_bits(f32_bits)
}

fn bfloat16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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

fn format_percent(value: f32) -> String {
    format!("{:.1}%", value * 100.0)
}

fn format_optional_score(value: Option<f32>) -> String {
    value
        .map(|score| format!("{score:.3}"))
        .unwrap_or_else(|| "-".to_string())
}

fn markdown_optional(value: Option<&str>) -> String {
    value.map(markdown_cell).unwrap_or_else(|| "-".to_string())
}

fn markdown_joined_list(values: &[String]) -> String {
    if values.is_empty() {
        "-".to_string()
    } else {
        markdown_cell(&values.join(", "))
    }
}

fn markdown_cell(value: &str) -> String {
    if value.is_empty() {
        "-".to_string()
    } else {
        value.replace('|', "\\|").replace('\n', "<br>")
    }
}

fn provider_score_list(provider_scores: &HashMap<String, f32>) -> String {
    if provider_scores.is_empty() {
        return "-".to_string();
    }

    let mut entries: Vec<_> = provider_scores.iter().collect();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    entries
        .into_iter()
        .map(|(provider, score)| format!("{provider}: {score:.3}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn report_score_keys(results: &[EvalResult]) -> Vec<String> {
    let mut keys: Vec<_> = results
        .iter()
        .flat_map(|result| result.scores.keys().cloned())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    keys.sort();
    keys
}

fn csv_cell(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
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

    #[derive(Debug, Default)]
    struct ConcurrencyTracker {
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    impl ConcurrencyTracker {
        fn enter(&self) {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
        }

        fn exit(&self) {
            self.active.fetch_sub(1, Ordering::SeqCst);
        }

        fn max_active(&self) -> usize {
            self.max_active.load(Ordering::SeqCst)
        }
    }

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

    #[derive(Debug)]
    struct DelayedEvalProvider {
        name: String,
        delay_ms: u64,
        tracker: Arc<ConcurrencyTracker>,
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

    impl DelayedEvalProvider {
        fn new(name: &str, delay_ms: u64, tracker: Arc<ConcurrencyTracker>) -> Self {
            Self {
                name: name.to_string(),
                delay_ms,
                tracker,
            }
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

    #[test]
    fn test_tensor_values_supports_half_precision() {
        let half_values = TensorData::Float16(vec![0x3c00, 0xc000, 0x3800]);
        let bfloat16_values = TensorData::BFloat16(vec![0x3f80, 0xc000, 0x3f00]);

        assert_eq!(tensor_values(&half_values), vec![1.0, -2.0, 0.5]);
        assert_eq!(tensor_values(&bfloat16_values), vec![1.0, -2.0, 0.5]);
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
                embed: false,
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
                embed: false,
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

    #[async_trait]
    impl WorldModelProvider for DelayedEvalProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                predict: true,
                generate: false,
                reason: false,
                transfer: false,
                embed: false,
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
            self.tracker.enter();
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            self.tracker.exit();

            let mut output_state = state.clone();
            output_state.time.step += 1;

            Ok(Prediction {
                id: uuid::Uuid::new_v4(),
                provider: self.name.clone(),
                model: "delayed-eval".to_string(),
                input_state: state.clone(),
                action: action.clone(),
                output_state,
                video: None,
                confidence: 0.9,
                physics_scores: PhysicsScores {
                    overall: 0.9,
                    object_permanence: 0.9,
                    gravity_compliance: 0.9,
                    collision_accuracy: 0.9,
                    spatial_consistency: 0.9,
                    temporal_consistency: 0.9,
                },
                latency_ms: self.delay_ms,
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
                latency_ms: self.delay_ms,
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

    #[test]
    fn test_eval_report_format_parsing_supports_known_variants() {
        assert_eq!(
            "json".parse::<EvalReportFormat>().unwrap(),
            EvalReportFormat::Json
        );
        assert_eq!(
            "markdown".parse::<EvalReportFormat>().unwrap(),
            EvalReportFormat::Markdown
        );
        assert_eq!(
            "md".parse::<EvalReportFormat>().unwrap(),
            EvalReportFormat::Markdown
        );
        assert_eq!(
            "csv".parse::<EvalReportFormat>().unwrap(),
            EvalReportFormat::Csv
        );
        assert!("yaml".parse::<EvalReportFormat>().is_err());
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

    #[tokio::test]
    async fn test_eval_report_renderers_are_roundtrip_and_deterministic() {
        let suite = EvalSuite::physics_standard();
        let provider = MockProvider::new();
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];
        let report = suite.run(&providers).await.unwrap();

        let json = report.to_json_pretty().unwrap();
        let roundtrip = EvalReport::from_json_str(&json).unwrap();
        assert_eq!(roundtrip.suite, report.suite);
        assert_eq!(roundtrip.results.len(), report.results.len());

        let markdown = report.to_markdown().unwrap();
        assert!(markdown.contains("# Evaluation Report: Physics Standard"));
        assert!(markdown.contains("## Leaderboard"));
        assert!(markdown.contains("## Provider Summaries"));
        assert!(markdown.contains("## Scenario Summaries"));

        let csv = report.to_csv().unwrap();
        let lines: Vec<_> = csv.lines().collect();
        assert_eq!(lines.len(), report.results.len() + 1);
        assert!(lines[0].contains("suite,provider,scenario,overall_score"));
        assert!(lines[0].contains("object_permanence"));
        assert!(csv.contains("Physics Standard,mock,object_drop"));
    }

    #[tokio::test]
    async fn test_eval_report_render_many_deduplicates_and_preserves_order() {
        let suite = EvalSuite::physics_standard();
        let provider = MockProvider::new();
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];
        let report = suite.run(&providers).await.unwrap();

        let rendered = report
            .render_many([
                EvalReportFormat::Markdown,
                EvalReportFormat::Csv,
                EvalReportFormat::Markdown,
                EvalReportFormat::Json,
            ])
            .unwrap();

        assert_eq!(rendered.len(), 3);
        assert_eq!(rendered[0].0, EvalReportFormat::Markdown);
        assert!(rendered[0]
            .1
            .contains("# Evaluation Report: Physics Standard"));
        assert_eq!(rendered[1].0, EvalReportFormat::Csv);
        assert!(rendered[1].1.contains("suite,provider,scenario"));
        assert_eq!(rendered[2].0, EvalReportFormat::Json);
        assert!(rendered[2].1.contains("\"suite\": \"Physics Standard\""));
    }

    #[test]
    fn test_eval_report_csv_escapes_text_fields() {
        let report = EvalReport {
            suite: "Physics, Standard".to_string(),
            results: vec![EvalResult {
                provider: "mock".to_string(),
                scenario: "drop \"mug\"".to_string(),
                scores: HashMap::from([(String::from("overall"), 0.95)]),
                latency_ms: 12,
                video: None,
                video_metrics: None,
                outcomes: vec![OutcomeResult {
                    description: "prediction".to_string(),
                    passed: true,
                    details: None,
                }],
            }],
            leaderboard: Vec::new(),
            provider_summaries: Vec::new(),
            dimension_summaries: Vec::new(),
            scenario_summaries: Vec::new(),
            outcomes_passed: 1,
            total_outcomes: 1,
        };

        let csv = report.to_csv().unwrap();
        assert!(csv.contains("\"Physics, Standard\""));
        assert!(csv.contains("\"drop \"\"mug\"\"\""));
    }

    #[tokio::test]
    async fn test_eval_suite_run_with_world_state_overlays_scenario_fixture() {
        use worldforge_core::scene::SceneObject;
        use worldforge_core::types::{BBox, Pose, Position};

        let object_id = uuid::Uuid::new_v4();
        let mut scenario_state = WorldState::new("overlay_fixture", "eval");
        let mut fixture_object = SceneObject::new(
            "cube",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: -0.1,
                    y: -0.1,
                    z: -0.1,
                },
                max: Position {
                    x: 0.1,
                    y: 0.1,
                    z: 0.1,
                },
            },
        );
        fixture_object.id = object_id;
        scenario_state.scene.add_object(fixture_object);

        let suite = EvalSuite {
            name: "World Overlay".to_string(),
            scenarios: vec![EvalScenario {
                name: "object_position".to_string(),
                description: "Check whether the cube remains at the fixture position".to_string(),
                initial_state: scenario_state.clone(),
                actions: Vec::new(),
                expected_outcomes: vec![ExpectedOutcome::ObjectPosition {
                    name: "cube".to_string(),
                    position: Position {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    tolerance: 0.05,
                }],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::SpatialConsistency],
            providers: vec![],
        };
        let provider = MockProvider::new();
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];

        let baseline = suite.run(&providers).await.unwrap();
        assert!(baseline.results[0].outcomes[0].passed);

        let mut overlay_state = scenario_state;
        overlay_state.scene.set_object_position(
            &object_id,
            Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        );

        let world_aware = suite
            .run_with_world_state(&providers, &overlay_state)
            .await
            .unwrap();
        assert!(!world_aware.results[0].outcomes[0].passed);
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
        assert!(suite
            .dimensions
            .contains(&EvalDimension::MaterialUnderstanding));
    }

    #[tokio::test]
    async fn test_manipulation_suite_run() {
        let suite = EvalSuite::manipulation_standard();
        let provider = MockProvider::new();
        let providers: Vec<&dyn WorldModelProvider> = vec![&provider];
        let report = suite.run(&providers).await.unwrap();
        assert_eq!(report.results.len(), 3);
        assert_eq!(report.leaderboard[0].provider, "mock");
        let action_summary = report
            .dimension_summaries
            .iter()
            .find(|summary| summary.dimension == "action_prediction_accuracy")
            .unwrap();
        assert!(!action_summary.provider_scores.is_empty());
        let material_summary = report
            .dimension_summaries
            .iter()
            .find(|summary| summary.dimension == "material_understanding")
            .unwrap();
        assert!(!material_summary.provider_scores.is_empty());
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
        let spatial_summary = report
            .dimension_summaries
            .iter()
            .find(|summary| summary.dimension == "spatial_reasoning")
            .unwrap();
        assert!(!spatial_summary.provider_scores.is_empty());
    }

    #[test]
    fn test_comprehensive_suite_creation() {
        let suite = EvalSuite::comprehensive();
        assert_eq!(suite.name, "Comprehensive");
        // 2 physics + 3 manipulation + 2 spatial = 7
        assert_eq!(suite.scenarios.len(), 7);
        assert_eq!(suite.dimensions.len(), 8);
        assert!(suite
            .dimensions
            .contains(&EvalDimension::MaterialUnderstanding));
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
        assert_eq!(restored.providers, vec!["mock".to_string()]);
    }

    #[test]
    fn test_effective_provider_names_prefers_explicit_then_suite_defaults() {
        let mut suite = EvalSuite::physics_standard();
        suite.providers = vec!["suite-a".to_string(), "suite-b".to_string()];

        assert_eq!(
            suite.effective_provider_names(&[]),
            vec!["suite-a".to_string(), "suite-b".to_string()]
        );
        assert_eq!(
            suite.effective_provider_names(&[
                " explicit ".to_string(),
                "explicit".to_string(),
                "".to_string(),
            ]),
            vec!["explicit".to_string()]
        );
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
            providers: vec![],
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
            providers: vec![],
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
            providers: vec![],
        };
        let provider = MockProvider::new();
        let report = suite
            .run(&[&provider as &dyn WorldModelProvider])
            .await
            .unwrap();
        let result = &report.results[0];

        assert!(result.outcomes.iter().all(|outcome| outcome.passed));
        assert!(result.scores["action_prediction_accuracy"] >= 0.75);
    }

    #[tokio::test]
    async fn test_outcomes_imply_derived_scores_without_explicit_dimensions() {
        let target = Position {
            x: 1.0,
            y: 0.5,
            z: -0.2,
        };
        let suite = EvalSuite {
            name: "Implied Derived Scores".to_string(),
            scenarios: vec![EvalScenario {
                name: "implied_scores".to_string(),
                description: "Outcomes should surface action/spatial/material derived scores"
                    .to_string(),
                initial_state: visual_fixture_state(
                    Position {
                        x: 0.0,
                        y: 0.5,
                        z: -0.2,
                    },
                    Some("mug"),
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
                ],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::SpatialConsistency],
            providers: vec![],
        };
        let provider = VisualFixtureProvider::new(
            "implied-visual",
            visual_fixture_state(target, Some("mug")),
            visual_fixture_clip(),
        );

        let report = suite
            .run(&[&provider as &dyn WorldModelProvider])
            .await
            .unwrap();
        let result = &report.results[0];

        assert!(result.scores.contains_key("action_prediction_accuracy"));
        assert!(result.scores.contains_key("spatial_reasoning"));
        assert!(result.scores.contains_key("material_understanding"));
    }

    #[tokio::test]
    async fn test_action_prediction_uses_action_effect_fallback_without_relevant_outcomes() {
        let mut output_state = visual_fixture_state(Position::default(), Some("mug"));
        output_state
            .metadata
            .tags
            .push("camera-look-at:0.0,0.5,-2.0".to_string());

        let suite = EvalSuite {
            name: "Action Fallback".to_string(),
            scenarios: vec![EvalScenario {
                name: "camera_look".to_string(),
                description: "When outcomes do not describe final state, use direct action effect"
                    .to_string(),
                initial_state: visual_fixture_state(Position::default(), Some("mug")),
                actions: vec![Action::CameraLookAt {
                    target: Position {
                        x: 0.0,
                        y: 0.5,
                        z: -2.0,
                    },
                }],
                expected_outcomes: vec![],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::ActionPredictionAccuracy],
            providers: vec![],
        };
        let provider =
            VisualFixtureProvider::new("action-fallback", output_state, visual_fixture_clip());

        let report = suite
            .run(&[&provider as &dyn WorldModelProvider])
            .await
            .unwrap();
        assert_eq!(report.results[0].scores["action_prediction_accuracy"], 1.0);
    }

    #[tokio::test]
    async fn test_threshold_request_for_action_prediction_is_not_circular() {
        let suite = EvalSuite {
            name: "Non Circular Threshold".to_string(),
            scenarios: vec![EvalScenario {
                name: "derived_threshold".to_string(),
                description: "Derived thresholds should not influence their own score".to_string(),
                initial_state: WorldState::new("non_circular", "eval"),
                actions: vec![Action::SetLighting { time_of_day: 0.5 }],
                expected_outcomes: vec![ExpectedOutcome::MinPhysicsScore {
                    dimension: EvalDimension::ActionPredictionAccuracy,
                    threshold: 0.8,
                }],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::SpatialConsistency],
            providers: vec![],
        };
        let provider = SequencedEvalProvider::new(
            "non-circular",
            vec![(
                PhysicsScores {
                    overall: 0.25,
                    object_permanence: 0.25,
                    gravity_compliance: 0.25,
                    collision_accuracy: 0.25,
                    spatial_consistency: 0.25,
                    temporal_consistency: 0.25,
                },
                0.9,
            )],
        );

        let report = suite
            .run(&[&provider as &dyn WorldModelProvider])
            .await
            .unwrap();
        let result = &report.results[0];
        assert_eq!(result.scores["action_prediction_accuracy"], 0.25);
        assert!(!result.outcomes[0].passed);
    }

    #[test]
    fn test_spatial_relationship_signal_scores_scene_relationships() {
        use worldforge_core::scene::SceneObject;

        let mut state = WorldState::new("spatial_signal", "eval");
        let object_a = SceneObject::new(
            "a",
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
        let object_b = SceneObject::new(
            "b",
            Pose {
                position: Position {
                    x: 0.12,
                    y: 0.5,
                    z: 0.0,
                },
                ..Default::default()
            },
            BBox {
                min: Position {
                    x: 0.02,
                    y: 0.4,
                    z: -0.1,
                },
                max: Position {
                    x: 0.22,
                    y: 0.6,
                    z: 0.1,
                },
            },
        );
        state.scene.add_object(object_a);
        state.scene.add_object(object_b);

        let score = spatial_relationship_signal_score(&state).unwrap();
        assert!(score > 0.5);
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
            providers: vec![],
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
            providers: vec![],
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
            providers: vec![],
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

    #[tokio::test]
    async fn test_eval_run_executes_provider_scenarios_concurrently_and_preserves_order() {
        let tracker = Arc::new(ConcurrencyTracker::default());
        let suite = EvalSuite {
            name: "Concurrent Eval".to_string(),
            scenarios: vec![
                EvalScenario {
                    name: "first".to_string(),
                    description: "first scenario".to_string(),
                    initial_state: WorldState::new("first", "eval"),
                    actions: vec![Action::SetLighting { time_of_day: 0.25 }],
                    expected_outcomes: vec![ExpectedOutcome::MinPhysicsScore {
                        dimension: EvalDimension::ObjectPermanence,
                        threshold: 0.5,
                    }],
                    ground_truth: None,
                },
                EvalScenario {
                    name: "second".to_string(),
                    description: "second scenario".to_string(),
                    initial_state: WorldState::new("second", "eval"),
                    actions: vec![Action::SetLighting { time_of_day: 0.75 }],
                    expected_outcomes: vec![ExpectedOutcome::MinPhysicsScore {
                        dimension: EvalDimension::ObjectPermanence,
                        threshold: 0.5,
                    }],
                    ground_truth: None,
                },
            ],
            dimensions: vec![EvalDimension::ObjectPermanence],
            providers: vec![],
        };
        let alpha = DelayedEvalProvider::new("alpha", 25, Arc::clone(&tracker));
        let beta = DelayedEvalProvider::new("beta", 25, Arc::clone(&tracker));

        let report = suite
            .run(&[
                &alpha as &dyn WorldModelProvider,
                &beta as &dyn WorldModelProvider,
            ])
            .await
            .unwrap();
        let ordering: Vec<_> = report
            .results
            .iter()
            .map(|result| format!("{}:{}", result.provider, result.scenario))
            .collect();

        assert_eq!(
            ordering,
            vec![
                "alpha:first".to_string(),
                "alpha:second".to_string(),
                "beta:first".to_string(),
                "beta:second".to_string(),
            ]
        );
        assert!(report
            .results
            .iter()
            .all(|result| result.outcomes.iter().all(|outcome| outcome.passed)));
        assert!(tracker.max_active() >= 2);
    }

    #[tokio::test]
    async fn test_custom_confidence_dimension_is_reported_and_drives_threshold_outcomes() {
        let suite = EvalSuite {
            name: "Custom Confidence".to_string(),
            scenarios: vec![EvalScenario {
                name: "confidence_threshold".to_string(),
                description: "Confidence should be exposed as a named custom metric".to_string(),
                initial_state: WorldState::new("confidence_threshold", "eval"),
                actions: vec![Action::SetLighting { time_of_day: 0.5 }],
                expected_outcomes: vec![ExpectedOutcome::MinPhysicsScore {
                    dimension: EvalDimension::Custom {
                        name: "confidence".to_string(),
                    },
                    threshold: 0.8,
                }],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::Custom {
                name: "confidence".to_string(),
            }],
            providers: vec![],
        };
        let provider = SequencedEvalProvider::new(
            "sequenced",
            vec![(
                PhysicsScores {
                    overall: 0.5,
                    object_permanence: 0.5,
                    gravity_compliance: 0.5,
                    collision_accuracy: 0.5,
                    spatial_consistency: 0.5,
                    temporal_consistency: 0.5,
                },
                0.85,
            )],
        );

        let report = suite
            .run(&[&provider as &dyn WorldModelProvider])
            .await
            .unwrap();
        let result = &report.results[0];

        assert_eq!(result.scores["confidence"], 0.85);
        assert!(result.outcomes.iter().all(|outcome| outcome.passed));
        assert_eq!(report.dimension_summaries[0].dimension, "confidence");
        assert_eq!(
            report.dimension_summaries[0].best_provider.as_deref(),
            Some("sequenced")
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
            providers: vec![],
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
            providers: vec![],
        };

        assert!(suite.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_unknown_custom_dimension() {
        let suite = EvalSuite {
            name: "custom-bad".to_string(),
            scenarios: vec![EvalScenario {
                name: "unsupported".to_string(),
                description: "Reject unsupported custom metrics".to_string(),
                initial_state: WorldState::new("unsupported", "eval"),
                actions: vec![Action::SetLighting { time_of_day: 0.5 }],
                expected_outcomes: vec![],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::Custom {
                name: "latency".to_string(),
            }],
            providers: vec![],
        };

        let error = suite.validate().unwrap_err().to_string();
        assert!(error.contains("unsupported custom evaluation dimension 'latency'"));
    }

    #[test]
    fn test_validate_rejects_custom_video_similarity_dimension_without_ground_truth() {
        let suite = EvalSuite {
            name: "video-dimension-bad".to_string(),
            scenarios: vec![EvalScenario {
                name: "needs_video".to_string(),
                description: "Custom video similarity dimensions require references".to_string(),
                initial_state: WorldState::new("needs_video", "eval"),
                actions: vec![Action::SetLighting { time_of_day: 0.5 }],
                expected_outcomes: vec![],
                ground_truth: None,
            }],
            dimensions: vec![EvalDimension::Custom {
                name: "video_similarity".to_string(),
            }],
            providers: vec![],
        };

        let error = suite.validate().unwrap_err().to_string();
        assert!(error.contains("custom dimension 'video_similarity' requires ground truth video"));
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
            providers: vec![],
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
