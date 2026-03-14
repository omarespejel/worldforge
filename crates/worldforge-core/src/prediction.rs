//! Prediction engine for WorldForge.
//!
//! Handles forward prediction of world states, multi-provider comparison,
//! and planning through world models.

use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::guardrail::GuardrailResult;
use crate::provider::CostEstimate;
use crate::state::WorldState;
use crate::types::{PredictionId, VideoClip};

/// Result of a single forward prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Unique identifier for this prediction.
    pub id: PredictionId,
    /// Provider that generated this prediction.
    pub provider: String,
    /// Model identifier used.
    pub model: String,
    /// Input world state.
    pub input_state: WorldState,
    /// Action that was applied.
    pub action: Action,
    /// Predicted output world state.
    pub output_state: WorldState,
    /// Generated video of the transition (if requested).
    pub video: Option<VideoClip>,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Physics plausibility scores.
    pub physics_scores: PhysicsScores,
    /// Latency of the prediction in milliseconds.
    pub latency_ms: u64,
    /// Cost of the prediction.
    pub cost: CostEstimate,
    /// Guardrail evaluation results.
    pub guardrail_results: Vec<GuardrailResult>,
    /// When the prediction was generated.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Physics plausibility scores for a prediction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhysicsScores {
    /// Overall physics plausibility (0.0–1.0).
    pub overall: f32,
    /// Object permanence score.
    pub object_permanence: f32,
    /// Gravity compliance score.
    pub gravity_compliance: f32,
    /// Collision accuracy score.
    pub collision_accuracy: f32,
    /// Spatial consistency score.
    pub spatial_consistency: f32,
    /// Temporal consistency score.
    pub temporal_consistency: f32,
}

/// Configuration for a prediction request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Number of future steps to predict.
    pub steps: u32,
    /// Output resolution `(width, height)`.
    pub resolution: (u32, u32),
    /// Output frames per second.
    pub fps: f32,
    /// Whether to return the generated video.
    pub return_video: bool,
    /// Whether to return depth maps.
    pub return_depth: bool,
    /// Whether to return segmentation maps.
    pub return_segmentation: bool,
    /// Guardrail configurations to apply.
    pub guardrails: Vec<crate::guardrail::GuardrailConfig>,
    /// Maximum latency before timeout (in milliseconds).
    pub max_latency_ms: Option<u64>,
    /// Fallback provider if primary fails.
    pub fallback_provider: Option<String>,
    /// Number of samples for uncertainty estimation.
    pub num_samples: u32,
    /// Sampling temperature.
    pub temperature: f32,
}

/// Result of multi-provider prediction comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPrediction {
    /// Individual predictions from each provider.
    pub predictions: Vec<Prediction>,
    /// Agreement score between providers (0.0–1.0).
    pub agreement_score: f32,
    /// Index of the highest-quality prediction.
    pub best_prediction: usize,
    /// Detailed comparison report.
    pub comparison: ComparisonReport,
}

/// Comparison report across multiple provider predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Per-provider scores.
    pub scores: Vec<ProviderScore>,
    /// Summary text.
    pub summary: String,
}

/// Score for a single provider in a comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderScore {
    /// Provider name.
    pub provider: String,
    /// Physics scores.
    pub physics_scores: PhysicsScores,
    /// Latency in milliseconds.
    pub latency_ms: u64,
    /// Cost estimate.
    pub cost: CostEstimate,
}

// ---------------------------------------------------------------------------
// Planning types
// ---------------------------------------------------------------------------

/// Request for planning a sequence of actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    /// Current world state.
    pub current_state: WorldState,
    /// Goal to achieve.
    pub goal: PlanGoal,
    /// Maximum number of planning steps.
    pub max_steps: u32,
    /// Guardrails to enforce during planning.
    pub guardrails: Vec<crate::guardrail::GuardrailConfig>,
    /// Planning algorithm to use.
    pub planner: PlannerType,
    /// Maximum planning time in seconds.
    pub timeout_seconds: f64,
}

/// Goal specification for planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanGoal {
    /// A condition that must be satisfied.
    Condition(crate::action::Condition),
    /// A target world state to reach.
    TargetState(Box<WorldState>),
    /// An image depicting the goal state.
    GoalImage(crate::types::Tensor),
    /// A natural language description of the goal.
    Description(String),
}

/// Planning algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlannerType {
    /// Gradient-based optimization.
    Gradient {
        learning_rate: f32,
        num_iterations: u32,
    },
    /// Random sampling.
    Sampling { num_samples: u32, top_k: u32 },
    /// Cross-entropy method.
    CEM {
        population_size: u32,
        elite_fraction: f32,
        num_iterations: u32,
    },
    /// Model predictive control.
    MPC {
        horizon: u32,
        num_samples: u32,
        replanning_interval: u32,
    },
    /// Use the provider's native planner.
    ProviderNative,
}

/// Result of a planning operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Planned action sequence.
    pub actions: Vec<Action>,
    /// Predicted world states at each step.
    pub predicted_states: Vec<WorldState>,
    /// Predicted videos for each step (if requested).
    pub predicted_videos: Option<Vec<VideoClip>>,
    /// Total estimated cost.
    pub total_cost: f32,
    /// Probability of success (0.0–1.0).
    pub success_probability: f32,
    /// Guardrail compliance at each step.
    pub guardrail_compliance: Vec<Vec<GuardrailResult>>,
    /// Time spent planning in milliseconds.
    pub planning_time_ms: u64,
    /// Number of iterations used.
    pub iterations_used: u32,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            steps: 1,
            resolution: (640, 480),
            fps: 24.0,
            return_video: false,
            return_depth: false,
            return_segmentation: false,
            guardrails: Vec::new(),
            max_latency_ms: None,
            fallback_provider: None,
            num_samples: 1,
            temperature: 1.0,
        }
    }
}

impl Default for PhysicsScores {
    fn default() -> Self {
        Self {
            overall: 0.0,
            object_permanence: 0.0,
            gravity_compliance: 0.0,
            collision_accuracy: 0.0,
            spatial_consistency: 0.0,
            temporal_consistency: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_config_default() {
        let config = PredictionConfig::default();
        assert_eq!(config.steps, 1);
        assert_eq!(config.resolution, (640, 480));
        assert!(!config.return_video);
    }

    #[test]
    fn test_physics_scores_default() {
        let scores = PhysicsScores::default();
        assert_eq!(scores.overall, 0.0);
    }

    #[test]
    fn test_planner_type_serialization() {
        let planner = PlannerType::CEM {
            population_size: 100,
            elite_fraction: 0.1,
            num_iterations: 50,
        };
        let json = serde_json::to_string(&planner).unwrap();
        let planner2: PlannerType = serde_json::from_str(&json).unwrap();
        match planner2 {
            PlannerType::CEM {
                population_size, ..
            } => assert_eq!(population_size, 100),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_plan_goal_description() {
        let goal = PlanGoal::Description("stack the blocks".to_string());
        let json = serde_json::to_string(&goal).unwrap();
        let goal2: PlanGoal = serde_json::from_str(&json).unwrap();
        match goal2 {
            PlanGoal::Description(s) => assert_eq!(s, "stack the blocks"),
            _ => panic!("wrong variant"),
        }
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_physics_scores() -> impl Strategy<Value = PhysicsScores> {
            (
                0.0f32..=1.0,
                0.0f32..=1.0,
                0.0f32..=1.0,
                0.0f32..=1.0,
                0.0f32..=1.0,
                0.0f32..=1.0,
            )
                .prop_map(|(overall, op, gc, ca, sc, tc)| PhysicsScores {
                    overall,
                    object_permanence: op,
                    gravity_compliance: gc,
                    collision_accuracy: ca,
                    spatial_consistency: sc,
                    temporal_consistency: tc,
                })
        }

        fn arb_planner_type() -> impl Strategy<Value = PlannerType> {
            prop_oneof![
                (
                    any::<f32>().prop_filter("finite", |v| v.is_finite()),
                    any::<u32>()
                )
                    .prop_map(|(lr, ni)| PlannerType::Gradient {
                        learning_rate: lr,
                        num_iterations: ni,
                    }),
                (any::<u32>(), any::<u32>()).prop_map(|(ns, tk)| PlannerType::Sampling {
                    num_samples: ns,
                    top_k: tk,
                }),
                Just(PlannerType::ProviderNative),
            ]
        }

        proptest! {
            #[test]
            fn physics_scores_roundtrip(scores in arb_physics_scores()) {
                let json = serde_json::to_string(&scores).unwrap();
                let scores2: PhysicsScores = serde_json::from_str(&json).unwrap();
                prop_assert!((scores.overall - scores2.overall).abs() < f32::EPSILON);
                prop_assert!((scores.object_permanence - scores2.object_permanence).abs() < f32::EPSILON);
            }

            #[test]
            fn prediction_config_roundtrip(
                steps in 1u32..100,
                w in 1u32..4096,
                h in 1u32..4096,
                fps in 1.0f32..120.0,
            ) {
                let config = PredictionConfig {
                    steps,
                    resolution: (w, h),
                    fps,
                    ..PredictionConfig::default()
                };
                let json = serde_json::to_string(&config).unwrap();
                let config2: PredictionConfig = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(config2.steps, steps);
                prop_assert_eq!(config2.resolution, (w, h));
            }

            #[test]
            fn planner_type_roundtrip(pt in arb_planner_type()) {
                let json = serde_json::to_string(&pt).unwrap();
                let _pt2: PlannerType = serde_json::from_str(&json).unwrap();
            }

            #[test]
            fn plan_goal_description_roundtrip(desc in ".*") {
                let goal = PlanGoal::Description(desc.clone());
                let json = serde_json::to_string(&goal).unwrap();
                let goal2: PlanGoal = serde_json::from_str(&json).unwrap();
                match goal2 {
                    PlanGoal::Description(s) => prop_assert_eq!(s, desc),
                    _ => prop_assert!(false, "wrong variant"),
                }
            }
        }
    }
}
