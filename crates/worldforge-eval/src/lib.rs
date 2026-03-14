//! WorldForge Evaluation Framework
//!
//! Provides standardized evaluation of world foundation models across
//! dimensions like physics plausibility, spatial consistency, and
//! temporal coherence.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use worldforge_core::action::Action;
use worldforge_core::error::Result;
use worldforge_core::prediction::{PhysicsScores, PredictionConfig};
use worldforge_core::provider::WorldModelProvider;
use worldforge_core::state::WorldState;
use worldforge_core::types::VideoClip;

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

impl EvalSuite {
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

    /// Run the evaluation suite against a set of providers.
    pub async fn run(&self, providers: &[&dyn WorldModelProvider]) -> Result<EvalReport> {
        let config = PredictionConfig::default();
        let mut all_results = Vec::new();

        for provider in providers {
            for scenario in &self.scenarios {
                let start = std::time::Instant::now();

                let mut scores = HashMap::new();
                let mut outcomes = Vec::new();

                // Run prediction for each action
                let mut current_state = scenario.initial_state.clone();
                for action in &scenario.actions {
                    match provider.predict(&current_state, action, &config).await {
                        Ok(prediction) => {
                            // Record physics scores
                            record_physics_scores(&prediction.physics_scores, &mut scores);

                            // Check expected outcomes
                            for expected in &scenario.expected_outcomes {
                                outcomes.push(check_outcome(expected, &prediction));
                            }

                            current_state = prediction.output_state;
                        }
                        Err(e) => {
                            outcomes.push(OutcomeResult {
                                description: "prediction".to_string(),
                                passed: false,
                                details: Some(e.to_string()),
                            });
                        }
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

        // Build leaderboard
        let leaderboard = build_leaderboard(&all_results, self.scenarios.len());

        Ok(EvalReport {
            suite: self.name.clone(),
            results: all_results,
            leaderboard,
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
    prediction: &worldforge_core::prediction::Prediction,
) -> OutcomeResult {
    match expected {
        ExpectedOutcome::MinPhysicsScore {
            dimension,
            threshold,
        } => {
            let score = match dimension {
                EvalDimension::ObjectPermanence => prediction.physics_scores.object_permanence,
                EvalDimension::GravityCompliance => prediction.physics_scores.gravity_compliance,
                EvalDimension::CollisionAccuracy => prediction.physics_scores.collision_accuracy,
                EvalDimension::SpatialConsistency => prediction.physics_scores.spatial_consistency,
                EvalDimension::TemporalConsistency => {
                    prediction.physics_scores.temporal_consistency
                }
                _ => prediction.physics_scores.overall,
            };
            OutcomeResult {
                description: format!("{dimension:?} >= {threshold}"),
                passed: score >= *threshold,
                details: Some(format!("score: {score:.3}")),
            }
        }
        ExpectedOutcome::MinConfidence { threshold } => OutcomeResult {
            description: format!("confidence >= {threshold}"),
            passed: prediction.confidence >= *threshold,
            details: Some(format!("confidence: {:.3}", prediction.confidence)),
        },
        ExpectedOutcome::ObjectExists { name } => {
            let exists = prediction
                .output_state
                .scene
                .objects
                .values()
                .any(|o| o.name == *name);
            OutcomeResult {
                description: format!("object '{name}' exists"),
                passed: exists,
                details: None,
            }
        }
        ExpectedOutcome::ObjectNotExists { name } => {
            let exists = prediction
                .output_state
                .scene
                .objects
                .values()
                .any(|o| o.name == *name);
            OutcomeResult {
                description: format!("object '{name}' does not exist"),
                passed: !exists,
                details: None,
            }
        }
    }
}

fn build_leaderboard(results: &[EvalResult], total_scenarios: usize) -> Vec<LeaderboardEntry> {
    let mut by_provider: HashMap<String, Vec<&EvalResult>> = HashMap::new();
    for r in results {
        by_provider.entry(r.provider.clone()).or_default().push(r);
    }

    let mut entries: Vec<LeaderboardEntry> = by_provider
        .into_iter()
        .map(|(provider, results)| {
            let avg_score = if results.is_empty() {
                0.0
            } else {
                let total: f32 = results.iter().filter_map(|r| r.scores.get("overall")).sum();
                total / results.len() as f32
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

            LeaderboardEntry {
                provider,
                average_score: avg_score,
                average_latency_ms: avg_latency,
                scenarios_passed: passed,
                total_scenarios,
            }
        })
        .collect();

    entries.sort_by(|a, b| {
        b.average_score
            .partial_cmp(&a.average_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_providers::MockProvider;

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
        assert_eq!(report.leaderboard[0].provider, "mock");
    }
}
