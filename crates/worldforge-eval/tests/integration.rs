//! Cross-crate integration tests for worldforge-eval.
//!
//! Tests the evaluation framework with real providers and
//! verifies the full eval pipeline: suite creation → run → report.

use async_trait::async_trait;
use uuid::Uuid;
use worldforge_core::action::Condition;
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::prediction::{PhysicsScores, Prediction, PredictionSamplingMetadata};
use worldforge_core::provider::{
    CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
    ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
    WorldModelProvider,
};
use worldforge_core::scene::SceneObject;
use worldforge_core::state::WorldState;
use worldforge_core::types::{BBox, DType, Frame, Pose, Position, SimTime, Tensor, VideoClip};
use worldforge_eval::EvalSuite;
use worldforge_eval::{EvalDimension, EvalScenario, ExpectedOutcome};
use worldforge_providers::MockProvider;

fn sample_scene_suite() -> EvalSuite {
    let mut state = WorldState::new("integration_eval", "eval");
    let mut object = SceneObject::new(
        "cube",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.8,
                z: 0.0,
            },
            ..Default::default()
        },
        BBox {
            min: Position {
                x: -0.1,
                y: 0.7,
                z: -0.1,
            },
            max: Position {
                x: 0.1,
                y: 0.9,
                z: 0.1,
            },
        },
    );
    let cube_id = object.id;
    object.semantic_label = Some("cube".to_string());
    state.scene.add_object(object);

    EvalSuite {
        name: "Integration Scene Suite".to_string(),
        scenarios: vec![EvalScenario {
            name: "move_cube".to_string(),
            description: "Move a cube while preserving physics and identity".to_string(),
            initial_state: state,
            actions: vec![worldforge_core::action::Action::Move {
                target: Position {
                    x: 0.2,
                    y: 0.8,
                    z: 0.0,
                },
                speed: 1.0,
            }],
            expected_outcomes: vec![
                ExpectedOutcome::MinPhysicsScore {
                    dimension: EvalDimension::GravityCompliance,
                    threshold: 0.8,
                },
                ExpectedOutcome::MinConfidence { threshold: 0.5 },
                ExpectedOutcome::FinalStateCondition {
                    condition: Condition::And(vec![
                        Condition::ObjectExists { object: cube_id },
                        Condition::ObjectAt {
                            object: cube_id,
                            position: Position {
                                x: 0.2,
                                y: 0.8,
                                z: 0.0,
                            },
                            tolerance: 0.001,
                        },
                    ]),
                },
                ExpectedOutcome::ObjectPosition {
                    name: "cube".to_string(),
                    position: Position {
                        x: 0.2,
                        y: 0.8,
                        z: 0.0,
                    },
                    tolerance: 0.001,
                },
                ExpectedOutcome::ObjectSemanticLabel {
                    name: "cube".to_string(),
                    label: "cube".to_string(),
                },
            ],
            ground_truth: Some(sample_ground_truth_clip()),
        }],
        dimensions: vec![
            EvalDimension::ObjectPermanence,
            EvalDimension::GravityCompliance,
            EvalDimension::SpatialConsistency,
            EvalDimension::ActionPredictionAccuracy,
            EvalDimension::MaterialUnderstanding,
        ],
        providers: vec![],
    }
}

fn sample_ground_truth_clip() -> VideoClip {
    let frame = Frame {
        data: Tensor::zeros(vec![2, 2, 3], DType::UInt8),
        timestamp: SimTime {
            step: 0,
            seconds: 0.0,
            dt: 0.0,
        },
        camera: None,
        depth: None,
        segmentation: None,
    };

    VideoClip {
        frames: vec![
            frame.clone(),
            Frame {
                timestamp: SimTime {
                    step: 1,
                    seconds: 0.5,
                    dt: 0.5,
                },
                ..frame
            },
        ],
        fps: 2.0,
        resolution: (2, 2),
        duration: 1.0,
    }
}

#[derive(Debug, Default)]
struct SamplingFixtureProvider;

impl SamplingFixtureProvider {
    fn new() -> Self {
        Self
    }
}

fn sampled_prediction(
    provider: &str,
    state: &WorldState,
    action: &worldforge_core::action::Action,
    confidence: f32,
    physics: PhysicsScores,
) -> Prediction {
    let mut output_state = state.clone();
    output_state.time.step += 1;

    Prediction {
        id: Uuid::new_v4(),
        provider: provider.to_string(),
        model: "sampling-fixture".to_string(),
        input_state: state.clone(),
        action: action.clone(),
        output_state,
        video: None,
        confidence,
        physics_scores: physics,
        latency_ms: 1,
        cost: CostEstimate::default(),
        provenance: None,
        sampling: None,
        guardrail_results: Vec::new(),
        timestamp: chrono::Utc::now(),
    }
}

#[async_trait]
impl WorldModelProvider for SamplingFixtureProvider {
    fn name(&self) -> &str {
        "sampling-fixture"
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
        action: &worldforge_core::action::Action,
        _config: &worldforge_core::prediction::PredictionConfig,
    ) -> Result<Prediction> {
        let samples = vec![
            sampled_prediction(
                self.name(),
                state,
                action,
                0.82,
                PhysicsScores {
                    overall: 0.8,
                    object_permanence: 0.81,
                    gravity_compliance: 0.8,
                    collision_accuracy: 0.79,
                    spatial_consistency: 0.83,
                    temporal_consistency: 0.8,
                },
            ),
            sampled_prediction(
                self.name(),
                state,
                action,
                0.91,
                PhysicsScores {
                    overall: 0.9,
                    object_permanence: 0.89,
                    gravity_compliance: 0.92,
                    collision_accuracy: 0.91,
                    spatial_consistency: 0.9,
                    temporal_consistency: 0.93,
                },
            ),
        ];
        let sampling = PredictionSamplingMetadata::from_predictions(&samples, 4, 1);
        let mut output_state = state.clone();
        output_state.time.step += 1;

        Ok(Prediction {
            id: Uuid::new_v4(),
            provider: self.name().to_string(),
            model: "sampling-fixture".to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video: None,
            confidence: 0.87,
            physics_scores: PhysicsScores {
                overall: 0.85,
                object_permanence: 0.85,
                gravity_compliance: 0.85,
                collision_accuracy: 0.84,
                spatial_consistency: 0.86,
                temporal_consistency: 0.85,
            },
            latency_ms: 1,
            cost: CostEstimate::default(),
            provenance: None,
            sampling: Some(sampling),
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
            provider: self.name().to_string(),
            capability: "generate".to_string(),
        })
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: self.name().to_string(),
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
            provider: self.name().to_string(),
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

#[tokio::test]
async fn test_physics_suite_with_mock() {
    let suite = EvalSuite::physics_standard();
    let mock = MockProvider::new();
    let providers: Vec<&dyn WorldModelProvider> = vec![&mock];

    let report = suite.run(&providers).await.unwrap();
    assert_eq!(report.suite, "Physics Standard");
    assert!(!report.results.is_empty());
    assert!(!report.leaderboard.is_empty());
    assert_eq!(report.provider_summaries.len(), 1);
    assert_eq!(report.dimension_summaries.len(), suite.dimensions.len());
    assert_eq!(report.scenario_summaries.len(), suite.scenarios.len());

    let entry = &report.leaderboard[0];
    assert_eq!(entry.provider, "mock");
    assert!(entry.total_scenarios > 0);
}

#[tokio::test]
async fn test_manipulation_suite_with_mock() {
    let suite = EvalSuite::manipulation_standard();
    let mock = MockProvider::new();
    let providers: Vec<&dyn WorldModelProvider> = vec![&mock];

    let report = suite.run(&providers).await.unwrap();
    assert_eq!(report.suite, "Manipulation Standard");
    assert!(!report.results.is_empty());
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

#[tokio::test]
async fn test_spatial_reasoning_suite_with_mock() {
    let suite = EvalSuite::spatial_reasoning();
    let mock = MockProvider::new();
    let providers: Vec<&dyn WorldModelProvider> = vec![&mock];

    let report = suite.run(&providers).await.unwrap();
    assert_eq!(report.suite, "Spatial Reasoning");
    assert!(!report.results.is_empty());
    let spatial_summary = report
        .dimension_summaries
        .iter()
        .find(|summary| summary.dimension == "spatial_reasoning")
        .unwrap();
    assert!(!spatial_summary.provider_scores.is_empty());
}

#[tokio::test]
async fn test_comprehensive_suite_with_mock() {
    let suite = EvalSuite::comprehensive();
    let mock = MockProvider::new();
    let providers: Vec<&dyn WorldModelProvider> = vec![&mock];

    let report = suite.run(&providers).await.unwrap();
    assert_eq!(report.suite, "Comprehensive");
    // Comprehensive should include all scenarios from other suites
    assert!(report.results.len() >= 7);
}

#[tokio::test]
async fn test_multi_provider_eval() {
    let suite = EvalSuite::physics_standard();
    let mock1 = MockProvider::new();
    let mock2 = MockProvider::with_name("mock2");
    let providers: Vec<&dyn WorldModelProvider> = vec![&mock1, &mock2];

    let report = suite.run(&providers).await.unwrap();
    // Should have results for both providers
    assert!(report.leaderboard.len() >= 2);

    // Leaderboard should be sorted by score (descending)
    if report.leaderboard.len() >= 2 {
        assert!(report.leaderboard[0].average_score >= report.leaderboard[1].average_score);
    }
}

#[tokio::test]
async fn test_eval_report_serialization() {
    let suite = EvalSuite::physics_standard();
    let mock = MockProvider::new();
    let providers: Vec<&dyn WorldModelProvider> = vec![&mock];

    let report = suite.run(&providers).await.unwrap();
    let json = serde_json::to_string(&report).unwrap();
    assert!(!json.is_empty());
    // Should be deserializable
    let deserialized: worldforge_eval::EvalReport = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.suite, report.suite);
    assert_eq!(
        deserialized.provider_summaries[0].provider,
        report.provider_summaries[0].provider
    );
}

#[tokio::test]
async fn test_custom_suite_scores_concrete_scene_thresholds() {
    let suite = sample_scene_suite();
    let mock = MockProvider::new();
    let providers: Vec<&dyn WorldModelProvider> = vec![&mock];

    let report = suite.run(&providers).await.unwrap();
    assert_eq!(report.results.len(), 1);

    let result = &report.results[0];
    assert_eq!(result.provider, "mock");
    assert_eq!(result.scenario, "move_cube");
    assert_eq!(result.outcomes.len(), 5);
    assert!(result.outcomes.iter().all(|outcome| outcome.passed));
    assert!(result.scores["action_prediction_accuracy"] >= 0.75);
    assert!(result.scores["material_understanding"] >= 0.75);
    assert!(result.scores["gravity_compliance"] >= 0.8);
    assert!(result.scores["overall"] >= 0.0);
}

#[tokio::test]
async fn test_sampling_diagnostics_flow_into_report_metrics() {
    let suite = EvalSuite {
        name: "Sampling Diagnostics".to_string(),
        scenarios: vec![EvalScenario {
            name: "sampled_move".to_string(),
            description: "Verify sampling metadata is preserved in evaluation reports".to_string(),
            initial_state: WorldState::new("sampling", "eval"),
            actions: vec![worldforge_core::action::Action::Move {
                target: Position {
                    x: 0.1,
                    y: 0.0,
                    z: 0.0,
                },
                speed: 0.5,
            }],
            expected_outcomes: vec![],
            ground_truth: None,
        }],
        dimensions: vec![EvalDimension::ObjectPermanence],
        providers: vec![],
    };
    let provider = SamplingFixtureProvider::new();
    let providers: Vec<&dyn WorldModelProvider> = vec![&provider];

    let report = suite.run(&providers).await.unwrap();
    assert_eq!(report.results.len(), 1);

    let result = &report.results[0];
    let sampling = result.sampling.as_ref().expect("sampling diagnostics");
    assert_eq!(sampling.summary.prediction_steps, 1);
    assert_eq!(sampling.summary.sampled_steps, 1);
    assert_eq!(sampling.summary.requested_samples, 4);
    assert_eq!(sampling.summary.completed_samples, 2);
    assert!(result.scores["sampling_completion_rate"] > 0.0);
    assert_eq!(result.scores["sampling_requested_samples"], 4.0);
    assert_eq!(result.scores["sampling_completed_samples"], 2.0);
    assert!(report.provider_summaries[0].sampling.is_some());
    assert!(report.scenario_summaries[0].sampling.is_some());

    let markdown = report.to_markdown().unwrap();
    assert!(markdown.contains("Sampling completion rate"));
    assert!(markdown.contains("Sampling steps"));
}

#[test]
fn test_ground_truth_video_roundtrips_through_suite_json() {
    let suite = sample_scene_suite();
    let json = serde_json::to_value(&suite).unwrap();

    assert_eq!(
        json["scenarios"][0]["ground_truth"]["frames"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        json["scenarios"][0]["ground_truth"]["resolution"],
        serde_json::json!([2, 2])
    );

    let roundtripped = EvalSuite::from_json_str(&serde_json::to_string(&suite).unwrap()).unwrap();
    let clip = roundtripped.scenarios[0].ground_truth.as_ref().unwrap();

    assert_eq!(clip.frames.len(), 2);
    assert_eq!(clip.fps, 2.0);
    assert_eq!(clip.resolution, (2, 2));
    assert_eq!(clip.duration, 1.0);
}

#[test]
fn test_final_state_condition_outcome_roundtrips_through_suite_json() {
    let suite = sample_scene_suite();
    let json = serde_json::to_string(&suite).unwrap();
    let roundtripped = EvalSuite::from_json_str(&json).unwrap();

    let condition = roundtripped.scenarios[0]
        .expected_outcomes
        .iter()
        .find_map(|outcome| match outcome {
            ExpectedOutcome::FinalStateCondition { condition } => Some(condition),
            _ => None,
        })
        .expect("final state condition outcome should survive roundtrip");

    match condition {
        Condition::And(conditions) => {
            assert_eq!(conditions.len(), 2);
            assert!(conditions
                .iter()
                .any(|condition| matches!(condition, Condition::ObjectExists { .. })));
            assert!(conditions
                .iter()
                .any(|condition| matches!(condition, Condition::ObjectAt { .. })));
        }
        other => panic!("expected And condition, got {other:?}"),
    }
}

#[tokio::test]
async fn test_ground_truth_video_similarity_is_reported() {
    let mock = MockProvider::new();
    let initial_state = WorldState::new("video_similarity", "mock");
    let action = worldforge_core::action::Action::SetLighting { time_of_day: 0.3 };
    let config = worldforge_core::prediction::PredictionConfig {
        return_video: true,
        ..Default::default()
    };
    let reference = mock
        .predict(&initial_state, &action, &config)
        .await
        .unwrap();

    let suite = EvalSuite {
        name: "Ground Truth Video Similarity".to_string(),
        scenarios: vec![EvalScenario {
            name: "video_similarity".to_string(),
            description: "Report a stable similarity score for a deterministic clip".to_string(),
            initial_state,
            actions: vec![action],
            expected_outcomes: vec![ExpectedOutcome::MinVideoSimilarity { threshold: 0.95 }],
            ground_truth: reference.video.clone(),
        }],
        dimensions: vec![EvalDimension::Custom {
            name: "video_similarity".to_string(),
        }],
        providers: vec![],
    };
    let providers: Vec<&dyn WorldModelProvider> = vec![&mock];

    let report = suite.run(&providers).await.unwrap();
    let result = &report.results[0];

    assert!(result.video.is_some());
    let metrics = result.video_metrics.as_ref().unwrap();
    assert!(metrics.overall_similarity >= 0.95);
    assert!(result.scores["video_similarity"] >= 0.95);
    assert!(result.outcomes.iter().all(|outcome| outcome.passed));
}
