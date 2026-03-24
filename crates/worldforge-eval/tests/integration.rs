//! Cross-crate integration tests for worldforge-eval.
//!
//! Tests the evaluation framework with real providers and
//! verifies the full eval pipeline: suite creation → run → report.

use worldforge_core::provider::WorldModelProvider;
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
        ],
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
}

#[tokio::test]
async fn test_spatial_reasoning_suite_with_mock() {
    let suite = EvalSuite::spatial_reasoning();
    let mock = MockProvider::new();
    let providers: Vec<&dyn WorldModelProvider> = vec![&mock];

    let report = suite.run(&providers).await.unwrap();
    assert_eq!(report.suite, "Spatial Reasoning");
    assert!(!report.results.is_empty());
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
    assert_eq!(result.outcomes.len(), 4);
    assert!(result.outcomes.iter().all(|outcome| outcome.passed));
    assert!(result.scores["gravity_compliance"] >= 0.8);
    assert!(result.scores["overall"] >= 0.0);
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
