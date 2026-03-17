//! Cross-crate integration tests for worldforge-eval.
//!
//! Tests the evaluation framework with real providers and
//! verifies the full eval pipeline: suite creation → run → report.

use worldforge_core::provider::WorldModelProvider;
use worldforge_eval::EvalSuite;
use worldforge_providers::MockProvider;

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
