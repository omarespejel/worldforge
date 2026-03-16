//! World state management and orchestration.
//!
//! The `World` struct is the primary user-facing object for interacting
//! with a simulated world through one or more providers.

use std::time::Duration;

use tracing::instrument;

use crate::action::Action;
use crate::error::{Result, WorldForgeError};
use crate::guardrail::{evaluate_guardrails, has_blocking_violation};
use crate::prediction::{
    ComparisonReport, MultiPrediction, Plan, PlanRequest, PlannerType, Prediction,
    PredictionConfig, ProviderScore,
};
use crate::provider::ProviderRegistry;
use crate::state::{HistoryEntry, PredictionSummary, WorldState};
use crate::types::SimTime;

/// A live world instance backed by one or more providers.
pub struct World {
    /// Current world state.
    pub state: WorldState,
    /// Default provider name for predictions.
    pub default_provider: String,
    /// Reference to the provider registry.
    registry: std::sync::Arc<ProviderRegistry>,
}

impl World {
    /// Create a new world with the given state and provider registry.
    pub fn new(
        state: WorldState,
        default_provider: impl Into<String>,
        registry: std::sync::Arc<ProviderRegistry>,
    ) -> Self {
        Self {
            state,
            default_provider: default_provider.into(),
            registry,
        }
    }

    /// Get the world's unique ID.
    pub fn id(&self) -> crate::types::WorldId {
        self.state.id
    }

    /// Predict the next world state after applying an action.
    #[instrument(skip(self, config))]
    pub async fn predict(
        &mut self,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        self.predict_with_provider(action, config, &self.default_provider.clone())
            .await
    }

    /// Predict using a specific provider.
    #[instrument(skip(self, config))]
    pub async fn predict_with_provider(
        &mut self,
        action: &Action,
        config: &PredictionConfig,
        provider_name: &str,
    ) -> Result<Prediction> {
        let mut prediction = self
            .run_prediction_with_fallback(&self.state, action, config, provider_name)
            .await?;

        // Evaluate guardrails on the predicted state
        if !config.guardrails.is_empty() {
            let results = evaluate_guardrails(&config.guardrails, &prediction.output_state);
            if has_blocking_violation(&results) {
                return Err(WorldForgeError::GuardrailBlocked {
                    reason: results
                        .iter()
                        .filter(|r| !r.passed)
                        .map(|r| {
                            format!(
                                "{}: {}",
                                r.guardrail_name,
                                r.violation_details.as_deref().unwrap_or("violation")
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("; "),
                });
            }
            prediction.guardrail_results = results;
        }

        // Update world state and history
        let hash = compute_state_hash(&prediction.output_state);
        self.state.history.push(HistoryEntry {
            time: self.state.time,
            state_hash: hash,
            action: Some(action.clone()),
            prediction: Some(PredictionSummary {
                confidence: prediction.confidence,
                physics_score: prediction.physics_scores.overall,
                latency_ms: prediction.latency_ms,
            }),
            provider: prediction.provider.clone(),
        });

        self.state.time = SimTime {
            step: self.state.time.step + config.steps as u64,
            seconds: self.state.time.seconds + (config.steps as f64 / config.fps as f64),
            dt: 1.0 / config.fps as f64,
        };
        self.state.scene = prediction.output_state.scene.clone();

        Ok(prediction)
    }

    /// Run prediction with multiple providers and compare results.
    #[instrument(skip(self, config))]
    pub async fn predict_multi(
        &self,
        action: &Action,
        provider_names: &[&str],
        config: &PredictionConfig,
    ) -> Result<MultiPrediction> {
        let mut predictions = Vec::new();

        for &name in provider_names {
            let pred = self
                .run_prediction(&self.state, action, config, name)
                .await?;
            predictions.push(pred);
        }

        if predictions.is_empty() {
            return Err(WorldForgeError::InternalError(
                "no predictions generated".to_string(),
            ));
        }

        // Find best prediction by overall physics score
        let best_idx = predictions
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.physics_scores
                    .overall
                    .partial_cmp(&b.physics_scores.overall)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Compute agreement score as average pairwise physics score similarity
        let agreement = if predictions.len() > 1 {
            let mut total = 0.0f32;
            let mut count = 0;
            for i in 0..predictions.len() {
                for j in (i + 1)..predictions.len() {
                    let diff = (predictions[i].physics_scores.overall
                        - predictions[j].physics_scores.overall)
                        .abs();
                    total += 1.0 - diff;
                    count += 1;
                }
            }
            if count > 0 {
                total / count as f32
            } else {
                1.0
            }
        } else {
            1.0
        };

        let scores = predictions
            .iter()
            .map(|p| ProviderScore {
                provider: p.provider.clone(),
                physics_scores: p.physics_scores,
                latency_ms: p.latency_ms,
                cost: p.cost.clone(),
            })
            .collect();

        Ok(MultiPrediction {
            agreement_score: agreement,
            best_prediction: best_idx,
            comparison: ComparisonReport {
                scores,
                summary: format!(
                    "Compared {} providers, best: {}",
                    predictions.len(),
                    predictions[best_idx].provider
                ),
            },
            predictions,
        })
    }

    /// Plan a sequence of actions to achieve a goal.
    ///
    /// Uses the specified planning algorithm to search for an action sequence
    /// that satisfies the goal while respecting guardrails. Currently supports
    /// the `Sampling` planner; other planner types will use sampling as a
    /// fallback.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::PlanningTimeout` if the timeout is exceeded,
    /// or `WorldForgeError::NoFeasiblePlan` if no valid plan is found.
    #[instrument(skip(self, request))]
    pub async fn plan(&self, request: &PlanRequest) -> Result<Plan> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs_f64(request.timeout_seconds);

        let (num_samples, top_k) = match &request.planner {
            PlannerType::Sampling { num_samples, top_k } => (*num_samples, *top_k),
            // Other planners fall back to sampling with sensible defaults
            PlannerType::CEM {
                population_size,
                num_iterations,
                ..
            } => (*population_size * *num_iterations, 1),
            PlannerType::MPC {
                num_samples,
                horizon,
                ..
            } => (*num_samples * *horizon, 1),
            _ => (32, 1),
        };

        let provider_name = &self.default_provider;
        let provider = self.registry.get(provider_name)?;
        let config = PredictionConfig::default();

        // Generate candidate action sequences and evaluate them
        let candidate_actions = generate_candidate_actions(&request.current_state, num_samples);
        let mut best_plan: Option<(Plan, f32)> = None;

        for actions in candidate_actions
            .iter()
            .take(top_k.max(num_samples) as usize)
        {
            if start.elapsed() > timeout {
                return Err(WorldForgeError::PlanningTimeout {
                    elapsed_ms: start.elapsed().as_millis() as u64,
                });
            }

            // Simulate the action sequence forward
            let mut sim_state = request.current_state.clone();
            let mut predicted_states = Vec::new();
            let mut guardrail_compliance = Vec::new();
            let mut total_score = 0.0f32;
            let mut feasible = true;

            for action in actions {
                let prediction = provider.predict(&sim_state, action, &config).await?;

                // Check guardrails
                let gr_results = if !request.guardrails.is_empty() {
                    let results =
                        evaluate_guardrails(&request.guardrails, &prediction.output_state);
                    if has_blocking_violation(&results) {
                        feasible = false;
                        break;
                    }
                    results
                } else {
                    Vec::new()
                };

                total_score += prediction.physics_scores.overall;
                sim_state = prediction.output_state;
                predicted_states.push(sim_state.clone());
                guardrail_compliance.push(gr_results);
            }

            if !feasible {
                continue;
            }

            // Evaluate goal satisfaction
            let goal_score = evaluate_goal_score(&request.goal, &sim_state);
            let combined_score = if actions.is_empty() {
                goal_score
            } else {
                (total_score / actions.len() as f32) * 0.3 + goal_score * 0.7
            };

            if best_plan
                .as_ref()
                .is_none_or(|(_, score)| combined_score > *score)
            {
                let planning_time_ms = start.elapsed().as_millis() as u64;
                best_plan = Some((
                    Plan {
                        actions: actions.clone(),
                        predicted_states,
                        predicted_videos: None,
                        total_cost: 0.0,
                        success_probability: goal_score,
                        guardrail_compliance,
                        planning_time_ms,
                        iterations_used: 1,
                    },
                    combined_score,
                ));
            }
        }

        match best_plan {
            Some((mut plan, _)) => {
                plan.planning_time_ms = start.elapsed().as_millis() as u64;
                Ok(plan)
            }
            None => Err(WorldForgeError::NoFeasiblePlan {
                goal: format!("{:?}", request.goal),
                reason: "no candidate action sequence passed guardrails".to_string(),
            }),
        }
    }

    /// Get the current world state.
    pub fn current_state(&self) -> &WorldState {
        &self.state
    }

    async fn run_prediction(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
        provider_name: &str,
    ) -> Result<Prediction> {
        let provider = self.registry.get(provider_name)?;

        if let Some(timeout_ms) = config.max_latency_ms {
            tokio::time::timeout(
                Duration::from_millis(timeout_ms),
                provider.predict(state, action, config),
            )
            .await
            .map_err(|_| WorldForgeError::ProviderTimeout {
                provider: provider_name.to_string(),
                timeout_ms,
            })?
        } else {
            provider.predict(state, action, config).await
        }
    }

    async fn run_prediction_with_fallback(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
        provider_name: &str,
    ) -> Result<Prediction> {
        match self
            .run_prediction(state, action, config, provider_name)
            .await
        {
            Ok(prediction) => Ok(prediction),
            Err(primary_error) => {
                let Some(fallback_provider) = config
                    .fallback_provider
                    .as_deref()
                    .filter(|fallback| *fallback != provider_name)
                else {
                    return Err(primary_error);
                };

                tracing::warn!(
                    provider = provider_name,
                    fallback = fallback_provider,
                    error = %primary_error,
                    "prediction failed on primary provider, attempting fallback"
                );

                match self
                    .run_prediction(state, action, config, fallback_provider)
                    .await
                {
                    Ok(prediction) => Ok(prediction),
                    Err(fallback_error) => Err(WorldForgeError::ProviderUnavailable {
                        provider: provider_name.to_string(),
                        reason: format!(
                            "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                        ),
                    }),
                }
            }
        }
    }
}

/// Generate candidate action sequences for planning.
///
/// Creates a set of simple single-action sequences based on objects
/// present in the current scene. For each movable object, generates
/// move, push, and rotate actions.
fn generate_candidate_actions(state: &WorldState, num_samples: u32) -> Vec<Vec<Action>> {
    use crate::types::{Position, Vec3};

    let mut candidates = Vec::new();
    let objects: Vec<_> = state
        .scene
        .objects
        .values()
        .filter(|o| !o.physics.is_static)
        .collect();

    if objects.is_empty() {
        // If no movable objects, generate a single no-op
        candidates.push(vec![Action::Move {
            target: Position::default(),
            speed: 1.0,
        }]);
        return candidates;
    }

    // Generate actions for each movable object
    let directions = [
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        Vec3 {
            x: -1.0,
            y: 0.0,
            z: 0.0,
        },
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
    ];

    for obj in &objects {
        // Push in each direction
        for dir in &directions {
            candidates.push(vec![Action::Push {
                object: obj.id,
                direction: *dir,
                force: 1.0,
            }]);
        }

        // Move to displaced positions
        let offsets = [1.0f32, -1.0, 2.0, -2.0];
        for &dx in &offsets {
            candidates.push(vec![Action::Place {
                object: obj.id,
                target: Position {
                    x: obj.pose.position.x + dx,
                    y: obj.pose.position.y,
                    z: obj.pose.position.z,
                },
            }]);
        }

        // Rotate around Y axis
        candidates.push(vec![Action::Rotate {
            object: obj.id,
            axis: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            angle: std::f32::consts::FRAC_PI_4,
        }]);

        if candidates.len() >= num_samples as usize {
            break;
        }
    }

    candidates.truncate(num_samples as usize);
    candidates
}

/// Evaluate how well a state satisfies a planning goal.
///
/// Returns a score between 0.0 (no progress) and 1.0 (goal achieved).
fn evaluate_goal_score(goal: &crate::prediction::PlanGoal, state: &WorldState) -> f32 {
    use crate::prediction::PlanGoal;

    match goal {
        PlanGoal::Condition(condition) => {
            if crate::action::evaluate_condition(condition, state) {
                1.0
            } else {
                0.0
            }
        }
        PlanGoal::TargetState(target) => {
            // Compare object positions between current and target state
            if target.scene.objects.is_empty() {
                return 0.5;
            }
            let mut total_similarity = 0.0f32;
            let mut count = 0;
            for (id, target_obj) in &target.scene.objects {
                if let Some(current_obj) = state.scene.get_object(id) {
                    let dx = current_obj.pose.position.x - target_obj.pose.position.x;
                    let dy = current_obj.pose.position.y - target_obj.pose.position.y;
                    let dz = current_obj.pose.position.z - target_obj.pose.position.z;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    // Convert distance to similarity (closer = higher score)
                    total_similarity += 1.0 / (1.0 + dist);
                    count += 1;
                }
            }
            if count > 0 {
                total_similarity / count as f32
            } else {
                0.0
            }
        }
        PlanGoal::Description(_) | PlanGoal::GoalImage(_) => {
            // Natural language and image goals require provider-level reasoning
            // Return a neutral score as we can't evaluate locally
            0.5
        }
    }
}

/// Compute a non-cryptographic fingerprint of a world state for history tracking.
///
/// Uses multiple independent hash rounds to populate all 32 bytes. This is
/// **not** a cryptographic hash — it is only used for quick equality checks
/// and deduplication within the state history.
fn compute_state_hash(state: &WorldState) -> [u8; 32] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut result = [0u8; 32];

    // Round 1: world identity
    let mut h1 = DefaultHasher::new();
    state.id.hash(&mut h1);
    state.time.step.hash(&mut h1);
    result[..8].copy_from_slice(&h1.finish().to_le_bytes());

    // Round 2: scene contents
    let mut h2 = DefaultHasher::new();
    state.scene.objects.len().hash(&mut h2);
    for name in state.scene.objects.values().map(|o| &o.name) {
        name.hash(&mut h2);
    }
    result[8..16].copy_from_slice(&h2.finish().to_le_bytes());

    // Round 3: temporal state
    let mut h3 = DefaultHasher::new();
    state.time.seconds.to_bits().hash(&mut h3);
    state.history.len().hash(&mut h3);
    result[16..24].copy_from_slice(&h3.finish().to_le_bytes());

    // Round 4: metadata
    let mut h4 = DefaultHasher::new();
    state.metadata.name.hash(&mut h4);
    state.metadata.created_by.hash(&mut h4);
    result[24..32].copy_from_slice(&h4.finish().to_le_bytes());

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    use crate::error::WorldForgeError;
    use crate::prediction::PlanGoal;
    use crate::prediction::{PhysicsScores, Prediction};
    use crate::provider::{
        CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
        ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
        WorldModelProvider,
    };
    use crate::types::VideoClip;

    #[test]
    fn test_compute_state_hash() {
        let state = WorldState::new("test", "mock");
        let hash = compute_state_hash(&state);
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn test_generate_candidate_actions_empty_scene() {
        let state = WorldState::new("test", "mock");
        let candidates = generate_candidate_actions(&state, 10);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_generate_candidate_actions_with_objects() {
        let mut state = WorldState::new("test", "mock");
        let obj = crate::scene::SceneObject::new(
            "ball",
            crate::types::Pose::default(),
            crate::types::BBox {
                min: crate::types::Position {
                    x: -0.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: crate::types::Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );
        state.scene.add_object(obj);
        let candidates = generate_candidate_actions(&state, 20);
        // Should generate push (4 dirs) + place (4 offsets) + rotate (1) = 9 for one object
        assert!(candidates.len() >= 9);
    }

    #[test]
    fn test_evaluate_goal_score_condition() {
        let state = WorldState::new("test", "mock");
        let fake_id = uuid::Uuid::new_v4();

        // Condition not met => 0.0
        let goal = PlanGoal::Condition(crate::action::Condition::ObjectExists { object: fake_id });
        assert_eq!(evaluate_goal_score(&goal, &state), 0.0);

        // NOT(ObjectExists) => met => 1.0
        let goal = PlanGoal::Condition(crate::action::Condition::Not(Box::new(
            crate::action::Condition::ObjectExists { object: fake_id },
        )));
        assert_eq!(evaluate_goal_score(&goal, &state), 1.0);
    }

    #[test]
    fn test_evaluate_goal_score_description() {
        let state = WorldState::new("test", "mock");
        let goal = PlanGoal::Description("stack the blocks".to_string());
        assert_eq!(evaluate_goal_score(&goal, &state), 0.5);
    }

    #[test]
    fn test_evaluate_goal_score_target_state() {
        let mut state = WorldState::new("test", "mock");
        let obj = crate::scene::SceneObject::new(
            "ball",
            crate::types::Pose {
                position: crate::types::Position {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                ..Default::default()
            },
            crate::types::BBox {
                min: crate::types::Position {
                    x: -0.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: crate::types::Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );
        let id = obj.id;
        state.scene.add_object(obj);

        // Target is the same position => high score
        let mut target = WorldState::new("target", "mock");
        let mut target_obj = crate::scene::SceneObject::new(
            "ball",
            crate::types::Pose {
                position: crate::types::Position {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                ..Default::default()
            },
            crate::types::BBox {
                min: crate::types::Position {
                    x: -0.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: crate::types::Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );
        // Use same ID so we can compare
        target_obj.id = id;
        target.scene.objects.insert(id, target_obj);

        let goal = PlanGoal::TargetState(Box::new(target));
        let score = evaluate_goal_score(&goal, &state);
        // Distance is 0, so similarity = 1/(1+0) = 1.0
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_predict_uses_fallback_provider() {
        let state = WorldState::new("fallback", "primary");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(FailingProvider::new("primary")));
            registry.register(Box::new(SuccessfulProvider::new("fallback")));
            registry
        });
        let mut world = World::new(state, "primary", registry);
        let action = Action::Move {
            target: crate::types::Position::default(),
            speed: 1.0,
        };
        let config = PredictionConfig {
            fallback_provider: Some("fallback".to_string()),
            ..PredictionConfig::default()
        };

        let prediction = world.predict(&action, &config).await.unwrap();

        assert_eq!(prediction.provider, "fallback");
        assert_eq!(world.current_state().history.len(), 1);
        assert_eq!(
            world.current_state().history.latest().unwrap().provider,
            "fallback"
        );
    }

    #[tokio::test]
    async fn test_predict_timeout_uses_fallback_provider() {
        let state = WorldState::new("timeout", "slow");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(SlowProvider::new("slow", 25)));
            registry.register(Box::new(SuccessfulProvider::new("fallback")));
            registry
        });
        let mut world = World::new(state, "slow", registry);
        let action = Action::Move {
            target: crate::types::Position::default(),
            speed: 1.0,
        };
        let config = PredictionConfig {
            fallback_provider: Some("fallback".to_string()),
            max_latency_ms: Some(1),
            ..PredictionConfig::default()
        };

        let prediction = world.predict(&action, &config).await.unwrap();

        assert_eq!(prediction.provider, "fallback");
        assert_eq!(world.current_state().time.step, 1);
    }

    #[tokio::test]
    async fn test_predict_records_guardrail_results() {
        let state = WorldState::new("guardrails", "mock");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(SuccessfulProvider::new("mock")));
            registry
        });
        let mut world = World::new(state, "mock", registry);
        let action = Action::Move {
            target: crate::types::Position::default(),
            speed: 1.0,
        };
        let config = PredictionConfig {
            guardrails: vec![crate::guardrail::GuardrailConfig {
                guardrail: crate::guardrail::Guardrail::NoCollisions,
                blocking: false,
            }],
            ..PredictionConfig::default()
        };

        let prediction = world.predict(&action, &config).await.unwrap();

        assert_eq!(prediction.guardrail_results.len(), 1);
        assert!(prediction.guardrail_results[0].passed);
    }

    #[derive(Debug, Clone)]
    struct FailingProvider {
        name: String,
    }

    impl FailingProvider {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    #[derive(Debug, Clone)]
    struct SlowProvider {
        name: String,
        delay_ms: u64,
    }

    impl SlowProvider {
        fn new(name: &str, delay_ms: u64) -> Self {
            Self {
                name: name.to_string(),
                delay_ms,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct SuccessfulProvider {
        name: String,
    }

    impl SuccessfulProvider {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    fn test_capabilities() -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: false,
            reason: false,
            transfer: false,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 4.0,
            max_resolution: (640, 480),
            fps_range: (1.0, 30.0),
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

    fn dummy_prediction(provider: &str, state: &WorldState, action: &Action) -> Prediction {
        Prediction {
            id: uuid::Uuid::new_v4(),
            provider: provider.to_string(),
            model: format!("{provider}-model"),
            input_state: state.clone(),
            action: action.clone(),
            output_state: state.clone(),
            video: None,
            confidence: 0.5,
            physics_scores: PhysicsScores::default(),
            latency_ms: 0,
            cost: CostEstimate::default(),
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    #[async_trait]
    impl WorldModelProvider for FailingProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            test_capabilities()
        }

        async fn predict(
            &self,
            _state: &WorldState,
            _action: &Action,
            _config: &PredictionConfig,
        ) -> Result<Prediction> {
            Err(WorldForgeError::ProviderUnavailable {
                provider: self.name.clone(),
                reason: "simulated failure".to_string(),
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
                healthy: false,
                message: "simulated failure".to_string(),
                latency_ms: 0,
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }

    #[async_trait]
    impl WorldModelProvider for SlowProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            test_capabilities()
        }

        async fn predict(
            &self,
            state: &WorldState,
            action: &Action,
            _config: &PredictionConfig,
        ) -> Result<Prediction> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(dummy_prediction(&self.name, state, action))
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
                message: "slow but healthy".to_string(),
                latency_ms: self.delay_ms,
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }

    #[async_trait]
    impl WorldModelProvider for SuccessfulProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            test_capabilities()
        }

        async fn predict(
            &self,
            state: &WorldState,
            action: &Action,
            _config: &PredictionConfig,
        ) -> Result<Prediction> {
            Ok(dummy_prediction(&self.name, state, action))
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
                latency_ms: 0,
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }
}
