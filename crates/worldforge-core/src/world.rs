//! World state management and orchestration.
//!
//! The `World` struct is the primary user-facing object for interacting
//! with a simulated world through one or more providers.

use std::time::Duration;

use tracing::instrument;

use crate::action::{Action, Condition, Weather};
use crate::error::{Result, WorldForgeError};
use crate::goal_image;
use crate::guardrail::{evaluate_guardrails, has_blocking_violation};
use crate::prediction::{
    MultiPrediction, Plan, PlanExecution, PlanRequest, PlannerType, Prediction, PredictionConfig,
};
use crate::provider::{
    GenerationConfig, GenerationPrompt, Operation, ProviderRegistry, ReasoningInput,
    ReasoningOutput, SpatialControls, TransferConfig, WorldModelProvider,
};
use crate::scene::{SceneObject, SceneObjectPatch};
use crate::state::{PredictionSummary, WorldState};
use crate::types::{ObjectId, Pose, Position, SimTime, Vec3};

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

    /// Add a new object to the world state.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if another object already uses
    /// the same ID.
    pub fn add_object(&mut self, object: SceneObject) -> Result<()> {
        if self.state.scene.get_object(&object.id).is_some() {
            return Err(WorldForgeError::InvalidState(format!(
                "object already exists: {}",
                object.id
            )));
        }

        self.state.scene.add_object(object);
        Ok(())
    }

    /// Replace an existing object in the world by ID.
    ///
    /// The replacement preserves the object's ID, keeps the scene hierarchy
    /// consistent, and refreshes inferred spatial relationships.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if the object does not exist.
    pub fn replace_object(&mut self, object: SceneObject) -> Result<SceneObject> {
        let object_id = object.id;
        self.state
            .scene
            .replace_object(object)
            .ok_or_else(|| WorldForgeError::InvalidState(format!("object not found: {object_id}")))
    }

    /// Get an object in the world by ID.
    pub fn get_object(&self, object_id: &ObjectId) -> Option<&SceneObject> {
        self.state.scene.get_object(object_id)
    }

    /// Get an object in the world by its human-readable name.
    pub fn get_object_by_name(&self, name: &str) -> Option<&SceneObject> {
        self.state.scene.find_object_by_name(name)
    }

    /// List all objects in the world in deterministic order.
    pub fn list_objects(&self) -> Vec<SceneObject> {
        self.state
            .scene
            .list_objects()
            .into_iter()
            .cloned()
            .collect()
    }

    /// Remove an object from the world by ID.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if the object does not exist.
    pub fn remove_object(&mut self, object_id: &ObjectId) -> Result<SceneObject> {
        self.state
            .scene
            .remove_object(object_id)
            .ok_or_else(|| WorldForgeError::InvalidState(format!("object not found: {object_id}")))
    }

    /// Update an object in the world by ID.
    ///
    /// The returned object preserves the original ID and reflects the updated scene state.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if the object does not exist.
    pub fn update_object(
        &mut self,
        object_id: &ObjectId,
        patch: SceneObjectPatch,
    ) -> Result<SceneObject> {
        self.state
            .scene
            .update_object(object_id, patch)
            .ok_or_else(|| WorldForgeError::InvalidState(format!("object not found: {object_id}")))
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
        let prediction = self
            .predict_from_state(&self.state, action, config, provider_name)
            .await?;

        let mut next_state = prediction.output_state.clone();
        if next_state.time == prediction.input_state.time {
            next_state.time = SimTime {
                step: prediction.input_state.time.step + config.steps as u64,
                seconds: prediction.input_state.time.seconds
                    + (config.steps as f64 / config.fps as f64),
                dt: 1.0 / config.fps as f64,
            };
        }

        next_state.record_current_state(
            Some(action.clone()),
            Some(PredictionSummary {
                confidence: prediction.confidence,
                physics_score: prediction.physics_scores.overall,
                latency_ms: prediction.latency_ms,
            }),
            prediction.provider.clone(),
        )?;

        self.state = next_state;

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
                .predict_from_state(&self.state, action, config, name)
                .await?;
            predictions.push(pred);
        }

        if predictions.is_empty() {
            return Err(WorldForgeError::InternalError(
                "no predictions generated".to_string(),
            ));
        }
        MultiPrediction::try_from_predictions(predictions)
    }

    /// Generate a video clip with the world's default provider.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<crate::types::VideoClip> {
        self.generate_with_provider(prompt, config, &self.default_provider)
            .await
    }

    /// Generate a video clip with a specific provider.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate_with_provider(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
        provider_name: &str,
    ) -> Result<crate::types::VideoClip> {
        self.generate_with_provider_and_fallback(prompt, config, provider_name, None)
            .await
            .map(|(_, clip)| clip)
    }

    /// Generate a video clip with a specific provider and optional fallback provider.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate_with_provider_and_fallback(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
        provider_name: &str,
        fallback_provider: Option<&str>,
    ) -> Result<(String, crate::types::VideoClip)> {
        match self.run_generate(prompt, config, provider_name).await {
            Ok(clip) => Ok((provider_name.to_string(), clip)),
            Err(primary_error) => {
                let Some(fallback_provider) =
                    fallback_provider.filter(|fallback| *fallback != provider_name)
                else {
                    return Err(primary_error);
                };

                tracing::warn!(
                    provider = provider_name,
                    fallback = fallback_provider,
                    error = %primary_error,
                    "generation failed on primary provider, attempting fallback"
                );

                match self.run_generate(prompt, config, fallback_provider).await {
                    Ok(clip) => Ok((fallback_provider.to_string(), clip)),
                    Err(fallback_error) => Err(combine_fallback_errors(
                        provider_name,
                        fallback_provider,
                        primary_error,
                        fallback_error,
                    )),
                }
            }
        }
    }

    /// Transfer spatial controls over an existing source clip with the world's default provider.
    #[instrument(skip(self, source, controls, config))]
    pub async fn transfer(
        &self,
        source: &crate::types::VideoClip,
        controls: &SpatialControls,
        config: &TransferConfig,
    ) -> Result<crate::types::VideoClip> {
        self.transfer_with_provider(source, controls, config, &self.default_provider)
            .await
    }

    /// Transfer spatial controls over an existing source clip with a specific provider.
    #[instrument(skip(self, source, controls, config))]
    pub async fn transfer_with_provider(
        &self,
        source: &crate::types::VideoClip,
        controls: &SpatialControls,
        config: &TransferConfig,
        provider_name: &str,
    ) -> Result<crate::types::VideoClip> {
        self.transfer_with_provider_and_fallback(source, controls, config, provider_name, None)
            .await
            .map(|(_, clip)| clip)
    }

    /// Transfer spatial controls over an existing source clip with a specific provider and optional fallback provider.
    #[instrument(skip(self, source, controls, config))]
    pub async fn transfer_with_provider_and_fallback(
        &self,
        source: &crate::types::VideoClip,
        controls: &SpatialControls,
        config: &TransferConfig,
        provider_name: &str,
        fallback_provider: Option<&str>,
    ) -> Result<(String, crate::types::VideoClip)> {
        match self
            .run_transfer(source, controls, config, provider_name)
            .await
        {
            Ok(clip) => Ok((provider_name.to_string(), clip)),
            Err(primary_error) => {
                let Some(fallback_provider) =
                    fallback_provider.filter(|fallback| *fallback != provider_name)
                else {
                    return Err(primary_error);
                };

                tracing::warn!(
                    provider = provider_name,
                    fallback = fallback_provider,
                    error = %primary_error,
                    "transfer failed on primary provider, attempting fallback"
                );

                match self
                    .run_transfer(source, controls, config, fallback_provider)
                    .await
                {
                    Ok(clip) => Ok((fallback_provider.to_string(), clip)),
                    Err(fallback_error) => Err(combine_fallback_errors(
                        provider_name,
                        fallback_provider,
                        primary_error,
                        fallback_error,
                    )),
                }
            }
        }
    }

    /// Ask the world's default provider to reason about the current state.
    #[instrument(skip(self, query))]
    pub async fn reason(&self, query: &str) -> Result<ReasoningOutput> {
        self.reason_with_provider(query, &self.default_provider)
            .await
    }

    /// Ask a specific provider to reason about the current world state.
    #[instrument(skip(self, query))]
    pub async fn reason_with_provider(
        &self,
        query: &str,
        provider_name: &str,
    ) -> Result<ReasoningOutput> {
        self.reason_with_provider_and_fallback(query, provider_name, None)
            .await
            .map(|(_, output)| output)
    }

    /// Ask a specific provider to reason about the current world state with an optional fallback provider.
    #[instrument(skip(self, query))]
    pub async fn reason_with_provider_and_fallback(
        &self,
        query: &str,
        provider_name: &str,
        fallback_provider: Option<&str>,
    ) -> Result<(String, ReasoningOutput)> {
        match self.run_reason(query, provider_name).await {
            Ok(output) => Ok((provider_name.to_string(), output)),
            Err(primary_error) => {
                let Some(fallback_provider) =
                    fallback_provider.filter(|fallback| *fallback != provider_name)
                else {
                    return Err(primary_error);
                };

                tracing::warn!(
                    provider = provider_name,
                    fallback = fallback_provider,
                    error = %primary_error,
                    "reasoning failed on primary provider, attempting fallback"
                );

                match self.run_reason(query, fallback_provider).await {
                    Ok(output) => Ok((fallback_provider.to_string(), output)),
                    Err(fallback_error) => Err(combine_fallback_errors(
                        provider_name,
                        fallback_provider,
                        primary_error,
                        fallback_error,
                    )),
                }
            }
        }
    }

    /// Plan a sequence of actions to achieve a goal.
    ///
    /// Uses the specified planning algorithm to search for an action sequence
    /// that satisfies the goal while respecting guardrails.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::PlanningTimeout` if the timeout is exceeded,
    /// or `WorldForgeError::NoFeasiblePlan` if no valid plan is found.
    #[instrument(skip(self, request))]
    pub async fn plan(&self, request: &PlanRequest) -> Result<Plan> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs_f64(request.timeout_seconds);
        let provider_name = &self.default_provider;
        let provider = self.registry.get(provider_name)?;
        let goal_hints = derive_goal_hints(&request.goal, &request.current_state);
        let seed = planning_seed(&request.current_state, &request.goal);
        let context = PlanningContext {
            provider,
            request,
            goal_hints: &goal_hints,
            start,
            timeout,
        };

        let candidate = match &request.planner {
            PlannerType::Sampling { num_samples, top_k } => {
                sampling_search(&context, *num_samples, *top_k, seed).await?
            }
            PlannerType::CEM {
                population_size,
                elite_fraction,
                num_iterations,
            } => {
                cem_search(
                    &context,
                    *population_size,
                    *elite_fraction,
                    *num_iterations,
                    seed,
                )
                .await?
            }
            PlannerType::MPC {
                horizon,
                num_samples,
                replanning_interval,
            } => mpc_search(&context, *horizon, *num_samples, *replanning_interval, seed).await?,
            PlannerType::Gradient {
                learning_rate,
                num_iterations,
            } => gradient_search(&context, *learning_rate, *num_iterations, seed).await?,
            PlannerType::ProviderNative => {
                if !provider.capabilities().supports_planning {
                    return Err(WorldForgeError::UnsupportedCapability {
                        provider: provider_name.to_string(),
                        capability: "native planning".to_string(),
                    });
                }

                let plan = tokio::time::timeout(timeout, provider.plan(request))
                    .await
                    .map_err(|_| WorldForgeError::PlanningTimeout {
                        elapsed_ms: timeout.as_millis() as u64,
                    })??;
                return finalize_provider_plan(provider_name, request, plan, start.elapsed());
            }
        };

        let Some(mut candidate) = candidate else {
            return Err(WorldForgeError::NoFeasiblePlan {
                goal: format!("{:?}", request.goal),
                reason: "no candidate action sequence passed guardrails".to_string(),
            });
        };

        candidate.plan.planning_time_ms = start.elapsed().as_millis() as u64;
        Ok(candidate.plan)
    }

    /// Execute a materialized plan against the world's default provider.
    ///
    /// The world state is only committed if every step succeeds.
    #[instrument(skip(self, plan, config))]
    pub async fn execute_plan(
        &mut self,
        plan: &Plan,
        config: &PredictionConfig,
    ) -> Result<PlanExecution> {
        let provider_name = self.default_provider.clone();
        self.execute_plan_with_provider(plan, config, &provider_name)
            .await
    }

    /// Execute a materialized plan against a specific provider.
    ///
    /// The world state is only committed if every step succeeds.
    #[instrument(skip(self, plan, config))]
    pub async fn execute_plan_with_provider(
        &mut self,
        plan: &Plan,
        config: &PredictionConfig,
        provider_name: &str,
    ) -> Result<PlanExecution> {
        let start = std::time::Instant::now();
        let mut sandbox = Self::new(
            self.state.clone(),
            self.default_provider.clone(),
            std::sync::Arc::clone(&self.registry),
        );
        let mut predictions = Vec::with_capacity(plan.actions.len());

        for action in &plan.actions {
            let prediction = sandbox
                .predict_with_provider(action, config, provider_name)
                .await?;
            predictions.push(prediction);
        }

        let final_state = sandbox.state.clone();
        self.state = final_state.clone();

        Ok(PlanExecution::from_predictions(
            predictions,
            final_state,
            start.elapsed().as_millis() as u64,
        ))
    }

    /// Get the current world state.
    pub fn current_state(&self) -> &WorldState {
        &self.state
    }

    /// Reconstruct a prior world state from a recorded history checkpoint.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if the requested checkpoint is
    /// unavailable.
    pub fn history_state(&self, index: usize) -> Result<WorldState> {
        self.state.history_state(index)
    }

    /// Restore this world in place to a recorded history checkpoint.
    ///
    /// The world's default provider is updated to match the restored
    /// checkpoint's recorded provider.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if the requested checkpoint is
    /// unavailable.
    pub fn restore_history(&mut self, index: usize) -> Result<()> {
        self.state.restore_history(index)?;
        self.default_provider = self.state.current_state_provider();
        Ok(())
    }

    async fn predict_from_state(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
        provider_name: &str,
    ) -> Result<Prediction> {
        let mut state = state.clone();
        let provider = state.current_state_provider();
        state.ensure_history_initialized(provider)?;
        let mut prediction = self
            .run_prediction_with_fallback(&state, action, config, provider_name)
            .await?;

        let results = evaluate_guardrails(&config.guardrails, &prediction.output_state);
        if has_blocking_violation(&results) {
            return Err(WorldForgeError::GuardrailBlocked {
                reason: results
                    .iter()
                    .filter(|result| !result.passed)
                    .map(|result| {
                        format!(
                            "{}: {}",
                            result.guardrail_name,
                            result.violation_details.as_deref().unwrap_or("violation")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; "),
            });
        }
        prediction.guardrail_results = results;

        Ok(prediction)
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

    async fn run_generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
        provider_name: &str,
    ) -> Result<crate::types::VideoClip> {
        let provider = self.registry.get(provider_name)?;
        provider.generate(prompt, config).await
    }

    async fn run_transfer(
        &self,
        source: &crate::types::VideoClip,
        controls: &SpatialControls,
        config: &TransferConfig,
        provider_name: &str,
    ) -> Result<crate::types::VideoClip> {
        let provider = self.registry.get(provider_name)?;
        provider.transfer(source, controls, config).await
    }

    async fn run_reason(&self, query: &str, provider_name: &str) -> Result<ReasoningOutput> {
        let provider = self.registry.get(provider_name)?;
        let input = ReasoningInput {
            video: None,
            state: Some(self.state.clone()),
        };
        provider.reason(&input, query).await
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
                    Err(fallback_error) => Err(combine_fallback_errors(
                        provider_name,
                        fallback_provider,
                        primary_error,
                        fallback_error,
                    )),
                }
            }
        }
    }
}

fn combine_fallback_errors(
    provider_name: &str,
    fallback_provider: &str,
    primary_error: WorldForgeError,
    fallback_error: WorldForgeError,
) -> WorldForgeError {
    WorldForgeError::ProviderUnavailable {
        provider: provider_name.to_string(),
        reason: format!(
            "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
        ),
    }
}

fn finalize_provider_plan(
    provider_name: &str,
    request: &PlanRequest,
    mut plan: Plan,
    elapsed: Duration,
) -> Result<Plan> {
    let step_count = plan.actions.len();
    if step_count > request.max_steps as usize {
        return Err(WorldForgeError::PlanningFailed {
            reason: format!(
                "provider-native plan from '{provider_name}' exceeded max_steps ({} > {})",
                step_count, request.max_steps
            ),
        });
    }
    if plan.predicted_states.len() != step_count {
        return Err(WorldForgeError::PlanningFailed {
            reason: format!(
                "provider-native plan from '{provider_name}' returned {} predicted states for {step_count} actions",
                plan.predicted_states.len()
            ),
        });
    }
    if let Some(videos) = &plan.predicted_videos {
        if videos.len() != step_count {
            return Err(WorldForgeError::PlanningFailed {
                reason: format!(
                    "provider-native plan from '{provider_name}' returned {} videos for {step_count} actions",
                    videos.len()
                ),
            });
        }
    }

    let mut computed_guardrails = Vec::with_capacity(step_count);
    for (index, state) in plan.predicted_states.iter().enumerate() {
        let results = evaluate_guardrails(&request.guardrails, state);
        if has_blocking_violation(&results) {
            let reason = results
                .iter()
                .filter(|result| !result.passed)
                .map(|result| {
                    format!(
                        "{}: {}",
                        result.guardrail_name,
                        result.violation_details.as_deref().unwrap_or("violation")
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            return Err(WorldForgeError::NoFeasiblePlan {
                goal: format!("{:?}", request.goal),
                reason: format!(
                    "provider-native plan from '{provider_name}' violated blocking guardrails at step {}: {reason}",
                    index + 1
                ),
            });
        }
        computed_guardrails.push(results);
    }

    if !plan.guardrail_compliance.is_empty() && plan.guardrail_compliance.len() != step_count {
        return Err(WorldForgeError::PlanningFailed {
            reason: format!(
                "provider-native plan from '{provider_name}' returned {} guardrail steps for {step_count} actions",
                plan.guardrail_compliance.len()
            ),
        });
    }

    plan.guardrail_compliance = computed_guardrails;
    plan.planning_time_ms = elapsed.as_millis() as u64;
    Ok(plan)
}

#[derive(Debug, Clone)]
struct EvaluatedCandidate {
    plan: Plan,
    score: f32,
}

#[derive(Debug, Clone)]
enum GoalHint {
    ObjectAt {
        object_id: ObjectId,
        _object_name: String,
        target: Position,
        tolerance: f32,
    },
    ObjectMissing {
        object_id: ObjectId,
        _object_name: String,
    },
    ObjectExists {
        object_name: String,
    },
    ObjectExistsAt {
        object_name: String,
        target: Position,
        tolerance: f32,
    },
    ObjectsTouching {
        a: ObjectId,
        b: ObjectId,
    },
    Weather {
        weather: Weather,
    },
    Lighting {
        time_of_day: f32,
    },
}

#[derive(Debug, Clone)]
struct PlannerRng {
    state: u64,
}

impl PlannerRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    fn index(&mut self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            (self.next_u32() as usize) % len
        }
    }

    fn coin(&mut self, probability: f32) -> bool {
        self.next_f32() <= probability.clamp(0.0, 1.0)
    }
}

struct PlanningContext<'a> {
    provider: &'a dyn WorldModelProvider,
    request: &'a PlanRequest,
    goal_hints: &'a [GoalHint],
    start: std::time::Instant,
    timeout: Duration,
}

async fn sampling_search(
    context: &PlanningContext<'_>,
    num_samples: u32,
    top_k: u32,
    seed: u64,
) -> Result<Option<EvaluatedCandidate>> {
    let candidate_budget = num_samples.max(1).saturating_mul(top_k.max(1));
    let candidates = generate_candidate_actions(
        &context.request.current_state,
        context.request.max_steps,
        candidate_budget,
        context.goal_hints,
        seed,
    );
    evaluate_candidates(context, candidates, 1).await
}

async fn cem_search(
    context: &PlanningContext<'_>,
    population_size: u32,
    elite_fraction: f32,
    num_iterations: u32,
    seed: u64,
) -> Result<Option<EvaluatedCandidate>> {
    let population_size = population_size.max(4);
    let elite_count = ((population_size as f32 * elite_fraction.clamp(0.05, 1.0)).ceil() as usize)
        .clamp(1, population_size as usize);
    let mut rng = PlannerRng::new(seed);
    let mut population = generate_candidate_actions(
        &context.request.current_state,
        context.request.max_steps,
        population_size,
        context.goal_hints,
        seed,
    );
    let mut best: Option<EvaluatedCandidate> = None;

    for iteration in 0..num_iterations.max(1) {
        ensure_planning_budget(context.start, context.timeout)?;
        let Some(round_best) =
            evaluate_candidates(context, population.clone(), iteration + 1).await?
        else {
            break;
        };
        if best
            .as_ref()
            .is_none_or(|current| round_best.score > current.score)
        {
            best = Some(round_best);
        }

        let mut scored = Vec::new();
        for candidate in population {
            if let Some(scored_candidate) =
                evaluate_candidate_sequence(context, &context.request.current_state, &candidate)
                    .await?
            {
                scored.push((candidate, scored_candidate.score));
            }
        }
        if scored.is_empty() {
            break;
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let elites: Vec<Vec<Action>> = scored
            .into_iter()
            .take(elite_count)
            .map(|(candidate, _)| candidate)
            .collect();

        population = Vec::with_capacity(population_size as usize);
        while population.len() < population_size as usize {
            let elite = elites[rng.index(elites.len())].clone();
            population.push(mutate_candidate_actions(
                &context.request.current_state,
                &elite,
                context.request.max_steps,
                context.goal_hints,
                &mut rng,
                0.35f32.powi((iteration + 1) as i32).max(0.04),
            ));
        }
    }

    Ok(best.map(|mut candidate| {
        candidate.plan.iterations_used = num_iterations.max(1);
        candidate
    }))
}

async fn gradient_search(
    context: &PlanningContext<'_>,
    learning_rate: f32,
    num_iterations: u32,
    seed: u64,
) -> Result<Option<EvaluatedCandidate>> {
    let mut rng = PlannerRng::new(seed);
    let mut candidates = generate_gradient_candidates(
        &context.request.current_state,
        context.request.max_steps,
        context.goal_hints,
        learning_rate,
    );
    if candidates.is_empty() {
        candidates = generate_candidate_actions(
            &context.request.current_state,
            context.request.max_steps,
            24,
            context.goal_hints,
            seed,
        );
    }

    let mut best: Option<EvaluatedCandidate> = None;
    let iterations = num_iterations.max(1);

    for iteration in 0..iterations {
        ensure_planning_budget(context.start, context.timeout)?;
        if let Some(round_best) =
            evaluate_candidates(context, candidates.clone(), iteration + 1).await?
        {
            if best
                .as_ref()
                .is_none_or(|current| round_best.score > current.score)
            {
                best = Some(round_best);
            }
        }

        let shrink = (1.0 - learning_rate.clamp(0.01, 0.95)).powi((iteration + 1) as i32);
        candidates = candidates
            .iter()
            .map(|candidate| {
                mutate_candidate_actions(
                    &context.request.current_state,
                    candidate,
                    context.request.max_steps,
                    context.goal_hints,
                    &mut rng,
                    shrink.max(0.02),
                )
            })
            .collect();
    }

    Ok(best.map(|mut candidate| {
        candidate.plan.iterations_used = iterations;
        candidate
    }))
}

async fn mpc_search(
    context: &PlanningContext<'_>,
    horizon: u32,
    num_samples: u32,
    replanning_interval: u32,
    seed: u64,
) -> Result<Option<EvaluatedCandidate>> {
    let mut rng = PlannerRng::new(seed);
    let mut simulated_state = context.request.current_state.clone();
    let mut actions = Vec::new();
    let mut predicted_states = Vec::new();
    let mut guardrail_compliance = Vec::new();
    let mut total_cost = 0.0f32;
    let mut iterations = 0u32;

    while actions.len() < context.request.max_steps as usize {
        ensure_planning_budget(context.start, context.timeout)?;

        let remaining_steps = context
            .request
            .max_steps
            .saturating_sub(actions.len() as u32);
        let local_horizon = horizon.max(1).min(remaining_steps);
        let local_hints = derive_goal_hints(&context.request.goal, &simulated_state);
        let local_candidates = generate_candidate_actions(
            &simulated_state,
            local_horizon,
            num_samples.max(8),
            &local_hints,
            u64::from(rng.next_u32()),
        );

        let local_request = PlanRequest {
            current_state: simulated_state.clone(),
            goal: context.request.goal.clone(),
            max_steps: local_horizon,
            guardrails: context.request.guardrails.clone(),
            planner: PlannerType::Sampling {
                num_samples: num_samples.max(8),
                top_k: 1,
            },
            timeout_seconds: context.request.timeout_seconds,
        };
        let local_context = PlanningContext {
            provider: context.provider,
            request: &local_request,
            goal_hints: &local_hints,
            start: context.start,
            timeout: context.timeout,
        };
        let Some(local_best) =
            evaluate_candidates(&local_context, local_candidates, iterations + 1).await?
        else {
            break;
        };

        let commit = replanning_interval
            .max(1)
            .min(local_best.plan.actions.len() as u32)
            .min(remaining_steps) as usize;
        if commit == 0 {
            break;
        }

        let per_step_cost = if local_best.plan.actions.is_empty() {
            0.0
        } else {
            local_best.plan.total_cost / local_best.plan.actions.len() as f32
        };
        for idx in 0..commit {
            actions.push(local_best.plan.actions[idx].clone());
            predicted_states.push(local_best.plan.predicted_states[idx].clone());
            guardrail_compliance.push(local_best.plan.guardrail_compliance[idx].clone());
            total_cost += per_step_cost;
        }
        simulated_state = predicted_states
            .last()
            .cloned()
            .unwrap_or_else(|| simulated_state.clone());
        iterations += 1;

        if evaluate_goal_score(&context.request.goal, &simulated_state) >= 0.95 {
            break;
        }
    }

    if actions.is_empty() {
        return Ok(None);
    }

    Ok(Some(EvaluatedCandidate {
        score: evaluate_goal_score(&context.request.goal, &simulated_state),
        plan: Plan {
            actions,
            predicted_states,
            predicted_videos: None,
            total_cost,
            success_probability: evaluate_goal_score(&context.request.goal, &simulated_state),
            guardrail_compliance,
            planning_time_ms: context.start.elapsed().as_millis() as u64,
            iterations_used: iterations.max(1),
        },
    }))
}

async fn evaluate_candidates(
    context: &PlanningContext<'_>,
    candidates: Vec<Vec<Action>>,
    iterations_used: u32,
) -> Result<Option<EvaluatedCandidate>> {
    let mut best: Option<EvaluatedCandidate> = None;

    for candidate in candidates {
        if let Some(mut scored) =
            evaluate_candidate_sequence(context, &context.request.current_state, &candidate).await?
        {
            scored.plan.iterations_used = iterations_used;
            if best
                .as_ref()
                .is_none_or(|current| scored.score > current.score)
            {
                best = Some(scored);
            }
        }
    }

    Ok(best)
}

async fn evaluate_candidate_sequence(
    context: &PlanningContext<'_>,
    initial_state: &WorldState,
    actions: &[Action],
) -> Result<Option<EvaluatedCandidate>> {
    ensure_planning_budget(context.start, context.timeout)?;

    let config = PredictionConfig::default();
    let mut simulated_state = initial_state.clone();
    let mut predicted_states = Vec::new();
    let mut guardrail_compliance = Vec::new();
    let mut total_physics = 0.0f32;
    let mut total_cost = 0.0f32;

    for action in actions {
        ensure_planning_budget(context.start, context.timeout)?;
        let prediction = context
            .provider
            .predict(&simulated_state, action, &config)
            .await?;
        let gr_results = evaluate_guardrails(&context.request.guardrails, &prediction.output_state);
        if has_blocking_violation(&gr_results) {
            return Ok(None);
        }

        total_physics += prediction.physics_scores.overall;
        total_cost += prediction.cost.usd as f32;
        simulated_state = prediction.output_state;
        predicted_states.push(simulated_state.clone());
        guardrail_compliance.push(gr_results);
    }

    let goal_score = evaluate_goal_score(&context.request.goal, &simulated_state);
    let mean_physics = if actions.is_empty() {
        0.0
    } else {
        total_physics / actions.len() as f32
    };
    let combined_score = goal_score * 0.8
        + mean_physics * 0.15
        + length_bonus(actions.len()) * 0.05
        + estimated_goal_alignment(context.goal_hints, initial_state, &simulated_state) * 0.05;

    Ok(Some(EvaluatedCandidate {
        score: combined_score,
        plan: Plan {
            actions: actions.to_vec(),
            predicted_states,
            predicted_videos: None,
            total_cost: if total_cost == 0.0 {
                estimate_plan_cost(context.provider, actions.len() as u32, &config)
            } else {
                total_cost
            },
            success_probability: goal_score,
            guardrail_compliance,
            planning_time_ms: context.start.elapsed().as_millis() as u64,
            iterations_used: 1,
        },
    }))
}

fn ensure_planning_budget(start: std::time::Instant, timeout: Duration) -> Result<()> {
    if start.elapsed() > timeout {
        return Err(WorldForgeError::PlanningTimeout {
            elapsed_ms: start.elapsed().as_millis() as u64,
        });
    }
    Ok(())
}

fn estimate_plan_cost(
    provider: &dyn WorldModelProvider,
    steps: u32,
    config: &PredictionConfig,
) -> f32 {
    let estimate = provider.estimate_cost(&Operation::Predict {
        steps: steps.max(1),
        resolution: config.resolution,
    });
    estimate.usd as f32
}

/// Generate candidate action sequences for planning.
///
/// Combines goal-directed sequences with exploratory mutations so every
/// planner can search over a deterministic but diverse candidate set.
fn generate_candidate_actions(
    state: &WorldState,
    max_steps: u32,
    budget: u32,
    goal_hints: &[GoalHint],
    seed: u64,
) -> Vec<Vec<Action>> {
    let mut candidates = goal_directed_candidates(state, max_steps, goal_hints);
    candidates.extend(exploratory_candidates(state, max_steps));

    if candidates.is_empty() {
        candidates.push(vec![Action::Move {
            target: Position::default(),
            speed: 1.0,
        }]);
    }

    let mut rng = PlannerRng::new(seed);
    let initial = candidates.clone();
    while candidates.len() < budget.max(1) as usize {
        let template = initial[rng.index(initial.len())].clone();
        candidates.push(mutate_candidate_actions(
            state, &template, max_steps, goal_hints, &mut rng, 0.35,
        ));
    }

    candidates.truncate(budget.max(1) as usize);
    candidates
}

fn goal_directed_candidates(
    state: &WorldState,
    max_steps: u32,
    goal_hints: &[GoalHint],
) -> Vec<Vec<Action>> {
    let mut candidates = Vec::new();
    let step_budget = max_steps.clamp(1, 4) as usize;

    for hint in goal_hints {
        match hint {
            GoalHint::ObjectAt {
                object_id, target, ..
            } => {
                candidates.push(vec![Action::Place {
                    object: *object_id,
                    target: *target,
                }]);
                if let Some(current) = state.scene.get_object(object_id) {
                    let waypoints =
                        interpolate_positions(current.pose.position, *target, step_budget);
                    candidates.push(
                        waypoints
                            .into_iter()
                            .map(|position| Action::Place {
                                object: *object_id,
                                target: position,
                            })
                            .collect(),
                    );

                    let direction = direction_between(current.pose.position, *target);
                    candidates.push(vec![Action::Push {
                        object: *object_id,
                        direction,
                        force: distance(current.pose.position, *target).max(0.5),
                    }]);
                }
            }
            GoalHint::ObjectMissing { object_id, .. } => {
                candidates.push(vec![Action::RemoveObject { object: *object_id }]);
            }
            GoalHint::ObjectExists { object_name } => {
                candidates.push(vec![Action::SpawnObject {
                    template: object_name.clone(),
                    pose: Pose {
                        position: default_spawn_position(state),
                        ..Default::default()
                    },
                }]);
            }
            GoalHint::ObjectExistsAt {
                object_name,
                target,
                ..
            } => {
                candidates.push(vec![Action::SpawnObject {
                    template: object_name.clone(),
                    pose: Pose {
                        position: *target,
                        ..Default::default()
                    },
                }]);
            }
            GoalHint::ObjectsTouching { a, b } => {
                if let Some(target) = state.scene.get_object(b).map(|object| object.pose.position) {
                    candidates.push(vec![Action::Place { object: *a, target }]);
                }
                if let Some(target) = state.scene.get_object(a).map(|object| object.pose.position) {
                    candidates.push(vec![Action::Place { object: *b, target }]);
                }
            }
            GoalHint::Weather { weather } => {
                candidates.push(vec![Action::SetWeather { weather: *weather }]);
            }
            GoalHint::Lighting { time_of_day } => {
                candidates.push(vec![Action::SetLighting {
                    time_of_day: *time_of_day,
                }]);
            }
        }
    }

    candidates
}

fn exploratory_candidates(state: &WorldState, max_steps: u32) -> Vec<Vec<Action>> {
    let mut candidates = Vec::new();
    let mut objects: Vec<&SceneObject> = state
        .scene
        .objects
        .values()
        .filter(|object| !object.physics.is_static)
        .collect();
    objects.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.id.as_bytes().cmp(b.id.as_bytes()))
    });

    if objects.is_empty() {
        candidates.push(vec![Action::Move {
            target: Position::default(),
            speed: 1.0,
        }]);
        return candidates;
    }

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
            y: 0.0,
            z: 1.0,
        },
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
    ];
    let offsets = [
        Position {
            x: 0.5,
            y: 0.0,
            z: 0.0,
        },
        Position {
            x: -0.5,
            y: 0.0,
            z: 0.0,
        },
        Position {
            x: 0.0,
            y: 0.0,
            z: 0.5,
        },
        Position {
            x: 0.0,
            y: 0.0,
            z: -0.5,
        },
    ];

    for object in objects {
        for direction in directions {
            candidates.push(vec![Action::Push {
                object: object.id,
                direction,
                force: 1.0,
            }]);
        }
        for offset in offsets {
            let target = Position {
                x: object.pose.position.x + offset.x,
                y: object.pose.position.y + offset.y,
                z: object.pose.position.z + offset.z,
            };
            candidates.push(vec![Action::Place {
                object: object.id,
                target,
            }]);
            if max_steps > 1 {
                candidates.push(vec![
                    Action::Push {
                        object: object.id,
                        direction: direction_between(object.pose.position, target),
                        force: 0.75,
                    },
                    Action::Place {
                        object: object.id,
                        target,
                    },
                ]);
            }
        }
        candidates.push(vec![Action::Rotate {
            object: object.id,
            axis: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            angle: std::f32::consts::FRAC_PI_4,
        }]);
    }

    candidates
}

fn mutate_candidate_actions(
    state: &WorldState,
    template: &[Action],
    max_steps: u32,
    goal_hints: &[GoalHint],
    rng: &mut PlannerRng,
    scale: f32,
) -> Vec<Action> {
    let preferred_target = goal_hints.iter().find_map(|hint| match hint {
        GoalHint::ObjectAt {
            object_id, target, ..
        } => Some((*object_id, *target)),
        _ => None,
    });

    let mut mutated = Vec::new();
    for action in template.iter().take(max_steps.max(1) as usize) {
        let next = match action {
            Action::Place { object, target } => {
                let adjusted = preferred_target
                    .filter(|(preferred_object, _)| preferred_object == object)
                    .map(|(_, preferred)| {
                        lerp_position(*target, preferred, (1.0 - scale).clamp(0.1, 0.95))
                    })
                    .unwrap_or(*target);
                Action::Place {
                    object: *object,
                    target: jitter_position(adjusted, rng, scale),
                }
            }
            Action::Move { target, speed } => Action::Move {
                target: preferred_target
                    .map(|(_, preferred)| jitter_position(preferred, rng, scale))
                    .unwrap_or_else(|| jitter_position(*target, rng, scale)),
                speed: (*speed + rng.range_f32(-scale, scale)).clamp(0.1, 3.0),
            },
            Action::Push {
                object,
                direction,
                force,
            } => Action::Push {
                object: *object,
                direction: normalize_vec3(Vec3 {
                    x: direction.x + rng.range_f32(-scale, scale),
                    y: direction.y + rng.range_f32(-scale * 0.5, scale * 0.5),
                    z: direction.z + rng.range_f32(-scale, scale),
                }),
                force: (*force + rng.range_f32(-scale, scale)).clamp(0.2, 4.0),
            },
            Action::Rotate {
                object,
                axis,
                angle,
            } => Action::Rotate {
                object: *object,
                axis: *axis,
                angle: (*angle + rng.range_f32(-scale, scale)).clamp(0.1, std::f32::consts::PI),
            },
            Action::SpawnObject { template, pose } => Action::SpawnObject {
                template: template.clone(),
                pose: Pose {
                    position: jitter_position(pose.position, rng, scale),
                    rotation: pose.rotation,
                },
            },
            Action::SetLighting { time_of_day } => Action::SetLighting {
                time_of_day: (*time_of_day + rng.range_f32(-4.0 * scale, 4.0 * scale))
                    .clamp(0.0, 24.0),
            },
            _ => action.clone(),
        };
        mutated.push(next);
    }

    if mutated.is_empty() {
        mutated.push(Action::Move {
            target: Position::default(),
            speed: 1.0,
        });
    }

    if mutated.len() < max_steps as usize && rng.coin(0.3) {
        let fallback = exploratory_candidates(state, 1)
            .into_iter()
            .next()
            .unwrap_or_else(|| {
                vec![Action::Move {
                    target: Position::default(),
                    speed: 1.0,
                }]
            });
        if let Some(action) = fallback.into_iter().next() {
            mutated.push(action);
        }
    }

    mutated.truncate(max_steps.max(1) as usize);
    mutated
}

fn generate_gradient_candidates(
    state: &WorldState,
    max_steps: u32,
    goal_hints: &[GoalHint],
    learning_rate: f32,
) -> Vec<Vec<Action>> {
    let mut candidates = Vec::new();
    let step_count = max_steps.clamp(1, 4) as usize;
    let rate = learning_rate.clamp(0.05, 0.95);

    for hint in goal_hints {
        if let GoalHint::ObjectAt {
            object_id, target, ..
        } = hint
        {
            if let Some(current) = state.scene.get_object(object_id) {
                let mut sequence = Vec::new();
                let mut cursor = current.pose.position;
                for _ in 0..step_count {
                    cursor = lerp_position(cursor, *target, rate);
                    sequence.push(Action::Place {
                        object: *object_id,
                        target: cursor,
                    });
                }
                candidates.push(sequence);
            }
        }
    }

    if candidates.is_empty() {
        candidates.extend(goal_directed_candidates(state, max_steps, goal_hints));
    }

    candidates
}

fn derive_goal_hints(goal: &crate::prediction::PlanGoal, state: &WorldState) -> Vec<GoalHint> {
    use crate::prediction::PlanGoal;

    match goal {
        PlanGoal::Condition(condition) => condition_hints(condition, state),
        PlanGoal::TargetState(target) => target_state_hints(target, state),
        PlanGoal::Description(description) => parse_description_goal(description, state),
        PlanGoal::GoalImage(image) => {
            let Some(target) = goal_image::goal_image_target(image, state) else {
                return Vec::new();
            };

            if let Some(object_id) = primary_dynamic_object(state).map(|object| object.id) {
                vec![GoalHint::ObjectAt {
                    object_id,
                    _object_name: state
                        .scene
                        .get_object(&object_id)
                        .map(|object| object.name.clone())
                        .unwrap_or_else(|| object_id.to_string()),
                    target: target.position,
                    tolerance: 0.15f32.max(0.05 + (1.0 - target.confidence) * 0.2),
                }]
            } else {
                vec![GoalHint::ObjectExistsAt {
                    object_name: "goal-image-object".to_string(),
                    target: target.position,
                    tolerance: 0.15f32.max(0.05 + (1.0 - target.confidence) * 0.2),
                }]
            }
        }
    }
}

fn condition_hints(condition: &Condition, state: &WorldState) -> Vec<GoalHint> {
    match condition {
        Condition::ObjectAt {
            object,
            position,
            tolerance,
        } => {
            let object_name = state
                .scene
                .get_object(object)
                .map(|item| item.name.clone())
                .unwrap_or_else(|| object.to_string());
            vec![GoalHint::ObjectAt {
                object_id: *object,
                _object_name: object_name,
                target: *position,
                tolerance: *tolerance,
            }]
        }
        Condition::ObjectExists { object } => state
            .scene
            .get_object(object)
            .map(|item| {
                vec![GoalHint::ObjectExists {
                    object_name: item.name.clone(),
                }]
            })
            .unwrap_or_default(),
        Condition::ObjectsTouching { a, b } => vec![GoalHint::ObjectsTouching { a: *a, b: *b }],
        Condition::And(conditions) | Condition::Or(conditions) => conditions
            .iter()
            .flat_map(|condition| condition_hints(condition, state))
            .collect(),
        Condition::Not(inner) => match inner.as_ref() {
            Condition::ObjectExists { object } => state
                .scene
                .get_object(object)
                .map(|item| {
                    vec![GoalHint::ObjectMissing {
                        object_id: *object,
                        _object_name: item.name.clone(),
                    }]
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        },
    }
}

fn target_state_hints(target: &WorldState, state: &WorldState) -> Vec<GoalHint> {
    let mut hints = Vec::new();
    for (object_id, target_object) in &target.scene.objects {
        if state.scene.get_object(object_id).is_some() {
            hints.push(GoalHint::ObjectAt {
                object_id: *object_id,
                _object_name: target_object.name.clone(),
                target: target_object.pose.position,
                tolerance: 0.15,
            });
        } else {
            hints.push(GoalHint::ObjectExists {
                object_name: target_object.name.clone(),
            });
        }
    }
    hints
}

fn parse_description_goal(description: &str, state: &WorldState) -> Vec<GoalHint> {
    let normalized = description.to_lowercase();
    let mut hints = Vec::new();

    if let Some((weather, _)) = parse_weather_hint(&normalized) {
        hints.push(GoalHint::Weather { weather });
    }
    if let Some(time_of_day) = parse_lighting_hint(&normalized) {
        hints.push(GoalHint::Lighting { time_of_day });
    }

    let mentioned_objects = mentioned_objects(state, &normalized);
    let relative_target = parse_relative_target_hint(state, &normalized);
    if normalized.contains("touch") && mentioned_objects.len() >= 2 {
        hints.push(GoalHint::ObjectsTouching {
            a: mentioned_objects[0].id,
            b: mentioned_objects[1].id,
        });
    }

    if contains_any(&normalized, &["remove", "delete", "discard"]) {
        if let Some(object) = mentioned_objects.first() {
            hints.push(GoalHint::ObjectMissing {
                object_id: object.id,
                _object_name: object.name.clone(),
            });
        }
    }

    if contains_any(&normalized, &["spawn", "create", "add"]) {
        let object_name = infer_object_name_from_verb(description, &["spawn", "create", "add"])
            .or_else(|| {
                mentioned_objects
                    .iter()
                    .find(|object| Some(object.id) != relative_target.map(|hint| hint.anchor_id))
                    .map(|object| object.name.clone())
            })
            .unwrap_or_else(|| "object".to_string());

        if let Some(relative_target) = relative_target {
            hints.push(GoalHint::ObjectExistsAt {
                object_name,
                target: relative_target.target,
                tolerance: relative_target.tolerance,
            });
        } else {
            hints.push(GoalHint::ObjectExists { object_name });
        }
    } else if let Some(relative_target) = relative_target {
        let subject = mentioned_objects
            .iter()
            .copied()
            .find(|object| object.id != relative_target.anchor_id)
            .or_else(|| {
                primary_dynamic_object(state)
                    .filter(|object| object.id != relative_target.anchor_id)
            });
        if let Some(subject) = subject {
            hints.push(GoalHint::ObjectAt {
                object_id: subject.id,
                _object_name: subject.name.clone(),
                target: relative_target.target,
                tolerance: relative_target.tolerance,
            });
        }
    }

    if let Some(target) = parse_position_hint(description) {
        let object = mentioned_objects
            .first()
            .copied()
            .or_else(|| primary_dynamic_object(state));
        if let Some(object) = object {
            hints.push(GoalHint::ObjectAt {
                object_id: object.id,
                _object_name: object.name.clone(),
                target,
                tolerance: 0.2,
            });
        }
    }

    hints
}

fn parse_weather_hint(input: &str) -> Option<(Weather, &'static str)> {
    [
        (Weather::Rain, "rain"),
        (Weather::Snow, "snow"),
        (Weather::Fog, "fog"),
        (Weather::Cloudy, "cloudy"),
        (Weather::Night, "night"),
        (Weather::Clear, "clear"),
    ]
    .into_iter()
    .find(|(_, needle)| input.contains(needle))
}

fn parse_lighting_hint(input: &str) -> Option<f32> {
    let marker = ["lighting", "time of day", "time"];
    for needle in marker {
        if let Some(index) = input.find(needle) {
            let suffix = &input[index + needle.len()..];
            if let Some(value) = first_number(suffix) {
                return Some(value.clamp(0.0, 24.0));
            }
        }
    }
    None
}

fn parse_position_hint(input: &str) -> Option<Position> {
    if let (Some(start), Some(end)) = (input.find('('), input.rfind(')')) {
        if start < end {
            let values: Vec<f32> = input[start + 1..end]
                .split([',', ' '])
                .filter(|token| !token.trim().is_empty())
                .filter_map(|token| token.trim().parse::<f32>().ok())
                .collect();
            if values.len() >= 3 {
                return Some(Position {
                    x: values[0],
                    y: values[1],
                    z: values[2],
                });
            }
        }
    }

    let words: Vec<&str> = input.split_whitespace().collect();
    for window in words.windows(4) {
        if matches!(window[0], "to" | "at" | "position") {
            let parsed = (
                window[1].parse::<f32>(),
                window[2].parse::<f32>(),
                window[3].parse::<f32>(),
            );
            if let (Ok(x), Ok(y), Ok(z)) = parsed {
                return Some(Position { x, y, z });
            }
        }
    }
    None
}

fn first_number(input: &str) -> Option<f32> {
    input.split_whitespace().find_map(|token| {
        token
            .trim_matches(|ch: char| !ch.is_ascii_digit() && ch != '.' && ch != '-')
            .parse::<f32>()
            .ok()
    })
}

fn infer_object_name_from_verb(input: &str, verbs: &[&str]) -> Option<String> {
    let normalized = input.to_lowercase();
    let delimiters = [
        " next to ",
        " beside ",
        " near ",
        " on top of ",
        " above ",
        " below ",
        " under ",
        " left of ",
        " right of ",
        " in front of ",
        " behind ",
        " at ",
        " to position ",
        " to (",
    ];
    for verb in verbs {
        if let Some(index) = normalized.find(verb) {
            let mut remainder = input[index + verb.len()..].trim();
            let mut remainder_lower = remainder.to_lowercase();

            for article in ["a ", "an ", "the "] {
                if remainder_lower.starts_with(article) {
                    remainder = remainder[article.len()..].trim_start();
                    remainder_lower = remainder.to_lowercase();
                    break;
                }
            }

            if remainder.is_empty() {
                continue;
            }

            let end_index = delimiters
                .iter()
                .filter_map(|delimiter| remainder_lower.find(delimiter))
                .min()
                .unwrap_or(remainder.len());
            let token = remainder[..end_index].trim().trim_matches(|ch: char| {
                !ch.is_alphanumeric() && ch != ' ' && ch != '_' && ch != '-'
            });
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

#[derive(Debug, Clone, Copy)]
enum RelativePlacement {
    NextTo,
    OnTopOf,
    Below,
    LeftOf,
    RightOf,
    InFrontOf,
    Behind,
}

#[derive(Debug, Clone, Copy)]
struct RelativeTargetHint {
    anchor_id: ObjectId,
    target: Position,
    tolerance: f32,
}

fn parse_relative_target_hint(state: &WorldState, description: &str) -> Option<RelativeTargetHint> {
    let mut best: Option<(usize, usize, RelativeTargetHint)> = None;

    for object in state.scene.objects.values() {
        let mut aliases = vec![object.name.to_lowercase()];
        if let Some(label) = &object.semantic_label {
            let lowered = label.to_lowercase();
            if !aliases.contains(&lowered) {
                aliases.push(lowered);
            }
        }

        for alias in aliases {
            for (phrase, placement) in [
                ("next to", RelativePlacement::NextTo),
                ("beside", RelativePlacement::NextTo),
                ("near", RelativePlacement::NextTo),
                ("on top of", RelativePlacement::OnTopOf),
                ("above", RelativePlacement::OnTopOf),
                ("below", RelativePlacement::Below),
                ("under", RelativePlacement::Below),
                ("left of", RelativePlacement::LeftOf),
                ("right of", RelativePlacement::RightOf),
                ("in front of", RelativePlacement::InFrontOf),
                ("behind", RelativePlacement::Behind),
            ] {
                let position = [
                    format!("{phrase} {alias}"),
                    format!("{phrase} the {alias}"),
                    format!("{phrase} a {alias}"),
                    format!("{phrase} an {alias}"),
                ]
                .into_iter()
                .filter_map(|pattern| description.find(&pattern))
                .min();
                let Some(position) = position else { continue };

                let target = relative_target_position(object, placement);
                let tolerance = (approximate_radius(object) + 0.15).clamp(0.15, 0.5);
                let candidate = (
                    position,
                    alias.len(),
                    RelativeTargetHint {
                        anchor_id: object.id,
                        target,
                        tolerance,
                    },
                );

                match best {
                    Some((best_position, best_alias_len, _)) => {
                        if position < best_position
                            || (position == best_position && alias.len() > best_alias_len)
                        {
                            best = Some(candidate);
                        }
                    }
                    None => best = Some(candidate),
                }
            }
        }
    }

    best.map(|(_, _, hint)| hint)
}

fn relative_target_position(anchor: &SceneObject, placement: RelativePlacement) -> Position {
    let clearance = (approximate_radius(anchor) + 0.2).clamp(0.2, 0.8);
    match placement {
        RelativePlacement::NextTo => Position {
            x: anchor.pose.position.x + clearance,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z,
        },
        RelativePlacement::OnTopOf => Position {
            x: anchor.pose.position.x,
            y: anchor.pose.position.y + clearance,
            z: anchor.pose.position.z,
        },
        RelativePlacement::Below => Position {
            x: anchor.pose.position.x,
            y: anchor.pose.position.y - clearance,
            z: anchor.pose.position.z,
        },
        RelativePlacement::LeftOf => Position {
            x: anchor.pose.position.x - clearance,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z,
        },
        RelativePlacement::RightOf => Position {
            x: anchor.pose.position.x + clearance,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z,
        },
        RelativePlacement::InFrontOf => Position {
            x: anchor.pose.position.x,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z + clearance,
        },
        RelativePlacement::Behind => Position {
            x: anchor.pose.position.x,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z - clearance,
        },
    }
}

fn object_name_matches(object: &SceneObject, query: &str) -> bool {
    let name = object.name.to_lowercase();
    if name == query || name.contains(query) {
        return true;
    }
    object
        .semantic_label
        .as_ref()
        .map(|label| {
            let label = label.to_lowercase();
            label == query || label.contains(query)
        })
        .unwrap_or(false)
}

fn mentioned_objects<'a>(state: &'a WorldState, description: &str) -> Vec<&'a SceneObject> {
    let mut objects: Vec<&SceneObject> = state
        .scene
        .objects
        .values()
        .filter(|object| {
            let name = object.name.to_lowercase();
            let label_match = object
                .semantic_label
                .as_ref()
                .map(|label| description.contains(&label.to_lowercase()))
                .unwrap_or(false);
            description.contains(&name) || label_match
        })
        .collect();
    objects.sort_by(|a, b| b.name.len().cmp(&a.name.len()));
    objects
}

fn primary_dynamic_object(state: &WorldState) -> Option<&SceneObject> {
    state
        .scene
        .objects
        .values()
        .filter(|object| !object.physics.is_static)
        .min_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.id.as_bytes().cmp(b.id.as_bytes()))
        })
}

fn score_goal_hints(goal_hints: &[GoalHint], state: &WorldState) -> Option<f32> {
    if goal_hints.is_empty() {
        return None;
    }

    let total = goal_hints
        .iter()
        .map(|hint| score_goal_hint(hint, state))
        .sum::<f32>();
    Some((total / goal_hints.len() as f32).clamp(0.0, 1.0))
}

fn score_goal_hint(goal_hint: &GoalHint, state: &WorldState) -> f32 {
    match goal_hint {
        GoalHint::ObjectAt {
            object_id,
            target,
            tolerance,
            ..
        } => state
            .scene
            .get_object(object_id)
            .map(|object| distance_score(object.pose.position, *target, *tolerance))
            .unwrap_or(0.0),
        GoalHint::ObjectMissing { object_id, .. } => {
            if state.scene.get_object(object_id).is_none() {
                1.0
            } else {
                0.0
            }
        }
        GoalHint::ObjectExists { object_name } => {
            if mentioned_objects(state, &object_name.to_lowercase()).is_empty() {
                0.0
            } else {
                1.0
            }
        }
        GoalHint::ObjectExistsAt {
            object_name,
            target,
            tolerance,
        } => {
            let query = object_name.to_lowercase();
            state
                .scene
                .objects
                .values()
                .filter(|object| object_name_matches(object, &query))
                .map(|object| distance_score(object.pose.position, *target, *tolerance))
                .fold(0.0, f32::max)
        }
        GoalHint::ObjectsTouching { a, b } => touching_score(state, *a, *b),
        GoalHint::Weather { weather } => {
            let expected = format!("weather:{weather:?}").to_lowercase();
            if state
                .metadata
                .tags
                .iter()
                .any(|tag| tag.to_lowercase() == expected)
            {
                1.0
            } else {
                0.0
            }
        }
        GoalHint::Lighting { time_of_day } => {
            let observed = state.metadata.tags.iter().find_map(|tag| {
                tag.to_lowercase()
                    .strip_prefix("lighting:")
                    .and_then(|value| value.parse::<f32>().ok())
            });
            observed
                .map(|value| distance_score_scalar(value, *time_of_day, 0.5))
                .unwrap_or(0.0)
        }
    }
}

fn touching_score(state: &WorldState, a: ObjectId, b: ObjectId) -> f32 {
    if state.scene.relationships.iter().any(|relationship| {
        matches!(relationship, crate::scene::SpatialRelationship::Touching { a: ra, b: rb } if (*ra == a && *rb == b) || (*ra == b && *rb == a))
    }) {
        return 1.0;
    }

    let Some(first) = state.scene.get_object(&a) else {
        return 0.0;
    };
    let Some(second) = state.scene.get_object(&b) else {
        return 0.0;
    };

    let radius = approximate_radius(first) + approximate_radius(second);
    distance_score(first.pose.position, second.pose.position, radius.max(0.05))
}

fn approximate_radius(object: &SceneObject) -> f32 {
    let dx = object.bbox.max.x - object.bbox.min.x;
    let dy = object.bbox.max.y - object.bbox.min.y;
    let dz = object.bbox.max.z - object.bbox.min.z;
    (dx.mul_add(dx, dy * dy) + dz * dz).sqrt() * 0.5
}

fn estimated_goal_alignment(
    goal_hints: &[GoalHint],
    initial_state: &WorldState,
    final_state: &WorldState,
) -> f32 {
    let initial = score_goal_hints(goal_hints, initial_state).unwrap_or(0.5);
    let final_score = score_goal_hints(goal_hints, final_state).unwrap_or(0.5);
    (final_score - initial + 0.5).clamp(0.0, 1.0)
}

fn default_spawn_position(state: &WorldState) -> Position {
    let baseline = primary_dynamic_object(state)
        .map(|object| object.pose.position)
        .unwrap_or_default();
    Position {
        x: baseline.x + 0.25,
        y: baseline.y.max(0.0) + 0.5,
        z: baseline.z,
    }
}

fn contains_any(input: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| input.contains(term))
}

fn normalize_vec3(vector: Vec3) -> Vec3 {
    let magnitude = (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z).sqrt();
    if magnitude <= f32::EPSILON {
        Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    } else {
        Vec3 {
            x: vector.x / magnitude,
            y: vector.y / magnitude,
            z: vector.z / magnitude,
        }
    }
}

fn direction_between(from: Position, to: Position) -> Vec3 {
    normalize_vec3(Vec3 {
        x: to.x - from.x,
        y: to.y - from.y,
        z: to.z - from.z,
    })
}

fn jitter_position(position: Position, rng: &mut PlannerRng, scale: f32) -> Position {
    Position {
        x: position.x + rng.range_f32(-scale, scale),
        y: position.y + rng.range_f32(-scale * 0.25, scale * 0.25),
        z: position.z + rng.range_f32(-scale, scale),
    }
}

fn interpolate_positions(from: Position, to: Position, steps: usize) -> Vec<Position> {
    (1..=steps.max(1))
        .map(|index| {
            let factor = index as f32 / steps.max(1) as f32;
            lerp_position(from, to, factor)
        })
        .collect()
}

fn lerp_position(from: Position, to: Position, factor: f32) -> Position {
    let factor = factor.clamp(0.0, 1.0);
    Position {
        x: from.x + (to.x - from.x) * factor,
        y: from.y + (to.y - from.y) * factor,
        z: from.z + (to.z - from.z) * factor,
    }
}

fn distance(from: Position, to: Position) -> f32 {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let dz = to.z - from.z;
    (dx.mul_add(dx, dy * dy) + dz * dz).sqrt()
}

fn distance_score(from: Position, to: Position, tolerance: f32) -> f32 {
    let tolerance = tolerance.max(0.05);
    let dist = distance(from, to);
    if dist <= tolerance {
        1.0
    } else {
        (1.0 / (1.0 + (dist - tolerance) / tolerance)).clamp(0.0, 1.0)
    }
}

fn distance_score_scalar(value: f32, target: f32, tolerance: f32) -> f32 {
    let delta = (value - target).abs();
    if delta <= tolerance {
        1.0
    } else {
        (1.0 / (1.0 + (delta - tolerance) / tolerance.max(0.1))).clamp(0.0, 1.0)
    }
}

fn length_bonus(length: usize) -> f32 {
    if length == 0 {
        1.0
    } else {
        (1.0 / length as f32).clamp(0.0, 1.0)
    }
}

fn planning_seed(goal_state: &WorldState, goal: &crate::prediction::PlanGoal) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    goal_state.id.hash(&mut hasher);
    goal_state.time.step.hash(&mut hasher);
    format!("{goal:?}").hash(&mut hasher);
    hasher.finish()
}

/// Evaluate how well a state satisfies a planning goal.
///
/// Returns a score between 0.0 (no progress) and 1.0 (goal achieved).
fn evaluate_goal_score(goal: &crate::prediction::PlanGoal, state: &WorldState) -> f32 {
    match goal {
        crate::prediction::PlanGoal::Condition(condition) => {
            if crate::action::evaluate_condition(condition, state) {
                1.0
            } else {
                score_goal_hints(&derive_goal_hints(goal, state), state).unwrap_or(0.0)
            }
        }
        crate::prediction::PlanGoal::GoalImage(image) => {
            let image_score = goal_image::goal_image_similarity(image, state).unwrap_or(0.0);
            let hint_score =
                score_goal_hints(&derive_goal_hints(goal, state), state).unwrap_or(image_score);
            ((image_score * 0.7) + (hint_score * 0.3)).clamp(0.0, 1.0)
        }
        _ => score_goal_hints(&derive_goal_hints(goal, state), state).unwrap_or(0.5),
    }
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
    fn test_generate_candidate_actions_empty_scene() {
        let state = WorldState::new("test", "mock");
        let candidates = generate_candidate_actions(&state, 1, 10, &[], 7);
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
        let candidates = generate_candidate_actions(&state, 2, 20, &[], 7);
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
    fn test_evaluate_goal_score_description_with_position_hint() {
        let (state, object_id) = sample_planning_state();
        let goal = PlanGoal::Description("move ball to position (2.0, 0.0, 0.0)".to_string());

        assert!(evaluate_goal_score(&goal, &state) < 1.0);

        let mut updated = state.clone();
        updated
            .scene
            .get_object_mut(&object_id)
            .unwrap()
            .pose
            .position = crate::types::Position {
            x: 2.0,
            y: 0.0,
            z: 0.0,
        };
        assert_eq!(evaluate_goal_score(&goal, &updated), 1.0);
    }

    #[test]
    fn test_evaluate_goal_score_relational_spawn_requires_spawned_object() {
        let (state, _ball_id, _mug_id) = sample_relational_state();
        let goal = PlanGoal::Description("spawn cube next to the red mug".to_string());
        assert_eq!(evaluate_goal_score(&goal, &state), 0.0);

        let target = parse_description_goal("spawn cube next to the red mug", &state)
            .into_iter()
            .find_map(|hint| {
                if let GoalHint::ObjectExistsAt { target, .. } = hint {
                    Some(target)
                } else {
                    None
                }
            })
            .unwrap();

        let mut updated = state.clone();
        let cube = crate::scene::SceneObject::new(
            "cube",
            crate::types::Pose {
                position: target,
                ..Default::default()
            },
            crate::types::BBox {
                min: crate::types::Position {
                    x: -0.1,
                    y: -0.1,
                    z: -0.1,
                },
                max: crate::types::Position {
                    x: 0.1,
                    y: 0.1,
                    z: 0.1,
                },
            },
        );
        updated.scene.add_object(cube);
        assert!(evaluate_goal_score(&goal, &updated) > 0.95);
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
    async fn test_sampling_plan_supports_goal_image() {
        let (state, object_id) = sample_planning_state();
        let initial_position = state.scene.get_object(&object_id).unwrap().pose.position;
        let mut target_state = state.clone();
        target_state
            .scene
            .get_object_mut(&object_id)
            .unwrap()
            .set_position(crate::types::Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            });
        let goal_image = crate::goal_image::render_scene_goal_image(&target_state, (32, 24));

        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(PlanningProvider::new("planner", 1.0, false)));
            registry
        });
        let world = World::new(state.clone(), "planner", registry);
        let request = PlanRequest {
            current_state: state,
            goal: PlanGoal::GoalImage(goal_image),
            max_steps: 3,
            guardrails: Vec::new(),
            planner: PlannerType::Sampling {
                num_samples: 16,
                top_k: 4,
            },
            timeout_seconds: 5.0,
        };

        let plan = world.plan(&request).await.unwrap();
        let final_state = plan.predicted_states.last().unwrap();
        let final_position = final_state
            .scene
            .get_object(&object_id)
            .unwrap()
            .pose
            .position;

        assert!(!plan.actions.is_empty());
        assert!(final_position.x > initial_position.x);
        assert!(plan.success_probability > 0.9);
    }

    #[tokio::test]
    async fn test_sampling_plan_moves_object_to_goal() {
        let (state, object_id) = sample_planning_state();
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(PlanningProvider::new("planner", 1.0, false)));
            registry
        });
        let world = World::new(state.clone(), "planner", registry);
        let request = PlanRequest {
            current_state: state,
            goal: PlanGoal::Description("move ball to position (2.0, 0.0, 0.0)".to_string()),
            max_steps: 3,
            guardrails: Vec::new(),
            planner: PlannerType::Sampling {
                num_samples: 16,
                top_k: 4,
            },
            timeout_seconds: 5.0,
        };

        let plan = world.plan(&request).await.unwrap();
        assert!(!plan.actions.is_empty());
        let final_state = plan.predicted_states.last().unwrap();
        let final_position = final_state
            .scene
            .get_object(&object_id)
            .unwrap()
            .pose
            .position;
        assert!(
            distance(
                final_position,
                crate::types::Position {
                    x: 2.0,
                    y: 0.0,
                    z: 0.0
                }
            ) < 0.25
        );
        assert!(plan.success_probability > 0.9);
    }

    #[tokio::test]
    async fn test_cem_plan_reports_iteration_count() {
        let (state, _object_id) = sample_planning_state();
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(PlanningProvider::new("planner", 1.0, false)));
            registry
        });
        let world = World::new(state.clone(), "planner", registry);
        let request = PlanRequest {
            current_state: state,
            goal: PlanGoal::Description("spawn cube".to_string()),
            max_steps: 2,
            guardrails: Vec::new(),
            planner: PlannerType::CEM {
                population_size: 12,
                elite_fraction: 0.25,
                num_iterations: 3,
            },
            timeout_seconds: 5.0,
        };

        let plan = world.plan(&request).await.unwrap();
        assert_eq!(plan.iterations_used, 3);
        assert!(matches!(
            plan.actions.first(),
            Some(Action::SpawnObject { .. })
        ));
    }

    #[tokio::test]
    async fn test_sampling_plan_spawns_relative_to_anchor() {
        let (state, _ball_id, mug_id) = sample_relational_state();
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(PlanningProvider::new("planner", 1.0, false)));
            registry
        });
        let world = World::new(state.clone(), "planner", registry);
        let request = PlanRequest {
            current_state: state,
            goal: PlanGoal::Description("spawn cube next to the red mug".to_string()),
            max_steps: 2,
            guardrails: Vec::new(),
            planner: PlannerType::Sampling {
                num_samples: 16,
                top_k: 4,
            },
            timeout_seconds: 5.0,
        };

        let plan = world.plan(&request).await.unwrap();
        assert!(matches!(
            plan.actions.first(),
            Some(Action::SpawnObject { template, .. }) if template == "cube"
        ));

        let final_state = plan.predicted_states.last().unwrap();
        let mug_position = final_state.scene.get_object(&mug_id).unwrap().pose.position;
        let cube = final_state.scene.find_object_by_name("cube").unwrap();
        assert!(cube.pose.position.x > mug_position.x);
        assert!(distance(cube.pose.position, mug_position) < 1.0);
        assert!(plan.success_probability > 0.9);
    }

    #[test]
    fn test_planning_spawn_object_bbox_tracks_pose() {
        let mut state = WorldState::new("spawn-bbox", "planner");
        let pose = Pose {
            position: Position {
                x: 1.25,
                y: 0.75,
                z: -0.5,
            },
            ..Pose::default()
        };

        apply_planning_action(
            &mut state,
            &Action::SpawnObject {
                template: "cube".to_string(),
                pose,
            },
            1.0,
        );

        let cube = state.scene.find_object_by_name("cube").unwrap();
        assert_eq!(cube.pose.position, pose.position);
        assert_eq!(cube.bbox.center(), pose.position);
    }

    #[tokio::test]
    async fn test_provider_native_uses_planning_capability() {
        let (state, object_id) = sample_planning_state();
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(PlanningProvider::new("planner", 0.5, true)));
            registry
        });
        let world = World::new(state.clone(), "planner", registry);
        let request = PlanRequest {
            current_state: state,
            goal: PlanGoal::Description("move ball to position (2.0, 0.0, 0.0)".to_string()),
            max_steps: 4,
            guardrails: Vec::new(),
            planner: PlannerType::ProviderNative,
            timeout_seconds: 5.0,
        };

        let plan = world.plan(&request).await.unwrap();
        let final_state = plan.predicted_states.last().unwrap();
        let final_position = final_state
            .scene
            .get_object(&object_id)
            .unwrap()
            .pose
            .position;
        assert_eq!(plan.iterations_used, 97);
        assert!(final_position.x > 1.5);
    }

    #[tokio::test]
    async fn test_provider_native_requires_provider_support() {
        let (state, _) = sample_planning_state();
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(PlanningProvider::new("planner", 1.0, false)));
            registry
        });
        let world = World::new(state.clone(), "planner", registry);
        let request = PlanRequest {
            current_state: state,
            goal: PlanGoal::Description("spawn cube".to_string()),
            max_steps: 2,
            guardrails: Vec::new(),
            planner: PlannerType::ProviderNative,
            timeout_seconds: 5.0,
        };

        let error = world.plan(&request).await.unwrap_err();
        assert!(matches!(
            error,
            WorldForgeError::UnsupportedCapability { provider, capability }
                if provider == "planner" && capability == "native planning"
        ));
    }

    #[tokio::test]
    async fn test_provider_native_rejects_malformed_plan() {
        let state = WorldState::new("planning", "planner");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(MalformedPlanProvider::new("planner")));
            registry
        });
        let world = World::new(state.clone(), "planner", registry);
        let request = PlanRequest {
            current_state: state,
            goal: PlanGoal::Description("set weather fog".to_string()),
            max_steps: 1,
            guardrails: Vec::new(),
            planner: PlannerType::ProviderNative,
            timeout_seconds: 5.0,
        };

        let error = world.plan(&request).await.unwrap_err();
        assert!(matches!(
            error,
            WorldForgeError::PlanningFailed { reason }
                if reason.contains("predicted states")
        ));
    }

    #[tokio::test]
    async fn test_execute_plan_commits_final_state_and_history() {
        let (state, object_id) = sample_planning_state();
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(PlanningProvider::new("planner", 1.0, false)));
            registry
        });
        let planner_world = World::new(state.clone(), "planner", std::sync::Arc::clone(&registry));
        let request = PlanRequest {
            current_state: state.clone(),
            goal: PlanGoal::Description("move ball to position (2.0, 0.0, 0.0)".to_string()),
            max_steps: 3,
            guardrails: Vec::new(),
            planner: PlannerType::Sampling {
                num_samples: 16,
                top_k: 4,
            },
            timeout_seconds: 5.0,
        };
        let plan = planner_world.plan(&request).await.unwrap();

        let mut world = World::new(state, "planner", registry);
        let execution = world
            .execute_plan(&plan, &PredictionConfig::default())
            .await
            .unwrap();

        let final_position = world
            .current_state()
            .scene
            .get_object(&object_id)
            .unwrap()
            .pose
            .position;
        assert_eq!(execution.predictions.len(), plan.actions.len());
        assert_eq!(world.current_state().history.len(), plan.actions.len() + 1);
        assert_eq!(world.current_state().time.step, plan.actions.len() as u64);
        assert_eq!(execution.final_state.time.step, plan.actions.len() as u64);
        assert!(final_position.x > 1.5);
    }

    #[tokio::test]
    async fn test_execute_plan_is_atomic_on_guardrail_failure() {
        let (state, object_id) = sample_planning_state();
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(PlanningProvider::new("planner", 1.0, false)));
            registry
        });
        let initial_position = state.scene.get_object(&object_id).unwrap().pose.position;
        let mut world = World::new(state, "planner", registry);
        let plan = Plan {
            actions: vec![Action::Move {
                target: Position {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                speed: 1.0,
            }],
            predicted_states: Vec::new(),
            predicted_videos: None,
            total_cost: 0.0,
            success_probability: 1.0,
            guardrail_compliance: Vec::new(),
            planning_time_ms: 0,
            iterations_used: 1,
        };
        let config = PredictionConfig {
            guardrails: vec![crate::guardrail::GuardrailConfig {
                guardrail: crate::guardrail::Guardrail::BoundaryConstraint {
                    bounds: crate::types::BBox {
                        min: Position {
                            x: -0.25,
                            y: -0.25,
                            z: -0.25,
                        },
                        max: Position {
                            x: 0.25,
                            y: 0.25,
                            z: 0.25,
                        },
                    },
                },
                blocking: true,
            }],
            ..PredictionConfig::default()
        };

        let error = world.execute_plan(&plan, &config).await.unwrap_err();

        assert!(matches!(error, WorldForgeError::GuardrailBlocked { .. }));
        assert!(world.current_state().history.is_empty());
        assert_eq!(
            world
                .current_state()
                .scene
                .get_object(&object_id)
                .unwrap()
                .pose
                .position,
            initial_position
        );
    }

    #[tokio::test]
    async fn test_execute_plan_uses_fallback_provider() {
        let (state, object_id) = sample_planning_state();
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(FailingProvider::new("primary")));
            registry.register(Box::new(PlanningProvider::new("fallback", 1.0, false)));
            registry
        });
        let mut world = World::new(state, "primary", registry);
        let plan = Plan {
            actions: vec![Action::Move {
                target: Position {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                speed: 1.0,
            }],
            predicted_states: Vec::new(),
            predicted_videos: None,
            total_cost: 0.0,
            success_probability: 1.0,
            guardrail_compliance: Vec::new(),
            planning_time_ms: 0,
            iterations_used: 1,
        };
        let config = PredictionConfig {
            fallback_provider: Some("fallback".to_string()),
            ..PredictionConfig::default()
        };

        let execution = world.execute_plan(&plan, &config).await.unwrap();

        assert_eq!(execution.predictions.len(), 1);
        assert_eq!(execution.predictions[0].provider, "fallback");
        assert_eq!(
            world.current_state().history.latest().unwrap().provider,
            "fallback"
        );
        assert!(
            world
                .current_state()
                .scene
                .get_object(&object_id)
                .unwrap()
                .pose
                .position
                .x
                > 0.5
        );
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
        assert_eq!(world.current_state().history.len(), 2);
        let mut entries = world.current_state().history.states.iter();
        let initial = entries.next().unwrap();
        assert!(initial.action.is_none());
        assert!(initial.prediction.is_none());
        assert_eq!(
            world.current_state().history.latest().unwrap().provider,
            "fallback"
        );
    }

    #[tokio::test]
    async fn test_predict_multi_uses_fallback_provider() {
        let state = WorldState::new("fallback", "primary");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(FailingProvider::new("primary")));
            registry.register(Box::new(SuccessfulProvider::new("fallback")));
            registry
        });
        let world = World::new(state, "primary", registry);
        let action = Action::Move {
            target: crate::types::Position::default(),
            speed: 1.0,
        };
        let config = PredictionConfig {
            fallback_provider: Some("fallback".to_string()),
            ..PredictionConfig::default()
        };

        let multi = world
            .predict_multi(&action, &["primary"], &config)
            .await
            .unwrap();

        assert_eq!(multi.predictions.len(), 1);
        assert_eq!(multi.predictions[0].provider, "fallback");
    }

    #[tokio::test]
    async fn test_predict_multi_applies_guardrails_without_mutating_state() {
        let (state, object_id) = sample_planning_state();
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(PlanningProvider::new("planner", 1.0, false)));
            registry
        });
        let world = World::new(state, "planner", registry);
        let action = Action::Move {
            target: Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            speed: 1.0,
        };
        let config = PredictionConfig {
            guardrails: vec![crate::guardrail::GuardrailConfig {
                guardrail: crate::guardrail::Guardrail::BoundaryConstraint {
                    bounds: crate::types::BBox {
                        min: Position {
                            x: -0.25,
                            y: -0.25,
                            z: -0.25,
                        },
                        max: Position {
                            x: 0.25,
                            y: 0.25,
                            z: 0.25,
                        },
                    },
                },
                blocking: true,
            }],
            ..PredictionConfig::default()
        };

        let error = world
            .predict_multi(&action, &["planner"], &config)
            .await
            .unwrap_err();

        assert!(matches!(error, WorldForgeError::GuardrailBlocked { .. }));
        assert!(world.current_state().history.is_empty());
        assert_eq!(
            world
                .current_state()
                .scene
                .get_object(&object_id)
                .unwrap()
                .pose
                .position,
            Position::default()
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

    #[tokio::test]
    async fn test_predict_applies_default_guardrails() {
        let state = WorldState::new("default-guardrails", "mock");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(SuccessfulProvider::new("mock")));
            registry
        });
        let mut world = World::new(state, "mock", registry);
        let action = Action::SetWeather {
            weather: Weather::Rain,
        };

        let prediction = world
            .predict(&action, &PredictionConfig::default())
            .await
            .unwrap();

        assert_eq!(prediction.guardrail_results.len(), 2);
        assert_eq!(
            prediction.guardrail_results[0].guardrail_name,
            "NoCollisions"
        );
        assert_eq!(
            prediction.guardrail_results[1].guardrail_name,
            "EnergyConservation"
        );
    }

    #[tokio::test]
    async fn test_predict_disable_guardrails_skips_defaults() {
        let state = WorldState::new("disabled-guardrails", "mock");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(SuccessfulProvider::new("mock")));
            registry
        });
        let mut world = World::new(state, "mock", registry);
        let action = Action::SetWeather {
            weather: Weather::Rain,
        };

        let prediction = world
            .predict(&action, &PredictionConfig::default().disable_guardrails())
            .await
            .unwrap();

        assert!(prediction.guardrail_results.is_empty());
    }

    #[tokio::test]
    async fn test_generate_uses_default_provider() {
        let state = WorldState::new("media", "media");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(MediaProvider::new("media")));
            registry
        });
        let world = World::new(state, "media", registry);
        let prompt = GenerationPrompt {
            text: "a rolling sphere".to_string(),
            reference_image: None,
            negative_prompt: Some("low quality".to_string()),
        };
        let config = GenerationConfig {
            duration_seconds: 6.0,
            ..GenerationConfig::default()
        };

        let clip = world.generate(&prompt, &config).await.unwrap();

        assert_eq!(clip.duration, 6.0);
        assert_eq!(clip.resolution, config.resolution);
    }

    #[tokio::test]
    async fn test_generate_uses_fallback_provider() {
        let state = WorldState::new("media-fallback", "missing");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(MediaProvider::new("fallback")));
            registry
        });
        let world = World::new(state, "missing", registry);
        let prompt = GenerationPrompt {
            text: "a rolling sphere".to_string(),
            reference_image: None,
            negative_prompt: None,
        };
        let config = GenerationConfig {
            duration_seconds: 3.5,
            ..GenerationConfig::default()
        };

        let (provider, clip) = world
            .generate_with_provider_and_fallback(&prompt, &config, "missing", Some("fallback"))
            .await
            .unwrap();

        assert_eq!(provider, "fallback");
        assert_eq!(clip.duration, 3.5);
    }

    #[tokio::test]
    async fn test_transfer_uses_default_provider() {
        let state = WorldState::new("transfer", "media");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(MediaProvider::new("media")));
            registry
        });
        let world = World::new(state, "media", registry);
        let source = VideoClip {
            frames: Vec::new(),
            fps: 12.0,
            resolution: (640, 360),
            duration: 5.0,
        };
        let config = TransferConfig {
            resolution: (800, 600),
            fps: 18.0,
            control_strength: 0.6,
        };

        let clip = world
            .transfer(&source, &SpatialControls::default(), &config)
            .await
            .unwrap();

        assert_eq!(clip.duration, source.duration);
        assert_eq!(clip.resolution, source.resolution);
        assert_eq!(clip.fps, source.fps);
    }

    #[tokio::test]
    async fn test_transfer_uses_fallback_provider() {
        let state = WorldState::new("transfer-fallback", "missing");
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(MediaProvider::new("fallback")));
            registry
        });
        let world = World::new(state, "missing", registry);
        let source = VideoClip {
            frames: Vec::new(),
            fps: 10.0,
            resolution: (320, 180),
            duration: 2.0,
        };
        let config = TransferConfig {
            resolution: (1024, 768),
            fps: 30.0,
            control_strength: 0.75,
        };

        let (provider, clip) = world
            .transfer_with_provider_and_fallback(
                &source,
                &SpatialControls::default(),
                &config,
                "missing",
                Some("fallback"),
            )
            .await
            .unwrap();

        assert_eq!(provider, "fallback");
        assert_eq!(clip.duration, source.duration);
        assert_eq!(clip.resolution, source.resolution);
    }

    #[tokio::test]
    async fn test_reason_uses_current_state() {
        let mut state = WorldState::new("reasoning", "media");
        let object = crate::scene::SceneObject::new(
            "cube",
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
        state.scene.add_object(object);
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(MediaProvider::new("media")));
            registry
        });
        let world = World::new(state, "media", registry);

        let output = world.reason("what objects are present?").await.unwrap();

        assert!(output.answer.contains("1 object"));
        assert!(output.evidence.iter().any(|item| item.contains("cube")));
    }

    #[tokio::test]
    async fn test_reason_uses_fallback_provider() {
        let mut state = WorldState::new("reasoning-fallback", "missing");
        state.scene.add_object(crate::scene::SceneObject::new(
            "cube",
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
        ));
        let registry = std::sync::Arc::new({
            let mut registry = ProviderRegistry::new();
            registry.register(Box::new(MediaProvider::new("fallback")));
            registry
        });
        let world = World::new(state, "missing", registry);

        let (provider, output) = world
            .reason_with_provider_and_fallback(
                "what objects are present?",
                "missing",
                Some("fallback"),
            )
            .await
            .unwrap();

        assert_eq!(provider, "fallback");
        assert!(output.answer.contains("1 object"));
        assert!(output.evidence.iter().any(|item| item.contains("cube")));
    }

    #[test]
    fn test_add_and_list_objects() {
        let registry = std::sync::Arc::new(ProviderRegistry::new());
        let mut world = World::new(WorldState::new("scene-edit", "mock"), "mock", registry);

        let crate_object = SceneObject::new(
            "crate",
            Pose::default(),
            crate::types::BBox {
                min: Position {
                    x: -0.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );
        let sphere = SceneObject::new(
            "sphere",
            Pose::default(),
            crate::types::BBox {
                min: Position {
                    x: -0.25,
                    y: -0.25,
                    z: -0.25,
                },
                max: Position {
                    x: 0.25,
                    y: 0.25,
                    z: 0.25,
                },
            },
        );

        world.add_object(sphere).unwrap();
        world.add_object(crate_object).unwrap();

        let objects = world.list_objects();
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].name, "crate");
        assert_eq!(objects[1].name, "sphere");
        assert!(world.get_object_by_name("crate").is_some());
    }

    #[test]
    fn test_add_object_rejects_duplicate_id() {
        let registry = std::sync::Arc::new(ProviderRegistry::new());
        let mut world = World::new(WorldState::new("dup", "mock"), "mock", registry);

        let object = SceneObject::new(
            "cube",
            Pose::default(),
            crate::types::BBox {
                min: Position {
                    x: -0.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );

        let duplicate = object.clone();
        world.add_object(object).unwrap();
        let error = world.add_object(duplicate).unwrap_err();
        assert!(matches!(error, WorldForgeError::InvalidState(_)));
    }

    #[test]
    fn test_replace_object_updates_scene_in_place() {
        let registry = std::sync::Arc::new(ProviderRegistry::new());
        let mut world = World::new(WorldState::new("replace", "mock"), "mock", registry);

        let table = SceneObject::new(
            "table",
            Pose::default(),
            crate::types::BBox {
                min: Position {
                    x: -0.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );
        let mut mug = SceneObject::new(
            "mug",
            Pose::default(),
            crate::types::BBox {
                min: Position {
                    x: -0.25,
                    y: -0.25,
                    z: -0.25,
                },
                max: Position {
                    x: 0.25,
                    y: 0.25,
                    z: 0.25,
                },
            },
        );
        mug.pose.position = Position {
            x: 0.0,
            y: 0.55,
            z: 0.0,
        };
        let object_id = table.id;
        world.add_object(table).unwrap();
        world.add_object(mug).unwrap();
        assert!(!world.state.scene.relationships.is_empty());

        let mut replacement = SceneObject::new(
            "table_updated",
            Pose::default(),
            crate::types::BBox {
                min: Position {
                    x: 9.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 10.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );
        replacement.id = object_id;
        replacement.pose.position = Position {
            x: 10.0,
            y: 0.0,
            z: 0.0,
        };

        let previous = world.replace_object(replacement).unwrap();
        assert_eq!(previous.name, "table");
        assert_eq!(world.get_object(&object_id).unwrap().name, "table_updated");
        assert_eq!(world.state.scene.root.children.len(), 2);
        assert_eq!(world.get_object(&object_id).unwrap().pose.position.x, 10.0);
    }

    #[test]
    fn test_remove_object() {
        let registry = std::sync::Arc::new(ProviderRegistry::new());
        let mut world = World::new(WorldState::new("remove", "mock"), "mock", registry);

        let object = SceneObject::new(
            "cube",
            Pose::default(),
            crate::types::BBox {
                min: Position {
                    x: -0.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );
        let object_id = object.id;
        world.add_object(object).unwrap();

        let removed = world.remove_object(&object_id).unwrap();
        assert_eq!(removed.name, "cube");
        assert!(world.get_object(&object_id).is_none());
    }

    #[test]
    fn test_update_object() {
        let registry = std::sync::Arc::new(ProviderRegistry::new());
        let mut world = World::new(WorldState::new("update", "mock"), "mock", registry);

        let object = SceneObject::new(
            "cube",
            Pose::default(),
            crate::types::BBox {
                min: Position {
                    x: -0.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );
        let object_id = object.id;
        world.add_object(object).unwrap();

        let updated = world
            .update_object(
                &object_id,
                SceneObjectPatch {
                    name: Some("cube_renamed".to_string()),
                    position: Some(Position {
                        x: 2.0,
                        y: 1.0,
                        z: -1.0,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(updated.id, object_id);
        assert_eq!(updated.name, "cube_renamed");
        assert_eq!(updated.pose.position.x, 2.0);
        assert_eq!(
            world
                .current_state()
                .scene
                .get_object(&object_id)
                .unwrap()
                .name,
            "cube_renamed"
        );
        assert_eq!(world.current_state().history.len(), 0);
        assert_eq!(world.current_state().time.step, 0);
    }

    #[test]
    fn test_update_object_missing() {
        let registry = std::sync::Arc::new(ProviderRegistry::new());
        let mut world = World::new(WorldState::new("missing", "mock"), "mock", registry);

        let error = world
            .update_object(
                &uuid::Uuid::new_v4(),
                SceneObjectPatch {
                    position: Some(Position {
                        x: 1.0,
                        y: 2.0,
                        z: 3.0,
                    }),
                    ..Default::default()
                },
            )
            .unwrap_err();

        assert!(matches!(error, WorldForgeError::InvalidState(_)));
    }

    #[test]
    fn test_restore_history_rewinds_world_state_and_provider() {
        let registry = std::sync::Arc::new(ProviderRegistry::new());
        let mut world = World::new(WorldState::new("restore-world", "mock"), "mock", registry);

        world.state.ensure_history_initialized("mock").unwrap();
        world.state.time = SimTime {
            step: 1,
            seconds: 0.5,
            dt: 0.5,
        };
        world.state.metadata.name = "restore-world-step-1".to_string();
        world
            .state
            .record_current_state(None, None, "backup")
            .unwrap();

        world.state.time = SimTime {
            step: 2,
            seconds: 1.0,
            dt: 0.5,
        };
        world.state.metadata.name = "restore-world-step-2".to_string();
        world
            .state
            .record_current_state(None, None, "mock")
            .unwrap();

        let checkpoint = world.history_state(1).unwrap();
        assert_eq!(checkpoint.time.step, 1);
        assert_eq!(checkpoint.metadata.name, "restore-world-step-1");
        assert_eq!(checkpoint.history.len(), 2);

        world.restore_history(1).unwrap();
        assert_eq!(world.current_state().time.step, 1);
        assert_eq!(world.current_state().metadata.name, "restore-world-step-1");
        assert_eq!(world.current_state().history.len(), 2);
        assert_eq!(world.default_provider, "backup");
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

    #[derive(Debug, Clone)]
    struct PlanningProvider {
        name: String,
        movement_scale: f32,
        supports_planning: bool,
    }

    #[derive(Debug, Clone)]
    struct MalformedPlanProvider {
        name: String,
    }

    #[derive(Debug, Clone)]
    struct MediaProvider {
        name: String,
    }

    impl PlanningProvider {
        fn new(name: &str, movement_scale: f32, supports_planning: bool) -> Self {
            Self {
                name: name.to_string(),
                movement_scale,
                supports_planning,
            }
        }
    }

    impl MediaProvider {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl MalformedPlanProvider {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    fn test_capabilities(supports_planning: bool) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: false,
            reason: false,
            transfer: false,
            embed: false,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 4.0,
            max_resolution: (640, 480),
            fps_range: (1.0, 30.0),
            supported_action_spaces: Vec::new(),
            supports_depth: false,
            supports_segmentation: false,
            supports_planning,
            latency_profile: LatencyProfile {
                p50_ms: 1,
                p95_ms: 1,
                p99_ms: 1,
                throughput_fps: 1.0,
            },
        }
    }

    fn media_capabilities() -> ProviderCapabilities {
        ProviderCapabilities {
            generate: true,
            reason: true,
            transfer: true,
            ..test_capabilities(false)
        }
    }

    fn sample_planning_state() -> (WorldState, uuid::Uuid) {
        let mut state = WorldState::new("planning", "planner");
        let object = crate::scene::SceneObject::new(
            "ball",
            crate::types::Pose::default(),
            crate::types::BBox {
                min: crate::types::Position {
                    x: -0.1,
                    y: -0.1,
                    z: -0.1,
                },
                max: crate::types::Position {
                    x: 0.1,
                    y: 0.1,
                    z: 0.1,
                },
            },
        );
        let object_id = object.id;
        state.scene.add_object(object);
        (state, object_id)
    }

    fn sample_relational_state() -> (WorldState, uuid::Uuid, uuid::Uuid) {
        let mut state = WorldState::new("relational", "planner");

        let ball = crate::scene::SceneObject::new(
            "ball",
            crate::types::Pose {
                position: crate::types::Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                ..Default::default()
            },
            crate::types::BBox::from_center_half_extents(
                crate::types::Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                crate::types::Vec3 {
                    x: 0.1,
                    y: 0.1,
                    z: 0.1,
                },
            ),
        );
        let ball_id = ball.id;
        state.scene.add_object(ball);

        let mut mug = crate::scene::SceneObject::new(
            "red mug",
            crate::types::Pose {
                position: crate::types::Position {
                    x: 0.75,
                    y: 0.0,
                    z: 0.0,
                },
                ..Default::default()
            },
            crate::types::BBox::from_center_half_extents(
                crate::types::Position {
                    x: 0.75,
                    y: 0.0,
                    z: 0.0,
                },
                crate::types::Vec3 {
                    x: 0.08,
                    y: 0.08,
                    z: 0.08,
                },
            ),
        );
        mug.semantic_label = Some("mug".to_string());
        let mug_id = mug.id;
        state.scene.add_object(mug);

        (state, ball_id, mug_id)
    }

    fn apply_planning_action(state: &mut WorldState, action: &Action, movement_scale: f32) {
        match action {
            Action::Place { object, target } => {
                if let Some(item) = state.scene.get_object_mut(object) {
                    item.pose.position = lerp_position(item.pose.position, *target, movement_scale);
                }
            }
            Action::Move { target, .. } => {
                if let Some(item) = state.scene.objects.values_mut().next() {
                    item.pose.position = lerp_position(item.pose.position, *target, movement_scale);
                }
            }
            Action::Push {
                object,
                direction,
                force,
            } => {
                if let Some(item) = state.scene.get_object_mut(object) {
                    item.pose.position = Position {
                        x: item.pose.position.x + direction.x * force * movement_scale,
                        y: item.pose.position.y + direction.y * force * movement_scale,
                        z: item.pose.position.z + direction.z * force * movement_scale,
                    };
                }
            }
            Action::RemoveObject { object } => {
                state.scene.remove_object(object);
            }
            Action::SpawnObject { template, pose } => {
                let object = crate::scene::SceneObject::new(
                    template,
                    *pose,
                    crate::types::BBox::from_center_half_extents(
                        pose.position,
                        crate::types::Vec3 {
                            x: 0.1,
                            y: 0.1,
                            z: 0.1,
                        },
                    ),
                );
                state.scene.add_object(object);
            }
            Action::SetWeather { weather } => {
                state
                    .metadata
                    .tags
                    .retain(|tag| !tag.starts_with("weather:"));
                state.metadata.tags.push(format!("weather:{weather:?}"));
            }
            Action::SetLighting { time_of_day } => {
                state
                    .metadata
                    .tags
                    .retain(|tag| !tag.starts_with("lighting:"));
                state
                    .metadata
                    .tags
                    .push(format!("lighting:{time_of_day:.1}"));
            }
            Action::Sequence(actions) | Action::Parallel(actions) => {
                for action in actions {
                    apply_planning_action(state, action, movement_scale);
                }
            }
            Action::Conditional {
                condition,
                then,
                otherwise,
            } => {
                if crate::action::evaluate_condition(condition, state) {
                    apply_planning_action(state, then, movement_scale);
                } else if let Some(otherwise) = otherwise {
                    apply_planning_action(state, otherwise, movement_scale);
                }
            }
            _ => {}
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

    fn build_native_plan(request: &PlanRequest, movement_scale: f32, iterations_used: u32) -> Plan {
        let goal_hints = derive_goal_hints(&request.goal, &request.current_state);
        let mut state = request.current_state.clone();
        let mut actions = Vec::new();
        let mut predicted_states = Vec::new();

        for hint in goal_hints {
            if actions.len() >= request.max_steps as usize {
                break;
            }

            match hint {
                GoalHint::ObjectAt {
                    object_id,
                    target,
                    tolerance,
                    ..
                } => {
                    while actions.len() < request.max_steps as usize {
                        let distance_to_goal = state
                            .scene
                            .get_object(&object_id)
                            .map(|object| distance(object.pose.position, target))
                            .unwrap_or(f32::INFINITY);
                        if distance_to_goal <= tolerance {
                            break;
                        }
                        let action = Action::Place {
                            object: object_id,
                            target,
                        };
                        apply_planning_action(&mut state, &action, movement_scale);
                        actions.push(action);
                        predicted_states.push(state.clone());
                    }
                }
                GoalHint::ObjectExists { object_name } => {
                    let action = Action::SpawnObject {
                        template: object_name,
                        pose: Pose {
                            position: default_spawn_position(&state),
                            ..Pose::default()
                        },
                    };
                    apply_planning_action(&mut state, &action, movement_scale);
                    actions.push(action);
                    predicted_states.push(state.clone());
                }
                GoalHint::ObjectExistsAt {
                    object_name,
                    target,
                    ..
                } => {
                    let action = Action::SpawnObject {
                        template: object_name,
                        pose: Pose {
                            position: target,
                            ..Pose::default()
                        },
                    };
                    apply_planning_action(&mut state, &action, movement_scale);
                    actions.push(action);
                    predicted_states.push(state.clone());
                }
                GoalHint::ObjectMissing { object_id, .. } => {
                    let action = Action::RemoveObject { object: object_id };
                    apply_planning_action(&mut state, &action, movement_scale);
                    actions.push(action);
                    predicted_states.push(state.clone());
                }
                GoalHint::ObjectsTouching { a, b } => {
                    let Some(anchor) = state
                        .scene
                        .get_object(&b)
                        .map(|object| object.pose.position)
                    else {
                        continue;
                    };
                    let action = Action::Place {
                        object: a,
                        target: anchor,
                    };
                    apply_planning_action(&mut state, &action, movement_scale);
                    actions.push(action);
                    predicted_states.push(state.clone());
                }
                GoalHint::Weather { weather } => {
                    let action = Action::SetWeather { weather };
                    apply_planning_action(&mut state, &action, movement_scale);
                    actions.push(action);
                    predicted_states.push(state.clone());
                }
                GoalHint::Lighting { time_of_day } => {
                    let action = Action::SetLighting { time_of_day };
                    apply_planning_action(&mut state, &action, movement_scale);
                    actions.push(action);
                    predicted_states.push(state.clone());
                }
            }
        }

        Plan {
            actions,
            predicted_states,
            predicted_videos: None,
            total_cost: 0.0,
            success_probability: evaluate_goal_score(&request.goal, &state),
            guardrail_compliance: Vec::new(),
            planning_time_ms: 0,
            iterations_used,
        }
    }

    #[async_trait]
    impl WorldModelProvider for FailingProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            test_capabilities(false)
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
            test_capabilities(false)
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
            test_capabilities(false)
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

    #[async_trait]
    impl WorldModelProvider for PlanningProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            test_capabilities(self.supports_planning)
        }

        async fn predict(
            &self,
            state: &WorldState,
            action: &Action,
            _config: &PredictionConfig,
        ) -> Result<Prediction> {
            let mut output_state = state.clone();
            apply_planning_action(&mut output_state, action, self.movement_scale);
            Ok(Prediction {
                id: uuid::Uuid::new_v4(),
                provider: self.name.clone(),
                model: format!("{}-planner", self.name),
                input_state: state.clone(),
                action: action.clone(),
                output_state,
                video: None,
                confidence: 0.9,
                physics_scores: PhysicsScores {
                    overall: 0.95,
                    object_permanence: 0.95,
                    gravity_compliance: 0.9,
                    collision_accuracy: 0.9,
                    spatial_consistency: 0.95,
                    temporal_consistency: 0.95,
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
                message: "planning provider".to_string(),
                latency_ms: 1,
            })
        }

        async fn plan(&self, request: &PlanRequest) -> Result<Plan> {
            if !self.supports_planning {
                return Err(WorldForgeError::UnsupportedCapability {
                    provider: self.name.clone(),
                    capability: "native planning".to_string(),
                });
            }
            Ok(build_native_plan(request, self.movement_scale, 97))
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }

    #[async_trait]
    impl WorldModelProvider for MalformedPlanProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            test_capabilities(true)
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
                message: "malformed native planner".to_string(),
                latency_ms: 1,
            })
        }

        async fn plan(&self, _request: &PlanRequest) -> Result<Plan> {
            Ok(Plan {
                actions: vec![Action::SetWeather {
                    weather: Weather::Fog,
                }],
                predicted_states: Vec::new(),
                predicted_videos: None,
                total_cost: 0.0,
                success_probability: 0.5,
                guardrail_compliance: Vec::new(),
                planning_time_ms: 0,
                iterations_used: 1,
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }

    #[async_trait]
    impl WorldModelProvider for MediaProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn capabilities(&self) -> ProviderCapabilities {
            media_capabilities()
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
            config: &GenerationConfig,
        ) -> Result<VideoClip> {
            Ok(VideoClip {
                frames: Vec::new(),
                fps: config.fps,
                resolution: config.resolution,
                duration: config.duration_seconds,
            })
        }

        async fn reason(&self, input: &ReasoningInput, query: &str) -> Result<ReasoningOutput> {
            let object_names = input
                .state
                .as_ref()
                .map(|state| {
                    state
                        .scene
                        .objects
                        .values()
                        .map(|object| object.name.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let object_count = object_names.len();
            Ok(ReasoningOutput {
                answer: format!("Observed {object_count} object(s) while answering: {query}"),
                confidence: 0.88,
                evidence: object_names,
            })
        }

        async fn transfer(
            &self,
            source: &VideoClip,
            _controls: &SpatialControls,
            _config: &TransferConfig,
        ) -> Result<VideoClip> {
            Ok(source.clone())
        }

        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                healthy: true,
                message: "media provider".to_string(),
                latency_ms: 1,
            })
        }

        fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
            CostEstimate::default()
        }
    }
}
