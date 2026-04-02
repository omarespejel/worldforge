//! Planning algorithms for WorldForge (RFC-0016).
//!
//! Implements real planning algorithms that evaluate action sequences through
//! a pluggable evaluator interface, enabling integration with any
//! `WorldModelProvider::predict()` implementation.
//!
//! # Algorithms
//!
//! - **Sampling**: Random sampling of action sequences, return best.
//! - **CEM (Cross-Entropy Method)**: Iteratively refine action distributions
//!   by keeping elite trajectories.
//! - **MPC (Model Predictive Control)**: Rolling-horizon planning with
//!   re-planning after each execution step.

use std::time::Instant;

use crate::action::Action;
use crate::error::{Result, WorldForgeError};
use crate::guardrail::GuardrailResult;
use crate::prediction::{Plan, PlannerType};
use crate::state::WorldState;
use crate::types::Position;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Evaluates an action sequence from a given state and returns a scalar score.
///
/// Higher scores are better. Implementations typically call
/// `WorldModelProvider::predict()` in a loop and aggregate confidence /
/// physics scores.
#[async_trait::async_trait]
pub trait ActionEvaluator: Send + Sync {
    /// Score an action sequence starting from `state`.
    ///
    /// Returns `(score, predicted_states)` where `predicted_states[i]` is the
    /// world state after applying `actions[0..=i]`.
    async fn evaluate(
        &self,
        state: &WorldState,
        actions: &[Action],
    ) -> Result<(f32, Vec<WorldState>)>;
}

/// Samples a random action for a given world state.
///
/// Implementations should return actions drawn from the feasible action space
/// of the scene (e.g., moving existing objects to random positions).
pub trait ActionSampler: Send + Sync {
    /// Generate a random action given the current state and an RNG.
    fn sample(&self, state: &WorldState, rng: &mut SimpleRng) -> Action;
}

// ---------------------------------------------------------------------------
// Simple deterministic PRNG (no external dependency)
// ---------------------------------------------------------------------------

/// Minimal xorshift64 PRNG for planning algorithms.
///
/// This avoids pulling in an external `rand` crate while providing adequate
/// randomness for stochastic optimisation loops.
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with the given seed. A seed of 0 is replaced with 1.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Return the next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a `f32` in `[0.0, 1.0)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0x00FF_FFFF) as f32 / 16_777_216.0
    }

    /// Return an `f32` in `[lo, hi)`.
    pub fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Return a `usize` in `[0, n)`.
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n.max(1)
    }
}

// ---------------------------------------------------------------------------
// Default action sampler
// ---------------------------------------------------------------------------

/// Default action sampler that generates `Move` actions to random positions.
///
/// This is a simple baseline. Real deployments should provide a domain-specific
/// sampler that considers the actual action space.
#[derive(Debug, Clone)]
pub struct DefaultActionSampler {
    /// Spatial bounds for random positions `(min, max)`.
    pub bounds: (Position, Position),
}

impl Default for DefaultActionSampler {
    fn default() -> Self {
        Self {
            bounds: (
                Position {
                    x: -5.0,
                    y: 0.0,
                    z: -5.0,
                },
                Position {
                    x: 5.0,
                    y: 3.0,
                    z: 5.0,
                },
            ),
        }
    }
}

impl ActionSampler for DefaultActionSampler {
    fn sample(&self, _state: &WorldState, rng: &mut SimpleRng) -> Action {
        let target = Position {
            x: rng.next_f32_range(self.bounds.0.x, self.bounds.1.x),
            y: rng.next_f32_range(self.bounds.0.y, self.bounds.1.y),
            z: rng.next_f32_range(self.bounds.0.z, self.bounds.1.z),
        };
        let speed = rng.next_f32_range(0.5, 2.0);
        Action::Move { target, speed }
    }
}

// ---------------------------------------------------------------------------
// Planner output builder
// ---------------------------------------------------------------------------

fn build_plan(
    actions: Vec<Action>,
    predicted_states: Vec<WorldState>,
    score: f32,
    planning_time_ms: u64,
    iterations_used: u32,
) -> Plan {
    let num_steps = actions.len();
    Plan {
        actions,
        predicted_states,
        predicted_videos: None,
        total_cost: 0.0,
        success_probability: score.clamp(0.0, 1.0),
        guardrail_compliance: vec![Vec::<GuardrailResult>::new(); num_steps],
        planning_time_ms,
        iterations_used,
        stored_plan_id: None,
        verification_proof: None,
    }
}

// ---------------------------------------------------------------------------
// Sampling planner
// ---------------------------------------------------------------------------

/// Random-sampling planner: evaluate `num_samples` random trajectories and
/// return the best one.
pub async fn plan_sampling(
    evaluator: &dyn ActionEvaluator,
    sampler: &dyn ActionSampler,
    initial_state: &WorldState,
    num_samples: u32,
    top_k: u32,
    horizon: usize,
    seed: u64,
) -> Result<Plan> {
    let started = Instant::now();
    let mut rng = SimpleRng::new(seed);

    if horizon == 0 {
        return Err(WorldForgeError::InvalidState(
            "sampling planner requires horizon >= 1".to_string(),
        ));
    }

    let mut candidates: Vec<(f32, Vec<Action>, Vec<WorldState>)> =
        Vec::with_capacity(num_samples as usize);

    for _ in 0..num_samples {
        // Sample a random action sequence.
        let actions: Vec<Action> = (0..horizon)
            .map(|_| sampler.sample(initial_state, &mut rng))
            .collect();

        let (score, states) = evaluator.evaluate(initial_state, &actions).await?;
        candidates.push((score, actions, states));
    }

    // Sort descending by score.
    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(top_k as usize);

    let (best_score, best_actions, best_states) = candidates
        .into_iter()
        .next()
        .ok_or_else(|| WorldForgeError::NoFeasiblePlan {
            goal: "sampling".to_string(),
            reason: "no candidates generated".to_string(),
        })?;

    Ok(build_plan(
        best_actions,
        best_states,
        best_score,
        started.elapsed().as_millis() as u64,
        num_samples,
    ))
}

// ---------------------------------------------------------------------------
// Cross-Entropy Method (CEM) planner
// ---------------------------------------------------------------------------

/// Parameterised action distribution for CEM.
///
/// Each time step has a mean position and standard deviation for the `Move`
/// action target, plus mean speed. The distribution is Gaussian-like, fit to
/// the elite set each iteration.
#[derive(Debug, Clone)]
struct CemDistribution {
    /// Per-step mean target `(x, y, z)`.
    means: Vec<[f32; 4]>, // [x, y, z, speed]
    /// Per-step stddev.
    stddevs: Vec<[f32; 4]>,
}

impl CemDistribution {
    fn new(horizon: usize) -> Self {
        Self {
            means: vec![[0.0, 1.0, 0.0, 1.0]; horizon],
            stddevs: vec![[3.0, 1.5, 3.0, 0.5]; horizon],
        }
    }

    fn sample_sequence(&self, rng: &mut SimpleRng) -> Vec<Action> {
        self.means
            .iter()
            .zip(self.stddevs.iter())
            .map(|(mean, std)| {
                // Box-Muller-ish approximation using uniform samples.
                let params: [f32; 4] = std::array::from_fn(|i| {
                    // Sum of 4 uniforms approximates a Gaussian reasonably.
                    let u: f32 = (0..4).map(|_| rng.next_f32()).sum::<f32>() - 2.0; // ~N(0,1)
                    mean[i] + u * std[i]
                });
                Action::Move {
                    target: Position {
                        x: params[0],
                        y: params[1],
                        z: params[2],
                    },
                    speed: params[3].clamp(0.1, 5.0),
                }
            })
            .collect()
    }

    /// Re-fit the distribution to the elite set of action sequences.
    fn fit(&mut self, elite: &[Vec<Action>]) {
        if elite.is_empty() {
            return;
        }
        let n = elite.len() as f32;
        let horizon = self.means.len();

        for t in 0..horizon {
            let mut sum = [0.0f32; 4];
            for seq in elite {
                let params = action_to_params(&seq[t]);
                for (s, p) in sum.iter_mut().zip(params.iter()) {
                    *s += p;
                }
            }
            let mean: [f32; 4] = std::array::from_fn(|i| sum[i] / n);

            let mut var = [0.0f32; 4];
            for seq in elite {
                let params = action_to_params(&seq[t]);
                for (v, (p, m)) in var.iter_mut().zip(params.iter().zip(mean.iter())) {
                    *v += (p - m) * (p - m);
                }
            }
            let stddev: [f32; 4] = std::array::from_fn(|i| (var[i] / n).sqrt().max(0.01));

            self.means[t] = mean;
            self.stddevs[t] = stddev;
        }
    }
}

fn action_to_params(action: &Action) -> [f32; 4] {
    match action {
        Action::Move { target, speed } => [target.x, target.y, target.z, *speed],
        _ => [0.0, 1.0, 0.0, 1.0],
    }
}

/// Cross-Entropy Method planner.
///
/// 1. Initialise a distribution over action sequences.
/// 2. Sample `population_size` sequences.
/// 3. Evaluate each with the evaluator.
/// 4. Keep top `elite_fraction` (elite set).
/// 5. Re-fit the distribution to the elite set.
/// 6. Repeat for `num_iterations`.
/// 7. Return the best sequence found across all iterations.
pub async fn plan_cem(
    evaluator: &dyn ActionEvaluator,
    initial_state: &WorldState,
    population_size: u32,
    elite_fraction: f32,
    num_iterations: u32,
    horizon: usize,
    seed: u64,
) -> Result<Plan> {
    let started = Instant::now();

    if horizon == 0 {
        return Err(WorldForgeError::InvalidState(
            "CEM planner requires horizon >= 1".to_string(),
        ));
    }

    let elite_count = ((population_size as f32 * elite_fraction).ceil() as usize).max(1);
    let mut rng = SimpleRng::new(seed);
    let mut distribution = CemDistribution::new(horizon);

    let mut best_overall: Option<(f32, Vec<Action>, Vec<WorldState>)> = None;

    for _ in 0..num_iterations {
        let mut candidates: Vec<(f32, Vec<Action>, Vec<WorldState>)> =
            Vec::with_capacity(population_size as usize);

        for _ in 0..population_size {
            let actions = distribution.sample_sequence(&mut rng);
            let (score, states) = evaluator.evaluate(initial_state, &actions).await?;
            candidates.push((score, actions, states));
        }

        // Sort descending by score.
        candidates.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update global best.
        if let Some((score, actions, states)) = candidates.first() {
            if best_overall
                .as_ref()
                .is_none_or(|(best, _, _)| *score > *best)
            {
                best_overall = Some((*score, actions.clone(), states.clone()));
            }
        }

        // Extract elite set and re-fit distribution.
        let elite: Vec<Vec<Action>> = candidates
            .iter()
            .take(elite_count)
            .map(|(_, actions, _)| actions.clone())
            .collect();
        distribution.fit(&elite);
    }

    let (score, actions, states) =
        best_overall.ok_or_else(|| WorldForgeError::NoFeasiblePlan {
            goal: "cem".to_string(),
            reason: "CEM produced no candidates".to_string(),
        })?;

    let total_evals = population_size * num_iterations;

    Ok(build_plan(
        actions,
        states,
        score,
        started.elapsed().as_millis() as u64,
        total_evals,
    ))
}

// ---------------------------------------------------------------------------
// MPC planner
// ---------------------------------------------------------------------------

/// Model Predictive Control planner.
///
/// At each execution step:
/// 1. Plan over a horizon `H` using sampling.
/// 2. Execute the first `replanning_interval` actions.
/// 3. Observe the resulting state.
/// 4. Re-plan from the new state.
/// 5. Repeat until `max_steps` actions have been executed.
#[allow(clippy::too_many_arguments)]
pub async fn plan_mpc(
    evaluator: &dyn ActionEvaluator,
    sampler: &dyn ActionSampler,
    initial_state: &WorldState,
    horizon: u32,
    num_samples: u32,
    replanning_interval: u32,
    max_steps: u32,
    seed: u64,
) -> Result<Plan> {
    let started = Instant::now();
    let mut rng = SimpleRng::new(seed);

    if horizon == 0 || max_steps == 0 {
        return Err(WorldForgeError::InvalidState(
            "MPC planner requires horizon >= 1 and max_steps >= 1".to_string(),
        ));
    }

    let mut all_actions: Vec<Action> = Vec::new();
    let mut all_states: Vec<WorldState> = Vec::new();
    let mut current_state = initial_state.clone();
    let mut total_score: f32 = 0.0;
    let mut replan_count: u32 = 0;
    let replan_steps = (replanning_interval as usize).max(1);

    while all_actions.len() < max_steps as usize {
        let remaining = max_steps as usize - all_actions.len();
        let effective_horizon = (horizon as usize).min(remaining);

        // Use sampling to find best trajectory over the horizon.
        let mut best_score = f32::NEG_INFINITY;
        let mut best_actions: Vec<Action> = Vec::new();
        let mut best_states: Vec<WorldState> = Vec::new();

        for _ in 0..num_samples {
            let actions: Vec<Action> = (0..effective_horizon)
                .map(|_| sampler.sample(&current_state, &mut rng))
                .collect();
            let (score, states) = evaluator.evaluate(&current_state, &actions).await?;
            if score > best_score {
                best_score = score;
                best_actions = actions;
                best_states = states;
            }
        }

        if best_actions.is_empty() {
            break;
        }

        // Execute the first `replan_steps` actions from the best trajectory.
        let execute_count = replan_steps.min(best_actions.len());
        for i in 0..execute_count {
            all_actions.push(best_actions[i].clone());
            all_states.push(best_states[i].clone());
        }

        // Update current state to the state after executed actions.
        if let Some(last_state) = all_states.last() {
            current_state = last_state.clone();
        }

        total_score += best_score;
        replan_count += 1;
    }

    if all_actions.is_empty() {
        return Err(WorldForgeError::NoFeasiblePlan {
            goal: "mpc".to_string(),
            reason: "MPC produced no actions".to_string(),
        });
    }

    let avg_score = total_score / replan_count.max(1) as f32;

    Ok(build_plan(
        all_actions,
        all_states,
        avg_score,
        started.elapsed().as_millis() as u64,
        replan_count,
    ))
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Run a planning algorithm based on the `PlannerType` variant.
///
/// This dispatches to the appropriate planner implementation. For
/// `PlannerType::ProviderNative` and `PlannerType::Gradient`, callers should
/// handle those cases externally (this function returns an error).
pub async fn run_planner(
    planner_type: &PlannerType,
    evaluator: &dyn ActionEvaluator,
    sampler: &dyn ActionSampler,
    initial_state: &WorldState,
    max_steps: u32,
    seed: u64,
) -> Result<Plan> {
    match planner_type {
        PlannerType::Sampling { num_samples, top_k } => {
            plan_sampling(
                evaluator,
                sampler,
                initial_state,
                *num_samples,
                *top_k,
                max_steps as usize,
                seed,
            )
            .await
        }
        PlannerType::CEM {
            population_size,
            elite_fraction,
            num_iterations,
        } => {
            plan_cem(
                evaluator,
                initial_state,
                *population_size,
                *elite_fraction,
                *num_iterations,
                max_steps as usize,
                seed,
            )
            .await
        }
        PlannerType::MPC {
            horizon,
            num_samples,
            replanning_interval,
        } => {
            plan_mpc(
                evaluator,
                sampler,
                initial_state,
                *horizon,
                *num_samples,
                *replanning_interval,
                max_steps,
                seed,
            )
            .await
        }
        PlannerType::Gradient { .. } => Err(WorldForgeError::InvalidState(
            "gradient planner not yet implemented in planner module".to_string(),
        )),
        PlannerType::ProviderNative => Err(WorldForgeError::InvalidState(
            "provider-native planner must be handled by the provider adapter".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::WorldState;
    use crate::types::Position;

    /// Simple evaluator that scores action sequences by how close the
    /// first Move target is to a goal position.
    struct GoalEvaluator {
        goal: Position,
    }

    #[async_trait::async_trait]
    impl ActionEvaluator for GoalEvaluator {
        async fn evaluate(
            &self,
            state: &WorldState,
            actions: &[Action],
        ) -> Result<(f32, Vec<WorldState>)> {
            let mut current = state.clone();
            let mut states = Vec::with_capacity(actions.len());

            for _action in actions {
                // Simple simulation: just clone state (mock-like).
                let mut next = current.clone();
                next.time.step += 1;
                next.time.seconds += 0.1;
                states.push(next.clone());
                current = next;
            }

            // Score: inverse distance to goal from last Move target.
            let score = actions
                .iter()
                .filter_map(|a| match a {
                    Action::Move { target, .. } => {
                        let dx = target.x - self.goal.x;
                        let dy = target.y - self.goal.y;
                        let dz = target.z - self.goal.z;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        Some(1.0 / (1.0 + dist))
                    }
                    _ => None,
                })
                .fold(0.0f32, f32::max);

            Ok((score, states))
        }
    }

    fn test_state() -> WorldState {
        WorldState::new("test", "test-provider")
    }

    #[tokio::test]
    async fn test_sampling_planner_returns_plan() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let sampler = DefaultActionSampler::default();
        let state = test_state();

        let plan = plan_sampling(&evaluator, &sampler, &state, 16, 3, 4, 42)
            .await
            .unwrap();

        assert_eq!(plan.actions.len(), 4);
        assert_eq!(plan.predicted_states.len(), 4);
        assert!(plan.success_probability >= 0.0);
        assert!(plan.success_probability <= 1.0);
        assert!(plan.iterations_used == 16);
    }

    #[tokio::test]
    async fn test_sampling_planner_finds_better_with_more_samples() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let sampler = DefaultActionSampler::default();
        let state = test_state();

        let plan_few = plan_sampling(&evaluator, &sampler, &state, 4, 1, 2, 42)
            .await
            .unwrap();
        let plan_many = plan_sampling(&evaluator, &sampler, &state, 200, 1, 2, 42)
            .await
            .unwrap();

        // More samples should generally find a solution at least as good.
        assert!(plan_many.success_probability >= plan_few.success_probability * 0.8);
    }

    #[tokio::test]
    async fn test_cem_planner_returns_plan() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 2.0,
                y: 1.0,
                z: 0.0,
            },
        };
        let state = test_state();

        let plan = plan_cem(&evaluator, &state, 16, 0.25, 3, 3, 123)
            .await
            .unwrap();

        assert_eq!(plan.actions.len(), 3);
        assert_eq!(plan.predicted_states.len(), 3);
        assert!(plan.success_probability >= 0.0);
        assert!(plan.success_probability <= 1.0);
        // CEM should have done population_size * num_iterations evaluations.
        assert_eq!(plan.iterations_used, 48);
    }

    #[tokio::test]
    async fn test_cem_improves_over_iterations() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 1.0,
                y: 0.5,
                z: 0.0,
            },
        };
        let state = test_state();

        let plan_1iter = plan_cem(&evaluator, &state, 20, 0.2, 1, 2, 999)
            .await
            .unwrap();
        let plan_5iter = plan_cem(&evaluator, &state, 20, 0.2, 5, 2, 999)
            .await
            .unwrap();

        // More iterations should find a solution at least as good.
        assert!(plan_5iter.success_probability >= plan_1iter.success_probability * 0.9);
    }

    #[tokio::test]
    async fn test_mpc_planner_returns_plan() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 1.0,
                y: 0.0,
                z: 1.0,
            },
        };
        let sampler = DefaultActionSampler::default();
        let state = test_state();

        let plan = plan_mpc(&evaluator, &sampler, &state, 3, 8, 1, 5, 77)
            .await
            .unwrap();

        assert!(!plan.actions.is_empty());
        assert!(plan.actions.len() <= 5);
        assert_eq!(plan.actions.len(), plan.predicted_states.len());
        assert!(plan.success_probability >= 0.0);
    }

    #[tokio::test]
    async fn test_mpc_replanning_interval() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let sampler = DefaultActionSampler::default();
        let state = test_state();

        // With replanning_interval=2 and max_steps=6, we should get at most
        // 3 re-plan cycles, each executing 2 actions.
        let plan = plan_mpc(&evaluator, &sampler, &state, 4, 10, 2, 6, 55)
            .await
            .unwrap();

        assert!(plan.actions.len() <= 6);
        assert_eq!(plan.actions.len(), plan.predicted_states.len());
    }

    #[tokio::test]
    async fn test_run_planner_dispatch_sampling() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let sampler = DefaultActionSampler::default();
        let state = test_state();
        let planner = PlannerType::Sampling {
            num_samples: 8,
            top_k: 2,
        };

        let plan = run_planner(&planner, &evaluator, &sampler, &state, 3, 42)
            .await
            .unwrap();

        assert_eq!(plan.actions.len(), 3);
    }

    #[tokio::test]
    async fn test_run_planner_dispatch_cem() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let sampler = DefaultActionSampler::default();
        let state = test_state();
        let planner = PlannerType::CEM {
            population_size: 10,
            elite_fraction: 0.3,
            num_iterations: 2,
        };

        let plan = run_planner(&planner, &evaluator, &sampler, &state, 2, 42)
            .await
            .unwrap();

        assert_eq!(plan.actions.len(), 2);
    }

    #[tokio::test]
    async fn test_run_planner_dispatch_mpc() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let sampler = DefaultActionSampler::default();
        let state = test_state();
        let planner = PlannerType::MPC {
            horizon: 3,
            num_samples: 8,
            replanning_interval: 1,
        };

        let plan = run_planner(&planner, &evaluator, &sampler, &state, 4, 42)
            .await
            .unwrap();

        assert!(!plan.actions.is_empty());
        assert!(plan.actions.len() <= 4);
    }

    #[tokio::test]
    async fn test_run_planner_errors_on_native() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let sampler = DefaultActionSampler::default();
        let state = test_state();

        let result =
            run_planner(&PlannerType::ProviderNative, &evaluator, &sampler, &state, 3, 1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_sampling_zero_horizon_error() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let sampler = DefaultActionSampler::default();
        let state = test_state();

        let result = plan_sampling(&evaluator, &sampler, &state, 8, 1, 0, 42).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cem_zero_horizon_error() {
        let evaluator = GoalEvaluator {
            goal: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let state = test_state();

        let result = plan_cem(&evaluator, &state, 10, 0.2, 3, 0, 42).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_simple_rng_f32_range() {
        let mut rng = SimpleRng::new(12345);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "f32 out of range: {v}");
        }
    }

    #[test]
    fn test_default_action_sampler_produces_move() {
        let sampler = DefaultActionSampler::default();
        let state = WorldState::new("test", "test-provider");
        let mut rng = SimpleRng::new(99);

        for _ in 0..10 {
            let action = sampler.sample(&state, &mut rng);
            assert!(matches!(action, Action::Move { .. }));
        }
    }
}
