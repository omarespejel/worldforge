# RFC-0016: Planning System

- **Status:** Draft
- **Created:** 2026-04-02
- **Authors:** WorldForge Core Team
- **Requires:** RFC-0001 (Core Architecture), RFC-0003 (Provider System), RFC-0005 (Guardrails)

---

## Abstract

This RFC defines the planning system for WorldForge, which enables agents and
robots to plan sequences of actions by using world model predictions to evaluate
potential futures. The system implements four planner types: Cross-Entropy Method
(CEM), Sampling, Model Predictive Control (MPC), and Gradient-based planning.
Each planner uses the prediction engine to simulate action sequences, evaluates
outcomes against specified goals, and produces optimized plans with safety
guardrail integration. The system supports multi-provider planning, plan
execution monitoring, replanning on failure, and provides a clear integration
path for robotics applications via ROS2.

## Motivation

World models become truly useful when they enable planning -- deciding what
actions to take to achieve a desired goal. Current approaches to AI planning
either:

1. **Ignore physics entirely** (LLM-based planners generate plausible-sounding
   but physically impossible plans)
2. **Require hand-crafted physics engines** (game engines, rigid body simulators)
   that don't generalize to the real world
3. **Use model-free RL** which requires millions of environment interactions

WorldForge's planning system bridges this gap: it uses learned world models to
simulate action outcomes, enabling physically-grounded planning without
hand-crafted physics. A robot can plan how to stack blocks by actually
predicting what happens when it moves each block, rather than relying on
symbolic rules or exhaustive trial-and-error.

Key use cases:
- **Robotics**: Plan manipulation sequences (pick, place, push, pour)
- **Game AI**: Plan NPC actions in physically simulated game worlds
- **Autonomous vehicles**: Plan driving trajectories considering physics
- **Content creation**: Plan camera movements and scene editing sequences

## Detailed Design

### 1. Core Types

#### 1.1 Plan Structure

```rust
pub struct Plan {
    /// Ordered sequence of actions to execute
    pub actions: Vec<Action>,
    /// Predicted world states after each action
    pub predicted_states: Vec<WorldState>,
    /// Estimated probability of successfully reaching the goal
    pub success_probability: f64,
    /// Total expected cost/reward of the plan
    pub expected_reward: f64,
    /// Time budget consumed during planning
    pub planning_time: Duration,
    /// Which planner generated this plan
    pub planner_type: PlannerType,
    /// Guardrail check results for each action
    pub safety_checks: Vec<GuardrailResult>,
}

pub struct Action {
    /// Natural language description of the action
    pub description: String,
    /// Structured action parameters (optional)
    pub parameters: Option<ActionParams>,
    /// Expected duration of the action
    pub expected_duration: Duration,
    /// Provider used to predict this action's outcome
    pub provider: String,
}

pub struct ActionParams {
    /// For robotic actions: end-effector target pose
    pub target_pose: Option<Pose6D>,
    /// For movement: velocity vector
    pub velocity: Option<Vec3>,
    /// For manipulation: force/torque
    pub force: Option<Vec3>,
    /// Generic key-value parameters
    pub custom: HashMap<String, Value>,
}
```

#### 1.2 Goal Specification

Goals can be specified in three forms:

```rust
pub enum PlanGoal {
    /// Natural language description of desired outcome
    Text(String),
    /// Target image showing desired final state
    Image(ImageData),
    /// Structured state description
    State(TargetState),
}

pub struct TargetState {
    /// Object positions in the target state
    pub object_positions: HashMap<String, Position>,
    /// Object relationships (e.g., "cup ON table")
    pub relationships: Vec<Relationship>,
    /// Numeric constraints (e.g., "distance(A, B) < 0.1")
    pub constraints: Vec<Constraint>,
}

// Examples:
let goal_text = PlanGoal::Text("stack the red block on top of the blue block".into());
let goal_image = PlanGoal::Image(load_image("target_state.png")?);
let goal_state = PlanGoal::State(TargetState {
    object_positions: hashmap! {
        "red_block" => Position::new(300, 200),
        "blue_block" => Position::new(300, 250),
    },
    relationships: vec![
        Relationship::on("red_block", "blue_block"),
    ],
    constraints: vec![],
});
```

#### 1.3 Planner Types

```rust
pub enum PlannerType {
    /// Cross-Entropy Method: sample, evaluate, refine distribution
    Cem,
    /// Random sampling: sample many plans, pick the best
    Sampling,
    /// Model Predictive Control: plan-execute-replan loop
    Mpc,
    /// Gradient-based optimization through differentiable world model
    Gradient,
}
```

### 2. Planner Implementations

#### 2.1 Cross-Entropy Method (CEM) Planner

CEM iteratively refines a distribution over action sequences by keeping the
top-performing samples and fitting a new distribution.

**Algorithm:**

```
1. Initialize action distribution D (e.g., Gaussian for continuous actions,
   uniform for discrete actions)
2. For iteration = 1 to max_iterations:
   a. Sample N action sequences from D
   b. For each sequence, predict outcomes using world model
   c. Evaluate each sequence against goal (reward function)
   d. Select top K sequences (elite set, typically top 10%)
   e. Fit new distribution D to elite set
   f. If convergence criteria met, stop
3. Return best action sequence found
```

**Configuration:**

```rust
pub struct CemConfig {
    /// Number of action sequences to sample per iteration
    pub num_samples: usize,         // default: 200
    /// Fraction of samples to keep as elite set
    pub elite_fraction: f64,        // default: 0.1
    /// Maximum CEM iterations
    pub max_iterations: usize,      // default: 10
    /// Convergence threshold (variance of elite set)
    pub convergence_threshold: f64, // default: 0.01
    /// Action sequence length
    pub horizon: usize,             // default: 10
    /// Initial action distribution variance
    pub initial_variance: f64,      // default: 1.0
    /// Smoothing factor for distribution update
    pub alpha: f64,                 // default: 0.1
}
```

**Implementation:**

```rust
impl CemPlanner {
    pub async fn plan(
        &self,
        initial_state: &WorldState,
        goal: &PlanGoal,
        engine: &PredictionEngine,
        config: &CemConfig,
    ) -> Result<Plan> {
        let mut mean = vec![ActionDistribution::uniform(); config.horizon];
        let mut variance = vec![config.initial_variance; config.horizon];

        for iteration in 0..config.max_iterations {
            // Sample action sequences
            let sequences: Vec<Vec<Action>> = (0..config.num_samples)
                .map(|_| self.sample_sequence(&mean, &variance))
                .collect();

            // Evaluate all sequences via world model (batched)
            let evaluations = engine.batch_predict(
                initial_state,
                &sequences,
            ).await?;

            // Score each sequence against goal
            let mut scored: Vec<(f64, usize)> = evaluations.iter()
                .enumerate()
                .map(|(i, eval)| (self.score(eval, goal), i))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Elite set
            let elite_count = (config.num_samples as f64 * config.elite_fraction) as usize;
            let elite: Vec<&Vec<Action>> = scored[..elite_count]
                .iter()
                .map(|(_, i)| &sequences[*i])
                .collect();

            // Update distribution
            let (new_mean, new_variance) = self.fit_distribution(&elite);
            mean = self.smooth_update(&mean, &new_mean, config.alpha);
            variance = new_variance;

            // Check convergence
            if variance.iter().all(|v| *v < config.convergence_threshold) {
                break;
            }
        }

        // Return best plan
        self.construct_plan(&mean, initial_state, goal, engine).await
    }
}
```

#### 2.2 Sampling Planner

The simplest planner: generate many random plans, evaluate all, return the best.
Good for quick planning in low-dimensional action spaces.

**Algorithm:**

```
1. Generate N random action sequences
2. For each sequence, predict outcomes using world model
3. Score each sequence against goal
4. Return the highest-scoring sequence
```

**Configuration:**

```rust
pub struct SamplingConfig {
    /// Number of random plans to generate
    pub num_samples: usize,      // default: 1000
    /// Action sequence length
    pub horizon: usize,          // default: 10
    /// Action space specification
    pub action_space: ActionSpace,
    /// Whether to use stratified sampling
    pub stratified: bool,        // default: true
    /// Seed for reproducibility
    pub seed: Option<u64>,
}
```

**When to use:** Quick, low-stakes planning. Good for action spaces with
fewer than 20 discrete options per step, or when compute budget is very limited.

#### 2.3 Model Predictive Control (MPC) Planner

MPC plans a full sequence but only executes the first action, then replans
from the new observed state. This provides robustness to prediction errors.

**Algorithm:**

```
1. Observe current state s_t
2. Plan full action sequence [a_t, a_{t+1}, ..., a_{t+H}] using inner planner
3. Execute only a_t in the real environment
4. Observe new state s_{t+1}
5. If goal reached, stop. Otherwise, go to step 1.
```

**Configuration:**

```rust
pub struct MpcConfig {
    /// Inner planner for generating candidate plans
    pub inner_planner: PlannerType,  // default: Cem
    /// Inner planner configuration
    pub inner_config: PlannerConfig,
    /// Planning horizon
    pub horizon: usize,              // default: 10
    /// Maximum replanning iterations
    pub max_replan_steps: usize,     // default: 50
    /// Goal achievement threshold
    pub goal_threshold: f64,         // default: 0.9
    /// Whether to warm-start from previous plan
    pub warm_start: bool,            // default: true
    /// State observation function
    pub observe_fn: Option<ObserveFn>,
}
```

**Key Features:**
- **Warm starting**: Each replan iteration initializes from the previous plan
  (shifted by one step), dramatically reducing planning time.
- **State correction**: Real observed states replace predicted states, preventing
  error accumulation.
- **Adaptive horizon**: Can shorten planning horizon as goal approaches.

```rust
impl MpcPlanner {
    pub async fn execute(
        &self,
        initial_state: &WorldState,
        goal: &PlanGoal,
        engine: &PredictionEngine,
        executor: &dyn ActionExecutor,
        config: &MpcConfig,
    ) -> Result<ExecutionTrace> {
        let mut current_state = initial_state.clone();
        let mut previous_plan: Option<Plan> = None;
        let mut trace = ExecutionTrace::new();

        for step in 0..config.max_replan_steps {
            // Plan (with warm start from previous plan)
            let plan = self.inner_plan(
                &current_state,
                goal,
                engine,
                previous_plan.as_ref(),
                &config,
            ).await?;

            // Check guardrails on first action
            let safety = self.check_guardrails(&plan.actions[0]).await?;
            if !safety.is_safe() {
                return Err(PlanError::GuardrailViolation(safety));
            }

            // Execute first action
            let result = executor.execute(&plan.actions[0]).await?;
            trace.record(step, &plan.actions[0], &result);

            // Observe new state
            current_state = executor.observe().await?;

            // Check if goal is achieved
            if self.goal_achieved(&current_state, goal, config.goal_threshold)? {
                trace.mark_success();
                return Ok(trace);
            }

            // Warm-start: shift previous plan by one step
            previous_plan = Some(self.shift_plan(&plan));
        }

        trace.mark_timeout();
        Ok(trace)
    }
}
```

#### 2.4 Gradient-Based Planner

For differentiable world models, optimize action sequences directly via
gradient descent on the goal objective.

**Algorithm:**

```
1. Initialize action sequence a = [a_0, ..., a_H] randomly
2. For iteration = 1 to max_iterations:
   a. Forward pass: predict states s_1, ..., s_H using world model
   b. Compute loss L = goal_distance(s_H, goal) + regularization(a)
   c. Backward pass: compute dL/da via backpropagation through world model
   d. Update a = a - lr * dL/da
   e. Project a onto valid action space
3. Return optimized action sequence
```

**Configuration:**

```rust
pub struct GradientConfig {
    /// Learning rate for action optimization
    pub learning_rate: f64,          // default: 0.01
    /// Maximum optimization iterations
    pub max_iterations: usize,       // default: 100
    /// Optimizer type
    pub optimizer: Optimizer,        // default: Adam
    /// Action regularization weight
    pub regularization: f64,         // default: 0.01
    /// Whether to use learned action embeddings
    pub use_embeddings: bool,        // default: false
    /// Gradient clipping threshold
    pub grad_clip: f64,              // default: 1.0
    /// Convergence threshold for loss
    pub convergence_threshold: f64,  // default: 0.001
}

pub enum Optimizer {
    Sgd,
    Adam { beta1: f64, beta2: f64 },
    AdamW { beta1: f64, beta2: f64, weight_decay: f64 },
}
```

**Requirements:**
- The world model provider must support gradient computation (not all do).
- Actions must be representable in a continuous space (or embedded).
- Currently supported by: local model providers, providers with gradient API.

**Fallback behavior:**
When gradient computation is unavailable, the planner automatically falls back
to CEM with a warning:

```rust
if !engine.supports_gradients() {
    warn!("Provider does not support gradients, falling back to CEM");
    return CemPlanner::new(CemConfig::from_gradient_config(config))
        .plan(initial_state, goal, engine)
        .await;
}
```

### 3. Plan Optimization Loop

The core planning loop is shared across all planners:

```rust
pub struct PlanOptimizer {
    planner: Box<dyn Planner>,
    engine: PredictionEngine,
    guardrails: GuardrailSet,
    goal_evaluator: GoalEvaluator,
}

impl PlanOptimizer {
    pub async fn optimize(
        &self,
        initial_state: &WorldState,
        goal: &PlanGoal,
        options: &PlannerOptions,
    ) -> Result<Plan> {
        // 1. Generate candidate plan
        let mut plan = self.planner.plan(initial_state, goal, &self.engine, options).await?;

        // 2. Run guardrail checks on every action
        for (i, action) in plan.actions.iter().enumerate() {
            let check = self.guardrails.check(
                action,
                &plan.predicted_states[i],
            ).await?;

            plan.safety_checks.push(check.clone());

            if !check.is_safe() {
                // Try to repair: re-plan from this step with constraint
                plan = self.repair_plan(
                    &plan,
                    i,
                    &check.violation,
                    initial_state,
                    goal,
                    options,
                ).await?;
            }
        }

        // 3. Compute success probability
        plan.success_probability = self.goal_evaluator.evaluate(
            &plan.predicted_states.last().unwrap(),
            goal,
        )?;

        Ok(plan)
    }

    async fn repair_plan(
        &self,
        original: &Plan,
        violation_step: usize,
        violation: &GuardrailViolation,
        initial_state: &WorldState,
        goal: &PlanGoal,
        options: &PlannerOptions,
    ) -> Result<Plan> {
        // Add the guardrail violation as a constraint
        let constrained_options = options.clone()
            .add_constraint(Constraint::avoid(violation));

        // Re-plan from the state before the violation
        let replan_state = if violation_step > 0 {
            &original.predicted_states[violation_step - 1]
        } else {
            initial_state
        };

        self.planner.plan(replan_state, goal, &self.engine, &constrained_options).await
    }
}
```

### 4. Guardrail Integration

Every planned action is checked against guardrails before execution:

```rust
pub struct PlanGuardrails {
    /// Safety rules that must not be violated
    pub safety_rules: Vec<SafetyRule>,
    /// Physical feasibility checks
    pub feasibility_checks: Vec<FeasibilityCheck>,
    /// Custom user-defined constraints
    pub custom_constraints: Vec<CustomConstraint>,
}

// Built-in safety rules
pub enum SafetyRule {
    /// Objects must not exceed velocity threshold
    MaxVelocity(f64),
    /// Actions must not cause collisions with specified objects
    NoCollision(Vec<ObjectId>),
    /// Actions must keep objects within workspace bounds
    WorkspaceBounds(BoundingBox),
    /// Force limits for robotic manipulation
    MaxForce(f64),
    /// No actions that could cause irreversible damage
    Reversibility,
}

// Example usage:
let guardrails = PlanGuardrails::new()
    .add_safety_rule(SafetyRule::MaxVelocity(2.0))   // m/s
    .add_safety_rule(SafetyRule::MaxForce(50.0))       // Newtons
    .add_safety_rule(SafetyRule::WorkspaceBounds(workspace))
    .add_safety_rule(SafetyRule::NoCollision(vec!["human_hand".into()]))
    .add_feasibility(FeasibilityCheck::Reachable)
    .add_feasibility(FeasibilityCheck::GraspStable);
```

### 5. Multi-Provider Planning

Different providers may excel at different prediction types. The planning
system can use different providers for different steps or aspects:

```rust
pub struct MultiProviderPlan {
    /// Provider routing rules
    pub routing: ProviderRouting,
}

pub enum ProviderRouting {
    /// Use the same provider for all steps
    Single(String),
    /// Route based on action type
    ActionBased(HashMap<ActionType, String>),
    /// Route based on required physics dimension
    PhysicsBased {
        gravity_provider: String,
        collision_provider: String,
        fluid_provider: String,
        default_provider: String,
    },
    /// Use ensemble: predict with multiple providers, take consensus
    Ensemble {
        providers: Vec<String>,
        aggregation: EnsembleStrategy,
    },
}

pub enum EnsembleStrategy {
    /// Average predicted states (requires compatible output formats)
    Average,
    /// Use majority vote on discrete outcomes
    MajorityVote,
    /// Use the prediction with highest confidence
    MostConfident,
    /// Use provider with best eval score for relevant dimension
    BestForDimension,
}

// Example: use Genesis for gravity predictions, Cosmos for spatial
let routing = ProviderRouting::PhysicsBased {
    gravity_provider: "genesis".into(),
    collision_provider: "genesis".into(),
    fluid_provider: "cosmos".into(),
    default_provider: "genesis".into(),
};

let planner = PlanOptimizer::new()
    .with_routing(routing)
    .build()?;
```

### 6. Plan Execution and Monitoring

#### 6.1 Execution Interface

```rust
#[async_trait]
pub trait ActionExecutor {
    /// Execute an action in the real/simulated environment
    async fn execute(&self, action: &Action) -> Result<ExecutionResult>;
    /// Observe the current state after action execution
    async fn observe(&self) -> Result<WorldState>;
    /// Check if the executor is ready for the next action
    async fn is_ready(&self) -> bool;
    /// Emergency stop
    async fn emergency_stop(&self) -> Result<()>;
}

pub struct ExecutionResult {
    pub success: bool,
    pub actual_state: WorldState,
    pub predicted_state: WorldState,
    pub prediction_error: f64,
    pub execution_time: Duration,
}
```

#### 6.2 Execution Monitor

```rust
pub struct ExecutionMonitor {
    /// Maximum acceptable prediction error before triggering replan
    pub prediction_error_threshold: f64,  // default: 0.2
    /// Maximum execution time per action before timeout
    pub action_timeout: Duration,          // default: 30s
    /// Callback for monitoring events
    pub event_handler: Option<Box<dyn MonitorEventHandler>>,
}

pub enum MonitorEvent {
    ActionStarted { step: usize, action: Action },
    ActionCompleted { step: usize, result: ExecutionResult },
    PredictionDrift { step: usize, error: f64 },
    ReplanTriggered { step: usize, reason: ReplanReason },
    GoalAchieved { total_steps: usize, total_time: Duration },
    PlanFailed { step: usize, error: PlanError },
    EmergencyStop { step: usize, reason: String },
}
```

### 7. Replanning on Failure

#### 7.1 Replan Triggers

```rust
pub enum ReplanReason {
    /// Prediction error exceeded threshold
    PredictionDrift { error: f64, threshold: f64 },
    /// Action execution failed
    ExecutionFailure { action: Action, error: String },
    /// Unexpected obstacle detected
    ObstacleDetected { obstacle: Object },
    /// Goal changed during execution
    GoalChanged { old_goal: PlanGoal, new_goal: PlanGoal },
    /// Better plan found by background planning
    BetterPlanAvailable { improvement: f64 },
    /// Safety constraint became active
    SafetyConstraint { constraint: SafetyRule },
}
```

#### 7.2 Replan Strategy

```rust
pub struct ReplanStrategy {
    /// Maximum number of replanning attempts
    pub max_replans: usize,                // default: 5
    /// Whether to warm-start from the failed plan
    pub warm_start: bool,                  // default: true
    /// Time budget for each replan attempt
    pub replan_budget: Duration,           // default: 3s
    /// Whether to continue background planning during execution
    pub background_planning: bool,         // default: false
    /// Escalation: use more expensive planner after N failures
    pub escalation: Option<EscalationPolicy>,
}

pub struct EscalationPolicy {
    /// After this many failures, switch to more powerful planner
    pub failure_threshold: usize,          // default: 2
    /// Escalation chain: Sampling -> CEM -> Gradient
    pub planner_chain: Vec<PlannerType>,
}
```

#### 7.3 Replanning Loop

```rust
impl PlanExecutor {
    pub async fn execute_with_replanning(
        &self,
        initial_plan: Plan,
        executor: &dyn ActionExecutor,
        strategy: &ReplanStrategy,
    ) -> Result<ExecutionTrace> {
        let mut current_plan = initial_plan;
        let mut replan_count = 0;
        let mut trace = ExecutionTrace::new();

        loop {
            match self.execute_plan(&current_plan, executor, &mut trace).await {
                Ok(()) => {
                    trace.mark_success();
                    return Ok(trace);
                }
                Err(PlanError::NeedsReplan(reason)) => {
                    if replan_count >= strategy.max_replans {
                        return Err(PlanError::MaxReplansExceeded);
                    }

                    let current_state = executor.observe().await?;
                    let planner = self.select_planner(replan_count, strategy);

                    current_plan = planner.plan(
                        &current_state,
                        &self.goal,
                        &self.engine,
                        &PlannerOptions::with_budget(strategy.replan_budget),
                    ).await?;

                    replan_count += 1;
                    trace.record_replan(reason, replan_count);
                }
                Err(PlanError::Safety(violation)) => {
                    executor.emergency_stop().await?;
                    trace.mark_emergency_stop(violation);
                    return Ok(trace);
                }
                Err(e) => return Err(e),
            }
        }
    }
}
```

### 8. ROS2 Integration Path

WorldForge provides a clear integration path for robotics via ROS2:

#### 8.1 ROS2 Node Architecture

```
worldforge_planner_node
├── Subscribers:
│   ├── /camera/image_raw (sensor_msgs/Image) -> current state
│   ├── /joint_states (sensor_msgs/JointState) -> robot state
│   └── /goal (worldforge_msgs/PlanGoal) -> planning goal
├── Publishers:
│   ├── /plan (worldforge_msgs/Plan) -> computed plan
│   ├── /predicted_trajectory (nav_msgs/Path) -> predicted path
│   └── /plan_status (worldforge_msgs/PlanStatus) -> execution status
├── Services:
│   ├── /plan_action (worldforge_msgs/PlanAction) -> single-shot planning
│   └── /replan (worldforge_msgs/Replan) -> trigger replanning
└── Action Servers:
    └── /execute_plan (worldforge_msgs/ExecutePlan) -> plan + execute
```

#### 8.2 ROS2 Message Types

```
# worldforge_msgs/msg/PlanGoal.msg
string goal_text
sensor_msgs/Image goal_image
worldforge_msgs/TargetState goal_state
uint8 goal_type  # TEXT=0, IMAGE=1, STATE=2

# worldforge_msgs/msg/Plan.msg
worldforge_msgs/Action[] actions
sensor_msgs/Image[] predicted_states
float64 success_probability
float64 planning_time_secs
string planner_type

# worldforge_msgs/msg/Action.msg
string description
geometry_msgs/Pose target_pose
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 force
string provider
float64 expected_duration_secs
```

#### 8.3 Integration Example

```python
import rclpy
from rclpy.node import Node
from worldforge_msgs.msg import PlanGoal, Plan
import worldforge as wf

class WorldForgePlannerNode(Node):
    def __init__(self):
        super().__init__('worldforge_planner')
        self.engine = wf.Engine("genesis", api_key=self.get_parameter('api_key'))
        self.planner = wf.Planner(
            type="mpc",
            engine=self.engine,
            guardrails=wf.Guardrails.robotics_default(),
        )

        self.goal_sub = self.create_subscription(PlanGoal, '/goal', self.on_goal, 10)
        self.plan_pub = self.create_publisher(Plan, '/plan', 10)

    def on_goal(self, msg):
        current_image = self.get_camera_image()
        plan = self.planner.plan(
            image=current_image,
            goal=msg.goal_text,
            horizon=10,
        )
        self.plan_pub.publish(plan.to_ros_msg())
```

#### 8.4 MoveIt2 Integration

For robotic manipulation, WorldForge plans can feed into MoveIt2:

```python
from moveit2 import MoveGroupInterface

# WorldForge plans the high-level sequence
wf_plan = planner.plan(image=scene, goal="pick up the cup and place it on the shelf")

# MoveIt2 handles low-level motion planning for each action
for action in wf_plan.actions:
    if action.target_pose:
        move_group.set_pose_target(action.target_pose)
        moveit_plan = move_group.plan()
        move_group.execute(moveit_plan)
```

### 9. Performance Requirements

| Operation                        | Target     | Notes                              |
|----------------------------------|------------|------------------------------------|
| Plan generation (10-step, CEM)   | < 5s       | 200 samples, 10 iterations         |
| Plan generation (10-step, Sampling) | < 2s    | 1000 samples                       |
| Plan generation (10-step, Gradient) | < 3s    | 100 optimization steps             |
| MPC replan (single step)         | < 1s       | Warm-started from previous plan    |
| Guardrail check (per action)     | < 50ms     | All safety rules                   |
| Goal evaluation                  | < 100ms    | State comparison                   |
| Plan repair (single violation)   | < 2s       | Constrained replanning             |
| Full MPC execution (10 steps)    | < 30s      | Including replanning               |

Performance is achieved through:
- **Batched predictions**: All candidate plans predicted in parallel
- **Warm starting**: MPC reuses previous plan computations
- **Cached predictions**: Repeated state-action pairs use cached results
- **Async execution**: Planning runs concurrently with execution monitoring
- **Provider-side batching**: Providers that support batch inference

### 10. Python API

```python
import worldforge as wf

# Create planner
engine = wf.Engine("genesis")
planner = wf.Planner(
    type="cem",                    # or "sampling", "mpc", "gradient"
    engine=engine,
    horizon=10,
    guardrails=wf.Guardrails(
        max_velocity=2.0,
        workspace_bounds=((0, 0, 0), (1, 1, 1)),
    ),
)

# Plan
plan = planner.plan(
    image="current_scene.png",
    goal="stack the red block on the blue block",
)

print(f"Plan has {len(plan.actions)} steps")
print(f"Success probability: {plan.success_probability:.2%}")
for i, action in enumerate(plan.actions):
    print(f"  Step {i+1}: {action.description}")
    plan.predicted_states[i].show()  # Display predicted state

# Execute with monitoring (MPC mode)
mpc = wf.Planner(type="mpc", engine=engine, horizon=10)
trace = mpc.execute(
    image="current_scene.png",
    goal="stack the red block on the blue block",
    executor=my_robot_executor,
    on_replan=lambda reason: print(f"Replanning: {reason}"),
)
print(f"Execution {'succeeded' if trace.success else 'failed'}")
print(f"Total steps: {trace.total_steps}, Replans: {trace.replan_count}")
```

### 11. Configuration

```toml
# worldforge-planning.toml
[planning]
default_planner = "cem"
default_horizon = 10
max_planning_time_secs = 5

[planning.cem]
num_samples = 200
elite_fraction = 0.1
max_iterations = 10
convergence_threshold = 0.01

[planning.sampling]
num_samples = 1000
stratified = true

[planning.mpc]
inner_planner = "cem"
max_replan_steps = 50
warm_start = true
goal_threshold = 0.9

[planning.gradient]
learning_rate = 0.01
max_iterations = 100
optimizer = "adam"
grad_clip = 1.0

[planning.guardrails]
max_velocity = 2.0
max_force = 50.0
workspace_bounds = [[0, 0, 0], [1, 1, 1]]

[planning.replanning]
max_replans = 5
replan_budget_secs = 3
prediction_error_threshold = 0.2
background_planning = false

[planning.providers]
routing = "single"  # or "action_based", "physics_based", "ensemble"
default_provider = "genesis"
```

## Implementation Plan

### Phase 1: Core Planners (Weeks 1-3)
- Implement Sampling planner with goal evaluation
- Implement CEM planner with distribution fitting
- Goal specification (text, image, state)
- Basic guardrail integration
- Unit tests for both planners

### Phase 2: MPC and Execution (Weeks 4-6)
- Implement MPC planner with warm starting
- ActionExecutor trait and mock implementation
- Execution monitor and replanning loop
- Replan triggers and escalation policy
- Integration tests with mock executor

### Phase 3: Gradient Planner (Weeks 7-8)
- Implement gradient-based planner
- Automatic fallback to CEM when gradients unavailable
- Optimizer implementations (SGD, Adam, AdamW)
- Action space projection

### Phase 4: Multi-Provider and Advanced Features (Weeks 9-10)
- Multi-provider routing
- Ensemble planning
- Background planning during execution
- Performance optimization (batching, caching)

### Phase 5: ROS2 Integration (Weeks 11-13)
- ROS2 message type definitions
- WorldForge planner ROS2 node
- MoveIt2 integration bridge
- ROS2 integration tests
- Robotics tutorial

### Phase 6: Polish (Week 14)
- Performance benchmarking against targets
- Python API finalization
- Documentation and examples
- Configuration file support

## Testing Strategy

### Unit Tests
- Each planner tested with deterministic mock world models
- Goal evaluation functions tested with known states
- Guardrail checks tested with safe/unsafe action pairs
- Replan triggers tested with synthetic prediction errors

### Integration Tests
- End-to-end planning with mock provider and mock executor
- MPC loop with simulated environment that diverges from predictions
- Multi-provider routing with provider-specific mock responses
- Guardrail repair loop with constrained action spaces

### Property-Based Tests
- Plans always have correct length (== horizon)
- Success probability in [0.0, 1.0]
- Guardrail-checked plans never contain unsafe actions
- MPC always terminates (within max_replan_steps)

### Performance Tests
- CEM planning time < 5s for 10-step plans (200 samples, 10 iterations)
- Sampling planning time < 2s for 10-step plans (1000 samples)
- MPC replan time < 1s with warm start
- Guardrail check < 50ms per action

### Robotics Tests (simulation)
- Plan and execute block stacking in PyBullet simulation
- Plan and execute pick-and-place in Isaac Sim
- ROS2 node integration test with Gazebo

## Open Questions

1. **Action representation**: Should actions be purely text-based (flexible but
   harder to optimize) or structured (less flexible but amenable to gradient
   optimization)? Hybrid approach?

2. **Prediction caching**: How aggressively should we cache world model
   predictions? Stale predictions could lead to suboptimal plans, but
   re-prediction is expensive.

3. **Hierarchical planning**: Should we support hierarchical task decomposition
   (e.g., "make coffee" -> "pick up cup", "move to machine", "press button")?
   This is a natural extension but adds significant complexity.

4. **Real-time constraints**: For robotics, plans may need to be generated
   within strict real-time deadlines. Should we support anytime planning
   (return best plan found within deadline)?

5. **Multi-agent planning**: Should the planning system support multiple
   agents with potentially conflicting goals? This is important for
   multi-robot systems.

6. **Plan explanation**: Should plans include human-readable explanations
   of why each action was chosen? Important for trust and debugging.

7. **Sim-to-real gap**: How do we handle the gap between world model
   predictions and real-world physics? Should the planner learn correction
   factors from execution history?

8. **Safety certification**: For safety-critical robotics applications,
   what level of formal verification is needed for the guardrail system?
   Can we provide probabilistic safety guarantees?
