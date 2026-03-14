# WorldForge Technical Specification

**Version 0.1.0-draft | March 2026**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Principles](#2-design-principles)
3. [Core Type System](#3-core-type-system)
4. [Provider Abstraction Layer](#4-provider-abstraction-layer)
5. [World State Management](#5-world-state-management)
6. [Action System](#6-action-system)
7. [Prediction Engine](#7-prediction-engine)
8. [Planning System](#8-planning-system)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Guardrails & Safety](#10-guardrails--safety)
11. [ZK Verification Layer](#11-zk-verification-layer)
12. [API Design](#12-api-design)
13. [Provider Specifications](#13-provider-specifications)
14. [Error Handling](#14-error-handling)
15. [Performance Requirements](#15-performance-requirements)

---

## 1. Overview

### 1.1 Problem Statement

World foundation models (WFMs) are the next frontier of AI infrastructure. NVIDIA Cosmos, Runway GWM-1, Meta's JEPA family, Google's Genie, and World Labs' Marble each offer powerful capabilities for predicting, generating, and reasoning about the physical world. However:

- Each provider has its own SDK, API format, and data schema
- There is no standard way to compose multi-step world model workflows
- State management across inference calls is entirely the developer's responsibility
- Safety guardrails are provider-specific and non-portable
- Comparing outputs across providers requires manual effort
- No evaluation framework exists that works across providers

WorldForge solves all of these problems by providing a unified orchestration layer.

### 1.2 Scope

WorldForge covers:
- Unified provider abstraction (Cosmos, GWM, JEPA, Genie, Marble, local models)
- Persistent world state management
- Standardized action type system
- Multi-step prediction and planning
- Cross-provider evaluation and comparison
- Configurable safety guardrails
- Optional ZK verification for safety-critical applications
- REST API server for language-agnostic access
- CLI tool for rapid experimentation

WorldForge does NOT:
- Train world models (use provider tools for training)
- Replace provider SDKs (it wraps them)
- Provide compute infrastructure (use cloud providers)
- Build end-user applications (it's developer infrastructure)

### 1.3 Design Goals

| Goal | Metric |
|------|--------|
| Provider switching | < 1 line of code change to switch providers |
| Latency overhead | < 5ms per WorldForge call (excluding provider latency) |
| State persistence | Worlds survive process restarts (serializable) |
| Type safety | Zero runtime type errors in well-typed code |
| Extensibility | New providers via a single trait implementation |

---

## 2. Design Principles

### 2.1 Provider Agnostic, Provider Aware

WorldForge abstracts provider differences but doesn't hide them. Developers can always:
- Access provider-specific features via escape hatches
- See which capabilities are available per provider
- Get clear errors when a requested capability isn't supported

### 2.2 Composition Over Configuration

Complex workflows are built by composing simple primitives (predict, plan, evaluate, verify) rather than configuring a monolithic system.

### 2.3 Safety By Default

Guardrails are opt-out, not opt-in. Every prediction is checked against basic physical constraints (energy conservation, collision detection) unless explicitly disabled.

### 2.4 Rust Core, Python Interface

The core library is in Rust for performance, safety, and WASM portability. Python bindings via PyO3 provide ergonomic access for the ML/robotics community. The Rust core can also be compiled to WASM for browser and edge deployment.

### 2.5 Open Core

The core library, all provider adapters, and the evaluation framework are Apache 2.0. The cloud offering (managed hosting, caching, dashboards) is proprietary.

---

## 3. Core Type System

### 3.1 Tensor Types

```rust
/// A multi-dimensional array with shape and dtype metadata.
/// Wraps provider-specific tensor types (ndarray, torch, burn).
pub struct Tensor {
    pub data: TensorData,    // Actual tensor data
    pub shape: Vec<usize>,   // Dimension sizes
    pub dtype: DType,        // f16, f32, bf16, u8, etc.
    pub device: Device,      // CPU, CUDA, WASM
}

pub enum DType {
    Float16,
    Float32,
    BFloat16,
    UInt8,
    Int32,
    Int64,
}

pub enum Device {
    Cpu,
    Cuda(u32),      // GPU index
    Wasm,
    Remote(String),  // Provider endpoint
}
```

### 3.2 Spatial Types

```rust
/// 3D position in world coordinates.
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D rotation as quaternion.
pub struct Rotation {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Complete 6DoF pose.
pub struct Pose {
    pub position: Position,
    pub rotation: Rotation,
}

/// Axis-aligned bounding box.
pub struct BBox {
    pub min: Position,
    pub max: Position,
}

/// A 3D mesh representation.
pub struct Mesh {
    pub vertices: Vec<Position>,
    pub faces: Vec<[u32; 3]>,
    pub normals: Option<Vec<Position>>,
    pub uvs: Option<Vec<[f32; 2]>>,
}
```

### 3.3 Temporal Types

```rust
/// A timestamp in simulation time.
pub struct SimTime {
    pub step: u64,           // Discrete step index
    pub seconds: f64,        // Continuous time in seconds
    pub dt: f64,             // Time delta since last step
}

/// A time range.
pub struct TimeRange {
    pub start: SimTime,
    pub end: SimTime,
}

/// A trajectory: sequence of poses over time.
pub struct Trajectory {
    pub poses: Vec<(SimTime, Pose)>,
    pub velocities: Option<Vec<(SimTime, Velocity)>>,
}
```

### 3.4 Media Types

```rust
/// A single video frame.
pub struct Frame {
    pub data: Tensor,          // [H, W, C] image tensor
    pub timestamp: SimTime,
    pub camera: Option<CameraPose>,
    pub depth: Option<Tensor>, // [H, W] depth map
    pub segmentation: Option<Tensor>, // [H, W] semantic labels
}

/// A video clip: sequence of frames.
pub struct VideoClip {
    pub frames: Vec<Frame>,
    pub fps: f32,
    pub resolution: (u32, u32), // (width, height)
    pub duration: f64,          // seconds
}

/// Camera intrinsics and extrinsics.
pub struct CameraPose {
    pub extrinsics: Pose,       // Camera position and orientation
    pub fov: f32,               // Field of view in degrees
    pub near_clip: f32,
    pub far_clip: f32,
}
```

---

## 4. Provider Abstraction Layer

### 4.1 Provider Trait

Every world model provider implements this trait:

```rust
#[async_trait]
pub trait WorldModelProvider: Send + Sync {
    /// Provider identifier
    fn name(&self) -> &str;

    /// What this provider can do
    fn capabilities(&self) -> ProviderCapabilities;

    /// Generate a prediction from current state + action
    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction, WorldForgeError>;

    /// Generate a video from a text/image prompt
    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip, WorldForgeError>;

    /// Reason about a scene (VLM-style)
    async fn reason(
        &self,
        input: &ReasoningInput,
        query: &str,
    ) -> Result<ReasoningOutput, WorldForgeError>;

    /// Transfer: apply spatial controls to generate controlled output
    async fn transfer(
        &self,
        source: &VideoClip,
        controls: &SpatialControls,
        config: &TransferConfig,
    ) -> Result<VideoClip, WorldForgeError>;

    /// Check if provider is healthy and reachable
    async fn health_check(&self) -> Result<HealthStatus, WorldForgeError>;

    /// Estimate cost for an operation
    fn estimate_cost(&self, operation: &Operation) -> CostEstimate;
}

/// Capabilities advertised by a provider
pub struct ProviderCapabilities {
    pub predict: bool,
    pub generate: bool,
    pub reason: bool,
    pub transfer: bool,
    pub action_conditioned: bool,
    pub multi_view: bool,
    pub max_video_length_seconds: f32,
    pub max_resolution: (u32, u32),
    pub fps_range: (f32, f32),
    pub supported_action_spaces: Vec<ActionSpaceType>,
    pub supports_depth: bool,
    pub supports_segmentation: bool,
    pub supports_planning: bool,
    pub latency_profile: LatencyProfile,
}

pub struct LatencyProfile {
    pub p50_ms: u32,
    pub p95_ms: u32,
    pub p99_ms: u32,
    pub throughput_fps: f32,
}
```

### 4.2 Provider Registry

```rust
/// Global registry of available providers.
pub struct ProviderRegistry {
    providers: HashMap<String, Box<dyn WorldModelProvider>>,
}

impl ProviderRegistry {
    /// Register a new provider
    pub fn register(&mut self, provider: Box<dyn WorldModelProvider>);

    /// Get a provider by name
    pub fn get(&self, name: &str) -> Option<&dyn WorldModelProvider>;

    /// List all registered providers
    pub fn list(&self) -> Vec<&str>;

    /// Find providers with specific capabilities
    pub fn find_by_capability(&self, cap: &str) -> Vec<&dyn WorldModelProvider>;

    /// Auto-detect available providers from environment
    pub fn auto_detect() -> Self;
}
```

### 4.3 Provider Implementations

#### 4.3.1 NVIDIA Cosmos Provider

```rust
pub struct CosmosProvider {
    pub model: CosmosModel,
    pub api_key: String,
    pub endpoint: CosmosEndpoint,
    pub config: CosmosConfig,
}

pub enum CosmosModel {
    Predict2_5,           // Video generation / future prediction
    Transfer2_5,          // Spatial control to video
    Reason2,              // Physical reasoning VLM (7B)
    Embed1,               // Video-text embeddings
}

pub enum CosmosEndpoint {
    NimApi(String),       // NVIDIA NIM managed API
    NimLocal(String),     // Self-hosted NIM container
    HuggingFace,          // Direct model download
    DgxCloud(String),     // DGX Cloud deployment
}
```

Integration approach:
- For NIM API: HTTP REST calls to NVIDIA's hosted endpoints
- For NIM Local: Docker container with gRPC interface
- For HuggingFace: Download model weights, run with local PyTorch/TensorRT
- Map WorldForge action types to Cosmos input format (text prompt + control signals)
- Parse Cosmos video output into WorldForge VideoClip format
- Use Cosmos Reason for physics plausibility scoring

#### 4.3.2 Runway GWM Provider

```rust
pub struct RunwayProvider {
    pub model: RunwayModel,
    pub api_secret: String,
    pub endpoint: String,
}

pub enum RunwayModel {
    Gwm1Worlds,       // Explorable environments
    Gwm1Robotics,     // Robot action-conditioned prediction
    Gwm1Avatars,      // Conversational characters
}
```

Integration approach:
- HTTP REST API via Runway's official SDK (Python) or direct HTTP (Rust)
- GWM Robotics: action-conditioned video generation via Python SDK
- GWM Worlds: text/image prompt to explorable environment
- GWM Avatars: audio-driven character generation
- Map WorldForge actions to Runway's action format (camera pose, robot commands)

#### 4.3.3 JEPA Provider (Local)

```rust
pub struct JepaProvider {
    pub model_path: PathBuf,
    pub backend: JepaBackend,
}

pub enum JepaBackend {
    Burn,              // Rust-native (via jepa-rs)
    PyTorch,           // Via tch-rs bindings
    Onnx,              // Via ort-rs
    Safetensors,       // Direct weight loading
}
```

Integration approach:
- Load V-JEPA / V-JEPA 2 weights from safetensors
- Run inference locally using burn framework (Rust-native) or PyTorch bindings
- This is the fully open-source, self-hosted option
- Enables ZK verification (circuit runs locally, no API dependency)
- jepa-rs is the reference implementation for this provider

#### 4.3.4 Google Genie Provider

```rust
pub struct GenieProvider {
    pub model: GenieModel,
    pub api_key: String,
}

pub enum GenieModel {
    Genie3,
}
```

Note: Genie is still in research preview. Provider will be stubbed until public API is available.

---

## 5. World State Management

### 5.1 World State

```rust
/// Complete state of a simulated world at a point in time.
pub struct WorldState {
    pub id: WorldId,
    pub time: SimTime,
    pub scene: SceneGraph,
    pub history: StateHistory,
    pub metadata: WorldMetadata,
}

/// Unique world identifier.
pub type WorldId = uuid::Uuid;

/// Scene graph: hierarchical representation of objects in the world.
pub struct SceneGraph {
    pub root: SceneNode,
    pub objects: HashMap<ObjectId, SceneObject>,
    pub relationships: Vec<SpatialRelationship>,
}

pub struct SceneObject {
    pub id: ObjectId,
    pub name: String,
    pub pose: Pose,
    pub bbox: BBox,
    pub mesh: Option<Mesh>,
    pub physics: PhysicsProperties,
    pub semantic_label: Option<String>,
    pub visual_embedding: Option<Tensor>, // Provider-specific embedding
}

pub struct PhysicsProperties {
    pub mass: Option<f32>,          // kg
    pub friction: Option<f32>,      // coefficient
    pub restitution: Option<f32>,   // bounciness
    pub is_static: bool,
    pub is_graspable: bool,
    pub material: Option<String>,
}

pub enum SpatialRelationship {
    On { subject: ObjectId, surface: ObjectId },
    In { subject: ObjectId, container: ObjectId },
    Near { a: ObjectId, b: ObjectId, distance: f32 },
    Touching { a: ObjectId, b: ObjectId },
    Above { subject: ObjectId, reference: ObjectId },
    Below { subject: ObjectId, reference: ObjectId },
}
```

### 5.2 State Persistence

```rust
/// Manages world state persistence across sessions.
pub trait StateStore: Send + Sync {
    /// Save world state
    async fn save(&self, state: &WorldState) -> Result<(), WorldForgeError>;

    /// Load world state by ID
    async fn load(&self, id: &WorldId) -> Result<WorldState, WorldForgeError>;

    /// List all saved worlds
    async fn list(&self) -> Result<Vec<WorldId>, WorldForgeError>;

    /// Delete a world
    async fn delete(&self, id: &WorldId) -> Result<(), WorldForgeError>;
}

/// Implementations
pub struct FileStateStore { pub path: PathBuf }    // JSON/MessagePack files
pub struct SqliteStateStore { pub db: SqlitePool }  // SQLite database
pub struct RedisStateStore { pub url: String }      // Redis for distributed state
pub struct S3StateStore { pub bucket: String }      // S3 for cloud deployment
```

### 5.3 State History

```rust
/// History of all states a world has been through.
pub struct StateHistory {
    pub states: Vec<HistoryEntry>,
    pub max_entries: usize,       // Rolling window
    pub compression: Compression, // None, LZ4, Zstd
}

pub struct HistoryEntry {
    pub time: SimTime,
    pub state_hash: [u8; 32],  // SHA-256 of serialized state
    pub action: Option<Action>,
    pub prediction: Option<PredictionSummary>,
    pub provider: String,
}
```

---

## 6. Action System

### 6.1 Action Types

```rust
/// A standardized action that can be translated to any provider's format.
pub enum Action {
    // === Robot manipulation ===
    Move { target: Position, speed: f32 },
    Grasp { object: ObjectId, grip_force: f32 },
    Release { object: ObjectId },
    Push { object: ObjectId, direction: Vec3, force: f32 },
    Rotate { object: ObjectId, axis: Vec3, angle: f32 },
    Place { object: ObjectId, target: Position },

    // === Camera/navigation ===
    CameraMove { delta: Pose },
    CameraLookAt { target: Position },
    Navigate { waypoints: Vec<Position> },
    Teleport { destination: Pose },

    // === Environment ===
    SetWeather { weather: Weather },
    SetLighting { time_of_day: f32 },
    SpawnObject { template: String, pose: Pose },
    RemoveObject { object: ObjectId },

    // === Compound ===
    Sequence(Vec<Action>),
    Parallel(Vec<Action>),
    Conditional { condition: Condition, then: Box<Action>, otherwise: Option<Box<Action>> },

    // === Raw ===
    Raw { provider: String, data: serde_json::Value },
}

pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

pub enum Weather {
    Clear, Cloudy, Rain, Snow, Fog, Night,
}
```

### 6.2 Action Translation

Each provider adapter implements action translation:

```rust
pub trait ActionTranslator {
    /// Translate a WorldForge action to provider-specific format
    fn translate(&self, action: &Action) -> Result<ProviderAction, WorldForgeError>;

    /// Which actions this provider supports
    fn supported_actions(&self) -> Vec<ActionType>;
}
```

For Cosmos Predict: Actions become text prompts + control signals
For Runway GWM Robotics: Actions become robot command sequences via Python SDK
For JEPA: Actions become latent action vectors for the action-conditioned model

---

## 7. Prediction Engine

### 7.1 Prediction

```rust
/// The result of asking a world model "what happens next?"
pub struct Prediction {
    pub id: PredictionId,
    pub provider: String,
    pub model: String,
    pub input_state: WorldState,
    pub action: Action,
    pub output_state: WorldState,
    pub video: Option<VideoClip>,
    pub confidence: f32,            // 0.0 - 1.0
    pub physics_scores: PhysicsScores,
    pub latency_ms: u64,
    pub cost: CostEstimate,
    pub guardrail_results: Vec<GuardrailResult>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct PhysicsScores {
    pub overall: f32,
    pub object_permanence: f32,
    pub gravity_compliance: f32,
    pub collision_accuracy: f32,
    pub spatial_consistency: f32,
    pub temporal_consistency: f32,
}
```

### 7.2 Prediction Config

```rust
pub struct PredictionConfig {
    pub steps: u32,                // Number of future steps
    pub resolution: (u32, u32),    // Output resolution
    pub fps: f32,                  // Output FPS
    pub return_video: bool,        // Generate video frames
    pub return_depth: bool,        // Generate depth maps
    pub return_segmentation: bool, // Generate semantic segmentation
    pub guardrails: Vec<GuardrailConfig>,
    pub max_latency_ms: Option<u64>, // Timeout
    pub fallback_provider: Option<String>, // Use this if primary fails
    pub num_samples: u32,          // Multiple predictions for uncertainty estimation
    pub temperature: f32,          // Sampling temperature
}
```

### 7.3 Multi-Provider Prediction

```rust
/// Predict with multiple providers and compare
pub async fn predict_multi(
    world: &World,
    action: &Action,
    providers: &[&str],
    config: &PredictionConfig,
) -> Result<MultiPrediction, WorldForgeError>;

pub struct MultiPrediction {
    pub predictions: Vec<Prediction>,
    pub agreement_score: f32,     // How much providers agree
    pub best_prediction: usize,   // Index of highest-quality prediction
    pub comparison: ComparisonReport,
}
```

---

## 8. Planning System

### 8.1 Planning Interface

```rust
/// Find an action sequence that transforms current state into goal state.
pub struct PlanRequest {
    pub current_state: WorldState,
    pub goal: PlanGoal,
    pub max_steps: u32,
    pub guardrails: Vec<GuardrailConfig>,
    pub planner: PlannerType,
    pub timeout_seconds: f64,
}

pub enum PlanGoal {
    /// Reach a state where a condition is true
    Condition(Condition),
    /// Minimize distance to a target state
    TargetState(WorldState),
    /// Match a goal image
    GoalImage(Tensor),
    /// Reach a state described in natural language
    Description(String),
}

pub enum PlannerType {
    /// Gradient-based planning through differentiable world model (JEPA)
    Gradient {
        learning_rate: f32,
        num_iterations: u32,
    },
    /// Sampling-based: generate many random plans, pick the best
    Sampling {
        num_samples: u32,
        top_k: u32,
    },
    /// Cross-entropy method
    CEM {
        population_size: u32,
        elite_fraction: f32,
        num_iterations: u32,
    },
    /// Model Predictive Control: replan at every step
    MPC {
        horizon: u32,
        num_samples: u32,
        replanning_interval: u32,
    },
    /// Use provider's built-in planning if available
    ProviderNative,
}
```

### 8.2 Plan Output

```rust
pub struct Plan {
    pub actions: Vec<Action>,
    pub predicted_states: Vec<WorldState>,
    pub predicted_videos: Option<Vec<VideoClip>>,
    pub total_cost: f32,
    pub success_probability: f32,
    pub guardrail_compliance: Vec<Vec<GuardrailResult>>,
    pub planning_time_ms: u64,
    pub iterations_used: u32,
    pub verification_proof: Option<ZkProof>, // If ZK verification enabled
}
```

---

## 9. Evaluation Framework

### 9.1 Evaluation Dimensions

```rust
pub enum EvalDimension {
    /// Does the model track objects behind occluders?
    ObjectPermanence,
    /// Do unsupported objects fall?
    GravityCompliance,
    /// Do objects bounce/break on contact?
    CollisionAccuracy,
    /// Turn around — is the scene the same?
    SpatialConsistency,
    /// Fast forward and rewind — same state?
    TemporalConsistency,
    /// Given action A, does the outcome match physics?
    ActionPredictionAccuracy,
    /// Does the model understand materials (glass breaks, rubber bounces)?
    MaterialUnderstanding,
    /// Perception of depth, scale, and distance
    SpatialReasoning,
    /// Custom evaluation metric
    Custom { name: String, evaluator: Box<dyn Evaluator> },
}
```

### 9.2 Evaluation Suite

```rust
pub struct EvalSuite {
    pub scenarios: Vec<EvalScenario>,
    pub dimensions: Vec<EvalDimension>,
    pub providers: Vec<String>,
}

pub struct EvalScenario {
    pub name: String,
    pub description: String,
    pub initial_state: WorldState,
    pub actions: Vec<Action>,
    pub expected_outcomes: Vec<ExpectedOutcome>,
    pub ground_truth: Option<VideoClip>,
}

pub struct EvalResult {
    pub provider: String,
    pub scenario: String,
    pub scores: HashMap<EvalDimension, f32>,
    pub latency_ms: u64,
    pub video: Option<VideoClip>,
}
```

---

## 10. Guardrails & Safety

### 10.1 Guardrail Types

```rust
pub enum Guardrail {
    /// No object may pass through another
    NoCollisions,
    /// Specified objects must stay upright
    StayUpright { objects: Vec<ObjectId>, max_tilt_degrees: f32 },
    /// No object may leave the specified area
    BoundaryConstraint { bounds: BBox },
    /// Energy must be conserved (within tolerance)
    EnergyConservation { tolerance: f32 },
    /// Specific states are forbidden
    ForbiddenStates { conditions: Vec<Condition> },
    /// Custom cost function
    CostFunction { function: Box<dyn CostFn>, threshold: f32 },
    /// Maximum velocity for any object
    MaxVelocity { limit: f32 },
    /// Human safety zone: no robot action within radius of human
    HumanSafetyZone { radius: f32 },
}

pub struct GuardrailResult {
    pub guardrail: Guardrail,
    pub passed: bool,
    pub violation_details: Option<String>,
    pub severity: ViolationSeverity,
}

pub enum ViolationSeverity {
    Info,
    Warning,
    Critical,   // Prediction is returned but flagged
    Blocking,   // Prediction is rejected
}
```

### 10.2 Guardrail Pipeline

Every prediction passes through the guardrail pipeline:

```
Prediction request
  -> Provider inference
  -> Post-inference guardrail check
  -> If all pass: return prediction
  -> If warning: return prediction + warnings
  -> If critical: return prediction + critical flags
  -> If blocking: return error
  -> Optional: generate ZK proof of guardrail compliance
```

---

## 11. ZK Verification Layer

### 11.1 Overview

The ZK verification layer is WorldForge's unique differentiator. It provides cryptographic proof that:

1. A world model inference was computed correctly
2. All guardrails passed for every step of a plan
3. Input data was not tampered with

This is critical for safety-critical applications (surgery, AV, industrial automation) where trust in the world model's output is insufficient — you need mathematical proof.

### 11.2 Proof Types

```rust
pub enum ZkProofType {
    /// Prove that the forward pass was computed correctly
    InferenceVerification {
        model_hash: [u8; 32],
        input_hash: [u8; 32],
        output_hash: [u8; 32],
    },
    /// Prove that all guardrails passed
    GuardrailCompliance {
        plan_hash: [u8; 32],
        guardrail_hashes: Vec<[u8; 32]>,
        all_passed: bool,
    },
    /// Prove data provenance
    DataProvenance {
        data_hash: [u8; 32],
        timestamp: u64,
        source_commitment: [u8; 32],
    },
}
```

### 11.3 Implementation Strategy

Phase 1: EZKL-based proofs for small models (inference verification)
Phase 2: Cairo/STARK-based proofs for guardrail compliance
Phase 3: On-chain verification on Starknet for audit trails

The JEPA provider is the primary target for ZK verification because:
- Model runs locally (no API dependency)
- Architecture is relatively simple (ViT + predictor)
- The forward pass is deterministic and differentiable
- Cairo implementation can leverage existing STARK expertise

---

## 12. API Design

### 12.1 Python API

```python
from worldforge import WorldForge, World, Action, Guardrail
from worldforge.providers import CosmosProvider, RunwayProvider, JepaProvider
from worldforge.eval import EvalSuite, PhysicsEval
from worldforge.verify import ZkVerifier

# Initialize
wf = WorldForge()
wf.register_provider("cosmos", CosmosProvider(
    model="cosmos-predict-2.5",
    api_key=os.environ["NVIDIA_API_KEY"]
))
wf.register_provider("runway", RunwayProvider(
    model="gwm-1-robotics",
    api_secret=os.environ["RUNWAY_API_SECRET"]
))
wf.register_provider("jepa", JepaProvider(
    model_path="./models/v-jepa-2",
    backend="burn"
))

# Create world
world = wf.create_world(
    prompt="A robot arm next to a table with blocks",
    provider="cosmos"
)

# Predict
pred = world.predict(
    action=Action.grasp("red_block"),
    config={"steps": 10, "return_video": True}
)

# Plan
plan = world.plan(
    goal="Stack red block on blue block",
    max_steps=20,
    planner="cem",
    guardrails=[
        Guardrail.no_collisions(),
        Guardrail.stay_upright(["red_block", "blue_block"]),
    ]
)

# Evaluate across providers
suite = PhysicsEval.standard_suite()
results = suite.run(providers=["cosmos", "runway", "jepa"])
results.to_leaderboard()

# Verify (for safety-critical use)
verifier = ZkVerifier(backend="stark")
proof = verifier.prove_plan(plan)
assert verifier.verify(proof)
```

### 12.2 REST API

```
POST /v1/worlds                    Create a new world
GET  /v1/worlds/{id}               Get world state
POST /v1/worlds/{id}/predict       Predict next state
POST /v1/worlds/{id}/plan          Plan action sequence
POST /v1/worlds/{id}/evaluate      Run evaluation suite
POST /v1/worlds/{id}/verify        Generate ZK proof
GET  /v1/providers                 List available providers
GET  /v1/providers/{name}/health   Provider health check
POST /v1/compare                   Compare predictions across providers
```

### 12.3 CLI

```bash
# Create a world
worldforge create --prompt "A kitchen with a mug" --provider cosmos

# Predict
worldforge predict --world <id> --action "push mug left" --steps 10

# Plan
worldforge plan --world <id> --goal "mug in dishwasher" --planner cem

# Evaluate
worldforge eval --suite physics --providers cosmos,runway,jepa

# Compare
worldforge compare --world <id> --action "push mug" --providers cosmos,runway

# Verify
worldforge verify --plan <plan-id> --backend stark
```

---

## 13. Provider Specifications

### 13.1 NVIDIA Cosmos Integration

**API Surface:**
- Cosmos Predict 2.5: NIM API endpoint or self-hosted container
- Cosmos Transfer 2.5: NIM API or container
- Cosmos Reason 2: NIM API or container (7B VLM)
- Cosmos Embed 1: NIM API (OpenAI Embeddings-compatible)
- Cosmos Evaluator: Open-source evaluation pipeline
- Cosmos Curate: Data curation pipeline

**Authentication:** NVIDIA API key via NGC
**SDK:** Python (ngcsdk), direct HTTP REST
**Models on HuggingFace:** nvidia/Cosmos-Predict2.5-*, nvidia/Cosmos-Reason2-*
**License:** Apache 2.0 (code), NVIDIA Open Model License (models)

**WorldForge mapping:**
| WorldForge concept | Cosmos implementation |
|-------------------|----------------------|
| predict() | Cosmos Predict 2.5 (text/image/video prompt → video) |
| reason() | Cosmos Reason 2 (video/image + query → reasoning) |
| transfer() | Cosmos Transfer 2.5 (3D controls → video) |
| physics_score() | Cosmos Reason 2 + Cosmos Evaluator |
| embed() | Cosmos Embed 1 (video/text → embedding vector) |

### 13.2 Runway GWM Integration

**API Surface:**
- GWM-1 Worlds: Interactive environment generation
- GWM-1 Robotics: Action-conditioned video generation (Python SDK)
- GWM-1 Avatars: Audio-driven character generation (React/Node SDK)
- Gen-4.5: Base video generation

**Authentication:** Runway API secret
**SDK:** Python (sdk-python), Node.js (sdk-node), React (avatars-sdk-react)
**Access:** Request access via runwayml.com for Robotics SDK
**License:** Proprietary API

**WorldForge mapping:**
| WorldForge concept | Runway implementation |
|-------------------|----------------------|
| predict() | GWM-1 Robotics (action → video rollout) |
| generate() | GWM-1 Worlds (text → explorable environment) |
| reason() | Not available (use Cosmos Reason as fallback) |
| transfer() | GWM-1 Worlds with spatial controls |

### 13.3 JEPA Integration (Local)

**Models:**
- I-JEPA (facebookresearch/ijepa): Image JEPA
- V-JEPA (facebookresearch/jepa): Video JEPA
- V-JEPA 2: Video + action-conditioned planning
- EB-JEPA (facebookresearch/eb_jepa): Educational, includes world model examples

**Weights:** Safetensors format on HuggingFace
**Backend:** jepa-rs (Rust/burn) or PyTorch (tch-rs)
**License:** CC-BY-NC 4.0 (models), Apache 2.0 (code)

**WorldForge mapping:**
| WorldForge concept | JEPA implementation |
|-------------------|---------------------|
| predict() | V-JEPA 2 forward pass (context → target representation) |
| plan() | Gradient-based planning through differentiable model |
| physics_score() | Energy function (L2 distance in representation space) |
| verify() | ZK proof of forward pass (primary target for verification) |

---

## 14. Error Handling

```rust
pub enum WorldForgeError {
    // Provider errors
    ProviderNotFound(String),
    ProviderUnavailable { provider: String, reason: String },
    ProviderTimeout { provider: String, timeout_ms: u64 },
    ProviderRateLimited { provider: String, retry_after_ms: u64 },
    ProviderAuthError(String),

    // Capability errors
    UnsupportedAction { provider: String, action: String },
    UnsupportedCapability { provider: String, capability: String },

    // State errors
    WorldNotFound(WorldId),
    InvalidState(String),
    StateCorrupted { world_id: WorldId, details: String },

    // Guardrail errors
    GuardrailViolation { guardrail: String, details: String },
    GuardrailBlocked { violations: Vec<GuardrailResult> },

    // Planning errors
    PlanningFailed { reason: String },
    PlanningTimeout { elapsed_ms: u64 },
    NoFeasiblePlan { goal: String, reason: String },

    // Verification errors
    VerificationFailed { proof_type: String, details: String },

    // General
    SerializationError(String),
    NetworkError(String),
    InternalError(String),
}
```

---

## 15. Performance Requirements

### 15.1 Latency Budget

| Operation | WorldForge overhead | Provider latency (typical) | Total |
|-----------|--------------------|----|-------|
| predict() | < 5ms | 200-2000ms | < 2005ms |
| plan() (10 steps) | < 50ms | 2-20s | < 20.05s |
| evaluate() (1 scenario) | < 10ms | 500-5000ms | < 5010ms |
| verify() (tiny model) | < 30s (proof gen) | N/A | < 30s |
| verify() (verify proof) | < 100ms | N/A | < 100ms |

### 15.2 Memory Budget

| Component | Target |
|-----------|--------|
| WorldForge core | < 50MB |
| World state (1 world) | < 10MB |
| State history (1000 entries) | < 100MB |
| Provider adapter | < 20MB (excluding model weights) |
| JEPA model weights (local) | 1-5GB |

### 15.3 Scalability

| Dimension | Target |
|-----------|--------|
| Concurrent worlds | 10,000+ |
| Concurrent predictions | 1,000+ (limited by provider throughput) |
| State history depth | 100,000 entries per world |
| Providers per registry | 50+ |
