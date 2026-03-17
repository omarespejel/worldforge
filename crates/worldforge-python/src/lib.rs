// PyO3-generated code triggers this clippy lint; it's a known false positive.
#![allow(clippy::useless_conversion)]
//! Python bindings for WorldForge.
//!
//! Exposes core types, scene management, and the main WorldForge
//! orchestrator to Python via PyO3.

use std::path::Path;
use std::sync::Arc;

use pyo3::prelude::*;

use worldforge_core::guardrail::GuardrailConfig;
use worldforge_core::prediction::{PlannerType, PredictionConfig};
use worldforge_core::provider::{
    GenerationConfig, GenerationPrompt, SpatialControls, TransferConfig,
};
use worldforge_core::scene::SceneObject;
use worldforge_core::state::{DynStateStore, StateStoreKind, WorldState};
use worldforge_core::types::{BBox, Position, Rotation, Velocity, VideoClip};
use worldforge_verify::{
    prove_guardrail_plan as prove_guardrail_plan_bundle,
    prove_inference_transition as prove_inference_transition_bundle, verify_bundle, verify_proof,
    VerificationBundle, ZkVerifier,
};

fn auto_detect_registry() -> Arc<worldforge_core::provider::ProviderRegistry> {
    Arc::new(worldforge_providers::auto_detect())
}

fn resolve_provider_name<'a>(state: &'a WorldState, provider: Option<&'a str>) -> &'a str {
    provider
        .filter(|name| !name.is_empty())
        .unwrap_or(state.metadata.created_by.as_str())
}

fn new_runtime() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Runtime::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
    })
}

fn parse_world_id(world_id: &str) -> PyResult<uuid::Uuid> {
    world_id
        .parse()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid world ID (must be UUID)"))
}

fn state_store_kind(
    state_backend: &str,
    state_dir: &str,
    state_db_path: Option<&str>,
) -> PyResult<StateStoreKind> {
    match state_backend {
        "file" => Ok(StateStoreKind::File(state_dir.into())),
        "sqlite" => Ok(StateStoreKind::Sqlite(
            state_db_path
                .map(Into::into)
                .unwrap_or_else(|| Path::new(state_dir).join("worldforge.db")),
        )),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown state backend: {other}. Available: file, sqlite"
        ))),
    }
}

fn parse_guardrails_json(guardrails_json: Option<&str>) -> PyResult<Vec<GuardrailConfig>> {
    guardrails_json
        .map(|json| {
            serde_json::from_str(json).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "failed to parse guardrails JSON: {e}"
                ))
            })
        })
        .transpose()
        .map(|value| value.unwrap_or_default())
}

#[allow(clippy::too_many_arguments)]
fn planner_from_args(
    planner: &str,
    max_steps: u32,
    num_samples: Option<u32>,
    top_k: Option<u32>,
    population_size: Option<u32>,
    elite_fraction: Option<f32>,
    num_iterations: Option<u32>,
    learning_rate: Option<f32>,
    horizon: Option<u32>,
    replanning_interval: Option<u32>,
) -> PyResult<PlannerType> {
    match planner {
        "sampling" => Ok(PlannerType::Sampling {
            num_samples: num_samples.unwrap_or(32).max(1),
            top_k: top_k.unwrap_or(5).max(1),
        }),
        "cem" => Ok(PlannerType::CEM {
            population_size: population_size.unwrap_or(64).max(4),
            elite_fraction: elite_fraction.unwrap_or(0.2).clamp(0.05, 1.0),
            num_iterations: num_iterations.unwrap_or(5).max(1),
        }),
        "mpc" => Ok(PlannerType::MPC {
            horizon: horizon.unwrap_or(max_steps).max(1).min(max_steps.max(1)),
            num_samples: num_samples.unwrap_or(32).max(4),
            replanning_interval: replanning_interval.unwrap_or(1).max(1),
        }),
        "gradient" => Ok(PlannerType::Gradient {
            learning_rate: learning_rate.unwrap_or(0.25).clamp(0.01, 1.0),
            num_iterations: num_iterations.unwrap_or(24).max(1),
        }),
        "provider-native" | "provider_native" | "native" => Ok(PlannerType::ProviderNative),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown planner: {other}. Available: sampling, cem, mpc, gradient, provider-native"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Spatial types
// ---------------------------------------------------------------------------

/// 3D position in world coordinates.
#[pyclass(name = "Position")]
#[derive(Debug, Clone)]
pub struct PyPosition {
    inner: Position,
}

#[pymethods]
impl PyPosition {
    #[new]
    #[pyo3(signature = (x=0.0, y=0.0, z=0.0))]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: Position { x, y, z },
        }
    }

    #[getter]
    fn x(&self) -> f32 {
        self.inner.x
    }

    #[setter]
    fn set_x(&mut self, x: f32) {
        self.inner.x = x;
    }

    #[getter]
    fn y(&self) -> f32 {
        self.inner.y
    }

    #[setter]
    fn set_y(&mut self, y: f32) {
        self.inner.y = y;
    }

    #[getter]
    fn z(&self) -> f32 {
        self.inner.z
    }

    #[setter]
    fn set_z(&mut self, z: f32) {
        self.inner.z = z;
    }

    fn __repr__(&self) -> String {
        format!(
            "Position(x={}, y={}, z={})",
            self.inner.x, self.inner.y, self.inner.z
        )
    }

    /// Convert to JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Create from JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: Position = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { inner })
    }
}

/// Quaternion rotation (w, x, y, z).
#[pyclass(name = "Rotation")]
#[derive(Debug, Clone)]
pub struct PyRotation {
    inner: Rotation,
}

#[pymethods]
impl PyRotation {
    #[new]
    #[pyo3(signature = (w=1.0, x=0.0, y=0.0, z=0.0))]
    fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: Rotation { w, x, y, z },
        }
    }

    #[getter]
    fn w(&self) -> f32 {
        self.inner.w
    }

    #[getter]
    fn x(&self) -> f32 {
        self.inner.x
    }

    #[getter]
    fn y(&self) -> f32 {
        self.inner.y
    }

    #[getter]
    fn z(&self) -> f32 {
        self.inner.z
    }

    /// Get the tilt angle in degrees from upright.
    fn tilt_degrees(&self) -> f32 {
        self.inner.tilt_degrees()
    }

    fn __repr__(&self) -> String {
        format!(
            "Rotation(w={}, x={}, y={}, z={})",
            self.inner.w, self.inner.x, self.inner.y, self.inner.z
        )
    }
}

/// Axis-aligned bounding box.
#[pyclass(name = "BBox")]
#[derive(Debug, Clone)]
pub struct PyBBox {
    inner: BBox,
}

#[pymethods]
impl PyBBox {
    #[new]
    fn new(min: &PyPosition, max: &PyPosition) -> Self {
        Self {
            inner: BBox {
                min: min.inner,
                max: max.inner,
            },
        }
    }

    #[getter]
    fn min(&self) -> PyPosition {
        PyPosition {
            inner: self.inner.min,
        }
    }

    #[getter]
    fn max(&self) -> PyPosition {
        PyPosition {
            inner: self.inner.max,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BBox(min=({}, {}, {}), max=({}, {}, {}))",
            self.inner.min.x,
            self.inner.min.y,
            self.inner.min.z,
            self.inner.max.x,
            self.inner.max.y,
            self.inner.max.z
        )
    }
}

/// 3D velocity vector.
#[pyclass(name = "Velocity")]
#[derive(Debug, Clone)]
pub struct PyVelocity {
    inner: Velocity,
}

#[pymethods]
impl PyVelocity {
    #[new]
    #[pyo3(signature = (x=0.0, y=0.0, z=0.0))]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: Velocity { x, y, z },
        }
    }

    #[getter]
    fn x(&self) -> f32 {
        self.inner.x
    }

    #[getter]
    fn y(&self) -> f32 {
        self.inner.y
    }

    #[getter]
    fn z(&self) -> f32 {
        self.inner.z
    }

    /// Get the speed (magnitude of velocity).
    fn magnitude(&self) -> f32 {
        self.inner.magnitude()
    }

    fn __repr__(&self) -> String {
        format!(
            "Velocity(x={}, y={}, z={})",
            self.inner.x, self.inner.y, self.inner.z
        )
    }
}

// ---------------------------------------------------------------------------
// Scene types
// ---------------------------------------------------------------------------

/// A physical object in the scene.
#[pyclass(name = "SceneObject")]
#[derive(Debug, Clone)]
pub struct PySceneObject {
    pub(crate) inner: SceneObject,
}

#[pymethods]
impl PySceneObject {
    /// Create a new scene object.
    #[new]
    fn new(name: &str, position: &PyPosition, bbox: &PyBBox) -> Self {
        let pose = worldforge_core::types::Pose {
            position: position.inner,
            rotation: Rotation::default(),
        };
        Self {
            inner: SceneObject::new(name, pose, bbox.inner),
        }
    }

    /// Get the object's unique ID as a string.
    #[getter]
    fn id(&self) -> String {
        self.inner.id.to_string()
    }

    /// Get the object's name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Get the object's position.
    #[getter]
    fn position(&self) -> PyPosition {
        PyPosition {
            inner: self.inner.pose.position,
        }
    }

    /// Set the object's position.
    #[setter]
    fn set_position(&mut self, pos: &PyPosition) {
        self.inner.pose.position = pos.inner;
    }

    /// Get the object's velocity.
    #[getter]
    fn velocity(&self) -> PyVelocity {
        PyVelocity {
            inner: self.inner.velocity,
        }
    }

    /// Set the object's velocity.
    #[setter]
    fn set_velocity(&mut self, vel: &PyVelocity) {
        self.inner.velocity = vel.inner;
    }

    /// Get the object's semantic label.
    #[getter]
    fn semantic_label(&self) -> Option<&str> {
        self.inner.semantic_label.as_deref()
    }

    /// Set the object's semantic label.
    #[setter]
    fn set_semantic_label(&mut self, label: Option<String>) {
        self.inner.semantic_label = label;
    }

    /// Set the object as static (immovable).
    fn set_static(&mut self, is_static: bool) {
        self.inner.physics.is_static = is_static;
    }

    /// Set the object's mass in kilograms.
    fn set_mass(&mut self, mass: f32) {
        self.inner.physics.mass = Some(mass);
    }

    fn __repr__(&self) -> String {
        format!(
            "SceneObject(name='{}', pos=({}, {}, {}))",
            self.inner.name,
            self.inner.pose.position.x,
            self.inner.pose.position.y,
            self.inner.pose.position.z
        )
    }
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

/// A WorldForge world instance.
///
/// Wraps the Rust WorldState with Python-friendly methods.
#[pyclass(name = "World")]
#[derive(Debug, Clone)]
pub struct PyWorld {
    state: WorldState,
}

#[pymethods]
impl PyWorld {
    /// Create a new empty world.
    #[new]
    #[pyo3(signature = (name, provider="mock"))]
    fn new(name: &str, provider: &str) -> Self {
        Self {
            state: WorldState::new(name, provider),
        }
    }

    /// Get the world's unique ID.
    #[getter]
    fn id(&self) -> String {
        self.state.id.to_string()
    }

    /// Get the world's name.
    #[getter]
    fn name(&self) -> &str {
        &self.state.metadata.name
    }

    /// Get the current simulation step.
    #[getter]
    fn step(&self) -> u64 {
        self.state.time.step
    }

    /// Get the current simulation time in seconds.
    #[getter]
    fn time_seconds(&self) -> f64 {
        self.state.time.seconds
    }

    /// Get the number of objects in the scene.
    #[getter]
    fn object_count(&self) -> usize {
        self.state.scene.objects.len()
    }

    /// Add an object to the world.
    fn add_object(&mut self, obj: &PySceneObject) {
        self.state.scene.add_object(obj.inner.clone());
    }

    /// Get an object by name.
    fn get_object(&self, name: &str) -> Option<PySceneObject> {
        self.state
            .scene
            .objects
            .values()
            .find(|o| o.name == name)
            .map(|o| PySceneObject { inner: o.clone() })
    }

    /// Remove an object by name. Returns True if found.
    fn remove_object(&mut self, name: &str) -> bool {
        if let Some(id) = self
            .state
            .scene
            .objects
            .values()
            .find(|o| o.name == name)
            .map(|o| o.id)
        {
            self.state.scene.remove_object(&id);
            true
        } else {
            false
        }
    }

    /// List all object names in the scene.
    fn list_objects(&self) -> Vec<String> {
        self.state
            .scene
            .objects
            .values()
            .map(|o| o.name.clone())
            .collect()
    }

    /// Get the number of history entries.
    #[getter]
    fn history_length(&self) -> usize {
        self.state.history.len()
    }

    /// Export the world state as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.state).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Import a world state from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let state: WorldState = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { state })
    }

    fn __repr__(&self) -> String {
        format!(
            "World(name='{}', objects={}, step={})",
            self.state.metadata.name,
            self.state.scene.objects.len(),
            self.state.time.step
        )
    }

    /// Predict the next world state after applying an action.
    #[pyo3(signature = (action, steps=1, provider=None, fallback_provider=None, return_video=false, max_latency_ms=None))]
    fn predict(
        &mut self,
        action: &PyAction,
        steps: u32,
        provider: Option<&str>,
        fallback_provider: Option<&str>,
        return_video: bool,
        max_latency_ms: Option<u64>,
    ) -> PyResult<PyPrediction> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
        })?;

        let provider_name = resolve_provider_name(&self.state, provider);
        let mut world = worldforge_core::world::World::new(
            self.state.clone(),
            provider_name,
            auto_detect_registry(),
        );
        let config = PredictionConfig {
            steps,
            return_video,
            max_latency_ms,
            fallback_provider: fallback_provider.map(ToOwned::to_owned),
            ..PredictionConfig::default()
        };

        let prediction = rt
            .block_on(world.predict(&action.inner, &config))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("prediction failed: {e}"))
            })?;

        self.state = world.current_state().clone();

        Ok(PyPrediction { inner: prediction })
    }

    /// Plan a sequence of actions to achieve a natural-language goal.
    #[pyo3(signature = (goal, max_steps=10, timeout_seconds=30.0, provider=None, planner="sampling", num_samples=None, top_k=None, population_size=None, elite_fraction=None, num_iterations=None, learning_rate=None, horizon=None, replanning_interval=None, guardrails_json=None))]
    #[allow(clippy::too_many_arguments)]
    fn plan(
        &self,
        goal: &str,
        max_steps: u32,
        timeout_seconds: f64,
        provider: Option<&str>,
        planner: &str,
        num_samples: Option<u32>,
        top_k: Option<u32>,
        population_size: Option<u32>,
        elite_fraction: Option<f32>,
        num_iterations: Option<u32>,
        learning_rate: Option<f32>,
        horizon: Option<u32>,
        replanning_interval: Option<u32>,
        guardrails_json: Option<&str>,
    ) -> PyResult<PyPlan> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
        })?;

        let provider_name = resolve_provider_name(&self.state, provider);
        let world = worldforge_core::world::World::new(
            self.state.clone(),
            provider_name,
            auto_detect_registry(),
        );
        let planner = planner_from_args(
            planner,
            max_steps,
            num_samples,
            top_k,
            population_size,
            elite_fraction,
            num_iterations,
            learning_rate,
            horizon,
            replanning_interval,
        )?;
        let request = worldforge_core::prediction::PlanRequest {
            current_state: self.state.clone(),
            goal: worldforge_core::prediction::PlanGoal::Description(goal.to_string()),
            max_steps,
            guardrails: parse_guardrails_json(guardrails_json)?,
            planner,
            timeout_seconds,
        };

        let plan = rt.block_on(world.plan(&request)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("planning failed: {e}"))
        })?;

        Ok(PyPlan { inner: plan })
    }

    /// Ask a provider to reason about the current world state.
    #[pyo3(signature = (query, provider=None))]
    fn reason(&self, query: &str, provider: Option<&str>) -> PyResult<PyReasoningOutput> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
        })?;

        let provider_name = resolve_provider_name(&self.state, provider);
        let world = worldforge_core::world::World::new(
            self.state.clone(),
            provider_name,
            auto_detect_registry(),
        );

        let output = rt.block_on(world.reason(query)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("reasoning failed: {e}"))
        })?;

        Ok(PyReasoningOutput { inner: output })
    }
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/// The result of a world-model prediction.
#[pyclass(name = "Prediction")]
#[derive(Debug, Clone)]
pub struct PyPrediction {
    inner: worldforge_core::prediction::Prediction,
}

#[pymethods]
impl PyPrediction {
    /// Prediction identifier.
    #[getter]
    fn id(&self) -> String {
        self.inner.id.to_string()
    }

    /// Provider that generated the prediction.
    #[getter]
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    /// Model identifier used by the provider.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// Prediction confidence score.
    #[getter]
    fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    /// Overall physics plausibility score.
    #[getter]
    fn physics_score(&self) -> f32 {
        self.inner.physics_scores.overall
    }

    /// Provider latency in milliseconds.
    #[getter]
    fn latency_ms(&self) -> u64 {
        self.inner.latency_ms
    }

    /// Number of evaluated guardrails included in this prediction.
    #[getter]
    fn guardrail_count(&self) -> usize {
        self.inner.guardrail_results.len()
    }

    /// Get the predicted output state as a `World`.
    fn output_world(&self) -> PyWorld {
        PyWorld {
            state: self.inner.output_state.clone(),
        }
    }

    /// Serialize the prediction to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Prediction(provider='{}', confidence={:.2}, physics_score={:.2})",
            self.inner.provider, self.inner.confidence, self.inner.physics_scores.overall
        )
    }
}

/// A generated or transferred video clip.
#[pyclass(name = "VideoClip")]
#[derive(Debug, Clone)]
pub struct PyVideoClip {
    inner: VideoClip,
}

#[pymethods]
impl PyVideoClip {
    /// Number of frames in the clip.
    #[getter]
    fn frame_count(&self) -> usize {
        self.inner.frames.len()
    }

    /// Frames per second.
    #[getter]
    fn fps(&self) -> f32 {
        self.inner.fps
    }

    /// Resolution as `(width, height)`.
    #[getter]
    fn resolution(&self) -> (u32, u32) {
        self.inner.resolution
    }

    /// Duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration
    }

    /// Serialize the clip to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Deserialize a clip from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: VideoClip = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoClip(frames={}, fps={:.1}, resolution=({}, {}), duration={:.2}s)",
            self.inner.frames.len(),
            self.inner.fps,
            self.inner.resolution.0,
            self.inner.resolution.1,
            self.inner.duration
        )
    }
}

/// Output of a provider reasoning query.
#[pyclass(name = "ReasoningOutput")]
#[derive(Debug, Clone)]
pub struct PyReasoningOutput {
    inner: worldforge_core::provider::ReasoningOutput,
}

#[pymethods]
impl PyReasoningOutput {
    /// Natural-language answer.
    #[getter]
    fn answer(&self) -> &str {
        &self.inner.answer
    }

    /// Confidence score.
    #[getter]
    fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    /// Evidence returned by the provider.
    #[getter]
    fn evidence(&self) -> Vec<String> {
        self.inner.evidence.clone()
    }

    /// Serialize the reasoning output to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ReasoningOutput(confidence={:.2}, answer='{}')",
            self.inner.confidence, self.inner.answer
        )
    }
}

// ---------------------------------------------------------------------------
// Action types
// ---------------------------------------------------------------------------

/// A standardized action in the WorldForge action system.
///
/// Use factory methods like `Action.move_to()`, `Action.grasp()`, etc.
#[pyclass(name = "Action")]
#[derive(Debug, Clone)]
pub struct PyAction {
    inner: worldforge_core::action::Action,
}

#[pymethods]
impl PyAction {
    /// Create a Move action to a target position.
    #[staticmethod]
    #[pyo3(signature = (x, y, z, speed=1.0))]
    fn move_to(x: f32, y: f32, z: f32, speed: f32) -> Self {
        Self {
            inner: worldforge_core::action::Action::Move {
                target: worldforge_core::types::Position { x, y, z },
                speed,
            },
        }
    }

    /// Create a Grasp action on an object by ID.
    #[staticmethod]
    #[pyo3(signature = (object_id, grip_force=1.0))]
    fn grasp(object_id: &str, grip_force: f32) -> PyResult<Self> {
        let id: uuid::Uuid = object_id.parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("invalid object ID (must be UUID)")
        })?;
        Ok(Self {
            inner: worldforge_core::action::Action::Grasp {
                object: id,
                grip_force,
            },
        })
    }

    /// Create a Release action on an object by ID.
    #[staticmethod]
    fn release(object_id: &str) -> PyResult<Self> {
        let id: uuid::Uuid = object_id.parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("invalid object ID (must be UUID)")
        })?;
        Ok(Self {
            inner: worldforge_core::action::Action::Release { object: id },
        })
    }

    /// Create a Push action on an object.
    #[staticmethod]
    #[pyo3(signature = (object_id, dx, dy, dz, force=1.0))]
    fn push(object_id: &str, dx: f32, dy: f32, dz: f32, force: f32) -> PyResult<Self> {
        let id: uuid::Uuid = object_id.parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("invalid object ID (must be UUID)")
        })?;
        Ok(Self {
            inner: worldforge_core::action::Action::Push {
                object: id,
                direction: worldforge_core::types::Vec3 {
                    x: dx,
                    y: dy,
                    z: dz,
                },
                force,
            },
        })
    }

    /// Create a Place action for an object at a target position.
    #[staticmethod]
    fn place(object_id: &str, x: f32, y: f32, z: f32) -> PyResult<Self> {
        let id: uuid::Uuid = object_id.parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("invalid object ID (must be UUID)")
        })?;
        Ok(Self {
            inner: worldforge_core::action::Action::Place {
                object: id,
                target: worldforge_core::types::Position { x, y, z },
            },
        })
    }

    /// Create a SetWeather action.
    #[staticmethod]
    fn set_weather(weather: &str) -> PyResult<Self> {
        let w = match weather.to_lowercase().as_str() {
            "clear" => worldforge_core::action::Weather::Clear,
            "cloudy" => worldforge_core::action::Weather::Cloudy,
            "rain" => worldforge_core::action::Weather::Rain,
            "snow" => worldforge_core::action::Weather::Snow,
            "fog" => worldforge_core::action::Weather::Fog,
            "night" => worldforge_core::action::Weather::Night,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown weather: {other}"
                )))
            }
        };
        Ok(Self {
            inner: worldforge_core::action::Action::SetWeather { weather: w },
        })
    }

    /// Create a SetLighting action.
    #[staticmethod]
    fn set_lighting(time_of_day: f32) -> Self {
        Self {
            inner: worldforge_core::action::Action::SetLighting { time_of_day },
        }
    }

    /// Create a SpawnObject action.
    #[staticmethod]
    fn spawn_object(template: &str) -> Self {
        Self {
            inner: worldforge_core::action::Action::SpawnObject {
                template: template.to_string(),
                pose: worldforge_core::types::Pose::default(),
            },
        }
    }

    /// Create a Sequence of actions.
    #[staticmethod]
    fn sequence(actions: Vec<PyAction>) -> Self {
        Self {
            inner: worldforge_core::action::Action::Sequence(
                actions.into_iter().map(|a| a.inner).collect(),
            ),
        }
    }

    /// Create a raw provider-specific action from JSON.
    #[staticmethod]
    fn raw(provider: &str, data: &str) -> PyResult<Self> {
        let value: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid JSON: {e}")))?;
        Ok(Self {
            inner: worldforge_core::action::Action::Raw {
                provider: provider.to_string(),
                data: value,
            },
        })
    }

    /// Serialize the action to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Deserialize an action from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: worldforge_core::action::Action = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("Action({:?})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// Guardrail types
// ---------------------------------------------------------------------------

/// A safety guardrail that constrains predictions.
#[pyclass(name = "Guardrail")]
#[derive(Debug, Clone)]
pub struct PyGuardrail {
    inner: worldforge_core::guardrail::Guardrail,
}

#[pymethods]
impl PyGuardrail {
    /// No object may pass through another.
    #[staticmethod]
    fn no_collisions() -> Self {
        Self {
            inner: worldforge_core::guardrail::Guardrail::NoCollisions,
        }
    }

    /// Specified objects must stay upright within max tilt degrees.
    #[staticmethod]
    #[pyo3(signature = (object_ids, max_tilt_degrees=45.0))]
    fn stay_upright(object_ids: Vec<String>, max_tilt_degrees: f32) -> PyResult<Self> {
        let ids: std::result::Result<Vec<uuid::Uuid>, _> =
            object_ids.iter().map(|s| s.parse()).collect();
        let ids = ids.map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("all object IDs must be valid UUIDs")
        })?;
        Ok(Self {
            inner: worldforge_core::guardrail::Guardrail::StayUpright {
                objects: ids,
                max_tilt_degrees,
            },
        })
    }

    /// No object may leave the specified bounding box.
    #[staticmethod]
    fn boundary_constraint(bbox: &PyBBox) -> Self {
        Self {
            inner: worldforge_core::guardrail::Guardrail::BoundaryConstraint { bounds: bbox.inner },
        }
    }

    /// Energy must be conserved within a tolerance.
    #[staticmethod]
    #[pyo3(signature = (tolerance=0.1))]
    fn energy_conservation(tolerance: f32) -> Self {
        Self {
            inner: worldforge_core::guardrail::Guardrail::EnergyConservation { tolerance },
        }
    }

    /// Maximum velocity for any object.
    #[staticmethod]
    fn max_velocity(limit: f32) -> Self {
        Self {
            inner: worldforge_core::guardrail::Guardrail::MaxVelocity { limit },
        }
    }

    /// Human safety zone: no robot action within radius of human.
    #[staticmethod]
    fn human_safety_zone(radius: f32) -> Self {
        Self {
            inner: worldforge_core::guardrail::Guardrail::HumanSafetyZone { radius },
        }
    }

    /// Serialize the guardrail to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!("Guardrail({:?})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// WorldForge orchestrator
// ---------------------------------------------------------------------------

/// The main WorldForge orchestrator.
///
/// Manages provider registration and world creation.
#[pyclass(name = "WorldForge")]
pub struct PyWorldForge {
    inner: worldforge_core::WorldForge,
    store: DynStateStore,
}

#[pymethods]
impl PyWorldForge {
    /// Create a new WorldForge instance with auto-detected providers.
    #[new]
    #[pyo3(signature = (state_backend="file", state_dir=".worldforge", state_db_path=None))]
    fn new(state_backend: &str, state_dir: &str, state_db_path: Option<&str>) -> PyResult<Self> {
        let store_kind = state_store_kind(state_backend, state_dir, state_db_path)?;
        let rt = new_runtime()?;
        let store = rt.block_on(store_kind.open()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to open state store: {e}"))
        })?;
        let mut wf = worldforge_core::WorldForge::new();
        for provider in worldforge_providers::auto_detect().into_providers() {
            let _ = wf.register_provider(provider);
        }
        Ok(Self { inner: wf, store })
    }

    /// List all registered provider names.
    fn providers(&self) -> Vec<String> {
        self.inner
            .providers()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Create a new world with the given name and provider.
    #[pyo3(signature = (name, provider="mock"))]
    fn create_world(&self, name: &str, provider: &str) -> PyResult<PyWorld> {
        let world = self.inner.create_world(name, provider).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create world: {e}"))
        })?;
        Ok(PyWorld {
            state: world.current_state().clone(),
        })
    }

    /// Persist a world snapshot to the configured state store.
    fn save_world(&self, world: &PyWorld) -> PyResult<String> {
        let rt = new_runtime()?;
        rt.block_on(self.store.save(&world.state)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to save world: {e}"))
        })?;
        Ok(world.state.id.to_string())
    }

    /// Load a world snapshot from the configured state store.
    fn load_world(&self, world_id: &str) -> PyResult<PyWorld> {
        let id = parse_world_id(world_id)?;
        let rt = new_runtime()?;
        let state = rt.block_on(self.store.load(&id)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to load world: {e}"))
        })?;
        Ok(PyWorld { state })
    }

    /// List all persisted world IDs in the configured state store.
    fn list_worlds(&self) -> PyResult<Vec<String>> {
        let rt = new_runtime()?;
        let ids = rt.block_on(self.store.list()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to list worlds: {e}"))
        })?;
        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }

    /// Delete a persisted world snapshot by ID.
    fn delete_world(&self, world_id: &str) -> PyResult<()> {
        let id = parse_world_id(world_id)?;
        let rt = new_runtime()?;
        rt.block_on(self.store.delete(&id)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to delete world: {e}"))
        })?;
        Ok(())
    }

    /// Generate a video clip directly from a prompt with a specific provider.
    #[pyo3(signature = (prompt, provider="mock", duration_seconds=4.0, width=1280, height=720, fps=24.0, temperature=1.0, seed=None, negative_prompt=None))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        prompt: &str,
        provider: &str,
        duration_seconds: f64,
        width: u32,
        height: u32,
        fps: f32,
        temperature: f32,
        seed: Option<u64>,
        negative_prompt: Option<&str>,
    ) -> PyResult<PyVideoClip> {
        let provider_ref = self.inner.registry().get(provider).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("generation failed: {e}"))
        })?;
        let prompt = GenerationPrompt {
            text: prompt.to_string(),
            reference_image: None,
            negative_prompt: negative_prompt.map(ToOwned::to_owned),
        };
        let config = GenerationConfig {
            resolution: (width, height),
            fps,
            duration_seconds,
            temperature,
            seed,
        };
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
        })?;
        let clip = rt
            .block_on(provider_ref.generate(&prompt, &config))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("generation failed: {e}"))
            })?;
        Ok(PyVideoClip { inner: clip })
    }

    /// Transfer spatial controls over an existing clip with a specific provider.
    #[pyo3(signature = (clip, provider="mock", controls_json=None, width=1280, height=720, fps=24.0, control_strength=0.8))]
    #[allow(clippy::too_many_arguments)]
    fn transfer(
        &self,
        clip: &PyVideoClip,
        provider: &str,
        controls_json: Option<&str>,
        width: u32,
        height: u32,
        fps: f32,
        control_strength: f32,
    ) -> PyResult<PyVideoClip> {
        let provider_ref = self.inner.registry().get(provider).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("transfer failed: {e}"))
        })?;
        let controls = match controls_json {
            Some(json) => serde_json::from_str::<SpatialControls>(json).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid controls JSON: {e}"))
            })?,
            None => SpatialControls::default(),
        };
        let config = TransferConfig {
            resolution: (width, height),
            fps,
            control_strength,
        };
        let rt = new_runtime()?;
        let clip = rt
            .block_on(provider_ref.transfer(&clip.inner, &controls, &config))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("transfer failed: {e}"))
            })?;
        Ok(PyVideoClip { inner: clip })
    }

    fn __repr__(&self) -> String {
        format!("WorldForge(providers={:?})", self.inner.providers())
    }
}

// ---------------------------------------------------------------------------
// Planning types
// ---------------------------------------------------------------------------

/// Result of a planning operation.
#[pyclass(name = "Plan")]
#[derive(Debug, Clone)]
pub struct PyPlan {
    inner: worldforge_core::prediction::Plan,
}

#[pymethods]
impl PyPlan {
    /// Number of actions in the plan.
    #[getter]
    fn action_count(&self) -> usize {
        self.inner.actions.len()
    }

    /// Probability of success (0.0–1.0).
    #[getter]
    fn success_probability(&self) -> f32 {
        self.inner.success_probability
    }

    /// Time taken for planning in milliseconds.
    #[getter]
    fn planning_time_ms(&self) -> u64 {
        self.inner.planning_time_ms
    }

    /// Number of planner iterations used.
    #[getter]
    fn iterations_used(&self) -> u32 {
        self.inner.iterations_used
    }

    /// Total estimated cost.
    #[getter]
    fn total_cost(&self) -> f32 {
        self.inner.total_cost
    }

    /// Get actions as JSON array.
    fn actions_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.actions).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Serialize the full plan to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Deserialize a plan from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: worldforge_core::prediction::Plan = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Plan(actions={}, success_prob={:.2}, time={}ms)",
            self.inner.actions.len(),
            self.inner.success_probability,
            self.inner.planning_time_ms
        )
    }
}

/// Plan a sequence of actions in a world.
///
/// Supports sampling, CEM, MPC, gradient, and provider-native planning.
#[pyfunction]
#[pyo3(signature = (world, goal, max_steps=10, timeout_seconds=30.0, provider="mock", planner="sampling", num_samples=None, top_k=None, population_size=None, elite_fraction=None, num_iterations=None, learning_rate=None, horizon=None, replanning_interval=None, guardrails_json=None))]
#[allow(clippy::too_many_arguments)]
fn plan(
    world: &PyWorld,
    goal: &str,
    max_steps: u32,
    timeout_seconds: f64,
    provider: &str,
    planner: &str,
    num_samples: Option<u32>,
    top_k: Option<u32>,
    population_size: Option<u32>,
    elite_fraction: Option<f32>,
    num_iterations: Option<u32>,
    learning_rate: Option<f32>,
    horizon: Option<u32>,
    replanning_interval: Option<u32>,
    guardrails_json: Option<&str>,
) -> PyResult<PyPlan> {
    world.plan(
        goal,
        max_steps,
        timeout_seconds,
        Some(provider),
        planner,
        num_samples,
        top_k,
        population_size,
        elite_fraction,
        num_iterations,
        learning_rate,
        horizon,
        replanning_interval,
        guardrails_json,
    )
}

// ---------------------------------------------------------------------------
// Evaluation types
// ---------------------------------------------------------------------------

/// A single evaluation result entry.
#[pyclass(name = "EvalResult")]
#[derive(Debug, Clone)]
pub struct PyEvalResult {
    /// Provider name.
    provider: String,
    /// Average score.
    average_score: f32,
    /// Average latency in ms.
    average_latency_ms: u64,
    /// Number of scenarios passed.
    scenarios_passed: usize,
    /// Total scenarios.
    total_scenarios: usize,
}

#[pymethods]
impl PyEvalResult {
    #[getter]
    fn provider(&self) -> &str {
        &self.provider
    }

    #[getter]
    fn average_score(&self) -> f32 {
        self.average_score
    }

    #[getter]
    fn average_latency_ms(&self) -> u64 {
        self.average_latency_ms
    }

    #[getter]
    fn scenarios_passed(&self) -> usize {
        self.scenarios_passed
    }

    #[getter]
    fn total_scenarios(&self) -> usize {
        self.total_scenarios
    }

    fn __repr__(&self) -> String {
        format!(
            "EvalResult(provider='{}', score={:.2}, passed={}/{})",
            self.provider, self.average_score, self.scenarios_passed, self.total_scenarios
        )
    }
}

/// Run an evaluation suite against the mock provider.
///
/// Returns a list of EvalResult entries from the leaderboard.
#[pyfunction]
#[pyo3(signature = (suite_name="physics"))]
fn run_eval(suite_name: &str) -> PyResult<Vec<PyEvalResult>> {
    let suite = match suite_name {
        "physics" => worldforge_eval::EvalSuite::physics_standard(),
        "manipulation" => worldforge_eval::EvalSuite::manipulation_standard(),
        "spatial" => worldforge_eval::EvalSuite::spatial_reasoning(),
        "comprehensive" => worldforge_eval::EvalSuite::comprehensive(),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown eval suite: {other}. Available: physics, manipulation, spatial, comprehensive"
        )))
        }
    };

    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
    })?;

    let mock = worldforge_providers::MockProvider::new();
    let provider_list: Vec<&dyn worldforge_core::provider::WorldModelProvider> = vec![&mock];

    let report = rt.block_on(suite.run(&provider_list)).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("evaluation failed: {e}"))
    })?;

    Ok(report
        .leaderboard
        .iter()
        .map(|entry| PyEvalResult {
            provider: entry.provider.clone(),
            average_score: entry.average_score,
            average_latency_ms: entry.average_latency_ms,
            scenarios_passed: entry.scenarios_passed,
            total_scenarios: entry.total_scenarios,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// ZK Verification types
// ---------------------------------------------------------------------------

/// A ZK proof generated by the verification layer.
#[pyclass(name = "ZkProof")]
#[derive(Debug, Clone)]
pub struct PyZkProof {
    inner: worldforge_verify::ZkProof,
}

#[pymethods]
impl PyZkProof {
    /// Deserialize a proof from JSON.
    #[staticmethod]
    fn from_json(proof_json: &str) -> PyResult<Self> {
        let inner = serde_json::from_str(proof_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid proof JSON: {e}"))
        })?;
        Ok(Self { inner })
    }

    /// Size of the proof data in bytes.
    #[getter]
    fn proof_size(&self) -> usize {
        self.inner.proof_data.len()
    }

    /// Backend that generated this proof.
    #[getter]
    fn backend(&self) -> String {
        format!("{:?}", self.inner.backend)
    }

    /// Time taken to generate the proof in milliseconds.
    #[getter]
    fn generation_time_ms(&self) -> u64 {
        self.inner.generation_time_ms
    }

    /// Verify this proof and return (valid, details).
    fn verify(&self) -> PyResult<(bool, String)> {
        let verifier = worldforge_verify::MockVerifier::new();
        let result = verify_proof(&verifier, &self.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
        })?;
        Ok((result.valid, result.details))
    }

    /// Serialize the proof to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ZkProof(backend={:?}, size={} bytes)",
            self.inner.backend,
            self.inner.proof_data.len()
        )
    }
}

/// Generate a ZK proof for inference verification.
#[pyfunction]
fn prove_inference(
    model_data: &[u8],
    input_data: &[u8],
    output_data: &[u8],
) -> PyResult<PyZkProof> {
    let verifier = worldforge_verify::MockVerifier::new();
    let model_hash = worldforge_verify::sha256_hash(model_data);
    let input_hash = worldforge_verify::sha256_hash(input_data);
    let output_hash = worldforge_verify::sha256_hash(output_data);

    let proof = verifier
        .prove_inference(model_hash, input_hash, output_hash)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
        })?;

    Ok(PyZkProof { inner: proof })
}

/// Generate an inference proof from two serialized `WorldState` payloads.
#[pyfunction]
#[pyo3(signature = (input_state_json, output_state_json, provider=None))]
fn prove_inference_transition(
    input_state_json: &str,
    output_state_json: &str,
    provider: Option<&str>,
) -> PyResult<PyZkProof> {
    let verifier = worldforge_verify::MockVerifier::new();
    let input_state: WorldState = serde_json::from_str(input_state_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid input state JSON: {e}"))
    })?;
    let output_state: WorldState = serde_json::from_str(output_state_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid output state JSON: {e}"))
    })?;
    let provider_name = provider.unwrap_or(output_state.metadata.created_by.as_str());

    let bundle =
        prove_inference_transition_bundle(&verifier, provider_name, &input_state, &output_state)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
            })?;

    Ok(PyZkProof {
        inner: bundle.proof,
    })
}

/// Generate a guardrail-compliance proof from a `Plan`.
#[pyfunction]
fn prove_guardrail_plan(plan: &PyPlan) -> PyResult<PyZkProof> {
    let verifier = worldforge_verify::MockVerifier::new();
    let bundle = prove_guardrail_plan_bundle(&verifier, &plan.inner).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
    })?;

    Ok(PyZkProof {
        inner: bundle.proof,
    })
}

/// Generate a ZK proof for data provenance.
#[pyfunction]
fn prove_provenance(data: &[u8], timestamp: u64, source: &[u8]) -> PyResult<PyZkProof> {
    let verifier = worldforge_verify::MockVerifier::new();
    let data_hash = worldforge_verify::sha256_hash(data);
    let source_commitment = worldforge_verify::sha256_hash(source);

    let proof = verifier
        .prove_data_provenance(data_hash, timestamp, source_commitment)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
        })?;

    Ok(PyZkProof { inner: proof })
}

/// Re-verify a proof serialized as JSON.
#[pyfunction]
fn verify_proof_json(proof_json: &str) -> PyResult<(bool, String)> {
    let proof = PyZkProof::from_json(proof_json)?;
    proof.verify()
}

/// Re-verify a serialized verification bundle and return a JSON report.
#[pyfunction]
fn verify_bundle_json(bundle_json: &str, bundle_type: &str) -> PyResult<String> {
    let verifier = worldforge_verify::MockVerifier::new();

    let report_json = match bundle_type {
        "inference" => {
            let bundle: VerificationBundle<worldforge_verify::InferenceArtifact> =
                serde_json::from_str(bundle_json).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid inference bundle JSON: {e}"
                    ))
                })?;
            serde_json::to_string_pretty(&verify_bundle(&verifier, &bundle).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
            })?)
        }
        "guardrail" => {
            let bundle: VerificationBundle<worldforge_verify::GuardrailArtifact> =
                serde_json::from_str(bundle_json).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid guardrail bundle JSON: {e}"
                    ))
                })?;
            serde_json::to_string_pretty(&verify_bundle(&verifier, &bundle).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
            })?)
        }
        "provenance" => {
            let bundle: VerificationBundle<worldforge_verify::ProvenanceArtifact> =
                serde_json::from_str(bundle_json).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid provenance bundle JSON: {e}"
                    ))
                })?;
            serde_json::to_string_pretty(&verify_bundle(&verifier, &bundle).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
            })?)
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown bundle_type: {other}. Available: inference, guardrail, provenance"
            )));
        }
    }
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}")))?;

    Ok(report_json)
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// WorldForge Python module.
///
/// Provides Python bindings for the WorldForge world foundation model
/// orchestration layer.
#[pymodule]
fn worldforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPosition>()?;
    m.add_class::<PyRotation>()?;
    m.add_class::<PyBBox>()?;
    m.add_class::<PyVelocity>()?;
    m.add_class::<PySceneObject>()?;
    m.add_class::<PyWorld>()?;
    m.add_class::<PyPrediction>()?;
    m.add_class::<PyVideoClip>()?;
    m.add_class::<PyReasoningOutput>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyGuardrail>()?;
    m.add_class::<PyWorldForge>()?;
    m.add_class::<PyPlan>()?;
    m.add_class::<PyEvalResult>()?;
    m.add_class::<PyZkProof>()?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(run_eval, m)?)?;
    m.add_function(wrap_pyfunction!(prove_inference, m)?)?;
    m.add_function(wrap_pyfunction!(prove_inference_transition, m)?)?;
    m.add_function(wrap_pyfunction!(prove_guardrail_plan, m)?)?;
    m.add_function(wrap_pyfunction!(prove_provenance, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_json, m)?)?;
    m.add_function(wrap_pyfunction!(verify_bundle_json, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_worldforge() -> PyWorldForge {
        PyWorldForge::new("file", ".worldforge-python-tests", None).unwrap()
    }

    #[test]
    fn test_position_create() {
        let pos = PyPosition::new(1.0, 2.0, 3.0);
        assert_eq!(pos.x(), 1.0);
        assert_eq!(pos.y(), 2.0);
        assert_eq!(pos.z(), 3.0);
    }

    #[test]
    fn test_position_repr() {
        let pos = PyPosition::new(1.0, 2.0, 3.0);
        assert!(pos.__repr__().contains("1"));
    }

    #[test]
    fn test_position_json_roundtrip() {
        let pos = PyPosition::new(1.5, 2.5, 3.5);
        let json = pos.to_json().unwrap();
        let pos2 = PyPosition::from_json(&json).unwrap();
        assert_eq!(pos2.x(), 1.5);
        assert_eq!(pos2.y(), 2.5);
        assert_eq!(pos2.z(), 3.5);
    }

    #[test]
    fn test_rotation_defaults() {
        let rot = PyRotation::new(1.0, 0.0, 0.0, 0.0);
        assert_eq!(rot.w(), 1.0);
        assert!(rot.tilt_degrees() < 0.01);
    }

    #[test]
    fn test_velocity_magnitude() {
        let vel = PyVelocity::new(3.0, 4.0, 0.0);
        assert!((vel.magnitude() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bbox_create() {
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        assert_eq!(bbox.min().x(), -1.0);
        assert_eq!(bbox.max().x(), 1.0);
    }

    #[test]
    fn test_scene_object_create() {
        let pos = PyPosition::new(0.0, 1.0, 0.0);
        let min = PyPosition::new(-0.5, 0.5, -0.5);
        let max = PyPosition::new(0.5, 1.5, 0.5);
        let bbox = PyBBox::new(&min, &max);
        let obj = PySceneObject::new("mug", &pos, &bbox);
        assert_eq!(obj.name(), "mug");
        assert_eq!(obj.position().y(), 1.0);
    }

    #[test]
    fn test_scene_object_velocity() {
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        let mut obj = PySceneObject::new("ball", &pos, &bbox);

        let vel = PyVelocity::new(1.0, 2.0, 3.0);
        obj.set_velocity(&vel);
        assert_eq!(obj.velocity().x(), 1.0);
    }

    #[test]
    fn test_world_create() {
        let world = PyWorld::new("test_world", "mock");
        assert_eq!(world.name(), "test_world");
        assert_eq!(world.object_count(), 0);
        assert_eq!(world.step(), 0);
    }

    #[test]
    fn test_world_add_objects() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        let obj = PySceneObject::new("cube", &pos, &bbox);

        world.add_object(&obj);
        assert_eq!(world.object_count(), 1);
        assert!(world.get_object("cube").is_some());
        assert!(world.get_object("nonexistent").is_none());
    }

    #[test]
    fn test_world_list_objects() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);

        world.add_object(&PySceneObject::new("a", &pos, &bbox));
        world.add_object(&PySceneObject::new("b", &pos, &bbox));

        let names = world.list_objects();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }

    #[test]
    fn test_world_remove_object() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        world.add_object(&PySceneObject::new("cube", &pos, &bbox));

        assert!(world.remove_object("cube"));
        assert_eq!(world.object_count(), 0);
        assert!(!world.remove_object("cube")); // already removed
    }

    #[test]
    fn test_world_json_roundtrip() {
        let mut world = PyWorld::new("test_world", "mock");
        let pos = PyPosition::new(1.0, 2.0, 3.0);
        let min = PyPosition::new(-0.5, -0.5, -0.5);
        let max = PyPosition::new(0.5, 0.5, 0.5);
        let bbox = PyBBox::new(&min, &max);
        world.add_object(&PySceneObject::new("ball", &pos, &bbox));

        let json = world.to_json().unwrap();
        let world2 = PyWorld::from_json(&json).unwrap();
        assert_eq!(world2.name(), "test_world");
        assert_eq!(world2.object_count(), 1);
    }

    #[test]
    fn test_world_predict_updates_state() {
        let mut world = PyWorld::new("predict_world", "mock");
        let prediction = world
            .predict(
                &PyAction::move_to(1.0, 0.0, 0.0, 1.0),
                1,
                None,
                None,
                false,
                None,
            )
            .unwrap();

        assert_eq!(prediction.provider(), "mock");
        assert_eq!(world.step(), 1);
        assert_eq!(world.history_length(), 1);
    }

    #[test]
    fn test_world_predict_uses_fallback_provider() {
        let mut world = PyWorld::new("predict_world", "missing");
        let prediction = world
            .predict(
                &PyAction::move_to(1.0, 0.0, 0.0, 1.0),
                1,
                None,
                Some("mock"),
                false,
                None,
            )
            .unwrap();

        assert_eq!(prediction.provider(), "mock");
        assert_eq!(world.step(), 1);
    }

    #[test]
    fn test_world_reason() {
        let world = PyWorld::new("reason_world", "mock");
        let output = world.reason("will it fall?", None).unwrap();
        assert!(output.answer().contains("will it fall?"));
        assert!(output.confidence() > 0.0);
        assert_eq!(output.evidence(), vec!["mock evidence".to_string()]);
    }

    // --- Action tests ---

    #[test]
    fn test_action_move_to() {
        let action = PyAction::move_to(1.0, 2.0, 3.0, 1.5);
        let json = action.to_json().unwrap();
        assert!(json.contains("Move"));
        let action2 = PyAction::from_json(&json).unwrap();
        let json2 = action2.to_json().unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn test_action_set_weather() {
        let action = PyAction::set_weather("rain").unwrap();
        let json = action.to_json().unwrap();
        assert!(json.contains("Rain"));
    }

    #[test]
    fn test_action_set_weather_invalid() {
        let result = PyAction::set_weather("tornado");
        assert!(result.is_err());
    }

    #[test]
    fn test_action_set_lighting() {
        let action = PyAction::set_lighting(18.0);
        let json = action.to_json().unwrap();
        assert!(json.contains("18"));
    }

    #[test]
    fn test_action_spawn_object() {
        let action = PyAction::spawn_object("cube");
        let json = action.to_json().unwrap();
        assert!(json.contains("cube"));
    }

    #[test]
    fn test_action_sequence() {
        let a1 = PyAction::move_to(1.0, 0.0, 0.0, 1.0);
        let a2 = PyAction::set_lighting(12.0);
        let seq = PyAction::sequence(vec![a1, a2]);
        let json = seq.to_json().unwrap();
        assert!(json.contains("Sequence"));
    }

    #[test]
    fn test_action_raw() {
        let action = PyAction::raw("cosmos", r#"{"prompt":"test"}"#).unwrap();
        let json = action.to_json().unwrap();
        assert!(json.contains("cosmos"));
    }

    #[test]
    fn test_action_raw_invalid_json() {
        let result = PyAction::raw("test", "not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_action_repr() {
        let action = PyAction::move_to(1.0, 0.0, 0.0, 1.0);
        let repr = action.__repr__();
        assert!(repr.contains("Action"));
    }

    // --- Guardrail tests ---

    #[test]
    fn test_guardrail_no_collisions() {
        let g = PyGuardrail::no_collisions();
        let json = g.to_json().unwrap();
        assert!(json.contains("NoCollisions"));
    }

    #[test]
    fn test_guardrail_max_velocity() {
        let g = PyGuardrail::max_velocity(10.0);
        let json = g.to_json().unwrap();
        assert!(json.contains("10"));
    }

    #[test]
    fn test_guardrail_energy_conservation() {
        let g = PyGuardrail::energy_conservation(0.05);
        let json = g.to_json().unwrap();
        assert!(json.contains("EnergyConservation"));
    }

    #[test]
    fn test_guardrail_human_safety_zone() {
        let g = PyGuardrail::human_safety_zone(2.0);
        let json = g.to_json().unwrap();
        assert!(json.contains("HumanSafetyZone"));
    }

    #[test]
    fn test_guardrail_boundary_constraint() {
        let min = PyPosition::new(-10.0, -10.0, -10.0);
        let max = PyPosition::new(10.0, 10.0, 10.0);
        let bbox = PyBBox::new(&min, &max);
        let g = PyGuardrail::boundary_constraint(&bbox);
        let json = g.to_json().unwrap();
        assert!(json.contains("BoundaryConstraint"));
    }

    // --- WorldForge orchestrator tests ---

    #[test]
    fn test_worldforge_create() {
        let wf = test_worldforge();
        let providers = wf.providers();
        assert!(providers.contains(&"mock".to_string()));
    }

    #[test]
    fn test_worldforge_create_world() {
        let wf = test_worldforge();
        let world = wf.create_world("test_world", "mock").unwrap();
        assert_eq!(world.name(), "test_world");
        assert_eq!(world.object_count(), 0);
    }

    #[test]
    fn test_worldforge_create_world_unknown_provider() {
        let wf = test_worldforge();
        let result = wf.create_world("test", "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_worldforge_repr() {
        let wf = test_worldforge();
        let repr = wf.__repr__();
        assert!(repr.contains("WorldForge"));
    }

    #[test]
    fn test_worldforge_generate() {
        let wf = test_worldforge();
        let clip = wf
            .generate(
                "a spinning cube",
                "mock",
                5.0,
                640,
                360,
                12.0,
                0.7,
                Some(7),
                Some("blurry"),
            )
            .unwrap();

        assert_eq!(clip.duration(), 5.0);
        assert_eq!(clip.resolution(), (640, 360));
        assert_eq!(clip.fps(), 12.0);
        assert_eq!(clip.frame_count(), 0);
        assert!(clip.__repr__().contains("VideoClip"));
    }

    #[test]
    fn test_videoclip_json_roundtrip() {
        let wf = test_worldforge();
        let clip = wf
            .generate(
                "a rolling sphere",
                "mock",
                2.0,
                320,
                180,
                10.0,
                1.0,
                None,
                None,
            )
            .unwrap();
        let json = clip.to_json().unwrap();
        let restored = PyVideoClip::from_json(&json).unwrap();

        assert_eq!(restored.duration(), 2.0);
        assert_eq!(restored.resolution(), (320, 180));
        assert_eq!(restored.fps(), 10.0);
    }

    #[test]
    fn test_worldforge_transfer() {
        let wf = test_worldforge();
        let clip = wf
            .generate(
                "a rolling sphere",
                "mock",
                3.0,
                320,
                180,
                10.0,
                1.0,
                None,
                None,
            )
            .unwrap();
        let transferred = wf
            .transfer(&clip, "mock", None, 640, 360, 24.0, 0.5)
            .unwrap();

        assert_eq!(transferred.duration(), clip.duration());
        assert_eq!(transferred.resolution(), clip.resolution());
        assert_eq!(transferred.fps(), clip.fps());
    }

    #[test]
    fn test_worldforge_file_store_roundtrip() {
        let state_dir =
            std::env::temp_dir().join(format!("wf-python-file-{}", uuid::Uuid::new_v4()));
        let wf = PyWorldForge::new("file", state_dir.to_str().unwrap(), None).unwrap();
        let world = wf.create_world("persisted_world", "mock").unwrap();
        let world_id = wf.save_world(&world).unwrap();

        let listed = wf.list_worlds().unwrap();
        assert!(listed.contains(&world_id));

        let loaded = wf.load_world(&world_id).unwrap();
        assert_eq!(loaded.name(), "persisted_world");

        wf.delete_world(&world_id).unwrap();
        assert!(wf.load_world(&world_id).is_err());

        let _ = std::fs::remove_dir_all(&state_dir);
    }

    #[test]
    fn test_worldforge_sqlite_store_roundtrip() {
        let state_dir =
            std::env::temp_dir().join(format!("wf-python-sqlite-{}", uuid::Uuid::new_v4()));
        let state_db_path = state_dir.join("worldforge.db");
        let wf = PyWorldForge::new(
            "sqlite",
            state_dir.to_str().unwrap(),
            Some(state_db_path.to_str().unwrap()),
        )
        .unwrap();
        let world = wf.create_world("sqlite_world", "mock").unwrap();
        let world_id = wf.save_world(&world).unwrap();

        let loaded = wf.load_world(&world_id).unwrap();
        assert_eq!(loaded.name(), "sqlite_world");
        assert_eq!(wf.list_worlds().unwrap(), vec![world_id.clone()]);

        wf.delete_world(&world_id).unwrap();
        assert!(wf.list_worlds().unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&state_dir);
    }

    // --- Planning tests ---

    #[test]
    fn test_plan_world() {
        let world = PyWorld::new("plan_test", "mock");
        let plan = world
            .plan(
                "move forward",
                5,
                10.0,
                Some("mock"),
                "sampling",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        assert!(plan.action_count() > 0);
        assert!(plan.success_probability() >= 0.0);
        assert!(plan.planning_time_ms() < 30_000);
    }

    #[test]
    fn test_plan_json_roundtrip() {
        let world = PyWorld::new("plan_json", "mock");
        let p = plan(
            &world,
            "reach goal",
            5,
            10.0,
            "mock",
            "sampling",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let json = p.to_json().unwrap();
        let p2 = PyPlan::from_json(&json).unwrap();
        assert_eq!(p2.action_count(), p.action_count());
    }

    #[test]
    fn test_plan_repr() {
        let world = PyWorld::new("repr_test", "mock");
        let p = plan(
            &world, "go", 5, 10.0, "mock", "sampling", None, None, None, None, None, None, None,
            None, None,
        )
        .unwrap();
        let repr = p.__repr__();
        assert!(repr.contains("Plan"));
    }

    #[test]
    fn test_plan_world_with_cem() {
        let world = PyWorld::new("plan_cem", "mock");
        let plan = world
            .plan(
                "spawn cube",
                4,
                10.0,
                Some("mock"),
                "cem",
                None,
                None,
                Some(16),
                Some(0.25),
                Some(3),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        assert!(plan.action_count() > 0);
        assert_eq!(plan.iterations_used(), 3);
    }

    #[test]
    fn test_plan_world_with_guardrails_json() {
        let world = PyWorld::new("plan_guardrails", "mock");
        let guardrails_json = r#"[{"guardrail":"NoCollisions","blocking":true}]"#;
        let plan = world
            .plan(
                "stay collision free",
                4,
                10.0,
                Some("mock"),
                "sampling",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(guardrails_json),
            )
            .unwrap();

        assert!(plan.action_count() > 0);
    }

    // --- Evaluation tests ---

    #[test]
    fn test_run_eval_physics() {
        let results = run_eval("physics").unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].provider(), "mock");
    }

    #[test]
    fn test_run_eval_invalid_suite() {
        let result = run_eval("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_result_repr() {
        let results = run_eval("physics").unwrap();
        let repr = results[0].__repr__();
        assert!(repr.contains("EvalResult"));
    }

    // --- ZK Verification tests ---

    #[test]
    fn test_prove_inference_and_verify() {
        let proof = prove_inference(b"model", b"input", b"output").unwrap();
        assert_eq!(proof.proof_size(), 96);
        assert_eq!(proof.backend(), "Mock");

        let (valid, details) = proof.verify().unwrap();
        assert!(valid);
        assert!(details.contains("verified"));
    }

    #[test]
    fn test_prove_provenance_and_verify() {
        let proof = prove_provenance(b"sensor-data", 1710000000, b"camera-01").unwrap();
        assert_eq!(proof.proof_size(), 72);

        let (valid, _) = proof.verify().unwrap();
        assert!(valid);
    }

    #[test]
    fn test_prove_inference_transition_and_verify() {
        let input = serde_json::to_string(&WorldState::new("input", "mock")).unwrap();
        let output = serde_json::to_string(&WorldState::new("output", "mock")).unwrap();

        let proof = prove_inference_transition(&input, &output, Some("mock")).unwrap();
        assert_eq!(proof.proof_size(), 96);

        let (valid, _) = proof.verify().unwrap();
        assert!(valid);
    }

    #[test]
    fn test_prove_guardrail_plan_and_verify() {
        let world = PyWorld::new("verify_plan", "mock");
        let plan = world
            .plan(
                "spawn cube",
                4,
                10.0,
                Some("mock"),
                "sampling",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(r#"[{"guardrail":"NoCollisions","blocking":true}]"#),
            )
            .unwrap();

        let proof = prove_guardrail_plan(&plan).unwrap();
        let (valid, _) = proof.verify().unwrap();
        assert!(valid);
    }

    #[test]
    fn test_zkproof_json_roundtrip() {
        let proof = prove_inference(b"m", b"i", b"o").unwrap();
        let json = proof.to_json().unwrap();
        assert!(json.contains("Mock"));

        let restored = PyZkProof::from_json(&json).unwrap();
        let (valid, _) = restored.verify().unwrap();
        assert!(valid);
    }

    #[test]
    fn test_zkproof_repr() {
        let proof = prove_inference(b"m", b"i", b"o").unwrap();
        let repr = proof.__repr__();
        assert!(repr.contains("ZkProof"));
    }

    #[test]
    fn test_verify_proof_json() {
        let proof = prove_inference(b"model", b"input", b"output").unwrap();
        let json = proof.to_json().unwrap();

        let (valid, details) = verify_proof_json(&json).unwrap();

        assert!(valid);
        assert!(details.contains("verified"));
    }

    #[test]
    fn test_verify_bundle_json() {
        let world = PyWorld::new("verify_bundle", "mock");
        let plan = world
            .plan(
                "spawn cube",
                4,
                10.0,
                Some("mock"),
                "sampling",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(r#"[{"guardrail":"NoCollisions","blocking":true}]"#),
            )
            .unwrap();
        let verifier = worldforge_verify::MockVerifier::new();
        let bundle = worldforge_verify::prove_guardrail_plan(&verifier, &plan.inner).unwrap();
        let json = serde_json::to_string(&bundle).unwrap();

        let report = verify_bundle_json(&json, "guardrail").unwrap();

        assert!(report.contains("\"verification_matches_recorded\": true"));
    }
}
