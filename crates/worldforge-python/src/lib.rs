// PyO3-generated code triggers this clippy lint; it's a known false positive.
#![allow(clippy::useless_conversion)]
//! Python bindings for WorldForge.
//!
//! Exposes core types, scene management, and the main WorldForge
//! orchestrator to Python via PyO3.

use std::path::Path;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyModule, PyString};

use worldforge_core::guardrail::GuardrailConfig;
use worldforge_core::prediction::{
    ComparisonConsensus, GuardrailDiagnostics, MultiPrediction, ObjectDiagnostic, ObjectDrift,
    PairwiseAgreement, PlanExecution, PlanGoal, PlanGoalInput, PlannerType, PredictionConfig,
    ProviderScore, StateDiagnostics,
};
use worldforge_core::provider::{
    CostEstimate, EmbeddingInput, EmbeddingOutput, GenerationConfig, GenerationPrompt, Operation,
    ProviderCapabilities, ProviderDescriptor, ProviderHealthReport, ProviderRegistry,
    ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig, WorldModelProvider,
};
use worldforge_core::scene::{SceneObject, SceneObjectPatch};
use worldforge_core::state::{
    deserialize_world_state, serialize_world_state, HistoryEntry,
    StateFileFormat as CoreStateFileFormat, StateStoreKind, WorldState,
};
use worldforge_core::types::{BBox, Position, Rotation, TensorData, Velocity, VideoClip};
use worldforge_core::world::World as CoreWorld;
use worldforge_providers::{
    CosmosProvider, GenieProvider, JepaBackend, JepaProvider, MockProvider, RunwayProvider,
};
use worldforge_verify::{
    prove_guardrail_plan as prove_guardrail_plan_bundle,
    prove_inference_transition as prove_inference_transition_bundle,
    prove_latest_inference as prove_latest_inference_bundle,
    prove_provenance as prove_provenance_state_bundle, verify_bundle, verify_proof,
    BundleVerificationReport, GuardrailArtifact, InferenceArtifact, ProvenanceArtifact,
    VerificationBackend, VerificationBundle, VerificationResult as CoreVerificationResult,
    ZkVerifier,
};

fn auto_detect_registry() -> Arc<worldforge_core::provider::ProviderRegistry> {
    Arc::new(worldforge_providers::auto_detect())
}

fn resolve_provider_name(world: &CoreWorld, provider: Option<&str>) -> String {
    provider
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| world.default_provider.clone())
}

fn new_runtime() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Runtime::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
    })
}

fn parse_uuid(value: &str, label: &str) -> PyResult<uuid::Uuid> {
    value.parse().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid {label} ID (must be UUID)"))
    })
}

fn parse_world_id(world_id: &str) -> PyResult<uuid::Uuid> {
    parse_uuid(world_id, "world")
}

fn parse_object_id(object_id: &str) -> PyResult<uuid::Uuid> {
    parse_uuid(object_id, "object")
}

fn state_store_kind(
    state_backend: &str,
    state_dir: &str,
    state_db_path: Option<&str>,
    state_file_format: &str,
    state_redis_url: Option<&str>,
) -> PyResult<StateStoreKind> {
    match state_backend {
        "file" => {
            let format = state_file_format
                .parse::<CoreStateFileFormat>()
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok(StateStoreKind::FileWithFormat {
                path: state_dir.into(),
                format,
            })
        }
        "sqlite" => Ok(StateStoreKind::Sqlite(
            state_db_path
                .map(Into::into)
                .unwrap_or_else(|| Path::new(state_dir).join("worldforge.db")),
        )),
        "redis" => state_redis_url
            .map(|url| StateStoreKind::Redis(url.to_string()))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "state_redis_url is required when state_backend='redis'",
                )
            }),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown state backend: {other}. Available: file, sqlite, redis"
        ))),
    }
}

fn parse_snapshot_format(format: &str) -> PyResult<CoreStateFileFormat> {
    format
        .parse::<CoreStateFileFormat>()
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

fn snapshot_bytes(snapshot: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = snapshot.extract::<&[u8]>() {
        return Ok(bytes.to_vec());
    }

    if let Ok(text) = snapshot.extract::<&str>() {
        return Ok(text.as_bytes().to_vec());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "snapshot must be `str` for JSON or `bytes` for MessagePack",
    ))
}

fn world_from_state(state: WorldState, registry: Arc<ProviderRegistry>) -> PyWorld {
    let provider = state.current_state_provider();
    PyWorld {
        world: CoreWorld::new(state, provider, Arc::clone(&registry)),
        registry,
    }
}

fn normalize_imported_state(
    state: &mut WorldState,
    new_id: bool,
    name: Option<&str>,
) -> PyResult<()> {
    let mut rebase = new_id;

    if let Some(name) = name.map(str::trim).filter(|value| !value.is_empty()) {
        state.metadata.name = name.to_string();
        rebase = true;
    }

    if rebase {
        state.id = uuid::Uuid::new_v4();
        state.history.states.clear();
        let provider = state.current_state_provider();
        state.ensure_history_initialized(provider).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to normalize imported world state: {e}"
            ))
        })?;
    }

    Ok(())
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

fn parse_plan_goal(goal: Option<&str>, goal_json: Option<&str>) -> PyResult<PlanGoal> {
    match (goal, goal_json) {
        (Some(description), None) => Ok(PlanGoal::Description(description.to_string())),
        (None, Some(json)) => serde_json::from_str::<PlanGoalInput>(json)
            .map(Into::into)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "failed to parse plan goal JSON: {e}"
                ))
            }),
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(
            "goal and goal_json are mutually exclusive",
        )),
        (None, None) => Err(pyo3::exceptions::PyValueError::new_err(
            "either goal or goal_json is required",
        )),
    }
}

fn parse_provider_names(input: &str) -> Vec<String> {
    let mut provider_names: Vec<String> = input
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if provider_names.is_empty() {
        provider_names.push("mock".to_string());
    }
    provider_names
}

fn parse_provider_names_allow_empty(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn resolve_eval_provider_names(suite: &worldforge_eval::EvalSuite, providers: &str) -> Vec<String> {
    let explicit = if providers.trim().is_empty() {
        parse_provider_names_allow_empty(providers)
    } else {
        parse_provider_names(providers)
    };
    suite.effective_provider_names(&explicit)
}

fn format_hash_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn format_hash_list_hex(hashes: &[[u8; 32]]) -> Vec<String> {
    hashes.iter().map(format_hash_hex).collect()
}

fn tensor_data_to_f32_vec(data: &TensorData) -> Vec<f32> {
    match data {
        TensorData::Float32(values) => values.clone(),
        TensorData::Float64(values) => values.iter().map(|value| *value as f32).collect(),
        TensorData::Float16(values) => values
            .iter()
            .map(|value| half_bits_to_f32(*value))
            .collect(),
        TensorData::BFloat16(values) => values
            .iter()
            .map(|value| bfloat16_bits_to_f32(*value))
            .collect(),
        TensorData::UInt8(values) => values.iter().map(|value| *value as f32).collect(),
        TensorData::Int32(values) => values.iter().map(|value| *value as f32).collect(),
        TensorData::Int64(values) => values.iter().map(|value| *value as f32).collect(),
    }
}

fn half_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits & 0x7c00) >> 10) as u32;
    let mantissa = (bits & 0x03ff) as u32;

    let f32_bits = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                let mut mantissa = mantissa;
                let mut exponent = -14i32;
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | (((exponent as i32 - 15 + 127) as u32) << 23) | (mantissa << 13),
    };

    f32::from_bits(f32_bits)
}

fn bfloat16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn load_eval_suite(
    suite_name: &str,
    suite_json: Option<&str>,
) -> PyResult<worldforge_eval::EvalSuite> {
    match suite_json {
        Some(json) => worldforge_eval::EvalSuite::from_json_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to load custom evaluation suite: {e}"
            ))
        }),
        None => worldforge_eval::EvalSuite::from_builtin(suite_name).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unknown eval suite: {suite_name}. Available: {}. Original error: {e}",
                worldforge_eval::EvalSuite::builtin_names().join(", ")
            ))
        }),
    }
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

fn operation_from_args(
    operation: &str,
    steps: u32,
    duration_seconds: f64,
    width: u32,
    height: u32,
) -> PyResult<Operation> {
    match operation {
        "predict" => Ok(Operation::Predict {
            steps: steps.max(1),
            resolution: (width, height),
        }),
        "generate" => Ok(Operation::Generate {
            duration_seconds: duration_seconds.max(0.1),
            resolution: (width, height),
        }),
        "reason" => Ok(Operation::Reason),
        "transfer" => Ok(Operation::Transfer {
            duration_seconds: duration_seconds.max(0.1),
        }),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown operation: {other}. Available: predict, generate, reason, transfer"
        ))),
    }
}

fn parse_cosmos_model(model: &str) -> PyResult<worldforge_providers::cosmos::CosmosModel> {
    match model.to_ascii_lowercase().as_str() {
        "predict" | "predict-2.5" | "predict2_5" => {
            Ok(worldforge_providers::cosmos::CosmosModel::Predict2_5)
        }
        "transfer" | "transfer-2.5" | "transfer2_5" => {
            Ok(worldforge_providers::cosmos::CosmosModel::Transfer2_5)
        }
        "reason" | "reason-2" | "reason2" => Ok(worldforge_providers::cosmos::CosmosModel::Reason2),
        "embed" | "embed-1" | "embed1" => Ok(worldforge_providers::cosmos::CosmosModel::Embed1),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown Cosmos model: {other}. Available: predict-2.5, transfer-2.5, reason-2, embed-1"
        ))),
    }
}

fn parse_runway_model(model: &str) -> PyResult<worldforge_providers::runway::RunwayModel> {
    match model.to_ascii_lowercase().as_str() {
        "worlds" => Ok(worldforge_providers::runway::RunwayModel::Gwm1Worlds),
        "robotics" => Ok(worldforge_providers::runway::RunwayModel::Gwm1Robotics),
        "avatars" => Ok(worldforge_providers::runway::RunwayModel::Gwm1Avatars),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown Runway model: {other}. Available: worlds, robotics, avatars"
        ))),
    }
}

fn parse_jepa_backend(backend: &str) -> PyResult<JepaBackend> {
    match backend.to_ascii_lowercase().as_str() {
        "burn" => Ok(JepaBackend::Burn),
        "pytorch" => Ok(JepaBackend::PyTorch),
        "onnx" => Ok(JepaBackend::Onnx),
        "safetensors" => Ok(JepaBackend::Safetensors),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown JEPA backend: {other}. Available: burn, pytorch, onnx, safetensors"
        ))),
    }
}

fn parse_genie_model(model: &str) -> PyResult<worldforge_providers::genie::GenieModel> {
    match model.to_ascii_lowercase().as_str() {
        "genie3" | "genie-3" => Ok(worldforge_providers::genie::GenieModel::Genie3),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown Genie model: {other}. Available: genie3"
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

    /// Set the object's name.
    #[setter]
    fn set_name(&mut self, name: String) {
        self.inner.name = name;
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
        self.inner.set_position(pos.inner);
    }

    /// Get the object's rotation.
    #[getter]
    fn rotation(&self) -> PyRotation {
        PyRotation {
            inner: self.inner.pose.rotation,
        }
    }

    /// Set the object's rotation.
    #[setter]
    fn set_rotation(&mut self, rotation: &PyRotation) {
        self.inner.pose.rotation = rotation.inner;
    }

    /// Get the object's bounding box.
    #[getter]
    fn bbox(&self) -> PyBBox {
        PyBBox {
            inner: self.inner.bbox,
        }
    }

    /// Set the object's bounding box.
    #[setter]
    fn set_bbox(&mut self, bbox: &PyBBox) {
        self.inner.bbox = bbox.inner;
        self.inner.pose.position = self.inner.bbox.center();
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

    /// Whether the object is static (immovable).
    #[getter]
    fn is_static(&self) -> bool {
        self.inner.physics.is_static
    }

    /// Set the object as static (immovable).
    fn set_static(&mut self, is_static: bool) {
        self.inner.physics.is_static = is_static;
    }

    /// Get the object's mass in kilograms.
    #[getter]
    fn mass(&self) -> Option<f32> {
        self.inner.physics.mass
    }

    /// Set the object's mass in kilograms.
    fn set_mass(&mut self, mass: f32) {
        self.inner.physics.mass = Some(mass);
    }

    /// Get the object's friction coefficient.
    #[getter]
    fn friction(&self) -> Option<f32> {
        self.inner.physics.friction
    }

    /// Set the object's friction coefficient.
    #[pyo3(signature = (friction=None))]
    fn set_friction(&mut self, friction: Option<f32>) {
        self.inner.physics.friction = friction;
    }

    /// Get the object's restitution coefficient.
    #[getter]
    fn restitution(&self) -> Option<f32> {
        self.inner.physics.restitution
    }

    /// Set the object's restitution coefficient.
    #[pyo3(signature = (restitution=None))]
    fn set_restitution(&mut self, restitution: Option<f32>) {
        self.inner.physics.restitution = restitution;
    }

    /// Whether the object can be grasped.
    #[getter]
    fn is_graspable(&self) -> bool {
        self.inner.physics.is_graspable
    }

    /// Set whether the object can be grasped.
    fn set_graspable(&mut self, is_graspable: bool) {
        self.inner.physics.is_graspable = is_graspable;
    }

    /// Get the object's material.
    #[getter]
    fn material(&self) -> Option<&str> {
        self.inner.physics.material.as_deref()
    }

    /// Set the object's material.
    #[pyo3(signature = (material=None))]
    fn set_material(&mut self, material: Option<String>) {
        self.inner.physics.material = material;
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
        let inner: SceneObject = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { inner })
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

/// A partial update for an existing scene object.
#[pyclass(name = "SceneObjectPatch")]
#[derive(Debug, Clone, Default)]
pub struct PySceneObjectPatch {
    inner: SceneObjectPatch,
}

#[pymethods]
impl PySceneObjectPatch {
    /// Create an empty object patch.
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Set or clear the replacement name.
    #[pyo3(signature = (name=None))]
    fn set_name(&mut self, name: Option<String>) {
        self.inner.name = name;
    }

    /// Set or clear the replacement position.
    #[pyo3(signature = (position=None))]
    fn set_position(&mut self, position: Option<&PyPosition>) {
        self.inner.position = position.map(|position| position.inner);
    }

    /// Set or clear the replacement rotation.
    #[pyo3(signature = (rotation=None))]
    fn set_rotation(&mut self, rotation: Option<&PyRotation>) {
        self.inner.rotation = rotation.map(|rotation| rotation.inner);
    }

    /// Set or clear the replacement bounding box.
    #[pyo3(signature = (bbox=None))]
    fn set_bbox(&mut self, bbox: Option<&PyBBox>) {
        self.inner.bbox = bbox.map(|bbox| bbox.inner);
    }

    /// Set or clear the replacement velocity.
    #[pyo3(signature = (velocity=None))]
    fn set_velocity(&mut self, velocity: Option<&PyVelocity>) {
        self.inner.velocity = velocity.map(|velocity| velocity.inner);
    }

    /// Set or clear the replacement semantic label.
    #[pyo3(signature = (semantic_label=None))]
    fn set_semantic_label(&mut self, semantic_label: Option<String>) {
        self.inner.semantic_label = semantic_label;
    }

    /// Set or clear the replacement mass.
    #[pyo3(signature = (mass=None))]
    fn set_mass(&mut self, mass: Option<f32>) {
        self.inner.mass = mass;
    }

    /// Set or clear the replacement friction coefficient.
    #[pyo3(signature = (friction=None))]
    fn set_friction(&mut self, friction: Option<f32>) {
        self.inner.friction = friction;
    }

    /// Set or clear the replacement restitution coefficient.
    #[pyo3(signature = (restitution=None))]
    fn set_restitution(&mut self, restitution: Option<f32>) {
        self.inner.restitution = restitution;
    }

    /// Set or clear the replacement material name.
    #[pyo3(signature = (material=None))]
    fn set_material(&mut self, material: Option<String>) {
        self.inner.material = material;
    }

    /// Set or clear the replacement immovable flag.
    #[pyo3(signature = (is_static=None))]
    fn set_static(&mut self, is_static: Option<bool>) {
        self.inner.is_static = is_static;
    }

    /// Set or clear the replacement graspable flag.
    #[pyo3(signature = (is_graspable=None))]
    fn set_graspable(&mut self, is_graspable: Option<bool>) {
        self.inner.is_graspable = is_graspable;
    }

    /// Convert the patch to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Create a patch from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: SceneObjectPatch = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        let mut fields = Vec::new();
        if self.inner.name.is_some() {
            fields.push("name");
        }
        if self.inner.position.is_some() {
            fields.push("position");
        }
        if self.inner.rotation.is_some() {
            fields.push("rotation");
        }
        if self.inner.bbox.is_some() {
            fields.push("bbox");
        }
        if self.inner.velocity.is_some() {
            fields.push("velocity");
        }
        if self.inner.semantic_label.is_some() {
            fields.push("semantic_label");
        }
        if self.inner.mass.is_some() {
            fields.push("mass");
        }
        if self.inner.friction.is_some() {
            fields.push("friction");
        }
        if self.inner.restitution.is_some() {
            fields.push("restitution");
        }
        if self.inner.material.is_some() {
            fields.push("material");
        }
        if self.inner.is_static.is_some() {
            fields.push("is_static");
        }
        if self.inner.is_graspable.is_some() {
            fields.push("is_graspable");
        }

        format!("SceneObjectPatch(fields=[{}])", fields.join(", "))
    }
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

/// A WorldForge world instance.
///
/// Wraps the Rust WorldState with Python-friendly methods.
#[pyclass(name = "World")]
pub struct PyWorld {
    world: CoreWorld,
    registry: Arc<ProviderRegistry>,
}

#[pymethods]
impl PyWorld {
    /// Create a new empty world.
    #[new]
    #[pyo3(signature = (name, provider="mock"))]
    fn new(name: &str, provider: &str) -> Self {
        let registry = auto_detect_registry();
        Self {
            world: CoreWorld::new(
                WorldState::new(name, provider),
                provider,
                Arc::clone(&registry),
            ),
            registry,
        }
    }

    /// Get the world's unique ID.
    #[getter]
    fn id(&self) -> String {
        self.world.state.id.to_string()
    }

    /// Get the world's name.
    #[getter]
    fn name(&self) -> &str {
        &self.world.state.metadata.name
    }

    /// Get the world's prompt-derived description.
    #[getter]
    fn description(&self) -> &str {
        &self.world.state.metadata.description
    }

    /// Get the current simulation step.
    #[getter]
    fn step(&self) -> u64 {
        self.world.state.time.step
    }

    /// Get the current simulation time in seconds.
    #[getter]
    fn time_seconds(&self) -> f64 {
        self.world.state.time.seconds
    }

    /// Get the number of objects in the scene.
    #[getter]
    fn object_count(&self) -> usize {
        self.world.state.scene.objects.len()
    }

    /// Add an object to the world.
    fn add_object(&mut self, obj: &PySceneObject) -> PyResult<()> {
        self.world
            .add_object(obj.inner.clone())
            .map_err(|e| match e {
                worldforge_core::error::WorldForgeError::InvalidState(message) => {
                    pyo3::exceptions::PyKeyError::new_err(message)
                }
                _ => {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("failed to add object: {e}"))
                }
            })
    }

    /// Update an existing object in the world using a full object replacement.
    fn update_object(&mut self, obj: &PySceneObject) -> PyResult<()> {
        let object_id = obj.inner.id;
        if self.world.get_object(&object_id).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "object not found: {object_id}"
            )));
        }

        self.world
            .replace_object(obj.inner.clone())
            .map_err(|e| match e {
                worldforge_core::error::WorldForgeError::InvalidState(message) => {
                    pyo3::exceptions::PyKeyError::new_err(message)
                }
                _ => pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to replace object: {e}"
                )),
            })?;
        Ok(())
    }

    /// Get an object by name.
    fn get_object(&self, name: &str) -> Option<PySceneObject> {
        self.world
            .state
            .scene
            .find_object_by_name(name)
            .map(|o| PySceneObject { inner: o.clone() })
    }

    /// Get an object by its stable ID.
    fn get_object_by_id(&self, object_id: &str) -> PyResult<Option<PySceneObject>> {
        let object_id = parse_object_id(object_id)?;
        Ok(self
            .world
            .get_object(&object_id)
            .cloned()
            .map(|inner| PySceneObject { inner }))
    }

    /// Remove an object by name. Returns True if found.
    fn remove_object(&mut self, name: &str) -> bool {
        if let Some(object_id) = self.world.get_object_by_name(name).map(|object| object.id) {
            self.world.remove_object(&object_id).is_ok()
        } else {
            false
        }
    }

    /// Remove an object by its stable ID.
    fn remove_object_by_id(&mut self, object_id: &str) -> PyResult<Option<PySceneObject>> {
        let object_id = parse_object_id(object_id)?;
        if self.world.get_object(&object_id).is_none() {
            return Ok(None);
        }

        self.world
            .remove_object(&object_id)
            .map(|inner| Some(PySceneObject { inner }))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("failed to remove object: {e}"))
            })
    }

    /// List all object names in the scene.
    fn list_objects(&self) -> Vec<String> {
        self.world
            .state
            .scene
            .list_objects()
            .into_iter()
            .map(|o| o.name.clone())
            .collect()
    }

    /// List all scene objects with their IDs and metadata.
    fn objects(&self) -> Vec<PySceneObject> {
        self.world
            .list_objects()
            .into_iter()
            .map(|inner| PySceneObject { inner })
            .collect()
    }

    /// Apply a partial update to an object using its stable ID.
    fn update_object_patch(
        &mut self,
        object_id: &str,
        patch: &PySceneObjectPatch,
    ) -> PyResult<PySceneObject> {
        let object_id = parse_object_id(object_id)?;
        self.world
            .update_object(&object_id, patch.inner.clone())
            .map(|inner| PySceneObject { inner })
            .map_err(|e| match e {
                worldforge_core::error::WorldForgeError::InvalidState(message) => {
                    pyo3::exceptions::PyKeyError::new_err(message)
                }
                _ => pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to patch object: {e}"
                )),
            })
    }

    /// Get the number of history entries.
    #[getter]
    fn history_length(&self) -> usize {
        self.world.state.history.len()
    }

    /// Return the recorded history entries for this world.
    fn history(&self) -> Vec<PyHistoryEntry> {
        self.world
            .state
            .history
            .states
            .iter()
            .cloned()
            .map(|inner| PyHistoryEntry { inner })
            .collect()
    }

    /// Return a detached world view for a specific recorded history checkpoint.
    fn history_state(&self, index: usize) -> PyResult<Self> {
        let state = self.world.history_state(index).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to reconstruct history checkpoint: {e}"
            ))
        })?;
        Ok(world_from_state(state, Arc::clone(&self.registry)))
    }

    /// Restore this world in place to a specific history checkpoint.
    fn restore_history(&mut self, index: usize) -> PyResult<()> {
        self.world.restore_history(index).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to restore history checkpoint: {e}"
            ))
        })
    }

    /// Export the world state as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.world.state).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Import a world state from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let state: WorldState = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        let provider = state.current_state_provider();
        let registry = auto_detect_registry();
        Ok(Self {
            world: CoreWorld::new(state, provider, Arc::clone(&registry)),
            registry,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "World(name='{}', objects={}, step={})",
            self.world.state.metadata.name,
            self.world.state.scene.objects.len(),
            self.world.state.time.step
        )
    }

    /// Generate a typed verification bundle for the latest recorded inference.
    ///
    /// Requires at least two recorded history entries, which typically means
    /// the world has already gone through one prediction step.
    #[pyo3(signature = (provider=None))]
    fn prove_latest_inference_bundle(&self, provider: Option<&str>) -> PyResult<PyInferenceBundle> {
        let verifier = worldforge_verify::MockVerifier::new();
        let bundle = prove_latest_inference_bundle(&verifier, &self.world.state, provider)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
            })?;
        Ok(PyInferenceBundle { inner: bundle })
    }

    /// Generate a typed provenance-attestation bundle for the current world state.
    #[pyo3(signature = (source_label="worldforge-python", timestamp=None))]
    fn prove_provenance_bundle(
        &self,
        source_label: &str,
        timestamp: Option<u64>,
    ) -> PyResult<PyProvenanceBundle> {
        let verifier = worldforge_verify::MockVerifier::new();
        let bundle = prove_provenance_state_bundle(
            &verifier,
            &self.world.state,
            source_label,
            timestamp.unwrap_or_else(current_timestamp),
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
        })?;
        Ok(PyProvenanceBundle { inner: bundle })
    }

    /// Predict the next world state after applying an action.
    #[pyo3(signature = (action, steps=1, provider=None, fallback_provider=None, return_video=false, max_latency_ms=None, disable_guardrails=false))]
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &mut self,
        action: &PyAction,
        steps: u32,
        provider: Option<&str>,
        fallback_provider: Option<&str>,
        return_video: bool,
        max_latency_ms: Option<u64>,
        disable_guardrails: bool,
    ) -> PyResult<PyPrediction> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
        })?;

        let provider_name = resolve_provider_name(&self.world, provider);
        self.world.default_provider = provider_name.clone();
        let mut config = PredictionConfig {
            steps,
            return_video,
            max_latency_ms,
            fallback_provider: fallback_provider.map(ToOwned::to_owned),
            ..PredictionConfig::default()
        };
        if disable_guardrails {
            config = config.disable_guardrails();
        }

        let prediction = rt
            .block_on(self.world.predict(&action.inner, &config))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("prediction failed: {e}"))
            })?;
        self.world.state = self.world.current_state().clone();

        Ok(PyPrediction { inner: prediction })
    }

    /// Compare predictions from multiple providers without mutating the world state.
    #[pyo3(signature = (action, providers, steps=1, fallback_provider=None, return_video=false, max_latency_ms=None, guardrails_json=None, disable_guardrails=false))]
    #[allow(clippy::too_many_arguments)]
    fn compare(
        &self,
        action: &PyAction,
        providers: Vec<String>,
        steps: u32,
        fallback_provider: Option<&str>,
        return_video: bool,
        max_latency_ms: Option<u64>,
        guardrails_json: Option<&str>,
        disable_guardrails: bool,
    ) -> PyResult<PyMultiPrediction> {
        if providers.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "providers cannot be empty",
            ));
        }

        let rt = new_runtime()?;
        let mut config = PredictionConfig {
            steps,
            return_video,
            max_latency_ms,
            fallback_provider: fallback_provider.map(ToOwned::to_owned),
            guardrails: parse_guardrails_json(guardrails_json)?,
            ..PredictionConfig::default()
        };
        if disable_guardrails {
            config = config.disable_guardrails();
        }
        let provider_refs: Vec<&str> = providers.iter().map(String::as_str).collect();
        let comparison = rt
            .block_on(
                self.world
                    .predict_multi(&action.inner, &provider_refs, &config),
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("comparison failed: {e}"))
            })?;

        Ok(PyMultiPrediction { inner: comparison })
    }

    /// Plan a sequence of actions to achieve either a natural-language or structured goal.
    #[pyo3(signature = (goal=None, goal_json=None, max_steps=10, timeout_seconds=30.0, provider=None, planner="sampling", num_samples=None, top_k=None, population_size=None, elite_fraction=None, num_iterations=None, learning_rate=None, horizon=None, replanning_interval=None, guardrails_json=None, disable_guardrails=false))]
    #[allow(clippy::too_many_arguments)]
    fn plan(
        &self,
        goal: Option<&str>,
        goal_json: Option<&str>,
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
        disable_guardrails: bool,
    ) -> PyResult<PyPlan> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create runtime: {e}"))
        })?;

        let provider_name = resolve_provider_name(&self.world, provider);
        let world = CoreWorld::new(
            self.world.state.clone(),
            provider_name.clone(),
            Arc::clone(&self.registry),
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
        let goal = parse_plan_goal(goal, goal_json)?;
        let mut request = worldforge_core::prediction::PlanRequest {
            current_state: self.world.state.clone(),
            goal,
            max_steps,
            guardrails: parse_guardrails_json(guardrails_json)?,
            planner,
            timeout_seconds,
        };
        if disable_guardrails {
            request = request.disable_guardrails();
        }

        let plan = rt.block_on(world.plan(&request)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("planning failed: {e}"))
        })?;
        Ok(PyPlan { inner: plan })
    }

    /// Execute a previously generated plan against the live world state.
    #[pyo3(signature = (plan, steps=1, provider=None, fallback_provider=None, return_video=false, max_latency_ms=None, guardrails_json=None, disable_guardrails=false))]
    #[allow(clippy::too_many_arguments)]
    fn execute_plan(
        &mut self,
        plan: &PyPlan,
        steps: u32,
        provider: Option<&str>,
        fallback_provider: Option<&str>,
        return_video: bool,
        max_latency_ms: Option<u64>,
        guardrails_json: Option<&str>,
        disable_guardrails: bool,
    ) -> PyResult<PyPlanExecution> {
        let rt = new_runtime()?;
        let provider_name = resolve_provider_name(&self.world, provider);
        let mut world = CoreWorld::new(
            self.world.state.clone(),
            provider_name.clone(),
            Arc::clone(&self.registry),
        );
        let mut config = PredictionConfig {
            steps,
            return_video,
            max_latency_ms,
            fallback_provider: fallback_provider.map(ToOwned::to_owned),
            guardrails: parse_guardrails_json(guardrails_json)?,
            ..PredictionConfig::default()
        };
        if disable_guardrails {
            config = config.disable_guardrails();
        }

        let execution = rt
            .block_on(world.execute_plan_with_provider(&plan.inner, &config, &provider_name))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("plan execution failed: {e}"))
            })?;
        self.world.state = world.current_state().clone();

        Ok(PyPlanExecution { inner: execution })
    }

    /// Ask a provider to reason about the current world state.
    #[pyo3(signature = (query, provider=None, fallback_provider=None))]
    fn reason(
        &self,
        query: &str,
        provider: Option<&str>,
        fallback_provider: Option<&str>,
    ) -> PyResult<PyReasoningOutput> {
        let rt = new_runtime()?;

        let provider_name = resolve_provider_name(&self.world, provider);
        let output = rt
            .block_on(self.world.reason_with_provider_and_fallback(
                query,
                &provider_name,
                fallback_provider,
            ))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("reasoning failed: {e}"))
            })?
            .1;

        Ok(PyReasoningOutput { inner: output })
    }
}

/// One recorded checkpoint from a world's state history.
#[pyclass(name = "HistoryEntry")]
#[derive(Debug, Clone)]
pub struct PyHistoryEntry {
    inner: HistoryEntry,
}

#[pymethods]
impl PyHistoryEntry {
    /// Simulation step captured by this entry.
    #[getter]
    fn step(&self) -> u64 {
        self.inner.time.step
    }

    /// Continuous simulation time in seconds.
    #[getter]
    fn time_seconds(&self) -> f64 {
        self.inner.time.seconds
    }

    /// Provider recorded for this checkpoint.
    #[getter]
    fn provider(&self) -> String {
        self.inner.provider.clone()
    }

    /// SHA-256 state hash for this checkpoint, encoded as lowercase hex.
    #[getter]
    fn state_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.state_hash)
    }

    /// Serialized action JSON, if this checkpoint was produced by an action.
    #[getter]
    fn action_json(&self) -> PyResult<Option<String>> {
        self.inner
            .action
            .as_ref()
            .map(|action| {
                serde_json::to_string(action).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
                })
            })
            .transpose()
    }

    /// Prediction confidence, if present.
    #[getter]
    fn confidence(&self) -> Option<f32> {
        self.inner
            .prediction
            .as_ref()
            .map(|prediction| prediction.confidence)
    }

    /// Overall physics score, if present.
    #[getter]
    fn physics_score(&self) -> Option<f32> {
        self.inner
            .prediction
            .as_ref()
            .map(|prediction| prediction.physics_score)
    }

    /// Prediction latency in milliseconds, if present.
    #[getter]
    fn latency_ms(&self) -> Option<u64> {
        self.inner
            .prediction
            .as_ref()
            .map(|prediction| prediction.latency_ms)
    }

    /// Serialize this history entry to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "HistoryEntry(step={}, provider='{}', action_present={})",
            self.inner.time.step,
            self.inner.provider,
            self.inner.action.is_some()
        )
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
        let state = self.inner.output_state.clone();
        let provider = state.current_state_provider();
        let registry = auto_detect_registry();
        PyWorld {
            world: CoreWorld::new(state, provider, Arc::clone(&registry)),
            registry,
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

/// Stable object identity surfaced in compare diagnostics.
#[pyclass(name = "ObjectDiagnostic")]
#[derive(Debug, Clone)]
pub struct PyObjectDiagnostic {
    inner: ObjectDiagnostic,
}

#[pymethods]
impl PyObjectDiagnostic {
    /// Stable object identifier.
    #[getter]
    fn object_id(&self) -> String {
        self.inner.object_id.to_string()
    }

    /// Human-readable object name.
    #[getter]
    fn object_name(&self) -> &str {
        &self.inner.object_name
    }

    fn __repr__(&self) -> String {
        format!(
            "ObjectDiagnostic(object_id='{}', object_name='{}')",
            self.inner.object_id, self.inner.object_name
        )
    }
}

/// Per-object positional drift surfaced in compare diagnostics.
#[pyclass(name = "ObjectDrift")]
#[derive(Debug, Clone)]
pub struct PyObjectDrift {
    inner: ObjectDrift,
}

#[pymethods]
impl PyObjectDrift {
    /// Stable object identifier.
    #[getter]
    fn object_id(&self) -> String {
        self.inner.object_id.to_string()
    }

    /// Human-readable object name.
    #[getter]
    fn object_name(&self) -> &str {
        &self.inner.object_name
    }

    /// Euclidean drift distance in world units.
    #[getter]
    fn distance(&self) -> f32 {
        self.inner.distance
    }

    fn __repr__(&self) -> String {
        format!(
            "ObjectDrift(object_id='{}', object_name='{}', distance={:.3})",
            self.inner.object_id, self.inner.object_name, self.inner.distance
        )
    }
}

/// Guardrail diagnostics within a multi-provider comparison.
#[pyclass(name = "GuardrailDiagnostics")]
#[derive(Debug, Clone)]
pub struct PyGuardrailDiagnostics {
    inner: GuardrailDiagnostics,
}

#[pymethods]
impl PyGuardrailDiagnostics {
    /// Number of guardrails evaluated.
    #[getter]
    fn evaluated_count(&self) -> usize {
        self.inner.evaluated_count
    }

    /// Number of guardrails that passed.
    #[getter]
    fn passed_count(&self) -> usize {
        self.inner.passed_count
    }

    /// Number of guardrails that failed.
    #[getter]
    fn failed_count(&self) -> usize {
        self.inner.failed_count
    }

    /// Number of blocking guardrail failures.
    #[getter]
    fn blocking_failures(&self) -> usize {
        self.inner.blocking_failures
    }

    /// Fraction of evaluated guardrails that passed.
    #[getter]
    fn pass_rate(&self) -> f32 {
        self.inner.pass_rate
    }

    fn __repr__(&self) -> String {
        format!(
            "GuardrailDiagnostics(passed={}/{}, blocking_failures={})",
            self.inner.passed_count, self.inner.evaluated_count, self.inner.blocking_failures
        )
    }
}

/// State-level diagnostics within a multi-provider comparison.
#[pyclass(name = "StateDiagnostics")]
#[derive(Debug, Clone)]
pub struct PyStateDiagnostics {
    inner: StateDiagnostics,
}

#[pymethods]
impl PyStateDiagnostics {
    /// Number of input-state objects before prediction.
    #[getter]
    fn input_object_count(&self) -> usize {
        self.inner.input_object_count
    }

    /// Number of output-state objects after prediction.
    #[getter]
    fn output_object_count(&self) -> usize {
        self.inner.output_object_count
    }

    /// Signed object-count delta (`output - input`).
    #[getter]
    fn object_count_delta(&self) -> i64 {
        self.inner.object_count_delta
    }

    /// Number of preserved input objects.
    #[getter]
    fn preserved_object_count(&self) -> usize {
        self.inner.preserved_object_count
    }

    /// Number of dropped input objects.
    #[getter]
    fn dropped_object_count(&self) -> usize {
        self.inner.dropped_object_count
    }

    /// Number of newly introduced output objects.
    #[getter]
    fn novel_object_count(&self) -> usize {
        self.inner.novel_object_count
    }

    /// Fraction of input objects preserved in the output.
    #[getter]
    fn object_preservation_rate(&self) -> f32 {
        self.inner.object_preservation_rate
    }

    /// Number of input-state relationships before prediction.
    #[getter]
    fn input_relationship_count(&self) -> usize {
        self.inner.input_relationship_count
    }

    /// Number of output-state relationships after prediction.
    #[getter]
    fn output_relationship_count(&self) -> usize {
        self.inner.output_relationship_count
    }

    /// Signed relationship-count delta (`output - input`).
    #[getter]
    fn relationship_count_delta(&self) -> i64 {
        self.inner.relationship_count_delta
    }

    /// Number of input relationships preserved in the output.
    #[getter]
    fn preserved_relationship_count(&self) -> usize {
        self.inner.preserved_relationship_count
    }

    /// Fraction of input relationships preserved in the output.
    #[getter]
    fn relationship_preservation_rate(&self) -> f32 {
        self.inner.relationship_preservation_rate
    }

    /// Mean positional drift across preserved objects.
    #[getter]
    fn average_position_shift(&self) -> f32 {
        self.inner.average_position_shift
    }

    /// Maximum positional drift across preserved objects.
    #[getter]
    fn max_position_shift(&self) -> f32 {
        self.inner.max_position_shift
    }

    /// Stable identities of dropped objects.
    fn dropped_objects(&self) -> Vec<PyObjectDiagnostic> {
        self.inner
            .dropped_objects
            .iter()
            .cloned()
            .map(|inner| PyObjectDiagnostic { inner })
            .collect()
    }

    /// Stable identities of newly introduced objects.
    fn novel_objects(&self) -> Vec<PyObjectDiagnostic> {
        self.inner
            .novel_objects
            .iter()
            .cloned()
            .map(|inner| PyObjectDiagnostic { inner })
            .collect()
    }

    /// Drift details for preserved objects.
    fn object_drifts(&self) -> Vec<PyObjectDrift> {
        self.inner
            .object_drifts
            .iter()
            .cloned()
            .map(|inner| PyObjectDrift { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "StateDiagnostics(objects={} ({:+}), relationships={} ({:+}), avg_shift={:.3})",
            self.inner.output_object_count,
            self.inner.object_count_delta,
            self.inner.output_relationship_count,
            self.inner.relationship_count_delta,
            self.inner.average_position_shift
        )
    }
}

/// Pairwise agreement summary within a multi-provider comparison.
#[pyclass(name = "PairwiseAgreement")]
#[derive(Debug, Clone)]
pub struct PyPairwiseAgreement {
    inner: PairwiseAgreement,
}

#[pymethods]
impl PyPairwiseAgreement {
    /// First provider name.
    #[getter]
    fn provider_a(&self) -> &str {
        &self.inner.provider_a
    }

    /// Second provider name.
    #[getter]
    fn provider_b(&self) -> &str {
        &self.inner.provider_b
    }

    /// Aggregate agreement score for the provider pair.
    #[getter]
    fn agreement_score(&self) -> f32 {
        self.inner.agreement_score
    }

    /// Number of shared output objects by stable ID.
    #[getter]
    fn common_object_count(&self) -> usize {
        self.inner.common_object_count
    }

    /// Jaccard overlap across output object IDs.
    #[getter]
    fn object_overlap_rate(&self) -> f32 {
        self.inner.object_overlap_rate
    }

    /// Signed output object-count delta (`provider_a - provider_b`).
    #[getter]
    fn object_count_delta(&self) -> i64 {
        self.inner.object_count_delta
    }

    /// Jaccard overlap across output relationships.
    #[getter]
    fn relationship_overlap_rate(&self) -> f32 {
        self.inner.relationship_overlap_rate
    }

    /// Signed output relationship-count delta (`provider_a - provider_b`).
    #[getter]
    fn relationship_count_delta(&self) -> i64 {
        self.inner.relationship_count_delta
    }

    /// Position-based agreement score across shared objects.
    #[getter]
    fn position_agreement(&self) -> f32 {
        self.inner.position_agreement
    }

    /// Mean positional delta across shared objects.
    #[getter]
    fn average_position_delta(&self) -> f32 {
        self.inner.average_position_delta
    }

    /// Maximum positional delta across shared objects.
    #[getter]
    fn max_position_delta(&self) -> f32 {
        self.inner.max_position_delta
    }

    /// Fraction of guardrail outcomes that agree by name.
    #[getter]
    fn guardrail_agreement_rate(&self) -> f32 {
        self.inner.guardrail_agreement_rate
    }

    /// Absolute physics-score delta between providers.
    #[getter]
    fn physics_score_delta(&self) -> f32 {
        self.inner.physics_score_delta
    }

    /// Absolute confidence delta between providers.
    #[getter]
    fn confidence_delta(&self) -> f32 {
        self.inner.confidence_delta
    }

    /// Absolute latency delta in milliseconds.
    #[getter]
    fn latency_delta_ms(&self) -> u64 {
        self.inner.latency_delta_ms
    }

    /// Highest positional disagreements across shared objects.
    fn positional_disagreements(&self) -> Vec<PyObjectDrift> {
        self.inner
            .positional_disagreements
            .iter()
            .cloned()
            .map(|inner| PyObjectDrift { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PairwiseAgreement(provider_a='{}', provider_b='{}', agreement_score={:.2})",
            self.inner.provider_a, self.inner.provider_b, self.inner.agreement_score
        )
    }
}

/// Consensus diagnostics shared across all provider outputs in a comparison.
#[pyclass(name = "ComparisonConsensus")]
#[derive(Debug, Clone)]
pub struct PyComparisonConsensus {
    inner: ComparisonConsensus,
}

#[pymethods]
impl PyComparisonConsensus {
    /// Number of object identities shared by every provider output.
    #[getter]
    fn shared_object_count(&self) -> usize {
        self.inner.shared_object_count
    }

    /// Shared object identities sorted by name then ID.
    fn shared_objects(&self) -> Vec<PyObjectDiagnostic> {
        self.inner
            .shared_objects
            .iter()
            .cloned()
            .map(|inner| PyObjectDiagnostic { inner })
            .collect()
    }

    /// Number of relationships shared by every provider output.
    #[getter]
    fn shared_relationship_count(&self) -> usize {
        self.inner.shared_relationship_count
    }

    /// Mean provider confidence across the comparison set.
    #[getter]
    fn average_confidence(&self) -> f32 {
        self.inner.average_confidence
    }

    /// Mean guardrail pass rate across the comparison set.
    #[getter]
    fn average_guardrail_pass_rate(&self) -> f32 {
        self.inner.average_guardrail_pass_rate
    }

    /// Mean provider quality score across the comparison set.
    #[getter]
    fn average_quality_score(&self) -> f32 {
        self.inner.average_quality_score
    }

    /// Mean provider latency across the comparison set.
    #[getter]
    fn average_latency_ms(&self) -> u64 {
        self.inner.average_latency_ms
    }

    /// Mean pairwise position delta across all provider pairs.
    #[getter]
    fn average_pairwise_position_delta(&self) -> f32 {
        self.inner.average_pairwise_position_delta
    }

    fn __repr__(&self) -> String {
        format!(
            "ComparisonConsensus(shared_object_count={}, shared_relationship_count={}, average_quality_score={:.2})",
            self.inner.shared_object_count,
            self.inner.shared_relationship_count,
            self.inner.average_quality_score
        )
    }
}

/// Provider-specific score summary within a multi-provider comparison.
#[pyclass(name = "ProviderScore")]
#[derive(Debug, Clone)]
pub struct PyProviderScore {
    inner: ProviderScore,
}

#[pymethods]
impl PyProviderScore {
    /// Provider name.
    #[getter]
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    /// Provider-reported confidence score.
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

    /// Provider cost estimate.
    fn cost(&self) -> PyCostEstimate {
        PyCostEstimate {
            inner: self.inner.cost.clone(),
        }
    }

    /// Composite quality score used to rank the best provider.
    #[getter]
    fn quality_score(&self) -> f32 {
        self.inner.quality_score
    }

    /// Guardrail diagnostics for the provider prediction.
    fn guardrails(&self) -> PyGuardrailDiagnostics {
        PyGuardrailDiagnostics {
            inner: self.inner.guardrails.clone(),
        }
    }

    /// State-derived diagnostics for the provider prediction.
    fn state(&self) -> PyStateDiagnostics {
        PyStateDiagnostics {
            inner: self.inner.state.clone(),
        }
    }

    /// Serialize the score summary to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ProviderScore(provider='{}', physics_score={:.2}, confidence={:.2}, latency_ms={})",
            self.inner.provider,
            self.inner.physics_scores.overall,
            self.inner.confidence,
            self.inner.latency_ms
        )
    }
}

/// Result of comparing predictions across providers.
#[pyclass(name = "MultiPrediction")]
#[derive(Debug, Clone)]
pub struct PyMultiPrediction {
    inner: MultiPrediction,
}

#[pymethods]
impl PyMultiPrediction {
    /// Number of provider predictions included in the comparison.
    #[getter]
    fn prediction_count(&self) -> usize {
        self.inner.predictions.len()
    }

    /// Agreement score across providers.
    #[getter]
    fn agreement_score(&self) -> f32 {
        self.inner.agreement_score
    }

    /// Index of the best prediction within the comparison.
    #[getter]
    fn best_prediction_index(&self) -> usize {
        self.inner.best_prediction
    }

    /// Human-readable comparison summary.
    #[getter]
    fn summary(&self) -> &str {
        &self.inner.comparison.summary
    }

    /// Individual predictions returned by each provider.
    fn predictions(&self) -> Vec<PyPrediction> {
        self.inner
            .predictions
            .iter()
            .cloned()
            .map(|inner| PyPrediction { inner })
            .collect()
    }

    /// Provider-level score summaries for the comparison.
    fn provider_scores(&self) -> Vec<PyProviderScore> {
        self.inner
            .comparison
            .scores
            .iter()
            .cloned()
            .map(|inner| PyProviderScore { inner })
            .collect()
    }

    /// Pairwise agreement summaries across provider outputs.
    fn pairwise_agreements(&self) -> Vec<PyPairwiseAgreement> {
        self.inner
            .comparison
            .pairwise_agreements
            .iter()
            .cloned()
            .map(|inner| PyPairwiseAgreement { inner })
            .collect()
    }

    /// Consensus diagnostics shared across all compared providers.
    fn consensus(&self) -> PyComparisonConsensus {
        PyComparisonConsensus {
            inner: self.inner.comparison.consensus.clone(),
        }
    }

    /// The highest-quality prediction in the comparison.
    fn best_prediction(&self) -> PyPrediction {
        PyPrediction {
            inner: self.inner.predictions[self.inner.best_prediction].clone(),
        }
    }

    /// Serialize the comparison to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Deserialize a comparison from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: MultiPrediction = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiPrediction(predictions={}, agreement_score={:.2}, best='{}')",
            self.inner.predictions.len(),
            self.inner.agreement_score,
            self.inner.predictions[self.inner.best_prediction].provider
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

/// Output of a provider embedding request.
#[pyclass(name = "EmbeddingOutput")]
#[derive(Debug, Clone)]
pub struct PyEmbeddingOutput {
    inner: EmbeddingOutput,
}

#[pymethods]
impl PyEmbeddingOutput {
    /// Provider identifier that produced the embedding.
    #[getter]
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    /// Model identifier used for the embedding request.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// Embedding tensor shape.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.embedding.shape.clone()
    }

    /// Tensor element type.
    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.embedding.dtype)
    }

    /// Tensor device.
    #[getter]
    fn device(&self) -> String {
        format!("{:?}", self.inner.embedding.device)
    }

    /// Flattened embedding vector as `f32` values.
    #[getter]
    fn vector(&self) -> Vec<f32> {
        tensor_data_to_f32_vec(&self.inner.embedding.data)
    }

    /// Serialize the embedding output to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Deserialize an embedding output from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: EmbeddingOutput = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingOutput(provider='{}', model='{}', shape={:?})",
            self.inner.provider, self.inner.model, self.inner.embedding.shape
        )
    }
}

/// Output of a provider reasoning query.
#[pyclass(name = "ReasoningOutput")]
#[derive(Debug, Clone)]
pub struct PyReasoningOutput {
    inner: ReasoningOutput,
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

/// Mock provider exposed to Python for tests and offline workflows.
#[pyclass(name = "MockProvider")]
#[derive(Debug, Clone)]
pub struct PyMockProvider {
    inner: MockProvider,
}

#[pymethods]
impl PyMockProvider {
    #[new]
    #[pyo3(signature = (name="mock", latency_ms=10, default_confidence=0.85))]
    fn new(name: &str, latency_ms: u64, default_confidence: f32) -> Self {
        let mut inner = if name == "mock" {
            MockProvider::new()
        } else {
            MockProvider::with_name(name)
        };
        inner.latency_ms = latency_ms;
        inner.default_confidence = default_confidence;
        Self { inner }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn capabilities(&self) -> PyProviderCapabilities {
        PyProviderCapabilities {
            inner: self.inner.capabilities(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MockProvider(name='{}', latency_ms={}, default_confidence={:.2})",
            self.inner.name(),
            self.inner.latency_ms,
            self.inner.default_confidence
        )
    }
}

impl PyMockProvider {
    fn boxed_provider(&self) -> Box<dyn WorldModelProvider> {
        Box::new(self.inner.clone())
    }
}

/// NVIDIA Cosmos provider exposed to Python.
#[pyclass(name = "CosmosProvider")]
#[derive(Debug, Clone)]
pub struct PyCosmosProvider {
    inner: CosmosProvider,
}

#[pymethods]
impl PyCosmosProvider {
    #[new]
    #[pyo3(signature = (api_key, model="predict-2.5", endpoint="https://ai.api.nvidia.com"))]
    fn new(api_key: &str, model: &str, endpoint: &str) -> PyResult<Self> {
        let model = parse_cosmos_model(model)?;
        Ok(Self {
            inner: CosmosProvider::new(
                model,
                api_key,
                worldforge_providers::cosmos::CosmosEndpoint::NimApi(endpoint.to_string()),
            ),
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn capabilities(&self) -> PyProviderCapabilities {
        PyProviderCapabilities {
            inner: self.inner.capabilities(),
        }
    }

    fn __repr__(&self) -> String {
        let model = match self.inner.model {
            worldforge_providers::cosmos::CosmosModel::Predict2_5 => "predict-2.5",
            worldforge_providers::cosmos::CosmosModel::Transfer2_5 => "transfer-2.5",
            worldforge_providers::cosmos::CosmosModel::Reason2 => "reason-2",
            worldforge_providers::cosmos::CosmosModel::Embed1 => "embed-1",
        };
        let endpoint = match &self.inner.endpoint {
            worldforge_providers::cosmos::CosmosEndpoint::NimApi(endpoint)
            | worldforge_providers::cosmos::CosmosEndpoint::NimLocal(endpoint)
            | worldforge_providers::cosmos::CosmosEndpoint::DgxCloud(endpoint) => endpoint,
            worldforge_providers::cosmos::CosmosEndpoint::HuggingFace => "huggingface",
        };
        format!("CosmosProvider(model='{}', endpoint='{}')", model, endpoint)
    }
}

impl PyCosmosProvider {
    fn boxed_provider(&self) -> Box<dyn WorldModelProvider> {
        Box::new(self.inner.clone())
    }
}

/// Runway GWM provider exposed to Python.
#[pyclass(name = "RunwayProvider")]
#[derive(Debug, Clone)]
pub struct PyRunwayProvider {
    inner: RunwayProvider,
}

#[pymethods]
impl PyRunwayProvider {
    #[new]
    #[pyo3(signature = (api_secret, model="worlds", endpoint=None))]
    fn new(api_secret: &str, model: &str, endpoint: Option<&str>) -> PyResult<Self> {
        let model = parse_runway_model(model)?;
        let inner = match endpoint {
            Some(endpoint) => RunwayProvider::with_endpoint(model, api_secret, endpoint),
            None => RunwayProvider::new(model, api_secret),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn capabilities(&self) -> PyProviderCapabilities {
        PyProviderCapabilities {
            inner: self.inner.capabilities(),
        }
    }

    fn __repr__(&self) -> String {
        let model = match self.inner.model {
            worldforge_providers::runway::RunwayModel::Gwm1Worlds => "worlds",
            worldforge_providers::runway::RunwayModel::Gwm1Robotics => "robotics",
            worldforge_providers::runway::RunwayModel::Gwm1Avatars => "avatars",
        };
        format!(
            "RunwayProvider(model='{}', endpoint='{}')",
            model, self.inner.endpoint
        )
    }
}

impl PyRunwayProvider {
    fn boxed_provider(&self) -> Box<dyn WorldModelProvider> {
        Box::new(self.inner.clone())
    }
}

/// Local JEPA provider exposed to Python.
#[pyclass(name = "JepaProvider")]
#[derive(Debug, Clone)]
pub struct PyJepaProvider {
    inner: JepaProvider,
}

#[pymethods]
impl PyJepaProvider {
    #[new]
    #[pyo3(signature = (model_path, backend="burn"))]
    fn new(model_path: &str, backend: &str) -> PyResult<Self> {
        Ok(Self {
            inner: JepaProvider::new(model_path, parse_jepa_backend(backend)?),
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn capabilities(&self) -> PyProviderCapabilities {
        PyProviderCapabilities {
            inner: self.inner.capabilities(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "JepaProvider(model_path='{}', backend='{:?}')",
            self.inner.model_path.display(),
            self.inner.backend
        )
    }
}

impl PyJepaProvider {
    fn boxed_provider(&self) -> Box<dyn WorldModelProvider> {
        Box::new(self.inner.clone())
    }
}

/// Google Genie provider exposed to Python.
#[pyclass(name = "GenieProvider")]
#[derive(Debug, Clone)]
pub struct PyGenieProvider {
    inner: GenieProvider,
}

#[pymethods]
impl PyGenieProvider {
    #[new]
    /// Create a Genie provider wrapper.
    ///
    /// The API key is optional for the local surrogate.
    #[pyo3(signature = (api_key=None, model="genie3", endpoint=None))]
    fn new(api_key: Option<&str>, model: &str, endpoint: Option<&str>) -> PyResult<Self> {
        let model = parse_genie_model(model)?;
        let api_key = api_key.unwrap_or("");
        let inner = match endpoint {
            Some(endpoint) => GenieProvider::with_endpoint(model, api_key, endpoint),
            None => GenieProvider::new(model, api_key),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn capabilities(&self) -> PyProviderCapabilities {
        PyProviderCapabilities {
            inner: self.inner.capabilities(),
        }
    }

    fn __repr__(&self) -> String {
        let model = match self.inner.model {
            worldforge_providers::genie::GenieModel::Genie3 => "genie3",
        };
        format!(
            "GenieProvider(model='{}', endpoint='{}')",
            model, self.inner.endpoint
        )
    }
}

impl PyGenieProvider {
    fn boxed_provider(&self) -> Box<dyn WorldModelProvider> {
        Box::new(self.inner.clone())
    }
}

fn boxed_python_provider(provider: &Bound<'_, PyAny>) -> PyResult<Box<dyn WorldModelProvider>> {
    if let Ok(provider) = provider.extract::<PyRef<'_, PyMockProvider>>() {
        return Ok(provider.boxed_provider());
    }
    if let Ok(provider) = provider.extract::<PyRef<'_, PyCosmosProvider>>() {
        return Ok(provider.boxed_provider());
    }
    if let Ok(provider) = provider.extract::<PyRef<'_, PyRunwayProvider>>() {
        return Ok(provider.boxed_provider());
    }
    if let Ok(provider) = provider.extract::<PyRef<'_, PyJepaProvider>>() {
        return Ok(provider.boxed_provider());
    }
    if let Ok(provider) = provider.extract::<PyRef<'_, PyGenieProvider>>() {
        return Ok(provider.boxed_provider());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "unsupported provider object. Expected MockProvider, CosmosProvider, RunwayProvider, JepaProvider, or GenieProvider",
    ))
}

/// Provider capability metadata.
#[pyclass(name = "ProviderCapabilities")]
#[derive(Debug, Clone)]
pub struct PyProviderCapabilities {
    inner: ProviderCapabilities,
}

#[pymethods]
impl PyProviderCapabilities {
    #[getter]
    fn predict(&self) -> bool {
        self.inner.predict
    }

    #[getter]
    fn generate(&self) -> bool {
        self.inner.generate
    }

    #[getter]
    fn reason(&self) -> bool {
        self.inner.reason
    }

    #[getter]
    fn transfer(&self) -> bool {
        self.inner.transfer
    }

    #[getter]
    fn embed(&self) -> bool {
        self.inner.embed
    }

    #[getter]
    fn supports_planning(&self) -> bool {
        self.inner.supports_planning
    }

    #[getter]
    fn action_conditioned(&self) -> bool {
        self.inner.action_conditioned
    }

    #[getter]
    fn multi_view(&self) -> bool {
        self.inner.multi_view
    }

    #[getter]
    fn supports_depth(&self) -> bool {
        self.inner.supports_depth
    }

    #[getter]
    fn supports_segmentation(&self) -> bool {
        self.inner.supports_segmentation
    }

    #[getter]
    fn max_resolution(&self) -> (u32, u32) {
        self.inner.max_resolution
    }

    #[getter]
    fn fps_range(&self) -> (f32, f32) {
        self.inner.fps_range
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ProviderCapabilities(predict={}, generate={}, reason={}, transfer={}, embed={}, planning={})",
            self.inner.predict,
            self.inner.generate,
            self.inner.reason,
            self.inner.transfer,
            self.inner.embed,
            self.inner.supports_planning
        )
    }
}

/// Provider descriptor exposed to Python.
#[pyclass(name = "ProviderInfo")]
#[derive(Debug, Clone)]
pub struct PyProviderInfo {
    inner: ProviderDescriptor,
}

#[pymethods]
impl PyProviderInfo {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn capabilities(&self) -> PyProviderCapabilities {
        PyProviderCapabilities {
            inner: self.inner.capabilities.clone(),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!("ProviderInfo(name={})", self.inner.name)
    }
}

/// Provider metadata paired with a live health-check result.
#[pyclass(name = "ProviderHealthInfo")]
#[derive(Debug, Clone)]
pub struct PyProviderHealthInfo {
    inner: ProviderHealthReport,
}

#[pymethods]
impl PyProviderHealthInfo {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn capabilities(&self) -> PyProviderCapabilities {
        PyProviderCapabilities {
            inner: self.inner.capabilities.clone(),
        }
    }

    #[getter]
    fn healthy(&self) -> bool {
        self.inner.is_healthy()
    }

    #[getter]
    fn message(&self) -> Option<String> {
        self.inner
            .status
            .as_ref()
            .map(|status| status.message.clone())
    }

    #[getter]
    fn latency_ms(&self) -> Option<u64> {
        self.inner.status.as_ref().map(|status| status.latency_ms)
    }

    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error.clone()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        match (&self.inner.status, &self.inner.error) {
            (Some(status), None) => format!(
                "ProviderHealthInfo(name={}, healthy={}, latency_ms={})",
                self.inner.name, status.healthy, status.latency_ms
            ),
            (_, Some(error)) => {
                format!(
                    "ProviderHealthInfo(name={}, error={error})",
                    self.inner.name
                )
            }
            (None, None) => format!(
                "ProviderHealthInfo(name={}, healthy=false)",
                self.inner.name
            ),
        }
    }
}

/// Cost estimate exposed to Python.
#[pyclass(name = "CostEstimate")]
#[derive(Debug, Clone)]
pub struct PyCostEstimate {
    inner: CostEstimate,
}

#[pymethods]
impl PyCostEstimate {
    #[getter]
    fn usd(&self) -> f64 {
        self.inner.usd
    }

    #[getter]
    fn credits(&self) -> f64 {
        self.inner.credits
    }

    #[getter]
    fn estimated_latency_ms(&self) -> u64 {
        self.inner.estimated_latency_ms
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CostEstimate(usd={:.4}, credits={:.2}, latency_ms={})",
            self.inner.usd, self.inner.credits, self.inner.estimated_latency_ms
        )
    }
}

/// The main WorldForge orchestrator.
///
/// Manages provider registration and world creation.
#[pyclass(name = "WorldForge")]
pub struct PyWorldForge {
    inner: worldforge_core::WorldForge,
}

#[pymethods]
impl PyWorldForge {
    /// Create a new WorldForge instance with auto-detected providers.
    #[new]
    #[pyo3(signature = (state_backend="file", state_dir=".worldforge", state_db_path=None, state_file_format="json", state_redis_url=None))]
    fn new(
        state_backend: &str,
        state_dir: &str,
        state_db_path: Option<&str>,
        state_file_format: &str,
        state_redis_url: Option<&str>,
    ) -> PyResult<Self> {
        let store_kind = state_store_kind(
            state_backend,
            state_dir,
            state_db_path,
            state_file_format,
            state_redis_url,
        )?;
        let rt = new_runtime()?;
        let store = rt.block_on(store_kind.open()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to open state store: {e}"))
        })?;
        let mut wf = worldforge_providers::auto_detect_worldforge_with_state_store(store);
        ensure_python_default_genie(&mut wf)?;
        Ok(Self { inner: wf })
    }

    /// Register a provider before any worlds are created.
    ///
    /// This accepts the provider wrappers exported from `worldforge.providers`.
    /// If a world has already been created, the underlying registry is no longer
    /// mutable and the registration attempt fails with a runtime error.
    fn register_provider(&mut self, provider: &Bound<'_, PyAny>) -> PyResult<()> {
        let provider = boxed_python_provider(provider).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("failed to register provider: {e}"))
        })?;
        self.inner.register_provider(provider).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to register provider: {e}"))
        })
    }

    /// List all registered provider names.
    fn providers(&self) -> Vec<String> {
        self.inner
            .providers()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Describe all registered providers, optionally filtering by capability.
    #[pyo3(signature = (capability=None))]
    fn provider_infos(&self, capability: Option<&str>) -> Vec<PyProviderInfo> {
        let descriptors = self.inner.provider_infos(capability);
        descriptors
            .into_iter()
            .map(|inner| PyProviderInfo { inner })
            .collect()
    }

    /// Describe one registered provider.
    fn provider_info(&self, provider: &str) -> PyResult<PyProviderInfo> {
        let descriptor = self.inner.provider_info(provider).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to describe provider: {e}"))
        })?;
        Ok(PyProviderInfo { inner: descriptor })
    }

    /// Run live health checks across all registered providers, optionally filtering by capability.
    #[pyo3(signature = (capability=None))]
    fn provider_healths(&self, capability: Option<&str>) -> PyResult<Vec<PyProviderHealthInfo>> {
        let rt = new_runtime()?;
        let reports = rt.block_on(self.inner.provider_healths(capability));
        Ok(reports
            .into_iter()
            .map(|inner| PyProviderHealthInfo { inner })
            .collect())
    }

    /// Run a live health check for one registered provider.
    fn provider_health(&self, provider: &str) -> PyResult<PyProviderHealthInfo> {
        let rt = new_runtime()?;
        let report = rt
            .block_on(self.inner.provider_health(provider))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to check provider health: {e}"
                ))
            })?;
        Ok(PyProviderHealthInfo { inner: report })
    }

    /// Estimate the cost of running an operation on a provider.
    #[pyo3(signature = (provider, operation="predict", steps=1, duration_seconds=4.0, width=1280, height=720))]
    #[allow(clippy::too_many_arguments)]
    fn estimate_cost(
        &self,
        provider: &str,
        operation: &str,
        steps: u32,
        duration_seconds: f64,
        width: u32,
        height: u32,
    ) -> PyResult<PyCostEstimate> {
        let operation = operation_from_args(operation, steps, duration_seconds, width, height)?;
        let estimate = self
            .inner
            .estimate_cost(provider, &operation)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to estimate provider cost: {e}"
                ))
            })?;
        Ok(PyCostEstimate { inner: estimate })
    }

    /// Request an embedding from a provider for text and/or video input.
    #[pyo3(signature = (provider, text=None, video=None, fallback_provider=None))]
    fn embed(
        &self,
        provider: &str,
        text: Option<&str>,
        video: Option<&PyVideoClip>,
        fallback_provider: Option<&str>,
    ) -> PyResult<PyEmbeddingOutput> {
        let input = EmbeddingInput::new(
            text.map(ToOwned::to_owned),
            video.map(|clip| clip.inner.clone()),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid input: {e}")))?;

        let rt = new_runtime()?;
        let output = rt
            .block_on(
                self.inner
                    .embed_with_fallback(provider, &input, fallback_provider),
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("embedding failed: {e}"))
            })?
            .1;
        Ok(PyEmbeddingOutput { inner: output })
    }

    /// Ask a provider to reason about a supplied world state or video clip.
    #[pyo3(signature = (provider, query, world=None, world_json=None, video=None, fallback_provider=None))]
    fn reason(
        &self,
        provider: &str,
        query: &str,
        world: Option<&PyWorld>,
        world_json: Option<&str>,
        video: Option<&PyVideoClip>,
        fallback_provider: Option<&str>,
    ) -> PyResult<PyReasoningOutput> {
        let world_state = resolve_eval_world_state(world, world_json)?;
        let video = video.map(|clip| clip.inner.clone());

        if world_state.is_none() && video.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "reason requires world, world_json, or video",
            ));
        }

        let input = ReasoningInput {
            state: world_state,
            video,
        };

        let rt = new_runtime()?;
        let (_, output) = rt
            .block_on(
                self.inner
                    .reason_with_fallback(provider, &input, query, fallback_provider),
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("reasoning failed: {e}"))
            })?;

        Ok(PyReasoningOutput { inner: output })
    }

    /// Create a new world with the given name and provider.
    #[pyo3(signature = (name, provider="mock"))]
    fn create_world(&self, name: &str, provider: &str) -> PyResult<PyWorld> {
        let registry = self.inner.registry_arc();
        let world = self.inner.create_world(name, provider).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to create world: {e}"))
        })?;
        Ok(PyWorld { world, registry })
    }

    /// Create a new world seeded from a natural-language prompt.
    #[pyo3(signature = (prompt, provider="mock", name=None))]
    fn create_world_from_prompt(
        &self,
        prompt: &str,
        provider: &str,
        name: Option<&str>,
    ) -> PyResult<PyWorld> {
        let registry = self.inner.registry_arc();
        let rt = new_runtime()?;
        let world = rt
            .block_on(self.inner.create_world_from_prompt(prompt, provider, name))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to create world from prompt: {e}"
                ))
            })?;
        Ok(PyWorld { world, registry })
    }

    /// Persist a world snapshot to the configured state store.
    fn save_world(&self, world: &PyWorld) -> PyResult<String> {
        let rt = new_runtime()?;
        let id = rt
            .block_on(self.inner.save_state(&world.world.state))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("failed to save world: {e}"))
            })?;
        Ok(id.to_string())
    }

    /// Load a world snapshot from the configured state store.
    fn load_world(&self, world_id: &str) -> PyResult<PyWorld> {
        let id = parse_world_id(world_id)?;
        let rt = new_runtime()?;
        let registry = self.inner.registry_arc();
        let world = rt
            .block_on(self.inner.load_world_from_store(&id))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("failed to load world: {e}"))
            })?;
        Ok(PyWorld { world, registry })
    }

    /// Export a persisted world snapshot.
    ///
    /// Returns a Python `str` when `format="json"` and raw `bytes` when
    /// `format="msgpack"`.
    #[pyo3(signature = (world_id, format="json"))]
    fn export_world(&self, py: Python<'_>, world_id: &str, format: &str) -> PyResult<Py<PyAny>> {
        let id = parse_world_id(world_id)?;
        let format = parse_snapshot_format(format)?;
        let rt = new_runtime()?;
        let state = rt.block_on(self.inner.load_state(&id)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to export world: {e}"))
        })?;
        let bytes = serialize_world_state(format, &state).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to export world: {e}"))
        })?;

        Ok(match format {
            CoreStateFileFormat::Json => {
                let json = String::from_utf8(bytes).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "failed to convert JSON snapshot to text: {e}"
                    ))
                })?;
                PyString::new_bound(py, &json).into_any().unbind()
            }
            CoreStateFileFormat::MessagePack => PyBytes::new_bound(py, &bytes).into_any().unbind(),
        })
    }

    /// Import a world snapshot into the configured store and return the stored world.
    ///
    /// Accepts JSON text or bytes when `format="json"` and MessagePack bytes
    /// when `format="msgpack"`. If `new_id` or `name` is supplied, the imported
    /// world is rebased to a fresh initial history entry before persistence.
    #[pyo3(signature = (snapshot, format="json", new_id=false, name=None))]
    fn import_world(
        &self,
        snapshot: &Bound<'_, PyAny>,
        format: &str,
        new_id: bool,
        name: Option<&str>,
    ) -> PyResult<PyWorld> {
        let format = parse_snapshot_format(format)?;
        let mut state =
            deserialize_world_state(format, &snapshot_bytes(snapshot)?).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("failed to import world: {e}"))
            })?;
        normalize_imported_state(&mut state, new_id, name)?;

        let rt = new_runtime()?;
        let registry = self.inner.registry_arc();
        rt.block_on(self.inner.save_state(&state)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to import world: {e}"))
        })?;
        Ok(world_from_state(state, registry))
    }

    /// List all persisted world IDs in the configured state store.
    fn list_worlds(&self) -> PyResult<Vec<String>> {
        let rt = new_runtime()?;
        let ids = rt.block_on(self.inner.list_worlds()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to list worlds: {e}"))
        })?;
        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }

    /// Delete a persisted world snapshot by ID.
    fn delete_world(&self, world_id: &str) -> PyResult<()> {
        let id = parse_world_id(world_id)?;
        let rt = new_runtime()?;
        rt.block_on(self.inner.delete_world(&id)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to delete world: {e}"))
        })?;
        Ok(())
    }

    /// Generate a video clip directly from a prompt with a specific provider.
    #[pyo3(signature = (prompt, provider="mock", duration_seconds=4.0, width=1280, height=720, fps=24.0, temperature=1.0, seed=None, negative_prompt=None, fallback_provider=None))]
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
        fallback_provider: Option<&str>,
    ) -> PyResult<PyVideoClip> {
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
        let rt = new_runtime()?;
        let world = CoreWorld::new(
            WorldState::new("python-generate", provider),
            provider,
            self.inner.registry_arc(),
        );
        let clip = rt
            .block_on(world.generate_with_provider_and_fallback(
                &prompt,
                &config,
                provider,
                fallback_provider,
            ))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("generation failed: {e}"))
            })?
            .1;
        Ok(PyVideoClip { inner: clip })
    }

    /// Transfer spatial controls over an existing clip with a specific provider.
    #[pyo3(signature = (clip, provider="mock", controls_json=None, width=1280, height=720, fps=24.0, control_strength=0.8, fallback_provider=None))]
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
        fallback_provider: Option<&str>,
    ) -> PyResult<PyVideoClip> {
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
        let world = CoreWorld::new(
            WorldState::new("python-transfer", provider),
            provider,
            self.inner.registry_arc(),
        );
        let clip = rt
            .block_on(world.transfer_with_provider_and_fallback(
                &clip.inner,
                &controls,
                &config,
                provider,
                fallback_provider,
            ))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("transfer failed: {e}"))
            })?
            .1;
        Ok(PyVideoClip { inner: clip })
    }

    /// Compare previously generated predictions.
    fn compare(&self, predictions: Vec<PyPrediction>) -> PyResult<PyMultiPrediction> {
        let raw_predictions: Vec<_> = predictions
            .into_iter()
            .map(|prediction| prediction.inner)
            .collect();
        let comparison = self.inner.compare(raw_predictions).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("comparison failed: {e}"))
        })?;
        Ok(PyMultiPrediction { inner: comparison })
    }

    fn __repr__(&self) -> String {
        format!("WorldForge(providers={:?})", self.inner.providers())
    }
}

fn ensure_python_default_genie(wf: &mut worldforge_core::WorldForge) -> PyResult<()> {
    if wf.providers().contains(&"genie") {
        return Ok(());
    }

    wf.register_provider(Box::new(GenieProvider::new(
        worldforge_providers::genie::GenieModel::Genie3,
        "",
    )))
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to register default Genie surrogate: {e}"
        ))
    })
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

    /// Generate a typed guardrail-compliance verification bundle for this plan.
    fn prove_guardrail_bundle(&self) -> PyResult<PyGuardrailBundle> {
        let verifier = worldforge_verify::MockVerifier::new();
        let bundle = prove_guardrail_plan_bundle(&verifier, &self.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
        })?;
        Ok(PyGuardrailBundle { inner: bundle })
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

/// Result of executing a plan against a live world.
#[pyclass(name = "PlanExecution")]
#[derive(Debug, Clone)]
pub struct PyPlanExecution {
    inner: PlanExecution,
}

#[pymethods]
impl PyPlanExecution {
    /// Number of executed plan steps.
    #[getter]
    fn step_count(&self) -> usize {
        self.inner.predictions.len()
    }

    /// End-to-end execution time in milliseconds.
    #[getter]
    fn execution_time_ms(&self) -> u64 {
        self.inner.execution_time_ms
    }

    /// Aggregate execution cost.
    fn total_cost(&self) -> PyCostEstimate {
        PyCostEstimate {
            inner: self.inner.total_cost.clone(),
        }
    }

    /// Per-step prediction results emitted while replaying the plan.
    fn predictions(&self) -> Vec<PyPrediction> {
        self.inner
            .predictions
            .iter()
            .cloned()
            .map(|inner| PyPrediction { inner })
            .collect()
    }

    /// Final committed world state after the plan completed.
    fn final_world(&self) -> PyWorld {
        let state = self.inner.final_state.clone();
        let provider = state.current_state_provider();
        let registry = auto_detect_registry();
        PyWorld {
            world: CoreWorld::new(state, provider, Arc::clone(&registry)),
            registry,
        }
    }

    /// Serialize the execution report to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    /// Deserialize an execution report from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: PlanExecution = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("deserialization error: {e}"))
        })?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "PlanExecution(steps={}, execution_time_ms={})",
            self.inner.predictions.len(),
            self.inner.execution_time_ms
        )
    }
}

/// Plan a sequence of actions in a world.
///
/// Supports sampling, CEM, MPC, gradient, and provider-native planning.
#[pyfunction]
#[pyo3(signature = (world, goal=None, goal_json=None, max_steps=10, timeout_seconds=30.0, provider="mock", planner="sampling", num_samples=None, top_k=None, population_size=None, elite_fraction=None, num_iterations=None, learning_rate=None, horizon=None, replanning_interval=None, guardrails_json=None, disable_guardrails=false))]
#[allow(clippy::too_many_arguments)]
fn plan(
    world: &PyWorld,
    goal: Option<&str>,
    goal_json: Option<&str>,
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
    disable_guardrails: bool,
) -> PyResult<PyPlan> {
    world.plan(
        goal,
        goal_json,
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
        disable_guardrails,
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
    /// Fraction of scenarios that fully passed.
    scenario_pass_rate: f32,
    /// Number of passed outcomes.
    outcomes_passed: usize,
    /// Total number of outcomes.
    total_outcomes: usize,
    /// Fraction of individual outcomes that passed.
    outcome_pass_rate: f32,
    /// Average score per dimension.
    dimension_scores: std::collections::HashMap<String, f32>,
    /// Scenario-level overall scores.
    scenario_scores: std::collections::HashMap<String, f32>,
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

    #[getter]
    fn scenario_pass_rate(&self) -> f32 {
        self.scenario_pass_rate
    }

    #[getter]
    fn outcomes_passed(&self) -> usize {
        self.outcomes_passed
    }

    #[getter]
    fn total_outcomes(&self) -> usize {
        self.total_outcomes
    }

    #[getter]
    fn outcome_pass_rate(&self) -> f32 {
        self.outcome_pass_rate
    }

    #[getter]
    fn dimension_scores(&self) -> std::collections::HashMap<String, f32> {
        self.dimension_scores.clone()
    }

    #[getter]
    fn scenario_scores(&self) -> std::collections::HashMap<String, f32> {
        self.scenario_scores.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EvalResult(provider='{}', score={:.2}, passed={}/{})",
            self.provider, self.average_score, self.scenarios_passed, self.total_scenarios
        )
    }
}

/// Aggregated dimension-level evaluation metrics.
#[pyclass(name = "EvalDimensionSummary")]
#[derive(Debug, Clone)]
pub struct PyEvalDimensionSummary {
    dimension: String,
    provider_scores: std::collections::HashMap<String, f32>,
    best_provider: Option<String>,
    best_score: Option<f32>,
}

#[pymethods]
impl PyEvalDimensionSummary {
    #[getter]
    fn dimension(&self) -> &str {
        &self.dimension
    }

    #[getter]
    fn provider_scores(&self) -> std::collections::HashMap<String, f32> {
        self.provider_scores.clone()
    }

    #[getter]
    fn best_provider(&self) -> Option<String> {
        self.best_provider.clone()
    }

    #[getter]
    fn best_score(&self) -> Option<f32> {
        self.best_score
    }
}

/// Aggregated scenario-level evaluation metrics.
#[pyclass(name = "EvalScenarioSummary")]
#[derive(Debug, Clone)]
pub struct PyEvalScenarioSummary {
    scenario: String,
    description: String,
    provider_scores: std::collections::HashMap<String, f32>,
    passed_by: Vec<String>,
    failed_by: Vec<String>,
    best_provider: Option<String>,
    best_score: Option<f32>,
    outcomes_passed: usize,
    total_outcomes: usize,
}

#[pymethods]
impl PyEvalScenarioSummary {
    #[getter]
    fn scenario(&self) -> &str {
        &self.scenario
    }

    #[getter]
    fn description(&self) -> &str {
        &self.description
    }

    #[getter]
    fn provider_scores(&self) -> std::collections::HashMap<String, f32> {
        self.provider_scores.clone()
    }

    #[getter]
    fn passed_by(&self) -> Vec<String> {
        self.passed_by.clone()
    }

    #[getter]
    fn failed_by(&self) -> Vec<String> {
        self.failed_by.clone()
    }

    #[getter]
    fn best_provider(&self) -> Option<String> {
        self.best_provider.clone()
    }

    #[getter]
    fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    #[getter]
    fn outcomes_passed(&self) -> usize {
        self.outcomes_passed
    }

    #[getter]
    fn total_outcomes(&self) -> usize {
        self.total_outcomes
    }
}

/// Full structured evaluation report.
#[pyclass(name = "EvalReport")]
#[derive(Debug, Clone)]
pub struct PyEvalReport {
    inner: worldforge_eval::EvalReport,
    suite: String,
    provider_summaries: Vec<PyEvalResult>,
    dimension_summaries: Vec<PyEvalDimensionSummary>,
    scenario_summaries: Vec<PyEvalScenarioSummary>,
    outcomes_passed: usize,
    total_outcomes: usize,
}

#[pymethods]
impl PyEvalReport {
    #[getter]
    fn suite(&self) -> &str {
        &self.suite
    }

    #[getter]
    fn provider_summaries(&self) -> Vec<PyEvalResult> {
        self.provider_summaries.clone()
    }

    #[getter]
    fn dimension_summaries(&self) -> Vec<PyEvalDimensionSummary> {
        self.dimension_summaries.clone()
    }

    #[getter]
    fn scenario_summaries(&self) -> Vec<PyEvalScenarioSummary> {
        self.scenario_summaries.clone()
    }

    #[getter]
    fn outcomes_passed(&self) -> usize {
        self.outcomes_passed
    }

    #[getter]
    fn total_outcomes(&self) -> usize {
        self.total_outcomes
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_pretty().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        worldforge_eval::EvalReport::from_json_str(json)
            .map(to_py_eval_report)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid eval report JSON: {e}"))
            })
    }

    fn to_markdown(&self) -> PyResult<String> {
        self.inner
            .to_markdown()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("render error: {e}")))
    }

    fn to_csv(&self) -> PyResult<String> {
        self.inner
            .to_csv()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("render error: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "EvalReport(suite='{}', providers={}, scenarios={})",
            self.suite,
            self.provider_summaries.len(),
            self.scenario_summaries.len()
        )
    }
}

/// A single scenario within an evaluation suite.
#[pyclass(name = "EvalScenario")]
#[derive(Debug, Clone)]
pub struct PyEvalScenario {
    inner: worldforge_eval::EvalScenario,
}

#[pymethods]
impl PyEvalScenario {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    #[getter]
    fn action_count(&self) -> usize {
        self.inner.actions.len()
    }

    #[getter]
    fn expected_outcome_count(&self) -> usize {
        self.inner.expected_outcomes.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "EvalScenario(name='{}', actions={}, expected_outcomes={})",
            self.inner.name,
            self.inner.actions.len(),
            self.inner.expected_outcomes.len()
        )
    }
}

/// A reusable evaluation suite wrapper.
#[pyclass(name = "EvalSuite")]
#[derive(Debug, Clone)]
pub struct PyEvalSuite {
    inner: worldforge_eval::EvalSuite,
}

#[pymethods]
impl PyEvalSuite {
    #[staticmethod]
    fn from_builtin(name: &str) -> PyResult<Self> {
        let inner = worldforge_eval::EvalSuite::from_builtin(name).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown eval suite: {e}"))
        })?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = worldforge_eval::EvalSuite::from_json_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid eval suite JSON: {e}"))
        })?;
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn scenario_count(&self) -> usize {
        self.inner.scenarios.len()
    }

    #[getter]
    fn dimensions(&self) -> Vec<String> {
        self.inner
            .dimensions
            .iter()
            .map(crate::eval_dimension_name)
            .collect()
    }

    #[getter]
    fn providers(&self) -> Vec<String> {
        self.inner.providers.clone()
    }

    fn scenarios(&self) -> Vec<PyEvalScenario> {
        self.inner
            .scenarios
            .iter()
            .cloned()
            .map(|inner| PyEvalScenario { inner })
            .collect()
    }

    /// Run the evaluation suite and return provider summaries.
    ///
    /// When `world` or `world_json` is supplied, each scenario is evaluated
    /// against that world state instead of its baked-in initial state.
    #[pyo3(signature = (providers="", world=None, world_json=None))]
    fn run(
        &self,
        providers: &str,
        world: Option<&PyWorld>,
        world_json: Option<&str>,
    ) -> PyResult<Vec<PyEvalResult>> {
        let world_state = resolve_eval_world_state(world, world_json)?;
        Ok(
            run_eval_suite_report_with_world(&self.inner, providers, world_state.as_ref())?
                .provider_summaries
                .iter()
                .map(to_py_eval_result)
                .collect(),
        )
    }

    /// Run the evaluation suite and return the full report JSON.
    ///
    /// When `world` or `world_json` is supplied, each scenario is evaluated
    /// against that world state instead of its baked-in initial state.
    #[pyo3(signature = (providers="", world=None, world_json=None))]
    fn run_report(
        &self,
        providers: &str,
        world: Option<&PyWorld>,
        world_json: Option<&str>,
    ) -> PyResult<String> {
        let world_state = resolve_eval_world_state(world, world_json)?;
        serde_json::to_string_pretty(&run_eval_suite_report_with_world(
            &self.inner,
            providers,
            world_state.as_ref(),
        )?)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to serialize evaluation report: {e}"
            ))
        })
    }

    /// Run the evaluation suite and return the structured report object.
    ///
    /// When `world` or `world_json` is supplied, each scenario is evaluated
    /// against that world state instead of its baked-in initial state.
    #[pyo3(signature = (providers="", world=None, world_json=None))]
    fn run_report_data(
        &self,
        providers: &str,
        world: Option<&PyWorld>,
        world_json: Option<&str>,
    ) -> PyResult<PyEvalReport> {
        let world_state = resolve_eval_world_state(world, world_json)?;
        Ok(to_py_eval_report(run_eval_suite_report_with_world(
            &self.inner,
            providers,
            world_state.as_ref(),
        )?))
    }

    /// Run the evaluation suite against a supplied world and return provider summaries.
    #[pyo3(signature = (providers="", world=None, world_json=None))]
    fn run_with_world(
        &self,
        providers: &str,
        world: Option<&PyWorld>,
        world_json: Option<&str>,
    ) -> PyResult<Vec<PyEvalResult>> {
        self.run(providers, world, world_json)
    }

    /// Run the evaluation suite against a supplied world and return the full report JSON.
    #[pyo3(signature = (providers="", world=None, world_json=None))]
    fn run_report_with_world(
        &self,
        providers: &str,
        world: Option<&PyWorld>,
        world_json: Option<&str>,
    ) -> PyResult<String> {
        self.run_report(providers, world, world_json)
    }

    /// Run the evaluation suite against a supplied world and return the structured report.
    #[pyo3(signature = (providers="", world=None, world_json=None))]
    fn run_report_data_with_world(
        &self,
        providers: &str,
        world: Option<&PyWorld>,
        world_json: Option<&str>,
    ) -> PyResult<PyEvalReport> {
        self.run_report_data(providers, world, world_json)
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_pretty().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "EvalSuite(name='{}', scenarios={}, dimensions={})",
            self.inner.name,
            self.inner.scenarios.len(),
            self.inner.dimensions.len()
        )
    }
}

/// Built-in physics evaluation helpers.
#[pyclass(name = "PhysicsEval")]
#[derive(Debug, Clone, Copy)]
pub struct PyPhysicsEval;

#[pymethods]
impl PyPhysicsEval {
    #[staticmethod]
    fn standard_suite() -> PyResult<PyEvalSuite> {
        PyEvalSuite::from_builtin("physics")
    }
}

/// Built-in manipulation evaluation helpers.
#[pyclass(name = "ManipulationEval")]
#[derive(Debug, Clone, Copy)]
pub struct PyManipulationEval;

#[pymethods]
impl PyManipulationEval {
    #[staticmethod]
    fn standard_suite() -> PyResult<PyEvalSuite> {
        PyEvalSuite::from_builtin("manipulation")
    }
}

/// Built-in spatial reasoning evaluation helpers.
#[pyclass(name = "SpatialEval")]
#[derive(Debug, Clone, Copy)]
pub struct PySpatialEval;

#[pymethods]
impl PySpatialEval {
    #[staticmethod]
    fn standard_suite() -> PyResult<PyEvalSuite> {
        PyEvalSuite::from_builtin("spatial")
    }
}

/// Built-in comprehensive evaluation helpers.
#[pyclass(name = "ComprehensiveEval")]
#[derive(Debug, Clone, Copy)]
pub struct PyComprehensiveEval;

#[pymethods]
impl PyComprehensiveEval {
    #[staticmethod]
    fn standard_suite() -> PyResult<PyEvalSuite> {
        PyEvalSuite::from_builtin("comprehensive")
    }
}

fn to_py_eval_result(summary: &worldforge_eval::ProviderSummary) -> PyEvalResult {
    PyEvalResult {
        provider: summary.provider.clone(),
        average_score: summary.average_score,
        average_latency_ms: summary.average_latency_ms,
        scenarios_passed: summary.scenarios_passed,
        total_scenarios: summary.total_scenarios,
        scenario_pass_rate: summary.scenario_pass_rate,
        outcomes_passed: summary.outcomes_passed,
        total_outcomes: summary.total_outcomes,
        outcome_pass_rate: summary.outcome_pass_rate,
        dimension_scores: summary.dimension_scores.clone(),
        scenario_scores: summary.scenario_scores.clone(),
    }
}

fn to_py_eval_report(report: worldforge_eval::EvalReport) -> PyEvalReport {
    let provider_summaries = report
        .provider_summaries
        .iter()
        .map(to_py_eval_result)
        .collect();
    let dimension_summaries = report
        .dimension_summaries
        .iter()
        .cloned()
        .map(|summary| PyEvalDimensionSummary {
            dimension: summary.dimension,
            provider_scores: summary.provider_scores,
            best_provider: summary.best_provider,
            best_score: summary.best_score,
        })
        .collect();
    let scenario_summaries = report
        .scenario_summaries
        .iter()
        .cloned()
        .map(|summary| PyEvalScenarioSummary {
            scenario: summary.scenario,
            description: summary.description,
            provider_scores: summary.provider_scores,
            passed_by: summary.passed_by,
            failed_by: summary.failed_by,
            best_provider: summary.best_provider,
            best_score: summary.best_score,
            outcomes_passed: summary.outcomes_passed,
            total_outcomes: summary.total_outcomes,
        })
        .collect();

    PyEvalReport {
        suite: report.suite.clone(),
        provider_summaries,
        dimension_summaries,
        scenario_summaries,
        outcomes_passed: report.outcomes_passed,
        total_outcomes: report.total_outcomes,
        inner: report,
    }
}

fn eval_dimension_name(dimension: &worldforge_eval::EvalDimension) -> String {
    match dimension {
        worldforge_eval::EvalDimension::ObjectPermanence => "object_permanence".to_string(),
        worldforge_eval::EvalDimension::GravityCompliance => "gravity_compliance".to_string(),
        worldforge_eval::EvalDimension::CollisionAccuracy => "collision_accuracy".to_string(),
        worldforge_eval::EvalDimension::SpatialConsistency => "spatial_consistency".to_string(),
        worldforge_eval::EvalDimension::TemporalConsistency => "temporal_consistency".to_string(),
        worldforge_eval::EvalDimension::ActionPredictionAccuracy => {
            "action_prediction_accuracy".to_string()
        }
        worldforge_eval::EvalDimension::MaterialUnderstanding => {
            "material_understanding".to_string()
        }
        worldforge_eval::EvalDimension::SpatialReasoning => "spatial_reasoning".to_string(),
        worldforge_eval::EvalDimension::Custom { name } => name.clone(),
    }
}

fn run_eval_suite_report(
    suite: &worldforge_eval::EvalSuite,
    providers: &str,
) -> PyResult<worldforge_eval::EvalReport> {
    run_eval_suite_report_with_world(suite, providers, None)
}

fn run_eval_suite_report_with_world(
    suite: &worldforge_eval::EvalSuite,
    providers: &str,
    world_state: Option<&WorldState>,
) -> PyResult<worldforge_eval::EvalReport> {
    let rt = new_runtime()?;
    let registry = auto_detect_registry();
    let provider_names = resolve_eval_provider_names(suite, providers);
    let mut provider_list: Vec<&dyn worldforge_core::provider::WorldModelProvider> = Vec::new();
    for provider_name in &provider_names {
        let provider = registry.get(provider_name).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("provider lookup failed: {e}"))
        })?;
        provider_list.push(provider);
    }

    match world_state {
        Some(world_state) => rt.block_on(suite.run_with_world_state(&provider_list, world_state)),
        None => rt.block_on(suite.run(&provider_list)),
    }
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("evaluation failed: {e}")))
}

fn resolve_eval_world_state(
    world: Option<&PyWorld>,
    world_json: Option<&str>,
) -> PyResult<Option<WorldState>> {
    match (world, world_json) {
        (Some(_), Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(
            "world and world_json are mutually exclusive",
        )),
        (Some(world), None) => Ok(Some(world.world.state.clone())),
        (None, Some(json)) => serde_json::from_str::<WorldState>(json)
            .map(Some)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("failed to parse world JSON: {e}"))
            }),
        (None, None) => Ok(None),
    }
}

/// List the built-in evaluation suite names.
#[pyfunction]
fn list_eval_suites() -> Vec<String> {
    worldforge_eval::EvalSuite::builtin_names()
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

/// Run an evaluation suite and return provider summaries.
#[pyfunction]
#[pyo3(signature = (suite_name="physics", providers="", suite_json=None))]
fn run_eval(
    suite_name: &str,
    providers: &str,
    suite_json: Option<&str>,
) -> PyResult<Vec<PyEvalResult>> {
    let suite = load_eval_suite(suite_name, suite_json)?;
    let report = run_eval_suite_report(&suite, providers)?;
    Ok(report
        .provider_summaries
        .iter()
        .map(to_py_eval_result)
        .collect())
}

/// Run an evaluation suite and return the full report JSON.
#[pyfunction]
#[pyo3(signature = (suite_name="physics", providers="", suite_json=None))]
fn run_eval_report(
    suite_name: &str,
    providers: &str,
    suite_json: Option<&str>,
) -> PyResult<String> {
    let suite = load_eval_suite(suite_name, suite_json)?;
    serde_json::to_string_pretty(&run_eval_suite_report(&suite, providers)?).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to serialize evaluation report: {e}"
        ))
    })
}

/// Run an evaluation suite and return the structured report object.
#[pyfunction]
#[pyo3(signature = (suite_name="physics", providers="", suite_json=None))]
fn run_eval_report_data(
    suite_name: &str,
    providers: &str,
    suite_json: Option<&str>,
) -> PyResult<PyEvalReport> {
    let suite = load_eval_suite(suite_name, suite_json)?;
    Ok(to_py_eval_report(run_eval_suite_report(&suite, providers)?))
}

fn register_child_module(
    py: Python<'_>,
    root: &Bound<'_, PyModule>,
    name: &str,
    module: &Bound<'_, PyModule>,
) -> PyResult<()> {
    root.add(name, module)?;
    let sys = py.import_bound("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item(format!("worldforge.{name}"), module)?;
    Ok(())
}

fn build_providers_submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let module = PyModule::new_bound(py, "worldforge.providers")?;
    module.add_class::<PyProviderCapabilities>()?;
    module.add_class::<PyMockProvider>()?;
    module.add_class::<PyCosmosProvider>()?;
    module.add_class::<PyRunwayProvider>()?;
    module.add_class::<PyJepaProvider>()?;
    module.add_class::<PyGenieProvider>()?;
    Ok(module)
}

fn build_eval_submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let module = PyModule::new_bound(py, "worldforge.eval")?;
    module.add_class::<PyEvalScenario>()?;
    module.add_class::<PyEvalSuite>()?;
    module.add_class::<PyPhysicsEval>()?;
    module.add_class::<PyManipulationEval>()?;
    module.add_class::<PySpatialEval>()?;
    module.add_class::<PyComprehensiveEval>()?;
    module.add_class::<PyEvalResult>()?;
    module.add_class::<PyEvalDimensionSummary>()?;
    module.add_class::<PyEvalScenarioSummary>()?;
    module.add_class::<PyEvalReport>()?;
    module.add_function(wrap_pyfunction!(list_eval_suites, &module)?)?;
    module.add_function(wrap_pyfunction!(run_eval, &module)?)?;
    module.add_function(wrap_pyfunction!(run_eval_report, &module)?)?;
    module.add_function(wrap_pyfunction!(run_eval_report_data, &module)?)?;
    Ok(module)
}

fn build_verify_submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let module = PyModule::new_bound(py, "worldforge.verify")?;
    module.add_class::<PyPlan>()?;
    module.add_class::<PyZkProof>()?;
    module.add_class::<PyVerificationResult>()?;
    module.add_class::<PyInferenceBundle>()?;
    module.add_class::<PyGuardrailBundle>()?;
    module.add_class::<PyProvenanceBundle>()?;
    module.add_class::<PyInferenceVerificationReport>()?;
    module.add_class::<PyGuardrailVerificationReport>()?;
    module.add_class::<PyProvenanceVerificationReport>()?;
    module.add_class::<PyZkVerifier>()?;
    module.add_class::<PyMockVerifier>()?;
    module.add_function(wrap_pyfunction!(prove_inference, &module)?)?;
    module.add_function(wrap_pyfunction!(prove_inference_transition, &module)?)?;
    module.add_function(wrap_pyfunction!(
        prove_inference_transition_bundle_py,
        &module
    )?)?;
    module.add_function(wrap_pyfunction!(prove_guardrail_plan, &module)?)?;
    module.add_function(wrap_pyfunction!(prove_provenance, &module)?)?;
    module.add_function(wrap_pyfunction!(verify_proof_json, &module)?)?;
    module.add_function(wrap_pyfunction!(verify_bundle_json, &module)?)?;
    Ok(module)
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
        let verifier = self.inner.backend.verifier();
        let result = verify_proof(verifier.as_ref(), &self.inner).map_err(|e| {
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

/// Result of verifying a proof or verification bundle.
#[pyclass(name = "VerificationResult")]
#[derive(Debug, Clone)]
pub struct PyVerificationResult {
    inner: CoreVerificationResult,
}

#[pymethods]
impl PyVerificationResult {
    /// Deserialize a verification result from JSON.
    #[staticmethod]
    fn from_json(result_json: &str) -> PyResult<Self> {
        let inner = serde_json::from_str(result_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid verification result JSON: {e}"
            ))
        })?;
        Ok(Self { inner })
    }

    /// Whether verification succeeded.
    #[getter]
    fn valid(&self) -> bool {
        self.inner.valid
    }

    /// Time spent verifying, in milliseconds.
    #[getter]
    fn verification_time_ms(&self) -> u64 {
        self.inner.verification_time_ms
    }

    /// Human-readable verification details.
    #[getter]
    fn details(&self) -> &str {
        &self.inner.details
    }

    /// Serialize the verification result to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "VerificationResult(valid={}, verification_time_ms={})",
            self.inner.valid, self.inner.verification_time_ms
        )
    }
}

/// Typed inference-verification bundle.
#[pyclass(name = "InferenceBundle")]
#[derive(Debug, Clone)]
pub struct PyInferenceBundle {
    inner: VerificationBundle<InferenceArtifact>,
}

#[pymethods]
impl PyInferenceBundle {
    #[staticmethod]
    fn from_json(bundle_json: &str) -> PyResult<Self> {
        let inner = serde_json::from_str(bundle_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid inference bundle JSON: {e}"))
        })?;
        Ok(Self { inner })
    }

    #[getter]
    fn provider(&self) -> &str {
        &self.inner.artifact.provider
    }

    #[getter]
    fn model_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.model_hash)
    }

    #[getter]
    fn input_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.input_hash)
    }

    #[getter]
    fn output_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.output_hash)
    }

    #[getter]
    fn proof(&self) -> PyZkProof {
        PyZkProof {
            inner: self.inner.proof.clone(),
        }
    }

    #[getter]
    fn verification(&self) -> PyVerificationResult {
        PyVerificationResult {
            inner: self.inner.verification.clone(),
        }
    }

    fn verify(&self) -> PyResult<PyInferenceVerificationReport> {
        verify_inference_bundle_inner(&self.inner)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "InferenceBundle(provider='{}')",
            self.inner.artifact.provider
        )
    }
}

/// Typed guardrail-compliance verification bundle.
#[pyclass(name = "GuardrailBundle")]
#[derive(Debug, Clone)]
pub struct PyGuardrailBundle {
    inner: VerificationBundle<GuardrailArtifact>,
}

#[pymethods]
impl PyGuardrailBundle {
    #[staticmethod]
    fn from_json(bundle_json: &str) -> PyResult<Self> {
        let inner = serde_json::from_str(bundle_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid guardrail bundle JSON: {e}"))
        })?;
        Ok(Self { inner })
    }

    #[getter]
    fn plan_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.plan_hash)
    }

    #[getter]
    fn guardrail_hashes_hex(&self) -> Vec<String> {
        format_hash_list_hex(&self.inner.artifact.guardrail_hashes)
    }

    #[getter]
    fn all_passed(&self) -> bool {
        self.inner.artifact.all_passed
    }

    #[getter]
    fn action_count(&self) -> usize {
        self.inner.artifact.action_count
    }

    #[getter]
    fn predicted_state_count(&self) -> usize {
        self.inner.artifact.predicted_state_count
    }

    #[getter]
    fn guardrail_step_count(&self) -> usize {
        self.inner.artifact.guardrail_step_count
    }

    #[getter]
    fn proof(&self) -> PyZkProof {
        PyZkProof {
            inner: self.inner.proof.clone(),
        }
    }

    #[getter]
    fn verification(&self) -> PyVerificationResult {
        PyVerificationResult {
            inner: self.inner.verification.clone(),
        }
    }

    fn verify(&self) -> PyResult<PyGuardrailVerificationReport> {
        verify_guardrail_bundle_inner(&self.inner)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GuardrailBundle(actions={}, all_passed={})",
            self.inner.artifact.action_count, self.inner.artifact.all_passed
        )
    }
}

/// Typed provenance-attestation verification bundle.
#[pyclass(name = "ProvenanceBundle")]
#[derive(Debug, Clone)]
pub struct PyProvenanceBundle {
    inner: VerificationBundle<ProvenanceArtifact>,
}

#[pymethods]
impl PyProvenanceBundle {
    #[staticmethod]
    fn from_json(bundle_json: &str) -> PyResult<Self> {
        let inner = serde_json::from_str(bundle_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid provenance bundle JSON: {e}"))
        })?;
        Ok(Self { inner })
    }

    #[getter]
    fn world_id(&self) -> String {
        self.inner.artifact.world_id.to_string()
    }

    #[getter]
    fn provider(&self) -> &str {
        &self.inner.artifact.provider
    }

    #[getter]
    fn data_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.data_hash)
    }

    #[getter]
    fn history_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.history_hash)
    }

    #[getter]
    fn timestamp(&self) -> u64 {
        self.inner.artifact.timestamp
    }

    #[getter]
    fn source_commitment_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.source_commitment)
    }

    #[getter]
    fn proof(&self) -> PyZkProof {
        PyZkProof {
            inner: self.inner.proof.clone(),
        }
    }

    #[getter]
    fn verification(&self) -> PyVerificationResult {
        PyVerificationResult {
            inner: self.inner.verification.clone(),
        }
    }

    fn verify(&self) -> PyResult<PyProvenanceVerificationReport> {
        verify_provenance_bundle_inner(&self.inner)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ProvenanceBundle(world_id='{}', provider='{}')",
            self.inner.artifact.world_id, self.inner.artifact.provider
        )
    }
}

/// Verification report for an inference bundle.
#[pyclass(name = "InferenceVerificationReport")]
#[derive(Debug, Clone)]
pub struct PyInferenceVerificationReport {
    inner: BundleVerificationReport<InferenceArtifact>,
}

#[pymethods]
impl PyInferenceVerificationReport {
    #[staticmethod]
    fn from_json(report_json: &str) -> PyResult<Self> {
        let inner = serde_json::from_str(report_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid inference verification report JSON: {e}"
            ))
        })?;
        Ok(Self { inner })
    }

    #[getter]
    fn provider(&self) -> &str {
        &self.inner.artifact.provider
    }

    #[getter]
    fn model_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.model_hash)
    }

    #[getter]
    fn input_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.input_hash)
    }

    #[getter]
    fn output_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.output_hash)
    }

    #[getter]
    fn proof(&self) -> PyZkProof {
        PyZkProof {
            inner: self.inner.proof.clone(),
        }
    }

    #[getter]
    fn recorded_verification(&self) -> PyVerificationResult {
        PyVerificationResult {
            inner: self.inner.recorded_verification.clone(),
        }
    }

    #[getter]
    fn current_verification(&self) -> PyVerificationResult {
        PyVerificationResult {
            inner: self.inner.current_verification.clone(),
        }
    }

    #[getter]
    fn verification_matches_recorded(&self) -> bool {
        self.inner.verification_matches_recorded
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }
}

/// Verification report for a guardrail bundle.
#[pyclass(name = "GuardrailVerificationReport")]
#[derive(Debug, Clone)]
pub struct PyGuardrailVerificationReport {
    inner: BundleVerificationReport<GuardrailArtifact>,
}

#[pymethods]
impl PyGuardrailVerificationReport {
    #[staticmethod]
    fn from_json(report_json: &str) -> PyResult<Self> {
        let inner = serde_json::from_str(report_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid guardrail verification report JSON: {e}"
            ))
        })?;
        Ok(Self { inner })
    }

    #[getter]
    fn plan_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.plan_hash)
    }

    #[getter]
    fn guardrail_hashes_hex(&self) -> Vec<String> {
        format_hash_list_hex(&self.inner.artifact.guardrail_hashes)
    }

    #[getter]
    fn all_passed(&self) -> bool {
        self.inner.artifact.all_passed
    }

    #[getter]
    fn action_count(&self) -> usize {
        self.inner.artifact.action_count
    }

    #[getter]
    fn predicted_state_count(&self) -> usize {
        self.inner.artifact.predicted_state_count
    }

    #[getter]
    fn guardrail_step_count(&self) -> usize {
        self.inner.artifact.guardrail_step_count
    }

    #[getter]
    fn proof(&self) -> PyZkProof {
        PyZkProof {
            inner: self.inner.proof.clone(),
        }
    }

    #[getter]
    fn recorded_verification(&self) -> PyVerificationResult {
        PyVerificationResult {
            inner: self.inner.recorded_verification.clone(),
        }
    }

    #[getter]
    fn current_verification(&self) -> PyVerificationResult {
        PyVerificationResult {
            inner: self.inner.current_verification.clone(),
        }
    }

    #[getter]
    fn verification_matches_recorded(&self) -> bool {
        self.inner.verification_matches_recorded
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }
}

/// Verification report for a provenance bundle.
#[pyclass(name = "ProvenanceVerificationReport")]
#[derive(Debug, Clone)]
pub struct PyProvenanceVerificationReport {
    inner: BundleVerificationReport<ProvenanceArtifact>,
}

#[pymethods]
impl PyProvenanceVerificationReport {
    #[staticmethod]
    fn from_json(report_json: &str) -> PyResult<Self> {
        let inner = serde_json::from_str(report_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid provenance verification report JSON: {e}"
            ))
        })?;
        Ok(Self { inner })
    }

    #[getter]
    fn world_id(&self) -> String {
        self.inner.artifact.world_id.to_string()
    }

    #[getter]
    fn provider(&self) -> &str {
        &self.inner.artifact.provider
    }

    #[getter]
    fn data_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.data_hash)
    }

    #[getter]
    fn history_hash_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.history_hash)
    }

    #[getter]
    fn timestamp(&self) -> u64 {
        self.inner.artifact.timestamp
    }

    #[getter]
    fn source_commitment_hex(&self) -> String {
        format_hash_hex(&self.inner.artifact.source_commitment)
    }

    #[getter]
    fn proof(&self) -> PyZkProof {
        PyZkProof {
            inner: self.inner.proof.clone(),
        }
    }

    #[getter]
    fn recorded_verification(&self) -> PyVerificationResult {
        PyVerificationResult {
            inner: self.inner.recorded_verification.clone(),
        }
    }

    #[getter]
    fn current_verification(&self) -> PyVerificationResult {
        PyVerificationResult {
            inner: self.inner.current_verification.clone(),
        }
    }

    #[getter]
    fn verification_matches_recorded(&self) -> bool {
        self.inner.verification_matches_recorded
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialization error: {e}"))
        })
    }
}

fn verify_inference_bundle_inner(
    bundle: &VerificationBundle<InferenceArtifact>,
) -> PyResult<PyInferenceVerificationReport> {
    let verifier = bundle.proof.backend.verifier();
    let inner = verify_bundle(verifier.as_ref(), bundle).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
    })?;
    Ok(PyInferenceVerificationReport { inner })
}

fn verify_guardrail_bundle_inner(
    bundle: &VerificationBundle<GuardrailArtifact>,
) -> PyResult<PyGuardrailVerificationReport> {
    let verifier = bundle.proof.backend.verifier();
    let inner = verify_bundle(verifier.as_ref(), bundle).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
    })?;
    Ok(PyGuardrailVerificationReport { inner })
}

fn verify_provenance_bundle_inner(
    bundle: &VerificationBundle<ProvenanceArtifact>,
) -> PyResult<PyProvenanceVerificationReport> {
    let verifier = bundle.proof.backend.verifier();
    let inner = verify_bundle(verifier.as_ref(), bundle).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
    })?;
    Ok(PyProvenanceVerificationReport { inner })
}

fn parse_verification_backend(backend: &str) -> PyResult<VerificationBackend> {
    backend.parse::<VerificationBackend>().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid verification backend: {e}"))
    })
}

fn verifier_from_backend_name(backend: &str) -> PyResult<Box<dyn ZkVerifier>> {
    Ok(parse_verification_backend(backend)?.verifier())
}

/// Mock-backed verifier facade exposed to Python.
#[pyclass(name = "ZkVerifier")]
#[derive(Debug, Clone)]
pub struct PyZkVerifier {
    backend: String,
}

#[pymethods]
impl PyZkVerifier {
    #[new]
    #[pyo3(signature = (backend="mock"))]
    fn new(backend: &str) -> PyResult<Self> {
        parse_verification_backend(backend)?;
        Ok(Self {
            backend: backend.to_ascii_lowercase(),
        })
    }

    #[getter]
    fn backend(&self) -> &str {
        &self.backend
    }

    fn prove_inference(
        &self,
        model_data: &[u8],
        input_data: &[u8],
        output_data: &[u8],
    ) -> PyResult<PyZkProof> {
        let verifier = verifier_from_backend_name(&self.backend)?;
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

    #[pyo3(signature = (input_state_json, output_state_json, provider=None))]
    fn prove_inference_transition(
        &self,
        input_state_json: &str,
        output_state_json: &str,
        provider: Option<&str>,
    ) -> PyResult<PyZkProof> {
        Ok(PyZkProof {
            inner: self
                .prove_inference_transition_bundle(input_state_json, output_state_json, provider)?
                .inner
                .proof,
        })
    }

    #[pyo3(signature = (input_state_json, output_state_json, provider=None))]
    fn prove_inference_transition_bundle(
        &self,
        input_state_json: &str,
        output_state_json: &str,
        provider: Option<&str>,
    ) -> PyResult<PyInferenceBundle> {
        let verifier = verifier_from_backend_name(&self.backend)?;
        let input_state: WorldState = serde_json::from_str(input_state_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid input state JSON: {e}"))
        })?;
        let output_state: WorldState = serde_json::from_str(output_state_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid output state JSON: {e}"))
        })?;
        let provider_name = provider.unwrap_or(output_state.metadata.created_by.as_str());
        let bundle = prove_inference_transition_bundle(
            verifier.as_ref(),
            provider_name,
            &input_state,
            &output_state,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
        })?;
        Ok(PyInferenceBundle { inner: bundle })
    }

    fn prove_guardrail_plan(&self, plan: &PyPlan) -> PyResult<PyZkProof> {
        let verifier = verifier_from_backend_name(&self.backend)?;
        let bundle = prove_guardrail_plan_bundle(verifier.as_ref(), &plan.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
        })?;
        Ok(PyZkProof {
            inner: bundle.proof,
        })
    }

    fn verify_inference_bundle(
        &self,
        bundle: &PyInferenceBundle,
    ) -> PyResult<PyInferenceVerificationReport> {
        let verifier = verifier_from_backend_name(&self.backend)?;
        let inner = verify_bundle(verifier.as_ref(), &bundle.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
        })?;
        Ok(PyInferenceVerificationReport { inner })
    }

    fn verify_guardrail_bundle(
        &self,
        bundle: &PyGuardrailBundle,
    ) -> PyResult<PyGuardrailVerificationReport> {
        let verifier = verifier_from_backend_name(&self.backend)?;
        let inner = verify_bundle(verifier.as_ref(), &bundle.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
        })?;
        Ok(PyGuardrailVerificationReport { inner })
    }

    fn verify_provenance_bundle(
        &self,
        bundle: &PyProvenanceBundle,
    ) -> PyResult<PyProvenanceVerificationReport> {
        let verifier = verifier_from_backend_name(&self.backend)?;
        let inner = verify_bundle(verifier.as_ref(), &bundle.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
        })?;
        Ok(PyProvenanceVerificationReport { inner })
    }

    fn prove_provenance(&self, data: &[u8], timestamp: u64, source: &[u8]) -> PyResult<PyZkProof> {
        let verifier = verifier_from_backend_name(&self.backend)?;
        let data_hash = worldforge_verify::sha256_hash(data);
        let source_commitment = worldforge_verify::sha256_hash(source);
        let proof = verifier
            .prove_data_provenance(data_hash, timestamp, source_commitment)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("proof generation failed: {e}"))
            })?;
        Ok(PyZkProof { inner: proof })
    }

    fn verify(&self, proof: &PyZkProof) -> PyResult<(bool, String)> {
        let verifier = verifier_from_backend_name(&self.backend)?;
        let result = verify_proof(verifier.as_ref(), &proof.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}"))
        })?;
        Ok((result.valid, result.details))
    }

    fn __repr__(&self) -> String {
        format!("ZkVerifier(backend='{}')", self.backend)
    }
}

/// Explicit mock verifier alias for Python users.
#[pyclass(name = "MockVerifier")]
#[derive(Debug, Clone, Copy, Default)]
pub struct PyMockVerifier;

#[pymethods]
impl PyMockVerifier {
    #[new]
    fn new() -> Self {
        Self
    }

    #[getter]
    fn backend(&self) -> &'static str {
        "mock"
    }

    fn prove_inference(
        &self,
        model_data: &[u8],
        input_data: &[u8],
        output_data: &[u8],
    ) -> PyResult<PyZkProof> {
        prove_inference(model_data, input_data, output_data)
    }

    #[pyo3(signature = (input_state_json, output_state_json, provider=None))]
    fn prove_inference_transition(
        &self,
        input_state_json: &str,
        output_state_json: &str,
        provider: Option<&str>,
    ) -> PyResult<PyZkProof> {
        prove_inference_transition(input_state_json, output_state_json, provider)
    }

    #[pyo3(signature = (input_state_json, output_state_json, provider=None))]
    fn prove_inference_transition_bundle(
        &self,
        input_state_json: &str,
        output_state_json: &str,
        provider: Option<&str>,
    ) -> PyResult<PyInferenceBundle> {
        prove_inference_transition_bundle_py(input_state_json, output_state_json, provider)
    }

    fn prove_guardrail_plan(&self, plan: &PyPlan) -> PyResult<PyZkProof> {
        prove_guardrail_plan(plan)
    }

    fn verify_inference_bundle(
        &self,
        bundle: &PyInferenceBundle,
    ) -> PyResult<PyInferenceVerificationReport> {
        bundle.verify()
    }

    fn verify_guardrail_bundle(
        &self,
        bundle: &PyGuardrailBundle,
    ) -> PyResult<PyGuardrailVerificationReport> {
        bundle.verify()
    }

    fn verify_provenance_bundle(
        &self,
        bundle: &PyProvenanceBundle,
    ) -> PyResult<PyProvenanceVerificationReport> {
        bundle.verify()
    }

    fn prove_provenance(&self, data: &[u8], timestamp: u64, source: &[u8]) -> PyResult<PyZkProof> {
        prove_provenance(data, timestamp, source)
    }

    fn verify(&self, proof: &PyZkProof) -> PyResult<(bool, String)> {
        proof.verify()
    }

    fn __repr__(&self) -> &'static str {
        "MockVerifier()"
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
    Ok(PyZkProof {
        inner: prove_inference_transition_bundle_py(input_state_json, output_state_json, provider)?
            .inner
            .proof,
    })
}

/// Generate a typed inference-verification bundle from two serialized `WorldState` payloads.
#[pyfunction(name = "prove_inference_transition_bundle")]
#[pyo3(signature = (input_state_json, output_state_json, provider=None))]
fn prove_inference_transition_bundle_py(
    input_state_json: &str,
    output_state_json: &str,
    provider: Option<&str>,
) -> PyResult<PyInferenceBundle> {
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

    Ok(PyInferenceBundle { inner: bundle })
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
    let report_json = match bundle_type {
        "inference" => {
            let bundle: VerificationBundle<worldforge_verify::InferenceArtifact> =
                serde_json::from_str(bundle_json).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid inference bundle JSON: {e}"
                    ))
                })?;
            let verifier = bundle.proof.backend.verifier();
            serde_json::to_string_pretty(&verify_bundle(verifier.as_ref(), &bundle).map_err(
                |e| pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}")),
            )?)
        }
        "guardrail" => {
            let bundle: VerificationBundle<worldforge_verify::GuardrailArtifact> =
                serde_json::from_str(bundle_json).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid guardrail bundle JSON: {e}"
                    ))
                })?;
            let verifier = bundle.proof.backend.verifier();
            serde_json::to_string_pretty(&verify_bundle(verifier.as_ref(), &bundle).map_err(
                |e| pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}")),
            )?)
        }
        "provenance" => {
            let bundle: VerificationBundle<worldforge_verify::ProvenanceArtifact> =
                serde_json::from_str(bundle_json).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid provenance bundle JSON: {e}"
                    ))
                })?;
            let verifier = bundle.proof.backend.verifier();
            serde_json::to_string_pretty(&verify_bundle(verifier.as_ref(), &bundle).map_err(
                |e| pyo3::exceptions::PyRuntimeError::new_err(format!("verification failed: {e}")),
            )?)
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
    let py = m.py();
    m.add("__path__", PyList::empty_bound(py))?;
    let sys = py.import_bound("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("worldforge", m)?;

    m.add_class::<PyPosition>()?;
    m.add_class::<PyRotation>()?;
    m.add_class::<PyBBox>()?;
    m.add_class::<PyVelocity>()?;
    m.add_class::<PySceneObject>()?;
    m.add_class::<PySceneObjectPatch>()?;
    m.add_class::<PyWorld>()?;
    m.add_class::<PyHistoryEntry>()?;
    m.add_class::<PyPrediction>()?;
    m.add_class::<PyObjectDiagnostic>()?;
    m.add_class::<PyObjectDrift>()?;
    m.add_class::<PyGuardrailDiagnostics>()?;
    m.add_class::<PyStateDiagnostics>()?;
    m.add_class::<PyPairwiseAgreement>()?;
    m.add_class::<PyComparisonConsensus>()?;
    m.add_class::<PyProviderScore>()?;
    m.add_class::<PyMultiPrediction>()?;
    m.add_class::<PyVideoClip>()?;
    m.add_class::<PyEmbeddingOutput>()?;
    m.add_class::<PyReasoningOutput>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyGuardrail>()?;
    m.add_class::<PyMockProvider>()?;
    m.add_class::<PyCosmosProvider>()?;
    m.add_class::<PyRunwayProvider>()?;
    m.add_class::<PyJepaProvider>()?;
    m.add_class::<PyGenieProvider>()?;
    m.add_class::<PyProviderCapabilities>()?;
    m.add_class::<PyProviderInfo>()?;
    m.add_class::<PyProviderHealthInfo>()?;
    m.add_class::<PyCostEstimate>()?;
    m.add_class::<PyWorldForge>()?;
    m.add_class::<PyPlan>()?;
    m.add_class::<PyPlanExecution>()?;
    m.add_class::<PyEvalScenario>()?;
    m.add_class::<PyEvalSuite>()?;
    m.add_class::<PyPhysicsEval>()?;
    m.add_class::<PyManipulationEval>()?;
    m.add_class::<PySpatialEval>()?;
    m.add_class::<PyComprehensiveEval>()?;
    m.add_class::<PyEvalResult>()?;
    m.add_class::<PyEvalDimensionSummary>()?;
    m.add_class::<PyEvalScenarioSummary>()?;
    m.add_class::<PyEvalReport>()?;
    m.add_class::<PyZkProof>()?;
    m.add_class::<PyVerificationResult>()?;
    m.add_class::<PyInferenceBundle>()?;
    m.add_class::<PyGuardrailBundle>()?;
    m.add_class::<PyProvenanceBundle>()?;
    m.add_class::<PyInferenceVerificationReport>()?;
    m.add_class::<PyGuardrailVerificationReport>()?;
    m.add_class::<PyProvenanceVerificationReport>()?;
    m.add_class::<PyZkVerifier>()?;
    m.add_class::<PyMockVerifier>()?;
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(list_eval_suites, m)?)?;
    m.add_function(wrap_pyfunction!(run_eval, m)?)?;
    m.add_function(wrap_pyfunction!(run_eval_report, m)?)?;
    m.add_function(wrap_pyfunction!(run_eval_report_data, m)?)?;
    m.add_function(wrap_pyfunction!(prove_inference, m)?)?;
    m.add_function(wrap_pyfunction!(prove_inference_transition, m)?)?;
    m.add_function(wrap_pyfunction!(prove_inference_transition_bundle_py, m)?)?;
    m.add_function(wrap_pyfunction!(prove_guardrail_plan, m)?)?;
    m.add_function(wrap_pyfunction!(prove_provenance, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof_json, m)?)?;
    m.add_function(wrap_pyfunction!(verify_bundle_json, m)?)?;

    let providers = build_providers_submodule(py)?;
    register_child_module(py, m, "providers", &providers)?;
    let eval = build_eval_submodule(py)?;
    register_child_module(py, m, "eval", &eval)?;
    let verify = build_verify_submodule(py)?;
    register_child_module(py, m, "verify", &verify)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_worldforge() -> PyWorldForge {
        PyWorldForge::new("file", ".worldforge-python-tests", None, "json", None).unwrap()
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
    fn test_scene_object_set_position_keeps_bbox_coherent() {
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        let mut obj = PySceneObject::new("crate", &pos, &bbox);

        let new_pos = PyPosition::new(2.0, 3.0, 4.0);
        obj.set_position(&new_pos);
        let updated_bbox = obj.bbox();

        assert_eq!(obj.position().x(), 2.0);
        assert_eq!(obj.position().y(), 3.0);
        assert_eq!(obj.position().z(), 4.0);
        assert_eq!(updated_bbox.min().x(), 1.0);
        assert_eq!(updated_bbox.min().y(), 2.0);
        assert_eq!(updated_bbox.min().z(), 3.0);
        assert_eq!(updated_bbox.max().x(), 3.0);
        assert_eq!(updated_bbox.max().y(), 4.0);
        assert_eq!(updated_bbox.max().z(), 5.0);
    }

    #[test]
    fn test_scene_object_json_roundtrip() {
        let pos = PyPosition::new(0.0, 1.0, 0.0);
        let min = PyPosition::new(-0.5, 0.5, -0.5);
        let max = PyPosition::new(0.5, 1.5, 0.5);
        let bbox = PyBBox::new(&min, &max);
        let mut obj = PySceneObject::new("crate", &pos, &bbox);
        obj.set_semantic_label(Some("storage".to_string()));
        obj.set_mass(5.0);

        let json = obj.to_json().unwrap();
        let restored = PySceneObject::from_json(&json).unwrap();

        assert_eq!(restored.name(), "crate");
        assert_eq!(restored.semantic_label(), Some("storage"));
        assert_eq!(restored.position().y(), 1.0);
    }

    #[test]
    fn test_scene_object_patch_json_roundtrip() {
        let mut patch = PySceneObjectPatch::new();
        patch.set_name(Some("crate".to_string()));
        patch.set_position(Some(&PyPosition::new(2.0, 3.0, 4.0)));
        patch.set_mass(Some(2.5));
        patch.set_static(Some(true));

        let json = patch.to_json().unwrap();
        let restored = PySceneObjectPatch::from_json(&json).unwrap();

        let mut world = PyWorld::new("test", "mock");
        let origin = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        let object = PySceneObject::new("cube", &origin, &bbox);
        let object_id = object.id();

        world.add_object(&object).unwrap();
        let updated = world.update_object_patch(&object_id, &restored).unwrap();

        assert_eq!(updated.name(), "crate");
        assert_eq!(updated.position().x(), 2.0);
        assert_eq!(updated.position().y(), 3.0);
        assert_eq!(updated.position().z(), 4.0);
        assert_eq!(updated.mass(), Some(2.5));
        assert!(updated.is_static());
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

        world.add_object(&obj).unwrap();
        assert_eq!(world.object_count(), 1);
        assert!(world.get_object("cube").is_some());
        assert!(world.get_object("nonexistent").is_none());
    }

    #[test]
    fn test_world_add_object_rejects_duplicate_id() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        let original = PySceneObject::new("cube", &pos, &bbox);
        let duplicate = PySceneObject::from_json(&original.to_json().unwrap()).unwrap();

        world.add_object(&original).unwrap();
        let err = world.add_object(&duplicate).unwrap_err();
        assert!(err.to_string().contains("object already exists"));
        assert_eq!(world.object_count(), 1);
        assert_eq!(world.world.state.scene.root.children.len(), 1);
    }

    #[test]
    fn test_world_update_object_applies_mutations() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        world
            .add_object(&PySceneObject::new("cube", &pos, &bbox))
            .unwrap();

        let mut obj = world.get_object("cube").unwrap();
        let new_pos = PyPosition::new(2.0, 3.0, 4.0);
        obj.set_name("crate".to_string());
        obj.set_position(&new_pos);
        obj.set_rotation(&PyRotation::new(0.0, 1.0, 0.0, 0.0));
        obj.set_mass(2.5);
        obj.set_friction(Some(0.25));
        obj.set_restitution(Some(0.75));
        obj.set_graspable(true);
        obj.set_material(Some("wood".to_string()));
        world.update_object(&obj).unwrap();

        assert!(world.get_object("cube").is_none());
        let updated = world.get_object("crate").unwrap();
        assert_eq!(updated.name(), "crate");
        assert_eq!(updated.position().x(), 2.0);
        assert_eq!(updated.position().y(), 3.0);
        assert_eq!(updated.position().z(), 4.0);
        assert_eq!(updated.bbox().min().x(), 1.0);
        assert_eq!(updated.bbox().max().z(), 5.0);
        assert_eq!(updated.mass(), Some(2.5));
        assert_eq!(updated.friction(), Some(0.25));
        assert_eq!(updated.restitution(), Some(0.75));
        assert!(updated.is_graspable());
        assert_eq!(updated.material(), Some("wood"));
        assert_eq!(updated.rotation().x(), 1.0);
        assert_eq!(updated.id(), obj.id());
    }

    #[test]
    fn test_world_update_object_clears_optional_fields() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        let mut object = PySceneObject::new("ball", &pos, &bbox);
        object.set_semantic_label(Some("crate".to_string()));
        world.add_object(&object).unwrap();

        object = world.get_object("ball").unwrap();
        object.set_semantic_label(None);
        world.update_object(&object).unwrap();

        let updated = world.get_object("ball").unwrap();
        assert_eq!(updated.semantic_label(), None);
    }

    #[test]
    fn test_world_update_object_missing_returns_error() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        let obj = PySceneObject::new("ghost", &pos, &bbox);

        let err = world.update_object(&obj).unwrap_err();
        assert!(err.to_string().contains("object not found"));
    }

    #[test]
    fn test_world_list_objects() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);

        world
            .add_object(&PySceneObject::new("a", &pos, &bbox))
            .unwrap();
        world
            .add_object(&PySceneObject::new("b", &pos, &bbox))
            .unwrap();

        let names = world.list_objects();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }

    #[test]
    fn test_world_objects_returns_sorted_scene_objects() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);

        world
            .add_object(&PySceneObject::new("b", &pos, &bbox))
            .unwrap();
        world
            .add_object(&PySceneObject::new("a", &pos, &bbox))
            .unwrap();

        let objects = world.objects();
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].name(), "a");
        assert_eq!(objects[1].name(), "b");
    }

    #[test]
    fn test_world_object_id_access_and_patch_flow() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        let object = PySceneObject::new("cube", &pos, &bbox);
        let object_id = object.id();

        world.add_object(&object).unwrap();

        let fetched = world.get_object_by_id(&object_id).unwrap().unwrap();
        assert_eq!(fetched.id(), object_id);
        assert_eq!(fetched.name(), "cube");

        let mut patch = PySceneObjectPatch::new();
        patch.set_name(Some("crate".to_string()));
        patch.set_position(Some(&PyPosition::new(2.0, 0.5, -1.0)));
        patch.set_velocity(Some(&PyVelocity::new(0.0, 0.1, 0.0)));
        patch.set_material(Some("wood".to_string()));
        patch.set_graspable(Some(true));

        let updated = world.update_object_patch(&object_id, &patch).unwrap();
        assert_eq!(updated.id(), object_id);
        assert_eq!(updated.name(), "crate");
        assert_eq!(updated.position().x(), 2.0);
        assert_eq!(updated.position().y(), 0.5);
        assert_eq!(updated.position().z(), -1.0);
        assert_eq!(updated.velocity().y(), 0.1);
        assert_eq!(updated.material(), Some("wood"));
        assert!(updated.is_graspable());

        let removed = world.remove_object_by_id(&object_id).unwrap().unwrap();
        assert_eq!(removed.id(), object_id);
        assert_eq!(world.object_count(), 0);
        assert!(world.get_object_by_id(&object_id).unwrap().is_none());
        assert!(world.remove_object_by_id(&object_id).unwrap().is_none());
    }

    #[test]
    fn test_world_update_object_patch_missing_returns_error() {
        let mut world = PyWorld::new("test", "mock");
        let mut patch = PySceneObjectPatch::new();
        patch.set_name(Some("ghost".to_string()));

        let missing_id = uuid::Uuid::new_v4().to_string();
        let err = world.update_object_patch(&missing_id, &patch).unwrap_err();
        assert!(err.to_string().contains("object not found"));
    }

    #[test]
    fn test_world_get_object_by_id_rejects_invalid_uuid() {
        let world = PyWorld::new("test", "mock");
        let err = world.get_object_by_id("not-a-uuid").unwrap_err();
        assert!(err.to_string().contains("invalid object ID"));
    }

    #[test]
    fn test_world_remove_object() {
        let mut world = PyWorld::new("test", "mock");
        let pos = PyPosition::new(0.0, 0.0, 0.0);
        let min = PyPosition::new(-1.0, -1.0, -1.0);
        let max = PyPosition::new(1.0, 1.0, 1.0);
        let bbox = PyBBox::new(&min, &max);
        world
            .add_object(&PySceneObject::new("cube", &pos, &bbox))
            .unwrap();

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
        world
            .add_object(&PySceneObject::new("ball", &pos, &bbox))
            .unwrap();

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
                false,
            )
            .unwrap();

        assert_eq!(prediction.provider(), "mock");
        assert_eq!(world.step(), 1);
        assert_eq!(world.history_length(), 2);
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
                false,
            )
            .unwrap();

        assert_eq!(prediction.provider(), "mock");
        assert_eq!(world.step(), 1);
    }

    #[test]
    fn test_world_history_exposes_initial_and_transition_entries() {
        let mut world = PyWorld::new("history_world", "mock");
        world
            .predict(
                &PyAction::move_to(1.0, 0.0, 0.0, 1.0),
                1,
                None,
                None,
                false,
                None,
                false,
            )
            .unwrap();

        let history = world.history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].step(), 0);
        assert!(history[0].action_json().unwrap().is_none());
        assert_eq!(history[1].provider(), "mock");
        assert!(history[1].action_json().unwrap().is_some());
        assert!(history[1].confidence().is_some());
    }

    #[test]
    fn test_world_history_state_and_restore_rewind_checkpoint() {
        let mut world = PyWorld::new("history_restore_world", "mock");
        world
            .predict(
                &PyAction::move_to(1.0, 0.0, 0.0, 1.0),
                1,
                None,
                None,
                false,
                None,
                false,
            )
            .unwrap();

        let checkpoint = world.history_state(0).unwrap();
        assert_eq!(checkpoint.step(), 0);
        assert_eq!(checkpoint.history_length(), 1);

        world.restore_history(0).unwrap();
        assert_eq!(world.step(), 0);
        assert_eq!(world.history_length(), 1);
    }

    #[test]
    fn test_world_compare_returns_multi_prediction() {
        let mut world = PyWorld::new("compare_world", "mock");
        world
            .add_object(&PySceneObject::new(
                "mug",
                &PyPosition::new(0.0, 0.0, 0.0),
                &PyBBox::new(
                    &PyPosition::new(-0.1, -0.1, -0.1),
                    &PyPosition::new(0.1, 0.1, 0.1),
                ),
            ))
            .unwrap();

        let comparison = world
            .compare(
                &PyAction::move_to(1.0, 0.0, 0.0, 1.0),
                vec!["mock".to_string(), "mock".to_string()],
                1,
                None,
                false,
                None,
                None,
                false,
            )
            .unwrap();

        assert_eq!(comparison.prediction_count(), 2);
        assert_eq!(comparison.best_prediction().provider(), "mock");
        assert!(comparison.summary().contains("Compared 2 providers"));
        assert_eq!(comparison.provider_scores().len(), 2);
        assert_eq!(comparison.pairwise_agreements().len(), 1);
        assert!(comparison.provider_scores()[0].quality_score() > 0.0);
        assert!(
            comparison.provider_scores()[0]
                .state()
                .relationship_preservation_rate()
                >= 0.0
        );
        assert!(
            comparison.consensus().shared_object_count()
                <= comparison.provider_scores()[0]
                    .state()
                    .output_object_count()
        );
        assert!(
            comparison.provider_scores()[0]
                .state()
                .output_object_count()
                >= 1
        );
        assert_eq!(
            comparison.provider_scores()[0]
                .guardrails()
                .evaluated_count(),
            2
        );
    }

    #[test]
    fn test_world_compare_rejects_empty_provider_list() {
        let world = PyWorld::new("compare_world", "mock");
        let result = world.compare(
            &PyAction::move_to(0.0, 0.0, 0.0, 1.0),
            Vec::new(),
            1,
            None,
            false,
            None,
            None,
            false,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_world_compare_uses_fallback_provider() {
        let world = PyWorld::new("compare_world", "mock");
        let comparison = world
            .compare(
                &PyAction::move_to(0.5, 0.0, 0.0, 1.0),
                vec!["missing".to_string()],
                1,
                Some("mock"),
                false,
                None,
                None,
                false,
            )
            .unwrap();

        assert_eq!(comparison.prediction_count(), 1);
        assert_eq!(comparison.best_prediction().provider(), "mock");
    }

    #[test]
    fn test_world_compare_applies_guardrails() {
        let mut world = PyWorld::new("compare_world", "mock");
        world
            .add_object(&PySceneObject::new(
                "cube",
                &PyPosition::new(0.0, 0.0, 0.0),
                &PyBBox::new(
                    &PyPosition::new(-0.1, -0.1, -0.1),
                    &PyPosition::new(0.1, 0.1, 0.1),
                ),
            ))
            .unwrap();

        let result = world.compare(
            &PyAction::move_to(1.0, 0.0, 0.0, 1.0),
            vec!["mock".to_string()],
            1,
            None,
            false,
            None,
            Some(
                r#"[{"guardrail":{"BoundaryConstraint":{"bounds":{"min":{"x":-0.25,"y":-0.25,"z":-0.25},"max":{"x":0.25,"y":0.25,"z":0.25}}}},"blocking":true}]"#,
            ),
            false,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("guardrail"));
    }

    #[test]
    fn test_world_predict_disable_guardrails_skips_defaults() {
        let mut world = PyWorld::new("predict_guardrails", "mock");
        world
            .add_object(&PySceneObject::new(
                "left",
                &PyPosition::new(0.0, 0.0, 0.0),
                &PyBBox::new(
                    &PyPosition::new(0.0, 0.0, 0.0),
                    &PyPosition::new(1.0, 1.0, 1.0),
                ),
            ))
            .unwrap();
        world
            .add_object(&PySceneObject::new(
                "right",
                &PyPosition::new(0.0, 0.0, 0.0),
                &PyBBox::new(
                    &PyPosition::new(0.5, 0.5, 0.5),
                    &PyPosition::new(1.5, 1.5, 1.5),
                ),
            ))
            .unwrap();

        let default_result = world.predict(
            &PyAction::set_weather("rain").unwrap(),
            1,
            None,
            None,
            false,
            None,
            false,
        );
        assert!(default_result.is_err());

        let prediction = world
            .predict(
                &PyAction::set_weather("rain").unwrap(),
                1,
                None,
                None,
                false,
                None,
                true,
            )
            .unwrap();

        assert_eq!(prediction.guardrail_count(), 0);
    }

    #[test]
    fn test_world_reason() {
        let world = PyWorld::new("reason_world", "mock");
        let output = world.reason("will it fall?", None, None).unwrap();
        assert!(output.answer().contains("empty"));
        assert!(output.confidence() > 0.0);
        assert_eq!(output.evidence(), vec!["objects: none".to_string()]);
    }

    #[test]
    fn test_world_reason_uses_fallback_provider() {
        let world = PyWorld::new("reason_world", "missing");
        let output = world.reason("will it fall?", None, Some("mock")).unwrap();
        assert!(output.answer().contains("empty"));
        assert_eq!(output.evidence(), vec!["objects: none".to_string()]);
    }

    #[test]
    fn test_worldforge_reason_uses_world_json_and_video() {
        let wf = test_worldforge();
        let mut world = PyWorld::new("reason_world", "mock");
        let position = PyPosition::new(0.0, 0.5, 0.0);
        let bbox = PyBBox::new(
            &PyPosition::new(-0.1, 0.4, -0.1),
            &PyPosition::new(0.1, 0.6, 0.1),
        );
        world
            .add_object(&PySceneObject::new("mug", &position, &bbox))
            .unwrap();
        let world_json = world.to_json().unwrap();
        let clip = PyVideoClip {
            inner: VideoClip {
                frames: Vec::new(),
                fps: 8.0,
                resolution: (64, 64),
                duration: 1.0,
            },
        };

        let output = wf
            .reason(
                "mock",
                "how many objects are here?",
                None,
                Some(&world_json),
                Some(&clip),
                None,
            )
            .unwrap();

        assert_eq!(output.answer(), "I can account for 1 object(s): mug.");
        assert!(output.evidence().iter().any(|entry| entry.contains("mug")));
    }

    #[test]
    fn test_worldforge_reason_video_only_echoes_query() {
        let wf = test_worldforge();
        let clip = PyVideoClip {
            inner: VideoClip {
                frames: Vec::new(),
                fps: 8.0,
                resolution: (64, 64),
                duration: 1.0,
            },
        };

        let output = wf
            .reason("mock", "what do you see?", None, None, Some(&clip), None)
            .unwrap();

        assert!(output.answer().contains("echo the query"));
        assert_eq!(output.evidence(), vec!["state: unavailable".to_string()]);
    }

    #[test]
    fn test_worldforge_reason_rejects_missing_inputs() {
        let wf = test_worldforge();
        let error = wf
            .reason("mock", "what happens?", None, None, None, None)
            .unwrap_err();

        assert!(format!("{error}").contains("reason requires world, world_json, or video"));
    }

    #[test]
    fn test_genie_provider_construction_accepts_missing_api_key() {
        let provider = PyGenieProvider::new(None, "genie3", None).unwrap();
        assert_eq!(provider.name(), "genie");
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
        assert!(providers.contains(&"genie".to_string()));
    }

    #[test]
    fn test_worldforge_genie_surrogate_is_available_without_credentials() {
        let wf = test_worldforge();
        let health = wf.provider_health("genie").unwrap();
        assert!(health.healthy());

        let clip = PyVideoClip {
            inner: VideoClip {
                frames: Vec::new(),
                fps: 8.0,
                resolution: (64, 64),
                duration: 1.0,
            },
        };
        let output = wf
            .reason("genie", "what do you see?", None, None, Some(&clip), None)
            .unwrap();

        assert!(output.answer().contains("frame(s)"));
        assert!(output
            .evidence()
            .iter()
            .any(|entry| entry.contains("frames:")));
    }

    #[test]
    fn test_worldforge_package_submodules_and_manual_provider_registration() {
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::new_bound(py, "worldforge")?;
            worldforge(&module)?;

            let root = py.import_bound("worldforge")?;
            let providers = py.import_bound("worldforge.providers")?;
            let eval = py.import_bound("worldforge.eval")?;
            let verify = py.import_bound("worldforge.verify")?;

            assert!(root.hasattr("providers")?);
            assert!(root.hasattr("eval")?);
            assert!(root.hasattr("verify")?);
            assert!(providers.hasattr("MockProvider")?);
            assert!(providers.hasattr("CosmosProvider")?);
            assert!(providers.hasattr("RunwayProvider")?);
            assert!(providers.hasattr("JepaProvider")?);
            assert!(providers.hasattr("GenieProvider")?);
            assert!(eval.hasattr("run_eval")?);
            assert!(eval.hasattr("run_eval_report_data")?);
            assert!(verify.hasattr("prove_inference")?);
            assert!(verify.hasattr("verify_bundle_json")?);

            let wf_cls = root.getattr("WorldForge")?;
            let wf = wf_cls.call0()?;
            let provider_cls = providers.getattr("MockProvider")?;
            let provider = provider_cls.call1(("manual-mock",))?;
            wf.call_method1("register_provider", (provider,))?;

            let provider_names: Vec<String> = wf.call_method0("providers")?.extract()?;
            assert!(provider_names.contains(&"manual-mock".to_string()));

            let world = wf.call_method1("create_world", ("python-world", "manual-mock"))?;
            assert!(!world.is_none());

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_worldforge_register_provider_rejects_after_world_creation() {
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::new_bound(py, "worldforge")?;
            worldforge(&module)?;

            let root = py.import_bound("worldforge")?;
            let providers = py.import_bound("worldforge.providers")?;
            let wf_cls = root.getattr("WorldForge")?;
            let wf = wf_cls.call0()?;
            let world = wf.call_method1("create_world", ("python-world", "mock"))?;

            let provider_cls = providers.getattr("MockProvider")?;
            let provider = provider_cls.call1(("late-provider",))?;
            let err = wf
                .call_method1("register_provider", (provider,))
                .unwrap_err();
            assert!(err.to_string().contains("failed to register provider"));
            assert!(err.to_string().contains("cannot register provider"));
            assert!(!world.is_none());

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_worldforge_create_world_from_prompt_keeps_seeded_history() {
        let wf = test_worldforge();
        let world = wf
            .create_world_from_prompt("A kitchen with a mug", "mock", Some("seeded-kitchen"))
            .unwrap();

        assert_eq!(world.name(), "seeded-kitchen");
        assert_eq!(world.description(), "A kitchen with a mug");
        assert!(world.object_count() >= 2);
        assert_eq!(world.history_length(), 1);
    }

    #[test]
    fn test_worldforge_provider_info() {
        let wf = test_worldforge();
        let info = wf.provider_info("mock").unwrap();
        assert_eq!(info.name(), "mock");
        assert!(info.capabilities().predict());
        assert!(info.capabilities().embed());
    }

    #[test]
    fn test_worldforge_provider_infos_with_filter() {
        let wf = test_worldforge();
        let infos = wf.provider_infos(Some("predict"));
        assert!(!infos.is_empty());
        assert!(infos.iter().any(|info| info.name() == "mock"));
    }

    #[test]
    fn test_worldforge_provider_infos_with_embed_filter() {
        let wf = test_worldforge();
        let infos = wf.provider_infos(Some("embed"));
        assert!(!infos.is_empty());
        assert!(infos.iter().any(|info| info.name() == "mock"));
    }

    #[test]
    fn test_worldforge_provider_health() {
        let wf = test_worldforge();
        let health = wf.provider_health("mock").unwrap();
        assert_eq!(health.name(), "mock");
        assert!(health.healthy());
        assert_eq!(
            health.message().as_deref(),
            Some("mock provider is always healthy")
        );
        assert!(health.error().is_none());
    }

    #[test]
    fn test_worldforge_provider_healths_with_filter() {
        let wf = test_worldforge();
        let healths = wf.provider_healths(Some("planning")).unwrap();
        assert!(!healths.is_empty());
        assert!(healths.iter().any(|info| info.name() == "mock"));
        assert!(healths.iter().all(|info| info.healthy()));
    }

    #[test]
    fn test_worldforge_estimate_cost() {
        let wf = test_worldforge();
        let estimate = wf
            .estimate_cost("mock", "generate", 1, 5.0, 640, 360)
            .unwrap();
        assert_eq!(estimate.usd(), 0.0);
        assert_eq!(estimate.estimated_latency_ms(), 10);
    }

    #[test]
    fn test_worldforge_create_world() {
        let wf = test_worldforge();
        let world = wf.create_world("test_world", "mock").unwrap();
        assert_eq!(world.name(), "test_world");
        assert_eq!(world.description(), "");
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
    fn test_worldforge_compare_predictions() {
        let wf = test_worldforge();
        let mut world_a = wf.create_world("compare-a", "mock").unwrap();
        let mut world_b = wf.create_world("compare-b", "mock").unwrap();
        let action = PyAction::move_to(1.0, 0.0, 0.0, 1.0);
        let prediction_a = world_a
            .predict(&action, 1, None, None, false, None, false)
            .unwrap();
        let prediction_b = world_b
            .predict(&action, 1, None, None, false, None, false)
            .unwrap();

        let comparison = wf.compare(vec![prediction_a, prediction_b]).unwrap();

        assert_eq!(comparison.prediction_count(), 2);
        assert!(comparison.best_prediction_index() < comparison.prediction_count());
        assert_eq!(comparison.best_prediction().provider(), "mock");
        assert_eq!(comparison.provider_scores()[0].provider(), "mock");
        assert_eq!(comparison.pairwise_agreements().len(), 1);
        assert!(comparison.consensus().average_quality_score() > 0.0);
    }

    #[test]
    fn test_multi_prediction_json_roundtrip() {
        let wf = test_worldforge();
        let mut world_a = wf.create_world("compare-a", "mock").unwrap();
        let mut world_b = wf.create_world("compare-b", "mock").unwrap();
        let action = PyAction::move_to(1.0, 0.0, 0.0, 1.0);
        let prediction_a = world_a
            .predict(&action, 1, None, None, false, None, false)
            .unwrap();
        let prediction_b = world_b
            .predict(&action, 1, None, None, false, None, false)
            .unwrap();
        let comparison = wf.compare(vec![prediction_a, prediction_b]).unwrap();

        let json = comparison.to_json().unwrap();
        let restored = PyMultiPrediction::from_json(&json).unwrap();

        assert_eq!(restored.prediction_count(), 2);
        assert_eq!(restored.best_prediction().provider(), "mock");
        assert_eq!(restored.pairwise_agreements().len(), 1);
        assert_eq!(restored.provider_scores().len(), 2);
        assert!(
            restored.consensus().shared_object_count()
                <= restored.provider_scores()[0].state().output_object_count()
        );
        assert!(restored.__repr__().contains("MultiPrediction"));
    }

    #[test]
    fn test_worldforge_compare_rejects_empty_predictions() {
        let wf = test_worldforge();
        let result = wf.compare(Vec::new());

        assert!(result.is_err());
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
                None,
            )
            .unwrap();

        assert_eq!(clip.duration(), 5.0);
        assert_eq!(clip.resolution(), (640, 360));
        assert_eq!(clip.fps(), 12.0);
        assert!(clip.frame_count() > 0);
        assert!(clip.__repr__().contains("VideoClip"));
    }

    #[test]
    fn test_worldforge_embed() {
        let wf = test_worldforge();
        let clip = PyVideoClip {
            inner: VideoClip {
                frames: Vec::new(),
                fps: 8.0,
                resolution: (64, 64),
                duration: 1.0,
            },
        };

        let output = wf
            .embed("mock", Some("a red mug on a table"), Some(&clip), None)
            .unwrap();

        assert_eq!(output.provider(), "mock");
        assert_eq!(output.model(), "mock-embedding-v1");
        assert_eq!(output.shape(), vec![32]);
        assert_eq!(output.vector().len(), 32);
        assert!(output.__repr__().contains("EmbeddingOutput"));
    }

    #[test]
    fn test_embedding_output_json_roundtrip() {
        let wf = test_worldforge();
        let clip = PyVideoClip {
            inner: VideoClip {
                frames: Vec::new(),
                fps: 8.0,
                resolution: (64, 64),
                duration: 1.0,
            },
        };

        let output = wf
            .embed("mock", Some("text-only"), Some(&clip), None)
            .unwrap();
        let json = output.to_json().unwrap();
        let restored = PyEmbeddingOutput::from_json(&json).unwrap();

        assert_eq!(restored.provider(), "mock");
        assert_eq!(restored.model(), "mock-embedding-v1");
        assert_eq!(restored.shape(), vec![32]);
        assert_eq!(restored.vector().len(), 32);
    }

    #[test]
    fn test_tensor_data_to_f32_vec_supports_half_precision() {
        let half_values = TensorData::Float16(vec![0x3c00, 0xc000, 0x3800]);
        let bfloat16_values = TensorData::BFloat16(vec![0x3f80, 0xc000, 0x3f00]);

        assert_eq!(tensor_data_to_f32_vec(&half_values), vec![1.0, -2.0, 0.5]);
        assert_eq!(
            tensor_data_to_f32_vec(&bfloat16_values),
            vec![1.0, -2.0, 0.5]
        );
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
                None,
            )
            .unwrap();
        let transferred = wf
            .transfer(&clip, "mock", None, 640, 360, 24.0, 0.5, None)
            .unwrap();

        assert_eq!(transferred.duration(), clip.duration());
        assert_eq!(transferred.resolution(), (640, 360));
        assert_eq!(transferred.fps(), 24.0);
    }

    #[test]
    fn test_worldforge_embed_uses_fallback_provider() {
        let wf = test_worldforge();
        let clip = PyVideoClip {
            inner: VideoClip {
                frames: Vec::new(),
                fps: 8.0,
                resolution: (64, 64),
                duration: 1.0,
            },
        };

        let output = wf
            .embed(
                "missing",
                Some("a red mug on a table"),
                Some(&clip),
                Some("mock"),
            )
            .unwrap();

        assert_eq!(output.provider(), "mock");
        assert_eq!(output.model(), "mock-embedding-v1");
    }

    #[test]
    fn test_worldforge_file_store_roundtrip() {
        let state_dir =
            std::env::temp_dir().join(format!("wf-python-file-{}", uuid::Uuid::new_v4()));
        let wf =
            PyWorldForge::new("file", state_dir.to_str().unwrap(), None, "json", None).unwrap();
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
            "json",
            None,
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

    #[test]
    fn test_worldforge_msgpack_file_store_roundtrip() {
        let state_dir =
            std::env::temp_dir().join(format!("wf-python-msgpack-{}", uuid::Uuid::new_v4()));
        let wf =
            PyWorldForge::new("file", state_dir.to_str().unwrap(), None, "msgpack", None).unwrap();
        let world = wf.create_world("msgpack_world", "mock").unwrap();
        let world_id = wf.save_world(&world).unwrap();

        assert!(state_dir.join(format!("{world_id}.msgpack")).exists());

        let loaded = wf.load_world(&world_id).unwrap();
        assert_eq!(loaded.name(), "msgpack_world");
        assert_eq!(wf.list_worlds().unwrap(), vec![world_id.clone()]);

        wf.delete_world(&world_id).unwrap();
        assert!(wf.list_worlds().unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&state_dir);
    }

    #[test]
    fn test_state_store_kind_accepts_redis_backend() {
        let kind = state_store_kind(
            "redis",
            ".worldforge",
            None,
            "json",
            Some("redis://127.0.0.1:6379/0"),
        )
        .unwrap();

        assert_eq!(
            kind,
            StateStoreKind::Redis("redis://127.0.0.1:6379/0".to_string())
        );
    }

    #[test]
    fn test_state_store_kind_requires_redis_url() {
        let error = state_store_kind("redis", ".worldforge", None, "json", None).unwrap_err();
        assert!(error.to_string().contains("state_redis_url is required"));
    }

    #[test]
    fn test_worldforge_export_import_json_roundtrip() {
        let state_dir =
            std::env::temp_dir().join(format!("wf-python-export-json-{}", uuid::Uuid::new_v4()));
        let wf =
            PyWorldForge::new("file", state_dir.to_str().unwrap(), None, "json", None).unwrap();
        let mut world = wf.create_world("snapshot_world", "mock").unwrap();
        let position = PyPosition::new(0.0, 0.8, 0.0);
        let bbox = PyBBox::new(
            &PyPosition::new(-0.05, 0.75, -0.05),
            &PyPosition::new(0.05, 0.85, 0.05),
        );
        world
            .add_object(&PySceneObject::new("mug", &position, &bbox))
            .unwrap();
        let world_id = wf.save_world(&world).unwrap();

        pyo3::Python::with_gil(|py| {
            let snapshot = wf.export_world(py, &world_id, "json").unwrap();
            let json = snapshot.bind(py).extract::<String>().unwrap();
            assert!(json.contains("\"snapshot_world\""));
            assert!(json.contains("\"mug\""));

            let snapshot = pyo3::types::PyString::new_bound(py, &json).into_any();
            let imported = wf
                .import_world(&snapshot, "json", true, Some("snapshot_world_copy"))
                .unwrap();

            assert_eq!(imported.name(), "snapshot_world_copy");
            assert_ne!(imported.id(), world_id);
            assert_eq!(imported.object_count(), 1);
        });

        let _ = std::fs::remove_dir_all(&state_dir);
    }

    #[test]
    fn test_worldforge_export_import_msgpack_roundtrip() {
        let state_dir =
            std::env::temp_dir().join(format!("wf-python-export-msgpack-{}", uuid::Uuid::new_v4()));
        let wf =
            PyWorldForge::new("file", state_dir.to_str().unwrap(), None, "msgpack", None).unwrap();
        let mut world = wf.create_world("msgpack_snapshot_world", "mock").unwrap();
        let position = PyPosition::new(0.0, 0.8, 0.0);
        let bbox = PyBBox::new(
            &PyPosition::new(-0.05, 0.75, -0.05),
            &PyPosition::new(0.05, 0.85, 0.05),
        );
        world
            .add_object(&PySceneObject::new("cube", &position, &bbox))
            .unwrap();
        let world_id = wf.save_world(&world).unwrap();

        pyo3::Python::with_gil(|py| {
            let snapshot = wf.export_world(py, &world_id, "msgpack").unwrap();
            let bytes = snapshot.bind(py).extract::<Vec<u8>>().unwrap();
            assert!(!bytes.is_empty());

            let snapshot = pyo3::types::PyBytes::new_bound(py, &bytes).into_any();
            let imported = wf.import_world(&snapshot, "msgpack", false, None).unwrap();

            assert_eq!(imported.id(), world_id);
            assert_eq!(imported.name(), "msgpack_snapshot_world");
            assert_eq!(imported.object_count(), 1);
        });

        let _ = std::fs::remove_dir_all(&state_dir);
    }

    // --- Planning tests ---

    #[test]
    fn test_plan_world() {
        let world = PyWorld::new("plan_test", "mock");
        let plan = world
            .plan(
                Some("move forward"),
                None,
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
                false,
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
            Some("reach goal"),
            None,
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
            false,
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
            &world,
            Some("go"),
            None,
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
            false,
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
                Some("spawn cube"),
                None,
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
                false,
            )
            .unwrap();

        assert!(plan.action_count() > 0);
        assert_eq!(plan.iterations_used(), 3);
    }

    #[test]
    fn test_plan_world_with_provider_native() {
        let wf = test_worldforge();
        let supports_native = wf
            .provider_info("mock")
            .unwrap()
            .capabilities()
            .supports_planning();
        let world = PyWorld::new("plan_native", "mock");
        let result = world.plan(
            Some("spawn cube"),
            None,
            4,
            10.0,
            Some("mock"),
            "provider-native",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        );

        if supports_native {
            let plan = result.unwrap();
            assert!(plan.action_count() > 0);
            assert!(plan.iterations_used() > 0);
            assert!((0.0..=1.0).contains(&plan.success_probability()));
            return;
        }

        let error = result.unwrap_err().to_string().to_lowercase();
        assert!(error.contains("native planning") || error.contains("unsupported"));
    }

    #[test]
    fn test_plan_world_with_guardrails_json() {
        let world = PyWorld::new("plan_guardrails", "mock");
        let guardrails_json = r#"[{"guardrail":"NoCollisions","blocking":true}]"#;
        let plan = world
            .plan(
                Some("stay collision free"),
                None,
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
                false,
            )
            .unwrap();

        assert!(plan.action_count() > 0);
    }

    #[test]
    fn test_plan_world_with_structured_goal_json() {
        let mut world = PyWorld::new("plan_structured", "mock");
        let position = PyPosition::new(0.0, 0.5, 0.0);
        let bbox = PyBBox::new(
            &PyPosition::new(-0.1, 0.4, -0.1),
            &PyPosition::new(0.1, 0.6, 0.1),
        );
        let object = PySceneObject::new("ball", &position, &bbox);
        let object_id = object.id();
        world.add_object(&object).unwrap();

        let goal_json = serde_json::json!({
            "type": "condition",
            "condition": {
                "ObjectAt": {
                    "object": object_id,
                    "position": {"x": 1.0, "y": 0.5, "z": 0.0},
                    "tolerance": 0.05
                }
            }
        })
        .to_string();

        let plan = world
            .plan(
                None,
                Some(&goal_json),
                4,
                10.0,
                Some("mock"),
                "sampling",
                Some(48),
                Some(5),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        assert!(plan.action_count() > 0);
        let plan_json: serde_json::Value = serde_json::from_str(&plan.to_json().unwrap()).unwrap();
        let final_state = plan_json["predicted_states"]
            .as_array()
            .and_then(|states| states.last())
            .expect("structured goal plan should include predicted states");
        let ball = final_state["scene"]["objects"]
            .as_object()
            .unwrap()
            .values()
            .find(|object| object["name"] == "ball")
            .expect("ball should still exist");
        let x = ball["pose"]["position"]["x"].as_f64().unwrap();
        let y = ball["pose"]["position"]["y"].as_f64().unwrap();
        let z = ball["pose"]["position"]["z"].as_f64().unwrap();
        assert!((x - 1.0).abs() <= 0.15);
        assert!((y - 0.5).abs() <= 0.15);
        assert!(z.abs() <= 0.15);
    }

    #[test]
    fn test_plan_world_with_goal_image_json() {
        let mut world = PyWorld::new("plan_goal_image", "mock");
        let position = PyPosition::new(0.0, 0.5, 0.0);
        let bbox = PyBBox::new(
            &PyPosition::new(-0.1, 0.4, -0.1),
            &PyPosition::new(0.1, 0.6, 0.1),
        );
        let object = PySceneObject::new("ball", &position, &bbox);
        world.add_object(&object).unwrap();

        let mut target_state = WorldState::new("plan_goal_image_target", "mock");
        let target_object = SceneObject::new(
            "ball",
            worldforge_core::types::Pose {
                position: Position {
                    x: 0.0,
                    y: 0.5,
                    z: 0.0,
                },
                ..worldforge_core::types::Pose::default()
            },
            BBox {
                min: Position {
                    x: -0.1,
                    y: 0.4,
                    z: -0.1,
                },
                max: Position {
                    x: 0.1,
                    y: 0.6,
                    z: 0.1,
                },
            },
        );
        let target_object_id = target_object.id;
        target_state.scene.add_object(target_object);
        target_state
            .scene
            .get_object_mut(&target_object_id)
            .unwrap()
            .set_position(Position {
                x: 1.0,
                y: 0.5,
                z: 0.0,
            });

        let goal_json = serde_json::json!({
            "type": "goal_image",
            "image": worldforge_core::goal_image::render_scene_goal_image(&target_state, (32, 24))
        })
        .to_string();

        let plan = world
            .plan(
                None,
                Some(&goal_json),
                4,
                10.0,
                Some("mock"),
                "sampling",
                Some(48),
                Some(5),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();

        assert!(plan.action_count() > 0);
        let plan_json: serde_json::Value = serde_json::from_str(&plan.to_json().unwrap()).unwrap();
        let final_state = plan_json["predicted_states"]
            .as_array()
            .and_then(|states| states.last())
            .expect("goal-image plan should include predicted states");
        let ball = final_state["scene"]["objects"]
            .as_object()
            .unwrap()
            .values()
            .find(|object| object["name"] == "ball")
            .expect("ball should still exist");
        let x = ball["pose"]["position"]["x"].as_f64().unwrap();
        assert!(x > 0.5);
    }

    #[test]
    fn test_execute_plan_updates_world_and_roundtrips_json() {
        let mut world = PyWorld::new("execute_plan", "mock");
        let position = PyPosition::new(0.0, 0.5, 0.0);
        let bbox = PyBBox::new(
            &PyPosition::new(-0.1, 0.4, -0.1),
            &PyPosition::new(0.1, 0.6, 0.1),
        );
        let object = PySceneObject::new("ball", &position, &bbox);
        world.add_object(&object).unwrap();

        let plan = world
            .plan(
                Some("move ball to position (1.0, 0.5, 0.0)"),
                None,
                4,
                10.0,
                Some("mock"),
                "sampling",
                Some(48),
                Some(5),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )
            .unwrap();
        let execution = world
            .execute_plan(&plan, 1, Some("mock"), None, false, None, None, false)
            .unwrap();

        assert!(execution.step_count() > 0);
        assert_eq!(world.step(), execution.final_world().step());
        assert_eq!(world.history().len(), execution.step_count() + 1);
        assert!(
            execution
                .final_world()
                .get_object("ball")
                .unwrap()
                .position()
                .x()
                > 0.5
        );

        let json = execution.to_json().unwrap();
        let roundtrip = PyPlanExecution::from_json(&json).unwrap();
        assert_eq!(roundtrip.step_count(), execution.step_count());
    }

    // --- Evaluation tests ---

    #[test]
    fn test_run_eval_physics() {
        let results = run_eval("physics", "mock", None).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].provider(), "mock");
        assert!(results[0].dimension_scores().contains_key("overall"));
        assert!(results[0].scenario_pass_rate() > 0.0);
    }

    #[test]
    fn test_run_eval_uses_suite_default_providers_when_omitted() {
        let results = run_eval("physics", "", None).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].provider(), "mock");
    }

    #[test]
    fn test_run_eval_invalid_suite() {
        let result = run_eval("nonexistent", "mock", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_result_repr() {
        let results = run_eval("physics", "mock", None).unwrap();
        let repr = results[0].__repr__();
        assert!(repr.contains("EvalResult"));
    }

    #[test]
    fn test_list_eval_suites_includes_builtins() {
        let suites = list_eval_suites();
        assert!(suites.contains(&"physics".to_string()));
        assert!(suites.contains(&"comprehensive".to_string()));
    }

    #[test]
    fn test_run_eval_report_with_custom_suite_json() {
        let suite_json =
            serde_json::to_string(&worldforge_eval::EvalSuite::physics_standard()).unwrap();
        let report = run_eval_report("physics", "mock", Some(&suite_json)).unwrap();
        assert!(report.contains("\"suite\": \"Physics Standard\""));
        assert!(report.contains("\"provider\": \"mock\""));
    }

    #[test]
    fn test_run_eval_report_data_exposes_rollups() {
        let report = run_eval_report_data("physics", "mock", None).unwrap();
        assert_eq!(report.suite(), "Physics Standard");
        assert_eq!(report.provider_summaries().len(), 1);
        assert!(report
            .dimension_summaries()
            .iter()
            .any(|summary| summary.dimension() == "gravity_compliance"));
        assert!(report
            .scenario_summaries()
            .iter()
            .any(|summary| summary.scenario() == "object_drop"));
    }

    #[test]
    fn test_eval_report_roundtrips_and_renders_artifacts() {
        let report = run_eval_report_data("physics", "mock", None).unwrap();

        let json = report.to_json().unwrap();
        let restored = PyEvalReport::from_json(&json).unwrap();
        assert_eq!(restored.suite(), "Physics Standard");

        let markdown = restored.to_markdown().unwrap();
        assert!(markdown.contains("# Evaluation Report: Physics Standard"));
        assert!(markdown.contains("## Leaderboard"));

        let csv = restored.to_csv().unwrap();
        assert!(csv
            .lines()
            .next()
            .unwrap()
            .contains("suite,provider,scenario"));
        assert!(csv.contains("Physics Standard,mock,object_drop"));
    }

    #[test]
    fn test_eval_suite_run_report_data_with_world_object_uses_supplied_state() {
        let suite = PyEvalSuite {
            inner: worldforge_eval::EvalSuite {
                name: "World-aware".to_string(),
                scenarios: vec![worldforge_eval::EvalScenario {
                    name: "find_mug".to_string(),
                    description: "Check the supplied world state".to_string(),
                    initial_state: WorldState::new("empty", "eval"),
                    actions: vec![],
                    expected_outcomes: vec![worldforge_eval::ExpectedOutcome::ObjectExists {
                        name: "mug".to_string(),
                    }],
                    ground_truth: None,
                }],
                dimensions: vec![worldforge_eval::EvalDimension::ObjectPermanence],
                providers: vec![],
            },
        };

        let empty_report = suite
            .run_report_data_with_world("mock", None, None)
            .unwrap();
        assert_eq!(
            empty_report.provider_summaries()[0].scenario_pass_rate(),
            0.0
        );
        assert_eq!(
            empty_report.provider_summaries()[0].outcome_pass_rate(),
            0.0
        );

        let mut world = PyWorld::new("world-aware", "mock");
        let position = PyPosition::new(0.0, 0.5, 0.0);
        let bbox = PyBBox::new(
            &PyPosition::new(-0.1, 0.4, -0.1),
            &PyPosition::new(0.1, 0.6, 0.1),
        );
        world
            .add_object(&PySceneObject::new("mug", &position, &bbox))
            .unwrap();

        let report = suite
            .run_report_data_with_world("mock", Some(&world), None)
            .unwrap();
        assert_eq!(report.provider_summaries()[0].scenario_pass_rate(), 1.0);
        assert_eq!(report.provider_summaries()[0].outcome_pass_rate(), 1.0);
    }

    #[test]
    fn test_eval_suite_run_report_data_with_world_json_uses_supplied_state() {
        let suite = PyEvalSuite {
            inner: worldforge_eval::EvalSuite {
                name: "World-aware JSON".to_string(),
                scenarios: vec![worldforge_eval::EvalScenario {
                    name: "find_mug".to_string(),
                    description: "Check the supplied world JSON".to_string(),
                    initial_state: WorldState::new("empty", "eval"),
                    actions: vec![],
                    expected_outcomes: vec![worldforge_eval::ExpectedOutcome::ObjectExists {
                        name: "mug".to_string(),
                    }],
                    ground_truth: None,
                }],
                dimensions: vec![worldforge_eval::EvalDimension::ObjectPermanence],
                providers: vec![],
            },
        };

        let mut world = PyWorld::new("world-aware-json", "mock");
        let position = PyPosition::new(0.0, 0.5, 0.0);
        let bbox = PyBBox::new(
            &PyPosition::new(-0.1, 0.4, -0.1),
            &PyPosition::new(0.1, 0.6, 0.1),
        );
        world
            .add_object(&PySceneObject::new("mug", &position, &bbox))
            .unwrap();
        let world_json = world.to_json().unwrap();

        let report = suite
            .run_report_data_with_world("mock", None, Some(&world_json))
            .unwrap();
        assert_eq!(report.provider_summaries()[0].scenario_pass_rate(), 1.0);
        assert_eq!(report.provider_summaries()[0].outcome_pass_rate(), 1.0);
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
    fn test_prove_inference_transition_bundle_and_verify_report() {
        let input = serde_json::to_string(&WorldState::new("input", "mock")).unwrap();
        let output = serde_json::to_string(&WorldState::new("output", "mock")).unwrap();

        let bundle = prove_inference_transition_bundle_py(&input, &output, Some("mock")).unwrap();
        assert_eq!(bundle.provider(), "mock");

        let report = bundle.verify().unwrap();
        assert!(report.current_verification().valid());
        assert!(report.verification_matches_recorded());

        let verifier = PyZkVerifier::new("mock").unwrap();
        let verifier_report = verifier.verify_inference_bundle(&bundle).unwrap();
        assert!(verifier_report.current_verification().valid());
    }

    #[test]
    fn test_prove_guardrail_plan_and_verify() {
        let world = PyWorld::new("verify_plan", "mock");
        let plan = world
            .plan(
                Some("spawn cube"),
                None,
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
                false,
            )
            .unwrap();

        let proof = prove_guardrail_plan(&plan).unwrap();
        let (valid, _) = proof.verify().unwrap();
        assert!(valid);
    }

    #[test]
    fn test_plan_prove_guardrail_bundle_and_verify_report() {
        let world = PyWorld::new("verify_bundle_plan", "mock");
        let plan = world
            .plan(
                Some("spawn cube"),
                None,
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
                false,
            )
            .unwrap();

        let bundle = plan.prove_guardrail_bundle().unwrap();
        assert!(bundle.action_count() > 0);

        let report = bundle.verify().unwrap();
        assert!(report.current_verification().valid());
        assert!(report.verification_matches_recorded());

        let verifier = PyMockVerifier::new();
        let verifier_report = verifier.verify_guardrail_bundle(&bundle).unwrap();
        assert!(verifier_report.current_verification().valid());
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
                Some("spawn cube"),
                None,
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
                false,
            )
            .unwrap();
        let verifier = worldforge_verify::MockVerifier::new();
        let bundle = worldforge_verify::prove_guardrail_plan(&verifier, &plan.inner).unwrap();
        let json = serde_json::to_string(&bundle).unwrap();

        let report = verify_bundle_json(&json, "guardrail").unwrap();

        assert!(report.contains("\"verification_matches_recorded\": true"));
    }

    #[test]
    fn test_world_bundle_helpers_cover_latest_inference_and_provenance() {
        let mut world = PyWorld::new("verify_world", "mock");
        let position = PyPosition::new(0.0, 0.5, 0.0);
        let bbox_min = PyPosition::new(-0.1, 0.4, -0.1);
        let bbox_max = PyPosition::new(0.1, 0.6, 0.1);
        let bbox = PyBBox::new(&bbox_min, &bbox_max);
        let object = PySceneObject::new("cube", &position, &bbox);
        world.add_object(&object).unwrap();
        world
            .predict(
                &PyAction::move_to(0.25, 0.5, 0.0, 1.0),
                2,
                Some("mock"),
                None,
                false,
                None,
                false,
            )
            .unwrap();

        let inference_bundle = world.prove_latest_inference_bundle(None).unwrap();
        assert_eq!(inference_bundle.provider(), "mock");
        let inference_report = inference_bundle.verify().unwrap();
        assert!(inference_report.current_verification().valid());

        let provenance_bundle = world
            .prove_provenance_bundle("worldforge-python-tests", Some(1710000000))
            .unwrap();
        assert_eq!(provenance_bundle.provider(), "mock");
        assert_eq!(provenance_bundle.timestamp(), 1710000000);

        let verifier = PyZkVerifier::new("mock").unwrap();
        let provenance_report = verifier
            .verify_provenance_bundle(&provenance_bundle)
            .unwrap();
        assert!(provenance_report.current_verification().valid());
    }
}
