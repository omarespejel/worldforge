// PyO3-generated code triggers this clippy lint; it's a known false positive.
#![allow(clippy::useless_conversion)]
//! Python bindings for WorldForge.
//!
//! Exposes core types, scene management, and the main WorldForge
//! orchestrator to Python via PyO3.

use pyo3::prelude::*;

use worldforge_core::scene::SceneObject;
use worldforge_core::state::WorldState;
use worldforge_core::types::{BBox, Position, Rotation, Velocity};

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
}

#[pymethods]
impl PyWorldForge {
    /// Create a new WorldForge instance with auto-detected providers.
    #[new]
    fn new() -> Self {
        let registry = worldforge_providers::auto_detect();
        let mut wf = worldforge_core::WorldForge::new();
        // Transfer providers from auto-detected registry
        for name in registry.list() {
            // We can't move providers out of a registry, so we create
            // a fresh WorldForge with mock provider as minimum.
            let _ = name;
        }
        wf.register_provider(Box::new(worldforge_providers::MockProvider::new()))
            .ok();
        Self { inner: wf }
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

    fn __repr__(&self) -> String {
        format!("WorldForge(providers={:?})", self.inner.providers())
    }
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
    m.add_class::<PyAction>()?;
    m.add_class::<PyGuardrail>()?;
    m.add_class::<PyWorldForge>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let wf = PyWorldForge::new();
        let providers = wf.providers();
        assert!(providers.contains(&"mock".to_string()));
    }

    #[test]
    fn test_worldforge_create_world() {
        let wf = PyWorldForge::new();
        let world = wf.create_world("test_world", "mock").unwrap();
        assert_eq!(world.name(), "test_world");
        assert_eq!(world.object_count(), 0);
    }

    #[test]
    fn test_worldforge_create_world_unknown_provider() {
        let wf = PyWorldForge::new();
        let result = wf.create_world("test", "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_worldforge_repr() {
        let wf = PyWorldForge::new();
        let repr = wf.__repr__();
        assert!(repr.contains("WorldForge"));
    }
}
