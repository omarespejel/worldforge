# RFC-0017: Scene Graph & Physics Integration

| Field   | Value                          |
|---------|--------------------------------|
| Status  | Draft                          |
| Author  | WorldForge Core Team           |
| Created | 2026-04-02                     |
| Updated | 2026-04-02                     |

## Abstract

This RFC specifies the integration of a physics simulation engine into
WorldForge's scene graph system. Using rapier3d for Rust-native physics, the
system provides physically-grounded prediction validation, spatial indexing for
efficient scene queries, and import/export of standard 3D formats (glTF, USD,
URDF). A deterministic simulation mode ensures reproducible results for testing
and verification of world model predictions.

## Motivation

WorldForge's core mission is world modeling—predicting how physical environments
evolve over time. The current scene graph in `crates/worldforge-core/src/scene.rs`
provides `SceneObject` with basic spatial types, but lacks:

- **Physics simulation**: No way to validate whether predicted object trajectories
  comply with physical laws (gravity, momentum, collisions).
- **Spatial queries**: No efficient way to query "all objects within 10 meters of
  this point" or "objects whose bounding volumes overlap."
- **Standard format support**: No import/export to industry-standard 3D formats,
  making it difficult to integrate with existing 3D pipelines.
- **Deterministic replay**: No guarantee that running the same simulation twice
  produces identical results, which is critical for regression testing.

Physics integration transforms WorldForge from a pure prediction system into a
physically-grounded simulation platform where AI predictions can be validated
against Newtonian mechanics.

## Detailed Design

### 1. Rapier3D Integration

We use [rapier3d](https://rapier.rs/) (v0.22+) as the physics engine. Rapier is
written in pure Rust, has no C dependencies, supports deterministic simulation,
and provides rigid body dynamics, collision detection, and joint constraints.

```rust
use rapier3d::prelude::*;

/// The physics world, wrapping rapier3d state.
pub struct PhysicsWorld {
    /// Rigid body storage.
    pub bodies: RigidBodySet,
    /// Collider storage.
    pub colliders: ColliderSet,
    /// Gravity vector (default: -9.81 m/s² on Y axis).
    pub gravity: Vector<Real>,
    /// Integration parameters (timestep, solver iterations, etc.).
    pub integration_parameters: IntegrationParameters,
    /// Internal simulation pipeline.
    pipeline: PhysicsPipeline,
    /// Island manager for sleeping bodies.
    island_manager: IslandManager,
    /// Broad phase collision detection.
    broad_phase: DefaultBroadPhase,
    /// Narrow phase collision detection.
    narrow_phase: NarrowPhase,
    /// Impulse joint storage.
    impulse_joints: ImpulseJointSet,
    /// Multibody joint storage.
    multibody_joints: MultibodyJointSet,
    /// Collision/contact event handler.
    event_handler: PhysicsEventCollector,
    /// CCD solver.
    ccd_solver: CCDSolver,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            gravity: vector![0.0, -9.81, 0.0],
            integration_parameters: IntegrationParameters {
                dt: 1.0 / 60.0,  // 60 Hz default
                ..Default::default()
            },
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            event_handler: PhysicsEventCollector::new(),
            ccd_solver: CCDSolver::new(),
        }
    }

    /// Step the simulation forward by one timestep.
    pub fn step(&mut self) {
        self.pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            None,
            &(),
            &self.event_handler,
        );
    }

    /// Step forward by a specific duration, using fixed timesteps.
    pub fn step_duration(&mut self, duration: f64) -> Vec<PhysicsEvent> {
        let steps = (duration / self.integration_parameters.dt as f64).ceil() as usize;
        for _ in 0..steps {
            self.step();
        }
        self.event_handler.drain_events()
    }
}
```

#### Physics Event Collection

```rust
pub struct PhysicsEventCollector {
    collision_events: Mutex<Vec<CollisionEvent>>,
    contact_force_events: Mutex<Vec<ContactForceEvent>>,
}

#[derive(Debug, Clone)]
pub enum PhysicsEvent {
    CollisionStarted {
        body_a: SceneObjectId,
        body_b: SceneObjectId,
        contact_point: Vector3,
    },
    CollisionEnded {
        body_a: SceneObjectId,
        body_b: SceneObjectId,
    },
    ContactForce {
        body_a: SceneObjectId,
        body_b: SceneObjectId,
        total_force: Vector3,
        max_force_magnitude: f64,
    },
}
```

### 2. Enhanced Scene Graph

The existing `SceneObject` is extended to bridge scene graph nodes with physics
bodies:

```rust
/// A node in the scene graph, optionally backed by physics.
pub struct SceneNode {
    /// Unique identifier.
    pub id: SceneObjectId,
    /// Human-readable name.
    pub name: String,
    /// Transform relative to parent.
    pub local_transform: Transform3D,
    /// Cached world-space transform.
    world_transform: Cell<Option<Transform3D>>,
    /// Parent node (None for root).
    pub parent: Option<SceneObjectId>,
    /// Child nodes.
    pub children: Vec<SceneObjectId>,
    /// Visual representation.
    pub mesh: Option<MeshData>,
    /// Material/appearance.
    pub material: Option<MaterialData>,
    /// Physics rigid body handle (if physics-enabled).
    pub rigid_body: Option<RigidBodyHandle>,
    /// Physics collider handle (if physics-enabled).
    pub collider: Option<ColliderHandle>,
    /// User-defined metadata.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Whether this node is active in simulation.
    pub active: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Transform3D {
    pub position: Vector3,
    pub rotation: Quaternion,
    pub scale: Vector3,
}

impl Transform3D {
    pub fn identity() -> Self {
        Self {
            position: Vector3::zeros(),
            rotation: Quaternion::identity(),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    /// Compose this transform with a child transform.
    pub fn compose(&self, child: &Transform3D) -> Transform3D {
        Transform3D {
            position: self.position + self.rotation * child.position.component_mul(&self.scale),
            rotation: self.rotation * child.rotation,
            scale: self.scale.component_mul(&child.scale),
        }
    }
}
```

#### Scene Graph Operations

```rust
pub struct SceneGraph {
    nodes: HashMap<SceneObjectId, SceneNode>,
    root_nodes: Vec<SceneObjectId>,
    spatial_index: SpatialIndex,
    physics: Option<PhysicsWorld>,
    id_generator: AtomicU64,
}

impl SceneGraph {
    /// Add a new node to the scene.
    pub fn add_node(
        &mut self,
        parent: Option<SceneObjectId>,
        node: SceneNodeBuilder,
    ) -> Result<SceneObjectId, SceneError> {
        let id = self.next_id();
        let mut scene_node = node.build(id);

        if let Some(parent_id) = parent {
            let parent_node = self.nodes.get_mut(&parent_id)
                .ok_or(SceneError::NodeNotFound(parent_id))?;
            parent_node.children.push(id);
            scene_node.parent = Some(parent_id);
        } else {
            self.root_nodes.push(id);
        }

        // Register with physics if applicable
        if let Some(ref mut physics) = self.physics {
            if let Some(body_desc) = &node.rigid_body_desc {
                let rb = physics.bodies.insert(body_desc.clone());
                scene_node.rigid_body = Some(rb);

                if let Some(collider_desc) = &node.collider_desc {
                    let col = physics.colliders.insert_with_parent(
                        collider_desc.clone(), rb, &mut physics.bodies
                    );
                    scene_node.collider = Some(col);
                }
            }
        }

        // Insert into spatial index
        self.spatial_index.insert(id, &scene_node.world_transform());

        self.nodes.insert(id, scene_node);
        Ok(id)
    }

    /// Remove a node and all its descendants.
    pub fn remove_node(&mut self, id: SceneObjectId) -> Result<Vec<SceneNode>, SceneError> {
        let mut removed = Vec::new();
        let mut stack = vec![id];

        while let Some(node_id) = stack.pop() {
            if let Some(node) = self.nodes.remove(&node_id) {
                // Remove from physics
                if let Some(ref mut physics) = self.physics {
                    if let Some(rb_handle) = node.rigid_body {
                        physics.bodies.remove(
                            rb_handle,
                            &mut physics.island_manager,
                            &mut physics.colliders,
                            &mut physics.impulse_joints,
                            &mut physics.multibody_joints,
                            true,
                        );
                    }
                }

                // Remove from spatial index
                self.spatial_index.remove(node_id);

                // Queue children for removal
                stack.extend(&node.children);

                // Remove from parent's children list
                if let Some(parent_id) = node.parent {
                    if let Some(parent) = self.nodes.get_mut(&parent_id) {
                        parent.children.retain(|&c| c != node_id);
                    }
                }

                removed.push(node);
            }
        }

        self.root_nodes.retain(|&n| n != id);
        Ok(removed)
    }

    /// Reparent a node under a new parent.
    pub fn reparent(
        &mut self,
        node_id: SceneObjectId,
        new_parent: Option<SceneObjectId>,
    ) -> Result<(), SceneError> {
        // Validate no cycles
        if let Some(new_parent_id) = new_parent {
            if self.is_ancestor(node_id, new_parent_id) {
                return Err(SceneError::CycleDetected);
            }
        }

        let node = self.nodes.get_mut(&node_id)
            .ok_or(SceneError::NodeNotFound(node_id))?;
        let old_parent = node.parent;
        node.parent = new_parent;

        // Update old parent's children
        if let Some(old_parent_id) = old_parent {
            if let Some(parent) = self.nodes.get_mut(&old_parent_id) {
                parent.children.retain(|&c| c != node_id);
            }
        } else {
            self.root_nodes.retain(|&n| n != node_id);
        }

        // Update new parent's children
        if let Some(new_parent_id) = new_parent {
            let parent = self.nodes.get_mut(&new_parent_id)
                .ok_or(SceneError::NodeNotFound(new_parent_id))?;
            parent.children.push(node_id);
        } else {
            self.root_nodes.push(node_id);
        }

        // Invalidate world transforms for subtree
        self.invalidate_transforms(node_id);
        Ok(())
    }

    /// Query all nodes within a bounding sphere.
    pub fn query_sphere(
        &self,
        center: Vector3,
        radius: f64,
    ) -> Vec<SceneObjectId> {
        self.spatial_index.query_sphere(center, radius)
    }

    /// Query all nodes within an axis-aligned bounding box.
    pub fn query_aabb(&self, min: Vector3, max: Vector3) -> Vec<SceneObjectId> {
        self.spatial_index.query_aabb(min, max)
    }

    /// Ray cast against the scene, returning hits sorted by distance.
    pub fn ray_cast(
        &self,
        origin: Vector3,
        direction: Vector3,
        max_distance: f64,
    ) -> Vec<RayHit> {
        self.spatial_index.ray_cast(origin, direction, max_distance)
    }
}
```

### 3. Spatial Indexing

We use a BVH (Bounding Volume Hierarchy) as the primary spatial index, with
an optional octree for static geometry.

```rust
pub enum SpatialIndex {
    /// BVH for dynamic scenes (rebuilt on demand).
    Bvh(BvhIndex),
    /// Octree for predominantly static scenes.
    Octree(OctreeIndex),
}

pub struct BvhIndex {
    /// Flat array of BVH nodes.
    nodes: Vec<BvhNode>,
    /// Mapping from scene object IDs to leaf indices.
    object_to_leaf: HashMap<SceneObjectId, usize>,
    /// Whether the BVH needs rebuilding.
    dirty: bool,
}

pub struct BvhNode {
    pub aabb: Aabb,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub object_id: Option<SceneObjectId>,
}

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vector3,
    pub max: Vector3,
}

impl Aabb {
    pub fn contains_point(&self, point: &Vector3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    pub fn intersects_sphere(&self, center: &Vector3, radius: f64) -> bool {
        let closest = Vector3::new(
            center.x.clamp(self.min.x, self.max.x),
            center.y.clamp(self.min.y, self.max.y),
            center.z.clamp(self.min.z, self.max.z),
        );
        (closest - center).norm_squared() <= radius * radius
    }

    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    pub fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: Vector3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vector3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }
}

impl SpatialIndex {
    pub fn query_sphere(&self, center: Vector3, radius: f64) -> Vec<SceneObjectId> {
        match self {
            SpatialIndex::Bvh(bvh) => bvh.query_sphere(center, radius),
            SpatialIndex::Octree(octree) => octree.query_sphere(center, radius),
        }
    }

    pub fn query_aabb(&self, min: Vector3, max: Vector3) -> Vec<SceneObjectId> {
        let query_aabb = Aabb { min, max };
        match self {
            SpatialIndex::Bvh(bvh) => bvh.query_aabb(&query_aabb),
            SpatialIndex::Octree(octree) => octree.query_aabb(&query_aabb),
        }
    }

    pub fn ray_cast(
        &self,
        origin: Vector3,
        direction: Vector3,
        max_distance: f64,
    ) -> Vec<RayHit> {
        let ray = Ray { origin, direction: direction.normalize(), max_distance };
        match self {
            SpatialIndex::Bvh(bvh) => bvh.ray_cast(&ray),
            SpatialIndex::Octree(octree) => octree.ray_cast(&ray),
        }
    }
}

pub struct RayHit {
    pub object_id: SceneObjectId,
    pub distance: f64,
    pub point: Vector3,
    pub normal: Vector3,
}
```

### 4. Physics-Based Prediction Validation

The validation system compares AI-predicted trajectories against physics
simulation to detect physically implausible predictions:

```rust
pub struct PredictionValidator {
    physics: PhysicsWorld,
    config: ValidationConfig,
}

pub struct ValidationConfig {
    /// Maximum allowed position deviation from physics (meters).
    pub position_tolerance: f64,
    /// Maximum allowed velocity deviation (m/s).
    pub velocity_tolerance: f64,
    /// Whether to check momentum conservation.
    pub check_momentum: bool,
    /// Whether to check collision plausibility.
    pub check_collisions: bool,
    /// Whether to check gravity compliance.
    pub check_gravity: bool,
    /// Simulation timestep for validation.
    pub timestep: f64,
}

pub struct ValidationResult {
    pub valid: bool,
    pub violations: Vec<PhysicsViolation>,
    pub confidence: f64,
    pub simulated_trajectory: Vec<ObjectState>,
    pub predicted_trajectory: Vec<ObjectState>,
}

pub enum PhysicsViolation {
    GravityViolation {
        object_id: SceneObjectId,
        expected_accel: Vector3,
        actual_accel: Vector3,
        deviation: f64,
    },
    MomentumViolation {
        timestamp: f64,
        expected_momentum: Vector3,
        actual_momentum: Vector3,
        deviation_percent: f64,
    },
    CollisionMissed {
        body_a: SceneObjectId,
        body_b: SceneObjectId,
        penetration_depth: f64,
    },
    UnrealisticVelocity {
        object_id: SceneObjectId,
        velocity: Vector3,
        max_plausible: f64,
    },
    EnergyViolation {
        timestamp: f64,
        energy_before: f64,
        energy_after: f64,
        change_percent: f64,
    },
}

impl PredictionValidator {
    /// Validate a predicted trajectory against physics simulation.
    pub fn validate(
        &mut self,
        initial_state: &SceneState,
        predicted_states: &[TimestampedState],
    ) -> ValidationResult {
        // Set up physics world from initial state
        self.physics.reset();
        self.load_state(initial_state);

        let mut violations = Vec::new();
        let mut simulated = Vec::new();

        for predicted in predicted_states {
            // Step physics to the predicted timestamp
            let dt = predicted.timestamp - self.physics.current_time();
            self.physics.step_duration(dt);

            let sim_state = self.physics.snapshot();
            simulated.push(sim_state.clone());

            // Compare positions
            for (obj_id, pred_obj) in &predicted.objects {
                if let Some(sim_obj) = sim_state.get(obj_id) {
                    let pos_dev = (pred_obj.position - sim_obj.position).norm();
                    if pos_dev > self.config.position_tolerance {
                        violations.push(PhysicsViolation::GravityViolation {
                            object_id: *obj_id,
                            expected_accel: sim_obj.acceleration,
                            actual_accel: pred_obj.acceleration,
                            deviation: pos_dev,
                        });
                    }
                }
            }

            // Check momentum conservation
            if self.config.check_momentum {
                self.check_momentum(&predicted, &sim_state, &mut violations);
            }

            // Check collisions
            if self.config.check_collisions {
                self.check_collisions(&predicted, &mut violations);
            }
        }

        let confidence = 1.0 - (violations.len() as f64
            / (predicted_states.len() as f64 * 5.0)).min(1.0);

        ValidationResult {
            valid: violations.is_empty(),
            violations,
            confidence,
            simulated_trajectory: simulated,
            predicted_trajectory: predicted_states.iter()
                .map(|s| s.to_object_state()).collect(),
        }
    }
}
```

### 5. Scene Format Import/Export

#### glTF (3D Assets)

```rust
pub struct GltfImporter;

impl GltfImporter {
    /// Import a glTF/GLB file into the scene graph.
    pub fn import(
        path: &Path,
        scene: &mut SceneGraph,
        options: &GltfImportOptions,
    ) -> Result<Vec<SceneObjectId>, SceneFormatError> {
        let (document, buffers, images) = gltf::import(path)?;

        let mut imported_ids = Vec::new();

        for gltf_scene in document.scenes() {
            for node in gltf_scene.nodes() {
                let id = Self::import_node(
                    &node, &buffers, &images, scene, None, options
                )?;
                imported_ids.push(id);
            }
        }

        Ok(imported_ids)
    }

    fn import_node(
        node: &gltf::Node,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
        scene: &mut SceneGraph,
        parent: Option<SceneObjectId>,
        options: &GltfImportOptions,
    ) -> Result<SceneObjectId, SceneFormatError> {
        let transform = Self::convert_transform(node.transform());
        let mesh = node.mesh().map(|m| Self::convert_mesh(&m, buffers));

        let mut builder = SceneNodeBuilder::new(
            node.name().unwrap_or("unnamed").to_string()
        )
        .transform(transform);

        if let Some(mesh_data) = mesh {
            builder = builder.mesh(mesh_data?);
        }

        // Generate physics collider from mesh if requested
        if options.generate_colliders {
            if let Some(mesh_ref) = node.mesh() {
                let collider = Self::generate_collider(&mesh_ref, buffers, options)?;
                builder = builder.collider(collider);
            }
        }

        let id = scene.add_node(parent, builder)?;

        // Recursively import children
        for child in node.children() {
            Self::import_node(&child, buffers, images, scene, Some(id), options)?;
        }

        Ok(id)
    }
}

pub struct GltfExporter;

impl GltfExporter {
    /// Export the scene graph to a glTF file.
    pub fn export(
        scene: &SceneGraph,
        path: &Path,
        options: &GltfExportOptions,
    ) -> Result<(), SceneFormatError> {
        let mut root = json::object::Object::new();
        // ... build glTF JSON structure from scene graph
        // Write binary buffer for mesh data
        // Optionally embed as GLB
        Ok(())
    }
}
```

#### USD (Production Pipelines)

```rust
pub struct UsdImporter;

impl UsdImporter {
    pub fn import(
        path: &Path,
        scene: &mut SceneGraph,
        options: &UsdImportOptions,
    ) -> Result<Vec<SceneObjectId>, SceneFormatError> {
        // USD import via usd-rs bindings or subprocess call to usdcat
        todo!("USD import")
    }
}
```

#### URDF (Robotics)

```rust
pub struct UrdfImporter;

impl UrdfImporter {
    /// Import a URDF robot description into the scene graph with joints.
    pub fn import(
        path: &Path,
        scene: &mut SceneGraph,
        physics: &mut PhysicsWorld,
    ) -> Result<RobotDescription, SceneFormatError> {
        let urdf = urdf_rs::read_file(path)?;

        let mut link_to_node: HashMap<String, SceneObjectId> = HashMap::new();
        let mut joint_handles = Vec::new();

        // Import links as scene nodes with collision geometry
        for link in &urdf.links {
            let node = Self::link_to_scene_node(link)?;
            let id = scene.add_node(None, node)?;
            link_to_node.insert(link.name.clone(), id);
        }

        // Import joints as physics constraints
        for joint in &urdf.joints {
            let parent_id = link_to_node.get(&joint.parent.link)
                .ok_or(SceneFormatError::MissingLink(joint.parent.link.clone()))?;
            let child_id = link_to_node.get(&joint.child.link)
                .ok_or(SceneFormatError::MissingLink(joint.child.link.clone()))?;

            // Reparent in scene graph
            scene.reparent(*child_id, Some(*parent_id))?;

            // Create physics joint
            let joint_handle = Self::create_physics_joint(
                joint, *parent_id, *child_id, physics, scene
            )?;
            joint_handles.push(joint_handle);
        }

        Ok(RobotDescription {
            links: link_to_node,
            joints: joint_handles,
        })
    }
}
```

### 6. Real-Time Physics Stepping

For interactive use cases, physics runs on a dedicated thread with a fixed
timestep:

```rust
pub struct PhysicsLoop {
    world: Arc<Mutex<PhysicsWorld>>,
    scene: Arc<RwLock<SceneGraph>>,
    running: Arc<AtomicBool>,
    target_hz: f64,
}

impl PhysicsLoop {
    pub fn start(self) -> JoinHandle<()> {
        let running = self.running.clone();
        std::thread::spawn(move || {
            let dt = Duration::from_secs_f64(1.0 / self.target_hz);
            let mut accumulator = Duration::ZERO;
            let mut last_time = Instant::now();

            while running.load(Ordering::Relaxed) {
                let now = Instant::now();
                accumulator += now - last_time;
                last_time = now;

                // Fixed timestep loop
                while accumulator >= dt {
                    {
                        let mut physics = self.world.lock().unwrap();
                        physics.step();

                        // Sync transforms back to scene graph
                        let mut scene = self.scene.write().unwrap();
                        Self::sync_transforms(&physics, &mut scene);
                    }
                    accumulator -= dt;
                }

                // Sleep for remaining time
                let sleep_time = dt.saturating_sub(accumulator);
                if sleep_time > Duration::from_millis(1) {
                    std::thread::sleep(sleep_time);
                }
            }
        })
    }

    fn sync_transforms(physics: &PhysicsWorld, scene: &mut SceneGraph) {
        for (handle, body) in physics.bodies.iter() {
            if body.is_dynamic() {
                if let Some(node_id) = physics.handle_to_node.get(&handle) {
                    if let Some(node) = scene.get_node_mut(node_id) {
                        let pos = body.translation();
                        let rot = body.rotation();
                        node.local_transform.position = Vector3::new(pos.x, pos.y, pos.z);
                        node.local_transform.rotation = *rot;
                    }
                }
            }
        }
    }
}
```

### 7. Deterministic Simulation Mode

For reproducible results, WorldForge supports deterministic physics:

```rust
pub struct DeterministicConfig {
    /// Fixed random seed for any stochastic processes.
    pub seed: u64,
    /// Fixed timestep (no variable dt).
    pub fixed_dt: f64,
    /// Disable parallel solving (single-threaded for determinism).
    pub single_threaded: bool,
    /// Fixed solver iteration counts.
    pub velocity_iterations: usize,
    pub position_iterations: usize,
}

impl PhysicsWorld {
    pub fn enable_deterministic_mode(&mut self, config: DeterministicConfig) {
        self.integration_parameters.dt = config.fixed_dt as f32;
        self.integration_parameters.num_solver_iterations =
            NonZeroUsize::new(config.velocity_iterations).unwrap();
        self.integration_parameters.num_additional_friction_iterations =
            config.position_iterations;

        // Rapier supports cross-platform determinism when:
        // 1. Using the same floating point settings
        // 2. Processing bodies in the same order
        // 3. Using single-threaded solving
        self.deterministic = true;
    }

    /// Serialize the complete physics state for snapshot/restore.
    pub fn snapshot(&self) -> PhysicsSnapshot {
        PhysicsSnapshot {
            bodies: bincode::serialize(&self.bodies).unwrap(),
            colliders: bincode::serialize(&self.colliders).unwrap(),
            joints: bincode::serialize(&self.impulse_joints).unwrap(),
            narrow_phase: bincode::serialize(&self.narrow_phase).unwrap(),
            island_manager: bincode::serialize(&self.island_manager).unwrap(),
            timestamp: self.current_time,
        }
    }

    /// Restore physics state from a snapshot.
    pub fn restore(&mut self, snapshot: &PhysicsSnapshot) {
        self.bodies = bincode::deserialize(&snapshot.bodies).unwrap();
        self.colliders = bincode::deserialize(&snapshot.colliders).unwrap();
        self.impulse_joints = bincode::deserialize(&snapshot.joints).unwrap();
        self.narrow_phase = bincode::deserialize(&snapshot.narrow_phase).unwrap();
        self.island_manager = bincode::deserialize(&snapshot.island_manager).unwrap();
        self.current_time = snapshot.timestamp;
    }
}
```

## Implementation Plan

### Phase 1: Core Physics (3 weeks)
- Add `rapier3d` to `worldforge-core` dependencies.
- Implement `PhysicsWorld` wrapper around rapier3d.
- Extend `SceneObject` / `SceneNode` with physics handles.
- Basic physics stepping and transform synchronization.
- Unit tests for rigid body simulation.

### Phase 2: Scene Graph Enhancement (2 weeks)
- Implement add/remove/reparent operations.
- Build BVH spatial index.
- Implement sphere, AABB, and ray queries.
- Invalidation and dirty-flag system for world transforms.

### Phase 3: Prediction Validation (2 weeks)
- Implement `PredictionValidator` with configurable tolerances.
- Gravity compliance checking.
- Momentum conservation checking.
- Collision detection validation.
- Integration with prediction pipeline.

### Phase 4: Format Import/Export (3 weeks)
- glTF import with mesh and hierarchy (using `gltf` crate).
- glTF export for scene serialization.
- URDF import for robotics (using `urdf-rs` crate).
- USD import/export (may require Python subprocess or `usd-rs`).

### Phase 5: Real-Time & Determinism (2 weeks)
- Fixed-timestep physics loop on dedicated thread.
- Deterministic simulation configuration.
- Snapshot/restore for physics state.
- Cross-platform determinism testing.

### Phase 6: Performance (1 week)
- Benchmark spatial queries with 10K+ objects.
- Profile physics stepping with complex scenes.
- BVH rebuild optimization (incremental updates).
- Memory usage optimization for large scenes.

## Testing Strategy

### Unit Tests
- Transform composition (parent-child hierarchy).
- AABB intersection and containment.
- BVH construction and query correctness.
- Physics stepping produces expected motion under gravity.
- Deterministic mode produces identical results across runs.

### Integration Tests
- Import a glTF file, add physics, simulate, export back.
- URDF robot import with joint constraints.
- Prediction validation catches obvious physics violations.
- Scene graph operations maintain consistency under concurrent access.

### Determinism Tests
- Run the same simulation 100 times and assert bitwise-identical results.
- Test determinism across different platforms (Linux, macOS, Windows).
- Snapshot and restore produces identical forward simulation.

### Performance Benchmarks
- Scene graph operations (add/remove) with 10K, 100K, 1M nodes.
- Spatial queries: sphere query with 100K objects.
- Physics stepping: 1K rigid bodies at 60Hz.
- glTF import time for scenes of various complexity.

### Fuzz Tests
- Random scene graph operations (add, remove, reparent) for crash testing.
- Random physics configurations to find edge cases.

## Open Questions

1. **Rapier3d vs alternatives**: Should we consider `nphysics` or `bevy_rapier`
   instead? Rapier3d is actively maintained and widely used, but direct bevy
   integration might be relevant if we add visualization.

2. **2D physics**: Should we also support `rapier2d` for 2D world models, or
   treat 2D as a special case of 3D (constrained to a plane)?

3. **USD support complexity**: Full USD support is extremely complex. Should we
   limit to USDA (ASCII) initially and use a Python subprocess for USDC/USDZ?

4. **Physics LOD**: For very large scenes, should we implement distance-based
   level-of-detail for physics (simplified colliders for distant objects)?

5. **Soft body physics**: Rapier3d is rigid-body only. Should we plan for
   soft body / cloth simulation, and if so, which library?

6. **Scene graph concurrency**: The current design uses `RwLock<SceneGraph>`.
   Should we explore lock-free approaches for better concurrent read performance?

7. **Mesh generation**: Should WorldForge generate primitive colliders (box,
   sphere, capsule) automatically from imported meshes, or require explicit
   specification?
