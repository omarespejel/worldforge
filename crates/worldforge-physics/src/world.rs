//! Physics world with Euler integration.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use worldforge_core::scene::PhysicsProperties;
use worldforge_core::types::{BBox, Position, Velocity};

/// Unique identifier for physics objects.
pub type ObjectId = Uuid;

/// A rigid body in the physics world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsObject {
    pub id: ObjectId,
    pub position: Position,
    pub velocity: Velocity,
    pub bbox: BBox,
    pub properties: PhysicsProperties,
}

/// Snapshot of the physics world state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsState {
    pub objects: HashMap<ObjectId, PhysicsObject>,
    pub gravity: [f32; 3],
    pub time: f64,
}

/// Physics world with semi-implicit Euler integration.
#[derive(Debug)]
pub struct PhysicsWorld {
    objects: HashMap<ObjectId, PhysicsObject>,
    gravity: [f32; 3],
    time: f64,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self {
            objects: HashMap::new(),
            gravity: [0.0, -9.81, 0.0],
            time: 0.0,
        }
    }
}

impl PhysicsWorld {
    /// Create a new physics world with the given gravity vector.
    pub fn new(gravity: [f32; 3]) -> Self {
        Self {
            gravity,
            ..Default::default()
        }
    }

    /// Add an object to the world. Returns the assigned ObjectId.
    pub fn add_object(&mut self, obj: PhysicsObject) -> ObjectId {
        let id = obj.id;
        tracing::debug!("Adding physics object {}", id);
        self.objects.insert(id, obj);
        id
    }

    /// Remove an object by id. Returns the removed object if it existed.
    pub fn remove_object(&mut self, id: ObjectId) -> Option<PhysicsObject> {
        tracing::debug!("Removing physics object {}", id);
        self.objects.remove(&id)
    }

    /// Get an immutable reference to an object.
    pub fn get_object(&self, id: ObjectId) -> Option<&PhysicsObject> {
        self.objects.get(&id)
    }

    /// Get the current gravity vector.
    pub fn gravity(&self) -> [f32; 3] {
        self.gravity
    }

    /// Set the gravity vector.
    pub fn set_gravity(&mut self, gravity: [f32; 3]) {
        self.gravity = gravity;
    }

    /// Number of objects in the world.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Current simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// All objects as an iterator.
    pub fn objects(&self) -> impl Iterator<Item = &PhysicsObject> {
        self.objects.values()
    }

    /// Advance the simulation by `dt` seconds using semi-implicit Euler integration.
    ///
    /// For each non-static object:
    ///   1. Apply gravity to velocity: v += g * dt
    ///   2. Apply damping (friction approximation): v *= (1 - friction * dt)
    ///   3. Update position: p += v * dt
    ///   4. Simple ground-plane collision at y=0
    pub fn step(&mut self, dt: f32) {
        let gx = self.gravity[0];
        let gy = self.gravity[1];
        let gz = self.gravity[2];

        for obj in self.objects.values_mut() {
            if obj.properties.is_static {
                continue;
            }

            // Apply gravity
            obj.velocity.x += gx * dt;
            obj.velocity.y += gy * dt;
            obj.velocity.z += gz * dt;

            // Simple friction damping
            let friction = obj.properties.friction.unwrap_or(0.0);
            let damping = (1.0 - friction * dt).max(0.0);
            obj.velocity.x *= damping;
            obj.velocity.y *= damping;
            obj.velocity.z *= damping;

            // Update position (semi-implicit Euler: use updated velocity)
            obj.position.x += obj.velocity.x * dt;
            obj.position.y += obj.velocity.y * dt;
            obj.position.z += obj.velocity.z * dt;

            // Ground plane collision at y=0
            let half_height = (obj.bbox.max.y - obj.bbox.min.y) / 2.0;
            if obj.position.y - half_height < 0.0 {
                obj.position.y = half_height;
                let restitution = obj.properties.restitution.unwrap_or(0.5);
                obj.velocity.y = -obj.velocity.y * restitution;
                // If velocity is very small, just stop
                if obj.velocity.y.abs() < 0.01 {
                    obj.velocity.y = 0.0;
                }
            }

            // Update bbox to match new position
            let hw = (obj.bbox.max.x - obj.bbox.min.x) / 2.0;
            let hh = half_height;
            let hd = (obj.bbox.max.z - obj.bbox.min.z) / 2.0;
            obj.bbox.min = Position {
                x: obj.position.x - hw,
                y: obj.position.y - hh,
                z: obj.position.z - hd,
            };
            obj.bbox.max = Position {
                x: obj.position.x + hw,
                y: obj.position.y + hh,
                z: obj.position.z + hd,
            };
        }

        self.time += dt as f64;
    }

    /// Get a snapshot of the current world state.
    pub fn get_state(&self) -> PhysicsState {
        PhysicsState {
            objects: self.objects.clone(),
            gravity: self.gravity,
            time: self.time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_object(y: f32, is_static: bool) -> PhysicsObject {
        PhysicsObject {
            id: Uuid::new_v4(),
            position: Position { x: 0.0, y, z: 0.0 },
            velocity: Velocity::default(),
            bbox: BBox {
                min: Position {
                    x: -0.5,
                    y: y - 0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 0.5,
                    y: y + 0.5,
                    z: 0.5,
                },
            },
            properties: PhysicsProperties {
                mass: Some(1.0),
                friction: Some(0.1),
                restitution: Some(0.5),
                is_static,
                is_graspable: false,
                material: None,
            },
        }
    }

    #[test]
    fn test_add_remove_object() {
        let mut world = PhysicsWorld::default();
        let obj = make_object(5.0, false);
        let id = obj.id;
        world.add_object(obj);
        assert_eq!(world.object_count(), 1);
        assert!(world.get_object(id).is_some());

        let removed = world.remove_object(id);
        assert!(removed.is_some());
        assert_eq!(world.object_count(), 0);
    }

    #[test]
    fn test_gravity_affects_velocity() {
        let mut world = PhysicsWorld::default();
        let obj = make_object(10.0, false);
        let id = obj.id;
        world.add_object(obj);

        world.step(1.0);
        let o = world.get_object(id).unwrap();
        // After 1s of gravity, vy should be negative (falling)
        assert!(o.velocity.y < 0.0);
    }

    #[test]
    fn test_static_objects_dont_move() {
        let mut world = PhysicsWorld::default();
        let obj = make_object(5.0, true);
        let id = obj.id;
        world.add_object(obj);

        world.step(1.0);
        let o = world.get_object(id).unwrap();
        assert!((o.position.y - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ground_collision() {
        let mut world = PhysicsWorld::default();
        let obj = make_object(0.6, false);
        let id = obj.id;
        world.add_object(obj);

        // Step many times to let object settle
        for _ in 0..1000 {
            world.step(0.01);
        }
        let o = world.get_object(id).unwrap();
        // Object should be resting at or near ground (y = half_height = 0.5)
        assert!(o.position.y >= 0.0);
    }

    #[test]
    fn test_get_state() {
        let mut world = PhysicsWorld::new([0.0, -10.0, 0.0]);
        let obj = make_object(5.0, false);
        world.add_object(obj);
        world.step(0.1);

        let state = world.get_state();
        assert_eq!(state.objects.len(), 1);
        assert_eq!(state.gravity, [0.0, -10.0, 0.0]);
        assert!(state.time > 0.0);
    }

    #[test]
    fn test_time_advances() {
        let mut world = PhysicsWorld::default();
        assert!((world.time() - 0.0).abs() < f64::EPSILON);
        world.step(0.5);
        assert!((world.time() - 0.5).abs() < 1e-6);
        world.step(0.5);
        assert!((world.time() - 1.0).abs() < 1e-6);
    }
}
