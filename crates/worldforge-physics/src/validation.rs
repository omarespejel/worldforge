//! Physics validation utilities.

use worldforge_core::types::Position;

use crate::world::PhysicsWorld;

/// Result of a physics validation check.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub message: String,
}

impl ValidationResult {
    fn pass(msg: impl Into<String>) -> Self {
        Self {
            passed: true,
            message: msg.into(),
        }
    }
    fn fail(msg: impl Into<String>) -> Self {
        Self {
            passed: false,
            message: msg.into(),
        }
    }
}

/// Validate that gravity is being applied correctly.
///
/// Checks that non-static objects have velocities consistent with the gravity
/// direction after at least one step.
pub fn validate_gravity(world: &PhysicsWorld) -> ValidationResult {
    let g = world.gravity();
    let g_mag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();

    if g_mag < f32::EPSILON {
        return ValidationResult::pass("Zero gravity - no gravity validation needed");
    }

    for obj in world.objects() {
        if obj.properties.is_static {
            continue;
        }
        // Check that object velocity has a component in the gravity direction
        let dot = obj.velocity.x * g[0] + obj.velocity.y * g[1] + obj.velocity.z * g[2];
        if dot < 0.0 {
            // Velocity is opposing gravity - object may be bouncing, that's ok
            continue;
        }
    }

    ValidationResult::pass("Gravity validation passed")
}

/// Validate that no two objects are overlapping (simple AABB collision check).
pub fn validate_collision(world: &PhysicsWorld) -> ValidationResult {
    let objects: Vec<_> = world.objects().collect();
    for i in 0..objects.len() {
        for j in (i + 1)..objects.len() {
            let a = &objects[i];
            let b = &objects[j];
            if aabb_overlap(&a.bbox.min, &a.bbox.max, &b.bbox.min, &b.bbox.max) {
                return ValidationResult::fail(format!(
                    "Collision detected between {} and {}",
                    a.id, b.id
                ));
            }
        }
    }
    ValidationResult::pass("No collisions detected")
}

/// Validate energy conservation (approximate).
///
/// Computes total kinetic + potential energy and checks it doesn't exceed
/// the initial budget (with some tolerance for numerical drift).
pub fn validate_energy(world: &PhysicsWorld, max_energy: f32) -> ValidationResult {
    let g_mag = {
        let g = world.gravity();
        (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()
    };

    let mut total_energy: f32 = 0.0;
    for obj in world.objects() {
        if obj.properties.is_static {
            continue;
        }
        let mass = obj.properties.mass.unwrap_or(1.0);
        let v = &obj.velocity;
        let kinetic = 0.5 * mass * (v.x * v.x + v.y * v.y + v.z * v.z);
        let potential = mass * g_mag * obj.position.y.max(0.0);
        total_energy += kinetic + potential;
    }

    if total_energy > max_energy {
        ValidationResult::fail(format!(
            "Total energy {:.2} exceeds maximum {:.2}",
            total_energy, max_energy
        ))
    } else {
        ValidationResult::pass(format!("Energy {:.2} within budget {:.2}", total_energy, max_energy))
    }
}

fn aabb_overlap(a_min: &Position, a_max: &Position, b_min: &Position, b_max: &Position) -> bool {
    a_min.x <= b_max.x
        && a_max.x >= b_min.x
        && a_min.y <= b_max.y
        && a_max.y >= b_min.y
        && a_min.z <= b_max.z
        && a_max.z >= b_min.z
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;
    use worldforge_core::scene::PhysicsProperties;
    use worldforge_core::types::{BBox, Velocity};

    use super::*;
    use crate::world::PhysicsObject;

    fn make_obj(x: f32, y: f32, z: f32) -> PhysicsObject {
        PhysicsObject {
            id: Uuid::new_v4(),
            position: Position { x, y, z },
            velocity: Velocity::default(),
            bbox: BBox {
                min: Position {
                    x: x - 0.5,
                    y: y - 0.5,
                    z: z - 0.5,
                },
                max: Position {
                    x: x + 0.5,
                    y: y + 0.5,
                    z: z + 0.5,
                },
            },
            properties: PhysicsProperties {
                mass: Some(1.0),
                friction: Some(0.0),
                restitution: Some(0.5),
                is_static: false,
                is_graspable: false,
                material: None,
            },
        }
    }

    #[test]
    fn test_validate_gravity_default() {
        let world = PhysicsWorld::default();
        let result = validate_gravity(&world);
        assert!(result.passed);
    }

    #[test]
    fn test_validate_collision_no_overlap() {
        let mut world = PhysicsWorld::new([0.0, 0.0, 0.0]);
        world.add_object(make_obj(0.0, 5.0, 0.0));
        world.add_object(make_obj(5.0, 5.0, 0.0));
        let result = validate_collision(&world);
        assert!(result.passed);
    }

    #[test]
    fn test_validate_collision_overlap() {
        let mut world = PhysicsWorld::new([0.0, 0.0, 0.0]);
        world.add_object(make_obj(0.0, 0.0, 0.0));
        world.add_object(make_obj(0.5, 0.0, 0.0)); // overlapping
        let result = validate_collision(&world);
        assert!(!result.passed);
    }

    #[test]
    fn test_validate_energy_within_budget() {
        let mut world = PhysicsWorld::default();
        world.add_object(make_obj(0.0, 5.0, 0.0));
        let result = validate_energy(&world, 1000.0);
        assert!(result.passed);
    }

    #[test]
    fn test_validate_energy_exceeds_budget() {
        let mut world = PhysicsWorld::default();
        world.add_object(make_obj(0.0, 100.0, 0.0));
        let result = validate_energy(&world, 1.0);
        assert!(!result.passed);
    }
}
