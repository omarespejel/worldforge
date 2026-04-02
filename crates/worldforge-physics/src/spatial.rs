//! Spatial query utilities: sphere/box queries and raycasting.

use worldforge_core::types::Position;

use crate::world::{ObjectId, PhysicsObject, PhysicsWorld};

/// Result of a raycast hit.
#[derive(Debug, Clone)]
pub struct RayHit {
    pub object_id: ObjectId,
    /// Distance from ray origin to the hit point.
    pub distance: f32,
    /// Hit point in world coordinates.
    pub point: Position,
}

/// Spatial query interface for the physics world.
pub struct SpatialQuery;

impl SpatialQuery {
    /// Find all objects whose center is within a sphere.
    pub fn objects_in_sphere(
        world: &PhysicsWorld,
        center: &Position,
        radius: f32,
    ) -> Vec<ObjectId> {
        let r2 = radius * radius;
        world
            .objects()
            .filter(|obj| {
                let dx = obj.position.x - center.x;
                let dy = obj.position.y - center.y;
                let dz = obj.position.z - center.z;
                dx * dx + dy * dy + dz * dz <= r2
            })
            .map(|obj| obj.id)
            .collect()
    }

    /// Find all objects whose AABB intersects with the given box.
    pub fn objects_in_box(
        world: &PhysicsWorld,
        box_min: &Position,
        box_max: &Position,
    ) -> Vec<ObjectId> {
        world
            .objects()
            .filter(|obj| {
                obj.bbox.min.x <= box_max.x
                    && obj.bbox.max.x >= box_min.x
                    && obj.bbox.min.y <= box_max.y
                    && obj.bbox.max.y >= box_min.y
                    && obj.bbox.min.z <= box_max.z
                    && obj.bbox.max.z >= box_min.z
            })
            .map(|obj| obj.id)
            .collect()
    }

    /// Cast a ray and return hits sorted by distance.
    ///
    /// Uses ray-AABB intersection (slab method).
    pub fn raycast(
        world: &PhysicsWorld,
        origin: &Position,
        direction: &Position,
        max_distance: f32,
    ) -> Vec<RayHit> {
        // Normalize direction
        let len = (direction.x * direction.x
            + direction.y * direction.y
            + direction.z * direction.z)
            .sqrt();
        if len < f32::EPSILON {
            return vec![];
        }
        let dx = direction.x / len;
        let dy = direction.y / len;
        let dz = direction.z / len;

        let mut hits: Vec<RayHit> = world
            .objects()
            .filter_map(|obj| {
                ray_aabb_intersect(origin, dx, dy, dz, obj, max_distance)
            })
            .collect();

        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }
}

/// Ray-AABB intersection using the slab method.
fn ray_aabb_intersect(
    origin: &Position,
    dx: f32,
    dy: f32,
    dz: f32,
    obj: &PhysicsObject,
    max_distance: f32,
) -> Option<RayHit> {
    let inv_dx = if dx.abs() > f32::EPSILON { 1.0 / dx } else { f32::INFINITY };
    let inv_dy = if dy.abs() > f32::EPSILON { 1.0 / dy } else { f32::INFINITY };
    let inv_dz = if dz.abs() > f32::EPSILON { 1.0 / dz } else { f32::INFINITY };

    let t1x = (obj.bbox.min.x - origin.x) * inv_dx;
    let t2x = (obj.bbox.max.x - origin.x) * inv_dx;
    let t1y = (obj.bbox.min.y - origin.y) * inv_dy;
    let t2y = (obj.bbox.max.y - origin.y) * inv_dy;
    let t1z = (obj.bbox.min.z - origin.z) * inv_dz;
    let t2z = (obj.bbox.max.z - origin.z) * inv_dz;

    let tmin = t1x.min(t2x).max(t1y.min(t2y)).max(t1z.min(t2z));
    let tmax = t1x.max(t2x).min(t1y.max(t2y)).min(t1z.max(t2z));

    if tmax < 0.0 || tmin > tmax || tmin > max_distance {
        return None;
    }

    let t = if tmin >= 0.0 { tmin } else { tmax };
    if t > max_distance {
        return None;
    }

    Some(RayHit {
        object_id: obj.id,
        distance: t,
        point: Position {
            x: origin.x + dx * t,
            y: origin.y + dy * t,
            z: origin.z + dz * t,
        },
    })
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;
    use worldforge_core::scene::PhysicsProperties;
    use worldforge_core::types::{BBox, Velocity};

    use super::*;

    fn make_obj_at(x: f32, y: f32, z: f32) -> PhysicsObject {
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
                is_static: true,
                is_graspable: false,
                material: None,
            },
        }
    }

    #[test]
    fn test_objects_in_sphere() {
        let mut world = PhysicsWorld::new([0.0, 0.0, 0.0]);
        let near = make_obj_at(1.0, 0.0, 0.0);
        let far = make_obj_at(10.0, 0.0, 0.0);
        let near_id = near.id;
        world.add_object(near);
        world.add_object(far);

        let center = Position { x: 0.0, y: 0.0, z: 0.0 };
        let result = SpatialQuery::objects_in_sphere(&world, &center, 2.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], near_id);
    }

    #[test]
    fn test_objects_in_box() {
        let mut world = PhysicsWorld::new([0.0, 0.0, 0.0]);
        let inside = make_obj_at(1.0, 1.0, 1.0);
        let outside = make_obj_at(20.0, 20.0, 20.0);
        let inside_id = inside.id;
        world.add_object(inside);
        world.add_object(outside);

        let bmin = Position { x: -5.0, y: -5.0, z: -5.0 };
        let bmax = Position { x: 5.0, y: 5.0, z: 5.0 };
        let result = SpatialQuery::objects_in_box(&world, &bmin, &bmax);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], inside_id);
    }

    #[test]
    fn test_raycast_hit() {
        let mut world = PhysicsWorld::new([0.0, 0.0, 0.0]);
        let obj = make_obj_at(5.0, 0.0, 0.0);
        let obj_id = obj.id;
        world.add_object(obj);

        let origin = Position { x: 0.0, y: 0.0, z: 0.0 };
        let dir = Position { x: 1.0, y: 0.0, z: 0.0 };
        let hits = SpatialQuery::raycast(&world, &origin, &dir, 100.0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].object_id, obj_id);
        assert!(hits[0].distance > 0.0);
    }

    #[test]
    fn test_raycast_miss() {
        let mut world = PhysicsWorld::new([0.0, 0.0, 0.0]);
        let obj = make_obj_at(5.0, 0.0, 0.0);
        world.add_object(obj);

        let origin = Position { x: 0.0, y: 0.0, z: 0.0 };
        let dir = Position { x: 0.0, y: 1.0, z: 0.0 }; // pointing up, object is to the right
        let hits = SpatialQuery::raycast(&world, &origin, &dir, 100.0);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_raycast_max_distance() {
        let mut world = PhysicsWorld::new([0.0, 0.0, 0.0]);
        let obj = make_obj_at(50.0, 0.0, 0.0);
        world.add_object(obj);

        let origin = Position { x: 0.0, y: 0.0, z: 0.0 };
        let dir = Position { x: 1.0, y: 0.0, z: 0.0 };
        let hits = SpatialQuery::raycast(&world, &origin, &dir, 10.0);
        assert!(hits.is_empty());
    }
}
