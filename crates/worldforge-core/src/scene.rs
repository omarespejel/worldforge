//! Scene graph for spatial world representation.
//!
//! The scene graph is the primary data structure for representing
//! the spatial layout of objects in a world.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{BBox, Mesh, ObjectId, Pose, Tensor, Velocity};

/// Hierarchical scene representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneGraph {
    /// Root node of the scene hierarchy.
    pub root: SceneNode,
    /// All objects indexed by their unique ID.
    pub objects: HashMap<ObjectId, SceneObject>,
    /// Spatial relationships between objects.
    pub relationships: Vec<SpatialRelationship>,
}

/// A node in the scene hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneNode {
    /// Human-readable name.
    pub name: String,
    /// Child nodes.
    pub children: Vec<SceneNode>,
    /// Object ID if this node represents an object.
    pub object_id: Option<ObjectId>,
}

/// A physical object in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneObject {
    /// Unique identifier.
    pub id: ObjectId,
    /// Human-readable name.
    pub name: String,
    /// Position and orientation in world space.
    pub pose: Pose,
    /// Axis-aligned bounding box.
    pub bbox: BBox,
    /// Optional 3D mesh geometry.
    pub mesh: Option<Mesh>,
    /// Physical properties for simulation.
    pub physics: PhysicsProperties,
    /// Current velocity of the object.
    pub velocity: Velocity,
    /// Semantic label (e.g. "mug", "table").
    pub semantic_label: Option<String>,
    /// Provider-specific visual embedding vector.
    pub visual_embedding: Option<Tensor>,
}

/// Physical properties of a scene object.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhysicsProperties {
    /// Mass in kilograms.
    pub mass: Option<f32>,
    /// Friction coefficient.
    pub friction: Option<f32>,
    /// Restitution (bounciness) coefficient.
    pub restitution: Option<f32>,
    /// Whether the object is immovable.
    pub is_static: bool,
    /// Whether the object can be grasped.
    pub is_graspable: bool,
    /// Material name (e.g. "wood", "metal").
    pub material: Option<String>,
}

/// Spatial relationship between objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialRelationship {
    /// Subject is resting on a surface.
    On {
        subject: ObjectId,
        surface: ObjectId,
    },
    /// Subject is contained within a container.
    In {
        subject: ObjectId,
        container: ObjectId,
    },
    /// Two objects are within a given distance.
    Near {
        a: ObjectId,
        b: ObjectId,
        distance: f32,
    },
    /// Two objects are in contact.
    Touching { a: ObjectId, b: ObjectId },
    /// Subject is above the reference object.
    Above {
        subject: ObjectId,
        reference: ObjectId,
    },
    /// Subject is below the reference object.
    Below {
        subject: ObjectId,
        reference: ObjectId,
    },
}

impl SceneGraph {
    /// Create an empty scene graph.
    pub fn new() -> Self {
        Self {
            root: SceneNode {
                name: "root".to_string(),
                children: Vec::new(),
                object_id: None,
            },
            objects: HashMap::new(),
            relationships: Vec::new(),
        }
    }

    /// Add an object to the scene.
    pub fn add_object(&mut self, object: SceneObject) {
        let id = object.id;
        let name = object.name.clone();
        self.objects.insert(id, object);
        self.root.children.push(SceneNode {
            name,
            children: Vec::new(),
            object_id: Some(id),
        });
    }

    /// Get an object by its ID.
    pub fn get_object(&self, id: &ObjectId) -> Option<&SceneObject> {
        self.objects.get(id)
    }

    /// Get a mutable reference to an object by its ID.
    pub fn get_object_mut(&mut self, id: &ObjectId) -> Option<&mut SceneObject> {
        self.objects.get_mut(id)
    }

    /// Remove an object from the scene.
    pub fn remove_object(&mut self, id: &ObjectId) -> Option<SceneObject> {
        self.root
            .children
            .retain(|n| n.object_id.as_ref() != Some(id));
        self.relationships.retain(|r| !relationship_involves(r, id));
        self.objects.remove(id)
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a spatial relationship involves a given object.
fn relationship_involves(rel: &SpatialRelationship, id: &ObjectId) -> bool {
    match rel {
        SpatialRelationship::On { subject, surface } => subject == id || surface == id,
        SpatialRelationship::In { subject, container } => subject == id || container == id,
        SpatialRelationship::Near { a, b, .. } => a == id || b == id,
        SpatialRelationship::Touching { a, b } => a == id || b == id,
        SpatialRelationship::Above { subject, reference } => subject == id || reference == id,
        SpatialRelationship::Below { subject, reference } => subject == id || reference == id,
    }
}

impl SceneObject {
    /// Create a new scene object with minimal required fields.
    pub fn new(name: impl Into<String>, pose: Pose, bbox: BBox) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            name: name.into(),
            pose,
            bbox,
            velocity: Velocity {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            mesh: None,
            physics: PhysicsProperties::default(),
            semantic_label: None,
            visual_embedding: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Position;

    fn sample_object(name: &str) -> SceneObject {
        SceneObject::new(
            name,
            Pose::default(),
            BBox {
                min: Position {
                    x: -0.5,
                    y: -0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
            },
        )
    }

    #[test]
    fn test_scene_graph_add_and_get() {
        let mut sg = SceneGraph::new();
        let obj = sample_object("cube");
        let id = obj.id;
        sg.add_object(obj);
        assert!(sg.get_object(&id).is_some());
        assert_eq!(sg.objects.len(), 1);
    }

    #[test]
    fn test_scene_graph_remove() {
        let mut sg = SceneGraph::new();
        let obj = sample_object("cube");
        let id = obj.id;
        sg.add_object(obj);
        let removed = sg.remove_object(&id);
        assert!(removed.is_some());
        assert!(sg.get_object(&id).is_none());
        assert_eq!(sg.objects.len(), 0);
    }

    #[test]
    fn test_scene_serialization_roundtrip() {
        let mut sg = SceneGraph::new();
        sg.add_object(sample_object("a"));
        sg.add_object(sample_object("b"));
        let json = serde_json::to_string(&sg).unwrap();
        let sg2: SceneGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(sg2.objects.len(), 2);
    }
}
