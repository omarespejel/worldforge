//! Scene graph for spatial world representation.
//!
//! The scene graph is the primary data structure for representing
//! the spatial layout of objects in a world.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{BBox, Mesh, ObjectId, Pose, Position, Rotation, Tensor, Vec3, Velocity};

const RELATIONSHIP_EPSILON: f32 = 0.05;

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

/// Partial updates for an existing scene object.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneObjectPatch {
    /// Replacement human-readable name.
    pub name: Option<String>,
    /// Replacement world position.
    pub position: Option<Position>,
    /// Replacement axis-aligned bounding box.
    pub bbox: Option<BBox>,
    /// Replacement world rotation.
    pub rotation: Option<Rotation>,
    /// Replacement velocity vector.
    pub velocity: Option<Velocity>,
    /// Replacement 3D mesh geometry.
    pub mesh: Option<Mesh>,
    /// Replacement semantic label.
    pub semantic_label: Option<String>,
    /// Replacement provider-specific visual embedding.
    pub visual_embedding: Option<Tensor>,
    /// Replacement mass in kilograms.
    pub mass: Option<f32>,
    /// Replacement friction coefficient.
    pub friction: Option<f32>,
    /// Replacement restitution coefficient.
    pub restitution: Option<f32>,
    /// Replacement material name.
    pub material: Option<String>,
    /// Replacement immovable flag.
    pub is_static: Option<bool>,
    /// Replacement graspable flag.
    pub is_graspable: Option<bool>,
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
        self.sync_root_node(id, name);
        self.refresh_relationships();
    }

    /// Replace an existing object in the scene.
    ///
    /// Returns the previous object when the ID exists, or `None` when the
    /// scene does not contain the requested object.
    pub fn replace_object(&mut self, object: SceneObject) -> Option<SceneObject> {
        let id = object.id;
        let name = object.name.clone();

        if !self.objects.contains_key(&id) {
            return None;
        }

        let previous = self.objects.insert(id, object)?;
        self.sync_root_node(id, name);
        self.refresh_relationships();
        Some(previous)
    }

    /// Get an object by its ID.
    pub fn get_object(&self, id: &ObjectId) -> Option<&SceneObject> {
        self.objects.get(id)
    }

    /// Find an object by its human-readable name.
    pub fn find_object_by_name(&self, name: &str) -> Option<&SceneObject> {
        self.objects.values().find(|object| object.name == name)
    }

    /// Get a mutable reference to an object by its ID.
    pub fn get_object_mut(&mut self, id: &ObjectId) -> Option<&mut SceneObject> {
        self.objects.get_mut(id)
    }

    /// Find a mutable object by its human-readable name.
    pub fn find_object_by_name_mut(&mut self, name: &str) -> Option<&mut SceneObject> {
        self.objects.values_mut().find(|object| object.name == name)
    }

    /// Return objects sorted by name then ID for deterministic output.
    pub fn list_objects(&self) -> Vec<&SceneObject> {
        let mut objects: Vec<_> = self.objects.values().collect();
        objects.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
        });
        objects
    }

    /// Remove an object from the scene.
    pub fn remove_object(&mut self, id: &ObjectId) -> Option<SceneObject> {
        self.root
            .children
            .retain(|n| n.object_id.as_ref() != Some(id));
        let removed = self.objects.remove(id);
        self.refresh_relationships();
        removed
    }

    /// Update an object in the scene and refresh relationships.
    ///
    /// Returns the updated object when it exists, or `None` if the object ID is unknown.
    pub fn update_object(&mut self, id: &ObjectId, patch: SceneObjectPatch) -> Option<SceneObject> {
        let updated = {
            let object = self.objects.get_mut(id)?;
            object.apply_patch(&patch);
            object.clone()
        };

        self.sync_root_node(*id, updated.name.clone());

        self.refresh_relationships();
        Some(updated)
    }

    /// Set the world position for an object and recompute relationships.
    pub fn set_object_position(&mut self, id: &ObjectId, position: Position) -> bool {
        if let Some(object) = self.objects.get_mut(id) {
            object.set_position(position);
            self.refresh_relationships();
            true
        } else {
            false
        }
    }

    /// Translate an object by a delta and recompute relationships.
    pub fn translate_object(&mut self, id: &ObjectId, delta: Vec3) -> bool {
        if let Some(object) = self.objects.get_mut(id) {
            object.translate_by(delta);
            self.refresh_relationships();
            true
        } else {
            false
        }
    }

    /// Recompute spatial relationships from the current object geometry.
    pub fn refresh_relationships(&mut self) {
        self.relationships = infer_relationships(&self.objects);
    }

    fn sync_root_node(&mut self, id: ObjectId, name: String) {
        let mut found = false;
        self.root.children.retain(|node| {
            if node.object_id == Some(id) {
                if found {
                    false
                } else {
                    found = true;
                    true
                }
            } else {
                true
            }
        });

        if let Some(node) = self
            .root
            .children
            .iter_mut()
            .find(|node| node.object_id == Some(id))
        {
            node.name = name;
        } else {
            self.root.children.push(SceneNode {
                name,
                children: Vec::new(),
                object_id: Some(id),
            });
        }
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
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

    /// Return the half extents of this object's axis-aligned bounding box.
    pub fn half_extents(&self) -> Vec3 {
        self.bbox.size().scale(0.5)
    }

    /// Update the object's position while keeping its bounding box aligned.
    pub fn set_position(&mut self, position: Position) {
        let delta = Vec3 {
            x: position.x - self.pose.position.x,
            y: position.y - self.pose.position.y,
            z: position.z - self.pose.position.z,
        };
        self.pose.position = position;
        self.bbox.translate(delta);
    }

    /// Translate the object and its bounding box by a delta.
    pub fn translate_by(&mut self, delta: Vec3) {
        self.pose.position = self.pose.position.offset(delta);
        self.bbox.translate(delta);
    }

    fn apply_patch(&mut self, patch: &SceneObjectPatch) {
        if let Some(name) = &patch.name {
            self.name = name.clone();
        }
        if let Some(rotation) = patch.rotation {
            self.pose.rotation = rotation;
        }
        match (patch.position, patch.bbox) {
            (Some(position), Some(bbox)) => {
                self.pose.position = position;
                self.bbox = bbox;
            }
            (Some(position), None) => {
                self.set_position(position);
            }
            (None, Some(bbox)) => {
                self.bbox = bbox;
                self.pose.position = bbox.center();
            }
            (None, None) => {}
        }
        if let Some(velocity) = patch.velocity {
            self.velocity = velocity;
        }
        if let Some(mesh) = &patch.mesh {
            self.mesh = Some(mesh.clone());
        }
        if let Some(label) = &patch.semantic_label {
            self.semantic_label = Some(label.clone());
        }
        if let Some(visual_embedding) = &patch.visual_embedding {
            self.visual_embedding = Some(visual_embedding.clone());
        }
        if let Some(mass) = patch.mass {
            self.physics.mass = Some(mass);
        }
        if let Some(friction) = patch.friction {
            self.physics.friction = Some(friction);
        }
        if let Some(restitution) = patch.restitution {
            self.physics.restitution = Some(restitution);
        }
        if let Some(material) = &patch.material {
            self.physics.material = Some(material.clone());
        }
        if let Some(is_static) = patch.is_static {
            self.physics.is_static = is_static;
        }
        if let Some(is_graspable) = patch.is_graspable {
            self.physics.is_graspable = is_graspable;
        }
    }
}

fn infer_relationships(objects: &HashMap<ObjectId, SceneObject>) -> Vec<SpatialRelationship> {
    let mut ordered: Vec<_> = objects.values().collect();
    ordered.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });

    let mut relationships = Vec::new();
    for i in 0..ordered.len() {
        for j in (i + 1)..ordered.len() {
            let left = ordered[i];
            let right = ordered[j];

            if left.bbox.intersects_or_touches(&right.bbox) {
                relationships.push(SpatialRelationship::Touching {
                    a: left.id,
                    b: right.id,
                });
            }

            let distance = left.pose.position.distance(right.pose.position);
            let near_threshold =
                (left.half_extents().magnitude() + right.half_extents().magnitude()).max(0.25)
                    + 0.5;
            if distance <= near_threshold {
                relationships.push(SpatialRelationship::Near {
                    a: left.id,
                    b: right.id,
                    distance,
                });
            }

            if left.bbox.contains(&right.bbox) {
                relationships.push(SpatialRelationship::In {
                    subject: right.id,
                    container: left.id,
                });
            } else if right.bbox.contains(&left.bbox) {
                relationships.push(SpatialRelationship::In {
                    subject: left.id,
                    container: right.id,
                });
            }

            if is_on(left, right) {
                relationships.push(SpatialRelationship::On {
                    subject: left.id,
                    surface: right.id,
                });
            } else if is_on(right, left) {
                relationships.push(SpatialRelationship::On {
                    subject: right.id,
                    surface: left.id,
                });
            }

            if left.pose.position.y > right.pose.position.y + RELATIONSHIP_EPSILON {
                relationships.push(SpatialRelationship::Above {
                    subject: left.id,
                    reference: right.id,
                });
                relationships.push(SpatialRelationship::Below {
                    subject: right.id,
                    reference: left.id,
                });
            } else if right.pose.position.y > left.pose.position.y + RELATIONSHIP_EPSILON {
                relationships.push(SpatialRelationship::Above {
                    subject: right.id,
                    reference: left.id,
                });
                relationships.push(SpatialRelationship::Below {
                    subject: left.id,
                    reference: right.id,
                });
            }
        }
    }

    relationships
}

fn is_on(subject: &SceneObject, surface: &SceneObject) -> bool {
    let vertical_gap = (subject.bbox.min.y - surface.bbox.max.y).abs();
    vertical_gap <= RELATIONSHIP_EPSILON
        && horizontal_overlap(&subject.bbox, &surface.bbox)
        && subject.pose.position.y >= surface.pose.position.y
}

fn horizontal_overlap(a: &BBox, b: &BBox) -> bool {
    a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.z <= b.max.z && a.max.z >= b.min.z
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
        assert_eq!(sg.root.children.len(), 1);
    }

    #[test]
    fn test_scene_graph_add_object_deduplicates_root_children_for_existing_id() {
        let mut sg = SceneGraph::new();
        let original = sample_object("cube");
        let id = original.id;
        sg.add_object(original);

        let mut replacement = sample_object("cube_renamed");
        replacement.id = id;
        replacement.pose.position = Position {
            x: 3.0,
            y: 0.5,
            z: -1.0,
        };

        sg.add_object(replacement);

        assert_eq!(sg.objects.len(), 1);
        assert_eq!(sg.root.children.len(), 1);
        assert_eq!(sg.root.children[0].name, "cube_renamed");
        assert_eq!(sg.get_object(&id).unwrap().name, "cube_renamed");
    }

    #[test]
    fn test_scene_graph_replace_object_updates_hierarchy_and_relationships() {
        let mut sg = SceneGraph::new();
        let table = sample_object("table");
        let table_id = table.id;
        let mut mug = sample_object("mug");
        mug.pose.position = Position {
            x: 0.0,
            y: 0.55,
            z: 0.0,
        };
        mug.bbox = BBox {
            min: Position {
                x: -0.25,
                y: 0.05,
                z: -0.25,
            },
            max: Position {
                x: 0.25,
                y: 0.55,
                z: 0.25,
            },
        };

        sg.add_object(table);
        sg.add_object(mug);
        assert!(!sg.relationships.is_empty());

        let mut replacement = sample_object("table_updated");
        replacement.id = table_id;
        replacement.pose.position = Position {
            x: 10.0,
            y: 0.5,
            z: 0.0,
        };
        replacement.bbox = BBox {
            min: Position {
                x: 9.5,
                y: 0.0,
                z: -0.5,
            },
            max: Position {
                x: 10.5,
                y: 1.0,
                z: 0.5,
            },
        };

        let previous = sg.replace_object(replacement).expect("replacement");
        assert_eq!(previous.name, "table");
        assert_eq!(sg.objects.len(), 2);
        assert_eq!(sg.root.children.len(), 2);
        assert_eq!(sg.root.children[0].name, "table_updated");
        assert!(sg.relationships.is_empty());
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

    #[test]
    fn test_scene_graph_find_by_name_and_sorted_listing() {
        let mut sg = SceneGraph::new();
        let zebra = sample_object("zebra");
        let apple = sample_object("apple");
        let apple_id = apple.id;
        sg.add_object(zebra);
        sg.add_object(apple);

        assert_eq!(
            sg.find_object_by_name("apple").map(|object| object.id),
            Some(apple_id)
        );

        let listed = sg.list_objects();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].name, "apple");
        assert_eq!(listed[1].name, "zebra");
    }

    #[test]
    fn test_scene_graph_find_object_by_name_mut() {
        let mut sg = SceneGraph::new();
        sg.add_object(sample_object("cube"));

        let object = sg.find_object_by_name_mut("cube").unwrap();
        object.semantic_label = Some("block".to_string());

        assert_eq!(
            sg.find_object_by_name("cube")
                .and_then(|object| object.semantic_label.as_deref()),
            Some("block")
        );
    }

    #[test]
    fn test_scene_object_set_position_updates_bbox() {
        let mut object = sample_object("cube");
        object.set_position(Position {
            x: 2.0,
            y: 1.0,
            z: -1.0,
        });

        assert_eq!(object.pose.position.x, 2.0);
        assert_eq!(object.bbox.center().x, 2.0);
        assert_eq!(object.bbox.center().y, 1.0);
        assert_eq!(object.bbox.center().z, -1.0);
    }

    #[test]
    fn test_scene_graph_update_object_translates_bbox_with_position() {
        let mut sg = SceneGraph::new();
        let object = sample_object("cube");
        let id = object.id;
        sg.add_object(object);

        let updated = sg
            .update_object(
                &id,
                SceneObjectPatch {
                    position: Some(Position {
                        x: 2.0,
                        y: 1.0,
                        z: -1.0,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(updated.pose.position.x, 2.0);
        assert_eq!(updated.bbox.center().x, 2.0);
        assert_eq!(updated.bbox.min.x, 1.5);
        assert_eq!(updated.bbox.max.z, -0.5);
        assert_eq!(sg.root.children[0].name, "cube");
    }

    #[test]
    fn test_scene_graph_update_object_snaps_pose_to_bbox_center() {
        let mut sg = SceneGraph::new();
        let object = sample_object("cube");
        let id = object.id;
        sg.add_object(object);

        let updated = sg
            .update_object(
                &id,
                SceneObjectPatch {
                    bbox: Some(BBox {
                        min: Position {
                            x: 1.0,
                            y: 2.0,
                            z: 3.0,
                        },
                        max: Position {
                            x: 5.0,
                            y: 6.0,
                            z: 7.0,
                        },
                    }),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(updated.pose.position.x, 3.0);
        assert_eq!(updated.pose.position.y, 4.0);
        assert_eq!(updated.pose.position.z, 5.0);
        assert_eq!(updated.bbox.min.x, 1.0);
        assert_eq!(updated.bbox.max.z, 7.0);
    }

    #[test]
    fn test_scene_graph_update_object_applies_mesh_and_visual_embedding() {
        let mut sg = SceneGraph::new();
        let object = sample_object("cube");
        let id = object.id;
        let original_bbox = object.bbox;
        let original_pose = object.pose;
        sg.add_object(object);

        let mesh = Mesh {
            vertices: vec![
                Position {
                    x: -0.5,
                    y: 0.0,
                    z: -0.5,
                },
                Position {
                    x: 0.5,
                    y: 0.0,
                    z: -0.5,
                },
                Position {
                    x: 0.0,
                    y: 0.5,
                    z: 0.5,
                },
            ],
            faces: vec![[0, 1, 2]],
            normals: Some(vec![
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ]),
            uvs: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        };
        let visual_embedding = Tensor {
            data: crate::types::TensorData::Float32(vec![0.1, 0.2, 0.3, 0.4]),
            shape: vec![4],
            dtype: crate::types::DType::Float32,
            device: crate::types::Device::Cpu,
        };

        let updated = sg
            .update_object(
                &id,
                SceneObjectPatch {
                    mesh: Some(mesh.clone()),
                    visual_embedding: Some(visual_embedding.clone()),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(updated.pose, original_pose);
        assert_eq!(updated.bbox, original_bbox);
        let updated_mesh = updated.mesh.as_ref().expect("mesh should be set");
        assert_eq!(updated_mesh.vertices, mesh.vertices);
        assert_eq!(updated_mesh.faces, mesh.faces);
        assert_eq!(updated_mesh.normals, mesh.normals);
        assert_eq!(updated_mesh.uvs, mesh.uvs);

        let updated_embedding = updated
            .visual_embedding
            .as_ref()
            .expect("visual embedding should be set");
        assert_eq!(updated_embedding.shape, visual_embedding.shape);
        assert_eq!(updated_embedding.dtype, visual_embedding.dtype);
        assert_eq!(updated_embedding.device, visual_embedding.device);
        match (&updated_embedding.data, &visual_embedding.data) {
            (crate::types::TensorData::Float32(lhs), crate::types::TensorData::Float32(rhs)) => {
                assert_eq!(lhs, rhs)
            }
            other => panic!("unexpected tensor data variants: {other:?}"),
        }

        let stored = sg.get_object(&id).unwrap();
        assert_eq!(stored.mesh.as_ref().unwrap().faces, mesh.faces);
        assert_eq!(stored.visual_embedding.as_ref().unwrap().shape, vec![4]);
    }

    #[test]
    fn test_scene_graph_refresh_relationships_detects_geometry() {
        let mut sg = SceneGraph::new();

        let table = SceneObject::new(
            "table",
            Pose::default(),
            BBox {
                min: Position {
                    x: -1.0,
                    y: -0.5,
                    z: -1.0,
                },
                max: Position {
                    x: 1.0,
                    y: 0.5,
                    z: 1.0,
                },
            },
        );
        let table_id = table.id;

        let mug = SceneObject::new(
            "mug",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.6,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: -0.1,
                    y: 0.5,
                    z: -0.1,
                },
                max: Position {
                    x: 0.1,
                    y: 0.7,
                    z: 0.1,
                },
            },
        );
        let mug_id = mug.id;

        let crate_box = SceneObject::new(
            "crate",
            Pose {
                position: Position {
                    x: 2.0,
                    y: 0.5,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: 1.0,
                    y: 0.0,
                    z: -0.5,
                },
                max: Position {
                    x: 3.0,
                    y: 1.0,
                    z: 0.5,
                },
            },
        );
        let crate_id = crate_box.id;

        let ball = SceneObject::new(
            "ball",
            Pose {
                position: Position {
                    x: 2.0,
                    y: 0.5,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: 1.6,
                    y: 0.2,
                    z: -0.2,
                },
                max: Position {
                    x: 2.4,
                    y: 0.8,
                    z: 0.2,
                },
            },
        );
        let ball_id = ball.id;

        sg.add_object(table);
        sg.add_object(mug);
        sg.add_object(crate_box);
        sg.add_object(ball);

        assert!(sg.relationships.iter().any(|relationship| {
            matches!(
                relationship,
                SpatialRelationship::On { subject, surface }
                    if *subject == mug_id && *surface == table_id
            )
        }));
        assert!(sg.relationships.iter().any(|relationship| {
            matches!(
                relationship,
                SpatialRelationship::Above { subject, reference }
                    if *subject == mug_id && *reference == table_id
            )
        }));
        assert!(sg.relationships.iter().any(|relationship| {
            matches!(
                relationship,
                SpatialRelationship::In { subject, container }
                    if *subject == ball_id && *container == crate_id
            )
        }));
        assert!(sg.relationships.iter().any(|relationship| {
            matches!(
                relationship,
                SpatialRelationship::Touching { a, b }
                    if (*a == ball_id && *b == crate_id) || (*a == crate_id && *b == ball_id)
            )
        }));
    }
}
