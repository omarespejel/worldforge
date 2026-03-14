//! Action type system for WorldForge.
//!
//! Actions represent operations that can be performed on a world state,
//! ranging from robot manipulation to environment modifications.

use serde::{Deserialize, Serialize};

use crate::types::{ObjectId, Pose, Position, Vec3};

/// An action that modifies world state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    // -- Robot manipulation --
    /// Move an end-effector to a target position.
    Move { target: Position, speed: f32 },
    /// Grasp an object with a given grip force.
    Grasp { object: ObjectId, grip_force: f32 },
    /// Release a grasped object.
    Release { object: ObjectId },
    /// Push an object in a direction with force.
    Push {
        object: ObjectId,
        direction: Vec3,
        force: f32,
    },
    /// Rotate an object around an axis.
    Rotate {
        object: ObjectId,
        axis: Vec3,
        angle: f32,
    },
    /// Place an object at a target position.
    Place { object: ObjectId, target: Position },

    // -- Camera / navigation --
    /// Move camera by a relative pose delta.
    CameraMove { delta: Pose },
    /// Point camera at a target position.
    CameraLookAt { target: Position },
    /// Navigate through a sequence of waypoints.
    Navigate { waypoints: Vec<Position> },
    /// Instantly teleport to a destination.
    Teleport { destination: Pose },

    // -- Environment --
    /// Change the weather conditions.
    SetWeather { weather: Weather },
    /// Change lighting by setting time of day (0.0–24.0).
    SetLighting { time_of_day: f32 },
    /// Spawn a new object from a template.
    SpawnObject { template: String, pose: Pose },
    /// Remove an object from the scene.
    RemoveObject { object: ObjectId },

    // -- Compound actions --
    /// Execute actions in sequence.
    Sequence(Vec<Action>),
    /// Execute actions in parallel.
    Parallel(Vec<Action>),
    /// Conditional action execution.
    Conditional {
        condition: Condition,
        then: Box<Action>,
        otherwise: Option<Box<Action>>,
    },

    // -- Raw provider-specific --
    /// Provider-specific action with raw JSON payload.
    Raw {
        provider: String,
        data: serde_json::Value,
    },
}

/// Weather conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Weather {
    Clear,
    Cloudy,
    Rain,
    Snow,
    Fog,
    Night,
}

/// A condition that can be evaluated against world state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    /// Check if an object is at a position (within tolerance).
    ObjectAt {
        object: ObjectId,
        position: Position,
        tolerance: f32,
    },
    /// Check if two objects are in contact.
    ObjectsTouching { a: ObjectId, b: ObjectId },
    /// Check if an object exists in the scene.
    ObjectExists { object: ObjectId },
    /// Logical AND of conditions.
    And(Vec<Condition>),
    /// Logical OR of conditions.
    Or(Vec<Condition>),
    /// Logical NOT of a condition.
    Not(Box<Condition>),
}

/// The type of action space used by a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionSpaceType {
    /// Continuous action vectors.
    Continuous,
    /// Discrete action indices.
    Discrete,
    /// Text/language-based actions.
    Language,
    /// Video/image-based action specification.
    Visual,
}

/// A provider-specific action representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderAction {
    /// Provider name.
    pub provider: String,
    /// Serialized action data.
    pub data: serde_json::Value,
}

/// Trait for translating WorldForge actions to provider-specific format.
pub trait ActionTranslator {
    /// Translate a WorldForge action to a provider-specific action.
    fn translate(&self, action: &Action) -> crate::error::Result<ProviderAction>;

    /// List of action types supported by this translator.
    fn supported_actions(&self) -> Vec<ActionSpaceType>;
}

/// Evaluate a condition against a world state.
pub fn evaluate_condition(condition: &Condition, state: &crate::state::WorldState) -> bool {
    match condition {
        Condition::ObjectAt {
            object,
            position,
            tolerance,
        } => {
            if let Some(obj) = state.scene.get_object(object) {
                let dx = obj.pose.position.x - position.x;
                let dy = obj.pose.position.y - position.y;
                let dz = obj.pose.position.z - position.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                dist <= *tolerance
            } else {
                false
            }
        }
        Condition::ObjectsTouching { a, b } => state
            .scene
            .relationships
            .iter()
            .any(|r| matches!(r, crate::scene::SpatialRelationship::Touching { a: ra, b: rb } if (ra == a && rb == b) || (ra == b && rb == a))),
        Condition::ObjectExists { object } => state.scene.get_object(object).is_some(),
        Condition::And(conditions) => conditions.iter().all(|c| evaluate_condition(c, state)),
        Condition::Or(conditions) => conditions.iter().any(|c| evaluate_condition(c, state)),
        Condition::Not(inner) => !evaluate_condition(inner, state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_action_move_serialization() {
        let action = Action::Move {
            target: Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            speed: 0.5,
        };
        let json = serde_json::to_string(&action).unwrap();
        let action2: Action = serde_json::from_str(&json).unwrap();
        match action2 {
            Action::Move { target, speed } => {
                assert_eq!(target.x, 1.0);
                assert_eq!(speed, 0.5);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_compound_action_serialization() {
        let action = Action::Sequence(vec![
            Action::Move {
                target: Position::default(),
                speed: 1.0,
            },
            Action::SetWeather {
                weather: Weather::Rain,
            },
        ]);
        let json = serde_json::to_string(&action).unwrap();
        let action2: Action = serde_json::from_str(&json).unwrap();
        match action2 {
            Action::Sequence(actions) => assert_eq!(actions.len(), 2),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_conditional_action() {
        let obj_id = uuid::Uuid::new_v4();
        let action = Action::Conditional {
            condition: Condition::ObjectExists { object: obj_id },
            then: Box::new(Action::Release { object: obj_id }),
            otherwise: None,
        };
        let json = serde_json::to_string(&action).unwrap();
        let _: Action = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_evaluate_condition_object_exists() {
        let mut state = crate::state::WorldState::new("test", "mock");
        let obj = crate::scene::SceneObject::new(
            "cube",
            crate::types::Pose::default(),
            crate::types::BBox {
                min: crate::types::Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                max: crate::types::Position {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                },
            },
        );
        let id = obj.id;
        state.scene.add_object(obj);

        assert!(evaluate_condition(
            &Condition::ObjectExists { object: id },
            &state
        ));
        assert!(!evaluate_condition(
            &Condition::ObjectExists {
                object: uuid::Uuid::new_v4()
            },
            &state
        ));
    }

    #[test]
    fn test_evaluate_condition_object_at() {
        let mut state = crate::state::WorldState::new("test", "mock");
        let obj = crate::scene::SceneObject::new(
            "cube",
            crate::types::Pose {
                position: Position {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                },
                ..crate::types::Pose::default()
            },
            crate::types::BBox {
                min: Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                max: Position {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                },
            },
        );
        let id = obj.id;
        state.scene.add_object(obj);

        assert!(evaluate_condition(
            &Condition::ObjectAt {
                object: id,
                position: Position {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0
                },
                tolerance: 0.1,
            },
            &state
        ));
        assert!(!evaluate_condition(
            &Condition::ObjectAt {
                object: id,
                position: Position {
                    x: 10.0,
                    y: 0.0,
                    z: 0.0
                },
                tolerance: 0.1,
            },
            &state
        ));
    }

    #[test]
    fn test_evaluate_condition_and_or_not() {
        let state = crate::state::WorldState::new("test", "mock");
        let fake_id = uuid::Uuid::new_v4();

        // NOT(ObjectExists(fake)) => true
        assert!(evaluate_condition(
            &Condition::Not(Box::new(Condition::ObjectExists { object: fake_id })),
            &state
        ));

        // AND([NOT(exists), NOT(exists)]) => true
        assert!(evaluate_condition(
            &Condition::And(vec![
                Condition::Not(Box::new(Condition::ObjectExists { object: fake_id })),
                Condition::Not(Box::new(Condition::ObjectExists {
                    object: uuid::Uuid::new_v4()
                })),
            ]),
            &state
        ));

        // OR([exists, NOT(exists)]) => true
        assert!(evaluate_condition(
            &Condition::Or(vec![
                Condition::ObjectExists { object: fake_id },
                Condition::Not(Box::new(Condition::ObjectExists { object: fake_id })),
            ]),
            &state
        ));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_weather() -> impl Strategy<Value = Weather> {
            prop_oneof![
                Just(Weather::Clear),
                Just(Weather::Cloudy),
                Just(Weather::Rain),
                Just(Weather::Snow),
                Just(Weather::Fog),
                Just(Weather::Night),
            ]
        }

        fn arb_action_space_type() -> impl Strategy<Value = ActionSpaceType> {
            prop_oneof![
                Just(ActionSpaceType::Continuous),
                Just(ActionSpaceType::Discrete),
                Just(ActionSpaceType::Language),
                Just(ActionSpaceType::Visual),
            ]
        }

        proptest! {
            #[test]
            fn weather_roundtrip(w in arb_weather()) {
                let json = serde_json::to_string(&w).unwrap();
                let w2: Weather = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(w, w2);
            }

            #[test]
            fn action_space_type_roundtrip(ast in arb_action_space_type()) {
                let json = serde_json::to_string(&ast).unwrap();
                let ast2: ActionSpaceType = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(ast, ast2);
            }

            #[test]
            fn move_action_roundtrip(x in prop::num::f32::NORMAL, y in prop::num::f32::NORMAL, z in prop::num::f32::NORMAL, speed in prop::num::f32::NORMAL) {
                let action = Action::Move {
                    target: Position { x, y, z },
                    speed,
                };
                let json = serde_json::to_string(&action).unwrap();
                let action2: Action = serde_json::from_str(&json).unwrap();
                match action2 {
                    Action::Move { target, speed: s } => {
                        prop_assert_eq!(target.x, x);
                        prop_assert_eq!(target.y, y);
                        prop_assert_eq!(target.z, z);
                        prop_assert_eq!(s, speed);
                    }
                    _ => prop_assert!(false, "wrong variant"),
                }
            }

            #[test]
            fn set_weather_roundtrip(w in arb_weather()) {
                let action = Action::SetWeather { weather: w };
                let json = serde_json::to_string(&action).unwrap();
                let action2: Action = serde_json::from_str(&json).unwrap();
                match action2 {
                    Action::SetWeather { weather } => prop_assert_eq!(weather, w),
                    _ => prop_assert!(false, "wrong variant"),
                }
            }
        }
    }
}
