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
}
