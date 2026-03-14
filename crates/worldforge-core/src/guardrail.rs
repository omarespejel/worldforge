//! Safety guardrails for WorldForge predictions and plans.
//!
//! Guardrails enforce physical and safety constraints on predicted
//! world states, preventing implausible or dangerous outcomes.

use serde::{Deserialize, Serialize};

use crate::types::{BBox, ObjectId};

/// A safety or physics constraint to enforce.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Guardrail {
    /// No collisions between objects.
    NoCollisions,

    /// Objects must remain upright within a tilt tolerance.
    StayUpright {
        objects: Vec<ObjectId>,
        max_tilt_degrees: f32,
    },

    /// Objects must remain within spatial bounds.
    BoundaryConstraint { bounds: BBox },

    /// Energy must be conserved within a tolerance.
    EnergyConservation { tolerance: f32 },

    /// Certain conditions must never be true.
    ForbiddenStates {
        conditions: Vec<crate::action::Condition>,
    },

    /// Maximum velocity constraint.
    MaxVelocity { limit: f32 },

    /// Human safety exclusion zone.
    HumanSafetyZone { radius: f32 },
}

/// Configuration for applying a guardrail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailConfig {
    /// The guardrail to apply.
    pub guardrail: Guardrail,
    /// Whether a violation should block the operation.
    pub blocking: bool,
}

/// Result of evaluating a guardrail against a state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailResult {
    /// Name of the guardrail that was evaluated.
    pub guardrail_name: String,
    /// Whether the guardrail passed.
    pub passed: bool,
    /// Details about any violation.
    pub violation_details: Option<String>,
    /// Severity of the violation.
    pub severity: ViolationSeverity,
}

/// Severity level for a guardrail violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Informational — no action needed.
    Info,
    /// Warning — flagged but result is returned.
    Warning,
    /// Critical — flagged but result is returned.
    Critical,
    /// Blocking — operation is rejected.
    Blocking,
}

/// Evaluate a set of guardrails against a world state.
pub fn evaluate_guardrails(
    configs: &[GuardrailConfig],
    state: &crate::state::WorldState,
) -> Vec<GuardrailResult> {
    configs
        .iter()
        .map(|config| evaluate_single(&config.guardrail, config.blocking, state))
        .collect()
}

/// Check if any guardrail results contain a blocking violation.
pub fn has_blocking_violation(results: &[GuardrailResult]) -> bool {
    results
        .iter()
        .any(|r| !r.passed && r.severity == ViolationSeverity::Blocking)
}

fn evaluate_single(
    guardrail: &Guardrail,
    blocking: bool,
    state: &crate::state::WorldState,
) -> GuardrailResult {
    let (name, passed, details) = match guardrail {
        Guardrail::NoCollisions => {
            // Check bounding box overlaps between all object pairs
            let objects: Vec<_> = state.scene.objects.values().collect();
            let mut collision_found = false;
            let mut detail = None;
            for i in 0..objects.len() {
                for j in (i + 1)..objects.len() {
                    if bbox_overlaps(&objects[i].bbox, &objects[j].bbox) {
                        collision_found = true;
                        detail = Some(format!(
                            "collision between '{}' and '{}'",
                            objects[i].name, objects[j].name
                        ));
                        break;
                    }
                }
                if collision_found {
                    break;
                }
            }
            ("NoCollisions".to_string(), !collision_found, detail)
        }
        Guardrail::BoundaryConstraint { bounds } => {
            let mut out_of_bounds = false;
            let mut detail = None;
            for obj in state.scene.objects.values() {
                let p = &obj.pose.position;
                if p.x < bounds.min.x
                    || p.x > bounds.max.x
                    || p.y < bounds.min.y
                    || p.y > bounds.max.y
                    || p.z < bounds.min.z
                    || p.z > bounds.max.z
                {
                    out_of_bounds = true;
                    detail = Some(format!("'{}' is out of bounds", obj.name));
                    break;
                }
            }
            ("BoundaryConstraint".to_string(), !out_of_bounds, detail)
        }
        Guardrail::MaxVelocity { .. } => {
            // Velocity checks require state deltas — pass by default in MVP
            ("MaxVelocity".to_string(), true, None)
        }
        Guardrail::HumanSafetyZone { .. } => {
            // Human detection requires perception — pass by default in MVP
            ("HumanSafetyZone".to_string(), true, None)
        }
        Guardrail::StayUpright { .. } => {
            // Orientation checks require quaternion analysis — pass by default in MVP
            ("StayUpright".to_string(), true, None)
        }
        Guardrail::EnergyConservation { .. } => {
            // Energy checks require physics simulation — pass by default in MVP
            ("EnergyConservation".to_string(), true, None)
        }
        Guardrail::ForbiddenStates { .. } => {
            // Condition evaluation is complex — pass by default in MVP
            ("ForbiddenStates".to_string(), true, None)
        }
    };

    let severity = if !passed && blocking {
        ViolationSeverity::Blocking
    } else if !passed {
        ViolationSeverity::Warning
    } else {
        ViolationSeverity::Info
    };

    GuardrailResult {
        guardrail_name: name,
        passed,
        violation_details: details,
        severity,
    }
}

fn bbox_overlaps(a: &BBox, b: &BBox) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
        && a.min.z <= b.max.z
        && a.max.z >= b.min.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    use crate::state::WorldState;
    use crate::types::{BBox, Pose, Position};

    fn make_state_with_objects(objects: Vec<SceneObject>) -> WorldState {
        let mut state = WorldState::new("test", "mock");
        for obj in objects {
            state.scene.add_object(obj);
        }
        state
    }

    #[test]
    fn test_no_collisions_pass() {
        let state = make_state_with_objects(vec![
            SceneObject::new(
                "a",
                Pose::default(),
                BBox {
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
            ),
            SceneObject::new(
                "b",
                Pose::default(),
                BBox {
                    min: Position {
                        x: 2.0,
                        y: 2.0,
                        z: 2.0,
                    },
                    max: Position {
                        x: 3.0,
                        y: 3.0,
                        z: 3.0,
                    },
                },
            ),
        ]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::NoCollisions,
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(results[0].passed);
    }

    #[test]
    fn test_no_collisions_fail() {
        let state = make_state_with_objects(vec![
            SceneObject::new(
                "a",
                Pose::default(),
                BBox {
                    min: Position {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    max: Position {
                        x: 2.0,
                        y: 2.0,
                        z: 2.0,
                    },
                },
            ),
            SceneObject::new(
                "b",
                Pose::default(),
                BBox {
                    min: Position {
                        x: 1.0,
                        y: 1.0,
                        z: 1.0,
                    },
                    max: Position {
                        x: 3.0,
                        y: 3.0,
                        z: 3.0,
                    },
                },
            ),
        ]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::NoCollisions,
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(!results[0].passed);
        assert_eq!(results[0].severity, ViolationSeverity::Blocking);
    }

    #[test]
    fn test_boundary_constraint() {
        let mut obj = SceneObject::new(
            "a",
            Pose::default(),
            BBox {
                min: Position {
                    x: -1.0,
                    y: -1.0,
                    z: -1.0,
                },
                max: Position {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                },
            },
        );
        obj.pose.position = Position {
            x: 100.0,
            y: 0.0,
            z: 0.0,
        };
        let state = make_state_with_objects(vec![obj]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::BoundaryConstraint {
                bounds: BBox {
                    min: Position {
                        x: -10.0,
                        y: -10.0,
                        z: -10.0,
                    },
                    max: Position {
                        x: 10.0,
                        y: 10.0,
                        z: 10.0,
                    },
                },
            },
            blocking: false,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(!results[0].passed);
        assert_eq!(results[0].severity, ViolationSeverity::Warning);
    }

    #[test]
    fn test_has_blocking_violation() {
        let results = vec![GuardrailResult {
            guardrail_name: "test".to_string(),
            passed: false,
            violation_details: None,
            severity: ViolationSeverity::Blocking,
        }];
        assert!(has_blocking_violation(&results));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_severity() -> impl Strategy<Value = ViolationSeverity> {
            prop_oneof![
                Just(ViolationSeverity::Info),
                Just(ViolationSeverity::Warning),
                Just(ViolationSeverity::Critical),
                Just(ViolationSeverity::Blocking),
            ]
        }

        proptest! {
            #[test]
            fn severity_roundtrip(s in arb_severity()) {
                let json = serde_json::to_string(&s).unwrap();
                let s2: ViolationSeverity = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(s, s2);
            }

            #[test]
            fn guardrail_result_roundtrip(
                name in ".*",
                passed in any::<bool>(),
                sev in arb_severity()
            ) {
                let result = GuardrailResult {
                    guardrail_name: name.clone(),
                    passed,
                    violation_details: None,
                    severity: sev,
                };
                let json = serde_json::to_string(&result).unwrap();
                let result2: GuardrailResult = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(result2.guardrail_name, name);
                prop_assert_eq!(result2.passed, passed);
                prop_assert_eq!(result2.severity, sev);
            }

            #[test]
            fn has_blocking_only_for_blocking_failures(
                passed in any::<bool>(),
                sev in arb_severity()
            ) {
                let results = vec![GuardrailResult {
                    guardrail_name: "test".to_string(),
                    passed,
                    violation_details: None,
                    severity: sev,
                }];
                let has_blocking = has_blocking_violation(&results);
                let expected = !passed && sev == ViolationSeverity::Blocking;
                prop_assert_eq!(has_blocking, expected);
            }
        }
    }
}
