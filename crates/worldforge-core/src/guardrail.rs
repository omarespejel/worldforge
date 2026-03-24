//! Safety guardrails for WorldForge predictions and plans.
//!
//! Guardrails enforce physical and safety constraints on predicted
//! world states, preventing implausible or dangerous outcomes.

use serde::{Deserialize, Serialize};

use crate::types::{BBox, ObjectId};

const DEFAULT_ENERGY_TOLERANCE_JOULES: f32 = 5_000.0;

/// A safety or physics constraint to enforce.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Guardrail {
    /// Disable all guardrail evaluation for an operation.
    Disabled,

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
    resolve_guardrails(configs)
        .iter()
        .map(|config| evaluate_single(&config.guardrail, config.blocking, state))
        .collect()
}

/// Return the core default guardrails used when a request omits guardrails.
pub fn default_guardrails() -> Vec<GuardrailConfig> {
    vec![
        GuardrailConfig {
            guardrail: Guardrail::NoCollisions,
            blocking: true,
        },
        GuardrailConfig {
            guardrail: Guardrail::EnergyConservation {
                tolerance: DEFAULT_ENERGY_TOLERANCE_JOULES,
            },
            blocking: true,
        },
    ]
}

/// Resolve guardrail configs into the effective set used by core evaluation.
///
/// Empty input uses the default guardrail set. The explicit
/// [`Guardrail::Disabled`] sentinel disables guardrail evaluation entirely.
pub fn resolve_guardrails(configs: &[GuardrailConfig]) -> Vec<GuardrailConfig> {
    if configs
        .iter()
        .any(|config| matches!(config.guardrail, Guardrail::Disabled))
    {
        Vec::new()
    } else if configs.is_empty() {
        default_guardrails()
    } else {
        configs.to_vec()
    }
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
        Guardrail::Disabled => ("Disabled".to_string(), true, None),
        Guardrail::NoCollisions => {
            // Check bounding box overlaps between all object pairs
            let objects: Vec<_> = state.scene.objects.values().collect();
            let mut collision_found = false;
            let mut detail = None;
            for i in 0..objects.len() {
                for j in (i + 1)..objects.len() {
                    if bbox_intersects(&objects[i].bbox, &objects[j].bbox) {
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
        Guardrail::MaxVelocity { limit } => {
            let mut violation = false;
            let mut detail = None;
            for obj in state.scene.objects.values() {
                let speed = obj.velocity.magnitude();
                if speed > *limit {
                    violation = true;
                    detail = Some(format!(
                        "'{}' velocity {:.2} exceeds limit {:.2}",
                        obj.name, speed, limit
                    ));
                    break;
                }
            }
            ("MaxVelocity".to_string(), !violation, detail)
        }
        Guardrail::HumanSafetyZone { radius } => {
            // Find objects tagged as "human" and check that all other objects
            // maintain the required safety distance from them.
            let humans: Vec<_> = state
                .scene
                .objects
                .values()
                .filter(|o| {
                    o.semantic_label
                        .as_deref()
                        .map(|l| {
                            l.eq_ignore_ascii_case("human") || l.eq_ignore_ascii_case("person")
                        })
                        .unwrap_or(false)
                })
                .collect();
            let mut violation = false;
            let mut detail = None;
            'outer: for human in &humans {
                let hp = &human.pose.position;
                for obj in state.scene.objects.values() {
                    if obj.id == human.id {
                        continue;
                    }
                    let dx = obj.pose.position.x - hp.x;
                    let dy = obj.pose.position.y - hp.y;
                    let dz = obj.pose.position.z - hp.z;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < *radius {
                        violation = true;
                        detail = Some(format!(
                            "'{}' is {:.2}m from human '{}', safety radius is {:.2}m",
                            obj.name, dist, human.name, radius
                        ));
                        break 'outer;
                    }
                }
            }
            ("HumanSafetyZone".to_string(), !violation, detail)
        }
        Guardrail::StayUpright {
            objects,
            max_tilt_degrees,
        } => {
            let mut violation = false;
            let mut detail = None;
            for obj_id in objects {
                if let Some(obj) = state.scene.get_object(obj_id) {
                    let tilt = obj.pose.rotation.tilt_degrees();
                    if tilt > *max_tilt_degrees {
                        violation = true;
                        detail = Some(format!(
                            "'{}' tilted {:.1}° (max {:.1}°)",
                            obj.name, tilt, max_tilt_degrees
                        ));
                        break;
                    }
                }
            }
            ("StayUpright".to_string(), !violation, detail)
        }
        Guardrail::EnergyConservation { tolerance } => {
            // Compare total kinetic energy across objects.
            // Since we only have a single state snapshot, we compute
            // total KE and flag if any object has implausibly high energy
            // relative to the scene total. A more complete implementation
            // would compare input vs output states.
            let total_ke: f32 = state
                .scene
                .objects
                .values()
                .map(|obj| {
                    let mass = obj.physics.mass.unwrap_or(1.0);
                    let v2 = obj.velocity.magnitude().powi(2);
                    0.5 * mass * v2
                })
                .sum();

            // Flag if total KE exceeds a reasonable bound.
            // Using tolerance as the max allowed total KE in joules.
            let violation = total_ke > *tolerance;
            let detail = if violation {
                Some(format!(
                    "total kinetic energy {:.2}J exceeds tolerance {:.2}J",
                    total_ke, tolerance
                ))
            } else {
                None
            };
            ("EnergyConservation".to_string(), !violation, detail)
        }
        Guardrail::ForbiddenStates { conditions } => {
            use crate::action::evaluate_condition;
            let mut violation = false;
            let mut detail = None;
            for (i, cond) in conditions.iter().enumerate() {
                if evaluate_condition(cond, state) {
                    violation = true;
                    detail = Some(format!("forbidden condition #{} is satisfied", i));
                    break;
                }
            }
            ("ForbiddenStates".to_string(), !violation, detail)
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

fn bbox_intersects(a: &BBox, b: &BBox) -> bool {
    a.intersects(b)
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
    fn test_no_collisions_touching_faces_pass() {
        let state = make_state_with_objects(vec![
            SceneObject::new(
                "left",
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
                "right",
                Pose::default(),
                BBox {
                    min: Position {
                        x: 1.0,
                        y: 0.25,
                        z: 0.25,
                    },
                    max: Position {
                        x: 2.0,
                        y: 0.75,
                        z: 0.75,
                    },
                },
            ),
        ]);
        let results = evaluate_guardrails(
            &[GuardrailConfig {
                guardrail: Guardrail::NoCollisions,
                blocking: true,
            }],
            &state,
        );

        assert!(results[0].passed);
    }

    #[test]
    fn test_default_guardrails_include_collision_and_energy() {
        let defaults = default_guardrails();
        assert_eq!(defaults.len(), 2);
        assert!(matches!(defaults[0].guardrail, Guardrail::NoCollisions));
        assert!(matches!(
            defaults[1].guardrail,
            Guardrail::EnergyConservation { tolerance }
                if (tolerance - DEFAULT_ENERGY_TOLERANCE_JOULES).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn test_resolve_guardrails_empty_uses_defaults() {
        let resolved = resolve_guardrails(&[]);
        assert_eq!(resolved.len(), default_guardrails().len());
    }

    #[test]
    fn test_resolve_guardrails_disabled_returns_empty() {
        let resolved = resolve_guardrails(&[GuardrailConfig {
            guardrail: Guardrail::Disabled,
            blocking: false,
        }]);

        assert!(resolved.is_empty());
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
    fn test_max_velocity_pass() {
        let state = make_state_with_objects(vec![SceneObject::new(
            "ball",
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
        )]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::MaxVelocity { limit: 10.0 },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(results[0].passed);
    }

    #[test]
    fn test_max_velocity_fail() {
        let mut obj = SceneObject::new(
            "rocket",
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
        obj.velocity = crate::types::Velocity {
            x: 10.0,
            y: 10.0,
            z: 10.0,
        };
        let state = make_state_with_objects(vec![obj]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::MaxVelocity { limit: 5.0 },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(!results[0].passed);
        assert!(results[0]
            .violation_details
            .as_ref()
            .unwrap()
            .contains("rocket"));
    }

    #[test]
    fn test_stay_upright_pass() {
        let obj = SceneObject::new(
            "mug",
            Pose::default(), // identity rotation = upright
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
        let id = obj.id;
        let state = make_state_with_objects(vec![obj]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::StayUpright {
                objects: vec![id],
                max_tilt_degrees: 10.0,
            },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(results[0].passed);
    }

    #[test]
    fn test_stay_upright_fail() {
        use crate::types::Rotation;
        // 90 degree rotation around Z axis
        let angle = std::f32::consts::FRAC_PI_2;
        let mut obj = SceneObject::new(
            "cup",
            Pose {
                position: Position::default(),
                rotation: Rotation {
                    w: (angle / 2.0).cos(),
                    x: 0.0,
                    y: 0.0,
                    z: (angle / 2.0).sin(),
                },
            },
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
        let id = obj.id;
        // The rotation above doesn't tilt around X, so let's use a tilt around X
        let tilt_angle = std::f32::consts::FRAC_PI_4; // 45 degrees
        obj.pose.rotation = Rotation {
            w: (tilt_angle / 2.0).cos(),
            x: (tilt_angle / 2.0).sin(),
            y: 0.0,
            z: 0.0,
        };
        let state = make_state_with_objects(vec![obj]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::StayUpright {
                objects: vec![id],
                max_tilt_degrees: 10.0,
            },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(!results[0].passed);
    }

    #[test]
    fn test_energy_conservation_pass() {
        let state = make_state_with_objects(vec![SceneObject::new(
            "ball",
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
        )]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::EnergyConservation { tolerance: 100.0 },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(results[0].passed);
    }

    #[test]
    fn test_energy_conservation_fail() {
        let mut obj = SceneObject::new(
            "cannonball",
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
        obj.velocity = crate::types::Velocity {
            x: 100.0,
            y: 0.0,
            z: 0.0,
        };
        obj.physics.mass = Some(10.0);
        let state = make_state_with_objects(vec![obj]);
        // KE = 0.5 * 10 * 100^2 = 50000
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::EnergyConservation { tolerance: 1000.0 },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(!results[0].passed);
    }

    #[test]
    fn test_forbidden_states_pass() {
        let fake_id = uuid::Uuid::new_v4();
        let state = make_state_with_objects(vec![]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::ForbiddenStates {
                conditions: vec![crate::action::Condition::ObjectExists { object: fake_id }],
            },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        // Object doesn't exist, so forbidden condition is NOT satisfied => passes
        assert!(results[0].passed);
    }

    #[test]
    fn test_forbidden_states_fail() {
        let obj = SceneObject::new(
            "bomb",
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
        let id = obj.id;
        let state = make_state_with_objects(vec![obj]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::ForbiddenStates {
                conditions: vec![crate::action::Condition::ObjectExists { object: id }],
            },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        // Object exists => forbidden condition IS satisfied => fails
        assert!(!results[0].passed);
    }

    #[test]
    fn test_human_safety_zone_pass() {
        let mut human = SceneObject::new(
            "person_1",
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
        );
        human.semantic_label = Some("human".to_string());

        let mut robot = SceneObject::new(
            "robot_arm",
            Pose {
                position: Position {
                    x: 10.0,
                    y: 0.0,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: 9.0,
                    y: -0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 11.0,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );
        robot.semantic_label = Some("robot".to_string());

        let state = make_state_with_objects(vec![human, robot]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::HumanSafetyZone { radius: 2.0 },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(results[0].passed);
    }

    #[test]
    fn test_human_safety_zone_fail() {
        let mut human = SceneObject::new(
            "person_1",
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
        );
        human.semantic_label = Some("human".to_string());

        let robot = SceneObject::new(
            "robot_arm",
            Pose {
                position: Position {
                    x: 0.5,
                    y: 0.0,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: 0.0,
                    y: -0.5,
                    z: -0.5,
                },
                max: Position {
                    x: 1.0,
                    y: 0.5,
                    z: 0.5,
                },
            },
        );

        let state = make_state_with_objects(vec![human, robot]);
        let configs = vec![GuardrailConfig {
            guardrail: Guardrail::HumanSafetyZone { radius: 2.0 },
            blocking: true,
        }];
        let results = evaluate_guardrails(&configs, &state);
        assert!(!results[0].passed);
        assert!(results[0]
            .violation_details
            .as_ref()
            .unwrap()
            .contains("robot_arm"));
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
