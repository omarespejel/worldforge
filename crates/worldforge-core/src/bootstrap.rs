//! Prompt-based world bootstrap helpers.
//!
//! These utilities turn a natural-language scene prompt into a deterministic
//! initial `WorldState` with coarse object geometry and metadata.

use std::collections::HashMap;

use crate::error::{Result, WorldForgeError};
use crate::scene::SceneObject;
use crate::state::WorldState;
use crate::types::{BBox, Pose, Position, Rotation, Vec3};

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "around", "at", "beside", "by", "for", "from", "in", "inside", "into",
    "next", "of", "on", "onto", "or", "scene", "the", "to", "with", "world",
];

const ENVIRONMENT_KEYWORDS: &[(&str, &str)] = &[
    ("factory", "pallet"),
    ("kitchen", "counter"),
    ("lab", "bench"),
    ("laboratory", "bench"),
    ("office", "desk"),
    ("studio", "table"),
    ("warehouse", "pallet"),
    ("workshop", "workbench"),
];

const NON_OBJECT_KEYWORDS: &[&str] = &[
    "camera",
    "corner",
    "environment",
    "frame",
    "left",
    "middle",
    "right",
    "room",
    "space",
];

/// Deterministically seed a world state from a natural-language prompt.
///
/// The seeded state stores the prompt in metadata, derives a stable world
/// name when one is not provided, materializes a coarse support surface, and
/// places a small set of objects inferred from the prompt text.
pub fn seed_world_state_from_prompt(
    prompt: &str,
    provider: &str,
    name_override: Option<&str>,
) -> Result<WorldState> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err(WorldForgeError::InvalidState(
            "prompt cannot be empty".to_string(),
        ));
    }

    let tokens = normalized_tokens(prompt);
    let support_label = support_label_from_tokens(&tokens);
    let object_labels = object_labels_from_tokens(&tokens, support_label);

    let world_name = name_override
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| derive_world_name_from_prompt(prompt));

    let mut state = WorldState::new(world_name, provider);
    state.metadata.description = prompt.to_string();
    state.metadata.tags = metadata_tags(&tokens, support_label, &object_labels);

    let support_anchor = support_label.map(|label| {
        let surface = support_surface(label);
        let pose = Pose {
            position: surface.center,
            rotation: Rotation::default(),
        };
        let mut object = SceneObject::new(
            surface.name,
            pose,
            BBox::from_center_half_extents(surface.center, surface.half_extents),
        );
        object.semantic_label = Some(label.to_string());
        object.physics.is_static = true;
        object.physics.material = Some(surface.material.to_string());
        object
    });

    let support_top = support_anchor
        .as_ref()
        .map(|object| object.bbox.max.y)
        .unwrap_or(0.0);

    if let Some(anchor) = support_anchor {
        state.scene.add_object(anchor);
    }

    let mut name_counts = HashMap::<String, usize>::new();
    for (index, label) in object_labels.iter().enumerate() {
        let canonical = singularize_label(label);
        let half_extents = object_half_extents(&canonical);
        let support_offset = object_vertical_offset(&canonical);
        let spacing = 0.28;
        let origin = (object_labels.len() as f32 - 1.0) * 0.5;
        let x = (index as f32 - origin) * spacing;
        let z = if index % 2 == 0 { 0.0 } else { 0.16 };
        let pose = Pose {
            position: Position {
                x,
                y: support_top + half_extents.y + support_offset,
                z,
            },
            rotation: Rotation::default(),
        };

        let count = name_counts.entry(canonical.clone()).or_insert(0);
        *count += 1;
        let name = if *count == 1 {
            canonical.clone()
        } else {
            format!("{canonical}_{count}")
        };

        let mut object = SceneObject::new(
            name,
            pose,
            BBox::from_center_half_extents(pose.position, half_extents),
        );
        object.semantic_label = Some(canonical.clone());
        object.physics.mass = Some(object_mass(&canonical));
        object.physics.is_graspable = object_is_graspable(&canonical);
        object.physics.material = Some(object_material(&canonical).to_string());
        state.scene.add_object(object);
    }

    state.ensure_history_initialized(provider)?;
    Ok(state)
}

/// Derive a stable human-readable world name from a natural-language prompt.
pub fn derive_world_name_from_prompt(prompt: &str) -> String {
    let mut parts = normalized_tokens(prompt)
        .into_iter()
        .filter(|token| !NON_OBJECT_KEYWORDS.contains(&token.as_str()))
        .take(3)
        .map(|token| title_case(&token))
        .collect::<Vec<_>>();

    if parts.is_empty() {
        return "Seeded World".to_string();
    }

    if parts.last().is_none_or(|part| part != "Scene") {
        parts.push("Scene".to_string());
    }
    parts.join(" ")
}

#[derive(Clone, Copy)]
struct SupportSurface {
    name: &'static str,
    center: Position,
    half_extents: Vec3,
    material: &'static str,
}

fn normalized_tokens(prompt: &str) -> Vec<String> {
    let lowered = prompt.to_ascii_lowercase();
    let sanitized = lowered
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>();

    let raw_tokens = sanitized
        .split_whitespace()
        .filter(|token| !STOP_WORDS.contains(token))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let mut tokens = Vec::with_capacity(raw_tokens.len());
    let mut index = 0;
    while index < raw_tokens.len() {
        let current = raw_tokens[index].as_str();
        let next = raw_tokens.get(index + 1).map(String::as_str);
        match (current, next) {
            ("robot", Some("arm")) => {
                tokens.push("robot_arm".to_string());
                index += 2;
            }
            ("work", Some("bench")) => {
                tokens.push("workbench".to_string());
                index += 2;
            }
            _ => {
                tokens.push(raw_tokens[index].clone());
                index += 1;
            }
        }
    }

    tokens
}

fn support_label_from_tokens(tokens: &[String]) -> Option<&'static str> {
    tokens
        .iter()
        .find_map(|token| canonical_support_label(token))
        .or_else(|| {
            tokens.iter().find_map(|token| {
                ENVIRONMENT_KEYWORDS
                    .iter()
                    .find_map(|(keyword, support)| (token == keyword).then_some(*support))
            })
        })
        .or_else(|| (!tokens.is_empty()).then_some("floor"))
}

fn canonical_support_label(token: &str) -> Option<&'static str> {
    match token {
        "bench" => Some("bench"),
        "counter" | "countertop" => Some("counter"),
        "desk" => Some("desk"),
        "floor" | "ground" => Some("floor"),
        "pallet" => Some("pallet"),
        "shelf" => Some("shelf"),
        "table" => Some("table"),
        "workbench" => Some("workbench"),
        _ => None,
    }
}

fn object_labels_from_tokens(tokens: &[String], support_label: Option<&str>) -> Vec<String> {
    let mut seen = Vec::<String>::new();
    for token in tokens {
        if STOP_WORDS.contains(&token.as_str())
            || NON_OBJECT_KEYWORDS.contains(&token.as_str())
            || support_label == Some(token.as_str())
            || ENVIRONMENT_KEYWORDS
                .iter()
                .any(|(keyword, _)| keyword == &token.as_str())
        {
            continue;
        }

        let label = singularize_label(token);
        if !seen.contains(&label) {
            seen.push(label);
        }
    }

    seen.into_iter().take(4).collect()
}

fn metadata_tags(
    tokens: &[String],
    support_label: Option<&str>,
    object_labels: &[String],
) -> Vec<String> {
    let mut tags = Vec::new();
    if let Some(support_label) = support_label {
        tags.push(support_label.to_string());
    }

    for token in tokens {
        if ENVIRONMENT_KEYWORDS
            .iter()
            .any(|(keyword, _)| keyword == &token.as_str())
            && !tags.contains(token)
        {
            tags.push(token.clone());
        }
    }

    for label in object_labels {
        if !tags.contains(label) {
            tags.push(label.clone());
        }
    }

    tags
}

fn support_surface(label: &str) -> SupportSurface {
    match label {
        "bench" => SupportSurface {
            name: "bench",
            center: Position {
                x: 0.0,
                y: 0.72,
                z: 0.0,
            },
            half_extents: Vec3 {
                x: 0.8,
                y: 0.08,
                z: 0.4,
            },
            material: "metal",
        },
        "counter" => SupportSurface {
            name: "counter",
            center: Position {
                x: 0.0,
                y: 0.78,
                z: 0.0,
            },
            half_extents: Vec3 {
                x: 0.9,
                y: 0.08,
                z: 0.45,
            },
            material: "stone",
        },
        "desk" => SupportSurface {
            name: "desk",
            center: Position {
                x: 0.0,
                y: 0.74,
                z: 0.0,
            },
            half_extents: Vec3 {
                x: 0.75,
                y: 0.08,
                z: 0.4,
            },
            material: "wood",
        },
        "pallet" => SupportSurface {
            name: "pallet",
            center: Position {
                x: 0.0,
                y: 0.1,
                z: 0.0,
            },
            half_extents: Vec3 {
                x: 0.7,
                y: 0.1,
                z: 0.5,
            },
            material: "wood",
        },
        "shelf" => SupportSurface {
            name: "shelf",
            center: Position {
                x: 0.0,
                y: 1.2,
                z: 0.0,
            },
            half_extents: Vec3 {
                x: 0.6,
                y: 0.06,
                z: 0.25,
            },
            material: "wood",
        },
        "table" => SupportSurface {
            name: "table",
            center: Position {
                x: 0.0,
                y: 0.74,
                z: 0.0,
            },
            half_extents: Vec3 {
                x: 0.85,
                y: 0.08,
                z: 0.45,
            },
            material: "wood",
        },
        "workbench" => SupportSurface {
            name: "workbench",
            center: Position {
                x: 0.0,
                y: 0.76,
                z: 0.0,
            },
            half_extents: Vec3 {
                x: 0.95,
                y: 0.08,
                z: 0.45,
            },
            material: "metal",
        },
        _ => SupportSurface {
            name: "floor",
            center: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            half_extents: Vec3 {
                x: 1.4,
                y: 0.05,
                z: 1.4,
            },
            material: "concrete",
        },
    }
}

fn object_half_extents(label: &str) -> Vec3 {
    match label {
        "block" | "cube" => Vec3 {
            x: 0.06,
            y: 0.06,
            z: 0.06,
        },
        "bottle" => Vec3 {
            x: 0.04,
            y: 0.12,
            z: 0.04,
        },
        "box" | "crate" => Vec3 {
            x: 0.12,
            y: 0.12,
            z: 0.12,
        },
        "mug" | "cup" => Vec3 {
            x: 0.05,
            y: 0.08,
            z: 0.05,
        },
        "robot_arm" => Vec3 {
            x: 0.18,
            y: 0.32,
            z: 0.18,
        },
        _ => Vec3 {
            x: 0.09,
            y: 0.09,
            z: 0.09,
        },
    }
}

fn object_vertical_offset(label: &str) -> f32 {
    match label {
        "robot_arm" => 0.02,
        _ => 0.01,
    }
}

fn object_material(label: &str) -> &'static str {
    match label {
        "bottle" => "glass",
        "robot_arm" => "metal",
        _ => "plastic",
    }
}

fn object_mass(label: &str) -> f32 {
    match label {
        "robot_arm" => 12.0,
        "box" | "crate" => 1.5,
        _ => 0.35,
    }
}

fn object_is_graspable(label: &str) -> bool {
    !matches!(label, "robot_arm")
}

fn singularize_label(label: &str) -> String {
    if let Some(prefix) = label.strip_suffix("ies") {
        return format!("{prefix}y");
    }

    if label.len() > 3 && label.ends_with('s') && !label.ends_with("ss") {
        return label[..label.len() - 1].to_string();
    }

    label.to_string()
}

fn title_case(token: &str) -> String {
    token
        .split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => {
                    let mut rendered = first.to_ascii_uppercase().to_string();
                    rendered.push_str(chars.as_str());
                    rendered
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_world_state_from_prompt_sets_metadata_and_history() {
        let state = seed_world_state_from_prompt("A kitchen with a mug", "mock", None).unwrap();

        assert_eq!(state.metadata.description, "A kitchen with a mug");
        assert_eq!(state.metadata.created_by, "mock");
        assert_eq!(state.metadata.tags, vec!["counter", "kitchen", "mug"]);
        assert_eq!(state.history.len(), 1);
        assert!(state.scene.find_object_by_name("counter").is_some());
        assert!(state.scene.find_object_by_name("mug").is_some());
    }

    #[test]
    fn test_seed_world_state_from_prompt_respects_name_override() {
        let state = seed_world_state_from_prompt(
            "robot arm next to a table with blocks",
            "mock",
            Some("Assembly Cell"),
        )
        .unwrap();

        assert_eq!(state.metadata.name, "Assembly Cell");
        assert!(state.scene.find_object_by_name("table").is_some());
        assert!(state.scene.find_object_by_name("robot_arm").is_some());
        assert!(state.scene.find_object_by_name("block").is_some());
    }

    #[test]
    fn test_seed_world_state_from_prompt_is_deterministic() {
        let left = seed_world_state_from_prompt("A lab bench with blocks and a mug", "mock", None)
            .unwrap();
        let right = seed_world_state_from_prompt("A lab bench with blocks and a mug", "mock", None)
            .unwrap();

        assert_eq!(left.metadata.name, right.metadata.name);
        assert_eq!(left.metadata.tags, right.metadata.tags);

        let left_objects = left
            .scene
            .list_objects()
            .into_iter()
            .map(|object| (object.name.clone(), object.pose.position, object.bbox))
            .collect::<Vec<_>>();
        let right_objects = right
            .scene
            .list_objects()
            .into_iter()
            .map(|object| (object.name.clone(), object.pose.position, object.bbox))
            .collect::<Vec<_>>();
        assert_eq!(left_objects, right_objects);
    }

    #[test]
    fn test_seed_world_state_from_prompt_support_only_prompt_creates_non_empty_scene() {
        let state = seed_world_state_from_prompt("lab bench", "mock", None).unwrap();

        assert_eq!(state.scene.objects.len(), 1);
        assert_eq!(state.metadata.tags, vec!["bench", "lab"]);
        assert!(state.scene.find_object_by_name("bench").is_some());
    }

    #[test]
    fn test_derive_world_name_from_prompt_uses_content_tokens() {
        assert_eq!(
            derive_world_name_from_prompt("a robot arm beside a table"),
            "Robot Arm Table Scene"
        );
    }
}
