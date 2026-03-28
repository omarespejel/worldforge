//! Prompt-based world bootstrap helpers.
//!
//! These utilities turn a natural-language scene prompt into a deterministic
//! initial `WorldState` with coarse object geometry and metadata.

use std::collections::HashMap;

use crate::error::{Result, WorldForgeError};
use crate::scene::SceneObject;
use crate::state::WorldState;
use crate::types::{BBox, Pose, Position, Rotation, Vec3};

const SKIP_WORDS: &[&str] = &[
    "and", "around", "at", "by", "for", "from", "inside", "into", "or", "scene", "the", "with",
    "world",
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

const COUNT_WORDS: &[(&str, usize)] = &[
    ("a", 1),
    ("an", 1),
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
];

const COLOR_WORDS: &[&str] = &[
    "black", "blue", "brown", "gray", "green", "grey", "orange", "purple", "red", "silver",
    "white", "yellow",
];

const MATERIAL_WORDS: &[&str] = &[
    "ceramic", "glass", "metal", "metallic", "plastic", "stone", "wood", "wooden",
];

const MAX_BOOTSTRAP_OBJECTS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelationKind {
    NextTo,
    LeftOf,
    RightOf,
    On,
    Under,
    Behind,
    InFrontOf,
}

#[derive(Clone, Debug)]
enum RelationTarget {
    Support(&'static str),
    Object {
        base_label: String,
        descriptors: Vec<String>,
    },
}

#[derive(Clone, Debug)]
struct PendingRelation {
    kind: RelationKind,
    target: RelationTarget,
}

#[derive(Clone, Debug)]
struct ParsedObjectSpec {
    base_label: String,
    descriptors: Vec<String>,
    material_override: Option<String>,
    count: usize,
    relation: Option<PendingRelation>,
}

#[derive(Clone, Debug)]
struct SeedObjectSpec {
    name: String,
    base_label: String,
    descriptors: Vec<String>,
    material: String,
    relation: Option<PendingRelation>,
}

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
    let parsed_objects = parse_object_specs(&tokens, support_label);

    let world_name = name_override
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| derive_world_name_from_prompt(prompt));

    let mut state = WorldState::new(world_name, provider);
    state.metadata.description = prompt.to_string();
    state.metadata.tags = metadata_tags(&tokens, support_label, &parsed_objects);

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

    let seeds = expand_object_specs(&parsed_objects);
    let objects = place_objects_from_specs(&seeds, support_top);
    for object in objects {
        state.scene.add_object(object);
    }

    state.ensure_history_initialized(provider)?;
    Ok(state)
}

/// Derive a stable human-readable world name from a natural-language prompt.
pub fn derive_world_name_from_prompt(prompt: &str) -> String {
    let mut parts = normalized_tokens(prompt)
        .into_iter()
        .filter(|token| {
            !NON_OBJECT_KEYWORDS.contains(&token.as_str())
                && !is_relation_token(token)
                && count_for_token(token).is_none()
                && !is_descriptor_token(token)
        })
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
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let mut tokens = Vec::with_capacity(raw_tokens.len());
    let mut index = 0;
    while index < raw_tokens.len() {
        let current = raw_tokens[index].as_str();
        let next = raw_tokens.get(index + 1).map(String::as_str);
        let next_next = raw_tokens.get(index + 2).map(String::as_str);
        match (current, next) {
            ("robot", Some("arm")) => {
                tokens.push("robot_arm".to_string());
                index += 2;
            }
            ("work", Some("bench")) => {
                tokens.push("workbench".to_string());
                index += 2;
            }
            ("next", Some("to")) => {
                tokens.push("next_to".to_string());
                index += 2;
            }
            ("left", Some("of")) => {
                tokens.push("left_of".to_string());
                index += 2;
            }
            ("right", Some("of")) => {
                tokens.push("right_of".to_string());
                index += 2;
            }
            ("on", Some("top")) if next_next == Some("of") => {
                tokens.push("on_top_of".to_string());
                index += 3;
            }
            ("in", Some("front")) if next_next == Some("of") => {
                tokens.push("in_front_of".to_string());
                index += 3;
            }
            _ => {
                let normalized = match current {
                    "beside" | "near" => Some("next_to".to_string()),
                    "below" => Some("under".to_string()),
                    _ if SKIP_WORDS.contains(&current) => None,
                    _ => Some(raw_tokens[index].clone()),
                };
                if let Some(token) = normalized {
                    tokens.push(token);
                }
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

fn parse_object_specs(tokens: &[String], support_label: Option<&str>) -> Vec<ParsedObjectSpec> {
    let mut specs = Vec::new();
    let mut index = 0;

    while index < tokens.len() && specs.len() < MAX_BOOTSTRAP_OBJECTS {
        let mut count = 1;
        if let Some(parsed_count) = count_for_token(&tokens[index]) {
            count = parsed_count;
            index += 1;
        }

        let mut descriptors = Vec::new();
        let mut material_override = None;
        while index < tokens.len() {
            if let Some(descriptor) = normalized_color(&tokens[index]) {
                descriptors.push(descriptor.to_string());
                index += 1;
                continue;
            }
            if let Some(material) = normalized_material(&tokens[index]) {
                descriptors.push(material.to_string());
                material_override = Some(material.to_string());
                index += 1;
                continue;
            }
            break;
        }

        let Some(token) = tokens.get(index) else {
            break;
        };
        if !is_object_candidate(token, support_label) {
            index += 1;
            continue;
        }

        let base_label = singularize_label(token);
        index += 1;

        let relation = if index < tokens.len() {
            parse_relation(tokens, &mut index, support_label)
        } else {
            None
        };

        specs.push(ParsedObjectSpec {
            base_label,
            descriptors,
            material_override,
            count: count.clamp(1, MAX_BOOTSTRAP_OBJECTS),
            relation,
        });
    }

    specs
}

fn parse_relation(
    tokens: &[String],
    index: &mut usize,
    support_label: Option<&str>,
) -> Option<PendingRelation> {
    let relation_kind = relation_kind_for_token(tokens.get(*index)?)?;
    let target_start = *index + 1;
    let mut cursor = target_start;

    while cursor < tokens.len()
        && (count_for_token(&tokens[cursor]).is_some()
            || ENVIRONMENT_KEYWORDS
                .iter()
                .any(|(keyword, _)| keyword == &tokens[cursor].as_str()))
    {
        if let Some(surface) = support_from_environment(&tokens[cursor]) {
            *index = target_start;
            return Some(PendingRelation {
                kind: relation_kind,
                target: RelationTarget::Support(surface),
            });
        }
        cursor += 1;
    }

    let mut descriptors = Vec::new();
    while cursor < tokens.len() {
        if let Some(color) = normalized_color(&tokens[cursor]) {
            descriptors.push(color.to_string());
            cursor += 1;
            continue;
        }
        if let Some(material) = normalized_material(&tokens[cursor]) {
            descriptors.push(material.to_string());
            cursor += 1;
            continue;
        }
        break;
    }

    let target_token = tokens.get(cursor)?;
    if let Some(surface) = canonical_support_label(target_token) {
        *index = target_start;
        return Some(PendingRelation {
            kind: relation_kind,
            target: RelationTarget::Support(surface),
        });
    }
    if let Some(surface) = support_from_environment(target_token) {
        *index = target_start;
        return Some(PendingRelation {
            kind: relation_kind,
            target: RelationTarget::Support(surface),
        });
    }
    if !is_object_candidate(target_token, support_label) {
        *index += 1;
        return None;
    }

    let base_label = singularize_label(target_token);
    *index = target_start;
    Some(PendingRelation {
        kind: relation_kind,
        target: RelationTarget::Object {
            base_label,
            descriptors,
        },
    })
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

fn is_object_candidate(token: &str, support_label: Option<&str>) -> bool {
    count_for_token(token).is_none()
        && !is_descriptor_token(token)
        && !is_relation_token(token)
        && !NON_OBJECT_KEYWORDS.contains(&token)
        && support_label != Some(token)
        && canonical_support_label(token).is_none()
        && !ENVIRONMENT_KEYWORDS
            .iter()
            .any(|(keyword, _)| keyword == &token)
}

fn metadata_tags(
    tokens: &[String],
    support_label: Option<&str>,
    object_specs: &[ParsedObjectSpec],
) -> Vec<String> {
    let mut tags = Vec::new();
    if let Some(support_label) = support_label {
        push_unique(&mut tags, support_label.to_string());
    }

    for token in tokens {
        if ENVIRONMENT_KEYWORDS
            .iter()
            .any(|(keyword, _)| keyword == &token.as_str())
        {
            push_unique(&mut tags, token.clone());
        }
    }

    for spec in object_specs {
        push_unique(&mut tags, spec.base_label.clone());
        for descriptor in &spec.descriptors {
            push_unique(&mut tags, descriptor.clone());
        }
    }

    tags
}

fn expand_object_specs(specs: &[ParsedObjectSpec]) -> Vec<SeedObjectSpec> {
    let mut expanded = Vec::new();
    let mut name_counts = HashMap::<String, usize>::new();

    for spec in specs {
        for _ in 0..spec.count {
            if expanded.len() >= MAX_BOOTSTRAP_OBJECTS {
                return expanded;
            }

            let base_name = object_display_name(&spec.base_label, &spec.descriptors);
            let count = name_counts.entry(base_name.clone()).or_insert(0);
            *count += 1;
            let name = if *count == 1 {
                base_name
            } else {
                format!("{base_name}_{count}")
            };

            expanded.push(SeedObjectSpec {
                name,
                base_label: spec.base_label.clone(),
                descriptors: spec.descriptors.clone(),
                material: spec
                    .material_override
                    .clone()
                    .unwrap_or_else(|| object_material(&spec.base_label).to_string()),
                relation: spec.relation.clone(),
            });
        }
    }

    expanded
}

fn place_objects_from_specs(specs: &[SeedObjectSpec], support_top: f32) -> Vec<SceneObject> {
    let mut placed = vec![None; specs.len()];
    let mut remaining: Vec<_> = (0..specs.len()).collect();
    let mut default_slot = 0usize;
    let mut relation_slots = HashMap::<String, usize>::new();

    while !remaining.is_empty() {
        let mut next_remaining = Vec::new();
        let mut progress = false;

        for index in remaining {
            let spec = &specs[index];
            let half_extents = object_half_extents(&spec.base_label);

            let position = match spec.relation.as_ref() {
                Some(relation) => resolve_relation_position(
                    relation,
                    specs,
                    &placed,
                    &mut relation_slots,
                    &spec.base_label,
                    half_extents,
                ),
                None => None,
            };

            let position = if let Some(position) = position {
                position
            } else if relation_target_ready(spec.relation.as_ref(), specs, &placed) == Some(false) {
                next_remaining.push(index);
                continue;
            } else {
                let position = default_object_position(default_slot, support_top, &spec.base_label);
                default_slot += 1;
                position
            };

            let pose = Pose {
                position,
                rotation: Rotation::default(),
            };
            let mut object = SceneObject::new(
                spec.name.clone(),
                pose,
                BBox::from_center_half_extents(position, half_extents),
            );
            object.semantic_label = Some(spec.base_label.clone());
            object.physics.mass = Some(object_mass(&spec.base_label));
            object.physics.is_graspable = object_is_graspable(&spec.base_label);
            object.physics.material = Some(spec.material.clone());
            placed[index] = Some(object);
            progress = true;
        }

        if !progress {
            for index in next_remaining {
                let spec = &specs[index];
                let half_extents = object_half_extents(&spec.base_label);
                let position = default_object_position(default_slot, support_top, &spec.base_label);
                default_slot += 1;
                let pose = Pose {
                    position,
                    rotation: Rotation::default(),
                };
                let mut object = SceneObject::new(
                    spec.name.clone(),
                    pose,
                    BBox::from_center_half_extents(position, half_extents),
                );
                object.semantic_label = Some(spec.base_label.clone());
                object.physics.mass = Some(object_mass(&spec.base_label));
                object.physics.is_graspable = object_is_graspable(&spec.base_label);
                object.physics.material = Some(spec.material.clone());
                placed[index] = Some(object);
            }
            break;
        }

        remaining = next_remaining;
    }

    placed.into_iter().flatten().collect()
}

fn relation_target_ready(
    relation: Option<&PendingRelation>,
    specs: &[SeedObjectSpec],
    placed: &[Option<SceneObject>],
) -> Option<bool> {
    let relation = relation?;
    match &relation.target {
        RelationTarget::Support(_) => Some(true),
        RelationTarget::Object {
            base_label,
            descriptors,
        } => Some(placed.iter().enumerate().any(|(target_index, object)| {
            object.is_some()
                && relation_object_signature(
                    &specs[target_index].base_label,
                    &specs[target_index].descriptors,
                ) == relation_object_signature(base_label, descriptors)
        })),
    }
}

fn resolve_relation_position(
    relation: &PendingRelation,
    specs: &[SeedObjectSpec],
    placed: &[Option<SceneObject>],
    relation_slots: &mut HashMap<String, usize>,
    base_label: &str,
    half_extents: Vec3,
) -> Option<Position> {
    let support_offset = object_vertical_offset(base_label);

    match &relation.target {
        RelationTarget::Support(label) => {
            let surface = support_surface(label);
            let slot = next_relation_slot(
                relation_slots,
                &format!("support:{label}:{}", relation_key(relation.kind)),
            );
            Some(position_from_anchor(
                relation.kind,
                surface.center,
                surface.half_extents,
                half_extents,
                support_offset,
                slot,
            ))
        }
        RelationTarget::Object {
            base_label: target_label,
            descriptors,
        } => {
            let target_signature = relation_object_signature(target_label, descriptors);
            let target_object = specs.iter().zip(placed.iter()).find_map(|(spec, object)| {
                (object.is_some()
                    && relation_object_signature(&spec.base_label, &spec.descriptors)
                        == target_signature)
                    .then_some(object.as_ref()?)
            })?;
            let slot = next_relation_slot(
                relation_slots,
                &format!(
                    "object:{}:{}",
                    target_object.name,
                    relation_key(relation.kind)
                ),
            );
            Some(position_from_anchor(
                relation.kind,
                target_object.pose.position,
                target_object.half_extents(),
                half_extents,
                support_offset,
                slot,
            ))
        }
    }
}

fn default_object_position(slot: usize, support_top: f32, label: &str) -> Position {
    let half_extents = object_half_extents(label);
    let spacing = 0.26;
    let row = slot / 3;
    let col = slot % 3;
    let x = (col as f32 - 1.0) * spacing;
    let z = (row as f32) * 0.18 - 0.09;
    Position {
        x,
        y: support_top + half_extents.y + object_vertical_offset(label),
        z,
    }
}

fn position_from_anchor(
    relation: RelationKind,
    anchor_center: Position,
    anchor_half_extents: Vec3,
    half_extents: Vec3,
    support_offset: f32,
    slot: usize,
) -> Position {
    let gap_x = anchor_half_extents.x + half_extents.x + 0.08;
    let gap_z = anchor_half_extents.z + half_extents.z + 0.08;
    let layer = 1.0 + (slot / 4) as f32 * 0.6;
    let offset = relation_slot_offset(slot, gap_x * layer, gap_z * layer);

    match relation {
        RelationKind::NextTo => Position {
            x: anchor_center.x + offset.x,
            y: anchor_center.y,
            z: anchor_center.z + offset.z,
        },
        RelationKind::LeftOf => Position {
            x: anchor_center.x - gap_x,
            y: anchor_center.y,
            z: anchor_center.z + offset.z * 0.35,
        },
        RelationKind::RightOf => Position {
            x: anchor_center.x + gap_x,
            y: anchor_center.y,
            z: anchor_center.z + offset.z * 0.35,
        },
        RelationKind::Behind => Position {
            x: anchor_center.x + offset.x * 0.35,
            y: anchor_center.y,
            z: anchor_center.z + gap_z,
        },
        RelationKind::InFrontOf => Position {
            x: anchor_center.x + offset.x * 0.35,
            y: anchor_center.y,
            z: anchor_center.z - gap_z,
        },
        RelationKind::On => Position {
            x: anchor_center.x + offset.x * 0.25,
            y: anchor_center.y + anchor_half_extents.y + half_extents.y + support_offset,
            z: anchor_center.z + offset.z * 0.25,
        },
        RelationKind::Under => Position {
            x: anchor_center.x + offset.x * 0.2,
            y: anchor_center.y - anchor_half_extents.y - half_extents.y - support_offset,
            z: anchor_center.z + offset.z * 0.2,
        },
    }
}

fn relation_slot_offset(slot: usize, gap_x: f32, gap_z: f32) -> Vec3 {
    match slot % 4 {
        0 => Vec3 {
            x: gap_x,
            y: 0.0,
            z: 0.0,
        },
        1 => Vec3 {
            x: -gap_x,
            y: 0.0,
            z: 0.0,
        },
        2 => Vec3 {
            x: 0.0,
            y: 0.0,
            z: gap_z,
        },
        _ => Vec3 {
            x: 0.0,
            y: 0.0,
            z: -gap_z,
        },
    }
}

fn next_relation_slot(slots: &mut HashMap<String, usize>, key: &str) -> usize {
    let entry = slots.entry(key.to_string()).or_insert(0);
    let slot = *entry;
    *entry += 1;
    slot
}

fn relation_key(relation: RelationKind) -> &'static str {
    match relation {
        RelationKind::NextTo => "next_to",
        RelationKind::LeftOf => "left_of",
        RelationKind::RightOf => "right_of",
        RelationKind::On => "on",
        RelationKind::Under => "under",
        RelationKind::Behind => "behind",
        RelationKind::InFrontOf => "in_front_of",
    }
}

fn relation_object_signature(base_label: &str, descriptors: &[String]) -> String {
    object_display_name(base_label, descriptors)
}

fn object_display_name(base_label: &str, descriptors: &[String]) -> String {
    if descriptors.is_empty() {
        return base_label.to_string();
    }

    let mut name_parts = descriptors.to_vec();
    name_parts.push(base_label.to_string());
    name_parts.join("_")
}

fn count_for_token(token: &str) -> Option<usize> {
    COUNT_WORDS
        .iter()
        .find_map(|(word, count)| (*word == token).then_some(*count))
        .or_else(|| token.parse::<usize>().ok())
}

fn normalized_color(token: &str) -> Option<&str> {
    match token {
        "grey" => Some("gray"),
        value if COLOR_WORDS.contains(&value) => Some(value),
        _ => None,
    }
}

fn normalized_material(token: &str) -> Option<&str> {
    match token {
        "wooden" => Some("wood"),
        "metallic" => Some("metal"),
        value if MATERIAL_WORDS.contains(&value) => Some(value),
        _ => None,
    }
}

fn is_descriptor_token(token: &str) -> bool {
    normalized_color(token).is_some() || normalized_material(token).is_some()
}

fn is_relation_token(token: &str) -> bool {
    relation_kind_for_token(token).is_some()
}

fn relation_kind_for_token(token: &str) -> Option<RelationKind> {
    match token {
        "next_to" => Some(RelationKind::NextTo),
        "left_of" => Some(RelationKind::LeftOf),
        "right_of" => Some(RelationKind::RightOf),
        "on" | "on_top_of" => Some(RelationKind::On),
        "under" => Some(RelationKind::Under),
        "behind" => Some(RelationKind::Behind),
        "in_front_of" => Some(RelationKind::InFrontOf),
        _ => None,
    }
}

fn support_from_environment(token: &str) -> Option<&'static str> {
    ENVIRONMENT_KEYWORDS
        .iter()
        .find_map(|(keyword, support)| (*keyword == token).then_some(*support))
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
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
    fn test_seed_world_state_from_prompt_parses_counts_and_relations() {
        let state = seed_world_state_from_prompt(
            "Two red blocks next to a blue mug on a table",
            "mock",
            None,
        )
        .unwrap();

        let table = state.scene.find_object_by_name("table").unwrap();
        let mug = state.scene.find_object_by_name("blue_mug").unwrap();
        let block = state.scene.find_object_by_name("red_block").unwrap();
        let block_2 = state.scene.find_object_by_name("red_block_2").unwrap();

        assert_eq!(mug.semantic_label.as_deref(), Some("mug"));
        assert_eq!(block.semantic_label.as_deref(), Some("block"));
        assert_eq!(block_2.semantic_label.as_deref(), Some("block"));
        assert!(state.metadata.tags.contains(&"red".to_string()));
        assert!(state.metadata.tags.contains(&"blue".to_string()));
        assert!(mug.pose.position.y > table.bbox.max.y);
        assert!((block.pose.position.y - mug.pose.position.y).abs() < 0.05);
        assert!((block_2.pose.position.y - mug.pose.position.y).abs() < 0.05);
        assert!(block.pose.position.distance(mug.pose.position) < 0.35);
        assert!(block_2.pose.position.distance(mug.pose.position) < 0.35);
    }

    #[test]
    fn test_seed_world_state_from_prompt_supports_vertical_relations() {
        let state = seed_world_state_from_prompt(
            "A box on a table and a mug under the table",
            "mock",
            None,
        )
        .unwrap();

        let table = state.scene.find_object_by_name("table").unwrap();
        let box_object = state.scene.find_object_by_name("box").unwrap();
        let mug = state.scene.find_object_by_name("mug").unwrap();

        assert!(box_object.pose.position.y > table.bbox.max.y);
        assert!(mug.pose.position.y < table.bbox.min.y);
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
