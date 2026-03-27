//! Shared deterministic native-planning helpers for provider adapters.
//!
//! This module turns a `PlanRequest` into a concrete action sequence by
//! deriving goal-directed actions, simulating them against a mutable world
//! state, and assembling the final `Plan`.

use std::time::Instant;

use worldforge_core::action::{evaluate_condition, Action, Condition, Weather};
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::goal_image;
use worldforge_core::guardrail::{evaluate_guardrails, has_blocking_violation};
use worldforge_core::prediction::{Plan, PlanGoal, PlanRequest};
use worldforge_core::provider::CostEstimate;
use worldforge_core::scene::SceneObject;
use worldforge_core::state::WorldState;
use worldforge_core::types::{BBox, Pose, Position, Rotation, Vec3, Velocity};

const PLANNING_FPS: f32 = 24.0;
const DEFAULT_SPAWN_HALF_EXTENTS: Vec3 = Vec3 {
    x: 0.12,
    y: 0.12,
    z: 0.12,
};

/// Build a deterministic native plan for a provider.
///
/// The helper is intentionally provider-agnostic: it derives actions from the
/// goal, simulates each action locally, and performs the same guardrail and
/// success bookkeeping expected by the core orchestration layer.
pub(crate) fn plan_native(
    provider_name: &str,
    request: &PlanRequest,
    step_cost: CostEstimate,
) -> Result<Plan> {
    let started = Instant::now();
    let mut state = request.current_state.clone();
    let mut actions = derive_native_actions(&request.goal, &state)?;
    actions.truncate(request.max_steps as usize);

    let mut planned_actions = Vec::with_capacity(actions.len());
    let mut predicted_states = Vec::with_capacity(actions.len());
    let mut guardrail_compliance = Vec::with_capacity(actions.len());

    for action in actions {
        let next_state = simulate_action(&state, &action);
        let guardrail_results = if request.guardrails.is_empty() {
            Vec::new()
        } else {
            let results = evaluate_guardrails(&request.guardrails, &next_state);
            if has_blocking_violation(&results) {
                return Err(WorldForgeError::NoFeasiblePlan {
                    goal: format!("{:?}", request.goal),
                    reason: format!(
                        "{provider_name} native planner generated a guardrail-blocked step"
                    ),
                });
            }
            results
        };

        planned_actions.push(action);
        state = next_state;
        predicted_states.push(state.clone());
        guardrail_compliance.push(guardrail_results);

        if goal_satisfied(&request.goal, &state) {
            break;
        }
    }

    if !goal_satisfied(&request.goal, &state) {
        return Err(WorldForgeError::NoFeasiblePlan {
            goal: format!("{:?}", request.goal),
            reason: format!(
                "{provider_name} native planner exhausted the step budget before satisfying the goal"
            ),
        });
    }

    let iterations_used = u32::try_from(planned_actions.len()).unwrap_or(u32::MAX);

    Ok(Plan {
        actions: planned_actions,
        predicted_states,
        predicted_videos: None,
        total_cost: (step_cost.usd as f32) * iterations_used as f32,
        success_probability: goal_score(&request.goal, &state),
        guardrail_compliance,
        planning_time_ms: started.elapsed().as_millis() as u64,
        iterations_used,
        verification_proof: None,
    })
}

fn simulate_action(state: &WorldState, action: &Action) -> WorldState {
    let mut next_state = state.clone();
    apply_action(&mut next_state, action);
    next_state
}

fn apply_action(state: &mut WorldState, action: &Action) {
    match action {
        Action::Move { target, speed } => {
            if let Some(object_id) = primary_movable_object_id(state, Some(*target)) {
                if let Some(object) = state.scene.get_object_mut(&object_id) {
                    let blend = ((*speed * 0.15) + 0.2).clamp(0.15, 0.9);
                    let next_position = object.pose.position.lerp(*target, blend);
                    let delta = Vec3 {
                        x: next_position.x - object.pose.position.x,
                        y: next_position.y - object.pose.position.y,
                        z: next_position.z - object.pose.position.z,
                    };
                    object.set_position(next_position);
                    object.velocity = Velocity {
                        x: delta.x * (*speed).max(0.25),
                        y: delta.y * (*speed).max(0.25),
                        z: delta.z * (*speed).max(0.25),
                    };
                }
            }
        }
        Action::Grasp { object, grip_force } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                item.velocity = Velocity::default();
                let lift = (*grip_force).clamp(0.0, 20.0) * 0.001;
                item.pose.position.y += lift;
                item.bbox.translate(Vec3 {
                    x: 0.0,
                    y: lift,
                    z: 0.0,
                });
            }
        }
        Action::Release { object } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                item.velocity = Velocity::default();
            }
        }
        Action::Push {
            object,
            direction,
            force,
        } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                let push = direction
                    .normalized()
                    .scale((*force).clamp(0.0, 50.0) * 0.04);
                item.translate_by(push);
                item.velocity = Velocity {
                    x: push.x * 2.0,
                    y: push.y * 2.0,
                    z: push.z * 2.0,
                };
            }
        }
        Action::Rotate {
            object,
            axis,
            angle,
        } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                let delta = quaternion_from_axis_angle(*axis, *angle);
                item.pose.rotation = multiply_rotation(item.pose.rotation, delta);
            }
        }
        Action::Place { object, target } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                item.set_position(*target);
                item.velocity = Velocity::default();
            }
        }
        Action::CameraMove { .. } | Action::CameraLookAt { .. } => {}
        Action::Navigate { waypoints } => {
            if let Some(target) = waypoints.last().copied() {
                if let Some(object_id) = primary_movable_object_id(state, Some(target)) {
                    if let Some(item) = state.scene.get_object_mut(&object_id) {
                        item.set_position(target);
                        item.velocity = Velocity::default();
                    }
                }
            }
        }
        Action::Teleport { destination } => {
            if let Some(object_id) = primary_movable_object_id(state, Some(destination.position)) {
                if let Some(item) = state.scene.get_object_mut(&object_id) {
                    item.set_position(destination.position);
                    item.pose.rotation = destination.rotation;
                    item.velocity = Velocity::default();
                }
            }
        }
        Action::SetWeather { weather } => {
            replace_tag(
                &mut state.metadata.tags,
                "weather:",
                format!("weather:{weather:?}").to_lowercase(),
            );
        }
        Action::SetLighting { time_of_day } => {
            replace_tag(
                &mut state.metadata.tags,
                "lighting:",
                format!("lighting:{time_of_day:.2}"),
            );
        }
        Action::SpawnObject { template, pose } => {
            let mut object = SceneObject::new(
                template.clone(),
                *pose,
                BBox::from_center_half_extents(pose.position, DEFAULT_SPAWN_HALF_EXTENTS),
            );
            object.semantic_label = Some(template.clone());
            state.scene.add_object(object);
        }
        Action::RemoveObject { object } => {
            let _ = state.scene.remove_object(object);
        }
        Action::Sequence(actions) | Action::Parallel(actions) => {
            for nested in actions {
                apply_action(state, nested);
            }
        }
        Action::Conditional {
            condition,
            then,
            otherwise,
        } => {
            if evaluate_condition(condition, state) {
                apply_action(state, then);
            } else if let Some(otherwise) = otherwise {
                apply_action(state, otherwise);
            }
        }
        Action::Raw { .. } => {}
    }

    state.scene.refresh_relationships();
    bump_time(state);
}

fn bump_time(state: &mut WorldState) {
    state.time.step = state.time.step.saturating_add(1);
    state.time.seconds += 1.0 / PLANNING_FPS as f64;
    state.time.dt = 1.0 / PLANNING_FPS as f64;
}

fn derive_native_actions(goal: &PlanGoal, state: &WorldState) -> Result<Vec<Action>> {
    match goal {
        PlanGoal::Condition(condition) => actions_for_condition(condition, state),
        PlanGoal::TargetState(target) => Ok(actions_for_target_state(state, target)),
        PlanGoal::GoalImage(image) => actions_for_goal_image(image, state),
        PlanGoal::Description(description) => actions_for_description(description, state),
    }
}

fn actions_for_goal_image(
    goal_image_tensor: &worldforge_core::types::Tensor,
    state: &WorldState,
) -> Result<Vec<Action>> {
    let target = goal_image::goal_image_target(goal_image_tensor, state).ok_or_else(|| {
        WorldForgeError::NoFeasiblePlan {
            goal: "goal-image".to_string(),
            reason: "native planner could not interpret the goal image".to_string(),
        }
    })?;
    let tolerance = goal_image_tolerance(target.confidence);

    if let Some(object_id) = primary_movable_object_id(state, None) {
        if state
            .scene
            .get_object(&object_id)
            .is_some_and(|object| object.pose.position.distance(target.position) <= tolerance)
        {
            return Ok(Vec::new());
        }
        Ok(vec![Action::Place {
            object: object_id,
            target: target.position,
        }])
    } else {
        Ok(vec![Action::SpawnObject {
            template: "goal-image-object".to_string(),
            pose: Pose {
                position: target.position,
                ..Pose::default()
            },
        }])
    }
}

fn goal_image_tolerance(confidence: f32) -> f32 {
    (0.12 + (1.0 - confidence).clamp(0.0, 1.0) * 0.2).clamp(0.05, 0.5)
}

fn actions_for_condition(condition: &Condition, state: &WorldState) -> Result<Vec<Action>> {
    match condition {
        Condition::ObjectAt {
            object,
            position,
            tolerance,
        } => {
            let Some(item) = state.scene.get_object(object) else {
                return Err(WorldForgeError::NoFeasiblePlan {
                    goal: format!("{condition:?}"),
                    reason: "condition references an unknown object".to_string(),
                });
            };
            if item.pose.position.distance(*position) <= *tolerance {
                Ok(Vec::new())
            } else {
                Ok(vec![Action::Place {
                    object: *object,
                    target: *position,
                }])
            }
        }
        Condition::ObjectsTouching { a, b } => {
            if evaluate_condition(condition, state) {
                return Ok(Vec::new());
            }
            let Some(anchor) = state.scene.get_object(b) else {
                return Err(WorldForgeError::NoFeasiblePlan {
                    goal: format!("{condition:?}"),
                    reason: "touching condition references an unknown anchor object".to_string(),
                });
            };
            Ok(vec![Action::Place {
                object: *a,
                target: anchor.pose.position,
            }])
        }
        Condition::ObjectExists { object } => {
            if state.scene.get_object(object).is_some() {
                Ok(Vec::new())
            } else {
                Err(WorldForgeError::NoFeasiblePlan {
                    goal: format!("{condition:?}"),
                    reason: "ObjectExists cannot be satisfied because IDs are immutable"
                        .to_string(),
                })
            }
        }
        Condition::And(conditions) => {
            let mut simulated = state.clone();
            let mut actions = Vec::new();
            for condition in conditions {
                let step_actions = actions_for_condition(condition, &simulated)?;
                for action in step_actions {
                    simulated = simulate_action(&simulated, &action);
                    actions.push(action);
                }
            }
            Ok(actions)
        }
        Condition::Or(conditions) => {
            if conditions
                .iter()
                .any(|candidate| evaluate_condition(candidate, state))
            {
                return Ok(Vec::new());
            }

            for candidate in conditions {
                if let Ok(actions) = actions_for_condition(candidate, state) {
                    return Ok(actions);
                }
            }

            Err(WorldForgeError::NoFeasiblePlan {
                goal: format!("{condition:?}"),
                reason: "none of the OR branches can be satisfied".to_string(),
            })
        }
        Condition::Not(inner) => actions_to_negate_condition(inner, state),
    }
}

fn actions_to_negate_condition(condition: &Condition, state: &WorldState) -> Result<Vec<Action>> {
    if !evaluate_condition(condition, state) {
        return Ok(Vec::new());
    }

    match condition {
        Condition::ObjectExists { object } => Ok(vec![Action::RemoveObject { object: *object }]),
        Condition::ObjectAt {
            object,
            position,
            tolerance,
        } => Ok(vec![Action::Place {
            object: *object,
            target: Position {
                x: position.x + tolerance.max(0.1) + 0.2,
                y: position.y,
                z: position.z,
            },
        }]),
        Condition::ObjectsTouching { a, b } => {
            let Some(anchor) = state.scene.get_object(b) else {
                return Err(WorldForgeError::NoFeasiblePlan {
                    goal: format!("{condition:?}"),
                    reason: "cannot negate touching condition without anchor object".to_string(),
                });
            };
            Ok(vec![Action::Place {
                object: *a,
                target: Position {
                    x: anchor.pose.position.x + 0.6,
                    y: anchor.pose.position.y,
                    z: anchor.pose.position.z,
                },
            }])
        }
        _ => Err(WorldForgeError::NoFeasiblePlan {
            goal: format!("{condition:?}"),
            reason: "native planner cannot negate this compound condition".to_string(),
        }),
    }
}

fn actions_for_target_state(current: &WorldState, target: &WorldState) -> Vec<Action> {
    let mut actions = Vec::new();

    let mut current_objects: Vec<_> = current.scene.objects.values().collect();
    current_objects.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });

    for object in current_objects {
        if target.scene.get_object(&object.id).is_none() {
            actions.push(Action::RemoveObject { object: object.id });
        }
    }

    let mut target_objects: Vec<_> = target.scene.objects.values().collect();
    target_objects.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });

    for target_object in target_objects {
        if let Some(current_object) = current.scene.get_object(&target_object.id) {
            if current_object
                .pose
                .position
                .distance(target_object.pose.position)
                > 0.05
            {
                actions.push(Action::Place {
                    object: target_object.id,
                    target: target_object.pose.position,
                });
            }
        } else {
            actions.push(Action::SpawnObject {
                template: target_object.name.clone(),
                pose: target_object.pose,
            });
        }
    }

    if let Some(weather) = weather_from_tags(&target.metadata.tags) {
        if weather_from_tags(&current.metadata.tags) != Some(weather) {
            actions.push(Action::SetWeather { weather });
        }
    }

    if let Some(lighting) = lighting_from_tags(&target.metadata.tags) {
        let current_lighting = lighting_from_tags(&current.metadata.tags);
        if current_lighting
            .map(|value| (value - lighting).abs() > 0.25)
            .unwrap_or(true)
        {
            actions.push(Action::SetLighting {
                time_of_day: lighting,
            });
        }
    }

    actions
}

fn actions_for_description(description: &str, state: &WorldState) -> Result<Vec<Action>> {
    let normalized = description.to_lowercase();
    let mut actions = Vec::new();

    if let Some(weather) = parse_weather_hint(&normalized) {
        actions.push(Action::SetWeather { weather });
    }

    if let Some(time_of_day) = parse_lighting_hint(&normalized) {
        actions.push(Action::SetLighting { time_of_day });
    }

    if let Some(name) =
        infer_object_name_from_verb(description, &["remove", "delete", "discard", "drop"])
    {
        if let Some(object) = find_object_by_name_or_label(state, &name.to_lowercase()) {
            actions.push(Action::RemoveObject { object: object.id });
        }
    }

    if let Some(template) = infer_object_name_from_verb(description, &["spawn", "create", "add"]) {
        let pose = Pose {
            position: parse_relative_target_hint(state, &normalized)
                .map(|hint| hint.target)
                .or_else(|| parse_position_hint(description))
                .unwrap_or_else(|| default_spawn_position(state)),
            ..Pose::default()
        };
        actions.push(Action::SpawnObject { template, pose });
    }

    if let Some(target) = parse_position_hint(description)
        .or_else(|| parse_relative_target_hint(state, &normalized).map(|hint| hint.target))
    {
        if let Some(name) = infer_object_name_from_verb(description, &["move", "place", "put"]) {
            if let Some(object) = find_object_by_name_or_label(state, &name.to_lowercase()) {
                actions.push(Action::Place {
                    object: object.id,
                    target,
                });
            } else {
                actions.push(Action::SpawnObject {
                    template: name,
                    pose: Pose {
                        position: target,
                        ..Pose::default()
                    },
                });
            }
        } else if let Some(object_id) = primary_movable_object_id(state, Some(target)) {
            actions.push(Action::Place {
                object: object_id,
                target,
            });
        }
    } else if let Some(name) = infer_object_name_from_verb(description, &["move", "place", "put"]) {
        if let Some(object) = find_object_by_name_or_label(state, &name.to_lowercase()) {
            actions.push(Action::Place {
                object: object.id,
                target: default_spawn_position(state),
            });
        } else {
            actions.push(Action::SpawnObject {
                template: name,
                pose: Pose {
                    position: default_spawn_position(state),
                    ..Pose::default()
                },
            });
        }
    }

    if actions.is_empty() {
        if let Some(template) = infer_object_name_from_verb(
            description,
            &["spawn", "create", "add", "place", "move", "put"],
        ) {
            actions.push(Action::SpawnObject {
                template,
                pose: Pose {
                    position: default_spawn_position(state),
                    ..Pose::default()
                },
            });
        }
    }

    if actions.is_empty() {
        return Err(WorldForgeError::NoFeasiblePlan {
            goal: description.to_string(),
            reason: "native planner could not interpret the requested goal".to_string(),
        });
    }

    Ok(actions)
}

fn goal_satisfied(goal: &PlanGoal, state: &WorldState) -> bool {
    goal_score(goal, state) >= 0.95
}

fn goal_score(goal: &PlanGoal, state: &WorldState) -> f32 {
    match goal {
        PlanGoal::Condition(condition) => {
            if evaluate_condition(condition, state) {
                1.0
            } else {
                0.0
            }
        }
        PlanGoal::TargetState(target) => target_state_score(state, target),
        PlanGoal::GoalImage(goal_image_tensor) => {
            goal_image::goal_image_similarity(goal_image_tensor, state).unwrap_or(0.0)
        }
        PlanGoal::Description(description) => description_goal_score(description, state),
    }
}

fn target_state_score(current: &WorldState, target: &WorldState) -> f32 {
    if target.scene.objects.is_empty() && current.scene.objects.is_empty() {
        return 1.0;
    }

    let mut score = 0.0;
    let mut components = 0.0;
    for target_object in target.scene.objects.values() {
        components += 1.0;
        let object_score = current
            .scene
            .get_object(&target_object.id)
            .or_else(|| find_object_by_name_or_label(current, &target_object.name.to_lowercase()))
            .map(|current_object| {
                distance_score(
                    current_object.pose.position,
                    target_object.pose.position,
                    0.1,
                )
            })
            .unwrap_or(0.0);
        score += object_score;
    }

    if let Some(target_weather) = weather_from_tags(&target.metadata.tags) {
        components += 1.0;
        score += if weather_from_tags(&current.metadata.tags) == Some(target_weather) {
            1.0
        } else {
            0.0
        };
    }

    if let Some(target_lighting) = lighting_from_tags(&target.metadata.tags) {
        components += 1.0;
        score += lighting_from_tags(&current.metadata.tags)
            .map(|value| distance_score_scalar(value, target_lighting, 0.5))
            .unwrap_or(0.0);
    }

    if components <= f32::EPSILON {
        0.5
    } else {
        (score / components).clamp(0.0, 1.0)
    }
}

fn description_goal_score(description: &str, state: &WorldState) -> f32 {
    let normalized = description.to_lowercase();
    let mut score = 0.0;
    let mut components = 0.0;

    if let Some(weather) = parse_weather_hint(&normalized) {
        components += 1.0;
        score += if weather_from_tags(&state.metadata.tags) == Some(weather) {
            1.0
        } else {
            0.0
        };
    }

    if let Some(time_of_day) = parse_lighting_hint(&normalized) {
        components += 1.0;
        score += lighting_from_tags(&state.metadata.tags)
            .map(|value| distance_score_scalar(value, time_of_day, 0.5))
            .unwrap_or(0.0);
    }

    if let Some(name) = infer_object_name_from_verb(description, &["remove", "delete"]) {
        components += 1.0;
        score += if find_object_by_name_or_label(state, &name.to_lowercase()).is_none() {
            1.0
        } else {
            0.0
        };
    }

    if let Some(template) = infer_object_name_from_verb(
        description,
        &["spawn", "create", "add", "place", "move", "put"],
    ) {
        components += 1.0;
        let object = find_object_by_name_or_label(state, &template.to_lowercase());
        let placement_score = if let Some(hint) = parse_relative_target_hint(state, &normalized) {
            object
                .map(|item| distance_score(item.pose.position, hint.target, hint.tolerance))
                .unwrap_or(0.0)
        } else if let Some(target) = parse_position_hint(description) {
            object
                .map(|item| distance_score(item.pose.position, target, 0.2))
                .unwrap_or(0.0)
        } else if object.is_some() {
            1.0
        } else {
            0.0
        };
        score += placement_score;
    }

    if let Some(target) = parse_position_hint(description)
        .or_else(|| parse_relative_target_hint(state, &normalized).map(|hint| hint.target))
    {
        components += 1.0;
        let object = infer_object_name_from_verb(description, &["move", "place", "put"])
            .and_then(|name| find_object_by_name_or_label(state, &name.to_lowercase()))
            .or_else(|| primary_movable_object(state));
        score += object
            .map(|item| distance_score(item.pose.position, target, 0.2))
            .unwrap_or(0.0);
    } else if let Some(name) = infer_object_name_from_verb(description, &["move", "place", "put"]) {
        components += 1.0;
        score += if find_object_by_name_or_label(state, &name.to_lowercase()).is_some() {
            1.0
        } else {
            0.0
        };
    }

    if components == 0.0 {
        actions_for_description(description, state)
            .ok()
            .map(|actions| if actions.is_empty() { 0.0 } else { 0.5 })
            .unwrap_or(0.0)
    } else {
        (score / components).clamp(0.0, 1.0)
    }
}

fn distance_score(current: Position, target: Position, tolerance: f32) -> f32 {
    let distance = current.distance(target);
    if distance <= tolerance {
        1.0
    } else {
        (1.0 - ((distance - tolerance) / 2.0)).clamp(0.0, 1.0)
    }
}

fn distance_score_scalar(current: f32, target: f32, tolerance: f32) -> f32 {
    let distance = (current - target).abs();
    if distance <= tolerance {
        1.0
    } else {
        (1.0 - ((distance - tolerance) / 6.0)).clamp(0.0, 1.0)
    }
}

fn default_spawn_position(state: &WorldState) -> Position {
    if state.scene.objects.is_empty() {
        return Position::default();
    }

    let mut objects: Vec<_> = state.scene.objects.values().collect();
    objects.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });

    if let Some(anchor) = objects.last() {
        Position {
            x: anchor.pose.position.x + 0.6,
            y: anchor.pose.position.y.max(anchor.bbox.max.y + 0.2),
            z: anchor.pose.position.z,
        }
    } else {
        Position::default()
    }
}

fn parse_weather_hint(input: &str) -> Option<Weather> {
    [
        (Weather::Rain, "rain"),
        (Weather::Snow, "snow"),
        (Weather::Fog, "fog"),
        (Weather::Cloudy, "cloudy"),
        (Weather::Night, "night"),
        (Weather::Clear, "clear"),
    ]
    .into_iter()
    .find_map(|(weather, needle)| input.contains(needle).then_some(weather))
}

fn parse_lighting_hint(input: &str) -> Option<f32> {
    for marker in ["lighting", "time of day", "time"] {
        if let Some(index) = input.find(marker) {
            let suffix = &input[index + marker.len()..];
            if let Some(value) = first_number(suffix) {
                return Some(value.clamp(0.0, 24.0));
            }
        }
    }
    None
}

fn parse_position_hint(input: &str) -> Option<Position> {
    if let (Some(start), Some(end)) = (input.find('('), input.rfind(')')) {
        if start < end {
            let values: Vec<f32> = input[start + 1..end]
                .split([',', ' '])
                .filter(|token| !token.trim().is_empty())
                .filter_map(|token| token.trim().parse::<f32>().ok())
                .collect();
            if values.len() >= 3 {
                return Some(Position {
                    x: values[0],
                    y: values[1],
                    z: values[2],
                });
            }
        }
    }

    let words: Vec<&str> = input.split_whitespace().collect();
    for window in words.windows(4) {
        if matches!(window[0], "to" | "at" | "position") {
            let parsed = (
                window[1].parse::<f32>(),
                window[2].parse::<f32>(),
                window[3].parse::<f32>(),
            );
            if let (Ok(x), Ok(y), Ok(z)) = parsed {
                return Some(Position { x, y, z });
            }
        }
    }
    None
}

fn first_number(input: &str) -> Option<f32> {
    input.split_whitespace().find_map(|token| {
        token
            .trim_matches(|ch: char| !ch.is_ascii_digit() && ch != '.' && ch != '-')
            .parse::<f32>()
            .ok()
    })
}

fn infer_object_name_from_verb(input: &str, verbs: &[&str]) -> Option<String> {
    let normalized = input.to_lowercase();
    let delimiters = [
        " next to ",
        " beside ",
        " near ",
        " on top of ",
        " above ",
        " below ",
        " under ",
        " left of ",
        " right of ",
        " in front of ",
        " behind ",
        " at ",
        " to position ",
        " to (",
    ];

    for verb in verbs {
        if let Some(index) = normalized.find(verb) {
            let mut remainder = input[index + verb.len()..].trim();
            let mut remainder_lower = remainder.to_lowercase();

            for article in ["a ", "an ", "the "] {
                if remainder_lower.starts_with(article) {
                    remainder = remainder[article.len()..].trim_start();
                    remainder_lower = remainder.to_lowercase();
                    break;
                }
            }

            if remainder.is_empty() {
                continue;
            }

            let end_index = delimiters
                .iter()
                .filter_map(|delimiter| remainder_lower.find(delimiter))
                .min()
                .unwrap_or(remainder.len());

            let token = remainder[..end_index].trim().trim_matches(|ch: char| {
                !ch.is_alphanumeric() && ch != ' ' && ch != '_' && ch != '-'
            });
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }

    None
}

#[derive(Debug, Clone, Copy)]
enum RelativePlacement {
    NextTo,
    OnTopOf,
    Below,
    LeftOf,
    RightOf,
    InFrontOf,
    Behind,
}

#[derive(Debug, Clone, Copy)]
struct RelativeTargetHint {
    target: Position,
    tolerance: f32,
}

fn parse_relative_target_hint(state: &WorldState, description: &str) -> Option<RelativeTargetHint> {
    let mut best: Option<(usize, usize, RelativeTargetHint)> = None;

    for object in state.scene.objects.values() {
        let mut aliases = vec![object.name.to_lowercase()];
        if let Some(label) = &object.semantic_label {
            let lowered = label.to_lowercase();
            if !aliases.contains(&lowered) {
                aliases.push(lowered);
            }
        }

        for alias in aliases {
            for (phrase, placement) in [
                ("next to", RelativePlacement::NextTo),
                ("beside", RelativePlacement::NextTo),
                ("near", RelativePlacement::NextTo),
                ("on top of", RelativePlacement::OnTopOf),
                ("above", RelativePlacement::OnTopOf),
                ("below", RelativePlacement::Below),
                ("under", RelativePlacement::Below),
                ("left of", RelativePlacement::LeftOf),
                ("right of", RelativePlacement::RightOf),
                ("in front of", RelativePlacement::InFrontOf),
                ("behind", RelativePlacement::Behind),
            ] {
                let position = [
                    format!("{phrase} {alias}"),
                    format!("{phrase} the {alias}"),
                    format!("{phrase} a {alias}"),
                    format!("{phrase} an {alias}"),
                ]
                .into_iter()
                .filter_map(|pattern| description.find(&pattern))
                .min();
                let Some(position) = position else {
                    continue;
                };

                let target = relative_target_position(object, placement);
                let tolerance = (approximate_radius(object) + 0.15).clamp(0.15, 0.5);
                let candidate = (
                    position,
                    alias.len(),
                    RelativeTargetHint { target, tolerance },
                );

                match best {
                    Some((best_position, best_alias_len, _)) => {
                        if position < best_position
                            || (position == best_position && alias.len() > best_alias_len)
                        {
                            best = Some(candidate);
                        }
                    }
                    None => best = Some(candidate),
                }
            }
        }
    }

    best.map(|(_, _, hint)| hint)
}

fn relative_target_position(anchor: &SceneObject, placement: RelativePlacement) -> Position {
    let clearance = (approximate_radius(anchor) + 0.2).clamp(0.2, 0.8);
    match placement {
        RelativePlacement::NextTo => Position {
            x: anchor.pose.position.x + clearance,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z,
        },
        RelativePlacement::OnTopOf => Position {
            x: anchor.pose.position.x,
            y: anchor.pose.position.y + clearance,
            z: anchor.pose.position.z,
        },
        RelativePlacement::Below => Position {
            x: anchor.pose.position.x,
            y: anchor.pose.position.y - clearance,
            z: anchor.pose.position.z,
        },
        RelativePlacement::LeftOf => Position {
            x: anchor.pose.position.x - clearance,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z,
        },
        RelativePlacement::RightOf => Position {
            x: anchor.pose.position.x + clearance,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z,
        },
        RelativePlacement::InFrontOf => Position {
            x: anchor.pose.position.x,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z + clearance,
        },
        RelativePlacement::Behind => Position {
            x: anchor.pose.position.x,
            y: anchor.pose.position.y,
            z: anchor.pose.position.z - clearance,
        },
    }
}

fn approximate_radius(object: &SceneObject) -> f32 {
    let half_width = (object.bbox.max.x - object.bbox.min.x).abs() * 0.5;
    let half_depth = (object.bbox.max.z - object.bbox.min.z).abs() * 0.5;
    half_width.max(half_depth).max(0.05)
}

fn find_object_by_name_or_label<'a>(state: &'a WorldState, query: &str) -> Option<&'a SceneObject> {
    let mut objects: Vec<_> = state.scene.objects.values().collect();
    objects.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });
    objects
        .into_iter()
        .find(|object| object_name_matches(object, query))
}

fn object_name_matches(object: &SceneObject, query: &str) -> bool {
    let name = object.name.to_lowercase();
    if name == query || name.contains(query) {
        return true;
    }

    object
        .semantic_label
        .as_ref()
        .map(|label| {
            let label = label.to_lowercase();
            label == query || label.contains(query)
        })
        .unwrap_or(false)
}

fn primary_movable_object(state: &WorldState) -> Option<&SceneObject> {
    let mut objects: Vec<_> = state
        .scene
        .objects
        .values()
        .filter(|object| !object.physics.is_static)
        .collect();
    objects.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });
    objects.into_iter().next()
}

fn primary_movable_object_id(state: &WorldState, target: Option<Position>) -> Option<uuid::Uuid> {
    if let Some(target) = target {
        let mut objects: Vec<_> = state.scene.objects.values().collect();
        objects.sort_by(|left, right| {
            let left_distance = left.pose.position.distance(target);
            let right_distance = right.pose.position.distance(target);
            left_distance
                .partial_cmp(&right_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.physics.is_static.cmp(&right.physics.is_static))
                .then_with(|| left.name.cmp(&right.name))
                .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
        });
        objects
            .into_iter()
            .find(|object| !object.physics.is_static)
            .map(|object| object.id)
            .or_else(|| primary_movable_object(state).map(|object| object.id))
    } else {
        primary_movable_object(state).map(|object| object.id)
    }
}

fn weather_from_tags(tags: &[String]) -> Option<Weather> {
    tags.iter().find_map(|tag| {
        let normalized = tag.to_lowercase();
        if let Some(value) = normalized.strip_prefix("weather:") {
            match value {
                "clear" => Some(Weather::Clear),
                "cloudy" => Some(Weather::Cloudy),
                "rain" => Some(Weather::Rain),
                "snow" => Some(Weather::Snow),
                "fog" => Some(Weather::Fog),
                "night" => Some(Weather::Night),
                _ => None,
            }
        } else {
            None
        }
    })
}

fn lighting_from_tags(tags: &[String]) -> Option<f32> {
    tags.iter().find_map(|tag| {
        tag.to_lowercase()
            .strip_prefix("lighting:")
            .and_then(|value| value.parse::<f32>().ok())
    })
}

fn replace_tag(tags: &mut Vec<String>, prefix: &str, replacement: String) {
    tags.retain(|tag| !tag.to_lowercase().starts_with(prefix));
    tags.push(replacement);
    tags.sort();
}

fn quaternion_from_axis_angle(axis: Vec3, angle_degrees: f32) -> Rotation {
    let axis = axis.normalized();
    let radians = angle_degrees.to_radians() * 0.5;
    let sin = radians.sin();
    Rotation {
        w: radians.cos(),
        x: axis.x * sin,
        y: axis.y * sin,
        z: axis.z * sin,
    }
}

fn multiply_rotation(left: Rotation, right: Rotation) -> Rotation {
    Rotation {
        w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
        x: left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        y: left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        z: left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
    }
}
