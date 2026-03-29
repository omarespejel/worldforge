//! Marble provider adapter.
//!
//! This is an experimental deterministic local surrogate for the Marble
//! family named in the project spec. It does not implement a real remote
//! API; instead it provides stable prediction, generation, reasoning,
//! planning, transfer, embedding, and health-check behavior for development
//! and tests.

use std::collections::hash_map::DefaultHasher;
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use crate::native_planning;
use worldforge_core::action::{
    evaluate_condition, Action, ActionSpaceType, ActionTranslator, ActionType, Condition,
    ProviderAction,
};
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::prediction::{PhysicsScores, Plan, PlanRequest, Prediction, PredictionConfig};
use worldforge_core::provider::{
    CostEstimate, EmbeddingInput, EmbeddingOutput, GenerationConfig, GenerationPrompt,
    HealthStatus, LatencyProfile, Operation, ProviderCapabilities, ReasoningInput, ReasoningOutput,
    SpatialControls, TransferConfig, WorldModelProvider,
};
use worldforge_core::scene::{PhysicsProperties, SceneObject};
use worldforge_core::state::WorldState;
use worldforge_core::types::{
    BBox, CameraPose, DType, Device, Frame, Pose, Position, Rotation, SimTime, Tensor, TensorData,
    Vec3, Velocity, VideoClip,
};

const DEFAULT_NAME: &str = "marble";
const MODEL_NAME: &str = "marble-local-surrogate";
const EMBEDDING_DIM: usize = 48;
const MAX_OUTPUT_FRAMES: usize = 12;
const HEALTH_LATENCY_MS: u64 = 4;

/// Experimental deterministic Marble provider.
#[derive(Debug, Clone)]
pub struct MarbleProvider {
    name: String,
}

impl MarbleProvider {
    /// Create a new Marble provider with the default registry name.
    pub fn new() -> Self {
        Self {
            name: DEFAULT_NAME.to_string(),
        }
    }

    /// Create a Marble provider with a custom registry name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for MarbleProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MarbleProvider {
    fn action_payload(&self, action: &Action) -> Value {
        serde_json::json!({
            "kind": "marble_program",
            "model": MODEL_NAME,
            "registry_name": self.name,
            "space": self.action_space(action),
            "instruction": self.action_instruction(action),
            "parameters": self.action_parameters(action),
        })
    }

    fn action_space(&self, action: &Action) -> &'static str {
        match action {
            Action::Move { .. }
            | Action::Grasp { .. }
            | Action::Release { .. }
            | Action::Push { .. }
            | Action::Rotate { .. }
            | Action::Place { .. }
            | Action::CameraMove { .. }
            | Action::CameraLookAt { .. }
            | Action::Navigate { .. } => "continuous",
            Action::Teleport { .. }
            | Action::SetWeather { .. }
            | Action::SetLighting { .. }
            | Action::SpawnObject { .. }
            | Action::RemoveObject { .. } => "discrete",
            Action::Sequence(_)
            | Action::Parallel(_)
            | Action::Conditional { .. }
            | Action::Raw { .. } => "language",
        }
    }

    fn action_instruction(&self, action: &Action) -> &'static str {
        match action {
            Action::Move { .. } => "move object toward target",
            Action::Grasp { .. } => "grasp object",
            Action::Release { .. } => "release object",
            Action::Push { .. } => "push object along direction",
            Action::Rotate { .. } => "rotate object around axis",
            Action::Place { .. } => "place object at target",
            Action::CameraMove { .. } => "move camera pose",
            Action::CameraLookAt { .. } => "orient camera toward target",
            Action::Navigate { .. } => "navigate through waypoints",
            Action::Teleport { .. } => "teleport scene actor to pose",
            Action::SetWeather { .. } => "update weather state",
            Action::SetLighting { .. } => "update lighting state",
            Action::SpawnObject { .. } => "spawn object from template",
            Action::RemoveObject { .. } => "remove object from scene",
            Action::Sequence(_) => "execute ordered marble program",
            Action::Parallel(_) => "execute parallel marble program",
            Action::Conditional { .. } => "execute conditional marble program",
            Action::Raw { .. } => "inject provider-specific marble instruction",
        }
    }

    fn action_parameters(&self, action: &Action) -> Value {
        match action {
            Action::Move { target, speed } => serde_json::json!({
                "target": self.position_payload(target),
                "speed": speed,
                "policy": "trajectory",
            }),
            Action::Grasp { object, grip_force } => serde_json::json!({
                "object": object,
                "grip_force": grip_force,
                "policy": "manipulation",
            }),
            Action::Release { object } => serde_json::json!({
                "object": object,
                "policy": "manipulation",
            }),
            Action::Push {
                object,
                direction,
                force,
            } => serde_json::json!({
                "object": object,
                "direction": self.vec3_payload(direction),
                "force": force,
                "policy": "trajectory",
            }),
            Action::Rotate {
                object,
                axis,
                angle,
            } => serde_json::json!({
                "object": object,
                "axis": self.vec3_payload(axis),
                "angle": angle,
                "policy": "trajectory",
            }),
            Action::Place { object, target } => serde_json::json!({
                "object": object,
                "target": self.position_payload(target),
                "policy": "trajectory",
            }),
            Action::CameraMove { delta } => serde_json::json!({
                "delta": self.pose_payload(delta),
                "policy": "camera_control",
            }),
            Action::CameraLookAt { target } => serde_json::json!({
                "target": self.position_payload(target),
                "policy": "camera_control",
            }),
            Action::Navigate { waypoints } => serde_json::json!({
                "waypoints": waypoints
                    .iter()
                    .map(|waypoint| self.position_payload(waypoint))
                    .collect::<Vec<_>>(),
                "policy": "trajectory",
            }),
            Action::Teleport { destination } => serde_json::json!({
                "destination": self.pose_payload(destination),
                "policy": "instant_reposition",
            }),
            Action::SetWeather { weather } => serde_json::json!({
                "weather": weather,
                "policy": "scene_edit",
            }),
            Action::SetLighting { time_of_day } => serde_json::json!({
                "time_of_day": time_of_day,
                "policy": "scene_edit",
            }),
            Action::SpawnObject { template, pose } => serde_json::json!({
                "template": template,
                "pose": self.pose_payload(pose),
                "policy": "scene_edit",
            }),
            Action::RemoveObject { object } => serde_json::json!({
                "object": object,
                "policy": "scene_edit",
            }),
            Action::Sequence(actions) => serde_json::json!({
                "steps": actions.iter().map(|nested| self.action_payload(nested)).collect::<Vec<_>>(),
            }),
            Action::Parallel(actions) => serde_json::json!({
                "steps": actions.iter().map(|nested| self.action_payload(nested)).collect::<Vec<_>>(),
            }),
            Action::Conditional {
                condition,
                then,
                otherwise,
            } => serde_json::json!({
                "condition": self.condition_payload(condition),
                "then": self.action_payload(then),
                "otherwise": otherwise.as_deref().map(|nested| self.action_payload(nested)),
            }),
            Action::Raw { provider, data } => serde_json::json!({
                "provider": provider,
                "payload": data,
            }),
        }
    }

    fn condition_payload(&self, condition: &Condition) -> Value {
        match condition {
            Condition::ObjectAt {
                object,
                position,
                tolerance,
            } => serde_json::json!({
                "kind": "object_at",
                "object": object,
                "position": self.position_payload(position),
                "tolerance": tolerance,
            }),
            Condition::ObjectsTouching { a, b } => serde_json::json!({
                "kind": "objects_touching",
                "a": a,
                "b": b,
            }),
            Condition::ObjectExists { object } => serde_json::json!({
                "kind": "object_exists",
                "object": object,
            }),
            Condition::And(conditions) => serde_json::json!({
                "kind": "and",
                "conditions": conditions
                    .iter()
                    .map(|nested| self.condition_payload(nested))
                    .collect::<Vec<_>>(),
            }),
            Condition::Or(conditions) => serde_json::json!({
                "kind": "or",
                "conditions": conditions
                    .iter()
                    .map(|nested| self.condition_payload(nested))
                    .collect::<Vec<_>>(),
            }),
            Condition::Not(inner) => serde_json::json!({
                "kind": "not",
                "condition": self.condition_payload(inner),
            }),
        }
    }

    fn position_payload(&self, position: &Position) -> Value {
        serde_json::json!({
            "x": position.x,
            "y": position.y,
            "z": position.z,
        })
    }

    fn vec3_payload(&self, vector: &Vec3) -> Value {
        serde_json::json!({
            "x": vector.x,
            "y": vector.y,
            "z": vector.z,
        })
    }

    fn pose_payload(&self, pose: &Pose) -> Value {
        serde_json::json!({
            "position": self.position_payload(&pose.position),
            "rotation": {
                "w": pose.rotation.w,
                "x": pose.rotation.x,
                "y": pose.rotation.y,
                "z": pose.rotation.z,
            },
        })
    }
}

#[async_trait]
impl WorldModelProvider for MarbleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: true,
            transfer: true,
            embed: true,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 6.0,
            max_resolution: (1280, 720),
            fps_range: (6.0, 24.0),
            supported_action_spaces: vec![
                ActionSpaceType::Continuous,
                ActionSpaceType::Discrete,
                ActionSpaceType::Language,
            ],
            supports_depth: true,
            supports_segmentation: true,
            supports_planning: true,
            supports_gradient_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 22,
                p95_ms: 40,
                p99_ms: 90,
                throughput_fps: 48.0,
            },
        }
    }

    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        let mut output_state = state.clone();
        let changed = apply_action(&mut output_state, action);
        output_state.time = advance_time(state.time, config);
        if changed {
            output_state.scene.refresh_relationships();
        }

        let physics_scores = score_transition(state, &output_state, action, config.steps);
        let confidence = confidence_from_scores(&physics_scores);
        let latency_ms =
            estimate_prediction_latency(config.steps, output_state.scene.objects.len());
        let video = if config.return_video {
            let seed = transition_seed(state, action, config)?;
            Some(render_video_clip(
                seed,
                config.steps.max(1).min(MAX_OUTPUT_FRAMES as u32) as usize,
                config.resolution,
                config.fps,
                config.return_depth,
                config.return_segmentation,
            ))
        } else {
            None
        };

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: self.name.clone(),
            model: MODEL_NAME.to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video,
            confidence,
            physics_scores,
            latency_ms,
            cost: self.estimate_cost(&Operation::Predict {
                steps: config.steps,
                resolution: config.resolution,
            }),
            provenance: None,
            sampling: None,
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip> {
        let seed = hash_serialized(&(prompt, config, &self.name))?;
        Ok(render_video_clip(
            seed,
            requested_frame_count(config.duration_seconds, config.fps),
            config.resolution,
            config.fps,
            true,
            true,
        ))
    }

    async fn reason(&self, input: &ReasoningInput, query: &str) -> Result<ReasoningOutput> {
        if input.state.is_none() && input.video.is_none() {
            return Err(WorldForgeError::InvalidState(
                "reasoning input must include state and/or video".to_string(),
            ));
        }

        Ok(reason_about_input(input, query))
    }

    async fn embed(&self, input: &EmbeddingInput) -> Result<EmbeddingOutput> {
        input.validate()?;
        let seed = hash_serialized(&(input, &self.name))?;
        let embedding = Tensor {
            data: TensorData::Float32(deterministic_embedding(seed, EMBEDDING_DIM)),
            shape: vec![EMBEDDING_DIM],
            dtype: DType::Float32,
            device: Device::Cpu,
        };

        Ok(EmbeddingOutput {
            provider: self.name.clone(),
            model: MODEL_NAME.to_string(),
            embedding,
        })
    }

    async fn transfer(
        &self,
        source: &VideoClip,
        controls: &SpatialControls,
        config: &TransferConfig,
    ) -> Result<VideoClip> {
        let seed = hash_serialized(&(source, controls, config, &self.name))?;
        let frame_count = source.frames.len().clamp(1, MAX_OUTPUT_FRAMES);
        Ok(render_video_clip(
            seed,
            frame_count,
            config.resolution,
            config.fps,
            true,
            true,
        ))
    }

    async fn plan(&self, request: &PlanRequest) -> Result<Plan> {
        let step_cost = self.estimate_cost(&Operation::Predict {
            steps: 1,
            resolution: (320, 180),
        });
        native_planning::plan_native(self.name.as_str(), request, step_cost)
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            healthy: true,
            message: "marble local surrogate is ready".to_string(),
            latency_ms: HEALTH_LATENCY_MS,
        })
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        let (estimated_latency_ms, credits) = match operation {
            Operation::Predict { steps, resolution } => {
                let pixels = u64::from(resolution.0.max(1)) * u64::from(resolution.1.max(1));
                (
                    10 + u64::from((*steps).max(1)) * 4 + pixels / 250_000,
                    0.01 + (*steps as f64) * 0.002,
                )
            }
            Operation::Generate {
                duration_seconds,
                resolution,
            } => {
                let pixels = u64::from(resolution.0.max(1)) * u64::from(resolution.1.max(1));
                (
                    14 + duration_seconds.max(0.0).ceil() as u64 * 3 + pixels / 220_000,
                    0.02 + duration_seconds.max(0.0) * 0.004,
                )
            }
            Operation::Reason => (6, 0.005),
            Operation::Transfer { duration_seconds } => (
                12 + duration_seconds.max(0.0).ceil() as u64 * 2,
                0.015 + duration_seconds.max(0.0) * 0.003,
            ),
        };

        CostEstimate {
            usd: 0.0,
            credits,
            estimated_latency_ms,
        }
    }

    fn translate_action(&self, action: &Action) -> Result<ProviderAction> {
        ActionTranslator::translate(self, action)
    }

    fn supported_actions(&self) -> Vec<ActionType> {
        ActionTranslator::supported_actions(self)
    }
}

impl ActionTranslator for MarbleProvider {
    fn translate(&self, action: &Action) -> Result<ProviderAction> {
        Ok(ProviderAction {
            provider: self.name().to_string(),
            data: self.action_payload(action),
        })
    }

    fn supported_actions(&self) -> Vec<ActionType> {
        ActionType::all()
    }
}

fn advance_time(time: SimTime, config: &PredictionConfig) -> SimTime {
    let fps = config.fps.max(1.0) as f64;
    let steps = config.steps.max(1) as f64;
    SimTime {
        step: time.step.saturating_add(config.steps.max(1) as u64),
        seconds: time.seconds + steps / fps,
        dt: 1.0 / fps,
    }
}

fn estimate_prediction_latency(steps: u32, object_count: usize) -> u64 {
    10 + u64::from(steps.max(1)) * 4 + u64::try_from(object_count).unwrap_or(u64::MAX).min(32)
}

fn confidence_from_scores(scores: &PhysicsScores) -> f32 {
    (0.08 + scores.overall * 0.88).clamp(0.0, 1.0)
}

fn score_transition(
    input: &WorldState,
    output: &WorldState,
    action: &Action,
    steps: u32,
) -> PhysicsScores {
    let input_objects = input.scene.list_objects();
    let output_objects = output.scene.list_objects();
    let input_count = input_objects.len();
    let mut preserved = 0usize;
    let mut total_shift = 0.0f32;
    let mut max_shift = 0.0f32;
    for object in &input_objects {
        if let Some(next) = output.scene.get_object(&object.id) {
            preserved += 1;
            let shift = object.pose.position.distance(next.pose.position);
            total_shift += shift;
            max_shift = max_shift.max(shift);
        }
    }

    let object_preservation = if input_count == 0 {
        1.0
    } else {
        preserved as f32 / input_count as f32
    };
    let average_shift = if preserved == 0 {
        0.0
    } else {
        total_shift / preserved as f32
    };
    let spatial_consistency = (1.0 - average_shift / 1.6).clamp(0.0, 1.0);
    let gravity_compliance = if output_objects
        .iter()
        .all(|object| object.pose.position.y > -2.0)
    {
        0.95
    } else {
        0.72
    };
    let collision_accuracy = if output.scene.relationships.is_empty() {
        0.9
    } else {
        0.86
    };
    let action_weight = action_weight(action);
    let temporal_consistency = (1.0 - (steps as f32 * 0.012)).clamp(0.0, 1.0);
    let overall = (object_preservation * 0.34
        + spatial_consistency * 0.24
        + gravity_compliance * 0.15
        + collision_accuracy * 0.15
        + temporal_consistency * 0.12)
        .clamp(0.0, 1.0);

    PhysicsScores {
        overall: (overall * 0.9 + action_weight * 0.1).clamp(0.0, 1.0),
        object_permanence: object_preservation,
        gravity_compliance,
        collision_accuracy,
        spatial_consistency: (spatial_consistency * 0.9 + action_weight * 0.1).clamp(0.0, 1.0),
        temporal_consistency,
    }
}

fn action_weight(action: &Action) -> f32 {
    match action {
        Action::Move { .. } | Action::Place { .. } | Action::Teleport { .. } => 0.92,
        Action::Grasp { .. } | Action::Release { .. } => 0.88,
        Action::Push { .. } | Action::Rotate { .. } | Action::Navigate { .. } => 0.84,
        Action::CameraMove { .. } | Action::CameraLookAt { .. } => 0.9,
        Action::SetWeather { .. } | Action::SetLighting { .. } => 0.96,
        Action::SpawnObject { .. } | Action::RemoveObject { .. } => 0.8,
        Action::Sequence(actions) | Action::Parallel(actions) => {
            (0.78 + (actions.len() as f32 * 0.01)).clamp(0.0, 1.0)
        }
        Action::Conditional { .. } => 0.85,
        Action::Raw { .. } => 0.72,
    }
}

fn apply_action(state: &mut WorldState, action: &Action) -> bool {
    match action {
        Action::Move { target, speed } => {
            let Some(object_id) = select_object_id(state, Some(*target)) else {
                return false;
            };
            if let Some(object) = state.scene.get_object_mut(&object_id) {
                let blend = ((*speed * 0.14) + 0.2).clamp(0.12, 0.92);
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
                state.scene.refresh_relationships();
                delta.magnitude() > f32::EPSILON
            } else {
                false
            }
        }
        Action::Grasp { object, grip_force } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                let lift = (*grip_force).clamp(0.0, 25.0) * 0.001;
                item.pose.position.y += lift;
                item.bbox.translate(Vec3 {
                    x: 0.0,
                    y: lift,
                    z: 0.0,
                });
                item.velocity = Velocity::default();
                state.scene.refresh_relationships();
                true
            } else {
                false
            }
        }
        Action::Release { object } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                item.velocity = Velocity::default();
                true
            } else {
                false
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
                    .scale((*force).clamp(0.0, 50.0) * 0.035);
                item.translate_by(push);
                item.velocity = Velocity {
                    x: push.x * 2.0,
                    y: push.y * 2.0,
                    z: push.z * 2.0,
                };
                state.scene.refresh_relationships();
                push.magnitude() > f32::EPSILON
            } else {
                false
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
                true
            } else {
                false
            }
        }
        Action::Place { object, target } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                item.set_position(*target);
                item.velocity = Velocity::default();
                state.scene.refresh_relationships();
                true
            } else {
                false
            }
        }
        Action::CameraMove { .. } | Action::CameraLookAt { .. } => false,
        Action::Navigate { waypoints } => {
            let Some(final_waypoint) = waypoints.last().copied() else {
                return false;
            };
            let Some(object_id) = select_object_id(state, Some(final_waypoint)) else {
                return false;
            };
            if let Some(item) = state.scene.get_object_mut(&object_id) {
                item.set_position(final_waypoint);
                item.velocity = Velocity::default();
                state.scene.refresh_relationships();
                true
            } else {
                false
            }
        }
        Action::Teleport { destination } => {
            let Some(object_id) = select_object_id(state, Some(destination.position)) else {
                return false;
            };
            if let Some(item) = state.scene.get_object_mut(&object_id) {
                item.pose = *destination;
                let half_extents = item.half_extents();
                let half_extents = Vec3 {
                    x: half_extents.x.max(0.08),
                    y: half_extents.y.max(0.08),
                    z: half_extents.z.max(0.08),
                };
                item.bbox = BBox::from_center_half_extents(destination.position, half_extents);
                item.velocity = Velocity::default();
                state.scene.refresh_relationships();
                true
            } else {
                false
            }
        }
        Action::SetWeather { weather } => {
            replace_tag(
                &mut state.metadata.tags,
                "weather:",
                format!("weather:{weather:?}").to_lowercase(),
            );
            true
        }
        Action::SetLighting { time_of_day } => {
            replace_tag(
                &mut state.metadata.tags,
                "lighting:",
                format!("lighting:{time_of_day:.2}"),
            );
            true
        }
        Action::SpawnObject { template, pose } => {
            let object = spawn_object(template, *pose);
            state.scene.add_object(object);
            true
        }
        Action::RemoveObject { object } => state.scene.remove_object(object).is_some(),
        Action::Sequence(actions) | Action::Parallel(actions) => {
            let mut changed = false;
            for nested in actions {
                changed |= apply_action(state, nested);
            }
            if changed {
                state.scene.refresh_relationships();
            }
            changed
        }
        Action::Conditional {
            condition,
            then,
            otherwise,
        } => {
            let branch = if evaluate_condition(condition, state) {
                Some(then.as_ref())
            } else {
                otherwise.as_deref()
            };
            branch.is_some_and(|action| apply_action(state, action))
        }
        Action::Raw { provider, data } => {
            let payload = canonical_json(data);
            let hash = hash_string(&format!("{provider}:{payload}"));
            replace_tag(&mut state.metadata.tags, "raw:", format!("raw:{hash:016x}"));
            true
        }
    }
}

fn reason_about_input(input: &ReasoningInput, query: &str) -> ReasoningOutput {
    let normalized = query.to_lowercase();
    if let Some(state) = input.state.as_ref() {
        let objects = state.scene.list_objects();
        let object_names: Vec<String> = objects
            .iter()
            .map(|object| object.name.clone())
            .take(5)
            .collect();
        let evidence = objects
            .iter()
            .map(|object| {
                format!(
                    "{} at ({:.2}, {:.2}, {:.2})",
                    object.name,
                    object.pose.position.x,
                    object.pose.position.y,
                    object.pose.position.z
                )
            })
            .take(5)
            .collect::<Vec<_>>();

        let answer = if normalized.contains("how many") || normalized.contains("count") {
            format!(
                "Marble sees {} object(s): {}.",
                objects.len(),
                if object_names.is_empty() {
                    "none".to_string()
                } else {
                    object_names.join(", ")
                }
            )
        } else if normalized.contains("weather") {
            let weather = state
                .metadata
                .tags
                .iter()
                .find_map(|tag| tag.strip_prefix("weather:"))
                .unwrap_or("unknown");
            format!("The scene metadata suggests weather={weather}.")
        } else if normalized.contains("lighting") {
            let lighting = state
                .metadata
                .tags
                .iter()
                .find_map(|tag| tag.strip_prefix("lighting:"))
                .unwrap_or("unknown");
            format!("The scene metadata suggests lighting={lighting}.")
        } else {
            format!(
                "Marble can answer '{}' from a scene with {} object(s).",
                query,
                objects.len()
            )
        };

        return ReasoningOutput {
            answer,
            confidence: 0.84,
            evidence,
        };
    }

    if let Some(video) = input.video.as_ref() {
        return ReasoningOutput {
            answer: format!(
                "Marble analyzed {} frame(s) at {:.1} fps for '{}'.",
                video.frames.len(),
                video.fps,
                query
            ),
            confidence: 0.77,
            evidence: vec![format!(
                "video: {} frame(s), resolution {:?}, duration {:.2}s",
                video.frames.len(),
                video.resolution,
                video.duration
            )],
        };
    }

    ReasoningOutput {
        answer: format!("Marble has no scene or video context for '{query}'."),
        confidence: 0.3,
        evidence: vec!["context: unavailable".to_string()],
    }
}

fn select_object_id(state: &WorldState, target: Option<Position>) -> Option<uuid::Uuid> {
    let mut objects: Vec<&SceneObject> = state.scene.objects.values().collect();
    objects.sort_by(|a, b| {
        let a_score = target
            .map(|t| a.pose.position.distance(t))
            .unwrap_or_default();
        let b_score = target
            .map(|t| b.pose.position.distance(t))
            .unwrap_or_default();
        a_score
            .partial_cmp(&b_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.physics.is_static.cmp(&b.physics.is_static))
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.id.as_bytes().cmp(b.id.as_bytes()))
    });
    objects.first().map(|object| object.id)
}

fn spawn_object(template: &str, pose: Pose) -> SceneObject {
    let seed = hash_string(
        &(template.to_lowercase()
            + &format!(
                ":{:.3}:{:.3}:{:.3}",
                pose.position.x, pose.position.y, pose.position.z
            )),
    );
    let half_extents = Vec3 {
        x: 0.05 + (seed & 0x0f) as f32 * 0.008,
        y: 0.05 + ((seed >> 4) & 0x0f) as f32 * 0.008,
        z: 0.05 + ((seed >> 8) & 0x0f) as f32 * 0.008,
    };
    let mut object = SceneObject::new(
        if template.trim().is_empty() {
            "marble-object"
        } else {
            template
        },
        pose,
        BBox::from_center_half_extents(pose.position, half_extents),
    );
    object.semantic_label = Some(template.to_lowercase());
    object.physics = PhysicsProperties {
        mass: Some(0.5 + ((seed >> 12) & 0x3f) as f32 * 0.1),
        friction: Some(0.25 + ((seed >> 18) & 0x1f) as f32 * 0.01),
        restitution: Some(0.1 + ((seed >> 24) & 0x1f) as f32 * 0.01),
        is_static: template.to_lowercase().contains("table")
            || template.to_lowercase().contains("floor")
            || template.to_lowercase().contains("wall"),
        is_graspable: !template.to_lowercase().contains("table")
            && !template.to_lowercase().contains("floor")
            && !template.to_lowercase().contains("wall"),
        material: Some(material_name(seed).to_string()),
    };
    object
}

fn replace_tag(tags: &mut Vec<String>, prefix: &str, replacement: String) {
    tags.retain(|tag| !tag.starts_with(prefix));
    tags.push(replacement);
}

fn material_name(seed: u64) -> &'static str {
    match seed % 4 {
        0 => "wood",
        1 => "metal",
        2 => "glass",
        _ => "plastic",
    }
}

fn quaternion_from_axis_angle(axis: Vec3, angle_degrees: f32) -> Rotation {
    let normalized = axis.normalized();
    if normalized.magnitude() < f32::EPSILON {
        return Rotation {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }

    let half_angle = angle_degrees.to_radians() * 0.5;
    let (sin_half, cos_half) = half_angle.sin_cos();
    Rotation {
        w: cos_half,
        x: normalized.x * sin_half,
        y: normalized.y * sin_half,
        z: normalized.z * sin_half,
    }
}

fn multiply_rotation(lhs: Rotation, rhs: Rotation) -> Rotation {
    Rotation {
        w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
        x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
        y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
        z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
    }
}

fn requested_frame_count(duration_seconds: f64, fps: f32) -> usize {
    let duration_seconds = duration_seconds.max(0.0);
    let fps = fps.max(1.0);
    ((duration_seconds * f64::from(fps)).round() as usize).clamp(1, MAX_OUTPUT_FRAMES)
}

fn render_video_clip(
    seed: u64,
    frame_count: usize,
    resolution: (u32, u32),
    fps: f32,
    include_depth: bool,
    include_segmentation: bool,
) -> VideoClip {
    let fps = fps.max(1.0);
    let mut frames = Vec::with_capacity(frame_count);
    for frame_index in 0..frame_count {
        let frame_seed = splitmix64(seed.wrapping_add(frame_index as u64));
        frames.push(render_frame(
            frame_seed,
            frame_index as u64,
            resolution,
            fps,
            include_depth,
            include_segmentation,
        ));
    }

    VideoClip {
        frames,
        fps,
        resolution,
        duration: frame_count as f64 / f64::from(fps),
    }
}

fn render_frame(
    seed: u64,
    frame_index: u64,
    resolution: (u32, u32),
    fps: f32,
    include_depth: bool,
    include_segmentation: bool,
) -> Frame {
    let timestamp = SimTime {
        step: frame_index,
        seconds: frame_index as f64 / f64::from(fps.max(1.0)),
        dt: 1.0 / f64::from(fps.max(1.0)),
    };
    let width = resolution.0.max(1) as usize;
    let height = resolution.1.max(1) as usize;
    let frame_seed = splitmix64(seed ^ frame_index.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let data = build_rgb_tensor(frame_seed, width, height);
    let camera = Some(camera_pose(frame_seed));
    let depth = include_depth.then(|| build_depth_tensor(frame_seed, width, height));
    let segmentation =
        include_segmentation.then(|| build_segmentation_tensor(frame_seed, width, height));

    Frame {
        data,
        timestamp,
        camera,
        depth,
        segmentation,
    }
}

fn build_rgb_tensor(seed: u64, width: usize, height: usize) -> Tensor {
    let mut data = Vec::with_capacity(width * height * 3);
    for idx in 0..(width * height) {
        let pixel_seed = splitmix64(seed.wrapping_add(idx as u64));
        data.push((pixel_seed & 0xFF) as u8);
        data.push(((pixel_seed >> 8) & 0xFF) as u8);
        data.push(((pixel_seed >> 16) & 0xFF) as u8);
    }

    Tensor {
        data: TensorData::UInt8(data),
        shape: vec![height, width, 3],
        dtype: DType::UInt8,
        device: Device::Cpu,
    }
}

fn build_depth_tensor(seed: u64, width: usize, height: usize) -> Tensor {
    let mut values = Vec::with_capacity(width * height);
    for idx in 0..(width * height) {
        let value = normalized_float(splitmix64(seed ^ idx as u64));
        values.push(value);
    }

    Tensor {
        data: TensorData::Float32(values),
        shape: vec![height, width],
        dtype: DType::Float32,
        device: Device::Cpu,
    }
}

fn build_segmentation_tensor(seed: u64, width: usize, height: usize) -> Tensor {
    let mut values = Vec::with_capacity(width * height);
    for idx in 0..(width * height) {
        let value = (splitmix64(seed.wrapping_add((idx * 31) as u64)) % 8) as i32;
        values.push(value);
    }

    Tensor {
        data: TensorData::Int32(values),
        shape: vec![height, width],
        dtype: DType::Int32,
        device: Device::Cpu,
    }
}

fn camera_pose(seed: u64) -> CameraPose {
    let jitter = normalized_float(seed) * 0.2;
    CameraPose {
        extrinsics: Pose {
            position: Position {
                x: jitter,
                y: 1.1 + jitter * 0.5,
                z: 2.5,
            },
            rotation: Rotation {
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        },
        fov: 58.0 + jitter * 10.0,
        near_clip: 0.01,
        far_clip: 100.0,
    }
}

fn deterministic_embedding(seed: u64, dims: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(dims);
    let mut state = seed;
    for _ in 0..dims {
        state = splitmix64(state);
        let unit = normalized_float(state);
        values.push(unit * 2.0 - 1.0);
    }
    values
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn normalized_float(seed: u64) -> f32 {
    let value = splitmix64(seed);
    (value as f64 / u64::MAX as f64) as f32
}

fn transition_seed(state: &WorldState, action: &Action, config: &PredictionConfig) -> Result<u64> {
    let mut hasher = DefaultHasher::new();
    state_signature(state).hash(&mut hasher);
    action_signature(action).hash(&mut hasher);
    hash_serialized(config)?.hash(&mut hasher);
    Ok(hasher.finish())
}

fn state_signature(state: &WorldState) -> u64 {
    let mut hasher = DefaultHasher::new();
    state.id.hash(&mut hasher);
    state.time.step.hash(&mut hasher);
    state.time.seconds.to_bits().hash(&mut hasher);
    state.time.dt.to_bits().hash(&mut hasher);
    state.metadata.name.hash(&mut hasher);
    state.metadata.description.hash(&mut hasher);
    state.metadata.created_by.hash(&mut hasher);
    state
        .metadata
        .created_at
        .timestamp_nanos_opt()
        .unwrap_or_default()
        .hash(&mut hasher);
    let mut tags = state.metadata.tags.clone();
    tags.sort();
    tags.hash(&mut hasher);

    let objects = state.scene.list_objects();
    for object in objects {
        object.id.hash(&mut hasher);
        object.name.hash(&mut hasher);
        object.pose.position.x.to_bits().hash(&mut hasher);
        object.pose.position.y.to_bits().hash(&mut hasher);
        object.pose.position.z.to_bits().hash(&mut hasher);
        object.pose.rotation.w.to_bits().hash(&mut hasher);
        object.pose.rotation.x.to_bits().hash(&mut hasher);
        object.pose.rotation.y.to_bits().hash(&mut hasher);
        object.pose.rotation.z.to_bits().hash(&mut hasher);
        object.bbox.min.x.to_bits().hash(&mut hasher);
        object.bbox.min.y.to_bits().hash(&mut hasher);
        object.bbox.min.z.to_bits().hash(&mut hasher);
        object.bbox.max.x.to_bits().hash(&mut hasher);
        object.bbox.max.y.to_bits().hash(&mut hasher);
        object.bbox.max.z.to_bits().hash(&mut hasher);
        object.velocity.x.to_bits().hash(&mut hasher);
        object.velocity.y.to_bits().hash(&mut hasher);
        object.velocity.z.to_bits().hash(&mut hasher);
        object.semantic_label.hash(&mut hasher);
        object.physics.is_static.hash(&mut hasher);
        object.physics.is_graspable.hash(&mut hasher);
        if let Some(mesh) = object.mesh.as_ref() {
            mesh.vertices.len().hash(&mut hasher);
            mesh.faces.len().hash(&mut hasher);
            mesh.normals.as_ref().map(Vec::len).hash(&mut hasher);
            mesh.uvs.as_ref().map(Vec::len).hash(&mut hasher);
        }
        if let Some(embedding) = object.visual_embedding.as_ref() {
            hash_tensor(embedding).hash(&mut hasher);
        }
    }

    let mut relationships = state
        .scene
        .relationships
        .iter()
        .map(relationship_signature)
        .collect::<Vec<_>>();
    relationships.sort();
    relationships.hash(&mut hasher);

    hasher.finish()
}

fn hash_tensor(tensor: &Tensor) -> u64 {
    let mut hasher = DefaultHasher::new();
    tensor.shape.hash(&mut hasher);
    tensor.dtype.hash(&mut hasher);
    match &tensor.data {
        TensorData::Float16(values) => values.hash(&mut hasher),
        TensorData::Float32(values) => {
            for value in values {
                value.to_bits().hash(&mut hasher);
            }
        }
        TensorData::Float64(values) => {
            for value in values {
                value.to_bits().hash(&mut hasher);
            }
        }
        TensorData::BFloat16(values) => values.hash(&mut hasher),
        TensorData::UInt8(values) => values.hash(&mut hasher),
        TensorData::Int32(values) => values.hash(&mut hasher),
        TensorData::Int64(values) => values.hash(&mut hasher),
    }
    hasher.finish()
}

fn relationship_signature(relationship: &worldforge_core::scene::SpatialRelationship) -> String {
    match relationship {
        worldforge_core::scene::SpatialRelationship::On { subject, surface } => {
            format!("on:{subject}:{surface}")
        }
        worldforge_core::scene::SpatialRelationship::In { subject, container } => {
            format!("in:{subject}:{container}")
        }
        worldforge_core::scene::SpatialRelationship::Near { a, b, distance } => {
            format!("near:{a}:{b}:{distance:.4}")
        }
        worldforge_core::scene::SpatialRelationship::Touching { a, b } => {
            format!("touching:{a}:{b}")
        }
        worldforge_core::scene::SpatialRelationship::Above { subject, reference } => {
            format!("above:{subject}:{reference}")
        }
        worldforge_core::scene::SpatialRelationship::Below { subject, reference } => {
            format!("below:{subject}:{reference}")
        }
    }
}

fn action_signature(action: &Action) -> u64 {
    let mut hasher = DefaultHasher::new();
    match action {
        Action::Move { target, speed } => {
            "move".hash(&mut hasher);
            target.x.to_bits().hash(&mut hasher);
            target.y.to_bits().hash(&mut hasher);
            target.z.to_bits().hash(&mut hasher);
            speed.to_bits().hash(&mut hasher);
        }
        Action::Grasp { object, grip_force } => {
            "grasp".hash(&mut hasher);
            object.hash(&mut hasher);
            grip_force.to_bits().hash(&mut hasher);
        }
        Action::Release { object } => {
            "release".hash(&mut hasher);
            object.hash(&mut hasher);
        }
        Action::Push {
            object,
            direction,
            force,
        } => {
            "push".hash(&mut hasher);
            object.hash(&mut hasher);
            direction.x.to_bits().hash(&mut hasher);
            direction.y.to_bits().hash(&mut hasher);
            direction.z.to_bits().hash(&mut hasher);
            force.to_bits().hash(&mut hasher);
        }
        Action::Rotate {
            object,
            axis,
            angle,
        } => {
            "rotate".hash(&mut hasher);
            object.hash(&mut hasher);
            axis.x.to_bits().hash(&mut hasher);
            axis.y.to_bits().hash(&mut hasher);
            axis.z.to_bits().hash(&mut hasher);
            angle.to_bits().hash(&mut hasher);
        }
        Action::Place { object, target } => {
            "place".hash(&mut hasher);
            object.hash(&mut hasher);
            target.x.to_bits().hash(&mut hasher);
            target.y.to_bits().hash(&mut hasher);
            target.z.to_bits().hash(&mut hasher);
        }
        Action::CameraMove { delta } => {
            "camera-move".hash(&mut hasher);
            delta.position.x.to_bits().hash(&mut hasher);
            delta.position.y.to_bits().hash(&mut hasher);
            delta.position.z.to_bits().hash(&mut hasher);
            delta.rotation.w.to_bits().hash(&mut hasher);
            delta.rotation.x.to_bits().hash(&mut hasher);
            delta.rotation.y.to_bits().hash(&mut hasher);
            delta.rotation.z.to_bits().hash(&mut hasher);
        }
        Action::CameraLookAt { target } => {
            "camera-look-at".hash(&mut hasher);
            target.x.to_bits().hash(&mut hasher);
            target.y.to_bits().hash(&mut hasher);
            target.z.to_bits().hash(&mut hasher);
        }
        Action::Navigate { waypoints } => {
            "navigate".hash(&mut hasher);
            for waypoint in waypoints {
                waypoint.x.to_bits().hash(&mut hasher);
                waypoint.y.to_bits().hash(&mut hasher);
                waypoint.z.to_bits().hash(&mut hasher);
            }
        }
        Action::Teleport { destination } => {
            "teleport".hash(&mut hasher);
            destination.position.x.to_bits().hash(&mut hasher);
            destination.position.y.to_bits().hash(&mut hasher);
            destination.position.z.to_bits().hash(&mut hasher);
            destination.rotation.w.to_bits().hash(&mut hasher);
            destination.rotation.x.to_bits().hash(&mut hasher);
            destination.rotation.y.to_bits().hash(&mut hasher);
            destination.rotation.z.to_bits().hash(&mut hasher);
        }
        Action::SetWeather { weather } => {
            "set-weather".hash(&mut hasher);
            weather.hash(&mut hasher);
        }
        Action::SetLighting { time_of_day } => {
            "set-lighting".hash(&mut hasher);
            time_of_day.to_bits().hash(&mut hasher);
        }
        Action::SpawnObject { template, pose } => {
            "spawn-object".hash(&mut hasher);
            template.hash(&mut hasher);
            pose.position.x.to_bits().hash(&mut hasher);
            pose.position.y.to_bits().hash(&mut hasher);
            pose.position.z.to_bits().hash(&mut hasher);
            pose.rotation.w.to_bits().hash(&mut hasher);
            pose.rotation.x.to_bits().hash(&mut hasher);
            pose.rotation.y.to_bits().hash(&mut hasher);
            pose.rotation.z.to_bits().hash(&mut hasher);
        }
        Action::RemoveObject { object } => {
            "remove-object".hash(&mut hasher);
            object.hash(&mut hasher);
        }
        Action::Sequence(actions) => {
            "sequence".hash(&mut hasher);
            for nested in actions {
                action_signature(nested).hash(&mut hasher);
            }
        }
        Action::Parallel(actions) => {
            "parallel".hash(&mut hasher);
            for nested in actions {
                action_signature(nested).hash(&mut hasher);
            }
        }
        Action::Conditional {
            condition,
            then,
            otherwise,
        } => {
            "conditional".hash(&mut hasher);
            condition_signature(condition).hash(&mut hasher);
            action_signature(then).hash(&mut hasher);
            otherwise
                .as_deref()
                .map(action_signature)
                .unwrap_or_default()
                .hash(&mut hasher);
        }
        Action::Raw { provider, data } => {
            "raw".hash(&mut hasher);
            provider.hash(&mut hasher);
            canonical_json(data).hash(&mut hasher);
        }
    }

    hasher.finish()
}

fn condition_signature(condition: &Condition) -> String {
    match condition {
        Condition::ObjectAt {
            object,
            position,
            tolerance,
        } => format!(
            "object-at:{object}:{:.4}:{:.4}:{:.4}:{tolerance:.4}",
            position.x, position.y, position.z
        ),
        Condition::ObjectsTouching { a, b } => format!("touching:{a}:{b}"),
        Condition::ObjectExists { object } => format!("exists:{object}"),
        Condition::And(conditions) => {
            let inner = conditions
                .iter()
                .map(condition_signature)
                .collect::<Vec<_>>()
                .join("|");
            format!("and:[{inner}]")
        }
        Condition::Or(conditions) => {
            let inner = conditions
                .iter()
                .map(condition_signature)
                .collect::<Vec<_>>()
                .join("|");
            format!("or:[{inner}]")
        }
        Condition::Not(inner) => format!("not:{}", condition_signature(inner)),
    }
}

fn hash_serialized<T: Serialize>(value: &T) -> Result<u64> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| WorldForgeError::SerializationError(error.to_string()))?;
    Ok(hash_bytes(&bytes))
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn hash_string(value: &str) -> u64 {
    hash_bytes(value.as_bytes())
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string()),
        Value::Array(values) => {
            let mut out = String::from("[");
            for (idx, value) in values.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                out.push_str(&canonical_json(value));
            }
            out.push(']');
            out
        }
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            let mut out = String::from("{");
            for (idx, (key, value)) in entries.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                let _ = write!(
                    &mut out,
                    "{}",
                    serde_json::to_string(key).unwrap_or_default()
                );
                out.push(':');
                out.push_str(&canonical_json(value));
            }
            out.push('}');
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::action::{ActionTranslator, ActionType};
    use worldforge_core::prediction::{PlanGoal, PlanRequest, PlannerType};

    use worldforge_core::action::{Action, Condition, Weather};
    use worldforge_core::provider::{EmbeddingInput, GenerationConfig, GenerationPrompt};
    use worldforge_core::scene::SceneObject;

    fn sample_world() -> (WorldState, uuid::Uuid) {
        let mut world = WorldState::new("marble-sample", "marble");
        let object = SceneObject::new(
            "mug",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.8,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox::from_center_half_extents(
                Position {
                    x: 0.0,
                    y: 0.8,
                    z: 0.0,
                },
                Vec3 {
                    x: 0.05,
                    y: 0.05,
                    z: 0.05,
                },
            ),
        );
        let object_id = object.id;
        world.scene.add_object(object);
        (world, object_id)
    }

    #[test]
    fn marble_capabilities_are_conservative() {
        let provider = MarbleProvider::new();
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(caps.generate);
        assert!(caps.reason);
        assert!(caps.transfer);
        assert!(caps.embed);
        assert!(caps.supports_depth);
        assert!(caps.supports_segmentation);
        assert!(caps.supports_planning);
        assert!(!caps.supports_gradient_planning);
        assert_eq!(provider.name(), "marble");
    }

    #[test]
    fn marble_action_translator_exposes_multi_space_payloads() {
        let provider = MarbleProvider::new();
        assert_eq!(
            worldforge_core::provider::WorldModelProvider::supported_actions(&provider),
            ActionType::all()
        );

        let action = Action::Sequence(vec![
            Action::Move {
                target: Position {
                    x: 0.4,
                    y: 0.9,
                    z: -0.2,
                },
                speed: 0.6,
            },
            Action::SetWeather {
                weather: Weather::Snow,
            },
        ]);

        let translated = provider.translate(&action).unwrap();
        assert_eq!(translated.provider, "marble");
        assert_eq!(translated.data["kind"], "marble_program");
        assert_eq!(translated.data["space"], "language");
        assert_eq!(
            translated.data["instruction"],
            "execute ordered marble program"
        );
        assert_eq!(
            translated.data["parameters"]["steps"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            translated.data["parameters"]["steps"][0]["space"],
            "continuous"
        );
        assert_eq!(
            translated.data["parameters"]["steps"][0]["parameters"]["policy"],
            "trajectory"
        );
        assert_eq!(
            translated.data["parameters"]["steps"][1]["space"],
            "discrete"
        );
        assert_eq!(
            translated.data["parameters"]["steps"][1]["parameters"]["policy"],
            "scene_edit"
        );
    }

    #[tokio::test]
    async fn marble_predict_is_deterministic_and_applies_actions() {
        let provider = MarbleProvider::new();
        let (world, object_id) = sample_world();
        let action = Action::Sequence(vec![
            Action::Place {
                object: object_id,
                target: Position {
                    x: 0.3,
                    y: 0.8,
                    z: 0.0,
                },
            },
            Action::Conditional {
                condition: Condition::ObjectExists { object: object_id },
                then: Box::new(Action::SetWeather {
                    weather: Weather::Rain,
                }),
                otherwise: Some(Box::new(Action::SetWeather {
                    weather: Weather::Clear,
                })),
            },
        ]);
        let config = PredictionConfig {
            steps: 3,
            resolution: (320, 180),
            fps: 12.0,
            return_video: true,
            return_depth: true,
            return_segmentation: true,
            ..PredictionConfig::default()
        };

        let first = provider.predict(&world, &action, &config).await.unwrap();
        let second = provider.predict(&world, &action, &config).await.unwrap();

        assert_eq!(first.provider, "marble");
        assert_eq!(first.model, MODEL_NAME);
        assert_eq!(first.output_state.time.step, world.time.step + 3);
        assert!(first
            .output_state
            .metadata
            .tags
            .iter()
            .any(|tag| tag == "weather:rain"));
        let moved = first.output_state.scene.get_object(&object_id).unwrap();
        assert!((moved.pose.position.x - 0.3).abs() < f32::EPSILON);
        assert_eq!(first.physics_scores.overall, second.physics_scores.overall);
        assert_eq!(
            first.output_state.scene.objects.len(),
            second.output_state.scene.objects.len()
        );
        assert!(first.video.is_some());
        assert_eq!(
            serde_json::to_string(&first.video).unwrap(),
            serde_json::to_string(&second.video).unwrap()
        );
    }

    #[tokio::test]
    async fn marble_media_reasoning_and_embedding_are_deterministic() {
        let provider = MarbleProvider::with_name("marble-test");
        let (world, _) = sample_world();

        let prompt = GenerationPrompt {
            text: "A bright kitchen with a red mug".to_string(),
            reference_image: None,
            negative_prompt: Some("low quality".to_string()),
        };
        let config = GenerationConfig {
            resolution: (256, 144),
            fps: 12.0,
            duration_seconds: 1.5,
            temperature: 0.7,
            seed: Some(7),
        };
        let clip_a = provider.generate(&prompt, &config).await.unwrap();
        let clip_b = provider.generate(&prompt, &config).await.unwrap();

        assert_eq!(clip_a.resolution, (256, 144));
        assert!(!clip_a.frames.is_empty());
        assert_eq!(
            serde_json::to_string(&clip_a).unwrap(),
            serde_json::to_string(&clip_b).unwrap()
        );

        let reasoning = provider
            .reason(
                &ReasoningInput {
                    video: None,
                    state: Some(world.clone()),
                },
                "how many objects are in the scene?",
            )
            .await
            .unwrap();
        assert!(reasoning.answer.contains("1 object"));
        assert!(reasoning.evidence.iter().any(|entry| entry.contains("mug")));

        let embedding_input = EmbeddingInput::from_text("mug on a counter");
        let embedding_a = provider.embed(&embedding_input).await.unwrap();
        let embedding_b = provider.embed(&embedding_input).await.unwrap();
        assert_eq!(embedding_a.provider, "marble-test");
        assert_eq!(embedding_a.model, MODEL_NAME);
        assert_eq!(
            serde_json::to_string(&embedding_a).unwrap(),
            serde_json::to_string(&embedding_b).unwrap()
        );

        let transfer_config = TransferConfig {
            resolution: (160, 90),
            fps: 8.0,
            control_strength: 0.6,
        };
        let transferred_a = provider
            .transfer(&clip_a, &SpatialControls::default(), &transfer_config)
            .await
            .unwrap();
        let transferred_b = provider
            .transfer(&clip_a, &SpatialControls::default(), &transfer_config)
            .await
            .unwrap();
        assert_eq!(transferred_a.resolution, (160, 90));
        assert_eq!(
            serde_json::to_string(&transferred_a).unwrap(),
            serde_json::to_string(&transferred_b).unwrap()
        );

        let health = provider.health_check().await.unwrap();
        assert!(health.healthy);
        assert!(health.message.contains("marble"));
    }

    #[tokio::test]
    async fn marble_native_planning_is_deterministic() {
        let provider = MarbleProvider::new();
        let (world, object_id) = sample_world();
        let mut target = world.clone();
        {
            let object = target.scene.get_object_mut(&object_id).unwrap();
            object.set_position(Position {
                x: 0.4,
                y: 0.8,
                z: 0.0,
            });
        }

        let request = PlanRequest {
            current_state: world.clone(),
            goal: PlanGoal::TargetState(Box::new(target)),
            max_steps: 4,
            guardrails: Vec::new(),
            planner: PlannerType::Sampling {
                num_samples: 8,
                top_k: 2,
            },
            timeout_seconds: 2.0,
            fallback_provider: None,
        };

        let first = provider.plan(&request).await.unwrap();
        let second = provider.plan(&request).await.unwrap();

        assert_eq!(
            serde_json::to_string(&first.actions).unwrap(),
            serde_json::to_string(&second.actions).unwrap()
        );
        assert!(!first.actions.is_empty());
        assert_eq!(first.predicted_states.len(), first.actions.len());
        assert!(matches!(
            first.actions.first(),
            Some(Action::Place { object, .. }) if *object == object_id
        ));
        assert!(first.success_probability >= 0.95);
    }

    #[tokio::test]
    async fn marble_native_planning_rejects_impossible_conditions() {
        let provider = MarbleProvider::new();
        let (world, _) = sample_world();
        let missing = uuid::Uuid::new_v4();
        let request = PlanRequest {
            current_state: world,
            goal: PlanGoal::Condition(Condition::ObjectExists { object: missing }),
            max_steps: 2,
            guardrails: Vec::new(),
            planner: PlannerType::Sampling {
                num_samples: 4,
                top_k: 1,
            },
            timeout_seconds: 2.0,
            fallback_provider: None,
        };

        let error = provider.plan(&request).await.unwrap_err();
        match error {
            WorldForgeError::NoFeasiblePlan { goal, reason } => {
                assert!(goal.contains("ObjectExists"));
                assert!(reason.contains("immutable"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
