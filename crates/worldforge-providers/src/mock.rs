//! Mock provider for testing and development.
//!
//! Unlike a shallow stub, this provider maintains scene geometry,
//! infers spatial relationships, emits lightweight preview clips,
//! and answers simple reasoning queries from world state.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;
use worldforge_core::action::{evaluate_condition, Action, ActionSpaceType};
use worldforge_core::error::Result;
use worldforge_core::prediction::{PhysicsScores, Prediction, PredictionConfig};
use worldforge_core::provider::{
    CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
    ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
    WorldModelProvider,
};
use worldforge_core::scene::{SceneObject, SpatialRelationship};
use worldforge_core::state::WorldState;
use worldforge_core::types::{
    BBox, CameraPose, DType, Device, Frame, Pose, Position, Rotation, SimTime, Tensor, TensorData,
    Vec3, Velocity, VideoClip,
};

const MAX_PREVIEW_DIMENSION: u32 = 96;
const MAX_PREVIEW_FRAMES: usize = 8;

#[derive(Clone, Copy)]
struct RasterRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// A mock provider that returns deterministic predictions.
#[derive(Debug, Clone)]
pub struct MockProvider {
    /// Name of this mock instance.
    name: String,
    /// Simulated latency in milliseconds.
    pub latency_ms: u64,
    /// Default confidence score for predictions.
    pub default_confidence: f32,
}

impl MockProvider {
    /// Create a new mock provider with default settings.
    pub fn new() -> Self {
        Self {
            name: "mock".to_string(),
            latency_ms: 10,
            default_confidence: 0.85,
        }
    }

    /// Create a named mock provider.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::new()
        }
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorldModelProvider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: true,
            transfer: true,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 10.0,
            max_resolution: (1920, 1080),
            fps_range: (8.0, 30.0),
            supported_action_spaces: vec![
                ActionSpaceType::Continuous,
                ActionSpaceType::Discrete,
                ActionSpaceType::Language,
            ],
            supports_depth: true,
            supports_segmentation: true,
            supports_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 10,
                p95_ms: 20,
                p99_ms: 50,
                throughput_fps: 60.0,
            },
        }
    }

    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        if self.latency_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
        }

        let trajectory = simulate_prediction_trajectory(state, action, config);
        let output_state = trajectory.last().cloned().unwrap_or_else(|| state.clone());
        let physics_scores = score_prediction(state, &output_state, action);
        let confidence =
            (self.default_confidence * physics_scores.overall.max(0.35)).clamp(0.0, 1.0);
        let video = config.return_video.then(|| {
            render_state_clip(
                &trajectory,
                config.resolution,
                config.fps,
                config.return_depth,
                config.return_segmentation,
            )
        });

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: self.name.clone(),
            model: "mock-v2".to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video,
            confidence,
            physics_scores,
            latency_ms: self.latency_ms,
            cost: CostEstimate::default(),
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip> {
        tracing::info!(prompt = %prompt.text, "mock: generating preview clip");
        Ok(render_prompt_clip(prompt, config))
    }

    async fn reason(&self, input: &ReasoningInput, query: &str) -> Result<ReasoningOutput> {
        let normalized = query.to_lowercase();
        let (answer, evidence, confidence) = match input.state.as_ref() {
            Some(state) => reason_about_state(state, &normalized),
            None => (
                format!("No world state was provided, so I can only echo the query: {query}"),
                vec!["state: unavailable".to_string()],
                0.35,
            ),
        };

        Ok(ReasoningOutput {
            answer,
            confidence,
            evidence,
        })
    }

    async fn transfer(
        &self,
        source: &VideoClip,
        controls: &SpatialControls,
        config: &TransferConfig,
    ) -> Result<VideoClip> {
        Ok(render_transfer_clip(source, controls, config))
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            healthy: true,
            message: "mock provider is always healthy".to_string(),
            latency_ms: 1,
        })
    }

    fn estimate_cost(&self, _operation: &Operation) -> CostEstimate {
        CostEstimate {
            usd: 0.0,
            credits: 0.0,
            estimated_latency_ms: self.latency_ms,
        }
    }
}

fn simulate_prediction_trajectory(
    initial_state: &WorldState,
    action: &Action,
    config: &PredictionConfig,
) -> Vec<WorldState> {
    let steps = config.steps.max(1);
    let fps = config.fps.max(1.0);
    let dt = 1.0 / fps as f64;
    let dt_f32 = dt as f32;
    let mut state = initial_state.clone();
    let mut trajectory = Vec::with_capacity(steps as usize);

    for step_index in 0..steps {
        decay_velocities(&mut state);
        apply_mock_action_step(&mut state, action, step_index, steps, dt_f32);
        state.time.step += 1;
        state.time.seconds += dt;
        state.time.dt = dt;
        state.scene.refresh_relationships();
        trajectory.push(state.clone());
    }

    trajectory
}

fn apply_mock_action_step(
    state: &mut WorldState,
    action: &Action,
    step_index: u32,
    total_steps: u32,
    dt: f32,
) {
    match action {
        Action::Move { target, speed } => {
            if let Some(object_id) = primary_movable_object_id(state) {
                move_object_toward(
                    state,
                    object_id,
                    *target,
                    *speed,
                    step_index,
                    total_steps,
                    dt,
                );
            }
        }
        Action::Place { object, target } => {
            move_object_toward(state, *object, *target, 1.0, step_index, total_steps, dt);
        }
        Action::Push {
            object,
            direction,
            force,
        } => {
            let delta = direction
                .normalized()
                .scale(force.max(0.1) * 0.15 / total_steps.max(1) as f32);
            translate_object(state, *object, delta, dt);
        }
        Action::Rotate {
            object,
            axis,
            angle,
        } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                let progress = (step_index + 1) as f32 / total_steps.max(1) as f32;
                item.pose.rotation = axis_angle_rotation(*axis, *angle * progress);
            }
        }
        Action::Teleport { destination } => {
            if step_index == 0 {
                if let Some(object_id) = primary_movable_object_id(state) {
                    if state
                        .scene
                        .set_object_position(&object_id, destination.position)
                    {
                        if let Some(item) = state.scene.get_object_mut(&object_id) {
                            item.pose.rotation = destination.rotation;
                            item.velocity = Velocity::default();
                        }
                    }
                }
            }
        }
        Action::Navigate { waypoints } => {
            if let Some(object_id) = primary_movable_object_id(state) {
                if let Some(target) = waypoint_for_progress(waypoints, step_index, total_steps) {
                    move_object_toward(state, object_id, target, 1.0, step_index, total_steps, dt);
                }
            }
        }
        Action::CameraMove { delta } => {
            if step_index == 0 {
                upsert_metadata_tag(
                    state,
                    "camera-offset:",
                    format!(
                        "{:.2},{:.2},{:.2}",
                        delta.position.x, delta.position.y, delta.position.z
                    ),
                );
            }
        }
        Action::CameraLookAt { target } => {
            if step_index == 0 {
                upsert_metadata_tag(
                    state,
                    "camera-look-at:",
                    format!("{:.2},{:.2},{:.2}", target.x, target.y, target.z),
                );
            }
        }
        Action::SetWeather { weather } => {
            if step_index == 0 {
                upsert_metadata_tag(state, "weather:", format!("{weather:?}"));
            }
        }
        Action::SetLighting { time_of_day } => {
            if step_index == 0 {
                upsert_metadata_tag(state, "lighting:", format!("{time_of_day:.1}"));
            }
        }
        Action::SpawnObject { template, pose } => {
            if step_index == 0 {
                state.scene.add_object(spawn_object(template, *pose));
            }
        }
        Action::RemoveObject { object } => {
            if step_index == 0 {
                state.scene.remove_object(object);
            }
        }
        Action::Grasp { object, .. } => {
            if step_index == 0 {
                upsert_metadata_tag(state, "grasped:", object.to_string());
                if let Some(item) = state.scene.get_object_mut(object) {
                    item.velocity = Velocity::default();
                }
            }
        }
        Action::Release { object } => {
            if step_index == 0 {
                state
                    .metadata
                    .tags
                    .retain(|tag| tag != &format!("grasped:{object}"));
            }
        }
        Action::Sequence(actions) | Action::Parallel(actions) => {
            for sub_action in actions {
                apply_mock_action_step(state, sub_action, step_index, total_steps, dt);
            }
        }
        Action::Conditional {
            condition,
            then,
            otherwise,
        } => {
            if evaluate_condition(condition, state) {
                apply_mock_action_step(state, then, step_index, total_steps, dt);
            } else if let Some(otherwise) = otherwise {
                apply_mock_action_step(state, otherwise, step_index, total_steps, dt);
            }
        }
        Action::Raw { provider, .. } => {
            if step_index == 0 {
                upsert_metadata_tag(state, "raw-provider:", provider.clone());
            }
        }
    }
}

fn primary_movable_object_id(state: &WorldState) -> Option<uuid::Uuid> {
    let mut objects: Vec<_> = state.scene.objects.values().collect();
    objects.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });
    objects
        .iter()
        .find(|object| !object.physics.is_static)
        .or_else(|| objects.first())
        .map(|object| object.id)
}

fn move_object_toward(
    state: &mut WorldState,
    object_id: uuid::Uuid,
    target: Position,
    speed: f32,
    step_index: u32,
    total_steps: u32,
    dt: f32,
) {
    let Some(current) = state
        .scene
        .get_object(&object_id)
        .map(|object| object.pose.position)
    else {
        return;
    };

    let remaining_steps = total_steps.saturating_sub(step_index).max(1) as f32;
    let alpha = (speed.max(0.1) / remaining_steps).clamp(0.15, 1.0);
    let next = current.lerp(target, alpha);
    let delta = Vec3 {
        x: next.x - current.x,
        y: next.y - current.y,
        z: next.z - current.z,
    };

    if state.scene.set_object_position(&object_id, next) {
        set_object_velocity(state, object_id, delta, dt);
    }
}

fn translate_object(state: &mut WorldState, object_id: uuid::Uuid, delta: Vec3, dt: f32) {
    if state.scene.translate_object(&object_id, delta) {
        set_object_velocity(state, object_id, delta, dt);
    }
}

fn set_object_velocity(state: &mut WorldState, object_id: uuid::Uuid, delta: Vec3, dt: f32) {
    if let Some(object) = state.scene.get_object_mut(&object_id) {
        let safe_dt = dt.max(1e-3);
        object.velocity = Velocity {
            x: delta.x / safe_dt,
            y: delta.y / safe_dt,
            z: delta.z / safe_dt,
        };
    }
}

fn decay_velocities(state: &mut WorldState) {
    for object in state.scene.objects.values_mut() {
        object.velocity.x *= 0.5;
        object.velocity.y *= 0.5;
        object.velocity.z *= 0.5;
    }
}

fn spawn_object(template: &str, pose: Pose) -> SceneObject {
    let mut object = SceneObject::new(
        template,
        pose,
        BBox::from_center_half_extents(
            pose.position,
            Vec3 {
                x: 0.2,
                y: 0.2,
                z: 0.2,
            },
        ),
    );
    object.semantic_label = Some(template.to_string());
    object
}

fn upsert_metadata_tag(state: &mut WorldState, prefix: &str, value: String) {
    state.metadata.tags.retain(|tag| !tag.starts_with(prefix));
    state.metadata.tags.push(format!("{prefix}{value}"));
}

fn waypoint_for_progress(
    waypoints: &[Position],
    step_index: u32,
    total_steps: u32,
) -> Option<Position> {
    if waypoints.is_empty() {
        return None;
    }

    let progress = (step_index + 1) as usize * waypoints.len() / total_steps.max(1) as usize;
    let index = progress.saturating_sub(1).min(waypoints.len() - 1);
    Some(waypoints[index])
}

fn axis_angle_rotation(axis: Vec3, angle: f32) -> Rotation {
    let axis = axis.normalized();
    if axis.magnitude() < f32::EPSILON {
        return Rotation::default();
    }

    let half_angle = angle * 0.5;
    let sin_half = half_angle.sin();
    Rotation {
        w: half_angle.cos(),
        x: axis.x * sin_half,
        y: axis.y * sin_half,
        z: axis.z * sin_half,
    }
}

fn score_prediction(input: &WorldState, output: &WorldState, action: &Action) -> PhysicsScores {
    let collision_count = output
        .scene
        .relationships
        .iter()
        .filter(|relationship| matches!(relationship, SpatialRelationship::Touching { .. }))
        .count();
    let mean_speed = if output.scene.objects.is_empty() {
        0.0
    } else {
        output
            .scene
            .objects
            .values()
            .map(|object| object.velocity.magnitude())
            .sum::<f32>()
            / output.scene.objects.len() as f32
    };
    let object_delta = output.scene.objects.len() as i32 - input.scene.objects.len() as i32;
    let object_permanence = match action {
        Action::SpawnObject { .. } | Action::RemoveObject { .. } => 0.92,
        _ => (1.0 - object_delta.abs() as f32 * 0.2).clamp(0.5, 1.0),
    };
    let gravity_compliance = if output
        .scene
        .objects
        .values()
        .any(|object| object.bbox.min.y < -0.25)
    {
        0.55
    } else {
        0.95
    };
    let collision_accuracy = (1.0 - collision_count as f32 * 0.15).clamp(0.3, 1.0);
    let spatial_consistency = if output
        .scene
        .objects
        .values()
        .all(|object| object.pose.position.distance(object.bbox.center()) < 0.01)
    {
        0.96
    } else {
        0.65
    };
    let temporal_consistency = (1.0 - (mean_speed / 12.0)).clamp(0.35, 0.98);
    let overall = ((object_permanence
        + gravity_compliance
        + collision_accuracy
        + spatial_consistency
        + temporal_consistency)
        / 5.0)
        .clamp(0.0, 1.0);

    PhysicsScores {
        overall,
        object_permanence,
        gravity_compliance,
        collision_accuracy,
        spatial_consistency,
        temporal_consistency,
    }
}

fn render_state_clip(
    states: &[WorldState],
    resolution: (u32, u32),
    fps: f32,
    include_depth: bool,
    include_segmentation: bool,
) -> VideoClip {
    let sample_indices = sample_indices(states.len());
    let frames = sample_indices
        .into_iter()
        .map(|index| {
            render_state_frame(
                &states[index],
                resolution,
                include_depth,
                include_segmentation,
                None,
            )
        })
        .collect();

    VideoClip {
        frames,
        fps,
        resolution,
        duration: states.len() as f64 / fps.max(1.0) as f64,
    }
}

fn render_prompt_clip(prompt: &GenerationPrompt, config: &GenerationConfig) -> VideoClip {
    let (width, height) = preview_dimensions(config.resolution);
    let frame_count = preview_frame_count(config.duration_seconds, config.fps);
    let seed = hash_value(&(prompt.text.as_str(), prompt.negative_prompt.as_deref()));
    let background = color_from_seed(seed);
    let accent = color_from_seed(seed.rotate_left(11));

    let mut frames = Vec::with_capacity(frame_count);
    for index in 0..frame_count {
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        fill_background(&mut pixels, width, height, background);
        let progress = progress(index, frame_count);
        let rect_width = (width / 4).max(4);
        let rect_height = (height / 3).max(4);
        let x = ((width.saturating_sub(rect_width)) as f32 * progress) as u32;
        let y = ((height.saturating_sub(rect_height)) as f32 * (1.0 - progress * 0.5)) as u32;
        fill_rect(
            &mut pixels,
            width,
            height,
            RasterRect {
                x,
                y,
                width: rect_width,
                height: rect_height,
            },
            accent,
        );

        frames.push(Frame {
            data: uint8_tensor(pixels, width, height, 3),
            timestamp: SimTime {
                step: index as u64,
                seconds: progress as f64 * config.duration_seconds,
                dt: 1.0 / config.fps.max(1.0) as f64,
            },
            camera: Some(default_camera()),
            depth: None,
            segmentation: None,
        });
    }

    VideoClip {
        frames,
        fps: config.fps,
        resolution: config.resolution,
        duration: config.duration_seconds,
    }
}

fn render_transfer_clip(
    source: &VideoClip,
    controls: &SpatialControls,
    config: &TransferConfig,
) -> VideoClip {
    let (width, height) = preview_dimensions(config.resolution);
    let frame_count = sample_count_from_source(source);
    let seed = hash_value(&(
        source.duration.to_bits(),
        source.resolution,
        source.fps.to_bits(),
    ));
    let background = color_from_seed(seed);
    let accent = color_from_seed(seed.rotate_left(7));
    let duration = source
        .duration
        .max(frame_count as f64 / config.fps.max(1.0) as f64);

    let mut frames = Vec::with_capacity(frame_count);
    for index in 0..frame_count {
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        fill_background(&mut pixels, width, height, background);
        let progress = progress(index, frame_count);
        let bar_width = ((width as f32 * config.control_strength.clamp(0.1, 1.0)).round() as u32)
            .clamp(4, width.max(4));
        let x = ((width.saturating_sub(bar_width)) as f32 * progress) as u32;
        fill_rect(
            &mut pixels,
            width,
            height,
            RasterRect {
                x,
                y: height / 3,
                width: bar_width.min(width),
                height: (height / 4).max(4),
            },
            accent,
        );

        frames.push(Frame {
            data: uint8_tensor(pixels, width, height, 3),
            timestamp: SimTime {
                step: index as u64,
                seconds: progress as f64 * duration,
                dt: 1.0 / config.fps.max(1.0) as f64,
            },
            camera: camera_for_transfer(controls, index, frame_count),
            depth: None,
            segmentation: None,
        });
    }

    VideoClip {
        frames,
        fps: config.fps,
        resolution: config.resolution,
        duration,
    }
}

fn render_state_frame(
    state: &WorldState,
    resolution: (u32, u32),
    include_depth: bool,
    include_segmentation: bool,
    camera: Option<CameraPose>,
) -> Frame {
    let (width, height) = preview_dimensions(resolution);
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    let mut depth = include_depth.then(|| vec![1.0f32; (width * height) as usize]);
    let mut segmentation = include_segmentation.then(|| vec![0i32; (width * height) as usize]);
    let background = background_from_state(state);
    fill_background(&mut pixels, width, height, background);

    let (min_x, max_x, min_z, max_z) = world_bounds(state);
    let x_span = (max_x - min_x).max(0.5);
    let z_span = (max_z - min_z).max(0.5);
    let mut objects: Vec<_> = state.scene.objects.values().collect();
    objects.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });

    for (index, object) in objects.into_iter().enumerate() {
        let color = color_from_seed(hash_value(&(object.name.as_str(), object.id)));
        let x0 = project_range(object.bbox.min.x, min_x, x_span, width);
        let x1 = project_range(object.bbox.max.x, min_x, x_span, width);
        let z0 = project_range(object.bbox.min.z, min_z, z_span, height);
        let z1 = project_range(object.bbox.max.z, min_z, z_span, height);
        let y = height.saturating_sub(z1.max(1) + 1);
        let rect_height = z1.saturating_sub(z0).max(3);
        let rect_width = x1.saturating_sub(x0).max(3);
        let rect = RasterRect {
            x: x0.min(width.saturating_sub(1)),
            y: y.min(height.saturating_sub(1)),
            width: rect_width.min(width),
            height: rect_height.min(height),
        };
        fill_rect(&mut pixels, width, height, rect, color);

        if let Some(depth_map) = depth.as_mut() {
            fill_scalar_rect(
                depth_map,
                width,
                height,
                rect,
                (1.0 - object.pose.position.y.abs() / 10.0).clamp(0.05, 1.0),
            );
        }
        if let Some(segmentation_map) = segmentation.as_mut() {
            fill_int_rect(segmentation_map, width, height, rect, (index + 1) as i32);
        }
    }

    Frame {
        data: uint8_tensor(pixels, width, height, 3),
        timestamp: state.time,
        camera: Some(camera.unwrap_or_else(default_camera)),
        depth: depth.map(|buffer| float32_tensor(buffer, width, height)),
        segmentation: segmentation.map(|buffer| int32_tensor(buffer, width, height)),
    }
}

fn reason_about_state(state: &WorldState, query: &str) -> (String, Vec<String>, f32) {
    if state.scene.objects.is_empty() {
        return (
            "The scene is currently empty.".to_string(),
            vec!["objects: none".to_string()],
            0.9,
        );
    }

    let object_names = sorted_object_names(state);
    let relationships = relationship_descriptions(state);

    if query.contains("how many") || query.contains("count") {
        return (
            format!(
                "I can account for {} object(s): {}.",
                object_names.len(),
                object_names.join(", ")
            ),
            vec![format!("objects: {}", object_names.join(", "))],
            0.92,
        );
    }

    if query.contains("where") || query.contains("position") {
        if let Some(object) = find_mentioned_object(state, query) {
            return (
                format!(
                    "{} is at ({:.2}, {:.2}, {:.2}).",
                    object.name,
                    object.pose.position.x,
                    object.pose.position.y,
                    object.pose.position.z
                ),
                vec![format!(
                    "position:{}={:.2},{:.2},{:.2}",
                    object.name,
                    object.pose.position.x,
                    object.pose.position.y,
                    object.pose.position.z
                )],
                0.9,
            );
        }
    }

    if query.contains("touch") || query.contains("collision") {
        let touching = relationships
            .iter()
            .filter(|entry| entry.starts_with("touching:"))
            .cloned()
            .collect::<Vec<_>>();
        if touching.is_empty() {
            return (
                "I do not see any touching objects or collisions.".to_string(),
                vec!["touching: none".to_string()],
                0.84,
            );
        }
        return (
            format!("Touching relationships detected: {}.", touching.join("; ")),
            touching,
            0.82,
        );
    }

    if query.contains("fall") || query.contains("stable") {
        let unsupported = unsupported_objects(state);
        if unsupported.is_empty() {
            return (
                "The scene looks stable: I do not see an unsupported elevated object.".to_string(),
                relationships,
                0.76,
            );
        }
        return (
            format!(
                "{} may fall because it is elevated without support.",
                unsupported.join(", ")
            ),
            unsupported
                .iter()
                .map(|name| format!("unsupported:{name}"))
                .collect(),
            0.7,
        );
    }

    (
        format!(
            "The scene contains {} object(s): {}.",
            object_names.len(),
            object_names.join(", ")
        ),
        relationships,
        0.72,
    )
}

fn sorted_object_names(state: &WorldState) -> Vec<String> {
    let mut names: Vec<_> = state
        .scene
        .objects
        .values()
        .map(|object| object.name.clone())
        .collect();
    names.sort();
    names
}

fn find_mentioned_object<'a>(state: &'a WorldState, query: &str) -> Option<&'a SceneObject> {
    let mut objects: Vec<_> = state.scene.objects.values().collect();
    objects.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });
    objects.into_iter().find(|object| {
        query.contains(&object.name.to_lowercase())
            || object
                .semantic_label
                .as_ref()
                .is_some_and(|label| query.contains(&label.to_lowercase()))
    })
}

fn unsupported_objects(state: &WorldState) -> Vec<String> {
    let mut unsupported = Vec::new();
    for object in state.scene.objects.values() {
        if object.physics.is_static || object.bbox.min.y <= 0.05 {
            continue;
        }

        let supported = state.scene.relationships.iter().any(|relationship| {
            matches!(
                relationship,
                SpatialRelationship::On { subject, .. } | SpatialRelationship::In { subject, .. }
                    if *subject == object.id
            )
        });
        if !supported {
            unsupported.push(object.name.clone());
        }
    }
    unsupported.sort();
    unsupported
}

fn relationship_descriptions(state: &WorldState) -> Vec<String> {
    let mut descriptions: Vec<_> = state
        .scene
        .relationships
        .iter()
        .filter_map(|relationship| describe_relationship(state, relationship))
        .collect();
    descriptions.sort();
    descriptions
}

fn describe_relationship(state: &WorldState, relationship: &SpatialRelationship) -> Option<String> {
    match relationship {
        SpatialRelationship::Touching { a, b } => Some(format!(
            "touching:{}:{}",
            object_name(state, *a)?,
            object_name(state, *b)?
        )),
        SpatialRelationship::On { subject, surface } => Some(format!(
            "on:{}:{}",
            object_name(state, *subject)?,
            object_name(state, *surface)?
        )),
        SpatialRelationship::In { subject, container } => Some(format!(
            "in:{}:{}",
            object_name(state, *subject)?,
            object_name(state, *container)?
        )),
        SpatialRelationship::Near { a, b, distance } => Some(format!(
            "near:{}:{}:{distance:.2}",
            object_name(state, *a)?,
            object_name(state, *b)?
        )),
        SpatialRelationship::Above { subject, reference } => Some(format!(
            "above:{}:{}",
            object_name(state, *subject)?,
            object_name(state, *reference)?
        )),
        SpatialRelationship::Below { subject, reference } => Some(format!(
            "below:{}:{}",
            object_name(state, *subject)?,
            object_name(state, *reference)?
        )),
    }
}

fn object_name(state: &WorldState, object_id: uuid::Uuid) -> Option<&str> {
    state
        .scene
        .get_object(&object_id)
        .map(|object| object.name.as_str())
}

fn sample_indices(len: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let samples = len.min(MAX_PREVIEW_FRAMES);
    let mut indices = Vec::with_capacity(samples);
    for slot in 0..samples {
        let index = slot * len / samples;
        indices.push(index.min(len - 1));
    }
    indices
}

fn preview_frame_count(duration_seconds: f64, fps: f32) -> usize {
    ((duration_seconds.max(0.1) * fps.max(1.0) as f64).round() as usize)
        .clamp(1, MAX_PREVIEW_FRAMES)
}

fn sample_count_from_source(source: &VideoClip) -> usize {
    if source.frames.is_empty() {
        preview_frame_count(source.duration.max(0.25), source.fps.max(1.0))
    } else {
        source.frames.len().clamp(1, MAX_PREVIEW_FRAMES)
    }
}

fn preview_dimensions(resolution: (u32, u32)) -> (u32, u32) {
    let width = resolution.0.max(1);
    let height = resolution.1.max(1);
    let scale = (MAX_PREVIEW_DIMENSION as f32 / width as f32)
        .min(MAX_PREVIEW_DIMENSION as f32 / height as f32)
        .min(1.0);
    (
        (width as f32 * scale).round().max(1.0) as u32,
        (height as f32 * scale).round().max(1.0) as u32,
    )
}

fn world_bounds(state: &WorldState) -> (f32, f32, f32, f32) {
    if state.scene.objects.is_empty() {
        return (-2.0, 2.0, -2.0, 2.0);
    }

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for object in state.scene.objects.values() {
        min_x = min_x.min(object.bbox.min.x);
        max_x = max_x.max(object.bbox.max.x);
        min_z = min_z.min(object.bbox.min.z);
        max_z = max_z.max(object.bbox.max.z);
    }

    (min_x - 0.5, max_x + 0.5, min_z - 0.5, max_z + 0.5)
}

fn background_from_state(state: &WorldState) -> [u8; 3] {
    if let Some(weather) = state
        .metadata
        .tags
        .iter()
        .find_map(|tag| tag.strip_prefix("weather:"))
    {
        match weather.to_ascii_lowercase().as_str() {
            "rain" => return [55, 80, 120],
            "snow" => return [220, 228, 236],
            "fog" => return [180, 188, 196],
            "night" => return [24, 28, 48],
            _ => {}
        }
    }

    if let Some(time_of_day) = state
        .metadata
        .tags
        .iter()
        .find_map(|tag| tag.strip_prefix("lighting:"))
        .and_then(|value| value.parse::<f32>().ok())
    {
        if !(6.0..=20.0).contains(&time_of_day) {
            return [32, 36, 56];
        }
    }

    [176, 196, 214]
}

fn camera_for_transfer(
    controls: &SpatialControls,
    index: usize,
    frame_count: usize,
) -> Option<CameraPose> {
    controls
        .camera_trajectory
        .as_ref()
        .and_then(|trajectory| {
            if trajectory.poses.is_empty() {
                None
            } else {
                let sample = trajectory.poses[index * trajectory.poses.len() / frame_count];
                Some(CameraPose {
                    extrinsics: sample.1,
                    fov: 60.0,
                    near_clip: 0.1,
                    far_clip: 100.0,
                })
            }
        })
        .or_else(|| Some(default_camera()))
}

fn default_camera() -> CameraPose {
    CameraPose {
        extrinsics: Pose {
            position: Position {
                x: 0.0,
                y: 5.0,
                z: 5.0,
            },
            rotation: Rotation::default(),
        },
        fov: 60.0,
        near_clip: 0.1,
        far_clip: 100.0,
    }
}

fn project_range(value: f32, min: f32, span: f32, extent: u32) -> u32 {
    let normalized = ((value - min) / span).clamp(0.0, 1.0);
    (normalized * extent.saturating_sub(1) as f32) as u32
}

fn progress(index: usize, frame_count: usize) -> f32 {
    if frame_count <= 1 {
        0.0
    } else {
        index as f32 / (frame_count - 1) as f32
    }
}

fn fill_background(pixels: &mut [u8], width: u32, height: u32, color: [u8; 3]) {
    fill_rect(
        pixels,
        width,
        height,
        RasterRect {
            x: 0,
            y: 0,
            width,
            height,
        },
        color,
    );
}

fn fill_rect(pixels: &mut [u8], width: u32, height: u32, rect: RasterRect, color: [u8; 3]) {
    for row in rect.y..(rect.y + rect.height).min(height) {
        for col in rect.x..(rect.x + rect.width).min(width) {
            let offset = ((row * width + col) * 3) as usize;
            pixels[offset] = color[0];
            pixels[offset + 1] = color[1];
            pixels[offset + 2] = color[2];
        }
    }
}

fn fill_scalar_rect(buffer: &mut [f32], width: u32, height: u32, rect: RasterRect, value: f32) {
    for row in rect.y..(rect.y + rect.height).min(height) {
        for col in rect.x..(rect.x + rect.width).min(width) {
            buffer[(row * width + col) as usize] = value;
        }
    }
}

fn fill_int_rect(buffer: &mut [i32], width: u32, height: u32, rect: RasterRect, value: i32) {
    for row in rect.y..(rect.y + rect.height).min(height) {
        for col in rect.x..(rect.x + rect.width).min(width) {
            buffer[(row * width + col) as usize] = value;
        }
    }
}

fn uint8_tensor(data: Vec<u8>, width: u32, height: u32, channels: usize) -> Tensor {
    Tensor {
        data: TensorData::UInt8(data),
        shape: vec![height as usize, width as usize, channels],
        dtype: DType::UInt8,
        device: Device::Cpu,
    }
}

fn float32_tensor(data: Vec<f32>, width: u32, height: u32) -> Tensor {
    Tensor {
        data: TensorData::Float32(data),
        shape: vec![height as usize, width as usize],
        dtype: DType::Float32,
        device: Device::Cpu,
    }
}

fn int32_tensor(data: Vec<i32>, width: u32, height: u32) -> Tensor {
    Tensor {
        data: TensorData::Int32(data),
        shape: vec![height as usize, width as usize],
        dtype: DType::Int32,
        device: Device::Cpu,
    }
}

fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn color_from_seed(seed: u64) -> [u8; 3] {
    [
        64 + (seed as u8 % 160),
        64 + ((seed >> 8) as u8 % 160),
        64 + ((seed >> 16) as u8 % 160),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::action::Condition;
    use worldforge_core::types::Position;

    fn sample_state() -> (WorldState, uuid::Uuid, uuid::Uuid) {
        let mut state = WorldState::new("test", "mock");
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
                    x: -0.5,
                    y: 0.7,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: -0.7,
                    y: 0.5,
                    z: -0.2,
                },
                max: Position {
                    x: -0.3,
                    y: 0.9,
                    z: 0.2,
                },
            },
        );
        let mug_id = mug.id;

        state.scene.add_object(table);
        state.scene.add_object(mug);
        (state, table_id, mug_id)
    }

    #[tokio::test]
    async fn test_mock_predict_updates_geometry_and_relationships() {
        let provider = MockProvider::new();
        let (state, table_id, mug_id) = sample_state();
        let action = Action::Place {
            object: mug_id,
            target: Position {
                x: 0.0,
                y: 0.7,
                z: 0.0,
            },
        };
        let prediction = provider
            .predict(
                &state,
                &action,
                &PredictionConfig {
                    steps: 3,
                    ..PredictionConfig::default()
                },
            )
            .await
            .unwrap();

        let moved = prediction.output_state.scene.get_object(&mug_id).unwrap();
        assert_eq!(moved.pose.position.x, 0.0);
        assert_eq!(moved.bbox.center().x, 0.0);
        assert!(prediction
            .output_state
            .scene
            .relationships
            .iter()
            .any(|relationship| {
                matches!(
                    relationship,
                    SpatialRelationship::On { subject, surface }
                        if *subject == mug_id && *surface == table_id
                )
            }));
    }

    #[tokio::test]
    async fn test_mock_predict_returns_preview_video_depth_and_segmentation() {
        let provider = MockProvider::new();
        let (state, _, mug_id) = sample_state();
        let prediction = provider
            .predict(
                &state,
                &Action::Place {
                    object: mug_id,
                    target: Position {
                        x: 0.5,
                        y: 0.7,
                        z: 0.0,
                    },
                },
                &PredictionConfig {
                    steps: 4,
                    resolution: (640, 360),
                    fps: 12.0,
                    return_video: true,
                    return_depth: true,
                    return_segmentation: true,
                    ..PredictionConfig::default()
                },
            )
            .await
            .unwrap();

        let clip = prediction.video.unwrap();
        assert!(!clip.frames.is_empty());
        assert!(clip.frames[0].depth.is_some());
        assert!(clip.frames[0].segmentation.is_some());
    }

    #[tokio::test]
    async fn test_mock_generate_populates_frames() {
        let provider = MockProvider::new();
        let clip = provider
            .generate(
                &GenerationPrompt {
                    text: "A kitchen with a mug".to_string(),
                    reference_image: None,
                    negative_prompt: None,
                },
                &GenerationConfig {
                    resolution: (640, 360),
                    fps: 12.0,
                    duration_seconds: 5.0,
                    ..GenerationConfig::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(clip.resolution, (640, 360));
        assert!(!clip.frames.is_empty());
    }

    #[tokio::test]
    async fn test_mock_reason_uses_scene_state() {
        let provider = MockProvider::new();
        let (state, _, _) = sample_state();
        let output = provider
            .reason(
                &ReasoningInput {
                    video: None,
                    state: Some(state),
                },
                "how many objects are here?",
            )
            .await
            .unwrap();

        assert!(output.answer.contains("2 object"));
        assert!(output.evidence.iter().any(|entry| entry.contains("table")));
    }

    #[tokio::test]
    async fn test_mock_transfer_respects_output_config() {
        let provider = MockProvider::new();
        let clip = provider
            .transfer(
                &VideoClip {
                    frames: Vec::new(),
                    fps: 24.0,
                    resolution: (640, 360),
                    duration: 3.0,
                },
                &SpatialControls::default(),
                &TransferConfig {
                    resolution: (1280, 720),
                    fps: 12.0,
                    control_strength: 0.75,
                },
            )
            .await
            .unwrap();

        assert_eq!(clip.resolution, (1280, 720));
        assert_eq!(clip.fps, 12.0);
        assert!(!clip.frames.is_empty());
    }

    #[tokio::test]
    async fn test_mock_predict_supports_conditionals_and_touching() {
        let provider = MockProvider::new();
        let (state, table_id, mug_id) = sample_state();
        let prediction = provider
            .predict(
                &state,
                &Action::Conditional {
                    condition: Condition::ObjectExists { object: mug_id },
                    then: Box::new(Action::Place {
                        object: mug_id,
                        target: Position {
                            x: 0.0,
                            y: 0.7,
                            z: 0.0,
                        },
                    }),
                    otherwise: None,
                },
                &PredictionConfig::default(),
            )
            .await
            .unwrap();

        assert!(prediction
            .output_state
            .scene
            .relationships
            .iter()
            .any(|relationship| {
                matches!(
                    relationship,
                    SpatialRelationship::On { subject, surface }
                        if *subject == mug_id && *surface == table_id
                )
            }));
    }

    #[tokio::test]
    async fn test_mock_health() {
        let provider = MockProvider::new();
        let status = provider.health_check().await.unwrap();
        assert!(status.healthy);
    }

    #[test]
    fn test_mock_capabilities() {
        let provider = MockProvider::new();
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(caps.generate);
        assert!(caps.transfer);
        assert!(caps.supports_depth);
        assert!(caps.supports_segmentation);
    }

    #[test]
    fn test_score_prediction_detects_support_for_stability() {
        let (state, _, mug_id) = sample_state();
        let mut moved = state.clone();
        moved.scene.set_object_position(
            &mug_id,
            Position {
                x: 0.0,
                y: 2.0,
                z: 0.0,
            },
        );
        moved.scene.refresh_relationships();

        let scores = score_prediction(
            &state,
            &moved,
            &Action::Place {
                object: mug_id,
                target: Position {
                    x: 0.0,
                    y: 2.0,
                    z: 0.0,
                },
            },
        );
        assert!(scores.gravity_compliance < 0.95);
    }

    #[test]
    fn test_render_state_frame_tracks_segmentation() {
        let (state, _, _) = sample_state();
        let frame = render_state_frame(&state, (640, 360), true, true, None);
        assert!(matches!(frame.data.data, TensorData::UInt8(_)));
        assert!(matches!(
            frame.depth.as_ref().unwrap().data,
            TensorData::Float32(_)
        ));
        assert!(matches!(
            frame.segmentation.as_ref().unwrap().data,
            TensorData::Int32(_)
        ));
    }
}
