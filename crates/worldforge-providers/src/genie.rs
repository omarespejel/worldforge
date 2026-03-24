//! Google Genie provider adapter.
//!
//! Implements the `WorldModelProvider` trait for Genie-style interactive
//! world generation using a deterministic local surrogate. The adapter does
//! not call an external API, but it produces usable predictions and video
//! clips that are stable across runs and distinct from the mock provider.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use worldforge_core::action::{evaluate_condition, Action, ActionSpaceType, Weather};
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::prediction::{PhysicsScores, Prediction, PredictionConfig};
use worldforge_core::provider::{
    CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
    ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
    WorldModelProvider,
};
use worldforge_core::scene::SceneObject;
use worldforge_core::state::WorldState;
use worldforge_core::types::{
    BBox, CameraPose, DType, Device, Frame, Pose, Position, Rotation, SimTime, Tensor, TensorData,
    Vec3, Velocity, VideoClip,
};

const MAX_RESOLUTION: (u32, u32) = (256, 256);
const MAX_DURATION_SECONDS: f64 = 8.0;
const MIN_GENERATION_FPS: f32 = 6.0;
const MAX_GENERATION_FPS: f32 = 12.0;
const PREDICTION_BASE_LATENCY_MS: u64 = 38;
const GENERATION_BASE_LATENCY_MS: u64 = 54;
const HEALTH_BASE_LATENCY_MS: u64 = 12;

/// Google Genie model variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenieModel {
    /// Genie 3 interactive world generation.
    Genie3,
}

/// Google Genie provider adapter.
#[derive(Debug, Clone)]
pub struct GenieProvider {
    /// Model variant.
    pub model: GenieModel,
    /// API key for authentication when a real backend is attached.
    api_key: String,
    /// API endpoint URL.
    pub endpoint: String,
}

impl GenieProvider {
    /// Create a new Genie provider.
    pub fn new(model: GenieModel, api_key: impl Into<String>) -> Self {
        Self {
            model,
            api_key: api_key.into(),
            endpoint: "https://generativelanguage.googleapis.com".to_string(),
        }
    }

    /// Create a Genie provider with a custom endpoint.
    pub fn with_endpoint(
        model: GenieModel,
        api_key: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Self::new(model, api_key)
        }
    }

    fn model_name(&self) -> &'static str {
        match self.model {
            GenieModel::Genie3 => "genie-3-local-surrogate",
        }
    }

    fn is_endpoint_valid(&self) -> bool {
        self.endpoint.starts_with("http://") || self.endpoint.starts_with("https://")
    }

    fn selected_object_id(
        &self,
        state: &WorldState,
        target: Option<Position>,
    ) -> Option<uuid::Uuid> {
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

    fn apply_action(&self, state: &mut WorldState, action: &Action) -> (bool, f32) {
        match action {
            Action::Move { target, speed } => {
                let Some(object_id) = self.selected_object_id(state, Some(*target)) else {
                    return (false, 0.05);
                };
                if let Some(object) = state.scene.get_object_mut(&object_id) {
                    let blend = ((*speed * 0.12) + 0.25).clamp(0.15, 0.9);
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
                    (delta.magnitude() > f32::EPSILON, 0.22)
                } else {
                    (false, 0.05)
                }
            }
            Action::Grasp { object, grip_force } => {
                if let Some(item) = state.scene.get_object_mut(object) {
                    item.velocity = Velocity::default();
                    item.pose.position.y += (*grip_force).clamp(0.0, 20.0) * 0.001;
                    item.bbox.translate(Vec3 {
                        x: 0.0,
                        y: (*grip_force).clamp(0.0, 20.0) * 0.001,
                        z: 0.0,
                    });
                    (true, 0.24)
                } else {
                    (false, 0.12)
                }
            }
            Action::Release { object } => {
                if let Some(item) = state.scene.get_object_mut(object) {
                    item.velocity = Velocity::default();
                    (true, 0.08)
                } else {
                    (false, 0.08)
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
                    (push.magnitude() > f32::EPSILON, 0.26)
                } else {
                    (false, 0.14)
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
                    (true, 0.22)
                } else {
                    (false, 0.18)
                }
            }
            Action::Place { object, target } => {
                if let Some(item) = state.scene.get_object_mut(object) {
                    item.set_position(*target);
                    item.velocity = Velocity::default();
                    (true, 0.18)
                } else {
                    (false, 0.16)
                }
            }
            Action::CameraMove { .. } => (false, 0.05),
            Action::CameraLookAt { .. } => (false, 0.05),
            Action::Navigate { waypoints } => {
                let Some(final_waypoint) = waypoints.last().copied() else {
                    return (false, 0.03);
                };
                let Some(object_id) = self.selected_object_id(state, Some(final_waypoint)) else {
                    return (false, 0.03);
                };
                if let Some(item) = state.scene.get_object_mut(&object_id) {
                    item.set_position(final_waypoint);
                    item.velocity = Velocity::default();
                    (true, 0.2)
                } else {
                    (false, 0.03)
                }
            }
            Action::Teleport { destination } => {
                let Some(object_id) = self.selected_object_id(state, Some(destination.position))
                else {
                    return (false, 0.04);
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
                    (true, 0.22)
                } else {
                    (false, 0.04)
                }
            }
            Action::SetWeather { weather } => {
                state
                    .metadata
                    .tags
                    .retain(|tag| !tag.starts_with("weather:"));
                state
                    .metadata
                    .tags
                    .push(format!("weather:{weather:?}").to_lowercase());
                (true, 0.08)
            }
            Action::SetLighting { time_of_day } => {
                replace_tag(
                    &mut state.metadata.tags,
                    "lighting:",
                    format!("lighting:{time_of_day:.2}"),
                );
                (true, 0.08)
            }
            Action::SpawnObject { template, pose } => {
                let object = spawn_object(template, *pose);
                state.scene.add_object(object);
                (true, 0.18)
            }
            Action::RemoveObject { object } => (state.scene.remove_object(object).is_some(), 0.16),
            Action::Sequence(actions) => {
                let mut changed = false;
                let mut complexity = 0.0;
                for nested in actions {
                    let (nested_changed, nested_complexity) = self.apply_action(state, nested);
                    changed |= nested_changed;
                    complexity += nested_complexity;
                }
                (changed, complexity.max(0.1))
            }
            Action::Parallel(actions) => {
                let mut changed = false;
                let mut complexity = 0.0;
                for nested in actions {
                    let (nested_changed, nested_complexity) = self.apply_action(state, nested);
                    changed |= nested_changed;
                    complexity += nested_complexity;
                }
                (changed, complexity.max(0.1))
            }
            Action::Conditional {
                condition,
                then,
                otherwise,
            } => {
                if evaluate_condition(condition, state) {
                    self.apply_action(state, then)
                } else if let Some(otherwise) = otherwise {
                    self.apply_action(state, otherwise)
                } else {
                    (false, 0.08)
                }
            }
            Action::Raw { provider, data } => {
                if provider.eq_ignore_ascii_case("genie") {
                    self.apply_raw_action(state, data)
                } else {
                    (false, 0.02)
                }
            }
        }
    }

    fn apply_raw_action(&self, state: &mut WorldState, data: &serde_json::Value) -> (bool, f32) {
        let Some(kind) = data
            .get("type")
            .or_else(|| data.get("kind"))
            .and_then(|value| value.as_str())
        else {
            return (false, 0.03);
        };

        match kind {
            "move" => {
                let target = data
                    .get("target")
                    .and_then(parse_position_value)
                    .unwrap_or_default();
                let speed = data
                    .get("speed")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(1.0) as f32;
                self.apply_action(state, &Action::Move { target, speed })
            }
            "place" => {
                let object = parse_uuid_value(data.get("object").or_else(|| data.get("object_id")));
                let target = data
                    .get("target")
                    .or_else(|| data.get("position"))
                    .and_then(parse_position_value);
                match (object, target) {
                    (Some(object), Some(target)) => {
                        self.apply_action(state, &Action::Place { object, target })
                    }
                    _ => (false, 0.05),
                }
            }
            "spawn" => {
                let template = data
                    .get("template")
                    .or_else(|| data.get("name"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("genie-object")
                    .to_string();
                let pose = data
                    .get("pose")
                    .and_then(parse_pose_value)
                    .unwrap_or_default();
                self.apply_action(state, &Action::SpawnObject { template, pose })
            }
            "remove" => {
                if let Some(object) =
                    parse_uuid_value(data.get("object").or_else(|| data.get("object_id")))
                {
                    self.apply_action(state, &Action::RemoveObject { object })
                } else {
                    (false, 0.04)
                }
            }
            "set_weather" | "weather" => {
                if let Some(weather) = data
                    .get("weather")
                    .or_else(|| data.get("value"))
                    .and_then(parse_weather_value)
                {
                    self.apply_action(state, &Action::SetWeather { weather })
                } else {
                    (false, 0.04)
                }
            }
            "set_lighting" | "lighting" => {
                let time_of_day = data
                    .get("time_of_day")
                    .or_else(|| data.get("value"))
                    .and_then(|value| value.as_f64())
                    .unwrap_or(12.0) as f32;
                self.apply_action(state, &Action::SetLighting { time_of_day })
            }
            "push" => {
                let Some(object) =
                    parse_uuid_value(data.get("object").or_else(|| data.get("object_id")))
                else {
                    return (false, 0.04);
                };
                let direction = data
                    .get("direction")
                    .and_then(parse_vec3_value)
                    .unwrap_or(Vec3 {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                    });
                let force = data
                    .get("force")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(1.0) as f32;
                self.apply_action(
                    state,
                    &Action::Push {
                        object,
                        direction,
                        force,
                    },
                )
            }
            "rotate" => {
                let Some(object) =
                    parse_uuid_value(data.get("object").or_else(|| data.get("object_id")))
                else {
                    return (false, 0.04);
                };
                let axis = data.get("axis").and_then(parse_vec3_value).unwrap_or(Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                });
                let angle = data
                    .get("angle")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(15.0) as f32;
                self.apply_action(
                    state,
                    &Action::Rotate {
                        object,
                        axis,
                        angle,
                    },
                )
            }
            _ => (false, 0.03),
        }
    }

    fn update_time(state: &mut WorldState, steps: u32, fps: f32) {
        let fps = fps.max(1.0);
        let steps = steps.max(1);
        state.time.step = state.time.step.saturating_add(steps as u64);
        state.time.seconds += steps as f64 / fps as f64;
        state.time.dt = 1.0 / fps as f64;
    }

    fn compute_prediction_scores(
        &self,
        state: &WorldState,
        output_state: &WorldState,
        changed: bool,
        complexity: f32,
    ) -> PhysicsScores {
        let object_count = output_state.scene.objects.len() as f32;
        let relationship_count = output_state.scene.relationships.len() as f32;
        let scene_density = (object_count / 16.0).clamp(0.0, 1.0);
        let relationship_factor = (relationship_count / 20.0).clamp(0.0, 1.0);
        let mut base = 0.92 - scene_density * 0.14 - relationship_factor * 0.05 - complexity * 0.09
            + if changed { 0.04 } else { -0.06 };
        base += stable_fraction(&prediction_seed(state, output_state, complexity, changed)) * 0.04;
        base = base.clamp(0.35, 0.99);

        PhysicsScores {
            overall: base,
            object_permanence: (base + 0.03).clamp(0.0, 1.0),
            gravity_compliance: (base - 0.02).clamp(0.0, 1.0),
            collision_accuracy: (base - 0.01).clamp(0.0, 1.0),
            spatial_consistency: (base + 0.01).clamp(0.0, 1.0),
            temporal_consistency: (base - 0.03).clamp(0.0, 1.0),
        }
    }

    fn compute_confidence(
        &self,
        physics_scores: PhysicsScores,
        changed: bool,
        complexity: f32,
    ) -> f32 {
        let mut confidence = physics_scores.overall * 0.92
            + physics_scores.object_permanence * 0.04
            + physics_scores.spatial_consistency * 0.04
            - complexity * 0.02
            + if changed { 0.04 } else { -0.02 };
        confidence = confidence.clamp(0.25, 0.99);
        confidence
    }

    fn prediction_latency_ms(&self, config: &PredictionConfig, object_count: usize) -> u64 {
        let pixels = (config.resolution.0 as u64).saturating_mul(config.resolution.1 as u64);
        PREDICTION_BASE_LATENCY_MS
            + u64::from(config.steps.max(1)) * 4
            + (pixels / 8_000)
            + (object_count as u64 * 2)
    }

    fn render_prediction_clip(
        &self,
        input_state: &WorldState,
        output_state: &WorldState,
        config: &PredictionConfig,
    ) -> VideoClip {
        let resolution = clamp_resolution(config.resolution);
        let fps = config.fps.clamp(MIN_GENERATION_FPS, MAX_GENERATION_FPS);
        let frames = config.steps.clamp(1, 32) as usize;
        let frame_duration = 1.0 / fps as f64;
        let seed = prediction_seed(input_state, output_state, 0.0, true);

        let mut rendered = Vec::with_capacity(frames);
        for index in 0..frames {
            let alpha = if frames == 1 {
                1.0
            } else {
                index as f32 / (frames - 1) as f32
            };
            rendered.push(render_world_frame(
                output_state,
                resolution,
                seed ^ (index as u64).wrapping_mul(0x9e37_79b9),
                alpha,
                frame_duration * index as f64,
                config.return_depth,
                config.return_segmentation,
            ));
        }

        VideoClip {
            frames: rendered,
            fps,
            resolution,
            duration: frames as f64 / fps as f64,
        }
    }

    fn render_generation_clip(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> VideoClip {
        let resolution = clamp_resolution(config.resolution);
        let fps = config.fps.clamp(MIN_GENERATION_FPS, MAX_GENERATION_FPS);
        let duration = config.duration_seconds.clamp(0.5, MAX_DURATION_SECONDS);
        let frame_count = ((duration * fps as f64).round() as usize).clamp(1, 128);
        let seed = prompt_seed(prompt);
        let reference_seed = prompt
            .reference_image
            .as_ref()
            .map(tensor_fingerprint)
            .unwrap_or(0);

        let mut frames = Vec::with_capacity(frame_count);
        for index in 0..frame_count {
            let alpha = if frame_count == 1 {
                1.0
            } else {
                index as f32 / (frame_count - 1) as f32
            };
            frames.push(render_prompt_frame(
                prompt,
                resolution,
                seed ^ reference_seed ^ (index as u64).wrapping_mul(0x517c_c1b7),
                alpha,
                index as f64 / fps as f64,
            ));
        }

        VideoClip {
            frames,
            fps,
            resolution,
            duration,
        }
    }
}

#[async_trait]
impl WorldModelProvider for GenieProvider {
    fn name(&self) -> &str {
        "genie"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: false,
            transfer: false,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: MAX_DURATION_SECONDS as f32,
            max_resolution: MAX_RESOLUTION,
            fps_range: (MIN_GENERATION_FPS, MAX_GENERATION_FPS),
            supported_action_spaces: vec![ActionSpaceType::Discrete, ActionSpaceType::Language],
            supports_depth: false,
            supports_segmentation: false,
            supports_planning: false,
            latency_profile: LatencyProfile {
                p50_ms: 220,
                p95_ms: 500,
                p99_ms: 900,
                throughput_fps: 12.0,
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
        let (changed, complexity) = self.apply_action(&mut output_state, action);
        let effective_steps = config.steps.max(1);
        Self::update_time(&mut output_state, effective_steps, config.fps.max(1.0));

        let physics_scores =
            self.compute_prediction_scores(state, &output_state, changed, complexity);
        let confidence = self.compute_confidence(physics_scores, changed, complexity);
        let video = if config.return_video {
            Some(self.render_prediction_clip(state, &output_state, config))
        } else {
            None
        };
        let latency_ms = self.prediction_latency_ms(config, output_state.scene.objects.len());
        let cost = self.estimate_cost(&Operation::Predict {
            steps: effective_steps,
            resolution: config.resolution,
        });

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: self.name().to_string(),
            model: self.model_name().to_string(),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video,
            confidence,
            physics_scores,
            latency_ms,
            cost,
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn generate(
        &self,
        prompt: &GenerationPrompt,
        config: &GenerationConfig,
    ) -> Result<VideoClip> {
        let _estimated_latency = self.estimate_cost(&Operation::Generate {
            duration_seconds: config.duration_seconds,
            resolution: config.resolution,
        });
        Ok(self.render_generation_clip(prompt, config))
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: self.name().to_string(),
            capability: "reason (Genie surrogate does not expose a reasoning backend)".to_string(),
        })
    }

    async fn transfer(
        &self,
        _source: &VideoClip,
        _controls: &SpatialControls,
        _config: &TransferConfig,
    ) -> Result<VideoClip> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: self.name().to_string(),
            capability: "transfer (Genie surrogate does not expose spatial transfer)".to_string(),
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let healthy = !self.api_key.trim().is_empty() && self.is_endpoint_valid();
        Ok(HealthStatus {
            healthy,
            message: if healthy {
                format!(
                    "Genie surrogate ready: {} at {}",
                    self.model_name(),
                    self.endpoint
                )
            } else if self.api_key.trim().is_empty() {
                "missing Genie API key".to_string()
            } else {
                format!("invalid Genie endpoint: {}", self.endpoint)
            },
            latency_ms: HEALTH_BASE_LATENCY_MS,
        })
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        let model_multiplier = match self.model {
            GenieModel::Genie3 => 1.0,
        };

        match operation {
            Operation::Predict { steps, resolution } => {
                let pixels = (resolution.0 as f64) * (resolution.1 as f64);
                let step_factor = (*steps).max(1) as f64;
                let usd = (0.0012 + step_factor * 0.00035 + pixels / 1_000_000.0 * 0.0007)
                    * model_multiplier;
                CostEstimate {
                    usd,
                    credits: usd * 1_000.0 + step_factor * 0.25,
                    estimated_latency_ms: PREDICTION_BASE_LATENCY_MS
                        + (*steps).max(1) as u64 * 4
                        + (pixels as u64 / 8_000),
                }
            }
            Operation::Generate {
                duration_seconds,
                resolution,
            } => {
                let duration = duration_seconds.clamp(0.5, MAX_DURATION_SECONDS);
                let pixels = (resolution.0 as f64) * (resolution.1 as f64);
                let usd =
                    (0.0020 + duration * 0.0010 + pixels / 1_000_000.0 * 0.0011) * model_multiplier;
                CostEstimate {
                    usd,
                    credits: usd * 1_000.0 + duration * 6.0,
                    estimated_latency_ms: GENERATION_BASE_LATENCY_MS
                        + (duration * 8.0) as u64
                        + (pixels as u64 / 6_000),
                }
            }
            Operation::Reason => CostEstimate {
                usd: 0.0006 * model_multiplier,
                credits: 0.75 * model_multiplier,
                estimated_latency_ms: HEALTH_BASE_LATENCY_MS + 18,
            },
            Operation::Transfer { duration_seconds } => {
                let duration = duration_seconds.clamp(0.5, MAX_DURATION_SECONDS);
                let usd = (0.0015 + duration * 0.00085) * model_multiplier;
                CostEstimate {
                    usd,
                    credits: usd * 1_000.0 + duration * 4.0,
                    estimated_latency_ms: GENERATION_BASE_LATENCY_MS + (duration * 7.0) as u64,
                }
            }
        }
    }
}

fn clamp_resolution(resolution: (u32, u32)) -> (u32, u32) {
    (
        resolution.0.max(1).min(MAX_RESOLUTION.0),
        resolution.1.max(1).min(MAX_RESOLUTION.1),
    )
}

fn replace_tag(tags: &mut Vec<String>, prefix: &str, replacement: String) {
    tags.retain(|tag| !tag.starts_with(prefix));
    tags.push(replacement);
}

fn parse_uuid_value(value: Option<&serde_json::Value>) -> Option<uuid::Uuid> {
    value
        .and_then(|value| value.as_str())
        .and_then(|value| uuid::Uuid::parse_str(value).ok())
}

fn parse_position_value(value: &serde_json::Value) -> Option<Position> {
    serde_json::from_value::<Position>(value.clone()).ok()
}

fn parse_pose_value(value: &serde_json::Value) -> Option<Pose> {
    serde_json::from_value::<Pose>(value.clone()).ok()
}

fn parse_vec3_value(value: &serde_json::Value) -> Option<Vec3> {
    serde_json::from_value::<Vec3>(value.clone()).ok()
}

fn parse_weather_value(value: &serde_json::Value) -> Option<Weather> {
    if let Some(name) = value.as_str() {
        match name.to_ascii_lowercase().as_str() {
            "clear" => Some(Weather::Clear),
            "cloudy" => Some(Weather::Cloudy),
            "rain" => Some(Weather::Rain),
            "snow" => Some(Weather::Snow),
            "fog" => Some(Weather::Fog),
            "night" => Some(Weather::Night),
            _ => None,
        }
    } else {
        serde_json::from_value::<Weather>(value.clone()).ok()
    }
}

fn spawn_object(template: &str, pose: Pose) -> SceneObject {
    let seed = stable_hash(template.as_bytes());
    let half_extents = Vec3 {
        x: 0.06 + (stable_fraction(&seed) * 0.14),
        y: 0.06 + (stable_fraction(&seed.rotate_left(7)) * 0.16),
        z: 0.06 + (stable_fraction(&seed.rotate_left(13)) * 0.14),
    };
    let mut object = SceneObject::new(
        template.to_string(),
        pose,
        BBox::from_center_half_extents(pose.position, half_extents),
    );
    object.semantic_label = Some(template.to_string());
    object.physics.mass = Some(0.5 + stable_fraction(&seed.rotate_left(19)) * 4.0);
    object.physics.friction = Some(0.2 + stable_fraction(&seed.rotate_left(23)) * 0.6);
    object.physics.restitution = Some(0.05 + stable_fraction(&seed.rotate_left(29)) * 0.35);
    object.physics.is_graspable = half_extents.x.max(half_extents.y).max(half_extents.z) < 0.18;
    object
}

fn quaternion_from_axis_angle(axis: Vec3, angle_degrees: f32) -> Rotation {
    let axis = axis.normalized();
    let half_angle = angle_degrees.to_radians() * 0.5;
    let sin_half = half_angle.sin();
    Rotation {
        w: half_angle.cos(),
        x: axis.x * sin_half,
        y: axis.y * sin_half,
        z: axis.z * sin_half,
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

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn stable_fraction(hash: &u64) -> f32 {
    let mixed = hash.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17);
    (mixed as f64 / u64::MAX as f64) as f32
}

fn prediction_seed(
    state: &WorldState,
    output_state: &WorldState,
    complexity: f32,
    changed: bool,
) -> u64 {
    let mut seed = stable_hash(state.metadata.name.as_bytes());
    seed ^= stable_hash(output_state.metadata.name.as_bytes()).rotate_left(11);
    seed ^= stable_hash(state.metadata.created_by.as_bytes()).rotate_left(23);
    seed ^= stable_hash(output_state.metadata.created_by.as_bytes()).rotate_left(31);
    seed ^= (state.scene.objects.len() as u64).rotate_left(7);
    seed ^= (output_state.scene.objects.len() as u64).rotate_left(19);
    seed ^= (complexity.to_bits() as u64).rotate_left(13);
    if changed {
        seed ^= 0xa5a5_a5a5_a5a5_a5a5;
    }
    seed
}

fn prompt_seed(prompt: &GenerationPrompt) -> u64 {
    let mut seed = stable_hash(prompt.text.as_bytes());
    if let Some(negative) = &prompt.negative_prompt {
        seed ^= stable_hash(negative.as_bytes()).rotate_left(17);
    }
    seed
}

fn tensor_fingerprint(tensor: &Tensor) -> u64 {
    let mut seed =
        stable_hash(format!("{:?}{:?}{:?}", tensor.shape, tensor.dtype, tensor.device).as_bytes());
    match &tensor.data {
        TensorData::Float32(values) => {
            for value in values.iter().take(32) {
                seed ^= stable_hash(&value.to_bits().to_le_bytes()).rotate_left(7);
            }
        }
        TensorData::Float64(values) => {
            for value in values.iter().take(32) {
                seed ^= stable_hash(&value.to_bits().to_le_bytes()).rotate_left(7);
            }
        }
        TensorData::UInt8(values) => {
            for value in values.iter().take(64) {
                seed ^= u64::from(*value).rotate_left((*value % 23) as u32);
            }
        }
        TensorData::Int32(values) => {
            for value in values.iter().take(32) {
                seed ^= stable_hash(&value.to_le_bytes()).rotate_left(5);
            }
        }
        TensorData::Int64(values) => {
            for value in values.iter().take(32) {
                seed ^= stable_hash(&value.to_le_bytes()).rotate_left(5);
            }
        }
    }
    seed
}

#[derive(Clone)]
struct DrawableObject {
    center_x: f32,
    center_y: f32,
    width: f32,
    height: f32,
    color: [u8; 3],
}

fn render_world_frame(
    state: &WorldState,
    resolution: (u32, u32),
    seed: u64,
    alpha: f32,
    timestamp_seconds: f64,
    return_depth: bool,
    return_segmentation: bool,
) -> Frame {
    let objects = drawable_objects_from_state(state, alpha, seed);
    let mut pixels = vec![0u8; (resolution.0 as usize) * (resolution.1 as usize) * 3];
    let mut depth =
        return_depth.then(|| vec![1.0f32; (resolution.0 as usize) * (resolution.1 as usize)]);
    let mut segmentation =
        return_segmentation.then(|| vec![0u8; (resolution.0 as usize) * (resolution.1 as usize)]);

    paint_background(
        &mut pixels,
        resolution,
        seed,
        state.metadata.tags.as_slice(),
        alpha,
    );

    let bounds = scene_bounds_from_drawables(&objects);
    for (index, drawable) in objects.iter().enumerate() {
        let rect = project_drawable(drawable, bounds, resolution);
        let fade = (0.45 + alpha * 0.55).clamp(0.0, 1.0);
        draw_rect_rgb(&mut pixels, resolution, rect, drawable.color, fade);
        if let Some(depth) = depth.as_mut() {
            draw_rect_f32(
                depth,
                resolution,
                rect,
                1.0 - drawable.center_y.clamp(0.0, 1.0),
            );
        }
        if let Some(segmentation) = segmentation.as_mut() {
            draw_rect_u8(segmentation, resolution, rect, (index + 1) as u8);
        }
    }

    let image_tensor = Tensor {
        data: TensorData::UInt8(pixels),
        shape: vec![resolution.1 as usize, resolution.0 as usize, 3],
        dtype: DType::UInt8,
        device: Device::Cpu,
    };

    Frame {
        data: image_tensor,
        timestamp: SimTime {
            step: state.time.step,
            seconds: state.time.seconds + timestamp_seconds,
            dt: state.time.dt,
        },
        camera: Some(CameraPose {
            extrinsics: Pose {
                position: Position {
                    x: 0.0,
                    y: 2.5 + alpha * 0.35,
                    z: 4.0,
                },
                rotation: Rotation::default(),
            },
            fov: 55.0,
            near_clip: 0.1,
            far_clip: 100.0,
        }),
        depth: depth.map(|depth| Tensor {
            data: TensorData::Float32(depth),
            shape: vec![resolution.1 as usize, resolution.0 as usize],
            dtype: DType::Float32,
            device: Device::Cpu,
        }),
        segmentation: segmentation.map(|segmentation| Tensor {
            data: TensorData::UInt8(segmentation),
            shape: vec![resolution.1 as usize, resolution.0 as usize],
            dtype: DType::UInt8,
            device: Device::Cpu,
        }),
    }
}

fn render_prompt_frame(
    prompt: &GenerationPrompt,
    resolution: (u32, u32),
    seed: u64,
    alpha: f32,
    timestamp_seconds: f64,
) -> Frame {
    let synthetic_state = prompt_scene_state(prompt, seed, alpha);
    render_world_frame(
        &synthetic_state,
        resolution,
        seed ^ 0xfeed_face_cafe_babe,
        alpha,
        timestamp_seconds,
        false,
        false,
    )
}

fn prompt_scene_state(prompt: &GenerationPrompt, seed: u64, alpha: f32) -> WorldState {
    let mut state = WorldState::new(
        format!("genie:{}", prompt.text.chars().take(24).collect::<String>()),
        "genie",
    );
    let words: Vec<_> = prompt.text.split_whitespace().take(4).collect();
    let count = words.len().max(2);
    for index in 0..count {
        let word = words.get(index).copied().unwrap_or("scene");
        let jitter = stable_fraction(&seed.rotate_left((index as u32 * 11) % 63));
        let x = -1.0 + index as f32 * 1.0 + jitter * 0.4 + alpha * 0.2;
        let y = 0.8 + (jitter * 0.4);
        let z = -0.5 + jitter * 0.8;
        let pose = Pose {
            position: Position { x, y, z },
            rotation: Rotation::default(),
        };
        let half = Vec3 {
            x: 0.16 + jitter * 0.12,
            y: 0.16 + jitter * 0.12,
            z: 0.16 + jitter * 0.12,
        };
        let mut object = SceneObject::new(
            format!("{}-{}", word, index),
            pose,
            BBox::from_center_half_extents(pose.position, half),
        );
        object.semantic_label = Some(word.to_string());
        object.velocity = Velocity {
            x: 0.05 + jitter * 0.1,
            y: 0.0,
            z: 0.03 + jitter * 0.08,
        };
        state.scene.add_object(object);
    }
    state
}

fn drawable_objects_from_state(state: &WorldState, alpha: f32, seed: u64) -> Vec<DrawableObject> {
    let mut objects: Vec<&SceneObject> = state.scene.objects.values().collect();
    objects.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.id.as_bytes().cmp(b.id.as_bytes()))
    });

    objects
        .into_iter()
        .map(|object| DrawableObject {
            center_x: object.pose.position.x + alpha * 0.03,
            center_y: object.pose.position.y,
            width: object.bbox.size().x.abs().max(0.05),
            height: object.bbox.size().z.abs().max(0.05),
            color: color_from_seed(
                seed ^ stable_hash(object.name.as_bytes())
                    ^ stable_hash(object.semantic_label.as_deref().unwrap_or("").as_bytes()),
            ),
        })
        .collect()
}

fn scene_bounds_from_drawables(drawables: &[DrawableObject]) -> (f32, f32, f32, f32) {
    if drawables.is_empty() {
        return (-1.0, 1.0, -1.0, 1.0);
    }

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for drawable in drawables {
        min_x = min_x.min(drawable.center_x - drawable.width);
        max_x = max_x.max(drawable.center_x + drawable.width);
        min_y = min_y.min(drawable.center_y - drawable.height);
        max_y = max_y.max(drawable.center_y + drawable.height);
    }

    let pad_x = ((max_x - min_x) * 0.18).max(0.35);
    let pad_y = ((max_y - min_y) * 0.18).max(0.35);
    (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)
}

fn project_drawable(
    drawable: &DrawableObject,
    bounds: (f32, f32, f32, f32),
    resolution: (u32, u32),
) -> (i32, i32, i32, i32) {
    let (min_x, max_x, min_y, max_y) = bounds;
    let width = resolution.0 as f32;
    let height = resolution.1 as f32;
    let center_x = map_range(drawable.center_x, min_x, max_x, 0.0, width - 1.0);
    let center_y = map_range(drawable.center_y, min_y, max_y, 0.0, height - 1.0);
    let half_w = ((drawable.width / (max_x - min_x).max(0.01)) * width * 0.85).max(2.0);
    let half_h = ((drawable.height / (max_y - min_y).max(0.01)) * height * 0.85).max(2.0);
    (
        (center_x - half_w).round() as i32,
        (center_y - half_h).round() as i32,
        (center_x + half_w).round() as i32,
        (center_y + half_h).round() as i32,
    )
}

fn paint_background(
    pixels: &mut [u8],
    resolution: (u32, u32),
    seed: u64,
    tags: &[String],
    alpha: f32,
) {
    let width = resolution.0 as usize;
    let height = resolution.1 as usize;
    let sky = color_from_seed(seed ^ 0x1a2b_3c4d_5e6f_7788);
    let ground = color_from_seed(seed ^ 0x8877_6655_4433_2211);
    let weather_boost = tags.iter().fold(0.0f32, |acc, tag| {
        if tag.starts_with("weather:") {
            acc + 0.07
        } else if tag.starts_with("lighting:") {
            acc + 0.03
        } else {
            acc
        }
    });

    for y in 0..height {
        let t = if height <= 1 {
            0.0
        } else {
            y as f32 / (height - 1) as f32
        };
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let pulse = ((seed.rotate_left((x as u32 % 23) + (y as u32 % 17)) & 0xff) as f32
                / 255.0)
                * 0.08
                * alpha;
            pixels[idx] = blend_channel(sky[0], ground[0], t, pulse + weather_boost);
            pixels[idx + 1] = blend_channel(sky[1], ground[1], t, pulse);
            pixels[idx + 2] = blend_channel(sky[2], ground[2], t, pulse - weather_boost * 0.5);
        }
    }
}

fn color_from_seed(seed: u64) -> [u8; 3] {
    let r = (seed & 0xff) as u8;
    let g = ((seed >> 8) & 0xff) as u8;
    let b = ((seed >> 16) & 0xff) as u8;
    [
        64u8.saturating_add(r / 2),
        64u8.saturating_add(g / 2),
        64u8.saturating_add(b / 2),
    ]
}

fn blend_channel(base: u8, target: u8, t: f32, pulse: f32) -> u8 {
    let t = t.clamp(0.0, 1.0);
    let value = (base as f32 * (1.0 - t) + target as f32 * t) + pulse * 255.0;
    value.clamp(0.0, 255.0) as u8
}

fn draw_rect_rgb(
    pixels: &mut [u8],
    resolution: (u32, u32),
    rect: (i32, i32, i32, i32),
    color: [u8; 3],
    alpha: f32,
) {
    let (x0, y0, x1, y1) = rect;
    let width = resolution.0 as i32;
    let height = resolution.1 as i32;
    let alpha = alpha.clamp(0.0, 1.0);
    let start_x = x0.clamp(0, width.saturating_sub(1));
    let start_y = y0.clamp(0, height.saturating_sub(1));
    let end_x = x1.clamp(0, width.saturating_sub(1));
    let end_y = y1.clamp(0, height.saturating_sub(1));

    if start_x >= end_x || start_y >= end_y {
        return;
    }

    for y in start_y..=end_y {
        for x in start_x..=end_x {
            let idx = ((y as usize) * resolution.0 as usize + x as usize) * 3;
            pixels[idx] = blend_pixel(pixels[idx], color[0], alpha);
            pixels[idx + 1] = blend_pixel(pixels[idx + 1], color[1], alpha);
            pixels[idx + 2] = blend_pixel(pixels[idx + 2], color[2], alpha);
        }
    }
}

fn draw_rect_f32(
    depth: &mut [f32],
    resolution: (u32, u32),
    rect: (i32, i32, i32, i32),
    value: f32,
) {
    let (x0, y0, x1, y1) = rect;
    let width = resolution.0 as i32;
    let height = resolution.1 as i32;
    let start_x = x0.clamp(0, width.saturating_sub(1));
    let start_y = y0.clamp(0, height.saturating_sub(1));
    let end_x = x1.clamp(0, width.saturating_sub(1));
    let end_y = y1.clamp(0, height.saturating_sub(1));

    if start_x >= end_x || start_y >= end_y {
        return;
    }

    for y in start_y..=end_y {
        for x in start_x..=end_x {
            let idx = (y as usize) * resolution.0 as usize + x as usize;
            depth[idx] = value.clamp(0.0, 1.0);
        }
    }
}

fn draw_rect_u8(mask: &mut [u8], resolution: (u32, u32), rect: (i32, i32, i32, i32), value: u8) {
    let (x0, y0, x1, y1) = rect;
    let width = resolution.0 as i32;
    let height = resolution.1 as i32;
    let start_x = x0.clamp(0, width.saturating_sub(1));
    let start_y = y0.clamp(0, height.saturating_sub(1));
    let end_x = x1.clamp(0, width.saturating_sub(1));
    let end_y = y1.clamp(0, height.saturating_sub(1));

    if start_x >= end_x || start_y >= end_y {
        return;
    }

    for y in start_y..=end_y {
        for x in start_x..=end_x {
            let idx = (y as usize) * resolution.0 as usize + x as usize;
            mask[idx] = value;
        }
    }
}

fn blend_pixel(base: u8, target: u8, alpha: f32) -> u8 {
    ((base as f32 * (1.0 - alpha) + target as f32 * alpha).clamp(0.0, 255.0)) as u8
}

fn map_range(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    if (from_max - from_min).abs() < f32::EPSILON {
        return to_min;
    }
    let ratio = (value - from_min) / (from_max - from_min);
    to_min + ratio.clamp(0.0, 1.0) * (to_max - to_min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::action::Condition;

    fn sample_state() -> (WorldState, uuid::Uuid) {
        let mut state = WorldState::new("genie-test", "genie");
        let object = SceneObject::new(
            "crate",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.5,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: -0.1,
                    y: 0.4,
                    z: -0.1,
                },
                max: Position {
                    x: 0.1,
                    y: 0.6,
                    z: 0.1,
                },
            },
        );
        let object_id = object.id;
        state.scene.add_object(object);
        (state, object_id)
    }

    #[test]
    fn test_genie_capabilities_are_distinct() {
        let provider = GenieProvider::new(GenieModel::Genie3, "key");
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(caps.generate);
        assert!(!caps.reason);
        assert!(!caps.transfer);
        assert_eq!(caps.max_resolution, (256, 256));
        assert_eq!(caps.fps_range, (6.0, 12.0));
    }

    #[tokio::test]
    async fn test_genie_predict_applies_action() {
        let provider = GenieProvider::new(GenieModel::Genie3, "key");
        let (state, object_id) = sample_state();
        let action = Action::Place {
            object: object_id,
            target: Position {
                x: 1.25,
                y: 0.75,
                z: -0.5,
            },
        };
        let config = PredictionConfig {
            steps: 4,
            resolution: (320, 180),
            fps: 12.0,
            return_video: true,
            return_depth: true,
            return_segmentation: true,
            ..PredictionConfig::default()
        };

        let prediction = provider.predict(&state, &action, &config).await.unwrap();
        let updated = prediction
            .output_state
            .scene
            .get_object(&object_id)
            .unwrap();
        assert_eq!(prediction.provider, "genie");
        assert_eq!(prediction.model, "genie-3-local-surrogate");
        assert!(
            updated.pose.position.distance(Position {
                x: 1.25,
                y: 0.75,
                z: -0.5,
            }) < 0.0001
        );
        assert!(prediction.confidence > 0.0);
        assert!(prediction.physics_scores.overall > 0.0);
        assert!(
            prediction.latency_ms
                >= provider
                    .estimate_cost(&Operation::Predict {
                        steps: 4,
                        resolution: (320, 180),
                    })
                    .estimated_latency_ms
        );
        assert!(prediction.video.is_some());
        assert!(prediction.guardrail_results.is_empty());
        assert_eq!(prediction.output_state.time.step, 4);
    }

    #[tokio::test]
    async fn test_genie_predict_handles_sequence_and_weather() {
        let provider = GenieProvider::new(GenieModel::Genie3, "key");
        let (state, object_id) = sample_state();
        let action = Action::Sequence(vec![
            Action::Move {
                target: Position {
                    x: 0.8,
                    y: 0.5,
                    z: 0.2,
                },
                speed: 2.0,
            },
            Action::SetLighting { time_of_day: 18.5 },
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

        let prediction = provider
            .predict(&state, &action, &PredictionConfig::default())
            .await
            .unwrap();
        let updated = prediction
            .output_state
            .scene
            .get_object(&object_id)
            .unwrap();
        assert_ne!(updated.pose.position, Position::default());
        assert!(prediction
            .output_state
            .metadata
            .tags
            .iter()
            .any(|tag| tag.starts_with("lighting:")));
        assert!(prediction
            .output_state
            .metadata
            .tags
            .iter()
            .any(|tag| tag.starts_with("weather:")));
    }

    #[tokio::test]
    async fn test_genie_generate_builds_lower_resolution_clip() {
        let provider = GenieProvider::new(GenieModel::Genie3, "key");
        let prompt = GenerationPrompt {
            text: "A robot moves across a bright lab".to_string(),
            reference_image: None,
            negative_prompt: Some("blur".to_string()),
        };
        let clip = provider
            .generate(
                &prompt,
                &GenerationConfig {
                    resolution: (512, 384),
                    fps: 30.0,
                    duration_seconds: 10.0,
                    temperature: 0.8,
                    seed: Some(7),
                },
            )
            .await
            .unwrap();

        assert_eq!(clip.resolution, (256, 256));
        assert!((clip.fps - 12.0).abs() < f32::EPSILON);
        assert_eq!(clip.frames.len(), 96);
        assert!(clip.duration <= MAX_DURATION_SECONDS + 0.01);
        assert_eq!(clip.frames[0].camera.as_ref().unwrap().fov, 55.0);
    }

    #[tokio::test]
    async fn test_genie_health_and_cost_are_non_trivial() {
        let provider = GenieProvider::new(GenieModel::Genie3, "secret");
        let health = provider.health_check().await.unwrap();
        assert!(health.healthy);
        assert!(health.message.contains("Genie surrogate ready"));

        let cost = provider.estimate_cost(&Operation::Generate {
            duration_seconds: 2.5,
            resolution: (320, 180),
        });
        assert!(cost.usd > 0.0);
        assert!(cost.credits > 0.0);
        assert!(cost.estimated_latency_ms > 0);
    }
}
