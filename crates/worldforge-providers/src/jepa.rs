//! Meta JEPA provider adapter (local inference).
//!
//! Implements the `WorldModelProvider` trait for Meta's JEPA family:
//! - I-JEPA: Image JEPA
//! - V-JEPA: Video JEPA
//! - V-JEPA 2: Video + action-conditioned planning
//!
//! This provider now offers a deterministic local inference path driven by
//! model asset inspection plus action-conditioned scene dynamics. It is still
//! a surrogate for full neural execution, but it behaves like a usable local
//! provider rather than a placeholder.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use worldforge_core::action::{evaluate_condition, Action, ActionSpaceType, Condition, Weather};
use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::goal_image;
use worldforge_core::guardrail::{evaluate_guardrails, has_blocking_violation};
use worldforge_core::prediction::{
    PhysicsScores, Plan, PlanGoal, PlanRequest, Prediction, PredictionConfig,
};
use worldforge_core::provider::{
    CostEstimate, GenerationConfig, GenerationPrompt, HealthStatus, LatencyProfile, Operation,
    ProviderCapabilities, ReasoningInput, ReasoningOutput, SpatialControls, TransferConfig,
    WorldModelProvider,
};
use worldforge_core::scene::{SceneGraph, SceneObject, SpatialRelationship};
use worldforge_core::state::WorldState;
use worldforge_core::types::{BBox, Pose, Position, Rotation, Vec3, Velocity, VideoClip};

const DEFAULT_MODEL_NAME: &str = "v-jepa-2-surrogate";
const MANIFEST_FILES: &[&str] = &["worldforge-jepa.json", "jepa.json", "config.json"];
const WEIGHT_EXTENSIONS: &[&str] = &["safetensors", "pt", "pth", "bin", "onnx", "ckpt"];

/// Backend for running JEPA inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JepaBackend {
    /// Rust-native via burn framework.
    Burn,
    /// PyTorch via tch-rs bindings.
    PyTorch,
    /// ONNX via ort-rs runtime.
    Onnx,
    /// Direct weight loading from safetensors.
    Safetensors,
}

impl JepaBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Burn => "burn",
            Self::PyTorch => "pytorch",
            Self::Onnx => "onnx",
            Self::Safetensors => "safetensors",
        }
    }

    fn base_latency_ms(self) -> u64 {
        match self {
            Self::Burn => 55,
            Self::PyTorch => 85,
            Self::Onnx => 70,
            Self::Safetensors => 45,
        }
    }
}

impl FromStr for JepaBackend {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "burn" => Ok(Self::Burn),
            "pytorch" | "torch" | "tch" => Ok(Self::PyTorch),
            "onnx" => Ok(Self::Onnx),
            "safetensors" | "st" => Ok(Self::Safetensors),
            other => Err(format!("unknown JEPA backend: {other}")),
        }
    }
}

/// Lightweight manifest describing how a local JEPA asset bundle should be
/// interpreted by the surrogate inference path.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JepaModelManifest {
    /// Optional model identifier to surface in predictions and health checks.
    pub model_name: Option<String>,
    /// Representation width used by the local model.
    pub representation_dim: Option<u32>,
    /// Multiplier applied to action-conditioned motion.
    pub action_gain: Option<f32>,
    /// Smoothing factor used during latent rollouts.
    pub temporal_smoothness: Option<f32>,
    /// Relative strength of gravity-like updates.
    pub gravity_bias: Option<f32>,
    /// Relative strength of collision handling.
    pub collision_bias: Option<f32>,
    /// Confidence adjustment applied to predictions.
    pub confidence_bias: Option<f32>,
}

impl JepaModelManifest {
    fn effective_model_name(&self) -> &str {
        self.model_name.as_deref().unwrap_or(DEFAULT_MODEL_NAME)
    }

    fn representation_dim(&self) -> u32 {
        self.representation_dim.unwrap_or(1024).clamp(128, 16_384)
    }

    fn action_gain(&self) -> f32 {
        self.action_gain.unwrap_or(1.0).clamp(0.1, 4.0)
    }

    fn temporal_smoothness(&self) -> f32 {
        self.temporal_smoothness.unwrap_or(0.82).clamp(0.1, 0.99)
    }

    fn gravity_bias(&self) -> f32 {
        self.gravity_bias.unwrap_or(0.9).clamp(0.0, 1.5)
    }

    fn collision_bias(&self) -> f32 {
        self.collision_bias.unwrap_or(0.85).clamp(0.0, 1.5)
    }

    fn confidence_bias(&self) -> f32 {
        self.confidence_bias.unwrap_or(0.0).clamp(-0.25, 0.25)
    }
}

#[derive(Debug, Clone)]
struct JepaAssets {
    manifest: JepaModelManifest,
    weight_files: Vec<PathBuf>,
    total_bytes: u64,
    fingerprint: u64,
}

/// Meta JEPA provider for local inference.
///
/// Loads V-JEPA / V-JEPA 2 assets from disk and performs a deterministic local
/// rollout based on the configured backend plus the inspected asset metadata.
/// This keeps the provider useful for planning and verification workflows even
/// before a full neural backend is wired in.
#[derive(Debug, Clone)]
pub struct JepaProvider {
    /// Path to model weights or a directory containing them.
    pub model_path: PathBuf,
    /// Inference backend.
    pub backend: JepaBackend,
}

impl JepaProvider {
    /// Create a new JEPA provider with the given model path and backend.
    pub fn new(model_path: impl Into<PathBuf>, backend: JepaBackend) -> Self {
        Self {
            model_path: model_path.into(),
            backend,
        }
    }

    /// Check if the configured model path exists on disk.
    pub fn weights_exist(&self) -> bool {
        self.model_path.exists()
    }

    fn inspect_assets(&self) -> Result<JepaAssets> {
        if !self.model_path.exists() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "jepa".to_string(),
                reason: format!("model path not found: {}", self.model_path.display()),
            });
        }

        let manifest_root = if self.model_path.is_dir() {
            self.model_path.as_path()
        } else {
            self.model_path.parent().unwrap_or_else(|| Path::new("."))
        };
        let manifest = load_manifest(manifest_root)?;
        let mut weight_files = collect_weight_files(&self.model_path)?;
        weight_files.sort();

        if weight_files.is_empty() {
            return Err(WorldForgeError::ProviderUnavailable {
                provider: "jepa".to_string(),
                reason: format!(
                    "no JEPA weight files found under {}",
                    self.model_path.display()
                ),
            });
        }

        let (total_bytes, fingerprint) = fingerprint_assets(&weight_files)?;
        Ok(JepaAssets {
            manifest,
            weight_files,
            total_bytes,
            fingerprint,
        })
    }

    fn estimate_local_latency_ms(&self, steps: u32, assets: &JepaAssets) -> u64 {
        let dim_penalty = u64::from(assets.manifest.representation_dim()) / 96;
        let size_penalty = (assets.total_bytes / (8 * 1024 * 1024)).min(64);
        self.backend.base_latency_ms() + u64::from(steps.max(1)) * 18 + dim_penalty + size_penalty
    }
}

#[async_trait]
impl WorldModelProvider for JepaProvider {
    fn name(&self) -> &str {
        "jepa"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: false,
            reason: false,
            transfer: false,
            embed: false,
            action_conditioned: true,
            multi_view: false,
            max_video_length_seconds: 5.0,
            max_resolution: (224, 224),
            fps_range: (8.0, 16.0),
            supported_action_spaces: vec![ActionSpaceType::Continuous],
            supports_depth: false,
            supports_segmentation: false,
            supports_planning: true,
            latency_profile: LatencyProfile {
                p50_ms: 90,
                p95_ms: 180,
                p99_ms: 320,
                throughput_fps: 30.0,
            },
        }
    }

    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        let assets = self.inspect_assets()?;
        let start = Instant::now();
        let output_state = simulate_prediction(state, action, config, &assets);
        let physics_scores = score_prediction(state, &output_state, action, &assets);
        let confidence = estimate_confidence(action, &physics_scores, &assets, self.backend);
        let latency_ms = self.estimate_local_latency_ms(config.steps, &assets)
            + start.elapsed().as_millis() as u64;

        Ok(Prediction {
            id: uuid::Uuid::new_v4(),
            provider: "jepa".to_string(),
            model: format!(
                "{}-{}-{:016x}",
                assets.manifest.effective_model_name(),
                self.backend.as_str(),
                assets.fingerprint
            ),
            input_state: state.clone(),
            action: action.clone(),
            output_state,
            video: None,
            confidence,
            physics_scores,
            latency_ms,
            cost: self.estimate_cost(&Operation::Predict {
                steps: config.steps,
                resolution: config.resolution,
            }),
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn generate(
        &self,
        _prompt: &GenerationPrompt,
        _config: &GenerationConfig,
    ) -> Result<VideoClip> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "jepa".to_string(),
            capability: "generate (JEPA models operate in representation space, not pixel space)"
                .to_string(),
        })
    }

    async fn reason(&self, _input: &ReasoningInput, _query: &str) -> Result<ReasoningOutput> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "jepa".to_string(),
            capability: "reason (use Cosmos Reason as fallback)".to_string(),
        })
    }

    async fn transfer(
        &self,
        _source: &VideoClip,
        _controls: &SpatialControls,
        _config: &TransferConfig,
    ) -> Result<VideoClip> {
        Err(WorldForgeError::UnsupportedCapability {
            provider: "jepa".to_string(),
            capability: "transfer (JEPA does not support spatial control transfer)".to_string(),
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        match self.inspect_assets() {
            Ok(assets) => Ok(HealthStatus {
                healthy: true,
                message: format!(
                    "JEPA {} ready via {} with {} weight file(s) totaling {} bytes",
                    assets.manifest.effective_model_name(),
                    self.backend.as_str(),
                    assets.weight_files.len(),
                    assets.total_bytes
                ),
                latency_ms: self.estimate_local_latency_ms(1, &assets) / 3,
            }),
            Err(err) => Ok(HealthStatus {
                healthy: false,
                message: err.to_string(),
                latency_ms: 0,
            }),
        }
    }

    async fn plan(&self, request: &PlanRequest) -> Result<Plan> {
        let assets = self.inspect_assets()?;
        let start = Instant::now();
        let native_goal =
            derive_native_goal(&request.goal, &request.current_state).ok_or_else(|| {
                WorldForgeError::NoFeasiblePlan {
                    goal: format!("{:?}", request.goal),
                    reason: "JEPA native planner could not interpret the requested goal"
                        .to_string(),
                }
            })?;

        let mut state = request.current_state.clone();
        let config = PredictionConfig::default();
        let mut actions = Vec::new();
        let mut predicted_states = Vec::new();
        let mut guardrail_compliance = Vec::new();

        while actions.len() < request.max_steps as usize {
            if native_goal_satisfied(&native_goal, &state) {
                break;
            }

            let Some(action) = next_native_action(&native_goal, &state) else {
                break;
            };

            let next_state = simulate_prediction(&state, &action, &config, &assets);
            let guardrail_results = if request.guardrails.is_empty() {
                Vec::new()
            } else {
                let results = evaluate_guardrails(&request.guardrails, &next_state);
                if has_blocking_violation(&results) {
                    return Err(WorldForgeError::NoFeasiblePlan {
                        goal: format!("{:?}", request.goal),
                        reason: "JEPA native planner generated a guardrail-blocked step"
                            .to_string(),
                    });
                }
                results
            };

            actions.push(action);
            state = next_state;
            predicted_states.push(state.clone());
            guardrail_compliance.push(guardrail_results);
        }

        if !native_goal_satisfied(&native_goal, &state) {
            return Err(WorldForgeError::NoFeasiblePlan {
                goal: format!("{:?}", request.goal),
                reason: "JEPA native planner exhausted the step budget before satisfying the goal"
                    .to_string(),
            });
        }

        let step_cost = self.estimate_cost(&Operation::Predict {
            steps: 1,
            resolution: config.resolution,
        });
        let total_cost = step_cost.usd as f32 * actions.len() as f32;

        let iterations_used = u32::try_from(predicted_states.len()).unwrap_or(u32::MAX);

        Ok(Plan {
            actions,
            predicted_states,
            predicted_videos: None,
            total_cost,
            success_probability: native_goal_score(&native_goal, &state),
            guardrail_compliance,
            planning_time_ms: start.elapsed().as_millis() as u64,
            iterations_used,
            verification_proof: None,
        })
    }

    fn estimate_cost(&self, operation: &Operation) -> CostEstimate {
        match operation {
            Operation::Predict { steps, .. } => CostEstimate {
                usd: 0.0,
                credits: 0.0,
                estimated_latency_ms: self.backend.base_latency_ms() + u64::from(*steps) * 18,
            },
            _ => CostEstimate::default(),
        }
    }
}

fn load_manifest(root: &Path) -> Result<JepaModelManifest> {
    for candidate in MANIFEST_FILES {
        let manifest_path = root.join(candidate);
        if manifest_path.exists() {
            let raw = fs::read_to_string(&manifest_path).map_err(|err| {
                WorldForgeError::InternalError(format!(
                    "failed to read JEPA manifest {}: {err}",
                    manifest_path.display()
                ))
            })?;
            return serde_json::from_str(&raw).map_err(|err| {
                WorldForgeError::InvalidState(format!(
                    "invalid JEPA manifest {}: {err}",
                    manifest_path.display()
                ))
            });
        }
    }

    Ok(JepaModelManifest::default())
}

fn collect_weight_files(root: &Path) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(if is_weight_file(root) {
            vec![root.to_path_buf()]
        } else {
            Vec::new()
        });
    }

    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();

    while let Some(dir) = pending.pop() {
        let entries = fs::read_dir(&dir).map_err(|err| {
            WorldForgeError::InternalError(format!(
                "failed to read JEPA model directory {}: {err}",
                dir.display()
            ))
        })?;

        for entry in entries {
            let entry = entry.map_err(|err| {
                WorldForgeError::InternalError(format!(
                    "failed to read JEPA directory entry in {}: {err}",
                    dir.display()
                ))
            })?;
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if is_weight_file(&path) {
                files.push(path);
            }
        }
    }

    Ok(files)
}

fn is_weight_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            WEIGHT_EXTENSIONS
                .iter()
                .any(|candidate| ext.eq_ignore_ascii_case(candidate))
        })
        .unwrap_or(false)
}

fn fingerprint_assets(weight_files: &[PathBuf]) -> Result<(u64, u64)> {
    let mut total_bytes = 0u64;
    let mut fingerprint = 0xcbf2_9ce4_8422_2325u64;

    for path in weight_files {
        let metadata = fs::metadata(path).map_err(|err| {
            WorldForgeError::InternalError(format!(
                "failed to stat JEPA asset {}: {err}",
                path.display()
            ))
        })?;
        let size = metadata.len();
        total_bytes += size;
        fingerprint = fnv1a_bytes(fingerprint, path.to_string_lossy().as_bytes());
        fingerprint = fnv1a_bytes(fingerprint, &size.to_le_bytes());

        let mut sample = Vec::new();
        let mut file = fs::File::open(path).map_err(|err| {
            WorldForgeError::InternalError(format!(
                "failed to open JEPA asset {}: {err}",
                path.display()
            ))
        })?;
        file.by_ref()
            .take(4096)
            .read_to_end(&mut sample)
            .map_err(|err| {
                WorldForgeError::InternalError(format!(
                    "failed to read JEPA asset sample {}: {err}",
                    path.display()
                ))
            })?;
        fingerprint = fnv1a_bytes(fingerprint, &sample);
    }

    Ok((total_bytes, fingerprint))
}

fn fnv1a_bytes(mut state: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x1000_0000_01b3);
    }
    state
}

fn simulate_prediction(
    state: &WorldState,
    action: &Action,
    config: &PredictionConfig,
    assets: &JepaAssets,
) -> WorldState {
    let mut output_state = state.clone();
    let fps = config.fps.max(1.0);
    let horizon = config.steps.max(1) as f32 / fps;
    apply_action_conditioned_update(&mut output_state, action, horizon, assets);
    apply_latent_relaxation(&mut output_state, horizon, assets);
    refresh_touching_relationships(&mut output_state.scene);
    output_state.time.step = output_state
        .time
        .step
        .saturating_add(u64::from(config.steps.max(1)));
    output_state.time.seconds += f64::from(config.steps.max(1)) / f64::from(fps);
    output_state.time.dt = 1.0 / f64::from(fps);
    output_state
}

#[derive(Debug, Clone)]
enum NativePlanGoal {
    AlreadySatisfied,
    ObjectAt {
        object_id: uuid::Uuid,
        target: Position,
        tolerance: f32,
    },
    ObjectExists {
        object_name: String,
        pose: Pose,
    },
    ObjectMissing {
        object_id: uuid::Uuid,
    },
    ObjectsTouching {
        mover: uuid::Uuid,
        anchor: uuid::Uuid,
    },
    Weather {
        weather: Weather,
    },
    Lighting {
        time_of_day: f32,
    },
}

fn derive_native_goal(goal: &PlanGoal, state: &WorldState) -> Option<NativePlanGoal> {
    match goal {
        PlanGoal::Condition(condition) => native_goal_from_condition(condition, state),
        PlanGoal::TargetState(target) => native_goal_from_target_state(target, state),
        PlanGoal::Description(description) => native_goal_from_description(description, state),
        PlanGoal::GoalImage(image) => {
            let target = goal_image::goal_image_target(image, state)?;
            let tolerance = goal_image_tolerance(target.confidence);
            if let Some(object_id) = primary_dynamic_object_id(&state.scene) {
                if state.scene.get_object(&object_id).is_some_and(|object| {
                    object.pose.position.distance(target.position) <= tolerance
                }) {
                    return Some(NativePlanGoal::AlreadySatisfied);
                }
                Some(NativePlanGoal::ObjectAt {
                    object_id,
                    target: target.position,
                    tolerance,
                })
            } else {
                Some(NativePlanGoal::ObjectExists {
                    object_name: "goal-image-object".to_string(),
                    pose: Pose {
                        position: target.position,
                        ..Pose::default()
                    },
                })
            }
        }
    }
}

fn goal_image_tolerance(confidence: f32) -> f32 {
    (0.12 + (1.0 - confidence).clamp(0.0, 1.0) * 0.2).clamp(0.05, 0.5)
}

fn native_goal_from_condition(condition: &Condition, state: &WorldState) -> Option<NativePlanGoal> {
    match condition {
        Condition::ObjectAt {
            object,
            position,
            tolerance,
        } => Some(NativePlanGoal::ObjectAt {
            object_id: *object,
            target: *position,
            tolerance: *tolerance,
        }),
        Condition::ObjectsTouching { a, b } => Some(NativePlanGoal::ObjectsTouching {
            mover: *a,
            anchor: *b,
        }),
        Condition::ObjectExists { object } => state
            .scene
            .get_object(object)
            .map(|_| NativePlanGoal::AlreadySatisfied),
        Condition::And(conditions) | Condition::Or(conditions) => conditions
            .iter()
            .find_map(|condition| native_goal_from_condition(condition, state)),
        Condition::Not(inner) => match inner.as_ref() {
            Condition::ObjectExists { object } => {
                if state.scene.get_object(object).is_some() {
                    Some(NativePlanGoal::ObjectMissing { object_id: *object })
                } else {
                    Some(NativePlanGoal::AlreadySatisfied)
                }
            }
            _ => None,
        },
    }
}

fn native_goal_from_target_state(
    target: &WorldState,
    state: &WorldState,
) -> Option<NativePlanGoal> {
    for (object_id, target_object) in &target.scene.objects {
        match state.scene.get_object(object_id) {
            Some(current) => {
                if position_distance(current.pose.position, target_object.pose.position) > 0.15 {
                    return Some(NativePlanGoal::ObjectAt {
                        object_id: *object_id,
                        target: target_object.pose.position,
                        tolerance: 0.15,
                    });
                }
            }
            None => {
                return Some(NativePlanGoal::ObjectExists {
                    object_name: target_object.name.clone(),
                    pose: target_object.pose,
                });
            }
        }
    }

    for object_id in state.scene.objects.keys() {
        if target.scene.get_object(object_id).is_none() {
            return Some(NativePlanGoal::ObjectMissing {
                object_id: *object_id,
            });
        }
    }

    Some(NativePlanGoal::AlreadySatisfied)
}

fn native_goal_from_description(description: &str, state: &WorldState) -> Option<NativePlanGoal> {
    let normalized = description.to_lowercase();
    let mentioned = mentioned_scene_objects(state, &normalized);

    if contains_any(&normalized, &["remove", "delete", "discard"]) {
        if let Some(object) = mentioned.first() {
            return Some(NativePlanGoal::ObjectMissing {
                object_id: object.id,
            });
        }
    }

    if normalized.contains("touch") && mentioned.len() >= 2 {
        return Some(NativePlanGoal::ObjectsTouching {
            mover: mentioned[0].id,
            anchor: mentioned[1].id,
        });
    }

    if let Some((weather, _)) = parse_weather_hint(&normalized) {
        return Some(NativePlanGoal::Weather { weather });
    }

    if let Some(time_of_day) = parse_lighting_hint(&normalized) {
        return Some(NativePlanGoal::Lighting { time_of_day });
    }

    if contains_any(&normalized, &["spawn", "create", "add"]) {
        return infer_object_name_from_verb(description, &["spawn", "create", "add"]).map(
            |object_name| NativePlanGoal::ObjectExists {
                pose: infer_spawn_pose(description, state),
                object_name,
            },
        );
    }

    parse_position_hint(description).and_then(|target| {
        mentioned
            .first()
            .map(|object| object.id)
            .or_else(|| primary_dynamic_object_id(&state.scene))
            .map(|object_id| NativePlanGoal::ObjectAt {
                object_id,
                target,
                tolerance: 0.15,
            })
    })
}

fn native_goal_satisfied(goal: &NativePlanGoal, state: &WorldState) -> bool {
    match goal {
        NativePlanGoal::AlreadySatisfied => true,
        NativePlanGoal::ObjectAt {
            object_id,
            target,
            tolerance,
        } => state
            .scene
            .get_object(object_id)
            .map(|object| position_distance(object.pose.position, *target) <= *tolerance)
            .unwrap_or(false),
        NativePlanGoal::ObjectExists { object_name, .. } => {
            !mentioned_scene_objects(state, &object_name.to_lowercase()).is_empty()
        }
        NativePlanGoal::ObjectMissing { object_id } => state.scene.get_object(object_id).is_none(),
        NativePlanGoal::ObjectsTouching { mover, anchor } => {
            touching_goal_satisfied(state, *mover, *anchor)
        }
        NativePlanGoal::Weather { weather } => {
            let expected = format!("weather:{weather:?}").to_lowercase();
            state
                .metadata
                .tags
                .iter()
                .any(|tag| tag.to_lowercase() == expected)
        }
        NativePlanGoal::Lighting { time_of_day } => state
            .metadata
            .tags
            .iter()
            .find_map(|tag| {
                tag.to_lowercase()
                    .strip_prefix("lighting:")
                    .and_then(|value| value.parse::<f32>().ok())
            })
            .map(|observed| (observed - *time_of_day).abs() <= 0.5)
            .unwrap_or(false),
    }
}

fn native_goal_score(goal: &NativePlanGoal, state: &WorldState) -> f32 {
    match goal {
        NativePlanGoal::AlreadySatisfied => 1.0,
        NativePlanGoal::ObjectAt {
            object_id,
            target,
            tolerance,
        } => state
            .scene
            .get_object(object_id)
            .map(|object| distance_score(object.pose.position, *target, *tolerance))
            .unwrap_or(0.0),
        NativePlanGoal::ObjectExists { object_name, .. } => {
            if mentioned_scene_objects(state, &object_name.to_lowercase()).is_empty() {
                0.0
            } else {
                1.0
            }
        }
        NativePlanGoal::ObjectMissing { object_id } => {
            if state.scene.get_object(object_id).is_none() {
                1.0
            } else {
                0.0
            }
        }
        NativePlanGoal::ObjectsTouching { mover, anchor } => touching_score(state, *mover, *anchor),
        NativePlanGoal::Weather { .. } | NativePlanGoal::Lighting { .. } => {
            if native_goal_satisfied(goal, state) {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn next_native_action(goal: &NativePlanGoal, state: &WorldState) -> Option<Action> {
    match goal {
        NativePlanGoal::AlreadySatisfied => None,
        NativePlanGoal::ObjectAt {
            object_id, target, ..
        } => state.scene.get_object(object_id).map(|_| Action::Place {
            object: *object_id,
            target: *target,
        }),
        NativePlanGoal::ObjectExists { object_name, pose } => {
            mentioned_scene_objects(state, &object_name.to_lowercase())
                .is_empty()
                .then(|| Action::SpawnObject {
                    template: object_name.clone(),
                    pose: *pose,
                })
        }
        NativePlanGoal::ObjectMissing { object_id } => state
            .scene
            .get_object(object_id)
            .map(|_| Action::RemoveObject { object: *object_id }),
        NativePlanGoal::ObjectsTouching { mover, anchor } => {
            state.scene.get_object(anchor).map(|target| Action::Place {
                object: *mover,
                target: target.pose.position,
            })
        }
        NativePlanGoal::Weather { weather } => (!native_goal_satisfied(goal, state))
            .then_some(Action::SetWeather { weather: *weather }),
        NativePlanGoal::Lighting { time_of_day } => (!native_goal_satisfied(goal, state))
            .then_some(Action::SetLighting {
                time_of_day: *time_of_day,
            }),
    }
}

fn parse_weather_hint(input: &str) -> Option<(Weather, &'static str)> {
    [
        (Weather::Rain, "rain"),
        (Weather::Snow, "snow"),
        (Weather::Fog, "fog"),
        (Weather::Cloudy, "cloudy"),
        (Weather::Night, "night"),
        (Weather::Clear, "clear"),
    ]
    .into_iter()
    .find(|(_, needle)| input.contains(needle))
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
            if let (Ok(x), Ok(y), Ok(z)) = (
                window[1].parse::<f32>(),
                window[2].parse::<f32>(),
                window[3].parse::<f32>(),
            ) {
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
    for verb in verbs {
        if let Some(index) = normalized.find(verb) {
            let remainder = input[index + verb.len()..].trim();
            let token = remainder
                .split_whitespace()
                .next()
                .map(|value| value.trim_matches(|ch: char| !ch.is_alphanumeric()));
            if let Some(token) = token {
                if !token.is_empty() {
                    return Some(token.to_string());
                }
            }
        }
    }
    None
}

fn mentioned_scene_objects<'a>(state: &'a WorldState, description: &str) -> Vec<&'a SceneObject> {
    let mut objects: Vec<_> = state
        .scene
        .objects
        .values()
        .filter(|object| {
            let name = object.name.to_lowercase();
            let label_match = object
                .semantic_label
                .as_ref()
                .map(|label| description.contains(&label.to_lowercase()))
                .unwrap_or(false);
            description.contains(&name) || label_match
        })
        .collect();
    objects.sort_by(|left, right| {
        right
            .name
            .len()
            .cmp(&left.name.len())
            .then_with(|| left.name.cmp(&right.name))
    });
    objects
}

fn infer_spawn_pose(description: &str, state: &WorldState) -> Pose {
    let normalized = description.to_lowercase();
    if contains_any(&normalized, &["next to", "beside", "near"]) {
        if let Some(anchor) = mentioned_scene_objects(state, &normalized).first() {
            let dx = (anchor.bbox.max.x - anchor.bbox.min.x).abs().max(0.15) + 0.15;
            return Pose {
                position: Position {
                    x: anchor.pose.position.x + dx,
                    y: anchor.pose.position.y,
                    z: anchor.pose.position.z,
                },
                ..Pose::default()
            };
        }
    }

    Pose {
        position: default_spawn_position_for_state(state),
        ..Pose::default()
    }
}

fn default_spawn_position_for_state(state: &WorldState) -> Position {
    let baseline = primary_dynamic_object_id(&state.scene)
        .and_then(|object_id| {
            state
                .scene
                .get_object(&object_id)
                .map(|object| object.pose.position)
        })
        .unwrap_or_default();
    Position {
        x: baseline.x + 0.25,
        y: baseline.y.max(0.0) + 0.5,
        z: baseline.z,
    }
}

fn touching_goal_satisfied(state: &WorldState, mover: uuid::Uuid, anchor: uuid::Uuid) -> bool {
    if state.scene.relationships.iter().any(|relationship| {
        matches!(relationship, SpatialRelationship::Touching { a, b } if (*a == mover && *b == anchor) || (*a == anchor && *b == mover))
    }) {
        return true;
    }

    let Some(left) = state.scene.get_object(&mover) else {
        return false;
    };
    let Some(right) = state.scene.get_object(&anchor) else {
        return false;
    };
    let radius = approximate_radius(left) + approximate_radius(right);
    position_distance(left.pose.position, right.pose.position) <= radius.max(0.05)
}

fn touching_score(state: &WorldState, mover: uuid::Uuid, anchor: uuid::Uuid) -> f32 {
    if touching_goal_satisfied(state, mover, anchor) {
        return 1.0;
    }

    let Some(left) = state.scene.get_object(&mover) else {
        return 0.0;
    };
    let Some(right) = state.scene.get_object(&anchor) else {
        return 0.0;
    };
    let radius = approximate_radius(left) + approximate_radius(right);
    distance_score(left.pose.position, right.pose.position, radius.max(0.05))
}

fn approximate_radius(object: &SceneObject) -> f32 {
    let dx = object.bbox.max.x - object.bbox.min.x;
    let dy = object.bbox.max.y - object.bbox.min.y;
    let dz = object.bbox.max.z - object.bbox.min.z;
    (dx.mul_add(dx, dy * dy) + dz * dz).sqrt() * 0.5
}

fn distance_score(left: Position, right: Position, tolerance: f32) -> f32 {
    let delta = position_distance(left, right);
    if delta <= tolerance {
        1.0
    } else {
        (1.0 / (1.0 + (delta - tolerance) / tolerance.max(0.1))).clamp(0.0, 1.0)
    }
}

fn contains_any(input: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| input.contains(term))
}

fn apply_action_conditioned_update(
    state: &mut WorldState,
    action: &Action,
    horizon: f32,
    assets: &JepaAssets,
) {
    let gain = assets.manifest.action_gain();
    match action {
        Action::Move { target, speed } => {
            if let Some(object_id) = primary_dynamic_object_id(&state.scene) {
                if let Some(object) = state.scene.get_object_mut(&object_id) {
                    let current = object.pose.position;
                    let factor = (speed.max(0.1) * gain * horizon).clamp(0.15, 1.0);
                    let next = lerp_position(current, *target, factor);
                    let velocity = velocity_between(current, next, horizon.max(0.05));
                    set_object_position(object, next);
                    object.velocity = velocity;
                }
            }
        }
        Action::Grasp { object, grip_force } => {
            if let Some(target) = state.scene.get_object_mut(object) {
                target.velocity = Velocity::default();
                translate_object(
                    target,
                    Position {
                        x: 0.0,
                        y: grip_force.clamp(0.0, 1.5) * 0.03,
                        z: 0.0,
                    },
                );
            }
        }
        Action::Release { object } => {
            if let Some(target) = state.scene.get_object_mut(object) {
                let gravity_drop = assets.manifest.gravity_bias() * horizon * 0.35;
                translate_object(
                    target,
                    Position {
                        x: 0.0,
                        y: -gravity_drop,
                        z: 0.0,
                    },
                );
                target.velocity.y -= assets.manifest.gravity_bias() * horizon * 2.5;
            }
        }
        Action::Push {
            object,
            direction,
            force,
        } => {
            let displacement = scaled_direction(*direction, force.max(0.1) * gain * horizon * 0.22);
            let mut propagated_origin = None;
            if let Some(target) = state.scene.get_object_mut(object) {
                translate_object(target, displacement);
                target.velocity = Velocity {
                    x: displacement.x / horizon.max(0.05),
                    y: displacement.y / horizon.max(0.05),
                    z: displacement.z / horizon.max(0.05),
                };
                propagated_origin = Some(target.pose.position);
            }
            if let Some(origin) = propagated_origin {
                propagate_impulse(state, *object, origin, displacement, assets);
            }
        }
        Action::Rotate {
            object,
            axis,
            angle,
        } => {
            if let Some(target) = state.scene.get_object_mut(object) {
                let delta = quaternion_from_axis_angle(*axis, *angle * gain);
                target.pose.rotation =
                    normalize_rotation(multiply_rotation(delta, target.pose.rotation));
            }
        }
        Action::Place { object, target } => {
            if let Some(item) = state.scene.get_object_mut(object) {
                let current = item.pose.position;
                set_object_position(item, *target);
                item.velocity = velocity_between(current, *target, horizon.max(0.05));
            }
        }
        Action::CameraMove { .. } | Action::CameraLookAt { .. } => {}
        Action::Navigate { waypoints } => {
            if let Some(destination) = waypoints.last().copied() {
                if let Some(object_id) = primary_dynamic_object_id(&state.scene) {
                    if let Some(target) = state.scene.get_object_mut(&object_id) {
                        let current = target.pose.position;
                        set_object_position(target, destination);
                        target.velocity = velocity_between(current, destination, horizon.max(0.05));
                    }
                }
            }
        }
        Action::Teleport { destination } => {
            if let Some(object_id) = primary_dynamic_object_id(&state.scene) {
                if let Some(target) = state.scene.get_object_mut(&object_id) {
                    set_object_position(target, destination.position);
                    target.pose.rotation = destination.rotation;
                    target.velocity = Velocity::default();
                }
            }
        }
        Action::SetWeather { weather } => {
            replace_tag(
                &mut state.metadata.tags,
                "weather:",
                &format!("weather:{weather:?}"),
            );
        }
        Action::SetLighting { time_of_day } => {
            replace_tag(
                &mut state.metadata.tags,
                "lighting:",
                &format!("lighting:{time_of_day:.1}"),
            );
        }
        Action::SpawnObject { template, pose } => {
            let extent = 0.12 + 0.03 * gain;
            let semantic_label = Some(template.clone());
            let object = SceneObject {
                id: uuid::Uuid::new_v4(),
                name: template.clone(),
                pose: *pose,
                bbox: centered_bbox(pose.position, extent),
                mesh: None,
                physics: Default::default(),
                velocity: Velocity::default(),
                semantic_label,
                visual_embedding: None,
            };
            state.scene.add_object(object);
        }
        Action::RemoveObject { object } => {
            state.scene.remove_object(object);
        }
        Action::Sequence(actions) | Action::Parallel(actions) => {
            for nested in actions {
                apply_action_conditioned_update(state, nested, horizon, assets);
            }
        }
        Action::Conditional {
            condition,
            then,
            otherwise,
        } => {
            if evaluate_condition(condition, state) {
                apply_action_conditioned_update(state, then, horizon, assets);
            } else if let Some(otherwise) = otherwise {
                apply_action_conditioned_update(state, otherwise, horizon, assets);
            }
        }
        Action::Raw { provider, data } => {
            if provider == "jepa" {
                apply_raw_jepa_action(state, data, horizon);
            }
        }
    }
}

fn apply_raw_jepa_action(state: &mut WorldState, data: &serde_json::Value, horizon: f32) {
    let Some(kind) = data.get("type").and_then(serde_json::Value::as_str) else {
        return;
    };

    if kind == "nudge" {
        let dx = data
            .get("dx")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        let dy = data
            .get("dy")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        let dz = data
            .get("dz")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        if let Some(object_id) = primary_dynamic_object_id(&state.scene) {
            if let Some(target) = state.scene.get_object_mut(&object_id) {
                let delta = Position {
                    x: dx * horizon,
                    y: dy * horizon,
                    z: dz * horizon,
                };
                translate_object(target, delta);
                target.velocity = Velocity {
                    x: delta.x / horizon.max(0.05),
                    y: delta.y / horizon.max(0.05),
                    z: delta.z / horizon.max(0.05),
                };
            }
        }
    }
}

fn apply_latent_relaxation(state: &mut WorldState, horizon: f32, assets: &JepaAssets) {
    let damping = assets.manifest.temporal_smoothness();
    let gravity = assets.manifest.gravity_bias() * 0.45;
    let object_ids = ordered_object_ids(&state.scene);

    for object_id in object_ids {
        let Some(object) = state.scene.get_object_mut(&object_id) else {
            continue;
        };
        if object.physics.is_static {
            continue;
        }

        translate_object(
            object,
            Position {
                x: object.velocity.x * horizon * 0.2,
                y: object.velocity.y * horizon * 0.2,
                z: object.velocity.z * horizon * 0.2,
            },
        );
        object.velocity.x *= damping;
        object.velocity.z *= damping;
        object.velocity.y = object.velocity.y * damping - gravity * horizon;

        if object.pose.position.y > 0.0 {
            let drop = gravity * horizon * horizon * 0.5;
            translate_object(
                object,
                Position {
                    x: 0.0,
                    y: -drop.min(object.pose.position.y),
                    z: 0.0,
                },
            );
        }

        if object.bbox.min.y < 0.0 {
            translate_object(
                object,
                Position {
                    x: 0.0,
                    y: -object.bbox.min.y,
                    z: 0.0,
                },
            );
            object.velocity.y = object.velocity.y.max(0.0) * 0.2;
        }
    }
}

fn propagate_impulse(
    state: &mut WorldState,
    source_id: uuid::Uuid,
    source_position: Position,
    displacement: Position,
    assets: &JepaAssets,
) {
    let transfer_scale = 0.15 * assets.manifest.collision_bias();
    let object_ids = ordered_object_ids(&state.scene);

    for object_id in object_ids {
        if object_id == source_id {
            continue;
        }
        let should_transfer = state
            .scene
            .get_object(&object_id)
            .map(|object| position_distance(object.pose.position, source_position) < 0.45)
            .unwrap_or(false);
        if !should_transfer {
            continue;
        }

        if let Some(target) = state.scene.get_object_mut(&object_id) {
            let delta = Position {
                x: displacement.x * transfer_scale,
                y: displacement.y * transfer_scale * 0.2,
                z: displacement.z * transfer_scale,
            };
            translate_object(target, delta);
            target.velocity = Velocity {
                x: delta.x,
                y: delta.y,
                z: delta.z,
            };
        }
    }
}

fn refresh_touching_relationships(scene: &mut SceneGraph) {
    scene
        .relationships
        .retain(|rel| !matches!(rel, SpatialRelationship::Touching { .. }));

    let object_ids = ordered_object_ids(scene);
    for (index, left) in object_ids.iter().enumerate() {
        for right in object_ids.iter().skip(index + 1) {
            let Some(a) = scene.get_object(left) else {
                continue;
            };
            let Some(b) = scene.get_object(right) else {
                continue;
            };
            if bbox_overlaps(&a.bbox, &b.bbox) {
                scene.relationships.push(SpatialRelationship::Touching {
                    a: *left,
                    b: *right,
                });
            }
        }
    }
}

fn score_prediction(
    input_state: &WorldState,
    output_state: &WorldState,
    action: &Action,
    assets: &JepaAssets,
) -> PhysicsScores {
    let input_objects = input_state.scene.objects.len() as f32;
    let output_objects = output_state.scene.objects.len() as f32;
    let object_delta = (input_objects - output_objects).abs();
    let count_penalty = (object_delta / input_objects.max(1.0)).min(1.0) * 0.35;
    let intent_bonus = if matches!(
        action,
        Action::SpawnObject { .. } | Action::RemoveObject { .. }
    ) {
        0.08
    } else {
        0.0
    };
    let object_permanence = (0.9 - count_penalty + intent_bonus).clamp(0.35, 0.99);

    let overlap_count = count_overlaps(&output_state.scene) as f32;
    let collision_accuracy =
        (0.88 + assets.manifest.collision_bias() * 0.04 - overlap_count * 0.12).clamp(0.2, 0.98);

    let avg_speed = average_speed(&output_state.scene);
    let temporal_consistency =
        (0.72 + assets.manifest.temporal_smoothness() * 0.22 - avg_speed * 0.05).clamp(0.2, 0.99);

    let avg_displacement = average_displacement(input_state, output_state);
    let spatial_consistency = (0.92 - avg_displacement * 0.08
        + assets.manifest.temporal_smoothness() * 0.04)
        .clamp(0.2, 0.99);

    let avg_vertical_velocity = average_vertical_velocity(&output_state.scene);
    let gravity_event_bonus = if action_mentions_vertical_dynamics(action) {
        0.12
    } else {
        0.0
    };
    let gravity_compliance = (0.72
        + assets.manifest.gravity_bias() * 0.12
        + avg_vertical_velocity.max(0.0) * 0.06
        + gravity_event_bonus)
        .clamp(0.2, 0.99);

    let overall = (object_permanence
        + collision_accuracy
        + spatial_consistency
        + temporal_consistency
        + gravity_compliance)
        / 5.0;

    PhysicsScores {
        overall,
        object_permanence,
        gravity_compliance,
        collision_accuracy,
        spatial_consistency,
        temporal_consistency,
    }
}

fn estimate_confidence(
    action: &Action,
    physics_scores: &PhysicsScores,
    assets: &JepaAssets,
    backend: JepaBackend,
) -> f32 {
    let backend_bias = match backend {
        JepaBackend::Burn => 0.08,
        JepaBackend::PyTorch => 0.05,
        JepaBackend::Onnx => 0.06,
        JepaBackend::Safetensors => 0.03,
    };
    let asset_richness = (assets.weight_files.len() as f32 * 0.02)
        + ((assets.total_bytes as f32 / 32_000_000.0).min(1.0) * 0.08);
    let complexity_penalty = action_complexity(action) as f32 * 0.025;

    (0.34
        + backend_bias
        + asset_richness
        + assets.manifest.confidence_bias()
        + physics_scores.overall * 0.42
        - complexity_penalty)
        .clamp(0.05, 0.98)
}

fn action_complexity(action: &Action) -> usize {
    match action {
        Action::Sequence(actions) | Action::Parallel(actions) => {
            1 + actions.iter().map(action_complexity).sum::<usize>()
        }
        Action::Conditional {
            then, otherwise, ..
        } => 1 + action_complexity(then) + otherwise.as_deref().map(action_complexity).unwrap_or(0),
        _ => 1,
    }
}

fn action_mentions_vertical_dynamics(action: &Action) -> bool {
    match action {
        Action::Release { .. } | Action::Place { .. } | Action::Grasp { .. } => true,
        Action::Sequence(actions) | Action::Parallel(actions) => {
            actions.iter().any(action_mentions_vertical_dynamics)
        }
        Action::Conditional {
            then, otherwise, ..
        } => {
            action_mentions_vertical_dynamics(then)
                || otherwise
                    .as_deref()
                    .map(action_mentions_vertical_dynamics)
                    .unwrap_or(false)
        }
        _ => false,
    }
}

fn average_speed(scene: &SceneGraph) -> f32 {
    if scene.objects.is_empty() {
        return 0.0;
    }
    scene
        .objects
        .values()
        .map(|object| object.velocity.magnitude())
        .sum::<f32>()
        / scene.objects.len() as f32
}

fn average_vertical_velocity(scene: &SceneGraph) -> f32 {
    let mut dynamic_objects = 0usize;
    let mut total = 0.0f32;

    for object in scene.objects.values() {
        if object.physics.is_static {
            continue;
        }
        dynamic_objects += 1;
        total += (-object.velocity.y).max(0.0);
    }

    if dynamic_objects == 0 {
        0.0
    } else {
        total / dynamic_objects as f32
    }
}

fn average_displacement(input_state: &WorldState, output_state: &WorldState) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;

    for (object_id, input_object) in &input_state.scene.objects {
        if let Some(output_object) = output_state.scene.objects.get(object_id) {
            total += position_distance(input_object.pose.position, output_object.pose.position);
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn count_overlaps(scene: &SceneGraph) -> usize {
    let object_ids = ordered_object_ids(scene);
    let mut overlaps = 0usize;

    for (index, left) in object_ids.iter().enumerate() {
        for right in object_ids.iter().skip(index + 1) {
            let Some(a) = scene.get_object(left) else {
                continue;
            };
            let Some(b) = scene.get_object(right) else {
                continue;
            };
            if bbox_overlaps(&a.bbox, &b.bbox) {
                overlaps += 1;
            }
        }
    }

    overlaps
}

fn ordered_object_ids(scene: &SceneGraph) -> Vec<uuid::Uuid> {
    let mut object_ids: Vec<_> = scene.objects.keys().copied().collect();
    object_ids.sort_by_key(|id| id.to_string());
    object_ids
}

fn primary_dynamic_object_id(scene: &SceneGraph) -> Option<uuid::Uuid> {
    let mut dynamic: Vec<_> = scene
        .objects
        .iter()
        .filter_map(|(id, object)| {
            (!object.physics.is_static).then_some((*id, object.name.clone()))
        })
        .collect();
    dynamic.sort_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)));
    dynamic
        .into_iter()
        .map(|(id, _)| id)
        .next()
        .or_else(|| ordered_object_ids(scene).into_iter().next())
}

fn translate_object(object: &mut SceneObject, delta: Position) {
    object.pose.position.x += delta.x;
    object.pose.position.y += delta.y;
    object.pose.position.z += delta.z;
    object.bbox.min.x += delta.x;
    object.bbox.min.y += delta.y;
    object.bbox.min.z += delta.z;
    object.bbox.max.x += delta.x;
    object.bbox.max.y += delta.y;
    object.bbox.max.z += delta.z;
}

fn set_object_position(object: &mut SceneObject, position: Position) {
    let delta = Position {
        x: position.x - object.pose.position.x,
        y: position.y - object.pose.position.y,
        z: position.z - object.pose.position.z,
    };
    translate_object(object, delta);
}

fn centered_bbox(center: Position, extent: f32) -> BBox {
    BBox {
        min: Position {
            x: center.x - extent,
            y: center.y - extent,
            z: center.z - extent,
        },
        max: Position {
            x: center.x + extent,
            y: center.y + extent,
            z: center.z + extent,
        },
    }
}

fn bbox_overlaps(left: &BBox, right: &BBox) -> bool {
    left.min.x <= right.max.x
        && left.max.x >= right.min.x
        && left.min.y <= right.max.y
        && left.max.y >= right.min.y
        && left.min.z <= right.max.z
        && left.max.z >= right.min.z
}

fn replace_tag(tags: &mut Vec<String>, prefix: &str, replacement: &str) {
    tags.retain(|tag| !tag.starts_with(prefix));
    tags.push(replacement.to_string());
}

fn lerp_position(start: Position, end: Position, factor: f32) -> Position {
    Position {
        x: start.x + (end.x - start.x) * factor,
        y: start.y + (end.y - start.y) * factor,
        z: start.z + (end.z - start.z) * factor,
    }
}

fn velocity_between(start: Position, end: Position, horizon: f32) -> Velocity {
    Velocity {
        x: (end.x - start.x) / horizon,
        y: (end.y - start.y) / horizon,
        z: (end.z - start.z) / horizon,
    }
}

fn position_distance(left: Position, right: Position) -> f32 {
    let dx = right.x - left.x;
    let dy = right.y - left.y;
    let dz = right.z - left.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn scaled_direction(direction: Vec3, scale: f32) -> Position {
    let length =
        (direction.x * direction.x + direction.y * direction.y + direction.z * direction.z).sqrt();
    if length < f32::EPSILON {
        return Position::default();
    }
    Position {
        x: direction.x / length * scale,
        y: direction.y / length * scale,
        z: direction.z / length * scale,
    }
}

fn quaternion_from_axis_angle(axis: Vec3, angle_degrees: f32) -> Rotation {
    let length = (axis.x * axis.x + axis.y * axis.y + axis.z * axis.z).sqrt();
    let normalized = if length < f32::EPSILON {
        Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    } else {
        Vec3 {
            x: axis.x / length,
            y: axis.y / length,
            z: axis.z / length,
        }
    };
    let half_angle = angle_degrees.to_radians() * 0.5;
    let sin = half_angle.sin();
    Rotation {
        w: half_angle.cos(),
        x: normalized.x * sin,
        y: normalized.y * sin,
        z: normalized.z * sin,
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

fn normalize_rotation(rotation: Rotation) -> Rotation {
    let magnitude = (rotation.w * rotation.w
        + rotation.x * rotation.x
        + rotation.y * rotation.y
        + rotation.z * rotation.z)
        .sqrt();
    if magnitude < f32::EPSILON {
        return Rotation::default();
    }
    Rotation {
        w: rotation.w / magnitude,
        x: rotation.x / magnitude,
        y: rotation.y / magnitude,
        z: rotation.z / magnitude,
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use worldforge_core::types::Pose;

    fn manifest_json(model_name: &str) -> String {
        format!(
            r#"{{
                "model_name": "{model_name}",
                "representation_dim": 1536,
                "action_gain": 1.2,
                "temporal_smoothness": 0.88,
                "gravity_bias": 0.92,
                "collision_bias": 0.9,
                "confidence_bias": 0.07
            }}"#
        )
    }

    struct TestModelDir {
        path: PathBuf,
    }

    impl TestModelDir {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "worldforge-jepa-tests-{name}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn write_weights(&self, filename: &str, bytes: &[u8]) {
            fs::write(self.path.join(filename), bytes).unwrap();
        }

        fn write_manifest(&self, body: &str) {
            fs::write(self.path.join("worldforge-jepa.json"), body).unwrap();
        }
    }

    impl Drop for TestModelDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn sample_state() -> (WorldState, uuid::Uuid) {
        let mut state = WorldState::new("local-jepa", "jepa");
        let object = SceneObject::new(
            "block",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                ..Default::default()
            },
            centered_bbox(
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                0.15,
            ),
        );
        let object_id = object.id;
        state.scene.add_object(object);
        (state, object_id)
    }

    #[test]
    fn test_jepa_provider_creation() {
        let provider = JepaProvider::new("/tmp/models/v-jepa-2", JepaBackend::Burn);
        assert_eq!(provider.name(), "jepa");
    }

    #[test]
    fn test_jepa_capabilities() {
        let provider = JepaProvider::new("/tmp/models", JepaBackend::Burn);
        let caps = provider.capabilities();
        assert!(caps.predict);
        assert!(!caps.generate);
        assert!(!caps.reason);
        assert!(!caps.transfer);
        assert!(caps.supports_planning);
        assert!(caps.action_conditioned);
    }

    #[test]
    fn test_jepa_cost_is_zero() {
        let provider = JepaProvider::new("/tmp/models", JepaBackend::Burn);
        let cost = provider.estimate_cost(&Operation::Predict {
            steps: 10,
            resolution: (224, 224),
        });
        assert_eq!(cost.usd, 0.0);
        assert!(cost.estimated_latency_ms > 0);
    }

    #[test]
    fn test_jepa_backend_serialization() {
        let backends = vec![
            JepaBackend::Burn,
            JepaBackend::PyTorch,
            JepaBackend::Onnx,
            JepaBackend::Safetensors,
        ];
        for backend in backends {
            let json = serde_json::to_string(&backend).unwrap();
            let roundtrip: JepaBackend = serde_json::from_str(&json).unwrap();
            assert_eq!(backend, roundtrip);
        }
    }

    #[test]
    fn test_jepa_backend_from_str() {
        assert_eq!("burn".parse::<JepaBackend>().unwrap(), JepaBackend::Burn);
        assert_eq!(
            "torch".parse::<JepaBackend>().unwrap(),
            JepaBackend::PyTorch
        );
        assert!("unknown".parse::<JepaBackend>().is_err());
    }

    #[tokio::test]
    async fn test_jepa_generate_unsupported() {
        let provider = JepaProvider::new("/tmp/models", JepaBackend::Burn);
        let result = provider
            .generate(
                &GenerationPrompt {
                    text: "test".to_string(),
                    reference_image: None,
                    negative_prompt: None,
                },
                &GenerationConfig::default(),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_jepa_reason_unsupported() {
        let provider = JepaProvider::new("/tmp/models", JepaBackend::Burn);
        let result = provider
            .reason(
                &ReasoningInput {
                    video: None,
                    state: None,
                },
                "will it fall?",
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_jepa_health_check_no_weights() {
        let provider = JepaProvider::new("/nonexistent/path", JepaBackend::Burn);
        let status = provider.health_check().await.unwrap();
        assert!(!status.healthy);
    }

    #[tokio::test]
    async fn test_jepa_health_check_with_assets() {
        let model_dir = TestModelDir::new("health");
        model_dir.write_weights("model.safetensors", b"weights");
        model_dir.write_manifest(&manifest_json("vjepa2-local"));

        let provider = JepaProvider::new(&model_dir.path, JepaBackend::Burn);
        let status = provider.health_check().await.unwrap();

        assert!(status.healthy);
        assert!(status.message.contains("vjepa2-local"));
    }

    #[tokio::test]
    async fn test_jepa_predict_requires_assets() {
        let provider = JepaProvider::new("/nonexistent/path", JepaBackend::Burn);
        let (state, object_id) = sample_state();
        let result = provider
            .predict(
                &state,
                &Action::Release { object: object_id },
                &PredictionConfig::default(),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_jepa_predict_with_assets_updates_state() {
        let model_dir = TestModelDir::new("predict");
        model_dir.write_weights("model.safetensors", b"latent-weights-go-here");
        model_dir.write_manifest(&manifest_json("vjepa2-local"));

        let provider = JepaProvider::new(&model_dir.path, JepaBackend::Burn);
        let (state, object_id) = sample_state();
        let prediction = provider
            .predict(
                &state,
                &Action::Push {
                    object: object_id,
                    direction: Vec3 {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    force: 2.5,
                },
                &PredictionConfig {
                    steps: 4,
                    fps: 8.0,
                    ..PredictionConfig::default()
                },
            )
            .await
            .unwrap();

        let before = state.scene.get_object(&object_id).unwrap().pose.position;
        let after = prediction
            .output_state
            .scene
            .get_object(&object_id)
            .unwrap()
            .pose
            .position;

        assert!(after.x > before.x);
        assert!(prediction.confidence > 0.4);
        assert!(prediction.physics_scores.overall > 0.4);
        assert!(prediction.latency_ms > 0);
        assert!(prediction.model.contains("vjepa2-local"));
    }

    #[tokio::test]
    async fn test_jepa_native_plan_moves_object_to_goal() {
        let model_dir = TestModelDir::new("native-plan");
        model_dir.write_weights("model.safetensors", b"latent-weights-go-here");
        model_dir.write_manifest(&manifest_json("vjepa2-local"));

        let provider = JepaProvider::new(&model_dir.path, JepaBackend::Burn);
        let (state, object_id) = sample_state();
        let request = PlanRequest {
            current_state: state,
            goal: PlanGoal::Description("move block to position (1.2, 1.0, 0.0)".to_string()),
            max_steps: 4,
            guardrails: Vec::new(),
            planner: worldforge_core::prediction::PlannerType::ProviderNative,
            timeout_seconds: 5.0,
        };

        let plan = provider.plan(&request).await.unwrap();
        let final_state = plan.predicted_states.last().unwrap();
        let final_position = final_state
            .scene
            .get_object(&object_id)
            .unwrap()
            .pose
            .position;

        assert!(!plan.actions.is_empty());
        assert!(final_position.x > 1.0);
        assert!(plan.success_probability > 0.9);
    }

    #[tokio::test]
    async fn test_jepa_native_plan_supports_goal_image() {
        let model_dir = TestModelDir::new("goal-image");
        model_dir.write_weights("model.safetensors", b"latent-weights-go-here");
        model_dir.write_manifest(&manifest_json("vjepa2-local"));

        let provider = JepaProvider::new(&model_dir.path, JepaBackend::Burn);
        let (state, object_id) = sample_state();
        let initial_position = state.scene.get_object(&object_id).unwrap().pose.position;
        let mut target_state = state.clone();
        target_state
            .scene
            .get_object_mut(&object_id)
            .unwrap()
            .set_position(Position {
                x: 1.2,
                y: 1.0,
                z: 0.0,
            });
        let goal = PlanGoal::GoalImage(worldforge_core::goal_image::render_scene_goal_image(
            &target_state,
            (32, 24),
        ));
        let request = PlanRequest {
            current_state: state,
            goal,
            max_steps: 4,
            guardrails: Vec::new(),
            planner: worldforge_core::prediction::PlannerType::ProviderNative,
            timeout_seconds: 5.0,
        };

        let plan = provider.plan(&request).await.unwrap();
        let final_state = plan.predicted_states.last().unwrap();
        let final_position = final_state
            .scene
            .get_object(&object_id)
            .unwrap()
            .pose
            .position;

        assert!(!plan.actions.is_empty());
        assert!(final_position.x > initial_position.x);
        assert!(plan.success_probability > 0.9);
    }
}
