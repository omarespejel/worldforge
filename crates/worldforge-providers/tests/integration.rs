//! Cross-crate integration tests for worldforge-providers.
//!
//! Tests the full workflow of provider registration, capability
//! querying, prediction, health checks, and multi-provider comparison.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use worldforge_core::action::{Action, ActionSpaceType, ActionTranslator, ActionType, Weather};
use worldforge_core::error::WorldForgeError;
use worldforge_core::guardrail::{Guardrail, GuardrailConfig};
use worldforge_core::prediction::{PlanGoal, PlanRequest, PlannerType, PredictionConfig};
use worldforge_core::provider::{
    EmbeddingInput, GenerationConfig, GenerationPrompt, ProviderRegistry, ReasoningInput,
    SpatialControls, TransferConfig, WorldModelProvider,
};
use worldforge_core::scene::SceneObject;
use worldforge_core::state::WorldState;
use worldforge_core::types::{
    BBox, DType, Pose, Position, SimTime, Tensor, TensorData, Trajectory, Vec3, VideoClip,
};
use worldforge_core::world::World;
use worldforge_providers::cosmos::{
    CosmosConfig, CosmosEndpoint, CosmosModel, CosmosPhysicsScores, CosmosPredictResponse,
};
use worldforge_providers::genie::{GenieModel, GenieProvider};
use worldforge_providers::runway::RunwayModel;
use worldforge_providers::{
    auto_detect, auto_detect_worldforge, auto_detect_worldforge_with_state_store,
    CosmosActionTranslator, CosmosProvider, JepaBackend, JepaProvider, MarbleProvider,
    MockProvider, RunwayActionTranslator, RunwayProvider,
};

struct TestModelDir {
    path: std::path::PathBuf,
}

impl TestModelDir {
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "worldforge-providers-integration-{name}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn write_assets(&self) {
        fs::write(self.path.join("model.safetensors"), b"jepa-weights").unwrap();
        fs::write(
            self.path.join("worldforge-jepa.json"),
            r#"{
                "model_name": "vjepa2-local",
                "representation_dim": 2048,
                "action_gain": 1.25,
                "temporal_smoothness": 0.9,
                "gravity_bias": 0.95,
                "collision_bias": 0.88,
                "confidence_bias": 0.08
            }"#,
        )
        .unwrap();
    }
}

impl Drop for TestModelDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn sample_jepa_state() -> (WorldState, uuid::Uuid) {
    let mut state = WorldState::new("jepa-world", "jepa");
    let object = SceneObject::new(
        "crate",
        Pose {
            position: Position {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            ..Default::default()
        },
        BBox {
            min: Position {
                x: -0.15,
                y: 0.85,
                z: -0.15,
            },
            max: Position {
                x: 0.15,
                y: 1.15,
                z: 0.15,
            },
        },
    );
    let object_id = object.id;
    state.scene.add_object(object);
    (state, object_id)
}

#[derive(Debug)]
struct RecordedRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

fn spawn_fake_http_server(
    response_body: String,
) -> (
    String,
    mpsc::Receiver<RecordedRequest>,
    thread::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();

    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());

        let mut request_line = String::new();
        reader.read_line(&mut request_line).unwrap();
        let request_line = request_line.trim_end_matches(['\r', '\n']);
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or_default().to_string();
        let path = parts.next().unwrap_or_default().to_string();

        let mut headers = HashMap::new();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                break;
            }

            if let Some((name, value)) = line.split_once(':') {
                let key = name.trim().to_ascii_lowercase();
                let value = value.trim().to_string();
                if key == "content-length" {
                    content_length = value.parse().unwrap_or(0);
                }
                headers.insert(key, value);
            }
        }

        let mut body_bytes = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body_bytes).unwrap();
        }

        let body = String::from_utf8(body_bytes).unwrap_or_default();
        tx.send(RecordedRequest {
            method,
            path,
            headers,
            body,
        })
        .unwrap();

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.flush().unwrap();
    });

    (format!("http://{}", addr), rx, handle)
}

fn spawn_route_http_server(
    responses: HashMap<String, String>,
) -> (
    String,
    mpsc::Receiver<RecordedRequest>,
    thread::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();
    let request_count = responses.len();

    let handle = thread::spawn(move || {
        for _ in 0..request_count {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());

            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let request_line = request_line.trim_end_matches(['\r', '\n']);
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let path = parts.next().unwrap_or_default().to_string();

            let mut headers = HashMap::new();
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let line = line.trim_end_matches(['\r', '\n']);
                if line.is_empty() {
                    break;
                }

                if let Some((name, value)) = line.split_once(':') {
                    let key = name.trim().to_ascii_lowercase();
                    let value = value.trim().to_string();
                    if key == "content-length" {
                        content_length = value.parse().unwrap_or(0);
                    }
                    headers.insert(key, value);
                }
            }

            let mut body_bytes = vec![0u8; content_length];
            if content_length > 0 {
                reader.read_exact(&mut body_bytes).unwrap();
            }

            let body = String::from_utf8(body_bytes).unwrap_or_default();
            let response_body = responses
                .get(&path)
                .unwrap_or_else(|| panic!("unexpected request path: {path}"));
            tx.send(RecordedRequest {
                method,
                path,
                headers,
                body,
            })
            .unwrap();

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        }
    });

    (format!("http://{}", addr), rx, handle)
}

fn sample_video_clip() -> VideoClip {
    VideoClip {
        frames: vec![worldforge_core::types::Frame {
            data: Tensor::zeros(vec![2, 2, 3], DType::UInt8),
            timestamp: SimTime::default(),
            camera: None,
            depth: None,
            segmentation: None,
        }],
        fps: 12.0,
        resolution: (2, 2),
        duration: 0.083333333,
    }
}

fn sample_genie_state() -> WorldState {
    let mut state = WorldState::new("genie-world", "genie");
    state.scene.add_object(SceneObject::new(
        "cube",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.5,
                z: 0.0,
            },
            ..Default::default()
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
    ));
    state
}

fn sample_genie_action() -> Action {
    Action::SetWeather {
        weather: Weather::Cloudy,
    }
}

fn sample_genie_prompt() -> GenerationPrompt {
    GenerationPrompt {
        text: "A small robot exploring a playable kitchen".to_string(),
        reference_image: None,
        negative_prompt: Some("blurry".to_string()),
    }
}

fn sample_genie_reasoning_state() -> WorldState {
    let mut state = WorldState::new("genie-reason-world", "genie");

    state.scene.add_object(SceneObject::new(
        "table",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            ..Pose::default()
        },
        BBox {
            min: Position {
                x: -0.75,
                y: -0.05,
                z: -0.75,
            },
            max: Position {
                x: 0.75,
                y: 0.05,
                z: 0.75,
            },
        },
    ));
    state.scene.add_object(SceneObject::new(
        "mug",
        Pose {
            position: Position {
                x: 0.1,
                y: 0.82,
                z: 0.0,
            },
            ..Pose::default()
        },
        BBox {
            min: Position {
                x: 0.05,
                y: 0.77,
                z: -0.05,
            },
            max: Position {
                x: 0.15,
                y: 0.87,
                z: 0.05,
            },
        },
    ));

    state
}

fn sample_marble_state() -> WorldState {
    let mut state = WorldState::new("marble-world", "marble");

    state.scene.add_object(SceneObject::new(
        "cube",
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
    ));

    state
}

fn sample_genie_transfer_controls() -> SpatialControls {
    SpatialControls {
        camera_trajectory: Some(Trajectory {
            poses: vec![
                (
                    SimTime {
                        step: 0,
                        seconds: 0.0,
                        dt: 0.0,
                    },
                    Pose::default(),
                ),
                (
                    SimTime {
                        step: 1,
                        seconds: 1.0,
                        dt: 1.0,
                    },
                    Pose {
                        position: Position {
                            x: 1.0,
                            y: 1.5,
                            z: 2.0,
                        },
                        ..Pose::default()
                    },
                ),
            ],
            velocities: None,
        }),
        depth_map: None,
        segmentation_map: None,
    }
}

fn sample_planning_scene(provider: &str) -> WorldState {
    let mut state = WorldState::new("planning-scene", provider);

    state.scene.add_object(SceneObject::new(
        "cube",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.8,
                z: 0.0,
            },
            ..Pose::default()
        },
        BBox {
            min: Position {
                x: -0.1,
                y: 0.7,
                z: -0.1,
            },
            max: Position {
                x: 0.1,
                y: 0.9,
                z: 0.1,
            },
        },
    ));
    state.scene.add_object(SceneObject::new(
        "mug",
        Pose {
            position: Position {
                x: 0.8,
                y: 0.8,
                z: 0.0,
            },
            ..Pose::default()
        },
        BBox {
            min: Position {
                x: 0.7,
                y: 0.7,
                z: -0.1,
            },
            max: Position {
                x: 0.9,
                y: 0.9,
                z: 0.1,
            },
        },
    ));

    state
}

fn action_kind(action: &Action) -> &'static str {
    match action {
        Action::Move { .. } => "move",
        Action::Grasp { .. } => "grasp",
        Action::Release { .. } => "release",
        Action::Push { .. } => "push",
        Action::Rotate { .. } => "rotate",
        Action::Place { .. } => "place",
        Action::CameraMove { .. } => "camera_move",
        Action::CameraLookAt { .. } => "camera_look_at",
        Action::Navigate { .. } => "navigate",
        Action::Teleport { .. } => "teleport",
        Action::SetWeather { .. } => "set_weather",
        Action::SetLighting { .. } => "set_lighting",
        Action::SpawnObject { .. } => "spawn",
        Action::RemoveObject { .. } => "remove",
        Action::Sequence(_) => "sequence",
        Action::Parallel(_) => "parallel",
        Action::Conditional { .. } => "conditional",
        Action::Raw { .. } => "raw",
    }
}

fn action_kinds(actions: &[Action]) -> Vec<&'static str> {
    actions.iter().map(action_kind).collect()
}

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn env_lock() -> &'static Mutex<()> {
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvVarGuard {
    previous: Vec<(String, Option<std::ffi::OsString>)>,
}

impl EnvVarGuard {
    fn new(vars: &[(&str, &str)]) -> Self {
        let previous = vars
            .iter()
            .map(|(name, _)| ((*name).to_string(), std::env::var_os(name)))
            .collect();

        for (name, value) in vars {
            std::env::set_var(name, value);
        }

        Self { previous }
    }

    fn clear(vars: &[&str]) -> Self {
        let previous = vars
            .iter()
            .map(|name| ((*name).to_string(), std::env::var_os(name)))
            .collect();

        for name in vars {
            std::env::remove_var(name);
        }

        Self { previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for (name, previous_value) in self.previous.drain(..) {
            match previous_value {
                Some(value) => std::env::set_var(&name, value),
                None => std::env::remove_var(&name),
            }
        }
    }
}

fn with_env_vars<T>(vars: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
    let _guard = env_lock().lock().unwrap();
    let _env = EnvVarGuard::new(vars);
    f()
}

#[test]
fn test_auto_detect_registry_mock_present() {
    let registry = auto_detect();
    assert!(registry.get("mock").is_ok());
}

#[test]
fn test_auto_detect_registry_includes_genie_without_env_vars() {
    let _guard = env_lock().lock().unwrap();
    let _env = EnvVarGuard::clear(&[
        "NVIDIA_API_KEY",
        "NVIDIA_API_ENDPOINT",
        "RUNWAY_API_SECRET",
        "JEPA_MODEL_PATH",
        "JEPA_BACKEND",
        "GENIE_API_KEY",
        "GENIE_API_ENDPOINT",
    ]);

    let registry = auto_detect();
    assert!(registry.get("genie").is_ok());
    assert!(registry
        .find_by_capability("reason")
        .iter()
        .any(|provider| provider.name() == "genie"));
}

#[test]
fn test_auto_detect_registry_registers_marble_when_available() {
    let registry = auto_detect();
    let Ok(_) = registry.get("marble") else {
        return;
    };

    assert!(registry.list().contains(&"marble"));

    let descriptor = registry.describe("marble").unwrap();
    assert_eq!(descriptor.name, "marble");
}

#[test]
fn test_full_stack_cosmos_constructor_exposes_merged_capabilities() {
    let provider = CosmosProvider::full_stack(
        "cosmos-test-key",
        CosmosEndpoint::NimApi("https://example.invalid/cosmos".to_string()),
    );
    let capabilities = provider.capabilities();

    assert!(capabilities.predict);
    assert!(capabilities.generate);
    assert!(capabilities.reason);
    assert!(capabilities.transfer);
    assert!(capabilities.embed);
}

#[tokio::test]
async fn test_cosmos_full_stack_native_planning_surface() {
    let provider = CosmosProvider::full_stack(
        "cosmos-test-key",
        CosmosEndpoint::NimApi("https://example.invalid/cosmos".to_string()),
    );
    let capabilities = provider.capabilities();

    let request = PlanRequest {
        current_state: WorldState::new("cosmos-native-plan", "cosmos"),
        goal: PlanGoal::Description("spawn a cube near the mug".to_string()),
        max_steps: 4,
        guardrails: Vec::new(),
        planner: PlannerType::ProviderNative,
        timeout_seconds: 5.0,
        fallback_provider: None,
    };

    let plan = provider.plan(&request).await.unwrap();
    assert!(capabilities.supports_planning);
    assert!(!plan.actions.is_empty());
    assert_eq!(plan.predicted_states.len(), plan.actions.len());
    assert_eq!(plan.guardrail_compliance.len(), plan.actions.len());
    let predicted_videos = plan
        .predicted_videos
        .as_ref()
        .expect("cosmos native planning should emit a storyboard");
    assert_eq!(predicted_videos.len(), plan.actions.len());
    assert!(predicted_videos.iter().all(|clip| !clip.frames.is_empty()));
    assert!(plan.iterations_used > 0);
    assert!((0.0..=1.0).contains(&plan.success_probability));
    assert!(plan.actions.iter().any(|action| matches!(
        action,
        Action::SpawnObject { template, .. } if template.to_lowercase().contains("cube")
    )));
    assert!(action_kinds(&plan.actions).contains(&"spawn"));
}

#[test]
fn test_auto_detect_registers_full_stack_cosmos() {
    with_env_vars(
        &[
            ("NVIDIA_API_KEY", "cosmos-test-key"),
            ("NVIDIA_API_ENDPOINT", "https://example.invalid/cosmos"),
        ],
        || {
            let registry = auto_detect();
            let descriptor = registry.describe("cosmos").unwrap();
            assert!(descriptor.capabilities.predict);
            assert!(descriptor.capabilities.generate);
            assert!(descriptor.capabilities.reason);
            assert!(descriptor.capabilities.transfer);
            assert!(descriptor.capabilities.embed);
            assert!(descriptor.capabilities.supports_planning);
            assert!(registry
                .find_by_capability("reason")
                .iter()
                .any(|provider| provider.name() == "cosmos"));
            assert!(registry
                .find_by_capability("transfer")
                .iter()
                .any(|provider| provider.name() == "cosmos"));
            assert!(registry
                .find_by_capability("embed")
                .iter()
                .any(|provider| provider.name() == "cosmos"));
            assert!(registry
                .find_by_capability("planning")
                .iter()
                .any(|provider| provider.name() == "cosmos"));
        },
    );
}

#[test]
fn test_full_stack_runway_constructor_exposes_merged_capabilities() {
    let provider = RunwayProvider::full_stack("runway-test-secret");
    let capabilities = provider.capabilities();

    assert!(capabilities.predict);
    assert!(capabilities.generate);
    assert!(capabilities.transfer);
    assert!(capabilities.action_conditioned);
    assert!(capabilities.multi_view);
    assert!(!capabilities.embed);
}

#[tokio::test]
async fn test_runway_full_stack_native_planning_surface() {
    let provider = RunwayProvider::full_stack("runway-test-secret");
    let capabilities = provider.capabilities();
    let current_state = sample_planning_scene("runway");
    let cube_id = current_state
        .scene
        .objects
        .values()
        .find(|object| object.name == "cube")
        .map(|object| object.id)
        .unwrap();
    let mut target_state = current_state.clone();
    target_state
        .scene
        .get_object_mut(&cube_id)
        .unwrap()
        .set_position(Position {
            x: 0.55,
            y: 0.8,
            z: 0.0,
        });

    let request = PlanRequest {
        current_state,
        goal: PlanGoal::TargetState(Box::new(target_state)),
        max_steps: 4,
        guardrails: Vec::new(),
        planner: PlannerType::ProviderNative,
        timeout_seconds: 5.0,
        fallback_provider: None,
    };

    let plan = provider.plan(&request).await.unwrap();
    assert!(capabilities.supports_planning);
    assert!(!plan.actions.is_empty());
    assert_eq!(plan.predicted_states.len(), plan.actions.len());
    assert_eq!(plan.guardrail_compliance.len(), plan.actions.len());
    let predicted_videos = plan
        .predicted_videos
        .as_ref()
        .expect("runway native planning should emit a storyboard");
    assert_eq!(predicted_videos.len(), plan.actions.len());
    assert!(predicted_videos.iter().all(|clip| !clip.frames.is_empty()));
    assert!(plan.iterations_used > 0);
    assert!((0.0..=1.0).contains(&plan.success_probability));
    let kinds = action_kinds(&plan.actions);
    assert_eq!(kinds, vec!["navigate", "grasp", "move", "place"]);
    assert!(matches!(
        &plan.actions[0],
        Action::Navigate { waypoints } if !waypoints.is_empty()
    ));
    assert!(matches!(
        &plan.actions[1],
        Action::Grasp { object, .. } if *object == cube_id
    ));
    assert!(matches!(&plan.actions[2], Action::Move { .. }));
    assert!(matches!(
        &plan.actions[3],
        Action::Place { object, .. } if *object == cube_id
    ));
    let final_state = plan.predicted_states.last().unwrap();
    let final_cube = final_state.scene.get_object(&cube_id).unwrap();
    assert_eq!(
        final_cube.pose.position,
        Position {
            x: 0.55,
            y: 0.8,
            z: 0.0,
        }
    );
}

#[tokio::test]
async fn test_provider_native_planning_profiles_diverge_on_same_goal() {
    let cosmos = CosmosProvider::full_stack(
        "cosmos-test-key",
        CosmosEndpoint::NimApi("https://example.invalid/cosmos".to_string()),
    );
    let runway = RunwayProvider::full_stack("runway-test-secret");
    let request = PlanRequest {
        current_state: sample_planning_scene("planning"),
        goal: PlanGoal::Description("place the cube next to the mug".to_string()),
        max_steps: 5,
        guardrails: Vec::new(),
        planner: PlannerType::ProviderNative,
        timeout_seconds: 5.0,
        fallback_provider: None,
    };

    let cosmos_plan = cosmos.plan(&request).await.unwrap();
    let runway_plan = runway.plan(&request).await.unwrap();

    assert_eq!(
        cosmos_plan.predicted_videos.as_ref().unwrap().len(),
        cosmos_plan.actions.len()
    );
    assert_eq!(
        runway_plan.predicted_videos.as_ref().unwrap().len(),
        runway_plan.actions.len()
    );
    assert!(cosmos_plan
        .predicted_videos
        .as_ref()
        .unwrap()
        .iter()
        .all(|clip| !clip.frames.is_empty()));
    assert!(runway_plan
        .predicted_videos
        .as_ref()
        .unwrap()
        .iter()
        .all(|clip| !clip.frames.is_empty()));

    let cosmos_kinds = action_kinds(&cosmos_plan.actions);
    let runway_kinds = action_kinds(&runway_plan.actions);
    assert_ne!(cosmos_kinds, runway_kinds);
    assert!(runway_kinds.iter().any(|kind| matches!(
        *kind,
        "move" | "grasp" | "release" | "navigate" | "teleport"
    )));
    assert!(cosmos_kinds
        .iter()
        .any(|kind| matches!(*kind, "place" | "spawn" | "conditional")));
}

#[tokio::test]
async fn test_runway_native_planning_blocks_boundary_constraint() {
    let provider = RunwayProvider::full_stack("runway-test-secret");
    let current_state = sample_planning_scene("runway-guardrail");
    let cube_id = current_state
        .scene
        .objects
        .values()
        .find(|object| object.name == "cube")
        .map(|object| object.id)
        .unwrap();
    let mut target_state = current_state.clone();
    target_state
        .scene
        .get_object_mut(&cube_id)
        .unwrap()
        .set_position(Position {
            x: 0.55,
            y: 0.8,
            z: 0.0,
        });
    let request = PlanRequest {
        current_state,
        goal: PlanGoal::TargetState(Box::new(target_state)),
        max_steps: 4,
        guardrails: vec![GuardrailConfig {
            guardrail: Guardrail::BoundaryConstraint {
                bounds: BBox {
                    min: Position {
                        x: -0.5,
                        y: 0.79,
                        z: -0.5,
                    },
                    max: Position {
                        x: 0.5,
                        y: 0.805,
                        z: 0.5,
                    },
                },
            },
            blocking: true,
        }],
        planner: PlannerType::ProviderNative,
        timeout_seconds: 5.0,
        fallback_provider: None,
    };

    let error = provider.plan(&request).await.unwrap_err();
    assert!(matches!(
        error,
        WorldForgeError::NoFeasiblePlan { reason, .. } if reason.contains("guardrail-blocked step")
    ));
}

#[test]
fn test_auto_detect_registers_full_stack_runway() {
    with_env_vars(&[("RUNWAY_API_SECRET", "runway-test-secret")], || {
        let registry = auto_detect();
        let descriptor = registry.describe("runway").unwrap();
        assert!(descriptor.capabilities.predict);
        assert!(descriptor.capabilities.generate);
        assert!(descriptor.capabilities.transfer);
        assert!(descriptor.capabilities.action_conditioned);
        assert!(descriptor.capabilities.multi_view);
        assert!(!descriptor.capabilities.embed);
        assert!(descriptor.capabilities.supports_planning);
        assert!(registry
            .find_by_capability("predict")
            .iter()
            .any(|provider| provider.name() == "runway"));
        assert!(registry
            .find_by_capability("generate")
            .iter()
            .any(|provider| provider.name() == "runway"));
        assert!(registry
            .find_by_capability("transfer")
            .iter()
            .any(|provider| provider.name() == "runway"));
        assert!(registry
            .find_by_capability("planning")
            .iter()
            .any(|provider| provider.name() == "runway"));
    });
}

#[test]
fn test_auto_detect_worldforge_creates_usable_world() {
    let worldforge = auto_detect_worldforge();

    assert!(worldforge.providers().contains(&"mock"));

    let world = worldforge.create_world("auto-world", "mock").unwrap();
    assert_eq!(world.default_provider, "mock");
    assert_eq!(world.current_state().metadata.created_by, "mock");
}

#[tokio::test]
async fn test_auto_detect_worldforge_with_state_store_roundtrip() {
    let dir = std::env::temp_dir().join(format!("wf-provider-auto-{}", uuid::Uuid::new_v4()));
    let store: worldforge_core::state::DynStateStore =
        Arc::new(worldforge_core::state::FileStateStore::new(&dir));
    let worldforge = auto_detect_worldforge_with_state_store(store);

    let mut world = worldforge.create_world("auto-store-world", "mock").unwrap();
    let object = SceneObject::new(
        "mug",
        Pose {
            position: Position {
                x: 0.2,
                y: 0.8,
                z: 0.0,
            },
            ..Default::default()
        },
        BBox {
            min: Position {
                x: 0.1,
                y: 0.7,
                z: -0.1,
            },
            max: Position {
                x: 0.3,
                y: 0.9,
                z: 0.1,
            },
        },
    );
    let object_id = object.id;
    world.add_object(object).unwrap();

    let saved_id = worldforge.save_world(&world).await.unwrap();
    assert_eq!(saved_id, world.id());

    let loaded_state = worldforge.load_state(&saved_id).await.unwrap();
    assert_eq!(loaded_state.metadata.created_by, "mock");
    assert!(loaded_state.scene.get_object(&object_id).is_some());

    let loaded_world = worldforge.load_world_from_store(&saved_id).await.unwrap();
    assert_eq!(loaded_world.default_provider, "mock");
    assert_eq!(
        loaded_world
            .current_state()
            .scene
            .get_object(&object_id)
            .unwrap()
            .name,
        "mug"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[test]
fn test_provider_capabilities_querying() {
    let mock = MockProvider::new();
    let caps = mock.capabilities();
    assert!(caps.predict);
    assert!(caps.generate);
    assert!(caps.action_conditioned);
    assert!(!caps.supported_action_spaces.is_empty());
}

#[test]
fn test_registry_find_by_capability() {
    let mut registry = ProviderRegistry::new();
    registry.register(Box::new(MockProvider::new()));
    registry.register(Box::new(JepaProvider::new(
        "/tmp/models",
        JepaBackend::Burn,
    )));

    let predictors = registry.find_by_capability("predict");
    assert_eq!(predictors.len(), 2);
    assert!(predictors.iter().any(|provider| provider.name() == "mock"));
    assert!(predictors.iter().any(|provider| provider.name() == "jepa"));

    let embedders = registry.find_by_capability("embed");
    assert_eq!(embedders.len(), 2);
    assert!(embedders.iter().any(|provider| provider.name() == "mock"));
    assert!(embedders.iter().any(|provider| provider.name() == "jepa"));

    let reasoners = registry.find_by_capability("reason");
    assert_eq!(reasoners.len(), 2);
    assert!(reasoners.iter().any(|provider| provider.name() == "mock"));
    assert!(reasoners.iter().any(|provider| provider.name() == "jepa"));

    let planners = registry.find_by_capability("planning");
    assert!(planners.iter().any(|provider| provider.name() == "jepa"));
}

#[test]
fn test_genie_provider_capabilities_are_distinctive() {
    let provider = GenieProvider::new(GenieModel::Genie3, "test-key");
    let caps = provider.capabilities();

    assert_eq!(provider.name(), "genie");
    assert!(caps.predict);
    assert!(caps.generate);
    assert!(caps.reason);
    assert!(caps.transfer);
    assert!(caps.action_conditioned);
    assert!(caps.supports_planning);
    assert_eq!(
        caps.supported_action_spaces,
        vec![ActionSpaceType::Discrete, ActionSpaceType::Language]
    );
    assert_eq!(caps.max_resolution, (256, 256));
    assert_eq!(caps.fps_range, (6.0, 12.0));
}

#[tokio::test]
async fn test_genie_reason_roundtrip_is_grounded_in_scene_state() {
    let provider = GenieProvider::new(GenieModel::Genie3, "genie-test-key");
    let input = ReasoningInput {
        video: None,
        state: Some(sample_genie_reasoning_state()),
    };

    let output = provider
        .reason(&input, "What objects are in the scene?")
        .await
        .unwrap();

    let answer = output.answer.to_lowercase();
    assert!(answer.contains("mug"));
    assert!(answer.contains("table"));
    assert!((0.0..=1.0).contains(&output.confidence));
    assert!(!output.evidence.is_empty());
    assert!(output
        .evidence
        .iter()
        .any(|entry| entry.to_lowercase().contains("mug")));
    assert!(output
        .evidence
        .iter()
        .any(|entry| entry.to_lowercase().contains("table")));
}

#[tokio::test]
async fn test_genie_transfer_roundtrip_respects_controls() {
    let provider = GenieProvider::new(GenieModel::Genie3, "genie-test-key");
    let source = sample_video_clip();
    let controls = sample_genie_transfer_controls();
    let config = TransferConfig {
        resolution: (256, 144),
        fps: 12.0,
        control_strength: 0.75,
    };

    let clip = provider
        .transfer(&source, &controls, &config)
        .await
        .unwrap();

    assert_eq!(clip.resolution, config.resolution);
    assert_eq!(clip.fps, config.fps);
    assert!((clip.duration - source.duration).abs() < 1.0e-6);
    assert!(!clip.frames.is_empty());
    assert!(clip.frames.iter().all(|frame| frame.camera.is_some()));
    assert!(clip
        .frames
        .iter()
        .any(|frame| frame.data.shape == vec![144, 256, 3]));
}

#[tokio::test]
async fn test_genie_provider_native_plan_spawn_goal() {
    let provider = GenieProvider::new(GenieModel::Genie3, "genie-test-key");
    let request = PlanRequest {
        current_state: WorldState::new("genie-native-plan", "genie"),
        goal: PlanGoal::Description("spawn cube".to_string()),
        max_steps: 4,
        guardrails: Vec::new(),
        planner: PlannerType::ProviderNative,
        timeout_seconds: 5.0,
        fallback_provider: None,
    };

    let caps = provider.capabilities();
    assert!(caps.supports_planning);

    let plan = provider.plan(&request).await.unwrap();
    assert!(!plan.actions.is_empty());
    assert_eq!(plan.predicted_states.len(), plan.actions.len());
    assert_eq!(plan.guardrail_compliance.len(), plan.actions.len());
    assert!(plan.iterations_used > 0);
    assert!((0.0..=1.0).contains(&plan.success_probability));
    assert!(plan.actions.iter().any(|action| matches!(
        action,
        Action::SpawnObject { template, .. } if template.to_lowercase().contains("cube")
    )));
}

fn mock_supports_native_planning() -> bool {
    MockProvider::new().capabilities().supports_planning
}

#[tokio::test]
async fn test_mock_provider_predict_workflow() {
    let mock = MockProvider::new();
    let state = WorldState::new("test_world", "mock");
    let action = Action::Move {
        target: Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = PredictionConfig::default();

    let prediction = mock.predict(&state, &action, &config).await.unwrap();
    assert_eq!(prediction.provider, "mock");
    assert!(prediction.confidence >= 0.0);
    assert!(prediction.confidence <= 1.0);
}

#[tokio::test]
async fn test_mock_provider_health_check() {
    let mock = MockProvider::new();
    let status = mock.health_check().await.unwrap();
    assert!(status.healthy);
}

#[tokio::test]
async fn test_mock_provider_embed_is_deterministic() {
    let mock = MockProvider::new();
    let input = EmbeddingInput::from_text("a red cube on a table");

    let first = mock.embed(&input).await.unwrap();
    let second = mock.embed(&input).await.unwrap();

    assert_eq!(first.provider, "mock");
    assert_eq!(first.model, "mock-embedding-v1");
    assert_eq!(second.provider, "mock");
    assert_eq!(second.model, "mock-embedding-v1");

    match (&first.embedding.data, &second.embedding.data) {
        (TensorData::Float32(left), TensorData::Float32(right)) => assert_eq!(left, right),
        other => panic!("unexpected embedding tensor data: {other:?}"),
    }
}

#[tokio::test]
async fn test_world_predict_with_mock() {
    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::new()));
        r
    });

    let state = WorldState::new("integration_world", "mock");
    let mut world = World::new(state, "mock", registry);

    let action = Action::Move {
        target: Position {
            x: 5.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 2.0,
    };
    let config = PredictionConfig::default();

    let prediction = world.predict(&action, &config).await.unwrap();
    assert_eq!(prediction.provider, "mock");

    // State should have advanced
    assert!(world.current_state().time.step > 0);
}

#[tokio::test]
async fn test_multi_provider_comparison() {
    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::new()));
        r.register(Box::new(MockProvider::with_name("mock2")));
        r
    });

    let state = WorldState::new("compare_world", "mock");
    let world = World::new(state, "mock", registry);

    let action = Action::Move {
        target: Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = PredictionConfig::default();

    let multi = world
        .predict_multi(&action, &["mock", "mock2"], &config)
        .await
        .unwrap();
    assert_eq!(multi.predictions.len(), 2);
    assert!(multi.agreement_score >= 0.0);
    assert!(multi.agreement_score <= 1.0);
}

#[test]
fn test_adapter_action_translator_surface_matches_helpers() {
    let cosmos = CosmosProvider::new(
        CosmosModel::Predict2_5,
        "cosmos-key",
        CosmosEndpoint::NimApi("https://example.invalid".to_string()),
    );
    let cosmos_action = Action::Move {
        target: Position {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
        speed: 0.5,
    };
    let cosmos_translation = cosmos.translate(&cosmos_action).unwrap();
    let cosmos_helper_translation = CosmosActionTranslator.translate(&cosmos_action).unwrap();
    assert_eq!(cosmos_translation.provider, "cosmos");
    assert_eq!(cosmos_translation.data, cosmos_helper_translation.data);
    assert_eq!(
        worldforge_core::provider::WorldModelProvider::supported_actions(&cosmos),
        ActionType::all()
    );
    assert_eq!(
        worldforge_core::provider::WorldModelProvider::supported_actions(&cosmos),
        CosmosActionTranslator.supported_actions()
    );

    let runway = RunwayProvider::new(RunwayModel::Gwm1Robotics, "runway-secret");
    let runway_action = Action::Move {
        target: Position {
            x: 4.0,
            y: 5.0,
            z: 6.0,
        },
        speed: 1.5,
    };
    let runway_translation = runway.translate(&runway_action).unwrap();
    let runway_helper_translation = RunwayActionTranslator.translate(&runway_action).unwrap();
    assert_eq!(runway_translation.provider, "runway");
    assert_eq!(runway_translation.data, runway_helper_translation.data);
    assert_eq!(
        worldforge_core::provider::WorldModelProvider::supported_actions(&runway),
        ActionType::all()
    );
    assert_eq!(
        worldforge_core::provider::WorldModelProvider::supported_actions(&runway),
        RunwayActionTranslator.supported_actions()
    );
}

#[tokio::test]
async fn test_provider_cost_estimation() {
    let mock = MockProvider::new();
    let cost = mock.estimate_cost(&worldforge_core::provider::Operation::Predict {
        steps: 10,
        resolution: (1280, 720),
    });
    // Mock provider should have zero or minimal cost
    assert!(cost.usd >= 0.0);
    assert_eq!(cost.estimated_latency_ms, mock.latency_ms);
}

#[tokio::test]
async fn test_cosmos_embed_roundtrip_with_fake_http_response() {
    let (endpoint, rx, handle) = spawn_fake_http_server(
        serde_json::json!({
            "request_id": "embed-1",
            "status": "ok",
            "model": "nvidia/cosmos-embed-1",
            "embedding": [0.1, 0.2, 0.3, 0.4]
        })
        .to_string(),
    );

    let provider = CosmosProvider::full_stack("cosmos-key", CosmosEndpoint::NimApi(endpoint));
    let output = provider
        .embed(&EmbeddingInput::from_text("a red cube on a table"))
        .await
        .unwrap();

    let request = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/embed");
    assert_eq!(
        request.headers.get("authorization").unwrap(),
        "Bearer cosmos-key"
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&request.body).unwrap()["model"],
        "nvidia/cosmos-embed-1"
    );
    assert_eq!(output.provider, "cosmos");
    assert_eq!(output.model, "nvidia/cosmos-embed-1");
    match output.embedding.data {
        TensorData::Float32(values) => assert_eq!(values, vec![0.1, 0.2, 0.3, 0.4]),
        other => panic!("unexpected embedding tensor: {other:?}"),
    }

    handle.join().unwrap();
}

#[tokio::test]
async fn test_mock_provider_native_plan_spawn_goal() {
    let mock = MockProvider::new();
    let request = PlanRequest {
        current_state: WorldState::new("native-plan", "mock"),
        goal: PlanGoal::Description("spawn cube".to_string()),
        max_steps: 4,
        guardrails: Vec::new(),
        planner: PlannerType::ProviderNative,
        timeout_seconds: 5.0,
        fallback_provider: None,
    };

    if mock_supports_native_planning() {
        let plan = mock.plan(&request).await.unwrap();
        assert!(!plan.actions.is_empty());
        assert_eq!(plan.predicted_states.len(), plan.actions.len());
        assert_eq!(plan.guardrail_compliance.len(), plan.actions.len());
        assert!((0.0..=1.0).contains(&plan.success_probability));
        assert!(plan.iterations_used > 0);
        return;
    }

    let error = mock.plan(&request).await.unwrap_err();
    assert!(matches!(
        error,
        WorldForgeError::UnsupportedCapability {
            provider,
            capability
        } if provider == "mock" && capability == "native planning"
    ));
}

#[tokio::test]
async fn test_mock_provider_native_plan_respects_blocking_guardrail() {
    let mock = MockProvider::new();
    let request = PlanRequest {
        current_state: WorldState::new("native-plan-guardrails", "mock"),
        goal: PlanGoal::Description("spawn cube".to_string()),
        max_steps: 4,
        guardrails: vec![GuardrailConfig {
            guardrail: Guardrail::BoundaryConstraint {
                bounds: BBox {
                    min: Position {
                        x: 100.0,
                        y: 100.0,
                        z: 100.0,
                    },
                    max: Position {
                        x: 101.0,
                        y: 101.0,
                        z: 101.0,
                    },
                },
            },
            blocking: true,
        }],
        planner: PlannerType::ProviderNative,
        timeout_seconds: 5.0,
        fallback_provider: None,
    };

    if mock_supports_native_planning() {
        let error = mock.plan(&request).await.unwrap_err();
        assert!(matches!(
            error,
            WorldForgeError::NoFeasiblePlan { .. } | WorldForgeError::GuardrailBlocked { .. }
        ));
        return;
    }

    let error = mock.plan(&request).await.unwrap_err();
    assert!(matches!(
        error,
        WorldForgeError::UnsupportedCapability {
            provider,
            capability
        } if provider == "mock" && capability == "native planning"
    ));
}

#[tokio::test]
async fn test_jepa_provider_predict_workflow() {
    let model_dir = TestModelDir::new("jepa-predict");
    model_dir.write_assets();

    let provider = JepaProvider::new(&model_dir.path, JepaBackend::Burn);
    let (state, object_id) = sample_jepa_state();
    let action = Action::Push {
        object: object_id,
        direction: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        force: 2.0,
    };
    let config = PredictionConfig {
        steps: 4,
        fps: 8.0,
        ..PredictionConfig::default()
    };

    let prediction = provider.predict(&state, &action, &config).await.unwrap();
    let before = state.scene.get_object(&object_id).unwrap().pose.position;
    let after = prediction
        .output_state
        .scene
        .get_object(&object_id)
        .unwrap()
        .pose
        .position;

    assert!(after.x > before.x);
    assert!(prediction.model.starts_with("vjepa2-local-burn-"));
    assert!(prediction.confidence > 0.4);
    assert!(prediction.physics_scores.overall > 0.4);
}

#[tokio::test]
async fn test_jepa_provider_reason_and_embed_workflow() {
    let model_dir = TestModelDir::new("jepa-reason-embed");
    model_dir.write_assets();

    let provider = JepaProvider::new(&model_dir.path, JepaBackend::Burn);
    let (state, _) = sample_jepa_state();

    let reasoning = provider
        .reason(
            &ReasoningInput {
                video: None,
                state: Some(state),
            },
            "Where is the crate?",
        )
        .await
        .unwrap();
    assert!(reasoning.answer.contains("crate"));
    assert!(reasoning
        .evidence
        .iter()
        .any(|entry| entry.starts_with("position:crate=")));

    let embedding = provider
        .embed(&EmbeddingInput::from_text("a crate on a table"))
        .await
        .unwrap();
    assert_eq!(embedding.provider, "jepa");
    assert_eq!(embedding.model, "vjepa2-local-burn-representation");
    assert_eq!(embedding.embedding.shape, vec![2048]);
}

#[tokio::test]
async fn test_world_predict_with_jepa() {
    let model_dir = TestModelDir::new("jepa-world");
    model_dir.write_assets();

    let registry = Arc::new({
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(MockProvider::new()));
        registry.register(Box::new(JepaProvider::new(
            &model_dir.path,
            JepaBackend::Burn,
        )));
        registry
    });

    let (state, object_id) = sample_jepa_state();
    let mut world = World::new(state, "jepa", registry);

    let action = Action::Release { object: object_id };
    let config = PredictionConfig {
        steps: 3,
        fps: 10.0,
        ..PredictionConfig::default()
    };

    let prediction = world.predict(&action, &config).await.unwrap();
    let updated = world.current_state().scene.get_object(&object_id).unwrap();
    let archived = world.current_state().history.latest().unwrap();

    assert_eq!(prediction.provider, "jepa");
    assert!(world.current_state().time.step > 0);
    assert!(updated.pose.position.y <= 1.0);
    assert_eq!(
        archived
            .prediction
            .as_ref()
            .and_then(|summary| summary.model.as_deref()),
        Some(prediction.model.as_str())
    );
}

#[tokio::test]
async fn test_genie_predict_flow_uses_discrete_actions() {
    let provider = GenieProvider::new(GenieModel::Genie3, "genie-test-key");
    let state = sample_genie_state();
    let action = sample_genie_action();
    let config = PredictionConfig {
        steps: 4,
        resolution: (1024, 768),
        fps: 12.0,
        return_video: true,
        return_depth: false,
        return_segmentation: false,
        ..PredictionConfig::default()
    };

    let prediction = provider.predict(&state, &action, &config).await.unwrap();

    assert_eq!(prediction.provider, "genie");
    assert!(prediction.model.to_lowercase().contains("genie"));
    assert_eq!(prediction.input_state.id, state.id);
    assert_eq!(
        prediction.output_state.time.step,
        state.time.step + u64::from(config.steps)
    );
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    assert!(prediction.physics_scores.overall >= 0.0 && prediction.physics_scores.overall <= 1.0);

    let video = prediction
        .video
        .as_ref()
        .expect("genie predict should return a video when requested");
    assert!(!video.frames.is_empty());
    assert!(video.resolution.0 <= 256);
    assert!(video.resolution.1 <= 256);
    assert!(video.fps >= 6.0 && video.fps <= 12.0);
    assert!(prediction
        .output_state
        .metadata
        .tags
        .iter()
        .any(|tag| tag == "weather:cloudy"));
}

#[tokio::test]
async fn test_genie_generate_flow_respects_genie_limits() {
    let provider = GenieProvider::new(GenieModel::Genie3, "genie-test-key");
    let prompt = sample_genie_prompt();
    let config = GenerationConfig {
        resolution: (1920, 1080),
        fps: 24.0,
        duration_seconds: 6.0,
        temperature: 0.8,
        seed: Some(42),
    };

    let clip = provider.generate(&prompt, &config).await.unwrap();

    assert!(!clip.frames.is_empty());
    assert!(clip.resolution.0 <= 256);
    assert!(clip.resolution.1 <= 256);
    assert!(clip.fps >= 6.0 && clip.fps <= 12.0);
    assert!(clip.duration > 0.0);
    assert!(clip.duration <= config.duration_seconds);
}

#[tokio::test]
async fn test_genie_reason_flow_is_grounded_in_state() {
    let provider = GenieProvider::new(GenieModel::Genie3, "genie-test-key");
    let state = sample_genie_reasoning_state();

    let reasoning = provider
        .reason(
            &ReasoningInput {
                video: None,
                state: Some(state),
            },
            "Where is the mug?",
        )
        .await
        .unwrap();

    let answer = reasoning.answer.to_lowercase();
    assert!(
        answer.contains("mug")
            || reasoning
                .evidence
                .iter()
                .any(|entry| entry.to_lowercase().contains("mug"))
    );
    assert!(
        answer.contains('(')
            || reasoning
                .evidence
                .iter()
                .any(|entry| entry.contains("position:"))
    );
    assert!(reasoning.confidence > 0.5);
}

#[tokio::test]
async fn test_marble_provider_registry_world_flow_when_available() {
    let registry = Arc::new(auto_detect());
    let Ok(descriptor) = registry.describe("marble") else {
        return;
    };

    let mut world = World::new(sample_marble_state(), "marble", Arc::clone(&registry));

    if descriptor.capabilities.predict {
        let prediction = world
            .predict_with_provider(
                &Action::Move {
                    target: Position {
                        x: 0.25,
                        y: 0.75,
                        z: 0.0,
                    },
                    speed: 0.5,
                },
                &PredictionConfig::default(),
                "marble",
            )
            .await
            .unwrap();

        assert_eq!(prediction.provider, "marble");
        assert_eq!(world.state.current_state_provider(), "marble");
        assert_eq!(world.state.history.latest().unwrap().provider, "marble");
        return;
    }

    if descriptor.capabilities.generate {
        let (provider_name, clip) = world
            .generate_with_provider_and_fallback(
                &GenerationPrompt {
                    text: "A small robot navigating a simple room".to_string(),
                    reference_image: None,
                    negative_prompt: None,
                },
                &GenerationConfig::default(),
                "marble",
                None,
            )
            .await
            .unwrap();

        assert_eq!(provider_name, "marble");
        assert!(!clip.frames.is_empty());
        return;
    }

    if descriptor.capabilities.reason {
        let (provider_name, reasoning) = world
            .reason_with_provider_and_fallback(
                "Describe the visible objects in the scene.",
                "marble",
                None,
            )
            .await
            .unwrap();

        assert_eq!(provider_name, "marble");
        assert!(!reasoning.answer.is_empty());
        return;
    }

    if descriptor.capabilities.transfer {
        let (provider_name, clip) = world
            .transfer_with_provider_and_fallback(
                &sample_video_clip(),
                &sample_genie_transfer_controls(),
                &TransferConfig::default(),
                "marble",
                None,
            )
            .await
            .unwrap();

        assert_eq!(provider_name, "marble");
        assert!(!clip.frames.is_empty());
    }
}

#[tokio::test]
async fn test_marble_native_planning_plans_target_state() {
    let provider = MarbleProvider::new();
    let state = sample_marble_state();
    let object_id = *state.scene.objects.keys().next().unwrap();
    let mut target = state.clone();
    target
        .scene
        .get_object_mut(&object_id)
        .unwrap()
        .set_position(Position {
            x: 0.4,
            y: 0.8,
            z: 0.0,
        });

    let request = PlanRequest {
        current_state: state,
        goal: PlanGoal::TargetState(Box::new(target)),
        max_steps: 4,
        guardrails: Vec::new(),
        planner: PlannerType::ProviderNative,
        timeout_seconds: 5.0,
        fallback_provider: None,
    };

    let plan = provider.plan(&request).await.unwrap();

    assert!(provider.capabilities().supports_planning);
    assert_eq!(plan.actions.len(), plan.predicted_states.len());
    assert!(!plan.actions.is_empty());
    assert!(matches!(
        plan.actions.first(),
        Some(Action::Place { object, .. }) if *object == object_id
    ));
    assert!(plan.success_probability >= 0.95);

    let final_state = plan.predicted_states.last().unwrap();
    let final_object = final_state.scene.get_object(&object_id).unwrap();
    assert_eq!(
        final_object.pose.position,
        Position {
            x: 0.4,
            y: 0.8,
            z: 0.0,
        }
    );
}

#[tokio::test]
async fn test_genie_transfer_flow_emits_controlled_video() {
    let provider = GenieProvider::new(GenieModel::Genie3, "genie-test-key");
    let source = provider
        .generate(
            &sample_genie_prompt(),
            &GenerationConfig {
                resolution: (320, 180),
                fps: 8.0,
                duration_seconds: 2.0,
                temperature: 0.7,
                seed: Some(11),
            },
        )
        .await
        .unwrap();

    let transferred = provider
        .transfer(
            &source,
            &SpatialControls {
                camera_trajectory: Some(Trajectory {
                    poses: vec![
                        (
                            SimTime {
                                step: 0,
                                seconds: 0.0,
                                dt: 0.1,
                            },
                            Pose {
                                position: Position {
                                    x: 0.0,
                                    y: 2.5,
                                    z: 4.0,
                                },
                                ..Pose::default()
                            },
                        ),
                        (
                            SimTime {
                                step: 1,
                                seconds: 1.0,
                                dt: 0.1,
                            },
                            Pose {
                                position: Position {
                                    x: 1.0,
                                    y: 2.0,
                                    z: 3.5,
                                },
                                ..Pose::default()
                            },
                        ),
                    ],
                    velocities: None,
                }),
                depth_map: Some(Tensor {
                    data: worldforge_core::types::TensorData::Float32(vec![0.25; 16]),
                    shape: vec![4, 4],
                    dtype: DType::Float32,
                    device: worldforge_core::types::Device::Cpu,
                }),
                segmentation_map: None,
            },
            &TransferConfig {
                resolution: (640, 480),
                fps: 24.0,
                control_strength: 0.85,
            },
        )
        .await
        .unwrap();

    assert_eq!(transferred.resolution, (256, 256));
    assert!((transferred.fps - 12.0).abs() < f32::EPSILON);
    assert_eq!(transferred.frames.len(), source.frames.len());
    assert_eq!(
        transferred.frames[0]
            .camera
            .as_ref()
            .unwrap()
            .extrinsics
            .position
            .x,
        0.0
    );
}

#[tokio::test]
async fn test_genie_native_plan_supports_goal_image() {
    let provider = GenieProvider::new(GenieModel::Genie3, "genie-test-key");
    let state = sample_genie_state();
    let mut target = state.clone();
    let object_id = *target.scene.objects.keys().next().unwrap();
    target
        .scene
        .get_object_mut(&object_id)
        .unwrap()
        .set_position(Position {
            x: 1.25,
            y: 0.8,
            z: 0.1,
        });
    let goal_image = worldforge_core::goal_image::render_scene_goal_image(&target, (32, 24));

    let plan = provider
        .plan(&PlanRequest {
            current_state: state,
            goal: PlanGoal::GoalImage(goal_image),
            max_steps: 2,
            guardrails: Vec::new(),
            planner: PlannerType::ProviderNative,
            timeout_seconds: 5.0,
            fallback_provider: None,
        })
        .await
        .unwrap();

    assert!(!plan.actions.is_empty());
    assert_eq!(plan.actions.len(), plan.predicted_states.len());
    assert!(plan.success_probability > 0.7);
}

#[test]
fn test_all_provider_names_unique() {
    let registry = auto_detect();
    let names = registry.list();
    let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
    assert_eq!(names.len(), unique.len(), "provider names must be unique");
}

#[test]
fn test_cosmos_physics_scores_helper_converts_canned_payload() {
    let scores = CosmosPhysicsScores {
        overall: Some(0.91),
        object_permanence: Some(0.82),
        gravity_compliance: Some(0.88),
        collision_accuracy: Some(0.77),
        spatial_consistency: Some(0.93),
        temporal_consistency: Some(0.84),
    };

    let converted = scores.to_physics_scores();
    assert!((converted.overall - 0.91).abs() < f32::EPSILON);
    assert!((converted.object_permanence - 0.82).abs() < f32::EPSILON);
    assert!((converted.gravity_compliance - 0.88).abs() < f32::EPSILON);
    assert!((converted.collision_accuracy - 0.77).abs() < f32::EPSILON);
    assert!((converted.spatial_consistency - 0.93).abs() < f32::EPSILON);
    assert!((converted.temporal_consistency - 0.84).abs() < f32::EPSILON);

    let response = CosmosPredictResponse {
        request_id: "cosmos-request-1".to_string(),
        status: "succeeded".to_string(),
        confidence: Some(0.97),
        physics_scores: Some(scores),
        processing_time_ms: Some(321),
    };
    let json = serde_json::to_string(&response).unwrap();
    let roundtrip: CosmosPredictResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(roundtrip.request_id, "cosmos-request-1");
    assert_eq!(roundtrip.status, "succeeded");
    assert_eq!(roundtrip.processing_time_ms, Some(321));
}

#[tokio::test]
async fn test_cosmos_predict_roundtrip_with_fake_http_response() {
    let response_body = serde_json::json!({
        "request_id": "cosmos-predict-1",
        "status": "succeeded",
        "confidence": 0.87,
        "physics_scores": {
            "overall": 0.94,
            "object_permanence": 0.91,
            "gravity_compliance": 0.92,
            "collision_accuracy": 0.90,
            "spatial_consistency": 0.95,
            "temporal_consistency": 0.93
        },
        "processing_time_ms": 312,
        "video_url": "https://example.invalid/cosmos-predict-1.mp4"
    })
    .to_string();
    let (endpoint, requests, server) = spawn_fake_http_server(response_body);

    let provider = CosmosProvider::with_config(
        CosmosModel::Predict2_5,
        "cosmos-key",
        CosmosEndpoint::NimApi(endpoint),
        CosmosConfig {
            include_depth: true,
            ..CosmosConfig::default()
        },
    )
    .unwrap();

    let mut state = WorldState::new("cosmos-world", "cosmos");
    state.scene.add_object(SceneObject::new(
        "mug",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.8,
                z: 0.0,
            },
            ..Default::default()
        },
        BBox {
            min: Position {
                x: -0.05,
                y: 0.75,
                z: -0.05,
            },
            max: Position {
                x: 0.05,
                y: 0.85,
                z: 0.05,
            },
        },
    ));

    let action = Action::Move {
        target: Position {
            x: 0.25,
            y: 0.8,
            z: 0.0,
        },
        speed: 1.5,
    };
    let config = PredictionConfig {
        steps: 3,
        resolution: (640, 360),
        fps: 6.0,
        return_video: true,
        return_depth: true,
        return_segmentation: true,
        ..PredictionConfig::default()
    };

    let prediction = provider.predict(&state, &action, &config).await.unwrap();
    let request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    server.join().unwrap();

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/predict");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer cosmos-key")
    );

    let request_body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    assert_eq!(request_body["model"], "nvidia/cosmos-predict-2.5");
    assert_eq!(request_body["num_frames"], 18);
    assert_eq!(request_body["include_depth"], true);

    assert_eq!(prediction.provider, "cosmos");
    assert_eq!(prediction.model, "nvidia/cosmos-predict-2.5");
    assert!((prediction.confidence - 0.87).abs() < 1e-6);
    assert!((prediction.physics_scores.overall - 0.94).abs() < 1e-6);
    assert_eq!(prediction.output_state.time.step, state.time.step + 3);
    let video = prediction
        .video
        .as_ref()
        .expect("cosmos predict should emit video");
    assert!(!video.frames.is_empty());
    assert_eq!(video.resolution, (640, 360));
    assert_eq!(video.fps, 6.0);
}

#[tokio::test]
async fn test_cosmos_reason_roundtrip_with_fake_http_response() {
    let response_body = serde_json::json!({
        "request_id": "cosmos-reason-1",
        "status": "completed",
        "answer": "The mug stays upright because the table supports it.",
        "confidence": 0.81,
        "evidence": ["mug supported by table"],
        "citations": ["frame-1"]
    })
    .to_string();
    let (endpoint, requests, server) = spawn_fake_http_server(response_body);

    let provider = CosmosProvider::new(
        CosmosModel::Reason2,
        "cosmos-key",
        CosmosEndpoint::NimApi(endpoint),
    );

    let mut state = WorldState::new("cosmos-reason-world", "cosmos");
    state.scene.add_object(SceneObject::new(
        "mug",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.8,
                z: 0.0,
            },
            ..Default::default()
        },
        BBox {
            min: Position {
                x: -0.05,
                y: 0.75,
                z: -0.05,
            },
            max: Position {
                x: 0.05,
                y: 0.85,
                z: 0.05,
            },
        },
    ));

    let input = worldforge_core::provider::ReasoningInput {
        video: None,
        state: Some(state),
    };
    let output = provider.reason(&input, "Will the mug fall?").await.unwrap();
    let request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    server.join().unwrap();

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/reason");

    let request_body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    assert_eq!(request_body["model"], "nvidia/cosmos-reason-2");
    assert_eq!(request_body["query"], "Will the mug fall?");
    assert_eq!(request_body["has_state"], true);

    assert_eq!(
        output.answer,
        "The mug stays upright because the table supports it."
    );
    assert!((output.confidence - 0.81).abs() < 1e-6);
    assert_eq!(
        output.evidence,
        vec!["mug supported by table".to_string(), "frame-1".to_string()]
    );
}

#[tokio::test]
async fn test_cosmos_full_stack_routes_predict_reason_and_transfer() {
    let predict_body = serde_json::json!({
        "request_id": "cosmos-full-stack-predict-1",
        "status": "succeeded",
        "confidence": 0.83,
        "physics_scores": {
            "overall": 0.91,
            "object_permanence": 0.88,
            "gravity_compliance": 0.87,
            "collision_accuracy": 0.86,
            "spatial_consistency": 0.9,
            "temporal_consistency": 0.89
        },
        "processing_time_ms": 290,
        "video_url": "https://example.invalid/cosmos-full-stack-predict.mp4"
    })
    .to_string();

    let reason_body = serde_json::json!({
        "request_id": "cosmos-full-stack-reason-1",
        "status": "completed",
        "answer": "The cube remains visible in the workspace.",
        "confidence": 0.8,
        "evidence": ["cube stays within the tabletop bounds"]
    })
    .to_string();

    let transfer_body = serde_json::json!({
        "video_url": "https://example.invalid/cosmos-full-stack-transfer.mp4",
        "duration": 2.0,
        "fps": 12.0,
        "resolution": [320, 240]
    })
    .to_string();

    let mut responses = HashMap::new();
    responses.insert("/v1/predict".to_string(), predict_body);
    responses.insert("/v1/reason".to_string(), reason_body);
    responses.insert("/v1/transfer".to_string(), transfer_body);

    let (endpoint, requests, server) = spawn_route_http_server(responses);
    let provider = CosmosProvider::full_stack("cosmos-key", CosmosEndpoint::NimApi(endpoint));

    let state = WorldState::new("cosmos-full-stack-world", "cosmos");
    let prediction = provider
        .predict(
            &state,
            &Action::Move {
                target: Position {
                    x: 0.5,
                    y: 0.8,
                    z: 0.0,
                },
                speed: 1.0,
            },
            &PredictionConfig {
                steps: 2,
                resolution: (320, 240),
                fps: 6.0,
                return_video: true,
                ..PredictionConfig::default()
            },
        )
        .await
        .unwrap();
    let predict_request = requests.recv_timeout(Duration::from_secs(1)).unwrap();

    let reasoning = provider
        .reason(
            &ReasoningInput {
                video: None,
                state: Some(state.clone()),
            },
            "Will the cube remain in bounds?",
        )
        .await
        .unwrap();
    let reason_request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    let transferred = provider
        .transfer(
            &sample_video_clip(),
            &SpatialControls::default(),
            &TransferConfig {
                resolution: (320, 240),
                fps: 12.0,
                control_strength: 0.5,
            },
        )
        .await
        .unwrap();
    let transfer_request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    server.join().unwrap();

    assert_eq!(predict_request.path, "/v1/predict");
    assert_eq!(reason_request.path, "/v1/reason");
    assert_eq!(transfer_request.path, "/v1/transfer");
    assert_eq!(prediction.provider, "cosmos");
    assert_eq!(prediction.model, "nvidia/cosmos-predict-2.5");
    assert_eq!(
        reasoning.answer,
        "The cube remains visible in the workspace."
    );
    assert_eq!(transferred.resolution, (320, 240));
}

#[tokio::test]
async fn test_cosmos_nim_local_executes_without_http_transport() {
    let provider = CosmosProvider::full_stack(
        "cosmos-key",
        CosmosEndpoint::NimLocal("http://127.0.0.1:9".to_string()),
    );

    let mut state = WorldState::new("cosmos-nim-local-world", "cosmos");
    state.scene.add_object(SceneObject::new(
        "mug",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.8,
                z: 0.0,
            },
            ..Default::default()
        },
        BBox {
            min: Position {
                x: -0.05,
                y: 0.75,
                z: -0.05,
            },
            max: Position {
                x: 0.05,
                y: 0.85,
                z: 0.05,
            },
        },
    ));

    let prediction = provider
        .predict(
            &state,
            &Action::Move {
                target: Position {
                    x: 0.25,
                    y: 0.8,
                    z: 0.0,
                },
                speed: 1.0,
            },
            &PredictionConfig {
                steps: 3,
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

    let generated = provider
        .generate(
            &GenerationPrompt {
                text: "a mug sliding across a tabletop".to_string(),
                reference_image: None,
                negative_prompt: None,
            },
            &GenerationConfig {
                resolution: (640, 360),
                fps: 12.0,
                duration_seconds: 2.0,
                temperature: 0.7,
                seed: Some(11),
            },
        )
        .await
        .unwrap();

    let reasoning = provider
        .reason(
            &ReasoningInput {
                video: None,
                state: Some(state.clone()),
            },
            "Will the mug stay upright?",
        )
        .await
        .unwrap();

    let embedded = provider
        .embed(&EmbeddingInput::from_text("a mug on a tabletop"))
        .await
        .unwrap();

    let transferred = provider
        .transfer(
            &sample_video_clip(),
            &SpatialControls::default(),
            &TransferConfig {
                resolution: (640, 360),
                fps: 12.0,
                control_strength: 0.5,
            },
        )
        .await
        .unwrap();

    let health = provider.health_check().await.unwrap();

    assert_eq!(prediction.provider, "cosmos");
    assert!(prediction.confidence >= 0.0);
    assert!(!prediction.output_state.scene.objects.is_empty());
    assert!(prediction.video.is_some());
    assert!(!generated.frames.is_empty());
    assert_eq!(generated.resolution, (640, 360));
    assert_eq!(generated.fps, 12.0);
    assert!(!reasoning.answer.is_empty());
    assert!(!reasoning.evidence.is_empty());
    assert_eq!(embedded.provider, "cosmos");
    assert!(!embedded.embedding.shape.is_empty());
    assert!(!transferred.frames.is_empty());
    assert_eq!(transferred.resolution, (640, 360));
    assert!(health.healthy);
}

#[tokio::test]
async fn test_cosmos_huggingface_executes_without_http_transport() {
    let provider = CosmosProvider::full_stack("cosmos-key", CosmosEndpoint::HuggingFace);

    let state = WorldState::new("cosmos-hf-world", "cosmos");

    let prediction = provider
        .predict(
            &state,
            &Action::SetLighting { time_of_day: 18.0 },
            &PredictionConfig {
                steps: 2,
                resolution: (320, 180),
                fps: 8.0,
                return_video: true,
                ..PredictionConfig::default()
            },
        )
        .await
        .unwrap();

    let generated = provider
        .generate(
            &GenerationPrompt {
                text: "sunset over a kitchen counter".to_string(),
                reference_image: None,
                negative_prompt: Some("blurry".to_string()),
            },
            &GenerationConfig {
                resolution: (320, 180),
                fps: 8.0,
                duration_seconds: 1.5,
                temperature: 0.5,
                seed: None,
            },
        )
        .await
        .unwrap();

    let reasoning = provider
        .reason(
            &ReasoningInput {
                video: None,
                state: Some(state.clone()),
            },
            "Is the scene stable?",
        )
        .await
        .unwrap();

    let embedded = provider
        .embed(&EmbeddingInput::from_text("a kitchen counter at sunset"))
        .await
        .unwrap();

    let transferred = provider
        .transfer(
            &sample_video_clip(),
            &SpatialControls::default(),
            &TransferConfig {
                resolution: (320, 180),
                fps: 8.0,
                control_strength: 0.75,
            },
        )
        .await
        .unwrap();

    let health = provider.health_check().await.unwrap();

    assert_eq!(prediction.provider, "cosmos");
    assert!(prediction.confidence >= 0.0);
    assert!(prediction.video.is_some());
    assert!(!generated.frames.is_empty());
    assert_eq!(generated.resolution, (320, 180));
    assert_eq!(generated.fps, 8.0);
    assert!(!reasoning.answer.is_empty());
    assert!(!reasoning.evidence.is_empty());
    assert_eq!(embedded.provider, "cosmos");
    assert!(!embedded.embedding.shape.is_empty());
    assert!(!transferred.frames.is_empty());
    assert_eq!(transferred.resolution, (320, 180));
    assert!(health.healthy);
}

#[tokio::test]
async fn test_runway_predict_roundtrip_with_fake_http_response() {
    let response_body = serde_json::json!({
        "request_id": "runway-predict-1",
        "status": "ok",
        "confidence": 0.72,
        "physics_scores": {
            "overall": 0.76,
            "object_permanence": 0.75,
            "gravity_compliance": 0.80,
            "collision_accuracy": 0.77,
            "spatial_consistency": 0.78,
            "temporal_consistency": 0.79
        },
        "processing_time_ms": 410,
        "video_url": "https://example.invalid/runway-predict-1.mp4"
    })
    .to_string();
    let (endpoint, requests, server) = spawn_fake_http_server(response_body);

    let provider =
        RunwayProvider::with_endpoint(RunwayModel::Gwm1Robotics, "runway-secret", endpoint);

    let state = WorldState::new("runway-world", "runway");
    let action = Action::Push {
        object: uuid::Uuid::new_v4(),
        direction: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        force: 2.5,
    };
    let config = PredictionConfig {
        steps: 4,
        resolution: (1280, 720),
        return_video: true,
        fps: 8.0,
        ..PredictionConfig::default()
    };

    let prediction = provider.predict(&state, &action, &config).await.unwrap();
    let request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    server.join().unwrap();

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/robotics/predict");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some("Bearer runway-secret")
    );

    let request_body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    assert_eq!(request_body["model"], "gwm-1-robotics");
    assert_eq!(request_body["num_frames"], 32);
    assert_eq!(prediction.provider, "runway");
    assert_eq!(prediction.model, "gwm-1-robotics");
    assert_eq!(prediction.output_state.time.step, state.time.step + 4);
    let video = prediction
        .video
        .as_ref()
        .expect("runway predict should emit video");
    assert!(!video.frames.is_empty());
    assert_eq!(video.resolution, (1280, 720));
    assert_eq!(video.fps, 8.0);
}

#[tokio::test]
async fn test_runway_generate_and_transfer_roundtrip_with_fake_http_response() {
    let generate_body = serde_json::json!({
        "video_url": "https://example.invalid/runway-generate-1.mp4",
        "duration": 4.0,
        "fps": 12.0,
        "resolution": [640, 360]
    })
    .to_string();
    let (generate_endpoint, generate_requests, generate_server) =
        spawn_fake_http_server(generate_body);

    let generate_provider =
        RunwayProvider::with_endpoint(RunwayModel::Gwm1Worlds, "runway-secret", generate_endpoint);

    let prompt = worldforge_core::provider::GenerationPrompt {
        text: "a cube rolling across a table".to_string(),
        reference_image: None,
        negative_prompt: Some("blurry".to_string()),
    };
    let generation_config = worldforge_core::provider::GenerationConfig {
        resolution: (640, 360),
        fps: 12.0,
        duration_seconds: 4.0,
        temperature: 0.8,
        seed: Some(7),
    };

    let generated = generate_provider
        .generate(&prompt, &generation_config)
        .await
        .unwrap();
    let generate_request = generate_requests
        .recv_timeout(Duration::from_secs(1))
        .unwrap();
    generate_server.join().unwrap();

    assert_eq!(generate_request.method, "POST");
    assert_eq!(generate_request.path, "/v1/worlds/generate");

    let generate_body: serde_json::Value = serde_json::from_str(&generate_request.body).unwrap();
    assert_eq!(generate_body["model"], "gwm-1-worlds");
    assert_eq!(generate_body["prompt"], "a cube rolling across a table");
    assert_eq!(generated.fps, 12.0);
    assert_eq!(generated.resolution, (640, 360));
    assert_eq!(generated.duration, 4.0);
    assert!(!generated.frames.is_empty());

    let transfer_body = serde_json::json!({
        "video_url": "https://example.invalid/runway-transfer-1.mp4",
        "duration": 4.0,
        "fps": 24.0,
        "resolution": [1280, 720]
    })
    .to_string();
    let (transfer_endpoint, transfer_requests, transfer_server) =
        spawn_fake_http_server(transfer_body);

    let transfer_provider =
        RunwayProvider::with_endpoint(RunwayModel::Gwm1Worlds, "runway-secret", transfer_endpoint);
    let source = sample_video_clip();
    let transfer_config = worldforge_core::provider::TransferConfig {
        resolution: (1280, 720),
        fps: 24.0,
        control_strength: 0.75,
    };

    let transferred = transfer_provider
        .transfer(
            &source,
            &worldforge_core::provider::SpatialControls::default(),
            &transfer_config,
        )
        .await
        .unwrap();
    let transfer_request = transfer_requests
        .recv_timeout(Duration::from_secs(1))
        .unwrap();
    transfer_server.join().unwrap();

    assert_eq!(transfer_request.method, "POST");
    assert_eq!(transfer_request.path, "/v1/worlds/transfer");

    let transfer_body: serde_json::Value = serde_json::from_str(&transfer_request.body).unwrap();
    assert_eq!(transfer_body["model"], "gwm-1-worlds");
    assert_eq!(transfer_body["resolution"], serde_json::json!([1280, 720]));
    assert_eq!(transferred.fps, 24.0);
    assert_eq!(transferred.resolution, (1280, 720));
    assert!(!transferred.frames.is_empty());
}
