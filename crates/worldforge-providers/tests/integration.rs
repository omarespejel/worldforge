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
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use worldforge_core::action::Action;
use worldforge_core::error::WorldForgeError;
use worldforge_core::guardrail::{Guardrail, GuardrailConfig};
use worldforge_core::prediction::{PlanGoal, PlanRequest, PlannerType, PredictionConfig};
use worldforge_core::provider::{ProviderRegistry, WorldModelProvider};
use worldforge_core::scene::SceneObject;
use worldforge_core::state::WorldState;
use worldforge_core::types::{BBox, DType, Pose, Position, SimTime, Tensor, Vec3, VideoClip};
use worldforge_core::world::World;
use worldforge_providers::cosmos::{
    CosmosConfig, CosmosEndpoint, CosmosModel, CosmosPhysicsScores, CosmosPredictResponse,
};
use worldforge_providers::runway::RunwayModel;
use worldforge_providers::{
    auto_detect, CosmosProvider, JepaBackend, JepaProvider, MockProvider, RunwayProvider,
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

#[test]
fn test_auto_detect_registry_mock_present() {
    let registry = auto_detect();
    assert!(registry.get("mock").is_ok());
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

    let predictors = registry.find_by_capability("predict");
    assert_eq!(predictors.len(), 1);
    assert_eq!(predictors[0].name(), "mock");

    let planners = registry.find_by_capability("planning");
    // MockProvider may or may not support planning
    assert!(planners.len() <= 1);
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
async fn test_mock_provider_native_plan_spawn_goal() {
    let mock = MockProvider::new();
    let request = PlanRequest {
        current_state: WorldState::new("native-plan", "mock"),
        goal: PlanGoal::Description("spawn cube".to_string()),
        max_steps: 4,
        guardrails: Vec::new(),
        planner: PlannerType::ProviderNative,
        timeout_seconds: 5.0,
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
    assert!(prediction.confidence > 0.4);
    assert!(prediction.physics_scores.overall > 0.4);
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

    assert_eq!(prediction.provider, "jepa");
    assert!(world.current_state().time.step > 0);
    assert!(updated.pose.position.y <= 1.0);
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
