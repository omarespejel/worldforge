//! End-to-end integration tests for worldforge-server.
//!
//! Tests the full REST API workflow: create world → predict →
//! list → show → delete, plus evaluation and comparison endpoints.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::task::JoinHandle;

use worldforge_core::provider::ProviderRegistry;
use worldforge_core::state::FileStateStore;
use worldforge_providers::MockProvider;
use worldforge_server::{Server, ServerConfig};

/// Helper to create a server config with a unique temp directory.
fn test_server_config() -> (FileStateStore, Arc<ProviderRegistry>) {
    let dir = std::env::temp_dir().join(format!("wf-integ-{}", uuid::Uuid::new_v4()));
    let store = FileStateStore::new(&dir);
    let mut registry = ProviderRegistry::new();
    registry.register(Box::new(MockProvider::new()));
    (store, Arc::new(registry))
}

struct TestServer {
    address: SocketAddr,
    state_dir: PathBuf,
    task: JoinHandle<anyhow::Result<()>>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
        let _ = std::fs::remove_dir_all(&self.state_dir);
    }
}

async fn spawn_test_server() -> TestServer {
    let state_dir = std::env::temp_dir().join(format!("wf-http-{}", uuid::Uuid::new_v4()));
    let mut registry = ProviderRegistry::new();
    registry.register(Box::new(MockProvider::new()));

    let server = Server::bind(
        ServerConfig {
            bind_address: "127.0.0.1:0".to_string(),
            state_dir: state_dir.display().to_string(),
            ..ServerConfig::default()
        },
        Arc::new(registry),
    )
    .await
    .unwrap();
    let address = server.local_addr().unwrap();
    let task = tokio::spawn(server.run());

    TestServer {
        address,
        state_dir,
        task,
    }
}

async fn spawn_test_server_sqlite() -> TestServer {
    let state_dir = std::env::temp_dir().join(format!("wf-http-sqlite-{}", uuid::Uuid::new_v4()));
    let mut registry = ProviderRegistry::new();
    registry.register(Box::new(MockProvider::new()));

    let server = Server::bind(
        ServerConfig {
            bind_address: "127.0.0.1:0".to_string(),
            state_dir: state_dir.display().to_string(),
            state_backend: "sqlite".to_string(),
            state_db_path: Some(state_dir.join("worldforge.db").display().to_string()),
        },
        Arc::new(registry),
    )
    .await
    .unwrap();
    let address = server.local_addr().unwrap();
    let task = tokio::spawn(server.run());

    TestServer {
        address,
        state_dir,
        task,
    }
}

async fn http_request(address: SocketAddr, method: &str, path: &str, body: &str) -> (u16, Value) {
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let response = raw_http_request(address, &request).await;
    let (status, response_body) = parse_http_response(&response);
    let json = serde_json::from_str(&response_body).unwrap();
    (status, json)
}

async fn raw_http_request(address: SocketAddr, request: &str) -> String {
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    response
}

fn parse_http_response(response: &str) -> (u16, String) {
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap()
        .parse()
        .unwrap();
    (status, body.to_string())
}

#[tokio::test]
async fn test_live_http_world_lifecycle() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"tcp_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let (status, list) = http_request(server.address, "GET", "/v1/worlds", "").await;
    assert_eq!(status, 200);
    assert!(list["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["id"] == world_id));

    let predict_body =
        r#"{"action":{"Move":{"target":{"x":1.0,"y":0.0,"z":0.0},"speed":1.0}},"provider":"mock"}"#;
    let (status, prediction) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/predict"),
        predict_body,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(prediction["data"]["provider"], "mock");

    let (status, world) =
        http_request(server.address, "GET", &format!("/v1/worlds/{world_id}"), "").await;
    assert_eq!(status, 200);
    assert_eq!(world["data"]["time"]["step"], 1);
    assert_eq!(
        world["data"]["history"]["states"].as_array().unwrap().len(),
        1
    );

    let (status, deleted) = http_request(
        server.address,
        "DELETE",
        &format!("/v1/worlds/{world_id}"),
        "",
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(deleted["data"]["deleted"], world_id);

    let (status, missing) =
        http_request(server.address, "GET", &format!("/v1/worlds/{world_id}"), "").await;
    assert_eq!(status, 404);
    assert_eq!(missing["success"], false);
}

#[tokio::test]
async fn test_live_http_world_lifecycle_sqlite() {
    let server = spawn_test_server_sqlite().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"sqlite_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let predict_body =
        r#"{"action":{"Move":{"target":{"x":1.0,"y":0.0,"z":0.0},"speed":1.0}},"provider":"mock"}"#;
    let (status, prediction) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/predict"),
        predict_body,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(prediction["data"]["provider"], "mock");

    let (status, world) =
        http_request(server.address, "GET", &format!("/v1/worlds/{world_id}"), "").await;
    assert_eq!(status, 200);
    assert_eq!(world["data"]["time"]["step"], 1);

    let (status, list) = http_request(server.address, "GET", "/v1/worlds", "").await;
    assert_eq!(status, 200);
    assert!(list["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["id"] == world_id));
}

#[tokio::test]
async fn test_live_http_predict_uses_fallback_provider() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"fallback_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let predict_body = r#"{"action":{"Move":{"target":{"x":1.0,"y":0.0,"z":0.0},"speed":1.0}},"provider":"missing","config":{"fallback_provider":"mock"}}"#;
    let (status, prediction) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/predict"),
        predict_body,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(prediction["data"]["provider"], "mock");
}

#[tokio::test]
async fn test_live_http_plan_uses_requested_planner() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"plan_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let plan_body = r#"{
        "goal":"spawn cube",
        "provider":"mock",
        "planner":"cem",
        "population_size":12,
        "elite_fraction":0.25,
        "num_iterations":3
    }"#;
    let (status, plan) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/plan"),
        plan_body,
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(plan["data"]["iterations_used"], 3);
    assert!(!plan["data"]["actions"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_live_http_rejects_oversized_body() {
    let server = spawn_test_server().await;
    let request = "POST /v1/worlds HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 4194305\r\nConnection: close\r\n\r\n";

    let response = raw_http_request(server.address, request).await;
    let (status, body) = parse_http_response(&response);
    let payload: Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, 413);
    assert_eq!(payload["success"], false);
    assert_eq!(payload["error"], "request body too large");
}

#[tokio::test]
async fn test_full_world_lifecycle() {
    let (store, _registry) = test_server_config();
    use worldforge_core::state::StateStore;

    // Create
    let state = worldforge_core::state::WorldState::new("lifecycle_test", "mock");
    let world_id = state.id;
    store.save(&state).await.unwrap();

    // Load
    let loaded = store.load(&world_id).await.unwrap();
    assert_eq!(loaded.metadata.name, "lifecycle_test");

    // List
    let ids = store.list().await.unwrap();
    assert!(ids.contains(&world_id));

    // Delete
    store.delete(&world_id).await.unwrap();
    assert!(store.load(&world_id).await.is_err());
}

#[tokio::test]
async fn test_prediction_updates_state() {
    let (_store, registry) = test_server_config();

    let state = worldforge_core::state::WorldState::new("pred_test", "mock");
    let mut world = worldforge_core::world::World::new(state, "mock", registry);

    let action = worldforge_core::action::Action::Move {
        target: worldforge_core::types::Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = worldforge_core::prediction::PredictionConfig::default();

    let prediction = world.predict(&action, &config).await.unwrap();
    assert_eq!(prediction.provider, "mock");

    // State should advance
    let new_state = world.current_state();
    assert!(new_state.time.step > 0);
}

#[tokio::test]
async fn test_eval_suite_via_providers() {
    let mock = MockProvider::new();
    let providers: Vec<&dyn worldforge_core::provider::WorldModelProvider> = vec![&mock];

    let suite = worldforge_eval::EvalSuite::physics_standard();
    let report = suite.run(&providers).await.unwrap();

    assert!(!report.leaderboard.is_empty());
    assert!(!report.results.is_empty());
    assert_eq!(report.leaderboard[0].provider, "mock");
}

#[tokio::test]
async fn test_multiple_worlds_persistence() {
    let (store, _registry) = test_server_config();
    use worldforge_core::state::StateStore;

    let ids: Vec<uuid::Uuid> = (0..5)
        .map(|i| {
            let state = worldforge_core::state::WorldState::new(format!("world_{i}"), "mock");
            state.id
        })
        .collect();

    // Save all worlds
    for (i, id) in ids.iter().enumerate() {
        let mut state = worldforge_core::state::WorldState::new(format!("world_{i}"), "mock");
        // Override the auto-generated ID so we can track it
        state.id = *id;
        store.save(&state).await.unwrap();
    }

    // List should contain all
    let listed = store.list().await.unwrap();
    for id in &ids {
        assert!(listed.contains(id), "world {id} should be in list");
    }

    // Delete odd-indexed worlds
    for (i, id) in ids.iter().enumerate() {
        if i % 2 == 1 {
            store.delete(id).await.unwrap();
        }
    }

    // Verify remaining
    let remaining = store.list().await.unwrap();
    assert_eq!(remaining.len(), 3);
    for (i, id) in ids.iter().enumerate() {
        if i % 2 == 0 {
            assert!(remaining.contains(id));
        } else {
            assert!(!remaining.contains(id));
        }
    }
}

#[tokio::test]
async fn test_guardrail_evaluation_in_prediction() {
    let (_store, registry) = test_server_config();

    let mut state = worldforge_core::state::WorldState::new("guardrail_test", "mock");

    // Add an object
    let obj = worldforge_core::scene::SceneObject::new(
        "ball",
        worldforge_core::types::Pose::default(),
        worldforge_core::types::BBox {
            min: worldforge_core::types::Position {
                x: -0.5,
                y: -0.5,
                z: -0.5,
            },
            max: worldforge_core::types::Position {
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
        },
    );
    state.scene.add_object(obj);

    let mut world = worldforge_core::world::World::new(state, "mock", registry);

    let action = worldforge_core::action::Action::Move {
        target: worldforge_core::types::Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };

    let config = worldforge_core::prediction::PredictionConfig {
        guardrails: vec![worldforge_core::guardrail::GuardrailConfig {
            guardrail: worldforge_core::guardrail::Guardrail::MaxVelocity { limit: 100.0 },
            blocking: false,
        }],
        ..worldforge_core::prediction::PredictionConfig::default()
    };

    // Prediction should succeed since MaxVelocity limit is high (100.0)
    let prediction = world.predict(&action, &config).await.unwrap();
    assert_eq!(prediction.provider, "mock");
    // State should have advanced even with guardrails configured
    assert!(world.current_state().time.step > 0);
}

#[tokio::test]
async fn test_verify_proof_roundtrip() {
    use worldforge_verify::{MockVerifier, ZkVerifier};

    let verifier = MockVerifier::new();
    let proof = verifier.prove_inference([1; 32], [2; 32], [3; 32]).unwrap();

    // Serialize and deserialize
    let json = serde_json::to_string(&proof).unwrap();
    let restored: worldforge_verify::ZkProof = serde_json::from_str(&json).unwrap();

    // Verify the restored proof
    let result = verifier.verify(&restored).unwrap();
    assert!(result.valid);
}

// ---------------------------------------------------------------------------
// End-to-end pipeline tests
// ---------------------------------------------------------------------------

/// Full pipeline: create world → add objects → plan → verify plan
#[tokio::test]
async fn test_e2e_plan_and_verify_pipeline() {
    use worldforge_core::prediction::{PlanGoal, PlanRequest, PlannerType};
    use worldforge_verify::{MockVerifier, ZkVerifier};

    let (_store, registry) = test_server_config();

    // 1. Create world with objects
    let mut state = worldforge_core::state::WorldState::new("e2e_plan_verify", "mock");
    let ball = worldforge_core::scene::SceneObject::new(
        "ball",
        worldforge_core::types::Pose {
            position: worldforge_core::types::Position {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            rotation: worldforge_core::types::Rotation::default(),
        },
        worldforge_core::types::BBox {
            min: worldforge_core::types::Position {
                x: -0.5,
                y: 0.5,
                z: -0.5,
            },
            max: worldforge_core::types::Position {
                x: 0.5,
                y: 1.5,
                z: 0.5,
            },
        },
    );
    state.scene.add_object(ball);

    // 2. Plan
    let world = worldforge_core::world::World::new(state.clone(), "mock", registry);
    let plan_request = PlanRequest {
        current_state: state.clone(),
        goal: PlanGoal::Description("move ball to position (2, 1, 0)".to_string()),
        max_steps: 5,
        guardrails: Vec::new(),
        planner: PlannerType::Sampling {
            num_samples: 16,
            top_k: 3,
        },
        timeout_seconds: 10.0,
    };

    let plan = world.plan(&plan_request).await.unwrap();
    assert!(!plan.actions.is_empty());
    assert!(plan.success_probability >= 0.0);

    // 3. Generate ZK proofs for the plan
    let verifier = MockVerifier::new();

    // 3a. Inference verification proof
    let model_hash = worldforge_verify::sha256_hash(b"mock-model");
    let input_hash = worldforge_verify::sha256_hash(&serde_json::to_vec(&state).unwrap());
    let output_hash = worldforge_verify::sha256_hash(
        &serde_json::to_vec(&plan.predicted_states.last().unwrap_or(&state)).unwrap(),
    );
    let inference_proof = verifier
        .prove_inference(model_hash, input_hash, output_hash)
        .unwrap();
    let inference_result = verifier.verify(&inference_proof).unwrap();
    assert!(inference_result.valid);

    // 3b. Guardrail compliance proof
    let guardrail_proof = verifier
        .prove_guardrail_compliance(&plan, &plan.guardrail_compliance)
        .unwrap();
    let guardrail_result = verifier.verify(&guardrail_proof).unwrap();
    assert!(guardrail_result.valid);

    // 3c. Data provenance proof
    let data_hash = worldforge_verify::sha256_hash(&serde_json::to_vec(&state).unwrap());
    let provenance_proof = verifier
        .prove_data_provenance(
            data_hash,
            1710000000,
            worldforge_verify::sha256_hash(b"test"),
        )
        .unwrap();
    let provenance_result = verifier.verify(&provenance_proof).unwrap();
    assert!(provenance_result.valid);
}

/// Full pipeline: create → predict → predict again → verify multi-step state evolution
#[tokio::test]
async fn test_e2e_multi_step_state_evolution() {
    let (_store, registry) = test_server_config();

    let state = worldforge_core::state::WorldState::new("multi_step", "mock");
    let mut world = worldforge_core::world::World::new(state, "mock", registry);

    let config = worldforge_core::prediction::PredictionConfig::default();

    // Apply 3 sequential predictions
    let actions = vec![
        worldforge_core::action::Action::Move {
            target: worldforge_core::types::Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            speed: 1.0,
        },
        worldforge_core::action::Action::SetWeather {
            weather: worldforge_core::action::Weather::Rain,
        },
        worldforge_core::action::Action::Move {
            target: worldforge_core::types::Position {
                x: 2.0,
                y: 0.0,
                z: 0.0,
            },
            speed: 1.5,
        },
    ];

    for action in &actions {
        let prediction = world.predict(action, &config).await.unwrap();
        assert_eq!(prediction.provider, "mock");
    }

    // State should have advanced 3 steps
    let final_state = world.current_state();
    assert_eq!(final_state.time.step, 3);
    assert!(final_state.history.len() >= 3);
}

/// Cross-provider comparison pipeline
#[tokio::test]
async fn test_e2e_cross_provider_comparison() {
    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::new()));
        r.register(Box::new(MockProvider::with_name("mock-2")));
        r
    });

    let state = worldforge_core::state::WorldState::new("comparison", "mock");
    let world = worldforge_core::world::World::new(state, "mock", registry);

    let action = worldforge_core::action::Action::Move {
        target: worldforge_core::types::Position {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        speed: 1.0,
    };
    let config = worldforge_core::prediction::PredictionConfig::default();

    let multi = world
        .predict_multi(&action, &["mock", "mock-2"], &config)
        .await
        .unwrap();

    assert_eq!(multi.predictions.len(), 2);
    assert!(multi.agreement_score >= 0.0);
    assert!(multi.agreement_score <= 1.0);
    assert!(multi.comparison.scores.len() == 2);
}

/// Evaluation with leaderboard generation
#[tokio::test]
async fn test_e2e_evaluation_all_suites() {
    let mock = MockProvider::new();
    let providers: Vec<&dyn worldforge_core::provider::WorldModelProvider> = vec![&mock];

    for suite_name in &["physics", "manipulation", "spatial", "comprehensive"] {
        let suite = match *suite_name {
            "physics" => worldforge_eval::EvalSuite::physics_standard(),
            "manipulation" => worldforge_eval::EvalSuite::manipulation_standard(),
            "spatial" => worldforge_eval::EvalSuite::spatial_reasoning(),
            "comprehensive" => worldforge_eval::EvalSuite::comprehensive(),
            _ => unreachable!(),
        };

        let report = suite.run(&providers).await.unwrap();
        assert!(
            !report.leaderboard.is_empty(),
            "{suite_name} leaderboard empty"
        );
        assert!(!report.results.is_empty(), "{suite_name} results empty");

        // Verify leaderboard has valid scores
        for entry in &report.leaderboard {
            assert!(entry.average_score >= 0.0);
            assert!(entry.total_scenarios > 0);
        }
    }
}

/// Verify that all three ZK proof types serialize/deserialize correctly through the pipeline
#[tokio::test]
async fn test_e2e_zk_proof_types_serialization() {
    use worldforge_verify::{MockVerifier, ZkVerifier};

    let verifier = MockVerifier::new();

    // Inference proof
    let inference_proof = verifier.prove_inference([1; 32], [2; 32], [3; 32]).unwrap();
    let json = serde_json::to_string(&inference_proof).unwrap();
    let restored: worldforge_verify::ZkProof = serde_json::from_str(&json).unwrap();
    assert!(verifier.verify(&restored).unwrap().valid);

    // Guardrail proof
    let plan = worldforge_core::prediction::Plan {
        actions: vec![worldforge_core::action::Action::Move {
            target: worldforge_core::types::Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            speed: 1.0,
        }],
        predicted_states: Vec::new(),
        predicted_videos: None,
        total_cost: 0.0,
        success_probability: 0.9,
        guardrail_compliance: Vec::new(),
        planning_time_ms: 100,
        iterations_used: 5,
    };
    let guardrail_proof = verifier.prove_guardrail_compliance(&plan, &[]).unwrap();
    let json = serde_json::to_string(&guardrail_proof).unwrap();
    let restored: worldforge_verify::ZkProof = serde_json::from_str(&json).unwrap();
    assert!(verifier.verify(&restored).unwrap().valid);

    // Provenance proof
    let provenance_proof = verifier
        .prove_data_provenance([4; 32], 1710000000, [5; 32])
        .unwrap();
    let json = serde_json::to_string(&provenance_proof).unwrap();
    let restored: worldforge_verify::ZkProof = serde_json::from_str(&json).unwrap();
    assert!(verifier.verify(&restored).unwrap().valid);
}
