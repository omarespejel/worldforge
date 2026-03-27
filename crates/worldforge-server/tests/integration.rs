//! End-to-end integration tests for worldforge-server.
//!
//! Tests the full REST API workflow: create world → predict →
//! list → show → delete, plus evaluation and comparison endpoints.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::task::JoinHandle;

use worldforge_core::provider::ProviderRegistry;
use worldforge_core::state::{FileStateStore, StateStore};
use worldforge_providers::genie::GenieModel;
use worldforge_providers::{GenieProvider, MockProvider};
use worldforge_server::{Server, ServerConfig};
use worldforge_verify::sha256_hash;

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

async fn spawn_test_server_msgpack() -> TestServer {
    let state_dir = std::env::temp_dir().join(format!("wf-http-msgpack-{}", uuid::Uuid::new_v4()));
    let mut registry = ProviderRegistry::new();
    registry.register(Box::new(MockProvider::new()));

    let server = Server::bind(
        ServerConfig {
            bind_address: "127.0.0.1:0".to_string(),
            state_dir: state_dir.display().to_string(),
            state_file_format: "msgpack".to_string(),
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

async fn spawn_test_server_with_genie() -> TestServer {
    let state_dir = std::env::temp_dir().join(format!("wf-http-genie-{}", uuid::Uuid::new_v4()));
    let mut registry = ProviderRegistry::new();
    registry.register(Box::new(MockProvider::new()));
    registry.register(Box::new(GenieProvider::new(
        GenieModel::Genie3,
        "genie-test-key",
    )));

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

async fn create_test_world(address: SocketAddr, name: &str) -> String {
    create_test_world_with_provider(address, name, "mock").await
}

async fn create_test_world_with_provider(
    address: SocketAddr,
    name: &str,
    provider: &str,
) -> String {
    let body = serde_json::json!({
        "name": name,
        "provider": provider,
    })
    .to_string();
    let (status, create) = http_request(address, "POST", "/v1/worlds", &body).await;
    assert_eq!(status, 201);
    create["data"]["id"].as_str().unwrap().to_string()
}

async fn create_test_object(
    address: SocketAddr,
    world_id: &str,
    body: serde_json::Value,
) -> String {
    let (status, created) = http_request(
        address,
        "POST",
        &format!("/v1/worlds/{world_id}/objects"),
        &body.to_string(),
    )
    .await;
    assert_eq!(status, 201);
    created["data"]["id"].as_str().unwrap().to_string()
}

fn eval_world_state_with_named_object(
    world_name: &str,
    object_name: &str,
    position: worldforge_core::types::Position,
) -> worldforge_core::state::WorldState {
    let mut state = worldforge_core::state::WorldState::new(world_name, "mock");
    let object = worldforge_core::scene::SceneObject::new(
        object_name,
        worldforge_core::types::Pose {
            position,
            ..worldforge_core::types::Pose::default()
        },
        worldforge_core::types::BBox {
            min: worldforge_core::types::Position {
                x: position.x - 0.1,
                y: position.y - 0.1,
                z: position.z - 0.1,
            },
            max: worldforge_core::types::Position {
                x: position.x + 0.1,
                y: position.y + 0.1,
                z: position.z + 0.1,
            },
        },
    );
    state.scene.add_object(object);
    state
}

fn object_exists_eval_suite(object_name: &str) -> worldforge_eval::EvalSuite {
    worldforge_eval::EvalSuite {
        name: format!("{object_name} overlay"),
        scenarios: vec![worldforge_eval::EvalScenario {
            name: "object_exists".to_string(),
            description: format!("Checks that {object_name} is present in the overlaid world"),
            initial_state: worldforge_core::state::WorldState::new("fixture", "eval"),
            actions: Vec::new(),
            expected_outcomes: vec![worldforge_eval::ExpectedOutcome::ObjectExists {
                name: object_name.to_string(),
            }],
            ground_truth: None,
        }],
        dimensions: vec![worldforge_eval::EvalDimension::ObjectPermanence],
        providers: vec!["mock".to_string()],
    }
}

async fn export_world_snapshot(address: SocketAddr, world_id: &str, format: &str) -> Value {
    let (status, response) = http_request(
        address,
        "GET",
        &format!("/v1/worlds/{world_id}/export?format={format}"),
        "",
    )
    .await;
    assert_eq!(status, 200);
    response
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn hex_decode(value: &str) -> Vec<u8> {
    fn hex_value(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    let bytes = value.as_bytes();
    assert!(bytes.len().is_multiple_of(2));

    let mut decoded = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let hi = hex_value(chunk[0]).expect("snapshot contains non-hex character");
        let lo = hex_value(chunk[1]).expect("snapshot contains non-hex character");
        decoded.push((hi << 4) | lo);
    }

    decoded
}

#[tokio::test]
async fn test_rest_eval_run_supports_standalone_suite() {
    let server = spawn_test_server().await;
    let body = serde_json::json!({
        "suite": "physics",
        "providers": ["mock"],
    })
    .to_string();

    let (status, response) = http_request(server.address, "POST", "/v1/evals/run", &body).await;
    assert_eq!(status, 200);
    assert_eq!(response["data"]["suite"], "Physics Standard");
    assert_eq!(
        response["data"]["provider_summaries"][0]["provider"],
        "mock"
    );
}

#[tokio::test]
async fn test_rest_eval_run_supports_world_id_overlay() {
    let server = spawn_test_server().await;
    let world_id = create_test_world(server.address, "eval-overlay").await;
    create_test_object(
        server.address,
        &world_id,
        serde_json::json!({
            "name": "cube",
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "bbox": {
                "min": {"x": -0.1, "y": -0.1, "z": -0.1},
                "max": {"x": 0.1, "y": 0.1, "z": 0.1}
            }
        }),
    )
    .await;

    let suite = object_exists_eval_suite("cube");
    let body = serde_json::json!({
        "suite_definition": suite,
        "providers": ["mock"],
        "world_id": world_id,
    })
    .to_string();

    let (status, response) = http_request(server.address, "POST", "/v1/evals/run", &body).await;
    assert_eq!(status, 200);
    assert_eq!(
        response["data"]["results"][0]["outcomes"][0]["passed"],
        true
    );
}

#[tokio::test]
async fn test_rest_eval_run_supports_inline_world_state_overlay() {
    let server = spawn_test_server().await;
    let suite = object_exists_eval_suite("sphere");
    let world_state = eval_world_state_with_named_object(
        "inline-eval",
        "sphere",
        worldforge_core::types::Position {
            x: 0.2,
            y: 0.4,
            z: -0.1,
        },
    );
    let body = serde_json::json!({
        "suite_definition": suite,
        "providers": ["mock"],
        "world_state": world_state,
        "report_format": "markdown",
    })
    .to_string();

    let (status, response) = http_request(server.address, "POST", "/v1/evals/run", &body).await;
    assert_eq!(status, 200);
    assert_eq!(response["data"]["format"], "markdown");
    assert_eq!(response["data"]["suite"], "sphere overlay");
    assert!(response["data"]["content"]
        .as_str()
        .unwrap()
        .contains("sphere overlay"));
}

#[tokio::test]
async fn test_rest_eval_run_rejects_ambiguous_world_inputs() {
    let server = spawn_test_server().await;
    let world_id = create_test_world(server.address, "eval-ambiguous").await;
    let world_state = eval_world_state_with_named_object(
        "inline-eval",
        "cube",
        worldforge_core::types::Position {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
    );
    let body = serde_json::json!({
        "suite": "physics",
        "providers": ["mock"],
        "world_id": world_id,
        "world_state": world_state,
    })
    .to_string();

    let (status, response) = http_request(server.address, "POST", "/v1/evals/run", &body).await;
    assert_eq!(status, 400);
    assert!(response["error"]
        .as_str()
        .unwrap()
        .contains("provide at most one of world_id or world_state"));
}

#[tokio::test]
async fn test_live_http_rejects_malformed_request_line() {
    let server = spawn_test_server().await;
    let request = "BROKEN\r\nHost: localhost\r\nConnection: close\r\n\r\n";

    let response = raw_http_response(server.address, request).await;
    assert_eq!(response.status, 400);

    let payload: Value = serde_json::from_str(&response.body).unwrap();
    assert_eq!(payload["success"], false);
    assert!(payload["error"]
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("bad request"));
}

#[tokio::test]
async fn test_live_http_rejects_invalid_content_length() {
    let server = spawn_test_server().await;
    let request = build_http_request(
        "POST",
        "/v1/worlds",
        &[
            ("Content-Length", "not-a-number"),
            ("Content-Type", "application/json"),
        ],
        r#"{"name":"invalid-length","provider":"mock"}"#,
    );

    let response = raw_http_response(server.address, &request).await;
    assert_eq!(response.status, 400);

    let payload: Value = serde_json::from_str(&response.body).unwrap();
    assert_eq!(payload["success"], false);
    assert!(payload["error"]
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("content-length"));
}

#[tokio::test]
async fn test_live_http_rejects_too_many_headers() {
    let server = spawn_test_server().await;
    let mut headers = Vec::new();
    for index in 0..65 {
        headers.push((format!("X-Header-{index}"), "value".to_string()));
    }
    let header_refs: Vec<_> = headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    let request = build_http_request("GET", "/v1/worlds", &header_refs, "");

    let response = raw_http_response(server.address, &request).await;
    assert_eq!(response.status, 400);

    let payload: Value = serde_json::from_str(&response.body).unwrap();
    assert_eq!(payload["success"], false);
    assert_eq!(payload["error"], "too many headers");
}

#[tokio::test]
async fn test_live_http_reports_405_for_known_path_wrong_method() {
    let server = spawn_test_server().await;
    let request = build_http_request("PATCH", "/v1/providers", &[], "");

    let response = raw_http_response(server.address, &request).await;
    assert_eq!(response.status, 405);

    let allow = response.header("allow").unwrap_or("");
    assert!(allow.contains("GET"));
}

#[tokio::test]
async fn test_live_http_head_on_get_endpoint_suppresses_body() {
    let server = spawn_test_server().await;
    let request = build_http_request("HEAD", "/v1/providers", &[], "");

    let response = raw_http_response(server.address, &request).await;
    assert_eq!(response.status, 200);
    assert!(response.body.is_empty());

    let content_type = response.header("content-type").unwrap_or("");
    assert!(
        content_type.contains("application/json"),
        "expected JSON response headers for HEAD request, got: {content_type}"
    );
}

#[tokio::test]
async fn test_live_http_decodes_query_parameters_for_provider_filtering() {
    let server = spawn_test_server().await;
    let request = build_http_request(
        "GET",
        "/v1/providers?capability=action%2Dconditioned",
        &[],
        "",
    );

    let response = raw_http_response(server.address, &request).await;
    assert_eq!(response.status, 200);

    let payload: Value = serde_json::from_str(&response.body).unwrap();
    let providers = payload["data"].as_array().unwrap();
    assert_eq!(providers.len(), 1);
    assert_eq!(providers[0]["name"], "mock");
}

#[tokio::test]
async fn test_live_http_rendered_eval_reports_use_content_type_and_body_semantics() {
    let server = spawn_test_server().await;

    for format in ["markdown", "csv"] {
        let body = serde_json::json!({
            "suite": "physics",
            "providers": ["mock"],
            "report_format": format,
        })
        .to_string();
        let request = build_http_request(
            "POST",
            "/v1/evals/run",
            &[("Content-Type", "application/json")],
            &body,
        );

        let response = raw_http_response(server.address, &request).await;
        assert_eq!(response.status, 200);

        let content_type = response.header("content-type").unwrap_or("");
        if content_type.contains("application/json") || response.body.trim_start().starts_with('{')
        {
            let payload: Value = serde_json::from_str(&response.body).unwrap();
            assert_eq!(payload["success"], true);
            assert_eq!(payload["data"]["format"], format);
            assert!(payload["data"]["content"]
                .as_str()
                .unwrap()
                .contains("Physics"));
        } else {
            match format {
                "markdown" => {
                    assert!(
                        content_type.contains("markdown") || content_type.contains("text/plain")
                    );
                    assert!(response.body.contains("# Evaluation Report"));
                }
                "csv" => {
                    assert!(content_type.contains("csv") || content_type.contains("text/plain"));
                    assert!(response.body.contains("suite,provider,scenario"));
                }
                _ => unreachable!(),
            }
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&sha256_hash(bytes))
}

fn assert_export_metadata(response: &Value, format: &str, encoding: &str, snapshot_bytes: &[u8]) {
    assert_eq!(response["data"]["format"], format);
    assert_eq!(response["data"]["encoding"], encoding);
    assert_eq!(response["data"]["sha256"], sha256_hex(snapshot_bytes));
}

async fn import_world_snapshot(
    address: SocketAddr,
    snapshot: &str,
    format: &str,
    new_id: bool,
    name: Option<&str>,
) -> Value {
    import_world_snapshot_with_metadata(address, snapshot, format, None, None, new_id, name).await
}

async fn import_world_snapshot_with_metadata(
    address: SocketAddr,
    snapshot: &str,
    format: &str,
    encoding: Option<&str>,
    sha256: Option<&str>,
    new_id: bool,
    name: Option<&str>,
) -> Value {
    let mut body = serde_json::json!({
        "format": format,
        "snapshot": snapshot,
        "new_id": new_id,
        "name": name,
    });

    if let Some(encoding) = encoding {
        body["encoding"] = serde_json::Value::String(encoding.to_string());
    }
    if let Some(sha256) = sha256 {
        body["sha256"] = serde_json::Value::String(sha256.to_string());
    }

    let (status, response) =
        http_request(address, "POST", "/v1/worlds/import", &body.to_string()).await;
    assert_eq!(status, 201);
    response
}

async fn http_request(address: SocketAddr, method: &str, path: &str, body: &str) -> (u16, Value) {
    let request = build_http_request(method, path, &[("Content-Type", "application/json")], body);
    let response = raw_http_request(address, &request).await;
    let (status, response_body) = parse_http_response(&response);
    let json = serde_json::from_str(&response_body).unwrap();
    (status, json)
}

#[derive(Debug, Clone)]
struct RawHttpResponse {
    status: u16,
    headers: HashMap<String, String>,
    body: String,
}

impl RawHttpResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }
}

async fn raw_http_request(address: SocketAddr, request: &str) -> String {
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    response
}

async fn raw_http_response(address: SocketAddr, request: &str) -> RawHttpResponse {
    let response = raw_http_request(address, request).await;
    parse_raw_http_response(&response)
}

fn build_http_request(
    method: &str,
    path: &str,
    extra_headers: &[(&str, &str)],
    body: &str,
) -> String {
    let has_content_length = extra_headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("content-length"));
    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n");

    for (name, value) in extra_headers {
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }

    if !has_content_length {
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }

    request.push_str("Connection: close\r\n\r\n");
    request.push_str(body);
    request
}

fn parse_raw_http_response(response: &str) -> RawHttpResponse {
    let (headers, body) = response.split_once("\r\n\r\n").unwrap_or((response, ""));
    let mut lines = headers.lines();
    let status = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);

    let headers = lines
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_ascii_lowercase(), value.trim().to_string()))
        })
        .collect();

    RawHttpResponse {
        status,
        headers,
        body: body.to_string(),
    }
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
        2
    );

    let (status, history) = http_request(
        server.address,
        "GET",
        &format!("/v1/worlds/{world_id}/history"),
        "",
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(history["data"].as_array().unwrap().len(), 2);
    assert!(history["data"][0]["action"].is_null());

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
async fn test_live_http_create_world_from_prompt_bootstraps_scene() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"prompt":"A kitchen with a mug","name":"seeded-kitchen","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    assert_eq!(create["data"]["name"], "seeded-kitchen");
    assert_eq!(create["data"]["description"], "A kitchen with a mug");
    assert!(create["data"]["object_count"].as_u64().unwrap() >= 2);

    let world_id = create["data"]["id"].as_str().unwrap();
    let (status, world) =
        http_request(server.address, "GET", &format!("/v1/worlds/{world_id}"), "").await;
    assert_eq!(status, 200);
    assert_eq!(
        world["data"]["metadata"]["description"],
        "A kitchen with a mug"
    );
    assert!(world["data"]["scene"]["objects"].as_object().unwrap().len() >= 2);
}

#[tokio::test]
async fn test_live_http_snapshot_export_import_json_roundtrip() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"json_export_world","prompt":"A kitchen with a mug","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let original_id = create["data"]["id"].as_str().unwrap().to_string();

    let predict_body =
        r#"{"action":{"Move":{"target":{"x":1.0,"y":0.8,"z":0.0},"speed":1.0}},"provider":"mock"}"#;
    let (status, prediction) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{original_id}/predict"),
        predict_body,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(prediction["data"]["provider"], "mock");

    let export = export_world_snapshot(server.address, &original_id, "json").await;
    assert_eq!(export["data"]["id"], original_id);

    let snapshot = export["data"]["snapshot"].as_str().unwrap();
    let snapshot_bytes = snapshot.as_bytes();
    assert_export_metadata(&export, "json", "utf-8", snapshot_bytes);
    let snapshot_json: Value = serde_json::from_str(snapshot).unwrap();
    assert_eq!(snapshot_json["metadata"]["name"], "json_export_world");
    assert_eq!(
        snapshot_json["metadata"]["description"],
        "A kitchen with a mug"
    );
    assert_eq!(snapshot_json["time"]["step"], 1);
    assert_eq!(
        snapshot_json["history"]["states"].as_array().unwrap().len(),
        2
    );

    let imported = import_world_snapshot(
        server.address,
        snapshot,
        "json",
        true,
        Some("json_snapshot_copy"),
    )
    .await;
    let imported_id = imported["data"]["id"].as_str().unwrap().to_string();
    assert_ne!(imported_id, original_id);
    assert_eq!(imported["data"]["metadata"]["name"], "json_snapshot_copy");
    assert_eq!(
        imported["data"]["metadata"]["description"],
        "A kitchen with a mug"
    );
    assert_eq!(imported["data"]["time"]["step"], 1);
    assert_eq!(
        imported["data"]["history"]["states"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn test_live_http_snapshot_export_import_msgpack_roundtrip() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"msgpack_export_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let original_id = create["data"]["id"].as_str().unwrap().to_string();

    let predict_body =
        r#"{"action":{"Move":{"target":{"x":0.5,"y":0.8,"z":0.0},"speed":1.0}},"provider":"mock"}"#;
    let (status, prediction) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{original_id}/predict"),
        predict_body,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(prediction["data"]["provider"], "mock");

    let export = export_world_snapshot(server.address, &original_id, "msgpack").await;
    assert_eq!(export["data"]["id"], original_id);

    let snapshot = export["data"]["snapshot"].as_str().unwrap();
    assert!(!snapshot.is_empty());
    assert!(snapshot.len() % 2 == 0);
    assert!(snapshot.chars().all(|ch| ch.is_ascii_hexdigit()));
    let snapshot_bytes = hex_decode(snapshot);
    assert_export_metadata(&export, "msgpack", "hex", &snapshot_bytes);

    let imported = import_world_snapshot(
        server.address,
        snapshot,
        "msgpack",
        true,
        Some("msgpack_snapshot_copy"),
    )
    .await;
    let imported_id = imported["data"]["id"].as_str().unwrap().to_string();
    assert_ne!(imported_id, original_id);
    assert_eq!(
        imported["data"]["metadata"]["name"],
        "msgpack_snapshot_copy"
    );
    assert_eq!(imported["data"]["time"]["step"], 1);
    assert_eq!(
        imported["data"]["history"]["states"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    let loaded = http_request(
        server.address,
        "GET",
        &format!("/v1/worlds/{imported_id}"),
        "",
    )
    .await;
    assert_eq!(loaded.0, 200);
    assert_eq!(
        loaded.1["data"]["metadata"]["name"],
        "msgpack_snapshot_copy"
    );
    assert_eq!(loaded.1["data"]["time"]["step"], 1);
}

#[tokio::test]
async fn test_live_http_snapshot_import_rejects_checksum_mismatch() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"checksum_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let export = export_world_snapshot(server.address, &world_id, "json").await;
    let snapshot = export["data"]["snapshot"].as_str().unwrap();
    let mut checksum = sha256_hex(snapshot.as_bytes());
    checksum.replace_range(0..1, if &checksum[0..1] == "0" { "1" } else { "0" });

    let body = serde_json::json!({
        "format": "json",
        "snapshot": snapshot,
        "encoding": "utf-8",
        "sha256": checksum,
        "new_id": true,
    });
    let (status, response) = http_request(
        server.address,
        "POST",
        "/v1/worlds/import",
        &body.to_string(),
    )
    .await;
    assert_eq!(status, 400);
    assert!(!response["success"].as_bool().unwrap_or(true));
}

#[tokio::test]
async fn test_live_http_snapshot_import_rejects_invalid_encoding_metadata() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"encoding_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let export = export_world_snapshot(server.address, &world_id, "msgpack").await;
    let snapshot = export["data"]["snapshot"].as_str().unwrap();

    let body = serde_json::json!({
        "format": "msgpack",
        "snapshot": snapshot,
        "encoding": "base64",
        "new_id": true,
    });
    let (status, response) = http_request(
        server.address,
        "POST",
        "/v1/worlds/import",
        &body.to_string(),
    )
    .await;
    assert_eq!(status, 400);
    assert!(!response["success"].as_bool().unwrap_or(true));
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
async fn test_live_http_world_lifecycle_msgpack_file_store() {
    let server = spawn_test_server_msgpack().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"msgpack_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let persisted_path = server.state_dir.join(format!("{world_id}.msgpack"));
    assert!(persisted_path.exists());

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
async fn test_live_http_provider_embed() {
    let server = spawn_test_server().await;
    let body = serde_json::json!({
        "text": "a red mug on a table",
        "video": {
            "frames": [],
            "fps": 8.0,
            "resolution": [64, 64],
            "duration": 1.0
        }
    })
    .to_string();

    let (status, response) =
        http_request(server.address, "POST", "/v1/providers/mock/embed", &body).await;
    assert_eq!(status, 200);
    assert_eq!(response["data"]["provider"], "mock");
    assert_eq!(response["data"]["model"], "mock-embedding-v1");
    assert_eq!(
        response["data"]["embedding"]["shape"],
        serde_json::json!([32])
    );
}

#[tokio::test]
async fn test_live_http_provider_reason_video_only() {
    let server = spawn_test_server().await;
    let body = serde_json::json!({
        "query": "what do you see?",
        "video": {
            "frames": [],
            "fps": 15.0,
            "resolution": [320, 240],
            "duration": 1.5
        }
    })
    .to_string();

    let (status, response) =
        http_request(server.address, "POST", "/v1/providers/mock/reason", &body).await;
    assert_eq!(status, 200);
    assert!(response["data"]["answer"]
        .as_str()
        .unwrap()
        .contains("echo the query"));
}

#[tokio::test]
async fn test_live_http_patch_object_persists_position_updates() {
    let server = spawn_test_server().await;
    let world_id = create_test_world(server.address, "patch_world").await;

    let object_id = create_test_object(
        server.address,
        &world_id,
        serde_json::json!({
            "name": "crate",
            "position": { "x": 0.0, "y": 1.0, "z": 0.0 },
            "bbox": {
                "min": { "x": -0.5, "y": 0.5, "z": -0.25 },
                "max": { "x": 0.5, "y": 1.5, "z": 0.25 }
            },
            "semantic_label": "storage"
        }),
    )
    .await;

    let patch_body = serde_json::json!({
        "position": { "x": 2.0, "y": 3.0, "z": -1.0 }
    })
    .to_string();
    let (status, updated) = http_request(
        server.address,
        "PATCH",
        &format!("/v1/worlds/{world_id}/objects/{object_id}"),
        &patch_body,
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(updated["data"]["id"], object_id);
    assert_eq!(updated["data"]["pose"]["position"]["x"], 2.0);
    assert_eq!(updated["data"]["pose"]["position"]["y"], 3.0);
    assert_eq!(updated["data"]["pose"]["position"]["z"], -1.0);

    let bbox_width = updated["data"]["bbox"]["max"]["x"].as_f64().unwrap()
        - updated["data"]["bbox"]["min"]["x"].as_f64().unwrap();
    let bbox_height = updated["data"]["bbox"]["max"]["y"].as_f64().unwrap()
        - updated["data"]["bbox"]["min"]["y"].as_f64().unwrap();
    let bbox_depth = updated["data"]["bbox"]["max"]["z"].as_f64().unwrap()
        - updated["data"]["bbox"]["min"]["z"].as_f64().unwrap();
    assert!((bbox_width - 1.0).abs() < f64::EPSILON);
    assert!((bbox_height - 1.0).abs() < f64::EPSILON);
    assert!((bbox_depth - 0.5).abs() < f64::EPSILON);

    let store = FileStateStore::new(&server.state_dir);
    let persisted = store.load(&world_id.parse().unwrap()).await.unwrap();
    let persisted_object = persisted
        .scene
        .get_object(&object_id.parse().unwrap())
        .unwrap();
    assert_eq!(persisted_object.pose.position.x, 2.0);
    assert_eq!(persisted_object.pose.position.y, 3.0);
    assert_eq!(persisted_object.pose.position.z, -1.0);
    assert_eq!(persisted_object.bbox.center().x, 2.0);
    assert_eq!(persisted_object.bbox.center().y, 3.0);
    assert_eq!(persisted_object.bbox.center().z, -1.0);
}

#[tokio::test]
async fn test_live_http_patch_object_validation_errors() {
    let server = spawn_test_server().await;
    let world_id = create_test_world(server.address, "patch_errors").await;
    let object_id = create_test_object(
        server.address,
        &world_id,
        serde_json::json!({
            "name": "cube",
            "position": { "x": 0.0, "y": 0.5, "z": 0.0 },
            "bbox": {
                "min": { "x": -0.25, "y": 0.25, "z": -0.25 },
                "max": { "x": 0.25, "y": 0.75, "z": 0.25 }
            }
        }),
    )
    .await;

    let patch_body = serde_json::json!({
        "position": { "x": 1.0, "y": 1.0, "z": 1.0 }
    })
    .to_string();

    let (status, invalid_world) = http_request(
        server.address,
        "PATCH",
        &format!("/v1/worlds/not-a-uuid/objects/{object_id}"),
        &patch_body,
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(invalid_world["success"], false);
    assert_eq!(invalid_world["error"], "invalid world ID");

    let (status, invalid_object) = http_request(
        server.address,
        "PATCH",
        &format!("/v1/worlds/{world_id}/objects/not-a-uuid"),
        &patch_body,
    )
    .await;
    assert_eq!(status, 400);
    assert_eq!(invalid_object["success"], false);
    assert_eq!(invalid_object["error"], "invalid object ID");

    let missing_world_id = uuid::Uuid::new_v4();
    let (status, missing_world) = http_request(
        server.address,
        "PATCH",
        &format!("/v1/worlds/{missing_world_id}/objects/{object_id}"),
        &patch_body,
    )
    .await;
    assert_eq!(status, 404);
    assert_eq!(missing_world["success"], false);
    assert!(missing_world["error"]
        .as_str()
        .unwrap_or_default()
        .contains("world not found"));

    let (status, missing_object) = http_request(
        server.address,
        "PATCH",
        &format!("/v1/worlds/{world_id}/objects/{}", uuid::Uuid::new_v4()),
        &patch_body,
    )
    .await;
    assert_eq!(status, 404);
    assert_eq!(missing_object["success"], false);
    assert!(missing_object["error"]
        .as_str()
        .unwrap_or_default()
        .contains("object not found"));
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
async fn test_live_http_provider_transfer() {
    let server = spawn_test_server().await;

    let transfer_body = r#"{
        "source":{"frames":[],"fps":10.0,"resolution":[320,180],"duration":2.0},
        "controls":{},
        "config":{"resolution":[640,360],"fps":24.0,"control_strength":0.5}
    }"#;
    let (status, clip) = http_request(
        server.address,
        "POST",
        "/v1/providers/mock/transfer",
        transfer_body,
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(clip["data"]["resolution"], serde_json::json!([640, 360]));
    assert_eq!(clip["data"]["fps"], 24.0);
    assert_eq!(clip["data"]["duration"], 2.0);
}

#[tokio::test]
async fn test_live_http_genie_reasoning_is_grounded_in_scene_state() {
    let server = spawn_test_server_with_genie().await;
    let world_id =
        create_test_world_with_provider(server.address, "genie_reason_world", "genie").await;

    let _ = create_test_object(
        server.address,
        &world_id,
        serde_json::json!({
            "name": "table",
            "position": { "x": 0.0, "y": 0.0, "z": 0.0 },
            "bbox": {
                "min": { "x": -0.75, "y": -0.05, "z": -0.75 },
                "max": { "x": 0.75, "y": 0.05, "z": 0.75 }
            }
        }),
    )
    .await;
    let _ = create_test_object(
        server.address,
        &world_id,
        serde_json::json!({
            "name": "mug",
            "position": { "x": 0.1, "y": 0.82, "z": 0.0 },
            "bbox": {
                "min": { "x": 0.05, "y": 0.77, "z": -0.05 },
                "max": { "x": 0.15, "y": 0.87, "z": 0.05 }
            }
        }),
    )
    .await;

    let (status, response) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/reason"),
        r#"{"query":"What objects are in the scene?"}"#,
    )
    .await;

    assert_eq!(status, 200);
    let answer = response["data"]["answer"]
        .as_str()
        .unwrap_or_default()
        .to_lowercase();
    assert!(answer.contains("mug"));
    assert!(answer.contains("table"));
    let evidence = response["data"]["evidence"].as_array().unwrap();
    assert!(!evidence.is_empty());
}

#[tokio::test]
async fn test_live_http_genie_native_plan_spawn_goal() {
    let server = spawn_test_server_with_genie().await;
    let world_id =
        create_test_world_with_provider(server.address, "genie_plan_world", "genie").await;

    let plan_body = r#"{
        "goal":"spawn cube",
        "provider":"genie",
        "planner":"provider-native",
        "max_steps":4
    }"#;
    let (status, plan_json) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/plan"),
        plan_body,
    )
    .await;

    assert_eq!(status, 200);
    let plan: worldforge_core::prediction::Plan =
        serde_json::from_value(plan_json["data"].clone()).unwrap();
    assert!(!plan.actions.is_empty());
    assert_eq!(plan.predicted_states.len(), plan.actions.len());
    assert_eq!(plan.guardrail_compliance.len(), plan.actions.len());
    assert!(plan.actions.iter().any(|action| matches!(
        action,
        worldforge_core::action::Action::SpawnObject { template, .. }
            if template.to_lowercase().contains("cube")
    )));
}

#[tokio::test]
async fn test_live_http_verify_proof_endpoint() {
    use worldforge_verify::{MockVerifier, ZkVerifier};

    let server = spawn_test_server().await;
    let verifier = MockVerifier::new();
    let proof = verifier.prove_inference([1; 32], [2; 32], [3; 32]).unwrap();
    let body = serde_json::json!({ "proof": proof }).to_string();

    let (status, response) = http_request(server.address, "POST", "/v1/verify/proof", &body).await;

    assert_eq!(status, 200);
    assert_eq!(response["data"]["verification"]["valid"], true);
    assert_eq!(response["data"]["proof"]["backend"], "Mock");
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
async fn test_live_http_execute_plan_persists_state() {
    let server = spawn_test_server().await;

    let world_id = create_test_world(server.address, "execute_plan_world").await;
    let _ = create_test_object(
        server.address,
        &world_id,
        serde_json::json!({
            "name": "ball",
            "position": { "x": 0.0, "y": 0.5, "z": 0.0 },
            "bbox": {
                "min": { "x": -0.1, "y": 0.4, "z": -0.1 },
                "max": { "x": 0.1, "y": 0.6, "z": 0.1 }
            }
        }),
    )
    .await;

    let execute_body = serde_json::json!({
        "plan": {
            "actions": [
                {
                    "Move": {
                        "target": { "x": 1.0, "y": 0.5, "z": 0.0 },
                        "speed": 1.0
                    }
                }
            ],
            "predicted_states": [],
            "predicted_videos": null,
            "total_cost": 0.0,
            "success_probability": 1.0,
            "guardrail_compliance": [],
            "planning_time_ms": 0,
            "iterations_used": 1
        }
    })
    .to_string();
    let (status, execution) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/execute-plan"),
        &execute_body,
    )
    .await;

    assert_eq!(status, 200);
    assert_eq!(
        execution["data"]["predictions"].as_array().unwrap().len(),
        1
    );
    assert_eq!(execution["data"]["final_state"]["time"]["step"], 1);

    let (status, world) =
        http_request(server.address, "GET", &format!("/v1/worlds/{world_id}"), "").await;
    assert_eq!(status, 200);
    assert_eq!(world["data"]["time"]["step"], 1);
    assert_eq!(
        world["data"]["history"]["states"].as_array().unwrap().len(),
        2
    );
}

#[tokio::test]
async fn test_live_http_plan_provider_native_with_mock() {
    let server = spawn_test_server().await;

    let (status, descriptor) = http_request(server.address, "GET", "/v1/providers/mock", "").await;
    assert_eq!(status, 200);
    let supports_native = descriptor["data"]["capabilities"]["supports_planning"]
        .as_bool()
        .unwrap_or(false);

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"native_plan_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let plan_body = r#"{
        "goal":"spawn cube",
        "provider":"mock",
        "planner":"provider-native",
        "max_steps":4
    }"#;
    let (status, response) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/plan"),
        plan_body,
    )
    .await;

    if supports_native {
        assert_eq!(status, 200);
        let actions = response["data"]["actions"].as_array().unwrap();
        let predicted_states = response["data"]["predicted_states"].as_array().unwrap();
        assert!(!actions.is_empty());
        assert_eq!(actions.len(), predicted_states.len());
        return;
    }

    assert!(status == 400 || status == 500);
    let error_message = response["error"]
        .as_str()
        .unwrap_or_default()
        .to_lowercase();
    assert!(
        error_message.contains("native planning") || error_message.contains("unsupported"),
        "expected unsupported native planning error, got: {error_message}"
    );
}

#[tokio::test]
async fn test_live_http_plan_relational_goal_spawns_object_near_anchor() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"relational_plan_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    // Seed the world with the anchor object referenced by the natural-language goal.
    let add_anchor_body = r#"{
        "name":"red mug",
        "position":{"x":1.0,"y":0.8,"z":0.0},
        "bbox":{"min":{"x":0.95,"y":0.75,"z":-0.05},"max":{"x":1.05,"y":0.85,"z":0.05}},
        "semantic_label":"mug"
    }"#;
    let (status, _object) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/objects"),
        add_anchor_body,
    )
    .await;
    assert_eq!(status, 201);

    let plan_body = r#"{
        "goal":"spawn cube next to the red mug",
        "provider":"mock",
        "planner":"sampling",
        "max_steps":4,
        "num_samples":48,
        "top_k":5
    }"#;
    let (status, plan_json) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/plan"),
        plan_body,
    )
    .await;
    assert_eq!(status, 200);

    let plan: worldforge_core::prediction::Plan =
        serde_json::from_value(plan_json["data"].clone()).unwrap();
    assert!(
        !plan.actions.is_empty(),
        "planning should produce actions for relational spawn goals"
    );

    // The goal should not be considered satisfied merely because the anchor exists.
    assert!(
        plan.actions.iter().any(|action| {
            matches!(
                action,
                worldforge_core::action::Action::SpawnObject { template, .. }
                    if template.to_lowercase().contains("cube")
            )
        }),
        "plan must include spawning the requested cube, not only operate on the anchor object"
    );

    let final_state = plan
        .predicted_states
        .last()
        .expect("plan should include predicted states");
    let anchor = final_state
        .scene
        .objects
        .values()
        .find(|object| object.name.to_lowercase() == "red mug")
        .expect("anchor object should remain in the world");
    let spawned_cubes: Vec<_> = final_state
        .scene
        .objects
        .values()
        .filter(|object| object.name.to_lowercase().contains("cube"))
        .collect();
    assert!(
        !spawned_cubes.is_empty(),
        "final state should contain a spawned cube"
    );

    let nearest_cube_distance = spawned_cubes
        .iter()
        .map(|cube| {
            let dx = cube.pose.position.x - anchor.pose.position.x;
            let dy = cube.pose.position.y - anchor.pose.position.y;
            let dz = cube.pose.position.z - anchor.pose.position.z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(f32::INFINITY, |best, distance| best.min(distance));
    assert!(
        nearest_cube_distance <= 0.8,
        "spawned cube should be near the named anchor; got distance {nearest_cube_distance}"
    );
}

#[tokio::test]
async fn test_live_http_plan_structured_condition_goal() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"structured_plan_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let add_object_body = r#"{
        "name":"ball",
        "position":{"x":0.0,"y":0.5,"z":0.0},
        "bbox":{"min":{"x":-0.1,"y":0.4,"z":-0.1},"max":{"x":0.1,"y":0.6,"z":0.1}}
    }"#;
    let (status, object) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/objects"),
        add_object_body,
    )
    .await;
    assert_eq!(status, 201);
    let object_id = object["data"]["id"].as_str().unwrap();

    let plan_body = serde_json::json!({
        "goal": {
            "type": "condition",
            "condition": {
                "ObjectAt": {
                    "object": object_id,
                    "position": {"x": 1.0, "y": 0.5, "z": 0.0},
                    "tolerance": 0.05
                }
            }
        },
        "provider": "mock",
        "planner": "sampling",
        "max_steps": 4,
        "num_samples": 48,
        "top_k": 5
    })
    .to_string();
    let (status, plan_json) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/plan"),
        &plan_body,
    )
    .await;
    assert_eq!(status, 200);

    let plan: worldforge_core::prediction::Plan =
        serde_json::from_value(plan_json["data"].clone()).unwrap();
    assert!(
        !plan.actions.is_empty(),
        "structured condition planning should produce at least one action"
    );

    let final_state = plan
        .predicted_states
        .last()
        .expect("plan should include predicted states");
    let moved_object = final_state
        .scene
        .find_object_by_name("ball")
        .expect("planned state should keep the moved object");
    assert!(
        moved_object
            .pose
            .position
            .distance(worldforge_core::types::Position {
                x: 1.0,
                y: 0.5,
                z: 0.0,
            })
            <= 0.15,
        "structured condition goal should move the object close to the target position"
    );
}

#[tokio::test]
async fn test_live_http_plan_goal_image() {
    let server = spawn_test_server().await;

    let (status, create) = http_request(
        server.address,
        "POST",
        "/v1/worlds",
        r#"{"name":"goal_image_plan_world","provider":"mock"}"#,
    )
    .await;
    assert_eq!(status, 201);
    let world_id = create["data"]["id"].as_str().unwrap().to_string();

    let add_object_body = r#"{
        "name":"ball",
        "position":{"x":0.0,"y":0.5,"z":0.0},
        "bbox":{"min":{"x":-0.1,"y":0.4,"z":-0.1},"max":{"x":0.1,"y":0.6,"z":0.1}}
    }"#;
    let (status, _) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/objects"),
        add_object_body,
    )
    .await;
    assert_eq!(status, 201);

    let mut target_state = worldforge_core::state::WorldState::new("goal-image-target", "mock");
    let object = worldforge_core::scene::SceneObject::new(
        "ball",
        worldforge_core::types::Pose {
            position: worldforge_core::types::Position {
                x: 0.0,
                y: 0.5,
                z: 0.0,
            },
            ..worldforge_core::types::Pose::default()
        },
        worldforge_core::types::BBox {
            min: worldforge_core::types::Position {
                x: -0.1,
                y: 0.4,
                z: -0.1,
            },
            max: worldforge_core::types::Position {
                x: 0.1,
                y: 0.6,
                z: 0.1,
            },
        },
    );
    let object_id = object.id;
    target_state.scene.add_object(object);
    target_state
        .scene
        .get_object_mut(&object_id)
        .unwrap()
        .set_position(worldforge_core::types::Position {
            x: 1.0,
            y: 0.5,
            z: 0.0,
        });

    let plan_body = serde_json::json!({
        "goal": {
            "type": "goal_image",
            "image": worldforge_core::goal_image::render_scene_goal_image(&target_state, (32, 24))
        },
        "provider": "mock",
        "planner": "sampling",
        "max_steps": 4,
        "num_samples": 48,
        "top_k": 5
    })
    .to_string();
    let (status, plan_json) = http_request(
        server.address,
        "POST",
        &format!("/v1/worlds/{world_id}/plan"),
        &plan_body,
    )
    .await;
    assert_eq!(status, 200);

    let plan: worldforge_core::prediction::Plan =
        serde_json::from_value(plan_json["data"].clone()).unwrap();
    assert!(
        !plan.actions.is_empty(),
        "goal-image planning should produce at least one action"
    );

    let final_state = plan
        .predicted_states
        .last()
        .expect("plan should include predicted states");
    let moved_object = final_state
        .scene
        .find_object_by_name("ball")
        .expect("planned state should keep the moved object");
    assert!(
        moved_object.pose.position.x > 0.5,
        "goal-image planning should move the object toward the rendered target"
    );
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
        fallback_provider: None,
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
    assert_eq!(multi.comparison.pairwise_agreements.len(), 1);
    assert!(multi.comparison.consensus.average_quality_score >= 0.0);
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
        verification_proof: None,
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
