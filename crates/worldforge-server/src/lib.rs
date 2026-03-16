//! WorldForge REST API Server
//!
//! A lightweight HTTP/JSON API server built on Tokio for interacting
//! with WorldForge functionality over the network.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::RwLock;

use worldforge_core::action::Action;
use worldforge_core::prediction::PredictionConfig;
use worldforge_core::provider::ProviderRegistry;
use worldforge_core::state::{FileStateStore, StateStore, WorldState};
use worldforge_core::types::WorldId;
use worldforge_eval::EvalSuite;
use worldforge_verify::{MockVerifier, ZkVerifier};

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to.
    pub bind_address: String,
    /// State storage directory.
    pub state_dir: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8080".to_string(),
            state_dir: ".worldforge".to_string(),
        }
    }
}

/// A bound WorldForge REST server ready to accept connections.
pub struct Server {
    listener: TcpListener,
    state: Arc<AppState>,
}

impl Server {
    /// Bind a server to the configured address.
    pub async fn bind(
        config: ServerConfig,
        registry: Arc<ProviderRegistry>,
    ) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(&config.bind_address).await?;
        let state = Arc::new(AppState {
            registry,
            store: FileStateStore::new(&config.state_dir),
            worlds: RwLock::new(HashMap::new()),
        });

        Ok(Self { listener, state })
    }

    /// Return the socket address the server is listening on.
    pub fn local_addr(&self) -> anyhow::Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Run the accept loop until the task is cancelled or an unrecoverable I/O error occurs.
    pub async fn run(self) -> anyhow::Result<()> {
        let local_addr = self.listener.local_addr()?;
        tracing::info!(address = %local_addr, "WorldForge server started");

        loop {
            let (stream, addr) = self.listener.accept().await?;
            let state = Arc::clone(&self.state);
            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, state).await {
                    tracing::error!(addr = %addr, error = %e, "request handling failed");
                }
            });
        }
    }
}

/// Shared application state.
struct AppState {
    registry: Arc<ProviderRegistry>,
    store: FileStateStore,
    worlds: RwLock<HashMap<WorldId, WorldState>>,
}

/// JSON request body for creating a world.
#[derive(Debug, Deserialize)]
struct CreateWorldRequest {
    name: String,
    provider: String,
}

/// JSON request body for prediction.
#[derive(Debug, Deserialize)]
struct PredictRequest {
    action: Action,
    #[serde(default)]
    config: PredictionConfig,
    #[serde(default = "default_provider")]
    provider: String,
}

fn default_provider() -> String {
    "mock".to_string()
}

/// JSON request body for planning.
#[derive(Debug, Deserialize)]
struct PlanRequest {
    goal: String,
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default = "default_provider")]
    provider: String,
}

fn default_max_steps() -> u32 {
    10
}

/// JSON request body for evaluation.
#[derive(Debug, Deserialize)]
struct EvaluateRequest {
    #[serde(default = "default_suite")]
    suite: String,
}

fn default_suite() -> String {
    "physics".to_string()
}

/// JSON request body for cross-provider comparison.
#[derive(Debug, Deserialize)]
struct CompareRequest {
    world_id: String,
    action: Action,
    providers: Vec<String>,
    #[serde(default)]
    config: PredictionConfig,
}

/// JSON request body for ZK verification.
#[derive(Debug, Deserialize)]
struct VerifyRequest {
    /// Proof type: "inference", "guardrail", or "provenance".
    #[serde(default = "default_proof_type")]
    proof_type: String,
}

fn default_proof_type() -> String {
    "inference".to_string()
}

/// JSON response envelope.
#[derive(Serialize)]
struct ApiResponse<T: Serialize> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    fn ok(data: T) -> String {
        serde_json::to_string(&ApiResponse {
            success: true,
            data: Some(data),
            error: None,
        })
        .unwrap_or_else(|_| r#"{"success":false,"error":"serialization failed"}"#.to_string())
    }
}

fn error_response(msg: &str) -> String {
    serde_json::to_string(&ApiResponse::<()> {
        success: false,
        data: None,
        error: Some(msg.to_string()),
    })
    .unwrap_or_else(|_| format!(r#"{{"success":false,"error":"{msg}"}}"#))
}

/// Start the WorldForge HTTP server.
pub async fn serve(config: ServerConfig, registry: Arc<ProviderRegistry>) -> anyhow::Result<()> {
    Server::bind(config, registry).await?.run().await
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    state: Arc<AppState>,
) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.split();
    let mut buf_reader = BufReader::new(reader);

    // Read request line
    let mut request_line = String::new();
    buf_reader.read_line(&mut request_line).await?;
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        send_response(&mut writer, 400, &error_response("bad request")).await?;
        return Ok(());
    }
    let method = parts[0];
    let path = parts[1];

    // Read headers
    const MAX_BODY_SIZE: usize = 4 * 1024 * 1024; // 4 MiB
    const MAX_HEADER_COUNT: usize = 64;
    let mut content_length: usize = 0;
    let mut header_count = 0;
    loop {
        let mut line = String::new();
        buf_reader.read_line(&mut line).await?;
        if line.trim().is_empty() {
            break;
        }
        header_count += 1;
        if header_count > MAX_HEADER_COUNT {
            send_response(&mut writer, 400, &error_response("too many headers")).await?;
            return Ok(());
        }
        // Case-insensitive header matching
        let lower = line.to_ascii_lowercase();
        if let Some(val) = lower.strip_prefix("content-length:") {
            content_length = val.trim().parse().unwrap_or(0);
        }
    }

    // Enforce body size limit
    if content_length > MAX_BODY_SIZE {
        send_response(&mut writer, 413, &error_response("request body too large")).await?;
        return Ok(());
    }

    // Read body
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        buf_reader.read_exact(&mut body).await?;
    }
    let body_str = String::from_utf8_lossy(&body);

    // Route request
    let (status, response_body) = route(method, path, &body_str, &state).await;
    send_response(&mut writer, status, &response_body).await?;
    Ok(())
}

async fn route(method: &str, path: &str, body: &str, state: &AppState) -> (u16, String) {
    // Trim trailing slash
    let path = path.trim_end_matches('/');

    match (method, path) {
        // GET /v1/providers
        ("GET", "/v1/providers") => {
            let names = state.registry.list();
            let providers: Vec<_> = names
                .into_iter()
                .map(|n| serde_json::json!({ "name": n }))
                .collect();
            (200, ApiResponse::ok(providers))
        }

        // GET /v1/providers/{name}/health
        ("GET", p) if p.starts_with("/v1/providers/") && p.ends_with("/health") => {
            let name = p
                .strip_prefix("/v1/providers/")
                .and_then(|s| s.strip_suffix("/health"))
                .unwrap_or("");
            match state.registry.get(name) {
                Ok(provider) => match provider.health_check().await {
                    Ok(status) => (200, ApiResponse::ok(status)),
                    Err(e) => (503, error_response(&e.to_string())),
                },
                Err(e) => (404, error_response(&e.to_string())),
            }
        }

        // POST /v1/worlds
        ("POST", "/v1/worlds") => match serde_json::from_str::<CreateWorldRequest>(body) {
            Ok(req) => {
                let ws = WorldState::new(&req.name, &req.provider);
                let id = ws.id;
                if let Err(e) = state.store.save(&ws).await {
                    return (500, error_response(&e.to_string()));
                }
                state.worlds.write().await.insert(id, ws.clone());
                (
                    201,
                    ApiResponse::ok(serde_json::json!({
                        "id": id.to_string(),
                        "name": req.name,
                        "provider": req.provider,
                    })),
                )
            }
            Err(e) => (400, error_response(&format!("invalid request: {e}"))),
        },

        // GET /v1/worlds — list all worlds
        ("GET", "/v1/worlds") => {
            let ids = state.store.list().await.unwrap_or_default();
            let mut worlds = Vec::new();
            for id in &ids {
                if let Ok(ws) = state.store.load(id).await {
                    worlds.push(serde_json::json!({
                        "id": id.to_string(),
                        "name": ws.metadata.name,
                        "provider": ws.metadata.created_by,
                        "step": ws.time.step,
                    }));
                }
            }
            (200, ApiResponse::ok(worlds))
        }

        // GET /v1/worlds/{id}
        ("GET", p)
            if p.starts_with("/v1/worlds/")
                && !p.contains("/predict")
                && !p.contains("/plan")
                && !p.contains("/evaluate")
                && !p.contains("/verify") =>
        {
            let id_str = p.strip_prefix("/v1/worlds/").unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match state.store.load(&id).await {
                    Ok(ws) => (200, ApiResponse::ok(ws)),
                    Err(e) => (404, error_response(&e.to_string())),
                },
                Err(_) => (400, error_response("invalid world ID")),
            }
        }

        // POST /v1/worlds/{id}/predict
        ("POST", p) if p.starts_with("/v1/worlds/") && p.ends_with("/predict") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|s| s.strip_suffix("/predict"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => {
                    match serde_json::from_str::<PredictRequest>(body) {
                        Ok(req) => {
                            let ws = match state.store.load(&id).await {
                                Ok(ws) => ws,
                                Err(e) => return (404, error_response(&e.to_string())),
                            };
                            let provider = match state.registry.get(&req.provider) {
                                Ok(p) => p,
                                Err(e) => return (404, error_response(&e.to_string())),
                            };
                            match provider.predict(&ws, &req.action, &req.config).await {
                                Ok(prediction) => {
                                    // Save updated state
                                    let _ = state.store.save(&prediction.output_state).await;
                                    (200, ApiResponse::ok(prediction))
                                }
                                Err(e) => (500, error_response(&e.to_string())),
                            }
                        }
                        Err(e) => (400, error_response(&format!("invalid request: {e}"))),
                    }
                }
                Err(_) => (400, error_response("invalid world ID")),
            }
        }

        // DELETE /v1/worlds/{id}
        ("DELETE", p) if p.starts_with("/v1/worlds/") => {
            let id_str = p.strip_prefix("/v1/worlds/").unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match state.store.delete(&id).await {
                    Ok(()) => {
                        state.worlds.write().await.remove(&id);
                        (
                            200,
                            ApiResponse::ok(serde_json::json!({"deleted": id.to_string()})),
                        )
                    }
                    Err(e) => (404, error_response(&e.to_string())),
                },
                Err(_) => (400, error_response("invalid world ID")),
            }
        }

        // POST /v1/worlds/{id}/plan
        ("POST", p) if p.starts_with("/v1/worlds/") && p.ends_with("/plan") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|s| s.strip_suffix("/plan"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match serde_json::from_str::<PlanRequest>(body) {
                    Ok(req) => {
                        let ws = match state.store.load(&id).await {
                            Ok(ws) => ws,
                            Err(e) => return (404, error_response(&e.to_string())),
                        };
                        if let Err(e) = state.registry.get(&req.provider) {
                            return (404, error_response(&e.to_string()));
                        }
                        let registry = Arc::clone(&state.registry);
                        let world =
                            worldforge_core::world::World::new(ws.clone(), &req.provider, registry);
                        let plan_req = worldforge_core::prediction::PlanRequest {
                            current_state: ws,
                            goal: worldforge_core::prediction::PlanGoal::Description(
                                req.goal.clone(),
                            ),
                            max_steps: req.max_steps,
                            guardrails: Vec::new(),
                            planner: worldforge_core::prediction::PlannerType::Sampling {
                                num_samples: 10,
                                top_k: 3,
                            },
                            timeout_seconds: 30.0,
                        };
                        match world.plan(&plan_req).await {
                            Ok(plan) => (200, ApiResponse::ok(plan)),
                            Err(e) => (500, error_response(&e.to_string())),
                        }
                    }
                    Err(e) => (400, error_response(&format!("invalid request: {e}"))),
                },
                Err(_) => (400, error_response("invalid world ID")),
            }
        }

        // POST /v1/worlds/{id}/verify
        ("POST", p) if p.starts_with("/v1/worlds/") && p.ends_with("/verify") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|s| s.strip_suffix("/verify"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match serde_json::from_str::<VerifyRequest>(body) {
                    Ok(req) => {
                        let ws = match state.store.load(&id).await {
                            Ok(ws) => ws,
                            Err(e) => return (404, error_response(&e.to_string())),
                        };
                        let verifier = MockVerifier::new();
                        let state_bytes = serde_json::to_vec(&ws).unwrap_or_default();

                        let proof_result = match req.proof_type.as_str() {
                            "inference" => {
                                let model_hash = worldforge_verify::sha256_hash(b"mock-model");
                                let input_hash =
                                    worldforge_verify::sha256_hash(&state_bytes);
                                let output_hash =
                                    worldforge_verify::sha256_hash(b"mock-output");
                                verifier.prove_inference(model_hash, input_hash, output_hash)
                            }
                            "guardrail" => {
                                let plan = worldforge_core::prediction::Plan {
                                    actions: Vec::new(),
                                    predicted_states: Vec::new(),
                                    predicted_videos: None,
                                    total_cost: 0.0,
                                    success_probability: 1.0,
                                    guardrail_compliance: Vec::new(),
                                    planning_time_ms: 0,
                                    iterations_used: 0,
                                };
                                verifier.prove_guardrail_compliance(&plan, &[])
                            }
                            "provenance" => {
                                let data_hash =
                                    worldforge_verify::sha256_hash(&state_bytes);
                                let timestamp = chrono::Utc::now().timestamp() as u64;
                                let source_commitment =
                                    worldforge_verify::sha256_hash(b"worldforge-server");
                                verifier.prove_data_provenance(
                                    data_hash,
                                    timestamp,
                                    source_commitment,
                                )
                            }
                            other => {
                                return (
                                    400,
                                    error_response(&format!(
                                        "unknown proof type: {other}. Available: inference, guardrail, provenance"
                                    )),
                                )
                            }
                        };

                        match proof_result {
                            Ok(proof) => {
                                let verification = verifier.verify(&proof);
                                match verification {
                                    Ok(result) => (
                                        200,
                                        ApiResponse::ok(serde_json::json!({
                                            "proof": proof,
                                            "verification": result,
                                        })),
                                    ),
                                    Err(e) => (500, error_response(&e.to_string())),
                                }
                            }
                            Err(e) => (500, error_response(&e.to_string())),
                        }
                    }
                    Err(e) => (400, error_response(&format!("invalid request: {e}"))),
                },
                Err(_) => (400, error_response("invalid world ID")),
            }
        }

        // POST /v1/worlds/{id}/evaluate
        ("POST", p) if p.starts_with("/v1/worlds/") && p.ends_with("/evaluate") => {
            let _id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|s| s.strip_suffix("/evaluate"))
                .unwrap_or("");
            match serde_json::from_str::<EvaluateRequest>(body) {
                Ok(req) => {
                    let suite = match req.suite.as_str() {
                        "physics" => EvalSuite::physics_standard(),
                        "manipulation" => EvalSuite::manipulation_standard(),
                        "spatial" => EvalSuite::spatial_reasoning(),
                        "comprehensive" => EvalSuite::comprehensive(),
                        other => {
                            return (400, error_response(&format!("unknown eval suite: {other}")))
                        }
                    };
                    // Run eval against all registered providers
                    let provider_refs: Vec<&dyn worldforge_core::provider::WorldModelProvider> =
                        state
                            .registry
                            .list()
                            .iter()
                            .filter_map(|name| state.registry.get(name).ok())
                            .collect();
                    match suite.run(&provider_refs).await {
                        Ok(report) => (200, ApiResponse::ok(report)),
                        Err(e) => (500, error_response(&e.to_string())),
                    }
                }
                Err(e) => (400, error_response(&format!("invalid request: {e}"))),
            }
        }

        // POST /v1/compare
        ("POST", "/v1/compare") => match serde_json::from_str::<CompareRequest>(body) {
            Ok(req) => {
                let id = match req.world_id.parse::<WorldId>() {
                    Ok(id) => id,
                    Err(_) => return (400, error_response("invalid world ID")),
                };
                let ws = match state.store.load(&id).await {
                    Ok(ws) => ws,
                    Err(e) => return (404, error_response(&e.to_string())),
                };
                let first_provider = req.providers.first().map(|s| s.as_str()).unwrap_or("mock");
                let registry = Arc::clone(&state.registry);
                let world = worldforge_core::world::World::new(ws, first_provider, registry);
                let provider_refs: Vec<&str> = req.providers.iter().map(|s| s.as_str()).collect();
                match world
                    .predict_multi(&req.action, &provider_refs, &req.config)
                    .await
                {
                    Ok(multi) => (200, ApiResponse::ok(multi)),
                    Err(e) => (500, error_response(&e.to_string())),
                }
            }
            Err(e) => (400, error_response(&format!("invalid request: {e}"))),
        },

        // Catch-all
        _ => (404, error_response(&format!("not found: {method} {path}"))),
    }
}

async fn send_response(
    writer: &mut (impl AsyncWriteExt + Unpin),
    status: u16,
    body: &str,
) -> anyhow::Result<()> {
    let status_text = match status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        413 => "Payload Too Large",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    };
    let response = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    writer.write_all(response.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_providers::MockProvider;

    fn test_state() -> Arc<AppState> {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(MockProvider::new()));
        Arc::new(AppState {
            registry: Arc::new(registry),
            store: FileStateStore::new(
                std::env::temp_dir().join(format!("wf-server-test-{}", uuid::Uuid::new_v4())),
            ),
            worlds: RwLock::new(HashMap::new()),
        })
    }

    #[tokio::test]
    async fn test_route_list_providers() {
        let state = test_state();
        let (status, body) = route("GET", "/v1/providers", "", &state).await;
        assert_eq!(status, 200);
        assert!(body.contains("mock"));
    }

    #[tokio::test]
    async fn test_route_create_world() {
        let state = test_state();
        let body = r#"{"name":"test world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        assert!(resp.contains("test world"));
    }

    #[tokio::test]
    async fn test_route_health_check() {
        let state = test_state();
        let (status, _) = route("GET", "/v1/providers/mock/health", "", &state).await;
        assert_eq!(status, 200);
    }

    #[tokio::test]
    async fn test_route_not_found() {
        let state = test_state();
        let (status, _) = route("GET", "/v1/nonexistent", "", &state).await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn test_route_get_world_not_found() {
        let state = test_state();
        let id = uuid::Uuid::new_v4();
        let (status, _) = route("GET", &format!("/v1/worlds/{id}"), "", &state).await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn test_route_list_worlds_empty() {
        let state = test_state();
        let (status, body) = route("GET", "/v1/worlds", "", &state).await;
        assert_eq!(status, 200);
        assert!(body.contains("[]") || body.contains("success"));
    }

    #[tokio::test]
    async fn test_route_list_worlds_with_entries() {
        let state = test_state();
        // Create a world first
        let body = r#"{"name":"w1","provider":"mock"}"#;
        let (status, _) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);

        let (status, resp) = route("GET", "/v1/worlds", "", &state).await;
        assert_eq!(status, 200);
        assert!(resp.contains("w1"));
    }

    #[tokio::test]
    async fn test_route_delete_world() {
        let state = test_state();
        // Create then delete
        let body = r#"{"name":"to_delete","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);

        // Extract ID
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = v["data"]["id"].as_str().unwrap();

        let (status, _) = route("DELETE", &format!("/v1/worlds/{id}"), "", &state).await;
        assert_eq!(status, 200);

        // Verify it's gone
        let (status, _) = route("GET", &format!("/v1/worlds/{id}"), "", &state).await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn test_route_delete_nonexistent() {
        let state = test_state();
        let id = uuid::Uuid::new_v4();
        let (status, _) = route("DELETE", &format!("/v1/worlds/{id}"), "", &state).await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn test_route_predict() {
        let state = test_state();
        let body = r#"{"name":"pred_world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = v["data"]["id"].as_str().unwrap();

        let pred_body = r#"{"action":{"Move":{"target":{"x":1.0,"y":0.0,"z":0.0},"speed":1.0}},"provider":"mock"}"#;
        let (status, _) = route(
            "POST",
            &format!("/v1/worlds/{id}/predict"),
            pred_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);
    }

    #[tokio::test]
    async fn test_route_evaluate() {
        let state = test_state();
        let body = r#"{"suite":"physics"}"#;
        let id = uuid::Uuid::new_v4();
        let (status, resp) =
            route("POST", &format!("/v1/worlds/{id}/evaluate"), body, &state).await;
        assert_eq!(status, 200);
        assert!(resp.contains("success"));
    }

    #[tokio::test]
    async fn test_route_evaluate_invalid_suite() {
        let state = test_state();
        let body = r#"{"suite":"nonexistent"}"#;
        let id = uuid::Uuid::new_v4();
        let (status, _) = route("POST", &format!("/v1/worlds/{id}/evaluate"), body, &state).await;
        assert_eq!(status, 400);
    }

    #[tokio::test]
    async fn test_route_verify_inference() {
        let state = test_state();
        let body = r#"{"name":"verify_world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = v["data"]["id"].as_str().unwrap();

        let verify_body = r#"{"proof_type":"inference"}"#;
        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/verify"),
            verify_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);
        assert!(resp.contains("proof"));
        assert!(resp.contains("verification"));

        // Check verification is valid
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(v["data"]["verification"]["valid"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_route_verify_provenance() {
        let state = test_state();
        let body = r#"{"name":"verify_prov","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = v["data"]["id"].as_str().unwrap();

        let verify_body = r#"{"proof_type":"provenance"}"#;
        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/verify"),
            verify_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(v["data"]["verification"]["valid"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_route_verify_invalid_type() {
        let state = test_state();
        let body = r#"{"name":"verify_bad","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = v["data"]["id"].as_str().unwrap();

        let verify_body = r#"{"proof_type":"nonexistent"}"#;
        let (status, _) = route(
            "POST",
            &format!("/v1/worlds/{id}/verify"),
            verify_body,
            &state,
        )
        .await;
        assert_eq!(status, 400);
    }

    #[tokio::test]
    async fn test_route_compare() {
        let state = test_state();
        // Create a world
        let body = r#"{"name":"cmp_world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = v["data"]["id"].as_str().unwrap();

        let cmp_body = format!(
            r#"{{"world_id":"{}","action":{{"Move":{{"target":{{"x":1.0,"y":0.0,"z":0.0}},"speed":1.0}}}},"providers":["mock"]}}"#,
            id
        );
        let (status, resp) = route("POST", "/v1/compare", &cmp_body, &state).await;
        assert_eq!(status, 200);
        assert!(resp.contains("success"));
    }
}
