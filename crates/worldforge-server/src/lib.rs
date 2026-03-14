//! WorldForge REST API Server
//!
//! A lightweight HTTP/JSON API server built on Tokio for interacting
//! with WorldForge functionality over the network.

use std::collections::HashMap;
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
    let state = Arc::new(AppState {
        registry,
        store: FileStateStore::new(&config.state_dir),
        worlds: RwLock::new(HashMap::new()),
    });

    let listener = TcpListener::bind(&config.bind_address).await?;
    tracing::info!(address = %config.bind_address, "WorldForge server started");

    loop {
        let (stream, addr) = listener.accept().await?;
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, state).await {
                tracing::error!(addr = %addr, error = %e, "request handling failed");
            }
        });
    }
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

        // GET /v1/worlds/{id}
        ("GET", p) if p.starts_with("/v1/worlds/") && !p.contains("/predict") => {
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
}
