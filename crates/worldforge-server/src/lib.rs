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
use worldforge_core::error::WorldForgeError;
use worldforge_core::guardrail::GuardrailConfig;
use worldforge_core::prediction::{PlannerType, PredictionConfig};
use worldforge_core::provider::{
    GenerationConfig, GenerationPrompt, Operation, ProviderRegistry, SpatialControls,
    TransferConfig,
};
use worldforge_core::scene::{PhysicsProperties, SceneObject};
use worldforge_core::state::{DynStateStore, StateStoreKind, WorldState};
use worldforge_core::types::{BBox, Pose, Position, Rotation, Velocity, VideoClip, WorldId};
use worldforge_eval::EvalSuite;
use worldforge_verify::{
    prove_guardrail_plan, prove_inference_transition, prove_latest_inference, prove_provenance,
    verify_bundle, verify_proof, MockVerifier, VerificationBundle, VerificationResult, ZkProof,
};

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to.
    pub bind_address: String,
    /// State storage directory for file mode and the default SQLite location.
    pub state_dir: String,
    /// Persistence backend to use: `file` or `sqlite`.
    pub state_backend: String,
    /// Optional SQLite database path override.
    pub state_db_path: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8080".to_string(),
            state_dir: ".worldforge".to_string(),
            state_backend: "file".to_string(),
            state_db_path: None,
        }
    }
}

impl ServerConfig {
    fn resolve_state_store_kind(&self) -> anyhow::Result<StateStoreKind> {
        match self.state_backend.as_str() {
            "file" => Ok(StateStoreKind::File(self.state_dir.clone().into())),
            "sqlite" => Ok(StateStoreKind::Sqlite(
                self.state_db_path
                    .as_deref()
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| std::path::Path::new(&self.state_dir).join("worldforge.db")),
            )),
            other => {
                anyhow::bail!("unknown state backend: {other}. Available backends: file, sqlite")
            }
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
        let store = config.resolve_state_store_kind()?.open().await?;
        let state = Arc::new(AppState {
            registry,
            store,
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
    store: DynStateStore,
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
    #[serde(default)]
    provider: Option<String>,
}

/// JSON request body for adding an object to a world scene.
#[derive(Debug, Deserialize)]
struct CreateObjectRequest {
    name: String,
    position: Position,
    bbox: BBox,
    #[serde(default)]
    rotation: Rotation,
    #[serde(default)]
    velocity: Velocity,
    #[serde(default)]
    semantic_label: Option<String>,
    #[serde(default)]
    mass: Option<f32>,
    #[serde(default)]
    friction: Option<f32>,
    #[serde(default)]
    restitution: Option<f32>,
    #[serde(default)]
    material: Option<String>,
    #[serde(default)]
    is_static: bool,
    #[serde(default)]
    is_graspable: bool,
}

/// JSON request body for planning.
#[derive(Debug, Deserialize)]
struct PlanRequest {
    goal: String,
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default = "default_timeout_seconds")]
    timeout_seconds: f64,
    #[serde(default = "default_planner_name")]
    planner: String,
    #[serde(default)]
    num_samples: Option<u32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    population_size: Option<u32>,
    #[serde(default)]
    elite_fraction: Option<f32>,
    #[serde(default)]
    num_iterations: Option<u32>,
    #[serde(default)]
    learning_rate: Option<f32>,
    #[serde(default)]
    horizon: Option<u32>,
    #[serde(default)]
    replanning_interval: Option<u32>,
    #[serde(default)]
    guardrails: Vec<GuardrailConfig>,
}

fn default_max_steps() -> u32 {
    10
}

fn default_timeout_seconds() -> f64 {
    30.0
}

fn default_planner_name() -> String {
    "sampling".to_string()
}

/// JSON request body for generation.
#[derive(Debug, Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    config: GenerationConfig,
}

/// JSON request body for provider transfer.
#[derive(Debug, Deserialize)]
struct TransferRequest {
    source: VideoClip,
    #[serde(default)]
    controls: SpatialControls,
    #[serde(default)]
    config: TransferConfig,
}

/// JSON request body for provider cost estimation.
#[derive(Debug, Deserialize)]
struct EstimateCostRequest {
    operation: Operation,
}

/// JSON request body for world-state reasoning.
#[derive(Debug, Deserialize)]
struct ReasonRequest {
    query: String,
    #[serde(default)]
    provider: Option<String>,
}

fn planner_from_request(request: &PlanRequest) -> std::result::Result<PlannerType, String> {
    match request.planner.as_str() {
        "sampling" => Ok(PlannerType::Sampling {
            num_samples: request.num_samples.unwrap_or(32).max(1),
            top_k: request.top_k.unwrap_or(5).max(1),
        }),
        "cem" => Ok(PlannerType::CEM {
            population_size: request.population_size.unwrap_or(64).max(4),
            elite_fraction: request.elite_fraction.unwrap_or(0.2).clamp(0.05, 1.0),
            num_iterations: request.num_iterations.unwrap_or(5).max(1),
        }),
        "mpc" => Ok(PlannerType::MPC {
            horizon: request
                .horizon
                .unwrap_or(request.max_steps)
                .max(1)
                .min(request.max_steps.max(1)),
            num_samples: request.num_samples.unwrap_or(32).max(4),
            replanning_interval: request.replanning_interval.unwrap_or(1).max(1),
        }),
        "gradient" => Ok(PlannerType::Gradient {
            learning_rate: request.learning_rate.unwrap_or(0.25).clamp(0.01, 1.0),
            num_iterations: request.num_iterations.unwrap_or(24).max(1),
        }),
        "provider-native" | "provider_native" | "native" => Ok(PlannerType::ProviderNative),
        other => Err(format!(
            "unknown planner: {other}. Available: sampling, cem, mpc, gradient, provider-native"
        )),
    }
}

fn planner_from_verify_request(
    request: &VerifyRequest,
) -> std::result::Result<PlannerType, String> {
    match request.planner.as_str() {
        "sampling" => Ok(PlannerType::Sampling {
            num_samples: request.num_samples.unwrap_or(32).max(1),
            top_k: request.top_k.unwrap_or(5).max(1),
        }),
        "cem" => Ok(PlannerType::CEM {
            population_size: request.population_size.unwrap_or(64).max(4),
            elite_fraction: request.elite_fraction.unwrap_or(0.2).clamp(0.05, 1.0),
            num_iterations: request.num_iterations.unwrap_or(5).max(1),
        }),
        "mpc" => Ok(PlannerType::MPC {
            horizon: request
                .horizon
                .unwrap_or(request.max_steps)
                .max(1)
                .min(request.max_steps.max(1)),
            num_samples: request.num_samples.unwrap_or(32).max(4),
            replanning_interval: request.replanning_interval.unwrap_or(1).max(1),
        }),
        "gradient" => Ok(PlannerType::Gradient {
            learning_rate: request.learning_rate.unwrap_or(0.25).clamp(0.01, 1.0),
            num_iterations: request.num_iterations.unwrap_or(24).max(1),
        }),
        "provider-native" | "provider_native" | "native" => Ok(PlannerType::ProviderNative),
        other => Err(format!(
            "unknown planner: {other}. Available: sampling, cem, mpc, gradient, provider-native"
        )),
    }
}

fn build_plan_request(
    current_state: WorldState,
    goal: String,
    max_steps: u32,
    guardrails: Vec<GuardrailConfig>,
    planner: PlannerType,
    timeout_seconds: f64,
) -> worldforge_core::prediction::PlanRequest {
    worldforge_core::prediction::PlanRequest {
        current_state,
        goal: worldforge_core::prediction::PlanGoal::Description(goal),
        max_steps,
        guardrails,
        planner,
        timeout_seconds,
    }
}

/// JSON request body for evaluation.
#[derive(Debug, Deserialize)]
struct EvaluateRequest {
    #[serde(default)]
    suite: Option<String>,
    #[serde(default)]
    suite_definition: Option<EvalSuite>,
    #[serde(default)]
    providers: Vec<String>,
}

fn resolve_eval_suite(request: &EvaluateRequest) -> std::result::Result<EvalSuite, String> {
    match &request.suite_definition {
        Some(suite) => suite
            .validate()
            .map(|_| suite.clone())
            .map_err(|e| e.to_string()),
        None => EvalSuite::from_builtin(request.suite.as_deref().unwrap_or("physics"))
            .map_err(|e| e.to_string()),
    }
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
    /// Optional provider override for planning or history-backed inference proofs.
    #[serde(default)]
    provider: Option<String>,
    /// Explicit input state for inference verification.
    #[serde(default)]
    input_state: Option<WorldState>,
    /// Explicit output state for inference/provenance verification.
    #[serde(default)]
    output_state: Option<WorldState>,
    /// Explicit plan for guardrail verification.
    #[serde(default)]
    plan: Option<worldforge_core::prediction::Plan>,
    /// Goal used to generate a plan before guardrail verification.
    #[serde(default)]
    goal: Option<String>,
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default = "default_timeout_seconds")]
    timeout_seconds: f64,
    #[serde(default = "default_planner_name")]
    planner: String,
    #[serde(default)]
    num_samples: Option<u32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    population_size: Option<u32>,
    #[serde(default)]
    elite_fraction: Option<f32>,
    #[serde(default)]
    num_iterations: Option<u32>,
    #[serde(default)]
    learning_rate: Option<f32>,
    #[serde(default)]
    horizon: Option<u32>,
    #[serde(default)]
    replanning_interval: Option<u32>,
    #[serde(default)]
    guardrails: Vec<GuardrailConfig>,
    #[serde(default = "default_source_label")]
    source_label: String,
}

/// JSON request body for standalone proof verification.
#[derive(Debug, Deserialize)]
struct VerifyProofRequest {
    #[serde(default)]
    proof: Option<ZkProof>,
    #[serde(default)]
    inference_bundle: Option<VerificationBundle<worldforge_verify::InferenceArtifact>>,
    #[serde(default)]
    guardrail_bundle: Option<VerificationBundle<worldforge_verify::GuardrailArtifact>>,
    #[serde(default)]
    provenance_bundle: Option<VerificationBundle<worldforge_verify::ProvenanceArtifact>>,
}

#[derive(Debug, Serialize)]
struct ProofVerificationReport {
    proof: ZkProof,
    verification: VerificationResult,
}

fn default_proof_type() -> String {
    "inference".to_string()
}

fn default_source_label() -> String {
    "worldforge-server".to_string()
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

fn api_error_status(error: &WorldForgeError) -> u16 {
    match error {
        WorldForgeError::ProviderNotFound(_) | WorldForgeError::WorldNotFound(_) => 404,
        WorldForgeError::UnsupportedAction { .. }
        | WorldForgeError::UnsupportedCapability { .. }
        | WorldForgeError::InvalidState(_)
        | WorldForgeError::SerializationError(_) => 400,
        WorldForgeError::GuardrailBlocked { .. } => 409,
        WorldForgeError::ProviderTimeout { .. } => 504,
        _ => 500,
    }
}

fn resolve_world_provider<'a>(state: &'a WorldState, requested: Option<&'a str>) -> &'a str {
    requested
        .filter(|provider| !provider.is_empty())
        .unwrap_or(state.metadata.created_by.as_str())
}

fn build_scene_object(request: CreateObjectRequest) -> SceneObject {
    let mut object = SceneObject::new(
        request.name,
        Pose {
            position: request.position,
            rotation: request.rotation,
        },
        request.bbox,
    );
    object.velocity = request.velocity;
    object.semantic_label = request.semantic_label;
    object.physics = PhysicsProperties {
        mass: request.mass,
        friction: request.friction,
        restitution: request.restitution,
        is_static: request.is_static,
        is_graspable: request.is_graspable,
        material: request.material,
    };
    object
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

fn split_path_and_query(path: &str) -> (&str, Option<&str>) {
    match path.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (path, None),
    }
}

fn query_param<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
    query.and_then(|query| {
        query.split('&').find_map(|pair| {
            let (candidate_key, candidate_value) = pair.split_once('=').unwrap_or((pair, ""));
            (candidate_key == key).then_some(candidate_value)
        })
    })
}

async fn route(method: &str, path: &str, body: &str, state: &AppState) -> (u16, String) {
    let (path, query) = split_path_and_query(path);
    // Trim trailing slash
    let path = path.trim_end_matches('/');

    match (method, path) {
        // POST /v1/verify/proof
        ("POST", "/v1/verify/proof") => match serde_json::from_str::<VerifyProofRequest>(body) {
            Ok(req) => {
                let verifier = MockVerifier::new();
                let provided_count = usize::from(req.proof.is_some())
                    + usize::from(req.inference_bundle.is_some())
                    + usize::from(req.guardrail_bundle.is_some())
                    + usize::from(req.provenance_bundle.is_some());
                if provided_count != 1 {
                    return (
                        400,
                        error_response(
                            "provide exactly one of proof, inference_bundle, guardrail_bundle, or provenance_bundle",
                        ),
                    );
                }

                if let Some(proof) = req.proof {
                    let verification = match verify_proof(&verifier, &proof) {
                        Ok(verification) => verification,
                        Err(error) => return (400, error_response(&error.to_string())),
                    };
                    return (
                        200,
                        ApiResponse::ok(ProofVerificationReport {
                            proof,
                            verification,
                        }),
                    );
                }

                if let Some(bundle) = req.inference_bundle {
                    return match verify_bundle(&verifier, &bundle) {
                        Ok(report) => (200, ApiResponse::ok(report)),
                        Err(error) => (400, error_response(&error.to_string())),
                    };
                }

                if let Some(bundle) = req.guardrail_bundle {
                    return match verify_bundle(&verifier, &bundle) {
                        Ok(report) => (200, ApiResponse::ok(report)),
                        Err(error) => (400, error_response(&error.to_string())),
                    };
                }

                if let Some(bundle) = req.provenance_bundle {
                    return match verify_bundle(&verifier, &bundle) {
                        Ok(report) => (200, ApiResponse::ok(report)),
                        Err(error) => (400, error_response(&error.to_string())),
                    };
                }

                (400, error_response("missing verification payload"))
            }
            Err(error) => (400, error_response(&format!("invalid request: {error}"))),
        },

        // GET /v1/providers
        ("GET", "/v1/providers") => {
            let providers = match query_param(query, "capability") {
                Some(capability) => state.registry.describe_by_capability(capability),
                None => state.registry.describe_all(),
            };
            (200, ApiResponse::ok(providers))
        }

        // GET /v1/providers/{name}
        ("GET", p) if p.starts_with("/v1/providers/") && !p.ends_with("/health") => {
            let name = p.strip_prefix("/v1/providers/").unwrap_or("");
            match state.registry.describe(name) {
                Ok(descriptor) => (200, ApiResponse::ok(descriptor)),
                Err(error) => (404, error_response(&error.to_string())),
            }
        }

        // GET /v1/evals/suites
        ("GET", "/v1/evals/suites") => {
            let suites: Vec<_> = EvalSuite::builtin_names()
                .iter()
                .map(|name| serde_json::json!({ "name": name }))
                .collect();
            (200, ApiResponse::ok(suites))
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

        // POST /v1/providers/{name}/estimate
        ("POST", p) if p.starts_with("/v1/providers/") && p.ends_with("/estimate") => {
            let provider_name = p
                .strip_prefix("/v1/providers/")
                .and_then(|value| value.strip_suffix("/estimate"))
                .unwrap_or("");
            match serde_json::from_str::<EstimateCostRequest>(body) {
                Ok(req) => match state.registry.estimate_cost(provider_name, &req.operation) {
                    Ok(estimate) => (
                        200,
                        ApiResponse::ok(serde_json::json!({
                            "provider": provider_name,
                            "operation": req.operation,
                            "estimate": estimate,
                        })),
                    ),
                    Err(error) => (404, error_response(&error.to_string())),
                },
                Err(error) => (400, error_response(&format!("invalid request: {error}"))),
            }
        }

        // POST /v1/worlds
        ("POST", "/v1/worlds") => match serde_json::from_str::<CreateWorldRequest>(body) {
            Ok(req) => {
                if let Err(error) = state.registry.get(&req.provider) {
                    return (404, error_response(&error.to_string()));
                }
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

        // POST /v1/providers/{name}/generate
        ("POST", p) if p.starts_with("/v1/providers/") && p.ends_with("/generate") => {
            let provider_name = p
                .strip_prefix("/v1/providers/")
                .and_then(|value| value.strip_suffix("/generate"))
                .unwrap_or("");
            match serde_json::from_str::<GenerateRequest>(body) {
                Ok(req) => match state.registry.get(provider_name) {
                    Ok(provider) => {
                        let prompt = GenerationPrompt {
                            text: req.prompt,
                            reference_image: None,
                            negative_prompt: req.negative_prompt,
                        };
                        match provider.generate(&prompt, &req.config).await {
                            Ok(clip) => (200, ApiResponse::ok(clip)),
                            Err(error) => {
                                (api_error_status(&error), error_response(&error.to_string()))
                            }
                        }
                    }
                    Err(error) => (404, error_response(&error.to_string())),
                },
                Err(e) => (400, error_response(&format!("invalid request: {e}"))),
            }
        }

        // POST /v1/providers/{name}/transfer
        ("POST", p) if p.starts_with("/v1/providers/") && p.ends_with("/transfer") => {
            let provider_name = p
                .strip_prefix("/v1/providers/")
                .and_then(|value| value.strip_suffix("/transfer"))
                .unwrap_or("");
            match serde_json::from_str::<TransferRequest>(body) {
                Ok(req) => match state.registry.get(provider_name) {
                    Ok(provider) => {
                        match provider
                            .transfer(&req.source, &req.controls, &req.config)
                            .await
                        {
                            Ok(clip) => (200, ApiResponse::ok(clip)),
                            Err(error) => {
                                (api_error_status(&error), error_response(&error.to_string()))
                            }
                        }
                    }
                    Err(error) => (404, error_response(&error.to_string())),
                },
                Err(e) => (400, error_response(&format!("invalid request: {e}"))),
            }
        }

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

        // GET /v1/worlds/{id}/objects
        ("GET", p) if p.starts_with("/v1/worlds/") && p.ends_with("/objects") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|value| value.strip_suffix("/objects"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match state.store.load(&id).await {
                    Ok(ws) => {
                        let provider = ws.metadata.created_by.clone();
                        let world = worldforge_core::world::World::new(
                            ws,
                            provider,
                            Arc::clone(&state.registry),
                        );
                        (200, ApiResponse::ok(world.list_objects()))
                    }
                    Err(error) => (404, error_response(&error.to_string())),
                },
                Err(_) => (400, error_response("invalid world ID")),
            }
        }

        // POST /v1/worlds/{id}/objects
        ("POST", p) if p.starts_with("/v1/worlds/") && p.ends_with("/objects") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|value| value.strip_suffix("/objects"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match serde_json::from_str::<CreateObjectRequest>(body) {
                    Ok(req) => {
                        let ws = match state.store.load(&id).await {
                            Ok(ws) => ws,
                            Err(error) => return (404, error_response(&error.to_string())),
                        };
                        let provider = ws.metadata.created_by.clone();
                        let mut world = worldforge_core::world::World::new(
                            ws,
                            provider,
                            Arc::clone(&state.registry),
                        );
                        let object = build_scene_object(req);
                        match world.add_object(object.clone()) {
                            Ok(()) => match state.store.save(world.current_state()).await {
                                Ok(()) => (201, ApiResponse::ok(object)),
                                Err(error) => (500, error_response(&error.to_string())),
                            },
                            Err(error) => {
                                (api_error_status(&error), error_response(&error.to_string()))
                            }
                        }
                    }
                    Err(error) => (400, error_response(&format!("invalid request: {error}"))),
                },
                Err(_) => (400, error_response("invalid world ID")),
            }
        }

        // GET /v1/worlds/{id}/objects/{object_id}
        ("GET", p) if p.starts_with("/v1/worlds/") && p.contains("/objects/") => {
            let Some((world_part, object_part)) = p
                .strip_prefix("/v1/worlds/")
                .and_then(|value| value.split_once("/objects/"))
            else {
                return (404, error_response(&format!("not found: {method} {path}")));
            };
            match (
                world_part.parse::<WorldId>(),
                object_part.parse::<uuid::Uuid>(),
            ) {
                (Ok(world_id), Ok(object_id)) => match state.store.load(&world_id).await {
                    Ok(ws) => {
                        let provider = ws.metadata.created_by.clone();
                        let world = worldforge_core::world::World::new(
                            ws,
                            provider,
                            Arc::clone(&state.registry),
                        );
                        match world.get_object(&object_id) {
                            Some(object) => (200, ApiResponse::ok(object.clone())),
                            None => (404, error_response("object not found")),
                        }
                    }
                    Err(error) => (404, error_response(&error.to_string())),
                },
                (Err(_), _) => (400, error_response("invalid world ID")),
                (_, Err(_)) => (400, error_response("invalid object ID")),
            }
        }

        // DELETE /v1/worlds/{id}/objects/{object_id}
        ("DELETE", p) if p.starts_with("/v1/worlds/") && p.contains("/objects/") => {
            let Some((world_part, object_part)) = p
                .strip_prefix("/v1/worlds/")
                .and_then(|value| value.split_once("/objects/"))
            else {
                return (404, error_response(&format!("not found: {method} {path}")));
            };
            match (
                world_part.parse::<WorldId>(),
                object_part.parse::<uuid::Uuid>(),
            ) {
                (Ok(world_id), Ok(object_id)) => match state.store.load(&world_id).await {
                    Ok(ws) => {
                        let provider = ws.metadata.created_by.clone();
                        let mut world = worldforge_core::world::World::new(
                            ws,
                            provider,
                            Arc::clone(&state.registry),
                        );
                        match world.remove_object(&object_id) {
                            Ok(object) => match state.store.save(world.current_state()).await {
                                Ok(()) => (200, ApiResponse::ok(object)),
                                Err(error) => (500, error_response(&error.to_string())),
                            },
                            Err(error) => {
                                (api_error_status(&error), error_response(&error.to_string()))
                            }
                        }
                    }
                    Err(error) => (404, error_response(&error.to_string())),
                },
                (Err(_), _) => (400, error_response("invalid world ID")),
                (_, Err(_)) => (400, error_response("invalid object ID")),
            }
        }

        // GET /v1/worlds/{id}
        ("GET", p)
            if p.starts_with("/v1/worlds/")
                && !p.contains("/predict")
                && !p.contains("/plan")
                && !p.contains("/objects")
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
                            let provider_name =
                                resolve_world_provider(&ws, req.provider.as_deref()).to_string();
                            let mut world = worldforge_core::world::World::new(
                                ws,
                                provider_name,
                                Arc::clone(&state.registry),
                            );
                            match world.predict(&req.action, &req.config).await {
                                Ok(prediction) => {
                                    // Save updated state
                                    let _ = state.store.save(world.current_state()).await;
                                    (200, ApiResponse::ok(prediction))
                                }
                                Err(e) => (api_error_status(&e), error_response(&e.to_string())),
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

        // POST /v1/worlds/{id}/reason
        ("POST", p) if p.starts_with("/v1/worlds/") && p.ends_with("/reason") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|value| value.strip_suffix("/reason"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match serde_json::from_str::<ReasonRequest>(body) {
                    Ok(req) => {
                        let ws = match state.store.load(&id).await {
                            Ok(ws) => ws,
                            Err(e) => return (404, error_response(&e.to_string())),
                        };
                        let provider_name =
                            resolve_world_provider(&ws, req.provider.as_deref()).to_string();
                        let world = worldforge_core::world::World::new(
                            ws,
                            provider_name,
                            Arc::clone(&state.registry),
                        );
                        match world.reason(&req.query).await {
                            Ok(output) => (200, ApiResponse::ok(output)),
                            Err(error) => {
                                (api_error_status(&error), error_response(&error.to_string()))
                            }
                        }
                    }
                    Err(e) => (400, error_response(&format!("invalid request: {e}"))),
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
                        let provider_name =
                            resolve_world_provider(&ws, req.provider.as_deref()).to_string();
                        if let Err(e) = state.registry.get(&provider_name) {
                            return (404, error_response(&e.to_string()));
                        }
                        let planner = match planner_from_request(&req) {
                            Ok(planner) => planner,
                            Err(error) => return (400, error_response(&error)),
                        };
                        let registry = Arc::clone(&state.registry);
                        let world = worldforge_core::world::World::new(
                            ws.clone(),
                            &provider_name,
                            registry,
                        );
                        let plan_req = build_plan_request(
                            ws,
                            req.goal.clone(),
                            req.max_steps,
                            req.guardrails,
                            planner,
                            req.timeout_seconds,
                        );
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
                        match req.proof_type.as_str() {
                            "inference" => {
                                let bundle = match (req.input_state.as_ref(), req.output_state.as_ref()) {
                                    (Some(input_state), Some(output_state)) => {
                                        let provider_name = req
                                            .provider
                                            .as_deref()
                                            .filter(|name| !name.is_empty())
                                            .unwrap_or(output_state.metadata.created_by.as_str());
                                        prove_inference_transition(
                                            &verifier,
                                            provider_name,
                                            input_state,
                                            output_state,
                                        )
                                    }
                                    (None, None) => {
                                        prove_latest_inference(&verifier, &ws, req.provider.as_deref())
                                    }
                                    _ => {
                                        return (
                                            400,
                                            error_response(
                                                "inference verification requires both input_state and output_state when either is provided",
                                            ),
                                        )
                                    }
                                };
                                match bundle {
                                    Ok(bundle) => (200, ApiResponse::ok(bundle)),
                                    Err(e) => (400, error_response(&e.to_string())),
                                }
                            }
                            "guardrail" => {
                                let plan = if let Some(plan) = req.plan.clone() {
                                    plan
                                } else {
                                    let goal = match req.goal.clone() {
                                        Some(goal) => goal,
                                        None => {
                                            return (
                                                400,
                                                error_response(
                                                    "guardrail verification requires either a plan or a goal",
                                                ),
                                            )
                                        }
                                    };
                                    let provider_name =
                                        resolve_world_provider(&ws, req.provider.as_deref()).to_string();
                                    if let Err(e) = state.registry.get(&provider_name) {
                                        return (404, error_response(&e.to_string()));
                                    }
                                    let planner = match planner_from_verify_request(&req) {
                                        Ok(planner) => planner,
                                        Err(error) => return (400, error_response(&error)),
                                    };
                                    let registry = Arc::clone(&state.registry);
                                    let world = worldforge_core::world::World::new(
                                        ws.clone(),
                                        &provider_name,
                                        registry,
                                    );
                                    let plan_req = build_plan_request(
                                        ws.clone(),
                                        goal,
                                        req.max_steps,
                                        req.guardrails,
                                        planner,
                                        req.timeout_seconds,
                                    );
                                    match world.plan(&plan_req).await {
                                        Ok(plan) => plan,
                                        Err(e) => return (500, error_response(&e.to_string())),
                                    }
                                };

                                match prove_guardrail_plan(&verifier, &plan) {
                                    Ok(bundle) => (200, ApiResponse::ok(bundle)),
                                    Err(e) => (500, error_response(&e.to_string())),
                                }
                            }
                            "provenance" => {
                                let target_state = req.output_state.as_ref().unwrap_or(&ws);
                                let timestamp = chrono::Utc::now().timestamp() as u64;
                                match prove_provenance(
                                    &verifier,
                                    target_state,
                                    &req.source_label,
                                    timestamp,
                                ) {
                                    Ok(bundle) => (200, ApiResponse::ok(bundle)),
                                    Err(e) => (500, error_response(&e.to_string())),
                                }
                            }
                            other => (
                                400,
                                error_response(&format!(
                                    "unknown proof type: {other}. Available: inference, guardrail, provenance"
                                )),
                            ),
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
                    let suite = match resolve_eval_suite(&req) {
                        Ok(suite) => suite,
                        Err(error) => return (400, error_response(&error)),
                    };

                    let provider_names = if req.providers.is_empty() {
                        state
                            .registry
                            .list()
                            .into_iter()
                            .map(str::to_string)
                            .collect::<Vec<_>>()
                    } else {
                        req.providers
                    };

                    let mut provider_refs: Vec<&dyn worldforge_core::provider::WorldModelProvider> =
                        Vec::new();
                    for provider_name in &provider_names {
                        match state.registry.get(provider_name) {
                            Ok(provider) => provider_refs.push(provider),
                            Err(error) => return (404, error_response(&error.to_string())),
                        }
                    }

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
        409 => "Conflict",
        413 => "Payload Too Large",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
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
    use worldforge_core::state::FileStateStore;
    use worldforge_providers::MockProvider;
    use worldforge_verify::ZkVerifier;

    fn test_state() -> Arc<AppState> {
        test_state_with_provider("mock")
    }

    fn test_state_with_provider(provider_name: &str) -> Arc<AppState> {
        test_state_with_providers(&[provider_name])
    }

    fn test_state_with_providers(provider_names: &[&str]) -> Arc<AppState> {
        let mut registry = ProviderRegistry::new();
        for provider_name in provider_names {
            registry.register(Box::new(MockProvider::with_name(*provider_name)));
        }
        Arc::new(AppState {
            registry: Arc::new(registry),
            store: Arc::new(FileStateStore::new(
                std::env::temp_dir().join(format!("wf-server-test-{}", uuid::Uuid::new_v4())),
            )),
            worlds: RwLock::new(HashMap::new()),
        })
    }

    #[test]
    fn test_server_config_resolves_file_store() {
        let config = ServerConfig {
            state_dir: "/tmp/worldforge".to_string(),
            ..ServerConfig::default()
        };

        assert_eq!(
            config.resolve_state_store_kind().unwrap(),
            StateStoreKind::File("/tmp/worldforge".into())
        );
    }

    #[test]
    fn test_server_config_resolves_sqlite_store() {
        let config = ServerConfig {
            state_dir: "/tmp/worldforge".to_string(),
            state_backend: "sqlite".to_string(),
            state_db_path: None,
            ..ServerConfig::default()
        };

        assert_eq!(
            config.resolve_state_store_kind().unwrap(),
            StateStoreKind::Sqlite("/tmp/worldforge/worldforge.db".into())
        );
    }

    #[tokio::test]
    async fn test_route_list_providers() {
        let state = test_state();
        let (status, body) = route("GET", "/v1/providers", "", &state).await;
        assert_eq!(status, 200);
        assert!(body.contains("mock"));
        assert!(body.contains("capabilities"));
    }

    #[tokio::test]
    async fn test_route_list_providers_with_capability_filter() {
        let state = test_state();
        let (status, body) = route("GET", "/v1/providers?capability=predict", "", &state).await;
        assert_eq!(status, 200);
        assert!(body.contains("mock"));
    }

    #[tokio::test]
    async fn test_route_get_provider_descriptor() {
        let state = test_state();
        let (status, body) = route("GET", "/v1/providers/mock", "", &state).await;
        assert_eq!(status, 200);
        assert!(body.contains("\"name\":\"mock\""));
        assert!(body.contains("\"predict\":true"));
    }

    #[tokio::test]
    async fn test_route_list_eval_suites() {
        let state = test_state();
        let (status, body) = route("GET", "/v1/evals/suites", "", &state).await;
        assert_eq!(status, 200);
        assert!(body.contains("physics"));
        assert!(body.contains("comprehensive"));
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
    async fn test_route_create_world_requires_registered_provider() {
        let state = test_state();
        let body = r#"{"name":"bad world","provider":"missing"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 404);
        assert!(resp.contains("provider not found"));
    }

    #[tokio::test]
    async fn test_route_health_check() {
        let state = test_state();
        let (status, _) = route("GET", "/v1/providers/mock/health", "", &state).await;
        assert_eq!(status, 200);
    }

    #[tokio::test]
    async fn test_route_estimate_cost() {
        let state = test_state();
        let body = r#"{"operation":{"Generate":{"duration_seconds":5.0,"resolution":[640,360]}}}"#;
        let (status, resp) = route("POST", "/v1/providers/mock/estimate", body, &state).await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["provider"], "mock");
        assert_eq!(
            value["data"]["estimate"]["estimated_latency_ms"].as_u64(),
            Some(10)
        );
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
    async fn test_route_add_and_list_objects() {
        let state = test_state();
        let body = r#"{"name":"object-world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();

        let object_body = serde_json::json!({
            "name": "crate",
            "position": { "x": 0.0, "y": 1.0, "z": 2.0 },
            "bbox": {
                "min": { "x": -0.5, "y": -0.5, "z": -0.5 },
                "max": { "x": 0.5, "y": 0.5, "z": 0.5 }
            },
            "velocity": { "x": 0.1, "y": 0.0, "z": 0.0 },
            "semantic_label": "storage",
            "mass": 5.0,
            "is_static": true,
            "is_graspable": true,
            "material": "wood"
        })
        .to_string();

        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/objects"),
            &object_body,
            &state,
        )
        .await;
        assert_eq!(status, 201);
        let created: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(created["data"]["name"], "crate");

        let (status, resp) = route("GET", &format!("/v1/worlds/{id}/objects"), "", &state).await;
        assert_eq!(status, 200);
        let listed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let objects = listed["data"].as_array().unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0]["semantic_label"], "storage");
    }

    #[tokio::test]
    async fn test_route_get_and_delete_object() {
        let state = test_state();
        let body = r#"{"name":"object-world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();

        let object_body = serde_json::json!({
            "name": "crate",
            "position": { "x": 0.0, "y": 1.0, "z": 2.0 },
            "bbox": {
                "min": { "x": -0.5, "y": -0.5, "z": -0.5 },
                "max": { "x": 0.5, "y": 0.5, "z": 0.5 }
            }
        })
        .to_string();

        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/objects"),
            &object_body,
            &state,
        )
        .await;
        assert_eq!(status, 201);
        let created: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let object_id = created["data"]["id"].as_str().unwrap();

        let (status, resp) = route(
            "GET",
            &format!("/v1/worlds/{id}/objects/{object_id}"),
            "",
            &state,
        )
        .await;
        assert_eq!(status, 200);
        let shown: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(shown["data"]["id"], object_id);

        let (status, _) = route(
            "DELETE",
            &format!("/v1/worlds/{id}/objects/{object_id}"),
            "",
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let (status, resp) = route("GET", &format!("/v1/worlds/{id}/objects"), "", &state).await;
        assert_eq!(status, 200);
        let listed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(listed["data"].as_array().unwrap().is_empty());
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
        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/predict"),
            pred_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let prediction: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(prediction["data"]["provider"], "mock");

        let (status, resp) = route("GET", &format!("/v1/worlds/{id}"), "", &state).await;
        assert_eq!(status, 200);

        let world: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(world["data"]["time"]["step"], 1);
        assert_eq!(
            world["data"]["history"]["states"].as_array().unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn test_route_predict_defaults_to_world_provider() {
        let state = test_state_with_provider("alt-mock");
        let body = r#"{"name":"alt world","provider":"alt-mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();

        let pred_body = r#"{"action":{"Move":{"target":{"x":1.0,"y":0.0,"z":0.0},"speed":1.0}}}"#;
        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/predict"),
            pred_body,
            &state,
        )
        .await;

        assert_eq!(status, 200);
        let prediction: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(prediction["data"]["provider"], "alt-mock");
    }

    #[tokio::test]
    async fn test_route_predict_uses_fallback_provider() {
        let state = test_state();
        let body = r#"{"name":"fallback_world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = v["data"]["id"].as_str().unwrap();

        let pred_body = r#"{"action":{"Move":{"target":{"x":1.0,"y":0.0,"z":0.0},"speed":1.0}},"provider":"missing","config":{"fallback_provider":"mock"}}"#;
        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/predict"),
            pred_body,
            &state,
        )
        .await;

        assert_eq!(status, 200);
        let prediction: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(prediction["data"]["provider"], "mock");
    }

    #[tokio::test]
    async fn test_route_evaluate() {
        let state = test_state();
        let body = r#"{"suite":"physics"}"#;
        let id = uuid::Uuid::new_v4();
        let (status, resp) =
            route("POST", &format!("/v1/worlds/{id}/evaluate"), body, &state).await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["suite"], "Physics Standard");
        assert_eq!(value["data"]["provider_summaries"][0]["provider"], "mock");
        assert_eq!(
            value["data"]["dimension_summaries"][0]["dimension"],
            "object_permanence"
        );
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
    async fn test_route_evaluate_custom_suite_for_selected_provider() {
        let state = test_state_with_providers(&["mock", "alt-mock"]);
        let suite = EvalSuite::physics_standard();
        let body = serde_json::json!({
            "suite_definition": suite,
            "providers": ["alt-mock"],
        })
        .to_string();
        let id = uuid::Uuid::new_v4();
        let (status, resp) =
            route("POST", &format!("/v1/worlds/{id}/evaluate"), &body, &state).await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let leaderboard = value["data"]["leaderboard"].as_array().unwrap();
        assert_eq!(leaderboard.len(), 1);
        assert_eq!(leaderboard[0]["provider"], "alt-mock");
        assert_eq!(
            value["data"]["provider_summaries"][0]["provider"],
            "alt-mock"
        );
    }

    #[tokio::test]
    async fn test_route_generate() {
        let state = test_state();
        let body =
            r#"{"prompt":"a cube rolling across the floor","config":{"duration_seconds":5.0}}"#;
        let (status, resp) = route("POST", "/v1/providers/mock/generate", body, &state).await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["duration"], 5.0);
        assert_eq!(value["data"]["fps"], 24.0);
    }

    #[tokio::test]
    async fn test_route_transfer() {
        let state = test_state();
        let body = r#"{
            "source":{"frames":[],"fps":15.0,"resolution":[320,240],"duration":1.5},
            "controls":{},
            "config":{"resolution":[800,600],"fps":24.0,"control_strength":0.7}
        }"#;
        let (status, resp) = route("POST", "/v1/providers/mock/transfer", body, &state).await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["resolution"], serde_json::json!([800, 600]));
        assert_eq!(value["data"]["fps"], 24.0);
        assert_eq!(value["data"]["duration"], 1.5);
    }

    #[tokio::test]
    async fn test_route_reason() {
        let state = test_state();
        let body = r#"{"name":"reason_world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();

        let reason_body = r#"{"query":"what happens next?"}"#;
        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/reason"),
            reason_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(value["data"]["answer"].as_str().unwrap().contains("empty"));
    }

    #[tokio::test]
    async fn test_route_verify_inference() {
        let state = test_state();
        let body = r#"{"name":"verify_world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = v["data"]["id"].as_str().unwrap();

        let input_state = WorldState::new("input", "mock");
        let output_state = WorldState::new("output", "mock");
        let verify_body = serde_json::json!({
            "proof_type": "inference",
            "input_state": input_state,
            "output_state": output_state,
            "provider": "mock",
        })
        .to_string();
        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/verify"),
            &verify_body,
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
    async fn test_route_verify_guardrail_from_goal() {
        let state = test_state();
        let body = r#"{"name":"verify_guardrail","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();

        let verify_body = serde_json::json!({
            "proof_type": "guardrail",
            "goal": "spawn cube",
            "guardrails": [
                {
                    "guardrail": "NoCollisions",
                    "blocking": true
                }
            ]
        })
        .to_string();
        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/verify"),
            &verify_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(value["data"]["verification"]["valid"].as_bool().unwrap());
        assert!(value["data"]["artifact"]["plan_hash"].is_array());
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
    async fn test_route_verify_proof_endpoint_accepts_raw_proof() {
        let state = test_state();
        let verifier = MockVerifier::new();
        let proof = verifier.prove_inference([1; 32], [2; 32], [3; 32]).unwrap();
        let body = serde_json::json!({ "proof": proof }).to_string();

        let (status, resp) = route("POST", "/v1/verify/proof", &body, &state).await;

        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["verification"]["valid"], true);
        assert_eq!(value["data"]["proof"]["backend"], "Mock");
    }

    #[tokio::test]
    async fn test_route_verify_proof_endpoint_accepts_bundle() {
        let state = test_state();
        let input_state = WorldState::new("input", "mock");
        let output_state = WorldState::new("output", "mock");
        let verifier = MockVerifier::new();
        let bundle =
            prove_inference_transition(&verifier, "mock", &input_state, &output_state).unwrap();
        let body = serde_json::json!({ "inference_bundle": bundle }).to_string();

        let (status, resp) = route("POST", "/v1/verify/proof", &body, &state).await;

        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["current_verification"]["valid"], true);
        assert_eq!(value["data"]["verification_matches_recorded"], true);
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
