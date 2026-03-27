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
use worldforge_core::guardrail::{Guardrail, GuardrailConfig};
use worldforge_core::prediction::{Plan, PlanGoal, PlanGoalInput, PlannerType, PredictionConfig};
use worldforge_core::provider::{
    EmbeddingInput, GenerationConfig, GenerationPrompt, Operation, ProviderRegistry,
    ReasoningInput, SpatialControls, TransferConfig,
};
use worldforge_core::scene::{PhysicsProperties, SceneObject, SceneObjectPatch};
use worldforge_core::state::{
    deserialize_world_state, serialize_world_state, DynStateStore, S3Config, StateFileFormat,
    StateStoreKind, WorldState,
};
use worldforge_core::types::{
    BBox, Mesh, Pose, Position, Rotation, Tensor, Velocity, VideoClip, WorldId,
};
use worldforge_core::world::World;
use worldforge_eval::{EvalReportFormat, EvalSuite};
use worldforge_verify::{
    prove_guardrail_plan, prove_inference_transition, prove_latest_inference, prove_provenance,
    sha256_hash, verifier_for_backend as verify_backend_resolver, verify_bundle, verify_proof,
    VerificationBackend, VerificationBundle, VerificationResult, ZkProof, ZkVerifier,
};

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to.
    pub bind_address: String,
    /// State storage directory for file mode and the default SQLite location.
    pub state_dir: String,
    /// Persistence backend to use: `file`, `sqlite`, `redis`, or `s3`.
    pub state_backend: String,
    /// Serialization format for the file-backed state store.
    pub state_file_format: String,
    /// Optional SQLite database path override.
    pub state_db_path: Option<String>,
    /// Optional Redis connection URL override.
    pub state_redis_url: Option<String>,
    /// Optional S3 bucket name override.
    pub state_s3_bucket: Option<String>,
    /// Optional S3 region override.
    pub state_s3_region: Option<String>,
    /// Optional S3 access key ID override.
    pub state_s3_access_key_id: Option<String>,
    /// Optional S3 secret access key override.
    pub state_s3_secret_access_key: Option<String>,
    /// Optional S3 endpoint override.
    pub state_s3_endpoint: Option<String>,
    /// Optional S3 session token override.
    pub state_s3_session_token: Option<String>,
    /// Optional S3 object-key prefix override.
    pub state_s3_prefix: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8080".to_string(),
            state_dir: ".worldforge".to_string(),
            state_backend: "file".to_string(),
            state_file_format: "json".to_string(),
            state_db_path: None,
            state_redis_url: None,
            state_s3_bucket: None,
            state_s3_region: None,
            state_s3_access_key_id: None,
            state_s3_secret_access_key: None,
            state_s3_endpoint: None,
            state_s3_session_token: None,
            state_s3_prefix: None,
        }
    }
}

impl ServerConfig {
    fn resolve_state_store_kind(&self) -> anyhow::Result<StateStoreKind> {
        match self.state_backend.as_str() {
            "file" => {
                let format = self
                    .state_file_format
                    .parse::<StateFileFormat>()
                    .map_err(anyhow::Error::msg)?;
                Ok(StateStoreKind::FileWithFormat {
                    path: self.state_dir.clone().into(),
                    format,
                })
            }
            "sqlite" => Ok(StateStoreKind::Sqlite(
                self.state_db_path
                    .as_deref()
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| std::path::Path::new(&self.state_dir).join("worldforge.db")),
            )),
            "redis" => self
                .state_redis_url
                .clone()
                .map(StateStoreKind::Redis)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "--state-redis-url is required when state_backend is set to redis"
                    )
                }),
            "s3" => Ok(StateStoreKind::S3 {
                config: resolve_s3_config(self)?,
                format: self
                    .state_file_format
                    .parse::<StateFileFormat>()
                    .map_err(anyhow::Error::msg)?,
            }),
            other => {
                anyhow::bail!(
                    "unknown state backend: {other}. Available backends: file, sqlite, redis, s3"
                )
            }
        }
    }
}

fn resolve_s3_config(config: &ServerConfig) -> anyhow::Result<S3Config> {
    let bucket = config.state_s3_bucket.as_deref().ok_or_else(|| {
        anyhow::anyhow!("--state-s3-bucket is required when state_backend is set to s3")
    })?;
    let region = config.state_s3_region.as_deref().ok_or_else(|| {
        anyhow::anyhow!("--state-s3-region is required when state_backend is set to s3")
    })?;
    let access_key_id = config.state_s3_access_key_id.as_deref().ok_or_else(|| {
        anyhow::anyhow!("--state-s3-access-key-id is required when state_backend is set to s3")
    })?;
    let secret_access_key = config
        .state_s3_secret_access_key
        .as_deref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--state-s3-secret-access-key is required when state_backend is set to s3"
            )
        })?;

    Ok(S3Config {
        bucket: bucket.to_string(),
        region: region.to_string(),
        endpoint: config.state_s3_endpoint.clone(),
        access_key_id: access_key_id.to_string(),
        secret_access_key: secret_access_key.to_string(),
        session_token: config.state_s3_session_token.clone(),
        prefix: config.state_s3_prefix.clone().unwrap_or_default(),
    })
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
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    provider: String,
}

/// JSON request body for importing a serialized world snapshot.
#[derive(Debug, Deserialize)]
struct ImportWorldRequest {
    /// Serialized world snapshot to import.
    #[serde(default)]
    state: Option<WorldState>,
    /// Snapshot format when importing from an exported payload.
    #[serde(default)]
    format: Option<String>,
    /// Snapshot payload when importing from an exported payload.
    #[serde(default)]
    snapshot: Option<String>,
    /// Snapshot encoding when importing from an exported payload.
    #[serde(default)]
    encoding: Option<String>,
    /// SHA-256 digest for the raw serialized snapshot bytes.
    #[serde(default)]
    sha256: Option<String>,
    /// Assign a new world ID before persisting the snapshot.
    #[serde(default)]
    new_id: bool,
    /// Optional replacement world name.
    #[serde(default)]
    name: Option<String>,
}

/// JSON request body for prediction.
#[derive(Debug, Deserialize)]
struct PredictRequest {
    action: Action,
    #[serde(default)]
    config: PredictionConfig,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    disable_guardrails: bool,
}

/// JSON request body for adding an object to a world scene.
#[derive(Debug, Deserialize)]
struct CreateObjectRequest {
    name: String,
    position: Position,
    bbox: BBox,
    #[serde(default)]
    mesh: Option<Mesh>,
    #[serde(default)]
    rotation: Rotation,
    #[serde(default)]
    velocity: Velocity,
    #[serde(default)]
    semantic_label: Option<String>,
    #[serde(default)]
    visual_embedding: Option<Tensor>,
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
    goal: PlanGoalInput,
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    fallback_provider: Option<String>,
    #[serde(default)]
    verification_backend: Option<String>,
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
    #[serde(default)]
    disable_guardrails: bool,
}

/// JSON request body for plan execution.
#[derive(Debug, Deserialize)]
struct ExecutePlanRequest {
    plan: Plan,
    #[serde(default)]
    config: PredictionConfig,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    disable_guardrails: bool,
}

/// JSON request body for restoring a world to a history checkpoint.
#[derive(Debug, Deserialize)]
struct RestoreWorldRequest {
    history_index: usize,
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
    fallback_provider: Option<String>,
    #[serde(default)]
    config: GenerationConfig,
}

/// JSON request body for provider embeddings.
#[derive(Debug, Deserialize)]
struct EmbedRequest {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    video: Option<VideoClip>,
    #[serde(default)]
    fallback_provider: Option<String>,
}

/// JSON request body for provider reasoning.
#[derive(Debug, Deserialize)]
struct ProviderReasonRequest {
    query: String,
    #[serde(default)]
    state: Option<WorldState>,
    #[serde(default)]
    video: Option<VideoClip>,
    #[serde(default)]
    fallback_provider: Option<String>,
}

/// JSON request body for provider transfer.
#[derive(Debug, Deserialize)]
struct TransferRequest {
    source: VideoClip,
    #[serde(default)]
    controls: SpatialControls,
    #[serde(default)]
    fallback_provider: Option<String>,
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
    #[serde(default)]
    fallback_provider: Option<String>,
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
    goal: PlanGoal,
    max_steps: u32,
    guardrails: Vec<GuardrailConfig>,
    planner: PlannerType,
    timeout_seconds: f64,
    fallback_provider: Option<String>,
) -> worldforge_core::prediction::PlanRequest {
    worldforge_core::prediction::PlanRequest {
        current_state,
        goal,
        max_steps,
        guardrails,
        planner,
        timeout_seconds,
        fallback_provider,
    }
}

fn resolve_guardrails(
    guardrails: Vec<GuardrailConfig>,
    disable_guardrails: bool,
) -> Vec<GuardrailConfig> {
    if disable_guardrails {
        vec![GuardrailConfig {
            guardrail: Guardrail::Disabled,
            blocking: false,
        }]
    } else {
        guardrails
    }
}

/// JSON request body for evaluation.
#[derive(Debug, Deserialize)]
struct EvaluateRequest {
    /// Optional built-in evaluation suite name.
    #[serde(default)]
    suite: Option<String>,
    /// Optional inline evaluation suite definition.
    #[serde(default)]
    suite_definition: Option<EvalSuite>,
    /// Optional explicit provider override; otherwise suite defaults are used.
    #[serde(default)]
    providers: Vec<String>,
    /// Optional rendered report format. Defaults to full JSON report data.
    #[serde(default)]
    report_format: Option<EvalReportFormat>,
    /// Optional persisted world ID whose state overlays each scenario fixture.
    #[serde(default)]
    world_id: Option<String>,
    /// Optional inline world state whose data overlays each scenario fixture.
    #[serde(default)]
    world_state: Option<WorldState>,
}

/// Rendered evaluation report artifact returned for non-JSON exports.
#[derive(Debug, Serialize)]
struct RenderedEvalReport {
    /// Suite name used for the evaluation.
    suite: String,
    /// Requested report format.
    format: EvalReportFormat,
    /// Rendered report body.
    content: String,
}

#[derive(Debug)]
struct ParsedRequest {
    method: String,
    path: String,
    body: String,
}

#[derive(Debug)]
struct WireResponse {
    status: u16,
    content_type: String,
    body: String,
    headers: Vec<(&'static str, String)>,
    content_length: Option<usize>,
}

impl WireResponse {
    fn json(status: u16, body: String) -> Self {
        Self {
            status,
            content_type: "application/json; charset=utf-8".to_string(),
            body,
            headers: Vec::new(),
            content_length: None,
        }
    }

    fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    fn without_body(mut self) -> Self {
        self.content_length = Some(self.body.len());
        self.body.clear();
        self
    }
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

fn render_eval_report_response(
    report: worldforge_eval::EvalReport,
    format: EvalReportFormat,
) -> (u16, String) {
    match format {
        EvalReportFormat::Json => (200, ApiResponse::ok(report)),
        other => match report.render(other) {
            Ok(content) => (
                200,
                ApiResponse::ok(RenderedEvalReport {
                    suite: report.suite.clone(),
                    format: other,
                    content,
                }),
            ),
            Err(error) => (500, error_response(&error.to_string())),
        },
    }
}

async fn resolve_eval_world_state(
    request: &EvaluateRequest,
    state: &AppState,
) -> std::result::Result<Option<WorldState>, (u16, String)> {
    match (&request.world_id, &request.world_state) {
        (Some(_), Some(_)) => Err((
            400,
            error_response("provide at most one of world_id or world_state"),
        )),
        (Some(world_id), None) => {
            let id = world_id
                .parse::<WorldId>()
                .map_err(|_| (400, error_response("invalid world ID")))?;
            state
                .store
                .load(&id)
                .await
                .map(Some)
                .map_err(|error| (404, error_response(&error.to_string())))
        }
        (None, Some(world_state)) => Ok(Some(world_state.clone())),
        (None, None) => Ok(None),
    }
}

async fn run_evaluation_request(
    request: &EvaluateRequest,
    state: &AppState,
    world_state: Option<&WorldState>,
) -> (u16, String) {
    let suite = match resolve_eval_suite(request) {
        Ok(suite) => suite,
        Err(error) => return (400, error_response(&error)),
    };

    let provider_names = suite.effective_provider_names(&request.providers);
    let mut provider_refs = Vec::with_capacity(provider_names.len());
    for provider_name in &provider_names {
        match state.registry.get(provider_name) {
            Ok(provider) => provider_refs.push(provider),
            Err(error) => return (404, error_response(&error.to_string())),
        }
    }

    let report = match world_state {
        Some(world_state) => {
            suite
                .run_with_world_state(&provider_refs, world_state)
                .await
        }
        None => suite.run(&provider_refs).await,
    };

    match report {
        Ok(report) => render_eval_report_response(
            report,
            request.report_format.unwrap_or(EvalReportFormat::Json),
        ),
        Err(error) => (500, error_response(&error.to_string())),
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
    #[serde(default)]
    disable_guardrails: bool,
}

/// JSON request body for ZK verification.
#[derive(Debug, Deserialize)]
struct VerifyRequest {
    /// Verification backend used to generate the proof.
    #[serde(default = "default_verification_backend")]
    backend: VerificationBackend,
    /// Proof type: "inference", "guardrail", or "provenance".
    #[serde(default = "default_proof_type")]
    proof_type: String,
    /// Optional provider override for planning or history-backed inference proofs.
    #[serde(default)]
    provider: Option<String>,
    /// Optional fallback provider when planning is generated for verification.
    #[serde(default)]
    fallback_provider: Option<String>,
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
    goal: Option<PlanGoalInput>,
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
    #[serde(default)]
    disable_guardrails: bool,
    #[serde(default = "default_source_label")]
    source_label: String,
}

/// JSON request body for standalone proof verification.
#[derive(Debug, Deserialize)]
struct VerifyProofRequest {
    #[serde(default)]
    backend: Option<VerificationBackend>,
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

fn default_verification_backend() -> VerificationBackend {
    VerificationBackend::Mock
}

fn default_source_label() -> String {
    "worldforge-server".to_string()
}

fn verifier_for_backend(backend: VerificationBackend) -> Box<dyn ZkVerifier> {
    verify_backend_resolver(backend)
}

fn parse_requested_verification_backend(
    backend: Option<&str>,
) -> std::result::Result<Option<VerificationBackend>, String> {
    backend
        .map(|value| {
            value
                .parse::<VerificationBackend>()
                .map_err(|e| e.to_string())
        })
        .transpose()
}

fn core_proof_from_verify_proof(
    proof: &ZkProof,
) -> std::result::Result<worldforge_core::proof::ZkProof, String> {
    let bytes = serde_json::to_vec(proof).map_err(|error| error.to_string())?;
    serde_json::from_slice(&bytes).map_err(|error| error.to_string())
}

fn attach_plan_verification(
    plan: &mut Plan,
    backend: Option<VerificationBackend>,
) -> std::result::Result<bool, String> {
    let Some(backend) = backend else {
        return Ok(false);
    };

    let verifier = verifier_for_backend(backend);
    let bundle =
        prove_guardrail_plan(verifier.as_ref(), plan).map_err(|error| error.to_string())?;
    plan.verification_proof = Some(core_proof_from_verify_proof(&bundle.proof)?);
    Ok(true)
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

fn encode_snapshot_payload(
    format: StateFileFormat,
    state: &WorldState,
) -> std::result::Result<SnapshotPayload, String> {
    let bytes = serialize_world_state(format, state).map_err(|error| error.to_string())?;
    match format {
        StateFileFormat::Json => Ok(SnapshotPayload {
            format: format.as_str().to_string(),
            encoding: "utf-8".to_string(),
            sha256: encode_hex(&sha256_hash(&bytes)),
            snapshot: String::from_utf8(bytes)
                .map_err(|error| format!("invalid JSON snapshot encoding: {error}"))?,
        }),
        StateFileFormat::MessagePack => Ok(SnapshotPayload {
            format: format.as_str().to_string(),
            encoding: "hex".to_string(),
            sha256: encode_hex(&sha256_hash(&bytes)),
            snapshot: encode_hex(&bytes),
        }),
    }
}

fn decode_snapshot_payload(
    format: StateFileFormat,
    snapshot: &str,
    encoding: Option<&str>,
    sha256: Option<&str>,
) -> std::result::Result<Vec<u8>, String> {
    let expected_encoding = match format {
        StateFileFormat::Json => "utf-8",
        StateFileFormat::MessagePack => "hex",
    };

    if let Some(encoding) = encoding {
        if !encoding.eq_ignore_ascii_case(expected_encoding) {
            return Err(format!(
                "snapshot encoding '{encoding}' does not match format '{}'",
                format.as_str()
            ));
        }
    }

    let bytes = match format {
        StateFileFormat::Json => snapshot.as_bytes().to_vec(),
        StateFileFormat::MessagePack => decode_hex(snapshot)?,
    };

    if let Some(expected_sha256) = sha256 {
        let actual_sha256 = encode_hex(&sha256_hash(&bytes));
        if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
            return Err(format!(
                "snapshot sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            ));
        }
    }

    Ok(bytes)
}

#[derive(Debug, Clone, Serialize)]
struct SnapshotPayload {
    format: String,
    encoding: String,
    sha256: String,
    snapshot: String,
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn decode_hex(value: &str) -> std::result::Result<Vec<u8>, String> {
    fn hex_value(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    let bytes = value.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return Err("msgpack snapshot must contain an even number of hex characters".to_string());
    }

    let mut decoded = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let hi = hex_value(chunk[0]).ok_or_else(|| {
            format!(
                "msgpack snapshot contains non-hex character: '{}'",
                chunk[0] as char
            )
        })?;
        let lo = hex_value(chunk[1]).ok_or_else(|| {
            format!(
                "msgpack snapshot contains non-hex character: '{}'",
                chunk[1] as char
            )
        })?;
        decoded.push((hi << 4) | lo);
    }

    Ok(decoded)
}

fn resolve_import_state(request: &ImportWorldRequest) -> std::result::Result<WorldState, String> {
    if let Some(state) = request.state.clone() {
        return Ok(state);
    }

    let format = request
        .format
        .as_deref()
        .ok_or_else(|| "import requires either `state` or `format` + `snapshot`".to_string())?
        .parse::<StateFileFormat>()
        .map_err(|error| error.to_string())?;
    let snapshot = request
        .snapshot
        .as_deref()
        .ok_or_else(|| "import requires either `state` or `format` + `snapshot`".to_string())?;
    let bytes = decode_snapshot_payload(
        format,
        snapshot,
        request.encoding.as_deref(),
        request.sha256.as_deref(),
    )?;
    deserialize_world_state(format, &bytes).map_err(|error| error.to_string())
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

fn resolve_world_provider(state: &WorldState, requested: Option<&str>) -> String {
    requested
        .filter(|provider| !provider.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| state.current_state_provider())
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
    object.mesh = request.mesh;
    object.velocity = request.velocity;
    object.semantic_label = request.semantic_label;
    object.visual_embedding = request.visual_embedding;
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
    let request = match read_request(&mut buf_reader).await {
        Ok(request) => request,
        Err(response) => {
            send_response(&mut writer, response).await?;
            return Ok(());
        }
    };

    let response = dispatch_request(&request, &state).await;
    send_response(&mut writer, response).await?;
    Ok(())
}

async fn read_request(
    reader: &mut (impl AsyncBufReadExt + AsyncReadExt + Unpin),
) -> std::result::Result<ParsedRequest, WireResponse> {
    const MAX_BODY_SIZE: usize = 4 * 1024 * 1024; // 4 MiB
    const MAX_HEADER_COUNT: usize = 64;

    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .await
        .map_err(|_| WireResponse::json(400, error_response("bad request")))?;

    let (method, path) = parse_request_line(&request_line)?;

    let mut content_length = 0usize;
    let mut header_count = 0usize;
    loop {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .await
            .map_err(|_| WireResponse::json(400, error_response("bad request")))?;
        if line.trim().is_empty() {
            break;
        }

        header_count += 1;
        if header_count > MAX_HEADER_COUNT {
            return Err(WireResponse::json(400, error_response("too many headers")));
        }

        let Some((name, value)) = line.split_once(':') else {
            return Err(WireResponse::json(400, error_response("malformed header")));
        };

        if name.trim().eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse::<usize>().map_err(|_| {
                WireResponse::json(400, error_response("invalid content-length header"))
            })?;
        }
    }

    if content_length > MAX_BODY_SIZE {
        return Err(WireResponse::json(
            413,
            error_response("request body too large"),
        ));
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .await
            .map_err(|_| WireResponse::json(400, error_response("bad request")))?;
    }

    Ok(ParsedRequest {
        method,
        path,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn parse_request_line(line: &str) -> std::result::Result<(String, String), WireResponse> {
    let line = line.trim_end_matches(['\r', '\n']);
    let mut parts = line.split(' ');
    let Some(method) = parts.next() else {
        return Err(WireResponse::json(400, error_response("bad request")));
    };
    let Some(path) = parts.next() else {
        return Err(WireResponse::json(400, error_response("bad request")));
    };
    let Some(version) = parts.next() else {
        return Err(WireResponse::json(400, error_response("bad request")));
    };

    if method.is_empty()
        || path.is_empty()
        || version.is_empty()
        || parts.next().is_some()
        || !path.starts_with('/')
        || !matches!(version, "HTTP/1.0" | "HTTP/1.1")
    {
        return Err(WireResponse::json(400, error_response("bad request")));
    }

    Ok((method.to_string(), path.to_string()))
}

fn split_path_and_query(path: &str) -> (&str, Option<&str>) {
    match path.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (path, None),
    }
}

fn query_param(query: Option<&str>, key: &str) -> Option<String> {
    query.and_then(|query| {
        query.split('&').find_map(|pair| {
            let (candidate_key, candidate_value) = pair.split_once('=').unwrap_or((pair, ""));
            let decoded_key = percent_decode_component(candidate_key)?;
            (decoded_key == key).then(|| percent_decode_component(candidate_value))
        })?
    })
}

fn query_flag(query: Option<&str>, key: &str) -> bool {
    query_param(query, key).is_some_and(|value| {
        value.is_empty() || matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on")
    })
}

fn percent_decode_component(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                decoded.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hi = hex_value(bytes[index + 1])?;
                let lo = hex_value(bytes[index + 2])?;
                decoded.push((hi << 4) | lo);
                index += 3;
            }
            b'%' => return None,
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }

    String::from_utf8(decoded).ok()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteKind {
    VerifyProof,
    ProvidersCollection,
    ProviderDescriptor,
    ProviderHealth,
    ProviderEstimate,
    EvalsSuites,
    EvalsRun,
    WorldsCollection,
    WorldsImport,
    WorldExport,
    ProviderGenerate,
    ProviderEmbed,
    ProviderReason,
    ProviderTransfer,
    WorldHistory,
    WorldRestore,
    WorldObjects,
    WorldObject,
    WorldItem,
    WorldPredict,
    WorldReason,
    WorldPlan,
    WorldExecutePlan,
    WorldVerify,
    WorldEvaluate,
    Compare,
    Unknown,
}

impl RouteKind {
    fn allowed_methods(self) -> &'static [&'static str] {
        match self {
            Self::VerifyProof => &["POST"],
            Self::ProvidersCollection => &["GET", "HEAD"],
            Self::ProviderDescriptor => &["GET", "HEAD"],
            Self::ProviderHealth => &["GET", "HEAD"],
            Self::ProviderEstimate => &["POST"],
            Self::EvalsSuites => &["GET", "HEAD"],
            Self::EvalsRun => &["POST"],
            Self::WorldsCollection => &["GET", "HEAD", "POST"],
            Self::WorldsImport => &["POST"],
            Self::WorldExport => &["GET", "HEAD"],
            Self::ProviderGenerate => &["POST"],
            Self::ProviderEmbed => &["POST"],
            Self::ProviderReason => &["POST"],
            Self::ProviderTransfer => &["POST"],
            Self::WorldHistory => &["GET", "HEAD"],
            Self::WorldRestore => &["POST"],
            Self::WorldObjects => &["GET", "HEAD", "POST"],
            Self::WorldObject => &["GET", "HEAD", "PATCH", "DELETE"],
            Self::WorldItem => &["GET", "HEAD", "DELETE"],
            Self::WorldPredict => &["POST"],
            Self::WorldReason => &["POST"],
            Self::WorldPlan => &["POST"],
            Self::WorldExecutePlan => &["POST"],
            Self::WorldVerify => &["POST"],
            Self::WorldEvaluate => &["POST"],
            Self::Compare => &["POST"],
            Self::Unknown => &[],
        }
    }

    fn allows(self, method: &str) -> bool {
        self.allowed_methods().contains(&method)
    }
}

fn classify_route_kind(path: &str) -> RouteKind {
    let segments: Vec<&str> = path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();

    match segments.as_slice() {
        ["v1", "verify", "proof"] => RouteKind::VerifyProof,
        ["v1", "providers"] => RouteKind::ProvidersCollection,
        ["v1", "providers", _, "health"] => RouteKind::ProviderHealth,
        ["v1", "providers", _, "estimate"] => RouteKind::ProviderEstimate,
        ["v1", "providers", _, "generate"] => RouteKind::ProviderGenerate,
        ["v1", "providers", _, "embed"] => RouteKind::ProviderEmbed,
        ["v1", "providers", _, "reason"] => RouteKind::ProviderReason,
        ["v1", "providers", _, "transfer"] => RouteKind::ProviderTransfer,
        ["v1", "providers", _] => RouteKind::ProviderDescriptor,
        ["v1", "evals", "suites"] => RouteKind::EvalsSuites,
        ["v1", "evals", "run"] => RouteKind::EvalsRun,
        ["v1", "worlds"] => RouteKind::WorldsCollection,
        ["v1", "worlds", "import"] => RouteKind::WorldsImport,
        ["v1", "worlds", _, "export"] => RouteKind::WorldExport,
        ["v1", "worlds", _, "history"] => RouteKind::WorldHistory,
        ["v1", "worlds", _, "restore"] => RouteKind::WorldRestore,
        ["v1", "worlds", _, "objects", _] => RouteKind::WorldObject,
        ["v1", "worlds", _, "objects"] => RouteKind::WorldObjects,
        ["v1", "worlds", _, "predict"] => RouteKind::WorldPredict,
        ["v1", "worlds", _, "reason"] => RouteKind::WorldReason,
        ["v1", "worlds", _, "plan"] => RouteKind::WorldPlan,
        ["v1", "worlds", _, "execute-plan"] => RouteKind::WorldExecutePlan,
        ["v1", "worlds", _, "verify"] => RouteKind::WorldVerify,
        ["v1", "worlds", _, "evaluate"] => RouteKind::WorldEvaluate,
        ["v1", "worlds", _] => RouteKind::WorldItem,
        ["v1", "compare"] => RouteKind::Compare,
        _ => RouteKind::Unknown,
    }
}

async fn dispatch_request(request: &ParsedRequest, state: &AppState) -> WireResponse {
    let head_request = request.method.eq_ignore_ascii_case("HEAD");
    let effective_method = if head_request {
        "GET"
    } else {
        request.method.as_str()
    };

    let (path_only, _) = split_path_and_query(&request.path);
    let normalized_path = path_only.trim_end_matches('/');
    let route_kind = classify_route_kind(normalized_path);
    if route_kind != RouteKind::Unknown && !route_kind.allows(&request.method) {
        let response = WireResponse::json(405, error_response("method not allowed"))
            .with_header("Allow", route_kind.allowed_methods().join(", "));
        return if head_request {
            response.without_body()
        } else {
            response
        };
    }

    let (status, body) = route(effective_method, &request.path, &request.body, state).await;
    let response = WireResponse::json(status, body);
    if head_request {
        response.without_body()
    } else {
        response
    }
}

async fn route(method: &str, path: &str, body: &str, state: &AppState) -> (u16, String) {
    let (path, query) = split_path_and_query(path);
    // Trim trailing slash
    let path = path.trim_end_matches('/');

    match (method, path) {
        // POST /v1/verify/proof
        ("POST", "/v1/verify/proof") => match serde_json::from_str::<VerifyProofRequest>(body) {
            Ok(req) => {
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
                    let verifier = verifier_for_backend(req.backend.unwrap_or(proof.backend));
                    let verification = match verify_proof(verifier.as_ref(), &proof) {
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
                    let verifier =
                        verifier_for_backend(req.backend.unwrap_or(bundle.proof.backend));
                    return match verify_bundle(verifier.as_ref(), &bundle) {
                        Ok(report) => (200, ApiResponse::ok(report)),
                        Err(error) => (400, error_response(&error.to_string())),
                    };
                }

                if let Some(bundle) = req.guardrail_bundle {
                    let verifier =
                        verifier_for_backend(req.backend.unwrap_or(bundle.proof.backend));
                    return match verify_bundle(verifier.as_ref(), &bundle) {
                        Ok(report) => (200, ApiResponse::ok(report)),
                        Err(error) => (400, error_response(&error.to_string())),
                    };
                }

                if let Some(bundle) = req.provenance_bundle {
                    let verifier =
                        verifier_for_backend(req.backend.unwrap_or(bundle.proof.backend));
                    return match verify_bundle(verifier.as_ref(), &bundle) {
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
            if query_flag(query, "health") {
                let providers = match query_param(query, "capability") {
                    Some(capability) => {
                        state.registry.health_check_by_capability(&capability).await
                    }
                    None => state.registry.health_check_all().await,
                };
                (200, ApiResponse::ok(providers))
            } else {
                match query_param(query, "capability") {
                    Some(capability) => {
                        let providers = state.registry.describe_by_capability(&capability);
                        (200, ApiResponse::ok(providers))
                    }
                    None => {
                        let providers = state.registry.describe_all();
                        (200, ApiResponse::ok(providers))
                    }
                }
            }
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

        // POST /v1/evals/run
        ("POST", "/v1/evals/run") => match serde_json::from_str::<EvaluateRequest>(body) {
            Ok(req) => match resolve_eval_world_state(&req, state).await {
                Ok(world_state) => run_evaluation_request(&req, state, world_state.as_ref()).await,
                Err(response) => response,
            },
            Err(error) => (400, error_response(&format!("invalid request: {error}"))),
        },

        // GET /v1/providers/{name}/health
        ("GET", p) if p.starts_with("/v1/providers/") && p.ends_with("/health") => {
            let name = p
                .strip_prefix("/v1/providers/")
                .and_then(|s| s.strip_suffix("/health"))
                .unwrap_or("");
            match state.registry.health_check(name).await {
                Ok(report) => match report.status {
                    Some(status) => (200, ApiResponse::ok(status)),
                    None => (
                        503,
                        error_response(report.error.as_deref().unwrap_or("health check failed")),
                    ),
                },
                Err(error) => (404, error_response(&error.to_string())),
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
                let world_name = req
                    .name
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty());
                let prompt = req
                    .prompt
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty());
                let ws = match (world_name, prompt) {
                    (Some(name), Some(prompt)) => {
                        match WorldState::from_prompt(prompt, &req.provider, Some(name)) {
                            Ok(state) => state,
                            Err(error) => {
                                return (
                                    api_error_status(&error),
                                    error_response(&error.to_string()),
                                );
                            }
                        }
                    }
                    (None, Some(prompt)) => {
                        match WorldState::from_prompt(prompt, &req.provider, None) {
                            Ok(state) => state,
                            Err(error) => {
                                return (
                                    api_error_status(&error),
                                    error_response(&error.to_string()),
                                );
                            }
                        }
                    }
                    (Some(name), None) => WorldState::new(name, &req.provider),
                    (None, None) => {
                        return (
                            400,
                            error_response("create world requires at least one of name or prompt"),
                        );
                    }
                };
                let id = ws.id;
                if let Err(e) = state.store.save(&ws).await {
                    return (500, error_response(&e.to_string()));
                }
                state.worlds.write().await.insert(id, ws.clone());
                (
                    201,
                    ApiResponse::ok(serde_json::json!({
                        "id": id.to_string(),
                        "name": ws.metadata.name,
                        "description": ws.metadata.description,
                        "provider": req.provider,
                        "object_count": ws.scene.objects.len(),
                    })),
                )
            }
            Err(e) => (400, error_response(&format!("invalid request: {e}"))),
        },

        // POST /v1/worlds/import
        ("POST", "/v1/worlds/import") => match serde_json::from_str::<ImportWorldRequest>(body) {
            Ok(req) => {
                let mut imported = match resolve_import_state(&req) {
                    Ok(state) => state,
                    Err(error) => return (400, error_response(&error)),
                };

                if req.new_id {
                    imported.id = WorldId::new_v4();
                }

                if let Some(name) = req
                    .name
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    imported.metadata.name = name.to_string();
                }

                let id = imported.id;
                if let Err(error) = state.store.save(&imported).await {
                    return (500, error_response(&error.to_string()));
                }
                state.worlds.write().await.insert(id, imported.clone());
                (201, ApiResponse::ok(imported))
            }
            Err(e) => (400, error_response(&format!("invalid request: {e}"))),
        },

        // GET /v1/worlds/{id}/export
        ("GET", p) if p.starts_with("/v1/worlds/") && p.ends_with("/export") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|value| value.strip_suffix("/export"))
                .unwrap_or("");
            let Some(format) = query_param(query, "format") else {
                return (
                    400,
                    error_response("export requires a `format` query parameter"),
                );
            };

            match (id_str.parse::<WorldId>(), format.parse::<StateFileFormat>()) {
                (Ok(id), Ok(format)) => match state.store.load(&id).await {
                    Ok(ws) => match encode_snapshot_payload(format, &ws) {
                        Ok(payload) => (
                            200,
                            ApiResponse::ok(serde_json::json!({
                                "id": id.to_string(),
                                "format": payload.format,
                                "encoding": payload.encoding,
                                "sha256": payload.sha256,
                                "snapshot": payload.snapshot,
                            })),
                        ),
                        Err(error) => (500, error_response(&error)),
                    },
                    Err(error) => (404, error_response(&error.to_string())),
                },
                (Err(_), _) => (400, error_response("invalid world ID")),
                (_, Err(_)) => (400, error_response("invalid export format")),
            }
        }

        // POST /v1/providers/{name}/generate
        ("POST", p) if p.starts_with("/v1/providers/") && p.ends_with("/generate") => {
            let provider_name = p
                .strip_prefix("/v1/providers/")
                .and_then(|value| value.strip_suffix("/generate"))
                .unwrap_or("");
            match serde_json::from_str::<GenerateRequest>(body) {
                Ok(req) => {
                    let prompt = GenerationPrompt {
                        text: req.prompt,
                        reference_image: None,
                        negative_prompt: req.negative_prompt,
                    };
                    let world = World::new(
                        WorldState::new("server-generate", provider_name),
                        provider_name,
                        Arc::clone(&state.registry),
                    );
                    match world
                        .generate_with_provider_and_fallback(
                            &prompt,
                            &req.config,
                            provider_name,
                            req.fallback_provider.as_deref(),
                        )
                        .await
                    {
                        Ok((_, clip)) => (200, ApiResponse::ok(clip)),
                        Err(error) => {
                            (api_error_status(&error), error_response(&error.to_string()))
                        }
                    }
                }
                Err(e) => (400, error_response(&format!("invalid request: {e}"))),
            }
        }

        // POST /v1/providers/{name}/embed
        ("POST", p) if p.starts_with("/v1/providers/") && p.ends_with("/embed") => {
            let provider_name = p
                .strip_prefix("/v1/providers/")
                .and_then(|value| value.strip_suffix("/embed"))
                .unwrap_or("");
            match serde_json::from_str::<EmbedRequest>(body) {
                Ok(req) => {
                    let input = match EmbeddingInput::new(req.text, req.video) {
                        Ok(input) => input,
                        Err(error) => return (400, error_response(&error.to_string())),
                    };

                    match state.registry.get(provider_name) {
                        Ok(provider) => match provider.embed(&input).await {
                            Ok(output) => (200, ApiResponse::ok(output)),
                            Err(primary_error) => {
                                let Some(fallback_provider) = req
                                    .fallback_provider
                                    .as_deref()
                                    .filter(|fallback| *fallback != provider_name)
                                else {
                                    return (
                                        api_error_status(&primary_error),
                                        error_response(&primary_error.to_string()),
                                    );
                                };

                                match state.registry.get(fallback_provider) {
                                    Ok(provider) => match provider.embed(&input).await {
                                        Ok(output) => (200, ApiResponse::ok(output)),
                                        Err(fallback_error) => (
                                            api_error_status(&fallback_error),
                                            error_response(
                                                &WorldForgeError::ProviderUnavailable {
                                                    provider: provider_name.to_string(),
                                                    reason: format!(
                                                        "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                                                    ),
                                                }
                                                .to_string(),
                                            ),
                                        ),
                                    },
                                    Err(fallback_error) => (
                                        api_error_status(&fallback_error),
                                        error_response(
                                            &WorldForgeError::ProviderUnavailable {
                                                provider: provider_name.to_string(),
                                                reason: format!(
                                                    "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                                                ),
                                            }
                                            .to_string(),
                                        ),
                                    ),
                                }
                            }
                        },
                        Err(primary_error) => {
                            let Some(fallback_provider) = req
                                .fallback_provider
                                .as_deref()
                                .filter(|fallback| *fallback != provider_name)
                            else {
                                return (
                                    api_error_status(&primary_error),
                                    error_response(&primary_error.to_string()),
                                );
                            };

                            match state.registry.get(fallback_provider) {
                                Ok(provider) => match provider.embed(&input).await {
                                    Ok(output) => (200, ApiResponse::ok(output)),
                                    Err(fallback_error) => (
                                        api_error_status(&fallback_error),
                                        error_response(
                                            &WorldForgeError::ProviderUnavailable {
                                                provider: provider_name.to_string(),
                                                reason: format!(
                                                    "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                                                ),
                                            }
                                            .to_string(),
                                        ),
                                    ),
                                },
                                Err(fallback_error) => (
                                    api_error_status(&fallback_error),
                                    error_response(
                                        &WorldForgeError::ProviderUnavailable {
                                            provider: provider_name.to_string(),
                                            reason: format!(
                                                "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                                            ),
                                        }
                                        .to_string(),
                                    ),
                                ),
                            }
                        }
                    }
                }
                Err(e) => (400, error_response(&format!("invalid request: {e}"))),
            }
        }

        // POST /v1/providers/{name}/reason
        ("POST", p) if p.starts_with("/v1/providers/") && p.ends_with("/reason") => {
            let provider_name = p
                .strip_prefix("/v1/providers/")
                .and_then(|value| value.strip_suffix("/reason"))
                .unwrap_or("");
            match serde_json::from_str::<ProviderReasonRequest>(body) {
                Ok(req) => {
                    let input = ReasoningInput {
                        state: req.state,
                        video: req.video,
                    };
                    if input.state.is_none() && input.video.is_none() {
                        return (
                            400,
                            error_response("reasoning input requires state and/or video"),
                        );
                    }

                    match state.registry.get(provider_name) {
                        Ok(provider) => match provider.reason(&input, &req.query).await {
                            Ok(output) => (200, ApiResponse::ok(output)),
                            Err(primary_error) => {
                                let Some(fallback_provider) = req
                                    .fallback_provider
                                    .as_deref()
                                    .filter(|fallback| *fallback != provider_name)
                                else {
                                    return (
                                        api_error_status(&primary_error),
                                        error_response(&primary_error.to_string()),
                                    );
                                };

                                match state.registry.get(fallback_provider) {
                                    Ok(provider) => match provider.reason(&input, &req.query).await
                                    {
                                        Ok(output) => (200, ApiResponse::ok(output)),
                                        Err(fallback_error) => (
                                            api_error_status(&fallback_error),
                                            error_response(
                                                &WorldForgeError::ProviderUnavailable {
                                                    provider: provider_name.to_string(),
                                                    reason: format!(
                                                        "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                                                    ),
                                                }
                                                .to_string(),
                                            ),
                                        ),
                                    },
                                    Err(fallback_error) => (
                                        api_error_status(&fallback_error),
                                        error_response(
                                            &WorldForgeError::ProviderUnavailable {
                                                provider: provider_name.to_string(),
                                                reason: format!(
                                                    "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                                                ),
                                            }
                                            .to_string(),
                                        ),
                                    ),
                                }
                            }
                        },
                        Err(primary_error) => {
                            let Some(fallback_provider) = req
                                .fallback_provider
                                .as_deref()
                                .filter(|fallback| *fallback != provider_name)
                            else {
                                return (
                                    api_error_status(&primary_error),
                                    error_response(&primary_error.to_string()),
                                );
                            };

                            match state.registry.get(fallback_provider) {
                                Ok(provider) => match provider.reason(&input, &req.query).await {
                                    Ok(output) => (200, ApiResponse::ok(output)),
                                    Err(fallback_error) => (
                                        api_error_status(&fallback_error),
                                        error_response(
                                            &WorldForgeError::ProviderUnavailable {
                                                provider: provider_name.to_string(),
                                                reason: format!(
                                                    "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                                                ),
                                            }
                                            .to_string(),
                                        ),
                                    ),
                                },
                                Err(fallback_error) => (
                                    api_error_status(&fallback_error),
                                    error_response(
                                        &WorldForgeError::ProviderUnavailable {
                                            provider: provider_name.to_string(),
                                            reason: format!(
                                                "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                                            ),
                                        }
                                        .to_string(),
                                    ),
                                ),
                            }
                        }
                    }
                }
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
                Ok(req) => {
                    let world = World::new(
                        WorldState::new("server-transfer", provider_name),
                        provider_name,
                        Arc::clone(&state.registry),
                    );
                    match world
                        .transfer_with_provider_and_fallback(
                            &req.source,
                            &req.controls,
                            &req.config,
                            provider_name,
                            req.fallback_provider.as_deref(),
                        )
                        .await
                    {
                        Ok((_, clip)) => (200, ApiResponse::ok(clip)),
                        Err(error) => {
                            (api_error_status(&error), error_response(&error.to_string()))
                        }
                    }
                }
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

        // GET /v1/worlds/{id}/history
        ("GET", p) if p.starts_with("/v1/worlds/") && p.ends_with("/history") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|value| value.strip_suffix("/history"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match state.store.load(&id).await {
                    Ok(ws) => {
                        let entries: Vec<_> = ws.history.states.iter().cloned().collect();
                        (200, ApiResponse::ok(entries))
                    }
                    Err(error) => (404, error_response(&error.to_string())),
                },
                Err(_) => (400, error_response("invalid world ID")),
            }
        }

        // POST /v1/worlds/{id}/restore
        ("POST", p) if p.starts_with("/v1/worlds/") && p.ends_with("/restore") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|value| value.strip_suffix("/restore"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match serde_json::from_str::<RestoreWorldRequest>(body) {
                    Ok(req) => match state.store.load(&id).await {
                        Ok(mut ws) => match ws.restore_history(req.history_index) {
                            Ok(()) => match state.store.save(&ws).await {
                                Ok(()) => {
                                    state.worlds.write().await.insert(id, ws.clone());
                                    (200, ApiResponse::ok(ws))
                                }
                                Err(error) => (
                                    500,
                                    error_response(&format!("failed to save world: {error}")),
                                ),
                            },
                            Err(error) => {
                                (api_error_status(&error), error_response(&error.to_string()))
                            }
                        },
                        Err(error) => (404, error_response(&error.to_string())),
                    },
                    Err(error) => (400, error_response(&format!("invalid request: {error}"))),
                },
                Err(_) => (400, error_response("invalid world ID")),
            }
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
                        let provider = ws.current_state_provider();
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
                        let provider = ws.current_state_provider();
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

        // PATCH /v1/worlds/{id}/objects/{object_id}
        ("PATCH", p) if p.starts_with("/v1/worlds/") && p.contains("/objects/") => {
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
                (Ok(world_id), Ok(object_id)) => {
                    match serde_json::from_str::<SceneObjectPatch>(body) {
                        Ok(req) => match state.store.load(&world_id).await {
                            Ok(ws) => {
                                let provider = ws.current_state_provider();
                                let mut world = worldforge_core::world::World::new(
                                    ws,
                                    provider,
                                    Arc::clone(&state.registry),
                                );
                                match world.update_object(&object_id, req) {
                                    Ok(object) => {
                                        match state.store.save(world.current_state()).await {
                                            Ok(()) => (200, ApiResponse::ok(object)),
                                            Err(error) => (500, error_response(&error.to_string())),
                                        }
                                    }
                                    Err(error) => {
                                        let status = match &error {
                                            WorldForgeError::InvalidState(message)
                                                if message.starts_with("object not found: ") =>
                                            {
                                                404
                                            }
                                            _ => api_error_status(&error),
                                        };
                                        (status, error_response(&error.to_string()))
                                    }
                                }
                            }
                            Err(error) => (404, error_response(&error.to_string())),
                        },
                        Err(error) => (400, error_response(&format!("invalid request: {error}"))),
                    }
                }
                (Err(_), _) => (400, error_response("invalid world ID")),
                (_, Err(_)) => (400, error_response("invalid object ID")),
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
                        let provider = ws.current_state_provider();
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
                        let provider = ws.current_state_provider();
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
                && !p.contains("/verify")
                && !p.contains("/export") =>
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
                            let config = if req.disable_guardrails {
                                req.config.disable_guardrails()
                            } else {
                                req.config
                            };
                            let provider_name =
                                resolve_world_provider(&ws, req.provider.as_deref());
                            let mut world = worldforge_core::world::World::new(
                                ws,
                                provider_name,
                                Arc::clone(&state.registry),
                            );
                            match world.predict(&req.action, &config).await {
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
                        let provider_name = resolve_world_provider(&ws, req.provider.as_deref());
                        let world = worldforge_core::world::World::new(
                            ws,
                            provider_name.clone(),
                            Arc::clone(&state.registry),
                        );
                        match world
                            .reason_with_provider_and_fallback(
                                &req.query,
                                &provider_name,
                                req.fallback_provider.as_deref(),
                            )
                            .await
                        {
                            Ok((_, output)) => (200, ApiResponse::ok(output)),
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
        ("POST", p)
            if p.starts_with("/v1/worlds/")
                && p.ends_with("/plan")
                && !p.ends_with("/execute-plan") =>
        {
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
                        let provider_name = resolve_world_provider(&ws, req.provider.as_deref());
                        if req.fallback_provider.is_none() {
                            if let Err(e) = state.registry.get(&provider_name) {
                                return (404, error_response(&e.to_string()));
                            }
                        }
                        if let Some(fallback_provider) = req.fallback_provider.as_deref() {
                            if let Err(e) = state.registry.get(fallback_provider) {
                                return (404, error_response(&e.to_string()));
                            }
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
                            req.goal.clone().into(),
                            req.max_steps,
                            resolve_guardrails(req.guardrails, req.disable_guardrails),
                            planner,
                            req.timeout_seconds,
                            req.fallback_provider,
                        );
                        match world.plan(&plan_req).await {
                            Ok(mut plan) => {
                                let verification_backend =
                                    match parse_requested_verification_backend(
                                        req.verification_backend.as_deref(),
                                    ) {
                                        Ok(backend) => backend,
                                        Err(error) => return (400, error_response(&error)),
                                    };
                                match attach_plan_verification(&mut plan, verification_backend) {
                                    Ok(_) => (200, ApiResponse::ok(plan)),
                                    Err(error) => (500, error_response(&error)),
                                }
                            }
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

        // POST /v1/worlds/{id}/execute-plan
        ("POST", p) if p.starts_with("/v1/worlds/") && p.ends_with("/execute-plan") => {
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|s| s.strip_suffix("/execute-plan"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match serde_json::from_str::<ExecutePlanRequest>(body) {
                    Ok(req) => {
                        let ws = match state.store.load(&id).await {
                            Ok(ws) => ws,
                            Err(e) => return (404, error_response(&e.to_string())),
                        };
                        let provider_name = resolve_world_provider(&ws, req.provider.as_deref());
                        if let Err(e) = state.registry.get(&provider_name) {
                            return (404, error_response(&e.to_string()));
                        }

                        let mut world = worldforge_core::world::World::new(
                            ws,
                            &provider_name,
                            Arc::clone(&state.registry),
                        );
                        let mut config = req.config;
                        if req.disable_guardrails {
                            config = config.disable_guardrails();
                        }

                        match world
                            .execute_plan_with_provider(&req.plan, &config, &provider_name)
                            .await
                        {
                            Ok(report) => match state.store.save(world.current_state()).await {
                                Ok(()) => (200, ApiResponse::ok(report)),
                                Err(error) => (
                                    500,
                                    error_response(&format!("failed to save world: {error}")),
                                ),
                            },
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
                        let verifier = verifier_for_backend(req.backend);
                        match req.proof_type.as_str() {
                            "inference" => {
                                let bundle = match (
                                    req.input_state.as_ref(),
                                    req.output_state.as_ref(),
                                ) {
                                    (Some(input_state), Some(output_state)) => {
                                        let provider_name = req
                                            .provider
                                            .as_deref()
                                            .filter(|name| !name.is_empty())
                                            .unwrap_or(output_state.metadata.created_by.as_str());
                                        prove_inference_transition(
                                            verifier.as_ref(),
                                            provider_name,
                                            input_state,
                                            output_state,
                                        )
                                    }
                                    (None, None) => prove_latest_inference(
                                        verifier.as_ref(),
                                        &ws,
                                        req.provider.as_deref(),
                                    ),
                                    _ => {
                                        return (
                                            400,
                                            error_response(
                                                "inference verification requires both input_state and output_state when either is provided",
                                            ),
                                        );
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
                                            );
                                        }
                                    };
                                    let provider_name =
                                        resolve_world_provider(&ws, req.provider.as_deref());
                                    if req.fallback_provider.is_none() {
                                        if let Err(e) = state.registry.get(&provider_name) {
                                            return (404, error_response(&e.to_string()));
                                        }
                                    }
                                    if let Some(fallback_provider) =
                                        req.fallback_provider.as_deref()
                                    {
                                        if let Err(e) = state.registry.get(fallback_provider) {
                                            return (404, error_response(&e.to_string()));
                                        }
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
                                        goal.into(),
                                        req.max_steps,
                                        resolve_guardrails(
                                            req.guardrails.clone(),
                                            req.disable_guardrails,
                                        ),
                                        planner,
                                        req.timeout_seconds,
                                        req.fallback_provider.clone(),
                                    );
                                    match world.plan(&plan_req).await {
                                        Ok(plan) => plan,
                                        Err(error) => {
                                            return (
                                                api_error_status(&error),
                                                error_response(&error.to_string()),
                                            );
                                        }
                                    }
                                };

                                match prove_guardrail_plan(verifier.as_ref(), &plan) {
                                    Ok(bundle) => (200, ApiResponse::ok(bundle)),
                                    Err(e) => (500, error_response(&e.to_string())),
                                }
                            }
                            "provenance" => {
                                let target_state = req.output_state.as_ref().unwrap_or(&ws);
                                let timestamp = chrono::Utc::now().timestamp() as u64;
                                match prove_provenance(
                                    verifier.as_ref(),
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
            let id_str = p
                .strip_prefix("/v1/worlds/")
                .and_then(|s| s.strip_suffix("/evaluate"))
                .unwrap_or("");
            match id_str.parse::<WorldId>() {
                Ok(id) => match state.store.load(&id).await {
                    Ok(ws) => match serde_json::from_str::<EvaluateRequest>(body) {
                        Ok(req) => {
                            if req.world_id.is_some() || req.world_state.is_some() {
                                return (
                                    400,
                                    error_response(
                                        "world-scoped evaluation uses the path world ID; omit world_id/world_state or use /v1/evals/run",
                                    ),
                                );
                            }

                            run_evaluation_request(&req, state, Some(&ws)).await
                        }
                        Err(e) => (400, error_response(&format!("invalid request: {e}"))),
                    },
                    Err(error) => (404, error_response(&error.to_string())),
                },
                Err(_) => (400, error_response("invalid world ID")),
            }
        }

        // POST /v1/compare
        ("POST", "/v1/compare") => match serde_json::from_str::<CompareRequest>(body) {
            Ok(req) => {
                if req.providers.is_empty() {
                    return (
                        400,
                        error_response("compare requires at least one provider"),
                    );
                }

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
                let config = if req.disable_guardrails {
                    req.config.disable_guardrails()
                } else {
                    req.config
                };
                match world
                    .predict_multi(&req.action, &provider_refs, &config)
                    .await
                {
                    Ok(multi) => (200, ApiResponse::ok(multi)),
                    Err(error) => (api_error_status(&error), error_response(&error.to_string())),
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
    response: WireResponse,
) -> anyhow::Result<()> {
    let status_text = match response.status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Payload Too Large",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown",
    };
    let content_length = response.content_length.unwrap_or(response.body.len());
    let mut raw = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        response.status, status_text, response.content_type, content_length
    );
    for (name, value) in response.headers {
        raw.push_str(name);
        raw.push_str(": ");
        raw.push_str(&value);
        raw.push_str("\r\n");
    }
    raw.push_str("\r\n");
    raw.push_str(&response.body);
    writer.write_all(raw.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::state::FileStateStore;
    use worldforge_core::types::{DType, Device, TensorData};
    use worldforge_providers::MockProvider;
    use worldforge_verify::{MockVerifier, ZkVerifier};

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

    fn sample_mesh() -> Mesh {
        Mesh {
            vertices: vec![
                Position {
                    x: -0.5,
                    y: 0.0,
                    z: -0.5,
                },
                Position {
                    x: 0.5,
                    y: 0.0,
                    z: -0.5,
                },
                Position {
                    x: 0.0,
                    y: 0.5,
                    z: 0.5,
                },
            ],
            faces: vec![[0, 1, 2]],
            normals: Some(vec![
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ]),
            uvs: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        }
    }

    fn sample_visual_embedding() -> serde_json::Value {
        serde_json::json!({
            "data": { "Float32": [0.1, 0.2, 0.3, 0.4] },
            "shape": [4],
            "dtype": "Float32",
            "device": "Cpu"
        })
    }

    fn sample_visual_embedding_tensor() -> Tensor {
        Tensor {
            data: TensorData::Float32(vec![0.1, 0.2, 0.3, 0.4]),
            shape: vec![4],
            dtype: DType::Float32,
            device: Device::Cpu,
        }
    }

    fn sample_export_state(name: &str) -> WorldState {
        let mut state = WorldState::new(name, "mock");
        state.metadata.description = "portable snapshot".to_string();
        state.scene.add_object(SceneObject::new(
            "anchor",
            Pose::default(),
            BBox {
                min: Position {
                    x: -0.25,
                    y: 0.0,
                    z: -0.25,
                },
                max: Position {
                    x: 0.25,
                    y: 0.5,
                    z: 0.25,
                },
            },
        ));
        state
    }

    async fn seed_colliding_world(state: &Arc<AppState>) -> WorldId {
        let mut world = WorldState::new("colliding-world", "mock");
        world.scene.add_object(SceneObject::new(
            "left",
            Pose::default(),
            BBox {
                min: Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                max: Position {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                },
            },
        ));
        world.scene.add_object(SceneObject::new(
            "right",
            Pose::default(),
            BBox {
                min: Position {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
                max: Position {
                    x: 1.5,
                    y: 1.5,
                    z: 1.5,
                },
            },
        ));
        let id = world.id;
        state.store.save(&world).await.unwrap();
        id
    }

    async fn seed_world(state: &Arc<AppState>, name: &str, provider: &str) -> WorldId {
        let world = WorldState::new(name, provider);
        let id = world.id;
        state.store.save(&world).await.unwrap();
        id
    }

    async fn seed_world_with_object(
        state: &Arc<AppState>,
        name: &str,
        provider: &str,
        object_name: &str,
        position: Position,
    ) -> (WorldId, uuid::Uuid) {
        let mut world = WorldState::new(name, provider);
        let object = SceneObject::new(
            object_name,
            Pose {
                position,
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: position.x - 0.1,
                    y: position.y - 0.1,
                    z: position.z - 0.1,
                },
                max: Position {
                    x: position.x + 0.1,
                    y: position.y + 0.1,
                    z: position.z + 0.1,
                },
            },
        );
        let object_id = object.id;
        world.scene.add_object(object);
        let id = world.id;
        state.store.save(&world).await.unwrap();
        (id, object_id)
    }

    #[test]
    fn test_server_config_resolves_file_store() {
        let config = ServerConfig {
            state_dir: "/tmp/worldforge".to_string(),
            ..ServerConfig::default()
        };

        assert_eq!(
            config.resolve_state_store_kind().unwrap(),
            StateStoreKind::FileWithFormat {
                path: "/tmp/worldforge".into(),
                format: StateFileFormat::Json,
            }
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

    #[test]
    fn test_server_config_resolves_redis_store() {
        let config = ServerConfig {
            state_backend: "redis".to_string(),
            state_redis_url: Some("redis://127.0.0.1:6379/1".to_string()),
            ..ServerConfig::default()
        };

        assert_eq!(
            config.resolve_state_store_kind().unwrap(),
            StateStoreKind::Redis("redis://127.0.0.1:6379/1".to_string())
        );
    }

    #[test]
    fn test_server_config_resolves_s3_store() {
        let config = ServerConfig {
            state_backend: "s3".to_string(),
            state_s3_bucket: Some("worldforge-states".to_string()),
            state_s3_region: Some("us-east-1".to_string()),
            state_s3_access_key_id: Some("test-access".to_string()),
            state_s3_secret_access_key: Some("test-secret".to_string()),
            state_s3_endpoint: Some("http://localhost:9000".to_string()),
            state_s3_session_token: Some("test-session".to_string()),
            state_s3_prefix: Some("states".to_string()),
            ..ServerConfig::default()
        };

        assert_eq!(
            config.resolve_state_store_kind().unwrap(),
            StateStoreKind::S3 {
                config: S3Config {
                    bucket: "worldforge-states".to_string(),
                    region: "us-east-1".to_string(),
                    endpoint: Some("http://localhost:9000".to_string()),
                    access_key_id: "test-access".to_string(),
                    secret_access_key: "test-secret".to_string(),
                    session_token: Some("test-session".to_string()),
                    prefix: "states".to_string(),
                },
                format: StateFileFormat::Json,
            }
        );
    }

    #[test]
    fn test_server_config_requires_redis_url() {
        let config = ServerConfig {
            state_backend: "redis".to_string(),
            ..ServerConfig::default()
        };

        let error = config.resolve_state_store_kind().unwrap_err();
        assert!(error
            .to_string()
            .contains("--state-redis-url is required when state_backend is set to redis"));
    }

    #[test]
    fn test_server_config_requires_s3_bucket() {
        let config = ServerConfig {
            state_backend: "s3".to_string(),
            ..ServerConfig::default()
        };

        let error = config.resolve_state_store_kind().unwrap_err();
        assert!(error
            .to_string()
            .contains("--state-s3-bucket is required when state_backend is set to s3"));
    }

    #[test]
    fn test_server_config_resolves_msgpack_file_store() {
        let config = ServerConfig {
            state_dir: "/tmp/worldforge".to_string(),
            state_file_format: "msgpack".to_string(),
            ..ServerConfig::default()
        };

        assert_eq!(
            config.resolve_state_store_kind().unwrap(),
            StateStoreKind::FileWithFormat {
                path: "/tmp/worldforge".into(),
                format: StateFileFormat::MessagePack,
            }
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
    async fn test_route_list_providers_with_health() {
        let state = test_state();
        let (status, body) = route("GET", "/v1/providers?health=true", "", &state).await;
        assert_eq!(status, 200);
        assert!(body.contains("mock"));
        assert!(body.contains("\"healthy\":true"));
    }

    #[tokio::test]
    async fn test_route_list_providers_with_health_and_capability_filter() {
        let state = test_state();
        let (status, body) = route(
            "GET",
            "/v1/providers?capability=planning&health=1",
            "",
            &state,
        )
        .await;
        assert_eq!(status, 200);
        assert!(body.contains("mock"));
        assert!(body.contains("\"status\":"));
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
    async fn test_route_create_world_from_prompt_bootstraps_scene() {
        let state = test_state();
        let body = r#"{"prompt":"A kitchen with a mug","name":"seeded-kitchen","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["name"], "seeded-kitchen");
        assert_eq!(value["data"]["description"], "A kitchen with a mug");
        assert!(value["data"]["object_count"].as_u64().unwrap() >= 2);

        let world_id = value["data"]["id"].as_str().unwrap();
        let (status, world) = route("GET", &format!("/v1/worlds/{world_id}"), "", &state).await;
        assert_eq!(status, 200);

        let loaded: serde_json::Value = serde_json::from_str(&world).unwrap();
        assert_eq!(
            loaded["data"]["metadata"]["description"],
            "A kitchen with a mug"
        );
        assert!(
            loaded["data"]["scene"]["objects"]
                .as_object()
                .unwrap()
                .len()
                >= 2
        );
    }

    #[tokio::test]
    async fn test_route_import_world_persists_cache_and_store() {
        let state = test_state();
        let mut source = WorldState::new("original-world", "mock");
        source.metadata.description = "Imported from snapshot".to_string();
        source.scene.add_object(SceneObject::new(
            "anchor",
            Pose::default(),
            BBox {
                min: Position {
                    x: -0.5,
                    y: 0.0,
                    z: -0.5,
                },
                max: Position {
                    x: 0.5,
                    y: 1.0,
                    z: 0.5,
                },
            },
        ));
        let body = serde_json::json!({
            "state": source,
        });

        let (status, resp) = route("POST", "/v1/worlds/import", &body.to_string(), &state).await;
        assert_eq!(status, 201);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();
        assert_eq!(value["data"]["metadata"]["name"], "original-world");
        assert_eq!(
            value["data"]["metadata"]["description"],
            "Imported from snapshot"
        );

        let parsed_id = id.parse::<WorldId>().unwrap();
        let stored = state.store.load(&parsed_id).await.unwrap();
        assert_eq!(stored.id.to_string(), id);
        assert_eq!(stored.metadata.description, "Imported from snapshot");

        let cache = state.worlds.read().await;
        assert!(cache.contains_key(&parsed_id));
        assert_eq!(
            cache.get(&parsed_id).unwrap().metadata.description,
            "Imported from snapshot"
        );
    }

    #[tokio::test]
    async fn test_route_import_world_allows_name_override() {
        let state = test_state();
        let source = WorldState::new("original-world", "mock");
        let body = serde_json::json!({
            "state": source,
            "name": "renamed-world",
        });

        let (status, resp) = route("POST", "/v1/worlds/import", &body.to_string(), &state).await;
        assert_eq!(status, 201);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();
        assert_eq!(value["data"]["metadata"]["name"], "renamed-world");

        let parsed_id = id.parse::<WorldId>().unwrap();
        let stored = state.store.load(&parsed_id).await.unwrap();
        assert_eq!(stored.metadata.name, "renamed-world");

        let cache = state.worlds.read().await;
        assert_eq!(
            cache.get(&parsed_id).unwrap().metadata.name,
            "renamed-world"
        );
    }

    #[tokio::test]
    async fn test_route_import_world_with_new_id_creates_new_snapshot() {
        let state = test_state();
        let source = WorldState::new("source-world", "mock");
        let original_id = source.id;
        state.store.save(&source).await.unwrap();
        let body = serde_json::json!({
            "state": source,
            "new_id": true,
        });

        let (status, resp) = route("POST", "/v1/worlds/import", &body.to_string(), &state).await;
        assert_eq!(status, 201);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let imported_id = value["data"]["id"]
            .as_str()
            .unwrap()
            .parse::<WorldId>()
            .unwrap();
        assert_ne!(imported_id, original_id);

        let imported = state.store.load(&imported_id).await.unwrap();
        assert_eq!(imported.metadata.name, "source-world");

        let original = state.store.load(&original_id).await.unwrap();
        assert_eq!(original.id, original_id);

        let cache = state.worlds.read().await;
        assert!(cache.contains_key(&imported_id));
        assert!(!cache.contains_key(&original_id));
    }

    #[tokio::test]
    async fn test_route_export_world_json_roundtrip() {
        let state = test_state();
        let source = sample_export_state("export-json-world");
        let source_id = source.id;
        state.store.save(&source).await.unwrap();

        let (status, resp) = route(
            "GET",
            &format!("/v1/worlds/{source_id}/export?format=json"),
            "",
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["id"], source_id.to_string());
        assert_eq!(value["data"]["format"], "json");
        assert_eq!(value["data"]["encoding"], "utf-8");
        assert_eq!(
            value["data"]["sha256"],
            encode_hex(&sha256_hash(
                value["data"]["snapshot"].as_str().unwrap().as_bytes()
            ))
        );

        let snapshot = value["data"]["snapshot"].as_str().unwrap();
        let restored: WorldState = serde_json::from_str(snapshot).unwrap();
        assert_eq!(restored.metadata.description, "portable snapshot");
        assert_eq!(restored.scene.objects.len(), source.scene.objects.len());

        let import_body = serde_json::json!({
            "format": "json",
            "encoding": "utf-8",
            "sha256": value["data"]["sha256"],
            "snapshot": snapshot,
            "new_id": true,
            "name": "imported-json-world",
        });

        let (status, imported_resp) = route(
            "POST",
            "/v1/worlds/import",
            &import_body.to_string(),
            &state,
        )
        .await;
        assert_eq!(status, 201);

        let imported_value: serde_json::Value = serde_json::from_str(&imported_resp).unwrap();
        assert_eq!(
            imported_value["data"]["metadata"]["name"],
            "imported-json-world"
        );
        assert_eq!(
            imported_value["data"]["metadata"]["description"],
            "portable snapshot"
        );
        assert_ne!(imported_value["data"]["id"], source_id.to_string());
    }

    #[tokio::test]
    async fn test_route_export_world_msgpack_roundtrip() {
        let state = test_state();
        let source = sample_export_state("export-msgpack-world");
        let source_id = source.id;
        state.store.save(&source).await.unwrap();

        let (status, resp) = route(
            "GET",
            &format!("/v1/worlds/{source_id}/export?format=msgpack"),
            "",
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["id"], source_id.to_string());
        assert_eq!(value["data"]["format"], "msgpack");
        assert_eq!(value["data"]["encoding"], "hex");
        assert_eq!(
            value["data"]["sha256"],
            encode_hex(&sha256_hash(
                &decode_hex(value["data"]["snapshot"].as_str().unwrap()).unwrap()
            ))
        );

        let snapshot = value["data"]["snapshot"].as_str().unwrap();
        assert!(!snapshot.is_empty());
        assert!(snapshot.chars().all(|c| c.is_ascii_hexdigit()));

        let import_body = serde_json::json!({
            "format": "msgpack",
            "encoding": "hex",
            "sha256": value["data"]["sha256"],
            "snapshot": snapshot,
            "name": "imported-msgpack-world",
        });

        let (status, imported_resp) = route(
            "POST",
            "/v1/worlds/import",
            &import_body.to_string(),
            &state,
        )
        .await;
        assert_eq!(status, 201);

        let imported_value: serde_json::Value = serde_json::from_str(&imported_resp).unwrap();
        assert_eq!(
            imported_value["data"]["metadata"]["name"],
            "imported-msgpack-world"
        );
        assert_eq!(
            imported_value["data"]["metadata"]["description"],
            "portable snapshot"
        );
        assert_eq!(
            imported_value["data"]["scene"]["objects"]
                .as_object()
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn test_route_import_world_rejects_checksum_mismatch() {
        let state = test_state();
        let source = sample_export_state("import-checksum-world");
        let source_id = source.id;
        state.store.save(&source).await.unwrap();

        let (status, resp) = route(
            "GET",
            &format!("/v1/worlds/{source_id}/export?format=json"),
            "",
            &state,
        )
        .await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();

        let import_body = serde_json::json!({
            "format": "json",
            "encoding": "utf-8",
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "snapshot": value["data"]["snapshot"],
        });

        let (status, resp) = route(
            "POST",
            "/v1/worlds/import",
            &import_body.to_string(),
            &state,
        )
        .await;
        assert_eq!(status, 400);
        assert!(resp.contains("sha256 mismatch"));
    }

    #[tokio::test]
    async fn test_route_import_world_rejects_wrong_encoding() {
        let state = test_state();
        let source = sample_export_state("import-encoding-world");
        let source_id = source.id;
        state.store.save(&source).await.unwrap();

        let (status, resp) = route(
            "GET",
            &format!("/v1/worlds/{source_id}/export?format=msgpack"),
            "",
            &state,
        )
        .await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();

        let import_body = serde_json::json!({
            "format": "msgpack",
            "encoding": "utf-8",
            "snapshot": value["data"]["snapshot"],
        });

        let (status, resp) = route(
            "POST",
            "/v1/worlds/import",
            &import_body.to_string(),
            &state,
        )
        .await;
        assert_eq!(status, 400);
        assert!(resp.contains("does not match format"));
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
    async fn test_route_create_world_requires_name_or_prompt() {
        let state = test_state();
        let body = r#"{"provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 400);
        assert!(resp.contains("name or prompt"));
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
    async fn test_route_get_world_history() {
        let state = test_state();
        let id = seed_world(&state, "history-world", "mock").await;
        let predict_body =
            r#"{"action":{"SetWeather":{"weather":"Rain"}},"disable_guardrails":true}"#;

        let (status, _) = route(
            "POST",
            &format!("/v1/worlds/{id}/predict"),
            predict_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let (status, resp) = route("GET", &format!("/v1/worlds/{id}/history"), "", &state).await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let entries = value["data"].as_array().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[0]["action"].is_null());
        assert_eq!(entries[1]["provider"], "mock");
        assert_eq!(entries[1]["prediction"]["model"], "mock-v2");
    }

    #[tokio::test]
    async fn test_route_restore_world_history_checkpoint() {
        let state = test_state();
        let id = seed_world(&state, "restore-world", "mock").await;
        let predict_body =
            r#"{"action":{"SetWeather":{"weather":"Rain"}},"disable_guardrails":true}"#;

        let (status, _) = route(
            "POST",
            &format!("/v1/worlds/{id}/predict"),
            predict_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let restore_body = r#"{"history_index":0}"#;
        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/restore"),
            restore_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["time"]["step"], 0);
        assert_eq!(
            value["data"]["history"]["states"].as_array().unwrap().len(),
            1
        );

        let persisted = state.store.load(&id).await.unwrap();
        assert_eq!(persisted.time.step, 0);
        assert_eq!(persisted.history.len(), 1);
    }

    #[tokio::test]
    async fn test_route_restore_world_history_rejects_unknown_index() {
        let state = test_state();
        let id = seed_world(&state, "restore-error", "mock").await;

        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/restore"),
            r#"{"history_index":99}"#,
            &state,
        )
        .await;
        assert_eq!(status, 400);
        assert!(resp.contains("history index 99 out of range"));
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
        let expected_mesh = serde_json::to_value(sample_mesh()).unwrap();
        let expected_embedding = serde_json::to_value(sample_visual_embedding()).unwrap();

        let object_body = serde_json::json!({
            "name": "crate",
            "position": { "x": 0.0, "y": 1.0, "z": 2.0 },
            "bbox": {
                "min": { "x": -0.5, "y": -0.5, "z": -0.5 },
                "max": { "x": 0.5, "y": 0.5, "z": 0.5 }
            },
            "mesh": sample_mesh(),
            "visual_embedding": sample_visual_embedding(),
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
        assert_eq!(created["data"]["mesh"], expected_mesh);
        assert_eq!(created["data"]["visual_embedding"], expected_embedding);

        let (status, resp) = route("GET", &format!("/v1/worlds/{id}/objects"), "", &state).await;
        assert_eq!(status, 200);
        let listed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let objects = listed["data"].as_array().unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0]["semantic_label"], "storage");
        assert_eq!(objects[0]["mesh"], expected_mesh);
        assert_eq!(objects[0]["visual_embedding"], expected_embedding);
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
            },
            "mesh": sample_mesh(),
            "visual_embedding": sample_visual_embedding()
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
        let expected_mesh = serde_json::to_value(sample_mesh()).unwrap();
        let expected_embedding = serde_json::to_value(sample_visual_embedding()).unwrap();
        assert_eq!(shown["data"]["mesh"], expected_mesh);
        assert_eq!(shown["data"]["visual_embedding"], expected_embedding);

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
    async fn test_route_patch_object_updates_persisted_state() {
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
            "mesh": sample_mesh(),
            "visual_embedding": sample_visual_embedding()
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

        let patch_body = serde_json::json!({
            "position": { "x": 2.0, "y": 3.0, "z": 4.0 }
        })
        .to_string();

        let (status, resp) = route(
            "PATCH",
            &format!("/v1/worlds/{id}/objects/{object_id}"),
            &patch_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);
        let patched: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(patched["data"]["id"], object_id);
        assert_eq!(patched["data"]["pose"]["position"]["x"], 2.0);
        assert_eq!(patched["data"]["bbox"]["min"]["x"], 1.5);
        assert_eq!(patched["data"]["bbox"]["max"]["z"], 2.5);
        let expected_mesh = serde_json::to_value(sample_mesh()).unwrap();
        let expected_embedding = serde_json::to_value(sample_visual_embedding()).unwrap();
        let expected_embedding_tensor =
            serde_json::to_value(sample_visual_embedding_tensor()).unwrap();
        assert_eq!(patched["data"]["mesh"], expected_mesh);
        assert_eq!(patched["data"]["visual_embedding"], expected_embedding);

        let world_id = id.parse::<WorldId>().unwrap();
        let persisted = state.store.load(&world_id).await.unwrap();
        let world =
            worldforge_core::world::World::new(persisted, "mock", Arc::clone(&state.registry));
        let object = world.get_object(&object_id.parse().unwrap()).unwrap();
        assert_eq!(object.pose.position.x, 2.0);
        assert_eq!(object.bbox.min.x, 1.5);
        assert_eq!(object.bbox.max.z, 2.5);
        assert_eq!(
            serde_json::to_value(object.mesh.as_ref().unwrap()).unwrap(),
            serde_json::to_value(sample_mesh()).unwrap()
        );
        assert_eq!(
            serde_json::to_value(object.visual_embedding.as_ref().unwrap()).unwrap(),
            expected_embedding_tensor
        );
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
            2
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
    async fn test_route_predict_disable_guardrails_allows_colliding_scene() {
        let state = test_state();
        let id = seed_colliding_world(&state).await;
        let baseline = state.store.load(&id).await.unwrap();
        assert_eq!(baseline.history.len(), 1);
        let initial_entry = baseline.history.latest().unwrap();
        assert!(initial_entry.action.is_none());
        assert!(initial_entry.prediction.is_none());

        let pred_body = r#"{"action":{"SetWeather":{"weather":"Rain"}}}"#;

        let (status, _) = route(
            "POST",
            &format!("/v1/worlds/{id}/predict"),
            pred_body,
            &state,
        )
        .await;
        assert_eq!(status, 409);

        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/predict"),
            r#"{"action":{"SetWeather":{"weather":"Rain"}},"disable_guardrails":true}"#,
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let prediction: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(
            prediction["data"]["guardrail_results"],
            serde_json::json!([])
        );

        let persisted = state.store.load(&id).await.unwrap();
        assert_eq!(persisted.time.step, 1);
        assert_eq!(persisted.history.len(), 2);
        let transition = persisted.history.latest().unwrap();
        assert_eq!(transition.provider, "mock");
        assert!(transition.prediction.is_some());
        assert!(matches!(
            transition.action,
            Some(worldforge_core::action::Action::SetWeather {
                weather: worldforge_core::action::Weather::Rain
            })
        ));
    }

    #[tokio::test]
    async fn test_route_plan_attaches_verification_proof_when_requested() {
        let state = test_state();
        let id = seed_world(&state, "plan-verified", "mock").await;
        let body = serde_json::json!({
            "goal": "spawn cube",
            "provider": "missing",
            "fallback_provider": "mock",
            "verification_backend": "mock"
        })
        .to_string();

        let (status, resp) = route("POST", &format!("/v1/worlds/{id}/plan"), &body, &state).await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["verification_proof"]["backend"], "Mock");
        assert_eq!(
            value["data"]["verification_proof"]["proof_type"]["GuardrailCompliance"]["all_passed"],
            true
        );
    }

    #[tokio::test]
    async fn test_route_execute_plan_commits_world_state() {
        let state = test_state();
        let body = r#"{"name":"execute_world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();

        let execute_body = serde_json::json!({
            "plan": {
                "actions": [
                    {
                        "Move": {
                            "target": { "x": 1.0, "y": 0.0, "z": 0.0 },
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

        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/execute-plan"),
            &execute_body,
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let execution: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(
            execution["data"]["predictions"].as_array().unwrap().len(),
            1
        );
        assert_eq!(execution["data"]["final_state"]["time"]["step"], 1);

        let (status, resp) = route("GET", &format!("/v1/worlds/{id}"), "", &state).await;
        assert_eq!(status, 200);
        let persisted: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(persisted["data"]["time"]["step"], 1);
        assert_eq!(
            persisted["data"]["history"]["states"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn test_route_execute_plan_guardrail_failure_is_atomic() {
        let state = test_state();
        let body = r#"{"name":"execute_guardrails","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();

        let object_body = serde_json::json!({
            "name": "ball",
            "position": { "x": 0.0, "y": 0.5, "z": 0.0 },
            "bbox": {
                "min": { "x": -0.1, "y": 0.4, "z": -0.1 },
                "max": { "x": 0.1, "y": 0.6, "z": 0.1 }
            }
        })
        .to_string();
        let (status, _) = route(
            "POST",
            &format!("/v1/worlds/{id}/objects"),
            &object_body,
            &state,
        )
        .await;
        assert_eq!(status, 201);
        let world_id = id.parse::<WorldId>().unwrap();
        let baseline = state.store.load(&world_id).await.unwrap();

        let execute_body = serde_json::json!({
            "plan": {
                "actions": [
                    {
                        "Move": {
                            "target": { "x": 1.0, "y": 0.0, "z": 0.0 },
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
            },
            "config": {
                "guardrails": [
                    {
                        "guardrail": {
                            "BoundaryConstraint": {
                                "bounds": {
                                    "min": { "x": -0.25, "y": -0.25, "z": -0.25 },
                                    "max": { "x": 0.25, "y": 0.25, "z": 0.25 }
                                }
                            }
                        },
                        "blocking": true
                    }
                ]
            }
        })
        .to_string();

        let (status, _) = route(
            "POST",
            &format!("/v1/worlds/{id}/execute-plan"),
            &execute_body,
            &state,
        )
        .await;
        assert_eq!(status, 409);

        let persisted = state.store.load(&world_id).await.unwrap();
        assert_eq!(persisted.time.step, baseline.time.step);
        assert_eq!(persisted.history.len(), baseline.history.len());
        assert_eq!(
            persisted
                .scene
                .find_object_by_name("ball")
                .unwrap()
                .pose
                .position,
            baseline
                .scene
                .find_object_by_name("ball")
                .unwrap()
                .pose
                .position
        );
    }

    #[tokio::test]
    async fn test_route_evaluate() {
        let state = test_state();
        let body = r#"{"suite":"physics"}"#;
        let id = seed_world(&state, "eval-world", "mock").await;
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
    async fn test_route_evaluate_uses_persisted_world_state() {
        let state = test_state();
        let (world_id, object_id) = seed_world_with_object(
            &state,
            "eval-overlay-world",
            "mock",
            "cube",
            Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        )
        .await;

        let mut fixture_state = WorldState::new("eval-overlay-world", "mock");
        let mut fixture_object = SceneObject::new(
            "cube",
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
                    x: -0.1,
                    y: -0.1,
                    z: -0.1,
                },
                max: Position {
                    x: 0.1,
                    y: 0.1,
                    z: 0.1,
                },
            },
        );
        fixture_object.id = object_id;
        fixture_state.scene.add_object(fixture_object);

        let suite = EvalSuite {
            name: "World Overlay".to_string(),
            scenarios: vec![worldforge_eval::EvalScenario {
                name: "object_position".to_string(),
                description: "Evaluate the persisted world state".to_string(),
                initial_state: fixture_state.clone(),
                actions: Vec::new(),
                expected_outcomes: vec![worldforge_eval::ExpectedOutcome::ObjectPosition {
                    name: "cube".to_string(),
                    position: Position {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    tolerance: 0.05,
                }],
                ground_truth: None,
            }],
            dimensions: vec![worldforge_eval::EvalDimension::SpatialConsistency],
            providers: vec![],
        };

        let body = serde_json::json!({
            "suite_definition": suite,
            "providers": ["mock"],
        })
        .to_string();

        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{world_id}/evaluate"),
            &body,
            &state,
        )
        .await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["results"][0]["outcomes"][0]["passed"], true);

        let mut mutated = state.store.load(&world_id).await.unwrap();
        mutated.scene.set_object_position(
            &object_id,
            Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        );
        state.store.save(&mutated).await.unwrap();

        let (status, resp) = route(
            "POST",
            &format!("/v1/worlds/{world_id}/evaluate"),
            &body,
            &state,
        )
        .await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["results"][0]["outcomes"][0]["passed"], false);
    }

    #[tokio::test]
    async fn test_route_evaluate_invalid_suite() {
        let state = test_state();
        let body = r#"{"suite":"nonexistent"}"#;
        let id = seed_world(&state, "eval-world", "mock").await;
        let (status, _) = route("POST", &format!("/v1/worlds/{id}/evaluate"), body, &state).await;
        assert_eq!(status, 400);
    }

    #[tokio::test]
    async fn test_route_evaluate_uses_suite_default_providers() {
        let state = test_state_with_providers(&["mock", "alt-mock"]);
        let mut suite = EvalSuite::physics_standard();
        suite.providers = vec!["alt-mock".to_string()];
        let body = serde_json::json!({
            "suite_definition": suite,
        })
        .to_string();
        let id = seed_world(&state, "eval-world", "mock").await;
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
    async fn test_route_evaluate_renders_markdown_report() {
        let state = test_state();
        let id = seed_world(&state, "eval-world", "mock").await;
        let body = r#"{"suite":"physics","report_format":"markdown"}"#;

        let (status, resp) =
            route("POST", &format!("/v1/worlds/{id}/evaluate"), body, &state).await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["format"], "markdown");
        assert_eq!(value["data"]["suite"], "Physics Standard");
        assert!(value["data"]["content"]
            .as_str()
            .unwrap()
            .contains("# Evaluation Report: Physics Standard"));
    }

    #[tokio::test]
    async fn test_route_evaluate_renders_csv_report() {
        let state = test_state();
        let id = seed_world(&state, "eval-world", "mock").await;
        let body = r#"{"suite":"physics","report_format":"csv"}"#;

        let (status, resp) =
            route("POST", &format!("/v1/worlds/{id}/evaluate"), body, &state).await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["format"], "csv");
        assert_eq!(value["data"]["suite"], "Physics Standard");
        let content = value["data"]["content"].as_str().unwrap();
        assert!(content
            .lines()
            .next()
            .unwrap()
            .contains("suite,provider,scenario"));
        assert!(content.contains("Physics Standard,mock,object_drop"));
    }

    #[tokio::test]
    async fn test_route_evaluate_missing_world_returns_not_found() {
        let state = test_state();
        let body = r#"{"suite":"physics"}"#;
        let id = uuid::Uuid::new_v4();

        let (status, _) = route("POST", &format!("/v1/worlds/{id}/evaluate"), body, &state).await;

        assert_eq!(status, 404);
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
    async fn test_route_generate_uses_fallback_provider() {
        let state = test_state();
        let body = r#"{
            "prompt":"a cube rolling across the floor",
            "fallback_provider":"mock",
            "config":{"duration_seconds":2.0,"resolution":[320,180],"fps":12.0}
        }"#;
        let (status, resp) = route("POST", "/v1/providers/missing/generate", body, &state).await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["data"]["duration"], 2.0);
        assert_eq!(value["data"]["resolution"], serde_json::json!([320, 180]));
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
    async fn test_route_provider_reason() {
        let state = test_state();
        let body = serde_json::json!({
            "query": "how many objects are here?",
            "state": worldforge_core::state::WorldState::new("reason-world", "mock"),
        });
        let (status, resp) = route(
            "POST",
            "/v1/providers/mock/reason",
            &body.to_string(),
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(value["data"]["answer"].as_str().unwrap().contains("empty"));
    }

    #[tokio::test]
    async fn test_route_provider_reason_uses_fallback_provider() {
        let state = test_state();
        let body = serde_json::json!({
            "query": "what do you see?",
            "video": {
                "frames": [],
                "fps": 15.0,
                "resolution": [320, 240],
                "duration": 1.5
            },
            "fallback_provider": "mock"
        });
        let (status, resp) = route(
            "POST",
            "/v1/providers/missing/reason",
            &body.to_string(),
            &state,
        )
        .await;
        assert_eq!(status, 200);

        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(value["data"]["answer"]
            .as_str()
            .unwrap()
            .contains("echo the query"));
    }

    #[tokio::test]
    async fn test_route_provider_reason_rejects_missing_inputs() {
        let state = test_state();
        let body = r#"{"query":"what happens?"}"#;
        let (status, resp) = route("POST", "/v1/providers/mock/reason", body, &state).await;
        assert_eq!(status, 400);
        assert!(resp.contains("state and/or video"));
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
    async fn test_route_reason_uses_fallback_provider() {
        let state = test_state();
        let body = r#"{"name":"reason_world","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();

        let reason_body =
            r#"{"query":"what happens next?","provider":"missing","fallback_provider":"mock"}"#;
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
            "provider": "missing",
            "fallback_provider": "mock",
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
    async fn test_route_verify_guardrail_from_structured_goal() {
        let state = test_state();
        let body = r#"{"name":"verify_structured_guardrail","provider":"mock"}"#;
        let (status, resp) = route("POST", "/v1/worlds", body, &state).await;
        assert_eq!(status, 201);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let id = value["data"]["id"].as_str().unwrap();

        let object_body = r#"{
            "name":"ball",
            "position":{"x":0.0,"y":0.5,"z":0.0},
            "bbox":{"min":{"x":-0.1,"y":0.4,"z":-0.1},"max":{"x":0.1,"y":0.6,"z":0.1}}
        }"#;
        let (status, object_resp) = route(
            "POST",
            &format!("/v1/worlds/{id}/objects"),
            object_body,
            &state,
        )
        .await;
        assert_eq!(status, 201);
        let object_json: serde_json::Value = serde_json::from_str(&object_resp).unwrap();
        let object_id = object_json["data"]["id"].as_str().unwrap();

        let verify_body = serde_json::json!({
            "proof_type": "guardrail",
            "provider": "missing",
            "fallback_provider": "mock",
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
            r#"{{"world_id":"{}","action":{{"Move":{{"target":{{"x":1.0,"y":0.0,"z":0.0}},"speed":1.0}}}},"providers":["mock","mock"]}}"#,
            id
        );
        let (status, resp) = route("POST", "/v1/compare", &cmp_body, &state).await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(
            value["data"]["comparison"]["scores"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            value["data"]["comparison"]["pairwise_agreements"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert!(value["data"]["comparison"]["scores"][0]["state"]["output_object_count"].is_u64());
        assert!(value["data"]["comparison"]["scores"][0]["quality_score"].is_number());
        assert!(value["data"]["comparison"]["consensus"]["shared_object_count"].is_u64());
    }

    #[tokio::test]
    async fn test_route_compare_requires_providers() {
        let state = test_state();
        let id = seed_world(&state, "cmp-world", "mock").await;
        let body = serde_json::json!({
            "world_id": id,
            "action": {
                "Move": {
                    "target": { "x": 1.0, "y": 0.0, "z": 0.0 },
                    "speed": 1.0
                }
            },
            "providers": [],
        })
        .to_string();

        let (status, resp) = route("POST", "/v1/compare", &body, &state).await;

        assert_eq!(status, 400);
        assert!(resp.contains("at least one provider"));
    }
}
