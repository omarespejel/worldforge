//! WorldForge CLI
//!
//! Command-line interface for interacting with world foundation models.
//! Supports world creation, prediction, planning, evaluation, and comparison.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::de::DeserializeOwned;
use serde::Serialize;

use worldforge_core::action::{Action, Weather};
use worldforge_core::error::WorldForgeError;
use worldforge_core::guardrail::{Guardrail, GuardrailConfig};
use worldforge_core::prediction::{
    ComparisonReportFormat, MultiPrediction, Plan, PlanExecution, PlanGoal, PlanGoalInput,
    PlanRequest, PlannerOptions, PlannerType, Prediction, PredictionConfig,
    PredictionSamplingMetadata, StoredPlanRecord,
};
use worldforge_core::provider::{
    EmbeddingInput, GenerationConfig, GenerationPrompt, Operation, ProviderDescriptor,
    ProviderHealthReport, ProviderRegistry, ReasoningInput, ReasoningOutput, SpatialControls,
    TransferConfig, WorldModelProvider,
};
use worldforge_core::scene::{PhysicsProperties, SceneObject, SceneObjectPatch};
use worldforge_core::state::{
    deserialize_world_state, infer_state_file_format, inspect_world_state_snapshot,
    serialize_world_state, DynStateStore, S3Config, StateFileFormat as CoreStateFileFormat,
    StateHistory, StateStore, StateStoreKind, WorldState, WORLD_STATE_SNAPSHOT_SCHEMA_VERSION,
};
use worldforge_core::types::{BBox, Pose, Position, Rotation, Vec3, Velocity, VideoClip};
use worldforge_eval::{EvalReportFormat, EvalSuite};
use worldforge_verify::{
    prove_guardrail_plan, prove_inference_transition, prove_latest_inference,
    prove_prediction_inference, prove_provenance, verifier_for_backend as verify_backend_resolver,
    verify_bundle, verify_proof, BundleVerificationReport, VerificationBackend, VerificationBundle,
    VerificationResult, ZkProof, ZkVerifier,
};

/// Persistence backend used by the CLI.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum StateBackend {
    /// Store world states as JSON or MessagePack files in a directory.
    File,
    /// Store world states in a SQLite database file.
    Sqlite,
    /// Store world states in a Redis database.
    Redis,
    /// Store world states in an S3 bucket.
    S3,
}

impl StateBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Sqlite => "sqlite",
            Self::Redis => "redis",
            Self::S3 => "s3",
        }
    }
}

/// Serialization format for file-backed world persistence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum StateFileFormat {
    /// Persist human-readable JSON files.
    Json,
    /// Persist compact MessagePack files.
    Msgpack,
}

impl StateFileFormat {
    fn as_core(self) -> CoreStateFileFormat {
        match self {
            Self::Json => CoreStateFileFormat::Json,
            Self::Msgpack => CoreStateFileFormat::MessagePack,
        }
    }
}

#[derive(Clone, Copy)]
struct CliStateStoreConfig<'a> {
    state_dir: &'a Path,
    state_backend: StateBackend,
    state_file_format: StateFileFormat,
    state_db_path: Option<&'a Path>,
    state_redis_url: Option<&'a str>,
    state_s3_bucket: Option<&'a str>,
    state_s3_region: Option<&'a str>,
    state_s3_access_key_id: Option<&'a str>,
    state_s3_secret_access_key: Option<&'a str>,
    state_s3_endpoint: Option<&'a str>,
    state_s3_session_token: Option<&'a str>,
    state_s3_prefix: Option<&'a str>,
}

impl<'a> CliStateStoreConfig<'a> {
    fn from_cli(cli: &'a Cli) -> Self {
        Self {
            state_dir: &cli.state_dir,
            state_backend: cli.state_backend,
            state_file_format: cli.state_file_format,
            state_db_path: cli.state_db_path.as_deref(),
            state_redis_url: cli.state_redis_url.as_deref(),
            state_s3_bucket: cli.state_s3_bucket.as_deref(),
            state_s3_region: cli.state_s3_region.as_deref(),
            state_s3_access_key_id: cli.state_s3_access_key_id.as_deref(),
            state_s3_secret_access_key: cli.state_s3_secret_access_key.as_deref(),
            state_s3_endpoint: cli.state_s3_endpoint.as_deref(),
            state_s3_session_token: cli.state_s3_session_token.as_deref(),
            state_s3_prefix: cli.state_s3_prefix.as_deref(),
        }
    }
}

/// Verification backend used for proof generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum VerifyBackend {
    /// Deterministic local mock backend.
    Mock,
    /// Deterministic EZKL compatibility backend.
    Ezkl,
    /// Deterministic STARK compatibility backend.
    Stark,
}

impl VerifyBackend {
    fn as_core(self) -> VerificationBackend {
        match self {
            Self::Mock => VerificationBackend::Mock,
            Self::Ezkl => VerificationBackend::Ezkl,
            Self::Stark => VerificationBackend::Stark,
        }
    }
}

/// Operation type for provider cost estimation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum EstimateOperationKind {
    /// Estimate a forward prediction call.
    Predict,
    /// Estimate a prompt-to-video generation call.
    Generate,
    /// Estimate a reasoning request.
    Reason,
    /// Estimate a transfer call.
    Transfer,
}

/// WorldForge — orchestration layer for world foundation models.
#[derive(Parser)]
#[command(name = "worldforge", version, about)]
pub struct Cli {
    /// State storage directory for file mode and the default SQLite location.
    #[arg(long, default_value = ".worldforge", global = true)]
    pub state_dir: PathBuf,

    /// Persistence backend for world state.
    #[arg(long, value_enum, default_value_t = StateBackend::File, global = true)]
    pub state_backend: StateBackend,

    /// Serialization format for file-backed state storage.
    #[arg(long, value_enum, default_value_t = StateFileFormat::Json, global = true)]
    pub state_file_format: StateFileFormat,

    /// Explicit SQLite database path when using the sqlite backend.
    #[arg(long, global = true)]
    pub state_db_path: Option<PathBuf>,

    /// Explicit Redis connection URL when using the redis backend.
    #[arg(long, global = true)]
    pub state_redis_url: Option<String>,

    /// Explicit S3 bucket name when using the s3 backend.
    #[arg(long = "state-s3-bucket", global = true)]
    pub state_s3_bucket: Option<String>,

    /// Explicit S3 region when using the s3 backend.
    #[arg(long = "state-s3-region", global = true)]
    pub state_s3_region: Option<String>,

    /// Explicit S3 access key ID when using the s3 backend.
    #[arg(long = "state-s3-access-key-id", global = true)]
    pub state_s3_access_key_id: Option<String>,

    /// Explicit S3 secret access key when using the s3 backend.
    #[arg(long = "state-s3-secret-access-key", global = true)]
    pub state_s3_secret_access_key: Option<String>,

    /// Optional custom S3 endpoint when using the s3 backend.
    #[arg(long = "state-s3-endpoint", global = true)]
    pub state_s3_endpoint: Option<String>,

    /// Optional S3 session token when using the s3 backend.
    #[arg(long = "state-s3-session-token", global = true)]
    pub state_s3_session_token: Option<String>,

    /// Optional S3 object-key prefix when using the s3 backend.
    #[arg(long = "state-s3-prefix", global = true)]
    pub state_s3_prefix: Option<String>,

    /// Log verbosity level.
    #[arg(long, default_value = "info", global = true)]
    pub log_level: String,

    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI commands.
#[derive(Subcommand)]
pub enum Commands {
    /// Create a new world.
    Create {
        /// Text description of the world.
        #[arg(long)]
        prompt: String,
        /// Optional explicit world name. Defaults to a name derived from the prompt.
        #[arg(long)]
        name: Option<String>,
        /// Provider to use.
        #[arg(long, default_value = "mock")]
        provider: String,
    },

    /// Predict the next state after an action.
    Predict {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Action description (e.g. "move 1 0 0", "set-weather rain").
        #[arg(long)]
        action: String,
        /// Number of prediction steps.
        #[arg(long, default_value = "1")]
        steps: u32,
        /// Provider to use.
        #[arg(long, default_value = "mock")]
        provider: String,
        /// Number of prediction samples to request.
        #[arg(long)]
        num_samples: Option<u32>,
        /// Optional fallback provider if the primary provider fails.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// Maximum time to wait for a provider response before timing out.
        #[arg(long)]
        timeout_ms: Option<u64>,
        /// Disable WorldForge's automatic guardrail checks.
        #[arg(long, default_value_t = false)]
        disable_guardrails: bool,
    },

    /// Generate a video clip directly from a prompt.
    Generate {
        /// Prompt describing the desired video.
        #[arg(long)]
        prompt: String,
        /// Provider to use.
        #[arg(long, default_value = "mock")]
        provider: String,
        /// Optional fallback provider if the primary provider fails.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// Optional negative prompt.
        #[arg(long)]
        negative_prompt: Option<String>,
        /// Output duration in seconds.
        #[arg(long, default_value = "4.0")]
        duration_seconds: f64,
        /// Output width in pixels.
        #[arg(long, default_value = "1280")]
        width: u32,
        /// Output height in pixels.
        #[arg(long, default_value = "720")]
        height: u32,
        /// Output frames per second.
        #[arg(long, default_value = "24.0")]
        fps: f32,
        /// Sampling temperature.
        #[arg(long, default_value = "1.0")]
        temperature: f32,
        /// Optional random seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Optional path to write the generated clip JSON payload.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Embed text and/or video input with a provider.
    Embed {
        /// Provider to use.
        #[arg(long, default_value = "mock")]
        provider: String,
        /// Optional fallback provider if the primary provider fails.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// Optional text to embed.
        #[arg(long)]
        text: Option<String>,
        /// Optional JSON file containing the source `VideoClip`.
        #[arg(long)]
        video_json: Option<PathBuf>,
        /// Optional path to write the embedding JSON payload.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Transfer spatial controls onto an existing source clip.
    Transfer {
        /// Provider to use.
        #[arg(long, default_value = "mock")]
        provider: String,
        /// Optional fallback provider if the primary provider fails.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// JSON file containing the source `VideoClip`.
        #[arg(long)]
        source_json: PathBuf,
        /// Optional JSON file containing `SpatialControls`.
        #[arg(long)]
        controls_json: Option<PathBuf>,
        /// Optional path to write the transferred clip JSON payload.
        #[arg(long)]
        output_json: Option<PathBuf>,
        /// Output width in pixels.
        #[arg(long, default_value = "1280")]
        width: u32,
        /// Output height in pixels.
        #[arg(long, default_value = "720")]
        height: u32,
        /// Output frames per second.
        #[arg(long, default_value = "24.0")]
        fps: f32,
        /// Spatial control strength.
        #[arg(long, default_value = "0.8")]
        control_strength: f32,
    },

    /// Ask a provider to reason about the current world state.
    Reason {
        /// Optional world ID. When omitted, supply `--state-json` and/or `--video-json`.
        #[arg(long)]
        world: Option<String>,
        /// Optional JSON file containing a serialized `WorldState`.
        #[arg(long)]
        state_json: Option<PathBuf>,
        /// Optional JSON file containing a serialized `VideoClip`.
        #[arg(long)]
        video_json: Option<PathBuf>,
        /// Natural-language reasoning query.
        #[arg(long)]
        query: String,
        /// Optional provider override.
        #[arg(long)]
        provider: Option<String>,
        /// Optional fallback provider if the selected provider fails.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// Optional path to write the reasoning JSON payload.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// List all saved worlds.
    List,

    /// Show details of a world.
    Show {
        /// World ID.
        world: String,
    },

    /// Show recorded history entries for a world.
    History {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Optional path to write the history entries as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Restore a persisted world to a recorded history checkpoint.
    Restore {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Zero-based history checkpoint index to restore.
        #[arg(long)]
        history_index: usize,
        /// Optional path to write the restored world JSON payload.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Fork a persisted world into a new branch.
    Fork {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Optional zero-based history checkpoint index to fork from.
        #[arg(long)]
        history_index: Option<usize>,
        /// Optional replacement world name.
        #[arg(long)]
        name: Option<String>,
    },

    /// Delete a world.
    Delete {
        /// World ID.
        world: String,
    },

    /// Export a persisted world snapshot to JSON or MessagePack.
    Export {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Output snapshot path.
        #[arg(long)]
        output: PathBuf,
        /// Snapshot format. If omitted, inferred from the output path extension.
        #[arg(long, value_parser = parse_snapshot_format)]
        format: Option<CoreStateFileFormat>,
    },

    /// Import a world snapshot into the configured state store.
    Import {
        /// Snapshot input path.
        #[arg(long)]
        input: PathBuf,
        /// Snapshot format. If omitted, inferred from the input path extension.
        #[arg(long, value_parser = parse_snapshot_format)]
        format: Option<CoreStateFileFormat>,
        /// Assign a fresh world ID before saving.
        #[arg(long, default_value_t = false)]
        new_id: bool,
        /// Optional replacement world name.
        #[arg(long)]
        name: Option<String>,
    },

    /// Manage scene objects in a persisted world.
    Objects {
        #[command(subcommand)]
        command: ObjectCommands,
    },

    /// Manage stored plans in a persisted world.
    Plans {
        #[command(subcommand)]
        command: PlanCommands,
    },

    /// List registered providers and their capabilities.
    Providers {
        /// Optional capability filter (for example: predict, generate, embed, planning, depth).
        #[arg(long)]
        capability: Option<String>,
        /// Run live health checks for the listed providers.
        #[arg(long, default_value_t = false)]
        health: bool,
    },

    /// Estimate provider cost for an operation.
    Estimate {
        /// Provider name.
        #[arg(long, default_value = "mock")]
        provider: String,
        /// Operation kind to estimate.
        #[arg(long, value_enum, default_value_t = EstimateOperationKind::Predict)]
        operation: EstimateOperationKind,
        /// Prediction step count for `predict`.
        #[arg(long, default_value = "1")]
        steps: u32,
        /// Duration in seconds for `generate` or `transfer`.
        #[arg(long, default_value = "4.0")]
        duration_seconds: f64,
        /// Output width for `predict` or `generate`.
        #[arg(long, default_value = "1280")]
        width: u32,
        /// Output height for `predict` or `generate`.
        #[arg(long, default_value = "720")]
        height: u32,
    },

    /// Run an evaluation suite.
    Eval {
        /// Evaluation suite name.
        #[arg(long)]
        suite: Option<String>,
        /// JSON file containing a custom `EvalSuite` definition.
        #[arg(long)]
        suite_json: Option<PathBuf>,
        /// Optional world ID whose persisted state seeds each evaluation scenario.
        #[arg(long, conflicts_with = "world_snapshot")]
        world: Option<String>,
        /// Optional exported world snapshot (JSON or MessagePack) to seed each scenario.
        #[arg(long, value_name = "PATH", conflicts_with = "world")]
        world_snapshot: Option<PathBuf>,
        /// Optional comma-separated list of providers.
        ///
        /// When omitted, the suite's baked-in providers are used.
        #[arg(long)]
        providers: Option<String>,
        /// Optional number of samples to request per prediction during evaluation.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
        num_samples: Option<u32>,
        /// Print the built-in suite names and exit.
        #[arg(long, default_value_t = false)]
        list_suites: bool,
        /// Print the available evaluation metric names and exit.
        #[arg(long, default_value_t = false)]
        list_metrics: bool,
        /// Optional path to write the evaluation report as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
        /// Optional path to write the evaluation report as Markdown.
        #[arg(long)]
        output_markdown: Option<PathBuf>,
        /// Optional path to write the evaluation report as CSV.
        #[arg(long)]
        output_csv: Option<PathBuf>,
    },

    /// Compare predictions across providers.
    Compare {
        /// World ID.
        #[arg(
            long,
            required_unless_present_any = ["prediction_json", "world_snapshot"],
            conflicts_with = "world_snapshot"
        )]
        world: Option<String>,
        /// Exported world snapshot (JSON or MessagePack) to compare without persistence.
        #[arg(
            long,
            value_name = "PATH",
            required_unless_present_any = ["prediction_json", "world"],
            conflicts_with = "world"
        )]
        world_snapshot: Option<PathBuf>,
        /// Action description.
        #[arg(long, required_unless_present = "prediction_json")]
        action: Option<String>,
        /// Comma-separated list of providers.
        #[arg(long, required_unless_present = "prediction_json")]
        providers: Option<String>,
        /// One or more JSON files containing serialized `Prediction` payloads to compare directly.
        #[arg(
            long = "prediction-json",
            value_name = "PATH",
            action = clap::ArgAction::Append,
            conflicts_with_all = [
                "world",
                "world_snapshot",
                "action",
                "providers",
                "steps",
                "fallback_provider",
                "timeout_ms",
                "guardrails_json",
                "disable_guardrails",
            ],
        )]
        prediction_json: Vec<PathBuf>,
        /// Number of prediction steps to compare.
        #[arg(long, default_value = "1")]
        steps: u32,
        /// Optional fallback provider if a listed provider fails.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// Maximum time to wait for each provider response before timing out.
        #[arg(long)]
        timeout_ms: Option<u64>,
        /// Optional JSON file containing `Vec<GuardrailConfig>`.
        #[arg(long)]
        guardrails_json: Option<PathBuf>,
        /// Disable WorldForge's automatic guardrail checks.
        #[arg(long, default_value_t = false)]
        disable_guardrails: bool,
        /// Optional path to write the comparison report as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
        /// Optional path to write the comparison report as Markdown.
        #[arg(long)]
        output_markdown: Option<PathBuf>,
        /// Optional path to write the comparison report as CSV.
        #[arg(long)]
        output_csv: Option<PathBuf>,
    },

    /// Plan a sequence of actions to achieve a goal.
    Plan {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Goal description (natural language).
        #[arg(
            long,
            required_unless_present = "goal_json",
            conflicts_with = "goal_json"
        )]
        goal: Option<String>,
        /// JSON file containing either a bare goal string or a structured `PlanGoalInput`.
        #[arg(long, conflicts_with = "goal")]
        goal_json: Option<PathBuf>,
        /// Maximum number of planning steps.
        #[arg(long, default_value = "10")]
        max_steps: u32,
        /// Planning algorithm (sampling, cem, mpc, gradient).
        #[arg(long, default_value = "sampling")]
        planner: String,
        /// Optional sample count for sampling or MPC planners.
        #[arg(long)]
        num_samples: Option<u32>,
        /// Optional top-k survivor count for the sampling planner.
        #[arg(long)]
        top_k: Option<u32>,
        /// Optional population size for CEM.
        #[arg(long)]
        population_size: Option<u32>,
        /// Optional elite fraction for CEM.
        #[arg(long)]
        elite_fraction: Option<f32>,
        /// Optional iteration count for CEM or gradient planning.
        #[arg(long)]
        num_iterations: Option<u32>,
        /// Optional learning rate for gradient planning.
        #[arg(long)]
        learning_rate: Option<f32>,
        /// Optional horizon for MPC.
        #[arg(long)]
        horizon: Option<u32>,
        /// Optional replanning interval for MPC.
        #[arg(long)]
        replanning_interval: Option<u32>,
        /// Planning timeout in seconds.
        #[arg(long, default_value = "30")]
        timeout: f64,
        /// Provider to use.
        #[arg(long, default_value = "mock")]
        provider: String,
        /// Optional fallback provider if the primary provider fails.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// Attach a guardrail-compliance proof to the generated plan.
        #[arg(long, value_enum)]
        verify_backend: Option<VerifyBackend>,
        /// Optional JSON file containing `Vec<GuardrailConfig>`.
        #[arg(long)]
        guardrails_json: Option<PathBuf>,
        /// Disable WorldForge's automatic guardrail checks.
        #[arg(long, default_value_t = false)]
        disable_guardrails: bool,
        /// Optional path to write the generated `Plan` as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Execute a previously generated plan against a persisted world.
    ExecutePlan {
        /// World ID.
        #[arg(long)]
        world: String,
        /// JSON file containing a serialized `Plan`.
        #[arg(long, conflicts_with = "plan_id", required_unless_present = "plan_id")]
        plan_json: Option<PathBuf>,
        /// Stored plan ID persisted on the target world.
        #[arg(
            long,
            conflicts_with = "plan_json",
            required_unless_present = "plan_json"
        )]
        plan_id: Option<String>,
        /// Optional provider override. Defaults to the world's current provider.
        #[arg(long)]
        provider: Option<String>,
        /// Optional fallback provider if the primary provider fails.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// Number of prediction steps per action during execution.
        #[arg(long, default_value = "1")]
        steps: u32,
        /// Maximum time to wait for each provider response before timing out.
        #[arg(long)]
        timeout_ms: Option<u64>,
        /// Optional JSON file containing `Vec<GuardrailConfig>`.
        #[arg(long)]
        guardrails_json: Option<PathBuf>,
        /// Return per-step preview clips in the execution report.
        #[arg(long, default_value_t = false)]
        return_video: bool,
        /// Disable WorldForge's automatic guardrail checks.
        #[arg(long, default_value_t = false)]
        disable_guardrails: bool,
        /// Optional path to write the execution report as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Generate and verify a ZK proof for a plan.
    Verify {
        /// World ID for state-backed proofs or plan generation.
        #[arg(long)]
        world: Option<String>,
        /// Verification backend to use when generating proofs.
        #[arg(long, value_enum, default_value_t = VerifyBackend::Mock)]
        backend: VerifyBackend,
        /// Proof type: inference, guardrail, provenance.
        #[arg(long, default_value = "inference")]
        proof_type: String,
        /// JSON file containing the input `WorldState` for inference verification.
        #[arg(long)]
        input_state_json: Option<PathBuf>,
        /// JSON file containing the output `WorldState` for inference verification.
        #[arg(long)]
        output_state_json: Option<PathBuf>,
        /// JSON file containing a serialized `Prediction` for archived inference verification.
        #[arg(long)]
        prediction_json: Option<PathBuf>,
        /// JSON file containing a fully materialized `Plan` for guardrail verification.
        #[arg(long, conflicts_with = "plan_id")]
        plan_json: Option<PathBuf>,
        /// Stored plan ID persisted on the target world for guardrail verification.
        #[arg(long, conflicts_with = "plan_json")]
        plan_id: Option<String>,
        /// Natural-language goal used to generate a plan before guardrail verification.
        #[arg(long, conflicts_with = "goal_json")]
        goal: Option<String>,
        /// JSON file containing either a bare goal string or a structured `PlanGoalInput`.
        #[arg(long, conflicts_with = "goal")]
        goal_json: Option<PathBuf>,
        /// Maximum number of planning steps when generating a plan for verification.
        #[arg(long, default_value = "10")]
        max_steps: u32,
        /// Planning algorithm when generating a plan for guardrail verification.
        #[arg(long, default_value = "sampling")]
        planner: String,
        /// Optional sample count for sampling or MPC planners.
        #[arg(long)]
        num_samples: Option<u32>,
        /// Optional top-k survivor count for the sampling planner.
        #[arg(long)]
        top_k: Option<u32>,
        /// Optional population size for CEM.
        #[arg(long)]
        population_size: Option<u32>,
        /// Optional elite fraction for CEM.
        #[arg(long)]
        elite_fraction: Option<f32>,
        /// Optional iteration count for CEM or gradient planning.
        #[arg(long)]
        num_iterations: Option<u32>,
        /// Optional learning rate for gradient planning.
        #[arg(long)]
        learning_rate: Option<f32>,
        /// Optional horizon for MPC.
        #[arg(long)]
        horizon: Option<u32>,
        /// Optional replanning interval for MPC.
        #[arg(long)]
        replanning_interval: Option<u32>,
        /// Planning timeout in seconds when generating a plan for guardrail verification.
        #[arg(long, default_value = "30")]
        timeout: f64,
        /// Optional provider override for generated plans or history-backed inference proofs.
        #[arg(long)]
        provider: Option<String>,
        /// Optional fallback provider when a plan must be generated for verification.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// Optional JSON file containing `Vec<GuardrailConfig>` for generated plans.
        #[arg(long)]
        guardrails_json: Option<PathBuf>,
        /// Disable WorldForge's automatic guardrail checks.
        #[arg(long, default_value_t = false)]
        disable_guardrails: bool,
        /// Source label to attest for provenance proofs.
        #[arg(long, default_value = "worldforge-cli")]
        source_label: String,
        /// Optional path to write the verification bundle as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Re-verify a previously exported proof or verification bundle.
    VerifyProof {
        /// JSON file containing a raw `ZkProof`.
        #[arg(long, conflicts_with_all = ["inference_bundle_json", "guardrail_bundle_json", "provenance_bundle_json"])]
        proof_json: Option<PathBuf>,
        /// JSON file containing `VerificationBundle<InferenceArtifact>`.
        #[arg(long, conflicts_with_all = ["proof_json", "guardrail_bundle_json", "provenance_bundle_json"])]
        inference_bundle_json: Option<PathBuf>,
        /// JSON file containing `VerificationBundle<GuardrailArtifact>`.
        #[arg(long, conflicts_with_all = ["proof_json", "inference_bundle_json", "provenance_bundle_json"])]
        guardrail_bundle_json: Option<PathBuf>,
        /// JSON file containing `VerificationBundle<ProvenanceArtifact>`.
        #[arg(long, conflicts_with_all = ["proof_json", "inference_bundle_json", "guardrail_bundle_json"])]
        provenance_bundle_json: Option<PathBuf>,
        /// Optional path to write the verification report as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Check provider health.
    Health {
        /// Provider name (or "all").
        #[arg(default_value = "all")]
        provider: String,
    },

    /// Start the WorldForge REST API server.
    Serve {
        /// Address to bind the HTTP server to.
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,
    },
}

/// Object-management subcommands for persisted worlds.
#[derive(Subcommand)]
pub enum ObjectCommands {
    /// Add a new object to a world scene.
    Add {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Human-readable object name.
        #[arg(long)]
        name: String,
        /// Object position as `x y z`.
        #[arg(long, num_args = 3, allow_hyphen_values = true)]
        position: Vec<f32>,
        /// Bounding-box minimum as `x y z`.
        #[arg(long = "bbox-min", num_args = 3, allow_hyphen_values = true)]
        bbox_min: Vec<f32>,
        /// Bounding-box maximum as `x y z`.
        #[arg(long = "bbox-max", num_args = 3, allow_hyphen_values = true)]
        bbox_max: Vec<f32>,
        /// Optional velocity as `x y z`.
        #[arg(long, num_args = 3, allow_hyphen_values = true)]
        velocity: Option<Vec<f32>>,
        /// Optional semantic label.
        #[arg(long)]
        semantic_label: Option<String>,
        /// Optional path to a JSON-serialized `Mesh`.
        #[arg(long)]
        mesh_json: Option<PathBuf>,
        /// Optional path to a JSON-serialized `Tensor` for the visual embedding.
        #[arg(long)]
        visual_embedding_json: Option<PathBuf>,
        /// Optional mass in kilograms.
        #[arg(long)]
        mass: Option<f32>,
        /// Optional friction coefficient.
        #[arg(long)]
        friction: Option<f32>,
        /// Optional restitution coefficient.
        #[arg(long)]
        restitution: Option<f32>,
        /// Optional material name.
        #[arg(long)]
        material: Option<String>,
        /// Mark the object as immovable.
        #[arg(long = "static", default_value_t = false, action = clap::ArgAction::SetTrue)]
        is_static: bool,
        /// Mark the object as graspable.
        #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
        graspable: bool,
        /// Optional path to write the created object as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// List all objects in a world scene.
    List {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Optional path to write the object list as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Show one object in a world scene.
    Show {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Object ID.
        #[arg(long)]
        object_id: String,
        /// Optional path to write the object as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Update an object in a world scene.
    Update {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Object ID.
        #[arg(long)]
        object_id: String,
        /// Optional replacement name.
        #[arg(long)]
        name: Option<String>,
        /// Optional replacement position as `x y z`.
        #[arg(long, num_args = 3, allow_hyphen_values = true)]
        position: Option<Vec<f32>>,
        /// Optional replacement rotation as quaternion `w x y z`.
        #[arg(long, num_args = 4, allow_hyphen_values = true)]
        rotation: Option<Vec<f32>>,
        /// Optional replacement bounding-box minimum as `x y z`.
        #[arg(long = "bbox-min", num_args = 3, allow_hyphen_values = true)]
        bbox_min: Option<Vec<f32>>,
        /// Optional replacement bounding-box maximum as `x y z`.
        #[arg(long = "bbox-max", num_args = 3, allow_hyphen_values = true)]
        bbox_max: Option<Vec<f32>>,
        /// Optional replacement velocity as `x y z`.
        #[arg(long, num_args = 3, allow_hyphen_values = true)]
        velocity: Option<Vec<f32>>,
        /// Optional replacement semantic label.
        #[arg(long)]
        semantic_label: Option<String>,
        /// Optional path to a JSON-serialized `Mesh`.
        #[arg(long)]
        mesh_json: Option<PathBuf>,
        /// Optional path to a JSON-serialized `Tensor` for the visual embedding.
        #[arg(long)]
        visual_embedding_json: Option<PathBuf>,
        /// Optional replacement mass in kilograms.
        #[arg(long)]
        mass: Option<f32>,
        /// Optional replacement friction coefficient.
        #[arg(long)]
        friction: Option<f32>,
        /// Optional replacement restitution coefficient.
        #[arg(long)]
        restitution: Option<f32>,
        /// Optional replacement material name.
        #[arg(long)]
        material: Option<String>,
        /// Optional replacement immovable flag.
        #[arg(long = "static", num_args = 1)]
        is_static: Option<bool>,
        /// Optional replacement graspable flag.
        #[arg(long, num_args = 1)]
        graspable: Option<bool>,
        /// Optional path to write the updated object as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Remove an object from a world scene.
    Remove {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Object ID.
        #[arg(long)]
        object_id: String,
    },
}

/// Plan-management subcommands for persisted worlds.
#[derive(Subcommand)]
pub enum PlanCommands {
    /// List stored plans for a world.
    List {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Optional path to write the plan list as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },
    /// Show a stored plan.
    Show {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Stored plan ID.
        #[arg(long)]
        plan_id: String,
        /// Optional path to write the stored plan as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },
    /// Delete a stored plan.
    Delete {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Stored plan ID.
        #[arg(long)]
        plan_id: String,
    },
}

/// Run the CLI application.
pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt().init();

    match &cli.command {
        Commands::Serve { bind } => {
            return cmd_serve(&cli, bind).await;
        }
        Commands::Providers { capability, health } => {
            return cmd_providers(capability.as_deref(), *health).await;
        }
        Commands::Estimate {
            provider,
            operation,
            steps,
            duration_seconds,
            width,
            height,
        } => {
            return cmd_estimate(
                provider,
                EstimateOptions {
                    operation: *operation,
                    steps: *steps,
                    duration_seconds: *duration_seconds,
                    resolution: (*width, *height),
                },
            );
        }
        Commands::Embed {
            provider,
            fallback_provider,
            text,
            video_json,
            output_json,
        } => {
            return cmd_embed(
                provider,
                EmbedOptions {
                    fallback_provider: fallback_provider.as_deref(),
                    text: text.as_deref(),
                    video_json: video_json.as_deref(),
                    output_json: output_json.as_deref(),
                },
            )
            .await;
        }
        Commands::Health { provider } => return cmd_health(provider).await,
        _ => {}
    }

    let store = open_state_store(&cli).await?;

    match cli.command {
        Commands::Create {
            prompt,
            name,
            provider,
        } => cmd_create(store.as_ref(), &prompt, name.as_deref(), &provider).await,
        Commands::Predict {
            world,
            action,
            steps,
            provider,
            num_samples,
            fallback_provider,
            timeout_ms,
            disable_guardrails,
        } => {
            cmd_predict(
                store.as_ref(),
                &world,
                &action,
                steps,
                &provider,
                num_samples,
                fallback_provider.as_deref(),
                timeout_ms,
                disable_guardrails,
            )
            .await
        }
        Commands::Generate {
            prompt,
            provider,
            fallback_provider,
            negative_prompt,
            duration_seconds,
            width,
            height,
            fps,
            temperature,
            seed,
            output_json,
        } => {
            cmd_generate(
                &prompt,
                &provider,
                GenerateOptions {
                    fallback_provider: fallback_provider.as_deref(),
                    negative_prompt: negative_prompt.as_deref(),
                    duration_seconds,
                    resolution: (width, height),
                    fps,
                    temperature,
                    seed,
                    output_json: output_json.as_deref(),
                },
            )
            .await
        }
        Commands::Transfer {
            provider,
            fallback_provider,
            source_json,
            controls_json,
            output_json,
            width,
            height,
            fps,
            control_strength,
        } => {
            cmd_transfer(
                &provider,
                TransferOptions {
                    fallback_provider: fallback_provider.as_deref(),
                    source_json: &source_json,
                    controls_json: controls_json.as_deref(),
                    output_json: output_json.as_deref(),
                    resolution: (width, height),
                    fps,
                    control_strength,
                },
            )
            .await
        }
        Commands::Embed { .. } => {
            unreachable!("embed command handled before store initialization")
        }
        Commands::Reason {
            world,
            state_json,
            video_json,
            query,
            provider,
            fallback_provider,
            output_json,
        } => {
            cmd_reason(
                store.as_ref(),
                ReasonOptions {
                    world: world.as_deref(),
                    state_json: state_json.as_deref(),
                    video_json: video_json.as_deref(),
                    query: &query,
                    provider: provider.as_deref(),
                    fallback_provider: fallback_provider.as_deref(),
                    output_json: output_json.as_deref(),
                },
            )
            .await
        }
        Commands::List => cmd_list(store.as_ref()).await,
        Commands::Show { world } => cmd_show(store.as_ref(), &world).await,
        Commands::History { world, output_json } => {
            cmd_history(store.as_ref(), &world, output_json.as_deref()).await
        }
        Commands::Restore {
            world,
            history_index,
            output_json,
        } => {
            cmd_restore(
                store.as_ref(),
                &world,
                history_index,
                output_json.as_deref(),
            )
            .await
        }
        Commands::Fork {
            world,
            history_index,
            name,
        } => cmd_fork(store.as_ref(), &world, history_index, name.as_deref())
            .await
            .map(|_| ()),
        Commands::Delete { world } => cmd_delete(store.as_ref(), &world).await,
        Commands::Export {
            world,
            output,
            format,
        } => cmd_export(store.as_ref(), &world, output.as_path(), format).await,
        Commands::Import {
            input,
            format,
            new_id,
            name,
        } => cmd_import(
            store.as_ref(),
            input.as_path(),
            format,
            new_id,
            name.as_deref(),
        )
        .await
        .map(|_| ()),
        Commands::Objects { command } => match command {
            ObjectCommands::Add {
                world,
                name,
                position,
                bbox_min,
                bbox_max,
                velocity,
                semantic_label,
                mesh_json,
                visual_embedding_json,
                mass,
                friction,
                restitution,
                material,
                is_static,
                graspable,
                output_json,
            } => {
                cmd_objects_add(
                    store.as_ref(),
                    &world,
                    &name,
                    ObjectAddOptions {
                        position: &position,
                        bbox_min: &bbox_min,
                        bbox_max: &bbox_max,
                        velocity: velocity.as_deref(),
                        semantic_label: semantic_label.as_deref(),
                        mesh_json: mesh_json.as_deref(),
                        visual_embedding_json: visual_embedding_json.as_deref(),
                        mass,
                        friction,
                        restitution,
                        material: material.as_deref(),
                        is_static,
                        graspable,
                        output_json: output_json.as_deref(),
                    },
                )
                .await
            }
            ObjectCommands::List { world, output_json } => {
                cmd_objects_list(
                    store.as_ref(),
                    &world,
                    ObjectOutputOptions {
                        output_json: output_json.as_deref(),
                    },
                )
                .await
            }
            ObjectCommands::Show {
                world,
                object_id,
                output_json,
            } => {
                cmd_objects_show(
                    store.as_ref(),
                    &world,
                    &object_id,
                    ObjectOutputOptions {
                        output_json: output_json.as_deref(),
                    },
                )
                .await
            }
            ObjectCommands::Update {
                world,
                object_id,
                name,
                position,
                rotation,
                bbox_min,
                bbox_max,
                velocity,
                semantic_label,
                mesh_json,
                visual_embedding_json,
                mass,
                friction,
                restitution,
                material,
                is_static,
                graspable,
                output_json,
            } => {
                cmd_objects_update(
                    store.as_ref(),
                    &world,
                    &object_id,
                    ObjectUpdateOptions {
                        name: name.as_deref(),
                        position: position.as_deref(),
                        rotation: rotation.as_deref(),
                        bbox_min: bbox_min.as_deref(),
                        bbox_max: bbox_max.as_deref(),
                        velocity: velocity.as_deref(),
                        semantic_label: semantic_label.as_deref(),
                        mesh_json: mesh_json.as_deref(),
                        visual_embedding_json: visual_embedding_json.as_deref(),
                        mass,
                        friction,
                        restitution,
                        material: material.as_deref(),
                        is_static,
                        graspable,
                        output_json: output_json.as_deref(),
                    },
                )
                .await
            }
            ObjectCommands::Remove { world, object_id } => {
                cmd_objects_remove(store.as_ref(), &world, &object_id).await
            }
        },
        Commands::Plans { command } => match command {
            PlanCommands::List { world, output_json } => {
                cmd_plans_list(
                    store.as_ref(),
                    &world,
                    PlanOutputOptions {
                        output_json: output_json.as_deref(),
                    },
                )
                .await
            }
            PlanCommands::Show {
                world,
                plan_id,
                output_json,
            } => {
                cmd_plans_show(
                    store.as_ref(),
                    &world,
                    &plan_id,
                    PlanOutputOptions {
                        output_json: output_json.as_deref(),
                    },
                )
                .await
            }
            PlanCommands::Delete { world, plan_id } => {
                cmd_plans_delete(store.as_ref(), &world, &plan_id).await
            }
        },
        Commands::Providers { .. } => {
            unreachable!("providers command handled before store initialization")
        }
        Commands::Estimate { .. } => {
            unreachable!("estimate command handled before store initialization")
        }
        Commands::Eval {
            suite,
            suite_json,
            world,
            world_snapshot,
            providers,
            num_samples,
            list_suites,
            list_metrics,
            output_json,
            output_markdown,
            output_csv,
        } => {
            cmd_eval(
                Some(store.as_ref()),
                EvalOptions {
                    suite_name: suite.as_deref(),
                    suite_json: suite_json.as_deref(),
                    world: world.as_deref(),
                    world_snapshot: world_snapshot.as_deref(),
                    providers: providers.as_deref(),
                    num_samples,
                    list_suites,
                    list_metrics,
                    output_json: output_json.as_deref(),
                    output_markdown: output_markdown.as_deref(),
                    output_csv: output_csv.as_deref(),
                },
            )
            .await
        }
        Commands::Compare {
            world,
            world_snapshot,
            action,
            providers,
            prediction_json,
            steps,
            fallback_provider,
            timeout_ms,
            guardrails_json,
            disable_guardrails,
            output_json,
            output_markdown,
            output_csv,
        } => {
            cmd_compare(
                store.as_ref(),
                world.as_deref(),
                world_snapshot.as_deref(),
                action.as_deref(),
                providers.as_deref(),
                CompareOptions {
                    prediction_json: &prediction_json,
                    steps,
                    fallback_provider: fallback_provider.as_deref(),
                    timeout_ms,
                    guardrails_json: guardrails_json.as_deref(),
                    disable_guardrails,
                    output_json: output_json.as_deref(),
                    output_markdown: output_markdown.as_deref(),
                    output_csv: output_csv.as_deref(),
                },
            )
            .await
        }
        Commands::Plan {
            world,
            goal,
            goal_json,
            max_steps,
            planner,
            num_samples,
            top_k,
            population_size,
            elite_fraction,
            num_iterations,
            learning_rate,
            horizon,
            replanning_interval,
            timeout,
            provider,
            fallback_provider,
            verify_backend,
            guardrails_json,
            disable_guardrails,
            output_json,
        } => {
            cmd_plan(
                store.as_ref(),
                &world,
                goal.as_deref(),
                PlanOptions {
                    max_steps,
                    planner_name: &planner,
                    planner_options: PlannerOptions {
                        num_samples,
                        top_k,
                        population_size,
                        elite_fraction,
                        num_iterations,
                        learning_rate,
                        horizon,
                        replanning_interval,
                    },
                    timeout,
                    provider: &provider,
                    fallback_provider: fallback_provider.as_deref(),
                    verify_backend: verify_backend.map(VerifyBackend::as_core),
                    goal_json: goal_json.as_deref(),
                    guardrails_json: guardrails_json.as_deref(),
                    disable_guardrails,
                    output_json: output_json.as_deref(),
                },
            )
            .await
        }
        Commands::ExecutePlan {
            world,
            plan_json,
            plan_id,
            provider,
            fallback_provider,
            steps,
            timeout_ms,
            guardrails_json,
            return_video,
            disable_guardrails,
            output_json,
        } => {
            cmd_execute_plan(
                store.as_ref(),
                &world,
                ExecutePlanOptions {
                    plan_json: plan_json.as_deref(),
                    plan_id: plan_id.as_deref(),
                    provider: provider.as_deref(),
                    fallback_provider: fallback_provider.as_deref(),
                    steps,
                    timeout_ms,
                    guardrails_json: guardrails_json.as_deref(),
                    return_video,
                    disable_guardrails,
                    output_json: output_json.as_deref(),
                },
            )
            .await
        }
        Commands::Verify {
            world,
            backend,
            proof_type,
            input_state_json,
            output_state_json,
            prediction_json,
            plan_json,
            plan_id,
            goal,
            goal_json,
            max_steps,
            planner,
            num_samples,
            top_k,
            population_size,
            elite_fraction,
            num_iterations,
            learning_rate,
            horizon,
            replanning_interval,
            timeout,
            provider,
            fallback_provider,
            guardrails_json,
            disable_guardrails,
            source_label,
            output_json,
        } => {
            cmd_verify(
                store.as_ref(),
                world.as_deref(),
                VerifyOptions {
                    backend,
                    proof_type: &proof_type,
                    input_state_json: input_state_json.as_deref(),
                    output_state_json: output_state_json.as_deref(),
                    prediction_json: prediction_json.as_deref(),
                    plan_json: plan_json.as_deref(),
                    plan_id: plan_id.as_deref(),
                    goal: goal.as_deref(),
                    goal_json: goal_json.as_deref(),
                    max_steps,
                    planner_name: &planner,
                    planner_options: PlannerOptions {
                        num_samples,
                        top_k,
                        population_size,
                        elite_fraction,
                        num_iterations,
                        learning_rate,
                        horizon,
                        replanning_interval,
                    },
                    timeout,
                    provider: provider.as_deref(),
                    fallback_provider: fallback_provider.as_deref(),
                    guardrails_json: guardrails_json.as_deref(),
                    disable_guardrails,
                    source_label: &source_label,
                    output_json: output_json.as_deref(),
                },
            )
            .await
        }
        Commands::Health { .. } => {
            unreachable!("health command handled before store initialization")
        }
        Commands::VerifyProof {
            proof_json,
            inference_bundle_json,
            guardrail_bundle_json,
            provenance_bundle_json,
            output_json,
        } => {
            cmd_verify_proof(VerifyProofOptions {
                proof_json: proof_json.as_deref(),
                inference_bundle_json: inference_bundle_json.as_deref(),
                guardrail_bundle_json: guardrail_bundle_json.as_deref(),
                provenance_bundle_json: provenance_bundle_json.as_deref(),
                output_json: output_json.as_deref(),
            })
            .await
        }
        Commands::Serve { .. } => {
            unreachable!("serve command handled before store initialization")
        }
    }
}

fn state_store_kind(config: &CliStateStoreConfig<'_>) -> Result<StateStoreKind> {
    match config.state_backend {
        StateBackend::File => Ok(StateStoreKind::FileWithFormat {
            path: config.state_dir.to_path_buf(),
            format: config.state_file_format.as_core(),
        }),
        StateBackend::Sqlite => Ok(StateStoreKind::Sqlite(
            config
                .state_db_path
                .map(Path::to_path_buf)
                .unwrap_or_else(|| config.state_dir.join("worldforge.db")),
        )),
        StateBackend::Redis => config
            .state_redis_url
            .map(|url| StateStoreKind::Redis(url.to_string()))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "--state-redis-url is required when --state-backend redis is selected"
                )
            }),
        StateBackend::S3 => Ok(StateStoreKind::S3 {
            config: resolve_s3_config(
                config.state_s3_bucket,
                config.state_s3_region,
                config.state_s3_access_key_id,
                config.state_s3_secret_access_key,
                config.state_s3_endpoint,
                config.state_s3_session_token,
                config.state_s3_prefix,
            )?,
            format: config.state_file_format.as_core(),
        }),
    }
}

fn resolve_s3_config(
    state_s3_bucket: Option<&str>,
    state_s3_region: Option<&str>,
    state_s3_access_key_id: Option<&str>,
    state_s3_secret_access_key: Option<&str>,
    state_s3_endpoint: Option<&str>,
    state_s3_session_token: Option<&str>,
    state_s3_prefix: Option<&str>,
) -> Result<S3Config> {
    let bucket = state_s3_bucket.ok_or_else(|| {
        anyhow::anyhow!("--state-s3-bucket is required when --state-backend s3 is selected")
    })?;
    let region = state_s3_region.ok_or_else(|| {
        anyhow::anyhow!("--state-s3-region is required when --state-backend s3 is selected")
    })?;
    let access_key_id = state_s3_access_key_id.ok_or_else(|| {
        anyhow::anyhow!("--state-s3-access-key-id is required when --state-backend s3 is selected")
    })?;
    let secret_access_key = state_s3_secret_access_key.ok_or_else(|| {
        anyhow::anyhow!(
            "--state-s3-secret-access-key is required when --state-backend s3 is selected"
        )
    })?;

    Ok(S3Config {
        bucket: bucket.to_string(),
        region: region.to_string(),
        endpoint: state_s3_endpoint
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned),
        access_key_id: access_key_id.to_string(),
        secret_access_key: secret_access_key.to_string(),
        session_token: state_s3_session_token
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned),
        prefix: state_s3_prefix.unwrap_or_default().to_string(),
    })
}

async fn open_state_store(cli: &Cli) -> Result<DynStateStore> {
    state_store_kind(&CliStateStoreConfig::from_cli(cli))?
        .open()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn auto_detect_registry() -> ProviderRegistry {
    worldforge_providers::auto_detect()
}

fn parse_provider_names(input: &str) -> Vec<String> {
    let mut provider_names: Vec<String> = input
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if provider_names.is_empty() {
        provider_names.push("mock".to_string());
    }
    provider_names
}

fn parse_provider_names_allow_empty(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn resolve_eval_provider_names(explicit: Option<&str>, suite: &EvalSuite) -> Vec<String> {
    let explicit = explicit
        .map(parse_provider_names_allow_empty)
        .unwrap_or_default();
    suite.effective_provider_names(&explicit)
}

fn available_provider_names(registry: &ProviderRegistry) -> String {
    let mut names: Vec<String> = registry.list().into_iter().map(str::to_string).collect();
    names.sort();
    names.join(", ")
}

fn available_eval_suite_names() -> String {
    EvalSuite::builtin_names().join(", ")
}

fn available_eval_metric_names() -> String {
    EvalSuite::builtin_metric_names().join(", ")
}

fn available_provider_capabilities() -> &'static str {
    "predict, generate, reason, transfer, embed, planning, action-conditioned, multi-view, depth, segmentation"
}

fn resolve_provider_name<'a>(state: &'a WorldState, provider: Option<&'a str>) -> &'a str {
    provider
        .filter(|name| !name.is_empty())
        .unwrap_or(state.metadata.created_by.as_str())
}

fn scene_edit_world(state: WorldState) -> worldforge_core::world::World {
    let provider = state.metadata.created_by.clone();
    worldforge_core::world::World::new(state, provider, Arc::new(ProviderRegistry::new()))
}

fn parse_position_triplet(values: &[f32], label: &str) -> Result<Position> {
    match values {
        [x, y, z] => Ok(Position {
            x: *x,
            y: *y,
            z: *z,
        }),
        _ => anyhow::bail!("{label} requires exactly 3 values"),
    }
}

fn parse_rotation_quaternion(values: &[f32], label: &str) -> Result<Rotation> {
    match values {
        [w, x, y, z] => Ok(Rotation {
            w: *w,
            x: *x,
            y: *y,
            z: *z,
        }),
        _ => anyhow::bail!("{label} requires exactly 4 values"),
    }
}

fn build_scene_object(name: &str, options: &ObjectAddOptions<'_>) -> Result<SceneObject> {
    let position = parse_position_triplet(options.position, "--position")?;
    let bbox_min = parse_position_triplet(options.bbox_min, "--bbox-min")?;
    let bbox_max = parse_position_triplet(options.bbox_max, "--bbox-max")?;
    let mut object = SceneObject::new(
        name,
        Pose {
            position,
            rotation: Rotation::default(),
        },
        BBox {
            min: bbox_min,
            max: bbox_max,
        },
    );
    if let Some(velocity) = options.velocity {
        let velocity = parse_position_triplet(velocity, "--velocity")?;
        object.velocity = Velocity {
            x: velocity.x,
            y: velocity.y,
            z: velocity.z,
        };
    }
    object.semantic_label = options.semantic_label.map(ToOwned::to_owned);
    object.mesh = load_optional_json(options.mesh_json)?;
    object.visual_embedding = load_optional_json(options.visual_embedding_json)?;
    object.physics = PhysicsProperties {
        mass: options.mass,
        friction: options.friction,
        restitution: options.restitution,
        is_static: options.is_static,
        is_graspable: options.graspable,
        material: options.material.map(ToOwned::to_owned),
    };
    Ok(object)
}

fn build_scene_object_patch(options: &ObjectUpdateOptions<'_>) -> Result<SceneObjectPatch> {
    let position = options
        .position
        .map(|values| parse_position_triplet(values, "--position"))
        .transpose()?;
    let rotation = options
        .rotation
        .map(|values| parse_rotation_quaternion(values, "--rotation"))
        .transpose()?;
    let bbox = match (options.bbox_min, options.bbox_max) {
        (Some(min), Some(max)) => Some(BBox {
            min: parse_position_triplet(min, "--bbox-min")?,
            max: parse_position_triplet(max, "--bbox-max")?,
        }),
        (None, None) => None,
        _ => anyhow::bail!("--bbox-min and --bbox-max must be provided together"),
    };

    Ok(SceneObjectPatch {
        name: options.name.map(ToOwned::to_owned),
        position,
        rotation,
        bbox,
        velocity: options
            .velocity
            .map(|values| {
                parse_position_triplet(values, "--velocity").map(|velocity| Velocity {
                    x: velocity.x,
                    y: velocity.y,
                    z: velocity.z,
                })
            })
            .transpose()?,
        semantic_label: options.semantic_label.map(ToOwned::to_owned),
        mesh: load_optional_json(options.mesh_json)?,
        visual_embedding: load_optional_json(options.visual_embedding_json)?,
        mass: options.mass,
        friction: options.friction,
        restitution: options.restitution,
        material: options.material.map(ToOwned::to_owned),
        is_static: options.is_static,
        is_graspable: options.graspable,
    })
}

fn print_scene_object(object: &SceneObject) {
    println!("  ID: {}", object.id);
    println!("  Name: {}", object.name);
    println!(
        "  Position: {:.2}, {:.2}, {:.2}",
        object.pose.position.x, object.pose.position.y, object.pose.position.z
    );
    println!(
        "  BBox: min=({:.2}, {:.2}, {:.2}) max=({:.2}, {:.2}, {:.2})",
        object.bbox.min.x,
        object.bbox.min.y,
        object.bbox.min.z,
        object.bbox.max.x,
        object.bbox.max.y,
        object.bbox.max.z
    );
    println!(
        "  Velocity: {:.2}, {:.2}, {:.2}",
        object.velocity.x, object.velocity.y, object.velocity.z
    );
    if let Some(label) = &object.semantic_label {
        println!("  Semantic label: {label}");
    }
    match &object.mesh {
        Some(mesh) => {
            println!(
                "  Mesh: vertices={} faces={} normals={} uvs={}",
                mesh.vertices.len(),
                mesh.faces.len(),
                mesh.normals.as_ref().map_or(0, Vec::len),
                mesh.uvs.as_ref().map_or(0, Vec::len)
            );
        }
        None => println!("  Mesh: none"),
    }
    match &object.visual_embedding {
        Some(embedding) => {
            println!(
                "  Visual embedding: shape={:?} dtype={:?} device={:?}",
                embedding.shape, embedding.dtype, embedding.device
            );
        }
        None => println!("  Visual embedding: none"),
    }
    println!(
        "  Physics: static={} graspable={} mass={:?} friction={:?} restitution={:?} material={:?}",
        object.physics.is_static,
        object.physics.is_graspable,
        object.physics.mass,
        object.physics.friction,
        object.physics.restitution,
        object.physics.material
    );
}

fn print_stored_plan(record: &StoredPlanRecord) {
    println!("  ID: {}", record.id);
    println!("  Provider: {}", record.provider);
    println!("  Planner: {}", record.planner);
    println!("  Goal: {}", record.goal_summary);
    println!("  Created: {}", record.created_at);
    println!("  Actions: {}", record.plan.actions.len());
    println!("  Predicted states: {}", record.plan.predicted_states.len());
    println!(
        "  Verification proof: {}",
        if record.plan.verification_proof.is_some() {
            "present"
        } else {
            "none"
        }
    );
}

struct GenerateOptions<'a> {
    fallback_provider: Option<&'a str>,
    negative_prompt: Option<&'a str>,
    duration_seconds: f64,
    resolution: (u32, u32),
    fps: f32,
    temperature: f32,
    seed: Option<u64>,
    output_json: Option<&'a Path>,
}

struct ObjectAddOptions<'a> {
    position: &'a [f32],
    bbox_min: &'a [f32],
    bbox_max: &'a [f32],
    velocity: Option<&'a [f32]>,
    semantic_label: Option<&'a str>,
    mesh_json: Option<&'a Path>,
    visual_embedding_json: Option<&'a Path>,
    mass: Option<f32>,
    friction: Option<f32>,
    restitution: Option<f32>,
    material: Option<&'a str>,
    is_static: bool,
    graspable: bool,
    output_json: Option<&'a Path>,
}

struct ObjectUpdateOptions<'a> {
    name: Option<&'a str>,
    position: Option<&'a [f32]>,
    rotation: Option<&'a [f32]>,
    bbox_min: Option<&'a [f32]>,
    bbox_max: Option<&'a [f32]>,
    velocity: Option<&'a [f32]>,
    semantic_label: Option<&'a str>,
    mesh_json: Option<&'a Path>,
    visual_embedding_json: Option<&'a Path>,
    mass: Option<f32>,
    friction: Option<f32>,
    restitution: Option<f32>,
    material: Option<&'a str>,
    is_static: Option<bool>,
    graspable: Option<bool>,
    output_json: Option<&'a Path>,
}

struct ObjectOutputOptions<'a> {
    output_json: Option<&'a Path>,
}

struct PlanOutputOptions<'a> {
    output_json: Option<&'a Path>,
}

struct EstimateOptions {
    operation: EstimateOperationKind,
    steps: u32,
    duration_seconds: f64,
    resolution: (u32, u32),
}

struct TransferOptions<'a> {
    fallback_provider: Option<&'a str>,
    source_json: &'a Path,
    controls_json: Option<&'a Path>,
    output_json: Option<&'a Path>,
    resolution: (u32, u32),
    fps: f32,
    control_strength: f32,
}

struct EmbedOptions<'a> {
    fallback_provider: Option<&'a str>,
    text: Option<&'a str>,
    video_json: Option<&'a Path>,
    output_json: Option<&'a Path>,
}

struct ReasonOptions<'a> {
    world: Option<&'a str>,
    state_json: Option<&'a Path>,
    video_json: Option<&'a Path>,
    query: &'a str,
    provider: Option<&'a str>,
    fallback_provider: Option<&'a str>,
    output_json: Option<&'a Path>,
}

struct PlanOptions<'a> {
    max_steps: u32,
    planner_name: &'a str,
    planner_options: PlannerOptions,
    timeout: f64,
    provider: &'a str,
    fallback_provider: Option<&'a str>,
    verify_backend: Option<VerificationBackend>,
    goal_json: Option<&'a Path>,
    guardrails_json: Option<&'a Path>,
    disable_guardrails: bool,
    output_json: Option<&'a Path>,
}

struct ExecutePlanOptions<'a> {
    plan_json: Option<&'a Path>,
    plan_id: Option<&'a str>,
    provider: Option<&'a str>,
    fallback_provider: Option<&'a str>,
    steps: u32,
    timeout_ms: Option<u64>,
    guardrails_json: Option<&'a Path>,
    return_video: bool,
    disable_guardrails: bool,
    output_json: Option<&'a Path>,
}

struct CompareOptions<'a> {
    prediction_json: &'a [PathBuf],
    steps: u32,
    fallback_provider: Option<&'a str>,
    timeout_ms: Option<u64>,
    guardrails_json: Option<&'a Path>,
    disable_guardrails: bool,
    output_json: Option<&'a Path>,
    output_markdown: Option<&'a Path>,
    output_csv: Option<&'a Path>,
}

struct EvalOptions<'a> {
    suite_name: Option<&'a str>,
    suite_json: Option<&'a Path>,
    world: Option<&'a str>,
    world_snapshot: Option<&'a Path>,
    providers: Option<&'a str>,
    num_samples: Option<u32>,
    list_suites: bool,
    list_metrics: bool,
    output_json: Option<&'a Path>,
    output_markdown: Option<&'a Path>,
    output_csv: Option<&'a Path>,
}

struct VerifyOptions<'a> {
    backend: VerifyBackend,
    proof_type: &'a str,
    input_state_json: Option<&'a Path>,
    output_state_json: Option<&'a Path>,
    prediction_json: Option<&'a Path>,
    plan_json: Option<&'a Path>,
    plan_id: Option<&'a str>,
    goal: Option<&'a str>,
    goal_json: Option<&'a Path>,
    max_steps: u32,
    planner_name: &'a str,
    planner_options: PlannerOptions,
    timeout: f64,
    provider: Option<&'a str>,
    fallback_provider: Option<&'a str>,
    guardrails_json: Option<&'a Path>,
    disable_guardrails: bool,
    source_label: &'a str,
    output_json: Option<&'a Path>,
}

struct VerifyProofOptions<'a> {
    proof_json: Option<&'a Path>,
    inference_bundle_json: Option<&'a Path>,
    guardrail_bundle_json: Option<&'a Path>,
    provenance_bundle_json: Option<&'a Path>,
    output_json: Option<&'a Path>,
}

#[derive(Serialize)]
struct ProofVerificationReport {
    proof: ZkProof,
    verification: VerificationResult,
}

#[derive(Serialize)]
struct ReasoningResponse {
    provider: String,
    reasoning: ReasoningOutput,
}

fn require_provider<'a>(
    registry: &'a ProviderRegistry,
    provider: &str,
) -> Result<&'a dyn WorldModelProvider> {
    registry.get(provider).map_err(|e| {
        anyhow::anyhow!(
            "{e}. Available providers: {}",
            available_provider_names(registry)
        )
    })
}

fn verifier_for_backend(backend: VerificationBackend) -> Box<dyn ZkVerifier> {
    verify_backend_resolver(backend)
}

fn verifier_for_proof(proof: &ZkProof) -> Box<dyn ZkVerifier> {
    verifier_for_backend(proof.backend)
}

fn core_proof_from_verify_proof(proof: &ZkProof) -> Result<worldforge_core::proof::ZkProof> {
    let bytes = serde_json::to_vec(proof).context("failed to serialize verification proof")?;
    serde_json::from_slice(&bytes).context("failed to convert verification proof into core shape")
}

fn attach_plan_verification(plan: &mut Plan, backend: Option<VerificationBackend>) -> Result<bool> {
    let Some(backend) = backend else {
        return Ok(false);
    };

    let verifier = verifier_for_backend(backend);
    let bundle = prove_guardrail_plan(verifier.as_ref(), plan)
        .map_err(|e| anyhow::anyhow!("failed to attach plan verification proof: {e}"))?;
    plan.verification_proof = Some(core_proof_from_verify_proof(&bundle.proof)?);
    Ok(true)
}

fn read_json_file<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read JSON from {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse JSON from {}", path.display()))
}

fn load_optional_json<T: DeserializeOwned>(path: Option<&Path>) -> Result<Option<T>> {
    path.map(read_json_file).transpose()
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let contents = serde_json::to_string_pretty(value).context("failed to serialize JSON")?;
    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

fn write_text_file(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

fn parse_snapshot_format(value: &str) -> std::result::Result<CoreStateFileFormat, String> {
    value.parse()
}

fn infer_snapshot_format(path: &Path) -> Result<CoreStateFileFormat> {
    infer_state_file_format(path)
        .map_err(anyhow::Error::new)
        .with_context(|| {
            format!(
                "unable to infer snapshot format from {}. Use --format json|msgpack",
                path.display()
            )
        })
}

fn resolve_snapshot_format(
    path: &Path,
    format: Option<CoreStateFileFormat>,
) -> Result<CoreStateFileFormat> {
    match format {
        Some(format) => Ok(format),
        None => infer_snapshot_format(path),
    }
}

fn read_world_state_snapshot(
    path: &Path,
    format: Option<CoreStateFileFormat>,
) -> Result<WorldState> {
    read_world_state_snapshot_with_metadata(path, format).map(|(state, _)| state)
}

fn read_world_state_snapshot_with_metadata(
    path: &Path,
    format: Option<CoreStateFileFormat>,
) -> Result<(WorldState, worldforge_core::state::StateSnapshotMetadata)> {
    let format = resolve_snapshot_format(path, format)?;
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read world snapshot from {}", path.display()))?;
    let metadata = inspect_world_state_snapshot(format, &bytes).map_err(anyhow::Error::new)?;
    let state = deserialize_world_state(format, &bytes).map_err(anyhow::Error::new)?;
    Ok((state, metadata))
}

fn write_world_state_snapshot(
    path: &Path,
    state: &WorldState,
    format: CoreStateFileFormat,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let bytes = serialize_world_state(format, state).map_err(anyhow::Error::new)?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn read_guardrails(path: Option<&Path>) -> Result<Vec<GuardrailConfig>> {
    match path {
        Some(path) => read_json_file(path),
        None => Ok(Vec::new()),
    }
}

fn load_plan_goal(goal: Option<&str>, goal_json: Option<&Path>) -> Result<PlanGoal> {
    match (goal, goal_json) {
        (Some(description), None) => Ok(PlanGoal::Description(description.to_string())),
        (None, Some(path)) => {
            let input: PlanGoalInput = read_json_file(path)?;
            Ok(input.into())
        }
        (Some(_), Some(_)) => Err(anyhow::anyhow!(
            "--goal and --goal-json are mutually exclusive"
        )),
        (None, None) => Err(anyhow::anyhow!("either --goal or --goal-json is required")),
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

fn load_eval_suite(suite_name: Option<&str>, suite_json: Option<&Path>) -> Result<EvalSuite> {
    match suite_json {
        Some(path) => EvalSuite::from_json_path(path).map_err(|e| anyhow::anyhow!("{e}")),
        None => EvalSuite::from_builtin(suite_name.unwrap_or("physics"))
            .map_err(|e| anyhow::anyhow!("{e}")),
    }
}

fn planner_from_name(
    planner_name: &str,
    max_steps: u32,
    planner_options: PlannerOptions,
) -> Result<PlannerType> {
    PlannerType::from_name(planner_name, max_steps, planner_options)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn build_operation(kind: EstimateOperationKind, options: &EstimateOptions) -> Operation {
    match kind {
        EstimateOperationKind::Predict => Operation::Predict {
            steps: options.steps.max(1),
            resolution: options.resolution,
        },
        EstimateOperationKind::Generate => Operation::Generate {
            duration_seconds: options.duration_seconds.max(0.1),
            resolution: options.resolution,
        },
        EstimateOperationKind::Reason => Operation::Reason,
        EstimateOperationKind::Transfer => Operation::Transfer {
            duration_seconds: options.duration_seconds.max(0.1),
        },
    }
}

fn summarize_capabilities(descriptor: &ProviderDescriptor) -> String {
    let mut labels = Vec::new();
    if descriptor.capabilities.predict {
        labels.push("predict");
    }
    if descriptor.capabilities.generate {
        labels.push("generate");
    }
    if descriptor.capabilities.reason {
        labels.push("reason");
    }
    if descriptor.capabilities.transfer {
        labels.push("transfer");
    }
    if descriptor.capabilities.embed {
        labels.push("embed");
    }
    if descriptor.capabilities.supports_planning {
        labels.push("planning");
    }
    if descriptor.capabilities.supports_depth {
        labels.push("depth");
    }
    if descriptor.capabilities.supports_segmentation {
        labels.push("segmentation");
    }
    labels.join(", ")
}

fn print_provider_descriptor(descriptor: &ProviderDescriptor) {
    let caps = &descriptor.capabilities;
    println!(
        "{} [{}]",
        descriptor.name,
        summarize_capabilities(descriptor)
    );
    println!(
        "  resolution: {}x{} | fps: {:.1}-{:.1} | max length: {:.1}s",
        caps.max_resolution.0,
        caps.max_resolution.1,
        caps.fps_range.0,
        caps.fps_range.1,
        caps.max_video_length_seconds
    );
    println!(
        "  action conditioned: {} | multi-view: {} | latency p50/p95/p99: {}/{}/{} ms",
        caps.action_conditioned,
        caps.multi_view,
        caps.latency_profile.p50_ms,
        caps.latency_profile.p95_ms,
        caps.latency_profile.p99_ms
    );
}

fn print_provider_health_report(report: &ProviderHealthReport) {
    match (&report.status, &report.error) {
        (Some(status), None) => {
            let icon = if status.healthy { "OK" } else { "UNHEALTHY" };
            println!(
                "  [{icon}] {}: {} ({}ms)",
                report.name, status.message, status.latency_ms
            );
        }
        (_, Some(error)) => {
            println!("  [ERROR] {}: {error}", report.name);
        }
        (None, None) => {
            println!("  [ERROR] {}: health check returned no status", report.name);
        }
    }
}

fn print_verification_bundle<T: Serialize>(
    label: &str,
    bundle: &VerificationBundle<T>,
) -> Result<()> {
    println!("ZK Proof generated:");
    println!("  Type: {label}");
    println!("  Backend: {:?}", bundle.proof.backend);
    println!("  Proof size: {} bytes", bundle.proof.proof_data.len());
    println!("  Generation time: {}ms", bundle.proof.generation_time_ms);
    println!();
    println!("Verification:");
    println!("  Valid: {}", bundle.verification.valid);
    println!("  Details: {}", bundle.verification.details);
    println!(
        "  Verification time: {}ms",
        bundle.verification.verification_time_ms
    );
    println!();
    println!("Artifact:");
    println!(
        "{}",
        serde_json::to_string_pretty(&bundle.artifact).context("failed to serialize artifact")?
    );
    Ok(())
}

fn print_proof_verification(report: &ProofVerificationReport) {
    println!("Proof verified:");
    println!("  Backend: {:?}", report.proof.backend);
    println!("  Proof size: {} bytes", report.proof.proof_data.len());
    println!("  Valid: {}", report.verification.valid);
    println!("  Details: {}", report.verification.details);
    println!(
        "  Verification time: {}ms",
        report.verification.verification_time_ms
    );
}

fn print_bundle_verification<T: Serialize>(
    label: &str,
    report: &BundleVerificationReport<T>,
) -> Result<()> {
    println!("Bundle re-verified:");
    println!("  Type: {label}");
    println!("  Backend: {:?}", report.proof.backend);
    println!("  Proof size: {} bytes", report.proof.proof_data.len());
    println!(
        "  Matches recorded verdict: {}",
        report.verification_matches_recorded
    );
    println!("  Current valid: {}", report.current_verification.valid);
    println!("  Current details: {}", report.current_verification.details);
    println!();
    println!("Artifact:");
    println!(
        "{}",
        serde_json::to_string_pretty(&report.artifact).context("failed to serialize artifact")?
    );
    Ok(())
}

fn print_plan_execution(report: &PlanExecution) {
    println!("Plan executed:");
    println!("  Steps: {}", report.predictions.len());
    println!("  Execution time: {}ms", report.execution_time_ms);
    println!("  Total USD: {:.4}", report.total_cost.usd);
    println!("  Total credits: {:.2}", report.total_cost.credits);
    println!(
        "  Estimated latency: {}ms",
        report.total_cost.estimated_latency_ms
    );
    println!("  Final step: {}", report.final_state.time.step);
}

async fn cmd_create(
    store: &(impl StateStore + ?Sized),
    prompt: &str,
    name: Option<&str>,
    provider: &str,
) -> Result<()> {
    let registry = auto_detect_registry();
    require_provider(&registry, provider).context("failed to create world")?;
    let state = WorldState::from_prompt(prompt, provider, name)?;
    store
        .save(&state)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("Created world: {}", state.id);
    println!("  Name: {}", state.metadata.name);
    println!("  Description: {}", state.metadata.description);
    println!("  Provider: {provider}");
    println!("  Seeded objects: {}", state.scene.objects.len());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn cmd_predict(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    action_str: &str,
    steps: u32,
    provider: &str,
    num_samples: Option<u32>,
    fallback_provider: Option<&str>,
    timeout_ms: Option<u64>,
    disable_guardrails: bool,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    let registry = Arc::new(auto_detect_registry());
    if fallback_provider.is_none() {
        require_provider(&registry, provider)?;
    }
    if let Some(fallback_provider) = fallback_provider {
        require_provider(&registry, fallback_provider)
            .context("invalid fallback provider for predict command")?;
    }
    let action = parse_action(action_str, &state)?;
    let mut world = worldforge_core::world::World::new(state, provider, registry);
    let mut config = build_predict_config(steps, num_samples, fallback_provider, timeout_ms);
    if disable_guardrails {
        config = config.disable_guardrails();
    }

    let prediction = world
        .predict(&action, &config)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Save updated state
    store
        .save(world.current_state())
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Prediction completed:");
    println!("  Provider: {}", prediction.provider);
    if let Some(fallback_provider) = fallback_provider {
        println!("  Fallback provider: {fallback_provider}");
    }
    println!("  Confidence: {:.2}", prediction.confidence);
    println!("  Physics score: {:.2}", prediction.physics_scores.overall);
    println!("  Latency: {}ms", prediction.latency_ms);
    if let Some(summary) = predict_sampling_summary(prediction.sampling.as_ref()) {
        println!("  {summary}");
    }
    println!("  New time step: {}", world.current_state().time.step);

    Ok(())
}

fn build_predict_config(
    steps: u32,
    num_samples: Option<u32>,
    fallback_provider: Option<&str>,
    timeout_ms: Option<u64>,
) -> PredictionConfig {
    PredictionConfig {
        steps,
        num_samples: num_samples.unwrap_or(1).max(1),
        fallback_provider: fallback_provider.map(ToOwned::to_owned),
        max_latency_ms: timeout_ms,
        ..PredictionConfig::default()
    }
}

fn predict_sampling_summary(sampling: Option<&PredictionSamplingMetadata>) -> Option<String> {
    sampling.map(|sampling| {
        format!(
            "Sampling: {} requested, {} completed, best sample #{}",
            sampling.requested_samples,
            sampling.completed_samples,
            sampling.selected_sample_index + 1
        )
    })
}

fn eval_sampling_summary(sampling: Option<&worldforge_eval::SamplingSummary>) -> Option<String> {
    sampling.map(|sampling| {
        format!(
            "{}/{} steps, {} requests, {:.1}% complete",
            sampling.sampled_steps,
            sampling.prediction_steps,
            sampling.requested_samples,
            sampling.completion_rate * 100.0
        )
    })
}

async fn cmd_generate(
    prompt: &str,
    provider_name: &str,
    options: GenerateOptions<'_>,
) -> Result<()> {
    let wf = worldforge_core::WorldForge::from_registry(auto_detect_registry());
    let prompt = GenerationPrompt {
        text: prompt.to_string(),
        reference_image: None,
        negative_prompt: options.negative_prompt.map(ToOwned::to_owned),
    };
    let config = GenerationConfig {
        duration_seconds: options.duration_seconds,
        resolution: options.resolution,
        fps: options.fps,
        temperature: options.temperature,
        seed: options.seed,
    };

    let (resolved_provider, clip) = wf
        .generate_with_fallback(provider_name, &prompt, &config, options.fallback_provider)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Generation completed:");
    println!("  Provider: {resolved_provider}");
    println!("  Duration: {:.2}s", clip.duration);
    println!("  Resolution: {}x{}", clip.resolution.0, clip.resolution.1);
    println!("  FPS: {:.1}", clip.fps);
    println!("  Frames: {}", clip.frames.len());
    if let Some(path) = options.output_json {
        write_json_file(path, &clip)?;
        println!("  Output JSON: {}", path.display());
    }

    Ok(())
}

async fn cmd_transfer(provider_name: &str, options: TransferOptions<'_>) -> Result<()> {
    let wf = worldforge_core::WorldForge::from_registry(auto_detect_registry());
    let source: VideoClip = read_json_file(options.source_json)?;
    let controls = match options.controls_json {
        Some(path) => read_json_file(path)?,
        None => SpatialControls::default(),
    };
    let config = TransferConfig {
        resolution: options.resolution,
        fps: options.fps,
        control_strength: options.control_strength,
    };

    let (resolved_provider, clip) = wf
        .transfer_with_fallback(
            provider_name,
            &source,
            &controls,
            &config,
            options.fallback_provider,
        )
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Transfer completed:");
    println!("  Provider: {resolved_provider}");
    println!("  Duration: {:.2}s", clip.duration);
    println!("  Resolution: {}x{}", clip.resolution.0, clip.resolution.1);
    println!("  FPS: {:.1}", clip.fps);
    println!("  Frames: {}", clip.frames.len());
    if let Some(path) = options.output_json {
        write_json_file(path, &clip)?;
        println!("  Output JSON: {}", path.display());
    }

    Ok(())
}

async fn cmd_embed(provider_name: &str, options: EmbedOptions<'_>) -> Result<()> {
    let wf = worldforge_core::WorldForge::from_registry(auto_detect_registry());
    let video = match options.video_json {
        Some(path) => Some(read_json_file(path)?),
        None => None,
    };
    let input = EmbeddingInput::new(options.text.map(ToOwned::to_owned), video)?;
    let (resolved_provider, output) = wf
        .embed_with_fallback(provider_name, &input, options.fallback_provider)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Embedding completed:");
    println!("  Provider: {resolved_provider}");
    println!("  Model: {}", output.model);
    println!("  Embedding shape: {:?}", output.embedding.shape);
    if let Some(path) = options.output_json {
        write_json_file(path, &output)?;
        println!("  Output JSON: {}", path.display());
    }

    Ok(())
}

async fn cmd_reason(store: &(impl StateStore + ?Sized), options: ReasonOptions<'_>) -> Result<()> {
    let registry = Arc::new(auto_detect_registry());
    let world_state = load_reasoning_state(store, options.world, options.state_json).await?;
    let video = match options.video_json {
        Some(path) => Some(read_json_file(path)?),
        None => None,
    };

    if world_state.is_none() && video.is_none() {
        return Err(anyhow::anyhow!(
            "reason requires --world, --state-json, or --video-json"
        ));
    }

    let provider_name = match options.provider.filter(|value| !value.is_empty()) {
        Some(provider) => provider.to_string(),
        None if options.world.is_some() => world_state
            .as_ref()
            .map(WorldState::current_state_provider)
            .ok_or_else(|| anyhow::anyhow!("failed to load world state for reasoning"))?,
        None => {
            return Err(anyhow::anyhow!(
                "reason requires --provider when --world is not provided"
            ));
        }
    };

    let use_world_flow =
        options.world.is_some() && options.state_json.is_none() && options.video_json.is_none();

    let (resolved_provider, output) = if use_world_flow {
        let state = world_state.expect("world flow requires loaded state");
        let world =
            worldforge_core::world::World::new(state, &provider_name, Arc::clone(&registry));
        world
            .reason_with_provider_and_fallback(
                options.query,
                &provider_name,
                options.fallback_provider,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        let input = ReasoningInput {
            state: world_state,
            video,
        };
        reason_with_provider_and_fallback(
            registry.as_ref(),
            &provider_name,
            &input,
            options.query,
            options.fallback_provider,
        )
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?
    };

    println!("Reasoning completed:");
    println!("  Provider: {resolved_provider}");
    println!("  Answer: {}", output.answer);
    println!("  Confidence: {:.2}", output.confidence);
    if output.evidence.is_empty() {
        println!("  Evidence: none");
    } else {
        println!("  Evidence:");
        for evidence in &output.evidence {
            println!("    - {evidence}");
        }
    }

    if let Some(path) = options.output_json {
        write_json_file(
            path,
            &ReasoningResponse {
                provider: resolved_provider,
                reasoning: output,
            },
        )?;
        println!("  Output JSON: {}", path.display());
    }

    Ok(())
}

async fn reason_with_provider_and_fallback(
    registry: &ProviderRegistry,
    provider_name: &str,
    input: &ReasoningInput,
    query: &str,
    fallback_provider: Option<&str>,
) -> Result<(String, ReasoningOutput)> {
    match registry.get(provider_name) {
        Ok(provider) => match provider.reason(input, query).await {
            Ok(output) => Ok((provider_name.to_string(), output)),
            Err(primary_error) => {
                let Some(fallback_provider) =
                    fallback_provider.filter(|fallback| *fallback != provider_name)
                else {
                    return Err(primary_error.into());
                };

                match registry.get(fallback_provider) {
                    Ok(provider) => match provider.reason(input, query).await {
                        Ok(output) => Ok((fallback_provider.to_string(), output)),
                        Err(fallback_error) => Err(WorldForgeError::ProviderUnavailable {
                            provider: provider_name.to_string(),
                            reason: format!(
                                "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                            ),
                        }
                        .into()),
                    },
                    Err(fallback_error) => Err(WorldForgeError::ProviderUnavailable {
                        provider: provider_name.to_string(),
                        reason: format!(
                            "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                        ),
                    }
                    .into()),
                }
            }
        },
        Err(primary_error) => {
            let Some(fallback_provider) =
                fallback_provider.filter(|fallback| *fallback != provider_name)
            else {
                return Err(primary_error.into());
            };

            match registry.get(fallback_provider) {
                Ok(provider) => match provider.reason(input, query).await {
                    Ok(output) => Ok((fallback_provider.to_string(), output)),
                    Err(fallback_error) => Err(WorldForgeError::ProviderUnavailable {
                        provider: provider_name.to_string(),
                        reason: format!(
                            "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                        ),
                    }
                    .into()),
                },
                Err(fallback_error) => Err(WorldForgeError::ProviderUnavailable {
                    provider: provider_name.to_string(),
                    reason: format!(
                        "primary provider error: {primary_error}; fallback provider '{fallback_provider}' error: {fallback_error}"
                    ),
                }
                .into()),
            }
        }
    }
}

async fn load_reasoning_state(
    store: &(impl StateStore + ?Sized),
    world: Option<&str>,
    state_json: Option<&Path>,
) -> Result<Option<WorldState>> {
    match (world, state_json) {
        (Some(_), Some(_)) => Err(anyhow::anyhow!(
            "--world and --state-json are mutually exclusive"
        )),
        (Some(world_id), None) => {
            let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
            let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(Some(state))
        }
        (None, Some(path)) => read_json_file(path).map(Some),
        (None, None) => Ok(None),
    }
}

async fn cmd_list(store: &(impl StateStore + ?Sized)) -> Result<()> {
    let ids = store.list().await.map_err(|e| anyhow::anyhow!("{e}"))?;
    if ids.is_empty() {
        println!("No worlds found.");
    } else {
        println!("Saved worlds:");
        for id in &ids {
            match store.load(id).await {
                Ok(state) => {
                    println!(
                        "  {} — {} (step {})",
                        id, state.metadata.name, state.time.step
                    );
                }
                Err(_) => {
                    println!("  {} — (unreadable)", id);
                }
            }
        }
    }
    Ok(())
}

async fn cmd_show(store: &(impl StateStore + ?Sized), world_id: &str) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("World: {}", state.id);
    println!("  Name: {}", state.metadata.name);
    println!("  Provider: {}", state.metadata.created_by);
    println!("  Created: {}", state.metadata.created_at);
    println!("  Time step: {}", state.time.step);
    println!("  Time (seconds): {:.2}", state.time.seconds);
    println!("  Objects: {}", state.scene.objects.len());
    for obj in state.scene.list_objects() {
        println!(
            "    - {} [{}] (pos: {:.1}, {:.1}, {:.1})",
            obj.name, obj.id, obj.pose.position.x, obj.pose.position.y, obj.pose.position.z
        );
    }
    println!("  History entries: {}", state.history.len());
    Ok(())
}

async fn cmd_history(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    output_json: Option<&Path>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let entries: Vec<_> = state.history.states.iter().cloned().collect();

    if let Some(path) = output_json {
        write_json_file(path, &entries)?;
        println!("Wrote world history JSON to {}", path.display());
        return Ok(());
    }

    println!("History for world: {}", state.id);
    if entries.is_empty() {
        println!("  No recorded history entries.");
        return Ok(());
    }

    for (index, entry) in entries.iter().enumerate() {
        let action = entry
            .action
            .as_ref()
            .map(|action| format!("{action:?}"))
            .unwrap_or_else(|| "initial checkpoint".to_string());
        let prediction = entry
            .prediction
            .as_ref()
            .map(|prediction| {
                let model = prediction
                    .model
                    .as_deref()
                    .map(|model| format!(", model {model}"))
                    .unwrap_or_default();
                format!(
                    "confidence {:.2}, physics {:.2}, latency {}ms{}",
                    prediction.confidence, prediction.physics_score, prediction.latency_ms, model
                )
            })
            .unwrap_or_else(|| "no prediction summary".to_string());

        println!(
            "  [{}] step {} provider={} action={}",
            index, entry.time.step, entry.provider, action
        );
        println!("      prediction={prediction}");
    }

    Ok(())
}

async fn cmd_restore(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    history_index: usize,
    output_json: Option<&Path>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let mut state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    state
        .restore_history(history_index)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    store
        .save(&state)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if let Some(path) = output_json {
        write_json_file(path, &state)?;
        println!("Wrote restored world JSON to {}", path.display());
    }

    println!(
        "Restored world {} to history checkpoint {} (step {}, {} entries retained)",
        state.id,
        history_index,
        state.time.step,
        state.history.len()
    );
    Ok(())
}

async fn cmd_fork(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    history_index: Option<usize>,
    name: Option<&str>,
) -> Result<WorldState> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let source = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let forked = fork_world_state(&source, history_index, name)?;
    store
        .save(&forked)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Forked world:");
    println!("  Source: {}", source.id);
    println!("  Fork: {}", forked.id);
    println!("  Name: {}", forked.metadata.name);
    println!("  Provider: {}", forked.metadata.created_by);
    println!("  Time step: {}", forked.time.step);
    println!("  History entries: {}", forked.history.len());

    Ok(forked)
}

fn default_fork_name(source_name: &str, name_override: Option<&str>) -> String {
    if let Some(name) = name_override.map(str::trim).filter(|name| !name.is_empty()) {
        return name.to_string();
    }

    let source_name = source_name.trim();
    if source_name.is_empty() {
        "Forked World".to_string()
    } else {
        format!("{source_name} Fork")
    }
}

fn fork_world_state(
    source: &WorldState,
    history_index: Option<usize>,
    name_override: Option<&str>,
) -> Result<WorldState> {
    let source = match history_index {
        Some(index) => source
            .history_state(index)
            .map_err(|e| anyhow::anyhow!("{e}"))?,
        None => source.clone(),
    };
    let provider = source.current_state_provider();
    let mut forked = WorldState {
        id: uuid::Uuid::new_v4(),
        time: source.time,
        scene: source.scene,
        history: StateHistory {
            states: std::collections::VecDeque::new(),
            max_entries: source.history.max_entries,
            compression: source.history.compression,
        },
        metadata: source.metadata,
        stored_plans: source.stored_plans,
    };
    forked.metadata.name = default_fork_name(&forked.metadata.name, name_override);
    forked.metadata.created_by = provider.clone();
    forked.metadata.created_at = chrono::Utc::now();
    forked
        .record_current_state(None, None, provider)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(forked)
}

async fn cmd_delete(store: &(impl StateStore + ?Sized), world_id: &str) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    store
        .delete(&id)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("Deleted world: {id}");
    Ok(())
}

async fn cmd_export(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    output: &Path,
    format: Option<CoreStateFileFormat>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let format = resolve_snapshot_format(output, format)?;

    write_world_state_snapshot(output, &state, format)?;

    println!("Exported world snapshot:");
    println!("  World: {id}");
    println!("  Schema version: {}", WORLD_STATE_SNAPSHOT_SCHEMA_VERSION);
    println!("  Format: {}", format.as_str());
    println!("  Output: {}", output.display());
    Ok(())
}

async fn cmd_import(
    store: &(impl StateStore + ?Sized),
    input: &Path,
    format: Option<CoreStateFileFormat>,
    new_id: bool,
    name: Option<&str>,
) -> Result<WorldState> {
    let resolved_format = resolve_snapshot_format(input, format)?;
    let (mut state, snapshot_metadata) =
        read_world_state_snapshot_with_metadata(input, Some(resolved_format))?;
    if new_id {
        state.id = uuid::Uuid::new_v4();
    }
    if let Some(name) = name {
        state.metadata.name = name.to_string();
    }

    store
        .save(&state)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Imported world snapshot:");
    println!("  World: {}", state.id);
    println!("  Name: {}", state.metadata.name);
    println!("  Schema version: {}", snapshot_metadata.schema_version);
    println!("  Legacy payload: {}", snapshot_metadata.legacy_payload);
    println!("  Format: {}", resolved_format.as_str());
    println!("  Source: {}", input.display());

    Ok(state)
}

async fn cmd_objects_add(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    name: &str,
    options: ObjectAddOptions<'_>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut world = scene_edit_world(state);
    let object = build_scene_object(name, &options)?;

    world
        .add_object(object.clone())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    store
        .save(world.current_state())
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Added object to world: {world_id}");
    print_scene_object(&object);
    if let Some(path) = options.output_json {
        write_json_file(path, &object)?;
        println!("  Output JSON: {}", path.display());
    }
    Ok(())
}

async fn cmd_objects_update(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    object_id: &str,
    options: ObjectUpdateOptions<'_>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let object_id: uuid::Uuid = object_id.parse().context("invalid object ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut world = scene_edit_world(state);
    let patch = build_scene_object_patch(&options)?;
    let object = world
        .update_object(&object_id, patch)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    store
        .save(world.current_state())
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Updated object in world: {world_id}");
    print_scene_object(&object);
    if let Some(path) = options.output_json {
        write_json_file(path, &object)?;
        println!("  Output JSON: {}", path.display());
    }
    Ok(())
}

async fn cmd_objects_list(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    options: ObjectOutputOptions<'_>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let world = scene_edit_world(state);
    let objects = world.list_objects();

    if objects.is_empty() {
        println!("No objects found.");
    } else {
        println!("Objects in world {world_id}:");
        for object in &objects {
            println!(
                "  {} — {} ({:.2}, {:.2}, {:.2})",
                object.id,
                object.name,
                object.pose.position.x,
                object.pose.position.y,
                object.pose.position.z
            );
        }
    }

    if let Some(path) = options.output_json {
        write_json_file(path, &objects)?;
        println!("Saved object list: {}", path.display());
    }

    Ok(())
}

async fn cmd_objects_show(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    object_id: &str,
    options: ObjectOutputOptions<'_>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let object_id: uuid::Uuid = object_id.parse().context("invalid object ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let world = scene_edit_world(state);
    let object = world
        .get_object(&object_id)
        .cloned()
        .context("object not found")?;

    println!("Object in world {world_id}:");
    print_scene_object(&object);
    if let Some(path) = options.output_json {
        write_json_file(path, &object)?;
        println!("Saved object JSON: {}", path.display());
    }
    Ok(())
}

async fn cmd_objects_remove(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    object_id: &str,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let object_id: uuid::Uuid = object_id.parse().context("invalid object ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut world = scene_edit_world(state);
    let object = world
        .remove_object(&object_id)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    store
        .save(world.current_state())
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Removed object from world: {world_id}");
    print_scene_object(&object);
    Ok(())
}

async fn cmd_plans_list(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    options: PlanOutputOptions<'_>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let plans: Vec<StoredPlanRecord> = state.stored_plans.values().cloned().collect();

    if plans.is_empty() {
        println!("No stored plans found.");
    } else {
        println!("Stored plans for world {}:", state.id);
        for record in &plans {
            println!(
                "  {} — {} (planner {}, provider {}, actions {})",
                record.id,
                record.goal_summary,
                record.planner,
                record.provider,
                record.plan.actions.len()
            );
        }
    }

    if let Some(path) = options.output_json {
        write_json_file(path, &plans)?;
        println!("Saved stored plan list: {}", path.display());
    }

    Ok(())
}

async fn cmd_plans_show(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    plan_id: &str,
    options: PlanOutputOptions<'_>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let plan_id: uuid::Uuid = plan_id.parse().context("invalid plan ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let plan = state
        .stored_plan(&plan_id)
        .cloned()
        .context("stored plan not found")?;

    println!("Stored plan for world {}:", state.id);
    print_stored_plan(&plan);

    if let Some(path) = options.output_json {
        write_json_file(path, &plan)?;
        println!("Saved stored plan JSON: {}", path.display());
    }

    Ok(())
}

async fn cmd_plans_delete(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    plan_id: &str,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let plan_id: uuid::Uuid = plan_id.parse().context("invalid plan ID")?;
    let mut state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let plan = state
        .stored_plans
        .remove(&plan_id)
        .context("stored plan not found")?;
    store
        .save(&state)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Deleted stored plan from world {}:", state.id);
    print_stored_plan(&plan);
    Ok(())
}

async fn cmd_eval(store: Option<&dyn StateStore>, options: EvalOptions<'_>) -> Result<()> {
    if options.list_suites || options.list_metrics {
        if options.list_suites {
            println!("Built-in evaluation suites:");
            for suite_name in EvalSuite::builtin_names() {
                println!("  {suite_name}");
            }
        }
        if options.list_metrics {
            println!("Available evaluation metrics:");
            for metric_name in available_eval_metric_names().split(", ") {
                println!("  {metric_name}");
            }
        }
        return Ok(());
    }

    let suite = load_eval_suite(options.suite_name, options.suite_json)
        .with_context(|| format!("available suites: {}", available_eval_suite_names()))?;
    let registry = auto_detect_registry();
    let provider_names = resolve_eval_provider_names(options.providers, &suite);
    let mut provider_list: Vec<&dyn WorldModelProvider> = Vec::new();
    for provider_name in &provider_names {
        provider_list.push(require_provider(&registry, provider_name)?);
    }

    let world_state =
        load_optional_world_state_input(store, options.world, options.world_snapshot).await?;
    let run_options = worldforge_eval::EvalRunOptions {
        num_samples: options.num_samples,
        ..Default::default()
    };
    let report = match world_state.as_ref() {
        Some(state) => suite
            .run_with_world_state_and_options(&provider_list, state, &run_options)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?,
        None => suite
            .run_with_options(&provider_list, &run_options)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?,
    };

    if let Some(path) = options.output_json {
        write_text_file(
            path,
            &report
                .render(EvalReportFormat::Json)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        )?;
        println!("Saved evaluation JSON: {}", path.display());
    }
    if let Some(path) = options.output_markdown {
        write_text_file(
            path,
            &report
                .render(EvalReportFormat::Markdown)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        )?;
        println!("Saved evaluation Markdown: {}", path.display());
    }
    if let Some(path) = options.output_csv {
        write_text_file(
            path,
            &report
                .render(EvalReportFormat::Csv)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        )?;
        println!("Saved evaluation CSV: {}", path.display());
    }

    println!("Evaluation Report: {}", report.suite);
    println!();
    for entry in &report.leaderboard {
        println!(
            "  {} — avg score: {:.2}, latency: {}ms, passed: {}/{}",
            entry.provider,
            entry.average_score,
            entry.average_latency_ms,
            entry.scenarios_passed,
            entry.total_scenarios
        );
    }
    println!();
    println!("Provider summaries:");
    for summary in &report.provider_summaries {
        println!(
            "  {} — scenario pass: {:.0}%, outcome pass: {:.0}%",
            summary.provider,
            summary.scenario_pass_rate * 100.0,
            summary.outcome_pass_rate * 100.0
        );
        let mut dimensions: Vec<_> = summary.dimension_scores.iter().collect();
        dimensions.sort_by(|a, b| a.0.cmp(b.0));
        for (dimension, score) in dimensions {
            println!("    {dimension}: {:.2}", score);
        }
        if let Some(sampling) = eval_sampling_summary(summary.sampling.as_ref()) {
            println!("    sampling: {sampling}");
        }
    }
    println!();
    println!("Dimension summaries:");
    for summary in &report.dimension_summaries {
        match (&summary.best_provider, summary.best_score) {
            (Some(provider), Some(score)) => {
                println!(
                    "  {} — best: {} ({:.2})",
                    summary.dimension, provider, score
                );
            }
            _ => println!("  {} — no scores recorded", summary.dimension),
        }
    }
    println!();
    println!("Scenario summaries:");
    for summary in &report.scenario_summaries {
        match (&summary.best_provider, summary.best_score) {
            (Some(provider), Some(score)) => println!(
                "  {} — best: {} ({:.2}), passed by: {}",
                summary.scenario,
                provider,
                score,
                summary.passed_by.join(", ")
            ),
            _ => println!("  {} — no scored providers", summary.scenario),
        }
    }
    println!();
    for result in &report.results {
        println!(
            "  Scenario: {} (provider: {})",
            result.scenario, result.provider
        );
        for outcome in &result.outcomes {
            let status = if outcome.passed { "PASS" } else { "FAIL" };
            println!(
                "    [{status}] {}{}",
                outcome.description,
                outcome
                    .details
                    .as_ref()
                    .map(|d| format!(" ({d})"))
                    .unwrap_or_default()
            );
        }
    }
    Ok(())
}

async fn cmd_compare(
    store: &dyn StateStore,
    world_id: Option<&str>,
    world_snapshot: Option<&Path>,
    action_str: Option<&str>,
    providers_str: Option<&str>,
    options: CompareOptions<'_>,
) -> Result<()> {
    let multi = if !options.prediction_json.is_empty() {
        let predictions = load_prediction_files(options.prediction_json)?;
        MultiPrediction::try_from_predictions(predictions).map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        let action_str = action_str
            .context("compare requires --action when --prediction-json is not provided")?;
        let providers_str = providers_str
            .context("compare requires --providers when --prediction-json is not provided")?;
        let state = load_optional_world_state_input(Some(store), world_id, world_snapshot)
            .await?
            .context(
                "compare requires --world or --world-snapshot when --prediction-json is not provided",
            )?;

        let provider_names = parse_provider_names(providers_str);
        let registry = Arc::new(auto_detect_registry());
        for provider_name in &provider_names {
            require_provider(&registry, provider_name)?;
        }
        if let Some(fallback_provider) = options.fallback_provider {
            require_provider(&registry, fallback_provider)?;
        }

        let default_provider = provider_names.first().map(String::as_str).unwrap_or("mock");
        let action = parse_action(action_str, &state)?;
        let world = worldforge_core::world::World::new(state, default_provider, registry);
        let config = PredictionConfig {
            steps: options.steps,
            guardrails: resolve_guardrails(
                read_guardrails(options.guardrails_json)?,
                options.disable_guardrails,
            ),
            max_latency_ms: options.timeout_ms,
            fallback_provider: options.fallback_provider.map(ToOwned::to_owned),
            ..PredictionConfig::default()
        };

        let provider_names: Vec<&str> = provider_names.iter().map(String::as_str).collect();
        world
            .predict_multi(&action, &provider_names, &config)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
    };

    print_comparison_results(&multi);
    if let Some(path) = options.output_json {
        write_json_file(path, &multi)?;
        println!();
        println!("Saved comparison JSON: {}", path.display());
    }
    if let Some(path) = options.output_markdown {
        write_text_file(path, &multi.render(ComparisonReportFormat::Markdown)?)?;
        println!("Saved comparison Markdown: {}", path.display());
    }
    if let Some(path) = options.output_csv {
        write_text_file(path, &multi.render(ComparisonReportFormat::Csv)?)?;
        println!("Saved comparison CSV: {}", path.display());
    }
    Ok(())
}

fn load_prediction_files(paths: &[PathBuf]) -> Result<Vec<Prediction>> {
    paths
        .iter()
        .map(|path| read_json_file(path.as_path()))
        .collect()
}

async fn load_optional_world_state_input(
    store: Option<&dyn StateStore>,
    world_id: Option<&str>,
    world_snapshot: Option<&Path>,
) -> Result<Option<WorldState>> {
    match (world_id, world_snapshot) {
        (Some(_), Some(_)) => anyhow::bail!("world ID and world snapshot are mutually exclusive"),
        (Some(world_id), None) => {
            let store = store.context("world-backed operation requires a state store")?;
            let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
            let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(Some(state))
        }
        (None, Some(snapshot)) => read_world_state_snapshot(snapshot, None).map(Some),
        (None, None) => Ok(None),
    }
}

fn print_comparison_results(multi: &MultiPrediction) {
    println!("Comparison results:");
    println!("  Agreement score: {:.2}", multi.agreement_score);
    println!(
        "  Best provider: {}",
        multi.predictions[multi.best_prediction].provider
    );
    println!(
        "  Consensus: {} shared objects, {} shared relationships, avg quality {:.2}, avg latency {}ms",
        multi.comparison.consensus.shared_object_count,
        multi.comparison.consensus.shared_relationship_count,
        multi.comparison.consensus.average_quality_score,
        multi.comparison.consensus.average_latency_ms
    );
    println!("  Summary: {}", multi.comparison.summary);
    println!();
    for score in &multi.comparison.scores {
        println!(
            "  {} — quality: {:.2}, physics: {:.2}, confidence: {:.2}, latency: {}ms, guardrails: {}/{} passed",
            score.provider,
            score.quality_score,
            score.physics_scores.overall,
            score.confidence,
            score.latency_ms,
            score.guardrails.passed_count,
            score.guardrails.evaluated_count
        );
        println!(
            "    objects: {} ({:+}), preserved {}/{}, relationships: {} ({:+}), preserved {}/{}",
            score.state.output_object_count,
            score.state.object_count_delta,
            score.state.preserved_object_count,
            score.state.input_object_count,
            score.state.output_relationship_count,
            score.state.relationship_count_delta,
            score.state.preserved_relationship_count,
            score.state.input_relationship_count
        );
        println!(
            "    drift: avg {:.3}, max {:.3}",
            score.state.average_position_shift, score.state.max_position_shift
        );
        if !score.state.novel_objects.is_empty() {
            println!(
                "    novel: {}",
                score
                    .state
                    .novel_objects
                    .iter()
                    .map(|object| object.object_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if !score.state.dropped_objects.is_empty() {
            println!(
                "    dropped: {}",
                score
                    .state
                    .dropped_objects
                    .iter()
                    .map(|object| object.object_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    if !multi.comparison.consensus.shared_objects.is_empty() {
        println!();
        println!(
            "Shared objects: {}",
            multi
                .comparison
                .consensus
                .shared_objects
                .iter()
                .map(|object| object.object_name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    if !multi.comparison.pairwise_agreements.is_empty() {
        println!();
        println!("Pairwise agreement:");
        for pair in &multi.comparison.pairwise_agreements {
            println!(
                "  {} vs {} — overall {:.2}, objects {:.2}, relationships {:.2}, avg position delta {:.3}, guardrails {:.2}",
                pair.provider_a,
                pair.provider_b,
                pair.agreement_score,
                pair.object_overlap_rate,
                pair.relationship_overlap_rate,
                pair.average_position_delta,
                pair.guardrail_agreement_rate
            );
        }
    }
}

async fn cmd_plan(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    goal: Option<&str>,
    options: PlanOptions<'_>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    let registry = Arc::new(auto_detect_registry());
    if options.fallback_provider.is_none() {
        require_provider(&registry, options.provider)?;
    }
    if let Some(fallback_provider) = options.fallback_provider {
        require_provider(&registry, fallback_provider)?;
    }
    let mut world = worldforge_core::world::World::new(state.clone(), options.provider, registry);
    let planner = planner_from_name(
        options.planner_name,
        options.max_steps,
        options.planner_options,
    )?;
    let guardrails = resolve_guardrails(
        read_guardrails(options.guardrails_json)?,
        options.disable_guardrails,
    );
    let goal = load_plan_goal(goal, options.goal_json)?;

    let request = PlanRequest {
        current_state: state,
        goal,
        max_steps: options.max_steps,
        guardrails,
        planner,
        timeout_seconds: options.timeout,
        fallback_provider: options.fallback_provider.map(ToOwned::to_owned),
    };

    let record = world
        .plan_and_store(&request)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut plan = record.plan;
    let proof_attached = attach_plan_verification(&mut plan, options.verify_backend)?;
    world
        .state
        .store_plan_record(worldforge_core::prediction::StoredPlanRecord::from_request(
            options.provider,
            &request,
            &plan,
        ));
    store
        .save(world.current_state())
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Plan generated:");
    println!("  Plan ID: {}", plan.stored_plan_id.unwrap_or(record.id));
    println!("  Actions: {}", plan.actions.len());
    println!("  Success probability: {:.2}", plan.success_probability);
    println!("  Planning time: {}ms", plan.planning_time_ms);
    println!("  Iterations: {}", plan.iterations_used);
    if proof_attached {
        let backend = plan
            .verification_proof
            .as_ref()
            .map(|proof| proof.backend.as_str())
            .unwrap_or("unknown");
        println!("  Verification proof: attached via {backend}");
    }
    println!();
    for (i, action) in plan.actions.iter().enumerate() {
        println!("  Step {}: {:?}", i + 1, action);
        if let Some(gr) = plan.guardrail_compliance.get(i) {
            for r in gr {
                let status = if r.passed { "PASS" } else { "FAIL" };
                println!("    [{status}] {}", r.guardrail_name);
            }
        }
    }

    if let Some(path) = options.output_json {
        write_json_file(path, &plan)?;
        println!();
        println!("Saved plan JSON: {}", path.display());
    }

    Ok(())
}

async fn cmd_execute_plan(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    options: ExecutePlanOptions<'_>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let plan = match (options.plan_json, options.plan_id) {
        (Some(path), None) => read_json_file(path)?,
        (None, Some(plan_id)) => {
            let plan_id: uuid::Uuid = plan_id.parse().context("invalid plan ID")?;
            state
                .stored_plan(&plan_id)
                .map(|record| record.plan.clone())
                .context("stored plan not found")?
        }
        _ => anyhow::bail!("execute-plan requires exactly one of --plan-json or --plan-id"),
    };

    let provider_name = options
        .provider
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| state.current_state_provider());
    let provider_name = provider_name.as_str();
    let registry = Arc::new(auto_detect_registry());
    require_provider(&registry, provider_name)?;
    if let Some(fallback_provider) = options.fallback_provider {
        require_provider(&registry, fallback_provider)
            .context("invalid fallback provider for execute-plan command")?;
    }

    let mut world = worldforge_core::world::World::new(state, provider_name, registry);
    let mut config = PredictionConfig {
        steps: options.steps,
        return_video: options.return_video,
        guardrails: read_guardrails(options.guardrails_json)?,
        fallback_provider: options.fallback_provider.map(ToOwned::to_owned),
        max_latency_ms: options.timeout_ms,
        ..PredictionConfig::default()
    };
    if options.disable_guardrails {
        config = config.disable_guardrails();
    }

    let report = world
        .execute_plan_with_provider(&plan, &config, provider_name)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    store
        .save(world.current_state())
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    print_plan_execution(&report);
    for (index, prediction) in report.predictions.iter().enumerate() {
        println!(
            "  Step {}: {} via {} (physics {:.2}, latency {}ms)",
            index + 1,
            serde_json::to_string(&prediction.action)
                .unwrap_or_else(|_| format!("{:?}", prediction.action)),
            prediction.provider,
            prediction.physics_scores.overall,
            prediction.latency_ms
        );
    }

    if let Some(path) = options.output_json {
        write_json_file(path, &report)?;
        println!("Saved execution JSON: {}", path.display());
    }

    Ok(())
}

async fn cmd_verify(
    store: &(impl StateStore + ?Sized),
    world_id: Option<&str>,
    options: VerifyOptions<'_>,
) -> Result<()> {
    let verifier = verifier_for_backend(options.backend.as_core());
    let loaded_state = match world_id {
        Some(world_id) => {
            let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
            Some(store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?)
        }
        None => None,
    };

    match options.proof_type {
        "inference" => {
            let bundle = match (
                options.prediction_json,
                options.input_state_json,
                options.output_state_json,
            ) {
                (Some(prediction_path), None, None) => {
                    let prediction: Prediction = read_json_file(prediction_path)?;
                    prove_prediction_inference(verifier.as_ref(), &prediction)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                }
                (None, Some(input_path), Some(output_path)) => {
                    let input_state: WorldState = read_json_file(input_path)?;
                    let output_state: WorldState = read_json_file(output_path)?;
                    let provider_name = options
                        .provider
                        .filter(|name| !name.is_empty())
                        .unwrap_or(output_state.metadata.created_by.as_str());
                    prove_inference_transition(
                        verifier.as_ref(),
                        provider_name,
                        &input_state,
                        &output_state,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                }
                (None, None, None) => {
                    let state = loaded_state.as_ref().context(
                        "inference verification requires --prediction-json, --world with at least two recorded history entries, or both --input-state-json and --output-state-json",
                    )?;
                    prove_latest_inference(verifier.as_ref(), state, options.provider)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                }
                _ => anyhow::bail!(
                    "inference verification requires exactly one of --prediction-json, both --input-state-json and --output-state-json, or --world"
                ),
            };

            print_verification_bundle("InferenceVerification", &bundle)?;
            if let Some(path) = options.output_json {
                write_json_file(path, &bundle)?;
                println!();
                println!("Saved verification bundle: {}", path.display());
            }
        }
        "guardrail" => {
            let plan = if let Some(plan_path) = options.plan_json {
                read_json_file(plan_path)?
            } else if let Some(plan_id) = options.plan_id {
                let state = loaded_state
                    .as_ref()
                    .context("guardrail verification with --plan-id requires --world")?;
                let plan_id: uuid::Uuid = plan_id.parse().context("invalid plan ID")?;
                state
                    .stored_plan(&plan_id)
                    .map(|record| record.plan.clone())
                    .context("stored plan not found")?
            } else {
                let state = loaded_state.as_ref().context(
                    "guardrail verification requires --plan-json, --plan-id with --world, or --world together with --goal/--goal-json",
                )?;
                let goal = load_plan_goal(options.goal, options.goal_json).context(
                    "guardrail verification requires --goal or --goal-json when neither --plan-json nor --plan-id is provided",
                )?;
                let provider_name = resolve_provider_name(state, options.provider).to_string();
                let registry = Arc::new(auto_detect_registry());
                if options.fallback_provider.is_none() {
                    require_provider(&registry, &provider_name)?;
                }
                if let Some(fallback_provider) = options.fallback_provider {
                    require_provider(&registry, fallback_provider)?;
                }
                let world =
                    worldforge_core::world::World::new(state.clone(), &provider_name, registry);
                let request = PlanRequest {
                    current_state: state.clone(),
                    goal,
                    max_steps: options.max_steps,
                    guardrails: resolve_guardrails(
                        read_guardrails(options.guardrails_json)?,
                        options.disable_guardrails,
                    ),
                    planner: planner_from_name(
                        options.planner_name,
                        options.max_steps,
                        options.planner_options,
                    )?,
                    timeout_seconds: options.timeout,
                    fallback_provider: options.fallback_provider.map(ToOwned::to_owned),
                };
                world
                    .plan(&request)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };

            let bundle = prove_guardrail_plan(verifier.as_ref(), &plan)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            print_verification_bundle("GuardrailCompliance", &bundle)?;
            if let Some(path) = options.output_json {
                write_json_file(path, &bundle)?;
                println!();
                println!("Saved verification bundle: {}", path.display());
            }
        }
        "provenance" => {
            let state = match options.output_state_json.or(options.input_state_json) {
                Some(state_path) => read_json_file(state_path)?,
                None => loaded_state
                    .as_ref()
                    .cloned()
                    .context("provenance verification requires --world or a state JSON input")?,
            };
            let timestamp = chrono::Utc::now().timestamp() as u64;
            let bundle =
                prove_provenance(verifier.as_ref(), &state, options.source_label, timestamp)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            print_verification_bundle("DataProvenance", &bundle)?;
            if let Some(path) = options.output_json {
                write_json_file(path, &bundle)?;
                println!();
                println!("Saved verification bundle: {}", path.display());
            }
        }
        other => anyhow::bail!(
            "unknown proof type: {other}. Available: inference, guardrail, provenance"
        ),
    }

    Ok(())
}

async fn cmd_verify_proof(options: VerifyProofOptions<'_>) -> Result<()> {
    if let Some(path) = options.proof_json {
        let proof: ZkProof = read_json_file(path)?;
        let verifier = verifier_for_proof(&proof);
        let report = ProofVerificationReport {
            verification: verify_proof(verifier.as_ref(), &proof)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            proof,
        };
        print_proof_verification(&report);
        if let Some(output_path) = options.output_json {
            write_json_file(output_path, &report)?;
            println!();
            println!("Saved verification report: {}", output_path.display());
        }
        return Ok(());
    }

    if let Some(path) = options.inference_bundle_json {
        let bundle: VerificationBundle<worldforge_verify::InferenceArtifact> =
            read_json_file(path)?;
        let verifier = verifier_for_backend(bundle.proof.backend);
        let report =
            verify_bundle(verifier.as_ref(), &bundle).map_err(|e| anyhow::anyhow!("{e}"))?;
        print_bundle_verification("InferenceVerification", &report)?;
        if let Some(output_path) = options.output_json {
            write_json_file(output_path, &report)?;
            println!();
            println!("Saved verification report: {}", output_path.display());
        }
        return Ok(());
    }

    if let Some(path) = options.guardrail_bundle_json {
        let bundle: VerificationBundle<worldforge_verify::GuardrailArtifact> =
            read_json_file(path)?;
        let verifier = verifier_for_backend(bundle.proof.backend);
        let report =
            verify_bundle(verifier.as_ref(), &bundle).map_err(|e| anyhow::anyhow!("{e}"))?;
        print_bundle_verification("GuardrailCompliance", &report)?;
        if let Some(output_path) = options.output_json {
            write_json_file(output_path, &report)?;
            println!();
            println!("Saved verification report: {}", output_path.display());
        }
        return Ok(());
    }

    if let Some(path) = options.provenance_bundle_json {
        let bundle: VerificationBundle<worldforge_verify::ProvenanceArtifact> =
            read_json_file(path)?;
        let verifier = verifier_for_backend(bundle.proof.backend);
        let report =
            verify_bundle(verifier.as_ref(), &bundle).map_err(|e| anyhow::anyhow!("{e}"))?;
        print_bundle_verification("DataProvenance", &report)?;
        if let Some(output_path) = options.output_json {
            write_json_file(output_path, &report)?;
            println!();
            println!("Saved verification report: {}", output_path.display());
        }
        return Ok(());
    }

    anyhow::bail!(
        "verify-proof requires one of --proof-json, --inference-bundle-json, --guardrail-bundle-json, or --provenance-bundle-json"
    )
}

async fn cmd_health(provider_name: &str) -> Result<()> {
    let registry = auto_detect_registry();
    if provider_name == "all" {
        for report in registry.health_check_all().await {
            print_provider_health_report(&report);
        }
        return Ok(());
    }

    match registry.health_check(provider_name).await {
        Ok(report) => print_provider_health_report(&report),
        Err(error) => println!("  [NOT FOUND] {provider_name}: {error}"),
    }
    Ok(())
}

async fn cmd_providers(capability: Option<&str>, include_health: bool) -> Result<()> {
    let registry = auto_detect_registry();

    if include_health {
        let reports = match capability {
            Some(capability) => {
                let reports = registry.health_check_by_capability(capability).await;
                if reports.is_empty() {
                    anyhow::bail!(
                        "no providers matched capability '{capability}'. Available capability filters: {}",
                        available_provider_capabilities()
                    );
                }
                reports
            }
            None => registry.health_check_all().await,
        };

        for report in reports {
            print_provider_descriptor(&ProviderDescriptor {
                name: report.name.clone(),
                capabilities: report.capabilities.clone(),
            });
            print_provider_health_report(&report);
        }
        return Ok(());
    }

    let descriptors = match capability {
        Some(capability) => {
            let descriptors = registry.describe_by_capability(capability);
            if descriptors.is_empty() {
                anyhow::bail!(
                    "no providers matched capability '{capability}'. Available capability filters: {}",
                    available_provider_capabilities()
                );
            }
            descriptors
        }
        None => registry.describe_all(),
    };

    for descriptor in descriptors {
        print_provider_descriptor(&descriptor);
    }

    Ok(())
}

fn cmd_estimate(provider_name: &str, options: EstimateOptions) -> Result<()> {
    let registry = auto_detect_registry();
    let operation = build_operation(options.operation, &options);
    let estimate = registry
        .estimate_cost(provider_name, &operation)
        .map_err(|e| {
            anyhow::anyhow!(
                "{e}. Available providers: {}",
                available_provider_names(&registry)
            )
        })?;

    println!("Provider: {provider_name}");
    println!("Operation: {:?}", operation);
    println!("Estimated USD: {:.4}", estimate.usd);
    println!("Estimated credits: {:.2}", estimate.credits);
    println!("Estimated latency: {}ms", estimate.estimated_latency_ms);
    Ok(())
}

async fn cmd_serve(cli: &Cli, bind: &str) -> Result<()> {
    let state_store = CliStateStoreConfig::from_cli(cli);
    let registry = Arc::new(auto_detect_registry());
    let server_config = worldforge_server::ServerConfig {
        bind_address: bind.to_string(),
        state_dir: state_store.state_dir.display().to_string(),
        state_backend: state_store.state_backend.as_str().to_string(),
        state_file_format: state_store.state_file_format.as_core().as_str().to_string(),
        state_db_path: state_store
            .state_db_path
            .map(|path| path.display().to_string()),
        state_redis_url: state_store.state_redis_url.map(ToOwned::to_owned),
        state_s3_bucket: state_store.state_s3_bucket.map(ToOwned::to_owned),
        state_s3_region: state_store.state_s3_region.map(ToOwned::to_owned),
        state_s3_access_key_id: state_store.state_s3_access_key_id.map(ToOwned::to_owned),
        state_s3_secret_access_key: state_store
            .state_s3_secret_access_key
            .map(ToOwned::to_owned),
        state_s3_endpoint: state_store.state_s3_endpoint.map(ToOwned::to_owned),
        state_s3_session_token: state_store.state_s3_session_token.map(ToOwned::to_owned),
        state_s3_prefix: state_store.state_s3_prefix.map(ToOwned::to_owned),
    };

    worldforge_server::serve(server_config, registry)
        .await
        .context("failed to start server")
}

/// Parse a CLI action string into a standardized Action.
fn parse_action(s: &str, state: &WorldState) -> Result<Action> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        anyhow::bail!("empty action string");
    }

    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return serde_json::from_str(trimmed)
            .context("failed to parse typed action JSON; use the core Action serde shape");
    }

    if let Some(rest) = trimmed.strip_prefix("sequence ") {
        return Ok(Action::Sequence(parse_compound_actions(rest, state)?));
    }
    if let Some(rest) = trimmed.strip_prefix("parallel ") {
        return Ok(Action::Parallel(parse_compound_actions(rest, state)?));
    }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    match parts[0] {
        "move" => {
            if parts.len() < 4 {
                anyhow::bail!("move expects `move <x> <y> <z> [speed]`");
            }
            let target = parse_position_tokens(&parts[1..4])?;
            let speed = parse_optional_f32(parts.get(4).copied(), 1.0, "speed")?;
            Ok(Action::Move { target, speed })
        }
        "grasp" => Ok(Action::Grasp {
            object: resolve_object_reference(parts.get(1).copied(), state)?,
            grip_force: parse_optional_f32(parts.get(2).copied(), 1.0, "grip_force")?,
        }),
        "release" => Ok(Action::Release {
            object: resolve_object_reference(parts.get(1).copied(), state)?,
        }),
        "push" => {
            let object = resolve_object_reference(parts.get(1).copied(), state)?;
            let (direction, consumed) = parse_direction_spec(&parts[2..])?;
            let force = parse_optional_f32(parts.get(2 + consumed).copied(), 1.0, "force")?;
            Ok(Action::Push {
                object,
                direction,
                force,
            })
        }
        "rotate" => {
            let object = resolve_object_reference(parts.get(1).copied(), state)?;
            let (axis, consumed) = parse_axis_spec(&parts[2..])?;
            let angle = parse_required_f32(
                parts.get(2 + consumed).copied(),
                "angle",
                "rotate expects `rotate <object> <axis|x y z> <angle>`",
            )?;
            Ok(Action::Rotate {
                object,
                axis,
                angle,
            })
        }
        "place" => {
            let object = resolve_object_reference(parts.get(1).copied(), state)?;
            if parts.len() < 5 {
                anyhow::bail!("place expects `place <object> <x> <y> <z>`");
            }
            Ok(Action::Place {
                object,
                target: parse_position_tokens(&parts[2..5])?,
            })
        }
        "camera-move" => Ok(Action::CameraMove {
            delta: parse_pose_spec(
                &parts[1..],
                "camera-move expects `camera-move <x> <y> <z> [w x y z]`",
            )?,
        }),
        "camera-look-at" | "look-at" => {
            if parts.len() < 4 {
                anyhow::bail!("camera-look-at expects `camera-look-at <x> <y> <z>`");
            }
            Ok(Action::CameraLookAt {
                target: parse_position_tokens(&parts[1..4])?,
            })
        }
        "navigate" => {
            if parts.len() < 4 || !(parts.len() - 1).is_multiple_of(3) {
                anyhow::bail!("navigate expects `navigate <x1> <y1> <z1> [<x2> <y2> <z2> ...]`");
            }
            let mut waypoints = Vec::new();
            for chunk in parts[1..].chunks(3) {
                waypoints.push(parse_position_tokens(chunk)?);
            }
            Ok(Action::Navigate { waypoints })
        }
        "teleport" => Ok(Action::Teleport {
            destination: parse_pose_spec(
                &parts[1..],
                "teleport expects `teleport <x> <y> <z> [w x y z]`",
            )?,
        }),
        "set-weather" => {
            let weather = match parts
                .get(1)
                .copied()
                .unwrap_or("clear")
                .to_ascii_lowercase()
                .as_str()
            {
                "clear" => Weather::Clear,
                "cloudy" => Weather::Cloudy,
                "rain" => Weather::Rain,
                "snow" => Weather::Snow,
                "fog" => Weather::Fog,
                "night" => Weather::Night,
                other => anyhow::bail!("unknown weather: {other}"),
            };
            Ok(Action::SetWeather { weather })
        }
        "set-lighting" => Ok(Action::SetLighting {
            time_of_day: parse_optional_f32(parts.get(1).copied(), 12.0, "time_of_day")?,
        }),
        "spawn" => {
            let template = parts.get(1).copied().unwrap_or("object").to_string();
            let pose = if parts.len() > 2 {
                parse_pose_spec(
                    &parts[2..],
                    "spawn expects `spawn <template> [x y z [w x y z]]`",
                )?
            } else {
                Pose::default()
            };
            Ok(Action::SpawnObject { template, pose })
        }
        "remove" | "remove-object" => Ok(Action::RemoveObject {
            object: resolve_object_reference(parts.get(1).copied(), state)?,
        }),
        "raw" => {
            let provider = parts
                .get(1)
                .copied()
                .context("raw expects `raw <provider> <json>`")?;
            let payload = parts
                .get(2..)
                .filter(|tokens| !tokens.is_empty())
                .map(|tokens| tokens.join(" "))
                .context("raw expects `raw <provider> <json>`")?;
            let data = serde_json::from_str(&payload).with_context(|| {
                format!("failed to parse raw JSON payload for provider `{provider}`")
            })?;
            Ok(Action::Raw {
                provider: provider.to_string(),
                data,
            })
        }
        _ => Ok(Action::Raw {
            provider: "cli".to_string(),
            data: serde_json::json!({ "text": trimmed }),
        }),
    }
}

fn parse_compound_actions(input: &str, state: &WorldState) -> Result<Vec<Action>> {
    let actions: Result<Vec<_>> = input
        .split(';')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(|segment| parse_action(segment, state))
        .collect();
    let actions = actions?;
    if actions.is_empty() {
        anyhow::bail!("compound actions require at least one sub-action");
    }
    Ok(actions)
}

fn resolve_object_reference(reference: Option<&str>, state: &WorldState) -> Result<uuid::Uuid> {
    let reference = reference.context("missing object reference")?;
    if let Ok(id) = reference.parse::<uuid::Uuid>() {
        return Ok(id);
    }
    state
        .scene
        .find_object_by_name(reference)
        .map(|object| object.id)
        .with_context(|| {
            format!("unknown object reference `{reference}`; use an object name from the world or a UUID")
        })
}

fn parse_optional_f32(token: Option<&str>, default: f32, name: &str) -> Result<f32> {
    token
        .map(|value| parse_required_f32(Some(value), name, &format!("invalid {name} value")))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_required_f32(token: Option<&str>, name: &str, message: &str) -> Result<f32> {
    let token = token.ok_or_else(|| anyhow::anyhow!("{message}"))?;
    token
        .parse::<f32>()
        .with_context(|| format!("invalid {name}: {token}"))
}

fn parse_position_tokens(tokens: &[&str]) -> Result<Position> {
    if tokens.len() != 3 {
        anyhow::bail!("expected exactly three coordinates");
    }
    Ok(Position {
        x: parse_required_f32(tokens.first().copied(), "x", "missing x coordinate")?,
        y: parse_required_f32(tokens.get(1).copied(), "y", "missing y coordinate")?,
        z: parse_required_f32(tokens.get(2).copied(), "z", "missing z coordinate")?,
    })
}

fn parse_vec3_tokens(tokens: &[&str]) -> Result<Vec3> {
    let position = parse_position_tokens(tokens)?;
    Ok(Vec3 {
        x: position.x,
        y: position.y,
        z: position.z,
    })
}

fn parse_pose_spec(tokens: &[&str], message: &str) -> Result<Pose> {
    match tokens.len() {
        3 => Ok(Pose {
            position: parse_position_tokens(tokens)?,
            rotation: Rotation::default(),
        }),
        7 => Ok(Pose {
            position: parse_position_tokens(&tokens[..3])?,
            rotation: Rotation {
                w: parse_required_f32(tokens.get(3).copied(), "w", message)?,
                x: parse_required_f32(tokens.get(4).copied(), "x", message)?,
                y: parse_required_f32(tokens.get(5).copied(), "y", message)?,
                z: parse_required_f32(tokens.get(6).copied(), "z", message)?,
            },
        }),
        _ => anyhow::bail!("{message}"),
    }
}

fn parse_direction_spec(tokens: &[&str]) -> Result<(Vec3, usize)> {
    match tokens.first().copied() {
        Some("left") => Ok((
            Vec3 {
                x: -1.0,
                y: 0.0,
                z: 0.0,
            },
            1,
        )),
        Some("right") => Ok((
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            1,
        )),
        Some("forward") => Ok((
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            1,
        )),
        Some("backward") => Ok((
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
            1,
        )),
        Some("up") => Ok((
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            1,
        )),
        Some("down") => Ok((
            Vec3 {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            },
            1,
        )),
        Some(_) if tokens.len() >= 3 => Ok((parse_vec3_tokens(&tokens[..3])?, 3)),
        _ => anyhow::bail!(
            "push expects `push <object> <left|right|forward|backward|up|down|x y z> [force]`"
        ),
    }
}

fn parse_axis_spec(tokens: &[&str]) -> Result<(Vec3, usize)> {
    match tokens.first().copied() {
        Some("x") => Ok((
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            1,
        )),
        Some("y") => Ok((
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            1,
        )),
        Some("z") => Ok((
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            1,
        )),
        Some(_) if tokens.len() >= 3 => Ok((parse_vec3_tokens(&tokens[..3])?, 3)),
        _ => anyhow::bail!("rotate expects `rotate <object> <x|y|z|ax ay az> <angle>`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use worldforge_core::scene::SceneObject;
    use worldforge_core::types::{DType, Device, Mesh, Tensor, TensorData};
    use worldforge_verify::{MockVerifier, ZkVerifier};

    #[allow(dead_code)]
    mod fake_state_backends {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/support/fake_state_backends.rs"
        ));
    }

    fn env_mutex() -> &'static Mutex<()> {
        static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_MUTEX.get_or_init(|| Mutex::new(()))
    }

    fn with_env_vars<F>(vars: &[(&str, Option<&str>)], test: F)
    where
        F: FnOnce(),
    {
        let _guard = env_mutex().lock().unwrap();
        let saved: Vec<_> = vars
            .iter()
            .map(|(name, _)| ((*name).to_string(), std::env::var(name).ok()))
            .collect();

        for (name, value) in vars {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(test));

        for (name, value) in saved {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(&name),
            }
        }

        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    struct TestJepaModelDir {
        path: PathBuf,
    }

    impl TestJepaModelDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "wf-cli-jepa-{label}-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn write_assets(&self, representation_dim: u32) {
            fs::write(
                self.path.join("model.safetensors"),
                b"cli-jepa-latent-weights",
            )
            .unwrap();
            fs::write(
                self.path.join("worldforge-jepa.json"),
                format!(
                    r#"{{
                        "model_name": "vjepa2-local",
                        "representation_dim": {representation_dim},
                        "action_gain": 1.15,
                        "temporal_smoothness": 0.88,
                        "gravity_bias": 0.91,
                        "collision_bias": 0.89,
                        "confidence_bias": 0.05
                    }}"#
                ),
            )
            .unwrap();
        }
    }

    impl Drop for TestJepaModelDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn world_with_named_object(name: &str) -> (WorldState, uuid::Uuid) {
        let mut state = WorldState::new("cli-action-tests", "mock");
        let object = SceneObject::new(
            name,
            Pose::default(),
            BBox {
                min: Position {
                    x: -0.05,
                    y: -0.05,
                    z: -0.05,
                },
                max: Position {
                    x: 0.05,
                    y: 0.05,
                    z: 0.05,
                },
            },
        );
        let id = object.id;
        state.scene.add_object(object);
        (state, id)
    }

    fn sample_snapshot_state(name: &str) -> WorldState {
        WorldState::from_prompt(
            "A kitchen counter with a red mug and a wooden block",
            "mock",
            Some(name),
        )
        .unwrap()
    }

    fn legacy_snapshot_state(name: &str) -> WorldState {
        let mut state = sample_snapshot_state(name);
        state.history = StateHistory::default();

        let state_hash = worldforge_core::state::canonical_state_hash(&state).unwrap();
        state.history.push(worldforge_core::state::HistoryEntry {
            time: state.time,
            state_hash,
            action: None,
            prediction: None,
            provider: state.metadata.created_by.clone(),
            snapshot: None,
        });

        state
    }

    fn sample_stored_plan_record(plan_id: uuid::Uuid) -> StoredPlanRecord {
        let mut predicted_state = WorldState::new("stored-plan-state", "mock");
        predicted_state.stored_plans.clear();

        StoredPlanRecord {
            id: plan_id,
            provider: "mock".to_string(),
            planner: "sampling".to_string(),
            goal_summary: "spawn cube".to_string(),
            created_at: chrono::Utc::now(),
            plan: Plan {
                actions: vec![Action::Move {
                    target: Position {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    speed: 1.0,
                }],
                predicted_states: vec![predicted_state],
                predicted_videos: None,
                total_cost: 0.0,
                success_probability: 1.0,
                guardrail_compliance: vec![Vec::new()],
                planning_time_ms: 1,
                iterations_used: 1,
                stored_plan_id: Some(plan_id),
                verification_proof: None,
            },
        }
    }

    fn sample_prediction(provider: &str, output_x: f32, latency_ms: u64) -> Prediction {
        let mut input_state = WorldState::new(format!("{provider}-input"), provider);
        input_state.scene.add_object(SceneObject::new(
            "cube",
            Pose::default(),
            BBox {
                min: Position {
                    x: -0.05,
                    y: -0.05,
                    z: -0.05,
                },
                max: Position {
                    x: 0.05,
                    y: 0.05,
                    z: 0.05,
                },
            },
        ));
        let mut output_state = input_state.clone();
        let updated = output_state
            .scene
            .find_object_by_name_mut("cube")
            .expect("cube should exist");
        updated.set_position(Position {
            x: output_x,
            y: 0.0,
            z: 0.0,
        });

        Prediction {
            id: uuid::Uuid::new_v4(),
            provider: provider.to_string(),
            model: format!("{provider}-model"),
            input_state,
            action: Action::Move {
                target: Position {
                    x: output_x,
                    y: 0.0,
                    z: 0.0,
                },
                speed: 1.0,
            },
            output_state,
            video: None,
            confidence: 0.8,
            provenance: None,
            physics_scores: worldforge_core::prediction::PhysicsScores {
                overall: 0.8,
                object_permanence: 0.8,
                gravity_compliance: 0.8,
                collision_accuracy: 0.8,
                spatial_consistency: 0.8,
                temporal_consistency: 0.8,
            },
            latency_ms,
            cost: worldforge_core::provider::CostEstimate {
                usd: latency_ms as f64 / 1_000.0,
                credits: 1.0,
                estimated_latency_ms: latency_ms,
            },
            sampling: None,
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_parse_action_move() {
        let state = WorldState::new("parse", "mock");
        let action = parse_action("move 1 2 3", &state).unwrap();
        match action {
            Action::Move { target, .. } => {
                assert_eq!(target.x, 1.0);
                assert_eq!(target.y, 2.0);
                assert_eq!(target.z, 3.0);
            }
            _ => panic!("expected Move"),
        }
    }

    #[test]
    fn test_parse_action_weather() {
        let state = WorldState::new("parse", "mock");
        let action = parse_action("set-weather rain", &state).unwrap();
        match action {
            Action::SetWeather { weather } => assert_eq!(weather, Weather::Rain),
            _ => panic!("expected SetWeather"),
        }
    }

    #[test]
    fn test_parse_action_raw() {
        let state = WorldState::new("parse", "mock");
        let action = parse_action("dance mug", &state).unwrap();
        match action {
            Action::Raw { provider, .. } => assert_eq!(provider, "cli"),
            _ => panic!("expected Raw"),
        }
    }

    #[test]
    fn test_parse_action_spawn() {
        let state = WorldState::new("parse", "mock");
        let action = parse_action("spawn cube 1 2 3", &state).unwrap();
        match action {
            Action::SpawnObject { template, pose } => {
                assert_eq!(template, "cube");
                assert_eq!(pose.position.x, 1.0);
                assert_eq!(pose.position.y, 2.0);
                assert_eq!(pose.position.z, 3.0);
            }
            _ => panic!("expected SpawnObject"),
        }
    }

    #[test]
    fn test_parse_action_set_lighting() {
        let state = WorldState::new("parse", "mock");
        let action = parse_action("set-lighting 18.5", &state).unwrap();
        match action {
            Action::SetLighting { time_of_day } => {
                assert!((time_of_day - 18.5).abs() < f32::EPSILON)
            }
            _ => panic!("expected SetLighting"),
        }
    }

    #[test]
    fn test_parse_action_grasp_resolves_object_name() {
        let (state, object_id) = world_with_named_object("mug");
        let action = parse_action("grasp mug 0.7", &state).unwrap();
        match action {
            Action::Grasp { object, grip_force } => {
                assert_eq!(object, object_id);
                assert!((grip_force - 0.7).abs() < f32::EPSILON);
            }
            _ => panic!("expected Grasp"),
        }
    }

    #[test]
    fn test_parse_action_push_resolves_object_name_and_direction_keyword() {
        let (state, object_id) = world_with_named_object("mug");
        let action = parse_action("push mug left 2.5", &state).unwrap();
        match action {
            Action::Push {
                object,
                direction,
                force,
            } => {
                assert_eq!(object, object_id);
                assert_eq!(
                    direction,
                    Vec3 {
                        x: -1.0,
                        y: 0.0,
                        z: 0.0
                    }
                );
                assert!((force - 2.5).abs() < f32::EPSILON);
            }
            _ => panic!("expected Push"),
        }
    }

    #[test]
    fn test_parse_action_rotate_supports_axis_keyword() {
        let (state, object_id) = world_with_named_object("cube");
        let action = parse_action("rotate cube z 90", &state).unwrap();
        match action {
            Action::Rotate {
                object,
                axis,
                angle,
                ..
            } => {
                assert_eq!(object, object_id);
                assert_eq!(
                    axis,
                    Vec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0
                    }
                );
                assert!((angle - 90.0).abs() < f32::EPSILON);
            }
            _ => panic!("expected Rotate"),
        }
    }

    #[test]
    fn test_parse_action_navigation_and_camera_variants() {
        let state = WorldState::new("parse", "mock");

        let navigate = parse_action("navigate 0 0 0 1 1 1", &state).unwrap();
        match navigate {
            Action::Navigate { waypoints } => assert_eq!(waypoints.len(), 2),
            _ => panic!("expected Navigate"),
        }

        let teleport = parse_action("teleport 1 2 3 1 0 0 0", &state).unwrap();
        match teleport {
            Action::Teleport { destination } => {
                assert_eq!(destination.position.x, 1.0);
                assert_eq!(destination.rotation.w, 1.0);
            }
            _ => panic!("expected Teleport"),
        }
    }

    #[test]
    fn test_parse_action_sequence_supports_compound_text() {
        let (state, object_id) = world_with_named_object("mug");
        let action = parse_action("sequence grasp mug; place mug 1 2 3", &state).unwrap();
        match action {
            Action::Sequence(actions) => {
                assert_eq!(actions.len(), 2);
                match &actions[0] {
                    Action::Grasp { object, .. } => assert_eq!(*object, object_id),
                    _ => panic!("expected Grasp"),
                }
                match &actions[1] {
                    Action::Place { object, target } => {
                        assert_eq!(*object, object_id);
                        assert_eq!(target.x, 1.0);
                    }
                    _ => panic!("expected Place"),
                }
            }
            _ => panic!("expected Sequence"),
        }
    }

    #[test]
    fn test_parse_action_supports_typed_json_for_complex_actions() {
        let (state, object_id) = world_with_named_object("mug");
        let json = format!(
            r#"{{"Conditional":{{"condition":{{"ObjectExists":{{"object":"{object_id}"}}}},"then":{{"Release":{{"object":"{object_id}"}}}},"otherwise":{{"RemoveObject":{{"object":"{object_id}"}}}}}}}}"#
        );

        let action = parse_action(&json, &state).unwrap();
        match action {
            Action::Conditional {
                condition,
                then,
                otherwise,
            } => {
                assert!(matches!(
                    condition,
                    worldforge_core::action::Condition::ObjectExists { object }
                        if object == object_id
                ));
                assert!(matches!(*then, Action::Release { object } if object == object_id));
                assert!(matches!(
                    otherwise.map(|action| *action),
                    Some(Action::RemoveObject { object }) if object == object_id
                ));
            }
            _ => panic!("expected Conditional"),
        }
    }

    #[test]
    fn test_parse_provider_names_trims_and_splits() {
        assert_eq!(
            parse_provider_names(" mock , jepa ,,cosmos "),
            vec!["mock", "jepa", "cosmos"]
        );
    }

    #[test]
    fn test_parse_provider_names_defaults_to_mock() {
        assert_eq!(parse_provider_names(" , "), vec!["mock"]);
    }

    #[test]
    fn test_build_operation_predict() {
        let operation = build_operation(
            EstimateOperationKind::Predict,
            &EstimateOptions {
                operation: EstimateOperationKind::Predict,
                steps: 0,
                duration_seconds: 2.0,
                resolution: (640, 360),
            },
        );

        assert_eq!(
            operation,
            Operation::Predict {
                steps: 1,
                resolution: (640, 360),
            }
        );
    }

    #[test]
    fn test_build_operation_transfer() {
        let operation = build_operation(
            EstimateOperationKind::Transfer,
            &EstimateOptions {
                operation: EstimateOperationKind::Transfer,
                steps: 4,
                duration_seconds: 0.0,
                resolution: (640, 360),
            },
        );

        assert_eq!(
            operation,
            Operation::Transfer {
                duration_seconds: 0.1,
            }
        );
    }

    #[test]
    fn test_cli_parse_serve_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "--state-dir",
            ".wf-test",
            "serve",
            "--bind",
            "127.0.0.1:9000",
        ])
        .unwrap();

        assert_eq!(cli.state_dir, PathBuf::from(".wf-test"));
        assert_eq!(cli.state_backend, StateBackend::File);
        assert_eq!(cli.state_file_format, StateFileFormat::Json);
        match cli.command {
            Commands::Serve { bind } => assert_eq!(bind, "127.0.0.1:9000"),
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn test_cli_parse_msgpack_file_backend() {
        let cli =
            Cli::try_parse_from(["worldforge", "--state-file-format", "msgpack", "list"]).unwrap();

        assert_eq!(cli.state_backend, StateBackend::File);
        assert_eq!(cli.state_file_format, StateFileFormat::Msgpack);
        assert!(matches!(cli.command, Commands::List));
    }

    #[test]
    fn test_cli_parse_create_command_with_name_override() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "create",
            "--prompt",
            "A kitchen with a mug",
            "--name",
            "kitchen-counter",
            "--provider",
            "mock",
        ])
        .unwrap();

        match cli.command {
            Commands::Create {
                prompt,
                name,
                provider,
            } => {
                assert_eq!(prompt, "A kitchen with a mug");
                assert_eq!(name.as_deref(), Some("kitchen-counter"));
                assert_eq!(provider, "mock");
            }
            _ => panic!("expected Create"),
        }
    }

    #[test]
    fn test_cli_parse_sqlite_backend() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "--state-backend",
            "sqlite",
            "--state-db-path",
            "/tmp/worldforge/state.db",
            "list",
        ])
        .unwrap();

        assert_eq!(cli.state_backend, StateBackend::Sqlite);
        assert_eq!(
            cli.state_db_path,
            Some(PathBuf::from("/tmp/worldforge/state.db"))
        );
        assert!(matches!(cli.command, Commands::List));
    }

    #[test]
    fn test_cli_parse_redis_backend() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "--state-backend",
            "redis",
            "--state-redis-url",
            "redis://127.0.0.1:6379/2",
            "list",
        ])
        .unwrap();

        assert_eq!(cli.state_backend, StateBackend::Redis);
        assert_eq!(
            cli.state_redis_url.as_deref(),
            Some("redis://127.0.0.1:6379/2")
        );
        assert!(matches!(cli.command, Commands::List));
    }

    #[test]
    fn test_cli_parse_s3_backend() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "--state-backend",
            "s3",
            "--state-s3-bucket",
            "worldforge-states",
            "--state-s3-region",
            "us-east-1",
            "--state-s3-access-key-id",
            "test-access",
            "--state-s3-secret-access-key",
            "test-secret",
            "--state-s3-endpoint",
            "http://localhost:9000",
            "--state-s3-session-token",
            "test-session",
            "--state-s3-prefix",
            "states",
            "list",
        ])
        .unwrap();

        assert_eq!(cli.state_backend, StateBackend::S3);
        assert_eq!(cli.state_s3_bucket.as_deref(), Some("worldforge-states"));
        assert_eq!(cli.state_s3_region.as_deref(), Some("us-east-1"));
        assert_eq!(cli.state_s3_access_key_id.as_deref(), Some("test-access"));
        assert_eq!(
            cli.state_s3_secret_access_key.as_deref(),
            Some("test-secret")
        );
        assert_eq!(
            cli.state_s3_endpoint.as_deref(),
            Some("http://localhost:9000")
        );
        assert_eq!(cli.state_s3_session_token.as_deref(), Some("test-session"));
        assert_eq!(cli.state_s3_prefix.as_deref(), Some("states"));
        assert!(matches!(cli.command, Commands::List));
    }

    #[test]
    fn test_cli_parse_history_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "history",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--output-json",
            "/tmp/history.json",
        ])
        .unwrap();

        match cli.command {
            Commands::History { world, output_json } => {
                assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(output_json, Some(PathBuf::from("/tmp/history.json")));
            }
            _ => panic!("expected History"),
        }
    }

    #[test]
    fn test_cli_parse_restore_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "restore",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--history-index",
            "2",
            "--output-json",
            "/tmp/restored.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Restore {
                world,
                history_index,
                output_json,
            } => {
                assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(history_index, 2);
                assert_eq!(output_json, Some(PathBuf::from("/tmp/restored.json")));
            }
            _ => panic!("expected Restore"),
        }
    }

    #[test]
    fn test_cli_parse_fork_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "fork",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--history-index",
            "1",
            "--name",
            "branched-world",
        ])
        .unwrap();

        match cli.command {
            Commands::Fork {
                world,
                history_index,
                name,
            } => {
                assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(history_index, Some(1));
                assert_eq!(name.as_deref(), Some("branched-world"));
            }
            _ => panic!("expected Fork"),
        }
    }

    #[test]
    fn test_cli_parse_export_command_infers_format_from_output_path() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "export",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--output",
            "/tmp/world.snapshot.msgpack",
        ])
        .unwrap();

        match cli.command {
            Commands::Export {
                world,
                output,
                format,
            } => {
                assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(output, PathBuf::from("/tmp/world.snapshot.msgpack"));
                assert!(format.is_none());
            }
            _ => panic!("expected Export"),
        }
    }

    #[test]
    fn test_cli_parse_import_command_with_override_and_new_id() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "import",
            "--input",
            "/tmp/world.snapshot.json",
            "--format",
            "msgpack",
            "--new-id",
            "--name",
            "restored-world",
        ])
        .unwrap();

        match cli.command {
            Commands::Import {
                input,
                format,
                new_id,
                name,
            } => {
                assert_eq!(input, PathBuf::from("/tmp/world.snapshot.json"));
                assert_eq!(format, Some(CoreStateFileFormat::MessagePack));
                assert!(new_id);
                assert_eq!(name.as_deref(), Some("restored-world"));
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn test_cli_parse_plan_list_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "plans",
            "list",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--output-json",
            "/tmp/plans.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Plans {
                command: PlanCommands::List { world, output_json },
            } => {
                assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(output_json, Some(PathBuf::from("/tmp/plans.json")));
            }
            _ => panic!("expected Plans::List"),
        }
    }

    #[test]
    fn test_cli_parse_plan_show_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "plans",
            "show",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--plan-id",
            "223e4567-e89b-12d3-a456-426614174000",
            "--output-json",
            "/tmp/plan.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Plans {
                command:
                    PlanCommands::Show {
                        world,
                        plan_id,
                        output_json,
                    },
            } => {
                assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(plan_id, "223e4567-e89b-12d3-a456-426614174000");
                assert_eq!(output_json, Some(PathBuf::from("/tmp/plan.json")));
            }
            _ => panic!("expected Plans::Show"),
        }
    }

    #[test]
    fn test_cli_parse_plan_delete_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "plans",
            "delete",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--plan-id",
            "223e4567-e89b-12d3-a456-426614174000",
        ])
        .unwrap();

        match cli.command {
            Commands::Plans {
                command: PlanCommands::Delete { world, plan_id },
            } => {
                assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(plan_id, "223e4567-e89b-12d3-a456-426614174000");
            }
            _ => panic!("expected Plans::Delete"),
        }
    }

    fn test_state_store_config() -> CliStateStoreConfig<'static> {
        CliStateStoreConfig {
            state_dir: Path::new(".wf"),
            state_backend: StateBackend::File,
            state_file_format: StateFileFormat::Json,
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

    #[test]
    fn test_state_store_kind_defaults_sqlite_path_under_state_dir() {
        let config = CliStateStoreConfig {
            state_backend: StateBackend::Sqlite,
            ..test_state_store_config()
        };

        assert_eq!(
            state_store_kind(&config).unwrap(),
            StateStoreKind::Sqlite(PathBuf::from(".wf/worldforge.db"))
        );
    }

    #[test]
    fn test_state_store_kind_uses_explicit_file_format() {
        let config = CliStateStoreConfig {
            state_file_format: StateFileFormat::Msgpack,
            ..test_state_store_config()
        };

        assert_eq!(
            state_store_kind(&config).unwrap(),
            StateStoreKind::FileWithFormat {
                path: PathBuf::from(".wf"),
                format: CoreStateFileFormat::MessagePack,
            }
        );
    }

    #[test]
    fn test_state_store_kind_uses_redis_url() {
        let config = CliStateStoreConfig {
            state_backend: StateBackend::Redis,
            state_redis_url: Some("redis://127.0.0.1:6379/0"),
            ..test_state_store_config()
        };
        let store_kind = state_store_kind(&config).unwrap();

        assert_eq!(
            store_kind,
            StateStoreKind::Redis("redis://127.0.0.1:6379/0".to_string())
        );
    }

    #[test]
    fn test_state_store_kind_uses_s3_bucket() {
        let config = CliStateStoreConfig {
            state_backend: StateBackend::S3,
            state_s3_bucket: Some("worldforge-states"),
            state_s3_region: Some("us-east-1"),
            state_s3_access_key_id: Some("test-access"),
            state_s3_secret_access_key: Some("test-secret"),
            state_s3_endpoint: Some("http://localhost:9000"),
            state_s3_session_token: Some("test-session"),
            state_s3_prefix: Some("states"),
            ..test_state_store_config()
        };
        let store_kind = state_store_kind(&config).unwrap();

        assert_eq!(
            store_kind,
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
                format: CoreStateFileFormat::Json,
            }
        );
    }

    #[test]
    fn test_open_state_store_requires_s3_bucket() {
        let cli = Cli::try_parse_from(["worldforge", "--state-backend", "s3", "list"]).unwrap();
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(open_state_store(&cli));
        match result {
            Ok(_) => panic!("expected missing S3 bucket error"),
            Err(error) => assert!(error
                .to_string()
                .contains("--state-s3-bucket is required when --state-backend s3 is selected")),
        }
    }

    #[test]
    fn test_open_state_store_requires_redis_url() {
        let cli = Cli::try_parse_from(["worldforge", "--state-backend", "redis", "list"]).unwrap();
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(open_state_store(&cli));
        match result {
            Ok(_) => panic!("expected missing Redis URL error"),
            Err(error) => {
                assert!(error.to_string().contains(
                    "--state-redis-url is required when --state-backend redis is selected"
                ))
            }
        }
    }

    #[tokio::test]
    async fn test_cmd_create_roundtrip_with_redis_state_store() {
        let redis = fake_state_backends::FakeRedisServer::spawn().await;
        let redis_url = redis.url(4);
        let cli = Cli::try_parse_from([
            "worldforge",
            "--state-backend",
            "redis",
            "--state-redis-url",
            &redis_url,
            "list",
        ])
        .unwrap();

        let store = open_state_store(&cli).await.unwrap();
        cmd_create(
            store.as_ref(),
            "A workshop with a crate",
            Some("redis-cli-world"),
            "mock",
        )
        .await
        .unwrap();

        let ids = store.list().await.unwrap();
        assert_eq!(ids.len(), 1);
        let world_id = ids[0].to_string();

        drop(store);

        let reopened = open_state_store(&cli).await.unwrap();
        let loaded = reopened.load(&ids[0]).await.unwrap();
        assert_eq!(loaded.metadata.name, "redis-cli-world");

        cmd_delete(reopened.as_ref(), &world_id).await.unwrap();
        assert!(reopened.list().await.unwrap().is_empty());

        let commands = redis.commands.lock().await;
        assert!(commands
            .iter()
            .any(|command| command == &vec!["SELECT".to_string(), "4".to_string()]));
        assert!(commands
            .iter()
            .any(|command| command.first().map(String::as_str) == Some("SET")));
        assert!(commands
            .iter()
            .any(|command| command.first().map(String::as_str) == Some("GET")));
        assert!(commands
            .iter()
            .any(|command| command.first().map(String::as_str) == Some("DEL")));
    }

    #[tokio::test]
    async fn test_cmd_export_import_roundtrip_with_s3_state_store() {
        let s3 = fake_state_backends::FakeS3Server::spawn().await;
        let s3_config = fake_state_backends::test_s3_config(s3.endpoint());
        let cli = Cli::try_parse_from([
            "worldforge",
            "--state-backend",
            "s3",
            "--state-s3-bucket",
            s3_config.bucket.as_str(),
            "--state-s3-region",
            s3_config.region.as_str(),
            "--state-s3-access-key-id",
            s3_config.access_key_id.as_str(),
            "--state-s3-secret-access-key",
            s3_config.secret_access_key.as_str(),
            "--state-s3-endpoint",
            s3_config.endpoint.as_deref().unwrap(),
            "--state-s3-session-token",
            s3_config.session_token.as_deref().unwrap(),
            "--state-s3-prefix",
            s3_config.prefix.as_str(),
            "list",
        ])
        .unwrap();

        let store = open_state_store(&cli).await.unwrap();
        cmd_create(
            store.as_ref(),
            "A shelf with a blue bin",
            Some("s3-cli-world"),
            "mock",
        )
        .await
        .unwrap();

        let ids = store.list().await.unwrap();
        assert_eq!(ids.len(), 1);
        let world_id = ids[0].to_string();

        let temp_dir = std::env::temp_dir().join(format!("wf-cli-s3-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        let export_path = temp_dir.join("world.json");

        cmd_export(store.as_ref(), &world_id, &export_path, None)
            .await
            .unwrap();
        cmd_delete(store.as_ref(), &world_id).await.unwrap();
        assert!(store.list().await.unwrap().is_empty());

        let imported = cmd_import(
            store.as_ref(),
            &export_path,
            None,
            true,
            Some("s3-imported"),
        )
        .await
        .unwrap();
        let loaded = store.load(&imported.id).await.unwrap();
        assert_eq!(loaded.metadata.name, "s3-imported");

        let requests = s3.requests.lock().await;
        assert!(requests.iter().any(|request| request.method == "PUT"));
        assert!(requests.iter().any(|request| request.method == "GET"));
        assert!(requests.iter().any(|request| request.method == "DELETE"));
        assert!(requests
            .iter()
            .any(|request| request.query.contains("list-type=2")));

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cli_parse_predict_with_fallback() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "predict",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--action",
            "move 1 2 3",
            "--provider",
            "runway",
            "--num-samples",
            "4",
            "--fallback-provider",
            "mock",
            "--timeout-ms",
            "250",
        ])
        .unwrap();

        match cli.command {
            Commands::Predict {
                provider,
                num_samples,
                fallback_provider,
                timeout_ms,
                disable_guardrails,
                ..
            } => {
                assert_eq!(provider, "runway");
                assert_eq!(num_samples, Some(4));
                assert_eq!(fallback_provider.as_deref(), Some("mock"));
                assert_eq!(timeout_ms, Some(250));
                assert!(!disable_guardrails);
            }
            _ => panic!("expected Predict"),
        }
    }

    #[test]
    fn test_build_predict_config_uses_requested_sample_count() {
        let config = build_predict_config(3, Some(8), Some("mock"), Some(120));

        assert_eq!(config.steps, 3);
        assert_eq!(config.num_samples, 8);
        assert_eq!(config.fallback_provider.as_deref(), Some("mock"));
        assert_eq!(config.max_latency_ms, Some(120));
    }

    #[test]
    fn test_predict_sampling_summary_uses_prediction_metadata() {
        assert_eq!(predict_sampling_summary(None), None);

        let sampling = PredictionSamplingMetadata {
            requested_samples: 4,
            completed_samples: 4,
            selected_sample_index: 1,
            confidence_mean: 0.5,
            confidence_stddev: 0.1,
            physics_mean: 0.4,
            physics_stddev: 0.05,
            quality_mean: 0.45,
            quality_stddev: 0.06,
            sample_summaries: Vec::new(),
        };

        assert_eq!(
            predict_sampling_summary(Some(&sampling)).as_deref(),
            Some("Sampling: 4 requested, 4 completed, best sample #2")
        );
    }

    #[test]
    fn test_cli_parse_compare_with_guardrails_and_fallback() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "compare",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--action",
            "move 1 2 3",
            "--providers",
            "runway,cosmos",
            "--steps",
            "4",
            "--fallback-provider",
            "mock",
            "--timeout-ms",
            "300",
            "--guardrails-json",
            "/tmp/guardrails.json",
            "--output-json",
            "/tmp/compare.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Compare {
                world,
                action,
                providers,
                prediction_json,
                steps,
                fallback_provider,
                timeout_ms,
                guardrails_json,
                output_json,
                output_markdown,
                output_csv,
                ..
            } => {
                assert_eq!(
                    world.as_deref(),
                    Some("123e4567-e89b-12d3-a456-426614174000")
                );
                assert_eq!(action.as_deref(), Some("move 1 2 3"));
                assert_eq!(providers.as_deref(), Some("runway,cosmos"));
                assert!(prediction_json.is_empty());
                assert_eq!(steps, 4);
                assert_eq!(fallback_provider.as_deref(), Some("mock"));
                assert_eq!(timeout_ms, Some(300));
                assert_eq!(guardrails_json, Some(PathBuf::from("/tmp/guardrails.json")));
                assert_eq!(output_json, Some(PathBuf::from("/tmp/compare.json")));
                assert!(output_markdown.is_none());
                assert!(output_csv.is_none());
            }
            _ => panic!("expected Compare"),
        }
    }

    #[test]
    fn test_cli_parse_compare_prediction_json_inputs() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "compare",
            "--prediction-json",
            "/tmp/prediction-a.json",
            "--prediction-json",
            "/tmp/prediction-b.json",
            "--output-json",
            "/tmp/compare.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Compare {
                world,
                action,
                providers,
                prediction_json,
                output_json,
                output_markdown,
                output_csv,
                ..
            } => {
                assert!(world.is_none());
                assert!(action.is_none());
                assert!(providers.is_none());
                assert_eq!(
                    prediction_json,
                    vec![
                        PathBuf::from("/tmp/prediction-a.json"),
                        PathBuf::from("/tmp/prediction-b.json")
                    ]
                );
                assert_eq!(output_json, Some(PathBuf::from("/tmp/compare.json")));
                assert!(output_markdown.is_none());
                assert!(output_csv.is_none());
            }
            _ => panic!("expected Compare"),
        }
    }

    #[test]
    fn test_cli_parse_compare_with_world_snapshot() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "compare",
            "--world-snapshot",
            "/tmp/world.msgpack",
            "--action",
            "move 1 2 3",
            "--providers",
            "mock,runway",
        ])
        .unwrap();

        match cli.command {
            Commands::Compare {
                world,
                world_snapshot,
                action,
                providers,
                prediction_json,
                ..
            } => {
                assert!(world.is_none());
                assert_eq!(world_snapshot, Some(PathBuf::from("/tmp/world.msgpack")));
                assert_eq!(action.as_deref(), Some("move 1 2 3"));
                assert_eq!(providers.as_deref(), Some("mock,runway"));
                assert!(prediction_json.is_empty());
            }
            _ => panic!("expected Compare"),
        }
    }

    #[test]
    fn test_cli_parse_generate_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "generate",
            "--prompt",
            "a bouncing sphere",
            "--provider",
            "mock",
            "--fallback-provider",
            "alt-mock",
            "--duration-seconds",
            "5.5",
            "--width",
            "640",
            "--height",
            "360",
            "--fps",
            "12.0",
            "--temperature",
            "0.7",
            "--seed",
            "42",
            "--output-json",
            "/tmp/generated.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Generate {
                provider,
                fallback_provider,
                duration_seconds,
                width,
                height,
                fps,
                temperature,
                seed,
                output_json,
                ..
            } => {
                assert_eq!(provider, "mock");
                assert_eq!(fallback_provider.as_deref(), Some("alt-mock"));
                assert_eq!(duration_seconds, 5.5);
                assert_eq!(width, 640);
                assert_eq!(height, 360);
                assert_eq!(fps, 12.0);
                assert_eq!(temperature, 0.7);
                assert_eq!(seed, Some(42));
                assert_eq!(output_json, Some(PathBuf::from("/tmp/generated.json")));
            }
            _ => panic!("expected Generate"),
        }
    }

    #[test]
    fn test_cli_parse_transfer_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "transfer",
            "--provider",
            "mock",
            "--fallback-provider",
            "alt-mock",
            "--source-json",
            "/tmp/source.json",
            "--controls-json",
            "/tmp/controls.json",
            "--output-json",
            "/tmp/output.json",
            "--width",
            "800",
            "--height",
            "600",
            "--fps",
            "18.0",
            "--control-strength",
            "0.4",
        ])
        .unwrap();

        match cli.command {
            Commands::Transfer {
                provider,
                fallback_provider,
                source_json,
                controls_json,
                output_json,
                width,
                height,
                fps,
                control_strength,
            } => {
                assert_eq!(provider, "mock");
                assert_eq!(fallback_provider.as_deref(), Some("alt-mock"));
                assert_eq!(source_json, PathBuf::from("/tmp/source.json"));
                assert_eq!(controls_json, Some(PathBuf::from("/tmp/controls.json")));
                assert_eq!(output_json, Some(PathBuf::from("/tmp/output.json")));
                assert_eq!(width, 800);
                assert_eq!(height, 600);
                assert_eq!(fps, 18.0);
                assert!((control_strength - 0.4).abs() < f32::EPSILON);
            }
            _ => panic!("expected Transfer"),
        }
    }

    #[test]
    fn test_cli_parse_embed_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "embed",
            "--provider",
            "mock",
            "--fallback-provider",
            "alt-mock",
            "--text",
            "a red mug on a table",
            "--video-json",
            "/tmp/video.json",
            "--output-json",
            "/tmp/embedding.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Embed {
                provider,
                fallback_provider,
                text,
                video_json,
                output_json,
            } => {
                assert_eq!(provider, "mock");
                assert_eq!(fallback_provider.as_deref(), Some("alt-mock"));
                assert_eq!(text.as_deref(), Some("a red mug on a table"));
                assert_eq!(video_json, Some(PathBuf::from("/tmp/video.json")));
                assert_eq!(output_json, Some(PathBuf::from("/tmp/embedding.json")));
            }
            _ => panic!("expected Embed"),
        }
    }

    #[test]
    fn test_cli_parse_reason_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "reason",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--query",
            "will the mug fall?",
            "--provider",
            "cosmos",
            "--fallback-provider",
            "mock",
        ])
        .unwrap();

        match cli.command {
            Commands::Reason {
                world,
                state_json,
                video_json,
                query,
                provider,
                fallback_provider,
                output_json,
            } => {
                assert_eq!(
                    world,
                    Some("123e4567-e89b-12d3-a456-426614174000".to_string())
                );
                assert!(state_json.is_none());
                assert!(video_json.is_none());
                assert_eq!(query, "will the mug fall?");
                assert_eq!(provider.as_deref(), Some("cosmos"));
                assert_eq!(fallback_provider.as_deref(), Some("mock"));
                assert!(output_json.is_none());
            }
            _ => panic!("expected Reason"),
        }
    }

    #[test]
    fn test_cli_parse_reason_command_direct_provider() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "reason",
            "--state-json",
            "/tmp/state.json",
            "--video-json",
            "/tmp/video.json",
            "--query",
            "what do you see?",
            "--provider",
            "mock",
            "--fallback-provider",
            "cosmos",
            "--output-json",
            "/tmp/reason.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Reason {
                world,
                state_json,
                video_json,
                query,
                provider,
                fallback_provider,
                output_json,
            } => {
                assert!(world.is_none());
                assert_eq!(state_json, Some(PathBuf::from("/tmp/state.json")));
                assert_eq!(video_json, Some(PathBuf::from("/tmp/video.json")));
                assert_eq!(query, "what do you see?");
                assert_eq!(provider.as_deref(), Some("mock"));
                assert_eq!(fallback_provider.as_deref(), Some("cosmos"));
                assert_eq!(output_json, Some(PathBuf::from("/tmp/reason.json")));
            }
            _ => panic!("expected Reason"),
        }
    }

    #[test]
    fn test_cli_parse_objects_add_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "objects",
            "add",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--name",
            "crate",
            "--position",
            "0.0",
            "1.0",
            "2.0",
            "--bbox-min",
            "-0.5",
            "-0.5",
            "-0.5",
            "--bbox-max",
            "0.5",
            "0.5",
            "0.5",
            "--velocity",
            "0.1",
            "0.0",
            "0.0",
            "--semantic-label",
            "storage",
            "--mesh-json",
            "/tmp/mesh.json",
            "--visual-embedding-json",
            "/tmp/embedding.json",
            "--static",
            "--graspable",
            "--output-json",
            "/tmp/object.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Objects { command } => match command {
                ObjectCommands::Add {
                    world,
                    name,
                    position,
                    bbox_min,
                    bbox_max,
                    velocity,
                    semantic_label,
                    mesh_json,
                    visual_embedding_json,
                    is_static,
                    graspable,
                    output_json,
                    ..
                } => {
                    assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                    assert_eq!(name, "crate");
                    assert_eq!(position, vec![0.0, 1.0, 2.0]);
                    assert_eq!(bbox_min, vec![-0.5, -0.5, -0.5]);
                    assert_eq!(bbox_max, vec![0.5, 0.5, 0.5]);
                    assert_eq!(velocity, Some(vec![0.1, 0.0, 0.0]));
                    assert_eq!(semantic_label.as_deref(), Some("storage"));
                    assert_eq!(mesh_json, Some(PathBuf::from("/tmp/mesh.json")));
                    assert_eq!(
                        visual_embedding_json,
                        Some(PathBuf::from("/tmp/embedding.json"))
                    );
                    assert!(is_static);
                    assert!(graspable);
                    assert_eq!(output_json, Some(PathBuf::from("/tmp/object.json")));
                }
                _ => panic!("expected Objects::Add"),
            },
            _ => panic!("expected Objects command"),
        }
    }

    #[test]
    fn test_cli_parse_objects_update_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "objects",
            "update",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--object-id",
            "123e4567-e89b-12d3-a456-426614174001",
            "--position",
            "1.0",
            "2.0",
            "3.0",
            "--rotation",
            "0.0",
            "1.0",
            "0.0",
            "0.0",
            "--semantic-label",
            "container",
            "--mesh-json",
            "/tmp/mesh.json",
            "--visual-embedding-json",
            "/tmp/embedding.json",
            "--static",
            "true",
            "--output-json",
            "/tmp/object.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Objects { command } => match command {
                ObjectCommands::Update {
                    world,
                    object_id,
                    position,
                    rotation,
                    semantic_label,
                    mesh_json,
                    visual_embedding_json,
                    is_static,
                    output_json,
                    ..
                } => {
                    assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                    assert_eq!(object_id, "123e4567-e89b-12d3-a456-426614174001");
                    assert_eq!(position, Some(vec![1.0, 2.0, 3.0]));
                    assert_eq!(rotation, Some(vec![0.0, 1.0, 0.0, 0.0]));
                    assert_eq!(semantic_label.as_deref(), Some("container"));
                    assert_eq!(mesh_json, Some(PathBuf::from("/tmp/mesh.json")));
                    assert_eq!(
                        visual_embedding_json,
                        Some(PathBuf::from("/tmp/embedding.json"))
                    );
                    assert_eq!(is_static, Some(true));
                    assert_eq!(output_json, Some(PathBuf::from("/tmp/object.json")));
                }
                _ => panic!("expected Objects::Update"),
            },
            _ => panic!("expected Objects command"),
        }
    }

    #[test]
    fn test_cli_parse_eval_with_custom_suite_and_output() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "eval",
            "--suite-json",
            "/tmp/custom-suite.json",
            "--providers",
            "mock,jepa",
            "--num-samples",
            "4",
            "--output-json",
            "/tmp/eval-report.json",
            "--output-markdown",
            "/tmp/eval-report.md",
            "--output-csv",
            "/tmp/eval-report.csv",
        ])
        .unwrap();

        match cli.command {
            Commands::Eval {
                suite,
                suite_json,
                world,
                world_snapshot,
                providers,
                num_samples,
                list_suites,
                list_metrics,
                output_json,
                output_markdown,
                output_csv,
            } => {
                assert!(suite.is_none());
                assert_eq!(suite_json, Some(PathBuf::from("/tmp/custom-suite.json")));
                assert!(world.is_none());
                assert!(world_snapshot.is_none());
                assert_eq!(providers.as_deref(), Some("mock,jepa"));
                assert_eq!(num_samples, Some(4));
                assert!(!list_suites);
                assert!(!list_metrics);
                assert_eq!(output_json, Some(PathBuf::from("/tmp/eval-report.json")));
                assert_eq!(output_markdown, Some(PathBuf::from("/tmp/eval-report.md")));
                assert_eq!(output_csv, Some(PathBuf::from("/tmp/eval-report.csv")));
            }
            _ => panic!("expected Eval"),
        }
    }

    #[test]
    fn test_cli_parse_eval_with_world_seed() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "eval",
            "--suite",
            "physics",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
        ])
        .unwrap();

        match cli.command {
            Commands::Eval {
                suite,
                world,
                providers,
                list_suites,
                list_metrics,
                output_markdown,
                output_csv,
                ..
            } => {
                assert_eq!(suite.as_deref(), Some("physics"));
                assert_eq!(
                    world.as_deref(),
                    Some("123e4567-e89b-12d3-a456-426614174000")
                );
                assert!(providers.is_none());
                assert!(!list_suites);
                assert!(!list_metrics);
                assert!(output_markdown.is_none());
                assert!(output_csv.is_none());
            }
            _ => panic!("expected Eval"),
        }
    }

    #[test]
    fn test_cli_parse_eval_with_world_snapshot() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "eval",
            "--suite",
            "physics",
            "--world-snapshot",
            "/tmp/eval-world.msgpack",
        ])
        .unwrap();

        match cli.command {
            Commands::Eval {
                suite,
                world,
                world_snapshot,
                ..
            } => {
                assert_eq!(suite.as_deref(), Some("physics"));
                assert!(world.is_none());
                assert_eq!(
                    world_snapshot,
                    Some(PathBuf::from("/tmp/eval-world.msgpack"))
                );
            }
            _ => panic!("expected Eval"),
        }
    }

    #[test]
    fn test_cli_parse_eval_list_metrics() {
        let cli = Cli::try_parse_from(["worldforge", "eval", "--list-metrics"]).unwrap();

        match cli.command {
            Commands::Eval {
                list_suites,
                list_metrics,
                ..
            } => {
                assert!(!list_suites);
                assert!(list_metrics);
            }
            _ => panic!("expected Eval"),
        }
    }

    #[test]
    fn test_available_eval_metric_names_includes_custom_metrics() {
        let metrics = available_eval_metric_names();
        assert!(metrics.contains("latency_efficiency"));
        assert!(metrics.contains("outcome_pass_rate"));
    }

    #[test]
    fn test_cli_parse_providers_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "providers",
            "--capability",
            "planning",
            "--health",
        ])
        .unwrap();

        match cli.command {
            Commands::Providers { capability, health } => {
                assert_eq!(capability.as_deref(), Some("planning"));
                assert!(health);
            }
            _ => panic!("expected Providers"),
        }
    }

    #[test]
    fn test_cli_parse_estimate_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "estimate",
            "--provider",
            "cosmos",
            "--operation",
            "generate",
            "--duration-seconds",
            "5.5",
            "--width",
            "640",
            "--height",
            "360",
        ])
        .unwrap();

        match cli.command {
            Commands::Estimate {
                provider,
                operation,
                duration_seconds,
                width,
                height,
                ..
            } => {
                assert_eq!(provider, "cosmos");
                assert_eq!(operation, EstimateOperationKind::Generate);
                assert_eq!(duration_seconds, 5.5);
                assert_eq!(width, 640);
                assert_eq!(height, 360);
            }
            _ => panic!("expected Estimate"),
        }
    }

    #[test]
    fn test_cli_parse_plan_command_with_guardrails_and_output() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "plan",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--goal",
            "spawn cube",
            "--verify-backend",
            "stark",
            "--guardrails-json",
            "/tmp/guardrails.json",
            "--output-json",
            "/tmp/plan.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Plan {
                verify_backend,
                guardrails_json,
                disable_guardrails,
                output_json,
                ..
            } => {
                assert_eq!(verify_backend, Some(VerifyBackend::Stark));
                assert_eq!(guardrails_json, Some(PathBuf::from("/tmp/guardrails.json")));
                assert!(!disable_guardrails);
                assert_eq!(output_json, Some(PathBuf::from("/tmp/plan.json")));
            }
            _ => panic!("expected Plan"),
        }
    }

    #[test]
    fn test_cli_parse_plan_command_with_goal_json() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "plan",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--goal-json",
            "/tmp/goal.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Plan {
                goal, goal_json, ..
            } => {
                assert!(goal.is_none());
                assert_eq!(goal_json, Some(PathBuf::from("/tmp/goal.json")));
            }
            _ => panic!("expected Plan"),
        }
    }

    #[test]
    fn test_cli_parse_plan_command_with_planner_tuning_flags() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "plan",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--goal",
            "spawn cube",
            "--planner",
            "cem",
            "--population-size",
            "12",
            "--elite-fraction",
            "0.25",
            "--num-iterations",
            "3",
        ])
        .unwrap();

        match cli.command {
            Commands::Plan {
                planner,
                population_size,
                elite_fraction,
                num_iterations,
                ..
            } => {
                assert_eq!(planner, "cem");
                assert_eq!(population_size, Some(12));
                assert_eq!(elite_fraction, Some(0.25));
                assert_eq!(num_iterations, Some(3));
            }
            _ => panic!("expected Plan"),
        }
    }

    #[test]
    fn test_planner_from_name_uses_shared_core_defaults() {
        let planner = planner_from_name("gradient", 10, PlannerOptions::default()).unwrap();
        assert!(matches!(
            planner,
            PlannerType::Gradient {
                learning_rate,
                num_iterations: 24
            } if (learning_rate - 0.25).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn test_cli_parse_execute_plan_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "execute-plan",
            "--world",
            "123e4567-e89b-12d3-a456-426614174000",
            "--plan-json",
            "/tmp/plan.json",
            "--provider",
            "runway",
            "--fallback-provider",
            "mock",
            "--steps",
            "2",
            "--timeout-ms",
            "50",
            "--guardrails-json",
            "/tmp/guardrails.json",
            "--return-video",
            "--disable-guardrails",
            "--output-json",
            "/tmp/execution.json",
        ])
        .unwrap();

        match cli.command {
            Commands::ExecutePlan {
                world,
                plan_json,
                plan_id,
                provider,
                fallback_provider,
                steps,
                timeout_ms,
                guardrails_json,
                return_video,
                disable_guardrails,
                output_json,
            } => {
                assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(plan_json, Some(PathBuf::from("/tmp/plan.json")));
                assert_eq!(plan_id, None);
                assert_eq!(provider.as_deref(), Some("runway"));
                assert_eq!(fallback_provider.as_deref(), Some("mock"));
                assert_eq!(steps, 2);
                assert_eq!(timeout_ms, Some(50));
                assert_eq!(guardrails_json, Some(PathBuf::from("/tmp/guardrails.json")));
                assert!(return_video);
                assert!(disable_guardrails);
                assert_eq!(output_json, Some(PathBuf::from("/tmp/execution.json")));
            }
            _ => panic!("expected ExecutePlan"),
        }
    }

    #[test]
    fn test_cli_parse_verify_command_with_artifacts() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "verify",
            "--proof-type",
            "guardrail",
            "--plan-json",
            "/tmp/plan.json",
            "--output-json",
            "/tmp/proof.json",
            "--source-label",
            "ci",
        ])
        .unwrap();

        match cli.command {
            Commands::Verify {
                world,
                proof_type,
                prediction_json,
                plan_json,
                disable_guardrails,
                output_json,
                source_label,
                ..
            } => {
                assert!(world.is_none());
                assert_eq!(proof_type, "guardrail");
                assert!(prediction_json.is_none());
                assert_eq!(plan_json, Some(PathBuf::from("/tmp/plan.json")));
                assert!(!disable_guardrails);
                assert_eq!(output_json, Some(PathBuf::from("/tmp/proof.json")));
                assert_eq!(source_label, "ci");
            }
            _ => panic!("expected Verify"),
        }
    }

    #[test]
    fn test_cli_parse_verify_command_with_prediction_json() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "verify",
            "--proof-type",
            "inference",
            "--prediction-json",
            "/tmp/prediction.json",
            "--output-json",
            "/tmp/bundle.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Verify {
                prediction_json,
                input_state_json,
                output_state_json,
                output_json,
                ..
            } => {
                assert_eq!(prediction_json, Some(PathBuf::from("/tmp/prediction.json")));
                assert!(input_state_json.is_none());
                assert!(output_state_json.is_none());
                assert_eq!(output_json, Some(PathBuf::from("/tmp/bundle.json")));
            }
            _ => panic!("expected Verify"),
        }
    }

    #[test]
    fn test_cli_parse_verify_proof_command() {
        let cli = Cli::try_parse_from([
            "worldforge",
            "verify-proof",
            "--guardrail-bundle-json",
            "/tmp/bundle.json",
            "--output-json",
            "/tmp/report.json",
        ])
        .unwrap();

        match cli.command {
            Commands::VerifyProof {
                proof_json,
                guardrail_bundle_json,
                output_json,
                ..
            } => {
                assert!(proof_json.is_none());
                assert_eq!(
                    guardrail_bundle_json,
                    Some(PathBuf::from("/tmp/bundle.json"))
                );
                assert_eq!(output_json, Some(PathBuf::from("/tmp/report.json")));
            }
            _ => panic!("expected VerifyProof"),
        }
    }

    #[test]
    fn test_resolve_provider_name_defaults_to_world_provider() {
        let state = WorldState::new("default-provider", "mock");
        assert_eq!(resolve_provider_name(&state, None), "mock");
        assert_eq!(resolve_provider_name(&state, Some("runway")), "runway");
    }

    #[tokio::test]
    async fn test_cmd_generate_writes_output_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-generate-{}", uuid::Uuid::new_v4()));
        let output = dir.join("clip.json");

        cmd_generate(
            "a bouncing sphere",
            "mock",
            GenerateOptions {
                fallback_provider: None,
                negative_prompt: None,
                duration_seconds: 2.5,
                resolution: (640, 360),
                fps: 12.0,
                temperature: 1.0,
                seed: Some(7),
                output_json: Some(&output),
            },
        )
        .await
        .unwrap();

        let clip: VideoClip = read_json_file(&output).unwrap();
        assert_eq!(clip.duration, 2.5);
        assert_eq!(clip.resolution, (640, 360));
        assert_eq!(clip.fps, 12.0);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_generate_uses_fallback_provider() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-generate-fallback-{}", uuid::Uuid::new_v4()));
        let output = dir.join("clip.json");

        cmd_generate(
            "a bouncing sphere",
            "missing",
            GenerateOptions {
                fallback_provider: Some("mock"),
                negative_prompt: None,
                duration_seconds: 1.5,
                resolution: (320, 180),
                fps: 8.0,
                temperature: 1.0,
                seed: None,
                output_json: Some(&output),
            },
        )
        .await
        .unwrap();

        let clip: VideoClip = read_json_file(&output).unwrap();
        assert_eq!(clip.duration, 1.5);
        assert_eq!(clip.resolution, (320, 180));

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_compare_prediction_json_writes_output_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-compare-{}", uuid::Uuid::new_v4()));
        let prediction_a = dir.join("prediction-a.json");
        let prediction_b = dir.join("prediction-b.json");
        let output = dir.join("compare.json");
        fs::create_dir_all(&dir).unwrap();
        write_json_file(&prediction_a, &sample_prediction("mock", 0.25, 120)).unwrap();
        write_json_file(&prediction_b, &sample_prediction("mock-2", 0.35, 180)).unwrap();

        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let prediction_paths = [prediction_a.clone(), prediction_b.clone()];

        cmd_compare(
            store.as_ref(),
            None,
            None,
            None,
            None,
            CompareOptions {
                prediction_json: &prediction_paths,
                steps: 1,
                fallback_provider: None,
                timeout_ms: None,
                guardrails_json: None,
                disable_guardrails: false,
                output_json: Some(&output),
                output_markdown: None,
                output_csv: None,
            },
        )
        .await
        .unwrap();

        let comparison: MultiPrediction = read_json_file(&output).unwrap();
        assert_eq!(comparison.predictions.len(), 2);
        assert_eq!(comparison.comparison.scores.len(), 2);
        assert_eq!(comparison.comparison.pairwise_agreements.len(), 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_compare_world_snapshot_writes_output_json() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-compare-snapshot-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let snapshot = dir.join("world.msgpack");
        let output = dir.join("compare.json");
        let state = WorldState::new("compare-snapshot", "mock");
        write_world_state_snapshot(&snapshot, &state, CoreStateFileFormat::MessagePack).unwrap();

        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();

        cmd_compare(
            store.as_ref(),
            None,
            Some(&snapshot),
            Some("move 1 2 3"),
            Some("mock,mock"),
            CompareOptions {
                prediction_json: &[],
                steps: 1,
                fallback_provider: None,
                timeout_ms: None,
                guardrails_json: None,
                disable_guardrails: false,
                output_json: Some(&output),
                output_markdown: None,
                output_csv: None,
            },
        )
        .await
        .unwrap();

        let comparison: MultiPrediction = read_json_file(&output).unwrap();
        assert_eq!(comparison.predictions.len(), 2);
        assert_eq!(comparison.comparison.pairwise_agreements.len(), 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_compare_prediction_json_writes_markdown_and_csv() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-compare-artifacts-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let prediction_a = dir.join("prediction-a.json");
        let prediction_b = dir.join("prediction-b.json");
        let markdown = dir.join("compare.md");
        let csv = dir.join("compare.csv");
        write_json_file(&prediction_a, &sample_prediction("mock", 0.25, 120)).unwrap();
        write_json_file(&prediction_b, &sample_prediction("mock-2", 0.35, 180)).unwrap();

        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let prediction_paths = [prediction_a, prediction_b];

        cmd_compare(
            store.as_ref(),
            None,
            None,
            None,
            None,
            CompareOptions {
                prediction_json: &prediction_paths,
                steps: 1,
                fallback_provider: None,
                timeout_ms: None,
                guardrails_json: None,
                disable_guardrails: false,
                output_json: None,
                output_markdown: Some(&markdown),
                output_csv: Some(&csv),
            },
        )
        .await
        .unwrap();

        let markdown_content = fs::read_to_string(markdown).unwrap();
        assert!(markdown_content.contains("# Multi-Provider Comparison"));
        assert!(markdown_content.contains("## Provider Scores"));

        let csv_content = fs::read_to_string(csv).unwrap();
        assert!(csv_content.contains("provider,is_best,rank,quality_score"));
        assert!(csv_content.contains("mock"));
        assert!(csv_content.contains("mock-2"));

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_eval_loads_custom_suite_and_writes_output_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-eval-{}", uuid::Uuid::new_v4()));
        let suite_path = dir.join("suite.json");
        let report_path = dir.join("report.json");
        let suite = EvalSuite::physics_standard();
        write_json_file(&suite_path, &suite).unwrap();

        cmd_eval(
            None,
            EvalOptions {
                suite_name: None,
                suite_json: Some(&suite_path),
                world: None,
                world_snapshot: None,
                providers: None,
                num_samples: Some(4),
                list_suites: false,
                list_metrics: false,
                output_json: Some(&report_path),
                output_markdown: None,
                output_csv: None,
            },
        )
        .await
        .unwrap();

        let report: serde_json::Value = read_json_file(&report_path).unwrap();
        assert_eq!(report["suite"], "Physics Standard");
        assert_eq!(report["leaderboard"][0]["provider"], "mock");
        assert_eq!(report["provider_summaries"][0]["provider"], "mock");
        assert_eq!(
            report["provider_summaries"][0]["sampling"]["requested_samples"],
            8
        );
        assert_eq!(
            report["dimension_summaries"][0]["dimension"],
            "object_permanence"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_eval_uses_world_state_when_provided() {
        let dir = std::env::temp_dir().join(format!("wf-cli-eval-world-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();

        let mut world_state = WorldState::new("eval-world", "mock");
        let mug = SceneObject::new(
            "mug",
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
        );
        world_state.scene.add_object(mug);
        store.save(&world_state).await.unwrap();

        let suite_path = dir.join("suite.json");
        let report_path = dir.join("report.json");
        let suite = EvalSuite {
            name: "World-aware eval".to_string(),
            scenarios: vec![worldforge_eval::EvalScenario {
                name: "world-seeded-object-check".to_string(),
                description: "Checks that the persisted world state seeds evaluation".to_string(),
                initial_state: {
                    let mut state = WorldState::new("fixture", "mock");
                    let cube = SceneObject::new(
                        "cube",
                        Pose {
                            position: Position {
                                x: 1.0,
                                y: 0.8,
                                z: 0.0,
                            },
                            ..Pose::default()
                        },
                        BBox {
                            min: Position {
                                x: 0.95,
                                y: 0.75,
                                z: -0.05,
                            },
                            max: Position {
                                x: 1.05,
                                y: 0.85,
                                z: 0.05,
                            },
                        },
                    );
                    state.scene.add_object(cube);
                    state
                },
                actions: Vec::new(),
                expected_outcomes: vec![
                    worldforge_eval::ExpectedOutcome::ObjectExists {
                        name: "cube".to_string(),
                    },
                    worldforge_eval::ExpectedOutcome::ObjectExists {
                        name: "mug".to_string(),
                    },
                ],
                ground_truth: None,
            }],
            dimensions: vec![worldforge_eval::EvalDimension::ObjectPermanence],
            providers: vec!["mock".to_string()],
        };
        write_json_file(&suite_path, &suite).unwrap();

        cmd_eval(
            Some(store.as_ref()),
            EvalOptions {
                suite_name: None,
                suite_json: Some(&suite_path),
                world: Some(&world_state.id.to_string()),
                world_snapshot: None,
                providers: None,
                num_samples: None,
                list_suites: false,
                list_metrics: false,
                output_json: Some(&report_path),
                output_markdown: None,
                output_csv: None,
            },
        )
        .await
        .unwrap();

        let report: serde_json::Value = read_json_file(&report_path).unwrap();
        assert_eq!(report["suite"], "World-aware eval");
        assert_eq!(report["results"][0]["outcomes"][0]["passed"], true);
        assert_eq!(report["results"][0]["outcomes"][1]["passed"], true);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_eval_uses_world_snapshot_when_provided() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-eval-snapshot-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();

        let mut world_state = WorldState::new("eval-world-snapshot", "mock");
        world_state.scene.add_object(SceneObject::new(
            "mug",
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

        let snapshot = dir.join("world.json");
        write_world_state_snapshot(&snapshot, &world_state, CoreStateFileFormat::Json).unwrap();

        let suite_path = dir.join("suite.json");
        let report_path = dir.join("report.json");
        let suite = EvalSuite {
            name: "Snapshot-aware eval".to_string(),
            scenarios: vec![worldforge_eval::EvalScenario {
                name: "snapshot-seeded-object-check".to_string(),
                description: "Checks that an exported snapshot seeds evaluation".to_string(),
                initial_state: WorldState::new("fixture", "mock"),
                actions: Vec::new(),
                expected_outcomes: vec![worldforge_eval::ExpectedOutcome::ObjectExists {
                    name: "mug".to_string(),
                }],
                ground_truth: None,
            }],
            dimensions: vec![worldforge_eval::EvalDimension::ObjectPermanence],
            providers: vec!["mock".to_string()],
        };
        write_json_file(&suite_path, &suite).unwrap();

        cmd_eval(
            None,
            EvalOptions {
                suite_name: None,
                suite_json: Some(&suite_path),
                world: None,
                world_snapshot: Some(&snapshot),
                providers: None,
                num_samples: None,
                list_suites: false,
                list_metrics: false,
                output_json: Some(&report_path),
                output_markdown: None,
                output_csv: None,
            },
        )
        .await
        .unwrap();

        let report: serde_json::Value = read_json_file(&report_path).unwrap();
        assert_eq!(report["suite"], "Snapshot-aware eval");
        assert_eq!(report["results"][0]["outcomes"][0]["passed"], true);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_eval_writes_markdown_and_csv_reports() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-eval-artifacts-{}", uuid::Uuid::new_v4()));
        let markdown_path = dir.join("report.md");
        let csv_path = dir.join("report.csv");

        cmd_eval(
            None,
            EvalOptions {
                suite_name: Some("physics"),
                suite_json: None,
                world: None,
                world_snapshot: None,
                providers: None,
                num_samples: None,
                list_suites: false,
                list_metrics: false,
                output_json: None,
                output_markdown: Some(&markdown_path),
                output_csv: Some(&csv_path),
            },
        )
        .await
        .unwrap();

        let markdown = fs::read_to_string(&markdown_path).unwrap();
        let csv = fs::read_to_string(&csv_path).unwrap();
        assert!(markdown.contains("# Evaluation Report: Physics Standard"));
        assert!(markdown.contains("## Leaderboard"));
        assert!(csv
            .lines()
            .next()
            .unwrap()
            .contains("suite,provider,scenario"));
        assert!(csv.contains("Physics Standard,mock,object_drop"));

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_transfer_roundtrips_clip_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-transfer-{}", uuid::Uuid::new_v4()));
        let source_path = dir.join("source.json");
        let output_path = dir.join("output.json");
        let source = VideoClip {
            frames: Vec::new(),
            fps: 10.0,
            resolution: (320, 180),
            duration: 3.0,
        };
        write_json_file(&source_path, &source).unwrap();

        cmd_transfer(
            "mock",
            TransferOptions {
                fallback_provider: None,
                source_json: &source_path,
                controls_json: None,
                output_json: Some(&output_path),
                resolution: (800, 600),
                fps: 24.0,
                control_strength: 0.5,
            },
        )
        .await
        .unwrap();

        let clip: VideoClip = read_json_file(&output_path).unwrap();
        assert_eq!(clip.duration, source.duration);
        assert_eq!(clip.resolution, (800, 600));
        assert_eq!(clip.fps, 24.0);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_embed_writes_output_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-embed-{}", uuid::Uuid::new_v4()));
        let video_path = dir.join("video.json");
        let output_path = dir.join("output.json");
        let clip = VideoClip {
            frames: Vec::new(),
            fps: 8.0,
            resolution: (64, 64),
            duration: 1.0,
        };
        write_json_file(&video_path, &clip).unwrap();

        cmd_embed(
            "mock",
            EmbedOptions {
                fallback_provider: None,
                text: Some("a red mug on a table"),
                video_json: Some(&video_path),
                output_json: Some(&output_path),
            },
        )
        .await
        .unwrap();

        let value: serde_json::Value = read_json_file(&output_path).unwrap();
        assert_eq!(value["provider"], "mock");
        assert_eq!(value["model"], "mock-embedding-v1");
        assert_eq!(value["embedding"]["shape"], serde_json::json!([32]));

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_reason_direct_provider_writes_output_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-reason-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let video_path = dir.join("video.json");
        let output_path = dir.join("output.json");
        let clip = VideoClip {
            frames: Vec::new(),
            fps: 12.0,
            resolution: (320, 180),
            duration: 1.5,
        };
        write_json_file(&video_path, &clip).unwrap();

        cmd_reason(
            store.as_ref(),
            ReasonOptions {
                world: None,
                state_json: None,
                video_json: Some(&video_path),
                query: "what do you see?",
                provider: Some("mock"),
                fallback_provider: None,
                output_json: Some(&output_path),
            },
        )
        .await
        .unwrap();

        let value: serde_json::Value = read_json_file(&output_path).unwrap();
        assert_eq!(value["provider"], "mock");
        assert!(value["reasoning"]["answer"]
            .as_str()
            .unwrap()
            .contains("echo the query"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_cmd_jepa_embed_and_reason_via_auto_detect() {
        let model_dir = TestJepaModelDir::new("auto-detect");
        model_dir.write_assets(1536);
        let model_dir_string = model_dir.path.to_string_lossy().to_string();

        with_env_vars(
            &[
                ("JEPA_MODEL_PATH", Some(model_dir_string.as_str())),
                ("JEPA_BACKEND", Some("burn")),
            ],
            || {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async {
                    let dir = std::env::temp_dir()
                        .join(format!("wf-cli-jepa-cmd-{}", uuid::Uuid::new_v4()));
                    let state_path = dir.join("state.json");
                    let embed_output = dir.join("embed.json");
                    let reason_output = dir.join("reason.json");
                    fs::create_dir_all(&dir).unwrap();

                    let store = StateStoreKind::File(dir.join("store"))
                        .open()
                        .await
                        .unwrap();
                    let (mut state, object_id) = world_with_named_object("block");
                    let object = state.scene.objects.get_mut(&object_id).unwrap();
                    object.pose.position = Position {
                        x: 0.45,
                        y: 0.25,
                        z: -0.10,
                    };
                    write_json_file(&state_path, &state).unwrap();

                    cmd_embed(
                        "jepa",
                        EmbedOptions {
                            fallback_provider: None,
                            text: Some("stack the block on the shelf"),
                            video_json: None,
                            output_json: Some(&embed_output),
                        },
                    )
                    .await
                    .unwrap();

                    let embedding: serde_json::Value = read_json_file(&embed_output).unwrap();
                    assert_eq!(embedding["provider"], "jepa");
                    assert_eq!(embedding["embedding"]["shape"], serde_json::json!([1536]));

                    cmd_reason(
                        store.as_ref(),
                        ReasonOptions {
                            world: None,
                            state_json: Some(&state_path),
                            video_json: None,
                            query: "Where is the block?",
                            provider: Some("jepa"),
                            fallback_provider: None,
                            output_json: Some(&reason_output),
                        },
                    )
                    .await
                    .unwrap();

                    let reasoning: serde_json::Value = read_json_file(&reason_output).unwrap();
                    assert_eq!(reasoning["provider"], "jepa");
                    assert!(reasoning["reasoning"]["answer"]
                        .as_str()
                        .unwrap()
                        .contains("block"));
                    assert!(reasoning["reasoning"]["evidence"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|entry| {
                            entry
                                .as_str()
                                .is_some_and(|value| value.starts_with("position:block="))
                        }));

                    let _ = fs::remove_dir_all(&dir);
                });
            },
        );
    }

    #[tokio::test]
    async fn test_cmd_reason_requires_provider_without_world() {
        let dir = std::env::temp_dir().join(format!(
            "wf-cli-reason-missing-provider-{}",
            uuid::Uuid::new_v4()
        ));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let video_path = dir.join("video.json");
        let clip = VideoClip {
            frames: Vec::new(),
            fps: 12.0,
            resolution: (320, 180),
            duration: 1.5,
        };
        write_json_file(&video_path, &clip).unwrap();

        let error = cmd_reason(
            store.as_ref(),
            ReasonOptions {
                world: None,
                state_json: None,
                video_json: Some(&video_path),
                query: "what do you see?",
                provider: None,
                fallback_provider: None,
                output_json: None,
            },
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("requires --provider"));

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_plan_writes_output_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-plan-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("plan-output", "mock");
        store.save(&state).await.unwrap();

        let guardrails_path = dir.join("guardrails.json");
        let plan_path = dir.join("plan.json");
        write_json_file(
            &guardrails_path,
            &vec![GuardrailConfig {
                guardrail: worldforge_core::guardrail::Guardrail::MaxVelocity { limit: 100.0 },
                blocking: true,
            }],
        )
        .unwrap();

        cmd_plan(
            store.as_ref(),
            &state.id.to_string(),
            Some("spawn cube"),
            PlanOptions {
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: "mock",
                fallback_provider: None,
                verify_backend: Some(VerificationBackend::Mock),
                goal_json: None,
                guardrails_json: Some(&guardrails_path),
                disable_guardrails: false,
                output_json: Some(&plan_path),
            },
        )
        .await
        .unwrap();

        let plan: worldforge_core::prediction::Plan = read_json_file(&plan_path).unwrap();
        assert!(!plan.actions.is_empty());
        assert!(plan.stored_plan_id.is_some());
        assert_eq!(
            plan.verification_proof
                .as_ref()
                .map(|proof| proof.backend.as_str()),
            Some("mock")
        );
        let persisted = store.load(&state.id).await.unwrap();
        assert_eq!(persisted.stored_plans.len(), 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_execute_plan_uses_stored_plan_id() {
        let dir = std::env::temp_dir().join(format!("wf-cli-execute-id-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("execute-plan-id", "mock");
        store.save(&state).await.unwrap();

        cmd_plan(
            store.as_ref(),
            &state.id.to_string(),
            Some("spawn cube"),
            PlanOptions {
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: "mock",
                fallback_provider: None,
                verify_backend: None,
                goal_json: None,
                guardrails_json: None,
                disable_guardrails: false,
                output_json: None,
            },
        )
        .await
        .unwrap();

        let persisted = store.load(&state.id).await.unwrap();
        let plan_id = persisted.stored_plans.keys().next().unwrap().to_string();

        cmd_execute_plan(
            store.as_ref(),
            &state.id.to_string(),
            ExecutePlanOptions {
                plan_json: None,
                plan_id: Some(&plan_id),
                provider: None,
                fallback_provider: None,
                steps: 1,
                timeout_ms: None,
                guardrails_json: None,
                return_video: false,
                disable_guardrails: false,
                output_json: None,
            },
        )
        .await
        .unwrap();

        let updated = store.load(&state.id).await.unwrap();
        assert!(updated.time.step >= 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_plans_list_show_delete_roundtrip() {
        let dir = std::env::temp_dir().join(format!("wf-cli-plans-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("stored-plans-world", "mock");
        store.save(&state).await.unwrap();

        let plan_id = uuid::Uuid::new_v4();
        let mut persisted = store.load(&state.id).await.unwrap();
        let record = sample_stored_plan_record(plan_id);
        persisted.store_plan_record(record.clone());
        store.save(&persisted).await.unwrap();

        let list_path = dir.join("plans.json");
        cmd_plans_list(
            store.as_ref(),
            &state.id.to_string(),
            PlanOutputOptions {
                output_json: Some(&list_path),
            },
        )
        .await
        .unwrap();

        let listed: Vec<StoredPlanRecord> = read_json_file(&list_path).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, plan_id);

        let show_path = dir.join("plan.json");
        cmd_plans_show(
            store.as_ref(),
            &state.id.to_string(),
            &plan_id.to_string(),
            PlanOutputOptions {
                output_json: Some(&show_path),
            },
        )
        .await
        .unwrap();

        let shown: StoredPlanRecord = read_json_file(&show_path).unwrap();
        assert_eq!(shown.id, plan_id);
        assert_eq!(shown.goal_summary, "spawn cube");

        cmd_plans_delete(store.as_ref(), &state.id.to_string(), &plan_id.to_string())
            .await
            .unwrap();

        let reloaded = store.load(&state.id).await.unwrap();
        assert!(reloaded.stored_plans.is_empty());

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_plan_reads_structured_goal_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-plan-json-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let mut state = WorldState::new("plan-structured", "mock");
        let object = SceneObject::new(
            "ball",
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
        store.save(&state).await.unwrap();

        let goal_path = dir.join("goal.json");
        let plan_path = dir.join("plan.json");
        write_json_file(
            &goal_path,
            &serde_json::json!({
                "type": "condition",
                "condition": {
                    "ObjectAt": {
                        "object": object_id,
                        "position": {"x": 1.0, "y": 0.5, "z": 0.0},
                        "tolerance": 0.05
                    }
                }
            }),
        )
        .unwrap();

        cmd_plan(
            store.as_ref(),
            &state.id.to_string(),
            None,
            PlanOptions {
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: "mock",
                fallback_provider: None,
                verify_backend: None,
                goal_json: Some(&goal_path),
                guardrails_json: None,
                disable_guardrails: false,
                output_json: Some(&plan_path),
            },
        )
        .await
        .unwrap();

        let plan: worldforge_core::prediction::Plan = read_json_file(&plan_path).unwrap();
        assert!(!plan.actions.is_empty());
        let final_state = plan.predicted_states.last().unwrap();
        let moved = final_state.scene.get_object(&object_id).unwrap();
        assert!(
            moved.pose.position.distance(Position {
                x: 1.0,
                y: 0.5,
                z: 0.0,
            }) <= 0.15
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_plan_reads_goal_image_json() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-plan-goal-image-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let mut state = WorldState::new("plan-goal-image", "mock");
        let object = SceneObject::new(
            "ball",
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
        store.save(&state).await.unwrap();

        let mut target_state = state.clone();
        target_state
            .scene
            .get_object_mut(&object_id)
            .unwrap()
            .set_position(Position {
                x: 1.0,
                y: 0.5,
                z: 0.0,
            });

        let goal_path = dir.join("goal-image.json");
        let plan_path = dir.join("plan.json");
        write_json_file(
            &goal_path,
            &serde_json::json!({
                "type": "goal_image",
                "image": worldforge_core::goal_image::render_scene_goal_image(&target_state, (32, 24))
            }),
        )
        .unwrap();

        cmd_plan(
            store.as_ref(),
            &state.id.to_string(),
            None,
            PlanOptions {
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: "mock",
                fallback_provider: None,
                verify_backend: None,
                goal_json: Some(&goal_path),
                guardrails_json: None,
                disable_guardrails: false,
                output_json: Some(&plan_path),
            },
        )
        .await
        .unwrap();

        let plan: worldforge_core::prediction::Plan = read_json_file(&plan_path).unwrap();
        assert!(!plan.actions.is_empty());
        let final_state = plan.predicted_states.last().unwrap();
        let moved = final_state.scene.get_object(&object_id).unwrap();
        assert!(moved.pose.position.x > 0.5);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_plan_provider_native_uses_fallback_provider() {
        let dir = std::env::temp_dir().join(format!("wf-cli-plan-native-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("plan-native-output", "mock");
        store.save(&state).await.unwrap();
        let plan_path = dir.join("plan-native.json");

        let supports_native = auto_detect_registry()
            .describe("mock")
            .unwrap()
            .capabilities
            .supports_planning;

        let result = cmd_plan(
            store.as_ref(),
            &state.id.to_string(),
            Some("spawn cube"),
            PlanOptions {
                max_steps: 4,
                planner_name: "provider-native",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: "missing",
                fallback_provider: Some("mock"),
                verify_backend: None,
                goal_json: None,
                guardrails_json: None,
                disable_guardrails: false,
                output_json: Some(&plan_path),
            },
        )
        .await;

        if supports_native {
            result.unwrap();
            let plan: worldforge_core::prediction::Plan = read_json_file(&plan_path).unwrap();
            assert!(!plan.actions.is_empty());
            assert_eq!(plan.actions.len(), plan.predicted_states.len());
        } else {
            let error = result.unwrap_err().to_string().to_lowercase();
            assert!(error.contains("native planning") || error.contains("unsupported"));
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_execute_plan_writes_output_json_and_persists_state() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-execute-plan-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let mut state = WorldState::new("execute-plan", "mock");
        let object = SceneObject::new(
            "ball",
            Pose::default(),
            BBox::from_center_half_extents(
                Position::default(),
                Vec3 {
                    x: 0.1,
                    y: 0.1,
                    z: 0.1,
                },
            ),
        );
        let object_id = object.id;
        state.scene.add_object(object);
        store.save(&state).await.unwrap();

        let plan = Plan {
            actions: vec![Action::Move {
                target: Position {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                speed: 1.0,
            }],
            predicted_states: Vec::new(),
            predicted_videos: None,
            total_cost: 0.0,
            success_probability: 1.0,
            guardrail_compliance: Vec::new(),
            planning_time_ms: 0,
            iterations_used: 1,
            stored_plan_id: None,
            verification_proof: None,
        };
        let plan_path = dir.join("plan.json");
        let execution_path = dir.join("execution.json");
        write_json_file(&plan_path, &plan).unwrap();

        cmd_execute_plan(
            store.as_ref(),
            &state.id.to_string(),
            ExecutePlanOptions {
                plan_json: Some(&plan_path),
                plan_id: None,
                provider: None,
                fallback_provider: None,
                steps: 1,
                timeout_ms: None,
                guardrails_json: None,
                return_video: false,
                disable_guardrails: false,
                output_json: Some(&execution_path),
            },
        )
        .await
        .unwrap();

        let report: PlanExecution = read_json_file(&execution_path).unwrap();
        assert_eq!(report.predictions.len(), 1);
        assert_eq!(report.final_state.time.step, 1);

        let persisted = store.load(&state.id).await.unwrap();
        assert_eq!(persisted.time.step, 1);
        assert_eq!(persisted.history.len(), 2);
        assert!(
            persisted
                .scene
                .get_object(&object_id)
                .unwrap()
                .pose
                .position
                .x
                > 0.5
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_predict_disable_guardrails_allows_colliding_scene() {
        let dir = std::env::temp_dir().join(format!("wf-cli-predict-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let mut state = WorldState::new("predict-guardrails", "mock");
        state.scene.add_object(SceneObject::new(
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
        state.scene.add_object(SceneObject::new(
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
        store.save(&state).await.unwrap();

        let persisted_before = store.load(&state.id).await.unwrap();
        assert_eq!(persisted_before.history.len(), 1);
        let initial_entry = persisted_before.history.latest().unwrap();
        assert!(initial_entry.action.is_none());
        assert!(initial_entry.prediction.is_none());

        let result = cmd_predict(
            store.as_ref(),
            &state.id.to_string(),
            "set-weather rain",
            1,
            "mock",
            None,
            None,
            None,
            false,
        )
        .await;
        let error = result.unwrap_err().to_string();
        assert!(error.contains("NoCollisions"));
        assert!(error.contains("collision between"));
        let after_failed = store.load(&state.id).await.unwrap();
        assert_eq!(after_failed.time, persisted_before.time);
        assert_eq!(after_failed.history.len(), persisted_before.history.len());
        let failed_entry = after_failed.history.latest().unwrap();
        assert!(failed_entry.action.is_none());
        assert!(failed_entry.prediction.is_none());

        cmd_predict(
            store.as_ref(),
            &state.id.to_string(),
            "set-weather rain",
            1,
            "mock",
            None,
            None,
            None,
            true,
        )
        .await
        .unwrap();

        let updated = store.load(&state.id).await.unwrap();
        assert_eq!(updated.time.step, 1);
        assert!(updated.history.len() > after_failed.history.len());
        let transition = updated.history.latest().unwrap();
        assert_eq!(transition.provider, "mock");
        assert!(transition.prediction.is_some());
        assert!(transition.snapshot.is_some());
        assert!(matches!(
            transition.action,
            Some(worldforge_core::action::Action::SetWeather {
                weather: worldforge_core::action::Weather::Rain
            })
        ));

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_history_writes_output_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-history-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("history", "mock");
        let world_id = state.id.to_string();
        store.save(&state).await.unwrap();

        cmd_predict(
            store.as_ref(),
            &world_id,
            "set-weather rain",
            1,
            "mock",
            None,
            None,
            None,
            true,
        )
        .await
        .unwrap();

        let history_path = dir.join("history.json");
        cmd_history(store.as_ref(), &world_id, Some(&history_path))
            .await
            .unwrap();

        let value: serde_json::Value = read_json_file(&history_path).unwrap();
        let entries = value.as_array().unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[0]["action"].is_null());
        assert_eq!(entries[1]["provider"], "mock");

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_restore_rewinds_persisted_world() {
        let dir = std::env::temp_dir().join(format!("wf-cli-restore-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("restore", "mock");
        let world_id = state.id.to_string();
        store.save(&state).await.unwrap();

        cmd_predict(
            store.as_ref(),
            &world_id,
            "set-weather rain",
            1,
            "mock",
            None,
            None,
            None,
            true,
        )
        .await
        .unwrap();

        let restored_path = dir.join("restored.json");
        cmd_restore(store.as_ref(), &world_id, 0, Some(&restored_path))
            .await
            .unwrap();

        let restored: WorldState = read_json_file(&restored_path).unwrap();
        assert_eq!(restored.time.step, 0);
        assert_eq!(restored.history.len(), 1);

        let persisted = store.load(&state.id).await.unwrap();
        assert_eq!(persisted.time.step, 0);
        assert_eq!(persisted.history.len(), 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_objects_roundtrip() {
        let dir = std::env::temp_dir().join(format!("wf-cli-objects-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("objects", "mock");
        let world_id = state.id.to_string();
        store.save(&state).await.unwrap();

        let created_path = dir.join("created.json");
        let updated_path = dir.join("updated.json");
        let list_path = dir.join("objects.json");
        let shown_path = dir.join("shown.json");
        let mesh_path = dir.join("mesh.json");
        let embedding_path = dir.join("embedding.json");
        let updated_mesh_path = dir.join("mesh-updated.json");
        let updated_embedding_path = dir.join("embedding-updated.json");

        let mesh = Mesh {
            vertices: vec![
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Position {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ],
            faces: vec![[0, 1, 2]],
            normals: Some(vec![
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            ]),
            uvs: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        };
        let embedding = Tensor {
            data: TensorData::Float32(vec![0.1, 0.2, 0.3, 0.4]),
            shape: vec![4],
            dtype: DType::Float32,
            device: Device::Cpu,
        };
        let updated_mesh = Mesh {
            vertices: vec![
                Position {
                    x: -1.0,
                    y: 0.0,
                    z: 0.0,
                },
                Position {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            ],
            faces: vec![[0, 1, 2]],
            normals: None,
            uvs: None,
        };
        let updated_embedding = Tensor {
            data: TensorData::Float32(vec![1.0, 2.0]),
            shape: vec![2],
            dtype: DType::Float32,
            device: Device::Cpu,
        };
        write_json_file(&mesh_path, &mesh).unwrap();
        write_json_file(&embedding_path, &embedding).unwrap();
        write_json_file(&updated_mesh_path, &updated_mesh).unwrap();
        write_json_file(&updated_embedding_path, &updated_embedding).unwrap();

        cmd_objects_add(
            store.as_ref(),
            &world_id,
            "crate",
            ObjectAddOptions {
                position: &[0.0, 1.0, 2.0],
                bbox_min: &[-0.5, -0.5, -0.5],
                bbox_max: &[0.5, 0.5, 0.5],
                velocity: Some(&[0.1, 0.0, 0.0]),
                semantic_label: Some("storage"),
                mesh_json: Some(&mesh_path),
                visual_embedding_json: Some(&embedding_path),
                mass: Some(5.0),
                friction: Some(0.3),
                restitution: Some(0.1),
                material: Some("wood"),
                is_static: true,
                graspable: true,
                output_json: Some(&created_path),
            },
        )
        .await
        .unwrap();

        let created: SceneObject = read_json_file(&created_path).unwrap();
        assert_eq!(created.name, "crate");
        assert_eq!(created.semantic_label.as_deref(), Some("storage"));
        assert_eq!(
            created.mesh.as_ref().map(|mesh| mesh.vertices.len()),
            Some(3)
        );
        assert_eq!(
            created
                .visual_embedding
                .as_ref()
                .map(|tensor| tensor.shape.clone()),
            Some(vec![4])
        );
        assert!(created.physics.is_static);

        cmd_objects_update(
            store.as_ref(),
            &world_id,
            &created.id.to_string(),
            ObjectUpdateOptions {
                name: Some("crate-updated"),
                position: Some(&[2.0, 3.0, 4.0]),
                rotation: Some(&[0.0, 1.0, 0.0, 0.0]),
                bbox_min: None,
                bbox_max: None,
                velocity: Some(&[0.2, 0.0, 0.0]),
                semantic_label: Some("updated-storage"),
                mesh_json: Some(&updated_mesh_path),
                visual_embedding_json: Some(&updated_embedding_path),
                mass: Some(7.5),
                friction: Some(0.4),
                restitution: Some(0.2),
                material: Some("metal"),
                is_static: Some(false),
                graspable: Some(false),
                output_json: Some(&updated_path),
            },
        )
        .await
        .unwrap();

        let updated: SceneObject = read_json_file(&updated_path).unwrap();
        assert_eq!(updated.id, created.id);
        assert_eq!(updated.name, "crate-updated");
        assert_eq!(updated.pose.position.x, 2.0);
        assert_eq!(updated.pose.rotation.x, 1.0);
        assert_eq!(updated.bbox.center().x, 2.0);
        assert_eq!(updated.semantic_label.as_deref(), Some("updated-storage"));
        assert_eq!(
            updated.mesh.as_ref().map(|mesh| mesh.vertices.len()),
            Some(3)
        );
        assert_eq!(
            updated
                .visual_embedding
                .as_ref()
                .map(|tensor| tensor.shape.clone()),
            Some(vec![2])
        );
        assert_eq!(updated.physics.mass, Some(7.5));

        cmd_objects_list(
            store.as_ref(),
            &world_id,
            ObjectOutputOptions {
                output_json: Some(&list_path),
            },
        )
        .await
        .unwrap();
        let listed: Vec<SceneObject> = read_json_file(&list_path).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, created.id);
        assert_eq!(listed[0].name, "crate-updated");
        assert!(listed[0].mesh.is_some());
        assert!(listed[0].visual_embedding.is_some());

        cmd_objects_show(
            store.as_ref(),
            &world_id,
            &created.id.to_string(),
            ObjectOutputOptions {
                output_json: Some(&shown_path),
            },
        )
        .await
        .unwrap();
        let shown: SceneObject = read_json_file(&shown_path).unwrap();
        assert_eq!(shown.id, created.id);
        assert_eq!(shown.name, "crate-updated");
        assert!(shown.mesh.is_some());
        assert!(shown.visual_embedding.is_some());

        cmd_objects_remove(store.as_ref(), &world_id, &created.id.to_string())
            .await
            .unwrap();

        let updated = store.load(&state.id).await.unwrap();
        assert!(updated.scene.objects.is_empty());

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_create_seeds_world_from_prompt() {
        let dir = std::env::temp_dir().join(format!("wf-cli-create-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();

        cmd_create(
            store.as_ref(),
            "Two red blocks next to a blue mug on a table",
            Some("seeded-kitchen"),
            "mock",
        )
        .await
        .unwrap();

        let world_ids = store.list().await.unwrap();
        assert_eq!(world_ids.len(), 1);

        let state = store.load(&world_ids[0]).await.unwrap();
        assert_eq!(state.metadata.name, "seeded-kitchen");
        assert_eq!(
            state.metadata.description,
            "Two red blocks next to a blue mug on a table"
        );
        let table = state.scene.find_object_by_name("table").unwrap();
        let mug = state.scene.find_object_by_name("blue_mug").unwrap();
        let block = state.scene.find_object_by_name("red_block").unwrap();
        let block_2 = state.scene.find_object_by_name("red_block_2").unwrap();
        assert!(mug.pose.position.y > table.bbox.max.y);
        assert!(block.pose.position.distance(mug.pose.position) < 0.35);
        assert!(block_2.pose.position.distance(mug.pose.position) < 0.35);
        assert_eq!(state.history.len(), 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_export_and_import_json_roundtrip() {
        let dir = std::env::temp_dir().join(format!("wf-cli-export-json-{}", uuid::Uuid::new_v4()));
        let source_store = StateStoreKind::File(dir.join("source"))
            .open()
            .await
            .unwrap();
        let import_store = StateStoreKind::File(dir.join("import-json"))
            .open()
            .await
            .unwrap();
        let state = sample_snapshot_state("json-source");
        let source_id = state.id;
        source_store.save(&state).await.unwrap();

        let snapshot_path = dir.join("snapshot.json");
        cmd_export(
            source_store.as_ref(),
            &source_id.to_string(),
            &snapshot_path,
            None,
        )
        .await
        .unwrap();

        let exported_json: serde_json::Value =
            serde_json::from_slice(&fs::read(&snapshot_path).unwrap()).unwrap();
        assert_eq!(
            exported_json["schema_version"],
            serde_json::json!(WORLD_STATE_SNAPSHOT_SCHEMA_VERSION)
        );
        let exported = read_world_state_snapshot(&snapshot_path, None).unwrap();
        assert_eq!(exported.id, source_id);
        assert_eq!(exported.metadata.name, "json-source");
        assert!(!exported.scene.objects.is_empty());

        let imported = cmd_import(
            import_store.as_ref(),
            &snapshot_path,
            None,
            true,
            Some("json-restored"),
        )
        .await
        .unwrap();
        assert_ne!(imported.id, source_id);
        assert_eq!(imported.metadata.name, "json-restored");
        assert_eq!(imported.metadata.created_by, "mock");
        assert_eq!(imported.history.len(), exported.history.len());
        assert_eq!(imported.scene.objects.len(), exported.scene.objects.len());

        let persisted = import_store.load(&imported.id).await.unwrap();
        assert_eq!(persisted.id, imported.id);
        assert_eq!(persisted.metadata.name, "json-restored");
        assert_eq!(persisted.scene.objects.len(), exported.scene.objects.len());

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_import_legacy_snapshot_can_restore_history_checkpoint() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-legacy-import-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let import_store = StateStoreKind::File(dir.join("import"))
            .open()
            .await
            .unwrap();
        let legacy_state = legacy_snapshot_state("legacy-import-world");
        let snapshot_path = dir.join("legacy.json");

        fs::write(
            &snapshot_path,
            serde_json::to_vec_pretty(&legacy_state).unwrap(),
        )
        .unwrap();

        let imported = cmd_import(
            import_store.as_ref(),
            &snapshot_path,
            Some(CoreStateFileFormat::Json),
            false,
            None,
        )
        .await
        .unwrap();

        assert_eq!(imported.history.len(), 1);
        assert!(imported.history.latest().unwrap().snapshot.is_some());

        let restored_path = dir.join("restored.json");
        cmd_restore(
            import_store.as_ref(),
            &imported.id.to_string(),
            0,
            Some(&restored_path),
        )
        .await
        .unwrap();

        let restored = import_store.load(&imported.id).await.unwrap();
        assert!(!restored.history.is_empty());
        assert!(restored.history.latest().unwrap().snapshot.is_some());

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_export_and_import_msgpack_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-export-msgpack-{}", uuid::Uuid::new_v4()));
        let source_store = StateStoreKind::File(dir.join("source"))
            .open()
            .await
            .unwrap();
        let import_store = StateStoreKind::File(dir.join("import-msgpack"))
            .open()
            .await
            .unwrap();
        let state = sample_snapshot_state("msgpack-source");
        let source_id = state.id;
        source_store.save(&state).await.unwrap();

        let snapshot_path = dir.join("snapshot.bin");
        cmd_export(
            source_store.as_ref(),
            &source_id.to_string(),
            &snapshot_path,
            Some(CoreStateFileFormat::MessagePack),
        )
        .await
        .unwrap();

        let snapshot_bytes = fs::read(&snapshot_path).unwrap();
        let metadata =
            inspect_world_state_snapshot(CoreStateFileFormat::MessagePack, &snapshot_bytes)
                .unwrap();
        assert_eq!(metadata.schema_version, WORLD_STATE_SNAPSHOT_SCHEMA_VERSION);
        assert!(!metadata.legacy_payload);
        let exported =
            read_world_state_snapshot(&snapshot_path, Some(CoreStateFileFormat::MessagePack))
                .unwrap();
        assert_eq!(exported.id, source_id);
        assert_eq!(exported.metadata.name, "msgpack-source");
        assert!(!exported.scene.objects.is_empty());

        let imported = cmd_import(
            import_store.as_ref(),
            &snapshot_path,
            Some(CoreStateFileFormat::MessagePack),
            false,
            None,
        )
        .await
        .unwrap();
        assert_eq!(imported.id, source_id);
        assert_eq!(imported.metadata.name, "msgpack-source");
        assert_eq!(imported.history.len(), exported.history.len());
        assert_eq!(imported.scene.objects.len(), exported.scene.objects.len());

        let persisted = import_store.load(&source_id).await.unwrap();
        assert_eq!(persisted.id, source_id);
        assert_eq!(persisted.metadata.name, "msgpack-source");
        assert_eq!(persisted.scene.objects.len(), exported.scene.objects.len());

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_fork_current_state_rebases_history_and_persists_new_world() {
        let dir = std::env::temp_dir().join(format!("wf-cli-fork-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("store"))
            .open()
            .await
            .unwrap();
        let source = sample_snapshot_state("fork-source");
        let source_id = source.id;
        store.save(&source).await.unwrap();

        let forked = cmd_fork(
            store.as_ref(),
            &source_id.to_string(),
            None,
            Some("branched-world"),
        )
        .await
        .unwrap();

        assert_ne!(forked.id, source_id);
        assert_eq!(forked.metadata.name, "branched-world");
        assert_eq!(forked.history.len(), 1);
        assert_eq!(forked.metadata.description, source.metadata.description);

        let persisted = store.load(&forked.id).await.unwrap();
        assert_eq!(persisted.id, forked.id);
        assert_eq!(persisted.metadata.name, "branched-world");
        assert_eq!(persisted.history.len(), 1);

        let original = store.load(&source_id).await.unwrap();
        assert_eq!(original.id, source_id);
        assert_eq!(original.metadata.name, "fork-source");

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_fork_history_checkpoint_uses_core_fork_semantics() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-fork-history-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("store"))
            .open()
            .await
            .unwrap();
        let mut source = sample_snapshot_state("fork-history-source");
        source.ensure_history_initialized("mock").unwrap();
        source.time.step = 2;
        source.metadata.name = "checkpoint".to_string();
        source.record_current_state(None, None, "mock").unwrap();
        source.time.step = 3;
        source.metadata.name = "latest".to_string();
        source.record_current_state(None, None, "mock").unwrap();
        let source_id = source.id;
        store.save(&source).await.unwrap();

        let forked = cmd_fork(store.as_ref(), &source_id.to_string(), Some(1), None)
            .await
            .unwrap();

        assert_ne!(forked.id, source_id);
        assert_eq!(forked.metadata.name, "checkpoint Fork");
        assert_eq!(forked.time.step, 2);
        assert_eq!(forked.history.len(), 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_verify_inference_from_state_jsons() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-verify-infer-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let input_state = WorldState::new("input", "mock");
        let output_state = WorldState::new("output", "mock");
        let input_path = dir.join("input.json");
        let output_path = dir.join("output.json");
        let bundle_path = dir.join("bundle.json");
        write_json_file(&input_path, &input_state).unwrap();
        write_json_file(&output_path, &output_state).unwrap();

        cmd_verify(
            store.as_ref(),
            None,
            VerifyOptions {
                backend: VerifyBackend::Mock,
                proof_type: "inference",
                input_state_json: Some(&input_path),
                output_state_json: Some(&output_path),
                prediction_json: None,
                plan_json: None,
                plan_id: None,
                goal: None,
                goal_json: None,
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: Some("mock"),
                fallback_provider: None,
                guardrails_json: None,
                disable_guardrails: false,
                source_label: "worldforge-cli",
                output_json: Some(&bundle_path),
            },
        )
        .await
        .unwrap();

        let bundle: serde_json::Value = read_json_file(&bundle_path).unwrap();
        assert_eq!(bundle["verification"]["valid"], true);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_verify_inference_from_prediction_json() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-verify-prediction-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let prediction_path = dir.join("prediction.json");
        let bundle_path = dir.join("bundle.json");
        let mut prediction = sample_prediction("mock", 0.25, 120);
        let model_hash = worldforge_core::state::sha256_hash(b"prediction-json-model");
        prediction.provenance = Some(worldforge_core::prediction::PredictionProvenance {
            model_hash,
            asset_fingerprint: Some(42),
            backend: Some("fixture".to_string()),
        });
        write_json_file(&prediction_path, &prediction).unwrap();

        cmd_verify(
            store.as_ref(),
            None,
            VerifyOptions {
                backend: VerifyBackend::Mock,
                proof_type: "inference",
                input_state_json: None,
                output_state_json: None,
                prediction_json: Some(&prediction_path),
                plan_json: None,
                plan_id: None,
                goal: None,
                goal_json: None,
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: None,
                fallback_provider: None,
                guardrails_json: None,
                disable_guardrails: false,
                source_label: "worldforge-cli",
                output_json: Some(&bundle_path),
            },
        )
        .await
        .unwrap();

        let bundle: serde_json::Value = read_json_file(&bundle_path).unwrap();
        assert_eq!(bundle["verification"]["valid"], true);
        assert_eq!(
            bundle["artifact"]["model_hash"],
            serde_json::json!(model_hash)
        );
        assert_eq!(bundle["artifact"]["provenance"]["backend"], "fixture");

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_verify_guardrail_from_plan_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-verify-plan-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("verify-plan", "mock");
        store.save(&state).await.unwrap();

        let plan_path = dir.join("plan.json");
        cmd_plan(
            store.as_ref(),
            &state.id.to_string(),
            Some("spawn cube"),
            PlanOptions {
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: "mock",
                fallback_provider: None,
                verify_backend: None,
                goal_json: None,
                guardrails_json: None,
                disable_guardrails: false,
                output_json: Some(&plan_path),
            },
        )
        .await
        .unwrap();

        let bundle_path = dir.join("bundle.json");
        cmd_verify(
            store.as_ref(),
            None,
            VerifyOptions {
                backend: VerifyBackend::Mock,
                proof_type: "guardrail",
                input_state_json: None,
                output_state_json: None,
                prediction_json: None,
                plan_json: Some(&plan_path),
                plan_id: None,
                goal: None,
                goal_json: None,
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: None,
                fallback_provider: None,
                guardrails_json: None,
                disable_guardrails: false,
                source_label: "worldforge-cli",
                output_json: Some(&bundle_path),
            },
        )
        .await
        .unwrap();

        let bundle: serde_json::Value = read_json_file(&bundle_path).unwrap();
        assert_eq!(bundle["verification"]["valid"], true);
        assert!(bundle["artifact"]["plan_hash"].is_array());

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_verify_guardrail_from_plan_id() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-verify-plan-id-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("verify-plan-id", "mock");
        store.save(&state).await.unwrap();

        cmd_plan(
            store.as_ref(),
            &state.id.to_string(),
            Some("spawn cube"),
            PlanOptions {
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: "mock",
                fallback_provider: None,
                verify_backend: None,
                goal_json: None,
                guardrails_json: None,
                disable_guardrails: false,
                output_json: None,
            },
        )
        .await
        .unwrap();

        let persisted = store.load(&state.id).await.unwrap();
        let plan_id = persisted.stored_plans.keys().next().unwrap().to_string();
        let bundle_path = dir.join("bundle.json");

        cmd_verify(
            store.as_ref(),
            Some(&state.id.to_string()),
            VerifyOptions {
                backend: VerifyBackend::Mock,
                proof_type: "guardrail",
                input_state_json: None,
                output_state_json: None,
                prediction_json: None,
                plan_json: None,
                plan_id: Some(&plan_id),
                goal: None,
                goal_json: None,
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: None,
                fallback_provider: None,
                guardrails_json: None,
                disable_guardrails: false,
                source_label: "worldforge-cli",
                output_json: Some(&bundle_path),
            },
        )
        .await
        .unwrap();

        let bundle: serde_json::Value = read_json_file(&bundle_path).unwrap();
        assert_eq!(bundle["verification"]["valid"], true);

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_verify_guardrail_from_goal_uses_fallback_provider() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-verify-fallback-{}", uuid::Uuid::new_v4()));
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let state = WorldState::new("verify-fallback", "mock");
        store.save(&state).await.unwrap();
        let world_id = state.id.to_string();

        let guardrails_path = dir.join("guardrails.json");
        write_json_file(
            &guardrails_path,
            &serde_json::json!([{
                "guardrail": "NoCollisions",
                "blocking": true
            }]),
        )
        .unwrap();
        let bundle_path = dir.join("bundle.json");
        cmd_verify(
            store.as_ref(),
            Some(&world_id),
            VerifyOptions {
                backend: VerifyBackend::Mock,
                proof_type: "guardrail",
                input_state_json: None,
                output_state_json: None,
                prediction_json: None,
                plan_json: None,
                plan_id: None,
                goal: Some("spawn cube"),
                goal_json: None,
                max_steps: 4,
                planner_name: "sampling",
                planner_options: PlannerOptions::default(),
                timeout: 10.0,
                provider: Some("missing"),
                fallback_provider: Some("mock"),
                guardrails_json: Some(&guardrails_path),
                disable_guardrails: false,
                source_label: "worldforge-cli",
                output_json: Some(&bundle_path),
            },
        )
        .await
        .unwrap();

        let bundle: serde_json::Value = read_json_file(&bundle_path).unwrap();
        assert_eq!(bundle["verification"]["valid"], true);
        assert!(bundle["artifact"]["plan_hash"].is_array());

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_verify_proof_from_json() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-verify-proof-{}", uuid::Uuid::new_v4()));
        let proof_path = dir.join("proof.json");
        let report_path = dir.join("report.json");
        let verifier = MockVerifier::new();
        let proof = verifier.prove_inference([1; 32], [2; 32], [3; 32]).unwrap();
        write_json_file(&proof_path, &proof).unwrap();

        cmd_verify_proof(VerifyProofOptions {
            proof_json: Some(&proof_path),
            inference_bundle_json: None,
            guardrail_bundle_json: None,
            provenance_bundle_json: None,
            output_json: Some(&report_path),
        })
        .await
        .unwrap();

        let report: serde_json::Value = read_json_file(&report_path).unwrap();
        assert_eq!(report["verification"]["valid"], true);
        assert_eq!(report["proof"]["backend"], "mock");

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_cmd_verify_bundle_from_json() {
        let dir =
            std::env::temp_dir().join(format!("wf-cli-verify-bundle-{}", uuid::Uuid::new_v4()));
        let bundle_path = dir.join("bundle.json");
        let report_path = dir.join("report.json");
        let store = StateStoreKind::File(dir.join("state"))
            .open()
            .await
            .unwrap();
        let input_state = WorldState::new("input", "mock");
        let output_state = WorldState::new("output", "mock");
        let bundle =
            prove_inference_transition(&MockVerifier::new(), "mock", &input_state, &output_state)
                .unwrap();
        write_json_file(&bundle_path, &bundle).unwrap();

        cmd_verify_proof(VerifyProofOptions {
            proof_json: None,
            inference_bundle_json: Some(&bundle_path),
            guardrail_bundle_json: None,
            provenance_bundle_json: None,
            output_json: Some(&report_path),
        })
        .await
        .unwrap();

        let report: serde_json::Value = read_json_file(&report_path).unwrap();
        assert_eq!(report["current_verification"]["valid"], true);
        assert_eq!(report["verification_matches_recorded"], true);

        drop(store);
        let _ = fs::remove_dir_all(dir);
    }
}
