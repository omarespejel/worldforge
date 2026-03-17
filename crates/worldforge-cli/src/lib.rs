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
use worldforge_core::guardrail::GuardrailConfig;
use worldforge_core::prediction::{PlanGoal, PlanRequest, PlannerType, PredictionConfig};
use worldforge_core::provider::{
    GenerationConfig, GenerationPrompt, ProviderRegistry, SpatialControls, TransferConfig,
    WorldModelProvider,
};
use worldforge_core::state::{DynStateStore, StateStore, StateStoreKind, WorldState};
use worldforge_core::types::{Position, VideoClip};
use worldforge_eval::EvalSuite;
use worldforge_verify::{
    prove_guardrail_plan, prove_inference_transition, prove_latest_inference, prove_provenance,
    verify_bundle, verify_proof, BundleVerificationReport, MockVerifier, VerificationBundle,
    VerificationResult, ZkProof,
};

/// Persistence backend used by the CLI.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum StateBackend {
    /// Store world states as JSON files in a directory.
    File,
    /// Store world states in a SQLite database file.
    Sqlite,
}

impl StateBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Sqlite => "sqlite",
        }
    }
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

    /// Explicit SQLite database path when using the sqlite backend.
    #[arg(long, global = true)]
    pub state_db_path: Option<PathBuf>,

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
        /// Optional fallback provider if the primary provider fails.
        #[arg(long)]
        fallback_provider: Option<String>,
        /// Maximum time to wait for a provider response before timing out.
        #[arg(long)]
        timeout_ms: Option<u64>,
    },

    /// Generate a video clip directly from a prompt.
    Generate {
        /// Prompt describing the desired video.
        #[arg(long)]
        prompt: String,
        /// Provider to use.
        #[arg(long, default_value = "mock")]
        provider: String,
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

    /// Transfer spatial controls onto an existing source clip.
    Transfer {
        /// Provider to use.
        #[arg(long, default_value = "mock")]
        provider: String,
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
        /// World ID.
        #[arg(long)]
        world: String,
        /// Natural-language reasoning query.
        #[arg(long)]
        query: String,
        /// Optional provider override.
        #[arg(long)]
        provider: Option<String>,
    },

    /// List all saved worlds.
    List,

    /// Show details of a world.
    Show {
        /// World ID.
        world: String,
    },

    /// Delete a world.
    Delete {
        /// World ID.
        world: String,
    },

    /// Run an evaluation suite.
    Eval {
        /// Evaluation suite name.
        #[arg(long)]
        suite: Option<String>,
        /// JSON file containing a custom `EvalSuite` definition.
        #[arg(long)]
        suite_json: Option<PathBuf>,
        /// Comma-separated list of providers.
        #[arg(long, default_value = "mock")]
        providers: String,
        /// Print the built-in suite names and exit.
        #[arg(long, default_value_t = false)]
        list_suites: bool,
        /// Optional path to write the evaluation report as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Compare predictions across providers.
    Compare {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Action description.
        #[arg(long)]
        action: String,
        /// Comma-separated list of providers.
        #[arg(long)]
        providers: String,
    },

    /// Plan a sequence of actions to achieve a goal.
    Plan {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Goal description (natural language).
        #[arg(long)]
        goal: String,
        /// Maximum number of planning steps.
        #[arg(long, default_value = "10")]
        max_steps: u32,
        /// Planning algorithm (sampling, cem, mpc, gradient).
        #[arg(long, default_value = "sampling")]
        planner: String,
        /// Planning timeout in seconds.
        #[arg(long, default_value = "30")]
        timeout: f64,
        /// Provider to use.
        #[arg(long, default_value = "mock")]
        provider: String,
        /// Optional JSON file containing `Vec<GuardrailConfig>`.
        #[arg(long)]
        guardrails_json: Option<PathBuf>,
        /// Optional path to write the generated `Plan` as JSON.
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Generate and verify a ZK proof for a plan.
    Verify {
        /// World ID for state-backed proofs or plan generation.
        #[arg(long)]
        world: Option<String>,
        /// Proof type: inference, guardrail, provenance.
        #[arg(long, default_value = "inference")]
        proof_type: String,
        /// JSON file containing the input `WorldState` for inference verification.
        #[arg(long)]
        input_state_json: Option<PathBuf>,
        /// JSON file containing the output `WorldState` for inference verification.
        #[arg(long)]
        output_state_json: Option<PathBuf>,
        /// JSON file containing a fully materialized `Plan` for guardrail verification.
        #[arg(long)]
        plan_json: Option<PathBuf>,
        /// Natural-language goal used to generate a plan before guardrail verification.
        #[arg(long)]
        goal: Option<String>,
        /// Maximum number of planning steps when generating a plan for verification.
        #[arg(long, default_value = "10")]
        max_steps: u32,
        /// Planning algorithm when generating a plan for guardrail verification.
        #[arg(long, default_value = "sampling")]
        planner: String,
        /// Planning timeout in seconds when generating a plan for guardrail verification.
        #[arg(long, default_value = "30")]
        timeout: f64,
        /// Optional provider override for generated plans or history-backed inference proofs.
        #[arg(long)]
        provider: Option<String>,
        /// Optional JSON file containing `Vec<GuardrailConfig>` for generated plans.
        #[arg(long)]
        guardrails_json: Option<PathBuf>,
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

/// Run the CLI application.
pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt().init();

    if let Commands::Serve { bind } = &cli.command {
        return cmd_serve(
            &cli.state_dir,
            cli.state_backend,
            cli.state_db_path.as_deref(),
            bind,
        )
        .await;
    }

    let store = open_state_store(&cli).await?;

    match cli.command {
        Commands::Create { prompt, provider } => {
            cmd_create(store.as_ref(), &prompt, &provider).await
        }
        Commands::Predict {
            world,
            action,
            steps,
            provider,
            fallback_provider,
            timeout_ms,
        } => {
            cmd_predict(
                store.as_ref(),
                &world,
                &action,
                steps,
                &provider,
                fallback_provider.as_deref(),
                timeout_ms,
            )
            .await
        }
        Commands::Generate {
            prompt,
            provider,
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
        Commands::Reason {
            world,
            query,
            provider,
        } => cmd_reason(store.as_ref(), &world, &query, provider.as_deref()).await,
        Commands::List => cmd_list(store.as_ref()).await,
        Commands::Show { world } => cmd_show(store.as_ref(), &world).await,
        Commands::Delete { world } => cmd_delete(store.as_ref(), &world).await,
        Commands::Eval {
            suite,
            suite_json,
            providers,
            list_suites,
            output_json,
        } => {
            cmd_eval(EvalOptions {
                suite_name: suite.as_deref(),
                suite_json: suite_json.as_deref(),
                providers: &providers,
                list_suites,
                output_json: output_json.as_deref(),
            })
            .await
        }
        Commands::Compare {
            world,
            action,
            providers,
        } => cmd_compare(store.as_ref(), &world, &action, &providers).await,
        Commands::Plan {
            world,
            goal,
            max_steps,
            planner,
            timeout,
            provider,
            guardrails_json,
            output_json,
        } => {
            cmd_plan(
                store.as_ref(),
                &world,
                &goal,
                PlanOptions {
                    max_steps,
                    planner_name: &planner,
                    timeout,
                    provider: &provider,
                    guardrails_json: guardrails_json.as_deref(),
                    output_json: output_json.as_deref(),
                },
            )
            .await
        }
        Commands::Verify {
            world,
            proof_type,
            input_state_json,
            output_state_json,
            plan_json,
            goal,
            max_steps,
            planner,
            timeout,
            provider,
            guardrails_json,
            source_label,
            output_json,
        } => {
            cmd_verify(
                store.as_ref(),
                world.as_deref(),
                VerifyOptions {
                    proof_type: &proof_type,
                    input_state_json: input_state_json.as_deref(),
                    output_state_json: output_state_json.as_deref(),
                    plan_json: plan_json.as_deref(),
                    goal: goal.as_deref(),
                    max_steps,
                    planner_name: &planner,
                    timeout,
                    provider: provider.as_deref(),
                    guardrails_json: guardrails_json.as_deref(),
                    source_label: &source_label,
                    output_json: output_json.as_deref(),
                },
            )
            .await
        }
        Commands::Health { provider } => cmd_health(&provider).await,
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
        Commands::Serve { .. } => unreachable!("serve command handled before store initialization"),
    }
}

fn state_store_kind(
    state_dir: &Path,
    state_backend: StateBackend,
    state_db_path: Option<&Path>,
) -> StateStoreKind {
    match state_backend {
        StateBackend::File => StateStoreKind::File(state_dir.to_path_buf()),
        StateBackend::Sqlite => StateStoreKind::Sqlite(
            state_db_path
                .map(Path::to_path_buf)
                .unwrap_or_else(|| state_dir.join("worldforge.db")),
        ),
    }
}

async fn open_state_store(cli: &Cli) -> Result<DynStateStore> {
    state_store_kind(
        &cli.state_dir,
        cli.state_backend,
        cli.state_db_path.as_deref(),
    )
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

fn available_provider_names(registry: &ProviderRegistry) -> String {
    let mut names: Vec<String> = registry.list().into_iter().map(str::to_string).collect();
    names.sort();
    names.join(", ")
}

fn available_eval_suite_names() -> String {
    EvalSuite::builtin_names().join(", ")
}

fn resolve_provider_name<'a>(state: &'a WorldState, provider: Option<&'a str>) -> &'a str {
    provider
        .filter(|name| !name.is_empty())
        .unwrap_or(state.metadata.created_by.as_str())
}

struct GenerateOptions<'a> {
    negative_prompt: Option<&'a str>,
    duration_seconds: f64,
    resolution: (u32, u32),
    fps: f32,
    temperature: f32,
    seed: Option<u64>,
    output_json: Option<&'a Path>,
}

struct TransferOptions<'a> {
    source_json: &'a Path,
    controls_json: Option<&'a Path>,
    output_json: Option<&'a Path>,
    resolution: (u32, u32),
    fps: f32,
    control_strength: f32,
}

struct PlanOptions<'a> {
    max_steps: u32,
    planner_name: &'a str,
    timeout: f64,
    provider: &'a str,
    guardrails_json: Option<&'a Path>,
    output_json: Option<&'a Path>,
}

struct EvalOptions<'a> {
    suite_name: Option<&'a str>,
    suite_json: Option<&'a Path>,
    providers: &'a str,
    list_suites: bool,
    output_json: Option<&'a Path>,
}

struct VerifyOptions<'a> {
    proof_type: &'a str,
    input_state_json: Option<&'a Path>,
    output_state_json: Option<&'a Path>,
    plan_json: Option<&'a Path>,
    goal: Option<&'a str>,
    max_steps: u32,
    planner_name: &'a str,
    timeout: f64,
    provider: Option<&'a str>,
    guardrails_json: Option<&'a Path>,
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

fn read_json_file<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read JSON from {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse JSON from {}", path.display()))
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let contents = serde_json::to_string_pretty(value).context("failed to serialize JSON")?;
    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

fn read_guardrails(path: Option<&Path>) -> Result<Vec<GuardrailConfig>> {
    match path {
        Some(path) => read_json_file(path),
        None => Ok(Vec::new()),
    }
}

fn load_eval_suite(suite_name: Option<&str>, suite_json: Option<&Path>) -> Result<EvalSuite> {
    match suite_json {
        Some(path) => EvalSuite::from_json_path(path).map_err(|e| anyhow::anyhow!("{e}")),
        None => EvalSuite::from_builtin(suite_name.unwrap_or("physics"))
            .map_err(|e| anyhow::anyhow!("{e}")),
    }
}

fn planner_from_name(planner_name: &str, max_steps: u32) -> Result<PlannerType> {
    match planner_name {
        "sampling" => Ok(PlannerType::Sampling {
            num_samples: 32,
            top_k: 5,
        }),
        "cem" => Ok(PlannerType::CEM {
            population_size: 64,
            elite_fraction: 0.1,
            num_iterations: 5,
        }),
        "mpc" => Ok(PlannerType::MPC {
            horizon: max_steps,
            num_samples: 32,
            replanning_interval: 1,
        }),
        "gradient" => Ok(PlannerType::Gradient {
            learning_rate: 0.01,
            num_iterations: 100,
        }),
        "provider-native" | "provider_native" | "native" => Ok(PlannerType::ProviderNative),
        other => anyhow::bail!(
            "unknown planner: {other}. Available: sampling, cem, mpc, gradient, provider-native"
        ),
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

async fn cmd_create(
    store: &(impl StateStore + ?Sized),
    prompt: &str,
    provider: &str,
) -> Result<()> {
    let registry = auto_detect_registry();
    require_provider(&registry, provider).context("failed to create world")?;
    let state = WorldState::new(prompt, provider);
    store
        .save(&state)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("Created world: {}", state.id);
    println!("  Name: {prompt}");
    println!("  Provider: {provider}");
    Ok(())
}

async fn cmd_predict(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    action_str: &str,
    steps: u32,
    provider: &str,
    fallback_provider: Option<&str>,
    timeout_ms: Option<u64>,
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
    let mut world = worldforge_core::world::World::new(state, provider, registry);

    let action = parse_action(action_str)?;
    let config = PredictionConfig {
        steps,
        fallback_provider: fallback_provider.map(ToOwned::to_owned),
        max_latency_ms: timeout_ms,
        ..PredictionConfig::default()
    };

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
    println!("  New time step: {}", world.current_state().time.step);

    Ok(())
}

async fn cmd_generate(
    prompt: &str,
    provider_name: &str,
    options: GenerateOptions<'_>,
) -> Result<()> {
    let registry = auto_detect_registry();
    let provider = require_provider(&registry, provider_name)?;
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

    let clip = provider
        .generate(&prompt, &config)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Generation completed:");
    println!("  Provider: {provider_name}");
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
    let registry = auto_detect_registry();
    let provider = require_provider(&registry, provider_name)?;
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

    let clip = provider
        .transfer(&source, &controls, &config)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Transfer completed:");
    println!("  Provider: {provider_name}");
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

async fn cmd_reason(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    query: &str,
    provider: Option<&str>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let provider_name = resolve_provider_name(&state, provider).to_string();
    let registry = Arc::new(auto_detect_registry());
    require_provider(&registry, &provider_name)?;
    let world = worldforge_core::world::World::new(state, &provider_name, registry);

    let output = world
        .reason(query)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Reasoning completed:");
    println!("  Provider: {provider_name}");
    println!("  Answer: {}", output.answer);
    println!("  Confidence: {:.2}", output.confidence);
    if output.evidence.is_empty() {
        println!("  Evidence: none");
    } else {
        println!("  Evidence:");
        for evidence in output.evidence {
            println!("    - {evidence}");
        }
    }

    Ok(())
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
    for obj in state.scene.objects.values() {
        println!(
            "    - {} (pos: {:.1}, {:.1}, {:.1})",
            obj.name, obj.pose.position.x, obj.pose.position.y, obj.pose.position.z
        );
    }
    println!("  History entries: {}", state.history.len());
    Ok(())
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

async fn cmd_eval(options: EvalOptions<'_>) -> Result<()> {
    if options.list_suites {
        println!("Built-in evaluation suites:");
        for suite_name in EvalSuite::builtin_names() {
            println!("  {suite_name}");
        }
        return Ok(());
    }

    let suite = load_eval_suite(options.suite_name, options.suite_json)
        .with_context(|| format!("available suites: {}", available_eval_suite_names()))?;
    let registry = auto_detect_registry();
    let provider_names = parse_provider_names(options.providers);
    let mut provider_list: Vec<&dyn WorldModelProvider> = Vec::new();
    for provider_name in &provider_names {
        provider_list.push(require_provider(&registry, provider_name)?);
    }

    let report = suite
        .run(&provider_list)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if let Some(path) = options.output_json {
        write_json_file(path, &report)?;
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
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    action_str: &str,
    providers_str: &str,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    let provider_names = parse_provider_names(providers_str);
    let registry = Arc::new(auto_detect_registry());
    for provider_name in &provider_names {
        require_provider(&registry, provider_name)?;
    }

    let default_provider = provider_names.first().map(String::as_str).unwrap_or("mock");
    let world = worldforge_core::world::World::new(state, default_provider, registry);
    let action = parse_action(action_str)?;
    let config = PredictionConfig::default();

    let provider_names: Vec<&str> = provider_names.iter().map(String::as_str).collect();
    let multi = world
        .predict_multi(&action, &provider_names, &config)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Comparison results:");
    println!("  Agreement score: {:.2}", multi.agreement_score);
    println!(
        "  Best provider: {}",
        multi.predictions[multi.best_prediction].provider
    );
    println!();
    for score in &multi.comparison.scores {
        println!(
            "  {} — physics: {:.2}, latency: {}ms",
            score.provider, score.physics_scores.overall, score.latency_ms
        );
    }
    Ok(())
}

async fn cmd_plan(
    store: &(impl StateStore + ?Sized),
    world_id: &str,
    goal: &str,
    options: PlanOptions<'_>,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    let registry = Arc::new(auto_detect_registry());
    require_provider(&registry, options.provider)?;
    let world = worldforge_core::world::World::new(state.clone(), options.provider, registry);
    let planner = planner_from_name(options.planner_name, options.max_steps)?;
    let guardrails = read_guardrails(options.guardrails_json)?;

    let request = PlanRequest {
        current_state: state,
        goal: PlanGoal::Description(goal.to_string()),
        max_steps: options.max_steps,
        guardrails,
        planner,
        timeout_seconds: options.timeout,
    };

    let plan = world
        .plan(&request)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Plan generated:");
    println!("  Actions: {}", plan.actions.len());
    println!("  Success probability: {:.2}", plan.success_probability);
    println!("  Planning time: {}ms", plan.planning_time_ms);
    println!("  Iterations: {}", plan.iterations_used);
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

async fn cmd_verify(
    store: &(impl StateStore + ?Sized),
    world_id: Option<&str>,
    options: VerifyOptions<'_>,
) -> Result<()> {
    let verifier = MockVerifier::new();
    let loaded_state = match world_id {
        Some(world_id) => {
            let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
            Some(store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?)
        }
        None => None,
    };

    match options.proof_type {
        "inference" => {
            let bundle = match (options.input_state_json, options.output_state_json) {
                (Some(input_path), Some(output_path)) => {
                    let input_state: WorldState = read_json_file(input_path)?;
                    let output_state: WorldState = read_json_file(output_path)?;
                    let provider_name = options
                        .provider
                        .filter(|name| !name.is_empty())
                        .unwrap_or(output_state.metadata.created_by.as_str());
                    prove_inference_transition(
                        &verifier,
                        provider_name,
                        &input_state,
                        &output_state,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                }
                (None, None) => {
                    let state = loaded_state.as_ref().context(
                        "inference verification requires either --world with at least two recorded history entries, or both --input-state-json and --output-state-json",
                    )?;
                    prove_latest_inference(&verifier, state, options.provider)
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                }
                _ => anyhow::bail!(
                    "inference verification requires both --input-state-json and --output-state-json"
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
            } else {
                let state = loaded_state.as_ref().context(
                    "guardrail verification requires --plan-json or --world together with --goal",
                )?;
                let goal = options.goal.context(
                    "guardrail verification requires --goal when --plan-json is not provided",
                )?;
                let provider_name = resolve_provider_name(state, options.provider).to_string();
                let registry = Arc::new(auto_detect_registry());
                require_provider(&registry, &provider_name)?;
                let world =
                    worldforge_core::world::World::new(state.clone(), &provider_name, registry);
                let request = PlanRequest {
                    current_state: state.clone(),
                    goal: PlanGoal::Description(goal.to_string()),
                    max_steps: options.max_steps,
                    guardrails: read_guardrails(options.guardrails_json)?,
                    planner: planner_from_name(options.planner_name, options.max_steps)?,
                    timeout_seconds: options.timeout,
                };
                world
                    .plan(&request)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };

            let bundle =
                prove_guardrail_plan(&verifier, &plan).map_err(|e| anyhow::anyhow!("{e}"))?;
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
            let bundle = prove_provenance(&verifier, &state, options.source_label, timestamp)
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
    let verifier = MockVerifier::new();

    if let Some(path) = options.proof_json {
        let proof: ZkProof = read_json_file(path)?;
        let report = ProofVerificationReport {
            verification: verify_proof(&verifier, &proof).map_err(|e| anyhow::anyhow!("{e}"))?,
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
        let report = verify_bundle(&verifier, &bundle).map_err(|e| anyhow::anyhow!("{e}"))?;
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
        let report = verify_bundle(&verifier, &bundle).map_err(|e| anyhow::anyhow!("{e}"))?;
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
        let report = verify_bundle(&verifier, &bundle).map_err(|e| anyhow::anyhow!("{e}"))?;
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
    let providers_to_check: Vec<&str> = if provider_name == "all" {
        registry.list()
    } else {
        vec![provider_name]
    };

    for name in providers_to_check {
        match registry.get(name) {
            Ok(provider) => match provider.health_check().await {
                Ok(status) => {
                    let icon = if status.healthy { "OK" } else { "UNHEALTHY" };
                    println!(
                        "  [{icon}] {name}: {} ({}ms)",
                        status.message, status.latency_ms
                    );
                }
                Err(e) => {
                    println!("  [ERROR] {name}: {e}");
                }
            },
            Err(e) => {
                println!("  [NOT FOUND] {name}: {e}");
            }
        }
    }
    Ok(())
}

async fn cmd_serve(
    state_dir: &Path,
    state_backend: StateBackend,
    state_db_path: Option<&Path>,
    bind: &str,
) -> Result<()> {
    let registry = Arc::new(auto_detect_registry());
    let config = worldforge_server::ServerConfig {
        bind_address: bind.to_string(),
        state_dir: state_dir.display().to_string(),
        state_backend: state_backend.as_str().to_string(),
        state_db_path: state_db_path.map(|path| path.display().to_string()),
    };

    worldforge_server::serve(config, registry)
        .await
        .context("failed to start server")
}

/// Parse a simple action string into an Action.
fn parse_action(s: &str) -> Result<Action> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        anyhow::bail!("empty action string");
    }

    match parts[0] {
        "move" => {
            let x = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let y = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let z = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            Ok(Action::Move {
                target: Position { x, y, z },
                speed: 1.0,
            })
        }
        "set-weather" => {
            let weather = match parts.get(1).copied().unwrap_or("clear") {
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
        "set-lighting" => {
            let time = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(12.0);
            Ok(Action::SetLighting { time_of_day: time })
        }
        "spawn" => {
            let template = parts.get(1).unwrap_or(&"object").to_string();
            Ok(Action::SpawnObject {
                template,
                pose: worldforge_core::types::Pose::default(),
            })
        }
        _ => {
            // Treat as a raw action
            Ok(Action::Raw {
                provider: "cli".to_string(),
                data: serde_json::json!({ "text": s }),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_verify::ZkVerifier;

    #[test]
    fn test_parse_action_move() {
        let action = parse_action("move 1 2 3").unwrap();
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
        let action = parse_action("set-weather rain").unwrap();
        match action {
            Action::SetWeather { weather } => assert_eq!(weather, Weather::Rain),
            _ => panic!("expected SetWeather"),
        }
    }

    #[test]
    fn test_parse_action_raw() {
        let action = parse_action("push mug left").unwrap();
        match action {
            Action::Raw { provider, .. } => assert_eq!(provider, "cli"),
            _ => panic!("expected Raw"),
        }
    }

    #[test]
    fn test_parse_action_spawn() {
        let action = parse_action("spawn cube").unwrap();
        match action {
            Action::SpawnObject { template, .. } => assert_eq!(template, "cube"),
            _ => panic!("expected SpawnObject"),
        }
    }

    #[test]
    fn test_parse_action_set_lighting() {
        let action = parse_action("set-lighting 18.5").unwrap();
        match action {
            Action::SetLighting { time_of_day } => {
                assert!((time_of_day - 18.5).abs() < f32::EPSILON)
            }
            _ => panic!("expected SetLighting"),
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
        match cli.command {
            Commands::Serve { bind } => assert_eq!(bind, "127.0.0.1:9000"),
            _ => panic!("expected Serve"),
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
    fn test_state_store_kind_defaults_sqlite_path_under_state_dir() {
        assert_eq!(
            state_store_kind(Path::new(".wf"), StateBackend::Sqlite, None),
            StateStoreKind::Sqlite(PathBuf::from(".wf/worldforge.db"))
        );
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
            "--fallback-provider",
            "mock",
            "--timeout-ms",
            "250",
        ])
        .unwrap();

        match cli.command {
            Commands::Predict {
                provider,
                fallback_provider,
                timeout_ms,
                ..
            } => {
                assert_eq!(provider, "runway");
                assert_eq!(fallback_provider.as_deref(), Some("mock"));
                assert_eq!(timeout_ms, Some(250));
            }
            _ => panic!("expected Predict"),
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
                source_json,
                controls_json,
                output_json,
                width,
                height,
                fps,
                control_strength,
            } => {
                assert_eq!(provider, "mock");
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
        ])
        .unwrap();

        match cli.command {
            Commands::Reason {
                world,
                query,
                provider,
            } => {
                assert_eq!(world, "123e4567-e89b-12d3-a456-426614174000");
                assert_eq!(query, "will the mug fall?");
                assert_eq!(provider.as_deref(), Some("cosmos"));
            }
            _ => panic!("expected Reason"),
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
            "--output-json",
            "/tmp/eval-report.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Eval {
                suite,
                suite_json,
                providers,
                list_suites,
                output_json,
            } => {
                assert!(suite.is_none());
                assert_eq!(suite_json, Some(PathBuf::from("/tmp/custom-suite.json")));
                assert_eq!(providers, "mock,jepa");
                assert!(!list_suites);
                assert_eq!(output_json, Some(PathBuf::from("/tmp/eval-report.json")));
            }
            _ => panic!("expected Eval"),
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
            "--guardrails-json",
            "/tmp/guardrails.json",
            "--output-json",
            "/tmp/plan.json",
        ])
        .unwrap();

        match cli.command {
            Commands::Plan {
                guardrails_json,
                output_json,
                ..
            } => {
                assert_eq!(guardrails_json, Some(PathBuf::from("/tmp/guardrails.json")));
                assert_eq!(output_json, Some(PathBuf::from("/tmp/plan.json")));
            }
            _ => panic!("expected Plan"),
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
                plan_json,
                output_json,
                source_label,
                ..
            } => {
                assert!(world.is_none());
                assert_eq!(proof_type, "guardrail");
                assert_eq!(plan_json, Some(PathBuf::from("/tmp/plan.json")));
                assert_eq!(output_json, Some(PathBuf::from("/tmp/proof.json")));
                assert_eq!(source_label, "ci");
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
    async fn test_cmd_eval_loads_custom_suite_and_writes_output_json() {
        let dir = std::env::temp_dir().join(format!("wf-cli-eval-{}", uuid::Uuid::new_v4()));
        let suite_path = dir.join("suite.json");
        let report_path = dir.join("report.json");
        let suite = EvalSuite::physics_standard();
        write_json_file(&suite_path, &suite).unwrap();

        cmd_eval(EvalOptions {
            suite_name: None,
            suite_json: Some(&suite_path),
            providers: "mock",
            list_suites: false,
            output_json: Some(&report_path),
        })
        .await
        .unwrap();

        let report: serde_json::Value = read_json_file(&report_path).unwrap();
        assert_eq!(report["suite"], "Physics Standard");
        assert_eq!(report["leaderboard"][0]["provider"], "mock");

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
        assert_eq!(clip.resolution, source.resolution);
        assert_eq!(clip.fps, source.fps);

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
            "spawn cube",
            PlanOptions {
                max_steps: 4,
                planner_name: "sampling",
                timeout: 10.0,
                provider: "mock",
                guardrails_json: Some(&guardrails_path),
                output_json: Some(&plan_path),
            },
        )
        .await
        .unwrap();

        let plan: worldforge_core::prediction::Plan = read_json_file(&plan_path).unwrap();
        assert!(!plan.actions.is_empty());

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
                proof_type: "inference",
                input_state_json: Some(&input_path),
                output_state_json: Some(&output_path),
                plan_json: None,
                goal: None,
                max_steps: 4,
                planner_name: "sampling",
                timeout: 10.0,
                provider: Some("mock"),
                guardrails_json: None,
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
            "spawn cube",
            PlanOptions {
                max_steps: 4,
                planner_name: "sampling",
                timeout: 10.0,
                provider: "mock",
                guardrails_json: None,
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
                proof_type: "guardrail",
                input_state_json: None,
                output_state_json: None,
                plan_json: Some(&plan_path),
                goal: None,
                max_steps: 4,
                planner_name: "sampling",
                timeout: 10.0,
                provider: None,
                guardrails_json: None,
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
        assert_eq!(report["proof"]["backend"], "Mock");

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
