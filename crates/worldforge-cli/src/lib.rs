//! WorldForge CLI
//!
//! Command-line interface for interacting with world foundation models.
//! Supports world creation, prediction, planning, evaluation, and comparison.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use worldforge_core::action::{Action, Weather};
use worldforge_core::prediction::{PlanGoal, PlanRequest, PlannerType, PredictionConfig};
use worldforge_core::provider::{ProviderRegistry, WorldModelProvider};
use worldforge_core::state::{FileStateStore, StateStore, WorldState};
use worldforge_core::types::Position;
use worldforge_eval::EvalSuite;
use worldforge_verify::{MockVerifier, ZkVerifier};

/// WorldForge — orchestration layer for world foundation models.
#[derive(Parser)]
#[command(name = "worldforge", version, about)]
pub struct Cli {
    /// State storage directory.
    #[arg(long, default_value = ".worldforge", global = true)]
    pub state_dir: PathBuf,

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
        #[arg(long, default_value = "physics")]
        suite: String,
        /// Comma-separated list of providers.
        #[arg(long, default_value = "mock")]
        providers: String,
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
    },

    /// Generate and verify a ZK proof for a plan.
    Verify {
        /// World ID.
        #[arg(long)]
        world: String,
        /// Proof type: inference, guardrail, provenance.
        #[arg(long, default_value = "inference")]
        proof_type: String,
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

    let store = FileStateStore::new(&cli.state_dir);

    match cli.command {
        Commands::Create { prompt, provider } => cmd_create(&store, &prompt, &provider).await,
        Commands::Predict {
            world,
            action,
            steps,
            provider,
            fallback_provider,
            timeout_ms,
        } => {
            cmd_predict(
                &store,
                &world,
                &action,
                steps,
                &provider,
                fallback_provider.as_deref(),
                timeout_ms,
            )
            .await
        }
        Commands::List => cmd_list(&store).await,
        Commands::Show { world } => cmd_show(&store, &world).await,
        Commands::Delete { world } => cmd_delete(&store, &world).await,
        Commands::Eval { suite, providers } => cmd_eval(&suite, &providers).await,
        Commands::Compare {
            world,
            action,
            providers,
        } => cmd_compare(&store, &world, &action, &providers).await,
        Commands::Plan {
            world,
            goal,
            max_steps,
            planner,
            timeout,
            provider,
        } => {
            cmd_plan(
                &store, &world, &goal, max_steps, &planner, timeout, &provider,
            )
            .await
        }
        Commands::Verify { world, proof_type } => cmd_verify(&store, &world, &proof_type).await,
        Commands::Health { provider } => cmd_health(&provider).await,
        Commands::Serve { bind } => cmd_serve(&cli.state_dir, &bind).await,
    }
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

async fn cmd_create(store: &FileStateStore, prompt: &str, provider: &str) -> Result<()> {
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
    store: &FileStateStore,
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

async fn cmd_list(store: &FileStateStore) -> Result<()> {
    // Ensure directory exists
    if !store.path.exists() {
        println!("No worlds found.");
        return Ok(());
    }
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

async fn cmd_show(store: &FileStateStore, world_id: &str) -> Result<()> {
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

async fn cmd_delete(store: &FileStateStore, world_id: &str) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    store
        .delete(&id)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("Deleted world: {id}");
    Ok(())
}

async fn cmd_eval(suite_name: &str, providers_str: &str) -> Result<()> {
    let suite = match suite_name {
        "physics" => EvalSuite::physics_standard(),
        "manipulation" => EvalSuite::manipulation_standard(),
        "spatial" => EvalSuite::spatial_reasoning(),
        "comprehensive" => EvalSuite::comprehensive(),
        _ => anyhow::bail!(
            "unknown eval suite: {suite_name}. Available: physics, manipulation, spatial, comprehensive"
        ),
    };

    let registry = auto_detect_registry();
    let provider_names = parse_provider_names(providers_str);
    let mut provider_list: Vec<&dyn WorldModelProvider> = Vec::new();
    for provider_name in &provider_names {
        provider_list.push(require_provider(&registry, provider_name)?);
    }

    let report = suite
        .run(&provider_list)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

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
    store: &FileStateStore,
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

#[allow(clippy::too_many_arguments)]
async fn cmd_plan(
    store: &FileStateStore,
    world_id: &str,
    goal: &str,
    max_steps: u32,
    planner_name: &str,
    timeout: f64,
    provider: &str,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    let registry = Arc::new(auto_detect_registry());
    require_provider(&registry, provider)?;
    let world = worldforge_core::world::World::new(state.clone(), provider, registry);

    let planner = match planner_name {
        "sampling" => PlannerType::Sampling {
            num_samples: 32,
            top_k: 5,
        },
        "cem" => PlannerType::CEM {
            population_size: 64,
            elite_fraction: 0.1,
            num_iterations: 5,
        },
        "mpc" => PlannerType::MPC {
            horizon: max_steps,
            num_samples: 32,
            replanning_interval: 1,
        },
        "gradient" => PlannerType::Gradient {
            learning_rate: 0.01,
            num_iterations: 100,
        },
        other => anyhow::bail!("unknown planner: {other}. Available: sampling, cem, mpc, gradient"),
    };

    let request = PlanRequest {
        current_state: state,
        goal: PlanGoal::Description(goal.to_string()),
        max_steps,
        guardrails: Vec::new(),
        planner,
        timeout_seconds: timeout,
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

    Ok(())
}

async fn cmd_verify(store: &FileStateStore, world_id: &str, proof_type: &str) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    let verifier = MockVerifier::new();

    match proof_type {
        "inference" => {
            let model_hash = worldforge_verify::sha256_hash(b"mock-model");
            let input_hash =
                worldforge_verify::sha256_hash(&serde_json::to_vec(&state).unwrap_or_default());
            let output_hash = worldforge_verify::sha256_hash(b"mock-output");

            let proof = verifier
                .prove_inference(model_hash, input_hash, output_hash)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            println!("ZK Proof generated:");
            println!("  Type: InferenceVerification");
            println!("  Backend: {:?}", proof.backend);
            println!("  Proof size: {} bytes", proof.proof_data.len());
            println!("  Generation time: {}ms", proof.generation_time_ms);

            let result = verifier
                .verify(&proof)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!();
            println!("Verification:");
            println!("  Valid: {}", result.valid);
            println!("  Details: {}", result.details);
            println!("  Verification time: {}ms", result.verification_time_ms);
        }
        "guardrail" => {
            use worldforge_core::prediction::Plan;

            let plan = Plan {
                actions: Vec::new(),
                predicted_states: Vec::new(),
                predicted_videos: None,
                total_cost: 0.0,
                success_probability: 1.0,
                guardrail_compliance: Vec::new(),
                planning_time_ms: 0,
                iterations_used: 0,
            };

            let proof = verifier
                .prove_guardrail_compliance(&plan, &[])
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            println!("ZK Proof generated:");
            println!("  Type: GuardrailCompliance");
            println!("  Backend: {:?}", proof.backend);
            println!("  Proof size: {} bytes", proof.proof_data.len());
            println!("  Generation time: {}ms", proof.generation_time_ms);

            let result = verifier
                .verify(&proof)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!();
            println!("Verification:");
            println!("  Valid: {}", result.valid);
            println!("  Details: {}", result.details);
            println!("  Verification time: {}ms", result.verification_time_ms);
        }
        "provenance" => {
            let data_hash =
                worldforge_verify::sha256_hash(&serde_json::to_vec(&state).unwrap_or_default());
            let timestamp = chrono::Utc::now().timestamp() as u64;
            let source_commitment = worldforge_verify::sha256_hash(b"worldforge-cli");

            let proof = verifier
                .prove_data_provenance(data_hash, timestamp, source_commitment)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            println!("ZK Proof generated:");
            println!("  Type: DataProvenance");
            println!("  Backend: {:?}", proof.backend);
            println!("  Proof size: {} bytes", proof.proof_data.len());
            println!("  Generation time: {}ms", proof.generation_time_ms);

            let result = verifier
                .verify(&proof)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!();
            println!("Verification:");
            println!("  Valid: {}", result.valid);
            println!("  Details: {}", result.details);
            println!("  Verification time: {}ms", result.verification_time_ms);
        }
        other => anyhow::bail!(
            "unknown proof type: {other}. Available: inference, guardrail, provenance"
        ),
    }

    Ok(())
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

async fn cmd_serve(state_dir: &Path, bind: &str) -> Result<()> {
    let registry = Arc::new(auto_detect_registry());
    let config = worldforge_server::ServerConfig {
        bind_address: bind.to_string(),
        state_dir: state_dir.display().to_string(),
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
        match cli.command {
            Commands::Serve { bind } => assert_eq!(bind, "127.0.0.1:9000"),
            _ => panic!("expected Serve"),
        }
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
}
