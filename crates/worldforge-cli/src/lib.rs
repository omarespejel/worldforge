//! WorldForge CLI
//!
//! Command-line interface for interacting with world foundation models.
//! Supports world creation, prediction, planning, evaluation, and comparison.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use worldforge_core::action::{Action, Weather};
use worldforge_core::prediction::PredictionConfig;
use worldforge_core::provider::ProviderRegistry;
use worldforge_core::state::{FileStateStore, StateStore};
use worldforge_core::types::Position;
use worldforge_core::WorldForge;
use worldforge_eval::EvalSuite;
use worldforge_providers::MockProvider;

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

    /// Check provider health.
    Health {
        /// Provider name (or "all").
        #[arg(default_value = "all")]
        provider: String,
    },
}

/// Run the CLI application.
pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt().init();

    let store = FileStateStore::new(&cli.state_dir);
    let mut wf = WorldForge::new();

    // Register available providers
    wf.register_provider(Box::new(MockProvider::new()));

    match cli.command {
        Commands::Create { prompt, provider } => cmd_create(&wf, &store, &prompt, &provider).await,
        Commands::Predict {
            world,
            action,
            steps,
            provider,
        } => cmd_predict(&wf, &store, &world, &action, steps, &provider).await,
        Commands::List => cmd_list(&store).await,
        Commands::Show { world } => cmd_show(&store, &world).await,
        Commands::Delete { world } => cmd_delete(&store, &world).await,
        Commands::Eval { suite, providers } => cmd_eval(&wf, &suite, &providers).await,
        Commands::Compare {
            world,
            action,
            providers,
        } => cmd_compare(&wf, &store, &world, &action, &providers).await,
        Commands::Health { provider } => cmd_health(&wf, &provider).await,
    }
}

async fn cmd_create(
    wf: &WorldForge,
    store: &FileStateStore,
    prompt: &str,
    provider: &str,
) -> Result<()> {
    let world = wf
        .create_world(prompt, provider)
        .context("failed to create world")?;
    store
        .save(world.current_state())
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("Created world: {}", world.id());
    println!("  Name: {prompt}");
    println!("  Provider: {provider}");
    Ok(())
}

async fn cmd_predict(
    _wf: &WorldForge,
    store: &FileStateStore,
    world_id: &str,
    action_str: &str,
    steps: u32,
    provider: &str,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        r.register(Box::new(MockProvider::new()));
        r
    });
    let mut world = worldforge_core::world::World::new(state, provider, registry);

    let action = parse_action(action_str)?;
    let config = PredictionConfig {
        steps,
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

async fn cmd_eval(_wf: &WorldForge, suite_name: &str, _providers_str: &str) -> Result<()> {
    let suite = match suite_name {
        "physics" => EvalSuite::physics_standard(),
        _ => anyhow::bail!("unknown eval suite: {suite_name}"),
    };

    let mock = MockProvider::new();
    let provider_list: Vec<&dyn worldforge_core::provider::WorldModelProvider> = vec![&mock];

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
    _wf: &WorldForge,
    store: &FileStateStore,
    world_id: &str,
    action_str: &str,
    providers_str: &str,
) -> Result<()> {
    let id: uuid::Uuid = world_id.parse().context("invalid world ID")?;
    let state = store.load(&id).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    let registry = Arc::new({
        let mut r = ProviderRegistry::new();
        // Register mock providers with different names for comparison
        for name in providers_str.split(',') {
            r.register(Box::new(MockProvider::with_name(name.trim())));
        }
        r
    });

    let world = worldforge_core::world::World::new(state, "mock", registry);
    let action = parse_action(action_str)?;
    let config = PredictionConfig::default();

    let provider_names: Vec<&str> = providers_str.split(',').map(|s| s.trim()).collect();
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

async fn cmd_health(wf: &WorldForge, provider_name: &str) -> Result<()> {
    let registry = wf.registry();
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
}
