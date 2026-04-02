//! Basic prediction example using WorldForge.
//!
//! This example demonstrates how to create a world and run a single prediction
//! using the auto-detected provider registry.
//!
//! # Running
//!
//! ```bash
//! cargo run --example basic_prediction
//! ```

use worldforge_core::{Action, PredictionConfig};
use worldforge_providers::auto_detect_worldforge;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Auto-detect all available providers from environment variables.
    // The Mock provider is always available.
    let wf = auto_detect_worldforge();

    // List available providers
    let providers = wf.list_providers();
    println!("Available providers: {:?}", providers);

    // Create a world using the mock provider (always available)
    let world = wf.create_world("kitchen", "mock")?;
    println!("Created world: kitchen (provider: mock)");

    // Define an action: move an object to coordinates (0.5, 0.8, 0.0)
    let action = Action::move_to(0.5, 0.8, 0.0);

    // Run a prediction for 10 steps
    let config = PredictionConfig {
        steps: 10,
        ..Default::default()
    };
    let prediction = world.predict(&action, &config).await?;

    // Print results
    println!("Prediction complete:");
    println!("  Physics score: {:.2}", prediction.physics_score);
    println!("  Frames generated: {}", prediction.frames.len());

    Ok(())
}
