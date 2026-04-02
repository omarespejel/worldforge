//! Cross-provider comparison example.
//!
//! This example shows how to run the same prediction across multiple providers
//! and compare their results side by side.
//!
//! # Running
//!
//! Set at least two provider API keys, then:
//!
//! ```bash
//! cargo run --example cross_provider_compare
//! ```

use worldforge_core::{Action, ComparisonConfig};
use worldforge_providers::auto_detect_worldforge;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let wf = auto_detect_worldforge();

    // List detected providers
    let providers = wf.list_providers();
    println!("Detected {} providers:", providers.len());
    for p in &providers {
        println!("  - {}", p.name);
    }

    // Create a world for comparison
    let world = wf.create_world("comparison-scene", "mock")?;

    // Define the action to compare
    let action = Action::move_to(0.5, 0.8, 0.0);

    // Compare across all available providers
    let provider_names: Vec<&str> = providers.iter().map(|p| p.name.as_str()).collect();
    let config = ComparisonConfig {
        providers: provider_names,
        steps: 10,
        ..Default::default()
    };

    let comparison = world.compare(&action, &config).await?;

    // Print a markdown comparison table
    println!("\n{}", comparison.to_markdown());

    // Access individual results
    for result in &comparison.results {
        println!(
            "Provider: {} | Physics: {:.2} | Latency: {:?}",
            result.provider, result.physics_score, result.latency
        );
    }

    Ok(())
}
