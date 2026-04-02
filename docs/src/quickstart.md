# Quick Start

This guide walks you through installing WorldForge and running your first
prediction in under five minutes.

## Prerequisites

- Rust 1.80+ (for building from source)
- Python 3.10+ (for Python bindings)
- At least one provider API key (or use the built-in Mock provider)

## Installation

### Rust Library

Add WorldForge crates to your `Cargo.toml`:

```toml
[dependencies]
worldforge-core = "0.1"
worldforge-providers = "0.1"
```

Or install via cargo:

```bash
cargo add worldforge-core worldforge-providers
```

### Python Bindings

```bash
pip install worldforge
```

### CLI

```bash
cargo install worldforge-cli
```

## Configure Provider Credentials

WorldForge auto-detects providers from environment variables:

```bash
export NVIDIA_API_KEY="your-key"      # Cosmos
export RUNWAY_API_SECRET="your-key"   # Runway GWM
export OPENAI_API_KEY="your-key"      # Sora 2
export GOOGLE_API_KEY="your-key"      # Veo 3
export PAN_API_KEY="your-key"         # PAN
```

No key? No problem. The **Mock** provider is always available for testing.

## Your First Prediction (Python)

```python
from worldforge import WorldForge, Action

# Initialize — auto-detects providers from env vars
wf = WorldForge()

# Create a world using a specific provider
world = wf.create_world("kitchen", provider="mock")

# Predict what happens when an object moves
prediction = world.predict(
    Action.move_to(0.5, 0.8, 0.0),
    steps=10,
)

print(f"Physics score: {prediction.physics_score}")
print(f"Frames generated: {len(prediction.frames)}")
```

## Your First Prediction (Rust)

```rust
use worldforge_providers::auto_detect_worldforge;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let wf = auto_detect_worldforge();
    let world = wf.create_world("kitchen", "mock")?;

    let action = worldforge_core::Action::move_to(0.5, 0.8, 0.0);
    let config = worldforge_core::PredictionConfig::default();
    let prediction = world.predict(&action, &config).await?;

    println!("Physics score: {}", prediction.physics_score);
    Ok(())
}
```

## Using the CLI

```bash
# Start the REST server
worldforge serve

# Run a prediction from the command line
worldforge predict --world kitchen --provider mock \
  --action '{"move_to": [0.5, 0.8, 0.0]}' --steps 10

# Run an evaluation suite
worldforge eval --suite physics --providers mock \
  --output-markdown report.md
```

## Next Steps

- Read the [Architecture](./architecture.md) overview to understand crate layout.
- Browse the [Providers](./providers/README.md) table for capabilities.
- Explore the [REST API](./api/rest.md) reference.
- Try the [Python SDK](./api/python.md) for advanced usage.
