# Contributing

Thank you for your interest in contributing to WorldForge! This guide covers
the most common contribution: adding a new provider adapter.

## Development Setup

```bash
git clone https://github.com/AbdelStark/worldforge.git
cd worldforge
cargo build
cargo test
```

## Adding a New Provider

### Step 1: Create the Module

Add a new file in `crates/worldforge-providers/src/`:

```
crates/worldforge-providers/src/my_provider.rs
```

### Step 2: Implement the WorldProvider Trait

```rust
use worldforge_core::{
    WorldProvider, WorldState, Action, Prediction,
    PredictionConfig, ProviderCapabilities, ProviderInfo,
};

pub struct MyProvider {
    api_key: String,
    client: reqwest::Client,
}

impl MyProvider {
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("MY_PROVIDER_API_KEY").ok()?;
        Some(Self {
            api_key,
            client: reqwest::Client::new(),
        })
    }
}

#[async_trait::async_trait]
impl WorldProvider for MyProvider {
    fn name(&self) -> &str { "my_provider" }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: false,
            transfer: false,
            embed: false,
            plan: false,
        }
    }

    async fn predict(
        &self,
        state: &WorldState,
        action: &Action,
        config: &PredictionConfig,
    ) -> Result<Prediction> {
        // Implement your provider's prediction logic here
        todo!()
    }
}
```

### Step 3: Register the Provider

In `crates/worldforge-providers/src/lib.rs`, add your provider to the
auto-detection function:

```rust
if let Some(p) = MyProvider::from_env() {
    registry.register(Box::new(p));
}
```

### Step 4: Add Tests

Create tests in your module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_my_provider_predict() {
        // Test with mock responses
    }
}
```

### Step 5: Update Documentation

- Add a row to the providers table in `docs/src/providers/README.md`.
- Update the provider count badge in `README.md`.

## Code Standards

- Run `cargo clippy -- -D warnings` — zero warnings policy.
- Run `cargo fmt` before committing.
- All public types need doc comments.
- Tests are required for all provider implementations.

## Pull Request Process

1. Fork the repository.
2. Create a feature branch: `git checkout -b add-provider-xyz`.
3. Implement and test your changes.
4. Submit a pull request with a clear description.
5. Ensure CI passes (build, test, clippy, fmt).

## Questions?

Open an issue on GitHub or start a discussion in the repository.
