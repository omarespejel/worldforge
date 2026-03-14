---
name: provider-integration
description: How to implement a WorldModelProvider adapter for WorldForge. Activate when creating a new provider (Cosmos, Runway, JEPA, Genie), modifying provider logic, or working with external world model APIs. Also activate when discussing provider capabilities, action translation, or multi-provider comparison.
prerequisites: worldforge-core types compiled, provider API access or docs
---

# Provider Integration

<purpose>
Guides implementation of provider adapters that connect WorldForge to external world model APIs (NVIDIA Cosmos, Runway GWM, Meta JEPA, Google Genie). Each provider implements the WorldModelProvider trait from worldforge-core.
</purpose>

<context>
— Provider trait defined in SPECIFICATION.md section 4.
— Provider mappings in SPECIFICATION.md section 13.
— All provider code lives in crates/worldforge-providers/src/.
— Each provider is a separate module (cosmos.rs, runway.rs, jepa.rs, genie.rs).
— Providers must handle: authentication, HTTP calls, action translation, response parsing.
— Provider capabilities are introspectable via ProviderCapabilities struct.
</context>

<procedure>
1. Read SPECIFICATION.md section 4 (Provider Trait) and section 13 (Provider Specs) for the target provider.
2. Create module file: `crates/worldforge-providers/src/{provider_name}.rs`.
3. Define provider struct with config fields (api_key, endpoint, model enum).
4. Implement `WorldModelProvider` trait:
   — `name()` → static provider name
   — `capabilities()` → what this provider can do
   — `predict()` → main inference call
   — `generate()` → video generation (if supported)
   — `reason()` → VLM-style reasoning (if supported)
   — `transfer()` → spatial control to video (if supported)
   — `health_check()` → verify connectivity
   — `estimate_cost()` → cost estimation
5. Implement `ActionTranslator` for provider-specific action mapping.
6. Handle authentication (API keys from environment or config).
7. Write integration tests with mock HTTP responses.
8. Re-export from `crates/worldforge-providers/src/lib.rs`.
</procedure>

<patterns>
<do>
  — Return `WorldForgeError::UnsupportedCapability` for operations the provider doesn't support.
  — Use reqwest with timeout and retry logic for HTTP calls.
  — Log all provider calls with tracing (request metadata, latency, response status).
  — Implement `health_check()` as a lightweight ping (not a full inference call).
  — Map provider-specific errors to WorldForgeError variants.
  — Use `#[cfg(test)]` mocks — don't require live API access for unit tests.
</do>
<dont>
  — Don't hardcode API keys — read from config or environment variables.
  — Don't add provider-specific types to worldforge-core — keep them in the provider module.
  — Don't panic on provider errors — always return Result.
  — Don't skip capability checking — verify the provider supports the requested operation.
</dont>
</patterns>

<examples>
Example: Provider struct and capability declaration

```rust
use worldforge_core::provider::{WorldModelProvider, ProviderCapabilities};

pub struct CosmosProvider {
    model: CosmosModel,
    api_key: String,
    endpoint: String,
    client: reqwest::Client,
}

impl CosmosProvider {
    pub fn new(model: CosmosModel, api_key: String) -> Self {
        Self {
            model,
            api_key,
            endpoint: "https://ai.api.nvidia.com".into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl WorldModelProvider for CosmosProvider {
    fn name(&self) -> &str { "cosmos" }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            predict: true,
            generate: true,
            reason: matches!(self.model, CosmosModel::Reason2),
            transfer: matches!(self.model, CosmosModel::Transfer2_5),
            // ...
        }
    }
    // ...
}
```
</examples>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| Provider timeout | API latency > configured timeout | Increase timeout in PredictionConfig, check provider health |
| 401 Unauthorized | Invalid or expired API key | Verify key, check provider dashboard |
| UnsupportedCapability error | Calling predict() on a reason-only model | Check capabilities() before calling |
| Serialization error on response | Provider API changed response format | Update response parsing, check provider changelog |

</troubleshooting>

<references>
— SPECIFICATION.md section 4: Provider trait definition
— SPECIFICATION.md section 13: Provider-specific mappings
— crates/worldforge-core/src/provider.rs: trait definition (to be implemented)
— crates/worldforge-providers/src/: provider implementations
</references>
