# Providers

## Implemented provider

| Provider | Status | Notes |
| --- | --- | --- |
| `mock` | implemented | deterministic local provider used by tests and examples |

## Scaffolded adapters

| Provider | Env var | Status |
| --- | --- | --- |
| `cosmos` | `NVIDIA_API_KEY` | Python adapter scaffold |
| `runway` | `RUNWAY_API_SECRET` | Python adapter scaffold |
| `jepa` | `JEPA_MODEL_PATH` | Python adapter scaffold |
| `genie` | `GENIE_API_KEY` | Python adapter scaffold |

## Capability model

Each provider declares capabilities through `ProviderCapabilities`.

Current capability names:

- `predict`
- `generate`
- `reason`
- `embed`
- `plan`
- `transfer`
- `verify`

## Rule

Documentation and provider metadata must match real implementation depth. A scaffolded adapter is not a supported production integration.
