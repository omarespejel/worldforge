# Providers

## Implemented in-repo

| Provider | Status | Notes |
| --- | --- | --- |
| `mock` | implemented | deterministic local provider used by tests, examples, and framework development |

## Scaffold adapters

| Provider | Env var | Status |
| --- | --- | --- |
| `cosmos` | `NVIDIA_API_KEY` | scaffold adapter |
| `runway` | `RUNWAY_API_SECRET` | scaffold adapter |
| `jepa` | `JEPA_MODEL_PATH` | scaffold adapter |
| `genie` | `GENIE_API_KEY` | scaffold adapter |

## Capability model

Providers can declare support for:

- `predict`
- `generate`
- `reason`
- `embed`
- `plan`
- `transfer`
