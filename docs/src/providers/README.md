# Providers

## In-repo providers

| Provider | Status | Auto-registration rule | Notes |
| --- | --- | --- | --- |
| `mock` | stable | always registered | deterministic local provider used by tests, examples, framework development, and adapter contract checks |
| `cosmos` | beta | register when `COSMOS_BASE_URL` is set | real HTTP adapter for Cosmos NIM; `NVIDIA_API_KEY` is optional and sent as bearer auth when present |
| `runway` | beta | register when `RUNWAYML_API_SECRET` or `RUNWAY_API_SECRET` is set | real HTTP adapter for Runway image-to-video and video-to-video APIs |
| `jepa` | scaffold | register when `JEPA_MODEL_PATH` is set | credential-gated stub backed by deterministic mock behavior |
| `genie` | scaffold | register when `GENIE_API_KEY` is set | credential-gated stub backed by deterministic mock behavior |

## Provider profiles

Every provider exposes a profile describing:

- supported task surface derived from capabilities
- deterministic vs stochastic behavior
- local vs remote runtime
- implementation status such as `stable`, `beta`, or `scaffold`
- credential requirements and environment variables
- supported modalities and artifact types
- maintainer notes for caveats

Programmatically:

```python
from worldforge import WorldForge

forge = WorldForge()
profile = forge.provider_profile("mock")
print(profile.supported_tasks, profile.deterministic)
```

From the CLI:

```bash
uv run worldforge provider list
uv run worldforge provider info mock
uv run worldforge doctor
```

## Capability model

Providers can declare support for:

- `predict`
- `generate`
- `reason`
- `embed`
- `plan`
- `transfer`

## Operational notes

- `doctor()` includes known providers by default so missing configuration shows up in diagnostics.
- Missing local asset paths now fail before the outbound request instead of being treated as opaque remote strings.
- `cosmos` and `runway` expose a typed `ProviderRequestPolicy` through `provider_profile()` and CLI JSON output.
- Health checks, polling, and downloads retry with backoff by default. Create-style POST requests remain single-attempt unless a caller passes a custom policy.
- `cosmos` and `runway` are the only in-repo adapters that currently perform real HTTP requests.
