# Providers

## Implemented in-repo

| Provider | Status | Notes |
| --- | --- | --- |
| `mock` | stable | deterministic local provider used by tests, examples, framework development, and adapter contract checks |

## Scaffold adapters

| Provider | Env var | Status |
| --- | --- | --- |
| `cosmos` | `NVIDIA_API_KEY` | scaffold adapter |
| `runway` | `RUNWAY_API_SECRET` | scaffold adapter |
| `jepa` | `JEPA_MODEL_PATH` | scaffold adapter |
| `genie` | `GENIE_API_KEY` | scaffold adapter |

## Provider profiles

Every provider now exposes a profile describing:

- supported task surface derived from capabilities
- deterministic vs stochastic behavior
- local vs remote runtime
- implementation status such as `stable` or `scaffold`
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
