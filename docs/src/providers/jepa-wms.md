# JEPA WMS Provider

Status: scaffold candidate with fake-runtime and host-owned torch-hub contract tests

Taxonomy category: JEPA latent predictive world model

This is a candidate scaffold for a future adapter around
[`facebookresearch/jepa-wms`](https://github.com/facebookresearch/jepa-wms), the Meta FAIR code,
data, weights, training loops, shared planning components, and planning evaluations for
joint-embedding predictive world models.

It is not exported from `worldforge.providers`, not registered by `WorldForge._known_providers`,
and does not auto-import the upstream research repository. Keep it that way until the adapter calls
the real upstream runtime and returns validated WorldForge models.

By default the provider advertises no public capabilities. When a test or host experiment injects
`runtime=` or uses `JEPAWMSProvider.from_torch_hub(...)`, it advertises `score` and exercises the
WorldForge-side scoring contract. These direct construction paths are intentionally not wired into
auto-registration.

The scaffold health check remains unhealthy when only `JEPA_WMS_MODEL_PATH` is set because a path
does not prove the integration can execute. It becomes healthy only when both `JEPA_WMS_MODEL_PATH`
or `model_path=` and an injected runtime, including the torch-hub runtime, are present.

## Contract Status

- [x] `score` contract implemented behind injected fake/runtime object.
- [x] Fixture-driven tests for malformed input, upstream error payloads, non-finite scores, score
  count mismatches, provider contract checks, and event emission.
- [x] Host-owned torch-hub runtime shim for `facebookresearch/jepa-wms` model loading and
  encode/unroll/latent-distance scoring.
- [x] `World.plan(...)` score-planning coverage through the JEPA-WMS candidate.
- [ ] Real upstream smoke against downloaded `facebook/jepa-wms` weights completed.
- [ ] Auto-registration decision made.

The intended public surface is `score_actions(...) -> ActionScoreResult`, matching the
LeWorldModel-shaped planning path. A future implementation should rank candidate action sequences
from task-shaped observations, goals, and action candidates. It should not expose `predict=True`
unless it can return a complete validated WorldForge `PredictionPayload`, not just a latent rollout
internal to JEPA-WMS.

## Configuration

- Required environment variable: `JEPA_WMS_MODEL_PATH`.
- Optional torch-hub model name: `JEPA_WMS_MODEL_NAME`, for example `jepa_wm_pusht`.
- Optional torch device: `JEPA_WMS_DEVICE`, for example `cpu` or `cuda:0`.

- Optional dependencies: expected to be host-owned until a real adapter is implemented. Do not add
  `jepa-wms`, PyTorch, datasets, checkpoints, or simulator dependencies to WorldForge's base
  install.
- Registration rule: none yet. The scaffold is not auto-registered. A real adapter can consider
  registration when `JEPA_WMS_MODEL_PATH` points at a supported checkpoint or local repo/runtime
  layout.

## Current Fake Runtime Contract

For tests and host experiments, instantiate the provider directly:

```python
from worldforge.providers.jepa_wms import JEPAWMSProvider

provider = JEPAWMSProvider(
    model_path="/models/jepa-wms/checkpoint.pt",
    runtime=fake_or_host_runtime,
)
```

The injected runtime must be callable or expose:

```python
score_actions(*, model_path: str, info: dict, action_candidates: object) -> object
```

Input validation currently requires:

- `info["observation"]`: tensor-like object or rectangular nested finite numeric sequence with at
  least two dimensions.
- `info["goal"]`: tensor-like object or rectangular nested finite numeric sequence with at least
  two dimensions.
- `info["action_history"]`: optional tensor-like object or rectangular nested finite numeric
  sequence with at least two dimensions.
- `action_candidates`: tensor-like object or rectangular nested finite numeric sequence shaped as
  `(batch, samples, horizon, action_dim)`.

The runtime success response must be a JSON object:

```json
{
  "scores": [0.4, 0.12, 0.9],
  "lower_is_better": true,
  "metadata": {
    "score_units": "latent_cost"
  }
}
```

`best_index` is optional. If omitted, WorldForge derives it from `scores` and
`lower_is_better`. The number of scores must equal the `samples` dimension in
`action_candidates`.

The runtime failure response must be a JSON object:

```json
{
  "error": {
    "type": "checkpoint_expired",
    "message": "checkpoint artifact is expired or unavailable"
  }
}
```

Failure responses are converted to `ProviderError` and emit a `ProviderEvent` with
`operation="score"` and `phase="failure"`.

## Host-Owned Torch-Hub Runtime

The candidate includes a lazy torch-hub runtime shim:

```python
from worldforge.providers.jepa_wms import JEPAWMSProvider

provider = JEPAWMSProvider.from_torch_hub(
    model_name="jepa_wm_pusht",
    device="cpu",
)
```

This follows the upstream loading pattern:

```python
model, preprocessor = torch.hub.load(
    "facebookresearch/jepa-wms",
    "jepa_wm_pusht",
)
```

The runtime lazily imports `torch` and lazily calls `torch.hub.load(...)` only when
`score_actions(...)` is invoked. It first delegates to a model-native
`score_actions(...)`, `score_action_candidates(...)`, or `compute_scores(...)` method if a host
wraps one. If the loaded model does not expose a scoring method, it uses the JEPA-WMS planning
shape described upstream:

```text
observation -> model.encode(..., act=True) -> z_init
goal        -> model.encode(..., act=False) -> z_goal
actions     -> model.unroll(z_init, act_suffix=actions)
score       -> latent L1/L2 distance between final predicted latent and goal latent
```

The torch-hub runtime expects `action_candidates` shaped as
`(1, samples, horizon, action_dim)`. WorldForge returns one score per sample and rejects multi-batch
payloads until the public `ActionScoreResult` contract defines batched score semantics.

Current `score_info` keys:

- `observation`: tensor-like or nested numeric observation payload accepted by the upstream model.
- `goal`: tensor-like or nested numeric goal payload accepted by the upstream model.
- `objective`: optional, `l2` by default; `l1` is also supported.
- `actions_are_normalized`: optional, `true` by default. Set `false` only when the loaded
  preprocessor exposes `normalize_actions(...)`.

## Remaining Upstream Contract To Define

- Mapping between JEPA-WMS candidate tensors and WorldForge `Action` sequences.
- Provider-specific limits such as context window, rollout horizon, action dimension, batch size,
  device placement, dataset/task family, and checkpoint compatibility.
- Failure modes for missing checkpoints, unsupported task configs, malformed tensor inputs,
  non-finite model outputs, unavailable simulator assets, and optional dependency failures.

## Tests

- `tests/test_jepa_wms_provider.py` covers the fake-runtime happy path, provider contract helper,
  missing model path, missing runtime, upstream error payloads, malformed fixtures, non-finite
  score output, score-count mismatches, torch-hub runtime delegation, score-planning through
  `World.plan(...)`, and success/failure event emission.
- `tests/fixtures/providers/jepa_wms_*.json` defines the current contract fixtures.

## Release Checklist

- [ ] Provider capabilities are narrow and truthful.
- [x] Provider profile metadata is complete for the fake-runtime candidate state.
- [x] Public API docs mention current failure modes.
- [x] `docs/src/providers/README.md` links this provider page.
- [x] `AGENTS.md` documents current dependencies and gotchas.
- [x] `CHANGELOG.md` records the user-visible behavior.
