# LeWorldModel Provider

Capability: `score`

Taxonomy category: JEPA latent predictive world model

`leworldmodel` wraps LeWorldModel's `stable_worldmodel.policy.AutoCostModel` surface. It ranks
candidate action sequences from task-shaped observations, goals, and actions. WorldForge models it
as an action scorer because the upstream runtime returns costs; it does not generate video, answer
text questions, or mutate WorldForge state directly.

```text
pixels + action history + goal + candidate actions
  -> LeWorldModel cost model
  -> ActionScoreResult(scores, best_index, lower_is_better=True)
```

## Runtime Ownership

WorldForge owns provider registration, input validation, score-result validation, planning
selection, metadata, and provider events.

The host owns:

- `stable_worldmodel` and torch installation
- checkpoint download, extraction, and compatibility
- `$STABLEWM_HOME` or `LEWORLDMODEL_CACHE_DIR`
- preprocessing sensor data into checkpoint-shaped `pixels`, `goal`, and `action` tensors
- mapping model-native action candidates back to WorldForge `Action` sequences

WorldForge does not add LeWorldModel, torch, checkpoint archives, or datasets to its base package.

## Configuration

- `LEWORLDMODEL_POLICY` or `LEWM_POLICY`: required for auto-registration. Value is the checkpoint
  run name relative to `$STABLEWM_HOME`, without the `_object.ckpt` suffix. Example: `pusht/lewm`.
- `LEWORLDMODEL_CACHE_DIR`: optional checkpoint root override.
- `LEWORLDMODEL_DEVICE`: optional torch device string such as `cpu`, `cuda`, or `cuda:0`.

Programmatic construction can inject a loader and tensor module for tests or host-owned runtimes:

```python
from worldforge.providers import LeWorldModelProvider

provider = LeWorldModelProvider(
    policy="pusht/lewm",
    cache_dir="/models/stable-wm",
    device="cpu",
)
```

## Input Contract

`score_actions(...)` requires:

```python
result = forge.score_actions(
    "leworldmodel",
    info={
        "pixels": pixels,
        "goal": goal,
        "action": action_history,
    },
    action_candidates=action_candidate_tensor,
)
```

Validation rules:

- `info` must be a JSON object.
- `info["pixels"]`, `info["goal"]`, and `info["action"]` are required.
- Required info fields must be tensors or rectangular nested numeric arrays with at least three
  dimensions.
- `action_candidates` must be a tensor or rectangular nested numeric array shaped as
  `(batch, samples, horizon, action_dim)`.
- Returned costs must flatten to finite numeric scores.
- The number of returned scores must match the scored candidate set.

Scores are costs: lower values are better. WorldForge sets `best_index` to the lowest-cost
candidate unless the provider returns a validated explicit index.

## Planning

Score planning keeps WorldForge actions separate from model-native tensors:

```python
from worldforge import Action

plan = world.plan(
    goal="select the lowest-cost LeWorldModel candidate",
    provider="leworldmodel",
    planner="leworldmodel-mpc",
    candidate_actions=[
        [Action.move_to(0.1, 0.5, 0.0)],
        [Action.move_to(0.4, 0.5, 0.0)],
    ],
    score_info={
        "pixels": pixels,
        "goal": goal_pixels,
        "action": action_history,
    },
    score_action_candidates=action_candidate_tensor,
    execution_provider="mock",
)
```

The selected `Plan.actions` come from `candidate_actions[best_index]`. `Plan.predicted_states`
stays empty because a score provider ranks candidate futures without returning a WorldForge state
rollout. Use `execute_plan(...)` with an execution provider that supports `predict`.

## Demo And Smoke

Checkout-safe provider/planner demo:

```bash
uv run worldforge-demo-leworldmodel
uv run worldforge-demo-leworldmodel --json-only
```

This injects a deterministic LeWorldModel-compatible cost runtime. It validates WorldForge's
provider, score-planning, execution, persistence, and reload path without loading an upstream
checkpoint.

Real-checkpoint smoke:

```bash
scripts/lewm-real \
  --checkpoint ~/.stable-wm/pusht/lewm_object.ckpt \
  --device cpu
```

Equivalent explicit `uv` command:

```bash
uv run --python 3.10 \
  --with "stable-worldmodel[train] @ git+https://github.com/galilai-group/stable-worldmodel.git" \
  --with "datasets>=2.21" \
  lewm-real \
    --checkpoint ~/.stable-wm/pusht/lewm_object.ckpt \
    --device cpu
```

`scripts/lewm-real` is the complete uv-backed task: it installs the host-owned upstream runtime in
uv's ephemeral environment, then runs the short `lewm-real` console alias. The live smoke prints a
visual pipeline by default: what the run demonstrates, checkpoint resolution, runtime preflight,
deterministic synthetic tensor construction, real `score_actions` inference, ranked candidate
costs, score statistics, provider events, and latency metrics. Pass `--json-only` when a script
needs only the machine-readable payload, or `--json-output lewm-real-summary.json` to keep the
visual terminal output and also write the run data.

The input tensors are synthetic PushT-shaped tensors. The smoke demonstrates checkpoint loading,
provider health, WorldForge's LeWorldModel tensor contract, real cost-model scoring, and candidate
ranking. It does not claim task-specific image preprocessing quality or robot execution.

The smoke requires an extracted object checkpoint such as
`~/.stable-wm/pusht/lewm_object.ckpt`. If you have Hugging Face LeWM `config.json` and
`weights.pt` assets instead, build the object checkpoint first:

```bash
uv run --python 3.10 \
  --with "stable-worldmodel[train] @ git+https://github.com/galilai-group/stable-worldmodel.git" \
  --with "datasets>=2.21" \
  --with huggingface_hub \
  worldforge-build-leworldmodel-checkpoint \
  --stablewm-home ~/.stable-wm \
  --policy pusht/lewm
```

Real LeRobot + LeWorldModel robotics showcase:

```bash
scripts/robotics-showcase
```

This showcase uses `LeWorldModelProvider` as the score half of a real policy-plus-score plan. It
loads the default PushT LeRobot policy, reads the default `pusht/lewm` object checkpoint, builds
PushT score tensors through the upstream environment, and ranks packaged action candidates through
WorldForge. For a non-PushT task, use `scripts/lewm-lerobot-real --help` and provide task-aligned
`pixels`, `goal`, `action`, and `action_candidates` tensors. WorldForge does not infer LeWorldModel
preprocessing from LeRobot output. If the policy action chunk is not already checkpoint-compatible,
provide a task-specific `--candidate-builder`.

## Failure Modes

- Missing `LEWORLDMODEL_POLICY` and `LEWM_POLICY` leaves the provider unregistered.
- Missing `torch` or `stable_worldmodel.policy.AutoCostModel` is reported by `health()`.
- Checkpoint loading failures are wrapped in `ProviderError`.
- Missing required `pixels`, `goal`, or `action` fields fail before model invocation.
- Ragged nested arrays, non-finite values, or low-rank tensors fail before model invocation.
- Non-four-dimensional action candidates fail before model invocation.
- Non-finite model outputs fail before `ActionScoreResult` is returned.
- Score planning fails if the score result cannot identify an in-range candidate.

## Tests

- `tests/test_leworldmodel_provider.py` covers input validation, score outputs, provider health,
  event emission, and planning integration.
- `tests/test_leworldmodel_e2e_demo.py` covers the checkout-safe end-to-end demo.
- `tests/test_leworldmodel_smoke_script.py` and `tests/test_leworldmodel_uv_tasks.py` cover smoke
  command parsing and checkpoint-builder behavior without requiring a real checkpoint.
- `tests/test_lerobot_leworldmodel_smoke_script.py` and `tests/test_robotics_showcase.py` cover
  the combined LeRobot + LeWorldModel runner and showcase defaults without requiring optional
  runtimes.

## Primary References

- [LeWorldModel paper](https://arxiv.org/abs/2603.19312)
- [LeWorldModel project page](https://le-wm.github.io/)
- [LeWorldModel code](https://github.com/lucas-maes/le-wm)
