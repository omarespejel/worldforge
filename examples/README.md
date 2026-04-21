# Examples

Runnable examples are split into checkout scripts and packaged console commands.

Use the CLI index for the current command list and JSON metadata:

```bash
uv run worldforge examples
uv run worldforge examples --format json
```

## Visual Harness

| Example | Surface | Command |
| --- | --- | --- |
| `theworldharness` | E2E flows, provider diagnostics, benchmark comparison | `uv run --extra harness worldforge-harness` |

```bash
uv run --extra harness worldforge-harness
uv run --extra harness worldforge-harness --flow lerobot
uv run --extra harness worldforge-harness --flow diagnostics
```

TheWorldHarness is optional and depends on Textual through the `harness` extra. It currently
includes score-planning, policy-plus-score planning, and provider diagnostics plus benchmark
comparison flows.

## Prediction And Evaluation

| Example | Surface | Command |
| --- | --- | --- |
| `basic-prediction` | prediction, planning, evaluation | `uv run python examples/basic_prediction.py` |

## Provider Comparison

| Example | Surface | Command |
| --- | --- | --- |
| `cross-provider-compare` | provider registration, comparison | `uv run python examples/cross_provider_compare.py` |

## Score Planning

| Example | Surface | Command |
| --- | --- | --- |
| `leworldmodel-score-planning` | score provider, planning, persistence | `uv run worldforge-demo-leworldmodel` |

## Policy Plus Score Planning

| Example | Surface | Command |
| --- | --- | --- |
| `lerobot-policy-score-planning` | policy provider, score provider, planning, persistence | `uv run worldforge-demo-lerobot` |

## Optional Runtime Smoke

| Example | Surface | Command |
| --- | --- | --- |
| `leworldmodel-real-checkpoint-smoke` | real checkpoint smoke | `scripts/lewm-real --checkpoint ~/.stable-wm/pusht/lewm_object.ckpt --device cpu` |

## Runtime Boundary

The packaged demos use real WorldForge provider surfaces with injected deterministic runtimes. They
are intended to verify the framework path in a clean checkout.

- `worldforge-demo-leworldmodel` exercises `LeWorldModelProvider`, score planning, execution,
  persistence, and reload without installing `stable_worldmodel`, torch, or checkpoints.
- `worldforge-demo-lerobot` exercises `LeRobotPolicyProvider`, policy-plus-score planning,
  execution, persistence, and reload without installing LeRobot, torch, or policy checkpoints.

Optional live smoke scripts are separate because they require host-owned model runtimes,
credentials, checkpoints, robot observations, or action translators.
