# Examples

Runnable examples are split into checkout scripts and packaged console commands.

Use the CLI index when you need the current command list:

```bash
uv run worldforge examples
uv run worldforge examples --format json
```

| Example | Surface | Command |
| --- | --- | --- |
| `basic-prediction` | prediction, planning, evaluation | `uv run python examples/basic_prediction.py` |
| `cross-provider-compare` | provider registration, comparison | `uv run python examples/cross_provider_compare.py` |
| `leworldmodel-score-planning` | score provider, planning, persistence | `uv run worldforge-demo-leworldmodel` |
| `lerobot-policy-score-planning` | policy provider, score provider, planning, persistence | `uv run worldforge-demo-lerobot` |

## Runtime Boundary

The packaged demos use real WorldForge provider surfaces with injected deterministic runtimes. They
are intended to verify the framework path in a clean checkout.

- `worldforge-demo-leworldmodel` exercises `LeWorldModelProvider`, score planning, execution,
  persistence, and reload without installing `stable_worldmodel`, torch, or checkpoints.
- `worldforge-demo-lerobot` exercises `LeRobotPolicyProvider`, policy-plus-score planning,
  execution, persistence, and reload without installing LeRobot, torch, or policy checkpoints.

Optional live smoke scripts are separate because they require host-owned model runtimes,
credentials, checkpoints, robot observations, or action translators.
