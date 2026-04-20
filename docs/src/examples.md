# Examples And CLI Commands

Use the CLI index for the current runnable examples and optional smoke paths:

```bash
uv run worldforge examples
uv run worldforge examples --format json
```

## Prediction And Evaluation

| Example | Command | Purpose |
| --- | --- | --- |
| `basic-prediction` | `uv run python examples/basic_prediction.py` | Create a mock world, predict, plan, and print a physics evaluation report. |

## Provider Comparison

| Example | Command | Purpose |
| --- | --- | --- |
| `cross-provider-compare` | `uv run python examples/cross_provider_compare.py` | Register a second deterministic provider and compare prediction outputs. |

## Score Planning

| Example | Command | Runtime boundary |
| --- | --- | --- |
| `leworldmodel-score-planning` | `uv run worldforge-demo-leworldmodel` | Uses `LeWorldModelProvider` with an injected deterministic cost runtime. |

## Policy Plus Score Planning

| Example | Command | Runtime boundary |
| --- | --- | --- |
| `lerobot-policy-score-planning` | `uv run worldforge-demo-lerobot` | Uses `LeRobotPolicyProvider` with an injected deterministic policy runtime. |

Both packaged demos validate the WorldForge adapter, planning, execution, persistence, reload, and
event path in a clean checkout. They do not install optional ML runtimes or run upstream neural
checkpoint inference.

## Optional Runtime Smoke

| Example | Command | Runtime boundary |
| --- | --- | --- |
| `leworldmodel-real-checkpoint-smoke` | `uv run --python 3.10 --with "stable-worldmodel[train,env] @ git+https://github.com/galilai-group/stable-worldmodel.git" --with "datasets>=2.21" worldforge-smoke-leworldmodel` | Requires host-owned `stable_worldmodel`, torch, datasets, and LeWM checkpoint assets. |

## Operational Commands

```bash
uv run worldforge doctor
uv run worldforge provider list
uv run worldforge provider docs
uv run worldforge provider info mock
uv run worldforge predict kitchen --provider mock --x 0.3 --y 0.8 --z 0.0 --steps 2
uv run worldforge eval --suite planning --provider mock --format json
uv run worldforge benchmark --provider mock --iterations 5 --format json
```
