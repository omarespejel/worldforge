# Examples And CLI Commands

Use the CLI index for the current runnable examples:

```bash
uv run worldforge examples
uv run worldforge examples --format json
```

## Checkout Scripts

| Example | Command | Purpose |
| --- | --- | --- |
| Basic prediction | `uv run python examples/basic_prediction.py` | Create a mock world, predict, plan, and print an evaluation report. |
| Cross-provider compare | `uv run python examples/cross_provider_compare.py` | Register a second deterministic provider and compare prediction outputs. |

## Packaged Demos

| Demo | Command | Runtime boundary |
| --- | --- | --- |
| LeWorldModel score planning | `uv run worldforge-demo-leworldmodel` | Uses `LeWorldModelProvider` with an injected deterministic cost runtime. |
| LeRobot policy-plus-score planning | `uv run worldforge-demo-lerobot` | Uses `LeRobotPolicyProvider` with an injected deterministic policy runtime. |

Both packaged demos validate the WorldForge adapter, planning, execution, persistence, reload, and
event path in a clean checkout. They do not install optional ML runtimes or run upstream neural
checkpoint inference.

## Operational Commands

```bash
uv run worldforge doctor
uv run worldforge provider list
uv run worldforge provider info mock
uv run worldforge predict kitchen --provider mock --x 0.3 --y 0.8 --z 0.0 --steps 2
uv run worldforge eval --suite planning --provider mock --format json
uv run worldforge benchmark --provider mock --iterations 5 --format json
```
