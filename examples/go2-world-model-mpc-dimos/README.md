# Go2 World-Model MPC + DimOS Shadow Bridge

Run a checkout-safe one-step MPC demo over the public
`espejelomar/go2-air-controlbench-v1` dataset. The demo fits a transparent
deadband-aware command-outcome world model, ranks bounded Unitree Sport Move
candidate commands against target motions, and writes a DimOS shadow-bridge plan
plus validated DecisionTrace artifacts.

This example does not import DimOS, connect to a robot, start a simulator, or send
hardware commands. DimOS is represented as the host-owned runtime and memory
adapter that would execute a bounded Sport Move only after separate live gates.

## Run

```bash
uv run python examples/go2-world-model-mpc-dimos/run.py \
  --out .worldforge/go2-world-model-mpc-dimos
```

Use a local public-preview CSV instead of fetching Hugging Face:

```bash
uv run python examples/go2-world-model-mpc-dimos/run.py \
  --dataset-csv <path-to-all_trials_normalized.csv> \
  --out .worldforge/go2-world-model-mpc-dimos
```

Expected success signal:

- `.worldforge/go2-world-model-mpc-dimos/world-model-mpc-summary.json` exists.
- `.worldforge/go2-world-model-mpc-dimos/world-model-mpc-report.md` exists.
- `.worldforge/go2-world-model-mpc-dimos/dimos-shadow-bridge-plan.json` exists.
- One `decision-trace-go2-world-model-mpc-*.json` file exists per target and
  validates as DecisionTrace v1.

## Boundary

This is an offline ControlBench decision demo. It proves that WorldForge can use
measured Go2 outcomes to rank candidate commands and emit DimOS-shaped evidence.
It does not prove autonomous Go2 control, live DimOS execution, learned model
superiority, Cosmos 3 usefulness, safety certification, or external-GT-grade
position accuracy.
