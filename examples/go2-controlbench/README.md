# Go2 ControlBench

Run a small benchmark over a host-owned Unitree Go2 Air native-rate system-identification capture.
The benchmark evaluates command-to-outcome baselines against real measured odometry without
copying raw telemetry, LiDAR sidecars, RGB frames, IP addresses, serial numbers, or host-local
paths into generated public artifacts.

## Run

```bash
uv run python examples/go2-controlbench/run.py \
  --capture-dir <capture-dir> \
  --out .worldforge/go2-controlbench
```

Expected success signal:

- `.worldforge/go2-controlbench/controlbench-summary.json` exists.
- `.worldforge/go2-controlbench/controlbench-report.md` exists.
- `.worldforge/go2-controlbench/decision-trace-go2-controlbench.json` exists and validates as
  `DecisionTrace v1`.
- The report lists Task A command-outcome prediction, Task B inverse-control ranking, and Task C
  external-ground-truth status.

## Boundary

This is a benchmark over an operator-scripted hardware capture, not autonomous Go2 control. The
baselines are deterministic command-to-outcome estimates. The generated DecisionTrace overlay
records how a baseline ranked command candidates against a real measured target and what regret
was observed from measured outcomes.
