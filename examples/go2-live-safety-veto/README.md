# Go2 Live Safety Veto Shadow Demo

This checkout-safe example fits the transparent Go2 ControlBench command-outcome model from the
public `espejelomar/go2-air-controlbench-v1` normalized trial table, scores a proposed
`forward_50cm` command against live-shaped obstacle evidence, and emits:

- `live-safety-veto-summary.json`
- `dimos-live-safety-bridge-plan.json`
- `decision-trace-go2-live-safety-veto.json`
- `live-safety-veto-report.md`

Run the default veto-shaped fixture:

```bash
uv run worldforge-demo-go2-live-safety-veto
```

Run the two-case shadow demo pair:

```bash
uv run worldforge-demo-go2-live-safety-veto --demo-pair
```

Run an open-space shadow authorization:

```bash
uv run worldforge-demo-go2-live-safety-veto \
  --forward-clearance-m 2.0 \
  --stopmove-verified
```

Runtime boundary: this example does not import DimOS, connect to a robot, send Unitree commands,
or use Cosmos 3. It prepares the DecisionTrace and DimOS bridge contract only. A host-owned live
demo must re-check telemetry freshness, obstacle evidence, StopMove, operator approval, and the
bounded command envelope before any execution.
