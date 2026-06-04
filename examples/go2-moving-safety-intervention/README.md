# Go2 Moving Safety Intervention

This checkout-safe example fits the transparent Go2 ControlBench deadband-affine command-outcome
model, computes a conservative stopping envelope for tiny forward Sport Move chunks, and emits a
chained DecisionTrace sequence:

```text
moving_clear -> obstacle_veto -> hold_blocked -> resume_clear
```

Run the default moving-intervention shadow sequence:

```bash
uv run worldforge-demo-go2-moving-safety-intervention
```

Run only the stop-proof planning rung:

```bash
uv run worldforge-demo-go2-moving-safety-intervention --mode stop-proof
```

Run with an empirical stop-proof measurement supplied by a host:

```bash
uv run worldforge-demo-go2-moving-safety-intervention \
  --mode moving-intervention \
  --stop-proof-distance-m 0.03 \
  --stop-proof-time-s 0.22 \
  --command-rtt-ms 80
```

Runtime boundary: this example does not import DimOS, connect to the Go2, send Unitree commands,
or use Cosmos 3. DimOS remains the host-owned runtime for odom, LiDAR/costmap, positive heartbeat,
`Move`, and `StopMove`. WorldForge owns candidate scoring, authorization/veto, and DecisionTrace
evidence. A live host must run the ladder before any moving recording: dry-run, stop-proof,
open-space bounded motion, then obstacle intervention.
