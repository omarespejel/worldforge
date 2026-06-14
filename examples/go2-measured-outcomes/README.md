# Go2 Measured Outcomes

Sanitize a host-owned Unitree Go2 native-rate system-identification capture into a
DecisionTrace-compatible `real_measured` artifact.

This example is intentionally private-data aware: it reads local capture summaries, writes
sanitized artifacts under `.worldforge/`, and does not copy raw telemetry, LiDAR sidecars, RGB
frames, IP addresses, serial numbers, or host-local paths into the generated trace.

## Run

```bash
uv run python examples/go2-measured-outcomes/run.py \
  --capture-dir <capture-dir> \
  --out .worldforge/go2-measured-outcomes
```

Expected success signal:

- `.worldforge/go2-measured-outcomes/decision-trace-go2-real-measured.json` exists.
- `.worldforge/go2-measured-outcomes/go2-measured-outcomes-report.md` exists.
- The trace validates as `DecisionTrace v1` with `outcome_kind=real_measured`,
  `score_kind=measured_metric`, and `hardware_executed=true`.
- Raw Go2 capture files remain outside the repository.

First triage step: if the command raises `WorldForgeError` mentioning
`command_group_stats`, confirm the capture directory contains
`analysis/command_group_stats.csv` with non-empty command group rows. If it mentions missing
`ROBOTODOM` or `ULIDAR_ARRAY` stream references/messages, confirm
`streams/topics/robotodom.jsonl` and `streams/topics/ulidar_array.jsonl` exist and the capture
summary `topic_counts` for those topics are nonzero.

## Boundary

This is measured-outcome evidence, not an autonomous planner result. The operator/script executed
the command grid on hardware; WorldForge ranks the measured command-tracking outcomes after the
fact so future planners and scorers can compare against real Go2 response.
