# DimOS Go2 Trace Judge

Experimental checkout-safe example for
[issue #329](https://github.com/AbdelStark/worldforge/issues/329).

This example shows how a host-owned DimOS + Unitree Go2 integration can use WorldForge as an
inspectable decision/evidence layer without making WorldForge responsible for robot networking,
controller execution, or physical safety.

```text
DimOS observes/maps/exposes robot skills
Host proposes candidate robot actions
WorldForge scores and records the decision trace
DimOS executes only if the host chooses to execute
```

The checked-in runner uses mocked, sanitized DimOS-style inputs. It does not import DimOS, connect
to a robot, read credentials, or execute robot commands.

## Run

```bash
uv run python examples/dimos-go2-trace-judge/app.py run
```

The default output directory is:

```text
.worldforge/dimos-go2-trace-judge/
```

For a deterministic sample run:

```bash
uv run python examples/dimos-go2-trace-judge/app.py run \
  --run-id dimos-go2-sample \
  --output-dir examples/dimos-go2-trace-judge/sample_run
```

Expected: exit code `0`, with six JSON artifacts plus `report.md` written to the selected output
directory. For the default command, inspect `.worldforge/dimos-go2-trace-judge/run_manifest.json`.
For the deterministic sample command, inspect
`examples/dimos-go2-trace-judge/sample_run/run_manifest.json`.

If the run fails before writing artifacts, check the printed validation error. If artifacts exist
but the result looks wrong, start with `run_manifest.json`, then compare `candidate_scores.json`
and `selected_action.json`.

Run against a venue input file:

```bash
uv run python examples/dimos-go2-trace-judge/app.py run \
  --input-json examples/dimos-go2-trace-judge/fixtures/venue_input.sample.json \
  --output-dir .worldforge/dimos-go2-trace-judge/venue-smoke
```

`--input-json` must contain `observation_summary`, `task`, and `candidates[]`. The optional
`--goal` flag overrides `task.human_goal` while preserving the rest of the task representation.

## Artifacts

The runner writes:

| Artifact | Purpose |
| --- | --- |
| `score_info.json` | The exact score-provider input: task, observation summary, host runtime, and action candidates. |
| `observation_summary.json` | Sanitized host observation summary: pose, costmap, visual target, navigation state. |
| `candidate_scores.json` | Candidate actions, transparent features, WorldForge score result, selected candidate. |
| `selected_action.json` | The action the host could execute through DimOS. |
| `outcome_after_execution.json` | Mocked post-action outcome shape for future learned-score training. |
| `run_manifest.json` | Issue-safe manifest with capability, safety boundary, and artifact paths. |
| `report.md` | Human-readable decision trace. |

See [trace_schema.md](./trace_schema.md) for the JSON shape.

## What This Proves

- WorldForge can rank host-proposed robot action candidates through the `score` capability.
- The decision can be preserved as JSON-native evidence before any physical robot execution.
- The same trace shape can become future training data for a learned scorer:
  `observation + goal + candidate action -> outcome quality`.

## What This Does Not Claim

- No new Go2 world model is claimed.
- WorldForge is not claiming control or safety certification for the robot.
- DimOS is not added as a WorldForge dependency.
- Robot IPs, credentials, venue labels, and private operator details are not exposed.
- Generated video is not treated as score evidence.

## Host-Owned Live Integration Sketch

A live host adapter can translate DimOS state into this example's input shape:

```text
DimOS map/costmap/frontiers/visual detections
  -> observation_summary.json

DimOS MCP or host planner action proposals
  -> candidates[]

WorldForge TransparentGo2ScoreProvider
  -> candidate_scores.json + selected_action.json

Host-side DimOS controller
  -> optional execution + outcome_after_execution.json
```

The host must own operator supervision, emergency stop, velocity limits, obstacle avoidance,
networking, authentication, and final execution approval. WorldForge remains the decision trace and
scoring surface.

## Fork-Only Live Bridge

For hackathon testing, `live_dimos_bridge.py` adds a host-owned DimOS CLI wrapper on top of the
offline trace judge. It shells out to the documented `dimos mcp` CLI and keeps execution gated.
This is useful for proving connectivity before touching the real robot.

Safe probe:

```bash
uv run python examples/dimos-go2-trace-judge/live_dimos_bridge.py probe
```

Expected: exit code `0`, `probe.json`, `bridge_manifest.json`, and `report.md` written under
`.worldforge/dimos-go2-live-bridge/`. If the DimOS CLI or MCP server is missing, the command still
writes artifacts with failed command results so the operator can inspect what is unavailable.

Dry-run the selected WorldForge action as a DimOS MCP command:

```bash
uv run python examples/dimos-go2-trace-judge/live_dimos_bridge.py dry-run-selected --with-probe
```

Expected: the offline trace judge runs, `selected_mcp_command.json` is written, and
`will_execute` is `false`. If this fails, inspect `bridge_manifest.json` first, then run the
offline trace judge by itself to confirm the score path still works before debugging DimOS.

Dry-run against a venue input file:

```bash
uv run python examples/dimos-go2-trace-judge/live_dimos_bridge.py dry-run-selected \
  --input-json .worldforge/dimos-go2-live-bridge/venue/venue_input.json \
  --with-probe
```

Live execution is intentionally harder:

```bash
WORLDFORGE_DIMOS_ENABLE_EXECUTE=1 \
uv run python examples/dimos-go2-trace-judge/live_dimos_bridge.py execute-selected \
  --confirm LIVE_DIMOS_GO2_EXECUTE
```

Execution is bounded to a narrow allowlist (`relative_move` and `wait`) with conservative parameter
limits. It should only be used with the robot on the floor, operator supervision, venue approval,
and an emergency stop path available. This bridge is fork-only until the live assumptions are
validated. A successful run exits `0` and writes `execution_result.json`, `bridge_manifest.json`,
and `report.md`. If execution is blocked or fails, first verify the environment variable and
confirmation flag, then inspect `execution_result.json` and `bridge_manifest.json` for details.

## Venue Input Collector

`collect_venue_input.py` prepares robot-time artifacts without moving the robot. It safely runs the
same DimOS MCP probe commands as the live bridge:

```bash
uv run python examples/dimos-go2-trace-judge/collect_venue_input.py \
  --output-dir .worldforge/dimos-go2-live-bridge/venue
```

Expected: `venue_probe.json`, `venue_input.json`, `run_manifest.json`, and `report.md`. The starter
`venue_input.json` is intentionally editable. Before live execution, replace the sample feature
values with current venue observations such as target bearing, obstacle risk, expected progress,
frontier count, stuck risk, and information gain.

The robot-time sequence should be:

```bash
uv run python examples/dimos-go2-trace-judge/live_dimos_bridge.py probe \
  --output-dir .worldforge/dimos-go2-live-bridge/venue-probe

uv run python examples/dimos-go2-trace-judge/collect_venue_input.py \
  --output-dir .worldforge/dimos-go2-live-bridge/venue

uv run python examples/dimos-go2-trace-judge/live_dimos_bridge.py dry-run-selected \
  --input-json .worldforge/dimos-go2-live-bridge/venue/venue_input.json \
  --with-probe \
  --output-dir .worldforge/dimos-go2-live-bridge/venue-dry-run

WORLDFORGE_DIMOS_ENABLE_EXECUTE=1 \
uv run python examples/dimos-go2-trace-judge/live_dimos_bridge.py execute-selected \
  --input-json .worldforge/dimos-go2-live-bridge/venue/venue_input.json \
  --confirm LIVE_DIMOS_GO2_EXECUTE \
  --output-dir .worldforge/dimos-go2-live-bridge/venue-execute
```

The collector is grounded in the DimOS Go2 source/docs that expose `unitree-go2-agentic`,
`dimos mcp list-tools`, `dimos mcp call relative_move`, Rerun visualization, costmap navigation,
and frontier exploration. It still keeps DimOS as a host-owned runtime and does not import DimOS
into WorldForge.

## Related Work

- Related to [#274](https://github.com/AbdelStark/worldforge/issues/274) for adopt-your-robot
  guided demos.
- Related to [#328](https://github.com/AbdelStark/worldforge/issues/328) for caller-owned
  candidate generation and score-based planning.
- Related to [#321](https://github.com/AbdelStark/worldforge/issues/321) for real hardware
  showcase boundaries.

This is intentionally experimental and open to maintainer changes.
