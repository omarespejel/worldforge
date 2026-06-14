# Examples

Runnable examples are split into checkout scripts and packaged console commands.

Use the CLI index for the current command list and JSON metadata:

```bash
uv run worldforge examples
uv run worldforge examples --format json
```

## Robotics Showcase TUI

| Example | Surface | Command |
| --- | --- | --- |
| `robotics-showcase` | real LeRobot policy plus LeWorldModel score report | `scripts/robotics-showcase` |

```bash
scripts/robotics-showcase
scripts/robotics-showcase --no-tui
```

The robotics showcase report is optional and depends on Textual through the `harness` extra. The
terminal and JSON paths remain available with `--no-tui`.

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

## Robot Decision Traces

| Example | Surface | Command |
| --- | --- | --- |
| `so101-replay-trace` | score provider, decision trace, counterfactuals, mock replay | `uv run worldforge-demo-so101-replay-trace` |
| `go2-controlbench-decisiontrace` | score provider, public decision trace, measured regret | `uv run python examples/go2-controlbench-decisiontrace/run.py` |
| `go2-world-model-mpc-dimos` | Go2 ControlBench world-model MPC, DimOS shadow bridge, DecisionTrace | `uv run worldforge-demo-go2-world-model-mpc` |
| `go2-live-safety-veto` | Go2 predictive safety-filter shadow gate, DimOS bridge contract, DecisionTrace | `uv run worldforge-demo-go2-live-safety-veto` |

The SO-101 replay trace demo uses a deterministic pick-and-place decision point shaped after the
public `lerobot/svla_so101_pickplace` metadata. It scores candidate 6D joint-action futures,
selects the lowest-cost manipulation action, executes the selected object placement through the
local mock provider, and emits an `observation -> goal -> candidate_actions -> candidate_scores ->
selected_action -> outcome -> counterfactuals` trace without installing LeRobot, torch, DimOS, or
connecting robot hardware.

The Go2 ControlBench DecisionTrace demo consumes a compact public trace from
`espejelomar/go2-air-controlbench-v1`. It reranks candidate Unitree Go2 sport-mode commands through
WorldForge's `score` capability, preserves the score-selected action, exposes the measured
counterfactual that would have done better, and reports native-odometry regret without importing
DimOS, Unitree SDKs, Hugging Face datasets, or controlling hardware.

The Go2 world-model MPC plus DimOS shadow bridge example consumes the public
`espejelomar/go2-air-controlbench-v1` normalized trial table at pinned revision
`bb80a77c63f0b502267e5500fef83e65535ced5b`, fits a transparent deadband-aware command-outcome
world model, ranks bounded Sport Move candidates for target body motions, and emits
`world-model-mpc-summary.json`, `dimos-shadow-bridge-plan.json`, and one DecisionTrace per target.
DimOS is represented as a host-owned shadow runtime and memory bridge; the example does not import
DimOS, connect to a robot, or send live hardware commands. The default remote CSV fetch is limited
to the public Hugging Face dataset host, capped to the expected CSV size, and redacted from emitted
artifacts. First triage step: rerun with a local `--dataset-csv all_trials_normalized.csv` from
revision `bb80a77c63f0b502267e5500fef83e65535ced5b`, then confirm the summary still reports
`shadow_replay_no_execution` and `live_execution_allowed=false`. Success signal:
`world-model-mpc-summary.json` contains those two values and the run emitted a non-empty
DecisionTrace JSON file for each target.

The Go2 live safety-veto shadow example scores a proposed `forward_50cm` command against
live-shaped obstacle evidence and emits `live-safety-veto-summary.json`,
`dimos-live-safety-bridge-plan.json`, and `decision-trace-go2-live-safety-veto.json`. The default
fixture rejects the forward action and selects `stop_hold`; an open-space shadow path is available
with `--forward-clearance-m 2.0 --stopmove-verified`. For a two-case terminal run, use
`uv run worldforge-demo-go2-live-safety-veto --demo-pair` to run the obstacle-veto and
open-space-control cases back to back. It does not import DimOS, connect to hardware, send robot
commands, use Cosmos 3, call `relative_move`, or claim certified safety.

## Service Host Reference

| Example | Surface | Command |
| --- | --- | --- |
| `service-host` | HTTP liveness/readiness, provider diagnostics, and mock prediction workflow | `uv run python examples/hosts/service/app.py --provider mock --port 8080` |

The service host uses Python's stdlib HTTP server so it does not add a base dependency. It is a
reference for embedding WorldForge in a host process, not a WorldForge deployment boundary:
production services still own authentication, alerting, telemetry export, upstream SLA handling,
and artifact retention.

## Batch Evaluation Host

| Example | Surface | Command |
| --- | --- | --- |
| `batch-eval-host` | eval/benchmark jobs, run manifests, budget exits | `uv run python examples/hosts/batch-eval/app.py benchmark --provider mock` |

The batch host uses only the base WorldForge package and Python stdlib. It writes
`.worldforge/batch-eval/runs/<run-id>/run_manifest.json`, report exports, and copied benchmark
input or budget files. A benchmark budget violation exits `1` after preserving the failed run.

## Robotics Operator Host

| Example | Surface | Command |
| --- | --- | --- |
| `robotics-operator-host` | offline policy+score review, dry-run approval, replay artifacts | `uv run python examples/hosts/robotics-operator/app.py review --sample-translator` |

The robotics operator host is non-mutating by default and never talks to robot controllers. It
requires an explicit action translator, records host-owned checklist and dry-run approval state, and
writes selected action chunks, score rationale, provider events, and a replay artifact under
`.worldforge/robotics-operator/runs/<run-id>/`. Controller execution is only an application hook:
WorldForge does not certify robot safety, controller semantics, interlocks, or deployment readiness.

Deployment recipes for the service host, batch evaluation host, and robotics operator host live in
the public examples documentation. Each recipe includes an env template, process command, readiness
command, smoke command, logging command, evidence export command, expected success signal, first
triage step, and owned boundary for checkout-safe, prepared-host, credentialed, GPU-bound, and
robotics-lab paths.

## Optional Runtime Smoke

| Example | Surface | Command |
| --- | --- | --- |
| `leworldmodel-real-checkpoint-smoke` | real checkpoint smoke | `scripts/lewm-real --checkpoint ~/.stable-wm/pusht/lewm_object.ckpt --device cpu` |
| `lerobot-leworldmodel-real-robotics` | real policy plus real world-model scoring replay | `scripts/robotics-showcase` |

The explicit command behind the wrapper is:

```bash
uv run --python 3.13 \
  --with "stable-worldmodel[train] @ git+https://github.com/galilai-group/stable-worldmodel.git" \
  --with "datasets>=2.21" \
  --with "opencv-python" \
  --with "imageio" \
  lewm-real \
    --checkpoint ~/.stable-wm/pusht/lewm_object.ckpt \
    --device cpu
```

It runs real LeWorldModel checkpoint scoring over deterministic synthetic PushT-shaped tensors
through the official `stable_worldmodel.policy.AutoCostModel` loading path and prints a visual
pipeline, tensor shapes, latency metrics, provider events, and ranked candidate costs.

The real robotics showcase composes LeRobot and LeWorldModel through WorldForge policy-plus-score
planning with a packaged PushT bridge. The wrapper opens a Textual report by default and writes the
same run summary to `/tmp/worldforge-robotics-showcase/real-run.json`. The report reveals each stage
in sequence, includes an illustrative animated robot-arm replay, and gives candidate ranking plus
tabletop replay their own full-width sections. Use `--no-tui` for the plain terminal report.

```bash
scripts/robotics-showcase
```

Use the lower-level runner when bringing a different task observation, score tensor source, or
candidate bridge:

```bash
scripts/lewm-lerobot-real \
  --policy-path lerobot/diffusion_pusht \
  --policy-type diffusion \
  --checkpoint ~/.stable-wm/pusht/lewm_object.ckpt \
  --device cpu \
  --mode select_action \
  --bridge pusht
```

For non-PushT tasks, pass explicit host-owned hooks instead of the packaged bridge:

```bash
scripts/lewm-lerobot-real \
  --policy-path /path/to/task-policy \
  --checkpoint /path/to/task/lewm_object.ckpt \
  --observation-module /path/to/task_obs.py:build_observation \
  --score-info-npz /path/to/lewm_score_tensors.npz \
  --translator /path/to/task_translator.py:translate_actions \
  --candidate-builder /path/to/task_lewm_bridge.py:build_action_candidates
```

Use custom hooks only with task-aligned inputs: the LeRobot policy, observation, LeWorldModel score
tensors, and candidate bridge must describe the same robotics task. The wrapper runs real model
inference, then executes the selected WorldForge action chunk in the local mock world for reporting.

## Runtime Boundary

The packaged demos use real WorldForge provider surfaces with injected deterministic runtimes. They
are intended to verify the framework path in a clean checkout.

- `worldforge-demo-leworldmodel` exercises `LeWorldModelProvider`, score planning, execution,
  persistence, and reload without installing `stable_worldmodel`, torch, or checkpoints.
- `worldforge-demo-lerobot` exercises `LeRobotPolicyProvider`, policy-plus-score planning,
  execution, persistence, and reload without installing LeRobot, torch, or policy checkpoints.

Optional live smoke scripts are separate because they require host-owned model runtimes,
credentials, checkpoints, robot observations, or action translators.

## DimOS Go2 Replay Arena

The DimOS Go2 replay arena is a checkout-safe robotics decision-evidence example. It does not
import DimOS, start a simulator, or connect to hardware. It consumes a small replay-shaped JSON
fixture, scores candidate Go2 actions with transparent costs, and writes a decision trace plus a
compact report.

```bash
uv run python examples/dimos-go2-replay-arena/run.py \
  --fixture examples/dimos-go2-replay-arena/fixtures/go2_office_replay_frame.json \
  --out .worldforge/dimos-go2-replay-arena
```

```bash
uv run python examples/dimos-go2-replay-arena/run.py \
  --all-fixtures \
  --out .worldforge/dimos-go2-replay-arena-batch
```

Expected success signal: the output directory contains `decision-trace.json` and `report.md`, and
the trace includes a selected action, rejected counterfactuals, score margin, and baseline regret.
Batch mode also writes `batch-report.json` and `batch-report.md`.
The bundled fixtures include one replay where WorldForge rejects the hardcoded baseline and one
clear-hallway replay where the baseline remains the best action.
