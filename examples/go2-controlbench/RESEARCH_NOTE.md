# Go2 ControlBench Research Note

## Thesis

Go2 ControlBench is a command-to-outcome benchmark, not a robot video dataset. The useful question
is whether a planner, scorer, or world model can predict and rank real measured robot outcomes
better than naive command math.

## Why This Belongs With WorldForge

WorldForge should not become a robot runtime. The runtime sends commands and records telemetry.
WorldForge evaluates the decision surface:

```text
observation -> goal -> candidate actions -> predicted scores -> selected action -> measured outcome
```

For WMCP, the future lower-layer role is `rollout/evaluate`: a planner asks a world model to
predict outcomes for candidate commands. For DecisionTrace, the durable role is evidence: record
which candidates were considered, how they were scored, which one won, and whether real measured
outcomes supported that decision.

## Tasks

Task A: command-outcome prediction.

- Input: commanded body-frame integral `(forward_m, lateral_m, yaw_rad)`.
- Output: measured body-frame outcome `(forward_m, left_m, dyaw_rad)`.
- Metrics: per-axis RMSE and deadband classification F1.

Task B: inverse-control ranking.

- Input: target measured body motion and candidate command cells.
- Output: ranked candidate commands.
- Metrics: top-1, inverse-control regret, and rank correlation between predicted and true
  measured outcome distances.

Task C: odometry versus external ground truth.

- Status: pending external GT capture.
- Intended signal: compare native Go2 odometry against derived overhead/AprilTag/floor-grid
  trajectories while keeping raw room video private.

## Baselines

- `command_integral`: assumes commanded integral equals measured outcome.
- `affine_linear_system_id`: leave-one-command-cell-out affine fit from command to measured
  outcome.
- `deadband_affine_system_id`: leave-one-command-cell-out affine fit after deterministic
  deadband threshold selection.

## Claim Boundary

Current outputs are real measured hardware evidence, but not autonomous planning evidence. The
capture was operator/script driven, RGB should stay private unless scrubbed, and external ground
truth is still needed before strong claims about true physical displacement.
