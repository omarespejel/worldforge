# Cross-Embodiment Decision Evidence

This bundle validates Go2 replay, PimSim export, and SO-101 manipulation traces against one `DecisionTrace v1` contract. It is replay evidence, not a live-robot or learned-model claim.

The comparable cross-embodiment column is `Value Signal` on a normalized `[0, 1]` scale. `Local Margin` and `Local Baseline Regret` are embodiment-local hand-cost diagnostics and should not be compared across embodiments as raw units.

`Outcome Kind` is the DecisionTrace measurement class. `Outcome Provenance` is the concrete source; `mock_replay_execution` under `analytic` means a fixture/mock analytic check, not sim-measured or real-measured success.

| Trace | Embodiment | Selected | Value Signal | Desirability | Local Margin | Local Baseline Regret | Score Kind | Outcome Kind | Outcome Provenance |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `decision-trace-go2` | unitree_go2_air / quadruped_navigation | `stop_relocalize` | 0.656151 | 1.000000 | 0.205603 | 1.401039 | hand_cost | analytic | analytic_replay_estimate |
| `decision-trace-pimsim` | unitree_go2_air / quadruped_navigation | `arc_left_clear` | 0.696969 | 1.000000 | 0.066455 | 1.637225 | hand_cost | analytic | analytic_replay_estimate |
| `decision-trace-so101` | so101 / manipulation | `lift-place-stable` | 0.875163 | 1.000000 | 0.930300 | 1.197197 | hand_cost | analytic | mock_replay_execution |

## DecisionTrace v1 Contract

- One DecisionTrace v1 semantic validator accepts navigation and manipulation decisions.
- Actions are self-describing with `type`, `params`, and `units`.
- Goals include ordered `sub_goals` and partial-credit success criteria.
- Reproducibility records provider version, checkpoint/model-card refs, input digest, seed, and code ref.
- Claim boundaries explicitly separate hand-cost scoring from learned-latent scoring and analytic outcomes from sim/real measured outcomes.

## Kill Criterion

If WorldForge cannot choose, explain, compare, or expose counterfactual robot actions better than a hardcoded DimOS command or plain script, stop pushing this integration and pivot.

For learned-scorer claims, the stricter gate is independent held-out referee performance above chance and above the proprio/hand-cost baselines. A marginal below-chance improvement is logged as negative evidence, not as value.

## go2

- Task: move toward the inspection waypoint without clipping the chair leg
- Selected: `stop_relocalize`
- Value signal: 0.656151
- Why: Lowest transparent replay cost after combining goal distance, obstacle risk, map cost, uncertainty, and relocalization terms; progress=0.000m.
- Counterfactuals: 4
- Outcome provenance: analytic_replay_estimate
- Limitations: No live Go2 command was sent.; Outcome is an analytic endpoint estimate from replay metadata.; The scorer is transparent hand cost, not a learned latent world model.

## pimsim

- Task: navigate around the PimSim chair leg toward the inspection marker
- Selected: `arc_left_clear`
- Value signal: 0.696969
- Why: Lowest transparent replay cost after combining goal distance, obstacle risk, map cost, uncertainty, and relocalization terms; progress=0.947m.
- Counterfactuals: 4
- Outcome provenance: analytic_replay_estimate
- Limitations: No live Go2 command was sent.; Outcome is an analytic endpoint estimate from replay metadata.; The scorer is transparent hand cost, not a learned latent world model.

## so101

- Task: Select a replay action chunk for tabletop pick-and-place.
- Selected: `lift-place-stable`
- Value signal: 0.875163
- Why: Lowest weighted replay cost: target placement inside tolerance with high grasp confidence and low contact risk.
- Counterfactuals: 3
- Outcome provenance: mock_replay_execution
- Limitations: No SO-101 hardware, camera, calibration, or LeRobot runtime was used.; Outcome is a replay/mock placement check, not a physical execution result.; Joint deltas are fixture units and must not be treated as calibrated commands.
