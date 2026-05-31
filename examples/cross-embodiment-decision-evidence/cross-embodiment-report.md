# Cross-Embodiment Decision Evidence

This bundle validates Go2 replay, PimSim export, and SO-101 manipulation traces against one `DecisionTrace v1` contract. It is replay evidence, not a live-robot or learned-model claim.

| Trace | Embodiment | Selected | Margin | Baseline Regret | Score Kind | Outcome Kind |
| --- | --- | --- | ---: | ---: | --- | --- |
| `decision-trace-go2` | unitree_go2_air / quadruped_navigation | `stop_relocalize` | 0.205603 | 1.401039 | hand_cost | analytic |
| `decision-trace-pimsim` | unitree_go2_air / quadruped_navigation | `arc_left_clear` | 0.066455 | 1.637225 | hand_cost | analytic |
| `decision-trace-so101` | so101 / manipulation | `lift-place-stable` | 0.930300 | 1.197197 | hand_cost | analytic |

## DecisionTrace v1 Contract

- One schema validates navigation and manipulation decisions.
- Actions are self-describing with `type`, `params`, and `units`.
- Goals include ordered `sub_goals` and partial-credit success criteria.
- Reproducibility records provider version, checkpoint/model-card refs, input digest, seed, and code ref.
- Claim boundaries explicitly separate hand-cost scoring from learned-latent scoring and analytic outcomes from sim/real measured outcomes.

## Kill Criterion

If WorldForge cannot choose, explain, compare, or expose counterfactual robot actions better than a hardcoded DimOS command or plain script, stop pushing this integration and pivot.

## go2

- Task: move toward the inspection waypoint without clipping the chair leg
- Selected: `stop_relocalize`
- Why: Lowest transparent replay cost after combining goal distance, obstacle risk, map cost, uncertainty, and relocalization terms; progress=0.000m.
- Counterfactuals: 4
- Limitations: No live Go2 command was sent.; Outcome is an analytic endpoint estimate from replay metadata.; The scorer is transparent hand cost, not a learned latent world model.

## pimsim

- Task: navigate around the PimSim chair leg toward the inspection marker
- Selected: `arc_left_clear`
- Why: Lowest transparent replay cost after combining goal distance, obstacle risk, map cost, uncertainty, and relocalization terms; progress=0.947m.
- Counterfactuals: 4
- Limitations: No live Go2 command was sent.; Outcome is an analytic endpoint estimate from replay metadata.; The scorer is transparent hand cost, not a learned latent world model.

## so101

- Task: Select a replay action chunk for tabletop pick-and-place.
- Selected: `lift-place-stable`
- Why: Lowest weighted replay cost: target placement inside tolerance with high grasp confidence and low contact risk.
- Counterfactuals: 3
- Limitations: No SO-101 hardware, camera, calibration, or LeRobot runtime was used.; Outcome is a replay/mock placement check, not a physical execution result.; Joint deltas are fixture units and must not be treated as calibrated commands.
