# Robotics Decision Trace Schema

This schema is example-local for the DimOS Go2 trace judge. It is intentionally JSON-native and
host-owned so it can evolve before becoming a shared WorldForge artifact contract.

## Observation Summary

```json
{
  "pose": {"x": 0.0, "y": 0.0, "yaw_degrees": 0.0},
  "costmap_summary": {
    "min_clearance_m": 0.8,
    "unknown_area_ratio": 0.42,
    "frontier_count": 4,
    "blocked_ahead": false
  },
  "visual_summary": {
    "target_confidence": 0.64,
    "target_bearing_degrees": 30,
    "gesture_direction_degrees": 35
  },
  "navigation_state": {
    "localized": true,
    "stuck_probability": 0.06,
    "active_goal": null
  }
}
```

## Candidate Scores

`candidate_scores.json` records the goal, candidate actions, transparent scoring features, the
selected candidate, and the raw WorldForge `ActionScoreResult`.

## Score Info

`score_info.json` records the exact score-provider boundary. This is the clearest artifact for
debugging or replacing the transparent scorer with a learned score provider later:

```json
{
  "provider": "transparent-go2-score",
  "capability": "score",
  "score_info": {
    "embodiment": "unitree_go2",
    "host_runtime": "dimos",
    "task": {"human_goal": "inspect the area"},
    "observation_summary": {}
  },
  "action_candidates": []
}
```

The live bridge and collector can feed the trace judge with a venue input file:

```json
{
  "schema_version": 1,
  "embodiment": "unitree_go2",
  "host_runtime": "dimos",
  "task": {
    "human_goal": "inspect the area and move toward the indicated target",
    "goal_representation": {
      "type": "host_interpreted_goal",
      "gesture_direction_degrees": 35
    }
  },
  "observation_summary": {
    "pose": {"x": 0.0, "y": 0.0, "yaw_degrees": 0.0},
    "costmap_summary": {
      "min_clearance_m": 0.8,
      "unknown_area_ratio": 0.42,
      "frontier_count": 4
    }
  },
  "candidates": [
    {
      "id": "scan_left",
      "action": "relative_move",
      "params": {"degrees": 35},
      "features": {
        "goal_alignment": 0.88,
        "information_gain": 0.83,
        "progress": 0.21,
        "obstacle_risk": 0.05,
        "stuck_risk": 0.02,
        "execution_cost": 0.16
      },
      "reason_hint": "high information gain and aligned with the gesture bearing"
    }
  ]
}
```

Candidate feature values are normalized to `[0, 1]`:

| Feature | Meaning |
| --- | --- |
| `goal_alignment` | Candidate heading or target bearing agreement with the host-interpreted goal. |
| `information_gain` | Expected useful new map/visual information. |
| `progress` | Expected physical or path-progress gain. |
| `obstacle_risk` | Estimated collision or poor-clearance risk. |
| `stuck_risk` | Estimated chance the action worsens localization or locomotion state. |
| `execution_cost` | Relative time, motion, or operator cost. |

The transparent scorer uses a utility score where higher is better:

```text
score =
  0.10
  + 0.28 * goal_alignment
  + 0.27 * information_gain
  + 0.25 * progress
  + 0.15 * (1 - obstacle_risk)
  + 0.05 * (1 - stuck_risk)
  - 0.05 * execution_cost
```

## Selected Action

`selected_action.json` is the only artifact a live host would need to translate into a DimOS MCP
tool call. It explicitly records that WorldForge is not executing the robot:

```json
{
  "selected_candidate_id": "detour_right",
  "action": "relative_move",
  "params": {"forward": 0.25, "left": -0.2},
  "execute_with": "host_runtime:mock-dimos",
  "live_execute_with": "host_runtime:dimos",
  "worldforge_executes_robot": false
}
```

## Outcome

`outcome_after_execution.json` is mocked in this checkout-safe example. A live host can replace it
with real post-action measurements:

```json
{
  "outcome_after_execution": {
    "duration_s": 5.0,
    "progress_delta_m": 0.28,
    "coverage_delta": 0.12,
    "target_confidence_delta": 0.18,
    "blocked": false,
    "manual_intervention": false
  }
}
```

This is the missing shape needed for future learned-score training:

```text
observation_summary + goal + candidate_features + selected_action + outcome_after_execution
```
