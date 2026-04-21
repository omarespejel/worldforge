# Capability Matrix

Authoritative mapping from `ProviderCapabilities` flag → required method →
expected return type. If you cannot point at the matching method that returns
the matching type with validated WorldForge models, the flag must stay `False`.

| Capability | Required method | Returns | Notes |
| --- | --- | --- | --- |
| `predict` | `predict_state(world, action)` | `WorldState` | Future world state from current state + action. Not a score, not a clip. |
| `generate` | `generate(prompt, duration, ...)` | `MediaClip` | Video / media synthesis. Pure synthesis only. |
| `transfer` | `transfer(clip, prompt, ...)` | `MediaClip` | Edits an existing clip; do not advertise unless an input clip is genuinely required. |
| `reason` | `reason(question, context)` | `ReasoningResult` | Natural-language reasoning over scene/state. |
| `embed` | `embed(items)` | `list[EmbeddingVector]` | Vector embeddings for retrieval / similarity. |
| `plan` | `plan_actions(world, goal)` | `Plan` | Returns a sequence of `Action`s; not the same as `policy`. |
| `score` | `score_actions(world, actions)` | `ActionScoreResult` | Per-action scalar / ranking. World models that score candidates live here, not under `predict`. |
| `policy` | `select_actions(observation, ...)` | `list[Action]` + raw metadata | Robot / agent policy. Raw action arrays must be translated to `Action` by the host before execution. |

## Disambiguation

- **score vs predict**: `predict` advances state forward. `score` evaluates given candidates against a learned model. A world-model that ranks N candidate clips/actions is `score`, even if internally it rolls state forward.
- **policy vs plan**: `plan` is offline / search-based and returns a sequence; `policy` is reactive and returns the next action(s) for the current observation. GR00T and LeRobot are `policy`. A planner over a world model is `plan`.
- **generate vs transfer**: `generate` synthesises from prompt only; `transfer` requires an input clip. Most "video edit" providers are `transfer`; pure text-to-video is `generate`.

## Fail-closed default

`ProviderCapabilities()` with no arguments sets every flag `False`. This is the correct default for every scaffold and every uncertain capability. Flipping a flag is a one-way contract: downstream consumers will call the matching method without further checks.
