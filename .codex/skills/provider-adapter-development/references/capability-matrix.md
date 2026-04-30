# Capability Matrix

| Provider | Surface | Registration | Runtime ownership |
| --- | --- | --- | --- |
| `mock` | `predict`, `generate`, `transfer`, `reason`, `embed` | always | in-repo deterministic local provider |
| `cosmos` | `generate` | `COSMOS_BASE_URL` | host supplies reachable Cosmos deployment and optional `NVIDIA_API_KEY` |
| `runway` | `generate`, `transfer` | `RUNWAYML_API_SECRET` or `RUNWAY_API_SECRET` | host supplies credentials and persists returned artifacts |
| `leworldmodel` | `score` | `LEWORLDMODEL_POLICY` or `LEWM_POLICY` | host installs `stable_worldmodel`, torch, and compatible checkpoints |
| `gr00t` | `policy` | `GROOT_POLICY_HOST` | host runs or reaches Isaac GR00T policy server |
| `lerobot` | `policy` | `LEROBOT_POLICY_PATH` or `LEROBOT_POLICY` | host installs LeRobot and compatible policy checkpoints |
| `jepa`, `genie` | scaffold only | env-gated reservations | not real upstream integrations |
| `jepa-wms` | direct-construction candidate | none | not exported or auto-registered |

Method mapping:
- `predict` -> `BaseProvider.predict(...)` -> `PredictionPayload`
- `generate` -> `BaseProvider.generate(...)` -> `VideoClip`
- `transfer` -> `BaseProvider.transfer(...)` -> `VideoClip`
- `reason` -> `BaseProvider.reason(...)` -> `ReasoningResult`
- `embed` -> `BaseProvider.embed(...)` -> `EmbeddingResult`
- `score` -> `BaseProvider.score_actions(...)` -> `ActionScoreResult`
- `policy` -> `BaseProvider.select_actions(...)` -> `ActionPolicyResult`

`plan` is a WorldForge facade workflow, not a direct provider benchmark operation.

