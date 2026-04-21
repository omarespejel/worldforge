# Operations

WorldForge is a Python library plus CLI. Operational responsibility lives in the host application
that imports it. This page documents the runtime assumptions and minimum runbook for developers using
WorldForge in services, jobs, or provider-evaluation pipelines.

For task-specific runbooks, use [User And Operator Playbooks](./playbooks.md). That page covers
clean checkout validation, provider availability, adapter promotion, persistence recovery, remote
artifact handling, optional runtime smokes, benchmarks, and release gates.

## Operational Modes

| Mode | Suitable use | Boundary |
| --- | --- | --- |
| local development | examples, unit tests, adapter prototyping, deterministic demos | `mock` provider and local JSON state |
| provider evaluation job | fixture-backed provider checks, benchmarks, optional runtime smokes | host owns credentials, checkpoints, outputs, and run artifacts |
| embedded service/library use | application calls WorldForge APIs inside a larger system | host owns request IDs, telemetry export, persistence, retries around jobs, and alerts |
| real robot or simulator loop | host supplies policy observations and action translators | host owns safety interlocks, controller semantics, and embodiment-specific execution |

Minimum startup preflight for a host process:

```bash
uv run worldforge doctor --registered-only
uv run worldforge provider health
```

## Configuration

Configuration comes from constructor arguments and environment variables documented in
`.env.example`.

- `COSMOS_BASE_URL` enables the Cosmos adapter.
- `NVIDIA_API_KEY` is optional bearer auth for Cosmos.
- `RUNWAYML_API_SECRET` enables the Runway adapter.
- `RUNWAY_API_SECRET` remains supported as the legacy Runway alias.
- `RUNWAYML_BASE_URL` overrides the default Runway API endpoint.
- `LEWORLDMODEL_POLICY` or `LEWM_POLICY` enables the optional LeWorldModel adapter.
- `LEWORLDMODEL_CACHE_DIR` overrides the LeWorldModel checkpoint root.
- `LEWORLDMODEL_DEVICE` selects the optional torch device for LeWorldModel scoring.
- `GROOT_POLICY_HOST` enables the optional GR00T embodied-policy adapter.
- `GROOT_POLICY_PORT` defaults to `5555`.
- `GROOT_POLICY_TIMEOUT_MS` defaults to `15000`.
- `GROOT_POLICY_API_TOKEN`, `GROOT_POLICY_STRICT`, and `GROOT_EMBODIMENT_TAG` are optional GR00T
  PolicyClient settings.
- `LEROBOT_POLICY_PATH` or `LEROBOT_POLICY` enables the optional LeRobot embodied-policy adapter.
- `LEROBOT_POLICY_TYPE`, `LEROBOT_DEVICE`, `LEROBOT_CACHE_DIR`, and `LEROBOT_EMBODIMENT_TAG` are
  optional LeRobot settings.
- `JEPA_MODEL_PATH` and `GENIE_API_KEY` enable scaffold adapters only.

Validate configuration when the host process starts:

```bash
uv run worldforge doctor --registered-only
uv run worldforge provider health
```

## Persistence

World state is persisted as local JSON under `.worldforge/worlds` by default or under the
`state_dir` passed to `WorldForge`.

The same local store is available from the CLI for checkout jobs and operator handoffs:

```bash
uv run worldforge world create lab --provider mock
uv run worldforge world add-object <world-id> cube --x 0 --y 0.5 --z 0 --object-id cube-1
uv run worldforge world update-object <world-id> cube-1 --x 0.2 --y 0.5 --z 0
uv run worldforge world predict <world-id> --object-id cube-1 --x 0.4 --y 0.5 --z 0
uv run worldforge world list
uv run worldforge world objects <world-id>
uv run worldforge world history <world-id>
uv run worldforge world export <world-id> --output world.json
uv run worldforge world import world.json --new-id --name lab-copy
uv run worldforge world fork <world-id> --history-index 0 --name lab-start
```

This store is suitable for local development, tests, examples, and single-writer workflows. It is
not a concurrent database. Services that need multi-writer persistence should store exported world
payloads in their own database and apply their own locking, backup, and retention policy.

Persistence remains explicitly host-owned beyond local JSON import/export. The reason is boundary
clarity: host applications own deployment topology, durability, locking semantics, backup policy,
and retention requirements. WorldForge should not imply production durability guarantees that a
local JSON store cannot enforce.

Supported persistence invariants:

- World IDs are validated as file-safe local storage identifiers before any read or write. Path
  separators, traversal-shaped IDs, empty strings, and non-string IDs are rejected.
- CLI object mutations and persisted predictions load the world, apply typed `SceneObject`,
  `SceneObjectPatch`, or `Action` values, and write through `save_world(...)`.
- `world predict` saves the provider-updated world unless `--dry-run` is supplied.
- Local JSON imports reject malformed scene object IDs, non-object state payloads, invalid
  metadata, invalid history, negative steps, history entries from future steps, empty history
  summaries, malformed serialized actions, and invalid historical snapshot states.
- `save_world(...)` validates the serialized world before writing and replaces the destination file
  atomically through a temporary file in the same directory.
- README and operations docs state that multi-writer persistence is host-owned.
- Any future built-in persistence backend must be introduced as an explicit adapter with its own
  locking, migration, and recovery documentation.

## Observability

Attach a provider event handler at `WorldForge(event_handler=...)` or provider construction time.
Use `compose_event_handlers(...)` to fan out events to:

- `JsonLoggerSink` for structured JSON logs.
- `ProviderMetricsSink` for request, retry, error, and latency aggregates.
- `InMemoryRecorderSink` for tests and local debugging.

Host services should add request or trace IDs through `JsonLoggerSink(extra_fields=...)` and
include those IDs in surrounding application logs.

## Failure Modes

- Invalid caller input raises `WorldForgeError`.
- Malformed persisted or provider-supplied state raises `WorldStateError`.
- Provider runtime, transport, credential, and upstream failures raise `ProviderError`.
- Missing remote credentials leave the provider unregistered unless inspected through
  `doctor()`.
- Remote create-style requests are single-attempt by default; health checks, polling, and
  downloads retry according to `ProviderRequestPolicy`.
- Cosmos and Runway validate typed upstream response payloads before creating returned media
  objects.
- Runway artifact downloads fail explicitly on expired/unavailable URLs, empty downloads, and
  explicit unsupported content types.
- LeWorldModel scoring fails explicitly when optional dependencies are unavailable, the checkpoint
  cannot load, required `pixels` / `goal` / `action` fields are missing, action candidates are not
  four-dimensional, or returned scores are not finite.
- GR00T policy selection fails explicitly when the PolicyClient dependency is unavailable, the
  policy server is unreachable, observations are malformed, raw actions are not JSON-compatible,
  or no host-owned action translator is provided.
- LeRobot policy selection fails explicitly when the LeRobot dependency is unavailable, policy
  loading fails, observations are malformed, raw actions are not JSON-compatible, or no host-owned
  action translator is provided.

## Recovery

- For local state corruption, restore from the host application's backup of exported world JSON.
- For missing credentials, fix the environment and restart the host process so provider
  auto-registration runs again.
- For transient remote failures, inspect emitted `ProviderEvent` records for `operation`,
  `phase`, `status_code`, `attempt`, and `target`.
- For expired Runway artifact URLs, regenerate or persist downloaded outputs immediately after
  task completion.
- For LeWorldModel failures, run `worldforge provider health leworldmodel`, verify
  `stable-worldmodel[env]` and `torch` are installed in the host environment, then confirm the
  configured policy exists under `$STABLEWM_HOME` or `LEWORLDMODEL_CACHE_DIR`.
- To smoke-test a real LeWorldModel checkpoint, run the packaged uv command with upstream
  dependencies:
  `uv run --python 3.10 --with "stable-worldmodel[train,env] @ git+https://github.com/galilai-group/stable-worldmodel.git" --with "datasets>=2.21" worldforge-smoke-leworldmodel
  --stablewm-home ~/.stable-wm --policy pusht/lewm`.
  This is the real inference smoke: it requires an extracted object checkpoint such as
  `~/.stable-wm/pusht/lewm_object.ckpt`, builds task-shaped tensors, and calls the upstream
  `stable_worldmodel.policy.AutoCostModel` path through
  `LeWorldModelProvider`.
- If you have Hugging Face LeWM `config.json` and `weights.pt` assets rather than an extracted
  `*_object.ckpt` archive, build the object checkpoint first with
  `uv run --python 3.10 --with "stable-worldmodel[train,env] @ git+https://github.com/galilai-group/stable-worldmodel.git" --with "datasets>=2.21" --with huggingface_hub
  worldforge-build-leworldmodel-checkpoint --stablewm-home ~/.stable-wm --policy pusht/lewm`.
  The builder downloads assets to `~/.cache/worldforge/leworldmodel` by default and writes the
  object checkpoint under `$STABLEWM_HOME`.
- To demonstrate the LeWorldModel planning flow without optional dependencies, run
  `uv run worldforge-demo-leworldmodel`. It uses the real `LeWorldModelProvider` interface
  with an injected deterministic cost runtime and exercises score planning, execution,
  persistence, and reload. It is not a real upstream-checkpoint inference run; use
  `worldforge-smoke-leworldmodel` for that path. The demo should report
  `uses_leworldmodel_provider: true`, `uses_worldforge_score_planning: true`, and
  `uses_real_upstream_checkpoint: false`.
- To demonstrate LeRobot policy-plus-score planning without optional dependencies, run
  `uv run worldforge-demo-lerobot`. It uses the real `LeRobotPolicyProvider` interface with an
  injected deterministic policy runtime and exercises policy selection, score ranking, execution,
  persistence, and reload. It is not a real LeRobot checkpoint inference run.
- To smoke-test a real GR00T policy server, install or check out NVIDIA Isaac-GR00T, prepare a
  host-specific observation factory and action translator, then run
  `python scripts/smoke_gr00t_policy.py --gr00t-root /path/to/Isaac-GR00T --start-server ...`.
  The script can also connect to an existing server with `GROOT_POLICY_HOST` and
  `--policy-info-json` or `--observation-module`.
- Starting the upstream GR00T server requires a compatible NVIDIA/Linux runtime for its CUDA and
  TensorRT dependencies. On unsupported hosts, point WorldForge at an already running remote GR00T
  policy server.

## Release Checklist

Before publishing a release:

```bash
uv sync --group dev
uv lock --check
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run python scripts/generate_provider_docs.py --check
uv run --extra harness pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90
bash scripts/test_package.sh
```

Also update `CHANGELOG.md`, the README, and provider documentation for any public behavior change.

## Provider Hardening Criteria

- Cosmos and Runway response parsers cover success and malformed upstream payloads with fixture
  tests.
- Remote provider non-happy-path tests cover transport retries, malformed JSON, missing task IDs,
  failed tasks, partial outputs, expired artifacts, bad artifact content types, and provider
  limits.
- Persistence remains documented as host-owned unless a dedicated persistence adapter is designed.
- API documentation lists the public exception families and provider workflow failure modes.
- Remaining work is tracked with measurable exit criteria before provider capabilities are
  advertised as complete.
