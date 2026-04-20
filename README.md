# WorldForge

Build physical-AI workflows from world models, not one-off scripts.

WorldForge is a Python integration layer for predictive world models, action scorers, embodied
policies, video generators, and evaluation harnesses. It gives each model a clear capability
contract, then adds the missing engineering layer around it: world state, planning loops, provider
diagnostics, evaluation suites, benchmarks, persistence, and a CLI.

Use it when you want to wire physical-AI systems together without pretending every model is the
same thing.

It is deliberately not a hosted service or a lowest-common-denominator model API. Optional model
runtimes, robot stacks, credentials, checkpoints, production telemetry, and durable storage stay
owned by the host application.

## Why It Exists

Physical-AI tooling is fragmented. A JEPA cost model, a robot policy server, a video simulator,
and a remote media API all have different inputs, runtimes, failure modes, and claims. That
fragmentation is fine. Hiding it is where projects get brittle.

WorldForge keeps those differences visible while making the workflow composable:

- adapters declare exactly what a provider can do.
- planning code can combine policy providers, score providers, and predictive providers.
- evaluation and benchmark runs exercise the same surfaces users call.
- optional heavy runtimes stay out of the base package.
- diagnostics make missing credentials, missing dependencies, and capability mismatches obvious.

## What You Can Build

- provider adapters for new world-model runtimes and APIs.
- score-based planners that rank candidate action sequences.
- policy-plus-score workflows where an embodied policy proposes actions and a world model ranks
  them.
- local evaluation harnesses for planning, reasoning, generation, transfer, and provider behavior.
- benchmark runs for latency, retries, throughput, and report export.
- checkout-safe demos that validate integration paths without downloading large models.
- optional live smokes for host environments with checkpoints, robot stacks, or remote servers.

## Who It's For

- ML researchers comparing model surfaces without rewriting the harness each time.
- robotics and physical-AI engineers wiring policies, scorers, simulators, and media providers
  around host-owned systems.
- Python developers building adapter packages, CLI workflows, and reproducible demos.
- builders and enthusiasts who want something that runs from a clean checkout before they install
  CUDA stacks or download checkpoints.

## Capability Model

WorldForge treats "world model" as an operational interface, not a marketing label.

- `predict`: state + action -> predicted state
- `score`: observations + goal + action candidates -> ranked candidates
- `policy`: observation + instruction -> action chunks
- `generate`: prompt/options -> media artifact
- `transfer`: media artifact + prompt/options -> media artifact
- `reason` and `embed`: narrow auxiliary contracts where providers implement them directly

This separation is deliberate. LeWorldModel is a score provider, not a video generator. GR00T and
LeRobot are policy providers, not predictive world models. Cosmos and Runway are media generation
adapters, not proof of controllable physical planning semantics.

The typical loop looks like this:

```text
observe state
  -> propose candidate actions
  -> score or roll out possible futures
  -> select an action sequence
  -> execute through a provider
  -> persist, evaluate, and observe again
```

## Install

Application project:

```bash
uv add worldforge
```

Repository development:

```bash
uv sync --group dev
cp .env.example .env
```

## Quick Start

```python
from worldforge import Action, BBox, Position, SceneObject, StructuredGoal, WorldForge

forge = WorldForge()
world = forge.create_world("kitchen", provider="mock")

world.add_object(
    SceneObject(
        "red_mug",
        Position(0.0, 0.8, 0.0),
        BBox(Position(-0.05, 0.75, -0.05), Position(0.05, 0.85, 0.05)),
    )
)

prediction = world.predict(Action.move_to(0.3, 0.8, 0.0), steps=2)
print(prediction.provider, prediction.physics_score)

plan = world.plan(
    goal_spec=StructuredGoal.object_at(
        object_name="red_mug",
        position=Position(0.3, 0.8, 0.0),
    )
)
print(plan.action_count, plan.success_probability)

doctor = forge.doctor()
print(doctor.healthy_provider_count, doctor.provider_count)
```

CLI:

```bash
uv run worldforge examples
uv run worldforge doctor
uv run worldforge provider list
uv run worldforge provider docs
uv run worldforge provider info mock
uv run worldforge predict kitchen --provider mock --x 0.3 --y 0.8 --z 0.0 --steps 2
uv run worldforge eval --suite planning --provider mock --format json
uv run worldforge benchmark --provider mock --iterations 5 --format json
```

## Provider Surfaces

<!-- provider-catalog-readme:start -->
| Provider | Capability surface | Registration | Runtime ownership |
| --- | --- | --- | --- |
| `mock` | `predict`, `generate`, `transfer`, `reason`, `embed`, `plan` | always registered | in-repo deterministic local provider |
| [`cosmos`](./docs/src/providers/cosmos.md) | `generate` | `COSMOS_BASE_URL` | host supplies a reachable Cosmos deployment and optional `NVIDIA_API_KEY` |
| [`runway`](./docs/src/providers/runway.md) | `generate`, `transfer` | `RUNWAYML_API_SECRET` or `RUNWAY_API_SECRET` | host supplies Runway credentials and persists returned artifacts |
| [`leworldmodel`](./docs/src/providers/leworldmodel.md) | `score` | `LEWORLDMODEL_POLICY` or `LEWM_POLICY` | host installs `stable_worldmodel`, torch, and compatible checkpoints |
| [`gr00t`](./docs/src/providers/gr00t.md) | `policy` | `GROOT_POLICY_HOST` | host runs or reaches an Isaac GR00T policy server |
| [`lerobot`](./docs/src/providers/lerobot.md) | `policy` | `LEROBOT_POLICY_PATH` or `LEROBOT_POLICY` | host installs LeRobot and compatible policy checkpoints |
| `jepa` | scaffold | `JEPA_MODEL_PATH` | credential-gated mock-backed reservation, not a real JEPA runtime |
| `genie` | scaffold | `GENIE_API_KEY` | credential-gated mock-backed reservation, not a real Genie runtime |
<!-- provider-catalog-readme:end -->

Provider candidate scaffolds stay outside package exports and auto-registration until they have a
validated runtime path, typed parser coverage, limits, and docs. The active candidate is
[`jepa-wms`](./docs/src/providers/jepa-wms.md), a direct-construction scaffold for future
`facebookresearch/jepa-wms` score-provider work.

## Architecture

```text
Host application / CLI
  |
  v
WorldForge facade
  provider catalog, registry, diagnostics, persistence helpers
  |
  v
World runtime
  state, history, planning, execution
  |
  v
Provider adapter
  capability contract, validation, events
  |
  v
Upstream runtime or API
  local model, robot policy server, media API, deterministic mock
```

Repository responsibilities:

| Path | Responsibility |
| --- | --- |
| `src/worldforge/models.py` | Domain models, serialization, validation errors, provider metadata, result types, request policies |
| `src/worldforge/framework.py` | `WorldForge`, `World`, persistence, planning, prediction, comparison, diagnostics |
| `src/worldforge/providers/catalog.py` | In-repo provider factories and auto-registration policy |
| `src/worldforge/providers/base.py` | Provider interfaces, `ProviderError`, remote-provider behavior, `PredictionPayload` |
| `src/worldforge/providers/` | Concrete adapters for mock, Cosmos, Runway, LeWorldModel, GR00T, LeRobot, JEPA, and Genie |
| `src/worldforge/evaluation/` | Built-in deterministic evaluation suites and report renderers |
| `src/worldforge/benchmark.py` | Capability-aware latency, retry, throughput, and event benchmark harness |
| `src/worldforge/observability.py` | `ProviderEvent` sinks for logs, recording, and metrics |
| `src/worldforge/testing/` | Reusable provider contract assertions |

Read the detailed [architecture](./docs/src/architecture.md), [taxonomy](./docs/src/world-model-taxonomy.md),
and [provider authoring guide](./docs/src/provider-authoring-guide.md) before adding a new
adapter.

## Operating Boundaries

- Provider capabilities are contracts. Do not advertise an operation unless the adapter implements
  it end to end and returns the typed WorldForge result. Capability names are strict; unknown
  names fail instead of behaving like empty filters.
- Optional runtimes remain host-owned. WorldForge does not install torch, LeWorldModel, LeRobot,
  Isaac GR00T, CUDA, TensorRT, robot controllers, checkpoints, or datasets as base dependencies.
- Embodiment-specific translation remains host-owned. Policy providers preserve raw actions, but a
  caller must translate them into executable `Action` objects for the target robot or simulator.
- Local JSON persistence is for deterministic single-writer workflows. Services needing locking,
  transactions, migrations, object storage, or backup policy should own that persistence layer.
- Built-in evaluation suites are deterministic contract harnesses. They do not claim physical
  fidelity, media quality, or real-world safety.
- Scaffold adapters are reservations for future work. They must not be presented as real JEPA,
  Genie, or jepa-wms integrations.
- World IDs are local storage identifiers. They may contain letters, numbers, `.`, `_`, and `-`;
  path separators and traversal-shaped IDs are rejected before any persistence read or write.

## User And Operator Playbooks

The [playbooks](./docs/src/playbooks.md) collect concrete runbooks for common work:

- bootstrap a clean checkout and verify provider docs.
- choose the right provider capability for a workflow.
- add or promote a provider adapter without overstating capabilities.
- diagnose provider registration, health, and capability mismatches.
- operate and recover local JSON persistence.
- run evaluation, benchmarks, optional runtime smokes, and release gates.

Use the playbooks with [operations](./docs/src/operations.md) when embedding WorldForge in a job
or service. Production credentials, telemetry export, dashboards, artifact retention, robot safety,
and durable storage are still host-owned.

## Demos And Optional Smokes

List runnable examples:

```bash
uv run worldforge examples
uv run worldforge examples --format json
```

Visual harness:

```bash
uv run --extra harness worldforge-harness
uv run --extra harness worldforge-harness --flow lerobot
uv run --extra harness worldforge-harness --flow diagnostics
uv run worldforge harness --list
uv run worldforge harness --list --format json
```

`TheWorldHarness` is a Textual-based optional TUI for running integration flows as visible,
inspectable traces. It keeps Textual outside the base dependency set and presents each run through
a timeline, metrics inspector, persisted-state summary, and structured transcript.

| Flow | What it exercises |
| --- | --- |
| `leworldmodel` | Score-provider planning with deterministic LeWorldModel-shaped costs, selected action path, execution, persistence, reload, and provider events. |
| `lerobot` | Policy-plus-score planning with deterministic LeRobot-shaped action chunks, translation, ranking, execution, persistence, reload, and provider events. |
| `diagnostics` | Provider catalog diagnostics plus a mock-provider benchmark comparison across predict, reason, generate, and transfer. |

Checkout-safe packaged demos:

```bash
uv run worldforge-demo-leworldmodel
uv run worldforge-demo-leworldmodel --json-only
uv run worldforge-demo-lerobot
uv run worldforge-demo-lerobot --json-only
```

These use real WorldForge provider interfaces with injected deterministic runtimes. They verify the
adapter, planning, execution, persistence, and reload path without installing optional model
runtimes or downloading checkpoints.

Real-checkpoint smoke:

```bash
uv run --python 3.10 \
  --with "stable-worldmodel[train,env] @ git+https://github.com/galilai-group/stable-worldmodel.git" \
  --with "datasets>=2.21" \
  worldforge-smoke-leworldmodel \
  --stablewm-home ~/.stable-wm \
  --policy pusht/lewm \
  --device cpu
```

The checkpoint smoke requires host-provided LeWorldModel dependencies and an extracted object
checkpoint such as `~/.stable-wm/pusht/lewm_object.ckpt`. If you have Hugging Face LeWM assets
instead of an object checkpoint, use `worldforge-build-leworldmodel-checkpoint` first.

## Development

Primary local gates:

```bash
uv lock --check
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run python scripts/generate_provider_docs.py --check
uv run pytest
uv run pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90
bash scripts/test_package.sh
```

Provider scaffolding:

```bash
uv run python scripts/scaffold_provider.py "Acme WM" \
  --taxonomy "JEPA latent predictive world model" \
  --planned-capability score
```

The scaffold creates adapter, fixture, test, and docs-stub files without advertising public
capabilities until a real implementation is complete.

## Scope

WorldForge owns the software layer around world-model workflows:

1. provider capability contracts.
2. world state, actions, scene objects, media artifacts, and result models.
3. planning loops that compose predictive, score, policy, and execution surfaces.
4. diagnostics, evaluation, benchmarks, and provider events.
5. deterministic demos, smoke commands, and adapter contract tests.

WorldForge does not own model training, robot safety, checkpoint hosting, credential storage,
production telemetry, dashboards, or durable multi-writer persistence.

## Current State

WorldForge is pre-1.0 beta. It is useful today for:

- local provider adapter development.
- deterministic planning and evaluation experiments.
- checkout-safe demos and optional-runtime smoke tests.
- contract testing for provider packages.
- CLI diagnostics around provider registration, health, and capabilities.

Known limits:

- JEPA and Genie are credential-gated scaffold adapters backed by deterministic mock behavior.
- `jepa-wms` is a direct-construction candidate and is not exported or auto-registered.
- local JSON persistence is single-writer only.
- built-in evaluation scores are contract signals, not physical-fidelity, media-quality, or
  real-world safety claims.
- optional model runtimes, checkpoints, robot dependencies, trace export, dashboards, and
  production telemetry remain host-owned.

## Links

- Documentation: [docs/src](./docs/src)
- Playbooks: [docs/src/playbooks.md](./docs/src/playbooks.md)
- Repository: <https://github.com/AbdelStark/worldforge>
- Issues: <https://github.com/AbdelStark/worldforge/issues>
