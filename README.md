# WorldForge

Typed local-first framework for physical-AI world-model workflows.

WorldForge gives Python developers a strict boundary between host application code, model/provider
runtimes, world state, planning loops, evaluation, and diagnostics. It is built for ML engineers
working with predictive world models, action scorers, embodied policies, video generation
adapters, and local evaluation harnesses.

It is a library and CLI, not a hosted service. Optional model runtimes, robot stacks, credentials,
checkpoints, production telemetry, and durable persistence stay owned by the host environment.

## What It Provides

- Typed world state, actions, poses, scene objects, media artifacts, provider profiles, and result
  objects.
- Capability-specific provider contracts for prediction, generation, transfer, reasoning,
  embedding, action scoring, and embodied policy selection.
- Planning paths for predictive rollouts, score-based candidate selection, policy action
  selection, and policy-plus-score composition.
- Deterministic local `mock` provider for tests, examples, and adapter contract checks.
- Provider diagnostics through Python and CLI surfaces, including missing environment variables
  and health checks for optional adapters.
- Built-in evaluation suites and a provider benchmark harness for adapter behavior, latency,
  retries, throughput, and report export.
- Provider event hooks for JSON logging, in-memory recording, and metrics aggregation.

## Design Center

WorldForge treats "world model" as an operational interface, not a loose label.

```text
observe state
  -> propose candidate actions
  -> score or roll out possible futures
  -> select an action sequence
  -> execute through a provider
  -> persist, evaluate, and observe the result
```

The core architecture is shaped around explicit capabilities:

- `predict`: state + action -> predicted state
- `score`: observations + goal + action candidates -> ranked candidates
- `policy`: observation + instruction -> action chunks
- `generate`: prompt/options -> media artifact
- `transfer`: media artifact + prompt/options -> media artifact
- `reason` and `embed`: narrow auxiliary contracts where providers implement them directly

This separation is deliberate. LeWorldModel is a score provider, not a video generator. GR00T and
LeRobot are policy providers, not predictive world models. Cosmos and Runway are media generation
adapters, not proof of controllable physical planning semantics.

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
uv run worldforge provider info mock
uv run worldforge predict kitchen --provider mock --x 0.3 --y 0.8 --z 0.0 --steps 2
uv run worldforge eval --suite planning --provider mock --format json
uv run worldforge benchmark --provider mock --iterations 5 --format json
```

## Provider Surfaces

| Provider | Surface | Registration | Runtime ownership |
| --- | --- | --- | --- |
| `mock` | `predict`, `generate`, `transfer`, `reason`, `embed`, `plan` | always registered | in-repo deterministic local provider |
| [`leworldmodel`](./docs/src/providers/leworldmodel.md) | `score` | `LEWORLDMODEL_POLICY` or `LEWM_POLICY` | host installs `stable_worldmodel`, torch, and checkpoints |
| [`gr00t`](./docs/src/providers/gr00t.md) | `policy` | `GROOT_POLICY_HOST` | host runs or reaches an Isaac GR00T policy server |
| [`lerobot`](./docs/src/providers/lerobot.md) | `policy` | `LEROBOT_POLICY_PATH` or `LEROBOT_POLICY` | host installs LeRobot and policy checkpoints |
| [`cosmos`](./docs/src/providers/cosmos.md) | `generate` | `COSMOS_BASE_URL` | host supplies reachable Cosmos deployment and optional `NVIDIA_API_KEY` |
| [`runway`](./docs/src/providers/runway.md) | `generate`, `transfer` | `RUNWAYML_API_SECRET` or `RUNWAY_API_SECRET` | host supplies Runway credentials |
| `jepa` | scaffold | `JEPA_MODEL_PATH` | credential-gated mock-backed reservation |
| `genie` | scaffold | `GENIE_API_KEY` | credential-gated mock-backed reservation |

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
  it end to end and returns the typed WorldForge result.
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

## Demos And Optional Smokes

List runnable examples:

```bash
uv run worldforge examples
uv run worldforge examples --format json
```

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

## Technical Scope

WorldForge has a narrow role in the physical-AI stack:

1. Make provider capability boundaries explicit enough that ML engineers can compare predictive
   models, action scorers, policy actors, and media generators without collapsing them into one
   vague interface.
2. Support planning loops where policy providers propose candidate actions, score providers rank
   candidate futures, and execution providers apply selected actions through a typed world state.
3. Keep optional heavy runtimes host-owned while still providing deterministic demos, smoke
   commands, and contract tests for adapter development.
4. Expand world, observation, scene, and evaluation contracts only when the library can validate
   them clearly.

## Links

- Documentation: [docs/src](./docs/src)
- Repository: <https://github.com/AbdelStark/worldforge>
- Issues: <https://github.com/AbdelStark/worldforge/issues>
