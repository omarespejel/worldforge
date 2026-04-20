<div align="center">

# WorldForge

**An integration layer for physical-AI world models.**

One typed interface for predictive models, action scorers, embodied policies, and media generators —
so you can compose them, plan through them, evaluate them, and ship them without pretending every
model is the same thing.

[![CI](https://img.shields.io/github/actions/workflow/status/AbdelStark/worldforge/ci.yml?branch=main&label=CI&style=plastic)](https://github.com/AbdelStark/worldforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=plastic&logo=python&logoColor=white)](https://github.com/AbdelStark/worldforge/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=plastic)](./LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen?style=plastic)](./.github/workflows/ci.yml)
[![Typed](https://img.shields.io/badge/typed-py.typed-3f7cac?style=plastic)](./src/worldforge/py.typed)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=plastic)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=plastic)](https://github.com/astral-sh/uv)
[![Status: pre-1.0](https://img.shields.io/badge/status-pre--1.0%20beta-orange?style=plastic)](#project-status)

[**Quickstart**](#quickstart) ·
[**Providers**](#provider-surfaces) ·
[**Capability Model**](#capability-model) ·
[**Architecture**](#architecture) ·
[**Docs**](./docs/src) ·
[**Playbooks**](./docs/src/playbooks.md) ·
[**Changelog**](./CHANGELOG.md)

</div>

---

## Overview

Physical-AI tooling is fragmented. A JEPA-style cost model, a robot policy server, a video
simulator, and a remote media API have different inputs, runtimes, failure modes, and claims.
Hiding that fragmentation is where projects get brittle.

**WorldForge keeps the differences visible while making the workflow composable.** Each provider
adapter declares exactly what it can do — `predict`, `score`, `policy`, `generate`, `transfer`,
`reason`, `embed`, `plan` — behind a strict, fail-closed capability contract. Planning, evaluation,
benchmarks, diagnostics, and persistence are built on that contract, not on top of any particular
runtime.

It is **not** a hosted service, a lowest-common-denominator model API, or a training framework.
Optional model runtimes, robot stacks, credentials, checkpoints, and durable storage stay owned by
the host application.

## Highlights

| | |
| --- | --- |
| **Strict capability contracts** | Eight named capabilities. Adapters advertise only what they implement end-to-end and return typed WorldForge results. Unknown capability names fail loudly. |
| **Composable planning** | Combine predictive, score, and policy providers in a single planning loop. Rank candidates, roll out futures, execute actions, persist state. |
| **Deterministic-by-default** | A built-in `mock` provider, reusable contract assertions (`worldforge.testing`), and deterministic demos make every workflow runnable from a clean checkout. |
| **Host-owned runtimes** | No torch, CUDA, robot controllers, or checkpoints in base dependencies. Integrate LeWorldModel, GR00T, LeRobot, Cosmos, and Runway through their official surfaces — on your terms. |
| **First-class diagnostics** | `worldforge doctor`, provider events, benchmark and evaluation harnesses, and a Textual TUI (`TheWorldHarness`) for inspectable traces. |
| **Typed, linted, covered** | `py.typed`, ruff on `src tests examples scripts`, ≥90% coverage gate, wheel + sdist contract tests in CI across Python 3.10 – 3.13. |

## Install

### Library (recommended)

```bash
uv add "worldforge @ git+https://github.com/AbdelStark/worldforge"
```

### Repository development

```bash
git clone https://github.com/AbdelStark/worldforge.git
cd worldforge
uv sync --group dev
cp .env.example .env
```

Optional extras:

```bash
uv sync --group dev --extra harness   # TheWorldHarness Textual TUI
```

Python 3.10+. Base install is pure-Python (httpx only). Every heavy runtime is an optional,
host-owned integration.

## Quickstart

### Python

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

### CLI

```bash
uv run worldforge examples                                              # runnable scripts index
uv run worldforge doctor                                                # provider health
uv run worldforge provider list                                         # registered providers
uv run worldforge provider info mock                                    # capability surface
uv run worldforge predict kitchen --provider mock --x 0.3 --y 0.8 --z 0.0 --steps 2
uv run worldforge eval --suite planning --provider mock --format json
uv run worldforge benchmark --provider mock --iterations 5 --format json
```

## Capability Model

WorldForge treats "world model" as an operational interface, not a marketing label.

| Capability | Signature | Example providers |
| --- | --- | --- |
| `predict` | `state + action → predicted state` | `mock` |
| `score` | `observations + goal + candidates → ranked candidates` | `leworldmodel` |
| `policy` | `observation + instruction → action chunks` | `gr00t`, `lerobot` |
| `generate` | `prompt + options → media artifact` | `cosmos`, `runway`, `mock` |
| `transfer` | `artifact + prompt/options → artifact` | `runway`, `mock` |
| `reason` | structured reasoning over state | `mock` |
| `embed` | observation → embedding | `mock` |
| `plan` | facade over composed surfaces | `mock` |

This separation is deliberate. LeWorldModel is a score provider, not a video generator. GR00T and
LeRobot are policy providers, not predictive world models. Cosmos and Runway are media generation
adapters, not controllable physical-planning semantics.

The canonical loop:

```text
observe state
  → propose candidate actions
  → score or roll out possible futures  (score / predict)
  → select an action sequence            (plan)
  → execute through a provider           (policy / predict)
  → persist, evaluate, observe again
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

Scaffold adapters stay outside package exports and auto-registration until they have a validated
runtime path, typed parser coverage, request limits, and docs. The active candidate is
[`jepa-wms`](./docs/src/providers/jepa-wms.md), a direct-construction scaffold targeting future
`facebookresearch/jepa-wms` score-provider work.

## Architecture

```text
  ┌──────────────────────────────────────────────┐
  │  Host application / CLI                      │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │  WorldForge facade                           │
  │  catalog · registry · diagnostics · persist  │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │  World runtime                               │
  │  state · history · planning · execution      │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │  Provider adapter                            │
  │  capability contract · validation · events   │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │  Upstream runtime or API                     │
  │  local model · policy server · media API     │
  └──────────────────────────────────────────────┘
```

| Path | Responsibility |
| --- | --- |
| `src/worldforge/models.py` | Domain models, serialization, validation errors, provider metadata, result types, request policies |
| `src/worldforge/framework.py` | `WorldForge`, `World`, persistence, planning, prediction, comparison, diagnostics |
| `src/worldforge/providers/catalog.py` | In-repo provider factories and auto-registration policy |
| `src/worldforge/providers/base.py` | Provider interfaces, `ProviderError`, remote-provider behavior, `PredictionPayload` |
| `src/worldforge/providers/` | Concrete adapters: mock, Cosmos, Runway, LeWorldModel, GR00T, LeRobot, JEPA, Genie |
| `src/worldforge/evaluation/` | Deterministic evaluation suites and report renderers |
| `src/worldforge/benchmark.py` | Capability-aware latency, retry, throughput, and event benchmark harness |
| `src/worldforge/observability.py` | `ProviderEvent` sinks for logs, recording, and metrics |
| `src/worldforge/testing/` | Reusable provider contract assertions |

Read [architecture](./docs/src/architecture.md) ·
[world-model taxonomy](./docs/src/world-model-taxonomy.md) ·
[provider authoring guide](./docs/src/provider-authoring-guide.md)
before adding a new adapter.

## Demos

Checkout-safe packaged demos run against deterministic injected runtimes — no checkpoints, no
credentials, no GPU:

```bash
uv run worldforge-demo-leworldmodel          # score-based planning end-to-end
uv run worldforge-demo-lerobot               # policy + score planning end-to-end
```

Visual TUI (optional `harness` extra):

```bash
uv run --extra harness worldforge-harness
uv run --extra harness worldforge-harness --flow leworldmodel
uv run --extra harness worldforge-harness --flow lerobot
uv run --extra harness worldforge-harness --flow diagnostics
```

| Flow | Exercises |
| --- | --- |
| `leworldmodel` | Score-provider planning with LeWorldModel-shaped costs, path selection, execution, persistence, reload, events. |
| `lerobot` | Policy-plus-score planning with LeRobot-shaped action chunks, translation, ranking, execution, persistence, reload, events. |
| `diagnostics` | Provider catalog diagnostics and a mock-provider benchmark across predict, reason, generate, transfer. |

Real-checkpoint live smoke (host-provided dependencies and assets):

```bash
uv run --python 3.10 \
  --with "stable-worldmodel[train,env] @ git+https://github.com/galilai-group/stable-worldmodel.git" \
  --with "datasets>=2.21" \
  worldforge-smoke-leworldmodel \
  --stablewm-home ~/.stable-wm \
  --policy pusht/lewm \
  --device cpu
```

See [examples/](./examples) and [`uv run worldforge examples`](./docs/src/examples.md) for the full
runnable index.

## Who It's For

- **Researchers** comparing world-model surfaces without rewriting the harness each time.
- **Robotics and physical-AI engineers** wiring policies, scorers, simulators, and media providers
  around host-owned systems.
- **Framework builders** shipping adapter packages, CLI workflows, and reproducible demos.
- **Enthusiasts** who want something that runs from a clean checkout before installing CUDA stacks
  or downloading checkpoints.

## Operating Boundaries

- Capabilities are contracts. Don't advertise an operation unless the adapter implements it and
  returns the typed WorldForge result.
- Optional runtimes remain host-owned. No torch, LeWorldModel, LeRobot, GR00T, CUDA, TensorRT,
  controllers, checkpoints, or datasets in base dependencies.
- Embodiment-specific action translation is host-owned. Policy providers preserve raw actions; the
  caller converts them into executable `Action` objects.
- Local JSON persistence is single-writer, deterministic. Services needing locking, transactions,
  or migrations own that layer.
- Built-in evaluation suites are deterministic contract harnesses — not physical fidelity, media
  quality, or real-world safety claims.
- Scaffold adapters (`jepa`, `genie`, `jepa-wms`) are reservations. They are never presented as
  real integrations.
- World IDs are local storage identifiers; path separators and traversal-shaped IDs are rejected.

## Development

Primary local gate (same as CI):

```bash
uv lock --check
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run python scripts/generate_provider_docs.py --check
uv run pytest
uv run pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90
bash scripts/test_package.sh
```

Scaffold a new provider:

```bash
uv run python scripts/scaffold_provider.py "Acme WM" \
  --taxonomy "JEPA latent predictive world model" \
  --planned-capability score
```

Full contributor guide: [CONTRIBUTING.md](./CONTRIBUTING.md). Repository agent context:
[AGENTS.md](./AGENTS.md).

## Project Status

WorldForge is **pre-1.0 beta**. Minor releases may still include breaking changes when the public
API needs to tighten.

**Useful today for**

- local provider adapter development
- deterministic planning and evaluation experiments
- checkout-safe demos and optional-runtime smoke tests
- contract testing for third-party provider packages
- CLI diagnostics around provider registration, health, and capabilities

**Known limits**

- `jepa` and `genie` are credential-gated scaffold adapters backed by mock behavior
- `jepa-wms` is a direct-construction candidate, not exported or auto-registered
- local JSON persistence is single-writer only
- evaluation scores are contract signals, not physical-fidelity or safety claims
- optional runtimes, checkpoints, trace export, dashboards, and production telemetry stay
  host-owned

## Citing WorldForge

If you use WorldForge in academic work, please cite it:

```bibtex
@software{worldforge,
  title   = {WorldForge: An integration layer for physical-AI world models},
  author  = {{WorldForge contributors}},
  year    = {2026},
  url     = {https://github.com/AbdelStark/worldforge},
  version = {0.3.0}
}
```

## Contributing

Issues, discussions, and pull requests are welcome. Please read
[CONTRIBUTING.md](./CONTRIBUTING.md) and open an issue for non-trivial changes before sending a
patch. The [provider authoring guide](./docs/src/provider-authoring-guide.md) and
[playbooks](./docs/src/playbooks.md) are the fastest paths to a merge-ready contribution.

## License

WorldForge is released under the [MIT License](./LICENSE).

## Links

- **Documentation** — [docs/src](./docs/src)
- **Quickstart** — [docs/src/quickstart.md](./docs/src/quickstart.md)
- **Playbooks** — [docs/src/playbooks.md](./docs/src/playbooks.md)
- **Architecture** — [docs/src/architecture.md](./docs/src/architecture.md)
- **World-model taxonomy** — [docs/src/world-model-taxonomy.md](./docs/src/world-model-taxonomy.md)
- **Repository** — <https://github.com/AbdelStark/worldforge>
- **Issues** — <https://github.com/AbdelStark/worldforge/issues>
