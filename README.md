# WorldForge

**The orchestration layer for world models.**

WorldForge is a developer toolkit that provides a unified interface for building applications on top of world foundation models (WFMs). It abstracts the differences between providers (NVIDIA Cosmos, Runway GWM, Meta JEPA, Google Genie, and others) behind a single, ergonomic API — letting developers focus on what to build rather than how to integrate.

Think LangChain for world models. Or Vercel for physical AI.

## Why WorldForge Exists

The world model ecosystem in 2026 looks like the LLM ecosystem in early 2023:

- **Foundation models exist** (Cosmos Predict/Transfer/Reason, GWM-1 Worlds/Robotics/Avatars, V-JEPA 2, Genie 3, Marble)
- **Each has its own SDK** (NVIDIA NIM + NGC, Runway Python/Node SDK, Meta research code, Google research preview)
- **No unified abstraction** — every developer writes custom integration code for each provider
- **No orchestration** — composing multi-step workflows (predict → evaluate → plan → verify) requires manual plumbing
- **No persistent state** — world models are stateless; managing scene state across calls is the developer's problem
- **No safety layer** — guardrails are provider-specific and non-portable

WorldForge fills every gap.

## Core Concepts

### Worlds
A `World` is a persistent, stateful environment. It wraps a world model provider and maintains scene state, history, and configuration across inference calls.

### Actions
An `Action` is a standardized representation of something an agent can do. Move, grasp, rotate, navigate — defined once, translated to each provider's format automatically.

### Predictions
A `Prediction` is the result of asking a world model "what happens next?" It contains the predicted future state, confidence scores, and optional video/image output.

### Plans
A `Plan` is a sequence of actions optimized to reach a goal state. WorldForge can use gradient-based planning (for differentiable world models like JEPA) or sampling-based planning (for generative models like Cosmos/GWM).

### Guardrails
A `Guardrail` is a safety constraint. Define forbidden states, energy thresholds, or physical laws. WorldForge checks every prediction against guardrails before returning results. Optional: ZK verification of guardrail compliance for safety-critical applications.

## Quick Example

```python
from worldforge import WorldForge, World, Action
from worldforge.providers import CosmosProvider, RunwayProvider

# Initialize with any provider
wf = WorldForge(provider=CosmosProvider(model="cosmos-predict-2.5"))

# Create a world from a text description
world = wf.create_world("A kitchen counter with a red mug and a plate")

# Predict what happens if we push the mug
prediction = world.predict(
    action=Action.push(target="red_mug", direction="left", force=0.5),
    steps=10
)

# Check physics plausibility
score = prediction.physics_score()  # 0.0 - 1.0

# Plan a sequence of actions to achieve a goal
plan = world.plan(
    goal="The red mug is inside the dishwasher",
    max_steps=20,
    planner="cem",
    guardrails=["no_collisions", "mug_stays_upright"]
)

# Switch providers seamlessly
world.set_provider(RunwayProvider(model="gwm-1-robotics"))
prediction_2 = world.predict(action=plan.actions[0])

# Compare predictions across providers
comparison = wf.compare([prediction, prediction_2])

# Transfer camera controls onto a generated clip
clip = wf.generate("A robot arm reaching across a workbench", provider="mock")
transferred = wf.transfer(clip, provider="mock")

# Persist Python-managed worlds with either backend
wf = WorldForge(state_backend="sqlite", state_db_path=".worldforge/worldforge.db")
wf.save_world(world)
same_world = wf.load_world(world.id)
```

## Architecture

```
worldforge/
├── crates/
│   ├── worldforge-core/        # Core library: types, traits, state, orchestration
│   │   └── src/
│   │       ├── lib.rs          # WorldForge entry point + provider registry
│   │       ├── types.rs        # Tensor, spatial, temporal, media types
│   │       ├── world.rs        # World orchestration + planning
│   │       ├── action.rs       # Action type system (18 variants)
│   │       ├── prediction.rs   # Prediction engine, multi-provider comparison
│   │       ├── provider.rs     # WorldModelProvider trait + registry
│   │       ├── scene.rs        # Scene graph (objects, relationships, physics)
│   │       ├── guardrail.rs    # Safety constraints (7 guardrail types)
│   │       ├── state.rs        # State persistence (file + SQLite stores)
│   │       └── error.rs        # WorldForgeError enum (18 variants)
│   ├── worldforge-providers/   # Provider adapters
│   │   └── src/
│   │       ├── mock.rs         # Mock provider (deterministic, for testing)
│   │       ├── cosmos.rs       # NVIDIA Cosmos adapter
│   │       ├── runway.rs       # Runway GWM-1 adapter
│   │       ├── jepa.rs         # Meta V-JEPA adapter
│   │       └── genie.rs        # Google Genie adapter
│   ├── worldforge-eval/        # Evaluation framework (4 built-in suites)
│   ├── worldforge-verify/      # ZK verification (optional)
│   ├── worldforge-server/      # REST API server (Tokio TCP)
│   ├── worldforge-cli/         # CLI tool (Clap)
│   └── worldforge-python/      # Python bindings (PyO3)
├── SPECIFICATION.md            # Technical specification (source of truth)
├── architecture/ADR.md         # Architecture Decision Records
└── CONTRIBUTING.md             # Development setup guide
```

## Development

```bash
# Build
cargo build

# Test
cargo test

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt

# Run CLI
cargo run -p worldforge-cli -- create --prompt "A kitchen with a mug"
cargo run -p worldforge-cli -- providers
cargo run -p worldforge-cli -- providers --capability planning
cargo run -p worldforge-cli -- estimate --provider cosmos --operation generate --duration-seconds 5 --width 1280 --height 720
cargo run -p worldforge-cli -- list
cargo run -p worldforge-cli -- --state-backend sqlite --state-db-path .worldforge/worldforge.db list
cargo run -p worldforge-cli -- predict --world <id> --action "move 1 0 0" --provider runway --fallback-provider mock --timeout-ms 500
cargo run -p worldforge-cli -- plan --world <id> --goal "spawn cube" --planner cem
cargo run -p worldforge-cli -- plan --world <id> --goal "spawn cube" --planner cem --guardrails-json guardrails.json --output-json plans/generated.json
cargo run -p worldforge-cli -- generate --provider mock --prompt "A cube rolling across a table" --duration-seconds 5 --output-json clips/generated.json
cargo run -p worldforge-cli -- transfer --provider mock --source-json clips/generated.json --output-json clips/transferred.json
cargo run -p worldforge-cli -- reason --world <id> --query "Will the mug fall if pushed?"
cargo run -p worldforge-cli -- verify --proof-type guardrail --plan-json plans/generated.json --output-json proofs/guardrail.json
cargo run -p worldforge-cli -- verify --proof-type inference --input-state-json states/before.json --output-state-json states/after.json --provider mock
cargo run -p worldforge-cli -- verify-proof --guardrail-bundle-json proofs/guardrail.json --output-json proofs/guardrail-report.json
cargo run -p worldforge-cli -- verify-proof --proof-json proofs/raw-proof.json
cargo run -p worldforge-cli -- eval --list-suites
cargo run -p worldforge-cli -- eval --suite physics
cargo run -p worldforge-cli -- eval --suite-json evals/custom.json --providers mock,jepa --output-json reports/custom-eval.json
cargo run -p worldforge-cli -- serve --bind 127.0.0.1:8080

# Or run the dedicated server binary
cargo run -p worldforge-server -- --bind 127.0.0.1:8080 --state-dir .worldforge

# Use auto-detected local JEPA weights from the CLI
JEPA_MODEL_PATH=/path/to/v-jepa-2 cargo run -p worldforge-cli -- create --prompt "A lab bench" --provider jepa
JEPA_MODEL_PATH=/path/to/v-jepa-2 cargo run -p worldforge-cli -- health jepa
```

## REST API

Start the server with either the CLI or the dedicated binary:

```bash
worldforge serve --bind 127.0.0.1:8080
# or
worldforge-server --bind 127.0.0.1:8080 --state-dir .worldforge
worldforge-server --bind 127.0.0.1:8080 --state-backend sqlite --state-db-path .worldforge/worldforge.db
```

Then call the HTTP API directly:

```bash
curl -X POST http://127.0.0.1:8080/v1/worlds \
  -H 'content-type: application/json' \
  -d '{"name":"Kitchen counter","provider":"mock"}'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/plan \
  -H 'content-type: application/json' \
  -d '{"goal":"spawn cube","planner":"cem","population_size":12,"elite_fraction":0.25,"num_iterations":3,"guardrails":[{"guardrail":"NoCollisions","blocking":true}]}'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/verify \
  -H 'content-type: application/json' \
  -d '{"proof_type":"guardrail","goal":"spawn cube","guardrails":[{"guardrail":"NoCollisions","blocking":true}]}'

curl -X POST http://127.0.0.1:8080/v1/verify/proof \
  -H 'content-type: application/json' \
  -d @proofs/verify-request.json

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/reason \
  -H 'content-type: application/json' \
  -d '{"query":"Will the spawned cube stay stable?"}'

curl -X POST http://127.0.0.1:8080/v1/providers/mock/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"A cube rolling across the floor","config":{"duration_seconds":5.0}}'

curl -X POST http://127.0.0.1:8080/v1/providers/mock/transfer \
  -H 'content-type: application/json' \
  -d '{"source":{"frames":[],"fps":12.0,"resolution":[640,360],"duration":5.0},"controls":{},"config":{"resolution":[1280,720],"fps":24.0,"control_strength":0.8}}'

curl http://127.0.0.1:8080/v1/evals/suites

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/evaluate \
  -H 'content-type: application/json' \
  -d '{"suite":"physics","providers":["mock"]}'

curl http://127.0.0.1:8080/v1/providers

curl http://127.0.0.1:8080/v1/providers?capability=predict

curl http://127.0.0.1:8080/v1/providers/mock

curl -X POST http://127.0.0.1:8080/v1/providers/mock/estimate \
  -H 'content-type: application/json' \
  -d '{"operation":{"Generate":{"duration_seconds":5.0,"resolution":[1280,720]}}}'
```

## Status

Pre-alpha. Core types, provider trait, state management, guardrails, evaluation
framework, CLI, server, Python bindings, and the mock plus JEPA local providers
are implemented. Prediction fallback and timeout handling are wired through the
core orchestration layer and exposed in the CLI, REST server, and Python API.
Planning now supports distinct sampling, CEM, MPC, gradient, and provider-native
execution paths in the core, with planner selection exposed across the CLI,
REST server, and Python bindings. Direct provider generation and world-state
reasoning are now exposed across the CLI, REST server, and Python bindings as
well, with REST requests defaulting to each stored world's configured provider
instead of hard-coding `mock`. Provider transfer is now exposed end-to-end in
the core, CLI, REST server, and Python bindings with JSON clip round-tripping
for reusable workflows. File-backed and SQLite-backed world persistence are
both supported through the shared `StateStore` abstraction across the core,
CLI, REST server, and Python bindings. Cosmos and Runway adapters have API wiring in place,
while Genie remains a research-preview stub pending public access. Planning now
accepts serialized guardrail configurations across the CLI, REST server, and
Python bindings, and verification now operates on explicit state transitions or
real generated plans instead of placeholder proof inputs. The CLI can export
plan JSON for reuse, and the REST server can generate guardrail proofs directly
from a goal plus guardrail set. Exported proofs and verification bundles can
now be re-verified offline across the CLI, REST server, and Python bindings,
and verification inputs are hashed with real SHA-256 digests. Evaluation now
supports built-in suite discovery, JSON-defined custom suites, provider
selection, and CLI report export across the CLI, REST server, and Python
bindings. Provider discovery now exposes capability metadata across the CLI,
REST server, and Python bindings, and provider adapters' cost estimates are
queryable end-to-end for predict, generate, reason, and transfer operations.

## License

Apache 2.0 (core library)

## Links

- [Specification](./SPECIFICATION.md)
- [Architecture Decision Records](./architecture/)
- [Contributing](./CONTRIBUTING.md)
