# WorldForge

**The orchestration layer for world models.**

WorldForge is a developer toolkit that provides a unified interface for building applications on top of world foundation models (WFMs). It abstracts the differences between providers (NVIDIA Cosmos, Runway GWM, Meta JEPA, Google Genie, and experimental Marble local-surrogate support) behind a single, ergonomic API — letting developers focus on what to build rather than how to integrate.

Cross-provider comparison, provider health fanout, and evaluation suite execution run concurrently while preserving deterministic result ordering.

Think LangChain for world models. Or Vercel for physical AI.

## Why WorldForge Exists

The world model ecosystem in 2026 looks like the LLM ecosystem in early 2023:

- **Foundation models exist** (Cosmos Predict/Transfer/Reason, GWM-1 Worlds/Robotics/Avatars, V-JEPA 2, Genie 3, and Marble as an experimental local surrogate)
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

### Embeddings
An `Embedding` is a provider-generated vector representation of text and/or video input. WorldForge treats embeddings as a first-class capability alongside prediction, generation, reasoning, and transfer.

### Plans
A `Plan` is a sequence of actions optimized to reach a goal state. WorldForge can use gradient-based planning (for differentiable world models like JEPA) or sampling-based planning (for generative models like Cosmos/GWM).

### Guardrails
A `Guardrail` is a safety constraint. Define forbidden states, energy thresholds, or physical laws. WorldForge checks every prediction against guardrails before returning results. Optional: ZK verification of guardrail compliance for safety-critical applications.

Prediction and planning requests apply conservative collision and energy checks by default. Use `disable_guardrails=True` in Python, `--disable-guardrails` in the CLI, or `"disable_guardrails": true` in REST payloads to opt out explicitly.

## Quick Example

```python
from worldforge import Action, BBox, Position, SceneObject, SceneObjectPatch, WorldForge
from worldforge.providers import MockProvider

# Initialize with auto-detected providers
wf = WorldForge()
wf.register_provider(MockProvider(name="manual-mock"))

# Create a world and seed it with a scene object
world = wf.create_world("kitchen-counter", provider="manual-mock")
world.add_object(
    SceneObject(
        "red_mug",
        Position(0.0, 0.8, 0.0),
        BBox(Position(-0.05, 0.75, -0.05), Position(0.05, 0.85, 0.05)),
    )
)

# Or bootstrap a starter scene directly from a prompt
seeded = wf.create_world_from_prompt(
    "A kitchen counter with a red mug",
    provider="manual-mock",
    name="kitchen-counter-seeded",
)
assert seeded.object_count > 0

# Inspect and patch scene objects by stable ID
mug = world.objects()[0]
patch = SceneObjectPatch()
patch.set_position(Position(0.1, 0.8, 0.0))
patch.set_graspable(True)
world.update_object_patch(mug.id, patch)
assert world.get_object_by_id(mug.id).is_graspable

# Scene-object JSON payloads also preserve optional mesh geometry and
# visual embeddings across Python, CLI, and REST object workflows.

# Predict the next state
prediction = world.predict(Action.move_to(0.25, 0.8, 0.0), steps=10)

# Check physics plausibility
score = prediction.physics_score  # 0.0 - 1.0

# Plan a sequence of actions to achieve a goal
plan = world.plan(
    goal="spawn cube next to the red mug",
    max_steps=8,
    planner="cem",
)

# Or attach a guardrail-compliance proof while planning
verified_plan = world.plan(
    goal="spawn cube next to the red mug",
    max_steps=8,
    planner="cem",
    verify_backend="mock",  # or "stark" / "ezkl"
)
assert verified_plan.verification_proof is not None

# Or send a structured goal JSON payload for condition or goal-image planning
goal_json = """
{
  "type": "condition",
  "condition": {
    "ObjectAt": {
      "object": "00000000-0000-0000-0000-000000000123",
      "position": {"x": 1.0, "y": 0.8, "z": 0.0},
      "tolerance": 0.05
    }
  }
}
"""
plan = world.plan(goal_json=goal_json, max_steps=4, planner="sampling")

# Execute the materialized plan against the live world state
execution = world.execute_plan(plan, provider="mock")
assert execution.step_count == len(plan.actions)
assert execution.final_world().step == world.step

# Compare provider outputs for the same action
comparison = world.compare(
    Action.move_to(0.5, 0.8, 0.0),
    providers=["mock", "runway"],
    steps=4,
)
best = comparison.best_prediction()
provider_score = comparison.provider_scores()[0]
state_diag = provider_score.state()
pair = comparison.pairwise_agreements()[0]
consensus = comparison.consensus()
assert state_diag.object_preservation_rate >= 0.0
assert pair.object_overlap_rate >= 0.0
assert consensus.shared_object_count >= 0

# Check live provider health
health = wf.provider_health("mock")
assert health.healthy

# Or compare previously captured predictions directly
prediction_2 = world.predict(Action.move_to(0.5, 0.8, 0.0), provider="runway")
comparison_from_predictions = wf.compare([prediction, prediction_2])

# Transfer camera controls onto a generated clip
clip = wf.generate("A robot arm reaching across a workbench", provider="mock")
transferred = wf.transfer(clip, provider="mock")

# Build embeddings from text and/or video
embedding = wf.embed("mock", text="a mug on a kitchen counter")
assert embedding.shape == [32]
assert len(embedding.vector) == 32

# Persist Python-managed worlds with multiple backends
wf = WorldForge(state_backend="sqlite", state_db_path=".worldforge/worldforge.db")
wf.save_world(world)
same_world = wf.load_world(world.id)

wf_msgpack = WorldForge(state_backend="file", state_file_format="msgpack")
wf_msgpack.save_world(world)

wf_redis = WorldForge(
    state_backend="redis",
    state_redis_url="redis://127.0.0.1:6379/0",
)
wf_redis.save_world(world)

wf_s3 = WorldForge(
    state_backend="s3",
    state_s3_bucket="worldforge-states",
    state_s3_region="us-east-1",
    state_s3_access_key_id="test-access",
    state_s3_secret_access_key="test-secret",
    state_s3_prefix="states",
)
wf_s3.save_world(world)

# Export/import portable snapshots from the configured store
snapshot_json = wf.export_world(world.id, format="json")
restored = wf.import_world(snapshot_json, format="json", new_id=True, name="kitchen-copy")
assert restored.id != world.id

snapshot_msgpack = wf_msgpack.export_world(world.id, format="msgpack")
same_world = wf_msgpack.import_world(snapshot_msgpack, format="msgpack")
assert same_world.id == world.id

# Inspect retained state history
history = world.history()
assert len(history) >= 2
assert history[0].action_json is None
assert history[-1].provider == "mock"

# Reconstruct or restore a prior checkpoint
checkpoint = world.history_state(0)
assert checkpoint.step == 0
world.restore_history(0)
assert world.step == 0

# Branch a new scenario from that checkpoint
branch = world.fork(history_index=0, name="kitchen-counter-branch")
assert branch.id != world.id
assert branch.history_length == 1

saved_branch = wf.fork_world(world.id, history_index=0, name="kitchen-counter-branch")
assert saved_branch.name == branch.name
assert saved_branch.history_length == 1
```

The Python package also exposes `worldforge.providers`, `worldforge.eval`, and
`worldforge.verify` as importable submodules. `worldforge.providers` includes
`MarbleProvider` as an experimental local surrogate alongside the other
provider wrappers. Register providers before you create worlds so each `World`
captures the intended registry snapshot.

```python
from worldforge.eval import EvalSuite
from worldforge.verify import ZkVerifier

suite = EvalSuite.from_builtin("physics")
report = suite.run_report_data()
markdown = report.to_markdown()
csv = report.to_csv()
artifacts = suite.run_report_artifacts()
assert markdown.startswith("# Evaluation Report:")
assert "suite,provider,scenario" in csv
assert set(artifacts) == {"json", "markdown", "csv"}

verifier = ZkVerifier(backend="stark")  # or "mock" / "ezkl"
guardrail_bundle = plan.prove_guardrail_bundle()
guardrail_report = verifier.verify_guardrail_bundle(guardrail_bundle)
assert guardrail_report.current_verification.valid

# One prediction is enough to retain both the initial checkpoint and the latest transition
world.predict(Action.move_to(0.35, 0.8, 0.0), steps=2)
inference_bundle = world.prove_latest_inference_bundle()
assert inference_bundle.verify().current_verification.valid

# Archived predictions can be reloaded and verified offline later
archived_prediction = Prediction.from_json(prediction.to_json())
archived_bundle = archived_prediction.prove_inference_bundle("mock")
assert archived_bundle.verify().current_verification.valid
```

Custom `EvalSuite` JSON can embed its default providers. The Python, CLI, and
REST evaluation entry points use those suite defaults when you omit an explicit
provider override, and explicit provider lists still take precedence. REST can
run suites directly via `POST /v1/evals/run`, optionally overlaying either a
persisted `world_id` or an inline `world_state` onto every scenario fixture.

The CLI can export the same evaluation report as multiple artifacts in one run:

```bash
worldforge eval \
  --suite physics \
  --providers mock \
  --output-json /tmp/eval-report.json \
  --output-markdown /tmp/eval-report.md \
  --output-csv /tmp/eval-report.csv

worldforge plan \
  --world <world-id> \
  --goal "spawn cube next to the red mug" \
  --provider mock \
  --fallback-provider mock \
  --verify-backend mock \
  --output-json /tmp/verified-plan.json
```

REST planning accepts the same opt-in switch with a `verification_backend`
field on `POST /v1/worlds/{id}/plan`; when supplied, the returned `Plan`
includes `verification_proof`. Planning also supports a `fallback_provider`
for resilient provider selection across the CLI, REST API, and Python bindings.

## Rust Quickstart

```rust
use worldforge_providers::auto_detect_worldforge;

let wf = auto_detect_worldforge();
let world = wf.create_world("kitchen-counter", "mock")?;
```

## Performance Benchmarks

The workspace now includes offline Criterion harnesses for the core, provider,
evaluation, and verification crates. Run a single suite with `cargo bench -p
<crate>` or compile it without executing the benchmarks with `cargo bench -p
<crate> --no-run`.

```bash
cargo bench -p worldforge-core
cargo bench -p worldforge-providers
cargo bench -p worldforge-eval
cargo bench -p worldforge-verify
```

Evaluation suites can now declare default `providers` in the suite definition.
The CLI, REST API, and Python bindings use that list when callers omit an
explicit provider list, and explicit provider arguments still override the
suite defaults. For example, `worldforge eval --suite physics` uses the suite's
providers, while `worldforge eval --suite-json evals/custom.json --providers
mock,jepa` forces an explicit list.

To attach persistence up front, open a `StateStore` and pass it to
`worldforge_providers::auto_detect_worldforge_with_state_store(...)`.
When `NVIDIA_API_KEY` is present, auto-detection registers a capability-complete
`cosmos` provider covering predict/generate/reason/transfer/embed plus
adapter-native deterministic planning. When `RUNWAY_API_SECRET` is present,
auto-detection registers a `runway` provider covering predict/generate/transfer
plus adapter-native deterministic planning, and that auto-detected `runway`
alias adds Cosmos-backed reasoning only when `NVIDIA_API_KEY` is also present.
Auto-detection also registers `genie` and `marble` as experimental local
surrogates so offline workflows can exercise provider orchestration without a
remote dependency; `MarbleProvider` is distinct from the mock backend and is
surfaced as experimental support rather than a vendor-faithful API binding.
Unlike the mock backend, Marble now exposes deterministic native planning in
addition to prediction, generation, reasoning, transfer, and embedding.

The same embedding surface is exposed over the CLI as `worldforge embed` and
over the REST API as `POST /v1/providers/{name}/embed`.

Provider-scoped reasoning is also available without persisting a world first.
In Python, `WorldForge()` auto-registers Genie as a local surrogate, so you
can use it immediately. `GENIE_API_KEY` and `GENIE_API_ENDPOINT` are optional
remote hints, not discovery gates. You can still supply a world snapshot, a
JSON snapshot, a clip, or any combination of those inputs:

```python
from worldforge import WorldForge

wf = WorldForge()
assert "genie" in wf.providers()
world = wf.create_world_from_prompt("A kitchen counter with a mug", provider="mock")
reasoning = wf.reason(
    "genie",
    "how many objects are here?",
    world_json=world.to_json(),
)
assert "object" in reasoning.answer
```

The same stateless surface is available over REST:

```bash
curl -X POST http://127.0.0.1:8080/v1/providers/mock/reason \
  -H 'content-type: application/json' \
  -d '{"query":"what do you see?","video":{"frames":[],"fps":8.0,"resolution":[64,64],"duration":1.0}}'
```

The CLI `reason` command follows the same provider-scoped pattern for either
stored worlds or direct snapshot/video inputs.

Portable world snapshots are now first-class across the user surfaces:
- Python: `WorldForge.export_world(...)` / `WorldForge.import_world(...)`
- CLI: `worldforge export ...` / `worldforge import ...`
- REST: `GET /v1/worlds/{id}/export?format=json|msgpack` and `POST /v1/worlds/import`

REST export returns a JSON envelope with `format`, `encoding`, `sha256`, and
`snapshot`. JSON snapshots use `encoding: "utf-8"`, while MessagePack snapshots
use `encoding: "hex"` so they stay text-safe over HTTP. Import accepts either a
raw `state` object or the exported snapshot payload shape.

Recoverable history checkpoints are also available end-to-end:
- Python: `world.history_state(index)` / `world.restore_history(index)`
- CLI: `worldforge restore --world <id> --history-index <n>`
- REST: `POST /v1/worlds/{id}/restore` with `{"history_index": <n>}`

## Python Installation

The Python bindings now ship as an installable package from this repository via
[`maturin`](https://github.com/PyO3/maturin):

```bash
bash scripts/test_python_package.sh
```

That script creates an isolated virtual environment, installs the package in
editable mode, verifies that `worldforge` and its submodules import from the
installed package, and then runs `python -m unittest discover -s python/tests`
against the installed bindings. The editable install is defined by
[`crates/worldforge-python/Cargo.toml`](./crates/worldforge-python/Cargo.toml)
and the root [`pyproject.toml`](./pyproject.toml), so no separate global
`maturin` installation is required.

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
│   │       ├── mock.rs         # Mock provider (scene-aware offline reference backend)
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
cargo run -p worldforge-cli -- create --prompt "A kitchen with a mug" --name kitchen-counter
cargo run -p worldforge-cli -- objects add --world <id> --name red_mug --position 0 0.8 0 --bbox-min -0.05 0.75 -0.05 --bbox-max 0.05 0.85 0.05 --semantic-label mug
cargo run -p worldforge-cli -- objects list --world <id>
cargo run -p worldforge-cli -- objects show --world <id> --object-id <object-id>
cargo run -p worldforge-cli -- objects update --world <id> --object-id <object-id> --position 0.25 0.8 0.0 --semantic-label mug
cargo run -p worldforge-cli -- objects remove --world <id> --object-id <object-id>
cargo run -p worldforge-cli -- providers
cargo run -p worldforge-cli -- providers --capability planning
cargo run -p worldforge-cli -- providers --health
cargo run -p worldforge-cli -- estimate --provider cosmos --operation generate --duration-seconds 5 --width 1280 --height 720
cargo run -p worldforge-cli -- list
cargo run -p worldforge-cli -- --state-backend sqlite --state-db-path .worldforge/worldforge.db list
cargo run -p worldforge-cli -- --state-backend redis --state-redis-url redis://127.0.0.1:6379/0 list
cargo run -p worldforge-cli -- --state-backend s3 --state-s3-bucket worldforge-states --state-s3-region us-east-1 --state-s3-access-key-id test-access --state-s3-secret-access-key test-secret --state-s3-prefix states list
cargo run -p worldforge-cli -- --state-file-format msgpack list
cargo run -p worldforge-cli -- export --world <id> --output snapshots/world.json
cargo run -p worldforge-cli -- export --world <id> --output snapshots/world.msgpack --format msgpack
cargo run -p worldforge-cli -- import --input snapshots/world.msgpack --format msgpack --new-id --name kitchen-copy
cargo run -p worldforge-cli -- history --world <id> --output-json histories/<id>.json
cargo run -p worldforge-cli -- restore --world <id> --history-index 0 --output-json restored/<id>.json
cargo run -p worldforge-cli -- predict --world <id> --action "move 1 0 0" --provider runway --fallback-provider mock --timeout-ms 500
cargo run -p worldforge-cli -- compare --prediction-json runs/mock.json --prediction-json runs/runway.json --output-json reports/compare.json
cargo run -p worldforge-cli -- compare --world-snapshot snapshots/world.msgpack --action "move 1 0 0" --providers mock,runway --output-json reports/compare-from-snapshot.json
cargo run -p worldforge-cli -- plan --world <id> --goal "spawn cube" --planner cem
cargo run -p worldforge-cli -- plan --world <id> --goal "spawn cube" --planner cem --guardrails-json guardrails.json --output-json plans/generated.json
cargo run -p worldforge-cli -- plan --world <id> --goal-json goals/object-at.json --planner sampling --output-json plans/object-at.json
cargo run -p worldforge-cli -- execute-plan --world <id> --plan-json plans/generated.json --output-json plans/executed.json
cargo run -p worldforge-cli -- generate --provider mock --prompt "A cube rolling across a table" --duration-seconds 5 --output-json clips/generated.json
cargo run -p worldforge-cli -- transfer --provider mock --source-json clips/generated.json --output-json clips/transferred.json
cargo run -p worldforge-cli -- reason --world <id> --query "Will the mug fall if pushed?"
cargo run -p worldforge-cli -- verify --backend stark --proof-type guardrail --plan-json plans/generated.json --output-json proofs/guardrail.json
cargo run -p worldforge-cli -- verify --proof-type guardrail --world <id> --goal-json goals/object-at.json --output-json proofs/object-at-guardrail.json
cargo run -p worldforge-cli -- verify --proof-type inference --input-state-json states/before.json --output-state-json states/after.json --provider mock
cargo run -p worldforge-cli -- verify --proof-type inference --prediction-json runs/mock.json --output-json proofs/inference-from-prediction.json
cargo run -p worldforge-cli -- verify-proof --guardrail-bundle-json proofs/guardrail.json --output-json proofs/guardrail-report.json
cargo run -p worldforge-cli -- verify-proof --proof-json proofs/raw-proof.json
cargo run -p worldforge-cli -- eval --list-suites
cargo run -p worldforge-cli -- eval --suite physics
cargo run -p worldforge-cli -- eval --suite physics --world-snapshot snapshots/world.json --output-json reports/eval-from-snapshot.json
cargo run -p worldforge-cli -- eval --suite physics --world <id>
cargo run -p worldforge-cli -- eval --suite-json evals/custom.json
cargo run -p worldforge-cli -- eval --suite-json evals/custom.json --providers mock,jepa --output-json reports/custom-eval.json
cargo run -p worldforge-cli -- serve --bind 127.0.0.1:8080

# Or run the dedicated server binary
cargo run -p worldforge-server -- --bind 127.0.0.1:8080 --state-dir .worldforge
cargo run -p worldforge-server -- --bind 127.0.0.1:8080 --state-dir .worldforge --state-file-format msgpack

# Build and smoke-test the Python package
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s python/tests

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
worldforge-server --bind 127.0.0.1:8080 --state-dir .worldforge --state-file-format msgpack
worldforge-server --bind 127.0.0.1:8080 --state-backend sqlite --state-db-path .worldforge/worldforge.db
worldforge-server --bind 127.0.0.1:8080 --state-backend redis --state-redis-url redis://127.0.0.1:6379/0
worldforge-server --bind 127.0.0.1:8080 --state-backend s3 --state-s3-bucket worldforge-states --state-s3-region us-east-1 --state-s3-access-key-id test-access --state-s3-secret-access-key test-secret --state-s3-prefix states
```

Then call the HTTP API directly:

The server now enforces a few stricter transport rules that are useful in real
clients and tests:
- malformed request lines or invalid `Content-Length` headers return `400`
- known paths return `405 Method Not Allowed` with an `Allow` header when the method is wrong
- `HEAD` is supported on the existing `GET` endpoints
- percent-encoded query parameters are decoded before routing and filtering
- rendered evaluation reports still return JSON envelopes when requested through the API
  with the rendered markdown/CSV exposed in the response payload
- evaluation requests can fan out multiple artifact formats from one run via
  `report_formats`
```bash
curl -X POST http://127.0.0.1:8080/v1/worlds \
  -H 'content-type: application/json' \
  -d '{"prompt":"A kitchen with a mug","name":"Kitchen counter","provider":"mock"}'

curl "http://127.0.0.1:8080/v1/worlds/<world-id>/export?format=json"

curl "http://127.0.0.1:8080/v1/worlds/<world-id>/export?format=msgpack"

curl -X POST http://127.0.0.1:8080/v1/worlds/import \
  -H 'content-type: application/json' \
  -d '{
    "format":"json",
    "encoding":"utf-8",
    "sha256":"<snapshot-sha256>",
    "snapshot":"{...json snapshot...}",
    "new_id":true,
    "name":"Kitchen copy"
  }'

curl -X POST http://127.0.0.1:8080/v1/worlds/import \
  -H 'content-type: application/json' \
  -d '{
    "format":"msgpack",
    "encoding":"hex",
    "sha256":"<snapshot-sha256>",
    "snapshot":"<hex-encoded-msgpack>",
    "new_id":true
  }'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/restore \
  -H 'content-type: application/json' \
  -d '{"history_index":0}'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/objects \
  -H 'content-type: application/json' \
  -d '{"name":"red_mug","position":{"x":0.0,"y":0.8,"z":0.0},"bbox":{"min":{"x":-0.05,"y":0.75,"z":-0.05},"max":{"x":0.05,"y":0.85,"z":0.05}},"semantic_label":"mug"}'

curl http://127.0.0.1:8080/v1/worlds/<world-id>/objects

curl http://127.0.0.1:8080/v1/worlds/<world-id>/objects/<object-id>

curl -X PATCH http://127.0.0.1:8080/v1/worlds/<world-id>/objects/<object-id> \
  -H 'content-type: application/json' \
  -d '{"position":{"x":0.25,"y":0.8,"z":0.0},"semantic_label":"mug"}'

curl -X DELETE http://127.0.0.1:8080/v1/worlds/<world-id>/objects/<object-id>

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/plan \
  -H 'content-type: application/json' \
  -d '{"goal":"spawn cube","planner":"cem","population_size":12,"elite_fraction":0.25,"num_iterations":3,"guardrails":[{"guardrail":"NoCollisions","blocking":true}]}'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/plan \
  -H 'content-type: application/json' \
  -d '{"goal":{"type":"condition","condition":{"ObjectAt":{"object":"<object-id>","position":{"x":1.0,"y":0.8,"z":0.0},"tolerance":0.05}}},"planner":"sampling","num_samples":48,"top_k":5}'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/execute-plan \
  -H 'content-type: application/json' \
  -d @plans/execution-request.json

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/verify \
  -H 'content-type: application/json' \
  -d '{"backend":"stark","proof_type":"guardrail","goal":"spawn cube","guardrails":[{"guardrail":"NoCollisions","blocking":true}]}'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/verify \
  -H 'content-type: application/json' \
  -d '{"proof_type":"guardrail","goal":{"type":"condition","condition":{"ObjectAt":{"object":"<object-id>","position":{"x":1.0,"y":0.8,"z":0.0},"tolerance":0.05}}},"guardrails":[{"guardrail":"NoCollisions","blocking":true}]}'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/verify \
  -H 'content-type: application/json' \
  -d @proofs/inference-from-prediction-request.json

curl -X POST http://127.0.0.1:8080/v1/verify/proof \
  -H 'content-type: application/json' \
  -d @proofs/verify-request.json

For inference verification, submit exactly one of `prediction`, both
`input_state` and `output_state`, or neither to reuse the latest archived world
transition. Verification backends are serialized as lowercase strings:
`mock`, `ezkl`, and `stark`.

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

curl -X POST http://127.0.0.1:8080/v1/evals/run \
  -H 'content-type: application/json' \
  -d '{"suite":"physics","providers":["mock"]}'

curl -X POST http://127.0.0.1:8080/v1/evals/run \
  -H 'content-type: application/json' \
  -d '{"suite":"physics","providers":["mock"],"world_id":"<world-id>","report_format":"markdown"}'

curl -X POST http://127.0.0.1:8080/v1/evals/run \
  -H 'content-type: application/json' \
  -d '{"suite":"physics","providers":["mock"],"report_formats":["json","markdown","csv"]}'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/evaluate \
  -H 'content-type: application/json' \
  -d '{"suite":"physics","providers":["mock"]}'

curl -X POST http://127.0.0.1:8080/v1/worlds/<world-id>/evaluate \
  -H 'content-type: application/json' \
  -d '{"suite":"physics","providers":["mock"],"report_format":"markdown"}'

curl http://127.0.0.1:8080/v1/providers

curl http://127.0.0.1:8080/v1/providers?capability=predict

curl http://127.0.0.1:8080/v1/providers?health=true

curl http://127.0.0.1:8080/v1/providers/mock

curl http://127.0.0.1:8080/v1/providers/mock/health

curl -X POST http://127.0.0.1:8080/v1/providers/mock/estimate \
  -H 'content-type: application/json' \
  -d '{"operation":{"Generate":{"duration_seconds":5.0,"resolution":[1280,720]}}}'

# `compare-body.json` should contain `{"predictions":[...]}`
# using previously exported `Prediction` JSON payloads.
curl -X POST http://127.0.0.1:8080/v1/compare \
  -H 'content-type: application/json' \
  -d @compare-body.json

# `compare-state-body.json` should contain
# `{"world_state": {...}, "action": {...}, "providers": ["mock","runway"]}`.
curl -X POST http://127.0.0.1:8080/v1/compare \
  -H 'content-type: application/json' \
  -d @compare-state-body.json
```

## Status

Pre-alpha. Core types, provider trait, state management, guardrails, evaluation
framework, CLI, server, Python bindings, and the mock plus JEPA local providers
are implemented. Prediction fallback and timeout handling are wired through the
core orchestration layer and exposed in the CLI, REST server, and Python API.
Planning now supports distinct sampling, CEM, MPC, gradient, and provider-native
execution paths in the core, with planner selection exposed across the CLI,
REST server, and Python bindings. Provider-native planning now dispatches
through an explicit provider hook instead of aliasing core heuristics, with the
local JEPA adapter and the full-stack Cosmos and Runway adapters supplying
deterministic adapter-native plans on top of WorldForge-managed surrogate
dynamics rather than vendor planning endpoints. Heuristic planners now parse
relational natural-language goals like spawning an object next to a named
anchor instead of collapsing those requests into plain anchor existence checks.
Planning requests can also set `fallback_provider` so the core retries against
another registered provider when the primary planner or prediction path fails.
Direct provider generation and transfer now flow through the shared Rust
`WorldForge` facade and are exposed across the CLI, REST server, and Python
bindings as well, with REST requests defaulting to each stored world's
configured provider instead of hard-coding `mock`. Stateless
provider reasoning now accepts either a persisted world snapshot or a raw video
clip in Python and over REST, and Python registers Genie as a local surrogate by
default. `GENIE_API_KEY` and `GENIE_API_ENDPOINT` remain optional remote hints,
not discovery gates. The CLI
`eval` command can now overlay a persisted world onto each scenario fixture via
`--world` or an exported snapshot via `--world-snapshot`, which keeps the public
evaluation workflow aligned with stored state without dropping suite-specific
setup. Cross-provider comparison now accepts portable world snapshots in the
CLI and inline `world_state` payloads over REST in addition to persisted
`world_id`s and offline prediction JSON inputs. Provider transfer is now exposed end-to-end in
the core, CLI, REST server, and Python bindings with JSON clip round-tripping
for reusable workflows. File-backed (JSON or MessagePack), SQLite-backed, Redis-backed, and
S3-backed world persistence are all supported through the shared `StateStore`
abstraction across the core, CLI, REST server, and Python bindings. Use
`state_s3_bucket` / `--state-s3-bucket`, `state_s3_region` /
`--state-s3-region`, and the S3 access key fields for deployment. Cosmos and
Runway adapters have API wiring in place,
and Genie ships as a deterministic low-resolution surrogate backend for
interactive world generation, scene-grounded reasoning, controlled transfer,
and provider-native planning. `GENIE_API_KEY` and `GENIE_API_ENDPOINT` are
optional remote hints, not discovery gates. Planning now
accepts serialized guardrail configurations across the CLI, REST server, and
Python bindings, and verification now operates on explicit state transitions or
real generated plans instead of placeholder proof inputs. The CLI can export
plan JSON for reuse, and the REST server can generate guardrail proofs directly
from a goal plus guardrail set. Exported proofs and verification bundles can
now be re-verified offline across the CLI, REST server, and Python bindings,
and verification inputs are hashed with real SHA-256 digests. Verification
backend selection is now exposed end to end across the verify crate, CLI, REST
server, and Python bindings for the deterministic `mock`, `ezkl`, and `stark`
compatibility paths. Cross-provider comparison now reuses the same guardrail
and fallback pipeline as single-provider prediction, with comparison config
exposed in the CLI, REST server, and Python bindings. The CLI and REST API can
also compare previously captured `Prediction` payloads directly, so expensive
provider runs can be analyzed offline without replaying inference, and archived
predictions now carry provenance into reusable inference-verification bundles
across the verify crate, CLI, REST server, and Python bindings. Evaluation now supports
built-in suite discovery, JSON-defined custom suites, suite-level default
providers, provider selection, and aggregated leaderboard, provider, scenario,
and dimension rollups. When a suite declares `providers`, the CLI, REST API,
and Python bindings use that list by default; explicit provider arguments still
override it. Custom suites can now assert concrete scene outcomes such as final
object positions and semantic labels, and can score deterministic clips against
optional ground-truth video references. They can also assert final-state
conditions using the core `Condition` semantics for relational checks. Named
custom dimensions now resolve to real metrics instead of placeholder labels:
`overall`, the built-in physics score keys, `confidence`,
`video_similarity`, and derived dimensions such as
`action_prediction_accuracy`, `spatial_reasoning`, and
`material_understanding` when the suite definition and scenario evidence
support them. Structured
`condition` and `goal_image` planning payloads are exercised end to end across
the CLI, REST server, and Python bindings, and serialized plans can now be
executed against persisted worlds through each surface with atomic state
commit-on-success semantics. Scene object seeding and inspection are
now exposed across the CLI and REST server as first-class operations instead of
requiring direct JSON state editing, and Python scene objects can round-trip
through JSON for interop with those workflows. Provider discovery now exposes
capability metadata across the CLI, REST server, and Python bindings, and
provider adapters' cost estimates are queryable end-to-end for predict,
generate, reason, and transfer operations. Registry-level provider health
reporting is now available across the core, CLI, REST server, and Python
bindings, including optional live health data when listing providers. The
Python bindings now reuse the shared Rust core facade for provider discovery,
health checks, prediction comparison, and state persistence instead of
duplicating orchestration logic.
The mock provider now serves as a higher-fidelity offline reference backend:
object motion keeps bounding boxes and inferred relationships in sync,
predictions can emit lightweight preview video/depth/segmentation outputs, and
reasoning answers are grounded in the current scene instead of fixed strings.
Python scene management now exposes stable object-ID lookup, full object
listing, and partial patch updates in addition to the original name-based
helpers, which brings the Python surface in line with the CLI and REST object
editing workflows.

## License

Apache 2.0 (core library)

## Links

- [Specification](./SPECIFICATION.md)
- [Architecture Decision Records](./architecture/)
- [Contributing](./CONTRIBUTING.md)
