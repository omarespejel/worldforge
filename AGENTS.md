# Agent Guide

## Project Identity

WorldForge is a Python integration layer for physical-AI world-model workflows: provider adapters,
world state, planning, evaluation, benchmarking, diagnostics, and host-owned optional runtimes. It
is a library and CLI, not a hosted service, on-chain contract, or end-user application.

Intended users are Python developers building provider adapters, local world-model experiments,
evaluation harnesses, and testable prototypes.

## Architecture Map

- `src/worldforge/models.py`: domain models, serialization helpers, validation errors, request
  policies, provider metadata, media/result types, and structured planning goals.
- `src/worldforge/framework.py`: `WorldForge`, `World`, persistence, planning, prediction,
  comparison, diagnostics, and top-level evaluation helpers.
- `src/worldforge/providers/base.py`: provider interfaces, `ProviderError`, remote-provider
  base behavior, and `PredictionPayload`.
- `src/worldforge/providers/catalog.py`: provider factories and auto-registration policy for the
  in-repo provider catalog.
- `src/worldforge/providers/mock.py`: deterministic local provider used by tests, examples, and
  contract checks.
- `src/worldforge/providers/cosmos.py` and `runway.py`: real HTTP adapters with typed timeout,
  retry, polling, download policies, and response parsers.
- `src/worldforge/providers/leworldmodel.py`: real optional LeWorldModel JEPA cost-model adapter
  for scoring action candidates through `stable_worldmodel.policy.AutoCostModel`.
- `src/worldforge/providers/gr00t.py`: experimental host-owned NVIDIA Isaac GR00T PolicyClient
  adapter for selecting embodied action chunks through the `policy` capability.
- `src/worldforge/providers/lerobot.py`: host-owned Hugging Face LeRobot `PreTrainedPolicy`
  adapter for selecting embodied action chunks through the `policy` capability.
- `src/worldforge/providers/jepa_wms.py`: candidate contract scaffold for
  `facebookresearch/jepa-wms` score-provider work; it supports injected test/runtime scoring and a
  host-owned torch-hub runtime but is intentionally not exported or registered.
- `src/worldforge/providers/remote.py`: credential-gated scaffold providers for `jepa` and
  `genie`; these intentionally use deterministic mock behavior after credential checks.
- `src/worldforge/evaluation/`: built-in generation, physics, planning, reasoning, and transfer
  suites plus report renderers.
- `src/worldforge/benchmark.py`: capability-aware provider latency, retry, and throughput harness.
- `src/worldforge/observability.py`: composable `ProviderEvent` sinks for JSON logging, in-memory
  recording, and metrics aggregation.
- `src/worldforge/testing/`: reusable adapter contract helpers.
- `src/worldforge/demos/`: packaged demo entry points exposed through `uv run` console scripts.
- `src/worldforge/demos/lerobot_e2e.py`: packaged LeRobot policy-plus-score planning demo exposed
  through `uv run worldforge-demo-lerobot`.
- `src/worldforge/harness/`: optional TheWorldHarness TUI package. Keep flow metadata and runners
  independent from Textual; `tui.py` is the only Textual-dependent module. Current flows cover
  LeWorldModel score planning, LeRobot policy-plus-score planning, and provider diagnostics plus
  benchmark comparison.
- `src/worldforge/smoke/`: packaged optional-runtime smoke entry points exposed through `uv run`
  console scripts.
- `src/worldforge/smoke/lerobot_leworldmodel.py`: optional host-owned real robotics showcase that
  composes a LeRobot policy checkpoint with a LeWorldModel score checkpoint through
  `World.plan(..., planning_mode="policy+score")`.
- `src/worldforge/smoke/robotics_showcase.py`: one-command PushT real robotics showcase that wires
  the packaged PushT observation, score, translator, and candidate bridge defaults into
  `lewm-lerobot-real`.
- `src/worldforge/smoke/pusht_showcase_inputs.py`: packaged PushT showcase hooks for building the
  LeRobot observation, LeWorldModel score tensors, and checkpoint-native action candidates.
- `src/worldforge/smoke/leworldmodel_checkpoint.py`: optional host-owned builder for creating the
  LeWorldModel `*_object.ckpt` file expected by `AutoCostModel` from Hugging Face LeWM assets.
- `examples/leworldmodel_e2e_demo.py`: checkout-safe end-to-end LeWorldModel provider-surface
  score-planning compatibility wrapper for `uv run worldforge-demo-leworldmodel`.
- `examples/lerobot_e2e_demo.py`: checkout-safe end-to-end LeRobot policy-plus-score planning
  compatibility wrapper with an injected deterministic policy.
- `scripts/generate_provider_docs.py`: provider catalog documentation generator and drift check.
- `scripts/scaffold_provider.py`: safe scaffold generator for new provider adapter files,
  fixture placeholders, tests, and docs stubs.
- `scripts/smoke_leworldmodel.py`: compatibility wrapper for
  `uv run --python 3.13 --with "stable-worldmodel[train] @ git+https://github.com/galilai-group/stable-worldmodel.git" --with "datasets>=2.21" worldforge-smoke-leworldmodel`.
- `scripts/smoke_gr00t_policy.py`: optional live GR00T PolicyClient smoke for host environments
  with Isaac-GR00T or a reachable policy server.
- `scripts/smoke_lerobot_policy.py`: optional live LeRobot `PreTrainedPolicy` smoke for host
  environments with LeRobot and robot-specific dependencies.

## Tech Stack

- Python `>=3.13,<3.14`, with CI workflows standardized on Python 3.13.
- Packaging/build: `hatchling`, `uv`, `uv.lock`.
- Runtime dependency: `httpx`.
- Optional TheWorldHarness runtime: `textual`, supplied only by the `harness` extra.
- Optional LeWorldModel runtime: `stable-worldmodel[train]` and `torch`, supplied by the host
  environment only when using `leworldmodel`.
- Optional GR00T runtime: `gr00t.policy.server_client.PolicyClient`, CUDA/TensorRT/checkpoints,
  and robot-specific dependencies supplied by the host environment only when using `gr00t`.
- Optional LeRobot runtime: `lerobot.policies.pretrained.PreTrainedPolicy`, torch/checkpoints, and
  robot-specific dependencies supplied by the host environment only when using `lerobot`.
- Development tools: `pytest`, `pytest-cov`, `ruff`, `pip-audit` in CI.
- License: MIT.

## Commands

Run these from the repository root:

```bash
uv sync --group dev
uv lock --check
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run python scripts/generate_provider_docs.py --check
uv run mkdocs build --strict
uv run pytest
uv run --extra harness pytest --cov=src/worldforge --cov-report=term-missing --cov-fail-under=90
bash scripts/test_package.sh
```

Discover and run examples:

```bash
uv run worldforge examples
uv run worldforge world create lab --provider mock
uv run worldforge world add-object <world-id> cube --x 0 --y 0.5 --z 0 --object-id cube-1
uv run worldforge world predict <world-id> --object-id cube-1 --x 0.4 --y 0.5 --z 0
uv run worldforge world list
uv run worldforge world objects <world-id>
uv run worldforge world history <world-id>
uv run worldforge world export <world-id> --output world.json
uv run worldforge world delete <world-id>
uv run worldforge provider docs
uv run --extra harness worldforge-harness
uv run worldforge-demo-leworldmodel
uv run worldforge-demo-lerobot
scripts/robotics-showcase
scripts/lewm-lerobot-real --help
uv run worldforge benchmark --provider mock --operation generate --budget-file examples/benchmark-budget.json
uv run worldforge benchmark --provider mock --operation embed --input-file examples/benchmark-inputs.json
```

Generate a provider scaffold:

```bash
uv run python scripts/scaffold_provider.py "Acme WM" \
  --taxonomy "JEPA latent predictive world model" \
  --planned-capability score
```

Local security audit:

```bash
tmp_req="$(mktemp requirements-audit.XXXXXX)"
uv export --frozen --all-groups --no-emit-project --no-hashes -o "$tmp_req" >/dev/null
uvx --from pip-audit pip-audit -r "$tmp_req" --no-deps --disable-pip --progress-spinner off
rm -f "$tmp_req"
```

## Documentation Map

- `README.md`: front-door identity, quickstart, provider matrix, common commands, operating
  boundaries, status, and roadmap.
- `docs/src/architecture.md`: system map, module responsibilities, planning pipelines,
  operational ownership, data contracts, failure boundaries, observability, persistence, and design
  rationale.
- `docs/src/playbooks.md`: checkout validation, provider capability selection, adapter promotion,
  provider diagnostics, local persistence recovery, remote artifacts, optional runtime smokes,
  benchmarks, incident triage, and release gates.
- `docs/src/operations.md`: configuration, operational modes, persistence, observability, failure
  modes, recovery, release checklist, and provider hardening criteria.
- `docs/src/provider-authoring-guide.md`: provider taxonomy, capability, validation,
  observability, testing, documentation, and review checklist.
- `docs/src/api/python.md`: public Python entry points, capability workflows, examples, and
  exception families.
- `docs/src/providers/`: generated provider catalog plus provider-specific config, contracts,
  limits, failure modes, and validation notes.
- `docs/src/assets/`: images used by the MkDocs Material site and README showcase.
- `mkdocs.yml`: GitHub Pages navigation, theme, and strict docs-build configuration.
- `CONTRIBUTING.md` and `docs/src/contributing.md`: contributor setup, validation gates,
  repository map, provider rules, and documentation routing.

## Conventions

- Public inputs fail explicitly with `WorldForgeError`; malformed persisted/provider state fails
  with `WorldStateError`; provider/runtime integration failures fail with `ProviderError`.
- Provider capabilities must only advertise operations that are implemented end to end.
  `ProviderCapabilities()` intentionally advertises no operations by default; opt into each
  supported capability explicitly. Valid capability names are `predict`, `generate`, `reason`,
  `embed`, `plan`, `transfer`, `score`, and `policy`; reject unknown names instead of treating
  them as unsupported.
- `leworldmodel` exposes `score`, not `predict`, `generate`, or `reason`; do not fake those
  capabilities around a cost model.
- `gr00t` exposes `policy`, not `predict`, `score`, or `generate`; do not call an embodied policy
  a predictive world model.
- `lerobot` exposes `policy`, not `predict`, `score`, or `generate`; keep embodiment-specific
  action translation host-owned.
- TheWorldHarness must keep Textual optional. Do not import Textual from `worldforge.__init__`,
  `worldforge.cli`, or non-TUI harness modules.
- Remote create/mutation requests are single-attempt by default; health, polling, and downloads
  use retry/backoff policy.
- Provider events are log-facing records. Keep `target`, `message`, and `metadata` sanitized so
  bearer tokens, API keys, signed URL query strings, and secret-like metadata never reach event
  sinks.
- Keep public API models typed and serializable. Validate boundary values before persistence or
  outbound network I/O.
- Add regression tests for every bug fix and every documented failure mode.
- Put remote provider payload fixtures under `tests/fixtures/providers/` and assert both parser
  errors and public provider errors.
- Update README, docs, changelog, playbooks, and this file when public behavior changes.
- Keep operator docs concrete: every new runtime, provider, persistence, or release workflow
  should state the command to run, the expected success signal, and the first triage step.
- Keep `mkdocs.yml` navigation synchronized with `docs/src/SUMMARY.md` when adding or removing
  public docs pages.

## Critical Constraints

- Do not replace scaffold providers with claims of real JEPA/Genie integration; they are
  credential-gated mock-backed adapters until real provider behavior is implemented.
- Do not export or auto-register `JEPAWMSProvider` until provider-specific limits are validated
  against real upstream weights. The torch-hub path is direct-construction only and keeps
  PyTorch plus JEPA-WMS dependencies host-owned.
- Do not add `stable_worldmodel`, `torch`, checkpoint archives, or downloaded datasets to the base
  dependency set or repository. Keep LeWorldModel optional and host-owned.
- Do not add Isaac GR00T, CUDA, TensorRT, robot checkpoints, or robot controller dependencies to
  the base dependency set. Keep GR00T host-owned and require explicit action translators.
- Do not add LeRobot, torch, robot checkpoints, simulation packages, or robot controller
  dependencies to the base dependency set. Keep LeRobot host-owned and require explicit action
  translators.
- Do not auto-register optional providers unless their required environment variables are present.
- Do not add hardcoded credentials, test secrets, or environment-specific endpoints.
- Do not weaken coverage gates or remove package validation from CI.
- Do not silently coerce invalid world state. A loud failure is preferable to persisted
  incoherence.

## Gotchas

- `WorldForge.doctor()` includes known unregistered providers by default so missing remote
  configuration appears as diagnostics.
- `ProviderMetricsSink.request_count` counts emitted provider events, not necessarily logical
  user-level operations; retry events increment both `request_count` and `retry_count`.
- World persistence is local JSON under `.worldforge/worlds` by default and is not a concurrent
  multi-writer store. Use `worldforge world ...` for CLI create/list/show/history/object
  mutation/predict/export/import/fork flows, and keep service-grade durability host-owned.
- World IDs are file stems for local JSON persistence. Reject path separators, traversal-shaped
  values, and other non-file-safe IDs before loading, importing, or saving world state.
- Persisted history is part of the state contract: history entries must have non-negative steps,
  non-empty summaries, valid snapshot states, valid serialized `Action` payloads when present, and
  no entry step greater than the current world step. Scene object add/update/remove mutations
  should append typed history entries without advancing provider time.
- Position patches must keep scene-object bounding boxes translated with their poses.
- Persistence is host-owned beyond local JSON import/export; do not add a lock file, SQLite store,
  or service adapter without an explicit design.
- Built-in evaluation suites are deterministic contract harnesses, not claims of physical or
  media-quality fidelity.
- LeWorldModel expects preprocessed pixel/action/goal tensors or rectangular nested numeric
  arrays shaped for the configured checkpoint. WorldForge validates the adapter boundary but does
  not infer task-specific image transforms.
- Use `uv run worldforge-demo-leworldmodel` when you need a working LeWorldModel story in a clean
  checkout. It deliberately injects a deterministic cost runtime instead of requiring optional
  `stable_worldmodel` or `torch` dependencies, so it proves the WorldForge adapter/planner path
  rather than real LeWorldModel neural inference.
- GR00T returns embodiment-specific raw action arrays. WorldForge preserves those raw actions but
  requires a host-supplied `action_translator` before it can return executable `Action` objects.
- LeRobot returns embodiment-specific raw policy actions. WorldForge preserves those raw actions
  but requires a host-supplied `action_translator` before it can return executable `Action`
  objects.
- Policy+score planning uses `policy_provider="gr00t"` or `policy_provider="lerobot"` plus
  `score_provider="leworldmodel"` or another score provider; score tensors remain
  host-preprocessed and provider-native.
- `scripts/robotics-showcase` is the prominent PushT real robotics entrypoint. It installs the
  optional host-owned runtime packages for the process, uses packaged PushT hooks, and filters
  common macOS native-library warning noise. Set `WORLDFORGE_SHOW_RUNTIME_WARNINGS=1` to see raw
  third-party stderr.
- `lewm-lerobot-real` is an optional real policy-plus-score smoke. It requires a task-aligned
  LeRobot policy, observation builder, LeWorldModel score tensors, and candidate bridge. Do not
  pad, project, or otherwise reinterpret mismatched action spaces inside WorldForge.
- `worldforge-smoke-leworldmodel` is an optional real-checkpoint smoke. Run it through
  `uv run --python 3.13 --with "stable-worldmodel[train] @ git+https://github.com/galilai-group/stable-worldmodel.git" --with "datasets>=2.21" ...`;
  do not add those dependencies to WorldForge's base package. The upstream default storage root is
  `~/.stable-wm`; object checkpoints must already be extracted there or supplied through
  `--cache-dir`.
- `worldforge-build-leworldmodel-checkpoint` is an optional host-owned object-checkpoint builder
  for Hugging Face LeWM `config.json` and `weights.pt` assets. Run it with the same upstream
  LeWorldModel runtime plus `huggingface_hub` and `matplotlib` (transitive import requirement of
  `stable_pretraining`); do not add those dependencies to WorldForge's base package or commit
  downloaded assets/checkpoints.
- `scripts/smoke_gr00t_policy.py` is an optional live PolicyClient smoke. It can start
  `gr00t/eval/run_gr00t_server.py` from a host-owned Isaac-GR00T checkout, but it still requires
  the host to provide real observations and an embodiment-specific action translator.
- Starting the upstream GR00T server requires a compatible NVIDIA/Linux runtime for CUDA and
  TensorRT dependencies. On unsupported hosts, connect to an already running remote GR00T policy
  server.
- `scripts/smoke_lerobot_policy.py` is an optional live LeRobot policy smoke. It requires the host
  to provide real observations and an embodiment-specific action translator.
- If GitHub Actions checks fail before execution because repository/account billing or spending
  limits prevent jobs from starting, treat local `uv`/package validation as the available gate.
- `JEPA_WMS_MODEL_PATH`, `JEPA_WMS_MODEL_NAME`, and `JEPA_WMS_DEVICE` are documented by the
  `jepa-wms` candidate only. They do not make `JEPAWMSProvider` available through `WorldForge`;
  direct tests must inject `runtime=` or use `JEPAWMSProvider.from_torch_hub(...)`.
- `RUNWAYML_API_SECRET` is preferred, but `RUNWAY_API_SECRET` remains supported as a legacy alias.
- `.env.example` is tracked via an explicit `!.env.example` rule in `.gitignore` (the general
  `.env.*` pattern would otherwise exclude it). Keep both the template and the exception in sync
  when adding new provider environment variables.
- `make lint` and `make format` run against `src tests examples scripts` to match CI and the
  commands documented in `README.md`. Do not drop `scripts` from either target.
- `make docs-check` checks generated provider docs and builds the MkDocs Material site with
  `--strict`. A warning in the published docs build is a release blocker.
- `worldforge benchmark --budget-file <path>` evaluates direct provider benchmark results against
  JSON thresholds and exits non-zero on violations. Keep benchmark budgets tied to preserved run
  artifacts when using them for release or paper claims.
- `worldforge benchmark --input-file <path>` loads deterministic benchmark inputs from JSON.
  Relative transfer clip paths resolve next to the input file; inline `frames_base64` is available
  when the clip bytes must be preserved inside the fixture.

## Technical Scope

WorldForge is scoped as a typed Python framework layer for local physical-AI world-model work:
truthful provider capabilities, score/policy planning composition, deterministic adapter
contracts, host-owned optional runtimes, strict validation, and clear operational boundaries.
Keep the front face serious and precise. Do not present scaffold adapters as real integrations,
do not imply physical fidelity from deterministic evaluation suites, and do not move host-owned
runtime, persistence, credential, or robot-controller responsibilities into the base package
without an explicit design.
