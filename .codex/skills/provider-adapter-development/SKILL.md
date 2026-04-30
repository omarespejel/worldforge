---
name: provider-adapter-development
description: "Use for any WorldForge provider work: adding adapters, promoting scaffolds, changing capability declarations, debugging provider failures, updating catalog docs, or touching Cosmos, Runway, LeWorldModel, GR00T, LeRobot, JEPA, Genie, or JEPA-WMS."
prerequisites: "uv, pytest, ruff"
---

# Provider Adapter Development

<purpose>
Keep provider adapters truthful, typed, tested, and honest about runtime ownership.
</purpose>

<context>
WorldForge providers expose strict capability names: `predict`, `generate`, `reason`, `embed`, `plan`, `transfer`, `score`, `policy`. `ProviderCapabilities()` is fail-closed by default. Catalog auto-registration lives in `src/worldforge/providers/catalog.py`; direct protocol registration lives on `WorldForge.register_*`. Optional runtimes, checkpoints, credentials, datasets, CUDA, and robot controllers stay host-owned.
</context>

<procedure>
1. Read the closest existing adapter plus `src/worldforge/providers/base.py`, `src/worldforge/providers/catalog.py`, and `docs/src/providers/README.md`.
2. Classify the real callable surface before editing. If the runtime only scores candidates, expose `score`; if it selects embodied actions, expose `policy`.
3. For new scaffolds, prefer `uv run python scripts/scaffold_provider.py ...`; keep capabilities unadvertised until methods return validated WorldForge models.
4. Validate public inputs before network or runtime calls. Return `PredictionPayload`, `VideoClip`, `ReasoningResult`, `EmbeddingResult`, `ActionScoreResult`, or `ActionPolicyResult` as appropriate.
5. Add fixtures under `tests/fixtures/providers/` for success, malformed upstream payload, and public provider error paths.
6. Add contract coverage with `worldforge.testing.assert_provider_contract()` when a capability is advertised.
7. Update `.env.example`, provider docs, README/generated catalog, `AGENTS.md`, and `CLAUDE.md` only when public behavior or environment variables change.
8. Run focused provider tests, then `uv run python scripts/generate_provider_docs.py --check`, ruff, pytest, and coverage when public surfaces changed.
</procedure>

<patterns>
<do>
- Use `ProviderProfileSpec` metadata to state implementation status, credential requirements, supported modalities, artifact types, and notes.
- Use `ProviderRequestPolicy` for remote timeouts/retries; keep create/mutation requests single-attempt unless idempotency is proven.
- Emit sanitized `ProviderEvent` records only; signed URL query strings, bearer tokens, and secret-like values must not reach sinks.
</do>
<dont>
- Do not label embodied policies as world models; GR00T and LeRobot expose `policy`.
- Do not expose `leworldmodel` as `predict`, `generate`, or `reason`; it exposes `score`.
- Do not export or auto-register `JEPAWMSProvider` without explicit validated-runtime design approval.
- Do not add optional runtime packages to `project.dependencies`.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| Provider appears in docs with wrong surface | `ProviderCapabilities` declaration drifted | Fix adapter capabilities, run provider docs generator, update tests |
| Optional provider missing from `doctor` | Required env var absent | Confirm variable name from `.env.example`; do not read `.env` |
| Contract helper fails on JSON | Metadata/raw payload not JSON-native | Validate at construction and convert tuples/objects before return |
| Remote test leaks URL/query | Event target/message metadata not sanitized | Add regression in `tests/test_observability.py` or provider test |
</troubleshooting>

<references>
- `references/capability-matrix.md`: capability surfaces and provider ownership.
- `src/worldforge/providers/base.py`: provider base behavior and validation helpers.
- `src/worldforge/providers/catalog.py`: catalog registration and docs rendering.
- `src/worldforge/testing/providers.py`: adapter contract assertions.
- `docs/src/provider-authoring-guide.md`: maintainer-facing provider guide.
</references>
