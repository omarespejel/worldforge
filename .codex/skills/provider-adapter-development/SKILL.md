---
name: provider-adapter-development
description: Use whenever the task touches a WorldForge provider adapter, provider capability flag, the provider catalog, generated provider docs, an HTTP parser, an optional runtime wrapper, or a provider fixture. Trigger on names like Cosmos, Runway, LeWorldModel, GR00T, LeRobot, JEPA, Genie, jepa-wms, or on phrases like "add a provider", "promote scaffold", "fix capability", "provider returns wrong shape", "doctor says missing", "provider docs drift", "scaffold adapter", "contract test failed". Also trigger when the user asks whether a provider can predict / generate / score / plan / act, since that question is really about capability truthfulness.
---

# Provider Adapter Development

The project's main correctness risk is providers lying about what they can do — `doctor`, planning, evaluation, and benchmarking all read `ProviderCapabilities` and trust it. Every change in this area must keep that flag honest end to end.

## Fast start

Most provider tasks are one of three shapes. Pick the closest, then read the matching section below.

```bash
# 1. Add a brand-new provider scaffold (remote HTTP)
uv run python scripts/scaffold_provider.py "Acme World" \
  --taxonomy generation --planned-capability generate \
  --remote --env-var ACME_API_KEY

# 2. Re-check honesty after editing capabilities or methods
uv run pytest tests/test_provider_contracts.py -k <provider>
uv run python scripts/generate_provider_docs.py --check

# 3. Diagnose a "missing"/"unregistered" complaint from doctor
uv run worldforge doctor --capability <name>
```

Generated capability matrix and method-to-capability mapping live in
`references/capability-matrix.md`. Read it before flipping any capability flag.

## Why this skill exists

`ProviderCapabilities` is the single source of truth used by:

- `worldforge doctor` to report what the host can run.
- the planner to decide whether `plan_actions` is callable.
- evaluation suites to skip / fail early on unsupported capabilities.
- benchmark to label rows.
- the README provider catalog block (regenerated, not hand-edited).

A `True` flag without a working method underneath is a silent contract break that surfaces as runtime `AttributeError`s or, worse, as wrong evaluation conclusions in published claims. Keeping flags strict and fail-closed is therefore non-negotiable; everything else in this skill is in service of that.

## The procedure (annotated)

1. **Classify by what the upstream actually returns**, not by marketing label. A score model is `score`, not `predict`. A robot policy is `policy`, not `predict`. A media generator is `generate` (and `transfer` only if it edits an existing clip).
2. **Scaffold first** if the provider is new. The scaffold gives you a `health()`, fail-closed `ProviderCapabilities()`, and the right place to put the env-gated registration. Leave every capability flag `False` until the matching method returns a validated WorldForge model end to end.
3. **Validate inputs at the boundary** before any outbound HTTP / runtime call — invalid public input must raise `WorldForgeError`, never reach the wire.
4. **Parse upstream responses through explicit helpers**, not inline `dict.get`. Malformed shapes raise `ProviderError` with provider context (provider name, operation, retry count if relevant).
5. **Fixture the failure modes** under `tests/fixtures/providers/<provider>/`: success, missing required field, malformed shape, provider error envelope, and (for media providers) partial / expired artifact URL. Failure paths are where contract drift hides.
6. **Assert the contract**: `worldforge.testing.assert_provider_contract(provider)` exercises every advertised capability with deterministic input. If it skips a capability you set `True`, the flag is wrong.
7. **Register in the catalog** only when auto-detection is safe. Env-gate anything that needs credentials so a fresh checkout doesn't spuriously register.
8. **Regenerate provider docs**: `uv run python scripts/generate_provider_docs.py`. The README catalog block and `docs/src/providers/README.md` are generated; never hand-edit them.
9. **Validate** with focused tests, then docs check, then ruff, then coverage gate, and `bash scripts/test_package.sh` if any public API moved.

## Examples

**Adding `generate` to a remote provider:**

```python
provider = AcmeProvider(transport=fake_transport)  # injected, no live network
assert provider.capabilities.generate
clip = provider.generate("move the cube", duration=1.0)
assert clip.metadata["provider"] == "acme"

report = assert_provider_contract(provider)
assert "generate" in report.exercised_operations
```

**Promoting a scaffold from "planned" to real:** the scaffold has
`ProviderCapabilities()` (all `False`). After implementing `generate_clip`
and adding fixtures + contract test, flip exactly the one flag that's now
backed by code. Do not flip the rest.

**A score model that "feels like" a world model:** still `score=True`,
not `predict=True`. The `score` capability returns `ActionScoreResult`;
`predict` returns a future world state. Different shape, different consumers.

## Activation cues

Trigger on:
- "add a provider", "scaffold provider", "register provider", "remove `auto_register`"
- "capability filter rejected …", "doctor says capability missing"
- "provider returns wrong shape", "parser fails on …", "fixtures for …"
- mention of any provider name (Cosmos, Runway, LeWorldModel, GR00T, LeRobot, JEPA, Genie, jepa-wms)
- mention of `ProviderCapabilities`, `ProviderRequestPolicy`, `assert_provider_contract`, `PROVIDER_CATALOG`

Do **not** trigger for:
- pure planner / evaluation / benchmark logic that happens to consume a provider — load `evaluation-benchmarking` instead
- world persistence / history / state validation — load `persistence-state`
- live LeWorldModel / GR00T / LeRobot smoke runs — load `optional-runtime-smokes`

## Stop and ask the user

- before exporting `JEPAWMSProvider` or auto-registering it (the project explicitly forbids this without validated-runtime design approval)
- before adding any optional runtime package to base `[project.dependencies]`
- before changing public capability semantics or registration semantics in `providers/base.py` / `providers/catalog.py`

## Patterns

**Do:**
- Use `ProviderCapabilities(score=True)` only when `score_actions(...)` returns `ActionScoreResult`.
- Use `ProviderCapabilities(policy=True)` only when `select_actions(...)` returns executable `Action` objects plus raw action metadata.
- Use `ProviderRequestPolicy.remote_defaults()` and the shared HTTP helpers for remote adapters unless the provider documents stricter behavior.
- Keep `health()` cheap — credential presence, optional-dep availability, endpoint reachability — never paid inference.
- Import optional dependencies lazily, inside the runtime path.

**Don't:**
- Call a score model `predict`, or call GR00T / LeRobot a world model.
- Retry create / mutation requests unless the upstream documents idempotency.
- Log bearer tokens, signed URLs, checkpoint paths containing secrets, or raw credentials — even at debug level.
- Make tests depend on live credentials. Use a fake transport or an injected runtime.

## Troubleshooting

| Symptom | Likely cause | First fix |
| --- | --- | --- |
| Provider appears in README with wrong capability | catalog / `ProviderCapabilities` drift | fix the flag, rerun `generate_provider_docs.py`, add a catalog test |
| Unsupported method returns a generic `AttributeError` | adapter overrides base method prematurely | remove the override, or raise `ProviderError` with provider context |
| Tests reach a live API | fake transport / injected runtime missing | replace the network path with a fixture-backed fake |
| Optional dependency import explodes on base import | eager top-level import leaked | move the import inside the configured runtime path |
| `assert_provider_contract` passes but `doctor` says "missing" | env-gated registration not satisfied | set the env var the catalog inspects, or document the requirement |

## References

- `references/capability-matrix.md` — capability ↔ method ↔ return-type lookup; consult before flipping flags
- `docs/src/provider-authoring-guide.md` — full authoring walkthrough
- `docs/src/playbooks.md` — provider diagnostics and release gates
- `tests/test_provider_contracts.py` — in-repo expectations
- `src/worldforge/providers/base.py` — base interfaces
- `src/worldforge/providers/catalog.py` — registration + auto-detect rules
