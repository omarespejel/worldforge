---
name: provider-adapter-development
description: Use when adding, promoting, reviewing, or debugging a WorldForge provider adapter, provider capability, provider catalog entry, provider docs page, HTTP parser, optional runtime wrapper, or provider fixture. Also use when a task mentions Cosmos, Runway, LeWorldModel, GR00T, LeRobot, JEPA, Genie, jepa-wms, capability truthfulness, or adapter contract tests.
prerequisites: uv, pytest, ruff; live provider credentials are optional and host-owned.
---

# Provider Adapter Development

<purpose>
Keep provider work honest: classify the upstream contract, expose only implemented capabilities, validate every boundary, and prove behavior with fixtures plus contract tests.
</purpose>

<context>
- Base interfaces: `src/worldforge/providers/base.py`.
- Catalog and generated docs: `src/worldforge/providers/catalog.py`, `scripts/generate_provider_docs.py`.
- Public capability model: `ProviderCapabilities` in `src/worldforge/models.py`.
- Contract helper: `worldforge.testing.assert_provider_contract()` in `src/worldforge/testing/providers.py`.
- Provider docs: `docs/src/providers/`; parser fixtures: `tests/fixtures/providers/`.
- `JEPAWMSProvider` is direct-construction only; do not export or auto-register it.
</context>

<procedure>
1. Classify the provider by actual callable behavior: `predict`, `generate`, `transfer`, `reason`, `embed`, `score`, or `policy`.
2. If starting new, run `uv run python scripts/scaffold_provider.py "<Provider Name>" --taxonomy "<category>" --planned-capability <capability>` with `--remote --env-var <ENV>` when applicable.
3. Keep scaffold capabilities disabled until the implementation returns validated WorldForge models end to end.
4. Validate caller inputs before outbound HTTP/runtime calls where possible.
5. Parse upstream responses through explicit helpers; malformed outputs must raise `ProviderError`.
6. Add fixtures under `tests/fixtures/providers/` for success, missing fields, malformed shapes, provider errors, and partial/expired artifacts where relevant.
7. Add tests that call the adapter method and `assert_provider_contract()` for advertised capabilities.
8. Register in `PROVIDER_CATALOG` only when auto-detection is safe and env-gated as needed.
9. Update `docs/src/providers/<provider>.md`; run `uv run python scripts/generate_provider_docs.py`.
10. Validate with focused provider tests, docs check, ruff, coverage, and package contract when public API changed.
</procedure>

<patterns>
<do>
- Use `ProviderCapabilities(score=True)` only when `score_actions(...)` returns `ActionScoreResult`.
- Use `ProviderCapabilities(policy=True)` only when `select_actions(...)` returns executable `Action` objects plus raw action metadata.
- Use `ProviderRequestPolicy.remote_defaults()` and shared HTTP helpers for remote adapters unless the provider needs stricter documented behavior.
- Keep `health()` cheap: credentials, optional dependency availability, and endpoint readiness, not expensive inference.
- Keep optional dependencies imported lazily inside runtime paths.
</do>
<dont>
- Do not call a score model `predict`; expose `score`.
- Do not call GR00T or LeRobot a world model; expose `policy`.
- Do not retry create/mutation requests unless idempotency is documented.
- Do not log bearer tokens, signed URLs, checkpoint paths containing secrets, or raw credentials.
- Do not make tests depend on live credentials; use fake transports or injected runtimes.
</dont>
</patterns>

<example>
```python
provider = AcmeProvider(transport=fake_transport)
assert provider.capabilities.generate
clip = provider.generate("move the cube", 1.0)
assert clip.metadata["provider"] == "acme"
report = assert_provider_contract(provider)
assert "generate" in report.exercised_operations
```
</example>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| provider appears in docs with wrong capability | catalog/profile drift | fix `ProviderCapabilities`, rerun docs generator, add catalog test |
| unsupported method returns generic error | adapter overrides base method prematurely | remove override or raise `ProviderError` with provider context |
| tests reach live API | fake transport/injected runtime missing | replace network path with fixture-backed fake |
| optional dependency import fails on base import | eager import leaked | move import into configured runtime path |
</troubleshooting>

<references>
- `docs/src/provider-authoring-guide.md`: full provider checklist.
- `docs/src/playbooks.md`: provider diagnostics and release gates.
- `tests/test_provider_contracts.py`: in-repo contract expectations.
</references>
