# Changelog

All notable user-visible changes to this project will be documented in this file.

## Unreleased

### Added

- Typed `StructuredGoal` planning inputs for `object_at`, `object_near`, `spawn_object`, and `swap_objects` workflows, plus targeted `Action.move_to(..., object_id=...)` execution support in the mock provider.
- Built-in `generation` and `transfer` evaluation suites so video-capable providers receive scored scenario coverage alongside physics, planning, and reasoning.
- `ProviderBenchmarkHarness`, benchmark report exporters, and `worldforge benchmark` for provider latency, retry, and throughput measurements.
- Explicit `WorldForgeError` and `WorldStateError` surfaces for invalid input and malformed persisted state.
- Regression coverage for invalid runtime inputs, malformed imports, missing local provider assets, and invalid remote payloads.
- `.env.example` and `AGENTS.md` so contributors and coding agents share the same live project contract.
- Typed `RetryPolicy`, `RequestOperationPolicy`, and `ProviderRequestPolicy` models exported through the public API.
- Typed `ProviderEvent` records and `event_handler=` plumbing on `WorldForge` and provider constructors for host-side observability.
- Capability-aware built-in evaluation suites for `physics`, `planning`, and `reasoning`, with scenario-level pass/fail results and provider summaries.

### Changed

- Public workflows now reject invalid values such as `steps <= 0`, `max_steps <= 0`, and missing scene object ids instead of silently coercing them.
- Remote asset handling now fails fast when a local file path is missing instead of treating the path string as a remote URI.
- README and provider documentation now reflect the real provider status split: `mock` stable, `cosmos` and `runway` beta, `jepa` and `genie` scaffold.
- `cosmos` and `runway` now share one typed timeout and retry contract, with retried read operations and single-attempt mutation requests by default.
- Builtin providers, manually registered providers, and scaffold adapters now share one provider-event contract, so host applications can observe local and remote execution through the same callback surface.
- Evaluation reports now include suite ids, provider pass/fail summaries, scenario metrics, and CLI export formats for Markdown, JSON, and CSV.
- The built-in planning suite now exercises relational goal execution for neighbor placement and object swaps in addition to relocation and spawn flows.
- `worldforge eval` now accepts repeated `--provider` arguments so one report can compare multiple providers in a single run.
- `list_eval_suites()` now returns `generation` and `transfer` in addition to the existing built-in suites.
- Unknown evaluation suite names and missing suite capabilities now fail with explicit `WorldForgeError` messages.

### Fixed

- Cosmos response decoding now validates the returned base64 payload instead of accepting malformed video data.
- Runway task polling now verifies that completed tasks include a non-empty output list before download.
- Transient retryable failures on remote health, polling, and download operations now recover automatically under the configured request policy.
