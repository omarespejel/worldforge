# Changelog

All notable user-visible changes to this project will be documented in this file.

## Unreleased

### Added

- Explicit `WorldForgeError` and `WorldStateError` surfaces for invalid input and malformed persisted state.
- Regression coverage for invalid runtime inputs, malformed imports, missing local provider assets, and invalid remote payloads.
- `.env.example` and `AGENTS.md` so contributors and coding agents share the same live project contract.
- Typed `RetryPolicy`, `RequestOperationPolicy`, and `ProviderRequestPolicy` models exported through the public API.

### Changed

- Public workflows now reject invalid values such as `steps <= 0`, `max_steps <= 0`, and missing scene object ids instead of silently coercing them.
- Remote asset handling now fails fast when a local file path is missing instead of treating the path string as a remote URI.
- README and provider documentation now reflect the real provider status split: `mock` stable, `cosmos` and `runway` beta, `jepa` and `genie` scaffold.
- `cosmos` and `runway` now share one typed timeout and retry contract, with retried read operations and single-attempt mutation requests by default.

### Fixed

- Cosmos response decoding now validates the returned base64 payload instead of accepting malformed video data.
- Runway task polling now verifies that completed tasks include a non-empty output list before download.
- Transient retryable failures on remote health, polling, and download operations now recover automatically under the configured request policy.
