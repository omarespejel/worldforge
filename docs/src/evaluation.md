# Evaluation

WorldForge ships five built-in suites:

- `generation`: prompt-only and image-conditioned video generation checks
- `physics`: deterministic object stability and action-response checks
- `planning`: relocation, neighbor placement, swap, and spawn execution validation over the predict-driven planner
- `reasoning`: scene-count and scene-identity checks for providers that implement `reason()`
- `transfer`: prompt-guided and reference-guided video transfer checks

## Python

```python
from worldforge.evaluation import EvaluationSuite

suite = EvaluationSuite.from_builtin("planning")
report = suite.run_report(["mock"], forge=forge)

print(report.results[0].passed)
print(report.to_json())
```

## CLI

```bash
uv run worldforge eval --suite generation --provider mock
uv run worldforge eval --suite physics --provider mock
uv run worldforge eval --suite planning --provider mock --format json
uv run worldforge eval --suite reasoning --provider mock --format csv
uv run worldforge eval --suite transfer --provider mock
```

Repeat `--provider` to compare multiple registered providers in one report.

The same built-in suites are available from TheWorldHarness. Launch
`uv run --extra harness worldforge-harness --flow eval`, pick a suite and provider, and the TUI
writes the canonical JSON report under `.worldforge/reports/` before opening the Run Inspector.
Capability mismatches remain `WorldForgeError` failures; the TUI surfaces the message instead of
silently skipping the suite.

## Report formats

- Markdown: provider summary table plus scenario-level detail table
- JSON: `suite_id`, `suite`, `provider_summaries`, and scenario `results`
- CSV: one row per provider/scenario pair with serialized metrics payloads

## Capability checks

Each suite declares the provider capabilities it needs. For example:

- `generation` requires `generate`
- `physics` and `planning` require `predict`
- `reasoning` requires `reason`
- `transfer` requires `transfer`

WorldForge raises `WorldForgeError` when a caller asks a provider to run a suite it cannot satisfy.
