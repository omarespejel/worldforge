# Evaluation

WorldForge currently ships three built-in suites:

- `physics`: deterministic object stability and action-response checks
- `planning`: heuristic plan generation plus execution validation
- `reasoning`: scene-count and scene-identity checks for providers that implement `reason()`

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
uv run worldforge eval --suite physics --provider mock
uv run worldforge eval --suite planning --provider mock --format json
uv run worldforge eval --suite reasoning --provider mock --format csv
```

Repeat `--provider` to compare multiple registered providers in one report.

## Report formats

- Markdown: provider summary table plus scenario-level detail table
- JSON: `suite_id`, `suite`, `provider_summaries`, and scenario `results`
- CSV: one row per provider/scenario pair with serialized metrics payloads

## Capability checks

Each suite declares the provider capabilities it needs. For example:

- `physics` and `planning` require `predict`
- `reasoning` requires `reason`

WorldForge raises `WorldForgeError` when a caller asks a provider to run a suite it cannot satisfy.
