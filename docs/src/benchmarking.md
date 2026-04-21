# Benchmarking

WorldForge includes a capability-aware benchmark harness for registered providers. It can measure
direct provider surfaces: `predict`, `reason`, `generate`, `transfer`, `embed`, `score`, and
`policy`. `plan` remains a WorldForge facade workflow, so benchmark score providers and policy
providers directly when you need planning-path latency.

## Python

```python
from worldforge import ProviderBenchmarkHarness

harness = ProviderBenchmarkHarness(forge=forge)
report = harness.run(
    ["mock"],
    operations=["predict", "generate", "transfer", "embed"],
    iterations=5,
    concurrency=2,
)

print(report.to_markdown())
```

Score and policy providers use the same benchmark runner with provider-native inputs supplied by
the host:

```python
from worldforge import BenchmarkInputs, ProviderBenchmarkHarness

inputs = BenchmarkInputs(
    score_info={
        "pixels": [[[[0.0]]]],
        "goal": [[[0.3, 0.5, 0.0]]],
        "action": [[[0.0, 0.5, 0.0]]],
    },
    score_action_candidates=[[[[0.0, 0.5, 0.0]], [[0.3, 0.5, 0.0]]]],
    policy_info={
        "observation": {
            "state": {"cube": [0.0, 0.5, 0.0]},
            "language": "move the cube",
        },
        "mode": "select_action",
    },
)

report = ProviderBenchmarkHarness(forge=forge).run(
    ["leworldmodel", "lerobot"],
    iterations=3,
    inputs=inputs,
)
```

## CLI

```bash
uv run worldforge benchmark --provider mock --iterations 5
uv run worldforge benchmark --provider mock --operation generate --format json
uv run worldforge benchmark --provider mock --operation embed --format markdown
```

Use a budget file when a benchmark run is part of a release gate, regression check, or public
claim. Budget selectors can pin a provider and operation, or omit either field to apply the
threshold to every matching result:

```json
{
  "budgets": [
    {
      "provider": "mock",
      "operation": "generate",
      "min_success_rate": 1.0,
      "max_error_count": 0,
      "max_retry_count": 0,
      "max_average_latency_ms": 250.0,
      "max_p95_latency_ms": 400.0,
      "min_throughput_per_second": 2.0
    }
  ]
}
```

```bash
uv run worldforge benchmark \
  --provider mock \
  --operation generate \
  --iterations 5 \
  --format json \
  --budget-file benchmark-budget.json
```

With `--budget-file`, the command prints both the benchmark report and a gate report. A failing gate
exits non-zero after printing violations such as latency, retry, error-count, success-rate, or
unmatched-budget checks. JSON output contains `benchmark` and `gate` objects; Markdown prints both
reports; CSV prints the gate violation table.

## Report contents

- per-provider, per-operation success and error counts
- retry totals derived from emitted `ProviderEvent` records
- total wall-clock time and throughput
- average, min/max, p50, and p95 latency
- serialized provider-operation event aggregates for deeper inspection
- optional budget-gate results for release or claim-oriented thresholds

The benchmark harness is synthetic. It measures operation latency, retries, and throughput for the selected provider adapter path; it does not score media quality or replace a distributed load-test setup.
