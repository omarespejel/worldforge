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

## Report contents

- per-provider, per-operation success and error counts
- retry totals derived from emitted `ProviderEvent` records
- total wall-clock time and throughput
- average, min/max, p50, and p95 latency
- serialized provider-operation event aggregates for deeper inspection

The benchmark harness is synthetic. It measures operation latency, retries, and throughput for the selected provider adapter path; it does not score media quality or replace a distributed load-test setup.
