# Benchmarking

WorldForge includes a capability-aware benchmark harness for registered providers.

## Python

```python
from worldforge import ProviderBenchmarkHarness

harness = ProviderBenchmarkHarness(forge=forge)
report = harness.run(
    ["mock"],
    operations=["predict", "generate", "transfer"],
    iterations=5,
    concurrency=2,
)

print(report.to_markdown())
```

## CLI

```bash
uv run worldforge benchmark --provider mock --iterations 5
uv run worldforge benchmark --provider mock --operation generate --format json
```

## Report contents

- per-provider, per-operation success and error counts
- retry totals derived from emitted `ProviderEvent` records
- total wall-clock time and throughput
- average, min/max, p50, and p95 latency
- serialized provider-operation event aggregates for deeper inspection

The benchmark harness is synthetic. It measures operation latency, retries, and throughput for the current provider adapter path; it does not score media quality or replace a distributed load-test setup.
