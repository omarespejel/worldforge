---
name: evaluation-benchmarking
description: Use when working on WorldForge evaluation suites, benchmark harnesses, planning reports, latency/throughput measurements, exported JSON/CSV/Markdown reports, provider comparison, or any claim based on evaluation or benchmark output.
prerequisites: uv, pytest; live providers are optional and must be host-owned.
---

# Evaluation And Benchmarking

<purpose>
Keep evaluation and benchmark work deterministic, capability-aware, and scoped to adapter behavior rather than physical or media-quality claims.
</purpose>

<context>
- Built-in suites live in `src/worldforge/evaluation/`.
- Benchmark harness lives in `src/worldforge/benchmark.py`.
- CLI surfaces: `worldforge eval` and `worldforge benchmark`.
- Evaluation suites are deterministic contract harnesses, not real-world fidelity evidence.
- Benchmark metrics include provider operation, latency, retry count, throughput, and report format.
</context>

<procedure>
1. Identify the required provider capability before choosing a suite or benchmark operation.
2. Use `mock` or injected deterministic runtimes for checkout-safe tests.
3. For capability-mismatch behavior, assert explicit `WorldForgeError` instead of silent skips unless an existing suite intentionally skips.
4. Preserve generated benchmark/evaluation artifacts when using them for docs, release notes, papers, or claims.
5. Validate CLI output formats when changing renderers: JSON, Markdown, and CSV where supported.
6. Run focused tests, then coverage gate for public behavior changes.
</procedure>

<commands>
```bash
uv run worldforge eval --suite planning --provider mock --format markdown
uv run worldforge eval --suite generation --provider mock --format json
uv run worldforge benchmark --provider mock --iterations 5 --format markdown
uv run worldforge benchmark --provider mock --iterations 5 --format json
uv run pytest tests/test_evaluation_and_planning.py tests/test_benchmark.py
```
</commands>

<patterns>
<do>
- Gate each suite by the exact capability it requires.
- Keep test data deterministic and small.
- Include provider, operation, status, latency, and retry evidence in reports.
- Label checkout-safe demos as injected deterministic runtime checks.
</do>
<dont>
- Do not claim physical fidelity, media quality, or real-world safety from built-in deterministic suites.
- Do not rewrite claims around one unpreserved benchmark run.
- Do not make benchmark tests depend on live credentials or network timing.
- Do not hide provider capability errors by catching broad exceptions.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| suite fails on unsupported provider | wrong capability selected | run `uv run worldforge doctor --capability <name>` and pick matching provider |
| benchmark retry counts look too high | `ProviderMetricsSink.request_count` counts emitted events/attempts | document attempt semantics; do not reinterpret as user operations |
| JSON renderer breaks tests | non-serializable model field | convert through `to_dict()` and add renderer regression test |
</troubleshooting>

<references>
- `docs/src/evaluation.md`: user-facing suite semantics.
- `docs/src/benchmarking.md`: benchmark usage and reports.
- `tests/test_evaluation_and_planning.py`: suite behavior.
- `tests/test_benchmark.py`: benchmark behavior.
</references>
