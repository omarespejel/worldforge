---
name: evaluation-benchmarking
description: "Use for WorldForge evaluation suites, benchmark harness changes, benchmark input fixtures, budget gates, report rendering, metrics semantics, and any claims based on benchmark or evaluation output."
prerequisites: "uv, pytest"
---

# Evaluation And Benchmarking

<purpose>
Keep evaluation and benchmark outputs deterministic, coherent, and claim-bounded.
</purpose>

<context>
Built-in evaluation suites are deterministic contract harnesses, not evidence of physical fidelity or media quality. Benchmarks measure adapter-path latency, retries, throughput, and errors for direct provider operations: `predict`, `reason`, `generate`, `transfer`, `embed`, `score`, and `policy`. `plan()` remains a WorldForge facade workflow.
</context>

<procedure>
1. Read `src/worldforge/evaluation/suites.py` for eval changes or `src/worldforge/benchmark.py` for benchmark changes.
2. Preserve claim-boundary metadata and metric semantics in rendered JSON/Markdown/CSV reports.
3. Validate benchmark inputs eagerly through `BenchmarkInputs` / `load_benchmark_inputs(...)`; reject unknown keys and non-finite metrics.
4. Resolve relative transfer clip paths next to the input fixture; use `frames_base64` when fixture bytes must be preserved inline.
5. If operations change, update CLI help snapshots, harness diagnostics text, README, docs benchmarking/API/playbooks, and changelog together.
6. Test direct operation behavior and budget violations; run docs check when report text or docs change.
</procedure>

<patterns>
<do>
- Benchmark `score` and `policy` directly with `examples/benchmark-inputs.json`.
- Preserve `BenchmarkBudget` non-zero exit behavior on violations.
- Sum `request_count` fields in operation metrics when reporting request volume; event rows can aggregate attempts.
</do>
<dont>
- Do not route benchmark `score` or `policy` through `plan()`.
- Do not turn deterministic suite scores into physical-fidelity claims.
- Do not force non-JSON provider-native tensors through JSON; preview type and shape when needed.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| `--input-file` rejects fixture | Unknown key or non-JSON payload | Match allowed `BenchmarkInputs` keys and JSON-native values |
| Budget gate exits non-zero | Threshold violation | Preserve artifact, inspect report, adjust code or documented budget with evidence |
| CLI help snapshot fails | Operation/help text changed | Update `tests/test_cli_help_snapshots.py` intentionally |
</troubleshooting>

<references>
- `src/worldforge/benchmark.py`: benchmark inputs, budgets, harness, renderers.
- `src/worldforge/evaluation/suites.py`: deterministic suite definitions.
- `examples/benchmark-inputs.json`: reproducible input fixture.
- `examples/benchmark-budget.json`: budget-gate fixture.
- `docs/src/benchmarking.md`: public benchmark contract.
</references>
