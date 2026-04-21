---
name: evaluation-benchmarking
description: Use whenever the task involves a WorldForge evaluation suite, the benchmark harness, planning report output, latency / throughput / retry measurements, JSON / CSV / Markdown report renderers, provider comparisons, or any external claim derived from `worldforge eval` or `worldforge benchmark`. Trigger on phrases like "run an eval", "benchmark this provider", "compare providers", "report says X is faster", "publish numbers", "regression in latency", "the suite is skipping", "renderer is broken", "what does this metric mean". Also trigger when the user is about to put a number from this repo into a slide, paper, README, or PR description.
---

# Evaluation And Benchmarking

The deterministic evaluation suites and the benchmark harness exist to measure **adapter behavior**, not real-world fidelity. Their outputs land in slides, papers, and PR descriptions, so the bar is: every claim that leaves the repo must be traceable to a preserved run with a documented capability and a documented runtime.

## Fast start

```bash
# Eval a capability-gated suite against the always-on mock provider
uv run worldforge eval --suite planning   --provider mock --format markdown
uv run worldforge eval --suite generation --provider mock --format json

# Benchmark a provider for latency / retry / throughput
uv run worldforge benchmark --provider mock --iterations 5 --format markdown
uv run worldforge benchmark --provider mock --iterations 5 --format json

# Validate after changing suites, renderers, or metrics
uv run pytest tests/test_evaluation_and_planning.py tests/test_benchmark.py
```

If the suite reports a capability mismatch, that is the suite working as intended — pick a matching provider via `worldforge doctor --capability <name>` rather than catching the error.

## Why this skill exists

The two systems answer different questions and must not be conflated:

- **Evaluation suites** (`src/worldforge/evaluation/`) are deterministic *contract* harnesses. They prove that a provider's advertised capability is actually callable end-to-end with validated WorldForge models. They do **not** measure physical realism, video quality, or robot success rate — anyone reading the output that way is being misled.
- **Benchmarks** (`src/worldforge/benchmark.py`) measure provider operations: latency, retry counts, throughput, status. Each row is one adapter call against the local process clock.

Two failure modes worth defending against:

1. **Misclaimed scope.** "The benchmark says Cosmos beats Runway on quality" — no, it doesn't, because no built-in suite scores quality. Latency/retry/throughput are the only honest claims.
2. **Unpreserved one-shot runs.** A single benchmark on a noisy network, deleted, then quoted in a PR. If a number is going to be cited, the JSON report it came from must be saved alongside the citation.

`ProviderMetricsSink.request_count` counts emitted attempts (incl. retries), not user operations. Mention this whenever a "retry count looks high" question comes up.

## The procedure

1. **Pick the capability first**, then the provider that advertises it. The suite uses the capability to decide what to call; mismatched capability → explicit `WorldForgeError`, not a silent skip (preserve this behavior).
2. **Use deterministic providers for checkout-safe runs**: `mock`, or an injected deterministic runtime under `worldforge.demos.*`. Live providers are host-owned and may be absent.
3. **Preserve every report** that will be cited externally. Store the exact JSON next to the artifact that quotes it (slide, paper section, release note).
4. **Validate every renderer** you change — the JSON, Markdown, and CSV outputs all have regression tests; add one if you add a field.
5. **Run focused tests first** (`tests/test_evaluation_and_planning.py`, `tests/test_benchmark.py`), then the coverage gate from `testing-validation` if public behavior changed.

## Examples

**Honest claim:**
> "On 5 iterations against `mock`, `worldforge benchmark` reports a median latency of 2.1 ms per `predict` call (`benchmark-2026-04-21.json`)."

**Dishonest claim (do not produce):**
> "WorldForge's `planning` suite shows our provider achieves 92 % real-world success."
> The suite is a deterministic contract harness; it measures whether `plan_actions` returns a validated `Plan`, not real-world success.

**Capability mismatch:**
```bash
$ uv run worldforge eval --suite generation --provider leworldmodel
WorldForgeError: provider 'leworldmodel' does not advertise 'generate'
```
This is correct. Either pick a provider with `generate`, or pick a different suite. Do not catch the error.

## Activation cues

Trigger on:
- "run eval", "benchmark", "compare providers", "latency", "throughput", "retries"
- "renderer broke", "JSON output", "Markdown report", "CSV"
- "report claim", "publish numbers", "put in the README", "for the paper"
- "suite skipped", "WorldForgeError on suite"

Do **not** trigger for:
- adding / fixing the provider being benchmarked — load `provider-adapter-development`
- world persistence / save-load for snapshots being scored — load `persistence-state`
- live optional-runtime smoke runs — load `optional-runtime-smokes`

## Stop and ask the user

- before turning on a network-dependent benchmark in tests
- before publishing a comparison built on a single un-preserved run
- before catching a `WorldForgeError` from a suite (silent skip turns the contract harness into a no-op)

## Patterns

**Do:**
- Gate each suite by the exact capability it requires.
- Keep test data deterministic and small — fixtures, not generated noise.
- Include provider, operation, status, latency, and retry evidence in every report.
- Label checkout-safe demos as injected deterministic runtime checks, not real inference.

**Don't:**
- Claim physical fidelity, media quality, or robot safety from a deterministic suite.
- Quote a benchmark number whose underlying JSON wasn't preserved.
- Make benchmark tests depend on live credentials or wall-clock timing.
- Hide capability errors behind `except Exception`.

## Troubleshooting

| Symptom | Likely cause | First fix |
| --- | --- | --- |
| Suite errors on chosen provider | suite needs a capability the provider doesn't advertise | `worldforge doctor --capability <name>` then pick a matching provider |
| Benchmark retry counts look "too high" | `request_count` counts attempts, including retries | document attempt semantics; do not reinterpret as user operations |
| JSON renderer breaks tests | non-serialisable model field added | route through `to_dict()`; add a renderer regression test |
| Numbers swing wildly between runs | unpreserved noise / live network / shared host load | preserve the run, repeat N times, report median + range |

## References

- `docs/src/evaluation.md` — user-facing suite semantics
- `docs/src/benchmarking.md` — benchmark usage and reports
- `tests/test_evaluation_and_planning.py` — suite behavior
- `tests/test_benchmark.py` — benchmark behavior, including renderer regressions
- `src/worldforge/evaluation/` — suite implementations
- `src/worldforge/benchmark.py` — benchmark harness and `ProviderMetricsSink`
