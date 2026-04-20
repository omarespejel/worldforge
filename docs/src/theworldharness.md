# TheWorldHarness

TheWorldHarness is an optional Textual TUI for running WorldForge integration flows as visible,
inspectable traces. It is the default integration reference for how provider surfaces, planning,
execution, persistence, diagnostics, benchmarks, and event inspection fit together.

It is a local tool. It does not require optional ML runtimes unless a selected flow explicitly
does, and the current flows use deterministic checkout-safe paths.

## Install Boundary

Textual is optional. The base package keeps `httpx` as its only runtime dependency.

```bash
uv run --extra harness worldforge-harness
uv run --extra harness worldforge-harness --flow lerobot
uv run --extra harness worldforge-harness --flow diagnostics
uv run worldforge harness --list
uv run worldforge harness --list --format json
```

Installed package:

```bash
pip install "worldforge[harness]"
worldforge-harness
```

Without the `harness` extra, metadata commands still work:

```bash
uv run worldforge harness --list
uv run worldforge harness --list --format json
```

Launching the TUI without Textual exits with an install hint instead of importing optional
dependencies at package import time.

## Current Flows

| Flow | Provider surface | What it visualizes |
| --- | --- | --- |
| `leworldmodel` | `score` | Deterministic LeWorldModel-shaped cost runtime, candidate scoring, score planning, execution, persistence, reload, provider events. |
| `lerobot` | `policy` plus score provider | Deterministic LeRobot-shaped policy, action translation, policy candidate ranking, execution, persistence, reload, provider events. |
| `diagnostics` | provider catalog plus benchmark harness | `doctor()` provider scan, registered/unregistered provider status, mock benchmark matrix across predict/reason/generate/transfer, latency/throughput comparison, provider events. |

## What The Interface Shows

Each flow is rendered from the same structured `HarnessRun` object used in tests:

- Timeline: ordered execution stages, boundary details, and produced artifacts.
- Inspector: compact metrics for the current run.
- Transcript: deterministic key-value output suitable for comparing runs.
- Flow rail: selectable integration references with keyboard shortcuts.

The diagnostics flow maps directly to the non-TUI commands:

```bash
uv run worldforge doctor
uv run worldforge provider list
uv run worldforge benchmark --provider mock --iterations 2 --format json
```

## Interface Contract

The TUI is intentionally separated from the rest of the project:

| Module | Dependency boundary |
| --- | --- |
| `worldforge.harness.models` | Dataclasses only; no Textual import. |
| `worldforge.harness.flows` | Runs packaged demos and builds timeline, metrics, and transcript data; no Textual import. |
| `worldforge.harness.cli` | Lists flows without Textual; imports the TUI only when launching it. |
| `worldforge.harness.tui` | The only Textual-dependent module. |

The harness does not replace the Python APIs or command-line demos. It makes the same flows
observable: selected candidates, costs, action paths, saved world ids, final object positions,
provider health, benchmark latency, benchmark throughput, and provider event phases.

## Interaction Model

- `r`: run the selected flow.
- `1`: select LeWorldModel score planning.
- `2`: select LeRobot policy-plus-score planning.
- `3`: select provider diagnostics and benchmark comparison.
- `q`: quit.

Each run reveals stages through a timeline, then fills the inspector and transcript panes from the
same structured `HarnessRun` data used by tests.
