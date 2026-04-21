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
| `diagnostics` | provider catalog plus benchmark harness | `doctor()` provider scan, registered/unregistered provider status, mock benchmark matrix across predict/reason/generate/transfer/embed, latency/throughput comparison, provider events. |

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

## Roadmap

TheWorldHarness is evolving from the read-only demo viewer described above into the project's
front-door interactive workspace — keyboard-first, command-palette-driven, and the canonical
example of how to compose WorldForge from Python. The work is broken into six milestones
(M0–M5), each with a published spec triad (`spec.md` + `plan.md` + `tasks.md`) under
[`specs/`](https://github.com/AbdelStark/worldforge/tree/main/specs):

| Milestone | What it adds |
| --- | --- |
| [M0 — Theme + chrome reset](https://github.com/AbdelStark/worldforge/tree/main/specs/theworldharness-M0-theme-chrome) | Registered light/dark themes, semantic CSS variables, header clock and breadcrumb. |
| [M1 — Screen architecture](https://github.com/AbdelStark/worldforge/tree/main/specs/theworldharness-M1-screen-architecture) | App split into named `Screen`s, `push_screen` navigation, `?` help overlay, `Ctrl+P` system commands. |
| [M2 — Worlds CRUD](https://github.com/AbdelStark/worldforge/tree/main/specs/theworldharness-M2-worlds-crud) | Create / edit / save / fork / delete worlds entirely from the TUI through the public `WorldForge` API. |
| [M3 — Live providers](https://github.com/AbdelStark/worldforge/tree/main/specs/theworldharness-M3-live-providers) | `ProvidersScreen` with capability matrix and one real provider call streamed through a worker; `Esc` cancels. |
| [M4 — Eval + Benchmark](https://github.com/AbdelStark/worldforge/tree/main/specs/theworldharness-M4-eval-benchmark) | `EvalScreen` and `BenchmarkScreen`; capability mismatch as a hard toast; reports preserved to disk and exportable. |
| [M5 — Polish + showcase](https://github.com/AbdelStark/worldforge/tree/main/specs/theworldharness-M5-polish-showcase) | High-contrast theme, dynamic command-palette provider, recent items, snapshot test matrix, README screenshot refresh. |

The intent and design language behind these milestones live in the contributor-facing roadmap
spec at `.codex/skills/tui-development/references/roadmap.md`.
