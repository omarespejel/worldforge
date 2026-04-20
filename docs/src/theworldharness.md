# TheWorldHarness

TheWorldHarness is an optional Textual TUI for running WorldForge E2E demos as visible provider
workflows. It is meant to be the default integration reference for how provider surfaces, planning,
execution, persistence, and event inspection fit together.

## Install Boundary

Textual is optional. The base package keeps `httpx` as its only runtime dependency.

```bash
uv run --extra harness worldforge-harness
uv run --extra harness worldforge-harness --flow lerobot
uv run worldforge harness --list
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

## Interface Contract

The TUI is intentionally separated from the rest of the project:

| Module | Dependency boundary |
| --- | --- |
| `worldforge.harness.models` | Dataclasses only; no Textual import. |
| `worldforge.harness.flows` | Runs packaged demos and builds timeline, metrics, and transcript data; no Textual import. |
| `worldforge.harness.cli` | Lists flows without Textual; imports the TUI only when launching it. |
| `worldforge.harness.tui` | The only Textual-dependent module. |

The harness does not replace the Python APIs or command-line demos. It makes the same flows
observable: selected candidates, costs, action path, saved world id, final object position, and
provider event phases.

## Interaction Model

- `r`: run the selected flow.
- `1`: select LeWorldModel score planning.
- `2`: select LeRobot policy-plus-score planning.
- `q`: quit.

Each run reveals stages through a timeline, then fills the inspector and transcript panes from the
same structured `HarnessRun` data used by tests.
