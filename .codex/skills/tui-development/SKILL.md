---
name: tui-development
description: Use whenever the task touches TheWorldHarness — the optional Textual TUI under `src/worldforge/harness/`. Trigger on phrases like "the TUI", "TheWorldHarness", "harness screen", "Textual", "command palette", "DataTable", "RichLog", "ModalScreen", "TCSS", "snapshot test", "harness flow", "live event log", "demo flow", "make the harness pretty / interactive / glamorous", "add a screen for X". Also trigger on issues that surface as "TUI lags", "blocks the event loop", "wrong color in light mode", "key binding doesn't show in footer", or any change to `harness/tui.py`, `harness/flows.py`, `harness/cli.py`, or `harness/models.py`. Treat the harness as the project's *front face* — every change is also a UX choice, not just code.
---

# TUI Development (TheWorldHarness)

TheWorldHarness is the visual integration reference for WorldForge. It must be the most polished surface in the repo — first impression for every newcomer, and the canonical example of how WorldForge composes. Treat each change as a UX decision, not just an implementation detail.

The end-state vision and feature roadmap live in
`references/roadmap.md` — read it before proposing a new screen, flow,
or visual refactor.

## Fast start

```bash
# Run the harness (Textual extra required)
uv run --extra harness worldforge-harness

# Pick an initial flow / state dir from the wrapper CLI
uv run --extra harness worldforge-harness --flow leworldmodel
uv run --extra harness worldforge-harness --flow diagnostics --state-dir .worldforge/harness

# Tests + snapshot tests
uv run --extra harness pytest tests/test_harness_tui.py -x
uv run --extra harness pytest tests/test_harness_flows.py tests/test_harness_cli.py
```

Files you'll touch:

| Concern | Lives in |
| --- | --- |
| App, screens, widgets, bindings, theme | `src/worldforge/harness/tui.py` |
| Flow definitions and run orchestration | `src/worldforge/harness/flows.py` |
| Wrapper CLI entry point | `src/worldforge/harness/cli.py` |
| Typed flow / step / metric models | `src/worldforge/harness/models.py` |
| Textual extra gate | `pyproject.toml` `[project.optional-dependencies].harness` |

The Textual import boundary is **strict**: `tui.py` is the only module that may import Textual. Keep `flows.py`, `cli.py`, and `models.py` Textual-free so the rest of WorldForge can keep installing on a base `httpx`-only profile.

## Why this skill exists

Three failure modes worth defending against:

1. **Treating the harness as a slideshow.** The current implementation
   plays canned `asyncio.sleep` step animations. Every new feature should
   move it closer to a *real* harness — interactive worlds, live provider
   events, executable plans — and away from pre-baked transcripts. The
   roadmap reference catalogues that direction.
2. **Blocking the Textual event loop.** Provider calls, planner runs,
   evals, and benchmarks all take real time. They must run in `@work`
   workers, with `call_from_thread` for any UI mutation. Once the loop
   stalls, the harness *visibly fails*, which is the worst outcome for the
   project's front face.
3. **Theme drift.** The current TUI hard-codes hex colors (`#d8c46a`,
   `#8ec5a3`, …) inline. That ships an opinion about background color and
   breaks instantly on a light terminal. New work moves toward semantic
   CSS variables (`$accent`, `$success`, `$warning`, `$error`, `$panel`,
   `$boost`) and a registered `Theme` so light/dark/high-contrast all work.

## SOTA Textual practices (the short version)

Read the Textual docs sections named in `references/roadmap.md §3`
before doing anything load-bearing. The crystallised rules:

- **One `App`, many `Screen`s.** New top-level views are `Screen`
  subclasses pushed onto the stack — never `display:none` toggling on
  one mega-screen. Type `ModalScreen[T]` so dismiss results are checked.
- **Compose, don't subclass.** Reach for `Static` + CSS before writing a
  new `Widget`. Custom widgets are for new render or input models (e.g.,
  a 3-D world preview), not for re-skinning a panel.
- **Reactives are the state model.** `selected_world: reactive[str | None]
  = reactive(None)`; pair with `watch_selected_world(self, old, new)`.
  Use `validate_<name>` to enforce invariants — same fail-loud philosophy
  as the rest of WorldForge.
- **Messages over reach-across.** Children post `class WorldSelected
  (Message): world_id: str`; parents handle `on_world_selected`. Never
  `self.app.query_one(...).do_thing()` from a sibling.
- **Bindings on the screen, with `show=True` for the footer.** Internal
  bindings (`show=False`) keep the footer clean.
- **Command palette is the discovery surface.** Implement
  `get_system_commands` on the `App`; register a `textual.command.Provider`
  for fuzzy-searchable dynamic items (worlds, providers, recent runs).
  Every action must be reachable from `Ctrl+P`.
- **Long-running work uses workers.**

  ```python
  @work(exclusive=True, group="provider", thread=True, name="cosmos.generate")
  def run_generate(self, request) -> None:
      log = self.query_one(RichLog)
      for event in self.provider.stream(request):           # ProviderEvent
          if get_current_worker().is_cancelled:
              return
          self.app.call_from_thread(log.write, format_event(event))
      self.app.call_from_thread(self.post_message, ProviderDone(request.id))
  ```

  `exclusive=True` cancels in-flight work when re-invoked. `group="provider"`
  lets one `self.workers.cancel_group("provider")` bind to `Esc`.
  **Never** mutate widgets from a thread worker without `call_from_thread`.
- **CSS via TCSS, semantic variables only.** No hex literals in widget
  CSS. Use `$accent`, `$panel`, `:focus-within`, etc. Hierarchy comes from
  borders + padding + `Rule()`, not blank lines.
- **Test with `Pilot` and snapshots.** `async with app.run_test() as pilot`
  → drive with `pilot.press(...)` / `pilot.click(...)`. Visual regressions
  use `pytest-textual-snapshot` (`assert snap_compare(app, press=[...],
  terminal_size=(120, 40))`); commit the SVGs and review diffs in PRs.

## The procedure

1. **Decide which roadmap milestone the change belongs to** (`references/roadmap.md`). If it doesn't fit, that's the conversation to have *before* writing code.
2. **Pick a Screen, not a panel-on-an-existing-Screen.** Each user-visible mode (Worlds, Providers, Eval, Benchmark, Diagnostics, Run Inspector) is its own `Screen` subclass.
3. **Wire the data flow.** Reactives for state, messages for cross-widget signalling, workers for any call into `WorldForge`, `provider.*`, `eval`, `benchmark`, persistence.
4. **Add the binding *and* a command palette entry.** Mouse + keyboard parity. Footer never lies.
5. **Theme via semantic CSS variables.** If a hex literal feels necessary, it belongs in the registered `Theme`, not in widget CSS.
6. **Write a Pilot test for the interaction**, and a snapshot test for the visual state.
7. **Update `references/roadmap.md`** if the change resolves or reshapes a milestone — the spec must stay honest.

## Activation cues

Trigger on:
- "harness", "TheWorldHarness", "TUI", "Textual", "TCSS"
- "screen / push_screen / ModalScreen / push_screen_wait"
- "DataTable", "RichLog", "Tree", "TabbedContent", "Footer", "Header"
- "command palette", "Ctrl+P", "system commands", "command provider"
- "live event log", "stream events", "@work", "call_from_thread"
- "snapshot test", "pytest-textual-snapshot", "Pilot"
- any task editing `src/worldforge/harness/*.py` or `tests/test_harness_*.py`
- "make the harness <prettier | interactive | usable>", "harness roadmap"

Do **not** trigger for:
- Provider implementation work (load `provider-adapter-development`) — even if "the harness should call this provider" is in the user's words; first build the provider, then surface it.
- Benchmark or eval semantics (load `evaluation-benchmarking`) — the harness is the *display* layer for those.
- Persistence shape changes (load `persistence-state`) — the harness reads through the persistence API, not around it.

## Stop and ask the user

- before adding a runtime dependency that isn't already in the
  `harness` extra (Textual is the only one allowed today; rich/rich-style
  packages already come with Textual).
- before importing Textual outside `tui.py` — the boundary is load-bearing
  for the base package's installability rule.
- before introducing a new file under `src/worldforge/harness/` — confirm
  it can be imported without Textual when the extra is absent (or move
  Textual-using code to `tui.py`).
- before reshaping the public flow models (`HarnessFlow`, `HarnessRun`,
  `HarnessStep`, `HarnessMetric`) — they're exported from
  `worldforge.harness` and are part of the public surface that tests assert.
- before declaring a milestone in `references/roadmap.md` "done" — call
  out which screens / tests / docs land for the user to confirm.

## Patterns

**Do:**
- Run any provider / eval / benchmark / persistence call inside `@work`
  with explicit `group=` and `name=`.
- Stream events into `RichLog(highlight=True, markup=True, max_lines=5000)`.
- Show empty states ("No worlds yet — press [b]n[/] to create one"), not
  blank panels.
- Type modal results: `class ConfirmDelete(ModalScreen[bool])`.
- Pair every footer binding with a click target and a tooltip.
- Snapshot test default screens at `terminal_size=(120, 40)`.

**Don't:**
- Block the event loop with `time.sleep`, sync HTTP, or unawaited disk I/O.
- Mutate widgets from a thread worker without `call_from_thread`.
- Ship inline hex colors in TCSS — use semantic variables.
- Mount widgets in `__init__` (DOM doesn't exist yet) — use `compose` /
  `on_mount`.
- Reach across the DOM from inside a child widget — post a `Message`
  instead.
- Add more "demo flow" hard-codings; build the real surface that subsumes
  the demo.

## Troubleshooting

| Symptom | Likely cause | First fix |
| --- | --- | --- |
| TUI freezes during a provider call | sync work on the main task | wrap in `@work(thread=True, group="provider")` and stream via `call_from_thread` |
| Footer is empty / missing keys | bindings declared with `show=False` or on the App when they should be on the Screen | move binding to the active Screen; flip `show=True` |
| Light terminal looks broken | hex literals in widget CSS | replace with `$accent` / `$panel` / `$boost`; register a `Theme` if needed |
| Snapshot tests flake | terminal size or animation timing not pinned | pin `terminal_size=(120, 40)`; await `pilot.pause()` before assertion |
| `push_screen_wait` raises | called outside a worker context | wrap caller in `@work` |
| `harness` extra missing on import | `--extra harness` not provided to uv | add it; the import is intentionally gated |

## References

- `references/roadmap.md` — TheWorldHarness vision, screen inventory,
  flows, design language, milestone summaries (with status table)
- `specs/theworldharness-M{0..5}-*/` — per-milestone Spec Kit triad
  (`spec.md` = WHAT/WHY, `plan.md` = HOW, `tasks.md` = ordered
  PR-sized units). Read the relevant milestone's triad before
  implementing tasks for that milestone — the roadmap summary is the
  intent; the spec triad is the source of truth for scope, acceptance
  criteria, architecture decisions, and dependencies.
- `src/worldforge/harness/tui.py` — App, screens, widgets (current)
- `src/worldforge/harness/flows.py` — flow registry and run orchestration
- `src/worldforge/harness/models.py` — typed `HarnessFlow` / `HarnessRun`
- `tests/test_harness_tui.py` — Pilot-driven TUI tests
- Textual docs: Screens, Workers, Reactivity, Events & Messages, CSS,
  Command Palette, Animation, Testing
- `pytest-textual-snapshot` — visual regression tests
