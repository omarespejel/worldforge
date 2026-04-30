---
name: tui-development
description: "Use for TheWorldHarness work: Textual screens, flows, launchpad, command palette, world editor, screenshots, visual tests, and any change under `src/worldforge/harness/`."
prerequisites: "uv, pytest, textual extra"
---

# TUI Development

<purpose>
Keep TheWorldHarness usable while preserving Textual as an optional dependency.
</purpose>

<context>
The harness split is intentional: `harness.models` contains dataclasses, `harness.flows` contains runnable logic and summaries, `harness.cli` lists/launches flows without importing Textual, and `harness.tui` is the only Textual import surface. Current flows cover LeWorldModel score planning, LeRobot policy-plus-score planning, and provider diagnostics with benchmark comparison.
</context>

<procedure>
1. Read `src/worldforge/harness/models.py`, `flows.py`, `cli.py`, and only then `tui.py` or `worlds_view.py`.
2. Keep flow metadata/runners independent from Textual so `worldforge harness --list --format json` works without the `harness` extra.
3. Add or update tests in `tests/test_harness_cli.py`, `tests/test_harness_flows.py`, `tests/test_harness_guards.py`, `tests/test_harness_tui.py`, `tests/test_harness_worlds_view.py`, or `tests/test_harness_snapshots.py`.
4. For visual polish, update screenshots through existing scripts only when the UI behavior actually changed.
5. Run focused harness tests, then coverage with `--extra harness`.
</procedure>

<patterns>
<do>
- Use deterministic `mock` provider and temporary state dirs for diagnostics flows.
- Keep screen state, flow records, and summaries JSON-native where rendered or tested.
- Preserve keyboard/help affordances covered by snapshots.
</do>
<dont>
- Do not import Textual from `worldforge.__init__`, `worldforge.cli`, or non-TUI harness modules.
- Do not require live optional runtimes for the default harness tests.
- Do not commit generated screenshots unless they replace documented assets intentionally.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| Base import fails without Textual | Optional import leaked | Move Textual import under `harness/tui.py` or guarded launch path |
| Harness list command fails | Flow metadata coupled to TUI | Move metadata back into `models.py`/`flows.py` |
| Snapshot drift | UI text or command surface changed | Update snapshot only after verifying behavior |
</troubleshooting>

<references>
- `references/roadmap.md`: current M0-M5 spec map.
- `src/worldforge/harness/flows.py`: flow logic.
- `src/worldforge/harness/tui.py`: Textual app.
- `docs/src/theworldharness.md`: public harness docs.
</references>
