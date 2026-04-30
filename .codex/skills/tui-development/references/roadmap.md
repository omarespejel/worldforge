# TheWorldHarness Spec Map

Current spec triads:
- `specs/theworldharness-M0-theme-chrome/{spec.md,plan.md,tasks.md}`
- `specs/theworldharness-M1-screen-architecture/{spec.md,plan.md,tasks.md}`
- `specs/theworldharness-M2-worlds-crud/{spec.md,plan.md,tasks.md}`
- `specs/theworldharness-M3-live-providers/{spec.md,plan.md,tasks.md}`
- `specs/theworldharness-M4-eval-benchmark/{spec.md,plan.md,tasks.md}`
- `specs/theworldharness-M5-polish-showcase/{spec.md,plan.md,tasks.md}`

When implementing a new harness milestone:
1. Update or add the spec triad first.
2. Keep non-visual logic in `harness.models` and `harness.flows`.
3. Keep `harness.cli` usable without Textual.
4. Put Textual-only code in `harness.tui` or view modules imported from it.
5. Validate with harness-focused tests and the `--extra harness` coverage gate.

