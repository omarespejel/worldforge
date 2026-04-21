---
name: persistence-state
description: Use whenever the task touches WorldForge world IDs, the local JSON persistence layer, save / load / import / export / fork / clone behavior, history entries, world state validation, the `.worldforge/worlds` directory, or recovery guidance for a corrupted or rejected persisted state. Trigger on phrases like "save the world", "load by id", "import this snapshot", "history rejected", "step number off", "world id invalid", "persistence path", "state directory", "round-trip", ".worldforge ignored", "what happens if the JSON is malformed". Also trigger on any suggestion to add locks, SQLite, migrations, or a database — those are out of scope for this layer and need explicit design approval.
---

# Persistence And State

WorldForge's local JSON persistence is a **deterministic single-writer contract** with strict validation at every boundary. It is not a database, not a migration framework, not a multi-writer store, and not a backup system. Treat persisted JSON as untrusted input: validate before applying, reject loudly on shape violations, and never coerce silently.

## Fast start

```python
from worldforge import WorldForge

forge = WorldForge(state_dir=".worldforge/worlds")
world = forge.create_world("lab", provider="mock")
world_id = forge.save_world(world)

payload = forge.export_world(world_id)
restored = forge.import_world(payload, new_id=True, name="lab-copy")
forge.save_world(restored)
```

```bash
# CLI parity for the same flow
uv run worldforge world create lab --provider mock
uv run worldforge world add-object <id> cube --x 0 --y 0.5 --z 0
uv run worldforge world predict <id> --x 0.4 --y 0.5 --z 0
```

Focused tests for any change here:
```bash
uv run pytest tests/test_world_lifecycle.py tests/test_helper_validations.py
```

## Why this skill exists

Three properties matter for this layer, and they're the entire reason it stays
small:

1. **Single-writer determinism.** Concurrent writers to the same JSON file is a
   data-loss bug class we do not want to debug. The contract is "one
   `WorldForge` process per `state_dir`". No locks, no coordination — just the
   contract.
2. **Strict validation at the boundary.** Persisted JSON is untrusted: it could
   be hand-edited, copied across versions, or produced by a bug in an earlier
   release. Every load, import, and history-replay validates shape *before*
   applying state. A silent coercion here corrupts the world.
3. **Deliberately small surface.** Adding SQLite, migrations, a lock file, or
   service persistence each opens a new failure mode and a new on-call
   commitment. They are out of scope until a dedicated design is approved.

`.worldforge/` is runtime data — gitignored, host-owned. Nothing in the repo
should treat it as a backup or as durable cross-environment storage.

## The procedure

1. **Validate world IDs before any path construction.** Reject path separators,
   traversal-shaped values (`..`, leading `/`, leading `~`), empty IDs, and
   anything outside file-safe characters (`A-Za-z0-9._-`). The ID becomes a
   filename; an unsafe ID is an unsafe filesystem operation.
2. **Validate persisted payloads before applying or replacing state**, even on
   "trusted" round trips. A round trip from a corrupted store still produces
   garbage; the validator is the firewall.
3. **Validate history entries**: non-negative `step`, non-empty summary, valid
   snapshot, valid serialized `Action` payload when present, and no history
   step greater than the world's current step. A history past the present is
   how silent state divergence looks.
4. **Add tests for malformed JSON, malformed scene objects, invalid history,
   traversal IDs, and successful round trips.** Failure paths are where this
   contract has historically broken.
5. **Do not add locks, SQLite, migrations, object storage, or a service**
   without explicit design approval. If the user asks for one of these, route
   to the gated list rather than improvising.

## Examples

**Safe round trip:**
```python
forge = WorldForge(state_dir=".worldforge/worlds")
world = forge.create_world("lab", provider="mock")
world_id = forge.save_world(world)        # validated on save
payload = forge.export_world(world_id)    # serialised payload
restored = forge.import_world(payload, new_id=True, name="lab-copy")  # validated on import
```

**Rejection (correct behavior — do not "fix" by coercing):**
```python
forge.save_world(World(id="../escape"))   # raises WorldStateError: unsafe world id
forge.import_world({"history": [{"step": 99, ...}]})  # raises if world.step < 99
```

## Activation cues

Trigger on:
- "save world", "load world", "export", "import", "fork", "clone"
- "history rejected", "step too high", "round trip"
- "world id invalid", "path traversal", "unsafe filename"
- "`.worldforge/`", "state dir", "persistence path"
- "should we use SQLite / locks / a database here?" — answer is the **stop and ask** section

Do **not** trigger for:
- provider-side state (that's `provider-adapter-development`)
- evaluation snapshots used for scoring (that's `evaluation-benchmarking`)
- TUI persistence integration tests if the question is about Textual rendering rather than persistence

## Stop and ask the user

- before adding lock files, SQLite, migrations, object storage, or any
  service-backed persistence — these are explicitly gated by the project's
  `<priority_rules>` (preserve local-first scope)
- before relaxing world-ID validation
- before changing the persisted JSON shape in a way that previous saves
  cannot be loaded — that's a migration question, which itself is gated

## Patterns

**Do:**
- Use model `to_dict()` / `from_dict()` helpers; do not hand-mutate JSON.
- Copy state payloads before storing or returning them — shared mutable refs
  cause action-at-a-distance bugs.
- Treat provider-supplied state as untrusted until validated.
- Document recovery as "restore from a host-owned export"; the repo is not a
  backup service.

**Don't:**
- Silently coerce invalid persisted state (truncate history, clamp steps,
  drop unknown fields). Raise `WorldStateError` instead.
- Write outside the configured `state_dir`.
- Commit `.worldforge/` runtime data.
- Make local JSON multi-writer by adding a lock file without design approval.

## Troubleshooting

| Symptom | Likely cause | First fix |
| --- | --- | --- |
| `load_world` rejects ID | unsafe file-stem input | use only `A-Za-z0-9._-`; reject the input upstream |
| `import_world` rejects history | invalid step / action / snapshot contract | fix the producer or add a validation test; do not coerce |
| State disappears between runs | `.worldforge/` is gitignored runtime data | persist or export through host-owned storage when needed |
| Two processes writing same `state_dir` | single-writer contract violated | run a single `WorldForge` per directory; do not add a lock to "fix" it |
| Round trip fails after upgrade | persisted shape changed | this is a migration question; ask before changing shape |

## References

- `src/worldforge/framework.py` — persistence implementation
- `src/worldforge/models.py` — domain serialisation + validation models
- `tests/test_world_lifecycle.py` — persistence and lifecycle behavior
- `tests/test_helper_validations.py` — validation edge cases
- `docs/src/playbooks.md` — local JSON operations and recovery
- `docs/src/operations.md` — persistence boundaries
