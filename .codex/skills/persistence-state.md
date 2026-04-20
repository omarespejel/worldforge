---
name: persistence-state
description: Use when changing WorldForge world IDs, local JSON persistence, save/load/import/export/fork behavior, history entries, world state validation, `.worldforge/worlds`, or recovery guidance for corrupted persisted state.
prerequisites: uv, pytest.
---

# Persistence And State

<purpose>
Protect local JSON world persistence as a deterministic single-writer contract with strict validation and clear recovery boundaries.
</purpose>

<context>
- Persistence implementation lives in `src/worldforge/framework.py`.
- Domain serialization and validation models live in `src/worldforge/models.py`.
- Default local state path is `.worldforge/worlds`; it is ignored runtime data.
- Persistence is not a concurrent database, migration layer, backup system, or service adapter.
</context>

<procedure>
1. Validate world IDs before any path construction, load, import, save, or export.
2. Reject path separators, traversal-shaped values, empty IDs, and non-file-safe IDs.
3. Validate persisted payloads before applying or replacing state.
4. Validate history entries: non-negative steps, non-empty summaries, valid snapshots, valid serialized `Action` payloads when present, and no history step greater than current world step.
5. Add tests for malformed JSON, malformed scene objects, invalid history, traversal IDs, and successful round trips.
6. Do not add locks, SQLite, migrations, object storage, or service persistence without explicit design approval.
</procedure>

<patterns>
<do>
- Use model `to_dict()`/`from_dict()` helpers instead of ad hoc JSON mutation.
- Copy state payloads before storing or returning them.
- Keep state recovery docs explicit: restore from host-owned backup/export.
- Treat provider-supplied state as untrusted until validated.
</do>
<dont>
- Do not silently coerce invalid persisted state.
- Do not write outside the configured state directory.
- Do not commit `.worldforge/` runtime data.
- Do not make local JSON multi-writer by adding a lock file without design approval.
</dont>
</patterns>

<example>
```python
forge = WorldForge(state_dir=".worldforge/worlds")
world = forge.create_world("lab", provider="mock")
world_id = forge.save_world(world)
payload = forge.export_world(world_id)
restored = forge.import_world(payload, new_id=True, name="lab-copy")
forge.save_world(restored)
```
</example>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| load rejects world ID | unsafe file-stem input | use file-safe letters, numbers, `.`, `_`, `-` |
| import rejects history | invalid step/action/snapshot contract | fix producer or add validation test; do not coerce |
| state disappears between runs | `.worldforge/` is runtime ignored data | persist/export through host-owned storage when needed |
</troubleshooting>

<references>
- `tests/test_world_lifecycle.py`: persistence and world lifecycle behavior.
- `tests/test_helper_validations.py`: validation edge cases.
- `docs/src/playbooks.md`: local JSON operations and recovery.
- `docs/src/operations.md`: persistence boundaries.
</references>
