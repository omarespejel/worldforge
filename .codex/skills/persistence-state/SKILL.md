---
name: persistence-state
description: "Use for WorldForge local JSON persistence, world CLI commands, world IDs, history entries, import/export/fork behavior, state validation, scene-object mutations, and recovery of `.worldforge/worlds` data."
prerequisites: "uv, pytest"
---

# Persistence And State

<purpose>
Maintain coherent local JSON world state and CLI persistence behavior.
</purpose>

<context>
World persistence is local JSON under `.worldforge/worlds` by default and is single-writer only. Hosts own durable storage beyond import/export. World IDs are file stems and must reject path separators, traversal, and unsafe names before load/import/save.
</context>

<procedure>
1. Use `WorldForge.create_world`, `create_world_from_prompt`, `save_world`, `load_world`, `export_world`, `import_world`, and `fork_world`; keep CLI handlers thin.
2. Validate world IDs and serialized state before filesystem reads/writes.
3. Preserve history contract: non-negative steps, non-empty summaries, valid snapshot states, valid serialized `Action` payloads when present, and no entry step greater than current world step.
4. Scene object add/update/remove mutations should append typed history entries without advancing provider time.
5. Position patches must translate bounding boxes with object poses.
6. Test CLI flows with `tmp_path` state dirs; do not use the user's real `.worldforge/` state.
</procedure>

<patterns>
<do>
- Raise `WorldStateError` for malformed persisted/provider state.
- Keep metadata, action parameters, scene objects, history, and patches JSON-native.
- Add import/export/fork regression tests when serialized contracts change.
</do>
<dont>
- Do not add lock files, SQLite, remote persistence, or service-grade durability without explicit design.
- Do not silently coerce invalid world state.
- Do not delete `.worldforge/` user data during tests or cleanup.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
| --- | --- | --- |
| Load rejects world ID | Unsafe file-stem shape | Use CLI-generated ID or sanitize before save/import |
| Fork/import fails | Snapshot history or action payload invalid | Validate serialized payload against framework validators |
| Object bbox drifts after move | Pose patch did not translate bounds | Update bbox by same delta as position |
</troubleshooting>

<references>
- `src/worldforge/framework.py`: world lifecycle, persistence, planning, validation.
- `src/worldforge/models.py`: state, action, scene, history models.
- `src/worldforge/cli.py`: shell-facing world commands.
- `tests/test_world_lifecycle.py`: state contract coverage.
- `tests/test_cli_world_commands.py`: CLI persistence coverage.
</references>
