# DimOS Go2 Trace Judge

- run_id: dimos-go2-sample
- selected_candidate_id: detour_right
- selected_score: 0.8832
- worldforge_executes_robot: false
- host_runtime: mock-dimos

| Candidate | Score | Decision | Reason |
| --- | ---: | --- | --- |
| `scan_left` | 0.8065 | rejected | rejected: high information gain and aligned with the gesture bearing |
| `go_forward_small` | 0.6477 | rejected | rejected: good physical progress, but higher obstacle and stuck risk |
| `detour_right` | 0.8832 | selected | selected: best balance of safe progress, target alignment, and new coverage |
| `stop_relocalize` | 0.3708 | rejected | rejected: safe fallback, but not needed while the host reports localization |

## Mock Outcome

- progress_delta_m: 0.28
- coverage_delta: 0.12
- target_confidence_delta: 0.18
- blocked: false
- manual_intervention: false

This artifact is an offline, experimental host-integration example. DimOS or another host runtime owns robot networking, controller execution, operator supervision, and emergency stop behavior.
