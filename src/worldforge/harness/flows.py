"""Flow definitions and runners for TheWorldHarness."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from worldforge.demos import lerobot_e2e, leworldmodel_e2e
from worldforge.harness.models import HarnessFlow, HarnessMetric, HarnessRun, HarnessStep
from worldforge.models import JSONDict

FlowRunner = Callable[..., JSONDict]

FLOWS: tuple[HarnessFlow, ...] = (
    HarnessFlow(
        id="leworldmodel",
        title="LeWorldModel Score Planning",
        short_title="LeWorldModel",
        focus="score planning",
        provider="LeWorldModelProvider",
        capability="score",
        command="uv run worldforge-demo-leworldmodel",
        accent="#d8c46a",
        summary=(
            "Inject a deterministic LeWorldModel-shaped cost runtime, score three action "
            "candidates, execute the selected plan, and verify persisted world state."
        ),
    ),
    HarnessFlow(
        id="lerobot",
        title="LeRobot Policy + Score Planning",
        short_title="LeRobot",
        focus="policy plus score planning",
        provider="LeRobotPolicyProvider",
        capability="policy",
        command="uv run worldforge-demo-lerobot",
        accent="#8ec5a3",
        summary=(
            "Inject a deterministic LeRobot-shaped policy, translate raw action chunks, rank "
            "them with a score provider, execute, persist, and reload the resulting world."
        ),
    ),
)

_RUNNERS: dict[str, FlowRunner] = {
    "leworldmodel": leworldmodel_e2e.run_demo,
    "lerobot": lerobot_e2e.run_demo,
}


def available_flows() -> tuple[HarnessFlow, ...]:
    """Return flows available through TheWorldHarness."""

    return FLOWS


def flow_index() -> dict[str, HarnessFlow]:
    """Return available flows keyed by id."""

    return {flow.id: flow for flow in FLOWS}


def flow_to_dicts() -> tuple[JSONDict, ...]:
    """Return flow metadata for CLI JSON output."""

    return tuple(flow.to_dict() for flow in FLOWS)


def run_flow(flow_id: str, *, state_dir: Path | None = None) -> HarnessRun:
    """Execute one harness flow and return visualizable run data."""

    flows = flow_index()
    if flow_id not in flows:
        valid = ", ".join(sorted(flows))
        raise ValueError(f"unknown harness flow '{flow_id}'. Valid flows: {valid}.")

    flow = flows[flow_id]
    resolved_state_dir = state_dir or Path(
        tempfile.mkdtemp(prefix=f"worldforge-harness-{flow_id}-")
    )
    summary = _RUNNERS[flow_id](state_dir=resolved_state_dir, emit=False)
    return HarnessRun(
        flow=flow,
        state_dir=resolved_state_dir,
        summary=summary,
        steps=_steps_for(flow_id, summary),
        metrics=_metrics_for(flow_id, summary),
        transcript=_transcript_for(flow_id, summary),
    )


def _steps_for(flow_id: str, summary: JSONDict) -> tuple[HarnessStep, ...]:
    if flow_id == "leworldmodel":
        return (
            HarnessStep(
                "Register provider surface",
                "LeWorldModelProvider receives an injected deterministic cost runtime.",
                (
                    "Provider health is configured; optional checkpoint inference stays outside "
                    "the base package."
                ),
                "provider=leworldmodel capability=score",
            ),
            HarnessStep(
                "Build planning world",
                "Create a local world, add blue_cube, and encode an object_at goal.",
                _goal_result(summary),
                "world=leworldmodel-score-planning-demo",
            ),
            HarnessStep(
                "Score candidate futures",
                "Send pixel/action/goal tensors through score_actions.",
                _cost_result(summary),
                f"selected_candidate={summary['selected_candidate_index']}",
            ),
            HarnessStep(
                "Plan and execute",
                "World.plan consumes the score result and mock executes the selected actions.",
                _action_result(summary),
                f"planner={summary['plan']['planner']}",
            ),
            HarnessStep(
                "Persist and reload",
                "Save the final world to local JSON and reload it through WorldForge.",
                _final_position_result(summary),
                f"saved_world_id={summary['saved_world_id']}",
            ),
            HarnessStep(
                "Inspect provider events",
                "Capture emitted provider phases from the model boundary.",
                _event_result(summary),
                "event_handler=recording",
            ),
        )
    if flow_id == "lerobot":
        return (
            HarnessStep(
                "Register policy surface",
                (
                    "LeRobotPolicyProvider receives an injected deterministic policy and action "
                    "translator."
                ),
                "Provider health is configured; torch and policy checkpoints remain host-owned.",
                "provider=lerobot capability=policy",
            ),
            HarnessStep(
                "Build task world",
                "Create a local world, add blue_cube, and define the placement goal.",
                _goal_result(summary),
                "world=lerobot-policy-plus-score-demo",
            ),
            HarnessStep(
                "Select action chunks",
                "Call select_actions and preserve raw policy candidates before translation.",
                f"{summary['policy_candidate_count']} translated action chunks returned.",
                f"policy_select_calls={summary['policy_select_calls']}",
            ),
            HarnessStep(
                "Rank policy candidates",
                "Score translated action chunks by final distance to the goal.",
                _cost_result(summary),
                f"selected_candidate={summary['selected_candidate_index']}",
            ),
            HarnessStep(
                "Execute and persist",
                (
                    "Execute selected WorldForge actions, save the final world, and reload it "
                    "from disk."
                ),
                _final_position_result(summary),
                f"saved_world_id={summary['saved_world_id']}",
            ),
            HarnessStep(
                "Inspect provider events",
                "Capture provider phases and policy lifecycle calls.",
                _event_result(summary),
                f"reset_calls={summary['policy_reset_calls']}",
            ),
        )
    raise ValueError(f"unknown harness flow '{flow_id}'")


def _metrics_for(flow_id: str, summary: JSONDict) -> tuple[HarnessMetric, ...]:
    flow_label = "score" if flow_id == "leworldmodel" else "policy+score"
    return (
        HarnessMetric("Flow", flow_label, "WorldForge planning mode"),
        HarnessMetric("Candidates", str(len(summary["candidate_costs"])), "ranked action paths"),
        HarnessMetric("Selected", f"#{summary['selected_candidate_index']}", "lowest-cost path"),
        HarnessMetric("Final position", _position(summary), "reloaded world state"),
        HarnessMetric(
            "Events",
            str(len(summary["event_phases"])),
            ", ".join(summary["event_phases"]),
        ),
        HarnessMetric("State", Path(str(summary["state_dir"])).name, "local persistence root"),
    )


def _transcript_for(flow_id: str, summary: JSONDict) -> tuple[str, ...]:
    lines = [
        f"flow: {flow_id}",
        f"providers: {', '.join(summary['providers'])}",
        f"candidate_costs: {', '.join(str(score) for score in summary['candidate_costs'])}",
        f"selected_candidate: {summary['selected_candidate_index']}",
        f"selected_actions: {len(summary['selected_actions'])}",
        f"final_position: {_position(summary)}",
        f"saved_world_id: {summary['saved_world_id']}",
        f"events: {', '.join(summary['event_phases'])}",
    ]
    if flow_id == "lerobot":
        lines.extend(
            [
                f"policy_candidate_count: {summary['policy_candidate_count']}",
                f"policy_select_calls: {summary['policy_select_calls']}",
                f"policy_reset_calls: {summary['policy_reset_calls']}",
            ]
        )
    return tuple(lines)


def _goal_result(summary: JSONDict) -> str:
    goal = summary["goal"]["position"]
    return f"Goal position encoded at ({goal['x']:.2f}, {goal['y']:.2f}, {goal['z']:.2f})."


def _cost_result(summary: JSONDict) -> str:
    costs = ", ".join(f"{cost:.4f}" for cost in summary["candidate_costs"])
    return f"Costs [{costs}], selected #{summary['selected_candidate_index']}."


def _action_result(summary: JSONDict) -> str:
    return f"{len(summary['selected_actions'])} actions selected for execution."


def _final_position_result(summary: JSONDict) -> str:
    return f"Final cube position {_position(summary)} after reload."


def _event_result(summary: JSONDict) -> str:
    return f"Provider phases: {', '.join(summary['event_phases'])}."


def _position(summary: JSONDict) -> str:
    final = summary["final_cube_position"]
    return f"({final['x']:.2f}, {final['y']:.2f}, {final['z']:.2f})"
