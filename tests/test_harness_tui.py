from __future__ import annotations

import asyncio

import pytest


def test_the_world_harness_app_runs_leworldmodel_flow(tmp_path) -> None:
    pytest.importorskip("textual")

    from worldforge.harness.tui import TheWorldHarnessApp

    async def scenario() -> None:
        app = TheWorldHarnessApp(
            initial_flow_id="leworldmodel",
            state_dir=tmp_path,
            step_delay=0.0,
        )
        async with app.run_test(size=(130, 42)) as pilot:
            await pilot.press("r")
            await pilot.pause()
            assert app.last_run is not None
            assert app.last_run.flow.id == "leworldmodel"
            assert app.last_run.summary["selected_candidate_index"] == 1
            assert app.query_one("#inspector") is not None

    asyncio.run(scenario())


def test_the_world_harness_app_switches_to_lerobot_flow(tmp_path) -> None:
    pytest.importorskip("textual")

    from worldforge.harness.tui import TheWorldHarnessApp

    async def scenario() -> None:
        app = TheWorldHarnessApp(
            initial_flow_id="leworldmodel",
            state_dir=tmp_path,
            step_delay=0.0,
        )
        async with app.run_test(size=(130, 42)) as pilot:
            await pilot.press("2")
            await pilot.press("r")
            await pilot.pause()
            assert app.last_run is not None
            assert app.last_run.flow.id == "lerobot"
            assert app.last_run.summary["policy_candidate_count"] == 3

    asyncio.run(scenario())


def test_the_world_harness_app_switches_to_diagnostics_flow(tmp_path) -> None:
    pytest.importorskip("textual")

    from worldforge.harness.tui import TheWorldHarnessApp

    async def scenario() -> None:
        app = TheWorldHarnessApp(
            initial_flow_id="leworldmodel",
            state_dir=tmp_path,
            step_delay=0.0,
        )
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.press("3")
            await pilot.press("r")
            await pilot.pause()
            assert app.last_run is not None
            assert app.last_run.flow.id == "diagnostics"
            assert app.last_run.summary["benchmark_operation_count"] == 4

    asyncio.run(scenario())
