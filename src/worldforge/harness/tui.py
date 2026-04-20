"""Textual TUI for TheWorldHarness."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Select, Static

from worldforge.harness.flows import available_flows, run_flow
from worldforge.harness.models import HarnessFlow, HarnessRun, HarnessStep


class HeroPane(Static):
    """Top-level harness identity panel."""

    def compose_panel(self, flow: HarnessFlow | None, running: bool) -> RenderableType:
        title = Text("TheWorldHarness", style="bold")
        title.append("  /  visual WorldForge integration reference", style="dim")
        command = flow.command if flow else "select a flow"
        status = "RUNNING" if running else "READY"
        status_style = "black on #d8c46a" if running else "black on #8ec5a3"
        return Panel(
            Group(
                title,
                Text(""),
                Text(
                    flow.summary if flow else "Run an E2E flow and inspect every boundary.",
                    style="dim",
                ),
                Text(""),
                Text(f"Command: {command}", style="bold #d8c46a"),
                Text(f"Status: {status}", style=status_style),
            ),
            title="WORLD FORGE / HARNESS",
            border_style="#d8c46a" if running else "#8ec5a3",
        )


class FlowCard(Static):
    """Flow selection card."""

    def render_flow(self, flow: HarnessFlow, selected: bool) -> None:
        marker = ">>" if selected else "  "
        style = "bold #d8c46a" if selected else "#d3d6cf"
        self.update(
            Panel(
                Group(
                    Text(f"{marker} {flow.short_title}", style=style),
                    Text(flow.focus, style="dim"),
                    Text(flow.provider, style="#8ec5a3"),
                ),
                border_style=flow.accent if selected else "#3b423e",
            )
        )


class TimelinePane(Static):
    """Visual execution timeline."""

    def render_steps(
        self,
        flow: HarnessFlow,
        steps: tuple[HarnessStep, ...],
        active_index: int,
        complete_count: int,
    ) -> None:
        rows: list[RenderableType] = []
        for index, step in enumerate(steps):
            if index < complete_count:
                symbol = "OK"
                style = "#8ec5a3"
            elif index == active_index:
                symbol = ">>"
                style = "#d8c46a"
            else:
                symbol = "--"
                style = "#6f7770"
            rows.append(Text(f"{symbol} {index + 1:02d}. {step.title}", style=f"bold {style}"))
            rows.append(Text(f"    {step.detail}", style="dim"))
            if index < complete_count:
                rows.append(Text(f"    {step.result}", style="#d3d6cf"))
                if step.artifact:
                    rows.append(Text(f"    {step.artifact}", style="#8ec5a3"))
            rows.append(Text(""))
        self.update(
            Panel(
                Group(*rows),
                title=f"{flow.title} / execution trace",
                border_style=flow.accent,
            )
        )


class InspectorPane(Static):
    """Run metrics and state summary."""

    def render_empty(self) -> None:
        self.update(
            Panel(
                Align.center(
                    Text(
                        "Run a flow to populate metrics, state, and provider events.", style="dim"
                    ),
                    vertical="middle",
                ),
                title="Inspector",
                border_style="#3b423e",
            )
        )

    def render_run(self, run: HarnessRun) -> None:
        table = Table.grid(expand=True)
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="right", ratio=1)
        for metric in run.metrics:
            table.add_row(Text(metric.label, style="dim"), Text(metric.value, style="bold #d8c46a"))
            if metric.detail:
                table.add_row("", Text(metric.detail, style="#8ec5a3"))
        self.update(Panel(table, title="Inspector", border_style=run.flow.accent))


class TranscriptPane(Static):
    """Structured transcript from the completed flow."""

    def render_empty(self) -> None:
        self.update(Panel(Text("Awaiting run output.", style="dim"), title="Run Transcript"))

    def render_run(self, run: HarnessRun) -> None:
        lines = [Text(line, style="#d3d6cf") for line in run.transcript]
        self.update(Panel(Group(*lines), title="Run Transcript", border_style=run.flow.accent))


class TheWorldHarnessApp(App[None]):
    """Visual TUI harness for WorldForge E2E demos."""

    TITLE = "TheWorldHarness"
    SUB_TITLE = "WorldForge visual integration harness"
    BINDINGS = [
        ("r", "run_selected", "Run"),
        ("1", "select_flow('leworldmodel')", "LeWorldModel"),
        ("2", "select_flow('lerobot')", "LeRobot"),
        ("3", "select_flow('diagnostics')", "Diagnostics"),
        ("q", "quit", "Quit"),
    ]
    CSS = """
    Screen {
        background: #101512;
        color: #d3d6cf;
    }

    Header {
        background: #171f1a;
        color: #e4dfc5;
    }

    Footer {
        background: #171f1a;
        color: #9ea89f;
    }

    #root {
        height: 1fr;
        padding: 1 2;
    }

    #hero {
        height: 10;
        margin-bottom: 1;
    }

    #body {
        height: 1fr;
    }

    #rail {
        width: 30;
        margin-right: 1;
    }

    #timeline {
        width: 1fr;
        margin-right: 1;
    }

    #inspector-column {
        width: 42;
    }

    FlowCard {
        height: 6;
        margin-bottom: 1;
    }

    Select {
        margin-bottom: 1;
    }

    Button {
        margin-bottom: 1;
    }

    #transcript {
        height: 14;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        *,
        initial_flow_id: str = "leworldmodel",
        state_dir: Path | None = None,
        step_delay: float = 0.18,
    ) -> None:
        super().__init__()
        self.flows = {flow.id: flow for flow in available_flows()}
        self.selected_flow_id = initial_flow_id if initial_flow_id in self.flows else "leworldmodel"
        self.state_dir = state_dir
        self.step_delay = step_delay
        self.running = False
        self.last_run: HarnessRun | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="root"):
            yield HeroPane(id="hero")
            with Horizontal(id="body"):
                with Vertical(id="rail"):
                    yield Select(
                        [(flow.title, flow.id) for flow in self.flows.values()],
                        value=self.selected_flow_id,
                        allow_blank=False,
                        id="flow-select",
                    )
                    yield Button("Run selected flow", id="run-button", variant="warning")
                    for flow in self.flows.values():
                        yield FlowCard(id=f"flow-card-{flow.id}")
                yield TimelinePane(id="timeline")
                with Vertical(id="inspector-column"):
                    yield InspectorPane(id="inspector")
                    yield TranscriptPane(id="transcript")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_static()

    @on(Select.Changed, "#flow-select")
    def _on_flow_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, str):
            self.selected_flow_id = event.value
            self._refresh_static()

    @on(Button.Pressed, "#run-button")
    async def _on_run_pressed(self) -> None:
        await self.action_run_selected()

    def action_select_flow(self, flow_id: str) -> None:
        if flow_id in self.flows:
            self.selected_flow_id = flow_id
            self.query_one("#flow-select", Select).value = flow_id
            self._refresh_static()

    async def action_run_selected(self) -> None:
        if self.running:
            return
        await self._run_flow(self.selected_flow_id)

    async def _run_flow(self, flow_id: str) -> None:
        self.running = True
        flow = self.flows[flow_id]
        self.query_one("#run-button", Button).disabled = True
        self._refresh_static()
        run = run_flow(flow_id, state_dir=self.state_dir)
        self.last_run = run
        timeline = self.query_one("#timeline", TimelinePane)
        for index, _step in enumerate(run.steps):
            timeline.render_steps(flow, run.steps, active_index=index, complete_count=index)
            await asyncio.sleep(self.step_delay)
            timeline.render_steps(flow, run.steps, active_index=index, complete_count=index + 1)
        self.query_one("#inspector", InspectorPane).render_run(run)
        self.query_one("#transcript", TranscriptPane).render_run(run)
        self.running = False
        self.query_one("#run-button", Button).disabled = False
        self._refresh_static()

    def _refresh_static(self) -> None:
        selected = self.flows[self.selected_flow_id]
        self.query_one("#hero", HeroPane).update(
            self.query_one("#hero", HeroPane).compose_panel(selected, self.running)
        )
        for flow in self.flows.values():
            self.query_one(f"#flow-card-{flow.id}", FlowCard).render_flow(
                flow,
                selected=flow.id == self.selected_flow_id,
            )
        if self.last_run is None:
            self.query_one("#timeline", TimelinePane).render_steps(
                selected,
                (
                    HarnessStep(
                        "Ready",
                        "Select a flow and press Run to visualize the integration path.",
                        "Waiting for execution.",
                    ),
                ),
                active_index=0,
                complete_count=0,
            )
            self.query_one("#inspector", InspectorPane).render_empty()
            self.query_one("#transcript", TranscriptPane).render_empty()
