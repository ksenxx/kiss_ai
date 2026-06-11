# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ``update_settings(auto_commit=...)`` in SorcarAgent.

Reproduces the bug where ``update_settings(auto_commit=False)`` recorded
nothing and returned the false message ``"No settings were changed (all
arguments were None)."`` even though an argument WAS provided.  Every other
flag (demo_mode / is_parallel / use_web_browser) handles falsy values
properly; ``auto_commit=False`` must likewise be acknowledged as a provided
argument (one-shot action explicitly NOT triggered) without committing
anything or broadcasting any setting event.

No mocks: the real agent is constructed and the real tool closure returned
by ``SorcarAgent._get_tools()`` is invoked directly (no LLM calls).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import kiss.agents.vscode.vscode_config as vscode_config
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.json_printer import JsonPrinter


class RecordingPrinter(JsonPrinter):
    """Printer subclass that captures every broadcast event (unfiltered)."""

    def __init__(self) -> None:
        super().__init__()
        self.all_events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the event and delegate to the parent implementation."""
        self.all_events.append(event)
        super().broadcast(event)


def _make_update_settings(
    work_dir: Path | None = None,
) -> tuple[SorcarAgent, RecordingPrinter, Any]:
    """Build a real SorcarAgent and return its ``update_settings`` tool."""
    agent = SorcarAgent("test-auto-commit")
    printer = RecordingPrinter()
    agent.printer = printer
    # Disable web tools so _get_tools doesn't launch Chromium.
    agent._use_web_tools = False
    if work_dir is not None:
        agent.work_dir = str(work_dir)
    tools = agent._get_tools()
    matches = [
        t for t in tools if callable(t) and t.__name__ == "update_settings"
    ]
    assert matches, "update_settings tool not found"
    return agent, printer, matches[0]


def _setting_events(printer: RecordingPrinter) -> list[dict[str, Any]]:
    """Return only updateSetting events from the recording."""
    return [e for e in printer.all_events if e.get("type") == "updateSetting"]


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path) -> Any:
    """Redirect vscode_config to a temp dir so tests never touch real config."""
    orig_dir = vscode_config.CONFIG_DIR
    orig_path = vscode_config.CONFIG_PATH

    test_dir = tmp_path / ".kiss"
    test_dir.mkdir()
    vscode_config.CONFIG_DIR = test_dir
    vscode_config.CONFIG_PATH = test_dir / "config.json"

    yield

    vscode_config.CONFIG_DIR = orig_dir
    vscode_config.CONFIG_PATH = orig_path


class TestAutoCommitFalse:
    """``auto_commit=False`` is a provided argument and must be reported."""

    def test_false_is_not_reported_as_all_none(self, tmp_path: Path) -> None:
        """The lying no-change message must not be returned for False."""
        _agent, _printer, update = _make_update_settings(tmp_path)
        result = update(auto_commit=False)
        assert result != "No settings were changed (all arguments were None)."
        assert "No settings were changed" not in result

    def test_false_returns_accurate_message(self, tmp_path: Path) -> None:
        """False is acknowledged as 'not triggered', mirroring True's report."""
        _agent, _printer, update = _make_update_settings(tmp_path)
        result = update(auto_commit=False)
        assert result.startswith("Updated: ")
        assert "auto_commit=not triggered (False)" in result

    def test_false_broadcasts_no_setting_event(self, tmp_path: Path) -> None:
        """False must not broadcast updateSetting nor commit anything."""
        _agent, printer, update = _make_update_settings(tmp_path)
        update(auto_commit=False)
        assert _setting_events(printer) == []


class TestAutoCommitNoneAndTrue:
    """Behavior for omitted and True values is unchanged."""

    def test_no_args_still_reports_no_change(self, tmp_path: Path) -> None:
        """Calling with no arguments must keep the no-change message."""
        _agent, printer, update = _make_update_settings(tmp_path)
        result = update()
        assert "No settings were changed" in result
        assert _setting_events(printer) == []

    def test_true_triggers_and_broadcasts(self, tmp_path: Path) -> None:
        """auto_commit=True in a non-git dir still reports 'triggered'."""
        _agent, printer, update = _make_update_settings(tmp_path)
        result = update(auto_commit=True)
        assert "auto_commit=triggered" in result
        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "auto_commit"
        assert evts[0]["value"] is True

    def test_false_combined_with_other_setting(self, tmp_path: Path) -> None:
        """False combines with other provided settings in one report."""
        agent, _printer, update = _make_update_settings(tmp_path)
        result = update(is_parallel=True, auto_commit=False)
        assert "is_parallel=True" in result
        assert "auto_commit=not triggered (False)" in result
        assert agent._is_parallel is True
