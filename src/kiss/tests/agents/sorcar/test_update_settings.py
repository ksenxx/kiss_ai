"""Integration tests for the update_settings tool in SorcarAgent.

Verifies that each setting parameter:
  - mutates the correct agent attribute,
  - persists config-level changes to disk,
  - broadcasts the correct UI event.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.vscode.vscode_config as vscode_config
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RecordingPrinter(BaseBrowserPrinter):
    """Subclass that captures every broadcast event (unfiltered)."""

    def __init__(self) -> None:
        super().__init__()
        self.all_events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record event and delegate to parent."""
        self.all_events.append(event)
        super().broadcast(event)


def _make_agent_and_printer() -> tuple[SorcarAgent, RecordingPrinter, list]:
    """Build a SorcarAgent with a recording printer and extract its tools."""
    agent = SorcarAgent("test-settings")
    printer = RecordingPrinter()
    agent.printer = printer
    # Disable web tools so _get_tools doesn't launch Chromium
    agent._use_web_tools = False
    tools = agent._get_tools()
    return agent, printer, tools


def _find_tool(tools: list, name: str) -> Any:
    """Find a tool function by its __name__."""
    matches = [t for t in tools if callable(t) and t.__name__ == name]
    assert matches, f"Tool '{name}' not found in {[t.__name__ for t in tools if callable(t)]}"
    return matches[0]


def _setting_events(printer: RecordingPrinter) -> list[dict[str, Any]]:
    """Return only updateSetting events from the recording."""
    return [e for e in printer.all_events if e.get("type") == "updateSetting"]


# ---------------------------------------------------------------------------
# Config isolation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path) -> Any:
    """Redirect vscode_config to a temp directory so tests never touch real config."""
    orig_dir = vscode_config.CONFIG_DIR
    orig_path = vscode_config.CONFIG_PATH

    test_dir = tmp_path / ".kiss"
    test_dir.mkdir()
    vscode_config.CONFIG_DIR = test_dir
    vscode_config.CONFIG_PATH = test_dir / "config.json"

    yield

    vscode_config.CONFIG_DIR = orig_dir
    vscode_config.CONFIG_PATH = orig_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUpdateSettingsToolPresence:
    """Verify the tool appears in the agent's tool list."""

    def test_update_settings_in_tools(self) -> None:
        _agent, _printer, tools = _make_agent_and_printer()
        names = [t.__name__ for t in tools if callable(t)]
        assert "update_settings" in names

    def test_ask_user_question_also_present(self) -> None:
        _agent, _printer, tools = _make_agent_and_printer()
        names = [t.__name__ for t in tools if callable(t)]
        assert "ask_user_question" in names


class TestNoArgs:
    """Calling with no arguments changes nothing."""

    def test_returns_no_change_message(self) -> None:
        _agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        result = update()
        assert "No settings were changed" in result
        assert _setting_events(printer) == []


class TestIsParallel:
    """is_parallel sets agent._is_parallel and broadcasts."""

    def test_enable(self) -> None:
        agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        assert agent._is_parallel is False

        result = update(is_parallel=True)
        assert "is_parallel=True" in result
        assert agent._is_parallel is True

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "is_parallel"
        assert evts[0]["value"] is True

    def test_disable(self) -> None:
        agent, printer, tools = _make_agent_and_printer()
        agent._is_parallel = True
        update = _find_tool(tools, "update_settings")

        result = update(is_parallel=False)
        assert "is_parallel=False" in result
        assert agent._is_parallel is False

        evts = _setting_events(printer)
        assert evts[0]["value"] is False

    def test_run_parallel_tool_appears_after_enable(self) -> None:
        agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        # Before enabling, run_parallel should not be in tools
        names_before = [t.__name__ for t in tools if callable(t)]
        assert "run_parallel" not in names_before

        update(is_parallel=True)
        # Rebuild tools — now run_parallel should appear
        new_tools = agent._get_tools()
        names_after = [t.__name__ for t in new_tools if callable(t)]
        assert "run_parallel" in names_after


class TestIsWorktree:
    """is_worktree only broadcasts (no agent attribute to change)."""

    def test_broadcast_only(self) -> None:
        _agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(is_worktree=True)
        assert "is_worktree=True" in result

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "is_worktree"
        assert evts[0]["value"] is True


class TestModel:
    """model sets agent.model_name and broadcasts."""

    def test_change_model(self) -> None:
        agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(model="gpt-4o")
        assert "model=gpt-4o" in result
        assert agent.model_name == "gpt-4o"

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "model"
        assert evts[0]["value"] == "gpt-4o"


class TestMaxBudget:
    """max_budget sets agent.max_budget, persists to config, and broadcasts."""

    def test_set_budget(self) -> None:
        agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(max_budget=7.5)
        assert "max_budget=7.5" in result
        assert agent.max_budget == 7.5

        # Check persistence
        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["max_budget"] == 7.5

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "max_budget"
        assert evts[0]["value"] == 7.5

    def test_budget_converts_to_float(self) -> None:
        agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        update(max_budget=3)  # int, not float
        assert isinstance(agent.max_budget, float)
        assert agent.max_budget == 3.0


class TestWorkingDirectory:
    """working_directory resolves, creates, sets agent.work_dir, persists."""

    def test_set_directory(self, tmp_path: Path) -> None:
        agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        target = tmp_path / "new_workdir"
        assert not target.exists()

        result = update(working_directory=str(target))
        resolved = str(target.resolve())
        assert f"working_directory={resolved}" in result
        assert agent.work_dir == resolved
        assert target.exists()

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["work_dir"] == resolved

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "working_directory"
        assert evts[0]["value"] == resolved

    def test_existing_directory_ok(self, tmp_path: Path) -> None:
        agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        target = tmp_path / "existing"
        target.mkdir()
        update(working_directory=str(target))
        assert agent.work_dir == str(target.resolve())


class TestUseWebBrowser:
    """use_web_browser sets agent._use_web_tools and persists."""

    def test_disable(self) -> None:
        agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        agent._use_web_tools = True

        result = update(use_web_browser=False)
        assert "use_web_browser=False" in result
        assert agent._use_web_tools is False

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["use_web_browser"] is False

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "use_web_browser"
        assert evts[0]["value"] is False

    def test_enable(self) -> None:
        agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        agent._use_web_tools = False

        update(use_web_browser=True)
        assert agent._use_web_tools is True


class TestRemotePassword:
    """remote_password persists to config; broadcast value is True (not password)."""

    def test_persist_password(self) -> None:
        _agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(remote_password="s3cret")
        assert "remote_password=<updated>" in result

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["remote_password"] == "s3cret"

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "remote_password"
        # Value broadcast is True, not the actual password
        assert evts[0]["value"] is True


class TestDemoMode:
    """demo_mode only broadcasts (no persistence)."""

    def test_enable(self) -> None:
        _agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(demo_mode=True)
        assert "demo_mode=True" in result

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "demo_mode"
        assert evts[0]["value"] is True

        # No config persistence for demo_mode
        assert not vscode_config.CONFIG_PATH.exists() or \
            "demo_mode" not in json.loads(vscode_config.CONFIG_PATH.read_text())


class TestAutoCommit:
    """auto_commit broadcasts only when True."""

    def test_trigger(self) -> None:
        _agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(auto_commit=True)
        assert "auto_commit=triggered" in result

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "auto_commit"
        assert evts[0]["value"] is True

    def test_false_does_nothing(self) -> None:
        _agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(auto_commit=False)
        assert "No settings were changed" in result
        assert _setting_events(printer) == []


class TestMultipleSettings:
    """Multiple settings changed in one call."""

    def test_two_settings(self) -> None:
        agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(is_parallel=True, model="gemini-2.5-pro")
        assert "is_parallel=True" in result
        assert "model=gemini-2.5-pro" in result
        assert agent._is_parallel is True
        assert agent.model_name == "gemini-2.5-pro"

        evts = _setting_events(printer)
        assert len(evts) == 2
        keys = {e["key"] for e in evts}
        assert keys == {"is_parallel", "model"}

    def test_all_settings_at_once(self, tmp_path: Path) -> None:
        agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        wd = tmp_path / "all_at_once"
        result = update(
            is_parallel=True,
            is_worktree=True,
            model="o3",
            max_budget=10.0,
            working_directory=str(wd),
            use_web_browser=False,
            remote_password="pw",
            demo_mode=True,
            auto_commit=True,
        )

        assert "is_parallel=True" in result
        assert "is_worktree=True" in result
        assert "model=o3" in result
        assert "max_budget=10.0" in result
        assert f"working_directory={wd.resolve()}" in result
        assert "use_web_browser=False" in result
        assert "remote_password=<updated>" in result
        assert "demo_mode=True" in result
        assert "auto_commit=triggered" in result

        # 9 settings → 9 broadcast events
        evts = _setting_events(printer)
        assert len(evts) == 9


class TestNoPrinterBroadcast:
    """Settings work even without a printer (no broadcast)."""

    def test_no_printer(self) -> None:
        agent = SorcarAgent("test-no-printer")
        agent._use_web_tools = False
        agent.printer = None
        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        result = update(is_parallel=True, model="test-model")
        assert "is_parallel=True" in result
        assert "model=test-model" in result
        assert agent._is_parallel is True
        assert agent.model_name == "test-model"

    def test_printer_without_broadcast(self) -> None:
        """A printer that has no broadcast attribute is handled gracefully."""
        from kiss.core.printer import Printer

        class MinimalPrinter(Printer):
            """Concrete printer without a broadcast method."""

            def print(self, content: Any, type: str = "text", **kw: Any) -> str:
                return ""

            def reset(self) -> None:
                pass

            def token_callback(self, token: str) -> None:
                pass

        agent = SorcarAgent("test-basic-printer")
        agent._use_web_tools = False
        agent.printer = MinimalPrinter()
        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        result = update(is_parallel=True)
        assert "is_parallel=True" in result
        assert agent._is_parallel is True


class TestConfigPersistenceIsolation:
    """Config writes are isolated between tests via the fixture."""

    def test_write_and_read_back(self) -> None:
        _agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        update(max_budget=42.0)
        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["max_budget"] == 42.0

    def test_previous_write_not_visible(self) -> None:
        """Confirm the previous test's max_budget=42 is not persisted here."""
        if vscode_config.CONFIG_PATH.exists():
            cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
            assert cfg.get("max_budget") != 42.0
        # else: config file doesn't exist yet → isolated correctly
