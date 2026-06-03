# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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
from kiss.agents.vscode.json_printer import JsonPrinter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RecordingPrinter(JsonPrinter):
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
    """model_name sets agent.model_name and broadcasts."""

    def test_set_model(self) -> None:
        agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(model_name="gpt-4o")
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

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["demo_mode"] is True


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

        result = update(is_parallel=True, model_name="gemini-2.5-pro")
        assert "is_parallel=True" in result
        assert "model=gemini-2.5-pro" in result
        assert agent._is_parallel is True
        assert agent.model_name == "gemini-2.5-pro"

        evts = _setting_events(printer)
        assert len(evts) == 2
        keys = {e["key"] for e in evts}
        assert keys == {"is_parallel", "model"}

    def test_all_settings_at_once(self) -> None:
        agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(
            is_parallel=True,
            is_worktree=True,
            model_name="o3",
            max_budget=10.0,
            use_web_browser=False,
            remote_password="pw",
            demo_mode=True,
            auto_commit=True,
            custom_endpoint="http://localhost:8080/v1",
            custom_headers="X-Custom:val",
        )

        assert "is_parallel=True" in result
        assert "is_worktree=True" in result
        assert "model=o3" in result
        assert "max_budget=10.0" in result
        assert "use_web_browser=False" in result
        assert "remote_password=<updated>" in result
        assert "demo_mode=True" in result
        assert "auto_commit=triggered" in result
        assert "custom_endpoint=http://localhost:8080/v1" in result
        assert "custom_headers=<updated>" in result

        # 10 settings → 10 broadcast events
        evts = _setting_events(printer)
        assert len(evts) == 10


class TestCustomEndpoint:
    """custom_endpoint persists to config and broadcasts."""

    def test_set_endpoint(self) -> None:
        _agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(custom_endpoint="http://localhost:8080/v1")
        assert "custom_endpoint=http://localhost:8080/v1" in result

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["custom_endpoint"] == "http://localhost:8080/v1"

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "custom_endpoint"
        assert evts[0]["value"] == "http://localhost:8080/v1"

    def test_clear_endpoint(self) -> None:
        _agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        update(custom_endpoint="http://example.com")
        printer.all_events.clear()

        result = update(custom_endpoint="")
        assert "custom_endpoint=" in result

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["custom_endpoint"] == ""


class TestApiKeysRejected:
    """API key parameters are no longer accepted by update_settings."""

    @pytest.mark.parametrize("param", [
        "custom_api_key",
        "gemini_api_key",
        "openai_api_key",
        "anthropic_api_key",
        "together_api_key",
        "openrouter_api_key",
        "minimax_api_key",
    ])
    def test_api_key_param_raises_type_error(self, param: str) -> None:
        """Passing any API key parameter raises TypeError (unexpected kwarg)."""
        _agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        with pytest.raises(TypeError):
            update(**{param: "sk-test-12345"})


class TestCustomHeaders:
    """custom_headers persists to config and broadcasts."""

    def test_set_headers(self) -> None:
        _agent, printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        result = update(custom_headers="X-Custom:val1\nX-Other:val2")
        assert "custom_headers=<updated>" in result

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["custom_headers"] == "X-Custom:val1\nX-Other:val2"

        evts = _setting_events(printer)
        assert len(evts) == 1
        assert evts[0]["key"] == "custom_headers"
        assert evts[0]["value"] is True


class TestNoPrinterBroadcast:
    """Settings work even without a printer (no broadcast)."""

    def test_no_printer(self) -> None:
        agent = SorcarAgent("test-no-printer")
        agent._use_web_tools = False
        agent.printer = None
        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        result = update(is_parallel=True, model_name="test-model")
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


class TestWorktreeSorcarAgentNoPrinter:
    """Verify update_settings works on WorktreeSorcarAgent without a printer.

    Mirrors the Slack channel poller's ``_run_sorcar()`` which creates a
    ``WorktreeSorcarAgent`` and calls ``agent.run()`` without passing a
    printer.  All config-level settings must persist; no errors should
    occur when ``self.printer is None``.
    """

    def test_update_settings_in_tools(self) -> None:
        """Tool is present even without a printer."""
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        agent = WorktreeSorcarAgent("test-slack-poller")
        agent.printer = None
        agent._use_web_tools = False
        tools = agent._get_tools()
        names = [t.__name__ for t in tools if callable(t)]
        assert "update_settings" in names

    def test_config_persists_without_printer(self) -> None:
        """Config-level settings persist to disk even without a printer."""
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        agent = WorktreeSorcarAgent("test-slack-persist")
        agent.printer = None
        agent._use_web_tools = False
        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        result = update(
            is_parallel=True,
            is_worktree=True,
            model_name="gemini-2.5-pro",
            max_budget=5.0,
            use_web_browser=False,
            custom_endpoint="http://localhost:11434/v1",
            custom_headers="X-Slack:true",
        )

        # Agent attributes updated
        assert agent._is_parallel is True
        assert agent.model_name == "gemini-2.5-pro"
        assert agent.max_budget == 5.0
        assert agent._use_web_tools is False

        # Verify result message
        assert "is_parallel=True" in result
        assert "is_worktree=True" in result
        assert "model=gemini-2.5-pro" in result
        assert "max_budget=5.0" in result
        assert "use_web_browser=False" in result
        assert "custom_endpoint=http://localhost:11434/v1" in result
        assert "custom_headers=<updated>" in result

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["max_budget"] == 5.0
        assert cfg["is_worktree"] is True
        assert cfg["is_parallel"] is True
        assert cfg["use_web_browser"] is False
        assert cfg["custom_endpoint"] == "http://localhost:11434/v1"
        assert cfg["custom_headers"] == "X-Slack:true"

    def test_api_key_param_rejected_without_printer(self) -> None:
        """API key parameters raise TypeError (no longer accepted)."""
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        agent = WorktreeSorcarAgent("test-slack-apikey")
        agent.printer = None
        agent._use_web_tools = False
        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        with pytest.raises(TypeError):
            update(gemini_api_key="slack-gem-key")

    def test_remote_password_and_demo_mode(self) -> None:
        """remote_password and demo_mode work without a printer."""
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        agent = WorktreeSorcarAgent("test-slack-misc")
        agent.printer = None
        agent._use_web_tools = False
        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        result = update(remote_password="pw123", demo_mode=True, auto_commit=True)
        assert "remote_password=<updated>" in result
        assert "demo_mode=True" in result
        assert "auto_commit=triggered" in result

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["remote_password"] == "pw123"

    def test_no_args_returns_no_change(self) -> None:
        """Calling with no args returns 'no change' message without error."""
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        agent = WorktreeSorcarAgent("test-slack-noargs")
        agent.printer = None
        agent._use_web_tools = False
        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        result = update()
        assert "No settings were changed" in result


class TestWebPrinterBroadcast:
    """Verify update_settings events are recorded by WebPrinter.

    Mirrors the web app context where ``WebPrinter`` is used as the
    printer.  Events should be recorded via ``_record_event`` and be
    retrievable through ``stop_recording``.
    """

    def test_events_recorded(self) -> None:
        """updateSetting events are captured in WebPrinter's recording."""
        from kiss.agents.vscode.web_server import WebPrinter

        agent = SorcarAgent("test-web-printer")
        printer = WebPrinter()
        agent.printer = printer
        agent._use_web_tools = False

        # Recordings are task-id keyed; mimic the production lifecycle
        # by tagging this thread with a task id before recording.
        printer._thread_local.task_id = "task-test-events-recorded"
        # Start recording (mimics what the web server does per-task)
        printer.start_recording()

        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        result = update(is_parallel=True, model_name="claude-sonnet-4-20250514")
        assert "is_parallel=True" in result
        assert "model=claude-sonnet-4-20250514" in result

        # Peek at the raw recording (unfiltered)
        with printer._lock:
            key = printer._task_key()
            raw = list(printer._recordings.get(key, []))

        setting_evts = [e for e in raw if e.get("type") == "updateSetting"]
        assert len(setting_evts) == 2
        keys = {e["key"] for e in setting_evts}
        assert keys == {"is_parallel", "model"}

    def test_all_settings_recorded(self) -> None:
        """All 11 setting types are recorded by WebPrinter."""
        from kiss.agents.vscode.web_server import WebPrinter

        agent = SorcarAgent("test-web-all")
        printer = WebPrinter()
        agent.printer = printer
        agent._use_web_tools = False
        printer._thread_local.task_id = "task-test-all-settings"
        printer.start_recording()

        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        update(
            is_parallel=True,
            is_worktree=True,
            model_name="o3",
            max_budget=10.0,
            use_web_browser=False,
            remote_password="pw",
            demo_mode=True,
            auto_commit=True,
            custom_endpoint="http://localhost:8080/v1",
            custom_headers="X-Web:true",
        )

        with printer._lock:
            key = printer._task_key()
            raw = list(printer._recordings.get(key, []))

        setting_evts = [e for e in raw if e.get("type") == "updateSetting"]
        assert len(setting_evts) == 10

        expected_keys = {
            "is_parallel", "is_worktree", "model", "max_budget",
            "use_web_browser", "remote_password",
            "demo_mode", "auto_commit", "custom_endpoint",
            "custom_headers",
        }
        actual_keys = {e["key"] for e in setting_evts}
        assert actual_keys == expected_keys

    def test_secret_values_masked(self) -> None:
        """Secret settings broadcast True, not the actual value."""
        from kiss.agents.vscode.web_server import WebPrinter

        agent = SorcarAgent("test-web-secrets")
        printer = WebPrinter()
        agent.printer = printer
        agent._use_web_tools = False
        printer.start_recording()

        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        update(
            remote_password="top-secret",
            custom_headers="Authorization:Bearer xyz",
        )

        with printer._lock:
            key = printer._task_key()
            raw = list(printer._recordings.get(key, []))

        setting_evts = [e for e in raw if e.get("type") == "updateSetting"]
        for evt in setting_evts:
            assert evt["value"] is True, (
                f"{evt['key']} should broadcast True, got {evt['value']!r}"
            )
            # The actual secret value must NOT appear in the event
            assert "top-secret" not in json.dumps(evt)
            assert "Bearer xyz" not in json.dumps(evt)

    def test_config_persists_with_web_printer(self) -> None:
        """Config-level settings persist to disk when using WebPrinter."""
        from kiss.agents.vscode.web_server import WebPrinter

        agent = SorcarAgent("test-web-persist")
        printer = WebPrinter()
        agent.printer = printer
        agent._use_web_tools = False

        tools = agent._get_tools()
        update = _find_tool(tools, "update_settings")

        update(max_budget=25.0, custom_endpoint="http://local:1234/v1")

        cfg = json.loads(vscode_config.CONFIG_PATH.read_text())
        assert cfg["max_budget"] == 25.0
        assert cfg["custom_endpoint"] == "http://local:1234/v1"
        assert agent.max_budget == 25.0


class TestApiKeysCannotBeUpdated:
    """API keys cannot be updated through the update_settings tool.

    For security reasons, the tool rejects all API key parameters with a
    TypeError (Python's "unexpected keyword argument" error).  Users must
    set API keys through the settings UI or environment variables.
    """

    @pytest.mark.parametrize("api_key_param", [
        "custom_api_key",
        "gemini_api_key",
        "openai_api_key",
        "anthropic_api_key",
        "together_api_key",
        "openrouter_api_key",
        "minimax_api_key",
    ])
    def test_api_key_param_not_in_signature(self, api_key_param: str) -> None:
        """The API key parameter is not part of update_settings' signature."""
        import inspect

        _agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        sig = inspect.signature(update)
        assert api_key_param not in sig.parameters

    def test_api_key_not_advertised_in_docstring(self) -> None:
        """API keys should not appear in the docstring's Args list."""
        _agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        doc = update.__doc__ or ""
        # The docstring may mention API keys to explain they're disallowed,
        # but they must not be listed under "Args:".
        args_section_marker = "Args:"
        if args_section_marker in doc:
            args_section = doc.split(args_section_marker, 1)[1]
            # Split out the "Returns:" trailer so we only inspect the args.
            args_section = args_section.split("Returns:", 1)[0]
            for forbidden in (
                "custom_api_key:",
                "gemini_api_key:",
                "openai_api_key:",
                "anthropic_api_key:",
                "together_api_key:",
                "openrouter_api_key:",
                "minimax_api_key:",
            ):
                assert forbidden not in args_section, (
                    f"API key '{forbidden}' should not be documented as a"
                    " supported update_settings parameter."
                )


class TestSettingsPersistAcrossTasks:
    """Settings changed via update_settings must survive into the next task.

    Regression test: ``save_config`` previously dropped keys that were
    missing from ``DEFAULTS`` (e.g. ``is_worktree``, ``demo_mode``,
    ``work_dir``, ``is_parallel``), so those settings silently reverted
    to their defaults on the next ``load_config()`` call.
    """

    def test_is_parallel_persists(self) -> None:
        """is_parallel must be saved to config and reloaded."""
        _agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        update(is_parallel=True)

        cfg = vscode_config.load_config()
        assert cfg.get("is_parallel") is True

    def test_is_worktree_persists(self) -> None:
        """is_worktree must be saved to config and reloaded."""
        _agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        update(is_worktree=True)

        cfg = vscode_config.load_config()
        assert cfg.get("is_worktree") is True

    def test_demo_mode_persists(self) -> None:
        """demo_mode must be saved to config and reloaded."""
        _agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        update(demo_mode=True)

        cfg = vscode_config.load_config()
        assert cfg.get("demo_mode") is True

    def test_model_name_persists(self) -> None:
        """model_name must be persisted via _save_last_model."""
        from kiss.agents.sorcar.persistence import _load_last_model

        _agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")
        update(model_name="gemini-2.5-pro")

        assert _load_last_model() == "gemini-2.5-pro"

    def test_all_settings_survive_roundtrip(self) -> None:
        """Every config-persisted setting survives save → load."""
        _agent, _printer, tools = _make_agent_and_printer()
        update = _find_tool(tools, "update_settings")

        update(
            is_parallel=True,
            is_worktree=True,
            max_budget=42.0,
            use_web_browser=False,
            demo_mode=True,
            remote_password="secret",
            custom_endpoint="http://localhost:8080/v1",
            custom_headers="X-Test:val",
        )

        cfg = vscode_config.load_config()
        assert cfg["is_parallel"] is True
        assert cfg["is_worktree"] is True
        assert cfg["max_budget"] == 42.0
        assert cfg["use_web_browser"] is False
        assert cfg["demo_mode"] is True
        assert cfg["remote_password"] == "secret"
        assert cfg["custom_endpoint"] == "http://localhost:8080/v1"
        assert cfg["custom_headers"] == "X-Test:val"


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
