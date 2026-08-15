# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ChatSorcarAgent's no-op ``summary`` tool.

Feature: the agent may periodically summarize what it did in the last
6 steps by calling ``summary(description=...)``, as requested by the
SYSTEM.md instructions.  The tool itself does nothing (the chat
webview reacts to the ``tool_call`` event by nesting and collapsing
the preceding panels — covered by the jsdom suite in
``src/kiss/agents/vscode/test/summaryToolCollapse.test.js``).

There is deliberately NO mechanical enforcement of the every-5-steps
cadence: ``ChatSorcarAgent`` installs no ``tool_call_guard`` and no
``pre_step_hook`` for summaries (the hardwired gate that once rejected
every other tool call on steps divisible by 5 has been removed).

This module verifies the Python side end-to-end:

* the ``summary`` tool function is a no-op returning a confirmation;
* ``ChatSorcarAgent`` registers the tool;
* ``ChatSorcarAgent`` leaves ``tool_call_guard`` / ``pre_step_hook``
  as the plain inherited attributes — no summary gate, no reminder
  hook, no rejection of other tools;
* the generic ``KISSAgent._execute_tool`` blocked-dispatch path (still
  used by ``SorcarAgent._block_finish_when_user_message_pending``)
  keeps working: a guard-blocked tool is not executed and its printed
  ``tool_result`` event carries ``is_error=True``.
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent, summary
from kiss.core.printer import Printer


class _RecordingPrinter(Printer):
    """Real Printer that records every emitted event for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any, dict[str, Any]]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Record the event and return no streamed text.

        Args:
            content: The content to display.
            type: Content type (e.g. "tool_call", "tool_result").
            **kwargs: Type-specific options (e.g. ``tool_input``).

        Returns:
            An empty string (no streamed text extracted).
        """
        self.events.append((type, content, kwargs))
        return ""

    def token_callback(self, token: str) -> None:
        """Ignore streamed tokens.

        Args:
            token: The text token (unused).
        """

    def reset(self) -> None:
        """Nothing to reset between messages."""


def test_summary_tool_is_noop() -> None:
    """Calling the tool performs no action and confirms."""
    assert summary("Did six things, then six more.") == "Summary recorded."


def test_chat_agent_registers_summary_tool() -> None:
    """ChatSorcarAgent's toolset must include the ``summary`` tool."""
    agent = ChatSorcarAgent("summary-tool-registration")
    agent._use_web_tools = False
    tools = agent._get_tools()
    names = [getattr(t, "__name__", "") for t in tools]
    assert "summary" in names
    tool = tools[names.index("summary")]
    assert tool("one. two. three. four. five.") == "Summary recorded."


def test_no_hardwired_summary_enforcement() -> None:
    """ChatSorcarAgent carries no summary gate or reminder hook.

    The enforcement machinery (``_summary_tool_guard``,
    ``_summary_reminder_hook``, ``_SUMMARY_GATE_REJECTION`` and the
    ``tool_call_guard`` / ``pre_step_hook`` property overrides) was
    removed: a fresh agent exposes the plain inherited attributes with
    their ``None`` defaults, plain assignment round-trips (the class no
    longer intercepts it with a delegating setter), and the module
    exports none of the old enforcement symbols.
    """
    import kiss.agents.sorcar.chat_sorcar_agent as mod

    for symbol in (
        "_SUMMARY_GATE_REJECTION",
        "_summary_tool_guard",
        "_summary_reminder_hook",
    ):
        assert not hasattr(mod, symbol)
        assert not hasattr(ChatSorcarAgent, symbol)
    for attr in ("tool_call_guard", "pre_step_hook"):
        assert not isinstance(getattr(ChatSorcarAgent, attr, None), property)

    agent = ChatSorcarAgent("no-summary-enforcement")
    # A never-run agent has no instance attributes yet (RelentlessAgent
    # assigns them in _reset() at run() time); with the property
    # overrides gone, reads no longer return bound gate methods.
    assert getattr(agent, "tool_call_guard", None) is None
    assert getattr(agent, "pre_step_hook", None) is None
    tmpdir = tempfile.mkdtemp(prefix="kiss_no_enforcement_")
    try:
        agent._reset(None, None, None, None, tmpdir, None)
        assert agent.tool_call_guard is None
        assert agent.pre_step_hook is None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    def guard(name: str, args: dict[str, Any]) -> str | None:
        """Sample guard used to prove plain assignment round-trips.

        Args:
            name: The tool name (unused).
            args: The tool arguments (unused).

        Returns:
            Always ``None`` (allow).
        """
        del name, args
        return None

    agent.tool_call_guard = guard
    assert agent.tool_call_guard is guard
    agent.tool_call_guard = None
    assert agent.tool_call_guard is None


def test_blocked_tool_is_not_executed_and_prints_error() -> None:
    """A guard-blocked tool is not executed; blocked finish not terminal.

    Drives the real ``KISSAgent._execute_tool`` dispatch with a guard
    rejection (the generic mechanism still used by
    ``SorcarAgent._block_finish_when_user_message_pending``): the tool
    function must NOT run, the rejection must be returned as the
    result, and the printed ``tool_result`` event must carry
    ``is_error=True`` so the webview renders the red FAILED panel (a
    plain-string rejection would otherwise be hidden inside a streamed
    Bash panel).
    """
    from kiss.core.kiss_agent import KISSAgent

    printer = _RecordingPrinter()
    executor: Any = KISSAgent("guard-dispatch-executor")
    executor.printer = printer
    executor.verbose = False
    calls: list[str] = []

    def finish(result: str) -> str:
        """Terminal finish stand-in recording invocations.

        Args:
            result: The final result.

        Returns:
            The result unchanged.
        """
        calls.append(result)
        return result

    executor.function_map = {"finish": finish}
    name, response = executor._execute_tool(
        {"name": "finish", "arguments": {"result": "done"}},
        blocked="Error: a queued user message is pending.",
    )
    assert name == "finish"
    assert response == "Error: a queued user message is pending."
    assert calls == [], "a blocked tool must not execute"
    results = [
        (content, kwargs)
        for etype, content, kwargs in printer.events
        if etype == "tool_result"
    ]
    assert len(results) == 1
    assert results[0][0] == "Error: a queued user message is pending."
    assert results[0][1].get("is_error") is True

    name, response = executor._execute_tool(
        {"name": "finish", "arguments": {"result": "done"}}
    )
    assert (name, response) == ("finish", "done")
    assert calls == ["done"]
