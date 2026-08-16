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
