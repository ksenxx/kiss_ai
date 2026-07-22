# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ChatSorcarAgent's no-op ``summary`` tool.

Feature: after every 5 steps the agent summarizes what it did in the
last 6 steps and calls ``summary(description=...)``.  The tool itself
does nothing (the chat webview reacts to the ``tool_call`` event by
nesting and collapsing the preceding panels — covered by the jsdom
suite in ``src/kiss/agents/vscode/test/summaryToolCollapse.test.js``).

This module verifies the Python side end-to-end:

* the ``summary`` tool function is a no-op returning a confirmation;
* ``ChatSorcarAgent`` registers the tool;
* a REAL agent run against a live LLM actually calls
  ``summary(description=...)`` after ~5 steps, proving the SYSTEM.md
  instruction works (no mocks — this talks to the real model).
"""

from __future__ import annotations

import re
import shutil
import tempfile
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent, summary
from kiss.core.printer import Printer

LIVE_MODEL = "claude-fable-5"


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


def test_summary_guard_rejects_every_tool_until_summary() -> None:
    """While a summary is due, every other tool call is blocked.

    Exercises ``ChatSorcarAgent.tool_call_guard`` (the real
    :meth:`_summary_tool_guard`) against a real executor object: with
    the gate armed, Bash / finish / arbitrary caller tools are all
    blocked with the instructive rejection; a ``summary`` call clears
    the gate; and the gate keeps rejecting indefinitely (no escape
    hatch) until summary is called.
    """
    from kiss.agents.sorcar.chat_sorcar_agent import _SUMMARY_GATE_REJECTION
    from kiss.core.kiss_agent import KISSAgent

    agent = ChatSorcarAgent("summary-guard-verify")
    guard = agent.tool_call_guard
    # ``Any``-typed on purpose: the gate stores its ``_summary_due``
    # state as a dynamic attribute on the per-session executor (same
    # pattern as ``_summary_reminder_step``).
    executor: Any = KISSAgent("summary-guard-executor")

    # No executor yet: everything is allowed.
    assert guard("Bash", {"command": "ls"}) is None

    agent._current_executor = executor
    # Gate not armed: everything is allowed.
    assert guard("Bash", {"command": "ls"}) is None
    assert guard("finish", {"result": "done"}) is None

    # Armed: every tool — including finish and custom tools — is
    # blocked; the gate never opens by itself.
    executor._summary_due = True
    for _ in range(5):
        assert guard("Bash", {"command": "ls"}) == _SUMMARY_GATE_REJECTION
        assert guard("finish", {"result": "done"}) == _SUMMARY_GATE_REJECTION
        assert guard("my_custom_tool", {}) == _SUMMARY_GATE_REJECTION
    assert executor._summary_due is True

    # A summary call clears the gate; subsequent calls run again.
    assert guard("summary", {"description": "one. two. three."}) is None
    assert executor._summary_due is False
    assert guard("Bash", {"command": "ls"}) is None
    assert guard("finish", {"result": "done"}) is None


def test_blocked_finish_is_not_terminal_and_prints_error() -> None:
    """A guard-blocked tool is not executed; blocked finish not terminal.

    Drives the real ``KISSAgent._execute_tool`` dispatch with a guard
    rejection: the tool function must NOT run, the rejection must be
    returned as the result, and the printed ``tool_result`` event must
    carry ``is_error=True`` so the webview renders the red FAILED
    panel (a plain-string rejection would otherwise be hidden inside a
    streamed Bash panel).
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
        blocked="Error: call summary first.",
    )
    assert name == "finish"
    assert response == "Error: call summary first."
    assert calls == [], "a blocked tool must not execute"
    results = [
        (content, kwargs)
        for etype, content, kwargs in printer.events
        if etype == "tool_result"
    ]
    assert len(results) == 1
    assert results[0][0] == "Error: call summary first."
    assert results[0][1].get("is_error") is True

    # Unblocked, the same call executes and returns normally.
    name, response = executor._execute_tool(
        {"name": "finish", "arguments": {"result": "done"}}
    )
    assert (name, response) == ("finish", "done")
    assert calls == ["done"]


def _tool_calls_by_step(
    events: list[tuple[str, Any, dict[str, Any]]],
) -> list[tuple[int, str]]:
    """Attribute recorded tool_call events to executor step numbers.

    ``KISSAgent._execute_step`` prints the step's ``usage_info`` event
    (carrying ``total_steps``) right after ``generate`` and BEFORE the
    step's ``tool_call`` events, so every ``tool_call`` belongs to the
    most recently seen ``total_steps``.

    Args:
        events: The (type, content, kwargs) tuples recorded by the
            printer.

    Returns:
        Ordered (step, tool_name) pairs.
    """
    step = 0
    calls: list[tuple[int, str]] = []
    for etype, content, kwargs in events:
        if etype == "usage_info":
            step = int(kwargs.get("total_steps", step) or step)
        elif etype == "tool_call":
            calls.append((step, str(content)))
    return calls


def test_live_agent_calls_summary_on_every_step_divisible_by_5() -> None:
    """A real run calls summary exactly on steps 5, 10, 15, ....

    Gives the agent a task whose steps are data-dependent (each Bash
    command needs the previous command's output), forcing one tool
    call per model step so the step counter crosses several 5-step
    boundaries, and asserts the live model calls ``summary`` (with a
    multi-sentence natural-language description) on EVERY step
    divisible by 5 — the enforcement gate rejects any other tool call
    on those steps, so compliance is no longer best-effort.
    """
    tmpdir = tempfile.mkdtemp(prefix="kiss_summary_live_")
    printer = _RecordingPrinter()
    agent = ChatSorcarAgent("summary-live-verify")
    try:
        agent.run(
            prompt_template=(
                "Compute a chain of numbers with the Bash tool. Start "
                "with x=3. Do 8 iterations: in each iteration run "
                "`echo $((x * 2 + 1))` where x is the number printed "
                "by your PREVIOUS command, so you MUST wait for each "
                "command's output before issuing the next - exactly "
                "one Bash tool call per turn, never batched, no loops, "
                "never combine iterations. Do not read or write any "
                "files and do not use the internet. After the 8th "
                "iteration, finish with success=True reporting the "
                "final number."
            ),
            model_name=LIVE_MODEL,
            work_dir=tmpdir,
            printer=printer,
            max_budget=8.0,
            max_steps=30,
            web_tools=False,
        )
        calls = _tool_calls_by_step(printer.events)
        assert calls, "no tool calls recorded"
        summary_steps = sorted({s for s, name in calls if name == "summary"})
        assert summary_steps, (
            "the agent never called the summary tool; tool calls were: "
            f"{calls}"
        )
        # Every summary call must land on a step divisible by 5.
        off_boundary = [s for s in summary_steps if s % 5]
        assert not off_boundary, (
            f"summary called on steps not divisible by 5: {off_boundary}; "
            f"all calls: {calls}"
        )
        # Every 5-step boundary reached before the final step must have
        # a summary call (the run may finish before the last boundary).
        max_step = max(s for s, _ in calls)
        expected = list(range(5, max_step + 1, 5))
        missed = [b for b in expected if b not in summary_steps]
        assert not missed, (
            f"summary missing on boundary steps {missed}; all calls: {calls}"
        )
        # The description must be a substantial natural-language summary.
        # A live model phrases its digest nondeterministically — some
        # runs use semicolons, arrows, or bullet fragments instead of
        # period-terminated sentences — so counting "." separators alone
        # is brittle.  Accept any description that has at least two
        # sentence-like segments (split on ., !, ?, ;, or newlines) and
        # is long enough to be a real summary rather than a stub.
        descriptions = [
            str((kwargs.get("tool_input") or {}).get("description", ""))
            for (etype, content, kwargs) in printer.events
            if etype == "tool_call" and content == "summary"
        ]
        for description in descriptions:
            segments = [
                s
                for s in re.split(r"[.!?;\n]", description)
                if s.strip()
            ]
            assert len(segments) >= 2 and len(description) >= 80, (
                "description must be a natural-language summary of several "
                f"sentences, got: {description!r}"
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
