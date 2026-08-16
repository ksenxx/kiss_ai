# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ChatSorcarAgent's no-op ``summary`` tool.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.agents.sorcar.test_summary_tool``; the non-core tests remain there.
"""

from __future__ import annotations

from typing import Any

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
