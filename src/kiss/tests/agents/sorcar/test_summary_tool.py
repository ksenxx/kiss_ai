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


def test_live_agent_calls_summary_every_5_steps() -> None:
    """A real run calls summary(description=...) after ~5 steps.

    Gives the agent a task whose steps are data-dependent (each Bash
    command needs the previous command's output), forcing one tool
    call per model step so the step counter actually crosses 5, and
    asserts the live model interleaves a ``summary`` tool call with a
    multi-sentence natural-language description.
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
        tool_calls = [
            (content, kwargs.get("tool_input") or {})
            for (etype, content, kwargs) in printer.events
            if etype == "tool_call"
        ]
        summary_calls = [tc for tc in tool_calls if tc[0] == "summary"]
        assert summary_calls, (
            "the agent never called the summary tool; tool calls were: "
            f"{[name for name, _ in tool_calls]}"
        )
        description = str(summary_calls[0][1].get("description", ""))
        assert description.strip(), "summary called without a description"
        sentences = [s for s in description.replace("\n", " ").split(".") if s.strip()]
        assert len(sentences) >= 3, (
            "description must be a natural-language summary of several "
            f"sentences, got: {description!r}"
        )
        # The first summary must come after roughly 5 working steps —
        # i.e. at least 4 non-summary tool calls precede it.
        first_idx = next(
            i for i, (name, _) in enumerate(tool_calls) if name == "summary"
        )
        assert first_idx >= 4, (
            "summary was called too early (after only "
            f"{first_idx} tool calls): {[name for name, _ in tool_calls]}"
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
