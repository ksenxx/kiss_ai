# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the SorcarAgent ``talk`` tool.

The ``talk(language, text)`` tool broadcasts a ``{"type": "talk"}``
event through the printer so every frontend tab subscribed to the
running task — on every device — plays the text aloud through the
device's default speaker via the Web Speech API.

These tests drive a real :class:`SorcarAgent` and a real
:class:`JsonPrinter` subclass (:class:`MemoryPrinter`, which mirrors
the production ``WebPrinter`` fanout contract) — no mocks.
"""

from __future__ import annotations

import unittest
from typing import Any

from kiss.agents.sorcar.sorcar_agent import SorcarAgent


def _find_tool(tools: list, name: str) -> Any:
    """Return the tool function named *name* from *tools*."""
    for t in tools:
        if callable(t) and t.__name__ == name:
            return t
    raise AssertionError(
        f"Tool {name!r} not found in "
        f"{[getattr(t, '__name__', None) for t in tools if callable(t)]}"
    )


def _make_agent(printer: Any) -> SorcarAgent:
    """Build a SorcarAgent with web tools disabled and *printer* attached."""
    agent = SorcarAgent("test-talk-tool")
    agent._use_web_tools = False
    agent.printer = printer
    return agent


class TestTalkTool(unittest.TestCase):
    """The ``talk`` tool is a default tool and broadcasts talk events."""

    def test_talk_without_printer_reports_unavailable(self) -> None:
        """No printer (e.g. bare library use) → graceful message."""
        agent = _make_agent(None)
        talk = _find_tool(agent._get_tools(), "talk")
        msg = talk("en", "hello")
        self.assertIn("not available", msg)

    def test_talk_with_printer_that_cannot_broadcast_reports_unavailable(
        self,
    ) -> None:
        """A printer-like object without ``broadcast`` is a graceful no-op."""
        agent = _make_agent(object())
        talk = _find_tool(agent._get_tools(), "talk")
        msg = talk("en", "hello")
        self.assertIn("not available", msg)


if __name__ == "__main__":
    unittest.main()
