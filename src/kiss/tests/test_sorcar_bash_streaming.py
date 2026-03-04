"""Tests that SorcarAgent bash streaming works when tools are created before printer is set."""

import queue
import time

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


def _drain(q: queue.Queue) -> list[dict]:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


class TestSorcarBashStreaming:

    def test_stream_callback_noop_when_printer_never_set(self):
        agent = SorcarAgent("test")
        assert agent.printer is None
        tools = agent._get_tools()
        bash_tool = tools[0]

        result = bash_tool(command="echo no_printer", description="test echo")
        assert "no_printer" in result

    def test_multiline_bash_streams_all_lines(self):
        agent = SorcarAgent("test")
        tools = agent._get_tools()
        bash_tool = tools[0]

        printer = BaseBrowserPrinter()
        cq = printer.add_client()
        agent.printer = printer

        result = bash_tool(
            command="printf 'line1\\nline2\\nline3\\n'",
            description="multiline",
        )
        printer._flush_bash()

        assert "line1" in result
        events = _drain(cq)
        sys_text = "".join(e["text"] for e in events if e["type"] == "system_output")
        assert "line1" in sys_text
        assert "line2" in sys_text
        assert "line3" in sys_text
