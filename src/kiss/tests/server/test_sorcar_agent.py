# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bash-streaming test extracted from
``kiss.tests.agents.third_party_agents.test_sorcar_agent``.

Moved here because its dependency closure is kiss.agents.sorcar (the
agent and its bash tool) + kiss.server (JsonPrinter) only.  Building
the default toolset does list the third-party package directory (the
``run_agent`` tool's docstring is expanded via ``available_channels()``,
which reads filenames only and imports no channel module); per the
series adjudication that incidental listing is an implementation
detail of the sorcar toolset, not a rule-6 third-party dependency —
the same boundary applied to every ``_get_tools()``-calling test in
tests/server and tests/agents/sorcar since Task 7.
"""

from __future__ import annotations

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.json_printer import JsonPrinter


class TestSorcarBashStreaming:
    def test_multiline_bash_streams_all_lines(self):
        agent = SorcarAgent("test")
        tools = agent._get_tools()
        bash_tool = tools[0]

        printer = JsonPrinter()
        printer._thread_local.task_id = "0"
        printer.start_recording()
        agent.printer = printer

        result = bash_tool(
            command="printf 'line1\\nline2\\nline3\\n'",
            description="multiline",
        )
        printer._flush_bash()

        assert "line1" in result
        events = printer.stop_recording()
        sys_text = "".join(e["text"] for e in events if e["type"] == "system_output")
        assert "line1" in sys_text
        assert "line2" in sys_text
        assert "line3" in sys_text
