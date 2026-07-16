# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests pinning the prompt/tool contract fixes in core (no mocks).

Covers:
- The summarizer prompt in relentless_agent instructs a ``finish(...)``
  call that the summarizer's actually-registered finish tool accepts
  (the summarizer registers ``KISSAgent.finish(result: str)`` because
  its tool list has no tool named "finish").
- ``kiss.core.utils.finish`` emits the success/is_continue/summary
  contract that ``kiss.core.printer.parse_result_yaml`` recognizes.
"""

import inspect
import re
import subprocess
import sys
import unittest
from collections.abc import Callable
from typing import Any

import yaml

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.kiss_agent import KISSAgent
from kiss.core.printer import parse_result_yaml
from kiss.core.relentless_agent import SUMMARIZER_PROMPT
from kiss.core.relentless_agent import finish as relentless_finish
from kiss.core.utils import finish as utils_finish


def _build_summarizer_registry() -> KISSAgent:
    """Build a KISSAgent tool registry exactly as the summarizer session does.

    Mirrors ``RelentlessAgent.perform_task``'s summarizer wiring
    (``tools=[shell_tools.Read, shell_tools.Bash]``) followed by
    ``KISSAgent._setup_tools``'s fallback: when no supplied tool is named
    "finish", the agent's own ``finish`` method is appended.

    Returns:
        The KISSAgent with its ``function_map`` populated.
    """
    agent = KISSAgent("summarizer registry test")
    agent.function_map = {}
    shell_tools = UsefulTools()
    tools: list[Callable[..., Any]] = [shell_tools.Read, shell_tools.Bash]
    tool_names = {getattr(tool, "__name__", None) for tool in tools}
    if "finish" not in tool_names:  # same fallback as KISSAgent._setup_tools
        tools.append(agent.finish)
    agent._add_functions(tools)
    return agent


class SummarizerFinishContract(unittest.TestCase):
    def test_prompt_instructs_kwargs_accepted_by_registered_finish(self) -> None:
        """The finish(...) kwargs named in SUMMARIZER_PROMPT bind to the real tool."""
        agent = _build_summarizer_registry()
        finish_tool = agent.function_map["finish"]

        matches = re.findall(r"finish\(\s*(\w+)\s*=", SUMMARIZER_PROMPT)
        self.assertTrue(
            matches, "SUMMARIZER_PROMPT must instruct a keyword finish(...) call"
        )
        sig = inspect.signature(finish_tool)
        for kwarg in matches:
            sig.bind(**{kwarg: "detailed summary of work done so far"})

    def test_execute_tool_with_prompt_instructed_call_succeeds(self) -> None:
        """Executing finish exactly as the prompt instructs returns the summary."""
        agent = _build_summarizer_registry()
        name, response = agent._execute_tool(
            {
                "name": "finish",
                "arguments": {"result": "detailed summary of work done so far"},
            }
        )
        self.assertEqual(name, "finish")
        self.assertEqual(response, "detailed summary of work done so far")
        # relentless_agent parses the summarizer result via yaml.safe_load and
        # uses it verbatim when it is not a dict — a plain string round-trips.
        parsed = yaml.safe_load(response)
        self.assertNotIsInstance(parsed, dict)

    def test_old_success_summary_call_does_not_bind(self) -> None:
        """The pre-fix prompt call finish(success=..., summary=...) is invalid."""
        agent = _build_summarizer_registry()
        finish_tool = agent.function_map["finish"]
        sig = inspect.signature(finish_tool)
        with self.assertRaises(TypeError):
            sig.bind(success=True, summary="detailed summary")


class UtilsFinishContract(unittest.TestCase):
    def test_relentless_reuses_canonical_finish(self) -> None:
        """Core exposes one finish implementation, not two drifting copies."""
        self.assertIs(relentless_finish, utils_finish)

    def test_utils_finish_recognized_by_parse_result_yaml(self) -> None:
        raw = utils_finish(True, False, "the final code")
        parsed = parse_result_yaml(raw)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(
            parsed,
            {"success": True, "is_continue": False, "summary": "the final code"},
        )


class FreshImportContract(unittest.TestCase):
    def test_relentless_agent_imports_in_fresh_interpreter(self) -> None:
        """Direct core import must not cycle through sorcar.__init__."""
        completed = subprocess.run(
            [sys.executable, "-c", "import kiss.core.relentless_agent"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
