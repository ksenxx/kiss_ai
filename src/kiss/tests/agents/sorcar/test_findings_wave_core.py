# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests pinning the prompt/tool contract fixes in core (no mocks).

Covers:
- The summarizer prompt in relentless_agent instructs a ``finish(...)``
  call that the summarizer's actually-registered finish tool accepts
  (the summarizer registers ``SummarizerFinish.finish(result: str)``,
  which returns the structured continuation result).
- ``relentless_agent`` reuses the canonical ``kiss.core.utils.finish``
  (whose ``parse_result_yaml`` contract is pinned in
  ``kiss.tests.core.test_findings_wave_core``).
"""

import inspect
import re
import subprocess
import sys
import unittest
from collections.abc import Callable
from typing import Any

import yaml

from kiss.agents.sorcar.relentless_agent import SUMMARIZER_PROMPT, SummarizerFinish
from kiss.agents.sorcar.relentless_agent import finish as relentless_finish
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.kiss_agent import KISSAgent
from kiss.core.utils import finish as utils_finish


def _build_summarizer_registry() -> KISSAgent:
    """Build a KISSAgent tool registry exactly as the summarizer session does.

    Mirrors ``RelentlessAgent._summarize_failed_session``'s summarizer
    wiring: ``SummarizerFinish.finish`` is named "finish", so
    ``KISSAgent._setup_tools`` registers it instead of the built-in.

    Returns:
        The KISSAgent with its ``function_map`` populated.
    """
    agent = KISSAgent("summarizer registry test")
    agent.function_map = {}
    shell_tools = UsefulTools()
    tools: list[Callable[..., Any]] = [
        shell_tools.Read,
        shell_tools.Bash,
        SummarizerFinish.finish,
    ]
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
        """Executing finish as the prompt instructs yields the continuation result."""
        agent = _build_summarizer_registry()
        name, response = agent._execute_tool(
            {
                "name": "finish",
                "arguments": {"result": "<p>detailed summary of work done so far</p>"},
            }
        )
        self.assertEqual(name, "finish")
        parsed = yaml.safe_load(response)
        self.assertEqual(
            parsed,
            {
                "success": False,
                "is_continue": True,
                "summary": "<p>detailed summary of work done so far</p>",
            },
        )

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


class FreshImportContract(unittest.TestCase):
    def test_relentless_agent_imports_in_fresh_interpreter(self) -> None:
        """Direct core import must not cycle through sorcar.__init__."""
        completed = subprocess.run(
            [sys.executable, "-c", "import kiss.agents.sorcar.relentless_agent"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
