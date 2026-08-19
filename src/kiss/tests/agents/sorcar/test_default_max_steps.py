# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the default step budget of the sorcar agents is 10000.

When a caller does not pass ``max_steps``, the auto-continuing
:class:`RelentlessAgent` / :class:`SorcarAgent` family must allow 10000
steps per session, and the per-step usage banner handed to the model must
advertise that same number.  The plain :class:`KISSAgent` half of this
contract lives in ``kiss.tests.core.test_default_max_steps``, which owns
the shared ``_ScriptedServer`` harness imported below.
"""

from __future__ import annotations

import tempfile
import unittest

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.core.test_default_max_steps import (
    DEFAULT_MAX_STEPS,
    _all_text,
    _ScriptedServer,
    ping,
)


class TestDefaultMaxSteps(unittest.TestCase):
    """The resolved default of ``max_steps`` is 10000 for the sorcar agents."""

    def test_relentless_agent_default_and_banner(self) -> None:
        """``RelentlessAgent`` defaults to 10000 and tells the model so."""
        server = _ScriptedServer(
            [
                ("ping", {}),
                (
                    "finish",
                    {
                        "success": True,
                        "is_continue": False,
                        "summary_in_html": "<p>done</p>",
                    },
                ),
            ]
        )
        try:
            agent = RelentlessAgent("DefaultStepsRelentless")
            with tempfile.TemporaryDirectory() as work_dir:
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Call ping, then finish.",
                    tools=[ping],
                    work_dir=work_dir,
                    model_config=server.model_config(),
                    verbose=False,
                )
            self.assertEqual(agent.max_steps, DEFAULT_MAX_STEPS)
            self.assertIn(
                f"Steps: 1/{DEFAULT_MAX_STEPS}", _all_text(server.request_bodies)
            )
        finally:
            server.stop()

    def test_sorcar_agent_default(self) -> None:
        """``SorcarAgent`` inherits the 10000-step default."""
        server = _ScriptedServer(
            [
                (
                    "finish",
                    {
                        "success": True,
                        "is_continue": False,
                        "summary_in_html": "<p>done</p>",
                    },
                )
            ]
        )
        try:
            agent = SorcarAgent("DefaultStepsSorcar")
            with tempfile.TemporaryDirectory() as work_dir:
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Call finish immediately.",
                    work_dir=work_dir,
                    model_config=server.model_config(),
                    web_tools=False,
                    verbose=False,
                )
            self.assertEqual(agent.max_steps, DEFAULT_MAX_STEPS)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
