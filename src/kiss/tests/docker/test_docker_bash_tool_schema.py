# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The docker ``Bash`` tool must expose both of its limits.

:class:`~kiss.agents.sorcar.docker_manager.DockerManager`'s ``Bash``
honours ``timeout_seconds`` and truncates at ``max_output_chars``, but
:meth:`SorcarAgent._get_tools` handed the model a two-parameter shim in
docker mode, and ``RelentlessAgent._docker_bash`` forwarded only those
two arguments.  Both limits were therefore pinned to the manager's
defaults: the model could not raise the 30-second cap for a slow build
nor widen the output slice, although the non-docker
``UsefulTools.Bash`` has always let it do exactly that.

Every test drives a **real** container through the **real** docker
daemon, using the **real** tool object the agent hands to the model.
No mock, patch, fake or test double is used.
"""

from __future__ import annotations

import inspect
import time
import unittest
from collections.abc import Callable
from typing import Any, cast

import docker

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


def is_docker_available() -> bool:
    """Return True when a docker daemon is reachable."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerBashToolExposesBothLimits(unittest.TestCase):
    """The Bash tool the model sees must carry both limit parameters."""

    manager: DockerManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = DockerManager("python:3.11-slim")
        cls.manager.open()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.manager.close()

    def setUp(self) -> None:
        self.agent = SorcarAgent("K2 docker shim")
        # A real docker manager makes _get_tools build the docker
        # toolset; web tools are off so the tool list does not start a
        # browser this test never uses.
        self.agent.docker_manager = self.manager
        self.agent._use_web_tools = False

    def _bash_tool(self) -> Callable[..., Any]:
        """Return the ``Bash`` tool the agent offers the model."""
        for tool in self.agent._get_tools():
            if getattr(tool, "__name__", "") == "Bash":
                return cast("Callable[..., Any]", tool)
        raise AssertionError("the docker toolset offers no Bash tool")

    def test_model_can_raise_the_timeout(self) -> None:
        """A command slower than the default cap must be allowed to run."""
        result = self._bash_tool()(
            "sleep 4; echo finished-after-sleep",
            "outlive the default docker timeout",
            timeout_seconds=40,
        )
        self.assertIn("finished-after-sleep", result)
        self.assertNotIn("timed out", result)

    def test_model_can_lower_the_timeout(self) -> None:
        """A short timeout must actually cut a hung command short."""
        start = time.monotonic()
        result = self._bash_tool()("sleep 60", "hang", timeout_seconds=2)
        elapsed = time.monotonic() - start
        self.assertIn("timed out", result)
        self.assertLess(
            elapsed, 30, f"the tool ignored timeout_seconds ({elapsed:.1f}s)",
        )

    def test_model_can_widen_the_output_cap(self) -> None:
        """``max_output_chars`` must reach the manager's truncation."""
        big = "python -c \"print('x' * 120000)\""
        narrow = self._bash_tool()(big, "big output", max_output_chars=1000)
        self.assertLessEqual(len(narrow), 1000)
        self.assertIn("truncated", narrow)

        wide = self._bash_tool()(big, "big output", max_output_chars=150000)
        self.assertGreater(
            len(wide),
            1000,
            "a widened max_output_chars never reached the docker manager",
        )
        self.assertNotIn("truncated", wide)

    def test_tool_schema_offers_both_limits(self) -> None:
        """The model's tool schema is built from this signature."""
        parameters = inspect.signature(self._bash_tool()).parameters
        self.assertEqual(
            list(parameters),
            ["command", "description", "timeout_seconds", "max_output_chars"],
        )

    def test_docker_bash_without_a_manager_still_raises(self) -> None:
        """The widened forwarder keeps the base class's guard."""
        from kiss.core.kiss_error import KISSError

        agent = SorcarAgent("K2 docker shim, no manager")
        with self.assertRaises(KISSError):
            agent._docker_bash("echo hi", "no manager attached")


if __name__ == "__main__":  # pragma: no cover — manual runs
    unittest.main()
