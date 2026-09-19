# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``DockerManager.run_commands_parallel`` and the ``_exec`` refactor, on a real container."""

from __future__ import annotations

import threading
import time
import unittest

import docker

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.kiss_error import KISSError
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    begin_tool_call,
    end_tool_call,
    interrupt_tool_call,
    unregister_tool_call,
)


def is_docker_available() -> bool:
    """Return whether a Docker daemon answers."""
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerRunCommandsParallel(unittest.TestCase):
    """Real ``docker exec`` runs; no mocks."""

    env: DockerManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("python:3.11-slim")
        cls.env.open()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_bash_formats_unchanged(self) -> None:
        self.assertEqual(self.env.Bash("echo hi", "ok").strip(), "hi")
        failed = self.env.Bash("echo boom >&2; exit 3", "fail")
        self.assertIn("boom", failed)
        self.assertIn("[exit code: 3]", failed)
        self.assertEqual(
            self.env.Bash("sleep 5", "slow", timeout_seconds=1),
            "Error: command timed out after 1s",
        )

    def test_report_in_input_order_and_concurrent(self) -> None:
        started = time.monotonic()
        out = self.env.run_commands_parallel(
            '["sleep 1; echo alpha", "sleep 1; echo beta >&2; exit 3", "sleep 5", "sleep 1"]',
            timeout_seconds=2,
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 8.0, elapsed)
        self.assertTrue(
            out.startswith("4 commands: 2 succeeded, 1 failed, 1 timed out."), out,
        )
        self.assertLess(out.index("[1/4] exit 0"), out.index("[2/4] exit 3"))
        self.assertLess(out.index("[2/4] exit 3"), out.index("[3/4] TIMED OUT"))
        self.assertLess(out.index("[3/4] TIMED OUT"), out.index("[4/4] exit 0"))
        self.assertIn("$ sleep 1; echo alpha\nalpha", out)
        self.assertIn("beta", out)

    def test_max_workers_serializes(self) -> None:
        started = time.monotonic()
        out = self.env.run_commands_parallel('["sleep 1", "sleep 1"]', max_workers=1)
        self.assertGreaterEqual(time.monotonic() - started, 2.0)
        self.assertTrue(out.startswith("2 commands: 2 succeeded"), out)

    def test_invalid_arguments(self) -> None:
        self.assertTrue(
            self.env.run_commands_parallel("$(cat cmds.json)").startswith(
                "Error: commands must be a JSON array of strings"
            )
        )
        self.assertEqual(
            self.env.run_commands_parallel('["echo x"]', max_workers=-1),
            "Error: max_workers must be 0 or a positive integer, got -1.",
        )

    def test_exec_failure_is_reported_not_raised(self) -> None:
        """A closed container fails every exec; the report carries the error."""
        other = DockerManager("python:3.11-slim")
        other.open()
        container = other.container
        assert container is not None
        other.close()
        exit_code, output, _seconds = other._timed_exec(
            container, "echo x", threading.Event(), 5,
        )
        self.assertEqual(exit_code, -1)
        self.assertTrue(output.startswith("Error: "), output)
        with self.assertRaises(KISSError):
            other.run_commands_parallel('["echo x"]')

    def test_task_stop_kills_running_and_skips_queued(self) -> None:
        stop = threading.Event()
        self.env.stop_event = stop
        try:
            threading.Timer(1.0, stop.set).start()
            started = time.monotonic()
            out = self.env.run_commands_parallel(
                '["sleep 20; touch /tmp/stop-marker", "sleep 20", "touch /tmp/stop-marker"]',
                max_workers=2,
            )
            elapsed = time.monotonic() - started
        finally:
            self.env.stop_event = None
        self.assertLess(elapsed, 15, elapsed)
        self.assertTrue(out.startswith("3 commands: 0 succeeded, 3 failed, 0 timed out."), out)
        self.assertEqual(out.count("Killed: the task was stopped."), 2, out)
        self.assertEqual(out.count("Not started: the task was stopped."), 1, out)
        time.sleep(1)
        self.assertIn("No such file", self.env.Bash("ls /tmp/stop-marker", "check"))
        # The container-side sleeps were reaped, not left running.
        self.assertNotIn("sleep 20", self.env.Bash("ps -eo args", "ps"))

    def test_tool_panel_interrupt_raises_and_kills(self) -> None:
        outcome: dict[str, object] = {}

        def run() -> None:
            token = begin_tool_call("run_commands_parallel")
            try:
                try:
                    outcome["result"] = self.env.run_commands_parallel(
                        '["sleep 20; touch /tmp/int-marker", "sleep 20; touch /tmp/int-marker"]'
                    )
                    end_tool_call(token)
                except ToolCallInterrupted:
                    outcome["interrupted"] = True
            finally:
                unregister_tool_call(token)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(1.5)
        assert thread.ident is not None
        self.assertTrue(interrupt_tool_call(thread.ident, "run_commands_parallel"))
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, {"interrupted": True})
        time.sleep(1)
        self.assertIn("No such file", self.env.Bash("ls /tmp/int-marker", "check"))

    def test_registered_in_docker_tool_list(self) -> None:
        agent = SorcarAgent("docker-tool-list")
        agent.docker_manager = self.env
        agent._use_web_tools = False
        names = [t.__name__ for t in agent._get_tools()]
        self.assertIn("Bash", names)
        self.assertIn("run_commands_parallel", names)


if __name__ == "__main__":
    unittest.main()
