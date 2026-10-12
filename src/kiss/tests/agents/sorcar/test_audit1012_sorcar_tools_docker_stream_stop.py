# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop must end a running docker ``Bash`` promptly and kill its exec.

Audit 2026-10-12 (sorcar-tools).  ``DockerManager._bash_streaming`` (the
path production uses: ``relentless_agent`` installs a ``stream_callback``
whenever a printer exists) waited on its output queue with a single
``get(timeout=<whole deadline>)``: one blocking C call for the entire
command timeout.  An injected ``ToolCallInterrupted`` (tool-panel Stop) or
``KeyboardInterrupt`` (task Stop) only lands at a bytecode boundary, so a
``sleep 600`` with the default 900 s timeout ignored the Stop for 600 s.
When the injection finally landed, neither docker path killed the
container-side process, so ``bash -c "sleep 20; touch marker"`` went on to
create the marker after the tool had reported itself interrupted.

Every test drives a real container through the real daemon; the Stop is
delivered through the real :mod:`kiss.core.tool_interrupt` machinery.
"""

from __future__ import annotations

import threading
import time
import unittest

import docker

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    begin_tool_call,
    end_tool_call,
    interrupt_tool_call,
    unregister_tool_call,
)

#: Counts container processes whose command line mentions ``sleep 20``,
#: reading /proc directly because the slim image ships no ``ps``.
_COUNT_SLEEP_20 = (
    "count=0\n"
    "for d in /proc/[0-9]*; do\n"
    '  args=$(tr "\\0" " " < "$d/cmdline" 2>/dev/null)\n'
    # The needle is spelled in two pieces so this script's own command
    # line (which is itself a container process) never matches.
    '  case "$args" in *"sle""ep 20"*) count=$((count + 1));; esac\n'
    "done\n"
    "echo $count"
)


def is_docker_available() -> bool:
    """Return whether a Docker daemon answers."""
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerBashStop(unittest.TestCase):
    """Real ``docker exec`` runs; no mocks."""

    env: DockerManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("python:3.11-slim")
        cls.env.open()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def tearDown(self) -> None:
        self.env.stream_callback = None
        self.env.stop_event = None

    def _interrupted_bash(self, marker: str) -> tuple[dict[str, object], float]:
        """Run ``sleep 20; touch marker`` as a tool call and Stop it after 1 s.

        Returns:
            The outcome dict (``{"interrupted": True}`` on success) and
            the seconds the tool call took to return after the Stop.
        """
        outcome: dict[str, object] = {}

        def run() -> None:
            token = begin_tool_call("Bash")
            try:
                try:
                    outcome["result"] = self.env.Bash(
                        f"sleep 20; touch {marker}",
                        "slow",
                        timeout_seconds=60,
                    )
                    end_tool_call(token)
                except ToolCallInterrupted:
                    outcome["interrupted"] = True
            finally:
                unregister_tool_call(token)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(1.0)
        assert thread.ident is not None
        started = time.monotonic()
        self.assertTrue(interrupt_tool_call(thread.ident, "Bash"))
        thread.join(timeout=30)
        elapsed = time.monotonic() - started
        self.assertFalse(thread.is_alive(), "Bash ignored the Stop for 30 s")
        return outcome, elapsed

    def _assert_marker_never_appears(self, marker: str) -> None:
        # The Stop must have killed ``bash -c "sleep 20; touch ..."``
        # inside the container, not just abandoned it.
        time.sleep(1)
        alive = self.env.Bash(_COUNT_SLEEP_20, "count leftover processes")
        self.assertEqual(alive.strip().splitlines()[-1], "0", alive)
        self.assertIn("No such file", self.env.Bash(f"ls {marker}", "check"))

    def test_streaming_bash_stops_promptly_and_kills_exec(self) -> None:
        streamed: list[str] = []
        self.env.stream_callback = streamed.append
        outcome, elapsed = self._interrupted_bash("/tmp/stream-stop-marker")
        self.assertEqual(outcome, {"interrupted": True})
        self.assertLess(elapsed, 10, elapsed)
        self._assert_marker_never_appears("/tmp/stream-stop-marker")

    def test_non_streaming_bash_stops_promptly_and_kills_exec(self) -> None:
        outcome, elapsed = self._interrupted_bash("/tmp/plain-stop-marker")
        self.assertEqual(outcome, {"interrupted": True})
        self.assertLess(elapsed, 10, elapsed)
        self._assert_marker_never_appears("/tmp/plain-stop-marker")

    def test_streaming_bash_honours_task_stop_event(self) -> None:
        streamed: list[str] = []
        self.env.stream_callback = streamed.append
        stop = threading.Event()
        self.env.stop_event = stop
        threading.Timer(1.0, stop.set).start()
        started = time.monotonic()
        result = self.env.Bash("sleep 20; touch /tmp/event-stop-marker", "slow", timeout_seconds=60)
        elapsed = time.monotonic() - started
        self.env.stop_event = None
        self.assertLess(elapsed, 10, elapsed)
        self.assertIn("Killed: the task was stopped.", result)
        self._assert_marker_never_appears("/tmp/event-stop-marker")

    def test_non_streaming_bash_honours_task_stop_event(self) -> None:
        stop = threading.Event()
        self.env.stop_event = stop
        threading.Timer(1.0, stop.set).start()
        started = time.monotonic()
        result = self.env.Bash(
            "sleep 20; touch /tmp/event-stop-marker2",
            "slow",
            timeout_seconds=60,
        )
        elapsed = time.monotonic() - started
        self.env.stop_event = None
        self.assertLess(elapsed, 10, elapsed)
        self.assertIn("Killed: the task was stopped.", result)
        self._assert_marker_never_appears("/tmp/event-stop-marker2")


if __name__ == "__main__":
    unittest.main()
