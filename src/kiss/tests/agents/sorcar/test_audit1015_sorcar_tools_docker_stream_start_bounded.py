# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real-container regression: the STREAMING docker Bash start is bounded.

Audit 2026-10-15 (sorcar-tools).  ``DockerManager`` ran two separate
``Bash`` engines.  The streaming one (installed ``stream_callback``)
called ``exec_create`` and ``exec_start`` on the task thread *before*
its deadline started, so a daemon that stalled there blocked ``Bash``
past its timeout, and a start delayed past the deadline ran the
command after ``Bash`` had returned — the non-streaming engine had
been fixed for both (committed-start guard + reaper) but the fix never
reached the streaming one.  Both now share one engine; these tests
drive the streaming path through the same latency shims the
non-streaming tests use.  The exec, container and daemon are real.
"""

import threading
import time
import unittest
from typing import Any, cast

from kiss.agents.sorcar.docker_manager import _KILL_WAIT_S, DockerManager
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available
from kiss.tests.agents.sorcar.test_docker_nonstream_timeout_kill import (
    _count_processes_script,
)


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestStreamingStartBounded(unittest.TestCase):
    """The streaming ``Bash`` honours its timeout during exec startup."""

    env: DockerManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("python:3.11-slim")
        cls.env.open()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.streamed: list[str] = []
        self.stop = threading.Event()
        self.env.stream_callback = self.streamed.append

    def tearDown(self) -> None:
        self.env.stream_callback = None

    def _stop_once_output_arrives(self, text: str) -> None:
        """Stream callback that records *text* and fires the task Stop."""
        self.streamed.append(text)
        self.stop.set()

    def _wait_until_no_process(self, needle_head: str, needle_tail: str) -> None:
        """Poll the container until no process matches the needle."""
        script = _count_processes_script(needle_head, needle_tail)
        deadline = time.monotonic() + 20
        count = ""
        while time.monotonic() < deadline:
            count = self.env.Bash(script, "count leftover processes")
            count = count.strip().splitlines()[-1]
            if count == "0":
                return
            time.sleep(0.5)
        self.fail(f"command still running in the container ({count} processes)")

    def test_timeout_while_exec_create_is_delayed_never_runs_command(self) -> None:
        """A delayed ``exec_create`` neither blocks past the deadline nor runs the command."""
        api = cast(Any, self.env.client.api)
        real_exec_create = api.exec_create
        first_call = threading.Event()

        def delayed_exec_create(*args: Any, **kwargs: Any) -> Any:
            if not first_call.is_set():  # delay only the command under test
                first_call.set()
                time.sleep(3)
            return real_exec_create(*args, **kwargs)

        api.exec_create = delayed_exec_create
        try:
            start = time.monotonic()
            result = self.env.Bash("sleep 71", "hang", timeout_seconds=1)
            elapsed = time.monotonic() - start
        finally:
            del api.exec_create
        self.assertLess(elapsed, 3)
        self.assertIn("timed out", result)
        time.sleep(3)  # the shim returns now; a cancelled worker must not start
        self._wait_until_no_process("sle", "ep 71")

    def test_timeout_while_exec_start_is_delayed_still_kills_command(self) -> None:
        """A start delayed past the deadline is killed by the reaper once it exists.

        The start is delayed 6 s: the timeout path may legitimately take
        the 1 s deadline plus up to ``_KILL_WAIT_S`` for the reaper's
        first sweep, so the bound on ``Bash`` is that plus a margin, and
        the delay must exceed the bound for a blocked start to show.
        """
        api = cast(Any, self.env.client.api)
        real_exec_start = api.exec_start
        first_call = threading.Event()

        def delayed_exec_start(*args: Any, **kwargs: Any) -> Any:
            if not first_call.is_set():  # delay only the command under test
                first_call.set()
                time.sleep(6)
            return real_exec_start(*args, **kwargs)

        api.exec_start = delayed_exec_start
        try:
            start = time.monotonic()
            result = self.env.Bash("sleep 73", "hang", timeout_seconds=1)
            elapsed = time.monotonic() - start
        finally:
            del api.exec_start
        self.assertLess(elapsed, 1 + _KILL_WAIT_S + 1)
        self.assertIn("timed out", result)
        time.sleep(6.5)  # only now does the container-side process start
        self._wait_until_no_process("sle", "ep 73")

    def test_output_streams_and_stop_keeps_partial_output(self) -> None:
        """Output reaches the callback live; a task Stop returns what was printed."""
        result = self.env.Bash("echo one; echo two >&2", "small")
        self.assertEqual(result.strip().split(), ["one", "two"])  # stdout, then stderr
        # The live callbacks follow the daemon's frame order, which does
        # not keep stdout ahead of stderr; only the final result does.
        self.assertEqual(sorted("".join(self.streamed).split()), ["one", "two"])

        # The Stop is armed by the first streamed chunk, not by a timer:
        # on a busy daemon the exec may not have started 1 s in, and a
        # Stop landing before ``early`` is printed is a different test.
        self.env.stop_event = self.stop
        self.env.stream_callback = self._stop_once_output_arrives
        try:
            result = self.env.Bash("echo early; sleep 20", "slow", timeout_seconds=60)
        finally:
            self.env.stop_event = None
        self.assertIn("early", result)
        self.assertIn("Killed: the task was stopped.", result)
        self.assertIn("[exit code: -1]", result)
        self._wait_until_no_process("sle", "ep 20")


if __name__ == "__main__":
    unittest.main()
