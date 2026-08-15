# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real-container regression: the NON-streaming docker Bash timeout kill.

The streaming path of :meth:`DockerManager.Bash` has always killed a
timed-out command inside the container (token-tagged exec +
``_kill_exec``), and its docstring calls the alternative out as a bug:
"Without this the hung command keeps running (and holding the stream
open) for the rest of the container's life."  The non-streaming path —
used whenever no ``stream_callback`` is installed — had the exact same
flaw: on timeout it returned the error but left the command running in
the container forever.

These tests drive a real container through the real docker daemon; no
mocks, patches or doubles.
"""

import threading
import time
import unittest
from typing import Any, cast

import docker

from kiss.agents.sorcar.docker_manager import DockerManager


def is_docker_available() -> bool:
    """Return True when a docker daemon is reachable."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def _count_processes_script(needle_head: str, needle_tail: str) -> str:
    """Return a shell script counting container processes by command line.

    Reads /proc directly because the slim image ships no ``ps``.  The
    needle is passed in two pieces and spliced inside the script so the
    script's own command line (which is itself a container process)
    never matches.

    Args:
        needle_head: First half of the command-line needle.
        needle_tail: Second half of the command-line needle.

    Returns:
        A script printing the number of matching processes.
    """
    return (
        "count=0\n"
        "for d in /proc/[0-9]*; do\n"
        '  args=$(tr "\\0" " " < "$d/cmdline" 2>/dev/null)\n'
        f'  case "$args" in *"{needle_head}""{needle_tail}"*) count=$((count + 1));; esac\n'
        "done\n"
        "echo $count"
    )


_COUNT_SLEEP_53 = _count_processes_script("sle", "ep 53")


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestNonStreamingTimeoutKillsCommand(unittest.TestCase):
    """A timed-out non-streaming command must die inside the container."""

    env: DockerManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("python:3.11-slim")
        cls.env.open()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_timeout_does_not_leave_the_command_running(self) -> None:
        """The hung command's process tree is killed on timeout."""
        start = time.monotonic()
        result = self.env.Bash("sleep 53", "hang", timeout_seconds=2)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 15)
        self.assertIn("timed out", result)
        time.sleep(1)
        alive = self.env.Bash(_COUNT_SLEEP_53, "count leftover processes")
        self.assertEqual(alive.strip().splitlines()[-1], "0")

    def test_output_and_exit_code_still_reported(self) -> None:
        """The rewritten exec path keeps output and exit-code parity."""
        self.assertEqual(self.env.Bash("echo hello", "small").strip(), "hello")
        result = self.env.Bash("echo boom >&2; exit 3", "fail")
        self.assertIn("boom", result)
        self.assertIn("[exit code: 3]", result)

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
        """A command whose creation the daemon delays past the deadline never runs.

        The single kill scan of the old implementation ran before the
        delayed exec existed, so the command started — and ran forever —
        *after* Bash had already returned the timeout error.  Now the
        worker refuses to start a cancelled exec at all.  The delay is a
        thin latency shim in front of the real ``exec_create``; the
        exec, container and daemon are all real.
        """
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
            result = self.env.Bash("sleep 61", "hang", timeout_seconds=1)
            elapsed = time.monotonic() - start
        finally:
            del api.exec_create
        self.assertLess(elapsed, 3)
        self.assertIn("timed out", result)
        # Wait out the shim so the worker's exec_create has returned and
        # the cancelled worker has had every chance to (wrongly) start.
        time.sleep(3)
        self._wait_until_no_process("sle", "ep 61")

    def test_timeout_while_exec_start_is_delayed_still_kills_command(self) -> None:
        """A command whose start the daemon delays past the deadline is killed.

        Here the worker has already committed to ``exec_start`` when the
        deadline expires, so the process starts *after* Bash returned;
        the reaper must kill it once it exists.  The delay is a thin
        latency shim in front of the real ``exec_start``.
        """
        api = cast(Any, self.env.client.api)
        real_exec_start = api.exec_start
        first_call = threading.Event()

        def delayed_exec_start(*args: Any, **kwargs: Any) -> Any:
            if not first_call.is_set():  # delay only the command under test
                first_call.set()
                time.sleep(3)
            return real_exec_start(*args, **kwargs)

        api.exec_start = delayed_exec_start
        try:
            start = time.monotonic()
            result = self.env.Bash("sleep 67", "hang", timeout_seconds=1)
            elapsed = time.monotonic() - start
        finally:
            del api.exec_start
        self.assertLess(elapsed, 3)
        self.assertIn("timed out", result)
        # Wait out the shim: only now does the container-side process
        # actually start, well after Bash returned.
        time.sleep(3.5)
        self._wait_until_no_process("sle", "ep 67")


if __name__ == "__main__":
    unittest.main()
