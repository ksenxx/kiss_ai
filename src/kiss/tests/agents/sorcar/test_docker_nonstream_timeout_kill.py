# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real-container regression: a timed-out docker ``Bash`` kills its command.

:meth:`DockerManager.Bash` once left a timed-out command running in the
container forever when no ``stream_callback`` was installed (only the
streaming variant killed it).  There is a single exec engine now, so
these tests pin the plain (no callback) call; the exec-startup latency
cases live in ``test_audit1015_sorcar_tools_docker_stream_start_bounded``
and cover both forms of the call.

These tests drive a real container through the real docker daemon; no
mocks, patches or doubles.
"""

import time
import unittest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available


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
    """A timed-out command run without a stream callback dies in the container."""

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


if __name__ == "__main__":
    unittest.main()
