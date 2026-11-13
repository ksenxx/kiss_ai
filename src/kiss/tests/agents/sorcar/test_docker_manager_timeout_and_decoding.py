# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real-container regressions for the docker Bash timeout and decoding:
:class:`~kiss.agents.sorcar.docker_manager.DockerManager`.

F3 (a) ``Bash`` forwarded to ``_bash_streaming`` without ``timeout_seconds``,
so the documented timeout was dead code on the only path production uses
(``relentless_agent`` installs a ``stream_callback`` whenever a printer
exists).  (b) Neither docker path truncated its output, unlike
``UsefulTools.Bash``'s ``max_output_chars``.

F4 every decode was strict UTF-8, so binary output raised
``UnicodeDecodeError`` *out of the tool*, and even valid UTF-8 split across
two docker stream frames raised nondeterministically.

Audit 2026-11-07 (A08 claude-2): the docker ``Bash`` keeps a command's
output bounded in memory while it streams, as the host ``Bash`` does
(head and tail once it passes twice ``_CAPTURE_CHARS``), and the
truncation marker still counts the whole output.

Audit 2026-11-10 (B/S4-F3): output the worker read ahead of a slow stream
callback waits in a bounded queue, the command is killed at the deadline or
a Stop before that queue is taken, and no callback runs after a Stop.

Every test here drives a real container through the real docker daemon; no
mocks, patches or doubles.
"""

import threading
import time
import tracemalloc
import unittest

from kiss.agents.sorcar.coding_gate import bash_exit_code
from kiss.agents.sorcar.docker_manager import STOPPED_MESSAGE, DockerManager
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available
from kiss.tests.conftest import wait_for

#: 20 MB of output at once (over 600 stream frames), then a wait: a command that has
#: printed far ahead of a slow reader and is still running at any deadline.
_FLOOD = "head -c 20000000 /dev/zero | tr '\\0' a; sleep 30"

#: Python source that emits the two bytes of "é" in two separate writes,
#: each flushed and separated by a sleep, so the docker exec stream
#: delivers them in two distinct frames.
_SPLIT_MULTIBYTE = (
    'python -c "import sys, time\n'
    "raw = chr(233).encode()\n"
    "sys.stdout.buffer.write(raw[:1]); sys.stdout.buffer.flush()\n"
    "time.sleep(0.6)\n"
    'sys.stdout.buffer.write(raw[1:]); sys.stdout.buffer.flush()"'
)

#: Counts container processes whose command line mentions ``sleep 47``,
#: reading /proc directly because the slim image ships no ``ps``.
_COUNT_SLEEP_47 = (
    "count=0\n"
    "for d in /proc/[0-9]*; do\n"
    '  args=$(tr "\\0" " " < "$d/cmdline" 2>/dev/null)\n'
    # The needle is spelled in two pieces so this script's own command
    # line (which is itself a container process) never matches.
    '  case "$args" in *"sle""ep 47"*) count=$((count + 1));; esac\n'
    "done\n"
    "echo $count"
)

#: Python source that emits invalid UTF-8 followed by readable text.
_BINARY_OUTPUT = (
    'python -c "import sys\n'
    'sys.stdout.buffer.write(b\'\\xff\\xfe hello\')"'
)


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerManagerTimeoutAndTruncation(unittest.TestCase):
    """F3 — the streaming path must honour timeouts and truncate output."""

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

    def test_streaming_bash_honours_timeout(self) -> None:
        """A hung command must return the timeout error, not block forever."""
        chunks: list[str] = []
        self.env.stream_callback = chunks.append
        start = time.monotonic()
        result = self.env.Bash("sleep 30", "hang", timeout_seconds=2)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 15, f"streaming Bash ignored the timeout: {elapsed}s")
        self.assertIn("timed out", result)

    def test_non_streaming_bash_honours_timeout(self) -> None:
        """The printer-less path keeps its existing timeout behaviour."""
        start = time.monotonic()
        result = self.env.Bash("sleep 30", "hang", timeout_seconds=2)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 15)
        self.assertIn("timed out", result)

    def test_streaming_output_is_truncated(self) -> None:
        """200 000 chars of output must come back truncated, as UsefulTools does."""
        chunks: list[str] = []
        self.env.stream_callback = chunks.append
        result = self.env.Bash(
            "python -c \"print('x' * 200000)\"", "big", max_output_chars=50000,
        )
        self.assertLessEqual(len(result), 50000)
        self.assertIn("truncated", result)
        self.assertTrue("".join(chunks), "live streaming must still emit output")

    def test_non_streaming_output_is_truncated(self) -> None:
        """The printer-less path truncates too."""
        result = self.env.Bash(
            "python -c \"print('x' * 200000)\"", "big", max_output_chars=50000,
        )
        self.assertLessEqual(len(result), 50000)
        self.assertIn("truncated", result)

    def test_short_output_is_not_truncated(self) -> None:
        """Output under the cap is returned verbatim (no marker)."""
        result = self.env.Bash("echo hello", "small")
        self.assertEqual(result.strip(), "hello")

    def test_the_status_of_a_command_survives_any_output_budget(self) -> None:
        """A small ``max_output_chars`` cuts the output, never the status that follows it.

        The ``[exit code: N]`` line and the timeout sentence were cut with
        the output, so a failed command read as one that passed (cross
        finding X7 of 2026-11-12; the exec bridge's twin is in
        ``test_exec_bridge_until.py``).
        """
        for budget in (1, 20, 59, 200):
            result = self.env.Bash("seq 1 200; exit 7", "fails", max_output_chars=budget)
            self.assertTrue(result.endswith("\n[exit code: 7]"), (budget, result))
            self.assertEqual(bash_exit_code(result, container=True), "7", (budget, result))
        result = self.env.Bash(
            "seq 1 200; exit 7", "polled", timeout_seconds=2, max_output_chars=5, until="NEVER",
            poll_seconds=1,
        )
        self.assertIn("\n[exit code: 7]\n[until 'NEVER': not matched after ", result)
        result = self.env.Bash(
            "seq 1 200; sleep 30", "hangs", timeout_seconds=2, max_output_chars=5,
        )
        self.assertTrue(result.startswith("Error: command timed out after 2"), result)
        self.assertEqual(bash_exit_code(result, container=True), "timeout", result)
        # The status is data too (``run_bash``, which the agent's tool hands to the coding
        # gate): in the text, a line the command printed read as the tool's (round 2, B2 #9).
        self.assertEqual(self.env.run_bash("echo '[exit code: 7]'", "prints").code, "0")
        self.assertEqual(self.env.run_bash("echo '[exit code: 0]'; exit 7", "fails").code, "7")
        self.assertEqual(self.env.run_bash("sleep 30", "hangs", timeout_seconds=1).code, "timeout")
        self.assertEqual(self.env.run_bash("rm -rf /", "refused").code, "refused")

    def test_a_flood_of_output_is_kept_bounded_in_memory(self) -> None:
        """50 MB of output never sits whole in the daemon; the marker counts it all (claude-2)."""
        tracemalloc.start()
        try:
            result = self.env.Bash(
                "head -c 50000000 /dev/zero | tr '\\0' a", "flood",
                timeout_seconds=120, max_output_chars=4000,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLessEqual(len(result), 4000)
        self.assertRegex(result, r"truncated 4999\d{4} chars")
        self.assertLess(peak, 24_000_000, f"the output was kept whole: peak {peak} bytes")

    def test_a_slow_stream_callback_delays_neither_the_deadline_nor_a_stop(self) -> None:
        """The kill comes before the queued output, and a Stop ends the streaming (B/S4-F3).

        A 30 ms callback under a flood: the unfixed code kept every frame it had read
        ahead and, at the deadline or the Stop, put all of them (some 600) through the
        callback before it killed the command, 18 s after a 1 s deadline.
        """
        earlier = set(threading.enumerate())
        calls: list[str] = []
        stop = threading.Event()

        def slow(text: str) -> None:
            calls.append(text)
            time.sleep(0.03)

        def stopping(text: str) -> None:
            slow(text)
            if len(calls) == 20:
                stop.set()

        def failing(text: str) -> None:
            slow(text)
            if len(calls) == 20:
                raise RuntimeError("callback failed")

        def exec_threads() -> list[str]:
            return [thread.name for thread in set(threading.enumerate()) - earlier
                    if thread.name.startswith("docker-exec-")]

        self.env.stream_callback = slow
        start = time.monotonic()
        result = self.env.Bash(_FLOOD, "flood", timeout_seconds=1)
        self.assertLess(time.monotonic() - start, 10)
        self.assertIn("timed out", result)
        self.assertLess(len(calls), 200, "the callback was fed a backlog no queue should hold")
        calls.clear()
        self.env.stream_callback, self.env.stop_event = stopping, stop
        self.addCleanup(setattr, self.env, "stop_event", None)
        start = time.monotonic()
        result = self.env.Bash(_FLOOD, "flood", timeout_seconds=60)
        self.assertLess(time.monotonic() - start, 10)
        self.assertIn(STOPPED_MESSAGE, result)
        self.assertEqual(len(calls), 20, "output was streamed after the Stop")
        # A callback that fails leaves the worker's queue full for good: that worker
        # must end all the same, as must the workers of the two calls above.
        calls.clear()
        self.env.stream_callback, self.env.stop_event = failing, None
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            self.env.Bash(_FLOOD, "flood", timeout_seconds=60)
        self.assertTrue(wait_for(lambda: not exec_threads()), exec_threads())

    def test_timeout_does_not_leave_the_command_running(self) -> None:
        """A timed-out streaming command is killed inside the container."""
        chunks: list[str] = []
        self.env.stream_callback = chunks.append
        self.env.Bash("sleep 47", "hang", timeout_seconds=2)
        self.env.stream_callback = None
        time.sleep(1)
        # python:3.11-slim has no `ps`, so read the process table directly.
        alive = self.env.Bash(_COUNT_SLEEP_47, "count leftover processes")
        self.assertEqual(alive.strip().splitlines()[-1], "0")


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerManagerDecoding(unittest.TestCase):
    """F4 — decoding must never raise out of the tool."""

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

    def test_streaming_binary_output_does_not_raise(self) -> None:
        """Invalid UTF-8 is replaced, and the readable tail survives."""
        chunks: list[str] = []
        self.env.stream_callback = chunks.append
        result = self.env.Bash(_BINARY_OUTPUT, "binary")
        self.assertIn("hello", result)
        self.assertIn("hello", "".join(chunks))

    def test_non_streaming_binary_output_does_not_raise(self) -> None:
        """The printer-less path replaces invalid bytes as well."""
        result = self.env.Bash(_BINARY_OUTPUT, "binary")
        self.assertIn("hello", result)

    def test_streaming_multibyte_split_across_frames(self) -> None:
        """A character straddling two stream frames must decode, not raise."""
        chunks: list[str] = []
        self.env.stream_callback = chunks.append
        result = self.env.Bash(_SPLIT_MULTIBYTE, "split")
        self.assertIn("\u00e9", result)
        self.assertIn("\u00e9", "".join(chunks))

    def test_streaming_stderr_is_decoded_too(self) -> None:
        """stderr uses its own incremental decoder."""
        chunks: list[str] = []
        self.env.stream_callback = chunks.append
        result = self.env.Bash(
            'python -c "import sys\n'
            "sys.stderr.buffer.write(b'\\xff' + chr(233).encode())\n"
            'sys.stderr.flush()"',
            "stderr",
        )
        self.assertIn("\u00e9", result)


if __name__ == "__main__":
    unittest.main()
