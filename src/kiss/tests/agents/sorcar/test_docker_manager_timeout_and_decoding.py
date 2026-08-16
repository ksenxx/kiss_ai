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

Every test here drives a real container through the real docker daemon; no
mocks, patches or doubles.
"""

import time
import unittest

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
