# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real-container regressions for the docker file tools:
:class:`~kiss.agents.sorcar.docker_tools.DockerTools`.

``Write`` detected failure by sniffing for the ``[exit code:`` marker, which
the timeout return of ``DockerManager.Bash`` does not carry — so a write that
never happened was reported as ``Successfully wrote N characters``.  ``Read``
was also missing the ``max_lines < 1`` rejection and the ``(file is empty)``
sentinel that ``UsefulTools.Read`` has, and ``Edit`` was missing the
empty-``old_string`` guard, so ``replace_all=True`` with an empty needle
corrupted the file instead of erroring.

The bash function passed to ``DockerTools`` here is the real
``DockerManager.Bash`` bound to a real container; the "slow container" is
simulated by really running the generated script behind a real ``sleep``.
"""

import unittest

import docker

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools


def is_docker_available() -> bool:
    """Return True when a docker daemon is reachable."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerToolsGuards(unittest.TestCase):
    """F2 — Write/Read/Edit must match UsefulTools' contract."""

    env: DockerManager
    tools: DockerTools

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("python:3.11-slim")
        cls.env.open()
        cls.tools = DockerTools(cls.env.Bash)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def tearDown(self) -> None:
        self.env.stream_callback = None

    def _slow_tools(self) -> DockerTools:
        """DockerTools whose commands really outlive a 1 s container timeout."""
        def slow_bash(command: str, description: str) -> str:
            return self.env.Bash(f"sleep 3; {command}", description, timeout_seconds=1)

        return DockerTools(slow_bash)

    def _exists(self, path: str) -> bool:
        out = self.env.Bash(f"test -f {path} && echo yes || echo no", "check")
        return out.strip().splitlines()[-1] == "yes"

    def test_write_that_timed_out_is_not_reported_as_success(self) -> None:
        """A timed-out write must surface the error, not a success message."""
        path = "/tmp/g_write_timeout.txt"
        self.env.Bash(f"rm -f {path}", "clean")
        result = self._slow_tools().Write(path, "payload that never lands")
        self.assertNotIn("Successfully wrote", result)
        self.assertIn("timed out", result)
        self.assertFalse(self._exists(path))

    def test_streaming_write_that_timed_out_is_not_success(self) -> None:
        """Same on the streaming path production actually uses."""
        chunks: list[str] = []
        self.env.stream_callback = chunks.append
        path = "/tmp/g_write_timeout_stream.txt"
        self.env.Bash(f"rm -f {path}", "clean")
        result = self._slow_tools().Write(path, "payload that never lands")
        self.env.stream_callback = None
        self.assertNotIn("Successfully wrote", result)
        self.assertFalse(self._exists(path))

    def test_write_success_still_reports_success(self) -> None:
        """The happy path is unchanged."""
        path = "/tmp/g_write_ok.txt"
        result = self.tools.Write(path, "hello world")
        self.assertIn("Successfully wrote 11 characters", result)
        self.assertIn("hello world", self.tools.Read(path))

    def test_write_failure_still_reports_the_exit_code(self) -> None:
        """A genuinely failing write still returns the shell error."""
        result = self.tools.Write("/dev/null/impossible/test.txt", "fail")
        self.assertNotIn("Successfully wrote", result)
        self.assertIn("exit code:", result)

    def test_read_rejects_max_lines_below_one(self) -> None:
        """``max_lines=0`` is an error, not a silent empty read."""
        path = "/tmp/g_read_maxlines.txt"
        self.tools.Write(path, "a\nb\nc\n")
        result = self.tools.Read(path, max_lines=0)
        self.assertIn("max_lines must be >= 1", result)

    def test_read_empty_file_returns_sentinel(self) -> None:
        """An empty file is distinguishable from an empty command result."""
        path = "/tmp/g_read_empty.txt"
        self.tools.Write(path, "")
        self.assertEqual(self.tools.Read(path).strip(), "(file is empty)")

    def test_edit_rejects_empty_old_string(self) -> None:
        """``replace_all`` with an empty needle must not corrupt the file."""
        path = "/tmp/g_edit_empty.txt"
        self.tools.Write(path, "aaa")
        result = self.tools.Edit(path, "", "X", replace_all=True)
        self.assertIn("old_string must not be empty", result)
        self.assertIn("aaa", self.tools.Read(path))
        self.assertNotIn("XaXaXaX", self.tools.Read(path))


if __name__ == "__main__":
    unittest.main()
