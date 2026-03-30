"""Tests for DockerTools — Read, Write, Edit inside Docker containers."""

import unittest

import docker

from kiss.docker.docker_manager import DockerManager
from kiss.docker.docker_tools import DockerTools


def is_docker_available() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestDockerTools(unittest.TestCase):
    """Integration tests for DockerTools using a real Docker container."""

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

    # ── Write tests ──────────────────────────────────────────────

    # ── Read tests ───────────────────────────────────────────────

    # ── Edit tests ───────────────────────────────────────────────

    def test_write_error(self) -> None:
        # /dev/null/subdir is guaranteed to fail (not a directory)
        result = self.tools.Write("/dev/null/impossible/test.txt", "fail")
        self.assertIn("exit code:", result)

    def test_edit_with_quotes(self) -> None:
        self.tools.Write("/tmp/test_edit_q.txt", "say 'hello' and \"bye\"")
        result = self.tools.Edit(
            "/tmp/test_edit_q.txt", "'hello'", "'world'"
        )
        self.assertIn("Successfully replaced", result)
        content = self.tools.Read("/tmp/test_edit_q.txt")
        self.assertIn("'world'", content)
        self.assertIn('"bye"', content)


if __name__ == "__main__":
    unittest.main()
