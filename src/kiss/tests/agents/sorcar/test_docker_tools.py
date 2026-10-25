# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for DockerTools — Read, Write, Edit inside Docker containers."""

import unittest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available


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


    def test_write_error(self) -> None:
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

    def test_edit_applies_several_pairs_in_order_and_stops_at_the_first_failure(self) -> None:
        """``edits=`` applies the pairs one by one; a bad pair stops the batch and is named."""
        import json

        self.tools.Write("/tmp/test_edit_batch.txt", "a = 1\nb = 2\nc = 3\n")
        batch = json.dumps([
            {"old_string": "a = 1", "new_string": "a = 10"},
            {"old_string": "c = 3", "new_string": "c = 30"},
        ])
        result = self.tools.Edit("/tmp/test_edit_batch.txt", edits=batch)
        self.assertIn("edit 1: Successfully replaced", result)
        self.assertIn("edit 2: Successfully replaced", result)
        self.assertEqual(
            self.tools.Read("/tmp/test_edit_batch.txt").splitlines()[:3],
            ["a = 10", "b = 2", "c = 30"],
        )
        failing = json.dumps([
            {"old_string": "b = 2", "new_string": "b = 20"},
            {"old_string": "missing", "new_string": "x"},
            {"old_string": "c = 30", "new_string": "c = 300"},
        ])
        result = self.tools.Edit("/tmp/test_edit_batch.txt", edits=failing)
        self.assertTrue(result.startswith("Error: edit 2: edits 1-1 applied; "), result)
        content = self.tools.Read("/tmp/test_edit_batch.txt")
        self.assertIn("b = 20", content)
        self.assertIn("c = 30\n", content + "\n")
        self.assertNotIn("c = 300", content)
        self.assertTrue(self.tools.Edit("/tmp/test_edit_batch.txt", edits="[").startswith("Error"))


if __name__ == "__main__":
    unittest.main()
