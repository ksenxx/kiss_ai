# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for DockerTools — Read, Write, Edit inside Docker containers."""

import functools
import unittest

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.core.config import DEFAULT_CONFIG
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available


class _Steps:
    """The agent's step counter a test sets: what ``DockerTools(step_count=...)`` cites."""

    def __init__(self) -> None:
        self.step = 0

    def __call__(self) -> int:
        return self.step


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

    def _session(self, step_count: _Steps | None = None) -> DockerTools:
        """A fresh DockerTools with the file fetch: one session of the Read dedupe."""
        return DockerTools(self.env.Bash, self.env.read_bytes, step_count)

    def test_read_returns_the_unchanged_note_for_lines_already_returned(self) -> None:
        """A Read whose lines an earlier Read returned (any range, any path) is the note."""
        tools = self._session()
        tools.Write("/tmp/dedupe/a.txt", "x\ny\nx\ny\nz\n")
        tools.Write("/tmp/dedupe/b.txt", "x\ny\nx\ny\nz\n")
        first = tools.Read("/tmp/dedupe/a.txt", max_lines=2)
        self.assertEqual(first, "x\ny\n[truncated: 3 more lines]\n")
        # Another range of the same file with the same lines (the truncation note aside).
        self.assertEqual(
            tools.Read("/tmp/dedupe/a.txt", max_lines=2, start_line=3),
            "Unchanged since your earlier Read at step 1 (lines 1-2 of /tmp/dedupe/a.txt)",
        )
        # The whole file; then a sub-range of that request whose window is the same
        # text; then another path with the same content.
        self.assertEqual(tools.Read("/tmp/dedupe/a.txt"), "x\ny\nx\ny\nz\n")
        self.assertEqual(
            tools.Read("/tmp/dedupe/a.txt", max_lines=5, start_line=1),
            "Unchanged since your earlier Read at step 3 (lines 1-5 of /tmp/dedupe/a.txt)",
        )
        self.assertEqual(
            tools.Read("/tmp/dedupe/b.txt"),
            "Unchanged since your earlier Read at step 3 (lines 1-5 of /tmp/dedupe/a.txt)",
        )
        # A changed file is read again; force=True re-sends an unchanged window and
        # becomes the showing the next note cites.
        self.assertIn("Successfully", tools.Edit("/tmp/dedupe/a.txt", "z", "w"))
        self.assertEqual(tools.Read("/tmp/dedupe/a.txt"), "x\ny\nx\ny\nw\n")
        self.assertEqual(tools.Read("/tmp/dedupe/b.txt", force=True), "x\ny\nx\ny\nz\n")
        self.assertEqual(
            tools.Read("/tmp/dedupe/b.txt"),
            "Unchanged since your earlier Read at step 7 (lines 1-5 of /tmp/dedupe/b.txt)",
        )
        # The reset of a context hand-off: the content is sent again.
        tools.reset_reads()
        self.assertEqual(tools.Read("/tmp/dedupe/b.txt"), "x\ny\nx\ny\nz\n")
        self.assertEqual(
            tools.Read("/tmp/dedupe/b.txt"),
            "Unchanged since your earlier Read at step 9 (lines 1-5 of /tmp/dedupe/b.txt)",
        )

    def test_read_note_cites_the_harness_step_and_errors_are_never_deduped(self) -> None:
        """The note cites the agent's step; errors, sentinels and failed reads come back whole."""
        steps = _Steps()
        tools = self._session(steps)
        tools.Write("/tmp/dedupe/steps.txt", "one\n")
        tools.Write("/tmp/dedupe/empty.txt", "")
        steps.step = 12
        self.assertEqual(tools.Read("/tmp/dedupe/steps.txt"), "one\n")
        steps.step = 15
        self.assertEqual(
            tools.Read("/tmp/dedupe/steps.txt"),
            "Unchanged since your earlier Read at step 12 (lines 1-1 of /tmp/dedupe/steps.txt)",
        )
        for _ in range(2):
            self.assertTrue(tools.Read("/tmp/dedupe/empty.txt").startswith("(file is empty)"))
            self.assertTrue(tools.Read("/tmp/dedupe/none.txt").startswith("Error: File not found"))
            past = tools.Read("/tmp/dedupe/steps.txt", start_line=5)
            self.assertTrue(past.startswith("Error: start_line=5 is past EOF"), past)
            # A file that exists but cannot be read fails the command (no window to record).
            failed = tools.Read("/proc/self/mem", max_lines=2)
            self.assertIn("Error: cannot read /proc/self/mem", failed)
            self.assertTrue(failed.rstrip().endswith("[exit code: 1]"), failed)
        # KISS_READ_DEDUPE=0 turns the note off.
        enabled = DEFAULT_CONFIG.read_dedupe
        DEFAULT_CONFIG.read_dedupe = False
        try:
            self.assertEqual(tools.Read("/tmp/dedupe/steps.txt"), "one\n")
        finally:
            DEFAULT_CONFIG.read_dedupe = enabled

    def test_read_counts_the_lines_of_a_path_shaped_like_an_assignment(self) -> None:
        """A relative ``name=value`` path is read like any other (sweep 2026-11-11).

        ``awk`` takes an operand of that shape for a variable assignment
        and reads its standard input instead, which is empty: the line
        count was 0 and ``Read`` answered ``(file is empty)`` for a file
        with lines (a partition directory such as ``date=2024-01-01/``).
        """
        tools = self._session()
        tools.Write("date=2024-01-01/part.csv", "1\n2\n3\n")
        head = tools.Read("date=2024-01-01/part.csv", max_lines=2)
        self.assertEqual(head, "1\n2\n[truncated: 1 more lines]\n")
        self.assertEqual(tools.Read("date=2024-01-01/part.csv", start_line=3), "3\n")

    def test_edit_tolerates_whitespace_and_keeps_the_file_indentation(self) -> None:
        """An old_string off only by blanks edits the original lines; an exact match stays exact."""
        tools = self._session()
        tools.Write(
            "/tmp/tolerant/f.py",
            "def f(x):\n    if x:\n        return foo(a,  b)   \n    return 0\n",
        )
        # Less indentation, one blank for two, no trailing blanks: the matched
        # lines keep their indentation, the new text gets it too.
        result = tools.Edit(
            "/tmp/tolerant/f.py",
            "  if x:\n      return foo(a, b)",
            "  if x:\n      return bar(a, b)\n  else:\n      pass",
        )
        self.assertIn("Successfully replaced 1 occurrence(s)", result)
        self.assertEqual(
            tools.Read("/tmp/tolerant/f.py"),
            "def f(x):\n    if x:\n        return bar(a, b)\n    else:\n        pass\n"
            "    return 0\n",
        )
        # An exact match takes the exact path even where the tolerant match is ambiguous.
        tools.Write("/tmp/tolerant/g.txt", "a = 1\na =  1\n")
        self.assertIn("Successfully", tools.Edit("/tmp/tolerant/g.txt", "a = 1", "a = 2"))
        self.assertEqual(tools.Read("/tmp/tolerant/g.txt"), "a = 2\na =  1\n")
        # A CRLF file gets the new text with its line endings.
        tools.Write("/tmp/tolerant/h.txt", "one\r\ntwo  =  2\r\nthree\r\n")
        result = tools.Edit("/tmp/tolerant/h.txt", "one\ntwo = 2", "uno\ndos = 2")
        self.assertIn("Successfully", result)
        self.assertEqual(self.env.read_bytes("/tmp/tolerant/h.txt"), b"uno\r\ndos = 2\r\nthree\r\n")

    def test_edit_names_the_ambiguity_or_the_closest_region_when_nothing_matches(self) -> None:
        """Several tolerant matches: their count; none: the closest lines, usable as old_string."""
        tools = self._session()
        path = "/tmp/tolerant/n.py"
        tools.Write(path, "import os\n\ndef main():\n    run(1)\n    run(2)\n    run(1)\n")
        # replace_all keeps the exact semantics: no tolerant substitution, the region named.
        self.assertEqual(
            tools.Edit(path, "run(  2)", "stop()", replace_all=True),
            "Error: String not found in file. The closest region is lines 5-5 of "
            f"{path}; pass its text verbatim as old_string:\n    run(2)",
        )
        self.assertEqual(
            tools.Edit(path, " run(1)   ", "x"),
            "Error: String not found in file; ignoring runs of blanks and trailing "
            "whitespace it matches 2 places, so add surrounding lines to old_string "
            "to make it unique",
        )
        region = "def main():\n    run(1)\n    run(2)"
        self.assertEqual(
            tools.Edit(path, "def main():\n    run(3)\n    run(2)", "x"),
            "Error: String not found in file. The closest region is lines 3-5 of "
            f"{path}; pass its text verbatim as old_string:\n{region}",
        )
        self.assertIn("Successfully", tools.Edit(path, region, "def main():\n    pass"))
        self.assertEqual(tools.Read(path), "import os\n\ndef main():\n    pass\n    run(1)\n")
        # Without the file's text (no fetch, a fetch over its cap, an empty file) the
        # exact error stands.
        bare = "Error: String not found in file"
        self.assertTrue(DockerTools(self.env.Bash).Edit(path, "nope", "x").startswith(bare))
        capped = DockerTools(self.env.Bash, functools.partial(self.env.read_bytes, max_bytes=4))
        self.assertTrue(capped.Edit(path, "nope", "x").startswith(bare))
        tools.Write("/tmp/tolerant/empty.txt", "")
        empty = tools.Edit("/tmp/tolerant/empty.txt", "nope", "x")
        self.assertEqual(empty.split(), f"{bare}\n[exit code: 1]".split())

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
