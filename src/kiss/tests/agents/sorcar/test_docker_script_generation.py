# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Docker Bash/Read/Write/Edit guards, without a docker daemon.

``DockerTools`` is defined against *any* ``bash_fn(command, description)``,
so the exact shell scripts it generates can be executed by a real local
``/bin/bash`` subprocess.  That covers the F2 guards (Write success
detection, Read's ``max_lines``/empty-file handling, Edit's empty-needle
refusal) on machines with no docker daemon, using real files in real temp
directories and a real timeout — no mocks, patches or doubles.

The F3 truncation helper and the F4 incremental decoder are exercised
directly with real byte sequences, because those are the pieces of
``DockerManager`` that carry the fix.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.docker_manager import (
    MAX_OUTPUT_CHARS,
    _new_utf8_decoder,
    _with_exit_code,
)
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.agents.sorcar.useful_tools import _truncate_output


def host_bash(command: str, description: str, timeout_seconds: float = 30) -> str:
    """Run *command* in the real local bash, shaped like ``DockerManager.Bash``.

    Args:
        command: The shell script to run.
        description: Ignored (kept for signature parity).
        timeout_seconds: Real wall-clock timeout; on expiry the exact
            timeout string ``DockerManager.Bash`` returns is produced.

    Returns:
        The combined output plus the ``[exit code: N]`` marker on failure,
        or the timeout error.
    """
    del description
    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_seconds}s"
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    return _with_exit_code(output, proc.returncode)


def slow_host_bash(command: str, description: str) -> str:
    """Run *command* behind a real 3 s sleep with a 1 s timeout."""
    return host_bash(f"sleep 3; {command}", description, timeout_seconds=1)


@unittest.skipIf(sys.platform == "win32", "needs a POSIX shell")
class TestDockerToolsWithoutDocker(unittest.TestCase):
    """F2 — the generated scripts, run by a real local bash."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="kiss_g_tools_")
        self.dir = Path(self._tmp.name)
        self.tools = DockerTools(host_bash)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_write_that_timed_out_is_not_reported_as_success(self) -> None:
        """The timeout return carries no exit-code marker; Write must still fail."""
        target = self.dir / "never.txt"
        result = DockerTools(slow_host_bash).Write(str(target), "payload")
        self.assertNotIn("Successfully wrote", result)
        self.assertIn("timed out", result)
        self.assertFalse(target.exists())

    def test_write_success_reports_success_and_writes_bytes(self) -> None:
        """The happy path really creates the file with the exact content."""
        target = self.dir / "sub" / "ok.txt"
        result = self.tools.Write(str(target), "hello world")
        self.assertIn("Successfully wrote 11 characters", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "hello world")

    def test_write_failure_reports_the_exit_code(self) -> None:
        """A write into a non-directory fails loudly."""
        blocker = self.dir / "blocker"
        blocker.write_text("x", encoding="utf-8")
        result = self.tools.Write(str(blocker / "nested.txt"), "fail")
        self.assertNotIn("Successfully wrote", result)
        self.assertIn("exit code:", result)

    def test_write_marker_is_not_echoed_to_the_model(self) -> None:
        """The success sentinel never leaks into the returned message."""
        target = self.dir / "clean.txt"
        result = self.tools.Write(str(target), "content")
        self.assertNotIn("KISS_WRITE_OK", result)

    def test_read_rejects_max_lines_below_one(self) -> None:
        """``max_lines=0`` errors instead of silently reading nothing."""
        target = self.dir / "lines.txt"
        target.write_text("a\nb\nc\n", encoding="utf-8")
        self.assertIn(
            "max_lines must be >= 1", self.tools.Read(str(target), max_lines=0),
        )
        self.assertIn(
            "max_lines must be >= 1", self.tools.Read(str(target), max_lines=-5),
        )

    def test_read_rejects_start_line_below_one(self) -> None:
        """The pre-existing start_line guard is unchanged."""
        target = self.dir / "lines.txt"
        target.write_text("a\n", encoding="utf-8")
        self.assertIn(
            "start_line must be >= 1", self.tools.Read(str(target), start_line=0),
        )

    def test_read_empty_file_returns_sentinel(self) -> None:
        """An empty file is not indistinguishable from an empty result."""
        target = self.dir / "empty.txt"
        target.write_text("", encoding="utf-8")
        self.assertEqual(self.tools.Read(str(target)).strip(), "(file is empty)")

    def test_read_non_empty_file_is_unaffected(self) -> None:
        """The empty-file branch does not swallow real content."""
        target = self.dir / "content.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        result = self.tools.Read(str(target))
        self.assertIn("alpha", result)
        self.assertIn("beta", result)

    def test_read_missing_file_reports_not_found(self) -> None:
        """A missing file still reports the not-found error."""
        result = self.tools.Read(str(self.dir / "nope.txt"))
        self.assertIn("Error: File not found", result)

    def test_read_truncates_and_windows(self) -> None:
        """max_lines/start_line windowing still works after the guards."""
        target = self.dir / "many.txt"
        target.write_text("".join(f"line{i}\n" for i in range(10)), encoding="utf-8")
        result = self.tools.Read(str(target), max_lines=2, start_line=3)
        self.assertIn("line2", result)
        self.assertIn("line3", result)
        self.assertNotIn("line4", result)
        self.assertIn("truncated: 6 more lines", result)

    def test_edit_rejects_empty_old_string(self) -> None:
        """replace_all with an empty needle must not corrupt the file."""
        target = self.dir / "edit.txt"
        target.write_text("aaa", encoding="utf-8")
        result = self.tools.Edit(str(target), "", "X", replace_all=True)
        self.assertIn("old_string must not be empty", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "aaa")

    def test_edit_rejects_empty_old_string_without_replace_all(self) -> None:
        """The guard applies to the single-replacement mode too."""
        target = self.dir / "edit.txt"
        target.write_text("aaa", encoding="utf-8")
        self.assertIn(
            "old_string must not be empty", self.tools.Edit(str(target), "", "X"),
        )
        self.assertEqual(target.read_text(encoding="utf-8"), "aaa")

    def test_edit_still_replaces(self) -> None:
        """The happy path is unchanged by the new guard."""
        target = self.dir / "edit.txt"
        target.write_text("aaa bbb", encoding="utf-8")
        result = self.tools.Edit(str(target), "bbb", "ccc")
        self.assertIn("Successfully replaced", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "aaa ccc")


class TestDockerManagerHelpers(unittest.TestCase):
    """F3 + F4 — the helpers that carry the manager's fixes."""

    def test_decoder_handles_split_multibyte_sequence(self) -> None:
        """A character split across two frames decodes once both arrive."""
        decoder = _new_utf8_decoder()
        raw = "\u00e9".encode()
        self.assertEqual(decoder.decode(raw[:1]), "")
        self.assertEqual(decoder.decode(raw[1:]), "\u00e9")

    def test_decoder_replaces_invalid_bytes(self) -> None:
        """Binary output is degraded, never fatal."""
        decoder = _new_utf8_decoder()
        text = decoder.decode(b"\xff\xfe hello")
        self.assertIn("hello", text)
        self.assertIn("\ufffd", text)

    def test_decoder_flush_emits_dangling_partial(self) -> None:
        """A truncated sequence at end of stream becomes U+FFFD, not an error."""
        decoder = _new_utf8_decoder()
        self.assertEqual(decoder.decode("\u00e9".encode()[:1]), "")
        self.assertEqual(decoder.decode(b"", True), "\ufffd")

    def test_decoders_are_independent(self) -> None:
        """stdout and stderr must not share decoder state."""
        first, second = _new_utf8_decoder(), _new_utf8_decoder()
        raw = "\u00e9".encode()
        self.assertEqual(first.decode(raw[:1]), "")
        self.assertEqual(second.decode(b"x"), "x")
        self.assertEqual(first.decode(raw[1:]), "\u00e9")

    def test_truncation_cap_matches_useful_tools(self) -> None:
        """The docker default cap is the same 50 000 chars UsefulTools uses."""
        self.assertEqual(MAX_OUTPUT_CHARS, 50000)
        truncated = _truncate_output("x" * 200000, MAX_OUTPUT_CHARS)
        self.assertLessEqual(len(truncated), MAX_OUTPUT_CHARS)
        self.assertIn("truncated", truncated)

    def test_with_exit_code_marks_only_failures(self) -> None:
        """The exit-code marker is appended exactly when the command failed."""
        self.assertEqual(_with_exit_code("out", 0), "out")
        self.assertEqual(_with_exit_code("out", 3), "out\n[exit code: 3]")
        self.assertEqual(_with_exit_code("", 3), "[exit code: 3]")
        self.assertEqual(_with_exit_code("", 0), "")


if __name__ == "__main__":
    unittest.main()
