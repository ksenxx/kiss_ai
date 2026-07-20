# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt regression: DockerTools.Read line counting on no-trailing-newline files.

``DockerTools.Read`` historically counted lines with ``wc -l``, which
counts *newline characters*, not lines: a file whose last line lacks a
trailing newline is undercounted by one.  Consequences:

* ``Read(path, start_line=N)`` for the real last line reported a bogus
  ``past EOF`` error even though the line exists (``tail -n +N`` would
  have printed it just fine), and
* the ``[truncated: N more lines]`` marker was omitted (or undercounted)
  when the window ended before a final unterminated line.

The tests exercise the generated shell command through a REAL local bash
subprocess (the same execution contract as ``DockerManager.Bash``), so
they validate end-to-end behavior without any mocks.
"""

import subprocess
from pathlib import Path

from kiss.core.docker_tools import DockerTools


def _local_bash(command: str, description: str) -> str:
    """Execute *command* with real bash, mirroring DockerManager.Bash output."""
    del description
    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=30,
    )
    parts = [p for p in (completed.stdout, completed.stderr) if p]
    output = "\n".join(parts)
    if completed.returncode != 0:
        suffix = f"[exit code: {completed.returncode}]"
        output = f"{output}\n{suffix}" if output else suffix
    return output


def test_read_last_line_without_trailing_newline(tmp_path: Path) -> None:
    """start_line pointing at an unterminated final line must return it."""
    f = tmp_path / "no_newline.txt"
    f.write_bytes(b"line1\nline2")  # no trailing newline
    tools = DockerTools(_local_bash)
    result = tools.Read(str(f), start_line=2)
    assert "past EOF" not in result, result
    assert "line2" in result


def test_read_truncation_marker_counts_unterminated_line(tmp_path: Path) -> None:
    """The truncated-lines marker must count a final unterminated line."""
    f = tmp_path / "no_newline.txt"
    f.write_bytes(b"line1\nline2")  # 2 lines, no trailing newline
    tools = DockerTools(_local_bash)
    result = tools.Read(str(f), max_lines=1)
    assert "line1" in result
    assert "[truncated: 1 more lines]" in result, result


def test_read_with_trailing_newline_still_works(tmp_path: Path) -> None:
    """Regular files keep their existing semantics."""
    f = tmp_path / "normal.txt"
    f.write_text("a\nb\nc\n")
    tools = DockerTools(_local_bash)

    assert tools.Read(str(f), start_line=2).splitlines()[0] == "b"
    past_eof = tools.Read(str(f), start_line=4)
    assert "past EOF" in past_eof
    assert "3 lines" in past_eof

    windowed = tools.Read(str(f), max_lines=2)
    assert "a" in windowed and "b" in windowed
    assert "[truncated: 1 more lines]" in windowed
