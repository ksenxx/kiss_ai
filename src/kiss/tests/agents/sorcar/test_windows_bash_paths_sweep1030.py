# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-30, slice S7, F16: the host shell guards on Windows.

Host commands run in Git Bash there, which spells ``C:\\Users\\x`` as
``/c/Users/x`` and reads a ``C:/x`` word as that path.  The guards used
to resolve targets against ``C:/Users/x`` with POSIX path rules, so a
``C:/...`` target became ``<cwd>/C:/...`` and a ``/c/...`` target was
looked up on the current drive: the here-document guard let an
overwrite of an existing file through, and a ``cat`` of a file did not
count as a read.

These tests run only on a Windows host with Git Bash.  Elsewhere
``shell_scan.bash_path`` and ``host_path`` return their argument
unchanged, and the Windows branch cannot be reached without faking
``sys.platform``, which would be a test double; the POSIX behaviour is
pinned by test_heredoc_guard.py and test_edit_guards_sweep1030.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Git Bash paths of a Windows host")


def _spellings(directory: Path) -> tuple[str, str]:
    """Return *directory* as a drive path (``C:/x``) and as Git Bash spells it (``/c/x``)."""
    drive = directory.resolve().as_posix()
    return drive, f"/{drive[0].lower()}{drive[2:]}"


def test_heredoc_guard_resolves_drive_and_git_bash_paths(tmp_path: Path) -> None:
    """An overwrite of an existing file is refused however its path is spelled."""
    (tmp_path / "notes.md").write_text("old\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    drive, git_bash = _spellings(tmp_path)
    for command in [
        f"cat > {drive}/notes.md <<EOF\nnew\nEOF",
        f"cat > {git_bash}/notes.md <<EOF\nnew\nEOF",
        f"cd .. && cat > {tmp_path.name}/notes.md <<EOF\nnew\nEOF",
        f"cd {git_bash} && cat > notes.md <<EOF\nnew\nEOF",
    ]:
        assert tools.Bash(command, "overwrite").startswith("REFUSED"), command
    assert (tmp_path / "notes.md").read_text() == "old\n"
    assert not tools.Bash(f"cat > {git_bash}/new.md <<EOF\nnew\nEOF", "create").startswith(
        "REFUSED"
    )
    assert (tmp_path / "new.md").read_text() == "new\n"


def test_cat_of_a_git_bash_path_counts_as_a_read(tmp_path: Path) -> None:
    """``cat /c/.../a.py`` shows the file, so an Edit of it may follow."""
    (tmp_path / "a.py").write_text("x = 1\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    _, git_bash = _spellings(tmp_path)
    assert tools.Bash(f"cat {git_bash}/a.py", "view") == "x = 1\n"
    assert tools.Edit(str(tmp_path / "a.py"), "x = 1", "x = 2").startswith("Successfully")
