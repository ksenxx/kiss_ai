# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 1, R2 finding 1: a host Bash command runs in the directory its guards checked.

A concurrent merge or cleanup removes the task's worktree right after
the guards approved a here-document that creates ``tmp/notes.md`` in it
(the worktree has no such file) and before the shell starts.  The launch
used to resolve the directory again, fall back to the parent repository
and overwrite the repository's ``tmp/notes.md``, a file no guard had
looked at.  The guards and the launch now share one directory: the
command runs there or not at all, and the next call checks the fallback
afresh.

The removal is placed between the two by :class:`Pause` (a
``sys.settrace`` hook that runs it on the launching thread at the entry
of the real launch function), not by a test double.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.edit_guards import HEREDOC_VERDICT
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.tests.conftest import posix_only
from kiss.tests.server.pause import Pause

pytestmark = posix_only("the host Bash tool runs bash")

COMMAND = "cat > tmp/notes.md <<'EOF'\nreplacement\nEOF"


def _repo_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(notes, worktree)``: the repo's existing ``tmp/notes.md`` and a live worktree."""
    repo = tmp_path / "repo"
    (repo / "tmp").mkdir(parents=True)
    notes = repo / "tmp" / "notes.md"
    notes.write_text("old\n")
    worktree = repo / ".kiss-worktrees" / "kiss_wt-live"
    (worktree / "tmp").mkdir(parents=True)
    return notes, worktree


class _RemoveAtLaunch:
    """A :class:`Pause` matcher that removes *worktree* when the launch starts, never pausing."""

    def __init__(self, worktree: Path) -> None:
        self.worktree = worktree
        self.calls = 0

    def __call__(self, frame_locals: dict[str, Any]) -> bool:
        self.calls += 1
        shutil.rmtree(self.worktree)
        return False


def test_foreground_command_does_not_run_in_a_directory_no_guard_checked(
    tmp_path: Path,
) -> None:
    notes, worktree = _repo_with_worktree(tmp_path)
    tools = UsefulTools(work_dir=str(worktree))
    remove = _RemoveAtLaunch(worktree)
    with Pause(UsefulTools._spawn.__code__, remove):
        out = tools.Bash(COMMAND, "write the notes")
    assert remove.calls == 1
    assert out.startswith("Error: ") and str(worktree) in out, out
    assert notes.read_text() == "old\n"
    assert not worktree.exists()
    # The next call picks the parent repository and checks it.
    assert tools.Bash(COMMAND, "write the notes") == HEREDOC_VERDICT.format(path=notes)
    assert notes.read_text() == "old\n"


def test_parallel_command_does_not_run_in_a_directory_no_guard_checked(tmp_path: Path) -> None:
    notes, worktree = _repo_with_worktree(tmp_path)
    tools = UsefulTools(work_dir=str(worktree))
    remove = _RemoveAtLaunch(worktree)
    with Pause(UsefulTools._spawn.__code__, remove):
        report = tools.run_commands_parallel(json.dumps([COMMAND]))
    assert remove.calls == 1
    assert f"Error: [Errno 2] No such file or directory: '{worktree}'" in report, report
    assert notes.read_text() == "old\n"


@pytest.mark.parametrize("on_timeout", ["", "background"])
def test_background_job_does_not_start_in_a_directory_no_guard_checked(
    tmp_path: Path, on_timeout: str,
) -> None:
    """``background=True`` and ``on_timeout="background"`` launch through the job starter."""
    notes, worktree = _repo_with_worktree(tmp_path)
    tools = UsefulTools(work_dir=str(worktree))
    remove = _RemoveAtLaunch(worktree)
    with Pause(UsefulTools._start_background_job.__code__, remove):
        if on_timeout:
            out = tools.Bash(COMMAND, "write the notes", timeout_seconds=5, on_timeout=on_timeout)
        else:
            out = tools.Bash(COMMAND, "write the notes", background=True)
    assert remove.calls == 1
    assert out.startswith("Error: could not start background job: "), out
    assert notes.read_text() == "old\n"
    assert not (notes.parent / "bash_jobs").exists()
