# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 1, R2 finding 4: a Stop is honoured between the steps of a checkpoint or rollback.

The wait for a git command looked at the Stops only while the command
ran past its first 0.2 s poll, and no step looked before starting: a
checkpoint or rollback made of fast git commands ran to completion after
a Stop, and a Stop between planning a rollback and applying it still
deleted and rewrote files.  Each git command and each deletion now
checks the Stops before it starts; a command already running finishes or
is ended as before.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import checkpoints
from kiss.agents.sorcar.checkpoints import CheckpointError, apply_restore, plan_restore, snapshot
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.tests.server.pause import Pause


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout


def _repo(path: Path) -> Path:
    """A repository with one commit holding ``a.txt``."""
    path.mkdir()
    _git(path, "init", "-q")
    (path / "a.txt").write_text("v1\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path


def test_a_stop_before_a_fast_checkpoint_saves_nothing(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    stop = threading.Event()
    tools = UsefulTools(stop_event=stop, work_dir=str(repo))
    stop.set()
    assert tools.checkpoint("base") == (
        "Error: checkpoint 'base' not saved: git rev-parse was stopped")
    stop.clear()
    assert tools.rollback("base") == "Error: unknown checkpoint 'base'. Known: none."


def test_a_stop_between_planning_and_applying_a_rollback_changes_nothing(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    stop = threading.Event()
    target = snapshot(str(repo), stop)
    (repo / "a.txt").write_text("v2\n")
    (repo / "new.txt").write_text("new\n")
    rewrite, missing, delete = plan_restore(target, snapshot(str(repo), stop), repo, stop)
    assert (rewrite, missing, delete) == (["a.txt"], [], ["new.txt"])
    stop.set()
    with pytest.raises(CheckpointError, match="^deleting new.txt was stopped$"):
        apply_restore(target, rewrite, missing, delete, repo, stop)
    # Nothing to delete: the rewrite is not started either.
    with pytest.raises(CheckpointError, match="^git read-tree was stopped$"):
        apply_restore(target, rewrite, missing, [], repo, stop)
    assert (repo / "a.txt").read_text() == "v2\n"
    assert (repo / "new.txt").read_text() == "new\n"


class _StopAtFirstDeletion:
    """A :class:`Pause` matcher: set *stop* as the first file is about to be deleted or replaced."""

    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop

    def __call__(self, frame_locals: dict[str, Any]) -> bool:
        self.stop.set()
        return False


class _StopAtCheck:
    """A :class:`Pause` matcher on ``_raise_if_stopped``: set *stop* at the *nth* check of *step*.

    The check is entered with the Stop not yet set and finds it set: *step* ends right there.
    """

    def __init__(self, stop: threading.Event, step: str, nth: int) -> None:
        self.stop, self.step, self.left = stop, step, nth

    def __call__(self, frame_locals: dict[str, Any]) -> bool:
        if frame_locals["step"] == self.step:
            self.left -= 1
            if self.left == 0:
                self.stop.set()
        return False


class _StopBeforeSecondRewrite:
    """Set *stop* when a second file is about to be rewritten, regardless of ordering."""

    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.seen: set[str] = set()

    def __call__(self, frame_locals: dict[str, Any]) -> bool:
        step = frame_locals["step"]
        if step.startswith("rewriting ") and step not in self.seen:
            if self.seen:
                self.stop.set()
            self.seen.add(step)
        return False


def test_a_stop_during_a_rollback_ends_it_before_the_next_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    stop = threading.Event()
    tools = UsefulTools(stop_event=stop, work_dir=str(repo))
    assert tools.checkpoint("base").startswith("Saved checkpoint 'base'")
    (repo / "a.txt").write_text("v2\n")
    (repo / "n1.txt").write_text("1\n")
    (repo / "n2.txt").write_text("2\n")
    with Pause(checkpoints._delete.__code__, _StopAtFirstDeletion(stop)):
        out = tools.rollback("base")
    assert out == "Error: rollback to 'base' failed: deleting n2.txt was stopped"
    assert not (repo / "n1.txt").exists()
    assert (repo / "n2.txt").read_text() == "2\n"
    assert (repo / "a.txt").read_text() == "v2\n"
    # The state the rollback replaced was saved before its first change.
    stop.clear()
    assert tools.rollback("before-rollback").startswith("Rolled ")
    assert (repo / "n1.txt").read_text() == "1\n"
    # The same between two files it replaces: exactly one has reached its old text.
    assert tools.checkpoint("three").startswith("Saved checkpoint 'three'")
    (repo / "n1.txt").write_text("1 changed\n")
    (repo / "n2.txt").write_text("2 changed\n")
    between_files = _StopBeforeSecondRewrite(stop)
    with Pause(checkpoints._raise_if_stopped.__code__, between_files):
        out = tools.rollback("three")
    assert out.startswith("Error: rollback to 'three' failed: rewriting ")
    assert out.endswith(" was stopped")
    restored = [
        path.read_text() if path.exists() else None
        for path in (repo / "n1.txt", repo / "n2.txt")
    ]
    if checkpoints._DIR_FD:
        assert sum(text in {"1\n", "2\n"} for text in restored) == 1
        assert sum(text in {"1 changed\n", "2 changed\n"} for text in restored) == 1
    else:
        # Native Windows cannot rename staged files over open paths. The fallback removes
        # one old file before reaching the next, then writes replacements in one checkout.
        assert sum(text is None for text in restored) == 1
        assert sum(text in {"1 changed\n", "2 changed\n"} for text in restored) == 1
    assert not list((repo / ".git").glob("kiss-rollback-*"))  # where the new files waited
    stop.clear()
    assert tools.rollback("before-rollback").startswith("Rolled ")
    assert (repo / "n1.txt").read_text() == "1 changed\n"
    # The same between two of the empty directories left where a file goes back.
    assert tools.checkpoint("four").startswith("Saved checkpoint 'four'")
    for name in ("n1.txt", "n2.txt"):
        (repo / name).unlink()
        (repo / name / "a").mkdir(parents=True)
        (repo / name / "b").mkdir()
    second = _StopAtCheck(stop, "rewriting n1.txt", 2)
    with Pause(checkpoints._raise_if_stopped.__code__, second):
        out = tools.rollback("four")
    assert out == "Error: rollback to 'four' failed: rewriting n1.txt was stopped"
    assert [len(list((repo / name).iterdir())) for name in ("n1.txt", "n2.txt")] == [1, 2]
    stop.clear()
    assert tools.rollback("four").startswith("Rolled ")
    assert (repo / "n2.txt").read_text() == "2 changed\n"
