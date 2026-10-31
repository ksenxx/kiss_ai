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
    rewrite, delete = plan_restore(target, snapshot(str(repo), stop), repo, stop)
    assert (rewrite, delete) == (["a.txt"], ["new.txt"])
    stop.set()
    with pytest.raises(CheckpointError, match="^deleting new.txt was stopped$"):
        apply_restore(target, rewrite, delete, repo, stop)
    # Nothing to delete: the rewrite is not started either.
    with pytest.raises(CheckpointError, match="^git read-tree was stopped$"):
        apply_restore(target, rewrite, [], repo, stop)
    assert (repo / "a.txt").read_text() == "v2\n"
    assert (repo / "new.txt").read_text() == "new\n"


class _StopAtFirstDeletion:
    """A :class:`Pause` matcher: set *stop* when the first file is about to be deleted."""

    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop

    def __call__(self, frame_locals: dict[str, Any]) -> bool:
        self.stop.set()
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
