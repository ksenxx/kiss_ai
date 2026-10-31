# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-30, slice S7: the host ``checkpoint`` / ``rollback`` tools.

* F4: the git commands of a checkpoint or a rollback ran through
  ``subprocess.run`` with a 300 s timeout and ignored both Stops; hashing
  a large untracked file kept the agent thread for the whole command.
* F13: a rollback refused before it changed anything (something no
  checkpoint holds in the way) still overwrote ``before-rollback``, so
  the state an earlier rollback replaced was lost.

Not covered: the 300 s ``GIT_TIMEOUT_SECONDS`` deadline of a git
command, which takes a git command running for five minutes (or a
patched constant, a test double).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    end_tool_call,
    inject_async_exception,
    interrupt_tool_call,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
)
from kiss.tests.conftest import posix_only


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout


def _repo(path: Path) -> Path:
    """A repository with one commit: ``a.txt`` and ``f``, ``*.log`` ignored."""
    path.mkdir()
    _git(path, "init", "-q")
    (path / ".gitignore").write_text("*.log\n")
    (path / "a.txt").write_text("v1\n")
    (path / "f").write_text("file\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path


def _git_processes(repo: Path) -> str:
    """The command lines of the git processes still working on *repo*.

    ``ps`` rather than ``pgrep -af``: macOS's pgrep prints bare pids
    for ``-a`` (there it means "include ancestors"), so the ``add`` the
    callers wait for never showed up.
    """
    listing = subprocess.run(
        ["ps", "-ww", "-e", "-o", "args="], capture_output=True, text=True,
    ).stdout
    return "".join(line + "\n" for line in listing.splitlines() if f"git -C {repo}" in line)


@posix_only("ps and sparse files")
def test_a_stop_ends_a_slow_checkpoint_at_once(tmp_path: Path) -> None:
    """F4: a checkpoint stopped before it starts never reaches the slow ``git add``.

    Review 1 (R2): its first git command already sees the Stop.
    """
    repo = _repo(tmp_path / "repo")
    with (repo / "big.bin").open("wb") as big:
        big.truncate(2 << 30)
    stop = threading.Event()
    tools = UsefulTools(stop_event=stop, work_dir=str(repo))
    stop.set()
    started = time.monotonic()
    out = tools.checkpoint("slow")
    elapsed = time.monotonic() - started
    assert out == "Error: checkpoint 'slow' not saved: git rev-parse was stopped", out
    assert elapsed < 5, elapsed
    assert _git_processes(repo) == ""
    # Without a stop the same checkpoint completes.
    stop.clear()
    (repo / "big.bin").unlink()
    assert tools.checkpoint("fast").startswith("Saved checkpoint 'fast'")


def _big_repo(tmp_path: Path) -> Path:
    """A repository with an untracked 2 GB sparse file, which ``git add`` hashes for seconds."""
    repo = _repo(tmp_path / "repo")
    with (repo / "big.bin").open("wb") as big:
        big.truncate(2 << 30)
    return repo


def _wait_for_git_add(repo: Path) -> None:
    """Wait until ``git add`` works on *repo*, then half a second more."""
    deadline = time.monotonic() + 10
    while " add " not in _git_processes(repo) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert " add " in _git_processes(repo)
    time.sleep(0.5)


def _checkpoint(tools: UsefulTools, outcome: dict) -> None:
    """Run ``tools.checkpoint("slow")``; record what it returned or raised."""
    try:
        outcome["result"] = tools.checkpoint("slow")
    except BaseException as exc:  # noqa: BLE001 — the test records whatever escaped
        outcome["error"] = exc


@posix_only("ps and sparse files")
def test_a_stop_while_git_works_ends_it(tmp_path: Path) -> None:
    """F4: a task Stop that comes while ``git add`` hashes ends it within a poll."""
    repo = _big_repo(tmp_path)
    stop = threading.Event()
    tools = UsefulTools(stop_event=stop, work_dir=str(repo))
    outcome: dict = {}
    thread = threading.Thread(target=_checkpoint, args=(tools, outcome))
    thread.start()
    _wait_for_git_add(repo)
    started = time.monotonic()
    stop.set()
    thread.join(10)
    assert not thread.is_alive()
    assert time.monotonic() - started < 5
    assert outcome.get("result", "").startswith(
        "Error: checkpoint 'slow' not saved: git add was stopped"
    ), outcome
    assert _git_processes(repo) == ""


@posix_only("ps and sparse files")
def test_a_stop_injected_into_a_slow_checkpoint_ends_git(tmp_path: Path) -> None:
    """F4: the ``KeyboardInterrupt`` a task Stop injects ends the git process group too."""
    repo = _big_repo(tmp_path)
    tools = UsefulTools(work_dir=str(repo))
    outcome: dict = {}
    thread = threading.Thread(target=_checkpoint, args=(tools, outcome))
    thread.start()
    _wait_for_git_add(repo)
    assert thread.ident is not None
    inject_async_exception(thread.ident, KeyboardInterrupt)
    thread.join(10)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), KeyboardInterrupt), outcome
    assert _git_processes(repo) == ""


def _checkpoint_inside_tool_call(tools: UsefulTools, outcome: dict) -> None:
    """Run ``tools.checkpoint`` inside a registered tool call; record what it returned or raised."""
    token = new_tool_call("checkpoint")
    register_tool_call(token)
    outcome["token"] = token
    try:
        outcome["result"] = tools.checkpoint("slow")
    except BaseException as exc:  # noqa: BLE001 — the test records whatever escaped
        outcome["error"] = exc
        unregister_tool_call(token)  # as the agent's ``except ToolCallInterrupted`` does
    else:
        end_tool_call(token)


@posix_only("ps and sparse files")
def test_the_tool_panel_stop_ends_a_slow_checkpoint(tmp_path: Path) -> None:
    """F4: the tool panel's Stop kills git and surfaces as the tool-call interrupt."""
    repo = _repo(tmp_path / "repo")
    with (repo / "big.bin").open("wb") as big:
        big.truncate(2 << 30)
    tools = UsefulTools(work_dir=str(repo))
    outcome: dict = {}
    thread = threading.Thread(target=_checkpoint_inside_tool_call, args=(tools, outcome))
    thread.start()
    deadline = time.monotonic() + 10
    while "git -C" not in _git_processes(repo) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert thread.ident is not None
    started = time.monotonic()
    assert interrupt_tool_call(thread.ident, "checkpoint")
    thread.join(10)
    assert not thread.is_alive()
    assert time.monotonic() - started < 5
    assert isinstance(outcome.get("error"), ToolCallInterrupted), outcome
    assert _git_processes(repo) == ""


def test_a_refused_rollback_keeps_the_previous_before_rollback(tmp_path: Path) -> None:
    """F13: only a rollback that goes on to change files replaces ``before-rollback``."""
    repo = _repo(tmp_path / "repo")
    tools = UsefulTools(work_dir=str(repo))
    assert tools.checkpoint("base").startswith("Saved")
    (repo / "a.txt").write_text("v2\n")  # the work the first rollback replaces
    assert tools.rollback("base").startswith("Rolled ")
    assert (repo / "a.txt").read_text() == "v1\n"
    # f becomes a directory holding an ignored file: rolling back to base is refused.
    (repo / "f").unlink()
    (repo / "f").mkdir()
    (repo / "f" / "debug.log").write_text("log\n")
    out = tools.rollback("base")
    assert out.startswith("Error: rollback to 'base' failed: nothing was changed: f is now a "
                          "directory holding f/debug.log"), out
    shutil.rmtree(repo / "f")
    out = tools.rollback("before-rollback")
    assert out.startswith("Rolled "), out
    assert (repo / "a.txt").read_text() == "v2\n"
    assert (repo / "f").read_text() == "file\n"
