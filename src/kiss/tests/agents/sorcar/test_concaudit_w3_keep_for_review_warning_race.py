# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``_keep_for_review`` must append to the pending merge warning atomically.

``WorktreeSorcarAgent._keep_for_review`` (worktree_sorcar_agent.py)
used to read ``_merge_conflict_warning`` under ``_warning_lock``,
RELEASE the lock, and then write the concatenation back through
``_set_warnings`` — two separate lock holds around one
read-modify-write.  A ``_flush_warnings`` landing in between (a server
teardown flushing while the agent thread parks a worktree) took and
broadcast the old warning, after which the write put it back, so the
same warning reached the user twice.  Symmetrically, a ``_set_warnings``
landing in between was silently overwritten.

The test drives the real failure path with a real git repository:
a held ``.git/config.lock`` makes ``save_preserve_marker`` fail, so
``_keep_for_review`` takes its warning-appending branch.  Many
concurrent flushes race it; after the fix the pre-existing warning
text is broadcast exactly once.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


class _RecordingPrinter:
    """Minimal broadcast sink: collects every warning message."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.messages.append(str(event.get("message", "")))


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-q", "--allow-empty", "-m", "init"],
        check=True,
    )


def test_prior_warning_is_broadcast_exactly_once_under_concurrent_flush(
    tmp_path: Path,
) -> None:
    """Concurrent flushes never re-deliver the warning ``_keep_for_review`` appends to."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    # What a concurrent ``git config`` holds: makes the marker write fail.
    (repo / ".git" / "config.lock").write_text("")

    wt = GitWorktree(
        repo_root=repo,
        branch="kiss/wt-test",
        original_branch="main",
        wt_dir=repo / ".kiss-worktrees" / "kiss_wt-test",
        baseline_commit=None,
        work_dir=None,
    )
    agent = WorktreeSorcarAgent("keep-for-review-race")
    printer = _RecordingPrinter()
    prior = "PRIOR-WARNING"
    agent._set_warnings(merge=prior)

    keep_done = threading.Event()
    keep_result: list[bool] = []

    def flush_until_kept() -> None:
        # ``_pending_review`` flips right after the marker write fails
        # and right before the warning is appended, so flushing from
        # that moment on overlaps the read-modify-write itself rather
        # than the preceding ``git config`` call.
        while not agent._pending_review and not keep_done.is_set():
            pass
        while not keep_done.is_set():
            agent._flush_warnings(printer)

    def keep() -> None:
        try:
            keep_result.append(agent._keep_for_review(wt))
        finally:
            keep_done.set()

    keeper = threading.Thread(target=keep)
    flushers = [threading.Thread(target=flush_until_kept) for _ in range(4)]
    keeper.start()
    for t in flushers:
        t.start()
    for t in [keeper, *flushers]:
        t.join(timeout=30)
        assert not t.is_alive()
    assert keep_result == [False]
    # Drain whatever is still pending so every broadcast is observed.
    agent._flush_warnings(printer)

    joined = "\n".join(printer.messages)
    assert joined.count(prior) == 1, printer.messages
    assert "could not be recorded in git config" in joined
    assert agent._pending_review is True
