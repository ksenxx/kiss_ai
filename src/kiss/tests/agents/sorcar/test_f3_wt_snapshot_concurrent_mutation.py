# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``SorcarAgent._wt`` & co. are read by server threads while the task thread grows ``_wts``.

``merge_flow`` reads ``agent._wt`` from the web server's request
threads and the continuation prompt / branch bookkeeping iterate the
pending worktrees, while the task thread keeps storing into and
popping from ``self._wts`` (``_setup_worktree_locked``, ``_merge_one``,
``_discard_worktree``).  A Python-level loop over the live dict raises
``RuntimeError: dictionary changed size during iteration``.  Every
reader now iterates a C-level snapshot (``self.worktrees``) instead.

The writer thread below does exactly what the task thread does to
``_wts`` (a store and a pop of one key) with a short GIL switch
interval so the two threads interleave inside the readers' loops.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.sorcar_agent import SorcarAgent

_RESIDENT = 4_000
_ROUNDS = 60


def _wt(i: int) -> GitWorktree:
    return GitWorktree(
        repo_root=Path(f"/repo{i}"),
        branch=f"kiss/wt-{i}",
        original_branch="main",
        wt_dir=Path(f"/repo{i}/.kiss-worktrees/wt{i}"),
    )


def _grow_and_shrink(agent: SorcarAgent, stop: threading.Event) -> None:
    """Task thread: isolate one more repository, then retire one, repeatedly."""
    i = _RESIDENT
    while not stop.is_set():
        i += 1
        agent._wts[Path(f"/repo{i}")] = _wt(i)
        agent._wts.pop(Path(f"/repo{i}"), None)


def test_worktree_readers_survive_concurrent_wts_mutation(tmp_path: Path) -> None:
    agent = SorcarAgent("f3-wt-snapshot")
    agent.work_dir = str(tmp_path)
    for i in range(_RESIDENT):
        agent._wts[Path(f"/repo{i}")] = _wt(i)
    first = agent._wts[Path("/repo0")]
    stop = threading.Event()
    writer = threading.Thread(target=_grow_and_shrink, args=(agent, stop), daemon=True)
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        writer.start()
        for _ in range(_ROUNDS):
            # work_dir is outside every worktree: ``_wt`` scans them all.
            assert agent._wt is first
            assert "kiss/wt-0" in agent._live_worktree_branches()
            state = agent._continuation_state()
            assert "- Active worktree (unmerged" in state
            assert str(first.wt_dir) in state
    finally:
        stop.set()
        writer.join(timeout=10)
        sys.setswitchinterval(old_interval)
    assert not writer.is_alive()
