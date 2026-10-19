# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``SorcarAgent.isolated_repos`` is read by a sub-task's thread while the parent grows ``_wts``.

``task_runner._parent_isolated_repos`` calls the parent agent's
:meth:`SorcarAgent.isolated_repos` on the CHILD task's thread at its
run start, while the parent thread keeps calling ``create_worktree``
for further repositories (``_setup_worktree_locked`` stores into
``self._wts``) and retiring merged ones (``_merge_one`` pops).  A
Python-level loop over the live dict then raises ``RuntimeError:
dictionary changed size during iteration`` and the sub-task aborts.
The fix snapshots the dict atomically (one C-level copy under the
GIL) before building the result.

The writer thread below does exactly what the parent's worktree code
does to ``_wts`` (a store and a pop of one key), with a short GIL
switch interval so the two threads interleave inside the loop.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.sorcar_agent import SorcarAgent

_RESIDENT = 20_000
_ROUNDS = 400


def _wt(i: int) -> GitWorktree:
    return GitWorktree(
        repo_root=Path(f"/repo{i}"),
        branch=f"kiss/wt-{i}",
        original_branch="main",
        wt_dir=Path(f"/repo{i}/.kiss-worktrees/wt{i}"),
    )


def _grow_and_shrink(agent: SorcarAgent, stop: threading.Event) -> None:
    """Parent thread: isolate one more repository, then retire one, repeatedly."""
    i = _RESIDENT
    while not stop.is_set():
        i += 1
        agent._wts[Path(f"/repo{i}")] = _wt(i)
        agent._wts.pop(Path(f"/repo{i}"), None)


def test_isolated_repos_survives_concurrent_worktree_creation() -> None:
    agent = SorcarAgent("conc1018-f-parent")
    for i in range(_RESIDENT):
        agent._wts[Path(f"/repo{i}")] = _wt(i)
    # inherit_worktrees stores repo.resolve(); on Windows that turns the drive-less
    # "/ancestor" into "C:\ancestor", so look the entry up under the resolved key.
    ancestor = Path("/ancestor").resolve()
    agent.inherit_worktrees({Path("/ancestor"): Path("/ancestor/.kiss-worktrees/x")})
    stop = threading.Event()
    writer = threading.Thread(target=_grow_and_shrink, args=(agent, stop), daemon=True)
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        writer.start()
        for _ in range(_ROUNDS):
            repos = agent.isolated_repos()
            assert len(repos) >= _RESIDENT + 1
            assert repos[ancestor] == Path("/ancestor/.kiss-worktrees/x")
            assert repos[Path("/repo0")] == Path("/repo0/.kiss-worktrees/wt0")
    finally:
        stop.set()
        writer.join(timeout=10)
        sys.setswitchinterval(old_interval)
    assert not writer.is_alive()
