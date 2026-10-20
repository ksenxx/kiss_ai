# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Direct accessors for the merge-flow answers the ``worktree_done`` event carries."""

from __future__ import annotations

from typing import Any

from kiss.server import agent_state
from kiss.server.merge_flow import _pending_worktrees


def merge_conflict(server: Any, tab_id: str = "") -> bool:
    """Whether merging any pending worktree of *tab_id* into its original branch would conflict.

    The same per-worktree ``_conflict_of`` probe ``worktree_done`` folds
    into its ``hasConflict`` field, asked directly so a test can check
    the answer without presenting the worktree.  Pure query: nothing
    is committed or otherwise mutated.
    """
    state = agent_state.find_by_tab(tab_id)
    wt_agent = state.agent if state is not None else None
    return any(server._conflict_of(wt) for wt in _pending_worktrees(wt_agent))
