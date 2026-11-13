# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Direct accessor for the changed-file listing that decides whether ``worktree_done`` is sent."""

from __future__ import annotations

from typing import Any

from kiss.server import agent_state
from kiss.server.merge_flow import _flatten_changed


def worktree_changed_files(server: Any, tab_id: str) -> list[str]:
    """The files changed in the pending worktrees of *tab_id* against their original branches.

    The list that decides whether ``worktree_done`` is sent, asked
    directly: files of the worktree the task's work dir is in are
    relative to it, files of any other repository's worktree are
    prefixed with that repository's root, and a worktree git cannot
    list contributes nothing.  ``[]`` for a tab without an agent.
    """
    state = agent_state.find_by_tab(tab_id)
    if state is None or state.agent is None:
        return []
    return _flatten_changed(server._list_pending(state.agent), state.agent._wt)
