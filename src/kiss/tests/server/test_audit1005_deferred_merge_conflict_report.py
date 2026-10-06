# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a deferred-merge retry reports the merge it ran.

``_merge_deferred_worktrees`` retries a worktree merge that was
deferred behind a busy main tree and broadcasts the outcome to the tab
as a ``worktree_result``.  It used to decide whether to broadcast by
re-reading the tab's ``wt_merge_deferred_branch`` marker after the
action returned: a marker still set meant "a guard refused before
anything ran".  But the marker is also re-set by a user's Merge click
that is refused because the main tree got busy again — and when that
click lands between the retry's claim release and its re-read, the
result of the merge the retry DID run (a conflict that keeps the
branch) was swallowed: the tab only ever saw the click's refusal.

``_handle_worktree_action`` now reports whether the retry took the
worktree (``_DEFERRAL_KEPT`` for every exit before ownership), and the
trigger broadcasts every outcome of a merge it ran.

Harness: ``_DeferredMergeBase`` (a real server, repo and worktree task;
the main tree is occupied and edited by a registered tab).  The
deferred retry's call of ``_handle_worktree_action`` is wrapped on the
server instance so that, right after the real call returns, the
occupant is back with a tracked edit and a Merge click is refused —
the re-deferral the race needs.  The merge conflicts on purpose
(``agent_out.txt`` committed on the main branch with other content)
and auto-commit is off, so no merge SEA runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.server import agent_state
from kiss.tests.server.test_worktree_deferred_auto_merge import (
    _DeferredMergeBase,
)
from kiss.tests.server.test_worktree_no_autocommit_branch import _run_git
from kiss.tests.server.test_worktree_repo_aware_busy_guard import _WT_TAB


class TestDeferredRetryReportsItsConflict(_DeferredMergeBase):
    """The conflict of a merge the retry ran reaches the tab."""

    def _redefer_after_deferred_call(self) -> list[dict[str, Any]]:
        """Wrap ``_handle_worktree_action``: a refused Merge click follows the retry's call."""
        calls: list[dict[str, Any]] = []
        real = self.server._handle_worktree_action

        def retry_then_refused_click(
            action: str, tab_id: str = "", **kwargs: Any,
        ) -> dict[str, Any]:
            result = real(action, tab_id, **kwargs)
            calls.append({"action": action, "kwargs": kwargs, "result": result})
            if kwargs.get("deferred_branch") is not None:
                # The main tree is occupied and edited again, and the
                # user clicks Merge: refused, and the merge is deferred
                # once more (the marker is set again).
                self._mark_non_wt_task(Path(self.repo))
                click = real("merge", _WT_TAB)
                calls.append({"action": "click", "result": click})
                self._unmark_non_wt_task()
            return result

        self.server._handle_worktree_action = (  # type: ignore[method-assign]
            retry_then_refused_click
        )
        self.addCleanup(setattr, self.server, "_handle_worktree_action", real)
        return calls

    def test_conflict_of_the_retried_merge_is_broadcast(self) -> None:
        self._strand_worktree()
        state = self._wt_state()
        with self.server._state_lock:
            state.auto_commit_mode = False
        # The main branch gets its own ``agent_out.txt``: the worktree
        # branch adds the same file with other content, so the retried
        # merge conflicts and keeps the branch.
        (Path(self.repo) / "agent_out.txt").write_text("main tree version\n")
        _run_git(self.repo, "add", "agent_out.txt")
        _run_git(self.repo, "commit", "-q", "-m", "conflicting file on main")
        calls = self._redefer_after_deferred_call()

        self.server._merge_deferred_worktrees(Path(self.repo))

        retry = [c for c in calls if c["action"] == "merge"]
        click = [c for c in calls if c["action"] == "click"]
        assert len(retry) == 1 and len(click) == 1, calls
        # The retry ran the merge and hit the conflict; the click was
        # refused and re-deferred the merge.
        assert retry[0]["result"]["success"] is False, retry
        assert "conflict" in retry[0]["result"]["message"].lower(), retry
        assert "merged automatically" in click[0]["result"]["message"], click
        assert state.agent is not None and state.agent._wt_pending
        assert state.wt_merge_deferred_branch == state.agent._wt_branch
        # The outcome of the merge the retry ran reached the tab.
        results = [
            e for e in self._worktree_results() if e.get("tabId") == _WT_TAB
        ]
        assert results, "the retried merge's conflict was never reported"
        assert results[-1]["success"] is False
        assert "conflict" in results[-1]["message"].lower(), results
        assert agent_state.find_by_tab(_WT_TAB) is state
