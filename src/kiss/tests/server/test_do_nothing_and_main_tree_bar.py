# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The "Do nothing" worktree action.

The post-task flow is covered end to end, driving the real
:class:`VSCodeServer` command handlers against a real git repository
with only the LLM-backed pieces replaced by deterministic stubs (no
mocks): a task whose agent isolated itself with ``create_worktree``
and finished without ``commit_and_merge`` presents the Merge / Discard
/ Do nothing bar.  The webview's third bar button sends
``worktreeAction`` with ``action: "nothing"``.  The daemon must detach
from the pending worktree while leaving its branch, its directory, and
any uncommitted changes in it untouched on disk, persist the
preserve-for-review marker (so a future process's orphan reclaim never
silently publishes the parked work), and answer with a successful
``worktree_result`` carrying ``kept: True`` (so the VS Code host keeps
its file-link fallback into the still-existing worktree).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
import kiss.server.merge_flow as _merge_flow_module
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import init_repo, kiss_wt_branches, run_git


def _head_sha(repo: str) -> str:
    return run_git(repo, "rev-parse", "HEAD").stdout.strip()


def _porcelain(repo: str) -> str:
    return run_git(repo, "status", "--porcelain", "-uall").stdout


class _Base(unittest.TestCase):
    """Fresh git repo + isolated persistence DB + real server per test."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-do-nothing-test-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        init_repo(self.repo)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "history.db"
        _persistence._db_conn = None

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        # Deterministic commit-message generation for the commit paths.
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff

        def fake_compose(
            diff_text: str,
            user_prompt: str | None = None,
            task_result: str | None = None,
        ) -> str:
            return "test: deterministic autocommit"

        _merge_flow_module.generate_commit_message_from_diff = fake_compose  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen

        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- helpers ---------------------------------------------------

    def _patch_run(
        self,
        filename: str | None = "agent_out.txt",
        result: str = "success: true\nsummary: stub\n",
    ) -> None:
        """Make the agent isolate itself in a worktree, write *filename*
        there, then return *result* without ``commit_and_merge``."""

        def stub_run(_agent: object, **kwargs: object) -> str:
            agent = cast(ChatSorcarAgent, _agent)
            agent._enter_work_dir(str(kwargs.get("work_dir") or ""))
            message = agent.create_worktree()
            assert agent._wt_pending, message
            if filename is not None:
                (Path(agent.work_dir) / filename).write_text("agent output\n")
            return result

        self._parent_class.run = stub_run

    def _run_task(self, tab_id: str) -> None:
        self.server._run_task_inner({
            "prompt": "make a change",
            "workDir": self.repo,
            "tabId": tab_id,
            "model": "",
        })
        # In production the run's worker thread dies when the task
        # returns; a direct synchronous call leaves the *test* thread
        # installed as `task_thread`, which would keep the state
        # `busy()` forever.  Reproduce the post-run idle state.
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
            if (
                state is not None
                and state.task_thread is threading.current_thread()
            ):
                state.task_thread = None
                state.is_task_active = False

    def _types(self) -> list[str]:
        return [str(e.get("type", "")) for e in self.events]

    def _only(self, event_type: str) -> dict[str, Any]:
        found = [e for e in self.events if e.get("type") == event_type]
        assert len(found) == 1, (
            f"expected exactly one {event_type!r}; got {self._types()}"
        )
        return found[0]


class TestWorktreeDoNothing(_Base):
    """worktreeAction 'nothing' detaches and leaves the worktree alone."""

    def _pending_agent(self, tab_id: str) -> Any:
        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        agent = state.agent
        assert agent is not None and agent._wt_pending
        return agent

    def test_nothing_keeps_worktree_and_branch_untouched(self) -> None:
        """The button's whole contract, end to end.

        After an unmerged worktree task presented its bar, the
        "Do nothing" click must: answer success with ``kept: True``,
        release the tab (no pending worktree, not busy), persist the
        preserve marker, and leave the branch, the worktree directory,
        the uncommitted file inside it, and the main checkout all
        exactly as they were.
        """
        self._patch_run()
        self._run_task("tab-wt")
        assert "worktree_done" in self._types(), (
            f"an unmerged worktree run must present the bar; "
            f"got {self._types()}"
        )
        agent = self._pending_agent("tab-wt")
        branch = agent._wt_branch
        wt_dir = Path(agent._wt_dir)
        pre_head = _head_sha(self.repo)
        assert (wt_dir / "agent_out.txt").exists()

        self.server._handle_command({
            "type": "worktreeAction",
            "action": "nothing",
            "tabId": "tab-wt",
        })

        result = self._only("worktree_result")
        assert result["success"] is True, result
        assert result.get("kept") is True, (
            f"'nothing' must flag the worktree as kept: {result}"
        )
        assert branch in result["message"], result
        assert str(wt_dir) in result["message"], result

        # Detached in memory…
        state = agent_state.find_by_tab("tab-wt")
        assert state is not None and state.agent is not None
        assert not state.agent._wt_pending, (
            "'Do nothing' must release the tab's claim on the worktree"
        )
        assert not state.is_merging, "the busy claim must be released"

        # …but untouched on disk.
        assert wt_dir.is_dir(), "the worktree directory must survive"
        assert (wt_dir / "agent_out.txt").exists(), (
            "the uncommitted work must stay in the worktree"
        )
        assert kiss_wt_branches(self.repo) == [branch], (
            "the task branch must survive"
        )
        assert _head_sha(self.repo) == pre_head, (
            "the main checkout must not move"
        )
        assert "agent_out.txt" not in _porcelain(self.repo), (
            "nothing may leak into the main working tree"
        )
        assert GitWorktreeOps.load_preserve_marker(Path(self.repo), branch), (
            "the preserve-for-review marker must be persisted so a "
            "future orphan reclaim never auto-merges the parked work"
        )

    @posix_only("chmod permission bits")
    def test_nothing_fails_closed_when_marker_cannot_be_saved(self) -> None:
        """No detach without the durable preserve marker.

        Dropping the in-memory claim while the marker write failed
        would let a later orphan reclaim silently merge the branch the
        user explicitly parked (gpt-5.6-sol review finding #4).  The
        action must instead report a retryable failure and leave the
        worktree pending, so the bar keeps its controls.
        """
        self._patch_run()
        self._run_task("tab-wt3")
        agent = self._pending_agent("tab-wt3")
        self.events.clear()

        # As if the run-end marker write had failed too: no marker on
        # disk, and a read-only marker directory refuses a new one.
        marker = GitWorktreeOps._preserve_marker_path(Path(self.repo), agent._wt.branch)
        assert marker is not None
        marker.unlink()
        marker.parent.chmod(0o555)
        try:
            self.server._handle_command({
                "type": "worktreeAction",
                "action": "nothing",
                "tabId": "tab-wt3",
            })
        finally:
            marker.parent.chmod(0o755)

        result = self._only("worktree_result")
        assert result["success"] is False, result
        assert result.get("retryable") is True, (
            f"the failure must keep the bar's retry controls: {result}"
        )
        assert "keep decision" in result["message"], result
        state = agent_state.find_by_tab("tab-wt3")
        assert state is not None and state.agent is not None
        assert state.agent._wt_pending, (
            "a failed detach must leave the worktree pending"
        )
        assert not state.is_merging, "the busy claim must be released"

    def test_nothing_survives_a_second_click_and_new_tasks(self) -> None:
        """After 'Do nothing', the tab is free and never re-nagged.

        A second click must fail cleanly (nothing is pending anymore),
        and a following worktree task on the same tab must get a fresh
        branch while the parked one survives.
        """
        self._patch_run()
        self._run_task("tab-wt2")
        agent = self._pending_agent("tab-wt2")
        parked_branch = agent._wt_branch
        self.server._handle_command({
            "type": "worktreeAction",
            "action": "nothing",
            "tabId": "tab-wt2",
        })
        self.events.clear()

        self.server._handle_command({
            "type": "worktreeAction",
            "action": "nothing",
            "tabId": "tab-wt2",
        })
        result = self._only("worktree_result")
        assert result["success"] is False, result
        assert "No pending worktree changes" in result["message"], result

        self.events.clear()
        self._run_task("tab-wt2")
        assert "worktree_done" in self._types()
        branches = kiss_wt_branches(self.repo)
        assert parked_branch in branches, (
            f"the parked branch must survive a new task: {branches}"
        )
        assert len(branches) == 2, (
            f"the new task must get its own fresh branch: {branches}"
        )


if __name__ == "__main__":
    unittest.main()
