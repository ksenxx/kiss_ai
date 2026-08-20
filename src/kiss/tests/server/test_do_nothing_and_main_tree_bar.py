# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The "Do nothing" worktree action and the non-worktree post-task bar.

Two post-task flows are covered end to end, each driving the real
:class:`VSCodeServer` command handlers against a real git repository
with only the LLM-backed pieces replaced by deterministic stubs
(exactly like the sibling autocommit tests — no mocks):

1. **Worktree + manual-commit — the "Do nothing" button.**  The
   webview's third bar button sends ``worktreeAction`` with
   ``action: "nothing"``.  The daemon must detach from the pending
   worktree while leaving its branch, its directory, and any
   uncommitted changes in it untouched on disk, persist the
   preserve-for-review marker (so a future process's orphan reclaim
   never silently publishes the parked work), and answer with a
   successful ``worktree_result`` carrying ``kept: True`` (so the
   VS Code host keeps its file-link fallback into the still-existing
   worktree).

2. **Non-worktree + manual-commit — the post-task action bar.**  When
   a direct-checkout task ends without an effective auto-commit and
   left the tree dirty, the daemon broadcasts ``main_tree_done``
   (the webview renders it as the Auto commit / Discard / Do nothing
   bar).  The bar's Discard and Do-nothing buttons send
   ``mainTreeAction``, answered by a broadcast ``main_tree_result``:
   ``discard`` reverts tracked changes and removes untracked files
   (never ignored ones), ``nothing`` leaves the tree exactly as it is.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
import kiss.server.merge_flow as _merge_flow_module
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


def _kiss_wt_branches(repo: str) -> list[str]:
    out = _run_git(repo, "branch", "--list", "kiss/wt-*").stdout
    return [
        line.strip().lstrip("*+ ").strip()
        for line in out.splitlines()
        if line.strip()
    ]


def _head_sha(repo: str) -> str:
    return _run_git(repo, "rev-parse", "HEAD").stdout.strip()


def _porcelain(repo: str) -> str:
    return _run_git(repo, "status", "--porcelain", "-uall").stdout


class _Base(unittest.TestCase):
    """Fresh git repo + isolated persistence DB + real server per test."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-do-nothing-test-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        # Deterministic commit-message generation for autocommit paths.
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
        """Make the agent write *filename* into its work dir, then
        return *result* (a failure summary flips ``task_failed``)."""

        def stub_run(_agent: object, **kwargs: object) -> str:
            work_dir = kwargs.get("work_dir")
            if filename is not None and isinstance(work_dir, str) and work_dir:
                (Path(work_dir) / filename).write_text("agent output\n")
            return result

        self._parent_class.run = stub_run

    def _run_task(
        self, tab_id: str, *, use_worktree: bool, auto_commit: bool,
    ) -> None:
        self.server._run_task_inner({
            "prompt": "make a change",
            "workDir": self.repo,
            "tabId": tab_id,
            "useWorktree": use_worktree,
            "autoCommit": auto_commit,
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

        After a manual-commit worktree task presented its bar, the
        "Do nothing" click must: answer success with ``kept: True``,
        release the tab (no pending worktree, not busy), persist the
        preserve marker, and leave the branch, the worktree directory,
        the uncommitted file inside it, and the main checkout all
        exactly as they were.
        """
        self._patch_run()
        self._run_task("tab-wt", use_worktree=True, auto_commit=False)
        assert "worktree_done" in self._types(), (
            f"manual-commit worktree run must present the bar; "
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
        assert _kiss_wt_branches(self.repo) == [branch], (
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

    def test_nothing_fails_closed_when_marker_cannot_be_saved(self) -> None:
        """No detach without the durable preserve marker.

        Dropping the in-memory claim while the marker write failed
        would let a later orphan reclaim silently merge the branch the
        user explicitly parked (gpt-5.6-sol review finding #4).  The
        action must instead report a retryable failure and leave the
        worktree pending, so the bar keeps its controls.
        """
        self._patch_run()
        self._run_task("tab-wt3", use_worktree=True, auto_commit=False)
        self._pending_agent("tab-wt3")
        self.events.clear()

        # `git config` writes .git/config.lock first; a read-only .git
        # makes that fail exactly like a locked/unwritable config.
        git_dir = Path(self.repo) / ".git"
        git_dir.chmod(0o555)
        try:
            self.server._handle_command({
                "type": "worktreeAction",
                "action": "nothing",
                "tabId": "tab-wt3",
            })
        finally:
            git_dir.chmod(0o755)

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
        self._run_task("tab-wt2", use_worktree=True, auto_commit=False)
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
        self._run_task("tab-wt2", use_worktree=True, auto_commit=False)
        assert "worktree_done" in self._types()
        branches = _kiss_wt_branches(self.repo)
        assert parked_branch in branches, (
            f"the parked branch must survive a new task: {branches}"
        )
        assert len(branches) == 2, (
            f"the new task must get its own fresh branch: {branches}"
        )


class TestMainTreeDonePresentation(_Base):
    """When (and only when) the non-worktree post-task bar appears."""

    def test_manual_commit_dirty_tree_presents_the_bar(self) -> None:
        """Manual-commit run that dirtied the checkout → main_tree_done."""
        self._patch_run()
        self._run_task("tab-mt", use_worktree=False, auto_commit=False)

        done = self._only("main_tree_done")
        assert done["tabId"] == "tab-mt"
        assert done["workDir"] == self.repo
        assert done["changedFiles"] == ["agent_out.txt"], done
        # The bar replaces nothing: the edits really are uncommitted.
        assert "autocommit_done" not in self._types()
        assert "agent_out.txt" in _porcelain(self.repo)

    def test_autocommit_run_presents_no_bar(self) -> None:
        """With auto-commit on, the commit happens and no bar appears."""
        self._patch_run()
        self._run_task("tab-mt", use_worktree=False, auto_commit=True)

        assert "main_tree_done" not in self._types(), self._types()
        assert "autocommit_done" in self._types()
        assert _porcelain(self.repo).strip() == ""

    def test_clean_tree_presents_no_bar(self) -> None:
        """A task that changed nothing has nothing to ask about."""
        self._patch_run(filename=None)
        self._run_task("tab-mt", use_worktree=False, auto_commit=False)
        assert "main_tree_done" not in self._types(), self._types()

    def test_failed_run_with_autocommit_on_presents_the_bar(self) -> None:
        """A failed task never auto-commits, so the user gets the bar.

        Mirrors the worktree rule: ``effective_auto_commit`` is off for
        a failed run, and the half-finished edits sitting dirty in the
        checkout are exactly what the bar exists to resolve.
        """
        self._patch_run(result="success: false\nsummary: stub failed\n")
        self._run_task("tab-mt", use_worktree=False, auto_commit=True)

        done = self._only("main_tree_done")
        assert done["changedFiles"] == ["agent_out.txt"], done
        assert "autocommit_done" not in self._types()

    def test_refused_autocommit_presents_the_bar(self) -> None:
        """Auto-commit ON, but git refuses the commit → the bar appears.

        A pre-commit hook rejecting the post-task auto-commit leaves
        the checkout exactly as dirty as manual-commit mode would, so
        gating the bar on the requested mode instead of the tree's
        actual state would strand the user with a failure line and no
        controls (gpt-5.6-sol review finding #1).
        """
        hook_dir = Path(self.tmpdir) / "hooks"
        hook_dir.mkdir(parents=True, exist_ok=True)
        hook = hook_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        _run_git(self.repo, "config", "core.hooksPath", str(hook_dir))

        self._patch_run()
        self._run_task("tab-mt", use_worktree=False, auto_commit=True)

        done_ac = self._only("autocommit_done")
        assert done_ac["success"] is False, (
            f"the hook must have refused the commit: {done_ac}"
        )
        done = self._only("main_tree_done")
        assert "agent_out.txt" in done["changedFiles"], done
        assert "agent_out.txt" in _porcelain(self.repo), (
            "the refused commit must leave the tree dirty"
        )


class TestMainTreeAction(_Base):
    """The mainTreeAction command behind the bar's Discard / Do nothing."""

    def _dirty_repo(self) -> None:
        """One tracked modification, one untracked file, one ignored file."""
        Path(self.repo, "seed.txt").write_text("modified\n")
        Path(self.repo, "untracked.txt").write_text("new\n")
        Path(self.repo, ".gitignore").write_text("ignored.txt\n")
        _run_git(self.repo, "add", ".gitignore")
        _run_git(self.repo, "commit", "-q", "-m", "add gitignore")
        Path(self.repo, "ignored.txt").write_text("keep me\n")

    def _action(self, action: str, work_dir: str | None = None) -> dict[str, Any]:
        self.server._handle_command({
            "type": "mainTreeAction",
            "action": action,
            "tabId": "tab-mt",
            "workDir": self.repo if work_dir is None else work_dir,
        })
        return self._only("main_tree_result")

    def test_nothing_leaves_the_dirty_tree_alone(self) -> None:
        self._dirty_repo()
        before = _porcelain(self.repo)
        result = self._action("nothing")
        assert result["success"] is True, result
        assert result["tabId"] == "tab-mt"
        assert _porcelain(self.repo) == before, (
            "'Do nothing' must not touch the working tree"
        )

    def test_discard_reverts_tracked_and_removes_untracked(self) -> None:
        self._dirty_repo()
        pre_head = _head_sha(self.repo)
        result = self._action("discard")
        assert result["success"] is True, result
        assert "2" in result["message"], (
            f"two files (seed.txt, untracked.txt) were dirty: {result}"
        )
        assert Path(self.repo, "seed.txt").read_text() == "seed\n", (
            "the tracked modification must be reverted"
        )
        assert not Path(self.repo, "untracked.txt").exists(), (
            "the untracked file must be removed"
        )
        assert Path(self.repo, "ignored.txt").read_text() == "keep me\n", (
            "ignored files must survive a discard (clean runs without -x)"
        )
        assert _head_sha(self.repo) == pre_head
        assert _porcelain(self.repo).strip() == ""

    def test_discard_reports_failure_when_dirt_survives(self) -> None:
        """`git clean -fd` cannot remove an untracked nested git repo.

        Both git commands return 0 while `git status` still reports
        the nested repository, so claiming success would dismiss the
        bar over a still-dirty tree (gpt-5.6-sol review finding #7).
        The post-discard re-probe must turn that into a failure that
        names the leftover.
        """
        nested = Path(self.repo) / "nested"
        nested.mkdir()
        _init_repo(str(nested))
        assert "nested/" in _porcelain(self.repo), (
            "the nested repo must show up as dirt in the outer repo"
        )
        result = self._action("discard")
        assert result["success"] is False, result
        assert "nested" in result["message"], result
        assert nested.is_dir(), "git clean -fd must have left the nested repo"

    def test_discard_on_a_clean_tree_is_a_noop(self) -> None:
        result = self._action("discard")
        assert result["success"] is True, result
        assert "Nothing to discard" in result["message"], result

    def test_unknown_action_fails(self) -> None:
        result = self._action("shred")
        assert result["success"] is False, result
        assert "Unknown action" in result["message"], result

    def test_non_git_folder_fails(self) -> None:
        plain = str(Path(self.tmpdir) / "plain")
        Path(plain).mkdir()
        result = self._action("discard", work_dir=plain)
        assert result["success"] is False, result
        assert "not inside a git repository" in result["message"], result

    def test_discard_refused_while_a_task_runs_in_the_repo(self) -> None:
        """The reset would yank files out from under the live agent."""
        self._dirty_repo()
        state = agent_state.AgentState("task-busy", tab_id="tab-busy")
        agent_state.register(state)
        with agent_state.STATE_LOCK:
            state.is_running_non_wt = True
            state.non_wt_repo_root = Path(self.repo)
        try:
            result = self._action("discard")
        finally:
            with agent_state.STATE_LOCK:
                state.is_running_non_wt = False
                state.non_wt_repo_root = None
        assert result["success"] is False, result
        assert "task is still running" in result["message"], result
        assert Path(self.repo, "untracked.txt").exists(), (
            "a refused discard must not delete anything"
        )


if __name__ == "__main__":
    unittest.main()
