"""Regression tests for worktree-empty + autoCommit=True post-task path.

User requirement: when ``use_worktree=True``, ``autoCommit=True``, the
task succeeds, and the agent made NO file modifications inside the
worktree, the server MUST NOT auto-commit / auto-merge the empty
worktree (which would produce a no-op merge or leave a stale
``kiss/wt-*`` branch behind).  Instead the worktree branch must be
discarded so the repository is left in the same state as before the
task ran.

The fix lives in ``_TaskRunnerMixin._run_task_inner``'s post-task
finally block: when ``effective_auto_commit`` is True and
``_get_worktree_changed_files(tab_id)`` is empty, the runner now
invokes ``_handle_worktree_action("discard", tab_id)`` instead of
``"merge"``.

These tests drive ``VSCodeServer._run_task_inner`` with a real git
repo and a stub for the parent ``SorcarAgent.run`` (deterministic — not
a mock).  Each test cleans up its temporary git repo and persistence
database in ``tearDown`` so it does not pollute later tests.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer


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


def _list_kiss_wt_branches(repo: str) -> list[str]:
    result = _run_git(repo, "branch", "--list", "kiss/wt-*")
    return [
        line.strip().lstrip("* ").strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _head_sha(repo: str) -> str:
    return _run_git(repo, "rev-parse", "HEAD").stdout.strip()


class _WorktreeAutocommitEmptyBase(unittest.TestCase):
    """Shared setUp / tearDown — fresh git repo, isolated persistence DB."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-empty-")
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
        self.events: list[dict] = []

        def capture(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run

        for tab in list(_RunningAgentState.running_agent_states.values()):
            if tab.agent is not None and tab.agent._wt_pending:
                try:
                    tab.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
        _RunningAgentState.running_agent_states.clear()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _types(self) -> list[str]:
        return [e["type"] for e in self.events]


def _patch_parent_run_create_file(filename: str | None) -> Any:
    """Replace ``ChatSorcarAgent``'s parent ``run`` with a deterministic stub.

    When *filename* is not None, the stub writes that file inside the
    per-task work_dir (the worktree directory).  When *filename* is
    None, the stub makes no file changes — exercising the empty-
    worktree code path that is the subject of these tests.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def stub_run(self_agent: object, **kwargs: object) -> str:
        if filename is not None:
            work_dir = kwargs.get("work_dir")
            if isinstance(work_dir, str) and work_dir:
                (Path(work_dir) / filename).write_text("agent output\n")
        return "success: true\nsummary: stub\n"

    parent_class.run = stub_run
    return original


class TestWorktreeAutocommitEmptyDiscards(_WorktreeAutocommitEmptyBase):
    """When the worktree is empty after a successful task and
    ``autoCommit=True``, the worktree branch must be discarded
    (not auto-merged)."""

    def test_no_changes_autocommit_on_discards_branch(self) -> None:
        """The ``kiss/wt-*`` branch must NOT exist after a successful
        task that made no file changes when auto-commit is on."""
        self._original_run = _patch_parent_run_create_file(None)
        pre_head = _head_sha(self.repo)
        self.server._run_task_inner({
            "prompt": "task with no changes",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": True,
            "model": "",
        })

        # Sanity: the agent did create a worktree branch during the run.
        assert "worktree_created" in self._types(), self._types()

        # Bug repro / fix verification: after the run the branch must
        # be gone (auto-discarded), not lingering as an empty branch
        # and not auto-merged into the original branch.
        branches = _list_kiss_wt_branches(self.repo)
        assert branches == [], (
            f"BUG: when use_worktree=True, autoCommit=True, task succeeds, "
            f"and no files changed, the kiss/wt-* branch should be "
            f"discarded.  Found branches: {branches}.  "
            f"Events: {self._types()}"
        )

        # And HEAD must be unchanged — no auto-merge commit was created
        # (merging an empty branch into the original would otherwise
        # advance HEAD).
        post_head = _head_sha(self.repo)
        assert post_head == pre_head, (
            f"BUG: HEAD advanced from {pre_head} to {post_head} despite "
            f"the worktree having no file modifications.  No commit "
            f"should have been created."
        )

        # In auto-commit mode a task that changed no files must stay
        # SILENT: the empty branch is discarded internally but NO
        # ``worktree_result`` (or any merge/discard status) is
        # broadcast to the user.  Surfacing a "Discarded branch …"
        # notification for a no-op turn would violate the requirement
        # that auto-commit mode prints nothing when nothing changed.
        wt_results = [e for e in self.events if e["type"] == "worktree_result"]
        assert wt_results == [], (
            f"Expected NO worktree_result event for an empty (no-change) "
            f"auto-commit task; got: {wt_results}"
        )

    def test_with_changes_autocommit_on_still_merges(self) -> None:
        """Sanity baseline: when the worktree DOES have changes and
        auto-commit is on, the existing auto-merge behavior is
        preserved (the fix must not regress the with-changes path)."""
        self._original_run = _patch_parent_run_create_file("agent_out.txt")
        pre_head = _head_sha(self.repo)
        self.server._run_task_inner({
            "prompt": "task with changes",
            "workDir": self.repo,
            "tabId": "1",
            "useWorktree": True,
            "autoCommit": True,
            "model": "",
        })

        # Branch should be gone (squash-merged + cleaned up).
        branches = _list_kiss_wt_branches(self.repo)
        assert branches == [], (
            f"Expected the worktree branch to be cleaned up after "
            f"auto-merge; got: {branches}"
        )

        # HEAD advanced (the squash-merge commit was created).
        post_head = _head_sha(self.repo)
        assert post_head != pre_head, (
            f"Expected HEAD to advance after auto-merging real "
            f"changes; both were {post_head}"
        )

        # The new file is present in the main working tree.
        assert (Path(self.repo) / "agent_out.txt").is_file(), (
            "Expected agent_out.txt to be merged into the main tree"
        )

        # A worktree_result with a merge-success message was broadcast.
        wt_results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(wt_results) == 1, (
            f"Expected exactly one worktree_result event; got: {wt_results}"
        )
        assert wt_results[0].get("success") is True, wt_results[0]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
