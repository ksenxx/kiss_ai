# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""When a task fails or is stopped, treat ``autoCommit=True`` as if it
were off.

The user-observed expectation:

    "when auto-commit mode is on, worktree mode is on, files in the
    worktree have been modified and the task fails, behave as if
    auto commit is off, i.e. explicitly show the user the diff/merge
    workflow followed by the worktree merge workflow."

The implementation in :meth:`_TaskRunnerMixin._run_task_inner`'s
finally block computes::

    task_failed = task_end_event.type in ("task_error", "task_stopped")
    effective_auto_commit = tab.auto_commit_mode and not task_failed

and both decision sites (non-worktree autocommit/merge gate and the
worktree merge-review gate) consult ``effective_auto_commit`` instead
of the raw ``tab.auto_commit_mode``.  Therefore on failure / user-stop
the user gets the explicit diff/merge workflow (non-worktree) or the
worktree merge-review (worktree), and on success the original
auto-commit / auto-merge fast path is preserved as a regression guard.

Each test drives the real :meth:`VSCodeServer._run_task_inner` against
a fresh git repo, replacing the stateful agent's parent ``run`` with
a deterministic stub (no mocks).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
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


def _list_kiss_wt_branches(repo: str) -> list[str]:
    result = _run_git(repo, "branch", "--list", "kiss/wt-*")
    return [
        line.strip().lstrip("* ").strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _head_sha(repo: str) -> str:
    return _run_git(repo, "rev-parse", "HEAD").stdout.strip()


class _Base(unittest.TestCase):
    """Fresh git repo + isolated persistence DB per test."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-failure-test-")
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

        from kiss.agents.sorcar.running_agent_state import _RunningAgentState
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


def _patch_run(
    filename: str | None,
    raises: BaseException | None,
) -> Any:
    """Replace ``ChatSorcarAgent``'s parent ``run`` with a stub.

    The stub creates *filename* (relative to work_dir) when given,
    then either raises *raises* or returns a success summary.

    Returns the original ``run`` for restoration in tearDown.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def stub_run(self_agent: object, **kwargs: object) -> str:
        if filename is not None:
            work_dir = kwargs.get("work_dir")
            if isinstance(work_dir, str) and work_dir:
                (Path(work_dir) / filename).write_text("agent output\n")
        if raises is not None:
            raise raises
        return "success: true\nsummary: stub\n"

    parent_class.run = stub_run
    return original


class TestWorktreeFailureWithAutocommit(_Base):
    """Worktree mode + autoCommit=True + task failure must NOT
    auto-merge; the branch must be preserved for manual review."""

    def test_runtime_error_preserves_branch_no_auto_merge(self) -> None:
        """``RuntimeError`` from the agent → ``task_error`` end event.

        Expect: no ``worktree_result`` event (auto-merge skipped),
        ``kiss/wt-*`` branch still present.
        """
        self._original_run = _patch_run(
            "agent_out.txt", raises=RuntimeError("boom"),
        )
        self.server._run_task_inner({
            "prompt": "task that fails",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": True,
            "model": "",
        })

        types = self._types()
        assert "worktree_result" not in types, (
            f"Auto-merge must NOT run on task_error; got events: {types}"
        )
        branches = _list_kiss_wt_branches(self.repo)
        assert len(branches) == 1, (
            f"Worktree branch must survive failed task for manual review; "
            f"branches={branches}, events={types}"
        )

    def test_keyboard_interrupt_preserves_branch_no_auto_merge(self) -> None:
        """``KeyboardInterrupt`` → ``task_stopped`` end event.

        Same expectations as task_error: no auto-merge, branch preserved.
        """
        self._original_run = _patch_run(
            "agent_out.txt", raises=KeyboardInterrupt(),
        )
        self.server._run_task_inner({
            "prompt": "task that is stopped",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": True,
            "model": "",
        })

        types = self._types()
        assert "worktree_result" not in types, (
            f"Auto-merge must NOT run on task_stopped; got events: {types}"
        )
        branches = _list_kiss_wt_branches(self.repo)
        assert len(branches) == 1, (
            f"Worktree branch must survive stopped task; "
            f"branches={branches}, events={types}"
        )


class TestNonWorktreeFailureWithAutocommit(_Base):
    """Non-worktree + autoCommit=True + task failure must NOT silently
    commit; user must get the interactive diff/merge or autocommit
    prompt instead."""

    def test_runtime_error_no_silent_commit(self) -> None:
        """Stub creates a file in the working tree, then raises.

        Expect: HEAD unchanged (no silent auto-commit), and the
        interactive diff/merge workflow is presented (either a
        ``merge_data`` event or an ``autocommit_prompt`` event).
        """
        pre_head = _head_sha(self.repo)
        self._original_run = _patch_run(
            "agent_out.txt", raises=RuntimeError("boom"),
        )
        self.server._run_task_inner({
            "prompt": "task that fails after editing",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": False,
            "autoCommit": True,
            "model": "",
        })

        post_head = _head_sha(self.repo)
        assert pre_head == post_head, (
            f"HEAD must not advance when task fails with autoCommit=True; "
            f"pre={pre_head} post={post_head}, events={self._types()}"
        )
        # The agent's untracked file should still exist on disk for
        # the user to inspect via the merge view.
        assert (Path(self.repo) / "agent_out.txt").exists()

        types = self._types()
        assert "merge_data" in types or "autocommit_prompt" in types, (
            f"User must be shown interactive diff/merge or autocommit "
            f"prompt on failure; got: {types}"
        )


class TestWorktreeSuccessAutoMergeRegression(_Base):
    """Regression guard: when the task SUCCEEDS with autoCommit=True
    and worktree mode, the auto-merge fast path must still run.
    """

    def test_success_still_auto_merges(self) -> None:
        self._original_run = _patch_run("agent_out.txt", raises=None)
        self.server._run_task_inner({
            "prompt": "task that succeeds",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": True,
            "autoCommit": True,
            "model": "",
        })

        types = self._types()
        assert "worktree_result" in types, (
            f"Successful task with autoCommit=True must auto-merge "
            f"the worktree; events: {types}"
        )
        # After a successful auto-merge the branch is cleaned up.
        branches = _list_kiss_wt_branches(self.repo)
        assert branches == [], (
            f"Auto-merge should delete the kiss/wt-* branch after "
            f"merging; remaining: {branches}"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
