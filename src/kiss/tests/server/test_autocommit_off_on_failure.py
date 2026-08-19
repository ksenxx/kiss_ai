# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""When a worktree task fails or is stopped, treat ``autoCommit=True``
as if it were off.

The implementation in :meth:`_TaskRunnerMixin._run_task_inner`'s
finally block computes::

    task_failed = task_end_event.type in ("task_error", "task_stopped")
    effective_auto_commit = tab.auto_commit_mode and not task_failed

and the worktree finalization gate consults ``effective_auto_commit``
instead of the raw ``tab.auto_commit_mode``.  Therefore on failure /
user-stop the user gets the explicit ``worktree_done`` Merge / Discard
prompt with the branch preserved, and on success the auto-merge fast
path is preserved as a regression guard.

Non-worktree tasks obey exactly the same rule, and for the same
reason.  ``effective_auto_commit`` gates the main tree's post-task
commit too, so a run that carried ``autoCommit: false`` — or that
failed — leaves its edits in the user's checkout as ordinary
uncommitted changes.  Nothing is stranded: those files are in the
working tree the user is looking at, listed by ``git status`` and by
the editor's source-control view, which is precisely where a
half-finished change belongs.  Committing it anyway would make the
visible per-run checkbox meaningless and would bake a failed task's
partial edits into the user's history.

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
import kiss.server.merge_flow as _merge_flow_module
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

        # Deterministic commit-message generation: the post-task
        # autocommit path calls this module-level extension point.
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

        from kiss.server import agent_state
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
        assert "worktree_done" in types, (
            f"The Merge / Discard prompt must be shown for the failed "
            f"task's branch; got events: {types}"
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
        assert "worktree_done" in types, (
            f"The Merge / Discard prompt must be shown for the stopped "
            f"task's branch; got events: {types}"
        )
        branches = _list_kiss_wt_branches(self.repo)
        assert len(branches) == 1, (
            f"Worktree branch must survive stopped task; "
            f"branches={branches}, events={types}"
        )


class TestNonWorktreeCommitObeysTheRun(_Base):
    """A task on the user's own checkout commits only when asked to."""

    def test_successful_run_with_the_toggle_on_commits(self) -> None:
        """The happy path still commits, so the toggle is real."""
        pre_head = _head_sha(self.repo)
        self._original_run = _patch_run("agent_out.txt", raises=None)
        self.server._run_task_inner({
            "prompt": "task with the toggle on",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": False,
            "autoCommit": True,
            "model": "",
        })

        assert pre_head != _head_sha(self.repo), (
            f"autoCommit=True must commit the task's work; "
            f"events={self._types()}"
        )
        status = _run_git(self.repo, "status", "--porcelain").stdout.strip()
        assert status == "", f"tree must be clean after autocommit: {status}"

        types = self._types()
        assert "autocommit_done" in types, (
            f"autocommit_done must be broadcast; got: {types}"
        )
        done = next(
            e for e in self.events if e["type"] == "autocommit_done"
        )
        assert done["success"] is True
        assert done["committed"] is True

    def test_failed_run_leaves_its_edits_uncommitted(self) -> None:
        """Stub creates a file in the working tree, then raises.

        The half-finished edit stays in the checkout as an ordinary
        uncommitted change — visible in ``git status`` and in the
        editor — instead of being baked into the user's history by a
        task that did not succeed.
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

        assert pre_head == _head_sha(self.repo), (
            f"a failed task's partial changes were committed; "
            f"events={self._types()}"
        )
        assert (Path(self.repo) / "agent_out.txt").exists(), (
            "the work must still be in the user's checkout"
        )
        status = _run_git(self.repo, "status", "--porcelain").stdout.strip()
        assert "agent_out.txt" in status, (
            f"the work must be visible as an uncommitted change: {status}"
        )

    def test_autocommit_off_leaves_the_dirty_tree_alone(self) -> None:
        """``autoCommit: false`` must not commit in the user's checkout."""
        pre_head = _head_sha(self.repo)
        self._original_run = _patch_run("agent_out.txt", raises=None)
        self.server._run_task_inner({
            "prompt": "task with the toggle off",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        })

        assert pre_head == _head_sha(self.repo), (
            f"autoCommit=False still committed; events={self._types()}"
        )
        status = _run_git(self.repo, "status", "--porcelain").stdout.strip()
        assert "agent_out.txt" in status, (
            f"the work must be left in the checkout: {status}"
        )

    def test_clean_tree_emits_no_autocommit_events(self) -> None:
        """A task that changes nothing produces no autocommit events."""
        pre_head = _head_sha(self.repo)
        self._original_run = _patch_run(None, raises=None)
        self.server._run_task_inner({
            "prompt": "task that changes nothing",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": False,
            "autoCommit": True,
            "model": "",
        })

        assert pre_head == _head_sha(self.repo)
        types = self._types()
        assert "autocommit_done" not in types, (
            f"a clean tree must stay event-free; got: {types}"
        )
        assert "autocommit_progress" not in types, (
            f"a clean tree must stay event-free; got: {types}"
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
        branches = _list_kiss_wt_branches(self.repo)
        assert branches == [], (
            f"Auto-merge should delete the kiss/wt-* branch after "
            f"merging; remaining: {branches}"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
