# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Non-worktree auto-commit must also commit repos other than work_dir.

The user-observed defect:

    "in the auto-commit and non worktree mode, if a task changes
    files, it does not auto-commit the files."

Root cause: at task end the auto-commit pass ran ``git add -A`` in
the repository containing the tab's *work_dir* — and nowhere else.  A
task is free to change files anywhere (the file tools take absolute
paths), so files it wrote in a DIFFERENT repository were silently left
uncommitted (observed in production: tasks with ``work_dir`` in one
project editing a sibling project's checkout).

The fix tracks the paths of ``Write`` / ``Edit`` tool calls per task
(in the printer, since event persistence is asynchronous), groups them
by containing repository at task end, and commits each extra
repository — staging ONLY the recorded paths so unrelated dirty state
in a repository the user never designated as work_dir is not swept in.

Each test drives the real :meth:`VSCodeServer._run_task_inner` against
fresh git repos, replacing only the stateful agent's parent ``run``
with a deterministic stub that changes files and reports the same
``tool_call`` events the real tool loop broadcasts (no mocks).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
import kiss.server.merge_flow as _merge_flow_module
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer

_COMMIT_MSG = "test: deterministic cross-repo commit message"


def _fixed_message(
    diff_text: str,
    user_prompt: str | None = None,
    task_result: str | None = None,
) -> str:
    """Deterministic stand-in for the LLM commit-message generator."""
    return _COMMIT_MSG


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


def _head_sha(repo: str) -> str:
    return _run_git(repo, "rev-parse", "HEAD").stdout.strip()


def _head_files(repo: str) -> list[str]:
    out = _run_git(repo, "show", "--name-only", "--format=", "HEAD").stdout
    return [line for line in out.splitlines() if line.strip()]


class _TwoRepoBase(unittest.TestCase):
    """Work-dir repo + a second independent repo, isolated DB."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-crossrepo-test-")
        self.repo = str(Path(self.tmpdir) / "workrepo")
        self.other = str(Path(self.tmpdir) / "otherrepo")
        for r in (self.repo, self.other):
            Path(r).mkdir(parents=True, exist_ok=True)
            _init_repo(r)

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
        orig_broadcast = self.server.printer.broadcast

        def capture(event: dict) -> None:
            self.events.append(event)
            orig_broadcast(event)

        # Wrap rather than replace: the real broadcast is what tracks
        # the per-task changed paths the fix under test consumes.
        self.server.printer.broadcast = capture  # type: ignore[assignment]

        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff
        _merge_flow_module.generate_commit_message_from_diff = _fixed_message

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen

        from kiss.server import agent_state
        with agent_state.STATE_LOCK:
            states = list(agent_state.agent_states.values())
        for state in states:
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
        self,
        body: Callable[[Any], None],
        raises: BaseException | None = None,
    ) -> None:
        """Replace the parent ``run`` with a stub executing *body*.

        *body* receives the agent instance; it changes files and
        reports them through the server printer exactly the way the
        real tool loop does (``printer.print(name, type="tool_call",
        tool_input=...)``).  The thread-local task id is already set
        by ``ChatSorcarAgent.run`` at this point, as in production.
        """
        def stub_run(self_agent: object, **kwargs: object) -> str:
            body(self_agent)
            if raises is not None:
                raise raises
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def _report_write(self, path: Path, content: str) -> None:
        """Write *path* and broadcast the matching Write tool_call."""
        path.write_text(content)
        self.server.printer.print(
            "Write", type="tool_call",
            tool_input={"file_path": str(path), "content": content},
        )

    def _submit(self, *, auto_commit: bool = True) -> None:
        self.server._run_task_inner({
            "prompt": "task that edits another repo",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": False,
            "autoCommit": auto_commit,
            "model": "",
        })


class TestCrossRepoAutoCommit(_TwoRepoBase):
    """Files changed outside the work_dir repo must be committed."""

    def test_changes_in_another_repo_are_committed(self) -> None:
        """The reported defect: a Write into a second repo, task
        succeeds, auto-commit ON — the second repo must gain a commit
        containing exactly that file."""
        target = Path(self.other) / "made-by-task.txt"
        pre = _head_sha(self.other)
        self._patch_run(lambda agent: self._report_write(target, "hi\n"))
        self._submit()

        post = _head_sha(self.other)
        assert pre != post, (
            f"Other repo must gain a commit; events={self._types()}"
        )
        assert _head_files(self.other) == ["made-by-task.txt"]
        msg = _run_git(self.other, "log", "-1", "--format=%s").stdout.strip()
        assert msg == _COMMIT_MSG
        status = _run_git(self.other, "status", "--porcelain").stdout
        assert status.strip() == ""
        done = [
            e for e in self.events
            if e["type"] == "autocommit_done" and e.get("committed")
            and "otherrepo" in e.get("message", "")
        ]
        assert done, f"No autocommit_done for other repo: {self._types()}"

    def test_unrelated_dirty_files_in_other_repo_stay_uncommitted(
        self,
    ) -> None:
        """Only the recorded paths enter the commit — pre-existing
        dirty state in the other repo must survive untouched."""
        Path(self.other, "unrelated.txt").write_text("user's own edit\n")
        Path(self.other, "seed.txt").write_text("user's tracked edit\n")
        target = Path(self.other) / "made-by-task.txt"
        self._patch_run(lambda agent: self._report_write(target, "hi\n"))
        self._submit()

        assert _head_files(self.other) == ["made-by-task.txt"]
        status = _run_git(self.other, "status", "--porcelain").stdout
        assert "unrelated.txt" in status
        assert "seed.txt" in status

    def test_no_cross_repo_commit_when_autocommit_off(self) -> None:
        target = Path(self.other) / "made-by-task.txt"
        pre = _head_sha(self.other)
        self._patch_run(lambda agent: self._report_write(target, "hi\n"))
        self._submit(auto_commit=False)

        assert _head_sha(self.other) == pre
        assert target.exists()

    def test_no_cross_repo_commit_when_task_fails(self) -> None:
        target = Path(self.other) / "made-by-task.txt"
        pre = _head_sha(self.other)
        self._patch_run(
            lambda agent: self._report_write(target, "hi\n"),
            raises=RuntimeError("boom"),
        )
        self._submit()

        assert _head_sha(self.other) == pre
        assert target.exists()

    def test_deleted_file_in_other_repo_is_committed(self) -> None:
        """``git add -A -- <path>`` records deletions too."""
        victim = Path(self.other) / "seed.txt"

        def body(agent: Any) -> None:
            victim.unlink()
            self.server.printer.print(
                "Edit", type="tool_call",
                tool_input={"file_path": str(victim)},
            )

        pre = _head_sha(self.other)
        self._patch_run(body)
        self._submit()

        assert _head_sha(self.other) != pre
        assert _head_files(self.other) == ["seed.txt"]
        out = _run_git(self.other, "ls-files", "seed.txt").stdout
        assert out.strip() == ""

    def test_path_outside_any_repo_is_ignored(self) -> None:
        """A write into a plain directory has no repo to commit to."""
        outside = Path(self.tmpdir) / "norepo"
        outside.mkdir()
        target = outside / "note.txt"
        self._patch_run(lambda agent: self._report_write(target, "x\n"))
        self._submit()

        assert target.exists()
        failed = [
            e for e in self.events
            if e["type"] == "autocommit_done" and not e.get("success", True)
        ]
        assert failed == [], f"No failure toast expected: {failed}"

    def test_kiss_worktree_checkout_is_left_alone(self) -> None:
        """Writes into a ``.kiss-worktrees`` checkout belong to the
        worktree merge flow; this pass must not commit there."""
        wt_dir = Path(self.other) / ".kiss-worktrees" / "kiss_wt-test"
        wt_dir.parent.mkdir(parents=True)
        res = _run_git(
            self.other, "worktree", "add",
            str(wt_dir), "-b", "kiss/wt-test",
        )
        assert res.returncode == 0, res.stderr
        wt_head_pre = _run_git(str(wt_dir), "rev-parse", "HEAD").stdout
        target = wt_dir / "wt-file.txt"
        self._patch_run(lambda agent: self._report_write(target, "x\n"))
        self._submit()

        wt_head_post = _run_git(str(wt_dir), "rev-parse", "HEAD").stdout
        assert wt_head_pre == wt_head_post
        status = _run_git(str(wt_dir), "status", "--porcelain").stdout
        assert "wt-file.txt" in status

    def test_work_dir_repo_is_not_committed_twice(self) -> None:
        """A recorded path inside the work_dir repo is the main
        pass's job; the cross-repo pass must skip it."""
        target = Path(self.repo) / "in-work-repo.txt"
        pre_count = _run_git(
            self.repo, "rev-list", "--count", "HEAD",
        ).stdout.strip()
        self._patch_run(lambda agent: self._report_write(target, "x\n"))
        self._submit()

        post_count = _run_git(
            self.repo, "rev-list", "--count", "HEAD",
        ).stdout.strip()
        assert int(post_count) == int(pre_count) + 1, (
            f"Exactly one commit expected in work repo; "
            f"{pre_count} -> {post_count}"
        )

    def test_subtask_paths_from_db_are_committed(self) -> None:
        """Files a sub-task changed (persisted under its own task id,
        linked by parent_task_id) are committed too."""
        target = Path(self.other) / "by-subtask.txt"

        def body(agent: Any) -> None:
            parent_id = str(self.server.printer._thread_local.task_id)
            sub_id, _ = _persistence._add_task(
                "sub work", extra={"parent_task_id": parent_id},
            )
            target.write_text("sub\n")
            _persistence._append_chat_event(
                {"type": "tool_call", "name": "Write", "path": str(target)},
                task_id=sub_id,
            )

        pre = _head_sha(self.other)
        self._patch_run(body)
        self._submit()

        assert _head_sha(self.other) != pre
        assert _head_files(self.other) == ["by-subtask.txt"]

    def test_user_staged_changes_in_other_repo_stay_staged(self) -> None:
        """A pathspec-limited commit must not sweep entries the USER
        had already staged in the other repo into the task's commit."""
        Path(self.other, "seed.txt").write_text("user's staged edit\n")
        _run_git(self.other, "add", "seed.txt")
        target = Path(self.other) / "made-by-task.txt"
        self._patch_run(lambda agent: self._report_write(target, "hi\n"))
        self._submit()

        assert _head_files(self.other) == ["made-by-task.txt"]
        staged = _run_git(
            self.other, "diff", "--cached", "--name-only",
        ).stdout.split()
        assert staged == ["seed.txt"], (
            f"User's staged entry must survive; staged={staged}"
        )

    def test_detached_head_repo_is_left_uncommitted(self) -> None:
        """A commit no branch points at (e.g. a submodule checkout,
        detached by default) would be unreachable — refuse it."""
        head = _head_sha(self.other)
        _run_git(self.other, "checkout", "-q", "--detach", head)
        target = Path(self.other) / "made-by-task.txt"
        self._patch_run(lambda agent: self._report_write(target, "hi\n"))
        self._submit()

        assert _head_sha(self.other) == head
        status = _run_git(self.other, "status", "--porcelain").stdout
        assert "made-by-task.txt" in status
        toasts = [
            e for e in self.events
            if e["type"] == "autocommit_done"
            and "detached" in e.get("message", "")
        ]
        assert toasts, f"Expected a detached-HEAD toast: {self._types()}"

    def test_relative_path_is_resolved_against_work_dir(self) -> None:
        """The file tools accept relative paths (cwd = work_dir); a
        recorded ``../otherrepo/...`` must reach the other repo."""
        rel = "../otherrepo/rel.txt"
        target = Path(self.repo) / rel

        def body(agent: Any) -> None:
            target.write_text("rel\n")
            self.server.printer.print(
                "Write", type="tool_call",
                tool_input={"file_path": rel, "content": "rel\n"},
            )

        pre = _head_sha(self.other)
        self._patch_run(body)
        self._submit()

        assert _head_sha(self.other) != pre
        assert _head_files(self.other) == ["rel.txt"]

    def test_sequential_task_tags_all_commit(self) -> None:
        """A multi-``<task>`` submission persists (and cleans up) each
        non-final task separately; files changed by EARLIER tasks must
        still reach the cross-repo commit at the end of the run."""
        calls: list[int] = []

        def body(agent: Any) -> None:
            n = len(calls)
            calls.append(n)
            self._report_write(
                Path(self.other) / f"task-{n}.txt", f"t{n}\n",
            )

        self._patch_run(body)
        self.server._run_task_inner({
            "prompt": "<task>first</task><task>second</task>",
            "workDir": self.repo,
            "tabId": "0",
            "useWorktree": False,
            "autoCommit": True,
            "model": "",
        })

        assert len(calls) == 2, f"Expected two sequential runs: {calls}"
        committed = set(_head_files(self.other))
        assert committed == {"task-0.txt", "task-1.txt"}, (
            f"Both tasks' files must be committed; got {committed}"
        )

    def test_subtask_printer_entry_is_popped_not_leaked(self) -> None:
        """A sub-agent's in-memory record must be consumed (its id is
        popped) — otherwise every file-changing sub-agent would leak
        one entry for the daemon's lifetime — and its not-yet-persisted
        paths must still be committed."""
        target = Path(self.other) / "by-subagent-memory.txt"

        def body(agent: Any) -> None:
            printer = self.server.printer
            parent_id = str(printer._thread_local.task_id)
            sub_id, _ = _persistence._add_task(
                "sub work", extra={"parent_task_id": parent_id},
            )
            printer._thread_local.task_id = sub_id
            try:
                self._report_write(target, "sub\n")
            finally:
                printer._thread_local.task_id = parent_id

        pre = _head_sha(self.other)
        self._patch_run(body)
        self._submit()

        assert _head_sha(self.other) != pre
        assert _head_files(self.other) == ["by-subagent-memory.txt"]
        assert self.server.printer._changed_paths == {}, (
            f"Leaked entries: {self.server.printer._changed_paths}"
        )

    def test_db_walk_failure_still_commits_the_tasks_own_paths(
        self,
    ) -> None:
        """A failure loading the sub-task record costs only the
        sub-task paths — the task's own changes are still committed."""
        target = Path(self.other) / "made-by-task.txt"
        orig = _persistence._descendant_task_ids

        def boom(root_task_id: str) -> list[str]:
            raise RuntimeError("db unavailable")

        _persistence._descendant_task_ids = boom
        try:
            pre = _head_sha(self.other)
            self._patch_run(
                lambda agent: self._report_write(target, "hi\n"),
            )
            self._submit()
        finally:
            _persistence._descendant_task_ids = orig

        assert _head_sha(self.other) != pre
        assert _head_files(self.other) == ["made-by-task.txt"]

    def test_glob_filename_cannot_match_other_files(self) -> None:
        """A recorded filename containing pathspec magic (``*``) must
        match itself literally, never other dirty files."""
        Path(self.other, "starXX.txt").write_text("not the task's\n")
        target = Path(self.other) / "star*.txt"
        self._patch_run(lambda agent: self._report_write(target, "hi\n"))
        self._submit()

        assert _head_files(self.other) == ["star*.txt"]
        status = _run_git(self.other, "status", "--porcelain").stdout
        assert "starXX.txt" in status

    def test_users_staged_rename_source_is_not_swept_in(self) -> None:
        """When the task writes the NEW side of a rename the user had
        staged, the rename's old-path deletion must stay the user's
        staged business, not enter the task's commit."""
        _run_git(self.other, "mv", "seed.txt", "moved.txt")
        target = Path(self.other) / "moved.txt"
        self._patch_run(
            lambda agent: self._report_write(target, "task content\n"),
        )
        self._submit()

        assert _head_files(self.other) == ["moved.txt"]
        staged = _run_git(
            self.other, "diff", "--cached", "--name-status",
        ).stdout
        assert "seed.txt" in staged, (
            f"User's staged deletion must survive; staged={staged!r}"
        )

    def test_autocommit_done_event_is_persisted(self) -> None:
        """The cross-repo commit must be replayable from the DB."""
        target = Path(self.other) / "made-by-task.txt"
        self._patch_run(lambda agent: self._report_write(target, "hi\n"))
        self._submit()

        db = _persistence._get_db()
        rows = db.execute(
            "SELECT event_json FROM events WHERE event_json LIKE "
            "'%autocommit_done%'",
        ).fetchall()
        assert any("otherrepo" in r["event_json"] for r in rows), (
            "autocommit_done for the other repo must be persisted"
        )


class TestChangedPathTracking(unittest.TestCase):
    """The printer's per-task record of file-mutating tool calls."""

    def setUp(self) -> None:
        self.printer = JsonPrinter()
        self.printer._thread_local.task_id = "t-track"

    def _tool_call(self, name: str, path: str) -> None:
        self.printer.print(
            name, type="tool_call", tool_input={"file_path": path},
        )

    def test_write_and_edit_paths_tracked_and_popped(self) -> None:
        self._tool_call("Write", "/tmp/a.txt")
        self._tool_call("Edit", "/tmp/b.txt")
        assert self.printer.pop_changed_paths("t-track") == {
            "/tmp/a.txt", "/tmp/b.txt",
        }

    def test_read_and_bash_are_not_tracked(self) -> None:
        self._tool_call("Read", "/tmp/a.txt")
        self.printer.print(
            "Bash", type="tool_call", tool_input={"command": "touch /tmp/x"},
        )
        assert self.printer.pop_changed_paths("t-track") == set()

    def test_pop_clears_the_record(self) -> None:
        self._tool_call("Write", "/tmp/a.txt")
        assert self.printer.pop_changed_paths("t-track")
        assert self.printer.pop_changed_paths("t-track") == set()

    def test_paths_are_kept_per_task(self) -> None:
        self._tool_call("Write", "/tmp/a.txt")
        self.printer._thread_local.task_id = "t-other"
        self._tool_call("Write", "/tmp/b.txt")
        assert self.printer.pop_changed_paths("t-track") == {"/tmp/a.txt"}
        assert self.printer.pop_changed_paths("t-other") == {"/tmp/b.txt"}

    def test_cleanup_task_frees_untaken_paths(self) -> None:
        self._tool_call("Write", "/tmp/a.txt")
        self.printer.cleanup_task("t-track")
        assert self.printer.pop_changed_paths("t-track") == set()

    def test_no_task_id_tracks_nothing(self) -> None:
        self.printer._thread_local.task_id = ""
        self._tool_call("Write", "/tmp/a.txt")
        assert self.printer.pop_changed_paths("") == set()


class TestWebPrinterChangedPathTracking(unittest.TestCase):
    """WebPrinter.broadcast must feed the same changed-path record.

    The remote-access daemon installs WebPrinter (a JsonPrinter
    subclass whose ``broadcast`` override re-implements the recording
    path) in place of the base printer; without a mirrored
    ``_track_changed_path`` call the cross-repo auto-commit saw no
    changed paths for tasks run through it (gpt-5.6-sol merge-review
    finding).
    """

    def _broadcast(self, printer: object, name: str, path: str, task: str) -> None:
        from kiss.server.web_server import WebPrinter
        assert isinstance(printer, WebPrinter)
        printer.broadcast({
            "type": "tool_call",
            "name": name,
            "path": path,
            "taskId": task,
        })

    def test_write_path_tracked_through_web_printer(self) -> None:
        from kiss.server.web_server import WebPrinter

        printer = WebPrinter()
        self._broadcast(printer, "Write", "/tmp/web-tracked.txt", "t-web-track")
        assert printer.pop_changed_paths("t-web-track") == {
            "/tmp/web-tracked.txt",
        }

    def test_read_not_tracked_through_web_printer(self) -> None:
        from kiss.server.web_server import WebPrinter

        printer = WebPrinter()
        self._broadcast(printer, "Read", "/tmp/web-read.txt", "t-web-track-2")
        assert printer.pop_changed_paths("t-web-track-2") == set()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
