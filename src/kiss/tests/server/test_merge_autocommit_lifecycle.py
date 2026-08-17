# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the full non-worktree task lifecycle.

With the interactive diff/merge review removed, a non-worktree task
that leaves the main working tree dirty is always auto-committed at
task end:

  agent modifies file → task ends → _main_dirty_files(work_dir)
  → _autocommit_changes → autocommit_progress events
  → autocommit_done (committed=True).

A clean tree (or a non-git folder) stays event-free.  The tests drive
the real :meth:`VSCodeServer._run_task_inner` against on-disk git
repositories, replacing the stateful agent's parent ``run`` with a
deterministic stub and the commit-message LLM call with a module-level
override (no mocks).

Also covers the work-dir plumbing bugs of the shared ``kiss-web``
daemon: ``_autocommit_changes``, ``generateCommitMessage`` and
``_main_dirty_files`` must act on the work dir carried by the
command / caller (the tab's own folder), not the daemon-wide
``self.work_dir``.
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
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    """Create a git repo with one committed file so HEAD exists."""
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "README.md").write_text("# Hello\n\nSome content\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "initial commit")


def _register_tab(
    tab_id: str, *, use_worktree: bool = False,
) -> agent_state.AgentState:
    """Register a server-owned AgentState for *tab_id* and return it."""
    state = agent_state.AgentState(
        f"task-{tab_id}", tab_id=tab_id, server_owned=True,
    )
    state.use_worktree = use_worktree
    agent_state.register(state)
    return state


def _make_server(work_dir: str) -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with captured events."""
    server = VSCodeServer()
    server.work_dir = work_dir
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _event_types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _find_event(events: list[dict], type_: str) -> dict:
    for e in events:
        if e["type"] == type_:
            return e
    raise AssertionError(f"No event of type {type_!r}: {_event_types(events)}")


class _LifecycleHarness(unittest.TestCase):
    """Real git repo + isolated persistence DB + stubbed agent run."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        self._db_tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self._db_tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

        self.server, self.events = _make_server(self.tmpdir)
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff

        def fake_gen(
            diff_text: str,
            user_prompt: str | None = None,
            task_result: str | None = None,
        ) -> str:
            return "chore: auto-commit test"

        _merge_flow_module.generate_commit_message_from_diff = fake_gen  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        agent_state.agent_states.clear()
        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self._db_tmpdir, ignore_errors=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_run(self, write: dict[str, str] | None) -> None:
        """Stub the agent to write *write* files into work_dir, succeed."""

        def stub_run(_agent: object, **kwargs: object) -> str:
            work_dir = kwargs.get("work_dir")
            if write and isinstance(work_dir, str) and work_dir:
                for name, content in write.items():
                    (Path(work_dir) / name).write_text(content)
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def _run_task(self, tab_id: str, *, auto_commit: bool = True) -> None:
        self.server._run_task_inner({
            "prompt": "make a change",
            "workDir": self.tmpdir,
            "tabId": tab_id,
            "useWorktree": False,
            "autoCommit": auto_commit,
            "model": "",
        })


class TestTaskEndAutocommitsModifiedFile(_LifecycleHarness):
    """A task that modifies a tracked file is committed at task end."""

    def test_modification_is_committed(self) -> None:
        """Modifying README.md yields autocommit_progress → done."""
        tab_id = "test-tab-1"
        pre_head = _git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        self._patch_run({"README.md": "# Hello\n\nUpdated by the agent\n"})

        self._run_task(tab_id)

        types = _event_types(self.events)
        assert "autocommit_progress" in types, types
        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is True
        assert done["committed"] is True
        assert done["tabId"] == tab_id
        assert types.index("autocommit_progress") < types.index(
            "autocommit_done",
        )

        post_head = _git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        assert pre_head != post_head, "the commit must land on the branch"
        status = _git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert status == "", f"Expected clean working tree, got: {status}"
        log = _git(self.tmpdir, "log", "-1", "--pretty=%s").stdout.strip()
        assert log == "chore: auto-commit test"

    def test_new_untracked_file_is_committed(self) -> None:
        """A brand-new untracked file is staged and committed too."""
        tab_id = "test-tab-2"
        self._patch_run({"new_file.txt": "brand new file\n"})

        self._run_task(tab_id)

        done = _find_event(self.events, "autocommit_done")
        assert done["committed"] is True
        show = _git(
            self.tmpdir, "show", "--name-only", "--pretty=", "HEAD",
        ).stdout
        assert "new_file.txt" in show

    def test_autocommit_toggle_off_leaves_the_change_uncommitted(self) -> None:
        """With the toggle off the change stays in the working tree.

        The user turned auto-commit off for this run, so the edit is
        theirs to review: it must show up as an ordinary modification
        instead of a commit, and no auto-commit strip is shown.
        """
        tab_id = "test-tab-toggle-off"
        pre_head = _git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        self._patch_run({"README.md": "# Toggle off\n"})

        self._run_task(tab_id, auto_commit=False)

        types = _event_types(self.events)
        assert "autocommit_done" not in types, types
        assert "autocommit_progress" not in types, types
        assert pre_head == _git(
            self.tmpdir, "rev-parse", "HEAD",
        ).stdout.strip()
        status = _git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert "README.md" in status, status


class TestTaskEndNoEventsWhenClean(_LifecycleHarness):
    """When the agent doesn't modify any files, no autocommit happens."""

    def test_no_events_when_no_changes(self) -> None:
        tab_id = "test-tab-noop"
        pre_head = _git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        self._patch_run(None)

        self._run_task(tab_id)

        types = _event_types(self.events)
        assert "autocommit_progress" not in types, types
        assert "autocommit_done" not in types, types
        post_head = _git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        assert pre_head == post_head

    def test_no_events_in_non_git_dir(self) -> None:
        """A non-git work dir must stay event-free (no failure toast)."""
        non_git = tempfile.mkdtemp()
        try:
            if _git(non_git, "rev-parse", "--is-inside-work-tree").returncode == 0:
                self.skipTest(f"{non_git} is inside a git repo")
            tab_id = "test-tab-nongit"
            self._patch_run({"loose.txt": "not tracked by anything\n"})
            self.server._run_task_inner({
                "prompt": "task in a non-git folder",
                "workDir": non_git,
                "tabId": tab_id,
                "useWorktree": False,
                "autoCommit": True,
                "model": "",
            })
            types = _event_types(self.events)
            assert "autocommit_progress" not in types, types
            assert "autocommit_done" not in types, types
            assert Path(non_git, "loose.txt").exists()
        finally:
            shutil.rmtree(non_git, ignore_errors=True)


class TestTaskEndCommitIncludesPreexistingDirt(_LifecycleHarness):
    """Pre-existing dirty files are committed along with agent work.

    ``git add -A`` stages the whole tree; with the pre-task snapshot
    machinery gone there is no distinction between user dirt and agent
    changes any more — everything dirty at task end is committed.
    """

    def test_pre_dirty_file_committed_with_agent_change(self) -> None:
        Path(self.tmpdir, "README.md").write_text("# Pre-existing dirty\n")
        tab_id = "test-tab-predirty"
        self._patch_run({"new_agent_file.py": "print('hello')\n"})

        self._run_task(tab_id)

        done = _find_event(self.events, "autocommit_done")
        assert done["committed"] is True
        show = _git(
            self.tmpdir, "show", "--name-only", "--pretty=", "HEAD",
        ).stdout
        assert "new_agent_file.py" in show
        assert "README.md" in show
        status = _git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert status == ""


class TestAutocommitUsesCommandWorkDir(unittest.TestCase):
    """``_autocommit_changes`` must act on the work_dir passed by the
    caller (the tab's own folder), not the daemon-wide ``self.work_dir``.

    Reproduces the reported bug: the shared ``kiss-web`` daemon was
    launched from (or synced to) a non-git folder, so the post-task
    autocommit reported ``"Not a git repository."`` even though the
    tab's workspace was a real git repository.  The task runner passes
    the tab's ``work_dir`` and the backend prefers it over
    ``self.work_dir``.
    """

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        _init_repo(self.repo)
        self.nongit = tempfile.mkdtemp()
        if _git(self.nongit, "rev-parse", "--is-inside-work-tree").returncode == 0:
            self.skipTest(f"{self.nongit} is inside a git repo")
        self.server, self.events = _make_server(self.nongit)
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff

        def fake_gen(
            diff_text: str,
            user_prompt: str | None = None,
            task_result: str | None = None,
        ) -> str:
            return "chore: auto-commit test"

        _merge_flow_module.generate_commit_message_from_diff = fake_gen  # type: ignore[assignment]

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.nongit, ignore_errors=True)
        agent_state.agent_states.clear()

    def test_commit_uses_caller_work_dir(self) -> None:
        """A dirty git repo passed via ``work_dir`` commits successfully
        even though ``self.work_dir`` is a non-git folder."""
        tab_id = "t-wd"
        _register_tab(tab_id)
        Path(self.repo, "new.txt").write_text("hello\n")

        self.server._autocommit_changes(tab_id, work_dir=self.repo)

        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is True, done
        assert done["committed"] is True, done
        status = _git(self.repo, "status", "--porcelain").stdout.strip()
        assert status == "", f"Expected clean tree, got: {status}"

    def test_commit_without_work_dir_still_fails_on_nongit(self) -> None:
        """Without a ``work_dir`` the handler falls back to the non-git
        ``self.work_dir`` and reports the original failure."""
        tab_id = "t-nowd"
        _register_tab(tab_id)
        self.server._autocommit_changes(tab_id)
        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is False, done
        assert done["message"] == "Not a git repository.", done


class TestCommitMessageUsesCommandWorkDir(unittest.TestCase):
    """``generateCommitMessage`` must act on the work_dir carried by the
    command (the tab's own folder), not the daemon-wide ``self.work_dir``.

    Reproduces the reported bug: the shared ``kiss-web`` daemon was
    launched from (or synced to) a non-git folder, so requesting a
    commit message reported ``"Not a git repository."`` even though the
    tab's workspace was a real git repository.  The frontend now stamps
    the tab's ``workDir`` on the ``generateCommitMessage`` command and
    the backend prefers it over ``self.work_dir``.
    """

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        _init_repo(self.repo)
        self.nongit = tempfile.mkdtemp()
        if _git(self.nongit, "rev-parse", "--is-inside-work-tree").returncode == 0:
            self.skipTest(f"{self.nongit} is inside a git repo")
        self.server, self.events = _make_server(self.nongit)

    def tearDown(self) -> None:
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.nongit, ignore_errors=True)
        agent_state.agent_states.clear()

    def test_no_git_error_with_command_work_dir(self) -> None:
        """A git repo passed via ``workDir`` is recognised even though
        ``self.work_dir`` is a non-git folder.

        With no staged changes the handler proceeds past the repo check
        and reports the (non-misleading) "no staged changes" error
        instead of "Not a git repository.".
        """
        tab_id = "t-cm"
        self.server._generate_commit_message(tab_id, work_dir=self.repo)
        msg = _find_event(self.events, "commitMessage")
        assert msg.get("error") != "Not a git repository.", msg
        assert "No staged changes" in msg.get("error", ""), msg

    def test_git_error_without_work_dir_on_nongit(self) -> None:
        """Without a ``workDir`` the handler falls back to the non-git
        ``self.work_dir`` and reports "Not a git repository."."""
        tab_id = "t-cm2"
        self.server._generate_commit_message(tab_id)
        msg = _find_event(self.events, "commitMessage")
        assert msg.get("error") == "Not a git repository.", msg

    def test_command_dispatch_forwards_work_dir(self) -> None:
        """``generateCommitMessage`` command forwards ``workDir`` so the
        repo check runs against the tab's folder, not ``self.work_dir``."""
        import time

        self.server._handle_command({
            "type": "generateCommitMessage",
            "tabId": "t-cm3",
            "workDir": self.repo,
        })
        for _ in range(50):
            if any(e["type"] == "commitMessage" for e in self.events):
                break
            time.sleep(0.05)
        msg = _find_event(self.events, "commitMessage")
        assert msg.get("error") != "Not a git repository.", msg


class TestMainDirtyFilesUsesCommandWorkDir(unittest.TestCase):
    """``_main_dirty_files`` must scan the work_dir passed by the caller
    (the tab's own folder), not the daemon-wide ``self.work_dir``.

    Mirrors the autocommit / commit-message fixes: the shared
    ``kiss-web`` daemon may run from a non-git folder, so the post-task
    dirty-file scan must use the tab's repository to detect changes.
    """

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        _init_repo(self.repo)
        self.nongit = tempfile.mkdtemp()
        if _git(self.nongit, "rev-parse", "--is-inside-work-tree").returncode == 0:
            self.skipTest(f"{self.nongit} is inside a git repo")
        self.server, self.events = _make_server(self.nongit)

    def tearDown(self) -> None:
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.nongit, ignore_errors=True)
        agent_state.agent_states.clear()

    def test_scan_uses_command_work_dir(self) -> None:
        """A dirty file in the command repo is reported even though
        ``self.work_dir`` is a non-git folder."""
        Path(self.repo, "new.txt").write_text("hello\n")
        changed = self.server._main_dirty_files(self.repo)
        assert "new.txt" in changed, changed

    def test_scan_without_work_dir_returns_empty_on_nongit(self) -> None:
        """Without a work_dir the scan falls back to the non-git
        ``self.work_dir`` and reports no changes."""
        Path(self.repo, "new.txt").write_text("hello\n")
        changed = self.server._main_dirty_files()
        assert changed == [], changed


if __name__ == "__main__":
    unittest.main()
