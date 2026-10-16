# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the manual Git Commit flow of the daemon.

The settings panel's Git Commit button runs
:meth:`AgentServer._commit_main_tree`:

  git add -A → generate_commit_message_from_diff(staged diff)
  → git commit → notification + autocommit_done (manual=True).

A clean tree reports "Nothing to commit." without committing.  The
tests drive the real handler against on-disk git repositories,
replacing only the commit-message LLM call with a module-level
override (no mocks).

Also covers the work-dir plumbing bug of the shared ``kiss-web``
daemon: ``_commit_main_tree`` must act on the work dir carried by
the command / caller (the tab's own folder), not the daemon-wide
``self.work_dir``.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path

import kiss.server.merge_flow as _merge_flow_module
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import run_git


def _init_repo(repo: str) -> None:
    """Create a git repo with one committed file so HEAD exists."""
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Test User")
    run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "README.md").write_text("# Hello\n\nSome content\n")
    run_git(repo, "add", "README.md")
    run_git(repo, "commit", "-q", "-m", "initial commit")


def _register_tab(tab_id: str) -> agent_state.AgentState:
    """Register a server-owned AgentState for *tab_id* and return it."""
    state = agent_state.AgentState(
        f"task-{tab_id}", tab_id=tab_id, server_owned=True,
    )
    agent_state.register(state)
    return state


def _make_server(work_dir: str) -> tuple[AgentServer, list[dict]]:
    """Create a AgentServer with captured events."""
    server = AgentServer()
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


def _fake_gen(
    diff_text: str,
    user_prompt: str | None = None,
    task_result: str | None = None,
) -> str:
    """Deterministic stand-in for the LLM commit-message generator."""
    del diff_text, user_prompt, task_result
    return "chore: manual commit test"


class _ManualCommitHarness(unittest.TestCase):
    """Real git repo + non-git daemon work dir + offline commit messages."""

    def setUp(self) -> None:
        self.repo = tempfile.mkdtemp()
        _init_repo(self.repo)
        self.nongit = tempfile.mkdtemp()
        if run_git(self.nongit, "rev-parse", "--is-inside-work-tree").returncode == 0:
            self.skipTest(f"{self.nongit} is inside a git repo")
        self.server, self.events = _make_server(self.nongit)
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff
        _merge_flow_module.generate_commit_message_from_diff = _fake_gen  # type: ignore[assignment]

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.nongit, ignore_errors=True)
        agent_state.agent_states.clear()


class TestManualCommitCommitsDirtyTree(_ManualCommitHarness):
    """The Git Commit button stages and commits everything dirty."""

    def test_modification_is_committed(self) -> None:
        """Modifying README.md yields a progress toast → done event."""
        tab_id = "test-tab-1"
        _register_tab(tab_id)
        pre_head = run_git(self.repo, "rev-parse", "HEAD").stdout.strip()
        Path(self.repo, "README.md").write_text("# Hello\n\nUpdated by the user\n")

        self.server._commit_main_tree(tab_id, work_dir=self.repo)

        types = _event_types(self.events)
        assert "notification" in types, types
        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is True, done
        assert done["committed"] is True, done
        assert done["manual"] is True, done
        assert done["tabId"] == tab_id
        assert done["workDir"] == self.repo
        assert done["commitMessage"] == "chore: manual commit test"
        assert types.index("notification") < types.index("autocommit_done")

        post_head = run_git(self.repo, "rev-parse", "HEAD").stdout.strip()
        assert pre_head != post_head, "the commit must land on the branch"
        status = run_git(self.repo, "status", "--porcelain").stdout.strip()
        assert status == "", f"Expected clean working tree, got: {status}"
        log = run_git(self.repo, "log", "-1", "--pretty=%s").stdout.strip()
        assert log == "chore: manual commit test"

    def test_new_untracked_file_is_committed(self) -> None:
        """A brand-new untracked file is staged and committed too."""
        tab_id = "test-tab-2"
        _register_tab(tab_id)
        Path(self.repo, "new_file.txt").write_text("brand new file\n")

        self.server._commit_main_tree(tab_id, work_dir=self.repo)

        done = _find_event(self.events, "autocommit_done")
        assert done["committed"] is True, done
        show = run_git(self.repo, "show", "--name-only", "--pretty=", "HEAD").stdout
        assert "new_file.txt" in show

    def test_pre_dirty_file_committed_with_new_file(self) -> None:
        """``git add -A`` stages the whole tree: every dirty file lands."""
        tab_id = "test-tab-predirty"
        _register_tab(tab_id)
        Path(self.repo, "README.md").write_text("# Pre-existing dirty\n")
        Path(self.repo, "new_agent_file.py").write_text("print('hello')\n")

        self.server._commit_main_tree(tab_id, work_dir=self.repo)

        done = _find_event(self.events, "autocommit_done")
        assert done["committed"] is True, done
        show = run_git(self.repo, "show", "--name-only", "--pretty=", "HEAD").stdout
        assert "new_agent_file.py" in show
        assert "README.md" in show


class TestManualCommitNothingToCommit(_ManualCommitHarness):
    """A clean tree is a cheap no-op and a non-git folder is an error."""

    def test_clean_tree_reports_nothing_to_commit(self) -> None:
        tab_id = "test-tab-noop"
        _register_tab(tab_id)
        pre_head = run_git(self.repo, "rev-parse", "HEAD").stdout.strip()

        self.server._commit_main_tree(tab_id, work_dir=self.repo)

        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is True, done
        assert done["committed"] is False, done
        assert done["message"] == "Nothing to commit.", done
        assert "commitMessage" not in done, done
        assert pre_head == run_git(self.repo, "rev-parse", "HEAD").stdout.strip()

    def test_non_git_dir_reports_failure(self) -> None:
        """A non-git work dir reports "Not a git repository." (error toast)."""
        tab_id = "test-tab-nongit"
        _register_tab(tab_id)
        Path(self.nongit, "loose.txt").write_text("not tracked by anything\n")

        self.server._commit_main_tree(tab_id, work_dir=self.nongit)

        toast = _find_event(self.events, "notification")
        assert toast["severity"] == "error", toast
        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is False, done
        assert done["committed"] is False, done
        assert done["message"] == "Not a git repository.", done
        assert Path(self.nongit, "loose.txt").exists()


class TestManualCommitUsesCommandWorkDir(_ManualCommitHarness):
    """``_commit_main_tree`` must act on the work_dir passed by the
    caller (the tab's own folder), not the daemon-wide ``self.work_dir``.

    Reproduces the reported bug: the shared ``kiss-web`` daemon was
    launched from (or synced to) a non-git folder, so the commit
    reported ``"Not a git repository."`` even though the tab's
    workspace was a real git repository.  The command carries the
    tab's ``work_dir`` and the backend prefers it over
    ``self.work_dir``.
    """

    def test_commit_uses_caller_work_dir(self) -> None:
        """A dirty git repo passed via ``work_dir`` commits successfully
        even though ``self.work_dir`` is a non-git folder."""
        tab_id = "t-wd"
        _register_tab(tab_id)
        Path(self.repo, "new.txt").write_text("hello\n")

        self.server._commit_main_tree(tab_id, work_dir=self.repo)

        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is True, done
        assert done["committed"] is True, done
        status = run_git(self.repo, "status", "--porcelain").stdout.strip()
        assert status == "", f"Expected clean tree, got: {status}"

    def test_commit_without_work_dir_still_fails_on_nongit(self) -> None:
        """Without a ``work_dir`` the handler falls back to the non-git
        ``self.work_dir`` and reports the original failure."""
        tab_id = "t-nowd"
        _register_tab(tab_id)
        self.server._commit_main_tree(tab_id)
        done = _find_event(self.events, "autocommit_done")
        assert done["success"] is False, done
        assert done["message"] == "Not a git repository.", done
        assert done["workDir"] == self.nongit, done


if __name__ == "__main__":
    unittest.main()
