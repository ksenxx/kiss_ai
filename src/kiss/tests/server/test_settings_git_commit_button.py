# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the settings panel's "Git Commit" button.

The button (``#autocommit-btn`` in ``media/chat.html``) posts an
``autocommitAction`` command carrying the tab's id and working
directory.  The command flows through the Sorcar API catalog
(``kiss.server.sorcar.API``) to the backend dispatcher
(``VSCodeServer._HANDLERS``), which stages the tab's working tree,
generates a commit message from the staged diff alone, commits, and
reports through toast ``notification`` events plus a terminal
``autocommit_done`` event marked ``manual: True`` (no
``autocommit_progress`` transcript events — the chat stays clean;
only a failure's reason is rendered there).

Contract locked in here:

* ``autocommitAction`` exists in the server API catalog, in the
  browser catalog (``media/api.js``), and in the backend dispatcher.
* Dispatching it against a dirty real git repository creates a real
  commit and broadcasts ``autocommit_done`` with ``committed=True``.
* A clean tree is a no-op reported as "Nothing to commit.".
* A non-git working directory is a failure, not a crash.
* While a non-worktree task is running in the repository the commit
  is refused (``git add -A`` would snapshot a half-written agent
  state), while a task in a different repository does not block it.
* While a tab's autocommit is in flight, duplicate clicks are dropped
  (no second commit), and the claim is released afterwards.
* The webview wires the button: the markup is in ``chat.html`` and
  ``main.js`` posts ``autocommitAction`` with tab id and work dir.

No mocks, patches, fakes, or test doubles: real git repositories and
a real ``VSCodeServer``.  The commit-message LLM helper falls back to
``"kiss: auto-commit agent work"`` when no model is reachable, so the
commit itself is always real.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer
from kiss.server.sorcar import API

_DONE_TIMEOUT_S = 120.0


def _git(cwd: str | Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout


def _make_repo(path: Path) -> None:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    (path / "f.txt").write_text("one\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")


class TestGitCommitCommandCatalog(unittest.TestCase):
    """``autocommitAction`` is present in every command catalog."""

    def test_in_server_api_catalog(self) -> None:
        self.assertIn("autocommitAction", API)
        self.assertEqual(API["autocommitAction"].handler, "forward")
        self.assertEqual(API["autocommitAction"].required, ())

    def test_in_backend_dispatcher(self) -> None:
        self.assertIn("autocommitAction", VSCodeServer._HANDLERS)


class TestGitCommitCommandEndToEnd(unittest.TestCase):
    """Dispatching ``autocommitAction`` commits the tab's tree."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-gitcommit-btn-")
        self.repo = Path(self.tmpdir) / "repo"
        _make_repo(self.repo)
        self.server = VSCodeServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]
        self.tab_id = "t-gitcommit-btn"

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _dispatch(self, work_dir: str | None = None) -> None:
        self.server._handle_command({
            "type": "autocommitAction",
            "tabId": self.tab_id,
            "workDir": str(self.repo) if work_dir is None else work_dir,
        })

    def _wait_done(self, count: int = 1) -> list[dict]:
        deadline = time.monotonic() + _DONE_TIMEOUT_S
        while time.monotonic() < deadline:
            done = [
                e for e in self.events if e.get("type") == "autocommit_done"
            ]
            if len(done) >= count:
                return done
            time.sleep(0.05)
        raise AssertionError(
            f"no autocommit_done after {_DONE_TIMEOUT_S}s: {self.events}",
        )

    def test_dirty_tree_is_committed(self) -> None:
        """The button's command stages and commits real changes."""
        (self.repo / "f.txt").write_text("one\ntwo\n")
        (self.repo / "new.txt").write_text("brand new\n")
        before = _git(self.repo, "rev-parse", "HEAD").strip()

        self._dispatch()
        done = self._wait_done()

        self.assertEqual(len(done), 1, self.events)
        self.assertTrue(done[0]["success"], done[0])
        self.assertTrue(done[0]["committed"], done[0])
        self.assertEqual(done[0]["tabId"], self.tab_id)
        self.assertTrue(done[0]["message"].startswith("Committed: "), done[0])
        after = _git(self.repo, "rev-parse", "HEAD").strip()
        self.assertNotEqual(before, after, "a new commit must exist")
        status = _git(self.repo, "status", "--porcelain", "-uall")
        self.assertEqual(status.strip(), "", "working tree must be clean")
        shown = _git(self.repo, "show", "--stat", "--format=%B", "HEAD")
        self.assertIn("new.txt", shown)
        progress = [
            e for e in self.events if e.get("type") == "autocommit_progress"
        ]
        self.assertEqual(
            progress, [],
            "a manual commit must not write progress into the chat",
        )
        notes = [
            e for e in self.events if e.get("type") == "notification"
        ]
        self.assertTrue(notes, "toast notifications must be broadcast")
        self.assertEqual(
            notes[0]["message"], "Auto-generating commit message…",
        )
        self.assertEqual(notes[0]["tabId"], self.tab_id)
        self.assertTrue(
            notes[-1]["message"].startswith("Committed: "), notes[-1],
        )
        self.assertTrue(done[0].get("manual"), done[0])
        with self.server._state_lock:
            self.assertNotIn(
                self.tab_id, self.server._autocommit_tabs,
                "the in-flight claim must be released after the commit",
            )

    def test_clean_tree_is_a_noop(self) -> None:
        """A clean tree reports "Nothing to commit." and no new commit."""
        before = _git(self.repo, "rev-parse", "HEAD").strip()
        self._dispatch()
        done = self._wait_done()
        self.assertTrue(done[0]["success"], done[0])
        self.assertFalse(done[0]["committed"], done[0])
        self.assertEqual(done[0]["message"], "Nothing to commit.")
        self.assertEqual(before, _git(self.repo, "rev-parse", "HEAD").strip())

    def test_non_git_dir_reports_failure(self) -> None:
        """A non-git working directory fails cleanly."""
        plain = Path(self.tmpdir) / "plain"
        plain.mkdir()
        self._dispatch(work_dir=str(plain))
        done = self._wait_done()
        self.assertFalse(done[0]["success"], done[0])
        self.assertFalse(done[0]["committed"], done[0])
        self.assertEqual(done[0]["message"], "Not a git repository.")

    def _mark_running_non_wt_task(self, repo_root: Path | None) -> None:
        """Register another tab running a non-worktree task.

        Mirrors what ``_run_task_inner`` records for a running
        non-worktree task: ``is_running_non_wt = True`` plus the
        resolved repo root of its ``work_dir``.
        """
        with self.server._state_lock:
            state = AgentState(
                "other-task-key", tab_id="t-other", server_owned=True,
            )
            agent_state.register(state)
            state.is_task_active = True
            state.is_running_non_wt = True
            state.non_wt_repo_root = (
                repo_root.resolve() if repo_root else None
            )

    def test_running_task_in_same_repo_refuses_commit(self) -> None:
        """A running non-worktree task blocks the manual commit: the
        half-written agent state must not be snapshotted."""
        (self.repo / "f.txt").write_text("half-written agent state\n")
        self._mark_running_non_wt_task(self.repo)
        before = _git(self.repo, "rev-parse", "HEAD").strip()

        self._dispatch()
        done = self._wait_done()

        self.assertFalse(done[0]["success"], done[0])
        self.assertFalse(done[0]["committed"], done[0])
        self.assertIn("task is still running", done[0]["message"])
        self.assertEqual(before, _git(self.repo, "rev-parse", "HEAD").strip())
        status = _git(self.repo, "status", "--porcelain", "-uall")
        self.assertNotEqual(status.strip(), "", "changes must stay dirty")

    def test_running_task_in_other_repo_does_not_block(self) -> None:
        """A task running in a different repository must not block."""
        other = Path(self.tmpdir) / "other-repo"
        _make_repo(other)
        self._mark_running_non_wt_task(other)
        (self.repo / "f.txt").write_text("one\ntwo\n")

        self._dispatch()
        done = self._wait_done()

        self.assertTrue(done[0]["success"], done[0])
        self.assertTrue(done[0]["committed"], done[0])

    def test_duplicate_clicks_are_dropped_while_in_flight(self) -> None:
        """A second click during an in-flight autocommit is ignored."""
        (self.repo / "f.txt").write_text("one\ntwo\n")
        with self.server._state_lock:
            self.server._autocommit_tabs.add(self.tab_id)
        try:
            self._dispatch()
            time.sleep(0.3)
            self.assertEqual(
                [e for e in self.events if e.get("type", "").startswith(
                    "autocommit",
                )],
                [],
                "a duplicate click must be dropped silently",
            )
            status = _git(self.repo, "status", "--porcelain", "-uall")
            self.assertNotEqual(
                status.strip(), "",
                "the dropped duplicate must not have committed anything",
            )
        finally:
            with self.server._state_lock:
                self.server._autocommit_tabs.discard(self.tab_id)
        # Once the claim is released, the button works again.
        self._dispatch()
        done = self._wait_done()
        self.assertTrue(done[0]["committed"], done[0])


if __name__ == "__main__":
    unittest.main()
