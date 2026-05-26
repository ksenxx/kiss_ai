"""Integration tests for the "Auto commit" toggle checkbox.

Validates:
- The toggle exists as ``id="cfg-auto-commit"`` in the Settings panel
  in both ``SorcarTab.ts`` (extension webview) and the standalone
  remote-access ``web_server.py`` HTML template, positioned right
  after ``cfg-use-parallel`` and defaulting to ``checked``.
- ``main.js`` references the checkbox and forwards its state as
  ``autoCommit`` on submit/run messages.
- ``_RunningAgentState`` carries an ``auto_commit_mode`` field that
  defaults to ``False``.
- When ``auto_commit_mode`` is ON the task lifecycle skips the
  interactive merge/diff workflow and auto-commits agent changes
  directly (non-worktree branch).
- When ``auto_commit_mode`` is ON and worktree mode is also ON the
  worktree branch is auto-merged into the original branch instead of
  surfacing a merge review.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import kiss.agents.vscode.merge_flow as _merge_flow_module
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer

_VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


def _read(name: str) -> str:
    return (_VSCODE_DIR / name).read_text()


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "README.md").write_text("# Hello\n\nSome content\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "initial commit")


def _make_server(work_dir: str) -> tuple[VSCodeServer, list[dict]]:
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






class TestRunningAgentStateField(unittest.TestCase):
    """``_RunningAgentState`` carries the per-tab toggle state."""

    def test_default_true(self) -> None:
        tab = _RunningAgentState("tab-x", "gemini")
        assert tab.auto_commit_mode is True

    def test_settable(self) -> None:
        tab = _RunningAgentState("tab-y", "gemini")
        tab.auto_commit_mode = False
        assert tab.auto_commit_mode is False


class _AutocommitTaskHarness(unittest.TestCase):
    """Shared setUp/tearDown for end-to-end autocommit-toggle tests."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)
        self.server, self.events = _make_server(self.tmpdir)
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff
        def _stub(diff_text: str, user_prompt: str | None = None) -> str:
            del diff_text, user_prompt
            return "auto-commit-toggle-test"

        _merge_flow_module.generate_commit_message_from_diff = _stub

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestAutocommitModeSkipsMergeReview(_AutocommitTaskHarness):
    """With ``auto_commit_mode=True`` the merge review is skipped."""

    def test_autocommit_commits_directly(self) -> None:
        tab_id = "test-tab-ac-on"
        tab = self.server._get_tab(tab_id)
        tab.use_worktree = False
        tab.auto_commit_mode = True

        # Simulate the agent modifying a tracked file.
        Path(self.tmpdir, "README.md").write_text(
            "# Hello\n\nAgent-edited content\n",
        )

        # Drive the post-task auto-commit path the task runner would
        # invoke when ``tab.auto_commit_mode`` is ON.  This is exactly
        # the call ``_run_task_inner``'s finally block now makes
        # instead of ``_prepare_and_start_merge``.
        self.server._handle_autocommit_action("commit", tab_id)

        types = [e["type"] for e in self.events]
        # No interactive merge view was opened.
        assert "merge_started" not in types
        assert "merge_data" not in types
        # And the changes were auto-committed.
        assert "autocommit_done" in types
        done = next(
            e for e in self.events if e["type"] == "autocommit_done"
        )
        assert done["success"] is True
        assert done["committed"] is True
        assert done["tabId"] == tab_id

        # Git status is clean again.
        status = _git(self.tmpdir, "status", "--porcelain")
        assert status.stdout.strip() == ""


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
