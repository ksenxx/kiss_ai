# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the "Auto commit" toggle checkbox.

Validates:
- ``AgentState`` carries an ``auto_commit_mode`` field defaulting to
  ``True`` (the run command overwrites it per task from the frontend
  toggle).
- ``_autocommit_changes`` commits agent changes on the main tree
  directly (non-worktree branch), with no interactive review.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import kiss.server.merge_flow as _merge_flow_module
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer


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


class TestAgentStateField(unittest.TestCase):
    """``AgentState`` carries the per-task auto-commit toggle state."""

    def test_default_true(self) -> None:
        state = AgentState("task-x")
        assert state.auto_commit_mode is True

    def test_settable(self) -> None:
        state = AgentState("task-y")
        state.auto_commit_mode = False
        assert state.auto_commit_mode is False


class _AutocommitTaskHarness(unittest.TestCase):
    """Shared setUp/tearDown for end-to-end autocommit-toggle tests."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)
        self.server, self.events = _make_server(self.tmpdir)
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff
        def _stub(
            diff_text: str,
            user_prompt: str | None = None,
            task_result: str | None = None,
        ) -> str:
            del diff_text, user_prompt, task_result
            return "auto-commit-toggle-test"

        _merge_flow_module.generate_commit_message_from_diff = _stub

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestAutocommitCommitsDirectly(_AutocommitTaskHarness):
    """Post-task autocommit commits directly, with no review events."""

    def test_autocommit_commits_directly(self) -> None:
        tab_id = "test-tab-ac-on"

        Path(self.tmpdir, "README.md").write_text(
            "# Hello\n\nAgent-edited content\n",
        )

        self.server._autocommit_changes(tab_id)

        types = [e["type"] for e in self.events]
        assert "merge_started" not in types
        assert "merge_data" not in types
        assert "autocommit_done" in types
        done = next(
            e for e in self.events if e["type"] == "autocommit_done"
        )
        assert done["success"] is True
        assert done["committed"] is True
        assert done["tabId"] == tab_id

        status = _git(self.tmpdir, "status", "--porcelain")
        assert status.stdout.strip() == ""


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
