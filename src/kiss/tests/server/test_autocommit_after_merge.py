# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the non-worktree post-task auto-commit.

With the interactive diff/merge review workflow removed, a non-worktree
task that leaves the main working tree dirty is always committed
directly: :meth:`_MergeFlowMixin._autocommit_changes` stages
everything, generates a commit message and commits to the current
branch, reporting progress through ``autocommit_progress`` /
``autocommit_done`` events.  The old ``autocommit_prompt`` event and
the ``mergeAction`` command no longer exist; ``autocommitAction`` is
back as the settings panel's manual "Git Commit" command and runs
``_autocommit_changes`` directly (see
``test_settings_git_commit_button.py``).

These tests drive :class:`VSCodeServer` with real ``git`` state — no
mocks, no test doubles.  The LLM call for commit-message generation is
replaced with a deterministic override of the module-level
``generate_commit_message_from_diff`` function in ``merge_flow``,
since this is a module-level extension point, not a mock / patch of
dependency internals.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import kiss.server.merge_flow as _merge_flow_module
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    """Create a git repo with one committed file so HEAD exists."""
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


class _ServerHarness(unittest.TestCase):
    """Shared setUp/tearDown — real git repo + event capture."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff

        def capture(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _types(self) -> list[str]:
        return [e["type"] for e in self.events]

    def _event(self, type_: str) -> dict:
        for e in self.events:
            if e["type"] == type_:
                return e
        raise AssertionError(f"No event of type {type_!r}: {self._types()}")


class TestAutocommitChanges(_ServerHarness):
    """``_autocommit_changes`` stages everything (including untracked),
    generates a commit message, and commits to the current branch."""

    def setUp(self) -> None:
        super().setUp()
        self._messages: list[str] = []

        def fake_compose(
            diff_text: str,
            user_prompt: str | None = None,
            task_result: str | None = None,
        ) -> str:
            self._messages.append(diff_text)
            return "feat: deterministic test commit"

        _merge_flow_module.generate_commit_message_from_diff = fake_compose  # type: ignore[assignment]

    def test_commit_stages_and_commits_tracked_and_untracked(self) -> None:
        Path(self.tmpdir, "seed.txt").write_text("updated seed\n")
        Path(self.tmpdir, "new.txt").write_text("brand new\n")

        before = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        self.server._autocommit_changes("t1")
        after = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()

        assert before != after
        status = _run_git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert status == ""
        log = _run_git(self.tmpdir, "log", "-1", "--pretty=%s").stdout.strip()
        assert log == "feat: deterministic test commit"
        show = _run_git(
            self.tmpdir, "show", "--name-only", "--pretty=", "HEAD",
        ).stdout
        assert "seed.txt" in show
        assert "new.txt" in show
        assert len(self._messages) == 1
        assert "seed.txt" in self._messages[0] or "new.txt" in self._messages[0]

        evt = self._event("autocommit_done")
        assert evt["success"] is True
        assert evt["committed"] is True
        assert evt["tabId"] == "t1"

    def test_progress_events_precede_done(self) -> None:
        """The webview shows a staged progress toast before the result."""
        Path(self.tmpdir, "seed.txt").write_text("progress\n")

        self.server._autocommit_changes("t1")

        types = self._types()
        assert "autocommit_progress" in types
        assert types.index("autocommit_progress") < types.index(
            "autocommit_done",
        )

    def test_commit_with_only_untracked(self) -> None:
        """Commit handles the case with only untracked files."""
        Path(self.tmpdir, "only_new.txt").write_text("hi\n")

        self.server._autocommit_changes("t1")

        status = _run_git(self.tmpdir, "status", "--porcelain").stdout.strip()
        assert status == ""
        show = _run_git(
            self.tmpdir, "show", "--name-only", "--pretty=", "HEAD",
        ).stdout
        assert "only_new.txt" in show
        evt = self._event("autocommit_done")
        assert evt["success"] is True
        assert evt["committed"] is True

    def test_commit_when_nothing_to_commit(self) -> None:
        """If there's nothing to commit (race), broadcast success with
        ``committed: False`` instead of failing."""
        before = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()

        self.server._autocommit_changes("t1")

        after = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        assert before == after
        evt = self._event("autocommit_done")
        assert evt["success"] is True
        assert evt["committed"] is False
        assert self._messages == []

    def test_commit_in_non_git_dir_reports_failure(self) -> None:
        non_git = tempfile.mkdtemp()
        try:
            self.server.work_dir = non_git
            self.server._autocommit_changes("t1")
            evt = self._event("autocommit_done")
            assert evt["success"] is False
            assert evt["committed"] is False
        finally:
            shutil.rmtree(non_git, ignore_errors=True)

    def test_explicit_work_dir_preferred_over_server_work_dir(self) -> None:
        """The ``work_dir`` keyword wins over the daemon-wide folder."""
        non_git = tempfile.mkdtemp()
        try:
            self.server.work_dir = non_git
            Path(self.tmpdir, "seed.txt").write_text("via keyword\n")
            self.server._autocommit_changes("t1", work_dir=self.tmpdir)
            evt = self._event("autocommit_done")
            assert evt["success"] is True
            assert evt["committed"] is True
            status = _run_git(
                self.tmpdir, "status", "--porcelain",
            ).stdout.strip()
            assert status == ""
        finally:
            shutil.rmtree(non_git, ignore_errors=True)


class TestRemovedCommandsAreUnknown(_ServerHarness):
    """The retired review commands are plain unknown commands now."""

    def test_autocommit_action_is_known_again(self) -> None:
        """``autocommitAction`` is live again (the settings panel's
        manual "Git Commit" button) and must not be rejected as an
        unknown command."""
        import time

        before = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        self.server._handle_command(
            {"type": "autocommitAction", "tabId": "t1",
             "workDir": self.tmpdir},
        )
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if any(e["type"] == "autocommit_done" for e in self.events):
                break
            time.sleep(0.05)
        assert "error" not in self._types(), self.events
        evt = self._event("autocommit_done")
        assert evt["success"] is True
        assert evt["committed"] is False
        assert evt["message"] == "Nothing to commit."
        after = _run_git(self.tmpdir, "rev-parse", "HEAD").stdout.strip()
        assert before == after, "a clean tree must not gain a commit"

    def test_merge_action_is_unknown(self) -> None:
        self.server._handle_command(
            {"type": "mergeAction", "action": "all-done", "tabId": "t1"},
        )
        evt = self._event("error")
        assert "Unknown command" in evt["text"]

    def test_unknown_cmd_still_rejected(self) -> None:
        """Safety check: other unknown command types still broadcast
        the generic unknown-command error."""
        self.server._handle_command({"type": "nosuchcmd"})
        evt = self._event("error")
        assert "Unknown command" in evt["text"]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
