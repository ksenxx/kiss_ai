# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the manual Git Commit button flow.

When the user presses the settings panel's Git Commit button, the
daemon runs ``_cmd_autocommit_action`` → ``_autocommit_changes`` with
``manual=True``.  A manual commit must:

- Generate the commit message from the staged diff ALONE: no
  ``User prompt:`` / ``Result:`` sections, even when the tab has a
  recorded last prompt and result summary.
- Report progress and outcome through toast ``notification`` events
  ("Auto-generating commit message…" first, then "Committed: <subject>"
  on success or the failure reason with ``severity: "error"``).
- Emit NO ``autocommit_progress`` transcript events, and mark its
  ``autocommit_done`` event ``manual: True`` so a SUCCESS adds no
  transcript text (the webview suppresses it) while a FAILURE still
  shows its reason in the chat webview.

The post-task autocommit path (``manual=False``) must keep its
existing behavior: transcript progress events, prompt/result sections
in the message, and no toast notifications.

These tests drive the real ``VSCodeServer._cmd_autocommit_action``
against on-disk git repositories; only the billed LLM call inside the
commit-message helper is replaced by a module-level recording override
(the same pattern as ``test_merge_autocommit_lifecycle.py``).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import kiss.server.merge_flow as _merge_flow_module
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


class _ManualCommitHarness(unittest.TestCase):
    """Real git repo + captured events + recording commit-message stub."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        _init_repo(self.tmpdir)

        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff
        self.gen_calls: list[dict] = []

        def fake_gen(
            diff_text: str,
            user_prompt: str | None = None,
            task_result: str | None = None,
        ) -> str:
            self.gen_calls.append({
                "diff_text": diff_text,
                "user_prompt": user_prompt,
                "task_result": task_result,
            })
            msg = "feat: update readme"
            if user_prompt:
                msg += f"\n\nUser prompt:\n{user_prompt}"
            if task_result:
                msg += f"\n\nResult:\n{task_result}"
            return msg

        _merge_flow_module.generate_commit_message_from_diff = fake_gen  # type: ignore[assignment]

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _register_tab_with_history(self, tab_id: str) -> None:
        """Register a tab whose agent state carries a prompt + result."""
        state = agent_state.AgentState(
            f"task-{tab_id}", tab_id=tab_id, server_owned=True,
        )
        state.last_user_prompt = "please update the readme"
        state.last_result_summary = "updated the readme"
        agent_state.register(state)

    def _manual_commit(self, tab_id: str, work_dir: str) -> dict:
        """Press the Git Commit button and wait for autocommit_done."""
        self.server._cmd_autocommit_action({
            "tabId": tab_id, "workDir": work_dir,
        })
        deadline = time.time() + 30
        while time.time() < deadline:
            for e in list(self.events):
                if e.get("type") == "autocommit_done":
                    return e
            time.sleep(0.02)
        raise AssertionError(
            f"No autocommit_done event: {[e.get('type') for e in self.events]}"
        )

    def _notifications(self) -> list[dict]:
        return [e for e in self.events if e.get("type") == "notification"]


class TestManualCommitMessageIsDiffOnly(_ManualCommitHarness):
    """Manual commit: message from the diff, no prompt/result sections."""

    def test_no_user_prompt_or_result_in_message(self) -> None:
        """The LLM helper gets no prompt/result and git log shows none."""
        tab_id = "tab-manual-1"
        self._register_tab_with_history(tab_id)
        Path(self.tmpdir, "README.md").write_text("# Hello\n\nEdited\n")

        done = self._manual_commit(tab_id, self.tmpdir)

        assert done["success"] is True
        assert done["committed"] is True
        assert done["manual"] is True
        assert done["tabId"] == tab_id

        assert len(self.gen_calls) == 1
        assert self.gen_calls[0]["user_prompt"] is None
        assert self.gen_calls[0]["task_result"] is None
        assert "README.md" in self.gen_calls[0]["diff_text"]

        body = _git(self.tmpdir, "log", "-1", "--pretty=%B").stdout
        assert "User prompt:" not in body
        assert "Result:" not in body
        assert body.strip() == "feat: update readme"

    def test_notifications_replace_transcript_text(self) -> None:
        """Generating + committed toasts, no transcript progress events."""
        tab_id = "tab-manual-2"
        Path(self.tmpdir, "README.md").write_text("# Hello\n\nEdited again\n")

        done = self._manual_commit(tab_id, self.tmpdir)

        types = [e.get("type") for e in self.events]
        assert "autocommit_progress" not in types, types

        notes = self._notifications()
        assert len(notes) == 2, notes
        generating, committed = notes
        assert generating["message"] == "Auto-generating commit message…"
        assert generating["severity"] == "info"
        assert generating["tabId"] == tab_id
        assert committed["message"] == "Committed: feat: update readme"
        assert committed["severity"] == "info"
        # Same stable id so the outcome toast replaces the generating
        # toast in place instead of stacking a second one.
        assert generating["id"] == committed["id"] == f"manual-commit:{tab_id}"
        # The generating toast must precede the terminal event.
        assert self.events.index(generating) < self.events.index(done)

    def test_nothing_to_commit_is_a_silent_success_toast(self) -> None:
        """A clean tree yields an info toast and a silent manual done."""
        tab_id = "tab-manual-3"

        done = self._manual_commit(tab_id, self.tmpdir)

        assert done["success"] is True
        assert done["committed"] is False
        assert done["manual"] is True
        notes = self._notifications()
        assert len(notes) == 1, notes
        assert notes[0]["message"] == "Nothing to commit."
        assert notes[0]["severity"] == "info"


class TestManualCommitFailure(_ManualCommitHarness):
    """Manual commit failures toast an error AND stay in the chat."""

    def test_non_git_folder_fails_with_error_toast(self) -> None:
        """A non-git workDir reports failure via an error notification."""
        tab_id = "tab-manual-fail"
        non_git = tempfile.mkdtemp()
        try:
            done = self._manual_commit(tab_id, non_git)
        finally:
            shutil.rmtree(non_git, ignore_errors=True)

        assert done["success"] is False
        assert done["committed"] is False
        # A failed manual commit is NOT silent: the webview renders the
        # reason of any non-successful manual event in the chat, so the
        # event must still carry the reason alongside the manual flag.
        assert done["manual"] is True
        assert done["message"] == "Not a git repository."
        notes = self._notifications()
        assert len(notes) == 1, notes
        assert notes[0]["severity"] == "error"
        assert notes[0]["message"] == "Not a git repository."

    def test_commit_hook_failure_reports_reason(self) -> None:
        """A pre-commit hook rejection surfaces its ACTUAL output.

        The user must see WHY the commit failed — the hook's own
        message, not a generic "pre-commit hook?" guess.
        """
        tab_id = "tab-manual-hook"
        hooks = Path(self.tmpdir, ".git", "hooks")
        hooks.mkdir(parents=True, exist_ok=True)
        hook = hooks / "pre-commit"
        hook.write_text(
            "#!/bin/sh\necho 'lint check failed: bad style' >&2\nexit 1\n"
        )
        hook.chmod(0o755)
        Path(self.tmpdir, "README.md").write_text("# Hook fail\n")

        done = self._manual_commit(tab_id, self.tmpdir)

        assert done["success"] is False
        assert done["manual"] is True
        assert done["message"] == (
            "git commit failed: lint check failed: bad style"
        )
        errors = [n for n in self._notifications() if n["severity"] == "error"]
        assert len(errors) == 1
        assert errors[0]["message"] == done["message"]


class TestPostTaskAutocommitUnchanged(_ManualCommitHarness):
    """The post-task path (manual=False) keeps its existing behavior."""

    def test_prompt_and_result_still_included(self) -> None:
        """Direct _autocommit_changes keeps prompt/result and progress."""
        tab_id = "tab-posttask"
        self._register_tab_with_history(tab_id)
        Path(self.tmpdir, "README.md").write_text("# Post task\n")

        self.server._autocommit_changes(tab_id, work_dir=self.tmpdir)

        types = [e.get("type") for e in self.events]
        assert "autocommit_progress" in types, types
        assert "notification" not in types, types
        done = next(
            e for e in self.events if e.get("type") == "autocommit_done"
        )
        assert done["success"] is True
        assert "manual" not in done
        assert len(self.gen_calls) == 1
        assert self.gen_calls[0]["user_prompt"] == "please update the readme"
        assert self.gen_calls[0]["task_result"] == "updated the readme"
        body = _git(self.tmpdir, "log", "-1", "--pretty=%B").stdout
        assert "User prompt:" in body
        assert "Result:" in body


if __name__ == "__main__":
    unittest.main()
