# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sibling-repo auto-commit must report git's real failure reason.

``_MergeFlowMixin._autocommit_paths_in_repo`` (the pass that commits
files a task changed OUTSIDE its work_dir repository) used to report
every failed ``git commit`` as ``git commit failed in <repo>
(pre-commit hook?).`` — a guess, while the work_dir pass next to it
already reported the first line of git's stderr.  With a pre-commit
hook that prints why it rejected the commit, the user saw the guess
and not the reason.

Before the fix the test below fails because the ``autocommit_done``
message does not contain the hook's output; after it the first line
of the hook's stderr is carried verbatim.  Real git repository, real
hook, real server; only the LLM commit-message generator is replaced
by a module-level function (the convention every auto-commit test in
this directory uses, since the message is generated before the hook
runs and must not need a model).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import kiss.server.merge_flow as _merge_flow_module
from kiss.server.server import VSCodeServer
from kiss.tests.conftest import posix_only

_HOOK_REASON = "HOOK-REJECT: commits are refused by this hook"


def _fixed_message(
    diff_text: str,
    user_prompt: str | None = None,
    task_result: str | None = None,
) -> str:
    """Deterministic stand-in for the LLM commit-message generator."""
    return "test: hook rejection message"


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


@posix_only("executable pre-commit hook script")
class TestSiblingRepoCommitReportsHookOutput(unittest.TestCase):
    """A rejecting pre-commit hook's stderr reaches the user."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-hook-error-test-")
        self.repo = str(Path(self.tmpdir) / "otherrepo")
        Path(self.repo).mkdir()
        _run_git(self.repo, "init", "-q")
        _run_git(self.repo, "config", "user.email", "test@example.com")
        _run_git(self.repo, "config", "user.name", "Test User")
        _run_git(self.repo, "config", "commit.gpgsign", "false")
        Path(self.repo, "seed.txt").write_text("seed\n")
        _run_git(self.repo, "add", "seed.txt")
        _run_git(self.repo, "commit", "-q", "-m", "seed")
        hook = Path(self.repo, ".git", "hooks", "pre-commit")
        hook.write_text(f"#!/bin/sh\necho '{_HOOK_REASON}' >&2\nexit 1\n")
        hook.chmod(hook.stat().st_mode | 0o111)

        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []
        orig_broadcast = self.server.printer.broadcast

        def capture(event: dict) -> None:
            self.events.append(event)
            orig_broadcast(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff
        _merge_flow_module.generate_commit_message_from_diff = _fixed_message

    def tearDown(self) -> None:
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_autocommit_done_carries_hook_stderr(self) -> None:
        target = Path(self.repo, "made-by-task.txt")
        target.write_text("hi\n")
        head = _run_git(self.repo, "rev-parse", "HEAD").stdout.strip()

        self.server._autocommit_paths_in_repo(
            Path(self.repo), [str(target)], "tab-hook",
        )

        assert _run_git(self.repo, "rev-parse", "HEAD").stdout.strip() == head, (
            "precondition: the hook must have rejected the commit"
        )
        done = [e for e in self.events if e["type"] == "autocommit_done"]
        assert len(done) == 1, f"expected one autocommit_done: {self.events}"
        assert done[0]["success"] is False
        assert done[0]["committed"] is False
        assert _HOOK_REASON in done[0]["message"], done[0]["message"]
        assert os.path.basename(self.repo) in done[0]["message"]
        assert "pre-commit hook?" not in done[0]["message"]


if __name__ == "__main__":
    unittest.main()
