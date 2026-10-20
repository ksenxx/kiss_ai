# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Git Commit releases the repository lock while the message is generated
and re-checks the exact staged tree before committing; a pending worktree
is listed once when it is presented.

Real git repositories, a real :class:`AgentServer`, real threads.  The
commit-message generator is the real one: the provider keys are blanked
(one bogus OpenAI key left so a model is picked) and it takes its
documented fallback path.  :class:`Pause` holds the commit thread at
the generator's entry so the test can act inside the unlocked window.
The git processes a presentation spawns are counted by a real ``git``
executable on ``PATH`` that logs its arguments and hands over to the
installed git.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from kiss.agents.sorcar import commit_message, git_worktree
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, repo_lock
from kiss.core import config as config_module
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import redirect_persistence, restore_persistence
from kiss.tests.server.pause import Pause

_KEY_FIELDS = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "ZAI_API_KEY",
    "MOONSHOT_API_KEY",
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    (path / "f.txt").write_text("one\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path


def _set_credentials(values: dict[str, str]) -> None:
    """Set the live config's provider keys (blank ones send the generator down its fallback)."""
    for name, value in values.items():
        setattr(config_module.DEFAULT_CONFIG, name, value)


def _wait_for(events: list[dict], kind: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for event in list(events):
            if event.get("type") == kind:
                return event
        time.sleep(0.02)
    raise AssertionError(f"no {kind} event: {[e.get('type') for e in events]}")


class _CommitHarness(unittest.TestCase):
    """A repository with one modified file and a server whose commits run unlocked."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-v6-commit-")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        (self.repo / "f.txt").write_text("two\n")
        saved = {name: getattr(config_module.DEFAULT_CONFIG, name) for name in _KEY_FIELDS}
        _set_credentials(dict.fromkeys(_KEY_FIELDS, "") | {"OPENAI_API_KEY": "kiss-v6-standin"})
        self.addCleanup(_set_credentials, saved)
        self.server = AgentServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

    def _commit_held_at_generator(self) -> tuple[threading.Thread, Pause]:
        """Start a Git Commit and return once its thread waits at the message generator."""
        pause = Pause(
            commit_message.generate_commit_message_from_diff.__code__, lambda _: True,
        )
        pause.__enter__()
        self.addCleanup(pause.__exit__, None, None, None)
        thread = threading.Thread(
            target=self.server._commit_main_tree, args=("tab-v6", str(self.repo)),
        )
        thread.start()
        self.addCleanup(thread.join, 120)
        assert pause.paused.wait(30), "the commit never reached the message generator"
        return thread, pause


class TestCommitDoesNotHoldRepoLockDuringMessageGeneration(_CommitHarness):
    """The lock's other users (worktree setup, Source Control, Merge) must not wait on the LLM."""

    def test_lock_is_free_while_the_message_is_generated(self) -> None:
        thread, pause = self._commit_held_at_generator()
        lock = repo_lock(self.repo)
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        pause.release()
        thread.join(120)
        assert acquired, "repo_lock was held across the commit-message generation"
        done = _wait_for(self.events, "autocommit_done")
        assert done["committed"] is True, done
        assert _git(self.repo, "status", "--porcelain") == ""

    def test_tree_changed_during_generation_is_refused_not_committed(self) -> None:
        head = _git(self.repo, "rev-parse", "HEAD")
        thread, pause = self._commit_held_at_generator()
        (self.repo / "f.txt").write_text("three\n")
        pause.release()
        thread.join(120)
        done = _wait_for(self.events, "autocommit_done")
        assert done["success"] is False and done["committed"] is False, done
        assert "changed while the commit message" in done["message"], done
        assert _git(self.repo, "rev-parse", "HEAD") == head

    def test_staging_failure_after_generation_is_reported(self) -> None:
        head = _git(self.repo, "rev-parse", "HEAD")
        thread, pause = self._commit_held_at_generator()
        index_lock = self.repo / ".git" / "index.lock"
        index_lock.write_text("")
        pause.release()
        thread.join(120)
        index_lock.unlink()
        done = _wait_for(self.events, "autocommit_done")
        assert done["success"] is False and done["committed"] is False, done
        assert done["message"].startswith("Staging failed: "), done
        assert _git(self.repo, "rev-parse", "HEAD") == head

    @posix_only("chmod; a read-only directory does not stop writes on Windows")
    def test_write_tree_failure_is_reported_as_staging_failure(self) -> None:
        """The index is identified by ``git write-tree``; its failure is a staging failure."""
        head = _git(self.repo, "rev-parse", "HEAD")
        objects = self.repo / ".git" / "objects"
        self.addCleanup(subprocess.run, ["chmod", "-R", "u+w", str(objects)], check=False)

        locked: list[str] = []

        def lock_objects_before_write_tree(frame_locals: dict) -> bool:
            if frame_locals.get("args", ())[:1] == ("write-tree",):
                subprocess.run(["chmod", "-R", "a-w", str(objects)], check=True)
                locked.append("write-tree")
            return False

        with Pause(git_worktree._git.__code__, lock_objects_before_write_tree):
            thread = threading.Thread(
                target=self.server._commit_main_tree, args=("tab-v6", str(self.repo)),
            )
            thread.start()
            thread.join(120)
        assert locked == ["write-tree"], "git add -A must have run before write-tree"
        done = _wait_for(self.events, "autocommit_done")
        assert done["success"] is False and done["committed"] is False, done
        assert done["message"].startswith("Staging failed: "), done
        assert "permission" in done["message"], done
        assert _git(self.repo, "rev-parse", "HEAD") == head


class TestCommitGuardIsExactBeyondThePromptCap(_CommitHarness):
    """The freshness check must see every staged byte, not the bounded prompt text.

    The patch handed to the model is capped at
    ``COMMIT_MESSAGE_DIFF_LIMIT_BYTES``; a change to a file past that cap
    that keeps the ``--shortstat`` / ``--stat`` figures leaves the prompt
    text identical, so only an exact identity of the index (and of the
    parent commit) can tell the staged content moved.
    """

    def setUp(self) -> None:
        super().setUp()
        (self.repo / "a.txt").write_text("".join(f"old {n:06d}\n" for n in range(18_000)))
        (self.repo / "z.txt").write_text("base\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-qm", "big")
        (self.repo / "a.txt").write_text("".join(f"new {n:06d}\n" for n in range(18_000)))
        (self.repo / "z.txt").write_text("first\n")
        _git(self.repo, "add", "-A")
        prompt = GitWorktreeOps.staged_diff(self.repo)
        assert prompt.startswith("[Patch truncated"), prompt[:80]
        assert "z.txt" not in prompt.split("\n\n", 2)[2], "z.txt must lie past the patch cap"

    def test_change_past_the_cap_during_generation_is_refused(self) -> None:
        head = _git(self.repo, "rev-parse", "HEAD")
        thread, pause = self._commit_held_at_generator()
        (self.repo / "z.txt").write_text("later\n")
        pause.release()
        thread.join(120)
        done = _wait_for(self.events, "autocommit_done")
        assert done["success"] is False and done["committed"] is False, done
        assert "changed while the commit message" in done["message"], done
        assert _git(self.repo, "rev-parse", "HEAD") == head
        assert _git(self.repo, "status", "--porcelain") != ""

    def test_commit_by_someone_else_during_generation_is_refused(self) -> None:
        thread, pause = self._commit_held_at_generator()
        _git(self.repo, "commit", "-qam", "elsewhere")
        other = _git(self.repo, "rev-parse", "HEAD")
        pause.release()
        thread.join(120)
        done = _wait_for(self.events, "autocommit_done")
        assert done["success"] is False and done["committed"] is False, done
        assert "changed while the commit message" in done["message"], done
        assert _git(self.repo, "rev-parse", "HEAD") == other


@posix_only("the logging git shim is a shell script")
class TestPendingWorktreeIsListedOnce(unittest.TestCase):
    """Presenting a worktree spawns the git processes of one listing plus the conflict probe."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-v6-present-")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        saved = redirect_persistence(self.tmpdir)
        self.addCleanup(restore_persistence, saved)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.server = AgentServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]
        self.agent = ChatSorcarAgent("wt_agent")
        self.agent._chat_id = "tab-v6-present"
        assert self.agent._try_setup_worktree(self.repo, str(self.repo)) is not None
        wt = self.agent._wt
        assert wt is not None
        self.addCleanup(GitWorktreeOps.delete_branch, self.repo, wt.branch)
        self.addCleanup(GitWorktreeOps.prune, self.repo)
        self.addCleanup(GitWorktreeOps.remove, self.repo, wt.wt_dir)
        (wt.wt_dir / "f.txt").write_text("worktree edit\n")
        (wt.wt_dir / "new.txt").write_text("new\n")
        self.state = agent_state.AgentState(
            "task-v6-present", agent=self.agent, tab_id="tab-v6-present", server_owned=True,
        )
        agent_state.register(self.state)
        self.addCleanup(agent_state.unregister, self.state.task_id, self.state)
        self.log = Path(self.tmpdir) / "git-calls.log"
        shim_dir = Path(self.tmpdir) / "bin"
        shim_dir.mkdir()
        real_git = shutil.which("git")
        assert real_git
        shim = shim_dir / "git"
        shim.write_text(f'#!/bin/sh\necho "$@" >> {self.log}\nexec {real_git} "$@"\n')
        shim.chmod(0o755)
        saved_path = os.environ["PATH"]
        os.environ["PATH"] = f"{shim_dir}{os.pathsep}{saved_path}"
        self.addCleanup(os.environ.__setitem__, "PATH", saved_path)

    def _git_calls(self, action) -> int:
        self.log.write_text("")
        action()
        return len(self.log.read_text().splitlines())

    def test_presentation_lists_each_worktree_once(self) -> None:
        wt = self.agent._wt
        assert wt is not None
        listing = self._git_calls(lambda: self.server._get_worktree_changed_files("tab-v6-present"))
        conflict = self._git_calls(lambda: self.server._conflict_of(wt))
        assert listing > 0 and conflict > 0
        present = self._git_calls(
            lambda: self.server._present_pending_worktree(
                "tab-v6-present", discard_if_empty=False,
            ),
        )
        assert present == listing + conflict, (present, listing, conflict)
        done = _wait_for(self.events, "worktree_done", timeout=1)
        assert done["changedFiles"] == ["f.txt", "new.txt"]
        assert done["worktrees"][0]["changedFiles"] == ["f.txt", "new.txt"]
        assert done["hasConflict"] is False
        assert self.agent._wt_pending

    def test_presentation_of_an_unchanged_worktree_emits_nothing(self) -> None:
        wt = self.agent._wt
        assert wt is not None
        (wt.wt_dir / "f.txt").write_text("one\n")
        (wt.wt_dir / "new.txt").unlink()
        listing = self._git_calls(lambda: self.server._get_worktree_changed_files("tab-v6-present"))
        present = self._git_calls(
            lambda: self.server._present_pending_worktree(
                "tab-v6-present", discard_if_empty=False,
            ),
        )
        assert present == listing, (present, listing)
        assert self.events == []
        assert self.agent._wt_pending


if __name__ == "__main__":
    unittest.main()
