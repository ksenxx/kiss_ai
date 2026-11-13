# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Git Commit holds the repository's merge locks around its git steps only,
re-checks the exact staged tree before committing and never takes staged
changes for a clean tree; a pending worktree is listed once when it is
presented, and a Merge that fails keeps its bar.

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
import sys
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
from kiss.tests.conftest import is_root, posix_only
from kiss.tests.server.conftest import redirect_persistence, restore_persistence
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.worktree_helpers import worktree_changed_files

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


def _blank_provider_keys(test: unittest.TestCase) -> None:
    """Leave *test* one bogus OpenAI key, so the real generator picks a model and falls back."""
    saved = {name: getattr(config_module.DEFAULT_CONFIG, name) for name in _KEY_FIELDS}
    _set_credentials(dict.fromkeys(_KEY_FIELDS, "") | {"OPENAI_API_KEY": "kiss-v6-standin"})
    test.addCleanup(_set_credentials, saved)


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
        _blank_provider_keys(self)
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


class TestStagedChangesAreNeverReportedAsNothingToCommit(_CommitHarness):
    """"Nothing to commit." is said of a clean index, not of a patch git could not print."""

    def test_missing_textconv_program_is_committed_with_the_fallback_message(self) -> None:
        (self.repo / ".git" / "info").mkdir(exist_ok=True)
        (self.repo / ".git" / "info" / "attributes").write_text("*.txt diff=gone\n")
        _git(self.repo, "config", "diff.gone.textconv", str(self.repo / "no-such-program"))
        _git(self.repo, "add", "-A")
        assert GitWorktreeOps.staged_diff(self.repo) == "", "git must fail to print the patch"
        self.server._commit_main_tree("tab-v6", str(self.repo))
        done = _wait_for(self.events, "autocommit_done", timeout=1)
        assert done["committed"] is True, done
        assert done["commitMessage"] == commit_message.FALLBACK_SUBJECT, done
        assert _git(self.repo, "status", "--porcelain") == ""


_PROBE_MERGE_FLOCK = """
import fcntl, sys
with open(sys.argv[1], "a+") as handle:
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("held")
    else:
        print("free")
"""


def _runs(frame_locals: dict, command: str) -> bool:
    """Whether the ``_git`` call entered with *frame_locals* runs git *command*."""
    return bool(frame_locals.get("args", ())[:1] == (command,))


@posix_only("fcntl.flock probed from a second process")
class TestCommitHoldsTheMergeLocksAroundEachGitStep(_CommitHarness):
    """Git Commit and a merge in another process or thread never run their git steps together.

    The commit is made in a linked worktree and held at each of its git
    steps; there, a second process must find the repository's
    ``kiss-reclaim.lock`` taken and this one the main worktree's
    ``repo_lock``, the two locks a merge holds for its whole transaction.
    Neither is held while the message is generated.
    """

    def _merge_locks_held(self, main: Path, git_dir: Path) -> tuple[bool, bool]:
        """Whether the flock in *git_dir* (probed by a new process) and *main*'s lock are taken."""
        probe = subprocess.run(
            [sys.executable, "-c", _PROBE_MERGE_FLOCK, str(git_dir / "kiss-reclaim.lock")],
            capture_output=True, text=True, timeout=120, check=True,
        )
        lock = repo_lock(main)
        free = lock.acquire(blocking=False)
        if free:
            lock.release()
        return probe.stdout.strip() == "held", not free

    def test_a_checkout_whose_git_dir_is_kept_apart_takes_the_same_locks(self) -> None:
        """git lists that git dir, not the checkout, as the main worktree: one lock all the same."""
        tmp = Path(self.tmpdir)
        apart, meta, linked = tmp / "apart", tmp / "meta", tmp / "linked"
        _git(tmp, "init", "-q", "-b", "main", f"--separate-git-dir={meta}", str(apart))
        _git(apart, "config", "user.email", "t@t")
        _git(apart, "config", "user.name", "t")
        _git(apart, "commit", "-q", "--allow-empty", "-m", "init")
        _git(apart, "worktree", "add", "-q", "-b", "side", str(linked))
        (linked / "f.txt").write_text("three\n")
        staging = Pause(git_worktree._git.__code__, lambda frame: _runs(frame, "add"))
        thread = threading.Thread(
            target=self.server._commit_main_tree, args=("tab-v6", str(linked)),
        )
        with staging:
            thread.start()
            self.addCleanup(thread.join, 120)
            assert staging.paused.wait(60), "the commit never reached its git add"
            held = self._merge_locks_held(apart, meta)
            staging.release()
            thread.join(120)
        assert held == (True, True)
        assert _wait_for(self.events, "autocommit_done")["committed"] is True

    def test_both_git_phases_run_under_the_flock_and_the_main_repo_lock(self) -> None:
        linked = Path(self.tmpdir) / "linked"
        _git(self.repo, "worktree", "add", "-q", "-b", "side", str(linked))
        (linked / "f.txt").write_text("three\n")
        generating = Pause(
            commit_message.generate_commit_message_from_diff.__code__, lambda _: True,
        )
        staging = Pause(git_worktree._git.__code__, lambda frame: _runs(frame, "add"))
        restaging = Pause(
            git_worktree._git.__code__,
            lambda frame: generating.paused.is_set() and _runs(frame, "add"),
        )
        committing = Pause(git_worktree._git.__code__, lambda frame: _runs(frame, "commit"))
        thread = threading.Thread(
            target=self.server._commit_main_tree, args=("tab-v6", str(linked)),
        )
        with Pauses(staging, generating, restaging, committing):
            thread.start()
            self.addCleanup(thread.join, 120)
            for step, pause, held in (
                ("git add", staging, True),
                ("message generation", generating, False),
                ("second git add", restaging, True),
                ("git commit", committing, True),
            ):
                assert pause.paused.wait(60), f"the commit never reached its {step}"
                assert self._merge_locks_held(self.repo, self.repo / ".git") == (held, held), step
                pause.release()
            thread.join(120)
        done = _wait_for(self.events, "autocommit_done")
        assert done["committed"] is True, done
        assert _git(linked, "status", "--porcelain") == ""
        assert self._merge_locks_held(self.repo, self.repo / ".git") == (False, False)


def _taken_elsewhere(lock: threading.RLock, timeout: float) -> bool:
    """Whether a thread other than the caller's takes *lock* within *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not lock.acquire(blocking=False):
            return True
        lock.release()
        time.sleep(0.01)
    return False


class TestTabsOnTheSameRepositoriesTakeTheirLocksInOneOrder(unittest.TestCase):
    """Two tabs with work pending in the same two repositories cannot deadlock.

    One tab reaches the repositories through their main checkouts
    (``a-main``, ``b-main``), the other through linked checkouts named
    the other way round (``c-b-linked``, ``d-a-linked``).  A checkout
    takes its repository's lock, so in the order of the checkout paths
    the first tab took A's lock then B's and the second B's then A's,
    and each waited for ever for the lock the other held.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-lock-order-")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        saved = redirect_persistence(self.tmpdir)
        self.addCleanup(restore_persistence, saved)
        tmp = Path(self.tmpdir)
        self.a = _make_repo(tmp / "a-main")
        self.b = _make_repo(tmp / "b-main")
        b_linked, a_linked = tmp / "c-b-linked", tmp / "d-a-linked"
        _git(self.b, "worktree", "add", "-q", "-b", "linked", str(b_linked))
        _git(self.a, "worktree", "add", "-q", "-b", "linked", str(a_linked))
        self.server = AgentServer()
        self.server.printer.broadcast = [].append  # type: ignore[assignment]
        self._leave_pending("tab-ab", [self.a, self.b])
        self._leave_pending("tab-ba", [b_linked, a_linked])
        self.results: dict[str, dict] = {}

    def _leave_pending(self, tab: str, checkouts: list[Path]) -> None:
        """Register *tab* with a task's worktree pending in each of *checkouts*."""
        agent = ChatSorcarAgent("wt_agent")
        agent._chat_id = tab
        for checkout in checkouts:
            assert agent._try_setup_worktree(checkout, str(checkout)) is not None
        state = agent_state.AgentState(f"task-{tab}", agent=agent, tab_id=tab, server_owned=True)
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, state.task_id, state)

    def _detach(self, tab: str) -> None:
        self.results[tab] = self.server._handle_worktree_action("nothing", tab)

    def test_each_tab_takes_the_two_locks_in_the_same_order(self) -> None:
        """While this thread holds B's lock, an action that takes A's first shows it taken."""
        lock_a, lock_b = repo_lock(self.a), repo_lock(self.b)
        assert lock_a is not lock_b
        takes_a_first = {}
        for tab in ("tab-ab", "tab-ba"):
            thread = threading.Thread(target=self._detach, args=(tab,), daemon=True)
            with lock_b:
                thread.start()
                takes_a_first[tab] = _taken_elsewhere(lock_a, timeout=10)
            thread.join(60)
            assert not thread.is_alive(), tab
        assert takes_a_first["tab-ab"] == takes_a_first["tab-ba"], takes_a_first
        assert [self.results[tab]["success"] for tab in ("tab-ab", "tab-ba")] == [True, True]


@posix_only("the logging git shim is a shell script")
class TestPendingWorktreeIsListedOnce(unittest.TestCase):
    """Presenting a worktree spawns the git processes of one listing."""

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
        files: list[str] = []
        listing = self._git_calls(
            lambda: files.extend(worktree_changed_files(self.server, "tab-v6-present")),
        )
        assert files == ["f.txt", "new.txt"] and listing > 0
        present = self._git_calls(
            lambda: self.server._present_pending_worktree(
                "tab-v6-present", discard_if_empty=False,
            ),
        )
        assert present == listing, (present, listing)
        # Every field has a reader: the page (the branches, the tab) or the
        # daemon's file-link fallback (the directories).
        assert _wait_for(self.events, "worktree_done", timeout=1) == {
            "type": "worktree_done",
            "branch": wt.branch,
            "worktreeDir": str(wt.wt_dir),
            "worktreeWorkDir": str(self.agent._wt_work_dir),
            "worktrees": [{"branch": wt.branch, "worktreeDir": str(wt.wt_dir)}],
            "tabId": "tab-v6-present",
        }
        assert self.agent._wt_pending

    def test_merge_refused_by_a_hook_keeps_the_bar_retryable(self) -> None:
        """A failed Merge leaves the worktree pending, so its result must keep the bar."""
        _blank_provider_keys(self)
        hook = self.repo / ".git" / "hooks" / "pre-commit"
        hook.parent.mkdir(exist_ok=True)
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        result = self.server._handle_worktree_action("merge", "tab-v6-present")
        assert result["success"] is False and result.get("retryable") is True, result
        assert self.agent._wt_pending

    @unittest.skipIf(is_root(), "needs permission bits that deny the owner")
    def test_merge_that_raises_keeps_the_bar_retryable(self) -> None:
        """A Merge that dies on a filesystem error leaves the worktree pending too.

        The repository is unreadable for the merge (mode 000), so the
        worktree's own ``stat`` raises ``PermissionError`` out of the
        action.  Its ``worktree_result`` carried no ``retryable``, and
        every page dropped the bar of a worktree still holding the work.
        """
        self.repo.chmod(0)
        self.addCleanup(self.repo.chmod, 0o755)
        self.server._handle_command(
            {"type": "worktreeAction", "action": "merge", "tabId": "tab-v6-present"},
        )
        self.repo.chmod(0o755)
        result = _wait_for(self.events, "worktree_result", timeout=1)
        assert result["success"] is False and "Permission denied" in result["message"], result
        assert result.get("retryable") is True, result
        assert self.agent._wt_pending and not self.state.is_merging

    def test_presentation_of_an_unchanged_worktree_emits_nothing(self) -> None:
        wt = self.agent._wt
        assert wt is not None
        (wt.wt_dir / "f.txt").write_text("one\n")
        (wt.wt_dir / "new.txt").unlink()
        listing = self._git_calls(lambda: worktree_changed_files(self.server, "tab-v6-present"))
        present = self._git_calls(
            lambda: self.server._present_pending_worktree(
                "tab-v6-present", discard_if_empty=False,
            ),
        )
        assert present == listing, (present, listing)
        assert self.events == []
        assert self.agent._wt_pending

    @posix_only("a FIFO that blocks git's read of a .gitignore")
    def test_failed_untracked_listing_is_unknown_not_clean(self) -> None:
        """A worktree whose new files git cannot list is presented, not discarded as clean.

        ``git ls-files --others --exclude-standard`` reads every
        ``.gitignore`` it walks past; a FIFO by that name holds the
        read until ``_git``'s timeout fails the command (returncode
        124) while the tracked diffs, which read no ignore file, come
        back clean.  Taking that failure for "no new files" would
        auto-discard the branch with the task's only output.
        """
        wt = self.agent._wt
        assert wt is not None
        (wt.wt_dir / "f.txt").write_text("one\n")
        (wt.wt_dir / "sub").mkdir()
        fifo = wt.wt_dir / "sub" / ".gitignore"
        os.mkfifo(fifo)
        self.addCleanup(fifo.unlink)
        saved = git_worktree._GIT_TIMEOUT_SECONDS
        git_worktree._GIT_TIMEOUT_SECONDS = 1.0
        self.addCleanup(setattr, git_worktree, "_GIT_TIMEOUT_SECONDS", saved)
        [(_, files)] = self.server._list_pending(self.agent)
        assert files is None, files
        assert self.server._present_pending_worktree("tab-v6-present") is True
        done = _wait_for(self.events, "worktree_done", timeout=1)
        assert [row["branch"] for row in done["worktrees"]] == [wt.branch]
        assert self.agent._wt_pending and (wt.wt_dir / "new.txt").exists()


if __name__ == "__main__":
    unittest.main()
