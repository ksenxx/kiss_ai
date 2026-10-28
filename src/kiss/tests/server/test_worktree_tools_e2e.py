"""End-to-end tests of the ``create_worktree`` / ``commit_and_merge`` tools.

A real ``AgentServer`` run on a scripted stand-in model endpoint
exercises the whole protocol: the tool-call guard refuses a ``Write``
of a tracked file until ``create_worktree`` ran, lets a git-ignored
path through, refuses the first ``finish`` while the worktree is
unmerged, and ``commit_and_merge`` lands the change on the user's
branch and removes the worktree.  A run that never merges leaves its
worktree for the user's Merge / Discard bar.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import WORKTREE_TOOL_NAMES, WorkerSea
from kiss.agents.sorcar.agent_dispatch import current_work_dir
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    _reclaim_process_lock,
    repo_lock,
)
from kiss.agents.sorcar.sorcar_agent import (
    TOOL_GROUPS,
    TOOL_PROFILES,
    SorcarAgent,
    resolve_tool_profile,
)
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import run_git
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)
from kiss.tests.server.test_autocommit_wire_toggle import _FakeCredentials

_MARKER = "WORKTREE-TOOLS-E2E"
REFUSED_WRITE = "Call create_worktree() first"
REFUSED_FINISH = "has not been merged"


def _write(path: str, content: str) -> dict[str, Any]:
    return tool_call_response("Write", {"file_path": path, "content": content})


class _ScriptedRun(unittest.TestCase):
    """A server run of one scripted task agent on an isolated repo."""

    script: list[dict[str, Any] | Callable[[], dict[str, Any]]] = []

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-worktree-tools-e2e-")
        self.repo = self.home.repo
        (self.repo / "f.txt").write_text("a\nb\nc\n")
        (self.repo / ".gitignore").write_text("tmp/\n")
        run_git(self.repo, "add", "f.txt", ".gitignore")
        run_git(self.repo, "commit", "-qm", "add f and gitignore")
        self.home.write_config(max_budget=5.0, use_web_browser=False)
        self.credentials = _FakeCredentials()
        self.requests: list[str] = []
        self._lock = threading.Lock()
        self.standin = StandInModelServer(self._respond)
        self.printer = CapturePrinter()
        self.server = AgentServer(printer=self.printer)
        self.server.work_dir = str(self.repo)

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            states = list(agent_state.agent_states.values())
        for state in states:
            agent = state.agent
            if agent is not None and getattr(agent, "_wt", None) is not None:
                try:
                    agent.discard()
                except Exception:
                    pass
        self.standin.stop()
        self.credentials.restore()
        self.home.cleanup()

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer the n-th model call with the n-th scripted tool call."""
        text = request_text(request)
        with self._lock:
            self.requests.append(text)
            index = len(self.requests) - 1
        if index < len(self.script):
            entry = self.script[index]
            return entry() if callable(entry) else entry
        return finish_response("task done")

    def _run(self) -> None:
        self.server._run_task({
            "type": "run",
            "tabId": "wt-tools-tab",
            "prompt": f"{_MARKER}: add g.txt",
            "model": STANDIN_MODEL,
            "workDir": str(self.repo),
            "isParallel": False,
            "useWebTools": False,
            "classifyTasks": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        })


class TestCreateWorktreeAndCommitAndMerge(_ScriptedRun):
    """The full protocol: guard, create, edit, finish gate, merge, finish."""

    script = [
        _write("g.txt", "new\n"),                     # 0: refused, new file in the tracked tree
        _write("tmp/notes.md", "scratch\n"),          # 1: allowed, git-ignored
        tool_call_response("create_worktree", {}),    # 2
        _write("g.txt", "new\n"),                     # 3: allowed, in the worktree
        finish_response("done without merging"),      # 4: refused once
        tool_call_response("commit_and_merge", {}),   # 5
        finish_response("merged and done"),           # 6
    ]

    def test_protocol(self) -> None:
        self._run()
        self.assertEqual(len(self.requests), 7, [r[-300:] for r in self.requests])

        self.assertIn(REFUSED_WRITE, self.requests[1])
        self.assertIn("is not git-ignored", self.requests[1])
        # The git-ignored scratch file was written in place, in the checkout.
        self.assertEqual((self.repo / "tmp" / "notes.md").read_text(), "scratch\n")
        self.assertNotIn(REFUSED_WRITE, self.requests[2][len(self.requests[1]):])

        (created,) = self.printer.events_of_type("worktree_created")
        branch = created["branch"]
        self.assertTrue(branch.startswith("kiss/wt-"))
        self.assertIn("Created worktree", self.requests[3])
        self.assertIn("The work dir is now", self.requests[3])

        # The Write after create_worktree landed in the worktree, not the checkout.
        self.assertIn("Successfully wrote", self.requests[4])
        self.assertIn(REFUSED_FINISH, self.requests[5])
        self.assertIn("commit_and_merge", self.requests[5])

        (merged,) = self.printer.events_of_type("worktree_result")
        self.assertEqual((merged["action"], merged["success"], merged["branch"]),
                         ("merge", True, branch))
        self.assertIn("The work dir is", self.requests[6])
        self.assertIn(str(self.repo), self.requests[6])

        self.assertEqual((self.repo / "g.txt").read_text(), "new\n")
        self.assertEqual(run_git(self.repo, "branch", "--list", "kiss/wt-*").stdout.strip(), "")
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
        self.assertEqual(list((self.repo / ".kiss-worktrees").glob("kiss_wt-*")), [])
        self.assertEqual(self.printer.events_of_type("worktree_done"), [])
        results = self.printer.events_of_type("result")
        self.assertTrue(results and results[-1].get("success") is not False, results)


class TestUnmergedWorktreeIsPresentedToTheUser(_ScriptedRun):
    """A second ``finish`` passes; the pending worktree reaches the Merge bar."""

    script = [
        tool_call_response("create_worktree", {}),
        _write("g.txt", "new\n"),
        finish_response("first finish"),
        finish_response("second finish"),
    ]

    def test_pending_worktree(self) -> None:
        self._run()
        self.assertEqual(len(self.requests), 4, [r[-300:] for r in self.requests])
        self.assertIn(REFUSED_FINISH, self.requests[3])

        (done,) = self.printer.events_of_type("worktree_done")
        self.assertEqual(done["changedFiles"], ["g.txt"])
        current = run_git(self.repo, "branch", "--show-current").stdout.strip()
        self.assertEqual(done["originalBranch"], current)
        # Untouched checkout, change only on the task branch.
        self.assertFalse((self.repo / "g.txt").exists())
        branches = run_git(
            self.repo, "branch", "--list", "--format=%(refname:short)", "kiss/wt-*",
        ).stdout.split()
        self.assertEqual(branches, [done["branch"]])
        self.assertEqual(self.printer.events_of_type("worktree_result"), [])


class TestCreateWorktreeIsIdempotentAndRefusesOutsideARepo(_ScriptedRun):
    """A second ``create_worktree`` keeps the first worktree; no repo, no worktree."""

    script = [
        tool_call_response("create_worktree", {}),
        tool_call_response("create_worktree", {}),
        tool_call_response("commit_and_merge", {}),
        tool_call_response("commit_and_merge", {}),
    ]

    def test_second_calls_are_no_ops(self) -> None:
        self._run()
        self.assertEqual(len(self.requests), 5, [r[-300:] for r in self.requests])
        self.assertIn("A worktree is already active", self.requests[2])
        self.assertEqual(len(self.printer.events_of_type("worktree_created")), 1)
        self.assertIn("No worktree is active", self.requests[4])
        self.assertEqual(len(self.printer.events_of_type("worktree_result")), 1)
        self.assertEqual(run_git(self.repo, "branch", "--list", "kiss/wt-*").stdout.strip(), "")

    def test_outside_a_git_repo_edits_in_place(self) -> None:
        scratch = self.home.tmpdir / "plain"
        scratch.mkdir()
        self.script = [
            tool_call_response("create_worktree", {}),
            _write("note.txt", "hello\n"),
        ]
        self.server.work_dir = str(scratch)
        self.server._run_task({
            "type": "run",
            "tabId": "plain-tab",
            "prompt": f"{_MARKER}: write note.txt",
            "model": STANDIN_MODEL,
            "workDir": str(scratch),
            "isParallel": False,
            "useWebTools": False,
            "classifyTasks": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        })
        self.assertEqual(len(self.requests), 3, [r[-300:] for r in self.requests])
        self.assertIn("not inside a git repository", self.requests[1])
        self.assertIn("Successfully wrote", self.requests[2])
        self.assertEqual((scratch / "note.txt").read_text(), "hello\n")
        self.assertEqual(self.printer.events_of_type("worktree_created"), [])


class TestSequentialRunsOnOneTab(_ScriptedRun):
    """A second task on the tab gets its own worktree; the first is preserved."""

    script = [
        tool_call_response("create_worktree", {}),
        _write("first.txt", "1\n"),
        finish_response("first, unmerged"),
        finish_response("first, unmerged"),
        tool_call_response("create_worktree", {}),
        _write("second.txt", "2\n"),
        finish_response("second, unmerged"),
        finish_response("second, unmerged"),
    ]

    def test_second_run_does_not_inherit_the_pending_worktree(self) -> None:
        self._run()
        self._run()
        self.assertEqual(len(self.requests), 8, [r[-300:] for r in self.requests])
        first, second = self.printer.events_of_type("worktree_created")
        self.assertNotEqual(first["branch"], second["branch"])
        self.assertIn("Created worktree", self.requests[5])
        # The checkout never received either file.
        self.assertFalse((self.repo / "first.txt").exists())
        self.assertFalse((self.repo / "second.txt").exists())
        self.assertEqual(GitWorktreeOps.status_porcelain(self.repo), "")
        # The first branch was committed and parked; the second is pending.
        self.assertEqual(
            run_git(self.repo, "show", f"{first['branch']}:first.txt").stdout, "1\n",
        )
        self.assertFalse(list((self.repo / ".kiss-worktrees").glob(f"*{first['branch'][8:]}*")))
        self.assertTrue(GitWorktreeOps.load_preserve_marker(self.repo, first["branch"]))
        self.assertTrue(GitWorktreeOps.load_preserve_marker(self.repo, second["branch"]))
        done = self.printer.events_of_type("worktree_done")
        self.assertEqual([d["branch"] for d in done], [first["branch"], second["branch"]])
        self.assertEqual(done[1]["changedFiles"], ["second.txt"])


class TestFailedMergeKeepsTheWorktreeUsable(_ScriptedRun):
    """A rejected merge commit hands the worktree back; a retry merges."""

    def setUp(self) -> None:
        super().setUp()
        self.hook = self.repo / ".git" / "hooks" / "pre-commit"
        self.script = [
            tool_call_response("create_worktree", {}),
            _write("g.txt", "new\n"),
            self._install_rejecting_hook,
            self._remove_hook,
            finish_response("merged after the retry"),
        ]

    def _install_rejecting_hook(self) -> dict[str, Any]:
        # Rejects commits on the user's branch only: the task branch
        # commits, the squash-merge commit on the original branch fails.
        self.hook.write_text(
            "#!/bin/sh\ncase \"$(git symbolic-ref --short HEAD)\" in kiss/wt-*) exit 0;; esac\n"
            "echo rejected >&2\nexit 1\n"
        )
        self.hook.chmod(self.hook.stat().st_mode | stat.S_IXUSR)
        return tool_call_response("commit_and_merge", {})

    def _remove_hook(self) -> dict[str, Any]:
        os.unlink(self.hook)
        return tool_call_response("commit_and_merge", {})

    def test_retry_after_the_hook_is_fixed(self) -> None:
        self._run()
        self.assertEqual(len(self.requests), 5, [r[-300:] for r in self.requests])
        failed, merged = self.printer.events_of_type("worktree_result")
        self.assertFalse(failed["success"])
        self.assertIn("checked out again", self.requests[3])
        self.assertIn("call commit_and_merge again", self.requests[3])
        self.assertTrue(merged["success"])
        self.assertEqual((self.repo / "g.txt").read_text(), "new\n")
        self.assertEqual(run_git(self.repo, "branch", "--list", "kiss/wt-*").stdout.strip(), "")
        self.assertEqual(list((self.repo / ".kiss-worktrees").glob("kiss_wt-*")), [])


class TestFailedMergeFromAnIgnoredSubdirectory(_ScriptedRun):
    """Reattaching recreates a work dir that git does not restore.

    The task runs in the ignored ``tmp/`` subdirectory of the repo.
    After the rejected merge removed the worktree, the reattached
    checkout has no ``tmp/`` (git restores tracked files only), so the
    tool must create it before pointing the tools back at it.
    """

    def setUp(self) -> None:
        super().setUp()
        (self.repo / "tmp").mkdir()
        self.hook = self.repo / ".git" / "hooks" / "pre-commit"
        self.work_dir_after_failure: str | None = None
        self.work_dir_existed = False
        self.script = [
            tool_call_response("create_worktree", {}),
            self._write_tracked_file_next_to_tmp,
            self._install_rejecting_hook,
            self._inspect_work_dir_then_remove_hook,
            finish_response("merged after the retry"),
        ]

    def _agent(self) -> ChatSorcarAgent:
        with agent_state.STATE_LOCK:
            (state,) = [s for s in agent_state.agent_states.values() if s.tab_id == "wt-tools-tab"]
        assert isinstance(state.agent, ChatSorcarAgent)
        return state.agent

    def _write_tracked_file_next_to_tmp(self) -> dict[str, Any]:
        return _write(str(Path(self._agent().work_dir).parent / "g.txt"), "new\n")

    def _install_rejecting_hook(self) -> dict[str, Any]:
        self.hook.write_text(
            "#!/bin/sh\ncase \"$(git symbolic-ref --short HEAD)\" in kiss/wt-*) exit 0;; esac\n"
            "echo rejected >&2\nexit 1\n"
        )
        self.hook.chmod(self.hook.stat().st_mode | stat.S_IXUSR)
        return tool_call_response("commit_and_merge", {})

    def _inspect_work_dir_then_remove_hook(self) -> dict[str, Any]:
        self.work_dir_after_failure = self._agent().work_dir
        self.work_dir_existed = Path(self.work_dir_after_failure).is_dir()
        os.unlink(self.hook)
        return tool_call_response("commit_and_merge", {})

    def _run(self) -> None:
        self.server._run_task({
            "type": "run",
            "tabId": "wt-tools-tab",
            "prompt": f"{_MARKER}: add g.txt from tmp",
            "model": STANDIN_MODEL,
            "workDir": str(self.repo / "tmp"),
            "isParallel": False,
            "useWebTools": False,
            "classifyTasks": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
        })

    def test_reattached_work_dir_exists(self) -> None:
        self._run()
        self.assertEqual(len(self.requests), 5, [r[-300:] for r in self.requests])
        (created,) = self.printer.events_of_type("worktree_created")
        self.assertEqual(
            created["worktreeWorkDir"], os.path.join(created["worktreeDir"], "tmp")
        )
        self.assertIn("checked out again", self.requests[3])
        self.assertEqual(self.work_dir_after_failure, created["worktreeWorkDir"])
        self.assertTrue(self.work_dir_existed, "reattach did not recreate the work dir")
        failed, merged = self.printer.events_of_type("worktree_result")
        self.assertEqual((failed["success"], merged["success"]), (False, True))
        self.assertEqual((self.repo / "g.txt").read_text(), "new\n")
        self.assertEqual(list((self.repo / ".kiss-worktrees").glob("kiss_wt-*")), [])


class TestPreserveMarkerDoesNotWaitForRepoLocks(unittest.TestCase):
    """The end-of-run preserve marker is queued behind no lock at all.

    Another thread (standing in for a sibling task's merge, whose
    conflict-resolving agent can hold ``repo_lock`` + the cross-process
    reclaim lock for minutes, and whose git commands hold
    ``.git/config.lock``) holds all three; the marker is a file of its
    own, so it lands while the locks are still held, is durable (a
    fresh process sees it), and this task's run can finish.
    """

    # The marker write runs one ``git rev-parse``, which a loaded Windows
    # box has taken most of a second to answer.
    HOLD_SECONDS = 3.0

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-preserve-marker-lock-")
        self.repo = self.home.repo
        self.agent = ChatSorcarAgent("wt")
        self.agent.work_dir = str(self.repo)
        self.assertIn("Created worktree", self.agent.create_worktree())

    def tearDown(self) -> None:
        if self.agent._wt is not None:
            self.agent.discard()
        self.home.cleanup()

    def _hold_locks(self, released_at: list[float]) -> None:
        config_lock = self.repo / ".git" / "config.lock"
        config_lock.write_text("")
        try:
            with repo_lock(self.repo), _reclaim_process_lock(self.repo):
                time.sleep(self.HOLD_SECONDS)
                released_at.append(time.monotonic())
        finally:
            config_lock.unlink()

    def test_marker_is_written_while_the_locks_are_held(self) -> None:
        wt = self.agent._wt
        assert wt is not None
        released_at: list[float] = []
        holder = threading.Thread(target=self._hold_locks, args=(released_at,))
        holder.start()
        time.sleep(0.1)  # let the holder take the locks first
        self.agent._mark_unmerged_worktree_preserved()
        written_at = time.monotonic()
        marked = GitWorktreeOps.load_preserve_marker(self.repo, wt.branch)
        durable = subprocess.run(
            [
                sys.executable, "-c",
                "import sys; from pathlib import Path\n"
                "from kiss.agents.sorcar.git_worktree import GitWorktreeOps\n"
                "print(GitWorktreeOps.load_preserve_marker(Path(sys.argv[1]), sys.argv[2]))",
                str(self.repo), wt.branch,
            ],
            capture_output=True, text=True, timeout=120,
        ).stdout.strip()
        holder.join()
        self.assertTrue(marked)
        self.assertEqual(durable, "True")
        self.assertLess(written_at, released_at[0])


class TestEmptyWorktreeIsDiscardedWithRescue(_ScriptedRun):
    """Ignored-only output is rescued and the empty branch dropped at task end."""

    script = [
        tool_call_response("create_worktree", {}),
        _write("tmp/out.txt", "report\n"),
        finish_response("only ignored output"),
        finish_response("only ignored output"),
    ]

    def test_discard_result_is_broadcast(self) -> None:
        self._run()
        (created,) = self.printer.events_of_type("worktree_created")
        (result,) = self.printer.events_of_type("worktree_result")
        self.assertEqual(
            (result["action"], result["success"], result["branch"], result["tabId"]),
            ("discard", True, created["branch"], "wt-tools-tab"),
        )
        self.assertEqual((self.repo / "tmp" / "out.txt").read_text(), "report\n")
        self.assertEqual(run_git(self.repo, "branch", "--list", "kiss/wt-*").stdout.strip(), "")
        self.assertEqual(self.printer.events_of_type("worktree_done"), [])


class TestToolWiring(unittest.TestCase):
    """Where the two tools live and who gets them."""

    def test_edit_group_and_full_profile_carry_the_tools(self) -> None:
        self.assertTrue(WORKTREE_TOOL_NAMES <= TOOL_GROUPS["edit"])
        self.assertIsNone(TOOL_PROFILES["full"])  # every tool, the worktree tools among them
        self.assertTrue(WORKTREE_TOOL_NAMES <= (resolve_tool_profile("shell+edit") or set()))
        self.assertFalse(WORKTREE_TOOL_NAMES & (resolve_tool_profile("review") or set()))

    def test_every_sorcar_agent_provides_them(self) -> None:
        for agent in (SorcarAgent("plain"), ChatSorcarAgent("chat")):
            names = {agent.create_worktree.__name__, agent.commit_and_merge.__name__}
            self.assertEqual(names, set(WORKTREE_TOOL_NAMES))

    def test_a_worker_sea_edits_in_place(self) -> None:
        agent = ChatSorcarAgent("wt")
        kept = WorkerSea().tools([agent.create_worktree, agent.commit_and_merge, agent.merge])
        self.assertEqual([tool.__name__ for tool in kept], ["merge"])

    def test_dispatch_follows_the_agent_into_its_worktree(self) -> None:
        agent = ChatSorcarAgent("wt")
        self.assertEqual(current_work_dir(None, "/built"), "/built")
        self.assertEqual(current_work_dir(agent, "/built"), "/built")
        agent.work_dir = "/moved"
        self.assertEqual(current_work_dir(agent, "/built"), "/moved")


if __name__ == "__main__":
    unittest.main()
