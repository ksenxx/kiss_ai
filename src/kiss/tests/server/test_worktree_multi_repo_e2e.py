# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the worktree protocol across several repositories.

A task edits its own repository (repoA, the work dir) and a second one
(repoB) discovered on the way: ``create_worktree`` is called once per
repository, the write guard maps repoB's checkout paths into repoB's
worktree, and one ``commit_and_merge`` merges both.  A sub-agent never
creates a worktree for a repository its calling task already isolates.
"""

from __future__ import annotations

import threading
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.agent_dispatch import _isolated_repos_of
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.server.task_runner import _inherited_isolated_repos
from kiss.server.web_server import WebPrinter
from kiss.tests.server.conftest import run_git
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    init_repo,
    request_text,
    tool_call_response,
)
from kiss.tests.server.test_autocommit_wire_toggle import _FakeCredentials

TAB = "wt-multi-repo-tab"


def _write(path: str, content: str) -> dict[str, Any]:
    return tool_call_response("Write", {"file_path": path, "content": content})


def _branches(repo: Path) -> list[str]:
    return run_git(
        repo, "branch", "--list", "--format=%(refname:short)", "kiss/wt-*",
    ).stdout.split()


class _TwoRepoRun(unittest.TestCase):
    """A server run of one scripted task agent on repoA, with repoB beside it."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-worktree-multi-repo-")
        self.repo = self.home.repo
        self.repo_b = self.home.tmpdir / "repoB"
        init_repo(self.repo_b)
        self.home.write_config(max_budget=5.0, use_web_browser=False)
        self.credentials = _FakeCredentials()
        self.requests: list[str] = []
        self._lock = threading.Lock()
        self.script: list[dict[str, Any]] = []
        # Answer to the n-th model call; tests that need a path only
        # known once a worktree exists replace it with a function.
        self.step: Callable[[int], dict[str, Any]] = self._scripted_step
        self.wt_b: Path = Path()
        self.standin = StandInModelServer(self._respond)
        self.printer = CapturePrinter()
        self.server = AgentServer(printer=self.printer)
        self.server.work_dir = str(self.repo)

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            states = list(agent_state.agent_states.values())
        for state in states:
            agent = state.agent
            if agent is not None and getattr(agent, "_wt_pending", False):
                try:
                    agent.discard()
                except Exception:
                    pass
        self.standin.stop()
        self.credentials.restore()
        self.home.cleanup()

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        with self._lock:
            self.requests.append(text)
            index = len(self.requests) - 1
        return self.step(index)

    def _scripted_step(self, index: int) -> dict[str, Any]:
        if index < len(self.script):
            return self.script[index]
        return finish_response("task done")

    def _two_worktrees_then_finish(self, index: int) -> dict[str, Any]:
        """Isolate both repositories, edit a file in each, finish unmerged."""
        if index == 0:
            return tool_call_response("create_worktree", {})
        if index == 1:
            return tool_call_response("create_worktree", {"path": str(self.repo_b)})
        if index == 2:
            wt_b = self._worktree_dir_of(self.repo_b)
            return _write(str(wt_b / "b.txt"), "B\n")
        if index == 3:
            return _write("a.txt", "A\n")
        return finish_response("leaving both unmerged")

    def _run(self, work_dir: Path | None = None, parent_task_id: str = "") -> None:
        self.server._run_task({
            "type": "run",
            "tabId": TAB,
            "prompt": "edit both repositories",
            "model": STANDIN_MODEL,
            "workDir": str(work_dir or self.repo),
            "isParallel": False,
            "useWebTools": False,
            "classifyTasks": False,
            "maxBudget": 5.0,
            "modelConfig": self.standin.model_config,
            "parentTaskId": parent_task_id,
        })

    def _agent(self) -> ChatSorcarAgent:
        state = agent_state.find_by_tab(TAB)
        assert state is not None and state.agent is not None
        return state.agent

    def _worktree_dir_of(self, repo: Path) -> Path:
        """The one worktree directory of *repo* (created by the run), resolved
        the way the product names it (macOS's /var tmpdir is /private/var)."""
        dirs = sorted((repo / ".kiss-worktrees").glob("kiss_wt-*"))
        self.assertEqual(len(dirs), 1, dirs)
        return dirs[0].resolve()


class TestSecondRepositoryIsIsolatedOnDemand(_TwoRepoRun):
    """Guard, per-repo create_worktree, mapped paths, one merge for both."""

    def test_both_repositories_are_merged(self) -> None:
        b_file = str(self.repo_b / "b.txt")
        self.script = [
            tool_call_response("create_worktree", {}),            # 0: repoA, work dir moves
            _write(b_file, "from the task\n"),                    # 1: refused, repoB not isolated
            tool_call_response("create_worktree", {"path": b_file}),  # 2: repoB worktree
            _write(b_file, "from the task\n"),                    # 3: refused, names the wt path
        ]

        def step(index: int) -> dict[str, Any]:
            if index < len(self.script):
                return self.script[index]
            if index == 4:  # the path carries the worktree's slug
                self.wt_b = self._worktree_dir_of(self.repo_b)
                return _write(str(self.wt_b / "b.txt"), "from the task\n")
            if index == 5:
                return _write("a.txt", "in repo A\n")
            if index == 6:
                return tool_call_response("commit_and_merge", {})
            return finish_response("merged both")

        self.step = step
        self._run()
        reqs = self.requests
        self.assertEqual(len(reqs), 8, [r[-200:] for r in reqs])
        self.assertIn("Created worktree", reqs[1])
        self.assertIn("work dir is now", reqs[1])
        # The refusal names the repository as git resolves it (macOS's
        # /var/folders tmpdir is really /private/var/folders).
        self.assertIn(f"Call create_worktree({str(self.repo_b.resolve())!r}) first", reqs[2])
        self.assertIn("Created worktree", reqs[3])
        self.assertIn("The work dir is unchanged", reqs[3])
        self.assertIn(f"Write {self.wt_b / 'b.txt'} instead", reqs[4])
        # The write under repoB's worktree and the one into repoA's
        # worktree (relative path) both passed, then one
        # commit_and_merge merged both repositories.
        self.assertIn("Successfully merged branch", reqs[-1])
        results = self.printer.events_of_type("worktree_result")
        self.assertEqual([r["success"] for r in results], [True, True])
        self.assertEqual({r["action"] for r in results}, {"merge"})
        self.assertEqual((self.repo / "a.txt").read_text(), "in repo A\n")
        self.assertEqual((self.repo_b / "b.txt").read_text(), "from the task\n")
        for repo in (self.repo, self.repo_b):
            self.assertEqual(_branches(repo), [], repo)
            self.assertEqual(list((repo / ".kiss-worktrees").glob("kiss_wt-*")), [])
        self.assertEqual(self.printer.events_of_type("worktree_done"), [])


class TestUnmergedMultiRepoWorkReachesTheMergeBar(_TwoRepoRun):
    """Both pending worktrees are presented together and merged by one action."""

    def test_worktree_done_lists_both_and_merge_acts_on_both(self) -> None:
        self.step = self._two_worktrees_then_finish
        self._run()
        self.assertEqual(len(self.requests), 6, [r[-200:] for r in self.requests])

        (done,) = self.printer.events_of_type("worktree_done")
        agent = self._agent()
        self.assertEqual(len(agent.worktrees), 2)
        wt_a, wt_b = agent.worktrees
        self.assertEqual(wt_a.repo_root.resolve(), self.repo.resolve())
        self.assertEqual(wt_b.repo_root.resolve(), self.repo_b.resolve())
        # The singular fields describe the work dir's worktree (repoA).
        self.assertEqual(done["branch"], wt_a.branch)
        self.assertEqual(done["worktreeDir"], str(wt_a.wt_dir))
        self.assertEqual(
            done["changedFiles"], sorted(["a.txt", f"{wt_b.repo_root}/b.txt"]),
        )
        self.assertEqual(
            [(w["repoRoot"], w["branch"], w["changedFiles"]) for w in done["worktrees"]],
            [
                (str(wt_a.repo_root), wt_a.branch, ["a.txt"]),
                (str(wt_b.repo_root), wt_b.branch, ["b.txt"]),
            ],
        )
        self.assertFalse(done["hasConflict"])
        self.assertFalse((self.repo / "a.txt").exists())
        self.assertFalse((self.repo_b / "b.txt").exists())

        reply = self.server._handle_worktree_action("merge", TAB)
        self.assertTrue(reply["success"], reply)
        self.assertFalse(agent._wt_pending)
        self.assertEqual((self.repo / "a.txt").read_text(), "A\n")
        self.assertEqual((self.repo_b / "b.txt").read_text(), "B\n")
        self.assertEqual(_branches(self.repo), [])
        self.assertEqual(_branches(self.repo_b), [])

    def test_discard_throws_both_away(self) -> None:
        self.step = self._two_worktrees_then_finish
        self._run()
        agent = self._agent()
        branches = [wt.branch for wt in agent.worktrees]
        self.assertEqual(len(branches), 2)

        reply = self.server._handle_worktree_action("discard", TAB)
        self.assertTrue(reply["success"], reply)
        self.assertFalse(agent._wt_pending)
        for branch in branches:
            self.assertIn(f"Discarded branch '{branch}'", reply["message"])
        self.assertFalse((self.repo / "a.txt").exists())
        self.assertFalse((self.repo_b / "b.txt").exists())
        self.assertEqual(_branches(self.repo), [])
        self.assertEqual(_branches(self.repo_b), [])


class TestSubAgentReusesTheCallingTasksWorktrees(_TwoRepoRun):
    """A sub-agent never isolates a repository its calling task already did."""

    def setUp(self) -> None:
        super().setUp()
        # The calling task: its work dir moved into repoA's worktree and
        # it also isolates repoB.  It is registered as a running task
        # so the daemon hands its worktrees to the child.
        self.parent = ChatSorcarAgent("parent")
        self.parent.work_dir = str(self.repo)
        self.parent._run_tool_names = frozenset({"create_worktree"})
        self.assertIn("Created worktree", self.parent.create_worktree())
        self.assertIn("Created worktree", self.parent.create_worktree(str(self.repo_b)))
        self.parent_task_id = "0123456789abcdef0123456789abcdef"
        agent_state.register(AgentState(
            self.parent_task_id, agent=self.parent, tab_id="parent-tab", is_task_active=True,
        ))

    def tearDown(self) -> None:
        agent_state.unregister(self.parent_task_id)
        self.parent.discard()
        super().tearDown()

    def test_child_edits_the_parents_worktrees(self) -> None:
        wt_a, wt_b = self.parent.worktrees
        b_file = str(self.repo_b / "b.txt")
        self.script = [
            tool_call_response("create_worktree", {"path": str(self.repo_b)}),  # 0
            _write(b_file, "child\n"),                                        # 1: mapped path
            _write(str(wt_b.wt_dir / "b.txt"), "child\n"),                    # 2: passes
            tool_call_response("create_worktree", {}),                        # 3: inside repoA's
            _write("a.txt", "child\n"),                                       # 4: passes
        ]
        children: list[ChatSorcarAgent] = []

        def step(index: int) -> dict[str, Any]:
            # The daemon drops a finished sub-task's agent: grab it live.
            children.append(self._agent())
            return self._scripted_step(index)

        self.step = step
        self._run(work_dir=wt_a.wt_dir, parent_task_id=self.parent_task_id)
        reqs = self.requests
        self.assertEqual(len(reqs), 6, [r[-200:] for r in reqs])
        self.assertIn(
            f"the calling task already isolates {self.repo_b.resolve()} in the "
            f"worktree {wt_b.wt_dir}", reqs[1],
        )
        self.assertIn(f"Write {wt_b.wt_dir / 'b.txt'} instead", reqs[2])
        self.assertIn("already inside the worktree of the calling task", reqs[4])
        self.assertEqual((wt_b.wt_dir / "b.txt").read_text(), "child\n")
        self.assertEqual((wt_a.wt_dir / "a.txt").read_text(), "child\n")
        # No worktree of its own, nothing to merge, no bar for the child.
        child = children[-1]
        self.assertEqual(child.worktrees, [])
        self.assertEqual(
            child.isolated_repos(),
            {self.repo.resolve(): wt_a.wt_dir, self.repo_b.resolve(): wt_b.wt_dir},
        )
        self.assertEqual(self.printer.events_of_type("worktree_done"), [])
        self.assertEqual(len(_branches(self.repo)), 1)
        self.assertEqual(len(_branches(self.repo_b)), 1)
        self.assertFalse((self.repo_b / "b.txt").exists())


class TestAgentLevelMultiRepo(unittest.TestCase):
    """Direct agent checks that need no server: idempotence, inheritance, merge order."""

    def setUp(self) -> None:
        self.home = IsolatedKissHome(prefix="kiss-worktree-multi-agent-")
        self.repo = self.home.repo
        self.repo_b = self.home.tmpdir / "repoB"
        init_repo(self.repo_b)
        self.agent = ChatSorcarAgent("t")
        self.agent.work_dir = str(self.repo)
        self.agent._run_tool_names = frozenset({"create_worktree"})

    def tearDown(self) -> None:
        if self.agent._wt_pending:
            self.agent.discard()
        self.home.cleanup()

    def test_second_call_for_an_isolated_repo_names_its_worktree(self) -> None:
        self.agent.create_worktree(str(self.repo_b))
        (wt_b,) = self.agent.worktrees
        again = self.agent.create_worktree(str(self.repo_b / "sub" / "deep.txt"))
        self.assertIn(f"already active for {wt_b.repo_root}: {wt_b.wt_dir}", again)
        self.assertEqual(len(self.agent.worktrees), 1)
        self.assertEqual(self.agent.work_dir, str(self.repo))
        # A path inside the worktree itself is already isolated.
        inside = self.agent.create_worktree(str(wt_b.wt_dir / "x.txt"))
        self.assertIn("already active", inside)
        # Outside any repository: nothing to isolate.
        self.assertIn("not inside a git repository", self.agent.create_worktree("/"))

    def test_guard_maps_a_secondary_checkout_path(self) -> None:
        self.agent.create_worktree(str(self.repo_b))
        (wt_b,) = self.agent.worktrees
        refusal = self.agent._guard_worktree(
            "Edit", {"file_path": str(self.repo_b / "seed.txt")},
        )
        assert refusal is not None
        self.assertIn(f"Edit {wt_b.wt_dir / 'seed.txt'} instead", refusal)
        self.assertIsNone(self.agent._guard_worktree(
            "Edit", {"file_path": str(wt_b.wt_dir / "seed.txt")},
        ))
        # The work dir's repository is still unisolated: asks for the no-arg call.
        own = self.agent._guard_worktree("Write", {"file_path": "new.txt"})
        assert own is not None
        self.assertIn("Call create_worktree() first", own)
        gate = self.agent._guard_worktree("finish", {})
        assert gate is not None
        self.assertIn(f"worktree on branch '{wt_b.branch}' has not been merged", gate)

    def test_inherited_repo_is_never_isolated_again(self) -> None:
        parent_wt = self.repo_b / ".kiss-worktrees" / "kiss_wt-parent"
        self.agent.inherit_worktrees({self.repo_b: parent_wt})
        msg = self.agent.create_worktree(str(self.repo_b / "b.txt"))
        self.assertIn(f"already isolates {self.repo_b.resolve()} in the worktree {parent_wt}", msg)
        self.assertEqual(self.agent.worktrees, [])
        refusal = self.agent._guard_worktree("Write", {"file_path": str(self.repo_b / "b.txt")})
        assert refusal is not None
        self.assertIn(f"Write {parent_wt / 'b.txt'} instead", refusal)
        # Its own repository is unaffected and the map is passed down whole.
        self.assertIn("Created worktree", self.agent.create_worktree())
        (wt_a,) = self.agent.worktrees
        self.assertEqual(
            self.agent.isolated_repos(),
            {self.repo_b.resolve(): parent_wt, self.repo.resolve(): wt_a.wt_dir},
        )

    def test_finish_gate_names_every_branch_and_leave_as_is_keeps_both(self) -> None:
        self.agent.create_worktree()
        self.agent.create_worktree(str(self.repo_b))
        wt_a, wt_b = self.agent.worktrees
        gate = self.agent._guard_worktree("finish", {})
        assert gate is not None
        self.assertIn(f"worktrees on branch '{wt_a.branch}', '{wt_b.branch}'", gate)
        self.assertIsNone(self.agent._guard_worktree("finish", {}))
        (wt_a.wt_dir / "a.txt").write_text("A\n")
        (wt_b.wt_dir / "b.txt").write_text("B\n")
        msg = self.agent.leave_as_is()
        for wt in (wt_a, wt_b):
            self.assertIn(f"Left branch '{wt.branch}'", msg)
            self.assertTrue(GitWorktreeOps.load_preserve_marker(wt.repo_root, wt.branch))
            self.assertTrue(wt.wt_dir.exists())
        self.assertEqual(self.agent.worktrees, [])

    def test_a_failed_merge_keeps_only_that_worktree_pending(self) -> None:
        self.agent.create_worktree()
        self.agent.create_worktree(str(self.repo_b))
        wt_a, wt_b = self.agent.worktrees
        (wt_a.wt_dir / "a.txt").write_text("A\n")
        (wt_b.wt_dir / "seed.txt").write_text("task\n")
        # repoB's checkout moves on in a conflicting way.
        (self.repo_b / "seed.txt").write_text("user\n")
        run_git(self.repo_b, "commit", "-qam", "user edit")
        msg = self.agent.merge()
        self.assertIn(f"Successfully merged branch '{wt_a.branch}'", msg)
        self.assertNotIn(f"Successfully merged branch '{wt_b.branch}'", msg)
        self.assertEqual([wt.branch for wt in self.agent.worktrees], [wt_b.branch])
        self.assertEqual((self.repo / "a.txt").read_text(), "A\n")
        self.assertEqual((self.repo_b / "seed.txt").read_text(), "user\n")

    def test_every_worktree_is_protected_from_reclaim(self) -> None:
        self.agent.create_worktree()
        self.agent.create_worktree(str(self.repo_b))
        state = AgentState("multi-live-1", agent=self.agent, tab_id="t", is_task_active=True)
        agent_state.register(state)
        try:
            self.assertEqual(
                JsonPrinter().live_worktree_branches(),
                {wt.branch for wt in self.agent.worktrees},
            )
        finally:
            agent_state.unregister("multi-live-1", state)

    def test_isolation_map_travels_the_wire(self) -> None:
        self.agent.inherit_worktrees({self.repo_b: self.repo_b / "wt-parent"})
        self.agent.create_worktree()
        (wt_a,) = self.agent.worktrees
        wire = _isolated_repos_of(self.agent)
        self.assertEqual(
            wire,
            {str(self.repo_b.resolve()): str(self.repo_b / "wt-parent"),
             str(self.repo.resolve()): str(wt_a.wt_dir)},
        )
        self.assertEqual(_isolated_repos_of(None), {})
        # The daemon merges the wire map with the live parent's (which
        # may have grown since the dispatch) and ignores junk.
        child = ChatSorcarAgent("child")
        child.inherit_worktrees(_inherited_isolated_repos(
            {"isolatedRepos": {**wire, "": "x", 3: "y"}}, "",
        ))
        self.assertEqual(child.isolated_repos(), {
            self.repo_b.resolve(): self.repo_b / "wt-parent",
            self.repo.resolve(): wt_a.wt_dir,
        })
        state = AgentState("multi-parent-1", agent=self.agent, tab_id="p", is_task_active=True)
        agent_state.register(state)
        try:
            self.agent.create_worktree(str(self.repo_b))  # no-op: inherited
            merged = _inherited_isolated_repos({"isolatedRepos": "junk"}, "multi-parent-1")
        finally:
            agent_state.unregister("multi-parent-1", state)
        self.assertEqual(merged, self.agent.isolated_repos())

    def test_nested_repository_worktree_is_not_remapped_into_the_outer_one(self) -> None:
        # repoC lives under repoA's git-ignored tmp/ and is isolated on
        # its own; its worktree path must reach repoC's worktree, not
        # repoA's copy of the path, for Write and for Bash.
        (self.repo / ".gitignore").write_text("tmp/\n")
        run_git(self.repo, "add", ".gitignore")
        run_git(self.repo, "commit", "-qm", "ignore tmp")
        repo_c = self.repo / "tmp" / "repoC"
        init_repo(repo_c)
        self.agent.create_worktree()
        self.agent.create_worktree(str(repo_c))
        wt_a, wt_c = self.agent.worktrees
        self.assertEqual(wt_c.repo_root.resolve(), repo_c.resolve())
        tools = UsefulTools(work_dir=str(wt_a.work_dir))
        target = wt_c.wt_dir / "c.txt"
        self.assertIsNone(self.agent._guard_worktree("Write", {"file_path": str(target)}))
        tools.Write(str(target), "C\n")
        self.assertEqual(target.read_text(), "C\n")
        self.assertFalse((wt_a.wt_dir / "tmp" / "repoC").exists())
        out = tools.Bash(f"cat {target}", "read it back")
        self.assertIn("C", out)
        self.assertNotIn("parent-repo path", out)
        # repoC's checkout is still guarded: the mapped path is named.
        refusal = self.agent._guard_worktree("Write", {"file_path": str(repo_c / "c.txt")})
        assert refusal is not None
        self.assertIn(f"Write {wt_c.wt_dir / 'c.txt'} instead", refusal)


class TestRemoteFallbackDirSurvivesSecondaryWorktrees(unittest.TestCase):
    """The per-tab file-link fallback stays on the work dir's worktree."""

    def setUp(self) -> None:
        self.printer = WebPrinter()
        self.printer.subscribe_tab("task-1", "tab-1")

    def _broadcast(self, **event: Any) -> None:
        self.printer.broadcast({"taskId": "task-1", **event})

    def test_secondary_created_and_result_leave_the_primary_alone(self) -> None:
        self._broadcast(
            type="worktree_created", worktreeDir="/a/.kiss-worktrees/kiss_wt-a",
            worktreeWorkDir="/a/.kiss-worktrees/kiss_wt-a/sub",
        )
        self._broadcast(type="worktree_created", worktreeDir="/b/.kiss-worktrees/kiss_wt-b")
        self.assertEqual(
            self.printer.worktree_dir_for_tab("tab-1"), "/a/.kiss-worktrees/kiss_wt-a/sub",
        )
        # repoB merged first (commit_and_merge goes repository by repository).
        self._broadcast(
            type="worktree_result", action="merge", success=True,
            worktreeDir="/b/.kiss-worktrees/kiss_wt-b", branch="kiss/wt-b",
        )
        self.assertEqual(
            self.printer.worktree_dir_for_tab("tab-1"), "/a/.kiss-worktrees/kiss_wt-a/sub",
        )
        self._broadcast(
            type="worktree_result", action="merge", success=True,
            worktreeDir="/a/.kiss-worktrees/kiss_wt-a", branch="kiss/wt-a",
        )
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-1"), "")

    def test_secondary_alone_is_recorded_and_a_bar_result_clears_all(self) -> None:
        self._broadcast(type="worktree_created", worktreeDir="/b/.kiss-worktrees/kiss_wt-b")
        self.assertEqual(
            self.printer.worktree_dir_for_tab("tab-1"), "/b/.kiss-worktrees/kiss_wt-b",
        )
        self._broadcast(type="worktree_result", action="discard", success=True)
        self.assertEqual(self.printer.worktree_dir_for_tab("tab-1"), "")


if __name__ == "__main__":
    unittest.main()
