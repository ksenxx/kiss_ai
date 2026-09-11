# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Per-repo main-tree claims close the mutator-vs-task-start TOCTOU.

Review findings 2 and 3 (gpt-5.6-sol): the main-tree Discard
(``_handle_main_tree_action``) and the manual Git Commit
(``_cmd_autocommit_action``) checked ``_any_non_wt_running`` under
``STATE_LOCK`` but published NO per-repository claim afterwards — a
direct (non-worktree) task admitted between the check and the ``git
reset --hard`` / ``git add -A`` had its half-written files reset or
staged out from under it.

The fix publishes a per-repo main-tree claim in the SAME locked
section as the busy check; non-worktree task admission
(``_run_task_inner``) refuses to start while a claim is held, and the
mutators refuse while another mutator's claim is held.

These tests drive the real :class:`VSCodeServer` command handlers
against a real git repository, exactly like the sibling autocommit
tests: only the LLM-backed commit-message composer and the agent's
LLM run are replaced by deterministic functions — no mocks.  The
composer stub BLOCKS on an event, deterministically holding the
manual-commit worker (and therefore its claim) mid-flight; pre-fix, a
task started during that window with no refusal (the discriminating
assertion of ``test_task_start_refused_while_manual_commit_in_flight``).

Branch-coverage note (unreachable-without-doubles exception): the
``threading.Thread(...).start()`` failure branch in
``_cmd_autocommit_action`` (which releases the claims when the worker
never ran) requires the OS to refuse spawning a thread — not
reachable in a real end-to-end setup.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as _persistence
import kiss.server.merge_flow as _merge_flow_module
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )


def _init_repo(repo: str) -> None:
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n")
    _run_git(repo, "add", "seed.txt")
    _run_git(repo, "commit", "-q", "-m", "seed")


class TestMainTreeClaims(unittest.TestCase):
    """Mutator claims vs direct task admission, in both directions."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-main-tree-claim-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        _init_repo(self.repo)

        self._saved_db = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None

        self.server = VSCodeServer()
        self.server.work_dir = self.repo
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        # Deterministic, GATED commit-message generation: the manual
        # commit worker blocks inside it (holding its main-tree claim)
        # until the test releases the gate.
        self._orig_gen = _merge_flow_module.generate_commit_message_from_diff
        self.gen_entered = threading.Event()
        self.gen_release = threading.Event()

        def gated_compose(
            diff_text: str,
            user_prompt: str | None = None,
            task_result: str | None = None,
        ) -> str:
            self.gen_entered.set()
            assert self.gen_release.wait(timeout=30), "gate never released"
            return "test: deterministic commit"

        _merge_flow_module.generate_commit_message_from_diff = gated_compose  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

        def stub_run(_agent: object, **kwargs: object) -> str:
            work_dir = kwargs.get("work_dir")
            if isinstance(work_dir, str) and work_dir:
                (Path(work_dir) / "agent_out.txt").write_text("agent output\n")
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self.gen_release.set()
        self._wait_for_idle_commit_worker()
        self._parent_class.run = self._original_run
        _merge_flow_module.generate_commit_message_from_diff = self._orig_gen
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- helpers ---------------------------------------------------

    def _dirty_repo(self) -> None:
        Path(self.repo, "seed.txt").write_text("modified\n")

    def _run_direct_task(self, tab_id: str) -> None:
        """Run a non-worktree task synchronously through the real gate."""
        self.server._run_task_inner({
            "prompt": "make a change",
            "workDir": self.repo,
            "tabId": tab_id,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        })
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
            if (
                state is not None
                and state.task_thread is threading.current_thread()
            ):
                state.task_thread = None
                state.is_task_active = False

    def _events_of(self, event_type: str) -> list[dict[str, Any]]:
        with self._events_lock:
            return [e for e in self.events if e.get("type") == event_type]

    def _wait_for_idle_commit_worker(self, timeout: float = 10.0) -> None:
        """Poll until every in-flight autocommit worker has finished."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._state_lock:
                if not self.server._autocommit_tabs:
                    return
            time.sleep(0.01)
        raise AssertionError("autocommit worker never finished")

    def _start_gated_manual_commit(self, tab_id: str = "tab-commit") -> None:
        """Dispatch a manual commit and wait until it holds its claim."""
        self._dirty_repo()
        self.server._cmd_autocommit_action({
            "tabId": tab_id, "workDir": self.repo,
        })
        self.assertTrue(
            self.gen_entered.wait(timeout=10),
            "manual-commit worker never reached message generation",
        )

    # -- finding 3: manual commit vs task start --------------------

    def test_task_start_refused_while_manual_commit_in_flight(self) -> None:
        """The discriminating regression: pre-fix the task was admitted.

        While the manual-commit worker is between its dispatch-time
        busy check and its ``git commit``, a direct task must be
        refused — pre-fix it started and its writes could be staged
        into the user's commit.
        """
        self._start_gated_manual_commit()

        self._run_direct_task("tab-task")
        errors = self._events_of("error")
        self.assertTrue(
            any("manual commit is in progress" in str(e.get("text", ""))
                for e in errors),
            f"expected a refusal error, got events: {self.events}",
        )
        # The refused task never ran: the stub agent wrote nothing.
        self.assertFalse(Path(self.repo, "agent_out.txt").exists())

        # Release the worker; the commit completes and the claim is
        # withdrawn.
        self.gen_release.set()
        self._wait_for_idle_commit_worker()
        done = self._events_of("autocommit_done")
        self.assertTrue(done and done[-1].get("success") is True, done)
        log = _run_git(self.repo, "log", "--oneline").stdout
        self.assertIn("test: deterministic commit", log)

        # With the claim released the same task starts and runs.
        self._run_direct_task("tab-task-2")
        self.assertTrue(Path(self.repo, "agent_out.txt").exists())

    def test_second_manual_commit_on_other_tab_is_refused(self) -> None:
        """Two tabs, one repo: the second dispatch cannot double-claim."""
        self._start_gated_manual_commit("tab-one")

        events_before = len(self._events_of("autocommit_done"))
        self.server._cmd_autocommit_action({
            "tabId": "tab-two", "workDir": self.repo,
        })
        done = self._events_of("autocommit_done")[events_before:]
        self.assertEqual(len(done), 1, done)
        self.assertFalse(done[0].get("success"))
        self.assertIn(
            "Another operation is modifying", done[0].get("message") or "",
        )

        # A duplicate click on the SAME tab is silently dropped (the
        # per-tab in-flight marker, unchanged behavior) — no extra
        # ``autocommit_done`` and, crucially, no claim wedging.
        self.server._cmd_autocommit_action({
            "tabId": "tab-one", "workDir": self.repo,
        })
        self.assertEqual(
            len(self._events_of("autocommit_done")), events_before + 1,
        )

        self.gen_release.set()
        self._wait_for_idle_commit_worker()
        # The single successful commit landed and released the claim:
        # a discard now proceeds (clean tree → friendly no-op).
        result = self.server._handle_main_tree_action(
            "discard", work_dir=self.repo,
        )
        self.assertTrue(result["success"], result)

    # -- finding 2: discard vs task start / other mutators ----------

    def test_discard_refused_while_manual_commit_in_flight(self) -> None:
        """A discard must not interleave with a claimed manual commit.

        The manual-commit dispatch publishes its claim before its
        worker takes the repo lock; a discard arriving in that window
        observes the claim and refuses instead of resetting the tree
        the commit is about to stage.
        """
        # Simulate the dispatch window: claim held, repo lock free —
        # exactly the state ``_cmd_autocommit_action`` leaves between
        # its locked claim and the worker's ``repo_lock`` entry.
        repo = Path(self.repo)
        test_claims: list = []
        with self.server._state_lock:
            self.assertTrue(
                self.server._claim_main_tree(
                    repo, "manual commit", holder=test_claims,
                ),
            )
        try:
            self._dirty_repo()
            result = self.server._handle_main_tree_action(
                "discard", work_dir=self.repo,
            )
            self.assertFalse(result["success"], result)
            self.assertIn("Another operation is modifying", result["message"])
            # Nothing was reset.
            self.assertEqual(
                Path(self.repo, "seed.txt").read_text(), "modified\n",
            )
        finally:
            with self.server._state_lock:
                for _claim in test_claims:
                    self.server._release_main_tree_claim(_claim)

        # Claim released: the discard proceeds and reverts the file.
        result = self.server._handle_main_tree_action(
            "discard", work_dir=self.repo,
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(Path(self.repo, "seed.txt").read_text(), "seed\n")

    def test_task_start_refused_while_discard_claim_held(self) -> None:
        """Task admission refuses during a discard's git-op window."""
        repo = Path(self.repo)
        test_claims: list = []
        with self.server._state_lock:
            self.assertTrue(
                self.server._claim_main_tree(
                    repo, "discard", holder=test_claims,
                )
            )
        try:
            self._run_direct_task("tab-task")
            errors = self._events_of("error")
            self.assertTrue(
                any("discard is in progress" in str(e.get("text", ""))
                    for e in errors),
                f"expected a refusal error, got events: {self.events}",
            )
            self.assertFalse(Path(self.repo, "agent_out.txt").exists())
        finally:
            with self.server._state_lock:
                for _claim in test_claims:
                    self.server._release_main_tree_claim(_claim)

    def test_discard_claims_and_releases_around_its_git_ops(self) -> None:
        """A normal discard still works and withdraws its claim."""
        self._dirty_repo()
        Path(self.repo, "untracked.txt").write_text("junk\n")
        result = self.server._handle_main_tree_action(
            "discard", work_dir=self.repo,
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(Path(self.repo, "seed.txt").read_text(), "seed\n")
        self.assertFalse(Path(self.repo, "untracked.txt").exists())
        # The claim is gone: a direct task starts immediately.
        self._run_direct_task("tab-after-discard")
        self.assertTrue(Path(self.repo, "agent_out.txt").exists())

    def test_manual_commit_outside_a_repo_publishes_no_claim(self) -> None:
        """A non-git workDir short-circuits without touching claims."""
        outside = Path(self.tmpdir) / "not-a-repo"
        outside.mkdir()
        self.server._cmd_autocommit_action({
            "tabId": "tab-outside", "workDir": str(outside),
        })
        self._wait_for_idle_commit_worker()
        done = self._events_of("autocommit_done")
        self.assertTrue(done, self.events)
        self.assertIn("Not a git repository", done[-1].get("message", ""))
