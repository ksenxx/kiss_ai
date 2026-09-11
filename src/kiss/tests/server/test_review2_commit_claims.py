# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Claims must cover the repository a commit ACTUALLY mutates.

Review 2 (gpt-5.6-sol), missed wiring 1 and 2:

* **Manual commit claimed the pre-fallback path.**
  ``_cmd_autocommit_action`` discovered and claimed the repository
  from the submitted ``workDir`` verbatim, but the worker's
  ``_autocommit_changes`` remaps a vanished
  ``.kiss-worktrees/kiss_wt-*`` path to its PARENT repository before
  staging.  A path already stale at dispatch produced NO claim at all
  (``discover_repo`` returned ``None``) while the worker still
  mutated the parent; a path that vanished after dispatch left the
  claim on the worktree while the parent was mutated.  Fixed by
  resolving the effective repository (``_effective_commit_repo``) at
  dispatch, and by making ``_autocommit_changes`` re-run the atomic
  busy-check + claim on the repository it actually resolved whenever
  that differs from the dispatcher's claim.

* **Automatic post-task commits published no claim.**
  The non-worktree task-completion path ran whole-repository
  ``_autocommit_changes`` and the sibling-repository
  ``_autocommit_changed_repos`` pass with no main-tree claim, so a
  direct task admitted during the commit had its half-written output
  swept into the commit (and, in the other direction, a finishing
  task's ``git add -A`` could stage a concurrently running task's
  files).  Fixed by wiring both passes into the same claim protocol:
  each publishes a per-repo "post-task commit" claim in the same
  locked section as a busy check that EXCLUDES the finishing task's
  own still-active admission (reentrancy), refuses/skips when the
  repository is otherwise occupied, and releases in ``finally``.

Real :class:`VSCodeServer` command handlers against real git
repositories, like the round-1 claim tests: only the LLM-backed
commit-message composer and the agent's LLM run are replaced by
deterministic functions (no mocks).  The composer stub BLOCKS on an
event, deterministically holding a commit (and therefore its claim)
mid-flight.

Branch-coverage note (unreachable-without-doubles exception): the
``resolve()``-raises-``OSError`` fallbacks in the claim-repo
comparison require an unresolvable path mid-operation and are not
reachable in a real end-to-end setup.
"""

from __future__ import annotations

import re
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


def _commit_count(repo: str) -> int:
    out = _run_git(repo, "rev-list", "--count", "HEAD").stdout.strip()
    return int(out or "0")


def _head_files(repo: str) -> set[str]:
    out = _run_git(
        repo, "show", "--name-only", "--pretty=format:", "HEAD",
    ).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


class _Base(unittest.TestCase):
    """Real server, real repo, gated composer, prompt-keyed stub agent."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-review2-claims-")
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

        # Deterministic, GATED commit-message generation: a commit
        # worker blocks inside it (holding its main-tree claim) until
        # the test releases the gate.
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

        # Prompt-keyed stub agent: the submitted prompt carries a
        # ``rvw2-<name>`` marker (the runner may wrap the prompt in a
        # template, so the marker is extracted by pattern); the stub
        # writes ``<name>.txt`` in its work dir, and a ``block-…``
        # name additionally parks mid-run until released — a real
        # running non-worktree task.
        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run
        self.blocker_entered = threading.Event()
        self.blocker_release = threading.Event()
        blocker_entered = self.blocker_entered
        blocker_release = self.blocker_release

        def stub_run(_agent: object, **kwargs: object) -> str:
            prompt = str(kwargs.get("prompt_template", "") or "")
            work_dir = kwargs.get("work_dir")
            match = re.search(r"rvw2-([A-Za-z0-9-]+)", prompt)
            name = match.group(1) if match else "out"
            if isinstance(work_dir, str) and work_dir:
                (Path(work_dir) / f"{name}.txt").write_text("agent output\n")
            if name.startswith("block-"):
                blocker_entered.set()
                assert blocker_release.wait(timeout=30), "blocker never freed"
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self.gen_release.set()
        self.blocker_release.set()
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

    def _run_direct_task(
        self,
        tab_id: str,
        prompt: str = "out",
        *,
        work_dir: str | None = None,
        auto_commit: bool = False,
    ) -> None:
        """Run a non-worktree task synchronously through the real gate."""
        self.server._run_task_inner({
            "prompt": f"rvw2-{prompt}",
            "workDir": work_dir or self.repo,
            "tabId": tab_id,
            "useWorktree": False,
            "autoCommit": auto_commit,
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

    def _assert_refused(self, needle: str) -> None:
        errors = self._events_of("error")
        self.assertTrue(
            any(needle in str(e.get("text", "")) for e in errors),
            f"expected a refusal containing {needle!r}, got: {self.events}",
        )

    def _claim_on(self, repo: str) -> str | None:
        with self.server._state_lock:
            return self.server._main_tree_claim_reason(Path(repo))

    def _wait_for_idle_commit_worker(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._state_lock:
                if not self.server._autocommit_tabs:
                    return
            time.sleep(0.01)
        raise AssertionError("autocommit worker never finished")


class TestManualCommitClaimsEffectiveRepo(_Base):
    """Missed wiring 1: the claim must cover the post-fallback repo."""

    def _stale_wt_path(self) -> str:
        return str(
            Path(self.repo) / ".kiss-worktrees" / "kiss_wt-1781574606-49147541",
        )

    def test_stale_worktree_dispatch_claims_the_parent_repo(self) -> None:
        """The reviewer's repro: workDir already stale at dispatch.

        Pre-fix ``discover_repo(stale_path)`` returned ``None`` — no
        busy check, no claim — while the worker fell back to the
        parent repository and mutated it; a direct task admitted
        meanwhile had its half-written output swept into the manual
        commit.  Post-fix the dispatch claims the PARENT and the task
        is refused (the discriminating assertion).
        """
        Path(self.repo, "manual.txt").write_text("manual change\n")
        self.server._cmd_autocommit_action({
            "tabId": "tab-stale", "workDir": self._stale_wt_path(),
        })
        self.assertTrue(
            self.gen_entered.wait(timeout=10),
            "manual-commit worker never reached message generation",
        )
        self.assertEqual(self._claim_on(self.repo), "manual commit")

        before = _commit_count(self.repo)
        self._run_direct_task("tab-task", "task-write")
        self._assert_refused("manual commit is in progress")
        self.assertFalse(Path(self.repo, "task-write.txt").exists())

        self.gen_release.set()
        self._wait_for_idle_commit_worker()
        self.assertIsNone(self._claim_on(self.repo))
        self.assertEqual(_commit_count(self.repo), before + 1)
        self.assertEqual(_head_files(self.repo), {"manual.txt"})

    def test_worker_reclaims_when_the_target_repo_changed(self) -> None:
        """The vanish-after-dispatch half: the worker re-claims.

        The dispatcher claimed the (worktree) path it resolved, but by
        the time the worker runs, the path is stale and the fallback
        targets the parent repository.  The worker must atomically
        busy-check + claim the parent before any git mutation — a
        direct task in the parent is refused while the commit is in
        flight — and release it afterwards, while the dispatcher's
        own claim stays untouched for its ``finally``.
        """
        stale_wt = self._stale_wt_path()
        claimed = Path(stale_wt)
        test_claims: list = []
        with self.server._state_lock:
            self.assertTrue(
                self.server._claim_main_tree(
                    claimed, "manual commit", holder=test_claims,
                ),
            )
        try:
            Path(self.repo, "manual.txt").write_text("manual change\n")
            worker = threading.Thread(
                target=self.server._autocommit_changes,
                args=("tab-vanish",),
                kwargs={
                    "work_dir": stale_wt,
                    "manual": True,
                    "claimed_repo": claimed,
                },
                daemon=True,
            )
            worker.start()
            self.assertTrue(self.gen_entered.wait(timeout=10))
            self.assertEqual(self._claim_on(self.repo), "manual commit")

            self._run_direct_task("tab-task", "task-write")
            self._assert_refused("manual commit is in progress")
            self.assertFalse(Path(self.repo, "task-write.txt").exists())

            self.gen_release.set()
            worker.join(timeout=15)
            self.assertFalse(worker.is_alive(), "commit worker wedged")
            # The parent's claim (taken by the worker) is released;
            # the dispatcher's claim on the stale path is NOT its to
            # release.
            self.assertIsNone(self._claim_on(self.repo))
            with self.server._state_lock:
                self.assertEqual(
                    self.server._main_tree_claim_reason(claimed),
                    "manual commit",
                )
            self.assertEqual(_head_files(self.repo), {"manual.txt"})
        finally:
            with self.server._state_lock:
                for _claim in test_claims:
                    self.server._release_main_tree_claim(_claim)


class TestManualReclaimRefusals(_Base):
    """The worker's re-claim refuses like the dispatch would have."""

    def _stale_wt_path(self) -> str:
        return str(
            Path(self.repo) / ".kiss-worktrees" / "kiss_wt-1781574606-49147541",
        )

    def test_manual_reclaim_refused_while_target_repo_is_busy(self) -> None:
        """Target changed to a repo with a RUNNING task: refuse."""
        blocker = threading.Thread(
            target=self._run_direct_task,
            args=("tab-block", "block-half"),
            daemon=True,
        )
        blocker.start()
        self.assertTrue(self.blocker_entered.wait(timeout=15))

        stale_wt = self._stale_wt_path()
        Path(self.repo, "manual.txt").write_text("manual change\n")
        before = _commit_count(self.repo)
        self.server._autocommit_changes(
            "tab-vanish", work_dir=stale_wt, manual=True,
            claimed_repo=Path(stale_wt),
        )
        self.assertEqual(_commit_count(self.repo), before)
        self.assertFalse(self.gen_entered.is_set())
        done = self._events_of("autocommit_done")
        self.assertTrue(done, self.events)
        self.assertFalse(done[-1].get("success"))
        self.assertIn(
            "wait for it to finish before committing",
            done[-1].get("message", ""),
        )
        self.assertIsNone(self._claim_on(self.repo))

        self.blocker_release.set()
        blocker.join(timeout=15)
        self.assertFalse(blocker.is_alive(), "blocking task wedged")

    def test_manual_reclaim_refused_while_target_repo_is_claimed(self) -> None:
        """Target changed to a repo another mutator claimed: refuse."""
        repo = Path(self.repo)
        test_claims: list = []
        with self.server._state_lock:
            self.assertTrue(
                self.server._claim_main_tree(
                    repo, "discard", holder=test_claims,
                )
            )
        try:
            stale_wt = self._stale_wt_path()
            Path(self.repo, "manual.txt").write_text("manual change\n")
            before = _commit_count(self.repo)
            self.server._autocommit_changes(
                "tab-vanish", work_dir=stale_wt, manual=True,
                claimed_repo=Path(stale_wt),
            )
            self.assertEqual(_commit_count(self.repo), before)
            done = self._events_of("autocommit_done")
            self.assertTrue(done, self.events)
            self.assertFalse(done[-1].get("success"))
            self.assertIn(
                "Another operation is modifying", done[-1].get("message", ""),
            )
            # The discard's claim was not touched.
            self.assertEqual(self._claim_on(self.repo), "discard")
        finally:
            with self.server._state_lock:
                for _claim in test_claims:
                    self.server._release_main_tree_claim(_claim)

    def test_dispatch_for_a_nonexistent_dir_publishes_no_claim(self) -> None:
        """A vanished non-worktree workDir short-circuits claim-free."""
        gone = str(Path(self.tmpdir) / "never-existed")
        self.server._cmd_autocommit_action({
            "tabId": "tab-gone", "workDir": gone,
        })
        self._wait_for_idle_commit_worker()
        done = self._events_of("autocommit_done")
        self.assertTrue(done, self.events)
        self.assertIn("Not a git repository", done[-1].get("message", ""))
        with self.server._state_lock:
            self.assertFalse(self.server._main_tree_claims)


class TestAutoCommitClaims(_Base):
    """Missed wiring 2: automatic post-task commits are claimed."""

    def test_task_admission_refused_during_post_task_commit(self) -> None:
        """The reviewer's repro, direction 1: admission vs auto-commit.

        While a finishing task's automatic whole-repo commit is
        between its ``git add -A`` and its ``git commit``, a second
        direct task must be refused — pre-fix it was admitted and its
        output could be swept into the auto-commit.
        """
        finisher = threading.Thread(
            target=self._run_direct_task,
            args=("tab-a", "outA"),
            kwargs={"auto_commit": True},
            daemon=True,
        )
        finisher.start()
        self.assertTrue(
            self.gen_entered.wait(timeout=15),
            "post-task auto-commit never reached message generation",
        )
        self.assertEqual(self._claim_on(self.repo), "post-task commit")

        before = _commit_count(self.repo)
        self._run_direct_task("tab-b", "outB")
        self._assert_refused("post-task commit is in progress")
        self.assertFalse(Path(self.repo, "outB.txt").exists())

        self.gen_release.set()
        finisher.join(timeout=15)
        self.assertFalse(finisher.is_alive(), "finishing task wedged")
        self.assertIsNone(self._claim_on(self.repo))
        self.assertEqual(_commit_count(self.repo), before + 1)
        self.assertEqual(_head_files(self.repo), {"outA.txt"})

        # With the claim released, the refused task's resubmission
        # runs (and, with auto-commit off, leaves its file dirty).
        self._run_direct_task("tab-b2", "outB")
        self.assertTrue(Path(self.repo, "outB.txt").exists())

    def test_auto_commit_skips_while_another_task_runs(self) -> None:
        """The reviewer's repro, direction 2: no sweeping ``add -A``.

        A finishing task must NOT whole-repo commit while another
        already-admitted task is still running in the same repository:
        pre-fix its ``git add -A`` staged the other task's half-written
        files into the auto-commit.  Post-fix the commit is skipped
        (failure event; both tasks' files stay uncommitted) — the
        finishing task's OWN still-active admission is excluded from
        the busy check, so only the OTHER task blocks it.
        """
        blocker = threading.Thread(
            target=self._run_direct_task,
            args=("tab-block", "block-half"),
            daemon=True,
        )
        blocker.start()
        self.assertTrue(
            self.blocker_entered.wait(timeout=15),
            "blocking task never started running",
        )

        before = _commit_count(self.repo)
        self._run_direct_task("tab-a", "outA", auto_commit=True)
        self.assertEqual(
            _commit_count(self.repo), before,
            "auto-commit swept a running task's files into a commit",
        )
        self.assertFalse(
            self.gen_entered.is_set(),
            "auto-commit reached message generation despite a busy repo",
        )
        done = self._events_of("autocommit_done")
        self.assertTrue(done, self.events)
        self.assertFalse(done[-1].get("success"))
        self.assertIn("still running", done[-1].get("message", ""))
        self.assertIsNone(self._claim_on(self.repo))

        self.blocker_release.set()
        blocker.join(timeout=15)
        self.assertFalse(blocker.is_alive(), "blocking task wedged")

    def test_auto_commit_skips_while_another_claim_is_held(self) -> None:
        """A held mutator claim (e.g. a discard) skips the auto-commit."""
        repo = Path(self.repo)
        test_claims: list = []
        with self.server._state_lock:
            self.assertTrue(
                self.server._claim_main_tree(
                    repo, "discard", holder=test_claims,
                )
            )
        try:
            Path(self.repo, "dirty.txt").write_text("dirty\n")
            before = _commit_count(self.repo)
            self.server._autocommit_changes("tab-x", work_dir=self.repo)
            self.assertEqual(_commit_count(self.repo), before)
            done = self._events_of("autocommit_done")
            self.assertTrue(done, self.events)
            self.assertFalse(done[-1].get("success"))
            self.assertIn(
                "Another operation is modifying", done[-1].get("message", ""),
            )
            # The discard's claim was not touched.
            self.assertEqual(self._claim_on(self.repo), "discard")
        finally:
            with self.server._state_lock:
                for _claim in test_claims:
                    self.server._release_main_tree_claim(_claim)


class TestSiblingRepoPassClaims(_Base):
    """Missed wiring 2b: the sibling-repository pass claims too."""

    def setUp(self) -> None:
        super().setUp()
        self.sibling = str(Path(self.tmpdir) / "sibling")
        Path(self.sibling).mkdir(parents=True, exist_ok=True)
        _init_repo(self.sibling)

    def test_sibling_commit_claims_its_repo(self) -> None:
        """A direct task in the sibling repo is refused mid-commit."""
        lib = Path(self.sibling, "lib.py")
        lib.write_text("print('changed')\n")
        worker = threading.Thread(
            target=self.server._autocommit_changed_repos,
            args=("tab-sib",),
            kwargs={
                "work_dir": self.repo,
                "task_id": "review2-sib-task",
                "extra_paths": {str(lib)},
            },
            daemon=True,
        )
        worker.start()
        self.assertTrue(
            self.gen_entered.wait(timeout=15),
            "sibling-repo commit never reached message generation",
        )
        self.assertEqual(self._claim_on(self.sibling), "post-task commit")

        before = _commit_count(self.sibling)
        self._run_direct_task("tab-task", "outS", work_dir=self.sibling)
        self._assert_refused("post-task commit is in progress")
        self.assertFalse(Path(self.sibling, "outS.txt").exists())

        self.gen_release.set()
        worker.join(timeout=15)
        self.assertFalse(worker.is_alive(), "sibling commit wedged")
        self.assertIsNone(self._claim_on(self.sibling))
        self.assertEqual(_commit_count(self.sibling), before + 1)
        self.assertEqual(_head_files(self.sibling), {"lib.py"})

    def test_sibling_commit_skips_a_busy_repo(self) -> None:
        """A task running in the sibling repo blocks its pathspec commit."""
        blocker = threading.Thread(
            target=self._run_direct_task,
            args=("tab-block", "block-half"),
            kwargs={"work_dir": self.sibling},
            daemon=True,
        )
        blocker.start()
        self.assertTrue(self.blocker_entered.wait(timeout=15))

        lib = Path(self.sibling, "lib.py")
        lib.write_text("print('changed')\n")
        before = _commit_count(self.sibling)
        self.server._autocommit_changed_repos(
            "tab-sib",
            work_dir=self.repo,
            task_id="review2-sib-task-2",
            extra_paths={str(lib)},
        )
        self.assertEqual(
            _commit_count(self.sibling), before,
            "sibling pass committed files in a repo with a running task",
        )
        self.assertFalse(self.gen_entered.is_set())
        done = self._events_of("autocommit_done")
        self.assertTrue(done, self.events)
        self.assertFalse(done[-1].get("success"))
        self.assertIn("uncommitted", done[-1].get("message", ""))
        self.assertIsNone(self._claim_on(self.sibling))

        self.blocker_release.set()
        blocker.join(timeout=15)
        self.assertFalse(blocker.is_alive(), "blocking task wedged")

    def test_sibling_commit_skips_a_claimed_repo(self) -> None:
        """A mutator's claim on the sibling repo blocks its commit."""
        sibling = Path(self.sibling)
        test_claims: list = []
        with self.server._state_lock:
            self.assertTrue(
                self.server._claim_main_tree(
                    sibling, "discard", holder=test_claims,
                ),
            )
        try:
            lib = Path(self.sibling, "lib.py")
            lib.write_text("print('changed')\n")
            before = _commit_count(self.sibling)
            self.server._autocommit_changed_repos(
                "tab-sib",
                work_dir=self.repo,
                task_id="review2-sib-task-3",
                extra_paths={str(lib)},
            )
            self.assertEqual(_commit_count(self.sibling), before)
            done = self._events_of("autocommit_done")
            self.assertTrue(done, self.events)
            self.assertFalse(done[-1].get("success"))
            self.assertIn("uncommitted", done[-1].get("message", ""))
            # The discard's claim was not touched.
            self.assertEqual(self._claim_on(self.sibling), "discard")
        finally:
            with self.server._state_lock:
                for _claim in test_claims:
                    self.server._release_main_tree_claim(_claim)


if __name__ == "__main__":
    unittest.main()
