# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""M-C1: ``_present_pending_worktree``'s empty-branch discard must
re-check ownership under the lock before claiming ``is_merging``.

The empty-branch discard in ``_present_pending_worktree`` was the only
``is_merging`` claim site with no ``state.busy()`` / pending re-check:
it saved, overwrote and blindly restored the flag guarded solely by
``_any_non_wt_running(wt_dir)`` — which sees only NON-worktree tasks.
Two consequences in the unlocked window between
``_finalize_pending_worktree`` returning ``PRESENT`` and the discard's
locked section:

1. **Own-task variant** — a run submitted on the tab in that window
   sets up a fresh (still empty) worktree; the presenter's probe finds
   it empty and ``wt_agent.discard`` deletes the directory and branch
   out from under the running task.

2. **Foreign-holder variant** — another thread legitimately holding
   the tab's ``is_merging`` (a resume's ``PRESENT_CLAIMED``
   presentation) has its live flag captured as ``prev_merging=True``
   and blindly restored after the real owner cleared it, wedging the
   tab (every later run / merge / discard refused) until restart.

The fix re-checks, in the same critical section that takes the claim:
the state and agent are still the ones read outside the lock, the
worktree is still pending, and the tab is idle or busy solely with the
CURRENT thread's own claim (the ``PRESENT_CLAIMED`` caller).

Everything here is real: a real git repository, a real linked worktree
created by ``WorktreeSorcarAgent``, real ``run`` / ``worktreeAction``
commands dispatched through ``VSCodeServer._handle_command``, real
threads.  Only the LLM call is replaced by a deterministic stub
function (no mocks), and the racing threads are parked at natural
seams via wrappers that delegate to the real methods — the same
technique as ``test_review2_stale_run_undo._Base``.
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
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter


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


def _kiss_wt_branches(repo: str) -> list[str]:
    out = _run_git(repo, "branch", "--list", "kiss/wt-*").stdout
    return [
        line.strip().lstrip("*+ ").strip()
        for line in out.splitlines()
        if line.strip()
    ]


class TestPresentDiscardOwnership(unittest.TestCase):
    """Real server, real git repo, isolated persistence, stubbed LLM."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-conc2026-mc1-")
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

        self.printer = MemoryPrinter()
        self.server = VSCodeServer(self.printer)
        self.server.work_dir = self.repo

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        for state in agent_state.snapshot():
            if state.agent is not None and state.agent._wt_pending:
                try:
                    state.agent.discard()
                except Exception:  # pragma: no cover — cleanup best-effort
                    pass
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

    def _stub_llm(self, filename: str | None) -> None:
        """Deterministic agent: write *filename* (or nothing), succeed."""

        def stub_run(_agent: object, **kwargs: object) -> str:
            work_dir = kwargs.get("work_dir")
            if filename and isinstance(work_dir, str) and work_dir:
                Path(work_dir, filename).write_text("agent output\n")
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def _status_ends(self, tab_id: str) -> list[dict[str, Any]]:
        return [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("type") == "status"
            and ev.get("running") is False
            and ev.get("tabId") == tab_id
        ]

    def _dispatch_run(self, tab_id: str) -> None:
        self.server._handle_command({
            "type": "run",
            "prompt": "task",
            "workDir": self.repo,
            "tabId": tab_id,
            "useWorktree": True,
            "autoCommit": False,
            "model": "",
        })

    def _wait_task_end(self, tab_id: str, before: int) -> None:
        deadline = time.time() + 60
        while time.time() < deadline:
            if len(self._status_ends(tab_id)) > before:
                return
            time.sleep(0.01)
        raise AssertionError(f"task on tab {tab_id!r} never finished")

    def _make_pending_empty_worktree(self, tab_id: str) -> tuple[Path, str]:
        """Run a real worktree task that changes nothing.

        With ``autoCommit=False`` the post-task path presents the
        (empty) worktree with ``discard_if_empty=False``, so the tab
        is left holding a pending, empty ``kiss/wt-*`` worktree —
        exactly the state a later resume's presentation probes.
        """
        self._stub_llm(None)
        before = len(self._status_ends(tab_id))
        self._dispatch_run(tab_id)
        self._wait_task_end(tab_id, before)
        state = agent_state.find_by_tab(tab_id)
        assert state is not None and state.agent is not None
        assert state.agent._wt_pending, "run must leave a pending worktree"
        wt_dir = state.agent._wt_dir
        branch = state.agent._wt_branch
        assert wt_dir is not None and branch is not None
        self.assertTrue(wt_dir.exists())
        self.assertIn(branch, _kiss_wt_branches(self.repo))
        return wt_dir, str(branch)

    # -- tests -----------------------------------------------------

    def test_present_refuses_to_discard_a_running_tasks_fresh_worktree(
        self,
    ) -> None:
        """Own-task variant: a resume that observed "idle, nothing
        pending" (``PRESENT``) and was descheduled must NOT discard the
        fresh, still-empty worktree of a task that started on the tab
        in the meantime."""
        tab_id = "tab-mc1-own"

        # Park thread A between _finalize_pending_worktree (PRESENT)
        # and _present_pending_worktree — the production race window.
        orig_finalize = self.server._finalize_pending_worktree
        a_parked = threading.Event()
        a_release = threading.Event()
        parked_once = threading.Event()

        def parking_finalize(t: str) -> Any:
            outcome = orig_finalize(t)
            if not parked_once.is_set():
                parked_once.set()
                a_parked.set()
                a_release.wait(timeout=60)
            return outcome

        self.server._finalize_pending_worktree = (  # type: ignore[method-assign]
            parking_finalize  # type: ignore[assignment]
        )

        # The task's LLM stub parks so the worktree stays live (and
        # empty) while thread A resumes.
        stub_started = threading.Event()
        stub_release = threading.Event()

        def parked_stub(_agent: object, **kwargs: object) -> str:
            stub_started.set()
            stub_release.wait(timeout=60)
            return "success: true\nsummary: stub\n"

        self._parent_class.run = parked_stub

        a_error: list[BaseException] = []

        def emit() -> None:
            try:
                self.server._emit_pending_worktree(tab_id)
            except BaseException as exc:  # pragma: no cover — must not raise
                a_error.append(exc)

        thread_a = threading.Thread(target=emit, daemon=True)
        try:
            thread_a.start()
            self.assertTrue(a_parked.wait(timeout=30), "A never finalized")

            # A new run starts on the tab inside A's window and sets
            # up a fresh worktree.
            self._dispatch_run(tab_id)
            deadline = time.time() + 30
            wt_dir: Path | None = None
            while time.time() < deadline:
                state = agent_state.find_by_tab(tab_id)
                if (
                    state is not None
                    and state.agent is not None
                    and state.agent._wt_pending
                    and stub_started.is_set()
                ):
                    wt_dir = state.agent._wt_dir
                    break
                time.sleep(0.005)
            self.assertIsNotNone(wt_dir, "run never set up its worktree")
            assert wt_dir is not None
            self.assertTrue(wt_dir.exists())
            branches = _kiss_wt_branches(self.repo)
            self.assertEqual(len(branches), 1)

            # A resumes: its presentation probes the fresh worktree,
            # finds it empty — and must now refuse the discard because
            # the tab is busy with a thread that is not A.
            a_release.set()
            thread_a.join(timeout=30)
            self.assertFalse(thread_a.is_alive(), "A never finished")
            self.assertEqual(a_error, [])

            self.assertTrue(
                wt_dir.exists(),
                "presentation discarded the running task's worktree",
            )
            self.assertEqual(_kiss_wt_branches(self.repo), branches)
            state = agent_state.find_by_tab(tab_id)
            assert state is not None
            self.assertFalse(
                state.is_merging,
                "presentation left a stale is_merging claim",
            )
            self.assertIsNone(state.merge_thread)
        finally:
            a_release.set()
            stub_release.set()
            self.server._finalize_pending_worktree = (  # type: ignore[method-assign]
                orig_finalize
            )

        # The parked task finishes normally afterwards.
        self._wait_task_end(tab_id, before=0)

    def test_present_refuses_while_a_foreign_thread_holds_the_claim(
        self,
    ) -> None:
        """Foreign-holder variant: while thread B legitimately holds
        the tab's ``is_merging`` (``PRESENT_CLAIMED`` presentation), a
        concurrent presenter must refuse instead of capturing B's live
        flag and blindly restoring it after B cleared it."""
        tab_id = "tab-mc1-foreign"
        wt_dir, branch = self._make_pending_empty_worktree(tab_id)

        # Park thread B inside its presentation, right after the
        # changed-files probe, while it holds the PRESENT_CLAIMED
        # is_merging claim taken by _finalize_pending_worktree.
        orig_probe = self.server._get_worktree_changed_files
        b_thread_box: list[threading.Thread] = []
        b_parked = threading.Event()
        b_release = threading.Event()
        parked_once = threading.Event()

        def parking_probe(t: str = "") -> list[str]:
            result = orig_probe(t)
            if (
                b_thread_box
                and threading.current_thread() is b_thread_box[0]
                and not parked_once.is_set()
            ):
                parked_once.set()
                b_parked.set()
                b_release.wait(timeout=60)
            return result

        self.server._get_worktree_changed_files = (  # type: ignore[method-assign]
            parking_probe  # type: ignore[assignment]
        )

        b_error: list[BaseException] = []

        def emit() -> None:
            try:
                self.server._emit_pending_worktree(tab_id)
            except BaseException as exc:  # pragma: no cover — must not raise
                b_error.append(exc)

        thread_b = threading.Thread(target=emit, daemon=True)
        b_thread_box.append(thread_b)
        try:
            thread_b.start()
            self.assertTrue(b_parked.wait(timeout=30), "B never probed")

            state = agent_state.find_by_tab(tab_id)
            assert state is not None
            self.assertTrue(state.is_merging)
            self.assertIs(state.merge_thread, thread_b)

            # A concurrent presenter (a second resume whose finalize
            # saw PRESENT an instant before B claimed) must refuse.
            self.server._present_pending_worktree(tab_id)

            self.assertTrue(
                wt_dir.exists(),
                "concurrent presenter discarded the claimed worktree",
            )
            self.assertIn(branch, _kiss_wt_branches(self.repo))
            self.assertTrue(
                state.is_merging,
                "concurrent presenter clobbered the owner's claim",
            )
            self.assertIs(
                state.merge_thread,
                thread_b,
                "concurrent presenter overwrote the owner's merge_thread",
            )

            # B resumes and, as the claim owner, discards the empty
            # branch itself; the claim is released afterwards — the
            # tab must NOT be wedged by a stale restore.
            b_release.set()
            thread_b.join(timeout=30)
            self.assertFalse(thread_b.is_alive(), "B never finished")
            self.assertEqual(b_error, [])
        finally:
            b_release.set()
            self.server._get_worktree_changed_files = (  # type: ignore[method-assign]
                orig_probe
            )

        self.assertFalse(wt_dir.exists(), "owner's own discard never ran")
        self.assertNotIn(branch, _kiss_wt_branches(self.repo))
        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        self.assertFalse(
            state.is_merging,
            "stale is_merging restore wedged the tab",
        )
        self.assertIsNone(state.merge_thread)
        self.assertFalse(state.busy())

    def test_present_refuses_when_the_pending_worktree_is_already_gone(
        self,
    ) -> None:
        """Pending-cleared variant: a user Discard landing between the
        presenter's probe and its locked claim clears ``_wt_pending``;
        the presenter must stand down instead of re-discarding."""
        tab_id = "tab-mc1-gone"
        wt_dir, branch = self._make_pending_empty_worktree(tab_id)

        orig_probe = self.server._get_worktree_changed_files
        a_thread_box: list[threading.Thread] = []
        a_parked = threading.Event()
        a_release = threading.Event()
        parked_once = threading.Event()

        def parking_probe(t: str = "") -> list[str]:
            result = orig_probe(t)
            if (
                a_thread_box
                and threading.current_thread() is a_thread_box[0]
                and not parked_once.is_set()
            ):
                parked_once.set()
                a_parked.set()
                a_release.wait(timeout=60)
            return result

        self.server._get_worktree_changed_files = (  # type: ignore[method-assign]
            parking_probe  # type: ignore[assignment]
        )

        a_error: list[BaseException] = []

        def present() -> None:
            try:
                self.server._present_pending_worktree(tab_id)
            except BaseException as exc:  # pragma: no cover — must not raise
                a_error.append(exc)

        thread_a = threading.Thread(target=present, daemon=True)
        a_thread_box.append(thread_a)
        try:
            thread_a.start()
            self.assertTrue(a_parked.wait(timeout=30), "A never probed")

            # The user clicks Discard while A is parked: the real
            # worktreeAction command discards the pending branch.
            self.server._handle_command({
                "type": "worktreeAction",
                "action": "discard",
                "tabId": tab_id,
            })
            self.assertFalse(wt_dir.exists())
            state = agent_state.find_by_tab(tab_id)
            assert state is not None
            agent = state.agent
            assert agent is not None
            self.assertFalse(agent._wt_pending)

            a_release.set()
            thread_a.join(timeout=30)
            self.assertFalse(thread_a.is_alive(), "A never finished")
            self.assertEqual(a_error, [])

            self.assertFalse(
                state.is_merging,
                "stood-down presenter left a stale is_merging claim",
            )
            self.assertIsNone(state.merge_thread)
            self.assertFalse(state.busy())
        finally:
            a_release.set()
            self.server._get_worktree_changed_files = (  # type: ignore[method-assign]
                orig_probe
            )

    def test_stale_empty_probe_never_discards_late_task_output(self) -> None:
        """Review finding 8: the discard decision must not rest on a
        probe taken BEFORE the ownership claim.

        Schedule: the presenter probes the running task's worktree and
        sees ``[]``; the task then writes ordinary output and fully
        finishes (clearing ``busy``, keeping the pending worktree);
        the presenter resumes — state, agent and pending all still
        match and the busy check passes.  Pre-fix it accepted the
        stale empty answer and ``discard(rescue_ignored=True)`` deleted
        the completed output and its branch.  Post-fix the probe is
        repeated under the claim, so the presenter sees the output and
        presents the Merge/Discard buttons instead.
        """
        tab_id = "tab-mc1-stale-empty"
        stub_started = threading.Event()
        allow_write = threading.Event()

        def late_writer(_agent: object, **kwargs: object) -> str:
            work_dir = kwargs.get("work_dir")
            assert isinstance(work_dir, str) and work_dir
            stub_started.set()
            assert allow_write.wait(30)
            Path(work_dir, "late-output.txt").write_text("must survive\n")
            return "success: true\nsummary: late writer\n"

        self._parent_class.run = late_writer

        orig_probe = self.server._get_worktree_changed_files
        presenter_box: list[threading.Thread] = []
        presenter_probed = threading.Event()
        allow_presenter = threading.Event()
        probed_once = threading.Event()

        def parking_probe(t: str = "") -> list[str]:
            result = orig_probe(t)
            if (
                presenter_box
                and threading.current_thread() is presenter_box[0]
                and not probed_once.is_set()
            ):
                probed_once.set()
                assert result == [], "the first probe must be the empty one"
                presenter_probed.set()
                assert allow_presenter.wait(60)
            return result

        self.server._get_worktree_changed_files = (  # type: ignore[method-assign]
            parking_probe  # type: ignore[assignment]
        )

        errors: list[BaseException] = []

        def present() -> None:
            try:
                self.server._present_pending_worktree(tab_id)
            except BaseException as exc:  # pragma: no cover — must not raise
                errors.append(exc)

        presenter = threading.Thread(target=present, daemon=True)
        presenter_box.append(presenter)
        before = len(self._status_ends(tab_id))
        try:
            self._dispatch_run(tab_id)
            self.assertTrue(stub_started.wait(30), "task never started")
            deadline = time.time() + 30
            state = None
            while time.time() < deadline:
                state = agent_state.find_by_tab(tab_id)
                if (
                    state is not None
                    and state.agent is not None
                    and state.agent._wt_pending
                ):
                    break
                time.sleep(0.005)
            assert state is not None and state.agent is not None
            self.assertTrue(state.agent._wt_pending, "no pending worktree")
            wt_dir = state.agent._wt_dir
            branch = state.agent._wt_branch
            assert wt_dir is not None and branch is not None
            self.assertTrue(wt_dir.exists())

            presenter.start()
            self.assertTrue(
                presenter_probed.wait(30), "presenter never probed",
            )

            # The task writes AFTER the empty probe, then fully ends.
            allow_write.set()
            self._wait_task_end(tab_id, before)
            self.assertTrue(Path(wt_dir, "late-output.txt").is_file())
            state = agent_state.find_by_tab(tab_id)
            assert state is not None
            deadline = time.time() + 30
            while state.busy() and time.time() < deadline:
                time.sleep(0.005)
            self.assertFalse(state.busy())

            emitted_before = len(self.printer.emitted)
            allow_presenter.set()
            presenter.join(30)
            self.assertFalse(presenter.is_alive(), "presenter never ended")
            self.assertEqual(errors, [])

            # The completed output survives, on disk and on the branch.
            self.assertTrue(
                wt_dir.exists(),
                "stale empty probe discarded the finished task's worktree",
            )
            self.assertTrue(Path(wt_dir, "late-output.txt").is_file())
            self.assertIn(branch, _kiss_wt_branches(self.repo))
            # The presenter re-probed under its claim and presented
            # the Merge/Discard buttons for the late output instead.
            done_events = [
                ev for ev in self.printer.emitted[emitted_before:]
                if ev.get("type") == "worktree_done"
                and ev.get("tabId") == tab_id
            ]
            self.assertTrue(
                any(
                    "late-output.txt" in (ev.get("changedFiles") or [])
                    for ev in done_events
                ),
                f"no worktree_done presenting the late output: "
                f"{done_events!r}",
            )
            self.assertFalse(state.is_merging, "stale is_merging claim")
            self.assertIsNone(state.merge_thread)
        finally:
            allow_write.set()
            allow_presenter.set()
            self.server._get_worktree_changed_files = (  # type: ignore[method-assign]
                orig_probe
            )

    def test_idle_empty_pending_worktree_is_still_auto_discarded(self) -> None:
        """The ownership re-check must not break the legitimate path:
        an idle tab's empty pending worktree is auto-discarded by the
        PRESENT_CLAIMED presentation on resume."""
        tab_id = "tab-mc1-idle"
        wt_dir, branch = self._make_pending_empty_worktree(tab_id)

        self.server._emit_pending_worktree(tab_id)

        self.assertFalse(wt_dir.exists(), "empty branch was not discarded")
        self.assertNotIn(branch, _kiss_wt_branches(self.repo))
        state = agent_state.find_by_tab(tab_id)
        assert state is not None
        self.assertFalse(state.is_merging)
        self.assertIsNone(state.merge_thread)
        self.assertFalse(state.busy())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
