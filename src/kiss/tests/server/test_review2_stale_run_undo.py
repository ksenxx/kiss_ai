# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A stale run-side undo must not delete a later legitimate reopen.

Review 2 (gpt-5.6-sol), introduced bug 1: ``_cmd_run``'s compensating
close (added for review finding 4) read ``state.frontend_closed``
under ``STATE_LOCK``, released the lock, and then called the
UNCONDITIONAL ``tab_registry.close_tab(tab_id)``.  The read and the
removal were not one operation, so this ordered schedule broke:

1. a ``closeTab`` marks the pre-start run state closed and removes
   the registry row;
2. the run publishes its row (``create=True``), reads
   ``frontend_closed=True``, and stalls before its compensating close;
3. a later ``resumeSession`` legitimately reopens the tab — clears
   ``frontend_closed`` and recreates the registry row;
4. the stale run resumes and its unconditional close deletes the
   RESUME's row.

Final state: no registry tab, but a live backend state with
``frontend_closed=False`` — the reverse mixed state of the original
finding-4 bug, and a real-time-ordering violation (the explicit close
completed before the resume began, yet stale work from the older run
removed the resume's tab).

The fix makes the undo verify it is undoing ITS OWN recreation: every
``TabRegistry.update_tab`` stamps the tab with a fresh in-memory
generation token, ``_cmd_run`` captures the token its own publication
produced, and the undo goes through the new atomic
``TabRegistry.close_tab_if_generation`` — a registry row republished
by anyone else (the resume) carries a NEWER token, so the stale undo
no-ops.

Real :class:`VSCodeServer`, real tab registry, real threads, real
``_cmd_run`` / ``_close_tab`` / ``_replay_session``; only the agent's
LLM run is replaced by a deterministic function (no mocks).  The
deterministic schedules park threads at two natural seams — inside the
publication's ``tabs_state`` broadcast and at the entry of the
registry-close call — via wrappers that delegate to the real methods.
"""

from __future__ import annotations

import shutil
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


class _Base(unittest.TestCase):
    """Real server + private tab registry + stubbed LLM run."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-stale-undo-")
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
        self.server.use_private_tab_registry(Path(self.tmpdir) / "tabs.json")
        self.work_dir = str(Path(self.tmpdir) / "work")
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self.server.work_dir = self.work_dir

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

        def stub_run(_agent: object, **kwargs: object) -> str:
            return "success: true\nsummary: stub\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
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

    def _run_cmd(self, tab_id: str) -> dict[str, Any]:
        return {
            "type": "run",
            "prompt": "do something",
            "workDir": self.work_dir,
            "tabId": tab_id,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        }

    def _wait_for_task_end(self, tab_id: str, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with agent_state.STATE_LOCK:
                state = agent_state.find_by_tab(tab_id)
                done = state is None or (
                    state.task_thread is None and not state.is_task_active
                )
            if done:
                return
            time.sleep(0.005)
        raise AssertionError(f"task on {tab_id} never finished")

    def _park_run_inside_publication(
        self, runner_holder: list[threading.Thread],
    ) -> tuple[threading.Event, threading.Event]:
        """Block the run thread inside its first ``tabs_state`` broadcast.

        Only the RUN thread's first ``tabs_state`` blocks; every other
        broadcast (from the test thread's close/resume, and the run's
        own later events) passes through untouched.
        """
        entered = threading.Event()
        release = threading.Event()
        latched = threading.Event()
        orig_broadcast = self.server.printer.broadcast

        def gating_broadcast(event: dict[str, Any]) -> None:
            if (
                event.get("type") == "tabs_state"
                and runner_holder
                and threading.current_thread() is runner_holder[0]
                and not latched.is_set()
            ):
                latched.set()
                entered.set()
                assert release.wait(timeout=30), "publication never released"

        self.server.printer.broadcast = gating_broadcast  # type: ignore[assignment]
        self._orig_broadcast = orig_broadcast
        return entered, release

    def _gate_registry_close(
        self, gated_threads: list[threading.Thread],
    ) -> tuple[threading.Event, threading.Event]:
        """Park listed threads at the entry of the registry-close call.

        Wraps BOTH ``close_tab`` and (when present, post-fix)
        ``close_tab_if_generation`` so the same schedule discriminates
        the pre-fix unconditional close and exercises the fixed
        token-conditional close.  The wrappers delegate to the real
        registry methods.
        """
        entered = threading.Event()
        release = threading.Event()
        reg = self.server.tab_registry

        def park_if_gated() -> None:
            if any(
                threading.current_thread() is t for t in gated_threads
            ):
                entered.set()
                assert release.wait(timeout=30), "registry close never released"

        orig_close = reg.close_tab

        def gated_close(tab_id: str) -> int:
            park_if_gated()
            return orig_close(tab_id)

        reg.close_tab = gated_close  # type: ignore[method-assign]

        orig_cond = getattr(reg, "close_tab_if_generation", None)
        if orig_cond is not None:
            def gated_cond(tab_id: str, generation: int) -> bool:
                park_if_gated()
                return bool(orig_cond(tab_id, generation))

            reg.close_tab_if_generation = gated_cond  # type: ignore[method-assign]
        return entered, release


class TestStaleRunUndoVsResume(_Base):
    """The reviewer's four-step schedule, driven deterministically."""

    def test_stale_undo_spares_a_later_resume_recreation(self) -> None:
        """closeTab → run publication → resume reopen → stale undo.

        The stale undo must NOT delete the resume's recreated tab: the
        final state must be the ``closeTab → run → resume`` serial
        outcome (tab open, state live, ``frontend_closed=False``).
        Pre-fix the unconditional ``close_tab`` deleted the resume's
        row, leaving no registry tab but a live open backend state.
        """
        tab_id = "tab-stale-undo"
        runner_holder: list[threading.Thread] = []
        pub_entered, pub_release = self._park_run_inside_publication(
            runner_holder,
        )
        undo_entered, undo_release = self._gate_registry_close(
            cast("list[threading.Thread]", runner_holder),
        )

        runner = threading.Thread(
            target=self.server._cmd_run, args=(self._run_cmd(tab_id),),
            daemon=True,
        )
        runner_holder.append(runner)
        runner.start()
        self.assertTrue(
            pub_entered.wait(timeout=10),
            "run never reached its registry publication",
        )
        # Step 1: the close lands now — it marks the pre-start run
        # state closed and removes the registry row the run created.
        self.server._close_tab(tab_id)
        self.assertFalse(self.server.tab_registry.has_tab(tab_id))
        # Step 2: the run proceeds, reads ``frontend_closed=True`` and
        # parks at the entry of its compensating registry close.
        pub_release.set()
        self.assertTrue(
            undo_entered.wait(timeout=10),
            "run never reached its compensating close",
        )
        # Step 3: a later resume legitimately reopens the tab.  The
        # chat id has no history and no running match, so the resume
        # takes the no-result path: clear ``frontend_closed``, then
        # recreate the registry row (``create=True``).
        self.server._replay_session("review2-reopen-chat", tab_id)
        self.assertTrue(self.server.tab_registry.has_tab(tab_id))
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
            self.assertIsNotNone(state)
            self.assertFalse(state.frontend_closed)  # type: ignore[union-attr]
        # Step 4: the stale undo fires.  Post-fix it no-ops (the row
        # carries the resume's newer generation token).
        undo_release.set()
        runner.join(timeout=15)
        self.assertFalse(runner.is_alive(), "run dispatch wedged")
        self._wait_for_task_end(tab_id)

        self.assertTrue(
            self.server.tab_registry.has_tab(tab_id),
            "stale run-side close deleted the resume's recreated tab",
        )
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
            self.assertIsNotNone(state)
            self.assertFalse(state.frontend_closed)  # type: ignore[union-attr]

    def test_undo_still_closes_when_no_reopen_intervened(self) -> None:
        """Same schedule WITHOUT the resume: the undo must still fire.

        This pins the original finding-4 convergence: when nobody
        reopened the tab between the run's publication and its undo,
        the token still matches and the recreate is undone — the
        ``run → closeTab`` serial outcome (tab closed; the task still
        runs to completion; state disposed at task end).

        The close is parked between its ``frontend_closed`` mark and
        its own registry removal, so it is the RUN's conditional close
        (token match) that removes the row — covering the match branch
        deterministically.
        """
        tab_id = "tab-undo-no-reopen"
        runner_holder: list[threading.Thread] = []
        pub_entered, pub_release = self._park_run_inside_publication(
            runner_holder,
        )
        closer_holder: list[threading.Thread] = []
        close_entered, close_release = self._gate_registry_close(
            cast("list[threading.Thread]", closer_holder),
        )

        runner = threading.Thread(
            target=self.server._cmd_run, args=(self._run_cmd(tab_id),),
            daemon=True,
        )
        runner_holder.append(runner)
        runner.start()
        self.assertTrue(pub_entered.wait(timeout=10))
        closer = threading.Thread(
            target=self.server._close_tab, args=(tab_id,), daemon=True,
        )
        closer_holder.append(closer)
        closer.start()
        # The close marked the state and parked before its removal.
        self.assertTrue(close_entered.wait(timeout=10))
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
            self.assertIsNotNone(state)
            self.assertTrue(state.frontend_closed)  # type: ignore[union-attr]
        # The run resumes: it re-checks the flag (True) and its
        # conditional close finds ITS OWN token — the row is removed.
        pub_release.set()
        runner.join(timeout=15)
        self.assertFalse(runner.is_alive(), "run dispatch wedged")
        self.assertFalse(self.server.tab_registry.has_tab(tab_id))
        # The parked close's own removal is now a no-op.
        close_release.set()
        closer.join(timeout=10)
        self.assertFalse(closer.is_alive(), "closeTab wedged")
        self.assertFalse(self.server.tab_registry.has_tab(tab_id))
        self._wait_for_task_end(tab_id)
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
        self.assertTrue(state is None or state.frontend_closed)


class TestRunCloseReopenSweep(_Base):
    """Three-actor jittered sweep: run vs close vs reopen."""

    def test_run_close_reopen_never_leaves_reverse_mixed_state(self) -> None:
        """Whenever a backend state survives, the registry must agree
        with its ``frontend_closed`` flag.  The reviewer's bug produced
        a live OPEN state with no registry row (reverse mixed state);
        the original finding-4 bug produced a registry row for a
        retired state — both are non-serializable outcomes.

        A reopen that lands after the run's state was already disposed
        legitimately recreates a bare history tab (no backend state),
        so the invariant is only asserted while a state exists.
        """
        for i in range(40):
            tab_id = f"tab-3sweep-{i}"
            self.server.tab_registry.open_tab(tab_id, "sweep", self.work_dir)
            start = threading.Barrier(3)

            def do_run(tab: str = tab_id) -> None:
                start.wait(timeout=10)
                self.server._cmd_run(self._run_cmd(tab))

            def do_close(tab: str = tab_id, jitter: int = i) -> None:
                start.wait(timeout=10)
                time.sleep((jitter % 20) * 0.0002)
                self.server._close_tab(tab)

            def do_reopen(tab: str = tab_id, jitter: int = i) -> None:
                start.wait(timeout=10)
                time.sleep(((jitter + 7) % 20) * 0.0002)
                self.server._replay_session(f"sweep-chat-{tab}", tab)

            threads = [
                threading.Thread(target=fn, daemon=True)
                for fn in (do_run, do_close, do_reopen)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=20)
            self.assertFalse(any(t.is_alive() for t in threads))
            self._wait_for_task_end(tab_id)
            with agent_state.STATE_LOCK:
                state = agent_state.find_by_tab(tab_id)
                state_open = (
                    state is not None and not state.frontend_closed
                )
                state_exists = state is not None
            if state_exists:
                self.assertEqual(
                    self.server.tab_registry.has_tab(tab_id), state_open,
                    f"mixed outcome on {tab_id}: has_tab="
                    f"{self.server.tab_registry.has_tab(tab_id)}, "
                    f"state_open={state_open}",
                )


if __name__ == "__main__":
    unittest.main()
