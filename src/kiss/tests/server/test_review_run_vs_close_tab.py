# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Concurrent ``run`` and ``closeTab`` must converge (review finding 4).

``_cmd_run`` registers its worker state under ``STATE_LOCK``, releases
the lock, and only later publishes the tab with
``_registry_update_tab(..., create=True)``.  A ``closeTab`` dispatched
on another connection could land in that window: it removed the
canonical tab, found the unstarted worker busy (a created-but-not-yet-
started thread counts as alive), marked the state
``frontend_closed = True``, and deferred disposal — then the run's
publication resurrected the tab, which nothing ever removed again,
while ``_dispose_if_closed`` retired the run's backend state at task
end.  Final state: the registry (and every client UI) showed an open
tab whose backend was gone — matching NEITHER serial order.

The fix has two halves that together make every interleaving converge
on the ``run → closeTab`` serial outcome (tab closed; the already-
admitted task still runs to completion, exactly like a close during a
started run):

* ``_close_tab`` sets ``frontend_closed`` BEFORE its registry removal
  (mark-then-remove);
* ``_cmd_run`` re-checks the flag right AFTER its registry publication
  and undoes the recreate when it is up (recreate-then-recheck).

The pre-fix interleaving was confirmed by temporarily inserting a
<0.1s sleep between ``_cmd_run``'s state registration and its registry
publication and closing the tab inside that window (sleep removed
again).  The permanent tests below reproduce the race two ways — a
deterministic schedule that parks the run inside its publication
broadcast, and a brute-force concurrency sweep of the whole window —
and assert the open/closed convergence invariant.  Real
:class:`VSCodeServer`, real threads, real tab registry; only the
agent's LLM run is replaced by a deterministic function (no mocks).
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
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-run-vs-close-")
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
        """Poll until the tab's run (if any) fully finished."""
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

    def _assert_consistent(self, tab_id: str) -> None:
        """The convergence invariant: registry and backend state agree.

        After the run finished and the close returned, either serial
        order is legal — ``closeTab → run`` leaves the tab open with a
        live idle state, ``run → closeTab`` leaves the tab closed with
        the state disposed.  The pre-fix bug produced the mixed state:
        tab present in the registry, backend state gone.
        """
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
        has_tab = self.server.tab_registry.has_tab(tab_id)
        state_alive = state is not None and not state.frontend_closed
        self.assertEqual(
            has_tab, state_alive,
            f"mixed open/closed outcome on {tab_id}: registry has_tab="
            f"{has_tab}, state={'alive' if state is not None else 'gone'}"
            f"{'' if state is None else f' frontend_closed={state.frontend_closed}'}",
        )


class TestCloseDuringRunPublication(_Base):
    """Deterministic schedule: close lands inside the publication."""

    def test_close_during_publication_leaves_tab_closed(self) -> None:
        """Park the run inside its ``tabs_state`` publication broadcast,
        close the tab from the test thread, release, and require the
        run→closeTab serial outcome: tab closed, task ran, state
        disposed at its end."""
        tab_id = "tab-close-mid-publish"
        events: list[dict[str, Any]] = []
        events_lock = threading.Lock()
        publish_entered = threading.Event()
        release = threading.Event()
        latched = threading.Event()

        def blocking_broadcast(event: dict[str, Any]) -> None:
            do_block = False
            with events_lock:
                events.append(event)
                if event.get("type") == "tabs_state" and not latched.is_set():
                    latched.set()
                    do_block = True
            if do_block:
                publish_entered.set()
                assert release.wait(timeout=30)

        self.server.printer.broadcast = blocking_broadcast  # type: ignore[assignment]

        runner = threading.Thread(
            target=self.server._cmd_run, args=(self._run_cmd(tab_id),),
            daemon=True,
        )
        runner.start()
        self.assertTrue(
            publish_entered.wait(timeout=10),
            "run never reached its registry publication",
        )
        # The run has recreated/created the tab and registered its
        # unstarted worker; the close lands NOW.
        closer = threading.Thread(
            target=self.server._close_tab, args=(tab_id,), daemon=True,
        )
        closer.start()
        # The close defers disposal (pre-start worker is busy) and
        # returns; it must never block on the parked broadcast.
        closer.join(timeout=10)
        self.assertFalse(closer.is_alive(), "closeTab wedged")
        release.set()
        runner.join(timeout=15)
        self.assertFalse(runner.is_alive(), "run dispatch wedged")

        self._wait_for_task_end(tab_id)
        self.assertFalse(self.server.tab_registry.has_tab(tab_id))
        self._assert_consistent(tab_id)

    def test_close_of_unknown_tab_is_a_quiet_noop(self) -> None:
        """Closing a tab with no registry entry and no state is safe."""
        self.server._close_tab("tab-never-existed")
        self.assertFalse(self.server.tab_registry.has_tab("tab-never-existed"))


class TestRunVsCloseSweep(_Base):
    """Brute-force sweep of the registration→publication window."""

    def test_concurrent_run_and_close_always_converge(self) -> None:
        """Race ``run`` against ``closeTab`` across the whole submit
        window; every iteration must end registry-consistent.  Pre-fix
        the close sometimes hit the pre-publication window and left an
        open registry tab with a disposed backend state."""
        for i in range(60):
            tab_id = f"tab-sweep-{i}"
            self.server.tab_registry.open_tab(tab_id, "sweep", self.work_dir)
            start = threading.Barrier(2)

            def do_run(tab: str = tab_id) -> None:
                start.wait(timeout=10)
                self.server._cmd_run(self._run_cmd(tab))

            def do_close(tab: str = tab_id, jitter: int = i) -> None:
                start.wait(timeout=10)
                # Sweep the window: the racy region is microseconds
                # wide, so stagger the close a little further into the
                # submit path on each iteration.
                time.sleep((jitter % 20) * 0.0002)
                self.server._close_tab(tab)

            t_run = threading.Thread(target=do_run, daemon=True)
            t_close = threading.Thread(target=do_close, daemon=True)
            t_run.start()
            t_close.start()
            t_run.join(timeout=20)
            t_close.join(timeout=20)
            self.assertFalse(t_run.is_alive() or t_close.is_alive())
            self._wait_for_task_end(tab_id)
            self._assert_consistent(tab_id)
