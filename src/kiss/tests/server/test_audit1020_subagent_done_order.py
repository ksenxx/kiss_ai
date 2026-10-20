# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a sub-agent tab announced during the child's teardown still closes.

Audit finding C3-F1.  A ``run_agent`` child used to broadcast its
``subagentDone`` from inside ``_run_task_inner`` — before ``_run_task``
released the state's thread slot (``_release_run_state``).  In that
window the child's state still reported ``running()``, so a parent
replay (``_open_persisted_subagent_tabs``) announced the sub-agent tab
with ``isDone: False`` and subscribed it to the child's stream — but
the fan-out signal had already gone out, and the run ended with only
a ``status running=False`` for the tab.  The tab stayed open although
the sub-agent had finished.

The reproduction parks the child's task thread inside that window
deterministically instead of racing it, with the daemon's own locks:
the printer's ``_model_pick_lock`` gates the thread after its terminal
event (the model-pick restore and ``cleanup_task`` take it) while the
test takes the state lock, which the mandatory-cleanup finally and
``_release_run_state`` both need; once ``cleanup_task`` has run the
thread is blocked right before the release, every pre-fix broadcast
already out.  The replay runs on the test thread (the lock is
re-entrant), exactly as a reconnecting client's would, and the viewer
connection records what the daemon broadcast.  Only the LLM boundary
is stubbed.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from kiss.agents.sorcar import daemon_client
from kiss.server import agent_state
from kiss.server.server import _subagent_is_done
from kiss.tests.server.test_run_agent_subagent_tab import (
    PARENT_TAB_ID,
    DaemonLocalHarness,
    _viewer_sub_tab_id,
)


class SubagentDoneOrderTest(DaemonLocalHarness):
    """The last-row ``subagentDone`` is ordered after the thread slot is released."""

    def test_replay_during_child_teardown_still_closes_the_tab(self) -> None:
        """A tab announced as live during the child's teardown gets its ``subagentDone``."""
        child_marker = "child subagent task order-9f"
        child_release = threading.Event()
        self._install_stub({}, child_release, child_marker)
        received = self._open_viewer()

        parent = daemon_client.run(
            "parent seed task",
            work_dir=self.repo,
            endpoint_file=self.endpoint_file,
            timeout=60,
        )
        assert parent.success is True and parent.task_id

        child_out: dict[str, Any] = {}

        def dispatch_child() -> None:
            child_out["result"] = daemon_client.run(
                child_marker,
                work_dir=self.repo,
                parent_task_id=parent.task_id,
                parent_tab_id=PARENT_TAB_ID,
                endpoint_file=self.endpoint_file,
                timeout=60,
            )

        dispatcher = threading.Thread(target=dispatch_child, daemon=True)
        dispatcher.start()
        printer = self.server._printer
        agent_server = self.server._agent_server
        parked = False
        state_locked = False
        try:
            new_tab = self._wait_for(
                lambda: next(
                    (
                        e
                        for e in list(received)
                        if e.get("type") == "new_tab" and e.get("parent_tab_id") == PARENT_TAB_ID
                    ),
                    None,
                ),
                what="new_tab broadcast with the parent tab id",
            )
            child_task_id = str(new_tab["task_id"])
            viewer_sub_tab_id = _viewer_sub_tab_id(child_task_id)
            self._send_from_viewer(
                {
                    "type": "resumeSession",
                    "taskId": child_task_id,
                    "tabId": viewer_sub_tab_id,
                }
            )
            self._wait_for(
                lambda: viewer_sub_tab_id in printer._fanout_targets(child_task_id),
                what="viewer sub-agent tab subscription",
            )
            # Park the child's thread after ``cleanup_task`` and before
            # ``_release_run_state``, in two steps.  The run drops
            # ``is_task_active`` right before its terminal event; from
            # there to the mandatory-cleanup finally it needs the
            # printer's ``_model_pick_lock`` (model-pick restore,
            # ``cleanup_task``) but not the state lock, so the first
            # lock gates the thread while the test takes the second,
            # which the finally's own ``is_task_active`` clear and the
            # release both need.
            printer._model_pick_lock.acquire()
            parked = True
            child_release.set()
            child_state = self._wait_for(
                lambda: agent_state.get(child_task_id),
                what="the child's state",
            )
            self._wait_for(
                lambda: not child_state.is_task_active,
                what="the child's run to end",
            )
            agent_server._state_lock.acquire()
            state_locked = True
            printer._model_pick_lock.release()
            parked = False
            self._wait_for(
                lambda: child_task_id in printer._closed_tasks,
                what="the child's cleanup_task",
            )
            assert not _subagent_is_done(child_task_id), (
                "the child's thread slot must still be held while its teardown is parked"
            )

            # A client replaying the parent now (a reconnect, a history
            # click) announces the child's tab from the state it sees.
            announced_from = len(received)
            agent_server._open_persisted_subagent_tabs(
                parent_task_id=parent.task_id,
                parent_tab_id=PARENT_TAB_ID,
            )
            announcement = self._wait_for(
                lambda: next(
                    (
                        e
                        for e in list(received)[announced_from:]
                        if e.get("type") == "openSubagentTab"
                        and e.get("tab_id") == viewer_sub_tab_id
                    ),
                    None,
                ),
                what="the replay's sub-agent tab announcement",
            )
            assert announcement["isDone"] is False
        finally:
            if parked:
                printer._model_pick_lock.release()
            if state_locked:
                agent_server._state_lock.release()
            child_release.set()
        dispatcher.join(timeout=60)
        assert not dispatcher.is_alive(), "child dispatch never returned"
        assert child_out["result"].success is True
        self._wait_for(
            lambda: _subagent_is_done(child_task_id),
            what="the child's thread slot release",
        )

        # The requirement: a sub-agent tab closes as soon as the
        # sub-agent finishes.  The tab was announced live, so a
        # ``subagentDone`` for it must follow that announcement.
        def _after_announcement() -> list[dict[str, Any]]:
            events = list(received)
            return events[events.index(announcement, announced_from) + 1 :]

        def _closed_after_announcement() -> bool:
            return any(
                e.get("type") == "subagentDone" and e.get("tab_id") == viewer_sub_tab_id
                for e in _after_announcement()
            )

        deadline = time.monotonic() + 5.0
        while not _closed_after_announcement():
            assert time.monotonic() < deadline, (
                "no subagentDone for the tab announced as live; events: "
                f"{[(e.get('type'), e.get('tab_id'), e.get('isDone')) for e in list(received)]}"
            )
            time.sleep(0.02)
