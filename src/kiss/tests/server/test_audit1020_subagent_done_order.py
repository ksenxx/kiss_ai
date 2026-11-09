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

Audit finding A20-1 (second test): the same window existed for every
INTERMEDIATE row of a multi-``<task>`` child, whose ``subagentDone``
went out before the row was persisted and while the registry still
reported the row live (the state is re-keyed only when the NEXT row is
allocated).  A parent replay in that window reopened the finished row
as a live sub-agent tab that nothing closed afterwards: the end-of-run
signal names the last row only.  The child's thread is held at the
exact statement by a trace hook the stub installs on it.
"""

from __future__ import annotations

import sys
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

    def test_replay_between_subtasks_closes_the_finished_intermediate_row(self) -> None:
        """A parent replay between two ``<task>`` blocks never keeps the first row's tab open."""
        first_marker = "first child subtask a20-1"
        second_marker = "second child subtask a20-1"
        held = threading.Event()
        release = threading.Event()
        second_release = threading.Event()

        def hold_at_first_row_persistence(frame: Any, event: str, _arg: Any) -> None:
            # Holds the child's thread on entry to the first
            # ``_persist_subtask_row`` call (the intermediate row's),
            # i.e. in the inter-subtask bookkeeping, before the next
            # row is allocated and the state re-keyed to it.
            if event == "call" and frame.f_code.co_name == "_persist_subtask_row":
                sys.settrace(None)
                held.set()
                release.wait(timeout=30)
            return None

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            prompt = str(kwargs.get("prompt_template", ""))
            if first_marker in prompt:
                sys.settrace(hold_at_first_row_persistence)
            elif second_marker in prompt:
                second_release.wait(timeout=30)
            self_agent.total_tokens_used = 5
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
            if printer is not None:  # pragma: no branch
                printer.print(raw, type="result", step_count=1, total_tokens=5, cost="$0.0010")
            return raw

        self._parent_class.run = stub_run
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
                f"<task>{first_marker}</task>\n<task>{second_marker}</task>",
                work_dir=self.repo,
                parent_task_id=parent.task_id,
                parent_tab_id=PARENT_TAB_ID,
                endpoint_file=self.endpoint_file,
                timeout=60,
            )

        def new_tab_rows() -> list[str]:
            return [
                str(e["task_id"])
                for e in list(received)
                if e.get("type") == "new_tab" and e.get("parent_tab_id") == PARENT_TAB_ID
            ]

        def announcement_after(start: int, sub_tab_id: str) -> dict[str, Any] | None:
            return next(
                (
                    e
                    for e in list(received)[start:]
                    if e.get("type") == "openSubagentTab" and e.get("tab_id") == sub_tab_id
                ),
                None,
            )

        dispatcher = threading.Thread(target=dispatch_child, daemon=True)
        dispatcher.start()
        agent_server = self.server._agent_server
        try:
            assert held.wait(timeout=30), "the child never reached its first row's persistence"
            first_row = new_tab_rows()[0]
            first_sub_tab = _viewer_sub_tab_id(first_row)
            # A client replaying the parent now (a reconnect, a history
            # click) announces the first row's tab from the state it sees.
            announced_from = len(received)
            agent_server._open_persisted_subagent_tabs(
                parent_task_id=parent.task_id,
                parent_tab_id=PARENT_TAB_ID,
            )
            announcement = self._wait_for(
                lambda: announcement_after(announced_from, first_sub_tab),
                what="the replay's announcement of the first row",
            )
            release.set()
            # The second subtask runs on the same state, re-keyed to its
            # own row: the first row is history now, the second is live.
            second_row = self._wait_for(
                lambda: next((r for r in new_tab_rows() if r != first_row), None),
                what="the second row's new_tab",
            )
            assert _subagent_is_done(first_row)
            assert not _subagent_is_done(second_row)
            replayed_from = len(received)
            agent_server._open_persisted_subagent_tabs(
                parent_task_id=parent.task_id,
                parent_tab_id=PARENT_TAB_ID,
            )
            first_again = self._wait_for(
                lambda: announcement_after(replayed_from, first_sub_tab),
                what="the second replay's announcement of the first row",
            )
            assert first_again["isDone"] is True
            second_live = self._wait_for(
                lambda: announcement_after(replayed_from, _viewer_sub_tab_id(second_row)),
                what="the second replay's announcement of the second row",
            )
            assert second_live["isDone"] is False
        finally:
            release.set()
            second_release.set()
        dispatcher.join(timeout=60)
        assert not dispatcher.is_alive(), "child dispatch never returned"
        assert child_out["result"].success is True
        self._wait_for(
            lambda: _subagent_is_done(second_row),
            what="the child's thread slot release",
        )
        if announcement["isDone"] is False:
            # Announced live: its ``subagentDone`` must follow the
            # announcement, as for the last row.
            events = list(received)
            after = events[events.index(announcement, announced_from) + 1 :]
            assert any(
                e.get("type") == "subagentDone" and e.get("tab_id") == first_sub_tab
                for e in after
            ), (
                "the first row's tab was announced live and never closed; events: "
                f"{[(e.get('type'), e.get('tab_id'), e.get('isDone')) for e in after]}"
            )
