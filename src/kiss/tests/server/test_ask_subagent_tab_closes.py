# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the ``/ask`` sub-agent tab closes when its run ends and stays closed.

A question typed into a RUNNING task's tab (steer mode) is answered by
a side-channel sub-agent: ``_cmd_append_user_message`` dispatches
``ask_sea`` as a child of the running task, the frontend shows it as a
nested sub-agent tab under the asking tab, and the answer lands in the
parent's transcript as an ``ask_answer`` panel.  The nested tab is
scaffolding — once the answering run ends it must be closed, on every
client, and it must not come back:

1. live: the daemon broadcasts ``subagentDone`` for every tab watching
   the child when the run ends (the frontend closes the tab on it);
2. finished before a client attached (the answer came back faster
   than the client's ``resumeSession``): the child's replay must close
   the tab, not re-open it as a finished sub-agent tab;
3. parent replay (every webview ``ready``, reload, reconnect, history
   click re-announces the parent's persisted sub-agent rows): a
   finished ``/ask`` child must be announced as closed, never as an
   open finished tab.  ``run_parallel`` / ``run_agent`` children keep
   their finished tabs (their fan-out panel owns them); the ``/ask``
   child has no such owner, which is what made it reappear.

Only the LLM boundary (``RelentlessAgent.run``) is stubbed; the daemon
runs the parent, accepts the ``/ask`` over its own socket, persists the
child under the parent and replays both for real.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar import persistence as _persistence
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_run_agent_subagent_tab import DaemonUdsHarness

pytestmark = requires_unix_sockets

PARENT_TAB_ID = "webtab-asker-1"
VIEWER_SUB_TAB_ID = "webtab-asker-1__sub_child"
PARENT_MARKER = "long running parent task ax3"
QUESTION = "what is the parent doing right now ax3?"


def _result_text() -> str:
    return "success: true\nis_continue: false\nsummary: done\n"


class AskSubagentTabClosesTest(DaemonUdsHarness):
    """The ``/ask`` child's tab is closed once its run ends, everywhere."""

    def setUp(self) -> None:
        super().setUp()
        # The daemon-side /ask dispatch goes back through the daemon's
        # own socket, which the scheduler records at boot.
        self._saved_daemon_sock = cron_agent._daemon_sock_path
        cron_agent._daemon_sock_path = self.sock_path

    def tearDown(self) -> None:
        cron_agent._daemon_sock_path = self._saved_daemon_sock
        super().tearDown()

    def _install_stub(  # type: ignore[override]
        self,
        parent_release: threading.Event,
        child_release: threading.Event | None,
    ) -> None:
        """Parent blocks on *parent_release*; the /ask child on *child_release*."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            prompt = str(kwargs.get("prompt_template", ""))
            # The child's prompt also quotes the parent's task (chat
            # context), so the question is matched first.
            if QUESTION in prompt:
                if child_release is not None:
                    child_release.wait(timeout=60)
            elif PARENT_MARKER in prompt:
                parent_release.wait(timeout=60)
            self_agent.total_tokens_used = 5
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = _result_text()
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1,
                    total_tokens=5, cost="$0.0010",
                )
            return raw

        self._parent_class.run = stub_run

    def _start_parent(self, received: list[dict[str, Any]]) -> str:
        """Run the blocked parent on the viewer's tab; return its task id."""
        self._send_from_viewer({
            "type": "run",
            "tabId": PARENT_TAB_ID,
            "prompt": PARENT_MARKER,
            "workDir": self.repo,
            "useWorktree": False,
            "autoCommit": False,
        })
        settings = self._wait_for(
            lambda: next(
                (
                    e for e in list(received)
                    if e.get("type") == "task_settings"
                    and e.get("tabId") == PARENT_TAB_ID
                    and (e.get("settings") or {}).get("task_id")
                ),
                None,
            ),
            what="the parent's persisted task id",
        )
        return str(settings["settings"]["task_id"])

    def _ask(self, received: list[dict[str, Any]]) -> str:
        """Type ``/ask`` into the running tab; return the child's task id."""
        self._send_from_viewer({
            "type": "appendUserMessage",
            "tabId": PARENT_TAB_ID,
            "prompt": f"/ask {QUESTION}",
        })
        new_tab = self._wait_for(
            lambda: next(
                (
                    e for e in list(received)
                    if e.get("type") == "new_tab"
                    and e.get("parent_tab_id") == PARENT_TAB_ID
                ),
                None,
            ),
            what="new_tab broadcast for the /ask child",
        )
        child_task_id = str(new_tab.get("task_id") or "")
        assert child_task_id, f"new_tab carries no task id: {new_tab!r}"
        return child_task_id

    @staticmethod
    def _events_for_tab(
        received: list[dict[str, Any]], kind: str, tab_id: str,
    ) -> list[dict[str, Any]]:
        return [
            e for e in list(received)
            if e.get("type") == kind and e.get("tab_id") == tab_id
        ]

    def _wait_for_answer(self, received: list[dict[str, Any]]) -> None:
        self._wait_for(
            lambda: any(
                e.get("type") == "ask_answer"
                and e.get("tabId") == PARENT_TAB_ID
                for e in list(received)
            ),
            what="the ask_answer panel in the parent tab",
        )

    def _finish_parent(
        self,
        parent_release: threading.Event,
        received: list[dict[str, Any]],
        parent_task_id: str,
    ) -> None:
        parent_release.set()
        self._wait_for(
            lambda: any(
                e.get("type") == "result"
                and e.get("tabId") == PARENT_TAB_ID
                and str(e.get("taskId")) == parent_task_id
                for e in list(received)
            ),
            what="the parent's result",
        )

    def test_live_done_closes_tab_and_replays_keep_it_closed(self) -> None:
        """Watching client: ``subagentDone`` live, and again on every replay."""
        parent_release = threading.Event()
        child_release = threading.Event()
        self._install_stub(parent_release, child_release)
        received = self._open_viewer()
        parent_task_id = self._start_parent(received)
        try:
            child_task_id = self._ask(received)
            # What media/main.js does on new_tab: open a nested tab and
            # subscribe it to the child's stream.
            self._send_from_viewer({
                "type": "resumeSession",
                "taskId": child_task_id,
                "tabId": VIEWER_SUB_TAB_ID,
            })
            self._wait_for(
                lambda: VIEWER_SUB_TAB_ID
                in self.server._printer._fanout_targets(child_task_id),
                what="viewer sub-agent tab subscription",
            )
            child_release.set()
            # 1. Live completion closes the watching tab.
            self._wait_for(
                lambda: self._events_for_tab(
                    received, "subagentDone", VIEWER_SUB_TAB_ID,
                ),
                what="subagentDone for the watching sub-agent tab",
            )
            self._wait_for_answer(received)

            # The child row is persisted under the parent and marked as
            # a side-channel child: that mark is what the replays below
            # key on.
            nested = _persistence._load_subagent_rows_by_parent_task_id(
                parent_task_id,
            )
            child_row = next(
                row for row in nested
                if str(row.get("task_id")) == child_task_id
            )
            sub_extra = json.loads(str(child_row.get("extra") or "{}"))
            assert sub_extra.get("subagent", {}).get("side_channel") is True, (
                f"child row is not marked side_channel: {sub_extra!r}"
            )

            # 2. Parent replay while the parent still runs (a reload of
            #    the webview): the finished child is announced closed.
            parent_sub_tab = f"{PARENT_TAB_ID}__sub_{child_task_id}"
            self._send_from_viewer({
                "type": "resumeSession",
                "taskId": parent_task_id,
                "tabId": PARENT_TAB_ID,
            })
            self._wait_for(
                lambda: self._events_for_tab(
                    received, "subagentDone", parent_sub_tab,
                ),
                what="subagentDone for the finished child on parent replay",
            )
            assert self._events_for_tab(
                received, "openSubagentTab", parent_sub_tab,
            ) == [], "a finished /ask child must not be re-opened"
        finally:
            parent_release.set()
            child_release.set()
        self._finish_parent(parent_release, received, parent_task_id)

        # 3. Parent replay after the parent finished (history click,
        #    reconnect): still announced closed.
        before = len(received)
        self._send_from_viewer({
            "type": "resumeSession",
            "taskId": parent_task_id,
            "tabId": PARENT_TAB_ID,
        })
        parent_sub_tab = f"{PARENT_TAB_ID}__sub_{child_task_id}"
        self._wait_for(
            lambda: [
                e for e in list(received)[before:]
                if e.get("type") == "subagentDone"
                and e.get("tab_id") == parent_sub_tab
            ],
            what="subagentDone on the finished parent's replay",
        )
        assert self._events_for_tab(
            received, "openSubagentTab", parent_sub_tab,
        ) == [], "a finished /ask child must not be re-opened"
        # The parent's own transcript still carries the answer.
        replayed_parent = [
            e for e in list(received)[before:]
            if e.get("type") == "task_events"
            and e.get("tabId") == PARENT_TAB_ID
        ]
        assert replayed_parent, "parent replay produced no task_events"
        assert any(
            ev.get("type") == "ask_answer"
            for ev in replayed_parent[-1].get("events", [])
        ), "the ask_answer panel must survive the parent replay"

    def test_child_finished_before_client_attached_is_closed(self) -> None:
        """A fast answer: the child's own replay closes the tab."""
        parent_release = threading.Event()
        self._install_stub(parent_release, None)
        received = self._open_viewer()
        parent_task_id = self._start_parent(received)
        try:
            child_task_id = self._ask(received)
            # The answer arrives before this client subscribed to the
            # child, so no live subagentDone can reach its tab.
            self._wait_for_answer(received)
            assert self._events_for_tab(
                received, "subagentDone", VIEWER_SUB_TAB_ID,
            ) == []
            # The late resumeSession the frontend sends on new_tab.
            self._send_from_viewer({
                "type": "resumeSession",
                "taskId": child_task_id,
                "tabId": VIEWER_SUB_TAB_ID,
            })
            self._wait_for(
                lambda: self._events_for_tab(
                    received, "subagentDone", VIEWER_SUB_TAB_ID,
                ),
                what="subagentDone from the finished child's replay",
            )
            assert self._events_for_tab(
                received, "openSubagentTab", VIEWER_SUB_TAB_ID,
            ) == [], "a finished /ask child must not be re-opened"
        finally:
            parent_release.set()
        self._finish_parent(parent_release, received, parent_task_id)

    def test_running_child_is_still_announced_open(self) -> None:
        """Replays while the child runs keep showing it (with the spinner)."""
        parent_release = threading.Event()
        child_release = threading.Event()
        self._install_stub(parent_release, child_release)
        received = self._open_viewer()
        parent_task_id = self._start_parent(received)
        try:
            child_task_id = self._ask(received)
            parent_sub_tab = f"{PARENT_TAB_ID}__sub_{child_task_id}"
            # A second client (or a reload) replays the parent while the
            # answer is still being computed: the child tab is opened
            # running, exactly like any other live sub-agent.
            self._send_from_viewer({
                "type": "resumeSession",
                "taskId": parent_task_id,
                "tabId": PARENT_TAB_ID,
            })
            announced = self._wait_for(
                lambda: self._events_for_tab(
                    received, "openSubagentTab", parent_sub_tab,
                ),
                what="openSubagentTab for the running child",
            )
            assert announced[0].get("isDone") is False
            assert self._events_for_tab(
                received, "subagentDone", parent_sub_tab,
            ) == []
            # And the reattached tab gets its completion signal live.
            child_release.set()
            self._wait_for(
                lambda: self._events_for_tab(
                    received, "subagentDone", parent_sub_tab,
                ),
                what="subagentDone for the reattached child tab",
            )
            self._wait_for_answer(received)
        finally:
            parent_release.set()
            child_release.set()
        self._finish_parent(parent_release, received, parent_task_id)
        time.sleep(0.2)
        done_ids = [
            str(e.get("tab_id")) for e in list(received)
            if e.get("type") == "subagentDone"
        ]
        assert len(done_ids) == len(set(done_ids)), (
            f"subagentDone double-fired: {done_ids!r}"
        )
