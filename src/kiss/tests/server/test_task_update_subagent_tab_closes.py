# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a finished task-update child never re-opens its tab.

The task-info panel polls ``getTaskUpdate`` while a tab shows a running
task; the daemon answers by running the task-update agent
(:func:`kiss.server.task_update.run_task_update_sea`) as an in-process
sub-agent of that task.  Like the ``/ask`` answerer it is a side
channel: its report goes to the task-info panel, so its nested tab is
open only while it runs.  Before the ``side_channel`` stamp, every
reload of the webview (a ``resumeSession`` of the parent) re-announced
each finished update as ``openSubagentTab{isDone: true}`` — one dead
tab per periodic update.

Real daemon over a UDS socket (:class:`DaemonUdsHarness`); only the LLM
boundary (``RelentlessAgent.run``) is stubbed.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from kiss.agents.seas import task_update_sea
from kiss.agents.sorcar import persistence as _persistence
from kiss.server.server import _is_side_channel_row
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_run_agent_subagent_tab import DaemonUdsHarness

pytestmark = requires_unix_sockets

PARENT_TAB_ID = "webtab-updated-1"
PARENT_MARKER = "long running parent task tu7"
# ``task_update_sea.PROMPT_TEMPLATE`` minus its ``{task_id}`` tail.
CHILD_MARKER = task_update_sea.PROMPT_TEMPLATE.split("{", 1)[0]


def _result_text() -> str:
    return (
        "success: true\nis_continue: false\n"
        "summary: <h4>Done so far</h4><ul><li>read parser.py</li></ul>\n"
    )


class TaskUpdateSubagentTabClosesTest(DaemonUdsHarness):
    """The task-update child's tab closes when done and stays closed on replay."""

    def _install_stub(  # type: ignore[override]
        self, parent_release: threading.Event,
    ) -> None:
        """Parent blocks on *parent_release*; the task-update child answers at once."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            prompt = str(kwargs.get("prompt_template", ""))
            # The child's prompt quotes the parent's (chat context), so
            # the child marker is tested first.
            if CHILD_MARKER not in prompt and PARENT_MARKER in prompt:
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

    def _request_update(self, received: list[dict[str, Any]]) -> str:
        """Poll ``getTaskUpdate`` for the parent's tab; return the child's task id."""
        # ``refresh: true`` is the panel's refresh button: it forces a
        # run even when a fresh report exists.  The reply itself is a
        # direct ``taskUpdate``; the child's ``new_tab`` is broadcast.
        self._send_from_viewer({
            "type": "getTaskUpdate",
            "tabId": PARENT_TAB_ID,
            "knownSig": "",
            "token": "t1",
            "refresh": True,
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
            what="new_tab broadcast for the task-update child",
        )
        child_task_id = str(new_tab.get("task_id") or "")
        assert child_task_id, f"new_tab carries no task id: {new_tab!r}"
        # What the webview does on ``new_tab``: open the nested tab and
        # attach it to the child's session.
        self._send_from_viewer({
            "type": "resumeSession",
            "taskId": child_task_id,
            "tabId": f"{PARENT_TAB_ID}__sub_{child_task_id}",
        })
        return child_task_id

    @staticmethod
    def _events_for_tab(
        received: list[dict[str, Any]], kind: str, tab_id: str,
    ) -> list[dict[str, Any]]:
        return [
            e for e in list(received)
            if e.get("type") == kind and e.get("tab_id") == tab_id
        ]

    def _assert_replay_closes_child(
        self,
        received: list[dict[str, Any]],
        parent_task_id: str,
        child_task_id: str,
    ) -> None:
        """Replay the parent (what a reload does) and check the child's fate."""
        parent_sub_tab = f"{PARENT_TAB_ID}__sub_{child_task_id}"
        before = len(received)
        self._send_from_viewer({
            "type": "resumeSession",
            "taskId": parent_task_id,
            "tabId": PARENT_TAB_ID,
        })
        self._wait_for(
            lambda: [
                e for e in list(received)[before:]
                if e.get("type") == "subagentDone"
                and e.get("tab_id") == parent_sub_tab
            ],
            what="subagentDone for the finished task-update child on replay",
        )
        # The replay is complete once the parent's transcript arrived;
        # by then any openSubagentTab for the child would be in too.
        self._wait_for(
            lambda: [
                e for e in list(received)[before:]
                if e.get("type") == "task_events"
                and e.get("tabId") == PARENT_TAB_ID
            ],
            what="the parent's replayed transcript",
        )
        assert self._events_for_tab(
            received, "openSubagentTab", parent_sub_tab,
        ) == [], "a finished task-update child must not be re-opened"

    def test_finished_update_child_stays_closed_on_replay(self) -> None:
        """Live ``subagentDone`` closes the tab; every replay repeats it."""
        parent_release = threading.Event()
        self._install_stub(parent_release)
        received = self._open_viewer()
        parent_task_id = self._start_parent(received)
        try:
            child_task_id = self._request_update(received)
            child_tab_prefix = f"task-{parent_task_id}__update-"
            parent_sub_tab = f"{PARENT_TAB_ID}__sub_{child_task_id}"
            # 1. The live run announces its own tab closed when done —
            # both the daemon-side id and the webview's nested tab.
            self._wait_for(
                lambda: [
                    e for e in list(received)
                    if e.get("type") == "subagentDone"
                    and str(e.get("tab_id", "")).startswith(child_tab_prefix)
                ],
                what="live subagentDone for the task-update child",
            )
            self._wait_for(
                lambda: self._events_for_tab(
                    received, "subagentDone", parent_sub_tab,
                ),
                what="subagentDone for the webview's nested update tab",
            )
            # The row is persisted under the parent as a side channel:
            # the very predicate the daemon's replays key on.
            child_row = self._wait_for(
                lambda: next(
                    (
                        row for row in
                        _persistence._load_subagent_rows_by_parent_task_id(
                            parent_task_id,
                        )
                        if str(row.get("task_id")) == child_task_id
                    ),
                    None,
                ),
                what="the persisted task-update child row",
            )
            assert _is_side_channel_row(child_row), (
                f"task-update child is not a side channel: {child_row!r}"
            )
            # Wait until the child is no longer registered as running;
            # ``_subagent_is_done`` is what decides the replay branch.
            from kiss.server.server import _subagent_is_done

            self._wait_for(
                lambda: _subagent_is_done(child_task_id),
                what="the child to leave the agent-state registry",
            )

            # 2. Reload while the parent still runs.
            self._assert_replay_closes_child(
                received, parent_task_id, child_task_id,
            )
        finally:
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
        # 3. Reload / history click after the parent finished.
        time.sleep(0.05)
        self._assert_replay_closes_child(
            received, parent_task_id, child_task_id,
        )
