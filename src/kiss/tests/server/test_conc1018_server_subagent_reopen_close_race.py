# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a sub-agent tab reopened by ``resumeSession`` while another client
closes it ends up consistent -- open AND streaming, or closed.

``_replay_session`` for a sub-agent tab used to commit in three
separate ``_state_lock`` sections (attach + subscribe; clear
``frontend_closed``; forget the user close) with the ``openSubagentTab``
announcement after them, while ``_close_tab``'s tail (mark closed,
broadcast ``closeSubagentTab``, remember the close, unsubscribe) runs in
ONE section.  A close landing between the attach and the commit left
the tab marked open on the reopening client, yet unsubscribed: the
sub-agent's later events never reached it (findings H1-F1).

Real daemon, two real client connections, scripted LLM loop (the
parent fans out one child through ``run_agents_parallel``; the child
blocks, then emits a ``tool_call`` on demand).  The attach-to-commit
window is widened with the suite's ``KISS_RACE_DELAY`` hook and the
close is sent while the replay thread sits in it (frame probe), so
client B sees ``closeSubagentTab`` and THEN its own ``openSubagentTab``:
B's tab is open, and the child's next event must reach it.
"""

from __future__ import annotations

import os
import sys
import threading
import uuid
from types import FrameType
from typing import Any

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar import persistence as _persistence
from kiss.core.models.model_info import get_available_models
from kiss.tests.server.test_subagent_tabs_two_clients import _fan_out
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

PARENT_PROMPT = "fan out one child conc1018"
CHILD_MARK = "CONC1018CHILD"
STEP_TOOL = "conc1018_step"


def _replay_thread_in_window() -> bool:
    """Whether some thread sits in ``_race_delay`` called by ``_replay_session``."""
    for top in sys._current_frames().values():
        names: set[str] = set()
        frame: FrameType | None = top
        while frame is not None:
            names.add(frame.f_code.co_name)
            frame = frame.f_back
        if {"_race_delay", "_replay_session"} <= names:
            return True
    return False


def _last_index(client: Client, event_type: str, **fields: Any) -> int:
    """Index in ``client.events`` of the latest *event_type* matching *fields* (-1: none)."""
    events = list(client.events)
    for i in range(len(events) - 1, -1, -1):
        e = events[i]
        if e.get("type") == event_type and all(e.get(k) == v for k, v in fields.items()):
            return i
    return -1


class SubagentReopenCloseRaceTest(TwoClientHarness):
    """Reopen racing a close: the reopened tab keeps the live fan-out."""

    def setUp(self) -> None:
        if not get_available_models():
            self.skipTest("no model credentials: the daemon refuses every run")
        super().setUp()
        self._cron_stop = cron_agent.start_scheduler_thread(
            interval=3600,
            endpoint_file=str(self.endpoint_file),
        )
        self.step_release = threading.Event()
        self.leaf_release = threading.Event()
        self._install_race_stub()

    def tearDown(self) -> None:
        os.environ.pop("KISS_RACE_DELAY", None)
        self.step_release.set()
        self.leaf_release.set()
        cron_agent.stop_scheduler_thread(self._cron_stop)
        super().tearDown()

    def _install_race_stub(self) -> None:
        """Replace the LLM loop: the parent fans out, the child blocks then emits a tool call."""
        step_release = self.step_release
        leaf_release = self.leaf_release

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
            assert printer is not None
            self_agent.printer = printer
            self_agent.model_name = str(kwargs.get("model_name") or "")
            self_agent.work_dir = str(kwargs.get("work_dir") or ".")
            prompt = str(kwargs.get("prompt_template", ""))
            if CHILD_MARK in prompt:
                step_release.wait(timeout=60)
                printer.print(
                    STEP_TOOL,
                    type="tool_call",
                    tool_input={"n": 1},
                    call_id=uuid.uuid4().hex,
                )
                leaf_release.wait(timeout=60)
            elif PARENT_PROMPT in prompt:
                _fan_out(self_agent, [f"{CHILD_MARK} blocking leaf"])
            self_agent.total_tokens_used = 5
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            printer.print(
                raw,
                type="result",
                step_count=1,
                total_tokens=5,
                cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run

    def _child_task_id(self) -> str | None:
        parents = [
            r for r in _persistence._load_history() if PARENT_PROMPT in str(r.get("task", ""))
        ]
        if not parents:
            return None
        rows = _persistence._load_subagent_rows_by_parent_task_id(str(parents[0]["id"]))
        return str(rows[0]["task_id"]) if rows else None

    def _resume(self, client: Client, child: str, sub_tab: str) -> None:
        """``resumeSession`` *sub_tab* on *client* and wait for ITS transcript.

        Counted, not merely awaited: an unscoped replay requested by
        another client also reaches this one.
        """
        before = len(client.of_type("task_events", tabId=sub_tab))
        client.send({"type": "resumeSession", "taskId": child, "tabId": sub_tab})
        self.wait_for(
            lambda: len(client.of_type("task_events", tabId=sub_tab)) > before,
            what=f"{client.name}: transcript of the sub-agent tab",
        )

    def _reopen_during_close(
        self,
        a: Client,
        b: Client,
        child: str,
        sub_tab: str,
    ) -> bool:
        """One attempt: B reopens *sub_tab* while A closes it inside the window.

        Returns ``True`` when the close landed between B's attach and
        its commit (B saw ``closeSubagentTab`` before its own
        ``openSubagentTab``), ``False`` when the window was missed.
        """
        closes_before = len(b.of_type("closeSubagentTab", tab_id=sub_tab))
        opens_before = len(b.of_type("openSubagentTab", tab_id=sub_tab))
        os.environ["KISS_RACE_DELAY"] = "0.1"
        try:
            b.send({"type": "resumeSession", "taskId": child, "tabId": sub_tab})
            self.wait_for(
                _replay_thread_in_window,
                timeout=10,
                what="B's replay thread to enter the attach-to-commit window",
            )
            a.send({"type": "closeTab", "tabId": sub_tab})
            self.wait_for(
                lambda: (
                    len(b.of_type("closeSubagentTab", tab_id=sub_tab)) > closes_before
                    and len(b.of_type("openSubagentTab", tab_id=sub_tab)) > opens_before
                ),
                what="B: the close and the reopen of the sub-agent tab",
            )
        finally:
            os.environ.pop("KISS_RACE_DELAY", None)
        close_at = _last_index(b, "closeSubagentTab", tab_id=sub_tab)
        open_at = _last_index(b, "openSubagentTab", tab_id=sub_tab)
        return close_at < open_at

    def test_reopened_tab_keeps_the_live_fan_out(self) -> None:
        a = self.open_client("a")
        a.ready()
        b = self.open_client("b")
        b.ready()
        self.wait_event(b, "tabs_state")
        a.send(
            {
                "type": "submit",
                "prompt": PARENT_PROMPT,
                "model": "",
                "attachments": [],
                "workDir": self.repo,
                "tabId": a.tab_id,
            }
        )
        child = self.wait_for(self._child_task_id, what="the child's history row")
        sub_tab = f"{a.tab_id}__sub_{child}"
        for client in (a, b):
            self.wait_event(client, "new_tab", task_id=child)
            self._resume(client, child, sub_tab)

        hit = False
        for _ in range(6):
            if self._reopen_during_close(a, b, child, sub_tab):
                hit = True
                break
            # The close landed after B's commit: B's tab is closed.
            # Reopen it on B and try the window again.
            self._resume(b, child, sub_tab)
        self.assertTrue(hit, "never landed the close inside B's attach-to-commit window")

        # B's tab is open (its last event for the tab is the reopen):
        # the child's next event must reach it.
        self.step_release.set()
        self.wait_event(b, "tool_call", timeout=10, tabId=sub_tab, name=STEP_TOOL)
        self.leaf_release.set()
        self.wait_for(
            lambda: a.of_type("status", running=False, tabId=a.tab_id),
            what="parent task to finish",
        )
