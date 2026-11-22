# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a sub-agent tab resumed while another client closes it ends up open AND streaming.

``_replay_session`` for a sub-agent tab used to commit in separate
``_state_lock`` sections (attach + subscribe; clear
``frontend_closed``) with the ``openSubagentTab`` announcement after
them, while ``_close_tab``'s tail (mark closed, unsubscribe) runs in
ONE section.  A close landing between the attach and the commit left
the tab announced to the resuming client, yet unsubscribed: the
sub-agent's later events never reached it (findings H1-F1).

Real daemon, two real client connections, scripted LLM loop (the
parent fans out one child through ``run_agents_parallel``; the child
blocks, then emits a ``tool_call`` on demand).  B's replay is held
between its attach and its commit (``Pause``, a trace hook on the real
code path) until A's ``closeTab`` of the same tab has unsubscribed it
(a page sends none for a sub-agent's tab; an older page or a script
can): B is then announced the running sub-agent, and the child's next
event must reach B's tab.

The daemon's own teardown takes the same section: a replay held the
same way while the child ends tells B the end and leaves nothing
subscribed or pinned for the tab, with no page closing it.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar import persistence as _persistence
from kiss.core.models.model_info import get_available_models
from kiss.server.server import AgentServer
from kiss.tests.server.pause import Pause
from kiss.tests.server.test_subagent_tabs_two_clients import _fan_out
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

PARENT_PROMPT = "fan out one sub-agent conc1018"
CHILD_MARK = "CONC1018CHILD"
STEP_TOOL = "conc1018_step"


class SubagentResumeCloseRaceTest(TwoClientHarness):
    """A resume racing a close: the resumed tab keeps the live fan-out."""

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
            # What the real ``run`` records: the dispatcher reads it to see
            # that the task asks for a sub-agent (a same-family child).
            self_agent.task_description = prompt
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

    def test_resumed_tab_keeps_the_live_fan_out(self) -> None:
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

        # A's close lands between the attach and the commit of B's resume.
        watchers = self.server._printer._fanout_targets
        announced = len(b.of_type("openSubagentTab", tab_id=sub_tab))
        held = Pause(
            AgentServer._announce_or_end_subagent_tab.__code__,
            lambda local: local["tab_id"] == sub_tab,
        )
        with held:
            b.send({"type": "resumeSession", "taskId": child, "tabId": sub_tab})
            self.assertTrue(held.paused.wait(30), "B's replay never got to its commit")
            a.send({"type": "closeTab", "tabId": sub_tab})
            self.wait_for(
                lambda: sub_tab not in watchers(child), what="A's close of the sub-agent tab",
            )
        self.wait_for(
            lambda: len(b.of_type("openSubagentTab", tab_id=sub_tab)) > announced,
            what="B: the running sub-agent announced",
        )

        # B's tab is open: the child's next event must reach it.
        self.step_release.set()
        self.wait_event(b, "tool_call", timeout=10, tabId=sub_tab, name=STEP_TOOL)

        # B asks once more and the child ends between that replay's
        # attach and its commit: B is told the end by both, and the
        # daemon holds nothing for the tab though no page closed it.
        told = len(b.of_type("subagentDone", tab_id=sub_tab))
        ending = Pause(
            AgentServer._announce_or_end_subagent_tab.__code__,
            lambda local: local["tab_id"] == sub_tab,
        )
        with ending:
            b.send({"type": "resumeSession", "taskId": child, "tabId": sub_tab})
            self.assertTrue(ending.paused.wait(30), "B's second replay never got to its commit")
            self.leaf_release.set()
            self.wait_event(a, "subagentDone", tab_id=sub_tab, task_id=child)
        self.wait_for(
            lambda: len(b.of_type("subagentDone", tab_id=sub_tab)) == told + 2,
            what="B: the end from the child and from its own replay",
        )
        server = self.server._agent_server
        with server._state_lock:
            self.assertEqual(server.printer.tasks_for_tab(sub_tab), [])
            self.assertNotIn(sub_tab, server._tab_opened_task_ids)
        self.wait_for(
            lambda: a.of_type("status", running=False, tabId=a.tab_id),
            what="parent task to finish",
        )
