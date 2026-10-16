# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a sub-agent's question reaches every client; one answer settles it.

A child sub-task blocked in ``ask_user_question`` is a running sub-agent
whose tab must be open on every surface, with its question showing, so
whoever is looking can answer -- and that one answer must complete the
child and close its tab everywhere.

This test runs the REAL daemon on its local WSS endpoint with several
real client connections (``two_client_harness``); the LLM loop is
scripted (the parent fans out one child through ``run_agents_parallel``;
the child asks through the daemon's real question callback and records
the answer it gets).  Checked on the wire, per client:

1. Both connected clients are told to open the child's tab (``new_tab``)
   and, once they watch it, get its ``askUser`` question.
2. A client connecting MID-QUESTION gets the running child from its
   ``ready`` replay and the pending question once it opens the tab.
3. Closing the child's tab on one client closes it everywhere
   (``closeSubagentTab``); reopening it from one client (what expanding
   the parent's fan-out panel does) announces it again on every client,
   question included.
4. One client's ``userAnswer`` reaches the real child, every client gets
   ``askUserDone`` then ``subagentDone``, and the parent finishes.
5. A client connecting AFTER completion is told of no running child and
   no pending question.
"""

from __future__ import annotations

import json
import uuid
from functools import partial
from typing import Any

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.agent_dispatch import run_agents_parallel
from kiss.core.models.model_info import get_available_models
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

PARENT_PROMPT = "fan out one asking child zq7"
CHILD_MARK = "ASKINGCHILD"
QUESTION = "Which colour should the child use? (zq7)"
ANSWER = "teal zq7"


class RunAgentChildAskTwoClientsTest(TwoClientHarness):
    """Child tab and question on every client; one answer closes it everywhere."""

    def setUp(self) -> None:
        if not get_available_models():
            self.skipTest("no model credentials: the daemon refuses every run")
        super().setUp()
        self._cron_stop = cron_agent.start_scheduler_thread(
            interval=3600, endpoint_file=str(self.endpoint_file),
        )
        self.answers: dict[str, str] = {}
        self._install_plan()

    def tearDown(self) -> None:
        cron_agent.stop_scheduler_thread(self._cron_stop)
        super().tearDown()

    def _install_plan(self) -> None:
        """Replace the LLM loop: the parent fans out; the child asks."""
        answers = self.answers

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
            assert printer is not None, "a daemon run always has a printer"
            self_agent.printer = printer
            self_agent.work_dir = str(
                kwargs.get("work_dir")
                or getattr(self_agent, "work_dir", ".") or ".",
            )
            prompt = str(kwargs.get("prompt_template", ""))
            if CHILD_MARK in prompt:
                # The daemon's real question callback: it publishes the
                # question to every watcher and blocks for the answer.
                answers["child"] = str(
                    self_agent._ask_user_question_callback(QUESTION),
                )
            elif PARENT_PROMPT in prompt:
                tasks = [f"{CHILD_MARK} ask the user"]
                tool_input = {"tasks": json.dumps(tasks)}
                printer.print(
                    "run_parallel", type="tool_call", tool_input=tool_input,
                    call_id=uuid.uuid4().hex,
                )
                results = run_agents_parallel(
                    self_agent.work_dir, tasks, parent_agent=self_agent,
                )
                printer.print(
                    results, type="tool_result", tool_name="run_parallel",
                    tool_input=tool_input, is_error=False, interrupted=False,
                )
            self_agent.total_tokens_used = 5
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            printer.print(
                raw, type="result", step_count=1, total_tokens=5, cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run

    # -- helpers -----------------------------------------------------------

    def _child_task_id(self) -> str | None:
        for row in _persistence._load_history():
            if PARENT_PROMPT not in str(row.get("task", "")):
                continue
            for child in _persistence._load_subagent_rows_by_parent_task_id(
                str(row["id"]),
            ):
                if CHILD_MARK in str(child.get("task", "")):
                    return str(child["task_id"])
        return None

    def _watch_child(self, client: Client, task_id: str, parent_tab: str) -> str:
        """Subscribe *client* to the child's stream, as a page does on ``new_tab``."""
        sub_tab = f"{parent_tab}__sub_{task_id}"
        client.send({"type": "resumeSession", "taskId": task_id, "tabId": sub_tab})
        return sub_tab

    def _open_deferred_tab(self, client: Client, tab_id: str) -> None:
        """Open the registry tab *tab_id* on *client* as its user would."""
        offer = self.wait_event(client, "replay_deferred", tabId=tab_id)
        client.send({
            "type": "resumeSession",
            "id": offer["chatId"],
            "tabId": tab_id,
            "scoped": True,
        })
        self.wait_event(client, "task_events", tabId=tab_id)

    @staticmethod
    def _more_than(
        client: Client, event_type: str, key: str, value: str, count: int,
    ) -> bool:
        """Whether *client* got more than *count* *event_type* events with ``key == value``."""
        return len(client.of_type(event_type, **{key: value})) > count

    def _running_child_tabs(self, client: Client) -> set[str]:
        opened: set[str] = set()
        for e in list(client.events):
            if e.get("type") == "openSubagentTab" and not e.get("isDone"):
                opened.add(str(e["tab_id"]))
            elif e.get("type") in ("closeSubagentTab", "subagentDone"):
                opened.discard(str(e.get("tab_id")))
        return opened

    # -- the test ------------------------------------------------------------

    def test_child_tab_and_question_on_every_client(self) -> None:
        a = self.open_client("a")
        a.ready()
        b = self.open_client("b")
        b.ready()
        self.wait_event(b, "tabs_state")
        a.send({
            "type": "submit",
            "prompt": PARENT_PROMPT,
            "model": "",
            "attachments": [],
            "workDir": self.repo,
            "tabId": a.tab_id,
        })
        child_id = self.wait_for(self._child_task_id, what="child row")
        child_tab = f"{a.tab_id}__sub_{child_id}"

        # 1. Both clients open the child's tab and, watching it, get
        #    the question stamped with the child's tab.
        for client in (a, b):
            ev = self.wait_event(client, "new_tab", task_id=child_id)
            self.assertEqual(ev["parent_tab_id"], a.tab_id, (client.name, ev))
            self._watch_child(client, child_id, a.tab_id)
            ask = self.wait_event(client, "askUser", tabId=child_tab)
            self.assertEqual(ask["question"], QUESTION, (client.name, ask))

        # 2. A client connecting mid-question gets the running child from
        #    its replay and the pending question once it opens the tab.
        c = self.open_client("c")
        c.ready()
        self.wait_for(
            lambda: child_tab in self._running_child_tabs(c),
            what="c: running child on replay",
        )
        self._watch_child(c, child_id, a.tab_id)
        self.assertEqual(
            self.wait_event(c, "askUser", tabId=child_tab)["question"], QUESTION,
        )

        # 3. The user closes the child's tab on one client: closed
        #    everywhere (the child keeps running); reopening it from one
        #    client announces it again everywhere, question included.
        b.send({"type": "closeTab", "tabId": child_tab})
        for client in (a, b, c):
            self.wait_event(client, "closeSubagentTab", tab_id=child_tab)
        self.assertNotIn("child", self.answers, "the child must still be waiting")
        before = {
            cl.name: (
                len(cl.of_type("openSubagentTab", tab_id=child_tab)),
                len(cl.of_type("askUser", tabId=child_tab)),
            )
            for cl in (a, b, c)
        }
        self._watch_child(a, child_id, a.tab_id)
        for client in (a, b, c):
            self.wait_for(
                partial(self._more_than, client, "openSubagentTab", "tab_id",
                        child_tab, before[client.name][0]),
                what=f"{client.name}: child tab reopened",
            )
            self.wait_for(
                partial(self._more_than, client, "askUser", "tabId",
                        child_tab, before[client.name][1]),
                what=f"{client.name}: question shown again",
            )
            self.assertFalse(
                self.wait_event(client, "openSubagentTab", tab_id=child_tab)
                .get("isDone"),
            )

        # 4. One client answers: the real child gets the answer, every
        #    client is told the question is over and the child done.
        c.send({"type": "userAnswer", "tabId": child_tab, "answer": ANSWER})
        self.wait_for(lambda: self.answers.get("child"), what="the child's answer")
        self.assertEqual(self.answers["child"], ANSWER)
        for client in (a, b, c):
            self.wait_event(client, "askUserDone")
            self.wait_event(client, "subagentDone", tab_id=child_tab)
        self.wait_for(
            lambda: a.of_type("status", running=False, tabId=a.tab_id),
            what="parent task to finish",
        )
        _persistence._flush_chat_events(child_id)
        child_session = _persistence._load_chat_events_by_task_id(child_id)
        assert child_session is not None, "the child's session is persisted"
        self.assertEqual(child_session["task"], f"{CHILD_MARK} ask the user")

        # 5. A client connecting after completion: no running child, no
        #    pending question.
        late = self.open_client("late")
        late.ready()
        self._open_deferred_tab(late, a.tab_id)
        self.assertEqual(self._running_child_tabs(late), set())
        self.assertEqual(late.of_type("askUser"), [])
