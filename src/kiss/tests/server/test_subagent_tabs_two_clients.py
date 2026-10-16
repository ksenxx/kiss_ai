# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a running sub-agent's tab is announced to EVERY client, and closed
on every client once the sub-agent is over.

The invariant (the user's ``AGENTS.md`` under the daemon home, ``~/.s10s/``)::

    Across all surfaces of KISS Sorcar you must show the same tabs.
    If a subagent is running, its tab must be open unless the user has
    closed the tab.  A subagent tab must be closed as soon as the
    subagent finishes its task.

A surface is one chat page, i.e. one connection to the daemon.  This
test runs the REAL daemon on its local WSS endpoint with several real
client connections (``two_client_harness``); only the LLM loop is
scripted (the parent fans out through ``run_agents_parallel``, whose
children are sub-tasks of the same daemon; one child nests a
grandchild, and the leaves block until released).

Checked on the wire, per client:

1. While the leaves run: the two children AND the nested grandchild are
   announced (``new_tab``) to every client that was connected when they
   spawned; every announcement of the grandchild names the nesting
   child's tab as its parent.
2. A client connecting MID-RUN gets the same running sub-agents from its
   ``ready`` replay (``openSubagentTab``, not done).  A running child's
   tab the user closes on one client is closed on all of them
   (``closeSubagentTab``) and stays closed for a client connecting
   afterwards; the other running sub-agents keep their tabs.
3. After the children finish: every client gets ``subagentDone`` for the
   sub-agents it still shows.
4. A client connecting AFTER completion is never told to open a finished
   sub-agent's tab as running.
5. (second test) The user closes the RUNNING chat's tab -- the registry
   drops it for every client, whose pages close the child's tab with
   it -- and reopens the chat under a new tab id: the running child, and
   a child spawned after the reopen, are announced under the new tab to
   every client.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from functools import partial
from typing import Any

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.agent_dispatch import run_agents_parallel
from kiss.core.models.model_info import get_available_models
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

PARENT_PROMPT = "fan out: spawn two children xq9"
CHILD_MARK = "CHILDTASK"
NEST_MARK = "NESTING"
GRANDCHILD_MARK = "GRANDCHILD"
WAVE2_MARK = "SECONDWAVE"


def _fan_out(agent: Any, tasks: list[str]) -> None:
    """Call ``run_parallel`` the way the model does: the ``tool_call`` /
    ``tool_result`` events frame the fan-out in the transcript.  The
    children are daemon sub-tasks of the calling task
    (:func:`run_agents_parallel`, dispatched to the harness daemon)."""
    tool_input = {"tasks": json.dumps(tasks)}
    agent.printer.print(
        "run_parallel", type="tool_call", tool_input=tool_input,
        call_id=uuid.uuid4().hex,
    )
    results = run_agents_parallel(agent.work_dir, tasks, parent_agent=agent)
    agent.printer.print(
        results, type="tool_result", tool_name="run_parallel",
        tool_input=tool_input, is_error=False, interrupted=False,
    )


class SubagentTabsTwoClientsTest(TwoClientHarness):
    """Sub-agent tabs open everywhere, close everywhere."""

    def setUp(self) -> None:
        if not get_available_models():
            self.skipTest("no model credentials: the daemon refuses every run")
        super().setUp()
        # ``run_parallel`` children are dispatched to the daemon hosting
        # the parent; record this harness daemon's endpoint the way
        # kiss-web does at boot.
        self._cron_stop = cron_agent.start_scheduler_thread(
            interval=3600, endpoint_file=str(self.endpoint_file),
        )

    def tearDown(self) -> None:
        cron_agent.stop_scheduler_thread(self._cron_stop)
        super().tearDown()

    def _install_plan(self, plan: Any) -> None:
        """Replace the LLM loop: *plan(agent, prompt)* does the run's work."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            self_agent.printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            self_agent.model_name = str(
                kwargs.get("model_name")
                or getattr(self_agent, "model_name", "") or "",
            )
            self_agent.work_dir = str(
                kwargs.get("work_dir")
                or getattr(self_agent, "work_dir", ".") or ".",
            )
            plan(self_agent, str(kwargs.get("prompt_template", "")))
            self_agent.total_tokens_used = 5
            self_agent.budget_used = 0.001
            self_agent.total_steps = 1
            raw = "success: true\nis_continue: false\nsummary: done\n"
            if self_agent.printer is not None:  # pragma: no branch
                self_agent.printer.print(
                    raw, type="result", step_count=1,
                    total_tokens=5, cost="$0.0010",
                )
            return raw

        self._parent_class.run = stub_run

    def _install_fanout_stub(self, leaf_release: threading.Event) -> None:
        """Parent fans out two children; one nests a grandchild."""

        def plan(agent: Any, prompt: str) -> None:
            if GRANDCHILD_MARK in prompt:
                leaf_release.wait(timeout=60)
            elif NEST_MARK in prompt:
                # Fan out IMMEDIATELY: no client can have subscribed to
                # this sub-agent's tab yet, the worst case for parenting
                # the grandchild under a tab every client knows.
                _fan_out(agent, [f"{GRANDCHILD_MARK} leaf"])
            elif CHILD_MARK in prompt:
                leaf_release.wait(timeout=60)
            elif PARENT_PROMPT in prompt:
                _fan_out(agent, [
                    f"{CHILD_MARK} plain leaf",
                    f"{CHILD_MARK} {NEST_MARK} parent of a grandchild",
                ])

        self._install_plan(plan)

    def _install_two_wave_stub(
        self, wave1_release: threading.Event, wave2_release: threading.Event,
    ) -> None:
        """Parent fans out one child, then -- once it is over -- another."""

        def plan(agent: Any, prompt: str) -> None:
            if WAVE2_MARK in prompt:
                wave2_release.wait(timeout=60)
            elif CHILD_MARK in prompt:
                wave1_release.wait(timeout=60)
            elif PARENT_PROMPT in prompt:
                _fan_out(agent, [f"{CHILD_MARK} first wave"])
                _fan_out(agent, [f"{CHILD_MARK} {WAVE2_MARK} second wave"])

        self._install_plan(plan)

    # -- helpers -----------------------------------------------------------

    def _submit(self, client: Client, prompt: str) -> None:
        client.send({
            "type": "submit",
            "prompt": prompt,
            "model": "",
            "attachments": [],
            "workDir": self.repo,
            "tabId": client.tab_id,
        })

    def _parent_task_id(self) -> str | None:
        parents = [
            r for r in _persistence._load_history()
            if PARENT_PROMPT in str(r.get("task", ""))
        ]
        return str(parents[0]["id"]) if parents else None

    def _child_rows(self, parent_task_id: str) -> list[dict[str, Any]]:
        return _persistence._load_subagent_rows_by_parent_task_id(
            parent_task_id,
        )

    def _running_announced(self, client: Client) -> set[str]:
        """Sub-agent tabs *client* was told to open as running, minus closes."""
        opened: set[str] = set()
        for e in list(client.events):
            if e.get("type") == "openSubagentTab" and not e.get("isDone"):
                opened.add(str(e["tab_id"]))
            elif e.get("type") == "closeSubagentTab":
                opened.discard(str(e["tab_id"]))
            elif e.get("type") == "subagentDone":
                opened.discard(str(e.get("tab_id")))
        return opened

    def _watch_child(self, client: Client, task_id: str, parent_tab: str) -> str:
        """Subscribe *client* to a spawned child's stream, as a page does on ``new_tab``.

        Returns the shared sub-agent tab id ``{parent_tab}__sub_{task_id}``.
        """
        sub_tab = f"{parent_tab}__sub_{task_id}"
        client.send({"type": "resumeSession", "taskId": task_id, "tabId": sub_tab})
        return sub_tab

    @staticmethod
    def _registry_dropped(client: Client, tab_id: str) -> bool:
        """Whether a ``tabs_state`` snapshot without *tab_id* reached *client*."""
        return any(
            not any(t.get("tabId") == tab_id for t in e.get("tabs", []))
            for e in client.of_type("tabs_state")
        )

    def _open_deferred_tab(self, client: Client, tab_id: str) -> None:
        """Open the registry tab *tab_id* on *client* as its user would.

        A connecting client is offered the other tabs' transcripts
        (``replay_deferred``) and asks for one with a connection-scoped
        ``resumeSession`` when the user opens the tab (main.js
        ``requestDeferredReplay``).
        """
        offer = self.wait_event(client, "replay_deferred", tabId=tab_id)
        client.send({
            "type": "resumeSession",
            "id": offer["chatId"],
            "tabId": tab_id,
            "scoped": True,
        })
        self.wait_event(client, "task_events", tabId=tab_id)

    # -- the tests -----------------------------------------------------------

    def test_subagent_tabs_open_and_close_on_every_client(self) -> None:
        leaf_release = threading.Event()
        self._install_fanout_stub(leaf_release)

        # Two clients connected before the run.
        a = self.open_client("a")
        a.ready()
        b = self.open_client("b")
        b.ready()
        self.wait_event(b, "tabs_state")
        self._submit(a, PARENT_PROMPT)

        try:
            # Persisted rows name the children and the grandchild.
            def _rows() -> dict[str, str] | None:
                parent_task_id = self._parent_task_id()
                if parent_task_id is None:
                    return None
                children = self._child_rows(parent_task_id)
                if len(children) != 2:
                    return None
                nesting = [
                    c for c in children if NEST_MARK in str(c.get("task", ""))
                ]
                grand = (
                    self._child_rows(str(nesting[0]["task_id"]))
                    if nesting else []
                )
                if len(grand) != 1:
                    return None
                return {
                    "parent": parent_task_id,
                    "plain": str(next(
                        c["task_id"] for c in children
                        if NEST_MARK not in str(c.get("task", ""))
                    )),
                    "nesting": str(nesting[0]["task_id"]),
                    "grand": str(grand[0]["task_id"]),
                }

            ids = self.wait_for(_rows, what="child and grandchild rows")
            plain_tab = f"{a.tab_id}__sub_{ids['plain']}"
            nesting_tab = f"{a.tab_id}__sub_{ids['nesting']}"
            grand_tab = f"{nesting_tab}__sub_{ids['grand']}"

            # 1. Live spawn: both connected clients are told to open all
            #    three running sub-agents, each under the right parent.
            for client in (a, b):
                for task_id, parent in (
                    (ids["plain"], a.tab_id),
                    (ids["nesting"], a.tab_id),
                    (ids["grand"], nesting_tab),
                ):
                    ev = self.wait_event(client, "new_tab", task_id=task_id)
                    self.assertEqual(ev["parent_tab_id"], parent, (client.name, ev))
                    self._watch_child(client, task_id, parent)

            # 2. A client connecting MID-RUN gets the same running
            #    sub-agents from its ready replay.
            c = self.open_client("c")
            c.ready()
            self.wait_for(
                lambda: self._running_announced(c)
                == {plain_tab, nesting_tab, grand_tab},
                what=f"c: running sub-agent tabs on replay, got "
                f"{self._running_announced(c)}",
            )
            # Every announcement of the grandchild -- the live spawn and
            # each replay -- names the NESTING child as its parent, never
            # the top-level chat tab.
            for client in (a, b, c):
                for ev in list(client.events):
                    if ev.get("task_id") == ids["grand"] and ev.get("type") in (
                        "new_tab", "openSubagentTab",
                    ):
                        self.assertEqual(
                            ev["parent_tab_id"], nesting_tab, (client.name, ev),
                        )

            # 2b. The user closes a running child's tab on one client: the
            #     close mirrors to every client, the other running
            #     sub-agents stay, and a client connecting afterwards does
            #     not get the closed tab back from its replay.
            b.send({"type": "closeTab", "tabId": plain_tab})
            for client in (a, b, c):
                self.wait_event(client, "closeSubagentTab", tab_id=plain_tab)
            d = self.open_client("d")
            d.ready()
            self.wait_for(
                lambda: self._running_announced(d) == {nesting_tab, grand_tab},
                what=f"d: replay without the user-closed tab, got "
                f"{self._running_announced(d)}",
            )
            time.sleep(0.5)
            self.assertEqual(d.of_type("openSubagentTab", tab_id=plain_tab), [])
        finally:
            leaf_release.set()

        # 3. Once the sub-agents are over, every client is told so.
        for client in (a, b, c, d):
            for tab in (nesting_tab, grand_tab):
                self.wait_event(client, "subagentDone", tab_id=tab)
        self.wait_for(
            lambda: a.of_type("status", running=False, tabId=a.tab_id),
            what="parent task to finish",
        )

        # 4. A client connecting after completion is never told to open a
        #    finished sub-agent as running.
        late = self.open_client("late")
        late.ready()
        self._open_deferred_tab(late, a.tab_id)
        time.sleep(0.5)
        self.assertEqual(
            self._running_announced(late), set(), late.of_type("openSubagentTab"),
        )

    def test_children_follow_a_chat_closed_and_reopened_while_running(
        self,
    ) -> None:
        wave1_release = threading.Event()
        wave2_release = threading.Event()
        self._install_two_wave_stub(wave1_release, wave2_release)

        a = self.open_client("a")
        a.ready()
        b = self.open_client("b")
        b.ready()
        self.wait_event(b, "tabs_state")
        self._submit(a, PARENT_PROMPT)

        try:
            def _wave1() -> dict[str, str] | None:
                parent_task_id = self._parent_task_id()
                if parent_task_id is None:
                    return None
                children = self._child_rows(parent_task_id)
                if len(children) != 1:
                    return None
                return {"parent": parent_task_id, "child": str(children[0]["task_id"])}

            ids = self.wait_for(_wave1, what="first-wave child row")
            for client in (a, b):
                self.wait_event(client, "new_tab", task_id=ids["child"])
                self._watch_child(client, ids["child"], a.tab_id)

            # The user closes the RUNNING chat's tab: the registry drops
            # it for every client (each page then closes the child's tab
            # along with its parent).
            a.send({"type": "closeTab", "tabId": a.tab_id})
            for client in (a, b):
                self.wait_for(
                    partial(self._registry_dropped, client, a.tab_id),
                    what=f"{client.name}: registry without the closed chat",
                )

            # The chat is reopened from history under a NEW tab id (what
            # a History-panel click does): the running child is announced
            # under the new tab on every client.
            parent_row = next(
                r for r in _persistence._load_history()
                if str(r["id"]) == ids["parent"]
            )
            new_tab = f"reopened-{uuid.uuid4().hex[:8]}"
            b.send({
                "type": "resumeSession",
                "id": parent_row["chat_id"],
                "taskId": ids["parent"],
                "tabId": new_tab,
            })
            new_child_tab = f"{new_tab}__sub_{ids['child']}"
            for client in (a, b):
                ev = self.wait_event(client, "openSubagentTab", tab_id=new_child_tab)
                self.assertEqual(ev["parent_tab_id"], new_tab, (client.name, ev))
                self.assertFalse(ev.get("isDone"), (client.name, ev))
        finally:
            wave1_release.set()

        try:
            # A child spawned AFTER the reopen is announced under the new
            # tab too.
            def _wave2() -> str | None:
                children = self._child_rows(ids["parent"])
                second = [
                    c for c in children if WAVE2_MARK in str(c.get("task", ""))
                ]
                return str(second[0]["task_id"]) if second else None

            second_id = self.wait_for(_wave2, what="second-wave child row")
            for client in (a, b):
                ev = self.wait_event(client, "new_tab", task_id=second_id)
                self.assertEqual(ev["parent_tab_id"], new_tab, (client.name, ev))
                self._watch_child(client, second_id, new_tab)
        finally:
            wave2_release.set()

        for client in (a, b):
            self.wait_event(
                client, "subagentDone", tab_id=f"{new_tab}__sub_{second_id}",
            )
