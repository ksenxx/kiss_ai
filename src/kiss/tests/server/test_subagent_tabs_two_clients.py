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
   afterwards, whose replay repeats the close and never says the
   running sub-agent is done; the other running sub-agents keep their
   tabs.
3. After the children finish: every client gets ``subagentDone`` for the
   sub-agents it still shows.
4. A client connecting AFTER completion is never told to open a finished
   sub-agent's tab as running.
5. (second test) The user closes the RUNNING chat's tab -- the registry
   drops it for every client, whose pages close the child's tab with
   it -- and reopens the chat under a new tab id: the running child, and
   a child spawned after the reopen, are announced under the new tab to
   every client.
6. (third test) A surface is away while the user closes a running
   child's tab and reconnects; the user then reopens the child.  The
   frames that surface got are played into the real chat page
   (``media/main.js`` under jsdom): it shows the child's tab again and
   never asks the daemon to close it.
7. (fourth test) The close a page sends when a child finishes reaches
   the daemon only after a user reopened the child's tab on another
   surface, and another page's is sent twice: the reopened tab stays
   open everywhere.  A page that closes the reopened tab is obeyed.
8. (fifth test) A reopened grandchild's tab closes with its parent's
   when the parent finishes, and the daemon holds it open no longer.
9. (sixth test) Another chat resumed into a running child's tab id
   keeps its tab when the child ends.
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
from kiss.tests.server.two_client_harness import Client, TwoClientHarness, page_after

PARENT_PROMPT = "fan out: spawn two sub-agents xq9"
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
            # What the real ``run`` records: the dispatcher reads it to see
            # that the task asks for sub-agents (same-family children).
            self_agent.task_description = str(kwargs.get("prompt_template", ""))
            plan(self_agent, self_agent.task_description)
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
                    f"{CHILD_MARK} {NEST_MARK} parent of a grandchild sub-agent",
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

    def _two_pages(self) -> tuple[Client, Client]:
        """Connect two pages; the second has the registry's first snapshot."""
        a = self.open_client("a")
        a.ready()
        b = self.open_client("b")
        b.ready()
        self.wait_event(b, "tabs_state")
        return a, b

    def _children(self, parent_task_id: str | None, count: int) -> list[str] | None:
        """The task ids of *parent_task_id*'s sub-agents once it has *count*."""
        rows = self._child_rows(parent_task_id) if parent_task_id else []
        return [str(r["task_id"]) for r in rows] if len(rows) == count else None

    def _watch_spawn(self, clients: tuple[Client, ...], task_id: str, parent_tab: str) -> str:
        """Have every page open the spawned *task_id*'s tab; return the tab id."""
        sub_tab = f"{parent_tab}__sub_{task_id}"
        for client in clients:
            self.wait_event(client, "new_tab", task_id=task_id)
            self._watch_child(client, task_id, parent_tab)
            self.wait_event(client, "openSubagentTab", tab_id=sub_tab)
        return sub_tab

    def _settled(self, client: Client) -> None:
        """Wait until the daemon handled every command *client* sent so far.

        A connection's commands are handled in the order sent, so the
        answer to one sent now follows all of them.
        """
        before = len(client.of_type("inputHistory"))
        client.send({"type": "getInputHistory"})
        self.wait_for(
            lambda: len(client.of_type("inputHistory")) > before,
            what=f"{client.name}: commands handled",
        )

    @staticmethod
    def _registry_lists(client: Client, tab_id: str) -> bool:
        """Whether the last ``tabs_state`` *client* got lists *tab_id*."""
        snapshots = client.of_type("tabs_state")
        return bool(snapshots) and any(
            t.get("tabId") == tab_id for t in snapshots[-1].get("tabs", [])
        )

    @staticmethod
    def _run_ended(client: Client, tab_id: str) -> bool:
        """Whether *client* saw the run in *tab_id* start and then end."""
        running = [e.get("running") for e in client.of_type("status", tabId=tab_id)]
        return True in running and running[-1] is False

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
            # d's replay says the tab is closed (a surface that was away
            # during the close still shows it), never that the running
            # sub-agent finished: a page keeps a finished sub-agent's
            # tab closed for good, so it would not show this one again
            # when a user reopens it.  The others were told once.
            self.wait_event(d, "closeSubagentTab", tab_id=plain_tab)
            time.sleep(0.5)
            self.assertEqual(d.of_type("openSubagentTab", tab_id=plain_tab), [])
            self.assertEqual(d.of_type("subagentDone", tab_id=plain_tab), [])
            for client in (a, b, c):
                closes = client.of_type("closeSubagentTab", tab_id=plain_tab)
                self.assertEqual(len(closes), 1, (client.name, closes))
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

    def test_a_page_away_during_a_close_shows_the_reopened_child(self) -> None:
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
            def _child() -> str | None:
                parent_task_id = self._parent_task_id()
                children = self._child_rows(parent_task_id) if parent_task_id else []
                return str(children[0]["task_id"]) if children else None

            child = self.wait_for(_child, what="child row")
            for client in (a, b):
                self.wait_event(client, "new_tab", task_id=child)
                child_tab = self._watch_child(client, child, a.tab_id)
            self.wait_event(b, "openSubagentTab", tab_id=child_tab)

            # b's surface goes away with the child's tab open; the user
            # closes that tab on a; b's surface comes back showing the
            # chat (a new connection, its page as it was).
            b.close()
            a.send({"type": "closeTab", "tabId": child_tab})
            self.wait_event(a, "closeSubagentTab", tab_id=child_tab)
            back = self.open_client("b-back")
            back.tab_id = a.tab_id
            back.ready()
            self.wait_for(
                lambda: [e for e in list(back.events) if e.get("tab_id") == child_tab],
                what="b-back: the replay's word on the closed child",
            )

            # The user expands the fan-out panel on a: the child's tab
            # is asked for again and opens on every surface.
            a.send({
                "type": "resumeSession", "taskId": child, "tabId": child_tab,
                "parentTabId": a.tab_id, "reopen": True,
            })
            self.wait_event(back, "openSubagentTab", tab_id=child_tab, reopened=True)

            self.assertEqual(
                page_after(a.tab_id, list(b.events) + list(back.events)),
                {"shown": [child_tab], "closed": []},
            )
        finally:
            wave1_release.set()
            wave2_release.set()
        self.wait_for(
            lambda: a.of_type("status", running=False, tabId=a.tab_id),
            what="parent task to finish",
        )

    def test_a_late_echo_of_a_finished_close_spares_the_reopened_tab(self) -> None:
        wave1_release = threading.Event()
        wave2_release = threading.Event()
        self._install_two_wave_stub(wave1_release, wave2_release)
        a, b = self._two_pages()
        self._submit(a, PARENT_PROMPT)
        try:
            (child,) = self.wait_for(
                lambda: self._children(self._parent_task_id(), 1), what="child row",
            )
            child_tab = self._watch_spawn((a, b), child, a.tab_id)

            # The child finishes: each page closes its tab and tells the
            # daemon so.
            wave1_release.set()
            echoes = []
            for client in (a, b):
                self.wait_event(client, "subagentDone", tab_id=child_tab)
                (echo,) = page_after(a.tab_id, list(client.events))["closed"]
                self.assertEqual(echo["tabId"], child_tab, client.name)
                echoes.append(echo)
            echo_a, echo_b = echoes

            # a's close arrives, and a user expands the fan-out panel on a.
            a.send(echo_a)
            self.wait_event(b, "closeSubagentTab", tab_id=child_tab)
            a.send({
                "type": "resumeSession", "taskId": child, "tabId": child_tab,
                "parentTabId": a.tab_id, "reopen": True,
            })
            reopened = self.wait_event(b, "openSubagentTab", tab_id=child_tab, reopened=True)

            # b's close arrives only now, and a's again (the transport
            # re-sends what a dropped socket left unconfirmed).
            b.send(echo_b)
            a.send(echo_a)
            for client in (a, b):
                self._settled(client)
            for client in (a, b):
                self.assertEqual(
                    len(client.of_type("closeSubagentTab", tab_id=child_tab)), 1, client.name,
                )
                self.assertIn(
                    child_tab, page_after(a.tab_id, list(client.events))["shown"], client.name,
                )
            c = self.open_client("c")
            c.tab_id = a.tab_id
            c.ready()
            self.wait_event(c, "openSubagentTab", tab_id=child_tab, reopened=True)

            # The frames a page gets when the reopening lands between the
            # child's release and its ``subagentDone`` (two statements
            # apart in ``_run_task``, so no test can hold the daemon
            # there): the tab it then closes is the reopened one, and the
            # daemon takes its word.
            done = next(
                i for i, e in enumerate(b.events)
                if e.get("type") == "subagentDone" and e.get("tab_id") == child_tab
            )
            (echo,) = page_after(a.tab_id, [*b.events[:done], reopened, b.events[done]])["closed"]
            self.assertEqual(
                (echo["tabId"], echo["incarnation"]), (child_tab, reopened["incarnation"]),
            )
            b.send(echo)
            self.wait_for(
                lambda: len(a.of_type("closeSubagentTab", tab_id=child_tab)) == 2,
                what="a: the reopened tab closed",
            )
            d = self.open_client("d")
            d.tab_id = a.tab_id
            d.ready()
            self.wait_event(d, "openSubagentTab", tab_id=child_tab, closedHistory=True)
        finally:
            wave1_release.set()
            wave2_release.set()
        self.wait_for(partial(self._run_ended, a, a.tab_id), what="parent task to finish")

    def test_a_reopened_grandchild_closes_with_its_finished_parent(self) -> None:
        grand_release = threading.Event()
        nest_release = threading.Event()

        def plan(agent: Any, prompt: str) -> None:
            if GRANDCHILD_MARK in prompt:
                grand_release.wait(timeout=60)
            elif NEST_MARK in prompt:
                _fan_out(agent, [f"{GRANDCHILD_MARK} leaf"])
                nest_release.wait(timeout=60)
            elif PARENT_PROMPT in prompt:
                _fan_out(agent, [f"{CHILD_MARK} {NEST_MARK} parent of a grandchild sub-agent"])

        self._install_plan(plan)
        a, b = self._two_pages()
        self._submit(a, PARENT_PROMPT)
        try:
            (nesting,) = self.wait_for(
                lambda: self._children(self._parent_task_id(), 1), what="child row",
            )
            (grand,) = self.wait_for(lambda: self._children(nesting, 1), what="grandchild row")
            nesting_tab = self._watch_spawn((a, b), nesting, a.tab_id)
            grand_tab = self._watch_spawn((a, b), grand, nesting_tab)

            # The grandchild finishes and its tab closes everywhere.
            grand_release.set()
            for client in (a, b):
                self.wait_event(client, "subagentDone", tab_id=grand_tab)
            (echo,) = page_after(a.tab_id, list(a.events))["closed"]
            a.send(echo)
            self.wait_event(b, "closeSubagentTab", tab_id=grand_tab)

            # Users on both surfaces expand its fan-out panel at once.
            for client in (a, b):
                client.send({
                    "type": "resumeSession", "taskId": grand, "tabId": grand_tab,
                    "parentTabId": nesting_tab, "reopen": True,
                })
            self.wait_for(
                lambda: len(a.of_type("openSubagentTab", tab_id=grand_tab, reopened=True)) == 2,
                what="a: both reopenings",
            )
            first, second = a.of_type("openSubagentTab", tab_id=grand_tab, reopened=True)
            self.assertLess(first["incarnation"], second["incarnation"])

            # Its parent finishes: a page closes both tabs, the
            # grandchild's as it was last announced.
            nest_release.set()
            for client in (a, b):
                self.wait_event(client, "subagentDone", tab_id=nesting_tab)
            *_, nesting_close, grand_close = page_after(a.tab_id, list(a.events))["closed"]
            self.assertEqual(nesting_close["tabId"], nesting_tab)
            self.assertEqual(
                (grand_close["tabId"], grand_close["incarnation"]),
                (grand_tab, second["incarnation"]),
            )
            a.send(nesting_close)
            a.send(grand_close)
            self.wait_for(
                lambda: len(b.of_type("closeSubagentTab", tab_id=grand_tab)) == 2,
                what="b: the reopened grandchild closed",
            )
            # The daemon holds it reopened no longer: a surface connecting
            # now is told it is over.
            c = self.open_client("c")
            c.tab_id = a.tab_id
            c.ready()
            self.wait_event(c, "subagentDone", tab_id=grand_tab)
            self.assertEqual(c.of_type("openSubagentTab", tab_id=grand_tab), [])
        finally:
            grand_release.set()
            nest_release.set()
        self.wait_for(partial(self._run_ended, a, a.tab_id), what="parent task to finish")

    def test_a_chat_resumed_into_a_child_tab_survives_the_childs_end(self) -> None:
        wave1_release = threading.Event()
        wave2_release = threading.Event()
        self._install_two_wave_stub(wave1_release, wave2_release)
        a, b = self._two_pages()

        # Another chat, whose tab the user then closes.
        other_prompt = "say hello zq4"
        self._submit(b, other_prompt)
        self.wait_for(partial(self._run_ended, b, b.tab_id), what="the other chat's task to finish")
        other = next(
            r for r in _persistence._load_history() if other_prompt in str(r.get("task", ""))
        )
        b.send({"type": "closeTab", "tabId": b.tab_id})
        self.wait_for(
            lambda: not self._registry_lists(a, b.tab_id), what="the other chat's tab closed",
        )

        self._submit(a, PARENT_PROMPT)
        try:
            (child,) = self.wait_for(
                lambda: self._children(self._parent_task_id(), 1), what="child row",
            )
            child_tab = self._watch_spawn((a, b), child, a.tab_id)
            # A client resumes the other chat into the running child's tab.
            b.send({
                "type": "resumeSession", "id": other["chat_id"],
                "taskId": str(other["id"]), "tabId": child_tab,
            })
            for client in (a, b):
                self.wait_event(client, "task_events", tabId=child_tab, chat_id=other["chat_id"])
                self.wait_for(
                    partial(self._registry_lists, client, child_tab),
                    what=f"{client.name}: the tab in the registry",
                )

            # The child ends: the parent spawns the next one.
            wave1_release.set()
            _, second = self.wait_for(
                lambda: self._children(self._parent_task_id(), 2), what="second child row",
            )
            for client in (a, b):
                self.wait_event(client, "new_tab", task_id=second)
            time.sleep(0.5)
            for client in (a, b):
                self.assertEqual(client.of_type("subagentDone", tab_id=child_tab), [], client.name)
                page = page_after(a.tab_id, list(client.events))
                self.assertEqual(page["closed"], [], client.name)
                self.assertIn(child_tab, page["shown"], client.name)

            # A page that closed the sub-agent's tab before it heard of
            # the other chat: its close is not for that chat's tab, and
            # the snapshot that still lists the tab answers the close.
            a.send({"type": "closeTab", "tabId": child_tab, "incarnation": 0, "ack": "kept"})
            self.wait_for(
                lambda: [e for e in a.of_type("tabs_state") if "kept" in e.get("acks", [])],
                what="a: the kept tab's snapshot",
            )
            for client in (a, b):
                self.assertTrue(self._registry_lists(client, child_tab), client.name)
        finally:
            wave1_release.set()
            wave2_release.set()
        self.wait_for(partial(self._run_ended, a, a.tab_id), what="parent task to finish")
