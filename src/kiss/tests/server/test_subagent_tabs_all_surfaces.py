# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a running sub-agent's tab is open on EVERY surface, and closed
on every surface once the sub-agent is over.

The invariant (``~/.kiss/SORCAR.md``)::

    Across all surfaces of KISS Sorcar you must show the same tabs.
    If a task is running, the corresponding tab must be open unless
    the user has closed the tab.  If a subagent is running, its tab
    must be open unless the user has closed the tab.  A subagent tab
    must be closed as soon as the subagent finishes its task.

"Surfaces" are the real chat webviews (``media/main.js``): the VS Code
sidebar view, a VS Code editor-tab panel and browser tabs of the
remote web app.  This test runs the REAL daemon on a Unix socket and
REAL webviews under jsdom (``test/multiSurfaceBridge.js``), one daemon
connection per webview, exactly like production; only the LLM loop is
scripted (the parent fans out through ``_run_tasks_parallel``, one
child nests a grandchild, and the leaves block until released).

Checked, on every surface, from the rendered tab bar:

1. While the leaves run: the two children AND the nested grandchild
   have a sub-agent tab (spinner, not done) on the surfaces that were
   connected when they spawned (live ``new_tab``) — sidebar and remote.
2. A surface connecting MID-RUN — a second remote browser tab and an
   editor-tab panel pinned to the parent chat — gets the same running
   sub-agent tabs from its ``ready`` replay; every announcement of the
   grandchild names the nesting child's tab as its parent.
   A running child's tab the user closes by hand on one surface
   closes on all of them and stays closed for a surface connecting
   afterwards; the other running sub-agents keep their tabs.
3. After the children finish: every surface has zero sub-agent tabs.
4. A surface connecting AFTER completion never shows a finished
   sub-agent tab.
5. (second test) The user closes a RUNNING chat's tab — its running
   child goes with it everywhere — and reopens the chat from history
   under a new tab id: the running child, and a child spawned after
   the reopen, show under the new tab on every surface.

Three daemon defects this test caught (all fixed):

* ``SorcarAgent._subagent_parent_tab_id`` looked the child's webview
  tab up in the printer's viewer registry, which is empty until a
  client's ``resumeSession`` arrives, so a sub-agent fanning out right
  away parented its children under its synthetic ``task-…__sub_N`` id
  and no surface opened their tabs.  It now derives the deterministic
  ``{parent_tab_id}__sub_{task_id}`` id.
* ``_open_persisted_subagent_tabs`` announced only direct children on
  a replay, so a surface connecting mid-run never saw a running
  grandchild.  It now recurses into running children.
* ``_resolve_parent_tab_id_for_sub`` skipped sub-agent states, so a
  grandchild's own replay named the top-level chat tab as its parent
  and re-parented it there on every surface.  The tab-id suffix now
  settles the parent first.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
import unittest
import uuid
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as _persistence
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_run_agent_subagent_tab import DaemonUdsHarness

pytestmark = requires_unix_sockets

_VSCODE_DIR = Path(__file__).resolve().parents[2] / "agents" / "vscode"
_BRIDGE = _VSCODE_DIR / "test" / "multiSurfaceBridge.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"

PARENT_PROMPT = "fan out: spawn two children xq9"
CHILD_MARK = "CHILDTASK"
NEST_MARK = "NESTING"
GRANDCHILD_MARK = "GRANDCHILD"
WAVE2_MARK = "SECONDWAVE"


class SurfaceBridge:
    """Drive ``multiSurfaceBridge.js``: real webviews over the daemon UDS."""

    def __init__(self, sock_path: str) -> None:
        self.proc = subprocess.Popen(
            ["node", str(_BRIDGE), sock_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.replies: queue.Queue[dict[str, Any]] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self.replies.put(json.loads(line))

    def call(self, op: str, **kw: Any) -> dict[str, Any]:
        """Send one command and return the bridge's reply."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"op": op, **kw}) + "\n")
        self.proc.stdin.flush()
        return self.replies.get(timeout=30)

    def tabs(self, name: str) -> list[dict[str, Any]]:
        """Rendered tab bar of surface *name*; fails on webview errors."""
        reply = self.call("tabs", name=name)
        assert reply["op"] == "tabs", reply
        assert not reply["errors"], (
            f"webview {name!r} raised: {reply['errors']!r}"
        )
        return list(reply["tabs"])

    def sub_tabs(self, name: str) -> list[dict[str, Any]]:
        """Sub-agent tabs rendered on surface *name*."""
        return [t for t in self.tabs(name) if t["isSubagentTab"]]

    def open(self, name: str, body_attrs: str = "") -> None:
        """Open a surface and wait for its first registry snapshot."""
        reply = self.call("open", name=name, bodyAttrs=body_attrs, state=None)
        assert reply["op"] == "opened", reply
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if "tabs_state" in self.event_types(name):
                return
            time.sleep(0.05)
        raise AssertionError(f"surface {name!r} never got tabs_state")

    def events(self, name: str) -> list[Any]:
        """Events surface *name* received: lifecycle events whole, else type."""
        return list(self.call("events", name=name)["events"])

    def event_types(self, name: str) -> list[str]:
        """Types of every event surface *name* received, in order."""
        return [
            e["type"] if isinstance(e, dict) else str(e)
            for e in self.events(name)
        ]

    def quit(self) -> None:
        """Stop the bridge process."""
        try:
            self.call("quit")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _fan_out(agent: Any, tasks: list[str]) -> None:
    """Call ``run_parallel`` the way the model does: the ``tool_call`` /
    ``tool_result`` events frame the fan-out in the transcript (the
    webview's fan-out panel, which keeps finished children's tabs
    closed on later replays, is built from them)."""
    tool_input = {"tasks": json.dumps(tasks)}
    agent.printer.print(
        "run_parallel", type="tool_call", tool_input=tool_input,
        call_id=uuid.uuid4().hex,
    )
    results = agent._run_tasks_parallel(tasks)
    agent.printer.print(
        json.dumps(results), type="tool_result", tool_name="run_parallel",
        tool_input=tool_input, is_error=False, interrupted=False,
    )


class SubagentTabsAllSurfacesTest(DaemonUdsHarness):
    """Open-everywhere while running, closed-everywhere when done."""

    def setUp(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest("jsdom is not installed under agents/vscode")
        super().setUp()
        self.bridge = SurfaceBridge(self.sock_path)

    def tearDown(self) -> None:
        self.bridge.quit()
        super().tearDown()

    def _install_plan(self, plan: Any) -> None:
        """Replace the LLM loop: *plan(agent, name)* does the run's work.

        The level comes from ``self.name`` (``Parallel-{task}``), never
        from the prompt: sub-agent prompts embed the chat's earlier
        tasks.  Any other agent the daemon runs alongside (e.g. the
        "Task update" side channel) is a plain leaf that returns at
        once.
        """

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
            plan(self_agent, str(getattr(self_agent, "name", "")))
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
        """Parent fans out two children; one nests a grandchild.

        Leaves (the plain child and the grandchild) block on
        *leaf_release* so the mid-run state can be inspected.
        """

        def plan(agent: Any, name: str) -> None:
            if GRANDCHILD_MARK in name:
                leaf_release.wait(timeout=60)
            elif NEST_MARK in name:
                # Fan out IMMEDIATELY: no client can have subscribed to
                # this sub-agent's tab yet, the worst case for parenting
                # the grandchild under a tab every surface knows.
                _fan_out(agent, [f"{GRANDCHILD_MARK} leaf"])
            elif CHILD_MARK in name:
                leaf_release.wait(timeout=60)
            elif getattr(agent, "_subagent_info", None) is None:
                _fan_out(agent, [
                    f"{CHILD_MARK} plain leaf",
                    f"{CHILD_MARK} {NEST_MARK} parent of a grandchild",
                ])

        self._install_plan(plan)

    def _install_two_wave_stub(
        self, wave1_release: threading.Event, wave2_release: threading.Event,
    ) -> None:
        """Parent fans out one child, then — once it is over — another.

        Each child blocks on its wave's release event.
        """

        def plan(agent: Any, name: str) -> None:
            if WAVE2_MARK in name:
                wave2_release.wait(timeout=60)
            elif CHILD_MARK in name:
                wave1_release.wait(timeout=60)
            elif getattr(agent, "_subagent_info", None) is None:
                _fan_out(agent, [f"{CHILD_MARK} first wave"])
                _fan_out(agent, [f"{CHILD_MARK} {WAVE2_MARK} second wave"])

        self._install_plan(plan)

    def _child_rows(self, parent_task_id: str) -> list[dict[str, Any]]:
        return _persistence._load_subagent_rows_by_parent_task_id(
            parent_task_id,
        )

    def _parent_task_id(self) -> str | None:
        parents = [
            r for r in _persistence._load_history()
            if PARENT_PROMPT in str(r.get("task", ""))
        ]
        return str(parents[0]["id"]) if parents else None

    def _settled_sub_tabs(self, name: str) -> set[str]:
        """Sub-agent tab ids on *name* once its replay has landed."""
        self._wait_for(
            lambda: "task_events" in self.bridge.event_types(name),
            what=f"{name} replay",
        )
        time.sleep(0.5)
        return {t["id"] for t in self.bridge.sub_tabs(name)}

    def _wait_sub_tabs(
        self, name: str, expected_ids: set[str], what: str,
    ) -> list[dict[str, Any]]:
        # Wrapped in a dict: an empty match (no sub-agent tabs) must
        # still count as found.
        def _probe() -> dict[str, list[dict[str, Any]]] | None:
            subs = self.bridge.sub_tabs(name)
            if {t["id"] for t in subs} != expected_ids:
                return None
            return {"subs": subs}

        try:
            found: dict[str, list[dict[str, Any]]] = self._wait_for(
                _probe, timeout=20, what=what,
            )
            return found["subs"]
        except AssertionError:
            got = [t["id"] for t in self.bridge.sub_tabs(name)]
            events = self.bridge.events(name)
            raise AssertionError(
                f"{what}: surface {name!r} shows sub-agent tabs {got!r}, "
                f"expected {sorted(expected_ids)!r}; events={events!r}",
            ) from None

    def test_subagent_tabs_open_and_close_on_every_surface(self) -> None:
        leaf_release = threading.Event()
        self._install_fanout_stub(leaf_release)

        # Two surfaces connected before the run: a sidebar view and a
        # remote browser tab.
        self.bridge.open("sidebar", "")
        self.bridge.open("remote", ' class="remote-chat"')

        # The user submits the prompt from the sidebar's placeholder
        # tab; the registry then mirrors that tab to the remote page.
        parent_tab_id = self.bridge.tabs("sidebar")[0]["id"]
        self.bridge.call("submit", name="sidebar", text=PARENT_PROMPT)
        self._wait_for(
            lambda: any(
                t["id"] == parent_tab_id for t in self.bridge.tabs("remote")
            ),
            what="remote page to mirror the parent tab",
        )

        try:
            # Persisted rows name the children and the grandchild.
            def _rows() -> dict[str, str] | None:
                parents = [
                    r for r in _persistence._load_history()
                    if PARENT_PROMPT in str(r.get("task", ""))
                ]
                if not parents:
                    return None
                parent_task_id = str(parents[0]["id"])
                children = _persistence._load_subagent_rows_by_parent_task_id(
                    parent_task_id,
                )
                if len(children) != 2:
                    return None
                nesting = [
                    c for c in children if NEST_MARK in str(c.get("task", ""))
                ]
                grand = (
                    _persistence._load_subagent_rows_by_parent_task_id(
                        str(nesting[0]["task_id"]),
                    ) if nesting else []
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

            ids = self._wait_for(_rows, what="child and grandchild rows")
            plain_tab = f"{parent_tab_id}__sub_{ids['plain']}"
            nesting_tab = f"{parent_tab_id}__sub_{ids['nesting']}"
            grand_tab = f"{nesting_tab}__sub_{ids['grand']}"
            running = {plain_tab, nesting_tab, grand_tab}

            # 1. Live spawn: both pre-connected surfaces show all three
            #    running sub-agents.
            for name in ("sidebar", "remote"):
                subs = self._wait_sub_tabs(
                    name, running, "running sub-agent tabs (live spawn)",
                )
                for t in subs:
                    assert not t["subagentDone"], (name, t)

            # 2. Surfaces connecting MID-RUN get the same tabs from
            #    their ready replay: a second browser tab, and an
            #    editor-tab panel pinned to the parent chat.
            self.bridge.open("remote2", ' class="remote-chat"')
            self.bridge.open(
                "editor",
                ' class="editor-tab-mode"'
                f' data-kiss-tab-id="{parent_tab_id}"'
                ' data-kiss-in-registry="1"',
            )
            for name in ("remote2", "editor"):
                subs = self._wait_sub_tabs(
                    name, running, "running sub-agent tabs (mid-run connect)",
                )
                for t in subs:
                    assert not t["subagentDone"], (name, t)

            # Every announcement of the grandchild — the live spawn and
            # each replay — names the NESTING child as its parent, never
            # the top-level chat tab (which would re-parent it there).
            for name in ("sidebar", "remote", "remote2", "editor"):
                for ev in self.bridge.events(name):
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("task_id") == ids["grand"] and ev["type"] in (
                        "new_tab", "openSubagentTab",
                    ):
                        assert ev["parent_tab_id"] == nesting_tab, (name, ev)

            # 2b. The user closes a running child's tab on one surface:
            #     the close mirrors to every surface, the other running
            #     sub-agents stay, and a surface connecting afterwards
            #     does not get the closed tab back from its replay.
            closed = self.bridge.call("closeTab", name="remote", tabId=plain_tab)
            assert closed["found"], closed
            kept = {nesting_tab, grand_tab}
            for name in ("sidebar", "remote", "remote2", "editor"):
                self._wait_sub_tabs(
                    name, kept, "user-closed running sub-agent tab gone",
                )
            self.bridge.open("remote3", ' class="remote-chat"')
            assert self._settled_sub_tabs("remote3") == kept, (
                self.bridge.events("remote3"),
            )
        finally:
            leaf_release.set()

        # 3. Once the sub-agents are over, every surface closes their
        #    tabs — and the parent tab itself stays.
        for name in ("sidebar", "remote", "remote2", "editor", "remote3"):
            self._wait_sub_tabs(name, set(), "sub-agent tabs closed on done")
            assert any(
                t["id"] == parent_tab_id for t in self.bridge.tabs(name)
            ), f"parent tab vanished on {name!r}"

        # 4. A surface connecting after completion never shows a
        #    finished sub-agent's tab.
        self._wait_for(
            lambda: not any(
                t["running"] for t in self.bridge.tabs("sidebar")
                if t["id"] == parent_tab_id
            ),
            what="parent task to finish",
        )
        self.bridge.open("late", ' class="remote-chat"')
        self._wait_for(
            lambda: "task_events" in self.bridge.event_types("late"),
            what="late surface replay",
        )
        time.sleep(0.5)
        assert self.bridge.sub_tabs("late") == [], (
            self.bridge.sub_tabs("late"),
            self.bridge.events("late"),
        )

    def test_children_follow_a_chat_closed_and_reopened_while_running(
        self,
    ) -> None:
        """The user closes a running chat's tab (its running child goes
        with it, everywhere), reopens the chat from history under a NEW
        tab id, and every surface shows the running child under the new
        tab — including a child spawned AFTER the reopen, which must not
        be dropped for naming the closed tab as its parent."""
        wave1_release = threading.Event()
        wave2_release = threading.Event()
        self._install_two_wave_stub(wave1_release, wave2_release)
        self.bridge.open("sidebar", "")
        self.bridge.open("remote", ' class="remote-chat"')
        old_tab = self.bridge.tabs("sidebar")[0]["id"]
        self.bridge.call("submit", name="sidebar", text=PARENT_PROMPT)

        def _wave1() -> dict[str, str] | None:
            parent = self._parent_task_id()
            rows = self._child_rows(parent) if parent else []
            if parent is None or len(rows) != 1:
                return None
            return {"parent": parent, "child": str(rows[0]["task_id"])}

        try:
            ids = self._wait_for(_wave1, what="first-wave child row")
            for name in ("sidebar", "remote"):
                self._wait_sub_tabs(
                    name, {f"{old_tab}__sub_{ids['child']}"},
                    "first-wave child (live spawn)",
                )

            # The user closes the running chat's tab: the tab and its
            # running child leave every surface.
            closed = self.bridge.call("closeTab", name="sidebar", tabId=old_tab)
            assert closed["found"], closed
            for name in ("sidebar", "remote"):
                self._wait_for(
                    lambda: not any(
                        t["id"] == old_tab or t["isSubagentTab"]
                        for t in self.bridge.tabs(name)
                    ),
                    what=f"closed chat and its child gone on {name}",
                )

            # Reopen the still-running chat from the history list (the
            # drawer loads its rows when the menu button opens it).
            opened = self.bridge.call("click", name="sidebar", selector="#menu-btn")
            assert opened["found"], opened

            def _history_click() -> bool:
                return bool(self.bridge.call(
                    "click", name="sidebar",
                    selector="#history-list .sidebar-item",
                    text=PARENT_PROMPT,
                )["found"])

            self._wait_for(_history_click, what="history row for the chat")

            def _new_parent() -> str | None:
                for t in self.bridge.tabs("sidebar"):
                    if t["running"] and not t["isSubagentTab"]:
                        return str(t["id"])
                return None

            new_tab = self._wait_for(_new_parent, what="reopened chat tab")
            assert new_tab != old_tab
            for name in ("sidebar", "remote"):
                self._wait_sub_tabs(
                    name, {f"{new_tab}__sub_{ids['child']}"},
                    "running child under the reopened tab",
                )
        finally:
            wave1_release.set()

        def _wave2() -> str | None:
            rows = self._child_rows(ids["parent"])
            second = [r for r in rows if WAVE2_MARK in str(r.get("task", ""))]
            return str(second[0]["task_id"]) if second else None

        try:
            child2 = self._wait_for(_wave2, what="second-wave child row")
            # Spawned after the reopen: its tab hangs off the NEW tab on
            # every surface (the daemon must not name the closed tab).
            for name in ("sidebar", "remote"):
                self._wait_sub_tabs(
                    name, {f"{new_tab}__sub_{child2}"},
                    "second-wave child under the reopened tab",
                )
        finally:
            wave2_release.set()
        for name in ("sidebar", "remote"):
            self._wait_sub_tabs(name, set(), "sub-agent tabs closed on done")


if __name__ == "__main__":
    unittest.main()
