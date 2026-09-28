# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a running sub-agent's tab is open on EVERY surface, and closed
on every surface once the sub-agent is over.

The invariant (``~/.kiss/SORCAR.md``)::

    whenever a subagent task is running its tab must be open across
    all surfaces and must be closed from all surfaces once the
    subagent's task is over.

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
   A tab the user closes by hand on one surface closes on all of them.
3. After the children finish: every surface has zero sub-agent tabs.
4. A surface connecting AFTER completion never shows a finished
   sub-agent tab.

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

    def _install_fanout_stub(self, leaf_release: threading.Event) -> None:
        """Parent fans out two children; one nests a grandchild.

        Leaves (the plain child and the grandchild) block on
        *leaf_release* so the mid-run state can be inspected.
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
            # The level comes from ``self.name`` (``Parallel-{task}``),
            # never from the prompt: sub-agent prompts embed the chat's
            # earlier tasks.  Only the root agent fans out; any other
            # agent the daemon runs alongside (e.g. the "Task update"
            # side channel) is a plain leaf that returns at once.
            name = str(getattr(self_agent, "name", ""))
            if GRANDCHILD_MARK in name:
                leaf_release.wait(timeout=60)
            elif NEST_MARK in name:
                # Fan out IMMEDIATELY: no client can have subscribed to
                # this sub-agent's tab yet, the worst case for parenting
                # the grandchild under a tab every surface knows.
                _fan_out(self_agent, [f"{GRANDCHILD_MARK} leaf"])
            elif CHILD_MARK in name:
                leaf_release.wait(timeout=60)
            elif getattr(self_agent, "_subagent_info", None) is None:
                _fan_out(self_agent, [
                    f"{CHILD_MARK} plain leaf",
                    f"{CHILD_MARK} {NEST_MARK} parent of a grandchild",
                ])
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

            # 2b. The surfaces never disagree: a sub-agent tab the user
            #     closes by hand on one surface closes on every other
            #     (daemon ``closeSubagentTab`` mirror), while the other
            #     running sub-agents' tabs stay everywhere.
            closed = self.bridge.call("closeTab", name="remote", tabId=plain_tab)
            assert closed["found"], closed
            for name in ("sidebar", "remote", "remote2", "editor"):
                self._wait_sub_tabs(
                    name, {nesting_tab, grand_tab},
                    "hand-closed sub-agent tab mirrored to every surface",
                )
        finally:
            leaf_release.set()

        # 3. Once the sub-agents are over, every surface closes their
        #    tabs — and the parent tab itself stays.
        for name in ("sidebar", "remote", "remote2", "editor"):
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


if __name__ == "__main__":
    unittest.main()
