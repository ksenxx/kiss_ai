# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression test: nested ``run_parallel`` sub-agents must
open tabs in the frontend.

Bug
---
When a sub-agent spawned by ``run_parallel`` itself calls
``run_parallel``, the nested (grand-child) sub-agents never open any
tabs in the VS Code webview.

Root cause
----------
``ChatSorcarAgent._run_tasks_parallel`` resolves ``parent_tab_id`` by
scanning ``_RunningAgentState.running_agent_states`` for the entry
whose ``state.agent is self``.  For a TOP-LEVEL parent that registry
key is the real frontend tab id (set by the VS Code server), so the
child's ``new_tab`` broadcast passes the frontend guard::

    if (ev.parent_tab_id && !tabs.find(t => t.id === ev.parent_tab_id))
      break;

But a NESTED parent (a sub-agent that calls ``run_parallel``) is
registered under the BACKEND synthetic key
``task-{grandparent_task_id}__sub_{idx}`` — while its frontend viewer
tab was created by ``createBackgroundSubagentTab`` with a RANDOM
frontend id.  The nested children's ``new_tab`` broadcasts therefore
carry a ``parent_tab_id`` that no frontend tab has, every webview
drops them, and no tabs open.

Fix
---
When the parent is itself a sub-agent (``self._subagent_info is not
None``), resolve the FRONTEND viewer tab id from the printer's
subscriber map (``_fanout_targets(self._last_task_id)``) — populated
when the frontend posted ``resumeSession`` for this sub-agent's tab —
and use it as ``parent_tab_id``, falling back to the registry key.

Test harness
------------
The test drives the REAL ``ChatSorcarAgent.run`` (task persistence,
registry mirroring, ``new_tab`` broadcast, subscriber routing) against
a printer that faithfully simulates the ``main.js`` webview: it keeps
a ``tabs[]`` list, applies the exact ``new_tab`` guard, materialises
background sub-agent tabs with random frontend ids, and posts
``resumeSession`` (``subscribe_tab``) back to the backend.  Only the
LLM-driven body (``SorcarAgent.run``) is replaced by a scripted body
that invokes ``_run_tasks_parallel`` exactly as the ``run_parallel``
tool would.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.json_printer import JsonPrinter

ROOT_TAB_ID = "frontend-root-tab"


class _FrontendSimPrinter(JsonPrinter):
    """Printer that additionally simulates the ``main.js`` webview.

    Mirrors the frontend's ``case 'new_tab':`` handler byte-for-byte in
    behaviour:

    1. Guard: drop the event when ``parent_tab_id`` is set and no local
       tab carries that id.
    2. ``createBackgroundSubagentTab``: allocate a RANDOM frontend tab
       id (never the backend's synthetic ``task-...__sub_N`` key) with
       ``parentTabId`` linking to the parent tab.
    3. ``resumeSession``: the server's ``_cmd_resume_session`` handler
       subscribes the new frontend tab to the task's event stream via
       ``subscribe_tab``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.tabs: list[dict[str, str]] = [
            {"id": ROOT_TAB_ID, "parentTabId": ""},
        ]
        self.dropped_new_tabs: list[dict[str, Any]] = []
        self._sim_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the event and run the simulated frontend handler."""
        event = self._inject_task_id(event)
        with self._sim_lock:
            self.events.append(dict(event))
            if event.get("type") == "new_tab":
                self._handle_new_tab(event)
        with self._lock:
            self._record_event(event)

    def _handle_new_tab(self, ev: dict[str, Any]) -> None:
        """Faithful port of ``main.js`` ``case 'new_tab':``."""
        parent_tab_id = ev.get("parent_tab_id") or ""
        if parent_tab_id and not any(
            t["id"] == parent_tab_id for t in self.tabs
        ):
            self.dropped_new_tabs.append(dict(ev))
            return
        if ev.get("task_id") is None:
            return
        # createBackgroundSubagentTab: random frontend id.
        frontend_tab_id = "fe-" + uuid.uuid4().hex[:12]
        self.tabs.append(
            {"id": frontend_tab_id, "parentTabId": parent_tab_id},
        )
        # resumeSession → server _cmd_resume_session → subscribe_tab.
        self.subscribe_tab(ev.get("task_id"), frontend_tab_id)


def _scripted_base_run(
    self: SorcarAgent, prompt_template: str = "", **kwargs: Any
) -> str:
    """Scripted replacement for the LLM-driven ``SorcarAgent.run`` body.

    Replicates the base-run attribute setup that
    ``_run_tasks_parallel`` depends on, then acts out the scenario:

    * LEVEL0 (top-level parent): the model calls ``run_parallel`` with
      one sub-task.
    * LEVEL1 (sub-agent): the model calls ``run_parallel`` again with
      two nested sub-tasks — the buggy case.
    * LEVEL2 (nested sub-agents): the model just finishes.

    The level is derived from ``self.name`` — NOT from
    ``prompt_template`` — because ``build_chat_prompt`` embeds the
    PREVIOUS tasks of the shared chat session into every sub-agent's
    augmented prompt, so a LEVEL2 agent's prompt also contains the
    literal strings "LEVEL0"/"LEVEL1" and prompt matching would
    recurse forever.  ``_run_tasks_parallel`` names each sub-agent
    ``Parallel-{task[:40]}``, which uniquely encodes its own level.
    """
    self.printer = kwargs.get("printer") or getattr(self, "printer", None)
    self.model_name = str(
        kwargs.get("model_name") or getattr(self, "model_name", "") or "",
    )
    self.work_dir = str(
        kwargs.get("work_dir") or getattr(self, "work_dir", ".") or ".",
    )
    name = getattr(self, "name", "")
    if "nested-tabs-parent" in name:
        self._run_tasks_parallel(["LEVEL1 mid task that nests"])
    elif "LEVEL1" in name:
        self._run_tasks_parallel(["LEVEL2 inner task A", "LEVEL2 inner task B"])
    return "success: true\nsummary: done"


class TestNestedRunParallelOpensTabs:
    """Nested ``run_parallel`` children must open frontend tabs."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self.saved_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self.saved_states = dict(_RunningAgentState.running_agent_states)
        _RunningAgentState.running_agent_states.clear()
        self.saved_run = SorcarAgent.run
        SorcarAgent.run = cast(  # type: ignore[method-assign]
            Any, _scripted_base_run,
        )

    def teardown_method(self) -> None:
        SorcarAgent.run = self.saved_run  # type: ignore[method-assign]
        _RunningAgentState.running_agent_states.clear()
        _RunningAgentState.running_agent_states.update(self.saved_states)
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_nested_subagents_open_frontend_tabs(self) -> None:
        """A sub-agent's own ``run_parallel`` children must materialise
        frontend tabs (their ``new_tab`` broadcasts must carry a
        ``parent_tab_id`` the webview recognises)."""
        printer = _FrontendSimPrinter()

        parent = ChatSorcarAgent("nested-tabs-parent")
        parent.printer = printer  # type: ignore[assignment]
        # The VS Code server registers the top-level agent under the
        # REAL frontend tab id before running it.
        parent_state = _RunningAgentState(ROOT_TAB_ID, "test-model")
        parent_state.agent = parent  # type: ignore[assignment]
        _RunningAgentState.register(ROOT_TAB_ID, parent_state)
        try:
            result = parent.run(
                prompt_template="LEVEL0 root task",
                model_name="test-model",
                work_dir=self.tmpdir,
                printer=printer,
                is_parallel=True,
            )
        finally:
            _RunningAgentState.unregister(ROOT_TAB_ID)

        assert "success" in result

        # --- Level 1: the direct child of the root run_parallel. ---
        mid_tabs = [
            t for t in printer.tabs if t["parentTabId"] == ROOT_TAB_ID
        ]
        assert len(mid_tabs) == 1, (
            f"Expected exactly 1 first-level sub-agent tab under "
            f"{ROOT_TAB_ID!r}; tabs={printer.tabs!r}, "
            f"dropped={printer.dropped_new_tabs!r}"
        )
        mid_tab_id = mid_tabs[0]["id"]

        # --- Level 2: the nested run_parallel children (THE BUG). ---
        inner_tabs = [
            t for t in printer.tabs if t["parentTabId"] == mid_tab_id
        ]
        assert len(inner_tabs) == 2, (
            "Nested run_parallel sub-agents did not open any tabs: "
            f"expected 2 tabs under the sub-agent viewer tab "
            f"{mid_tab_id!r}, got {inner_tabs!r}. All tabs: "
            f"{printer.tabs!r}. Dropped new_tab events (frontend guard "
            f"rejected their parent_tab_id): {printer.dropped_new_tabs!r}"
        )

        # No new_tab broadcast may be dropped by the frontend guard.
        assert printer.dropped_new_tabs == [], (
            "new_tab events were dropped because their parent_tab_id "
            "matches no frontend tab: "
            f"{printer.dropped_new_tabs!r}"
        )

    def test_nested_new_tab_carries_frontend_viewer_tab_id(self) -> None:
        """The nested children's ``new_tab`` events must carry the
        FRONTEND viewer tab id of their parent sub-agent, never the
        backend synthetic ``task-...__sub_N`` registry key."""
        printer = _FrontendSimPrinter()

        parent = ChatSorcarAgent("nested-tabs-parent-2")
        parent.printer = printer  # type: ignore[assignment]
        parent_state = _RunningAgentState(ROOT_TAB_ID, "test-model")
        parent_state.agent = parent  # type: ignore[assignment]
        _RunningAgentState.register(ROOT_TAB_ID, parent_state)
        try:
            parent.run(
                prompt_template="LEVEL0 root task",
                model_name="test-model",
                work_dir=self.tmpdir,
                printer=printer,
                is_parallel=True,
            )
        finally:
            _RunningAgentState.unregister(ROOT_TAB_ID)

        new_tab_events = [
            e for e in printer.events if e.get("type") == "new_tab"
        ]
        # 1 (level-1 child) + 2 (level-2 nested children).
        assert len(new_tab_events) == 3, (
            f"Expected 3 new_tab broadcasts, got {new_tab_events!r}"
        )
        frontend_tab_ids = {t["id"] for t in printer.tabs}
        for ev in new_tab_events:
            ptid = ev.get("parent_tab_id", "")
            assert not str(ptid).startswith("task-"), (
                "new_tab must never carry the backend synthetic "
                f"registry key as parent_tab_id: {ev!r}"
            )
            assert ptid in frontend_tab_ids, (
                f"new_tab parent_tab_id {ptid!r} is not a frontend tab "
                f"id (frontend tabs: {frontend_tab_ids!r}); the webview "
                f"guard drops this event and no tab opens: {ev!r}"
            )
