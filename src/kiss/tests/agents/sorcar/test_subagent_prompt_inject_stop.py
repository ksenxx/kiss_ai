# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: prompt injection + Stop on a running SUB-AGENT tab.

Reproduces two user-reported bugs:

1. A message typed into a sub-agent tab's input textbox
   (``appendUserMessage``) must show up as a ``prompt`` panel in the
   sub-agent's trajectory — both live AND after any ``task_events``
   replay / reload.  Before the fix, the echoed ``prompt`` event
   carried only ``tabId`` (no ``taskId``), so ``WebPrinter.broadcast``
   treated it as a transient targeted system event: never recorded in
   the printer's in-memory recording and never persisted to the
   ``events`` table.  Sub-agent tabs rebuild their transcript from the
   recording on every reopen, so the injected prompt vanished.

2. Pressing the Stop button on a sub-agent tab
   (``{"type": "stop", "tabId": <viewer>}``) must set ONLY that
   sub-agent's stop event — the parent task and sibling sub-agents
   keep running.

The tests replicate the exact production wiring: real
:class:`ChatSorcarAgent` instances, real ``task_history`` rows
allocated via ``_add_task``, real :class:`_RunningAgentState`
registrations mirroring ``ChatSorcarAgent._run_tasks_parallel``'s
``_run_single`` (synthetic ``task-{parent}__sub_{idx}`` tab ids,
``_SubagentStopEvent`` chained to the parent's), a real
:class:`WebPrinter` and a real :class:`VSCodeServer` dispatching the
same commands the webview posts.  No mocks, patches, or test doubles.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent, _SubagentStopEvent
from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.web_server import WebPrinter
from kiss.core.models.anthropic_model import AnthropicModel


@pytest.fixture()
def isolated_db(tmp_path: Path):
    """Redirect the persistence DB to a temp dir for the test."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = tmp_path / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    yield
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


class _Harness:
    """Production-mirroring parent + sub-agent wiring on one server."""

    def __init__(self) -> None:
        self.printer = WebPrinter()
        self.server = VSCodeServer(self.printer)

        # Parent task: real ChatSorcarAgent with a real task_history row,
        # registered under its frontend tab id (as _run_task_inner does).
        self.parent = ChatSorcarAgent("Parent")
        parent_task_id, chat_id = _add_task("parent task")
        self.parent._last_task_id = parent_task_id
        self.parent._chat_id = chat_id
        self.parent_stop = threading.Event()
        self.parent_state = _RunningAgentState(
            "tab-parent",
            "test-model",
            agent=self.parent,  # type: ignore[arg-type]
            chat_id=chat_id,
            is_task_active=True,
            stop_event=self.parent_stop,
        )
        _RunningAgentState.register("tab-parent", self.parent_state)
        self.printer.subscribe_tab(parent_task_id, "tab-parent")

        # Sub-agent: mirrors chat_sorcar_agent._run_tasks_parallel's
        # _run_single — synthetic backend tab id, _SubagentStopEvent
        # chained to the parent's, is_subagent=True, agent._tab_id set.
        self.sub = ChatSorcarAgent("Parallel-sub")
        self.sub.resume_chat_by_id(chat_id)
        sub_task_id, _ = _add_task(
            "sub task",
            chat_id=chat_id,
            extra={"subagent": {"parent_task_id": parent_task_id}},
        )
        self.sub._last_task_id = sub_task_id
        self.sub_task_id = str(sub_task_id)
        self.parent_task_id = str(parent_task_id)
        self.sub_tab_id = f"task-{parent_task_id}__sub_0"
        self.sub._tab_id = self.sub_tab_id  # type: ignore[attr-defined]
        self.sub_stop = _SubagentStopEvent(self.parent_stop)
        self.sub_state = _RunningAgentState(
            self.sub_tab_id,
            "test-model",
            agent=self.sub,  # type: ignore[arg-type]
            chat_id=chat_id,
            is_subagent=True,
            parent_task_id=str(parent_task_id),
            is_task_active=True,
            stop_event=self.sub_stop,
        )
        _RunningAgentState.register(self.sub_tab_id, self.sub_state)

        # Register both agents in the printer's persistence map and open
        # in-memory recordings, exactly as ChatSorcarAgent.run does on
        # the agent thread.
        for key, agent in (
            (self.parent_task_id, self.parent),
            (self.sub_task_id, self.sub),
        ):
            self.printer._persist_agents[key] = agent
            self.printer._thread_local.task_id = key
            self.printer.start_recording()
        # Commands arrive on a thread with no task binding.
        self.printer._thread_local.task_id = ""

        # The frontend materialised a viewer tab for the sub-agent and
        # posted resumeSession → subscribe_tab (main.js 'new_tab').
        self.viewer_tab = "tab-sub-viewer"
        self.printer.subscribe_tab(sub_task_id, self.viewer_tab)

    def recording(self, task_id: str) -> list[dict[str, Any]]:
        with self.printer._lock:
            return list(self.printer._recordings.get(task_id, []))

    def persisted_events(self, task_id: str) -> list[dict[str, Any]]:
        th._flush_chat_events()
        row = th._load_chat_events_by_task_id(task_id)
        events = (row or {}).get("events")
        if not isinstance(events, list):
            return []
        return [e for e in events if isinstance(e, dict)]


def _persisted_prompts(rows: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    return [
        e for e in rows
        if e.get("type") == "prompt" and text in str(e.get("text", ""))
    ]


class TestSubagentPromptInjection:
    """appendUserMessage on a sub-agent viewer tab (bug 1)."""

    def test_prompt_routed_to_subagent_queue_only(self, isolated_db) -> None:
        """The injected prompt lands in the SUB-agent's queue, not the parent's."""
        h = _Harness()
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "HELLO SUBAGENT INJECTION",
            "tabId": h.viewer_tab,
        })
        assert h.sub_state.pending_user_messages == ["HELLO SUBAGENT INJECTION"]
        assert h.parent_state.pending_user_messages == []

    def test_prompt_recorded_in_subagent_trajectory(self, isolated_db) -> None:
        """The echo is recorded under the sub-agent's task (replay survives)."""
        h = _Harness()
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "HELLO SUBAGENT INJECTION",
            "tabId": h.viewer_tab,
        })
        rec = h.recording(h.sub_task_id)
        prompts = [
            e for e in rec
            if e.get("type") == "prompt"
            and "HELLO SUBAGENT INJECTION" in str(e.get("text", ""))
        ]
        assert prompts, f"prompt echo not recorded; recording={rec}"
        # The recorded copy must not pin a stale viewer tab id — replay
        # re-stamps events with the subscribing tab's own id.
        assert "tabId" not in prompts[0]
        assert prompts[0].get("taskId") == h.sub_task_id
        # Nothing leaked into the parent's recording.
        assert not [
            e for e in h.recording(h.parent_task_id)
            if "HELLO SUBAGENT INJECTION" in str(e.get("text", ""))
        ]

    def test_prompt_persisted_to_subagent_events(self, isolated_db) -> None:
        """The echo is persisted to the sub-agent's events rows (reload survives)."""
        h = _Harness()
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "HELLO SUBAGENT INJECTION",
            "tabId": h.viewer_tab,
        })
        rows = h.persisted_events(h.sub_task_id)
        assert _persisted_prompts(rows, "HELLO SUBAGENT INJECTION")
        assert not _persisted_prompts(
            h.persisted_events(h.parent_task_id), "HELLO SUBAGENT INJECTION",
        )

    def test_prompt_on_main_tab_recorded_and_persisted(self, isolated_db) -> None:
        """The same trajectory guarantees hold for a MAIN (parent) tab."""
        h = _Harness()
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "HELLO PARENT INJECTION",
            "tabId": "tab-parent",
        })
        assert h.parent_state.pending_user_messages == ["HELLO PARENT INJECTION"]
        rec = h.recording(h.parent_task_id)
        assert [
            e for e in rec
            if e.get("type") == "prompt"
            and "HELLO PARENT INJECTION" in str(e.get("text", ""))
        ]
        rows = h.persisted_events(h.parent_task_id)
        assert _persisted_prompts(rows, "HELLO PARENT INJECTION")

    def test_blank_prompt_ignored(self, isolated_db) -> None:
        """Whitespace-only prompts are dropped before any routing."""
        h = _Harness()
        h.server._handle_command({
            "type": "appendUserMessage", "prompt": "   ", "tabId": h.viewer_tab,
        })
        assert h.sub_state.pending_user_messages == []
        assert h.recording(h.sub_task_id) == []

    def test_prompt_without_owner_task_deferred_then_flushed_by_drain(
        self, isolated_db,
    ) -> None:
        """A pre-allocation prompt is deferred, then persisted at drain time.

        When the owner's agent has not allocated its ``task_history``
        row yet (fast follow-up during ``run()`` startup) the echo is
        emitted transiently (tab-only, so the user sees it at once)
        and a durable copy is deferred via
        ``unattributed_prompt_echoes``: the REAL drain hook records +
        persists it (``recordOnly``) from the agent thread, where the
        printer's thread-local task id names the task that actually
        consumed the message.
        """
        h = _Harness()
        h.parent._last_task_id = None
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "NO TASK YET",
            "tabId": "tab-parent",
        })
        # Queued for the drain hook and marked for deferred durable
        # echoing — the immediate echo is transient (tab-only), so
        # nothing is recorded yet.
        assert h.parent_state.pending_user_messages == ["NO TASK YET"]
        assert h.parent_state.unattributed_prompt_echoes == ["NO TASK YET"]
        assert h.recording(h.parent_task_id) == []

        # The agent thread allocates the task row and reaches its first
        # model step: the drain hook injects the message and flushes
        # the deferred echo under the consuming task's id.
        h.parent._last_task_id = h.parent_task_id  # run(): _add_task done
        h.parent.printer = h.printer
        h.parent._tab_id = "tab-parent"  # type: ignore[attr-defined]
        model = AnthropicModel(
            "claude-haiku-4-5",
            os.environ.get("ANTHROPIC_API_KEY", "test-key"),
        )
        h.printer._thread_local.task_id = h.parent_task_id
        try:
            h.parent._drain_pending_user_messages(model)
        finally:
            h.printer._thread_local.task_id = ""
        assert h.parent_state.pending_user_messages == []
        assert h.parent_state.unattributed_prompt_echoes == []
        assert [
            m for m in model.conversation
            if m.get("role") == "user" and "NO TASK YET" in str(m.get("content"))
        ]
        rec = h.recording(h.parent_task_id)
        assert [
            e for e in rec
            if e.get("type") == "prompt" and "NO TASK YET" in str(e.get("text", ""))
        ]
        assert _persisted_prompts(
            h.persisted_events(h.parent_task_id), "NO TASK YET",
        )

    def test_prompt_on_idle_viewer_dropped(self, isolated_db) -> None:
        """A viewer of a finished sub-agent silently drops the prompt."""
        h = _Harness()
        h.sub_state.is_task_active = False
        h.sub_state.stop_event = None
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "TOO LATE",
            "tabId": h.viewer_tab,
        })
        assert h.sub_state.pending_user_messages == []
        assert h.recording(h.sub_task_id) == []

    def test_drain_without_printer_drops_deferred_echo_silently(
        self, isolated_db,
    ) -> None:
        """A printer-less agent drains deferred echoes without crashing."""
        h = _Harness()
        h.parent._last_task_id = None
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "SILENT DEFER",
            "tabId": "tab-parent",
        })
        assert h.parent_state.unattributed_prompt_echoes == ["SILENT DEFER"]
        h.parent._last_task_id = h.parent_task_id
        h.parent.printer = None
        h.parent._tab_id = "tab-parent"  # type: ignore[attr-defined]
        model = AnthropicModel(
            "claude-haiku-4-5",
            os.environ.get("ANTHROPIC_API_KEY", "test-key"),
        )
        h.parent._drain_pending_user_messages(model)
        # Message still consumed; deferred echo dropped, no crash.
        assert [
            m for m in model.conversation
            if m.get("role") == "user" and "SILENT DEFER" in str(m.get("content"))
        ]
        assert h.parent_state.unattributed_prompt_echoes == []
        assert h.recording(h.parent_task_id) == []

    def test_drain_with_attributed_prompt_emits_no_duplicate_echo(
        self, isolated_db,
    ) -> None:
        """An already-attributed prompt is echoed exactly once.

        The command handler stamps and records the echo immediately;
        the subsequent drain must inject the message into the model
        WITHOUT broadcasting a second ``prompt`` event (the deferred
        list is empty), so no duplicate panel lands in the trajectory.
        """
        h = _Harness()
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "ONCE ONLY",
            "tabId": "tab-parent",
        })
        assert h.parent_state.unattributed_prompt_echoes == []
        h.parent.printer = h.printer
        h.parent._tab_id = "tab-parent"  # type: ignore[attr-defined]
        model = AnthropicModel(
            "claude-haiku-4-5",
            os.environ.get("ANTHROPIC_API_KEY", "test-key"),
        )
        h.printer._thread_local.task_id = h.parent_task_id
        try:
            h.parent._drain_pending_user_messages(model)
        finally:
            h.printer._thread_local.task_id = ""
        assert [
            m for m in model.conversation
            if m.get("role") == "user" and "ONCE ONLY" in str(m.get("content"))
        ]
        echoes = [
            e for e in h.recording(h.parent_task_id)
            if e.get("type") == "prompt" and "ONCE ONLY" in str(e.get("text", ""))
        ]
        assert len(echoes) == 1

    def test_prompt_with_agentless_owner_deferred(self, isolated_db) -> None:
        """An owner state with NO agent attached defers the echo."""
        h = _Harness()
        h.parent_state.agent = None
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "AGENTLESS OWNER",
            "tabId": "tab-parent",
        })
        # Queued for the drain hook; unattributable → deferred echo,
        # nothing recorded yet.
        assert h.parent_state.pending_user_messages == ["AGENTLESS OWNER"]
        assert h.parent_state.unattributed_prompt_echoes == ["AGENTLESS OWNER"]
        assert h.recording(h.parent_task_id) == []

    def test_prompt_dropped_when_resolved_source_went_inactive(
        self, isolated_db,
    ) -> None:
        """A resolvable source that is no longer task-active drops the prompt.

        ``_find_source_tab_for_viewer`` resolves the sub-agent by its
        live ``stop_event`` + matching task id, but the owner guard in
        ``_cmd_append_user_message`` must still reject it once
        ``is_task_active`` has been cleared (task in teardown — the
        pending queue would never be drained).
        """
        h = _Harness()
        h.sub_state.is_task_active = False  # stop_event stays live
        h.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "TEARDOWN RACE",
            "tabId": h.viewer_tab,
        })
        assert h.sub_state.pending_user_messages == []
        assert h.recording(h.sub_task_id) == []


class TestBusyRunConversion:
    """A stale ``run``/submit on a busy tab is queued AND persisted.

    A reopened webview can believe a tab is idle (before the resume's
    ``status running:true`` arrives) and send typed text as a
    ``submit`` (→ ``run``) instead of ``appendUserMessage``.
    ``_cmd_run`` converts it into a queued follow-up; its echoed
    ``prompt`` event must carry the owning task id so the message
    survives ``task_events`` replay exactly like the
    ``appendUserMessage`` path (it previously stayed transient).
    """

    def test_busy_run_queues_and_persists_prompt(self, isolated_db) -> None:
        """run on a busy tab queues the prompt and persists the echo."""
        h = _Harness()
        h.parent_state.task_thread = threading.Thread()  # created, unstarted
        h.server._handle_command({
            "type": "run",
            "prompt": "STALE SUBMIT FOLLOW-UP",
            "tabId": "tab-parent",
        })
        # Queued into the LIVE task — no second task started.
        assert h.parent_state.pending_user_messages == [
            "STALE SUBMIT FOLLOW-UP",
        ]
        rec = h.recording(h.parent_task_id)
        prompts = [
            e for e in rec
            if e.get("type") == "prompt"
            and "STALE SUBMIT FOLLOW-UP" in str(e.get("text", ""))
        ]
        assert prompts, f"busy-run echo not recorded; recording={rec}"
        assert "tabId" not in prompts[0]
        assert prompts[0].get("taskId") == h.parent_task_id
        rows = h.persisted_events(h.parent_task_id)
        assert _persisted_prompts(rows, "STALE SUBMIT FOLLOW-UP")

    def test_busy_run_without_task_id_deferred_then_flushed_by_drain(
        self, isolated_db,
    ) -> None:
        """Busy run before task-row allocation defers, then persists.

        A fast double-submit lands after the winning submit armed
        ``task_thread`` but before its ``run()`` allocated the task
        row: the prompt must not be mis-attributed to a previous task
        NOR lost on replay — the drain hook flushes the echo under
        the id of the task that consumed it.
        """
        h = _Harness()
        h.parent_state.task_thread = threading.Thread()
        h.parent._last_task_id = None  # run() entered, _add_task not yet
        h.server._handle_command({
            "type": "run",
            "prompt": "PRE-ALLOCATION SUBMIT",
            "tabId": "tab-parent",
        })
        assert h.parent_state.pending_user_messages == [
            "PRE-ALLOCATION SUBMIT",
        ]
        assert h.parent_state.unattributed_prompt_echoes == [
            "PRE-ALLOCATION SUBMIT",
        ]
        # Not attributable yet: nothing recorded, nothing persisted.
        assert h.recording(h.parent_task_id) == []
        assert not _persisted_prompts(
            h.persisted_events(h.parent_task_id), "PRE-ALLOCATION SUBMIT",
        )

        # The worker thread's run() allocates the row and reaches the
        # first model step — the drain flushes the durable echo.
        h.parent._last_task_id = h.parent_task_id  # run(): _add_task done
        h.parent.printer = h.printer
        h.parent._tab_id = "tab-parent"  # type: ignore[attr-defined]
        model = AnthropicModel(
            "claude-haiku-4-5",
            os.environ.get("ANTHROPIC_API_KEY", "test-key"),
        )
        h.printer._thread_local.task_id = h.parent_task_id
        try:
            h.parent._drain_pending_user_messages(model)
        finally:
            h.printer._thread_local.task_id = ""
        assert h.parent_state.unattributed_prompt_echoes == []
        assert _persisted_prompts(
            h.persisted_events(h.parent_task_id), "PRE-ALLOCATION SUBMIT",
        )

    def test_blank_run_on_busy_tab_ignored(self, isolated_db) -> None:
        """A whitespace-only run on a busy tab is neither queued nor echoed."""
        h = _Harness()
        h.parent_state.task_thread = threading.Thread()
        h.server._handle_command({
            "type": "run", "prompt": "   ", "tabId": "tab-parent",
        })
        assert h.parent_state.pending_user_messages == []
        assert h.recording(h.parent_task_id) == []


class TestWebPrinterTargetedEventGuard:
    """Only ``prompt`` echoes may cross from transient to task events."""

    def test_nonprompt_tab_event_with_task_id_stays_transient(
        self, isolated_db,
    ) -> None:
        """A tab-stamped ``status`` with a taskId must NOT be recorded.

        Launcher/viewer ``status`` events carry a client-supplied
        CORRELATION ``taskId`` (not a task-stream key); recording them
        would contaminate the task's recording and replay.
        """
        h = _Harness()
        h.printer.broadcast({
            "type": "status",
            "running": True,
            "tabId": h.viewer_tab,
            "taskId": h.sub_task_id,
        })
        assert h.recording(h.sub_task_id) == []
        assert not [
            e for e in h.persisted_events(h.sub_task_id)
            if e.get("type") == "status"
        ]

    def test_prompt_echo_for_unknown_task_is_harmless(
        self, isolated_db,
    ) -> None:
        """A stamped echo for a task with no recording/agent is a no-op."""
        h = _Harness()
        h.printer.broadcast({
            "type": "prompt",
            "text": "ORPHAN ECHO",
            "tabId": h.viewer_tab,
            "taskId": "999999",
        })
        assert h.recording(h.sub_task_id) == []
        assert h.recording(h.parent_task_id) == []

    def test_record_only_prompt_recorded_and_persisted(
        self, isolated_db,
    ) -> None:
        """A ``recordOnly`` echo is recorded + persisted, marker stripped.

        This is the drain hook's durable copy of a deferred prompt
        echo: broadcast from the agent thread (thread-local task id
        bound), it must land in the task's recording and ``events``
        rows WITHOUT the ``recordOnly`` marker, so ``task_events``
        replay renders it as a plain prompt panel.
        """
        h = _Harness()
        h.printer._thread_local.task_id = h.sub_task_id
        try:
            h.printer.broadcast({
                "type": "prompt",
                "text": "DURABLE COPY",
                "recordOnly": True,
            })
        finally:
            h.printer._thread_local.task_id = ""
        rec = [
            e for e in h.recording(h.sub_task_id)
            if e.get("type") == "prompt"
            and "DURABLE COPY" in str(e.get("text", ""))
        ]
        assert rec, "recordOnly prompt was not recorded"
        assert "recordOnly" not in rec[0]
        assert "tabId" not in rec[0]
        assert _persisted_prompts(
            h.persisted_events(h.sub_task_id), "DURABLE COPY",
        )
        # Sibling/parent recordings untouched.
        assert h.recording(h.parent_task_id) == []

    def test_plain_task_event_still_recorded_and_fanned_out(
        self, isolated_db,
    ) -> None:
        """A normal (non-``recordOnly``) task event keeps full routing.

        The ``recordOnly`` early-return must not swallow ordinary
        agent-thread events: they are recorded, persisted AND handed to
        the per-tab fan-out exactly as before.
        """
        h = _Harness()
        h.printer._thread_local.task_id = h.sub_task_id
        try:
            h.printer.broadcast({
                "type": "prompt",
                "text": "PLAIN TASK EVENT",
            })
        finally:
            h.printer._thread_local.task_id = ""
        rec = [
            e for e in h.recording(h.sub_task_id)
            if e.get("type") == "prompt"
            and "PLAIN TASK EVENT" in str(e.get("text", ""))
        ]
        assert rec, "plain task event was not recorded"
        assert _persisted_prompts(
            h.persisted_events(h.sub_task_id), "PLAIN TASK EVENT",
        )

    def test_record_only_prompt_without_task_id_dropped(
        self, isolated_db,
    ) -> None:
        """A ``recordOnly`` echo with no resolvable task id is dropped.

        It was already rendered live at queueing time; with nowhere to
        record it, re-sending it verbatim would duplicate the prompt
        panel — the printer must swallow it entirely.
        """
        h = _Harness()
        h.printer.broadcast({
            "type": "prompt",
            "text": "NOWHERE TO GO",
            "recordOnly": True,
        })
        assert h.recording(h.sub_task_id) == []
        assert h.recording(h.parent_task_id) == []
        assert not _persisted_prompts(
            h.persisted_events(h.sub_task_id), "NOWHERE TO GO",
        )
        assert not _persisted_prompts(
            h.persisted_events(h.parent_task_id), "NOWHERE TO GO",
        )


class TestSubagentStop:
    """Stop on a sub-agent viewer tab (bug 2)."""

    def test_stop_on_viewer_stops_only_subagent(self, isolated_db) -> None:
        """Stop with the viewer tab id sets ONLY the sub-agent's event."""
        h = _Harness()
        h.server._handle_command({"type": "stop", "tabId": h.viewer_tab})
        assert h.sub_stop.is_set()
        assert not h.parent_stop.is_set()

    def test_stop_on_backend_sub_tab_id(self, isolated_db) -> None:
        """Stop addressed at the synthetic backend tab id also works."""
        h = _Harness()
        h.server._handle_command({"type": "stop", "tabId": h.sub_tab_id})
        assert h.sub_stop.is_set()
        assert not h.parent_stop.is_set()

    def test_stop_on_parent_propagates_to_subagent(self, isolated_db) -> None:
        """Stopping the parent chains into the sub-agent's stop event."""
        h = _Harness()
        h.server._handle_command({"type": "stop", "tabId": "tab-parent"})
        assert h.parent_stop.is_set()
        # _SubagentStopEvent chains the parent's event.
        assert h.sub_stop.is_set()

    def test_stop_on_sibling_viewer_leaves_this_subagent_running(
        self, isolated_db,
    ) -> None:
        """A second sub-agent's Stop must not leak into this one."""
        h = _Harness()
        sib = ChatSorcarAgent("Parallel-sib")
        sib.resume_chat_by_id(h.parent._chat_id)
        sib_task_id, _ = _add_task(
            "sibling task",
            chat_id=h.parent._chat_id,
            extra={"subagent": {"parent_task_id": h.parent_task_id}},
        )
        sib._last_task_id = sib_task_id
        sib_tab_id = f"task-{h.parent_task_id}__sub_1"
        sib._tab_id = sib_tab_id  # type: ignore[attr-defined]
        sib_stop = _SubagentStopEvent(h.parent_stop)
        sib_state = _RunningAgentState(
            sib_tab_id,
            "test-model",
            agent=sib,  # type: ignore[arg-type]
            chat_id=h.parent._chat_id,
            is_subagent=True,
            parent_task_id=h.parent_task_id,
            is_task_active=True,
            stop_event=sib_stop,
        )
        _RunningAgentState.register(sib_tab_id, sib_state)
        h.printer.subscribe_tab(sib_task_id, "tab-sib-viewer")

        h.server._handle_command({"type": "stop", "tabId": "tab-sib-viewer"})
        assert sib_stop.is_set()
        assert not h.sub_stop.is_set()
        assert not h.parent_stop.is_set()
