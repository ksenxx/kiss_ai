# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the "Suggested next" bar comes from ``finish()`` itself.

The agent proposes its own follow-up through the 4th ``finish``
parameter, ``suggested_next_task``.  ``_run_task`` reads it from the
result YAML the agent returns and, after the ``task_done`` event,
emits ONE ``followup_suggestion`` event per watching tab and persists it
exactly once — no separate follow-up LLM call is made any more.

Drives the real ``VSCodeServer._run_task`` lifecycle with a real
``WorktreeSorcarAgent`` subclass whose ``run`` returns a scripted
``finish()`` result (the same harness as ``test_wave2_runner_bugs.py``),
a real ``JsonPrinter`` subclass that records every broadcast, and the
real SQLite persistence layer.  No mocks or patches.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.core.utils import finish
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


class _CapturePrinter(JsonPrinter):
    """Real printer subclass that records every broadcast event."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, then run the real record/persist path."""
        self.events.append(dict(event))
        super().broadcast(event)


class _ScriptedAgent(WorktreeSorcarAgent):
    """Real agent subclass whose ``run`` returns a scripted ``finish()``.

    Mirrors what the task runner relies on from ``ChatSorcarAgent.run``
    under ``_skip_persistence=True``: a fresh ``task_history`` row per
    call, ``_last_task_id`` published under ``_task_id_lock``, and the
    printer's thread-local ``task_id`` set for the run and cleared on
    exit.  The follow-up is taken from the prompt text so each test
    controls it: ``prompt`` → suggestion ``"next after <prompt>"``;
    a prompt containing ``[no-followup]`` yields no suggestion; a prompt
    containing ``[unparsable]`` returns a non-YAML result; a prompt
    containing ``[raise]`` fails after allocating its task row.
    """

    viewer_tab: str = ""

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Allocate a task row and return the scripted result."""
        prompt_template = kwargs.get("prompt_template", "")
        printer = kwargs.get("printer")
        task_id, self._chat_id = _add_task(
            prompt_template, chat_id=self._chat_id or "",
        )
        with self._task_id_lock:
            self._last_task_id = task_id
        if printer is not None:
            printer._thread_local.task_id = str(task_id)
            printer._thread_local.task_id = ""
            if self.viewer_tab:
                # A second tab watching the task (history-resume / chat
                # viewer) subscribes once the run has published its id.
                printer.subscribe_tab(str(task_id), self.viewer_tab)
        if "[unparsable]" in prompt_template:
            return "not yaml at all: [unbalanced"
        if "[raise]" in prompt_template:
            raise RuntimeError("scripted failure")
        suggestion = (
            "" if "[no-followup]" in prompt_template else f"next after {prompt_template}"
        )
        return finish(
            True,
            summary_in_html=f"<p>done {prompt_template}</p>",
            suggested_next_task=suggestion,
        )


def _pop_states(prefix: str) -> None:
    """Remove every agent state whose tab id starts with *prefix*."""
    with agent_state.STATE_LOCK:
        stale = [st.task_id for st in agent_state.snapshot() if st.tab_id.startswith(prefix)]
    for task_id in stale:
        agent_state.unregister(task_id)


def _run_scripted_task(
    tmp_path: Path, tab_id: str, prompt: str, viewer_tab: str = "",
) -> tuple[_CapturePrinter, str]:
    """Run *prompt* through the real ``_run_task``; return (printer, task_id)."""
    models = get_available_models()
    if not models:
        pytest.skip("no models configured in this environment")
    printer = _CapturePrinter()
    server = VSCodeServer(printer=printer)
    agent = _ScriptedAgent("Sorcar VS Code")
    agent.viewer_tab = viewer_tab
    state = AgentState(f"pre-{tab_id}", agent=agent, tab_id=tab_id, server_owned=True)
    agent_state.register(state)
    server._run_task({
        "tabId": tab_id,
        "prompt": prompt,
        "workDir": str(tmp_path),
        "model": models[0],
        "_state_key": state.task_id,
    })
    return printer, str(agent._last_task_id)


def _followup_rows(task_id: str) -> list[dict[str, Any]]:
    """Return the persisted ``followup_suggestion`` events of *task_id*."""
    persistence._flush_chat_events()
    conn = sqlite3.connect(str(persistence._DB_PATH))
    try:
        rows = conn.execute(
            "SELECT event_json FROM events WHERE task_id = ? ORDER BY rowid",
            (task_id,),
        ).fetchall()
    finally:
        conn.close()
    events = [json.loads(r[0]) for r in rows]
    return [e for e in events if e.get("type") == "followup_suggestion"]


def _followup_broadcasts(printer: _CapturePrinter) -> list[dict[str, Any]]:
    return [e for e in printer.events if e.get("type") == "followup_suggestion"]


def test_suggestion_is_broadcast_after_task_done_and_persisted_once(
    tmp_path: Path,
) -> None:
    tab_id = "snt-basic-tab"
    try:
        printer, task_id = _run_scripted_task(tmp_path, tab_id, "snt basic")
        broadcasts = _followup_broadcasts(printer)
        assert len(broadcasts) == 1
        ev = broadcasts[0]
        assert ev["text"] == "next after snt basic"
        assert ev["tabId"] == tab_id
        assert ev["ts"] > 0
        types = [e.get("type") for e in printer.events]
        assert types.index("task_done") < types.index("followup_suggestion")
        rows = _followup_rows(task_id)
        assert [r["text"] for r in rows] == ["next after snt basic"]
        assert rows[0]["ts"] == ev["ts"]
        assert "tabId" not in rows[0]
    finally:
        _pop_states("snt-")


def test_suggestion_reaches_every_watching_tab(tmp_path: Path) -> None:
    tab_id = "snt-fanout-tab"
    try:
        printer, task_id = _run_scripted_task(
            tmp_path, tab_id, "snt fanout", viewer_tab="snt-viewer-tab",
        )
        broadcasts = _followup_broadcasts(printer)
        assert sorted(e["tabId"] for e in broadcasts) == [tab_id, "snt-viewer-tab"]
        assert len(_followup_rows(task_id)) == 1
    finally:
        _pop_states("snt-")


def test_no_suggestion_means_no_event_and_no_row(tmp_path: Path) -> None:
    tab_id = "snt-none-tab"
    try:
        printer, task_id = _run_scripted_task(tmp_path, tab_id, "snt quiet [no-followup]")
        assert _followup_broadcasts(printer) == []
        assert _followup_rows(task_id) == []
        assert any(e.get("type") == "task_done" for e in printer.events)
    finally:
        _pop_states("snt-")


def test_unparsable_result_means_no_suggestion(tmp_path: Path) -> None:
    tab_id = "snt-raw-tab"
    try:
        printer, task_id = _run_scripted_task(tmp_path, tab_id, "snt raw [unparsable]")
        assert _followup_broadcasts(printer) == []
        assert _followup_rows(task_id) == []
        assert any(e.get("type") == "task_done" for e in printer.events)
    finally:
        _pop_states("snt-")


def test_multi_subtask_run_uses_last_subtasks_suggestion(tmp_path: Path) -> None:
    tab_id = "snt-multi-tab"
    try:
        printer, task_id = _run_scripted_task(
            tmp_path,
            tab_id,
            "<task>snt first</task>\n<task>snt second</task>",
        )
        broadcasts = _followup_broadcasts(printer)
        assert [e["text"] for e in broadcasts] == ["next after snt second"]
        assert [r["text"] for r in _followup_rows(task_id)] == ["next after snt second"]
    finally:
        _pop_states("snt-")


def test_failed_later_subtask_does_not_republish_earlier_suggestion(
    tmp_path: Path,
) -> None:
    tab_id = "snt-stale-tab"
    try:
        printer, _task_id = _run_scripted_task(
            tmp_path,
            tab_id,
            "<task>snt ok first</task>\n<task>snt boom [raise]</task>",
        )
        assert _followup_broadcasts(printer) == []
        assert any(e.get("type") == "task_error" for e in printer.events)
        persistence._flush_chat_events()
        conn = sqlite3.connect(str(persistence._DB_PATH))
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_json LIKE '%followup_suggestion%' "
                "AND event_json LIKE '%snt ok first%'",
            ).fetchone()
        finally:
            conn.close()
        assert rows[0] == 0
    finally:
        _pop_states("snt-")
