# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: a ChatSorcarAgent / WorktreeSorcarAgent run from
OUTSIDE a chat webview still records a replayable event stream.

When the agent runs inside a chat webview, the VS Code server's
recording ``JsonPrinter`` / ``WebPrinter`` streams every event into the
``events`` table, so the chat can be reopened and replayed.  When the
agent runs OUTSIDE a chat webview — the CLI, a third-party channel
agent, or a remote webapp invocation with a non-recording printer — no
events were persisted, so the chat webview loaded a blank session even
though the task and its result were saved in ``task_history``.

``ChatSorcarAgent.run`` now synthesizes a minimal replayable event
stream (a ``prompt`` event followed by a ``result`` event) in its
``finally`` block whenever the run produced no events of its own, so
the run can still be opened and replayed in the chat webview.  When a
recording printer already persisted the full event stream, the
synthesis is skipped so events are never duplicated.

The tests drive the REAL ``ChatSorcarAgent.run`` code path against a
real temp-dir SQLite database.  The model invocation is avoided purely
through inheritance (MRO): the offline agent multiply-inherits from
``ChatSorcarAgent`` and a ``SorcarAgent`` subclass whose ``run`` returns
a canned result, so ``ChatSorcarAgent.run``'s ``super().run()`` resolves
to the canned implementation — no mocks, patches, or monkeypatching.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _flush_chat_events,
    _load_chat_events_by_task_id,
    _load_latest_chat_events_by_chat_id,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.json_printer import JsonPrinter

_CANNED_RESULT: str = yaml.dump(
    {"success": True, "summary": "all done"}, sort_keys=False,
)


class _CannedModelAgent(SorcarAgent):
    """A ``SorcarAgent`` whose ``run`` returns a canned YAML result.

    Stands in for the model-invoking ``SorcarAgent.run`` so the chat
    bookkeeping in ``ChatSorcarAgent.run`` can be exercised without a
    live model.  Optionally streams a ``result`` event through the
    printer to emulate the recording path of a real webview run.
    """

    broadcast_result_event = False

    def run(self, prompt_template: str = "", **kwargs: Any) -> str:  # type: ignore[override]
        printer = kwargs.get("printer")
        if self.broadcast_result_event and printer is not None:
            printer.broadcast({"type": "result", "text": _CANNED_RESULT})
        return _CANNED_RESULT


class _OfflineChatAgent(ChatSorcarAgent, _CannedModelAgent):
    """Runs the real ``ChatSorcarAgent.run`` bookkeeping.

    Its MRO is ``[_OfflineChatAgent, ChatSorcarAgent, _CannedModelAgent,
    SorcarAgent, ...]`` so the ``super().run()`` call inside
    ``ChatSorcarAgent.run`` dispatches to ``_CannedModelAgent.run`` — no
    model is invoked and no test double is used.
    """


def _redirect(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore(saved: tuple) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


class TestReplayEventsOutsideWebview:
    """Runs outside a chat webview must still be replayable."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_printer_run_persists_replayable_events(self) -> None:
        agent = _OfflineChatAgent("offline")
        result = agent.run(
            prompt_template="do the thing",
            model_name="canned",
            work_dir=str(self.tmpdir),
        )
        assert "all done" in result

        task_id = agent._last_task_id
        assert task_id is not None

        _flush_chat_events()
        loaded = _load_chat_events_by_task_id(task_id)
        assert loaded is not None
        events = loaded.get("events")
        assert isinstance(events, list)
        types = [str(e.get("type")) for e in events]
        # A run with no recording printer must still have a replayable
        # prompt + result so the chat webview renders the exchange.
        assert "prompt" in types, types
        assert "result" in types, types

        prompt_ev = next(e for e in events if e.get("type") == "prompt")
        assert "do the thing" in str(prompt_ev.get("text", ""))

        result_ev = next(e for e in events if e.get("type") == "result")
        assert result_ev.get("success") is True
        assert result_ev.get("summary") == "all done"
        assert str(result_ev.get("cost", "")).startswith("$")

    def test_chat_id_lookup_loads_outside_webview_run(self) -> None:
        agent = _OfflineChatAgent("offline")
        agent.run(
            prompt_template="research grills",
            model_name="canned",
            work_dir=str(self.tmpdir),
        )
        _flush_chat_events()
        # The webview's post-restart ``resumeSession`` loads a chat by
        # its chat_id; it must find the synthesized events.
        loaded = _load_latest_chat_events_by_chat_id(agent.chat_id)
        assert loaded is not None
        events = loaded.get("events")
        assert isinstance(events, list)
        assert any(e.get("type") == "result" for e in events), events

    def test_recording_printer_run_is_not_duplicated(self) -> None:
        agent = _OfflineChatAgent("offline")
        agent.broadcast_result_event = True
        printer = JsonPrinter()
        result = agent.run(
            prompt_template="do the thing",
            model_name="canned",
            work_dir=str(self.tmpdir),
            printer=printer,
        )
        assert "all done" in result

        task_id = agent._last_task_id
        assert task_id is not None
        _flush_chat_events()
        loaded = _load_chat_events_by_task_id(task_id)
        assert loaded is not None
        events = loaded.get("events")
        assert isinstance(events, list)
        # The recording printer already persisted exactly one result
        # event; the finally-block synthesis must NOT add a second
        # result (and must NOT add a synthesized prompt either).
        result_count = sum(1 for e in events if e.get("type") == "result")
        assert result_count == 1, events
        prompt_count = sum(1 for e in events if e.get("type") == "prompt")
        assert prompt_count == 0, events

    def test_persist_replay_events_is_idempotent(self) -> None:
        # Directly exercise the helper (shared by WorktreeSorcarAgent
        # via inheritance) against a real DB: a second call must not
        # duplicate the events once the task already has some.
        agent = WorktreeSorcarAgent("wt")
        agent.total_tokens_used = 5
        agent.budget_used = 0.25
        task_id, _ = _add_task("a worktree task")

        agent._persist_replay_events_if_missing(
            task_id=task_id,
            prompt="a worktree task",
            result_raw=_CANNED_RESULT,
            result_summary="all done",
        )
        agent._persist_replay_events_if_missing(
            task_id=task_id,
            prompt="a worktree task",
            result_raw=_CANNED_RESULT,
            result_summary="all done",
        )
        _flush_chat_events()
        loaded = _load_chat_events_by_task_id(task_id)
        assert loaded is not None
        events = loaded.get("events", [])
        assert isinstance(events, list)
        assert sum(1 for e in events if e.get("type") == "result") == 1, events
        assert sum(1 for e in events if e.get("type") == "prompt") == 1, events
