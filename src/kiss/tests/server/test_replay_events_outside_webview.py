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

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _flush_chat_events,
    _load_chat_events_by_task_id,
)
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.test_replay_events_outside_webview import (  # noqa: F401
    _CANNED_RESULT,
    _CannedModelAgent,
    _OfflineChatAgent,
    _redirect,
    _restore,
)


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
        result_count = sum(1 for e in events if e.get("type") == "result")
        assert result_count == 1, events
        prompt_count = sum(1 for e in events if e.get("type") == "prompt")
        assert prompt_count == 0, events
