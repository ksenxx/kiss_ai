# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: restoring a parent agent that fanned out
sub-agents (``run_parallel``) after a VS Code restart.

After a VS Code restart the webview restores its tabs from persisted
state and sends ``resumeSession {chatId, tabId}`` — chat id only, no
task id — for each restored tab (see ``init()`` in media/main.js and
the ``ready`` handler in SorcarSidebarView.ts).  For a parent agent
that spawned sub-agents the restored parent tab must:

1. load the PARENT's own chat events into the parent tab (NOT the
   events of the most recently persisted sub-agent row, which shares
   the parent's chat_id and was inserted later),
2. NOT be converted into a sub-agent tab, and
3. reopen every persisted sub-agent row in its own sub-agent tab
   (``openSubagentTab`` + ``task_events``) anchored to the parent tab
   so the restored layout mirrors the live execution layout.
"""

from __future__ import annotations

import shutil
import tempfile
import threading

import kiss.agents.sorcar.persistence as th
from kiss.server.server import VSCodeServer
from kiss.tests.agents.sorcar.test_restore_tabs_with_subagents import (  # noqa: F401
    _redirect,
    _restore,
    _seed_parent_with_subagents,
)


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer whose broadcasts go into an in-memory list."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        ev = server.printer._inject_task_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        with lock:
            events.append(ev)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestRestoreParentTabWithSubagents:
    """Simulates the post-restart ``resumeSession`` (chat id only, no
    task id) that the webview sends for a restored parent tab."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _resume_restored_parent_tab(
        self,
    ) -> tuple[VSCodeServer, list[dict], str, list[str], str]:
        chat_id = "chat-restart-1"
        parent_id, sub_ids = _seed_parent_with_subagents(chat_id)
        server, events = _make_server()
        parent_tab_id = "tab-restored-parent"
        server._cmd_resume_session({
            "type": "resumeSession",
            "chatId": chat_id,
            "tabId": parent_tab_id,
        })
        return server, events, parent_id, sub_ids, parent_tab_id

    def test_parent_tab_loads_its_own_events(self) -> None:
        """The restored parent tab must replay the PARENT's events,
        not the events of the latest sub-agent row in the chat."""
        _, events, parent_id, _, parent_tab_id = (
            self._resume_restored_parent_tab()
        )
        parent_replays = [
            e for e in events
            if e.get("type") == "task_events"
            and e.get("tabId") == parent_tab_id
        ]
        assert len(parent_replays) == 1, f"events={events}"
        replay = parent_replays[0]
        assert replay["task_id"] == parent_id
        assert replay["task"] == "parent task with fanout"
        assert any(
            ev.get("type") == "text_delta" and ev.get("text") == "parent-event"
            for ev in replay["events"]
        ), f"parent tab replayed wrong events: {replay['events']}"

    def test_parent_tab_is_not_converted_into_subagent_tab(self) -> None:
        """No ``openSubagentTab`` may target the restored parent tab id."""
        _, events, _, _, parent_tab_id = self._resume_restored_parent_tab()
        assert not any(
            e.get("type") == "openSubagentTab"
            and e.get("tab_id") == parent_tab_id
            for e in events
        ), f"parent tab was converted into a sub-agent tab: {events}"

    def test_subagent_tabs_reopen_right_of_parent_with_own_events(
        self,
    ) -> None:
        """Each persisted sub-agent row reopens in its own sub-agent
        tab anchored to the parent tab, replaying its own events."""
        _, events, _, sub_ids, parent_tab_id = (
            self._resume_restored_parent_tab()
        )
        opens = [e for e in events if e.get("type") == "openSubagentTab"]
        assert len(opens) == len(sub_ids), f"events={events}"
        for idx, (op, sub_id) in enumerate(zip(opens, sub_ids)):
            assert op["parent_tab_id"] == parent_tab_id
            assert op["isSubagentTab"] is True
            assert op["isDone"] is True
            assert op["description"] == f"sub task {idx}"
            sub_tab_id = op["tab_id"]
            assert sub_tab_id != parent_tab_id
            sub_replays = [
                e for e in events
                if e.get("type") == "task_events"
                and e.get("tabId") == sub_tab_id
            ]
            assert len(sub_replays) == 1
            assert sub_replays[0]["task_id"] == sub_id
            assert any(
                ev.get("type") == "text_delta"
                and ev.get("text") == f"sub-event-{idx}"
                for ev in sub_replays[0]["events"]
            )

    def test_parent_events_replay_before_subagent_tabs_open(self) -> None:
        """The parent tab's own events must be replayed first; the
        sub-agent tabs open after (to the right of) the parent."""
        _, events, _, _, parent_tab_id = self._resume_restored_parent_tab()
        parent_idx = next(
            i for i, e in enumerate(events)
            if e.get("type") == "task_events"
            and e.get("tabId") == parent_tab_id
        )
        open_idxs = [
            i for i, e in enumerate(events)
            if e.get("type") == "openSubagentTab"
        ]
        assert open_idxs, f"no sub-agent tabs opened: {events}"
        assert all(parent_idx < i for i in open_idxs)
