# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: restoring a parent agent that fanned out
sub-agents (``run_parallel``) after a daemon restart.

After a daemon restart the chat page restores its tabs from persisted
state and sends ``resumeSession {chatId, tabId}`` — chat id only, no
task id — for each restored tab (see ``init()`` in media/main.js).
For a parent agent
that spawned sub-agents the restored parent tab must:

1. load the PARENT's own chat events into the parent tab (NOT the
   events of the most recently persisted sub-agent row, which shares
   the parent's chat_id and was inserted later),
2. NOT be converted into a sub-agent tab, and
3. end the tab of every persisted sub-agent (``subagentDone``): none
   of them runs after a restart, and a sub-agent has a tab only while
   it runs.
"""

from __future__ import annotations

import shutil
import tempfile
import threading

import kiss.agents.sorcar.persistence as th
from kiss.server.server import AgentServer
from kiss.tests.agents.sorcar.test_restore_tabs_with_subagents import (  # noqa: F401
    _redirect,
    _restore,
    _seed_parent_with_subagents,
)


def _make_server() -> tuple[AgentServer, list[dict]]:
    """Create a AgentServer whose broadcasts go into an in-memory list."""
    server = AgentServer()
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
    task id) that the chat page sends for a restored parent tab."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        th._close_thread_db()
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _resume_restored_parent_tab(
        self,
    ) -> tuple[AgentServer, list[dict], str, list[str], str]:
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

    def test_persisted_subagents_are_ended_after_the_parents_replay(self) -> None:
        """No persisted sub-agent row gets a tab, a transcript or an
        offer of one: each one's ``subagentDone`` (its tab and task)
        follows the parent's own events and closes the tab a page kept
        from before the restart."""
        _, events, _, sub_ids, parent_tab_id = self._resume_restored_parent_tab()
        told = [
            e for e in events
            if e.get("type") in ("openSubagentTab", "replay_deferred", "subagentDone")
        ]
        assert [e["type"] for e in told] == ["subagentDone"] * len(sub_ids), f"events={events}"
        assert [(e["tab_id"], e["task_id"]) for e in told] == [
            (f"{parent_tab_id}__sub_{sub_id}", sub_id) for sub_id in sub_ids
        ]
        replayed = [e["tabId"] for e in events if e.get("type") == "task_events"]
        assert replayed == [parent_tab_id]
        assert events.index(told[0]) > next(
            i for i, e in enumerate(events) if e.get("type") == "task_events"
        )
