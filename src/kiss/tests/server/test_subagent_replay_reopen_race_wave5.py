# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A replay's late completion signal never closes a tab reopened since.

A parent replay (``_open_persisted_subagent_tabs``) announces a running
child, sends its transcript offer, and then checks again whether the
child finished meanwhile, telling the clients ``subagentDone`` if so.
If the child finishes in that window and a user reopens its tab
(``resumeSession`` with ``reopen: true``, which announces the finished
tab ``reopened`` on every surface), that late ``subagentDone`` used to
follow the reopening: every page closes the tab on ``subagentDone`` and
sends ``closeTab``, which the daemon mirrors to every surface, so the
user's reopening was undone.  The replay is held at its pending-ask
step (``Pause``, a trace hook on the real code path) while the child
finishes and the reopening runs.  A real ``AgentServer``, real agent
states and a real history database; nothing is mocked.
"""

from __future__ import annotations

import threading
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.pause import Pause
from kiss.tests.server.test_subagent_stop_review_fixes import (
    _DbRedirectBase,
    _RecordingPrinter,
)


def _register_running(task_id: str, chat_id: str, tab_id: str, parent_id: str = "") -> None:
    """Register a live agent state for *task_id*, as a running task has."""
    agent_state.register(
        agent_state.AgentState(
            task_id,
            agent=ChatSorcarAgent("agent"),
            chat_id=chat_id,
            tab_id=tab_id,
            parent_task_id=parent_id or None,
            stop_event=threading.Event(),
            is_task_active=True,
        )
    )


def _tab_frames(events: list[dict[str, Any]], tab_id: str) -> list[tuple[str, Any, Any]]:
    """The (type, isDone, reopened) of every sub-agent tab frame about *tab_id*."""
    return [
        (ev["type"], ev.get("isDone"), ev.get("reopened"))
        for ev in events
        if ev.get("tab_id") == tab_id and ev.get("type") in ("openSubagentTab", "subagentDone")
    ]


class TestReplayCompletionAfterReopen(_DbRedirectBase):
    """The replay's completion check yields to a reopening recorded before it."""

    def _finish_mid_replay(self, reopen: bool) -> list[tuple[str, Any, Any]]:
        """Replay a running child, finish it (and reopen its tab) mid-replay; return its frames."""
        chat_id = "chat-replay-reopen"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        sub_id, _ = th._add_task(
            "sub task", chat_id=chat_id, extra={"subagent": {"parent_task_id": parent_id}},
        )
        _register_running(parent_id, chat_id, "tab-parent")
        _register_running(sub_id, chat_id, f"task-{parent_id}__sub_0", parent_id)
        printer = _RecordingPrinter()
        server = AgentServer(printer=printer)
        sub_tab = f"tab-parent__sub_{sub_id}"
        replay = threading.Thread(
            target=server._open_persisted_subagent_tabs,
            kwargs={
                "parent_task_id": parent_id,
                "parent_tab_id": "tab-parent",
                "conn_id": "conn-new",
                "transcripts": False,
            },
        )
        held = Pause(
            AgentServer._emit_pending_ask.__code__, lambda local: local["tab_id"] == sub_tab,
        )
        with held:
            replay.start()
            assert held.paused.wait(30), "the replay never reached the child's pending ask"
            agent_state.unregister(sub_id)
            if reopen:
                server._replay_session(chat_id, sub_tab, task_id=sub_id, reopen=True)
            held.release()
            replay.join(30)
        assert not replay.is_alive(), "the replay never ended"
        return _tab_frames(printer.events, sub_tab)

    def test_a_child_finishing_and_reopened_mid_replay_stays_open(self) -> None:
        """No ``subagentDone`` from the replay follows the reopened announcement."""
        frames = self._finish_mid_replay(reopen=True)

        assert frames == [
            ("openSubagentTab", False, None),
            ("openSubagentTab", True, True),
        ], frames

    def test_a_child_finishing_mid_replay_is_closed(self) -> None:
        """Without a reopening, the replay closes the tab it announced running."""
        frames = self._finish_mid_replay(reopen=False)

        assert frames == [("openSubagentTab", False, None), ("subagentDone", None, None)], frames
