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

The same harness covers the other replay-vs-lifecycle races of a
sub-agent tab: a child finishing while its tab is replayed by name,
and a chat resumed into a child's tab while the parent replay loads
the children's rows.
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

    def test_a_child_finishing_mid_named_replay_is_closed(self) -> None:
        """A child's tab replayed by name (``resumeSession``) closes when it finishes mid-replay.

        The child's own ``subagentDone`` lands before the announcement
        on a page still opening the tab, which keeps no record of it
        without a tab: the replay must send the close again, as the
        parent replay does for the children it announces.
        """
        chat_id = "chat-named-replay"
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
            target=server._replay_session, args=(chat_id, sub_tab), kwargs={"task_id": sub_id},
        )
        held = Pause(
            AgentServer._emit_pending_ask.__code__, lambda local: local["tab_id"] == sub_tab,
        )
        with held:
            replay.start()
            assert held.paused.wait(30), "the replay never reached the child's pending ask"
            agent_state.unregister(sub_id)
            held.release()
            replay.join(30)
        assert not replay.is_alive(), "the replay never ended"
        frames = _tab_frames(printer.events, sub_tab)

        assert frames == [("openSubagentTab", False, None), ("subagentDone", None, None)], frames

    def _finish_before_named_announcement(self, side_channel: bool) -> list[tuple[str, Any, Any]]:
        """Replay a running child by name and finish it while its announcement is built.

        Returns the frames the replay sent.

        The replay is held while building the announcement: after the
        unlocked ``is_done`` sample, before the locked broadcast.
        """
        chat_id = "chat-named-late"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        sub_id, _ = th._add_task(
            "sub task",
            chat_id=chat_id,
            extra={"subagent": {"parent_task_id": parent_id, "side_channel": side_channel}},
        )
        _register_running(parent_id, chat_id, "tab-parent")
        _register_running(sub_id, chat_id, f"task-{parent_id}__sub_0", parent_id)
        printer = _RecordingPrinter()
        server = AgentServer(printer=printer)
        sub_tab = f"tab-parent__sub_{sub_id}"
        replay = threading.Thread(
            target=server._replay_session, args=(chat_id, sub_tab), kwargs={"task_id": sub_id},
        )
        held = Pause(
            AgentServer._subagent_announcement.__code__,
            lambda local: local["sub_tab_id"] == sub_tab,
        )
        with held:
            replay.start()
            assert held.paused.wait(30), "the replay never built the child's announcement"
            agent_state.unregister(sub_id)
            held.release()
            replay.join(30)
        assert not replay.is_alive(), "the replay never ended"
        return _tab_frames(printer.events, sub_tab)

    def test_a_child_finishing_before_the_named_announcement_is_announced_done(self) -> None:
        """The named replay's announcement carries the child's state at the moment it is broadcast.

        ``_replay_session`` samples ``is_done`` outside the lock; the
        child's release (``_release_run_state``, under that lock) and
        its ``subagentDone`` can land before ``_reopen_subagent_tab``
        broadcasts, which then said ``isDone: false`` AFTER the done
        signal — every client had to tolerate a running announcement
        for a finished child.  Re-sampled under the lock that
        broadcasts, the announcement says finished, and the replay's
        late completion check has nothing left to correct.
        """
        frames = self._finish_before_named_announcement(side_channel=False)

        assert frames == [("openSubagentTab", True, None)], frames

    def test_a_side_channel_finishing_before_its_named_announcement_is_closed(self) -> None:
        """A side channel (the /ask answerer) finishing in that window is closed, not announced."""
        frames = self._finish_before_named_announcement(side_channel=True)

        assert frames == [("subagentDone", None, None)], frames

    def test_a_child_tab_rebound_to_another_chat_mid_replay_keeps_that_chat(self) -> None:
        """A descendant replay yields to a chat resumed into the child's tab meanwhile.

        The parent replay loads its children's rows outside the lock;
        a chat resumed into a child's tab in that window binds the tab
        to the chat.  The old replay must then neither announce the
        child into the tab, nor subscribe the tab to the child, nor
        follow the chat's transcript with the offer of the child's.
        """
        chat_a, chat_b = "chat-A", "chat-B"
        parent_id, _ = th._add_task("parent task", chat_id=chat_a)
        sub_id, _ = th._add_task(
            "sub task", chat_id=chat_a, extra={"subagent": {"parent_task_id": parent_id}},
        )
        th._add_task("b task", chat_id=chat_b)
        _register_running(parent_id, chat_a, "tab-parent")
        _register_running(sub_id, chat_a, f"task-{parent_id}__sub_0", parent_id)
        printer = _RecordingPrinter()
        server = AgentServer(printer=printer)
        sub_tab = f"tab-parent__sub_{sub_id}"
        replay = threading.Thread(
            target=server._open_persisted_subagent_tabs,
            kwargs={
                "parent_task_id": parent_id,
                "parent_tab_id": "tab-parent",
                "conn_id": "",
            },
        )
        held = Pause(
            th._load_subagent_rows_by_parent_task_id.__code__,
            lambda local: local.get("parent_task_id") == parent_id,
        )
        with held:
            replay.start()
            assert held.paused.wait(30), "the replay never loaded the children's rows"
            server._replay_session(chat_b, sub_tab)
            held.release()
            replay.join(30)
        assert not replay.is_alive(), "the replay never ended"
        frames = [
            (ev["type"], ev.get("chat_id"))
            for ev in printer.events
            if sub_tab in (ev.get("tab_id"), ev.get("tabId"))
            and ev["type"] in ("openSubagentTab", "task_events", "replay_deferred", "subagentDone")
        ]

        assert frames == [("task_events", chat_b)], frames
        assert sub_tab not in printer.viewer_tab_ids(sub_id)
        assert server._tab_chat_views.get(sub_tab) == chat_b
