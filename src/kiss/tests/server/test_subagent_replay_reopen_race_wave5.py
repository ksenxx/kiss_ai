# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A sub-agent that finishes while its tab is replayed is closed by that replay.

A parent replay (``_open_persisted_subagent_tabs``) announces a running
child, sends its transcript offer, and then checks again whether the
child finished meanwhile, telling the clients ``subagentDone`` if so:
the child's own ``subagentDone`` may have preceded the announcement on
a page still opening the tab.  A replay of the child's tab by name
(``resumeSession``) does the same, and announces nothing for a child
that is over when the replay gets to its announcement.  The replay is
held at a step of the real code path (``Pause``, a trace hook) while
the child finishes.  A real ``AgentServer``, real agent states and a
real history database; nothing is mocked.

The same harness covers a chat resumed into a child's tab while the
parent replay loads the children's rows, and another sub-agent replayed
into a child's tab while that child's own end is on its way, before the
end reaches the tab or right after it.
"""

from __future__ import annotations

import threading
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import _notify_subagent_done
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
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


def _tab_frames(events: list[dict[str, Any]], tab_id: str) -> list[tuple[str, str]]:
    """The (type, task id) of every sub-agent tab frame about *tab_id*."""
    return [
        (ev["type"], ev["task_id"])
        for ev in events
        if ev.get("tab_id") == tab_id and ev.get("type") in ("openSubagentTab", "subagentDone")
    ]


class TestSubagentFinishingMidReplay(_DbRedirectBase):
    """A replay closes the tab of a child that finishes while it runs."""

    def test_a_child_finishing_mid_replay_is_closed(self) -> None:
        """The parent replay closes the tab it announced running."""
        chat_id = "chat-replay-finish"
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
            held.release()
            replay.join(30)
        assert not replay.is_alive(), "the replay never ended"
        frames = _tab_frames(printer.events, sub_tab)

        assert frames == [("openSubagentTab", sub_id), ("subagentDone", sub_id)], frames

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

        assert frames == [("openSubagentTab", sub_id), ("subagentDone", sub_id)], frames

    def test_a_child_over_before_its_named_announcement_is_told_done_only(self) -> None:
        """A child over when the named replay gets to its announcement is not announced.

        ``_replay_session`` attaches the tab while the child runs; the
        child's release (``_release_run_state``) and its ``subagentDone``
        can land before ``_announce_or_end_subagent_tab`` decides.  It samples
        and broadcasts under the lock the release runs under, so the
        end is the only frame about the tab: no announcement, status or
        transcript follows a done signal.  The daemon forgets the tab
        with that answer (no page reports that it closed a sub-agent's
        tab), so the answer goes to every connection: the tab was a
        viewer of the child, and the child's own end no longer finds it.
        """
        chat_id = "chat-named-late"
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
            target=server._replay_session,
            args=(chat_id, sub_tab),
            kwargs={"task_id": sub_id, "conn_id": "conn-asking"},
        )
        held = Pause(
            AgentServer._announce_or_end_subagent_tab.__code__,
            lambda local: local["tab_id"] == sub_tab,
        )
        with held:
            replay.start()
            assert held.paused.wait(30), "the replay never got to the child's announcement"
            agent_state.unregister(sub_id)
            held.release()
            replay.join(30)
        assert not replay.is_alive(), "the replay never ended"
        about_tab = [
            (ev["type"], ev.get("task_id"), ev.get("connId"))
            for ev in printer.events
            if sub_tab in (ev.get("tab_id"), ev.get("tabId"))
        ]

        assert about_tab == [("subagentDone", sub_id, None)], about_tab
        assert printer.tasks_for_tab(sub_tab) == [], "the ended tab is still a viewer"
        assert sub_tab not in server._tab_opened_task_ids, "the ended tab is still pinned"

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

    def test_a_childs_end_leaves_the_sub_agent_replayed_into_its_tab_meanwhile(self) -> None:
        """A child's end closes no other sub-agent replayed into its tab since the end began.

        The ending child's thread samples the tabs watching it, then
        asks the daemon to end each.  Another running sub-agent replayed
        into one of them in between owns the tab: it is told nothing of
        the first child's end, which would close it on every page, and
        keeps its pin and its subscription.
        """
        chat_id = "chat-end-vs-takeover"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        subs = [
            th._add_task(
                name, chat_id=chat_id, extra={"subagent": {"parent_task_id": parent_id}},
            )[0]
            for name in ("first sub", "second sub")
        ]
        _register_running(parent_id, chat_id, "tab-parent")
        for idx, sub_id in enumerate(subs):
            _register_running(sub_id, chat_id, f"task-{parent_id}__sub_{idx}", parent_id)
        first, second = subs
        printer = _RecordingPrinter()
        server = AgentServer(printer=printer)
        sub_tab = f"tab-parent__sub_{first}"
        server._replay_session(chat_id, sub_tab, task_id=first)
        agent_state.unregister(first)
        ending = threading.Thread(
            target=_notify_subagent_done, args=(printer, first, "", "", "tab-parent"),
        )
        held = Pause(
            AgentServer._end_subagent_view.__code__, lambda local: local["tab_id"] == sub_tab,
        )
        with held:
            ending.start()
            assert held.paused.wait(30), "the child's end never reached its tab"
            server._replay_session(chat_id, sub_tab, task_id=second)
            held.release()
            ending.join(30)
        assert not ending.is_alive(), "the child's end never returned"
        frames = _tab_frames(printer.events, sub_tab)

        assert frames == [("openSubagentTab", first), ("openSubagentTab", second)], frames
        assert printer.tasks_for_tab(sub_tab) == [second]
        assert server._tab_opened_task_ids.get(sub_tab) == second

    def test_a_childs_end_restores_no_picker_of_a_chat_resumed_into_its_tab_since(self) -> None:
        """A child's end hands its tab's model picker back before the tab can be taken over.

        The end tells the tab and restores its picker in one step.  A
        running chat resumed into the tab from then on shows the model
        its agent picked; a restore sent after the takeover would put
        the ended child's model over it, with no later event to fix it.
        """
        chat_id = "chat-end-vs-model"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        sub_id, _ = th._add_task(
            "sub task", chat_id=chat_id, extra={"subagent": {"parent_task_id": parent_id}},
        )
        other_id, other_chat = th._add_task("another chat's task")
        _register_running(parent_id, chat_id, "tab-parent")
        _register_running(sub_id, chat_id, f"task-{parent_id}__sub_0", parent_id)
        _register_running(other_id, other_chat, "tab-other")
        printer = _RecordingPrinter()
        server = AgentServer(printer=printer)
        sub_tab = f"tab-parent__sub_{sub_id}"
        server._replay_session(chat_id, sub_tab, task_id=sub_id)
        printer.broadcast_agent_model_pick("child-model", "", sub_id)
        printer.broadcast_agent_model_pick("other-model", "tab-other", other_id)
        agent_state.unregister(sub_id)
        ending = threading.Thread(
            target=_notify_subagent_done,
            args=(printer, sub_id, "", "launch-model", "tab-parent"),
        )
        takeover = threading.Thread(target=server._replay_session, args=(other_chat, sub_tab))
        held = Pause(
            JsonPrinter.restore_model_pick.__code__, lambda local: local["tab_id"] == sub_tab,
        )
        with held:
            ending.start()
            assert held.paused.wait(30), "the child's end never got to the tab's picker"
            takeover.start()
            takeover.join(0.5)  # over before the release only when the restore is a later step
            held.release()
            ending.join(30)
            takeover.join(30)
        assert not ending.is_alive(), "the child's end never returned"
        assert not takeover.is_alive(), "the takeover never returned"
        picks = [
            (ev["model"], ev["source"])
            for ev in printer.events
            if ev["type"] == "modelPick" and ev.get("tabId") == sub_tab
        ]

        assert picks[0] == ("child-model", "agent"), picks
        assert picks[-1] == ("other-model", "agent"), picks
        assert sub_tab in printer._model_override_tabs, "the tab's picker is no longer the agent's"
