# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A replay superseded by another chat's rebinding of the same tab sends no transcript.

Audit 2026-11-01 (V3 F2): two surfaces can resume different chats into
the same shared tab at once.  ``AgentServer._replay_session`` publishes
and commits its binding under ownership checks, but the tail that
follows — the tab's status, the ``task_events`` transcript, the viewer
attach finalization — was unconditional.  Replay A, held after its
publication, let replay B publish chat B and deliver B's transcript;
A then broadcast A's transcript last, so every client showed chat A in
a tab the registry and ``_tab_chat_views`` bound to chat B, and the
next reconnect (or another surface) disagreed with what was displayed.
The tail is now delivered only while the tab's backend chat view is
still this replay's chat, checked under ``_state_lock`` (the status
helpers) and after the replay's FIFO send reservation (the
transcript), so a later publication's own frames always land last.

Real ``AgentServer`` with a private tab registry over a real sqlite
file; replay A runs on a thread held at the entry of the real
``_broadcast_replay_status`` with a ``sys.settrace`` hook
(:class:`Pause`) while replay B runs to completion.  The replies are
captured from the printer's broadcast.  No mocks.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
import kiss.server.server as server_module
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.agents.sorcar.test_steer_inputs import _redirect, _restore
from kiss.tests.local_ws import (
    LocalReader,
    LocalWriter,
    _generate_self_signed_cert,
    open_local_connection,
)
from kiss.tests.persistence_helpers import close_db
from kiss.tests.server.pause import Pause

_TAB = "shared-tab"


def _persist_child(parent_task_id: str, chat_id: str, side_channel: bool = False) -> str:
    """Persist a sub-agent row of *parent_task_id* in *chat_id* with one event; its task id."""
    child, _ = th._add_task(
        "child task",
        chat_id,
        extra={"subagent": {"parent_task_id": parent_task_id, "side_channel": side_channel}},
    )
    th._append_chat_event({"type": "text_delta", "text": "child transcript"}, task_id=child)
    return child


def _register_running(
    case: unittest.TestCase, task_id: str, chat_id: str, parent_task_id: str | None = None,
) -> None:
    """Register *task_id* as a running task of *chat_id* (*parent_task_id*'s sub-agent)."""
    state = agent_state.AgentState(
        task_id, chat_id=chat_id, tab_id="owner-" + task_id, parent_task_id=parent_task_id,
        server_owned=True, is_task_active=True,
    )
    agent_state.register(state)
    case.addCleanup(agent_state.unregister, task_id, state)


class TestSupersededReplay(unittest.TestCase):
    """The last transcript a shared tab shows is the chat the registry binds."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="kiss-superseded-replay-")
        self.saved = _redirect(self.tmp)
        self.server = AgentServer()
        self.server.use_private_tab_registry(Path(self.tmp) / "tabs.json")
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]
        self.task_a, self.chat_a = th._add_task("chat A task")
        self.task_b, self.chat_b = th._add_task("chat B task")
        th._append_chat_event({"type": "text_delta", "text": "A transcript"}, task_id=self.task_a)
        th._append_chat_event({"type": "text_delta", "text": "B transcript"}, task_id=self.task_b)

    def tearDown(self) -> None:
        close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _frames(self, kind: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == kind and e.get("tabId") == _TAB]

    def _replay(
        self, chat_id: str, task_id: str, conn_id: str, failures: list, *, reopen: bool = False,
    ) -> None:
        try:
            self.server._replay_session(
                chat_id, _TAB, task_id=task_id, conn_id=conn_id, reopen=reopen,
            )
        except BaseException as exc:  # noqa: BLE001 - reported by the test thread
            failures.append(exc)
        finally:
            th._close_thread_db()

    def _defer(self, chat_id: str, task_id: str, conn_id: str, failures: list) -> None:
        try:
            self.server._defer_replay(chat_id, _TAB, task_id, conn_id)
        except BaseException as exc:  # noqa: BLE001 - reported by the test thread
            failures.append(exc)
        finally:
            th._close_thread_db()

    def _walk(self, parent_task_id: str, parent_tab_id: str, conn_id: str, failures: list) -> None:
        """Walk the sub-agents of *parent_task_id*, scoped to *conn_id* when non-empty."""
        try:
            self.server._open_persisted_subagent_tabs(
                parent_task_id=parent_task_id, parent_tab_id=parent_tab_id, conn_id=conn_id,
            )
        except BaseException as exc:  # noqa: BLE001 - reported by the test thread
            failures.append(exc)
        finally:
            th._close_thread_db()

    def _about(self, tab_id: str, since: int = 0) -> list[str]:
        """The types of the frames about *tab_id* from *since* on."""
        return [
            ev["type"] for ev in self.events[since:]
            if tab_id in (ev.get("tab_id"), ev.get("tabId"))
        ]

    def _race(
        self,
        first: tuple[str, str, str],
        second: tuple[str, str, str],
        held_at: Any = AgentServer._broadcast_replay_status.__code__,
        first_call: Callable[..., None] | None = None,
    ) -> None:
        """Run replay *first* on a thread held at *held_at* while *second* completes."""
        failures: list[BaseException] = []
        with Pause(held_at, lambda _locals: threading.current_thread().name == "held") as pause:
            held = threading.Thread(
                target=first_call or self._replay, args=(*first, failures), name="held",
            )
            held.start()
            self.assertTrue(pause.paused.wait(10), "replay A never reached the hold")
            self.server._replay_session(second[0], _TAB, task_id=second[1], conn_id=second[2])
            pause.release()
            held.join(10)
        self.assertFalse(held.is_alive())
        self.assertEqual(failures, [])

    def _live(self, task_id: str, chat_id: str) -> None:
        """Register *task_id* as a running task of *chat_id* launched from its own tab."""
        _register_running(self, task_id, chat_id)

    def test_superseded_replay_sends_neither_status_nor_transcript(self) -> None:
        self._race((self.chat_a, self.task_a, ""), (self.chat_b, self.task_b, ""))
        self.assertEqual(self.server.tab_registry.bindings()[_TAB], self.chat_b)
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_b)
        transcripts = self._frames("task_events")
        self.assertEqual(
            [ev["chat_id"] for ev in transcripts], [self.chat_b],
            "a transcript of the superseded chat was delivered to the tab",
        )
        self.assertEqual(
            [ev.get("text") for ev in transcripts[-1]["events"] if ev.get("type") == "text_delta"],
            ["B transcript"],
        )
        self.assertEqual(len(self._frames("status")), 1)

    def test_undisturbed_replay_delivers_status_and_transcript(self) -> None:
        self.server._replay_session(self.chat_a, _TAB, task_id=self.task_a)
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_a)
        self.assertEqual([ev["chat_id"] for ev in self._frames("task_events")], [self.chat_a])
        self.assertEqual([ev["running"] for ev in self._frames("status")], [False])

    def test_two_connections_resuming_the_same_chat_both_get_the_transcript(self) -> None:
        """Reconnect replays of one chat by two clients do not supersede each other."""
        # A reconnect resumes a registry-bound tab (A18-3: a scoped
        # replay refreshes that binding and publishes nothing).
        self.server._replay_session(self.chat_a, _TAB, task_id=self.task_a)
        self.events.clear()
        self._race((self.chat_a, self.task_a, "conn-1"), (self.chat_a, self.task_a, "conn-2"))
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_a)
        self.assertEqual(
            sorted(ev["connId"] for ev in self._frames("task_events")), ["conn-1", "conn-2"],
        )
        self.assertEqual(
            sorted(ev["connId"] for ev in self._frames("status")), ["conn-1", "conn-2"],
        )

    def test_late_winning_publication_owns_subscriptions_and_pin(self) -> None:
        """Audit V5 F1: the replay that publishes last keeps only ITS live subscription/pin."""
        self._live(self.task_a, self.chat_a)
        self._live(self.task_b, self.chat_b)
        # A attached, B attached/published/sent, then A publishes (and wins).
        self._race(
            (self.chat_a, self.task_a, ""), (self.chat_b, self.task_b, ""),
            held_at=AgentServer._publish_replay_reopen.__code__,
        )
        self.assertEqual(self.server.tab_registry.bindings()[_TAB], self.chat_a)
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_a)
        self.assertEqual(
            self.server.printer.tasks_for_tab(_TAB), [self.task_a],
            "the tab stayed subscribed to the superseded chat's live task",
        )
        self.assertEqual(self.server._tab_opened_task_ids.get(_TAB), self.task_a)

    def test_older_task_of_the_same_chat_cannot_overwrite_a_newer_selection(self) -> None:
        """Audit V5 F2: supersession is by chat AND selected task, not chat alone."""
        task_a2, _ = th._add_task("chat A second task", self.chat_a)
        th._append_chat_event({"type": "text_delta", "text": "A2 transcript"}, task_id=task_a2)
        self._race((self.chat_a, self.task_a, ""), (self.chat_a, task_a2, ""))
        self.assertEqual(self.server._tab_opened_task_ids.get(_TAB), task_a2)
        self.assertEqual(
            [ev["task_id"] for ev in self._frames("task_events")], [task_a2],
            "the superseded task's transcript was delivered after the selected one",
        )
        self.assertEqual(len(self._frames("status")), 1)

    def test_neighbour_read_in_flight_across_a_rebind_is_discarded(self) -> None:
        """Audit V5 F3: an adjacent-task reply for the chat the tab no longer shows is dropped."""
        task_a2, _ = th._add_task("chat A second task", self.chat_a)
        self.server._replay_session(self.chat_a, _TAB, task_id=task_a2)

        def neighbour(chat_id: str, task_id: str, _conn: str, failures: list) -> None:
            try:
                self.server._get_adjacent_task(chat_id, task_id, "prev", _TAB)
            except BaseException as exc:  # noqa: BLE001 - reported by the test thread
                failures.append(exc)
            finally:
                th._close_thread_db()

        self._race(
            (self.chat_a, task_a2, ""), (self.chat_b, self.task_b, ""),
            held_at=th._get_adjacent_task_by_chat_id.__code__, first_call=neighbour,
        )
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_b)
        self.assertEqual(
            self._frames("adjacent_task_events"), [],
            "the previous task of the superseded chat was spliced into the rebound tab",
        )
        self.server._get_adjacent_task(self.chat_a, task_a2, "prev", "unbound-tab")
        self.assertEqual(
            [ev["task_id"] for ev in self.events if ev.get("type") == "adjacent_task_events"],
            [self.task_a],
        )
        # The same when the rebind lands between the command's resolution
        # of the tab's chat and the read's capture of the tab's view
        # (review 2026-11-04 R4): the fence then describes the new chat
        # while the read is the old chat's.
        self.server._replay_session(self.chat_a, _TAB, task_id=task_a2)
        self.events.clear()

        def command(_chat_id: str, task_id: str, _conn: str, failures: list) -> None:
            try:
                self.server._cmd_get_adjacent_task(
                    {"tabId": _TAB, "taskId": task_id, "direction": "prev"},
                )
            except BaseException as exc:  # noqa: BLE001 - reported by the test thread
                failures.append(exc)
            finally:
                th._close_thread_db()

        self._race(
            (self.chat_a, task_a2, ""), (self.chat_b, self.task_b, ""),
            held_at=AgentServer._get_adjacent_task.__code__, first_call=command,
        )
        self.assertEqual(self.server.tab_registry.bindings()[_TAB], self.chat_b)
        self.assertEqual(
            [ev["task_id"] for ev in self._frames("adjacent_task_events")], [],
            "the neighbour of the chat the command resolved was spliced into the rebound tab",
        )
        # And when the tab's selected task moves within the chat in that
        # interval (review 2026-11-04 R2): the predecessor of A2 would
        # follow A3's transcript, and the page, prepending it as the
        # oldest task shown, would skip A2 on the next scroll.
        task_a3, _ = th._add_task("chat A third task", self.chat_a)
        self.server._replay_session(self.chat_a, _TAB, task_id=task_a2)
        self.events.clear()
        self._race(
            (self.chat_a, task_a2, ""), (self.chat_a, task_a3, ""),
            held_at=AgentServer._get_adjacent_task.__code__, first_call=command,
        )
        self.assertEqual(self.server._tab_opened_task_ids.get(_TAB), task_a3)
        self.assertEqual([ev["task_id"] for ev in self._frames("task_events")], [task_a3])
        self.assertEqual(
            [ev["task_id"] for ev in self._frames("adjacent_task_events")], [],
            "the predecessor of the task the tab no longer selects followed the new selection",
        )

    def test_stale_deferred_replay_does_not_reattach_the_old_chat(self) -> None:
        """Audit V5 F4: a ready-time deferred replay of a rebound tab stands down."""
        self._live(self.task_a, self.chat_a)
        self._live(self.task_b, self.chat_b)
        self.server._replay_session(self.chat_a, _TAB, task_id=self.task_a)
        self.events.clear()
        # The ready snapshot named (chat A, task A); the tab is rebound to
        # chat B before the deferred replay acts on it.
        self._race(
            (self.chat_a, self.task_a, "reconnecting"), (self.chat_b, self.task_b, ""),
            held_at=AgentServer._defer_replay.__code__, first_call=self._defer,
        )
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_b)
        self.assertEqual(
            self.server.printer.tasks_for_tab(_TAB), [self.task_b],
            "the stale deferred replay subscribed the rebound tab to the old chat's task",
        )
        self.assertEqual(self._frames("replay_deferred"), [])
        self.assertEqual(self.server._tab_opened_task_ids.get(_TAB), self.task_b)
        self.assertEqual([ev.get("connId") for ev in self._frames("status")], [None])

    def test_deferred_replay_between_a_publication_and_its_commit_is_answered(self) -> None:
        """A deferral of the view a replay just published (not yet committed) gets its offer."""
        self.server._replay_session(self.chat_a, _TAB, task_id=self.task_a)
        self.events.clear()

        # Replay B is held after its registry publication, before its commit;
        # a second client's ready defers the tab as the registry names it: (B, B).
        failures: list[BaseException] = []
        with Pause(
            AgentServer._commit_replay_publication.__code__,
            lambda _locals: threading.current_thread().name == "held",
        ) as pause:
            held = threading.Thread(
                target=self._replay, args=(self.chat_b, self.task_b, "", failures), name="held",
            )
            held.start()
            self.assertTrue(pause.paused.wait(10), "replay B never reached its commit")
            self.server._defer_replay(self.chat_b, _TAB, self.task_b, "second-client")
            pause.release()
            held.join(10)
        self.assertFalse(held.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(
            [(ev["chatId"], ev["connId"]) for ev in self._frames("replay_deferred")],
            [(self.chat_b, "second-client")],
            "the deferral landing between B's publication and commit lost its offer",
        )
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_b)
        self.assertEqual(self.server._tab_opened_task_ids.get(_TAB), self.task_b)

    def _child_of(self, parent_task_id: str, chat_id: str) -> str:
        """Persist a sub-agent row of *parent_task_id* in *chat_id* with one event."""
        return _persist_child(parent_task_id, chat_id)

    def test_named_subagent_replay_superseded_by_a_chat_takeover_sends_nothing(self) -> None:
        """A18-1 (astra) / A18-2 (fable): a sub-agent view is fenced like a chat view.

        A sub-agent replay publishes no chat binding, so its status and
        transcript used to be unconditional: announced, then held, it
        let a chat takeover of the same tab publish and deliver chat B,
        and then sent the child's running status and transcript last —
        into a tab the registry and ``_tab_chat_views`` bind to B.
        """
        child = self._child_of(self.task_a, self.chat_a)
        state = agent_state.AgentState(
            child, chat_id=self.chat_a, tab_id="owner-" + child, parent_task_id=self.task_a,
            server_owned=True, is_task_active=True,
        )
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, child, state)
        # The child's replay is held after its announcement, at the
        # running status that precedes its transcript.
        self._race(
            (self.chat_a, child, "conn-1"), (self.chat_b, self.task_b, ""),
            held_at=AgentServer._broadcast_viewer_running.__code__,
        )
        self.assertEqual(self.server.tab_registry.bindings()[_TAB], self.chat_b)
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_b)
        self.assertEqual(
            [ev["task_id"] for ev in self._frames("task_events")], [self.task_b],
            "the superseded sub-agent's transcript was delivered after the chat takeover",
        )
        self.assertEqual(
            [ev["running"] for ev in self._frames("status")], [False],
            "the superseded sub-agent's running status was delivered after the chat takeover",
        )
        self.assertEqual(self.server.printer.tasks_for_tab(_TAB), [])

    def _subagent_frames(self, since: int = 0) -> list[tuple[str, Any]]:
        """The ``(type, isDone)`` of the sub-agent tab frames about ``_TAB`` from *since* on."""
        return [
            (ev["type"], ev.get("isDone"))
            for ev in self.events[since:]
            if ev.get("tab_id") == _TAB and ev["type"] in ("openSubagentTab", "subagentDone")
        ]

    def _live_child(self, side_channel: bool = False) -> str:
        """Persist a running child of chat A's task; return its task id."""
        child = _persist_child(self.task_a, self.chat_a, side_channel)
        _register_running(self, child, self.chat_a, self.task_a)
        return child

    def test_named_subagent_replay_held_before_its_announcement_yields_to_a_chat_takeover(
        self,
    ) -> None:
        """Audit V4 F1: the announcement's commit checks that the tab still shows the child.

        Held between its attach and ``_reopen_subagent_tab``, a child's
        replay let a chat takeover of the same tab publish chat B, then
        removed B's chat view, subscribed the tab to the child alongside
        B and announced the child — a tab the registry binds to B.
        """
        child = self._live_child()
        self._live(self.task_b, self.chat_b)
        # The replay is a reopening (any other stands down on a closed
        # tab at once): superseded, it leaves the user's close as it is.
        self.server._closed_subagent_tabs.add(_TAB)
        self._race(
            (self.chat_a, child, "conn-1"), (self.chat_b, self.task_b, ""),
            held_at=AgentServer._reopen_subagent_tab.__code__,
            first_call=lambda *replay: self._replay(*replay, reopen=True),
        )
        self.assertEqual(self.server.tab_registry.bindings()[_TAB], self.chat_b)
        self.assertEqual(self.server._tab_chat_views.get(_TAB), self.chat_b)
        self.assertEqual(self.server._tab_opened_task_ids.get(_TAB), self.task_b)
        self.assertEqual(
            self.server.printer.tasks_for_tab(_TAB), [self.task_b],
            "the superseded child's subscription was added alongside the chat takeover's",
        )
        self.assertNotIn(_TAB, self.server.printer.viewer_tab_ids(child))
        self.assertEqual(self._subagent_frames(), [], "the superseded child was announced")
        self.assertEqual([ev["task_id"] for ev in self._frames("task_events")], [self.task_b])
        self.assertIn(_TAB, self.server._closed_subagent_tabs)

    def test_stale_finished_side_channel_replay_does_not_close_a_tab_taken_over(self) -> None:
        """Audit V4 F2: a finished side channel's close is fenced like its transcript.

        A side channel's replay finds it finished and tells the client
        to close the tab instead of announcing it; sent outside the
        lock, that ``subagentDone`` closed a tab a chat takeover had
        published and delivered meanwhile (the page closes the tab it
        names whatever it shows).  Both samples are covered: the one
        taken before the announcement and the re-sample under the
        announcement's lock.
        """
        import kiss.server.server as server_module

        child, _ = th._add_task(
            "side channel",
            self.chat_a,
            extra={"subagent": {"parent_task_id": self.task_a, "side_channel": True}},
        )
        self._race(
            (self.chat_a, child, "conn-1"), (self.chat_b, self.task_b, ""),
            held_at=server_module._subagent_is_done.__code__,
        )
        self.assertEqual(self.server.tab_registry.bindings()[_TAB], self.chat_b)
        self.assertEqual(self.server._tab_chat_views.get(_TAB), self.chat_b)
        self.assertEqual(self._subagent_frames(), [], "a stale side-channel close was sent")
        self.assertEqual([ev["task_id"] for ev in self._frames("task_events")], [self.task_b])

        # Sampled running, finished while held before the announcement,
        # whose locked re-sample finds a finished side channel.
        running = self._live_child(side_channel=True)
        self.server._replay_session(self.chat_a, _TAB, task_id=self.task_a)
        since = len(self.events)
        failures: list[BaseException] = []
        with Pause(
            AgentServer._reopen_subagent_tab.__code__,
            lambda _locals: threading.current_thread().name == "held",
        ) as pause:
            held = threading.Thread(
                target=self._replay, args=(self.chat_a, running, "conn-1", failures), name="held",
            )
            held.start()
            self.assertTrue(pause.paused.wait(10), "the replay never reached the announcement")
            agent_state.unregister(running)
            self.server._replay_session(self.chat_b, _TAB, task_id=self.task_b)
            pause.release()
            held.join(10)
        self.assertFalse(held.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(self.server._tab_chat_views.get(_TAB), self.chat_b)
        self.assertEqual(self._subagent_frames(since), [], "a stale side-channel close was sent")

    def test_subagent_reopen_wins_over_a_close_landing_before_its_announcement(self) -> None:
        """conc1018 H1-F1 ordering kept: a close is no takeover, the reopen that follows it wins.

        The close dropped the tab's pin with the rest of its state; the
        reopen pins the child again, so the status and transcript that
        the pin fences follow the announcement.
        """
        child = self._live_child()
        sub_tab = f"parent-tab__sub_{child}"
        failures: list[BaseException] = []

        def reopen() -> None:
            try:
                self.server._replay_session(self.chat_a, sub_tab, task_id=child, reopen=True)
            except BaseException as exc:  # noqa: BLE001 - reported by the test thread
                failures.append(exc)
            finally:
                th._close_thread_db()

        with Pause(
            AgentServer._reopen_subagent_tab.__code__,
            lambda _locals: threading.current_thread().name == "held",
        ) as pause:
            held = threading.Thread(target=reopen, name="held")
            held.start()
            self.assertTrue(pause.paused.wait(10), "the replay never reached the announcement")
            self.server._close_tab(sub_tab)
            pause.release()
            held.join(10)
        self.assertFalse(held.is_alive())
        self.assertEqual(failures, [])
        kinds = [
            ev["type"]
            for ev in self.events
            if sub_tab in (ev.get("tab_id"), ev.get("tabId"))
            and ev["type"] in ("closeSubagentTab", "openSubagentTab", "status", "task_events")
        ]
        self.assertEqual(kinds, ["closeSubagentTab", "openSubagentTab", "status", "task_events"])
        self.assertEqual(self.server._tab_opened_task_ids.get(sub_tab), child)
        self.assertEqual(self.server.printer.tasks_for_tab(sub_tab), [child])
        self.assertNotIn(sub_tab, self.server._closed_subagent_tabs)

    def test_deferred_child_offer_is_not_sent_after_a_chat_takeover_of_its_tab(self) -> None:
        """A18-4 (astra): the child offer's view check and its send are one locked step.

        A parent's ready replay checked that the child's tab was not
        chat-bound, released the lock, and then sent ``replay_deferred``;
        a chat takeover of that tab landing in between delivered chat
        B's transcript first, and the page, installing the late offer,
        fetched the old child back into B's tab.
        """
        child = self._child_of(self.task_a, self.chat_a)
        child_tab = f"parent-tab__sub_{child}"
        paused, go = threading.Event(), threading.Event()
        self.addCleanup(go.set)

        def sink(event: dict) -> None:
            if (
                event.get("type") == "replay_deferred"
                and threading.current_thread().name == "held"
            ):
                paused.set()
                go.wait(10)
            self.events.append(event)

        self.server.printer.broadcast = sink  # type: ignore[assignment]
        failures: list[BaseException] = []

        def children() -> None:
            try:
                self.server._open_persisted_subagent_tabs(
                    parent_task_id=self.task_a, parent_tab_id="parent-tab",
                    conn_id="reconnecting",
                )
            except BaseException as exc:  # noqa: BLE001 - reported by the test thread
                failures.append(exc)
            finally:
                th._close_thread_db()

        def take_over() -> None:
            try:
                self.server._replay_session(self.chat_b, child_tab, task_id=self.task_b)
            except BaseException as exc:  # noqa: BLE001 - reported by the test thread
                failures.append(exc)
            finally:
                th._close_thread_db()

        held = threading.Thread(target=children, name="held")
        held.start()
        self.assertTrue(paused.wait(10), "the parent replay never reached the child's offer")
        takeover = threading.Thread(target=take_over, name="takeover")
        takeover.start()
        # Lands inside the window when there is one; blocks on the lock
        # the offer is now sent under otherwise.
        takeover.join(1.0)
        go.set()
        held.join(10)
        takeover.join(10)
        self.assertFalse(held.is_alive() or takeover.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(self.server._tab_chat_views[child_tab], self.chat_b)
        kinds = [
            (ev["type"], ev.get("chat_id") or ev.get("chatId"))
            for ev in self.events
            if ev.get("tabId") == child_tab and ev["type"] in ("replay_deferred", "task_events")
        ]
        self.assertIn(("task_events", self.chat_b), kinds)
        self.assertNotIn(
            ("replay_deferred", self.chat_a), kinds[kinds.index(("task_events", self.chat_b)):],
            "the child's offer was sent after the chat takeover's transcript",
        )

    def test_superseded_replay_opens_no_subagent_tab_below_the_tab_it_lost(self) -> None:
        """Audit V5 F1: the descendant walk is fenced by the view its replay gave the tab.

        Held at the read of its sub-agent rows, after its own transcript,
        a replay (or a ready's deferral) let another chat, or another
        task of its chat, take the tab, then announced and filled its
        child's tab below it.
        """
        child_tab = f"{_TAB}__sub_{self._child_of(self.task_a, self.chat_a)}"
        other_task, _ = th._add_task("chat A, another task", self.chat_a)
        rows = th._load_subagent_rows_by_parent_task_id.__code__
        for takeover in ((self.chat_b, self.task_b, ""), (self.chat_a, other_task, "")):
            self._race((self.chat_a, self.task_a, ""), takeover, held_at=rows)
            self.assertEqual(self._about(child_tab), [], f"announced below {takeover}")
        self.server._replay_session(self.chat_a, _TAB, task_id=self.task_a)
        self.events.clear()
        self._race(
            (self.chat_a, self.task_a, "reconnecting"), (self.chat_b, self.task_b, ""),
            held_at=rows, first_call=self._defer,
        )
        self.assertEqual(self._about(child_tab), [], "offered below the rebound tab")

    def _walk_losing_its_child_tab(self, held_at: Any, finish: bool = False) -> list[str]:
        """What a walk held at *held_at* sends about a child's tab another sub-agent took.

        The child also finishes inside the window when *finish*; a
        later, scoped walk of the same parent is included.
        """
        child = self._live_child()
        child_tab = f"parent-tab__sub_{child}"
        other = self._child_of(self.task_b, self.chat_b)
        taken: list[int] = []

        def take_over() -> None:
            if finish:
                agent_state.unregister(child)
            self.server._replay_session(self.chat_b, child_tab, task_id=other)
            taken.append(len(self.events))

        self._race_action((self.task_a, "parent-tab", ""), take_over, held_at, self._walk)
        failures: list[BaseException] = []
        self._walk(self.task_a, "parent-tab", "reconnecting", failures)
        self.assertEqual(failures, [])
        self.assertEqual(self.server._tab_view_locked(child_tab), ("", other))
        return self._about(child_tab, taken[0])

    def test_walk_yields_a_child_tab_another_subagent_replay_pinned(self) -> None:
        """Audit V5 F2: the walk's child guards read the tab's pin, not only its chat view.

        A sub-agent replay installs no chat view, only its task pin: a
        walk held before the child's completion signal sent the old
        child's frames into the tab another sub-agent's replay had
        taken; so did every later walk.  (The child's offer is checked
        and sent under the lock a takeover commits under, see
        ``test_deferred_child_offer_is_not_sent_after_a_chat_takeover_of_its_tab``.)
        """
        done = AgentServer._emit_pending_ask.__code__
        self.assertEqual(self._walk_losing_its_child_tab(done, finish=True), [], "completion")

    def test_scoped_replay_of_a_tab_closed_meanwhile_does_not_reopen_it(self) -> None:
        """A18-3 (astra) / A18-1 (fable): a ready's refresh of a closed tab stands down."""
        self._live(self.task_a, self.chat_a)
        self.server._replay_session(self.chat_a, _TAB, task_id=self.task_a)
        self.events.clear()

        def close() -> None:
            self.server._close_tab(_TAB)

        self._race_action((self.chat_a, self.task_a, "reconnecting"), close)
        self.assertFalse(
            self.server.tab_registry.has_tab(_TAB),
            "the stale scoped replay reopened the tab the user closed",
        )
        self.assertNotIn(_TAB, self.server._tab_chat_views)
        self._assert_closed_tab_left_alone(_TAB)

    def _assert_closed_tab_left_alone(self, tab_id: str) -> None:
        """A stale refresh gave the closed *tab_id* no subscription, no pin and no frame."""
        self.assertNotIn(
            tab_id, self.server.printer._fanout_plan(self.task_a)[0],
            "the closed tab was subscribed to the running task again",
        )
        self.assertNotIn(tab_id, self.server._tab_opened_task_ids)
        self.assertEqual(self._about(tab_id), [], "frames for a tab no surface shows")

    def test_ready_fan_out_leaves_a_tab_closed_since_its_snapshot_alone(self) -> None:
        """Audit 2026-11-11 (V2): neither resume of a ``ready`` revives a closed tab's subscription.

        The fan-out resumes the tabs of a registry snapshot.  A viewer
        of a running chat, or the run's own tab, that another surface
        closed before its resume ran was pinned and subscribed again:
        a viewer no surface shows (so a parent the run's next sub-agent
        tabs could be opened under), told ``running:true`` on every
        connection.
        """
        self._live(self.task_a, self.chat_a)
        own = "owner-" + self.task_a
        for tab, deferred in ((_TAB, True), (own, True), (own, False)):
            with self.subTest(tab=tab, deferred=deferred):
                resume = {
                    "type": "resumeSession", "chatId": self.chat_a, "tabId": tab,
                    "replayConnId": "reconnecting", "deferred": deferred,
                }
                if tab == own:
                    # As ``_cmd_run`` publishes and subscribes the launching tab.
                    self.server._registry_update_tab(tab, chat_id=self.chat_a, create=True)
                    self.server.printer.subscribe_tab(self.task_a, tab)
                else:
                    self.server._replay_session(self.chat_a, tab, task_id=self.task_a)
                    resume["taskId"] = self.task_a
                self.assertIn(tab, self.server.printer.viewer_tab_ids(self.task_a))
                self.server._close_tab(tab)
                self.events.clear()
                self.server._handle_command(resume)
                self._assert_closed_tab_left_alone(tab)

    def test_scoped_replay_of_a_rebound_tab_refreshes_to_the_current_chat(self) -> None:
        """A18-3 (astra) / A18-1 (fable): a ready's refresh of a rebound tab shows the new chat."""
        self.server._replay_session(self.chat_a, _TAB, task_id=self.task_a)
        self.events.clear()

        def rebind() -> None:
            self.server._replay_session(self.chat_b, _TAB, task_id=self.task_b)

        self._race_action((self.chat_a, self.task_a, "reconnecting"), rebind)
        self.assertEqual(self.server.tab_registry.bindings()[_TAB], self.chat_b)
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_b)
        self.assertEqual(self.server._tab_opened_task_ids.get(_TAB), self.task_b)
        self.assertEqual(
            [(ev["chat_id"], ev.get("connId")) for ev in self._frames("task_events")],
            [(self.chat_b, None), (self.chat_b, "reconnecting")],
            "the stale scoped replay rebound the tab to the chat it no longer shows",
        )

    def _race_action(
        self,
        first: tuple[str, str, str],
        action: Callable[[], None],
        held_at: Any = th._load_chat_events_by_task_id.__code__,
        first_call: Callable[..., None] | None = None,
    ) -> None:
        """Run *first* on a thread held at *held_at* (its history read) while *action* runs."""
        failures: list[BaseException] = []
        with Pause(held_at, lambda _locals: threading.current_thread().name == "held") as pause:
            held = threading.Thread(
                target=first_call or self._replay, args=(*first, failures), name="held",
            )
            held.start()
            self.assertTrue(pause.paused.wait(10), "the held thread never reached its hold")
            action()
            pause.release()
            held.join(10)
        self.assertFalse(held.is_alive())
        self.assertEqual(failures, [])

    def test_publishing_replay_subscribes_the_tab_as_a_viewer_only_at_its_commit(self) -> None:
        """A18-2 (astra): no live frame reaches a surface before the tab's new binding.

        The attach subscribed the tab to chat A's live task before the
        binding was published and committed, so a token A streamed in
        that window was fanned out to every surface stamped with a tab
        still showing chat B.  The attach now subscribes the tab
        headless (its copies reach local peers only, see
        ``WebPrinter._fanout``) and the commit makes it a viewer under
        the lock that binds the tab.
        """
        self._live(self.task_a, self.chat_a)
        self.server._replay_session(self.chat_b, _TAB, task_id=self.task_b)
        self.events.clear()
        failures: list[BaseException] = []
        with Pause(
            AgentServer._publish_replay_reopen.__code__,
            lambda _locals: threading.current_thread().name == "held",
        ) as pause:
            held = threading.Thread(
                target=self._replay, args=(self.chat_a, self.task_a, "", failures), name="held",
            )
            held.start()
            self.assertTrue(pause.paused.wait(10), "replay A never reached its publication")
            targets, headless = self.server.printer._fanout_plan(self.task_a)
            self.assertIn(_TAB, targets)
            self.assertIn(
                _TAB, headless,
                "the tab was a viewer of chat A's task while it still showed chat B",
            )
            self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_b)
            pause.release()
            held.join(10)
        self.assertFalse(held.is_alive())
        self.assertEqual(failures, [])
        targets, headless = self.server.printer._fanout_plan(self.task_a)
        self.assertEqual((_TAB in targets, _TAB in headless), (True, False))
        self.assertEqual(self.server._tab_chat_views[_TAB], self.chat_a)
        self.assertEqual([ev["running"] for ev in self._frames("status")], [True])
        self.assertEqual([ev["chat_id"] for ev in self._frames("task_events")], [self.chat_a])


class TestSupersededReplayLive(IsolatedAsyncioTestCase):
    """The V4 and V5 interleavings over a real daemon and live connections.

    The races above run against a bare server whose broadcast is
    captured.  Here the same ``Pause`` seams hold the daemon's own
    worker thread while the real ``WebPrinter`` delivers to two
    connections of a real ``RemoteAccessServer``: the one resuming the
    child into the shared tab and the one taking the tab over with
    chat B.  Each ``resumeSession`` carries a ``replayId``: its
    ``replayDone`` reaches the sending connection after everything its
    replay sent there, so what a connection received for the tab is
    read up to it.
    """

    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="kiss-superseded-replay-live-")
        self.saved = _redirect(self.tmp)
        certfile = Path(self.tmp) / "cert.pem"
        keyfile = Path(self.tmp) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.remote = RemoteAccessServer(
            host="127.0.0.1", port=0, certfile=str(certfile), keyfile=str(keyfile),
            url_file=Path(self.tmp) / "remote-url.json",
            local_endpoint_file=Path(self.tmp) / "sorcar-local.json",
        )
        await self.remote.start_async()
        self.server = self.remote._agent_server
        self.task_a, self.chat_a = th._add_task("chat A task")
        self.task_b, self.chat_b = th._add_task("chat B task")
        th._append_chat_event({"type": "text_delta", "text": "A transcript"}, task_id=self.task_a)
        th._append_chat_event({"type": "text_delta", "text": "B transcript"}, task_id=self.task_b)
        self.resumer = await open_local_connection(self.remote)
        self.taker = await open_local_connection(self.remote)

    async def asyncTearDown(self) -> None:
        for _, writer in (self.resumer, self.taker):
            writer.close()
        await self.remote.stop_async()
        close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def _resume(
        self, conn: tuple[LocalReader, LocalWriter], chat_id: str, task_id: str,
        replay_id: str, scoped: bool, reopen: bool = False, tab_id: str = _TAB,
    ) -> None:
        """Send *conn*'s ``resumeSession`` of *task_id* into *tab_id* (default: the shared tab)."""
        cmd: dict[str, Any] = {
            "type": "resumeSession", "chatId": chat_id, "taskId": task_id, "tabId": tab_id,
            "replayId": replay_id,
        }
        if scoped:
            cmd["scoped"] = True
        if reopen:
            cmd["reopen"] = True
        conn[1].write(json.dumps(cmd).encode("utf-8") + b"\n")
        await conn[1].drain()

    async def _close(self, tab_id: str) -> None:
        """The taker closes *tab_id*; returns once the daemon is through with the close."""
        cmd = {"type": "closeTab", "tabId": tab_id}
        self.taker[1].write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.taker[1].drain()
        await self._received(self.taker)

    async def _received(
        self, conn: tuple[LocalReader, LocalWriter], replay_id: str = "",
    ) -> list[dict[str, Any]]:
        """Every frame *conn* received up to the ``replayDone`` of *replay_id*.

        Without a *replay_id* the connection is drained up to the reply
        of a ``getInputHistory`` sent now: what a replay of another
        connection broadcast to this one was queued before it.
        """
        if not replay_id:
            conn[1].write(json.dumps({"type": "getInputHistory"}).encode("utf-8") + b"\n")
            await conn[1].drain()
        frames: list[dict[str, Any]] = []
        for _ in range(500):
            line = await asyncio.wait_for(conn[0].readline(), timeout=30)
            assert line, "the connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == "replayDone" and msg.get("replayId") == replay_id:
                return frames
            if not replay_id and msg.get("type") == "inputHistory":
                return frames
            frames.append(msg)
        raise AssertionError(f"no end of the frames of {replay_id or 'the drain'}")

    @staticmethod
    def _child_frames(frames: list[dict[str, Any]]) -> list[str]:
        """The sub-agent tab frames about the shared tab among *frames*."""
        return [
            ev["type"] for ev in frames
            if ev.get("tab_id") == _TAB and ev["type"] in ("openSubagentTab", "subagentDone")
        ]

    @staticmethod
    def _transcripts(frames: list[dict[str, Any]]) -> list[str]:
        """The task ids of the transcripts delivered to the shared tab among *frames*."""
        return [
            ev["task_id"] for ev in frames
            if ev.get("type") == "task_events" and ev.get("tabId") == _TAB
        ]

    async def _race(self, child: str, held_at: Any, matching: Callable[[dict[str, Any]], bool],
                    before_release: Callable[[], None] | None = None,
                    reopen: bool = False) -> None:
        """Hold the resumer's replay of *child* at *held_at* while the taker resumes chat B.

        *before_release* runs while the replay is held, after the
        takeover is through.  Both connections' frames up to their
        ``replayDone`` are left in ``self.resumer_frames`` and
        ``self.taker_frames``.
        """
        with Pause(held_at, matching) as pause:
            await self._resume(
                self.resumer, self.chat_a, child, "child", scoped=True, reopen=reopen,
            )
            self.assertTrue(
                await asyncio.to_thread(pause.paused.wait, 10),
                "the child's replay never reached the hold",
            )
            await self._resume(self.taker, self.chat_b, self.task_b, "takeover", scoped=False)
            self.taker_frames = await self._received(self.taker, "takeover")
            if before_release is not None:
                before_release()
            pause.release()
        self.resumer_frames = await self._received(self.resumer, "child")

    def _assert_tab_shows_chat_b(self) -> None:
        self.assertEqual(self.server.tab_registry.bindings()[_TAB], self.chat_b)
        self.assertEqual(self.server._tab_chat_views.get(_TAB), self.chat_b)
        self.assertEqual(self.server._tab_opened_task_ids.get(_TAB), self.task_b)
        for frames in (self.resumer_frames, self.taker_frames):
            self.assertEqual(self._transcripts(frames), [self.task_b], frames)
            self.assertEqual(self._child_frames(frames), [], frames)

    async def test_live_child_replay_held_before_its_announcement_yields_to_a_chat_takeover(
        self,
    ) -> None:
        """Audit V4 F1 on the wire: neither connection gets the superseded child.

        The resumer's connection, whose replay was held at the
        announcement while the taker published chat B into the tab,
        receives chat B's transcript (the takeover broadcast it) and no
        announcement of the child; the taker's receives the same.  The
        tab stays chat B's (registry, chat view, task pin, printer
        subscriptions), and the closed child stays closed.
        """
        child = _persist_child(self.task_a, self.chat_a)
        _register_running(self, child, self.chat_a, self.task_a)
        _register_running(self, self.task_b, self.chat_b)
        with self.server._state_lock:
            self.server._closed_subagent_tabs.add(_TAB)
        await self._race(
            child, AgentServer._reopen_subagent_tab.__code__,
            lambda loc: loc.get("task_id") == child, reopen=True,
        )
        self._assert_tab_shows_chat_b()
        self.assertEqual(self.server.printer.tasks_for_tab(_TAB), [self.task_b])
        self.assertNotIn(_TAB, self.server.printer.viewer_tab_ids(child))
        self.assertIn(_TAB, self.server._closed_subagent_tabs)

    async def test_live_stale_finished_side_channel_replay_does_not_close_a_tab_taken_over(
        self,
    ) -> None:
        """Audit V4 F2 on the wire: no connection is told to close the taken-over tab.

        A finished side channel's replay, held at its liveness sample
        while chat B takes the tab over, sends its ``subagentDone`` to
        nobody: both connections show chat B in the tab.  Then the
        other sample: a running side channel's replay, held at the
        announcement, whose child finishes and whose tab is taken over
        before it goes on, closes nothing either.
        """
        finished = _persist_child(self.task_a, self.chat_a, side_channel=True)
        await self._race(
            finished, server_module._subagent_is_done.__code__,
            lambda loc: loc.get("sub_task_id") == finished,
        )
        self._assert_tab_shows_chat_b()

        running = _persist_child(self.task_a, self.chat_a, side_channel=True)
        _register_running(self, running, self.chat_a, self.task_a)
        await self._resume(self.resumer, self.chat_a, self.task_a, "parent", scoped=False)
        await self._received(self.resumer, "parent")
        await self._received(self.taker)
        await self._race(
            running, AgentServer._reopen_subagent_tab.__code__,
            lambda loc: loc.get("task_id") == running,
            before_release=lambda: agent_state.unregister(running),
        )
        self._assert_tab_shows_chat_b()

    async def test_live_subagent_fetch_that_is_no_reopening_leaves_a_closed_tab_closed(
        self,
    ) -> None:
        """V5-F2 on the wire: a deferred tab's transcript fetch does not undo a user's close.

        The taker closes the running child's tab before the resumer
        fetches its transcript (``reopen`` unset), or while that fetch
        is held between its attach and its announcement.  The resumer
        gets the close and nothing of the tab after it, a connection
        opened afterwards that replays the parent is told the close
        again (never to open the tab, never that the running child is
        done), and the tab stays unsubscribed.
        """
        for close_first in (True, False):
            child = _persist_child(self.task_a, self.chat_a)
            _register_running(self, child, self.chat_a, self.task_a)
            sub_tab = f"{_TAB}__sub_{child}"
            fetch = (self.resumer, self.chat_a, child, "fetch")
            if close_first:
                await self._close(sub_tab)
                await self._resume(*fetch, scoped=True, tab_id=sub_tab)
            else:
                with Pause(
                    AgentServer._reopen_subagent_tab.__code__,
                    lambda loc: loc.get("task_id") == child,
                ) as pause:
                    await self._resume(*fetch, scoped=True, tab_id=sub_tab)
                    self.assertTrue(
                        await asyncio.to_thread(pause.paused.wait, 10),
                        "the fetch never reached the announcement",
                    )
                    await self._close(sub_tab)
            frames = await self._received(self.resumer, "fetch")
            later = await open_local_connection(self.remote)
            await self._resume(later, self.chat_a, self.task_a, "parent", scoped=True)
            replayed = await self._received(later, "parent")
            later[1].close()
            kinds = [
                [
                    ev["type"] for ev in got
                    if sub_tab in (ev.get("tab_id"), ev.get("tabId"))
                    and ev["type"] in (
                        "closeSubagentTab", "openSubagentTab", "subagentDone",
                        "status", "task_events",
                    )
                ]
                for got in (frames, replayed)
            ]
            with self.subTest(close_first=close_first):
                self.assertEqual(kinds, [["closeSubagentTab"], ["closeSubagentTab"]])
                self.assertEqual(self.server.printer.tasks_for_tab(sub_tab), [])
                self.assertIn(sub_tab, self.server._closed_subagent_tabs)


if __name__ == "__main__":
    unittest.main()
