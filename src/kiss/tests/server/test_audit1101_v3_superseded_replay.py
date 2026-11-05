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

import shutil
import tempfile
import threading
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.agents.sorcar.test_steer_inputs import _redirect, _restore
from kiss.tests.server.pause import Pause

_TAB = "shared-tab"


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
        th._close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _frames(self, kind: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == kind and e.get("tabId") == _TAB]

    def _replay(self, chat_id: str, task_id: str, conn_id: str, failures: list) -> None:
        try:
            self.server._replay_session(chat_id, _TAB, task_id=task_id, conn_id=conn_id)
        except BaseException as exc:  # noqa: BLE001 - reported by the test thread
            failures.append(exc)
        finally:
            th._close_thread_db()

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
        state = agent_state.AgentState(
            task_id, chat_id=chat_id, tab_id="owner-" + task_id,
            server_owned=True, is_task_active=True,
        )
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, task_id, state)

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

        def deferred(chat_id: str, task_id: str, conn_id: str, failures: list) -> None:
            try:
                self.server._defer_replay(chat_id, _TAB, task_id, conn_id)
            except BaseException as exc:  # noqa: BLE001 - reported by the test thread
                failures.append(exc)
            finally:
                th._close_thread_db()

        # The ready snapshot named (chat A, task A); the tab is rebound to
        # chat B before the deferred replay acts on it.
        self._race(
            (self.chat_a, self.task_a, "reconnecting"), (self.chat_b, self.task_b, ""),
            held_at=AgentServer._defer_replay.__code__, first_call=deferred,
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


if __name__ == "__main__":
    unittest.main()
