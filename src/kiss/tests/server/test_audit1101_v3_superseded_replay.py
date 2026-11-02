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
from pathlib import Path

import kiss.agents.sorcar.persistence as th
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

    def _race(self, first: tuple[str, str, str], second: tuple[str, str, str]) -> None:
        """Run replay *first* on a thread held after its publication while *second* completes."""
        failures: list[BaseException] = []
        with Pause(
            AgentServer._broadcast_replay_status.__code__,
            lambda frame_locals: frame_locals.get("tab_id") == _TAB,
        ) as pause:
            held = threading.Thread(target=self._replay, args=(*first, failures))
            held.start()
            self.assertTrue(pause.paused.wait(10), "replay A never reached its status")
            self.server._replay_session(second[0], _TAB, task_id=second[1], conn_id=second[2])
            pause.release()
            held.join(10)
        self.assertFalse(held.is_alive())
        self.assertEqual(failures, [])

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


if __name__ == "__main__":
    unittest.main()
