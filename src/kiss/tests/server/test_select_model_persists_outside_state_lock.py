# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a ``selectModel`` waiting for the history database does not stall the daemon.

``_cmd_select_model`` persists the pick with ``_record_model_usage``,
which takes the database's writer lock.  It used to do so while
holding ``STATE_LOCK`` (so that ``_refresh_default_model``, which
re-read the file under the same lock, could never revert the pick with
a stale file), and ``STATE_LOCK`` is on the path of every tab's
commands: while a long ``getHistory`` held the reader lock, one model
pick froze every connection's ``newChat`` / ``getModels`` (V3-3).

Now the pick bumps a counter under the lock and persists after
releasing it; ``_refresh_default_model`` reads the file outside the
lock and adopts it only when the counter did not move meanwhile.

Real daemon over its local WSS endpoint, two real client connections;
the reader holding the database lock is a thread holding the very
``_rw_lock`` the daemon's write waits on.
"""

from __future__ import annotations

import threading
import time

from kiss.agents.sorcar import persistence
from kiss.tests.server.pause import Pause
from kiss.tests.server.two_client_harness import TwoClientHarness

HOLD = 6.0
"""Seconds the reader holds the lock: well past what a free daemon needs to answer."""


class SelectModelOutsideStateLockTest(TwoClientHarness):
    """Another tab's ``newChat`` is answered while the pick waits for the writer lock."""

    def test_new_chat_is_answered_while_a_pick_waits_for_the_database(self) -> None:
        a = self.open_client("a")
        b = self.open_client("b")
        a.ready()
        b.ready()
        self.wait_event(a, "models")
        self.wait_event(b, "models")

        release = threading.Event()
        held = threading.Event()

        def hold_reader() -> None:
            with persistence._rw_lock.read_lock():
                held.set()
                release.wait(HOLD)

        reader = threading.Thread(target=hold_reader, name="history-reader", daemon=True)
        reader.start()
        self.addCleanup(reader.join, 5.0)
        self.addCleanup(release.set)
        self.assertTrue(held.wait(5.0))

        a.send({"type": "selectModel", "tabId": a.tab_id, "model": "gpt-4o-mini"})
        self.wait_for(
            lambda: persistence._rw_lock._any_alive(persistence._rw_lock._pending_refs),
            what="the pick waiting for the writer lock",
        )

        started = time.monotonic()
        b.send({"type": "newChat", "tabId": b.tab_id})
        welcome = self.wait_event(b, "showWelcome", tabId=b.tab_id)
        self.assertLess(
            time.monotonic() - started, HOLD / 2,
            "newChat waited for the pick's database write",
        )
        # The in-memory default is the pick, not a stale file: the
        # welcome of the new chat already names the picked model.
        self.assertEqual(welcome.get("model"), "gpt-4o-mini")

        release.set()
        self.wait_for(
            lambda: persistence._load_last_model() == "gpt-4o-mini",
            what="the pick persisted once the reader left",
        )

    def test_a_pick_landing_while_the_file_is_read_is_kept(self) -> None:
        """A pick between the refresh's file read and its re-check wins over the stale file.

        The refresh is held at the file read (``Pause`` on
        ``_load_last_model``); the pick lands meanwhile and its own
        write waits for the database, so the refresh wakes up with the
        OLD file in hand and a newer in-memory default.
        """
        agent_server = self.server._agent_server
        a = self.open_client("a")
        a.ready()
        self.wait_event(a, "models")
        a.send({"type": "selectModel", "tabId": a.tab_id, "model": "gpt-5.5"})
        self.wait_for(lambda: persistence._load_last_model() == "gpt-5.5", what="first pick")

        with Pause(persistence._load_last_model.__code__, lambda _locals: True) as pause:
            refresh = threading.Thread(target=agent_server._refresh_default_model, daemon=True)
            refresh.start()
            self.assertTrue(pause.paused.wait(5.0), "the refresh reached the file read")
            with persistence._rw_lock.read_lock():  # the pick's write waits behind this
                a.send({"type": "selectModel", "tabId": a.tab_id, "model": "gpt-4o-mini"})
                self.wait_for(
                    lambda: agent_server._model_picks_pending == 1,
                    what="the pick waiting for the writer lock",
                )
                pause.release()
                refresh.join(5.0)
                self.assertFalse(refresh.is_alive())
                self.assertEqual(persistence._load_last_model(), "gpt-5.5", "the file is stale")
                self.assertEqual(agent_server._default_model, "gpt-4o-mini")
        self.wait_for(
            lambda: persistence._load_last_model() == "gpt-4o-mini",
            what="the pick persisted once the reader left",
        )

    def test_an_older_pick_whose_write_finishes_last_does_not_revert_a_newer_one(self) -> None:
        """Two overlapping picks: the file ends with the newest whichever write lands last.

        Client A's pick is held just before its file write; client B's
        pick lands and is persisted meanwhile; A's write is released.
        The old write must not leave the file (and the next refresh)
        on A's model.
        """
        agent_server = self.server._agent_server
        a = self.open_client("a")
        b = self.open_client("b")
        a.ready()
        b.ready()
        self.wait_event(a, "models")
        self.wait_event(b, "models")
        with Pause(
            persistence._save_last_model.__code__,
            lambda frame_locals: frame_locals["model"] == "gpt-5.5",
        ) as pause:
            a.send({"type": "selectModel", "tabId": a.tab_id, "model": "gpt-5.5"})
            self.assertTrue(pause.paused.wait(5.0), "the first pick reached its file write")
            b.send({"type": "selectModel", "tabId": b.tab_id, "model": "gpt-4o-mini"})
            self.wait_for(
                lambda: persistence._load_last_model() == "gpt-4o-mini",
                what="the second pick persisted",
            )
            self.assertEqual(agent_server._default_model, "gpt-4o-mini")
            pause.release()
            self.wait_for(
                lambda: agent_server._model_picks_pending == 0, what="both picks finished",
            )
        self.assertEqual(persistence._load_last_model(), "gpt-4o-mini")
        b.send({"type": "newChat", "tabId": b.tab_id})
        welcome = self.wait_event(b, "showWelcome", tabId=b.tab_id)
        self.assertEqual(welcome.get("model"), "gpt-4o-mini")
