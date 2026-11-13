# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the persisted ``last_model`` and the daemon's default end on the same model pick.

``_cmd_select_model`` updates the in-memory default under
``_state_lock`` and persists the pick after releasing it.  Picks that
overlap (one per connection) save themselves one at a time, each with
one write and only while it is still the newest, so the file ends on
the newest pick.  A pick ``config.json`` cannot record puts the default
back on the last pick the file did record, which is the model a
restart would start new chats on.

Real daemon over its local WSS endpoint and real client connections;
:class:`Pause` holds daemon threads at the entry of the real
persistence functions to land each write in the losing order.
"""

from __future__ import annotations

import functools
import sqlite3
import sys
import unittest
from typing import Any

from kiss.agents.sorcar import persistence
from kiss.core import user_config
from kiss.tests.conftest import is_root
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

FIRST = "gpt-5.5"
SECOND = "gpt-4o-mini"
THIRD = "claude-opus-4-1"

_needs_unreadable_file = unittest.skipIf(
    sys.platform == "win32" or is_root(),
    "needs POSIX permission bits that deny the owner (not root, not Windows)",
)


def _note_write(writes: list[str], frame_locals: dict[str, Any]) -> bool:
    """Note the model a ``_save_last_model`` call is about to write; holds nobody."""
    writes.append(frame_locals["model"])
    return False


def _unreadable_once_saved(frame_locals: dict[str, Any]) -> bool:
    """Make ``config.json`` unreadable once it holds the model being written; holds nobody."""
    if persistence._load_last_model() == frame_locals["model"]:
        user_config.CONFIG_PATH.chmod(0)
    return False


class ModelPickPersistTest(TwoClientHarness):
    """The file names the newest pick it could record, and new chats start on that one."""

    def _pages(self) -> list[Client]:
        """The ready pages a, b and c."""
        pages = [self.open_client(name) for name in ("a", "b", "c")]
        for page in pages:
            page.ready()
            self.wait_event(page, "models")
        return pages

    def _pages_with_first_saved(self) -> list[Client]:
        """:meth:`_pages`, once a's pick of ``FIRST`` is saved and no pick is in flight."""
        pages = self._pages()
        pages[0].send({"type": "selectModel", "tabId": pages[0].tab_id, "model": FIRST})
        self._wait_picks_done(FIRST)
        self.assertEqual(persistence._load_last_model(), FIRST)
        return pages

    def _picks_done(self, newest: str) -> bool:
        """Whether *newest* is the newest pick and no pick is in flight."""
        agent_server = self.server._agent_server
        with agent_server._state_lock:
            return agent_server._model_pick[1] == newest and not agent_server._model_picks_pending

    def _wait_picks_done(self, newest: str) -> None:
        """Wait until *newest* is the newest pick and no pick is in flight."""
        self.wait_for(functools.partial(self._picks_done, newest), what="every pick finished")

    def _new_chat_model(self, page: Client) -> str:
        """The model a new chat of *page* starts on."""
        page.send({"type": "newChat", "tabId": page.tab_id})
        return str(self.wait_event(page, "showWelcome", tabId=page.tab_id).get("model"))

    def test_the_newest_of_overlapping_picks_is_the_one_saved(self) -> None:
        """A's write is held while B and then C pick: the file ends on C, and B never writes.

        A was the newest pick when its turn came, so its write lands
        first.  B is superseded by C before its turn and writes
        nothing, whichever of the two gets its turn first; a write of
        B's landing after C's would leave the file on B while memory
        said C.  B's tab still runs on B's pick, so it is counted like
        the other two.
        """
        agent_server = self.server._agent_server
        a, b, c = self._pages()
        counted = persistence._load_model_usage()
        writes: list[str] = []
        a_write = Pause(
            persistence._save_last_model.__code__,
            lambda frame_locals: frame_locals["model"] == FIRST,
        )
        every_write = Pause(
            persistence._save_last_model.__code__, functools.partial(_note_write, writes),
        )
        with Pauses(every_write, a_write):
            a.send({"type": "selectModel", "tabId": a.tab_id, "model": FIRST})
            self.assertTrue(a_write.paused.wait(10.0), "A reached its write")
            b.send({"type": "selectModel", "tabId": b.tab_id, "model": SECOND})
            self.wait_for(lambda: agent_server._model_pick[1] == SECOND, what="B picked")
            c.send({"type": "selectModel", "tabId": c.tab_id, "model": THIRD})
            self.wait_for(lambda: agent_server._model_pick[1] == THIRD, what="C picked")
            a_write.release()
            self._wait_picks_done(THIRD)
        self.assertEqual(writes, [FIRST, THIRD])
        usage = persistence._load_model_usage()
        self.assertEqual(
            [usage.get(model, 0) - counted.get(model, 0) for model in (FIRST, SECOND, THIRD)],
            [1, 1, 1],
        )
        self.assertEqual(persistence._load_last_model(), THIRD)
        self.assertEqual(self._new_chat_model(c), THIRD)

    def test_a_pick_whose_usage_insert_fails_is_still_persisted(self) -> None:
        """The usage counter cannot be written; the pick still reaches ``config.json``.

        The ``model_usage`` table is dropped behind the daemon's back,
        so the real insert raises ``sqlite3.OperationalError``.  The
        pick used to skip its file write, and the next ``newChat``
        reverted the default to the previous pick.
        """
        agent_server = self.server._agent_server
        a = self.open_client("a")
        a.ready()
        self.wait_event(a, "models")
        a.send({"type": "selectModel", "tabId": a.tab_id, "model": FIRST})
        self.wait_for(lambda: persistence._load_last_model() == FIRST, what="first pick")
        self.wait_for(lambda: agent_server._model_picks_pending == 0, what="first pick done")

        with sqlite3.connect(persistence._DB_PATH) as db:
            db.execute("DROP TABLE model_usage")
        a.send({"type": "selectModel", "tabId": a.tab_id, "model": SECOND})
        self.wait_for(
            lambda: agent_server._model_pick[1] == SECOND
            and agent_server._model_picks_pending == 0,
            what="the second pick done",
        )
        self.assertEqual(persistence._load_last_model(), SECOND)
        a.send({"type": "newChat", "tabId": a.tab_id})
        welcome = self.wait_event(a, "showWelcome", tabId=a.tab_id)
        self.assertEqual(welcome.get("model"), SECOND)

    @_needs_unreadable_file
    def test_picks_the_config_cannot_record_are_reported_and_not_the_default(self) -> None:
        """``config.json`` is unreadable: each sender is told, new chats keep the saved model.

        The write's ``PermissionError`` used to escape the handler: no
        answer, and the unsaved pick stayed the daemon's default, so
        every new chat started on a model a restart would forget.  A is
        held inside its write until B has picked, so the default B
        replaced is A's pick, which is not saved either: the default
        goes back to the one the file holds.  Each picking tab keeps
        the model its picker shows.
        """
        agent_server = self.server._agent_server
        a, b, c = self._pages_with_first_saved()
        a_write = Pause(
            persistence._save_last_model.__code__,
            lambda frame_locals: frame_locals["model"] == SECOND,
        )
        user_config.CONFIG_PATH.chmod(0)
        try:
            with a_write:
                a.send({"type": "selectModel", "tabId": a.tab_id, "model": SECOND})
                self.assertTrue(a_write.paused.wait(10.0), "A reached its own write")
                b.send({"type": "selectModel", "tabId": b.tab_id, "model": THIRD})
                self.wait_for(lambda: agent_server._model_pick[1] == THIRD, what="B picked")
                a_write.release()
                error = self.wait_event(b, "error", command="selectModel")
                self.assertIn("Could not save the model pick", error["text"])
                self.wait_event(a, "error", command="selectModel")
            self._wait_picks_done(THIRD)
            self.assertEqual(agent_server._tab_model(a.tab_id), SECOND)
            self.assertEqual(agent_server._tab_model(b.tab_id), THIRD)
            self.assertEqual(self._new_chat_model(c), FIRST)
            self.assertEqual(c.of_type("error"), [])
        finally:
            # Readable again before the daemon stops (it may read or write it).
            user_config.CONFIG_PATH.chmod(0o600)
        self.assertEqual(persistence._load_last_model(), FIRST)

    @_needs_unreadable_file
    def test_a_saved_pick_stays_when_the_config_turns_unreadable(self) -> None:
        """``config.json`` turns unreadable as soon as it holds the pick: the pick stays.

        The handler used to write the file once more after the write
        that saved the pick, and took the failure of that second write
        for an unsaved pick: the sender got an ``error`` and new chats
        went back to the previous model, which a restart would not.
        """
        a, _, c = self._pages_with_first_saved()
        try:
            with Pause(persistence._save_last_model.__code__, _unreadable_once_saved):
                a.send({"type": "selectModel", "tabId": a.tab_id, "model": SECOND})
                self._wait_picks_done(SECOND)
            user_config.CONFIG_PATH.chmod(0)
            self.assertEqual(self._new_chat_model(c), SECOND)
            self.assertEqual(a.of_type("error"), [])
        finally:
            user_config.CONFIG_PATH.chmod(0o600)
        self.assertEqual(persistence._load_last_model(), SECOND)

    @_needs_unreadable_file
    def test_a_pick_that_fails_goes_back_to_the_pick_saved_before_it(self) -> None:
        """A's pick is saved and B's newer one is not: new chats start on A's, as after a restart.

        B picks while A's write is held, so both picks are in flight
        together, and the file turns unreadable between A's write and
        B's.  The default used to go back to the one in force before
        both picks, although the file had moved on.
        """
        agent_server = self.server._agent_server
        a, b, c = self._pages_with_first_saved()
        a_write = Pause(
            persistence._save_last_model.__code__,
            lambda frame_locals: frame_locals["model"] == SECOND,
        )
        b_write = Pause(
            persistence._save_last_model.__code__,
            lambda frame_locals: frame_locals["model"] == THIRD,
        )
        try:
            with Pauses(a_write, b_write):
                a.send({"type": "selectModel", "tabId": a.tab_id, "model": SECOND})
                self.assertTrue(a_write.paused.wait(10.0), "A reached its write")
                b.send({"type": "selectModel", "tabId": b.tab_id, "model": THIRD})
                self.wait_for(lambda: agent_server._model_pick[1] == THIRD, what="B picked")
                a_write.release()
                self.assertTrue(b_write.paused.wait(10.0), "B reached its write")
                self.wait_for(
                    lambda: agent_server._model_picks_pending == 1, what="A's pick finished",
                )
                user_config.CONFIG_PATH.chmod(0)
                b_write.release()
                self.wait_event(b, "error", command="selectModel")
                self._wait_picks_done(THIRD)
            self.assertEqual(self._new_chat_model(c), SECOND)
            self.assertEqual(a.of_type("error"), [])
        finally:
            user_config.CONFIG_PATH.chmod(0o600)
        self.assertEqual(persistence._load_last_model(), SECOND)
