# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the persisted ``last_model`` always ends on the newest model pick.

``_cmd_select_model`` updates the in-memory default under
``_state_lock`` and persists the pick after releasing it.  Overlapping
picks (one per connection) wrote ``config.json`` in no particular
order, and a superseded pick re-wrote the newest one it had read
without any ordering against other picks' writes: with three picks the
stale re-write could land last, and the next ``newChat`` (or
``getModels``) adopted the older file over the newer in-memory pick.  A
pick whose usage insert failed never wrote the file at all, with the
same result.

Real daemon over its local WSS endpoint and real client connections;
:class:`Pause` holds daemon threads at the entry of the real
persistence functions to land each write in the losing order.
"""

from __future__ import annotations

import sqlite3

from kiss.agents.sorcar import persistence
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.two_client_harness import TwoClientHarness

FIRST = "gpt-5.5"
SECOND = "gpt-4o-mini"
THIRD = "claude-opus-4-1"


class ModelPickPersistTest(TwoClientHarness):
    """The file names the newest pick once every pick is done."""

    def test_a_stale_rewrite_landing_after_a_newer_pick_does_not_win(self) -> None:
        """A's re-write of B is held until C has picked and written; C stays.

        A is held inside its usage write until B has picked, so A finds
        B superseding it and re-writes B; that re-write is held while B
        and then C pick and write their own models.  Released last,
        A's write used to leave the file on B while memory said C.
        """
        agent_server = self.server._agent_server
        a = self.open_client("a")
        b = self.open_client("b")
        c = self.open_client("c")
        for client in (a, b, c):
            client.ready()
            self.wait_event(client, "models")
        a_write = Pause(
            persistence._save_last_model.__code__,
            lambda frame_locals: frame_locals["model"] == FIRST,
        )
        b_usage = Pause(
            persistence._record_model_usage.__code__,
            lambda frame_locals: frame_locals["model"] == SECOND,
        )
        a_rewrite = Pause(
            persistence._save_last_model.__code__,
            lambda frame_locals: frame_locals["model"] == SECOND,
        )
        with Pauses(a_write, b_usage, a_rewrite):
            a.send({"type": "selectModel", "tabId": a.tab_id, "model": FIRST})
            self.assertTrue(a_write.paused.wait(10.0), "A reached its own write")
            b.send({"type": "selectModel", "tabId": b.tab_id, "model": SECOND})
            self.assertTrue(b_usage.paused.wait(10.0), "B picked and reached its usage insert")
            a_write.release()
            self.assertTrue(a_rewrite.paused.wait(10.0), "A reached its re-write of B")
            b_usage.release()
            self.wait_for(
                lambda: persistence._load_last_model() == SECOND, what="B's own write",
            )
            c.send({"type": "selectModel", "tabId": c.tab_id, "model": THIRD})
            self.wait_for(
                lambda: persistence._load_last_model() == THIRD, what="C's own write",
            )
            a_rewrite.release()
            self.wait_for(
                lambda: agent_server._model_picks_pending == 0, what="every pick finished",
            )
        self.assertEqual(persistence._load_last_model(), THIRD)
        c.send({"type": "newChat", "tabId": c.tab_id})
        welcome = self.wait_event(c, "showWelcome", tabId=c.tab_id)
        self.assertEqual(welcome.get("model"), THIRD)

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
