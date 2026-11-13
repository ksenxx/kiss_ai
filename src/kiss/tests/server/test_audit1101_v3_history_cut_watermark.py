# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A deletion landing during a history read must not poison the delta watermark.

Audit 2026-11-01 (V3 F1): ``AgentServer._get_input_history`` read
``_input_history_cut`` twice — once to decide between a full list and
a delta, and again, after the database read, to stamp the reply's
``newest``.  A ``deleteTask`` completing between the two reads
(``_delete_history`` deletes, advances the cut, then broadcasts
``tasks_updated``) therefore let a reply that still listed the deleted
prompt advertise the deletion's own cut as its watermark.  The page's
refresh on ``tasks_updated`` passed that watermark back as ``since``,
which was not below the cut, so it got an empty delta instead of the
whole list and kept the deleted prompt until some unrelated full
refresh.

Real ``AgentServer`` over a real sqlite file in a temp directory; the
reader thread is held between its database read and its reply with a
``sys.settrace`` hook on the real code path (:class:`Pause`), the
deletion runs meanwhile.  No mocks.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest

import kiss.agents.sorcar.persistence as th
from kiss.server.server import AgentServer
from kiss.tests.agents.sorcar.test_steer_inputs import (
    _redirect,
    _restore,
    _set_task_timestamp,
)
from kiss.tests.persistence_helpers import close_db
from kiss.tests.server.pause import Pause


class TestHistoryCutWatermark(unittest.TestCase):
    """A reply that predates a deletion never carries the deletion's cut."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="kiss-hist-cut-")
        self.saved = _redirect(self.tmp)
        self.server = AgentServer()
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

    def tearDown(self) -> None:
        close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _replies(self) -> list[dict]:
        return [e for e in self.events if e.get("type") == "inputHistory"]

    def _read_history(self, failures: list[BaseException]) -> None:
        try:
            self.server._get_input_history(conn_id="client-A")
        except BaseException as exc:  # noqa: BLE001 - reported by the test thread
            failures.append(exc)
        finally:
            th._close_thread_db()

    def test_deletion_during_the_read_forces_the_next_refresh_to_be_full(self) -> None:
        task_id, _chat = th._add_task("deleted prompt")
        _set_task_timestamp(task_id, 10.0)
        failures: list[BaseException] = []
        # ``_safe_str`` is the first Python call ``_load_input_texts``
        # makes after its SELECT returned: the reader is parked with
        # the deleted prompt already loaded, its reply not built yet.
        with Pause(th._safe_str.__code__, lambda _locals: True) as pause:
            reader = threading.Thread(target=self._read_history, args=(failures,))
            reader.start()
            self.assertTrue(pause.paused.wait(10), "the history read never ran")
            self.server._delete_history(
                {}, "the task", th._task_tree, "", th._delete_task, task_id,
            )
            pause.release()
            reader.join(10)
        self.assertFalse(reader.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(th._load_input_texts(), [])

        stale = self._replies()[-1]
        self.assertEqual(stale["tasks"], ["deleted prompt"])
        self.assertLess(
            stale["newest"], self.server._input_history_cut,
            "a reply listing the deleted prompt advertised the deletion's cut",
        )
        # The page's refresh on ``tasks_updated``: the whole list again,
        # without the deleted prompt, not an empty delta.
        self.server._get_input_history(conn_id="client-A", since=stale["newest"])
        refresh = self._replies()[-1]
        self.assertNotIn("since", refresh, refresh)
        self.assertEqual(refresh["tasks"], [])


if __name__ == "__main__":
    unittest.main()
