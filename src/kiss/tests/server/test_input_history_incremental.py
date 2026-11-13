# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The composer's ArrowUp history is refreshed incrementally.

Every ``tasks_updated`` (a task or sub-agent starting or finishing)
and every ``ready`` used to make the page fetch the whole composer
history again: the 500 newest distinct texts, hundreds of kilobytes
on a slow network, down the same socket as the CHATS page and the
transcript the user is waiting for.  Now a reply carries ``newest``
(the time of the most recent use among its texts) and a request that
passes it back as ``since`` gets only the texts used after it, with
``since`` echoed (``AgentServer._get_input_history``,
``persistence._load_input_texts``).  A delta cannot name a prompt a
deleted task took with it, so a page whose list predates the last
deletion (``_input_history_cut``) gets the whole list again.

Real ``AgentServer`` over a real sqlite file in a temp directory; the
replies are captured from the printer's broadcast.
"""

from __future__ import annotations

import shutil
import tempfile
import time

import kiss.agents.sorcar.persistence as th
from kiss.server.server import AgentServer
from kiss.tests.agents.sorcar.test_steer_inputs import (
    _redirect,
    _restore,
    _set_steer_timestamp,
    _set_task_timestamp,
)
from kiss.tests.persistence_helpers import close_db


class TestIncrementalInputHistory:
    """A refresh with ``since`` carries only what was typed since."""

    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.saved = _redirect(self.tmp)
        self.server = AgentServer()
        self.events: list[dict] = []
        self.server.printer.broadcast = self.events.append  # type: ignore[assignment]

    def teardown_method(self) -> None:
        close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _replies(self) -> list[dict]:
        return [e for e in self.events if e.get("type") == "inputHistory"]

    def _seed(self) -> None:
        t1, _chat = th._add_task("first task")
        _set_task_timestamp(t1, 10.0)
        th._record_steer_input("steer one")
        _set_steer_timestamp("steer one", 20.0)
        t2, _chat = th._add_task("second task")
        _set_task_timestamp(t2, 30.0)

    def test_loader_filters_both_tables_by_time_of_use(self) -> None:
        self._seed()
        assert th._load_input_texts() == [
            ("second task", 30.0), ("steer one", 20.0), ("first task", 10.0),
        ]
        assert th._load_input_texts(since=15.0) == [("second task", 30.0), ("steer one", 20.0)]
        assert th._load_input_texts(since=30.0) == []
        # A text used again moves to the front with its newest use.
        th._record_steer_input("first task")
        _set_steer_timestamp("first task", 40.0)
        assert th._load_input_texts(since=30.0) == [("first task", 40.0)]
        assert th._load_input_texts(limit=2, since=15.0) == [
            ("first task", 40.0), ("second task", 30.0),
        ]

    def test_full_reply_names_its_newest_use_and_a_refresh_is_a_delta(self) -> None:
        self._seed()
        self.server._get_input_history()
        (full,) = self._replies()
        assert full["tasks"] == ["second task", "steer one", "first task"]
        assert full["newest"] == 30.0
        assert "since" not in full

        # Nothing new: an empty delta whose newest is still the client's.
        self.server._get_input_history(since=full["newest"])
        empty = self._replies()[1]
        assert empty == {"type": "inputHistory", "tasks": [], "newest": 30.0, "since": 30.0}

        # A task started and a message typed into it: the delta holds
        # those two, newest first, and the newest use moves on.
        t3, _chat = th._add_task("third task")
        _set_task_timestamp(t3, 50.0)
        th._record_steer_input("  first task  ")
        _set_steer_timestamp("  first task  ", 60.0)
        self.server._get_input_history(since=30.0)
        delta = self._replies()[2]
        assert delta["tasks"] == ["first task", "third task"]
        assert delta["newest"] == 60.0
        assert delta["since"] == 30.0

    def test_command_passes_since_and_ignores_a_malformed_one(self) -> None:
        self._seed()
        self.server._handle_command({"type": "getInputHistory", "since": 20.0})
        self.server._handle_command({"type": "getInputHistory", "since": "20"})
        # ``10**400`` is a valid JSON integer a client may send; it
        # overflows ``float`` and must count as malformed, not raise.
        for bad in ("abc", None, True, float("nan"), -5, 0, 10**400):
            self.server._handle_command({"type": "getInputHistory", "since": bad})
        replies = self._replies()
        assert [r["tasks"] for r in replies[:2]] == [["second task"], ["second task"]]
        assert all(r["since"] == 20.0 for r in replies[:2])
        assert len(replies) == 9
        for whole in replies[2:]:
            assert whole["tasks"] == ["second task", "steer one", "first task"]
            assert "since" not in whole

    def test_empty_database(self) -> None:
        self.server._get_input_history()
        self.server._get_input_history(since=5.0)
        assert self._replies() == [
            {"type": "inputHistory", "tasks": [], "newest": 0.0},
            {"type": "inputHistory", "tasks": [], "newest": 5.0, "since": 5.0},
        ]

    def test_a_deletion_makes_the_next_refresh_the_whole_list(self) -> None:
        """A list fetched before a task was deleted is replaced, not merged into."""
        self._seed()
        t2 = th._load_history(limit=1, offset=0, tag="", query="", work_dir="",
                              running_task_ids=set())[0]["id"]
        self.server._get_input_history()
        (full,) = self._replies()
        assert full["tasks"] == ["second task", "steer one", "first task"]

        before = time.time()
        self.server._handle_command({"type": "deleteTask", "taskId": t2})
        assert self.events[-1] == {"type": "tasks_updated"}
        assert before <= self.server._input_history_cut <= time.time()

        # The page's refresh after the nudge: the whole list, without
        # the deleted prompt, replacing the one on hand (no ``since``);
        # its cursor moves past the deletion.
        self.server._get_input_history(since=full["newest"])
        whole = self._replies()[1]
        assert whole["tasks"] == ["steer one", "first task"]
        assert "since" not in whole
        assert whole["newest"] == self.server._input_history_cut

        # From then on a refresh is a delta again.
        self.server._get_input_history(since=whole["newest"])
        delta = self._replies()[2]
        assert delta == {
            "type": "inputHistory", "tasks": [], "newest": whole["newest"],
            "since": whole["newest"],
        }

        # A page that reconnects with a cursor from before the deletion
        # (``ready`` with ``inputHistorySince``) gets the whole list too.
        self.server._handle_command(
            {"type": "getInputHistory", "since": full["newest"]},
        )
        again = self._replies()[3]
        assert again["tasks"] == ["steer one", "first task"]
        assert "since" not in again
