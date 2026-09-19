# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the History sidebar's chat panel headers.

The History sidebar groups a chat's tasks in one collapsible panel
whose header shows the chat's FIRST task — which may be older than any
row the current page carries, so the server's ``_get_history`` stamps
every session with ``chat_first_task``, looked up by
``persistence._chat_first_tasks`` over the same row set the sidebar
lists (sub-agent rows excluded).
"""

from __future__ import annotations

import shutil
import tempfile
import threading

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import _chat_first_tasks
from kiss.server.server import VSCodeServer

from .test_history_task_meta_server import _redirect, _restore


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _history_sessions(events: list[dict]) -> list[dict]:
    hist = [e for e in events if e.get("type") == "history"]
    assert len(hist) == 1, f"expected one history event, got {len(hist)}"
    sessions = hist[0]["sessions"]
    assert isinstance(sessions, list)
    return sessions  # type: ignore[no-any-return]


class TestChatFirstTask:
    """``_get_history`` stamps every session with its chat's first task."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_every_row_of_a_chat_names_its_first_task(self) -> None:
        """Each session carries the FIRST task text of ITS chat."""
        _, chat_a = th._add_task("first task of A")
        th._add_task("second task of A", chat_id=chat_a)
        _, chat_b = th._add_task("only task of B")
        server, events = _make_server()

        server._get_history(query=None)

        sessions = _history_sessions(events)
        assert len(sessions) == 3
        by_chat = {s["id"]: s for s in sessions}
        for s in sessions:
            if s["id"] == chat_a:
                assert s["chat_first_task"] == "first task of A"
            else:
                assert s["id"] == chat_b
                assert s["chat_first_task"] == "only task of B"
        # Both rows of chat A carry the same header text.
        assert by_chat[chat_a]["chat_first_task"] == "first task of A"

    def test_search_results_carry_the_header_too(self) -> None:
        """A filtered page still names each chat's overall first task —
        even when the first task itself does not match the query."""
        _, chat_a = th._add_task("first task of A")
        th._add_task("needle in A", chat_id=chat_a)
        server, events = _make_server()

        server._get_history(query="needle")

        sessions = _history_sessions(events)
        assert [s["preview"] for s in sessions] == ["needle in A"]
        assert sessions[0]["chat_first_task"] == "first task of A"

    def test_subagent_rows_never_name_the_chat(self) -> None:
        """The header ignores sub-agent rows, exactly like the list."""
        _, chat = th._add_task(
            "subagent scaffolding",
            extra={"subagent": {"parent_task_id": "deadbeef" * 4}},
        )
        th._add_task("real first task", chat_id=chat)
        server, events = _make_server()

        server._get_history(query=None)

        sessions = _history_sessions(events)
        assert [s["preview"] for s in sessions] == ["real first task"]
        assert sessions[0]["chat_first_task"] == "real first task"

    def test_first_task_text_is_bounded(self) -> None:
        """The header text is truncated to 1000 characters: it is
        stamped on EVERY row of its chat, so an unbounded prompt would
        multiply itself across the whole page's JSON (the client clamps
        the header to 3 lines anyway)."""
        huge = "x" * 5000
        _, chat = th._add_task(huge)
        th._add_task("second task", chat_id=chat)
        server, events = _make_server()

        server._get_history(query=None)

        sessions = _history_sessions(events)
        assert len(sessions) == 2
        for s in sessions:
            assert s["chat_first_task"] == "x" * 1000
        assert _chat_first_tasks([chat]) == {chat: "x" * 1000}

    def test_tied_timestamps_pick_the_first_inserted_row(self) -> None:
        """Coarse clocks (and imported databases) produce equal
        timestamps; insertion order (rowid) breaks the tie, matching
        the project's chronological order everywhere else."""
        _, chat = th._add_task("really first")
        th._add_task("second, same clock tick", chat_id=chat)
        with th._rw_lock.write_lock():
            db = th._get_db()
            db.execute("UPDATE task_history SET timestamp = 1700000000.0")
            db.commit()
        assert _chat_first_tasks([chat]) == {chat: "really first"}

    def test_chat_first_tasks_helper_edge_cases(self) -> None:
        """Empty input and empty ids are skipped; unknown chats are
        absent from the mapping."""
        assert _chat_first_tasks([]) == {}
        assert _chat_first_tasks(["", ""]) == {}
        _, chat = th._add_task("solo")
        mapping = _chat_first_tasks([chat, "", "no-such-chat"])
        assert mapping == {chat: "solo"}
