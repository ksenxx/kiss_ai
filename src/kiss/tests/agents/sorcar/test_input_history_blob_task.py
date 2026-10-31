# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the composer history readers tolerate a BLOB ``task``.

SQLite's dynamic typing lets a hand-edited or third-party ``history.db``
hold a BLOB in the TEXT ``task`` column (the case ``_safe_str`` was
added for).  The three readers behind the composer — ``_load_input_history``
(ArrowUp), ``_prefix_match_tasks`` (ghost completion) and
``_load_slash_command_usage`` (``/`` popup order) — handed the raw
``bytes`` on, so ``json.dumps`` of the ``inputHistory`` reply raised in
the daemon's command thread and the request was never answered: the
page's history stayed empty for as long as the row existed.  The replay
session dict of such a row (a resume, the adjacent-task reply) carried
the raw value too.  Every test runs against a real sqlite file in a
temp directory.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.agents.sorcar.test_steer_inputs import (
    _redirect,
    _restore,
    _set_task_timestamp,
)


def _store_blob_task(task_id: str, text: bytes) -> None:
    """Overwrite the ``task`` column of *task_id* with the BLOB *text*."""
    with closing(sqlite3.connect(th._DB_PATH)) as conn, conn:
        conn.execute("UPDATE task_history SET task = ? WHERE id = ?", (text, task_id))


class TestBlobTaskTexts:
    """A BLOB row is read back as text, like every other row."""

    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.saved = _redirect(self.tmp)
        plain_id, _chat = th._add_task("fix the login bug in the portal")
        _set_task_timestamp(plain_id, 1000.0)
        self.blob_id, _chat = th._add_task("placeholder")
        _set_task_timestamp(self.blob_id, 1001.0)
        _store_blob_task(self.blob_id, b"/discover the parser \xff\xfe")
        th._record_steer_input("fix the typo")

    def teardown_method(self) -> None:
        th._close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_arrow_up_history_decodes_the_blob(self) -> None:
        history = th._load_input_history()
        assert all(isinstance(text, str) for text in history)
        assert history[0] == "fix the typo"
        assert history[1:] == [
            "/discover the parser \ufffd\ufffd",
            "fix the login bug in the portal",
        ]
        json.dumps({"type": "inputHistory", "tasks": history})

    def test_ghost_completion_decodes_the_blob(self) -> None:
        matches = th._prefix_match_tasks("/disc")
        assert matches == ["/discover the parser \ufffd\ufffd"]
        assert th._prefix_match_tasks("fix the l") == ["fix the login bug in the portal"]

    def test_slash_command_usage_counts_the_blob_row(self) -> None:
        usage = th._load_slash_command_usage()
        assert usage == {"discover": (1, 1001.0)}

    def test_the_replay_session_of_the_blob_row_is_json(self) -> None:
        session = th._load_chat_events_by_task_id(self.blob_id)
        assert session is not None
        assert session["task"] == "/discover the parser \ufffd\ufffd"
        json.dumps(session)


class TestInputHistoryCommandIsAnswered:
    """The daemon's ``getInputHistory`` is answered with the BLOB row in place."""

    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.saved = _redirect(self.tmp)
        task_id, _chat = th._add_task("placeholder")
        _store_blob_task(task_id, b"fix the login bug in the portal")
        self.server = AgentServer()
        self.server.work_dir = self.tmp
        self.events: list[dict[str, Any]] = []
        self.server.printer.broadcast = self._capture  # type: ignore[method-assign]

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
        th._close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _capture(self, event: dict[str, Any]) -> None:
        """Stand in for the printer: keep the JSON the clients would get."""
        self.events.append(json.loads(json.dumps(event)))

    def test_the_reply_carries_the_decoded_text(self) -> None:
        self.server._handle_command({"type": "getInputHistory", "connId": "conn-1"})
        (reply,) = [e for e in self.events if e["type"] == "inputHistory"]
        assert reply["connId"] == "conn-1"
        assert reply["tasks"] == ["fix the login bug in the portal"]
