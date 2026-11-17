# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for a chat's favourite star (``chat_summaries.is_favorite``).

The CHATS panel's ``setChatFavorite`` command stores the star on the
chat's ``chat_summaries`` row, where it stays through every recompute
of that row, and ``_get_history`` stamps it on each row of the chat as
``chat_is_favorite``.
"""

from __future__ import annotations

import shutil
import tempfile

import kiss.agents.sorcar.persistence as th
from kiss.tests.server.conftest import (
    make_capturing_server,
    redirect_persistence,
    restore_persistence,
)

from .test_history_chat_first_task import _history_sessions


class TestChatFavorite:
    """The star a user gives a chat is stored, kept and listed."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)

    def teardown_method(self) -> None:
        th._close_thread_db()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _stars(self) -> dict[str, bool]:
        """Return what a fresh history page says of each listed task's chat."""
        server, events = make_capturing_server()
        server._get_history(query=None)
        return {s["title"]: s["chat_is_favorite"] for s in _history_sessions(events)}

    def test_the_command_stars_the_chat_and_its_history_rows(self) -> None:
        first_id, chat_id = th._add_task("write the intro", extra={"startTs": 1_000})
        th._save_task_extra({"endTs": 2_000}, task_id=first_id)
        th._add_task("another chat")
        assert th._chat_summaries([chat_id])[chat_id]["is_favorite"] is False
        assert self._stars() == {"write the intro": False, "another chat": False}

        server, events = make_capturing_server()
        server._handle_command({"type": "setChatFavorite", "chatId": chat_id, "isFavorite": True})
        # Every page is told to refetch: the other surfaces show the star too.
        assert events == [{"type": "tasks_updated"}]
        # A later task of the chat recomputes its row; the star stays.
        second_id, _ = th._add_task(
            "polish the abstract", chat_id=chat_id, extra={"startTs": 3_000}
        )
        th._save_task_extra({"endTs": 4_000}, task_id=second_id)
        row = th._chat_summaries([chat_id])[chat_id]
        assert (row["is_favorite"], row["last_launched"]) == (True, 3_000)
        assert self._stars() == {
            "write the intro": True, "polish the abstract": True, "another chat": False,
        }
        # So does it when a task of the chat is deleted.
        assert th._delete_task(second_id) == 1
        assert th._chat_summaries([chat_id])[chat_id]["is_favorite"] is True

        server._handle_command({"type": "setChatFavorite", "chatId": chat_id, "isFavorite": False})
        assert self._stars() == {"write the intro": False, "another chat": False}

    def test_a_chat_with_no_finished_task_can_be_starred(self) -> None:
        _, chat_id = th._add_task("still running", extra={"startTs": 20_000})
        assert th._chat_summaries([chat_id]) == {}
        assert th._set_chat_favorite(chat_id, True) is True
        assert th._chat_summaries([chat_id])[chat_id] == {
            "summary": "", "last_launched": 0, "last_finished": 0, "is_favorite": True,
        }
        # The title typed over the chat leaves its star alone.
        assert th._set_chat_summary(chat_id, "Named") is True
        assert th._chat_summaries([chat_id])[chat_id] == {
            "summary": "Named", "last_launched": 0, "last_finished": 0, "is_favorite": True,
        }

    def test_a_command_naming_no_chat_changes_nothing(self) -> None:
        server, events = make_capturing_server()
        server._handle_command({"type": "setChatFavorite", "isFavorite": True})
        assert events == []
        assert th._set_chat_favorite("", True) is False
        assert th._get_db().execute("SELECT COUNT(*) FROM chat_summaries").fetchone()[0] == 0
