# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt: ``_list_recent_chats`` must not surface sub-agent rows.

Every other chat/history reader in ``persistence.py`` —
``_load_history``, ``_search_history``,
``_prefix_match_task``, ``_load_chat_context``,
``_load_latest_chat_events_by_chat_id`` and
``_get_adjacent_task_by_chat_id`` — explicitly filters out sub-agent
rows (``extra.subagent``, written by
``ChatSorcarAgent._run_tasks_parallel``) because they are an internal
implementation detail of the parent's ``run_parallel`` tool call.

``_list_recent_chats`` (the backend of the CLI ``--list-chats``
resume menu, see ``cli_helpers._print_recent_chats``) lacked that
filter, so every parallel fan-out task leaked into the printed chat
sessions, and a chat whose only surviving rows were sub-agent rows
was listed as a resumable session with no visible tasks.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _list_recent_chats,
    _save_task_result,
)


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestListRecentChatsSkipsSubagents(_TempDbTestBase):
    """Sub-agent rows must be invisible in the recent-chats listing."""

    def test_subagent_rows_excluded_from_chat_tasks(self) -> None:
        parent_id, chat_id = _add_task("parent visible task")
        _save_task_result("parent result", task_id=parent_id)
        # Exact shape written by ChatSorcarAgent._run_tasks_parallel.
        _add_task(
            "subagent internal fan-out task",
            chat_id=chat_id,
            extra={
                "subagent": {
                    "parent_task_id": parent_id,
                    "parent_tab_id": "tab-1",
                },
                "model": "test-model",
            },
        )

        chats = _list_recent_chats(limit=10)
        assert len(chats) == 1
        tasks = chats[0]["tasks"]
        assert isinstance(tasks, list)
        assert [t["task"] for t in tasks] == ["parent visible task"]
        assert [t["result"] for t in tasks] == ["parent result"]

    def test_chat_with_only_subagent_rows_not_listed(self) -> None:
        visible_id, visible_chat = _add_task("normal chat task")
        # A legacy/orphaned chat whose only rows are sub-agent rows
        # (e.g. the parent row was deleted by an older build that did
        # not cascade) must not be offered as a resumable session.
        _, orphan_chat = _add_task(
            "orphaned subagent task",
            extra={
                "subagent": {
                    "parent_task_id":
                        "ffffffffffffffffffffffffffffffff",
                    "parent_tab_id": "t",
                },
            },
        )
        assert orphan_chat != visible_chat

        chats = _list_recent_chats(limit=10)
        listed_chat_ids = [c["chat_id"] for c in chats]
        assert orphan_chat not in listed_chat_ids
        assert visible_chat in listed_chat_ids
        assert visible_id is not None
