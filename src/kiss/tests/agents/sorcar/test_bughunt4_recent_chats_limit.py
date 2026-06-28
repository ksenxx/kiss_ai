# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: sub-agent-only chats must not consume ``_list_recent_chats`` slots.

``_list_recent_chats(limit)`` selected the most recent *limit* distinct
``chat_id`` groups FIRST and only then dropped chats whose every row is
a sub-agent row (``extra.subagent``, see ``_is_subagent_row``).  An
omitted chat therefore still consumed one of the *limit* slots: with
``limit=2`` and history [newest: sub-agent-only chat X, older: real
chat A, oldest: real chat B], the function returned only ``[A]`` even
though two real chats exist — the CLI's ``--list-chats`` (and any
resume picker built on it) silently hid chat B.

The fix filters sub-agent rows inside the chat-selection query itself
(``_HISTORY_NOT_SUBAGENT``), so only chats with at least one real task
are counted against *limit* — and chat recency is anchored to the
latest REAL task rather than a sub-agent row's timestamp.

Runs against a real SQLite database redirected to a temp dir.
No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import (
    _add_task,
    _list_recent_chats,
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

    def _set_timestamp(self, task_id: int, ts: float) -> None:
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET timestamp = ? WHERE id = ?",
                (ts, task_id),
            )
            db.commit()


class TestSubagentOnlyChatsDoNotConsumeLimit(_TempDbTestBase):
    """Omitted sub-agent-only chats must not eat ``limit`` slots."""

    def test_real_chat_beyond_raw_limit_still_returned(self) -> None:
        base = time.time()
        # Oldest: real chat B.
        b_id, chat_b = _add_task("real task B")
        self._set_timestamp(b_id, base - 30)
        # Middle: real chat A.
        a_id, chat_a = _add_task("real task A")
        self._set_timestamp(a_id, base - 20)
        # Newest: an orphaned sub-agent-only chat (its parent lives in
        # another chat / was created by a legacy run) — omitted from
        # the listing but previously still consuming a limit slot.
        x_id, _chat_x = _add_task(
            "orphaned subagent task",
            extra={"subagent": {
                "parent_task_id": "ffffffffffffffffffffffffffffffff"
            }},
        )
        self._set_timestamp(x_id, base - 10)

        chats = _list_recent_chats(limit=2)

        chat_ids = [c["chat_id"] for c in chats]
        assert chat_ids == [chat_a, chat_b]

    def test_chat_recency_anchored_to_real_task(self) -> None:
        base = time.time()
        # Chat A: real task at base-30, plus a sub-agent row at base-5
        # (sub-agents share the parent's chat_id).
        a_id, chat_a = _add_task("chat A real task")
        self._set_timestamp(a_id, base - 30)
        sub_id, _ = _add_task(
            "chat A subagent",
            chat_id=chat_a,
            extra={"subagent": {"parent_task_id": a_id}},
        )
        self._set_timestamp(sub_id, base - 5)
        # Chat B: real task at base-20 — more recent REAL activity
        # than chat A's real task.
        b_id, chat_b = _add_task("chat B real task")
        self._set_timestamp(b_id, base - 20)

        chats = _list_recent_chats(limit=10)

        chat_ids = [c["chat_id"] for c in chats]
        assert chat_ids == [chat_b, chat_a]
        # Sub-agent rows stay hidden from the per-chat task list.
        a_entry = next(c for c in chats if c["chat_id"] == chat_a)
        tasks = a_entry["tasks"]
        assert isinstance(tasks, list)
        assert [t["task"] for t in tasks] == ["chat A real task"]
