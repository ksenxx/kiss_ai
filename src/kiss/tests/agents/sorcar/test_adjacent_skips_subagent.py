"""Adjacent-task navigation in the chat webview MUST skip sub-agent rows.

Sub-agent rows are an internal implementation detail of the parent's
``run_parallel`` tool call; they should never appear in the main chat
thread when the user "adjacent-scrolls" between tasks (Cursor-style).
``_load_chat_context`` already filters sub-agent rows out of the LLM
context, but ``_get_adjacent_task_by_chat_id`` (used by the frontend's
overscroll-driven ``getAdjacentTask`` message) did not, so scrolling
between two parent tasks A → C would surface the sub-agent row B in
between.
"""

import json
import shutil
import tempfile
import time

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str):
    """Redirect the DB to a temp dir and reset the singleton connection."""
    from pathlib import Path

    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved):
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class TestAdjacentSkipsSubagent:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mark_subagent(self, task_id: int, parent_task_id: int) -> None:
        """Persist the subagent marker on a task_history row, mirroring
        what :class:`ChatSorcarAgent._run_tasks_parallel` writes."""
        th._save_task_extra(
            {"subagent": {"parent_task_id": parent_task_id}},
            task_id=task_id,
        )

    def test_prev_skips_subagent_row_in_between(self):
        """prev from C must reach A, skipping the sub-agent row B."""
        chat_id = "scroll-chat"
        a_id, _ = th._add_task("parent A", chat_id=chat_id)
        th._append_chat_event({"ev": "a"}, task_id=a_id)
        time.sleep(0.01)
        b_id, _ = th._add_task("subagent B", chat_id=chat_id)
        th._append_chat_event({"ev": "b"}, task_id=b_id)
        self._mark_subagent(b_id, parent_task_id=a_id)
        time.sleep(0.01)
        c_id, _ = th._add_task("parent C", chat_id=chat_id)
        th._append_chat_event({"ev": "c"}, task_id=c_id)

        prev = th._get_adjacent_task_by_chat_id(chat_id, c_id, "prev")
        assert prev is not None, "expected a previous task before C"
        assert prev["task_id"] == a_id, (
            f"adjacent scroll surfaced sub-agent row "
            f"{prev['task_id']} (task={prev['task']!r}); expected parent A"
        )
        assert prev["task"] == "parent A"

    def test_next_skips_subagent_row_in_between(self):
        """next from A must reach C, skipping the sub-agent row B."""
        chat_id = "scroll-chat-2"
        a_id, _ = th._add_task("parent A", chat_id=chat_id)
        th._append_chat_event({"ev": "a"}, task_id=a_id)
        time.sleep(0.01)
        b_id, _ = th._add_task("subagent B", chat_id=chat_id)
        th._append_chat_event({"ev": "b"}, task_id=b_id)
        self._mark_subagent(b_id, parent_task_id=a_id)
        time.sleep(0.01)
        c_id, _ = th._add_task("parent C", chat_id=chat_id)
        th._append_chat_event({"ev": "c"}, task_id=c_id)

        nxt = th._get_adjacent_task_by_chat_id(chat_id, a_id, "next")
        assert nxt is not None, "expected a next task after A"
        assert nxt["task_id"] == c_id, (
            f"adjacent scroll surfaced sub-agent row "
            f"{nxt['task_id']} (task={nxt['task']!r}); expected parent C"
        )
        assert nxt["task"] == "parent C"

    def test_no_adjacent_when_only_subagent_neighbours(self):
        """If the only earlier/later rows are sub-agents, return None
        rather than leaking them into the chat thread."""
        chat_id = "sub-only"
        a_id, _ = th._add_task("parent A", chat_id=chat_id)
        th._append_chat_event({"ev": "a"}, task_id=a_id)
        time.sleep(0.01)
        sub1_id, _ = th._add_task("sub 1", chat_id=chat_id)
        self._mark_subagent(sub1_id, parent_task_id=a_id)
        time.sleep(0.01)
        sub2_id, _ = th._add_task("sub 2", chat_id=chat_id)
        self._mark_subagent(sub2_id, parent_task_id=a_id)

        # next from the parent must return None — only sub-agents exist
        # after it.
        assert th._get_adjacent_task_by_chat_id(chat_id, a_id, "next") is None
        # prev from sub2 (a sub-agent row itself) must also skip the
        # sub-agent row before it and land on parent A.  Even though
        # the reference row is a sub-agent, the candidates returned
        # must never be sub-agents.
        prev = th._get_adjacent_task_by_chat_id(chat_id, sub2_id, "prev")
        assert prev is not None
        assert prev["task_id"] == a_id

    def test_extra_is_subagent_marker(self):
        """Sanity check: the marker we write matches the production
        format that ``_is_subagent_row`` recognises."""
        chat_id = "marker-check"
        a_id, _ = th._add_task("parent", chat_id=chat_id)
        sub_id, _ = th._add_task("sub", chat_id=chat_id)
        self._mark_subagent(sub_id, parent_task_id=a_id)
        db = th._get_db()
        row = db.execute(
            "SELECT extra FROM task_history WHERE id = ?", (sub_id,)
        ).fetchone()
        parsed = json.loads(row["extra"])
        assert "subagent" in parsed
        assert th._is_subagent_row(row["extra"]) is True
