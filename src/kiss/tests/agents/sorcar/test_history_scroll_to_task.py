# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test that resumeSession with a taskId loads the specific task, not the latest.

When a user clicks a specific task in the history panel, the backend should
load that task's events (not the most recent task in the chat session) so
the frontend can scroll to the clicked task.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar import persistence as th


class TestResumeSessionWithTaskId:
    """resumeSession with taskId loads the specific task, not the latest."""

    def test_load_chat_events_by_task_id(self, tmp_path: Path) -> None:
        """_load_chat_events_by_task_id returns the correct task."""
        orig_dir = th._KISS_DIR
        orig_db = th._DB_PATH
        orig_conn = th._db_conn
        try:
            th._db_conn = None
            th._KISS_DIR = tmp_path
            th._DB_PATH = tmp_path / "sorcar.db"

            task_id, chat_id = th._add_task("specific task", chat_id="0")
            th._append_chat_event(
                {"type": "text_delta", "text": "hi"}, task_id=task_id,
            )

            result = th._load_chat_events_by_task_id(task_id)
            assert result is not None
            assert result["task"] == "specific task"
            assert result["task_id"] == task_id
            assert result["chat_id"] == chat_id
            evts = result["events"]
            assert isinstance(evts, list)
            assert len(evts) == 1

            assert th._load_chat_events_by_task_id("999999") is None
        finally:
            th._close_db()
            th._db_conn = orig_conn
            th._KISS_DIR = orig_dir
            th._DB_PATH = orig_db
