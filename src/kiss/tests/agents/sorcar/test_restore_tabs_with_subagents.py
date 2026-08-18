# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: restoring a parent agent that fanned out
sub-agents (``run_parallel``) after a VS Code restart.

After a VS Code restart the webview restores its tabs from persisted
state and sends ``resumeSession {chatId, tabId}`` — chat id only, no
task id — for each restored tab (see ``init()`` in media/main.js and
the ``ready`` handler in SorcarSidebarView.ts).  For a parent agent
that spawned sub-agents the restored parent tab must:

1. load the PARENT's own chat events into the parent tab (NOT the
   events of the most recently persisted sub-agent row, which shares
   the parent's chat_id and was inserted later),
2. NOT be converted into a sub-agent tab, and
3. reopen every persisted sub-agent row in its own sub-agent tab
   (``openSubagentTab`` + ``task_events``) anchored to the parent tab
   so the restored layout mirrors the live execution layout.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    """Redirect the persistence DB to a temp dir; return saved state."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _seed_parent_with_subagents(
    chat_id: str,
) -> tuple[str, list[str]]:
    """Persist a finished parent task plus two finished sub-agent rows.

    Mirrors exactly what ``ChatSorcarAgent`` writes during a
    ``run_parallel`` fan-out: the parent ``task_history`` row is
    created first, then one row per sub-agent (sharing the parent's
    ``chat_id``) whose ``extra.subagent.parent_task_id`` points back
    at the parent row.

    Returns:
        Tuple of (parent task id, list of sub-agent task ids).
    """
    parent_id, _ = th._add_task("parent task with fanout", chat_id=chat_id)
    th._append_chat_event(
        {"type": "text_delta", "text": "parent-event"}, task_id=parent_id,
    )
    th._save_task_extra(
        {
            "model": "test-model",
            "work_dir": "/tmp",
            "version": "test",
            "tokens": 0,
            "cost": 0.0,
            "is_parallel": True,
            "is_worktree": False,
        },
        task_id=parent_id,
    )
    sub_ids: list[str] = []
    for idx in range(2):
        sub_id, _ = th._add_task(f"sub task {idx}", chat_id=chat_id)
        th._append_chat_event(
            {"type": "text_delta", "text": f"sub-event-{idx}"},
            task_id=sub_id,
        )
        th._save_task_extra(
            {
                "model": "test-model",
                "work_dir": "/tmp",
                "version": "test",
                "tokens": 0,
                "cost": 0.0,
                "is_parallel": True,
                "is_worktree": False,
                "subagent": {"parent_task_id": parent_id},
            },
            task_id=sub_id,
        )
        sub_ids.append(sub_id)
    return parent_id, sub_ids


class TestLatestChatEventsSkipSubagentRows:
    """``_load_latest_chat_events_by_chat_id`` must return the latest
    NON-sub-agent row: chat-id-only resumes always target the parent
    session, while sub-agent rows are loaded explicitly by task id."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_latest_skips_trailing_subagent_rows(self) -> None:
        chat_id = "chat-restart-2"
        parent_id, _ = _seed_parent_with_subagents(chat_id)
        result = th._load_latest_chat_events_by_chat_id(chat_id)
        assert result is not None
        assert result["task_id"] == parent_id
        assert result["task"] == "parent task with fanout"

    def test_latest_returns_newer_followup_parent_row(self) -> None:
        """A follow-up (non-sub-agent) task persisted after the fan-out
        is the new session tail and must win."""
        chat_id = "chat-restart-3"
        _seed_parent_with_subagents(chat_id)
        followup_id, _ = th._add_task("follow-up task", chat_id=chat_id)
        result = th._load_latest_chat_events_by_chat_id(chat_id)
        assert result is not None
        assert result["task_id"] == followup_id

    def test_chat_with_only_subagent_rows_returns_none(self) -> None:
        """Degenerate case: no parent row at all (e.g. parent row was
        deleted) — there is nothing chat-level to resume."""
        chat_id = "chat-restart-4"
        sub_id, _ = th._add_task("orphan sub task", chat_id=chat_id)
        th._save_task_extra(
            {"subagent": {
                "parent_task_id":
                    "ffffffffffffffffffffffffffffffff"
            }}, task_id=sub_id,
        )
        assert th._load_latest_chat_events_by_chat_id(chat_id) is None
