# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the `/resume` chat-list feature changes.

Two changes are exercised here against the real SQLite database and
the real `_print_recent_chats` / `_list_recent_chats` helpers:

1. `/resume` (no argument) must show the **latest 20** chat ids
   rather than the previous 10.
2. The listing must surface the **task id** and **parent task id**
   for every task it prints, so users can see the per-task identity
   and the sub-agent parent relationship when present.

No mocks, patches, fakes, or test doubles — every test redirects the
real `persistence` module to a fresh temp DB and exercises the
public helpers end-to-end via `capsys` capture of the actual printed
output.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.cli_helpers import _print_recent_chats
from kiss.agents.sorcar.persistence import (
    _add_task,
    _list_recent_chats,
    _save_task_result,
)

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after.

    Mirrors the pattern used by every other persistence-level test
    in this directory (e.g. ``test_bughunt4_recent_chats_limit.py``).
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved: tuple[Path, sqlite3.Connection | None, Path] = (
            th._DB_PATH, th._db_conn, th._KISS_DIR,
        )
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

    def _set_timestamp(self, task_id: str, ts: float) -> None:
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET timestamp = ? WHERE id = ?",
                (ts, task_id),
            )
            db.commit()


class TestListRecentChatsExposesTaskAndParentIds(_TempDbTestBase):
    """`_list_recent_chats` returns task_id and parent_task_id per task."""

    def test_top_level_task_has_task_id_and_empty_parent(self) -> None:
        task_id, chat_id = _add_task("only top-level task")
        _save_task_result("done", task_id=task_id)

        chats = _list_recent_chats(limit=20)

        assert len(chats) == 1
        assert chats[0]["chat_id"] == chat_id
        tasks = chats[0]["tasks"]
        assert isinstance(tasks, list)
        assert len(tasks) == 1
        only = tasks[0]
        # New required keys.
        assert "task_id" in only, (
            f"task entry missing 'task_id': {only!r}"
        )
        assert "parent_task_id" in only, (
            f"task entry missing 'parent_task_id': {only!r}"
        )
        assert only["task_id"] == task_id
        # Top-level tasks have no parent.
        assert only["parent_task_id"] == ""

    def test_multiple_chats_each_carry_task_ids(self) -> None:
        base = time.time()
        first_id, first_chat = _add_task("first chat task")
        self._set_timestamp(first_id, base - 30)
        second_id, second_chat = _add_task("second chat task")
        self._set_timestamp(second_id, base - 10)

        chats = _list_recent_chats(limit=20)

        # Most recent first.
        assert [c["chat_id"] for c in chats] == [second_chat, first_chat]
        for entry in chats:
            tasks = entry["tasks"]
            assert isinstance(tasks, list)
            for t in tasks:
                assert "task_id" in t
                assert "parent_task_id" in t
                assert _HEX32.fullmatch(str(t["task_id"])), (
                    f"task_id is not a 32-hex uuid: {t['task_id']!r}"
                )
                # Listed tasks are real (non-subagent), so parent
                # must be the empty string.
                assert t["parent_task_id"] == ""
        # Sanity: id wiring matches what _add_task returned.
        first_entry = next(c for c in chats if c["chat_id"] == first_chat)
        first_tasks: list[dict[str, object]] = first_entry["tasks"]  # type: ignore[assignment]
        assert first_tasks[0]["task_id"] == first_id
        second_entry = next(c for c in chats if c["chat_id"] == second_chat)
        second_tasks: list[dict[str, object]] = second_entry["tasks"]  # type: ignore[assignment]
        assert second_tasks[0]["task_id"] == second_id


class TestResumePrintsLatest20Chats(_TempDbTestBase):
    """`/resume` listing prints the latest 20 chat ids (not 10)."""

    def _create_chats(self, n: int) -> list[str]:
        """Create *n* distinct chats with strictly increasing timestamps.

        Returns the chat_ids in oldest-first order so callers can slice
        the most-recent tail directly.
        """
        base = time.time()
        chat_ids: list[str] = []
        for i in range(n):
            tid, cid = _add_task(f"task in chat {i}")
            # Space timestamps a full second apart so the recency
            # ordering is unambiguous even on coarse clocks.
            self._set_timestamp(tid, base - (n - i) * 1.0)
            chat_ids.append(cid)
        return chat_ids

    def test_print_recent_chats_caps_at_20(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        chat_ids = self._create_chats(25)

        _print_recent_chats()
        out = capsys.readouterr().out

        # Count distinct "Chat ID:" header lines actually printed.
        printed_chat_ids = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed_chat_ids) == 20, (
            f"expected 20 chats printed, got {len(printed_chat_ids)}: "
            f"{printed_chat_ids!r}"
        )
        # The 20 newest chats must be exactly the last 20 created.
        expected_newest_20 = set(chat_ids[-20:])
        assert set(printed_chat_ids) == expected_newest_20
        # The 5 oldest chats must NOT appear.
        for old_cid in chat_ids[:5]:
            assert old_cid not in printed_chat_ids

    def test_print_recent_chats_under_20_prints_all(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        chat_ids = self._create_chats(7)

        _print_recent_chats()
        out = capsys.readouterr().out

        printed_chat_ids = re.findall(r"Chat ID:\s+([0-9a-f]{32})", out)
        assert len(printed_chat_ids) == 7
        assert set(printed_chat_ids) == set(chat_ids)


class TestPrintShowsTaskIdAndParentTaskId(_TempDbTestBase):
    """Printed output must include task id and parent task id labels."""

    def test_print_includes_labels_for_top_level_task(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        task_id, chat_id = _add_task("a printable task")
        _save_task_result("a result", task_id=task_id)

        _print_recent_chats()
        out = capsys.readouterr().out

        assert f"Chat ID: {chat_id}" in out
        # Both labels must appear in the output.
        assert "Task ID:" in out
        assert "Parent Task ID:" in out
        # The actual task_id value must appear under the chat block.
        assert task_id in out

    def test_print_shows_empty_parent_for_top_level_task(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        task_id, _ = _add_task("top-level visible")
        _save_task_result("ok", task_id=task_id)

        _print_recent_chats()
        out = capsys.readouterr().out

        # The Parent Task ID line must be present even when empty.
        # We don't pin the exact placeholder string, but the label
        # must appear and the value must be recognizable as "no
        # parent" — i.e. empty / dash / "(none)".
        m = re.search(r"Parent Task ID:\s*(\S*)", out)
        assert m is not None, (
            f"no 'Parent Task ID:' line in output:\n{out}"
        )
        placeholder = m.group(1).strip()
        assert placeholder in ("", "-", "(none)", "None"), (
            f"unexpected non-empty parent value: {placeholder!r}"
        )

    def test_print_empty_no_chats(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        _print_recent_chats()
        out = capsys.readouterr().out
        assert "No chat sessions found." in out
        # And no chat / task id labels are emitted.
        assert "Chat ID:" not in out
        assert "Task ID:" not in out
        assert "Parent Task ID:" not in out


class TestPrintHandlesMultipleTasksInOneChat(_TempDbTestBase):
    """Every task in a multi-task chat must show its own task id line."""

    def test_each_task_has_its_own_task_id_line(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        first_id, chat_id = _add_task("first")
        _save_task_result("r1", task_id=first_id)
        second_id, _ = _add_task("second", chat_id=chat_id)
        _save_task_result("r2", task_id=second_id)
        third_id, _ = _add_task("third", chat_id=chat_id)
        _save_task_result("r3", task_id=third_id)

        _print_recent_chats()
        out = capsys.readouterr().out

        # All three task ids must appear in the printed block.
        assert first_id in out
        assert second_id in out
        assert third_id in out
        # And there should be three "Task ID:" lines (not counting
        # the "Parent Task ID:" lines, which share the substring).
        task_id_lines = re.findall(r"(?m)^\s+Task ID:\s+\S+", out)
        parent_lines = re.findall(r"(?m)^\s+Parent Task ID:\s+\S+", out)
        assert len(task_id_lines) == 3
        assert len(parent_lines) == 3
