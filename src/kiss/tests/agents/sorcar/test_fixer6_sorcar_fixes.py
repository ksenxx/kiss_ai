# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E regression tests for findings-6 fixes #4 and #10.

#4  ``ChatSorcarAgent.new_chat`` leaked the one-shot
    ``resume_from_task_id`` seed: after ``resume_from_task_id(tid);
    new_chat()`` the first prompt of the "new" chat was still
    augmented with the old task's parent-chain context.

#10 Sub-agent row filtering was SQL-side (``_HISTORY_NOT_SUBAGENT``)
    in some persistence readers but Python-side in
    ``_load_latest_chat_events_by_chat_id``, ``_load_chat_context``,
    and ``_list_recent_chats``' inner tasks query.  All readers must
    agree on the exact same result sets.

All tests run against real SQLite databases in temp dirs;
no kiss code is mocked.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestNewChatClearsContextSeed(_TempDbTestBase):
    """#4: ``new_chat`` must drop the pending ``resume_from_task_id`` seed."""

    def test_seed_augments_prompt_without_new_chat(self) -> None:
        """Sanity: the seed DOES augment the next prompt when not cleared."""
        task_id, _ = th._add_task("SECRET-OLD-TASK write the report")
        th._save_task_result("SECRET-OLD-RESULT done", task_id=task_id)
        agent = ChatSorcarAgent("fixer6-seed-sanity")
        agent.resume_from_task_id(task_id)
        prompt = agent.build_chat_prompt("follow-up prompt")
        assert "SECRET-OLD-TASK" in prompt
        assert "SECRET-OLD-RESULT" in prompt

    def test_new_chat_drops_pending_seed(self) -> None:
        """After ``new_chat`` the old task chain must NOT leak into the prompt."""
        task_id, _ = th._add_task("SECRET-OLD-TASK write the report")
        th._save_task_result("SECRET-OLD-RESULT done", task_id=task_id)
        agent = ChatSorcarAgent("fixer6-seed-clear")
        agent.resume_from_task_id(task_id)
        agent.new_chat()
        prompt = agent.build_chat_prompt("fresh prompt")
        assert "SECRET-OLD-TASK" not in prompt, (
            "new_chat leaked the resume_from_task_id seed: the first "
            "prompt of a brand-new chat carries the old task's context"
        )
        assert "SECRET-OLD-RESULT" not in prompt
        assert prompt == "# Task\nfresh prompt"


class TestSubagentFilteringConsistency(_TempDbTestBase):
    """#10: all chat readers must hide sub-agent rows identically."""

    def _seed_parent_and_subagents(self) -> tuple[str, str]:
        """Insert one parent task followed by two newer sub-agent rows."""
        parent_id, chat_id = th._add_task("parent task text")
        th._save_task_result("parent result text", task_id=parent_id)
        time.sleep(0.01)
        for i in range(2):
            th._add_task(
                f"subagent internal task {i}",
                chat_id=chat_id,
                extra={"subagent": {"parent_task_id": parent_id}},
            )
        return parent_id, chat_id

    def test_latest_chat_events_skips_newer_subagent_rows(self) -> None:
        parent_id, chat_id = self._seed_parent_and_subagents()
        latest = th._load_latest_chat_events_by_chat_id(chat_id)
        assert latest is not None
        assert latest["task"] == "parent task text"
        assert latest["task_id"] == parent_id

    def test_chat_context_excludes_subagent_rows(self) -> None:
        _, chat_id = self._seed_parent_and_subagents()
        ctx = th._load_chat_context(chat_id)
        assert [e["task"] for e in ctx] == ["parent task text"]
        assert [e["result"] for e in ctx] == ["parent result text"]

    def test_list_recent_chats_excludes_subagent_rows(self) -> None:
        _, chat_id = self._seed_parent_and_subagents()
        chats = th._list_recent_chats()
        assert [c["chat_id"] for c in chats] == [chat_id]
        tasks = cast("list[dict[str, object]]", chats[0]["tasks"])
        assert [t["task"] for t in tasks] == ["parent task text"]
        assert tasks[0]["parent_task_id"] == ""

    def test_subagent_only_chat_invisible_to_all_readers(self) -> None:
        """A chat holding ONLY sub-agent rows must vanish everywhere."""
        parent_id, _ = th._add_task("parent in its own chat")
        time.sleep(0.01)
        _, sub_chat = th._add_task(
            "orphan-chat subagent task",
            chat_id="fixer6subchat0000000000000000000",
            extra={"subagent": {"parent_task_id": parent_id}},
        )
        assert th._load_latest_chat_events_by_chat_id(sub_chat) is None
        assert th._load_chat_context(sub_chat) == []
        assert sub_chat not in [c["chat_id"] for c in th._list_recent_chats()]
