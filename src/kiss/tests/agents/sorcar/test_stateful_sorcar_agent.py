"""Tests for ChatSorcarAgent: chat context, prompt augmentation, persistence."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import MAX_TASKS, ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _load_last_chat_id,
    _save_task_result,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


def _redirect(tmpdir: str) -> tuple:
    """Redirect DB to a temp dir and reset the singleton connection."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved: tuple) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _patch_super_run(agent: ChatSorcarAgent, captured: dict[str, Any]) -> Any:
    """Monkey-patch RelentlessAgent.run to capture the prompt and return YAML."""
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original_run = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        captured["prompt_template"] = kwargs.get("prompt_template", "")
        return "success: true\nsummary: test done\n"

    parent_class.run = fake_run
    return original_run


class TestChatSorcarAgent:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_last_chat_id_returns_most_recent(self) -> None:
        agent = ChatSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="some task")
        finally:
            parent_class.run = original_run

        assert _load_last_chat_id() == agent.chat_id


def _seed_chat(n: int) -> str:
    """Insert *n* tasks into a single chat session and return its chat_id."""
    chat_id = ""
    for i in range(1, n + 1):
        task_id, chat_id = _add_task(f"task {i}", chat_id=chat_id)
        _save_task_result(result=f"result {i}", task_id=task_id)
    return chat_id


class TestBuildChatPromptTruncation:
    """Tests for the MAX_TASKS truncation in build_chat_prompt."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_max_tasks_constant_is_ten(self) -> None:
        """Document the truncation budget — change here if MAX_TASKS changes."""
        assert MAX_TASKS == 10

    def test_no_context_returns_prompt_unaugmented(self) -> None:
        """Empty chat history → just '# Task\\n' + prompt, no preamble."""
        agent = ChatSorcarAgent("test")
        out = agent.build_chat_prompt("hello")
        assert out == "# Task\nhello"

    def test_below_limit_renders_all_entries(self) -> None:
        """N < MAX_TASKS → all N entries rendered, no truncation."""
        chat_id = _seed_chat(5)
        agent = ChatSorcarAgent("test")
        agent.resume_chat_by_id(chat_id)
        out = agent.build_chat_prompt("now")

        for i in range(1, 6):
            assert f"### Task {i}\ntask {i}" in out
            assert f"### Result {i}\nresult {i}" in out
        assert "### Task 6" not in out
        assert out.endswith("# Task (work on it now)\n\nnow")

    def test_exactly_at_limit_renders_all_entries(self) -> None:
        """N == MAX_TASKS → all 10 entries rendered, no truncation."""
        chat_id = _seed_chat(MAX_TASKS)
        agent = ChatSorcarAgent("test")
        agent.resume_chat_by_id(chat_id)
        out = agent.build_chat_prompt("now")

        for i in range(1, MAX_TASKS + 1):
            assert f"task {i}" in out
            assert f"result {i}" in out
        assert "### Task 11" not in out

    def test_one_over_limit_drops_one_middle_entry(self) -> None:
        """N == MAX_TASKS+1 → drop exactly the third-oldest entry (index 2)."""
        chat_id = _seed_chat(MAX_TASKS + 1)
        agent = ChatSorcarAgent("test")
        agent.resume_chat_by_id(chat_id)
        out = agent.build_chat_prompt("now")

        assert "task 3" not in out
        assert "result 3" not in out
        for kept in (1, 2, 4, 5, 6, 7, 8, 9, 10, 11):
            assert f"task {kept}" in out
            assert f"result {kept}" in out
        assert "### Task 10" in out
        assert "### Task 11" not in out

    def test_far_over_limit_keeps_first_two_and_last_eight(self) -> None:
        """N >> MAX_TASKS → first 2 entries kept, last (MAX_TASKS-2) kept."""
        n = 20
        chat_id = _seed_chat(n)
        agent = ChatSorcarAgent("test")
        agent.resume_chat_by_id(chat_id)
        out = agent.build_chat_prompt("now")

        kept_inputs = {1, 2} | set(range(n - (MAX_TASKS - 2) + 1, n + 1))
        assert kept_inputs == {1, 2, 13, 14, 15, 16, 17, 18, 19, 20}
        for kept in kept_inputs:
            assert f"task {kept}" in out
            assert f"result {kept}" in out
        for dropped in set(range(1, n + 1)) - kept_inputs:
            assert f"task {dropped}" not in out
            assert f"result {dropped}" not in out
        assert out.count("### Task ") == MAX_TASKS
        assert "### Task 11" not in out

    def test_empty_result_omits_result_block(self) -> None:
        """Entries with empty result should not produce a '### Result' block."""
        chat_id = ""
        task_id, chat_id = _add_task("only-task", chat_id=chat_id)
        _save_task_result(result="", task_id=task_id)

        agent = ChatSorcarAgent("test")
        agent.resume_chat_by_id(chat_id)
        out = agent.build_chat_prompt("now")

        assert "### Task 1\nonly-task" in out
        assert "### Result 1" not in out
