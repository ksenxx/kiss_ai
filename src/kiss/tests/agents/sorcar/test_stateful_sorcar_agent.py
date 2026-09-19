# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ChatSorcarAgent: chat context, prompt augmentation, persistence."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import MAX_TASKS, ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _save_task_result,
)


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


def _seed_long_chat(n: int, result_chars: int) -> str:
    chat_id = ""
    for i in range(1, n + 1):
        task_id, chat_id = _add_task(f"task {i} " + "T" * 700, chat_id=chat_id)
        result = f"<h3>Result {i} heading</h3><p>{'r' * result_chars}</p><pre>code {i}</pre>"
        _save_task_result(result=result, task_id=task_id)
    return chat_id


class TestBuildChatPromptDigest:
    """Older results are digested and the prefix is bounded (cost lever WP5)."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_newest_two_full_older_digested(self) -> None:
        chat_id = _seed_long_chat(4, result_chars=1000)
        agent = ChatSorcarAgent("test")
        agent.resume_chat_by_id(chat_id)
        out = agent.build_chat_prompt("now")
        # Tasks 3 and 4: full HTML result and full task text.
        assert "<pre>code 3</pre>" in out and "<pre>code 4</pre>" in out
        assert "task 4 " + "T" * 700 in out
        # Tasks 1 and 2: tag-stripped digest cut to 300 chars, task text cut to 600.
        assert "<pre>code 1</pre>" not in out and "<h3>" not in out.split("### Task 3")[0]
        digest_1 = out.split("### Result 1\n")[1].split("\n\n### Task 2")[0]
        assert digest_1.startswith("Result 1 heading rrrr")
        assert digest_1.endswith(" …") and len(digest_1) <= 305
        task_1 = out.split("### Task 1\n")[1].split("\n\n### Result 1")[0]
        assert len(task_1) <= 605 and task_1.endswith(" …")
        assert out.endswith("# Task (work on it now)\n\nnow")

    def test_prefix_is_bounded_by_dropping_oldest(self) -> None:
        # 10 tasks with 4k-char results: everything older than the two
        # newest is dropped; two full results (~9.6k with their task text)
        # still exceed the 6k cap, so the older of the two is digested and
        # only the newest stays whole.
        chat_id = _seed_long_chat(10, result_chars=4000)
        agent = ChatSorcarAgent("test")
        agent.resume_chat_by_id(chat_id)
        out = agent.build_chat_prompt("now")
        assert out.count("### Task ") == 2
        assert "<pre>code 10</pre>" in out
        assert "<pre>code 9</pre>" not in out and "Result 9 heading" in out
        assert "code 8" not in out
        body = out.split("for reference\n\n")[1].split("\n\n---\n\n# Task")[0]
        assert len(body) <= 6_000
        # Numbering restarts at 1 for the kept entries.
        assert "### Task 1\n" in out and "### Task 2\n" in out

    def test_digest_disabled_keeps_full_history(self, monkeypatch) -> None:
        from kiss.core.config import DEFAULT_CONFIG

        monkeypatch.setattr(DEFAULT_CONFIG, "chat_history_digest", False)
        chat_id = _seed_long_chat(4, result_chars=1000)
        agent = ChatSorcarAgent("test")
        agent.resume_chat_by_id(chat_id)
        out = agent.build_chat_prompt("now")
        assert out.count("### Task ") == 4
        assert "<pre>code 1</pre>" in out

    def test_cap_wins_over_the_newest_full_results(self) -> None:
        # Two 20k-char results cannot both stay whole under a 6k cap.
        chat_id = _seed_long_chat(2, result_chars=20_000)
        agent = ChatSorcarAgent("test")
        agent.resume_chat_by_id(chat_id)
        out = agent.build_chat_prompt("now")
        body = out.split("for reference\n\n")[1].split("\n\n---\n\n# Task")[0]
        assert len(body) <= 6_000
        assert "<pre>code 2</pre>" not in out and "Result 2 heading" in out
