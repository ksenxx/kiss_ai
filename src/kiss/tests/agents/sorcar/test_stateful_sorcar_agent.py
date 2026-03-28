"""Tests for StatefulSorcarAgent: chat context, prompt augmentation, persistence."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.persistence import _load_last_chat_id
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent


def _redirect(tmpdir: str) -> tuple:
    """Redirect DB to a temp dir and reset the singleton connection."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore(saved: tuple) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _patch_super_run(agent: StatefulSorcarAgent, captured: dict[str, Any]) -> Any:
    """Monkey-patch RelentlessAgent.run to capture the prompt and return YAML."""
    parent_class = cast(Any, SorcarAgent.__mro__[1])  # RelentlessAgent
    original_run = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        captured["prompt_template"] = kwargs.get("prompt_template", "")
        return "success: true\nsummary: test done\n"

    parent_class.run = fake_run
    return original_run


class TestStatefulSorcarAgent:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_chat_id_on_init(self) -> None:
        agent = StatefulSorcarAgent("test")
        assert agent.chat_id
        assert len(agent.chat_id) == 32

    def test_new_chat_resets_id(self) -> None:
        agent = StatefulSorcarAgent("test")
        old_id = agent.chat_id
        agent.new_chat()
        assert agent.chat_id != old_id
        assert len(agent.chat_id) == 32

    def test_first_task_no_context_prompt_unmodified(self) -> None:
        agent = StatefulSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="do stuff")
        finally:
            parent_class.run = original_run

        assert captured["prompt_template"] == "# Task\ndo stuff"

    def test_task_persisted_to_history(self) -> None:
        agent = StatefulSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="my task")
        finally:
            parent_class.run = original_run

        entries = th._load_history(limit=100)
        tasks = [e["task"] for e in entries]
        assert "my task" in tasks

    def test_result_saved_to_history(self) -> None:
        agent = StatefulSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="my task")
        finally:
            parent_class.run = original_run

        context = th._load_chat_context(agent.chat_id)
        assert len(context) == 1
        assert context[0]["task"] == "my task"
        assert context[0]["result"] == "test done"

    def test_second_task_gets_chat_context(self) -> None:
        agent = StatefulSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="task one")
            agent.run(prompt_template="task two")
        finally:
            parent_class.run = original_run

        prompt = str(captured["prompt_template"])
        assert "## Previous tasks and results from the chat session for reference" in prompt
        assert "### Task 1\ntask one" in prompt
        assert "### Result 1\ntest done" in prompt
        assert "# Task (work on it now)\n\ntask two" in prompt

    def test_resume_chat_restores_chat_id(self) -> None:
        agent = StatefulSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="original task")
        finally:
            parent_class.run = original_run

        original_chat_id = agent.chat_id

        agent2 = StatefulSorcarAgent("test2")
        assert agent2.chat_id != original_chat_id
        agent2.resume_chat("original task")
        assert agent2.chat_id == original_chat_id

    def test_resume_chat_no_match_keeps_id(self) -> None:
        agent = StatefulSorcarAgent("test")
        old_id = agent.chat_id
        agent.resume_chat("nonexistent task")
        assert agent.chat_id == old_id

    def test_error_saves_failure_result(self) -> None:
        agent = StatefulSorcarAgent("test")
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        original_run = parent_class.run

        def failing_run(self_agent: object, **kwargs: object) -> str:
            raise RuntimeError("boom")

        parent_class.run = failing_run
        try:
            try:
                agent.run(prompt_template="failing task")
            except RuntimeError:
                pass
        finally:
            parent_class.run = original_run

        context = th._load_chat_context(agent.chat_id)
        assert len(context) == 1
        assert context[0]["task"] == "failing task"
        assert "Task failed" in str(context[0]["result"])

    def test_chat_context_entry_with_default_result(self) -> None:
        """When a prior task has only the default result, both headers appear."""
        agent = StatefulSorcarAgent("test")
        # _add_task inserts a default result ("Agent Failed Abrubptly")
        th._add_task("bare task", chat_id=agent.chat_id)

        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="follow up")
        finally:
            parent_class.run = original_run

        prompt = str(captured["prompt_template"])
        assert "### Task 1\nbare task" in prompt
        assert "### Result 1" in prompt

    def test_chat_id_property(self) -> None:
        agent = StatefulSorcarAgent("test")
        assert agent.chat_id == agent._chat_id

    def test_run_returns_super_result(self) -> None:
        agent = StatefulSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            result = agent.run(prompt_template="do stuff")
        finally:
            parent_class.run = original_run

        assert "success: true" in result
        assert "summary: test done" in result

    def test_kwargs_passed_through(self) -> None:
        agent = StatefulSorcarAgent("test")
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        original_run = parent_class.run
        captured: dict[str, Any] = {}

        def capture_run(self_agent: object, **kwargs: object) -> str:
            captured.update(kwargs)
            return "success: true\nsummary: ok\n"

        parent_class.run = capture_run
        try:
            agent.run(prompt_template="task", model_name="gpt-4", max_budget=5.0)
        finally:
            parent_class.run = original_run

        assert captured["model_name"] == "gpt-4"
        assert captured["max_budget"] == 5.0

    def test_multiple_tasks_build_full_context(self) -> None:
        agent = StatefulSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="task A")
            agent.run(prompt_template="task B")
            agent.run(prompt_template="task C")
        finally:
            parent_class.run = original_run

        prompt = str(captured["prompt_template"])
        assert "### Task 1\ntask A" in prompt
        assert "### Task 2\ntask B" in prompt
        assert "### Result 1\ntest done" in prompt
        assert "### Result 2\ntest done" in prompt
        assert "# Task (work on it now)\n\ntask C" in prompt

    def test_load_last_chat_id_returns_most_recent(self) -> None:
        agent = StatefulSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="some task")
        finally:
            parent_class.run = original_run

        assert _load_last_chat_id() == agent.chat_id

    def test_load_last_chat_id_empty_when_no_chat_id(self) -> None:
        """Tasks with empty chat_id return empty string."""
        th._add_task("bare task", chat_id="")
        assert _load_last_chat_id() == ""

    def test_build_chat_prompt_no_context(self) -> None:
        agent = StatefulSorcarAgent("test")
        result = agent.build_chat_prompt("do something")
        assert result == "# Task\ndo something"

    def test_build_chat_prompt_with_context(self) -> None:
        agent = StatefulSorcarAgent("test")
        th._add_task("prior task", chat_id=agent.chat_id)
        th._set_latest_chat_events([], task="prior task", result="prior result")
        result = agent.build_chat_prompt("new task")
        assert "## Previous tasks and results from the chat session for reference" in result
        assert "### Task 1\nprior task" in result
        assert "### Result 1\nprior result" in result
        assert "# Task (work on it now)\n\nnew task" in result

    def test_build_chat_prompt_default_result_entry(self) -> None:
        agent = StatefulSorcarAgent("test")
        th._add_task("bare task", chat_id=agent.chat_id)
        result = agent.build_chat_prompt("follow up")
        assert "### Task 1\nbare task" in result
        assert "### Result 1" in result
        assert "# Task (work on it now)\n\nfollow up" in result

    def test_new_chat_clears_context(self) -> None:
        agent = StatefulSorcarAgent("test")
        captured: dict[str, Any] = {}
        original_run = _patch_super_run(agent, captured)
        parent_class = cast(Any, SorcarAgent.__mro__[1])
        try:
            agent.run(prompt_template="old task")
            agent.new_chat()
            agent.run(prompt_template="fresh start")
        finally:
            parent_class.run = original_run

        # After new_chat, prompt should not include old context
        prompt = str(captured["prompt_template"])
        assert "old task" not in prompt
        assert prompt == "# Task\nfresh start"
