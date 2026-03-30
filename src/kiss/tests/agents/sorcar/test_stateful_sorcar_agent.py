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
