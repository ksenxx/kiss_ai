"""Tests for ChatSorcarAgent: chat context, prompt augmentation, persistence."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _load_last_chat_id
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
