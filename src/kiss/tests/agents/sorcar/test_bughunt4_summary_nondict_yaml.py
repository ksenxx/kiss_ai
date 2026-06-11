# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt4: ChatSorcarAgent.run must not persist '' for non-dict YAML results.

When the agent result string fails to parse as YAML, ``run()`` falls back
to persisting ``result[:500]``.  But when the result parses successfully
as a *non-dict* YAML document (a plain string, a list, a number), the
summary stayed ``""`` and an empty result was persisted to task history —
inconsistent with the parse-failure fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _load_chat_context
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


def _redirect_db(tmpdir: Path) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = tmpdir / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _run_and_get_persisted_result(tmp_path: Path, agent_result: str) -> str:
    """Run a ChatSorcarAgent whose underlying run returns *agent_result*."""
    saved = _redirect_db(tmp_path)
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original_run = parent_class.run

    def fake_run(self_agent: Any, **kwargs: Any) -> str:
        return agent_result

    parent_class.run = fake_run
    try:
        agent = ChatSorcarAgent("bughunt4-summary")
        agent.run(prompt_template="my task")
        context = _load_chat_context(agent.chat_id)
        assert context, "task was not persisted at all"
        return str(context[-1].get("result") or "")
    finally:
        parent_class.run = original_run
        _restore_db(saved)


def test_plain_string_result_persists_text(tmp_path: Path) -> None:
    """A plain-string result (valid non-dict YAML) must be persisted."""
    result = _run_and_get_persisted_result(
        tmp_path, "all done as a plain string",
    )
    assert result == "all done as a plain string"


def test_yaml_list_result_persists_text(tmp_path: Path) -> None:
    """A YAML-list result (valid non-dict YAML) must be persisted."""
    result = _run_and_get_persisted_result(tmp_path, "- alpha\n- beta\n")
    assert result == "- alpha\n- beta\n"


def test_dict_result_still_uses_summary_key(tmp_path: Path) -> None:
    """Regression guard: dict results keep persisting the summary value."""
    result = _run_and_get_persisted_result(
        tmp_path, "success: true\nsummary: the summary\n",
    )
    assert result == "the summary"
