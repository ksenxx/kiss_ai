# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt iteration 5: non-string ``summary`` values must not crash run().

``ChatSorcarAgent.run`` extracts ``result_yaml.get("summary", "")`` and
passes it verbatim to ``_save_task_result`` in the ``finally`` block.
LLMs routinely emit a YAML *list* (or nested dict) under ``summary``::

    success: true
    summary:
      - did this
      - did that

Pre-fix, the list/dict reached sqlite parameter binding which raised
``sqlite3.ProgrammingError: type 'list' is not supported`` FROM THE
``finally`` BLOCK — replacing the task's successful return value with an
exception and skipping ``_save_task_extra`` (tokens/cost lost too).
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
        agent = ChatSorcarAgent("bughunt5-summary")
        returned = agent.run(prompt_template="my task")
        assert returned == agent_result
        context = _load_chat_context(agent.chat_id)
        assert context, "task was not persisted at all"
        return str(context[-1].get("result") or "")
    finally:
        parent_class.run = original_run
        _restore_db(saved)


def test_list_summary_does_not_crash_and_persists_text(tmp_path: Path) -> None:
    """``summary:`` holding a YAML list must persist as text, not raise."""
    persisted = _run_and_get_persisted_result(
        tmp_path, "success: true\nsummary:\n  - item one\n  - item two\n",
    )
    assert "item one" in persisted and "item two" in persisted


def test_dict_summary_does_not_crash_and_persists_text(tmp_path: Path) -> None:
    """``summary:`` holding a nested mapping must persist as text."""
    persisted = _run_and_get_persisted_result(
        tmp_path, "success: true\nsummary:\n  outcome: fixed\n  files: 3\n",
    )
    assert "outcome" in persisted and "fixed" in persisted


def test_none_summary_persists_empty_string(tmp_path: Path) -> None:
    """``summary:`` with no value (None) must persist '' without crashing."""
    persisted = _run_and_get_persisted_result(
        tmp_path, "success: true\nsummary:\n",
    )
    assert persisted == ""


def test_string_summary_regression_guard(tmp_path: Path) -> None:
    """Plain string summaries keep working unchanged."""
    persisted = _run_and_get_persisted_result(
        tmp_path, "success: true\nsummary: the summary\n",
    )
    assert persisted == "the summary"
