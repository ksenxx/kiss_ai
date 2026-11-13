# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``frequent_tasks`` table of history.db: kept in the schema, no longer written.

The table fed the composer's Frequent-tasks menu.  The menu is gone and
no code reads the rows, so a run no longer upserts one.  The table
stays, so that a database from an earlier release keeps its rows and
``carry_over_tables.py`` can still move them into a replaced database.
"""

from __future__ import annotations

from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    history_rows,
)


def _finish(request: dict[str, Any]) -> dict[str, Any]:
    """Answer every model request with a ``finish`` call."""
    return finish_response("done")


def test_a_run_writes_no_row_into_the_table_it_keeps() -> None:
    """A top-level run lands in ``task_history`` and leaves ``frequent_tasks`` empty."""
    env = IsolatedKissHome("kiss-frequent-tasks-")
    server = StandInModelServer(_finish)
    try:
        ChatSorcarAgent("frequent-tasks-chat").run(
            prompt_template="plain user task",
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
        )
        assert [row["task"] for row in history_rows()] == ["plain user task"]
        assert th._get_db().execute("SELECT COUNT(*) FROM frequent_tasks").fetchone()[0] == 0
    finally:
        server.stop()
        env.cleanup()
