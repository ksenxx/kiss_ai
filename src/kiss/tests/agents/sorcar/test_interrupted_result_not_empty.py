# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the task-3624 empty-result data loss.

Incident (forensics from ``~/.kiss/sorcar.db``): when the kiss-web
daemon was SIGTERMed on 2026-06-11 00:37:45 with 7 parallel sub-agents
in flight, the shutdown's cooperative stop event made each sub-agent's
printer raise ``KeyboardInterrupt``.  One sub-agent (task_history row
3624) unwound far enough to run ``ChatSorcarAgent.run``'s ``finally``
block — which persisted ``result_summary`` still at its initial ``""``
because only ``except Exception`` (not ``KeyboardInterrupt``, a
``BaseException``) rewrites it.

That empty string *replaced* the ``"Agent Failed Abruptly"`` sentinel,
so the next startup's orphan sweep (which only repairs rows whose
result is exactly the sentinel) could no longer recover the row: task
3624 shows an empty result forever, indistinguishable from a task that
produced nothing.

Fix under test: an interruption (``KeyboardInterrupt`` or any other
non-``Exception`` ``BaseException``) persists the explicit
``"Task interrupted"`` marker instead of the empty string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

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


def _run_interrupted_and_get_persisted_result(
    tmp_path: Path, exc: BaseException,
) -> str:
    """Run a ChatSorcarAgent whose underlying run raises *exc*; return
    the result text persisted for the task row."""
    saved = _redirect_db(tmp_path)
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original_run = parent_class.run

    def raising_run(self_agent: Any, **kwargs: Any) -> str:
        raise exc

    parent_class.run = raising_run
    try:
        agent = ChatSorcarAgent("interrupted-result")
        with pytest.raises(type(exc)):
            agent.run(prompt_template="my task")
        context = _load_chat_context(agent.chat_id)
        assert context, "task was not persisted at all"
        return str(context[-1].get("result") or "")
    finally:
        parent_class.run = original_run
        _restore_db(saved)


def test_keyboard_interrupt_persists_explicit_marker(tmp_path: Path) -> None:
    """A KeyboardInterrupt (user Stop / daemon shutdown) must not persist
    an empty result — that wipes the orphan-sweep sentinel and leaves the
    row unrecoverable (the task-3624 incident)."""
    persisted = _run_interrupted_and_get_persisted_result(
        tmp_path, KeyboardInterrupt("simulated stop during shutdown"),
    )
    assert persisted != "", (
        "interrupted run persisted an empty result over the 'Agent Failed "
        "Abruptly' sentinel — the orphan sweep can never repair this row"
    )
    assert persisted == "Task interrupted"


def test_system_exit_persists_explicit_marker(tmp_path: Path) -> None:
    """Other non-Exception BaseExceptions take the same recovery path."""
    persisted = _run_interrupted_and_get_persisted_result(
        tmp_path, SystemExit(1),
    )
    assert persisted == "Task interrupted"


def test_plain_exception_still_persists_task_failed(tmp_path: Path) -> None:
    """Regression guard: ordinary exceptions keep the 'Task failed' label."""
    persisted = _run_interrupted_and_get_persisted_result(
        tmp_path, RuntimeError("boom"),
    )
    assert persisted == "Task failed"
