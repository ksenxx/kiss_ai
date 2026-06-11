# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt4: module-level run_tasks_parallel must propagate the parent stop_event.

``ChatSorcarAgent._run_tasks_parallel`` copies the parent thread's
``printer._thread_local.stop_event`` into each worker thread-local so
sub-agents can be aborted, but the module-level
:func:`kiss.agents.sorcar.sorcar_agent.run_tasks_parallel` (used by plain
``SorcarAgent``) only captured ``task_id`` — sub-agents spawned through it
never saw the parent stop event, so Stop could not kill their Bash
process groups.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, run_tasks_parallel


class _Printer:
    """Minimal printer stand-in: only the thread-local channel is needed."""

    def __init__(self) -> None:
        self._thread_local = threading.local()


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


def test_module_level_run_tasks_parallel_propagates_stop_event(
    tmp_path: Path,
) -> None:
    """The parent's stop_event must reach the sub-agent's ``_stop_event``.

    ``SorcarAgent.run`` resolves ``self._stop_event`` from the *worker*
    thread's ``printer._thread_local.stop_event``, so unless
    ``run_tasks_parallel`` copies the parent's event into the worker
    thread-local, the sub-agent sees ``None``.
    """
    saved = _redirect_db(tmp_path)
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original_run = parent_class.run
    captured: dict[str, Any] = {}

    def fake_run(self_agent: Any, **kwargs: Any) -> str:
        # SorcarAgent.run has already resolved self._stop_event from the
        # worker thread-local by the time it delegates here.
        captured["stop_event"] = getattr(self_agent, "_stop_event", None)
        return "success: true\nsummary: done\n"

    parent_class.run = fake_run
    printer = _Printer()
    ev = threading.Event()
    # The calling (parent) thread carries the stop event, exactly as the
    # VS Code task runner sets it before SorcarAgent.run executes.
    printer._thread_local.stop_event = ev
    printer._thread_local.task_id = "42"
    try:
        results = run_tasks_parallel(
            ["task one"], max_workers=1, printer=cast(Any, printer),
        )
    finally:
        parent_class.run = original_run
        _restore_db(saved)

    assert len(results) == 1
    assert captured.get("stop_event") is ev, (
        "module-level run_tasks_parallel did not propagate the parent "
        f"stop_event into the worker thread (got {captured.get('stop_event')!r})"
    )
