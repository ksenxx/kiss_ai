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
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _SubagentStopEvent,
    run_tasks_parallel,
)


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
    """A parent stop must reach the sub-agent's ``_stop_event``.

    ``SorcarAgent.run`` resolves ``self._stop_event`` from the *worker*
    thread's ``printer._thread_local.stop_event``, so unless
    ``run_tasks_parallel`` binds a stop event into the worker
    thread-local, the sub-agent sees ``None``.

    The child gets its OWN event chained to the parent's rather than
    the parent object itself, so that stopping one sub-agent does not
    stop the parent and its siblings; a parent stop still reaches the
    child through the chain.
    """
    saved = _redirect_db(tmp_path)
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original_run = parent_class.run
    captured: dict[str, Any] = {}

    def fake_run(self_agent: Any, **kwargs: Any) -> str:
        captured["stop_event"] = getattr(self_agent, "_stop_event", None)
        return "success: true\nsummary: done\n"

    parent_class.run = fake_run
    printer = _Printer()
    ev = threading.Event()
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
    child_ev = captured.get("stop_event")
    assert isinstance(child_ev, _SubagentStopEvent), (
        "run_tasks_parallel did not bind a per-sub-agent stop event "
        f"into the worker thread (got {child_ev!r})"
    )
    assert child_ev is not ev, (
        "the child was handed the PARENT's own event, so stopping the "
        "child would stop the parent and its siblings"
    )
    assert not child_ev.is_set()
    ev.set()
    assert child_ev.is_set(), (
        "a parent stop must still reach the child through the chain"
    )
