# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The tool modules' locks survive the runner's Stop injection on a task thread.

``sea_commands._lock`` is the most contended lock of the tool modules:
every ``run_agent`` call resolves its SEA through
:func:`~kiss.agents.sorcar.sea_commands.get_command`, every task start
checks its model through :func:`~kiss.agents.sorcar.sea_commands.model_sea`,
and the registry watcher republishes under ``_notify_lock`` -> ``_lock``
while those calls wait.  ``MCPManager._lock`` is taken by every MCP
tool call, and once more by the done callback a timed-out ``connect``
registers on its straggler's future.  The Stop watchdog delivers
``KeyboardInterrupt`` to the task thread with
``PyThreadState_SetAsyncExc``; a thread blocked in a contended
``acquire`` receives it at the check point right after ``__enter__``
returns, outside the ``with`` block (CPython gh-74174), so a plain
``with lock:`` there leaks the lock for the daemon's life and every
later call of every task hangs on it.  Every site a task thread
reaches therefore goes through :func:`~kiss.core.locked.locked_call`,
and the straggler's callback is registered on the manager loop: the
stdlib's ``Future.add_done_callback`` takes the future's condition
with a plain ``with``, which a task thread must not run.

Each hammer runs in a child process (a leaked lock would wedge the
pytest process): a contender thread holds the lock half the time so
the worker's ``acquire`` really blocks, and the worker loops one real
path while hundreds of interrupts are injected into it.  The test
fails on the pre-fix ``with`` sites and passes on the fixed ones.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from typing import Any

import pytest

from kiss.agents.sorcar import mcp_servers, sea_commands
from kiss.tests.lock_injection import hammer


class _ModelLookup:
    """``model_sea`` and a ``get_command`` hit on the worker: the registry reads under ``_lock``."""

    def __call__(self) -> None:
        sea_commands.model_sea("slack")
        sea_commands.get_command("slack")


class _MissRefresh:
    """``get_command`` of an unknown name: the rescan republishes under both registry locks."""

    def __init__(self) -> None:
        self.notified = 0
        sea_commands.subscribe(self._note)

    def _note(self, names: list[str]) -> None:
        self.notified += 1

    def __call__(self) -> None:
        sea_commands.get_command("audit1013-no-such-sea")


async def _finished() -> None:
    """A server task that is already over by the time its connection is reaped."""


class _ReapFinishedStraggler:
    """``_reap_straggler`` of a connection whose task is done, on the worker."""

    def __init__(self) -> None:
        self.manager = mcp_servers.MCPManager()
        config = mcp_servers.MCPServerConfig(name="audit1013", transport="stdio", command="x")
        self.conn = mcp_servers._Connection(config=config)
        self.conn.task = asyncio.run_coroutine_threadsafe(_finished(), self.manager._loop)
        self.conn.task.result(5.0)

    def __call__(self) -> None:
        self.manager._reap_straggler(self.conn, self.conn.task)


_DRIVERS: dict[str, type[_ModelLookup | _MissRefresh | _ReapFinishedStraggler]] = {
    "model_lookup": _ModelLookup, "miss_refresh": _MissRefresh,
    "reap_finished_straggler": _ReapFinishedStraggler,
}


def run_hammer(name: str) -> None:
    """Child-process body: hammer the path *name* and check its module's locks stay free."""
    sea_commands.list_commands()
    driver = _DRIVERS[name]()
    if isinstance(driver, _ReapFinishedStraggler):
        locks: dict[str, object] = {"MCPManager._lock": driver.manager._lock}
        contended = driver.manager._lock
    else:
        locks = {"_lock": sea_commands._lock, "_notify_lock": sea_commands._notify_lock}
        contended = sea_commands._notify_lock if name == "miss_refresh" else sea_commands._lock
    # The locks are checked once the contender is gone: it holds them most of the time.
    hammer(driver, locks, contend=contended)
    if isinstance(driver, _ReapFinishedStraggler):
        driver.manager.shutdown()
    # The registry still works afterwards: the daemon's next ``run_agent`` resolves.
    assert sea_commands.get_command("slack") is not None


@pytest.mark.parametrize("name", sorted(_DRIVERS))
def test_tool_lock_survives_injection(name: str) -> None:
    """No lock site a task thread runs in the tool modules leaks the lock under injection."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer({name!r})"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, f"{name}: {proc.stderr[-3000:]}"


_CONDITION_ENTER = threading.Condition.__enter__.__code__


def _stop_on_condition_enter(frame: Any, event: str, arg: Any) -> Any:
    """Trace function raising the Stop where ``Condition.__enter__`` returns (gh-74174)."""
    if event == "call" and frame.f_code is _CONDITION_ENTER:
        return _stop_on_enter_return
    return None


def _stop_on_enter_return(frame: Any, event: str, arg: Any) -> Any:
    if event == "return":
        raise KeyboardInterrupt
    return _stop_on_enter_return


def test_a_stop_in_the_straggler_reap_leaves_the_futures_condition_free() -> None:
    """A Stop on the task thread cannot leave the straggler's future condition held.

    ``Future.add_done_callback`` takes the future's condition with a
    plain ``with``; a Stop landing right after that ``__enter__`` on
    the task thread kept the condition for the thread's life, and the
    loop's own completion of the task, like every other taker, hung on
    it.  The Stop is delivered through a trace function at exactly that
    point, should the reap enter a condition on the task thread at all.
    """
    driver = _ReapFinishedStraggler()
    task = driver.conn.task
    stopped: list[bool] = []

    def reap_under_a_stop() -> None:
        sys.settrace(_stop_on_condition_enter)
        try:
            driver()
        except KeyboardInterrupt:
            stopped.append(True)
        finally:
            sys.settrace(None)

    try:
        worker = threading.Thread(target=reap_under_a_stop, daemon=True)
        worker.start()
        worker.join(10.0)
        assert not worker.is_alive(), "the reap hung"
        # A bounded take rather than a probe: the loop may still be inside
        # the registration the reap handed it, holding the condition briefly.
        taken = task._condition.acquire(timeout=5.0)
        assert taken, f"the future's condition stayed held: {stopped}"
        task._condition.release()
        assert task.result(5.0) is None  # another thread uses the future
    finally:
        driver.manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
