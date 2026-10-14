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
registers on its straggler's future: a future that already finished
runs that callback inline, on the task thread.  The Stop watchdog
delivers ``KeyboardInterrupt`` to the task thread with
``PyThreadState_SetAsyncExc``; a thread blocked in a contended
``acquire`` receives it at the check point right after ``__enter__``
returns, outside the ``with`` block (CPython gh-74174), so a plain
``with lock:`` there leaks the lock for the daemon's life and every
later call of every task hangs on it.  Every site a task thread
reaches therefore goes through :func:`~kiss.core.locked.locked_call`.

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
    """``_reap_straggler`` of a connection whose task is done: the callback runs on the worker."""

    def __init__(self) -> None:
        self.manager = mcp_servers.MCPManager()
        config = mcp_servers.MCPServerConfig(name="audit1013", transport="stdio", command="x")
        self.conn = mcp_servers._Connection(config=config)
        self.conn.task = asyncio.run_coroutine_threadsafe(_finished(), self.manager._loop)
        self.conn.task.result(5.0)

    def __call__(self) -> None:
        self.manager._reap_straggler(self.conn)


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


if __name__ == "__main__":
    pytest.main([__file__])
