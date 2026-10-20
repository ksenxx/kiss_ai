# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A memory write or delete waits a bounded time for ``<root>.lock``.

``MemoryDir._mutation_lock`` took the memory's cross-process lock with
no timeout: a ``flock`` blocking in C that a wedged holder (a sync
script stuck on a network filesystem, a daemon paused in a debugger)
kept forever, and that the task's Stop could not interrupt.  The wait
is now bounded by ``pages.MUTATION_LOCK_TIMEOUT`` and the tools turn
the ``TimeoutError`` into an ``Error:`` reply for the agent.

The tests hold the lock themselves through a second open file
description, which excludes the tool exactly as another process would,
with the bound lowered so the suite stays fast.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from kiss.core.file_lock import exclusive_file_lock
from kiss.core.memoryfield import pages
from kiss.core.memoryfield.index import hashed_embedding
from kiss.core.memoryfield.tools import MemoryTools


def _call_into(fn, args: tuple, results: list[str]) -> None:
    results.append(fn(*args))


def _reply_while_locked(tools: MemoryTools, fn, *args) -> str:
    """Call the tool *fn* on a thread while the memory lock is held; return its reply."""
    results: list[str] = []
    worker = threading.Thread(target=_call_into, args=(fn, args, results), daemon=True)
    with exclusive_file_lock(Path(f"{tools.memory.root}.lock")):
        worker.start()
        worker.join(5)
        assert not worker.is_alive(), "the mutation is still waiting for the lock"
    return results[0]


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryTools:
    monkeypatch.setattr(pages, "MUTATION_LOCK_TIMEOUT", 1.0)
    return MemoryTools(tmp_path / "memory", embed=hashed_embedding)


def test_memory_write_reports_a_lock_it_could_not_get(tools: MemoryTools) -> None:
    reply = _reply_while_locked(tools, tools.memory_write, "p", "body")
    assert reply.startswith("Error:") and "still locked by another process" in reply
    assert not tools.memory.page_path("p").exists()


def test_memory_delete_reports_a_lock_it_could_not_get(tools: MemoryTools) -> None:
    tools.memory_write("p", "body")
    reply = _reply_while_locked(tools, tools.memory_delete, "p")
    assert reply.startswith("Error:") and "still locked by another process" in reply
    assert tools.memory.page_path("p").exists()
    assert tools.memory_delete("p") == "Deleted p."
