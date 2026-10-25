# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A memory search can be stopped while its index sync runs on a worker thread.

``MemoryTools._search`` runs one sync-and-search per memory on a
``ThreadPoolExecutor`` and waited for them with ``future.result()`` and
the executor's ``shutdown(wait=True)``, both unbounded waits in C.  The
runner's Stop delivers ``KeyboardInterrupt`` with
``PyThreadState_SetAsyncExc``, which only lands when the thread runs
bytecode, so a first-time or format-bump index build (minutes of
embedding requests) made ``memory_search``, ``memory_pull`` and the
task-start ``recall`` deaf to Stop.  The task thread now waits in short
polled slices and leaves the pool without joining it.

The embedder here is a real (slow) embedding function: every page
embedding sleeps :data:`SYNC_SECONDS`, standing in for the network
round trips of a real build.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from kiss.core.memoryfield.index import hashed_embedding
from kiss.core.memoryfield.tools import MemoryTools
from kiss.tests.lock_injection import ignore_thread_interrupt, inject

SYNC_SECONDS = 3.0
STOP_AFTER_SECONDS = 0.3
STOP_LANDS_WITHIN_SECONDS = 1.0


def slow_embedding(text: str) -> list[float]:
    """The offline hashed embedding, as slow as one network request per text."""
    time.sleep(SYNC_SECONDS)
    return hashed_embedding(text)


def _call_into(tools: MemoryTools, tool: str, outcome: list[object]) -> None:
    try:
        outcome.append(getattr(tools, tool)("a page"))
    except BaseException as e:  # noqa: BLE001 - the injected interrupt is the point
        outcome.append(e)


@pytest.fixture
def tools(tmp_path: Path) -> MemoryTools:
    tools = MemoryTools(tmp_path / "memory", embed=slow_embedding)
    tools.memory.write("page", "a page the first search has to embed")
    return tools


@pytest.mark.parametrize("tool", ["memory_search", "memory_pull", "recall"])
def test_stop_lands_while_the_sync_runs(
    tools: MemoryTools, tool: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(threading, "excepthook", ignore_thread_interrupt)
    outcome: list[object] = []
    worker = threading.Thread(target=_call_into, args=(tools, tool, outcome), daemon=True)
    worker.start()
    time.sleep(STOP_AFTER_SECONDS)
    assert worker.is_alive(), "the search finished before the stop"
    assert inject(worker) == 1
    worker.join(STOP_LANDS_WITHIN_SECONDS)
    assert not worker.is_alive(), "the stop did not land while the sync ran"
    assert isinstance(outcome[0], KeyboardInterrupt)
