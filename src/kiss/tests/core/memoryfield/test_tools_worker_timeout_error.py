# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A search worker whose own failure is a ``TimeoutError`` must not hang the search.

``MemoryTools._search`` waits for each worker future in short
``future.result(timeout=...)`` slices and catches the ``TimeoutError``
those slices raise so a Stop can land between them.  A worker whose
embedder times out on the wire raises a ``TimeoutError`` of its own
(``socket.timeout`` is that very class), which ``future.result()``
re-raises instantly on every poll: the wait became an infinite busy
loop that hung ``memory_search``, ``memory_pull`` and the task-start
``recall`` and bypassed recall's error fallback.  The embedder here does
a real socket read with a 10 ms timeout, standing in for an embedding
request whose server never answers.
"""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest

from kiss.core.memoryfield.tools import MemoryTools

READ_TIMEOUT_SECONDS = 0.01
FINISHES_WITHIN_SECONDS = 2.0


def embedding_over_a_silent_socket(text: str) -> list[float]:
    """An embedding request to a server that never answers: times out on the read."""
    ours, theirs = socket.socketpair()
    try:
        ours.settimeout(READ_TIMEOUT_SECONDS)
        ours.recv(1)  # raises TimeoutError (socket.timeout) after READ_TIMEOUT_SECONDS
    finally:
        ours.close()
        theirs.close()
    raise AssertionError("the silent socket answered")


def _call_into(tools: MemoryTools, tool: str, outcome: list[object]) -> None:
    try:
        outcome.append(getattr(tools, tool)("a page"))
    except BaseException as e:  # noqa: BLE001 - the worker's TimeoutError is the point
        outcome.append(e)


@pytest.fixture
def tools(tmp_path: Path) -> MemoryTools:
    tools = MemoryTools(tmp_path / "memory", embed=embedding_over_a_silent_socket)
    tools.memory.write("page", "a page the first search has to embed")
    return tools


@pytest.mark.parametrize("tool", ["memory_search", "memory_pull"])
def test_search_and_pull_raise_the_workers_timeout_error(tools: MemoryTools, tool: str) -> None:
    outcome: list[object] = []
    worker = threading.Thread(target=_call_into, args=(tools, tool, outcome), daemon=True)
    worker.start()
    worker.join(FINISHES_WITHIN_SECONDS)
    assert not worker.is_alive(), f"{tool} never returned after its worker timed out"
    assert isinstance(outcome[0], TimeoutError)


def test_recall_falls_back_to_empty_when_its_worker_times_out(tools: MemoryTools) -> None:
    outcome: list[object] = []
    worker = threading.Thread(target=_call_into, args=(tools, "recall", outcome), daemon=True)
    worker.start()
    worker.join(FINISHES_WITHIN_SECONDS)
    assert not worker.is_alive(), "recall never returned after its worker timed out"
    assert outcome == [""]
