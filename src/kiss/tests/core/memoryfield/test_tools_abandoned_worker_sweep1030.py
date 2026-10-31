# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""What a stopped memory search leaves running, and how a panel Stop lands.

* ``MemoryTools._search`` abandons its sync-and-search workers when the
  task thread is stopped.  The worker used to run the WHOLE index build
  (``VectorIndex.sync`` had no stop check between embedding batches), and
  interpreter exit joins ``ThreadPoolExecutor`` threads, so a stopped
  first-time build kept the process alive until it ended.  The worker now
  starts no embedding batch after the stop.
* The task thread's polled wait did not check the tool panel's
  cooperative interrupt, so a panel Stop of ``memory_search`` waited for
  the watchdog's forced injection; it now waits in
  ``tool_interrupt.result_interruptible`` (as MCP tool calls do), which
  honours it at the next wait slice.

The embedder is a real (slow) embedding: each ``embed_many`` request
sleeps :data:`BATCH_SECONDS`, standing in for one network round trip.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from kiss.core import tool_interrupt
from kiss.core.memoryfield.index import EMBED_BATCH_INPUTS, hashed_embedding
from kiss.core.memoryfield.tools import MemoryTools
from kiss.tests.lock_injection import ignore_thread_interrupt, inject

BATCH_SECONDS = 1.0


class SlowBatchEmbedding:
    """Hashed embeddings; each batch request sleeps like a network round trip."""

    def __init__(self) -> None:
        self.batches: list[int] = []
        self.batch_started = threading.Event()
        self.worker: threading.Thread | None = None

    def __call__(self, text: str) -> list[float]:
        """Embed one text (the search query) at once."""
        return hashed_embedding(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed one sync batch, slowly, recording its size and thread."""
        self.batches.append(len(texts))
        self.worker = threading.current_thread()
        self.batch_started.set()
        time.sleep(BATCH_SECONDS)
        return [hashed_embedding(text) for text in texts]


def _tools(root: Path, embedding: SlowBatchEmbedding, pages: int) -> MemoryTools:
    tools = MemoryTools(root, embed=embedding)
    for i in range(pages):
        tools.memory.write(f"page-{i}", f"note number {i} about the build")
    return tools


def _search_into(tools: MemoryTools, outcome: list[object]) -> None:
    try:
        outcome.append(tools.memory_search("note about the build"))
    except BaseException as e:  # noqa: BLE001 - the stop is the point
        outcome.append(e)


def test_a_stopped_search_starts_no_further_embedding_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(threading, "excepthook", ignore_thread_interrupt)
    embedding = SlowBatchEmbedding()
    tools = _tools(tmp_path / "memory", embedding, EMBED_BATCH_INPUTS + 10)
    outcome: list[object] = []
    task = threading.Thread(target=_search_into, args=(tools, outcome), daemon=True)
    task.start()
    assert embedding.batch_started.wait(60), "the first embedding batch never started"
    assert inject(task) == 1
    task.join(5)
    assert not task.is_alive() and isinstance(outcome[0], KeyboardInterrupt)
    assert embedding.worker is not None
    embedding.worker.join(BATCH_SECONDS * 5)
    assert not embedding.worker.is_alive(), "the abandoned worker kept building the index"
    assert embedding.batches == [EMBED_BATCH_INPUTS]
    # The batch in flight was kept; the next search embeds only the rest.
    assert "page-" in tools.memory_search("note about the build")
    assert embedding.batches == [EMBED_BATCH_INPUTS, 10]


def _tool_call_into(tools: MemoryTools, box: dict[str, object]) -> None:
    """Run ``memory_search`` as a registered tool call, the way the agent does."""
    token = tool_interrupt.new_tool_call("memory_search")
    box["token"] = token
    tool_interrupt.register_tool_call(token)
    try:
        box["result"] = tools.memory_search("note about the build")
    except BaseException as e:  # noqa: BLE001 - the interrupt is the point
        box["result"] = e
    finally:
        tool_interrupt.end_tool_call(token)
        tool_interrupt.unregister_tool_call(token)


def test_a_tool_panel_stop_is_honoured_without_forced_injection(tmp_path: Path) -> None:
    embedding = SlowBatchEmbedding()
    tools = _tools(tmp_path / "memory", embedding, 1)
    box: dict[str, object] = {}
    task = threading.Thread(target=_tool_call_into, args=(tools, box), daemon=True)
    task.start()
    assert embedding.batch_started.wait(60), "the embedding batch never started"
    assert task.ident is not None
    assert tool_interrupt.interrupt_tool_call(task.ident, "memory_search")
    task.join(5)
    assert not task.is_alive()
    assert isinstance(box["result"], tool_interrupt.ToolCallInterrupted)
    token = box["token"]
    assert isinstance(token, tool_interrupt.ToolCallToken)
    assert not token.injected, "the Stop needed the watchdog's forced injection"
    assert embedding.worker is not None
    embedding.worker.join(BATCH_SECONDS * 5)
