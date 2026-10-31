# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``result_interruptible`` must not busy-spin on a finished future whose result is a TimeoutError.

Review 2026-10-12 (core).  ``concurrent.futures.TimeoutError`` is
``builtins.TimeoutError``, so ``future.result(timeout=slice)`` raises
the same type both when the slice expires and when the completed
operation itself raised ``TimeoutError`` (an ``asyncio.wait_for``
deadline inside an MCP call).  The sliced wait treated both as an
unfinished wait and re-raised the finished future's exception on
every slice until the outer deadline: a busy loop whose exception
grew one traceback chain per iteration.

The test drives the helper with a REAL ``asyncio`` loop thread and a
REAL coroutine that times out after 10 ms; no mocks.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator

import pytest

from kiss.core.tool_interrupt import result_interruptible


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    """A running event loop on its own thread, stopped and closed at teardown."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="mcp-result-timeout-loop")
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


async def _times_out_after_10ms() -> None:
    await asyncio.wait_for(asyncio.sleep(5), timeout=0.01)


async def _answers_after_10ms() -> str:
    await asyncio.sleep(0.01)
    return "ok"


def _traceback_depth(exc: BaseException) -> int:
    depth, tb = 0, exc.__traceback__
    while tb is not None:
        depth, tb = depth + 1, tb.tb_next
    return depth


def test_finished_future_raising_timeout_error_is_returned_at_once(
    loop: asyncio.AbstractEventLoop,
) -> None:
    future = asyncio.run_coroutine_threadsafe(_times_out_after_10ms(), loop)
    started, cpu = time.monotonic(), time.process_time()
    with pytest.raises(TimeoutError) as info:
        result_interruptible(future, 5.0)
    elapsed, cpu_used = time.monotonic() - started, time.process_time() - cpu
    assert future.done()
    assert elapsed < 2.0, f"spun until the outer deadline: {elapsed:.2f}s"
    assert cpu_used < 0.5, f"busy loop: {cpu_used:.2f}s of CPU"
    assert _traceback_depth(info.value) < 50


def test_unfinished_future_is_still_waited_for(loop: asyncio.AbstractEventLoop) -> None:
    future = asyncio.run_coroutine_threadsafe(_answers_after_10ms(), loop)
    assert result_interruptible(future, 5.0) == "ok"


def test_outer_deadline_still_raises_timeout_error(loop: asyncio.AbstractEventLoop) -> None:
    future = asyncio.run_coroutine_threadsafe(asyncio.sleep(5), loop)
    try:
        with pytest.raises(TimeoutError):
            result_interruptible(future, 0.5)
        assert not future.done()
    finally:
        future.cancel()
