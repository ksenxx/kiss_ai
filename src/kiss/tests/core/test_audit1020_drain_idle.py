# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``drain_pending_interrupt`` idles instead of spinning (audit 2026-10-20, A3-F1).

The drain gives an injected ``ToolCallInterrupted`` a place to land.  When
the tool swallowed the injection (``except BaseException``) nothing is
pending, so the drain runs out its full second: that second must cost
(almost) no CPU, not one core at 100 % with the GIL held.  The drain
must still deliver a pending injection promptly.
"""

from __future__ import annotations

import sys
import threading
import time

from kiss.core import tool_interrupt
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    drain_pending_interrupt,
    interrupt_tool_call,
    new_tool_call,
    register_tool_call,
    unregister_tool_call,
)


def _drain_with_nothing_pending(out: dict[str, float]) -> None:
    """Run the drain on a thread with no injection; record wall and CPU time."""
    wall = time.monotonic()
    cpu = time.thread_time()
    try:
        drain_pending_interrupt()
    except ToolCallInterrupted:
        out["wall"] = time.monotonic() - wall
        out["cpu"] = time.thread_time() - cpu


def test_drain_without_a_pending_injection_idles_rather_than_spins() -> None:
    out: dict[str, float] = {}
    thread = threading.Thread(target=_drain_with_nothing_pending, args=(out,), daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), "drain_pending_interrupt did not return"
    assert "wall" in out, "the explicit ToolCallInterrupted was not raised"
    # The explicit raise still waits the full drain period ...
    assert out["wall"] >= tool_interrupt._DRAIN_SECONDS * 0.9
    # ... but the thread sleeps through it instead of holding the GIL.
    assert out["cpu"] < tool_interrupt._DRAIN_SECONDS * 0.25, out


def _drain_on_a_tool_thread(ready: threading.Event, out: dict[str, float]) -> None:
    """Register a tool call and drain on it; record when the drain raised."""
    token = new_tool_call("swallower")
    register_tool_call(token)
    try:
        ready.set()
        try:
            drain_pending_interrupt()
        except ToolCallInterrupted:
            out["finished_at"] = time.monotonic()
    finally:
        unregister_tool_call(token)


def _inside_drain(thread: threading.Thread) -> bool:
    """Whether *thread* is currently executing ``drain_pending_interrupt``."""
    frame = sys._current_frames().get(thread.ident or 0)
    while frame is not None:
        if frame.f_code is drain_pending_interrupt.__code__:
            return True
        frame = frame.f_back
    return False


def test_drain_delivers_an_injection_that_lands_inside_it_promptly() -> None:
    ready = threading.Event()
    out: dict[str, float] = {}
    thread = threading.Thread(target=_drain_on_a_tool_thread, args=(ready, out), daemon=True)
    thread.start()
    assert ready.wait(5)
    # Inject only once the worker is provably inside the drain, so the
    # exception can land nowhere but in the drain's sleep loop.
    deadline = time.monotonic() + 5.0
    while not _inside_drain(thread):
        assert time.monotonic() < deadline, "the tool thread never entered the drain"
        time.sleep(0.001)
    token = tool_interrupt._RUNNING[thread.ident or 0]
    assert interrupt_tool_call(thread.ident or 0, "swallower")
    assert tool_interrupt._inject_unless_closing(token)
    injected_at = time.monotonic()
    thread.join(timeout=10)
    assert not thread.is_alive(), "the tool thread never finished draining"
    assert "finished_at" in out, "drain_pending_interrupt did not raise"
    # Delivered by the next 1 ms sleep, not by the explicit raise after
    # the full drain period.
    delay = out["finished_at"] - injected_at
    assert delay < tool_interrupt._DRAIN_SECONDS * 0.25, f"drain took {delay:.3f}s to deliver"
